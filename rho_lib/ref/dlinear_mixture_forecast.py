"""Moirai-aligned reference: DLinear forecaster with the same 4-component
Mixture NLL as Moirai itself.

Key differences vs `dlinear_normal.py`:

1. **Task = forecasting (suffix-only)** — input is the lookback window, target
   is the suffix. Matches Moirai's pretraining objective exactly. Random
   suffix length per sample (same range as Moirai: 15-50% of seq_len).

2. **Loss = 4-component Mixture NLL** — same `MixtureOutput` Moirai uses
   (StudentT + NormalFixedScale + NegativeBinomial + LogNormal). Units now
   match Moirai's `current_loss` exactly, eliminating the Normal-vs-Mixture
   bias of the legacy ref.

3. **Per-time-step ref table** — same `(N_windows, C99, T)` layout as
   `build_ref_loss_normal_timepoint` so Moirai's RHO lookup code does NOT need
   to change. The table now holds NLL only for time steps that fall in the
   suffix region (lookback steps get NLL=0, which the lookup never touches
   because Moirai only does RHO on prediction tokens).

Architecture:

    DLinear two-head: lookback (L) → 12 distr params per output time step
    Per-call random split: pick suffix length S in [0.15·T, 0.5·T]; lookback
    L = T - S; standardize-by-lookback (same as Moirai's PackedStdScaler does
    on observed positions only). Project lookback → suffix params, compute
    Mixture NLL on the suffix. Backprop on suffix loss only.

For ref-table construction we have to commit to a single (lookback, suffix)
split per window because the table is patch-size-agnostic per time step.
We use `mid-suffix` (suffix_len = 0.30·T = 154 steps for T=512) which sits
in the middle of Moirai's mask_ratio range — a reasonable surrogate for the
average forecasting NLL Moirai will face.
"""
import math
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from uni2ts.distribution import (
    MixtureOutput,
    StudentTOutput,
    NormalFixedScaleOutput,
    NegativeBinomialOutput,
    LogNormalOutput,
)

from ._train import set_seed


_DEFAULT_REF_SUFFIX_RATIO = 0.30   # used for the static ref-table build
_TRAIN_MIN_RATIO = 0.15
_TRAIN_MAX_RATIO = 0.50


def _build_distr_output() -> MixtureOutput:
    """Same 4-component mixture Moirai uses."""
    return MixtureOutput(components=[
        StudentTOutput(),
        NormalFixedScaleOutput(),
        NegativeBinomialOutput(),
        LogNormalOutput(),
    ])


def _split_param_proj(flat: torch.Tensor) -> dict:
    """Split a (..., 12) tensor into MixtureOutput's PyTree of distr params.

    Layout (12 dims/time-step total):
       0..3   weights_logits        (4)
       4..6   StudentT  {df, loc, scale}    (3)
       7      NormalFixedScale {loc}        (1)
       8..9   NegativeBinomial {total_count, logits}  (2)
       10..11 LogNormal {loc, scale}        (2)
    """
    # MixtureOutput.distribution accepts {weights_logits, components: [dict,...]}
    return {
        "weights_logits": flat[..., 0:4],
        "components": [
            {"df": flat[..., 4:5].squeeze(-1),
             "loc": flat[..., 5:6].squeeze(-1),
             "scale": flat[..., 6:7].squeeze(-1)},
            {"loc": flat[..., 7:8].squeeze(-1)},
            {"total_count": flat[..., 8:9].squeeze(-1),
             "logits": flat[..., 9:10].squeeze(-1)},
            {"loc": flat[..., 10:11].squeeze(-1),
             "scale": flat[..., 11:12].squeeze(-1)},
        ],
    }


# Apply MixtureOutput's domain_map to convert raw projections → valid params.
# domain_map is a PyTree of callables matching the args_dim structure; we
# walk it manually (uni2ts uses tree_map_multi which requires a container
# with the same shape — easier to just walk dict + list ourselves).
def _apply_domain_map(raw_params: dict, distr_output: MixtureOutput) -> dict:
    """Apply distr_output.domain_map (PyTree of callables) to raw_params.

    Adds NaN/Inf safety: clamp raw projections to a stable range BEFORE the
    domain_map callables (StudentT df constraints, scale > 0, etc.). Without
    clamping, large raw values from a barely-trained linear head produce NaN
    Categorical(logits) and crash distribution.log_prob.
    """
    SAFE_CLAMP = 30.0   # |raw| ≤ 30 → softplus(30) ≈ 30, exp not saturated
    def _safe(t):
        return torch.nan_to_num(t, nan=0.0, posinf=SAFE_CLAMP, neginf=-SAFE_CLAMP).clamp(-SAFE_CLAMP, SAFE_CLAMP)

    dm = distr_output.domain_map
    out = {
        "weights_logits": dm["weights_logits"](_safe(raw_params["weights_logits"])),
        "components": [
            {k: dm["components"][i][k](_safe(raw_params["components"][i][k]))
             for k in raw_params["components"][i]}
            for i in range(len(raw_params["components"]))
        ],
    }
    return out


class DLinearMixtureForecastRef(nn.Module):
    """DLinear that maps a lookback window to per-step Mixture distr params.

    forward(x_lb)        : (N, L) → (N, T_max, 12) raw distr params
                           (callers slice [:, :S, :] for the suffix length)
    distr(raw, target)   : build Mixture distribution and compute NLL
    """

    def __init__(self, max_seq_len: int):
        super().__init__()
        # 12 = total distr params per output step (see _split_param_proj).
        # The linear head is (max_lookback) → (max_seq_len * 12). At forward
        # time we slice the first L lookback inputs and the first S output
        # steps, so it's a single matmul regardless of (L, S) split.
        self.max_seq_len = max_seq_len
        self.linear = nn.Linear(max_seq_len, max_seq_len * 12, bias=True)
        self.distr_output = _build_distr_output()

    def forward(self, x_lb: torch.Tensor, max_pred_len: int) -> torch.Tensor:
        """
        x_lb : (N, L) lookback (possibly zero-padded to max_seq_len at right)
        returns raw_params : (N, max_pred_len, 12)
        """
        # Pad lookback to max_seq_len at the right (lookback occupies the LEFT
        # — Moirai mask is on the right suffix).
        N, L = x_lb.shape
        if L < self.max_seq_len:
            pad = torch.zeros(N, self.max_seq_len - L,
                              device=x_lb.device, dtype=x_lb.dtype)
            x = torch.cat([x_lb, pad], dim=1)
        else:
            x = x_lb[:, :self.max_seq_len]

        out = self.linear(x)                                     # (N, T*12)
        out = out.view(N, self.max_seq_len, 12)                  # (N, T, 12)
        return out[:, :max_pred_len, :]                          # (N, S, 12)

    def nll(self, raw_params: torch.Tensor,
            target: torch.Tensor,
            loc: torch.Tensor | None = None,
            scale: torch.Tensor | None = None) -> torch.Tensor:
        """
        raw_params : (N, S, 12)
        target     : (N, S)
        Returns nll : (N, S) per-time-step NLL.
        """
        params = _split_param_proj(raw_params)
        params = _apply_domain_map(params, self.distr_output)
        distr = self.distr_output.distribution(params, loc=loc, scale=scale)
        return -distr.log_prob(target)


def _train(dataset, *, seq_len, ref_epochs, batch_size, device, collate_fn,
           seed, tag, lr=1e-4, out_dir=None):
    set_seed(seed)
    c_sizes      = dataset.channel_counts_per_window()
    global_max_c = int(np.percentile(c_sizes, 99))
    model        = DLinearMixtureForecastRef(max_seq_len=seq_len).to(device)
    optimizer    = torch.optim.Adam(model.parameters(), lr=lr)
    loader       = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                              num_workers=0, collate_fn=collate_fn,
                              pin_memory=False)

    print(f'\n[{tag}] Training DLinear-Mixture forecaster: {ref_epochs} epochs')
    rng = random.Random(seed)

    step_log_f = None
    if out_dir is not None:
        from pathlib import Path as _P
        _od = _P(out_dir); _od.mkdir(parents=True, exist_ok=True)
        step_log_f = open(_od / 'ref_step_loss.csv', 'w')
        step_log_f.write('step,loss\n')
    global_step = 0

    for epoch in range(ref_epochs):
        model.train()
        total = 0.0
        n = 0
        nan_skips = 0
        for _, x, ch_mask in tqdm(loader, desc=f'  ref {epoch+1}/{ref_epochs}'):
            x = x.to(device, non_blocking=True)              # (B, C, T)
            ch_mask = ch_mask.to(device, non_blocking=True)
            B, C, T = x.shape
            # Random suffix ratio per BATCH (cheap, gives variety across epochs)
            ratio = rng.uniform(_TRAIN_MIN_RATIO, _TRAIN_MAX_RATIO)
            S = max(1, int(round(T * ratio)))
            L = T - S

            # Channel-flat
            valid = ch_mask.reshape(B * C)
            if not valid.any():
                continue
            x_flat  = x.reshape(B * C, T)
            x_lb    = x_flat[:, :L]
            target  = x_flat[:, L:L + S]

            # Self-standardize using lookback only (matches Moirai
            # PackedStdScaler which scales by observed positions, where the
            # suffix is unobserved during pretraining).
            mu  = x_lb.mean(dim=1, keepdim=True)
            sig = x_lb.std(dim=1, keepdim=True).clamp(min=1e-6)
            x_lb_n  = (x_lb - mu) / sig
            target_n = (target - mu) / sig

            raw = model(x_lb_n, max_pred_len=S)              # (B*C, S, 12)
            nll = model.nll(raw, target_n)                   # (B*C, S)
            # mask out padded channels
            nll_flat = nll.mean(dim=1)                       # (B*C,)
            loss     = (nll_flat * valid.float()).sum() / valid.float().sum().clamp(min=1)

            if not torch.isfinite(loss):
                nan_skips += 1
                optimizer.zero_grad(set_to_none=True)
                # If model params went non-finite, sanitize them so subsequent
                # batches don't propagate NaN.
                with torch.no_grad():
                    for p in model.parameters():
                        if not torch.isfinite(p).all():
                            p.nan_to_num_(nan=0.0, posinf=1.0, neginf=-1.0)
                continue
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            # Sanitize gradients before stepping
            for p in model.parameters():
                if p.grad is not None and not torch.isfinite(p.grad).all():
                    p.grad.nan_to_num_(nan=0.0, posinf=1.0, neginf=-1.0)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item(); n += 1
            global_step += 1
            if step_log_f is not None and global_step % 10 == 0:
                step_log_f.write(f'{global_step},{loss.item():.6f}\n')
                step_log_f.flush()
        print(f'  ref epoch {epoch+1}: nll={total/max(n,1):.4f}  '
              f'(nan_skips={nan_skips})')
    if step_log_f is not None:
        step_log_f.close()
    return model, global_max_c


def build_ref_loss_mixture_forecast_timepoint(
    dataset, *, seq_len, ref_epochs, batch_size, device, collate_fn,
    seed: int = 42, out_dir=None,
) -> torch.Tensor:
    """Build a `(N, C99, T)` per-time-step Mixture-NLL ref table.

    The TABLE BUILD step uses a single canonical split: lookback = 70% T,
    suffix = 30% T. The first L positions get NLL = 0 (never used because
    Moirai's RHO lookup is restricted to prediction tokens which always lie
    in the suffix region). Suffix positions get the actual per-step Mixture
    NLL produced by the trained DLinear forecaster.

    Returns: ref_table  shape (N, C99, T)  on CPU.
    """
    model, global_max_c = _train(
        dataset, seq_len=seq_len, ref_epochs=ref_epochs,
        batch_size=batch_size, device=device,
        collate_fn=collate_fn, seed=seed,
        tag='Ref-Mixture-Forecast', out_dir=out_dir,
    )

    N = len(dataset)
    ref_table = torch.zeros(N, global_max_c, seq_len, dtype=torch.float32)

    # Eval-time fixed ratio sits in the middle of Moirai's mask_ratio range.
    S = max(1, int(round(seq_len * _DEFAULT_REF_SUFFIX_RATIO)))
    L = seq_len - S

    eval_loader = DataLoader(
        dataset, batch_size=batch_size * 2, shuffle=False,
        num_workers=0, collate_fn=collate_fn, pin_memory=False,
    )
    model.eval()
    print(f'[Ref-Mixture-Forecast] table build: '
          f'lookback={L}, suffix={S} (ratio={_DEFAULT_REF_SUFFIX_RATIO})')
    with torch.no_grad():
        for indices, x, ch_mask in tqdm(eval_loader, desc='  ref eval'):
            x = x.to(device); B, C, T = x.shape
            C_eff = min(C, global_max_c)
            x_flat = x[:, :C_eff].reshape(B * C_eff, T)
            x_lb   = x_flat[:, :L]
            target = x_flat[:, L:L + S]

            mu  = x_lb.mean(dim=1, keepdim=True)
            sig = x_lb.std(dim=1, keepdim=True).clamp(min=1e-6)
            x_lb_n   = (x_lb - mu) / sig
            target_n = (target - mu) / sig

            raw = model(x_lb_n, max_pred_len=S)             # (B*C_eff, S, 12)
            nll = model.nll(raw, target_n)                  # (B*C_eff, S)
            nll = nll.reshape(B, C_eff, S).float()

            # Lookback positions stay 0 (RHO never reads them — pred tokens
            # always sit in the suffix region of the original time axis).
            ref_slice = torch.zeros(B, C_eff, T, dtype=torch.float32)
            ref_slice[:, :, L:L + S] = nll.cpu()
            ref_table[indices, :C_eff, :] = ref_slice

    suffix_only = ref_table[..., L:L + S]
    print(f'[Ref-Mixture-Forecast] shape={tuple(ref_table.shape)} '
          f'suffix_mean={suffix_only.mean():.4f} '
          f'suffix_std={suffix_only.std():.4f}')
    return ref_table
