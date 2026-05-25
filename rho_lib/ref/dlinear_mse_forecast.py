"""DLinear point-prediction reference trained with MSE.

Used by TimesFM for SPM selection: the reference's per-timestep squared
error is compared against the foundation model's own squared error to
compute rho = current_MSE - ref_MSE (both in identical units).
"""
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from ._train import set_seed


_DEFAULT_REF_SUFFIX_RATIO = 0.30   # used for the static ref-table build
_TRAIN_MIN_RATIO = 0.15
_TRAIN_MAX_RATIO = 0.50


class DLinearMSEForecastRef(nn.Module):
    """Plain Linear MSE forecaster: lookback → suffix point predictions.

    Trained with MSE on the suffix region. Compared against the foundation model's point
    prediction (mixture.mean) to compute MSE-based ρ.
    """

    def __init__(self, max_seq_len: int):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.linear = nn.Linear(max_seq_len, max_seq_len, bias=True)

    def forward(self, x_lb: torch.Tensor, max_pred_len: int) -> torch.Tensor:
        """
        x_lb : (N, L) lookback (zero-padded to max_seq_len at right)
        Returns pred : (N, max_pred_len) point predictions on suffix.
        """
        N, L = x_lb.shape
        if L < self.max_seq_len:
            pad = torch.zeros(N, self.max_seq_len - L,
                              device=x_lb.device, dtype=x_lb.dtype)
            x = torch.cat([x_lb, pad], dim=1)
        else:
            x = x_lb[:, :self.max_seq_len]
        out = self.linear(x)                        # (N, max_seq_len)
        return out[:, :max_pred_len]                # (N, max_pred_len)


def _train(dataset, *, seq_len, ref_epochs, batch_size, device, collate_fn,
           seed, tag, lr=1e-3, out_dir=None):
    set_seed(seed)
    c_sizes      = dataset.channel_counts_per_window()
    global_max_c = int(np.percentile(c_sizes, 99))
    model        = DLinearMSEForecastRef(max_seq_len=seq_len).to(device)
    optimizer    = torch.optim.Adam(model.parameters(), lr=lr)
    loader       = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                              num_workers=0, collate_fn=collate_fn,
                              pin_memory=False)

    print(f'\n[{tag}] Training DLinear-MSE forecaster: {ref_epochs} epochs (lr={lr})')
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
        total = 0.0; n = 0
        for _, x, ch_mask in tqdm(loader, desc=f'  ref {epoch+1}/{ref_epochs}'):
            x = x.to(device, non_blocking=True)
            ch_mask = ch_mask.to(device, non_blocking=True)
            B, C, T = x.shape
            ratio = rng.uniform(_TRAIN_MIN_RATIO, _TRAIN_MAX_RATIO)
            S = max(1, int(round(T * ratio)))
            L = T - S

            valid = ch_mask.reshape(B * C)
            if not valid.any():
                continue
            x_flat   = x.reshape(B * C, T)
            x_lb     = x_flat[:, :L]
            target   = x_flat[:, L:L + S]

            mu      = x_lb.mean(dim=1, keepdim=True)
            sig_raw = x_lb.std(dim=1, keepdim=True)
            # Drop near-constant lookback channels: dividing target by clamp
            # blows the loss when raw target jumps by O(1) → MSE 1e6.
            # Diagnosis showed exploding batches always had hundreds of such
            # channels; better to mask them out than rely on clamp/clip.
            nonconst = (sig_raw.squeeze(-1) > 1e-2)
            valid    = valid & nonconst
            if not valid.any():
                continue
            sig      = sig_raw.clamp(min=1e-2)
            x_lb_n   = (x_lb - mu) / sig
            target_n = (target - mu) / sig

            pred = model(x_lb_n, max_pred_len=S)              # (B*C, S)
            sq_err = (pred - target_n) ** 2                   # (B*C, S)
            per_bc = sq_err.mean(dim=1).clamp(max=100.0)      # cap outlier ch
            loss   = (per_bc * valid.float()).sum() / valid.float().sum().clamp(min=1)

            # Robustness: skip non-finite batches and clip grads
            if not torch.isfinite(loss):
                optimizer.zero_grad(set_to_none=True)
                continue
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item(); n += 1
            global_step += 1
            if step_log_f is not None and global_step % 10 == 0:
                step_log_f.write(f'{global_step},{loss.item():.6f}\n')
                step_log_f.flush()
        print(f'  ref epoch {epoch+1}: mse={total/max(n,1):.4f}')
    if step_log_f is not None:
        step_log_f.close()
    return model, global_max_c


def build_ref_model_only(
    dataset, *, seq_len, ref_epochs, batch_size, device, collate_fn,
    seed: int = 42, out_dir=None,
):
    """Train the DLinear MSE forecaster and return the model itself
    (without building a static (N, C99, T) ref table).

    Live ref mode: instead of caching ref MSE for a single
    fixed suffix ratio (0.30), the model is run per-batch with the actual
    sample-specific lookback/suffix split. Trades extra compute (~ref-forward
    per batch) for ratio-accurate ref MSE.

    Returns: (model, global_max_c)
    """
    return _train(
        dataset, seq_len=seq_len, ref_epochs=ref_epochs,
        batch_size=batch_size, device=device,
        collate_fn=collate_fn, seed=seed,
        tag='Ref-MSE-Forecast-Live', out_dir=out_dir,
    )

def build_ref_loss_mse_forecast_timepoint(
    dataset, *, seq_len, ref_epochs, batch_size, device, collate_fn,
    seed: int = 42, out_dir=None,
) -> torch.Tensor:
    """Build a `(N, C99, T)` per-time-step MSE ref table.

    Same layout as the Mixture-NLL ref table — suffix region holds the
    DLinear MSE forecaster's per-step squared error; lookback positions
    stay 0 (RHO never reads them since prediction tokens always lie in
    the suffix region).

    Returns: ref_table  shape (N, C99, T)  on CPU.
    """
    model, global_max_c = _train(
        dataset, seq_len=seq_len, ref_epochs=ref_epochs,
        batch_size=batch_size, device=device,
        collate_fn=collate_fn, seed=seed,
        tag='Ref-MSE-Forecast', out_dir=out_dir,
    )

    N = len(dataset)
    ref_table = torch.zeros(N, global_max_c, seq_len, dtype=torch.float32)

    S = max(1, int(round(seq_len * _DEFAULT_REF_SUFFIX_RATIO)))
    L = seq_len - S

    eval_loader = DataLoader(
        dataset, batch_size=batch_size * 2, shuffle=False,
        num_workers=0, collate_fn=collate_fn, pin_memory=False,
    )
    model.eval()
    print(f'[Ref-MSE-Forecast] table build: '
          f'lookback={L}, suffix={S} (ratio={_DEFAULT_REF_SUFFIX_RATIO})')
    with torch.no_grad():
        for indices, x, ch_mask in tqdm(eval_loader, desc='  ref eval'):
            x = x.to(device); B, C, T = x.shape
            C_eff  = min(C, global_max_c)
            x_flat = x[:, :C_eff].reshape(B * C_eff, T)
            x_lb   = x_flat[:, :L]
            target = x_flat[:, L:L + S]

            mu      = x_lb.mean(dim=1, keepdim=True)
            sig_raw = x_lb.std(dim=1, keepdim=True)
            # near-constant lookback channels: zero ref_sq → no spurious ρ
            nonconst = (sig_raw.squeeze(-1) > 1e-2)           # (B*C_eff,)
            sig      = sig_raw.clamp(min=1e-2)
            x_lb_n   = (x_lb - mu) / sig
            target_n = (target - mu) / sig

            pred   = model(x_lb_n, max_pred_len=S)            # (B*C_eff, S)
            sq_err = (pred - target_n) ** 2                   # (B*C_eff, S)
            sq_err = sq_err * nonconst.float().unsqueeze(-1)  # zero const channels
            sq_err = sq_err.reshape(B, C_eff, S).float()

            ref_slice = torch.zeros(B, C_eff, T, dtype=torch.float32)
            ref_slice[:, :, L:L + S] = sq_err.cpu()
            ref_table[indices, :C_eff, :] = ref_slice

    suffix_only = ref_table[..., L:L + S]
    print(f'[Ref-MSE-Forecast] shape={tuple(ref_table.shape)} '
          f'suffix_mean={suffix_only.mean():.4f} '
          f'suffix_std={suffix_only.std():.4f}')
    return ref_table
