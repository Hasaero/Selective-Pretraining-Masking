"""
Moirai-1.0 Pretraining with token-level RHO-CM masking.

Baseline : Moirai-small (random init) + suffix masked patch prediction (NLL)
rho_cm   : Moirai-small + per-token RHO weighting
           ρ[t] = current_NLL[t] - ref_NLL[t]   for each prediction token
                                                  (= sample × variate × time-patch)
           Drop bottom drop_pct% of prediction tokens per batch.
           ref = DLinear (Mixture NLL, suffix forecasting) trained briefly on same
           corpus, producing a (N_windows, C99, SEQ_LEN) per-time-step NLL
           table. The training loop aggregates ref to per-token granularity
           by averaging over each token's actual patch_size — supporting
           Moirai's per-sample patch_size sampling.

           This matches Moirai's loss aggregation unit (packed token NLL),
           consistent with MOMENT's patch-level and Timer's token-level RHO.

Usage
-----
# Baseline
python scripts/pretrain_moirai.py --mode baseline --epochs 2 --batch-size 32

# Token-level RHO-CM
python scripts/pretrain_moirai.py --mode rho_cm --epochs 2 --batch-size 32 \
    --ref-epochs 3 --drop-pct 10

# Evaluate (linear probe on ETTh1 forecasting)
python scripts/pretrain_moirai.py --mode eval \
    --ckpt results_moirai/rho_cm/best.pt \
    --eval-dataset ETTh1 --eval-horizon 96
"""

import argparse
import math
import random
import sys
import time
from pathlib import Path

# Allow `from rho_lib...` to resolve when this script is launched from
# anywhere — adds the project root (parent of scripts/) to sys.path.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# ── rho_lib (shared scaffolding) ──────────────────────────────────────────────
from rho_lib.data.utsd     import (UTSDPretrainDataset, make_collate_fn,
                                   load_utsd, compute_channel_cap)
from rho_lib.data.forecast import prepare_forecast_datasets
from rho_lib.ref.dlinear_mse_forecast import (
    build_ref_loss_mse_forecast_timepoint,
    build_ref_model_only,
)
from rho_lib.ref.dlinear_per_domain import build_per_domain_ref_table
from rho_lib.rho.mask      import compute_rho_mask
from rho_lib.eval.sweep    import run_eval_sweep
from rho_lib.train.checkpointing import save_epoch, dump_metrics

# ── uni2ts (Moirai) ───────────────────────────────────────────────────────────
# Load module.py directly to avoid lightning/torchvision crash in __init__.py
import importlib.util as _ilu
import types as _types

def _load_moirai_module():
    import uni2ts  # registers top-level package
    _uni2ts_root = Path(_ilu.find_spec('uni2ts').origin).parent

    # Register a stub for uni2ts.model.moirai so its __init__ is never executed
    _stub = _types.ModuleType('uni2ts.model.moirai')
    sys.modules.setdefault('uni2ts.model.moirai', _stub)

    _path = _uni2ts_root / 'model' / 'moirai' / 'module.py'
    _spec = _ilu.spec_from_file_location('uni2ts.model.moirai.module', _path)
    _mod  = _ilu.module_from_spec(_spec)
    sys.modules['uni2ts.model.moirai.module'] = _mod
    _spec.loader.exec_module(_mod)
    return _mod.MoiraiModule

MoiraiModule = _load_moirai_module()

from uni2ts.distribution import (
    MixtureOutput,
    StudentTOutput,
    NormalFixedScaleOutput,
    NegativeBinomialOutput,
    LogNormalOutput,
)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
SEQ_LEN      = 512
UTSD_PATH    = Path(__file__).resolve().parent.parent / "data" / 'utsd_repo' / 'UTSD-12G'
DATA_DIR     = Path(__file__).resolve().parent.parent / "data"


# Convenience: a SEQ_LEN-bound collate (Moirai uses fixed 512, no channel cap).
collate_fn = make_collate_fn(SEQ_LEN, channel_cap=None)


# ─────────────────────────────────────────────────────────────────────────────
# Optimizer / scheduler — matches MoiraiPretrain.configure_optimizers
# ─────────────────────────────────────────────────────────────────────────────

def _build_optimizer(module: nn.Module, lr: float, weight_decay: float) -> torch.optim.AdamW:
    """Param groups matching official MoiraiPretrain.configure_optimizers."""
    from uni2ts.module.position import (
        BinaryAttentionBias, LearnedEmbedding, LearnedProjection,
    )
    from uni2ts.module.norm import RMSNorm
    from uni2ts.module.ts_embed import MultiInSizeLinear, MultiOutSizeLinear

    whitelist = (LearnedProjection, MultiInSizeLinear, MultiOutSizeLinear, nn.Linear)
    blacklist = (BinaryAttentionBias, LearnedEmbedding, RMSNorm, nn.Embedding, nn.LayerNorm)

    decay, no_decay = set(), set()
    for mn, m in module.named_modules():
        for pn, p in m.named_parameters(recurse=False):
            if not p.requires_grad:
                continue
            fpn = f'{mn}.{pn}' if mn else pn
            if pn.endswith('bias'):
                no_decay.add(fpn)
            elif pn.endswith('weight') and isinstance(m, whitelist):
                decay.add(fpn)
            elif pn.endswith('weight') and isinstance(m, blacklist):
                no_decay.add(fpn)

    param_dict = {pn: p for pn, p in module.named_parameters() if p.requires_grad}
    inter = decay & no_decay
    union = decay | no_decay
    assert not inter, f'params in both groups: {inter}'
    missing = set(param_dict.keys()) - union
    assert not missing, f'params not classified: {missing}'

    groups = [
        {'params': [param_dict[pn] for pn in sorted(decay)],
         'weight_decay': weight_decay},
        {'params': [param_dict[pn] for pn in sorted(no_decay)],
         'weight_decay': 0.0},
    ]
    print(f'[Optim] decay={len(decay)}  no_decay={len(no_decay)}  '
          f'(wd={weight_decay} on decay group)')
    return torch.optim.AdamW(groups, lr=lr, betas=(0.9, 0.98), eps=1e-6)


def _build_scheduler_cosine_with_restarts(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float = 0.5,
):
    """Equivalent to transformers/uni2ts cosine_with_restarts (num_cycles=0.5
    is the default half-cosine — same shape used by uni2ts.optim.get_scheduler)."""
    def _lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = (float(current_step - num_warmup_steps)
                    / float(max(1, num_training_steps - num_warmup_steps)))
        if progress >= 1.0:
            return 0.0
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * ((num_cycles * progress) % 1.0))))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, _lambda)


# ─────────────────────────────────────────────────────────────────────────────
# Moirai constants (small variant, Moirai-1.0-R-small config)
# ─────────────────────────────────────────────────────────────────────────────
PATCH_SIZES   = (8, 16, 32, 64, 128)

# Domain-conditioned patch_size candidates derived from per-domain frequency
# (see domain_of() in UTSDPretrainDataset). Replaces uniform random sampling
# over PATCH_SIZES, which let high-token-count batches OOM at bs>=64.
DOMAIN_PATCH_SIZES = {
    'Web':         (32,),
    'Health':      (128,),
    'Nature':      (64,),
    'Energy':      (64,),
    'ERA5':        (64,),
    'IoT':         (128,),
    'Transport':   (64,),
    'Environment': (64,),
}
DEFAULT_PATCH_SIZES = (64,)
D_MODEL       = 384
NUM_LAYERS    = 6
MAX_SEQ_LEN   = 512   # max number of patches in one sequence
MIN_MASK_RATIO = 0.15
MAX_MASK_RATIO = 0.50
MAX_DIM       = 128   # max variate id space for AddVariateIndex (official: max_dim=128)

# ─────────────────────────────────────────────────────────────────────────────
# Build Moirai distribution output (4-component mixture, same as pretrained)
# ─────────────────────────────────────────────────────────────────────────────

def _build_distr_output() -> MixtureOutput:
    # Official Moirai mixture (cli/conf/pretrain/model/moirai_small.yaml)
    return MixtureOutput(components=[
        StudentTOutput(),
        NormalFixedScaleOutput(),
        NegativeBinomialOutput(),
        LogNormalOutput(),
    ])


# ─────────────────────────────────────────────────────────────────────────────
# Data preparation: convert (B, C, SEQ_LEN) batch → Moirai packed format
# ─────────────────────────────────────────────────────────────────────────────

def batch_to_moirai(
    x: torch.Tensor,        # (B, C, SEQ_LEN)
    ch_mask: torch.Tensor,  # (B, C) bool
    min_mask_ratio: float,
    max_mask_ratio: float,
    rng: random.Random,
    device: torch.device,
    domains: list[str] | None = None,  # per-sample domain prefix (optional)
) -> dict[str, torch.Tensor]:
    """
    Convert raw (B, C, T) batch into the flat packed format expected by MoiraiModule.

    Faithful to the official Moirai recipe:
      • patch size sampled per sample (GetPatchSize transform, patch.py)
      • prediction mask is a contiguous SUFFIX (forecasting-style) per sample
        (MaskedPrediction._generate_prediction_mask, transform/task.py)
      • variate ids are randomized in [0, MAX_DIM) per sample
        (AddVariateIndex(randomize=True, max_dim=128))

    The flat packed dict also carries `_orig_channel_id_flat` so the RHO-CM
    path can map each token back to its ORIGINAL channel position (which is
    what `rho_mask[b, c]` is indexed by), independent of the randomized id
    handed to the model.
    """
    B, _C_max, T = x.shape
    max_ps = max(PATCH_SIZES)   # 128
    valid_ps = [p for p in PATCH_SIZES if T >= p * 2]
    if not valid_ps:
        valid_ps = [PATCH_SIZES[0]]

    # Per-sample blocks; each is concatenated once at the end.
    blocks_target   = []
    blocks_obs      = []
    blocks_time     = []
    blocks_variate  = []
    blocks_orig_ch  = []
    blocks_pred     = []
    blocks_sample   = []
    blocks_psize    = []

    arange_max_p = torch.arange(max(T // p for p in valid_ps), device=device)  # reusable

    for b in range(B):
        real_channels = ch_mask[b].nonzero(as_tuple=True)[0]  # (C_real,) long, original idx
        C_real = int(real_channels.numel())
        if C_real == 0:
            continue

        # Per-sample patch size: domain-conditioned candidates if available,
        # else fall back to the (filtered) global PATCH_SIZES list.
        if domains is not None:
            cand = DOMAIN_PATCH_SIZES.get(domains[b], DEFAULT_PATCH_SIZES)
            cand_valid = [p for p in cand if p in valid_ps]
            if not cand_valid:
                cand_valid = valid_ps
            patch_size_b = rng.choice(cand_valid)
        else:
            patch_size_b = rng.choice(valid_ps)
        n_patches_b  = T // patch_size_b
        pad_w        = max_ps - patch_size_b

        # Suffix-only prediction mask
        mask_ratio_b = rng.uniform(min_mask_ratio, max_mask_ratio)
        n_masked_b   = max(1, int(round(n_patches_b * mask_ratio_b)))
        pred_flag_b  = torch.zeros(n_patches_b, dtype=torch.bool, device=device)
        pred_flag_b[-n_masked_b:] = True

        # Cap channels at MAX_DIM (rare: UTSD has a few series with C>128).
        # Original channel index is preserved for RHO lookup.
        if C_real > MAX_DIM:
            real_channels = real_channels[:MAX_DIM]
            C_real = MAX_DIM
        n_tokens_b = C_real * n_patches_b
        perm_b = torch.randperm(MAX_DIM, device=device)[:C_real]    # (C_real,)

        # ── Vectorised packing: build (C_real, P_b, ...) blocks then flatten ──
        # target: (C_real, P_b, max_ps)
        x_b   = x[b, real_channels]                                      # (C_real, T)
        tgt_b = x_b.reshape(C_real, n_patches_b, patch_size_b)
        if pad_w > 0:
            tgt_b = torch.nn.functional.pad(tgt_b, (0, pad_w))

        # observed_mask shared across all (C_real, P_b) — broadcast then expand
        obs_row = torch.zeros(max_ps, dtype=torch.bool, device=device)
        obs_row[:patch_size_b] = True
        obs_b   = obs_row.expand(C_real, n_patches_b, max_ps).contiguous()

        # 1-D ids of length n_tokens_b
        t_ids   = arange_max_p[:n_patches_b].repeat(C_real)              # (C_real*P_b,)
        v_ids   = perm_b.repeat_interleave(n_patches_b)
        oc_ids  = real_channels.repeat_interleave(n_patches_b)
        s_ids   = torch.full((n_tokens_b,), b, dtype=torch.long, device=device)
        ps_ids  = torch.full((n_tokens_b,), patch_size_b, dtype=torch.long, device=device)
        pred_ids = pred_flag_b.repeat(C_real)                            # (C_real*P_b,) bool

        blocks_target.append(tgt_b.reshape(n_tokens_b, max_ps))
        blocks_obs.append(obs_b.reshape(n_tokens_b, max_ps))
        blocks_time.append(t_ids)
        blocks_variate.append(v_ids)
        blocks_orig_ch.append(oc_ids)
        blocks_pred.append(pred_ids)
        blocks_sample.append(s_ids)
        blocks_psize.append(ps_ids)

    if not blocks_target:
        return None

    target_t  = torch.cat(blocks_target,  dim=0)
    obs_t     = torch.cat(blocks_obs,     dim=0)
    time_t    = torch.cat(blocks_time,    dim=0)
    variate_t = torch.cat(blocks_variate, dim=0)
    orig_ch_t = torch.cat(blocks_orig_ch, dim=0)
    pred_t    = torch.cat(blocks_pred,    dim=0)
    sample_t  = torch.cat(blocks_sample,  dim=0)
    psize_t   = torch.cat(blocks_psize,   dim=0)

    return dict(
        target          = target_t.unsqueeze(0).float(),
        observed_mask   = obs_t.unsqueeze(0),
        time_id         = time_t.unsqueeze(0),
        variate_id      = variate_t.unsqueeze(0),
        prediction_mask = pred_t.unsqueeze(0),
        sample_id       = sample_t.unsqueeze(0),
        patch_size      = psize_t.unsqueeze(0),
        # metadata for RHO-CM
        _pred_flag           = pred_t,          # (S_total,) bool
        _variate_id_flat     = variate_t,       # (S_total,) randomized id (for reference)
        _orig_channel_id_flat = orig_ch_t,      # (S_total,) ORIGINAL channel idx — RHO lookup
        _sample_id_flat      = sample_t,        # (S_total,)
    )


# ─────────────────────────────────────────────────────────────────────────────
# Per-channel NLL computation
# ─────────────────────────────────────────────────────────────────────────────

def moirai_nll_per_channel(
    distr,
    packed: dict,
    B: int,
    C_max: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute per-(sample, channel) NLL given an already-computed distribution.

    `distr` is reused from the main forward — no second forward pass.
    Caller is responsible for wrapping in torch.no_grad() if gradients
    should be cut for the rho computation.

    Returns:
        nll_bc   : (B, C_max) float  — 0 for padded channels
        valid_bc : (B, C_max) bool
    """
    # NLL per token: (1, S_total, max_ps)
    nll_token = -distr.log_prob(packed['target']).squeeze(0)   # (S_total, max_ps)

    pred_flag = packed['_pred_flag']                # (S_total,) bool
    orig_ch   = packed['_orig_channel_id_flat']     # (S_total,) ORIGINAL channel idx
    sample_id = packed['_sample_id_flat']           # (S_total,)

    # Average NLL across patch positions, on prediction tokens only
    nll_pred = nll_token[pred_flag].mean(dim=-1)    # (n_pred_tokens,)
    ch_pred  = orig_ch[pred_flag]
    smp_pred = sample_id[pred_flag]

    # Scatter into (B, C_max). Both indices are guaranteed in-range by
    # construction in batch_to_moirai (sample_id < B, orig_channel < C_max).
    flat_idx = smp_pred * C_max + ch_pred
    nll_bc   = torch.zeros(B * C_max, device=device)
    count_bc = torch.zeros(B * C_max, device=device)
    nll_bc.scatter_add_(0, flat_idx, nll_pred)
    count_bc.scatter_add_(0, flat_idx, torch.ones_like(nll_pred))
    nll_bc   = nll_bc.view(B, C_max)
    count_bc = count_bc.view(B, C_max)
    valid_bc = count_bc > 0
    nll_bc   = torch.where(valid_bc, nll_bc / count_bc.clamp(min=1.0), torch.zeros_like(nll_bc))
    return nll_bc, valid_bc


# ─────────────────────────────────────────────────────────────────────────────
# Scalar loss from packed distribution (with optional per-channel RHO weighting)
# ─────────────────────────────────────────────────────────────────────────────

def packed_nll_loss(
    distr,
    target: torch.Tensor,           # (1, S, patch_size)
    prediction_mask: torch.Tensor,  # (1, S) bool
    observed_mask: torch.Tensor,    # (1, S, patch_size) bool
    channel_weight: torch.Tensor | None,  # (S,) float or None
) -> torch.Tensor:
    """Compute masked NLL with optional per-token channel weighting."""
    nll = -distr.log_prob(target)        # (1, S, patch_size)
    nll = nll.squeeze(0)                 # (S, patch_size)
    obs  = observed_mask.squeeze(0)      # (S, patch_size)
    pred = prediction_mask.squeeze(0)   # (S,)

    # Only masked tokens contribute to loss
    mask = pred.unsqueeze(-1) * obs      # (S, patch_size)

    if channel_weight is not None:
        mask = mask * channel_weight.unsqueeze(-1)

    denom = mask.sum().clamp(min=1.0)
    return (nll * mask).sum() / denom


# ─────────────────────────────────────────────────────────────────────────────
# Main pretraining loop
# ─────────────────────────────────────────────────────────────────────────────

def pretrain(
    mode: str,
    epochs: int,
    batch_size: int,
    lr: float,
    drop_pct: float,
    ref_epochs: int,
    out_dir: Path,
    seed: int,
    max_series: int | None,
    device: torch.device,
    ref_mode: str = 'timepoint',
):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    rng = random.Random(seed)

    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Dataset ───────────────────────────────────────────────────────────────
    print('[Data] Loading UTSD from', UTSD_PATH, '...')
    hf_ds = load_utsd(UTSD_PATH, max_series=max_series)
    print(f'[Data] {len(hf_ds):,} rows loaded')
    dataset = UTSDPretrainDataset(
        hf_ds, seq_len=SEQ_LEN, stride=SEQ_LEN,
        max_series=max_series, standardize_per_series=False,  # Moirai feeds raw
    )

    # Cap channels to bound packed sequence length S = C * n_patches.
    # Moirai packs all (sample, channel, patch) tokens into one batch dim and
    # builds an (S,S) attention bias — memory grows quadratically. Empirically
    # channel_cap > 32 with batch >= 16 OOMs even on H200 (146 GB) due to
    # outlier batches with rare wide series. UTSD p99=33 covers 99% of windows
    # so cap=32 drops only 1.5% of training data.
    raw_cap = compute_channel_cap(dataset, 99.0)
    channel_cap = min(raw_cap, 32)  # SAFE_CAP for Moirai packed attention
    capped_collate = make_collate_fn(SEQ_LEN, channel_cap=channel_cap)
    print(f'[Data] channel cap: {channel_cap} (raw p99={raw_cap}, '
          f'safe_max=32 for Moirai packed attention)')

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=capped_collate,
        pin_memory=False,
        drop_last=True,
    )
    print(f'[Data] {len(loader)} batches/epoch')

    # ── Model: Moirai-small, random init ─────────────────────────────────────
    print('[Model] Initializing Moirai-small (random init)...')
    distr_output = _build_distr_output()
    module = MoiraiModule(
        distr_output   = distr_output,
        d_model        = D_MODEL,
        num_layers     = NUM_LAYERS,
        patch_sizes    = PATCH_SIZES,
        max_seq_len    = MAX_SEQ_LEN,
        attn_dropout_p = 0.0,   # official moirai_small.yaml
        dropout_p      = 0.0,
        scaling        = True,
    ).to(device)
    n_params = sum(p.numel() for p in module.parameters())
    print(f'[Model] {n_params/1e6:.1f}M parameters')

    total_steps  = len(loader) * epochs
    # Official: fixed 10k warmup. Cap at 10% on tiny smoke runs.
    warmup_steps = min(10_000, max(1, total_steps // 10))
    optimizer    = _build_optimizer(module, lr=lr, weight_decay=1e-1)
    scheduler    = _build_scheduler_cosine_with_restarts(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps,
    )

    # ── RHO-CM: DLinear reference (timepoint, per_domain, or live) ──
    ref_loss_table = None
    per_domain_ref = None
    ref_live_model = None  # set when ref_mode == 'live'
    ref_live_C99   = None
    sample_to_local_idx = None  # (N,) int: per-sample local idx in domain table
    if mode == 'rho_cm':
        t0 = time.time()
        if ref_mode == 'per_domain':
            per_domain_ref = build_per_domain_ref_table(
                dataset, seq_len=SEQ_LEN, ref_epochs=ref_epochs,
                batch_size=256, device=device, collate_fn=capped_collate,
                domain_patch_sizes=DOMAIN_PATCH_SIZES,
                default_patch_size=DEFAULT_PATCH_SIZES[0],
                seed=seed, out_dir=out_dir,
            )
            sample_to_local_idx = np.full(len(dataset), -1, dtype=np.int64)
            sample_to_domain_name_arr = [None] * len(dataset)
            for dom, info in per_domain_ref.items():
                for li, gi in enumerate(info['window_indices']):
                    sample_to_local_idx[gi] = li
                    sample_to_domain_name_arr[gi] = dom
            if torch.cuda.is_available():
                for dom, info in per_domain_ref.items():
                    t = info['table']
                    if t.element_size() * t.nelement() < 2 * (1 << 30):
                        info['table'] = t.to(device, non_blocking=True)
            n_doms = len(per_domain_ref)
            print(f'[RHO-CM per_domain] {n_doms} domain refs built in {time.time()-t0:.1f}s')
        elif ref_mode == 'live':
            # Live ref model — no static table. Run ref forward per batch
            # with the actual sample-specific (lookback, suffix) split so
            # ref MSE matches the prediction_mask exactly (ratio-accurate).
            ref_live_model, ref_live_C99 = build_ref_model_only(
                dataset, seq_len=SEQ_LEN, ref_epochs=ref_epochs,
                batch_size=256, device=device, collate_fn=capped_collate,
                seed=seed, out_dir=out_dir,
            )
            ref_live_model.eval()
            for p in ref_live_model.parameters():
                p.requires_grad_(False)
            print(f'[RHO-CM live] ref model trained in {time.time()-t0:.1f}s '
                  f'(C99={ref_live_C99}, no static table)')
        else:
            ref_loss_table = build_ref_loss_mse_forecast_timepoint(
                dataset, seq_len=SEQ_LEN, ref_epochs=ref_epochs,
                batch_size=256, device=device, collate_fn=capped_collate,
                seed=seed, out_dir=out_dir,
            )
            print(f'[RHO-CM timepoint] ref built in {time.time()-t0:.1f}s')
            if torch.cuda.is_available():
                ref_bytes = ref_loss_table.element_size() * ref_loss_table.nelement()
                if ref_bytes < 4 * (1 << 30):
                    ref_loss_table = ref_loss_table.to(device, non_blocking=True)
                    print(f'[RHO-CM] ref table on GPU ({ref_bytes/1e6:.1f} MB)')

    # ── Training ──────────────────────────────────────────────────────────────
    metrics    = {'train_loss': [], 'kept_ratio': []}
    best_loss  = float('inf')

    # Per-step loss log for convergence plots
    step_log_path = out_dir / 'step_loss.csv'
    step_log_f = open(step_log_path, 'w')
    step_log_f.write('step,loss,kept_ratio\n')
    step_log_every = 10

    for epoch in range(epochs):
        module.train()
        epoch_loss = torch.zeros((), device=device)
        epoch_kept = torch.zeros((), device=device)
        n_batches  = 0
        step_in_epoch = 0

        for indices, x, ch_mask in tqdm(loader, desc=f'epoch {epoch+1}/{epochs}'):
            indices = indices.to(device)
            x       = x.to(device)
            ch_mask = ch_mask.to(device)
            B, C_max, T = x.shape

            domains = [dataset.domain_of(int(i)) for i in indices.tolist()]
            packed = batch_to_moirai(
                x, ch_mask,
                MIN_MASK_RATIO, MAX_MASK_RATIO, rng, device,
                domains=domains,
            )
            if packed is None:
                continue

            if mode == 'baseline':
                distr = module(
                    target          = packed['target'],
                    observed_mask   = packed['observed_mask'],
                    sample_id       = packed['sample_id'],
                    time_id         = packed['time_id'],
                    variate_id      = packed['variate_id'],
                    prediction_mask = packed['prediction_mask'],
                    patch_size      = packed['patch_size'],
                )
                loss = packed_nll_loss(
                    distr,
                    packed['target'],
                    packed['prediction_mask'],
                    packed['observed_mask'],
                    channel_weight=None,
                )
                kept_ratio = torch.ones((), device=device)

            elif mode == 'rho_cm':
                # Token-level RHO: ρ[t] = current_NLL[t] - ref_NLL[t] for each
                # prediction token t (= a (sample, variate, time-patch) triple
                # in the packed format). Drop bottom drop_pct% of prediction
                # tokens across the batch. This matches Moirai's loss unit
                # exactly — packed_nll_loss aggregates per-token NLL — and
                # parallels MOMENT's patch-level / Timer's token-level RHO.

                # ── Live ref mode: build a batch-local (B, C99, T) ref table
                #    on the fly using the per-sample (lookback, suffix) split
                #    that batch_to_moirai actually picked. Lookup code below
                #    treats it identically to the static timepoint table, but
                #    indices into the small batch-local table (sids = b in 0..B).
                if ref_live_model is not None:
                    with torch.no_grad():
                        C99 = ref_live_C99
                        C_in = min(C_max, C99)
                        live_tbl = torch.zeros(B, C99, T, device=device)
                        # Per-sample suffix split — recover from packed metadata.
                        # Each sample b has unique psize_b and a contiguous suffix
                        # marked by pred_flag for its tokens.
                        pred_flag_all = packed['_pred_flag']           # (S_total,) bool
                        sample_id_all = packed['_sample_id_flat']      # (S_total,)
                        psize_all     = packed['patch_size'].squeeze(0)# (S_total,)
                        for b in range(B):
                            tok_mask_b = sample_id_all == b
                            if not tok_mask_b.any():
                                continue
                            psize_b = int(psize_all[tok_mask_b][0].item())
                            n_patches_b = T // psize_b
                            # n_masked = number of distinct time positions with pred_flag True
                            pred_b   = pred_flag_all[tok_mask_b]
                            time_b   = packed['time_id'].squeeze(0)[tok_mask_b]
                            masked_times = time_b[pred_b].unique()
                            if masked_times.numel() == 0:
                                continue
                            n_masked_b = int(masked_times.numel())
                            S_b = n_masked_b * psize_b
                            L_b = T - S_b
                            if L_b < 1 or S_b < 1:
                                continue

                            x_b = x[b, :C_in].reshape(C_in, T)         # (C_in, T)
                            x_lb_b = x_b[:, :L_b]
                            target_b = x_b[:, L_b:L_b + S_b]
                            mu_b  = x_lb_b.mean(dim=1, keepdim=True)
                            sig_raw_b = x_lb_b.std(dim=1, keepdim=True)
                            nonconst_b = (sig_raw_b.squeeze(-1) > 1e-2)
                            sig_b = sig_raw_b.clamp(min=1e-2)
                            x_lb_n = (x_lb_b - mu_b) / sig_b
                            target_n = (target_b - mu_b) / sig_b

                            pred = ref_live_model(x_lb_n, max_pred_len=S_b)  # (C_in, S_b)
                            sq_err = (pred - target_n) ** 2                  # (C_in, S_b)
                            sq_err = sq_err * nonconst_b.float().unsqueeze(-1)
                            live_tbl[b, :C_in, L_b:L_b + S_b] = sq_err
                    # Use this batch-local table via the same lookup code below.
                    # IMPORTANT: shadow ref_loss_table with batch-local one and
                    #            override sids with batch-local b indices (not
                    #            global dataset indices).
                    ref_loss_table_batch_local = live_tbl
                    C_eff = min(C_max, C99)
                elif per_domain_ref is not None:
                    # All per-domain tables share global_max_c
                    any_tbl = next(iter(per_domain_ref.values()))["table"]
                    C_eff = min(C_max, any_tbl.shape[1])
                    ref_loss_table_batch_local = None
                else:
                    C_eff = min(C_max, ref_loss_table.shape[1])
                    ref_loss_table_batch_local = None

                # Single forward; reused for both rho and the loss
                distr = module(
                    target          = packed['target'],
                    observed_mask   = packed['observed_mask'],
                    sample_id       = packed['sample_id'],
                    time_id         = packed['time_id'],
                    variate_id      = packed['variate_id'],
                    prediction_mask = packed['prediction_mask'],
                    patch_size      = packed['patch_size'],
                )

                pred_flag  = packed['_pred_flag']                 # (S_total,) bool
                orig_ch    = packed['_orig_channel_id_flat']      # (S_total,)
                sample_id_ = packed['_sample_id_flat']            # (S_total,)
                time_id    = packed['time_id'].squeeze(0)         # (S_total,) int — token's patch index
                psize_t    = packed['patch_size'].squeeze(0)      # (S_total,) int — patch_size at this token

                # MSE-based selection: training loss stays Mixture NLL (gradient),
                # but ρ is computed from POINT-PREDICTION MSE so it's directly
                # comparable to the DLinear MSE forecaster ref. distr.mean is
                # the 4-component weighted average — Moirai's natural point
                # prediction.
                with torch.no_grad():
                    mean_pred = distr.mean.squeeze(0)                         # (S_total, max_ps)
                    target_t  = packed['target'].squeeze(0)                   # (S_total, max_ps)
                    sq_err    = (mean_pred - target_t) ** 2                   # (S_total, max_ps)
                    arange    = torch.arange(sq_err.shape[1], device=device)
                    valid_p   = arange.unsqueeze(0) < psize_t.unsqueeze(1)    # (S_total, max_ps)
                    mse_tok   = (sq_err * valid_p).sum(dim=1) / psize_t.float()  # (S_total,)

                    # Look up ref NLL for the same (sample, channel, time-window).
                    # ref_loss_table is (N, C99, SEQ_LEN) of per-time-step NLL.
                    # For a token at time_id=p with patch_size=ps, its time
                    # window is [p*ps, (p+1)*ps). We average ref NLL over this
                    # window. Tokens whose channel exceeds C99 (no ref entry)
                    # are kept — token_weight stays 1 for them.
                    has_ref = pred_flag & (orig_ch < C_eff)
                    if has_ref.any():
                        # Global indices (used by static timepoint table)
                        sids   = indices[sample_id_[has_ref]]
                        # Batch-local indices (used by live mode's per-batch table)
                        sids_b = sample_id_[has_ref]
                        chs    = orig_ch[has_ref]
                        tps    = time_id[has_ref]
                        pss    = psize_t[has_ref]
                        ref_tok = torch.zeros(has_ref.sum(), device=device)

                        if per_domain_ref is not None:
                            # Per-domain ref: lookup is direct per-patch.
                            # Group tokens by domain (each sample maps to one domain).
                            sids_cpu = sids.cpu().numpy()
                            domains_b = [sample_to_domain_name_arr[i] for i in sids_cpu]
                            for dom in set(domains_b):
                                if dom not in per_domain_ref: continue
                                info = per_domain_ref[dom]
                                tbl = info['table']  # (G, C99, n_patches_dom)
                                ps_dom = info['patch_size']
                                # Mask of tokens whose sample is in this domain
                                dom_mask = torch.tensor(
                                    [d == dom for d in domains_b],
                                    dtype=torch.bool, device=device
                                )
                                if not dom_mask.any(): continue
                                s_d = sids[dom_mask]
                                c_d = chs[dom_mask]
                                t_d = tps[dom_mask]   # patch index (token position)
                                # Map global window idx -> local idx in domain table
                                local = torch.tensor(
                                    sample_to_local_idx[s_d.cpu().numpy()],
                                    dtype=torch.long, device=device,
                                )
                                if tbl.is_cuda:
                                    vals = tbl[local, c_d, t_d]
                                else:
                                    vals = tbl[local.cpu(), c_d.cpu(), t_d.cpu()].to(device)
                                ref_tok[dom_mask] = vals
                        elif ref_loss_table_batch_local is not None or ref_loss_table is not None:
                            # Pick which table + which sample-index to use:
                            #   live mode → batch-local (B, C99, T) tbl + sids_b
                            #   timepoint → global    (N, C99, T) tbl + sids
                            if ref_loss_table_batch_local is not None:
                                _tbl  = ref_loss_table_batch_local
                                _sids = sids_b
                            else:
                                _tbl  = ref_loss_table
                                _sids = sids
                            for ps_val in psize_t.unique().tolist():
                                sel = pss == ps_val
                                if not sel.any(): continue
                                s_sel = _sids[sel]
                                c_sel = chs[sel]
                                t_sel = tps[sel]
                                offsets = torch.arange(ps_val, device=device)
                                time_idx = t_sel.unsqueeze(1) * ps_val + offsets
                                if _tbl.is_cuda:
                                    rows = _tbl[
                                        s_sel.unsqueeze(1).expand(-1, ps_val),
                                        c_sel.unsqueeze(1).expand(-1, ps_val),
                                        time_idx,
                                    ]
                                else:
                                    rows = _tbl[
                                        s_sel.cpu().unsqueeze(1).expand(-1, ps_val),
                                        c_sel.cpu().unsqueeze(1).expand(-1, ps_val),
                                        time_idx.cpu(),
                                    ].to(device, non_blocking=True)
                                ref_tok[sel] = rows.mean(dim=1)
                    else:
                        ref_tok = torch.zeros(0, device=device)

                    # ρ per token (only where we have ref + prediction)
                    pred_mask = pred_flag & has_ref            # token participates in RHO
                    rho_pred  = torch.zeros(pred_flag.shape[0], device=device)
                    if pred_mask.any():
                        rho_pred[pred_mask] = mse_tok[pred_mask] - ref_tok

                    # Drop bottom drop_pct% of prediction tokens (by ρ ascending)
                    rho_kept = compute_rho_mask(rho_pred, pred_mask, drop_pct)
                    # rho_kept: True = keep prediction token; False = drop or non-prediction

                # Build per-token weight — drop excludes tokens from loss
                # (only prediction tokens can be dropped; non-prediction tokens
                # are unaffected — packed_nll_loss already excludes them via
                # prediction_mask, so weight=1 is fine for them).
                token_weight = torch.ones(pred_flag.shape[0], device=device)
                drop_tokens  = pred_flag & has_ref & (~rho_kept)
                token_weight[drop_tokens] = 0.0

                kept_ratio = (rho_kept.float().sum()
                              / pred_mask.float().sum().clamp(min=1))

                loss = packed_nll_loss(
                    distr,
                    packed['target'],
                    packed['prediction_mask'],
                    packed['observed_mask'],
                    channel_weight=token_weight,
                )
            else:
                raise ValueError(f'Unknown mode: {mode}')

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(module.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            # GPU-side accumulation — single .item() at epoch end avoids
            # per-step CUDA sync (saves a few us × thousands of steps).
            epoch_loss += loss.detach()
            epoch_kept += kept_ratio.detach()
            n_batches  += 1
            step_in_epoch += 1
            if step_in_epoch % step_log_every == 0:
                global_step = epoch * len(loader) + step_in_epoch
                step_log_f.write(f'{global_step},{loss.item():.6f},{kept_ratio.item():.4f}\n')
                step_log_f.flush()

        avg_loss = float((epoch_loss / max(n_batches, 1)).item())
        avg_kept = float((epoch_kept / max(n_batches, 1)).item())
        metrics['train_loss'].append(avg_loss)
        metrics['kept_ratio'].append(avg_kept)
        print(f'epoch {epoch+1}: loss={avg_loss:.4f}  kept={avg_kept*100:.1f}%')

        best_loss = save_epoch(out_dir, epoch + 1, module.state_dict(),
                               avg_loss, best_loss)

    step_log_f.close()
    dump_metrics(out_dir, metrics)
    print(f'\nDone. Best loss={best_loss:.4f}  ckpt: {out_dir}/best.pt')
    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# Zero-shot forecasting using pretrained MoiraiModule
# (ports MoiraiForecast._convert + _get_distr + _format_preds without Lightning)
# ─────────────────────────────────────────────────────────────────────────────

def _patched_seq_pad(patch_size, x, dim, left=True, value=None):
    if dim >= 0:
        dim = -x.ndim + dim
    pad_length = -x.size(dim) % patch_size
    pad = (pad_length, 0) if left else (0, pad_length)
    pad = (0, 0) * (abs(dim) - 1) + pad
    return torch.nn.functional.pad(x, pad, value=value)


def _moirai_convert(
    module: MoiraiModule,
    patch_size: int,
    past_target: torch.Tensor,           # (B, context_len, C)
    past_observed: torch.Tensor,         # (B, context_len, C) bool
    past_is_pad: torch.Tensor,           # (B, context_len) bool
    prediction_length: int,
    device: torch.device,
):
    """
    Ports MoiraiForecast._convert() without Lightning dependency.
    Returns (target, observed_mask, sample_id, time_id, variate_id, prediction_mask)
    each shaped (B, S, max_patch) or (B, S).
    """
    from einops import rearrange, reduce, repeat
    B = past_target.shape[0]
    C = past_target.shape[-1]
    max_ps = max(module.patch_sizes)
    context_tokens  = math.ceil(past_target.shape[1] / patch_size)
    pred_tokens     = math.ceil(prediction_length / patch_size)

    # time_id
    past_obs_pad = _patched_seq_pad(patch_size, past_observed, -2, left=True)
    past_seq_id  = reduce(past_obs_pad, '... (s p) d -> ... s', 'max', p=patch_size)
    past_seq_id  = torch.clamp(past_seq_id.cummax(-1).values.cumsum(-1) - 1, min=0)  # (B, ctx_tok)
    future_seq_id = (
        repeat(torch.arange(pred_tokens, device=device), 'f -> b f', b=B)
        + past_seq_id.max(-1, keepdim=True).values + 1
    )  # (B, pred_tok)

    def pad_patch(t, left=True):
        return torch.nn.functional.pad(t, (0, max_ps - patch_size))

    # past target patches: (B, C*ctx_tok, max_ps)
    past_pad   = _patched_seq_pad(patch_size, past_target, -2, left=True)
    past_tgt   = rearrange(past_pad,  'b (s p) d -> b (d s) p', p=patch_size)
    past_tgt   = pad_patch(past_tgt)

    # future target patches (zeros, will be masked): (B, C*pred_tok, max_ps)
    future_target = torch.zeros(B, prediction_length, C, dtype=past_target.dtype, device=device)
    future_pad  = _patched_seq_pad(patch_size, future_target, -2, left=False)
    future_tgt  = rearrange(future_pad, 'b (s p) d -> b (d s) p', p=patch_size)
    future_tgt  = pad_patch(future_tgt)

    target = torch.cat([past_tgt, future_tgt], dim=1)  # (B, C*(ctx+pred)_tok, max_ps)

    # observed_mask
    past_obs_p  = rearrange(_patched_seq_pad(patch_size, past_observed, -2, left=True),
                            'b (s p) d -> b (d s) p', p=patch_size)
    past_obs_p  = pad_patch(past_obs_p)
    future_obs  = torch.ones(B, pred_tokens * C, max_ps, dtype=torch.bool, device=device)
    future_obs[:, :, patch_size:] = False
    observed_mask = torch.cat([past_obs_p, future_obs], dim=1)

    # sample_id (1 = real, 0 = pad)
    is_pad_pad  = _patched_seq_pad(patch_size, past_is_pad, -1, left=True, value=1)
    past_sid    = reduce((is_pad_pad == 0).int(), 'b (s p) -> b s', 'max', p=patch_size)
    past_sid    = repeat(past_sid, 'b s -> b (d s)', d=C)
    future_sid  = torch.ones(B, pred_tokens * C, dtype=torch.long, device=device)
    sample_id   = torch.cat([past_sid, future_sid], dim=1)

    # time_id
    past_tid   = repeat(past_seq_id,   'b s -> b (d s)', d=C)
    future_tid = repeat(future_seq_id, 'b s -> b (d s)', d=C)
    time_id    = torch.cat([past_tid, future_tid], dim=1)

    # variate_id
    past_vid   = repeat(torch.arange(C, device=device), 'd -> b (d s)', b=B, s=context_tokens)
    future_vid = repeat(torch.arange(C, device=device), 'd -> b (d s)', b=B, s=pred_tokens)
    variate_id = torch.cat([past_vid, future_vid], dim=1)

    # prediction_mask: False for context, True for future
    pred_mask  = torch.cat([
        torch.zeros(B, C * context_tokens, dtype=torch.bool, device=device),
        torch.ones (B, C * pred_tokens,    dtype=torch.bool, device=device),
    ], dim=1)

    return target, observed_mask, sample_id, time_id, variate_id, pred_mask, context_tokens, pred_tokens


def _select_best_patch_size(
    module: MoiraiModule,
    past_target: torch.Tensor,
    past_observed: torch.Tensor,
    past_is_pad: torch.Tensor,
    prediction_length: int,
    device: torch.device,
) -> int:
    """Pick patch size with lowest val NLL on a single batch (cheap proxy).
    Called once per zero-shot evaluation rather than once per batch."""
    best_ps, best_loss = None, float('inf')
    with torch.no_grad():
        for ps in module.patch_sizes:
            tgt, obs, sid, tid, vid, pmask, _ctx_tok, _pred_tok = _moirai_convert(
                module, ps, past_target, past_observed, past_is_pad,
                prediction_length, device,
            )
            ps_t  = torch.full_like(tid, ps, dtype=torch.long)
            distr = module(tgt, obs, sid, tid, vid, pmask, ps_t)
            nll   = -distr.log_prob(tgt)
            mask  = pmask.unsqueeze(-1) * obs
            loss  = (nll * mask).sum() / mask.sum().clamp(min=1)
            if loss.item() < best_loss:
                best_loss, best_ps = loss.item(), ps
    return best_ps


def moirai_zero_shot_predict(
    module: MoiraiModule,
    past_target: torch.Tensor,    # (B, context_len, C)
    past_observed: torch.Tensor,  # (B, context_len, C) bool
    past_is_pad: torch.Tensor,    # (B, context_len) bool
    prediction_length: int,
    num_samples: int,
    device: torch.device,
    patch_size: str | int = 'auto',
) -> torch.Tensor:
    """
    Zero-shot forecast using pretrained MoiraiModule.
    Returns median prediction: (B, prediction_length, C).
    """
    from einops import rearrange

    C = past_target.shape[-1]
    max_ps = max(module.patch_sizes)

    if patch_size == 'auto':
        patch_size = _select_best_patch_size(
            module, past_target, past_observed, past_is_pad,
            prediction_length, device,
        )

    tgt, obs, sid, tid, vid, pmask, ctx_tok, pred_tok = _moirai_convert(
        module, patch_size, past_target, past_observed, past_is_pad,
        prediction_length, device
    )
    ps_t = torch.ones_like(tid, dtype=torch.long) * patch_size

    with torch.no_grad():
        distr = module(tgt, obs, sid, tid, vid, pmask, ps_t)
        samples = distr.sample(torch.Size((num_samples,)))  # (num_samples, B, S, max_ps)

    # extract future prediction tokens: positions C*ctx_tok .. C*(ctx_tok+pred_tok)
    start = C * ctx_tok
    end   = start + C * pred_tok
    samples = samples[..., start:end, :patch_size]          # (num_samples, B, C*pred_tok, patch_size)
    samples = rearrange(samples, 'n b (d s) p -> n b (s p) d', d=C)
    samples = samples[..., :prediction_length, :]           # (num_samples, B, pred_len, C)
    return samples.median(dim=0).values                     # (B, pred_len, C)


def eval_zero_shot(
    ckpt_path: Path | None,
    dataset_name: str,
    horizon: int,
    seed: int,
    device: torch.device,
    batch_size: int = 32,
    num_samples: int = 100,
    context_length: int = SEQ_LEN,
) -> dict:
    torch.manual_seed(seed); np.random.seed(seed)
    print(f'\n[Zero-shot] {dataset_name} H={horizon}  ckpt={ckpt_path}')

    _, _, test_ds = prepare_forecast_datasets(
        dataset_name, DATA_DIR, context_length, horizon, stride=1
    )

    module = MoiraiModule(
        distr_output=_build_distr_output(),
        d_model=D_MODEL, num_layers=NUM_LAYERS,
        patch_sizes=PATCH_SIZES, max_seq_len=MAX_SEQ_LEN,
        attn_dropout_p=0.0, dropout_p=0.0, scaling=True,
    ).to(device)

    if ckpt_path is not None:
        ckpt = torch.load(ckpt_path, map_location=device)
        module.load_state_dict(ckpt['state_dict'], strict=True)
        print(f'  loaded: {ckpt_path.name}')

    module.eval()
    mse_list, mae_list = [], []
    selected_ps = None  # cached after first batch

    with torch.no_grad():
        for batch in DataLoader(test_ds, batch_size=batch_size, num_workers=0):
            x_b, y_b, *_ = batch
            x_b = x_b.to(device)   # (B, C, context_len)
            y_b = y_b.to(device)   # (B, C, horizon)
            B, C, T = x_b.shape

            # MoiraiForecast expects (B, T, C)
            past_target   = x_b.permute(0, 2, 1).float()
            past_observed = torch.ones(B, T, C, dtype=torch.bool, device=device)
            past_is_pad   = torch.zeros(B, T, dtype=torch.bool, device=device)

            if selected_ps is None:
                selected_ps = _select_best_patch_size(
                    module, past_target, past_observed, past_is_pad,
                    horizon, device,
                )
                print(f'  selected patch_size={selected_ps}')

            pred = moirai_zero_shot_predict(
                module, past_target, past_observed, past_is_pad,
                prediction_length=horizon, num_samples=num_samples,
                device=device, patch_size=selected_ps,
            )  # (B, horizon, C)

            pred_bc = pred.permute(0, 2, 1)  # (B, C, horizon)
            mse_list.append(((pred_bc - y_b) ** 2).mean().item())
            mae_list.append((pred_bc - y_b).abs().mean().item())

    mse = float(np.mean(mse_list))
    mae = float(np.mean(mae_list))
    print(f'  → MSE={mse:.4f}  MAE={mae:.4f}')
    return {'dataset': dataset_name, 'horizon': horizon, 'test_mse': mse, 'test_mae': mae}


# ─────────────────────────────────────────────────────────────────────────────
# Downstream evaluation: linear probe via MoiraiModule encoder
# ─────────────────────────────────────────────────────────────────────────────

def eval_forecasting(
    ckpt_path: Path | None,
    dataset_name: str,
    horizon: int,
    seed: int,
    device: torch.device,
    probe_epochs: int = 5,
    probe_lr: float  = 1e-4,
    batch_size: int  = 64,
) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    print(f'\n[Eval] {dataset_name} H={horizon}  ckpt={ckpt_path}')

    train_ds, val_ds, test_ds = prepare_forecast_datasets(
        dataset_name, DATA_DIR, SEQ_LEN, horizon, stride=1
    )

    # Load encoder
    distr_output = _build_distr_output()
    module = MoiraiModule(
        distr_output   = distr_output,
        d_model        = D_MODEL,
        num_layers     = NUM_LAYERS,
        patch_sizes    = PATCH_SIZES,
        max_seq_len    = MAX_SEQ_LEN,
        attn_dropout_p = 0.0,
        dropout_p      = 0.0,
        scaling        = True,
    ).to(device)

    if ckpt_path is not None:
        ckpt  = torch.load(ckpt_path, map_location=device)
        state = ckpt['state_dict']
        module.load_state_dict(state, strict=False)
        print(f'  loaded checkpoint: {ckpt_path.name}')

    # Freeze encoder, add linear forecasting head
    for p in module.parameters():
        p.requires_grad_(False)

    head = nn.Linear(D_MODEL, horizon).to(device)

    def encode_batch(x_in):
        """x_in: (B, C, T) → repr: (B, C, D_MODEL)"""
        B, C, T = x_in.shape
        patch_size = PATCH_SIZES[2]  # 32, fixed for eval
        n_patches  = T // patch_size
        patches    = x_in.reshape(B * C, n_patches, patch_size)

        target_in = patches.unsqueeze(0)     # (1, B*C*n_patches, patch_size)
        obs_in    = torch.ones_like(target_in, dtype=torch.bool)
        pred_in   = torch.zeros(1, B * C * n_patches, dtype=torch.bool, device=device)
        s_id      = torch.arange(B * C, device=device).repeat_interleave(n_patches).unsqueeze(0)
        t_id      = torch.arange(n_patches, device=device).repeat(B * C).unsqueeze(0)
        v_id      = torch.zeros(1, B * C * n_patches, dtype=torch.long, device=device)
        ps        = torch.full((1, B * C * n_patches), patch_size, dtype=torch.long, device=device)

        # Reshape to (1, S, P)
        target_in = target_in.reshape(1, B * C * n_patches, patch_size)
        obs_in    = obs_in.reshape(1, B * C * n_patches, patch_size)

        from uni2ts.common.torch_util import mask_fill, packed_attention_mask
        loc, scale = module.scaler(target_in, obs_in * ~pred_in.unsqueeze(-1), s_id, v_id)
        scaled = (target_in - loc) / scale
        reprs  = module.in_proj(scaled, ps)
        reprs  = module.encoder(reprs, packed_attention_mask(s_id), time_id=t_id, var_id=v_id)
        # Average over patches per (sample, channel): reprs (1, S, D)
        reprs  = reprs.squeeze(0)   # (S, D)
        reprs  = reprs.reshape(B * C, n_patches, D_MODEL).mean(dim=1)  # (B*C, D)
        return reprs.reshape(B, C, D_MODEL)

    optimizer = torch.optim.Adam(head.parameters(), lr=probe_lr)

    for epoch in range(probe_epochs):
        head.train()
        epoch_loss, n = 0.0, 0
        for batch in DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0):
            x_b, y_b, *_ = batch
            x_b = x_b.to(device)
            y_b = y_b.to(device)
            with torch.no_grad():
                repr_bc = encode_batch(x_b)         # (B, C, D)
            pred = head(repr_bc.mean(dim=1))        # (B, horizon)
            loss = ((pred - y_b.mean(dim=1)) ** 2).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n += 1
        print(f'  probe epoch {epoch+1}/{probe_epochs}: loss={epoch_loss/n:.4f}')

    head.eval()
    mse_list, mae_list = [], []
    with torch.no_grad():
        for batch in DataLoader(test_ds, batch_size=batch_size, num_workers=0):
            x_b, y_b, *_ = batch
            x_b = x_b.to(device)
            y_b = y_b.to(device)
            repr_bc = encode_batch(x_b)
            pred    = head(repr_bc.mean(dim=1))
            y_mean  = y_b.mean(dim=1)
            mse_list.append(((pred - y_mean) ** 2).mean().item())
            mae_list.append((pred - y_mean).abs().mean().item())

    mse = float(np.mean(mse_list))
    mae = float(np.mean(mae_list))
    print(f'  → MSE={mse:.4f}  MAE={mae:.4f}')
    return {'dataset': dataset_name, 'horizon': horizon, 'test_mse': mse, 'test_mae': mae}


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _moirai_zero_shot_eval_fn(ckpt_path, ds, h, *, seed, device,
                              batch_size, num_samples):
    return eval_zero_shot(ckpt_path, ds, h, seed, device,
                          batch_size=batch_size, num_samples=num_samples)


def _moirai_probe_eval_fn(ckpt_path, ds, h, *, seed, device, probe_epochs):
    return eval_forecasting(ckpt_path, ds, h, seed, device,
                            probe_epochs=probe_epochs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['baseline', 'rho_cm', 'eval', 'eval_sweep',
                                           'eval_zero_shot', 'eval_zero_shot_sweep'], default='rho_cm')
    parser.add_argument('--epochs',     type=int,   default=2)
    parser.add_argument('--batch-size', type=int,   default=32)
    parser.add_argument('--lr',         type=float, default=1e-3)
    parser.add_argument('--drop-pct',   type=float, default=10.0)
    parser.add_argument('--ref-epochs', type=int,   default=3)
    parser.add_argument('--ref-mode',   type=str,   default='timepoint',
                        choices=['timepoint', 'per_domain', 'live'])
    parser.add_argument('--max-series', type=int,   default=None)
    parser.add_argument('--out-dir',    type=str,   default=None)
    parser.add_argument('--seed',       type=int,   default=42)
    # Eval
    parser.add_argument('--ckpt',              type=str, default=None)
    parser.add_argument('--eval-dataset',      type=str, default='ETTh1')
    parser.add_argument('--eval-horizon',      type=int, default=96)
    parser.add_argument('--probe-epochs',      type=int, default=5)
    parser.add_argument('--probe-lr',          type=float, default=1e-4)
    # Eval sweep / zero-shot
    parser.add_argument('--baseline-ckpt',     type=str, default=None)
    parser.add_argument('--rho-cm-ckpt',       type=str, default=None)
    parser.add_argument('--eval-datasets',     type=str, default='ETTh1,ETTh2,ETTm1,ETTm2,weather,exchange')
    parser.add_argument('--eval-horizons',     type=str, default='96,192,336,720')
    parser.add_argument('--num-samples',       type=int, default=100,
                        help='Number of MC samples for zero-shot prediction')
    parser.add_argument('--context-length',    type=int, default=SEQ_LEN,
                        help='Context window length for zero-shot eval')

    args   = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    if args.out_dir is None:
        args.out_dir = f'results_moirai/{args.mode}'

    if args.mode in ('baseline', 'rho_cm'):
        pretrain(
            mode       = args.mode,
            epochs     = args.epochs,
            batch_size = args.batch_size,
            lr         = args.lr,
            drop_pct   = args.drop_pct,
            ref_epochs = args.ref_epochs,
            out_dir    = Path(args.out_dir),
            seed       = args.seed,
            max_series = args.max_series,
            device     = device,
            ref_mode   = args.ref_mode,
        )
    elif args.mode == 'eval':
        eval_forecasting(
            ckpt_path    = Path(args.ckpt) if args.ckpt else None,
            dataset_name = args.eval_dataset,
            horizon      = args.eval_horizon,
            seed         = args.seed,
            device       = device,
            probe_epochs = args.probe_epochs,
            probe_lr     = args.probe_lr,
        )
    elif args.mode == 'eval_sweep':
        ckpt_paths = {}
        if args.baseline_ckpt:
            ckpt_paths['baseline'] = Path(args.baseline_ckpt)
        if args.rho_cm_ckpt:
            ckpt_paths['rho_cm'] = Path(args.rho_cm_ckpt)
        run_eval_sweep(
            _moirai_probe_eval_fn,
            ckpt_paths   = ckpt_paths,
            datasets     = args.eval_datasets.split(','),
            horizons     = [int(h) for h in args.eval_horizons.split(',')],
            title        = 'LINEAR PROBE RESULTS',
            out_filename = 'linear_probe_results.json',
            out_dir      = args.out_dir,
            seed         = args.seed,
            device       = device,
            probe_epochs = args.probe_epochs,
        )
    elif args.mode == 'eval_zero_shot':
        eval_zero_shot(
            ckpt_path      = Path(args.ckpt) if args.ckpt else None,
            dataset_name   = args.eval_dataset,
            horizon        = args.eval_horizon,
            seed           = args.seed,
            device         = device,
            batch_size     = args.batch_size,
            num_samples    = args.num_samples,
            context_length = args.context_length,
        )
    elif args.mode == 'eval_zero_shot_sweep':
        ckpt_paths = {}
        if args.baseline_ckpt:
            ckpt_paths['baseline'] = Path(args.baseline_ckpt)
        if args.rho_cm_ckpt:
            ckpt_paths['rho_cm'] = Path(args.rho_cm_ckpt)
        if not ckpt_paths and args.ckpt:
            ckpt_paths['model'] = Path(args.ckpt)
        run_eval_sweep(
            _moirai_zero_shot_eval_fn,
            ckpt_paths   = ckpt_paths,
            datasets     = args.eval_datasets.split(','),
            horizons     = [int(h) for h in args.eval_horizons.split(',')],
            title        = 'ZERO-SHOT RESULTS',
            out_filename = 'zero_shot_results.json',
            out_dir      = args.out_dir,
            seed         = args.seed,
            device       = device,
            batch_size   = args.batch_size,
            num_samples  = args.num_samples,
        )


if __name__ == '__main__':
    main()
