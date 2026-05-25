"""Timer pretraining (baseline + SPM variants).

Architecture
------------
Decoder-only causal Transformer (GPT-style), channel-independent.
Patch-based next-token prediction with MSE loss, RoPE positional embeddings.
Random-init from `OpenLTM/Timer-base` config (~84M params, patch_len=96).

Modes
-----
- baseline          : standard next-patch MSE (no selection).
- threshold_rho     : SPM with calibrated threshold τ.
                      Keep tokens with ρ = current_MSE − ref_MSE > τ.
                      τ is fit on the first `--calib-batches` to hit
                      `--target-keep-pct` (k).
- random_mask       : ablation, random per-token mask matching `--drop-pct`.
- top_loss_drop     : ablation, drop tokens with the highest current loss.
- bottom_loss_drop  : ablation, drop tokens with the lowest current loss.

Reference model: causal next-patch DLinear (see rho_lib/ref/dlinear_causal.py),
trained briefly on the same UTSD corpus for `--ref-epochs` epochs.

Eval
----
- eval_zero_shot, eval_zero_shot_sweep : direct forecasting on benchmark datasets.

Best config (see paper Table N)
-------------------------------
baseline       : --epochs 3 --batch-size 512 --lr 3e-4
SPM (calib)    : same + --target-keep-pct 0.4 --calib-batches 200 --ref-epochs 2
"""

import argparse
import math
import random
import sys
import time
from pathlib import Path

# Allow `from rho_lib...` to resolve when this script is launched from anywhere.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

# ── rho_lib (shared scaffolding) ──────────────────────────────────────────────
from rho_lib.data.utsd     import (
    UTSDPretrainDataset, make_collate_fn, load_utsd, compute_channel_cap,
    UTSDPretrainDatasetUnivariate, make_collate_fn_univariate,
    load_or_build_cached_univariate,
)
from rho_lib.data.forecast import prepare_forecast_datasets
from rho_lib.ref.dlinear_causal import build_ref_loss_causal
from rho_lib.rho.mask      import compute_rho_mask
from rho_lib.eval.sweep    import run_eval_sweep
from rho_lib.train.checkpointing import save_epoch, dump_metrics
from rho_lib.train.schedule import warmup_cosine_with_min_lr_lambda

# ─────────────────────────────────────────────────────────────────────────────
# Constants — Timer-base-84M architecture (thuml/timer-base-84m) with shorter
# context tuned for UTSD-12G distribution. UTSD median series length is ~896
# timesteps and only ~5% of series exceed 2880, so we keep the official
# patch_len=96 (token meaning preserved) and shorten n_patches 30→8 to make
# ~80%+ of series eligible while reducing attention O(N^2) cost ~14×.
# ─────────────────────────────────────────────────────────────────────────────
PATCH_LEN    = 96           # input_token_len (HF config "input_token_len": 96) — unchanged
TOKEN_NUM    = 8            # n_patches per sequence (OpenLTM default 30; reduced for UTSD)
SEQ_LEN      = PATCH_LEN * TOKEN_NUM   # 768 timesteps = 8 patches × 96
MAX_PATCHES  = SEQ_LEN // PATCH_LEN    # 8

UTSD_PATH    = Path(__file__).resolve().parent.parent / "data" / 'utsd_repo' / 'UTSD-12G'
DATA_DIR     = Path(__file__).resolve().parent.parent / "data"

# ─────────────────────────────────────────────────────────────────────────────
# Timer-base architecture (HF thuml/timer-base-84m config.json)
# ─────────────────────────────────────────────────────────────────────────────
D_MODEL    = 1024           # hidden_size
NUM_HEADS  = 8              # num_attention_heads
NUM_LAYERS = 8              # num_hidden_layers
D_FF       = 2048           # intermediate_size
DROPOUT    = 0.0            # attention_dropout
ROPE_THETA = 10000          # rope_theta

# ─────────────────────────────────────────────────────────────────────────────
# Timer model: decoder-only causal Transformer with RoPE
# ─────────────────────────────────────────────────────────────────────────────

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_len=10000, theta=ROPE_THETA):
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq, persistent=False)
        self.max_len = max_len
        # Pre-compute cos/sin for the full max_len once. Buffers move with the
        # module to GPU and are reused across all layers and forward calls.
        t = torch.arange(max_len).float()
        freqs = torch.outer(t, inv_freq)
        emb   = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer('cos_cache', emb.cos(), persistent=False)
        self.register_buffer('sin_cache', emb.sin(), persistent=False)

    def forward(self, seq_len, device):
        if seq_len > self.max_len:
            # Fallback for unusually long sequences (shouldn't happen with default max_len=10000)
            t = torch.arange(seq_len, device=device).float()
            freqs = torch.outer(t, self.inv_freq.to(device))
            emb   = torch.cat([freqs, freqs], dim=-1)
            return emb.cos(), emb.sin()
        return self.cos_cache[:seq_len], self.sin_cache[:seq_len]


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary(x, cos, sin):
    # x: (B, H, T, head_dim)
    return x * cos.unsqueeze(0).unsqueeze(0) + rotate_half(x) * sin.unsqueeze(0).unsqueeze(0)


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.0):
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.head_dim  = d_model // num_heads
        self.qkv  = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.drop = nn.Dropout(dropout)
        self.rope = RotaryEmbedding(self.head_dim)

    def forward(self, x):
        B, T, D = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)          # each (B, T, H, head_dim)
        q = q.transpose(1, 2)                # (B, H, T, head_dim)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        cos, sin = self.rope(T, x.device)
        q = apply_rotary(q, cos, sin)
        k = apply_rotary(k, cos, sin)

        # PyTorch ≥2.0 SDPA: FlashAttention/mem-efficient backend, causal-aware,
        # avoids materialising the (T,T) attention matrix and `triu(ones)` mask.
        out = F.scaled_dot_product_attention(
            q, k, v,
            is_causal=True,
            dropout_p=self.drop.p if self.training else 0.0,
        )                                     # (B, H, T, head_dim)
        out = out.transpose(1, 2).reshape(B, T, D)
        return self.proj(out)


class TimerBlock(nn.Module):
    """
    Post-norm decoder block matching thuml/timer-base-84m TimerDecoderLayer:
    norm1 applied AFTER attention residual, norm2 AFTER FFN residual.
    """
    def __init__(self, d_model, num_heads, d_ff, dropout=0.0):
        super().__init__()
        self.attn  = CausalSelfAttention(d_model, num_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff_gate = nn.Linear(d_model, d_ff, bias=False)
        self.ff_up   = nn.Linear(d_model, d_ff, bias=False)
        self.ff_down = nn.Linear(d_ff, d_model, bias=False)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop    = nn.Dropout(dropout)

    def forward(self, x):
        x = self.norm1(x + self.attn(x))
        h = F.silu(self.ff_gate(x)) * self.ff_up(x)
        x = self.norm2(x + self.drop(self.ff_down(h)))
        return x


class TimerModel(nn.Module):
    """
    Timer-base-84M: decoder-only causal Transformer (GPT-style).
    Channel-independent — each (sample, channel) is an independent patch sequence.

    Pretraining = autoregressive next-patch prediction:
        position t observes patches [0..t] and predicts patch t+1.
    No mask_token, no random masking; causal mask in attention only.

    RevIN-style instance normalization (OpenLTM --use_norm) is applied to each
    sequence at forward time and reversed on output.

    Input  : (N, n_patches, patch_len)  — N = real (sample, channel) pairs
    Output : (N, n_patches, patch_len)  — output at position t predicts patch t+1
    """
    def __init__(self, patch_len=PATCH_LEN, d_model=D_MODEL, num_heads=NUM_HEADS,
                 num_layers=NUM_LAYERS, d_ff=D_FF, dropout=DROPOUT, use_norm=True):
        super().__init__()
        self.patch_len = patch_len
        self.d_model   = d_model
        self.use_norm  = use_norm

        self.patch_embed = nn.Linear(patch_len, d_model, bias=False)
        self.blocks = nn.ModuleList([
            TimerBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, patch_len, bias=True)

    def forward(self, x):
        # x: (N, n_patches, patch_len)
        if self.use_norm:
            flat = x.reshape(x.shape[0], -1)               # (N, n_patches*patch_len)
            mu   = flat.mean(dim=-1, keepdim=True)
            sig  = flat.std(dim=-1, keepdim=True) + 1e-5
            flat = (flat - mu) / sig
            x_in = flat.reshape_as(x)
        else:
            x_in, mu, sig = x, None, None

        h = self.patch_embed(x_in)
        for block in self.blocks:
            h = block(h)
        h = self.norm(h)
        out = self.head(h)                                 # (N, n_patches, patch_len)

        if self.use_norm:
            out_flat = out.reshape(out.shape[0], -1) * sig + mu
            out = out_flat.reshape_as(out)
        return out


# ─────────────────────────────────────────────────────────────────────────────
# Pretraining loop
# ─────────────────────────────────────────────────────────────────────────────

def pretrain(
    mode, epochs, batch_size, lr, drop_pct, ref_epochs,
    out_dir, seed, max_series, max_windows, device,
    lambda_clip_max=None, meta_lr=1e-2, init_lambda=1.0,
    threshold_tau=0.0, univariate=False,
    target_keep_pct=None, calib_batches=200,
    save_every_n_steps=None,
):
    torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Dataset ───────────────────────────────────────────────────────────────
    if univariate:
        # OpenLTM-style SOM: each (series, channel) becomes a univariate window.
        # Uses pickle cache so 4-min HF iteration is paid once across all runs
        # with the same (seq_len, stride, standardize, max_series) config.
        dataset = load_or_build_cached_univariate(
            UTSD_PATH, seq_len=SEQ_LEN, stride=SEQ_LEN,
            standardize_per_series=True,
            max_series=max_series, max_windows=max_windows,
        )
        channel_cap = 1   # each batch item is already (1, sl)
        print(f'[Data] univariate mode (SOM): {len(dataset)} windows, no channel cap')
        collate = make_collate_fn_univariate(SEQ_LEN)
    else:
        print('[Data] Loading UTSD...')
        hf_ds = load_utsd(UTSD_PATH, max_series=max_series)
        print(f'[Data] {len(hf_ds):,} rows loaded')
        dataset = UTSDPretrainDataset(
            hf_ds, seq_len=SEQ_LEN, stride=SEQ_LEN,
            max_series=max_series, max_windows=max_windows,
            standardize_per_series=True,                # Timer default: pre-norm
        )
        # Cap channels at 99-percentile to bound N_real per batch. Heavy MVTS
        # series (traffic 862ch, ECL 321ch) otherwise produce N_real * 8 heads
        # > 65535, which exceeds CUDA's grid Y dim limit and crashes SDPA.
        channel_cap = compute_channel_cap(dataset, percentile=99.0, per='window')
        print(f'[Data] channel cap (p99): {channel_cap}; capping at p99')
        collate = make_collate_fn(SEQ_LEN, channel_cap)

    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        num_workers=0, collate_fn=collate,
        pin_memory=False, drop_last=True,
    )
    print(f'[Data] {len(loader)} batches/epoch')

    # ── Model ─────────────────────────────────────────────────────────────────
    print('[Model] Initializing Timer-base (random init)...')
    model = TimerModel().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'[Model] {n_params/1e6:.1f}M parameters')

    # Patched (2026-05-14): added warmup + per-step cosine scheduler.
    # Original OpenLTM setting (lr=5e-5, no warmup, cosine over epochs) wasn't
    # giving monotone training loss; switched to AdamW (wd=1e-2) + warmup_cosine.
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    total_steps  = len(loader) * epochs
    warmup_steps = min(1000, max(1, int(total_steps * 0.05)))
    scheduler    = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        warmup_cosine_with_min_lr_lambda(
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
            warmup_start_lr=1e-6, base_lr=lr, min_lr=lr * 0.05,
        ),
    )
    print(f'[Optim] AdamW lr={lr} wd=1e-2  warmup={warmup_steps}/{total_steps}')

    # ── RHO-CM reference ──────────────────────────────────────────────────────
    ref_loss_table = None
    if mode in ('rho_cm', 'soft_rho_learnable', 'soft_rho_fixed', 'threshold_rho'):
        t0 = time.time()
        # Timer's ref builder: causal next-patch DLinear with capped collate
        ref_collate = make_collate_fn(SEQ_LEN, channel_cap=channel_cap)
        ref_bs      = 256   # fixed: ref model is tiny, large batch reduces step count
        ref_loss_table = build_ref_loss_causal(
            dataset, seq_len=SEQ_LEN, patch_len=PATCH_LEN,
            ref_epochs=ref_epochs, batch_size=ref_bs,
            device=device, collate_fn=ref_collate, seed=seed,
            out_dir=out_dir,
        )
        print(f'[RHO-CM] ref built in {time.time()-t0:.1f}s')
        # Move to GPU once if it fits comfortably (≤2 GB on H200/A100).
        # Avoids per-batch CPU→GPU copies during training.
        ref_bytes = ref_loss_table.element_size() * ref_loss_table.nelement()
        if torch.cuda.is_available() and ref_bytes < 2 * (1 << 30):
            ref_loss_table = ref_loss_table.to(device, non_blocking=True)
            print(f'[RHO-CM] ref table on GPU ({ref_bytes/1e6:.1f} MB)')
        else:
            ref_loss_table = ref_loss_table.pin_memory()
            print(f'[RHO-CM] ref table on pinned CPU ({ref_bytes/1e6:.1f} MB)')

    # ── Learnable lambda (soft_rho_learnable only) ───────────────────────────
    # Use separate Adam optimizer so it doesn't interact with the model's
    # warmup-cosine LambdaLR scheduler (which is registered for one param group).
    if mode == 'soft_rho_learnable':
        lambda_ref = nn.Parameter(torch.tensor(float(init_lambda), device=device))
        log_temp   = nn.Parameter(torch.tensor(0.0, device=device))  # temp = exp(log_temp), init = 1.0
        meta_optimizer = torch.optim.Adam([lambda_ref, log_temp], lr=meta_lr)
        print(f'[soft_rho_learnable] lambda init={init_lambda}, temp init=1.0  '
              f'meta_lr={meta_lr}  clip_max={lambda_clip_max}')
    else:
        lambda_ref = None
        log_temp   = None
        meta_optimizer = None

    # ── Training ──────────────────────────────────────────────────────────────
    metrics   = {'train_loss': [], 'kept_ratio': [], 'lambda': [], 'temp': []}
    best_loss = float('inf')

    # Per-step loss log for convergence plots (separate from tqdm)
    step_log_path = out_dir / 'step_loss.csv'
    step_log_f = open(step_log_path, 'w')
    step_log_f.write('step,loss,kept_ratio,lambda,temp\n')
    step_log_every = 10  # write every N steps to keep file small

    # ── Calibration: estimate threshold_tau from initial rho distribution ────
    if mode == 'threshold_rho' and target_keep_pct is not None and ref_loss_table is not None:
        print(f'[Calib] target_keep_pct={target_keep_pct} '
              f'collecting rho from first {calib_batches} batches (no grad)...')
        t0 = time.time()
        rho_samples = []
        model.eval()
        calib_loader = loader   # reuse same loader, just take first N batches
        with torch.no_grad():
            for bi, (indices, x, ch_mask) in enumerate(calib_loader):
                if bi >= calib_batches:
                    break
                indices = indices.to(device); x = x.to(device); ch_mask = ch_mask.to(device)
                B, C_max, T = x.shape
                n_patches_b = T // PATCH_LEN
                real_idx_b = ch_mask.nonzero(as_tuple=False)
                if real_idx_b.shape[0] == 0:
                    continue
                b_idx_b, c_idx_b = real_idx_b[:, 0], real_idx_b[:, 1]
                x_real = x[b_idx_b, c_idx_b]
                x_in = x_real.reshape(-1, n_patches_b, PATCH_LEN)
                pred = model(x_in)
                mse_pos_b = ((pred[:, :-1] - x_in[:, 1:]) ** 2).mean(dim=-1)
                C_eff = min(C_max, ref_loss_table.shape[1])
                in_eff = c_idx_b < C_eff
                if in_eff.any():
                    mse_eff = mse_pos_b[in_eff]
                    b_eff = b_idx_b[in_eff]
                    c_eff = c_idx_b[in_eff]
                    if ref_loss_table.is_cuda:
                        ref_pos = ref_loss_table[indices[b_eff], c_eff]
                    else:
                        ref_pos = ref_loss_table[indices[b_eff].cpu(), c_eff.cpu()].to(device)
                    rho_b = (mse_eff - ref_pos).detach()
                    rho_samples.append(rho_b.flatten().cpu())
        model.train()
        all_rho = torch.cat(rho_samples)
        threshold_tau = torch.quantile(all_rho, 1.0 - target_keep_pct).item()
        print(f'[Calib] threshold_tau={threshold_tau:.4f} (n_rho={len(all_rho)}, '
              f'mean={all_rho.mean():.4f}, std={all_rho.std():.4f}) '
              f'in {time.time()-t0:.1f}s')

    for epoch in range(epochs):
        model.train()
        # GPU-side accumulation — single .item() at epoch end avoids per-step CUDA sync
        epoch_loss = torch.zeros((), device=device)
        epoch_kept = torch.zeros((), device=device)
        n_batches  = 0
        step_in_epoch = 0

        for indices, x, ch_mask in tqdm(loader, desc=f'epoch {epoch+1}/{epochs}'):
            indices = indices.to(device)
            x       = x.to(device)
            ch_mask = ch_mask.to(device)
            B, C_max, T = x.shape
            n_patches = T // PATCH_LEN
            n_pos     = n_patches - 1   # number of next-patch prediction positions

            # Channel-independent forward over real channels only (skip padding).
            # real_idx[i] = (b, c) with ch_mask[b, c] = True, so forward batch
            # size is N_real = sum(ch_mask), not B*C_max — saves wasted compute
            # when batches mix series with very different channel counts.
            real_idx = ch_mask.nonzero(as_tuple=False)                  # (N_real, 2)
            if real_idx.shape[0] == 0:
                continue
            b_idx, c_idx = real_idx[:, 0], real_idx[:, 1]
            x_real = x[b_idx, c_idx]                                    # (N_real, T)
            x_in   = x_real.reshape(-1, n_patches, PATCH_LEN)           # (N_real, P, patch_len)
            pred   = model(x_in)                                        # (N_real, P, patch_len)

            # Next-patch MSE per (real-sequence, position)
            mse_pos = ((pred[:, :-1] - x_in[:, 1:]) ** 2).mean(dim=-1)  # (N_real, P-1)

            if mode == 'baseline':
                loss = mse_pos.mean()
                kept_ratio = torch.ones((), device=device)

            elif mode == 'random_mask':
                # Random masking baseline: same drop_pct as RHO but uniformly
                # random ranking instead of ρ = current - ref.
                rho        = torch.rand_like(mse_pos)
                valid      = torch.ones_like(rho, dtype=torch.bool)
                token_mask = compute_rho_mask(rho, valid, drop_pct)
                weighted   = mse_pos * token_mask.float()
                denom      = token_mask.float().sum().clamp(min=1)
                loss       = weighted.sum() / denom
                kept_ratio = token_mask.float().mean()
            elif mode == 'top_loss_drop':
                # Drop the HARDEST tokens (highest current loss). Ranking by
                # raw loss; compute_rho_mask drops the lowest score, so use
                # NEGATIVE loss as score => lowest = highest loss = dropped.
                rho        = -mse_pos.detach()
                valid      = torch.ones_like(rho, dtype=torch.bool)
                token_mask = compute_rho_mask(rho, valid, drop_pct)
                weighted   = mse_pos * token_mask.float()
                denom      = token_mask.float().sum().clamp(min=1)
                loss       = weighted.sum() / denom
                kept_ratio = token_mask.float().mean()
            elif mode == 'bottom_loss_drop':
                # Drop the EASIEST tokens (lowest current loss). Ranking by
                # raw loss directly => lowest dropped.
                rho        = mse_pos.detach()
                valid      = torch.ones_like(rho, dtype=torch.bool)
                token_mask = compute_rho_mask(rho, valid, drop_pct)
                weighted   = mse_pos * token_mask.float()
                denom      = token_mask.float().sum().clamp(min=1)
                loss       = weighted.sum() / denom
                kept_ratio = token_mask.float().mean()
            elif mode == 'rho_cm':
                # Only cells with c < C_eff have a ref entry. Filter real_idx to those.
                C_eff   = min(C_max, ref_loss_table.shape[1])
                in_eff  = c_idx < C_eff                                  # (N_real,) bool
                if in_eff.any():
                    mse_eff   = mse_pos[in_eff]                          # (N_eff, P-1)
                    b_eff     = b_idx[in_eff]
                    c_eff     = c_idx[in_eff]
                    # ref_loss_table is on GPU (moved once before training);
                    # if not, fall back to per-batch transfer.
                    if ref_loss_table.is_cuda:
                        ref_pos = ref_loss_table[indices[b_eff], c_eff]   # (N_eff, P-1)
                    else:
                        ref_pos = ref_loss_table[indices[b_eff].cpu(), c_eff.cpu()].to(device)
                    rho     = mse_eff.detach() - ref_pos                  # (N_eff, P-1)
                    valid   = torch.ones_like(rho, dtype=torch.bool)
                    token_mask = compute_rho_mask(rho, valid, drop_pct)  # (N_eff, P-1)
                    weighted   = mse_eff * token_mask.float()
                    denom      = token_mask.float().sum().clamp(min=1)
                    loss       = weighted.sum() / denom
                    kept_ratio = token_mask.float().mean()
                else:
                    # All channels exceed C_eff → fall back to baseline loss
                    loss = mse_pos.mean()
                    kept_ratio = torch.ones((), device=device)

            elif mode == 'threshold_rho':
                # Hard mask with FIXED threshold tau instead of fixed drop_pct.
                # As model improves -> current_loss decreases -> rho shrinks ->
                # fewer tokens pass tau => selected ratio drops over time
                # (an emergent curriculum without scheduling).
                C_eff   = min(C_max, ref_loss_table.shape[1])
                in_eff  = c_idx < C_eff
                if in_eff.any():
                    mse_eff = mse_pos[in_eff]
                    b_eff   = b_idx[in_eff]
                    c_eff   = c_idx[in_eff]
                    if ref_loss_table.is_cuda:
                        ref_pos = ref_loss_table[indices[b_eff], c_eff]
                    else:
                        ref_pos = ref_loss_table[indices[b_eff].cpu(), c_eff.cpu()].to(device)
                    rho_val   = mse_eff.detach() - ref_pos
                    mask_bool = rho_val > threshold_tau
                    if mask_bool.any():
                        weighted = mse_eff * mask_bool.float()
                        denom    = mask_bool.float().sum().clamp(min=1)
                        loss     = weighted.sum() / denom
                        kept_ratio = mask_bool.float().mean()
                    else:
                        # No token above threshold -> fall back to mean to avoid 0 grad
                        loss = mse_eff.mean()
                        kept_ratio = torch.zeros((), device=device)
                else:
                    loss = mse_pos.mean()
                    kept_ratio = torch.ones((), device=device)

            elif mode == 'soft_rho_fixed':
                # Soft reweighting with FIXED lambda=1 and temp=1 (no learning).
                # Compares soft mechanism vs hard top-K, isolating the soft
                # weighting effect from learnable hyperparams.
                C_eff   = min(C_max, ref_loss_table.shape[1])
                in_eff  = c_idx < C_eff
                if in_eff.any():
                    mse_eff = mse_pos[in_eff]
                    b_eff   = b_idx[in_eff]
                    c_eff   = c_idx[in_eff]
                    if ref_loss_table.is_cuda:
                        ref_pos = ref_loss_table[indices[b_eff], c_eff]
                    else:
                        ref_pos = ref_loss_table[indices[b_eff].cpu(), c_eff.cpu()].to(device)
                    rho_val = mse_eff.detach() - ref_pos     # lambda=1
                    w       = torch.sigmoid(rho_val)         # temp=1
                    weighted = mse_eff * w
                    denom    = w.sum().clamp(min=1e-6)
                    loss     = weighted.sum() / denom
                    kept_ratio = w.mean()
                else:
                    loss = mse_pos.mean()
                    kept_ratio = torch.ones((), device=device)

            elif mode == 'soft_rho_learnable':
                # Soft reweighting with learnable lambda + temperature.
                # No hard top-K mask; all tokens contribute, weighted by
                #   w = sigmoid((current - lambda * ref) / temp)
                # lambda and temp are learned alongside model parameters.
                C_eff   = min(C_max, ref_loss_table.shape[1])
                in_eff  = c_idx < C_eff
                if in_eff.any():
                    mse_eff = mse_pos[in_eff]                                 # (N_eff, P-1)
                    b_eff   = b_idx[in_eff]
                    c_eff   = c_idx[in_eff]
                    if ref_loss_table.is_cuda:
                        ref_pos = ref_loss_table[indices[b_eff], c_eff]
                    else:
                        ref_pos = ref_loss_table[indices[b_eff].cpu(), c_eff.cpu()].to(device)
                    # ρ uses detached current so lambda only learns through w
                    rho_val = mse_eff.detach() - lambda_ref * ref_pos
                    temp    = torch.exp(log_temp).clamp(min=1e-3, max=10.0)
                    w       = torch.sigmoid(rho_val / temp)
                    weighted = mse_eff * w
                    denom    = w.sum().clamp(min=1e-6)
                    loss     = weighted.sum() / denom
                    kept_ratio = w.mean()
                else:
                    loss = mse_pos.mean()
                    kept_ratio = torch.ones((), device=device)

            else:
                raise ValueError(f'Unknown mode: {mode}')

            optimizer.zero_grad(set_to_none=True)
            if meta_optimizer is not None:
                meta_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            if meta_optimizer is not None:
                meta_optimizer.step()
                if lambda_clip_max is not None:
                    with torch.no_grad():
                        lambda_ref.clamp_(min=0.0, max=lambda_clip_max)
            scheduler.step()

            epoch_loss += loss.detach()
            epoch_kept += kept_ratio.detach()
            n_batches  += 1
            step_in_epoch += 1
            if step_in_epoch % step_log_every == 0:
                global_step = epoch * len(loader) + step_in_epoch
                lam_val = lambda_ref.item() if lambda_ref is not None else float('nan')
                tmp_val = (torch.exp(log_temp).item()
                           if log_temp is not None else float('nan'))
                step_log_f.write(f'{global_step},{loss.item():.6f},{kept_ratio.item():.4f},{lam_val:.4f},{tmp_val:.4f}\n')
                step_log_f.flush()
            # Step-level checkpoint for Rho-1 style learning curve
            if save_every_n_steps is not None and (epoch * len(loader) + step_in_epoch) % save_every_n_steps == 0:
                gs = epoch * len(loader) + step_in_epoch
                sckdir = out_dir / 'step_ckpts'
                sckdir.mkdir(parents=True, exist_ok=True)
                torch.save({'epoch': epoch+1, 'global_step': gs, 'state_dict': model.state_dict(), 'loss': float(loss.item())},
                           sckdir / f'step{gs:07d}.pt')

        # scheduler is per-step now (LambdaLR); no per-epoch step here
        avg_loss = float((epoch_loss / max(n_batches, 1)).item())
        avg_kept = float((epoch_kept / max(n_batches, 1)).item())
        metrics['train_loss'].append(avg_loss)
        metrics['kept_ratio'].append(avg_kept)
        print(f'epoch {epoch+1}: loss={avg_loss:.4f}  kept={avg_kept*100:.1f}%')

        best_loss = save_epoch(out_dir, epoch + 1, model.state_dict(),
                               avg_loss, best_loss)

    step_log_f.close()
    dump_metrics(out_dir, metrics)
    print(f'\nDone. Best loss={best_loss:.4f}  ckpt: {out_dir}/best.pt')
    return metrics




# ─────────────────────────────────────────────────────────────────────────────
# Zero-shot forecasting: sliding-window autoregressive decoding
# ─────────────────────────────────────────────────────────────────────────────

def timer_zero_shot_predict(
    model: TimerModel,
    past_target: torch.Tensor,   # (B, context_len, C)
    prediction_length: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Autoregressively generate prediction_length future time steps.

    To stay inside the model's training distribution we (1) keep the input
    sliding window at exactly MAX_PATCHES tokens — the length the model was
    trained on — by dropping the oldest patch each step, and (2) freeze the
    RevIN statistics to those of the original context so the autoregressive
    feedback loop doesn't re-normalize generated patches with shifting stats.
    """
    model.eval()
    B, ctx_len, C = past_target.shape
    patch_len = model.patch_len

    # Pad context to multiple of patch_len
    pad = (-ctx_len) % patch_len
    if pad > 0:
        past_target = F.pad(past_target, (0, 0, pad, 0))
    ctx_patches = past_target.reshape(B, -1, patch_len, C)         # (B, n_ctx, patch_len, C)
    n_ctx = ctx_patches.shape[1]

    # Use exactly the trained window length: take the last MAX_PATCHES tokens
    # of context as the seed sequence. If context shorter, left-pad with zeros.
    train_len = MAX_PATCHES
    if n_ctx >= train_len:
        seed = ctx_patches[:, -train_len:, :, :]                   # (B, train_len, patch_len, C)
    else:
        zeros = torch.zeros(B, train_len - n_ctx, patch_len, C, device=device)
        seed  = torch.cat([zeros, ctx_patches], dim=1)             # left-pad

    n_pred_patches = math.ceil(prediction_length / patch_len)

    # Toggle off the model's internal RevIN; we apply stable per-channel norm here
    saved_use_norm = model.use_norm
    model.use_norm = False

    generated = []
    with torch.no_grad():
        for c in range(C):
            seq = seed[:, :, :, c].contiguous()                    # (B, train_len, patch_len)

            # Frozen RevIN: stats from the seed only (not updated as we generate)
            seed_flat = seq.reshape(B, -1)
            mu  = seed_flat.mean(dim=-1, keepdim=True).unsqueeze(-1)   # (B, 1, 1)
            sig = seed_flat.std(dim=-1, keepdim=True).unsqueeze(-1) + 1e-5
            seq = (seq - mu) / sig                                  # normalize seed once

            preds = []
            for _ in range(n_pred_patches):
                out = model(seq)                                    # (B, train_len, patch_len) normalized
                next_patch = out[:, -1:, :]                         # (B, 1, patch_len)
                preds.append(next_patch)
                # Sliding window: drop oldest, append newest — keep length = train_len
                seq = torch.cat([seq[:, 1:, :], next_patch], dim=1)

            pred_patches = torch.cat(preds, dim=1)                  # (B, n_pred, patch_len) normalized
            # De-normalize back to original scale
            pred_patches = pred_patches * sig + mu
            pred_ts = pred_patches.reshape(B, -1)[:, :prediction_length]
            generated.append(pred_ts)

    model.use_norm = saved_use_norm
    return torch.stack(generated, dim=-1)  # (B, prediction_length, C)


def eval_zero_shot(
    ckpt_path, dataset_name, horizon, seed, device,
    batch_size=32, context_length=SEQ_LEN,
):
    torch.manual_seed(seed); np.random.seed(seed)
    print(f'\n[Zero-shot] {dataset_name} H={horizon}  ckpt={ckpt_path}')

    _, _, test_ds = prepare_forecast_datasets(
        dataset_name, DATA_DIR, context_length, horizon, stride=1
    )

    model = TimerModel().to(device)
    if ckpt_path is not None:
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        model.load_state_dict(ckpt['state_dict'], strict=True)
        print(f'  loaded: {Path(ckpt_path).name}')
    model.eval()

    mse_list, mae_list = [], []
    with torch.no_grad():
        for batch in DataLoader(test_ds, batch_size=batch_size, num_workers=0):
            x_b, y_b, *_ = batch
            x_b = x_b.to(device)   # (B, C, ctx_len)
            y_b = y_b.to(device)   # (B, C, horizon)

            past = x_b.permute(0, 2, 1).float()  # (B, ctx_len, C)
            pred = timer_zero_shot_predict(model, past, horizon, device)
            pred_bc = pred.permute(0, 2, 1)       # (B, C, horizon)

            mse_list.append(((pred_bc - y_b) ** 2).mean().item())
            mae_list.append((pred_bc - y_b).abs().mean().item())

    mse = float(np.mean(mse_list))
    mae = float(np.mean(mae_list))
    print(f'  → MSE={mse:.4f}  MAE={mae:.4f}')
    return {'dataset': dataset_name, 'horizon': horizon, 'test_mse': mse, 'test_mae': mae}


def _timer_zero_shot_eval_fn(ckpt_path, ds, h, *, seed, device, batch_size):
    return eval_zero_shot(ckpt_path, ds, h, seed, device, batch_size=batch_size)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['baseline', 'rho_cm', 'random_mask',
                                           'top_loss_drop', 'bottom_loss_drop',
                                           'soft_rho_learnable', 'soft_rho_fixed',
                                           'threshold_rho',
                                           'eval_zero_shot', 'eval_zero_shot_sweep'],
                        default='baseline')
    parser.add_argument('--epochs',      type=int,   default=10)      # OpenLTM train_epochs=10
    parser.add_argument('--batch-size',  type=int,   default=16384)   # OpenLTM batch_size=16384 (8 GPUs DP)
    parser.add_argument('--lr',          type=float, default=5e-5)    # OpenLTM learning_rate=5e-5
    parser.add_argument('--drop-pct',    type=float, default=10.0)
    parser.add_argument('--ref-epochs',  type=int,   default=10)
    parser.add_argument('--max-series',  type=int,   default=None,
                        help='Cap on number of source series (None = all)')
    parser.add_argument('--max-windows', type=int,   default=None,
                        help='Cap on number of training windows (None = all)')
    parser.add_argument('--out-dir',     type=str,   default=None)
    parser.add_argument('--seed',        type=int,   default=42)
    # Eval
    parser.add_argument('--ckpt',            type=str, default=None)
    parser.add_argument('--eval-dataset',    type=str, default='ETTh1')
    parser.add_argument('--eval-horizon',    type=int, default=96)
    parser.add_argument('--eval-datasets',   type=str, default='ETTh1,ETTh2,ETTm1,ETTm2,weather,exchange')
    parser.add_argument('--eval-horizons',   type=str, default='96,192,336,720')
    parser.add_argument('--baseline-ckpt',   type=str, default=None)
    parser.add_argument('--rho-cm-ckpt',     type=str, default=None)
    parser.add_argument('--context-length',  type=int, default=SEQ_LEN)
    # soft_rho_learnable specific
    parser.add_argument('--lambda-clip-max', type=float, default=None,
                        help='If set, clamp lambda_ref to [0, max] each step')
    parser.add_argument('--meta-lr',         type=float, default=1e-2,
                        help='lr for learnable lambda + temp (soft mode)')
    parser.add_argument('--init-lambda',     type=float, default=1.0,
                        help='Initial value of lambda_ref (soft mode)')
    parser.add_argument('--save-every-n-steps', type=int, default=None)
    parser.add_argument('--threshold-tau',   type=float, default=0.0,
                        help='Fixed tau for threshold_rho mode (rho > tau kept)')
    parser.add_argument('--univariate', action='store_true',
                        help='OpenLTM-style SOM: unroll multivariate to univariate windows')
    parser.add_argument('--target-keep-pct', type=float, default=None,
                        help='If set, calibrate threshold_tau via initial-batch percentile')
    parser.add_argument('--calib-batches', type=int, default=200)

    args   = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    if args.out_dir is None:
        args.out_dir = f'results_timer/{args.mode}'

    if args.mode in ('baseline', 'rho_cm', 'random_mask',
                     'top_loss_drop', 'bottom_loss_drop',
                     'soft_rho_learnable', 'soft_rho_fixed',
                     'threshold_rho'):
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
            max_windows = args.max_windows,
            device     = device,
            lambda_clip_max = args.lambda_clip_max,
            meta_lr         = args.meta_lr,
            init_lambda     = args.init_lambda,
            threshold_tau   = args.threshold_tau,
            univariate      = args.univariate,
            target_keep_pct = args.target_keep_pct,
            calib_batches   = args.calib_batches,
            save_every_n_steps = args.save_every_n_steps,
        )
    elif args.mode == 'eval_zero_shot':
        eval_zero_shot(
            ckpt_path      = Path(args.ckpt) if args.ckpt else None,
            dataset_name   = args.eval_dataset,
            horizon        = args.eval_horizon,
            seed           = args.seed,
            device         = device,
            batch_size     = args.batch_size,
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
            _timer_zero_shot_eval_fn,
            ckpt_paths   = ckpt_paths,
            datasets     = args.eval_datasets.split(','),
            horizons     = [int(h) for h in args.eval_horizons.split(',')],
            title        = 'ZERO-SHOT RESULTS',
            out_filename = 'zero_shot_results.json',
            out_dir      = args.out_dir,
            seed         = args.seed,
            device       = device,
            batch_size   = args.batch_size,
        )


if __name__ == '__main__':
    main()
