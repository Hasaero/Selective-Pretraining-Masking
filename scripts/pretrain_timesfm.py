"""TimesFM 1.0 (200M) pretraining (baseline + SPM variants).

Architecture
------------
Random-init TimesFM 1.0 v1 (Das et al. ICML 2024, vendored in
`rho_lib/_timesfm_v1_src/`). patch_len=32, horizon_len=128,
num_layers=20, hidden_size=1280 (~204M params).
Output: (B, N_input_patches=16, horizon_len=128, num_outputs=10).
  index 0 = mean (point prediction), indices 1..9 = 0.1..0.9 quantiles.

Per-patch random prefix masking (paper Section 3.2): for each sample, mask
the first r ∼ Uniform{0..patch_len-1} timesteps of the first input patch.

Modes
-----
- baseline          : mean MSE + quantile pinball over all (B, N, 128) targets.
- threshold_rho     : SPM with calibrated τ (recommended).
                      ρ = current_MSE − ref_MSE per (sample, position, horizon-step),
                      static (N, 16, 128) DLinear ref table built once.
- random_mask       : random per-token mask matching `--drop-pct` (ablation).
- top_loss_drop     : drop highest-loss tokens (ablation).
- bottom_loss_drop  : drop lowest-loss tokens (ablation).

Reference model: DLinear MSE forecaster (see rho_lib/ref/dlinear_mse_forecast.py).

Eval
----
- eval_zero_shot, eval_zero_shot_sweep : AR rollout per (dataset, horizon).

Best config
-----------
baseline       : --epochs 1 --batch-size 128 --lr 5e-6
SPM (calib)    : same + --target-keep-pct 0.4 --calib-batches 200 --ref-epochs 2
"""

import argparse
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# TimesFM v1 source (drop in cleared __init__ so we don't pull extra deps)
_TIMESFM_SRC = _PROJECT_ROOT / 'rho_lib' / '_timesfm_v1_src'
if str(_TIMESFM_SRC) not in sys.path:
    sys.path.insert(0, str(_TIMESFM_SRC))
# Ensure timesfm_v1/__init__.py is empty to avoid the bundled package's heavy imports
_init = _TIMESFM_SRC / 'timesfm_v1' / '__init__.py'
if _init.exists() and _init.read_text().strip():
    _init.write_text('')
from timesfm_v1.pytorch_patched_decoder import PatchedTimeSeriesDecoder, TimesFMConfig

from rho_lib.data.utsd import (
    load_or_build_cached_univariate, make_collate_fn_univariate,
)
from rho_lib.data.forecast import prepare_forecast_datasets
from rho_lib.ref.dlinear_mse_forecast import build_ref_model_only
from rho_lib.train.schedule import warmup_cosine_with_min_lr_lambda
from rho_lib.train.checkpointing import save_epoch, dump_metrics

# ─────────────────────────────────────────────────────────────────────────────
# TimesFM 1.0 200M architecture constants
# ─────────────────────────────────────────────────────────────────────────────
PATCH_LEN       = 32
HORIZON_LEN     = 128
N_INPUT_PATCHES = 16             # context_length = 512
CONTEXT_LEN     = N_INPUT_PATCHES * PATCH_LEN   # 512
TRAIN_FULL_LEN  = CONTEXT_LEN + HORIZON_LEN     # 640
MEAN_IDX        = 0              # index 0 in output last dim is the mean

UTSD_PATH = _PROJECT_ROOT / 'data' / 'utsd_repo' / 'UTSD-12G'
DATA_DIR  = _PROJECT_ROOT / 'data'


def build_timesfm_model() -> PatchedTimeSeriesDecoder:
    """Random-init TimesFM 1.0 200M (official config: 20 layers, hidden=1280)."""
    return PatchedTimeSeriesDecoder(TimesFMConfig())


def apply_input_masks(ctx: torch.Tensor, rng: torch.Generator) -> torch.Tensor:
    """TimesFM 1.0 paper random masking.
    Per sample, mask the first r ~ Uniform{0, ..., patch_len-1} timesteps of
    the FIRST patch only (Das et al. ICML 2024, Section 3.2).
    Returns padding (B, T) where 1.0 = masked.
    """
    B, T = ctx.shape
    device = ctx.device
    # Per-sample random r in [0, patch_len)
    r = torch.randint(0, PATCH_LEN, (B,), generator=rng, device=device)
    arange_T = torch.arange(T, device=device).view(1, T)
    padding = (arange_T < r.unsqueeze(-1)).float()    # (B, T) — only first r positions
    return padding


def pretrain(
    mode: str,
    epochs: int,
    batch_size: int,
    lr: float,
    out_dir: Path,
    seed: int,
    max_series: int | None,
    device: torch.device,
    threshold_tau: float = 0.0,
    ref_epochs: int = 2,
    target_keep_pct: float | None = None,
    calib_batches: int = 200,
    save_every_n_steps: int | None = None,
    drop_pct: float = 70.0,
):
    torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)
    rng = torch.Generator(device=device); rng.manual_seed(seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Dataset (UTSD univariate, window = 640) ─────────────────────────────
    dataset = load_or_build_cached_univariate(
        UTSD_PATH, seq_len=TRAIN_FULL_LEN, stride=TRAIN_FULL_LEN,
        standardize_per_series=True,
        max_series=max_series,
    )
    print(f'[Data] {len(dataset)} windows of length {TRAIN_FULL_LEN}')
    collate_fn = make_collate_fn_univariate(TRAIN_FULL_LEN)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        num_workers=0, collate_fn=collate_fn,
        pin_memory=False, drop_last=True,
    )
    print(f'[Data] DataLoader: {len(loader)} batches')

    # ── Model ───────────────────────────────────────────────────────────────
    model = build_timesfm_model().to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f'[Model] TimesFM 1.0 200M random-init: {n_params:.1f}M params')

    # Paper: Adam (not AdamW), no weight decay, default betas
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    total_steps  = len(loader) * epochs
    warmup_steps = min(2000, max(1, int(total_steps * 0.05)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        warmup_cosine_with_min_lr_lambda(
            num_warmup_steps=warmup_steps, num_training_steps=total_steps,
            warmup_start_lr=1e-5, base_lr=lr, min_lr=1e-5,
        ),
    )

    # ── Static ref table (N, 16, 128) ───────────────────────────────────────
    ref_table = None
    if mode == 'threshold_rho':
        t0 = time.time()
        ref_model, _ = build_ref_model_only(
            dataset, seq_len=TRAIN_FULL_LEN, ref_epochs=ref_epochs,
            batch_size=256, device=device, collate_fn=collate_fn, seed=seed,
            out_dir=out_dir,
        )
        ref_model.eval()
        for p in ref_model.parameters():
            p.requires_grad_(False)
        print(f'[Ref] DLinear trained in {time.time()-t0:.1f}s; building static table...')

        N = len(dataset)
        ref_table = torch.zeros(N, N_INPUT_PATCHES, HORIZON_LEN, dtype=torch.float32)
        eval_loader = DataLoader(
            dataset, batch_size=128, shuffle=False, num_workers=0,
            collate_fn=collate_fn, pin_memory=False,
        )
        with torch.no_grad():
            for indices, x_full_bc, ch_mask in tqdm(eval_loader, desc='  ref table'):
                B, C, _ = x_full_bc.shape
                x_flat = x_full_bc.reshape(B*C, TRAIN_FULL_LEN).to(device)
                idx_flat = indices.repeat_interleave(C)
                vmask = ch_mask.reshape(B*C)
                if not vmask.any():
                    continue
                x_v = x_flat[vmask]
                Bv = x_v.shape[0]
                per_sample_refs = torch.zeros(Bv, N_INPUT_PATCHES, HORIZON_LEN, device=device)
                for i in range(N_INPUT_PATCHES):
                    L = PATCH_LEN * (i + 1)
                    if L + HORIZON_LEN > TRAIN_FULL_LEN:
                        break
                    ctx_i = x_v[:, :L]
                    target_i = x_v[:, L:L + HORIZON_LEN]
                    mu_i = ctx_i.mean(dim=1, keepdim=True)
                    sig_i = ctx_i.std(dim=1, keepdim=True).clamp(min=1e-2)
                    ctx_n = (ctx_i - mu_i) / sig_i
                    ref_pred_n = ref_model(ctx_n, max_pred_len=HORIZON_LEN)
                    ref_pred = ref_pred_n * sig_i + mu_i
                    per_sample_refs[:, i, :] = (ref_pred - target_i) ** 2
                ref_table[idx_flat[vmask].cpu()] = per_sample_refs.cpu()
        del ref_model
        torch.cuda.empty_cache()
        ref_bytes = ref_table.element_size() * ref_table.nelement()
        if torch.cuda.is_available() and ref_bytes < 2 * (1 << 30):
            ref_table = ref_table.to(device)
            print(f'[Ref] static table on GPU ({ref_bytes/1e6:.1f} MB)')
        print(f'[Ref] static (N={N}, 16, 128) table ready in {time.time()-t0:.1f}s total')

    QUANTILES = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], device=device)

    def model_forward(x_full, training_mask=True):
        """x_full (B, 640) -> (full_out (B, 16, 128, 10), targets (B, 16, 128), valid_pos (16,))."""
        B = x_full.shape[0]
        ctx = x_full[:, :CONTEXT_LEN]
        if training_mask:
            padding = apply_input_masks(ctx, rng)
        else:
            padding = torch.zeros_like(ctx)
        freq = torch.zeros(B, 1, dtype=torch.long, device=ctx.device)
        out = model(ctx, padding, freq)                          # (B, 16, 128, 10)
        # Per-position targets (next 128 after position i)
        targets = torch.zeros(B, N_INPUT_PATCHES, HORIZON_LEN, device=x_full.device)
        valid_pos = torch.zeros(N_INPUT_PATCHES, dtype=torch.bool, device=x_full.device)
        for i in range(N_INPUT_PATCHES):
            L = PATCH_LEN * (i + 1)
            if L + HORIZON_LEN <= TRAIN_FULL_LEN:
                targets[:, i, :] = x_full[:, L:L + HORIZON_LEN]
                valid_pos[i] = True
        return out, targets, valid_pos

    def quantile_pinball_loss(quantile_preds, target, valid_mask):
        """quantile_preds (B, N, H, 9), target (B, N, H), valid_mask (B, N, H) bool.
        Returns scalar mean pinball loss over valid positions.
        """
        diff = target.unsqueeze(-1) - quantile_preds              # (B, N, H, 9)
        loss_q = torch.maximum(QUANTILES * diff, (QUANTILES - 1) * diff)
        # Average over quantiles, then mask
        per_pos = loss_q.mean(dim=-1)                             # (B, N, H)
        denom = valid_mask.float().sum().clamp(min=1)
        return (per_pos * valid_mask.float()).sum() / denom

    # ── Calibration ─────────────────────────────────────────────────────────
    if mode == 'threshold_rho' and target_keep_pct is not None and ref_table is not None:
        print(f'[Calib] target_keep_pct={target_keep_pct} '
              f'collecting rho from first {calib_batches} batches...')
        t0 = time.time()
        rho_samples = []
        model.eval()
        with torch.no_grad():
            for bi, (indices, x_full_bc, ch_mask) in enumerate(loader):
                if bi >= calib_batches:
                    break
                B, C, _ = x_full_bc.shape
                x_full = x_full_bc.reshape(B*C, TRAIN_FULL_LEN).to(device)
                vmask = ch_mask.reshape(B*C).to(device)
                idx_flat = indices.repeat_interleave(C).to(device)
                if not vmask.any():
                    continue
                x_full = x_full[vmask]
                idx_flat = idx_flat[vmask]
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    out, targets, valid_pos = model_forward(x_full, training_mask=True)
                mean_pred = out[..., MEAN_IDX].float()
                cur_sq = (mean_pred - targets) ** 2
                if ref_table.is_cuda:
                    ref_sq = ref_table[idx_flat]
                else:
                    ref_sq = ref_table[idx_flat.cpu()].to(device)
                rho = cur_sq - ref_sq
                valid = valid_pos.view(1, -1, 1).expand_as(rho)
                rho_samples.append(rho[valid].flatten().detach().cpu())
        model.train()
        if rho_samples:
            all_rho = torch.cat(rho_samples)
            # torch.quantile chokes on >16M elements; subsample if needed.
            if all_rho.numel() > 16_000_000:
                idx = torch.randperm(all_rho.numel())[:16_000_000]
                _rho_sub = all_rho[idx]
            else:
                _rho_sub = all_rho
            threshold_tau = torch.quantile(_rho_sub, 1.0 - target_keep_pct).item()
            print(f'[Calib] threshold_tau={threshold_tau:.4f} '
                  f'(n_rho={len(all_rho)}, mean={all_rho.mean():.4f}, '
                  f'std={all_rho.std():.4f}) in {time.time()-t0:.1f}s')
        else:
            print('[Calib] WARNING: no rho samples')

    # ── Training ────────────────────────────────────────────────────────────
    metrics = {'train_loss': [], 'kept_ratio': []}
    best_loss = float('inf')
    step_log = open(out_dir / 'step_loss.csv', 'w')
    step_log.write('step,loss,kept_ratio\n')
    log_every = 10

    for epoch in range(epochs):
        model.train()
        epoch_loss = torch.zeros((), device=device)
        epoch_kept = torch.zeros((), device=device)
        n_batches = 0
        step_in_epoch = 0
        for indices, x_full_bc, ch_mask in tqdm(loader, desc=f'epoch {epoch+1}/{epochs}'):
            B, C, _ = x_full_bc.shape
            x_full = x_full_bc.reshape(B*C, TRAIN_FULL_LEN).to(device)
            vmask = ch_mask.reshape(B*C).to(device)
            idx_flat = indices.repeat_interleave(C).to(device)
            if not vmask.any():
                continue
            x_full = x_full[vmask]
            idx_flat = idx_flat[vmask]

            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                out, targets, valid_pos = model_forward(x_full, training_mask=True)
                mean_pred = out[..., MEAN_IDX]
                quantile_preds = out[..., 1:]                                  # (B, 16, 128, 9)
                cur_sq = (mean_pred - targets) ** 2
                valid = valid_pos.view(1, -1, 1).expand_as(cur_sq)

                if mode == 'baseline':
                    denom = valid.float().sum().clamp(min=1)
                    mse_loss = (cur_sq * valid.float()).sum() / denom
                    q_loss = quantile_pinball_loss(quantile_preds, targets, valid)
                    loss = mse_loss + q_loss
                    kept_ratio = torch.ones((), device=device)
                elif mode == 'threshold_rho':
                    with torch.no_grad():
                        if ref_table.is_cuda:
                            ref_sq = ref_table[idx_flat]
                        else:
                            ref_sq = ref_table[idx_flat.cpu()].to(device)
                        rho = cur_sq.detach().float() - ref_sq
                    kept_mask = valid & (rho > threshold_tau)
                    kept_count = kept_mask.float().sum().clamp(min=1)
                    mse_loss = (cur_sq * kept_mask.float()).sum() / kept_count
                    q_loss = quantile_pinball_loss(quantile_preds, targets, kept_mask)
                    loss = mse_loss + q_loss
                    kept_ratio = kept_count / valid.float().sum().clamp(min=1)
                else:
                    # random_mask / top_loss_drop / bottom_loss_drop ablations.
                    # All keep keep_ratio (= 1 - drop_pct/100) of the valid tokens.
                    keep_ratio_target = 1.0 - drop_pct / 100.0
                    with torch.no_grad():
                        flat_valid = valid.flatten()
                        n_valid = int(flat_valid.sum().item())
                        n_keep = max(1, int(round(keep_ratio_target * n_valid)))
                        if mode == 'random_mask':
                            score = torch.rand_like(cur_sq.detach().float()).flatten()
                        elif mode == 'top_loss_drop':
                            # Drop highest losses: keep low losses -> rank by ascending loss, keep first n_keep
                            score = -cur_sq.detach().float().flatten()  # higher score = lower loss
                        elif mode == 'bottom_loss_drop':
                            # Drop lowest losses: keep high losses -> rank by descending loss
                            score = cur_sq.detach().float().flatten()
                        else:
                            raise ValueError(f'Unknown mode {mode}')
                        # mask invalid positions out with -inf so they are never picked
                        score = score.masked_fill(~flat_valid, float('-inf'))
                        topk_idx = torch.topk(score, n_keep).indices
                        kept_mask_flat = torch.zeros_like(flat_valid)
                        kept_mask_flat[topk_idx] = True
                        kept_mask = kept_mask_flat.reshape_as(valid)
                    kept_count = kept_mask.float().sum().clamp(min=1)
                    mse_loss = (cur_sq * kept_mask.float()).sum() / kept_count
                    q_loss = quantile_pinball_loss(quantile_preds, targets, kept_mask)
                    loss = mse_loss + q_loss
                    kept_ratio = kept_count / valid.float().sum().clamp(min=1)

            optimizer.zero_grad(set_to_none=True)
            if torch.isnan(loss):
                # Only skip NaN (not Inf). Inf grad is fine after clipping.
                scheduler.step()
                continue
            loss.backward()
            # Aggressive clipping for stability with 200M random init
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if torch.isnan(grad_norm):
                scheduler.step()
                continue
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.detach()
            epoch_kept += kept_ratio.detach()
            n_batches += 1
            step_in_epoch += 1
            if step_in_epoch % log_every == 0:
                gs = epoch * len(loader) + step_in_epoch
                step_log.write(f'{gs},{loss.item():.6f},{kept_ratio.item():.4f}\n')
            # Step-level checkpoint for Rho-1 style learning curve
            if save_every_n_steps is not None:
                _gs = epoch * len(loader) + step_in_epoch
                if _gs % save_every_n_steps == 0:
                    sckdir = out_dir / 'step_ckpts'
                    sckdir.mkdir(parents=True, exist_ok=True)
                    torch.save({'epoch': epoch+1, 'global_step': _gs, 'state_dict': model.state_dict(), 'loss': float(loss.item())},
                               sckdir / f'step{_gs:07d}.pt')

        if n_batches:
            avg_loss = (epoch_loss / n_batches).item()
            avg_kept = (epoch_kept / n_batches).item()
        else:
            avg_loss, avg_kept = 0.0, 0.0
        print(f'epoch {epoch+1}: loss={avg_loss:.4f}  kept={avg_kept*100:.1f}%')
        metrics['train_loss'].append(avg_loss)
        metrics['kept_ratio'].append(avg_kept)
        best_loss = save_epoch(
            out_dir, epoch=epoch+1,
            state_dict=model.state_dict(), loss=avg_loss, best_loss=best_loss,
        )
    step_log.close()
    dump_metrics(out_dir, metrics)
    print(f'Pretraining done. Best loss={best_loss:.4f}')


# ─────────────────────────────────────────────────────────────────────────────
# Eval: zero-shot AR forecast
# ─────────────────────────────────────────────────────────────────────────────
def eval_zero_shot(ckpt_path, dataset_name, horizon, seed, device, batch_size=32):
    torch.manual_seed(seed); np.random.seed(seed)
    print(f'\n[Zero-shot] {dataset_name} H={horizon}  ckpt={ckpt_path}')

    _, _, test_ds = prepare_forecast_datasets(
        dataset_name, DATA_DIR, CONTEXT_LEN, horizon, stride=1,
    )

    model = build_timesfm_model().to(device)
    if ckpt_path is not None:
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        model.load_state_dict(ckpt['state_dict'], strict=True)
        print(f'  loaded {Path(ckpt_path).name}')
    model.eval()

    mse_list, mae_list = [], []
    with torch.no_grad():
        for batch in DataLoader(test_ds, batch_size=batch_size, num_workers=0):
            x_b, y_b, *_ = batch
            x_b = x_b.to(device); y_b = y_b.to(device)
            B, C, _ = x_b.shape
            ctx_uv = x_b.permute(0, 2, 1).reshape(B*C, CONTEXT_LEN)
            # AR rollout: predict HORIZON_LEN at a time, append to context, repeat.
            full_pred = torch.zeros(B*C, 0, device=device)
            cur_ctx = ctx_uv
            steps_needed = horizon
            while full_pred.shape[1] < horizon:
                padding = torch.zeros_like(cur_ctx)
                freq = torch.zeros(cur_ctx.shape[0], 1, dtype=torch.long, device=device)
                out = model(cur_ctx, padding, freq)                # (B, N, 128, 10)
                # last input position gives the actual horizon prediction
                next_chunk = out[:, -1, :, MEAN_IDX]                # (B, 128)
                full_pred = torch.cat([full_pred, next_chunk], dim=1)
                # Slide context by appending next_chunk, drop oldest
                cur_ctx = torch.cat([cur_ctx, next_chunk], dim=1)[:, -CONTEXT_LEN:]
            pred = full_pred[:, :horizon]
            pred_bc = pred.reshape(B, C, horizon)
            mse_list.append(((pred_bc - y_b) ** 2).mean().item())
            mae_list.append((pred_bc - y_b).abs().mean().item())

    mse = float(np.mean(mse_list)); mae = float(np.mean(mae_list))
    print(f'  -> MSE={mse:.4f} MAE={mae:.4f}')
    return {'dataset': dataset_name, 'horizon': horizon,
            'test_mse': mse, 'test_mae': mae}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['baseline', 'threshold_rho',
                                           'random_mask', 'top_loss_drop', 'bottom_loss_drop',
                                           'eval_zero_shot', 'eval_zero_shot_sweep'],
                        required=True)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--ref-epochs', type=int, default=2)
    parser.add_argument('--save-every-n-steps', type=int, default=None)
    parser.add_argument('--threshold-tau', type=float, default=0.0)
    parser.add_argument('--drop-pct', type=float, default=70.0,
                        help='drop_pct for random_mask/top_loss_drop/bottom_loss_drop (%%)')
    parser.add_argument('--target-keep-pct', type=float, default=None)
    parser.add_argument('--calib-batches', type=int, default=200)
    parser.add_argument('--max-series', type=int, default=None)
    parser.add_argument('--out-dir', type=str, default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--ckpt', type=str, default=None)
    parser.add_argument('--baseline-ckpt', type=str, default=None)
    parser.add_argument('--rho-cm-ckpt', type=str, default=None)
    parser.add_argument('--eval-datasets', type=str,
                        default='ETTh1,ETTh2,ETTm1,ETTm2,weather,exchange')
    parser.add_argument('--eval-horizons', type=str, default='96,192,336,720')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.mode in ('baseline', 'threshold_rho', 'random_mask', 'top_loss_drop', 'bottom_loss_drop'):
        pretrain(
            mode=args.mode, epochs=args.epochs, batch_size=args.batch_size,
            lr=args.lr, out_dir=Path(args.out_dir), seed=args.seed,
            max_series=args.max_series, device=device,
            threshold_tau=args.threshold_tau, ref_epochs=args.ref_epochs,
            target_keep_pct=args.target_keep_pct,
            save_every_n_steps=args.save_every_n_steps,
            drop_pct=args.drop_pct,
            calib_batches=args.calib_batches,
        )
    elif args.mode == 'eval_zero_shot':
        eval_zero_shot(Path(args.ckpt), args.eval_datasets.split(',')[0],
                       int(args.eval_horizons.split(',')[0]), args.seed,
                       device, args.batch_size)
    elif args.mode == 'eval_zero_shot_sweep':
        from rho_lib.eval.reporting import print_results_table, print_delta_tables, dump_results_json
        ckpt_paths = {}
        if args.baseline_ckpt: ckpt_paths['baseline'] = Path(args.baseline_ckpt)
        if args.rho_cm_ckpt:   ckpt_paths['rho_cm']   = Path(args.rho_cm_ckpt)
        if not ckpt_paths and args.ckpt: ckpt_paths['model'] = Path(args.ckpt)
        datasets = args.eval_datasets.split(',')
        horizons = [int(h) for h in args.eval_horizons.split(',')]
        results = {label: [] for label in ckpt_paths}
        for label, ckpt in ckpt_paths.items():
            print(f'\n{"="*60}\nEvaluating: {label}  ckpt={ckpt}\n{"="*60}')
            for ds in datasets:
                for h in horizons:
                    try:
                        r = eval_zero_shot(ckpt, ds, h, args.seed,
                                           device, batch_size=args.batch_size)
                        results[label].append(r)
                    except Exception as e:
                        print(f'  SKIP {ds} H={h}: {e}')
        print_results_table(results, datasets, horizons, title='TIMESFM v1 ZERO-SHOT')
        print_delta_tables(results, datasets, horizons, baseline_label='baseline')
        dump_results_json(results, args.out_dir, 'zero_shot_results.json')


if __name__ == '__main__':
    main()
