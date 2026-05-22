"""
MOMENT Pretraining with patch-level RHO-CM masking.

Baseline      : MOMENT-small + random 30% patch mask (MAE objective)
patch_rho_cm  : MOMENT-small + per-(sample, channel, patch) RHO weighting
                rho[i, c, p] = current_recon_loss[i, c, p] - ref_loss[i, c, p]
                Drop bottom drop_pct% of (i, c, p) triples per batch.
                ref = DLinear trained briefly (ref_epochs) on the same corpus.

Usage
-----
# Baseline (random mask only)
python scripts/pretrain_moment.py --mode baseline --epochs 2 --batch-size 64

# Patch-level RHO-CM
python scripts/pretrain_moment.py --mode patch_rho_cm --epochs 2 --batch-size 64 \
    --ref-epochs 3 --drop-pct 10

# Evaluate (linear probe on ETTh1 forecasting)
python scripts/pretrain_moment.py --mode eval \
    --ckpt results_pretrain/patch_rho_cm/best.pt \
    --eval-dataset ETTh1 --eval-horizon 96
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
from torch.utils.data import DataLoader
from tqdm import tqdm

# ── momentfm ──────────────────────────────────────────────────────────────────
from momentfm import MOMENTPipeline

# ── rho_lib (shared scaffolding) ──────────────────────────────────────────────
from rho_lib.data.utsd     import (UTSDPretrainDataset, make_collate_fn,
                                   load_utsd, compute_channel_cap)
from rho_lib.data.forecast import prepare_forecast_datasets
from rho_lib.ref.dlinear_recon_masked import (
    build_ref_model_masked,
    DLinearMaskedReconRef,
)
from rho_lib.rho.mask      import compute_rho_mask
from rho_lib.eval.sweep    import run_eval_sweep
from rho_lib.train.schedule     import warmup_cosine_with_min_lr_lambda
from rho_lib.train.checkpointing import save_epoch, dump_metrics

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
SEQ_LEN   = 512   # MOMENT fixed input length
PATCH_LEN = 8
N_PATCHES = SEQ_LEN // PATCH_LEN   # 64

UTSD_PATH = Path(__file__).resolve().parent.parent / "data" / 'utsd_repo' / 'UTSD-12G'
DATA_DIR  = Path(__file__).resolve().parent.parent / "data"

# ─────────────────────────────────────────────────────────────────────────────
# MOMENT reconstruction loss helpers
# ─────────────────────────────────────────────────────────────────────────────

def _moment_forward_chunk(
    model,
    x: torch.Tensor,   # (B, C, T)
    ch_chunk: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Runs MOMENT in channel chunks. Returns:
        recon   : (B, C, T)  reconstruction
        pmask   : (B, C, T)  float, 0=masked timepoint
    Both on same device as x. Grad flows if model is in train mode.

    Single-pass path when C <= ch_chunk avoids the (B, C, T) zero-alloc + scatter.
    """
    B, C, T = x.shape
    device  = x.device

    if C <= ch_chunk:
        x_enc = x.reshape(B * C, 1, T)
        outputs = model(x_enc=x_enc, input_mask=torch.ones(B * C, T, device=device))
        recon = outputs.reconstruction.squeeze(1).reshape(B, C, T)
        # MOMENT returns pretrain_mask as long/bool; cast to float so callers
        # can do `(1 - pmask).mean(dim)` etc. without dtype errors.
        pmask = outputs.pretrain_mask.float().reshape(B, C, T)
        return recon, pmask

    # Multi-chunk path (C > ch_chunk). Reuse one (B, C, T) buffer + one input_mask.
    recon_out = torch.empty(B, C, T, device=device)
    pmask_out = torch.empty(B, C, T, device=device)
    input_mask = torch.ones(B * ch_chunk, T, device=device)

    for c_start in range(0, C, ch_chunk):
        c_end = min(c_start + ch_chunk, C)
        Cc    = c_end - c_start
        x_enc = x[:, c_start:c_end, :].reshape(B * Cc, 1, T)
        im    = input_mask[:B * Cc]
        outputs = model(x_enc=x_enc, input_mask=im)
        recon_out[:, c_start:c_end, :] = outputs.reconstruction.squeeze(1).reshape(B, Cc, T)
        pmask_out[:, c_start:c_end, :] = outputs.pretrain_mask.float().reshape(B, Cc, T)

    return recon_out, pmask_out


def moment_recon_loss_per_channel(
    model,
    x: torch.Tensor,        # (B, C, SEQ_LEN)
    ch_mask: torch.Tensor,  # (B, C) bool
    ch_chunk: int = 4,
) -> torch.Tensor:
    """Per-channel reconstruction MSE on masked timepoints. Used by the
    baseline mode's loss aggregation. Returns (B, C) float32."""
    recon, pmask = _moment_forward_chunk(model, x, ch_chunk)
    sq_err  = (recon - x) ** 2
    masked  = (1 - pmask)
    denom   = masked.sum(dim=2).clamp(min=1.0)
    return (sq_err * masked).sum(dim=2) / denom


def moment_recon_loss_per_patch(
    model,
    x: torch.Tensor,        # (B, C, SEQ_LEN)
    ch_mask: torch.Tensor,  # (B, C) bool
    ch_chunk: int = 4,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-(sample, channel, patch) reconstruction MSE + patch mask flag.
    Returns (per_patch_loss (B, C, N_PATCHES), patch_mask (B, C, N_PATCHES) float)."""
    recon, pmask = _moment_forward_chunk(model, x, ch_chunk)
    B, C, T = x.shape
    sq_err = (recon - x) ** 2

    sq_patches = sq_err.reshape(B, C, N_PATCHES, PATCH_LEN).mean(3)
    pm_patches = (1 - pmask).reshape(B, C, N_PATCHES, PATCH_LEN).mean(3) > 0.5
    return sq_patches, pm_patches.float()


# ─────────────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────────────

def pretrain(
    mode: str,          # 'baseline' | 'rho_cm' | 'patch_rho_cm' | 'timepoint_rho_cm'
    epochs: int,
    batch_size: int,
    lr: float,
    drop_pct: float,
    ref_epochs: int,
    out_dir: Path,
    seed: int,
    max_series: int | None,
    device: torch.device,
):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load dataset ──────────────────────────────────────────────────────────
    print('[Data] Loading UTSD...')
    hf_ds = load_utsd(UTSD_PATH, max_series=max_series)
    print(f'[Data] {len(hf_ds):,} rows loaded')
    dataset = UTSDPretrainDataset(
        hf_ds, seq_len=SEQ_LEN, stride=SEQ_LEN,
        max_series=max_series, standardize_per_series=True,  # MOMENT default
    )

    # Cap channels at the 99th percentile to bound effective forward batch
    # size (B × C). Without this, heavy multivariate series (e.g. Traffic
    # with 862 channels) cause OOM on a single batch even when the nominal
    # batch_size is moderate. Random subset preserves coverage across epochs.
    channel_cap = compute_channel_cap(dataset, percentile=99.0, per='window')
    print(f'[Data] channel cap (p99): {channel_cap}')
    collate_fn = make_collate_fn(SEQ_LEN, channel_cap=channel_cap)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn,
        pin_memory=False,
        drop_last=True,
    )
    print(f'[Data] DataLoader ready: {len(loader)} batches')

    # ── Load MOMENT-small (architecture only, random init) ───────────────────
    print('[Model] Initializing MOMENT-1-small from scratch...')
    model = MOMENTPipeline.from_pretrained(
        'AutonLab/MOMENT-1-small',
        local_files_only=True,
        model_kwargs={
            'task_name': 'reconstruction',
            'mask_ratio': 0.3,
            'freeze_encoder': False,
            'freeze_embedder': False,
            'freeze_head': False,
        },
    )
    def _reset_weights(m):
        if hasattr(m, 'reset_parameters'):
            m.reset_parameters()
    model.apply(_reset_weights)
    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    total_steps  = len(loader) * epochs
    warmup_steps = min(1000, max(1, int(total_steps * 0.05)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        warmup_cosine_with_min_lr_lambda(
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
            warmup_start_lr=1e-5, base_lr=lr, min_lr=1e-5,
        ),
    )

    # ── Patch-RHO-CM: train DLinear masked-recon ref, keep model live ──────
    # We apply MOMENT's actual per-step mask to the ref each batch so ρ uses
    # fully matched (sample, channel, patch) predictions. No N×C×P table.
    ref_model = None
    if mode == 'patch_rho_cm':
        t0 = time.time()
        # ref bs fixed to 256 — DLinear is tiny, large batch reduces step count
        ref_model = build_ref_model_masked(
            dataset,
            seq_len=SEQ_LEN, ref_epochs=ref_epochs, batch_size=256,
            device=device, collate_fn=collate_fn, seed=seed,
            patch_len=PATCH_LEN, out_dir=out_dir,
        )
        ref_model.eval()
        for p in ref_model.parameters():
            p.requires_grad_(False)
        print(f'[Patch-RHO-CM] ref build time: {time.time()-t0:.1f}s')

    # ── Training ──────────────────────────────────────────────────────────────
    metrics = {'train_loss': [], 'kept_ratio': []}
    best_loss = float('inf')

    # Per-step loss log
    step_log_path = out_dir / 'step_loss.csv'
    step_log_f = open(step_log_path, 'w')
    step_log_f.write('step,loss,kept_ratio\n')
    step_log_every = 5   # MOMENT has fewer steps (chunked forward), log denser

    for epoch in range(epochs):
        model.train()
        # GPU-side accumulation — single .item() at epoch end avoids per-step CUDA sync
        epoch_loss = torch.zeros((), device=device)
        epoch_kept = torch.zeros((), device=device)
        n_batches  = 0
        step_in_epoch = 0

        for indices, x, ch_mask in tqdm(loader, desc=f'epoch {epoch+1}/{epochs}'):
            indices = indices.to(device, non_blocking=True)
            x       = x.to(device, non_blocking=True)        # (B, C, SEQ_LEN)
            ch_mask = ch_mask.to(device, non_blocking=True)   # (B, C) bool
            B, C, T = x.shape

            if mode == 'baseline':
                per_bc = moment_recon_loss_per_channel(model, x, ch_mask)
                loss = (per_bc * ch_mask.float()).sum() / ch_mask.float().sum()
                kept_ratio = torch.ones((), device=device)

            elif mode == 'patch_rho_cm':
                # Single forward with grad; detach for rho
                per_patch, pmask_bool = moment_recon_loss_per_patch(model, x, ch_mask)

                # Apply MOMENT's actual mask to the DLinear ref so its MSE is
                # measured on the SAME quantity (sample, channel, patch).
                Bx, Cx, Px = pmask_bool.shape
                x_flat     = x.reshape(Bx * Cx, T)
                pmask_flat = (pmask_bool > 0.5).reshape(Bx * Cx, Px)
                with torch.no_grad():
                    ref_pred  = ref_model.predict_with_mask(x_flat, pmask_flat)
                    ref_sq    = (ref_pred - x_flat) ** 2
                    ref_patch = ref_sq.reshape(
                        Bx * Cx, Px, PATCH_LEN
                    ).mean(dim=2).reshape(Bx, Cx, Px)

                rho        = per_patch.detach() - ref_patch
                valid_mask = ch_mask.unsqueeze(-1) & (pmask_bool > 0.5)
                rho_patch_mask = compute_rho_mask(rho, valid_mask, drop_pct)
                kept_ratio = (rho_patch_mask.float().sum() /
                              valid_mask.float().sum().clamp(min=1))
                loss = (per_patch * rho_patch_mask.float()).sum() / \
                       rho_patch_mask.float().sum().clamp(min=1)

            elif mode == 'random_mask':
                # Ablation: random rho instead of DLinear-based ranking.
                # Tests whether SPM win is from selection or from random-drop regularization.
                per_patch, pmask_bool = moment_recon_loss_per_patch(model, x, ch_mask)
                rho = torch.rand_like(per_patch)
                valid_mask = ch_mask.unsqueeze(-1) & (pmask_bool > 0.5)
                rho_patch_mask = compute_rho_mask(rho, valid_mask, drop_pct)
                kept_ratio = (rho_patch_mask.float().sum() /
                              valid_mask.float().sum().clamp(min=1))
                loss = (per_patch * rho_patch_mask.float()).sum() /                        rho_patch_mask.float().sum().clamp(min=1)

            else:
                raise ValueError(f'Unknown mode: {mode}')

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            scheduler.step()

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

        best_loss = save_epoch(out_dir, epoch + 1, model.state_dict(),
                               avg_loss, best_loss)

    step_log_f.close()
    dump_metrics(out_dir, metrics)
    print(f'\nPretraining done. Best loss={best_loss:.4f}  ckpt: {out_dir}/best.pt')
    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# Downstream evaluation: linear probe on forecasting
# ─────────────────────────────────────────────────────────────────────────────

def load_moment_with_ckpt(ckpt_path: Path, horizon: int, device: torch.device):
    """Load MOMENT-small with forecasting head, injecting pretrained encoder weights."""
    model = MOMENTPipeline.from_pretrained(
        'AutonLab/MOMENT-1-small',
        model_kwargs={
            'task_name': 'reconstruction',
            'freeze_encoder': False,
            'freeze_embedder': False,
            'freeze_head': False,
        },
    )

    if ckpt_path is not None:
        ckpt = torch.load(ckpt_path, map_location='cpu')
        state = ckpt['state_dict']
        model_state = model.state_dict()
        filtered = {k: v for k, v in state.items()
                    if k in model_state and 'head' not in k}
        model_state.update(filtered)
        model.load_state_dict(model_state, strict=False)
        print(f'  loaded {len(filtered)} keys from {ckpt_path.name}')

    model.new_task_name = 'forecasting'
    model.config.forecast_horizon = horizon
    model.init()

    for name, param in model.named_parameters():
        param.requires_grad = 'head' in name

    return model.to(device)


def eval_forecasting(
    ckpt_path: Path | None,
    dataset_name: str,
    horizon: int,
    seed: int,
    device: torch.device,
    probe_epochs: int = 5,
    probe_lr: float = 1e-4,
    batch_size: int = 64,
    train_frac: float = 1.0,
) -> dict:
    """Linear probe: freeze encoder/embedder, train only the forecasting head."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    print(f'\n[Eval] {dataset_name} H={horizon}  ckpt={ckpt_path}')

    train_ds, val_ds, test_ds = prepare_forecast_datasets(
        dataset_name, DATA_DIR, SEQ_LEN, horizon, stride=1,
    )

    # Few-shot: subsample train data
    if train_frac < 1.0:
        from torch.utils.data import Subset
        n_full = len(train_ds)
        n_keep = max(1, int(round(n_full * train_frac)))
        rng = np.random.default_rng(seed)
        idxs = rng.choice(n_full, size=n_keep, replace=False)
        train_ds = Subset(train_ds, sorted(idxs.tolist()))
        print(f'  [few-shot] train_frac={train_frac}: kept {n_keep}/{n_full} samples')

    model = load_moment_with_ckpt(ckpt_path, horizon, device)

    def run_loader(ds, shuffle):
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                          num_workers=0, pin_memory=False)

    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=probe_lr
    )

    for epoch in range(probe_epochs):
        model.train()
        epoch_loss, n = 0.0, 0
        for batch in run_loader(train_ds, True):
            x, y, mask, *_ = batch
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            B, C, T = x.shape
            x_enc = x.reshape(B * C, 1, T)
            inp   = torch.ones(B * C, T, device=device)
            out   = model(x_enc=x_enc, input_mask=inp)
            pred  = out.forecast.reshape(B, C, horizon)
            loss  = ((pred - y) ** 2).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n += 1
        print(f'  probe epoch {epoch+1}/{probe_epochs}: loss={epoch_loss/n:.4f}')

    model.eval()
    mse_list, mae_list = [], []
    with torch.no_grad():
        for batch in run_loader(test_ds, False):
            x, y, mask, *_ = batch
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            B, C, T = x.shape
            x_enc = x.reshape(B * C, 1, T)
            inp   = torch.ones(B * C, T, device=device)
            out   = model(x_enc=x_enc, input_mask=inp)
            pred  = out.forecast.reshape(B, C, horizon)
            mse_list.append(((pred - y) ** 2).mean().item())
            mae_list.append((pred - y).abs().mean().item())

    mse = float(np.mean(mse_list))
    mae = float(np.mean(mae_list))
    print(f'  → MSE={mse:.4f}  MAE={mae:.4f}')
    return {'dataset': dataset_name, 'horizon': horizon, 'test_mse': mse, 'test_mae': mae}


def _moment_probe_eval_fn(ckpt_path, ds, h, *, seed, device, probe_epochs, train_frac=1.0):
    return eval_forecasting(ckpt_path, ds, h, seed, device,
                            probe_epochs=probe_epochs, train_frac=train_frac)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode',
                        choices=['baseline', 'patch_rho_cm', 'random_mask', 'eval', 'eval_sweep'],
                        default='patch_rho_cm')

    # Pretraining
    parser.add_argument('--epochs',      type=int,   default=2)
    parser.add_argument('--batch-size',  type=int,   default=32)
    parser.add_argument('--lr',          type=float, default=1e-4)
    parser.add_argument('--drop-pct',    type=float, default=10.0,
                        help='%% of (sample, channel, patch) triples to drop (patch_rho_cm only)')
    parser.add_argument('--ref-epochs',  type=int,   default=3,
                        help='DLinear reference training epochs (patch_rho_cm only)')
    parser.add_argument('--max-series',  type=int,   default=None,
                        help='Limit number of UTSD series (for quick tests)')
    parser.add_argument('--out-dir',     type=str,   default=None)
    parser.add_argument('--seed',        type=int,   default=42)

    # Eval
    parser.add_argument('--ckpt',            type=str, default=None)
    parser.add_argument('--eval-dataset',    type=str, default='ETTh1')
    parser.add_argument('--eval-horizon',    type=int, default=96)
    parser.add_argument('--probe-epochs',    type=int, default=1)
    parser.add_argument('--train-frac',      type=float, default=1.0,
                        help='Fraction of training data to use for linear probe (few-shot)')
    parser.add_argument('--probe-lr',        type=float, default=1e-4)
    parser.add_argument('--baseline-ckpt',     type=str, default=None,
                        help='baseline checkpoint for eval_sweep')
    parser.add_argument('--patch-rho-cm-ckpt', type=str, default=None,
                        help='patch_rho_cm checkpoint for eval_sweep')
    parser.add_argument('--eval-datasets',   type=str,
                        default='ETTh1,ETTh2,ETTm1,ETTm2,weather,exchange')
    parser.add_argument('--eval-horizons',   type=str, default='96,192,336,720')

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    if args.out_dir is None:
        args.out_dir = f'results_pretrain/{args.mode}'

    if args.mode == 'eval':
        assert args.ckpt, '--ckpt required for eval mode'
        eval_forecasting(
            Path(args.ckpt), args.eval_dataset, args.eval_horizon,
            args.seed, device,
            probe_epochs=args.probe_epochs, probe_lr=args.probe_lr,
        )
    elif args.mode == 'eval_sweep':
        ckpt_paths = {}
        for name, cli_val, default_path in [
            ('baseline',     args.baseline_ckpt,     'results_pretrain/baseline/best.pt'),
            ('patch_rho_cm', args.patch_rho_cm_ckpt, 'results_pretrain/patch_rho_cm/best.pt'),
        ]:
            if cli_val:
                ckpt_paths[name] = Path(cli_val)
            elif Path(default_path).exists():
                ckpt_paths[name] = Path(default_path)
        if not ckpt_paths:
            print('No checkpoints found. Pass --baseline-ckpt and/or --patch-rho-cm-ckpt')
        else:
            run_eval_sweep(
                _moment_probe_eval_fn,
                ckpt_paths   = ckpt_paths,
                datasets     = args.eval_datasets.split(','),
                horizons     = [int(h) for h in args.eval_horizons.split(',')],
                title        = 'LINEAR PROBE RESULTS',
                out_filename = 'linear_probe_results.json',
                out_dir      = Path(args.out_dir),
                seed         = args.seed,
                device       = device,
                probe_epochs = args.probe_epochs,
                train_frac   = args.train_frac,
            )
    else:
        pretrain(
            mode=args.mode,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            drop_pct=args.drop_pct,
            ref_epochs=args.ref_epochs,
            out_dir=Path(args.out_dir),
            seed=args.seed,
            max_series=args.max_series,
            device=device,
        )


if __name__ == '__main__':
    main()
