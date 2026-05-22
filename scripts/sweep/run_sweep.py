"""MOMENT pretraining + eval sweep harness.

Drives `pretrain_moment.py` over a grid of (mode, batch, lr, epochs, seed)
and a downstream eval sweep on ETT/electricity/.../weather × {96,192,336,720}.

Each run:
  1. Pretrain → checkpoint at `out_root/<run_name>/best.pt`
  2. Eval sweep → per-(dataset, horizon) MSE/MAE → `out_root/<run_name>/eval.json`
  3. Append summary row to `out_root/sweep_summary.jsonl`

Usage:
    python scripts/sweep/run_sweep.py --grid quick   # short timing/sanity grid
    python scripts/sweep/run_sweep.py --grid full    # full hparam search
    python scripts/sweep/run_sweep.py --grid match   # baseline vs rho at fixed hparams × seeds

Designed to be RESUMABLE: skips runs whose `eval.json` already exists, so
killing/restarting the script just continues the remaining work.
"""
import argparse
import itertools
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPT = ROOT / 'scripts' / 'pretrain_moment.py'
PYTHON = sys.executable

# Default eval — full benchmark (slow but comprehensive).
EVAL_DATASETS_FULL = 'ETTh1,ETTh2,ETTm1,ETTm2,weather,exchange,electricity'
EVAL_HORIZONS_FULL = '96,192,336,720'

# Slim eval — used by match grids to fit overnight budget. Drops electricity
# (largest, slowest) and the longest horizon (720). 5×3 = 15 results, ~3× faster
# than the full 28.
EVAL_DATASETS_SLIM = 'ETTh1,ETTh2,ETTm1,ETTm2,weather'
EVAL_HORIZONS_SLIM = '96,192,336'

EVAL_DATASETS = EVAL_DATASETS_FULL
EVAL_HORIZONS = EVAL_HORIZONS_FULL


def run_name(mode, batch, lr, epochs, seed, drop_pct=10):
    base = f'{mode}__bs{batch}_lr{lr:g}_e{epochs}_s{seed}'
    if mode != 'baseline' and drop_pct != 10:
        base += f'_d{drop_pct:g}'
    return base


def already_done(out_dir: Path) -> bool:
    """A run is done iff best.pt + linear_probe_results.json both exist."""
    return (out_dir / 'best.pt').exists() and (out_dir / 'linear_probe_results.json').exists()


def pretrain(mode, batch, lr, epochs, seed, out_dir, drop_pct=10):
    out_dir.mkdir(parents=True, exist_ok=True)
    log = out_dir / 'pretrain.log'
    cmd = [
        PYTHON, str(SCRIPT),
        '--mode', mode,
        '--epochs', str(epochs),
        '--batch-size', str(batch),
        '--lr', str(lr),
        '--seed', str(seed),
        '--out-dir', str(out_dir),
    ]
    if mode != 'baseline':
        cmd += ['--ref-epochs', '3', '--drop-pct', str(drop_pct)]
    print(f'[run] pretrain → {out_dir.name}')
    print(f'      log: {log}')
    t0 = time.time()
    with open(log, 'w') as f:
        f.write(f'cmd: {" ".join(cmd)}\n')
        f.flush()
        rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=str(ROOT)).returncode
    dt = time.time() - t0
    print(f'      done in {dt/60:.1f}min  rc={rc}')
    return rc, dt


def eval_sweep(ckpt_path, out_dir, *, probe_epochs=None, datasets=None, horizons=None):
    """Run linear-probe eval sweep against the checkpoint."""
    log = out_dir / 'eval.log'
    cmd = [
        PYTHON, str(SCRIPT),
        '--mode', 'eval_sweep',
        '--out-dir', str(out_dir),
        '--probe-epochs', str(probe_epochs or 5),
        '--eval-datasets', datasets or EVAL_DATASETS,
        '--eval-horizons', horizons or EVAL_HORIZONS,
        '--baseline-ckpt', str(ckpt_path),  # use baseline-ckpt slot to label as "the model"
    ]
    print(f'[run] eval     → {out_dir.name}')
    t0 = time.time()
    with open(log, 'w') as f:
        f.write(f'cmd: {" ".join(cmd)}\n')
        f.flush()
        rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=str(ROOT)).returncode
    dt = time.time() - t0
    print(f'      done in {dt/60:.1f}min  rc={rc}')
    return rc, dt


def summarize(out_dir):
    """Aggregate per-(dataset, horizon) results into mean MSE/MAE."""
    p = out_dir / 'linear_probe_results.json'
    if not p.exists():
        return None
    data = json.loads(p.read_text())
    # data is {label: [{dataset, horizon, test_mse, test_mae}, ...]}
    rows = next(iter(data.values()))  # only one label because we used baseline-ckpt slot
    if not rows:
        return None
    mse = sum(r['test_mse'] for r in rows) / len(rows)
    mae = sum(r['test_mae'] for r in rows) / len(rows)
    return {'n_results': len(rows), 'mean_mse': mse, 'mean_mae': mae,
            'per_setting': rows}


def append_summary(summary_path, row):
    with open(summary_path, 'a') as f:
        f.write(json.dumps(row) + '\n')


def grids():
    return {
        # Sanity grid (verifies harness end-to-end on a tiny subset).
        'quick': [
            ('baseline',     32, 1e-4, 1, 42),
            ('patch_rho_cm', 32, 1e-4, 1, 42),
        ],
        # Overnight plan A: lr search at fixed batch=64, 1 epoch, baseline only.
        # 3 runs × ~100min = ~5h. Picks the best lr; phase B uses it.
        'lr_search': [
            ('baseline', 64, 5e-5, 1, 42),
            ('baseline', 64, 1e-4, 1, 42),
            ('baseline', 64, 3e-4, 1, 42),
        ],
        # Overnight plan B: matched comparison at the most-likely-best config
        # (1e-4 from MOMENT default, 1 epoch, batch 64). 2 modes × 3 seeds = 6 runs.
        # ~10h total — fits in one overnight budget.
        'match_e1': [
            (mode, 64, 1e-4, 1, seed)
            for mode in ('baseline', 'patch_rho_cm')
            for seed in (42, 43, 44)
        ],
        # Plan C (longer training): 2 epochs at best config, 1 seed each.
        'match_e2': [
            ('baseline',     64, 1e-4, 2, 42),
            ('patch_rho_cm', 64, 1e-4, 2, 42),
        ],
        # drop_pct ablation at fixed best config (after match phase identifies it).
        'drop_pct': [
            ('patch_rho_cm', 64, 1e-4, 1, 42, dp) for dp in (5, 10, 20, 30)
        ],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--grid', required=True,
                    choices=['quick', 'lr_search', 'match_e1', 'match_e2', 'drop_pct'])
    ap.add_argument('--out-root', default='logs/sweep_v1')
    ap.add_argument('--slim-eval', action='store_true',
                    help='Use slim eval (5 datasets × 3 horizons, fewer probe epochs) — faster.')
    ap.add_argument('--probe-epochs', type=int, default=5)
    args = ap.parse_args()
    os.chdir(str(ROOT))
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    summary_path = out_root / 'sweep_summary.jsonl'

    g = grids()[args.grid]
    if not g:
        print(f'[!] grid {args.grid!r} is empty (must be populated dynamically).')
        return 1

    print(f'[sweep] {args.grid}: {len(g)} runs → {out_root}')
    started = datetime.now().isoformat(timespec='seconds')

    for i, spec in enumerate(g, 1):
        # spec is (mode, batch, lr, epochs, seed) or (mode, batch, lr, epochs, seed, drop_pct)
        if len(spec) == 5:
            mode, batch, lr, epochs, seed = spec
            drop_pct = 10
        else:
            mode, batch, lr, epochs, seed, drop_pct = spec
        rn = run_name(mode, batch, lr, epochs, seed, drop_pct)
        out_dir = out_root / rn
        print(f'\n[{i}/{len(g)}] {rn}')

        if already_done(out_dir):
            print(f'      [skip] eval.json already exists')
            continue

        # Pretrain (skip if best.pt already there)
        if (out_dir / 'best.pt').exists():
            print(f'      [skip pretrain] best.pt already exists')
        else:
            rc, _ = pretrain(mode, batch, lr, epochs, seed, out_dir, drop_pct=drop_pct)
            if rc != 0 or not (out_dir / 'best.pt').exists():
                print(f'      [!] pretrain failed (rc={rc}); skipping eval')
                append_summary(summary_path, {
                    'time': datetime.now().isoformat(timespec='seconds'),
                    'run': rn, 'status': 'pretrain_failed',
                    'mode': mode, 'batch': batch, 'lr': lr, 'epochs': epochs, 'seed': seed,
                    'drop_pct': drop_pct,
                })
                continue

        # Eval sweep
        if args.slim_eval:
            rc, _ = eval_sweep(out_dir / 'best.pt', out_dir,
                               probe_epochs=args.probe_epochs,
                               datasets=EVAL_DATASETS_SLIM,
                               horizons=EVAL_HORIZONS_SLIM)
        else:
            rc, _ = eval_sweep(out_dir / 'best.pt', out_dir,
                               probe_epochs=args.probe_epochs)
        s = summarize(out_dir)
        row = {
            'time': datetime.now().isoformat(timespec='seconds'),
            'run': rn, 'status': 'ok' if (s is not None and rc == 0) else 'eval_failed',
            'mode': mode, 'batch': batch, 'lr': lr, 'epochs': epochs, 'seed': seed,
            'drop_pct': drop_pct,
        }
        if s is not None:
            row['mean_mse'] = round(s['mean_mse'], 6)
            row['mean_mae'] = round(s['mean_mae'], 6)
            row['n_results'] = s['n_results']
        append_summary(summary_path, row)
        print(f'      summary: {row}')

    print(f'\n[sweep] DONE. started={started} ended={datetime.now().isoformat(timespec="seconds")}')
    print(f'        summary: {summary_path}')


if __name__ == '__main__':
    sys.exit(main() or 0)
