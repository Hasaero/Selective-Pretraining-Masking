"""Overnight matched-pair sweep harness for Moirai and Timer.

For each hparam config in the grid, runs:
  1. Pretrain baseline → out_root/<run>/baseline/best.pt + metrics.json
  2. Pretrain rho_cm   → out_root/<run>/rho_cm/best.pt   + metrics.json
  3. Eval zero-shot sweep on both checkpoints (ETT × {96,192,336,720} + weather/exchange)
  4. Append summary row to out_root/sweep_summary.jsonl

RESUMABLE: skips runs whose eval_results.json already exists.

Usage:
    python scripts/sweep/sweep_overnight.py --model moirai --grid main
    python scripts/sweep/sweep_overnight.py --model timer  --grid main
"""
import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
PYTHON = sys.executable

EVAL_DATASETS = 'ETTh1,ETTh2,ETTm1,ETTm2,weather,exchange'
EVAL_HORIZONS = '96,192,336,720'

SCRIPTS = {
    'moirai': ROOT / 'scripts' / 'pretrain_moirai.py',
    'timer':  ROOT / 'scripts' / 'pretrain_timer.py',
}

# Eval mode per model: zero-shot sweep is the headline metric the user cares about.
EVAL_MODES = {
    'moirai': 'eval_zero_shot_sweep',
    'timer':  'eval_zero_shot_sweep',
}

EVAL_OUT_FILENAME = {
    'moirai': 'zero_shot_results.json',
    'timer':  'zero_shot_results.json',
}


def grids(model):
    """Hparam grids per model. Each entry is (batch, lr, epochs, ref_epochs, drop_pct)."""
    if model == 'moirai':
        return {
            'quick': [
                (4, 1e-3, 1, 2, 10),
            ],
            # Main sweep: 3 lrs × 2 epoch settings; batch=4 is the highest stable
            # value for Moirai's packed attention even with channel cap=32 (max
            # H200 OOMs at batch>=8 due to outlier wide-channel batches).
            'main': [
                (4, 5e-4, 2, 3, 10),
                (4, 1e-3, 2, 3, 10),
                (4, 2e-3, 2, 3, 10),
                (4, 1e-3, 3, 3, 10),
            ],
            # Stable lr range: 1e-3 / 2e-3 NaN'd in main sweep (gradient
            # explosion in NLL head). Try smaller lrs + drop_pct ablation.
            'stable': [
                (4, 1e-4, 2, 3, 10),
                (4, 3e-4, 2, 3, 10),
                (4, 3e-4, 1, 3, 10),
                (4, 3e-4, 2, 3, 5),
                (4, 3e-4, 2, 3, 20),
            ],
        }
    elif model == 'timer':
        return {
            'quick': [
                (32, 3e-4, 1, 5, 10),
            ],
            # Timer: channel-independent — N_real scales with batch * avg_C.
            # batch=32 at channel cap p99 keeps N_real ~thousand level (safe).
            'main': [
                (32, 1e-4, 2, 5, 10),
                (32, 3e-4, 2, 5, 10),
                (32, 1e-3, 2, 5, 10),
                (32, 3e-4, 3, 5, 10),
            ],
            # Stable + drop_pct ablation, mirroring Moirai findings: low lr +
            # high drop_pct seem to favor rho_cm.
            'stable': [
                (32, 3e-4, 2, 5, 10),
                (32, 3e-4, 2, 5, 20),
                (32, 1e-4, 2, 5, 10),
                (32, 1e-4, 2, 5, 20),
                (32, 3e-4, 1, 5, 10),
            ],
            # Single best config for full-UTSD validation (no max_series cap).
            'best': [
                (32, 3e-4, 2, 5, 20),
            ],
        }
    raise ValueError(f'Unknown model: {model}')


def run_pretrain(model, mode, batch, lr, epochs, seed, ref_epochs, drop_pct,
                 out_dir, max_series=None):
    out_dir.mkdir(parents=True, exist_ok=True)
    log = out_dir / 'pretrain.log'
    cmd = [
        PYTHON, str(SCRIPTS[model]),
        '--mode', mode,
        '--epochs', str(epochs),
        '--batch-size', str(batch),
        '--lr', str(lr),
        '--seed', str(seed),
        '--out-dir', str(out_dir),
    ]
    if mode != 'baseline':
        cmd += ['--ref-epochs', str(ref_epochs), '--drop-pct', str(drop_pct)]
    if max_series is not None:
        cmd += ['--max-series', str(max_series)]
    print(f'[run] pretrain {model}/{mode} → {out_dir.relative_to(ROOT)}')
    print(f'      cmd: {" ".join(cmd[2:])}')
    t0 = time.time()
    with open(log, 'w') as f:
        f.write(f'cmd: {" ".join(cmd)}\n')
        f.flush()
        rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=str(ROOT)).returncode
    dt = time.time() - t0
    print(f'      done in {dt/60:.1f}min  rc={rc}')
    return rc, dt


def run_eval_sweep(model, baseline_ckpt, rho_cm_ckpt, out_dir, batch_size=32, num_samples=20):
    """Zero-shot eval sweep over baseline and rho_cm together (so they share datasets/horizons)."""
    log = out_dir / 'eval.log'
    cmd = [
        PYTHON, str(SCRIPTS[model]),
        '--mode', EVAL_MODES[model],
        '--out-dir', str(out_dir),
        '--baseline-ckpt', str(baseline_ckpt),
        '--rho-cm-ckpt',   str(rho_cm_ckpt),
        '--eval-datasets', EVAL_DATASETS,
        '--eval-horizons', EVAL_HORIZONS,
        '--batch-size',    str(batch_size),
    ]
    if model == 'moirai':
        cmd += ['--num-samples', str(num_samples)]
    print(f'[run] eval {model} sweep → {out_dir.relative_to(ROOT)}')
    t0 = time.time()
    with open(log, 'w') as f:
        f.write(f'cmd: {" ".join(cmd)}\n')
        f.flush()
        rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=str(ROOT)).returncode
    dt = time.time() - t0
    print(f'      done in {dt/60:.1f}min  rc={rc}')
    return rc, dt


def summarize(out_dir, eval_filename):
    p = out_dir / eval_filename
    if not p.exists():
        return None
    data = json.loads(p.read_text())
    # data is {label: [{dataset, horizon, test_mse, test_mae}, ...]}
    summary = {}
    for label, rows in data.items():
        if not rows:
            continue
        mse = sum(r['test_mse'] for r in rows) / len(rows)
        mae = sum(r['test_mae'] for r in rows) / len(rows)
        summary[label] = {
            'n_results': len(rows),
            'mean_mse': round(mse, 6),
            'mean_mae': round(mae, 6),
        }
    if 'baseline' in summary and 'rho_cm' in summary:
        summary['delta_mse'] = round(summary['rho_cm']['mean_mse'] - summary['baseline']['mean_mse'], 6)
        summary['delta_mae'] = round(summary['rho_cm']['mean_mae'] - summary['baseline']['mean_mae'], 6)
        summary['rho_wins'] = summary['delta_mse'] < 0
    return summary


def append_summary(summary_path, row):
    with open(summary_path, 'a') as f:
        f.write(json.dumps(row) + '\n')


def run_name(batch, lr, epochs, ref_epochs, drop_pct):
    return f'bs{batch}_lr{lr:g}_e{epochs}_re{ref_epochs}_d{drop_pct:g}'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True, choices=['moirai', 'timer'])
    ap.add_argument('--grid', required=True)
    ap.add_argument('--out-root', default=None,
                    help='default: logs/sweep_<model>_<grid>')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--max-series', type=int, default=None,
                    help='limit UTSD series for fast iteration')
    ap.add_argument('--eval-batch-size', type=int, default=32)
    ap.add_argument('--num-samples', type=int, default=20,
                    help='moirai-only: MC samples for zero-shot prediction')
    args = ap.parse_args()

    if args.out_root is None:
        args.out_root = f'logs/sweep_{args.model}_{args.grid}'
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    summary_path = out_root / 'sweep_summary.jsonl'
    os.chdir(str(ROOT))

    g = grids(args.model)[args.grid]
    print(f'[sweep] model={args.model} grid={args.grid}: {len(g)} configs → {out_root}')
    started = datetime.now().isoformat(timespec='seconds')

    eval_filename = EVAL_OUT_FILENAME[args.model]

    for i, spec in enumerate(g, 1):
        batch, lr, epochs, ref_epochs, drop_pct = spec
        rn = run_name(batch, lr, epochs, ref_epochs, drop_pct)
        run_dir = out_root / rn
        bdir = run_dir / 'baseline'
        rdir = run_dir / 'rho_cm'
        edir = run_dir / 'eval'
        print(f'\n[{i}/{len(g)}] {rn}')

        # Skip whole config if evaluation already done.
        if (edir / eval_filename).exists():
            print(f'      [skip] {edir / eval_filename} already exists')
            continue

        # Pretrain baseline (skip if best.pt exists)
        if not (bdir / 'best.pt').exists():
            rc, _ = run_pretrain(args.model, 'baseline', batch, lr, epochs,
                                  args.seed, ref_epochs, drop_pct, bdir,
                                  max_series=args.max_series)
            if rc != 0 or not (bdir / 'best.pt').exists():
                print(f'      [!] baseline pretrain failed (rc={rc}); skipping config')
                append_summary(summary_path, {
                    'time': datetime.now().isoformat(timespec='seconds'),
                    'run': rn, 'status': 'baseline_failed',
                    'batch': batch, 'lr': lr, 'epochs': epochs,
                    'ref_epochs': ref_epochs, 'drop_pct': drop_pct,
                })
                continue
        else:
            print(f'      [skip pretrain] baseline best.pt exists')

        # Pretrain rho_cm
        if not (rdir / 'best.pt').exists():
            rc, _ = run_pretrain(args.model, 'rho_cm', batch, lr, epochs,
                                  args.seed, ref_epochs, drop_pct, rdir,
                                  max_series=args.max_series)
            if rc != 0 or not (rdir / 'best.pt').exists():
                print(f'      [!] rho_cm pretrain failed (rc={rc}); skipping eval')
                append_summary(summary_path, {
                    'time': datetime.now().isoformat(timespec='seconds'),
                    'run': rn, 'status': 'rho_cm_failed',
                    'batch': batch, 'lr': lr, 'epochs': epochs,
                    'ref_epochs': ref_epochs, 'drop_pct': drop_pct,
                })
                continue
        else:
            print(f'      [skip pretrain] rho_cm best.pt exists')

        # Eval sweep on both checkpoints
        edir.mkdir(parents=True, exist_ok=True)
        rc, _ = run_eval_sweep(args.model, bdir / 'best.pt', rdir / 'best.pt',
                                edir, batch_size=args.eval_batch_size,
                                num_samples=args.num_samples)
        s = summarize(edir, eval_filename)
        row = {
            'time': datetime.now().isoformat(timespec='seconds'),
            'run': rn, 'status': 'ok' if (s is not None and rc == 0) else 'eval_failed',
            'batch': batch, 'lr': lr, 'epochs': epochs,
            'ref_epochs': ref_epochs, 'drop_pct': drop_pct,
            'summary': s,
        }
        append_summary(summary_path, row)
        print(f'      summary: {json.dumps(s, indent=2) if s else None}')

    print(f'\n[sweep] DONE. started={started} ended={datetime.now().isoformat(timespec="seconds")}')
    print(f'        summary: {summary_path}')


if __name__ == '__main__':
    sys.exit(main() or 0)
