"""drop_pct ablation analysis: scan all patch_rho_cm runs at lr=1e-4, e=1, s=42
with different drop_pct values, compare against baseline_s42."""
import json
import re
from pathlib import Path

ROOT = Path('/mnt/workspace/juyoung_ha/rho_pretrain/logs/sweep_v1')

SLIM_DS = {'ETTh1', 'ETTh2', 'ETTm1', 'ETTm2', 'weather'}
SLIM_H  = {96, 192, 336}
SLIM = {(d, h) for d in SLIM_DS for h in SLIM_H}


def load_run(run_dir):
    p = run_dir / 'linear_probe_results.json'
    if not p.exists():
        return None
    rows = next(iter(json.loads(p.read_text()).values()))
    return [r for r in rows if (r['dataset'], r['horizon']) in SLIM]


def mean_metrics(rows):
    if not rows:
        return None, None
    n = len(rows)
    return sum(r['test_mse'] for r in rows) / n, sum(r['test_mae'] for r in rows) / n


def parse_drop_pct(name):
    m = re.search(r'_d(\d+)', name)
    if m:
        return int(m.group(1))
    return 10  # default


def main():
    print(f'Reading from: {ROOT}\n')

    # baseline at the same lr/epochs/seed as the ablation
    base_dir = ROOT / 'baseline__bs64_lr0.0001_e1_s42'
    base_rows = load_run(base_dir)
    base_mse, base_mae = mean_metrics(base_rows) if base_rows else (None, None)
    print(f'baseline (lr=1e-4 e=1 s=42, slim n={len(base_rows) if base_rows else 0}): '
          f'MSE={base_mse:.4f} MAE={base_mae:.4f}\n')

    # all patch_rho_cm runs at lr=1e-4, e=1, s=42 with various drop_pct
    rho_runs = []
    for d in sorted(ROOT.glob('patch_rho_cm__bs64_lr0.0001_e1_s42*')):
        dp = parse_drop_pct(d.name)
        rows = load_run(d)
        if rows is None:
            continue
        mse, mae = mean_metrics(rows)
        rho_runs.append((dp, mse, mae, len(rows)))

    rho_runs.sort()
    print('=' * 60)
    print('Drop-pct ablation (single seed=42, slim subset)')
    print('=' * 60)
    print(f'{"drop%":<6} {"n":<4} {"MSE":>10} {"ΔMSE":>10} {"MAE":>10} {"ΔMAE":>10}')
    for dp, mse, mae, n in rho_runs:
        dmse = (mse - base_mse) if base_mse is not None else 0
        dmae = (mae - base_mae) if base_mae is not None else 0
        flag = ' ✓' if dmse < 0 else ''
        print(f'{dp:<6} {n:<4} {mse:>10.4f} {dmse:>+10.4f} '
              f'{mae:>10.4f} {dmae:>+10.4f}{flag}')


if __name__ == '__main__':
    main()
