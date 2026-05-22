"""Aggregate match_e1 results into a baseline-vs-RHO comparison table.

Reads `linear_probe_results.json` from each match_e1 run, restricts to the
slim eval subset (5 datasets × 3 horizons = 15 results) for fair comparison,
prints mean ± std across seeds and per-(dataset, horizon) Δ.
"""
import json
from pathlib import Path
from collections import defaultdict
import statistics

ROOT = Path('/mnt/workspace/juyoung_ha/rho_pretrain/logs/sweep_v1')

SLIM_DS = ['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2', 'weather']
SLIM_H  = [96, 192, 336]
SLIM = {(d, h) for d in SLIM_DS for h in SLIM_H}


def load_run(run_dir):
    p = run_dir / 'linear_probe_results.json'
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    rows = next(iter(d.values()))
    return [r for r in rows if (r['dataset'], r['horizon']) in SLIM]


def mean_metrics(rows):
    if not rows:
        return None, None
    mse = sum(r['test_mse'] for r in rows) / len(rows)
    mae = sum(r['test_mae'] for r in rows) / len(rows)
    return mse, mae


def aggregate_by_seed(mode, lr=1e-4, epochs=1):
    out = {}
    for d in sorted(ROOT.glob(f'{mode}__bs64_lr*_e{epochs}_s*')):
        # parse seed; skip drop_pct ablation runs (suffix _d5/_d20/_d30)
        parts = d.name.rsplit('_s', 1)
        seed_str = parts[1]
        if '_d' in seed_str:
            continue
        try:
            seed = int(seed_str)
        except ValueError:
            continue
        rows = load_run(d)
        if rows is None or len(rows) < 15:
            continue
        out[seed] = rows
    return out


def per_setting_mean(by_seed):
    """Mean per (dataset, horizon) over seeds."""
    out = {}
    for ds, h in SLIM:
        vals_mse = []
        vals_mae = []
        for seed, rows in by_seed.items():
            r = next((r for r in rows if r['dataset'] == ds and r['horizon'] == h), None)
            if r is not None:
                vals_mse.append(r['test_mse'])
                vals_mae.append(r['test_mae'])
        if vals_mse:
            out[(ds, h)] = (
                sum(vals_mse) / len(vals_mse), sum(vals_mae) / len(vals_mae)
            )
    return out


def main():
    print(f'Reading from: {ROOT}\n')

    base = aggregate_by_seed('baseline', lr=1e-4)
    rho  = aggregate_by_seed('patch_rho_cm', lr=1e-4)
    print(f'baseline seeds available: {sorted(base)}')
    print(f'rho      seeds available: {sorted(rho)}\n')

    # Aggregate-level
    print('=' * 60)
    print(f'OVERALL (slim subset, {len(SLIM)} settings)')
    print('=' * 60)
    print(f'{"mode":<14} {"seeds":<8} {"MSE mean":>10} {"MSE std":>10} {"MAE mean":>10} {"MAE std":>10}')
    for label, by_seed in [('baseline', base), ('patch_rho_cm', rho)]:
        if not by_seed:
            print(f'{label:<14} (no runs)')
            continue
        per_seed_mse = []
        per_seed_mae = []
        for seed, rows in sorted(by_seed.items()):
            m, a = mean_metrics(rows)
            per_seed_mse.append(m)
            per_seed_mae.append(a)
        print(f'{label:<14} {len(by_seed):<8} '
              f'{sum(per_seed_mse)/len(per_seed_mse):>10.4f} '
              f'{(statistics.stdev(per_seed_mse) if len(per_seed_mse)>=2 else 0):>10.4f} '
              f'{sum(per_seed_mae)/len(per_seed_mae):>10.4f} '
              f'{(statistics.stdev(per_seed_mae) if len(per_seed_mae)>=2 else 0):>10.4f}')

    # Per-seed table
    print('\n' + '=' * 60)
    print('PER-SEED MEAN (slim subset)')
    print('=' * 60)
    print(f'{"mode":<14} {"seed":<6} {"MSE":>10} {"MAE":>10}')
    for label, by_seed in [('baseline', base), ('patch_rho_cm', rho)]:
        for seed, rows in sorted(by_seed.items()):
            m, a = mean_metrics(rows)
            print(f'{label:<14} {seed:<6} {m:>10.4f} {a:>10.4f}')

    # Δ-per-setting (if both available)
    if base and rho:
        print('\n' + '=' * 60)
        print('Δ MSE / Δ MAE per (dataset, horizon) — averaged across seeds')
        print('rho - baseline; negative = rho better')
        print('=' * 60)
        bps = per_setting_mean(base)
        rps = per_setting_mean(rho)
        print(f'{"dataset":<10} {"H":>5} {"base MSE":>10} {"rho MSE":>10} {"ΔMSE":>10} '
              f'{"base MAE":>10} {"rho MAE":>10} {"ΔMAE":>10}')
        better_mse = better_mae = total = 0
        for ds in SLIM_DS:
            for h in SLIM_H:
                k = (ds, h)
                if k in bps and k in rps:
                    bm, ba = bps[k]
                    rm, ra = rps[k]
                    dm, da = rm - bm, ra - ba
                    flag = ('✓' if dm < 0 else '')
                    print(f'{ds:<10} {h:>5} {bm:>10.4f} {rm:>10.4f} {dm:>+10.4f} '
                          f'{ba:>10.4f} {ra:>10.4f} {da:>+10.4f}  {flag}')
                    total += 1
                    if dm < 0: better_mse += 1
                    if da < 0: better_mae += 1
        if total:
            print(f'\nrho better on MSE: {better_mse}/{total}, MAE: {better_mae}/{total}')


if __name__ == '__main__':
    main()
