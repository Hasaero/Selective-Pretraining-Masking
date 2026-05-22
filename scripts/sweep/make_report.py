"""Aggregate sweep_summary.jsonl + per-run zero_shot_results.json into a Markdown report.

Usage:
    python scripts/sweep/make_report.py <sweep_dir> [--out RESULTS.md]
"""
import argparse
import json
import sys
from pathlib import Path


def load_summary(sweep_dir: Path):
    p = sweep_dir / 'sweep_summary.jsonl'
    if not p.exists():
        return []
    rows = []
    for line in p.read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def load_per_run(sweep_dir: Path, run_name: str, eval_filename: str):
    p = sweep_dir / run_name / 'eval' / eval_filename
    if not p.exists():
        return None
    return json.loads(p.read_text())


def fmt_table(rows, headers):
    """Markdown table from list of lists."""
    out = ['| ' + ' | '.join(headers) + ' |',
           '| ' + ' | '.join('---' for _ in headers) + ' |']
    for r in rows:
        out.append('| ' + ' | '.join(str(c) for c in r) + ' |')
    return '\n'.join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('sweep_dir')
    ap.add_argument('--model', required=True, choices=['moirai', 'timer'])
    ap.add_argument('--out', default=None,
                    help='Output markdown (default: <sweep_dir>/RESULTS.md)')
    args = ap.parse_args()

    sweep_dir = Path(args.sweep_dir)
    out_path = Path(args.out) if args.out else sweep_dir / f'RESULTS_{args.model.upper()}.md'

    # Eval filename per model (zero-shot only — what the user asked for)
    eval_filename = 'zero_shot_results.json'

    summary = load_summary(sweep_dir)
    if not summary:
        print(f'No sweep_summary.jsonl in {sweep_dir}', file=sys.stderr)
        return 1

    md = []
    md.append(f'# {args.model.title()} sweep results — {sweep_dir.name}')
    md.append('')
    md.append('Each row is a (batch, lr, epochs) configuration. Both baseline '
              'and rho_cm were trained with identical hparams and seed=42, '
              'then evaluated zero-shot on 6 datasets × 4 horizons (24 settings).')
    md.append('')
    md.append('## Configuration ranking (by mean test MSE, baseline vs rho_cm)')
    md.append('')

    # Build ranking table
    rows = []
    for r in summary:
        if r['status'] != 'ok' or 'summary' not in r:
            continue
        s = r['summary']
        rows.append([
            r['run'],
            f"{s['baseline']['mean_mse']:.4f}",
            f"{s['rho_cm']['mean_mse']:.4f}",
            f"{s['delta_mse']:+.4f}",
            'YES' if s.get('rho_wins') else 'no',
        ])
    rows.sort(key=lambda x: float(x[2]))   # by rho_cm MSE ascending

    md.append(fmt_table(
        rows,
        ['Config', 'Baseline MSE', 'rho_cm MSE', 'ΔMSE (rho-base)', 'rho wins?'],
    ))
    md.append('')

    # Detailed per-config breakdown
    md.append('## Per-(dataset, horizon) breakdown')
    md.append('')
    for r in summary:
        if r['status'] != 'ok' or 'summary' not in r:
            continue
        rn = r['run']
        data = load_per_run(sweep_dir, rn, eval_filename)
        if data is None:
            continue
        s = r['summary']

        md.append(f'### Config: `{rn}`')
        md.append('')
        md.append(f'- Baseline: mean MSE = {s["baseline"]["mean_mse"]:.4f}, mean MAE = {s["baseline"]["mean_mae"]:.4f}')
        md.append(f'- rho_cm:   mean MSE = {s["rho_cm"]["mean_mse"]:.4f}, mean MAE = {s["rho_cm"]["mean_mae"]:.4f}')
        md.append(f'- ΔMSE = {s["delta_mse"]:+.4f}, ΔMAE = {s["delta_mae"]:+.4f} ({"rho_cm wins" if s.get("rho_wins") else "baseline wins"})')
        md.append('')

        # per-(ds, h) table
        bl_rows = {(rr['dataset'], rr['horizon']): rr for rr in data.get('baseline', [])}
        rh_rows = {(rr['dataset'], rr['horizon']): rr for rr in data.get('rho_cm', [])}
        keys = sorted(bl_rows.keys() & rh_rows.keys())
        rows = []
        for ds, h in keys:
            b = bl_rows[(ds, h)]
            rc = rh_rows[(ds, h)]
            d_mse = rc['test_mse'] - b['test_mse']
            rows.append([
                ds, h,
                f"{b['test_mse']:.4f}", f"{b['test_mae']:.4f}",
                f"{rc['test_mse']:.4f}", f"{rc['test_mae']:.4f}",
                f"{d_mse:+.4f}",
                '✓' if d_mse < 0 else '',
            ])
        md.append(fmt_table(
            rows,
            ['Dataset', 'H', 'Baseline MSE', 'Baseline MAE',
             'rho_cm MSE', 'rho_cm MAE', 'ΔMSE', 'rho wins'],
        ))
        md.append('')

    # Best config callout
    md.append('## Best configuration')
    md.append('')
    ok = [r for r in summary if r['status'] == 'ok' and 'summary' in r]
    if ok:
        best = min(ok, key=lambda r: r['summary']['rho_cm']['mean_mse'])
        md.append(f'Lowest rho_cm mean MSE: **`{best["run"]}`** with '
                  f'MSE={best["summary"]["rho_cm"]["mean_mse"]:.4f}, '
                  f'MAE={best["summary"]["rho_cm"]["mean_mae"]:.4f}.')
        md.append('')
        winning = [r for r in ok if r['summary'].get('rho_wins')]
        md.append(f'Configurations where rho_cm beats baseline: '
                  f'**{len(winning)}/{len(ok)}**.')

    out_path.write_text('\n'.join(md))
    print(f'Wrote {out_path}  ({len(md)} lines)')
    return 0


if __name__ == '__main__':
    sys.exit(main() or 0)
