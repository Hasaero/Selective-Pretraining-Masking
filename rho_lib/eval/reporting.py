"""Result-table printing + delta-vs-baseline reporting + JSON dump."""
import json
from pathlib import Path


def print_results_table(results: dict, datasets: list, horizons: list,
                        title: str = 'RESULTS') -> None:
    """Render one row per (dataset, horizon), one column-pair per ckpt label."""
    labels = list(results.keys())
    print(f'\n{"="*60}\n{title}\n{"="*60}')
    header = f"{'Dataset':<10} {'H':>4}  " + \
             '  '.join(f"{l+' MSE':>14} {l+' MAE':>10}" for l in labels)
    print(header)
    print('-' * len(header))
    for ds in datasets:
        for h in horizons:
            row = f'{ds:<10} {h:>4}  '
            for label in labels:
                entry = next((r for r in results[label]
                              if r['dataset'] == ds and r['horizon'] == h), None)
                row += (f"  {entry['test_mse']:>14.4f} {entry['test_mae']:>10.4f}"
                        if entry else f"  {'N/A':>14} {'N/A':>10}")
            print(row)


def print_delta_tables(results: dict, datasets: list, horizons: list,
                       baseline_label: str = 'baseline') -> None:
    """Print Δ MSE / Δ MAE for every non-baseline label vs baseline."""
    if baseline_label not in results:
        return
    labels = list(results.keys())
    for comp in [l for l in labels if l != baseline_label]:
        print(f'\n{"="*60}\nDELTA ({comp} - {baseline_label})\n{"="*60}')
        print(f"{'Dataset':<10} {'H':>4}  {'ΔMSE':>10} {'ΔMAE':>10}")
        print('-' * 40)
        for ds in datasets:
            for h in horizons:
                b = next((r for r in results[baseline_label]
                          if r['dataset'] == ds and r['horizon'] == h), None)
                g = next((r for r in results[comp]
                          if r['dataset'] == ds and r['horizon'] == h), None)
                if b and g:
                    dm = g['test_mse'] - b['test_mse']
                    da = g['test_mae'] - b['test_mae']
                    print(f"{ds:<10} {h:>4}  {dm:>+10.4f} {da:>+10.4f}"
                          f"{' ✓' if dm < 0 else ''}")


def dump_results_json(results: dict, out_dir: Path | str | None,
                      filename: str) -> None:
    if out_dir is None:
        return
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / filename, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nSaved → {out_dir}/{filename}')
