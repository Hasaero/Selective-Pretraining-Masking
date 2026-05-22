"""Generic eval sweep — runs `eval_fn` across (ckpt × dataset × horizon)."""
from pathlib import Path
from typing import Callable

from .reporting import print_results_table, print_delta_tables, dump_results_json


def run_eval_sweep(
    eval_fn: Callable[..., dict],
    ckpt_paths: dict,                        # {label: Path | None}
    datasets: list,
    horizons: list,
    *,
    title: str = 'RESULTS',
    out_filename: str = 'results.json',
    out_dir: Path | str | None = None,
    baseline_label: str = 'baseline',
    **eval_kwargs,
) -> dict:
    """`eval_fn` must accept `(ckpt_path, dataset_name, horizon, **eval_kwargs)`
    and return `{'dataset', 'horizon', 'test_mse', 'test_mae'}`. Failures
    print 'SKIP' and continue."""
    results: dict[str, list[dict]] = {}
    for label, ckpt_path in ckpt_paths.items():
        print(f'\n{"="*60}\nEvaluating: {label}\n{"="*60}')
        results[label] = []
        for ds in datasets:
            for h in horizons:
                try:
                    r = eval_fn(ckpt_path, ds, h, **eval_kwargs)
                    results[label].append(r)
                except Exception as e:
                    print(f'  SKIP {ds} H={h}: {e}')

    print_results_table(results, datasets, horizons, title=title)
    print_delta_tables(results, datasets, horizons, baseline_label=baseline_label)
    dump_results_json(results, out_dir, out_filename)
    return results
