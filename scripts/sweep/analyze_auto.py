"""Aggregate auto_overnight sweep results.

For each (host, backbone, name) run, compute:
- baseline mean MSE/MAE over (24 settings) × (3 eval seeds) = 72 numbers
- rho mean MSE/MAE same
- ΔMSE, ΔMAE
- "rho wins" iff ΔMSE < 0 AND ΔMAE < 0

Reads:
  logs/auto_overnight/keti{1,2}/{backbone}__{name}/
    eval_seed{42,43,44}/{linear_probe_results,zero_shot_results}.json

Run remotely: ssh keti_1 'python /tmp/analyze_auto.py'
"""
import json
from pathlib import Path
from collections import defaultdict


ROOT = Path("/mnt/workspace/juyoung_ha/rho_pretrain/logs/auto_overnight")


def load_eval(run_dir: Path, seed: int):
    """Returns dict {label: [{dataset,horizon,test_mse,test_mae}, ...]}"""
    eval_dir = run_dir / f"eval_seed{seed}"
    for fname in ("linear_probe_results.json", "zero_shot_results.json"):
        p = eval_dir / fname
        if p.exists():
            return json.loads(p.read_text())
    return None


def aggregate_run(run_dir: Path) -> dict | None:
    """Returns summary dict with baseline/rho per-seed and aggregated stats."""
    seeds = (42, 43, 44)
    per_seed = {}
    for s in seeds:
        per_seed[s] = load_eval(run_dir, s)

    # Find which labels present (baseline + rho_cm or patch_rho_cm)
    labels = set()
    for s, d in per_seed.items():
        if d:
            labels.update(d.keys())
    if "baseline" not in labels or len(labels) < 2:
        return None

    other_label = next(l for l in labels if l != "baseline")

    rows_per_label_seed = defaultdict(list)
    for s in seeds:
        d = per_seed[s]
        if not d:
            continue
        for label, rows in d.items():
            for r in rows:
                rows_per_label_seed[(label, s)].append(r)

    def agg(label):
        all_rows = []
        for s in seeds:
            all_rows.extend(rows_per_label_seed.get((label, s), []))
        if not all_rows:
            return None
        return {
            "n": len(all_rows),
            "mse": sum(r["test_mse"] for r in all_rows) / len(all_rows),
            "mae": sum(r["test_mae"] for r in all_rows) / len(all_rows),
        }

    base = agg("baseline")
    rho = agg(other_label)
    if base is None or rho is None:
        return None
    return {
        "run_dir": str(run_dir),
        "baseline": base,
        "rho": rho,
        "rho_label": other_label,
        "delta_mse": rho["mse"] - base["mse"],
        "delta_mae": rho["mae"] - base["mae"],
        "rho_wins": (rho["mse"] < base["mse"]) and (rho["mae"] < base["mae"]),
        "rho_wins_mse": rho["mse"] < base["mse"],
        "rho_wins_mae": rho["mae"] < base["mae"],
    }


def main():
    runs = []
    for host_dir in sorted(ROOT.glob("keti*")):
        if not host_dir.is_dir():
            continue
        for run_dir in sorted(host_dir.iterdir()):
            if not run_dir.is_dir():
                continue
            r = aggregate_run(run_dir)
            if r is None:
                # at least show it's incomplete
                runs.append({"run_dir": str(run_dir), "incomplete": True})
            else:
                runs.append(r)

    print(f"\n{'='*100}")
    print(f"{'Run':<55s} {'baseMSE':>9s} {'baseMAE':>9s} {'rhoMSE':>9s} {'rhoMAE':>9s} "
          f"{'ΔMSE':>9s} {'ΔMAE':>9s} {'WIN':>5s}")
    print("=" * 100)
    for r in runs:
        run = Path(r["run_dir"]).relative_to(ROOT)
        if r.get("incomplete"):
            print(f"{str(run):<55s}  (incomplete)")
            continue
        win = "✓" if r["rho_wins"] else ("M" if r["rho_wins_mse"] else ("A" if r["rho_wins_mae"] else "·"))
        print(
            f"{str(run):<55s} "
            f"{r['baseline']['mse']:>9.4f} {r['baseline']['mae']:>9.4f} "
            f"{r['rho']['mse']:>9.4f} {r['rho']['mae']:>9.4f} "
            f"{r['delta_mse']:>+9.4f} {r['delta_mae']:>+9.4f} "
            f"{win:>5s}"
        )

    print("\nLegend: ✓ rho wins on both MSE & MAE  |  M = MSE-only  |  A = MAE-only  |  · neither")
    wins = [r for r in runs if r.get("rho_wins")]
    print(f"\n{len(wins)}/{len([r for r in runs if not r.get('incomplete')])} "
          f"configurations have rho > baseline on BOTH MSE and MAE.")
    if wins:
        print("\nWinners:")
        for r in wins:
            run = Path(r["run_dir"]).relative_to(ROOT)
            print(f"  {run}  ΔMSE={r['delta_mse']:+.4f} ΔMAE={r['delta_mae']:+.4f}")


if __name__ == "__main__":
    main()
