"""Per-(dataset, horizon) breakdown for best Timer & Moirai configs."""
import json
from pathlib import Path
from collections import defaultdict

ROOT = Path("/mnt/workspace/juyoung_ha/rho_pretrain/logs/auto_overnight")

BEST = [
    ("Timer-base bs=64 lr=3e-4 d=20",
     ROOT / "keti2/timer__bs64_lr3e-4_e1_d20"),
    ("Moirai-small bs=32 lr=1e-4 d=20",
     ROOT / "keti1/moirai__bs32_lr1e-4_e1_d20"),
]

DATASETS = ["ETTh1", "ETTh2", "ETTm1", "ETTm2", "weather", "exchange"]
HORIZONS = [96, 192, 336, 720]
SEEDS = (42, 43, 44)


def load_seed(run_dir, seed):
    for fname in ("zero_shot_results.json", "linear_probe_results.json"):
        p = run_dir / f"eval_seed{seed}" / fname
        if p.exists():
            return json.loads(p.read_text())
    return None


def avg_per_setting(run_dir, label):
    """Returns {(ds, h): (mse, mae)} averaged across 3 seeds."""
    sums = defaultdict(lambda: [0.0, 0.0, 0])
    for s in SEEDS:
        d = load_seed(run_dir, s)
        if not d or label not in d:
            continue
        for r in d[label]:
            key = (r["dataset"], r["horizon"])
            sums[key][0] += r["test_mse"]
            sums[key][1] += r["test_mae"]
            sums[key][2] += 1
    return {k: (v[0]/v[2], v[1]/v[2]) for k, v in sums.items() if v[2] > 0}


for name, run in BEST:
    print(f"\n{'='*100}")
    print(f"{name}")
    print(f"  {run}")
    print(f"{'='*100}")

    base = avg_per_setting(run, "baseline")
    rho_label = "rho_cm" if "moirai" in str(run) or "timer" in str(run) else "patch_rho_cm"
    rho = avg_per_setting(run, rho_label)

    print(f"{'Dataset':<10s} {'H':>4s} | {'baseMSE':>9s} {'baseMAE':>9s} | "
          f"{'rhoMSE':>9s} {'rhoMAE':>9s} | {'ΔMSE':>9s} {'ΔMAE':>9s}  WIN")
    print("-" * 100)

    sum_b_mse = sum_b_mae = sum_r_mse = sum_r_mae = 0.0
    n_win = n_total = 0
    for ds in DATASETS:
        for h in HORIZONS:
            k = (ds, h)
            if k not in base or k not in rho:
                print(f"{ds:<10s} {h:>4d} |   missing")
                continue
            bM, bA = base[k]
            rM, rA = rho[k]
            dM, dA = rM - bM, rA - bA
            win = "✓" if (dM < 0 and dA < 0) else ("M" if dM < 0 else ("A" if dA < 0 else "·"))
            print(f"{ds:<10s} {h:>4d} | {bM:>9.4f} {bA:>9.4f} | "
                  f"{rM:>9.4f} {rA:>9.4f} | {dM:>+9.4f} {dA:>+9.4f}  {win}")
            sum_b_mse += bM; sum_b_mae += bA; sum_r_mse += rM; sum_r_mae += rA
            n_total += 1
            if dM < 0 and dA < 0:
                n_win += 1

    if n_total:
        print("-" * 100)
        print(f"{'MEAN':<10s} {'':>4s} | {sum_b_mse/n_total:>9.4f} {sum_b_mae/n_total:>9.4f} | "
              f"{sum_r_mse/n_total:>9.4f} {sum_r_mae/n_total:>9.4f} | "
              f"{(sum_r_mse-sum_b_mse)/n_total:>+9.4f} {(sum_r_mae-sum_b_mae)/n_total:>+9.4f}")
        print(f"\nrho wins on BOTH MSE&MAE: {n_win}/{n_total}")

    # per-dataset means
    print("\nper-dataset (averaged across 4 horizons × 3 seeds):")
    for ds in DATASETS:
        bM = bA = rM = rA = 0.0; nh = 0
        for h in HORIZONS:
            k = (ds, h)
            if k in base and k in rho:
                bM += base[k][0]; bA += base[k][1]; rM += rho[k][0]; rA += rho[k][1]; nh += 1
        if nh:
            bM/=nh; bA/=nh; rM/=nh; rA/=nh
            dM=rM-bM; dA=rA-bA
            win = "✓" if (dM < 0 and dA < 0) else ("M" if dM < 0 else ("A" if dA < 0 else "·"))
            print(f"  {ds:<10s} | base MSE/MAE = {bM:.4f}/{bA:.4f}  →  rho = {rM:.4f}/{rA:.4f}  "
                  f"Δ = {dM:+.4f}/{dA:+.4f}  {win}")
