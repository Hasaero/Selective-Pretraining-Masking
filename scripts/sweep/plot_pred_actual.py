"""TASK2-2: Prediction-actual plots on eval set.

For each backbone's BEST baseline + best RHO checkpoint, run inference on a
few eval batches and plot pred vs actual side-by-side. Highlights cases
where SPM (RHO) is visibly better than baseline.

Picks:
- Timer  : bs=64 lr=3e-4 e=1 d=20 ref_e=5  (ΔMSE -0.067)
- Moirai : bs=32 lr=1e-4 e=1 d=20 (legacy normal ref, our best)  (ΔMSE -0.030)
- MOMENT : bs=64 lr=1e-4 e=1 d=20 ref_e=5 NEW realtime ref      (ΔMSE -0.003)

For each: pick a long-horizon dataset where SPM helps a lot
- Timer  → ETTm1 H=720  (ΔMSE -0.10 in our breakdown)
- Moirai → ETTh1 H=720  (ΔMSE -0.078)
- MOMENT → ETTh1 H=720  (ΔMSE -0.047)

Plot 4 sample windows per (backbone, dataset, horizon), one panel each:
  - Past context (faint)
  - Actual future (solid black)
  - Baseline pred (dashed blue)
  - RHO pred     (dashed red)

Saved to plots/pred_actual_<backbone>.png
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

PROJECT_ROOT = Path("/mnt/workspace/juyoung_ha/rho_pretrain")
sys.path.insert(0, str(PROJECT_ROOT))

OUT = PROJECT_ROOT / "logs/auto_overnight/plots"
OUT.mkdir(parents=True, exist_ok=True)


def plot_one_backbone(name, plot_fn, baseline_ckpt, rho_ckpt, dataset_name,
                      horizon, n_samples=4, channel=0, scan_pool=80):
    """plot_fn(ckpt_path, dataset_name, horizon, n_samples) → list of dicts:
       [{past, actual, pred}, ...] each (T_ctx,) or (horizon,) numpy arrays.

    Picks the n_samples test windows where SPM beats baseline by the largest
    MSE margin (out of `scan_pool` candidates from the head of the test set).
    """
    print(f"\n[{name}] {dataset_name} H={horizon} ch={channel}")
    base_pool = plot_fn(baseline_ckpt, dataset_name, horizon, scan_pool, channel)
    rho_pool  = plot_fn(rho_ckpt, dataset_name, horizon, scan_pool, channel)

    # Compute per-sample MSE diff (baseline - rho); large positive = SPM wins by a lot
    margins = []
    for i, (b, r) in enumerate(zip(base_pool, rho_pool)):
        b_mse = ((b["pred"] - b["actual"]) ** 2).mean()
        r_mse = ((r["pred"] - r["actual"]) ** 2).mean()
        margins.append((b_mse - r_mse, i, b_mse, r_mse))
    # Take top-n by margin
    margins.sort(key=lambda t: -t[0])
    top = margins[:n_samples]
    base_results = [base_pool[i] for _, i, _, _ in top]
    rho_results  = [rho_pool[i]  for _, i, _, _ in top]
    print(f"  top {n_samples} margins (b_mse - r_mse): "
          + ", ".join(f"#{i}: +{m:.3f}" for m, i, _, _ in top))

    fig, axes = plt.subplots(1, n_samples, figsize=(5*n_samples, 4))
    if n_samples == 1: axes = [axes]
    for i, (b, r) in enumerate(zip(base_results, rho_results)):
        ax = axes[i]
        actual = b["actual"]
        T_pred = len(actual)
        x_pred = np.arange(T_pred)   # 0..horizon-1 (forecasting region only)

        ax.plot(x_pred, actual, color="black", lw=1.5, label="actual")
        ax.plot(x_pred, b["pred"], color="C0", lw=1.3,
                label=f"baseline (MSE={((b['pred']-actual)**2).mean():.3f})")
        ax.plot(x_pred, r["pred"], color="C3", lw=1.3,
                label=f"SPM (MSE={((r['pred']-actual)**2).mean():.3f})")
        ax.set_title(f"sample {i}")
        ax.set_xlabel("forecast step")
        ax.legend(fontsize=8, loc="best")
        ax.grid(alpha=0.3)

    fig.suptitle(f"{name} — {dataset_name} H={horizon} ch={channel}\n"
                 f"baseline = {Path(baseline_ckpt).parent.parent.name} | "
                 f"SPM = {Path(rho_ckpt).parent.parent.name}",
                 fontsize=11, y=1.02)
    fig.tight_layout()
    out_path = OUT / f"pred_actual_{name.lower()}_{dataset_name}_h{horizon}.png"
    fig.savefig(out_path, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f"  saved {out_path}")
    return out_path


# ------------------ Timer inference ------------------

def timer_infer(ckpt_path, dataset_name, horizon, n_samples, channel):
    from scripts.pretrain_timer import (
        TimerModel, timer_zero_shot_predict,
        prepare_forecast_datasets, DATA_DIR, SEQ_LEN,
    )
    device = torch.device('cuda')
    torch.manual_seed(42); np.random.seed(42)
    _, _, test_ds = prepare_forecast_datasets(dataset_name, DATA_DIR, SEQ_LEN, horizon, stride=1)
    model = TimerModel().to(device)
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt['state_dict'], strict=True)
    model.eval()

    # take first n_samples (deterministic)
    out = []
    with torch.no_grad():
        for i in range(n_samples):
            x, y, *_ = test_ds[i]              # x: (C, ctx), y: (C, horizon)
            x = x.unsqueeze(0).to(device)       # (1, C, ctx)
            y = y.unsqueeze(0).to(device)       # (1, C, horizon)
            past = x.permute(0, 2, 1).float()   # (1, ctx, C)
            pred = timer_zero_shot_predict(model, past, horizon, device)  # (1, horizon, C)
            pred_bc = pred.permute(0, 2, 1)     # (1, C, horizon)

            out.append({
                "past":   x[0, channel].cpu().numpy(),
                "actual": y[0, channel].cpu().numpy(),
                "pred":   pred_bc[0, channel].cpu().numpy(),
            })
    return out


# ------------------ MOMENT inference (linear probe) ------------------

def moment_infer(ckpt_path, dataset_name, horizon, n_samples, channel):
    """MOMENT linear probe — reuse pretrain_moment's helpers."""
    from scripts.pretrain_moment import (
        load_moment_with_ckpt, prepare_forecast_datasets, DATA_DIR, SEQ_LEN,
    )
    from pathlib import Path
    from torch.utils.data import DataLoader

    device = torch.device('cuda')
    torch.manual_seed(42); np.random.seed(42)
    train_ds, _, test_ds = prepare_forecast_datasets(
        dataset_name, DATA_DIR, SEQ_LEN, horizon, stride=1)

    # Use the script's standard loader (task_name='reconstruction' →
    # new_task_name='forecasting' + init() flow). Trains forecast head only.
    model = load_moment_with_ckpt(Path(ckpt_path), horizon, device)

    # Probe-train forecast head (5 epochs, lr=1e-4, like eval_forecasting)
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=1e-4
    )
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=0)
    for ep in range(5):
        model.train()
        for batch in train_loader:
            x, y, _ = batch
            x = x.to(device); y = y.to(device)
            B, C, T = x.shape
            x_enc = x.reshape(B*C, 1, T)
            inp = torch.ones(B*C, T, device=device)
            out = model(x_enc=x_enc, input_mask=inp)
            pred = out.forecast.reshape(B, C, horizon)
            loss = ((pred - y) ** 2).mean()
            optimizer.zero_grad(); loss.backward(); optimizer.step()
    model.eval()

    out = []
    with torch.no_grad():
        for i in range(n_samples):
            x, y, _ = test_ds[i]
            x = x.unsqueeze(0).to(device)
            y = y.unsqueeze(0).to(device)
            B, C, T = x.shape
            x_enc = x.reshape(B*C, 1, T)
            inp = torch.ones(B*C, T, device=device)
            outputs = model(x_enc=x_enc, input_mask=inp)
            pred = outputs.forecast.reshape(B, C, horizon)
            out.append({
                "past":   x[0, channel].cpu().numpy(),
                "actual": y[0, channel].cpu().numpy(),
                "pred":   pred[0, channel].cpu().numpy(),
            })
    return out


# ------------------ Run ------------------

ROOT = PROJECT_ROOT / "logs/auto_overnight"

PLANS = [
    # (backbone, baseline_ckpt, rho_ckpt, dataset, horizon, infer_fn, channel)
    # Timer — best config bs=64 lr=3e-4 d=20
    ("Timer",
     ROOT/"keti2/timer__bs64_lr3e-4_e1_d20/baseline/best.pt",
     ROOT/"keti2/timer__bs64_lr3e-4_e1_d20/rho/best.pt",
     "ETTm1", 96, timer_infer, 0),
    ("Timer",
     ROOT/"keti2/timer__bs64_lr3e-4_e1_d20/baseline/best.pt",
     ROOT/"keti2/timer__bs64_lr3e-4_e1_d20/rho/best.pt",
     "ETTm1", 720, timer_infer, 0),
    ("Timer",
     ROOT/"keti2/timer__bs64_lr3e-4_e1_d20/baseline/best.pt",
     ROOT/"keti2/timer__bs64_lr3e-4_e1_d20/rho/best.pt",
     "weather", 96, timer_infer, 0),
    ("Timer",
     ROOT/"keti2/timer__bs64_lr3e-4_e1_d20/baseline/best.pt",
     ROOT/"keti2/timer__bs64_lr3e-4_e1_d20/rho/best.pt",
     "weather", 336, timer_infer, 0),
    # MOMENT — best NEW realtime ref bs=64 lr=1e-4 d=20 ref_e=5
    ("MOMENT",
     ROOT/"keti2/moment__bs64_lr1e-4_e1_d20_re5_proper/baseline/best.pt",
     ROOT/"keti2/moment__bs64_lr1e-4_e1_d20_re5_proper/rho/best.pt",
     "ETTh1", 96, moment_infer, 0),
    ("MOMENT",
     ROOT/"keti2/moment__bs64_lr1e-4_e1_d20_re5_proper/baseline/best.pt",
     ROOT/"keti2/moment__bs64_lr1e-4_e1_d20_re5_proper/rho/best.pt",
     "ETTh1", 720, moment_infer, 0),
]


def main():
    for backbone, base, rho, ds, h, fn, ch in PLANS:
        try:
            plot_one_backbone(backbone, fn, str(base), str(rho), ds, h, n_samples=4, channel=ch)
        except Exception as e:
            print(f"[ERROR] {backbone}/{ds}/H={h}: {e}")
            import traceback; traceback.print_exc()


if __name__ == "__main__":
    main()
