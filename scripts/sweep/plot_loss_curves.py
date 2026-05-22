"""TASK2-1 (v2): Training loss curves — iteration step axis, 1 epoch only.

- x-axis = iteration step (NOT epoch)
- single 1-epoch run, all 3 backbones side by side
- baseline vs SPM (RHO) overlaid per panel
- ref training (DLinear pre-training) loss is EXCLUDED from main loss curve
  (it's a separate phase that runs BEFORE the model training)
- subsamples to ~200 points per series for clean visuals (raw tqdm yields
  thousands of step lines)

Usage: python plot_loss_curves.py
Output: logs/auto_overnight/plots/loss_curves_baseline_vs_spm.png
"""
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("/mnt/workspace/juyoung_ha/rho_pretrain/logs/auto_overnight")
OUT  = ROOT / "plots"
OUT.mkdir(exist_ok=True)


# Best configs (where SPM wins by largest ΔMSE)
BACKBONES = {
    "Timer (bs=64, lr=3e-4, drop=20)": "keti2/timer__bs64_lr3e-4_e1_d20",
    "Moirai (bs=32, lr=1e-4, drop=20)": "keti1/moirai__bs32_lr1e-4_e1_d20",
    "MOMENT (bs=64, lr=1e-4, drop=20, ref_e=5)": "keti2/moment__bs64_lr1e-4_e1_d20_re5_proper",
}

TARGET_POINTS = 200    # subsample every series to roughly this count
SMOOTH_WINDOW = 30     # moving avg window in raw step units


def parse_step_loss_csv(run_phase_dir: Path) -> tuple[list[int], list[float]]:
    """Read step_loss.csv (written by patched pretrain scripts).
    File format: step,loss,kept_ratio (one row per logged step).
    Returns (steps, losses).
    """
    p = run_phase_dir / 'step_loss.csv'
    if not p.exists():
        return [], []
    steps, losses = [], []
    with open(p) as f:
        next(f, None)  # header
        for line in f:
            parts = line.strip().split(',')
            if len(parts) < 2:
                continue
            try:
                steps.append(int(parts[0]))
                losses.append(float(parts[1]))
            except ValueError:
                continue
    return steps, losses


# Keep old log-parser as fallback
def parse_main_loss(log_path: Path):
    if not log_path.exists():
        return [], []
    raw = log_path.read_text(errors='ignore').replace('\r', '\n')
    pat_main = re.compile(
        r'^epoch\s+\d+/\d+:\s+\d+%\|.+?\|\s*(\d+)/(\d+).*?loss[=:]?\s*([0-9]*\.?[0-9]+)',
        re.MULTILINE,
    )
    steps, losses = [], []
    for m in pat_main.finditer(raw):
        steps.append(int(m.group(1)))
        losses.append(float(m.group(3)))
    if not steps:
        for m in re.finditer(r'^epoch\s+(\d+):\s*loss=([0-9]*\.?[0-9]+)', raw, re.MULTILINE):
            steps.append(int(m.group(1)))
            losses.append(float(m.group(2)))
    return steps, losses


def moving_average(values, window):
    if len(values) <= window:
        return list(values)
    out = []
    s = sum(values[:window])
    out.append(s / window)
    for i in range(window, len(values)):
        s += values[i] - values[i - window]
        out.append(s / window)
    pad = [out[0]] * (window - 1)
    return pad + out


def subsample(steps, values, target_points):
    n = len(steps)
    if n <= target_points:
        return steps, values
    stride = max(1, n // target_points)
    return steps[::stride], values[::stride]


def plot_pair(label: str, run_dir_rel: str, ax):
    run = ROOT / run_dir_rel
    # Prefer step_loss.csv (dense per-step), fall back to log parsing
    s_b, l_b = parse_step_loss_csv(run / "baseline")
    s_r, l_r = parse_step_loss_csv(run / "rho")
    if not s_b:
        s_b, l_b = parse_main_loss(run / "baseline" / "pretrain.log")
    if not s_r:
        s_r, l_r = parse_main_loss(run / "rho" / "pretrain.log")

    if not s_b and not s_r:
        ax.set_title(f"{label}\n(no log data)")
        return

    # Smooth (in raw resolution) then subsample for plotting
    if s_b:
        l_b_s = moving_average(l_b, SMOOTH_WINDOW)
        s_b_p, l_b_p = subsample(s_b, l_b_s, TARGET_POINTS)
        ax.plot(s_b_p, l_b_p, label=f"baseline", color="C0", lw=1.5)
    if s_r:
        l_r_s = moving_average(l_r, SMOOTH_WINDOW)
        s_r_p, l_r_p = subsample(s_r, l_r_s, TARGET_POINTS)
        ax.plot(s_r_p, l_r_p, label=f"SPM (RHO)", color="C3", lw=1.5)

    # Force x-axis to show iteration steps (not epochs)
    max_step = max((s_b[-1] if s_b else 0), (s_r[-1] if s_r else 0))
    ax.set_xlim(0, max_step)
    ax.set_xlabel("iteration step (1 epoch)")
    ax.set_ylabel(f"loss (MA-{SMOOTH_WINDOW})")
    ax.set_title(label)
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)


fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, (label, run_rel) in zip(axes, BACKBONES.items()):
    plot_pair(label, run_rel, ax)
fig.suptitle("Training loss vs iteration step (1 epoch on full UTSD)\n"
             "lower / earlier dip = faster convergence",
             fontsize=13, y=1.04)
fig.tight_layout()
out_path = OUT / "loss_curves_baseline_vs_spm.png"
fig.savefig(out_path, dpi=150, bbox_inches='tight')
print(f"Saved {out_path}")
plt.close(fig)
