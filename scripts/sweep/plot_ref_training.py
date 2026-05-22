"""Plot reference model training convergence.

If ref_step_loss.csv exists in {run}/rho/ → plot per-iteration step.
Otherwise fall back to per-epoch loss parsed from pretrain.log.
"""
import re
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("/mnt/workspace/juyoung_ha/rho_pretrain/logs/auto_overnight")
OUT  = ROOT / "plots"
OUT.mkdir(exist_ok=True)

# Show recent best RHO runs
RUNS = {
    "Timer (NLinear, drop=10)": "keti2/timer__bs64_lr3e-4_e1_d10_re5_nlinear",
    "Moirai (NLinear MSE-select, drop=20)": "keti1/moirai__bs32_lr1e-4_e1_d20_re5_nlinear",
    "MOMENT (NLinear, realtime, drop=20)": "keti2/moment__bs64_lr1e-4_e1_d20_re5_proper",  # uses Linear ref
}

# pattern: '  ref epoch K: <metric>=<value>  (... )?'
PAT = re.compile(r'^\s*ref epoch (\d+):\s+(\w+)=([0-9eE.\-+nan]+)', re.MULTILINE)


def parse_ref_loss(log_path: Path):
    """Returns list of (epoch, metric_name, loss)."""
    if not log_path.exists():
        return []
    raw = log_path.read_text(errors='ignore').replace('\r', '\n')
    out = []
    for m in PAT.finditer(raw):
        try:
            ep = int(m.group(1))
            name = m.group(2)
            val = float(m.group(3))
            out.append((ep, name, val))
        except ValueError:
            continue
    return out


def parse_step_csv(csv_path: Path):
    if not csv_path.exists():
        return [], []
    steps, losses = [], []
    with open(csv_path) as f:
        next(f, None)
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


def smooth(values, window=20):
    if len(values) <= window:
        return values
    out = []
    s = sum(values[:window])
    out.append(s / window)
    for i in range(window, len(values)):
        s += values[i] - values[i - window]
        out.append(s / window)
    pad = [out[0]] * (window - 1)
    return pad + out


fig, axes = plt.subplots(1, len(RUNS), figsize=(6*len(RUNS), 4.5))
if len(RUNS) == 1: axes = [axes]
for ax, (label, run_rel) in zip(axes, RUNS.items()):
    csv_path = ROOT / run_rel / "rho" / "ref_step_loss.csv"
    steps, losses = parse_step_csv(csv_path)
    if steps:
        # iteration-step plot
        losses_s = smooth(losses, window=20)
        ax.plot(steps, losses_s, lw=1.6, color='C2', label='ref MA-20')
        ax.plot(steps, losses, lw=0.4, color='C2', alpha=0.3, label='ref raw')
        ax.set_xlabel('ref training iteration step')
        ax.set_ylabel('ref loss')
        ax.set_title(f'{label}\n(step-level, {len(steps)} pts)')
    else:
        # fallback: per-epoch
        data = parse_ref_loss(ROOT / run_rel / "rho" / "pretrain.log")
        if not data:
            ax.set_title(f"{label}\n(no ref data)")
            ax.set_xlabel("step")
            continue
        eps = [d[0] for d in data]
        losses = [d[2] for d in data]
        metric = data[0][1]
        ax.plot(eps, losses, marker='o', lw=2, color='C2', label=metric)
        for e, _, v in data:
            ax.annotate(f"{v:.3f}", (e, v), textcoords='offset points',
                        xytext=(5, 8), fontsize=8)
        ax.set_xlabel('ref epoch (step-level not available)')
        ax.set_ylabel(f'ref loss ({metric})')
        ax.set_xticks(eps)
        ax.set_title(label)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

fig.suptitle("Reference (DLinear) training convergence",
             fontsize=13, y=1.03)
fig.tight_layout()
out_path = OUT / "ref_training_convergence.png"
fig.savefig(out_path, dpi=150, bbox_inches='tight')
print(f"Saved {out_path}")
plt.close(fig)
