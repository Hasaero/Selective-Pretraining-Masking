"""Plot reference (DLinear) training convergence per epoch.

Shows whether the ref model actually converges (loss decreases monotonically
and flattens) for each backbone.

Sources are pretrain.log files from completed RHO runs.
"""
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("/mnt/workspace/juyoung_ha/rho_pretrain/logs/auto_overnight")
OUT  = ROOT / "plots"
OUT.mkdir(exist_ok=True)


# (display label, run dir relative to ROOT, ref-tag for log parsing, color)
RUNS = [
    ("Timer (Linear causal, d=20)",
     "keti2/timer__bs64_lr3e-4_e1_d20",
     "Ref-Causal", "C0"),
    ("Moirai (Normal NLL recon, d=20)",
     "keti1/moirai__bs32_lr1e-4_e1_d20",
     "Ref-Normal-Timepoint", "C2"),
    ("Moirai (NLinear MSE-select, d=20)",
     "keti1/moirai__bs32_lr1e-4_e1_d20_re5_nlinear",
     "Ref-MSE-Forecast", "C3"),
    ("MOMENT (Linear masked-recon, d=20)",
     "keti2/moment__bs64_lr1e-4_e1_d20_re5_proper",
     "Ref-Masked-Recon", "C1"),
]

PAT_EPOCH = re.compile(r'\bref epoch (\d+):\s+\w+=([\-0-9.eE+nan]+)')


def parse_ref_epoch_loss(log_path: Path):
    """Returns list of (epoch, loss). Handles tqdm \\r noise."""
    if not log_path.exists():
        return []
    raw = log_path.read_text(errors='ignore').replace('\r', '\n')
    out = []
    for m in PAT_EPOCH.finditer(raw):
        try:
            ep = int(m.group(1))
            val = float(m.group(2))
            out.append((ep, val))
        except ValueError:
            continue
    return out


# 2x2 grid
fig, axes = plt.subplots(2, 2, figsize=(13, 9))
axes = axes.flatten()

for ax, (label, run_rel, tag, color) in zip(axes, RUNS):
    log_path = ROOT / run_rel / "rho" / "pretrain.log"
    data = parse_ref_epoch_loss(log_path)
    if not data:
        ax.set_title(f"{label}\n(no ref training data)")
        ax.set_xlabel("ref epoch")
        ax.grid(alpha=0.3)
        continue

    eps = [d[0] for d in data]
    losses = [d[1] for d in data]

    ax.plot(eps, losses, marker='o', lw=2, color=color, label='ref loss')
    for e, v in data:
        ax.annotate(f"{v:.4g}", (e, v), textcoords="offset points",
                    xytext=(7, 8), fontsize=9)

    converged = "converged" if losses[-1] <= min(losses) * 1.05 else "NOT converged"
    diverged  = max(losses) / max(min(losses), 1e-9) > 5
    status    = "DIVERGED" if diverged else converged
    color_status = "red" if diverged else "green"

    ax.set_title(f"{label}\n[{status}] final={losses[-1]:.4g}  min={min(losses):.4g}",
                 color=color_status)
    ax.set_xlabel("ref training epoch")
    ax.set_ylabel("ref loss")
    ax.set_xticks(eps)
    # for diverged runs use log scale to see them
    if diverged:
        ax.set_yscale('log')
    ax.legend(loc='best', fontsize=9)
    ax.grid(alpha=0.3)

fig.suptitle("Reference (DLinear) training convergence by epoch\n"
             "Green = converges (loss flat); Red = diverges (loss explodes)",
             fontsize=13, y=1.0)
fig.tight_layout()
out_path = OUT / "ref_training_per_epoch.png"
fig.savefig(out_path, dpi=150, bbox_inches='tight')
print(f"Saved {out_path}")
plt.close(fig)
