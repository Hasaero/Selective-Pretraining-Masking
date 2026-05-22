#!/usr/bin/env python3
"""Post-process a sweep run dir: emit step-loss plot, kept-ratio plot, overhead summary.

Usage: python postprocess_run.py <run_dir>
  Expects: <run_dir>/baseline/{step_loss.csv,pretrain.log}
           <run_dir>/rho/{step_loss.csv,pretrain.log}

Outputs:
  <run_dir>/plots/step_loss.png
  <run_dir>/plots/kept_ratio.png
  <run_dir>/plots/overhead.txt
"""
import sys, re, csv
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

run_dir = Path(sys.argv[1])
plot_dir = run_dir / 'plots'
plot_dir.mkdir(exist_ok=True)


def load_csv(p):
    if not p.exists():
        return None
    steps, losses, keeps = [], [], []
    with open(p) as f:
        rdr = csv.DictReader(f)
        for r in rdr:
            steps.append(int(r['step']))
            losses.append(float(r['loss']))
            keeps.append(float(r['kept_ratio']))
    return np.array(steps), np.array(losses), np.array(keeps)


def parse_overhead(log_path):
    """Returns dict: ref_build_s, train_total_s, eval_total_s, total_s"""
    if not log_path.exists():
        return {}
    txt = log_path.read_text(errors='ignore')
    out = {}
    m = re.search(r'\[RHO-CM\] ref built in ([\d.]+)s', txt)
    if m: out['ref_build_s'] = float(m.group(1))
    # First and last timestamp lines (or use file mtime)
    return out


def fmt_time(s):
    if s is None: return 'N/A'
    if s < 60: return f'{s:.1f}s'
    if s < 3600: return f'{s/60:.1f}min'
    return f'{s/3600:.2f}h'


# ─── Step-loss plot ───
fig, ax = plt.subplots(1, 1, figsize=(9, 5))
for label, sub, color in [('baseline', 'baseline', '#888888'),
                          ('rho_cm',   'rho',      '#d62728')]:
    csv_p = run_dir / sub / 'step_loss.csv'
    res = load_csv(csv_p)
    if res is None:
        continue
    steps, losses, _ = res
    # Smooth with rolling mean for readability
    if len(losses) > 50:
        win = max(5, len(losses) // 50)
        ker = np.ones(win) / win
        losses_s = np.convolve(losses, ker, mode='valid')
        steps_s = steps[win-1:]
        ax.plot(steps_s, losses_s, color=color, label=f'{label} (smoothed)', linewidth=2)
        ax.plot(steps, losses, color=color, alpha=0.2, linewidth=0.5)
    else:
        ax.plot(steps, losses, color=color, label=label, linewidth=2)

ax.set_xlabel('Training step')
ax.set_ylabel('Train loss (MSE)')
ax.set_title(f'Step-wise loss convergence — {run_dir.name}')
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(plot_dir / 'step_loss.png', dpi=150, bbox_inches='tight')
plt.close(fig)

# ─── Kept-ratio plot (rho only) ───
csv_p = run_dir / 'rho' / 'step_loss.csv'
res = load_csv(csv_p)
if res is not None:
    steps, _, keeps = res
    fig, ax = plt.subplots(1, 1, figsize=(9, 4))
    ax.plot(steps, keeps, color='#2ca02c', linewidth=1, alpha=0.5)
    if len(keeps) > 50:
        win = max(5, len(keeps) // 50)
        ker = np.ones(win) / win
        keeps_s = np.convolve(keeps, ker, mode='valid')
        ax.plot(steps[win-1:], keeps_s, color='#2ca02c', linewidth=2, label='smoothed')
    ax.axhline(y=keeps.mean(), color='k', linestyle='--', alpha=0.5,
               label=f'mean={keeps.mean():.3f}')
    ax.set_xlabel('Training step')
    ax.set_ylabel('Kept ratio (1 − drop_pct/100, after token selection)')
    ax.set_title(f'Token selection over training — {run_dir.name}')
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(plot_dir / 'kept_ratio.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

# ─── Overhead summary ───
oh = {}
oh['baseline'] = parse_overhead(run_dir / 'baseline' / 'pretrain.log')
oh['rho']      = parse_overhead(run_dir / 'rho' / 'pretrain.log')

# File mtimes for total wall time
def wall(p):
    pre = p / 'pretrain.log'
    if not pre.exists(): return None
    # crude: file_size_grow proxy is not reliable; use mtime - ctime
    import os
    s = os.stat(pre)
    return s.st_mtime - s.st_ctime

with open(plot_dir / 'overhead.txt', 'w') as f:
    f.write(f'=== Overhead summary: {run_dir.name} ===\n\n')
    for stage in ('baseline', 'rho'):
        f.write(f'[{stage}]\n')
        d = oh[stage]
        f.write(f'  ref_build:   {fmt_time(d.get("ref_build_s"))}\n')
        w = wall(run_dir / stage)
        f.write(f'  pretrain wallclock (mtime-ctime, approx): {fmt_time(w)}\n')
        f.write('\n')
    eval_log = run_dir / 'eval_seed42' / 'eval.log'
    if eval_log.exists():
        import os
        s = os.stat(eval_log)
        f.write(f'[eval]\n  wallclock (approx): {fmt_time(s.st_mtime - s.st_ctime)}\n')
    f.write('\n')
    # If kept_ratio CSV exists, summary stats
    res = load_csv(run_dir / 'rho' / 'step_loss.csv')
    if res is not None:
        _, _, keeps = res
        f.write(f'[selection]\n')
        f.write(f'  mean kept_ratio: {keeps.mean():.4f}  '
                f'(target: {1.0 - (float(re.search(r"_d(\d+)", run_dir.name).group(1)) if re.search(r"_d(\d+)", run_dir.name) else 0)/100:.2f})\n')
        f.write(f'  std kept_ratio:  {keeps.std():.4f}\n')

print(f'Wrote: {plot_dir}/step_loss.png, kept_ratio.png, overhead.txt')
