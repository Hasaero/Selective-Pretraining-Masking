"""Pretraining time + SPM overhead measurement.

Parses sweep_*.log timestamps to compute:
- baseline pretrain wallclock
- ref training wallclock
- rho pretrain wallclock (includes ref forward overhead per step)
- eval wallclock per seed
- SPM overhead = (rho pretrain + ref training) − baseline pretrain

Outputs JSON + a per-config table.
"""
import json
import re
from datetime import datetime
from pathlib import Path

ROOT = Path("/mnt/workspace/juyoung_ha/rho_pretrain/logs/auto_overnight")


TS = re.compile(r'\[(\d{2}:\d{2}:\d{2})\]\s+(.*)')


def parse_log_segments(log_path: Path):
    """
    Returns dict of phase wallclocks (seconds) extracted from a sweep host log.
    Looks for stanzas like:
        [HH:MM:SS] timer/<name> START
        [HH:MM:SS] baseline pretrain ...
        [HH:MM:SS] rho pretrain ...
        [HH:MM:SS] eval seed=42 ...
        [HH:MM:SS] eval seed=43 ...
        [HH:MM:SS] eval seed=44 ...
        [HH:MM:SS] timer/<name> DONE
    """
    if not log_path.exists():
        return {}
    text = log_path.read_text(errors='ignore')
    runs = {}
    cur = None
    for line in text.splitlines():
        m = TS.match(line.strip())
        if not m:
            continue
        ts_s, msg = m.group(1), m.group(2)
        try:
            ts = datetime.strptime(ts_s, "%H:%M:%S")
        except Exception:
            continue
        if 'START' in msg:
            mname = re.search(r'(timer|moirai|moment)/([^\s]+)\s+START', msg)
            if mname:
                cur = f"{mname.group(1)}/{mname.group(2)}"
                runs.setdefault(cur, {'phases': []})
        elif cur and any(k in msg for k in ('baseline pretrain', 'rho pretrain', 'eval seed', 'reuse', 'skip')):
            runs[cur]['phases'].append((ts, msg))
        elif cur and ' DONE' in msg:
            runs[cur]['phases'].append((ts, msg))
            cur = None
    return runs


def secs_between(t1: datetime, t2: datetime) -> int:
    delta = (t2 - t1).total_seconds()
    if delta < 0:
        delta += 86400  # day rollover
    return int(delta)


def summarize_run(phases):
    """Convert list of (ts, msg) into named segment durations."""
    out = {}
    for i, (ts, msg) in enumerate(phases[:-1]):
        next_ts = phases[i + 1][0]
        dur = secs_between(ts, next_ts)
        if 'baseline pretrain' in msg:
            out['baseline_pretrain_s'] = dur
        elif 'rho pretrain' in msg:
            out['rho_pretrain_total_s'] = dur   # includes ref training inside
        elif 'eval seed=42' in msg:
            out['eval_seed42_s'] = dur
        elif 'eval seed=43' in msg:
            out['eval_seed43_s'] = dur
        elif 'eval seed=44' in msg:
            out['eval_seed44_s'] = dur
    return out


def split_ref_overhead_from_rho_log(rho_log: Path):
    """Returns ref training wallclock seconds, or None."""
    if not rho_log.exists():
        return None
    text = rho_log.read_text(errors='ignore')
    for pat in (
        r'\[Patch-RHO-CM\] ref \(.*?\) build time: ([\d.]+)\s*s',
        r'\[Patch-RHO-CM\] ref build time: ([\d.]+)\s*s',
        r'\[RHO-CM\] ref(?: \(.*?\))? built in ([\d.]+)\s*s',
        r'\[RHO-CM\] ref built in ([\d.]+)\s*s',
    ):
        m = re.search(pat, text)
        if m:
            return float(m.group(1))
    return None


def main():
    all_records = []
    for host_log in ('keti1.log', 'keti2.log'):
        log_path = ROOT / host_log
        host = host_log.split('.')[0]
        runs = parse_log_segments(log_path)
        for run_name, info in runs.items():
            backbone, name = run_name.split('/', 1)
            durs = summarize_run(info['phases'])
            run_dir = ROOT / f"{host[:-1]}_{host[-1]}/{backbone}__{name}"
            # try alternative dir layout if needed
            if not run_dir.exists():
                run_dir = ROOT / f"keti{host[-1]}/{backbone}__{name}"
            rho_log = run_dir / 'rho' / 'pretrain.log'
            ref_train_s = split_ref_overhead_from_rho_log(rho_log)
            rec = {
                'host': host,
                'backbone': backbone,
                'name': name,
                **durs,
                'ref_train_s': ref_train_s,
            }
            if 'rho_pretrain_total_s' in rec and ref_train_s:
                rec['rho_model_train_s'] = max(rec['rho_pretrain_total_s'] - int(ref_train_s), 0)
            if 'baseline_pretrain_s' in rec and 'rho_pretrain_total_s' in rec:
                rec['spm_overhead_s'] = rec['rho_pretrain_total_s'] - rec['baseline_pretrain_s']
                rec['spm_overhead_pct'] = round(100 * rec['spm_overhead_s'] / max(rec['baseline_pretrain_s'], 1), 1)
            all_records.append(rec)

    out_json = ROOT / 'plots' / 'timing_analysis.json'
    out_json.parent.mkdir(exist_ok=True)
    out_json.write_text(json.dumps(all_records, indent=2))
    print(f"Wrote {out_json} ({len(all_records)} runs)")

    # Pretty-print summary table
    print(f"\n{'='*120}")
    print(f"{'backbone/name':<60s} {'baseline':>10} {'rho_total':>10} {'ref':>8} {'overhead':>10} {'%':>6}")
    print('='*120)
    def cell(v, suffix='s'):
        if v is None: return '-'
        if isinstance(v, float): return f"{v:.0f}{suffix}"
        if isinstance(v, int): return f"{v}{suffix}"
        return str(v)
    for r in all_records:
        b   = cell(r.get('baseline_pretrain_s'))
        rt  = cell(r.get('rho_pretrain_total_s'))
        ref = cell(r.get('ref_train_s'))
        ov  = cell(r.get('spm_overhead_s'))
        pct = cell(r.get('spm_overhead_pct'), suffix='%')
        print(f"{r['backbone']+'/'+r['name']:<60s} {b:>10} {rt:>10} {ref:>8} {ov:>10} {pct:>6}")


if __name__ == "__main__":
    main()
