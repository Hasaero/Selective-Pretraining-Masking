#!/usr/bin/env bash
# Overnight pipeline: lr_search → pick best → match_e1 (3 seeds × 2 modes).
# Runs sequentially in foreground; designed to be invoked via nohup.
set -e

cd /mnt/workspace/juyoung_ha/rho_pretrain
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate gracm

TS_START=$(date -Iseconds)
echo "[overnight] start: $TS_START"

# 1. lr_search (3 baseline runs at 1 epoch — different lr)
echo "[overnight] phase 1: lr_search"
python scripts/sweep/run_sweep.py --grid lr_search 2>&1 \
    | tee -a logs/sweep_v1/overnight.log

# 2. Pick best lr from sweep_summary.jsonl (lowest mean_mse among baseline runs)
BEST_LR=$(python -c "
import json
rows = [json.loads(l) for l in open('logs/sweep_v1/sweep_summary.jsonl')]
ok = [r for r in rows if r.get('status')=='ok' and r.get('mode')=='baseline'
      and r.get('epochs')==1 and r.get('batch')==64]
if not ok:
    # fall back to default
    print(0.0001); exit(0)
best = min(ok, key=lambda r: r['mean_mse'])
print(best['lr'])
")
echo "[overnight] best baseline lr at e1 = $BEST_LR"

# 3. match_e1 — but use the chosen lr by editing the grid spec on the fly via a
#    small env-driven Python launcher. Keep it simple: just call the pre-defined
#    match_e1 grid (lr=1e-4) UNLESS BEST_LR differs, in which case patch.
if [ "$BEST_LR" != "0.0001" ]; then
    echo "[overnight] best lr ($BEST_LR) differs from default 1e-4 — patching grid"
    python -c "
import re
p = 'scripts/sweep/run_sweep.py'
s = open(p).read()
s2 = s.replace('(mode, 64, 1e-4, 1, seed)', f'(mode, 64, $BEST_LR, 1, seed)')
open(p, 'w').write(s2)
print('grid patched')
"
fi

echo "[overnight] phase 2: match_e1"
python scripts/sweep/run_sweep.py --grid match_e1 2>&1 \
    | tee -a logs/sweep_v1/overnight.log

# 4. Optional phase: drop_pct ablation if we still have budget
echo "[overnight] phase 3 (optional): drop_pct ablation"
python scripts/sweep/run_sweep.py --grid drop_pct 2>&1 \
    | tee -a logs/sweep_v1/overnight.log || echo "[overnight] drop_pct phase skipped/failed"

TS_END=$(date -Iseconds)
echo "[overnight] end: $TS_END"
