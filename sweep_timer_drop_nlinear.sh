#!/usr/bin/env bash
# Timer drop_ratio ablation with NLinear ref (current dlinear_causal.py).
# bs=64, lr=3e-4, e=1, ref_epochs=5, seed=42 (pretrain + eval).
# Reuses baseline ckpt (drop-independent).
set -uo pipefail

ROOT=/mnt/workspace/juyoung_ha/rho_pretrain
SWEEP=$ROOT/logs/auto_overnight
mkdir -p "$SWEEP/keti2"
SUMMARY=$SWEEP/keti2/summary.jsonl
touch "$SUMMARY"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate gracm
cd "$ROOT"

EVAL_SEEDS=(42)   # ★ seed=42 only (per user task)
EVAL_DATASETS=ETTh1,ETTh2,ETTm1,ETTm2,weather,exchange
EVAL_HORIZONS=96,192,336,720
BASELINE_SRC=$SWEEP/keti2/timer__bs64_lr3e-4_e1_d20/baseline   # reuse

run_drop () {
  local drop=$1
  local name="bs64_lr3e-4_e1_d${drop}_re5_nlinear"
  local out=$SWEEP/keti2/timer__${name}
  mkdir -p "$out"
  local marker=$out/DONE
  if [[ -f $marker ]]; then echo "[skip] $name (DONE)"; return 0; fi

  echo "============================================"
  echo "[$(date '+%H:%M:%S')] timer/$name START (drop=${drop} ref_e=5 NLinear ref)"
  echo "============================================"

  if [[ ! -f $out/baseline/best.pt ]]; then
    if [[ -f $BASELINE_SRC/best.pt ]]; then
      echo "[reuse] baseline <- $BASELINE_SRC"
      mkdir -p "$out/baseline"
      ln -sf "$BASELINE_SRC/best.pt" "$out/baseline/best.pt"
      [[ -f "$BASELINE_SRC/metrics.json" ]] && ln -sf "$BASELINE_SRC/metrics.json" "$out/baseline/metrics.json"
    else
      echo "[$(date '+%H:%M:%S')] baseline pretrain (no source)"
      mkdir -p "$out/baseline"
      python scripts/pretrain_timer.py --mode baseline --epochs 1 --batch-size 64 \
        --lr 3e-4 --seed 42 --out-dir "$out/baseline" \
        > "$out/baseline/pretrain.log" 2>&1
      [[ ! -f $out/baseline/best.pt ]] && { tail -20 "$out/baseline/pretrain.log"; return 1; }
    fi
  fi

  if [[ ! -f $out/rho/best.pt ]]; then
    echo "[$(date '+%H:%M:%S')] rho pretrain"
    mkdir -p "$out/rho"
    python scripts/pretrain_timer.py --mode rho_cm --epochs 1 --batch-size 64 \
      --lr 3e-4 --seed 42 --drop-pct "$drop" --ref-epochs 5 \
      --out-dir "$out/rho" > "$out/rho/pretrain.log" 2>&1
    [[ ! -f $out/rho/best.pt ]] && { tail -30 "$out/rho/pretrain.log"; return 1; }
  fi

  for seed in "${EVAL_SEEDS[@]}"; do
    local eval_out=$out/eval_seed${seed}
    [[ -f $eval_out/zero_shot_results.json ]] && { echo "[skip] eval seed=$seed exists"; continue; }
    mkdir -p "$eval_out"
    echo "[$(date '+%H:%M:%S')] eval seed=$seed"
    python scripts/pretrain_timer.py --mode eval_zero_shot_sweep --seed "$seed" \
      --eval-datasets "$EVAL_DATASETS" --eval-horizons "$EVAL_HORIZONS" \
      --out-dir "$eval_out" \
      --baseline-ckpt "$out/baseline/best.pt" \
      --rho-cm-ckpt   "$out/rho/best.pt" \
      > "$eval_out/eval.log" 2>&1
    [[ ! -f $eval_out/zero_shot_results.json ]] && { tail -20 "$eval_out/eval.log"; return 1; }
  done

  touch "$marker"
  local now=$(date '+%Y-%m-%dT%H:%M:%S')
  echo "{\"time\":\"$now\",\"host\":\"keti_2\",\"backbone\":\"timer\",\"name\":\"$name\",\"batch\":64,\"lr\":3e-4,\"epochs\":1,\"drop_pct\":${drop},\"ref_epochs\":5,\"task\":\"drop_ablation_nlinear_seed42\",\"status\":\"ok\"}" >> "$SUMMARY"
  echo "[$(date '+%H:%M:%S')] timer/$name DONE"
}

for d in 10 20 30 40 50; do
  run_drop "$d"
done

echo "[$(date '+%H:%M:%S')] Timer NLinear drop ablation ALL DONE"
