#!/usr/bin/env bash
# TASK1 Step A on keti_1 — Timer drop_pct ablation half (10, 15, 20, 25, 30).
# Splits 9-config ablation across both hosts to halve wallclock.
# All paired (baseline+rho), full UTSD 1 epoch, bs=64, lr=3e-4, ref_epochs=5.
set -uo pipefail

ROOT=/mnt/workspace/juyoung_ha/rho_pretrain
SWEEP=$ROOT/logs/auto_overnight
mkdir -p "$SWEEP/keti1"
SUMMARY=$SWEEP/keti1/summary.jsonl
touch "$SUMMARY"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate gracm
cd "$ROOT"

EVAL_SEEDS=(42 43 44)
EVAL_DATASETS=ETTh1,ETTh2,ETTm1,ETTm2,weather,exchange
EVAL_HORIZONS=96,192,336,720

# Use the bs=64 lr=3e-4 baseline (already trained on keti_2). NFS-shared,
# so keti_1 can read it directly. baseline is identical regardless of drop_pct.
BASELINE_SRC=$SWEEP/keti2/timer__bs64_lr3e-4_e1_d20/baseline

run_pair_drop () {
  local drop=$1
  local name="bs64_lr3e-4_e1_d${drop}_re5"
  local out=$SWEEP/keti1/timer__${name}
  mkdir -p "$out"
  local marker=$out/DONE
  if [[ -f $marker ]]; then echo "[skip] timer/$name (DONE)"; return 0; fi

  # Also skip if a complete run exists on keti_2 (NFS-shared)
  local sister=$SWEEP/keti2/timer__${name}
  if [[ -f $sister/DONE ]]; then
    echo "[skip] timer/$name done on keti_2 (sister run); symlinking"
    rmdir "$out" 2>/dev/null || true
    ln -sfn "$sister" "$out"
    return 0
  fi

  echo "============================================"
  echo "[$(date '+%H:%M:%S')] timer/$name START (drop=${drop} ref_e=5) on keti_1"
  echo "============================================"

  if [[ ! -f $out/baseline/best.pt ]]; then
    if [[ -f $BASELINE_SRC/best.pt ]]; then
      echo "[reuse] baseline ← $BASELINE_SRC"
      mkdir -p "$out/baseline"
      ln -sf "$BASELINE_SRC/best.pt" "$out/baseline/best.pt"
      [[ -f "$BASELINE_SRC/metrics.json" ]] && ln -sf "$BASELINE_SRC/metrics.json" "$out/baseline/metrics.json"
    else
      echo "[$(date '+%H:%M:%S')] baseline pretrain (no source to reuse)"
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
  echo "{\"time\":\"$now\",\"host\":\"keti_1\",\"backbone\":\"timer\",\"name\":\"$name\",\"batch\":64,\"lr\":3e-4,\"epochs\":1,\"drop_pct\":${drop},\"ref_epochs\":5,\"task\":\"task1_drop_ablation\",\"status\":\"ok\"}" >> "$SUMMARY"
  echo "[$(date '+%H:%M:%S')] timer/$name DONE"
}

# keti_1 half: drop_pct 10, 20, 30
for d in 10 20 30; do
  run_pair_drop "$d"
done

echo "[$(date '+%H:%M:%S')] keti_1 task1 Timer drop ablation HALF DONE"
