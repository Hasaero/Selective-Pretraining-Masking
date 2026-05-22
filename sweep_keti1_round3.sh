#!/usr/bin/env bash
# Round-3: Moirai retry of failed bs=64 with smaller bs, plus extra explorations.
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

run_pair () {
  local backbone=$1 name=$2 batch=$3 lr=$4 epochs=$5 drop=$6 ref_e=$7
  local script rho_mode eval_mode
  case $backbone in
    moirai) script=scripts/pretrain_moirai.py; rho_mode=rho_cm; eval_mode=eval_zero_shot_sweep;;
    timer)  script=scripts/pretrain_timer.py;  rho_mode=rho_cm; eval_mode=eval_zero_shot_sweep;;
    moment) script=scripts/pretrain_moment.py; rho_mode=patch_rho_cm; eval_mode=eval_sweep;;
  esac

  local out=$SWEEP/keti1/${backbone}__${name}
  mkdir -p "$out"
  local marker=$out/DONE
  if [[ -f $marker ]]; then echo "[skip] $backbone/$name (DONE)"; return 0; fi

  echo "============================================"
  echo "[$(date '+%H:%M:%S')] $backbone/$name START"
  echo "============================================"

  if [[ ! -f $out/baseline/best.pt ]]; then
    echo "[$(date '+%H:%M:%S')] baseline pretrain ..."
    mkdir -p "$out/baseline"
    python "$script" --mode baseline --epochs "$epochs" --batch-size "$batch" \
      --lr "$lr" --seed 42 --out-dir "$out/baseline" \
      > "$out/baseline/pretrain.log" 2>&1
    if [[ ! -f $out/baseline/best.pt ]]; then
      echo "[FAIL] baseline pretrain — see $out/baseline/pretrain.log"
      tail -20 "$out/baseline/pretrain.log"; return 1
    fi
  fi

  if [[ ! -f $out/rho/best.pt ]]; then
    echo "[$(date '+%H:%M:%S')] rho pretrain ..."
    mkdir -p "$out/rho"
    python "$script" --mode "$rho_mode" --epochs "$epochs" --batch-size "$batch" \
      --lr "$lr" --seed 42 --drop-pct "$drop" --ref-epochs "$ref_e" \
      --out-dir "$out/rho" \
      > "$out/rho/pretrain.log" 2>&1
    if [[ ! -f $out/rho/best.pt ]]; then
      echo "[FAIL] rho pretrain — see $out/rho/pretrain.log"
      tail -20 "$out/rho/pretrain.log"; return 1
    fi
  fi

  local base_flag=--baseline-ckpt rho_flag=--rho-cm-ckpt eval_extra=""
  if [[ $backbone == moment ]]; then rho_flag=--patch-rho-cm-ckpt; eval_extra="--probe-epochs 5"; fi

  for seed in "${EVAL_SEEDS[@]}"; do
    local eval_out=$out/eval_seed${seed}
    local eval_json=$eval_out/zero_shot_results.json
    [[ $backbone == moment ]] && eval_json=$eval_out/linear_probe_results.json
    if [[ -f $eval_json ]]; then echo "[skip] eval seed=$seed exists"; continue; fi
    mkdir -p "$eval_out"
    echo "[$(date '+%H:%M:%S')] eval seed=$seed ..."
    python "$script" --mode "$eval_mode" --seed "$seed" \
      --eval-datasets "$EVAL_DATASETS" --eval-horizons "$EVAL_HORIZONS" \
      --out-dir "$eval_out" $eval_extra \
      $base_flag "$out/baseline/best.pt" $rho_flag "$out/rho/best.pt" \
      > "$eval_out/eval.log" 2>&1
    if [[ ! -f $eval_json ]]; then
      echo "[FAIL] eval seed=$seed — see $eval_out/eval.log"
      tail -20 "$eval_out/eval.log"; return 1
    fi
  done

  touch "$marker"
  local now=$(date '+%Y-%m-%dT%H:%M:%S')
  echo "{\"time\":\"$now\",\"host\":\"keti_1\",\"backbone\":\"$backbone\",\"name\":\"$name\",\"batch\":$batch,\"lr\":$lr,\"epochs\":$epochs,\"drop_pct\":$drop,\"ref_epochs\":$ref_e,\"status\":\"ok\"}" >> "$SUMMARY"
  echo "[$(date '+%H:%M:%S')] $backbone/$name DONE"
}

# Round-3: bs=64 was OOM on Moirai (H200 140GB exceeded by packed sequence).
# Retry with bs=48 (sqrt scaling: 32→48, lr 1e-4→1.2e-4 ≈ 1e-4 still safe).
run_pair moirai bs48_lr1e-4_e1_d20  48 1e-4 1 20 3
run_pair moirai bs48_lr3e-4_e1_d20  48 3e-4 1 20 3

# Best Moirai found: bs=32 lr=1e-4 d=20 (ΔMSE -0.030). Try fine-tuning around it.
run_pair moirai bs32_lr1e-4_e1_d30  32 1e-4 1 30 3   # heavier drop
run_pair moirai bs32_lr2e-4_e1_d20  32 2e-4 1 20 3   # in-between lr
run_pair moirai bs32_lr1e-4_e1_d15  32 1e-4 1 15 3   # lighter drop

echo "[$(date '+%H:%M:%S')] keti_1 round-3 ALL DONE"
