#!/usr/bin/env bash
# sweep_keti1.sh — runs Moirai paired (baseline+rho) sweep on keti_1.
# Self-contained, idempotent. Resumes by skipping any run whose summary.json exists.
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
  # Args: backbone name batch lr epochs drop_pct ref_epochs
  local backbone=$1 name=$2 batch=$3 lr=$4 epochs=$5 drop=$6 ref_e=$7
  local script
  case $backbone in
    moirai) script=scripts/pretrain_moirai.py; rho_mode=rho_cm; eval_mode=eval_zero_shot_sweep;;
    timer)  script=scripts/pretrain_timer.py;  rho_mode=rho_cm; eval_mode=eval_zero_shot_sweep;;
    moment) script=scripts/pretrain_moment.py; rho_mode=patch_rho_cm; eval_mode=eval_sweep;;
    *) echo "unknown backbone $backbone"; return 2;;
  esac

  local out=$SWEEP/keti1/${backbone}__${name}
  mkdir -p "$out"
  local marker=$out/DONE
  if [[ -f $marker ]]; then
    echo "[skip] $backbone/$name (DONE)"
    return 0
  fi

  echo "============================================"
  echo "[$(date '+%H:%M:%S')] $backbone/$name START"
  echo "============================================"

  # 1) baseline pretrain (skip if best.pt exists)
  if [[ ! -f $out/baseline/best.pt ]]; then
    echo "[$(date '+%H:%M:%S')] baseline pretrain ..."
    mkdir -p "$out/baseline"
    python "$script" --mode baseline --epochs "$epochs" --batch-size "$batch" \
      --lr "$lr" --seed 42 --out-dir "$out/baseline" \
      > "$out/baseline/pretrain.log" 2>&1
    if [[ ! -f $out/baseline/best.pt ]]; then
      echo "[FAIL] baseline pretrain — see $out/baseline/pretrain.log"
      tail -20 "$out/baseline/pretrain.log"
      return 1
    fi
  else
    echo "[skip] baseline best.pt exists"
  fi

  # 2) rho pretrain (skip if best.pt exists)
  if [[ ! -f $out/rho/best.pt ]]; then
    echo "[$(date '+%H:%M:%S')] rho pretrain ..."
    mkdir -p "$out/rho"
    python "$script" --mode "$rho_mode" --epochs "$epochs" --batch-size "$batch" \
      --lr "$lr" --seed 42 --drop-pct "$drop" --ref-epochs "$ref_e" \
      --out-dir "$out/rho" \
      > "$out/rho/pretrain.log" 2>&1
    if [[ ! -f $out/rho/best.pt ]]; then
      echo "[FAIL] rho pretrain — see $out/rho/pretrain.log"
      tail -20 "$out/rho/pretrain.log"
      return 1
    fi
  else
    echo "[skip] rho best.pt exists"
  fi

  # 3) eval at 3 seeds (rho needs different ckpt flag for moment)
  local base_flag rho_flag
  if [[ $backbone == moment ]]; then
    base_flag=--baseline-ckpt; rho_flag=--patch-rho-cm-ckpt
    eval_extra="--probe-epochs 5"
  else
    base_flag=--baseline-ckpt; rho_flag=--rho-cm-ckpt
    eval_extra=""
  fi

  for seed in "${EVAL_SEEDS[@]}"; do
    local eval_out=$out/eval_seed${seed}
    local eval_json
    if [[ $backbone == moment ]]; then
      eval_json=$eval_out/linear_probe_results.json
    else
      eval_json=$eval_out/zero_shot_results.json
    fi
    if [[ -f $eval_json ]]; then
      echo "[skip] eval seed=$seed exists"
      continue
    fi
    mkdir -p "$eval_out"
    echo "[$(date '+%H:%M:%S')] eval seed=$seed ..."
    python "$script" --mode "$eval_mode" --seed "$seed" \
      --eval-datasets "$EVAL_DATASETS" --eval-horizons "$EVAL_HORIZONS" \
      --out-dir "$eval_out" $eval_extra \
      $base_flag "$out/baseline/best.pt" $rho_flag "$out/rho/best.pt" \
      > "$eval_out/eval.log" 2>&1
    if [[ ! -f $eval_json ]]; then
      echo "[FAIL] eval seed=$seed — see $eval_out/eval.log"
      tail -20 "$eval_out/eval.log"
      return 1
    fi
  done

  touch "$marker"
  local now=$(date '+%Y-%m-%dT%H:%M:%S')
  echo "{\"time\":\"$now\",\"host\":\"keti_1\",\"backbone\":\"$backbone\",\"name\":\"$name\",\"batch\":$batch,\"lr\":$lr,\"epochs\":$epochs,\"drop_pct\":$drop,\"ref_epochs\":$ref_e,\"status\":\"ok\"}" >> "$SUMMARY"
  echo "[$(date '+%H:%M:%S')] $backbone/$name DONE"
}

# ------------------------------------------------------------
# Configs (run sequentially) — keti_1 takes Moirai (slowest)
# ------------------------------------------------------------

# best from prior sweep was lr=1e-4 e=2 d=10. With full UTSD we run e=1.
run_pair moirai bs4_lr1e-4_e1_d10  4 1e-4 1 10 3
run_pair moirai bs4_lr1e-4_e1_d20  4 1e-4 1 20 3
# Optionally try lower lr that the prior sweep didn't try
run_pair moirai bs8_lr1e-4_e1_d10  8 1e-4 1 10 3
run_pair moirai bs8_lr1e-4_e1_d20  8 1e-4 1 20 3

# Round 2: bs scaling — Timer found that bs=64 wins with lr=3e-4. Try same regime on Moirai.
# (Moirai bs scaling: lr ~ sqrt(bs) — bs=4→1e-4 implies bs=32→3e-4, bs=64→4e-4.)
run_pair moirai bs32_lr3e-4_e1_d20  32 3e-4 1 20 3
run_pair moirai bs32_lr3e-4_e1_d10  32 3e-4 1 10 3
run_pair moirai bs32_lr1e-4_e1_d20  32 1e-4 1 20 3
run_pair moirai bs64_lr3e-4_e1_d20  64 3e-4 1 20 3

echo "[$(date '+%H:%M:%S')] keti_1 sweep ALL DONE"
