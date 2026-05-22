#!/usr/bin/env bash
# Round-5 keti_2: MOMENT with realtime masked-recon ref (only ref now).
# Reuses baseline ckpts from earlier rounds.
set -uo pipefail

ROOT=/mnt/workspace/juyoung_ha/rho_pretrain
SWEEP=$ROOT/logs/auto_overnight
mkdir -p "$SWEEP/keti2"
SUMMARY=$SWEEP/keti2/summary.jsonl
touch "$SUMMARY"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate gracm
cd "$ROOT"

EVAL_SEEDS=(42 43 44)
EVAL_DATASETS=ETTh1,ETTh2,ETTm1,ETTm2,weather,exchange
EVAL_HORIZONS=96,192,336,720

run_pair_mr () {
  # name batch lr epochs drop_pct ref_epochs reused_baseline_dir
  local name=$1 batch=$2 lr=$3 epochs=$4 drop=$5 ref_e=$6 baseline_src=$7
  local script=scripts/pretrain_moment.py
  local out=$SWEEP/keti2/moment__${name}
  mkdir -p "$out"
  local marker=$out/DONE
  if [[ -f $marker ]]; then echo "[skip] moment/$name (DONE)"; return 0; fi

  echo "============================================"
  echo "[$(date '+%H:%M:%S')] moment/$name START (realtime masked-recon ref)"
  echo "============================================"

  if [[ ! -f $out/baseline/best.pt ]]; then
    if [[ -n "$baseline_src" && -f "$baseline_src/best.pt" ]]; then
      echo "[reuse] baseline ← $baseline_src"
      mkdir -p "$out/baseline"
      ln -sf "$baseline_src/best.pt"     "$out/baseline/best.pt"
      [[ -f "$baseline_src/metrics.json" ]] && ln -sf "$baseline_src/metrics.json" "$out/baseline/metrics.json"
    else
      echo "[$(date '+%H:%M:%S')] baseline pretrain (no reuse)"
      mkdir -p "$out/baseline"
      python "$script" --mode baseline --epochs "$epochs" --batch-size "$batch" \
        --lr "$lr" --seed 42 --out-dir "$out/baseline" \
        > "$out/baseline/pretrain.log" 2>&1
      [[ ! -f $out/baseline/best.pt ]] && { tail -20 "$out/baseline/pretrain.log"; return 1; }
    fi
  fi

  if [[ ! -f $out/rho/best.pt ]]; then
    echo "[$(date '+%H:%M:%S')] rho pretrain"
    mkdir -p "$out/rho"
    python "$script" --mode patch_rho_cm --epochs "$epochs" --batch-size "$batch" \
      --lr "$lr" --seed 42 --drop-pct "$drop" --ref-epochs "$ref_e" \
      --out-dir "$out/rho" > "$out/rho/pretrain.log" 2>&1
    [[ ! -f $out/rho/best.pt ]] && { tail -30 "$out/rho/pretrain.log"; return 1; }
  fi

  for seed in "${EVAL_SEEDS[@]}"; do
    local eval_out=$out/eval_seed${seed}
    [[ -f $eval_out/linear_probe_results.json ]] && { echo "[skip] eval seed=$seed exists"; continue; }
    mkdir -p "$eval_out"
    echo "[$(date '+%H:%M:%S')] eval seed=$seed"
    python "$script" --mode eval_sweep --seed "$seed" \
      --eval-datasets "$EVAL_DATASETS" --eval-horizons "$EVAL_HORIZONS" \
      --out-dir "$eval_out" --probe-epochs 5 \
      --baseline-ckpt     "$out/baseline/best.pt" \
      --patch-rho-cm-ckpt "$out/rho/best.pt" \
      > "$eval_out/eval.log" 2>&1
    [[ ! -f $eval_out/linear_probe_results.json ]] && { tail -20 "$eval_out/eval.log"; return 1; }
  done

  touch "$marker"
  local now=$(date '+%Y-%m-%dT%H:%M:%S')
  echo "{\"time\":\"$now\",\"host\":\"keti_2\",\"backbone\":\"moment\",\"name\":\"$name\",\"batch\":$batch,\"lr\":$lr,\"epochs\":$epochs,\"drop_pct\":$drop,\"ref_epochs\":$ref_e,\"ref\":\"masked_realtime\",\"status\":\"ok\"}" >> "$SUMMARY"
  echo "[$(date '+%H:%M:%S')] moment/$name DONE"
}

# Configs around closest legacy results — now with proper realtime ref.
run_pair_mr bs64_lr1e-4_e1_d10_mr   64 1e-4 1 10 3 "$SWEEP/keti2/moment__bs64_lr1e-4_e1_d10/baseline"
run_pair_mr bs64_lr1e-4_e1_d20_mr   64 1e-4 1 20 3 "$SWEEP/keti2/moment__bs64_lr1e-4_e1_d20/baseline"
run_pair_mr bs64_lr1e-4_e1_d30_mr   64 1e-4 1 30 3 "$SWEEP/keti2/moment__bs64_lr1e-4_e1_d30/baseline"
run_pair_mr bs128_lr2e-4_e1_d20_mr 128 2e-4 1 20 3 "$SWEEP/keti2/moment__bs128_lr2e-4_e1_d20/baseline"
run_pair_mr bs64_lr1e-4_e1_d20_re5_mr 64 1e-4 1 20 5 "$SWEEP/keti2/moment__bs64_lr1e-4_e1_d20/baseline"

echo "[$(date '+%H:%M:%S')] keti_2 round-5 ALL DONE"
