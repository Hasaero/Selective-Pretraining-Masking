#!/usr/bin/env bash
# Re-run drop=20 unified comparison using PROPER (new) refs:
# - Moirai: mixture_forecast ref (not legacy Normal NLL)
# - MOMENT: realtime masked-recon ref (not legacy identity)
# All paired, baseline reused, ref_epochs=5.
set -uo pipefail

ROOT=/mnt/workspace/juyoung_ha/rho_pretrain
SWEEP=$ROOT/logs/auto_overnight
HOST=$1   # keti_1 or keti_2
HOST_SHORT="${HOST//_/}"   # keti_1 -> keti1
mkdir -p "$SWEEP/$HOST_SHORT"
SUMMARY=$SWEEP/$HOST_SHORT/summary.jsonl
touch "$SUMMARY"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate gracm
cd "$ROOT"

EVAL_SEEDS=(42 43 44)
EVAL_DATASETS=ETTh1,ETTh2,ETTm1,ETTm2,weather,exchange
EVAL_HORIZONS=96,192,336,720
DROP=20

run_pair_proper () {
  local backbone=$1 name=$2 batch=$3 lr=$4 baseline_src=$5
  local script rho_mode eval_mode base_flag rho_flag eval_json eval_extra
  case $backbone in
    moirai) script=scripts/pretrain_moirai.py; rho_mode=rho_cm;       eval_mode=eval_zero_shot_sweep; base_flag=--baseline-ckpt; rho_flag=--rho-cm-ckpt;       eval_json=zero_shot_results.json; eval_extra="";;
    moment) script=scripts/pretrain_moment.py; rho_mode=patch_rho_cm; eval_mode=eval_sweep;            base_flag=--baseline-ckpt; rho_flag=--patch-rho-cm-ckpt; eval_json=linear_probe_results.json; eval_extra="--probe-epochs 5";;
  esac

  local out=$SWEEP/$HOST_SHORT/${backbone}__${name}
  mkdir -p "$out"
  local marker=$out/DONE
  if [[ -f $marker ]]; then echo "[skip] $backbone/$name (DONE)"; return 0; fi

  echo "============================================"
  echo "[$(date '+%H:%M:%S')] $backbone/$name START (drop=$DROP, ref_e=5, NEW ref)"
  echo "============================================"

  if [[ ! -f $out/baseline/best.pt ]]; then
    if [[ -f "$baseline_src/best.pt" ]]; then
      echo "[reuse] baseline ← $baseline_src"
      mkdir -p "$out/baseline"
      ln -sf "$baseline_src/best.pt" "$out/baseline/best.pt"
    else
      echo "[$(date '+%H:%M:%S')] baseline pretrain"
      mkdir -p "$out/baseline"
      python "$script" --mode baseline --epochs 1 --batch-size "$batch" --lr "$lr" \
        --seed 42 --out-dir "$out/baseline" > "$out/baseline/pretrain.log" 2>&1
      [[ ! -f $out/baseline/best.pt ]] && { tail -20 "$out/baseline/pretrain.log"; return 1; }
    fi
  fi

  if [[ ! -f $out/rho/best.pt ]]; then
    echo "[$(date '+%H:%M:%S')] rho pretrain"
    mkdir -p "$out/rho"
    python "$script" --mode "$rho_mode" --epochs 1 --batch-size "$batch" --lr "$lr" \
      --seed 42 --drop-pct "$DROP" --ref-epochs 5 \
      --out-dir "$out/rho" > "$out/rho/pretrain.log" 2>&1
    [[ ! -f $out/rho/best.pt ]] && {
      echo "[FAIL] rho pretrain — last 30 lines:";
      tail -30 "$out/rho/pretrain.log"
      # Detect NaN
      if grep -q 'nan' "$out/rho/pretrain.log"; then
        echo "[INFO] NaN detected; will not retry automatically — review log"
      fi
      return 1
    }
  fi

  for seed in "${EVAL_SEEDS[@]}"; do
    local eval_out=$out/eval_seed${seed}
    [[ -f $eval_out/$eval_json ]] && { echo "[skip] eval seed=$seed exists"; continue; }
    mkdir -p "$eval_out"
    echo "[$(date '+%H:%M:%S')] eval seed=$seed"
    python "$script" --mode "$eval_mode" --seed "$seed" \
      --eval-datasets "$EVAL_DATASETS" --eval-horizons "$EVAL_HORIZONS" \
      --out-dir "$eval_out" $eval_extra \
      $base_flag "$out/baseline/best.pt" $rho_flag "$out/rho/best.pt" \
      > "$eval_out/eval.log" 2>&1
    [[ ! -f $eval_out/$eval_json ]] && { tail -20 "$eval_out/eval.log"; return 1; }
  done

  touch "$marker"
  local now=$(date '+%Y-%m-%dT%H:%M:%S')
  echo "{\"time\":\"$now\",\"host\":\"$HOST\",\"backbone\":\"$backbone\",\"name\":\"$name\",\"batch\":$batch,\"lr\":$lr,\"epochs\":1,\"drop_pct\":$DROP,\"ref_epochs\":5,\"task\":\"d20_proper_ref\",\"status\":\"ok\"}" >> "$SUMMARY"
  echo "[$(date '+%H:%M:%S')] $backbone/$name DONE"
}

if [[ $HOST == keti_1 ]]; then
  # Moirai (uses NEW mixture_forecast ref by default — script already updated)
  run_pair_proper moirai "bs32_lr1e-4_e1_d20_re5_proper" 32 1e-4 \
    "$SWEEP/keti1/moirai__bs32_lr1e-4_e1_d20/baseline"
elif [[ $HOST == keti_2 ]]; then
  # MOMENT (uses NEW realtime masked-recon ref by default)
  run_pair_proper moment "bs64_lr1e-4_e1_d20_re5_proper" 64 1e-4 \
    "$SWEEP/keti2/moment__bs64_lr1e-4_e1_d20/baseline"
fi

echo "[$(date '+%H:%M:%S')] $HOST d20-proper ALL DONE"
