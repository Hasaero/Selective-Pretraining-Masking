# RHO-CM: Reference-Hard Online Channel Masking for Time-Series Foundation Model Pretraining

This directory contains the official implementation of RHO-CM applied to three
representative time-series foundation models — **Moirai**, **MOMENT**, and
**Timer** — pretrained on **UTSD-12G**.

## Repository structure

```
rho_pretrain/
├── README.md
├── requirements.txt
├── scripts/
│   ├── pretrain_moirai.py       # Moirai-small + token-level RHO-CM
│   ├── pretrain_moment.py       # MOMENT-small + patch-level RHO-CM
│   └── pretrain_timer.py        # Timer-base + RHO-CM
├── rho_lib/                     # Shared scaffolding (data, ref-loss, rho-mask, eval, train)
│   ├── data/
│   │   ├── utsd.py              # UTSDPretrainDataset, make_collate_fn, compute_channel_cap
│   │   └── forecast.py          # ETT/electricity/traffic/weather/exchange/ILI/ECL splits
│   ├── ref/
│   │   ├── dlinear_normal.py    # Moirai reference: Normal NLL
│   │   ├── dlinear_recon.py     # MOMENT reference: MSE reconstruction
│   │   └── dlinear_causal.py    # Timer reference: causal next-patch MSE
│   ├── rho/mask.py              # safe_quantile, compute_rho_mask
│   ├── eval/{sweep,reporting}.py
│   └── train/{schedule,checkpointing}.py
└── data/                        # see "Data" section
```

## Setup

```bash
# 1. Conda environment (Python 3.10)
conda create -n rho_pretrain -c conda-forge python=3.10 pip -y
conda activate rho_pretrain

# 2. PyTorch (CUDA 12.4 wheel)
pip install torch==2.4.1 torchvision==0.19.1 \
    --index-url https://download.pytorch.org/whl/cu124

# 3. Remaining dependencies
pip install -r requirements.txt
```

Tested on H200 NVL (CUDA 12.8 driver / 12.4 toolkit). The pinned versions of
`momentfm 0.1.4`, `transformers 4.33.3`, and `uni2ts 2.0.0` are mutually
compatible — do not upgrade individually.

## Data

### Pretraining: UTSD-12G

Save the HuggingFace UTSD-12G dataset to `data/utsd_repo/UTSD-12G/` so it can
be loaded via `datasets.load_from_disk`. The expected layout is 80 `.arrow`
shards plus `dataset_info.json` and `state.json`, totalling ≈ 290k rows /
3.9 GB. Source: https://huggingface.co/datasets/thuml/UTSD .

### Evaluation: forecasting benchmarks

Place the following CSVs under `data/`:

```
data/ETT-small/{ETTh1,ETTh2,ETTm1,ETTm2}.csv
data/electricity/electricity.csv
data/traffic/traffic.csv
data/weather/weather.csv
data/Exchange/Exchange.csv
data/ILI/ILI.csv
data/ECL/ECL.csv
```

Standard sources (Informer / Autoformer benchmark suite). Splits are computed
inside [rho_lib/data/forecast.py](rho_lib/data/forecast.py) following the
canonical ETT split rules (12-month train / 4-month val / 4-month test for
ETT; 70/10/20 for the rest).

## Reproducing main results

### Pretraining

```bash
# Moirai-small (14 M params)
python scripts/pretrain_moirai.py --mode baseline --epochs 5 --batch-size 32
python scripts/pretrain_moirai.py --mode rho_cm   --epochs 5 --batch-size 32 \
    --ref-epochs 3 --drop-pct 10

# MOMENT-small (37 M params)
python scripts/pretrain_moment.py --mode baseline --epochs 2 --batch-size 64
python scripts/pretrain_moment.py --mode patch_rho_cm --epochs 2 --batch-size 64 \
    --ref-epochs 3 --drop-pct 10

# Timer-base (84 M params)
python scripts/pretrain_timer.py  --mode baseline --epochs 10 --batch-size 32
python scripts/pretrain_timer.py  --mode rho_cm   --epochs 10 --batch-size 32 \
    --ref-epochs 10 --drop-pct 10
```

Checkpoints are written to `results_<model>/<mode>/{epoch%03d.pt, best.pt, metrics.json}`.

### Evaluation

```bash
# Linear probe (Moirai, MOMENT)
python scripts/pretrain_moirai.py --mode eval_sweep \
    --baseline-ckpt results_moirai/baseline/best.pt \
    --rho-cm-ckpt   results_moirai/rho_cm/best.pt

python scripts/pretrain_moment.py --mode eval_sweep \
    --baseline-ckpt     results_pretrain/baseline/best.pt \
    --patch-rho-cm-ckpt results_pretrain/patch_rho_cm/best.pt

# Zero-shot (Moirai, Timer)
python scripts/pretrain_moirai.py --mode eval_zero_shot_sweep \
    --baseline-ckpt results_moirai/baseline/best.pt \
    --rho-cm-ckpt   results_moirai/rho_cm/best.pt

python scripts/pretrain_timer.py  --mode eval_zero_shot_sweep \
    --baseline-ckpt results_timer/baseline/best.pt \
    --rho-cm-ckpt   results_timer/rho_cm/best.pt
```

Each sweep prints a `(dataset × horizon)` MSE/MAE table for every checkpoint,
followed by Δ-vs-baseline rows, and writes `*_results.json` to the chosen
`--out-dir`.

## RHO-CM in one paragraph

RHO-CM scores each *learning unit* of the model's loss by the gap to a small
DLinear reference trained on the same corpus, and drops the bottom `drop_pct%`
units per batch (the ones the model already handles as well as DLinear).
The granularity matches each backbone's loss aggregation:

- **Moirai**: per **token** (sample × variate × time-patch), Normal NLL ref.
- **MOMENT**: per **patch** (sample × channel × patch), MSE-recon ref.
- **Timer**:  per **token** (sample × channel × position), causal next-patch
  MSE ref.

In every case `ρ[u] = current_loss[u] − ref_loss[u]` — units with the lowest
gap (model is already as good as DLinear) are weighted out of the loss.

## CLI reference (common flags)

| Flag | Meaning |
|------|---------|
| `--mode {baseline,rho_cm,...}` | training/eval mode (see each script's `--help`) |
| `--epochs`, `--batch-size`, `--lr` | core training schedule |
| `--ref-epochs` | DLinear reference training epochs (rho_cm only) |
| `--drop-pct` | % of learning units (token/patch) to drop per batch (rho_cm only) |
| `--max-series` | cap on UTSD source series (for quick tests) |
| `--max-windows` | cap on training windows (Timer only) |
| `--seed` | RNG seed (default 42) |
| `--out-dir` | output directory (default `results_<model>/<mode>`) |

## Notes

- Moirai's [scripts/pretrain_moirai.py](scripts/pretrain_moirai.py) follows the official
  `cli/conf/pretrain/model/moirai_small.yaml` recipe: 4-component mixture
  output (StudentT, NormalFixedScale, NegativeBinomial, LogNormal),
  per-sample patch-size sampling over `{8, 16, 32, 64, 128}`, suffix-only
  prediction mask, randomized variate IDs in `[0, 128)`,
  decay/no-decay parameter group split with `wd = 0.1`,
  10k-step linear warmup + cosine-with-restarts.
- Moirai applies RHO-CM at the **token granularity** — matching the unit
  Moirai's `PackedNLLLoss` aggregates over. The DLinear reference produces a
  `(N_windows, C99, SEQ_LEN)` per-time-step NLL table; the training loop
  averages it over each token's actual `patch_size` (sampled per-sample from
  `{8, 16, 32, 64, 128}`) so the reference is patch-size-agnostic.
- MOMENT applies RHO-CM at the patch granularity: per-(sample, channel, patch)
  reconstruction MSE compared against a DLinear reference, with bottom-N%
  triples dropped per batch (`patch_rho_cm`).
- Timer applies RHO-CM at the token granularity: per-(sample, channel,
  position) next-patch MSE.
- Timer caps channels at the 99th percentile (`compute_channel_cap`) to keep
  `N_real × num_heads` under CUDA's grid limit on heavy multivariate series
  (traffic 862 ch, ECL 321 ch).
- The reference-loss table is moved to GPU once when it fits (2 GB threshold
  for Timer/MOMENT, 4 GB for Moirai's larger time-step table), avoiding
  per-batch CPU↔GPU copies in the training loop.
