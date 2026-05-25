# Selective Pretraining Masking (SPM) for Time-Series Foundation Models

> Pretraining-time token selection that drops easy examples and keeps only
> tokens where the foundation model's current loss exceeds a small reference
> model's loss. Applied to **Timer**, **MOMENT**, and **TimesFM 1.0**.

## TL;DR

For each pretraining example we compute

$$
\rho[u] = L_{\text{current}}[u] - L_{\text{ref}}[u]
$$

with a lightweight DLinear reference, and at every step we keep only tokens
with $\rho[u] > \tau$ — where $\tau$ is calibrated once on the first 200
batches to hit a target keep ratio $k$.

| Model | Params | Architecture | Best $k$ | Zero-shot $\Delta$MSE vs baseline |
|---|---|---|---|---|
| **Timer** | 84M | Causal decoder | 0.4 | **−6.94 %** (win 23/24) |
| **MOMENT** | 35M | BERT-style encoder + recon | 0.6 | **−1.43 %** (win 19/24) |
| **TimesFM 1.0** | 203M | Patched decoder + quantile head | 0.4 | **−8.80 %** (win 21/24) |

All numbers averaged over 6 benchmark datasets
(ETTh1, ETTh2, ETTm1, ETTm2, weather, exchange) and 4 horizons (96 / 192 / 336 / 720).

See [`docs/01_method.md`](docs/01_method.md) for the method writeup and
[`docs/03_timer_results.md`](docs/03_timer_results.md) for the Timer sweep.

## Repo layout

```
rho_pretrain/
├── README.md
├── requirements.txt
├── docs/                      # method writeup + experiment notes
├── scripts/
│   ├── pretrain_timer.py      # Timer pretrain (baseline + SPM variants)
│   ├── pretrain_moment.py     # MOMENT pretrain
│   ├── pretrain_timesfm.py    # TimesFM 1.0 pretrain (random-init 200M)
│   ├── timer_fewshot.py       # Timer few-shot fine-tuning
│   ├── timesfm_fewshot.py     # TimesFM few-shot fine-tuning
│   ├── postprocess_run.py
│   └── sweep/                 # plotting + sweep helpers
├── rho_lib/
│   ├── data/                  # UTSD loader + forecast eval datasets
│   ├── eval/                  # sweep harness + result tables
│   ├── ref/                   # DLinear reference (causal / mse / recon-masked)
│   ├── rho/                   # top-K ρ mask
│   ├── train/                 # checkpoint + LR schedule helpers
│   └── _timesfm_v1_src/       # TimesFM v1 PyTorch source (vendored)
├── data/                      # UTSD-12G + benchmark CSVs
├── logs/                      # ckpts + eval JSONs
└── plots/                     # generated figures (loss curves, sweeps, etc.)
```

## Quick start

### Environment

Two conda environments are required (transformers version conflict between
MOMENT and TimesFM):

```bash
# Timer + MOMENT
conda create -n gracm python=3.10 -y
conda activate gracm
pip install -r requirements.txt

# TimesFM
conda create -n gracm_ttm python=3.11 -y
conda activate gracm_ttm
pip install -r requirements.txt
```

### Pretraining (best SPM config per model)

```bash
# Timer baseline
python scripts/pretrain_timer.py --mode baseline --univariate \
    --epochs 3 --batch-size 512 --lr 3e-4 --seed 42 --out-dir <dir>

# Timer SPM (calibrated, k=0.4)
python scripts/pretrain_timer.py --mode threshold_rho --univariate \
    --epochs 3 --batch-size 512 --lr 3e-4 \
    --target-keep-pct 0.4 --calib-batches 200 --ref-epochs 2 \
    --seed 42 --out-dir <dir>

# MOMENT baseline
python scripts/pretrain_moment.py --mode baseline --univariate \
    --epochs 1 --batch-size 1024 --lr 1e-4 --seed 42 --out-dir <dir>

# MOMENT SPM (k=0.6)
python scripts/pretrain_moment.py --mode threshold_rho --univariate \
    --epochs 1 --batch-size 1024 --lr 1e-4 \
    --target-keep-pct 0.6 --calib-batches 200 --ref-epochs 2 \
    --seed 42 --out-dir <dir>

# TimesFM baseline
python scripts/pretrain_timesfm.py --mode baseline \
    --epochs 1 --batch-size 128 --lr 5e-6 --seed 42 --out-dir <dir>

# TimesFM SPM (k=0.4)
python scripts/pretrain_timesfm.py --mode threshold_rho \
    --epochs 1 --batch-size 128 --lr 5e-6 \
    --target-keep-pct 0.4 --calib-batches 200 --ref-epochs 2 \
    --seed 42 --out-dir <dir>
```

### Zero-shot evaluation

```bash
python scripts/pretrain_<model>.py --mode eval_zero_shot_sweep \
    --baseline-ckpt <baseline.pt> --rho-cm-ckpt <spm.pt> \
    --eval-datasets ETTh1,ETTh2,ETTm1,ETTm2,weather,exchange \
    --eval-horizons 96,192,336,720 \
    --out-dir <eval_dir>
```

### Few-shot fine-tuning

```bash
# Timer (full-FT, lr=1e-6, epochs=3)
python scripts/timer_fewshot.py --full-ft \
    --baseline-ckpt <baseline.pt> --rho-cm-ckpt <spm.pt> \
    --train-frac 0.01 --epochs 3 --batch-size 32 --lr 1e-6 \
    --out-json <out.json>

# TimesFM (full-FT, lr=1e-6, epochs=10)
python scripts/timesfm_fewshot.py --full-ft \
    --baseline-ckpt <baseline.pt> --rho-cm-ckpt <spm.pt> \
    --train-frac 0.01 --epochs 10 --batch-size 32 --lr 1e-6 \
    --out-json <out.json>
```

## Dependencies (key versions)

| Package | gracm | gracm_ttm |
|---|---|---|
| python | 3.10 | 3.11 |
| torch | 2.4.1 + cu121 | 2.10 + cu128 |
| transformers | 4.33.3 | 4.49.0 |
| momentfm | 0.1.x | — |
| granite-tsfm | — | 0.2.28 |

## License

This repository contains:
- Original SPM code: Apache 2.0.
- Vendored TimesFM v1 PyTorch source (`rho_lib/_timesfm_v1_src/`) by Google
  Research, Apache 2.0.
- MOMENT is loaded from `AutonLab/MOMENT-1-{small,base,large}` on HuggingFace.

## Citation

```bibtex
@article{spm_tsfm_2026,
  title  = {Selective Pretraining Masking for Time-Series Foundation Models},
  author = {Anonymous},
  year   = {2026},
  note   = {NeurIPS submission}
}
```
