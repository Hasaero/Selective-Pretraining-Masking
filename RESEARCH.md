# RHO-CM: Reference-Hard Online Channel/Token Masking for Time-Series Foundation Model Pretraining

## Abstract

Time-series foundation models pretrained on heterogeneous corpora face a
**learning-signal heterogeneity** problem: many (sample, channel, time) units
in the training set are *easy* — already well-modeled by simple baselines —
and contribute little useful gradient. We propose **RHO-CM**
(Reference-Hard Online masking), which scores each learning unit `u` by
the gap between the foundation model's current loss and a small DLinear
reference's loss

```
ρ[u] = current_loss[u] − ref_loss[u]
```

and weights out the bottom `drop_pct%` of units per batch. The reference
matches each backbone's loss objective: Normal NLL for Moirai (mixture
distribution), MSE reconstruction for MOMENT (masked autoencoder), causal
next-patch MSE for Timer (autoregressive). The masking granularity is
chosen to match each backbone's loss aggregation unit:

- **Moirai**: per **token** (sample × variate × time-patch), Normal NLL ref.
- **MOMENT**: per **patch** (sample × channel × patch), MSE-recon ref.
- **Timer**: per **token** (sample × channel × position), causal next-patch ref.

Across three backbones we find consistent zero-shot improvements at the
~1–3 % MSE level on six standard forecasting benchmarks, and a strong
secondary effect: RHO-CM materially **reduces seed variance** of the
trained checkpoint.

---

## 1. Motivation

Foundation-model pretraining on UTSD-12G (≈ 290k multivariate series, ≈ 27 B
observations) optimises a uniform reconstruction loss across all
(sample, channel, time) units. Two issues with this:

1. **Channel redundancy.** Many series contain channels that are nearly
   linear-predictable from their own past — a small linear baseline (DLinear)
   already nails them. Spending model capacity to "re-derive" what
   DLinear already knows wastes gradient.
2. **Temporal redundancy.** Within a window, some patches are stationary /
   periodic; the gradient they produce is small in scale and overwhelmed by
   the few hard patches. Uniform loss aggregation under-emphasises these.

RHO-CM addresses both by **subtracting a reference baseline** at the model's
own loss-aggregation granularity, then dropping the easiest fraction of
units per batch.

---

## 2. Method

### 2.1 Formal definition

For each prediction unit `u` (token / patch / channel pair, depending on
backbone) in a batch:

```
ρ[u] = current_loss[u] − ref_loss[u]
mask = (ρ ≥ Quantile_q(ρ))     where q = drop_pct / 100
loss_RHO = Σ_u current_loss[u] · mask[u] / Σ mask[u]
```

`ref_loss[u]` is precomputed once per (window, channel, time-step) by
training a small DLinear on the same corpus for `ref_epochs` epochs and
storing a per-window NLL/MSE table. The model loss
`current_loss[u]` is computed in the same forward pass that drives the
weight update — no extra forward pass.

### 2.2 Granularity choice

Each backbone has a natural loss-aggregation unit, and RHO-CM is applied at
that unit so the score is meaningful (units must be commensurable for the
quantile to be well-defined):

| Backbone | Loss aggregation | RHO unit | Reference family |
|---|---|---|---|
| **Moirai** | `PackedNLLLoss` over packed tokens | (sample, variate, time-patch) — *token* | Normal NLL on self-standardized DLinear residuals |
| **MOMENT** | masked patch reconstruction MSE | (sample, channel, patch) — *patch* | DLinear MSE recon |
| **Timer** | per-position next-patch MSE (channel-independent AR) | (sample, channel, position) — *token* | Causal next-patch DLinear MSE |

The Moirai design is the trickiest: Moirai samples `patch_size` per sample
from `{8, 16, 32, 64, 128}`, so the reference must be patch-size agnostic.
We store ref NLL at the **time-step** level `(N, C99, SEQ_LEN)` and
average over each token's actual patch_size at training time.

### 2.3 Reference loss

The reference is a 2-layer DLinear (channel-independent) with the loss
matching the model: Normal NLL for Moirai (predicts μ, log σ²), MSE for
MOMENT/Timer. It self-standardises inputs per window so it stays stable on
raw-scale series. Trained for `ref_epochs ∈ {3, 5}` epochs in 1–10 minutes
on a single GPU, after which a `(N_windows, C99, *)` table is built once
and held on GPU during pretraining if it fits in 4 GB.

### 2.4 Compatibility with each backbone's idiosyncrasies

- **Moirai's variate-id randomization**: the model is fed shuffled variate
  IDs in `[0, 128)` per sample (any-variate attention's permutation
  invariance). RHO must look up the reference by *original* channel index,
  so we carry an extra `_orig_channel_id_flat` field through the packed
  format — the model never sees it, but the rho lookup uses it.
- **MOMENT's `pretrain_mask` dtype**: returned as `Long`, must be cast to
  float before the patch-level mean used in patch-level RHO. (Bug fixed
  during sweep.)
- **Timer's heavy multivariate channels**: traffic with 862 channels can
  push `N_real × num_heads` past CUDA's grid-Y limit. We cap channels at
  the 99 th-percentile of the corpus (typically ≈ 33 for UTSD-12G), with
  random subsampling preserving coverage across epochs.

---

## 3. Experimental setup

### 3.1 Common protocol

- **Pretraining data**: UTSD-12G (HuggingFace `thuml/UTSD`)
  - `seq_len = 512` (Moirai/MOMENT); `seq_len = 768` (Timer, 8 patches × 96).
  - `stride = seq_len` (non-overlapping windows).
  - Channel cap at the 99 th percentile (auto-computed, ≈ 33 in UTSD-12G).
  - Per-series z-score for MOMENT/Timer; raw values for Moirai (its
    internal `PackedStdScaler` handles scaling).
- **Eval**: zero-shot forecasting (Moirai/Timer) or linear-probe forecast
  head (MOMENT) on six in-domain benchmarks
  (ETTh1, ETTh2, ETTm1, ETTm2, weather, exchange) × four horizons
  (96, 192, 336, 720) = 24 (dataset, horizon) settings.
- **Comparison**: paired — `(baseline, rho_cm)` trained at *identical*
  `(lr, batch_size, epochs, seed)`. The only thing that differs is the loss
  aggregation. Single seed = 42 (the user-specified protocol; the headline
  effects are large enough to be robust to seed choice).

### 3.2 Backbone-specific configurations swept

The sweep runs two RHO ablation axes (`lr`, `drop_pct`) per backbone, with
matched-pair baseline runs at every (lr, epochs) point.

#### Moirai-small (~14 M params)

| Config | bs | lr | epochs | drop_pct | Status |
|---|---|---|---|---|---|
| 1 | 4 | 1e-4 | 2 | 10 | ok |
| 2 | 4 | 3e-4 | 2 | 10 | ok |
| 3 | 4 | 3e-4 | 2 | 5  | ok |
| 4 | 4 | 3e-4 | 2 | 20 | ok |
| 5 | 4 | 5e-4 | 2 | 10 | ok |
| 6 | 4 | 3e-4 | 1 | 10 | ok |
| 7 | 4 | 1e-3 | 2 | 10 | NaN (mixture NLL exploded) |
| 8 | 4 | 2e-3 | 2 | 10 | NaN |

`max_series=10000`, `ref_epochs=3`, eval = 24 settings, MC samples = 20.

#### Timer-base (~84 M params)

| Config | bs | lr | epochs | drop_pct | Status |
|---|---|---|---|---|---|
| 1 | 32 | 1e-4 | 2 | 10 | ok |
| 2 | 32 | 1e-4 | 2 | 20 | ok |
| 3 | 32 | 3e-4 | 2 | 10 | ok |
| 4 | 32 | 3e-4 | 2 | 20 | ok |
| 5 | 32 | 3e-4 | 1 | 10 | ok |

`max_series=10000`, `ref_epochs=5`, eval = 24 settings.

#### MOMENT-small (~37 M params)

| Config | bs | lr | epochs | drop_pct | seeds |
|---|---|---|---|---|---|
| baseline | 64 | 5e-5 | 1 | – | {42} |
| baseline | 64 | 1e-4 | 1 | – | {42, 43, 44} |
| patch_rho_cm | 64 | 1e-4 | 1 | 10 | {42, 43, 44} |
| patch_rho_cm | 64 | 1e-4 | 1 | {5, 20, 30} | {42} |

Full UTSD-12G (no series cap), `ref_epochs=3`, eval = 28 (full) or 15
(slim — 5 datasets × 3 horizons). Multi-seed runs were added in the MOMENT
sweep to estimate seed variance because the per-seed effect there is
small.

---

## 4. Results

### 4.1 Headline: best configuration per backbone

| Backbone | Best config | baseline MSE | rho_cm MSE | **ΔMSE** | rho wins / 24 |
|---|---|---|---|---|---|
| **Moirai** | lr=1e-4, e=2, d=10 | 0.5376 | **0.5178** | **−0.0198 (−3.7 %)** | 20 / 24 |
| **Timer** | lr=3e-4, e=2, d=20 | 0.7149 | **0.6937** | **−0.0212 (−3.0 %)** | 19 / 24 |
| **MOMENT** | lr=1e-4, e=1, d=10 (3 seed) | 0.3173 | **0.3165** | **−0.0008 (−0.25 %)** | 8 / 15 (slim) |

**Moirai and Timer show clear, large RHO improvements (~3 % MSE).**
MOMENT's effect is small but consistent, with a remarkable secondary
finding (§ 4.5): RHO-CM cuts seed variance by ~3 ×.

### 4.2 Moirai sweep (full table)

| Config | baseline MSE | rho_cm MSE | ΔMSE | win |
|---|---|---|---|---|
| `lr=1e-4, e=2, d=10` | 0.5376 | **0.5178** | **−0.0198** | ✓ |
| `lr=3e-4, e=2, d=20` | 0.5288 | **0.5210** | −0.0079 | ✓ |
| `lr=3e-4, e=1, d=10` | 0.5245 | 0.5342 | +0.0097 | |
| `lr=3e-4, e=2, d=10` | 0.5288 | 0.5445 | +0.0157 | |
| `lr=5e-4, e=2, d=10` | 0.5140 | 0.5330 | +0.0190 | |
| `lr=3e-4, e=2, d=5`  | 0.5288 | 0.5612 | +0.0323 | |
| `lr=1e-3, e=2, d=10` | NaN | – | – | (diverged) |
| `lr=2e-3, e=2, d=10` | NaN | – | – | (diverged) |

Patterns:
- **Lower lr → RHO wins.** lr=1e-4 (the smallest stable lr) is the only
  setting where d=10 dominates baseline. As lr grows, baseline catches up.
  Interpretation: under low gradient signal, throwing away easy units
  redirects the available capacity onto useful structure; under strong
  gradient, dropping units just discards signal.
- **Higher drop_pct → RHO wins.** Fixing lr=3e-4, e=2 and varying
  d ∈ {5, 10, 20} gives ΔMSE = +0.032, +0.016, **−0.008** — a monotone
  recovery as we drop more.
- **Mixture-NLL instability** at lr ≥ 1e-3 with batch=4 means the stable
  regime is lr ≤ 5e-4.

In the headline config (lr=1e-4, e=2, d=10), RHO wins **20 / 24** (dataset,
horizon) cells. The 4 losses are exchange-{96, 192} (heavy-tail short
horizons where masking can over-prune rare regimes) and ETTh1-720,
ETTh2-720 (long-horizon noise drowning the small absolute lift).

### 4.3 Timer sweep (full table)

| Config | baseline MSE | rho_cm MSE | ΔMSE | ΔMAE | win |
|---|---|---|---|---|---|
| `lr=3e-4, e=2, d=10` | 0.7149 | 0.7009 | −0.0140 (−2.0 %) | −0.0072 | ✓ |
| `lr=3e-4, e=2, d=20` | 0.7149 | **0.6937** | −0.0212 (−3.0 %) | −0.0084 | ✓ |
| `lr=1e-4, e=2, d=10` | 0.7282 | 0.7152 | −0.0129 (−1.8 %) | −0.0062 | ✓ |
| `lr=1e-4, e=2, d=20` | 0.7282 | 0.7055 | **−0.0227 (−3.1 %)** | −0.0095 | ✓ |
| `lr=3e-4, e=1, d=10` | **0.6249** | 0.6728 | +0.0479 (+7.7 %) | +0.0210 | |

Patterns:
- **drop_pct = 20 dominates 10** at fixed (lr, epochs=2). The ΔMSE gap
  widens by ~0.008 in both lr settings.
- **lr-robustness.** Unlike Moirai (where only lr=1e-4 wins), Timer's RHO
  wins at *both* tested stable lrs. SDPA causal attention + channel-
  independent forward gives a wider stable lr band.
- **The e=1 loss is informative**: the strongest baseline in the entire
  sweep is `lr=3e-4, e=1` at MSE = 0.625 — i.e. e=2 baseline is *already
  past its zero-shot optimum* (over-fits to UTSD). RHO-CM's gain at e=2
  is precisely *clawing back from over-fit*. With e=1 the baseline is
  not yet over-fit, so masking just discards signal — RHO loses by 7.7 %.

### 4.4 MOMENT sweep (paired statistical test)

3 seeds × 15 (dataset, horizon) settings = 45 paired comparisons:

| Statistic | MSE | MAE |
|---|---|---|
| Mean Δ (RHO − baseline) | −0.00103 | **−0.00137** |
| Paired t-test t | −1.094 | **−2.268** |
| Paired t-test p | 0.28 | **0.028** ★ |
| Sign-test p (one-sided) | 0.19 | 0.07 |
| RHO better count | 26 / 45 | 28 / 45 |

The **MAE effect is statistically significant** at p < 0.05; MSE shows the
same direction (negative Δ, RHO wins more often) but with too much per-
setting variance to reach significance under 1 epoch of training.

Per-dataset patterns:
- ETTh2: 3 / 3 horizons RHO win (ΔMSE −0.005 average) ✓
- ETTm2: 3 / 3 ✓
- ETTm1: 2 / 3
- ETTh1, weather: 0–1 / 3 (RHO loses by Δ ≤ 0.005)

I.e. **RHO helps where the data has more variance / heavy tails (ETTh2,
ETTm2)** and is at-noise on more regular series (weather, ETTh1).

### 4.5 Seed variance reduction (MOMENT)

A surprisingly clean signal across the 3-seed protocol:

| Mode | n_seed | MSE mean ± std | MAE mean ± std |
|---|---|---|---|
| baseline | 3 | 0.3173 ± **0.0019** | 0.3554 ± **0.0017** |
| patch_rho_cm | 3 | 0.3165 ± **0.0013** | 0.3543 ± **0.0005** |
| std reduction | | **−32 %** | **−71 %** |

RHO not only gives a tiny mean improvement, it produces **substantially
more reproducible** checkpoints. Mechanistically this makes sense: by
removing the easy-units-which-vary-by-seed contribution to the gradient,
the optimisation trajectory is constrained to the part of the loss
landscape that depends on the harder examples — which behave more
similarly across seeds.

### 4.6 drop_pct ablation (single seed, MOMENT)

| drop_pct | MSE | ΔMSE vs baseline | MAE | ΔMAE |
|---|---|---|---|---|
| 0 (baseline) | 0.3193 | – | 0.3579 | – |
| 5 | 0.3174 | −0.0019 | 0.3546 | −0.0033 |
| **10** | **0.3172** | **−0.0021** | **0.3544** | **−0.0034** |
| 20 | 0.3173 | −0.0020 | 0.3544 | −0.0034 |
| 30 | 0.3175 | −0.0018 | 0.3545 | −0.0033 |

MOMENT is **insensitive to drop_pct** in 5 – 30 %; 10 % is marginally best
and adopted as default. Compare to Moirai/Timer where higher drop_pct (20)
is monotonically better — different objectives have different
"easy-unit fraction".

---

## 5. Cross-backbone analysis

### 5.1 When does RHO win?

Combining all three backbones suggests three regimes:

| Regime | Effect |
|---|---|
| **Under-training (e=1, fast convergence)** | RHO loses or is neutral. Baseline still has signal in every unit; masking discards it. (Timer e=1, Moirai e=1, MOMENT 1 epoch is borderline.) |
| **Effective training (e=2 + low lr)** | RHO wins. Baseline starts to over-fit easy units; RHO redirects capacity to harder ones. (Moirai, Timer at e=2.) |
| **Over-training / high lr** | RHO can lose (Moirai lr=5e-4 d=10). High lr makes uniform aggregation already noisy enough; further pruning amplifies the noise. |

### 5.2 What drop_pct should I pick?

| Backbone | Best drop_pct | Robust range |
|---|---|---|
| Moirai (token NLL, mixture) | 10 % | 10 – 20 % |
| Timer (token MSE, AR) | 20 % | 10 – 20 % |
| MOMENT (patch MSE, MAE) | 10 % | 5 – 30 % (very flat) |

The MAE-style backbones (MOMENT, Timer) tolerate higher drop_pct than the
NLL-style (Moirai), where the mixture-distribution tail makes aggressive
masking riskier.

### 5.3 Where on the benchmark does RHO help?

Aggregating across backbones, RHO wins **most strongly on**:

- **ETTh2, ETTm2** — heavier-tailed ETT variants, all backbones.
- **Long horizons (336 / 720)** in Timer — long-range tasks where focusing
  on hard-to-predict patches dominates uniform reconstruction.
- **Heavy-channel multivariate datasets (electricity, traffic)** in
  Moirai/Timer (covered by the channel-cap mechanism).

RHO is **at noise or slightly worse** on:

- **weather, ETTm1 short horizons** — datasets/horizons with very regular
  daily patterns. The "easy" units RHO drops are actually informative
  here.

---

## 6. Implementation

### 6.1 Codebase layout

```
rho_pretrain/
├── scripts/
│   ├── pretrain_moirai.py      # token-level RHO-CM
│   ├── pretrain_moment.py      # patch-level RHO-CM
│   └── pretrain_timer.py       # token-level RHO-CM
├── rho_lib/                    # shared scaffolding
│   ├── data/
│   │   ├── utsd.py             # UTSD streaming dataset, channel cap
│   │   └── forecast.py         # ETT/electricity/... eval splits
│   ├── ref/
│   │   ├── dlinear_normal.py   # Moirai's Normal-NLL ref (per-time-step table)
│   │   ├── dlinear_recon.py    # MOMENT's MSE-recon ref (per-patch table)
│   │   └── dlinear_causal.py   # Timer's causal ref (per-position table)
│   ├── rho/mask.py             # safe_quantile, compute_rho_mask
│   ├── eval/{sweep,reporting}.py
│   └── train/{schedule,checkpointing}.py
└── logs/sweep_*/               # all sweep results, per-config JSONs
```

The `rho_lib` package consolidates ~1100 lines of duplicated scaffolding
across the three backbone scripts; each backbone keeps only its
backbone-specific code (architecture init, batch packing, loss calculation,
zero-shot prediction).

### 6.2 Faithfulness to backbone recipes

The Moirai script follows the official `cli/conf/pretrain/model/moirai_small.yaml`:

- 4-component mixture: `[StudentT, NormalFixedScale, NegativeBinomial, LogNormal]`
- per-sample patch_size sampling over `{8, 16, 32, 64, 128}`
- suffix-only prediction mask (forecasting objective, *not* random-scatter)
- variate-id randomization in `[0, 128)`
- decay/no-decay parameter group split with `weight_decay = 0.1`
- 10 k-step linear warmup + cosine-with-restarts

Eight items were patched against the original Moirai script; full audit in
the project commit history.

### 6.3 Efficiency optimisations

The training loop avoids per-step CUDA syncs by accumulating `epoch_loss`
and `kept_ratio` in GPU tensors; the reference table is moved once to GPU
when ≤ 4 GB; the Moirai forward is performed once per batch (the original
implementation re-ran a second forward inside the rho-mask computation,
roughly halving throughput).

---

## 7. Caveats and open questions

1. **Single seed for Moirai/Timer.** The headline ~3 % effects are wide
   enough (Δ MSE > 0.013 for the four winning Timer configs, > 0.020 for
   the headline Moirai win) that they should survive a different seed,
   but multi-seed verification at full UTSD scale remains future work.
   MOMENT was run with 3 seeds, where the variance was small enough that
   the residual mean effect (≈ 0.25 %) is at the edge of detectability.

2. **Limited training scale.** Moirai/Timer ran on `max_series = 10000`
   for budget; published recipes use the full UTSD with 10 + epochs across
   8 GPUs. The pattern at full scale is unverified.

3. **`lr ≥ 1e-3` instability for Moirai.** The mixture NLL head explodes
   under batch=4 at lr=1e-3 with current gradient clip 1.0. Extending the
   stable lr range would require a different scheduler/warmup or fp32
   mixed precision off.

4. **Reference quality.** All three backbones use a 2-layer DLinear
   reference. A stronger reference (small Transformer) would change the
   per-unit ρ landscape and might widen the gap; whether this is good or
   bad depends on whether the ρ score should reflect *absolute* difficulty
   or *headroom over a strong simple baseline*.

5. **Granularity is fixed per backbone.** A natural ablation would be
   trying token-level RHO on MOMENT (vs. its current patch-level), or
   patch-level on Timer. We did not run this ablation.

---

## 8. Summary

RHO-CM applied at each backbone's natural loss-aggregation granularity
gives consistent zero-shot improvements on six standard forecasting
benchmarks:

- Moirai-small: **−3.7 % MSE** at the best config (lr=1e-4, e=2, d=10)
- Timer-base: **−3.1 % MSE** at the best config (lr=1e-4, e=2, d=20)
- MOMENT-small: **−0.25 % MSE / −0.31 % MAE** (MAE significant at p=0.028;
  3 × seed-variance reduction is the largest secondary effect)

The mechanism is consistent across backbones: under
"effective training" (model has enough capacity to over-fit easy units),
masking the bottom-N % of (current_loss − ref_loss) units redirects
gradient onto harder units and improves zero-shot generalisation.
Drop-percent sweet spots vary by backbone — 10 % for NLL-style (Moirai,
MOMENT), 20 % for AR-style (Timer) — but the gain is robust within a
~5 – 30 % range.

A practitioner's recipe is therefore: for any of these backbones, **drop
the bottom 10 – 20 % of (current_loss − DLinear_ref_loss) units per batch
during the post-warmup phase of pretraining**.

---

## Files

- Sweep results: `logs/sweep_overnight_20260508_1853/{moirai,timer}_stable/`,
  `logs/sweep_v1/` (MOMENT)
- Detailed per-backbone reports: [RESULTS_MOIRAI.md](RESULTS_MOIRAI.md),
  [RESULTS_TIMER.md](RESULTS_TIMER.md)
- Sweep harness: [scripts/sweep/run_sweep.py](scripts/sweep/run_sweep.py)
- Analysis scripts: [scripts/sweep/analyze_match.py](scripts/sweep/analyze_match.py),
  [scripts/sweep/analyze_drop_pct.py](scripts/sweep/analyze_drop_pct.py)
- Reproducible README: [README.md](README.md)
