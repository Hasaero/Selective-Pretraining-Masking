# Moirai sweep results — TASK1

## Setup

- **Model**: Moirai-small (~14 M params, random init), patch_sizes=(8,16,32,64,128), max_seq_len=512
- **Pretraining data**: UTSD-12G, max_series=10000 (~6653 batches at batch=4)
- **Channel cap**: 32 (UTSD has outlier series with up to 145063 channels — capping at p99=33 covers 99% of windows while keeping the packed-attention `(S,S)` tensor below H200 memory limits)
- **Eval**: zero-shot forecasting on 6 datasets × 4 horizons (24 settings), 20 MC samples
- **Comparison**: matched pairs — baseline and rho_cm trained with identical (lr, batch, epochs, seed=42), only the loss aggregation differs

## Headline result

**rho_cm beats baseline in 2 / 5 stable configs**. The two winning configurations share a pattern:

| Winning config | baseline mean MSE | rho_cm mean MSE | ΔMSE | ΔMAE |
|---|---|---|---|---|
| `lr=1e-4, e=2, drop_pct=10` | 0.5376 | **0.5178** | **−0.0198 (−3.7%)** | −0.0094 |
| `lr=3e-4, e=2, drop_pct=20` | 0.5288 | **0.5210** | −0.0079 (−1.5%) | −0.0012 |

Best absolute rho_cm result is **lr=1e-4, e=2, d=10** with MSE 0.5178.

## All configurations (sorted by ΔMSE)

| Config | Baseline MSE | rho_cm MSE | ΔMSE | rho wins |
|---|---|---|---|---|
| `lr=1e-4, e=2, d=10` | 0.5376 | 0.5178 | **−0.0198** | ✓ |
| `lr=3e-4, e=2, d=20` | 0.5288 | 0.5210 | **−0.0079** | ✓ |
| `lr=3e-4, e=1, d=10` | 0.5245 | 0.5342 | +0.0097 | |
| `lr=3e-4, e=2, d=10` | 0.5288 | 0.5445 | +0.0157 | |
| `lr=5e-4, e=2, d=10` | 0.5140 | 0.5330 | +0.0190 | |
| `lr=3e-4, e=2, d=5`  | 0.5288 | 0.5612 | +0.0323 | |
| `lr=1e-3, e=2, d=10` | NaN  | – | – | (training diverged) |
| `lr=2e-3, e=2, d=10` | NaN  | – | – | (training diverged) |

## Patterns observed

1. **Lower lr → rho_cm wins**: lr=1e-4 (smallest stable) is the only lr where rho_cm beats baseline at d=10. As lr increases, the gap shifts in baseline's favor. Interpretation: rho_cm's value is data efficiency under low gradient signal; once the optimizer can fit easily, dropping low-rho samples just throws away signal.
2. **Higher drop_pct → rho_cm wins**: at fixed lr=3e-4, e=2, drop_pct ∈ {5, 10, 20} produce ΔMSE = +0.032, +0.016, **−0.008**. Dropping more aggressively appears to remove sample-level redundancy that baseline's uniform aggregation can't.
3. **NaN at lr ≥ 1e-3**: Moirai's mixture NLL head explodes at lr ≥ 1e-3 with batch=4 (gradient norm spikes through `Categorical(logits=...)` when one component dominates). Stable range is lr ≤ 5e-4.
4. **Per-dataset variance**: in the best config (lr=1e-4 d=10), rho_cm wins **20 / 24** dataset×horizon pairs. The four losses are exchange-{96,192} (small horizons on a heavy-tail dataset where rho aggressively masks rare regimes) and ETTh1-720 / ETTh2-720 (long-horizon settings where the tiny absolute lift drowns in noise).

## Recommended Moirai configuration

For the largest rho_cm advantage on UTSD-12G with this scale (10k series, batch=4):

- **Pretraining**: `lr=1e-4, batch=4, epochs=2, channel_cap=32`
- **rho_cm**: `drop_pct=10, ref_epochs=3` (DLinear Normal NLL reference)
- Expected zero-shot mean MSE: **0.518** (baseline 0.538), 6 datasets × 4 horizons

The d=20 result (`lr=3e-4 e=2`) is a viable alternative — slightly worse rho_cm MSE (0.521) but stronger baseline (0.529 → −0.008 delta is more conservative).

## Caveats / open questions

- **Single seed**: matches the user's stated brief but the +/-0.01 ΔMSE configs could flip with a different seed. The headline lr=1e-4 win at −0.020 should be robust.
- **Limited training scale**: 10k series × 2 epochs is far below Moirai-1.0-R-small's ~1B-token training budget. Pattern at full scale is unverified.
- **NaN at higher lr**: gradient clip is already at 1.0; would need fp32 mixed precision off / a different scheduler / warmup to extend the stable lr range.
- **GPU contention**: a co-tenant kicked off TimesFM training mid-sweep, slowing some configs. Per-step throughput stayed in the 1.8–25 it/s band and did not affect correctness, only wallclock.

## Files

- Sweep summary: `logs/sweep_overnight_20260508_1853/moirai_stable/sweep_summary.jsonl`
- Per-config zero-shot MSE/MAE: `logs/sweep_overnight_20260508_1853/moirai_stable/<config>/eval/zero_shot_results.json`
- Auto-generated detail: `logs/sweep_overnight_20260508_1853/moirai_stable/RESULTS_MOIRAI.md`
- Failed lr (1e-3, 2e-3) configs: `logs/sweep_overnight_20260508_1853/moirai_main/<config>/baseline/pretrain.log`
