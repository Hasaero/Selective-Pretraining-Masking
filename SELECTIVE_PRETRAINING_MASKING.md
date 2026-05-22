# Selective Pretraining Masking for Time-Series Foundation Models

## 0. 요약

이 연구는 대규모 시계열 foundation model pretraining에서 모든 학습 단위를 동일하게
사용하는 기존 방식이 비효율적이라는 문제에서 출발한다. UTSD-12G 같은 이질적인
시계열 corpus에는 이미 간단한 선형 모델로도 잘 설명되는 쉬운 sample, channel,
patch가 많다. 이런 쉬운 단위까지 동일한 loss weight로 학습하면 모델은 중요한
gradient budget을 중복적이고 낮은 정보량의 신호에 소비한다.

본 연구의 핵심 아이디어는 각 학습 단위의 현재 모델 loss에서 가벼운 reference
model의 loss를 뺀 값을 **reference-relative excess loss**로 정의하고, excess
loss가 낮은 쉬운 단위를 pretraining loss에서 선택적으로 제거하는 것이다.

```text
excess_loss[u] = current_loss[u] - reference_loss[u]
```

여기서 `u`는 모델마다 다르다.

| Model | Learning unit `u` | Current loss | Reference loss |
|---|---|---|---|
| Moirai | sample x variate x time-patch token | token NLL | DLinear Normal NLL |
| MOMENT | sample x channel x patch | reconstruction MSE | DLinear masked-recon MSE |
| Timer | sample x channel x autoregressive position | next-patch MSE | DLinear causal next-patch MSE |

각 batch에서 `excess_loss`가 낮은 bottom `drop_pct%` 단위를 제거하고, 나머지 단위에
대해서만 backpropagation을 수행한다. 본 문서에서는 이 방법을 **Selective Pretraining
Masking, SPM**이라고 부르고, 기존 코드의 구현명인 `rho_cm`은 SPM의 한 구현으로
본다.

---

## 1. 연구 문제

### 1.1 배경

Time-series foundation model은 여러 도메인의 시계열을 하나의 대규모 corpus로
모아 pretraining한다. 본 repository에서는 UTSD-12G를 사용하며, 세 backbone을
대상으로 실험한다.

| Backbone | 구현 파일 | 모델 규모 | 주요 pretraining objective |
|---|---|---:|---|
| Moirai-small | `scripts/pretrain_moirai.py` | 약 14M | suffix masked patch prediction, probabilistic NLL |
| MOMENT-small | `scripts/pretrain_moment.py` | 약 37M | masked autoencoding, reconstruction MSE |
| Timer-base | `scripts/pretrain_timer.py` | 약 84M | autoregressive next-patch prediction, MSE |

세 모델 모두 입력 시계열을 window, channel, patch 또는 token 단위로 나누어 loss를
계산한다. 기존 pretraining은 이 단위들을 거의 균등하게 평균낸다.

### 1.2 문제 의식

대규모 시계열 corpus에는 다음과 같은 쉬운 학습 단위가 많다.

- 거의 선형 trend만 갖는 channel
- 반복적이고 주기성이 강한 patch
- 다른 channel과 중복적인 multivariate signal
- 짧은 horizon에서 단순 persistence 또는 linear extrapolation으로 충분한 구간

이런 단위는 작은 DLinear reference만으로도 낮은 loss를 달성할 수 있다. Foundation
model이 이 단위를 반복적으로 학습하는 것은 representation learning 관점에서
효율적이지 않을 수 있다. 반대로 reference보다 foundation model이 여전히 높은
loss를 내는 단위는 비선형성, long-range dependency, irregular regime을 포함할
가능성이 높다.

따라서 이 연구의 질문은 다음과 같다.

> Pretraining 중 모델이 이미 잘 처리하거나 단순 reference와 차이가 거의 없는 쉬운
> 단위를 줄이고, reference 대비 아직 어려운 단위에 gradient를 집중하면 downstream
> forecasting 성능이 좋아지는가?

---

## 2. 핵심 개념: Reference-Relative Excess Loss

### 2.1 표기

학습 단위 하나를 `u`라고 하자. `u`는 backbone에 따라 token, patch, 또는
autoregressive position이다.

- Foundation model: `f_theta`
- 고정된 reference model: `g_ref`
- 현재 모델의 단위 loss: `ell_theta(u)`
- reference model의 단위 loss: `ell_ref(u)`

본 연구는 다음 값을 정의한다.

```text
e_theta(u) = ell_theta(u) - ell_ref(u)
```

이를 **reference-relative excess loss**라고 부른다. 코드에서는 이 값이 `rho`로
구현되어 있다.

```text
rho[u] = current_loss[u] - ref_loss[u]
```

### 2.2 Excess risk와의 관계

데이터 분포에서 `u ~ D`라고 하면 기대값은 다음과 같다.

```text
E_D[e_theta(u)]
= E_D[ell_theta(u) - ell_ref(u)]
= R(theta) - R(ref)
```

즉 `rho`의 기대값은 reference model 대비 현재 foundation model의 excess risk이다.
고전적인 excess risk가 Bayes optimal risk `R*` 대비 `R(theta) - R*`를 뜻한다면,
여기서는 Bayes optimal 대신 고정된 DLinear reference를 기준점으로 둔다.

따라서 엄밀한 명칭은 다음이 가장 적절하다.

```text
reference-relative excess loss
DLinear-relative excess loss
```

### 2.3 NLL objective에서의 해석

Moirai처럼 probabilistic NLL을 쓰는 경우:

```text
ell_theta(u) = -log p_theta(y | x)
ell_ref(u)   = -log q_ref(y | x)
```

그러면:

```text
e_theta(u)
= -log p_theta(y | x) + log q_ref(y | x)
= log(q_ref(y | x) / p_theta(y | x))
```

기대값을 취하면:

```text
E[e_theta]
= CE(p*, p_theta) - CE(p*, q_ref)
= KL(p* || p_theta) - KL(p* || q_ref)
```

여기서 `p*`는 실제 데이터 생성 분포이다. 즉 Moirai의 `rho`는 reference 대비
excess cross-entropy 또는 excess KL로 해석할 수 있다.

### 2.4 MSE objective에서의 해석

MOMENT와 Timer처럼 MSE를 쓰는 경우:

```text
ell_theta(u) = ||y_u - f_theta(x_u)||^2
ell_ref(u)   = ||y_u - g_ref(x_u)||^2
```

따라서:

```text
e_theta(u)
= ||y_u - f_theta(x_u)||^2 - ||y_u - g_ref(x_u)||^2
```

이 값이 낮으면 현재 모델이 reference만큼 또는 reference보다 잘 맞추는 단위이고,
높으면 현재 모델이 reference보다 더 어려워하는 단위이다.

---

## 3. 방법: Selective Pretraining Masking

### 3.1 Batch-level masking rule

각 batch에서 valid learning units 집합을 `B`라고 하자. 각 단위에 대해
`e_theta(u)`를 계산한 뒤, 하위 `drop_pct%` quantile을 threshold로 둔다.

```text
tau = Quantile({e_theta(u) | u in B}, drop_pct / 100)
keep(u) = 1[e_theta(u) >= tau]
```

SPM loss는 다음과 같다.

```text
L_SPM(theta)
= sum_{u in B} keep(u) * ell_theta(u)
  / sum_{u in B} keep(u)
```

이때 `keep(u)=0`인 단위는 현재 step에서 gradient를 만들지 않는다.

### 3.2 직관

- `e_theta(u)`가 낮은 단위: 현재 모델이 reference와 비슷하거나 더 잘 처리한다.
- `e_theta(u)`가 높은 단위: 현재 모델이 reference보다 아직 못 처리한다.
- SPM은 낮은 excess loss 단위를 제거하고 높은 excess loss 단위를 남긴다.

즉, 단순히 loss가 큰 hard example을 고르는 것이 아니라, **reference 대비 아직
모델에 headroom이 남아 있는 단위**를 고른다. 이 차이가 중요하다. 원래 loss만
사용하면 scale이 큰 channel이나 noisy한 구간이 과도하게 선택될 수 있다. Reference
loss를 빼면 단순한 scale effect와 쉬운 선형 구조를 어느 정도 제거할 수 있다.

### 3.3 알고리즘

```text
Input:
  dataset D
  foundation model f_theta
  reference model g_ref
  drop percentage q

Step 1. Train reference model g_ref briefly on D.

Step 2. For each pretraining batch:
  1. Run f_theta forward once.
  2. Compute current loss ell_theta(u) at the backbone's natural unit.
  3. Compute or look up reference loss ell_ref(u).
  4. Compute excess loss e_theta(u) = ell_theta(u) - ell_ref(u).
  5. Drop bottom q percent of e_theta(u).
  6. Backpropagate only the kept units.
```

구현상 중요한 점은 current loss를 위한 forward와 rho 계산을 위한 forward를 분리하지
않는 것이다. Moirai 구현에서는 하나의 forward output distribution을 rho 계산과
최종 loss 계산에 모두 재사용한다.

---

## 4. 세 모델별 구현

### 4.1 전체 비교

| 항목 | Moirai-small | MOMENT-small | Timer-base |
|---|---|---|---|
| 모델 구조 | probabilistic Transformer, packed token format | Transformer 기반 masked autoencoder | decoder-only causal Transformer |
| 파라미터 수 | 약 14M | 약 37M | 약 84M |
| 입력 길이 | `SEQ_LEN=512` | `SEQ_LEN=512` | `SEQ_LEN=768` |
| patch 설정 | `{8,16,32,64,128}` 중 sample별 sampling | `PATCH_LEN=8`, 64 patches | `PATCH_LEN=96`, 8 patches |
| 기본 objective | suffix masked patch NLL | random 30% patch reconstruction MSE | autoregressive next-patch MSE |
| SPM 단위 | packed prediction token | sample x channel x patch | sample x channel x position |
| reference | DLinear Normal NLL table | DLinear masked reconstruction model | DLinear causal next-patch MSE table |
| 표준화 | Moirai internal scaler, raw input | per-series z-score | per-series z-score + Timer internal norm |
| 평가 | zero-shot, linear probe | linear probe | zero-shot autoregressive |

### 4.2 Moirai-small

Moirai는 probabilistic forecasting 계열 모델이다. 구현은 `MoiraiModule`을 직접
로드하며, `uni2ts.model.moirai.__init__`에서 발생하는 Lightning/torchvision import
충돌을 피하기 위해 `module.py`만 직접 import한다.

구조 설정:

```text
D_MODEL     = 384
NUM_LAYERS  = 6
PATCH_SIZES = (8, 16, 32, 64, 128)
MAX_SEQ_LEN = 512
MAX_DIM     = 128
```

Distribution output은 Moirai-small recipe와 맞춘 4-component mixture이다.

```text
StudentT
NormalFixedScale
NegativeBinomial
LogNormal
```

#### Baseline pretraining

Moirai baseline은 forecasting-style suffix masked patch prediction을 수행한다.

1. `(B, C, T)` batch를 Moirai packed format으로 변환한다.
2. sample마다 patch size를 `{8,16,32,64,128}`에서 선택한다.
3. suffix 영역을 prediction mask로 둔다.
4. prediction token에 대해 mixture NLL을 계산한다.

Loss:

```text
L_base = mean NLL over prediction tokens and valid patch positions
```

#### SPM pretraining

Moirai의 SPM은 token-level이다. 학습 단위는 다음이다.

```text
u = (sample, original_channel, time_patch)
```

Moirai는 모델 입력으로 randomized variate id를 사용하지만, reference lookup에는 원래
channel index가 필요하다. 그래서 packed dict에 `_orig_channel_id_flat`을 별도로
저장한다.

Reference table:

```text
ref_loss_table shape = (N_windows, C99, SEQ_LEN)
```

Reference는 time-step 단위 NLL을 저장한다. Moirai는 sample마다 patch size가 바뀌므로,
training step에서 해당 token의 실제 patch 구간에 맞게 reference NLL을 평균낸다.

```text
ref_token_loss[u]
= mean ref_loss_table[window, channel, t_start:t_end]
```

Current token NLL:

```text
current_token_loss[u]
= mean -log p_theta(target_patch | context)
```

Excess loss:

```text
excess_loss[u] = current_token_loss[u] - ref_token_loss[u]
```

그 뒤 bottom `drop_pct%` prediction tokens를 제거한다.

### 4.3 MOMENT-small

MOMENT는 masked autoencoding 방식이다. 구현은 `MOMENTPipeline.from_pretrained`로
architecture를 불러온 뒤, weight를 reset하여 random initialization에서 시작한다.

기본 설정:

```text
SEQ_LEN   = 512
PATCH_LEN = 8
N_PATCHES = 64
mask_ratio = 0.3
```

#### Baseline pretraining

MOMENT baseline은 random 30% patch mask를 만들고, masked timepoints에 대해서만
reconstruction MSE를 계산한다.

```text
L_base = mean reconstruction MSE on masked positions
```

Channel 수가 큰 series를 처리하기 위해 forward는 channel chunk 단위로 나누어 수행한다.

#### SPM pretraining

MOMENT의 SPM은 patch-level이다.

```text
u = (sample, channel, patch)
```

Current loss:

```text
current_patch_loss[u]
= mean squared reconstruction error within patch
```

Reference는 DLinear masked reconstruction model이다. 중요한 점은 reference table을
고정해서 쓰지 않고, 매 batch에서 MOMENT가 실제로 사용한 mask를 reference에도 동일하게
적용한다는 것이다.

```text
ref_patch_loss[u]
= DLinear masked-recon MSE on the same masked patch
```

Excess loss:

```text
excess_loss[u] = current_patch_MSE[u] - ref_patch_MSE[u]
```

Valid unit은 실제 channel이고, MOMENT mask에 의해 masked된 patch여야 한다.

```text
valid(u) = channel_is_real and patch_is_masked
```

SPM은 valid patch 중 bottom `drop_pct%`를 제거한다.

### 4.4 Timer-base

Timer는 channel-independent decoder-only causal Transformer이다. 각 `(sample, channel)`을
독립적인 univariate patch sequence로 보고, causal attention으로 다음 patch를 예측한다.

구조 설정:

```text
PATCH_LEN   = 96
TOKEN_NUM   = 8
SEQ_LEN     = 768
D_MODEL     = 1024
NUM_HEADS   = 8
NUM_LAYERS  = 8
D_FF        = 2048
```

#### Baseline pretraining

입력 시계열을 8개 patch로 나누고, position `t`에서 patch `t+1`을 예측한다.

```text
mse_pos[t] = mean((pred[t] - patch[t+1])^2)
L_base = mean mse_pos over real sample-channel pairs and positions
```

#### SPM pretraining

Timer의 SPM은 autoregressive token 또는 position-level이다.

```text
u = (sample, channel, position)
```

Reference는 causal DLinear이다. `seq_len - patch_len` 길이의 context를 사용해 다음
patch를 예측하며, 모든 autoregressive position에 대한 MSE table을 만든다.

```text
ref_loss_table shape = (N_windows, C99, n_pos)
n_pos = n_patches - 1 = 7
```

Current loss:

```text
current_loss[u] = next_patch_MSE[u]
```

Excess loss:

```text
excess_loss[u] = current_next_patch_MSE[u] - ref_next_patch_MSE[u]
```

SPM은 reference entry가 있는 channel에 대해 bottom `drop_pct%` positions를 제거한다.

---

## 5. 데이터와 학습 설정

### 5.1 Pretraining data

Pretraining corpus는 UTSD-12G이다. 데이터는 HuggingFace `load_from_disk` 형식으로
`data/utsd_repo/UTSD-12G` 아래에 있다고 가정한다.

공통 처리:

- series를 `(C, T)` 배열로 그룹화한다.
- `seq_len` 길이의 non-overlapping windows를 만든다.
- batch 안에서 channel 수가 다르면 zero padding하고 `ch_mask`를 만든다.
- channel 수가 너무 큰 series는 percentile cap으로 random subsampling한다.

모델별 차이:

| Model | `seq_len` | stride | standardization | channel cap |
|---|---:|---:|---|---|
| Moirai | 512 | 512 | raw input, internal scaler 사용 | p99를 계산하되 packed attention 때문에 최대 32 |
| MOMENT | 512 | 512 | per-series z-score | p99 |
| Timer | 768 | 768 | per-series z-score | p99 |

### 5.2 Reference training

Reference는 모두 작은 DLinear 계열 모델이다. 목적은 strong teacher를 만드는 것이
아니라, 단순 선형 모델이 이미 설명 가능한 쉬운 단위를 식별하는 것이다.

| Model | Reference | Training target | Output |
|---|---|---|---|
| Moirai | DLinear Normal NLL | self-standardized window reconstruction NLL | `(N, C99, T)` time-step NLL table |
| MOMENT | DLinear masked recon | masked patch reconstruction MSE | live reference model, batch mask와 동일 적용 |
| Timer | DLinear causal | causal next-patch MSE | `(N, C99, n_pos)` position MSE table |

Moirai에는 alternative로 4-component mixture forecasting reference 구현도 존재한다
(`dlinear_mixture_forecast.py`). 기본 결과에서는 legacy Normal NLL reference가 사용된다.

---

## 6. 실험 결과

### 6.1 Moirai-small 결과

Moirai는 `max_series=10000`, 6 datasets x 4 horizons zero-shot forecasting으로
평가되었다.

| Config | Baseline MSE | SPM MSE | Delta MSE | 결과 |
|---|---:|---:|---:|---|
| `lr=1e-4, e=2, drop=10` | 0.5376 | 0.5178 | -0.0198 (-3.7%) | SPM win |
| `lr=3e-4, e=2, drop=20` | 0.5288 | 0.5210 | -0.0079 (-1.5%) | SPM win |
| `lr=3e-4, e=1, drop=10` | 0.5245 | 0.5342 | +0.0097 | baseline win |
| `lr=3e-4, e=2, drop=10` | 0.5288 | 0.5445 | +0.0157 | baseline win |
| `lr=5e-4, e=2, drop=10` | 0.5140 | 0.5330 | +0.0190 | baseline win |
| `lr=3e-4, e=2, drop=5` | 0.5288 | 0.5612 | +0.0323 | baseline win |

핵심 관찰:

- 가장 좋은 SPM 결과는 `lr=1e-4, epochs=2, drop_pct=10`이다.
- 이 설정에서 SPM은 24개 dataset-horizon 조합 중 20개에서 baseline을 이겼다.
- `lr >= 1e-3`에서는 mixture NLL head가 불안정해져 NaN이 발생했다.
- Moirai에서는 낮은 learning rate와 적절한 drop ratio에서 SPM이 효과적이었다.

### 6.2 Timer-base 결과

Timer는 먼저 `max_series=10000` 제한 실험에서 6 datasets x 4 horizons zero-shot
forecasting으로 평가되었다.

| Config | Baseline MSE | SPM MSE | Delta MSE | 결과 |
|---|---:|---:|---:|---|
| `lr=3e-4, e=2, drop=10` | 0.7149 | 0.7009 | -0.0140 (-2.0%) | SPM win |
| `lr=3e-4, e=2, drop=20` | 0.7149 | 0.6937 | -0.0212 (-3.0%) | SPM win |
| `lr=1e-4, e=2, drop=10` | 0.7282 | 0.7152 | -0.0129 (-1.8%) | SPM win |
| `lr=1e-4, e=2, drop=20` | 0.7282 | 0.7055 | -0.0227 (-3.1%) | SPM win |
| `lr=3e-4, e=1, drop=10` | 0.6249 | 0.6728 | +0.0479 (+7.7%) | baseline win |

10k-series 제한 실험에서는 SPM이 5개 stable config 중 4개에서 이겼다. 특히
`drop_pct=20`이 `drop_pct=10`보다 일관되게 좋았다.

다만 full UTSD 재검증에서는 결론이 달라졌다.

| Scale | Baseline MSE | SPM MSE | Delta MSE | SPM wins |
|---|---:|---:|---:|---:|
| max-series=10000 | 0.7149 | 0.6937 | -0.0212 | 19 / 24 |
| full UTSD | 0.5952 | 0.5998 | +0.0046 | 0 / 24 |

해석:

- 제한된 data/compute regime에서는 SPM이 hard-unit focusing 효과를 낸다.
- full-scale에서는 baseline 자체가 충분히 좋아져서, bottom 20% token 제거가 정보 손실로
  작용한다.
- 따라서 Timer 결과는 "SPM이 항상 좋다"가 아니라, "SPM은 under-trained 또는
  compute-limited regime에서 overfitting/inefficient fitting을 완화할 수 있다"로
  해석해야 한다.

### 6.3 MOMENT-small 결과

MOMENT는 linear probe forecasting으로 평가되었고, 3 seeds x 15 settings paired
comparison이 수행되었다.

| Statistic | MSE | MAE |
|---|---:|---:|
| Mean Delta, SPM - baseline | -0.00103 | -0.00137 |
| Paired t-test t | -1.094 | -2.268 |
| Paired t-test p | 0.28 | 0.028 |
| SPM better count | 26 / 45 | 28 / 45 |

해석:

- MSE 개선은 방향은 일관되지만 통계적으로 강하지 않다.
- MAE는 p=0.028로 유의한 개선을 보인다.
- MOMENT에서는 평균 성능 개선보다 seed variance 감소가 더 뚜렷했다.

Seed variance:

| Mode | n_seed | MSE mean +/- std | MAE mean +/- std |
|---|---:|---:|---:|
| baseline | 3 | 0.3173 +/- 0.0019 | 0.3554 +/- 0.0017 |
| patch SPM | 3 | 0.3165 +/- 0.0013 | 0.3543 +/- 0.0005 |

SPM은 MOMENT에서 MSE std를 약 32%, MAE std를 약 71% 줄였다. 이는 쉬운 단위에서
발생하는 seed-dependent gradient variation을 줄이고, 더 안정적인 hard-unit 중심
trajectory를 만든다는 해석과 맞는다.

---

## 7. 종합 해석

### 7.1 SPM이 잘 작동하는 조건

현재 결과를 종합하면 SPM은 다음 조건에서 효과적이다.

| 조건 | 설명 |
|---|---|
| data/compute-limited regime | baseline이 모든 단위를 충분히 학습하지 못하는 경우 hard-unit focusing이 유리하다. |
| 낮거나 안정적인 learning rate | Moirai에서는 `lr=1e-4`에서 가장 좋은 결과가 나왔다. |
| pretraining이 쉬운 단위에 과도하게 끌리는 경우 | Timer의 10k-series e=2 실험처럼 baseline이 easy-unit fitting에 치우칠 때 SPM이 완화한다. |
| objective와 masking granularity가 잘 맞는 경우 | Moirai는 token NLL, MOMENT는 patch MSE, Timer는 position MSE 단위가 자연스럽다. |

### 7.2 SPM이 실패하거나 약해지는 조건

| 조건 | 관찰 |
|---|---|
| under-training | Timer e=1에서는 baseline이 아직 모든 단위에서 학습 신호를 필요로 하므로 SPM이 손해였다. |
| full-scale 충분 학습 | Timer full UTSD에서는 baseline이 더 좋아지고 SPM은 약하게 손해였다. |
| 과도한 drop ratio | useful easy units까지 제거할 수 있다. |
| reference mismatch | reference objective가 backbone objective와 다르면 excess loss ranking이 왜곡될 수 있다. |
| probabilistic head instability | Moirai mixture NLL은 high lr에서 NaN이 발생했다. |

### 7.3 Backbone별 결론

| Backbone | 현재 결론 |
|---|---|
| Moirai | 10k-series 제한 실험에서 SPM 효과가 가장 명확하다. Best config에서 MSE -3.7%, 20/24 wins. |
| MOMENT | 평균 MSE 개선은 작지만 MAE 개선과 seed variance 감소가 의미 있다. |
| Timer | 10k-series에서는 강한 개선, full UTSD에서는 약한 손해. Compute-limited setting에서 유효한 방법으로 해석해야 한다. |

---

## 8. 연구 기여

이 연구의 기여는 다음과 같이 정리할 수 있다.

1. **Reference-relative excess loss 기반 selective pretraining**
   - 단순 loss가 아니라 `current_loss - reference_loss`를 사용해 학습 단위의 상대적
     어려움을 측정한다.

2. **Backbone-specific masking granularity**
   - Moirai, MOMENT, Timer의 loss aggregation 단위에 맞춰 token, patch, position 단위로
     SPM을 적용한다.

3. **Time-series foundation model pretraining에 대한 실증 분석**
   - 세 가지 서로 다른 구조의 시계열 foundation model에 같은 원리를 적용했다.

4. **Data/compute regime에 따른 효과 분석**
   - SPM이 compute-limited regime에서는 성능을 높일 수 있지만, full-scale 충분 학습에서는
     정보 손실이 될 수 있음을 Timer full UTSD 실험으로 확인했다.

5. **Seed variance 감소 관찰**
   - MOMENT에서 SPM이 평균 성능뿐 아니라 checkpoint reproducibility를 개선할 수 있음을
     보였다.

---

## 9. 한계

1. **Reference 학습 비용**
   - DLinear는 작지만 별도 reference training과 reference table 생성이 필요하다.

2. **Reference table memory**
   - Moirai의 `(N, C99, T)` table은 full corpus에서 메모리 부담이 크다.

3. **Full-scale generalization 미확정**
   - Moirai와 MOMENT는 full UTSD multi-seed scale에서 아직 충분히 검증되지 않았다.
   - Timer는 full UTSD에서 SPM이 약하게 손해였다.

4. **drop_pct schedule 부재**
   - 현재는 fixed `drop_pct`를 사용한다. 학습 초반에는 낮게, 후반에는 높이는 schedule이
     더 적절할 수 있다.

5. **Reference choice 제한**
   - 현재 reference는 DLinear 계열이다. 더 강한 reference를 쓰면 ranking이 달라질 수 있다.

6. **Theoretical guarantee의 범위**
   - `rho`가 reference-relative excess loss라는 점은 수학적으로 명확하다.
   - 하지만 bottom excess loss masking이 항상 generalization을 개선한다는 보장은 추가
     가정 없이는 어렵다.

---

## 10. 향후 연구 방향

### 10.1 Adaptive masking schedule

고정된 `drop_pct` 대신 학습 단계에 따라 masking strength를 조절할 수 있다.

```text
early training: drop_pct = 0 or small
middle training: drop_pct increases
late training: drop_pct tuned by validation or rho distribution
```

이는 Timer full-scale 결과에서 보인 "충분한 data regime에서는 정보 손실" 문제를 완화할
수 있다.

### 10.2 Reference calibration

Reference loss scale이 backbone loss scale과 잘 맞는지 보정하는 것이 중요하다.

가능한 방법:

- z-score normalized excess loss
- per-channel quantile normalization
- moving average reference calibration
- objective-matched reference, 예를 들어 Moirai mixture NLL reference

### 10.3 Soft weighting

현재는 hard mask이다.

```text
weight(u) in {0, 1}
```

대안으로 soft weighting을 사용할 수 있다.

```text
weight(u) = sigmoid((excess_loss(u) - tau) / temperature)
```

Soft weighting은 useful easy units를 완전히 버리는 문제를 줄일 수 있다.

### 10.4 Granularity ablation

현재는 backbone의 자연스러운 loss 단위에 맞췄다. 추가 ablation으로 다음이 가능하다.

- Moirai: token-level vs channel-level
- MOMENT: patch-level vs timepoint-level
- Timer: position-level vs channel-level

### 10.5 Downstream task 확장

현재는 forecasting 중심이다. SPM은 다음 task에도 확장 가능하다.

- imputation
- anomaly detection
- classification
- representation transfer
- few-shot forecasting

---

## 11. 논문용 핵심 문장 후보

### 11.1 Problem statement

> Uniform pretraining over large heterogeneous time-series corpora allocates
> equal gradient budget to both informative hard units and redundant easy units.
> We argue that this is inefficient for time-series foundation models, where many
> channels and temporal patches are already well explained by simple linear
> references.

### 11.2 Method statement

> We define a reference-relative excess loss for each learning unit as the
> difference between the current foundation model loss and a lightweight DLinear
> reference loss. Selective Pretraining Masking drops the bottom quantile of this
> excess loss within each batch, focusing optimization on units where the
> foundation model still underperforms the reference.

### 11.3 Contribution statement

> Across Moirai, MOMENT, and Timer, we instantiate the same excess-loss principle
> at each backbone's natural loss granularity: packed probabilistic tokens for
> Moirai, reconstruction patches for MOMENT, and autoregressive positions for
> Timer.

### 11.4 Caveat statement

> The benefit of selective masking is regime-dependent. It improves
> data- or compute-limited pretraining, but can become mild data dropout when the
> baseline is already trained at sufficient scale.

---

## 12. 재현 명령

### Moirai

```bash
python scripts/pretrain_moirai.py --mode baseline --epochs 2 --batch-size 32

python scripts/pretrain_moirai.py --mode rho_cm \
  --epochs 2 \
  --batch-size 32 \
  --ref-epochs 3 \
  --drop-pct 10
```

### MOMENT

```bash
python scripts/pretrain_moment.py --mode baseline --epochs 2 --batch-size 64

python scripts/pretrain_moment.py --mode patch_rho_cm \
  --epochs 2 \
  --batch-size 64 \
  --ref-epochs 3 \
  --drop-pct 10
```

### Timer

```bash
python scripts/pretrain_timer.py --mode baseline --epochs 10 --batch-size 32

python scripts/pretrain_timer.py --mode rho_cm \
  --epochs 10 \
  --batch-size 32 \
  --ref-epochs 10 \
  --drop-pct 10
```

### Evaluation sweep

```bash
python scripts/pretrain_moirai.py --mode eval_zero_shot_sweep \
  --baseline-ckpt results_moirai/baseline/best.pt \
  --rho-cm-ckpt results_moirai/rho_cm/best.pt

python scripts/pretrain_moment.py --mode eval_sweep \
  --baseline-ckpt results_pretrain/baseline/best.pt \
  --patch-rho-cm-ckpt results_pretrain/patch_rho_cm/best.pt

python scripts/pretrain_timer.py --mode eval_zero_shot_sweep \
  --baseline-ckpt results_timer/baseline/best.pt \
  --rho-cm-ckpt results_timer/rho_cm/best.pt
```

---

## 13. 최종 정리

Selective Pretraining Masking은 시계열 foundation model pretraining에서 각 학습
단위의 중요도를 **reference-relative excess loss**로 측정하고, 쉬운 단위를 선택적으로
제거하는 방법이다.

이 연구의 가장 중요한 메시지는 두 가지다.

1. `current_loss - reference_loss`는 단순 heuristic이 아니라 reference 대비 excess
   loss로 해석할 수 있으며, NLL에서는 reference 대비 excess KL, MSE에서는 reference
   대비 excess squared error risk와 연결된다.

2. SPM의 효과는 학습 regime에 의존한다. 제한된 data/compute에서는 hard-unit focusing으로
   성능을 개선할 수 있지만, 충분한 full-scale 학습에서는 useful easy units까지 제거해
   약한 손해가 날 수 있다.

따라서 이 방법은 "항상 좋은 pretraining objective"라기보다, **학습 예산이 제한된
상황에서 gradient budget을 더 효율적으로 배분하는 selective pretraining strategy**로
정의하는 것이 가장 정확하다.
