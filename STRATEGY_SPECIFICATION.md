# KOSPI-S&P500 베타 잔차 평균 회귀 전략 기술 명세서

**버전**: 2.0 (Grid Search 최적화 반영)
**최종 수정일**: 2025-12-31
**적용 파일**: `base_longterm.py`

---

## 📋 목차

1. [전략 개요](#1-전략-개요)
2. [데이터 요구사항](#2-데이터-요구사항)
3. [전략 로직](#3-전략-로직)
4. [파라미터 설정](#4-파라미터-설정)
5. [구현 가이드](#5-구현-가이드)
6. [성과 지표](#6-성과-지표)
7. [코드 재현 프롬프트](#7-코드-재현-프롬프트)

---

## 1. 전략 개요

### 1.1 전략 철학
KOSPI와 S&P500의 동행성(베타)을 기반으로, KOSPI가 S&P500 대비 **과도하게 움직였을 때 평균 회귀**를 기대하는 통계적 차익거래 전략입니다.

### 1.2 핵심 가정
- KOSPI와 S&P500는 장기적으로 일정한 베타 관계 유지
- 단기적으로 베타에서 벗어난 잔차는 평균으로 회귀
- 극단적 변동성 구간에서는 평균 회귀 가설이 약화

### 1.3 거래 방향
- **Long (매수)**: KOSPI가 S&P500 대비 **저평가**되었을 때 (Z-Score ≤ -2.15)
- **Short (매도)**: KOSPI가 S&P500 대비 **고평가**되었을 때 (Z-Score ≥ +2.15)

---

## 2. 데이터 요구사항

### 2.1 필수 컬럼

| 컬럼명 | 설명 | 데이터 타입 | 비고 |
|--------|------|------------|------|
| `공통날짜` | 거래일 | datetime | YYYY-MM-DD 형식 |
| `kospi_t` | KOSPI 종가 (당일) | float | 1,000 단위 |
| `SPX_t-1` | S&P500 종가 (전일) | float | 시차 보정 필수 |
| `VIX_t-1` | VIX 지수 (전일) | float | 변동성 필터용 |
| `FX_t` | 원/달러 환율 (당일) | float | 환율 필터용 |

### 2.2 데이터 전처리 규칙

```python
# 1. 날짜 변환
df["공통날짜"] = pd.to_datetime(df["공통날짜"])

# 2. 숫자 변환 (천단위 콤마 제거)
for col in ["kospi_t", "SPX_t-1", "VIX_t-1", "FX_t"]:
    df[col] = pd.to_numeric(df[col].str.replace(",", ""), errors='coerce')

# 3. 결측치 처리 (Forward Fill)
df = df.sort_values("공통날짜").set_index("공통날짜").ffill().dropna()
```

**중요**: 결측치는 **ffill(앞 값으로 채우기)** 사용. 이는 실제 거래에서 마지막 관측값을 사용하는 것과 동일.

---

## 3. 전략 로직

### 3.1 로그 수익률 계산

```python
# KOSPI 로그 수익률
rK = log(kospi_t / kospi_t-1)

# S&P500 로그 수익률 (전일 데이터 사용)
rS = log(SPX_t-1 / SPX_t-2)

# 환율 로그 수익률
rFX = log(FX_t / FX_t-1)
```

**구현 코드**:
```python
out["rK"] = np.log(out["kospi_t"]).diff()
out["rS"] = np.log(out["SPX_t-1"]).diff()
out["rFX"] = np.log(out["FX_t"]).diff()
```

---

### 3.2 Rolling Beta 계산

**정의**: 최근 60일간 KOSPI와 S&P500의 공분산 / S&P500 분산

```python
beta_t = Cov(rK, rS, window=60) / Var(rS, window=60)
```

**구현 코드**:
```python
BETA_W = 60
out["beta"] = out["rK"].rolling(BETA_W).cov(out["rS"]) / out["rS"].rolling(BETA_W).var()
```

**해석**:
- beta = 1.2 → KOSPI는 S&P500보다 20% 더 민감
- beta = 0.8 → KOSPI는 S&P500보다 20% 덜 민감

---

### 3.3 잔차(Residual) 계산

**정의**: 실제 KOSPI 수익률 - 베타 예측 수익률

```python
resid_t = rK_t - beta_t × rS_t
```

**구현 코드**:
```python
out["resid"] = out["rK"] - out["beta"] * out["rS"]
```

**해석**:
- resid > 0 → KOSPI가 예상보다 **상승** (고평가)
- resid < 0 → KOSPI가 예상보다 **하락** (저평가)

---

### 3.4 Z-Score 계산 (Look-ahead Bias 제거)

**정의**: 잔차를 과거 60일 분포로 표준화

```python
# [중요] shift(1)으로 미래 정보 사용 방지
resid_mean_t = mean(resid, window=60).shift(1)
resid_std_t = std(resid, window=60).shift(1)

z_t = (resid_t - resid_mean_t) / resid_std_t
```

**구현 코드**:
```python
RES_W = 60
out["resid_mean"] = out["resid"].rolling(RES_W).mean().shift(1)
out["resid_std"] = out["resid"].rolling(RES_W).std().shift(1)
out["z"] = (out["resid"] - out["resid_mean"]) / out["resid_std"]
```

**중요 원칙**:
- **shift(1) 필수**: t일의 Z-Score는 t-1일까지의 통계량으로만 계산
- 이를 빠뜨리면 **Look-ahead Bias** 발생 (미래 정보 사용)

**해석**:
- Z = +2.0 → 잔차가 평균보다 2 표준편차 위 (고평가)
- Z = -2.0 → 잔차가 평균보다 2 표준편차 아래 (저평가)

---

### 3.5 리스크 필터

#### 3.5.1 VIX 필터

**목적**: 변동성이 과도하게 높은 구간 차단

```python
# VIX의 과거 252일(1년) 대비 순위 (백분위)
vix_rank_t = percentile_rank(VIX_t-1, window=252).shift(1)

# 필터: VIX가 과거 1년 대비 상위 6% 초과 시 거래 금지
allow_vix = (vix_rank_t <= 0.94)
```

**구현 코드**:
```python
FILTER_W = 252
out["vix_rank"] = out["VIX_t-1"].rolling(FILTER_W).rank(pct=True).shift(1)
allow_vix = (out["vix_rank"] <= 0.94)
```

#### 3.5.2 FX 충격 필터

**목적**: 환율 급변동 구간 차단

```python
# 환율 수익률의 Z-Score
fx_mean = mean(rFX, window=252)
fx_std = std(rFX, window=252)
fx_z = (rFX - fx_mean) / fx_std

# FX 충격의 과거 252일 대비 순위
fx_shock_t = percentile_rank(|fx_z|, window=252).shift(1)

# 필터: FX 충격이 과거 1년 대비 상위 4% 초과 시 거래 금지
allow_fx = (fx_shock_t <= 0.96)
```

**구현 코드**:
```python
out["fx_mean"] = out["rFX"].rolling(FILTER_W).mean()
out["fx_std"] = out["rFX"].rolling(FILTER_W).std()
out["fx_z"] = (out["rFX"] - out["fx_mean"]) / out["fx_std"]
out["fx_shock"] = out["fx_z"].abs().rolling(FILTER_W).rank(pct=True).shift(1)
allow_fx = (out["fx_shock"] <= 0.96)
```

#### 3.5.3 통합 필터

```python
allow = allow_vix AND allow_fx
```

---

### 3.6 신호 생성 로직 (State Machine)

```python
# 상태 변수
current_pos = 0  # -1(Short), 0(Neutral), +1(Long)

for each day t:
    # [1단계] 필터 체크
    if NOT allow[t]:
        current_pos = 0  # 강제 청산
        continue

    if z[t] is NaN:
        current_pos = 0
        continue

    # [2단계] 손절매 체크
    if current_pos != 0:
        if |z[t]| >= 7.095:  # Entry(2.15) × 3.3
            current_pos = 0  # 강제 청산
            continue

    # [3단계] 진입/청산 로직
    if current_pos == 0:
        # 진입 조건
        if z[t] <= -2.15:
            current_pos = +1  # Long 진입
        elif z[t] >= +2.15:
            current_pos = -1  # Short 진입
    else:
        # 청산 조건
        if |z[t]| <= 0.0:
            current_pos = 0  # 평균 회귀 시 청산

    position[t] = current_pos
```

**구현 코드**:
```python
pos = np.zeros(len(df))
z = df["z"].values
allow = df["allow"].values

ENTRY_T = 2.15
EXIT_T = 0.0
STOP_LOSS = 7.095

current_pos = 0

for i in range(1, len(df)):
    # 필터 체크
    if not allow[i]:
        current_pos = 0
        pos[i] = 0
        continue

    if np.isnan(z[i]):
        pos[i] = 0
        continue

    # 손절매
    if current_pos != 0:
        if abs(z[i]) >= STOP_LOSS:
            current_pos = 0
            pos[i] = 0
            continue

    # 진입/청산
    if current_pos == 0:
        if z[i] <= -ENTRY_T:
            current_pos = +1
        elif z[i] >= +ENTRY_T:
            current_pos = -1
    else:
        if abs(z[i]) <= EXIT_T:
            current_pos = 0

    pos[i] = current_pos
```

---

### 3.7 수익률 계산

```python
# 전략 수익률 (t-1일 포지션 × t일 KOSPI 수익률)
strategy_ret[t] = position[t-1] × rK[t]

# 거래 비용 (포지션 변화 × 0.02%)
turnover[t] = |position[t] - position[t-1]|
transaction_cost[t] = turnover[t] × 0.0002

# 순수익률
strategy_ret_net[t] = strategy_ret[t] - transaction_cost[t]

# 누적 수익률 (Equity Curve)
equity[t] = ∏(1 + strategy_ret_net[i]) for i in [1, t]
```

**구현 코드**:
```python
out["strategy_ret"] = out["pos"].shift(1) * out["rK"]

TC = 0.0002
out["turnover"] = out["pos"].diff().abs()
out["strategy_ret_net"] = out["strategy_ret"] - TC * out["turnover"]
out["equity"] = (1 + out["strategy_ret_net"].fillna(0)).cumprod()
```

---

## 4. 파라미터 설정

### 4.1 최적화 파라미터 (Grid Search 결과)

| 파라미터 | 값 | 설명 | 최적화 방법 |
|---------|-----|------|------------|
| `BETA_WINDOW` | 60 | 베타 계산 롤링 윈도우 (일) | 고정 |
| `RESID_WINDOW` | 60 | 잔차 Z-Score 롤링 윈도우 (일) | 고정 |
| `FILTER_WINDOW` | 252 | 필터 계산 롤링 윈도우 (일, 1년) | 고정 |
| `ENTRY_THRESHOLD` | **2.15** | Z-Score 진입 기준 | Grid Search |
| `EXIT_THRESHOLD` | **0.0** | Z-Score 청산 기준 | Grid Search |
| `STOP_LOSS_MULTIPLIER` | **3.3** | 손절매 승수 (Z=7.095) | Grid Search |
| `VIX_QUANTILE` | **0.94** | VIX 필터 분위수 | Grid Search |
| `FX_QUANTILE` | **0.96** | FX 필터 분위수 | Grid Search |
| `TRANSACTION_COST` | 0.0002 | 편도 거래 비용 (0.02%) | 고정 |

### 4.2 파라미터 클래스 구조

```python
class StrategyParams:
    # Window Sizes
    BETA_WINDOW = 60
    RESID_WINDOW = 60
    FILTER_WINDOW = 252

    # Entry/Exit Thresholds
    ENTRY_THRESHOLD = 2.15
    EXIT_THRESHOLD = 0.0
    STOP_LOSS_MULTIPLIER = 3.3

    # Risk Filters
    VIX_QUANTILE = 0.94
    FX_QUANTILE = 0.96

    # Transaction Cost
    TRANSACTION_COST = 0.0002
```

---

## 5. 구현 가이드

### 5.1 Look-ahead Bias 방지 체크리스트

✅ **필수 적용 사항**:

1. **Z-Score 계산**:
   ```python
   # ✅ 올바른 방법
   resid_mean = resid.rolling(60).mean().shift(1)

   # ❌ 잘못된 방법 (미래 정보 포함)
   resid_mean = resid.rolling(60).mean()
   ```

2. **VIX Rank 계산**:
   ```python
   # ✅ 올바른 방법
   vix_rank = VIX.rolling(252).rank(pct=True).shift(1)

   # ❌ 잘못된 방법
   vix_rank = VIX.rolling(252).rank(pct=True)
   ```

3. **FX Shock 계산**:
   ```python
   # ✅ 올바른 방법
   fx_shock = fx_z.abs().rolling(252).rank(pct=True).shift(1)

   # ❌ 잘못된 방법
   fx_shock = fx_z.abs().rolling(252).rank(pct=True)
   ```

**원칙**: 모든 rolling 통계량은 **현재 시점을 포함하므로**, 신호 생성에 사용할 때는 반드시 `.shift(1)` 적용.

---

### 5.2 포지션 로직 구현 주의사항

#### 1. **필터 우선순위**
```python
# 순서 중요!
1순위: 필터 체크 (allow == False → 강제 청산)
2순위: 손절매 체크 (|Z| >= 7.095 → 강제 청산)
3순위: 진입/청산 로직
```

#### 2. **포지션 상태 유지**
```python
# ✅ 올바른 방법: current_pos 변수로 상태 유지
current_pos = 0
for i in range(1, len(df)):
    pos[i] = current_pos  # 기본값: 이전 상태 유지

    if some_condition:
        current_pos = 1

    pos[i] = current_pos

# ❌ 잘못된 방법: 매번 pos[i-1] 참조 (느리고 복잡)
for i in range(1, len(df)):
    if some_condition:
        pos[i] = 1
    else:
        pos[i] = pos[i-1]
```

#### 3. **NaN 처리**
```python
# Z-Score가 NaN일 때는 포지션 0으로 설정
if np.isnan(z[i]):
    pos[i] = 0
    continue
```

---

### 5.3 성과 지표 계산

```python
def compute_summary(df):
    ann_factor = 252
    valid_ret = df["strategy_ret_net"].dropna()

    # 연환산 수익률/변동성
    ann_return = valid_ret.mean() * ann_factor
    ann_vol = valid_ret.std() * np.sqrt(ann_factor)

    # Sharpe Ratio
    sharpe = ann_return / ann_vol if ann_vol > 0 else 0.0

    # MDD (Maximum Drawdown)
    equity = (1 + valid_ret).cumprod()
    mdd = (equity / equity.cummax() - 1).min()

    # Hit Ratio (승률)
    hit_ratio = (valid_ret > 0).mean()

    return {
        "ann_return": ann_return,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "mdd": mdd,
        "hit_ratio": hit_ratio
    }
```

---

## 6. 성과 지표

### 6.1 백테스트 결과 (2000-2025, 26년)

| 지표 | 값 | 업계 기준 |
|------|-----|----------|
| 연환산 수익률 | **8.39%** | 5~10% (시장 중립) |
| 연환산 변동성 | **11.20%** | 10~15% (중위험) |
| Sharpe Ratio | **0.75** | > 0.5 (양호) |
| MDD | **-15.72%** | < -20% (양호) |
| 누적 수익률 | **576.80%** | - |
| Hit Ratio | **19.15%** | - |

### 6.2 거래 활동 통계

- **총 거래 횟수**: 198회 (26년)
- **연평균 거래**: 7.6회
- **포지션 보유율**: 35.6%
- **Long/Short 비율**: 50.0% / 50.0% (균형)

---

## 7. 코드 재현 프롬프트

### 7.1 전체 구조 재현 프롬프트

```
다음 명세를 따라 KOSPI-S&P500 베타 잔차 평균 회귀 전략을 Python으로 구현해주세요.

[데이터 구조]
- 입력: Excel 파일 (컬럼: 공통날짜, kospi_t, SPX_t-1, VIX_t-1, FX_t)
- 전처리: 날짜 인덱스, ffill로 결측치 처리

[전략 로직]
1. 로그 수익률 계산 (rK, rS, rFX)
2. Rolling Beta 계산 (60일, rK와 rS의 공분산/분산)
3. 잔차 계산 (resid = rK - beta × rS)
4. Z-Score 계산 (60일 rolling, **shift(1) 필수**)
5. 필터:
   - VIX: 252일 rolling rank ≤ 0.94 (shift(1) 필수)
   - FX: |fx_z|의 252일 rolling rank ≤ 0.96 (shift(1) 필수)
6. 신호 생성:
   - 필터 차단 시 강제 청산
   - 손절매: |Z| ≥ 7.095 시 청산
   - 진입: Z ≤ -2.15 → Long, Z ≥ +2.15 → Short
   - 청산: |Z| ≤ 0.0 → 평균 회귀 시 청산
7. 수익률: pos.shift(1) × rK - 거래비용(0.02%)

[파라미터]
- BETA_WINDOW = 60
- RESID_WINDOW = 60
- FILTER_WINDOW = 252
- ENTRY_THRESHOLD = 2.15
- EXIT_THRESHOLD = 0.0
- STOP_LOSS_MULTIPLIER = 3.3
- VIX_QUANTILE = 0.94
- FX_QUANTILE = 0.96
- TRANSACTION_COST = 0.0002

[필수 구현 원칙]
1. Look-ahead Bias 방지: 모든 rolling 통계량에 shift(1) 적용
2. 포지션 로직: current_pos 변수로 상태 머신 구현
3. 필터 우선순위: 필터 → 손절매 → 진입/청산 순서 엄수

[출력]
- compute_summary() 함수로 Sharpe, MDD, 연환산 수익률 계산
- diagnostic_report() 함수로 거래 내역, 연도별 통계 출력
```

### 7.2 개별 함수 재현 프롬프트

#### 7.2.1 Z-Score 계산 함수

```
다음 사양에 따라 Z-Score 계산 함수를 작성해주세요:

입력: pandas DataFrame with columns ['resid']
출력: DataFrame with columns ['resid_mean', 'resid_std', 'z']

로직:
1. resid의 60일 rolling mean 계산 후 shift(1) (Look-ahead bias 방지)
2. resid의 60일 rolling std 계산 후 shift(1)
3. z = (resid - resid_mean) / resid_std

중요: shift(1)을 빠뜨리면 미래 정보를 사용하게 되어 백테스트가 무효화됩니다.
```

#### 7.2.2 신호 생성 함수

```
다음 사양에 따라 신호 생성 함수를 작성해주세요:

입력:
- z: Z-Score 배열 (numpy array)
- allow: 필터 통과 여부 (numpy array, True/False)
- ENTRY = 2.15, EXIT = 0.0, STOP_LOSS = 7.095

출력: position 배열 (0: 중립, +1: Long, -1: Short)

로직 (순차 처리):
1. 필터 체크: allow[i] == False → pos[i] = 0
2. 손절매: |z[i]| >= STOP_LOSS and pos[i-1] != 0 → pos[i] = 0
3. 진입 (pos[i-1] == 0일 때):
   - z[i] <= -ENTRY → pos[i] = +1
   - z[i] >= +ENTRY → pos[i] = -1
4. 청산 (pos[i-1] != 0일 때):
   - |z[i]| <= EXIT → pos[i] = 0

구현 방식: for loop with current_pos 상태 변수
```

#### 7.2.3 필터 계산 함수

```
다음 사양에 따라 리스크 필터를 계산해주세요:

입력: DataFrame with columns ['VIX_t-1', 'rFX']
출력: DataFrame with column ['allow'] (True/False)

로직:
1. VIX Rank:
   - VIX_t-1의 252일 rolling percentile rank 계산
   - shift(1) 적용
   - allow_vix = (vix_rank <= 0.94)

2. FX Shock:
   - rFX의 252일 rolling mean, std 계산
   - fx_z = (rFX - fx_mean) / fx_std
   - |fx_z|의 252일 rolling percentile rank 계산
   - shift(1) 적용
   - allow_fx = (fx_shock <= 0.96)

3. 통합: allow = allow_vix & allow_fx

중요: rolling().rank(pct=True)는 현재 값을 포함하므로 shift(1) 필수
```

---

## 8. 검증 체크리스트

구현 후 다음 항목을 검증하세요:

### 8.1 Look-ahead Bias 검증
```python
# resid_mean이 shift되었는지 확인
assert df["resid_mean"].isna().sum() > df["resid"].rolling(60).mean().isna().sum()

# vix_rank가 shift되었는지 확인
assert df["vix_rank"].isna().sum() > df["VIX_t-1"].rolling(252).rank(pct=True).isna().sum()
```

### 8.2 필터 동작 검증
```python
# 필터가 실제로 작동했는지 확인
assert (~df["allow"]).sum() > 0  # 최소 1번은 필터 발동

# 필터 차단 시 포지션이 0인지 확인
assert df.loc[~df["allow"], "pos"].abs().sum() == 0
```

### 8.3 성과 지표 검증 (26년 백테스트)
```python
# 예상 결과 (오차 ±5%)
assert 7.0 < ann_return < 9.5  # 8.39%
assert 0.65 < sharpe < 0.85    # 0.75
assert -18.0 < mdd < -13.0     # -15.72%
```

---

## 9. 참고 자료

- **원본 명세서**: 사용자 제공 전략 명세 문서
- **Grid Search 결과**: `grid_search_improved/20251231_130402/`
- **최적화 보고서**: `OPTIMIZATION_RESULTS.md`
- **사용 가이드**: `GRID_SEARCH_GUIDE.md`

---

**작성자**: Claude Code (Anthropic)
**라이선스**: 내부 사용 전용
**버전 이력**:
- v1.0 (2025-12-31): 초안 작성
- v2.0 (2025-12-31): Grid Search 최적화 반영
