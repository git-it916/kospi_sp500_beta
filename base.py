import pandas as pd
import numpy as np

# -----------------------------
# 1) Load
# -----------------------------
# 파일이 탭/복수 공백 섞여 있을 수 있어서 engine='python' + sep 지정
path = r"C:\Users\10845\OneDrive - 이지스자산운용\문서\kospi_sp500_filtered.xlsx"  # 같은 폴더에 두거나 전체 경로로 바꾸기
df = pd.read_excel(path)

# 컬럼 정리
df.columns = [c.strip() for c in df.columns]
df["공통날짜"] = pd.to_datetime(df["공통날짜"])

def to_float(x):
    # "6,609.3" 같은 콤마 제거
    if isinstance(x, str):
        x = x.replace(",", "").strip()
    return pd.to_numeric(x, errors="coerce")

for c in ["kospi_t", "SPX_t-1", "VIX_t-1", "FX_t"]:
    df[c] = df[c].apply(to_float)

df = df.sort_values("공통날짜").set_index("공통날짜").dropna()

# -----------------------------
# 2) Returns (log)
# -----------------------------
# KOSPI: t 기준 레벨 -> 일간 수익률
df["rK"] = np.log(df["kospi_t"]).diff()

# SPX_t-1 레벨로부터 "미국 전일 수익률" 생성:
# rS(t) = log(SPX_{t-1}/SPX_{t-2})
df["rS"] = np.log(df["SPX_t-1"]).diff()

# VIX는 레벨 자체가 필터에 유용. FX는 변화율도 필요.
df["rFX"] = np.log(df["FX_t"]).diff()

# -----------------------------
# 3) Rolling beta (OLS closed-form)
# -----------------------------
def rolling_beta(y, x, window):
    cov = y.rolling(window).cov(x)
    var = x.rolling(window).var()
    return cov / var

BETA_W = 60
df["beta"] = rolling_beta(df["rK"], df["rS"], BETA_W)

# -----------------------------
# 4) Residual + zscore signal
# -----------------------------
RES_W = 60
df["resid"] = df["rK"] - df["beta"] * df["rS"]

df["resid_mean"] = df["resid"].rolling(RES_W).mean()
df["resid_std"]  = df["resid"].rolling(RES_W).std()
df["z"] = (df["resid"] - df["resid_mean"]) / df["resid_std"]

# -----------------------------
# 5) Thresholds (walk-forward friendly defaults)
# -----------------------------
# VIX threshold: rolling 252일 상위 80% (과열/리스크오프)
VIX_W = 252
df["vix_p80"] = df["VIX_t-1"].rolling(VIX_W).quantile(0.80)

# FX shock threshold: rFX의 rolling zscore 절대값이 상위 90%면 거래 중단
FX_W = 252
df["fx_mean"] = df["rFX"].rolling(FX_W).mean()
df["fx_std"]  = df["rFX"].rolling(FX_W).std()
df["fx_z"] = (df["rFX"] - df["fx_mean"]) / df["fx_std"]
df["fx_abs_p90"] = df["fx_z"].abs().rolling(FX_W).quantile(0.90)

# 필터: (VIX가 높거나) (FX 쇼크가 크면) 거래 금지
df["allow"] = (df["VIX_t-1"] <= df["vix_p80"]) & (df["fx_z"].abs() <= df["fx_abs_p90"])

# -----------------------------
# 6) Trading rule (Strategy A: residual mean reversion)
# -----------------------------
ENTRY = 1.0
EXIT  = 0.2

pos = np.zeros(len(df))
z = df["z"].values
allow = df["allow"].values

for i in range(1, len(df)):
    # 기본: 이전 포지션 유지
    pos[i] = pos[i-1]

    # 필터 꺼지면 청산
    if not allow[i]:
        pos[i] = 0
        continue

    # 진입/청산
    if pos[i-1] == 0:
        if z[i] <= -ENTRY:
            pos[i] = +1   # long kospi
        elif z[i] >= +ENTRY:
            pos[i] = -1   # short kospi (선물 가능 가정)
    else:
        if abs(z[i]) <= EXIT:
            pos[i] = 0

df["pos"] = pos

# -----------------------------
# 7) Backtest PnL (next-day execution 가정)
# -----------------------------
# 신호는 t에 계산 -> t+1 수익률에 적용 (룩어헤드 방지)
df["strategy_ret"] = df["pos"].shift(1) * df["rK"]

# 간단 비용 (코스피200 선물/ETF 가정): 포지션 변동에 비례
TC = 0.0002  # 2bp 예시
df["turnover"] = df["pos"].diff().abs()
df["strategy_ret_net"] = df["strategy_ret"] - TC * df["turnover"]

# 누적
df["equity"] = (1 + df["strategy_ret_net"].fillna(0)).cumprod()

# 요약
ann_factor = 252
mean = df["strategy_ret_net"].mean() * ann_factor
vol  = df["strategy_ret_net"].std() * np.sqrt(ann_factor)
sharpe = mean / vol if vol > 0 else np.nan
mdd = (df["equity"] / df["equity"].cummax() - 1).min()

print("Rows:", len(df))
print(f"Ann.Return: {mean:.3%}")
print(f"Ann.Vol:    {vol:.3%}")
print(f"Sharpe:     {sharpe:.2f}")
print(f"MDD:        {mdd:.2%}")
print("Hit ratio:", (df["strategy_ret_net"] > 0).mean())
