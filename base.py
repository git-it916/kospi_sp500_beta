import pandas as pd
import numpy as np

# 기본 데이터 경로 설정
PATH = r"C:\Users\10845\OneDrive - 이지스자산운용\문서\kospi_sp500_filtered.xlsx"

def to_float(x):
    """문자열 숫자의 콤마 제거 및 변환"""
    if isinstance(x, str):
        x = x.replace(",", "").strip()
    return pd.to_numeric(x, errors="coerce")

def load_data(path=PATH):
    """데이터 로드 및 전처리 (ffill 적용)"""
    df = pd.read_excel(path)
    df.columns = [c.strip() for c in df.columns]
    df["공통날짜"] = pd.to_datetime(df["공통날짜"])
    
    for c in ["kospi_t", "SPX_t-1", "VIX_t-1", "FX_t"]:
        df[c] = df[c].apply(to_float)
    
    # 날짜순 정렬 후 결측치 앞의 값으로 채우기(ffill) -> 이후 삭제
    return df.sort_values("공통날짜").set_index("공통날짜").ffill().dropna()

def compute_strategy(df):
    """전략 로직 계산 (파라미터 최적화 & Look-ahead bias 제거)"""
    out = df.copy()

    # 1. 로그 수익률
    out["rK"] = np.log(out["kospi_t"]).diff()
    out["rS"] = np.log(out["SPX_t-1"]).diff()
    out["rFX"] = np.log(out["FX_t"]).diff()

    # 2. Rolling Beta & Residual
    BETA_W = 60
    out["beta"] = out["rK"].rolling(BETA_W).cov(out["rS"]) / out["rS"].rolling(BETA_W).var()
    out["resid"] = out["rK"] - out["beta"] * out["rS"]

    # [중요] Z-Score 계산 시 Look-ahead Bias 제거
    # 당일의 잔차를 평가할 때, 평균과 표준편차는 '어제'까지의 분포를 사용
    RES_W = 60
    out["resid_mean"] = out["resid"].rolling(RES_W).mean().shift(1)
    out["resid_std"]  = out["resid"].rolling(RES_W).std().shift(1)
    out["z"] = (out["resid"] - out["resid_mean"]) / out["resid_std"]

    # 3. Filters (Optimized Params)
    VIX_W = 252
    FX_W  = 252
    VIX_Q = 0.75  # 최적화 값: 0.80 -> 0.75 (더 보수적)
    FX_Q  = 0.90  # 유지

    out["vix_th"] = out["VIX_t-1"].rolling(VIX_W).quantile(VIX_Q)

    out["fx_mean"] = out["rFX"].rolling(FX_W).mean()
    out["fx_std"]  = out["rFX"].rolling(FX_W).std()
    out["fx_z"] = (out["rFX"] - out["fx_mean"]) / out["fx_std"]
    out["fx_abs_th"] = out["fx_z"].abs().rolling(FX_W).quantile(FX_Q)

    out["allow"] = (out["VIX_t-1"] <= out["vix_th"]) & (out["fx_z"].abs() <= out["fx_abs_th"])

    # 4. Signal Generation
    ENTRY = 1.5  # 최적화 값: 1.0 -> 1.5 (진입 장벽 상향)
    EXIT  = 0.1  # 최적화 값: 0.2 -> 0.1 (빠른 청산)

    pos = np.zeros(len(out))
    z = out["z"].values
    allow = out["allow"].values

    for i in range(1, len(out)):
        pos[i] = pos[i-1]
        
        if not allow[i]:
            pos[i] = 0
            continue
        
        if np.isnan(z[i]):
            pos[i] = 0
            continue

        if pos[i-1] == 0:
            if z[i] <= -ENTRY:
                pos[i] = +1  # Long
            elif z[i] >= +ENTRY:
                pos[i] = -1  # Short
        else:
            if abs(z[i]) <= EXIT:
                pos[i] = 0

    out["pos"] = pos
    
    # 5. PnL (t-1 포지션 * t 수익률)
    out["strategy_ret"] = out["pos"].shift(1) * out["rK"]
    
    TC = 0.0002
    out["turnover"] = out["pos"].diff().abs()
    out["strategy_ret_net"] = out["strategy_ret"] - TC * out["turnover"]
    out["equity"] = (1 + out["strategy_ret_net"].fillna(0)).cumprod()
    
    return out

def compute_summary(df):
    """성과 지표 계산 (report.py 호환용 Dict 반환)"""
    ann_factor = 252
    valid_ret = df["strategy_ret_net"].dropna()
    
    if len(valid_ret) == 0:
        return {
            "ann_return": 0.0, "ann_vol": 0.0, "sharpe": 0.0, 
            "mdd": 0.0, "hit_ratio": 0.0
        }

    mean = valid_ret.mean() * ann_factor
    vol  = valid_ret.std() * np.sqrt(ann_factor)
    sharpe = mean / vol if vol > 0 else 0.0
    mdd = (df["equity"] / df["equity"].cummax() - 1).min()
    hit_ratio = (valid_ret > 0).mean()

    return {
        "ann_return": mean,
        "ann_vol": vol,
        "sharpe": sharpe,
        "mdd": mdd,
        "hit_ratio": hit_ratio
    }

if __name__ == "__main__":
    # 이 파일만 단독 실행 시 요약 출력
    df = load_data()
    df = compute_strategy(df)
    summary = compute_summary(df)
    
    print("="*40)
    print(" [Base Strategy Execution] ")
    print("="*40)
    print(f"Period:     {df.index[0].date()} ~ {df.index[-1].date()}")
    print(f"Ann.Return: {summary['ann_return']:.2%}")
    print(f"Ann.Vol:    {summary['ann_vol']:.2%}")
    print(f"Sharpe:     {summary['sharpe']:.2f}")
    print(f"MDD:        {summary['mdd']:.2%}")
    print(f"Hit Ratio:  {summary['hit_ratio']:.2%}")
    print(f"Total Ret:  {df['equity'].iloc[-1] - 1:.2%}")