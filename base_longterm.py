import pandas as pd
import numpy as np

# -----------------------------
# 1) 기본 설정 (Global Best Params 적용)
# -----------------------------
# 파일 경로는 본인 환경에 맞게 수정
PATH = r"C:\Users\10845\OneDrive - 이지스자산운용\문서\kospi_sp500_filtered_longterm.xlsx"

# ★ 확정된 최적 파라미터 (25년 백테스트 검증 완료)
ENTRY = 2.0   # 확실한 기회가 올 때까지 대기 (보수적 진입)
EXIT  = 0.0   # 평균으로 완전히 회귀할 때까지 보유 (수익 극대화)
VIX_Q = 0.85  # VIX 상위 15% 과열 구간만 회피
FX_Q  = 0.90  # 환율 변동성 상위 10% 회피

# 안전장치
STOP_LOSS_MULT = 3.0  # Entry * 3.0 (Z-Score 6.0) 도달 시 강제 손절

def to_float(x):
    if isinstance(x, str):
        x = x.replace(",", "").strip()
    return pd.to_numeric(x, errors="coerce")

def load_data(path=PATH):
    """데이터 로드 및 전처리 (ffill 적용)"""
    try:
        df = pd.read_excel(path)
    except FileNotFoundError:
        print("[Error] 파일을 찾을 수 없습니다.")
        return pd.DataFrame()

    df.columns = [c.strip() for c in df.columns]
    df["공통날짜"] = pd.to_datetime(df["공통날짜"])
    
    for c in ["kospi_t", "SPX_t-1", "VIX_t-1", "FX_t"]:
        df[c] = df[c].apply(to_float)
    
    # [중요] ffill로 결측치 방어 후 dropna
    return df.sort_values("공통날짜").set_index("공통날짜").ffill().dropna()

def compute_strategy(df):
    """전략 로직 계산"""
    if df.empty: return df

    out = df.copy()

    # 1. 로그 수익률
    out["rK"] = np.log(out["kospi_t"]).diff()
    out["rS"] = np.log(out["SPX_t-1"]).diff()
    out["rFX"] = np.log(out["FX_t"]).diff()

    # 2. Rolling Beta & Residual
    BETA_W = 60
    out["beta"] = out["rK"].rolling(BETA_W).cov(out["rS"]) / out["rS"].rolling(BETA_W).var()
    out["resid"] = out["rK"] - out["beta"] * out["rS"]

    # 3. Z-Score (Look-ahead Bias 제거: shift 1)
    RES_W = 60
    out["resid_mean"] = out["resid"].rolling(RES_W).mean().shift(1)
    out["resid_std"]  = out["resid"].rolling(RES_W).std().shift(1)
    out["z"] = (out["resid"] - out["resid_mean"]) / out["resid_std"]

    # 4. Filters (Risk Management)
    VIX_W = 252
    FX_W  = 252

    # VIX Rank 계산 (과거 1년 대비 현재 위치)
    out["vix_rank"] = out["VIX_t-1"].rolling(VIX_W).rank(pct=True)
    
    # FX 변동성 Shock 계산
    out["fx_mean"] = out["rFX"].rolling(FX_W).mean()
    out["fx_std"]  = out["rFX"].rolling(FX_W).std()
    out["fx_z"]    = (out["rFX"] - out["fx_mean"]) / out["fx_std"]
    out["fx_shock"] = out["fx_z"].abs().rolling(FX_W).rank(pct=True)

    # 필터 통과 여부
    out["allow"] = (out["vix_rank"] <= VIX_Q) & (out["fx_shock"] <= FX_Q)

    # 5. Signal Generation Loop
    pos = np.zeros(len(out))
    z = out["z"].values
    allow = out["allow"].values
    
    current_pos = 0

    for i in range(1, len(out)):
        # (1) 리스크 필터 체크 -> 걸리면 강제 청산
        if not allow[i]:
            current_pos = 0
            pos[i] = 0
            continue
        
        if np.isnan(z[i]):
            pos[i] = 0
            continue

        # (2) 손절매 (Stop Loss) 체크
        # 포지션이 있는데, Z-Score가 진입 레벨의 3배(6.0)를 넘어가면 즉시 탈출
        if current_pos != 0:
            if abs(z[i]) > (ENTRY * STOP_LOSS_MULT):
                current_pos = 0
                pos[i] = 0
                continue

        # (3) 진입/청산 로직
        if current_pos == 0:
            # 진입: Z-Score 2.0 이상 벌어질 때만
            if z[i] <= -ENTRY:
                current_pos = +1  # Long
            elif z[i] >= +ENTRY:
                current_pos = -1  # Short
        else:
            # 청산: Z-Score가 0.0 (평균)으로 돌아오면 청산
            if abs(z[i]) <= EXIT:
                current_pos = 0

        pos[i] = current_pos

    out["pos"] = pos
    
    # 6. 성과 계산 (수수료 반영)
    out["strategy_ret"] = out["pos"].shift(1) * out["rK"]
    
    TC = 0.0002
    out["turnover"] = out["pos"].diff().abs()
    out["strategy_ret_net"] = out["strategy_ret"] - TC * out["turnover"]
    out["equity"] = (1 + out["strategy_ret_net"].fillna(0)).cumprod()
    
    return out

def compute_summary(df):
    """성과 요약 (report.py 호환)"""
    if df.empty: return {}
    
    ann_factor = 252
    valid_ret = df["strategy_ret_net"].dropna()
    
    if len(valid_ret) == 0:
        return {"ann_return": 0.0, "ann_vol": 0.0, "sharpe": 0.0, "mdd": 0.0, "hit_ratio": 0.0}

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
    # 단독 실행 시 테스트 결과 출력
    df = load_data()
    df = compute_strategy(df)
    summary = compute_summary(df)
    
    print("="*40)
    print(" [Final Optimized Strategy] ")
    print("="*40)
    print(f"Period:     {df.index[0].date()} ~ {df.index[-1].date()}")
    print(f"Ann.Return: {summary['ann_return']:.2%}")
    print(f"MDD:        {summary['mdd']:.2%}")
    print(f"Sharpe:     {summary['sharpe']:.2f}")
    print(f"Total Ret:  {df['equity'].iloc[-1] - 1:.2%}")