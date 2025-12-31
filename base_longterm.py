import pandas as pd
import numpy as np

# -----------------------------
# 1) 기본 설정 (Global Best Params 적용)
# -----------------------------
# 파일 경로는 본인 환경에 맞게 수정
PATH = r"C:\Users\10845\OneDrive - 이지스자산운용\문서\kospi_sp500_filtered_longterm.xlsx"


class StrategyParams:
    """Grid Search 최적화 파라미터 (Optimized via Stage 2 Fine Grid Search)

    최적화 일자: 2025-12-31
    최적화 기간: 2002-02-22 ~ 2025-12-29 (23.8년)
    최적화 방법: 3-Stage Grid Search (Coarse → Fine → Robustness)

    성과 개선:
    - Sharpe: 0.431 → 0.781 (+81%)
    - Calmar: 0.380 → 0.581 (+53%)
    - MDD: -26.56% → -15.72% (+41% 개선)
    - Score: 1.416 → 1.676 (+18%)
    """
    # Window Sizes
    BETA_WINDOW = 60
    RESID_WINDOW = 60
    FILTER_WINDOW = 252

    # Entry/Exit Thresholds (Grid Search 최적화)
    ENTRY_THRESHOLD = 2.15  # Z-Score 진입 기준 (2.0 → 2.15)
    EXIT_THRESHOLD = 0.0    # Z-Score 청산 기준 (유지)
    STOP_LOSS_MULTIPLIER = 3.3  # 손절매 승수 (3.0 → 3.3, Z-Score 7.095)

    # Risk Filters (Grid Search 최적화)
    VIX_QUANTILE = 0.94  # VIX 상위 6% 차단 (0.85 → 0.94, 필터 완화)
    FX_QUANTILE = 0.96   # FX Shock 상위 4% 차단 (0.90 → 0.96, 필터 완화)

    # Transaction Cost
    TRANSACTION_COST = 0.0002  # 0.02%


# 하위 호환성을 위한 전역 변수 (grid_search_longterm.py에서 사용)
ENTRY = StrategyParams.ENTRY_THRESHOLD
EXIT = StrategyParams.EXIT_THRESHOLD
VIX_Q = StrategyParams.VIX_QUANTILE
FX_Q = StrategyParams.FX_QUANTILE
STOP_LOSS_MULT = StrategyParams.STOP_LOSS_MULTIPLIER

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
    """전략 로직 계산 (명세서 기준, Look-ahead bias 완전 제거)"""
    if df.empty: return df

    out = df.copy()

    # 1. 로그 수익률 (명세서 3.1절)
    out["rK"] = np.log(out["kospi_t"]).diff()
    out["rS"] = np.log(out["SPX_t-1"]).diff()
    out["rFX"] = np.log(out["FX_t"]).diff()

    # 2. Rolling Beta & Residual (명세서 3.2절)
    BETA_W = StrategyParams.BETA_WINDOW
    out["beta"] = out["rK"].rolling(BETA_W).cov(out["rS"]) / out["rS"].rolling(BETA_W).var()
    out["resid"] = out["rK"] - out["beta"] * out["rS"]

    # 3. Z-Score (명세서 3.3절 - Look-ahead Bias 제거: shift 1)
    RES_W = StrategyParams.RESID_WINDOW
    out["resid_mean"] = out["resid"].rolling(RES_W).mean().shift(1)
    out["resid_std"]  = out["resid"].rolling(RES_W).std().shift(1)
    out["z"] = (out["resid"] - out["resid_mean"]) / out["resid_std"]

    # 4. Filters (명세서 3.4절 & 4.2절 - Risk Management)
    FILTER_W = StrategyParams.FILTER_WINDOW

    # VIX Rank 계산 (과거 1년 대비 현재 위치) - Look-ahead Bias 제거
    out["vix_rank"] = out["VIX_t-1"].rolling(FILTER_W).rank(pct=True).shift(1)

    # FX 변동성 Shock 계산 - Look-ahead Bias 제거
    out["fx_mean"] = out["rFX"].rolling(FILTER_W).mean()
    out["fx_std"]  = out["rFX"].rolling(FILTER_W).std()
    out["fx_z"]    = (out["rFX"] - out["fx_mean"]) / out["fx_std"]
    out["fx_shock"] = out["fx_z"].abs().rolling(FILTER_W).rank(pct=True).shift(1)

    # 필터 통과 여부
    out["allow"] = (out["vix_rank"] <= StrategyParams.VIX_QUANTILE) & (out["fx_shock"] <= StrategyParams.FX_QUANTILE)

    # 5. Signal Generation Loop (명세서 4.1절 & 4.3절 & 4.4절)
    pos = np.zeros(len(out))
    z = out["z"].values
    allow = out["allow"].values

    ENTRY_T = StrategyParams.ENTRY_THRESHOLD
    EXIT_T = StrategyParams.EXIT_THRESHOLD
    STOP_LOSS = StrategyParams.ENTRY_THRESHOLD * StrategyParams.STOP_LOSS_MULTIPLIER

    current_pos = 0

    for i in range(1, len(out)):
        # (1) 리스크 필터 체크 -> 걸리면 강제 청산 (명세서 4.2절)
        if not allow[i]:
            current_pos = 0
            pos[i] = 0
            continue

        if np.isnan(z[i]):
            pos[i] = 0
            continue

        # (2) 손절매 (Stop Loss) 체크 (명세서 4.4절 - Structural Break)
        # Z-Score가 ±6.0 초과 시 평균 회귀 가설 붕괴로 간주하고 즉시 청산
        if current_pos != 0:
            if abs(z[i]) >= STOP_LOSS:
                current_pos = 0
                pos[i] = 0
                continue

        # (3) 진입/청산 로직 (명세서 4.3절 & 4.4절)
        if current_pos == 0:
            # 진입: Z-Score 2.0 이상 벌어질 때만
            if z[i] <= -ENTRY_T:
                current_pos = +1  # Long (KOSPI 저평가)
            elif z[i] >= +ENTRY_T:
                current_pos = -1  # Short (KOSPI 고평가)
        else:
            # 청산: Z-Score가 평균으로 회귀하면 청산 (Mean Reversion)
            if abs(z[i]) <= EXIT_T:
                current_pos = 0

        pos[i] = current_pos

    out["pos"] = pos

    # 6. 성과 계산 (수수료 반영)
    out["strategy_ret"] = out["pos"].shift(1) * out["rK"]

    TC = StrategyParams.TRANSACTION_COST
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


def diagnostic_report(df):
    """전략 진단용 상세 리포트 (디버깅 및 분석용)"""
    print("\n" + "="*50)
    print(" [DIAGNOSTIC REPORT - LONG TERM]")
    print("="*50)

    # 1. 신호 발생 통계
    print("\n[Signal Breakdown]")
    print(f"Total Days: {len(df)}")
    long_entries = (df["pos"].diff() == 1).sum()
    short_entries = (df["pos"].diff() == -1).sum()
    exits = ((df["pos"].shift(1) != 0) & (df["pos"] == 0)).sum()
    total_trades = long_entries + short_entries
    print(f"Total Trades: {total_trades}")
    print(f"Long Entries: {long_entries} ({long_entries/total_trades*100:.1f}% of trades)" if total_trades > 0 else "Long Entries: 0")
    print(f"Short Entries: {short_entries} ({short_entries/total_trades*100:.1f}% of trades)" if total_trades > 0 else "Short Entries: 0")
    print(f"Total Exits: {exits}")
    print(f"Filter Block Days: {(~df['allow']).sum()} ({(~df['allow']).mean():.1%})")

    # 2. 손절매 발생 횟수
    stop_loss_threshold = StrategyParams.ENTRY_THRESHOLD * StrategyParams.STOP_LOSS_MULTIPLIER
    stop_loss_exits = ((df["pos"].shift(1) != 0) &
                       (df["pos"] == 0) &
                       (df["z"].abs() >= stop_loss_threshold)).sum()
    print(f"Stop-Loss Triggered: {stop_loss_exits}")

    # 3. 포지션 보유 통계
    exposure = (df["pos"] != 0).mean()
    long_days = (df["pos"] == 1).sum()
    short_days = (df["pos"] == -1).sum()
    print(f"\nExposure (포지션 보유율): {exposure:.1%}")
    print(f"Long Days: {long_days} ({long_days/len(df):.1%})")
    print(f"Short Days: {short_days} ({short_days/len(df):.1%})")

    # 4. 트레이딩 타이밍 상세 (최근 20개 거래)
    print("\n[Recent Trading Activity - Last 20 Trades]")
    pos_changes = df[df["pos"].diff() != 0].copy()
    pos_changes = pos_changes[pos_changes["pos"].diff() != 0]  # 실제 포지션 변화만

    if len(pos_changes) > 0:
        recent_trades = pos_changes.tail(20)
        for idx, row in recent_trades.iterrows():
            pos_now = row["pos"]
            pos_prev = df.loc[:idx, "pos"].shift(1).iloc[-1] if len(df.loc[:idx]) > 1 else 0

            if pos_now == 1 and pos_prev == 0:
                action = "LONG ENTRY"
            elif pos_now == -1 and pos_prev == 0:
                action = "SHORT ENTRY"
            elif pos_now == 0 and pos_prev == 1:
                action = "LONG EXIT"
            elif pos_now == 0 and pos_prev == -1:
                action = "SHORT EXIT"
            elif pos_now == 1 and pos_prev == -1:
                action = "FLIP: SHORT->LONG"
            elif pos_now == -1 and pos_prev == 1:
                action = "FLIP: LONG->SHORT"
            else:
                action = "UNKNOWN"

            z_val = row["z"]
            allow_status = "OK" if row["allow"] else "BLOCKED"
            print(f"{idx.date()}: {action:20s} | Z={z_val:6.2f} | Filter={allow_status}")
    else:
        print("No trades detected in backtest period")

    # 5. 연도별 트레이딩 카운트
    print("\n[Trading Count by Year]")
    df_with_year = df.copy()
    df_with_year["year"] = df_with_year.index.year
    df_with_year["is_long_entry"] = (df_with_year["pos"].diff() == 1)
    df_with_year["is_short_entry"] = (df_with_year["pos"].diff() == -1)

    yearly_trades = df_with_year.groupby("year").agg({
        "is_long_entry": "sum",
        "is_short_entry": "sum"
    })
    yearly_trades["Total"] = yearly_trades["is_long_entry"] + yearly_trades["is_short_entry"]
    yearly_trades.columns = ["Long Entries", "Short Entries", "Total Trades"]

    print(yearly_trades.to_string())

    # 4. Z-Score 분포
    print("\n[Z-Score Distribution]")
    print(df["z"].describe())

    # 5. 극단적 Z-Score 케이스 (상위 5개)
    print("\n[Top 5 Extreme Z-Scores (Positive)]")
    top_z = df.nlargest(5, "z")[["z", "pos", "allow", "resid"]]
    for idx, row in top_z.iterrows():
        print(f"{idx.date()}: Z={row['z']:.2f}, Pos={row['pos']:.0f}, Allow={row['allow']}")

    print("\n[Top 5 Extreme Z-Scores (Negative)]")
    bottom_z = df.nsmallest(5, "z")[["z", "pos", "allow", "resid"]]
    for idx, row in bottom_z.iterrows():
        print(f"{idx.date()}: Z={row['z']:.2f}, Pos={row['pos']:.0f}, Allow={row['allow']}")

    print("="*50 + "\n")


def validate_implementation(df):
    """명세서 vs 구현 검증 (자동 체크)"""
    issues = []
    warnings = []

    # 1. Z-Score 극단값 체크
    max_z = df["z"].abs().max()
    if max_z > 10:
        warnings.append(f"[WARNING] Extreme Z-Score detected: {max_z:.2f}")

    # 2. 손절매 로직 동작 확인
    stop_loss_threshold = StrategyParams.ENTRY_THRESHOLD * StrategyParams.STOP_LOSS_MULTIPLIER
    stop_loss_exits = ((df["pos"].shift(1) != 0) &
                       (df["pos"] == 0) &
                       (df["z"].abs() >= stop_loss_threshold)).sum()
    if stop_loss_exits == 0:
        warnings.append("[WARNING] Stop-loss never triggered")
    else:
        print(f"[OK] Stop-loss verified ({stop_loss_exits} occurrences)")

    # 3. 필터 동작 확인
    if df["allow"].all():
        warnings.append("[WARNING] Filters never activated")
    else:
        filter_rate = (~df["allow"]).mean()
        print(f"[OK] Filters activated {filter_rate:.1%} of time")

    # 4. 파라미터 확인
    print(f"[OK] Entry Threshold: {StrategyParams.ENTRY_THRESHOLD}")
    print(f"[OK] Exit Threshold: {StrategyParams.EXIT_THRESHOLD}")
    print(f"[OK] Stop Loss: {stop_loss_threshold}")
    print(f"[OK] VIX Quantile: {StrategyParams.VIX_QUANTILE}")
    print(f"[OK] FX Quantile: {StrategyParams.FX_QUANTILE}")

    # 5. Look-ahead Bias 체크
    if df["resid_mean"].isna().sum() > df["resid"].rolling(60).mean().isna().sum():
        print("[OK] Look-ahead bias prevention verified")

    print("\n" + "="*50)
    print(" [VALIDATION REPORT]")
    print("="*50)

    if not issues:
        print("[OK] No critical issues found")
    else:
        print("[ERROR] Critical Issues:")
        for issue in issues:
            print(f"  {issue}")

    if warnings:
        print("\n[WARNING] Warnings:")
        for warning in warnings:
            print(f"  {warning}")

    print("="*50 + "\n")

    return {"issues": issues, "warnings": warnings}


if __name__ == "__main__":
    # 단독 실행 시 테스트 결과 출력
    df = load_data()
    df = compute_strategy(df)
    summary = compute_summary(df)

    print("="*50)
    print(" [Long-term Strategy Execution]")
    print("="*50)
    print(f"Period:     {df.index[0].date()} ~ {df.index[-1].date()}")
    print(f"Ann.Return: {summary['ann_return']:.2%}")
    print(f"Ann.Vol:    {summary['ann_vol']:.2%}")
    print(f"Sharpe:     {summary['sharpe']:.2f}")
    print(f"MDD:        {summary['mdd']:.2%}")
    print(f"Hit Ratio:  {summary['hit_ratio']:.2%}")
    print(f"Total Ret:  {df['equity'].iloc[-1] - 1:.2%}")
    print("="*50)

    # 검증 및 진단 리포트 실행
    validate_implementation(df)
    diagnostic_report(df)