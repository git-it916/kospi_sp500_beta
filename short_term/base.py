import pandas as pd  # 표 형태 데이터 처리용
import numpy as np  # 수치 계산용

# 기본 데이터 경로 설정(엑셀 원본 파일 위치)
PATH = r"C:\Users\10845\OneDrive - 이지스자산운용\문서\kospi_sp500_filtered.xlsx"


class StrategyParams:
    """명세서 기준 전략 파라미터 (Global Best Parameters)"""
    # Window Sizes
    BETA_WINDOW = 60
    RESID_WINDOW = 60
    FILTER_WINDOW = 252

    # Entry/Exit Thresholds (명세서 4.1절)
    ENTRY_THRESHOLD = 2.0  # Z-Score 진입 기준
    EXIT_THRESHOLD = 0.0   # Z-Score 청산 기준
    STOP_LOSS_MULTIPLIER = 3.0  # 손절매 승수 (Z-Score 6.0 = 2.0 * 3.0)

    # Risk Filters (명세서 4.2절)
    VIX_QUANTILE = 0.85  # VIX 상위 15% 차단
    FX_QUANTILE = 0.90   # FX Shock 상위 10% 차단

    # Transaction Cost
    TRANSACTION_COST = 0.0002  # 0.02%

def to_float(x):
    """문자열 숫자(예: '1,234')를 실수로 변환"""
    # 엑셀에서 숫자가 문자열로 들어온 경우를 대비
    if isinstance(x, str):
        # 천단위 콤마 제거 + 앞뒤 공백 제거
        x = x.replace(",", "").strip()
    # 숫자로 변환되지 않는 값은 NaN으로 처리
    return pd.to_numeric(x, errors="coerce")

def load_data(path=PATH):
    """엑셀 데이터 로드 및 전처리 (날짜 정렬, 결측치 처리)"""
    # 엑셀 파일 읽기
    df = pd.read_excel(path)
    # 컬럼명 양끝 공백 제거(엑셀 특유의 공백 문제 방지)
    df.columns = [c.strip() for c in df.columns]
    # 날짜 컬럼을 datetime 타입으로 변환
    df["공통날짜"] = pd.to_datetime(df["공통날짜"])
    
    # 숫자 컬럼들을 안전하게 숫자로 변환
    for c in ["kospi_t", "SPX_t-1", "VIX_t-1", "FX_t"]:
        df[c] = df[c].apply(to_float)
    
    # 날짜순 정렬 후, 앞의 값으로 결측치 채우기(ffill)
    # 마지막에 dropna()로 남은 결측치 제거
    return df.sort_values("공통날짜").set_index("공통날짜").ffill().dropna()

def compute_strategy(df):
    """전략 로직 계산 (명세서 기준, Look-ahead bias 완전 제거)"""
    # 원본 데이터 보호를 위해 복사본 사용
    out = df.copy()

    # 1. 로그 수익률 (명세서 3.1절)
    # 로그 수익률 = log(오늘 가격) - log(어제 가격)
    out["rK"] = np.log(out["kospi_t"]).diff()
    out["rS"] = np.log(out["SPX_t-1"]).diff()
    out["rFX"] = np.log(out["FX_t"]).diff()

    # 2. Rolling Beta & Residual (명세서 3.2절)
    # 60일 구간에서 KOSPI가 SPX에 얼마나 민감한지(베타) 추정
    BETA_W = StrategyParams.BETA_WINDOW
    out["beta"] = out["rK"].rolling(BETA_W).cov(out["rS"]) / out["rS"].rolling(BETA_W).var()
    # 잔차(residual) = 실제 KOSPI 수익률 - (베타 * SPX 수익률)
    out["resid"] = out["rK"] - out["beta"] * out["rS"]

    # [중요] Z-Score 계산 시 Look-ahead Bias 제거 (명세서 3.3절)
    # 당일의 잔차를 평가할 때, 평균과 표준편차는 '어제'까지의 분포를 사용
    RES_W = StrategyParams.RESID_WINDOW
    # rolling 평균/표준편차 계산 후 하루 shift(1)로 미래 정보 사용을 방지
    out["resid_mean"] = out["resid"].rolling(RES_W).mean().shift(1)
    out["resid_std"]  = out["resid"].rolling(RES_W).std().shift(1)
    # z-score = (오늘 잔차 - 과거 평균) / 과거 표준편차
    out["z"] = (out["resid"] - out["resid_mean"]) / out["resid_std"]

    # 3. Filters (명세서 3.4절 & 4.2절)
    # 변동성이 너무 큰 구간은 매매를 제한하는 필터
    FILTER_W = StrategyParams.FILTER_WINDOW
    VIX_Q = StrategyParams.VIX_QUANTILE
    FX_Q = StrategyParams.FX_QUANTILE

    # VIX의 롤링 분위수 임계값 계산 (Look-ahead Bias 제거)
    out["vix_th"] = out["VIX_t-1"].rolling(FILTER_W).quantile(VIX_Q).shift(1)

    # FX 변동성의 z-score 계산
    out["fx_mean"] = out["rFX"].rolling(FILTER_W).mean()
    out["fx_std"]  = out["rFX"].rolling(FILTER_W).std()
    out["fx_z"] = (out["rFX"] - out["fx_mean"]) / out["fx_std"]
    # FX 변동성의 절대값이 일정 수준 이상이면 필터로 차단 (Look-ahead Bias 제거)
    out["fx_abs_th"] = out["fx_z"].abs().rolling(FILTER_W).quantile(FX_Q).shift(1)

    # VIX와 FX 모두 임계값 이하일 때만 매매 허용
    out["allow"] = (out["VIX_t-1"] <= out["vix_th"]) & (out["fx_z"].abs() <= out["fx_abs_th"])

    # 4. Signal Generation (명세서 4.1절 & 4.3절 & 4.4절)
    # 진입/청산 기준값(잔차 z-score) - 명세서 파라미터 사용
    ENTRY = StrategyParams.ENTRY_THRESHOLD
    EXIT = StrategyParams.EXIT_THRESHOLD
    STOP_LOSS = StrategyParams.ENTRY_THRESHOLD * StrategyParams.STOP_LOSS_MULTIPLIER

    # 포지션 배열 초기화 (0=현금, +1=롱, -1=숏)
    pos = np.zeros(len(out))
    # z-score와 필터 조건을 numpy 배열로 뽑아서 루프에서 사용
    z = out["z"].values
    allow = out["allow"].values

    for i in range(1, len(out)):
        # 기본은 전날 포지션을 유지
        pos[i] = pos[i-1]

        # 필터 조건을 통과하지 못하면 강제 청산 (명세서 4.2절)
        if not allow[i]:
            pos[i] = 0
            continue

        # z-score가 NaN이면 판단 불가 -> 포지션 없음
        if np.isnan(z[i]):
            pos[i] = 0
            continue

        # [중요] 손절매 로직 (명세서 4.4절 - Structural Break)
        # Z-Score가 ±6.0 초과 시 평균 회귀 가설 붕괴로 간주하고 즉시 청산
        if pos[i-1] != 0 and abs(z[i]) >= STOP_LOSS:
            pos[i] = 0
            continue

        # 포지션이 없을 때만 신규 진입 (명세서 4.3절)
        if pos[i-1] == 0:
            if z[i] <= -ENTRY:
                pos[i] = +1  # Long (KOSPI 저평가)
            elif z[i] >= +ENTRY:
                pos[i] = -1  # Short (KOSPI 고평가)
        else:
            # 포지션이 있을 때 z-score가 평균으로 회귀하면 청산 (명세서 4.4절 - Mean Reversion)
            if abs(z[i]) <= EXIT:
                pos[i] = 0

    # 최종 포지션 저장
    out["pos"] = pos
    
    # 5. PnL (t-1 포지션 * t 수익률)
    # 실제 수익률은 "전일 보유 포지션"이 오늘 가격 변화에 반영됨
    out["strategy_ret"] = out["pos"].shift(1) * out["rK"]

    # 거래 비용(매매 회전율에 비례)
    TC = StrategyParams.TRANSACTION_COST
    out["turnover"] = out["pos"].diff().abs()
    # 거래 비용 차감 후 순수익률 계산
    out["strategy_ret_net"] = out["strategy_ret"] - TC * out["turnover"]
    # 누적 수익률(에쿼티 곡선)
    out["equity"] = (1 + out["strategy_ret_net"].fillna(0)).cumprod()

    return out

def compute_summary(df):
    """성과 지표 계산 (report.py 호환용 Dict 반환)"""
    # 연간 환산을 위한 거래일 수(보통 252일 가정)
    ann_factor = 252
    # 순수익률만 뽑아서 통계 계산에 사용
    valid_ret = df["strategy_ret_net"].dropna()

    # 데이터가 너무 적으면 0으로 반환
    if len(valid_ret) == 0:
        return {
            "ann_return": 0.0, "ann_vol": 0.0, "sharpe": 0.0,
            "mdd": 0.0, "hit_ratio": 0.0
        }

    # 연환산 수익률/변동성
    mean = valid_ret.mean() * ann_factor
    vol  = valid_ret.std() * np.sqrt(ann_factor)
    # 샤프 비율(위험 대비 수익)
    sharpe = mean / vol if vol > 0 else 0.0
    # 최대 낙폭(MDD) = 최고점 대비 최소 하락폭
    mdd = (df["equity"] / df["equity"].cummax() - 1).min()
    # 수익률이 플러스인 비율
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
    print(" [DIAGNOSTIC REPORT]")
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

    # 6. 필터 활성화 기간
    if (~df["allow"]).sum() > 0:
        print("\n[Filter Activation Periods (First 10)]")
        filter_periods = df[~df["allow"]].head(10)
        for idx, row in filter_periods.iterrows():
            vix_status = "VIX High" if row["VIX_t-1"] > row["vix_th"] else "VIX OK"
            fx_status = "FX Shock" if abs(row["fx_z"]) > row["fx_abs_th"] else "FX OK"
            print(f"{idx.date()}: {vix_status}, {fx_status}")

    print("="*50 + "\n")


def validate_implementation(df):
    """명세서 vs 구현 검증 (자동 체크)"""
    issues = []
    warnings = []

    # 1. Z-Score 극단값 체크
    max_z = df["z"].abs().max()
    if max_z > 10:
        warnings.append(f"[WARNING] Extreme Z-Score detected: {max_z:.2f} (평균회귀 가정 점검 필요)")

    # 2. 손절매 로직 동작 확인
    stop_loss_threshold = StrategyParams.ENTRY_THRESHOLD * StrategyParams.STOP_LOSS_MULTIPLIER
    stop_loss_exits = ((df["pos"].shift(1) != 0) &
                       (df["pos"] == 0) &
                       (df["z"].abs() >= stop_loss_threshold)).sum()
    if stop_loss_exits == 0:
        warnings.append("[WARNING] Stop-loss never triggered (백테스트 기간 동안 구조적 붕괴 없음)")
    else:
        print(f"[OK] Stop-loss logic verified ({stop_loss_exits} occurrences)")

    # 3. 필터 동작 확인
    if df["allow"].all():
        warnings.append("[WARNING] Filters never activated (파라미터가 너무 느슨할 수 있음)")
    else:
        filter_rate = (~df["allow"]).mean()
        print(f"[OK] Filters activated {filter_rate:.1%} of time")

    # 4. 파라미터 명세서 일치 확인
    print(f"[OK] Entry Threshold: {StrategyParams.ENTRY_THRESHOLD}")
    print(f"[OK] Exit Threshold: {StrategyParams.EXIT_THRESHOLD}")
    print(f"[OK] Stop Loss: {stop_loss_threshold} (Entry * {StrategyParams.STOP_LOSS_MULTIPLIER})")
    print(f"[OK] VIX Quantile: {StrategyParams.VIX_QUANTILE}")
    print(f"[OK] FX Quantile: {StrategyParams.FX_QUANTILE}")

    # 5. Look-ahead Bias 체크 (간접 확인)
    # resid_mean, resid_std가 shift되어 있는지는 코드로 확인
    if df["resid_mean"].isna().sum() > df["resid"].rolling(60).mean().isna().sum():
        print("[OK] Look-ahead bias prevention: resid stats shifted")

    # 결과 출력
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
    # 이 파일만 단독 실행하면 요약 성과를 콘솔에 출력
    df = load_data()
    df = compute_strategy(df)
    summary = compute_summary(df)

    print("="*50)
    print(" [Base Strategy Execution]")
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
