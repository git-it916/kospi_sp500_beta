import pandas as pd  # 표 형태 데이터 처리용
import numpy as np  # 수치 계산용

# 기본 데이터 경로 설정(엑셀 원본 파일 위치)
PATH = r"C:\Users\10845\OneDrive - 이지스자산운용\문서\kospi_sp500_filtered.xlsx"

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
    """전략 로직 계산 (파라미터 최적화 & Look-ahead bias 제거)"""
    # 원본 데이터 보호를 위해 복사본 사용
    out = df.copy()

    # 1. 로그 수익률
    # 로그 수익률 = log(오늘 가격) - log(어제 가격)
    out["rK"] = np.log(out["kospi_t"]).diff()
    out["rS"] = np.log(out["SPX_t-1"]).diff()
    out["rFX"] = np.log(out["FX_t"]).diff()

    # 2. Rolling Beta & Residual
    # 60일 구간에서 KOSPI가 SPX에 얼마나 민감한지(베타) 추정
    BETA_W = 60
    out["beta"] = out["rK"].rolling(BETA_W).cov(out["rS"]) / out["rS"].rolling(BETA_W).var()
    # 잔차(residual) = 실제 KOSPI 수익률 - (베타 * SPX 수익률)
    out["resid"] = out["rK"] - out["beta"] * out["rS"]

    # [중요] Z-Score 계산 시 Look-ahead Bias 제거
    # 당일의 잔차를 평가할 때, 평균과 표준편차는 '어제'까지의 분포를 사용
    RES_W = 60
    # rolling 평균/표준편차 계산 후 하루 shift(1)로 미래 정보 사용을 방지
    out["resid_mean"] = out["resid"].rolling(RES_W).mean().shift(1)
    out["resid_std"]  = out["resid"].rolling(RES_W).std().shift(1)
    # z-score = (오늘 잔차 - 과거 평균) / 과거 표준편차
    out["z"] = (out["resid"] - out["resid_mean"]) / out["resid_std"]

    # 3. Filters (Optimized Params)
    # 변동성이 너무 큰 구간은 매매를 제한하는 필터
    VIX_W = 252
    FX_W  = 252
    VIX_Q = 0.75  # 최적화 값: 0.80 -> 0.75 (더 보수적)
    FX_Q  = 0.90  # 유지

    # VIX의 롤링 분위수 임계값 계산
    out["vix_th"] = out["VIX_t-1"].rolling(VIX_W).quantile(VIX_Q)

    # FX 변동성의 z-score 계산
    out["fx_mean"] = out["rFX"].rolling(FX_W).mean()
    out["fx_std"]  = out["rFX"].rolling(FX_W).std()
    out["fx_z"] = (out["rFX"] - out["fx_mean"]) / out["fx_std"]
    # FX 변동성의 절대값이 일정 수준 이상이면 필터로 차단
    out["fx_abs_th"] = out["fx_z"].abs().rolling(FX_W).quantile(FX_Q)

    # VIX와 FX 모두 임계값 이하일 때만 매매 허용
    out["allow"] = (out["VIX_t-1"] <= out["vix_th"]) & (out["fx_z"].abs() <= out["fx_abs_th"])

    # 4. Signal Generation
    # 진입/청산 기준값(잔차 z-score)
    ENTRY = 1.5  # 최적화 값: 1.0 -> 1.5 (진입 장벽 상향)
    EXIT  = 0.1  # 최적화 값: 0.2 -> 0.1 (빠른 청산)

    # 포지션 배열 초기화 (0=현금, +1=롱, -1=숏)
    pos = np.zeros(len(out))
    # z-score와 필터 조건을 numpy 배열로 뽑아서 루프에서 사용
    z = out["z"].values
    allow = out["allow"].values

    for i in range(1, len(out)):
        # 기본은 전날 포지션을 유지
        pos[i] = pos[i-1]
        
        # 필터 조건을 통과하지 못하면 강제 청산
        if not allow[i]:
            pos[i] = 0
            continue
        
        # z-score가 NaN이면 판단 불가 -> 포지션 없음
        if np.isnan(z[i]):
            pos[i] = 0
            continue

        # 포지션이 없을 때만 신규 진입
        if pos[i-1] == 0:
            if z[i] <= -ENTRY:
                pos[i] = +1  # Long
            elif z[i] >= +ENTRY:
                pos[i] = -1  # Short
        else:
            # 포지션이 있을 때 z-score가 0 근처로 회귀하면 청산
            if abs(z[i]) <= EXIT:
                pos[i] = 0

    # 최종 포지션 저장
    out["pos"] = pos
    
    # 5. PnL (t-1 포지션 * t 수익률)
    # 실제 수익률은 "전일 보유 포지션"이 오늘 가격 변화에 반영됨
    out["strategy_ret"] = out["pos"].shift(1) * out["rK"]
    
    # 거래 비용(매매 회전율에 비례)
    TC = 0.0002
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

if __name__ == "__main__":
    # 이 파일만 단독 실행하면 요약 성과를 콘솔에 출력
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
