"""
walkforward_strategy.py  (FIXED VERSION)

Walk-forward grid search + OOS backtest for:
- Rolling beta (KOSPI vs SPX lag)
- Residual z-score mean reversion
- VIX & FX shock filters (rolling quantile thresholds with CONTEXT)

Input: Excel file with columns:
  공통날짜, kospi_t, SPX_t-1, VIX_t-1, FX_t

Outputs:
  - oos_equity.csv
  - wf_params_by_segment.csv
"""

import warnings
warnings.filterwarnings("ignore")

from dataclasses import dataclass
from datetime import datetime
from itertools import product
from pathlib import Path

import pandas as pd
import numpy as np


# =========================
# USER CONFIG
# =========================
# 엑셀 원본 데이터 경로
PATH = r"C:\Users\10845\OneDrive - 이지스자산운용\문서\kospi_sp500_filtered.xlsx"

# 필수 컬럼 이름 정의
DATE_COL = "공통날짜"
REQUIRED_COLS = ["kospi_t", "SPX_t-1", "VIX_t-1", "FX_t"]

# 전략 계산에 쓰이는 기본 윈도우 길이(일 단위)
BETA_W = 60
RES_W  = 60
Q_W    = 252        # VIX / FX rolling window

# 거래비용 가정(매매 1회당 비율)
TC = 0.0002         # transaction cost

# 워크-포워드 학습/검증 기간 설정
TRAIN_YEARS = 2
TEST_MONTHS = 6
STEP_MONTHS = 6

# 그리드 서치 후보 값들
ENTRY_GRID = [0.8, 1.0, 1.2, 1.5]
EXIT_GRID  = [0.1, 0.2, 0.3, 0.4]
VIX_Q_GRID = [0.75, 0.80, 0.85, 0.90]
FX_Q_GRID  = [0.75, 0.80, 0.85, 0.90]

# 숏 포지션 허용 여부
ALLOW_SHORT = True


# =========================
# DATA LOAD
# =========================
def to_float(x):
    # 문자열 숫자를 안전하게 실수로 변환
    if isinstance(x, str):
        x = x.replace(",", "").strip()
    return pd.to_numeric(x, errors="coerce")


def read_excel_data(path):
    # 엑셀 파일 읽기
    df = pd.read_excel(path)
    # 컬럼 공백 제거
    df.columns = [c.strip() for c in df.columns]

    # 날짜 컬럼 변환
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    # 필수 숫자 컬럼 변환
    for c in REQUIRED_COLS:
        df[c] = df[c].apply(to_float)

    # 날짜 기준 정렬 후 인덱스로 설정
    df = df.sort_values(DATE_COL).set_index(DATE_COL)
    # 필수 컬럼에 결측치가 있는 행 제거
    df = df.dropna(subset=REQUIRED_COLS)
    return df


# =========================
# FEATURE ENGINEERING
# =========================
def rolling_beta(y, x, window):
    # 롤링 윈도우에서 공분산/분산으로 베타 계산
    return y.rolling(window).cov(x) / x.rolling(window).var()


def prepare_features(df):
    # 원본 보호를 위해 복사
    df = df.copy()

    # 로그 수익률 계산
    df["rK"]  = np.log(df["kospi_t"]).diff()
    df["rS"]  = np.log(df["SPX_t-1"]).diff()
    df["rFX"] = np.log(df["FX_t"]).diff()

    # 롤링 베타와 잔차 계산
    df["beta"] = rolling_beta(df["rK"], df["rS"], BETA_W)
    df["resid"] = df["rK"] - df["beta"] * df["rS"]

    # 어제까지의 60일 데이터로 분포를 추정하고, 오늘 잔차를 평가(룩어헤드 방지)
    resid_mean = df["resid"].rolling(RES_W).mean().shift(1)
    resid_std  = df["resid"].rolling(RES_W).std().shift(1)
    df["z"] = (df["resid"] - resid_mean) / resid_std

    # FX 수익률의 변동성(표준화) 계산
    df["fx_mean"] = df["rFX"].rolling(Q_W).mean()
    df["fx_std"]  = df["rFX"].rolling(Q_W).std()
    df["fx_z"] = (df["rFX"] - df["fx_mean"]) / df["fx_std"]

    return df


# =========================
# OOS SIMULATION (FIXED)
# =========================
def simulate_strategy_oos(df_full, start, end,
                          entry, exit_, vix_q, fx_q,
                          allow_short=True):

    # 롤링 임계값 계산을 위해 충분한 과거 컨텍스트 확보
    context_start = start - pd.Timedelta(days=400)
    ctx = df_full.loc[context_start:end].copy()

    # rolling threshold는 컨텍스트 구간 전체에서 계산
    vix_th = ctx["VIX_t-1"].rolling(Q_W).quantile(vix_q)
    fx_th  = ctx["fx_z"].abs().rolling(Q_W).quantile(fx_q)

    # VIX/FX 조건을 동시에 만족하는 경우만 매매 허용
    allow = (ctx["VIX_t-1"] <= vix_th) & (ctx["fx_z"].abs() <= fx_th)

    # 루프에서 사용할 numpy 배열 준비
    z = ctx["z"].values
    allow_v = allow.values

    # 포지션 초기화(0=현금, +1=롱, -1=숏)
    pos = np.zeros(len(ctx))

    for i in range(1, len(ctx)):
        # 기본은 전일 포지션 유지
        pos[i] = pos[i-1]

        # 필터 미통과 또는 z-score 비정상 값은 포지션 해제
        if not allow_v[i] or not np.isfinite(z[i]):
            pos[i] = 0
            continue

        if pos[i-1] == 0:
            # 진입 조건: z-score가 임계값을 넘을 때
            if z[i] <= -entry:
                pos[i] = 1
            elif z[i] >= entry:
                # 숏 허용 여부에 따라 포지션 결정
                pos[i] = -1 if allow_short else 0
        else:
            # 청산 조건: z-score가 0 근처로 회귀
            if abs(z[i]) <= exit_:
                pos[i] = 0

    # 포지션 및 수익률 계산
    ctx["pos"] = pos
    ctx["turnover"] = ctx["pos"].diff().abs().fillna(0)
    ctx["strategy_ret"] = ctx["pos"].shift(1) * ctx["rK"]
    ctx["strategy_ret_net"] = ctx["strategy_ret"].fillna(0) - TC * ctx["turnover"]
    ctx["equity"] = (1 + ctx["strategy_ret_net"]).cumprod()

    # 최종 구간만 반환
    return ctx.loc[start:end]


# =========================
# METRICS
# =========================
def perf_stats(ret):
    # 성과 지표 계산(연환산 수익률/변동성, 샤프, MDD)
    ann = 252
    r = ret.dropna()
    # 표본이 너무 적으면 신뢰 어려움 -> NaN 반환
    if len(r) < 30:
        return dict(sharpe=np.nan, ann_ret=np.nan, ann_vol=np.nan, mdd=np.nan)

    ann_ret = r.mean() * ann
    ann_vol = r.std() * np.sqrt(ann)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else np.nan
    mdd = ((1 + r).cumprod() / (1 + r).cumprod().cummax() - 1).min()

    return dict(sharpe=sharpe, ann_ret=ann_ret, ann_vol=ann_vol, mdd=mdd)


@dataclass
class WFResult:
    # 워크-포워드 구간별 결과 저장용 구조체
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    best_params: dict
    train_stats: dict
    test_stats: dict


# =========================
# WALK-FORWARD GRID SEARCH
# =========================
def walk_forward_grid_search(df):

    results = []
    oos_parts = []

    # 전체 데이터 구간 설정
    start = df.index.min()
    end   = df.index.max()
    test_start = start + pd.DateOffset(years=TRAIN_YEARS)

    # 테스트 시작일을 앞으로 이동하며 반복(워크-포워드)
    while test_start < end:
        train_start = test_start - pd.DateOffset(years=TRAIN_YEARS)
        train_end   = test_start - pd.Timedelta(days=1)
        test_end    = min(test_start + pd.DateOffset(months=TEST_MONTHS) - pd.Timedelta(days=1), end)

        # 학습/검증 데이터 분리
        train_df = df.loc[train_start:train_end].dropna()
        test_df  = df.loc[test_start:test_end].dropna()

        # 데이터가 충분하지 않으면 다음 구간으로 이동
        if len(train_df) < 300 or len(test_df) < 60:
            test_start += pd.DateOffset(months=STEP_MONTHS)
            continue

        best_score = -np.inf
        best_params = None
        best_train_stats = None

        # 모든 파라미터 조합을 탐색(그리드 서치)
        for entry, exit_, vq, fq in product(ENTRY_GRID, EXIT_GRID, VIX_Q_GRID, FX_Q_GRID):
            # exit이 entry보다 크거나 같으면 논리적으로 부적합
            if exit_ >= entry:
                continue

            # 학습 구간에서 전략 시뮬레이션
            sim = simulate_strategy_oos(
                df, train_df.index[0], train_df.index[-1],
                entry, exit_, vq, fq, ALLOW_SHORT
            )

            stats = perf_stats(sim["strategy_ret_net"])
            if not np.isfinite(stats["sharpe"]):
                continue

            # 점수 = 샤프 - (MDD 가중 패널티)
            score = stats["sharpe"] - 0.5 * abs(stats["mdd"])

            if score > best_score:
                best_score = score
                best_params = dict(ENTRY=entry, EXIT=exit_, VIX_Q=vq, FX_Q=fq)
                best_train_stats = stats

        # 최적 파라미터로 테스트 구간 OOS 시뮬레이션
        sim_oos = simulate_strategy_oos(
            df, test_start, test_end,
            best_params["ENTRY"], best_params["EXIT"],
            best_params["VIX_Q"], best_params["FX_Q"],
            ALLOW_SHORT
        )

        test_stats = perf_stats(sim_oos["strategy_ret_net"])

        # 결과 저장
        results.append(WFResult(
            train_start, train_end,
            test_start, test_end,
            best_params, best_train_stats, test_stats
        ))

        # OOS 수익률만 합치기 위해 따로 저장
        oos_parts.append(sim_oos[["strategy_ret_net"]])
        test_start += pd.DateOffset(months=STEP_MONTHS)

    # 전체 OOS 구간을 하나로 합치고 누적 수익률 계산
    oos = pd.concat(oos_parts).sort_index()
    oos["oos_equity"] = (1 + oos["strategy_ret_net"]).cumprod()

    return results, oos


# =========================
# MAIN
# =========================
def main():
    print("Loading data...")
    # 원본 데이터를 로드하고 피처 생성
    df_raw = read_excel_data(PATH)
    df = prepare_features(df_raw)

    print("Running walk-forward grid search...")
    # 워크-포워드 그리드 서치 실행
    wf_results, oos = walk_forward_grid_search(df)

    # 각 테스트 구간별 결과 출력
    for i, r in enumerate(wf_results, 1):
        print(f"[{i}] Test {r.test_start.date()}~{r.test_end.date()} | "
              f"Best {r.best_params} | OOS Sharpe {r.test_stats['sharpe']:.2f}")

    # 전체 OOS 성과 요약
    overall = perf_stats(oos["strategy_ret_net"])
    print("\n=== Overall OOS ===")
    print(overall)

    # 결과 저장 폴더 생성(타임스탬프)
    base_dir = Path("grid_search")
    base_dir.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = base_dir / stamp
    suffix = 1
    while out_dir.exists():
        out_dir = base_dir / f"{stamp}_{suffix:02d}"
        suffix += 1
    out_dir.mkdir()

    # OOS 에쿼티와 구간별 최적 파라미터 저장
    oos.to_csv(out_dir / "oos_equity.csv")
    pd.DataFrame(
        [
            {
                "train_start": r.train_start,
                "test_start": r.test_start,
                **r.best_params,
                **r.test_stats,
            }
            for r in wf_results
        ]
    ).to_csv(out_dir / "wf_params_by_segment.csv", index=False)

    print(f"Saved oos_equity.csv, wf_params_by_segment.csv to: {out_dir}")


if __name__ == "__main__":
    main()
