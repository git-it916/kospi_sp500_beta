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

import pandas as pd
import numpy as np
from dataclasses import dataclass
from itertools import product


# =========================
# USER CONFIG
# =========================
PATH = r"C:\Users\10845\OneDrive - 이지스자산운용\문서\kospi_sp500_filtered.xlsx"

DATE_COL = "공통날짜"
REQUIRED_COLS = ["kospi_t", "SPX_t-1", "VIX_t-1", "FX_t"]

BETA_W = 60
RES_W  = 60
Q_W    = 252        # VIX / FX rolling window

TC = 0.0002         # transaction cost

TRAIN_YEARS = 2
TEST_MONTHS = 6
STEP_MONTHS = 6

ENTRY_GRID = [0.8, 1.0, 1.2, 1.5]
EXIT_GRID  = [0.1, 0.2, 0.3, 0.4]
VIX_Q_GRID = [0.75, 0.80, 0.85, 0.90]
FX_Q_GRID  = [0.75, 0.80, 0.85, 0.90]

ALLOW_SHORT = True


# =========================
# DATA LOAD
# =========================
def to_float(x):
    if isinstance(x, str):
        x = x.replace(",", "").strip()
    return pd.to_numeric(x, errors="coerce")


def read_excel_data(path):
    df = pd.read_excel(path)
    df.columns = [c.strip() for c in df.columns]

    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    for c in REQUIRED_COLS:
        df[c] = df[c].apply(to_float)

    df = df.sort_values(DATE_COL).set_index(DATE_COL)
    df = df.dropna(subset=REQUIRED_COLS)
    return df


# =========================
# FEATURE ENGINEERING
# =========================
def rolling_beta(y, x, window):
    return y.rolling(window).cov(x) / x.rolling(window).var()


def prepare_features(df):
    df = df.copy()

    df["rK"]  = np.log(df["kospi_t"]).diff()
    df["rS"]  = np.log(df["SPX_t-1"]).diff()
    df["rFX"] = np.log(df["FX_t"]).diff()

    df["beta"] = rolling_beta(df["rK"], df["rS"], BETA_W)
    df["resid"] = df["rK"] - df["beta"] * df["rS"]

    df["resid_mean"] = df["resid"].rolling(RES_W).mean()
    df["resid_std"]  = df["resid"].rolling(RES_W).std()
    df["z"] = (df["resid"] - df["resid_mean"]) / df["resid_std"]

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

    # 충분한 과거 컨텍스트 확보
    context_start = start - pd.Timedelta(days=400)
    ctx = df_full.loc[context_start:end].copy()

    # rolling threshold는 ctx 전체에서 계산
    vix_th = ctx["VIX_t-1"].rolling(Q_W).quantile(vix_q)
    fx_th  = ctx["fx_z"].abs().rolling(Q_W).quantile(fx_q)

    allow = (ctx["VIX_t-1"] <= vix_th) & (ctx["fx_z"].abs() <= fx_th)

    z = ctx["z"].values
    allow_v = allow.values

    pos = np.zeros(len(ctx))

    for i in range(1, len(ctx)):
        pos[i] = pos[i-1]

        if not allow_v[i] or not np.isfinite(z[i]):
            pos[i] = 0
            continue

        if pos[i-1] == 0:
            if z[i] <= -entry:
                pos[i] = 1
            elif z[i] >= entry:
                pos[i] = -1 if allow_short else 0
        else:
            if abs(z[i]) <= exit_:
                pos[i] = 0

    ctx["pos"] = pos
    ctx["turnover"] = ctx["pos"].diff().abs().fillna(0)
    ctx["strategy_ret"] = ctx["pos"].shift(1) * ctx["rK"]
    ctx["strategy_ret_net"] = ctx["strategy_ret"].fillna(0) - TC * ctx["turnover"]
    ctx["equity"] = (1 + ctx["strategy_ret_net"]).cumprod()

    return ctx.loc[start:end]


# =========================
# METRICS
# =========================
def perf_stats(ret):
    ann = 252
    r = ret.dropna()
    if len(r) < 30:
        return dict(sharpe=np.nan, ann_ret=np.nan, ann_vol=np.nan, mdd=np.nan)

    ann_ret = r.mean() * ann
    ann_vol = r.std() * np.sqrt(ann)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else np.nan
    mdd = ((1 + r).cumprod() / (1 + r).cumprod().cummax() - 1).min()

    return dict(sharpe=sharpe, ann_ret=ann_ret, ann_vol=ann_vol, mdd=mdd)


@dataclass
class WFResult:
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

    start = df.index.min()
    end   = df.index.max()
    test_start = start + pd.DateOffset(years=TRAIN_YEARS)

    while test_start < end:
        train_start = test_start - pd.DateOffset(years=TRAIN_YEARS)
        train_end   = test_start - pd.Timedelta(days=1)
        test_end    = min(test_start + pd.DateOffset(months=TEST_MONTHS) - pd.Timedelta(days=1), end)

        train_df = df.loc[train_start:train_end].dropna()
        test_df  = df.loc[test_start:test_end].dropna()

        if len(train_df) < 300 or len(test_df) < 60:
            test_start += pd.DateOffset(months=STEP_MONTHS)
            continue

        best_score = -np.inf
        best_params = None
        best_train_stats = None

        for entry, exit_, vq, fq in product(ENTRY_GRID, EXIT_GRID, VIX_Q_GRID, FX_Q_GRID):
            if exit_ >= entry:
                continue

            sim = simulate_strategy_oos(
                df, train_df.index[0], train_df.index[-1],
                entry, exit_, vq, fq, ALLOW_SHORT
            )

            stats = perf_stats(sim["strategy_ret_net"])
            if not np.isfinite(stats["sharpe"]):
                continue

            score = stats["sharpe"] - 0.5 * abs(stats["mdd"])

            if score > best_score:
                best_score = score
                best_params = dict(ENTRY=entry, EXIT=exit_, VIX_Q=vq, FX_Q=fq)
                best_train_stats = stats

        sim_oos = simulate_strategy_oos(
            df, test_start, test_end,
            best_params["ENTRY"], best_params["EXIT"],
            best_params["VIX_Q"], best_params["FX_Q"],
            ALLOW_SHORT
        )

        test_stats = perf_stats(sim_oos["strategy_ret_net"])

        results.append(WFResult(
            train_start, train_end,
            test_start, test_end,
            best_params, best_train_stats, test_stats
        ))

        oos_parts.append(sim_oos[["strategy_ret_net"]])
        test_start += pd.DateOffset(months=STEP_MONTHS)

    oos = pd.concat(oos_parts).sort_index()
    oos["oos_equity"] = (1 + oos["strategy_ret_net"]).cumprod()

    return results, oos


# =========================
# MAIN
# =========================
def main():
    print("Loading data...")
    df_raw = read_excel_data(PATH)
    df = prepare_features(df_raw)

    print("Running walk-forward grid search...")
    wf_results, oos = walk_forward_grid_search(df)

    for i, r in enumerate(wf_results, 1):
        print(f"[{i}] Test {r.test_start.date()}~{r.test_end.date()} | "
              f"Best {r.best_params} | OOS Sharpe {r.test_stats['sharpe']:.2f}")

    overall = perf_stats(oos["strategy_ret_net"])
    print("\n=== Overall OOS ===")
    print(overall)

    oos.to_csv("oos_equity.csv")
    pd.DataFrame([{
        "train_start": r.train_start,
        "test_start": r.test_start,
        **r.best_params,
        **r.test_stats
    } for r in wf_results]).to_csv("wf_params_by_segment.csv", index=False)

    print("Saved oos_equity.csv, wf_params_by_segment.csv")


if __name__ == "__main__":
    main()
