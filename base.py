import pandas as pd
import numpy as np

# =============================
# CONFIG
# =============================
PATH = r"C:\Users\10845\OneDrive - 이지스자산운용\문서\kospi_sp500_filtered.xlsx"

DATE_COL = "공통날짜"
REQUIRED_COLS = ["kospi_t", "SPX_t-1", "VIX_t-1", "FX_t"]

BETA_W = 60
RES_W = 60
VIX_W = 252
FX_W = 252

ENTRY = 1.0
EXIT = 0.2
TC = 0.0002


def to_float(x):
    # "6,609.3" 같은 콤마 제거
    if isinstance(x, str):
        x = x.replace(",", "").strip()
    return pd.to_numeric(x, errors="coerce")


def load_data(path=PATH):
    # -----------------------------
    # 1) Load
    # -----------------------------
    df = pd.read_excel(path)

    # 컬럼 정리
    df.columns = [c.strip() for c in df.columns]
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])

    for c in REQUIRED_COLS:
        df[c] = df[c].apply(to_float)

    return df.sort_values(DATE_COL).set_index(DATE_COL).dropna()


def rolling_beta(y, x, window):
    cov = y.rolling(window).cov(x)
    var = x.rolling(window).var()
    return cov / var


def compute_strategy(
    df,
    beta_w=BETA_W,
    res_w=RES_W,
    vix_w=VIX_W,
    fx_w=FX_W,
    entry=ENTRY,
    exit_=EXIT,
    tc=TC,
):
    out = df.copy()

    # -----------------------------
    # 2) Returns (log)
    # -----------------------------
    out["rK"] = np.log(out["kospi_t"]).diff()
    out["rS"] = np.log(out["SPX_t-1"]).diff()
    out["rFX"] = np.log(out["FX_t"]).diff()

    # -----------------------------
    # 3) Rolling beta (OLS closed-form)
    # -----------------------------
    out["beta"] = rolling_beta(out["rK"], out["rS"], beta_w)

    # -----------------------------
    # 4) Residual + zscore signal
    # -----------------------------
    out["resid"] = out["rK"] - out["beta"] * out["rS"]
    out["resid_mean"] = out["resid"].rolling(res_w).mean()
    out["resid_std"] = out["resid"].rolling(res_w).std()
    out["z"] = (out["resid"] - out["resid_mean"]) / out["resid_std"]

    # -----------------------------
    # 5) Thresholds (walk-forward friendly defaults)
    # -----------------------------
    out["vix_p80"] = out["VIX_t-1"].rolling(vix_w).quantile(0.80)

    out["fx_mean"] = out["rFX"].rolling(fx_w).mean()
    out["fx_std"] = out["rFX"].rolling(fx_w).std()
    out["fx_z"] = (out["rFX"] - out["fx_mean"]) / out["fx_std"]
    out["fx_abs_p90"] = out["fx_z"].abs().rolling(fx_w).quantile(0.90)

    out["allow"] = (out["VIX_t-1"] <= out["vix_p80"]) & (
        out["fx_z"].abs() <= out["fx_abs_p90"]
    )

    # -----------------------------
    # 6) Trading rule (Strategy A: residual mean reversion)
    # -----------------------------
    pos = np.zeros(len(out))
    z = out["z"].values
    allow = out["allow"].values

    for i in range(1, len(out)):
        pos[i] = pos[i - 1]
        if not allow[i]:
            pos[i] = 0
            continue
        if pos[i - 1] == 0:
            if z[i] <= -entry:
                pos[i] = +1
            elif z[i] >= +entry:
                pos[i] = -1
        else:
            if abs(z[i]) <= exit_:
                pos[i] = 0

    out["pos"] = pos

    # -----------------------------
    # 7) Backtest PnL (next-day execution 가정)
    # -----------------------------
    out["strategy_ret"] = out["pos"].shift(1) * out["rK"]

    out["turnover"] = out["pos"].diff().abs()
    out["strategy_ret_net"] = out["strategy_ret"] - tc * out["turnover"]

    out["equity"] = (1 + out["strategy_ret_net"].fillna(0)).cumprod()
    return out


def compute_summary(df):
    ann_factor = 252
    mean = df["strategy_ret_net"].mean() * ann_factor
    vol = df["strategy_ret_net"].std() * np.sqrt(ann_factor)
    sharpe = mean / vol if vol > 0 else np.nan
    mdd = (df["equity"] / df["equity"].cummax() - 1).min()
    hit = (df["strategy_ret_net"] > 0).mean()
    return {
        "ann_return": mean,
        "ann_vol": vol,
        "sharpe": sharpe,
        "mdd": mdd,
        "hit_ratio": hit,
    }


def run_backtest(path=PATH):
    df = load_data(path)
    df = compute_strategy(df)
    metrics = compute_summary(df)
    return df, metrics


def print_summary(df, metrics):
    print("Rows:", len(df))
    print(f"Ann.Return: {metrics['ann_return']:.3%}")
    print(f"Ann.Vol:    {metrics['ann_vol']:.3%}")
    print(f"Sharpe:     {metrics['sharpe']:.2f}")
    print(f"MDD:        {metrics['mdd']:.2%}")
    print("Hit ratio:", metrics["hit_ratio"])


def main():
    df, metrics = run_backtest(PATH)
    print_summary(df, metrics)


if __name__ == "__main__":
    main()
