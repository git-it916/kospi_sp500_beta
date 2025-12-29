import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

DEFAULT_DATA_PATH = (
    "C:\\Users\\10845\\OneDrive - 이지스자산운용\\문서\\kospi_sp500_filtered.xlsx"
)


def to_float(x):
    if isinstance(x, str):
        x = x.replace(",", "").strip()
    return pd.to_numeric(x, errors="coerce")


def load_data(path):
    df = pd.read_excel(path)
    df.columns = [c.strip() for c in df.columns]
    df["공통날짜"] = pd.to_datetime(df["공통날짜"])
    for c in ["kospi_t", "SPX_t-1", "VIX_t-1", "FX_t"]:
        df[c] = df[c].apply(to_float)
    return df.sort_values("공통날짜").set_index("공통날짜").dropna()


def compute_strategy(df):
    out = df.copy()

    out["rK"] = np.log(out["kospi_t"]).diff()
    out["rS"] = np.log(out["SPX_t-1"]).diff()
    out["rFX"] = np.log(out["FX_t"]).diff()

    beta_w = 60
    out["beta"] = out["rK"].rolling(beta_w).cov(out["rS"]) / out["rS"].rolling(
        beta_w
    ).var()

    res_w = 60
    out["resid"] = out["rK"] - out["beta"] * out["rS"]
    out["resid_mean"] = out["resid"].rolling(res_w).mean()
    out["resid_std"] = out["resid"].rolling(res_w).std()
    out["z"] = (out["resid"] - out["resid_mean"]) / out["resid_std"]

    vix_w = 252
    fx_w = 252
    out["vix_p80"] = out["VIX_t-1"].rolling(vix_w).quantile(0.80)

    out["fx_mean"] = out["rFX"].rolling(fx_w).mean()
    out["fx_std"] = out["rFX"].rolling(fx_w).std()
    out["fx_z"] = (out["rFX"] - out["fx_mean"]) / out["fx_std"]
    out["fx_abs_p90"] = out["fx_z"].abs().rolling(fx_w).quantile(0.90)

    out["allow"] = (out["VIX_t-1"] <= out["vix_p80"]) & (
        out["fx_z"].abs() <= out["fx_abs_p90"]
    )

    entry = 1.0
    exit_z = 0.2

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
            if abs(z[i]) <= exit_z:
                pos[i] = 0

    out["pos"] = pos
    out["strategy_ret"] = out["pos"].shift(1) * out["rK"]

    tc = 0.0002
    out["turnover"] = out["pos"].diff().abs()
    out["strategy_ret_net"] = out["strategy_ret"] - tc * out["turnover"]

    out["equity"] = (1 + out["strategy_ret_net"].fillna(0)).cumprod()
    return out


def compute_metrics(df):
    ann_factor = 252
    mean = df["strategy_ret_net"].mean() * ann_factor
    vol = df["strategy_ret_net"].std() * np.sqrt(ann_factor)
    sharpe = mean / vol if vol > 0 else np.nan
    mdd = (df["equity"] / df["equity"].cummax() - 1).min()
    hit = (df["strategy_ret_net"] > 0).mean()
    exposure = (df["pos"] != 0).mean()
    total_return = df["equity"].iloc[-1] - 1
    return mean, vol, sharpe, mdd, hit, exposure, total_return


def ensure_output_dir(base_dir):
    base_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = base_dir / stamp
    suffix = 1
    while out_dir.exists():
        out_dir = base_dir / f"{stamp}_{suffix:02d}"
        suffix += 1
    out_dir.mkdir()
    return out_dir


def save_plots(df, annual, quarterly, out_dir):
    plt.figure(figsize=(10, 4))
    plt.plot(df.index, df["equity"], color="#0B3C5D")
    plt.title("Strategy Equity Curve")
    plt.xlabel("Date")
    plt.ylabel("Equity")
    plt.tight_layout()
    plt.savefig(out_dir / "equity_curve.png", dpi=150)
    plt.close()

    plt.figure(figsize=(10, 3))
    drawdown = df["equity"] / df["equity"].cummax() - 1
    plt.plot(df.index, drawdown, color="#B5523B")
    plt.title("Drawdown")
    plt.xlabel("Date")
    plt.ylabel("Drawdown")
    plt.tight_layout()
    plt.savefig(out_dir / "drawdown.png", dpi=150)
    plt.close()

    plt.figure(figsize=(10, 4))
    annual.sort_index().plot(kind="bar", color="#2C6E49")
    plt.title("Annual Returns")
    plt.xlabel("Year")
    plt.ylabel("Return")
    plt.tight_layout()
    plt.savefig(out_dir / "annual_returns.png", dpi=150)
    plt.close()

    q = quarterly.copy()
    q.index = q.index.to_timestamp()
    heat = q.to_frame("ret")
    heat["year"] = heat.index.year
    heat["quarter"] = heat.index.quarter
    pivot = heat.pivot(index="year", columns="quarter", values="ret").sort_index()

    plt.figure(figsize=(8, 6))
    limit = np.nanmax(np.abs(pivot.values))
    limit = 0.1 if not np.isfinite(limit) or limit == 0 else limit
    img = plt.imshow(pivot, aspect="auto", cmap="RdYlGn", vmin=-limit, vmax=limit)
    plt.colorbar(img, label="Return")
    plt.title("Quarterly Returns (Heatmap)")
    plt.xlabel("Quarter")
    plt.ylabel("Year")
    plt.xticks(ticks=[0, 1, 2, 3], labels=["Q1", "Q2", "Q3", "Q4"])
    plt.yticks(ticks=np.arange(len(pivot.index)), labels=pivot.index)
    plt.tight_layout()
    plt.savefig(out_dir / "quarterly_returns_heatmap.png", dpi=150)
    plt.close()


def write_summary(df, metrics, annual, quarterly, out_dir):
    mean, vol, sharpe, mdd, hit, exposure, total_return = metrics
    lines = [
        f"Rows: {len(df)}",
        f"Period: {df.index.min().date()} to {df.index.max().date()}",
        f"Ann.Return: {mean:.3%}",
        f"Ann.Vol: {vol:.3%}",
        f"Sharpe: {sharpe:.2f}",
        f"MDD: {mdd:.2%}",
        f"Total Return: {total_return:.2%}",
        f"Exposure: {exposure:.2%}",
        f"Hit Ratio: {hit:.2%}",
    ]
    (out_dir / "summary.txt").write_text("\n".join(lines), encoding="utf-8")

    annual.to_csv(out_dir / "annual_returns.csv", header=["return"])
    quarterly.to_csv(out_dir / "quarterly_returns.csv", header=["return"])


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate strategy report outputs with timestamped folders."
    )
    parser.add_argument("--path", default=DEFAULT_DATA_PATH, help="Excel file path")
    parser.add_argument(
        "--out-dir", default="outputs", help="Base output directory"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    df = load_data(args.path)
    df = compute_strategy(df)

    annual = df["strategy_ret_net"].groupby(df.index.year).apply(
        lambda x: (1 + x).prod() - 1
    )
    quarterly = df["strategy_ret_net"].groupby(df.index.to_period("Q")).apply(
        lambda x: (1 + x).prod() - 1
    )

    out_dir = ensure_output_dir(Path(args.out_dir))
    metrics = compute_metrics(df)
    write_summary(df, metrics, annual, quarterly, out_dir)
    save_plots(df, annual, quarterly, out_dir)

    print(f"Report saved to: {out_dir}")


if __name__ == "__main__":
    main()
