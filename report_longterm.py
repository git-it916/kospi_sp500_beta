import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import base_longterm as lt

PATH = lt.PATH
ROLLING_WINDOW = 252


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


def max_drawdown_duration(equity):
    drawdown = equity / equity.cummax() - 1
    underwater = drawdown < 0
    max_len = 0
    current = 0
    for v in underwater:
        if v:
            current += 1
            if current > max_len:
                max_len = current
        else:
            current = 0
    return max_len


def rolling_metrics(ret, window=ROLLING_WINDOW):
    r = ret.dropna()
    rolling_mean = r.rolling(window).mean()
    rolling_std = r.rolling(window).std()
    rolling_ann_ret = rolling_mean * 252
    rolling_ann_vol = rolling_std * np.sqrt(252)
    rolling_ann_vol = rolling_ann_vol.replace(0, np.nan)
    rolling_sharpe = rolling_ann_ret / rolling_ann_vol

    return pd.DataFrame(
        {
            "rolling_ann_ret": rolling_ann_ret,
            "rolling_ann_vol": rolling_ann_vol,
            "rolling_sharpe": rolling_sharpe,
        }
    )


def compute_summary(df):
    if df.empty:
        return {}

    base = lt.compute_summary(df)
    ret = df["strategy_ret_net"].dropna()
    equity = df["equity"]

    total_return = equity.iloc[-1] - 1
    hit_ratio = (ret > 0).mean()
    exposure = (df["pos"] != 0).mean()
    avg_turnover = df["turnover"].mean()
    best_day = ret.max()
    worst_day = ret.min()
    skew = ret.skew()
    kurt = ret.kurt()
    calmar = base["ann_return"] / abs(base["mdd"]) if base["mdd"] < 0 else np.nan
    max_dd_days = max_drawdown_duration(equity)

    return {
        "rows": len(df),
        "start": df.index.min().date(),
        "end": df.index.max().date(),
        "ann_ret": base["ann_return"],
        "ann_vol": base["ann_vol"],
        "sharpe": base["sharpe"],
        "mdd": base["mdd"],
        "total_return": total_return,
        "hit_ratio": hit_ratio,
        "exposure": exposure,
        "avg_turnover": avg_turnover,
        "best_day": best_day,
        "worst_day": worst_day,
        "skew": skew,
        "kurt": kurt,
        "calmar": calmar,
        "max_dd_days": max_dd_days,
    }


def compute_period_returns(ret):
    annual = ret.groupby(ret.index.year).apply(lambda x: (1 + x).prod() - 1)
    quarterly = ret.groupby(ret.index.to_period("Q")).apply(lambda x: (1 + x).prod() - 1)
    monthly = ret.groupby(ret.index.to_period("M")).apply(lambda x: (1 + x).prod() - 1)
    return annual, quarterly, monthly


def write_summary(out_dir, summary, rolling, monthly):
    last_roll = rolling.dropna().tail(1)
    last_roll_ret = last_roll["rolling_ann_ret"].iloc[0] if len(last_roll) else np.nan
    last_roll_vol = last_roll["rolling_ann_vol"].iloc[0] if len(last_roll) else np.nan
    last_roll_sharpe = last_roll["rolling_sharpe"].iloc[0] if len(last_roll) else np.nan

    monthly_win = (monthly > 0).mean() if len(monthly) else np.nan

    lines = [
        "[BASIC]",
        f"Rows: {summary['rows']}",
        f"Period: {summary['start']} to {summary['end']}",
        f"Ann.Return: {summary['ann_ret']:.3%}",
        f"Ann.Vol: {summary['ann_vol']:.3%}",
        f"Sharpe: {summary['sharpe']:.2f}",
        f"MDD: {summary['mdd']:.2%}",
        f"Total Return: {summary['total_return']:.2%}",
        "",
        "[DETAIL]",
        f"Hit Ratio: {summary['hit_ratio']:.2%}",
        f"Exposure: {summary['exposure']:.2%}",
        f"Avg Turnover: {summary['avg_turnover']:.4f}",
        f"Best Day: {summary['best_day']:.2%}",
        f"Worst Day: {summary['worst_day']:.2%}",
        f"Skew: {summary['skew']:.2f}",
        f"Kurtosis: {summary['kurt']:.2f}",
        f"Calmar: {summary['calmar']:.2f}",
        f"Max DD Duration (days): {summary['max_dd_days']}",
        "",
        "[EXTENDED]",
        f"Rolling Window: {ROLLING_WINDOW}",
        f"Rolling Ann.Return (last): {last_roll_ret:.3%}",
        f"Rolling Ann.Vol (last): {last_roll_vol:.3%}",
        f"Rolling Sharpe (last): {last_roll_sharpe:.2f}",
        f"Monthly Win Rate: {monthly_win:.2%}",
        "",
        "[PARAMS]",
        f"ENTRY: {lt.ENTRY}",
        f"EXIT: {lt.EXIT}",
        f"VIX_Q: {lt.VIX_Q}",
        f"FX_Q: {lt.FX_Q}",
        f"STOP_LOSS_MULT: {lt.STOP_LOSS_MULT}",
    ]
    (out_dir / "summary.txt").write_text("\n".join(lines), encoding="utf-8")


def plot_basic(df, annual, quarterly, out_dir):
    plt.figure(figsize=(10, 4))
    plt.plot(df.index, df["equity"], color="#0B3C5D")
    plt.title("Equity Curve")
    plt.xlabel("Date")
    plt.ylabel("Equity")
    plt.tight_layout()
    plt.savefig(out_dir / "basic_equity_curve.png", dpi=150)
    plt.close()

    plt.figure(figsize=(10, 3))
    drawdown = df["equity"] / df["equity"].cummax() - 1
    plt.plot(df.index, drawdown, color="#B5523B")
    plt.title("Drawdown")
    plt.xlabel("Date")
    plt.ylabel("Drawdown")
    plt.tight_layout()
    plt.savefig(out_dir / "basic_drawdown.png", dpi=150)
    plt.close()

    plt.figure(figsize=(10, 4))
    annual.sort_index().plot(kind="bar", color="#2C6E49")
    plt.title("Annual Returns")
    plt.xlabel("Year")
    plt.ylabel("Return")
    plt.tight_layout()
    plt.savefig(out_dir / "basic_annual_returns.png", dpi=150)
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
    plt.savefig(out_dir / "basic_quarterly_returns_heatmap.png", dpi=150)
    plt.close()


def plot_detail(df, monthly, out_dir):
    m = monthly.copy()
    m.index = m.index.to_timestamp()
    heat = m.to_frame("ret")
    heat["year"] = heat.index.year
    heat["month"] = heat.index.month
    pivot = heat.pivot(index="year", columns="month", values="ret").sort_index()

    plt.figure(figsize=(10, 6))
    limit = np.nanmax(np.abs(pivot.values))
    limit = 0.05 if not np.isfinite(limit) or limit == 0 else limit
    img = plt.imshow(pivot, aspect="auto", cmap="RdYlGn", vmin=-limit, vmax=limit)
    plt.colorbar(img, label="Return")
    plt.title("Monthly Returns (Heatmap)")
    plt.xlabel("Month")
    plt.ylabel("Year")
    plt.xticks(ticks=np.arange(12), labels=[str(i) for i in range(1, 13)])
    plt.yticks(ticks=np.arange(len(pivot.index)), labels=pivot.index)
    plt.tight_layout()
    plt.savefig(out_dir / "detail_monthly_returns_heatmap.png", dpi=150)
    plt.close()

    ret = df["strategy_ret_net"].dropna()
    plt.figure(figsize=(8, 4))
    plt.hist(ret, bins=60, color="#4C72B0", alpha=0.8, density=True)
    plt.axvline(0, color="#333333", linewidth=1)
    plt.title("Return Distribution")
    plt.xlabel("Daily Return")
    plt.ylabel("Density")
    plt.tight_layout()
    plt.savefig(out_dir / "detail_return_distribution.png", dpi=150)
    plt.close()


def plot_extended(df, rolling, out_dir):
    roll = rolling.dropna()
    if len(roll) > 0:
        plt.figure(figsize=(10, 3))
        plt.plot(roll.index, roll["rolling_sharpe"], color="#6F2DBD")
        plt.title("Rolling Sharpe (1Y)")
        plt.xlabel("Date")
        plt.ylabel("Sharpe")
        plt.tight_layout()
        plt.savefig(out_dir / "extended_rolling_sharpe.png", dpi=150)
        plt.close()

        plt.figure(figsize=(10, 3))
        plt.plot(roll.index, roll["rolling_ann_vol"], color="#2A9D8F")
        plt.title("Rolling Volatility (1Y)")
        plt.xlabel("Date")
        plt.ylabel("Ann.Vol")
        plt.tight_layout()
        plt.savefig(out_dir / "extended_rolling_volatility.png", dpi=150)
        plt.close()

        plt.figure(figsize=(10, 3))
        plt.plot(roll.index, roll["rolling_ann_ret"], color="#E76F51")
        plt.title("Rolling Annual Return (1Y)")
        plt.xlabel("Date")
        plt.ylabel("Ann.Return")
        plt.tight_layout()
        plt.savefig(out_dir / "extended_rolling_return.png", dpi=150)
        plt.close()

    exposure_roll = (df["pos"] != 0).rolling(ROLLING_WINDOW).mean()
    plt.figure(figsize=(10, 3))
    plt.plot(exposure_roll.index, exposure_roll, color="#264653")
    plt.title("Rolling Exposure (1Y)")
    plt.xlabel("Date")
    plt.ylabel("Exposure")
    plt.tight_layout()
    plt.savefig(out_dir / "extended_rolling_exposure.png", dpi=150)
    plt.close()

    turnover_roll = df["turnover"].rolling(ROLLING_WINDOW).mean()
    plt.figure(figsize=(10, 3))
    plt.plot(turnover_roll.index, turnover_roll, color="#F4A261")
    plt.title("Rolling Turnover (1Y)")
    plt.xlabel("Date")
    plt.ylabel("Turnover")
    plt.tight_layout()
    plt.savefig(out_dir / "extended_rolling_turnover.png", dpi=150)
    plt.close()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate long-term report with rich visuals."
    )
    parser.add_argument("--path", default=PATH, help="Excel file path")
    parser.add_argument(
        "--out-dir", default="outputs_longterm", help="Base output directory"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    df = lt.load_data(args.path)
    if df.empty:
        print("No data loaded. Check the file path.")
        return

    df = lt.compute_strategy(df)
    if df.empty:
        print("No strategy output generated.")
        return

    ret = df["strategy_ret_net"].dropna()
    annual, quarterly, monthly = compute_period_returns(ret)
    rolling = rolling_metrics(ret)
    summary = compute_summary(df)

    out_dir = ensure_output_dir(Path(args.out_dir))
    df[["strategy_ret_net", "equity", "pos", "turnover"]].to_csv(
        out_dir / "strategy_timeseries.csv"
    )
    annual.to_csv(out_dir / "annual_returns.csv", header=["return"])
    quarterly.to_csv(out_dir / "quarterly_returns.csv", header=["return"])
    monthly.to_csv(out_dir / "monthly_returns.csv", header=["return"])
    rolling.to_csv(out_dir / "rolling_metrics.csv")

    write_summary(out_dir, summary, rolling, monthly)
    plot_basic(df, annual, quarterly, out_dir)
    plot_detail(df, monthly, out_dir)
    plot_extended(df, rolling, out_dir)

    print(f"Report saved to: {out_dir}")


if __name__ == "__main__":
    main()
