import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from base import PATH as DEFAULT_DATA_PATH, load_data, compute_strategy, compute_summary


def compute_report_metrics(df):
    summary = compute_summary(df)
    mean = summary["ann_return"]
    vol = summary["ann_vol"]
    sharpe = summary["sharpe"]
    mdd = summary["mdd"]
    hit = summary["hit_ratio"]
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
    metrics = compute_report_metrics(df)
    write_summary(df, metrics, annual, quarterly, out_dir)
    save_plots(df, annual, quarterly, out_dir)

    print(f"Report saved to: {out_dir}")


if __name__ == "__main__":
    main()
