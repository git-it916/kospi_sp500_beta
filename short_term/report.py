import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import matplotlib

# GUI 없이 저장만 하기 위해 Agg 백엔드 사용(서버/CLI 환경용)
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# base.py의 데이터 로딩/전략 계산 함수 재사용
from base import PATH as DEFAULT_DATA_PATH, load_data, compute_strategy, compute_summary


def compute_report_metrics(df):
    # base.py에서 계산한 요약 지표를 다시 꺼내서 정리
    summary = compute_summary(df)
    mean = summary["ann_return"]
    vol = summary["ann_vol"]
    sharpe = summary["sharpe"]
    mdd = summary["mdd"]
    hit = summary["hit_ratio"]
    # 포지션 보유 비율(시장에 노출된 날짜 비중)
    exposure = (df["pos"] != 0).mean()
    # 마지막 누적 수익률(에쿼티 기준)
    total_return = df["equity"].iloc[-1] - 1
    return mean, vol, sharpe, mdd, hit, exposure, total_return


def ensure_output_dir(base_dir):
    # 출력 폴더가 없으면 생성
    base_dir.mkdir(parents=True, exist_ok=True)
    # 타임스탬프 기반 폴더 생성
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = base_dir / stamp
    suffix = 1
    # 동일 이름 폴더가 있으면 뒤에 번호를 붙여서 중복 회피
    while out_dir.exists():
        out_dir = base_dir / f"{stamp}_{suffix:02d}"
        suffix += 1
    out_dir.mkdir()
    return out_dir


def save_plots(df, annual, quarterly, out_dir):
    # 1) 누적 수익률(에쿼티) 그래프 저장
    plt.figure(figsize=(10, 4))
    plt.plot(df.index, df["equity"], color="#0B3C5D")
    plt.title("Strategy Equity Curve")
    plt.xlabel("Date")
    plt.ylabel("Equity")
    plt.tight_layout()
    plt.savefig(out_dir / "equity_curve.png", dpi=150)
    plt.close()

    # 2) 드로우다운 그래프 저장
    plt.figure(figsize=(10, 3))
    drawdown = df["equity"] / df["equity"].cummax() - 1
    plt.plot(df.index, drawdown, color="#B5523B")
    plt.title("Drawdown")
    plt.xlabel("Date")
    plt.ylabel("Drawdown")
    plt.tight_layout()
    plt.savefig(out_dir / "drawdown.png", dpi=150)
    plt.close()

    # 3) 연도별 수익률 막대그래프 저장
    plt.figure(figsize=(10, 4))
    annual.sort_index().plot(kind="bar", color="#2C6E49")
    plt.title("Annual Returns")
    plt.xlabel("Year")
    plt.ylabel("Return")
    plt.tight_layout()
    plt.savefig(out_dir / "annual_returns.png", dpi=150)
    plt.close()

    # 4) 분기별 수익률 히트맵 생성
    q = quarterly.copy()
    # 분기 PeriodIndex를 실제 날짜 인덱스로 변환
    q.index = q.index.to_timestamp()
    heat = q.to_frame("ret")
    # 히트맵을 그리기 위해 연도/분기 컬럼 분리
    heat["year"] = heat.index.year
    heat["quarter"] = heat.index.quarter
    pivot = heat.pivot(index="year", columns="quarter", values="ret").sort_index()

    plt.figure(figsize=(8, 6))
    # 색상 범위를 대칭으로 맞춰 양/음 수익률을 균형 있게 표현
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
    # 텍스트 요약 파일 작성
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

    # 연도/분기별 수익률 CSV 저장
    annual.to_csv(out_dir / "annual_returns.csv", header=["return"])
    quarterly.to_csv(out_dir / "quarterly_returns.csv", header=["return"])


def parse_args():
    # 커맨드라인 인자 정의
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
    # 데이터 로드 및 전략 계산
    df = load_data(args.path)
    df = compute_strategy(df)

    # 연도/분기별 수익률 계산
    annual = df["strategy_ret_net"].groupby(df.index.year).apply(
        lambda x: (1 + x).prod() - 1
    )
    quarterly = df["strategy_ret_net"].groupby(df.index.to_period("Q")).apply(
        lambda x: (1 + x).prod() - 1
    )

    # 출력 폴더 생성 후 리포트 생성
    out_dir = ensure_output_dir(Path(args.out_dir))
    metrics = compute_report_metrics(df)
    write_summary(df, metrics, annual, quarterly, out_dir)
    save_plots(df, annual, quarterly, out_dir)

    print(f"Report saved to: {out_dir}")


if __name__ == "__main__":
    main()
