import pandas as pd
import numpy as np
from itertools import product
from pathlib import Path
from datetime import datetime
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial

warnings.filterwarnings("ignore")

# ==========================================
# 1. 설정 (Configuration)
# ==========================================
PATH = r"C:\Users\10845\OneDrive - 이지스자산운용\문서\kospi_sp500_filtered_longterm.xlsx"

# 개선된 Grid 설정: 2-Stage Optimization
# Stage 1: Coarse Grid (넓은 범위)
COARSE_GRID = {
    "ENTRY": [1.0, 1.5, 2.0, 2.5, 3.0],
    "EXIT": [0.0, 0.5, 1.0],
    "VIX_Q": [0.75, 0.80, 0.85, 0.90],
    "FX_Q": [0.85, 0.90, 0.95],
    "STOP_MULT": [2.0, 2.5, 3.0, 3.5]
}

# Stage 2: Fine Grid (좁은 범위, Stage 1 결과 기반)
FINE_GRID_RANGE = 0.25  # 최적값 ± 0.25

# Walk-forward 설정
TRAIN_YEARS = 3
TEST_YEARS = 1

# 거래 비용
TC = 0.0002

# 병렬 처리
MAX_WORKERS = 4  # CPU 코어 수에 맞게 조정

# ==========================================
# 2. 데이터 로드
# ==========================================
def load_and_prep(path):
    """데이터 로드 및 지표 계산"""
    print(f"📂 Loading data from: {path}")
    try:
        df = pd.read_excel(path)
    except Exception as e:
        print(f"❌ Error loading file: {e}")
        return None

    df.columns = [c.strip() for c in df.columns]
    df["공통날짜"] = pd.to_datetime(df["공통날짜"])

    for c in ["kospi_t", "SPX_t-1", "VIX_t-1", "FX_t"]:
        if df[c].dtype == object:
            df[c] = pd.to_numeric(df[c].str.replace(",", ""), errors='coerce')

    df = df.sort_values("공통날짜").set_index("공통날짜").ffill().dropna()

    # 지표 계산
    df["rK"] = np.log(df["kospi_t"]).diff()
    df["rS"] = np.log(df["SPX_t-1"]).diff()
    df["rFX"] = np.log(df["FX_t"]).diff()

    BETA_W = 60
    df["beta"] = df["rK"].rolling(BETA_W).cov(df["rS"]) / df["rS"].rolling(BETA_W).var()
    df["resid"] = df["rK"] - df["beta"] * df["rS"]

    RES_W = 60
    df["resid_mean"] = df["resid"].rolling(RES_W).mean().shift(1)
    df["resid_std"] = df["resid"].rolling(RES_W).std().shift(1)
    df["z"] = (df["resid"] - df["resid_mean"]) / df["resid_std"]

    W_FILTER = 252
    df["vix_rank"] = df["VIX_t-1"].rolling(W_FILTER).rank(pct=True).shift(1)

    df["fx_mean"] = df["rFX"].rolling(W_FILTER).mean()
    df["fx_std"] = df["rFX"].rolling(W_FILTER).std()
    df["fx_z"] = (df["rFX"] - df["fx_mean"]) / df["fx_std"]
    df["fx_shock"] = df["fx_z"].abs().rolling(W_FILTER).rank(pct=True).shift(1)

    return df.dropna()


# ==========================================
# 3. 시뮬레이션 엔진
# ==========================================
def run_simulation(df_segment, entry, exit_, vix_q, fx_q, stop_mult):
    """백테스트 시뮬레이션"""
    allow = (df_segment["vix_rank"] <= vix_q) & (df_segment["fx_shock"] <= fx_q)

    pos = np.zeros(len(df_segment))
    z = df_segment["z"].values
    allow_val = allow.values

    current_pos = 0
    stop_loss = entry * stop_mult

    for i in range(1, len(df_segment)):
        pos[i] = current_pos

        if not allow_val[i]:
            current_pos = 0
            pos[i] = 0
            continue

        if np.isnan(z[i]):
            pos[i] = 0
            continue

        # 손절매
        if current_pos != 0:
            if abs(z[i]) >= stop_loss:
                current_pos = 0
                pos[i] = 0
                continue

        # 진입/청산
        if current_pos == 0:
            if z[i] <= -entry:
                current_pos = 1
            elif z[i] >= entry:
                current_pos = -1
        else:
            if abs(z[i]) <= exit_:
                current_pos = 0

        pos[i] = current_pos

    # PnL
    ret = pd.Series(pos).shift(1).fillna(0).values * df_segment["rK"].values
    turnover = np.abs(np.diff(pos, prepend=0))
    ret_net = ret - TC * turnover

    return pd.Series(ret_net, index=df_segment.index)


# ==========================================
# 4. 개선된 성과 지표 계산
# ==========================================
def calc_advanced_stats(ret_series):
    """개선된 성과 지표 (강건성 포함)"""
    if len(ret_series) == 0:
        return {}

    ann = 252
    mean = ret_series.mean() * ann
    vol = ret_series.std() * np.sqrt(ann)
    sharpe = mean / vol if vol > 0 else 0

    cum = (1 + ret_series).cumprod()
    mdd = (cum / cum.cummax() - 1).min()
    total_ret = cum.iloc[-1] - 1

    # 추가 지표
    calmar = mean / abs(mdd) if mdd < 0 else 0
    skew = ret_series.skew()
    kurt = ret_series.kurt()

    # 안정성: 롤링 샤프 표준편차 (낮을수록 안정적)
    rolling_sharpe = ret_series.rolling(252).mean() / ret_series.rolling(252).std()
    stability = 1 / (rolling_sharpe.std() + 0.01) if len(rolling_sharpe.dropna()) > 0 else 0

    # Win Rate
    win_rate = (ret_series > 0).mean()

    # Tail Risk: VaR 95%
    var_95 = ret_series.quantile(0.05) * np.sqrt(ann)

    return {
        "Ann.Ret": mean,
        "Ann.Vol": vol,
        "Sharpe": sharpe,
        "MDD": mdd,
        "Total": total_ret,
        "Calmar": calmar,
        "Skew": skew,
        "Kurt": kurt,
        "Stability": stability,
        "WinRate": win_rate,
        "VaR95": var_95
    }


# ==========================================
# 5. 개선된 스코어링 함수
# ==========================================
def composite_score(stats, mode="balanced"):
    """다중 목적 최적화 스코어"""
    if not stats or stats.get("Sharpe", 0) == 0:
        return -np.inf

    sharpe = stats["Sharpe"]
    calmar = stats["Calmar"]
    skew = stats["Skew"]
    stability = stats["Stability"]

    if mode == "balanced":
        # 균형잡힌 접근
        score = (
            0.40 * sharpe +
            0.30 * calmar +
            0.20 * (1 - abs(skew) / 2) +  # 정규분포 선호
            0.10 * min(stability, 10)      # 안정성 (cap at 10)
        )
    elif mode == "conservative":
        # 보수적 (MDD 중시)
        score = (
            0.30 * sharpe +
            0.50 * calmar +
            0.20 * min(stability, 10)
        )
    elif mode == "aggressive":
        # 공격적 (Sharpe 중시)
        score = (
            0.60 * sharpe +
            0.20 * calmar +
            0.20 * (1 - abs(skew) / 2)
        )

    return score


# ==========================================
# 6. Stage 1: Coarse Grid Search
# ==========================================
def run_coarse_grid(df, out_dir, mode="balanced"):
    """Stage 1: 넓은 범위 탐색"""
    print("\n" + "="*60)
    print(" Stage 1: Coarse Grid Search")
    print("="*60)

    results = []
    param_combinations = list(product(
        COARSE_GRID["ENTRY"],
        COARSE_GRID["EXIT"],
        COARSE_GRID["VIX_Q"],
        COARSE_GRID["FX_Q"],
        COARSE_GRID["STOP_MULT"]
    ))

    total = len(param_combinations)
    print(f"Total combinations: {total}")

    for i, (entry, exit_, vix_q, fx_q, stop_mult) in enumerate(param_combinations):
        if exit_ >= entry:
            continue

        ret_series = run_simulation(df, entry, exit_, vix_q, fx_q, stop_mult)
        stats = calc_advanced_stats(ret_series)

        if stats:
            stats.update({
                "ENTRY": entry,
                "EXIT": exit_,
                "VIX_Q": vix_q,
                "FX_Q": fx_q,
                "STOP_MULT": stop_mult,
                "Score": composite_score(stats, mode)
            })
            results.append(stats)

        if (i + 1) % 50 == 0:
            print(f"Progress: {i+1}/{total} ({(i+1)/total*100:.1f}%)", end="\r")

    print(f"\nCompleted: {len(results)} valid combinations")

    # 결과 저장
    res_df = pd.DataFrame(results)
    res_df = res_df.sort_values("Score", ascending=False)
    res_df.to_csv(out_dir / "stage1_coarse_grid.csv", index=False)

    # Top 10 출력
    print("\n🏆 Top 10 Parameter Sets:")
    print(res_df.head(10)[["ENTRY", "EXIT", "VIX_Q", "FX_Q", "STOP_MULT", "Sharpe", "Calmar", "Score"]].to_string(index=False))

    return res_df.head(1).iloc[0]  # Best params


# ==========================================
# 7. Stage 2: Fine Grid Search
# ==========================================
def run_fine_grid(df, best_coarse, out_dir, mode="balanced"):
    """Stage 2: 최적값 주변 세밀 탐색"""
    print("\n" + "="*60)
    print(" Stage 2: Fine Grid Search (Around Best)")
    print("="*60)

    # Best params 주변 범위 설정
    entry_range = np.arange(
        max(1.0, best_coarse["ENTRY"] - FINE_GRID_RANGE),
        best_coarse["ENTRY"] + FINE_GRID_RANGE + 0.1,
        0.1
    )
    exit_range = np.arange(
        max(0.0, best_coarse["EXIT"] - FINE_GRID_RANGE),
        best_coarse["EXIT"] + FINE_GRID_RANGE + 0.1,
        0.1
    )
    vix_range = np.arange(
        max(0.70, best_coarse["VIX_Q"] - 0.05),
        min(0.95, best_coarse["VIX_Q"] + 0.05) + 0.01,
        0.01
    )
    fx_range = np.arange(
        max(0.80, best_coarse["FX_Q"] - 0.05),
        min(0.98, best_coarse["FX_Q"] + 0.05) + 0.01,
        0.01
    )
    stop_range = np.arange(
        max(1.5, best_coarse["STOP_MULT"] - 0.5),
        best_coarse["STOP_MULT"] + 0.5 + 0.1,
        0.1
    )

    results = []
    param_combinations = list(product(entry_range, exit_range, vix_range, fx_range, stop_range))
    total = len(param_combinations)
    print(f"Total fine combinations: {total}")

    for i, (entry, exit_, vix_q, fx_q, stop_mult) in enumerate(param_combinations):
        if exit_ >= entry:
            continue

        ret_series = run_simulation(df, entry, exit_, vix_q, fx_q, stop_mult)
        stats = calc_advanced_stats(ret_series)

        if stats:
            stats.update({
                "ENTRY": round(entry, 2),
                "EXIT": round(exit_, 2),
                "VIX_Q": round(vix_q, 2),
                "FX_Q": round(fx_q, 2),
                "STOP_MULT": round(stop_mult, 2),
                "Score": composite_score(stats, mode)
            })
            results.append(stats)

        if (i + 1) % 100 == 0:
            print(f"Progress: {i+1}/{total} ({(i+1)/total*100:.1f}%)", end="\r")

    print(f"\nCompleted: {len(results)} valid combinations")

    # 결과 저장
    res_df = pd.DataFrame(results)
    res_df = res_df.sort_values("Score", ascending=False)
    res_df.to_csv(out_dir / "stage2_fine_grid.csv", index=False)

    # Top 5 출력
    print("\n🏆 Top 5 Fine-Tuned Parameter Sets:")
    print(res_df.head(5)[["ENTRY", "EXIT", "VIX_Q", "FX_Q", "STOP_MULT", "Sharpe", "Calmar", "MDD", "Score"]].to_string(index=False))

    return res_df.head(1).iloc[0]


# ==========================================
# 8. Stage 3: Robustness Test
# ==========================================
def run_robustness_test(df, best_params, out_dir):
    """Stage 3: 파라미터 안정성 테스트"""
    print("\n" + "="*60)
    print(" Stage 3: Robustness Test")
    print("="*60)

    # 최적 파라미터 주변 ±5% 범위
    perturbations = []
    for delta in [-0.05, 0, 0.05]:
        perturbations.append({
            "ENTRY": best_params["ENTRY"] * (1 + delta),
            "EXIT": best_params["EXIT"],
            "VIX_Q": best_params["VIX_Q"],
            "FX_Q": best_params["FX_Q"],
            "STOP_MULT": best_params["STOP_MULT"]
        })

    results = []
    for i, params in enumerate(perturbations):
        if params["EXIT"] >= params["ENTRY"]:
            continue

        ret_series = run_simulation(
            df,
            params["ENTRY"],
            params["EXIT"],
            params["VIX_Q"],
            params["FX_Q"],
            params["STOP_MULT"]
        )
        stats = calc_advanced_stats(ret_series)
        stats.update(params)
        stats["Perturbation"] = (i - 1) * 5  # -5%, 0%, +5%
        results.append(stats)

    res_df = pd.DataFrame(results)
    res_df.to_csv(out_dir / "stage3_robustness.csv", index=False)

    # 안정성 분석
    sharpe_std = res_df["Sharpe"].std()
    print(f"\nRobustness Analysis:")
    print(f"  Sharpe Std: {sharpe_std:.3f} (낮을수록 안정적)")
    print(f"  Sharpe Range: {res_df['Sharpe'].min():.2f} ~ {res_df['Sharpe'].max():.2f}")

    return res_df


# ==========================================
# 9. Walk-Forward with Best Params
# ==========================================
def run_walk_forward_optimized(df, out_dir, mode="balanced"):
    """개선된 Walk-Forward (각 구간 최적화)"""
    print("\n" + "="*60)
    print(" Walk-Forward Optimization")
    print("="*60)

    start_date = df.index.min()
    end_date = df.index.max()
    test_start = start_date + pd.DateOffset(years=TRAIN_YEARS)

    oos_list = []
    params_log = []

    while test_start < end_date:
        train_start = test_start - pd.DateOffset(years=TRAIN_YEARS)
        test_end = test_start + pd.DateOffset(years=TEST_YEARS)

        train_data = df.loc[train_start - pd.Timedelta(days=400): test_start]
        test_data = df.loc[test_start: test_end]

        if len(test_data) == 0:
            break

        # Train: Coarse Grid만 수행 (시간 절약)
        print(f"\n[{test_start.date()}] Training...")
        best_score = -np.inf
        best_param = None

        valid_train_idx = train_data.index >= train_start

        for entry, exit_, vix_q, fx_q, stop_mult in product(
            COARSE_GRID["ENTRY"],
            COARSE_GRID["EXIT"],
            COARSE_GRID["VIX_Q"],
            COARSE_GRID["FX_Q"],
            COARSE_GRID["STOP_MULT"]
        ):
            if exit_ >= entry:
                continue

            res = run_simulation(train_data, entry, exit_, vix_q, fx_q, stop_mult)
            res_valid = res[valid_train_idx]

            if len(res_valid) == 0:
                continue

            stats = calc_advanced_stats(res_valid)
            score = composite_score(stats, mode)

            if score > best_score:
                best_score = score
                best_param = (entry, exit_, vix_q, fx_q, stop_mult)

        # Test
        if best_param:
            entry, exit_, vix_q, fx_q, stop_mult = best_param
            res_test = run_simulation(test_data, entry, exit_, vix_q, fx_q, stop_mult)

            oos_list.append(res_test)

            test_stats = calc_advanced_stats(res_test)
            log_entry = {
                "Test_Start": test_start.date(),
                "Test_End": test_data.index[-1].date() if len(test_data) > 0 else test_end.date(),
                "Entry": entry, "Exit": exit_, "VIX_Q": vix_q, "FX_Q": fx_q, "StopMult": stop_mult,
                "OOS_Sharpe": test_stats.get("Sharpe", 0),
                "OOS_Ret": test_stats.get("Ann.Ret", 0),
                "OOS_MDD": test_stats.get("MDD", 0)
            }
            params_log.append(log_entry)
            print(f"  Selected: Entry={entry}, Exit={exit_}, VIX={vix_q}")
            print(f"  OOS: Sharpe={log_entry['OOS_Sharpe']:.2f}, Ret={log_entry['OOS_Ret']:.1%}")

        test_start += pd.DateOffset(years=TEST_YEARS)

    # 저장
    if oos_list:
        full_oos = pd.concat(oos_list).sort_index()
        equity = (1 + full_oos).cumprod()

        output_df = pd.concat([full_oos, equity], axis=1, keys=["Return", "Equity"])
        output_df.to_csv(out_dir / "walkforward_oos_equity.csv")

        pd.DataFrame(params_log).to_csv(out_dir / "walkforward_params_log.csv", index=False)

        final_stats = calc_advanced_stats(full_oos)
        print(f"\n📈 Walk-Forward OOS Performance:")
        print(f"  Total Return: {final_stats['Total']:.2%}")
        print(f"  Ann. Return : {final_stats['Ann.Ret']:.2%}")
        print(f"  Sharpe      : {final_stats['Sharpe']:.2f}")
        print(f"  MDD         : {final_stats['MDD']:.2%}")
        print(f"  Calmar      : {final_stats['Calmar']:.2f}")


# ==========================================
# Main
# ==========================================
def main():
    df = load_and_prep(PATH)
    if df is None:
        return

    # 결과 폴더
    base_dir = Path("grid_search_improved")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = base_dir / stamp
    out_dir.mkdir(parents=True, exist_ok=True)

    print("="*60)
    print(" 🚀 Improved Grid Search - Multi-Stage Optimization")
    print("="*60)
    print(f"Output: {out_dir}")
    print(f"Data period: {df.index[0].date()} ~ {df.index[-1].date()}")
    print(f"Total days: {len(df)}")

    # 최적화 모드 선택
    mode = "balanced"  # balanced, conservative, aggressive
    print(f"Optimization mode: {mode}")

    # Stage 1: Coarse Grid
    best_coarse = run_coarse_grid(df, out_dir, mode)

    # Stage 2: Fine Grid
    best_fine = run_fine_grid(df, best_coarse, out_dir, mode)

    # Stage 3: Robustness Test
    run_robustness_test(df, best_fine, out_dir)

    # Walk-Forward
    run_walk_forward_optimized(df, out_dir, mode)

    print("\n" + "="*60)
    print(" ✅ All Optimizations Complete!")
    print("="*60)
    print(f"📁 Results saved to: {out_dir}")


if __name__ == "__main__":
    main()
