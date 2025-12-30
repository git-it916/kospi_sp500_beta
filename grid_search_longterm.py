import pandas as pd
import numpy as np
from itertools import product
from pathlib import Path
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")

# ==========================================
# 1. 설정 (Configuration)
# ==========================================
PATH = r"C:\Users\10845\OneDrive - 이지스자산운용\문서\kospi_sp500_filtered_longterm.xlsx"

# 탐색할 파라미터 범위 (더 세밀하게 확장)
ENTRY_GRID = [1.0, 1.5, 2.0, 2.5]
EXIT_GRID  = [0.0, 0.1, 0.2, 0.5]
VIX_Q_GRID = [0.75, 0.80, 0.85, 0.90]

# 워크포워드 설정
TRAIN_YEARS = 3
TEST_YEARS  = 1

# 거래 비용 & 손절매 설정
TC = 0.0002
STOP_LOSS_MULT = 3.0  # Entry의 3배수 이상 벌어지면 손절 (예: Z > 4.5)

# ==========================================
# 2. 데이터 로드 및 전처리
# ==========================================
def load_and_prep(path):
    print(f"📂 Loading data from: {path}")
    try:
        df = pd.read_excel(path)
    except Exception as e:
        print(f"❌ Error loading file: {e}")
        return None

    df.columns = [c.strip() for c in df.columns]
    df["공통날짜"] = pd.to_datetime(df["공통날짜"])
    
    # 숫자 변환
    for c in ["kospi_t", "SPX_t-1", "VIX_t-1", "FX_t"]:
        if df[c].dtype == object:
            df[c] = pd.to_numeric(df[c].str.replace(",", ""), errors='coerce')
            
    # ffill로 결측 메우기 (중요)
    df = df.sort_values("공통날짜").set_index("공통날짜").ffill().dropna()
    
    # --- 지표 계산 ---
    df["rK"] = np.log(df["kospi_t"]).diff()
    df["rS"] = np.log(df["SPX_t-1"]).diff()
    df["rFX"] = np.log(df["FX_t"]).diff()
    
    # Beta & Resid
    BETA_W = 60
    df["beta"] = df["rK"].rolling(BETA_W).cov(df["rS"]) / df["rS"].rolling(BETA_W).var()
    df["resid"] = df["rK"] - df["beta"] * df["rS"]
    
    # Z-Score (Shift 1 applied for look-ahead bias removal)
    RES_W = 60
    df["resid_mean"] = df["resid"].rolling(RES_W).mean().shift(1)
    df["resid_std"]  = df["resid"].rolling(RES_W).std().shift(1)
    df["z"] = (df["resid"] - df["resid_mean"]) / df["resid_std"]
    
    # Filters
    W_FILTER = 252
    df["vix_rank"] = df["VIX_t-1"].rolling(W_FILTER).rank(pct=True)
    
    df["fx_mean"] = df["rFX"].rolling(W_FILTER).mean()
    df["fx_std"]  = df["rFX"].rolling(W_FILTER).std()
    df["fx_z"]    = (df["rFX"] - df["fx_mean"]) / df["fx_std"]
    df["fx_shock"] = df["fx_z"].abs().rolling(W_FILTER).rank(pct=True)

    return df.dropna()

# ==========================================
# 3. 시뮬레이션 엔진
# ==========================================
def run_simulation(df_segment, entry, exit_, vix_q, fx_q=0.9):
    # 필터 조건
    allow = (df_segment["vix_rank"] <= vix_q) & (df_segment["fx_shock"] <= fx_q)
    
    pos = np.zeros(len(df_segment))
    z = df_segment["z"].values
    allow_val = allow.values
    
    current_pos = 0
    
    for i in range(1, len(df_segment)):
        # 기본: 포지션 유지
        pos[i] = current_pos
        
        # 1. 필터 컷
        if not allow_val[i]:
            current_pos = 0
            pos[i] = 0
            continue
            
        # 2. 손절매 (Structural Break)
        if current_pos != 0:
            if abs(z[i]) > (entry * STOP_LOSS_MULT):
                current_pos = 0
                pos[i] = 0
                continue
        
        # 3. 진입/청산
        if current_pos == 0:
            if z[i] <= -entry:
                current_pos = 1
            elif z[i] >= entry:
                current_pos = -1
        else:
            if abs(z[i]) <= exit_:
                current_pos = 0
                
        pos[i] = current_pos

    # PnL 계산
    ret = pd.Series(pos).shift(1).fillna(0).values * df_segment["rK"].values
    turnover = np.abs(np.diff(pos, prepend=0))
    ret_net = ret - TC * turnover
    
    return pd.Series(ret_net, index=df_segment.index)

def calc_stats(ret_series):
    if len(ret_series) == 0: return {}
    ann = 252
    mean = ret_series.mean() * ann
    vol = ret_series.std() * np.sqrt(ann)
    sharpe = mean / vol if vol > 0 else 0
    
    cum = (1 + ret_series).cumprod()
    mdd = (cum / cum.cummax() - 1).min()
    total_ret = cum.iloc[-1] - 1
    
    return {
        "Ann.Ret": mean, "Ann.Vol": vol, "Sharpe": sharpe, 
        "MDD": mdd, "Total": total_ret
    }

# ==========================================
# 4. Global Grid Search (전구간 전수조사)
# ==========================================
def run_global_grid(df, out_dir):
    print("\n🔎 Running Global Grid Search (Fixed Params over 25 years)...")
    results = []
    
    total_combinations = len(ENTRY_GRID) * len(EXIT_GRID) * len(VIX_Q_GRID)
    count = 0
    
    for entry, exit_, vix_q in product(ENTRY_GRID, EXIT_GRID, VIX_Q_GRID):
        if exit_ >= entry: continue
        
        ret_series = run_simulation(df, entry, exit_, vix_q)
        stats = calc_stats(ret_series)
        
        stats["ENTRY"] = entry
        stats["EXIT"]  = exit_
        stats["VIX_Q"] = vix_q
        results.append(stats)
        
        count += 1
        if count % 20 == 0:
            print(f"   Progress: {count}/{total_combinations}", end="\r")

    res_df = pd.DataFrame(results)
    res_df = res_df.sort_values("Sharpe", ascending=False)
    
    # 저장
    res_df.to_csv(out_dir / "global_grid_results.csv", index=False)
    print(f"\n✅ Global search finished. Saved to 'global_grid_results.csv'")
    
    # Top 3 출력
    print("\n🏆 Top 3 Fixed Parameter Sets (Global):")
    print(res_df.head(3)[["ENTRY", "EXIT", "VIX_Q", "Sharpe", "MDD", "Total"]].to_string(index=False))

# ==========================================
# 5. Walk-Forward Grid Search
# ==========================================
def run_walk_forward(df, out_dir):
    print("\n🚶 Running Walk-Forward Grid Search...")
    
    start_date = df.index.min()
    end_date   = df.index.max()
    test_start = start_date + pd.DateOffset(years=TRAIN_YEARS)
    
    oos_list = []
    params_log = []
    
    while test_start < end_date:
        train_start = test_start - pd.DateOffset(years=TRAIN_YEARS)
        test_end    = test_start + pd.DateOffset(years=TEST_YEARS)
        
        # Buffer 포함한 Train Data
        train_data = df.loc[train_start - pd.Timedelta(days=400) : test_start]
        test_data  = df.loc[test_start : test_end]
        
        if len(test_data) == 0: break
        
        # [Train]
        best_score = -np.inf
        best_param = None
        
        valid_train_idx = train_data.index >= train_start
        
        for entry, exit_, vix_q in product(ENTRY_GRID, EXIT_GRID, VIX_Q_GRID):
            if exit_ >= entry: continue
            
            res = run_simulation(train_data, entry, exit_, vix_q)
            res_valid = res[valid_train_idx]
            
            if len(res_valid) == 0: continue
            
            # Score: Sharpe + MDD Penalty
            # 최근 3년(Train) 장세에 가장 안정적인 파라미터 선정
            s = calc_stats(res_valid)
            score = s["Sharpe"] - 1.0 * abs(s["MDD"])
            
            if score > best_score:
                best_score = score
                best_param = (entry, exit_, vix_q)
                
        # [Test]
        if best_param:
            be, bx, bv = best_param
            res_test = run_simulation(test_data, be, bx, bv)
            
            oos_list.append(res_test)
            
            # 기록
            test_stats = calc_stats(res_test)
            log_entry = {
                "Test_Start": test_start.date(),
                "Test_End": test_data.index[-1].date() if len(test_data) > 0 else test_end.date(),
                "Best_Entry": be, "Best_Exit": bx, "Best_Vix": bv,
                "OOS_Ret": test_stats.get("Ann.Ret", 0),
                "OOS_MDD": test_stats.get("MDD", 0)
            }
            params_log.append(log_entry)
            print(f"   [{test_start.date()}] Selected {best_param} -> OOS Ret: {log_entry['OOS_Ret']:.1%}")
            
        test_start += pd.DateOffset(years=TEST_YEARS)
        
    # 저장
    if oos_list:
        full_oos = pd.concat(oos_list).sort_index()
        full_oos.name = "Strategy_Net_Return"
        
        # Equity Curve 생성
        equity = (1 + full_oos).cumprod()
        equity.name = "Equity"
        
        output_df = pd.concat([full_oos, equity], axis=1)
        output_df.to_csv(out_dir / "oos_equity_curve.csv")
        
        pd.DataFrame(params_log).to_csv(out_dir / "wf_params_log.csv", index=False)
        print(f"✅ Walk-forward finished. Saved 'oos_equity_curve.csv' & 'wf_params_log.csv'")
        
        # 최종 통계
        final_stats = calc_stats(full_oos)
        print(f"\n📈 Final OOS Performance:")
        print(f"   Total Return: {final_stats['Total']:.2%}")
        print(f"   Ann. Return : {final_stats['Ann.Ret']:.2%}")
        print(f"   MDD         : {final_stats['MDD']:.2%}")

# ==========================================
# Main
# ==========================================
def main():
    df = load_and_prep(PATH)
    if df is None: return
    
    # 결과 저장 폴더 생성
    base_dir = Path("grid_search_longterm")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = base_dir / stamp
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"🚀 Starting Long-term Analysis (Output: {out_dir})")
    
    # 1. Global Grid Search (전체 기간 고정 파라미터 효율 체크)
    run_global_grid(df, out_dir)
    
    # 2. Walk-Forward Analysis (구간별 최적화)
    run_walk_forward(df, out_dir)

if __name__ == "__main__":
    main()