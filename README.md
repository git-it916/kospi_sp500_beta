# KOSPI-SP500 베타 잔차 되돌림 전략

이 프로젝트는 KOSPI가 S&P500 대비 과도하게 움직였을 때 되돌림이 발생한다는 가정으로 거래하는 전략을 다룹니다.  
`base.py`(단일 백테스트), `report.py`(리포트 생성), `grid_search.py`(워크포워드 그리드 서치) 기준으로 작성했습니다.
(아직 초안이며 강건한 전략을 위해 grid search와 긴기간으로 테스트를 해볼 예정입니다)

## 파일 구성
- `base.py`: 단일 전략 백테스트 (콘솔 요약 출력)
- `report.py`: 그래프/CSV/요약 리포트 생성 (outputs/날짜_시간)
- `grid_search.py`: 워크포워드 그리드 서치 + OOS 백테스트 (grid_search/날짜_시간)
- `README.md`: 설명 문서

## 데이터
- 기본 경로: `C:\Users\10845\OneDrive - 이지스자산운용\문서\kospi_sp500_filtered.xlsx` [전처리 완료]
- 필수 컬럼
  - `공통날짜`: 날짜 (일별)
  - `kospi_t`: KOSPI 종가
  - `SPX_t-1`: S&P500 전일 종가 (시차 보정용)
  - `VIX_t-1`: VIX 전일 값
  - `FX_t`: 원/달러 환율

## 전략 로직 (쉽게 설명)
1. **동행성 측정**: 최근 60일을 기준으로 KOSPI가 S&P500에 얼마나 민감한지(베타)를 계산합니다.
2. **잔차 계산**: `예상 KOSPI 움직임 = 베타 × S&P500 변화`, 실제 움직임과의 차이를 잔차로 정의합니다.
3. **과열/과매도 판단**: 잔차를 Z-Score로 표준화해 과열/과매도를 판단합니다.
4. **리스크 필터**: VIX가 높거나 환율이 급변하면 거래를 쉬어 리스크를 줄입니다.

## 매매 규칙 (기본 전략)
- 진입
  - Z ≤ -1.0: KOSPI 매수 (롱)
  - Z ≥ +1.0: KOSPI 매도 (숏)
- 청산
  - |Z| ≤ 0.2: 포지션 종료
- 거래 제한
  - VIX가 과도하게 높거나, 환율 변동이 급격할 때는 거래하지 않음
- 체결 가정
  - 신호는 오늘 계산, 실제 매매는 다음 거래일에 실행

## 기본 파라미터
- 베타/잔차 롤링: 60일
- 리스크 필터 롤링: 252일
- VIX 기준: 상위 80% 분위
- FX 기준: 변동성 Z-Score 절대값 상위 90% 분위
- 거래 비용: 2bp

## 실행 방법
### 1) 단일 백테스트
```bash
python base.py
```
- 콘솔에 요약 지표를 출력합니다.
- 데이터 경로를 바꾸려면 `base.py` 상단 `path`를 수정합니다.

### 2) 리포트 생성 (출력 정리)
```bash
python report.py
```
- 실행할 때마다 `outputs/YYYYMMDD_HHMMSS/` 폴더가 생성됩니다.
- 다른 경로를 쓰려면:
```bash
python report.py --path "엑셀경로" --out-dir outputs
```

### 3) 워크포워드 그리드 서치 (OOS)
```bash
python grid_search.py
```
- 실행할 때마다 `grid_search/YYYYMMDD_HHMMSS/` 폴더가 생성됩니다.
- 데이터 경로를 바꾸려면 `grid_search.py` 상단 `PATH`를 수정합니다.

## 출력 설명
### report.py 결과물
- `outputs/YYYYMMDD_HHMMSS/summary.txt`: 기간, 연환산 수익률/변동성, 샤프, MDD 등 요약
- `outputs/YYYYMMDD_HHMMSS/equity_curve.png`: 누적 수익 곡선
- `outputs/YYYYMMDD_HHMMSS/drawdown.png`: 최대낙폭 추이
- `outputs/YYYYMMDD_HHMMSS/annual_returns.csv`: 연도별 수익률
- `outputs/YYYYMMDD_HHMMSS/annual_returns.png`: 연도별 수익률 그래프
- `outputs/YYYYMMDD_HHMMSS/quarterly_returns.csv`: 분기별 수익률
- `outputs/YYYYMMDD_HHMMSS/quarterly_returns_heatmap.png`: 분기별 수익률 히트맵

### grid_search.py 결과물
- `grid_search/YYYYMMDD_HHMMSS/oos_equity.csv`
  - 컬럼: `공통날짜`, `strategy_ret_net`, `oos_equity`
  - 워크포워드 구간별 OOS 누적 수익 곡선
- `grid_search/YYYYMMDD_HHMMSS/wf_params_by_segment.csv`
  - 컬럼: `train_start`, `test_start`, `ENTRY`, `EXIT`, `VIX_Q`, `FX_Q`, `sharpe`, `ann_ret`, `ann_vol`, `mdd`
  - 각 구간에서 선택된 최적 파라미터와 성과

## 워크포워드 그리드 서치 방식
- 학습 2년, 테스트 6개월, 6개월씩 전진
- 파라미터 탐색 범위
  - ENTRY: 0.8, 1.0, 1.2, 1.5
  - EXIT: 0.1, 0.2, 0.3, 0.4
  - VIX/FX 분위: 0.75, 0.80, 0.85, 0.90
- 평가 함수: `Sharpe - 0.5 × |MDD|`
- 롤링 필터 계산을 위해 테스트 구간 앞 400일 컨텍스트 포함

## 예시 결과 (outputs/20251229_155825 기준)
| 항목 | 값 |
| --- | ---: |
| 기간 | 2020-01-03 ~ 2025-12-26 |
| 거래일 수 | 1,421 |
| 연환산 수익률 | 2.15% |
| 연환산 변동성 | 9.40% |
| 샤프 | 0.23 |
| 최대낙폭(MDD) | -23.29% |
| 누적 수익률 | 10.11% |
| 시장 노출일 비중 | 23.72% |
| 일간 양(+) 수익 비중 | 11.75% |

### 연도별 수익률 (예시)
| Year | Return |
| --- | ---: |
| 2020 | 0.00% |
| 2021 | 0.00% |
| 2022 | 1.47% |
| 2023 | 7.52% |
| 2024 | -7.03% |
| 2025 | 8.55% |

### 분기별 수익률 (예시)
| Year | Q1 | Q2 | Q3 | Q4 |
| --- | ---: | ---: | ---: | ---: |
| 2020 | 0.00% | 0.00% | 0.00% | 0.00% |
| 2021 | 0.00% | 0.00% | 0.00% | 0.00% |
| 2022 | -0.04% | 4.30% | 5.36% | -7.63% |
| 2023 | 3.83% | 5.37% | 0.20% | -1.93% |
| 2024 | 0.79% | -2.11% | -5.32% | -0.47% |
| 2025 | -5.74% | -2.78% | -4.57% | 24.13% |

## 해석 가이드 (펀드매니저 관점)
- 성과가 특정 분기에 집중되는 경향이 있어, 전략 비중 조절이 중요합니다.
- 초기 구간은 롤링 윈도우 확보로 거래가 거의 없을 수 있습니다.
- VIX/환율 필터는 리스크를 낮추지만, 수익 기회도 동시에 줄입니다.
- 최대낙폭이 존재하므로, 단독 운용보다는 분산 포트폴리오 내 활용이 적합합니다.

## 최신 업데이트 (2025-12-31)

### Grid Search 최적화 완료
- **3-Stage 최적화**: Coarse → Fine → Robustness
- **파라미터 개선**: Entry 2.0→2.15, VIX 0.85→0.94, FX 0.90→0.96
- **성과 향상**: Sharpe 0.41→0.75 (+83%), MDD -26.56%→-15.72% (+41%)

### 주요 문서
- **전략 명세서**: `STRATEGY_SPECIFICATION.md` (상세 기술 문서)
- **빠른 재현 프롬프트**: `QUICK_PROMPT.txt` (원라이너 프롬프트)
- **최적화 결과**: `OPTIMIZATION_RESULTS.md` (Grid Search 분석)
- **Grid Search 가이드**: `GRID_SEARCH_GUIDE.md` (사용법)

## 참고
- 리포트 요약: `outputs/YYYYMMDD_HHMMSS/summary.txt`
- 연도별 수익률: `outputs/YYYYMMDD_HHMMSS/annual_returns.csv`
- 분기별 수익률: `outputs/YYYYMMDD_HHMMSS/quarterly_returns.csv`
- 워크포워드 결과: `grid_search/YYYYMMDD_HHMMSS/oos_equity.csv`
- 워크포워드 파라미터: `grid_search/YYYYMMDD_HHMMSS/wf_params_by_segment.csv`
- **Grid Search 개선 결과**: `grid_search_improved/YYYYMMDD_HHMMSS/`
