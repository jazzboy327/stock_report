# # src/utils/financial_analysis.py

import yfinance as yf
import pandas_datareader as pdr
from pykrx import stock
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def get_financial_statements(ticker):
    """기업의 재무제표를 가져옵니다."""
    company = yf.Ticker(ticker)
    return company.financials

def get_dividends(ticker):
    """기업의 배당금 정보를 가져옵니다."""
    company = yf.Ticker(ticker)
    return company.dividends

def analyze_risk_metrics(ticker, benchmark_ticker='^KS11', period='5y'):
    """
    주식의 위험 지표를 분석합니다.
    
    Args:
        ticker (str): 주식 티커 심볼
        benchmark_ticker (str): 벤치마크 지수 티커 심볼 (기본값: KOSPI)
        period (str): 분석 기간 (기본값: 5년)
        
    Returns:
        dict: 다양한 위험 지표를 포함한 딕셔너리
    """
    try:
        # 한국 주식 심볼 처리 (yfinance 형식에 맞게 변환)
        modified_ticker = ticker
        # 티커 형식이 이미 .KS 또는 .KQ로 끝나면 그대로 사용
        if '.KS' in ticker or '.KQ' in ticker:
            pass
        # 숫자로만 구성된 티커는 한국 주식으로 처리 - KOSPI 종목
        elif ticker.isdigit() and len(ticker) == 6:
            modified_ticker = f"{ticker}.KS"
        
        print(f"분석 대상 티커: {modified_ticker}, 벤치마크: {benchmark_ticker}")
        
        # 주가 데이터 가져오기 (세부 로그 추가)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365*5)  # 5년치 데이터
        
        print(f"데이터 조회 기간: {start_date.strftime('%Y-%m-%d')} ~ {end_date.strftime('%Y-%m-%d')}")
        stock_data = yf.download(modified_ticker, start=start_date, end=end_date, progress=False)
        benchmark_data = yf.download(benchmark_ticker, start=start_date, end=end_date, progress=False)
        
        # 데이터 구조 확인 로그
        print(f"주식 데이터 컬럼: {stock_data.columns.tolist()}")
        print(f"벤치마크 데이터 컬럼: {benchmark_data.columns.tolist()}")
        
        # 데이터 유효성 검사
        if stock_data.empty:
            raise ValueError(f"주식 데이터를 가져올 수 없습니다: {modified_ticker}")
        
        if benchmark_data.empty:
            raise ValueError(f"벤치마크 데이터를 가져올 수 없습니다: {benchmark_ticker}")
        
        print(f"수집된 주식 데이터: {len(stock_data)}행, 벤치마크 데이터: {len(benchmark_data)}행")
        
        # 종가 데이터 추출 - 'Adj Close'가 없으면 'Close' 사용
        if 'Adj Close' in stock_data.columns:
            stock_prices = stock_data['Adj Close']
            print("주식 데이터: Adj Close 컬럼 사용")
        else:
            stock_prices = stock_data['Close']
            print("주식 데이터: Close 컬럼 사용")
            
        if 'Adj Close' in benchmark_data.columns:
            benchmark_prices = benchmark_data['Adj Close']
            print("벤치마크 데이터: Adj Close 컬럼 사용")
        else:
            benchmark_prices = benchmark_data['Close']
            print("벤치마크 데이터: Close 컬럼 사용")
        
        # Series 타입인지 확인
        print(f"Stock prices type: {type(stock_prices)}")
        print(f"Benchmark prices type: {type(benchmark_prices)}")
        
        # DataFrame인 경우 Series로 변환
        if isinstance(stock_prices, pd.DataFrame):
            stock_prices = stock_prices.iloc[:, 0]
            print("주식 가격을 DataFrame에서 Series로 변환했습니다")
            
        if isinstance(benchmark_prices, pd.DataFrame):
            benchmark_prices = benchmark_prices.iloc[:, 0]
            print("벤치마크 가격을 DataFrame에서 Series로 변환했습니다")
        
        # 일간 수익률 계산
        stock_returns = stock_prices.pct_change().dropna()
        benchmark_returns = benchmark_prices.pct_change().dropna()
        
        # 데이터 수가 너무 적으면 오류 발생
        if len(stock_returns) < 30:
            raise ValueError(f"충분한 주가 데이터가 없습니다. 최소 30일 이상의 데이터가 필요합니다.")
        
        # 데이터 정렬
        aligned_data = pd.concat([stock_returns, benchmark_returns], axis=1).dropna()
        aligned_data.columns = ['stock', 'benchmark']
        
        # 스칼라 값 기반 계산 (Series.iloc[0] 사용)
        stock_returns_mean = stock_returns.mean()
        if isinstance(stock_returns_mean, pd.Series):
            stock_returns_mean = stock_returns_mean.iloc[0]
        else:
            stock_returns_mean = float(stock_returns_mean)
            
        benchmark_returns_mean = benchmark_returns.mean()
        if isinstance(benchmark_returns_mean, pd.Series):
            benchmark_returns_mean = benchmark_returns_mean.iloc[0]
        else:
            benchmark_returns_mean = float(benchmark_returns_mean)
            
        stock_returns_std = stock_returns.std()
        if isinstance(stock_returns_std, pd.Series):
            stock_returns_std = stock_returns_std.iloc[0]
        else:
            stock_returns_std = float(stock_returns_std)
        
        # 1. 베타 계산
        cov_matrix = aligned_data.cov()
        covariance = cov_matrix.iloc[0, 1]
        benchmark_variance = aligned_data['benchmark'].var()
        
        # Series인 경우 값 추출
        if isinstance(covariance, pd.Series):
            covariance = covariance.iloc[0]
        if isinstance(benchmark_variance, pd.Series):
            benchmark_variance = benchmark_variance.iloc[0]
            
        beta = covariance / benchmark_variance if benchmark_variance > 0 else 1.0
        
        # 2. 변동성 (연간 표준편차)
        annual_volatility = stock_returns_std * np.sqrt(252)
        
        # 3. 최대 낙폭 (MDD)
        cumulative_returns = (1 + stock_returns).cumprod()
        mdd_values = (cumulative_returns / cumulative_returns.cummax() - 1)
        max_dd_value = mdd_values.min()
        
        # Series인 경우 값 추출
        if isinstance(max_dd_value, pd.Series):
            max_dd = max_dd_value.iloc[0]
        else:
            max_dd = float(max_dd_value)
        
        # 4. 샤프 비율
        risk_free_rate = 0.03  # 무위험 수익률 가정 (3%)
        annual_return = (1 + stock_returns_mean) ** 252 - 1
        sharpe_ratio = (annual_return - risk_free_rate) / annual_volatility if annual_volatility > 0 else 0
        
        # 5. 하방 위험 (Downside Risk)
        negative_returns = stock_returns[stock_returns < 0]
        
        if len(negative_returns) > 0:
            downside_risk_std = negative_returns.std()
            if isinstance(downside_risk_std, pd.Series):
                downside_risk_std = downside_risk_std.iloc[0]
            downside_risk = downside_risk_std * np.sqrt(252)
        else:
            downside_risk = 0
        
        # 6. 월간 수익률 계산 (ME로 변경)
        monthly_returns = stock_prices.resample('ME').last().pct_change().dropna()
        positive_months = (monthly_returns > 0).sum()
        total_months = len(monthly_returns)
        
        # Series인 경우 값 추출
        if isinstance(positive_months, pd.Series):
            positive_months = positive_months.iloc[0]
            
        positive_month_ratio = positive_months / total_months if total_months > 0 else 0
        
        # 7. VaR (Value at Risk) - 95% 신뢰수준
        if len(stock_returns) > 20:
            var_95 = np.percentile(stock_returns, 5)
            # Series인 경우 값 추출
            if isinstance(var_95, pd.Series):
                var_95 = var_95.iloc[0]
        else:
            var_95 = -annual_volatility / np.sqrt(252)
            
        daily_var_95 = var_95 * stock_prices.iloc[-1]
        if isinstance(daily_var_95, pd.Series):
            daily_var_95 = daily_var_95.iloc[0]
        
        # 8. 상관관계
        correlation_matrix = aligned_data.corr()
        correlation = correlation_matrix.iloc[0, 1]
        
        # 9. 알파 (CAPM 모델 기반)
        risk_free_daily = risk_free_rate / 252
        alpha = stock_returns_mean - (risk_free_daily + beta * (benchmark_returns_mean - risk_free_daily))
        annual_alpha = (1 + alpha) ** 252 - 1
        
        # 10. 승률 (Winning Ratio)
        winning_days = (stock_returns > 0).sum()
        total_days = len(stock_returns)
        
        # Series인 경우 값 추출
        if isinstance(winning_days, pd.Series):
            winning_days = winning_days.iloc[0]
            
        winning_ratio = winning_days / total_days if total_days > 0 else 0
        
        # 위험 지표 결과 반환
        risk_metrics = {
            'beta': round(beta, 2),
            'annual_volatility': round(annual_volatility * 100, 2),  # 백분율로 변환
            'max_drawdown': round(max_dd * 100, 2),  # 백분율로 변환
            'sharpe_ratio': round(sharpe_ratio, 2),
            'downside_risk': round(downside_risk * 100, 2),  # 백분율로 변환
            'positive_month_ratio': round(positive_month_ratio * 100, 2),  # 백분율로 변환
            'var_95': round(var_95 * 100, 2),  # 백분율로 변환
            'daily_var_95': round(daily_var_95, 2),
            'correlation': round(correlation, 2),
            'annual_alpha': round(annual_alpha * 100, 2),  # 백분율로 변환
            'winning_ratio': round(winning_ratio * 100, 2)  # 백분율로 변환
        }
        
        print(f"위험 지표 분석 완료: {ticker}")
        return risk_metrics
    
    except Exception as e:
        import traceback
        error_msg = str(e)
        error_trace = traceback.format_exc()
        print(f"위험 지표 분석 중 오류 발생: {error_msg}")
        print(f"상세 오류: {error_trace}")
        
        # 오류 메시지에 따라 다른 기본값 반환
        if "주식 데이터를 가져올 수 없습니다" in error_msg:
            return {
                'error': True,
                'error_message': f"티커 심볼({ticker})에 대한 데이터를 찾을 수 없습니다. 심볼이 올바른지 확인하세요.",
                'beta': 1.0,
                'annual_volatility': 15.0,  # 시장 평균 가정
                'max_drawdown': 20.0,
                'sharpe_ratio': 0.5,
                'downside_risk': 10.0,
                'positive_month_ratio': 55.0,
                'var_95': -1.5,
                'daily_var_95': 1000.0,
                'correlation': 0.7,
                'annual_alpha': 0.0,
                'winning_ratio': 52.0
            }
        else:
            # 일반적인 오류 시 기본값
            return {
                'error': True,
                'error_message': f"위험 지표 분석 중 오류가 발생했습니다: {error_msg}",
                'beta': 1.0,
                'annual_volatility': 15.0,
                'max_drawdown': 20.0,
                'sharpe_ratio': 0.5,
                'downside_risk': 10.0,
                'positive_month_ratio': 55.0,
                'var_95': -1.5,
                'daily_var_95': 1000.0,
                'correlation': 0.7,
                'annual_alpha': 0.0,
                'winning_ratio': 52.0
            }



def analyze_growth_rates(ticker):
    """
    기업의 재무제표를 분석하여 성장률을 계산합니다.
    
    Args:
        ticker (str): 주식 티커 심볼
        
    Returns:
        dict: 매출, 영업이익, 순이익 등의 성장률 데이터를 포함한 딕셔너리
    """
    try:
        import yfinance as yf
        import pandas as pd
        import numpy as np
        from datetime import datetime
        import logging
        
        logger = logging.getLogger('StockAnalysisApp.FinancialAnalysis')
        
        # 회사 정보 가져오기
        company = yf.Ticker(ticker)
        
        # 재무제표 가져오기 (연간)
        income_stmt = company.financials  # 손익계산서
        balance_sheet = company.balance_sheet  # 대차대조표
        cash_flow = company.cashflow  # 현금흐름표
        
        # 분기별 데이터
        quarterly_income = company.quarterly_financials  # 분기별 손익계산서
        
        # 결과 저장 딕셔너리
        growth_data = {
            'annual': {
                'years': [],
                'revenue': [],
                'revenue_growth': [],
                'operating_income': [],
                'operating_income_growth': [],
                'net_income': [],
                'net_income_growth': [],
                'eps': [],
                'eps_growth': []
            },
            'quarterly': {
                'quarters': [],
                'revenue': [],
                'revenue_growth': [],
                'net_income': [],
                'net_income_growth': []
            }
        }
        
        # 연간 데이터 처리
        if not income_stmt.empty:
            # 연도 정보 추출 (column은 날짜)
            annual_years = [date.strftime('%Y') for date in income_stmt.columns]
            growth_data['annual']['years'] = annual_years
            
            # 총매출 (Total Revenue)
            revenue_row = None
            for possible_name in ['Total Revenue', 'Revenue', 'Gross Profit']:
                if possible_name in income_stmt.index:
                    revenue_row = possible_name
                    break
            
            if revenue_row:
                annual_revenue = income_stmt.loc[revenue_row].values
                growth_data['annual']['revenue'] = annual_revenue.tolist()
                
                # 성장률 계산
                revenue_growth = []
                for i in range(1, len(annual_revenue)):
                    if annual_revenue[i-1] != 0:  # 분모가 0이 아닌 경우에만
                        growth = ((annual_revenue[i] - annual_revenue[i-1]) / abs(annual_revenue[i-1])) * 100
                    else:
                        growth = 0
                    revenue_growth.append(growth)
                
                # 첫 해는 성장률 계산 불가능하므로 None 또는 0 추가
                revenue_growth.insert(0, None)
                growth_data['annual']['revenue_growth'] = revenue_growth
            
            # 영업이익 (Operating Income)
            operating_income_row = None
            for possible_name in ['Operating Income', 'EBIT', 'Operating Profit']:
                if possible_name in income_stmt.index:
                    operating_income_row = possible_name
                    break
            
            if operating_income_row:
                annual_operating_income = income_stmt.loc[operating_income_row].values
                growth_data['annual']['operating_income'] = annual_operating_income.tolist()
                
                # 성장률 계산
                operating_income_growth = []
                for i in range(1, len(annual_operating_income)):
                    if annual_operating_income[i-1] != 0:  # 분모가 0이 아닌 경우에만
                        growth = ((annual_operating_income[i] - annual_operating_income[i-1]) / abs(annual_operating_income[i-1])) * 100
                    else:
                        growth = 0
                    operating_income_growth.append(growth)
                
                operating_income_growth.insert(0, None)
                growth_data['annual']['operating_income_growth'] = operating_income_growth
            
            # 순이익 (Net Income)
            net_income_row = None
            for possible_name in ['Net Income', 'Net Income Common Stockholders']:
                if possible_name in income_stmt.index:
                    net_income_row = possible_name
                    break
            
            if net_income_row:
                annual_net_income = income_stmt.loc[net_income_row].values
                growth_data['annual']['net_income'] = annual_net_income.tolist()
                
                # 성장률 계산
                net_income_growth = []
                for i in range(1, len(annual_net_income)):
                    if annual_net_income[i-1] != 0:  # 분모가 0이 아닌 경우에만
                        growth = ((annual_net_income[i] - annual_net_income[i-1]) / abs(annual_net_income[i-1])) * 100
                    else:
                        growth = 0
                    net_income_growth.append(growth)
                
                net_income_growth.insert(0, None)
                growth_data['annual']['net_income_growth'] = net_income_growth
            
            # EPS (Basic EPS)
            eps_row = None
            for possible_name in ['Basic EPS', 'EPS', 'Diluted EPS']:
                if possible_name in income_stmt.index:
                    eps_row = possible_name
                    break
            
            if eps_row:
                annual_eps = income_stmt.loc[eps_row].values
                growth_data['annual']['eps'] = annual_eps.tolist()
                
                # 성장률 계산
                eps_growth = []
                for i in range(1, len(annual_eps)):
                    if annual_eps[i-1] != 0:  # 분모가 0이 아닌 경우에만
                        growth = ((annual_eps[i] - annual_eps[i-1]) / abs(annual_eps[i-1])) * 100
                    else:
                        growth = 0
                    eps_growth.append(growth)
                
                eps_growth.insert(0, None)
                growth_data['annual']['eps_growth'] = eps_growth
        
        # 분기별 데이터 처리
        if not quarterly_income.empty:
            # 분기 정보 추출
            quarterly_dates = [f"{date.year}-Q{date.quarter}" for date in quarterly_income.columns]


            # quarterly_dates = [date.strftime('%Y-Q%q').replace('Q1', 'Q1').replace('Q2', 'Q2').replace('Q3', 'Q3').replace('Q4', 'Q4') 
            #                 for date in quarterly_income.columns]
            growth_data['quarterly']['quarters'] = quarterly_dates
            
            # 분기별 매출
            if revenue_row and revenue_row in quarterly_income.index:
                quarterly_revenue = quarterly_income.loc[revenue_row].values
                growth_data['quarterly']['revenue'] = quarterly_revenue.tolist()
                
                # 전년 동기 대비 성장률 계산 (4분기 전과 비교)
                quarterly_revenue_growth = []
                for i in range(min(4, len(quarterly_revenue)), len(quarterly_revenue)):
                    if quarterly_revenue[i-4] != 0:
                        growth = ((quarterly_revenue[i] - quarterly_revenue[i-4]) / abs(quarterly_revenue[i-4])) * 100
                    else:
                        growth = 0
                    quarterly_revenue_growth.append(growth)
                
                # 첫 4분기는 성장률 계산 불가능
                for _ in range(min(4, len(quarterly_revenue))):
                    quarterly_revenue_growth.insert(0, None)
                
                growth_data['quarterly']['revenue_growth'] = quarterly_revenue_growth
            
            # 분기별 순이익
            if net_income_row and net_income_row in quarterly_income.index:
                quarterly_net_income = quarterly_income.loc[net_income_row].values
                growth_data['quarterly']['net_income'] = quarterly_net_income.tolist()
                
                # 전년 동기 대비 성장률 계산
                quarterly_net_income_growth = []
                for i in range(min(4, len(quarterly_net_income)), len(quarterly_net_income)):
                    if quarterly_net_income[i-4] != 0:
                        growth = ((quarterly_net_income[i] - quarterly_net_income[i-4]) / abs(quarterly_net_income[i-4])) * 100
                    else:
                        growth = 0
                    quarterly_net_income_growth.append(growth)
                
                # 첫 4분기는 성장률 계산 불가능
                for _ in range(min(4, len(quarterly_net_income))):
                    quarterly_net_income_growth.insert(0, None)
                
                growth_data['quarterly']['net_income_growth'] = quarterly_net_income_growth
        
        return growth_data
        
    except Exception as e:
        import traceback
        error_msg = str(e)
        error_trace = traceback.format_exc()
        logger.error(f"성장성 분석 중 오류 발생: {error_msg}")
        logger.error(f"상세 오류: {error_trace}")
        
        # 오류 발생 시 기본 데이터 반환
        return {
            'annual': {
                'years': ["2021", "2022", "2023"],
                'revenue': [100, 110, 120],
                'revenue_growth': [None, 10.0, 9.09],
                'operating_income': [20, 22, 24],
                'operating_income_growth': [None, 10.0, 9.09],
                'net_income': [15, 16, 18],
                'net_income_growth': [None, 6.67, 12.5],
                'eps': [1.5, 1.6, 1.8],
                'eps_growth': [None, 6.67, 12.5]
            },
            'quarterly': {
                'quarters': ["2022-Q4", "2023-Q1", "2023-Q2", "2023-Q3", "2023-Q4"],
                'revenue': [28, 29, 30, 31, 32],
                'revenue_growth': [None, None, None, None, 14.29],
                'net_income': [4.2, 4.3, 4.4, 4.5, 4.8],
                'net_income_growth': [None, None, None, None, 14.29]
            },
            'error': True,
            'error_message': f"성장성 분석 중 오류가 발생했습니다: {error_msg}"
        }