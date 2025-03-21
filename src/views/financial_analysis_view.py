# src/views/financial_analysis_view.py

import logging
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import altair as alt
from datetime import datetime, timedelta
import yfinance as yf
import traceback
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import io
import base64
from tensorflow import keras
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Sequential

logger = logging.getLogger('StockAnalysisApp.FinancialAnalysisView')


class FinancialAnalysisView:
    """재무 분석 뷰를 담당하는 클래스"""

    def __init__(self):
        """뷰 초기화"""
        pass

    def display(self, company_info):
        """재무 분석 탭 표시"""
        st.header("위험 지표 및 재무 분석")

        ticker = company_info['symbol']
        start, end = self._get_analysis_period()

        # 데이터 수집 정보 표시
        st.write(f"수집 기간: {start.strftime('%Y-%m-%d')} ~ {end.strftime('%Y-%m-%d')}")

        # 탭 생성 - 성장성 분석 탭 추가
        fin_tabs = st.tabs(["위험 지표 분석", "성장성 분석", "재무 지표"])

        # 위험 지표 분석 탭
        with fin_tabs[0]:
            self._display_risk_metrics_tab(ticker)

        # 성장성 분석 탭
        with fin_tabs[1]:
            self._display_growth_analysis_tab(ticker)

        # 재무제표 탭
        with fin_tabs[2]:
            self._display_financial_statements_tab(ticker)

    def _get_analysis_period(self):
        """5년치 분석 기간을 반환"""
        end_date = datetime.today()
        start_date = end_date - timedelta(days=5 * 365)
        return start_date, end_date

    def _display_risk_metrics_tab(self, ticker):
        """위험 지표 분석 탭 표시"""
        st.subheader("위험 지표 분석")

        # 로딩 상태 표시
        with st.spinner("위험 지표 분석 중..."):
            # 한국 시장이면 KOSPI를 벤치마크로, 그렇지 않으면 S&P 500을 벤치마크로 사용
            if '.KS' in ticker or '.KQ' in ticker:
                benchmark = '^KS11'  # KOSPI
                benchmark_name = 'KOSPI'
            else:
                benchmark = '^GSPC'  # S&P 500
                benchmark_name = 'S&P 500'

            # 티커 수정 (한국 주식 고려)
            if ticker.isdigit() and len(ticker) == 6:
                # 순수 숫자 6자리면 한국 주식으로 가정하고 .KS 추가
                display_ticker = ticker
                if '.KS' not in ticker and '.KQ' not in ticker:
                    analysis_ticker = f"{ticker}.KS"
                else:
                    analysis_ticker = ticker
            else:
                display_ticker = ticker
                analysis_ticker = ticker

            # 위험 지표 분석 실행
            from src.utils.financial_analysis import analyze_risk_metrics
            risk_metrics = analyze_risk_metrics(analysis_ticker, benchmark, period='5y')

            # 종합리포트에 데이터 등록 (추가된 부분)
            if 'comprehensive_report_view' in globals():
                try:
                    from src.views.comprehensive_report_view import ComprehensiveReportView
                    comprehensive_view = ComprehensiveReportView()
                    comprehensive_view.register_analysis_result('financial_analysis', {'risk_metrics': risk_metrics})
                except Exception as e:
                    logger.warning(f"종합리포트에 위험지표 데이터 등록 실패: {e}")
            else:
                # 세션 상태에 저장 (기존 코드 유지)
                st.session_state.risk_metrics = risk_metrics

            # 오류 확인 및 처리
            if risk_metrics.get('error', False):
                st.error(risk_metrics['error_message'])
                st.warning("기본 시장 통계를 기반으로 한 예상 위험 지표를 표시합니다.")
                st.info(f"정확한 분석을 위해 주식 티커({display_ticker})가 올바른지 확인하세요.")

            # 지표 설명
            self._display_risk_metrics_guide()

            # 메트릭 표시
            self._display_risk_metrics_values(risk_metrics)

            # 위험 평가
            self._display_risk_assessment(risk_metrics, benchmark_name)

            # 투자 성향별 적합도
            self._display_investor_suitability(risk_metrics)

            # 투자 조언 섹션 추가
            if not risk_metrics.get('error', False):
                self._display_investment_advice(risk_metrics)

    def _display_risk_metrics_guide(self):
        """위험 지표 해석 가이드 표시"""
        with st.expander("위험 지표 해석 가이드", expanded=False):
            st.markdown("""
            - **베타(β)**: 시장 대비 주가 변동성을 나타냅니다. 1보다 크면 시장보다 변동성이 큽니다.
            - **연간 변동성**: 주가의 연간 표준편차로, 값이 클수록 변동성이 큽니다.
            - **최대 낙폭(MDD)**: 고점 대비 최대 하락 폭을 나타냅니다. 작을수록 안정적입니다.
            - **샤프 비율**: 위험 대비 수익률을 나타냅니다. 높을수록 효율적인 투자입니다.
            - **알파(α)**: 시장 대비 초과 수익률을 나타냅니다. 양수면 시장보다 좋은 성과입니다.
            - **하방 위험**: 하락장에서의 변동성을 나타냅니다. 작을수록 안정적입니다.
            - **VaR (95%)**: 95% 신뢰수준에서의 최대 예상 손실률입니다.
            - **승률**: 주가가 상승한 날의 비율을 나타냅니다.
            """)

    def _display_risk_metrics_values(self, risk_metrics):
        """위험 지표 값 표시"""
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("베타(β)", f"{risk_metrics['beta']}")
            st.metric("최대 낙폭(MDD)", f"{risk_metrics['max_drawdown']}%")
            st.metric("상관계수", f"{risk_metrics['correlation']}")

        with col2:
            st.metric("연간 변동성", f"{risk_metrics['annual_volatility']}%")
            st.metric("샤프 비율", f"{risk_metrics['sharpe_ratio']}")
            st.metric("하방 위험", f"{risk_metrics['downside_risk']}%")

        with col3:
            st.metric("알파(α)", f"{risk_metrics['annual_alpha']}%")
            st.metric("VaR (95%)", f"{risk_metrics['var_95']}%")
            st.metric("승률", f"{risk_metrics['winning_ratio']}%")

    def _display_risk_assessment(self, risk_metrics, benchmark_name):
        """위험 평가 표시"""
        st.subheader("종합 위험 평가")

        # 베타 기반 위험도
        if risk_metrics['beta'] < 0.8:
            beta_risk = "낮음 (방어적)"
            beta_color = "green"
        elif risk_metrics['beta'] < 1.2:
            beta_risk = "중간 (시장과 유사)"
            beta_color = "orange"
        else:
            beta_risk = "높음 (공격적)"
            beta_color = "red"

        # 변동성 기반 위험도
        kospi_vol = 15.0  # KOSPI 평균 변동성 가정
        if risk_metrics['annual_volatility'] < kospi_vol * 0.8:
            vol_risk = "낮음"
            vol_color = "green"
        elif risk_metrics['annual_volatility'] < kospi_vol * 1.2:
            vol_risk = "중간"
            vol_color = "orange"
        else:
            vol_risk = "높음"
            vol_color = "red"

        # 하방 위험 기반
        if risk_metrics['downside_risk'] < 10:
            down_risk = "낮음"
            down_color = "green"
        elif risk_metrics['downside_risk'] < 20:
            down_risk = "중간"
            down_color = "orange"
        else:
            down_risk = "높음"
            down_color = "red"

        # 최대 낙폭 기반
        if risk_metrics['max_drawdown'] < 20:
            mdd_risk = "낮음"
            mdd_color = "green"
        elif risk_metrics['max_drawdown'] < 40:
            mdd_risk = "중간"
            mdd_color = "orange"
        else:
            mdd_risk = "높음"
            mdd_color = "red"

        # 위험 평가 표시
        st.markdown(f"""
        | 지표 | 값 | 위험도 |
        | --- | --- | --- |
        | 베타 | {risk_metrics['beta']} | <span style="color:{beta_color}">{beta_risk}</span> |
        | 연간 변동성 | {risk_metrics['annual_volatility']}% | <span style="color:{vol_color}">{vol_risk}</span> |
        | 하방 위험 | {risk_metrics['downside_risk']}% | <span style="color:{down_color}">{down_risk}</span> |
        | 최대 낙폭 | {risk_metrics['max_drawdown']}% | <span style="color:{mdd_color}">{mdd_risk}</span> |
        """, unsafe_allow_html=True)

        # 시장과 비교 벤치마크
        st.subheader(f"{benchmark_name} 대비 성과")

        if risk_metrics.get('error', False):
            st.info(f"정확한 비교를 위해서는 주식 코드가 올바르게 설정되어야 합니다.")
        elif risk_metrics['annual_alpha'] > 0:
            st.success(f"이 주식은 지난 5년간 {benchmark_name} 대비 연간 {risk_metrics['annual_alpha']}%의 초과 수익을 냈습니다.")
        else:
            st.error(f"이 주식은 지난 5년간 {benchmark_name} 대비 연간 {abs(risk_metrics['annual_alpha'])}%의 수익이 부족했습니다.")

        if not risk_metrics.get('error', False):
            if risk_metrics['beta'] > 1:
                st.warning(
                    f"베타가 {risk_metrics['beta']}로, 시장보다 {round((risk_metrics['beta'] - 1) * 100, 1)}% 더 변동성이 큽니다.")
            else:
                st.info(f"베타가 {risk_metrics['beta']}로, 시장보다 {round((1 - risk_metrics['beta']) * 100, 1)}% 더 안정적입니다.")

    def _display_investor_suitability(self, risk_metrics):
        """투자자 성향별 적합도 표시"""
        st.subheader("투자 성향별 적합도")

        # 위험 지표들의 가중 평균으로 위험 점수 계산 (0-100 사이)
        risk_score = (
                25 * min(risk_metrics['beta'] / 2, 1) +
                25 * min(risk_metrics['annual_volatility'] / 40, 1) +
                25 * min(risk_metrics['downside_risk'] / 30, 1) +
                25 * min(risk_metrics['max_drawdown'] / 60, 1)
        )

        # 투자 성향별 적합도 표시
        col1, col2, col3 = st.columns(3)

        with col1:
            conservative_score = max(0, 100 - risk_score)
            st.progress(conservative_score / 100)
            st.write(f"안정추구형: {round(conservative_score)}%")

        with col2:
            balanced_score = 100 - abs(risk_score - 50)
            st.progress(balanced_score / 100)
            st.write(f"균형투자형: {round(balanced_score)}%")

        with col3:
            aggressive_score = max(0, risk_score)
            st.progress(aggressive_score / 100)
            st.write(f"적극투자형: {round(aggressive_score)}%")

    def _display_investment_advice(self, risk_metrics):
        """투자 조언 표시"""
        st.subheader("투자 조언")

        advice = []

        # 베타 기반 조언
        if risk_metrics['beta'] > 1.3:
            advice.append("시장보다 변동성이 매우 큰 주식입니다. 시장이 상승할 때 더 큰 상승을 기대할 수 있지만, 하락장에서도 더 큰 하락을 겪을 수 있습니다.")
        elif risk_metrics['beta'] < 0.7:
            advice.append("방어적인 성격의 주식으로, 시장 하락 시 방어력이 높은 편입니다. 안정적인 포트폴리오를 구성하는 데 도움이 될 수 있습니다.")

        # 샤프 비율 기반 조언
        if risk_metrics['sharpe_ratio'] > 1:
            advice.append("위험 대비 수익률이 양호한 편으로, 효율적인 투자 대상이 될 수 있습니다.")
        elif risk_metrics['sharpe_ratio'] < 0.5:
            advice.append("감수하는 위험에 비해 수익률이 낮은 편입니다. 투자 목적과 기대수익을 재검토해보세요.")

        # 최대 낙폭 기반 조언
        if risk_metrics['max_drawdown'] > 40:
            advice.append(f"최대 낙폭이 {risk_metrics['max_drawdown']}%로 상당히 큰 편입니다. 큰 변동성을 감내할 수 있는 투자자에게 적합합니다.")

        # 승률 기반 조언
        if risk_metrics['winning_ratio'] > 55:
            advice.append(f"거래일 기준 승률이 {risk_metrics['winning_ratio']}%로 양호한 편입니다.")

        # 알파 기반 조언
        if risk_metrics['annual_alpha'] > 3:
            advice.append(f"시장 대비 연간 {risk_metrics['annual_alpha']}%의 초과 수익을 내고 있어 투자 매력도가 높습니다.")
        elif risk_metrics['annual_alpha'] < -3:
            advice.append(f"시장 대비 성과가 부진하므로 투자 이유를 재검토해보세요.")

        # 조언 표시
        if advice:
            for idx, adv in enumerate(advice):
                st.write(f"{idx + 1}. {adv}")
        else:
            st.write("시장과 유사한 성격의 주식입니다. 개별 기업의 성장성과 재무 건전성을 함께 고려하세요.")

    def _display_financial_statements_tab(self, ticker):
        """재무제표 탭 표시"""
        st.subheader("재무제표")

        # 재무제표 가져오기
        from src.utils.financial_analysis import get_financial_statements
        financials = get_financial_statements(ticker)

        # 재무제표 표시
        with st.expander("재무제표 데이터", expanded=True):
            st.write(financials)

        # 재무제표 인사이트 추출 및 표시
        self._display_financial_insights(financials)

        # 배당금 정보 가져오기
        st.subheader("배당금 정보")
        from src.utils.financial_analysis import get_dividends
        dividends = get_dividends(ticker)

        # 배당금 정보 표시
        with st.expander("배당금 데이터", expanded=True):
            st.write(dividends)

        # 배당금 인사이트 추출 및 표시
        self._display_dividend_insights(dividends, ticker)

    def _display_financial_insights(self, financials):
        """재무제표에서 인사이트 추출 및 표시"""
        st.subheader("💡 재무제표 인사이트")

        # 데이터가 없는 경우 처리
        if financials.empty:
            st.info("재무제표 데이터가 충분하지 않아 인사이트를 제공할 수 없습니다.")
            return

        try:
            # 성장성 인사이트
            st.write("#### 성장성 분석")

            # 수익 관련 항목 찾기
            revenue_row = None
            net_income_row = None
            gross_profit_row = None
            operating_income_row = None

            # 가능한 다양한 레이블 확인
            for possible_revenue in ['Total Revenue', 'Revenue', 'Sales', 'Net Sales']:
                if possible_revenue in financials.index:
                    revenue_row = possible_revenue
                    break

            for possible_net_income in ['Net Income', 'Net Income Common Stockholders', 'Net Earnings']:
                if possible_net_income in financials.index:
                    net_income_row = possible_net_income
                    break

            for possible_gross_profit in ['Gross Profit', 'Gross Income']:
                if possible_gross_profit in financials.index:
                    gross_profit_row = possible_gross_profit
                    break

            for possible_operating_income in ['Operating Income', 'Operating Profit', 'EBIT']:
                if possible_operating_income in financials.index:
                    operating_income_row = possible_operating_income
                    break

            # 성장률 계산 함수
            def calculate_growth(row_data):
                if len(row_data) >= 2:
                    earliest = row_data.iloc[-1]
                    latest = row_data.iloc[0]
                    if earliest != 0 and not pd.isna(earliest) and not pd.isna(latest):
                        return ((latest - earliest) / abs(earliest)) * 100
                return None

            # 연평균 성장률(CAGR) 계산 함수
            def calculate_cagr(row_data):
                if len(row_data) >= 2:
                    earliest = row_data.iloc[-1]
                    latest = row_data.iloc[0]
                    years = len(row_data) - 1
                    if earliest > 0 and not pd.isna(earliest) and not pd.isna(latest) and latest > 0:
                        return (((latest / earliest) ** (1 / years)) - 1) * 100
                return None

            col1, col2 = st.columns(2)

            with col1:
                if revenue_row:
                    revenue_growth = calculate_growth(financials.loc[revenue_row])
                    revenue_cagr = calculate_cagr(financials.loc[revenue_row])

                    if revenue_growth is not None:
                        growth_color = "green" if revenue_growth > 0 else "red"
                        st.write(f"총 매출 성장률: <span style='color:{growth_color};'>{revenue_growth:.2f}%</span>",
                                 unsafe_allow_html=True)

                    if revenue_cagr is not None:
                        cagr_color = "green" if revenue_cagr > 0 else "red"
                        st.write(f"매출 연평균 성장률(CAGR): <span style='color:{cagr_color};'>{revenue_cagr:.2f}%</span>",
                                 unsafe_allow_html=True)

                if operating_income_row:
                    op_income_growth = calculate_growth(financials.loc[operating_income_row])
                    if op_income_growth is not None:
                        growth_color = "green" if op_income_growth > 0 else "red"
                        st.write(f"영업이익 성장률: <span style='color:{growth_color};'>{op_income_growth:.2f}%</span>",
                                 unsafe_allow_html=True)

            with col2:
                if net_income_row:
                    net_income_growth = calculate_growth(financials.loc[net_income_row])
                    net_income_cagr = calculate_cagr(financials.loc[net_income_row])

                    if net_income_growth is not None:
                        growth_color = "green" if net_income_growth > 0 else "red"
                        st.write(f"순이익 성장률: <span style='color:{growth_color};'>{net_income_growth:.2f}%</span>",
                                 unsafe_allow_html=True)

                    if net_income_cagr is not None:
                        cagr_color = "green" if net_income_cagr > 0 else "red"
                        st.write(f"순이익 연평균 성장률(CAGR): <span style='color:{cagr_color};'>{net_income_cagr:.2f}%</span>",
                                 unsafe_allow_html=True)

                # 수익성 인사이트
                if revenue_row and net_income_row:
                    latest_margin = (financials.loc[net_income_row].iloc[0] / financials.loc[revenue_row].iloc[0]) * 100
                    margin_color = "green" if latest_margin > 10 else "orange" if latest_margin > 5 else "red"
                    st.write(f"최근 순이익률: <span style='color:{margin_color};'>{latest_margin:.2f}%</span>",
                             unsafe_allow_html=True)

            # 연도별 트렌드 시각화
            st.write("#### 주요 지표 추이")

            # 차트 데이터 준비
            chart_data = pd.DataFrame()

            if revenue_row:
                chart_data['매출'] = financials.loc[revenue_row]
            if net_income_row:
                chart_data['순이익'] = financials.loc[net_income_row]
            if operating_income_row:
                chart_data['영업이익'] = financials.loc[operating_income_row]

            # 차트 그리기
            if not chart_data.empty:
                import plotly.graph_objects as go

                fig = go.Figure()

                for column in chart_data.columns:
                    fig.add_trace(
                        go.Bar(
                            x=chart_data.columns,
                            y=chart_data.iloc[-1],
                            name=f'최근년도({chart_data.columns[-1]})'
                        )
                    )
                    fig.add_trace(
                        go.Bar(
                            x=chart_data.columns,
                            y=chart_data.iloc[0],
                            name=f'최근년도({chart_data.columns[0]})'
                        )
                    )

                fig.update_layout(
                    title="주요 재무 지표 비교",
                    xaxis_title="지표",
                    yaxis_title="금액",
                    barmode='group',
                    height=400
                )

                st.plotly_chart(fig, use_container_width=True)

            # 재무 건전성 분석
            st.write("#### 재무 비율 분석")

            # 자산, 부채, 자본 관련 항목 찾기
            total_assets_row = None
            total_liabilities_row = None
            total_equity_row = None

            for possible_assets in ['Total Assets', 'Assets']:
                if possible_assets in financials.index:
                    total_assets_row = possible_assets
                    break

            for possible_liabilities in ['Total Liabilities', 'Liabilities']:
                if possible_liabilities in financials.index:
                    total_liabilities_row = possible_liabilities
                    break

            for possible_equity in ['Total Equity', 'Stockholders Equity', 'Equity']:
                if possible_equity in financials.index:
                    total_equity_row = possible_equity
                    break

            col1, col2 = st.columns(2)

            with col1:
                # 부채비율 계산
                if total_liabilities_row and total_equity_row:
                    try:
                        latest_debt_ratio = (financials.loc[total_liabilities_row].iloc[0] /
                                             financials.loc[total_equity_row].iloc[0]) * 100
                        ratio_color = "green" if latest_debt_ratio < 100 else "orange" if latest_debt_ratio < 200 else "red"
                        st.write(f"부채비율: <span style='color:{ratio_color};'>{latest_debt_ratio:.2f}%</span>",
                                 unsafe_allow_html=True)

                        if latest_debt_ratio < 100:
                            st.write("👍 부채비율이 낮아 재무적으로 안정적입니다.")
                        elif latest_debt_ratio < 200:
                            st.write("⚠️ 부채비율이 중간 수준입니다. 업종 평균과 비교해 보세요.")
                        else:
                            st.write("⚠️ 부채비율이 높습니다. 재무 건전성에 주의가 필요합니다.")
                    except:
                        st.write("부채비율을 계산할 수 없습니다.")

            with col2:
                # ROE(자기자본이익률) 계산
                if net_income_row and total_equity_row:
                    try:
                        latest_roe = (financials.loc[net_income_row].iloc[0] / financials.loc[total_equity_row].iloc[
                            0]) * 100
                        roe_color = "green" if latest_roe > 15 else "orange" if latest_roe > 8 else "red"
                        st.write(f"ROE(자기자본이익률): <span style='color:{roe_color};'>{latest_roe:.2f}%</span>",
                                 unsafe_allow_html=True)

                        if latest_roe > 15:
                            st.write("👍 ROE가 높아 자본 대비 수익성이 우수합니다.")
                        elif latest_roe > 8:
                            st.write("✅ ROE가 평균 수준입니다.")
                        else:
                            st.write("⚠️ ROE가 낮습니다. 자본 대비 수익성에 주의가 필요합니다.")
                    except:
                        st.write("ROE를 계산할 수 없습니다.")

            # 종합 인사이트
            st.write("#### 종합 인사이트")

            insights = []

            # 성장성 인사이트
            if revenue_row and revenue_growth is not None:
                if revenue_growth > 10:
                    insights.append("📈 매출이 빠르게 성장하고 있습니다.")
                elif revenue_growth > 0:
                    insights.append("✅ 매출이 안정적으로 성장하고 있습니다.")
                else:
                    insights.append("📉 매출이 감소하고 있습니다. 성장성에 주의가 필요합니다.")

            # 수익성 인사이트
            if revenue_row and net_income_row:
                try:
                    latest_margin = (financials.loc[net_income_row].iloc[0] / financials.loc[revenue_row].iloc[0]) * 100
                    earliest_margin = (financials.loc[net_income_row].iloc[-1] / financials.loc[revenue_row].iloc[
                        -1]) * 100

                    if latest_margin > earliest_margin:
                        insights.append("💰 순이익률이 개선되고 있어 비용 효율성이 향상되고 있습니다.")
                    else:
                        insights.append("⚠️ 순이익률이 하락 추세입니다. 비용 구조 최적화가 필요할 수 있습니다.")
                except:
                    pass

            # 영업이익 vs 순이익 인사이트
            if operating_income_row and net_income_row:
                try:
                    op_income_growth = calculate_growth(financials.loc[operating_income_row])
                    net_income_growth = calculate_growth(financials.loc[net_income_row])

                    if op_income_growth > 0 and net_income_growth < 0:
                        insights.append("📊 본업은 성장하고 있으나 순이익이 감소하는 추세입니다. 이자비용이나 세금 부담을 확인해 보세요.")
                    elif op_income_growth < 0 and net_income_growth > 0:
                        insights.append("📊 본업 수익은 감소하나 순이익이 증가하는 특이한 패턴입니다. 일회성 수익이 있는지 확인해 보세요.")
                except:
                    pass

            # 부채비율 인사이트
            if total_liabilities_row and total_equity_row:
                try:
                    latest_debt_ratio = (financials.loc[total_liabilities_row].iloc[0] /
                                         financials.loc[total_equity_row].iloc[0]) * 100
                    earliest_debt_ratio = (financials.loc[total_liabilities_row].iloc[-1] /
                                           financials.loc[total_equity_row].iloc[-1]) * 100

                    if latest_debt_ratio > earliest_debt_ratio:
                        insights.append("🏦 부채비율이 증가 추세입니다. 재무 건전성에 주의가 필요합니다.")
                    else:
                        insights.append("🛡️ 부채비율이 개선되고 있어 재무 안정성이 강화되고 있습니다.")
                except:
                    pass

            # 인사이트 표시
            if insights:
                for insight in insights:
                    st.write(f"- {insight}")
            else:
                st.write("충분한 데이터가 없어 상세 인사이트를 제공할 수 없습니다.")

        except Exception as e:
            import traceback
            st.error(f"재무제표 인사이트 생성 중 오류가 발생했습니다: {str(e)}")
            logger.error(f"재무제표 인사이트 생성 오류: {str(e)}")
            logger.error(traceback.format_exc())

    def _display_dividend_insights(self, dividends, ticker):
        """배당금 정보에서 인사이트 추출 및 표시"""
        st.subheader("💡 배당금 인사이트")

        # 데이터가 없거나 비어있는 경우 처리
        if dividends is None or dividends.empty:
            st.info("배당금 데이터가 충분하지 않아 인사이트를 제공할 수 없습니다.")
            return

        try:
            # 배당 트렌드 분석
            st.write("#### 배당금 트렌드 분석")

            # 최근 5년 데이터로 제한 (데이터가 충분하다면)
            recent_years = 5
            if len(dividends) > recent_years:
                recent_dividends = dividends.tail(recent_years)
            else:
                recent_dividends = dividends

            # 배당금 증가율 계산
            if len(recent_dividends) >= 2:
                earliest_dividend = recent_dividends.iloc[0]
                latest_dividend = recent_dividends.iloc[-1]
                dividend_growth = ((latest_dividend - earliest_dividend) / earliest_dividend) * 100

                # 연평균 배당 증가율(CAGR) 계산
                years = len(recent_dividends) - 1
                if years > 0 and earliest_dividend > 0 and latest_dividend > 0:
                    dividend_cagr = (((latest_dividend / earliest_dividend) ** (1 / years)) - 1) * 100
                else:
                    dividend_cagr = None

                col1, col2 = st.columns(2)

                with col1:
                    if dividend_growth is not None:
                        growth_color = "green" if dividend_growth > 0 else "red"
                        st.write(f"배당금 성장률: <span style='color:{growth_color};'>{dividend_growth:.2f}%</span>",
                                 unsafe_allow_html=True)

                    # 배당 횟수 (연간)
                    years_with_dividends = recent_dividends.resample('Y').sum()
                    avg_annual_payments = len(recent_dividends) / len(years_with_dividends)
                    st.write(f"평균 연간 배당 횟수: {avg_annual_payments:.1f}회")

                with col2:
                    if dividend_cagr is not None:
                        cagr_color = "green" if dividend_cagr > 0 else "red"
                        st.write(f"배당금 연평균 성장률(CAGR): <span style='color:{cagr_color};'>{dividend_cagr:.2f}%</span>",
                                 unsafe_allow_html=True)

                    # 배당 주기 파악
                    if avg_annual_payments >= 4:
                        st.write("📊 분기 배당 기업")
                    elif avg_annual_payments >= 2:
                        st.write("📊 반기 배당 기업")
                    else:
                        st.write("📊 연간 배당 기업")

            # 배당 수익률 계산 (현재 주가 정보 필요)
            st.write("#### 배당 수익률 분석")

            try:
                # yfinance로 현재 주가 및 배당 정보 가져오기
                import yfinance as yf
                stock_info = yf.Ticker(ticker).info

                if 'dividendYield' in stock_info and stock_info['dividendYield'] is not None:
                    dividend_yield = stock_info['dividendYield'] * 100

                    yield_color = "green" if dividend_yield > 3 else "orange" if dividend_yield > 1 else "red"
                    st.write(f"현재 배당 수익률: <span style='color:{yield_color};'>{dividend_yield:.2f}%</span>",
                             unsafe_allow_html=True)

                    if dividend_yield > 5:
                        st.write("💰 높은 배당 수익률을 제공하는 고배당 주식입니다.")
                    elif dividend_yield > 3:
                        st.write("✅ 양호한 배당 수익률을 제공합니다.")
                    elif dividend_yield > 1:
                        st.write("📊 평균적인 배당 수익률을 제공합니다.")
                    else:
                        st.write("⚠️ 배당 수익률이 낮습니다. 성장주일 가능성이 높습니다.")

                if 'payoutRatio' in stock_info and stock_info['payoutRatio'] is not None:
                    payout_ratio = stock_info['payoutRatio'] * 100

                    ratio_color = "green" if payout_ratio < 70 else "orange" if payout_ratio < 90 else "red"
                    st.write(f"배당성향(Payout Ratio): <span style='color:{ratio_color};'>{payout_ratio:.2f}%</span>",
                             unsafe_allow_html=True)

                    if payout_ratio < 30:
                        st.write("💡 배당성향이 보수적입니다. 향후 배당 증가 여력이 있습니다.")
                    elif payout_ratio < 70:
                        st.write("✅ 적정한 배당성향을 유지하고 있습니다.")
                    elif payout_ratio < 90:
                        st.write("⚠️ 배당성향이 높은 편입니다. 배당 지속가능성을 체크해 보세요.")
                    else:
                        st.write("🚨 매우 높은 배당성향입니다. 수익 대비 과도한 배당금을 지급할 수 있습니다.")
            except:
                st.info("현재 주가 정보를 가져올 수 없어 배당 수익률과 배당성향을 계산할 수 없습니다.")

            # 배당금 추이 시각화
            if not recent_dividends.empty:
                st.write("#### 배당금 추이")

                import plotly.graph_objects as go

                fig = go.Figure()

                fig.add_trace(
                    go.Scatter(
                        x=recent_dividends.index,
                        y=recent_dividends.values,
                        mode='lines+markers',
                        name='배당금',
                        line=dict(color='green', width=2),
                        marker=dict(size=8)
                    )
                )

                fig.update_layout(
                    title='배당금 추이',
                    xaxis_title='날짜',
                    yaxis_title='배당금',
                    height=400
                )

                st.plotly_chart(fig, use_container_width=True)

            # 종합 인사이트
            st.write("#### 종합 인사이트")

            insights = []

            # 배당 성장성 인사이트
            if len(recent_dividends) >= 2:
                if dividend_growth > 0:
                    insights.append("📈 배당금이 꾸준히 증가하고 있어 배당 투자에 적합할 수 있습니다.")
                elif dividend_growth == 0:
                    insights.append("📊 배당금이 변동 없이 유지되고 있습니다. 안정적인 배당을 선호하는 투자자에게 적합합니다.")
                else:
                    insights.append("📉 배당금이 감소하고 있습니다. 배당 정책 변화에 주의가 필요합니다.")

            # 배당 주기 인사이트
            if avg_annual_payments >= 4:
                insights.append("🔄 분기 배당으로 현금흐름이 안정적입니다.")

            # 배당 수익률 인사이트 (yfinance 데이터가 있는 경우)
            try:
                if 'dividendYield' in stock_info and stock_info['dividendYield'] is not None:
                    dividend_yield = stock_info['dividendYield'] * 100

                    if dividend_yield > 3:
                        insights.append(f"💰 {dividend_yield:.2f}%의 배당 수익률은 시장 평균보다 높습니다.")

                # 배당성향과 배당 지속가능성
                if 'payoutRatio' in stock_info and 'dividendYield' in stock_info:
                    payout_ratio = stock_info['payoutRatio'] * 100
                    dividend_yield = stock_info['dividendYield'] * 100

                    if payout_ratio > 80 and dividend_yield > 5:
                        insights.append("⚠️ 높은 배당 수익률과 높은 배당성향은 배당의 지속가능성에 의문을 제기합니다.")
                    elif payout_ratio < 50 and dividend_yield > 3:
                        insights.append("👍 적절한 배당성향과 높은 배당 수익률은 이상적인 배당주 특성입니다.")
            except:
                pass

            # 배당 일관성 인사이트
            consecutive_years = 0
            if len(dividends) > 0:
                # 연도별 배당금 합계 계산
                annual_dividends = dividends.resample('Y').sum()

                # 연속 배당 년수 계산
                for i in range(len(annual_dividends) - 1, 0, -1):
                    if annual_dividends.iloc[i] > 0:
                        consecutive_years += 1
                    else:
                        break

                if consecutive_years >= 5:
                    insights.append(f"🏆 최소 {consecutive_years}년 연속 배당을 유지하고 있는 안정적인 배당주입니다.")
                elif consecutive_years >= 3:
                    insights.append(f"👍 {consecutive_years}년 연속 배당을 지급하고 있습니다.")

            # 인사이트 표시
            if insights:
                for insight in insights:
                    st.write(f"- {insight}")
            else:
                st.write("충분한 데이터가 없어 상세 인사이트를 제공할 수 없습니다.")

            # 배당 투자자 유형 적합도
            st.write("#### 배당 투자자 유형 적합도")

            try:
                # 적합도 점수 계산 (0-100)
                income_investor_score = 0
                growth_investor_score = 0

                # yfinance 정보가 있는 경우
                if 'dividendYield' in locals() and 'payout_ratio' in locals():
                    # 배당 수익률 기여도 (최대 50점)
                    income_investor_score += min(dividend_yield * 10, 50)

                    # 배당성향 기여도 (최대 20점)
                    if payout_ratio < 30:
                        growth_investor_score += 20
                    elif payout_ratio < 50:
                        income_investor_score += 10
                        growth_investor_score += 10
                    else:
                        income_investor_score += 20

                # 배당 성장률 기여도 (최대 30점)
                if 'dividend_growth' in locals() and dividend_growth is not None:
                    if dividend_growth > 10:
                        growth_investor_score += 30
                    elif dividend_growth > 5:
                        growth_investor_score += 20
                        income_investor_score += 10
                    elif dividend_growth > 0:
                        growth_investor_score += 10
                        income_investor_score += 5

                # 배당 주기 기여도 (최대 20점)
                if 'avg_annual_payments' in locals():
                    if avg_annual_payments >= 4:
                        income_investor_score += 20
                    elif avg_annual_payments >= 2:
                        income_investor_score += 10

                # 연속 배당 기여도 (최대 20점)
                if consecutive_years >= 10:
                    income_investor_score += 20
                elif consecutive_years >= 5:
                    income_investor_score += 10

                # 점수 정규화 (최대 100점)
                income_investor_score = min(income_investor_score, 100)
                growth_investor_score = min(growth_investor_score, 100)

                col1, col2 = st.columns(2)

                with col1:
                    st.write("소득형 투자자 적합도")
                    st.progress(income_investor_score / 100)
                    st.write(f"{income_investor_score}/100")

                    if income_investor_score >= 70:
                        st.write("👍 소득형 투자자에게 매우 적합")
                    elif income_investor_score >= 50:
                        st.write("✅ 소득형 투자자에게 적합")
                    else:
                        st.write("⚠️ 소득형 투자자에게 적합하지 않음")

                with col2:
                    st.write("성장형 배당 투자자 적합도")
                    st.progress(growth_investor_score / 100)
                    st.write(f"{growth_investor_score}/100")

                    if growth_investor_score >= 70:
                        st.write("👍 성장형 배당 투자자에게 매우 적합")
                    elif growth_investor_score >= 50:
                        st.write("✅ 성장형 배당 투자자에게 적합")
                    else:
                        st.write("⚠️ 성장형 배당 투자자에게 적합하지 않음")

            except Exception as e:
                st.info("투자자 유형 적합도를 계산할 수 없습니다.")

        except Exception as e:
            import traceback
            st.error(f"배당금 인사이트 생성 중 오류가 발생했습니다: {str(e)}")
            logger.error(f"배당금 인사이트 생성 오류: {str(e)}")
            logger.error(traceback.format_exc())

    def _display_growth_analysis_tab(self, ticker):
        """성장성 분석 탭 표시"""
        st.subheader("성장성 분석")

        # 로딩 상태 표시
        with st.spinner("성장성 지표 분석 중..."):
            # 성장성 분석 실행
            from src.utils.financial_analysis import analyze_growth_rates
            growth_data = analyze_growth_rates(ticker)

            # 세션 상태에 저장 및 growth_data 구조 개선
            st.session_state.growth_data = growth_data

            # 종합리포트에 데이터 등록 (추가된 부분)
            try:
                from src.views.comprehensive_report_view import ComprehensiveReportView
                comprehensive_view = ComprehensiveReportView()
                if 'financial_analysis' not in st.session_state.comprehensive_data:
                    st.session_state.comprehensive_data['financial_analysis'] = {}
                st.session_state.comprehensive_data['financial_analysis']['growth_data'] = growth_data
            except Exception as e:
                logger.warning(f"종합리포트에 성장률 데이터 등록 실패: {e}")

            # 오류 확인 및 처리
            if growth_data.get('error', False):
                st.error(growth_data.get('error_message', "성장성 분석 중 오류가 발생했습니다."))
                st.warning("예시 데이터를 기반으로 한 차트를 표시합니다.")
                st.info("정확한 분석을 위해 주식 티커가 올바른지 확인하세요.")

            # 성장성 지표 해석 가이드
            self._display_growth_metrics_guide()

            # 연간 성장률 차트 표시
            self._display_annual_growth_chart(growth_data)

            # 분기별 성장률 차트 표시
            self._display_quarterly_growth_chart(growth_data)

            # 실적 추이 (절대값) 차트 표시
            self._display_absolute_performance_chart(growth_data)

            # 성장성 종합 평가
            self._display_growth_assessment(growth_data)

    def _display_growth_metrics_guide(self):
        """성장성 지표 해석 가이드 표시"""
        with st.expander("성장성 지표 해석 가이드", expanded=False):
            st.markdown("""
                - **매출 성장률**: 회사의 총 매출이 얼마나 빠르게 성장하는지를 나타냅니다. 산업 평균보다 높으면 경쟁력이 있다고 볼 수 있습니다.
                - **영업이익 성장률**: 회사의 핵심 사업에서 발생하는 이익의 성장률입니다. 매출 성장률보다 높으면 수익성이 개선되고 있다는 의미입니다.
                - **순이익 성장률**: 모든 비용과 세금을 제외한 최종 이익의 성장률입니다. 회사의 전반적인 수익성을 보여줍니다.
                - **EPS 성장률**: 주당순이익의 성장률로, 주주 입장에서 가장 중요한 지표 중 하나입니다.
    
                **지속 가능한 성장**이란 장기간에 걸쳐 안정적인 성장률을 유지하는 것을 의미하며, 
                **급격한 변동**은 사업 모델의 불안정성을 암시할 수 있습니다.
                """)

    def _display_annual_growth_chart(self, growth_data):
        """연간 성장률 차트 표시"""
        # 데이터 준비 - 연간
        annual_data = growth_data.get('annual', {})
        years = annual_data.get('years', [])
        revenue_growth = annual_data.get('revenue_growth', [])
        operating_income_growth = annual_data.get('operating_income_growth', [])
        net_income_growth = annual_data.get('net_income_growth', [])
        eps_growth = annual_data.get('eps_growth', [])

        if years and len(years) > 1:
            st.subheader("연간 성장률 추이")

            # Plotly 그래프 작성
            fig = go.Figure()

            # 데이터가 있는 경우에만 차트에 추가
            if revenue_growth and any(x is not None for x in revenue_growth[1:]):
                fig.add_trace(go.Scatter(
                    x=years[1:],
                    y=[x if x is not None else 0 for x in revenue_growth[1:]],
                    mode='lines+markers',
                    name='매출 성장률',
                    line=dict(color='royalblue', width=3),
                    marker=dict(size=8)
                ))

            if operating_income_growth and any(x is not None for x in operating_income_growth[1:]):
                fig.add_trace(go.Scatter(
                    x=years[1:],
                    y=[x if x is not None else 0 for x in operating_income_growth[1:]],
                    mode='lines+markers',
                    name='영업이익 성장률',
                    line=dict(color='green', width=3),
                    marker=dict(size=8)
                ))

            if net_income_growth and any(x is not None for x in net_income_growth[1:]):
                fig.add_trace(go.Scatter(
                    x=years[1:],
                    y=[x if x is not None else 0 for x in net_income_growth[1:]],
                    mode='lines+markers',
                    name='순이익 성장률',
                    line=dict(color='firebrick', width=3),
                    marker=dict(size=8)
                ))

            if eps_growth and any(x is not None for x in eps_growth[1:]):
                fig.add_trace(go.Scatter(
                    x=years[1:],
                    y=[x if x is not None else 0 for x in eps_growth[1:]],
                    mode='lines+markers',
                    name='EPS 성장률',
                    line=dict(color='purple', width=3),
                    marker=dict(size=8)
                ))

            # 차트 레이아웃 설정
            fig.update_layout(
                title='연간 성장률 추이 (%)',
                xaxis_title='연도',
                yaxis_title='성장률 (%)',
                height=500,
                hovermode='x unified',
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=-0.2,
                    xanchor="center",
                    x=0.5
                )
            )

            # 0% 기준선 추가
            fig.add_hline(y=0, line_width=1, line_dash="dash", line_color="gray")

            st.plotly_chart(fig, use_container_width=True)

            # 성장률 요약 테이블
            st.subheader("성장률 요약")

            # 최근 3년간의 평균 성장률 계산
            recent_years = 3
            if len(years) >= recent_years + 1:  # 첫 항목은 성장률이 없으므로 +1
                recent_revenue_growth = [x for x in revenue_growth[-recent_years:] if x is not None]
                recent_operating_income_growth = [x for x in operating_income_growth[-recent_years:] if x is not None]
                recent_net_income_growth = [x for x in net_income_growth[-recent_years:] if x is not None]
                recent_eps_growth = [x for x in eps_growth[-recent_years:] if x is not None]

                avg_revenue_growth = sum(recent_revenue_growth) / len(recent_revenue_growth) if recent_revenue_growth else 0
                avg_operating_income_growth = sum(recent_operating_income_growth) / len(
                    recent_operating_income_growth) if recent_operating_income_growth else 0
                avg_net_income_growth = sum(recent_net_income_growth) / len(
                    recent_net_income_growth) if recent_net_income_growth else 0
                avg_eps_growth = sum(recent_eps_growth) / len(recent_eps_growth) if recent_eps_growth else 0

                # 요약 테이블 표시
                summary_data = {
                    "지표": ["매출 성장률", "영업이익 성장률", "순이익 성장률", "EPS 성장률"],
                    f"최근 {recent_years}년 평균": [
                        f"{avg_revenue_growth:.2f}%",
                        f"{avg_operating_income_growth:.2f}%",
                        f"{avg_net_income_growth:.2f}%",
                        f"{avg_eps_growth:.2f}%"
                    ],
                    "최근 성장률": [
                        f"{revenue_growth[-1]:.2f}%" if revenue_growth and revenue_growth[-1] is not None else "N/A",
                        f"{operating_income_growth[-1]:.2f}%" if operating_income_growth and operating_income_growth[
                            -1] is not None else "N/A",
                        f"{net_income_growth[-1]:.2f}%" if net_income_growth and net_income_growth[
                            -1] is not None else "N/A",
                        f"{eps_growth[-1]:.2f}%" if eps_growth and eps_growth[-1] is not None else "N/A"
                    ]
                }

                import pandas as pd
                summary_df = pd.DataFrame(summary_data)
                st.table(summary_df)

    def _display_quarterly_growth_chart(self, growth_data):
        """분기별 성장률 차트 표시"""
        # 분기별 데이터
        quarterly_data = growth_data.get('quarterly', {})
        quarters = quarterly_data.get('quarters', [])
        quarterly_revenue_growth = quarterly_data.get('revenue_growth', [])
        quarterly_net_income_growth = quarterly_data.get('net_income_growth', [])

        # 분기별 성장률 차트
        if quarters and len(quarters) > 4:  # 최소 5개 분기 데이터 필요 (4분기 전과 비교하므로)
            st.subheader("분기별 성장률 추이 (YoY)")

            # 유효한 데이터만 추출
            valid_data = []
            for i in range(len(quarters)):
                revenue_growth = quarterly_revenue_growth[i] if i < len(quarterly_revenue_growth) and \
                                                                quarterly_revenue_growth[i] is not None else None
                net_income_growth = quarterly_net_income_growth[i] if i < len(quarterly_net_income_growth) and \
                                                                      quarterly_net_income_growth[
                                                                          i] is not None else None

                if revenue_growth is not None or net_income_growth is not None:
                    if not np.isnan(revenue_growth) if revenue_growth is not None else True and not np.isnan(
                            net_income_growth) if net_income_growth is not None else True:
                        valid_data.append({
                            "quarter": quarters[i],
                            "revenue_growth": revenue_growth,
                            "net_income_growth": net_income_growth
                        })

            if valid_data:
                fig = go.Figure()

                # 매출 성장률 (YoY) - 막대 그래프
                fig.add_trace(go.Bar(
                    x=[d["quarter"] for d in valid_data],
                    y=[d["revenue_growth"] if d["revenue_growth"] is not None else 0 for d in valid_data],
                    name='매출 성장률 (YoY)',
                    marker_color='rgba(65, 105, 225, 0.7)',  # 로열 블루, 반투명
                    text=[f"{d['revenue_growth']:.1f}%" if d['revenue_growth'] is not None and not np.isnan(
                        d['revenue_growth']) else "" for d in valid_data],
                    textposition='outside',
                    width=0.6  # 막대 너비 조정
                ))

                # 순이익 성장률 (YoY) - 막대 그래프 (다른 색상)
                fig.add_trace(go.Bar(
                    x=[d["quarter"] for d in valid_data],
                    y=[d["net_income_growth"] if d["net_income_growth"] is not None else 0 for d in valid_data],
                    name='순이익 성장률 (YoY)',
                    marker_color='rgba(220, 20, 60, 0.7)',  # 크림슨, 반투명
                    text=[f"{d['net_income_growth']:.1f}%" if d['net_income_growth'] is not None and not np.isnan(
                        d['net_income_growth']) else "" for d in valid_data],
                    textposition='outside',
                    width=0.6  # 막대 너비 조정
                ))

                # 추세선 추가 (예외 처리 추가)
                if len(valid_data) >= 3:  # 최소 3개 이상의 데이터 필요
                    try:
                        # 매출 성장률 추세선
                        revenue_values = [d["revenue_growth"] for d in valid_data if
                                          d["revenue_growth"] is not None and not np.isnan(d["revenue_growth"])]
                        if len(revenue_values) >= 3:  # 최소 3개 이상의 유효한 데이터 필요
                            x_indices = list(range(len(revenue_values)))

                            # 데이터의 다양성 확인 (모두 같은 값인지)
                            if len(set(revenue_values)) > 1:  # 값이 최소 2개 이상 다른 경우에만 진행
                                z = np.polyfit(x_indices, revenue_values, 1)
                                p = np.poly1d(z)
                                trend_x = list(range(len(valid_data)))
                                trend_y = p(trend_x)

                                fig.add_trace(go.Scatter(
                                    x=[d["quarter"] for d in valid_data],
                                    y=trend_y,
                                    mode='lines',
                                    name='매출 성장 추세',
                                    line=dict(color='blue', width=2, dash='dot')
                                ))
                    except Exception as e:
                        st.warning(f"매출 성장 추세선 계산 중 오류 발생: {str(e)}")

                    try:
                        # 순이익 성장률 추세선
                        net_income_values = [d["net_income_growth"] for d in valid_data if
                                             d["net_income_growth"] is not None and not np.isnan(
                                                 d["net_income_growth"])]
                        if len(net_income_values) >= 3:  # 최소 3개 이상의 유효한 데이터 필요
                            x_indices = list(range(len(net_income_values)))

                            # 데이터의 다양성 확인 (모두 같은 값인지)
                            if len(set(net_income_values)) > 1:  # 값이 최소 2개 이상 다른 경우에만 진행
                                z = np.polyfit(x_indices, net_income_values, 1)
                                p = np.poly1d(z)
                                trend_x = list(range(len(valid_data)))
                                trend_y = p(trend_x)

                                fig.add_trace(go.Scatter(
                                    x=[d["quarter"] for d in valid_data],
                                    y=trend_y,
                                    mode='lines',
                                    name='순이익 성장 추세',
                                    line=dict(color='red', width=2, dash='dot')
                                ))
                    except Exception as e:
                        st.warning(f"순이익 성장 추세선 계산 중 오류 발생: {str(e)}")

                # 차트 레이아웃 설정 개선
                fig.update_layout(
                    title={
                        'text': '분기별 전년동기대비 성장률 (%)',
                        'font': {'size': 22}
                    },
                    xaxis_title='분기',
                    yaxis_title='성장률 (%)',
                    barmode='group',  # 막대를 그룹화하여 같은 분기의 데이터를 나란히 표시
                    height=600,  # 차트 높이 증가
                    hovermode='x unified',
                    hoverlabel=dict(
                        bgcolor="white",
                        font_size=14
                    ),
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=-0.2,
                        xanchor="center",
                        x=0.5,
                        font=dict(size=14)
                    ),
                    margin=dict(l=50, r=50, t=80, b=100)  # 여백 조정
                )

                # 0% 기준선 추가 및 스타일 개선
                fig.add_hline(
                    y=0,
                    line_width=1.5,
                    line_dash="solid",
                    line_color="gray",
                    annotation_text="성장/축소 경계선",
                    annotation_position="bottom right"
                )

                # y 최대/최소값 계산 (유효한 데이터만 사용)
                valid_revenue_values = [d["revenue_growth"] for d in valid_data if
                                        d["revenue_growth"] is not None and not np.isnan(d["revenue_growth"])]
                valid_income_values = [d["net_income_growth"] for d in valid_data if
                                       d["net_income_growth"] is not None and not np.isnan(d["net_income_growth"])]

                max_y = max([max(valid_revenue_values) if valid_revenue_values else 0,
                             max(valid_income_values) if valid_income_values else 0,
                             5]) * 1.2  # 최대값보다 20% 높게

                min_y = min([min(valid_revenue_values) if valid_revenue_values else 0,
                             min(valid_income_values) if valid_income_values else 0,
                             -5]) * 1.2  # 최소값보다 20% 낮게

                # 양수/음수 영역 색상 구분
                fig.add_hrect(
                    y0=0, y1=max_y,
                    fillcolor="rgba(0, 255, 0, 0.05)",
                    line_width=0,
                    annotation_text="성장 구간",
                    annotation_position="top right"
                )
                fig.add_hrect(
                    y0=min_y, y1=0,
                    fillcolor="rgba(255, 0, 0, 0.05)",
                    line_width=0,
                    annotation_text="축소 구간",
                    annotation_position="bottom right"
                )

                # X축 레이블 형식 개선
                fig.update_xaxes(
                    tickangle=45,
                    tickfont=dict(size=12)
                )

                # Y축 그리드 추가
                fig.update_yaxes(
                    gridcolor='lightgray',
                    griddash='dot',
                    zeroline=True,
                    zerolinecolor='black',
                    zerolinewidth=2
                )

                # 차트 표시
                st.plotly_chart(fig, use_container_width=True)

                # 분기별 성장률 해석 추가
                latest_quarters = valid_data[-4:] if len(valid_data) >= 4 else valid_data

                # 데이터 유효성 검증 추가
                valid_revenue_data = [q.get("revenue_growth") for q in latest_quarters
                                      if q.get("revenue_growth") is not None and not np.isnan(q.get("revenue_growth"))]
                valid_income_data = [q.get("net_income_growth") for q in latest_quarters
                                     if q.get("net_income_growth") is not None and not np.isnan(
                        q.get("net_income_growth"))]

                # 최근 추세 계산 (충분한 데이터가 있을 때만)
                if valid_revenue_data:
                    revenue_trend = "상승" if all(val >= 0 for val in valid_revenue_data) else "하락" if all(
                        val <= 0 for val in valid_revenue_data) else "혼조"
                else:
                    revenue_trend = "데이터 없음"

                if valid_income_data:
                    income_trend = "상승" if all(val >= 0 for val in valid_income_data) else "하락" if all(
                        val <= 0 for val in valid_income_data) else "혼조"
                else:
                    income_trend = "데이터 없음"

                # 인사이트 제공
                st.subheader("분기별 성장 해석")

                col1, col2 = st.columns(2)

                with col1:
                    # 최근 분기 성장률
                    latest = valid_data[-1] if valid_data else {}

                    rev_growth = latest.get("revenue_growth")
                    inc_growth = latest.get("net_income_growth")

                    if rev_growth is not None and not np.isnan(rev_growth):
                        color = "green" if rev_growth > 0 else "red"
                        st.markdown(f"**최근 분기 매출 성장률:** <span style='color:{color}'>{rev_growth:.1f}%</span>",
                                    unsafe_allow_html=True)
                    else:
                        st.markdown("**최근 분기 매출 성장률:** 데이터 없음")

                    if inc_growth is not None and not np.isnan(inc_growth):
                        color = "green" if inc_growth > 0 else "red"
                        st.markdown(f"**최근 분기 순이익 성장률:** <span style='color:{color}'>{inc_growth:.1f}%</span>",
                                    unsafe_allow_html=True)
                    else:
                        st.markdown("**최근 분기 순이익 성장률:** 데이터 없음")

                with col2:
                    st.markdown(f"**매출 성장 추세:** {revenue_trend}")
                    st.markdown(f"**순이익 성장 추세:** {income_trend}")

                # 추가 인사이트 (데이터가 충분할 때만)
                if revenue_trend != "데이터 없음" and income_trend != "데이터 없음":
                    if revenue_trend == "상승" and income_trend == "상승":
                        st.success("📈 매출과 순이익이 함께 성장하는 건전한 사업 확장 추세입니다.")
                    elif revenue_trend == "상승" and income_trend == "하락":
                        st.warning("⚠️ 매출은 증가하나 수익성이 하락하고 있습니다. 비용 구조를 확인할 필요가 있습니다.")
                    elif revenue_trend == "하락" and income_trend == "상승":
                        st.info("💡 매출은 줄었으나 수익성이 개선되고 있습니다. 효율적인 운영이 이루어지고 있습니다.")
                    elif revenue_trend == "하락" and income_trend == "하락":
                        st.error("📉 매출과 순이익이 함께 감소하고 있습니다. 사업 모델을 재검토할 필요가 있습니다.")
            else:
                st.info("분기별 성장률 데이터가 충분하지 않습니다.")
        else:
            st.info("분기별 성장률을 표시하기 위한 충분한 데이터(최소 5개 분기)가 없습니다.")


    def _display_absolute_performance_chart(self, growth_data):
        """실적 추이 (절대값) 차트 표시"""
        # 실적 추이 차트 (절대값)
        st.subheader("실적 추이 (절대값)")

        # 데이터 준비
        annual_data = growth_data.get('annual', {})
        years = annual_data.get('years', [])
        revenue = annual_data.get('revenue', [])
        operating_income = annual_data.get('operating_income', [])
        net_income = annual_data.get('net_income', [])

        if years and revenue and len(years) == len(revenue):
            # 단위 변환 (큰 숫자 처리)
            def format_value(value):
                if abs(value) >= 1e12:  # 1조 이상
                    return value / 1e12, "조원"
                elif abs(value) >= 1e9:  # 10억 이상
                    return value / 1e9, "십억원"
                elif abs(value) >= 1e6:  # 백만 이상
                    return value / 1e6, "백만원"
                else:
                    return value, "원"

            # 가장 큰 값 기준으로 단위 결정
            max_revenue = max(revenue) if revenue else 0
            divider, unit = format_value(max_revenue)

            # 매출 및 이익 데이터 변환
            revenue_scaled = [r / divider for r in revenue]
            operating_income_scaled = [oi / divider if oi is not None else None for oi in operating_income]
            net_income_scaled = [ni / divider if ni is not None else None for ni in net_income]

            # 실적 차트
            fig = go.Figure()

            # 매출
            fig.add_trace(go.Bar(
                x=years,
                y=revenue_scaled,
                name='매출',
                marker_color='rgba(58, 71, 80, 0.6)'
            ))

            # 영업이익
            fig.add_trace(go.Bar(
                x=years,
                y=operating_income_scaled if operating_income else [],
                name='영업이익',
                marker_color='rgba(34, 139, 34, 0.6)'
            ))

            # 순이익
            fig.add_trace(go.Bar(
                x=years,
                y=net_income_scaled if net_income else [],
                name='순이익',
                marker_color='rgba(178, 34, 34, 0.6)'
            ))

            # 차트 레이아웃 설정
            fig.update_layout(
                title=f'연간 실적 추이 ({unit})',
                xaxis_title='연도',
                yaxis_title=f'금액 ({unit})',
                barmode='group',
                height=500,
                hovermode='x unified',
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=-0.2,
                    xanchor="center",
                    x=0.5
                )
            )

            st.plotly_chart(fig, use_container_width=True)


    def _display_growth_assessment(self, growth_data):
        """성장성 종합 평가"""
        st.subheader("성장성 종합 평가")

        # 데이터 준비
        annual_data = growth_data.get('annual', {})
        years = annual_data.get('years', [])
        revenue_growth = annual_data.get('revenue_growth', [])
        operating_income_growth = annual_data.get('operating_income_growth', [])
        net_income_growth = annual_data.get('net_income_growth', [])

        # 성장성 평가 로직
        def evaluate_growth(growth_values, thresholds=(5, 15)):
            """성장률을 평가하여 낮음, 중간, 높음으로 구분"""
            if not growth_values or all(x is None for x in growth_values):
                return "평가 불가", "gray"

            # None이 아닌 값만 필터링
            valid_values = [x for x in growth_values if x is not None]
            if not valid_values:
                return "평가 불가", "gray"

            avg_growth = sum(valid_values) / len(valid_values)

            if avg_growth < thresholds[0]:
                return "낮음", "red"
            elif avg_growth < thresholds[1]:
                return "중간", "orange"
            else:
                return "높음", "green"

        # 최근 3년 데이터만 사용
        recent_years_count = min(3, len(years) - 1)  # 첫 해는 성장률이 없으므로 제외
        recent_revenue_growth = revenue_growth[-recent_years_count:] if revenue_growth else []
        recent_operating_income_growth = operating_income_growth[-recent_years_count:] if operating_income_growth else []
        recent_net_income_growth = net_income_growth[-recent_years_count:] if net_income_growth else []

        # 각 지표별 평가
        revenue_evaluation, revenue_color = evaluate_growth(recent_revenue_growth)
        operating_income_evaluation, oi_color = evaluate_growth(recent_operating_income_growth)
        net_income_evaluation, ni_color = evaluate_growth(recent_net_income_growth)

        # 평가 테이블 표시
        st.markdown(f"""
            | 지표 | 평가 | 해석 |
            | --- | --- | --- |
            | 매출 성장성 | <span style="color:{revenue_color}">{revenue_evaluation}</span> | {'높은 매출 성장률은 시장 점유율 확대와 사업 확장을 의미합니다.' if revenue_evaluation == '높음' else '중간 수준의 매출 성장은 안정적인 사업 운영을 나타냅니다.' if revenue_evaluation == '중간' else '낮은 매출 성장은 사업 성숙 또는 경쟁 심화를 의미할 수 있습니다.' if revenue_evaluation == '낮음' else '데이터 부족으로 평가할 수 없습니다.'} |
            | 영업이익 성장성 | <span style="color:{oi_color}">{operating_income_evaluation}</span> | {'높은 영업이익 성장은 운영 효율성 개선과 원가 관리 성공을 의미합니다.' if operating_income_evaluation == '높음' else '중간 수준의 영업이익 성장은 적절한 비용 관리를 나타냅니다.' if operating_income_evaluation == '중간' else '낮은 영업이익 성장은 비용 증가 또는 마진 압박을 의미할 수 있습니다.' if operating_income_evaluation == '낮음' else '데이터 부족으로 평가할 수 없습니다.'} |
            | 순이익 성장성 | <span style="color:{ni_color}">{net_income_evaluation}</span> | {'높은 순이익 성장은 전반적인 재무 건전성과 효율적 경영을 의미합니다.' if net_income_evaluation == '높음' else '중간 수준의 순이익 성장은 안정적 수익성을 나타냅니다.' if net_income_evaluation == '중간' else '낮은 순이익 성장은 수익성 약화 또는 세금/금융비용 증가를 의미할 수 있습니다.' if net_income_evaluation == '낮음' else '데이터 부족으로 평가할 수 없습니다.'} |
            """, unsafe_allow_html=True)

        # 성장성 종합 평가
        avg_score = 0
        count = 0

        if revenue_evaluation != "평가 불가":
            avg_score += 1 if revenue_evaluation == "낮음" else 2 if revenue_evaluation == "중간" else 3
            count += 1

        if operating_income_evaluation != "평가 불가":
            avg_score += 1 if operating_income_evaluation == "낮음" else 2 if operating_income_evaluation == "중간" else 3
            count += 1

        if net_income_evaluation != "평가 불가":
            avg_score += 1 if net_income_evaluation == "낮음" else 2 if net_income_evaluation == "중간" else 3
            count += 1

        if count > 0:
            final_score = avg_score / count

            st.subheader("종합 평가")

            if final_score >= 2.5:
                st.success("이 기업은 높은 성장성을 보이고 있으며, 지속적인 확장과 수익성 개선이 기대됩니다.")
            elif final_score >= 1.5:
                st.info("이 기업은 중간 수준의 성장성을 보이고 있으며, 안정적인 사업 운영이 이루어지고 있습니다.")
            else:
                st.warning("이 기업은 낮은 성장성을 보이고 있으며, 성장 동력 확보가 필요할 수 있습니다.")

            # 산업 평균과 비교 (가상 데이터)
            st.write("**산업 평균 대비 성장성**")

            industry_avg = {
                "revenue_growth": 8.5,
                "operating_income_growth": 7.2,
                "net_income_growth": 6.8
            }

            # 최근 성장률 계산
            recent_avg_revenue_growth = sum([x for x in recent_revenue_growth if x is not None]) / len(
                [x for x in recent_revenue_growth if x is not None]) if any(
                x is not None for x in recent_revenue_growth) else 0
            recent_avg_oi_growth = sum([x for x in recent_operating_income_growth if x is not None]) / len(
                [x for x in recent_operating_income_growth if x is not None]) if any(
                x is not None for x in recent_operating_income_growth) else 0
            recent_avg_ni_growth = sum([x for x in recent_net_income_growth if x is not None]) / len(
                [x for x in recent_net_income_growth if x is not None]) if any(
                x is not None for x in recent_net_income_growth) else 0

            # 성장성 비교 차트
            comparison_data = {
                "지표": ["매출 성장률", "영업이익 성장률", "순이익 성장률"],
                "기업": [recent_avg_revenue_growth, recent_avg_oi_growth, recent_avg_ni_growth],
                "산업 평균": [industry_avg["revenue_growth"], industry_avg["operating_income_growth"],
                          industry_avg["net_income_growth"]]
            }

            import pandas as pd
            import altair as alt

            comparison_df = pd.DataFrame(comparison_data)
            comparison_df_melted = pd.melt(comparison_df, id_vars=["지표"], var_name="구분", value_name="성장률")

            chart = alt.Chart(comparison_df_melted).mark_bar().encode(
                x=alt.X('지표:N', title=None),
                y=alt.Y('성장률:Q', title='성장률 (%)'),
                color=alt.Color('구분:N', scale=alt.Scale(domain=['기업', '산업 평균'], range=['#4169E1', '#2E8B57'])),
                column=alt.Column('구분:N', title=None)
            ).properties(
                width=300,
                height=300
            ).configure_axisX(
                labelAngle=0
            )

            st.altair_chart(chart, use_container_width=True)

            # 성장 지속성 평가
            st.write("**성장 지속성 평가**")

            # 성장률 변동성 계산
            def calc_volatility(growth_values):
                """성장률의 변동성 계산"""
                if not growth_values or all(x is None for x in growth_values):
                    return float('inf')

                valid_values = [x for x in growth_values if x is not None]
                if len(valid_values) < 2:
                    return float('inf')

                import numpy as np
                return np.std(valid_values)

            rev_volatility = calc_volatility(revenue_growth[1:])  # 첫 해 제외
            oi_volatility = calc_volatility(operating_income_growth[1:])
            ni_volatility = calc_volatility(net_income_growth[1:])

            volatility_data = {
                "지표": ["매출", "영업이익", "순이익"],
                "변동성": [rev_volatility, oi_volatility, ni_volatility]
            }

            vol_df = pd.DataFrame(volatility_data)

            # 변동성이 유효한 경우에만 표시
            valid_vol = [v for v in [rev_volatility, oi_volatility, ni_volatility] if v != float('inf')]
            if valid_vol:
                avg_volatility = sum(valid_vol) / len(valid_vol)

                # 변동성 해석
                if avg_volatility < 5:
                    st.success("매우 안정적인 성장 패턴을 보이고 있습니다. 지속 가능한 성장이 기대됩니다.")
                elif avg_volatility < 15:
                    st.info("일반적인 수준의 성장 변동성을 보이고 있습니다. 안정적인 사업 운영이 이루어지고 있습니다.")
                else:
                    st.warning("높은 성장 변동성을 보이고 있습니다. 성장이 불안정할 수 있으며 주의가 필요합니다.")
        else:
            st.warning("충분한 데이터가 없어 종합 평가를 진행할 수 없습니다.")