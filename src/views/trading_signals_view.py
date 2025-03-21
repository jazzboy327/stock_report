# src/views/trading_signals_view.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import logging
from datetime import datetime, timedelta

from src.utils.optimal_trading_analyzer import OptimalTradingAnalyzer

logger = logging.getLogger('StockAnalysisApp.TradingSignalsView')


class TradingSignalsView:
    """매매 신호 탭 뷰를 담당하는 클래스"""

    def __init__(self):
        """뷰 초기화"""
        self.analyzer = OptimalTradingAnalyzer()

        # 최적 매매 시점 분석에 필요한 최소 데이터 수 정의
        self.MIN_REQUIRED_DATA = 200  # 200일치 데이터가 최소 권장
        self.ABSOLUTE_MIN_DATA = 60  # 60일치는 최소 필요

    def display(self, ticker, stock_data_df=None):
        """매매 신호 탭 표시"""
        st.header("최적 매매 시점 분석")

        if stock_data_df is None:
            # 데이터가 제공되지 않은 경우 직접 가져오기 (기간 연장)
            try:
                import yfinance as yf
                from datetime import datetime, timedelta

                end_date = datetime.today()
                # 기존 1년에서 2년으로 연장
                start_date = end_date - timedelta(days=730)  # 2년치 데이터

                st.info(f"{ticker} 데이터를 가져오는 중...")
                stock_data = yf.download(ticker, start=start_date, end=end_date)

                if stock_data.empty:
                    st.error(f"'{ticker}' 심볼에 대한 데이터를 가져올 수 없습니다.")
                    return

                st.success(f"{len(stock_data)}개 데이터 포인트를 가져왔습니다.")
                stock_data_df = stock_data
            except Exception as e:
                st.error(f"데이터 로드 중 오류 발생: {str(e)}")
                return

        # 데이터 충분성 확인
        if len(stock_data_df) < 20:  # 최소 20일 필요
            st.error(f"데이터가 너무 부족합니다. 최소 20일 이상의 데이터가 필요합니다. (현재: {len(stock_data_df)}일)")
            
            # 기본 주가 차트 표시
            if not stock_data_df.empty:
                self._display_basic_price_chart(stock_data_df)
                
            # 종합 리포트를 위한 기본 데이터 생성
            default_signals = {
                "recommendation": "데이터 부족으로 분석 불가",
                "current_buy_strength": 0,
                "current_sell_strength": 0,
                "latest_buy": [{"날짜": "N/A", "근거": "데이터 부족"}],
                "latest_sell": [{"날짜": "N/A", "근거": "데이터 부족"}],
                "signal_heatmap": {
                    "SMA": 0, "MACD": 0, "RSI": 0, 
                    "BB": 0, "Stoch": 0, "ADX": 0
                },
                "data_quality": {
                    "available_days": len(stock_data_df),
                    "required_days": 30,  # 최소 필요 일수
                    "sufficient": False
                }
            }
            
            # 세션 상태에 저장
            st.session_state.trading_signals = default_signals
            
            # 종합리포트에 데이터 등록
            try:
                from src.views.comprehensive_report_view import ComprehensiveReportView
                comprehensive_view = ComprehensiveReportView()
                comprehensive_view.register_analysis_result('trading_signals', default_signals)
            except Exception as e:
                logger.warning(f"종합리포트에 기본 매매신호 데이터 등록 실패: {e}")
                
            return

        # 분석 기간 설정
        period_options = {
            "1개월": 30,
            "3개월": 90,
            "6개월": 180,
            "1년": 365,
            "모든 데이터": len(stock_data_df)
        }

        col1, col2 = st.columns([2, 3])

        with col1:
            selected_period = st.selectbox(
                "분석 기간 선택",
                options=list(period_options.keys()),
                index=min(3, len(period_options) - 1)  # 기본값은 1년 또는 가능한 최대 기간
            )

            window_size = min(period_options[selected_period], len(stock_data_df))

            sensitivity = st.slider(
                "신호 민감도",
                min_value=1,
                max_value=5,
                value=3,
                help="값이 높을수록 더 많은 매매 신호가 생성됩니다"
            )

        with col2:
            st.write("**분석에 사용되는 기술적 지표:**")
            st.markdown("""
            - **이동평균선 (SMA/EMA)**: 추세 확인
            - **MACD**: 모멘텀과 추세 전환 감지
            - **RSI**: 과매수/과매도 판단
            - **볼린저 밴드**: 가격 변동성과 이상치 감지
            - **스토캐스틱**: 가격 모멘텀과 반전 신호
            - **ADX**: 추세 강도 측정
            """)

        # 로딩 스피너 표시
        with st.spinner("기술적 지표 계산 중..."):
            try:
                # 분석 실행
                analyzed_data, performance = self.analyzer.analyze(stock_data_df)

                # 세션 상태에 매매 신호 결과 저장
                st.session_state.trading_signals = performance

                # 종합리포트에 데이터 등록
                try:
                    from src.views.comprehensive_report_view import ComprehensiveReportView
                    comprehensive_view = ComprehensiveReportView()
                    comprehensive_view.register_analysis_result('trading_signals', performance)
                except Exception as e:
                    logger.warning(f"종합리포트에 매매신호 데이터 등록 실패: {e}")

                # 표시할 데이터 선택
                if window_size < len(analyzed_data):
                    display_data = analyzed_data.iloc[-window_size:].copy()
                else:
                    display_data = analyzed_data.copy()

                # 결과 시각화
                st.subheader("매매 신호 차트")
                fig = self.analyzer.visualize_signals(analyzed_data, window_size)
                st.plotly_chart(fig, use_container_width=True)

                # 현재 매매 추천
                self._display_recommendation(performance)

                # 최근 신호 표시
                self._display_recent_signals(display_data)

                # 기술적 지표 상세 분석
                self._display_technical_indicators(display_data)

                # 신호 통계
                self._display_signal_statistics(performance)
                
                # 만약 신뢰도 정보가 있다면 표시
                if 'data_confidence' in performance and 'data_confidence_level' in performance:
                    self._display_data_confidence(performance)

            except ValueError as ve:
                # 데이터 부족 오류 처리
                error_msg = str(ve)
                logger.error(f"매매 시점 분석 중 오류: {error_msg}")
                
                if '데이터 길이' in error_msg and '부족' in error_msg:
                    # 데이터 부족 시 친절한 안내 메시지
                    st.error("📊 데이터 부족으로 인해 최적 매매 시점 분석을 진행할 수 없습니다.")
                    
                    # 현재 데이터 길이와 필요한 최소 길이 정보 추출
                    import re
                    current_length = re.search(r'데이터 길이\((\d+)일\)', error_msg)
                    min_required = re.search(r'최소 길이\((\d+)일\)', error_msg)
                    
                    current_days = int(current_length.group(1)) if current_length else 0
                    required_days = int(min_required.group(1)) if min_required else 30
                    
                    # 정보와 해결 방법 안내
                    st.info(f"""
                    ### 데이터 부족 안내
                    
                    현재 데이터는 **{current_days}일**치로, 분석에 필요한 최소 **{required_days}일**보다 부족합니다.
                    
                    #### 원인
                    - 최근 상장한 기업일 경우 거래 기록이 짧을 수 있습니다.
                    - 기술적 지표 계산으로 인해 데이터 일부가 제외되었습니다.
                    
                    #### 해결 방법
                    1. 더 긴 기간의 데이터를 사용해보세요. (기본 수집 기간을 2년으로 확장했습니다)
                    2. 해당 기업의 거래 기록이 더 쌓일 때까지 기다려보세요.
                    3. 기술적 분석 대신 기본적 분석을 통해 투자 결정을 해보세요.
                    """)
                    
                    # 대체 정보 제공: 기본 주가 추이 차트 표시
                    self._display_basic_price_chart(stock_data_df)
                    
                    # 종합 리포트 등록용 기본 데이터 생성
                    default_signals = {
                        "recommendation": "데이터 부족으로 분석 불가",
                        "current_buy_strength": 0,
                        "current_sell_strength": 0,
                        "latest_buy": [{"날짜": "N/A", "근거": "데이터 부족"}],
                        "latest_sell": [{"날짜": "N/A", "근거": "데이터 부족"}],
                        "signal_heatmap": {
                            "SMA": 0, "MACD": 0, "RSI": 0, 
                            "BB": 0, "Stoch": 0, "ADX": 0
                        },
                        "data_quality": {
                            "available_days": current_days,
                            "required_days": required_days,
                            "sufficient": False,
                            "missing_percent": round((1 - current_days/required_days) * 100, 1)
                        }
                    }
                    
                    # 세션 상태에 저장
                    st.session_state.trading_signals = default_signals
                    
                    # 종합리포트에 데이터 등록
                    try:
                        from src.views.comprehensive_report_view import ComprehensiveReportView
                        comprehensive_view = ComprehensiveReportView()
                        comprehensive_view.register_analysis_result('trading_signals', default_signals)
                    except Exception as e:
                        logger.warning(f"종합리포트에 기본 매매신호 데이터 등록 실패: {e}")
                        
                else:
                    # 기타 ValueError 처리
                    st.error(f"분석 중 오류가 발생했습니다: {error_msg}")
                    
            except Exception as e:
                # 기타 모든 예외 처리
                logger.error(f"분석 중 오류 발생: {str(e)}", exc_info=True)
                st.error(f"분석 중 오류가 발생했습니다: {str(e)}")
                
                # 기본 차트 표시
                self._display_basic_price_chart(stock_data_df)
                
                # 오류 발생 시에도 종합 리포트를 위한 기본 데이터 생성
                default_signals = {
                    "recommendation": f"오류 발생: {str(e)[:50]}...",
                    "current_buy_strength": 0,
                    "current_sell_strength": 0,
                    "error": True,
                    "error_message": str(e),
                    "latest_buy": [{"날짜": "N/A", "근거": "오류 발생"}],
                    "latest_sell": [{"날짜": "N/A", "근거": "오류 발생"}],
                    "signal_heatmap": {
                        "SMA": 0, "MACD": 0, "RSI": 0, 
                        "BB": 0, "Stoch": 0, "ADX": 0
                    }
                }
                
                # 세션 상태에 저장
                st.session_state.trading_signals = default_signals
                
                # 종합리포트에 데이터 등록
                try:
                    from src.views.comprehensive_report_view import ComprehensiveReportView
                    comprehensive_view = ComprehensiveReportView()
                    comprehensive_view.register_analysis_result('trading_signals', default_signals)
                except Exception as reg_e:
                    logger.warning(f"종합리포트에 기본 매매신호 데이터 등록 실패: {reg_e}")

    def _display_basic_price_chart(self, stock_data_df):
        """데이터 부족 시 기본 주가 차트 표시"""
        st.subheader("기본 주가 추이")
        
        if stock_data_df is not None and not stock_data_df.empty:
            fig = go.Figure()
            fig.add_trace(go.Candlestick(
                x=stock_data_df.index,
                open=stock_data_df['Open'],
                high=stock_data_df['High'],
                low=stock_data_df['Low'],
                close=stock_data_df['Close'],
                name="주가",
                increasing_line_color='red',  # 한국식: 상승 빨간색
                decreasing_line_color='blue'  # 한국식: 하락 파란색
            ))
            
            # 이동평균선 추가 (가능한 경우)
            try:
                # 데이터 길이에 맞춰 이동평균 조정
                if len(stock_data_df) >= 20:
                    ma_length = 20
                elif len(stock_data_df) >= 10:
                    ma_length = 10
                elif len(stock_data_df) >= 5:
                    ma_length = 5
                else:
                    ma_length = None
                    
                if ma_length:
                    stock_data_df[f'MA{ma_length}'] = stock_data_df['Close'].rolling(window=ma_length).mean()
                    fig.add_trace(go.Scatter(
                        x=stock_data_df.index,
                        y=stock_data_df[f'MA{ma_length}'],
                        name=f"{ma_length}일 이동평균",
                        line=dict(color='purple', width=1)
                    ))
                    
                # 더 짧은 이동평균선 추가
                short_ma = max(3, len(stock_data_df) // 8)
                stock_data_df[f'MA{short_ma}'] = stock_data_df['Close'].rolling(window=short_ma).mean()
                fig.add_trace(go.Scatter(
                    x=stock_data_df.index,
                    y=stock_data_df[f'MA{short_ma}'],
                    name=f"{short_ma}일 이동평균",
                    line=dict(color='orange', width=1)
                ))
            except Exception as e:
                logger.warning(f"이동평균선 추가 실패: {str(e)}")
            
            fig.update_layout(
                title="기본 주가 차트 (데이터 부족으로 일부 분석 생략)",
                xaxis_title="날짜",
                yaxis_title="가격",
                xaxis_rangeslider_visible=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # 간단한 거래량 차트 추가
            if 'Volume' in stock_data_df.columns:
                volume_fig = go.Figure()
                volume_fig.add_trace(go.Bar(
                    x=stock_data_df.index, 
                    y=stock_data_df['Volume'],
                    name='거래량',
                    marker_color='darkgrey'
                ))
                
                volume_fig.update_layout(
                    title="거래량",
                    xaxis_title="날짜",
                    yaxis_title="거래량",
                    height=300
                )
                
                st.plotly_chart(volume_fig, use_container_width=True)
                
    def _display_data_confidence(self, performance):
        """데이터 신뢰도 정보 표시"""
        st.subheader("데이터 신뢰도 정보")
        
        confidence = performance.get('data_confidence', 0)
        confidence_level = performance.get('data_confidence_level', '알 수 없음')
        
        # 신뢰도에 따른 색상 설정
        if confidence_level == "높음":
            color = "green"
        elif confidence_level == "보통":
            color = "orange"
        else:
            color = "red"
            
        st.markdown(f"""
        <div style="padding:10px; border-radius:5px; background-color:rgba({','.join(['200', '200', '200', '0.2'])});">
            <h4>신뢰도 수준: <span style="color:{color};">{confidence_level}</span> ({confidence:.1f}%)</h4>
            <p>{performance.get('warning', '신뢰도 정보가 충분합니다.')}</p>
        </div>
        """, unsafe_allow_html=True)

    def _display_recommendation(self, performance):
        """현재 매매 추천 표시"""
        st.subheader("현재 매매 추천")

        recommendation = performance['recommendation']
        buy_strength = performance['current_buy_strength']
        sell_strength = performance['current_sell_strength']

        # 추천 색상 설정
        if recommendation in ["강력 매수", "매수"]:
            color = "green"
        elif recommendation in ["강력 매도", "매도"]:
            color = "red"
        else:
            color = "orange"

        col1, col2, col3 = st.columns([2, 1, 1])

        with col1:
            st.markdown(f"### <span style='color:{color};'>{recommendation}</span>", unsafe_allow_html=True)

        with col2:
            st.metric("매수 신호 강도", f"{buy_strength:.1f}%")

        with col3:
            st.metric("매도 신호 강도", f"{sell_strength:.1f}%")

        # 신호 히트맵 (기술적 지표별 매수/매도 신호)
        signal_heatmap = performance['signal_heatmap']

        st.write("**기술적 지표별 신호**")

        heatmap_cols = st.columns(6)
        indicators = ["SMA", "MACD", "RSI", "BB", "Stoch", "ADX"]

        for i, indicator in enumerate(indicators):
            with heatmap_cols[i]:
                signal = signal_heatmap.get(indicator, 0)
                if signal == 1:
                    st.markdown(
                        f"<div style='text-align:center; background-color:#d4f1cc; padding:10px; border-radius:5px;'><b>{indicator}</b><br>매수</div>",
                        unsafe_allow_html=True)
                else:
                    st.markdown(
                        f"<div style='text-align:center; background-color:#f1ccc9; padding:10px; border-radius:5px;'><b>{indicator}</b><br>매도</div>",
                        unsafe_allow_html=True)

    def _display_recent_signals(self, data):
        """최근 매매 신호 표시"""
        st.subheader("최근 매매 신호")

        # 신호가 있는 행만 필터링
        signals = data[data['Signal'] != 0].copy()

        if len(signals) == 0:
            st.info("선택한 기간 동안 매매 신호가 없습니다.")
            # 세션 상태 업데이트 - 빈 신호 정보 추가
            if 'trading_signals' in st.session_state:
                st.session_state.trading_signals["latest_buy"] = [{"날짜": "N/A", "근거": "N/A"}]
                st.session_state.trading_signals["latest_sell"] = [{"날짜": "N/A", "근거": "N/A"}]
            return

        # 최근 10개 신호만 표시
        recent_signals = signals.iloc[-10:].copy() if len(signals) > 10 else signals

        # 신호 데이터 준비
        signal_data = []
        buy_signals = []
        sell_signals = []

        for idx, row in recent_signals.iterrows():
            date = idx.strftime('%Y-%m-%d')
            signal_type = "매수" if row['Signal'] == 1 else "매도"
            signal_strength = row['Buy_Signal_Strength'] if row['Signal'] == 1 else row['Sell_Signal_Strength']
            price = row['Close']
            reason = row['Signal_Type']

            signal_info = {
                "날짜": date,
                "신호": signal_type,
                "가격": f"{price:,.0f}원",
                "강도": f"{signal_strength:.1f}%",
                "근거": reason
            }

            signal_data.append(signal_info)

            # 매수/매도 신호 구분하여 저장
            if signal_type == "매수":
                buy_signals.append(signal_info)
            else:
                sell_signals.append(signal_info)

        # 세션 상태 업데이트 - 최신 매수/매도 신호 저장
        if 'trading_signals' in st.session_state:
            if buy_signals:
                st.session_state.trading_signals["latest_buy"] = buy_signals[-1:]  # 가장 최근 매수 신호
            else:
                st.session_state.trading_signals["latest_buy"] = [{"날짜": "N/A", "근거": "N/A"}]

            if sell_signals:
                st.session_state.trading_signals["latest_sell"] = sell_signals[-1:]  # 가장 최근 매도 신호
            else:
                st.session_state.trading_signals["latest_sell"] = [{"날짜": "N/A", "근거": "N/A"}]

        # 데이터프레임으로 변환 후 표시
        import pandas as pd
        signal_df = pd.DataFrame(signal_data[::-1])  # 역순으로 정렬 (최신순)

        # 신호 유형에 따라 행 색상 지정
        def highlight_signals(row):
            if row['신호'] == '매수':
                return ['background-color: #d4f1cc'] * len(row)
            else:
                return ['background-color: #f1ccc9'] * len(row)

        st.dataframe(signal_df.style.apply(highlight_signals, axis=1), use_container_width=True)

    def _display_technical_indicators(self, data):
        """기술적 지표 상세 분석"""
        st.subheader("기술적 지표 상세 분석")

        # 가장 최근 데이터
        latest = data.iloc[-1]

        # 탭으로 구분하여 표시
        tabs = st.tabs(["이동평균선", "MACD", "RSI & 스토캐스틱", "볼린저 밴드", "ADX"])

        # 1. 이동평균선 탭
        with tabs[0]:
            col1, col2 = st.columns(2)

            with col1:
                st.write("**이동평균선 (SMA)**")
                st.metric("SMA20", f"{latest['SMA20']:.2f}")
                st.metric("SMA50", f"{latest['SMA50']:.2f}")
                st.metric("SMA200", f"{latest['SMA200']:.2f}")

                # SMA 교차 상태
                if latest['SMA20'] > latest['SMA50']:
                    st.success("SMA20이 SMA50 위에 있습니다 (상승 추세)")
                else:
                    st.error("SMA20이 SMA50 아래에 있습니다 (하락 추세)")

                if latest['SMA50'] > latest['SMA200']:
                    st.success("SMA50이 SMA200 위에 있습니다 (장기 상승 추세)")
                else:
                    st.error("SMA50이 SMA200 아래에 있습니다 (장기 하락 추세)")

            with col2:
                st.write("**지수이동평균선 (EMA)**")
                st.metric("EMA9", f"{latest['EMA9']:.2f}")
                st.metric("EMA21", f"{latest['EMA21']:.2f}")

                # EMA 교차 상태
                if latest['EMA9'] > latest['EMA21']:
                    st.success("EMA9이 EMA21 위에 있습니다 (단기 상승 추세)")
                else:
                    st.error("EMA9이 EMA21 아래에 있습니다 (단기 하락 추세)")

                # 가격과 EMA 관계
                if latest['Close'] > latest['EMA9']:
                    st.success("현재가가 EMA9 위에 있습니다 (강한 단기 상승)")
                else:
                    st.error("현재가가 EMA9 아래에 있습니다 (약한 단기 모멘텀)")

        # 2. MACD 탭
        with tabs[1]:
            col1, col2 = st.columns(2)

            with col1:
                st.write("**MACD 지표 값**")
                st.metric("MACD", f"{latest['MACD_12_26_9']:.2f}")
                st.metric("Signal", f"{latest['MACDs_12_26_9']:.2f}")
                st.metric("Histogram", f"{latest['MACDh_12_26_9']:.2f}")

            with col2:
                st.write("**MACD 분석**")

                # MACD 신호 해석
                if latest['MACD_12_26_9'] > latest['MACDs_12_26_9']:
                    st.success("MACD가 Signal선 위에 있습니다 (상승 추세)")
                else:
                    st.error("MACD가 Signal선 아래에 있습니다 (하락 추세)")

                # 히스토그램 해석
                if latest['MACDh_12_26_9'] > 0:
                    st.success(f"MACD 히스토그램이 양수입니다 ({latest['MACDh_12_26_9']:.2f})")
                else:
                    st.error(f"MACD 히스토그램이 음수입니다 ({latest['MACDh_12_26_9']:.2f})")

                # 히스토그램 방향 해석
                if latest['MACDh_12_26_9'] > data['MACDh_12_26_9'].iloc[-2]:
                    st.success("히스토그램이 증가하고 있습니다 (모멘텀 상승)")
                else:
                    st.error("히스토그램이 감소하고 있습니다 (모멘텀 하락)")

        # 3. RSI & 스토캐스틱 탭
        with tabs[2]:
            col1, col2 = st.columns(2)

            with col1:
                st.write("**RSI (14)**")

                # RSI 값
                rsi_value = latest['RSI14']
                st.metric("RSI", f"{rsi_value:.2f}")

                # RSI 상태 해석
                if rsi_value > 70:
                    st.error("과매수 상태 (매도 고려)")
                elif rsi_value < 30:
                    st.success("과매도 상태 (매수 고려)")
                else:
                    st.info("중립 구간")

                # RSI 방향
                if rsi_value > data['RSI14'].iloc[-2]:
                    st.success("RSI 상승 중")
                else:
                    st.error("RSI 하락 중")

            with col2:
                st.write("**스토캐스틱 (14,3,3)**")

                # 스토캐스틱 값
                k_value = latest['STOCHk_14_3_3']
                d_value = latest['STOCHd_14_3_3']

                st.metric("%K", f"{k_value:.2f}")
                st.metric("%D", f"{d_value:.2f}")

                # 스토캐스틱 상태 해석
                if k_value > 80 and d_value > 80:
                    st.error("과매수 상태 (매도 고려)")
                elif k_value < 20 and d_value < 20:
                    st.success("과매도 상태 (매수 고려)")
                else:
                    st.info("중립 구간")

                # 스토캐스틱 교차
                if k_value > d_value:
                    st.success("%K가 %D 위에 있습니다 (상승 신호)")
                else:
                    st.error("%K가 %D 아래에 있습니다 (하락 신호)")

        # 4. 볼린저 밴드 탭
        with tabs[3]:
            col1, col2 = st.columns(2)

            with col1:
                st.write("**볼린저 밴드 (20, 2)**")

                # 볼린저 밴드 값
                upper = latest['BBU_20_2.0']
                middle = latest['BBM_20_2.0']
                lower = latest['BBL_20_2.0']

                st.metric("상단 밴드", f"{upper:.2f}")
                st.metric("중심선 (SMA20)", f"{middle:.2f}")
                st.metric("하단 밴드", f"{lower:.2f}")

                # 밴드폭
                bandwidth = (upper - lower) / middle * 100
                st.metric("밴드폭", f"{bandwidth:.2f}%")

            with col2:
                st.write("**볼린저 밴드 분석**")

                # 현재가 위치
                close = latest['Close']

                if close > upper:
                    st.error(f"상단 밴드 돌파 (과매수): {close:.2f} > {upper:.2f}")
                elif close < lower:
                    st.success(f"하단 밴드 돌파 (과매도): {close:.2f} < {lower:.2f}")
                else:
                    # 밴드 내에서의 위치
                    position = (close - lower) / (upper - lower) * 100
                    st.info(f"밴드 내 위치: {position:.1f}% (0%: 하단, 100%: 상단)")

                # 추가 분석
                if bandwidth > 5:
                    st.success(f"밴드폭이 넓음 ({bandwidth:.2f}%): 높은 변동성")
                else:
                    st.warning(f"밴드폭이 좁음 ({bandwidth:.2f}%): 낮은 변동성, 곧 큰 움직임 가능성")

                # 추세 분석
                if middle > data['BBM_20_2.0'].iloc[-10:].mean():
                    st.success("중심선 상승 중 (상승 추세)")
                else:
                    st.error("중심선 하락 중 (하락 추세)")

        # 5. ADX 탭
        with tabs[4]:
            col1, col2 = st.columns(2)

            with col1:
                st.write("**ADX (14)**")

                # ADX 값
                adx_value = latest['ADX_14']
                pdi_value = latest['DMP_14']
                ndi_value = latest['DMN_14']

                st.metric("ADX", f"{adx_value:.2f}")
                st.metric("+DI", f"{pdi_value:.2f}")
                st.metric("-DI", f"{ndi_value:.2f}")

            with col2:
                st.write("**ADX 분석**")

                # ADX 강도 해석
                if adx_value >= 25:
                    if adx_value >= 50:
                        st.success(f"매우 강한 추세 (ADX: {adx_value:.2f})")
                    else:
                        st.success(f"강한 추세 (ADX: {adx_value:.2f})")
                else:
                    st.warning(f"약한 추세 또는 횡보 (ADX: {adx_value:.2f})")

                # 추세 방향
                if pdi_value > ndi_value:
                    st.success(f"상승 추세 (+DI > -DI): {pdi_value:.2f} > {ndi_value:.2f}")
                else:
                    st.error(f"하락 추세 (+DI < -DI): {pdi_value:.2f} < {ndi_value:.2f}")

                # ADX 방향
                if adx_value > data['ADX_14'].iloc[-2]:
                    st.success("ADX 상승 중 (추세 강화)")
                else:
                    st.error("ADX 하락 중 (추세 약화)")

    def _display_signal_statistics(self, performance):
        """매매 신호 통계 표시"""
        st.subheader("매매 신호 통계")

        col1, col2 = st.columns(2)

        with col1:
            st.write("**신호 발생 횟수**")

            # 신호 횟수 차트
            chart_data = {
                "신호 유형": ["매수 신호", "매도 신호", "강한 매수", "강한 매도"],
                "횟수": [
                    performance['buy_signals'],
                    performance['sell_signals'],
                    performance['strong_buy_signals'],
                    performance['strong_sell_signals']
                ]
            }

            import pandas as pd
            chart_df = pd.DataFrame(chart_data)

            st.bar_chart(chart_df.set_index("신호 유형"), use_container_width=True)

        with col2:
            st.write("**신호 유형별 분포**")

            # 신호 유형 분포
            signal_types = performance['signal_types']

            # 가장 많은 신호 유형 표시
            if signal_types:
                most_common = max(signal_types.items(), key=lambda x: x[1])
                st.info(f"가장 많이 발생한 신호 유형: **{most_common[0]}** ({most_common[1]}회)")

                # 유형별 횟수 표시
                for signal_type, count in sorted(signal_types.items(), key=lambda x: x[1], reverse=True):
                    st.write(f"- {signal_type}: {count}회")
            else:
                st.info("아직 신호 유형 데이터가 없습니다.")

        # 데이터 신뢰도 표시 (추가된 부분)
        if 'data_confidence' in performance and 'data_confidence_level' in performance:
            st.subheader("데이터 신뢰도")

            confidence = performance['data_confidence']
            confidence_level = performance['data_confidence_level']
            data_length = performance.get('data_length', 0)

            # 신뢰도에 따른 색상 설정
            if confidence_level == "높음":
                color = "green"
            elif confidence_level == "보통":
                color = "orange"
            else:
                color = "red"

            # 신뢰도 메시지
            st.markdown(f"""
            <div style="padding:10px; border-radius:5px; background-color:rgba({','.join(['200', '200', '200', '0.2'])});">
                <h4>신뢰도 수준: <span style="color:{color};">{confidence_level}</span> ({confidence:.1f}%)</h4>
                <p>사용된 데이터: {data_length}일 / 권장 데이터: {self.MIN_REQUIRED_DATA}일</p>
                <p><i>참고: 신뢰도는 사용 가능한 데이터 양에 기반합니다. 데이터가 많을수록 분석의 정확도가 높아집니다.</i></p>
            </div>
            """, unsafe_allow_html=True)