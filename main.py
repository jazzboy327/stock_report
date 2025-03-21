# main.py

import streamlit as st
import logging
import traceback
import asyncio
import time
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import yfinance as yf
import os
import numpy as np
from dotenv import load_dotenv

# .env 파일에서 환경 변수 로드
load_dotenv()

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('StockAnalysisApp')

# 한국 주식 종목 매퍼 클래스 가져오기
from src.utils.korean_stock_symbol_mapper import KoreanStockSymbolMapper
from src.models.stock_info import StockInfo, StockController
from src.utils.stock_data_collector import StockDataCollector
from src.utils.financial_analysis import get_financial_statements, get_dividends
from src.views.financial_analysis_view import FinancialAnalysisView
from src.views.trading_signals_view import TradingSignalsView
from src.views.comprehensive_report_view import ComprehensiveReportView

# 로깅 오류 안전 처리를 위한 핸들러
class SafeLogHandler(logging.Handler):
    """로깅 오류를 안전하게 처리하는 핸들러"""

    def emit(self, record):
        try:
            msg = self.format(record)
            # 안전하게 로그 처리
            print(msg)
        except Exception as e:
            # 로깅 오류 자체를 무시하고 대체 로깅
            fallback_msg = f"로깅 오류 발생: {str(e)}"
            try:
                print(fallback_msg)
            except:
                pass  # 최후의 방어선

# 기존 logging 설정 코드 아래에 추가
def setup_safe_logging():
    """pykrx 라이브러리의 로깅 오류를 방지하기 위한 설정"""
    # pykrx 로깅 설정
    pykrx_logger = logging.getLogger('pykrx')
    pykrx_logger.setLevel(logging.WARNING)  # 필요한 로그 레벨로 조정

    # 기존 핸들러 제거 (선택적)
    for handler in pykrx_logger.handlers[:]:
        pykrx_logger.removeHandler(handler)

    # 안전한 핸들러 추가
    safe_handler = SafeLogHandler()
    safe_handler.setLevel(logging.WARNING)
    safe_handler.setFormatter(logging.Formatter('PYKRX: %(levelname)s - %(message)s'))
    pykrx_logger.addHandler(safe_handler)

    # root 로거에도 안전 핸들러 추가 (선택적)
    root_logger = logging.getLogger()
    safe_root_handler = SafeLogHandler()
    safe_root_handler.setLevel(logging.WARNING)
    safe_root_handler.setFormatter(logging.Formatter('%(asctime)s - ROOT: %(levelname)s - %(message)s'))
    root_logger.addHandler(safe_root_handler)

class StockAnalysisApp:
    """주식 분석 애플리케이션"""

    def __init__(self):
        """애플리케이션 초기화"""
        self.symbol_mapper = KoreanStockSymbolMapper()
        self.stock_controller = StockController()
        self.data_collector = StockDataCollector()

        # 안전 로깅 설정 추가
        setup_safe_logging()

        # 데이터 통합 매니저 초기화
        from src.utils.data_integration import DataIntegrationManager
        self.data_integration_manager = DataIntegrationManager()

        # 각 뷰 초기화
        self.financial_analysis_view = FinancialAnalysisView()
        self.trading_signals_view = TradingSignalsView()
        self.comprehensive_report_view = ComprehensiveReportView()

        # 이벤트 핸들러 설정
        from src.views.common_event_handlers import AnalysisEventHandler
        self.analysis_event_handler = AnalysisEventHandler()

        # 데이터 통합 이벤트 리스너 등록
        self.data_integration_manager.add_data_listener(self._on_data_updated)

        # 세션 상태 초기화
        if 'selected_company' not in st.session_state:
            st.session_state.selected_company = None
        if 'stock_info' not in st.session_state:
            st.session_state.stock_info = None
        if 'market_data' not in st.session_state:
            st.session_state.market_data = None
        if 'analysis_results' not in st.session_state:
            st.session_state.analysis_results = None
        if 'error_message' not in st.session_state:
            st.session_state.error_message = None
        # 현재 분석 중인 기업 코드 저장
        if 'current_analyzed_symbol' not in st.session_state:
            st.session_state.current_analyzed_symbol = None

    def _on_data_updated(self, category, data):
        """데이터 업데이트 이벤트 핸들러"""
        # 특정 카테고리의 데이터가 업데이트되면 관련 처리 수행
        logger.info(f"데이터 업데이트 감지: {category}")

        # 종합 리포트 갱신 여부 확인
        if category in ['technical_analysis', 'trading_signals', 'financial_analysis.risk_metrics',
                        'prediction_result']:
            # 종합 리포트 뷰에 데이터 변경 알림
            if hasattr(self, 'comprehensive_report_view'):
                try:
                    # 여기서는 단순히 로그만 남기지만, 필요시 추가 작업 수행 가능
                    logger.info(f"종합 리포트에 데이터 변경 알림: {category}")
                except Exception as e:
                    logger.error(f"데이터 변경 알림 처리 중 오류: {str(e)}", exc_info=True)

    def setup_page(self):
        """페이지 설정"""
        st.set_page_config(
            page_title="주식 분석 시스템",
            layout="wide"
        )
        st.sidebar.title("옵션 설정")

        # 앱 상태 초기화
        if 'current_tab_index' not in st.session_state:
            st.session_state.current_tab_index = 0

        # 종합 리포트 탭 플래그 초기화
        if 'in_comprehensive_tab' not in st.session_state:
            st.session_state.in_comprehensive_tab = False

    def get_analysis_period(self):
        """5년치 분석 기간을 반환"""
        end_date = datetime.today()
        start_date = end_date - timedelta(days=5 * 365)
        return start_date, end_date

    def render_search_section(self):
        """검색 섹션 렌더링"""
        st.sidebar.header("🔍 기업 검색")

        # 검색어 입력
        search_query = st.sidebar.text_input("기업명 입력", placeholder="예: 삼성전자")

        # 검색 기록 추적
        if 'last_search_query' not in st.session_state:
            st.session_state.last_search_query = None

        # 검색어가 변경된 경우 예측 상태 초기화
        if search_query != st.session_state.last_search_query:
            st.session_state.last_search_query = search_query
            if 'prediction_status' in st.session_state:
                st.session_state.prediction_status = 'ready'
            if 'prediction_result' in st.session_state:
                del st.session_state.prediction_result
            logger.info(f"검색어 변경: {search_query}, 예측 상태 초기화")

        if search_query:
            # 유사한 기업 검색
            similar_companies = self.symbol_mapper.search_companies(search_query)

            if not similar_companies:
                st.sidebar.warning(f"'{search_query}'와(과) 일치하는 기업을 찾을 수 없습니다.")
                return None

            # 검색 결과 표시
            st.sidebar.subheader("검색 결과")

            # 선택 추적
            if 'last_selected_option' not in st.session_state:
                st.session_state.last_selected_option = None

            # 기업 선택 옵션 생성
            company_options = [f"{company} ({market})" for company, symbol, market in similar_companies]
            selected_option = st.sidebar.selectbox("기업 선택", company_options)

            # 선택된 옵션이 변경된 경우 예측 상태 초기화
            if selected_option != st.session_state.last_selected_option:
                st.session_state.last_selected_option = selected_option
                if 'prediction_status' in st.session_state:
                    st.session_state.prediction_status = 'ready'
                if 'prediction_result' in st.session_state:
                    del st.session_state.prediction_result
                logger.info(f"기업 선택 변경: {selected_option}, 예측 상태 초기화")

            if selected_option:
                # 선택된 기업 정보 추출
                selected_idx = company_options.index(selected_option)
                selected_company, selected_symbol, selected_market = similar_companies[selected_idx]

                # 선택된 기업 정보 표시
                st.sidebar.info(f"선택된 기업: {selected_company}\n종목코드: {selected_symbol}\n시장: {selected_market}")

                # 분석 버튼
                if st.sidebar.button("주식 분석 시작", key="analyze_button"):
                    # 새로운 기업이 선택되었는지 확인
                    is_new_company = (st.session_state.current_analyzed_symbol != selected_symbol)

                    # 새로운 기업이 선택된 경우 세션 상태 초기화
                    if is_new_company:
                        logger.info(
                            f"새 기업 선택됨: {selected_company} ({selected_symbol}), 이전: {st.session_state.current_analyzed_symbol}")

                        # 예측 상태 초기화
                        st.session_state.prediction_status = 'ready'

                        # 예측 결과 제거
                        if 'prediction_result' in st.session_state:
                            del st.session_state.prediction_result

                        # 현재 예측 심볼도 함께 업데이트
                        if 'current_prediction_symbol' in st.session_state:
                            st.session_state.current_prediction_symbol = selected_symbol

                        # 히스토리 데이터 초기화
                        if 'history_data' in st.session_state:
                            del st.session_state.history_data

                        # 현재 분석 중인 심볼 업데이트
                        st.session_state.current_analyzed_symbol = selected_symbol

                    return {
                        'name': selected_company,
                        'symbol': selected_symbol,
                        'market': selected_market
                    }

            return None

    async def analyze_stock(self, company_info):
        """주식 분석 수행"""
        try:
            # 상태 표시
            status_container = st.empty()
            progress_bar = st.progress(0)
            status_container.info("💡 분석을 시작합니다...")
            progress_bar.progress(10)

            symbol = company_info['symbol']
            company_name = company_info['name']

            # 데이터 수집 및 분석 진행
            status_container.info("📊 주식 데이터를 수집하고 있습니다...")
            progress_bar.progress(30)

            # StockDataCollector를 통해 모든 데이터 수집
            stock_info, market_data, analysis_results = await self.data_collector.collect_all_data(symbol)

            progress_bar.progress(90)

            # 결과 저장
            st.session_state.selected_company = company_info
            st.session_state.stock_info = stock_info
            st.session_state.market_data = market_data
            st.session_state.analysis_results = analysis_results
            st.session_state.current_analyzed_symbol = symbol

            # 현재 예측 심볼도 함께 업데이트 (일관성 유지)
            st.session_state.current_prediction_symbol = symbol

            # 분석 시작 플래그 설정
            st.session_state.analysis_started = True

            # 이전 예측 관련 세션 상태 초기화
            if 'prediction_status' in st.session_state:
                st.session_state.prediction_status = 'ready'
            if 'prediction_result' in st.session_state:
                del st.session_state.prediction_result

            progress_bar.progress(100)
            status_container.success("✅ 분석이 완료되었습니다!")
            time.sleep(1)
            status_container.empty()
            progress_bar.empty()

            return True

        except Exception as e:
            logger.error(f"분석 중 오류 발생: {str(e)}")
            logger.error(traceback.format_exc())
            st.error(f"분석 중 오류가 발생했습니다: {str(e)}")
            st.session_state.error_message = str(e)
            return False

    def display_results(self):
        """분석 결과 표시"""
        if not st.session_state.selected_company or not st.session_state.market_data:
            return

        company_info = st.session_state.selected_company
        stock_info = st.session_state.stock_info
        market_data = st.session_state.market_data
        analysis_results = st.session_state.analysis_results

        # 기업명과 심볼 표시
        st.title(f"{company_info['name']} ({company_info['symbol']})")

        # 탭 생성
        tabs = st.tabs(["📊 주식 상세 정보", "📈 기술적 분석", "👥 투자자 동향", "💰 위험지표, 제무지표 분석", "🎯 최적 매매 시점", "📑 AI 예측 리포트"])

        # 현재 선택된 탭 인덱스 저장 (Streamlit이 자동으로 처리)
        # 참고: Streamlit에서는 탭 클릭에 따라 각 탭 내부의 코드가 자동으로 실행됨

        # 주식 상세 정보 탭
        with tabs[0]:
            self.display_stock_detail_tab(company_info, stock_info, market_data)
            # 주식 상세 데이터 등록 (종합리포트용)
            stock_detail_data = {
                "current_price": market_data['close'][-1] if market_data and len(
                    market_data.get('close', [])) > 0 else 0,
                "price_change": ((market_data['close'][-1] - market_data['close'][-2]) / market_data['close'][-2]) * 100
                if market_data and len(market_data.get('close', [])) >= 2 else 0,
                "volume": market_data['volume'][-1] if market_data and len(market_data.get('volume', [])) > 0 else 0,
                "market_cap": getattr(stock_info, 'market_cap', 0) or 0
            }
            self.comprehensive_report_view.register_analysis_result('stock_detail', stock_detail_data)

        # 기술적 분석 탭
        with tabs[1]:
            self.display_technical_analysis_tab(company_info, market_data, analysis_results)
            # 기술적 분석 데이터 등록 (종합리포트용)
            technical_analysis_data = {
                "trend": analysis_results.get('trend', 'N/A'),
                "ma5": analysis_results.get('ma5', 0),
                "ma20": analysis_results.get('ma20', 0),
                "rsi": analysis_results.get('rsi', 0),
                "rsi_status": analysis_results.get('rsi_status', 'N/A'),
                "volume_trend": analysis_results.get('volume_trend', 'N/A')
            }
            self.comprehensive_report_view.register_analysis_result('technical_analysis', technical_analysis_data)

        # 투자자 동향 탭
        with tabs[2]:
            self.display_investor_tab(market_data, analysis_results)
            # 투자자 동향 데이터 등록 (종합리포트용)
            investor_trends_data = {
                "main_buyer": analysis_results.get('main_buyer', 'N/A'),
                "main_seller": analysis_results.get('main_seller', 'N/A'),
                "investor_ratio": market_data.get('investor_ratio', [0, 0, 0, 0])
            }
            self.comprehensive_report_view.register_analysis_result('investor_trends', investor_trends_data)

        # 재무 분석 탭
        with tabs[3]:
            self.display_financial_analysis_tab(company_info)

        # 최적 매매 시점 탭
        with tabs[4]:
            history_df = self.get_history_data(company_info['symbol'])
            self.trading_signals_view.display(company_info['symbol'], history_df)

            # 추가: 매매 신호 데이터 등록
            if 'trading_signals' in st.session_state:
                self.comprehensive_report_view.register_analysis_result('trading_signals',
                                                                        st.session_state.trading_signals)
        # 종합 리포트 탭
        with tabs[5]:
            # 종합 리포트 탭 활성화 시 특수 플래그 설정
            st.session_state.in_comprehensive_tab = True

            history_df = self.get_history_data(company_info['symbol'])
            self.comprehensive_report_view.display(company_info, stock_info, market_data, analysis_results, history_df)

            # 플래그 초기화
            st.session_state.in_comprehensive_tab = False

    def get_history_data(self, symbol):
        """히스토리 데이터 가져오기 (캐싱 적용)"""
        try:
            # 기업이 변경된 경우 캐시 초기화
            if st.session_state.current_analyzed_symbol != symbol and 'history_data' in st.session_state:
                logger.info(f"기업 변경으로 히스토리 데이터 캐시 초기화: {symbol}")
                del st.session_state.history_data

            # 캐시된 데이터 확인
            if 'history_data' in st.session_state:
                logger.info(f"캐시된 히스토리 데이터 사용: {symbol}")
                return st.session_state.history_data

            # 데이터 수집
            logger.info(f"새로운 히스토리 데이터 로드 중: {symbol}")
            history_df = self.data_collector.get_stock_history_sync(symbol)
            st.session_state.history_data = history_df
            return history_df

        except Exception as e:
            st.error(f"주식 데이터 로드 실패: {str(e)}")
            logger.error(f"히스토리 데이터 로드 실패: {str(e)}", exc_info=True)
            return None

    def display_stock_detail_tab(self, company_info, stock_info, market_data):
        """주식 상세 정보 탭 표시"""
        # 회사 정보 헤더
        st.header("주식 상세 정보")

        # 현재가 등락률 계산
        if market_data and len(market_data.get('close', [])) >= 2:
            current_price = market_data['close'][-1]
            prev_price = market_data['close'][-2]
            price_change = ((current_price - prev_price) / prev_price) * 100
        else:
            current_price = 0
            price_change = 0

        # 기본 지표 표시
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "현재가",
                f"{current_price:,.0f}원",
                f"{price_change:+.2f}%"
            )

        with col2:
            volume = market_data['volume'][-1] if market_data and len(market_data.get('volume', [])) > 0 else 0
            volume_avg = np.mean(market_data['volume'][-5:]) if market_data and len(
                market_data.get('volume', [])) >= 5 else 0
            volume_change = ((volume - volume_avg) / volume_avg * 100) if volume_avg > 0 else 0

            st.metric(
                "거래량",
                f"{volume / 10000:,.0f}만주",
                f"{volume_change:+.2f}%" if volume_avg > 0 else None
            )

        with col3:
            market_cap = getattr(stock_info, 'market_cap', 0) or 0
            if market_cap >= 1e12:  # 1조 이상
                market_cap_str = f"{market_cap / 1e12:.2f}조원"
            elif market_cap >= 1e8:  # 1억 이상
                market_cap_str = f"{market_cap / 1e8:.2f}억원"
            else:
                market_cap_str = f"{market_cap:,.0f}원"

            st.metric(
                "시가총액",
                market_cap_str
            )

        with col4:
            st.metric(
                "시장",
                company_info['market']
            )

        # 주가 해석 추가 (새로운 섹션)
        st.subheader("주가 분석 해석")

        # 해석을 위한 4개 컬럼 생성
        interp_col1, interp_col2, interp_col3, interp_col4 = st.columns(4)

        with interp_col1:
            st.markdown("#### 가격 동향")
            # 5일 이동평균과 20일 이동평균 비교
            if market_data and 'MA5' in market_data and 'MA20' in market_data and len(market_data['MA5']) > 0 and len(
                    market_data['MA20']) > 0:
                ma5 = market_data['MA5'][-1]
                ma20 = market_data['MA20'][-1]

                if current_price > ma5 > ma20:
                    st.success("**강한 상승세**\n\n현재가가 5일선과 20일선 위에 있어 단기 및 중기 상승 추세입니다.")
                elif current_price > ma5:
                    st.info("**단기 상승세**\n\n현재가가 5일선 위에 있어 단기적으로 강세입니다.")
                elif current_price < ma5 < ma20:
                    st.error("**강한 하락세**\n\n현재가가 5일선과 20일선 아래에 있어 단기 및 중기 하락 추세입니다.")
                elif current_price < ma5:
                    st.warning("**단기 하락세**\n\n현재가가 5일선 아래에 있어 단기적으로 약세입니다.")
                else:
                    st.info("**횡보세**\n\n뚜렷한 추세가 나타나지 않고 있습니다.")
            else:
                st.info("이동평균 데이터가 충분하지 않습니다.")

        with interp_col2:
            st.markdown("#### 거래량 해석")
            # 거래량 분석
            if market_data and len(market_data.get('volume', [])) > 5:
                recent_volume = market_data['volume'][-1]
                avg_volume = np.mean(market_data['volume'][-6:-1])  # 최근 5일 평균 거래량
                volume_ratio = recent_volume / avg_volume if avg_volume > 0 else 0

                if volume_ratio > 2:
                    st.error(f"**거래량 급증**\n\n최근 5일 평균 대비 {volume_ratio:.1f}배 증가했습니다. 중요한 이벤트나 뉴스가 있을 수 있습니다.")
                elif volume_ratio > 1.5:
                    st.warning(f"**거래량 증가**\n\n5일 평균보다 {volume_ratio:.1f}배 많은 거래량으로 시장의 관심이 높아지고 있습니다.")
                elif volume_ratio > 1:
                    st.info(f"**정상 거래량**\n\n5일 평균과 비슷한 수준의 거래가 이루어지고 있습니다.")
                elif volume_ratio > 0.5:
                    st.warning(f"**거래량 감소**\n\n5일 평균보다 낮은 거래량으로 투자자들의 관심이 줄어들고 있습니다.")
                else:
                    st.error(f"**거래량 급감**\n\n5일 평균 대비 크게 감소했습니다. 관망세가 강해지고 있습니다.")
            else:
                st.info("거래량 데이터가 충분하지 않습니다.")

        with interp_col3:
            st.markdown("#### 변동성 분석")
            # 고가-저가 변동폭 분석
            if market_data and 'high' in market_data and 'low' in market_data and len(market_data['high']) > 0 and len(
                    market_data['low']) > 0:
                latest_high = market_data['high'][-1]
                latest_low = market_data['low'][-1]
                latest_range_pct = (latest_high - latest_low) / latest_low * 100

                # 최근 5일 평균 변동폭 계산
                avg_range_pct = 0
                if len(market_data['high']) >= 5 and len(market_data['low']) >= 5:
                    ranges = [(market_data['high'][i] - market_data['low'][i]) / market_data['low'][i] * 100
                              for i in range(-6, -1)]
                    avg_range_pct = np.mean(ranges)

                if latest_range_pct > 5:
                    st.error(f"**높은 변동성**\n\n당일 변동폭이 {latest_range_pct:.2f}%로 매우 큽니다. 주의가 필요합니다.")
                elif latest_range_pct > 3:
                    st.warning(f"**중간 변동성**\n\n당일 변동폭이 {latest_range_pct:.2f}%로 다소 큰 편입니다.")
                else:
                    st.success(f"**낮은 변동성**\n\n당일 변동폭이 {latest_range_pct:.2f}%로 비교적 안정적입니다.")

                # 평균 대비 변동성
                if avg_range_pct > 0:
                    volatility_ratio = latest_range_pct / avg_range_pct
                    if volatility_ratio > 1.5:
                        st.warning(f"최근 5일 평균보다 {volatility_ratio:.1f}배 높은 변동성을 보이고 있습니다.")
                    elif volatility_ratio < 0.5:
                        st.info(f"최근 5일 평균보다 변동성이 감소했습니다.")
            else:
                st.info("가격 변동폭 데이터가 충분하지 않습니다.")

        with interp_col4:
            st.markdown("#### 가격 저항/지지")
            # 저항선/지지선 분석
            if market_data and 'high' in market_data and 'low' in market_data and len(market_data['high']) > 20 and len(
                    market_data['low']) > 20:
                # 최근 20일 데이터에서 저항선/지지선 계산
                recent_highs = market_data['high'][-20:]
                recent_lows = market_data['low'][-20:]

                # 상위 3개 고가와 하위 3개 저가의 평균으로 저항선/지지선 추정
                resistance = np.mean(sorted(recent_highs, reverse=True)[:3])
                support = np.mean(sorted(recent_lows)[:3])

                # 현재 가격과의 거리 계산
                resistance_gap = (resistance - current_price) / current_price * 100
                support_gap = (current_price - support) / current_price * 100

                st.write(f"**저항선**: {resistance:,.0f}원 (현재가 +{resistance_gap:.2f}%)")
                st.write(f"**지지선**: {support:,.0f}원 (현재가 -{support_gap:.2f}%)")

                # 가격 위치 해석
                if resistance_gap < 1.5:
                    st.warning("현재 저항선 근처에서 거래되고 있습니다. 돌파 시 추가 상승 가능성이 있습니다.")
                elif support_gap < 1.5:
                    st.warning("현재 지지선 근처에서 거래되고 있습니다. 지지선 붕괴 시 추가 하락 가능성이 있습니다.")
                else:
                    price_position = support_gap / (support_gap + resistance_gap) * 100
                    st.info(f"저항-지지 구간의 {price_position:.1f}% 위치에서 거래 중입니다.")
            else:
                st.info("저항/지지선 분석을 위한 데이터가 충분하지 않습니다.")

        # 차트 섹션
        st.subheader("가격 차트")

        if market_data and len(market_data.get('dates', [])) > 0:
            # 캔들스틱 차트
            fig = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.15,  # 간격을 넓힘
                subplot_titles=("가격", "거래량"),
                row_heights=[0.7, 0.3]
            )

            # 캔들스틱 추가
            fig.add_trace(
                go.Candlestick(
                    x=market_data['dates'],
                    open=market_data['open'],
                    high=market_data['high'],
                    low=market_data['low'],
                    close=market_data['close'],
                    name="주가"
                ),
                row=1, col=1
            )

            # 이동평균선 추가
            if len(market_data.get('MA5', [])) > 0:
                fig.add_trace(
                    go.Scatter(
                        x=market_data['dates'],
                        y=market_data['MA5'],
                        name="5일 이동평균",
                        line=dict(color='orange')
                    ),
                    row=1, col=1
                )

            if len(market_data.get('MA20', [])) > 0:
                fig.add_trace(
                    go.Scatter(
                        x=market_data['dates'],
                        y=market_data['MA20'],
                        name="20일 이동평균",
                        line=dict(color='blue')
                    ),
                    row=1, col=1
                )

            # 거래량 바 추가
            fig.add_trace(
                go.Bar(
                    x=market_data['dates'],
                    y=market_data['volume'],
                    name="거래량",
                    marker_color='rgba(0, 0, 255, 0.5)'
                ),
                row=2, col=1
            )

            # 차트 레이아웃 설정
            fig.update_layout(
                title="일간 주가 변동",
                xaxis_title="날짜",
                yaxis_title="주가 (원)",
                height=600,
                xaxis_rangeslider_visible=False
            )

            st.plotly_chart(fig, use_container_width=True)

            # 가격 추가 해석
            st.subheader("추가 주가 분석")
            analysis_col1, analysis_col2 = st.columns(2)

            with analysis_col1:
                # 최근 주가 추세 해석
                if len(market_data['close']) >= 10:
                    recent_prices = market_data['close'][-10:]
                    price_5day_change = (recent_prices[-1] - recent_prices[-5]) / recent_prices[-5] * 100 if \
                        recent_prices[-5] > 0 else 0
                    price_10day_change = (recent_prices[-1] - recent_prices[0]) / recent_prices[0] * 100 if \
                        recent_prices[0] > 0 else 0

                    st.write("#### 최근 가격 변동")
                    st.write(f"- 5일 변동률: {price_5day_change:+.2f}%")
                    st.write(f"- 10일 변동률: {price_10day_change:+.2f}%")

                    # 추세 판단
                    if price_5day_change > 0 and price_10day_change > 0:
                        st.success("단기 및 중기 모두 상승세입니다.")
                    elif price_5day_change > 0 and price_10day_change <= 0:
                        st.info("단기적으로 반등하고 있습니다.")
                    elif price_5day_change <= 0 and price_10day_change > 0:
                        st.warning("최근 상승세가 둔화되고 있습니다.")
                    else:
                        st.error("단기 및 중기 모두 하락세입니다.")

            with analysis_col2:
                # 가격 변동성 분석
                if len(market_data['high']) >= 20 and len(market_data['low']) >= 20:
                    # 20일 변동성 계산 (고가-저가 범위의 평균)
                    ranges = [(market_data['high'][i] - market_data['low'][i]) / market_data['close'][i] * 100
                              for i in range(-20, 0)]
                    avg_volatility = np.mean(ranges)

                    st.write("#### 가격 변동성")
                    st.write(f"- 20일 평균 일변동성: {avg_volatility:.2f}%")

                    # 변동성 해석
                    if avg_volatility > 5:
                        st.error("높은 변동성으로 큰 가격 변동이 예상됩니다. 투자에 주의가 필요합니다.")
                    elif avg_volatility > 3:
                        st.warning("중간 수준의 변동성을 보이고 있습니다.")
                    else:
                        st.success("낮은 변동성으로 비교적 안정적인 가격 움직임을 보이고 있습니다.")

    def display_technical_analysis_tab(self, company_info, market_data, analysis_results):
        """기술적 분석 탭 표시"""
        st.header("기술적 분석")

        ticker = company_info['symbol']
        start, end = self.get_analysis_period()

        # 데이터 수집
        df = yf.download(ticker, start=start, end=end)
        df.columns = df.columns.get_level_values(0)  # 다중 인덱스 열 이름을 단순화
        df['20_MA'] = df['Close'].rolling(window=20).mean()
        df['50_MA'] = df['Close'].rolling(window=50).mean()

        # 현재가 계산
        if market_data and len(market_data.get('close', [])) >= 1:
            current_price = market_data['close'][-1]
        else:
            current_price = 0

        # 분석 결과 요약
        st.subheader("분석 요약")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("현재 추세", analysis_results['trend'])
            st.metric("RSI (14일)", f"{analysis_results['rsi']:.1f}")

        with col2:
            st.metric("5일 이동평균", f"{analysis_results['ma5']:,.0f}원")
            st.metric("RSI 상태", analysis_results['rsi_status'])

        with col3:
            st.metric("20일 이동평균", f"{analysis_results['ma20']:,.0f}원")
            st.metric("거래량 추이", analysis_results['volume_trend'])

        # 최근 5년간 최고가 및 최저가
        st.subheader("최근 5년간 최고가/최저가")
        highest_price = df['High'].max()
        lowest_price = df['Low'].min()

        if not df['High'].empty and not df['High'].isna().all():
            highest_price_date = df['High'].idxmax().strftime('%Y-%m-%d')
        else:
            highest_price_date = "데이터 없음"

        if not df['Low'].empty and not df['Low'].isna().all():
            lowest_price_date = df['Low'].idxmin().strftime('%Y-%m-%d')
        else:
            lowest_price_date = "데이터 없음"

        st.write(f"최고가: {highest_price} (날짜: {highest_price_date})")
        st.write(f"최저가: {lowest_price} (날짜: {lowest_price_date})")

        # 이동 평균
        st.subheader("이동 평균")
        st.line_chart(df[['Close', '20_MA', '50_MA']])

        # MACD 차트
        st.subheader("MACD 차트")

        st.markdown("""
        MACD(Moving Average Convergence Divergence)는 단기 이동평균선과 장기 이동평균선의 차이를 나타내는 추세 추종형 모멘텀 지표입니다.
        - **MACD 선(파란색)**: 12일 지수이동평균에서 26일 지수이동평균을 뺀 값
        - **시그널 선(주황색)**: MACD 선의 9일 지수이동평균
        - **히스토그램(회색)**: MACD 선과 시그널 선의 차이

        MACD 선이 시그널 선을 상향 돌파하면 매수 신호, 하향 돌파하면 매도 신호로 해석할 수 있습니다.
        """)

        if market_data and 'MACD' in market_data:
            fig_macd = go.Figure()

            # MACD 라인
            fig_macd.add_trace(go.Scatter(
                x=market_data['dates'],
                y=market_data['MACD'],
                name="MACD",
                line=dict(color='blue')
            ))

            # 시그널 라인
            fig_macd.add_trace(go.Scatter(
                x=market_data['dates'],
                y=market_data['MACD_Signal'],
                name="Signal",
                line=dict(color='orange')
            ))

            # MACD 히스토그램
            fig_macd.add_trace(go.Bar(
                x=market_data['dates'],
                y=market_data['MACD_Histogram'],
                name="Histogram",
                marker_color='gray'
            ))

            fig_macd.update_layout(
                title="MACD 지표",
                height=400
            )

            st.plotly_chart(fig_macd, use_container_width=True)

        if '20_MA' in df.columns and '50_MA' in df.columns:
            # 2. 데이터가 충분히 있는지 확인
            if not df['20_MA'].empty and not df['50_MA'].empty:
                # 기존 조건 검사 수행
                if current_price > df['20_MA'].iloc[-1] and current_price > df['50_MA'].iloc[-1]:
                    st.write("현재 주가는 이동 평균보다 높습니다. 이는 강세 신호일 수 있습니다.")
                else:
                    st.write("현재 주가는 이동 평균보다 낮습니다. 이는 약세 신호일 수 있습니다.")
                    pass
            else:
                # 데이터가 부족한 경우 처리
                st.warning("일부 기술적 지표에 충분한 데이터가 없습니다.")
        else:
            # 필요한 열이 없는 경우 처리
            st.warning("20일 또는 50일 이동평균선 데이터가 생성되지 않았습니다.")


        # 추세 분석
        st.markdown(f"""
        ### 추세 분석
        - **현재 추세**: {analysis_results['trend']}
        - **5일 이동평균**: {analysis_results['ma5']:,.0f}원
        - **20일 이동평균**: {analysis_results['ma20']:,.0f}원

        ### RSI 분석
        - **현재 RSI**: {analysis_results['rsi']:.1f}
        - **상태**: {analysis_results['rsi_status']}
        - **해석**: {'과매수 상태로 조정 가능성이 있습니다.' if analysis_results['rsi_status'] == '과매수' else '과매도 상태로 반등 가능성이 있습니다.' if analysis_results['rsi_status'] == '과매도' else '중립적인 상태입니다.'}

        ### 거래량 분석
        - **최근 거래량 추이**: {analysis_results['volume_trend']}
        - **해석**: {'거래량이 증가하고 있어 현재 추세가 강화될 수 있습니다.' if analysis_results['volume_trend'] == '증가세' else '거래량이 감소하고 있어 현재 추세가 약화될 수 있습니다.'}
        """)

    def display_investor_tab(self, market_data, analysis_results):
        """투자자 동향 탭 표시"""
        st.header("투자자 동향 분석")

        # 데이터 출처 표시
        is_default_data = market_data.get('is_default_data', False)
        if is_default_data:
            st.warning("⚠️ 투자자 데이터를 가져오지 못해 예시 데이터를 표시합니다. 실제 투자 결정에 참고하지 마세요.")
            st.markdown("""
            <div style="background-color: #fffacd; padding: 10px; border-radius: 5px; border-left: 5px solid #ffcc00;">
                <h4 style="color: #cc5500; margin-top: 0;">⚠️ 예시 데이터 사용 중</h4>
                <p>현재 표시되는 투자자 데이터는 <b>실제 시장 데이터가 아닌 예시 데이터</b>입니다.</p>
                <p>이 데이터는 UI 표시 목적으로만 제공되며, 실제 투자 결정에 사용해서는 안 됩니다.</p>
            </div>
            """, unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            # 투자자별 거래 비중 도넛 차트
            investor_ratio = market_data.get('investor_ratio', [40, 30, 25, 5])

            # 색상 및 레이블 설정
            colors = ['rgb(31, 119, 180)', 'rgb(255, 127, 14)',
                      'rgb(44, 160, 44)', 'rgb(214, 39, 40)']
            labels = ['기관', '외국인', '개인', '기타']
            
            fig_ratio = go.Figure(data=[go.Pie(
                labels=labels,
                values=investor_ratio,
                hole=.4,
                marker_colors=colors,
                textinfo='label+percent',
                hoverinfo='label+percent+value'
            )])
            
            title_text = "투자자별 거래 비중"
            if is_default_data:
                title_text += " (예시 데이터 - 실제 데이터 아님)"
            
            fig_ratio.update_layout(
                title=title_text,
                height=400,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=-0.2,
                    xanchor="center",
                    x=0.5
                )
            )
            
            st.plotly_chart(fig_ratio, use_container_width=True)
        
        with col2:
            # 투자자별 매수/매도 금액 막대 차트
            buy_amounts = market_data.get('buy_amounts', [1000, 800, 600, 200])
            sell_amounts = market_data.get('sell_amounts', [900, 850, 550, 250])
            
            # 데이터 전처리 - 매도 금액을 음수로 표시
            sell_amounts_neg = [-1 * amount for amount in sell_amounts]
            
            fig_amounts = go.Figure()
            
            # 매수 데이터 추가
            fig_amounts.add_trace(go.Bar(
                y=labels,
                x=buy_amounts,
                name='매수',
                orientation='h',
                marker=dict(
                    color='rgba(50, 171, 96, 0.7)',
                    line=dict(color='rgba(50, 171, 96, 1.0)', width=1)
                )
            ))
            
            # 매도 데이터 추가
            fig_amounts.add_trace(go.Bar(
                y=labels,
                x=sell_amounts_neg,
                name='매도',
                orientation='h',
                marker=dict(
                    color='rgba(219, 64, 82, 0.7)',
                    line=dict(color='rgba(219, 64, 82, 1.0)', width=1)
                )
            ))
            
            title_text = "투자자별 매수/매도 금액 (십억원)"
            if is_default_data:
                title_text += " (예시 데이터 - 실제 데이터 아님)"
            fig_amounts.update_layout(
                title=title_text,
                barmode='relative',
                height=400,
                xaxis=dict(
                    title=dict(text='금액 (십억원)', font=dict(size=14)),  # titlefont_size 수정
                    tickfont=dict(size=12),  # tickfont_size 수정
                ),
                yaxis=dict(
                    title=dict(font=dict(size=14)),  # titlefont_size 수정
                    tickfont=dict(size=12),  # tickfont_size 수정
                ),
                legend=dict(
                    x=0.5,
                    y=1.0,
                    bgcolor='rgba(255, 255, 255, 0)',
                    bordercolor='rgba(255, 255, 255, 0)',
                    orientation="h"
                )
            )
            # fig_amounts.update_layout(
            #     title=title_text,
            #     barmode='relative',
            #     height=400,
            #     xaxis=dict(
            #         title='금액 (십억원)',
            #         titlefont_size=14,
            #         tickfont_size=12,
            #     ),
            #     yaxis=dict(
            #         titlefont_size=14,
            #         tickfont_size=12,
            #     ),
            #     legend=dict(
            #         x=0.5,
            #         y=1.0,
            #         bgcolor='rgba(255, 255, 255, 0)',
            #         bordercolor='rgba(255, 255, 255, 0)',
            #         orientation="h"
            #     )
            # )
            
            st.plotly_chart(fig_amounts, use_container_width=True)
        
        # 투자자 동향 분석
        st.subheader("투자자 동향 분석")
        
        if is_default_data:
            st.markdown("""
            <div style="background-color: #ffebee; padding: 10px; border-radius: 5px; border-left: 5px solid #f44336; margin-bottom: 20px;">
                <h4 style="color: #b71c1c; margin-top: 0;">⚠️ 주의: 예시 데이터 기반 분석</h4>
                <p>아래 분석은 <b>실제 시장 데이터가 아닌 예시 데이터</b>를 기반으로 합니다.</p>
                <p>실제 투자 결정에 참고하지 마세요.</p>
            </div>
            """, unsafe_allow_html=True)

        # 안전하게 인덱스를 찾는 함수
        def safe_find_index(investor_type, labels_list):
            # '투자자' 접미사 제거
            cleaned_type = investor_type.replace('투자자', '').replace('법인', '').strip()

            # 정확한 일치 시도
            try:
                return labels_list.index(cleaned_type)
            except ValueError:
                # 부분 일치 시도
                for i, label in enumerate(labels_list):
                    if cleaned_type in label or label in cleaned_type:
                        return i
                # 기본값 반환
                return 0  # 기본적으로 첫 번째 인덱스 반환

        # 주요 매수/매도 세력 인덱스 안전하게 찾기
        main_buyer_idx = safe_find_index(analysis_results['main_buyer'], labels)
        main_seller_idx = safe_find_index(analysis_results['main_seller'], labels)

        st.markdown(f"""
        ### 주요 투자자 동향
        - **주요 매수세력**: {analysis_results['main_buyer']} ({investor_ratio[main_buyer_idx]}%)
        - **주요 매도세력**: {analysis_results['main_seller']} ({investor_ratio[main_seller_idx]}%)

        ### 투자자별 비중
        - **기관투자자**: {investor_ratio[0]}%
        - **외국인**: {investor_ratio[1]}%
        - **개인**: {investor_ratio[2]}%
        - **기타법인**: {investor_ratio[3]}%

        ### 해석
        {f"기관투자자의 비중이 높아 기관의 매매 동향에 주목할 필요가 있습니다." if investor_ratio[0] > 30 else ""}
        {f"외국인의 비중이 높아 외국인 투자자의 매매 동향이 주가에 큰 영향을 줄 수 있습니다." if investor_ratio[1] > 30 else ""}
        {f"개인 투자자의 비중이 높아 개인 투자심리가 주가에 영향을 줄 수 있습니다." if investor_ratio[2] > 30 else ""}
        """)
        
        if is_default_data:
            st.markdown("""
            <div style="background-color: #e8f5e9; padding: 10px; border-radius: 5px; border-left: 5px solid #4caf50; margin-top: 20px;">
                <h4 style="color: #2e7d32; margin-top: 0;">💡 실제 데이터 확인 방법</h4>
                <p>실제 투자자 데이터를 확인하려면 한국거래소(KRX) 공식 웹사이트나 증권사 HTS/MTS를 이용하세요.</p>
            </div>
            """, unsafe_allow_html=True)

    def display_financial_analysis_tab(self, company_info):
        """재무 분석 탭 표시 - FinancialAnalysisView 클래스 사용"""
        self.financial_analysis_view.display(company_info)

async def main():
    """메인 함수"""
    app = StockAnalysisApp()
    app.setup_page()
    
    # 검색 섹션 렌더링
    company_info = app.render_search_section()
    
    # 분석 실행
    if company_info:
        await app.analyze_stock(company_info)
    
    # 결과 표시
    app.display_results()

if __name__ == "__main__":
    asyncio.run(main())