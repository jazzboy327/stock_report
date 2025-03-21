# src/utils/stock_data_collector.py의 수정 버전

import yfinance as yf
import FinanceDataReader as fdr
from pykrx import stock
import pandas as pd
import numpy as np
import datetime
from datetime import timedelta
import logging
import traceback
import asyncio
from src.models.stock_info import StockController

logger = logging.getLogger('StockAnalysisApp.DataCollector')


class StockDataCollector:
    """주식 데이터 수집 클래스"""

    def __init__(self):
        """데이터 수집기 초기화"""
        # 기본 설정
        self.default_history_years = 3  # 기본 히스토리 수집 기간: 3년으로 확장
        self.max_history_years = 5  # 최대 수집 가능 기간: 5년
        self.investor_data_days = 30  # 투자자 데이터 수집 기간: 30일로 확장
        self.technical_chart_days = 90  # 기술적 차트 데이터: 90일로 확장

    async def get_stock_history(self, symbol, period=None, years=None):
        """주식 히스토리 데이터 수집 - 수집 기간 확장"""
        try:
            # 날짜 범위 설정
            end_date = datetime.datetime.now()

            # 기간 설정: years 인자가 있으면 그 값 사용, 없으면 기본값 사용
            years_to_collect = years if years else self.default_history_years
            # 최대 수집 기간 제한
            years_to_collect = min(years_to_collect, self.max_history_years)

            # 시작일 계산
            start_date = end_date - timedelta(days=int(365 * years_to_collect))

            logger.info(
                f"주식 데이터 수집 기간: {start_date.strftime('%Y-%m-%d')} ~ {end_date.strftime('%Y-%m-%d')} ({years_to_collect}년)")

            # FinanceDataReader로 실제 데이터 수집
            df = fdr.DataReader(symbol.replace('.KS', '').replace('.KQ', ''),
                                start_date.strftime('%Y-%m-%d'),
                                end_date.strftime('%Y-%m-%d'))

            if df.empty:
                raise ValueError("주식 데이터를 찾을 수 없습니다.")

            # 데이터 품질 확인 및 결측치 처리
            df = self._ensure_data_quality(df)

            return df
        except Exception as e:
            logger.error(f"주식 히스토리 데이터 수집 중 오류: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def _ensure_data_quality(self, df):
        """데이터 품질 확인 및 결측치 처리"""
        # 결측치 확인
        missing_values = df.isnull().sum()
        if missing_values.any():
            logger.warning(f"데이터에 결측치 발견: {missing_values}")

            # 전진 채우기 후 후진 채우기로 결측치 처리
            df = df.ffill().bfill()

            logger.info("결측치 처리 완료")

        # 이상치 확인 (예: 거래량이 0인 날짜)
        zero_volume_days = (df['Volume'] == 0).sum()
        if zero_volume_days > 0:
            logger.warning(f"거래량이 0인 거래일 발견: {zero_volume_days}일")

            # 거래량이 0인 날짜는 직전 거래일 데이터로 대체
            df.loc[df['Volume'] == 0, 'Volume'] = df['Volume'].replace(0, method='ffill')

        return df

    async def prepare_chart_data(self, df, days=None):
        """차트 데이터 준비 - 기간 확장"""
        try:
            # 표시 기간 설정 (명시적 지정 또는 기본값)
            display_days = days if days else self.technical_chart_days

            # 최근 n일 데이터 추출
            df_recent = df.tail(display_days).copy()

            # 이동평균선 계산 - 다양한 기간 추가
            df_recent['MA5'] = df_recent['Close'].rolling(window=5).mean()
            df_recent['MA10'] = df_recent['Close'].rolling(window=10).mean()
            df_recent['MA20'] = df_recent['Close'].rolling(window=20).mean()
            df_recent['MA60'] = df_recent['Close'].rolling(window=60).mean()
            df_recent['MA120'] = df_recent['Close'].rolling(window=120).mean()

            # VWAP (Volume Weighted Average Price) 추가
            df_recent['VWAP'] = (df_recent['Close'] * df_recent['Volume']).cumsum() / df_recent['Volume'].cumsum()

            # 차트 데이터 구성
            chart_data = {
                'dates': df_recent.index.strftime('%Y-%m-%d').tolist(),
                'open': df_recent['Open'].tolist(),
                'high': df_recent['High'].tolist(),
                'low': df_recent['Low'].tolist(),
                'close': df_recent['Close'].tolist(),
                'volume': df_recent['Volume'].tolist(),
                'MA5': df_recent['MA5'].tolist(),
                'MA10': df_recent['MA10'].tolist(),
                'MA20': df_recent['MA20'].tolist(),
                'MA60': df_recent['MA60'].tolist(),
                'MA120': df_recent['MA120'].tolist(),
                'VWAP': df_recent['VWAP'].tolist()
            }

            return chart_data, df_recent
        except Exception as e:
            logger.error(f"차트 데이터 준비 중 오류: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def calculate_rsi(self, prices, period=14):
        """RSI 계산"""
        try:
            deltas = np.diff(prices)
            seed = deltas[:period + 1]
            up = seed[seed >= 0].sum() / period
            down = -seed[seed < 0].sum() / period
            if down == 0:
                return 100
            rs = up / down
            rsi = np.zeros_like(prices)
            rsi[:period] = 100. - 100. / (1. + rs)

            for i in range(period, len(prices)):
                delta = deltas[i - 1]
                if delta > 0:
                    upval = delta
                    downval = 0.
                else:
                    upval = 0.
                    downval = -delta

                up = (up * (period - 1) + upval) / period
                down = (down * (period - 1) + downval) / period
                rs = up / down
                rsi[i] = 100. - 100. / (1. + rs)

            return rsi[-1]
        except Exception as e:
            logger.error(f"RSI 계산 중 오류: {str(e)}")
            return 50.0

    def calculate_macd(self, prices):
        """MACD 계산"""
        try:
            exp1 = prices.ewm(span=12, adjust=False).mean()
            exp2 = prices.ewm(span=26, adjust=False).mean()
            macd = exp1 - exp2
            signal = macd.ewm(span=9, adjust=False).mean()

            return {
                'MACD': macd.tolist(),
                'MACD_Signal': signal.tolist(),
                'MACD_Histogram': (macd - signal).tolist()
            }
        except Exception as e:
            logger.error(f"MACD 계산 중 오류: {str(e)}")
            return {'MACD': [], 'MACD_Signal': [], 'MACD_Histogram': []}

    async def get_investor_data(self, symbol):
        """투자자 데이터 수집 - 수집 기간 확장 및 에러 처리 강화"""
        try:
            # 날짜 범위 설정 (최대 30일)
            end_date = datetime.datetime.now()
            start_date = end_date - timedelta(days=self.investor_data_days)

            end_date_str = end_date.strftime("%Y%m%d")
            start_date_str = start_date.strftime("%Y%m%d")

            logger.info(f"투자자 데이터 수집 중: {symbol}, {start_date_str} ~ {end_date_str}")

            # 재시도 로직 추가
            max_retries = 3
            retry_count = 0
            retry_delay = 1  # 초 단위 대기 시간 (지수적으로 증가)

            while retry_count < max_retries:
                try:
                    # KRX API 호출 시도
                    df = stock.get_market_trading_value_by_date(
                        fromdate=start_date_str,
                        todate=end_date_str,
                        ticker=symbol
                    )

                    # 데이터 유효성 검사
                    if df is None:
                        logger.warning(f"투자자 데이터가 None으로 반환됨: {symbol}")
                        retry_count += 1
                        await asyncio.sleep(retry_delay)
                        retry_delay *= 2  # 지수적 백오프
                        continue

                    if df.empty:
                        logger.warning(f"투자자 데이터가 비어있음: {symbol}")
                        return {}

                    # 필수 컬럼 확인
                    expected_columns = ['기관합계', '외국인합계', '개인', '기타법인']
                    missing_columns = [col for col in expected_columns if col not in df.columns]

                    if missing_columns:
                        logger.warning(f"데이터에서 필요한 컬럼 누락: {missing_columns}")
                        retry_count += 1
                        await asyncio.sleep(retry_delay)
                        retry_delay *= 2
                        continue

                    # 성공적으로 데이터를 가져옴
                    break

                except requests.exceptions.JSONDecodeError as e:
                    logger.warning(f"JSON 파싱 오류 (시도 {retry_count + 1}/{max_retries}): {str(e)}")
                    retry_count += 1
                    if retry_count >= max_retries:
                        logger.error(f"최대 재시도 횟수 초과, 기본 데이터를 사용합니다.")
                        return {}
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 2  # 지수적 백오프

                except TypeError as e:
                    # 'NoneType' object is not subscriptable 에러 처리
                    if "'NoneType' object is not subscriptable" in str(e):
                        logger.warning(f"KRX API가 티커 정보를 찾지 못함: {symbol}")
                        return {}
                    # 그 외 TypeError는 재시도
                    logger.warning(f"유형 오류 (시도 {retry_count + 1}/{max_retries}): {str(e)}")
                    retry_count += 1
                    if retry_count >= max_retries:
                        return {}
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 2

                except Exception as e:
                    logger.error(f"투자자 데이터 수집 중 예상치 못한 오류: {str(e)}")
                    logger.error(traceback.format_exc())
                    return {}

            # 최대 재시도 횟수 초과
            if retry_count >= max_retries:
                logger.warning(f"KRX API 호출 최대 재시도 횟수 초과: {symbol}")
                return {}

            # 투자자 데이터 처리 및 투자자 분류별 추세 분석
            investor_trends = self._analyze_investor_trends(df)

            # 최신 데이터만 사용 (마지막 거래일)
            try:
                latest_date = df.index[-1]
                latest_data = df.loc[latest_date]

                # 각 투자자 유형별 순매수 금액 계산 - 안전한 접근
                inst_buy = latest_data.get('기관합계', 0)
                other_buy = latest_data.get('기타법인', 0)
                retail_buy = latest_data.get('개인', 0)
                foreign_buy = latest_data.get('외국인합계', 0)

                # 순매수 금액의 절대값 계산 (비율 계산용) - 데이터 타입 안전 처리
                try:
                    inst_buy_abs = abs(float(inst_buy))
                    other_buy_abs = abs(float(other_buy))
                    retail_buy_abs = abs(float(retail_buy))
                    foreign_buy_abs = abs(float(foreign_buy))
                except (ValueError, TypeError):
                    logger.warning(f"순매수 금액 변환 실패, 기본값 사용")
                    return {}

                # 총 거래 금액 계산
                total_trade = inst_buy_abs + other_buy_abs + retail_buy_abs + foreign_buy_abs

                # 투자자별 비율 계산
                if total_trade > 0:
                    inst_ratio = round((inst_buy_abs / total_trade) * 100)
                    foreign_ratio = round((foreign_buy_abs / total_trade) * 100)
                    retail_ratio = round((retail_buy_abs / total_trade) * 100)
                    other_ratio = round((other_buy_abs / total_trade) * 100)
                else:
                    logger.warning("총 거래금액이 0이어서 비율을 계산할 수 없음")
                    return {}

                # 비율 합이 100%가 되도록 조정
                total_ratio = inst_ratio + foreign_ratio + retail_ratio + other_ratio
                if total_ratio != 100:
                    diff = 100 - total_ratio
                    # 가장 큰 비율에 차이를 더함
                    max_ratio = max(inst_ratio, foreign_ratio, retail_ratio, other_ratio)
                    if max_ratio == inst_ratio:
                        inst_ratio += diff
                    elif max_ratio == foreign_ratio:
                        foreign_ratio += diff
                    elif max_ratio == retail_ratio:
                        retail_ratio += diff
                    else:
                        other_ratio += diff

                # 십억 단위로 변환 (차트 표시용)
                try:
                    buy_amounts = [
                        max(int(inst_buy / 1_000_000_000), 0) if inst_buy > 0 else 0,
                        max(int(foreign_buy / 1_000_000_000), 0) if foreign_buy > 0 else 0,
                        max(int(retail_buy / 1_000_000_000), 0) if retail_buy > 0 else 0,
                        max(int(other_buy / 1_000_000_000), 0) if other_buy > 0 else 0
                    ]

                    # 매도 금액은 음수값만 사용
                    sell_amounts = [
                        abs(int(inst_buy / 1_000_000_000)) if inst_buy < 0 else 0,
                        abs(int(foreign_buy / 1_000_000_000)) if foreign_buy < 0 else 0,
                        abs(int(retail_buy / 1_000_000_000)) if retail_buy < 0 else 0,
                        abs(int(other_buy / 1_000_000_000)) if other_buy < 0 else 0
                    ]
                except (ValueError, TypeError):
                    logger.warning("매수/매도 금액 변환 실패, 기본값 사용")
                    return {}

                # 결과에 투자자 추세 정보 추가
                result = {
                    'investor_ratio': [inst_ratio, foreign_ratio, retail_ratio, other_ratio],
                    'buy_amounts': buy_amounts,
                    'sell_amounts': sell_amounts,
                    'is_default_data': False,
                    'investor_trends': investor_trends
                }

                return result

            except (IndexError, KeyError) as e:
                logger.error(f"투자자 데이터 처리 중 인덱스/키 오류: {str(e)}")
                return {}

        except Exception as e:
            logger.error(f"투자자 데이터 수집 중 오류 발생: {str(e)}")
            logger.error(traceback.format_exc())
            return {}

    def _analyze_investor_trends(self, df):
        """투자자 추세 분석 - 기간별 누적 순매수 분석"""
        try:
            # 기간별 분석: 최근 5일, 10일, 30일
            periods = [5, 10, 30]
            trends = {}

            investor_types = ['기관합계', '외국인합계', '개인', '기타법인']

            for period in periods:
                if len(df) >= period:
                    period_df = df.tail(period)
                    period_trends = {}

                    for investor in investor_types:
                        if investor in period_df.columns:
                            # 순매수 금액 누적합 계산
                            net_buy_sum = period_df[investor].sum()

                            # 순매수 추세 판정
                            if net_buy_sum > 0:
                                trend = "순매수"
                            elif net_buy_sum < 0:
                                trend = "순매도"
                            else:
                                trend = "중립"

                            period_trends[investor] = {
                                'net_buy_sum': int(net_buy_sum),
                                'trend': trend
                            }

                    trends[f'{period}일'] = period_trends

            return trends

        except Exception as e:
            logger.error(f"투자자 추세 분석 중 오류: {str(e)}")
            return {}

    def generate_analysis_report(self, market_data):
        """기술적 분석 리포트 생성"""
        try:
            current_price = market_data['close'][-1]
            ma5 = market_data['MA5'][-1] if not np.isnan(market_data['MA5'][-1]) else 0
            ma20 = market_data['MA20'][-1] if not np.isnan(market_data['MA20'][-1]) else 0
            rsi = market_data['RSI']

            # 추세 판단
            if current_price > ma5 > ma20:
                trend = "강한 상승세"
            elif current_price > ma5:
                trend = "상승세"
            elif current_price < ma5 < ma20:
                trend = "강한 하락세"
            elif current_price < ma5:
                trend = "하락세"
            else:
                trend = "횡보세"

            # RSI 상태 판단
            if rsi > 70:
                rsi_status = "과매수"
            elif rsi < 30:
                rsi_status = "과매도"
            else:
                rsi_status = "중립"

            # 거래량 추이 판단
            volume_trend = "증가세" if market_data['volume'][-1] > np.mean(market_data['volume'][-5:]) else "감소세"

            # 투자자 동향
            investor_ratios = market_data.get('investor_ratio', [0, 0, 0, 0])
            investor_types = ['기관투자자', '외국인', '개인', '기타법인']

            # 가장 비중이 높은 투자자 유형 찾기
            max_buy_idx = np.argmax(market_data.get('buy_amounts', [0, 0, 0, 0]))
            max_sell_idx = np.argmax(market_data.get('sell_amounts', [0, 0, 0, 0]))

            main_buyer = investor_types[max_buy_idx]
            main_seller = investor_types[max_sell_idx]

            return {
                'trend': trend,
                'ma5': ma5,
                'ma20': ma20,
                'rsi': rsi,
                'rsi_status': rsi_status,
                'volume_trend': volume_trend,
                'main_buyer': main_buyer,
                'main_seller': main_seller,
                'investor_ratios': investor_ratios
            }
        except Exception as e:
            logger.error(f"분석 리포트 생성 중 오류: {str(e)}")
            logger.error(traceback.format_exc())
            return {
                'trend': "분석 불가",
                'ma5': 0,
                'ma20': 0,
                'rsi': 50,
                'rsi_status': "중립",
                'volume_trend': "분석 불가",
                'main_buyer': "분석 불가",
                'main_seller': "분석 불가",
                'investor_ratios': [25, 25, 25, 25]
            }

    async def collect_all_data(self, symbol):
        """모든 주식 데이터 수집 및 분석"""
        try:
            # 1. 주식 컨트롤러에서 기본 정보 가져오기
            stock_controller = StockController()
            stock_info, _ = await stock_controller.get_stock_data(symbol)

            # 2. 히스토리 데이터 가져오기
            df = await self.get_stock_history(symbol)

            # 3. 차트 데이터 준비
            chart_data, df_recent = await self.prepare_chart_data(df)

            # 4. 기술적 지표 계산
            chart_data['RSI'] = self.calculate_rsi(np.array(chart_data['close']))

            # 5. MACD 계산
            macd_data = self.calculate_macd(df_recent['Close'])
            chart_data.update(macd_data)

            # 6. 투자자 데이터 수집
            investor_data = await self.get_investor_data(symbol.replace('.KS', '').replace('.KQ', ''))

            if investor_data:
                chart_data.update(investor_data)
            else:
                # 기본값 사용
                chart_data.update({
                    'investor_ratio': [35, 28, 32, 5],  # 기관, 외국인, 개인, 기타
                    'buy_amounts': [350, 280, 320, 50],
                    'sell_amounts': [320, 260, 300, 45],
                    'is_default_data': True
                })

            # 7. 분석 결과 생성
            analysis_results = self.generate_analysis_report(chart_data)

            return stock_info, chart_data, analysis_results

        except Exception as e:
            logger.error(f"데이터 수집 중 오류 발생: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def get_stock_history_sync(self, symbol, period='1y'):
        """주식 히스토리 데이터 수집 (동기식 버전)"""
        try:
            # 날짜 범위 설정
            end_date = datetime.datetime.now()
            start_date = end_date - timedelta(days=365)  # 1년치 데이터

            # FinanceDataReader로 실제 데이터 수집
            df = fdr.DataReader(symbol.replace('.KS', '').replace('.KQ', ''),
                                start_date.strftime('%Y-%m-%d'),
                                end_date.strftime('%Y-%m-%d'))

            if df.empty:
                raise ValueError("주식 데이터를 찾을 수 없습니다.")

            return df
        except Exception as e:
            logger.error(f"주식 히스토리 데이터 수집 중 오류: {str(e)}")
            logger.error(traceback.format_exc())
            raise