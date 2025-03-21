# src/utils/optimal_trading_analyzer.py

import pandas as pd
import numpy as np
import pandas_ta as ta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import logging
from typing import Dict, List, Tuple, Optional, Union

logger = logging.getLogger('StockAnalysisApp.OptimalTradingAnalyzer')


class OptimalTradingAnalyzer:
    """최적 매매 시점 분석 클래스"""

    def __init__(self):
        """분석기 초기화"""
        # 기술적 지표 계산에 필요한 최소 데이터 수 정의
        self.MIN_REQUIRED_DATA = {
            'SMA20': 20,     # 20일 이동평균
            'SMA50': 30,     # 50일 이동평균 (원래 50이지만 30으로 조정)
            'SMA200': 60,    # 200일 이동평균 (원래 200이지만 60으로 조정)
            'RSI': 14,       # 14일 RSI
            'BB': 20,        # 20일 볼린저 밴드
            'MACD': 26,      # MACD (26일이 가장 긴 기간)
            'ADX': 14,       # 14일 ADX
            'Stochastic': 14 # 14일 스토캐스틱
        }
        
        # 최소 필수 데이터 기간 (여러 지표를 함께 사용하므로 충분해야 함)
        self.ABSOLUTE_MIN_DATA = 30  # 최소 요구 기간 (원래 60이었으나 30으로 완화)
        
        # 권장 최소 데이터 기간 (신뢰할 수 있는 분석을 위해)
        self.RECOMMENDED_MIN_DATA = 60  # 권장 기간 (원래 200이었으나 60으로 완화)

    def analyze(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """
        주가 데이터를 분석하여 매매 신호 생성 - 데이터 부족 시에도 제한적 분석 수행

        Args:
            df: 주가 데이터 DataFrame (Open, High, Low, Close, Volume 컬럼 필요)

        Returns:
            분석 결과가 추가된 DataFrame과 요약 정보를 담은 딕셔너리
        """
        try:
            # 데이터 길이 확인
            data_length = len(df)
            
            # 최소 필요 데이터 수 검증
            if data_length < self.ABSOLUTE_MIN_DATA:
                raise ValueError(f"데이터 길이({data_length}일)가 분석에 필요한 최소 길이({self.ABSOLUTE_MIN_DATA}일)보다 짧습니다.")
            
            # 권장 데이터 수보다 적으면 경고 로그
            if data_length < self.RECOMMENDED_MIN_DATA:
                logger.warning(f"데이터 길이({data_length}일)가 권장 최소 길이({self.RECOMMENDED_MIN_DATA}일)보다 짧아 분석 정확도가 떨어질 수 있습니다.")
                
            # 데이터 복사 및 전처리
            data = df.copy()

            # 컬럼명 표준화 (대소문자 등)
            column_map = {
                'open': 'Open', 'Open': 'Open', 'OPEN': 'Open',
                'high': 'High', 'High': 'High', 'HIGH': 'High',
                'low': 'Low', 'Low': 'Low', 'LOW': 'Low',
                'close': 'Close', 'Close': 'Close', 'CLOSE': 'Close',
                'volume': 'Volume', 'Volume': 'Volume', 'VOLUME': 'Volume'
            }

            # 컬럼 이름 표준화
            data.rename(columns={col: column_map[col] for col in data.columns if col in column_map}, inplace=True)

            # 필요한 컬럼 확인
            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            missing_columns = [col for col in required_columns if col not in data.columns]
            
            if missing_columns:
                raise ValueError(f"입력 데이터에 다음 컬럼이 없습니다: {', '.join(missing_columns)}")

            # 결측치 확인 및 처리
            if data.isnull().any().any():
                logger.warning("데이터에 결측치가 있습니다. 전방 채우기(ffill) 및 후방 채우기(bfill)로 처리합니다.")
                data = data.ffill().bfill()  # 앞의 값, 뒤의 값으로 결측치 대체

            # 1. 이동평균선 (Simple Moving Average) - 데이터 길이에 맞게 조정
            # SMA200은 데이터가 충분하지 않을 경우 사용 가능한 최대 기간으로 조정
            if data_length >= 60:
                data['SMA200'] = ta.sma(data['Close'], length=60)  # 원래 200이었으나 60으로 완화
            else:
                max_sma = (data_length // 3) * 2  # 데이터 길이의 약 2/3로 설정
                if max_sma >= 30:  # 최소 30일 이상은 되어야 의미있는 장기 이동평균
                    logger.warning(f"데이터가 60일 미만이어서 SMA60 대신 SMA{max_sma}를 사용합니다.")
                    data['SMA200'] = ta.sma(data['Close'], length=max_sma)
                else:
                    # 30일 미만일 경우 더 짧게 설정
                    max_sma = max(15, data_length // 2)  
                    logger.warning(f"데이터가 매우 부족하여 SMA60 대신 SMA{max_sma}를 사용합니다.")
                    data['SMA200'] = ta.sma(data['Close'], length=max_sma)
            
            # SMA50도 데이터 길이에 맞게 조정
            if data_length >= 50:
                data['SMA50'] = ta.sma(data['Close'], length=50)
            else:
                max_sma = min(30, data_length // 2)  # 최대 30일, 또는 데이터 길이의 1/2
                logger.warning(f"데이터가 50일 미만이어서 SMA50 대신 SMA{max_sma}를 사용합니다.")
                data['SMA50'] = ta.sma(data['Close'], length=max_sma)
            
            # 다른 이동평균선
            if data_length >= 20:
                data['SMA20'] = ta.sma(data['Close'], length=20)
            else:
                max_sma = max(5, data_length // 3)  # 최소 5일
                logger.warning(f"데이터가 20일 미만이어서 SMA20 대신 SMA{max_sma}를 사용합니다.")
                data['SMA20'] = ta.sma(data['Close'], length=max_sma)

            # 2. 지수이동평균선 (Exponential Moving Average)
            if data_length >= 21:
                data['EMA21'] = ta.ema(data['Close'], length=21)
            else:
                ema_length = max(5, data_length // 4)
                data['EMA21'] = ta.ema(data['Close'], length=ema_length)
                
            if data_length >= 9:
                data['EMA9'] = ta.ema(data['Close'], length=9)
            else:
                ema_length = max(3, data_length // 7)
                data['EMA9'] = ta.ema(data['Close'], length=ema_length)

            # 3. MACD (Moving Average Convergence Divergence)
            # MACD 기간 조정: 12, 26, 9 → 최대 6, 13, 4 (축소된 기간)
            macd_fast = min(12, max(3, data_length // 8))
            macd_slow = min(26, max(7, data_length // 4))
            macd_signal = min(9, max(3, data_length // 10))
            
            macd = ta.macd(data['Close'], fast=macd_fast, slow=macd_slow, signal=macd_signal)
            data = pd.concat([data, macd], axis=1)

            # 4. RSI (Relative Strength Index)
            rsi_length = min(14, max(7, data_length // 4))
            data['RSI14'] = ta.rsi(data['Close'], length=rsi_length)

            # 5. 볼린저 밴드 (Bollinger Bands)
            bb_length = min(20, max(10, data_length // 3))
            bbands = ta.bbands(data['Close'], length=bb_length, std=2)
            data = pd.concat([data, bbands], axis=1)

            # 6. 스토캐스틱 (Stochastic)
            stoch_k = min(14, max(5, data_length // 5))
            stoch_d = min(3, max(2, data_length // 15))
            stoch_smooth = min(3, max(2, data_length // 15))
            stoch = ta.stoch(data['High'], data['Low'], data['Close'], 
                            k=stoch_k, d=stoch_d, smooth_k=stoch_smooth)
            data = pd.concat([data, stoch], axis=1)

            # 7. ADX (Average Directional Index)
            adx_length = min(14, max(7, data_length // 4))
            adx = ta.adx(data['High'], data['Low'], data['Close'], length=adx_length)
            data = pd.concat([data, adx], axis=1)

            # 8. OBV (On-Balance Volume)
            data['OBV'] = ta.obv(data['Close'], data['Volume'])

            # NaN 값 추가 확인 및 처리
            if data.isnull().any().any():
                # NaN 발생 위치 및 개수 로깅
                nan_counts = data.isnull().sum()
                nan_cols = nan_counts[nan_counts > 0]
                logger.info(f"기술적 지표 계산 후 NaN 발생 컬럼 및 개수: {nan_cols.to_dict()}")
                
                # ffill과 bfill로 남은 NaN 처리 시도
                data = data.ffill().bfill()
                
                # 그래도 NaN이 남아있는지 확인
                if data.isnull().any().any():
                    # 앞부분의 NaN은 기술적 지표를 계산할 수 없어서 발생 - 이 행들은 제외
                    logger.info("기술적 지표 계산으로 인한 NaN을 제거합니다.")
                    # NaN이 있는 행 삭제 (대부분 초기 행들)
                    pre_drop_len = len(data)
                    data = data.dropna()
                    post_drop_len = len(data)
                    
                    dropped_rows = pre_drop_len - post_drop_len
                    logger.info(f"NaN 제거로 {dropped_rows}개 행이 삭제되었습니다.")
                    
                    # 데이터가 너무 적어지면 경고하지만, 30일 이상이면 계속 진행 
                    if len(data) < self.ABSOLUTE_MIN_DATA:
                        logger.error(f"NaN 제거 후 데이터가 {len(data)}일로 부족합니다.")
                        
                        # 20일 이상이면 제한된 분석 시도
                        if len(data) >= 20:
                            logger.warning(f"데이터가 부족하지만 제한된 분석을 시도합니다. 신뢰도가 낮을 수 있습니다.")
                        else:
                            raise ValueError(f"NaN 제거 후 데이터 길이({len(data)}일)가 분석에 필요한 최소 길이({self.ABSOLUTE_MIN_DATA}일)보다 짧습니다.")

            # 매매 신호 생성
            data = self._generate_signals(data)

            # 성과 분석
            performance = self._analyze_performance(data)
            
            # 데이터 신뢰도 정보 추가
            data_confidence = min(100, (len(data) / self.RECOMMENDED_MIN_DATA) * 100)
            performance['data_confidence'] = data_confidence
            
            if data_confidence < 50:
                performance['data_confidence_level'] = "매우 낮음"
                performance['warning'] = "데이터가 매우 부족하여 분석 신뢰도가 낮습니다."
            elif data_confidence < 75:
                performance['data_confidence_level'] = "낮음"
                performance['warning'] = "데이터가 부족하여 분석 신뢰도가 다소 낮습니다."
            elif data_confidence < 90:
                performance['data_confidence_level'] = "보통"
                performance['warning'] = "중간 수준의 데이터로 분석했습니다. 참고용으로만 활용하세요."
            else:
                performance['data_confidence_level'] = "높음"

            return data, performance

        except Exception as e:
            logger.error(f"매매 시점 분석 중 오류 발생: {str(e)}")
            raise

    def _generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        기술적 지표를 기반으로 매매 신호 생성

        Args:
            data: 기술적 지표가 추가된 데이터프레임

        Returns:
            매매 신호가 추가된 데이터프레임
        """
        try:
            # 초기 설정
            data['Signal'] = 0  # 0: 홀드, 1: 매수, -1: 매도
            data['Signal_Type'] = ""  # 신호 유형 설명

            # 1. 골든 크로스 / 데드 크로스 (SMA20과 SMA50 기준)
            data.loc[(data['SMA20'] > data['SMA50']) & (data['SMA20'].shift(1) <= data['SMA50'].shift(1)), 'Signal'] = 1
            data.loc[(data['SMA20'] > data['SMA50']) & (
                    data['SMA20'].shift(1) <= data['SMA50'].shift(1)), 'Signal_Type'] = "골든 크로스 (SMA20 > SMA50)"

            data.loc[
                (data['SMA20'] < data['SMA50']) & (data['SMA20'].shift(1) >= data['SMA50'].shift(1)), 'Signal'] = -1
            data.loc[(data['SMA20'] < data['SMA50']) & (
                    data['SMA20'].shift(1) >= data['SMA50'].shift(1)), 'Signal_Type'] = "데드 크로스 (SMA20 < SMA50)"

            # 2. MACD 신호
            data.loc[(data['MACD_12_26_9'] > data['MACDs_12_26_9']) & (
                    data['MACD_12_26_9'].shift(1) <= data['MACDs_12_26_9'].shift(1)), 'Signal'] = 1
            data.loc[(data['MACD_12_26_9'] > data['MACDs_12_26_9']) & (
                    data['MACD_12_26_9'].shift(1) <= data['MACDs_12_26_9'].shift(1)), 'Signal_Type'] = "MACD 골든 크로스"

            data.loc[(data['MACD_12_26_9'] < data['MACDs_12_26_9']) & (
                    data['MACD_12_26_9'].shift(1) >= data['MACDs_12_26_9'].shift(1)), 'Signal'] = -1
            data.loc[(data['MACD_12_26_9'] < data['MACDs_12_26_9']) & (
                    data['MACD_12_26_9'].shift(1) >= data['MACDs_12_26_9'].shift(1)), 'Signal_Type'] = "MACD 데드 크로스"

            # 3. RSI 과매수/과매도
            data.loc[(data['RSI14'] < 30) & (data['RSI14'].shift(1) >= 30), 'Signal'] = 1
            data.loc[(data['RSI14'] < 30) & (data['RSI14'].shift(1) >= 30), 'Signal_Type'] = "RSI 과매도 (매수 신호)"

            data.loc[(data['RSI14'] > 70) & (data['RSI14'].shift(1) <= 70), 'Signal'] = -1
            data.loc[(data['RSI14'] > 70) & (data['RSI14'].shift(1) <= 70), 'Signal_Type'] = "RSI 과매수 (매도 신호)"

            # 4. 볼린저 밴드 돌파
            data.loc[data['Close'] < data['BBL_20_2.0'], 'Signal'] = 1
            data.loc[data['Close'] < data['BBL_20_2.0'], 'Signal_Type'] = "볼린저 밴드 하단 돌파 (매수 신호)"

            data.loc[data['Close'] > data['BBU_20_2.0'], 'Signal'] = -1
            data.loc[data['Close'] > data['BBU_20_2.0'], 'Signal_Type'] = "볼린저 밴드 상단 돌파 (매도 신호)"

            # 5. 스토캐스틱 신호
            data.loc[(data['STOCHk_14_3_3'] > data['STOCHd_14_3_3']) & (
                    data['STOCHk_14_3_3'].shift(1) <= data['STOCHd_14_3_3'].shift(1)), 'Signal'] = 1
            data.loc[(data['STOCHk_14_3_3'] > data['STOCHd_14_3_3']) & (
                    data['STOCHk_14_3_3'].shift(1) <= data['STOCHd_14_3_3'].shift(1)), 'Signal_Type'] = "스토캐스틱 골든 크로스"

            data.loc[(data['STOCHk_14_3_3'] < data['STOCHd_14_3_3']) & (
                    data['STOCHk_14_3_3'].shift(1) >= data['STOCHd_14_3_3'].shift(1)), 'Signal'] = -1
            data.loc[(data['STOCHk_14_3_3'] < data['STOCHd_14_3_3']) & (
                    data['STOCHk_14_3_3'].shift(1) >= data['STOCHd_14_3_3'].shift(1)), 'Signal_Type'] = "스토캐스틱 데드 크로스"

            # 6. ADX 추세 강도 결합 신호
            data.loc[(data['ADX_14'] > 25) & (data['DMP_14'] > data['DMN_14']) & (
                    data['DMP_14'].shift(1) <= data['DMN_14'].shift(1)), 'Signal'] = 1
            data.loc[(data['ADX_14'] > 25) & (data['DMP_14'] > data['DMN_14']) & (
                    data['DMP_14'].shift(1) <= data['DMN_14'].shift(1)), 'Signal_Type'] = "ADX 강한 상승 추세"

            data.loc[(data['ADX_14'] > 25) & (data['DMP_14'] < data['DMN_14']) & (
                    data['DMP_14'].shift(1) >= data['DMN_14'].shift(1)), 'Signal'] = -1
            data.loc[(data['ADX_14'] > 25) & (data['DMP_14'] < data['DMN_14']) & (
                    data['DMP_14'].shift(1) >= data['DMN_14'].shift(1)), 'Signal_Type'] = "ADX 강한 하락 추세"

            # 복합 신호 강도 계산
            data['Buy_Strength'] = 0
            data['Sell_Strength'] = 0

            # 매수 강도 계산
            data.loc[data['SMA20'] > data['SMA50'], 'Buy_Strength'] += 1
            data.loc[data['MACD_12_26_9'] > data['MACDs_12_26_9'], 'Buy_Strength'] += 1
            data.loc[data['RSI14'] < 40, 'Buy_Strength'] += 1
            data.loc[data['Close'] < data['BBL_20_2.0'], 'Buy_Strength'] += 1
            data.loc[data['STOCHk_14_3_3'] > data['STOCHd_14_3_3'], 'Buy_Strength'] += 1
            data.loc[(data['ADX_14'] > 25) & (data['DMP_14'] > data['DMN_14']), 'Buy_Strength'] += 1

            # 매도 강도 계산
            data.loc[data['SMA20'] < data['SMA50'], 'Sell_Strength'] += 1
            data.loc[data['MACD_12_26_9'] < data['MACDs_12_26_9'], 'Sell_Strength'] += 1
            data.loc[data['RSI14'] > 60, 'Sell_Strength'] += 1
            data.loc[data['Close'] > data['BBU_20_2.0'], 'Sell_Strength'] += 1
            data.loc[data['STOCHk_14_3_3'] < data['STOCHd_14_3_3'], 'Sell_Strength'] += 1
            data.loc[(data['ADX_14'] > 25) & (data['DMP_14'] < data['DMN_14']), 'Sell_Strength'] += 1

            # 최종 신호 강도 계산 (0~100%)
            data['Buy_Signal_Strength'] = (data['Buy_Strength'] / 6) * 100
            data['Sell_Signal_Strength'] = (data['Sell_Strength'] / 6) * 100

            # 강한 신호 (80% 이상)만 별도 표시
            data['Strong_Signal'] = 0
            data.loc[data['Buy_Signal_Strength'] >= 80, 'Strong_Signal'] = 1
            data.loc[data['Sell_Signal_Strength'] >= 80, 'Strong_Signal'] = -1

            return data

        except Exception as e:
            logger.error(f"매매 신호 생성 중 오류 발생: {str(e)}")
            raise

    def _analyze_performance(self, data: pd.DataFrame) -> Dict:
        """
        생성된 매매 신호의 성과 분석

        Args:
            data: 매매 신호가 추가된 데이터프레임

        Returns:
            성과 지표를 담은 딕셔너리
        """
        try:
            # 매수/매도 신호 횟수
            buy_signals = data[data['Signal'] == 1].shape[0]
            sell_signals = data[data['Signal'] == -1].shape[0]
            strong_buy_signals = data[data['Strong_Signal'] == 1].shape[0]
            strong_sell_signals = data[data['Strong_Signal'] == -1].shape[0]

            # 신호 유형별 분석
            signal_types = data[data['Signal'] != 0]['Signal_Type'].value_counts().to_dict()

            # 가장 최근 신호
            latest_signals = data.iloc[-20:].copy()
            latest_buy = latest_signals[latest_signals['Signal'] == 1].iloc[-1:].to_dict('records') if len(
                latest_signals[latest_signals['Signal'] == 1]) > 0 else None
            latest_sell = latest_signals[latest_signals['Signal'] == -1].iloc[-1:].to_dict('records') if len(
                latest_signals[latest_signals['Signal'] == -1]) > 0 else None

            # 현재 신호 강도
            try:
                current_buy_strength = data['Buy_Signal_Strength'].iloc[-1]
                current_sell_strength = data['Sell_Signal_Strength'].iloc[-1]
            except IndexError:
                logger.warning("데이터 인덱싱 오류, 신호 강도에 기본값 사용")
                current_buy_strength = 0
                current_sell_strength = 0

            # 매매 추천
            try:
                if current_buy_strength >= 80:
                    recommendation = "강력 매수"
                elif current_buy_strength >= 60:
                    recommendation = "매수"
                elif current_sell_strength >= 80:
                    recommendation = "강력 매도"
                elif current_sell_strength >= 60:
                    recommendation = "매도"
                else:
                    recommendation = "관망"
            except Exception as e:
                logger.warning(f"매매 추천 생성 오류: {str(e)}, 기본값으로 대체")
                recommendation = "관망"

            # 시그널 히트맵 데이터
            try:
                last_row = data.iloc[-1]
                signal_heatmap = {
                    'SMA': 1 if last_row['SMA20'] > last_row['SMA50'] else -1,
                    'MACD': 1 if last_row['MACD_12_26_9'] > last_row['MACDs_12_26_9'] else -1,
                    'RSI': 1 if last_row['RSI14'] < 50 else -1,
                    'BB': 1 if last_row['Close'] < last_row['BBM_20_2.0'] else -1,
                    'Stoch': 1 if last_row['STOCHk_14_3_3'] > last_row['STOCHd_14_3_3'] else -1,
                    'ADX': 1 if (last_row['ADX_14'] > 25 and last_row['DMP_14'] > last_row['DMN_14']) else -1
                }
            except Exception as e:
                logger.warning(f"히트맵 데이터 생성 오류: {str(e)}, 기본값으로 대체")
                signal_heatmap = {
                    'SMA': 0, 'MACD': 0, 'RSI': 0,
                    'BB': 0, 'Stoch': 0, 'ADX': 0
                }

            # 데이터 길이에 따른 신뢰도 계산
            data_confidence = min(100, (len(data) / self.RECOMMENDED_MIN_DATA) * 100)
            if data_confidence < 50:
                data_confidence_level = "매우 낮음"
            elif data_confidence < 75:
                data_confidence_level = "낮음"
            elif data_confidence < 90:
                data_confidence_level = "보통"
            else:
                data_confidence_level = "높음"

            return {
                'buy_signals': buy_signals,
                'sell_signals': sell_signals,
                'strong_buy_signals': strong_buy_signals,
                'strong_sell_signals': strong_sell_signals,
                'signal_types': signal_types,
                'latest_buy': latest_buy,
                'latest_sell': latest_sell,
                'current_buy_strength': current_buy_strength,
                'current_sell_strength': current_sell_strength,
                'recommendation': recommendation,
                'signal_heatmap': signal_heatmap,
                'data_length': len(data),
                'data_confidence': data_confidence,
                'data_confidence_level': data_confidence_level
            }

        except Exception as e:
            logger.error(f"성과 분석 중 오류 발생: {str(e)}")
            # 오류 발생 시 최소한의 정보만 반환
            return {
                'buy_signals': 0,
                'sell_signals': 0,
                'strong_buy_signals': 0,
                'strong_sell_signals': 0,
                'signal_types': {},
                'latest_buy': None,
                'latest_sell': None,
                'current_buy_strength': 0,
                'current_sell_strength': 0,
                'recommendation': "데이터 오류",
                'signal_heatmap': {
                    'SMA': 0, 'MACD': 0, 'RSI': 0,
                    'BB': 0, 'Stoch': 0, 'ADX': 0
                },
                'data_length': len(data) if isinstance(data, pd.DataFrame) else 0,
                'data_confidence': 0,
                'data_confidence_level': "매우 낮음"
            }

    def visualize_signals(self, data: pd.DataFrame, window_size: int = 180) -> go.Figure:
        """
        매매 신호 시각화

        Args:
            data: 매매 신호가 포함된 데이터프레임
            window_size: 표시할 데이터 기간 (일)

        Returns:
            Plotly Figure 객체
        """
        try:
            # 데이터 길이 확인 및 경고
            if len(data) < self.ABSOLUTE_MIN_DATA:
                logger.warning(f"시각화 데이터 길이({len(data)}일)가 너무 짧아 신뢰성이 낮을 수 있습니다.")

            # 데이터 준비 (최신 n일 데이터만 사용)
            if len(data) > window_size:
                plot_data = data.iloc[-window_size:].copy()
            else:
                plot_data = data.copy()

            # 서브플롯 생성 (캔들차트, 지표 차트들)
            fig = make_subplots(
                rows=4, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.02,
                row_heights=[0.5, 0.2, 0.15, 0.15],
                subplot_titles=("가격 및 매매 신호", "MACD", "RSI", "스토캐스틱")
            )

            # 1. 캔들스틱 차트
            fig.add_trace(
                go.Candlestick(
                    x=plot_data.index,
                    open=plot_data['Open'],
                    high=plot_data['High'],
                    low=plot_data['Low'],
                    close=plot_data['Close'],
                    name="가격",
                    increasing_line_color='red',
                    decreasing_line_color='blue'
                ),
                row=1, col=1
            )

            # 이동평균선 추가
            fig.add_trace(
                go.Scatter(
                    x=plot_data.index,
                    y=plot_data['SMA20'],
                    name="SMA20",
                    line=dict(color='orange', width=1)
                ),
                row=1, col=1
            )

            fig.add_trace(
                go.Scatter(
                    x=plot_data.index,
                    y=plot_data['SMA50'],
                    name="SMA50",
                    line=dict(color='blue', width=1)
                ),
                row=1, col=1
            )

            fig.add_trace(
                go.Scatter(
                    x=plot_data.index,
                    y=plot_data['EMA9'],
                    name="EMA9",
                    line=dict(color='purple', width=1, dash='dot')
                ),
                row=1, col=1
            )

            # 볼린저 밴드 추가
            fig.add_trace(
                go.Scatter(
                    x=plot_data.index,
                    y=plot_data['BBU_20_2.0'],
                    name="BB Upper",
                    line=dict(color='rgba(0,176,246,0.2)', width=1),
                    showlegend=False
                ),
                row=1, col=1
            )

            fig.add_trace(
                go.Scatter(
                    x=plot_data.index,
                    y=plot_data['BBM_20_2.0'],
                    name="BB Middle",
                    line=dict(color='rgba(0,176,246,0.5)', width=1),
                    showlegend=False
                ),
                row=1, col=1
            )

            fig.add_trace(
                go.Scatter(
                    x=plot_data.index,
                    y=plot_data['BBL_20_2.0'],
                    name="BB Lower",
                    line=dict(color='rgba(0,176,246,0.2)', width=1),
                    fill='tonexty',
                    fillcolor='rgba(0,176,246,0.05)',
                    showlegend=False
                ),
                row=1, col=1
            )

            # 매수 신호 표시
            buy_signals = plot_data[plot_data['Signal'] == 1]
            fig.add_trace(
                go.Scatter(
                    x=buy_signals.index,
                    y=buy_signals['Low'] * 0.99,  # 약간 아래에 표시
                    mode='markers',
                    marker=dict(symbol='triangle-up', size=10, color='green'),
                    name="매수 신호"
                ),
                row=1, col=1
            )

            # 매도 신호 표시
            sell_signals = plot_data[plot_data['Signal'] == -1]
            fig.add_trace(
                go.Scatter(
                    x=sell_signals.index,
                    y=sell_signals['High'] * 1.01,  # 약간 위에 표시
                    mode='markers',
                    marker=dict(symbol='triangle-down', size=10, color='red'),
                    name="매도 신호"
                ),
                row=1, col=1
            )

            # 강한 매수 신호 표시
            strong_buy_signals = plot_data[plot_data['Strong_Signal'] == 1]
            fig.add_trace(
                go.Scatter(
                    x=strong_buy_signals.index,
                    y=strong_buy_signals['Low'] * 0.98,  # 아래에 표시
                    mode='markers',
                    marker=dict(symbol='triangle-up', size=15, color='darkgreen'),
                    name="강한 매수 신호"
                ),
                row=1, col=1
            )

            # 강한 매도 신호 표시
            strong_sell_signals = plot_data[plot_data['Strong_Signal'] == -1]
            fig.add_trace(
                go.Scatter(
                    x=strong_sell_signals.index,
                    y=strong_sell_signals['High'] * 1.02,  # 위에 표시
                    mode='markers',
                    marker=dict(symbol='triangle-down', size=15, color='darkred'),
                    name="강한 매도 신호"
                ),
                row=1, col=1
            )

            # 2. MACD 차트
            fig.add_trace(
                go.Scatter(
                    x=plot_data.index,
                    y=plot_data['MACD_12_26_9'],
                    name="MACD",
                    line=dict(color='blue', width=1.5)
                ),
                row=2, col=1
            )

            fig.add_trace(
                go.Scatter(
                    x=plot_data.index,
                    y=plot_data['MACDs_12_26_9'],
                    name="MACD Signal",
                    line=dict(color='red', width=1)
                ),
                row=2, col=1
            )

            # MACD 히스토그램
            colors = ['green' if val > 0 else 'red' for val in plot_data['MACDh_12_26_9']]
            fig.add_trace(
                go.Bar(
                    x=plot_data.index,
                    y=plot_data['MACDh_12_26_9'],
                    name="MACD Histogram",
                    marker_color=colors
                ),
                row=2, col=1
            )

            # 0선 추가
            fig.add_hline(y=0, line_width=1, line_dash="dot", line_color="gray", row=2, col=1)

            # 3. RSI 차트
            fig.add_trace(
                go.Scatter(
                    x=plot_data.index,
                    y=plot_data['RSI14'],
                    name="RSI14",
                    line=dict(color='purple', width=1.5)
                ),
                row=3, col=1
            )

            # RSI 기준선 추가
            fig.add_hline(y=70, line_width=1, line_dash="dot", line_color="red", row=3, col=1)
            fig.add_hline(y=30, line_width=1, line_dash="dot", line_color="green", row=3, col=1)
            fig.add_hline(y=50, line_width=1, line_dash="dot", line_color="gray", row=3, col=1)

            # 4. 스토캐스틱 차트
            fig.add_trace(
                go.Scatter(
                    x=plot_data.index,
                    y=plot_data['STOCHk_14_3_3'],
                    name="%K",
                    line=dict(color='blue', width=1.5)
                ),
                row=4, col=1
            )

            fig.add_trace(
                go.Scatter(
                    x=plot_data.index,
                    y=plot_data['STOCHd_14_3_3'],
                    name="%D",
                    line=dict(color='red', width=1)
                ),
                row=4, col=1
            )

            # 스토캐스틱 기준선 추가
            fig.add_hline(y=80, line_width=1, line_dash="dot", line_color="red", row=4, col=1)
            fig.add_hline(y=20, line_width=1, line_dash="dot", line_color="green", row=4, col=1)

            # 레이아웃 업데이트
            fig.update_layout(
                title="기술적 지표 기반 매매 신호",
                height=900,
                xaxis_rangeslider_visible=False,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )

            # Y축 타이틀 추가
            fig.update_yaxes(title_text="가격", row=1, col=1)
            fig.update_yaxes(title_text="MACD", row=2, col=1)
            fig.update_yaxes(title_text="RSI", row=3, col=1)
            fig.update_yaxes(title_text="Stoch", row=4, col=1)

            # 데이터 부족 표시 워터마크 추가
            if len(data) < self.RECOMMENDED_MIN_DATA:
                # 데이터가 권장량보다 적은 경우 워터마크 추가
                confidence_text = f"데이터 신뢰도: {int(len(data) / self.RECOMMENDED_MIN_DATA * 100)}% (권장 {self.RECOMMENDED_MIN_DATA}일)"
                fig.add_annotation(
                    text=confidence_text,
                    x=0.5,
                    y=0.5,
                    xref="paper",
                    yref="paper",
                    showarrow=False,
                    font=dict(size=20, color="rgba(150,150,150,0.5)"),
                    textangle=-30
                )

            return fig

        except Exception as e:
            logger.error(f"매매 신호 시각화 중 오류 발생: {str(e)}")
            # 오류 발생 시 빈 그래프 반환
            fig = go.Figure()
            fig.update_layout(
                title="시각화 오류 발생",
                annotations=[
                    dict(
                        text=f"시각화 중 오류 발생: {str(e)}",
                        xref="paper",
                        yref="paper",
                        x=0.5,
                        y=0.5,
                        showarrow=False
                    )
                ]
            )
            return fig