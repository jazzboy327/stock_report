# src/utils/stock_price_predictor_improved.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LSTM, Dropout, Input, MultiHeadAttention, LayerNormalization, Average, \
    GlobalAveragePooling1D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestRegressor
import logging
import os
from datetime import datetime, timedelta
import joblib
from pathlib import Path
import streamlit as st
import re

logger = logging.getLogger('StockAnalysisApp.PricePredictor')


class StockPricePredictor:
    """개선된 주식 가격 예측 클래스"""

    def __init__(self):
        """예측기 초기화"""
        self.model = None
        self.scaler = None
        self.model_path = Path(__file__).resolve().parent.parent.parent / 'models'
        self.model_path.mkdir(parents=True, exist_ok=True)

        # 예측 기간 조정
        self.prediction_days = 90  # 이전 120에서 90으로 축소
        self.forecast_days = 30  # 기본 예측 기간

        self.cached_prediction = None
        self.cached_settings = None
        self.is_trained = False  # 학습 상태 추적
        self.training_params = None  # 학습 파라미터 저장
        self.feature_importance = None  # 특성 중요도 저장

        # 학습 매개변수 개선
        self.default_batch_size = 32
        self.default_epochs = 150  # 이전 200에서 150으로 축소
        self.default_patience = 15  # 이전 20에서 15로 축소

        # 기본 학습률
        self.default_lr = 0.001

        # 자동 특성 선택 임계값 조정 (높여서 더 적은 특성 선택)
        self.auto_feature_threshold = 0.05  # 이전 0.03에서 0.05로 증가

        # 정규화 강화
        self.l2_lambda = 0.001  # L2 정규화 계수 추가

    def preprocess_data(self, stock_data, forecast_days=30, train_size=0.8, features=None, use_advanced_features=True):
        """
        데이터 전처리 - 고급 특성 포함 옵션 추가
        """
        try:
            # 최소 데이터 길이 확인
            MIN_REQUIRED_DATA = max(60, self.prediction_days * 2)  # 최소 60일 또는 예측일의 2배

            if len(stock_data) < MIN_REQUIRED_DATA:
                error_msg = f"데이터가 부족합니다. 최소 {MIN_REQUIRED_DATA}일 이상의 데이터가 필요합니다. (현재: {len(stock_data)}일)"
                logger.error(error_msg)
                raise ValueError(error_msg)

            # 특성 선택
            features = features or ['Close']

            # 기술적 지표 계산 및 추가 - 고급 특성 포함 여부에 따라 다른 메서드 호출
            if use_advanced_features:
                logger.info("고급 금융 특성을 포함하여 전처리합니다.")
                df = self._add_technical_indicators_enhanced(stock_data)
            else:
                logger.info("기본 기술적 지표만 사용하여 전처리합니다.")
                df = self._add_technical_indicators(stock_data)

            # 데이터 길이 확인 및 로깅
            original_length = len(df)
            logger.info(f"원본 데이터 길이: {original_length}일")

            # 사용 가능한 전체 특성 목록 로깅
            logger.info(f"사용 가능한 전체 특성 목록 ({len(df.columns)}개): {df.columns.tolist()}")

            # 데이터가 self.prediction_days보다 짧은 경우 처리
            if original_length < self.prediction_days:
                logger.warning(f"데이터가 필요한 길이({self.prediction_days}일)보다 짧습니다.")
                error_msg = f"데이터 길이({original_length}일)가 예측 기간({self.prediction_days}일)보다 짧습니다."
                logger.error(error_msg)
                raise ValueError(error_msg)

            # 선택된 특성만 사용
            if features != ['All']:
                selected_columns = []
                for feature in features:
                    # 기본 컬럼 또는 파생 컬럼인 경우 포함
                    if feature in df.columns:
                        selected_columns.append(feature)

                # 최소한 종가는 포함되어야 함
                if 'Close' not in selected_columns:
                    selected_columns.append('Close')

                logger.info(f"선택된 특성: {selected_columns}")
                df = df[selected_columns]

            # 결측치 확인 및 처리
            if df.isnull().any().any():
                null_counts = df.isnull().sum()
                logger.warning(f"결측치 발견: {null_counts[null_counts > 0]}")
                logger.info("전방 채우기(ffill) 및 후방 채우기(bfill)로 결측치 처리 중...")
                df = df.ffill().bfill()

            # 견고한 스케일러 사용 (이상치에 덜 민감함)
            self.scaler = RobustScaler()
            scaled_data = self.scaler.fit_transform(df)

            # 데이터 형태 로깅
            logger.info(f"스케일된 데이터 형태: {scaled_data.shape}")

            # 학습 데이터 생성
            X = []
            y = []

            # 시퀀스 데이터 생성 (스텝 조정)
            try:
                for i in range(self.prediction_days, len(scaled_data) - forecast_days, 3):  # 스텝 크기 3으로 증가
                    X.append(scaled_data[i - self.prediction_days:i])
                    # Close 가격만 예측 (항상 첫 번째 컬럼이 Close가 되도록 설정)
                    close_idx = df.columns.get_loc('Close')
                    y.append(scaled_data[i:i + forecast_days, close_idx])
            except Exception as seq_error:
                logger.error(f"시퀀스 데이터 생성 중 오류: {str(seq_error)}")
                raise ValueError(f"시퀀스 데이터 생성 실패: {str(seq_error)}")

            # 데이터 충분성 확인
            if len(X) == 0 or len(y) == 0:
                error_msg = f"생성된 학습 시퀀스가 없습니다. 데이터 길이({len(scaled_data)}일)가 충분하지 않습니다."
                logger.error(error_msg)
                raise ValueError(error_msg)

            X, y = np.array(X), np.array(y)

            # 배열 차원 확인
            if len(X.shape) < 3:
                logger.warning(f"X 배열 차원이 올바르지 않습니다: {X.shape}. 재구성 시도 중...")
                try:
                    # 3차원으로 변환 시도 (샘플 수, 시퀀스 길이, 특성 수)
                    if len(X.shape) == 2:
                        # 이미 2차원인 경우, 차원 추가
                        X = X.reshape(X.shape[0], X.shape[1], 1)
                        logger.info(f"X 배열 재구성 완료: {X.shape}")
                    else:
                        error_msg = f"X 배열 차원({X.shape})이 예상과 다릅니다. 3차원 배열이 필요합니다."
                        logger.error(error_msg)
                        raise ValueError(error_msg)
                except Exception as reshape_error:
                    logger.error(f"배열 재구성 중 오류: {str(reshape_error)}")
                    raise ValueError(f"데이터 형태 조정 실패: {str(reshape_error)}")

            # 데이터 형태 로깅
            logger.info(f"입력 데이터 X 형태: {X.shape}, 출력 데이터 y 형태: {y.shape}")

            # 설정된 비율로 데이터 분할 (시간 순서 유지)
            train_size = int(len(X) * train_size)
            if train_size == 0:
                logger.warning("학습 데이터가 너무 적어 train_size가 0이 됩니다. 최소 1로 설정합니다.")
                train_size = 1

            # 분할 시 데이터가 충분한지 확인
            if train_size >= len(X):
                train_size = max(1, len(X) - 1)  # 최소 1개는 테스트용으로 남김

            if train_size < 1 or len(X) - train_size < 1:
                error_msg = f"데이터 분할 후 학습/테스트 세트가 비어 있습니다. (전체: {len(X)}, 학습: {train_size})"
                logger.error(error_msg)
                raise ValueError(error_msg)

            X_train, X_test = X[:train_size], X[train_size:]
            y_train, y_test = y[:train_size], y[train_size:]

            logger.info(f"학습 데이터 형태: X_train {X_train.shape}, y_train {y_train.shape}")
            logger.info(f"테스트 데이터 형태: X_test {X_test.shape}, y_test {y_test.shape}")

            return X_train, y_train, X_test, y_test

        except Exception as e:
            logger.error(f"데이터 전처리 중 오류 발생: {str(e)}", exc_info=True)
            raise

    def _add_technical_indicators(self, df):
        """
        기술적 지표 계산 및 추가 - 중요 지표 위주로 축소
        """
        df_with_indicators = df.copy()

        # 중요한 이동평균만 추가
        df_with_indicators['MA5'] = df_with_indicators['Close'].rolling(window=5).mean()
        df_with_indicators['MA20'] = df_with_indicators['Close'].rolling(window=20).mean()
        df_with_indicators['MA60'] = df_with_indicators['Close'].rolling(window=60).mean()

        # VWAP (거래량 가중 평균 가격)
        df_with_indicators['VWAP'] = (df_with_indicators['Close'] * df_with_indicators['Volume']).rolling(
            window=20).sum() / df_with_indicators['Volume'].rolling(window=20).sum()

        # RSI 계산 (14일)
        delta = df_with_indicators['Close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
        rs = gain / loss
        df_with_indicators['RSI'] = 100 - (100 / (1 + rs))

        # MACD
        ema12 = df_with_indicators['Close'].ewm(span=12, adjust=False).mean()
        ema26 = df_with_indicators['Close'].ewm(span=26, adjust=False).mean()
        df_with_indicators['MACD'] = ema12 - ema26
        df_with_indicators['MACD_Signal'] = df_with_indicators['MACD'].ewm(span=9, adjust=False).mean()

        # 볼린저 밴드
        df_with_indicators['BB_Middle'] = df_with_indicators['Close'].rolling(window=20).mean()
        std = df_with_indicators['Close'].rolling(window=20).std()
        df_with_indicators['BB_Width'] = (df_with_indicators['BB_Middle'] + 2 * std - 
                                         (df_with_indicators['BB_Middle'] - 2 * std)) / df_with_indicators['BB_Middle']

        # 가격 변화율 (핵심 기간만)
        df_with_indicators['Price_Change_1D'] = df_with_indicators['Close'].pct_change(1)
        df_with_indicators['Price_Change_5D'] = df_with_indicators['Close'].pct_change(5)

        # 거래량 변화율
        df_with_indicators['Volume_Ratio'] = df_with_indicators['Volume'] / df_with_indicators['Volume'].rolling(
            window=20).mean()

        # 결측치 처리
        df_with_indicators = df_with_indicators.bfill().ffill()

        return df_with_indicators

    def add_advanced_financial_features(self, df):
        """금융 데이터 특화 고급 특성 추가"""
        logger.info("고급 금융 특성 추가 시작...")
        df_new = df.copy()

        # 1. 변동성 관련 특성
        df_new['Volatility_10D'] = df_new['Close'].rolling(window=10).std() / df_new['Close'].rolling(window=10).mean()

        # 2. 거래량 기반 특성
        df_new['Volume_Price_Trend'] = (df_new['Volume'] * (df_new['Close'] - df_new['Close'].shift(1))).rolling(
            window=14).sum()

        # 3. VWAP (Volume Weighted Average Price) 추가 기간
        df_new['VWAP_5D'] = (df_new['Close'] * df_new['Volume']).rolling(window=5).sum() / df_new['Volume'].rolling(
            window=5).sum()

        # 4. 시장 요일 효과 (계절성)
        df_new['DayOfWeek'] = df_new.index.dayofweek
        for i in range(5):  # 월-금 (0-4)
            df_new[f'Is_Day_{i}'] = (df_new['DayOfWeek'] == i).astype(int)

        # 5. Zigzag 지표 (추세 반전)
        def calculate_zigzag(prices, min_change=0.03):
            zigzag = np.zeros_like(prices)
            last_extreme_price = prices[0]
            last_extreme_idx = 0
            uptrend = True

            for i in range(1, len(prices)):
                price_change = (prices[i] - last_extreme_price) / last_extreme_price

                if uptrend and price_change <= -min_change:
                    # 하락 전환점
                    zigzag[last_extreme_idx] = 1  # 이전 고점
                    uptrend = False
                    last_extreme_price = prices[i]
                    last_extreme_idx = i
                elif not uptrend and price_change >= min_change:
                    # 상승 전환점
                    zigzag[last_extreme_idx] = -1  # 이전 저점
                    uptrend = True
                    last_extreme_price = prices[i]
                    last_extreme_idx = i

            return zigzag

        df_new['ZigZag'] = calculate_zigzag(df_new['Close'].values)

        # 6. 이동평균 교차 신호
        if 'MA5' in df_new.columns and 'MA20' in df_new.columns:
            df_new['MA_Cross_Signal'] = np.where(
                df_new['MA5'] > df_new['MA20'],
                1,  # 골든 크로스
                np.where(
                    df_new['MA5'] < df_new['MA20'],
                    -1,  # 데드 크로스
                    0  # 무변화
                )
            )

        # 7. ATR (Average True Range)
        if 'High' in df_new.columns and 'Low' in df_new.columns:
            high_low = df_new['High'] - df_new['Low']
            high_close = np.abs(df_new['High'] - df_new['Close'].shift())
            low_close = np.abs(df_new['Low'] - df_new['Close'].shift())

            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            df_new['ATR_14'] = true_range.rolling(14).mean()

        # 8. 가격 모멘텀 지표
        df_new['Momentum_14D'] = df_new['Close'] / df_new['Close'].shift(14) - 1

        # 9. 거래량 변화율
        df_new['Volume_Change_5D'] = df_new['Volume'] / df_new['Volume'].shift(5) - 1

        # 10. 가격 추세 강도 지표
        def calculate_adx(df, period=14):
            """
            ADX(Average Directional Index) 계산
            추세의 강도를 측정하는 지표 (0-100)
            """
            try:
                if 'High' not in df.columns or 'Low' not in df.columns:
                    return np.zeros(len(df))

                # True Range 계산
                tr1 = np.abs(df['High'] - df['Low'])
                tr2 = np.abs(df['High'] - df['Close'].shift())
                tr3 = np.abs(df['Low'] - df['Close'].shift())
                true_range = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)

                # +DM, -DM 계산
                up_move = df['High'] - df['High'].shift()
                down_move = df['Low'].shift() - df['Low']

                plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
                minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)

                # Exponential Moving Average 계산
                tr_ema = true_range.ewm(alpha=1 / period, adjust=False).mean()
                plus_di = 100 * (pd.Series(plus_dm).ewm(alpha=1 / period, adjust=False).mean() / tr_ema)
                minus_di = 100 * (pd.Series(minus_dm).ewm(alpha=1 / period, adjust=False).mean() / tr_ema)

                # Directional Index 계산
                dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)

                # Average Directional Index 계산
                adx = dx.ewm(alpha=1 / period, adjust=False).mean()

                return adx
            except Exception as e:
                logger.warning(f"ADX 계산 중 오류: {str(e)}")
                return np.zeros(len(df))

        df_new['ADX'] = calculate_adx(df_new)

        # 11. 가격 캔들 패턴 (상승/하락 추세 반전 패턴)
        if all(x in df_new.columns for x in ['Open', 'High', 'Low', 'Close']):
            # 도지 패턴 (몸통이 작고 그림자가 긴 캔들)
            df_new['Doji'] = (
                        (np.abs(df_new['Close'] - df_new['Open']) / (df_new['High'] - df_new['Low'])) < 0.1).astype(int)

            # 망치 패턴 (하단 그림자가 긴 캔들)
            df_new['Hammer'] = (
                    ((df_new['High'] - df_new['Low']) > 3 * (df_new['Open'] - df_new['Close'])) &
                    ((df_new['Close'] - df_new['Low']) / (0.001 + df_new['High'] - df_new['Low']) > 0.6) &
                    ((df_new['Open'] - df_new['Low']) / (0.001 + df_new['High'] - df_new['Low']) > 0.6)
            ).astype(int)

        # 12. 현재 거래일이 월말/월초인지 여부
        df_new['MonthStart'] = df_new.index.is_month_start.astype(int)
        df_new['MonthEnd'] = df_new.index.is_month_end.astype(int)

        # 13. 거래량 가격 비율 (금액/거래량)
        df_new['Price_Volume_Ratio'] = df_new['Close'] / (df_new['Volume'] + 1)  # 0으로 나누기 방지

        # 결측치 처리
        df_new = df_new.fillna(method='ffill').fillna(method='bfill')

        logger.info(f"고급 특성 {len(df_new.columns) - len(df)} 개 추가 완료")
        return df_new

    def _add_technical_indicators_enhanced(self, df):
        """
        기존 기술적 지표 계산에 고급 금융 특성을 추가한 개선된 버전
        """
        # 기존 기술적 지표 계산
        df_with_indicators = self._add_technical_indicators(df)

        # 고급 금융 특성 추가
        df_enhanced = self.add_advanced_financial_features(df_with_indicators)

        return df_enhanced

    def auto_feature_selection(self, stock_data, target_column='Close', n_features=None):
        """
        자동 특성 선택 알고리즘 - 결측치 문제를 해결하고 고급 금융 특성을 포함하도록 개선된 버전
        """
        try:
            # 기술적 지표 추가 - 고급 금융 특성 포함
            logger.info("자동 특성 선택을 위해 고급 금융 특성 계산 중...")
            df_with_indicators = self._add_technical_indicators_enhanced(stock_data)

            # 계산된 특성 목록 로깅
            logger.info(
                f"자동 특성 선택을 위한 전체 특성 목록 ({len(df_with_indicators.columns)}개): {df_with_indicators.columns.tolist()}")

            # 결측치 확인
            na_counts = df_with_indicators.isna().sum()
            has_na = na_counts.sum() > 0

            if has_na:
                # 결측치가 있는 열들과 개수 로깅
                na_columns = na_counts[na_counts > 0]
                logger.info(f"결측치가 있는 열 및 개수: {na_columns.to_dict()}")

                # 결측치 비율이 높은 열 제외 (예: 30% 이상)
                high_na_cols = na_counts[na_counts > len(df_with_indicators) * 0.3].index.tolist()
                if high_na_cols:
                    logger.info(f"결측치 비율이 높은 다음 열들은 제외합니다: {high_na_cols}")
                    df_with_indicators = df_with_indicators.drop(columns=high_na_cols)

                # 전체 결측치를 적절히 처리 - ffill 후 bfill
                logger.info("나머지 결측치는 전방/후방 채우기로 처리합니다.")
                df_with_indicators = df_with_indicators.ffill().bfill()

            # 결측치 처리 후 데이터 크기 로깅
            logger.info(f"결측치 처리 후 데이터 크기: {df_with_indicators.shape}")

            # 데이터가 충분한지 확인
            if len(df_with_indicators) < 60:  # 최소 데이터 개수 확인
                logger.warning("자동 특성 선택을 위한 데이터가 부족합니다. 기본 특성을 사용합니다.")
                return ['Close', 'MA20', 'RSI', 'MACD', 'Volume_Ratio']  # 확장된 기본 특성 세트 반환

            # 1. 상관관계 기반 특성 선택
            # 임계값을 0.05에서 0.01로 낮춰서 더 많은 특성이 선택되도록 함
            threshold = 0.01

            # 상관관계 계산 시 오류 처리
            try:
                correlations = df_with_indicators.corr()[target_column].abs()
                # 상관관계가 NaN인 열 처리
                correlations = correlations.fillna(0)
            except Exception as corr_error:
                logger.warning(f"상관관계 계산 중 오류: {str(corr_error)}")
                # 각 특성과 타겟 간의 상관관계를 개별적으로 계산
                correlations = {}
                for col in df_with_indicators.columns:
                    if col != target_column:
                        try:
                            corr = df_with_indicators[col].corr(df_with_indicators[target_column])
                            if not pd.isna(corr):
                                correlations[col] = abs(corr)
                            else:
                                correlations[col] = 0
                        except:
                            correlations[col] = 0

                # 딕셔너리를 Series로 변환
                correlations = pd.Series(correlations)

            # 상관관계 상위 특성들 로깅
            top_corr = correlations.sort_values(ascending=False).head(15)
            logger.info(f"상관관계 상위 15개 특성: \n{top_corr}")

            high_corr_features = correlations[correlations > threshold].index.tolist()

            if target_column in high_corr_features:
                high_corr_features.remove(target_column)

            # 상관관계 선택 결과 로깅
            logger.info(f"상관관계 기반 1차 필터링 결과: {len(high_corr_features)}개 특성 선택됨")
            if high_corr_features:
                logger.info(f"선택된 특성: {high_corr_features[:15]}{'...' if len(high_corr_features) > 15 else ''}")
            else:
                logger.warning("상관관계 기반 특성이 없습니다. 임계값을 낮춥니다.")

            # 첫번째 필터로 특성이 충분하지 않으면 임계값 낮춤
            if len(high_corr_features) < 5:  # 최소 5개 특성 확보
                lower_threshold = 0.005  # 더 낮은 임계값
                logger.info(f"선택된 특성이 적어 임계값을 {lower_threshold}로 낮춤")
                high_corr_features = correlations[correlations > lower_threshold].index.tolist()
                if target_column in high_corr_features:
                    high_corr_features.remove(target_column)
                logger.info(f"임계값 낮춘 후 1차 필터링 결과: {len(high_corr_features)}개 특성 선택됨")

                # 그래도 특성이 부족하면 상위 5개 강제 선택
                if len(high_corr_features) < 5:
                    logger.info("상관관계 기반 특성이 부족하여 상위 5개 특성 강제 선택")
                    # 상관관계가 0보다 큰 특성만 포함
                    positive_corr = correlations[correlations > 0].sort_values(ascending=False)
                    # 적어도 하나의 상관관계가 있는지 확인
                    if len(positive_corr) > 1:  # 타겟 제외하고 최소 1개
                        high_corr_features = positive_corr.iloc[1:min(6, len(positive_corr))].index.tolist()
                    else:
                        # 상관관계가 없으면 기본 특성 사용
                        logger.warning("유효한 상관관계가 없습니다. 기본 특성을 추가합니다.")
                        candidate_features = ['MA5', 'MA20', 'RSI', 'MACD', 'Volume_Ratio', 'Price_Change_1D',
                                              'BB_Width']
                        high_corr_features = [f for f in candidate_features if f in df_with_indicators.columns]

            # 너무 많은 특성이 선택된 경우 상위 20개만 유지
            if len(high_corr_features) > 20:
                logger.info(f"너무 많은 특성({len(high_corr_features)}개)이 선택되었습니다. 상위 20개만 유지합니다.")
                # 상관관계 기준으로 정렬하여 상위 20개 선택
                sorted_features = correlations.sort_values(ascending=False).index.tolist()
                # 타겟 제외하고 상위 20개
                sorted_features = [f for f in sorted_features if f != target_column and f in high_corr_features][:20]
                high_corr_features = sorted_features

            # 2. 랜덤 포레스트 기반 특성 중요도 분석 (2차 필터링)
            try:
                # 고급 특성이 너무 적으면 랜덤 포레스트 단계 건너뛰기
                if len(high_corr_features) <= 3:
                    logger.warning("선택된 특성이 3개 이하입니다. 랜덤 포레스트 분석을 건너뜁니다.")
                    raise ValueError("선택된 특성 부족")

                # 목표변수 준비 (미래 n일 후 가격 변화율)
                future_days = 5  # 5일 후 가격 변화율 예측
                # 데이터가 충분한지 확인
                if len(df_with_indicators) <= future_days:
                    logger.warning(f"데이터가 부족합니다({len(df_with_indicators)}일). 랜덤 포레스트 분석을 건너뜁니다.")
                    raise ValueError("데이터 부족")

                y = df_with_indicators[target_column].shift(-future_days).dropna()
                price_prev = df_with_indicators[target_column].iloc[:-future_days]

                # 변화율 계산에서 0으로 나누기 방지
                y = ((y.values / price_prev.values) - 1) * 100  # 변화율(%)

                # 특성 준비
                X = df_with_indicators.iloc[:-future_days][high_corr_features]

                # 데이터 크기 로깅
                logger.info(f"랜덤 포레스트 학습 데이터 크기: X({X.shape}), y({len(y)})")

                # 랜덤 포레스트 모델 학습
                model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)  # 병렬 처리 활성화
                model.fit(X, y)

                # 특성 중요도 계산
                feature_importance = pd.DataFrame({
                    'feature': high_corr_features,
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=False)

                # 특성 중요도 저장
                self.feature_importance = feature_importance

                # 전체 중요도 목록 로깅
                logger.info("랜덤 포레스트 특성 중요도:")
                for idx, row in feature_importance.iterrows():
                    logger.info(f"  {row['feature']}: {row['importance']:.4f}")

                # 특성 중요도에 따른 선택 - 최대 n_features개로 제한
                max_features = 10  # 최대 특성 수를 10개로 증가
                if n_features:
                    # 지정된 수의 특성 선택, 최대 max_features개로 제한
                    max_features = min(n_features, max_features)

                # 최소 중요도 임계값 설정 (0에 가까운 특성 제외)
                min_importance = 0.01
                valid_features = feature_importance[feature_importance['importance'] > min_importance]

                # 유효한 특성이 적으면 임계값 낮춤
                if len(valid_features) < 3:
                    logger.warning(f"중요도 {min_importance} 이상인 특성이 부족합니다. 임계값을 낮춥니다.")
                    min_importance = 0.001
                    valid_features = feature_importance[feature_importance['importance'] > min_importance]

                    # 그래도 특성이 적으면 모든 특성 사용
                    if len(valid_features) < 3:
                        logger.warning("유효한 특성 중요도가 적습니다. 전체 특성 중요도를 사용합니다.")
                        valid_features = feature_importance

                # 상위 특성 선택
                top_features = valid_features['feature'][:min(max_features, len(valid_features))].tolist()

                logger.info(f"랜덤 포레스트 중요도 기반 2차 필터링 결과: {len(top_features)}개 특성 선택됨")
                logger.info(f"선택된 특성: {top_features}")

                selected_features = top_features

                # 특성이 너무 적으면 상관관계 기반으로 추가
                if len(selected_features) < 3:
                    logger.warning("선택된 특성이 너무 적습니다. 상관관계 기반 특성으로 보완합니다.")
                    additional_features = [f for f in high_corr_features if f not in selected_features]
                    selected_features.extend(additional_features[:min(5, len(additional_features))])
                    logger.info(f"추가된 특성: {additional_features[:min(5, len(additional_features))]}")

            except Exception as rf_error:
                logger.warning(f"랜덤 포레스트 기반 특성 선택 중 오류: {str(rf_error)}")
                logger.info("1차 필터링 결과만 사용합니다.")
                # 1차 필터링 결과에서도 최대 10개만 선택
                selected_features = high_corr_features[:min(10, len(high_corr_features))]

            # 기본 특성 확인 및 추가 (종가는 반드시 포함)
            if 'Close' not in selected_features and 'Close' in df_with_indicators.columns:
                selected_features.insert(0, 'Close')  # 맨 앞에 추가
                logger.info("기본 특성 'Close'를 추가했습니다.")

            # 중요한 기술적 지표가 빠졌다면 추가
            important_indicators = ['MA20', 'RSI']
            for indicator in important_indicators:
                if indicator not in selected_features and indicator in df_with_indicators.columns:
                    selected_features.append(indicator)
                    logger.info(f"중요한 기술적 지표 '{indicator}'를 추가했습니다.")

            # 선택된 특성이 없거나 매우 적을 경우
            if not selected_features or len(selected_features) < 3:
                logger.warning("선택된 특성이 너무 적습니다. 확장된 기본 특성 세트를 사용합니다.")
                default_features = ['Close', 'MA20', 'RSI', 'MACD', 'Volume_Ratio']
                # 데이터프레임에 있는 특성만 선택
                selected_features = [f for f in default_features if f in df_with_indicators.columns]
                # 여전히 부족하면 추가 특성 고려
                if len(selected_features) < 3:
                    extra_features = ['MA5', 'MA60', 'BB_Width', 'Price_Change_1D', 'Momentum_14D']
                    for f in extra_features:
                        if f in df_with_indicators.columns and f not in selected_features:
                            selected_features.append(f)
                            if len(selected_features) >= 5:
                                break

            logger.info(f"최종 선택된 특성: {selected_features}")
            return selected_features

        except Exception as e:
            logger.error(f"자동 특성 선택 중 오류 발생: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            # 오류 발생 시 최소 기본 특성 반환
            return ['Close', 'MA20', 'RSI', 'MACD', 'Volume_Ratio']  # 기본 특성 확장

    def build_lstm_model(self, input_shape, forecast_days):
        """개선된 LSTM 모델 구축 - 레이어 수와 유닛 수 감소"""
        model = Sequential()

        # 입력 레이어: 첫 번째 LSTM 레이어
        model.add(LSTM(units=64, return_sequences=True, input_shape=input_shape,
                       recurrent_dropout=0.1, kernel_regularizer=l2(self.l2_lambda)))
        model.add(Dropout(0.2))
        model.add(BatchNormalization())

        # 중간 레이어: 1개만 사용 (레이어 감소)
        model.add(LSTM(units=48, return_sequences=True, kernel_regularizer=l2(self.l2_lambda)))
        model.add(Dropout(0.2))
        model.add(BatchNormalization())

        # 마지막 LSTM 레이어
        model.add(LSTM(units=32, kernel_regularizer=l2(self.l2_lambda)))
        model.add(Dropout(0.2))

        # 출력 전 완전 연결 레이어 (감소)
        model.add(Dense(units=32, activation='relu', kernel_regularizer=l2(self.l2_lambda)))
        model.add(Dense(units=forecast_days))

        # 학습률 스케줄러가 있는 Adam 옵티마이저 사용
        optimizer = Adam(learning_rate=self.default_lr)
        model.compile(optimizer=optimizer, loss='huber')  # Huber 손실 사용 (MSE보다 이상치에 강함)

        return model

    def build_transformer_model(self, input_shape, forecast_days):
        """
        수정된 Transformer 모델 구축 - 더 간단한 위치 인코딩 방식 사용
        """
        from tensorflow.keras.layers import Lambda
        # 위치 인코딩 함수 정의 - 클래스 대신 함수 사용
        def positional_encoding(inputs):
            batch_size = tf.shape(inputs)[0]
            seq_length = tf.shape(inputs)[1]
            d_model = tf.shape(inputs)[2]

            # 위치 인덱스 생성
            position = tf.cast(tf.range(seq_length), tf.float32)
            position = tf.reshape(position, [1, seq_length, 1])

            # 차원 인덱스 생성
            dim = tf.cast(tf.range(d_model), tf.float32)
            dim = tf.reshape(dim, [1, 1, d_model])

            # 짝수/홀수 차원에 따른 각도 계산
            angle = position / tf.pow(10000.0, (2.0 * (dim // 2.0)) / tf.cast(d_model, tf.float32))

            # sin, cos 적용
            encoding = tf.where(
                tf.cast(tf.math.floormod(dim, 2.0), tf.bool),  # 홀수 차원
                tf.sin(angle),  # sin 적용
                tf.cos(angle)  # 짝수 차원에 cos 적용
            )

            # 배치 차원으로 확장
            encoding = tf.tile(encoding, [batch_size, 1, 1])

            return inputs + encoding

        def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
            # 멀티헤드 어텐션
            x = LayerNormalization(epsilon=1e-6)(inputs)
            attention_output = MultiHeadAttention(
                key_dim=head_size, num_heads=num_heads, dropout=dropout
            )(x, x)
            attention_output = Dropout(dropout)(attention_output)
            res = attention_output + inputs

            # 피드 포워드 네트워크
            x = LayerNormalization(epsilon=1e-6)(res)
            x = Dense(ff_dim, activation="gelu", kernel_regularizer=l2(self.l2_lambda))(x)
            x = Dropout(dropout)(x)
            x = Dense(inputs.shape[-1], kernel_regularizer=l2(self.l2_lambda))(x)
            x = Dropout(dropout)(x)

            return x + res

        # 개선된 모델 파라미터 설정
        head_size = 64
        num_heads = 4  # 8에서 4로 감소
        ff_dim = 256
        encoder_layers = 3  # 6에서 3으로 감소
        dropout_rate = 0.2

        # 모델 구축
        inputs = Input(shape=input_shape)

        # 시간적 인코딩 추가 (클래스 대신 함수 사용)
        x = Lambda(positional_encoding)(inputs)

        # 트랜스포머 인코더 블록 (레이어 수 감소)
        for _ in range(encoder_layers):
            x = transformer_encoder(
                x, head_size=head_size, num_heads=num_heads,
                ff_dim=ff_dim, dropout=dropout_rate
            )

        # 시퀀스 처리 후 특성 추출
        x = GlobalAveragePooling1D()(x)

        # 완전 연결 레이어 (축소)
        x = Dense(64, activation="relu", kernel_regularizer=l2(self.l2_lambda))(x)
        x = Dropout(dropout_rate)(x)

        # 출력층
        outputs = Dense(forecast_days)(x)

        # 모델 정의
        model = Model(inputs, outputs)

        # 컴파일 - Huber 손실함수 사용
        model.compile(
            optimizer=Adam(learning_rate=self.default_lr),
            loss='huber'
        )

        return model

    def build_ensemble_model(self, input_shape, forecast_days):
        """
        개선된 앙상블 모델 구축 - Keras 연산 사용
        """
        try:
            from tensorflow.keras.layers import Lambda, Reshape, Concatenate

            # 위치 인코딩 함수 정의
            def positional_encoding(inputs):
                batch_size = tf.shape(inputs)[0]
                seq_length = tf.shape(inputs)[1]
                d_model = tf.shape(inputs)[2]

                # 위치 인덱스 생성
                position = tf.cast(tf.range(seq_length), tf.float32)
                position = tf.reshape(position, [1, seq_length, 1])

                # 차원 인덱스 생성
                dim = tf.cast(tf.range(d_model), tf.float32)
                dim = tf.reshape(dim, [1, 1, d_model])

                # 짝수/홀수 차원에 따른 각도 계산
                angle = position / tf.pow(10000.0, (2.0 * (dim // 2.0)) / tf.cast(d_model, tf.float32))

                # sin, cos 적용
                encoding = tf.where(
                    tf.cast(tf.math.floormod(dim, 2.0), tf.bool),
                    tf.sin(angle),
                    tf.cos(angle)
                )

                # 배치 차원으로 확장
                encoding = tf.tile(encoding, [batch_size, 1, 1])

                return inputs + encoding

            # 1. LSTM 서브모델
            lstm_input = Input(shape=input_shape, name='lstm_input')

            # LSTM 스택
            x1 = LSTM(units=64, return_sequences=True, recurrent_dropout=0.1,
                      kernel_regularizer=l2(self.l2_lambda))(lstm_input)
            x1 = BatchNormalization()(x1)
            x1 = Dropout(0.2)(x1)

            x1 = LSTM(units=48, return_sequences=False, kernel_regularizer=l2(self.l2_lambda))(x1)
            x1 = BatchNormalization()(x1)
            x1 = Dropout(0.2)(x1)

            x1 = Dense(units=32, activation='relu', kernel_regularizer=l2(self.l2_lambda))(x1)
            lstm_output = Dense(units=forecast_days, name='lstm_output')(x1)

            # 2. Transformer 서브모델
            transformer_input = Input(shape=input_shape, name='transformer_input')

            # 시간적 인코딩 추가 (함수 사용)
            x2 = Lambda(positional_encoding)(transformer_input)

            # Transformer 인코더 블록
            for _ in range(3):
                x2 = self._transformer_encoder(x2, head_size=64, num_heads=4, ff_dim=256, dropout=0.2)

            x2 = GlobalAveragePooling1D()(x2)
            x2 = Dense(64, activation='relu', kernel_regularizer=l2(self.l2_lambda))(x2)
            x2 = Dropout(0.3)(x2)
            transformer_output = Dense(units=forecast_days, name='transformer_output')(x2)

            # 앙상블 가중 결합 (Keras 레이어만 사용)
            # tf.expand_dims 대신 Reshape 사용
            lstm_reshaped = Reshape((1, forecast_days))(lstm_output)
            transformer_reshaped = Reshape((1, forecast_days))(transformer_output)

            # tf.keras.layers.Concatenate 사용
            ensemble_input = Concatenate(axis=1)([lstm_reshaped, transformer_reshaped])

            # 플랫 레이어로 변환
            flattened = tf.keras.layers.Flatten()(ensemble_input)

            # 가중치 계산을 위한 MLP
            attention = Dense(8, activation='relu')(flattened)
            attention_weights = Dense(2, activation='softmax')(attention)

            # 가중치를 적용하기 위한 커스텀 레이어
            class WeightedAverage(tf.keras.layers.Layer):
                def call(self, inputs):
                    ensemble_tensor, weights = inputs
                    # 가중치 차원 변환
                    weights_expanded = tf.expand_dims(weights, axis=2)
                    # 가중 평균 계산
                    weighted = ensemble_tensor * weights_expanded
                    return tf.reduce_sum(weighted, axis=1)

            # 가중치 형태 변환
            weights_reshaped = Reshape((2,))(attention_weights)

            # 가중 평균 계산
            ensemble_output = WeightedAverage()([ensemble_input, weights_reshaped])

            # 최종 모델 구성
            model = Model(
                inputs=[lstm_input, transformer_input],
                outputs=ensemble_output
            )

            # 모델 컴파일
            model.compile(
                optimizer=Adam(learning_rate=self.default_lr),
                loss='huber'
            )

            return model

        except Exception as e:
            logger.error(f"앙상블 모델 구축 중 오류: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())

            # 오류 발생 시 기본 LSTM 모델 반환
            logger.info("앙상블 모델 구축 실패, 기본 LSTM 모델로 대체합니다.")
            return self.build_lstm_model(input_shape, forecast_days)

    def build_hybrid_ensemble(self, stock_data, forecast_days=30, features=None):
        """여러 모델을 결합한 하이브리드 앙상블 모델 구축"""
        try:
            import traceback  # 필수 임포트 추가

            logger.info("하이브리드 앙상블 모델 구축 시작...")

            # 회사 식별자 추가 (가능한 경우)
            company_symbol = None
            if hasattr(stock_data, 'name'):
                company_symbol = stock_data.name
            elif 'symbol' in st.session_state and st.session_state.symbol is not None:
                company_symbol = st.session_state.symbol
            elif 'current_analyzed_symbol' in st.session_state:
                company_symbol = st.session_state.current_analyzed_symbol

            # 자동 특성 선택 사용 여부 확인
            if features and (features == ['Auto'] or 'Auto' in features):
                logger.info("자동 특성 선택 알고리즘 실행 중...")
                features = self.auto_feature_selection(stock_data)
                logger.info(f"자동 선택된 특성: {features}")

            # LSTM 모델 학습 및 예측
            logger.info("LSTM 모델 학습 및 예측 중...")
            self.train_model(stock_data, model_type='lstm', forecast_days=forecast_days, features=features)
            lstm_pred = self.predict_future(stock_data, model_type='lstm', days=forecast_days, features=features)

            # Transformer 모델 학습 및 예측
            logger.info("Transformer 모델 학습 및 예측 중...")
            self.train_model(stock_data, model_type='transformer', forecast_days=forecast_days, features=features)
            transformer_pred = self.predict_future(stock_data, model_type='transformer', days=forecast_days,
                                                   features=features)

            # 앙상블 모델 학습 및 예측 (기존 구현)
            logger.info("앙상블 모델 학습 및 예측 중...")
            self.train_model(stock_data, model_type='ensemble', forecast_days=forecast_days, features=features)
            ensemble_pred = self.predict_future(stock_data, model_type='ensemble', days=forecast_days,
                                                features=features)

            # TCN 모델 학습 및 예측 (새로 추가된 모델)
            try:
                logger.info("TCN 모델 학습 및 예측 중...")
                self.train_model(stock_data, model_type='tcn', forecast_days=forecast_days, features=features)
                tcn_pred = self.predict_future(stock_data, model_type='tcn', days=forecast_days, features=features)
            except Exception as tcn_error:
                logger.warning(f"TCN 모델 학습/예측 실패: {str(tcn_error)}. 앙상블에서 제외합니다.")
                tcn_pred = None

            # Prophet 예측 추가 (Prophet 통합 모듈이 있는 경우)
            prophet_result = None
            try:
                from src.utils.stock_price_predictor_prophet import ProphetIntegration
                prophet_integration = ProphetIntegration()

                logger.info("Prophet 모델 학습 및 예측 중...")
                prophet_result = prophet_integration.add_prophet_features(stock_data, forecast_days)

                if prophet_result is None:
                    logger.warning("Prophet 예측 결과가 없습니다. 앙상블에서 제외합니다.")
            except ImportError:
                logger.warning("Prophet 통합 모듈을 찾을 수 없습니다. 앙상블에서 제외합니다.")
            except Exception as prophet_error:
                logger.warning(f"Prophet 통합 중 오류: {str(prophet_error)}. 앙상블에서 제외합니다.")

            # 모델별 가중치 계산 (과거 성능 기반)
            # 기본 가중치 설정
            model_weights = {
                'lstm': 0.3,
                'transformer': 0.3,
                'ensemble': 0.4,
                'tcn': 0.0,
                'prophet': 0.0
            }

            # 모델 성능 기반 가중치 조정
            try:
                # LSTM 모델 평가
                lstm_eval = self.evaluate_model(stock_data, model_type='lstm', forecast_days=forecast_days,
                                                features=features)
                lstm_rmse = lstm_eval.get('rmse', 0)

                # Transformer 모델 평가
                transformer_eval = self.evaluate_model(stock_data, model_type='transformer',
                                                       forecast_days=forecast_days, features=features)
                transformer_rmse = transformer_eval.get('rmse', 0)

                # 앙상블 모델 평가
                ensemble_eval = self.evaluate_model(stock_data, model_type='ensemble', forecast_days=forecast_days,
                                                    features=features)
                ensemble_rmse = ensemble_eval.get('rmse', 0)

                # TCN 모델 평가 (있는 경우만)
                tcn_rmse = 0
                if tcn_pred is not None:
                    try:
                        tcn_eval = self.evaluate_model(stock_data, model_type='tcn', forecast_days=forecast_days,
                                                       features=features)
                        tcn_rmse = tcn_eval.get('rmse', 0)
                    except:
                        pass

                # RMSE의 역수를 사용한 가중치 계산 (RMSE가 낮을수록 가중치 높음)
                if lstm_rmse > 0 and transformer_rmse > 0 and ensemble_rmse > 0:
                    # 역수 계산
                    lstm_inverse = 1.0 / max(lstm_rmse, 0.001)
                    transformer_inverse = 1.0 / max(transformer_rmse, 0.001)
                    ensemble_inverse = 1.0 / max(ensemble_rmse, 0.001)
                    tcn_inverse = 1.0 / max(tcn_rmse, 0.001) if tcn_rmse > 0 else 0

                    # 총합
                    total_inverse = lstm_inverse + transformer_inverse + ensemble_inverse + tcn_inverse

                    if total_inverse > 0:
                        # 정규화된 가중치
                        model_weights['lstm'] = lstm_inverse / total_inverse * 0.8  # Prophet 가중치 20%를 위해 80%만 할당
                        model_weights['transformer'] = transformer_inverse / total_inverse * 0.8
                        model_weights['ensemble'] = ensemble_inverse / total_inverse * 0.8

                        if tcn_rmse > 0:
                            model_weights['tcn'] = tcn_inverse / total_inverse * 0.8

                        # Prophet에 20% 고정 가중치 할당 (있는 경우)
                        if prophet_result is not None:
                            model_weights['prophet'] = 0.2
                        else:
                            # Prophet이 없는 경우 나머지 모델에 비례 배분
                            factor = 1.0 / (1.0 - 0.2)
                            model_weights['lstm'] *= factor
                            model_weights['transformer'] *= factor
                            model_weights['ensemble'] *= factor
                            if tcn_rmse > 0:
                                model_weights['tcn'] *= factor

                logger.info(f"계산된 모델 가중치: {model_weights}")

            except Exception as e:
                logger.warning(f"성능 기반 가중치 계산 중 오류: {str(e)}. 기본 가중치를 사용합니다.")

            # 가중 평균 예측 계산
            weighted_prediction = np.zeros(forecast_days)
            total_weight = 0.0

            # LSTM 예측
            if 'predicted' in lstm_pred and len(lstm_pred['predicted']) == forecast_days and model_weights['lstm'] > 0:
                weighted_prediction += lstm_pred['predicted'] * model_weights['lstm']
                total_weight += model_weights['lstm']

            # Transformer 예측
            if 'predicted' in transformer_pred and len(transformer_pred['predicted']) == forecast_days and \
                    model_weights['transformer'] > 0:
                weighted_prediction += transformer_pred['predicted'] * model_weights['transformer']
                total_weight += model_weights['transformer']

            # 앙상블 예측
            if 'predicted' in ensemble_pred and len(ensemble_pred['predicted']) == forecast_days and model_weights[
                'ensemble'] > 0:
                weighted_prediction += ensemble_pred['predicted'] * model_weights['ensemble']
                total_weight += model_weights['ensemble']

            # TCN 예측 (있는 경우만)
            if tcn_pred is not None and 'predicted' in tcn_pred and len(tcn_pred['predicted']) == forecast_days and \
                    model_weights['tcn'] > 0:
                weighted_prediction += tcn_pred['predicted'] * model_weights['tcn']
                total_weight += model_weights['tcn']

            # Prophet 예측 (있는 경우만)
            if prophet_result is not None and 'predictions' in prophet_result and len(
                    prophet_result['predictions']) == forecast_days and model_weights['prophet'] > 0:
                weighted_prediction += prophet_result['predictions'] * model_weights['prophet']
                total_weight += model_weights['prophet']

            # 가중치 합계가 0보다 큰 경우에만 정규화
            if total_weight > 0:
                weighted_prediction /= total_weight
            else:
                # 가중치 합계가 0이면 LSTM 예측 사용
                logger.warning("유효한 가중치가 없습니다. LSTM 예측을 사용합니다.")
                weighted_prediction = lstm_pred['predicted']

            # 신뢰 구간 계산
            confidence_high = np.zeros(forecast_days)
            confidence_low = np.zeros(forecast_days)

            # 각 모델의 신뢰 구간을 가중 평균
            if 'confidence_high' in lstm_pred and 'confidence_low' in lstm_pred:
                confidence_high = lstm_pred['confidence_high']
                confidence_low = lstm_pred['confidence_low']

            # 기타 신뢰 구간 정보가 있으면 추가
            if prophet_result is not None and 'upper' in prophet_result and 'lower' in prophet_result:
                # 각 모델의 신뢰 구간 범위를 가중 평균
                confidence_range = (confidence_high - confidence_low) * 0.8  # 80% 가중치
                prophet_range = (prophet_result['upper'] - prophet_result['lower']) * 0.2  # 20% 가중치

                # 중앙값 기준으로 신뢰 구간 재계산
                total_range = confidence_range + prophet_range
                confidence_high = weighted_prediction + total_range / 2
                confidence_low = weighted_prediction - total_range / 2

            # 전체 앙상블 결과 반환
            ensemble_result = {
                'dates': lstm_pred['dates'],  # 날짜는 모두 동일
                'predicted': weighted_prediction,
                'confidence_high': confidence_high,
                'confidence_low': confidence_low,
                'last_price': lstm_pred['last_price'],
                'model_weights': model_weights,
                'individual_predictions': {
                    'lstm': lstm_pred['predicted'] if 'predicted' in lstm_pred else None,
                    'transformer': transformer_pred['predicted'] if 'predicted' in transformer_pred else None,
                    'ensemble': ensemble_pred['predicted'] if 'predicted' in ensemble_pred else None,
                    'tcn': tcn_pred['predicted'] if tcn_pred is not None and 'predicted' in tcn_pred else None,
                    'prophet': prophet_result[
                        'predictions'] if prophet_result is not None and 'predictions' in prophet_result else None
                },
                'historical_volatility': lstm_pred.get('historical_volatility', 0),
                'trend_strength': lstm_pred.get('trend_strength', 0),
                'company_symbol': company_symbol
            }

            logger.info("하이브리드 앙상블 예측 완료")
            return ensemble_result

        except Exception as e:
            logger.error(f"하이브리드 앙상블 구축 중 오류: {str(e)}")
            logger.error(traceback.format_exc())

            # 오류 발생 시 기본 모델(LSTM) 결과 반환
            logger.info("오류 발생으로 LSTM 모델 결과만 반환합니다.")
            return lstm_pred if 'lstm_pred' in locals() else None

    def train_model(self, stock_data, model_type='lstm', forecast_days=30, features=None, use_advanced_features=True):
        """
        모델 학습 - 고급 특성 사용 옵션 추가
        """
        try:
            # 회사 식별자 추가 (가능한 경우)
            company_symbol = None
            if hasattr(stock_data, 'name'):
                company_symbol = stock_data.name
            elif 'symbol' in st.session_state and st.session_state.symbol is not None:
                company_symbol = st.session_state.symbol
            elif 'current_analyzed_symbol' in st.session_state:
                company_symbol = st.session_state.current_analyzed_symbol

            if company_symbol:
                logger.info(f"회사 식별자: {company_symbol} 모델 학습 시작")

            # 학습 시작 시간 기록
            start_time = datetime.now()
            logger.info(f"모델 학습 시작: {model_type} 모델, 예측 기간 {forecast_days}일")

            # 고급 특성 사용 여부 로깅
            if use_advanced_features:
                logger.info("고급 금융 특성을 포함하여 학습합니다.")
            else:
                logger.info("기본 기술적 지표만 사용하여 학습합니다.")

            # 데이터 충분성 확인 - 최소 필요 데이터 수
            # 주가 예측은 최소 예측일의 3배 또는 60일 중 더 큰 값 필요
            MIN_REQUIRED_DATA = max(forecast_days * 3, 60)

            if stock_data is None or len(stock_data) < MIN_REQUIRED_DATA:
                error_msg = f"데이터가 부족합니다. 최소 {MIN_REQUIRED_DATA}일 이상의 데이터가 필요합니다. (현재: {len(stock_data) if stock_data is not None else 0}일)"
                logger.error(error_msg)
                raise ValueError(error_msg)

            # 자동 특성 선택 사용 시
            if features and (features == ['Auto'] or 'Auto' in features):
                logger.info("자동 특성 선택 알고리즘 실행 중...")
                features = self.auto_feature_selection(stock_data)
                logger.info(f"자동 특성 선택 완료: {features}")

            # 현재 설정 확인
            current_params = {
                'model_type': model_type,
                'forecast_days': forecast_days,
                'features': features or st.session_state.model_settings.get('prediction_features', ['Close']),
                'company_symbol': company_symbol,  # 회사 식별자 저장
                'use_advanced_features': use_advanced_features  # 고급 특성 사용 여부 저장
            }

            # 이미 동일한 파라미터로 학습된 경우 재학습 방지
            if self.is_trained and self.training_params == current_params:
                logger.info("모델이 이미 동일한 파라미터로 학습되어 있습니다.")
                return True

            # 세션 상태에서 설정값 적용
            patience = st.session_state.model_settings.get('patience', self.default_patience)
            forecast_days = st.session_state.model_settings.get('prediction_days', forecast_days)
            batch_size = st.session_state.model_settings.get('batch_size', self.default_batch_size)

            logger.info(f"학습 설정 - patience: {patience}, forecast_days: {forecast_days}, batch_size: {batch_size}")

            # 데이터 전처리 - 고급 특성 사용 여부에 따라 다른 메서드 호출
            logger.info("데이터 전처리 중...")
            try:
                # 전처리 함수에 use_advanced_features 파라미터 전달
                X_train, y_train, X_test, y_test = self.preprocess_data(
                    stock_data,
                    forecast_days=forecast_days,
                    features=current_params['features'],
                    use_advanced_features=use_advanced_features
                )
                logger.info(f"데이터 전처리 완료 - X_train 형태: {X_train.shape if X_train is not None else 'None'}")

                # 추가: 데이터 차원 확인 및 처리
                if X_train is None or len(X_train) == 0:
                    raise ValueError("학습 데이터 생성에 실패했습니다. 데이터가 충분하지 않거나 전처리 중 오류가 발생했습니다.")

                # X_train 차원 체크
                if len(X_train.shape) < 3:
                    error_msg = f"입력 데이터 형태가 올바르지 않습니다: {X_train.shape}. 3차원 배열이 필요합니다."
                    logger.error(error_msg)
                    raise ValueError(error_msg)

            except Exception as prep_error:
                logger.error(f"데이터 전처리 중 오류 발생: {str(prep_error)}")
                raise ValueError(f"데이터 전처리 실패: {str(prep_error)}")

            # 콜백 설정 (patience 감소)
            callbacks = [
                # 학습 중단 콜백
                EarlyStopping(
                    monitor='val_loss',
                    patience=patience,
                    restore_best_weights=True,
                    verbose=1
                ),
                # 학습률 조정 콜백 (patience 조정)
                ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=patience // 2,  # 조기 중단 인내심의 1/2로 조정
                    min_lr=1e-6,
                    verbose=1
                )
            ]

            # 모델 저장 콜백 추가 (옵션)
            if model_type != 'ensemble' and model_type != 'hybrid':  # 앙상블 모델은 저장이 복잡할 수 있음
                model_filename = f"{model_type}_{forecast_days}d_{datetime.now().strftime('%Y%m%d_%H%M')}.keras"
                model_path = os.path.join(str(self.model_path), model_filename)

                checkpoint = ModelCheckpoint(
                    model_path,
                    monitor='val_loss',
                    save_best_only=True,
                    verbose=1
                )
                callbacks.append(checkpoint)

            # 모델 구축 전 입력 형태 로깅
            input_shape = (X_train.shape[1], X_train.shape[2])
            logger.info(f"모델 입력 형태: {input_shape}, 출력 길이: {forecast_days}")

            # 모델 구축 부분 수정
            logger.info(f"{model_type} 모델 구축 중...")

            # 하이브리드 모델 특별 처리
            if model_type.lower() == 'hybrid':
                # 하이브리드 모델은 훈련 방식이 다름을 로깅
                logger.info("하이브리드 모델은 별도의 학습 흐름을 사용합니다.")

                # 하이브리드 모델은 build_hybrid_ensemble 메서드 직접 호출
                # 이 메서드는 내부적으로 여러 모델을 학습하고 결합하므로 별도의 fit 호출이 필요 없음
                self.model = "HYBRID_MODEL_DIRECT_CALL"  # 더미 모델 설정

                # 학습 완료 후 상태 업데이트
                self.is_trained = True
                self.training_params = current_params

                # 성공 메시지
                logger.info(f"하이브리드 앙상블 모델 학습 설정 완료")
                return True
            elif model_type.lower() == 'prophet':
                # Prophet 모델은 훈련 방식이 다름을 로깅
                logger.info("Prophet 모델은 별도의 학습 흐름을 사용합니다.")

                # Prophet 모델 특별 처리 - 학습 단계에서는 상태만 업데이트
                self.model = "PROPHET_MODEL_DIRECT_CALL"  # 더미 모델 설정

                # 학습 완료 후 상태 업데이트
                self.is_trained = True
                self.training_params = current_params

                # 성공 메시지
                logger.info(f"Prophet 모델 설정 완료")
                return True
            else:
                # 모델 팩토리 사용 (일반 모델)
                try:
                    from src.utils.model_factory import ModelFactory
                    model_factory = ModelFactory(self)
                    self.model = model_factory.create_model(model_type, input_shape, forecast_days)
                except ImportError:
                    # 기존 모델 빌더 사용
                    if model_type == 'lstm':
                        self.model = self.build_lstm_model(input_shape, forecast_days)
                    elif model_type == 'transformer':
                        self.model = self.build_transformer_model(input_shape, forecast_days)
                    elif model_type == 'ensemble':
                        self.model = self.build_ensemble_model(input_shape, forecast_days)
                    else:
                        raise ValueError(f"지원하지 않는 모델 유형: {model_type}")

            # 모델 구조 요약
            if hasattr(self.model, 'summary'):
                self.model.summary(print_fn=lambda x: logger.info(x))
            else:
                logger.info(f"{model_type} 모델은 summary 메소드를 지원하지 않습니다.")

            # 시간 측정을 위한 프로그레스 표시
            if 'progress_placeholder' in st.session_state:
                progress_placeholder = st.session_state.progress_placeholder
            else:
                progress_placeholder = None

            # 모델 학습
            logger.info("모델 학습 시작...")

            # 학습 시작 시간 출력
            current_time = datetime.now().strftime('%H:%M:%S')
            if progress_placeholder:
                progress_placeholder.text(f"학습 시작: {current_time}")

            # 모델 학습 과정
            if model_type == 'ensemble':
                # 앙상블 모델의 경우 입력이 여러 개임
                history = self.model.fit(
                    [X_train, X_train],  # 두 입력 모델에 동일한 데이터 전달
                    y_train,
                    epochs=self.default_epochs,
                    batch_size=batch_size,
                    validation_split=0.2,
                    callbacks=callbacks,
                    verbose=1
                )
            elif model_type != 'hybrid':  # 하이브리드 모델이 아닌 경우에만 fit 호출
                # 일반 모델의 경우
                history = self.model.fit(
                    X_train, y_train,
                    epochs=self.default_epochs,
                    batch_size=batch_size,
                    validation_split=0.2,
                    callbacks=callbacks,
                    verbose=1
                )
            else:
                # 하이브리드 모델은 이미 처리됨
                history = None

            # 학습 종료 시간 및 소요 시간 계산
            end_time = datetime.now()
            training_time = end_time - start_time
            logger.info(f"모델 학습 완료: 소요 시간 {training_time}")

            # 학습 히스토리 분석 (하이브리드 모델은 건너뜀)
            if model_type != 'hybrid' and history is not None:
                val_loss = history.history['val_loss']
                best_epoch = np.argmin(val_loss) + 1
                best_val_loss = val_loss[best_epoch - 1]
                logger.info(f"최적 모델 - Epoch: {best_epoch}/{len(val_loss)}, 검증 손실: {best_val_loss:.4f}")

                # 모델 성능 간단 평가 (학습 완료 직후) - 하이브리드 모델은 건너뜀
                if model_type == 'ensemble':
                    y_pred = self.model.predict([X_test, X_test])
                else:
                    y_pred = self.model.predict(X_test)

                test_loss = np.mean((y_pred - y_test) ** 2)
                logger.info(f"테스트 세트 MSE: {test_loss:.4f}")

            # 학습 완료 후 상태 업데이트
            self.is_trained = True
            self.training_params = current_params

            # 성공 메시지
            logger.info(f"{model_type} 모델 학습 성공!")
            return True

        except Exception as e:
            logger.error(f"모델 학습 중 오류 발생: {str(e)}", exc_info=True)
            # 세션 상태 업데이트 - 실패 정보 기록
            if 'model_settings' in st.session_state:
                st.session_state.model_settings['last_error'] = str(e)
                st.session_state.model_settings['last_error_time'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # 학습 실패 상태 설정
            self.is_trained = False  # 학습 실패로 표시

            raise

    def predict_future(self, stock_data, model_type='lstm', days=30, features=None):
        """미래 주가 예측 - 개선된 예측 프로세스, 회사 식별자 추가, 하이브리드 모델 특별 처리"""
        try:
            # 회사 식별자 추가 (가능한 경우)
            company_symbol = None
            if hasattr(stock_data, 'name'):
                company_symbol = stock_data.name
            elif 'symbol' in st.session_state and st.session_state.symbol is not None:
                company_symbol = st.session_state.symbol
            elif 'current_analyzed_symbol' in st.session_state:
                company_symbol = st.session_state.current_analyzed_symbol

            if company_symbol:
                logger.info(f"회사 식별자: {company_symbol} 예측 시작")

            logger.info(f"미래 주가 예측 시작: {model_type} 모델, {days}일 예측")

            # 예측 설정 저장
            prediction_config = {
                'model_type': model_type,
                'days': days,
                'features': features,
                'date': datetime.now().strftime('%Y-%m-%d'),
                'company_symbol': company_symbol  # 회사 식별자 추가
            }

            # 데이터 충분성 확인 - 최소 필요 데이터 수
            # 주가 예측은 최소 예측일의 3배 또는 60일 중 더 큰 값 필요
            MIN_REQUIRED_DATA = max(days * 3, 60)

            if stock_data is None or len(stock_data) < MIN_REQUIRED_DATA:
                error_msg = f"데이터가 부족합니다. 최소 {MIN_REQUIRED_DATA}일 이상의 데이터가 필요합니다. (현재: {len(stock_data) if stock_data is not None else 0}일)"
                logger.error(error_msg)

                # 오류를 포함한 결과 반환
                result = {
                    'error': True,
                    'error_message': error_msg,
                    'company_symbol': company_symbol,
                    'prediction_config': prediction_config
                }
                return result

            # 자동 특성 선택 사용 시
            if features and (features == ['Auto'] or 'Auto' in features):
                logger.info("자동 특성 선택 알고리즘 실행 중...")
                features = self.auto_feature_selection(stock_data)
                logger.info(f"자동 선택된 특성: {features}")
                prediction_config['features'] = features  # 실제 사용된 특성으로 업데이트

            # 하이브리드 모델 특별 처리
            if model_type.lower() == 'hybrid':
                logger.info("하이브리드 모델은 별도의 예측 흐름을 사용합니다.")
                # 하이브리드 모델은 build_hybrid_ensemble 메서드 직접 호출
                result = self.build_hybrid_ensemble(stock_data, days, features)

                # 예측 설정 정보 추가
                if result:
                    result['prediction_config'] = prediction_config

                    # 회사 식별자가 아직 없으면 추가
                    if 'company_symbol' not in result or not result['company_symbol']:
                        result['company_symbol'] = company_symbol

                    # 오류 표시 제거
                    result['error'] = False

                return result
            # Prophet 모델 특별 처리
            elif model_type.lower() == 'prophet':
                logger.info("Prophet 모델은 별도의 예측 흐름을 사용합니다.")

                try:
                    # Prophet 통합 모듈 가져오기
                    from src.utils.stock_price_predictor_prophet import ProphetIntegration
                    prophet_integration = ProphetIntegration()

                    if not prophet_integration.is_prophet_available:
                        error_msg = "Prophet 라이브러리가 설치되지 않았습니다. pip install prophet을 실행하세요."
                        logger.error(error_msg)
                        raise ImportError(error_msg)

                    # Prophet 예측 수행
                    prophet_result = prophet_integration.add_prophet_features(stock_data, days)

                    if prophet_result is None:
                        error_msg = "Prophet 예측 결과가 없습니다."
                        logger.error(error_msg)
                        raise ValueError(error_msg)

                    # Prophet 결과를 일반 예측 결과 형식으로 변환
                    last_price = stock_data['Close'].iloc[-1]

                    # 거래일 생성
                    prediction_dates = self.generate_trading_days(stock_data.index[-1], days)

                    # 결과 구성
                    result = {
                        'dates': prediction_dates,
                        'predicted': prophet_result['predictions'],
                        'confidence_high': prophet_result['upper'],
                        'confidence_low': prophet_result['lower'],
                        'last_price': last_price,
                        'company_symbol': company_symbol,
                        'error': False,
                        'prediction_config': prediction_config,
                        'prophet_forecast': prophet_result['forecast_df'] if 'forecast_df' in prophet_result else None
                    }

                    # 변동성 및 추세 강도 계산 (기존 방식 활용)
                    historical_volatility = np.std(
                        np.diff(stock_data['Close'].values[-60:]) / stock_data['Close'].values[-60:-1]) * 100
                    result['historical_volatility'] = historical_volatility

                    # 추세 강도 (신뢰 구간 기반)
                    if 'upper' in prophet_result and 'lower' in prophet_result:
                        price_change = (prophet_result['predictions'][-1] - last_price) / last_price
                        confidence_width = np.mean(
                            (prophet_result['upper'] - prophet_result['lower']) / prophet_result['predictions'])
                        trend_strength = abs(price_change) / confidence_width if confidence_width > 0 else 0
                        result['trend_strength'] = trend_strength

                    return result

                except (ImportError, Exception) as e:
                    import traceback
                    logger.error(f"Prophet 예측 중 오류: {str(e)}")
                    logger.error(traceback.format_exc())

                    # 오류를 포함한 결과 반환
                    return {
                        'error': True,
                        'error_message': str(e),
                        'company_symbol': company_symbol,
                        'dates': [],
                        'predicted': [],
                        'last_price': stock_data['Close'].iloc[-1] if isinstance(stock_data,
                                                                                 pd.DataFrame) and not stock_data.empty else 0,
                        'confidence_high': [],
                        'confidence_low': [],
                        'historical_volatility': 0,
                        'trend_strength': 0,
                        'prediction_config': prediction_config
                    }
            else:
                # 일반 모델 처리 (기존 코드)
                # 모델이 학습되지 않은 경우에만 학습 실행
                if not self.is_trained:
                    logger.info("학습된 모델이 없습니다. 모델 학습을 시작합니다...")
                    training_success = self.train_model(stock_data, model_type, days, features)

                    if not training_success:
                        raise ValueError("모델 학습에 실패했습니다.")

            # 데이터 준비
            logger.info("예측용 데이터 준비 중...")
            last_sequence = self._prepare_prediction_data(stock_data, features)

            # 학습된 모델 확인
            if self.model is None:
                raise ValueError("학습된 모델이 없습니다.")

            # 예측 수행
            if model_type == 'ensemble':
                # 앙상블 모델의 경우 여러 입력 필요
                prediction = self.model.predict([last_sequence, last_sequence])
            else:
                # 단일 모델의 경우
                prediction = self.model.predict(last_sequence)

            # 결과 처리 및 반환
            logger.info("예측 결과 처리 중...")
            result = self._process_prediction_result(prediction, stock_data, days)

            # 예측 설정 정보 추가
            result['prediction_config'] = prediction_config

            # 회사 식별자 추가
            result['company_symbol'] = company_symbol

            # 예측 결과 기본 분석 추가
            result['analysis'] = self._analyze_prediction(result)

            # 성공 메시지
            logger.info(f"예측 완료: {days}일 예측")

            # 오류 표시 제거
            result['error'] = False

            return result

        except Exception as e:
            logger.error(f"예측 중 오류 발생: {str(e)}", exc_info=True)

            # 오류를 포함한 결과 반환
            return {
                'error': True,
                'error_message': str(e),
                'company_symbol': company_symbol,
                'dates': [],
                'predicted': [],
                'last_price': stock_data['Close'].iloc[-1] if isinstance(stock_data,
                                                                         pd.DataFrame) and not stock_data.empty else 0,
                'confidence_high': [],
                'confidence_low': [],
                'historical_volatility': 0,
                'trend_strength': 0,
                'prediction_config': prediction_config if 'prediction_config' in locals() else {
                    'model_type': model_type,
                    'days': days,
                    'features': features,
                    'date': datetime.now().strftime('%Y-%m-%d'),
                    'company_symbol': company_symbol
                }
            }

    def _prepare_prediction_data(self, stock_data, features=None):
        """
        예측을 위한 입력 데이터 준비 - 고급 특성 일관성 문제 해결
        """
        try:
            # 자동 특성 선택 사용 시
            if features and (features == ['Auto'] or 'Auto' in features):
                features = self.auto_feature_selection(stock_data)
                logger.info(f"자동 특성 선택 결과: {features}")

            # 설정된 특성 가져오기
            features = features or st.session_state.model_settings.get('prediction_features', ['Close'])

            # 학습 시 사용된 특성 확인 (스케일러에 저장된 특성)
            scaler_features = []
            if hasattr(self.scaler, 'feature_names_in_'):
                scaler_features = list(self.scaler.feature_names_in_)
                logger.info(f"스케일러에 저장된 특성: {scaler_features}")

            # 기술적 지표 추가 (확장된 고급 특성 포함)
            df = self._add_technical_indicators_enhanced(stock_data)

            # 특성 일관성 확인
            missing_features = [f for f in scaler_features if f not in df.columns]
            if missing_features:
                logger.warning(f"스케일러에 있지만 현재 데이터에 없는 특성: {missing_features}")

                # 누락된 특성에 대한 더미 컬럼 추가
                for feature in missing_features:
                    logger.info(f"누락된 특성 '{feature}'에 대해 더미 컬럼 추가")
                    # 0으로 채워진 컬럼 추가
                    df[feature] = 0.0

            # 선택된 특성만 사용하는 대신, 스케일러에 저장된 모든 특성 사용
            if scaler_features:
                logger.info(f"스케일러에 저장된 모든 특성 사용: {scaler_features}")
                df_selected = df[scaler_features]
            else:
                # 선택된 특성만 사용 (대체 로직)
                if features != ['All']:
                    selected_columns = []
                    for feature in features:
                        if feature in df.columns:
                            selected_columns.append(feature)

                    # 최소한 종가는 포함되어야 함
                    if 'Close' not in selected_columns:
                        selected_columns.append('Close')

                    logger.info(f"선택된 특성: {selected_columns}")
                    df_selected = df[selected_columns]
                else:
                    df_selected = df

            # 열 순서 확인 로깅
            logger.info(f"스케일링 전 데이터프레임 열: {df_selected.columns.tolist()}")

            # 데이터 정규화
            scaled_data = self.scaler.transform(df_selected)
            logger.info(f"스케일링 완료: 형태 {scaled_data.shape}")

            # 마지막 시퀀스 준비
            last_sequence = scaled_data[-self.prediction_days:]
            last_sequence = np.reshape(last_sequence, (1, last_sequence.shape[0], last_sequence.shape[1]))
            logger.info(f"입력 시퀀스 형태: {last_sequence.shape}")

            return last_sequence

        except ValueError as ve:
            # 스케일러의 특성과 현재 데이터의 특성이 일치하지 않는 경우
            if "feature names" in str(ve):
                logger.error(f"특성 불일치 오류: {str(ve)}")

                # 스케일러 정보 로깅
                if hasattr(self.scaler, 'feature_names_in_'):
                    logger.info(f"스케일러 특성: {self.scaler.feature_names_in_}")

                # 현재 데이터프레임 정보 로깅
                logger.info(f"현재 데이터프레임 특성: {df.columns.tolist()}")

                # 특성을 맞추기 위해 스케일러 재학습 시도 (대체 해결책)
                logger.info("데이터 불일치로 인해 스케일러 재학습 시도")

                # 기본 특성만 사용하여 다시 시도
                basic_features = ['Close', 'MA20', 'RSI']
                basic_df = self._add_technical_indicators(stock_data)[basic_features]

                # 새 스케일러 생성 및 학습
                self.scaler = RobustScaler()
                scaled_basic_data = self.scaler.fit_transform(basic_df)

                # 마지막 시퀀스 준비
                last_sequence = scaled_basic_data[-self.prediction_days:]
                last_sequence = np.reshape(last_sequence, (1, last_sequence.shape[0], last_sequence.shape[1]))

                logger.info(f"재시도 후 입력 시퀀스 형태: {last_sequence.shape}")
                return last_sequence

            else:
                # 다른 ValueError 예외 재발생
                logger.error(f"예측 데이터 준비 중 오류: {str(ve)}")
                raise

        except Exception as e:
            logger.error(f"예측 데이터 준비 중 오류: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            raise

    def _process_prediction_result(self, prediction, stock_data, days, mc_std=None):
        """
        예측 결과 처리 - 신뢰 구간 개선
        """
        try:
            # 예측값 shape 확인 및 조정
            if len(prediction.shape) == 2:  # (1, forecast_days) 형태
                prediction = prediction[0]  # (forecast_days,) 형태로 변환

            # 예측값 길이 확인 및 조정
            if len(prediction) < days:
                logger.warning(f"예측 결과({len(prediction)}일)가 요청된 일수({days}일)보다 적습니다. 예측값을 확장합니다.")
                prediction = np.pad(
                    prediction,
                    (0, days - len(prediction)),
                    mode='edge'  # 마지막 값으로 패딩
                )

            # 예측값을 days 길이로 자르기
            prediction = prediction[:days]

            # 예측값 스케일 복원 전에 close 컬럼 인덱스 찾기
            close_idx = 0  # 기본값

            # 스케일러에서 열 이름 확인
            if hasattr(self.scaler, 'feature_names_in_'):
                feature_names = self.scaler.feature_names_in_
                if 'Close' in feature_names:
                    close_idx = np.where(feature_names == 'Close')[0][0]

            # 예측값 스케일 복원을 위해 shape 조정
            prediction_reshaped = prediction.reshape(-1, 1)
            # 스케일러 특성 수 확인
            n_features = self.scaler.n_features_in_
            prediction_copies = np.repeat(prediction_reshaped, n_features, axis=1)

            # 스케일 복원
            prediction_inverse = self.scaler.inverse_transform(prediction_copies)
            predicted_prices = prediction_inverse[:, close_idx]  # Close 가격만 추출

            # 마지막 실제 가격
            last_price = stock_data['Close'].iloc[-1]
            last_date = stock_data.index[-1]

            # 거래일(월-금)만 고려한 예측 날짜 생성
            trading_days = self.generate_trading_days(last_date, days)
            prediction_dates = trading_days

            # 과거 변동성 계산 (신뢰 구간 개선을 위해)
            past_days = min(60, len(stock_data))
            recent_prices = stock_data['Close'].iloc[-past_days:].values
            historical_volatility = np.std(np.diff(recent_prices) / recent_prices[:-1]) * 100  # 일일 변동성(%)
            annual_volatility = historical_volatility * np.sqrt(252)  # 연간 변동성으로 변환

            # 기간에 따른 변동성 증가 반영 (개선된 방식)
            volatility_factor = np.sqrt(np.arange(1, days + 1) / 252)  # 시간에 따른 불확실성 증가
            
            # 추가 수정: 볼린저 밴드 기반 변동성 조정
            if 'BB_Width' in stock_data.columns:
                # 볼린저 밴드 폭을 기반으로 변동성 조정
                bb_width_avg = stock_data['BB_Width'].iloc[-20:].mean()
                volatility_adjuster = max(0.8, min(1.2, bb_width_avg / 0.05))  # 기준값 0.05 대비 조정
                volatility_factor = volatility_factor * volatility_adjuster
                logger.info(f"볼린저 밴드 기반 변동성 조정 계수: {volatility_adjuster}")

            # 신뢰 구간 계산 - 과거 변동성 기반
            logger.info(f"과거 변동성({historical_volatility:.2f}%) 기반 신뢰 구간 계산")

            # 신뢰 수준 설정
            conf_level = 0.9  # 90% 신뢰 수준
            z_score = 1.645  # 90% 신뢰 수준에 해당하는 Z-score

            # 기간별 신뢰 구간 계산
            confidence_range = predicted_prices * (historical_volatility / 100) * z_score * volatility_factor
            confidence_high = predicted_prices + confidence_range
            confidence_low = predicted_prices - confidence_range

            # 추세 강도 계산
            trend_strength = self._calculate_trend_strength(predicted_prices, confidence_high, confidence_low)

            # 모든 배열의 길이가 동일한지 확인
            if not (len(predicted_prices) == len(confidence_high) == len(confidence_low) == len(
                    prediction_dates) == days):
                logger.error("배열 길이가 일치하지 않습니다.")
                logger.error(f"Shape mismatch - predicted: {predicted_prices.shape}, "
                             f"conf_high: {confidence_high.shape}, "
                             f"conf_low: {confidence_low.shape}, "
                             f"dates: {len(prediction_dates)}")
                raise ValueError("예측 결과 배열 길이가 일치하지 않습니다.")

            # 반환할 결과 구성
            result = {
                'dates': prediction_dates,
                'predicted': predicted_prices,
                'confidence_high': confidence_high,
                'confidence_low': confidence_low,
                'last_price': last_price,
                'historical_volatility': historical_volatility,
                'annual_volatility': annual_volatility,
                'trend_strength': trend_strength
            }

            return result

        except Exception as e:
            logger.error(f"예측 결과 처리 중 오류 발생: {str(e)}")
            logger.error(f"예측값 shape: {prediction.shape if hasattr(prediction, 'shape') else 'unknown'}")
            logger.error(f"요청된 예측 일수: {days}")
            import traceback
            logger.error(traceback.format_exc())
            raise

    def generate_trading_days(self, start_date, num_days):
        """
        주어진 시작일로부터 지정된 수의 거래일(월-금)을 생성합니다.
        공휴일은 고려하지 않습니다.
        """
        import pandas as pd
        from datetime import datetime, timedelta

        # 시작일이 datetime 객체가 아니면 변환
        if not isinstance(start_date, datetime) and not isinstance(start_date, pd.Timestamp):
            try:
                start_date = pd.Timestamp(start_date)
            except:
                start_date = datetime.now()

        # 시작일 다음 날부터 시작 (오늘은 이미 지난 거래일)
        start_date += timedelta(days=1)

        # 시작일이 주말이면 다음 월요일로 조정
        while start_date.weekday() > 4:  # 5: 토요일, 6: 일요일
            start_date += timedelta(days=1)  # 다음 날로 이동

        # 거래일 생성
        trading_days = []
        current_date = start_date

        while len(trading_days) < num_days:
            if current_date.weekday() < 5:  # 월요일(0)부터 금요일(4)까지만
                trading_days.append(current_date)
            current_date += timedelta(days=1)

        return trading_days

    def _calculate_trend_strength(self, predicted_prices, confidence_high, confidence_low):
        """
        예측 추세의 강도 계산 - 개선된 버전
        """
        try:
            # 전체 예측 기간의 가격 변화율
            price_change = (predicted_prices[-1] - predicted_prices[0]) / predicted_prices[0]

            # 평균 신뢰 구간 너비 (가격의 비율로)
            avg_confidence_width = np.mean((confidence_high - confidence_low) / predicted_prices)

            # 예측 변화량 / 신뢰 구간 너비 = 추세 강도
            # 값이 클수록 추세가 강함 (변화량이 불확실성보다 큼)
            if avg_confidence_width > 0:
                trend_strength = abs(price_change) / avg_confidence_width
                
                # 추가: 추세 방향에 따른 조정
                if price_change > 0:
                    # 상승 추세 강도는 그대로 유지
                    pass
                else:
                    # 하락 추세 강도는 10% 할인 (하락 추세의 불확실성이 더 크다고 가정)
                    trend_strength = trend_strength * 0.9
            else:
                trend_strength = 0

            return float(trend_strength)

        except Exception as e:
            logger.error(f"추세 강도 계산 중 오류: {str(e)}")
            return 0.0

    def _analyze_prediction(self, prediction_result):
        """
        예측 결과 분석 - 개선된 분석
        """
        try:
            # 필요한 데이터 추출
            prices = prediction_result['predicted']
            dates = prediction_result['dates']
            conf_high = prediction_result['confidence_high']
            conf_low = prediction_result['confidence_low']
            last_price = prediction_result['last_price']
            days = len(prices)

            # 결과 저장 딕셔너리
            analysis = {}

            # 기간별 구분
            short_term_idx = min(7, days)
            mid_term_idx = min(15, days)  # 21에서 15로 축소

            # 단기/중기/장기 가격 변화 계산
            short_term_change = (prices[short_term_idx - 1] - last_price) / last_price * 100
            
            if mid_term_idx > short_term_idx:
                mid_term_change = (prices[mid_term_idx - 1] - prices[short_term_idx - 1]) / prices[
                    short_term_idx - 1] * 100
            else:
                mid_term_change = 0

            long_term_change = (prices[-1] - prices[min(mid_term_idx, days) - 1]) / prices[
                min(mid_term_idx, days) - 1] * 100

            # 추세 분류 (더 세분화)
            def classify_trend(change):
                if change > 7:
                    return "강한 상승"
                elif change > 3:
                    return "상승"
                elif change > 0:
                    return "약한 상승"
                elif change > -3:
                    return "약한 하락"
                elif change > -7:
                    return "하락"
                else:
                    return "강한 하락"

            # 기간별 추세 분석
            analysis['short_term'] = {
                'period': f"1-{short_term_idx}일",
                'change_percent': short_term_change,
                'trend': classify_trend(short_term_change)
            }

            if mid_term_idx > short_term_idx:
                analysis['mid_term'] = {
                    'period': f"{short_term_idx + 1}-{mid_term_idx}일",
                    'change_percent': mid_term_change,
                    'trend': classify_trend(mid_term_change)
                }

            if days > mid_term_idx:
                analysis['long_term'] = {
                    'period': f"{mid_term_idx + 1}-{days}일",
                    'change_percent': long_term_change,
                    'trend': classify_trend(long_term_change)
                }

            # 예측 신뢰도 (신뢰 구간 너비 기반)
            confidence_width = np.mean((conf_high - conf_low) / prices) * 100

            if confidence_width < 5:
                confidence_level = "높음"
            elif confidence_width < 10:
                confidence_level = "중간"
            else:
                confidence_level = "낮음"

            analysis['confidence'] = {
                'level': confidence_level,
                'avg_width_percent': confidence_width
            }

            # 전체 예측 기간 추세
            overall_change = (prices[-1] - last_price) / last_price * 100
            analysis['overall'] = {
                'period': f"1-{days}일",
                'change_percent': overall_change,
                'trend': classify_trend(overall_change)
            }

            # 변동성 분석
            price_volatility = np.std(np.diff(prices) / prices[:-1]) * 100
            analysis['volatility'] = {
                'daily_percent': price_volatility,
                'annualized_percent': price_volatility * np.sqrt(252)
            }

            # 극대/극소점 감지 (주요 전환점) - 개선된 알고리즘
            peaks_idx = []
            valleys_idx = []

            # 더 엄격한 극대/극소점 기준 적용
            for i in range(3, len(prices) - 3):
                # 극대점: 전후 3일보다 높은 가격
                if all(prices[i] > prices[i-j] for j in range(1, 4)) and all(prices[i] > prices[i+j] for j in range(1, 4)):
                    peaks_idx.append(i)

                # 극소점: 전후 3일보다 낮은 가격
                if all(prices[i] < prices[i-j] for j in range(1, 4)) and all(prices[i] < prices[i+j] for j in range(1, 4)):
                    valleys_idx.append(i)

            # 감지된 전환점들
            turning_points = []

            for idx in sorted(peaks_idx + valleys_idx):
                point_type = "고점" if idx in peaks_idx else "저점"
                date = dates[idx].strftime('%Y-%m-%d')
                price = prices[idx]
                change = (price - last_price) / last_price * 100

                turning_points.append({
                    'date': date,
                    'day': idx + 1,  # 1부터 시작하는 날짜 인덱스
                    'type': point_type,
                    'price': float(price),
                    'change_percent': float(change)
                })

            analysis['turning_points'] = turning_points

            # 추가: 매매 의견 (추세 및 신뢰도에 기반한 의견)
            if overall_change > 5 and confidence_level != "낮음":
                recommendation = "매수 고려"
                reason = "뚜렷한 상승 추세와 적절한 신뢰도를 보입니다."
            elif overall_change < -5 and confidence_level != "낮음":
                recommendation = "매도 고려"
                reason = "뚜렷한 하락 추세와 적절한 신뢰도를 보입니다."
            else:
                recommendation = "관망"
                reason = "명확한 추세가 보이지 않거나 예측 신뢰도가 낮습니다."

            analysis['recommendation'] = {
                'opinion': recommendation,
                'reason': reason
            }

            return analysis

        except Exception as e:
            logger.error(f"예측 결과 분석 중 오류: {str(e)}")
            return {}

    def evaluate_model(self, stock_data, model_type='lstm', forecast_days=30, features=None):
        """모델 성능 평가 - 개선된 평가 지표 및 Prophet 모델 처리 추가"""
        try:
            # Prophet 모델 특별 처리
            if model_type.lower() == 'prophet':
                logger.info("Prophet 모델 평가 특별 처리 중...")

                try:
                    from src.utils.stock_price_predictor_prophet import ProphetIntegration
                    prophet_integration = ProphetIntegration()

                    if not prophet_integration.is_prophet_available:
                        raise ImportError("Prophet 라이브러리가 설치되지 않았습니다.")

                    # 데이터 준비 - 테스트 기간 결정
                    total_days = len(stock_data)
                    test_size = min(int(total_days * 0.2), 60)  # 테스트 데이터는 최대 20% 또는 60일
                    train_data = stock_data.iloc[:-test_size]
                    test_data = stock_data.iloc[-test_size:]

                    # 학습 데이터로 Prophet 학습 및 예측
                    prophet_result = prophet_integration.add_prophet_features(train_data, test_size)

                    if prophet_result is None:
                        raise ValueError("Prophet 예측 결과가 없습니다.")

                    # 예측값과 실제값 비교
                    y_pred = prophet_result['predictions']
                    y_test = test_data['Close'].values

                    # 예측 길이 조정 (필요한 경우)
                    min_len = min(len(y_pred), len(y_test))
                    y_pred = y_pred[:min_len]
                    y_test = y_test[:min_len]

                    # 평가 지표 계산
                    mse = np.mean((y_test - y_pred) ** 2)
                    rmse = np.sqrt(mse)
                    mape = np.mean(np.abs((y_test - y_pred) / np.maximum(y_test, 1e-10))) * 100

                    # 상대 RMSE 계산
                    current_price = stock_data['Close'].iloc[-1]
                    relative_rmse = (rmse / current_price) * 100

                    # 오차 계산
                    errors = y_test - y_pred
                    mean_error = np.mean(errors)
                    std_error = np.std(errors)

                    # 방향성 예측 정확도
                    actual_direction = np.sign(np.diff(y_test))
                    pred_direction = np.sign(np.diff(y_pred))
                    direction_match = (actual_direction == pred_direction).astype(int)
                    direction_accuracy = np.mean(direction_match) * 100 if len(direction_match) > 0 else 0

                    # R-squared 계산
                    y_test_mean = np.mean(y_test)
                    ss_total = np.sum((y_test - y_test_mean) ** 2)
                    ss_residual = np.sum((y_test - y_pred) ** 2)
                    r_squared = 1 - (ss_residual / ss_total) if ss_total > 0 else 0

                    # 결과 반환
                    return {
                        'mse': mse,
                        'rmse': rmse,
                        'mape': mape,
                        'relative_rmse': relative_rmse,
                        'errors': errors,
                        'mean_error': mean_error,
                        'std_error': std_error,
                        'direction_accuracy': direction_accuracy,
                        'r_squared': r_squared,
                        'y_test': y_test,
                        'y_pred': y_pred,
                        'bias': 'unbiased' if np.abs(mean_error) < std_error * 0.5 else 'biased'
                    }

                except Exception as prophet_error:
                    logger.error(f"Prophet 모델 평가 중 오류: {str(prophet_error)}")
                    import traceback
                    logger.error(traceback.format_exc())

                    # 기본 값 반환
                    return {
                        'mse': 0,
                        'rmse': 0,
                        'mape': 0,
                        'relative_rmse': 0,
                        'errors': np.array([0]),
                        'mean_error': 0,
                        'std_error': 0,
                        'direction_accuracy': 0,
                        'r_squared': 0,
                        'y_test': np.array([0]),
                        'y_pred': np.array([0]),
                        'bias': 'unknown'
                    }

            # 하이브리드 모델 특별 처리
            elif model_type.lower() == 'hybrid':
                logger.info("하이브리드 모델 평가 특별 처리 중...")

                # 하이브리드 모델은 각 구성 모델의 평가를 기반으로 처리
                try:
                    # LSTM 모델 평가를 기준으로 사용
                    lstm_eval = self.evaluate_model(stock_data, model_type='lstm', forecast_days=forecast_days,
                                                    features=features)

                    # 다른 모델들도 평가해서 평균을 낼 수 있지만, 복잡도를 고려하여 LSTM 결과만 사용
                    return lstm_eval

                except Exception as hybrid_error:
                    logger.error(f"하이브리드 모델 평가 중 오류: {str(hybrid_error)}")
                    import traceback
                    logger.error(traceback.format_exc())

                    # 기본 값 반환
                    return {
                        'mse': 0,
                        'rmse': 0,
                        'mape': 0,
                        'relative_rmse': 0,
                        'errors': np.array([0]),
                        'mean_error': 0,
                        'std_error': 0,
                        'direction_accuracy': 0,
                        'r_squared': 0,
                        'y_test': np.array([0]),
                        'y_pred': np.array([0]),
                        'bias': 'unknown'
                    }

            # 일반 모델 처리 (기존 코드)
            # 자동 특성 선택 사용 시
            if features and (features == ['Auto'] or 'Auto' in features):
                features = self.auto_feature_selection(stock_data)

            # 모델이 학습되지 않은 경우에만 학습 실행
            if not self.is_trained:
                self.train_model(stock_data, model_type, forecast_days, features)

            # 데이터 전처리
            X_train, y_train, X_test, y_test = self.preprocess_data(
                stock_data,
                forecast_days=forecast_days,
                features=features
            )

            # 예측 수행
            if model_type == 'ensemble':
                y_pred = self.model.predict([X_test, X_test])
            else:
                y_pred = self.model.predict(X_test)

            # 예측값과 실제값의 shape 확인 및 조정
            if y_pred.shape != y_test.shape:
                logger.warning(f"Shape mismatch - y_pred: {y_pred.shape}, y_test: {y_test.shape}")

            # min_length 정의
            min_samples = min(y_pred.shape[0], y_test.shape[0])
            min_length = 1
            if len(y_pred.shape) > 1:
                min_length = min(y_pred.shape[1], y_test.shape[1])

            # 배열 크기 조정
            y_pred = y_pred[:min_samples, :min_length] if len(y_pred.shape) > 1 else y_pred[:min_samples]
            y_test = y_test[:min_samples, :min_length] if len(y_test.shape) > 1 else y_test[:min_samples]

            # 스케일 복원
            # 데이터 형태에 맞게 스케일 복원 방식 조정
            n_features = self.scaler.n_features_in_

            # 종가 인덱스 찾기 (기본적으로 0으로 설정)
            close_idx = 0
            if features:
                try:
                    temp_df = self._add_technical_indicators(stock_data)
                    if 'Close' in temp_df.columns:
                        if isinstance(features, list) and 'Close' in features:
                            close_idx = features.index('Close')
                        else:
                            close_idx = list(temp_df.columns).index('Close')
                except:
                    close_idx = 0

            # 예측 결과 재구성 및 스케일 복원
            if len(y_test.shape) == 1:  # 1차원 배열인 경우
                # 1차원 결과를 2차원으로 확장
                y_test_inv = np.zeros(min_samples)
                y_pred_inv = np.zeros(min_samples)

                # 임시 배열을 만들어 스케일 복원
                for i in range(min_samples):
                    temp_test = np.zeros(n_features)
                    temp_test[close_idx] = y_test[i]

                    temp_pred = np.zeros(n_features)
                    temp_pred[close_idx] = y_pred[i]

                    y_test_inv[i] = self.scaler.inverse_transform([temp_test])[0, close_idx]
                    y_pred_inv[i] = self.scaler.inverse_transform([temp_pred])[0, close_idx]

            else:  # 2차원 배열인 경우
                y_test_inv = np.zeros((min_samples, min_length))
                y_pred_inv = np.zeros((min_samples, min_length))

                for i in range(min_samples):
                    for j in range(min_length):
                        temp_test = np.zeros(n_features)
                        temp_test[close_idx] = y_test[i, j]

                        temp_pred = np.zeros(n_features)
                        temp_pred[close_idx] = y_pred[i, j]

                        y_test_inv[i, j] = self.scaler.inverse_transform([temp_test])[0, close_idx]
                        y_pred_inv[i, j] = self.scaler.inverse_transform([temp_pred])[0, close_idx]

            # 평가 지표 계산을 위해 배열 펼치기
            y_test_flat = y_test_inv.flatten()
            y_pred_flat = y_pred_inv.flatten()

            # 평가 지표 계산
            mse = mean_squared_error(y_test_flat, y_pred_flat)
            rmse = np.sqrt(mse)

            # MAPE 계산 시 0 나누기 문제 방지
            mape = np.mean(np.abs((y_test_flat - y_pred_flat) / np.maximum(y_test_flat, 1e-10))) * 100

            # 상대 RMSE 계산 (현재 주가 대비)
            current_price = stock_data['Close'].iloc[-1]
            relative_rmse = (rmse / current_price) * 100

            # 오차 계산
            errors = y_test_flat - y_pred_flat
            mean_error = np.mean(errors)
            std_error = np.std(errors)

            # 추가 지표: Direction Accuracy (방향성 예측 정확도)
            # 연속된 두 값 간의 방향 예측이 맞는지 계산
            if len(y_test_flat) > 1 and len(y_pred_flat) > 1:
                actual_direction = np.sign(np.diff(y_test_flat))
                pred_direction = np.sign(np.diff(y_pred_flat))
                direction_match = (actual_direction == pred_direction).astype(int)
                direction_accuracy = np.mean(direction_match) * 100
            else:
                direction_accuracy = 0

            # R-squared (설명된 분산 비율) 계산
            y_test_mean = np.mean(y_test_flat)
            ss_total = np.sum((y_test_flat - y_test_mean) ** 2)
            ss_residual = np.sum((y_test_flat - y_pred_flat) ** 2)

            if ss_total > 0:
                r_squared = 1 - (ss_residual / ss_total)
            else:
                r_squared = 0

            return {
                'mse': mse,
                'rmse': rmse,
                'mape': mape,
                'relative_rmse': relative_rmse,
                'errors': errors,
                'mean_error': mean_error,
                'std_error': std_error,
                'direction_accuracy': direction_accuracy,
                'r_squared': r_squared,
                'y_test': y_test_inv,
                'y_pred': y_pred_inv,
                'bias': 'unbiased' if np.abs(mean_error) < std_error * 0.5 else 'biased'
            }

        except Exception as e:
            logger.error(f"모델 평가 중 오류 발생: {str(e)}", exc_info=True)
            raise

    def _transformer_encoder(self, inputs, head_size, num_heads, ff_dim, dropout=0):
        """Transformer 인코더 블록"""
        # 멀티헤드 어텐션
        x = LayerNormalization(epsilon=1e-6)(inputs)
        x = MultiHeadAttention(
            key_dim=head_size,
            num_heads=num_heads,
            dropout=dropout
        )(x, x)
        x = Dropout(dropout)(x)
        res = x + inputs

        # 피드 포워드 네트워크
        x = LayerNormalization(epsilon=1e-6)(res)
        x = Dense(ff_dim, activation='relu', kernel_regularizer=l2(self.l2_lambda))(x)
        x = Dropout(dropout)(x)
        x = Dense(inputs.shape[-1], kernel_regularizer=l2(self.l2_lambda))(x)

        return x + res