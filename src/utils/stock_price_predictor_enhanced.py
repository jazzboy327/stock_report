# src/utils/stock_price_predictor_enhanced.py

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, LSTM, GRU, Dropout, Concatenate,
    BatchNormalization, LayerNormalization, MultiHeadAttention,
    GlobalAveragePooling1D, Add, Lambda, Conv1D, Activation, SpatialDropout1D,
    AveragePooling1D, Reshape
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
import logging
from datetime import datetime, timedelta
import os
import traceback


# 기존 StockPricePredictor 클래스를 확장하는 추가 메소드들

class StockPricePredictorEnhanced:
    """
    StockPricePredictor 클래스에 추가될 메소드들
    실제 구현 시에는 StockPricePredictor 클래스에 직접 통합
    """

    def build_tft_model(self, input_shape, forecast_days):
        """Temporal Fusion Transformer 모델 구축"""
        # 입력 처리
        inputs = Input(shape=input_shape)

        # 변수 처리 네트워크
        variable_selection = Dense(64, activation='elu')(inputs)
        variable_selection = BatchNormalization()(variable_selection)

        # LSTM 인코더 (과거 정보 인코딩)
        encoder = LSTM(128, return_sequences=True)(variable_selection)
        encoder = LayerNormalization()(encoder)

        # Static Covariate Encoder (시간에 따라 변하지 않는 특성 처리)
        static_features = Dense(64, activation='elu')(inputs[:, 0, :])
        static_context = Dense(128)(static_features)

        # Self-Attention 메커니즘
        attention_input = encoder
        for _ in range(2):  # 2개의 Self-Attention 블록
            attention_output = MultiHeadAttention(
                num_heads=4, key_dim=32
            )(attention_input, attention_input)
            attention_output = Dropout(0.1)(attention_output)
            attention_input = LayerNormalization()(attention_output + attention_input)

        # 시간적 특성 반영 - 다중 해상도 처리
        daily_features = Dense(64, activation='elu')(attention_input)
        weekly_features = Dense(64, activation='elu')(
            AveragePooling1D(pool_size=5, strides=1, padding='same')(attention_input)
        )

        # 특성 결합
        combined = Concatenate()([
            daily_features[:, -1, :],
            weekly_features[:, -1, :],
            static_context
        ])

        # 최종 예측
        dense = Dense(128, activation='elu')(combined)
        dense = Dropout(0.1)(dense)
        dense = Dense(64, activation='elu')(dense)
        outputs = Dense(forecast_days)(dense)

        model = Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=Adam(learning_rate=self.default_lr),
            loss='huber'
        )

        return model

    def build_nbeats_model(self, input_shape, forecast_days):
        """N-BEATS 모델 구축"""
        # 하이퍼파라미터
        nb_blocks = 3
        hidden_units = 128
        theta_size = 5  # 베이시스 함수 수

        # 입력 및 초기화
        inputs = Input(shape=input_shape)
        x = Reshape((input_shape[0] * input_shape[1],))(inputs)

        # 블록 스택 구축
        backcast = x
        forecast = None

        for i in range(nb_blocks):
            # 기본 블록
            block_input = backcast

            # 완전 연결 레이어 스택
            for j in range(4):
                block_input = Dense(hidden_units, activation='relu')(block_input)

            # 예측 및 역투영
            theta = Dense(2 * theta_size)(block_input)

            # 역투영
            backcast_theta = Lambda(lambda x: x[:, :theta_size])(theta)
            backcast_out = Dense(input_shape[0] * input_shape[1])(backcast_theta)

            # 예측
            forecast_theta = Lambda(lambda x: x[:, theta_size:])(theta)
            forecast_out = Dense(forecast_days)(forecast_theta)

            # 잔차 연결
            backcast = Subtract()([backcast, backcast_out]) if 'Subtract' in dir(tf.keras.layers) else Lambda(
                lambda x: x[0] - x[1])([backcast, backcast_out])

            # 예측 합산
            if forecast is None:
                forecast = forecast_out
            else:
                forecast = Add()([forecast, forecast_out])

        model = Model(inputs=inputs, outputs=forecast)
        model.compile(
            optimizer=Adam(learning_rate=self.default_lr),
            loss='huber'
        )

        return model

    def build_tcn_model(self, input_shape, forecast_days):
        """Temporal Convolutional Network 모델 구축"""

        def residual_block(x, dilation_rate, nb_filters, kernel_size, dropout_rate=0.2):
            # Skip connection
            prev_x = x

            # Causal convolution 레이어 1
            conv1 = Conv1D(
                filters=nb_filters,
                kernel_size=kernel_size,
                dilation_rate=dilation_rate,
                padding='causal',
                kernel_regularizer=l2(self.l2_lambda)
            )(x)
            conv1 = BatchNormalization()(conv1)
            conv1 = Activation('relu')(conv1)
            conv1 = SpatialDropout1D(dropout_rate)(conv1)

            # Causal convolution 레이어 2
            conv2 = Conv1D(
                filters=nb_filters,
                kernel_size=kernel_size,
                dilation_rate=dilation_rate,
                padding='causal',
                kernel_regularizer=l2(self.l2_lambda)
            )(conv1)
            conv2 = BatchNormalization()(conv2)
            conv2 = Activation('relu')(conv2)
            conv2 = SpatialDropout1D(dropout_rate)(conv2)

            # Skip connection 처리
            if prev_x.shape[-1] != nb_filters:
                prev_x = Conv1D(nb_filters, kernel_size=1)(prev_x)

            # 결과 합산
            res_x = Add()([prev_x, conv2])

            return res_x

        # 모델 입력
        inputs = Input(shape=input_shape)

        # 초기 변환
        x = Conv1D(64, kernel_size=1, padding='same')(inputs)

        # 확장된 컨볼루션 블록
        x = residual_block(x, dilation_rate=1, nb_filters=64, kernel_size=3)
        x = residual_block(x, dilation_rate=2, nb_filters=64, kernel_size=3)
        x = residual_block(x, dilation_rate=4, nb_filters=64, kernel_size=3)
        x = residual_block(x, dilation_rate=8, nb_filters=64, kernel_size=3)

        # 글로벌 풀링
        x = GlobalAveragePooling1D()(x)

        # 예측 레이어
        x = Dense(64, activation='relu')(x)
        x = Dropout(0.2)(x)
        outputs = Dense(forecast_days)(x)

        model = Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=Adam(learning_rate=self.default_lr),
            loss='huber'
        )

        return model