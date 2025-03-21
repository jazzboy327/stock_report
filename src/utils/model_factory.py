# src/utils/model_factory.py

import logging
from tensorflow.keras.models import Model
import traceback
import numpy as np

logger = logging.getLogger('StockAnalysisApp.ModelFactory')


class ModelFactory:
    """
    다양한 모델을 생성하고 관리하는 팩토리 클래스
    StockPricePredictor 클래스와 함께 사용
    """

    def __init__(self, stock_predictor):
        """
        ModelFactory 초기화

        Args:
            stock_predictor: StockPricePredictor 인스턴스
        """
        self.predictor = stock_predictor
        self.available_models = {
            'lstm': self.predictor.build_lstm_model,
            'transformer': self.predictor.build_transformer_model,
            'ensemble': self.predictor.build_ensemble_model,
            'tcn': self.build_tcn_model,
            'tft': self.build_tft_model,
            'nbeats': self.build_nbeats_model,
            'prophet': self.build_prophet_model,
            'hybrid': self.predictor.build_hybrid_ensemble
        }

    def create_model(self, model_type, input_shape, forecast_days):
        """
        모델 유형에 따라 적절한 모델 생성

        Args:
            model_type: 모델 유형 ('lstm', 'transformer', 'ensemble', 'tcn', 'tft', 'nbeats', 'hybrid')
            input_shape: 입력 형태 (tuple)
            forecast_days: 예측 기간 (일)

        Returns:
            생성된 모델 인스턴스
        """
        try:
            model_type = model_type.lower()

            if model_type not in self.available_models:
                logger.warning(f"지원하지 않는 모델 유형: {model_type}, 기본 LSTM 모델을 사용합니다.")
                model_type = 'lstm'

            logger.info(f"{model_type} 모델 생성 중...")
            model_builder = self.available_models[model_type]

            # 'hybrid' 모델은 특별히 처리
            if model_type == 'hybrid':
                # 하이브리드 모델은 특별 처리가 필요함을 알림
                logger.info("하이브리드 모델은 직접 호출 방식으로 처리됩니다.")
                return "HYBRID_MODEL_DIRECT_CALL"  # 특별한 마커 반환
            else:
                # 일반 모델은 정상적으로 생성하여 반환
                return model_builder(input_shape, forecast_days)

        except Exception as e:
            logger.error(f"모델 생성 중 오류 발생: {str(e)}")
            logger.error(traceback.format_exc())
            # 오류 발생 시 기본 LSTM 모델 반환
            return self.predictor.build_lstm_model(input_shape, forecast_days)

    def build_tcn_model(self, input_shape, forecast_days):
        """
        Temporal Convolutional Network 모델 구축

        Args:
            input_shape: 입력 형태 (tuple)
            forecast_days: 예측 기간 (일)

        Returns:
            TCN 모델 인스턴스
        """
        try:
            # 기존 TCN 구현 불러오기
            from src.utils.stock_price_predictor_enhanced import StockPricePredictorEnhanced

            # 임시 인스턴스 생성
            enhanced_predictor = StockPricePredictorEnhanced()
            enhanced_predictor.l2_lambda = self.predictor.l2_lambda
            enhanced_predictor.default_lr = self.predictor.default_lr

            # TCN 모델 생성
            return enhanced_predictor.build_tcn_model(input_shape, forecast_days)

        except ImportError:
            logger.warning("StockPricePredictorEnhanced 모듈을 불러올 수 없습니다. 기본 LSTM 모델을 사용합니다.")
            return self.predictor.build_lstm_model(input_shape, forecast_days)
        except Exception as e:
            logger.error(f"TCN 모델 생성 중 오류: {str(e)}")
            logger.error(traceback.format_exc())
            return self.predictor.build_lstm_model(input_shape, forecast_days)

    def build_tft_model(self, input_shape, forecast_days):
        """
        Temporal Fusion Transformer 모델 구축

        Args:
            input_shape: 입력 형태 (tuple)
            forecast_days: 예측 기간 (일)

        Returns:
            TFT 모델 인스턴스
        """
        try:
            # 기존 TFT 구현 불러오기
            from src.utils.stock_price_predictor_enhanced import StockPricePredictorEnhanced

            # 임시 인스턴스 생성
            enhanced_predictor = StockPricePredictorEnhanced()
            enhanced_predictor.l2_lambda = self.predictor.l2_lambda
            enhanced_predictor.default_lr = self.predictor.default_lr

            # TFT 모델 생성
            return enhanced_predictor.build_tft_model(input_shape, forecast_days)

        except ImportError:
            logger.warning("StockPricePredictorEnhanced 모듈을 불러올 수 없습니다. 기본 LSTM 모델을 사용합니다.")
            return self.predictor.build_lstm_model(input_shape, forecast_days)
        except Exception as e:
            logger.error(f"TFT 모델 생성 중 오류: {str(e)}")
            logger.error(traceback.format_exc())
            return self.predictor.build_lstm_model(input_shape, forecast_days)

    def build_nbeats_model(self, input_shape, forecast_days):
        """
        N-BEATS 모델 구축

        Args:
            input_shape: 입력 형태 (tuple)
            forecast_days: 예측 기간 (일)

        Returns:
            N-BEATS 모델 인스턴스
        """
        try:
            # 기존 N-BEATS 구현 불러오기
            from src.utils.stock_price_predictor_enhanced import StockPricePredictorEnhanced

            # 임시 인스턴스 생성
            enhanced_predictor = StockPricePredictorEnhanced()
            enhanced_predictor.l2_lambda = self.predictor.l2_lambda
            enhanced_predictor.default_lr = self.predictor.default_lr

            # N-BEATS 모델 생성
            return enhanced_predictor.build_nbeats_model(input_shape, forecast_days)

        except ImportError:
            logger.warning("StockPricePredictorEnhanced 모듈을 불러올 수 없습니다. 기본 LSTM 모델을 사용합니다.")
            return self.predictor.build_lstm_model(input_shape, forecast_days)
        except Exception as e:
            logger.error(f"N-BEATS 모델 생성 중 오류: {str(e)}")
            logger.error(traceback.format_exc())
            return self.predictor.build_lstm_model(input_shape, forecast_days)

    def build_prophet_model(self, input_shape, forecast_days):
        """Prophet 모델 구축"""
        try:
            logger.info("Prophet 모델은 직접 호출 방식으로 처리됩니다.")
            return "PROPHET_MODEL_DIRECT_CALL"  # 특별한 마커 반환
        except Exception as e:
            logger.error(f"Prophet 모델 생성 중 오류: {str(e)}")
            logger.error(traceback.format_exc())
            return self.predictor.build_lstm_model(input_shape, forecast_days)