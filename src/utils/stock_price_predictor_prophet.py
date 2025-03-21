# src/utils/stock_price_predictor_prophet.py

import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
import os
import traceback

logger = logging.getLogger('StockAnalysisApp.ProphetPredictor')


class ProphetIntegration:
    """Prophet 기반 예측 모델 통합 클래스"""

    def __init__(self):
        """초기화"""
        self.prophet_model = None
        self.is_prophet_available = self._check_prophet_available()

    def _check_prophet_available(self):  # 정의된 메서드명
        """Prophet 라이브러리 사용 가능 여부 확인"""
        try:
            import prophet
            return True
        except ImportError:
            logger.warning("Prophet 라이브러리를 찾을 수 없습니다. Prophet 기능은 사용할 수 없습니다.")
            return False

    def add_prophet_features(self, stock_data, forecast_days=30):
        """Prophet으로 특성 생성 및 예측 결과 반환"""
        try:
            if not self.is_prophet_available:
                logger.warning("Prophet 라이브러리가 설치되지 않았습니다. pip install prophet을 실행하세요.")
                return None

            import pandas as pd
            from prophet import Prophet

            # Prophet용 데이터 준비
            df = pd.DataFrame({
                'ds': stock_data.index,
                'y': stock_data['Close']
            })

            # Prophet 모델 학습
            model = Prophet(
                daily_seasonality=True,
                yearly_seasonality=True,
                weekly_seasonality=True,
                changepoint_prior_scale=0.05
            )
            model.fit(df)

            # 미래 날짜 생성
            future = model.make_future_dataframe(periods=forecast_days)

            # 예측
            forecast = model.predict(future)

            # 원본 데이터에 Prophet 특성 추가
            prophet_features = pd.DataFrame(index=stock_data.index)

            # Prophet 특성 추가 (추세, 계절성 등)
            for column in ['trend', 'yhat', 'yhat_lower', 'yhat_upper']:
                if column in forecast.columns:
                    prophet_data = forecast[forecast['ds'].isin(stock_data.index)]
                    prophet_features[f'prophet_{column}'] = prophet_data[column].values

            # 예측 값 추출
            prophet_predictions = forecast.iloc[-forecast_days:]['yhat'].values
            prophet_lower = forecast.iloc[-forecast_days:]['yhat_lower'].values
            prophet_upper = forecast.iloc[-forecast_days:]['yhat_upper'].values

            # 계절성 구성 요소 추가
            for component in ['yearly', 'weekly', 'daily']:
                if f'{component}' in forecast.columns:
                    prophet_features[f'prophet_{component}'] = forecast[forecast['ds'].isin(stock_data.index)][
                        f'{component}'].values

            # 모델 저장
            self.prophet_model = model

            return {
                'features': prophet_features,
                'predictions': prophet_predictions,
                'lower': prophet_lower,
                'upper': prophet_upper,
                'forecast_df': forecast.iloc[-forecast_days:][['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
            }

        except Exception as e:
            logger.error(f"Prophet 특성 생성 중 오류: {str(e)}")
            logger.error(traceback.format_exc())
            return None

    def plot_prophet_components(self, stock_data, save_path=None):
        """Prophet 구성요소 시각화"""
        try:
            if self.prophet_model is None:
                logger.error("Prophet 모델이 학습되지 않았습니다. add_prophet_features를 먼저 호출하세요.")
                return None

            from prophet.plot import plot_components
            import matplotlib.pyplot as plt

            # 모델 구성요소 플롯 생성
            fig = plot_components(self.prophet_model, self.prophet_model.predict(self.prophet_model.history))

            # 저장 경로가 지정된 경우 저장
            if save_path:
                fig.savefig(save_path)
                logger.info(f"Prophet 구성요소 그래프가 {save_path}에 저장되었습니다.")

            return fig

        except Exception as e:
            logger.error(f"Prophet 구성요소 시각화 중 오류: {str(e)}")
            logger.error(traceback.format_exc())
            return None