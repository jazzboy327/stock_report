# src/views/comprehensive_report_view.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import logging
from datetime import datetime
import json
import requests
import base64
import io
import matplotlib.pyplot as plt
import traceback
import os
from dotenv import load_dotenv
import time

logger = logging.getLogger('StockAnalysisApp.ComprehensiveReportView')


class ComprehensiveReportView:
    """종합 분석 리포트 뷰를 담당하는 클래스"""

    def __init__(self):
        """뷰 초기화"""
        # .env 파일 로드 시도
        load_dotenv()
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self._initialize_data_store()

        # 주가 예측기 초기화
        from src.utils.stock_price_predictor import StockPricePredictor
        self.price_predictor = StockPricePredictor()

    def _initialize_data_store(self):
        """데이터 저장소 초기화 - 확장된 기간 및 기능 추가"""
        if 'comprehensive_data' not in st.session_state:
            st.session_state.comprehensive_data = {
                'stock_detail': {},
                'technical_analysis': {},
                'investor_trends': {},
                'financial_analysis': {
                    'risk_metrics': {},
                    'growth_data': {}
                },
                'trading_signals': {},
                'prediction_result': None,
                'last_update': {},
                'analysis_cache': {}
            }

        if 'report_options' not in st.session_state:
            st.session_state.report_options = {
                'has_generated': False,
                'llm_model': 'gpt-4o',
                'temperature': 0.7,
                'max_tokens': 2000,
                'include_prediction': True,  # 주가 예측 포함 여부 기본값을 True로 변경
                'language': '한국어',  # 언어 옵션 추가
                'analysis_depth': '고급'  # 분석 깊이 옵션 추가
            }

        # 주가 예측 관련 모델 설정 초기화
        if 'model_settings' not in st.session_state:
            st.session_state.model_settings = {
                'prediction_days': 30,
                'model_type': 'LSTM',
                'use_ensemble': False,
                'train_size': 0.8,
                'use_auto_features': True,  # 자동 특성 선택 알고리즘을 기본값으로 설정
                'prediction_features': ['Close']
            }

    def display(self, company_info, stock_info, market_data, analysis_results, history_df=None):
        """종합 분석 리포트 탭 표시 - 개선된 설정 섹션"""
        st.header("📑 종합 분석 리포트")

        # AI 설정 섹션
        st.subheader("🛠️ AI 설정")

        # AI 설정 폼
        with st.form(key="ai_settings_form"):
            # 1. 주가 예측 설정
            st.markdown("### 주가 예측 설정")

            pred_col1, pred_col2, pred_col3 = st.columns(3)

            with pred_col1:
                # 모델 유형 설정 - 확장된 모델 옵션
                model_options = ["LSTM", "Transformer", "앙상블", "TCN", "TFT", "N-BEATS", "Prophet", "하이브리드"]
                model_descriptions = {
                    "LSTM": "시계열 예측에 적합한 기본 모델",
                    "Transformer": "복잡한 패턴 인식에 강한 모델",
                    "앙상블": "LSTM과 Transformer 모델을 결합한 모델",
                    "TCN": "시간적 의존성을 효과적으로 포착하는 컨볼루션 기반 모델",
                    "TFT": "시간 특성을 고려한 트랜스포머 모델",
                    "N-BEATS": "계층적 구조와 역투영 메커니즘을 활용한 모델",
                    "Prophet": "Facebook의 시계열 예측 라이브러리 기반 모델",
                    "하이브리드": "모든 모델을 결합한 최적화된 앙상블 모델"
                }

                current_model = st.session_state.model_settings.get('model_type', 'LSTM')
                model_type = st.selectbox(
                    "모델 유형",
                    options=model_options,
                    index=model_options.index(current_model) if current_model in model_options else 0,
                    help="예측에 사용할 모델 유형을 선택합니다.",
                    format_func=lambda x: f"{x} - {model_descriptions[x]}"
                )

            with pred_col2:
                # 예측 기간 설정
                prediction_days = st.slider(
                    "예측 기간 (일)",
                    min_value=30,
                    max_value=180,
                    value=st.session_state.model_settings.get('prediction_days', 60),
                    step=30,
                    help="몇 일 후까지 예측할지 설정합니다."
                )

            with pred_col3:
                # Early Stopping Patience 설정 추가
                patience = st.slider(
                    "Early Stopping Patience",
                    min_value=5,
                    max_value=30,
                    value=st.session_state.model_settings.get('patience', 20),
                    step=5,
                    help="성능 개선이 없을 때 몇 번의 epoch를 더 기다릴지 설정합니다."
                )

            # 데이터 수집 기간 설정 (히스토리)
            history_years = st.slider(
                "데이터 수집 기간 (년)",
                min_value=1,
                max_value=5,
                value=st.session_state.model_settings.get('history_years', 3),
                step=1,
                help="예측에 사용할 과거 데이터 기간을 설정합니다."
            )

            # 신뢰도 설정 - 몬테카를로 방식 선택 및 신뢰 수준
            confidence_col1, confidence_col2 = st.columns(2)

            with confidence_col1:
                use_monte_carlo = st.checkbox(
                    "몬테카를로 시뮬레이션 활성화",
                    value=st.session_state.model_settings.get('use_monte_carlo', True),
                    help="다양한 시나리오를 시뮬레이션하여 더 정확한 신뢰 구간을 계산합니다."
                )

            with confidence_col2:
                confidence_options = [0.8, 0.9, 0.95, 0.99]
                current_confidence = st.session_state.model_settings.get('confidence_level', 0.9)
                confidence_level = st.select_slider(
                    "신뢰 수준",
                    options=confidence_options,
                    value=current_confidence if current_confidence in confidence_options else 0.9,
                    format_func=lambda x: f"{int(x * 100)}%",
                    help="예측 신뢰 구간의 확률적 범위입니다."
                )

            # 데이터 미리보기 - 전체 컬럼 표기 (고급 설정 위로 이동)
            if history_df is not None:
                with st.expander("데이터 미리보기", expanded=False):
                    # 기본 데이터
                    st.subheader("기본 주가 데이터")
                    st.dataframe(history_df.tail(10))

                    # 확장된 기술적 지표 데이터
                    st.subheader("확장된 기술적 지표")
                    # 기술적 지표 계산 - 모든 컬럼 표시 (수정)
                    from src.utils.stock_price_predictor import StockPricePredictor
                    predictor = StockPricePredictor()
                    extended_df = predictor._add_technical_indicators(history_df)
                    st.dataframe(extended_df.tail(10))

            # 새로운 고급 모델에 대한 추가 설정 섹션
            self._display_enhanced_model_settings(model_type)

            # 특성 선택 섹션 (고급 설정)
            with st.expander("고급 설정", expanded=False):
                # 자동 특성 선택 활성화 여부
                use_auto_features = st.checkbox(
                    "자동 특성 선택 활성화",
                    value=st.session_state.model_settings.get('use_auto_features', True),
                    help="최적의 예측 성능을 위해 중요한 특성을 자동으로 선택합니다."
                )

                if not use_auto_features:
                    # 수동 특성 선택
                    feature_options = [
                        "Close", "Open", "High", "Low", "Volume",
                        "MA5", "MA10", "MA20", "MA60", "MA120",
                        "RSI", "MACD", "MACD_Signal", "MACD_Hist",
                        "BB_Middle", "BB_Upper", "BB_Lower", "BB_Width",
                        "Stoch_K", "Stoch_D", "ATR", "OBV",
                        "Price_Change_1D", "Price_Change_5D", "Price_Change_20D",
                        "Volume_Change_1D", "Volume_Change_5D", "Volume_Ratio"
                    ]

                    default_features = st.session_state.model_settings.get('prediction_features', ["Close"])
                    # 기본값이 feature_options에 포함되어 있는지 확인
                    valid_defaults = [f for f in default_features if f in feature_options]
                    if not valid_defaults:
                        valid_defaults = ["Close"]

                    selected_features = st.multiselect(
                        "예측에 사용할 특성 선택",
                        options=feature_options,
                        default=valid_defaults,
                        help="주가 예측에 사용할 특성을 선택합니다. 5개 내외를 권장합니다."
                    )

                    # 선택된 특성이 없으면 기본값 사용
                    if not selected_features:
                        st.warning("특성을 선택하지 않으면 자동 특성 선택이 활성화됩니다.")
                        selected_features = ["Auto"]
                        use_auto_features = True
                else:
                    selected_features = ["Auto"]

            # 2. 리포트 생성 설정
            st.markdown("### 리포트 생성 설정")

            # OpenAI API 설정
            if self.openai_api_key is None:
                self.openai_api_key = st.text_input(
                    "OpenAI API 키를 입력하세요",
                    type="password",
                    help="AI 분석 생성을 위해 필요합니다"
                )
            else:
                st.success("API 키가 설정되었습니다. 변경이 필요하면 아래에 새 값을 입력하세요.")
                new_key = st.text_input(
                    "OpenAI API 키 변경 (필요시)",
                    type="password",
                    value="",
                    help="API 키를 변경하려면 새 값을 입력하세요"
                )
                if new_key:
                    self.openai_api_key = new_key

            # LLM 모델 설정
            llm_col1, llm_col2, llm_col3 = st.columns(3)

            with llm_col1:
                model_options = ["gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"]
                current_model = st.session_state.report_options.get('llm_model', 'gpt-4o')
                model_index = 0  # 기본값
                if current_model in model_options:
                    model_index = model_options.index(current_model)

                model = st.selectbox(
                    "LLM 모델",
                    options=model_options,
                    index=model_index,
                    help="분석에 사용할 OpenAI 모델을 선택하세요."
                )

            with llm_col2:
                temperature = st.slider(
                    "Temperature",
                    min_value=0.0,
                    max_value=1.0,
                    value=st.session_state.report_options.get('temperature', 0.7),
                    step=0.1,
                    help="값이 높을수록 더 창의적인 결과가 생성됩니다."
                )

            with llm_col3:
                max_tokens = st.slider(
                    "최대 토큰 수",
                    min_value=1000,
                    max_value=8000,
                    value=st.session_state.report_options.get('max_tokens', 2000),
                    step=500,
                    help="생성할 텍스트의 최대 길이를 설정합니다."
                )

            # 분석 깊이 및 언어 설정
            depth_lang_col1, depth_lang_col2 = st.columns(2)

            with depth_lang_col1:
                depth_options = ["기본", "심화", "전문가"]
                default_depth = st.session_state.report_options.get('analysis_depth', '심화')
                # '고급'이 아닌 '심화'로 기본값 설정
                if default_depth not in depth_options:
                    default_depth = '심화'

                analysis_depth = st.radio(
                    "분석 깊이",
                    options=depth_options,
                    index=depth_options.index(default_depth),
                    horizontal=True,
                    help="생성될 분석 보고서의 상세도를 설정합니다."
                )

            with depth_lang_col2:
                lang_options = ["한국어", "영어"]
                default_lang = st.session_state.report_options.get('language', '한국어')
                if default_lang not in lang_options:
                    default_lang = '한국어'

                language = st.radio(
                    "언어",
                    options=lang_options,
                    index=lang_options.index(default_lang),
                    horizontal=True,
                    help="보고서 생성 언어를 선택합니다."
                )

            # 폼 제출 버튼 (반드시 포함되어야 함)
            submit_button = st.form_submit_button("설정 저장")

            # 설정이 저장되면 상태 업데이트
            if submit_button:
                # 모델 설정 업데이트
                st.session_state.model_settings.update({
                    'model_type': model_type,
                    'prediction_days': prediction_days,
                    'history_years': history_years,
                    'patience': patience,
                    'use_auto_features': use_auto_features,
                    'prediction_features': selected_features,
                    'use_monte_carlo': use_monte_carlo,
                    'confidence_level': confidence_level
                })

                # 추가 모델별 설정 저장
                if model_type == "TFT":
                    st.session_state.model_settings.update({
                        'tft_num_heads': st.session_state.model_settings.get('tft_num_heads', 4),
                        'tft_encoder_layers': st.session_state.model_settings.get('tft_encoder_layers', 2),
                        'tft_multiresolution': st.session_state.model_settings.get('tft_multiresolution', True)
                    })
                elif model_type == "TCN":
                    st.session_state.model_settings.update({
                        'tcn_kernel_size': st.session_state.model_settings.get('tcn_kernel_size', 3),
                        'tcn_filters': st.session_state.model_settings.get('tcn_filters', 64),
                        'tcn_layers': st.session_state.model_settings.get('tcn_layers', 4)
                    })
                elif model_type == "N-BEATS":
                    st.session_state.model_settings.update({
                        'nbeats_blocks': st.session_state.model_settings.get('nbeats_blocks', 3),
                        'nbeats_units': st.session_state.model_settings.get('nbeats_units', 128),
                        'nbeats_seasonal': st.session_state.model_settings.get('nbeats_seasonal', True)
                    })
                elif model_type == "하이브리드":
                    st.session_state.model_settings.update({
                        'use_prophet': st.session_state.model_settings.get('use_prophet', True),
                        'auto_weights': st.session_state.model_settings.get('auto_weights', True)
                    })

                # 리포트 옵션 업데이트
                st.session_state.report_options.update({
                    'has_generated': st.session_state.report_options.get('has_generated', False),
                    'llm_model': model,
                    'temperature': temperature,
                    'max_tokens': max_tokens,
                    'analysis_depth': analysis_depth,
                    'language': language
                })

                st.success("설정이 저장되었습니다.")

            # 폼 종료 후, 현재 설정 정보 표시
            st.subheader("📋 현재 설정 정보")

            # 탭으로 분류하여 설정 정보 표시
            tab1, tab2 = st.tabs(["주가 예측 설정", "AI 분석 설정"])

            with tab1:
                # 주가 예측 설정 표시
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("##### 기본 설정")
                    st.markdown(f"""
                    - **모델 유형**: {st.session_state.model_settings['model_type']}
                    - **예측 기간**: {st.session_state.model_settings['prediction_days']}일
                    - **데이터 수집 기간**: {st.session_state.model_settings.get('history_years', 3)}년
                    - **Early Stopping Patience**: {st.session_state.model_settings.get('patience', 20)}
                    """)

                with col2:
                    st.markdown("##### 신뢰도 및 특성 설정")

                    # 자동 특성 선택 여부에 따른 표시
                    if st.session_state.model_settings.get('use_auto_features', True):
                        feature_info = "자동 특성 선택 활성화"
                    else:
                        feature_info = f"수동 선택: {', '.join(st.session_state.model_settings.get('prediction_features', ['Close']))}"

                    st.markdown(f"""
                    - **특성 선택**: {feature_info}
                    - **몬테카를로 시뮬레이션**: {'활성화' if st.session_state.model_settings.get('use_monte_carlo', True) else '비활성화'}
                    - **신뢰 수준**: {int(st.session_state.model_settings.get('confidence_level', 0.9) * 100)}%
                    """)

            with tab2:
                # AI 분석 설정 표시
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("##### 모델 설정")
                    st.markdown(f"""
                    - **LLM 모델**: {st.session_state.report_options['llm_model']}
                    - **Temperature**: {st.session_state.report_options['temperature']}
                    - **최대 토큰 수**: {st.session_state.report_options['max_tokens']}
                    """)

                with col2:
                    st.markdown("##### 출력 설정")
                    st.markdown(f"""
                    - **분석 깊이**: {st.session_state.report_options['analysis_depth']}
                    - **언어**: {st.session_state.report_options['language']}
                    - **리포트 생성 여부**: {'완료' if st.session_state.report_options.get('has_generated', False) else '미생성'}
                    """)

            # API 키 상태 표시
            api_key_status = "설정됨 ✅" if self.openai_api_key else "미설정 ❌"
            st.markdown(f"**OpenAI API 키 상태**: {api_key_status}")

            # 마지막 설정 변경 시간 표시 (옵션)
            if 'last_settings_update' not in st.session_state:
                st.session_state.last_settings_update = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            elif submit_button:
                st.session_state.last_settings_update = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            st.caption(f"마지막 설정 변경: {st.session_state.last_settings_update}")


        # 종합 리포트 생성 버튼 (폼 밖에 위치)
        if st.button("종합 리포트 생성", help="AI 기반 주가 예측과 종합 분석 리포트를 생성합니다."):
            if not self.openai_api_key:
                st.error("AI 기반 분석을 위해 OpenAI API 키가 필요합니다.")
                return

            # 리포트 생성 프로세스 실행
            self._generate_comprehensive_report(
                company_info,
                stock_info,
                market_data,
                analysis_results,
                history_df
            )

    def _display_enhanced_model_settings(self, model_type):
        """확장된 모델 유형에 대한 고급 설정 표시"""
        # 하이브리드 앙상블 설정
        if model_type == "하이브리드":
            st.write("#### 🔄 하이브리드 앙상블 설정")
            st.info("하이브리드 앙상블은 여러 모델의 예측을 결합하여 정확도를 높입니다.")

            # Prophet 사용 여부
            use_prophet = st.checkbox(
                "Prophet 모델 포함",
                value=st.session_state.model_settings.get('use_prophet', True),
                help="Facebook의 Prophet 모델을 앙상블에 포함합니다. 설치가 필요할 수 있습니다."
            )
            st.session_state.model_settings['use_prophet'] = use_prophet

            # 가중치 자동 계산 여부
            auto_weights = st.checkbox(
                "가중치 자동 계산",
                value=st.session_state.model_settings.get('auto_weights', True),
                help="각 모델의 성능을 기반으로 가중치를 자동 계산합니다."
            )
            st.session_state.model_settings['auto_weights'] = auto_weights

            # 수동 가중치 설정
            if not auto_weights:
                st.write("##### 모델별 가중치 설정")
                col1, col2, col3 = st.columns(3)

                with col1:
                    lstm_weight = st.slider(
                        "LSTM 가중치",
                        min_value=0.0,
                        max_value=1.0,
                        value=st.session_state.model_settings.get('lstm_weight', 0.3),
                        step=0.05
                    )
                    st.session_state.model_settings['lstm_weight'] = lstm_weight

                    tcn_weight = st.slider(
                        "TCN 가중치",
                        min_value=0.0,
                        max_value=1.0,
                        value=st.session_state.model_settings.get('tcn_weight', 0.2),
                        step=0.05
                    )
                    st.session_state.model_settings['tcn_weight'] = tcn_weight

                with col2:
                    transformer_weight = st.slider(
                        "Transformer 가중치",
                        min_value=0.0,
                        max_value=1.0,
                        value=st.session_state.model_settings.get('transformer_weight', 0.2),
                        step=0.05
                    )
                    st.session_state.model_settings['transformer_weight'] = transformer_weight

                    tft_weight = st.slider(
                        "TFT 가중치",
                        min_value=0.0,
                        max_value=1.0,
                        value=st.session_state.model_settings.get('tft_weight', 0.1),
                        step=0.05
                    )
                    st.session_state.model_settings['tft_weight'] = tft_weight

                with col3:
                    ensemble_weight = st.slider(
                        "앙상블 가중치",
                        min_value=0.0,
                        max_value=1.0,
                        value=st.session_state.model_settings.get('ensemble_weight', 0.1),
                        step=0.05
                    )
                    st.session_state.model_settings['ensemble_weight'] = ensemble_weight

                    prophet_weight = st.slider(
                        "Prophet 가중치",
                        min_value=0.0,
                        max_value=1.0,
                        value=st.session_state.model_settings.get('prophet_weight', 0.1),
                        step=0.05
                    )
                    st.session_state.model_settings['prophet_weight'] = prophet_weight

                # 가중치 합계 확인 및 정규화
                total_weight = lstm_weight + transformer_weight + ensemble_weight + tcn_weight + tft_weight
                if use_prophet:
                    total_weight += prophet_weight

                # 가중치 합계가 1을 초과하면 경고
                if total_weight > 1.0:
                    st.warning(f"가중치 합계가 1을 초과합니다 ({total_weight:.2f}). 자동으로 정규화됩니다.")

        # TFT 모델 설정
        elif model_type == "TFT":
            st.write("#### 🔄 Temporal Fusion Transformer 설정")

            col1, col2 = st.columns(2)

            with col1:
                # 어텐션 헤드 수
                num_heads = st.slider(
                    "어텐션 헤드 수",
                    min_value=1,
                    max_value=8,
                    value=st.session_state.model_settings.get('tft_num_heads', 4),
                    step=1,
                    help="멀티헤드 어텐션에 사용할 헤드의 수입니다. 수가 많을수록 다양한 특성 관계를 포착할 수 있습니다."
                )
                st.session_state.model_settings['tft_num_heads'] = num_heads

            with col2:
                # 인코더 레이어 수
                encoder_layers = st.slider(
                    "인코더 레이어 수",
                    min_value=1,
                    max_value=4,
                    value=st.session_state.model_settings.get('tft_encoder_layers', 2),
                    step=1,
                    help="트랜스포머 인코더 블록의 수입니다. 수가 많을수록 복잡한 패턴을 포착할 수 있지만 학습이 어려워집니다."
                )
                st.session_state.model_settings['tft_encoder_layers'] = encoder_layers

            # 다중 해상도 처리 활성화
            enable_multiresolution = st.checkbox(
                "다중 해상도 처리 활성화",
                value=st.session_state.model_settings.get('tft_multiresolution', True),
                help="일별, 주별 등 여러 시간 해상도의 패턴을 동시에 고려합니다."
            )
            st.session_state.model_settings['tft_multiresolution'] = enable_multiresolution

        # TCN 모델 설정
        elif model_type == "TCN":
            st.write("#### 🔄 Temporal Convolutional Network 설정")

            col1, col2 = st.columns(2)

            with col1:
                # 커널 크기
                kernel_size = st.select_slider(
                    "컨볼루션 커널 크기",
                    options=[2, 3, 5, 7],
                    value=st.session_state.model_settings.get('tcn_kernel_size', 3),
                    help="컨볼루션 필터의 크기입니다. 클수록 더 넓은 시간 범위를 고려합니다."
                )
                st.session_state.model_settings['tcn_kernel_size'] = kernel_size

            with col2:
                # 필터 수
                num_filters = st.select_slider(
                    "필터 수",
                    options=[32, 64, 128, 256],
                    value=st.session_state.model_settings.get('tcn_filters', 64),
                    help="각 컨볼루션 레이어의 필터 수입니다. 많을수록 복잡한 패턴을 포착할 수 있습니다."
                )
                st.session_state.model_settings['tcn_filters'] = num_filters

            # 확장 레이어 수
            num_layers = st.slider(
                "확장 컨볼루션 레이어 수",
                min_value=2,
                max_value=8,
                value=st.session_state.model_settings.get('tcn_layers', 4),
                step=1,
                help="더 많은 레이어는 더 긴 시간 의존성을 포착할 수 있습니다."
            )
            st.session_state.model_settings['tcn_layers'] = num_layers

        # N-BEATS 모델 설정
        elif model_type == "N-BEATS":
            st.write("#### 🔄 N-BEATS 설정")

            col1, col2 = st.columns(2)

            with col1:
                # 블록 수
                num_blocks = st.slider(
                    "블록 수",
                    min_value=2,
                    max_value=6,
                    value=st.session_state.model_settings.get('nbeats_blocks', 3),
                    step=1,
                    help="N-BEATS 블록 수입니다. 블록이 많을수록 복잡한 시계열 구성요소를 포착할 수 있습니다."
                )
                st.session_state.model_settings['nbeats_blocks'] = num_blocks

            with col2:
                # 히든 유닛 수
                hidden_units = st.select_slider(
                    "히든 유닛 수",
                    options=[64, 128, 256, 512],
                    value=st.session_state.model_settings.get('nbeats_units', 128),
                    help="각 블록의 히든 레이어 크기입니다."
                )
                st.session_state.model_settings['nbeats_units'] = hidden_units

            # 계절성 블록 추가
            add_seasonal = st.checkbox(
                "계절성 블록 추가",
                value=st.session_state.model_settings.get('nbeats_seasonal', True),
                help="주기적인 패턴을 포착하기 위한 특수 블록을 추가합니다."
            )
            st.session_state.model_settings['nbeats_seasonal'] = add_seasonal

    def _generate_comprehensive_report(self, company_info, stock_info, market_data, analysis_results, history_df):
        """종합 리포트 생성 프로세스 - 새 모델 지원 추가"""
        try:
            with st.spinner("종합 리포트 생성 중..."):
                # 현재 기업 식별 정보 저장
                current_company = company_info['symbol']

                # 세션 상태에 현재 기업 정보 저장
                st.session_state.current_report_company = current_company

                # 1. 주가 예측 실행
                with st.spinner("주가 예측 모델 실행 중..."):
                    st.info("주가 예측 모델을 학습하고 예측 중입니다. 잠시 기다려주세요...")
                    progress_bar = st.progress(0)

                    try:
                        # 데이터 충분성 확인
                        MIN_REQUIRED_DATA = 60  # 최소 60일치 데이터 필요

                        if history_df is None or len(history_df) < MIN_REQUIRED_DATA:
                            error_msg = f"데이터가 부족합니다. 최소 {MIN_REQUIRED_DATA}일 이상의 데이터가 필요합니다. (현재: {len(history_df) if history_df is not None else 0}일)"
                            st.error(error_msg)
                            logger.error(error_msg)

                            # 이전에 저장된 예측 결과가 있으면 현재 회사 정보와 비교하여 삭제
                            if ('prediction_result' in st.session_state.comprehensive_data and
                                    'current_prediction_symbol' in st.session_state and
                                    st.session_state.current_prediction_symbol != current_company):

                                # 이전 기업의 예측 결과를 삭제
                                logger.info(f"이전 기업({st.session_state.current_prediction_symbol})의 예측 결과 제거")
                                st.session_state.comprehensive_data['prediction_result'] = None

                                if 'model_evaluation' in st.session_state.comprehensive_data:
                                    st.session_state.comprehensive_data['model_evaluation'] = None

                            # 기본 예측 결과 생성 (오류 메시지 포함)
                            default_prediction = {
                                'error': True,
                                'error_message': error_msg,
                                'company_symbol': current_company,
                                'dates': [],
                                'predicted': [],
                                'last_price': history_df['Close'].iloc[
                                    -1] if history_df is not None and not history_df.empty else 0,
                                'confidence_high': [],
                                'confidence_low': [],
                                'historical_volatility': 0,
                                'trend_strength': 0
                            }

                            # 세션 상태에 예측 실패 정보 저장
                            st.session_state.comprehensive_data['prediction_result'] = default_prediction
                            st.session_state.comprehensive_data['prediction_status'] = 'failed'
                            st.session_state.current_prediction_symbol = current_company

                            progress_bar.empty()
                            raise ValueError(error_msg)

                        # 모델 설정 가져오기
                        model_type = st.session_state.model_settings['model_type']

                        # 모델 유형 정규화 - 한글/영문 모델명 처리
                        if model_type == "앙상블":
                            model_type = "ensemble"
                        elif model_type == "하이브리드":
                            model_type = "hybrid"

                        # 모델 유형별 특수 처리
                        if model_type.lower() == "hybrid":
                            # 하이브리드 앙상블 설정 추가
                            use_prophet = st.session_state.model_settings.get('use_prophet', True)
                            auto_weights = st.session_state.model_settings.get('auto_weights', True)

                            # 하이브리드 앙상블 설정 로깅
                            logger.info(f"하이브리드 앙상블 설정 - Prophet 사용: {use_prophet}, 자동 가중치: {auto_weights}")

                        elif model_type.upper() == "TFT":
                            # TFT 모델 특수 설정
                            tft_num_heads = st.session_state.model_settings.get('tft_num_heads', 4)
                            tft_encoder_layers = st.session_state.model_settings.get('tft_encoder_layers', 2)
                            tft_multiresolution = st.session_state.model_settings.get('tft_multiresolution', True)

                            # TFT 설정 로깅
                            logger.info(
                                f"TFT 모델 설정 - 헤드: {tft_num_heads}, 레이어: {tft_encoder_layers}, 다중해상도: {tft_multiresolution}")

                        elif model_type.upper() == "TCN":
                            # TCN 모델 특수 설정
                            tcn_kernel_size = st.session_state.model_settings.get('tcn_kernel_size', 3)
                            tcn_filters = st.session_state.model_settings.get('tcn_filters', 64)
                            tcn_layers = st.session_state.model_settings.get('tcn_layers', 4)

                            # TCN 설정 로깅
                            logger.info(f"TCN 모델 설정 - 커널: {tcn_kernel_size}, 필터: {tcn_filters}, 레이어: {tcn_layers}")

                        elif model_type.upper() == "N-BEATS":
                            # N-BEATS 모델 특수 설정
                            nbeats_blocks = st.session_state.model_settings.get('nbeats_blocks', 3)
                            nbeats_units = st.session_state.model_settings.get('nbeats_units', 128)
                            nbeats_seasonal = st.session_state.model_settings.get('nbeats_seasonal', True)

                            # N-BEATS 설정 로깅
                            logger.info(
                                f"N-BEATS 모델 설정 - 블록: {nbeats_blocks}, 유닛: {nbeats_units}, 계절성: {nbeats_seasonal}")

                        # 예측 기간 설정 가져오기
                        forecast_days = st.session_state.model_settings['prediction_days']

                        # 자동 특성 선택 알고리즘 사용 여부 확인
                        use_auto_features = st.session_state.model_settings.get('use_auto_features', True)

                        # 진행 상황 업데이트
                        if progress_bar:
                            progress_bar.progress(10, text="데이터 전처리 및 특성 선택 중...")

                        # 특성 선택
                        if use_auto_features:
                            # 자동 특성 선택 알고리즘 사용
                            selected_features = self.price_predictor.auto_feature_selection(history_df)
                            st.info(f"자동 선택된 특성: {', '.join(selected_features)}")
                        else:
                            # 사용자 선택 특성 사용
                            selected_features = st.session_state.model_settings['prediction_features']

                        # 로그에 설정값 기록
                        logger.info(
                            f"주가 예측 실행: 심볼={company_info['symbol']}, 모델={model_type}, 예측기간={forecast_days}일, 특성={selected_features}")

                        if progress_bar:
                            progress_bar.progress(20, text="모델 학습 준비 중...")

                        # 먼저 모델 학습
                        self.price_predictor.train_model(
                            history_df,
                            model_type=model_type,
                            forecast_days=forecast_days,
                            features=selected_features
                        )

                        if progress_bar:
                            progress_bar.progress(50, text="모델 학습 완료, 예측 중...")

                        # 예측 실행
                        prediction_result = self.price_predictor.predict_future(
                            history_df,
                            model_type=model_type,
                            days=forecast_days,
                            features=selected_features
                        )

                        # 진행 상황 업데이트
                        if progress_bar:
                            progress_bar.progress(100, text="예측 완료!")

                        if prediction_result:
                            # 모델 평가 실행
                            model_evaluation = self.price_predictor.evaluate_model(
                                history_df,
                                model_type=model_type,
                                features=selected_features
                            )

                            # 상대 RMSE 계산 추가
                            if 'rmse' in model_evaluation:
                                last_price = history_df['Close'].iloc[-1]
                                model_evaluation['relative_rmse'] = (model_evaluation['rmse'] / last_price) * 100

                            # 예측 결과와 모델 평가 결과 저장
                            st.session_state.comprehensive_data.update({
                                "prediction_result": prediction_result,
                                "model_evaluation": model_evaluation,
                                "selected_features": selected_features,
                                "history_df": history_df,  # 히스토리 데이터도 저장하여 그래프에서 사용
                                "prediction_status": 'completed',
                                "current_prediction_symbol": current_company,
                                "model_type": model_type  # 사용된 모델 타입도 저장
                            })

                            st.success("주가 예측이 완료되었습니다!")

                    except Exception as e:
                        st.error(f"주가 예측 중 오류가 발생했습니다: {str(e)}")
                        logger.error(f"주가 예측 오류: {str(e)}")
                        logger.error(traceback.format_exc())

                        # 이전에 저장된 예측 결과가 있으면 현재 회사 정보와 비교하여 삭제
                        if ('prediction_result' in st.session_state.comprehensive_data and
                                'current_prediction_symbol' in st.session_state and
                                st.session_state.current_prediction_symbol != current_company):

                            # 이전 기업의 예측 결과를 삭제
                            logger.info(f"이전 기업({st.session_state.current_prediction_symbol})의 예측 결과 제거")
                            st.session_state.comprehensive_data['prediction_result'] = None

                            if 'model_evaluation' in st.session_state.comprehensive_data:
                                st.session_state.comprehensive_data['model_evaluation'] = None

                        # 현재 기업에 대한 오류 정보를 담은 기본 예측 결과 생성
                        error_message = str(e)
                        default_prediction = {
                            'error': True,
                            'error_message': error_message,
                            'company_symbol': current_company,
                            'dates': [],
                            'predicted': [],
                            'last_price': history_df['Close'].iloc[
                                -1] if history_df is not None and not history_df.empty else 0,
                            'confidence_high': [],
                            'confidence_low': [],
                            'historical_volatility': 0,
                            'trend_strength': 0
                        }

                        # 세션 상태에 예측 실패 정보 저장
                        st.session_state.comprehensive_data['prediction_result'] = default_prediction
                        st.session_state.comprehensive_data['prediction_status'] = 'failed'
                        st.session_state.current_prediction_symbol = current_company

                    finally:
                        progress_bar.empty()

                # 2. LLM 분석 생성
                try:
                    comprehensive_data = self.get_comprehensive_data(
                        company_info, stock_info, market_data, analysis_results, history_df
                    )

                    # 예측 및 평가 결과 추가
                    if 'prediction_result' in st.session_state.comprehensive_data:
                        comprehensive_data['prediction_result'] = st.session_state.comprehensive_data[
                            'prediction_result']

                        # 새로운 모델에 대한 정보 추가
                        model_type = st.session_state.comprehensive_data.get('model_type', 'LSTM')
                        comprehensive_data['model_type'] = model_type

                        # 특화 모델 설정 정보 추가
                        if model_type.lower() == "hybrid":
                            comprehensive_data['hybrid_settings'] = {
                                'use_prophet': st.session_state.model_settings.get('use_prophet', True),
                                'auto_weights': st.session_state.model_settings.get('auto_weights', True)
                            }
                        elif model_type.upper() == "TFT":
                            comprehensive_data['tft_settings'] = {
                                'num_heads': st.session_state.model_settings.get('tft_num_heads', 4),
                                'encoder_layers': st.session_state.model_settings.get('tft_encoder_layers', 2),
                                'multiresolution': st.session_state.model_settings.get('tft_multiresolution', True)
                            }
                        elif model_type.upper() == "TCN":
                            comprehensive_data['tcn_settings'] = {
                                'kernel_size': st.session_state.model_settings.get('tcn_kernel_size', 3),
                                'filters': st.session_state.model_settings.get('tcn_filters', 64),
                                'layers': st.session_state.model_settings.get('tcn_layers', 4)
                            }
                        elif model_type.upper() == "N-BEATS":
                            comprehensive_data['nbeats_settings'] = {
                                'blocks': st.session_state.model_settings.get('nbeats_blocks', 3),
                                'units': st.session_state.model_settings.get('nbeats_units', 128),
                                'seasonal': st.session_state.model_settings.get('nbeats_seasonal', True)
                            }

                    if 'model_evaluation' in st.session_state.comprehensive_data:
                        comprehensive_data['model_evaluation'] = st.session_state.comprehensive_data['model_evaluation']
                    if 'history_df' in st.session_state.comprehensive_data:
                        comprehensive_data['history_df'] = st.session_state.comprehensive_data['history_df']

                    llm_data = self.prepare_llm_data(comprehensive_data, "comprehensive")

                    ai_analysis = self.generate_ai_analysis(
                        llm_data,
                        st.session_state.report_options['language'],
                        None,
                        st.session_state.report_options['llm_model'],
                        st.session_state.report_options['temperature'],
                        st.session_state.report_options['max_tokens'],
                        analysis_depth=st.session_state.report_options['analysis_depth']
                    )
                    comprehensive_data["ai_analysis"] = ai_analysis

                except Exception as e:
                    st.error(f"AI 분석 생성 중 오류가 발생했습니다: {str(e)}")
                    logger.error(f"AI 분석 생성 오류: {str(e)}")
                    logger.error(traceback.format_exc())

                # 3. 통합 리포트 표시
                self._display_unified_report(
                    comprehensive_data,
                    "AI 기반 고급 분석 리포트",
                    "한국어",
                    company_info,
                    True
                )

        except Exception as e:
            st.error(f"리포트 생성 중 오류가 발생했습니다: {str(e)}")
            logger.error(f"리포트 생성 오류: {str(e)}")
            logger.error(traceback.format_exc())
            
    def generate_dynamic_prompt(self, llm_data, language="한국어", analysis_focus=None, analysis_depth="심화"):
        """데이터에 기반한 동적 프롬프트 생성"""

        # 언어에 따라 다른 프롬프트 템플릿 사용
        if language == "영어":
            prompt_template = f"""
            You are a securities research analyst tasked with creating a comprehensive analysis report for {llm_data['company']['name']}({llm_data['company']['symbol']}).
            Please provide useful insights for investors based on various analysis results.
            """

            # 분석 깊이에 따른 영어 지시사항
            if analysis_depth == "기본":
                prompt_template += """
                # Analysis Depth
                Provide concise and essential information only. Focus on the most important points that investors can quickly understand.
                """
            elif analysis_depth == "전문가":
                prompt_template += """
                # Analysis Depth
                Provide expert-level in-depth analysis and advanced investment strategies. Include professional content such as market microstructure, 
                advanced aspects of technical analysis, and analysis linked to portfolio theory.
                """
            else:  # "심화" 기본값
                prompt_template += """
                # Analysis Depth
                Provide balanced analysis and practical investment insights. Balance fundamental and technical analysis, 
                and include in-depth consideration of possible investment scenarios.
                """

            prompt_template += """
            # Key Analysis Data
            """
        else:  # 기본값: 한국어
            prompt_template = f"""
            당신은 증권사의 리서치 애널리스트로, {llm_data['company']['name']}({llm_data['company']['symbol']})에 대한 종합 분석 리포트를 작성해야 합니다.
            다양한 분석 결과를 바탕으로 투자자에게 유용한 종합적인 인사이트를 제공해주세요.
            """

            # 분석 깊이에 따른 한국어 지시사항
            if analysis_depth == "기본":
                prompt_template += """
                # 분석 깊이
                간결하고 핵심적인 정보만 제공해주세요. 투자자가 빠르게 이해할 수 있도록 가장 중요한 포인트만 집중적으로 다루세요.
                """
            elif analysis_depth == "전문가":
                prompt_template += """
                # 분석 깊이
                전문가 수준의 깊이 있는 분석과 고급 투자 전략을 제시해주세요. 시장 미시구조, 기술적 분석의 고급 측면, 
                포트폴리오 이론과 연계한 분석 등 전문적인 내용을 포함해주세요.
                """
            else:  # "심화" 기본값
                prompt_template += """
                # 분석 깊이
                균형 잡힌 분석과 실용적인 투자 인사이트를 제공해주세요. 기본적 분석과 기술적 분석의 균형을 맞추고,
                가능한 투자 시나리오에 대한 심층적인 고려를 포함하세요.
                """

            prompt_template += """
            # 주요 분석 데이터
            """

        # 분석 데이터 섹션 구성 - 언어에 맞게 섹션 제목 조정
        sections = []

        # 기술적 분석 섹션
        if 'technical_analysis' in llm_data:
            if language == "영어":
                tech_section = "## Technical Analysis\n"
            else:
                tech_section = "## 기술적 분석\n"

            for key, value in llm_data['technical_analysis'].items():
                tech_section += f"- {key}: {value}\n"
            sections.append(tech_section)

        # 투자자 동향 섹션
        if 'investor_trends' in llm_data:
            if language == "영어":
                investor_section = "## Investor Trends\n"
            else:
                investor_section = "## 투자자 동향\n"

            for key, value in llm_data['investor_trends'].items():
                # investor_ratio는 리스트이므로 특별히 처리
                if key == 'investor_ratio' and isinstance(value, list) and len(value) >= 4:
                    if language == "영어":
                        investor_section += f"- Institutional investors: {value[0]}%\n"
                        investor_section += f"- Foreign investors: {value[1]}%\n"
                        investor_section += f"- Individual investors: {value[2]}%\n"
                        investor_section += f"- Other corporations: {value[3]}%\n"
                    else:
                        investor_section += f"- 기관투자자 비중: {value[0]}%\n"
                        investor_section += f"- 외국인 비중: {value[1]}%\n"
                        investor_section += f"- 개인 비중: {value[2]}%\n"
                        investor_section += f"- 기타법인 비중: {value[3]}%\n"
                else:
                    investor_section += f"- {key}: {value}\n"
            sections.append(investor_section)

        # 위험 지표 섹션
        if 'financial_analysis' in llm_data and 'risk_metrics' in llm_data['financial_analysis']:
            if language == "영어":
                risk_section = "## Risk Metrics\n"
            else:
                risk_section = "## 위험 지표\n"

            for key, value in llm_data['financial_analysis']['risk_metrics'].items():
                risk_section += f"- {key}: {value}\n"
            sections.append(risk_section)

        # 매매 신호 섹션
        if 'trading_signals' in llm_data:
            if language == "영어":
                signal_section = "## Optimal Trading Points\n"
            else:
                signal_section = "## 최적 매매 시점\n"

            for key, value in llm_data['trading_signals'].items():
                if not isinstance(value, list) and not isinstance(value, dict):
                    signal_section += f"- {key}: {value}\n"
            sections.append(signal_section)

        # 주가 예측 섹션
        if 'prediction_result' in llm_data:
            if language == "영어":
                prediction_section = "## Stock Price Prediction\n"
            else:
                prediction_section = "## 주가 예측\n"

            pred_data = llm_data['prediction_result']

            # 기본 예측 정보
            prediction_section += f"- last_price: {pred_data.get('last_price', 0)}\n"
            prediction_section += f"- final_prediction: {pred_data.get('final_prediction', 0)}\n"
            prediction_section += f"- overall_change_percent: {pred_data.get('overall_change_percent', 0)}%\n"
            prediction_section += f"- prediction_days: {pred_data.get('prediction_days', 0)}\n"

            # 기간별 예측 정보
            for period in ['short_term', 'mid_term', 'long_term']:
                if period in pred_data:
                    prediction_section += f"- {period}:\n"
                    for key, value in pred_data[period].items():
                        prediction_section += f"  - {key}: {value}\n"

            sections.append(prediction_section)

        # 분석 초점이 있는 경우 관련 영역 강조
        if analysis_focus:
            if language == "영어":
                prompt_template += f"\n# Analysis Focus\nPlease focus especially on {analysis_focus} in your analysis.\n"
            else:
                prompt_template += f"\n# 분석 초점\n특히 {analysis_focus}에 중점을 두고 분석해주세요.\n"

        # 모든 섹션을 프롬프트에 추가
        prompt_template += "\n".join(sections)

        # 요청사항 추가
        if language == "영어":
            prompt_template += """

            # Requirements
            1. Please provide useful insights to investors by comprehensively analyzing the above data.
            2. Include a SWOT analysis (Strengths, Weaknesses, Opportunities, Threats).
            3. Present short-term (1-3 months) and medium-term (6-12 months) outlooks.
            4. Suggest 2-3 action plans that will be helpful to investors.
            5. Present an overall investment opinion (Buy/Sell/Hold) and explain the reasons.

            # Format
            - Please write in markdown format.
            - Keep the response length around 1000 words.
            - Structure with titles and subtitles.
            - Use professional terminology but explain at a level that general investors can understand.
            """
        else:
            prompt_template += """

            # 요청사항
            1. 위 데이터를 종합적으로 분석하여 투자자에게 유용한 인사이트를 제공해주세요.
            2. SWOT 분석을 포함해주세요. (강점, 약점, 기회, 위협)
            3. 단기(1-3개월), 중기(6-12개월) 전망을 제시해주세요.
            4. 투자자에게 도움이 될 만한 액션 플랜을 2-3가지 제안해주세요.
            5. 전체적인 투자 의견(매수/매도/관망)을 제시하고 그 이유를 설명해주세요.

            # 형식
            - 마크다운 형식으로 작성해주세요.
            - 응답 길이는 1000단어 내외로 해주세요.
            - 제목과 소제목을 포함하여 구조화된 형태로 작성해주세요.
            - 전문 용어를 사용하되 일반 투자자도 이해할 수 있는 수준으로 설명해주세요.
            """

        return prompt_template

    def _run_stock_prediction(self, symbol, stock_data, progress_bar=None):
        """주가 예측 실행 - 개선된 오류 처리"""
        try:
            # 데이터 충분성 확인
            MIN_REQUIRED_DATA = 60  # 최소 60일치 데이터 필요
            if len(stock_data) < MIN_REQUIRED_DATA:
                error_msg = f"데이터가 부족합니다. 최소 {MIN_REQUIRED_DATA}일 이상의 데이터가 필요합니다. (현재: {len(stock_data)}일)"
                if progress_bar:
                    progress_bar.error(error_msg)
                logger.error(error_msg)

                # 기본 예측 결과 생성 (오류 메시지 포함)
                default_prediction = {
                    'error': True,
                    'error_message': error_msg,
                    'dates': [],
                    'predicted': [],
                    'last_price': stock_data['Close'].iloc[-1] if not stock_data.empty else 0,
                    'confidence_high': [],
                    'confidence_low': [],
                    'historical_volatility': 0,
                    'trend_strength': 0
                }
                return default_prediction

            # 모델 설정 가져오기
            model_type = st.session_state.model_settings.get('model_type', 'LSTM').lower()
            if model_type == "앙상블":
                model_type = "ensemble"

            # 예측 기간 설정 가져오기
            forecast_days = st.session_state.model_settings.get('prediction_days', 30)

            # 로그에 설정값 기록
            logger.info(f"주가 예측 실행: 심볼={symbol}, 모델={model_type}, 예측기간={forecast_days}일")

            # 진행 상황 업데이트
            if progress_bar:
                progress_bar.progress(10, text="데이터 전처리 중...")

            try:
                # 먼저 모델 학습
                train_success = self.price_predictor.train_model(
                    stock_data,
                    model_type=model_type,
                    forecast_days=forecast_days
                )

                if not train_success:
                    raise ValueError("모델 학습에 실패했습니다.")

                if progress_bar:
                    progress_bar.progress(50, text="모델 학습 완료, 예측 중...")

                # 예측 실행
                prediction_result = self.price_predictor.predict_future(
                    stock_data,
                    model_type=model_type,
                    days=forecast_days
                )

                # 진행 상황 업데이트
                if progress_bar:
                    progress_bar.progress(100, text="예측 완료!")

                return prediction_result

            except Exception as inner_e:
                error_msg = str(inner_e)
                logger.error(f"예측 함수 실행 중 오류: {error_msg}")
                import traceback
                logger.error(traceback.format_exc())

                # 오류 메시지 분류
                if 'tuple index out of range' in error_msg:
                    friendly_error = "데이터 부족 또는 형식이 맞지 않아 학습에 실패했습니다. 더 많은 데이터가 필요합니다."
                elif '부족' in error_msg or 'insufficient' in error_msg.lower():
                    friendly_error = f"데이터가 부족합니다. 더 긴 기간의 주가 데이터가 필요합니다."
                else:
                    friendly_error = f"예측 중 오류가 발생했습니다: {error_msg}"

                if progress_bar:
                    progress_bar.error(friendly_error)

                # 기본 예측 결과 생성 (오류 메시지 포함)
                default_prediction = {
                    'error': True,
                    'error_message': friendly_error,
                    'dates': [],
                    'predicted': [],
                    'last_price': stock_data['Close'].iloc[-1] if not stock_data.empty else 0,
                    'confidence_high': [],
                    'confidence_low': [],
                    'historical_volatility': 0,
                    'trend_strength': 0
                }
                return default_prediction

        except Exception as e:
            logger.error(f"주가 예측 중 오류 발생: {str(e)}")
            logger.error(traceback.format_exc())
            if progress_bar:
                progress_bar.error(f"예측 중 오류: {str(e)}")
                progress_bar.empty()

            # 기본 예측 결과 생성 (오류 메시지 포함)
            default_prediction = {
                'error': True,
                'error_message': f"예측 처리 중 오류: {str(e)}",
                'dates': [],
                'predicted': [],
                'last_price': stock_data['Close'].iloc[-1] if isinstance(stock_data,
                                                                         pd.DataFrame) and not stock_data.empty else 0,
                'confidence_high': [],
                'confidence_low': [],
                'historical_volatility': 0,
                'trend_strength': 0
            }
            return default_prediction

    def register_analysis_result(self, tab_name, data):
        """각 탭의 분석 결과를 등록"""
        if 'comprehensive_data' not in st.session_state:
            self._initialize_data_store()

        # 중첩 딕셔너리 구조인 경우 처리
        if tab_name == 'financial_analysis' and isinstance(data, dict):
            for key, value in data.items():
                if key in st.session_state.comprehensive_data['financial_analysis']:
                    st.session_state.comprehensive_data['financial_analysis'][key] = value
        else:
            st.session_state.comprehensive_data[tab_name] = data

        # 마지막 업데이트 시간 기록
        st.session_state.comprehensive_data['last_update'][tab_name] = datetime.now().isoformat()

        # 캐시 무효화
        st.session_state.comprehensive_data['analysis_cache'] = {}

        logger.info(f"탭 '{tab_name}' 분석 결과가 종합 데이터에 등록되었습니다.")

    def get_comprehensive_data(self, company_info, stock_info, market_data, analysis_results, history_df=None):
        """완전한 종합 데이터 가져오기"""
        # 데이터 저장소 초기화 확인
        if 'comprehensive_data' not in st.session_state:
            self._initialize_data_store()

        # 기본 정보 업데이트
        comprehensive_data = st.session_state.comprehensive_data.copy()
        comprehensive_data['company_info'] = company_info
        comprehensive_data['stock_info'] = stock_info
        comprehensive_data['market_data'] = market_data
        comprehensive_data['analysis_results'] = analysis_results
        comprehensive_data['history_df'] = history_df

        # 누락된 데이터 확인 및 필요시 수집
        self._ensure_essential_data(comprehensive_data)

        return comprehensive_data

    def _ensure_essential_data(self, data):
        """필수 데이터가 있는지 확인하고 없으면 기본값 설정 또는 다른 소스에서 가져오기"""
        # 기본 주식 상세 정보 확인
        if not data.get('stock_detail'):
            data['stock_detail'] = self._extract_stock_detail(data)

        # 기본 기술적 분석 데이터 확인
        if not data.get('technical_analysis'):
            data['technical_analysis'] = self._extract_technical_analysis(data)

        # 기본 투자자 동향 데이터 확인
        if not data.get('investor_trends'):
            data['investor_trends'] = self._extract_investor_trends(data)

        # 위험 지표 확인 - 세션 상태에서도 확인
        if not data.get('financial_analysis', {}).get('risk_metrics') and 'risk_metrics' in st.session_state:
            if 'financial_analysis' not in data:
                data['financial_analysis'] = {}
            data['financial_analysis']['risk_metrics'] = st.session_state.risk_metrics

        # 성장률 데이터 확인 - 세션 상태에서도 확인
        if not data.get('financial_analysis', {}).get('growth_data') and 'growth_data' in st.session_state:
            if 'financial_analysis' not in data:
                data['financial_analysis'] = {}
            data['financial_analysis']['growth_data'] = st.session_state.growth_data

        # 매매 신호 데이터 확인 - 세션 상태에서도 확인
        if not data.get('trading_signals') and 'trading_signals' in st.session_state:
            data['trading_signals'] = st.session_state.trading_signals

        # 주가 예측 데이터 확인 - 세션 상태에서도 확인
        if not data.get('prediction_result') and 'prediction_result' in st.session_state and st.session_state.get(
                'prediction_status') == 'completed':
            data['prediction_result'] = st.session_state.prediction_result

        # 결측 데이터 기본값 설정
        if not data.get('financial_analysis', {}).get('risk_metrics'):
            if 'financial_analysis' not in data:
                data['financial_analysis'] = {}
            data['financial_analysis']['risk_metrics'] = {
                "beta": 1.0,
                "annual_volatility": 15.0,
                "max_drawdown": 20.0,
                "sharpe_ratio": 0.5
            }

        if not data.get('financial_analysis', {}).get('growth_data'):
            if 'financial_analysis' not in data:
                data['financial_analysis'] = {}
            data['financial_analysis']['growth_data'] = {
                "annual": {
                    "revenue_growth": [],
                    "operating_income_growth": [],
                    "net_income_growth": []
                }
            }

        if not data.get('trading_signals'):
            data['trading_signals'] = {
                "recommendation": "관망",
                "current_buy_strength": 0,
                "current_sell_strength": 0,
                "latest_buy": [{"날짜": "N/A", "근거": "N/A"}],
                "latest_sell": [{"날짜": "N/A", "근거": "N/A"}]
            }

    def _extract_stock_detail(self, data):
        """마켓 데이터에서 주식 상세 정보 추출"""
        market_data = data.get('market_data', {})
        stock_info = data.get('stock_info', {})

        return {
            "current_price": market_data.get('close', [])[-1] if market_data.get('close') and len(
                market_data.get('close', [])) > 0 else 0,
            "price_change": ((market_data.get('close', [])[-1] - market_data.get('close', [])[-2]) /
                             market_data.get('close', [])[-2]) * 100
            if market_data.get('close') and len(market_data.get('close', [])) >= 2 else 0,
            "volume": market_data.get('volume', [])[-1] if market_data.get('volume') and len(
                market_data.get('volume', [])) > 0 else 0,
            "market_cap": getattr(stock_info, 'market_cap', 0) or 0
        }

    def _extract_technical_analysis(self, data):
        """분석 결과에서 기술적 분석 정보 추출"""
        analysis_results = data.get('analysis_results', {})

        return {
            "trend": analysis_results.get('trend', 'N/A'),
            "ma5": analysis_results.get('ma5', 0),
            "ma20": analysis_results.get('ma20', 0),
            "rsi": analysis_results.get('rsi', 0),
            "rsi_status": analysis_results.get('rsi_status', 'N/A'),
            "volume_trend": analysis_results.get('volume_trend', 'N/A')
        }

    def _extract_investor_trends(self, data):
        """분석 결과에서 투자자 동향 정보 추출"""
        analysis_results = data.get('analysis_results', {})
        market_data = data.get('market_data', {})

        return {
            "main_buyer": analysis_results.get('main_buyer', 'N/A'),
            "main_seller": analysis_results.get('main_seller', 'N/A'),
            "investor_ratio": market_data.get('investor_ratio', [0, 0, 0, 0])
        }

    def prepare_llm_data(self, report_data, detail_level="comprehensive"):
        """LLM 분석을 위한 데이터 준비"""
        # comprehensive 고정으로 사용하므로 다른 조건 삭제
        # 가능한 모든 데이터 포함 (단, 너무 크거나 복잡한 데이터는 제외)
        llm_data = {
            "company": report_data["company_info"],
            "stock_detail": report_data["stock_detail"],
            "technical_analysis": report_data["technical_analysis"],
            "investor_trends": report_data["investor_trends"],
            "financial_analysis": report_data["financial_analysis"],
            "trading_signals": report_data["trading_signals"]
        }

        # 예측 결과가 있으면 포함
        if report_data.get("prediction_result"):
            llm_data["prediction_result"] = self._format_prediction_data(report_data["prediction_result"])

        # 히스토리 데이터는 너무 크므로 제외
        if "history_df" in llm_data:
            del llm_data["history_df"]

        return llm_data

    def _format_prediction_data(self, prediction_result):
        """예측 데이터 포맷팅"""
        if not prediction_result:
            return None

        # NumPy 배열 처리
        predicted = prediction_result.get('predicted', [])
        if isinstance(predicted, np.ndarray):
            predicted = predicted.tolist()
        elif isinstance(predicted, pd.Series):
            predicted = predicted.values.tolist()

        last_price = prediction_result.get('last_price', 0)

        # 기간별 분석을 위한 데이터 준비
        short_term = predicted[:7] if len(predicted) >= 7 else predicted
        mid_term = predicted[7:21] if len(predicted) >= 21 else predicted[7:] if len(predicted) >= 7 else []
        long_term = predicted[21:] if len(predicted) >= 21 else []

        # 기간별 변화율 계산
        short_term_change = ((short_term[-1] - last_price) / last_price * 100) if short_term else 0
        mid_term_change = ((mid_term[-1] - last_price) / last_price * 100) if mid_term else 0
        long_term_change = ((long_term[-1] - last_price) / last_price * 100) if long_term else 0

        return {
            "last_price": last_price,
            "predicted": predicted,
            "prediction_days": len(predicted),
            "final_prediction": predicted[-1] if predicted else 0,
            "overall_change_percent": ((predicted[-1] - last_price) / last_price * 100) if predicted else 0,
            "short_term": {
                "period": "7일",
                "change_percent": short_term_change,
                "trend": "상승" if short_term_change > 0 else "하락"
            },
            "mid_term": {
                "period": "7-21일",
                "change_percent": mid_term_change,
                "trend": "상승" if mid_term_change > 0 else "하락" if mid_term_change < 0 else "데이터 없음"
            },
            "long_term": {
                "period": "21일 이상",
                "change_percent": long_term_change,
                "trend": "상승" if long_term_change > 0 else "하락" if long_term_change < 0 else "데이터 없음"
            }
        }

    def generate_ai_analysis(self, llm_data, language="한국어", analysis_focus=None,
                             model="gpt-4o", temperature=0.7, max_tokens=2000,
                             analysis_depth="심화"):
        """LLM 기반 AI 분석 생성"""
        try:
            # 캐시 키 생성 (모델, 온도, 토큰 수 포함)
            cache_key = f"{language}_{analysis_focus}_{model}_{temperature}_{max_tokens}_{analysis_depth}_{hash(json.dumps(llm_data, sort_keys=True, default=str))}"

            # 캐시된 분석 결과가 있으면 반환
            cached_analysis = st.session_state.comprehensive_data.get('analysis_cache', {}).get(cache_key)
            if cached_analysis:
                logger.info("캐시된 AI 분석 결과 사용")
                return cached_analysis

            # 동적 프롬프트 생성 - 언어와 분석 깊이 반영
            prompt = self.generate_dynamic_prompt(llm_data, language, analysis_focus, analysis_depth)

            # 언어에 따른 시스템 메시지 설정
            if language == "영어":
                system_message = "You are a professional financial analyst providing investment insights. Please respond in English."
            else:
                system_message = "당신은 투자 인사이트를 제공하는 전문 금융 애널리스트입니다. 한국어로 응답해주세요."

            # API 요청 데이터 구성
            api_url = "https://api.openai.com/v1/chat/completions"
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.openai_api_key}"
            }

            # API 요청 보내기
            data = {
                "model": model,
                "messages": [
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": prompt}
                ],
                "temperature": temperature,
                "max_tokens": max_tokens
            }

            # 디버깅을 위한 언어 설정 로깅
            logger.info(f"AI 분석 요청 - 언어: {language}, 분석 깊이: {analysis_depth}")

            response = requests.post(api_url, headers=headers, json=data)
            response_data = response.json()

            # 응답 처리
            if "choices" in response_data and len(response_data["choices"]) > 0:
                analysis = response_data["choices"][0]["message"]["content"]

                # 결과 캐싱
                if 'analysis_cache' not in st.session_state.comprehensive_data:
                    st.session_state.comprehensive_data['analysis_cache'] = {}
                st.session_state.comprehensive_data['analysis_cache'][cache_key] = analysis

                return analysis
            else:
                error_message = response_data.get("error", {}).get("message", "Unknown error")
                raise Exception(f"OpenAI API 오류: {error_message}")

        except Exception as e:
            logger.error(f"AI 분석 생성 중 오류 발생: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def _display_unified_report(self, report_data, report_type, language, company_info, include_predictions=True):
        """통합된 리포트 내용 표시 (탭 대신 단일 화면)"""
        # 회사 정보
        company_name = company_info["name"]
        company_symbol = company_info["symbol"]

        st.subheader(f"{company_name} ({company_symbol}) 종합 분석 리포트")
        st.caption(f"생성일: {datetime.now().strftime('%Y-%m-%d')}")

        # 주가 예측 섹션 - 설정에 따라 가장 먼저 표시
        if include_predictions:
            # 현재 표시할 예측 결과 확인
            prediction_result = report_data.get('prediction_result')
            
            # 1. 예측 결과가 없는 경우
            if prediction_result is None:
                st.warning("주가 예측 결과가 없습니다.")
            # 2. 예측 오류가 있는 경우
            elif 'error' in prediction_result and prediction_result['error']:
                st.error(f"주가 예측 실패: {prediction_result.get('error_message', '알 수 없는 오류')}")
                
                # 데이터 부족 문제인 경우 유용한 정보 제공
                if '부족' in prediction_result.get('error_message', ''):
                    st.info("""
                    ### 데이터 부족으로 인한 주가 예측 불가
                    
                    최근 상장된 기업이나 거래 데이터가 충분하지 않은 경우 딥러닝 기반 주가 예측이 어렵습니다.
                    다음과 같은 대안을 고려해보세요:
                    
                    1. **기술적 분석** - 제한된 데이터로도 기본적인 기술적 분석은 가능합니다.
                    2. **기본적 분석** - 재무제표와 뉴스 기반 분석을 활용하세요.
                    3. **유사 기업 분석** - 동종 업계의 유사한 기업 데이터를 참고하세요.
                    """)
            # 3. 정상적인 예측 결과가 있는 경우
            else:
                # 예측 회사와 현재 회사가 다른 경우 예측 결과 표시하지 않음
                prediction_company = prediction_result.get('company_symbol', company_symbol)
                if prediction_company != company_symbol:
                    logger.warning(f"예측 결과의 회사({prediction_company})와 현재 회사({company_symbol})가 일치하지 않습니다.")
                    st.warning("이 기업에 대한 주가 예측 결과가 없습니다.")
                else:
                    self._display_prediction_section(report_data, include_predictions, company_info)

        # AI 종합 분석
        st.markdown("### 🧠 종합 AI 분석")

        if "ai_analysis" in report_data and report_data["ai_analysis"]:
            st.markdown(report_data["ai_analysis"])
        else:
            st.warning("AI 분석 결과가 아직 생성되지 않았습니다. 리포트 생성 버튼을 클릭하여 분석을 실행해주세요.")
            
    def _display_prediction_section(self, report_data, include_predictions, company_info=None):
        """예측 섹션 표시 - 회사 일치 여부 확인 추가"""
        if not include_predictions:
            return

        st.write("## 📈 주가 예측 분석")

        try:
            # 예측 결과가 없는 경우 처리
            if 'prediction_result' not in report_data:
                st.warning("예측 결과가 없습니다.")
                return
                
            prediction_result = report_data['prediction_result']
            
            # 오류가 있는 경우 오류 메시지 표시 후 종료
            if 'error' in prediction_result and prediction_result['error']:
                st.error(f"주가 예측 실패: {prediction_result.get('error_message', '알 수 없는 오류')}")
                return
                
            # 예측 회사와 현재 회사가 다른 경우
            current_symbol = company_info.get("symbol") if company_info else None
            prediction_company = prediction_result.get('company_symbol', current_symbol)
            
            if prediction_company != current_symbol:
                st.warning(f"이 기업({current_symbol})에 대한 주가 예측 결과가 없습니다.")
                logger.warning(f"예측 결과의 회사({prediction_company})와 현재 회사({current_symbol})가 일치하지 않습니다.")
                return

            # 예측 결과 표시
            self._display_prediction_results(report_data, prediction_result)

            # 모델 평가 결과 표시
            if 'model_evaluation' in report_data:
                with st.expander("🎯 모델 성능 평가", expanded=False):
                    symbol = company_info.get("symbol", "unknown") if company_info else "unknown"
                    self._display_model_evaluation(
                        report_data['model_evaluation'],
                        symbol
                    )

        except Exception as e:
            st.error(f"예측 결과 표시 중 오류가 발생했습니다: {str(e)}")
            logger.error(f"예측 결과 표시 오류: {str(e)}")
            logger.error(traceback.format_exc())

    def _display_prediction_results(self, stock_data, prediction_data):
        """예측 결과 시각화 - 새로운 모델 지원 추가"""
        # 필요한 모듈 임포트
        import pandas as pd
        import numpy as np
        from datetime import datetime, timedelta

        try:
            if prediction_data is None:
                st.warning("예측 결과가 없습니다.")
                return

            st.subheader("주가 예측 차트")

            # 모델 유형 가져오기 및 표시
            model_type = st.session_state.comprehensive_data.get('model_type', 'LSTM')
            st.write(f"#### 사용 모델: {model_type}")

            # 모델별 특성 설명 추가
            model_descriptions = {
                "LSTM": "순환 신경망 기반으로 시간적 의존성을 잘 포착하는 모델입니다.",
                "Transformer": "어텐션 메커니즘을 활용하여 복잡한 패턴을 인식하는 모델입니다.",
                "ensemble": "LSTM과 Transformer 모델을 결합한 앙상블 모델입니다.",
                "TCN": "확장된 컨볼루션을 통해 다양한 시간 스케일의 패턴을 효율적으로 처리하는 모델입니다.",
                "TFT": "시간 특성을 여러 해상도로 처리하고 변수 중요도를 자동으로 학습하는 최신 모델입니다.",
                "N-BEATS": "계층적 구조와 역투영 메커니즘을 통해 복잡한 시계열 패턴을 포착하는 모델입니다.",
                "hybrid": "여러 예측 모델을 지능적으로 결합한 하이브리드 앙상블 모델입니다."
            }

            if model_type.lower() in model_descriptions:
                st.info(model_descriptions[model_type.lower()])

            # 하이브리드 앙상블인 경우 사용된 모델 가중치 표시
            if model_type.lower() == "hybrid" and 'model_weights' in prediction_data:
                st.write("##### 모델 가중치")
                weights = prediction_data['model_weights']

                # 가중치 정보를 표 형태로 표시
                weight_data = []
                for model, weight in weights.items():
                    if weight > 0:  # 가중치가 0보다 큰 모델만 표시
                        weight_data.append({"모델": model, "가중치": f"{weight:.2f}"})

                if weight_data:
                    weight_df = pd.DataFrame(weight_data)
                    st.dataframe(weight_df, use_container_width=True)

            # stock_data 타입 확인 및 필요시 조정
            history_df = None
            if stock_data is not None:
                if isinstance(stock_data, pd.DataFrame):
                    history_df = stock_data
                    logger.info("stock_data를 DataFrame으로 사용")
                elif isinstance(stock_data, dict) and 'history_df' in stock_data:
                    # stock_data가 dict이고 history_df 키가 있는 경우
                    history_df = stock_data['history_df']
                    logger.info("stock_data['history_df']를 DataFrame으로 사용")
                elif isinstance(stock_data, dict):
                    # prediction_data에 history_df가 있는지 확인
                    if 'history_df' in prediction_data:
                        history_df = prediction_data['history_df']
                        logger.info("prediction_data['history_df']를 DataFrame으로 사용")
                    else:
                        logger.warning("유효한 stock_data DataFrame을 찾을 수 없습니다.")

            # prediction_data 구조 확인 및 실제 예측 결과 추출
            prediction_result = None

            # case 1: prediction_data가 이미 예측 결과인 경우
            if 'predicted' in prediction_data or any(
                    k in prediction_data for k in ['predicted_prices', 'predictions', 'forecast']):
                prediction_result = prediction_data
                logger.info("직접 예측 결과 객체 사용")

            # case 2: 'prediction_result' 키가 있는 중첩 구조인 경우
            elif 'prediction_result' in prediction_data and prediction_data['prediction_result'] is not None:
                prediction_result = prediction_data['prediction_result']
                logger.info("prediction_result 키에서 예측 결과 추출")

            # 유효한 예측 결과가 없는 경우 예외 처리
            if prediction_result is None:
                st.warning("유효한 예측 결과를 찾을 수 없습니다.")
                logger.warning(f"prediction_data 키: {list(prediction_data.keys())}")

                # prediction_data가 dict의 dict인 경우, 모든 내부 dict를 확인
                for key, value in prediction_data.items():
                    if isinstance(value, dict) and (
                            'predicted' in value or 'predictions' in value or 'forecast' in value):
                        prediction_result = value
                        logger.info(f"'{key}' 키에서 예측 결과 추출")
                        break

            # 여전히 유효한 예측 결과가 없는 경우
            if prediction_result is None:
                st.error("예측 결과 데이터를 찾을 수 없습니다.")
                logger.error(f"예측 데이터 구조: {type(prediction_data)}")
                if isinstance(prediction_data, dict):
                    logger.error(f"사용 가능한 모든 키: {list(prediction_data.keys())}")
                    # 더 자세한 구조 확인을 위한 로깅
                    for key, value in prediction_data.items():
                        logger.error(f"Key '{key}' type: {type(value)}")
                        if isinstance(value, dict):
                            logger.error(f"Key '{key}' sub-keys: {list(value.keys())}")
                return

            # 예측 결과에서 필요한 데이터 추출
            # 'predicted' 키 또는 대체 키 확인
            predicted_prices = None
            for key in ['predicted', 'predicted_prices', 'predictions', 'forecast', 'pred']:
                if key in prediction_result:
                    predicted_prices = prediction_result[key]
                    logger.info(f"예측 가격에 '{key}' 키 사용")
                    break

            if predicted_prices is None:
                st.error("예측 가격 데이터를 찾을 수 없습니다.")
                logger.error(f"prediction_result 키: {list(prediction_result.keys())}")
                return

            # 신뢰 구간 (없을 경우 None 사용)
            confidence_high = prediction_result.get('confidence_high', None)
            confidence_low = prediction_result.get('confidence_low', None)

            # Prophet 예측 결과가 있는 경우 추가 (하이브리드 모델에서)
            if model_type.lower() == "hybrid" and 'prophet_predictions' in prediction_result:
                prophet_data = prediction_result['prophet_predictions']

                # Prophet 예측 추가 표시 여부 확인
                show_prophet = st.checkbox("Prophet 예측 결과 표시", value=True)

                if show_prophet and prophet_data is not None:
                    st.write("##### Prophet 예측 결과")
                    prophet_values = prophet_data.get('values', [])
                    prophet_lower = prophet_data.get('lower', [])
                    prophet_upper = prophet_data.get('upper', [])

                    if prophet_values and len(prophet_values) > 0:
                        # Prophet 예측 시각화 (간단한 차트)
                        try:
                            fig_prophet = go.Figure()
                            # 날짜 생성
                            dates = self.generate_trading_days(history_df.index[-1], len(prophet_values))

                            # Prophet 예측값
                            fig_prophet.add_trace(go.Scatter(
                                x=dates,
                                y=prophet_values,
                                mode='lines',
                                name='Prophet 예측',
                                line=dict(color='orange', width=2)
                            ))

                            # 신뢰 구간 (있는 경우)
                            if prophet_lower and prophet_upper and len(prophet_lower) == len(prophet_upper):
                                fig_prophet.add_trace(go.Scatter(
                                    x=dates + dates[::-1],
                                    y=prophet_upper + prophet_lower[::-1],
                                    fill='toself',
                                    fillcolor='rgba(255,165,0,0.2)',
                                    line=dict(color='rgba(255,255,255,0)'),
                                    name='Prophet 신뢰 구간'
                                ))

                            fig_prophet.update_layout(
                                title="Prophet 시계열 예측",
                                xaxis_title="날짜",
                                yaxis_title="가격",
                                height=300
                            )

                            st.plotly_chart(fig_prophet, use_container_width=True)
                        except Exception as e:
                            st.warning(f"Prophet 차트 생성 중 오류: {str(e)}")

            # 예측 날짜 (없을 경우 생성)
            if 'dates' in prediction_result:
                prediction_dates = prediction_result['dates']
            else:
                # 마지막 날짜 가져오기
                if history_df is not None and isinstance(history_df, pd.DataFrame) and not history_df.empty:
                    last_date = history_df.index[-1]
                    # 거래일(월-금) 기준 날짜 생성
                    prediction_dates = self.generate_trading_days(last_date, len(predicted_prices))
                else:
                    # 오늘부터 거래일(월-금) 기준 날짜 생성
                    prediction_dates = self.generate_trading_days(datetime.now(), len(predicted_prices))

            # 마지막 가격 (없을 경우 계산)
            if 'last_price' in prediction_result:
                last_price = prediction_result['last_price']
            else:
                if history_df is not None and isinstance(history_df, pd.DataFrame) and not history_df.empty:
                    last_price = history_df['Close'].iloc[-1]
                else:
                    # 예상 마지막 가격 (예측 시작점)
                    last_price = predicted_prices[0] if len(predicted_prices) > 0 else 0
                    logger.warning("마지막 실제 가격 정보가 없어 예측 첫 가격을 사용합니다.")

            # 가격 변화 추세 계산
            price_change = predicted_prices[-1] - last_price
            price_change_percent = (price_change / last_price) * 100 if last_price > 0 else 0

            # 예측 방향 표시
            if price_change > 0:
                arrow = "↗️"
                color = "green"
                direction = "상승"
            else:
                arrow = "↘️"
                color = "red"
                direction = "하락"

            st.markdown(
                f"### 예측 방향: <span style='color:{color};'>{arrow} {direction} ({price_change_percent:.2f}%)</span>",
                unsafe_allow_html=True)

            # 지표 표시
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "현재 가격",
                    f"{last_price:,.0f}원"
                )

            with col2:
                st.metric(
                    f"{len(predicted_prices)}일 후 예상 가격",
                    f"{predicted_prices[-1]:,.0f}원",
                    f"{price_change_percent:.2f}%"
                )

            with col3:
                if confidence_high is not None and confidence_low is not None:
                    uncertainty = ((confidence_high[-1] - confidence_low[-1]) / 2 / predicted_prices[-1] * 100)
                    st.metric(
                        "예측 불확실성",
                        f"±{uncertainty:.2f}%"
                    )

            # 최근 실제 가격과 예측 가격 차트
            fig = go.Figure()

            # 데이터 준비
            recent_dates = []
            recent_prices = []

            # 최근 실제 주가 데이터 준비
            if history_df is not None and isinstance(history_df, pd.DataFrame) and not history_df.empty:
                historical_period = min(30, len(history_df))
                recent_dates = history_df.index[-historical_period:].tolist()
                recent_prices = history_df['Close'].values[-historical_period:].tolist()

            # 예측 데이터 준비
            try:
                prediction_dates_list = list(prediction_dates) if not isinstance(prediction_dates,
                                                                                 list) else prediction_dates
                predicted_prices_list = list(predicted_prices) if not isinstance(predicted_prices,
                                                                                 list) else predicted_prices
            except Exception as e:
                logger.error(f"예측 데이터 변환 오류: {str(e)}")
                # 백업 방법으로 수동 변환 시도
                try:
                    prediction_dates_list = []
                    for date in prediction_dates:
                        prediction_dates_list.append(date)

                    predicted_prices_list = []
                    for price in predicted_prices:
                        predicted_prices_list.append(float(price))
                except Exception as e2:
                    logger.error(f"수동 데이터 변환도 실패: {str(e2)}")
                    st.error("예측 데이터 준비 중 오류가 발생했습니다.")
                    return

            # 신뢰 구간 데이터 준비
            confidence_high_list = []
            confidence_low_list = []

            if confidence_high is not None and confidence_low is not None:
                try:
                    confidence_high_list = list(confidence_high) if not isinstance(confidence_high,
                                                                                   list) else confidence_high
                    confidence_low_list = list(confidence_low) if not isinstance(confidence_low,
                                                                                 list) else confidence_low
                except Exception as e:
                    logger.error(f"신뢰 구간 변환 오류: {str(e)}")
                    # 신뢰 구간은 필수가 아니므로 무시하고 진행

            # ===== 핵심: Y축 범위 계산을 위한 코드 =====
            # 1. 모든 데이터를 숫자로 변환하고 이상치 제거
            def safe_float(x):
                try:
                    val = float(x)
                    if np.isnan(val) or np.isinf(val):
                        return None
                    return val
                except:
                    return None

            # 주요 데이터만 사용하여 Y축 범위 계산 (신뢰 구간 제외)
            core_prices = []

            # 실제 가격 추가
            core_prices.extend([safe_float(p) for p in recent_prices])

            # 마지막 실제 가격 추가
            if last_price is not None:
                core_prices.append(safe_float(last_price))

            # 예측 가격 추가
            core_prices.extend([safe_float(p) for p in predicted_prices_list])

            # None 값 필터링
            core_prices = [p for p in core_prices if p is not None]

            # 극단적 이상치 제거 (IQR 방식)
            if len(core_prices) > 4:  # 사분위수 계산에 필요한 최소 데이터 개수
                try:
                    # 사분위수 계산
                    q1 = np.percentile(core_prices, 25)
                    q3 = np.percentile(core_prices, 75)
                    iqr = q3 - q1

                    # 이상치 경계 계산 (일반적인 1.5 대신 3으로 설정하여 덜 엄격하게)
                    lower_bound = q1 - 3 * iqr
                    upper_bound = q3 + 3 * iqr

                    # 이상치 제거된 데이터
                    filtered_prices = [p for p in core_prices if lower_bound <= p <= upper_bound]

                    # 필터링 결과 로깅
                    logger.info(f"원본 데이터 개수: {len(core_prices)}, 필터링 후: {len(filtered_prices)}")
                    logger.info(f"Q1: {q1}, Q3: {q3}, IQR: {iqr}")
                    logger.info(f"하한: {lower_bound}, 상한: {upper_bound}")

                    # 필터링 결과가 너무 적으면 원본 사용
                    if len(filtered_prices) < len(core_prices) * 0.5:
                        logger.warning("필터링으로 데이터가 너무 많이 제거됨. 필터링 완화")
                        # 더 관대한 경계로 다시 필터링
                        lower_bound = q1 - 5 * iqr
                        upper_bound = q3 + 5 * iqr
                        filtered_prices = [p for p in core_prices if lower_bound <= p <= upper_bound]

                    # 최종 데이터 사용
                    if len(filtered_prices) > 0:
                        core_prices = filtered_prices
                except Exception as e:
                    logger.error(f"이상치 제거 중 오류: {str(e)}")
                    # 오류 발생 시 원본 데이터 유지

            # 2. Y축 범위 설정
            if core_prices:
                min_price = min(core_prices)
                max_price = max(core_prices)

                # 최소/최대 동일한 경우 처리
                if max_price == min_price:
                    padding = max(max_price * 0.05, 1)  # 최소 1 또는 5% 중 큰 값
                    min_price -= padding
                    max_price += padding
                else:
                    price_range = max_price - min_price

                    # 여백 추가 (좀더 넉넉하게)
                    padding = price_range * 0.12
                    min_price = min_price - padding
                    max_price = max_price + padding

                # 음수 방지
                min_price = max(0, min_price) if min_price > -0.1 * max_price else min_price

                logger.info(f"최종 Y축 범위: {min_price:.2f} ~ {max_price:.2f}")
            else:
                # 데이터가 없는 경우 자동 범위 사용
                logger.warning("유효한 가격 데이터가 없어 자동 범위 사용")
                min_price = None
                max_price = None

            # 그래프 데이터 추가
            # 실제 가격 추가
            if recent_dates and recent_prices:
                fig.add_trace(
                    go.Scatter(
                        x=recent_dates,
                        y=recent_prices,
                        name="실제 가격",
                        line=dict(color='royalblue', width=3)
                    )
                )

            # 예측 가격 추가
            if prediction_dates_list and predicted_prices_list:
                fig.add_trace(
                    go.Scatter(
                        x=prediction_dates_list,
                        y=predicted_prices_list,
                        mode='lines+markers',
                        name='예측 가격',
                        line=dict(color='red', width=2),
                        marker=dict(size=6)
                    )
                )

            # 신뢰 구간 추가 (옵션)
            if prediction_dates_list and confidence_high_list and confidence_low_list:
                try:
                    dates_x = prediction_dates_list
                    dates_x_rev = dates_x[::-1]

                    fig.add_trace(
                        go.Scatter(
                            x=dates_x + dates_x_rev,
                            y=confidence_high_list + confidence_low_list[::-1],
                            fill='toself',
                            fillcolor='rgba(0,100,80,0.2)',
                            line=dict(color='rgba(255,255,255,0)'),
                            name='90% 신뢰 구간',
                            showlegend=True
                        )
                    )
                except Exception as e:
                    logger.warning(f"신뢰 구간 표시 오류: {str(e)}")

            # 마지막 실제 가격 표시
            if history_df is not None and isinstance(history_df, pd.DataFrame) and not history_df.empty:
                last_actual_date = history_df.index[-1]
                fig.add_trace(
                    go.Scatter(
                        x=[last_actual_date],
                        y=[last_price],
                        mode='markers',
                        name='마지막 실제 가격',
                        marker=dict(color='blue', size=8, symbol='star')
                    )
                )

            # 경계 표시를 위한 주석 추가
            if history_df is not None and isinstance(history_df, pd.DataFrame) and not history_df.empty:
                current_date = history_df.index[-1]

                fig.add_annotation(
                    x=current_date,
                    y=1.05,
                    yref="paper",
                    text="현재",
                    showarrow=True,
                    arrowhead=2,
                    arrowcolor="gray",
                    arrowwidth=1.5,
                    arrowsize=1,
                    ax=0,
                    ay=-30
                )

                # 수직선 대신 시각적으로 영역 구분
                if len(prediction_dates) > 0:
                    fig.add_vrect(
                        x0=current_date,
                        x1=prediction_dates[0],
                        fillcolor="gray",
                        opacity=0.1,
                        layer="below",
                        line_width=0
                    )

            # Y축 범위 적용 및 그리드 설정 (한 번만 설정)
            if min_price is not None and max_price is not None:
                fig.update_yaxes(
                    range=[min_price, max_price],
                    autorange=False,  # 자동 범위 비활성화
                    showgrid=True,  # 그리드 표시
                    gridwidth=1,  # 그리드 두께
                    gridcolor='rgba(0,0,0,0.1)',  # 그리드 색상
                    zeroline=True,  # 0 기준선 표시
                    zerolinewidth=1.5,  # 0 기준선 두께
                    zerolinecolor='rgba(0,0,0,0.2)'  # 0 기준선 색상
                )
            else:
                # 자동 범위에도 그리드는 적용
                fig.update_yaxes(
                    autorange=True,
                    showgrid=True,
                    gridwidth=1,
                    gridcolor='rgba(0,0,0,0.1)'
                )

            # 차트 레이아웃 설정
            fig.update_layout(
                title='실제 가격과 예측 가격 비교',
                xaxis_title='날짜',
                yaxis_title='주가',
                hovermode='x unified',
                showlegend=True,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )

            # X축 날짜 포맷 설정
            fig.update_xaxes(
                tickformat="%m월 %d일 (%a)",  # 요일 표시
                tickformatstops=[
                    dict(dtickrange=[None, 86400000], value="%m월 %d일 (%a)"),
                    dict(dtickrange=[86400000, 604800000], value="%m월 %d일 (%a)"),
                    dict(dtickrange=[604800000, "M1"], value="%m월 %d일"),
                    dict(dtickrange=["M1", "M12"], value="%m월"),
                    dict(dtickrange=["M12", None], value="%Y년")
                ]
            )

            # 그래프 표시
            st.plotly_chart(fig, use_container_width=True)

            # 예측 결과 테이블
            with st.expander("상세 예측 가격", expanded=False):
                try:
                    # 날짜 포맷 변환
                    formatted_dates = []
                    formatted_weekdays = []

                    for date in prediction_dates:
                        try:
                            # datetime 객체 변환 및 포맷팅
                            if hasattr(date, 'strftime'):
                                formatted_dates.append(date.strftime('%Y-%m-%d (%a)'))
                                formatted_weekdays.append(date.strftime('%A'))
                            else:
                                # datetime이 아닌 경우 변환 시도
                                d = pd.Timestamp(date)
                                formatted_dates.append(d.strftime('%Y-%m-%d (%a)'))
                                formatted_weekdays.append(d.strftime('%A'))
                        except Exception as e:
                            # 변환 실패 시 원본 값 사용
                            logger.warning(f"날짜 형식 변환 실패: {str(e)}")
                            formatted_dates.append(str(date))
                            formatted_weekdays.append("Unknown")

                    # 데이터프레임 생성
                    pred_df = pd.DataFrame({
                        '날짜': formatted_dates,
                        '요일': formatted_weekdays,
                        '예측 가격': [f"{price:,.0f}원" for price in predicted_prices],
                        '변화율(%)': [(p - last_price) / last_price * 100 for p in predicted_prices]
                    })

                    if confidence_high is not None and confidence_low is not None:
                        pred_df['상한 신뢰구간'] = [f"{price:,.0f}원" for price in confidence_high]
                        pred_df['하한 신뢰구간'] = [f"{price:,.0f}원" for price in confidence_low]
                        pred_df['불확실성(±%)'] = [((high - low) / 2 / pred) * 100 for high, low, pred in
                                               zip(confidence_high, confidence_low, predicted_prices)]

                    # 요일에 따른 조건부 서식 설정
                    def highlight_weekday(s):
                        return [
                            'background-color: #f2f2f2' if '(Mon)' in val or '(월)' in val else ''
                            for val in s
                        ]

                    # 적용하기 전에 한글 요일로 변환
                    weekday_map = {
                        'Monday': '월요일',
                        'Tuesday': '화요일',
                        'Wednesday': '수요일',
                        'Thursday': '목요일',
                        'Friday': '금요일'
                    }
                    pred_df['요일'] = pred_df['요일'].map(lambda x: weekday_map.get(x, x))

                    # 스타일을 적용한 데이터프레임 표시
                    st.dataframe(pred_df.style.apply(highlight_weekday, subset=['날짜']), use_container_width=True)

                except Exception as e:
                    logger.error(f"예측 결과 테이블 생성 중 오류: {str(e)}")
                    # 간단한 테이블로 대체
                    simple_df = pd.DataFrame({
                        '날짜 인덱스': range(len(predicted_prices)),
                        '예측 가격': [f"{price:,.0f}원" for price in predicted_prices],
                        '변화율(%)': [f"{((p - last_price) / last_price * 100):.2f}%" for p in predicted_prices]
                    })
                    st.dataframe(simple_df, use_container_width=True)

            # 예측 해석 추가
            st.subheader("예측 결과 해석")

            # 예측 기간에 따른 분석
            short_term = predicted_prices[0:7]  # 1주
            mid_term = predicted_prices[7:21]  # 2-3주
            long_term = predicted_prices[21:]  # 3주 이상

            # 단기/중기/장기 추세 계산
            short_term_trend = "상승" if short_term[-1] > short_term[0] else "하락"
            mid_term_trend = "상승" if len(mid_term) > 0 and mid_term[-1] > mid_term[0] else "하락" if len(
                mid_term) > 0 else "데이터 없음"
            long_term_trend = "상승" if len(long_term) > 0 and long_term[-1] > long_term[0] else "하락" if len(
                long_term) > 0 else "데이터 없음"

            col1, col2, col3 = st.columns(3)

            with col1:
                st.write("#### 단기 전망 (7일)")
                short_change = (short_term[-1] - last_price) / last_price * 100
                short_color = "green" if short_change > 0 else "red"
                st.markdown(f"**예상 변동률:** <span style='color:{short_color};'>{short_change:+.2f}%</span>",
                            unsafe_allow_html=True)
                st.write(f"**추세:** {short_term_trend}")

                # 해석
                if short_change > 3:
                    st.success("단기적으로 강한 상승이 예상됩니다.")
                elif short_change > 0:
                    st.info("단기적으로 완만한 상승이 예상됩니다.")
                elif short_change > -3:
                    st.warning("단기적으로 소폭 하락이 예상됩니다.")
                else:
                    st.error("단기적으로 큰 폭의 하락이 예상됩니다.")

            with col2:
                st.write("#### 중기 전망 (7-21일)")
                if len(mid_term) > 0:
                    mid_change = (mid_term[-1] - last_price) / last_price * 100
                    mid_color = "green" if mid_change > 0 else "red"
                    st.markdown(f"**예상 변동률:** <span style='color:{mid_color};'>{mid_change:+.2f}%</span>",
                                unsafe_allow_html=True)
                    st.write(f"**추세:** {mid_term_trend}")

                    # 해석
                    if mid_change > 5:
                        st.success("중기적으로 강한 상승이 예상됩니다.")
                    elif mid_change > 0:
                        st.info("중기적으로 완만한 상승이 예상됩니다.")
                    elif mid_change > -5:
                        st.warning("중기적으로 소폭 하락이 예상됩니다.")
                    else:
                        st.error("중기적으로 큰 폭의 하락이 예상됩니다.")
                else:
                    st.info("중기 예측 데이터가 충분하지 않습니다.")

            with col3:
                st.write("#### 장기 전망 (21일 이상)")
                if len(long_term) > 0:
                    long_change = (long_term[-1] - last_price) / last_price * 100
                    long_color = "green" if long_change > 0 else "red"
                    st.markdown(f"**예상 변동률:** <span style='color:{long_color};'>{long_change:+.2f}%</span>",
                                unsafe_allow_html=True)
                    st.write(f"**추세:** {long_term_trend}")

                    # 해석
                    if long_change > 10:
                        st.success("장기적으로 강한 상승이 예상됩니다.")
                    elif long_change > 0:
                        st.info("장기적으로 완만한 상승이 예상됩니다.")
                    elif long_change > -10:
                        st.warning("장기적으로 소폭 하락이 예상됩니다.")
                    else:
                        st.error("장기적으로 큰 폭의 하락이 예상됩니다.")
                else:
                    st.info("장기 예측 데이터가 충분하지 않습니다.")

            # 종합 분석
            st.write("#### 종합 분석")

            # 추세 변화 감지
            trend_changes = []
            if short_term_trend != mid_term_trend and mid_term_trend != "데이터 없음":
                trend_changes.append(f"단기({short_term_trend})에서 중기({mid_term_trend})로 추세 변화")

            if mid_term_trend != long_term_trend and mid_term_trend != "데이터 없음" and long_term_trend != "데이터 없음":
                trend_changes.append(f"중기({mid_term_trend})에서 장기({long_term_trend})로 추세 변화")

            # 변동성 분석
            volatility = np.std(predicted_prices) / np.mean(predicted_prices) * 100

            # 종합 해석
            analysis_points = []

            # 추세 기반 분석
            if price_change_percent > 0:
                if volatility > 5:
                    analysis_points.append("전체적으로 상승 추세이나 변동성이 높아 주의가 필요합니다.")
                else:
                    analysis_points.append("전체적으로 안정적인 상승 추세가 예상됩니다.")
            else:
                if volatility > 5:
                    analysis_points.append("전체적으로 하락 추세이며 변동성이 높습니다. 투자에 신중한 접근이 필요합니다.")
                else:
                    analysis_points.append("전체적으로 완만한 하락 추세가 예상됩니다.")

            # 추세 변화 감지 시 분석
            if trend_changes:
                analysis_points.append("예측 기간 내 추세 변화가 감지되었습니다: " + ", ".join(trend_changes))

            # 변동성 분석
            if volatility < 2:
                analysis_points.append("예측된 주가 변동성이 낮아 비교적 안정적인 움직임이 예상됩니다.")
            elif volatility < 5:
                analysis_points.append("예측된 주가 변동성이 중간 수준입니다.")
            else:
                analysis_points.append(f"예측된 주가 변동성이 높습니다({volatility:.2f}%). 단기 매매 시 주의가 필요합니다.")

            # 사용된 특성 정보 표시
            if 'selected_features' in prediction_data:
                features = prediction_data['selected_features']
                if features:
                    if features == ['Auto'] or 'Auto' in features:
                        analysis_points.append("자동 특성 선택 알고리즘이 최적의 특성을 선택하여 예측했습니다.")
                    else:
                        analysis_points.append(f"예측에 사용된 주요 특성: {', '.join(features)}")

            # 모델 유형별 추가 분석
            if model_type and model_type.lower() in model_descriptions:
                if model_type.lower() == "hybrid":
                    analysis_points.append("하이브리드 앙상블 모델을 사용하여 여러 모델의 장점을 결합한 예측 결과입니다.")
                    if 'model_weights' in prediction_data:
                        weights = prediction_data['model_weights']
                        top_model = max(weights.items(), key=lambda x: x[1])[0]
                        analysis_points.append(f"앙상블에서 가장 큰 영향을 미친 모델은 '{top_model}'입니다.")
                elif model_type.upper() == "TFT":
                    analysis_points.append("TFT 모델은 시간적 특성과 변수 간 관계를 잘 포착하여 복잡한 패턴을 예측합니다.")
                elif model_type.upper() == "TCN":
                    analysis_points.append("TCN 모델은 다양한 시간 스케일의 패턴을 효율적으로 처리하여 예측 안정성을 높입니다.")
                elif model_type.upper() == "N-BEATS":
                    analysis_points.append("N-BEATS 모델은 추세와 계절성을 분리하여 분석하므로 복잡한 시계열 예측에 강점이 있습니다.")

            # 분석 결과 표시
            for point in analysis_points:
                st.write(f"- {point}")

        except Exception as e:
            st.error(f"예측 결과 표시 중 오류가 발생했습니다: {str(e)}")
            logger.error(f"예측 결과 표시 오류: {str(e)}")
            logger.error(traceback.format_exc())

            # 오류 시 예측 결과 구조 로깅
            if prediction_data is not None:
                logger.error(f"예측 데이터 키: {list(prediction_data.keys() if isinstance(prediction_data, dict) else [])}")
                logger.error(f"예측 데이터 구조: {type(prediction_data)}")

            st.info("예측 결과에 접근하는 데 문제가 발생했습니다. 개발자에게 로그를 확인하도록 요청하세요.")














    def generate_trading_days(self, start_date, num_days):
        """
        주어진 시작일로부터 지정된 수의 거래일(월-금)을 생성합니다.
        공휴일은 고려하지 않습니다.

        Args:
            start_date (datetime.date): 시작 날짜
            num_days (int): 생성할 거래일 수

        Returns:
            list: 거래일 목록 (datetime.date 객체)
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

    def _display_model_evaluation(self, evaluation_results, company_code=None):
        """모델 성능 평가 결과 표시 - y축 범위 개선"""
        try:
            # 고유한 key 생성을 위한 접두사 설정
            timestamp = int(time.time() * 1000)
            random_suffix = np.random.randint(1000, 9999)
            prefix = f"{company_code}_{timestamp}_{random_suffix}_" if company_code else f"eval_{timestamp}_{random_suffix}_"

            st.write("### 📊 모델 성능 평가")

            # 지표 표시
            st.write("**모델 성능 지표**")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("MSE (평균 제곱 오차)", f"{evaluation_results.get('mse', 0):.2f}")

            with col2:
                st.metric("RMSE (평균 제곱근 오차)", f"{evaluation_results.get('rmse', 0):.2f}")
                if 'relative_rmse' in evaluation_results:
                    st.metric("상대 RMSE (%)", f"{evaluation_results['relative_rmse']:.2f}%")

            with col3:
                st.metric("MAPE (평균 절대 백분율 오차)", f"{evaluation_results.get('mape', 0):.2f}%")

            # 해석 가이드
            if 'relative_rmse' in evaluation_results:
                if evaluation_results['relative_rmse'] < 3:
                    st.success("모델의 예측 오차가 현재 주가의 3% 미만으로 비교적 정확합니다.")
                elif evaluation_results['relative_rmse'] < 5:
                    st.info("모델의 예측 오차가 현재 주가의 3~5% 수준으로 보통 수준입니다.")
                else:
                    st.warning("모델의 예측 오차가 현재 주가의 5% 이상으로 신뢰도가 낮을 수 있습니다.")

            # 테스트 세트 예측 성능 그래프 개선
            if 'y_test' in evaluation_results and 'y_pred' in evaluation_results:
                # 데이터 준비 및 유효성 검사
                y_test_orig = evaluation_results['y_test']
                y_pred_orig = evaluation_results['y_pred']

                # 데이터 형태 검사 및 변환
                if isinstance(y_test_orig, list) or isinstance(y_test_orig, np.ndarray):
                    # 2차원 배열인 경우 1차원으로 변환
                    if hasattr(y_test_orig, 'shape') and len(y_test_orig.shape) > 1:
                        y_test_data = y_test_orig.flatten()
                    else:
                        y_test_data = np.array(y_test_orig)

                    # 2차원 배열인 경우 1차원으로 변환
                    if hasattr(y_pred_orig, 'shape') and len(y_pred_orig.shape) > 1:
                        y_pred_data = y_pred_orig.flatten()
                    else:
                        y_pred_data = np.array(y_pred_orig)

                    # 데이터 길이 제한 (너무 많은 포인트는 그래프를 느리게 만듦)
                    max_points = 100
                    if len(y_test_data) > max_points:
                        # 균등하게 샘플링
                        step = len(y_test_data) // max_points
                        y_test_data = y_test_data[::step]
                        y_pred_data = y_pred_data[::step]

                    # 실제로 데이터 변화가 있는지 확인
                    y_test_range = np.max(y_test_data) - np.min(y_test_data)
                    y_pred_range = np.max(y_pred_data) - np.min(y_pred_data)

                    if y_test_range > 0.001 and y_pred_range > 0.001:  # 의미 있는 변화가 있는 경우에만 그래프 생성
                        fig = go.Figure()

                        # X축 인덱스 생성
                        x_indices = np.arange(len(y_test_data))

                        # 절대 오차 계산
                        error_data = np.abs(y_test_data - y_pred_data)

                        # ===== Y축 범위 개선을 위한 코드 =====
                        # 모든 가격 데이터 수집 (실제 가격과 예측 가격)
                        all_prices = np.concatenate([y_test_data, y_pred_data])

                        # 최소/최대값 및 범위 계산
                        min_price = np.min(all_prices)
                        max_price = np.max(all_prices)
                        price_range = max_price - min_price

                        # 여백 추가 (범위의 10%)
                        padding = price_range * 0.1
                        y_min = min_price - padding
                        y_max = max_price + padding

                        # 실제 가격
                        fig.add_trace(
                            go.Scatter(
                                x=x_indices,
                                y=y_test_data,
                                mode='lines',
                                name='실제 가격',
                                line=dict(color='royalblue', width=3)
                            )
                        )

                        # 예측 가격
                        fig.add_trace(
                            go.Scatter(
                                x=x_indices,
                                y=y_pred_data,
                                mode='lines',
                                name='예측 가격',
                                line=dict(color='firebrick', width=3, dash='dash')
                            )
                        )

                        # Y축 범위 설정 (실제 가격과 예측 가격만 포함)
                        fig.update_layout(
                            yaxis=dict(
                                range=[y_min, y_max],
                                title=dict(text="주가", font=dict(color="black")), 
                                side="left"
                            )
                        )
                        # fig.update_layout(
                        #     yaxis=dict(
                        #         range=[y_min, y_max],
                        #         title="주가",
                        #         titlefont=dict(color="black"),
                        #         side="left"
                        #     )
                        # )

                        # 별도의 Y축에 오차 표시 (두 번째 Y축)
                        fig.add_trace(
                            go.Scatter(
                                x=x_indices,
                                y=error_data,
                                mode='lines',
                                name='절대 오차',
                                line=dict(color='green', width=2, dash='dot'),
                                yaxis="y2"  # 두 번째 Y축에 표시
                            )
                        )

                        # 두 번째 Y축 설정 (오차용)
                        error_max = np.max(error_data)
                        fig.update_layout(
                            yaxis2=dict(
                                title=dict(text="절대 오차", font=dict(color="green")),  
                                tickfont=dict(color="green"),
                                anchor="x",
                                overlaying="y",
                                side="right",
                                range=[0, error_max * 1.1]  # 오차의 최대값에 맞춰 설정
                            )
                        )
                        # fig.update_layout(
                        #     yaxis2=dict(
                        #         title="절대 오차",
                        #         titlefont=dict(color="green"),
                        #         tickfont=dict(color="green"),
                        #         anchor="x",
                        #         overlaying="y",
                        #         side="right",
                        #         range=[0, error_max * 1.1]  # 오차의 최대값에 맞춰 설정
                        #     )
                        # )

                        # 차트 레이아웃 설정
                        fig.update_layout(
                            title='테스트 세트 예측 성능',
                            xaxis_title='샘플 인덱스',
                            height=400,
                            hovermode='x unified',
                            # 여백 최소화
                            margin=dict(l=40, r=40, t=40, b=40),
                            legend=dict(
                                orientation="h",  # 수평 레이아웃
                                yanchor="bottom",
                                y=1.02,
                                xanchor="right",
                                x=1
                            )
                        )

                        # X축 숫자 테이블 형식 조정
                        fig.update_xaxes(
                            showticklabels=True,  # 눈금 레이블 표시
                            tickvals=x_indices[::max(1, len(x_indices) // 10)],  # 10개 정도의 눈금만 표시
                            ticktext=[f"{i + 1}" for i in range(0, len(x_indices), max(1, len(x_indices) // 10))]
                            # 1부터 시작하는 숫자로 표시
                        )

                        st.plotly_chart(fig, use_container_width=True, key=f"{prefix}test_vs_pred_chart")
                    else:
                        st.warning("데이터 변화가 너무 작아 의미 있는 그래프를 그릴 수 없습니다.")
                else:
                    st.warning("테스트 및 예측 데이터 형식이 지원되지 않습니다.")

                # 오차 분석
                st.write("**예측 오차 분석**")

                # 오차 히스토그램
                if 'errors' in evaluation_results and len(evaluation_results['errors']) > 0:
                    fig_hist = go.Figure()
                    fig_hist.add_trace(
                        go.Histogram(
                            x=evaluation_results['errors'],
                            nbinsx=30,
                            name='오차 분포',
                            marker_color='royalblue'
                        )
                    )

                    fig_hist.update_layout(
                        title='예측 오차 히스토그램',
                        xaxis_title='오차',
                        yaxis_title='빈도',
                        height=300,
                        margin=dict(l=40, r=40, t=40, b=40)  # 여백 최소화
                    )

                    st.plotly_chart(fig_hist, use_container_width=True, key=f"{prefix}error_distribution_chart")

                # 오차 통계
                mean_error = evaluation_results.get('mean_error', np.mean(evaluation_results.get('errors', [0])))
                std_error = evaluation_results.get('std_error', np.std(evaluation_results.get('errors', [0])))

                st.write(f"**오차 평균**: {mean_error:.2f}")
                st.write(f"**오차 표준편차**: {std_error:.2f}")

                # 편향성 분석
                bias = evaluation_results.get('bias', 'unknown')
                if bias == 'unbiased' or np.abs(mean_error) < std_error * 0.5:
                    st.success("예측 오차가 정규분포에 가까우며, 평균이 0에 가까워 모델이 편향되지 않았습니다.")
                else:
                    st.warning("예측 오차의 평균이 0에서 벗어나 있어 모델에 편향이 있을 수 있습니다.")

            # 신뢰도 평가
            if 'relative_rmse' in evaluation_results:
                st.write("**예측 신뢰도 평가**")

                relative_rmse = evaluation_results['relative_rmse']

                if relative_rmse < 3:
                    confidence = "높음"
                    confidence_color = "green"
                    confidence_text = "모델의 예측 신뢰도가 높습니다."
                elif relative_rmse < 5:
                    confidence = "중간"
                    confidence_color = "orange"
                    confidence_text = "모델의 예측 신뢰도가 보통 수준입니다."
                else:
                    confidence = "낮음"
                    confidence_color = "red"
                    confidence_text = "모델의 예측 신뢰도가 낮습니다. 결과 해석에 주의가 필요합니다."

                st.markdown(f"**신뢰도 수준:** <span style='color:{confidence_color};'>{confidence}</span>",
                            unsafe_allow_html=True)
                st.write(confidence_text)

        except Exception as e:
            st.error(f"모델 평가 결과 표시 중 오류가 발생했습니다: {str(e)}")
            logger.error(f"모델 평가 결과 표시 오류: {str(e)}")
            logger.error(traceback.format_exc())

    def get_prediction_data(self, symbol):
        """예측 데이터 가져오기"""
        data = load_and_preprocess_data(symbol)
        prediction = create_and_evaluate_model(data)
        return prediction

def load_and_preprocess_data(symbol):
    """데이터 로딩 및 전처리"""
    data = load_data(symbol)
    preprocessed_data = preprocess_data(data)
    return preprocessed_data

def create_and_evaluate_model(data, model_type='lstm'):
    """모델 생성 및 평가"""
    model = create_model(data, model_type)
    evaluation = evaluate_model(model, data)
    return evaluation

def display_prediction_results(prediction_data):
    """예측 결과 시각화"""
    st.subheader("예측 결과")
    st.line_chart(prediction_data)