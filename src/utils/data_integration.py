# src/utils/data_integration.py

import logging
import streamlit as st
from datetime import datetime
import json
import numpy as np
import pandas as pd

logger = logging.getLogger('StockAnalysisApp.DataIntegration')


class DataIntegrationManager:
    """
    다양한 분석 탭 간의 데이터 통합을 관리하는 클래스
    종합 리포트를 위한 중앙 데이터 수집 및 변환 담당
    """

    _instance = None

    def __init__(self):
        # Initialize the data_listeners attribute as an empty list
        self.data_listeners = []

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DataIntegrationManager, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        """데이터 저장소 초기화"""
        # 통합 데이터 세션 상태 초기화
        if 'integrated_data' not in st.session_state:
            st.session_state.integrated_data = {
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

        self.data_listeners = []  # 데이터 변경 리스너 목록

    def register_data(self, category, data, source=None):
        """
        특정 카테고리의 데이터 등록

        Args:
            category (str): 데이터 카테고리 (stock_detail, technical_analysis 등)
            data (dict): 등록할 데이터
            source (str, optional): 데이터 소스 (탭 이름 등)
        """
        try:
            # 중첩 딕셔너리 구조 처리
            if '.' in category:
                main_cat, sub_cat = category.split('.', 1)
                if main_cat not in st.session_state.integrated_data:
                    st.session_state.integrated_data[main_cat] = {}

                # 중첩 딕셔너리 업데이트
                if isinstance(st.session_state.integrated_data[main_cat], dict):
                    st.session_state.integrated_data[main_cat][sub_cat] = data
                else:
                    st.session_state.integrated_data[main_cat] = {sub_cat: data}
            else:
                # 일반 카테고리 업데이트
                st.session_state.integrated_data[category] = data

            # 마지막 업데이트 시간 기록
            st.session_state.integrated_data['last_update'][category] = {
                'timestamp': datetime.now().isoformat(),
                'source': source
            }

            # 캐시 무효화
            if 'analysis_cache' in st.session_state.integrated_data:
                st.session_state.integrated_data['analysis_cache'] = {}

            # 리스너 알림
            self._notify_listeners(category, data)

            logger.info(f"카테고리 '{category}' 데이터가 등록되었습니다. 소스: {source}")
            return True

        except Exception as e:
            logger.error(f"데이터 등록 중 오류 발생: {str(e)}", exc_info=True)
            return False

    def get_data(self, category=None):
        """
        등록된 데이터 조회

        Args:
            category (str, optional): 조회할 카테고리. None이면 전체 데이터 반환

        Returns:
            dict: 요청한 카테고리의 데이터 또는 전체 데이터
        """
        try:
            if not category:
                return st.session_state.integrated_data

            # 중첩 카테고리 처리
            if '.' in category:
                main_cat, sub_cat = category.split('.', 1)
                if main_cat in st.session_state.integrated_data and isinstance(
                        st.session_state.integrated_data[main_cat], dict):
                    return st.session_state.integrated_data[main_cat].get(sub_cat)
                return None

            return st.session_state.integrated_data.get(category)

        except Exception as e:
            logger.error(f"데이터 조회 중 오류 발생: {str(e)}", exc_info=True)
            return None

    def add_data_listener(self, listener_func):
        """
        데이터 변경 리스너 등록

        Args:
            listener_func: 데이터 변경 시 호출될 함수 (category, data를 인자로 받음)
        """
        if listener_func not in self.data_listeners:
            self.data_listeners.append(listener_func)

    def _notify_listeners(self, category, data):
        """등록된 리스너에게 데이터 변경 알림"""
        for listener in self.data_listeners:
            try:
                listener(category, data)
            except Exception as e:
                logger.error(f"리스너 호출 중 오류 발생: {str(e)}", exc_info=True)

    def prepare_for_llm(self, detail_level="standard"):
        """
        LLM(Large Language Model)에 전달할 데이터 준비

        Args:
            detail_level (str): 데이터 상세 수준 (minimal, standard, comprehensive)

        Returns:
            dict: LLM에 전달할 형태로 가공된 데이터
        """
        data = st.session_state.integrated_data

        # 모든 데이터를 JSON 직렬화 가능한 형태로 변환
        processed_data = self._process_for_serialization(data)

        # 상세 수준에 따라 데이터 필터링
        if detail_level == "minimal":
            return self._extract_minimal_data(processed_data)
        elif detail_level == "standard":
            return self._extract_standard_data(processed_data)
        else:  # comprehensive
            return processed_data

    def _process_for_serialization(self, data):
        """데이터를 JSON 직렬화 가능한 형태로 변환"""
        if isinstance(data, dict):
            result = {}
            for key, value in data.items():
                result[key] = self._process_for_serialization(value)
            return result
        elif isinstance(data, list):
            return [self._process_for_serialization(item) for item in data]
        elif isinstance(data, np.ndarray):
            return data.tolist()
        elif isinstance(data, pd.DataFrame) or isinstance(data, pd.Series):
            return "DataFrame (too large for serialization)"
        elif isinstance(data, datetime):
            return data.isoformat()
        else:
            return data

    def export_data(self, format="json"):
        """
        통합 데이터 내보내기

        Args:
            format (str): 내보내기 형식 ('json', 'csv' 등)

        Returns:
            str or bytes: 직렬화된 데이터
        """
        try:
            data = self._process_for_serialization(st.session_state.integrated_data)

            if format.lower() == "json":
                return json.dumps(data, indent=2, ensure_ascii=False)
            else:
                # 다른 형식 지원 필요시 구현
                raise ValueError(f"지원하지 않는 형식: {format}")
        except Exception as e:
            logger.error(f"데이터 내보내기 중 오류: {str(e)}", exc_info=True)
            return None