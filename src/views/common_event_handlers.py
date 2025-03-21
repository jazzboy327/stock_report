# src/views/common_event_handlers.py

import streamlit as st
import logging
from datetime import datetime
from src.utils.data_integration import DataIntegrationManager

logger = logging.getLogger('StockAnalysisApp.EventHandlers')


class AnalysisEventHandler:
    """
    분석 탭 간 이벤트 핸들링과 데이터 공유를 위한 클래스
    """

    @staticmethod
    def register_tab_completion(tab_name, data, target_view=None):
        """
        특정 탭의 분석 완료 이벤트 등록

        Args:
            tab_name (str): 완료된 탭 이름
            data (dict): 탭에서 생성된 분석 데이터
            target_view (str, optional): 특정 뷰를 지정 (기본값은 모든 뷰에 알림)
        """
        try:
            # 세션 상태에 저장 (기존 방식과의 호환성 유지)
            if tab_name == "risk_metrics":
                st.session_state.risk_metrics = data
            elif tab_name == "growth_data":
                st.session_state.growth_data = data
            elif tab_name == "trading_signals":
                st.session_state.trading_signals = data

            # 데이터 통합 매니저에 등록
            integration_manager = DataIntegrationManager()
            integration_manager.register_data(tab_name, data, source=f"{tab_name}_tab")

            # 이벤트 발생 시간 기록
            if 'analysis_events' not in st.session_state:
                st.session_state.analysis_events = {}

            st.session_state.analysis_events[tab_name] = {
                'timestamp': datetime.now().isoformat(),
                'status': 'completed'
            }

            logger.info(f"탭 '{tab_name}' 분석 완료 이벤트 등록됨")

            # 추가 작업이 필요한 경우 여기에 구현
            # 예: 종합 보고서 자동 갱신 등

        except Exception as e:
            logger.error(f"탭 완료 이벤트 등록 중 오류: {str(e)}", exc_info=True)

    @staticmethod
    def get_tab_status(tab_name):
        """
        특정 탭의 분석 상태 확인

        Args:
            tab_name (str): 확인할 탭 이름

        Returns:
            dict: 탭의 상태 정보 또는 None
        """
        if 'analysis_events' in st.session_state:
            return st.session_state.analysis_events.get(tab_name)
        return None

    @staticmethod
    def share_data_between_tabs(from_tab, to_tab, data_key, transform_func=None):
        """
        탭 간 데이터 공유

        Args:
            from_tab (str): 소스 탭 이름
            to_tab (str): 대상 탭 이름
            data_key (str): 공유할 데이터 키
            transform_func (callable, optional): 데이터 변환 함수
        """
        try:
            # 소스 탭 데이터 가져오기
            integration_manager = DataIntegrationManager()
            source_data = integration_manager.get_data(from_tab)

            if not source_data or data_key not in source_data:
                logger.warning(f"'{from_tab}' 탭에서 '{data_key}' 데이터를 찾을 수 없습니다.")
                return False

            # 필요시 데이터 변환
            if transform_func:
                shared_data = transform_func(source_data[data_key])
            else:
                shared_data = source_data[data_key]

            # 대상 탭에 데이터 등록
            integration_manager.register_data(f"{to_tab}.{data_key}", shared_data, source=from_tab)

            logger.info(f"'{from_tab}'에서 '{to_tab}'으로 '{data_key}' 데이터 공유 완료")
            return True

        except Exception as e:
            logger.error(f"탭 간 데이터 공유 중 오류: {str(e)}", exc_info=True)
            return False