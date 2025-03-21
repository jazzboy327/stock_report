#!/usr/bin/env python
"""
KRX API 직접 요청 테스트 스크립트
- 라이브러리 없이 직접 KRX API에 요청하여 응답 확인
- JSON 응답 구조 분석 및 저장
"""

import requests
import json
from datetime import datetime, timedelta
import logging
import sys
import time

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('krx_direct_test.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger('krx_direct_test')

# KRX API 엔드포인트
KRX_API_URL = "http://data.krx.co.kr/comm/bldAttendant/getJsonData.cmd"

def save_response_to_file(filename, data):
    """응답 데이터를 파일로 저장"""
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    logger.info(f"응답 저장 완료: {filename}")

def test_stock_listing_api():
    """상장 종목 검색 API 테스트"""
    logger.info("===== 상장 종목 검색 API 테스트 =====")
    
    params = {
        'bld': 'dbms/MDC/STAT/standard/MDCSTAT01901',
        'locale': 'ko_KR',
        'mktsel': 'ALL',  # 전체 시장
        'typeNo': 0
    }
    
    try:
        logger.info(f"요청 URL: {KRX_API_URL}")
        logger.info(f"요청 파라미터: {params}")
        
        response = requests.post(KRX_API_URL, data=params, timeout=10)
        
        logger.info(f"응답 상태 코드: {response.status_code}")
        logger.info(f"응답 헤더: {dict(response.headers)}")
        
        if response.status_code == 200:
            try:
                data = response.json()
                logger.info("JSON 파싱 성공!")
                
                # 응답 구조 확인
                logger.info(f"응답 키: {list(data.keys())}")
                
                if 'OutBlock_1' in data and data['OutBlock_1']:
                    total_count = len(data['OutBlock_1'])
                    logger.info(f"총 종목 수: {total_count}")
                    
                    # 첫 5개 종목 정보 출력
                    for i, item in enumerate(data['OutBlock_1'][:5]):
                        logger.info(f"종목 {i+1}: {item}")
                    
                    # 응답 저장
                    save_response_to_file('krx_listing_response.json', data)
                    
                    # 특정 종목 찾기 (문제가 있었던 종목)
                    problem_ticker = "030200"
                    found = False
                    
                    for item in data['OutBlock_1']:
                        if 'ISU_SRT_CD' in item and item['ISU_SRT_CD'] == problem_ticker:
                            logger.info(f"문제 종목({problem_ticker}) 정보 발견: {item}")
                            found = True
                            break
                    
                    if not found:
                        logger.warning(f"문제 종목({problem_ticker})이 상장 종목 목록에 없습니다.")
                else:
                    logger.warning("응답에 종목 목록이 없습니다.")
            
            except json.JSONDecodeError as e:
                logger.error(f"JSON 파싱 오류: {str(e)}")
                logger.error(f"응답 내용: {response.text[:200]}...")  # 응답 앞부분만 출력
                
                # 오류 위치 주변의 문자열 출력
                error_pos = e.pos
                start_pos = max(0, error_pos - 30)
                end_pos = min(len(response.text), error_pos + 30)
                
                logger.error(f"오류 위치 주변: '{response.text[start_pos:end_pos]}'")
                logger.error(f"오류 위치 문자 인덱스: {error_pos}")
                
                # 응답 전체를 파일로 저장
                with open('krx_listing_error_response.txt', 'w', encoding='utf-8') as f:
                    f.write(response.text)
                logger.info("오류 응답 저장 완료: krx_listing_error_response.txt")
        else:
            logger.error(f"HTTP 오류: {response.status_code}")
            logger.error(f"응답 내용: {response.text[:200]}...")
    
    except requests.RequestException as e:
        logger.error(f"요청 오류: {str(e)}")

def test_investor_trading_api(ticker="005930"):
    """투자자별 거래실적 API 테스트"""
    logger.info(f"===== 투자자별 거래실적 API 테스트 (종목: {ticker}) =====")
    
    # 날짜 범위 설정
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    
    start_date_str = start_date.strftime("%Y%m%d")
    end_date_str = end_date.strftime("%Y%m%d")
    
    params = {
        'bld': 'dbms/MDC/STAT/standard/MDCSTAT02201',
        'strtDd': start_date_str,
        'endDd': end_date_str,
        'isuCd': ticker,
        'inqTpCd': 2,  # 거래대금
        'trdVolTpCd': 1,  # 순매수
        'askBid': False
    }
    
    try:
        logger.info(f"요청 URL: {KRX_API_URL}")
        logger.info(f"요청 파라미터: {params}")
        
        response = requests.post(KRX_API_URL, data=params, timeout=10)
        
        logger.info(f"응답 상태 코드: {response.status_code}")
        logger.info(f"응답 헤더: {dict(response.headers)}")
        
        if response.status_code == 200:
            try:
                data = response.json()
                logger.info("JSON 파싱 성공!")
                
                # 응답 구조 확인
                logger.info(f"응답 키: {list(data.keys())}")
                
                if 'OutBlock_1' in data and data['OutBlock_1']:
                    total_count = len(data['OutBlock_1'])
                    logger.info(f"총 거래일 수: {total_count}")
                    
                    # 첫 5개 거래일 정보 출력
                    for i, item in enumerate(data['OutBlock_1'][:5]):
                        logger.info(f"거래일 {i+1}: {item}")
                    
                    # 응답 저장
                    filename = f'krx_investor_{ticker}_response.json'
                    save_response_to_file(filename, data)
                else:
                    logger.warning("응답에 거래 데이터가 없습니다.")
            
            except json.JSONDecodeError as e:
                logger.error(f"JSON 파싱 오류: {str(e)}")
                logger.error(f"응답 내용: {response.text[:200]}...")  # 응답 앞부분만 출력
                
                # 오류 위치 주변의 문자열 출력
                error_pos = e.pos
                start_pos = max(0, error_pos - 30)
                end_pos = min(len(response.text), error_pos + 30)
                
                logger.error(f"오류 위치 주변: '{response.text[start_pos:end_pos]}'")
                logger.error(f"오류 위치 문자 인덱스: {error_pos}")
                
                # 응답 전체를 파일로 저장
                filename = f'krx_investor_{ticker}_error_response.txt'
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(response.text)
                logger.info(f"오류 응답 저장 완료: {filename}")
        else:
            logger.error(f"HTTP 오류: {response.status_code}")
            logger.error(f"응답 내용: {response.text[:200]}...")
    
    except requests.RequestException as e:
        logger.error(f"요청 오류: {str(e)}")

def test_delisted_stocks_api():
    """상장폐지 종목 검색 API 테스트"""
    logger.info("===== 상장폐지 종목 검색 API 테스트 =====")
    
    params = {
        'bld': 'dbms/MDC/STAT/standard/MDCSTAT01801',
        'mktsel': 'ALL',  # 전체 시장
        'searchText': '',
        'typeNo': 0
    }
    
    try:
        logger.info(f"요청 URL: {KRX_API_URL}")
        logger.info(f"요청 파라미터: {params}")
        
        response = requests.post(KRX_API_URL, data=params, timeout=10)
        
        logger.info(f"응답 상태 코드: {response.status_code}")
        logger.info(f"응답 헤더: {dict(response.headers)}")
        
        if response.status_code == 200:
            try:
                data = response.json()
                logger.info("JSON 파싱 성공!")
                
                # 응답 구조 확인
                logger.info(f"응답 키: {list(data.keys())}")
                
                if 'OutBlock_1' in data and data['OutBlock_1']:
                    total_count = len(data['OutBlock_1'])
                    logger.info(f"총 상장폐지 종목 수: {total_count}")
                    
                    # 첫 5개 종목 정보 출력
                    for i, item in enumerate(data['OutBlock_1'][:5]):
                        logger.info(f"상장폐지 종목 {i+1}: {item}")
                    
                    # 응답 저장
                    save_response_to_file('krx_delisted_response.json', data)
                    
                    # 특정 종목 찾기 (문제가 있었던 종목)
                    problem_ticker = "030200"
                    found = False
                    
                    for item in data['OutBlock_1']:
                        if 'ISU_SRT_CD' in item and item['ISU_SRT_CD'] == problem_ticker:
                            logger.info(f"문제 종목({problem_ticker}) 정보 발견: {item}")
                            found = True
                            break
                    
                    if not found:
                        logger.warning(f"문제 종목({problem_ticker})이 상장폐지 종목 목록에도 없습니다.")
                else:
                    logger.warning("응답에 상장폐지 종목 목록이 없습니다.")
            
            except json.JSONDecodeError as e:
                logger.error(f"JSON 파싱 오류: {str(e)}")
                logger.error(f"응답 내용: {response.text[:200]}...")  # 응답 앞부분만 출력
                
                # 오류 위치 주변의 문자열 출력
                error_pos = e.pos
                start_pos = max(0, error_pos - 30)
                end_pos = min(len(response.text), error_pos + 30)
                
                logger.error(f"오류 위치 주변: '{response.text[start_pos:end_pos]}'")
                logger.error(f"오류 위치 문자 인덱스: {error_pos}")
                
                # 응답 전체를 파일로 저장
                with open('krx_delisted_error_response.txt', 'w', encoding='utf-8') as f:
                    f.write(response.text)
                logger.info("오류 응답 저장 완료: krx_delisted_error_response.txt")
        else:
            logger.error(f"HTTP 오류: {response.status_code}")
            logger.error(f"응답 내용: {response.text[:200]}...")
    
    except requests.RequestException as e:
        logger.error(f"요청 오류: {str(e)}")

def compare_stock_requests(normal_ticker="005930", problem_ticker="030200"):
    """정상 종목과 문제 종목의 요청 비교"""
    logger.info(f"===== 종목 요청 비교: 정상({normal_ticker}) vs 문제({problem_ticker}) =====")
    
    # 날짜 범위 설정
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    
    start_date_str = start_date.strftime("%Y%m%d")
    end_date_str = end_date.strftime("%Y%m%d")
    
    # 공통 파라미터
    common_params = {
        'bld': 'dbms/MDC/STAT/standard/MDCSTAT02201',
        'strtDd': start_date_str,
        'endDd': end_date_str,
        'inqTpCd': 2,  # 거래대금
        'trdVolTpCd': 1,  # 순매수
        'askBid': False
    }
    
    # 각 종목별 파라미터 생성
    normal_params = dict(common_params)
    normal_params['isuCd'] = normal_ticker
    
    problem_params = dict(common_params)
    problem_params['isuCd'] = problem_ticker
    
    # 헤더 설정 - user-agent 추가
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Accept': 'application/json, text/javascript, */*; q=0.01',
        'Accept-Language': 'ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7',
        'Content-Type': 'application/x-www-form-urlencoded; charset=UTF-8',
        'Origin': 'http://data.krx.co.kr',
        'Referer': 'http://data.krx.co.kr/contents/MDC/MDI/mdiLoader/index.cmd?menuId=MDC0201'
    }
    
    # 정상 종목 요청
    logger.info(f"정상 종목({normal_ticker}) 요청 시작...")
    
    try:
        normal_response = requests.post(KRX_API_URL, data=normal_params, headers=headers, timeout=10)
        logger.info(f"정상 종목 응답 상태 코드: {normal_response.status_code}")
        
        if normal_response.status_code == 200:
            try:
                normal_data = normal_response.json()
                logger.info("정상 종목 JSON 파싱 성공!")
                if 'OutBlock_1' in normal_data:
                    logger.info(f"정상 종목 데이터 건수: {len(normal_data['OutBlock_1'])}")
                
                # 응답 저장
                save_response_to_file(f'krx_compare_normal_{normal_ticker}.json', normal_data)
            except json.JSONDecodeError as e:
                logger.error(f"정상 종목 JSON 파싱 오류: {str(e)}")
        else:
            logger.error(f"정상 종목 HTTP 오류: {normal_response.status_code}")
    
    except requests.RequestException as e:
        logger.error(f"정상 종목 요청 오류: {str(e)}")
    
    # 약간의 간격을 두고 문제 종목 요청
    time.sleep(2)
    
    # 문제 종목 요청
    logger.info(f"문제 종목({problem_ticker}) 요청 시작...")
    
    try:
        problem_response = requests.post(KRX_API_URL, data=problem_params, headers=headers, timeout=10)
        logger.info(f"문제 종목 응답 상태 코드: {problem_response.status_code}")
        
        if problem_response.status_code == 200:
            # 파일에 응답 전체 저장 (파싱 전)
            with open(f'krx_compare_problem_{problem_ticker}_raw.txt', 'w', encoding='utf-8') as f:
                f.write(problem_response.text)
            logger.info(f"문제 종목 원시 응답 저장 완료: krx_compare_problem_{problem_ticker}_raw.txt")
            
            try:
                problem_data = problem_response.json()
                logger.info("문제 종목 JSON 파싱 성공!")
                if 'OutBlock_1' in problem_data:
                    logger.info(f"문제 종목 데이터 건수: {len(problem_data['OutBlock_1'])}")
                
                # 응답 저장
                save_response_to_file(f'krx_compare_problem_{problem_ticker}.json', problem_data)
            except json.JSONDecodeError as e:
                logger.error(f"문제 종목 JSON 파싱 오류: {str(e)}")
                # 오류 위치 주변의 문자열 출력
                error_pos = e.pos
                start_pos = max(0, error_pos - 30)
                end_pos = min(len(problem_response.text), error_pos + 30)
                
                logger.error(f"오류 위치 주변: '{problem_response.text[start_pos:end_pos]}'")
                logger.error(f"오류 위치 문자 인덱스: {error_pos}")
        else:
            logger.error(f"문제 종목 HTTP 오류: {problem_response.status_code}")
    
    except requests.RequestException as e:
        logger.error(f"문제 종목 요청 오류: {str(e)}")
    
    # 응답 비교 결과 요약
    logger.info("===== 요청 비교 결과 =====")
    logger.info(f"정상 종목({normal_ticker}) 응답 상태: {normal_response.status_code}")
    logger.info(f"문제 종목({problem_ticker}) 응답 상태: {problem_response.status_code}")
    
    if normal_response.status_code == 200 and problem_response.status_code == 200:
        logger.info("두 요청 모두 200 OK 상태코드를 반환했습니다.")
        
        # 응답 길이 비교
        normal_len = len(normal_response.text)
        problem_len = len(problem_response.text)
        
        logger.info(f"정상 종목 응답 길이: {normal_len} 바이트")
        logger.info(f"문제 종목 응답 길이: {problem_len} 바이트")
        
        if normal_len != problem_len:
            logger.info(f"응답 길이 차이: {abs(normal_len - problem_len)} 바이트")
    
    logger.info("자세한 비교는 저장된 파일을 확인하세요.")

def main():
    """메인 함수"""
    logger.info("===== KRX API 직접 요청 테스트 시작 =====")
    logger.info(f"시작 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 상장 종목 검색 API 테스트
    test_stock_listing_api()
    
    # 투자자별 거래실적 API 테스트 (정상 종목)
    test_investor_trading_api("005930")  # 삼성전자
    
    # 투자자별 거래실적 API 테스트 (문제 종목)
    test_investor_trading_api("030200")  # 문제가 발생했던 종목
    
    # 상장폐지 종목 검색 API 테스트
    test_delisted_stocks_api()
    
    # 정상 종목과 문제 종목 요청 비교 (헤더 추가, 원시 응답 저장)
    compare_stock_requests("005930", "030200")
    
    logger.info(f"종료 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("===== KRX API 직접 요청 테스트 종료 =====")

if __name__ == "__main__":
    main()