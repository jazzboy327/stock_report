#!/usr/bin/env python
"""
KRX API 테스트 스크립트
- KRX API의 연결 상태 및 응답 형식을 테스트
- 네트워크 오류 또는 JSON 파싱 오류를 확인
- pykrx 라이브러리의 내부 요청 상세 정보 확인
"""

import sys
import json
import traceback
import requests
import logging
from datetime import datetime, timedelta
import time

# 원시 HTTP 요청과 응답을 로깅
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('krx_api_test.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger('krx_api_test')

# KRX API 테스트 함수들
def test_krx_api_direct():
    """KRX에 직접 HTTP 요청을 보내 API 응답을 검증"""
    logger.info("===== KRX API 직접 요청 테스트 시작 =====")
    
    try:
        # KRX API 엔드포인트 - pykrx 라이브러리에서 사용하는 것과 동일
        base_url = "http://data.krx.co.kr/comm/bldAttendant/getJsonData.cmd"
        
        # 현재 날짜 기준으로 조회 기간 설정
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        
        end_date_str = end_date.strftime("%Y%m%d")
        start_date_str = start_date.strftime("%Y%m%d")
        
        # 테스트할 종목 코드
        ticker = "005930"  # 삼성전자 (일반적으로 잘 작동하는 종목)
        
        # 1. 상장종목 검색 API 테스트
        logger.info("1. 상장종목 검색 API 테스트")
        listing_params = {
            'bld': 'dbms/MDC/STAT/standard/MDCSTAT01901',
            'locale': 'ko_KR',
            'mktsel': 'ALL',  # 전체 시장
            'typeNo': 0
        }
        
        # 요청 전 로깅
        logger.info(f"요청 URL: {base_url}")
        logger.info(f"요청 파라미터: {listing_params}")
        
        # HTTP 요청 시도
        try:
            resp = requests.post(base_url, data=listing_params, timeout=10)
            logger.info(f"응답 상태 코드: {resp.status_code}")
            logger.info(f"응답 헤더: {dict(resp.headers)}")
            
            # 응답 내용 일부 로깅 (너무 길 수 있음)
            logger.info(f"응답 내용 미리보기: {resp.text[:500]}...")
            
            # JSON 파싱 시도
            try:
                json_data = resp.json()
                logger.info("JSON 파싱 성공!")
                
                # 응답 구조 확인
                if 'OutBlock_1' in json_data:
                    sample_data = json_data['OutBlock_1'][:2]  # 첫 2개 항목만 표시
                    logger.info(f"데이터 샘플: {json.dumps(sample_data, indent=2, ensure_ascii=False)}")
                    logger.info(f"총 종목 수: {len(json_data['OutBlock_1'])}")
                else:
                    logger.warning(f"예상 데이터 구조가 없음. 키: {list(json_data.keys())}")
            
            except json.JSONDecodeError as e:
                logger.error(f"JSON 파싱 오류: {str(e)}")
                logger.error(f"오류 위치 문자: {resp.text[max(0, e.pos-20):min(len(resp.text), e.pos+20)]}")
                logger.error(f"전체 응답 내용: {resp.text}")
        
        except requests.RequestException as e:
            logger.error(f"HTTP 요청 오류: {str(e)}")
        
        # 2. 투자자별 거래실적 API 테스트
        logger.info("\n2. 투자자별 거래실적 API 테스트")
        investor_params = {
            'bld': 'dbms/MDC/STAT/standard/MDCSTAT02201',
            'strtDd': start_date_str,
            'endDd': end_date_str,
            'isuCd': ticker,
            'inqTpCd': 2,  # 거래대금
            'trdVolTpCd': 1,  # 순매수
            'askBid': False
        }
        
        logger.info(f"요청 URL: {base_url}")
        logger.info(f"요청 파라미터: {investor_params}")
        
        try:
            resp = requests.post(base_url, data=investor_params, timeout=10)
            logger.info(f"응답 상태 코드: {resp.status_code}")
            logger.info(f"응답 헤더: {dict(resp.headers)}")
            
            # 응답 내용 일부 로깅
            logger.info(f"응답 내용 미리보기: {resp.text[:500]}...")
            
            # JSON 파싱 시도
            try:
                json_data = resp.json()
                logger.info("JSON 파싱 성공!")
                
                # 응답 구조 확인
                if 'OutBlock_1' in json_data:
                    sample_data = json_data['OutBlock_1'][:2]  # 첫 2개 항목만 표시
                    logger.info(f"데이터 샘플: {json.dumps(sample_data, indent=2, ensure_ascii=False)}")
                    logger.info(f"총 거래일 수: {len(json_data['OutBlock_1'])}")
                else:
                    logger.warning(f"예상 데이터 구조가 없음. 키: {list(json_data.keys())}")
            
            except json.JSONDecodeError as e:
                logger.error(f"JSON 파싱 오류: {str(e)}")
                logger.error(f"오류 위치 문자: {resp.text[max(0, e.pos-20):min(len(resp.text), e.pos+20)]}")
                logger.error(f"전체 응답 내용: {resp.text}")
        
        except requests.RequestException as e:
            logger.error(f"HTTP 요청 오류: {str(e)}")
        
        # 3. 문제가 있는 종목으로 테스트 (030200)
        logger.info("\n3. 문제가 발생했던 종목(030200)으로 테스트")
        problem_ticker = "030200"
        
        investor_params = {
            'bld': 'dbms/MDC/STAT/standard/MDCSTAT02201',
            'strtDd': start_date_str,
            'endDd': end_date_str,
            'isuCd': problem_ticker,
            'inqTpCd': 2,  # 거래대금
            'trdVolTpCd': 1,  # 순매수
            'askBid': False
        }
        
        logger.info(f"요청 URL: {base_url}")
        logger.info(f"요청 파라미터: {investor_params}")
        
        try:
            resp = requests.post(base_url, data=investor_params, timeout=10)
            logger.info(f"응답 상태 코드: {resp.status_code}")
            logger.info(f"응답 헤더: {dict(resp.headers)}")
            
            # 응답 내용 일부 로깅
            logger.info(f"응답 내용 미리보기: {resp.text[:500]}...")
            
            # JSON 파싱 시도
            try:
                json_data = resp.json()
                logger.info("JSON 파싱 성공!")
                
                # 응답 구조 확인
                if 'OutBlock_1' in json_data:
                    sample_data = json_data['OutBlock_1'][:2]  # 첫 2개 항목만 표시
                    logger.info(f"데이터 샘플: {json.dumps(sample_data, indent=2, ensure_ascii=False)}")
                    logger.info(f"총 거래일 수: {len(json_data['OutBlock_1'])}")
                else:
                    logger.warning(f"예상 데이터 구조가 없음. 키: {list(json_data.keys())}")
            
            except json.JSONDecodeError as e:
                logger.error(f"JSON 파싱 오류: {str(e)}")
                logger.error(f"오류 위치 문자: {resp.text[max(0, e.pos-20):min(len(resp.text), e.pos+20)]}")
                logger.error(f"전체 응답 내용: {resp.text}")
        
        except requests.RequestException as e:
            logger.error(f"HTTP 요청 오류: {str(e)}")
    
    except Exception as e:
        logger.error(f"테스트 중 예상치 못한 오류 발생: {str(e)}")
        logger.error(traceback.format_exc())
    
    logger.info("===== KRX API 직접 요청 테스트 종료 =====")


def test_pykrx_api():
    """pykrx 라이브러리를 사용한 API 테스트"""
    logger.info("\n===== pykrx 라이브러리 테스트 시작 =====")
    
    try:
        # pykrx 라이브러리 가져오기 (필요 시 설치)
        try:
            from pykrx import stock
        except ImportError:
            logger.error("pykrx 라이브러리가 설치되어 있지 않습니다.")
            logger.info("pip install pykrx 명령어로 설치할 수 있습니다.")
            return
        
        # 날짜 범위 설정
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        
        end_date_str = end_date.strftime("%Y%m%d")
        start_date_str = start_date.strftime("%Y%m%d")
        
        # 1. 정상 종목 테스트 (삼성전자)
        ticker = "005930"
        logger.info(f"1. 정상 종목 테스트 (삼성전자): {ticker}")
        
        try:
            logger.info(f"삼성전자 투자자 데이터 요청 시작: {start_date_str} ~ {end_date_str}")
            df = stock.get_market_trading_value_by_date(
                fromdate=start_date_str,
                todate=end_date_str,
                ticker=ticker
            )
            
            if df is not None and not df.empty:
                logger.info("데이터 조회 성공!")
                logger.info(f"데이터 행 수: {len(df)}")
                logger.info(f"데이터 컬럼: {df.columns.tolist()}")
                logger.info(f"데이터 미리보기:\n{df.head(2)}")
            else:
                logger.warning("데이터가 비어있거나 None입니다.")
        
        except Exception as e:
            logger.error(f"삼성전자 데이터 조회 오류: {str(e)}")
            logger.error(traceback.format_exc())
        
        # 2. 문제가 발생한 종목 테스트 (030200)
        problem_ticker = "030200"
        logger.info(f"\n2. 문제 종목 테스트: {problem_ticker}")
        
        try:
            logger.info(f"문제 종목 투자자 데이터 요청 시작: {start_date_str} ~ {end_date_str}")
            df = stock.get_market_trading_value_by_date(
                fromdate=start_date_str,
                todate=end_date_str,
                ticker=problem_ticker
            )
            
            if df is not None and not df.empty:
                logger.info("데이터 조회 성공!")
                logger.info(f"데이터 행 수: {len(df)}")
                logger.info(f"데이터 컬럼: {df.columns.tolist()}")
                logger.info(f"데이터 미리보기:\n{df.head(2)}")
            else:
                logger.warning("데이터가 비어있거나 None입니다.")
        
        except Exception as e:
            logger.error(f"문제 종목 데이터 조회 오류: {str(e)}")
            logger.error(traceback.format_exc())
            
        # 3. 종목 코드 체계 검증
        logger.info("\n3. 종목 코드 체계 검증")
        
        # 모든 종목 리스트 가져오기
        try:
            logger.info("KOSPI 종목 리스트 요청")
            kospi_tickers = stock.get_market_ticker_list(market="KOSPI")
            logger.info(f"KOSPI 종목 수: {len(kospi_tickers)}")
            if kospi_tickers:
                logger.info(f"첫 5개 KOSPI 종목: {kospi_tickers[:5]}")
            
            logger.info("\nKOSDAQ 종목 리스트 요청")
            kosdaq_tickers = stock.get_market_ticker_list(market="KOSDAQ")
            logger.info(f"KOSDAQ 종목 수: {len(kosdaq_tickers)}")
            if kosdaq_tickers:
                logger.info(f"첫 5개 KOSDAQ 종목: {kosdaq_tickers[:5]}")
            
            # 문제 종목이 리스트에 있는지 확인
            if problem_ticker in kospi_tickers:
                logger.info(f"문제 종목({problem_ticker})이 KOSPI 목록에 있습니다.")
            elif problem_ticker in kosdaq_tickers:
                logger.info(f"문제 종목({problem_ticker})이 KOSDAQ 목록에 있습니다.")
            else:
                logger.warning(f"문제 종목({problem_ticker})이 현재 상장 종목 목록에 없습니다.")
                
                # 상장폐지 종목인지 확인
                try:
                    logger.info("상장폐지 종목 확인 시도")
                    stock_name = stock.get_market_ticker_name(problem_ticker)
                    logger.info(f"종목명: {stock_name}")
                except Exception as e:
                    logger.warning(f"종목명 조회 실패: {str(e)}")
                    logger.info("이 종목은 상장폐지 되었거나 존재하지 않는 종목 코드일 수 있습니다.")
        
        except Exception as e:
            logger.error(f"종목 리스트 조회 오류: {str(e)}")
            logger.error(traceback.format_exc())
    
    except Exception as e:
        logger.error(f"pykrx 테스트 중 예상치 못한 오류 발생: {str(e)}")
        logger.error(traceback.format_exc())
    
    logger.info("===== pykrx 라이브러리 테스트 종료 =====")


def test_network_connectivity():
    """기본적인 네트워크 연결성 및 DNS 확인"""
    logger.info("\n===== 네트워크 연결성 테스트 시작 =====")
    
    # 1. KRX 도메인 연결 테스트
    krx_domains = [
        "www.krx.co.kr",
        "data.krx.co.kr",
        "kasp.krx.co.kr",
        "ops.krx.co.kr"
    ]
    
    for domain in krx_domains:
        logger.info(f"{domain} 연결 테스트...")
        
        # HTTP 요청 테스트
        try:
            url = f"http://{domain}"
            logger.info(f"GET 요청: {url}")
            resp = requests.get(url, timeout=10)
            logger.info(f"응답 상태 코드: {resp.status_code}")
            logger.info(f"응답 헤더: {dict(resp.headers)}")
        except requests.RequestException as e:
            logger.error(f"HTTP 연결 오류: {str(e)}")
        
        # DNS 확인 테스트
        try:
            import socket
            logger.info(f"{domain} DNS 조회 중...")
            ip_address = socket.gethostbyname(domain)
            logger.info(f"IP 주소: {ip_address}")
        except socket.gaierror as e:
            logger.error(f"DNS 조회 오류: {str(e)}")
    
    logger.info("===== 네트워크 연결성 테스트 종료 =====")


def main():
    """KRX API 테스트 메인 함수"""
    logger.info("===== KRX API 테스트 프로그램 시작 =====")
    logger.info(f"시작 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 네트워크 연결성 테스트
    test_network_connectivity()
    
    # 직접 API 요청 테스트
    test_krx_api_direct()
    
    # pykrx 라이브러리 테스트
    test_pykrx_api()
    
    logger.info(f"종료 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("===== KRX API 테스트 프로그램 종료 =====")


if __name__ == "__main__":
    main()
