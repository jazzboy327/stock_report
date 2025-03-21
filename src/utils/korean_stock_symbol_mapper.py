# src/utils/korean_stock_symbol_mapper.py

import logging
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Set
import FinanceDataReader as fdr
from datetime import datetime, timedelta
import json
import threading

class KoreanStockSymbolMapper:
    """한국 주식 종목 코드 매핑 클래스"""

    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not hasattr(self, 'logger'):
            # 로거 설정
            self.logger = logging.getLogger('StockAgent.KoreanStockSymbolMapper')
            
            self._symbols: Dict[str, str] = {}
            self._companies: Dict[str, str] = {}
            self._markets: Dict[str, str] = {}
            
            self._load_symbols()

    def _load_symbols(self):
        """종목 정보 로드"""
        try:
            # 데이터 파일 경로 설정
            data_path = Path(__file__).resolve().parent.parent.parent / 'data'
            symbols_file = data_path / 'stock_symbols.csv'
            last_update_file = data_path / 'last_update.json'
            
            # 데이터 디렉토리 생성
            data_path.mkdir(parents=True, exist_ok=True)
            
            update_needed = True
            
            # 마지막 업데이트 확인
            if last_update_file.exists() and symbols_file.exists():
                with open(last_update_file, 'r') as f:
                    last_update = datetime.fromisoformat(json.load(f)['last_update'])
                    today = datetime.now()
                    update_needed = (today - last_update).days >= 1
            
            if update_needed:
                self.logger.info("Updating stock symbols from KRX...")
                self._download_stock_symbols(symbols_file, last_update_file)
            
            # CSV 파일 읽기
            df = pd.read_csv(symbols_file)
            
            # 데이터 매핑
            for _, row in df.iterrows():
                company_name = row['Name']  # FinanceDataReader의 컬럼명
                symbol = row['Symbol']
                market = row['Market']
                
                self._symbols[company_name] = symbol
                self._companies[symbol] = company_name
                self._markets[company_name] = market
            
            self.logger.info(f"Successfully loaded {len(self._symbols)} companies")
            
        except Exception as e:
            self.logger.error(f"Error loading stock symbols: {str(e)}", exc_info=True)
            self._create_backup_data()

    def _download_stock_symbols(self, symbols_file: Path, last_update_file: Path):
        """KRX에서 종목 정보 다운로드"""
        try:
            # KOSPI 종목 목록
            kospi = fdr.StockListing('KOSPI')
            self.logger.debug(f"KOSPI columns: {kospi.columns.tolist()}")  # 실제 컬럼명 확인
            kospi['Market'] = 'KOSPI'
            
            # KOSDAQ 종목 목록
            kosdaq = fdr.StockListing('KOSDAQ')
            self.logger.debug(f"KOSDAQ columns: {kosdaq.columns.tolist()}")  # 실제 컬럼명 확인
            kosdaq['Market'] = 'KOSDAQ'
            
            # 데이터 합치기
            df = pd.concat([kospi, kosdaq])
            self.logger.debug(f"Combined columns: {df.columns.tolist()}")  # 결합된 데이터프레임의 컬럼명 확인
            
            # 컬럼명 매핑 (FinanceDataReader의 실제 컬럼명으로 수정)
            column_mapping = {
                'Code': 'Symbol',    # 종목 코드
                'Name': 'Name',      # 종목명
                'Market': 'Market'   # 시장 구분
            }
            
            # 필요한 컬럼 선택 및 이름 변경
            df = df[list(column_mapping.keys())].rename(columns=column_mapping)
            
            # Symbol 컬럼에 시장 구분자 추가
            df['Symbol'] = df.apply(
                lambda x: f"{x['Symbol']}.{'KS' if x['Market']=='KOSPI' else 'KQ'}", 
                axis=1
            )
            
            # CSV 파일로 저장
            df.to_csv(symbols_file, index=False, encoding='utf-8')
            
            # 마지막 업데이트 시간 저장
            with open(last_update_file, 'w') as f:
                json.dump({'last_update': datetime.now().isoformat()}, f)
            
            self.logger.info(f"Downloaded and saved {len(df)} companies")
            
        except Exception as e:
            self.logger.error(f"Error downloading stock symbols: {str(e)}", exc_info=True)
            raise  # 에러를 상위로 전파하여 백업 데이터 사용하도록 함

    def _create_backup_data(self):
        """백업 데이터 생성"""
        sample_data = {
            '삼성전자': ('005930.KS', 'KOSPI'),
            'SK하이닉스': ('000660.KS', 'KOSPI'),
            '네이버': ('035420.KS', 'KOSPI'),
            '카카오': ('035720.KS', 'KOSPI'),
            'LG에너지솔루션': ('373220.KS', 'KOSPI'),
            '현대차': ('005380.KS', 'KOSPI'),
            'POSCO홀딩스': ('005490.KS', 'KOSPI'),
            '삼성바이오로직스': ('207940.KS', 'KOSPI'),
            'LG화학': ('051910.KS', 'KOSPI'),
            '삼성SDI': ('006400.KS', 'KOSPI'),
            # KOSDAQ 샘플
            '셀트리온헬스케어': ('091990.KQ', 'KOSDAQ'),
            '에코프로': ('086520.KQ', 'KOSDAQ'),
            'SK바이오팜': ('326030.KQ', 'KOSDAQ'),
            '펄어비스': ('263750.KQ', 'KOSDAQ'),
            '카카오게임즈': ('293490.KQ', 'KOSDAQ')
        }
        
        for company, (symbol, market) in sample_data.items():
            self._symbols[company] = symbol
            self._companies[symbol] = company
            self._markets[company] = market
        
        self.logger.warning("Using backup data due to download failure")

    def get_symbol(self, company_name: str) -> Optional[str]:
        """종목명으로 종목 코드 조회"""
        try:
            symbol = self._symbols.get(company_name)
            if not symbol:
                self.logger.warning(f"Symbol not found for company: {company_name}")
            return symbol
        except Exception as e:
            self.logger.error(f"Error getting symbol: {str(e)}", exc_info=True)
            return None

    def get_company_name(self, symbol: str) -> Optional[str]:
        """종목 코드로 종목명 조회"""
        try:
            company_name = self._companies.get(symbol)
            if not company_name:
                self.logger.warning(f"Company name not found for symbol: {symbol}")
            return company_name
        except Exception as e:
            self.logger.error(f"Error getting company name: {str(e)}", exc_info=True)
            return None

    def get_market(self, company_name: str) -> Optional[str]:
        """종목명으로 시장 구분 조회"""
        try:
            market = self._markets.get(company_name)
            if not market:
                self.logger.warning(f"Market not found for company: {company_name}")
            return market
        except Exception as e:
            self.logger.error(f"Error getting market: {str(e)}", exc_info=True)
            return None

    def get_all_companies(self) -> Dict[str, str]:
        """모든 종목 정보 조회"""
        return self._symbols.copy()

    def search_companies(self, keyword: str) -> List[Tuple[str, str, str]]:
        """회사명으로 종목 검색 (모든 일치하는 결과 반환)
        Returns:
            List[Tuple[str, str, str]]: [(회사명, 종목코드, 시장구분)] 형태의 리스트
        """
        try:
            result = []
            keyword = keyword.lower()
            
            for company, symbol in self._symbols.items():
                if keyword in company.lower():
                    market = self._markets.get(company, "Unknown")
                    result.append((company, symbol, market))
            
            # 회사명 기준으로 정렬
            result.sort(key=lambda x: x[0])
            return result
        except Exception as e:
            self.logger.error(f"Error searching companies: {str(e)}", exc_info=True)
            return [] 