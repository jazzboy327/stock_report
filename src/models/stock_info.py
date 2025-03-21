# src/models/stock_info.py

from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime
import logging
import yfinance as yf
import asyncio
import streamlit as st

logger = logging.getLogger(__name__)

@dataclass
class StockInfo:
    """주식 기본 정보를 저장하는 데이터 클래스"""
    symbol: str
    company_name: str
    current_price: float
    open_prices: List[float]
    high_prices: List[float]
    low_prices: List[float]
    closing_prices: List[float]
    dates: List[str]
    volume: int
    market_cap: Optional[float] = None

    def __post_init__(self):
        """기본값 초기화 및 데이터 유효성 검증"""
        # 리스트 필드의 기본값 초기화
        if self.open_prices is None:
            self.open_prices = []
        if self.high_prices is None:
            self.high_prices = []
        if self.low_prices is None:
            self.low_prices = []
        if self.closing_prices is None:
            self.closing_prices = []
        if self.dates is None:
            self.dates = []

        # 데이터 길이 검증
        data_lengths = [
            len(self.open_prices),
            len(self.high_prices),
            len(self.low_prices),
            len(self.closing_prices),
            len(self.dates)
        ]
        if len(set(data_lengths)) > 1:
            raise ValueError("All price and date lists must have the same length")

    @property
    def price_change_percent(self) -> float:
        """가격 변동률을 계산하여 반환"""
        if len(self.closing_prices) >= 2:
            return ((self.closing_prices[-1] - self.closing_prices[0]) /
                    self.closing_prices[0] * 100)
        return 0.0

class StockController:
    """주식 분석 컨트롤러"""

    def __init__(self):
        """컨트롤러 초기화"""
        pass

    async def get_stock_data(self, symbol: str) -> Tuple[StockInfo, Dict[str, Any]]:
        """
        주식 기본 데이터를 가져옵니다.
        """
        try:
            # yfinance를 비동기로 실행
            stock_data = await asyncio.to_thread(yf.Ticker, symbol)
            info = await asyncio.to_thread(lambda: stock_data.info)

            # StockInfo 객체 생성
            stock_info = StockInfo(
                symbol=symbol,
                company_name=info.get("longName", "N/A"),
                current_price=info.get("currentPrice", 0.0),
                open_prices=[],  # 필요한 경우 히스토리 데이터에서 채움
                high_prices=[],
                low_prices=[],
                closing_prices=[],
                dates=[],
                volume=info.get("volume", 0),
                market_cap=info.get("marketCap", None)
            )

            # 시장 데이터 구성
            market_data = {
                "sector": info.get("sector", "N/A"),
                "industry": info.get("industry", "N/A"),
                "pe_ratio": info.get("trailingPE", 0),
                "52_week_high": info.get("fiftyTwoWeekHigh", 0),
                "52_week_low": info.get("fiftyTwoWeekLow", 0),
                "dividend_yield": info.get("dividendYield", 0),
                "beta": info.get("beta", 0),
            }

            return stock_info, market_data

        except Exception as e:
            logger.error(f"Error fetching stock data: {str(e)}")
            raise
