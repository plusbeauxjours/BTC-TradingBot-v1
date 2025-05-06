import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Binance API Configuration
BINANCE_API_KEY = os.getenv('BINANCE_API_KEY')
BINANCE_API_SECRET = os.getenv('BINANCE_API_SECRET')

# Trading Configuration
SYMBOL = "BTCUSDT"
MAIN_INTERVAL = "1h"
TREND_INTERVAL = "4h"
POSITION_SIZE = 0.001
MAX_POSITION_SIZE = 0.003

# Volatility-based Configuration
ATR_PERIOD = 14
ATR_MULTIPLIER = 1.5
MIN_ATR_THRESHOLD = 50
MAX_ATR_THRESHOLD = 1500
VOLATILITY_WINDOW = 24

# Dynamic Risk Management
BASE_STOP_LOSS = 0.015  # 1.5%로 증가
BASE_TAKE_PROFIT = 0.045  # 4.5%로 증가
TRAILING_STOP_ACTIVATION = 0.02  # 2%로 증가
TRAILING_STOP_MULTIPLIER = 0.7

# Technical Indicators Configuration
# MACD
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
MACD_THRESHOLD = 0.00005  # MACD 임계값 증가

# RSI
RSI_PERIOD = 14
RSI_OVERSOLD = 30  # RSI 과매도 기준 강화
RSI_OVERBOUGHT = 70  # RSI 과매수 기준 강화
RSI_TREND_PERIOD = 100

# Bollinger Bands
BB_PERIOD = 20
BB_STD = 2.0

# Moving Averages for Trend
MA_FAST = 10
MA_SLOW = 30
MA_TREND = 50
TREND_THRESHOLD = 0.001  # 추세 임계값 증가
MIN_TREND_STRENGTH = 0.001  # 최소 추세 강도 증가
MAX_TREND_STRENGTH = 0.05

# Volume Analysis
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5  # 거래량 임계값 증가
VOLUME_TREND_PERIOD = 50
MIN_VOLUME_INCREASE = 1.5  # 최소 거래량 증가율 증가

# Risk Management
MAX_DAILY_TRADES = 1  # 일일 최대 거래 횟수 제한
MAX_DAILY_DRAWDOWN = 0.02
RISK_PER_TRADE = 0.01
MAX_OPEN_TRADES = 1

# Consecutive Candles
MIN_CONSEC_CANDLES = 3  # 최소 연속 캔들 수 증가
MAX_CONSEC_CANDLES = 5

# Trend Following
TREND_CONFIRMATION_PERIOD = 3  # 추세 확인 기간
MIN_TREND_CANDLES = 2  # 최소 추세 캔들 수

def validate_api_keys():
    if not BINANCE_API_KEY or not BINANCE_API_SECRET:
        print("경고: Binance API 키가 설정되지 않았습니다. .env 파일을 확인하세요.")
        return False
    return True 