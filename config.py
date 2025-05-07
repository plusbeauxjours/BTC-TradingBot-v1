import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Binance API Configuration
BINANCE_API_KEY = os.getenv('BINANCE_API_KEY')
BINANCE_API_SECRET = os.getenv('BINANCE_API_SECRET')

# Trading Configuration
SYMBOL = "BTCUSDT"
MAIN_INTERVAL = "1h"  # 1시간 봉 유지
TREND_INTERVAL = "4h"  # 4시간 봉으로 추세 확인
POSITION_SIZE = 0.001
MAX_POSITION_SIZE = 0.003
VOLATILITY_ADJUSTMENT = True
MAX_VOLATILITY_MULTIPLIER = 1.5

# Volatility-based Configuration
ATR_PERIOD = 14
ATR_MULTIPLIER = 1.2
MIN_ATR_THRESHOLD = 30  # 낮은 변동성도 허용
MAX_ATR_THRESHOLD = 2000  # 높은 변동성도 허용
VOLATILITY_WINDOW = 24

# Dynamic Risk Management - 승률 70% 이상 목표
BASE_STOP_LOSS = 0.015  # 1.5% 손절
BASE_TAKE_PROFIT = 0.055  # 5.5% 익절로 수익 확대
TRAILING_STOP_ACTIVATION = 0.015  # 1.5%부터 트레일링 활성화
TRAILING_STOP_MULTIPLIER = 0.6  # 공격적 트레일링
STOP_LOSS_PCT = 0.015
TAKE_PROFIT_PCT = 0.055
TRAILING_STOP_PCT = 0.01

# Technical Indicators Configuration
# MACD
MACD_FAST = 8  # 단기 EMA 기간 단축 (12→8)
MACD_SLOW = 21  # 장기 EMA 기간 단축 (26→21)
MACD_SIGNAL = 9
MACD_THRESHOLD = 0.00005  # 낮은 값도 신호로 인식

# RSI
RSI_PERIOD = 14
RSI_OVERSOLD = 38  # 과매도 기준 완화 (32→38)
RSI_OVERBOUGHT = 62  # 과매수 기준 완화 (68→62)
RSI_TREND_PERIOD = 50  # 50으로 단축

# Bollinger Bands
BB_PERIOD = 20
BB_STD = 1.8  # 밴드 폭 좁힘 (2.0→1.8)

# Moving Averages for Trend
MA_FAST = 5  # 매우 단기 이평선 (9→5)
MA_SLOW = 13  # 중기 이평선 (25→13)
MA_TREND = 34  # 장기 이평선 (50→34)
TREND_THRESHOLD = 0.0008  # 작은 추세도 포착 (0.0015→0.0008)
MIN_TREND_STRENGTH = 0.0008  # 약한 추세도 인식 (0.0015→0.0008)
MAX_TREND_STRENGTH = 0.05  # 강한 추세까지 허용 (0.04→0.05)

# Volume Analysis
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.2  # 평균보다 약간만 높아도 OK (1.75→1.2)
VOLUME_TREND_PERIOD = 50
MIN_VOLUME_INCREASE = 1.1  # 최소 거래량 증가 기준 완화 (1.6→1.1)

# Risk Management
MAX_DAILY_TRADES = 3  # 일일 최대 거래 횟수 증가 (1→3)
MAX_DAILY_DRAWDOWN = 0.05  # 일일 최대 손실 한도 완화 (0.02→0.05)
RISK_PER_TRADE = 0.01
MAX_OPEN_TRADES = 1

# Consecutive Candles
MIN_CONSEC_CANDLES = 1  # 단일 캔들도 패턴으로 인정 (2→1)
MAX_CONSEC_CANDLES = 5  # 더 많은 연속 캔들 허용 (4→5)

# Trend Following
TREND_CONFIRMATION_PERIOD = 3  # 추세 확인 기간 단축 (4→3)
MIN_TREND_CANDLES = 1  # 추세 캔들 수 최소화 (2→1)

def validate_api_keys():
    if not BINANCE_API_KEY or not BINANCE_API_SECRET:
        print("경고: Binance API 키가 설정되지 않았습니다. .env 파일을 확인하세요.")
        return False
    return True 