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
BASE_STOP_LOSS = 0.015  # ⚙️ 1.5% 손절 - ⬆️ 높이면: 손실 폭 증가, 거래 빈도 감소, 승률 향상 / ⬇️ 낮추면: 손실 제한, 거래 빈도 증가, 승률 하락
BASE_TAKE_PROFIT = 0.055  # ⚙️ 5.5% 익절 - ⬆️ 높이면: 수익 폭 증가, 승률 하락, 홀딩 기간 증가 / ⬇️ 낮추면: 수익 제한, 승률 향상, 거래 빈도 증가
TRAILING_STOP_ACTIVATION = 0.015  # ⚙️ 1.5% 트레일링 활성화 - ⬆️ 높이면: 수익 실현 지연, 수익 변동성 증가 / ⬇️ 낮추면: 빠른 수익 실현, 최대 수익 제한
TRAILING_STOP_MULTIPLIER = 0.6  # ⚙️ 트레일링 배수 - ⬆️ 높이면: 수익 보존 증가, 리스크 감소 / ⬇️ 낮추면: 최대 수익 추구, 변동성 허용
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
RSI_OVERSOLD = 30  # ⚙️ 과매도 기준 - ⬆️ 높이면: 매수 신호 증가, 거래 빈도 증가, 정확도 하락 / ⬇️ 낮추면: 매수 신호 감소, 거래 빈도 감소, 정확도 향상
RSI_OVERBOUGHT = 70  # ⚙️ 과매수 기준 - ⬆️ 높이면: 매도 신호 감소, 홀딩 기간 증가 / ⬇️ 낮추면: 매도 신호 증가, 수익 실현 빨라짐, 최대 수익 제한
RSI_TREND_PERIOD = 50  # 50으로 단축

# Bollinger Bands
BB_PERIOD = 20
BB_STD = 2.0  # ⚙️ 표준편차 - ⬆️ 높이면: 신호 감소, 정확도 향상, 거래 빈도 감소 / ⬇️ 낮추면: 신호 증가, 거래 빈도 증가, 오신호 위험 증가

# Moving Averages for Trend
MA_FAST = 5  # ⚙️ 단기 이평선 - ⬆️ 높이면: 신호 지연, 노이즈 감소, 거래 빈도 감소 / ⬇️ 낮추면: 반응 속도 증가, 노이즈 증가, 거래 빈도 증가
MA_SLOW = 13  # ⚙️ 중기 이평선 - ⬆️ 높이면: 안정적 추세 확인, 신호 지연, 거래 타이밍 놓침 / ⬇️ 낮추면: 추세 변화 빠르게 감지, 오신호 증가
MA_TREND = 34  # ⚙️ 장기 이평선 - ⬆️ 높이면: 장기 추세 확인, 안정성 증가, 거래 기회 감소 / ⬇️ 낮추면: 중기 추세 포착, 변동성 증가
TREND_THRESHOLD = 0.0008  # ⚙️ 추세 감지 임계값 - ⬆️ 높이면: 강한 추세만 포착, 거래 빈도 감소, 정확도 향상 / ⬇️ 낮추면: 약한 추세도 포착, 거래 빈도 증가, 오신호 위험
MIN_TREND_STRENGTH = 0.0008  # ⚙️ 최소 추세 강도 - ⬆️ 높이면: 강한 추세만 인식, 거래 감소, 승률 향상 / ⬇️ 낮추면: 약한 추세도 인식, 거래 증가, 승률 하락 가능
MAX_TREND_STRENGTH = 0.05  # 강한 추세까지 허용 (0.04→0.05)

# Volume Analysis
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.2  # ⚙️ 거래량 임계값 - ⬆️ 높이면: 강한 거래량 신호만 포착, 거래 감소, 신뢰도 향상 / ⬇️ 낮추면: 약한 거래량도 신호로 인식, 거래 증가
VOLUME_TREND_PERIOD = 50
MIN_VOLUME_INCREASE = 1.1  # ⚙️ 최소 거래량 증가율 - ⬆️ 높이면: 확실한 거래량 증가만 인식, 거래 감소 / ⬇️ 낮추면: 작은 거래량 변화도 인식, 거래 증가

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