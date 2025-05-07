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

# Volatility-based Configuration - 변동성 기반 필터링
ATR_PERIOD = 14
ATR_MULTIPLIER = 1.2  # 변동성 필터 완화 (1.4 → 1.2) - 더 많은 거래 기회
MIN_ATR_THRESHOLD = 35  # 최소 변동성 임계값 완화 (45 → 35) - 낮은 변동성에서도 거래
MAX_ATR_THRESHOLD = 2000  # 최대 변동성 임계값 증가 (1600 → 2000) - 더 넓은 범위의 변동성 허용
VOLATILITY_WINDOW = 24

# Dynamic Risk Management - 승률 70% 이상 목표
BASE_STOP_LOSS = 0.048  # ⚙️ 손절 확대 (4.3% → 4.8%) - ⬆️ 높이면: 손실 폭 증가, 거래 빈도 감소, 승률 향상 / ⬇️ 낮추면: 손실 제한, 거래 빈도 증가, 승률 하락
BASE_TAKE_PROFIT = 0.018  # ⚙️ 익절 낮춤 (2.0% → 1.8%) - ⬆️ 높이면: 수익 폭 증가, 승률 하락, 홀딩 기간 증가 / ⬇️ 낮추면: 수익 제한, 승률 향상, 거래 빈도 증가
TRAILING_STOP_ACTIVATION = 0.0015  # ⚙️ 트레일링 활성화 낮춤 (0.2% → 0.15%) - ⬆️ 높이면: 수익 실현 지연, 수익 변동성 증가 / ⬇️ 낮추면: 빠른 수익 실현, 최대 수익 제한
TRAILING_STOP_MULTIPLIER = 0.12  # ⚙️ 트레일링 배수 낮춤 (0.2 → 0.12) - ⬆️ 높이면: 수익 보존 증가, 리스크 감소 / ⬇️ 낮추면: 최대 수익 추구, 변동성 허용
STOP_LOSS_PCT = 0.048
TAKE_PROFIT_PCT = 0.018
TRAILING_STOP_PCT = 0.01

# Technical Indicators Configuration - 기술적 지표 설정
# MACD - 이동평균수렴확산지수
MACD_FAST = 3  # 단기 EMA 기간 단축 (4 → 3) - 빠른 반응성
MACD_SLOW = 8  # 장기 EMA 기간 단축 (10 → 8) - 중단기 트렌드에 더 민감
MACD_SIGNAL = 3  # 시그널 기간 단축 (4 → 3) - 더 민감한 신호
MACD_THRESHOLD = 0.000005  # 낮은 임계값 완화 (0.000008 → 0.000005) - 더 많은 신호 감지

# RSI - 상대강도지수
RSI_PERIOD = 14
RSI_OVERSOLD = 47  # ⚙️ 과매도 기준 완화 (45 → 47) - ⬆️ 높이면: 매수 신호 증가, 거래 빈도 증가, 정확도 하락 / ⬇️ 낮추면: 매수 신호 감소, 거래 빈도 감소, 정확도 향상
RSI_OVERBOUGHT = 53  # ⚙️ 과매수 기준 강화 (55 → 53) - ⬆️ 높이면: 매도 신호 감소, 홀딩 기간 증가 / ⬇️ 낮추면: 매도 신호 증가, 수익 실현 빨라짐, 최대 수익 제한
RSI_TREND_PERIOD = 20  # 트렌드 기간 단축 (22 → 20) - 더 민감한 트렌드 감지

# Bollinger Bands - 볼린저 밴드
BB_PERIOD = 20
BB_STD = 1.3  # ⚙️ 표준편차 감소 (1.4 → 1.3) - ⬆️ 높이면: 신호 감소, 정확도 향상, 거래 빈도 감소 / ⬇️ 낮추면: 신호 증가, 거래 빈도 증가, 오신호 위험 증가

# Moving Averages for Trend - 이동평균선 지표
MA_FAST = 2  # ⚙️ 단기 이평선 단축 (3 → 2) - ⬆️ 높이면: 신호 지연, 노이즈 감소, 거래 빈도 감소 / ⬇️ 낮추면: 반응 속도 증가, 노이즈 증가, 거래 빈도 증가
MA_SLOW = 6  # ⚙️ 중기 이평선 단축 (7 → 6) - ⬆️ 높이면: 안정적 추세 확인, 신호 지연, 거래 타이밍 놓침 / ⬇️ 낮추면: 추세 변화 빠르게 감지, 오신호 증가
MA_TREND = 12  # ⚙️ 장기 이평선 단축 (15 → 12) - ⬆️ 높이면: 장기 추세 확인, 안정성 증가, 거래 기회 감소 / ⬇️ 낮추면: 중기 추세 포착, 변동성 증가
TREND_THRESHOLD = 0.00005  # ⚙️ 추세 감지 임계값 완화 (0.0001 → 0.00005) - ⬆️ 높이면: 강한 추세만 포착, 거래 빈도 감소, 정확도 향상 / ⬇️ 낮추면: 약한 추세도 포착, 거래 빈도 증가, 오신호 위험
MIN_TREND_STRENGTH = 0.00005  # ⚙️ 최소 추세 강도 완화 (0.0001 → 0.00005) - ⬆️ 높이면: 강한 추세만 인식, 거래 감소, 승률 향상 / ⬇️ 낮추면: 약한 추세도 인식, 거래 증가, 승률 하락 가능
MAX_TREND_STRENGTH = 0.1  # 강한 추세 허용 상한 확대 (0.09 → 0.1) - 강한 추세에서도 거래

# Volume Analysis - 거래량 분석
VOLUME_MA_PERIOD = 8  # 거래량 이동평균 기간 단축 (10 → 8) - 최근 거래량에 더 민감하게 반응
VOLUME_THRESHOLD = 1.005  # ⚙️ 거래량 임계값 완화 (1.01 → 1.005) - ⬆️ 높이면: 강한 거래량 신호만 포착, 거래 감소, 신뢰도 향상 / ⬇️ 낮추면: 약한 거래량도 신호로 인식, 거래 증가
VOLUME_TREND_PERIOD = 15  # 거래량 트렌드 기간 단축 (18 → 15) - 최근 추세에 더 민감하게 반응
MIN_VOLUME_INCREASE = 1.003  # ⚙️ 최소 거래량 증가율 완화 (1.005 → 1.003) - ⬆️ 높이면: 확실한 거래량 증가만 인식, 거래 감소 / ⬇️ 낮추면: 작은 거래량 변화도 인식, 거래 증가

# Risk Management - 리스크 관리
MAX_DAILY_TRADES = 10  # 일일 최대 거래 횟수 증가 (8 → 10) - 더 많은 거래 기회 포착
MAX_DAILY_DRAWDOWN = 0.1  # 일일 최대 손실 한도 완화 (0.09 → 0.1) - 손실 허용 범위 확대
RISK_PER_TRADE = 0.01
MAX_OPEN_TRADES = 1

# Consecutive Candles - 연속 캔들 패턴
MIN_CONSEC_CANDLES = 1  # 단일 캔들도 패턴으로 인정
MAX_CONSEC_CANDLES = 12  # 최대 연속 캔들 허용 확대 (10 → 12) - 더 긴 패턴도 인식

# Trend Following - 추세 추종
TREND_CONFIRMATION_PERIOD = 1  # 추세 확인 기간 단축 (2 → 1) - 빠른 추세 확인
MIN_TREND_CANDLES = 1  # 추세 캔들 수 최소화 (2 → 1) - 더 민감한 추세 감지

def validate_api_keys():
    if not BINANCE_API_KEY or not BINANCE_API_SECRET:
        print("경고: Binance API 키가 설정되지 않았습니다. .env 파일을 확인하세요.")
        return False
    return True 