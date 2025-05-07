import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from binance.client import Client
from config import *

# Binance 클라이언트 초기화
client = None

# 백테스트 설정
START_DATE = "1 Jan, 2023"  # 백테스트 시작일
END_DATE = "1 Jul, 2023"    # 백테스트 종료일
INITIAL_BALANCE = 10000     # 초기 잔고 (USDT)

# 지표 계산 함수
def calculate_indicators(df):
    # Moving Averages
    df['ma_fast'] = df['close'].rolling(window=MA_FAST).mean()
    df['ma_slow'] = df['close'].rolling(window=MA_SLOW).mean()
    df['ma_trend'] = df['close'].rolling(window=MA_TREND).mean()
    
    # MACD
    df['ema_fast'] = df['close'].ewm(span=MACD_FAST, adjust=False).mean()
    df['ema_slow'] = df['close'].ewm(span=MACD_SLOW, adjust=False).mean()
    df['macd'] = df['ema_fast'] - df['ema_slow']
    df['macd_signal'] = df['macd'].ewm(span=MACD_SIGNAL, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']

    # RSI
    delta = df['close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=RSI_PERIOD).mean()
    avg_loss = loss.rolling(window=RSI_PERIOD).mean()
    rs = avg_gain / avg_loss
    df['rsi'] = 100 - (100 / (1 + rs))

    # Bollinger Bands
    df['bb_middle'] = df['close'].rolling(window=BB_PERIOD).mean()
    df['bb_std'] = df['close'].rolling(window=BB_PERIOD).std()
    df['bb_upper'] = df['bb_middle'] + (df['bb_std'] * BB_STD)
    df['bb_lower'] = df['bb_middle'] - (df['bb_std'] * BB_STD)
    
    # Volume Analysis
    df['volume_ma'] = df['volume'].rolling(window=VOLUME_MA_PERIOD).mean()
    df['volume_ratio'] = df['volume'] / df['volume_ma']
    
    # 추가 지표 계산
    df['price_change'] = df['close'].pct_change()
    df['volume_change'] = df['volume'].pct_change()
    df['avg_volume'] = df['volume'].rolling(window=VOLUME_MA_PERIOD).mean()
    
    # Trend Analysis
    df['trend'] = (df['ma_fast'] - df['ma_slow']) / df['ma_slow']
    df['trend_strength'] = abs(df['trend'])
    
    # 연속 상승/하락 캔들 수 계산
    df['price_change'] = df['close'] - df['open']
    df['consec_up'] = 0
    df['consec_down'] = 0
    
    for i in range(1, len(df)):
        if df['price_change'].iloc[i] > 0:
            df.loc[df.index[i], 'consec_up'] = df['consec_up'].iloc[i-1] + 1
            df.loc[df.index[i], 'consec_down'] = 0
        elif df['price_change'].iloc[i] < 0:
            df.loc[df.index[i], 'consec_down'] = df['consec_down'].iloc[i-1] + 1
            df.loc[df.index[i], 'consec_up'] = 0
    
    return df

# 과거 데이터 가져오기
def fetch_historical_data(symbol, interval, start_date, end_date):
    # 날짜를 밀리초로 변환
    start_ts = int(datetime.strptime(start_date, "%d %b, %Y").timestamp() * 1000)
    end_ts = int(datetime.strptime(end_date, "%d %b, %Y").timestamp() * 1000)
    
    print("과거 데이터 다운로드 중...")
    
    # 데이터 가져오기
    klines = client.get_historical_klines(
        symbol=symbol,
        interval=interval,
        start_str=start_date,
        end_str=end_date
    )
    
    # 데이터프레임으로 변환
    data = pd.DataFrame(klines, columns=[
        'open_time', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'qav', 'num_trades', 'taker_base', 'taker_quote', 'ignore'
    ])
    
    # 데이터 형변환
    data['open_time'] = pd.to_datetime(data['open_time'], unit='ms')
    data['close_time'] = pd.to_datetime(data['close_time'], unit='ms')
    
    # 인덱스 설정
    data.set_index('open_time', inplace=True)
    
    for col in ['open', 'high', 'low', 'close', 'volume']:
        data[col] = data[col].astype(float)
    
    print(f"다운로드 완료: {len(data)} 개의 {interval} 데이터")
    return data

def calculate_atr(df, period=14):
    """Average True Range (ATR) 계산"""
    high = df['high'].astype(float)
    low = df['low'].astype(float)
    close = df['close'].astype(float)
    
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    
    return atr

def calculate_volatility_metrics(df):
    """변동성 지표 계산"""
    # ATR 계산
    df['atr'] = calculate_atr(df, ATR_PERIOD)
    
    # 변동성 상태 계산
    df['volatility_ratio'] = df['atr'] / df['atr'].rolling(VOLATILITY_WINDOW).mean()
    
    # 변동성 기반 동적 파라미터 계산
    df['dynamic_stop_loss'] = BASE_STOP_LOSS * df['volatility_ratio']
    df['dynamic_take_profit'] = BASE_TAKE_PROFIT * df['volatility_ratio']
    
    # 변동성 상태 분류
    df['volatility_state'] = pd.cut(
        df['volatility_ratio'],
        bins=[0, 0.8, 1.2, float('inf')],
        labels=['low', 'normal', 'high']
    )
    
    return df

def generate_signals(df, df_trend):
    df = df.copy()
    df['signal'] = 0  # 0: 중립, 1: 매수, -1: 매도
    
    # 변동성 지표 계산
    df = calculate_volatility_metrics(df)
    
    # 추세 분석
    df['trend_direction'] = (df['close'] > df['ma_trend']).astype(int)
    df['trend_strength'] = abs(df['close'] - df['ma_trend']) / df['ma_trend']
    
    # 상위 타임프레임 추세 분석
    higher_trend = df_trend['close'] > df_trend['ma_trend']
    higher_trend_strength = abs(df_trend['close'] - df_trend['ma_trend']) / df_trend['ma_trend']
    
    # 볼린저 밴드 위치
    df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
    
    # 캔들스틱 패턴 분석
    df['bullish_engulfing'] = False
    df['bearish_engulfing'] = False
    df['bullish_hammer'] = False
    df['bearish_shooting_star'] = False
    df['bullish_doji'] = False
    df['bearish_doji'] = False
    
    for i in range(1, len(df)):
        # 불리시 인걸핑 패턴 - 완화된 기준
        if (df['close'].iloc[i-1] < df['open'].iloc[i-1] and  # 이전 캔들이 음봉
            df['close'].iloc[i] > df['open'].iloc[i] and      # 현재 캔들이 양봉
            df['close'].iloc[i] > df['close'].iloc[i-1]):     # 종가가 이전 종가보다 높음
            df.loc[df.index[i], 'bullish_engulfing'] = True
            
        # 베어리시 인걸핑 패턴 - 완화된 기준
        if (df['close'].iloc[i-1] > df['open'].iloc[i-1] and  # 이전 캔들이 양봉
            df['close'].iloc[i] < df['open'].iloc[i] and      # 현재 캔들이 음봉
            df['close'].iloc[i] < df['close'].iloc[i-1]):     # 종가가 이전 종가보다 낮음
            df.loc[df.index[i], 'bearish_engulfing'] = True
            
        # 불리시 해머 - 완화된 기준
        body = abs(df['close'].iloc[i] - df['open'].iloc[i])
        lower_wick = min(df['open'].iloc[i], df['close'].iloc[i]) - df['low'].iloc[i]
        upper_wick = df['high'].iloc[i] - max(df['open'].iloc[i], df['close'].iloc[i])
        
        if (df['close'].iloc[i] > df['open'].iloc[i] and      # 양봉
            lower_wick >= body * 1.0 and                      # 아래 꼬리가 몸통 이상 (매우 완화)
            upper_wick < body):                               # 위 꼬리가 몸통보다 작음 (매우 완화)
            df.loc[df.index[i], 'bullish_hammer'] = True
            
        # 베어리시 슈팅스타 - 완화된 기준
        if (df['close'].iloc[i] < df['open'].iloc[i] and      # 음봉
            upper_wick >= body * 1.0 and                      # 위 꼬리가 몸통 이상 (매우 완화)
            lower_wick < body):                               # 아래 꼬리가 몸통보다 작음 (매우 완화)
            df.loc[df.index[i], 'bearish_shooting_star'] = True
            
        # 도지 패턴 추가 (몸통이 작은 캔들)
        candle_range = df['high'].iloc[i] - df['low'].iloc[i]
        if candle_range > 0 and body / candle_range < 0.3:  # 몸통이 전체 범위의 30% 미만
            if df['close'].iloc[i] > df['open'].iloc[i]:  # 양봉 도지
                df.loc[df.index[i], 'bullish_doji'] = True
            else:  # 음봉 도지
                df.loc[df.index[i], 'bearish_doji'] = True
    
    # MACD 교차 분석
    df['macd_cross_up'] = (df['macd'] > df['macd_signal']) & (df['macd'].shift(1) <= df['macd_signal'].shift(1))
    df['macd_cross_down'] = (df['macd'] < df['macd_signal']) & (df['macd'].shift(1) >= df['macd_signal'].shift(1))
    df['macd_trending_up'] = (df['macd'] > df['macd'].shift(1))
    df['macd_trending_down'] = (df['macd'] < df['macd'].shift(1))
    
    # RSI 방향성 분석
    df['rsi_trending_up'] = (df['rsi'] > df['rsi'].shift(1))
    df['rsi_trending_down'] = (df['rsi'] < df['rsi'].shift(1))
    
    # 이동평균선 교차 분석 (추가)
    df['ma_cross_up'] = (df['ma_fast'] > df['ma_slow']) & (df['ma_fast'].shift(1) <= df['ma_slow'].shift(1))
    df['ma_cross_down'] = (df['ma_fast'] < df['ma_slow']) & (df['ma_fast'].shift(1) >= df['ma_slow'].shift(1))
    
    # 연속 캔들 분석
    df['consec_up'] = 0
    df['consec_down'] = 0
    
    for i in range(1, len(df)):
        if df['close'].iloc[i] > df['open'].iloc[i]:  # 양봉
            df.loc[df.index[i], 'consec_up'] = df['consec_up'].iloc[i-1] + 1
            df.loc[df.index[i], 'consec_down'] = 0
        elif df['close'].iloc[i] < df['open'].iloc[i]:  # 음봉
            df.loc[df.index[i], 'consec_down'] = df['consec_down'].iloc[i-1] + 1
            df.loc[df.index[i], 'consec_up'] = 0
    
    # 매수 신호 조건들
    df['macd_condition'] = (
        df['macd_cross_up'] | 
        ((df['macd'] > df['macd_signal']) & df['macd_trending_up'])
    )
    
    df['rsi_condition'] = (
        (df['rsi'] <= RSI_OVERSOLD) | 
        ((df['rsi'] < 45) & df['rsi_trending_up'])
    )
    
    df['bb_condition'] = (
        (df['bb_position'] < 0.3) | 
        (df['close'] < df['bb_lower'])
    )
    
    df['trend_condition'] = (
        (df['trend_direction'] == 1) |
        (df['ma_cross_up']) |
        (df['close'] > df['ma_fast'])
    )
    
    df['volume_condition'] = (
        (df['volume'] > df['avg_volume']) |
        (df['volume_change'] > 0)
    )
    
    df['volatility_condition'] = (df['atr'] > 0)  # 항상 True (제약 조건 제거)
    
    # 패턴 조건 (추가된 패턴)
    df['pattern_condition'] = (
        df['bullish_engulfing'] | 
        df['bullish_hammer'] | 
        df['bullish_doji'] | 
        (df['consec_up'] >= 1)
    )
    
    # 매수 신호 조건 - 매우 완화된 조건
    # 다양한 진입 시나리오 구현
    
    # 시나리오 1: RSI 과매도 반등
    buy_scenario_1 = (
        df['rsi_condition'] &
        df['rsi_trending_up'] &
        ((df['close'] > df['ma_fast']) | (df['ma_cross_up']))
    )
    
    # 시나리오 2: MACD 상승 교차
    buy_scenario_2 = (
        df['macd_condition'] &
        df['trend_condition'] &
        (df['volume'] > df['volume'].shift(1))
    )
    
    # 시나리오 3: 볼린저 밴드 하단 반등
    buy_scenario_3 = (
        ((df['close'] < df['bb_lower']) | (df['bb_position'] < 0.2)) &
        df['rsi_trending_up'] &
        (df['close'] > df['close'].shift(1))
    )
    
    # 시나리오 4: 패턴 기반 진입
    buy_scenario_4 = (
        df['pattern_condition'] &
        (df['close'] > df['ma_fast']) &
        (df['volume'] > df['avg_volume'] * 0.8)
    )
    
    # 시나리오 5: 추세 상승 확인
    buy_scenario_5 = (
        (df['close'] > df['ma_trend']) &
        (df['ma_fast'] > df['ma_fast'].shift(3)) &
        (df['volume'] > df['volume'].shift(1))
    )
    
    # 종합 매수 조건 (시나리오 중 하나라도 충족)
    buy_condition = (
        buy_scenario_1 | 
        buy_scenario_2 | 
        buy_scenario_3 | 
        buy_scenario_4 | 
        buy_scenario_5
    )
    
    # 상위 타임프레임 필터링 (완화: 상승 추세이거나 횡보장일 때만)
    higher_timeframe_filter = (higher_trend | (higher_trend_strength < MIN_TREND_STRENGTH))
    
    # 최종 매수 신호
    df.loc[buy_condition & higher_timeframe_filter, 'signal'] = 1
    
    # 매도 신호 조건들
    df['macd_sell_condition'] = (
        df['macd_cross_down'] | 
        ((df['macd'] < df['macd_signal']) & df['macd_trending_down'])
    )
    
    df['rsi_sell_condition'] = (
        (df['rsi'] >= RSI_OVERBOUGHT) | 
        ((df['rsi'] > 55) & df['rsi_trending_down'])
    )
    
    df['bb_sell_condition'] = (
        (df['bb_position'] > 0.7) | 
        (df['close'] > df['bb_upper'])
    )
    
    df['trend_sell_condition'] = (
        (df['trend_direction'] == 0) |
        (df['ma_cross_down']) |
        (df['close'] < df['ma_fast'])
    )
    
    df['pattern_sell_condition'] = (
        df['bearish_engulfing'] | 
        df['bearish_shooting_star'] | 
        df['bearish_doji'] | 
        (df['consec_down'] >= 1)
    )
    
    # 매도 신호 다양한 시나리오
    
    # 시나리오 1: RSI 과매수 반전
    sell_scenario_1 = (
        df['rsi_sell_condition'] &
        df['rsi_trending_down'] &
        ((df['close'] < df['ma_fast']) | (df['ma_cross_down']))
    )
    
    # 시나리오 2: MACD 하락 교차
    sell_scenario_2 = (
        df['macd_sell_condition'] &
        df['trend_sell_condition'] &
        (df['volume'] > df['volume'].shift(1))
    )
    
    # 시나리오 3: 볼린저 밴드 상단 이탈
    sell_scenario_3 = (
        ((df['close'] > df['bb_upper']) | (df['bb_position'] > 0.8)) &
        df['rsi_trending_down'] &
        (df['close'] < df['close'].shift(1))
    )
    
    # 시나리오 4: 패턴 기반 매도
    sell_scenario_4 = (
        df['pattern_sell_condition'] &
        (df['close'] < df['ma_fast']) &
        (df['volume'] > df['avg_volume'] * 0.8)
    )
    
    # 시나리오 5: 추세 하락 확인
    sell_scenario_5 = (
        (df['close'] < df['ma_trend']) &
        (df['ma_fast'] < df['ma_fast'].shift(3)) &
        (df['volume'] > df['volume'].shift(1))
    )
    
    # 종합 매도 조건 (시나리오 중 하나라도 충족)
    sell_condition = (
        sell_scenario_1 | 
        sell_scenario_2 | 
        sell_scenario_3 | 
        sell_scenario_4 | 
        sell_scenario_5
    )
    
    # 상위 타임프레임 필터링 (완화: 하락 추세이거나 횡보장일 때만)
    higher_timeframe_filter_sell = (~higher_trend | (higher_trend_strength < MIN_TREND_STRENGTH))
    
    # 최종 매도 신호
    df.loc[sell_condition & higher_timeframe_filter_sell, 'signal'] = -1
    
    # 디버깅 정보 출력
    print("\n=== 매매 신호 분석 ===")
    print(f"총 데이터 포인트: {len(df)}")
    print("\n매수 조건 만족 횟수:")
    print(f"MACD 조건: {df['macd_condition'].sum()}")
    print(f"RSI 조건: {df['rsi_condition'].sum()}")
    print(f"볼린저 밴드 조건: {df['bb_condition'].sum()}")
    print(f"추세 조건: {df['trend_condition'].sum()}")
    print(f"거래량 조건: {df['volume_condition'].sum()}")
    print(f"변동성 조건: {df['volatility_condition'].sum()}")
    print(f"패턴 조건: {df['pattern_condition'].sum()}")
    
    print(f"시나리오 1 (RSI 반등): {buy_scenario_1.sum()}")
    print(f"시나리오 2 (MACD 교차): {buy_scenario_2.sum()}")
    print(f"시나리오 3 (BB 반등): {buy_scenario_3.sum()}")
    print(f"시나리오 4 (패턴 기반): {buy_scenario_4.sum()}")
    print(f"시나리오 5 (추세 확인): {buy_scenario_5.sum()}")
    
    print(f"최종 매수 신호: {(df['signal'] == 1).sum()}")
    
    print("\n매도 조건 만족 횟수:")
    print(f"MACD 조건: {df['macd_sell_condition'].sum()}")
    print(f"RSI 조건: {df['rsi_sell_condition'].sum()}")
    print(f"볼린저 밴드 조건: {df['bb_sell_condition'].sum()}")
    print(f"추세 조건: {df['trend_sell_condition'].sum()}")
    print(f"패턴 조건: {df['pattern_sell_condition'].sum()}")
    
    print(f"시나리오 1 (RSI 반전): {sell_scenario_1.sum()}")
    print(f"시나리오 2 (MACD 교차): {sell_scenario_2.sum()}")
    print(f"시나리오 3 (BB 이탈): {sell_scenario_3.sum()}")
    print(f"시나리오 4 (패턴 기반): {sell_scenario_4.sum()}")
    print(f"시나리오 5 (추세 확인): {sell_scenario_5.sum()}")
    
    print(f"최종 매도 신호: {(df['signal'] == -1).sum()}")
    
    # 월별 신호 개수 분석 추가
    df['month'] = df.index.month
    monthly_signals = df.groupby('month')['signal'].apply(lambda x: (x != 0).sum())
    print("\n월별 거래 신호 수:")
    print(monthly_signals)
    
    # 월 평균 거래 수 계산
    months_count = len(monthly_signals)
    if months_count > 0:
        avg_trades_per_month = monthly_signals.sum() / months_count
        print(f"\n월 평균 거래 수: {avg_trades_per_month:.2f}")
    
    return df[['close', 'signal', 'dynamic_stop_loss', 'dynamic_take_profit', 'volatility_state']]

def execute_trade(df, position, entry_price, entry_time, current_price, current_time):
    """향상된 변동성 기반 거래 실행 및 자금 관리"""
    if position == 0:  # 포지션 없음
        return 0, current_price, current_time
    
    # 현재 변동성 상태 확인
    current_volatility = df.loc[current_time, 'volatility_state']
    dynamic_stop_loss = df.loc[current_time, 'dynamic_stop_loss']
    dynamic_take_profit = df.loc[current_time, 'dynamic_take_profit']
    
    # 수익률 계산
    pnl = (current_price - entry_price) / entry_price if position > 0 else (entry_price - current_price) / entry_price
    
    # 거래 진입 시간 계산
    holding_time = (current_time - entry_time).total_seconds() / 3600  # 시간 단위
    
    # 시간 기반 동적 익절/손절 조정
    time_factor = min(1.5, max(0.8, 1 + (holding_time / 48)))  # 48시간 후 최대 50% 증가, 최소 20% 감소
    
    # 변동성 상태에 따른 추가 조정
    if current_volatility == 'high':
        vol_factor = 0.8  # 높은 변동성에서는 더 빨리 익절
    elif current_volatility == 'low':
        vol_factor = 1.2  # 낮은 변동성에서는 더 길게 홀딩
    else:
        vol_factor = 1.0  # 정상 변동성에서는 기본값
    
    # 최종 익절/손절 값 계산
    adjusted_take_profit = dynamic_take_profit * time_factor * vol_factor
    adjusted_stop_loss = dynamic_stop_loss * min(1.2, time_factor)  # 손절폭은 너무 넓어지지 않도록 제한
    
    # 트레일링 스탑 활성화 기준 (역시 시간에 따라 조정)
    trailing_activation = TRAILING_STOP_ACTIVATION * (0.8 if holding_time > 24 else 1.0)
    
    # 변동성 기반 청산 조건
    if position > 0:  # 롱 포지션
        # 손절
        if pnl <= -adjusted_stop_loss:
            print(f"손절 실행: {pnl*100:.2f}% (기준: {-adjusted_stop_loss*100:.2f}%)")
            return 0, current_price, current_time
        # 익절
        if pnl >= adjusted_take_profit:
            print(f"익절 실행: {pnl*100:.2f}% (기준: {adjusted_take_profit*100:.2f}%)")
            return 0, current_price, current_time
        # 트레일링 스탑
        if pnl >= trailing_activation:
            trailing_stop = pnl * TRAILING_STOP_MULTIPLIER
            if pnl <= trailing_stop:
                print(f"트레일링 스탑 발동: {pnl*100:.2f}% (기준: {trailing_stop*100:.2f}%)")
                return 0, current_price, current_time
    else:  # 숏 포지션
        # 손절
        if pnl <= -adjusted_stop_loss:
            print(f"손절 실행: {pnl*100:.2f}% (기준: {-adjusted_stop_loss*100:.2f}%)")
            return 0, current_price, current_time
        # 익절
        if pnl >= adjusted_take_profit:
            print(f"익절 실행: {pnl*100:.2f}% (기준: {adjusted_take_profit*100:.2f}%)")
            return 0, current_price, current_time
        # 트레일링 스탑
        if pnl >= trailing_activation:
            trailing_stop = pnl * TRAILING_STOP_MULTIPLIER
            if pnl <= trailing_stop:
                print(f"트레일링 스탑 발동: {pnl*100:.2f}% (기준: {trailing_stop*100:.2f}%)")
                return 0, current_price, current_time
    
    return position, entry_price, entry_time

# 백테스트 실행
def run_backtest(signals, initial_balance=10000):
    balance = initial_balance  # 초기 USDT 잔고
    btc_amount = 0             # 초기 BTC 보유량
    position = None            # 현재 포지션 (None: 없음, 'LONG': 롱, 'SHORT': 숏)
    entry_price = 0            # 진입 가격
    
    trades = []                # 모든 거래 내역
    balance_history = []       # 잔고 히스토리
    
    for i in range(len(signals)):
        current_time = signals.index[i]
        current_price = signals['close'].iloc[i]
        current_signal = signals['signal'].iloc[i]
        
        # 초기 잔고 기록
        if i == 0:
            balance_history.append({
                'time': current_time,
                'balance': balance,
                'btc_amount': btc_amount,
                'total_value': balance
            })
        
        # 매수 신호
        if current_signal == 1 and position is None:
            # 롱 포지션 진입
            btc_to_buy = (balance * 0.99) / current_price  # 1% 슬리피지 고려
            entry_price = current_price
            balance = 0
            btc_amount = btc_to_buy
            position = 'LONG'
            
            trades.append({
                'time': current_time,
                'type': 'ENTRY',
                'position': 'LONG',
                'price': current_price,
                'btc_amount': btc_amount,
                'balance': balance
            })
        
        # 매도 신호
        elif current_signal == -1 and position is None:
            # 숏 포지션 진입 (현물거래에서는 불가능하므로 생략)
            # 실제 숏을 구현하려면 더 복잡한 로직이 필요함
            pass
        
        # 포지션 종료 조건 확인 (손절/익절)
        if position == 'LONG':
            change_pct = (current_price - entry_price) / entry_price
            
            # 익절
            if change_pct >= TAKE_PROFIT_PCT:
                balance = btc_amount * current_price * 0.99  # 1% 슬리피지 고려
                trades.append({
                    'time': current_time,
                    'type': 'EXIT',
                    'position': 'LONG',
                    'price': current_price,
                    'btc_amount': btc_amount,
                    'balance': balance,
                    'profit_pct': change_pct * 100,
                    'reason': '익절'
                })
                btc_amount = 0
                position = None
            
            # 손절
            elif change_pct <= -STOP_LOSS_PCT:
                balance = btc_amount * current_price * 0.99  # 1% 슬리피지 고려
                trades.append({
                    'time': current_time,
                    'type': 'EXIT',
                    'position': 'LONG',
                    'price': current_price,
                    'btc_amount': btc_amount,
                    'balance': balance,
                    'profit_pct': change_pct * 100,
                    'reason': '손절'
                })
                btc_amount = 0
                position = None
        
        # 매일 잔고 기록
        if i > 0 and current_time.date() != signals.index[i-1].date():
            total_value = balance
            if btc_amount > 0:
                total_value = balance + (btc_amount * current_price)
            
            balance_history.append({
                'time': current_time,
                'balance': balance,
                'btc_amount': btc_amount,
                'total_value': total_value
            })
    
    # 마지막 포지션 정리
    if position == 'LONG':
        current_price = signals['close'].iloc[-1]
        balance = btc_amount * current_price * 0.99
        change_pct = (current_price - entry_price) / entry_price
        
        trades.append({
            'time': signals.index[-1],
            'type': 'EXIT',
            'position': 'LONG',
            'price': current_price,
            'btc_amount': btc_amount,
            'balance': balance,
            'profit_pct': change_pct * 100,
            'reason': '백테스트 종료'
        })
        btc_amount = 0
    
    # 최종 잔고 계산
    final_balance = balance
    
    # 수익률 계산
    profit_pct = ((final_balance - initial_balance) / initial_balance) * 100
    
    # 수익성 있는 거래 분석
    profitable_trades = [t for t in trades if t['type'] == 'EXIT' and t['profit_pct'] > 0]
    losing_trades = [t for t in trades if t['type'] == 'EXIT' and t['profit_pct'] <= 0]
    
    # 최대 낙폭 계산
    if balance_history:
        balance_df = pd.DataFrame(balance_history)
        if not balance_df.empty and 'total_value' in balance_df:
            balance_df['drawdown'] = balance_df['total_value'].cummax() - balance_df['total_value']
            balance_df['drawdown_pct'] = (balance_df['drawdown'] / balance_df['total_value'].cummax()) * 100
            max_drawdown = balance_df['drawdown_pct'].max()
        else:
            max_drawdown = 0
    else:
        max_drawdown = 0
    
    # 결과 요약
    results = {
        'initial_balance': initial_balance,
        'final_balance': final_balance,
        'profit_pct': profit_pct,
        'total_trades': len([t for t in trades if t['type'] == 'EXIT']),
        'profitable_trades': len(profitable_trades),
        'losing_trades': len(losing_trades),
        'win_rate': len(profitable_trades) / len([t for t in trades if t['type'] == 'EXIT']) if trades else 0,
        'max_drawdown': max_drawdown,
        'trades': trades,
        'balance_history': balance_history
    }
    
    return results

# 결과 시각화
def plot_results(df, signals, results):
    # 캔들차트, MACD, RSI, 볼린저 밴드, 이동평균, 거래량, 매매 시그널 시각화
    fig, axs = plt.subplots(7, 1, figsize=(14, 28), gridspec_kw={'height_ratios': [3, 1, 1, 1, 1, 1, 1]})
    
    # 캔들차트, 볼린저 밴드, 이동평균선
    axs[0].set_title(f'{SYMBOL} {MAIN_INTERVAL} 백테스트')
    axs[0].plot(df.index, df['close'], label='종가', color='black', alpha=0.3)
    axs[0].plot(df.index, df['bb_upper'], label='BB Upper', color='red', alpha=0.5)
    axs[0].plot(df.index, df['bb_middle'], label='BB Middle', color='blue', alpha=0.5)
    axs[0].plot(df.index, df['bb_lower'], label='BB Lower', color='green', alpha=0.5)
    axs[0].plot(df.index, df['ma_fast'], label=f'MA{MA_FAST}', color='orange', alpha=0.5)
    axs[0].plot(df.index, df['ma_slow'], label=f'MA{MA_SLOW}', color='purple', alpha=0.5)
    axs[0].plot(df.index, df['ma_trend'], label=f'MA{MA_TREND}', color='brown', alpha=0.5)
    
    # 매매 시그널 표시
    for trade in results['trades']:
        if trade['type'] == 'ENTRY':
            axs[0].scatter(trade['time'], trade['price'], color='green', marker='^', s=100)
        elif trade['type'] == 'EXIT':
            axs[0].scatter(trade['time'], trade['price'], color='red', marker='v', s=100)
    axs[0].legend()
    
    # MACD
    axs[1].set_title('MACD')
    axs[1].plot(df.index, df['macd'], label='MACD')
    axs[1].plot(df.index, df['macd_signal'], label='Signal')
    axs[1].bar(df.index, df['macd_hist'], label='Histogram', alpha=0.3)
    axs[1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
    axs[1].legend()
    
    # RSI
    axs[2].set_title('RSI')
    axs[2].plot(df.index, df['rsi'], label='RSI')
    axs[2].axhline(y=RSI_OVERSOLD, color='green', linestyle='--', alpha=0.5)
    axs[2].axhline(y=RSI_OVERBOUGHT, color='red', linestyle='--', alpha=0.5)
    axs[2].axhline(y=50, color='black', linestyle='-', alpha=0.3)
    axs[2].set_ylim(0, 100)
    axs[2].legend()
    
    # 볼린저 밴드 %B
    axs[3].set_title('Bollinger Bands %B')
    df['bb_pct'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
    axs[3].plot(df.index, df['bb_pct'], label='%B')
    axs[3].axhline(y=0, color='green', linestyle='--', alpha=0.5)
    axs[3].axhline(y=1, color='red', linestyle='--', alpha=0.5)
    axs[3].set_ylim(-0.5, 1.5)
    axs[3].legend()
    
    # 거래량
    axs[4].set_title('Volume Analysis')
    axs[4].bar(df.index, df['volume'], label='Volume', alpha=0.3)
    axs[4].plot(df.index, df['volume_ma'], label=f'Volume MA{VOLUME_MA_PERIOD}', color='red')
    axs[4].legend()
    
    # 추세 강도
    axs[5].set_title('Trend Strength')
    axs[5].plot(df.index, df['trend_strength'], label='Trend Strength')
    axs[5].axhline(y=MIN_TREND_STRENGTH, color='red', linestyle='--', alpha=0.5)
    axs[5].axhline(y=MAX_TREND_STRENGTH, color='red', linestyle='--', alpha=0.5)
    axs[5].legend()
    
    # 자산 가치 변화
    if results['balance_history']:
        balance_df = pd.DataFrame(results['balance_history'])
        axs[6].set_title('자산 가치 변화')
        axs[6].plot(balance_df['time'], balance_df['total_value'], label='총 자산가치')
        axs[6].legend()
    
    plt.tight_layout()
    plt.savefig('backtest_result.png', dpi=300)
    plt.show()

# 백테스트 실행 함수
def main():
    global client
    
    # API 키 검증
    if not validate_api_keys():
        print("API 키 설정 오류. 프로그램을 종료합니다.")
        return
    
    # Binance 클라이언트 초기화
    try:
        client = Client(BINANCE_API_KEY, BINANCE_API_SECRET)
        # 연결 테스트
        client.get_exchange_info()
        print("Binance API 연결 성공!")
    except Exception as e:
        print(f"Binance API 연결 실패: {e}")
        return
    
    print(f"백테스트 시작: {SYMBOL} {MAIN_INTERVAL}")
    
    # 백테스트 기간 설정
    end_time = datetime.now()
    start_time = end_time - timedelta(days=180)  # 6개월
    print(f"기간: {start_time.strftime('%d %b, %Y')} - {end_time.strftime('%d %b, %Y')}")
    
    # 과거 데이터 다운로드
    print("과거 데이터 다운로드 중...")
    df = fetch_historical_data(SYMBOL, MAIN_INTERVAL, start_time.strftime("%d %b, %Y"), end_time.strftime("%d %b, %Y"))
    print(f"다운로드 완료: {len(df)} 개의 {MAIN_INTERVAL} 데이터")
    
    print("과거 데이터 다운로드 중...")
    df_trend = fetch_historical_data(SYMBOL, TREND_INTERVAL, start_time.strftime("%d %b, %Y"), end_time.strftime("%d %b, %Y"))
    print(f"다운로드 완료: {len(df_trend)} 개의 {TREND_INTERVAL} 데이터")
    
    # 충분한 데이터가 있는지 확인
    if len(df) < 100 or len(df_trend) < 100:
        print("충분한 데이터가 없습니다.")
        return
    
    # 기술적 지표 계산
    df = calculate_indicators(df)
    df_trend = calculate_indicators(df_trend)
    
    # 매매 신호 생성
    signals = generate_signals(df, df_trend)
    
    # 백테스트 실행
    initial_capital = 10000
    capital = initial_capital
    position = 0
    entry_price = 0
    entry_time = None
    trades = []
    max_drawdown = 0
    peak_capital = initial_capital
    balance_history = []  # 잔고 히스토리 추가
    
    # 일일 거래 횟수 제한을 위한 변수
    current_date = None
    daily_trades = 0
    
    for i in range(len(df)):
        current_time = df.index[i]
        current_price = df['close'].iloc[i]
        
        # 새로운 날짜인 경우 일일 거래 횟수 초기화
        if current_date != current_time.date():
            current_date = current_time.date()
            daily_trades = 0
        
        # 일일 거래 횟수 제한 확인
        if daily_trades >= MAX_DAILY_TRADES:
            continue
        
        # 포지션이 없는 경우
        if position == 0:
            if signals['signal'].iloc[i] == 1 and daily_trades < MAX_DAILY_TRADES:  # 매수 신호
                position = 1
                entry_price = current_price
                entry_time = current_time
                daily_trades += 1
                trades.append({
                    'type': 'LONG',
                    'entry_time': entry_time,
                    'entry_price': entry_price,
                    'exit_time': None,
                    'exit_price': None,
                    'pnl': None
                })
            elif signals['signal'].iloc[i] == -1 and daily_trades < MAX_DAILY_TRADES:  # 매도 신호
                position = -1
                entry_price = current_price
                entry_time = current_time
                daily_trades += 1
                trades.append({
                    'type': 'SHORT',
                    'entry_time': entry_time,
                    'entry_price': entry_price,
                    'exit_time': None,
                    'exit_price': None,
                    'pnl': None
                })
        # 포지션이 있는 경우
        else:
            # 변동성 기반 청산 조건 확인
            position, exit_price, exit_time = execute_trade(
                signals, position, entry_price, entry_time, current_price, current_time
            )
            
            # 포지션 청산
            if position == 0:
                pnl = (exit_price - entry_price) / entry_price if trades[-1]['type'] == 'LONG' else (entry_price - exit_price) / entry_price
                trades[-1].update({
                    'exit_time': exit_time,
                    'exit_price': exit_price,
                    'pnl': pnl
                })
                capital *= (1 + pnl)
                peak_capital = max(peak_capital, capital)
                max_drawdown = max(max_drawdown, (peak_capital - capital) / peak_capital)
        
        # 잔고 히스토리 기록
        balance_history.append({
            'time': current_time,
            'balance': capital,
            'btc_amount': 0 if position == 0 else (capital / current_price if position == 1 else -capital / current_price),
            'total_value': capital
        })
    
    # 결과 출력
    print("\n===== 백테스트 결과 =====")
    print(f"초기 자본: {initial_capital:.2f} USDT")
    print(f"최종 자본: {capital:.2f} USDT")
    print(f"순이익: {capital - initial_capital:.2f} USDT")
    print(f"수익률: {(capital - initial_capital) / initial_capital * 100:.2f}%")
    print(f"총 거래 횟수: {len(trades)}")
    winning_trades = [t for t in trades if t['pnl'] and t['pnl'] > 0]
    losing_trades = [t for t in trades if t['pnl'] and t['pnl'] <= 0]
    if trades:
        print(f"승리한 거래: {len(winning_trades)}")
        print(f"손실한 거래: {len(losing_trades)}")
        print(f"승률: {len(winning_trades) / len(trades) * 100:.2f}%")
    print(f"최대 낙폭: {max_drawdown * 100:.2f}%")
    
    # 거래 내역 출력
    print("\n===== 거래 내역 =====")
    for i, trade in enumerate(trades, 1):
        if trade['exit_time']:
            print(f"{i}. {trade['entry_time']} - {trade['type']} {'익절' if trade['pnl'] > 0 else '손절'} @ {trade['exit_price']:.2f} USDT (손익: {trade['pnl']*100:.2f}%)")
        else:
            print(f"{i}. {trade['entry_time']} - {trade['type']} 백테스트 종료 @ {current_price:.2f} USDT (손익: {((current_price - trade['entry_price']) / trade['entry_price'])*100:.2f}%)")
    
    # 결과 시각화
    plot_results(df, signals, {
        'initial_capital': initial_capital,
        'final_capital': capital,
        'trades': trades,
        'max_drawdown': max_drawdown,
        'balance_history': balance_history
    })

if __name__ == '__main__':
    main() 