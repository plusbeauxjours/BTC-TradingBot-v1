import time
import numpy as np
import pandas as pd
from binance.client import Client
from binance.enums import *
from config import *

# Binance 클라이언트 초기화
client = None

# 지표 계산 함수
def calculate_indicators(df):
    # EMA
    df['ema_fast'] = df['close'].ewm(span=MACD_FAST, adjust=False).mean()
    df['ema_slow'] = df['close'].ewm(span=MACD_SLOW, adjust=False).mean()
    df['macd'] = df['ema_fast'] - df['ema_slow']
    df['macd_signal'] = df['macd'].ewm(span=MACD_SIGNAL, adjust=False).mean()

    # RSI
    delta = df['close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=RSI_PERIOD).mean()
    avg_loss = loss.rolling(window=RSI_PERIOD).mean()
    rs = avg_gain / avg_loss
    df['rsi'] = 100 - (100 / (1 + rs))
    return df

# K라인 불러오기
def fetch_klines(symbol, interval, limit=100):
    klines = client.get_klines(symbol=symbol, interval=interval, limit=limit)
    data = pd.DataFrame(klines, columns=[
        'open_time', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'qav', 'num_trades', 'taker_base', 'taker_quote', 'ignore'
    ])
    data['close'] = data['close'].astype(float)
    return data

# 매매 신호 체크
def check_signal(df):
    last = df.iloc[-1]
    prev = df.iloc[-2]

    # 골든 크로스
    macd_cross_up = (prev['macd'] < prev['macd_signal']) and (last['macd'] > last['macd_signal'])
    # 데드 크로스
    macd_cross_down = (prev['macd'] > prev['macd_signal']) and (last['macd'] < last['macd_signal'])

    rsi = last['rsi']

    if macd_cross_up and rsi <= RSI_OVERSOLD:
        return 'BUY'
    if macd_cross_down and rsi >= RSI_OVERBOUGHT:
        return 'SELL'
    return None

# 주문 실행
def execute_order(side, quantity):
    try:
        order = client.create_order(
            symbol=SYMBOL,
            side=SIDE_BUY if side == 'BUY' else SIDE_SELL,
            type=ORDER_TYPE_MARKET,
            quantity=quantity
        )
        print(f"{side} order executed: {order}")
        return order
    except Exception as e:
        print(f"Order failed: {e}")
        return None

# 메인 루프
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
        client.get_account()
        print("Binance API 연결 성공!")
    except Exception as e:
        print(f"Binance API 연결 실패: {e}")
        return
    
    print('자동매매 시작...')
    position = None
    entry_price = 0

    while True:
        try:
            df = fetch_klines(SYMBOL, INTERVAL)
            df = calculate_indicators(df)
            signal = check_signal(df)
            price = float(df['close'].iloc[-1])

            print(f"현재 시간: {time.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"현재 가격: {price}, RSI: {df['rsi'].iloc[-1]:.2f}, MACD: {df['macd'].iloc[-1]:.2f}, Signal: {df['macd_signal'].iloc[-1]:.2f}")
            
            # 포지션 진입
            if not position and signal == 'BUY':
                print("매수 신호 감지!")
                order = execute_order('BUY', POSITION_SIZE)
                if order:
                    position = 'LONG'
                    entry_price = price
                    print(f"롱 포지션 진입, 진입가: {entry_price}")
                    
            if not position and signal == 'SELL':
                print("매도 신호 감지!")
                order = execute_order('SELL', POSITION_SIZE)
                if order:
                    position = 'SHORT'
                    entry_price = price
                    print(f"숏 포지션 진입, 진입가: {entry_price}")

            # 손절/익절 체크
            if position:
                change_pct = (price - entry_price) / entry_price if position == 'LONG' else (entry_price - price) / entry_price
                print(f"현재 손익: {change_pct*100:.2f}%")
                
                if change_pct >= TAKE_PROFIT_PCT:
                    print(f"익절 조건 충족 ({TAKE_PROFIT_PCT*100}%)")
                    side = 'SELL' if position == 'LONG' else 'BUY'
                    order = execute_order(side, POSITION_SIZE)
                    if order:
                        print(f"포지션 종료 ({position}), 손익: {change_pct*100:.2f}%")
                        position = None
                        entry_price = 0
                        
                elif change_pct <= -STOP_LOSS_PCT:
                    print(f"손절 조건 충족 ({-STOP_LOSS_PCT*100}%)")
                    side = 'SELL' if position == 'LONG' else 'BUY'
                    order = execute_order(side, POSITION_SIZE)
                    if order:
                        print(f"포지션 종료 ({position}), 손익: {change_pct*100:.2f}%")
                        position = None
                        entry_price = 0
            
            print("=" * 50)
            time.sleep(60)  # 1분마다 체크 (실제 운영 시 60*60 설정으로 변경)
            
        except Exception as e:
            print(f"오류 발생: {e}")
            time.sleep(60)

if __name__ == '__main__':
    main()
