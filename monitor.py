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
    data['open_time'] = pd.to_datetime(data['open_time'], unit='ms')
    data['close_time'] = pd.to_datetime(data['close_time'], unit='ms')
    for col in ['open', 'high', 'low', 'close', 'volume']:
        data[col] = data[col].astype(float)
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

# 메인 모니터링 함수
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
    
    print('모니터링 시작...')
    
    # 가상 포지션 추적 (시뮬레이션용)
    virtual_position = None
    entry_price = 0
    trade_history = []
    
    print(f"대상 코인: {SYMBOL}")
    print(f"봉 주기: {INTERVAL}")
    print(f"RSI 설정: {RSI_PERIOD}일, 과매수: {RSI_OVERBOUGHT}, 과매도: {RSI_OVERSOLD}")
    print(f"MACD 설정: Fast: {MACD_FAST}, Slow: {MACD_SLOW}, Signal: {MACD_SIGNAL}")
    print(f"거래량: {POSITION_SIZE} BTC")
    print(f"손절: {STOP_LOSS_PCT*100}%, 익절: {TAKE_PROFIT_PCT*100}%")
    print("=" * 50)

    try:
        while True:
            df = fetch_klines(SYMBOL, INTERVAL)
            df = calculate_indicators(df)
            signal = check_signal(df)
            
            current_time = time.strftime("%Y-%m-%d %H:%M:%S")
            current_price = float(df['close'].iloc[-1])
            current_rsi = df['rsi'].iloc[-1]
            current_macd = df['macd'].iloc[-1]
            current_macd_signal = df['macd_signal'].iloc[-1]
            
            print(f"시간: {current_time}")
            print(f"가격: {current_price:.2f} USDT")
            print(f"RSI: {current_rsi:.2f}")
            print(f"MACD: {current_macd:.4f}, Signal: {current_macd_signal:.4f}, Diff: {(current_macd - current_macd_signal):.4f}")
            
            # 신호 체크 및 가상 매매
            if signal:
                print(f"신호 감지: {signal}")
                
                if not virtual_position and signal == 'BUY':
                    virtual_position = 'LONG'
                    entry_price = current_price
                    trade_history.append({
                        'time': current_time,
                        'type': 'ENTRY',
                        'position': 'LONG',
                        'price': current_price,
                        'rsi': current_rsi,
                        'macd': current_macd
                    })
                    print(f"가상 롱 포지션 진입, 진입가: {entry_price:.2f}")
                    
                elif not virtual_position and signal == 'SELL':
                    virtual_position = 'SHORT'
                    entry_price = current_price
                    trade_history.append({
                        'time': current_time,
                        'type': 'ENTRY',
                        'position': 'SHORT',
                        'price': current_price,
                        'rsi': current_rsi,
                        'macd': current_macd
                    })
                    print(f"가상 숏 포지션 진입, 진입가: {entry_price:.2f}")
            
            # 가상 포지션 손익 계산
            if virtual_position:
                change_pct = ((current_price - entry_price) / entry_price) if virtual_position == 'LONG' else ((entry_price - current_price) / entry_price)
                profit_usdt = change_pct * entry_price * POSITION_SIZE
                
                print(f"가상 포지션: {virtual_position}")
                print(f"진입가: {entry_price:.2f}")
                print(f"현재 손익: {change_pct*100:.2f}% (약 {profit_usdt:.2f} USDT)")
                
                # 손절/익절 체크
                if change_pct >= TAKE_PROFIT_PCT:
                    exit_type = "익절"
                    trade_history.append({
                        'time': current_time,
                        'type': 'EXIT',
                        'position': virtual_position,
                        'entry_price': entry_price,
                        'exit_price': current_price,
                        'profit_pct': change_pct*100,
                        'exit_type': exit_type
                    })
                    print(f"가상 {exit_type} 실행 ({TAKE_PROFIT_PCT*100}%)")
                    virtual_position = None
                    
                elif change_pct <= -STOP_LOSS_PCT:
                    exit_type = "손절"
                    trade_history.append({
                        'time': current_time,
                        'type': 'EXIT',
                        'position': virtual_position,
                        'entry_price': entry_price,
                        'exit_price': current_price,
                        'profit_pct': change_pct*100,
                        'exit_type': exit_type
                    })
                    print(f"가상 {exit_type} 실행 ({-STOP_LOSS_PCT*100}%)")
                    virtual_position = None
            
            # 가상 매매 내역 출력
            if len(trade_history) > 0:
                print("\n가상 매매 내역:")
                for i, trade in enumerate(trade_history[-3:], 1):
                    if trade['type'] == 'ENTRY':
                        print(f"{i}. {trade['time']} - {trade['position']} 진입 @ {trade['price']:.2f}")
                    else:
                        print(f"{i}. {trade['time']} - {trade['position']} {trade['exit_type']} @ {trade['exit_price']:.2f} (손익: {trade['profit_pct']:.2f}%)")
            
            print("=" * 50)
            time.sleep(60)  # 1분마다 갱신
            
    except KeyboardInterrupt:
        print("\n모니터링을 종료합니다.")
    except Exception as e:
        print(f"오류 발생: {e}")

if __name__ == '__main__':
    main() 