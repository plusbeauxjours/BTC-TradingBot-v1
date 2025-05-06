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
    
    for col in ['open', 'high', 'low', 'close', 'volume']:
        data[col] = data[col].astype(float)
    
    print(f"다운로드 완료: {len(data)} 개의 {interval} 데이터")
    return data

# 매매 신호 생성
def generate_signals(df):
    signals = pd.DataFrame(index=df.index)
    signals['price'] = df['close']
    signals['signal'] = 0  # 0: 중립, 1: 매수, -1: 매도
    
    # 매수 신호 (MACD 골든 크로스 + RSI 과매도)
    buy_signals = (df['macd'] > df['macd_signal']) & (df['macd'].shift(1) <= df['macd_signal'].shift(1)) & (df['rsi'] <= RSI_OVERSOLD)
    
    # 매도 신호 (MACD 데드 크로스 + RSI 과매수)
    sell_signals = (df['macd'] < df['macd_signal']) & (df['macd'].shift(1) >= df['macd_signal'].shift(1)) & (df['rsi'] >= RSI_OVERBOUGHT)
    
    signals.loc[buy_signals, 'signal'] = 1
    signals.loc[sell_signals, 'signal'] = -1
    
    return signals

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
        current_price = signals['price'].iloc[i]
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
        if i > 0 and current_time.day != signals.index[i-1].day:
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
        current_price = signals['price'].iloc[-1]
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
    # 캔들차트, MACD, RSI, 매매 시그널 시각화
    fig, axs = plt.subplots(4, 1, figsize=(14, 16), gridspec_kw={'height_ratios': [3, 1, 1, 1]})
    
    # 캔들차트
    axs[0].set_title(f'{SYMBOL} {INTERVAL} 백테스트')
    axs[0].plot(df.index, df['close'], label='종가', color='black', alpha=0.3)
    
    # 매매 시그널 표시
    for trade in results['trades']:
        if trade['type'] == 'ENTRY':
            axs[0].scatter(trade['time'], trade['price'], color='green', marker='^', s=100)
        elif trade['type'] == 'EXIT':
            axs[0].scatter(trade['time'], trade['price'], color='red', marker='v', s=100)
    
    # MACD
    axs[1].set_title('MACD')
    axs[1].plot(df.index, df['macd'], label='MACD')
    axs[1].plot(df.index, df['macd_signal'], label='Signal')
    axs[1].bar(df.index, df['macd'] - df['macd_signal'], label='Histogram', alpha=0.3)
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
    
    # 자산 가치 변화
    if results['balance_history']:
        balance_df = pd.DataFrame(results['balance_history'])
        axs[3].set_title('자산 가치 변화')
        axs[3].plot(balance_df['time'], balance_df['total_value'], label='총 자산가치')
        axs[3].legend()
    
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
    
    print(f'백테스트 시작: {SYMBOL} {INTERVAL}')
    print(f'기간: {START_DATE} - {END_DATE}')
    
    # 과거 데이터 가져오기
    df = fetch_historical_data(SYMBOL, INTERVAL, START_DATE, END_DATE)
    
    if len(df) < 50:
        print("충분한 데이터가 없습니다.")
        return
    
    # 지표 계산
    df = calculate_indicators(df)
    
    # RSI 계산에 필요한 충분한 데이터 확보를 위해 초기 데이터 제거
    df = df.iloc[RSI_PERIOD:]
    
    # 매매 신호 생성
    signals = generate_signals(df)
    
    # 백테스트 실행
    results = run_backtest(signals, INITIAL_BALANCE)
    
    # 결과 출력
    print("\n===== 백테스트 결과 =====")
    print(f"초기 자본: {results['initial_balance']} USDT")
    print(f"최종 자본: {results['final_balance']:.2f} USDT")
    print(f"순이익: {results['final_balance'] - results['initial_balance']:.2f} USDT")
    print(f"수익률: {results['profit_pct']:.2f}%")
    print(f"총 거래 횟수: {results['total_trades']}")
    print(f"승리한 거래: {results['profitable_trades']}")
    print(f"손실한 거래: {results['losing_trades']}")
    print(f"승률: {results['win_rate']*100:.2f}%")
    print(f"최대 낙폭: {results['max_drawdown']:.2f}%")
    
    print("\n===== 거래 내역 =====")
    for i, trade in enumerate([t for t in results['trades'] if t['type'] == 'EXIT']):
        print(f"{i+1}. {trade['time']} - {trade['position']} {trade['reason']} @ {trade['price']:.2f} USDT (손익: {trade['profit_pct']:.2f}%)")
    
    # 결과 시각화
    plot_results(df, signals, results)

if __name__ == '__main__':
    main() 