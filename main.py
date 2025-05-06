import time
import numpy as np
import pandas as pd
from binance.client import Client
from binance.enums import *
from config import *
from datetime import datetime, timedelta

# Binance client initialization
client = None

def calculate_atr(df, period=14):
    high_low = df['high'].astype(float) - df['low'].astype(float)
    high_close = np.abs(df['high'].astype(float) - df['close'].astype(float).shift())
    low_close = np.abs(df['low'].astype(float) - df['close'].astype(float).shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    return true_range.rolling(period).mean()

def calculate_indicators(df):
    # Convert string columns to float
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = df[col].astype(float)
    
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
    
    # Moving Averages
    df['ma_fast'] = df['close'].rolling(window=MA_FAST).mean()
    df['ma_slow'] = df['close'].rolling(window=MA_SLOW).mean()
    df['ma_trend'] = df['close'].rolling(window=MA_TREND).mean()
    
    # Bollinger Bands
    df['bb_middle'] = df['close'].rolling(window=BB_PERIOD).mean()
    df['bb_std'] = df['close'].rolling(window=BB_PERIOD).std()
    df['bb_upper'] = df['bb_middle'] + (df['bb_std'] * BB_STD)
    df['bb_lower'] = df['bb_middle'] - (df['bb_std'] * BB_STD)
    
    # Volume Analysis
    df['volume_ma'] = df['volume'].rolling(window=VOLUME_MA_PERIOD).mean()
    df['volume_ratio'] = df['volume'] / df['volume_ma']
    
    # ATR for volatility
    df['atr'] = calculate_atr(df, ATR_PERIOD)
    
    return df

def fetch_klines(symbol, interval, limit=100):
    try:
        klines = client.get_klines(symbol=symbol, interval=interval, limit=limit)
        data = pd.DataFrame(klines, columns=[
            'open_time', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'qav', 'num_trades', 'taker_base', 'taker_quote', 'ignore'
        ])
        data['timestamp'] = pd.to_datetime(data['open_time'], unit='ms')
        return data
    except Exception as e:
        print(f"Error fetching klines: {e}")
        return None

def analyze_trend(df_main, df_trend):
    """Enhanced trend analysis with multiple confirmation methods"""
    # Price action analysis
    last_main = df_main.iloc[-1]
    last_trend = df_trend.iloc[-1]
    
    # Calculate swing points
    def get_swing_points(df, window=20):
        highs = df['high'].rolling(window=window, center=True).max()
        lows = df['low'].rolling(window=window, center=True).min()
        return highs, lows
    
    main_highs, main_lows = get_swing_points(df_main)
    trend_highs, trend_lows = get_swing_points(df_trend)
    
    # Volume-weighted trend
    volume_price = df_main['close'] * df_main['volume']
    vwap = volume_price.rolling(window=20).sum() / df_main['volume'].rolling(window=20).sum()
    
    # Momentum analysis
    main_momentum = (df_main['close'] - df_main['close'].shift(5)) / df_main['close'].shift(5)
    trend_momentum = (df_trend['close'] - df_trend['close'].shift(5)) / df_trend['close'].shift(5)
    
    # Trend strength calculation
    main_trend_strength = abs(df_main['close'].iloc[-1] - df_main['ma_trend'].iloc[-1]) / df_main['ma_trend'].iloc[-1]
    higher_trend_strength = abs(df_trend['close'].iloc[-1] - df_trend['ma_trend'].iloc[-1]) / df_trend['ma_trend'].iloc[-1]
    
    # Market regime detection using volatility
    volatility = df_main['atr'].iloc[-1] / df_main['close'].iloc[-1]
    avg_volatility = (df_main['atr'] / df_main['close']).rolling(window=20).mean().iloc[-1]
    is_volatile = volatility > avg_volatility * 1.5
    
    # Combined trend analysis
    main_trend = df_main['close'].iloc[-1] > df_main['ma_trend'].iloc[-1]
    higher_trend = df_trend['close'].iloc[-1] > df_trend['ma_trend'].iloc[-1]
    
    # Volume trend
    volume_trend = df_main['volume_ratio'].iloc[-1] > VOLUME_THRESHOLD
    sustained_volume = df_main['volume_ratio'].rolling(window=3).mean().iloc[-1] > VOLUME_THRESHOLD
    
    return {
        'trend_aligned': main_trend == higher_trend,
        'is_uptrend': main_trend and higher_trend,
        'trend_strength': (main_trend_strength + higher_trend_strength) / 2,
        'volume_confirmed': volume_trend and sustained_volume,
        'momentum_aligned': (main_momentum.iloc[-1] > 0) == (trend_momentum.iloc[-1] > 0),
        'price_above_vwap': last_main['close'] > vwap.iloc[-1],
        'is_volatile_market': is_volatile,
        'swing_high': trend_highs.iloc[-1],
        'swing_low': trend_lows.iloc[-1]
    }

def calculate_dynamic_risk_params(df, trend_info):
    """Calculate dynamic risk parameters based on market conditions"""
    volatility = df['atr'].iloc[-1] / df['close'].iloc[-1]
    avg_volatility = (df['atr'] / df['close']).rolling(window=20).mean().iloc[-1]
    
    # Adjust stop loss based on volatility
    dynamic_stop = BASE_STOP_LOSS * (volatility / avg_volatility)
    dynamic_stop = min(max(dynamic_stop, BASE_STOP_LOSS * 0.8), BASE_STOP_LOSS * 1.5)
    
    # Adjust take profit based on trend strength
    trend_multiplier = 1 + trend_info['trend_strength']
    dynamic_tp = BASE_TAKE_PROFIT * trend_multiplier
    dynamic_tp = min(max(dynamic_tp, BASE_TAKE_PROFIT * 0.8), BASE_TAKE_PROFIT * 1.5)
    
    # Adjust trailing stop based on momentum
    trailing_activation = TRAILING_STOP_ACTIVATION
    if trend_info['momentum_aligned']:
        trailing_activation *= 0.8  # Tighter trailing stop in strong momentum
    
    return {
        'stop_loss': dynamic_stop,
        'take_profit': dynamic_tp,
        'trailing_activation': trailing_activation
    }

def calculate_position_size(df, base_size):
    """Calculate position size based on volatility"""
    if not VOLATILITY_ADJUSTMENT:
        return base_size
        
    current_atr = df['atr'].iloc[-1]
    avg_atr = df['atr'].rolling(window=ATR_PERIOD*2).mean().iloc[-1]
    
    volatility_ratio = current_atr / avg_atr
    size_multiplier = min(MAX_VOLATILITY_MULTIPLIER, max(0.5, 1/volatility_ratio))
    
    return min(MAX_POSITION_SIZE, base_size * size_multiplier)

def check_signal(df_main, df_trend):
    """Enhanced signal detection with multi-timeframe analysis"""
    last = df_main.iloc[-1]
    prev = df_main.iloc[-2]
    
    # Trend analysis
    trend_info = analyze_trend(df_main, df_trend)
    
    # Base conditions
    macd_cross_up = (prev['macd'] < prev['macd_signal']) and (last['macd'] > last['macd_signal'])
    macd_cross_down = (prev['macd'] > prev['macd_signal']) and (last['macd'] < last['macd_signal'])
    
    # Volume confirmation
    volume_confirmed = last['volume_ratio'] > VOLUME_THRESHOLD
    
    # RSI conditions
    rsi = last['rsi']
    rsi_trending_up = rsi > prev['rsi']
    rsi_trending_down = rsi < prev['rsi']
    
    # Price action confirmation
    price_above_ma = last['close'] > last['ma_fast']
    price_below_ma = last['close'] < last['ma_fast']
    
    # Buy signal
    if (macd_cross_up and 
        rsi <= RSI_OVERSOLD and 
        trend_info['trend_aligned'] and
        trend_info['volume_confirmed'] and
        price_above_ma and
        rsi_trending_up):
        return 'BUY'
    
    # Sell signal
    if (macd_cross_down and 
        rsi >= RSI_OVERBOUGHT and 
        trend_info['trend_aligned'] and
        trend_info['volume_confirmed'] and
        price_below_ma and
        rsi_trending_down):
        return 'SELL'
    
    return None

def execute_order(side, quantity):
    try:
        order = client.create_order(
            symbol=SYMBOL,
            side=side,
            type=ORDER_TYPE_MARKET,
            quantity=quantity
        )
        print(f"{side} order executed: {order}")
        return order
    except Exception as e:
        print(f"Order failed: {e}")
        return None

class TradingState:
    def __init__(self):
        self.position = None
        self.entry_price = 0
        self.trailing_stop = 0
        self.daily_trades = 0
        self.daily_pl = 0
        self.last_trade_date = None
        self.open_trades = 0
    
    def reset_daily_stats(self):
        """Reset daily statistics"""
        current_date = datetime.now().date()
        if self.last_trade_date != current_date:
            self.daily_trades = 0
            self.daily_pl = 0
            self.last_trade_date = current_date

def main():
    global client
    
    if not validate_api_keys():
        print("API key configuration error. Exiting program.")
        return
    
    try:
        client = Client(BINANCE_API_KEY, BINANCE_API_SECRET)
        client.get_account()
        print("Successfully connected to Binance API!")
    except Exception as e:
        print(f"Failed to connect to Binance API: {e}")
        return
    
    print('Starting automated trading...')
    state = TradingState()
    
    while True:
        try:
            # Reset daily stats if needed
            state.reset_daily_stats()
            
            # Fetch data for both timeframes
            df_main = fetch_klines(SYMBOL, MAIN_INTERVAL)
            df_trend = fetch_klines(SYMBOL, TREND_INTERVAL)
            
            if df_main is None or df_trend is None:
                print("Failed to fetch market data. Retrying...")
                time.sleep(60)
                continue
            
            # Calculate indicators
            df_main = calculate_indicators(df_main)
            df_trend = calculate_indicators(df_trend)
            
            current_price = float(df_main['close'].iloc[-1])
            signal = check_signal(df_main, df_trend)
            
            print(f"\nCurrent Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"Current Price: {current_price}")
            print(f"RSI: {df_main['rsi'].iloc[-1]:.2f}")
            print(f"MACD: {df_main['macd'].iloc[-1]:.2f}")
            print(f"Signal: {df_main['macd_signal'].iloc[-1]:.2f}")
            
            # Position entry logic
            if not state.position and state.daily_trades < MAX_DAILY_TRADES:
                position_size = calculate_position_size(df_main, POSITION_SIZE)
                
                if signal == 'BUY' and state.open_trades < MAX_OPEN_TRADES:
                    print("Buy signal detected!")
                    order = execute_order(SIDE_BUY, position_size)
                    if order:
                        state.position = 'LONG'
                        state.entry_price = current_price
                        state.trailing_stop = current_price * (1 - STOP_LOSS_PCT)
                        state.daily_trades += 1
                        state.open_trades += 1
                        print(f"Entered LONG position at {state.entry_price}")
                        
                elif signal == 'SELL' and state.open_trades < MAX_OPEN_TRADES:
                    print("Sell signal detected!")
                    order = execute_order(SIDE_SELL, position_size)
                    if order:
                        state.position = 'SHORT'
                        state.entry_price = current_price
                        state.trailing_stop = current_price * (1 + STOP_LOSS_PCT)
                        state.daily_trades += 1
                        state.open_trades += 1
                        print(f"Entered SHORT position at {state.entry_price}")
            
            # Position management
            if state.position:
                pnl = ((current_price - state.entry_price) / state.entry_price 
                       if state.position == 'LONG' 
                       else (state.entry_price - current_price) / state.entry_price)
                print(f"Current P&L: {pnl*100:.2f}%")
                
                # Update trailing stop if in profit
                if pnl > 0:
                    if state.position == 'LONG':
                        new_stop = current_price * (1 - TRAILING_STOP_PCT)
                        state.trailing_stop = max(state.trailing_stop, new_stop)
                    else:  # SHORT
                        new_stop = current_price * (1 + TRAILING_STOP_PCT)
                        state.trailing_stop = min(state.trailing_stop, new_stop)
                
                # Check exit conditions
                exit_position = False
                exit_reason = ""
                
                if state.position == 'LONG':
                    if current_price <= state.trailing_stop:
                        exit_reason = "Trailing Stop"
                        exit_position = True
                    elif pnl >= TAKE_PROFIT_PCT:
                        exit_reason = "Take Profit"
                        exit_position = True
                else:  # SHORT
                    if current_price >= state.trailing_stop:
                        exit_reason = "Trailing Stop"
                        exit_position = True
                    elif pnl >= TAKE_PROFIT_PCT:
                        exit_reason = "Take Profit"
                        exit_position = True
                
                if exit_position:
                    print(f"Exit signal: {exit_reason}")
                    side = SIDE_SELL if state.position == 'LONG' else SIDE_BUY
                    order = execute_order(side, POSITION_SIZE)
                    if order:
                        state.daily_pl += pnl
                        state.open_trades -= 1
                        print(f"Closed {state.position} position, P&L: {pnl*100:.2f}%")
                        state.position = None
                        state.entry_price = 0
                        state.trailing_stop = 0
                
                # Check daily drawdown limit
                if state.daily_pl <= -MAX_DAILY_DRAWDOWN:
                    print(f"Daily drawdown limit reached ({MAX_DAILY_DRAWDOWN*100}%). Stopping trading for today.")
                    state.daily_trades = MAX_DAILY_TRADES
            
            print("=" * 50)
            time.sleep(60)  # Check every minute
            
        except Exception as e:
            print(f"Error occurred: {e}")
            time.sleep(60)

if __name__ == '__main__':
    main()
