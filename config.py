import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Binance API configuration
BINANCE_API_KEY = os.getenv('BINANCE_API_KEY')
BINANCE_API_SECRET = os.getenv('BINANCE_API_SECRET')

# Trading parameters
SYMBOL = 'BTCUSDT'
INTERVAL = '1h'      # 1시간 봉
RSI_PERIOD = 14
RSI_OVERSOLD = 40
RSI_OVERBOUGHT = 60
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
POSITION_SIZE = 0.001  # BTC 단위
STOP_LOSS_PCT = 0.02   # 2%
TAKE_PROFIT_PCT = 0.03 # 3%

# Validate API keys
def validate_api_keys():
    if not BINANCE_API_KEY or BINANCE_API_KEY == 'your_api_key_here':
        print("경고: Binance API 키가 설정되지 않았습니다. .env 파일을 확인하세요.")
        return False
    
    if not BINANCE_API_SECRET or BINANCE_API_SECRET == 'your_api_secret_here':
        print("경고: Binance API 시크릿이 설정되지 않았습니다. .env 파일을 확인하세요.")
        return False
    
    return True 