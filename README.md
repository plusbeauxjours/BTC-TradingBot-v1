# 🤖 Bitcoin Auto Trading Bot | 비트코인 자동매매 봇

<div align="center">

[English](#-bitcoin-auto-trading-bot) | [한국어](#-비트코인-자동매매-봇)

</div>

---

## 🌎 ENGLISH

### 📋 Overview

This project is an automated Bitcoin trading bot based on RSI and MACD indicators. It uses the Binance API to automate BTC/USDT trading.

### ✨ Key Features

- 📊 **Real-time Market Monitoring**: Real-time market analysis using RSI and MACD indicators
- 🔄 **Automated Trading**: Automatic buy/sell execution based on configured strategies
- 📈 **Backtesting**: Strategy testing and performance analysis with historical data
- 🖥️ **Simulation Mode**: Monitor trading signals without using real funds

### 🛠️ Installation

1. Clone this repository:

   ```bash
   git clone <repository-url>
   cd AutoTradingBitcoinV1
   ```

2. Install required packages:

   ```bash
   pip install -r requirements.txt
   ```

3. Set up environment variables:
   - Copy the `env.example` file to create a `.env` file:
     ```bash
     cp env.example .env
     ```
   - Open the `.env` file and enter your Binance API key and secret.

### 🔑 How to Get API Keys

1. Log in to the [Binance official website](https://www.binance.com).
2. Go to Account -> API Management.
3. Click the "Create API" button and complete the security verification.
4. Enter the newly created API key and secret in your `.env` file.
5. **Important**: It is recommended to only grant appropriate permissions (read, spot trading) to the API key and disable withdrawal permissions.

### 📱 Usage

#### 🔍 Monitoring Mode

Monitor trading signals without actual trading:

```bash
python monitor.py
```

#### 🤝 Automated Trading Mode

Execute automated trading with real funds:

```bash
python main.py
```

#### 📊 Backtesting

Test the profitability of your strategy with historical data:

```bash
python backtest.py
```

### ⚙️ Trading Strategy Configuration

You can adjust the strategy by modifying the following parameters in the `config.py` file:

- `SYMBOL`: Trading symbol (default: BTCUSDT)
- `INTERVAL`: Chart time interval (default: 1h)
- `RSI_PERIOD`: RSI calculation period (default: 14)
- `RSI_OVERSOLD`: RSI oversold threshold (default: 40)
- `RSI_OVERBOUGHT`: RSI overbought threshold (default: 60)
- `MACD_FAST`, `MACD_SLOW`, `MACD_SIGNAL`: MACD settings
- `POSITION_SIZE`: Trade size (in BTC units)
- `STOP_LOSS_PCT`: Stop-loss percentage (default: 2%)
- `TAKE_PROFIT_PCT`: Take-profit percentage (default: 3%)

### ⚠️ Caution

- This bot is provided for educational and research purposes.
- Cryptocurrency trading involves high risk. Only trade with funds you can afford to lose.
- Before actual trading, thoroughly validate with backtesting and monitoring mode.
- Keep your API keys and secrets secure, and only use API keys without withdrawal permissions.

### 👥 How to Contribute

1. Fork this repository.
2. Create a new feature branch (`git checkout -b feature/amazing-feature`).
3. Commit your changes (`git commit -m 'Add some amazing feature'`).
4. Push to the branch (`git push origin feature/amazing-feature`).
5. Submit a Pull Request.

### 📜 License

This project is distributed under the MIT license.

---

## 🌎 한국어

### 📋 개요

이 프로젝트는 RSI와 MACD 지표를 기반으로 한 비트코인 자동매매 봇입니다. Binance API를 이용하여 BTC/USDT 거래를 자동화합니다.

### ✨ 주요 기능

- 📊 **실시간 시장 모니터링**: RSI 및 MACD 지표를 활용한 실시간 시장 분석
- 🔄 **자동 매매**: 설정된 전략에 따른 자동 매수/매도 실행
- 📈 **백테스팅**: 과거 데이터를 활용한 전략 테스트 및 성과 분석
- 🖥️ **시뮬레이션 모드**: 실제 자금을 사용하지 않고 매매 신호 모니터링

### 🛠️ 설치 방법

1. 이 저장소를 클론합니다:

   ```bash
   git clone <repository-url>
   cd AutoTradingBitcoinV1
   ```

2. 필요한 패키지를 설치합니다:

   ```bash
   pip install -r requirements.txt
   ```

3. 환경 변수 설정:
   - `env.example` 파일을 복사하여 `.env` 파일을 생성합니다:
     ```bash
     cp env.example .env
     ```
   - `.env` 파일을 열어 Binance API 키와 시크릿을 입력합니다.

### 🔑 API 키 발급 방법

1. [Binance 공식 사이트](https://www.binance.com)에 로그인합니다.
2. 계정 -> API 관리로 이동합니다.
3. "Create API" 버튼을 클릭하고 보안 인증을 완료합니다.
4. 새로 생성된 API 키와 시크릿을 `.env` 파일에 입력합니다.
5. **중요**: API 키에 적절한 권한(읽기, 현물 거래)만 부여하고 출금 권한은 비활성화하는 것을 권장합니다.

### 📱 사용 방법

#### 🔍 모니터링 모드

실제 거래 없이 매매 신호만 모니터링합니다:

```bash
python monitor.py
```

#### 🤝 자동매매 모드

실제 자금으로 자동 매매를 실행합니다:

```bash
python main.py
```

#### 📊 백테스트

과거 데이터로 전략의 수익성을 테스트합니다:

```bash
python backtest.py
```

### ⚙️ 매매 전략 설정

`config.py` 파일에서 다음 매개변수를 수정하여 전략을 조정할 수 있습니다:

- `SYMBOL`: 거래 심볼 (기본값: BTCUSDT)
- `INTERVAL`: 차트 시간 간격 (기본값: 1h)
- `RSI_PERIOD`: RSI 계산 기간 (기본값: 14)
- `RSI_OVERSOLD`: RSI 과매도 기준 (기본값: 40)
- `RSI_OVERBOUGHT`: RSI 과매수 기준 (기본값: 60)
- `MACD_FAST`, `MACD_SLOW`, `MACD_SIGNAL`: MACD 설정
- `POSITION_SIZE`: 거래 규모 (BTC 단위)
- `STOP_LOSS_PCT`: 손절 비율 (기본값: 2%)
- `TAKE_PROFIT_PCT`: 익절 비율 (기본값: 3%)

### ⚠️ 주의 사항

- 이 봇은 교육 및 연구 목적으로 제공됩니다.
- 암호화폐 거래에는 높은 위험이 따릅니다. 감당할 수 있는 자금으로만 거래하세요.
- 실제 거래 전에 반드시 백테스트와 모니터링 모드로 충분히 검증하세요.
- API 키와 시크릿은 안전하게 보관하고, 출금 권한이 없는 API 키만 사용하세요.

### 👥 기여 방법

1. 이 저장소를 포크합니다.
2. 새 기능 브랜치를 생성합니다 (`git checkout -b feature/amazing-feature`).
3. 변경 사항을 커밋합니다 (`git commit -m 'Add some amazing feature'`).
4. 브랜치에 푸시합니다 (`git push origin feature/amazing-feature`).
5. Pull Request를 제출합니다.

### 📜 라이센스

이 프로젝트는 MIT 라이센스 하에 배포됩니다.

# BTC-TradingBot-v1
