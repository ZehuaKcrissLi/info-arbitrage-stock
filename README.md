# Multi-Agent LLM Trading System

A sophisticated trading bot that mimics a professional trading team through specialized AI agents. This system implements a comprehensive multi-agent architecture for automated stock trading using large language models.

## 🏗️ System Architecture

The system follows a 7-layer multi-agent architecture:

### Layer 1: Data Collection
- **Information Monitoring Agent**: Real-time news and social media monitoring
- **Background Enhancement Agent**: RAG-based context enrichment

### Layer 2: Analysis
- **Fundamental Analysis Agent**: Financial metrics and valuation analysis
- **Technical Analysis Agent**: Chart patterns and technical indicators
- **Sentiment Analysis Agent**: Market sentiment from multiple sources
- **Macro Analysis Agent**: Economic indicators and market structure

### Layer 3: Decision Making
- **Bull Agent**: Argues for buying opportunities
- **Bear Agent**: Argues for selling or avoiding positions
- **Debate Coordinator**: Facilitates structured debates between agents

### Layer 4: Execution
- **Risk Management Agent**: Position sizing and risk validation
- **Trading Agent**: Signal generation and trade coordination
- **Execution Agent**: Order management and broker API integration

### Layer 5: Monitoring
- **Event Bus**: Redis-based inter-agent communication
- **Workflow Engine**: Process orchestration and state management

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Redis server
- API keys for trading and data providers

### Installation

1. **Clone and setup environment:**
```bash
git clone <repository-url>
cd info-arbitrage-stock
cp .env.example .env
```

2. **Configure API keys in `.env`:**
Edit the `.env` file and add your API keys (see `.env.example` for all required keys)

3. **Start the system:**
```bash
chmod +x start.sh
./start.sh
```

Or manually:
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python main.py
```

## 🔑 Required API Keys

### Essential (Core Functionality)
- **OpenAI API**: `OPENAI_API_KEY` - For LLM decision making
- **Alpaca Trading**: `ALPACA_API_KEY`, `ALPACA_SECRET_KEY` - For trade execution

### Market Data (Choose at least one)
- **Alpha Vantage**: `ALPHA_VANTAGE_API_KEY` - For fundamental data
- **Finnhub**: `FINNHUB_API_KEY` - For financial metrics
- **Polygon**: `POLYGON_API_KEY` - For market data
- **Yahoo Finance**: No key required (free tier)

### News & Sentiment (Optional but recommended)
- **News API**: `NEWS_API_KEY` - For news monitoring
- **Twitter API**: For social sentiment analysis
- **Reddit API**: For Reddit sentiment

### Notifications (Optional)
- **Slack Webhook**: `SLACK_WEBHOOK_URL` - For trade notifications

## 💡 Usage Examples

### Basic Startup
```bash
python main.py
```

### Development Mode with Example Events
```bash
ENVIRONMENT=development python main.py
```

### Paper Trading (Recommended for testing)
```bash
ALPACA_PAPER_TRADING=true python main.py
```

### Simulation Mode (No real API calls)
```bash
SIMULATION_MODE=true python main.py
```

## 🛡️ Safety Features

- **Paper Trading**: Test with virtual money before going live
- **Risk Management**: Multiple validation layers and position limits
- **Circuit Breakers**: Automatic shutdown on excessive losses
- **Real-time Monitoring**: Comprehensive logging and alerts

## 📊 Agent Responsibilities

| Agent | Primary Function | Key Capabilities |
|-------|-----------------|------------------|
| Monitoring | Data Collection | News feeds, social media, market events |
| Background | Context Enhancement | RAG search, web research, event enrichment |
| Fundamental | Financial Analysis | P/E ratios, earnings, financial health |
| Technical | Chart Analysis | RSI, MACD, support/resistance levels |
| Sentiment | Market Sentiment | News sentiment, social media analysis |
| Macro | Economic Analysis | VIX, yields, sector rotation, risk metrics |
| Bull/Bear | Debate System | Structured arguments for/against trades |
| Risk | Risk Management | Position sizing, stop losses, circuit breakers |
| Trading | Signal Generation | Final trade decisions and coordination |
| Execution | Order Management | Broker API integration and order tracking |

## ⚙️ Configuration

Key configuration options in `.env`:

```bash
# Trading Parameters
MAX_POSITION_PCT=0.05          # Max 5% of account per position
MAX_DAILY_LOSS=1000.0          # Daily loss limit
MAX_TRADES_PER_DAY=10          # Trade frequency limit

# Risk Management
MIN_RISK_REWARD_RATIO=2.0      # Minimum risk/reward
DEFAULT_STOP_LOSS_PCT=0.02     # 2% stop loss
DEFAULT_TAKE_PROFIT_PCT=0.04   # 4% take profit

# System Settings
LOG_LEVEL=INFO                 # Logging verbosity
PAPER_TRADING=true             # Start with paper trading
SIMULATION_MODE=false          # Use real APIs
```

## 🔧 Development

### Project Structure
```
src/
├── agents/           # All AI agents
├── models/           # Data models and schemas
├── communication/    # Event bus and coordination
├── config/           # Configuration management
└── utils/            # Utility functions

config.yaml          # System configuration
requirements.txt     # Python dependencies
main.py             # Application entry point
start.sh            # Startup script
```

### Adding New Agents
1. Create agent class in `src/agents/`
2. Implement required methods: `start()`, `stop()`, `analyze()`
3. Register in `AgentCoordinator`
4. Add to event routing

### Testing
```bash
pytest tests/
```

## 📈 Trading Workflow

1. **Event Detection**: Monitor news, social media, price movements
2. **Background Research**: Enhance events with additional context
3. **Multi-Agent Analysis**: Fundamental, technical, sentiment, macro analysis
4. **Structured Debate**: Bull vs Bear agents argue the case
5. **Risk Validation**: Validate trade against risk parameters
6. **Signal Generation**: Create final trading signal
7. **Execution**: Submit orders through broker API
8. **Monitoring**: Track positions and performance

## ⚠️ Important Disclaimers

- **Start with Paper Trading**: Always test with virtual money first
- **Not Financial Advice**: This is educational/experimental software
- **Use at Your Own Risk**: Trading involves substantial risk of loss
- **Monitor Continuously**: Automated systems require human oversight
- **Regulatory Compliance**: Ensure compliance with local trading regulations

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is for educational and research purposes. Please ensure compliance with all applicable regulations when using for live trading.

## 🆘 Support

For issues, questions, or contributions:
1. Check the logs in `logs/trading_system.log`
2. Verify your `.env` configuration
3. Ensure all required services (Redis) are running
4. Create an issue with detailed error information

---

**Remember**: This system handles real money when not in simulation mode. Always start with paper trading and thoroughly test before using real funds.