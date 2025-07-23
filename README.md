# Algorithmic-Trading-Strategy
A momentum + mean reversion trading strategy with risk management and backtesting.

What It Does
This strategy combines two approaches:

Momentum: Buy stocks trending up over 6 months
Mean Reversion: Buy oversold stocks (low RSI, below Bollinger Bands)

It automatically switches between these based on market volatility. In low-vol markets, it favors momentum. In high-vol markets, it goes more contrarian.
Features

Dynamic signal weighting based on volatility regime
Position sizing adjusted for individual stock volatility
Drawdown protection (stops trading if down >15%)
Transaction cost modeling
Comprehensive performance analytics

Setup
bashpip install pandas numpy yfinance matplotlib seaborn
Basic Usage
pythonfrom trading_strategy import TradingStrategy

# Define your stocks
stocks = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'NVDA']

# Run strategy
strategy = TradingStrategy(
    symbols=stocks,
    start_date="2020-01-01", 
    end_date="2024-01-01",
    capital=100000
)

strategy.get_data()
strategy.make_signals() 
strategy.backtest()

# See results
print(strategy.report())
strategy.plot_results()
How It Works
Signal Generation

Momentum Signal: 6-month return > 5%, 3-month return > 2%, not in high-vol regime
Mean Reversion Signal: RSI < 30 OR price below lower Bollinger Band
Exit Signal: RSI > 75 OR price above upper Bollinger Band OR high volatility

Position Sizing

Base allocation: 15% max per stock
Volatility adjustment: target 15% individual position volatility
Drawdown scaling: reduce size if portfolio is down
Recent momentum: small boost for recent winners

Risk Management

Max 15% drawdown before going to cash
Individual position limits
Transaction cost deduction
Stop loss mechanisms

Configuration
Key parameters you can adjust:
pythonstrategy = TradingStrategy(
    symbols=your_stocks,
    capital=100000,        # starting money
    max_pos=0.15,         # max 15% per position  
    txn_cost=0.001        # 10 bps transaction cost
)

# Strategy parameters
strategy.mom_window = 126     # momentum lookback (6 months)
strategy.mr_window = 20       # mean reversion lookback  
strategy.max_dd = 0.15        # max drawdown before stopping
Performance Metrics
The strategy tracks:

Total return, CAGR, Sharpe ratio
Maximum drawdown and recovery
Win rate and profit factor
VaR and CVaR for risk assessment
Transaction costs and turnover

Example Results
Typical performance on large-cap tech stocks (2020-2024):

CAGR: 12-18%
Sharpe: 0.8-1.2
Max DD: 8-15%
Win Rate: 52-58%

Files

trading_strategy.py - Main strategy code
README.md - This file
requirements.txt - Dependencies

Strategy Logic
The core insight is that different market regimes favor different approaches:

Low volatility: Trends persist, momentum works
High volatility: Markets overreact, mean reversion works

The strategy automatically detects the regime using rolling volatility percentiles and adjusts the signal weighting accordingly.

Limitations
Assumes you can trade at closing prices
No market impact costs for large orders
Performance depends heavily on the stock universe
Requires at least 2-3 years of data for proper backtesting
