import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class TradingStrategy:
    def __init__(self, symbols, start_date="2020-01-01", end_date="2024-01-01", 
                 capital=100000, txn_cost=0.001, max_pos=0.15):
        self.symbols = symbols
        self.start_date = start_date
        self.end_date = end_date
        self.capital = capital
        self.txn_cost = txn_cost
        self.max_pos = max_pos
        
        # strategy params - found these work well through testing
        self.mom_window = 126
        self.mr_window = 20
        self.vol_window = 30
        self.rsi_len = 14
        self.bb_std = 2.0
        
        # risk params
        self.max_dd = 0.15
        self.stop_pct = 0.08
        
        self.data = {}
        self.signals = {}
        self.portfolio = None
        
    def get_data(self):
        print("Getting market data...")
        
        for sym in self.symbols:
            try:
                ticker = yf.Ticker(sym)
                df = ticker.history(start=self.start_date, end=self.end_date)
                
                if len(df) < 200:
                    print(f"Not enough data for {sym}")
                    continue
                    
                df = df.dropna()
                df['ret'] = df['Close'].pct_change()
                df['log_ret'] = np.log(df['Close'] / df['Close'].shift(1))
                
                self.data[sym] = df
                print(f"Got {len(df)} days for {sym}")
                
            except Exception as e:
                print(f"Error with {sym}: {e}")
    
    def calc_indicators(self, sym):
        df = self.data[sym].copy()
        
        # momentum stuff
        df['mom_6m'] = df['Close'] / df['Close'].shift(self.mom_window) - 1
        df['mom_3m'] = df['Close'] / df['Close'].shift(63) - 1
        df['mom_1m'] = df['Close'] / df['Close'].shift(21) - 1
        
        # mean reversion
        df['sma'] = df['Close'].rolling(self.mr_window).mean()
        df['price_sma_ratio'] = df['Close'] / df['sma']
        
        # bollinger bands
        std = df['Close'].rolling(self.mr_window).std()
        df['bb_upper'] = df['sma'] + (std * self.bb_std)
        df['bb_lower'] = df['sma'] - (std * self.bb_std)
        df['bb_pos'] = (df['Close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # rsi
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0).rolling(self.rsi_len).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(self.rsi_len).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # vol
        df['vol'] = df['ret'].rolling(self.vol_window).std() * np.sqrt(252)
        df['vol_rank'] = df['vol'].rolling(252).rank(pct=True)
        
        # volume if available
        if 'Volume' in df.columns:
            df['vol_sma'] = df['Volume'].rolling(20).mean()
            df['vol_ratio'] = df['Volume'] / df['vol_sma']
        else:
            df['vol_ratio'] = 1
            
        return df
    
    def make_signals(self):
        print("Making signals...")
        
        for sym in self.data.keys():
            df = self.calc_indicators(sym)
            
            mom_sig = np.zeros(len(df))
            mr_sig = np.zeros(len(df))
            
            # momentum signal - trend following
            mom_cond = ((df['mom_6m'] > 0.05) & 
                       (df['mom_3m'] > 0.02) & 
                       (df['vol_rank'] < 0.8))
            mom_sig[mom_cond] = 1
            
            # mean reversion - contrarian
            mr_cond = ((df['rsi'] < 30) | (df['bb_pos'] < 0.1)) & (df['vol_rank'] < 0.7)
            mr_sig[mr_cond] = 1
            
            # exit signals
            exit_cond = ((df['rsi'] > 75) | 
                        (df['bb_pos'] > 0.9) | 
                        (df['vol_rank'] > 0.9))
            
            # combine based on vol regime
            regime_wt = np.where(df['vol_rank'] < 0.5, 0.7, 0.3)
            df['signal'] = regime_wt * mom_sig + (1 - regime_wt) * mr_sig
            df['signal'][exit_cond] = -1
            
            # smooth it out
            df['smooth_sig'] = df['signal'].rolling(3).mean()
            df['final_sig'] = np.where(df['smooth_sig'] > 0.3, 1,
                                      np.where(df['smooth_sig'] < -0.3, -1, 0))
            
            self.signals[sym] = df
    
    def calc_position_size(self, sym, signal, price, port_val, dd):
        if signal == 0:
            return 0
        
        base_size = self.max_pos
        
        # vol adjustment
        curr_vol = self.signals[sym]['vol'].iloc[-1]
        target_vol = 0.15
        vol_adj = min(target_vol / curr_vol, 2.0)
        
        # drawdown adjustment
        dd_adj = max(0.5, 1 - (dd / self.max_dd))
        
        # momentum boost
        recent_ret = self.signals[sym]['ret'].tail(20).mean()
        mom_adj = 1 + np.tanh(recent_ret * 100) * 0.2
        
        final_size = base_size * vol_adj * dd_adj * mom_adj * abs(signal)
        return min(final_size, self.max_pos)
    
    def backtest(self):
        print("Running backtest...")
        
        all_dates = set()
        for sym in self.signals.keys():
            all_dates.update(self.signals[sym].index)
        all_dates = sorted(list(all_dates))
        
        results = []
        positions = {sym: 0 for sym in self.symbols}
        cash = self.capital
        port_val = self.capital
        max_val = self.capital
        
        for date in all_dates:
            daily_pnl = 0
            turnover = 0
            
            # calc returns on existing positions
            for sym in positions.keys():
                if (sym in self.signals and date in self.signals[sym].index and positions[sym] != 0):
                    ret = self.signals[sym].loc[date, 'ret']
                    pos_val = positions[sym] * port_val
                    pos_pnl = pos_val * ret
                    daily_pnl += pos_pnl
            
            port_val += daily_pnl
            max_val = max(max_val, port_val)
            dd = (max_val - port_val) / max_val
            
            # new positions
            new_pos = {}
            for sym in self.symbols:
                if sym in self.signals and date in self.signals[sym].index:
                    signal = self.signals[sym].loc[date, 'final_sig']
                    price = self.signals[sym].loc[date, 'Close']
                    
                    if dd > self.max_dd:
                        signal = 0  # go to cash
                    
                    target_pos = self.calc_position_size(sym, signal, price, port_val, dd)
                    new_pos[sym] = target_pos
                    
                    # turnover calc
                    pos_change = abs(target_pos - positions.get(sym, 0))
                    turnover += pos_change * port_val
            
            # transaction costs
            costs = turnover * self.txn_cost
            port_val -= costs
            
            positions.update(new_pos)
            
            results.append({
                'Date': date,
                'port_val': port_val,
                'pnl': daily_pnl,
                'ret': daily_pnl / (port_val - daily_pnl) if port_val != daily_pnl else 0,
                'dd': dd,
                'costs': costs,
                'turnover': turnover,
                **{f'{sym}_pos': positions.get(sym, 0) for sym in self.symbols}
            })
        
        self.portfolio = pd.DataFrame(results).set_index('Date')
        print(f"Done. Final value: ${port_val:,.2f}")
    
    def get_metrics(self):
        if self.portfolio is None:
            return None
        
        rets = self.portfolio['ret'].dropna()
        vals = self.portfolio['port_val']
        
        total_ret = (vals.iloc[-1] / self.capital) - 1
        cagr = (vals.iloc[-1] / self.capital) ** (252 / len(rets)) - 1
        vol = rets.std() * np.sqrt(252)
        sharpe = (cagr - 0.02) / vol
        
        # drawdown
        cum_rets = (1 + rets).cumprod()
        running_max = cum_rets.expanding().max()
        dd_series = (cum_rets - running_max) / running_max
        max_dd = dd_series.min()
        
        sortino = (cagr - 0.02) / (rets[rets < 0].std() * np.sqrt(252))
        win_rate = (rets > 0).sum() / len(rets)
        profit_factor = rets[rets > 0].sum() / abs(rets[rets < 0].sum())
        
        var95 = np.percentile(rets, 5)
        cvar95 = rets[rets <= var95].mean()
        
        avg_turnover = self.portfolio['turnover'].mean()
        total_costs = self.portfolio['costs'].sum()
        
        return {
            'total_return': total_ret,
            'cagr': cagr,
            'volatility': vol,
            'sharpe': sharpe,
            'sortino': sortino,
            'max_drawdown': max_dd,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'var95': var95,
            'cvar95': cvar95,
            'avg_turnover': avg_turnover,
            'total_costs': total_costs
        }
    
    def plot_results(self):
        if self.portfolio is None:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # portfolio value
        axes[0, 0].plot(self.portfolio.index, self.portfolio['port_val'], 'b-', linewidth=2)
        axes[0, 0].axhline(y=self.capital, color='r', linestyle='--', alpha=0.7)
        axes[0, 0].set_title('Portfolio Value')
        axes[0, 0].grid(True, alpha=0.3)
        
        # drawdowns
        dd = self.portfolio['dd'] * 100
        axes[0, 1].fill_between(dd.index, dd, 0, alpha=0.3, color='red')
        axes[0, 1].plot(dd.index, dd, 'darkred', linewidth=1)
        axes[0, 1].set_title('Drawdown %')
        axes[0, 1].grid(True, alpha=0.3)
        
        # rolling sharpe
        rolling_rets = self.portfolio['ret'].rolling(63)
        rolling_sharpe = (rolling_rets.mean() * 252) / (rolling_rets.std() * np.sqrt(252))
        axes[1, 0].plot(rolling_sharpe.index, rolling_sharpe, 'g-', linewidth=2)
        axes[1, 0].axhline(y=1, color='orange', linestyle='--')
        axes[1, 0].set_title('Rolling 3M Sharpe')
        axes[1, 0].grid(True, alpha=0.3)
        
        # monthly returns heatmap
        monthly = self.portfolio['ret'].resample('M').apply(lambda x: (1 + x).prod() - 1)
        monthly_pivot = monthly.groupby([monthly.index.year, monthly.index.month]).first().unstack()
        
        sns.heatmap(monthly_pivot * 100, annot=True, fmt='.1f', cmap='RdYlGn', 
                   center=0, ax=axes[1, 1])
        axes[1, 1].set_title('Monthly Returns %')
        
        plt.tight_layout()
        plt.show()
    
    def report(self):
        metrics = self.get_metrics()
        if not metrics:
            return "No results available"
        
        return f"""
TRADING STRATEGY RESULTS
========================

Portfolio: {', '.join(self.symbols)}
Period: {self.start_date} to {self.end_date}
Starting Capital: ${self.capital:,.2f}
Ending Value: ${self.portfolio['port_val'].iloc[-1]:,.2f}

Performance:
- Total Return: {metrics['total_return']:.2%}
- CAGR: {metrics['cagr']:.2%}
- Volatility: {metrics['volatility']:.2%}
- Sharpe: {metrics['sharpe']:.2f}
- Sortino: {metrics['sortino']:.2f}
- Max Drawdown: {metrics['max_drawdown']:.2%}

Risk:
- VaR (95%): {metrics['var95']:.2%}
- CVaR (95%): {metrics['cvar95']:.2%}
- Win Rate: {metrics['win_rate']:.2%}
- Profit Factor: {metrics['profit_factor']:.2f}

Trading:
- Avg Daily Turnover: ${metrics['avg_turnover']:,.2f}
- Total Costs: ${metrics['total_costs']:,.2f}
"""

def main():
    # stock universe
    tickers = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'NVDA', 'META', 'AMZN', 'NFLX']
    
    strategy = TradingStrategy(
        symbols=tickers,
        start_date="2020-01-01",
        end_date="2024-01-01",
        capital=100000,
        txn_cost=0.001,
        max_pos=0.15
    )
    
    strategy.get_data()
    strategy.make_signals()
    strategy.backtest()
    
    print(strategy.report())
    strategy.plot_results()

if __name__ == "__main__":
    main()
