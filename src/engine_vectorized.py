import numpy as np
import pandas as pd
from .metrics import calculate_returns, calculate_cumulative_returns

class VectorizedEngine:
    """A high-performance vectorized backtesting engine for quick strategy evaluation."""
    
    def __init__(self, price_series: pd.Series, transaction_cost: float = 0.0):
        self.price_series = price_series
        self.transaction_cost = transaction_cost
        self.returns = calculate_returns(price_series)
        self.signals = None
        self.strategy_returns = None
    
    def run(self, signals: pd.Series):
        """Run the backtest given a series of signals."""
        self.signals = signals.shift(1) # Align signals to next bar's returns
        
        # Calculate raw strategy returns
        self.strategy_returns = self.signals * self.returns
        
        # Deduct transaction costs (when positions change)
        trades = self.signals.diff().abs().fillna(0)
        self.strategy_returns -= trades * self.transaction_cost
        
        return self.strategy_returns
    
    def get_summary(self):
        """Return a summary of backtest results."""
        if self.strategy_returns is None:
            return None
        
        cumulative_returns = calculate_cumulative_returns(self.strategy_returns)
        
        return {
            "Total Return": (cumulative_returns.iloc[-1] - 1) if not cumulative_returns.empty else 0,
            "Strategy Returns": self.strategy_returns,
            "Cumulative Returns": cumulative_returns
        }
