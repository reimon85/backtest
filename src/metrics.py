import numpy as np
import pandas as pd

def calculate_returns(price_series: pd.Series) -> pd.Series:
    """Calculate log returns of a price series."""
    return np.log(price_series / price_series.shift(1))

def calculate_cumulative_returns(returns: pd.Series) -> pd.Series:
    """Calculate cumulative returns from a series of log returns."""
    return returns.cumsum().apply(np.exp)

def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    """Calculate the Sharpe Ratio (annualized)."""
    excess_returns = returns - (risk_free_rate / 252)
    return np.sqrt(252) * excess_returns.mean() / excess_returns.std()

def calculate_sortino_ratio(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    """Calculate the Sortino Ratio (annualized)."""
    excess_returns = returns - (risk_free_rate / 252)
    downside_returns = excess_returns[excess_returns < 0]
    return np.sqrt(252) * excess_returns.mean() / downside_returns.std()

def calculate_max_drawdown(cumulative_returns: pd.Series) -> float:
    """Calculate the maximum drawdown from a cumulative returns series."""
    peak = cumulative_returns.cummax()
    drawdown = (cumulative_returns - peak) / peak
    return drawdown.min()

def calculate_win_rate(returns: pd.Series) -> float:
    """Calculate the percentage of positive return days."""
    winning_days = returns[returns > 0]
    return len(winning_days) / len(returns) if len(returns) > 0 else 0.0

def calculate_profit_factor(returns: pd.Series) -> float:
    """Calculate the ratio of gross profits to gross losses."""
    gross_profit = returns[returns > 0].sum()
    gross_loss = abs(returns[returns < 0].sum())
    return gross_profit / gross_loss if gross_loss > 0 else float('inf')
