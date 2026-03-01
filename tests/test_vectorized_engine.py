import pandas as pd
import numpy as np
import pytest
from src.engine_vectorized import VectorizedEngine
from src.metrics import calculate_cumulative_returns

def test_vectorized_engine_long_only():
    """Test a simple buy-and-hold strategy."""
    prices = pd.Series([100, 105, 110, 108, 115])
    signals = pd.Series([1, 1, 1, 1, 1])
    
    engine = VectorizedEngine(prices)
    strategy_returns = engine.run(signals)
    
    # Cumulative returns of buy-and-hold should match the price ratio (approx)
    cumulative_returns = calculate_cumulative_returns(strategy_returns)
    
    # Last price / first price = 1.15
    assert np.isclose(cumulative_returns.iloc[-1], 1.15)

def test_vectorized_engine_with_costs():
    """Test that transaction costs reduce returns."""
    prices = pd.Series([100, 110, 100, 110])
    signals = pd.Series([1, 0, 1, 0])
    
    engine_no_cost = VectorizedEngine(prices, transaction_cost=0.0)
    engine_with_cost = VectorizedEngine(prices, transaction_cost=0.01)
    
    returns_no_cost = engine_no_cost.run(signals)
    returns_with_cost = engine_with_cost.run(signals)
    
    assert returns_with_cost.sum() < returns_no_cost.sum()
