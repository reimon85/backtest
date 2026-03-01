import pandas as pd
import pytest
from src.engine_event import EventDrivenEngine, Strategy, Portfolio

class BuyAndHoldStrategy(Strategy):
    def on_bar(self, bar):
        # Only buy if we don't have a position
        if self.portfolio.positions.get("AAPL", 0) == 0:
            price = bar["AAPL"]
            quantity = self.portfolio.cash // price
            self.portfolio.buy("AAPL", quantity, price)

def test_event_driven_engine_basic():
    data = pd.DataFrame({
        "AAPL": [100, 105, 110, 108, 115]
    }, index=pd.date_range("2023-01-01", periods=5))
    
    engine = EventDrivenEngine(data, BuyAndHoldStrategy, initial_cash=1000.0)
    equity_curve = engine.run(symbol="AAPL")
    
    # Initial equity should be 1000
    # Final equity should be roughly 1000 * (115/100) = 1150
    assert equity_curve.iloc[0]["equity"] == 1000
    assert equity_curve.iloc[-1]["equity"] >= 1140 # Account for cash left over
