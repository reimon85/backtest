import pandas as pd
import numpy as np
from src.engine_vectorized import VectorizedEngine
from src.engine_event import EventDrivenEngine, Strategy
from src.metrics import calculate_sharpe_ratio, calculate_max_drawdown
from src.robustness import MonteCarloAnalyzer

# 1. Prepare Dummy Data
dates = pd.date_range("2023-01-01", periods=100)
prices = pd.Series(100 * np.exp(np.cumsum(np.random.normal(0, 0.01, 100))), index=dates)

print("--- Vectorized Engine Demo ---")
# 2. Vectorized Engine Example: Moving Average Crossover
short_ma = prices.rolling(5).mean()
long_ma = prices.rolling(20).mean()
signals = (short_ma > long_ma).astype(int)

vec_engine = VectorizedEngine(prices, transaction_cost=0.001)
strategy_returns = vec_engine.run(signals)
summary = vec_engine.get_summary()

print(f"Total Return: {summary['Total Return']:.2%}")
print(f"Sharpe Ratio: {calculate_sharpe_ratio(strategy_returns):.2f}")
print(f"Max Drawdown: {calculate_max_drawdown(summary['Cumulative Returns']):.2%}")

print("\n--- Event-Driven Engine Demo ---")
# 3. Event-Driven Engine Example: Simple Buy and Hold
class BuyAndHold(Strategy):
    def on_bar(self, bar):
        if self.portfolio.positions.get("Price", 0) == 0:
            price = bar["Price"]
            quantity = self.portfolio.cash // price
            self.portfolio.buy("Price", quantity, price)

data = pd.DataFrame({"Price": prices})
event_engine = EventDrivenEngine(data, BuyAndHold, initial_cash=10000.0)
equity_curve = event_engine.run(symbol="Price")

final_equity = equity_curve.iloc[-1]["equity"]
print(f"Initial Equity: 10,000.00")
print(f"Final Equity: {final_equity:,.2f}")
print(f"Total Return: {(final_equity / 10000.0 - 1):.2%}")

print("\n--- Robustness Tool Demo ---")
# 4. Monte Carlo Simulation
mc_analyzer = MonteCarloAnalyzer(strategy_returns)
sim_results = mc_analyzer.simulate(num_simulations=500)
print(f"Monte Carlo Mean Final Return: {(sim_results.mean() - 1):.2%}")
print(f"95% Confidence Interval: {(sim_results.quantile(0.025) - 1):.2%} to {(sim_results.quantile(0.975) - 1):.2%}")
