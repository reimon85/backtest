# Backtesting Framework

A production-grade backtesting system focusing on reliability and bias mitigation.

## Architecture

1.  **Event-Driven Engine:** Modular architecture using `Strategy`, `Portfolio`, and `ExecutionModel` for high-fidelity simulations.
2.  **Vectorized Engine:** High-performance alternative using NumPy and Pandas for rapid calculations.
3.  **Walk-Forward Optimizer:** Layer for parameter optimization and rolling window testing.
4.  **Monte Carlo Analyzer:** Statistical component for robustness testing using bootstrapping.

## Key Features

- **Realistic Execution:** Market, Limit, and Stop orders with slippage and commission models.
- **Comprehensive Metrics:** Sharpe, Sortino, Calmar ratios, and drawdown analysis.
- **Robustness Testing:** Walk-forward analysis and probability-of-loss simulations.
- **Portfolio Management:** Tracking of positions, equity, and PnL.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

(Coming Soon)
