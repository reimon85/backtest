import pandas as pd
import numpy as np
from src.robustness import MonteCarloAnalyzer

def test_monte_carlo_analyzer():
    # Symmetric returns (mean 0)
    returns = pd.Series([0.01, -0.01, 0.02, -0.02, 0.005, -0.005])
    
    analyzer = MonteCarloAnalyzer(returns)
    simulations = analyzer.simulate(num_simulations=100)
    
    # We should have 100 results
    assert len(simulations) == 100
    # Mean simulation result should be around 1.0 (no drift)
    assert 0.9 <= simulations.mean() <= 1.1
