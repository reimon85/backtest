import pandas as pd
import numpy as np
from src.robustness import MonteCarloAnalyzer, WalkForwardOptimizer


# ── Monte Carlo ──────────────────────────────────────────────────────

def test_monte_carlo_analyzer():
    """Backward-compatible: simulate returns self, final_returns gives Series."""
    returns = pd.Series([0.01, -0.01, 0.02, -0.02, 0.005, -0.005])

    analyzer = MonteCarloAnalyzer(returns)
    result = analyzer.simulate(num_simulations=100)

    assert result is analyzer  # returns self
    simulations = analyzer.final_returns
    assert len(simulations) == 100
    assert 0.9 <= simulations.mean() <= 1.1


def test_monte_carlo_summary():
    returns = pd.Series([0.01, -0.01, 0.02, -0.02, 0.005, -0.005])
    analyzer = MonteCarloAnalyzer(returns).simulate(num_simulations=200)

    summary = analyzer.summary(original_final_return=1.02)

    assert isinstance(summary, dict)
    for key in ("median_return", "p5", "p95", "risk_of_ruin", "dd_ci95"):
        assert key in summary
        assert isinstance(summary[key], float)
    assert "original_vs_median" in summary
    assert isinstance(summary["original_vs_median"], float)


def test_monte_carlo_risk_of_ruin():
    """With strongly negative returns over many steps, risk of ruin should be > 0."""
    returns = pd.Series([-0.05, -0.04, -0.03, -0.06, -0.02, -0.07])
    analyzer = MonteCarloAnalyzer(returns).simulate(num_simulations=500, sample_size=50)

    ruin = analyzer.risk_of_ruin(threshold=-0.30)
    assert isinstance(ruin, float)
    assert ruin > 0  # many negative draws → high ruin


def test_monte_carlo_confidence_interval():
    returns = pd.Series([0.01, -0.01, 0.02, -0.02, 0.005, -0.005])
    analyzer = MonteCarloAnalyzer(returns).simulate(num_simulations=200, sample_size=50)

    dd_ci = analyzer.confidence_interval_drawdown(ci=0.95)
    assert isinstance(dd_ci, float)
    assert dd_ci < 0  # drawdown is negative with enough steps


# ── WalkForward — basic ─────────────────────────────────────────────

def _make_data(n=100, seed=42):
    np.random.seed(seed)
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    return pd.DataFrame({"close": np.cumsum(np.random.randn(n)) + 100}, index=dates)


def test_walk_forward_no_optimization():
    """WalkForward without param_grid: uses empty params, returns trade-level cols."""
    data = _make_data()

    def run_fn(df, params):
        # Each pct_change value ≠ 0 is treated as a "trade"
        return df["close"].pct_change().dropna()

    wf = WalkForwardOptimizer(data, run_fn)
    results = wf.run(train_size=30, test_size=20, step=20)

    assert len(results) > 0
    for col in ("oos_pnl", "oos_num_trades", "oos_wr", "oos_pf", "best_params"):
        assert col in results.columns
    assert results.iloc[0]["best_params"] == {}


def test_walk_forward_with_optimization():
    """WalkForward with param_grid: selects best params per window."""
    data = _make_data()

    def run_fn(df, params):
        scale = params.get("scale", 1)
        return df["close"].pct_change().dropna() * scale

    param_grid = {"scale": [0.5, 1.0, 2.0]}
    wf = WalkForwardOptimizer(data, run_fn, param_grid=param_grid)
    results = wf.run(train_size=30, test_size=20, step=20)

    assert len(results) > 0
    for bp in results["best_params"]:
        assert "scale" in bp
        assert bp["scale"] in [0.5, 1.0, 2.0]


def test_walk_forward_backward_compat_series():
    """run_fn returning pd.Series still works (non-zero values → trades)."""
    data = _make_data()

    def run_fn(df, params):
        return df["close"].pct_change().dropna()

    wf = WalkForwardOptimizer(data, run_fn)
    results = wf.run(train_size=30, test_size=20, step=20)

    assert len(results) > 0
    assert results["oos_num_trades"].sum() > 0


# ── WalkForward — dict interface ────────────────────────────────────

def _make_wf_with_trades(positive_ratio=0.6, num_trades_per_fold=15, n_folds=6):
    """Helper: builds a WFO with a controlled trade generator.

    Returns a ready-to-query WalkForwardOptimizer (already .run()'d).
    """
    np.random.seed(0)
    # Need enough bars: train_size + test_size * n_folds (roughly)
    n = 30 + 20 * n_folds + 20
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    data = pd.DataFrame({"close": np.cumsum(np.random.randn(n)) + 100}, index=dates)

    def run_fn(df, params):
        np.random.seed(hash(str(df.index[0])) % 2**31)
        wins = int(num_trades_per_fold * positive_ratio)
        losses = num_trades_per_fold - wins
        trades = [abs(np.random.normal(50, 10)) for _ in range(wins)] + \
                 [-abs(np.random.normal(30, 10)) for _ in range(losses)]
        np.random.shuffle(trades)
        return {"trades": trades}

    wf = WalkForwardOptimizer(data, run_fn)
    wf.run(train_size=30, test_size=20, step=20)
    return wf


def test_walk_forward_compound_metrics():
    """compound_metrics() returns correct structure and types."""
    wf = _make_wf_with_trades()
    m = wf.compound_metrics()

    for key in ("pf_oos", "pnl_oos", "total_trades_oos", "wr_oos",
                "worst_fold_pnl", "folds_positive_pct", "quarterly_pf"):
        assert key in m, f"Missing key: {key}"
    assert isinstance(m["pf_oos"], float)
    assert isinstance(m["total_trades_oos"], int)
    assert isinstance(m["quarterly_pf"], dict)
    assert m["total_trades_oos"] > 0


def test_walk_forward_acceptance_pass():
    """Positive-biased trades → should pass acceptance."""
    wf = _make_wf_with_trades(positive_ratio=0.7, num_trades_per_fold=20, n_folds=6)
    acc = wf.acceptance()

    assert "verdict" in acc
    assert "obligatory" in acc
    assert "consistency" in acc
    assert "risk_pct" in acc
    assert "metrics" in acc
    # With 70% WR and bigger wins than losses, most criteria should pass
    assert acc["obligatory"]["pnl_gt_0"] is True
    assert acc["consistency"]["wr_gt_35pct"] is True


def test_walk_forward_acceptance_blocked():
    """Negative-biased trades → BLOQUEADA."""
    wf = _make_wf_with_trades(positive_ratio=0.2, num_trades_per_fold=20, n_folds=6)
    acc = wf.acceptance()

    # PnL should be negative with only 20% winners and smaller wins
    assert acc["obligatory"]["pnl_gt_0"] is False
    assert acc["verdict"] == "BLOQUEADA"
    assert acc["risk_pct"] == 0.0
