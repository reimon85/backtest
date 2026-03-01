import itertools
import numpy as np
import pandas as pd
from .metrics import calculate_cumulative_returns, calculate_sharpe_ratio, calculate_max_drawdown


class MonteCarloAnalyzer:
    """Simulates multiple return paths using bootstrapping for robustness testing."""

    def __init__(self, returns: pd.Series):
        self.returns = returns.dropna()
        self.equity_curves: np.ndarray | None = None

    def simulate(self, num_simulations: int = 1000, sample_size: int | None = None):
        """Perform Monte Carlo simulation on returns.

        Stores full equity curves and returns self for chaining.
        """
        if sample_size is None:
            sample_size = len(self.returns)

        curves = np.empty((num_simulations, sample_size))
        for i in range(num_simulations):
            bootstrapped = np.random.choice(self.returns.values, size=sample_size, replace=True)
            curves[i] = np.exp(np.cumsum(bootstrapped))

        self.equity_curves = curves
        return self

    @property
    def final_returns(self) -> pd.Series:
        """Final cumulative value for each simulation (backward-compatible)."""
        if self.equity_curves is None:
            raise RuntimeError("Call simulate() first")
        return pd.Series(self.equity_curves[:, -1])

    def risk_of_ruin(self, threshold: float = -0.30) -> float:
        """Fraction of simulations where equity drops more than threshold from peak."""
        if self.equity_curves is None:
            raise RuntimeError("Call simulate() first")
        count = 0
        for curve in self.equity_curves:
            peak = np.maximum.accumulate(curve)
            dd = (curve - peak) / peak
            if dd.min() <= threshold:
                count += 1
        return count / len(self.equity_curves)

    def confidence_interval_drawdown(self, ci: float = 0.95) -> float:
        """Percentile ci of max drawdown across all simulations."""
        if self.equity_curves is None:
            raise RuntimeError("Call simulate() first")
        max_dds = []
        for curve in self.equity_curves:
            peak = np.maximum.accumulate(curve)
            dd = (curve - peak) / peak
            max_dds.append(dd.min())
        return float(np.percentile(max_dds, ci * 100))

    def summary(self, original_final_return: float | None = None) -> dict:
        """Summary statistics of the simulation."""
        finals = self.final_returns
        result = {
            "median_return": float(finals.median()),
            "p5": float(finals.quantile(0.05)),
            "p95": float(finals.quantile(0.95)),
            "risk_of_ruin": self.risk_of_ruin(),
            "dd_ci95": self.confidence_interval_drawdown(),
        }
        if original_final_return is not None:
            result["original_vs_median"] = original_final_return - result["median_return"]
        return result


class WalkForwardOptimizer:
    """Walk-forward analysis aligned with PRODUCTION_STANDARD.md.

    Parameters
    ----------
    data : pd.DataFrame
        Full dataset (index must be datetime-like).
    run_fn : callable
        ``run_fn(data, params)`` returns either:
        - ``dict`` with key ``"trades"``: ``list[float]`` of per-trade PnL, or
        - ``pd.Series`` (backward compat): non-zero values treated as trades.
    param_grid : dict | None
        ``{"param_name": [v1, v2, ...], ...}`` for IS grid search.
        If None, runs with ``params={}`` (no re-optimization — production mode).
    """

    def __init__(self, data: pd.DataFrame, run_fn, param_grid: dict | None = None):
        self.data = data
        self.run_fn = run_fn
        self.param_grid = param_grid
        self._results: list[dict] | None = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _parse_run_result(self, result) -> list[float]:
        """Normalize run_fn output to a list of per-trade PnL."""
        if isinstance(result, dict):
            return list(result["trades"])
        if isinstance(result, pd.Series):
            return result[result != 0].tolist()
        raise TypeError("run_fn must return dict with 'trades' key or pd.Series")

    def _grid_combinations(self):
        keys = list(self.param_grid.keys())
        values = list(self.param_grid.values())
        for combo in itertools.product(*values):
            yield dict(zip(keys, combo))

    @staticmethod
    def _trade_metrics(trades: list[float]) -> dict:
        """Compute PnL, WR, PF from a list of per-trade PnL."""
        if not trades:
            return {"pnl": 0.0, "num_trades": 0, "wr": 0.0, "pf": 0.0}
        gross_profit = sum(t for t in trades if t > 0)
        gross_loss = abs(sum(t for t in trades if t < 0))
        return {
            "pnl": sum(trades),
            "num_trades": len(trades),
            "wr": len([t for t in trades if t > 0]) / len(trades),
            "pf": gross_profit / gross_loss if gross_loss > 0 else float("inf"),
        }

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(self, train_size: int, test_size: int, step: int):
        """Execute walk-forward analysis.

        Returns a DataFrame with per-fold OOS metrics.
        Full trade lists are kept internally for ``compound_metrics()``.
        """
        results = []
        total_len = len(self.data)

        start = 0
        while start + train_size + test_size <= total_len:
            train_data = self.data.iloc[start: start + train_size]
            test_data = self.data.iloc[start + train_size: start + train_size + test_size]

            # IS optimization (if grid provided)
            best_params: dict = {}
            if self.param_grid:
                best_sharpe = -np.inf
                for combo in self._grid_combinations():
                    is_trades = self._parse_run_result(self.run_fn(train_data, combo))
                    if len(is_trades) > 1:
                        sharpe = calculate_sharpe_ratio(pd.Series(is_trades))
                        if np.isfinite(sharpe) and sharpe > best_sharpe:
                            best_sharpe = sharpe
                            best_params = combo

            # IS evaluation
            is_trades = self._parse_run_result(self.run_fn(train_data, best_params))
            is_m = self._trade_metrics(is_trades)

            # OOS evaluation
            oos_trades = self._parse_run_result(self.run_fn(test_data, best_params))
            oos_m = self._trade_metrics(oos_trades)

            results.append({
                "window_start": test_data.index[0],
                "window_end": test_data.index[-1],
                "best_params": best_params,
                "is_pnl": is_m["pnl"],
                "oos_pnl": oos_m["pnl"],
                "oos_num_trades": oos_m["num_trades"],
                "oos_wr": oos_m["wr"],
                "oos_pf": oos_m["pf"],
                "_oos_trades": oos_trades,  # internal, stripped from display
            })

            start += step

        self._results = results

        # Return clean DataFrame (without raw trade lists)
        display = [{k: v for k, v in r.items() if k != "_oos_trades"} for r in results]
        return pd.DataFrame(display)

    # ------------------------------------------------------------------
    # Compound metrics (PRODUCTION_STANDARD §5)
    # ------------------------------------------------------------------

    def compound_metrics(self) -> dict:
        """Aggregate OOS metrics across all folds."""
        if self._results is None:
            raise RuntimeError("Call run() first")

        all_trades: list[float] = []
        for fold in self._results:
            all_trades.extend(fold["_oos_trades"])

        m = self._trade_metrics(all_trades)
        fold_pnls = [f["oos_pnl"] for f in self._results]

        # Quarterly PF
        quarterly_pf: dict[str, float] = {}
        q_buckets: dict[str, dict] = {}
        for fold in self._results:
            ws = fold["window_start"]
            q = f"{ws.year}Q{(ws.month - 1) // 3 + 1}"
            if q not in q_buckets:
                q_buckets[q] = {"profit": 0.0, "loss": 0.0}
            for t in fold["_oos_trades"]:
                if t > 0:
                    q_buckets[q]["profit"] += t
                else:
                    q_buckets[q]["loss"] += abs(t)
        for q, v in q_buckets.items():
            quarterly_pf[q] = v["profit"] / v["loss"] if v["loss"] > 0 else float("inf")

        return {
            "pf_oos": m["pf"],
            "pnl_oos": m["pnl"],
            "total_trades_oos": m["num_trades"],
            "wr_oos": m["wr"],
            "worst_fold_pnl": min(fold_pnls) if fold_pnls else 0.0,
            "folds_positive_pct": (
                len([p for p in fold_pnls if p > 0]) / len(fold_pnls)
                if fold_pnls else 0.0
            ),
            "quarterly_pf": quarterly_pf,
        }

    # ------------------------------------------------------------------
    # Acceptance criteria (PRODUCTION_STANDARD §5)
    # ------------------------------------------------------------------

    def acceptance(self, worst_fold_threshold: float = -500) -> dict:
        """Evaluate acceptance criteria per PRODUCTION_STANDARD.md §5.

        Parameters
        ----------
        worst_fold_threshold : float
            Absolute PnL floor for worst fold (default -500 pts).
        """
        metrics = self.compound_metrics()

        obligatory = {
            "pf_gt_1.05": metrics["pf_oos"] > 1.05,
            "pnl_gt_0": metrics["pnl_oos"] > 0,
            "trades_gte_50": metrics["total_trades_oos"] >= 50,
            "worst_fold_controlled": metrics["worst_fold_pnl"] > worst_fold_threshold,
        }

        consistency = {
            "folds_positive_gte_50pct": metrics["folds_positive_pct"] >= 0.50,
            "wr_gt_35pct": metrics["wr_oos"] > 0.35,
            "no_quarter_pf_lt_0.5": (
                all(pf >= 0.5 for pf in metrics["quarterly_pf"].values())
                if metrics["quarterly_pf"] else True
            ),
        }

        all_obligatory = all(obligatory.values())
        consistency_count = sum(consistency.values())

        if all_obligatory and consistency_count >= 2:
            verdict = "PRODUCCION_ESTANDAR"
            risk = 1.0
        elif all_obligatory and consistency_count == 1:
            verdict = "PRODUCCION_CONTROLADA"
            risk = 0.5
        elif all_obligatory:
            verdict = "NO_LISTA"
            risk = 0.0
        else:
            verdict = "BLOQUEADA"
            risk = 0.0

        return {
            "obligatory": obligatory,
            "consistency": consistency,
            "consistency_passed": consistency_count,
            "verdict": verdict,
            "risk_pct": risk,
            "metrics": metrics,
        }
