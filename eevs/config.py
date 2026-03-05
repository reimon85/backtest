"""EEVS config — Immutable configuration objects.

CostModel, COST_TABLE (PRODUCTION_STANDARD §4), StrategyConfig.
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class CostModel:
    """Per-asset cost model (PRODUCTION_STANDARD §4)."""
    spread: float
    slippage: float
    commission: float = 0.0
    cost_type: str = "points"  # "points" | "percentage"

    @property
    def cost_per_side(self) -> float:
        return self.spread + self.slippage

    @property
    def round_trip(self) -> float:
        return 2 * self.cost_per_side + self.commission


COST_TABLE: dict[str, CostModel] = {
    "NAS100":   CostModel(spread=2.5,   slippage=0.7,  commission=0.0, cost_type="points"),
    "US30":     CostModel(spread=2.5,   slippage=0.7,  commission=0.0, cost_type="points"),
    "XAUUSD":   CostModel(spread=0.3,   slippage=0.1,  commission=0.0, cost_type="points"),
    "GBPJPY":   CostModel(spread=0.02,  slippage=0.01, commission=0.0, cost_type="points"),
    "BTC_SPOT": CostModel(spread=0.05,  slippage=0.05, commission=0.10, cost_type="percentage"),
    "BTC_FUT":  CostModel(spread=0.02,  slippage=0.02, commission=0.05, cost_type="percentage"),
}


@dataclass(frozen=True)
class StrategyConfig:
    """Immutable strategy configuration for EEVS runs."""
    name: str
    version: str
    asset: str
    timeframe: str
    data_path: str
    params: dict[str, Any] = field(default_factory=dict)
    warmup_bars: int = 2000
    tp_points: float = 350.0
    sl_points: float = 250.0
    max_concurrent: int = 1
    worst_case_intrabar: bool = True
    entry_mode: str = "next_open"
    risk_pct: float = 0.01

    @property
    def cost_model(self) -> CostModel:
        return COST_TABLE[self.asset]
