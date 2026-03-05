"""EEVS simulator — Deterministic bar-by-bar simulation engine.

Rules (hardcoded, non-negotiable):
  - Entry: next_open (bar after signal)
  - Fill price: next_open ± cost_per_side
  - TP/SL collision: worst_case (SL wins)
  - Costs: applied per-side at entry and exit
"""

from dataclasses import dataclass, field
from typing import Callable, Any

import numpy as np
import pandas as pd

from .config import StrategyConfig, CostModel


@dataclass
class TradeRecord:
    """Complete record of a single trade for audit purposes."""
    signal_time: pd.Timestamp
    entry_time: pd.Timestamp
    direction: str              # "LONG" | "SHORT"
    signal_price: float         # close of signal bar
    fill_price: float           # next_open ± cost
    tp_level: float
    sl_level: float
    exit_time: pd.Timestamp | None = None
    exit_price: float | None = None
    exit_reason: str | None = None   # "TP" | "SL"
    pnl_gross: float = 0.0
    pnl_net: float = 0.0
    cost_applied: float = 0.0
    month: str = ""
    pnl_eq: float = 0.0


def simulate(
    df: pd.DataFrame,
    config: StrategyConfig,
    signal_fn: Callable[[pd.Series, dict[str, Any]], str | None],
) -> list[TradeRecord]:
    """Run deterministic bar-by-bar simulation.

    Parameters
    ----------
    df : DataFrame with pre-computed indicators + next_open column.
    config : Strategy configuration (immutable).
    signal_fn : Pure function (row, state_dict) -> "LONG" | "SHORT" | None.

    Returns
    -------
    List of TradeRecord with complete audit trail.
    """
    cost = config.cost_model
    trades: list[TradeRecord] = []
    state: dict[str, Any] = {}
    active: TradeRecord | None = None
    pending_signal: dict | None = None

    sim_df = df.iloc[config.warmup_bars:]

    for ts, row in sim_df.iterrows():
        hi = float(row["high"])
        lo = float(row["low"])
        op = float(row["open"])

        # ── Step 1: Activate pending signal (entry on this bar's open) ──
        if pending_signal is not None:
            direction = pending_signal["direction"]
            signal_price = pending_signal["signal_price"]
            signal_time = pending_signal["signal_time"]

            if cost.cost_type == "points":
                cost_side = cost.cost_per_side
            else:
                cost_side = op * cost.cost_per_side / 100.0

            if direction == "LONG":
                fill = op + cost_side
                tp_level = fill + config.tp_points
                sl_level = fill - config.sl_points
            else:
                fill = op - cost_side
                tp_level = fill - config.tp_points
                sl_level = fill + config.sl_points

            active = TradeRecord(
                signal_time=signal_time,
                entry_time=ts,
                direction=direction,
                signal_price=signal_price,
                fill_price=fill,
                tp_level=tp_level,
                sl_level=sl_level,
                cost_applied=cost.round_trip,
                month=ts.strftime("%Y-%m"),
            )
            pending_signal = None

        # ── Step 2: Resolve active trade ────────────────────────────────
        if active is not None:
            if active.direction == "LONG":
                hit_tp = hi >= active.tp_level
                hit_sl = lo <= active.sl_level
            else:
                hit_tp = lo <= active.tp_level
                hit_sl = hi >= active.sl_level

            # Worst-case collision: SL wins
            if hit_tp and hit_sl:
                hit_tp = False

            if hit_tp:
                active.exit_time = ts
                active.exit_price = active.tp_level
                active.exit_reason = "TP"
                active.pnl_gross = config.tp_points
                active.pnl_net = config.tp_points - cost.round_trip
                active.pnl_eq = config.risk_pct * (config.tp_points - cost.round_trip) / config.sl_points
                trades.append(active)
                active = None
            elif hit_sl:
                active.exit_time = ts
                active.exit_price = active.sl_level
                active.exit_reason = "SL"
                active.pnl_gross = -config.sl_points
                active.pnl_net = -(config.sl_points + cost.round_trip)
                active.pnl_eq = -config.risk_pct * (config.sl_points + cost.round_trip) / config.sl_points
                trades.append(active)
                active = None

        # ── Step 3: Check for new signal (only if no active trade) ──────
        if active is None and pending_signal is None:
            next_open = row.get("next_open")
            if pd.notna(next_open):
                signal = signal_fn(row, state)
                if signal in ("LONG", "SHORT"):
                    pending_signal = {
                        "direction": signal,
                        "signal_price": float(row["close"]),
                        "signal_time": ts,
                    }

    return trades
