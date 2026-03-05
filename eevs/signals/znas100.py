"""EEVS signal — ZNas100 v1.7.0.

Pure signal function: reads pre-computed columns, never computes.
State machine (WAIT/READY_LONG/READY_SHORT) in state dict.

Required pre-computed columns:
  zscore, ema_bias, pivot_bias, atr_d, next_open

Required config.params:
  z_door, z_fire_delta, start_hour, end_hour,
  use_strict_ema, use_pivot_filter, min_atr_d
"""

from typing import Any

import numpy as np
import pandas as pd

from ..config import StrategyConfig


# ── Default production parameters ───────────────────────────────────────────

ZNAS100_PARAMS = {
    "z_period":         20,
    "z_door":           2.5,
    "z_fire_delta":     0.7,
    "ema_fast":         34,
    "ema_mid":          89,
    "ema_slow":         144,
    "tp_points":        350.0,
    "sl_points":        250.0,
    "start_hour":       1,
    "end_hour":         21,
    "use_pivot_filter": True,
    "use_strict_ema":   True,
    "atr_d_period":     14,
    "min_atr_d":        250.0,
}


def make_config(data_path: str = "data/oanda_NAS100_USD_1m_800000.csv") -> StrategyConfig:
    """Create StrategyConfig for ZNas100 v1.7.0 with production params."""
    return StrategyConfig(
        name="ZNas100",
        version="1.7.0",
        asset="NAS100",
        timeframe="15min",
        data_path=data_path,
        params=ZNAS100_PARAMS,
        warmup_bars=2000,
        tp_points=ZNAS100_PARAMS["tp_points"],
        sl_points=ZNAS100_PARAMS["sl_points"],
    )


def signal_fn(row: pd.Series, state: dict[str, Any]) -> str | None:
    """ZNas100 v1.7.0 signal function.

    State machine:
      WAIT -> READY_LONG  (when z <= -z_door)
      WAIT -> READY_SHORT (when z >= +z_door)
      READY_LONG  -> fires LONG  (when delta_z >= z_fire_delta + filters)
      READY_SHORT -> fires SHORT (when delta_z >= z_fire_delta + filters)

    Returns "LONG", "SHORT", or None.
    """
    p = ZNAS100_PARAMS

    ts = row.name
    day = ts.weekday()
    hour = ts.hour

    # Weekend filter
    if day >= 5:
        state["machine"] = "WAIT"
        state["prev_z"] = np.nan
        return None

    # Hour filter
    if not (p["start_hour"] <= hour < p["end_hour"]):
        z_val = row.get("zscore")
        if z_val is not None and not pd.isna(z_val):
            state["prev_z"] = float(z_val)
        return None

    z = row.get("zscore")
    ema_b = row.get("ema_bias")
    pivot_b = row.get("pivot_bias")
    atr_d = row.get("atr_d")

    if z is None or pd.isna(z) or pivot_b is None or pd.isna(pivot_b):
        state["prev_z"] = np.nan
        return None

    z = float(z)
    ema_b = int(ema_b) if ema_b is not None and not pd.isna(ema_b) else 0
    atr_ok = atr_d is not None and not pd.isna(atr_d) and float(atr_d) >= p["min_atr_d"]

    machine = state.get("machine", "WAIT")
    prev_z = state.get("prev_z", np.nan)

    # Arming (always active, independent of ATR)
    if machine == "WAIT":
        if z <= -p["z_door"]:
            machine = "READY_LONG"
        elif z >= p["z_door"]:
            machine = "READY_SHORT"

    # Firing (ATR + EMA + Pivot filters)
    signal = None
    if not np.isnan(prev_z):
        if machine == "READY_LONG" and (z - prev_z) >= p["z_fire_delta"]:
            if (atr_ok
                    and (not p["use_strict_ema"] or ema_b == 1)
                    and (not p["use_pivot_filter"] or pivot_b == 1)):
                signal = "LONG"
            machine = "WAIT"

        elif machine == "READY_SHORT" and (prev_z - z) >= p["z_fire_delta"]:
            if (atr_ok
                    and (not p["use_strict_ema"] or ema_b == -1)
                    and (not p["use_pivot_filter"] or pivot_b == -1)):
                signal = "SHORT"
            machine = "WAIT"

    state["machine"] = machine
    state["prev_z"] = z
    return signal
