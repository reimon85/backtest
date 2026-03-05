"""EEVS data_factory — Pre-computation of indicators (single source of truth).

All indicators are computed with explicit parameters and anti-look-ahead
guarantees. The strategy only READS pre-computed columns, never computes.
"""

import pandas as pd
import numpy as np
from .config import StrategyConfig


# ── Indicator functions (explicit params, no ambiguity) ─────────────────────


def zscore(series: pd.Series, period: int, ddof: int = 1) -> pd.Series:
    """Causal rolling Z-Score with explicit ddof."""
    roll = series.rolling(window=period)
    return (series - roll.mean()) / roll.std(ddof=ddof)


def ema(series: pd.Series, span: int) -> pd.Series:
    """Causal EWM (adjust=False — no look-ahead)."""
    return series.ewm(span=span, adjust=False).mean()


def rsi_wilder(series: pd.Series, period: int) -> pd.Series:
    """RSI with Wilder smoothing (EWM alpha=1/period)."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def atr_daily(df: pd.DataFrame, period: int, shift: int = 1) -> pd.Series:
    """Daily ATR with shift to prevent look-ahead.

    Returns a daily series shifted by `shift` days (default=1 = previous day).
    """
    daily = df.resample("1D").agg({"high": "max", "low": "min", "close": "last"}).dropna()
    atr = (daily["high"] - daily["low"]).rolling(period).mean()
    return atr.shift(shift)


def pivot_bias(df: pd.DataFrame, shift: int = 1) -> pd.Series:
    """Daily pivot bias with anti-look-ahead shift.

    Compares pivot(shift) vs pivot(shift+1) to determine trend direction.
    Returns +1 (bullish), -1 (bearish), 0 (neutral).
    """
    daily = df.resample("1D").agg({"high": "max", "low": "min", "close": "last"}).dropna()
    daily["pivot"] = (daily["high"] + daily["low"] + daily["close"]) / 3.0
    bias = pd.Series(0, index=daily.index, dtype=int)
    bias.loc[daily["pivot"].shift(shift) > daily["pivot"].shift(shift + 1)] = 1
    bias.loc[daily["pivot"].shift(shift) < daily["pivot"].shift(shift + 1)] = -1
    return bias


# ── Main builder ────────────────────────────────────────────────────────────


def build_dataframe(config: StrategyConfig) -> pd.DataFrame:
    """Load CSV, resample to timeframe, compute all indicators as fixed columns.

    Returns a DataFrame with OHLCV + indicator columns + next_open.
    The strategy only reads from this DataFrame — never computes.
    """
    # Load raw data
    df_raw = pd.read_csv(config.data_path, parse_dates=["timestamp"])
    df_raw["timestamp"] = pd.to_datetime(df_raw["timestamp"], utc=True)
    df_raw.set_index("timestamp", inplace=True)
    df_raw.sort_index(inplace=True)

    # Resample to strategy timeframe
    df = df_raw.resample(config.timeframe).agg({
        "open":   "first",
        "high":   "max",
        "low":    "min",
        "close":  "last",
        "volume": "sum",
    }).dropna()

    # Entry mode: next_open
    df["next_open"] = df["open"].shift(-1)

    # Compute indicators based on strategy params
    p = config.params

    # Z-Score (causal rolling, ddof=1)
    if "z_period" in p:
        df["zscore"] = zscore(df["close"], period=p["z_period"], ddof=1)

    # EMAs (causal)
    if "ema_fast" in p:
        df["ema_fast"] = ema(df["close"], span=p["ema_fast"])
    if "ema_mid" in p:
        df["ema_mid"] = ema(df["close"], span=p["ema_mid"])
    if "ema_slow" in p:
        df["ema_slow"] = ema(df["close"], span=p["ema_slow"])

    # EMA bias (requires all three EMAs)
    if all(k in p for k in ("ema_fast", "ema_mid", "ema_slow")):
        df["ema_bias"] = 0
        df.loc[(df["ema_fast"] > df["ema_mid"]) & (df["ema_mid"] > df["ema_slow"]), "ema_bias"] = 1
        df.loc[(df["ema_fast"] < df["ema_mid"]) & (df["ema_mid"] < df["ema_slow"]), "ema_bias"] = -1

    # Pivot bias (daily, anti-look-ahead)
    if p.get("use_pivot_filter"):
        df["pivot_bias"] = pivot_bias(df, shift=1).reindex(df.index, method="ffill")

    # ATR daily (anti-look-ahead)
    if "atr_d_period" in p:
        df["atr_d"] = atr_daily(df, period=p["atr_d_period"], shift=1).reindex(df.index, method="ffill")

    # RSI (if needed by strategy)
    if "rsi_period" in p:
        df["rsi"] = rsi_wilder(df["close"], period=p["rsi_period"])

    return df
