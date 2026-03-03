import logging
import pandas as pd
import numpy as np
from decimal import Decimal
from datetime import timedelta, timezone

from eagleseye.core.enums import Direction
from eagleseye.core.models import OHLCVBar, Signal
from eagleseye.strategies.base import BaseStrategy

logger = logging.getLogger(__name__)

class BTCZGapStrategy(BaseStrategy):
    """
    BTCZGap Strategy (v5.8.0) - GOLDEN STANDARD REVERSION
    - Daily Trend Filter: EMA 89 > 144
    - Daily Safety Zone: Z-Score (34p) between -0.9 and 0.9
    - 1H Z-Door Precision: 20 periods with 1.5 StdDev threshold
    - 2H RSI(7) Velocity Filter: Arming < 40 / Firing > 45
    - Sniper Management: TP 3.0x / SL 2.0x ATR (Wilder)
    - Protection: 24h Cooldown & Min Daily ATR 1.2%
    """

    name = "btczgap"
    version = "5.8.0"
    
    default_params = {
        "ema_m": 89,
        "ema_s": 144,
        "z_period_daily": 34,
        "z_daily_limit": 0.9,
        "z_period_1h": 20,
        "z_1h_arm_lvl": 1.5,
        "rsi_period": 7,
        "rsi_arm_l": 40,
        "rsi_fire_l": 45,
        "rsi_arm_s": 60,
        "rsi_fire_s": 55,
        "atr_period": 14,
        "tp_mult": 3.0,
        "sl_mult": 2.0,
        "min_atr_daily_pct": 0.012,
        "cooldown_h": 24,
        "max_concurrent": 5
    }

    def __init__(self, params: dict | None = None):
        super().__init__(params)
        self._armed_l = False
        self._armed_s = False
        self._last_trigger_ts = None

    def required_candles(self) -> int:
        return 1500 

    def reset_backtest_state(self) -> None:
        self._armed_l = False
        self._armed_s = False
        self._last_trigger_ts = None

    def _zscore(self, series: pd.Series, period: int) -> pd.Series:
        mu = series.rolling(window=period).mean()
        std = series.rolling(window=period).std().replace(0, np.nan)
        return (series - mu) / std

    def _rsi_wilder(self, series: pd.Series, period: int) -> pd.Series:
        delta = series.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        return 100 - (100 / (1 + rs))

    async def analyze(self, symbol: str, timeframe: str, candles: list[OHLCVBar], active_trades: list = None, **kwargs) -> list[Signal] | None:
        if len(candles) < 500: return None
        
        # Convertimos a DataFrame
        df = pd.DataFrame([{"ts": b.timestamp, "c": float(b.close), "h": float(b.high), "l": float(b.low)} for b in candles])
        df.set_index(pd.to_datetime(df["ts"], utc=True), inplace=True)
        current_ts = df.index[-1]

        # 1. HORARIO (Dom 12:00 a Vie 21:00 UTC)
        day, hour = current_ts.weekday(), current_ts.hour
        is_th = not (day == 4 and hour >= 21 or day == 5 or day == 6 and hour < 12)
        if not is_th:
            self._armed_l = self._armed_s = False
            return None

        # 2. COOLDOWN
        if self._last_trigger_ts and (current_ts - self._last_trigger_ts) < timedelta(hours=self.params["cooldown_h"]):
            return None

        # 3. FILTROS DIARIOS (Resample 21:00 UTC)
        df_d = df[df.index.hour == 21].copy()
        if len(df_d) < 40: return None
        
        # ATR Diario Pct
        df_d["atr_d"] = (df["h"].resample("D").max() - df["l"].resample("D").min()).rolling(14).mean().reindex(df_d.index, method="ffill")
        atr_d_pct = df_d["atr_d"].iloc[-1] / df_d["c"].iloc[-1]
        if atr_d_pct < self.params["min_atr_daily_pct"]:
            return None

        # Tendencia y Z-Score Diario
        df_d["ema89"] = df_d["c"].ewm(span=self.params["ema_m"]).mean()
        df_d["ema144"] = df_d["c"].ewm(span=self.params["ema_s"]).mean()
        df_d["z_d"] = self._zscore(df_d["c"], self.params["z_period_daily"])
        
        last_ema89 = df_d["ema89"].iloc[-1]
        last_ema144 = df_d["ema144"].iloc[-1]
        last_z_d = df_d["z_d"].iloc[-1]
        
        can_l = (last_ema89 > last_ema144) and (last_z_d >= -self.params["z_daily_limit"])
        can_s = (last_ema89 < last_ema144) and (last_z_d <= self.params["z_daily_limit"])

        # 4. INDICADORES 1H Y 2H
        # Z-Score 1H (20p)
        z_1h = self._zscore(df["c"], self.params["z_period_1h"]).iloc[-1]
        
        # RSI 2H (7p)
        df_2h = df["c"].resample("2h").last().to_frame()
        rsi_2h_series = self._rsi_wilder(df_2h["c"], self.params["rsi_period"])
        rsi_2h = rsi_2h_series.iloc[-1]
        
        # ATR 1H Wilder para salidas
        h_l = df['h'] - df['l']
        h_pc = np.abs(df['h'] - df['c'].shift(1))
        l_pc = np.abs(df['l'] - df['c'].shift(1))
        atr_1h = np.max(pd.concat([h_l, h_pc, l_pc], axis=1), axis=1).rolling(self.params["atr_period"]).mean().iloc[-1]

        # 5. MÁQUINA DE ESTADOS (ARMADO)
        if can_l and z_1h <= -self.params["z_1h_arm_lvl"] and rsi_2h <= self.params["rsi_arm_l"]: self._armed_l = True
        if can_s and z_1h >= self.params["z_1h_arm_lvl"] and rsi_2h >= self.params["rsi_arm_s"]: self._armed_s = True
        
        # Reset de seguridad
        if z_1h > 0: self._armed_l = False
        if z_1h < 0: self._armed_s = False

        if active_trades and len(active_trades) >= self.params["max_concurrent"]: return None

        # 6. DISPARO (FIRING)
        last_close = Decimal(str(df["c"].iloc[-1]))
        
        if self._armed_l and rsi_2h >= self.params["rsi_fire_l"]:
            dist_tp = Decimal(str(atr_1h * self.params["tp_mult"]))
            dist_sl = Decimal(str(atr_1h * self.params["sl_mult"]))
            self._armed_l, self._last_trigger_ts = False, current_ts
            return [Signal(Direction.LONG, entry_price=last_close, take_profit=last_close + dist_tp, stop_loss=last_close - dist_sl)]
            
        if self._armed_s and rsi_2h <= self.params["rsi_fire_s"]:
            dist_tp = Decimal(str(atr_1h * self.params["tp_mult"]))
            dist_sl = Decimal(str(atr_1h * self.params["sl_mult"]))
            self._armed_s, self._last_trigger_ts = False, current_ts
            return [Signal(Direction.SHORT, entry_price=last_close, take_profit=last_close - dist_tp, stop_loss=last_close + dist_sl)]

        return None
