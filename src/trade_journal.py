"""
TradeJournal — Registro estándar de operaciones para backtesting.

Uso:
    journal = TradeJournal()
    tid = journal.open_trade(entry_ts, "LONG", 6840.0, tp=6936.0, sl=6743.6,
                             metadata={"atr": 96.4, "ref_level": 6850.0})
    journal.close_trade(tid, exit_ts, 6743.6, result="SL")
    journal.display()
"""

from __future__ import annotations

import pandas as pd
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Trade:
    """Registro completo del ciclo de vida de un trade."""
    trade_id: int
    entry_ts: pd.Timestamp
    direction: str          # "LONG" | "SHORT"
    entry_price: float
    tp: float
    sl: float
    multiplier: float = 1.0
    trade_type: str = ""    # "WEEKLY", "DAILY", etc.
    metadata: dict = field(default_factory=dict)
    # Campos de cierre (se rellenan al cerrar)
    exit_ts: pd.Timestamp | None = None
    exit_price: float | None = None
    result: str | None = None   # "TP" | "SL" | "TIMEOUT" | ...
    pnl: float | None = None
    commission: float = 0.0

    @property
    def duration(self) -> pd.Timedelta | None:
        if self.exit_ts is not None and self.entry_ts is not None:
            return self.exit_ts - self.entry_ts
        return None

    @property
    def is_closed(self) -> bool:
        return self.exit_ts is not None


def _fmt_duration(td: pd.Timedelta | None) -> str:
    """Formatea un Timedelta a algo legible: '4d 2h', '0d 3h', etc."""
    if td is None:
        return "OPEN"
    total_seconds = int(td.total_seconds())
    days = total_seconds // 86400
    hours = (total_seconds % 86400) // 3600
    minutes = (total_seconds % 3600) // 60
    if days > 0:
        return f"{days}d {hours}h"
    if hours > 0:
        return f"{hours}h {minutes}m"
    return f"{minutes}m"


class TradeJournal:
    """
    Componente estándar para registrar trades con ciclo de vida completo.

    Campos obligatorios por trade:
      - entry_ts, exit_ts, duration (auto)
      - direction, entry_price, exit_price
      - result, pnl
      - metadata (ATR, ref_level, etc.)
    """

    def __init__(self):
        self._trades: list[Trade] = []
        self._next_id: int = 0

    # ------------------------------------------------------------------
    # API pública
    # ------------------------------------------------------------------

    def open_trade(
        self,
        entry_ts: pd.Timestamp,
        direction: str,
        entry_price: float,
        *,
        tp: float = 0.0,
        sl: float = 0.0,
        multiplier: float = 1.0,
        trade_type: str = "",
        metadata: dict | None = None,
    ) -> int:
        """Registra la apertura de un trade. Devuelve el trade_id."""
        tid = self._next_id
        self._next_id += 1
        self._trades.append(
            Trade(
                trade_id=tid,
                entry_ts=entry_ts,
                direction=direction.upper(),
                entry_price=entry_price,
                tp=tp,
                sl=sl,
                multiplier=multiplier,
                trade_type=trade_type,
                metadata=metadata or {},
            )
        )
        return tid

    def close_trade(
        self,
        trade_id: int,
        exit_ts: pd.Timestamp,
        exit_price: float,
        *,
        result: str = "",
        pnl: float | None = None,
        commission: float = 0.0,
    ) -> Trade:
        """Cierra un trade abierto. Calcula PnL si no se proporciona."""
        trade = self._get(trade_id)
        trade.exit_ts = exit_ts
        trade.exit_price = exit_price
        trade.result = result.upper()
        trade.commission = commission

        if pnl is not None:
            trade.pnl = pnl
        else:
            # Auto-calcular PnL
            diff = exit_price - trade.entry_price
            if trade.direction == "SHORT":
                diff = -diff
            trade.pnl = round(diff * trade.multiplier - commission, 2)

        return trade

    @property
    def closed_trades(self) -> list[Trade]:
        return [t for t in self._trades if t.is_closed]

    @property
    def open_trades(self) -> list[Trade]:
        return [t for t in self._trades if not t.is_closed]

    def pnl_list(self) -> list[float]:
        """Lista de PnL por trade cerrado (compatible con WFA/MonteCarlo)."""
        return [t.pnl for t in self.closed_trades]

    def to_dataframe(self) -> pd.DataFrame:
        """DataFrame con el ciclo de vida completo de cada trade cerrado."""
        rows = []
        for t in self.closed_trades:
            rows.append({
                "Entry_TS": t.entry_ts,
                "Exit_TS": t.exit_ts,
                "Duration": _fmt_duration(t.duration),
                "Type": t.trade_type,
                "Dir": t.direction,
                "Entry": round(t.entry_price, 2),
                "Exit": round(t.exit_price, 2),
                "TP": round(t.tp, 2),
                "SL": round(t.sl, 2),
                "Result": t.result,
                "PnL": t.pnl,
                **{f"meta_{k}": v for k, v in t.metadata.items()},
            })
        return pd.DataFrame(rows)

    def display(self) -> None:
        """Imprime la tabla completa + métricas resumen."""
        df = self.to_dataframe()
        if df.empty:
            print("No hay trades cerrados.")
            return

        print(df.to_string(index=False))
        print()

        total_pnl = df["PnL"].sum()
        wins = (df["Result"] == "TP").sum()
        total = len(df)
        wr = wins / total if total else 0

        gross_profit = df.loc[df["PnL"] > 0, "PnL"].sum()
        gross_loss = abs(df.loc[df["PnL"] < 0, "PnL"].sum())
        pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        avg_win = df.loc[df["PnL"] > 0, "PnL"].mean() if wins else 0
        avg_loss = df.loc[df["PnL"] < 0, "PnL"].mean() if (total - wins) else 0

        durations = [t.duration for t in self.closed_trades if t.duration is not None]
        avg_dur = _fmt_duration(sum(durations, pd.Timedelta(0)) / len(durations)) if durations else "N/A"

        print(f"Total PnL:      {total_pnl:>12,.2f}")
        print(f"Trades:         {total:>12d}")
        print(f"Win Rate:       {wr:>12.2%}")
        print(f"Profit Factor:  {pf:>12.2f}")
        print(f"Avg Win:        {avg_win:>12,.2f}")
        print(f"Avg Loss:       {avg_loss:>12,.2f}")
        print(f"Avg Duration:   {avg_dur:>12s}")

    # ------------------------------------------------------------------
    # Internos
    # ------------------------------------------------------------------

    def _get(self, trade_id: int) -> Trade:
        for t in self._trades:
            if t.trade_id == trade_id:
                return t
        raise KeyError(f"Trade {trade_id} no encontrado")
