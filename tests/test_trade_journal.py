import pandas as pd
import pytest
from src.trade_journal import TradeJournal, _fmt_duration


class TestTradeJournal:
    def test_open_and_close_trade(self):
        j = TradeJournal()
        tid = j.open_trade(
            pd.Timestamp("2025-02-02 10:00", tz="UTC"),
            "LONG", 6840.0, tp=6936.0, sl=6743.6, multiplier=10.0,
            trade_type="WEEKLY", metadata={"atr": 96.4},
        )
        assert len(j.open_trades) == 1
        assert len(j.closed_trades) == 0

        trade = j.close_trade(
            tid, pd.Timestamp("2025-02-06 14:00", tz="UTC"), 6743.6, result="SL"
        )
        assert trade.is_closed
        assert trade.result == "SL"
        assert trade.pnl == round((6743.6 - 6840.0) * 10.0, 2)
        assert len(j.open_trades) == 0
        assert len(j.closed_trades) == 1

    def test_short_trade_pnl(self):
        j = TradeJournal()
        tid = j.open_trade(
            pd.Timestamp("2025-03-01"), "SHORT", 5000.0, multiplier=5.0
        )
        j.close_trade(tid, pd.Timestamp("2025-03-02"), 4900.0, result="TP")
        assert j.closed_trades[0].pnl == round(100.0 * 5.0, 2)

    def test_duration(self):
        j = TradeJournal()
        tid = j.open_trade(pd.Timestamp("2025-01-01 00:00"), "LONG", 100.0)
        j.close_trade(tid, pd.Timestamp("2025-01-05 02:30"), 110.0, result="TP")
        assert j.closed_trades[0].duration == pd.Timedelta(days=4, hours=2, minutes=30)

    def test_to_dataframe_has_required_columns(self):
        j = TradeJournal()
        tid = j.open_trade(
            pd.Timestamp("2025-01-01"), "LONG", 100.0,
            tp=110.0, sl=95.0, trade_type="WEEKLY", metadata={"atr": 10.0},
        )
        j.close_trade(tid, pd.Timestamp("2025-01-03"), 110.0, result="TP")
        df = j.to_dataframe()
        required = {"Entry_TS", "Exit_TS", "Duration", "Type", "Dir",
                     "Entry", "Exit", "TP", "SL", "Result", "PnL", "meta_atr"}
        assert required.issubset(set(df.columns))

    def test_pnl_list_compatible_with_wfa(self):
        j = TradeJournal()
        for i in range(5):
            tid = j.open_trade(pd.Timestamp(f"2025-01-{i+1:02d}"), "LONG", 100.0, multiplier=1.0)
            j.close_trade(tid, pd.Timestamp(f"2025-01-{i+2:02d}"), 100.0 + (i - 2) * 10, result="TP")
        pnls = j.pnl_list()
        assert len(pnls) == 5
        assert all(isinstance(p, float) for p in pnls)

    def test_explicit_pnl_overrides_auto(self):
        j = TradeJournal()
        tid = j.open_trade(pd.Timestamp("2025-01-01"), "LONG", 100.0, multiplier=1.0)
        j.close_trade(tid, pd.Timestamp("2025-01-02"), 110.0, result="TP", pnl=42.0)
        assert j.closed_trades[0].pnl == 42.0

    def test_commission_subtracted(self):
        j = TradeJournal()
        tid = j.open_trade(pd.Timestamp("2025-01-01"), "LONG", 100.0, multiplier=1.0)
        j.close_trade(tid, pd.Timestamp("2025-01-02"), 110.0, result="TP", commission=2.5)
        assert j.closed_trades[0].pnl == round(10.0 * 1.0 - 2.5, 2)


class TestFmtDuration:
    def test_days_hours(self):
        assert _fmt_duration(pd.Timedelta(days=4, hours=2)) == "4d 2h"

    def test_hours_minutes(self):
        assert _fmt_duration(pd.Timedelta(hours=3, minutes=15)) == "3h 15m"

    def test_minutes_only(self):
        assert _fmt_duration(pd.Timedelta(minutes=45)) == "45m"

    def test_none(self):
        assert _fmt_duration(None) == "OPEN"
