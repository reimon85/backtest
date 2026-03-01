import pandas as pd
import numpy as np
import asyncio
from datetime import datetime, timedelta
from decimal import Decimal
from strategies.spxunhitW import SPXUnhitStrategy
from src.engine_event import ExecutionModel
from src.trade_journal import TradeJournal

class MockOHLCVBar:
    def __init__(self, timestamp, open, high, low, close, volume):
        self.timestamp = timestamp
        self.open = Decimal(str(open))
        self.high = Decimal(str(high))
        self.low = Decimal(str(low))
        self.close = Decimal(str(close))
        self.volume = Decimal(str(volume))

async def run_spx_backtest(file_path):
    df = pd.read_csv(file_path, parse_dates=['timestamp'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    df.set_index('timestamp', inplace=True)

    end_date = df.index.max()
    start_date = end_date - timedelta(days=365)
    df_test = df[df.index >= start_date].copy()
    df_hist = df[df.index < start_date].tail(6000)

    strategy = SPXUnhitStrategy()
    execution = ExecutionModel(slippage_pct=0.00005, commission_pct=0.0001)
    journal = TradeJournal()

    # Map trade_id -> active trade info (tp, sl, multiplier, direction, metadata)
    active_trades = {}

    current_candles = [MockOHLCVBar(ts, row['open'], row['high'], row['low'], row['close'], row['volume'])
                       for ts, row in df_hist.iterrows()]

    print(f"Iniciando simulación de 365 días (V4.0.0)...")
    for ts, row in df_test.iterrows():
        new_bar = MockOHLCVBar(ts, row['open'], row['high'], row['low'], row['close'], row['volume'])
        current_candles.append(new_bar)
        if len(current_candles) > 7000: current_candles.pop(0)

        for tid in list(active_trades.keys()):
            info = active_trades[tid]
            curr_high, curr_low = float(row['high']), float(row['low'])
            hit_tp, hit_sl = False, False
            if info['direction'] == "LONG":
                if curr_high >= info['tp']: hit_tp = True
                elif curr_low <= info['sl']: hit_sl = True
            else:
                if curr_low <= info['tp']: hit_tp = True
                elif curr_high >= info['sl']: hit_sl = True

            if hit_tp or hit_sl:
                res = "TP" if hit_tp else "SL"
                exit_price = info['tp'] if hit_tp else info['sl']
                side = "SELL" if info['direction'] == "LONG" else "BUY"
                exec_price, comm = execution.execute(side, "SPX", 1.0, exit_price)
                journal.close_trade(tid, ts, exec_price, result=res, commission=comm)
                strategy.register_trade_result(0.0, timestamp=ts, result=res, metadata=info['metadata'])
                del active_trades[tid]

        signals = await strategy.analyze("SPX", "1h", current_candles)
        if signals:
            for sig in signals:
                if not active_trades:
                    meta = sig.metadata if hasattr(sig, 'metadata') else {}
                    tid = journal.open_trade(
                        entry_ts=ts,
                        direction=sig.direction.name,
                        entry_price=float(sig.entry_price),
                        tp=float(sig.take_profit),
                        sl=float(sig.stop_loss),
                        multiplier=sig.quantity_multiplier * 10,
                        trade_type="WEEKLY",
                        metadata=meta,
                    )
                    active_trades[tid] = {
                        "direction": sig.direction.name,
                        "tp": float(sig.take_profit),
                        "sl": float(sig.stop_loss),
                        "metadata": meta,
                    }

    return journal

if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    journal = loop.run_until_complete(run_spx_backtest("data/spx_data_1h_10k.csv"))

    print("\n--- DIARIO DE OPERACIONES SPX (Últimos 365 días) ---")
    journal.display()
