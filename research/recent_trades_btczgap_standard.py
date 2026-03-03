import pandas as pd
import numpy as np
from datetime import timedelta, timezone
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.trade_journal import TradeJournal

P = {
    "ema_m": 89, "ema_s": 144, "z_per_d": 34, "z_d_lim": 0.9,
    "z_per_1h": 20, "z_arm_1h": 1.5,
    "rsi_per": 7, "rsi_arm_l": 40, "rsi_fire_l": 45,
    "rsi_arm_s": 60, "rsi_fire_s": 55,
    "tp_m": 3.0, "sl_m": 2.0,
    "atr_per": 14, "min_atr_d": 0.012, "cooldown_h": 24, "comm": 0.0005
}

DATA_PATH = "data/binance_BTCUSDT_1h_25000.csv"

def _rsi_wilder(series, period):
    delta = series.diff()
    gain, loss = delta.where(delta > 0, 0.0), -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def precompute(df):
    df = df.copy()
    h_l, h_pc, l_pc = df['high']-df['low'], np.abs(df['high']-df['close'].shift(1)), np.abs(df['low']-df['close'].shift(1))
    df['tr'] = np.max(pd.concat([h_l, h_pc, l_pc], axis=1), axis=1)
    df['atr_1h'] = df['tr'].rolling(P["atr_per"]).mean()
    df_d = df.resample('D').agg({'high':'max', 'low':'min', 'close':'last'}).dropna()
    df_d['ema89'], df_d['ema144'] = df_d['close'].ewm(span=P["ema_m"]).mean(), df_d['close'].ewm(span=P["ema_s"]).mean()
    mu_d, std_d = df_d['close'].rolling(P["z_per_d"]).mean(), df_d['close'].rolling(P["z_per_d"]).std().replace(0, np.nan)
    df_d['z_d'] = (df_d['close'] - mu_d) / std_d
    df_d['atr_d_pct'] = (df_d['high'] - df_d['low']).rolling(14).mean() / df_d['close']
    daily = df_d[['ema89', 'ema144', 'z_d', 'atr_d_pct']].shift(1).reindex(df.index, method='ffill')
    df['ema89_d'], df['ema144_d'], df['z_d'], df['atr_d_pct'] = daily['ema89'], daily['ema144'], daily['z_d'], daily['atr_d_pct']
    df_2h = df['close'].resample('2H').last().to_frame()
    df_2h['rsi'] = _rsi_wilder(df_2h['close'], P["rsi_per"])
    df['rsi_2h'] = df_2h['rsi'].reindex(df.index, method='ffill')
    df['z_1h'] = (df['close'] - df['close'].rolling(20).mean()) / df['close'].rolling(20).std().replace(0, np.nan)
    return df

def simulate(df):
    journal = TradeJournal()
    active_tid = None
    armed_l, armed_s = False, False
    last_trigger = pd.Timestamp(0, tz='UTC')

    for ts, row in df.iterrows():
        is_th = not (ts.weekday()==4 and ts.hour>=21 or ts.weekday()==5 or ts.weekday()==6 and ts.hour<12)
        close, atr = float(row["close"]), float(row["atr_1h"])
        
        if active_tid is not None:
            t = journal._get(active_tid)
            h_tp = (float(row["high"]) >= t.tp) if t.direction == "LONG" else (float(row["low"]) <= t.tp)
            h_sl = (float(row["low"]) <= t.sl) if t.direction == "LONG" else (float(row["high"]) >= t.sl)
            if h_tp and h_sl: h_sl, h_tp = True, False
            
            if h_tp or h_sl:
                res = "TP" if h_tp else "SL"
                exit_p = t.tp if h_tp else t.sl
                journal.close_trade(active_tid, ts, exit_p, result=res, commission=exit_p * P["comm"])
                active_tid = None

        if not is_th or row["atr_d_pct"] < P["min_atr_d"]: armed_l = armed_s = False; continue
        if (ts - last_trigger) < timedelta(hours=P["cooldown_h"]): continue
        
        if (row["ema89_d"] > row["ema144_d"]) and (row["z_d"] >= -P["z_d_lim"]) and row["z_1h"] <= -P["z_arm_1h"] and row["rsi_2h"] <= P["rsi_arm_l"]: armed_l = True
        if (row["ema89_d"] < row["ema144_d"]) and (row["z_d"] <= P["z_d_lim"]) and row["z_1h"] >= P["z_arm_1h"] and row["rsi_2h"] >= P["rsi_arm_s"]: armed_s = True
        
        if row["z_1h"] > 0: armed_l = False
        if row["z_1h"] < 0: armed_s = False

        if active_tid is None:
            if armed_l and row["rsi_2h"] >= P["rsi_fire_l"]:
                d_tp, d_sl = atr*P["tp_m"], atr*P["sl_m"]
                active_tid = journal.open_trade(ts, "LONG", close, tp=close+d_tp, sl=close-d_sl, trade_type="BTCZGAP", metadata={"z_d": round(row["z_d"], 2), "rsi": round(row["rsi_2h"], 2)})
                armed_l, last_trigger = False, ts
            elif armed_s and row["rsi_2h"] <= P["rsi_fire_s"]:
                d_tp, d_sl = atr*P["tp_m"], atr*P["sl_m"]
                active_tid = journal.open_trade(ts, "SHORT", close, tp=close-d_tp, sl=close+d_sl, trade_type="BTCZGAP", metadata={"z_d": round(row["z_d"], 2), "rsi": round(row["rsi_2h"], 2)})
                armed_s, last_trigger = False, ts
                
    return journal

df_raw = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])
df_raw.set_index(pd.to_datetime(df_raw["timestamp"], utc=True), inplace=True)
df = precompute(df_raw)
journal = simulate(df)

# Filtramos los últimos 60 días para el display
df_all = journal.to_dataframe()
cutoff = pd.Timestamp('2026-01-01', tz='UTC')
df_recent = df_all[df_all["Entry_TS"] >= cutoff]

print("\n--- REPORTE ESTÁNDAR DE OPERACIONES (Últimos 60 días) ---")
if not df_recent.empty:
    # Mostramos el display estandar pero filtrado manualmente para no saturar
    print(df_recent.to_string(index=False))
    
    # Metricas del periodo filtrado
    total_pnl = df_recent["PnL"].sum()
    print(f"\nPNL ACUMULADO (Periodo): {total_pnl:,.2f}")
    print(f"TRADES (Periodo):      {len(df_recent)}")
else:
    print("No hay operaciones en el periodo solicitado.")
