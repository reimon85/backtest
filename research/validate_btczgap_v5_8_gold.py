import pandas as pd
import numpy as np
from datetime import timedelta

P = {
    "ema_m": 89, "ema_s": 144, "z_per_d": 34, "z_d_lim": 0.9,
    "z_per_1h": 20, "z_arm_1h": 1.5, # Golden Standard Z
    "rsi_per": 7, "rsi_arm": 40, "rsi_fire": 45, # Golden Standard RSI
    "tp_m": 3.0, "sl_m": 2.0,
    "atr_per": 14, "min_atr_d": 0.012, "cooldown": 24, "comm": 0.0005, "max_c": 5
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
    
    mu_1h = df['close'].rolling(window=P["z_per_1h"]).mean()
    std_1h = df['close'].rolling(window=P["z_per_1h"]).std().replace(0, np.nan)
    df['z_1h'] = (df['close'] - mu_1h) / std_1h
    
    return df

def simulate(df):
    active, trades = [], []
    armed_l, armed_s = False, False
    last_trigger = pd.Timestamp(0, tz='UTC')
    for ts, row in df.iterrows():
        is_th = not (ts.weekday()==4 and ts.hour>=21 or ts.weekday()==5 or ts.weekday()==6 and ts.hour<12)
        hi, lo, close, atr = float(row["high"]), float(row["low"]), float(row["close"]), float(row["atr_1h"])
        rem = []
        for t in active:
            h_tp = (hi >= t["tp"]) if t["dir"] == "L" else (lo <= t["tp"])
            h_sl = (lo <= t["sl"]) if t["dir"] == "L" else (hi >= t["sl"])
            if h_tp and h_sl: h_sl, h_tp = True, False
            if h_tp: trades.append((t["dist_tp"]/close) - (P["comm"]*2))
            elif h_sl: trades.append(-(t["dist_sl"]/close) - (P["comm"]*2))
            else: rem.append(t)
        active = rem

        if not is_th or row["atr_d_pct"] < P["min_atr_d"]: armed_l = armed_s = False; continue
        if (ts - last_trigger) < timedelta(hours=P["cooldown"]): continue

        # Golden Arming
        if (row["ema89_d"] > row["ema144_d"]) and (row["z_d"] >= -P["z_d_lim"]):
            if row["z_1h"] <= -P["z_arm_1h"] and row["rsi_2h"] <= P["rsi_arm"]: armed_l = True
        
        if (row["ema89_d"] < row["ema144_d"]) and (row["z_d"] <= P["z_d_lim"]):
            if row["z_1h"] >= P["z_arm_1h"] and row["rsi_2h"] >= 100-P["rsi_arm"]: armed_s = True
            
        if row["z_1h"] > 0: armed_l = False
        if row["z_1h"] < 0: armed_s = False

        if len(active) < P["max_c"] and not np.isnan(atr):
            if armed_l and row["rsi_2h"] >= P["rsi_fire"]:
                dist_tp, dist_sl = atr * P["tp_m"], atr * P["sl_m"]
                active.append({"dir":"L", "tp":close+dist_tp, "sl":close-dist_sl, "dist_tp":dist_tp, "dist_sl":dist_sl})
                armed_l, last_trigger = False, ts
            elif armed_s and row["rsi_2h"] <= 100-P["rsi_fire"]:
                dist_tp, dist_sl = atr * P["tp_m"], atr * P["sl_m"]
                active.append({"dir":"S", "tp":close-dist_tp, "sl":close+dist_sl, "dist_tp":dist_tp, "dist_sl":dist_sl})
                armed_s, last_trigger = False, ts
                
    return trades

df_raw = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])
df_raw.set_index(pd.to_datetime(df_raw["timestamp"], utc=True), inplace=True)
df = precompute(df_raw).dropna(subset=['atr_1h', 'ema89_d'])
trades = simulate(df)

if trades:
    pnls = np.array(trades)
    print(f"\n--- RESULTADOS GOLDEN v5.8.0 ---")
    print(f"Total Trades:  {len(trades)}")
    print(f"Profit Factor: {pnls[pnls>0].sum()/abs(pnls[pnls<0].sum()):.3f}")
    print(f"Retorno Total: {np.sum(trades)*100:.2f}%")
