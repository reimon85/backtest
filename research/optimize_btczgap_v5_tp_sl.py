import pandas as pd
import numpy as np
import itertools
from datetime import timedelta

P_BASE = {
    "ema_m": 89, "ema_s": 144, "z_per_d": 34, "z_d_lim": 0.9,
    "rsi_per": 4, "rsi_arm": 15, "rsi_fire": 40,
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
    df['atr_1h'] = df['tr'].rolling(P_BASE["atr_per"]).mean()
    df_d = df.resample('D').agg({'high':'max', 'low':'min', 'close':'last'}).dropna()
    df_d['ema89'], df_d['ema144'] = df_d['close'].ewm(span=P_BASE["ema_m"]).mean(), df_d['close'].ewm(span=P_BASE["ema_s"]).mean()
    mu_d, std_d = df_d['close'].rolling(P_BASE["z_per_d"]).mean(), df_d['close'].rolling(P_BASE["z_per_d"]).std().replace(0, np.nan)
    df_d['z_d'] = (df_d['close'] - mu_d) / std_d
    df_d['atr_d_pct'] = (df_d['high'] - df_d['low']).rolling(14).mean() / df_d['close']
    daily = df_d[['ema89', 'ema144', 'z_d', 'atr_d_pct']].shift(1).reindex(df.index, method='ffill')
    df['ema89_d'], df['ema144_d'], df['z_d'], df['atr_d_pct'] = daily['ema89'], daily['ema144'], daily['z_d'], daily['atr_d_pct']
    df_2h = df['close'].resample('2H').last().to_frame()
    df_2h['rsi'] = _rsi_wilder(df_2h['close'], P_BASE["rsi_per"])
    df['rsi_2h'] = df_2h['rsi'].reindex(df.index, method='ffill')
    return df

def simulate(df, tp_m, sl_m):
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
            if h_tp: trades.append((t["tp_dist"]/close) - (P_BASE["comm"]*2))
            elif h_sl: trades.append(-(t["sl_dist"]/close) - (P_BASE["comm"]*2))
            else: rem.append(t)
        active = rem
        if not is_th or row["atr_d_pct"] < P_BASE["min_atr_d"]: armed_l = armed_s = False; continue
        if (ts - last_trigger) < timedelta(hours=P_BASE["cooldown"]): continue
        if (row["ema89_d"] > row["ema144_d"]) and (row["z_d"] >= -P_BASE["z_d_lim"]) and row["rsi_2h"] <= P_BASE["rsi_arm"]: armed_l = True
        if (row["ema89_d"] < row["ema144_d"]) and (row["z_d"] <= P_BASE["z_d_lim"]) and row["rsi_2h"] >= 100-P_BASE["rsi_arm"]: armed_s = True
        if len(active) < P_BASE["max_c"] and not np.isnan(atr):
            if armed_l and row["rsi_2h"] >= P_BASE["rsi_fire"]:
                active.append({"dir":"L", "tp":close+(atr*tp_m), "sl":close-(atr*sl_m), "tp_dist":atr*tp_m, "sl_dist":atr*sl_m})
                armed_l, last_trigger = False, ts
            elif armed_s and row["rsi_2h"] <= 100-P_BASE["rsi_fire"]:
                active.append({"dir":"S", "tp":close-(atr*tp_m), "sl":close+(atr*sl_m), "tp_dist":atr*tp_m, "sl_dist":atr*sl_m})
                armed_s, last_trigger = False, ts
    return trades

df_raw = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])
df_raw.set_index(pd.to_datetime(df_raw["timestamp"], utc=True), inplace=True)
df = precompute(df_raw).dropna(subset=['atr_1h', 'ema89_d'])

results = []
tp_list, sl_list = [1.5, 2.0, 2.5, 3.0], [0.8, 1.0, 1.2, 1.5]

for tp, sl in itertools.product(tp_list, sl_list):
    t = simulate(df, tp, sl)
    if not t: continue
    pnls = np.array(t)
    pf = pnls[pnls > 0].sum() / abs(pnls[pnls < 0].sum()) if pnls[pnls < 0].size > 0 else 0
    results.append({"tp": tp, "sl": sl, "pf": pf, "ret": np.sum(t), "n": len(t)})

res_df = pd.DataFrame(results).sort_values("pf", ascending=False)
print("\nTop 5 Combinaciones TP/SL ATR (v5.5.0):")
print(res_df.head(5))
