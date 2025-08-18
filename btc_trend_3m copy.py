#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BTC 3m Trend Detector (Binance)
- Lấy klines 3 phút (120 nến)
- Tính EMA50, EMA100, RSI(14), MACD(12,26,9), VolumeMA(20)
- Nhận diện swing High/Low -> HH/HL hoặc LH/LL
- Chấm điểm & kết luận xu hướng
- Phát hiện tín hiệu đảo chiều: BOS, RSI Divergence, Engulfing, EMA/MACD cross
- Vẽ & lưu hình PNG + GẮN MARKER TÍN HIỆU
"""

import re, requests, numpy as np, pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime as dt
import pytz
import time
from typing import Tuple, List, Dict, Optional

SYMBOL = "BTCUSDT"
INTERVAL = "3m"
LIMIT = 120                               # <-- CHỈ LẤY 120 NẾN
SAVE_PRICE_PNG = "btc_3m_price.png"
SAVE_PRICE_15M_PNG = "btc_15m_price.png"
SAVE_RSI_PNG   = "btc_3m_rsi.png"
SAVE_MACD_PNG  = "btc_3m_macd.png"
BASE = "https://api.binance.com"
TZ = pytz.timezone("Asia/Ho_Chi_Minh")

TELEGRAM_BOT_TOKEN = "8226246719:AAHXDggFiFYpsgcq1vwTAWv7Gsz1URP4KEU"
#TELEGRAM_CHAT_ID = "-4706073326"
TELEGRAM_CHAT_ID = "-4967023521"


# ================= Indicators =================
def ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=length, adjust=False).mean()

def rsi(series: pd.Series, length: int = 14) -> pd.Series:
    d = series.diff()
    up = d.clip(lower=0.0)
    down = -d.clip(upper=0.0)
    ma_up = up.ewm(alpha=1/length, adjust=False).mean()
    ma_down = down.ewm(alpha=1/length, adjust=False).mean()
    rs = ma_up / (ma_down.replace(0, np.nan))
    out = 100 - (100 / (1 + rs))
    return out.fillna(50.0)

def macd(series: pd.Series, fast=12, slow=26, signal=9):
    e_fast = ema(series, fast)
    e_slow = ema(series, slow)
    macd_line = e_fast - e_slow
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

# ================= Swings =================
def swing_points(df: pd.DataFrame, window: int = 3) -> pd.DataFrame:
    highs = df["high"].values; lows = df["low"].values
    n = len(df); k = window
    sh = np.full(n, False); sl = np.full(n, False)
    for i in range(k, n-k):
        if highs[i] == np.max(highs[i-k:i+k+1]): sh[i] = True
        if lows[i]  == np.min(lows[i-k:i+k+1]):  sl[i] = True
    out = df.copy()
    out["swing_high"] = sh; out["swing_low"] = sl
    return out

def last_two_swing_idx(df: pd.DataFrame):
    h_idx = list(df.index[df["swing_high"]])
    l_idx = list(df.index[df["swing_low"]])
    return h_idx[-2:], l_idx[-2:]

# ================= Fetch =================
def get_klines(symbol=SYMBOL, interval=INTERVAL, limit=LIMIT) -> pd.DataFrame:
    url = f"{BASE}/api/v3/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    data = r.json()
    cols = ["open_time","open","high","low","close","volume",
            "close_time","qav","trades","taker_base","taker_quote","ignore"]
    df = pd.DataFrame(data, columns=cols)
    for c in ["open","high","low","close","volume"]:
        df[c] = df[c].astype(float)
    df["open_time"]  = pd.to_datetime(df["open_time"], unit="ms", utc=True).dt.tz_convert(TZ)
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True).dt.tz_convert(TZ)
    return df[["open_time","open","high","low","close","volume","close_time"]]

# Lấy dữ liệu 15 phút
def get_klines_15m(symbol=SYMBOL, limit=120) -> pd.DataFrame:
    return get_klines(symbol=symbol, interval="15m", limit=limit)

# ================= Trend Score =================
def score_trend(df: pd.DataFrame) -> dict:
    last = df.iloc[-1]; score = 0; reasons = []

    # EMA alignment & price location (EMA50 vs EMA100)
    if last["ema50"] > last["ema100"]:
        score += 1; reasons.append("EMA50 > EMA100")
    elif last["ema50"] < last["ema100"]:
        score -= 1; reasons.append("EMA50 < EMA100")

    if last["close"] > last["ema50"]:
        score += 1; reasons.append("Close > EMA50")
    elif last["close"] < last["ema50"]:
        score -= 1; reasons.append("Close < EMA50")

    # RSI regime
    if last["rsi"] > 55:
        score += 1; reasons.append("RSI > 55")
    elif last["rsi"] < 45:
        score -= 1; reasons.append("RSI < 45")

    # MACD momentum
    if last["macd_hist"] > 0 and last["macd_line"] > last["macd_signal"]:
        score += 1; reasons.append("MACD up")
    elif last["macd_hist"] < 0 and last["macd_line"] < last["macd_signal"]:
        score -= 1; reasons.append("MACD down")

    # Volume vs VolMA20
    if last["volume"] > last["vol_ma20"]:
        score += (1 if last["close"] >= df.iloc[-2]["close"] else -1)

    # Kết luận với EMA50/EMA100
    if score >= 2 and last["close"] > last["ema50"] > last["ema100"]:
        label = "UPTREND"
    elif score <= -2 and last["close"] < last["ema50"] < last["ema100"]:
        label = "DOWNTREND"
    else:
        label = "SIDEWAYS / MIXED"

    return {"score":score,"label":label,"last_time":last["open_time"],
            "last_close":last["close"],"ema50":last["ema50"],"ema100":last["ema100"],
            "rsi":last["rsi"],"macd_line":last["macd_line"],
            "macd_signal":last["macd_signal"],"macd_hist":last["macd_hist"],
            "volume":last["volume"],"vol_ma20":last["vol_ma20"]}

# ================= Reversal Signals =================
def detect_bos(df: pd.DataFrame) -> List[Dict]:
    sig = []
    h_idx, l_idx = last_two_swing_idx(df)
    if len(h_idx)==2 and len(l_idx)==2:
        h1,h2 = h_idx[-1], h_idx[-2]
        l1,l2 = l_idx[-1], l_idx[-2]
        hh_hl = (df.loc[h1,"high"]>df.loc[h2,"high"]) and (df.loc[l1,"low"]>df.loc[l2,"low"])
        lh_ll = (df.loc[h1,"high"]<df.loc[h2,"high"]) and (df.loc[l1,"low"]<df.loc[l2,"low"])
        last_close = df.iloc[-1]["close"]
        if hh_hl and last_close < df.loc[l1,"low"]:
            sig.append({"type":"BOS","side":"bearish","at":df.iloc[-1]["open_time"],
                        "level":float(df.loc[l1,"low"]), "idx":int(l1),
                        "note":f"Break HL {df.loc[l1,'low']:.2f}"})
        if lh_ll and last_close > df.loc[h1,"high"]:
            sig.append({"type":"BOS","side":"bullish","at":df.iloc[-1]["open_time"],
                        "level":float(df.loc[h1,"high"]), "idx":int(h1),
                        "note":f"Break LH {df.loc[h1,'high']:.2f}"})
    return sig

def detect_rsi_divergence(df: pd.DataFrame) -> List[Dict]:
    sig = []
    h_idx, l_idx = last_two_swing_idx(df)
    # Bearish: Price HH, RSI LH
    if len(h_idx)==2:
        h2,h1 = h_idx[-2],h_idx[-1]
        if df.loc[h1,"high"]>df.loc[h2,"high"] and df.loc[h1,"rsi"]<df.loc[h2,"rsi"]:
            sig.append({"type":"RSI Divergence","side":"bearish","at":df.loc[h1,"open_time"],
                        "price_pts":[(df.loc[h2,"open_time"],float(df.loc[h2,"high"])),
                                     (df.loc[h1,"open_time"],float(df.loc[h1,"high"]))],
                        "rsi_pts":[(df.loc[h2,"open_time"],float(df.loc[h2,"rsi"])),
                                   (df.loc[h1,"open_time"],float(df.loc[h1,"rsi"]))],
                        "note":"Price HH vs RSI LH"})
    # Bullish: Price LL, RSI HL
    if len(l_idx)==2:
        l2,l1 = l_idx[-2],l_idx[-1]
        if df.loc[l1,"low"]<df.loc[l2,"low"] and df.loc[l1,"rsi"]>df.loc[l2,"rsi"]:
            sig.append({"type":"RSI Divergence","side":"bullish","at":df.loc[l1,"open_time"],
                        "price_pts":[(df.loc[l2,"open_time"],float(df.loc[l2,"low"])),
                                     (df.loc[l1,"open_time"],float(df.loc[l1,"low"]))],
                        "rsi_pts":[(df.loc[l2,"open_time"],float(df.loc[l2,"rsi"])),
                                   (df.loc[l1,"open_time"],float(df.loc[l1,"rsi"]))],
                        "note":"Price LL vs RSI HL"})
    return sig

def detect_engulfing(df: pd.DataFrame, lookback: int = 10) -> List[Dict]:
    sig = []
    sub = df.tail(lookback+1).copy()
    for i in range(1,len(sub)):
        o1,c1 = sub.iloc[i-1]["open"], sub.iloc[i-1]["close"]
        o2,c2 = sub.iloc[i]["open"],  sub.iloc[i]["close"]
        # bullish engulfing gần swing low
        is_bull = (c2>o2) and not (c1>o1) and (max(o2,c2)>=max(o1,c1)) and (min(o2,c2)<=min(o1,c1))
        near_sl = bool(sub.iloc[i]["swing_low"]) or bool(sub.iloc[i-1]["swing_low"])
        if is_bull and near_sl:
            sig.append({"type":"Engulfing","side":"bullish","at":sub.iloc[i]["open_time"],
                        "price":float(c2), "note":"Bullish engulfing @ swing-low"})
        # bearish engulfing gần swing high
        is_bear = (c2<o2) and not (c1<o1) and (max(o2,c2)>=max(o1,c1)) and (min(o2,c2)<=min(o1,c1))
        near_sh = bool(sub.iloc[i]["swing_high"]) or bool(sub.iloc[i-1]["swing_high"])
        if is_bear and near_sh:
            sig.append({"type":"Engulfing","side":"bearish","at":sub.iloc[i]["open_time"],
                        "price":float(c2), "note":"Bearish engulfing @ swing-high"})
    return sig

def detect_ema_cross(df: pd.DataFrame, within:int=20)->List[Dict]:
    """Golden/Death cross (EMA50/EMA100) trong within nến gần nhất."""
    sig=[]; e50=df["ema50"].values; e100=df["ema100"].values
    n=len(df); start=max(1,n-within)
    for i in range(start,n):
        if (e50[i-1]<=e100[i-1]) and (e50[i]>e100[i]):
            sig.append({"type":"EMA Cross","side":"bullish","at":df.iloc[i]["open_time"],
                        "price":float(df.iloc[i]["close"]), "note":"Golden cross (50>100)"})
        if (e50[i-1]>=e100[i-1]) and (e50[i]<e100[i]):
            sig.append({"type":"EMA Cross","side":"bearish","at":df.iloc[i]["open_time"],
                        "price":float(df.iloc[i]["close"]), "note":"Death cross (50<100)"})
    return sig

def detect_macd_cross(df: pd.DataFrame, within:int=20)->List[Dict]:
    sig=[]; n=len(df); start=max(1,n-within)
    ml=df["macd_line"].values; sg=df["macd_signal"].values
    for i in range(start,n):
        if (ml[i-1]<=sg[i-1]) and (ml[i]>sg[i]):
            sig.append({"type":"MACD Cross","side":"bullish","at":df.iloc[i]["open_time"],
                        "value":float(ml[i]), "note":"MACD up-cross"})
        if (ml[i-1]>=sg[i-1]) and (ml[i]<sg[i]):
            sig.append({"type":"MACD Cross","side":"bearish","at":df.iloc[i]["open_time"],
                        "value":float(ml[i]), "note":"MACD down-cross"})
    return sig

def collect_reversal_signals(df: pd.DataFrame) -> List[Dict]:
    sig = []
    sig += detect_bos(df)
    sig += detect_rsi_divergence(df)
    sig += detect_engulfing(df, lookback=min(10, len(df)-2))
    sig += detect_ema_cross(df, within=min(20, len(df)-1))
    sig += detect_macd_cross(df, within=min(20, len(df)-1))
    return sorted(sig, key=lambda s: s["at"])

# ================= Plotting (+ markers) =================
def plot_price(df: pd.DataFrame, signals: List[Dict], save_path: str, interval: str = "3m"):
    plt.figure()
    plt.plot(df["open_time"], df["close"], linewidth=1.3, label="Close")
    plt.plot(df["open_time"], df["ema50"], linewidth=1.0, label="EMA50")
    plt.plot(df["open_time"], df["ema100"], linewidth=1.0, label="EMA100")

    # Swings
    sw = df[df["swing_high"] | df["swing_low"]]
    if not sw.empty:
        plt.scatter(sw["open_time"], sw["high"], s=15)
        plt.scatter(sw["open_time"], sw["low"], s=15)

    # Markers for signals on price chart
    for s in signals:
        t = s["at"]
        if s["type"] in ("EMA Cross","Engulfing"):
            y = s.get("price", float(df.loc[df["open_time"]<=t].tail(1)["close"]))
            marker = "^" if s["side"]=="bullish" else "v"
            plt.scatter([t],[y], marker=marker, s=60)
            plt.annotate(s["type"], (t,y), xytext=(6,10), textcoords="offset points", fontsize=8)
        elif s["type"]=="BOS":
            level = s.get("level")
            plt.axhline(level, linestyle="--", linewidth=0.8)
            marker = "v" if s["side"]=="bearish" else "^"
            y = level
            plt.scatter([t],[y], marker=marker, s=60)
            plt.annotate(f"BOS {s['side']}", (t,y), xytext=(6,10), textcoords="offset points", fontsize=8)
        elif s["type"]=="RSI Divergence":
            # nối 2 điểm giá
            pts = s.get("price_pts")
            if pts and len(pts)==2:
                xs=[pts[0][0], pts[1][0]]; ys=[pts[0][1], pts[1][1]]
                plt.plot(xs, ys, linewidth=1.0)
                m = "v" if s["side"]=="bearish" else "^"
                plt.scatter([xs[-1]],[ys[-1]], marker=m, s=60)
                plt.annotate("Div", (xs[-1],ys[-1]), xytext=(6,10), textcoords="offset points", fontsize=8)

    plt.title(f"{SYMBOL} – {interval} Price with EMA50/EMA100 (+Signals)")
    plt.xlabel("Time (Asia/Ho_Chi_Minh)"); plt.ylabel("Price")
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M', tz=TZ))
    plt.legend(); plt.tight_layout(); plt.savefig(save_path, dpi=150); plt.close()

def plot_rsi(df: pd.DataFrame, signals: List[Dict], save_path: str):
    plt.figure()
    plt.plot(df["open_time"], df["rsi"], linewidth=1.2)
    plt.axhline(45, linestyle="--", linewidth=0.8)
    plt.axhline(55, linestyle="--", linewidth=0.8)
    # Vẽ đường phân kỳ trên RSI
    for s in signals:
        if s["type"]=="RSI Divergence":
            pts = s.get("rsi_pts")
            if pts and len(pts)==2:
                xs=[pts[0][0], pts[1][0]]; ys=[pts[0][1], pts[1][1]]
                plt.plot(xs, ys, linewidth=1.0)
                m = "v" if s["side"]=="bearish" else "^"
                plt.scatter([xs[-1]],[ys[-1]], marker=m, s=60)
                plt.annotate("Div", (xs[-1],ys[-1]), xytext=(6,10), textcoords="offset points", fontsize=8)
    plt.title("RSI(14) + Divergence")
    plt.xlabel("Time (Asia/Ho_Chi_Minh)"); plt.ylabel("RSI")
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M', tz=TZ))
    plt.tight_layout(); plt.savefig(save_path, dpi=150); plt.close()

def plot_macd(df: pd.DataFrame, signals: List[Dict], save_path: str):
    plt.figure()
    plt.plot(df["open_time"], df["macd_line"], linewidth=1.0, label="MACD")
    plt.plot(df["open_time"], df["macd_signal"], linewidth=1.0, label="Signal")
    plt.plot(df["open_time"], df["macd_hist"], linewidth=0.8, label="Hist")
    plt.axhline(0, linewidth=0.8)
    # đánh dấu MACD cross
    for s in signals:
        if s["type"]=="MACD Cross":
            t = s["at"]; y = float(df.loc[df["open_time"]<=t].tail(1)["macd_line"])
            marker = "^" if s["side"]=="bullish" else "v"
            plt.scatter([t],[y], marker=marker, s=60)
            plt.annotate("MACD x", (t,y), xytext=(6,10), textcoords="offset points", fontsize=8)
    plt.title("MACD(12,26,9)")
    plt.xlabel("Time (Asia/Ho_Chi_Minh)"); plt.ylabel("Value")
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M', tz=TZ))
    plt.legend(); plt.tight_layout(); plt.savefig(save_path, dpi=150); plt.close()

def send_telegram_message(text: str):
    """Gửi tin nhắn đến Telegram chat."""
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": text,
        "parse_mode": "HTML",
        "disable_web_page_preview": True
    }
    try:
        requests.post(url, data=payload, timeout=10)
    except Exception as e:
        print(f"Telegram send error: {e}")

def send_telegram_photo(photo_path: str, caption: str = ""):
    """Gửi hình ảnh đến Telegram chat."""
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
    try:
        with open(photo_path, "rb") as photo:
            files = {"photo": photo}
            data = {
                "chat_id": TELEGRAM_CHAT_ID,
                "caption": caption,
                "parse_mode": "HTML",
                "disable_notification": True
            }
            requests.post(url, data=data, files=files, timeout=20)
    except Exception as e:
        print(f"Telegram photo send error: {e}")

# ================= Main =================
def main():
    last_signal_id = None  # Lưu id tín hiệu đảo chiều cuối cùng đã gửi
    while True:
        try:
            # 3m chart
            df = get_klines()
            df["ema50"]  = ema(df["close"], 50)
            df["ema100"] = ema(df["close"], 100)
            df["rsi"]    = rsi(df["close"], 14)
            macd_line, macd_signal, macd_hist = macd(df["close"], 12, 26, 9)
            df["macd_line"] = macd_line; df["macd_signal"] = macd_signal; df["macd_hist"] = macd_hist
            df["vol_ma20"] = df["volume"].rolling(20).mean()
            df = swing_points(df, window=3)
            dfp = df.tail(120).copy()
            result  = score_trend(dfp)
            signals = collect_reversal_signals(dfp)
            plot_price(dfp, signals, SAVE_PRICE_PNG, interval="3m")
            plot_rsi(dfp, signals, SAVE_RSI_PNG)
            plot_macd(dfp, signals, SAVE_MACD_PNG)

            # 15m chart
            df15 = get_klines_15m()
            df15["ema50"]  = ema(df15["close"], 50)
            df15["ema100"] = ema(df15["close"], 100)
            df15["rsi"]    = rsi(df15["close"], 14)
            macd_line15, macd_signal15, macd_hist15 = macd(df15["close"], 12, 26, 9)
            df15["macd_line"] = macd_line15; df15["macd_signal"] = macd_signal15; df15["macd_hist"] = macd_hist15
            df15["vol_ma20"] = df15["volume"].rolling(20).mean()
            df15 = swing_points(df15, window=3)
            dfp15 = df15.tail(120).copy()
            signals15 = collect_reversal_signals(dfp15)
            plot_price(dfp15, signals15, SAVE_PRICE_15M_PNG, interval="15m")

            print("=== BTC 3m Trend Detector ===")
            print(f"Time: {result['last_time'].strftime('%Y-%m-%d %H:%M:%S %Z')}")
            print(f"Close: {result['last_close']:.2f} | EMA50: {result['ema50']:.2f} | EMA100: {result['ema100']:.2f}")
            print(f"RSI(14): {result['rsi']:.2f}")
            print(f"MACD: line {result['macd_line']:.4f} | signal {result['macd_signal']:.4f} | hist {result['macd_hist']:.4f}")
            print(f"Volume: {result['volume']:.0f} | VolMA20: {result['vol_ma20']:.0f}")
            print(f"Score: {result['score']}  =>  Trend: {result['label']}")
            print(f"Saved: {SAVE_PRICE_PNG}, {SAVE_RSI_PNG}, {SAVE_MACD_PNG}\n")

            print("--- Reversal signals (latest) ---")
            if not signals:
                print("No strong signals in recent window.")
            else:
                for s in signals[-10:]:
                    print(f"[{s['at'].strftime('%m-%d %H:%M')}] {s['type']} | {s['side'].upper()} | {s.get('note','')}")
                latest = signals[-1]
                # Xác định id duy nhất cho tín hiệu đảo chiều (dựa trên type, side, at)
                signal_id = f"{latest['type']}_{latest['side']}_{latest['at']}"

                # Tính gợi ý SL/TP cho MACD Cross và RSI Divergence (dựa trên swing + EMA50/100)
                sl_tp_text = ""
                if latest['type'] in ("MACD Cross", "RSI Divergence"):
                    try:
                        h_idx, l_idx = last_two_swing_idx(dfp)
                        ema50 = float(result.get('ema50', 0.0))
                        ema100 = float(result.get('ema100', 0.0))
                        if latest['side'] == 'bullish':
                            sl = float(dfp.loc[l_idx[-1], 'low']) if l_idx else None
                            tp_candidates = [v for v in (ema50, ema100) if v and v > 0]
                            if h_idx:
                                tp_candidates.append(float(dfp.loc[h_idx[-1], 'high']))
                            tp = max(tp_candidates) if tp_candidates else None
                        else:
                            sl = float(dfp.loc[h_idx[-1], 'high']) if h_idx else None
                            tp_candidates = [v for v in (ema50, ema100) if v and v > 0]
                            if l_idx:
                                tp_candidates.append(float(dfp.loc[l_idx[-1], 'low']))
                            tp = min(tp_candidates) if tp_candidates else None
                        sl_text = f"{sl:.2f}" if sl is not None else "N/A"
                        tp_text = f"{tp:.2f}" if tp is not None else "N/A"
                        sl_tp_text = f"\n<b>SL/TP:</b> SL: {sl_text} | TP: {tp_text}"
                    except Exception:
                        sl_tp_text = ""

                if signal_id != last_signal_id:
                    if result['label'] == "UPTREND":
                        trend_action = "LONG"
                    elif result['label'] == "DOWNTREND":
                        trend_action = "SHORT"
                    else:
                        trend_action = "NO CLEAR DIRECTION"
                    msg = (
                        f"<b>BTC 3m Signal</b>\n"
                        f"Time: {latest['at'].strftime('%Y-%m-%d %H:%M')}\n"
                        f"Type: <b>{latest['type']}</b>\n"
                        f"Side: <b>{latest['side'].upper()}</b>\n"
                        f"Note: {latest.get('note','')}\n"
                        f"Close: {result['last_close']:.2f} | EMA50: {result['ema50']:.2f} | EMA100: {result['ema100']:.2f}\n"
                        f"RSI: {result['rsi']:.2f} | MACD: {result['macd_line']:.4f}\n"
                        f"<b>Kết luận: {trend_action}</b>" + sl_tp_text
                    )
                    send_telegram_message(msg)
                    # gửi cả ảnh 3m và 15m để tham khảo
                    send_telegram_photo(SAVE_PRICE_PNG)
                    try:
                        send_telegram_photo(SAVE_PRICE_15M_PNG)
                    except Exception:
                        pass
                    last_signal_id = signal_id
            # Chờ 60 giây trước khi lặp lại
            time.sleep(60)
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(60)

if __name__ == "__main__":
    main()
if __name__ == "__main__":
    main()

# requirements for pip install:
# pandas
# numpy
# matplotlib
# requests
# pytz

