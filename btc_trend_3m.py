#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BTC 3m Trend Detector (Binance) — Completed
- Lấy klines 3 phút (120 nến) + 15 phút (120 nến)
- Tính EMA50, EMA100, RSI(14), MACD(12,26,9), VolumeMA(20)
- Nhận diện swing High/Low -> HH/HL hoặc LH/LL
- Chấm điểm & kết luận xu hướng (3m + 15m)
- Phát hiện tín hiệu đảo chiều: BOS, RSI Divergence, Engulfing, EMA/MACD cross
- Vẽ & lưu hình PNG + GẮN MARKER TÍN HIỆU
- TÍNH SL/TP (theo swing gần nhất) + R:R
- Chống gửi trùng tín hiệu, tùy chọn confirm MTF (15m)
"""

import os, re, requests, numpy as np, pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime as dt
import pytz
import time
from typing import Tuple, List, Dict, Optional

# ============== CONFIG ==============
SYMBOLS = [
    "BTCUSDT",
    "ETHUSDT",
    "SUIUSDT",
    "SOLUSDT"
]
INTERVAL = "3m"
LIMIT = 120

def get_save_paths(symbol: str) -> dict:
    return {
        "price_3m": f"{symbol.lower()}_3m_price.png",
        "price_15m": f"{symbol.lower()}_15m_price.png"
    }
BASE = "https://api.binance.com"
TZ = pytz.timezone("Asia/Ho_Chi_Minh")


TELEGRAM_BOT_TOKEN = "8226246719:AAHXDggFiFYpsgcq1vwTAWv7Gsz1URP4KEU"
#TELEGRAM_CHAT_ID = "-4706073326"
TELEGRAM_CHAT_ID = "-4967023521"

# API Configs
API_TIMEOUT = 20               # Timeout cho API calls (giây)
TELEGRAM_TIMEOUT = 10         # Timeout cho Telegram API (giây)
RETRY_DELAY = 10             # Thời gian chờ giữa các lần retry khi lỗi (giây)
SYMBOL_DELAY = 1             # Delay giữa các lần fetch data cho mỗi coin (giây)

# Tham số vận hành
LOOP_SLEEP_SECONDS = 15         # nghỉ giữa mỗi vòng lặp
SEND_IMAGES = True              # gửi ảnh kèm tín hiệu
MTF_CONFIRM = True              # confirm theo trend 15m
MIN_RR_TO_SEND = 1.0           # chỉ gửi khi R:R >= ngưỡng (có thể đặt 1.2/1.5 tùy khẩu vị)

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
def get_klines(symbol: str, interval=INTERVAL, limit=LIMIT) -> pd.DataFrame:
    max_retries = 3
    for attempt in range(max_retries):
        try:
            url = f"{BASE}/api/v3/klines"
            params = {"symbol": symbol, "interval": interval, "limit": limit}
            r = requests.get(url, params=params, timeout=API_TIMEOUT)
            r.raise_for_status()
            data = r.json()
            
            cols = ["open_time","open","high","low","close","volume",
                   "close_time","qav","trades","taker_base","taker_quote","ignore"]
            df = pd.DataFrame(data, columns=cols)
            
            for c in ["open","high","low","close","volume"]:
                df[c] = df[c].astype(float)
                
            df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True).dt.tz_convert(TZ)
            df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True).dt.tz_convert(TZ)
            
            return df[["open_time","open","high","low","close","volume","close_time"]]
            
        except requests.exceptions.RequestException as e:
            print(f"Error fetching klines for {symbol} (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(RETRY_DELAY)
            else:
                raise

def get_klines_15m(symbol: str, limit=120) -> pd.DataFrame:
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

    return {
        "score":score, "label":label, "last_time":last["open_time"],
        "last_close":last["close"], "ema50":last["ema50"], "ema100":last["ema100"],
        "rsi":last["rsi"], "macd_line":last["macd_line"],
        "macd_signal":last["macd_signal"], "macd_hist":last["macd_hist"],
        "volume":last["volume"], "vol_ma20":last["vol_ma20"], "reasons":reasons
    }

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
def plot_price(df: pd.DataFrame, signals: List[Dict], save_path: str, interval: str = "3m", symbol: str = "BTCUSDT"):
    plt.figure()
    plt.plot(df["open_time"], df["close"], linewidth=1.0, label="Close")
    plt.plot(df["open_time"], df["ema50"], linewidth=1.0, label="EMA50")
    plt.plot(df["open_time"], df["ema100"], linewidth=1.0, label="EMA100")

    # Vẽ các tín hiệu
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
            pts = s.get("price_pts")
            if pts and len(pts)==2:
                xs=[pts[0][0], pts[1][0]]; ys=[pts[0][1], pts[1][1]]
                plt.plot(xs, ys, linewidth=1.0)
                m = "v" if s["side"]=="bearish" else "^"
                plt.scatter([xs[-1]],[ys[-1]], marker=m, s=60)
                plt.annotate("Div", (xs[-1],ys[-1]), xytext=(6,10), textcoords="offset points", fontsize=8)

    plt.title(f"{symbol} – {interval} Price with EMA50/EMA100 (+Signals)")
    plt.xlabel("Time (Asia/Ho_Chi_Minh)"); plt.ylabel("Price")
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M', tz=TZ))
    plt.legend(); plt.tight_layout(); plt.savefig(save_path, dpi=150); plt.close()

def plot_rsi(df: pd.DataFrame, signals: List[Dict], save_path: str):
    plt.figure()
    plt.plot(df["open_time"], df["rsi"], linewidth=1.2)
    plt.axhline(45, linestyle="--", linewidth=0.8)
    plt.axhline(55, linestyle="--", linewidth=0.8)
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

# ================= Messaging =================
def send_telegram_message(message: str, max_retries: int = 3) -> bool:
    if not TELEGRAM_BOT_TOKEN or "YOUR_TELEGRAM_BOT_TOKEN" in TELEGRAM_BOT_TOKEN:
        return False
        
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "HTML",
        "disable_notification": True
    }
    
    for attempt in range(max_retries):
        try:
            r = requests.post(url, data=payload, timeout=TELEGRAM_TIMEOUT)
            r.raise_for_status()
            return True
        except requests.exceptions.RequestException as e:
            print(f"Telegram send error (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(RETRY_DELAY)
    
    return False

def send_telegram_photo(photo_path: str, caption: str = "", max_retries: int = 3) -> bool:
    if not TELEGRAM_BOT_TOKEN or "YOUR_TELEGRAM_BOT_TOKEN" in TELEGRAM_BOT_TOKEN:
        return False
        
    if not os.path.exists(photo_path):
        print(f"Photo file not found: {photo_path}")
        return False
        
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
    
    for attempt in range(max_retries):
        try:
            with open(photo_path, "rb") as photo:
                files = {"photo": photo}
                data = {
                    "chat_id": TELEGRAM_CHAT_ID,
                    "caption": caption,
                    "parse_mode": "HTML",
                    "disable_notification": True
                }
                r = requests.post(url, data=data, files=files, timeout=TELEGRAM_TIMEOUT)
                r.raise_for_status()
                return True
        except (requests.exceptions.RequestException, IOError) as e:
            print(f"Telegram photo send error (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(RETRY_DELAY)
    
    return False

def send_telegram_photo(photo_path: str, caption: str = ""):
    if not TELEGRAM_BOT_TOKEN or "YOUR_TELEGRAM_BOT_TOKEN" in TELEGRAM_BOT_TOKEN:
        return
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

# ================= SL/TP & Helpers =================
def compute_sl_tp(dfp: pd.DataFrame, side: str):
    """SL/TP cơ bản dựa trên swing gần nhất (TP1 = swing đối diện)."""
    h_idx, l_idx = last_two_swing_idx(dfp)
    entry = float(dfp.iloc[-1]["close"])

    sl = tp = None
    if side == "bullish":
        if l_idx: sl = float(dfp.loc[l_idx[-1], "low"])
        if h_idx: tp = float(dfp.loc[h_idx[-1], "high"])
    else:
        if h_idx: sl = float(dfp.loc[h_idx[-1], "high"])
        if l_idx: tp = float(dfp.loc[l_idx[-1], "low"])

    rr = None
    if sl is not None and tp is not None:
        if side == "bullish" and entry > sl:
            denom = (entry - sl)
            rr = (tp - entry) / denom if denom > 0 else None
        elif side == "bearish" and sl > entry:
            denom = (sl - entry)
            rr = (entry - tp) / denom if denom > 0 else None

    return entry, sl, tp, rr

def aligned_with_15m(side: str, label15: str) -> Tuple[bool, str]:
    """Xác định độ đồng pha với khung 15m; trả kèm tag mô tả."""
    if label15 == "UPTREND" and side == "bullish":
        return True, "WITH 15m trend"
    if label15 == "DOWNTREND" and side == "bearish":
        return True, "WITH 15m trend"
    if label15 == "SIDEWAYS / MIXED":
        return True, "15m SIDEWAYS"
    return False, "COUNTER-TREND 15m"

def fmt(n: Optional[float], digits=2) -> str:
    return "—" if n is None else f"{n:.{digits}f}"

# ================= Main =================
def main():
    last_signal_ids = {symbol: None for symbol in SYMBOLS}
    first_run = True
    error_counts = {symbol: 0 for symbol in SYMBOLS}
    
    while True:
        try:
            for symbol in SYMBOLS:
                save_paths = get_save_paths(symbol)
                
                # ===== 3m =====
                df = get_klines(symbol=symbol)
                df["ema50"] = ema(df["close"], 50)
                df["ema100"] = ema(df["close"], 100)
                df["rsi"] = rsi(df["close"], 14)
                macd_line, macd_signal, macd_hist = macd(df["close"], 12, 26, 9)
                df["macd_line"] = macd_line
                df["macd_signal"] = macd_signal
                df["macd_hist"] = macd_hist
                df["vol_ma20"] = df["volume"].rolling(20).mean()
                df = swing_points(df, window=3)
                dfp = df.tail(120).copy()

                result3 = score_trend(dfp)
                signals = collect_reversal_signals(dfp)

                plot_price(dfp, signals, save_paths["price_3m"], interval="3m", symbol=symbol)

                # ===== 15m =====
                df15 = get_klines_15m(symbol=symbol)
                df15["ema50"] = ema(df15["close"], 50)
                df15["ema100"] = ema(df15["close"], 100)
            df15["rsi"]    = rsi(df15["close"], 14)
            macd_line15, macd_signal15, macd_hist15 = macd(df15["close"], 12, 26, 9)
            df15["macd_line"] = macd_line15; df15["macd_signal"] = macd_signal15; df15["macd_hist"] = macd_hist15
            df15["vol_ma20"] = df15["volume"].rolling(20).mean()
            df15 = swing_points(df15, window=3)
            dfp15 = df15.tail(120).copy()
            result15 = score_trend(dfp15)

            # ===== Console log =====
            print(f"=== {symbol} Trend Detector ===")
            print(f"Time: {result3['last_time'].strftime('%Y-%m-%d %H:%M:%S %Z')}")
            print(f"3m Close: {result3['last_close']:.2f} | EMA50: {result3['ema50']:.2f} | EMA100: {result3['ema100']:.2f}")
            print(f"3m Volume: {result3['volume']:.0f} | VolMA20: {result3['vol_ma20']:.0f}")
            print(f"3m Score: {result3['score']}  =>  Trend: {result3['label']}")
            print(f"15m Trend: {result15['label']}")
            print(f"Saved: {save_paths['price_3m']}, {save_paths['price_15m']}\n")

            # Vẽ 15m cuối cùng sau khi có result15
            signals15 = collect_reversal_signals(dfp15)
            plot_price(dfp15, signals15, save_paths["price_15m"], interval="15m", symbol=symbol)

            # ===== Xử lý tín hiệu đảo chiều mới nhất =====
            print("--- Reversal signals (latest) ---")
            if not signals:
                print("No strong signals in recent window.")
            else:
                for s in signals[-10:]:
                    print(f"[{s['at'].strftime('%m-%d %H:%M')}] {s['type']} | {s['side'].upper()} | {s.get('note','')}")
                latest = signals[-1]
                signal_id = f"{latest['type']}_{latest['side']}_{latest['at']}"

                # Chỉ xử lý khi tín hiệu mới
                if signal_id != last_signal_ids[symbol]:
                    side = latest['side']  # 'bullish' hoặc 'bearish'
                    entry, sl, tp, rr = compute_sl_tp(dfp, side)

                    # MTF filter
                    ok_15m, tag_15m = aligned_with_15m(side, result15['label'])
                    mtf_ok = (not MTF_CONFIRM) or ok_15m

                    # So ngưỡng R:R
                    rr_ok = (rr is not None) and (rr >= MIN_RR_TO_SEND)

                    # Soạn message
                    signal_time = latest['at'].strftime('%Y-%m-%d %H:%M:%S')
                    header = f"<b>{symbol} {INTERVAL}</b> — {latest['type']} ({side.upper()})"
                    time_line = f"Signal Time: {signal_time}"
                    info3m = (
                        f"Close: {result3['last_close']:.2f} | EMA50/100: "
                        f"{result3['ema50']:.2f}/{result3['ema100']:.2f} | "
                        f"RSI: {result3['rsi']:.2f} | Trend3m: {result3['label']}"
                    )
                    info15m = f"Trend15m: {result15['label']} ({tag_15m})"
                    entry_rr = f"Entry: {fmt(entry)} | R:R = {fmt(rr,2)}"
                    sl_line = f"SL: {fmt(sl)}"
                    tp_line = f"TP1: {fmt(tp)}"

                    note_list = [latest.get('note',''), "MTF OK" if mtf_ok else "MTF FAIL"]
                    if rr is None:
                        note_list.append("R:R N/A")
                    elif rr_ok:
                        note_list.append(f"R:R OK (≥ {MIN_RR_TO_SEND})")
                    else:
                        note_list.append(f"R:R FAIL (< {MIN_RR_TO_SEND})")

                    msg = (
                        f"{header}\n{time_line}\n{info3m}\n{info15m}\n"
                        f"{entry_rr}\n{sl_line}\n{tp_line}\n"
                        f"Note: " + " | ".join([n for n in note_list if n])
                    )

                    # In ra console
                    print(msg)

                    # Gửi Telegram khi thỏa điều kiện hoặc khi là lần chạy đầu tiên
                    should_send = mtf_ok and rr_ok
                    if should_send and (first_run or signal_id != last_signal_ids[symbol]):
                        send_telegram_message(msg)
                        if SEND_IMAGES:
                            send_telegram_photo(save_paths["price_3m"], f"{symbol} Price 3m")
                            send_telegram_photo(save_paths["price_15m"], f"{symbol} Price 15m")
                        last_signal_ids[symbol] = signal_id
                    elif not first_run:
                        # Chỉ set last_signal_ids khi không phải lần chạy đầu
                        last_signal_ids[symbol] = signal_id

                time.sleep(SYMBOL_DELAY)
                error_counts[symbol] = 0  # Reset error count nếu xử lý thành công

            # Reset first_run sau khi đã xử lý tất cả các coin
            first_run = False
            time.sleep(LOOP_SLEEP_SECONDS)

        except Exception as e:
            print(f"[CRITICAL ERROR] {e}")
            time.sleep(RETRY_DELAY)

# ============== Entry ==============
if __name__ == "__main__":
    send_telegram_photo('start_server.png', "Start server detector ....")
    main()
