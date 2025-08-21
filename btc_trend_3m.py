#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Advanced cryptocurrency trend detector with reversal signals."""

import os
import sys
import time
import numpy as np
import pandas as pd
import pytz
import requests
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from typing import Dict, Tuple, List, Optional

# ============== CONFIG ==============
SYMBOLS = [
    "BTCUSDT",
    "ETHUSDT", 
    "SUIUSDT",
    "SOLUSDT"
]
INTERVAL = "3m"
LIMIT = 120

# API Configs
BASE = "https://api.binance.com"
TZ = pytz.timezone("Asia/Ho_Chi_Minh")
API_TIMEOUT = 20               # Timeout for API calls (seconds)
TELEGRAM_TIMEOUT = 10         # Timeout for Telegram API (seconds)
RETRY_DELAY = 10             # Wait time between retries on error (seconds)
SYMBOL_DELAY = 1             # Delay between fetching data for each coin (seconds)

# Operational parameters
LOOP_SLEEP_SECONDS = 15         # Sleep between main loops
SEND_IMAGES = True              # Send images with signals
MTF_CONFIRM = True              # Confirm according to 15m trend
MIN_RR_TO_SEND = 1.5           # Only send when R:R >= threshold (increased from 1.0)

# Enhanced R:R Configuration
SYMBOL_SPECIFIC_RR = {
    "BTCUSDT": {"min_rr": 1.2, "risk_percent": 1.0, "atr_sl_mult": 1.2, "atr_tp_mult": 2.5},
    "ETHUSDT": {"min_rr": 1.5, "risk_percent": 0.8, "atr_sl_mult": 1.5, "atr_tp_mult": 3.0},
    "SUIUSDT": {"min_rr": 1.8, "risk_percent": 0.5, "atr_sl_mult": 1.8, "atr_tp_mult": 3.5},
    "SOLUSDT": {"min_rr": 1.8, "risk_percent": 0.5, "atr_sl_mult": 1.8, "atr_tp_mult": 3.5}
}
VOLUME_THRESHOLD = 1.2          # Volume ratio for confidence boost
VOLATILITY_THRESHOLD = 2.0      # High volatility threshold

# Telegram configuration
TELEGRAM_BOT_TOKEN = "8226246719:AAHXDggFiFYpsgcq1vwTAWv7Gsz1URP4KEU"
TELEGRAM_CHAT_ID = "-4967023521"

# Global state
error_counts: Dict[str, int] = {s: 0 for s in SYMBOLS}

def get_save_paths(symbol: str) -> dict:
    """Get file paths for saving charts."""
    return {
        "price_3m": f"{symbol.lower()}_3m_price.png",
        "price_15m": f"{symbol.lower()}_15m_price.png"
    }

# ================= Indicators =================
def ema(series: pd.Series, length: int) -> pd.Series:
    """Calculate Exponential Moving Average."""
    return series.ewm(span=length, adjust=False).mean()

def rsi(series: pd.Series, length: int = 14) -> pd.Series:
    """Calculate Relative Strength Index."""
    d = series.diff()
    up = d.clip(lower=0.0)
    down = -d.clip(upper=0.0)
    ma_up = up.ewm(alpha=1/length, adjust=False).mean()
    ma_down = down.ewm(alpha=1/length, adjust=False).mean()
    rs = ma_up / (ma_down.replace(0, np.nan))
    out = 100 - (100 / (1 + rs))
    return out.fillna(50.0)

def macd(series: pd.Series, fast=12, slow=26, signal=9):
    """Calculate MACD indicator."""
    e_fast = ema(series, fast)
    e_slow = ema(series, slow)
    macd_line = e_fast - e_slow
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate Average True Range for volatility measurement."""
    high = df['high']
    low = df['low']
    close = df['close'].shift(1)
    
    tr1 = high - low
    tr2 = abs(high - close)
    tr3 = abs(low - close)
    
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = true_range.rolling(window=period).mean()
    return atr

# ================= Swings =================
def swing_points(df: pd.DataFrame, window: int = 3) -> pd.DataFrame:
    """Identify swing high and low points."""
    highs = df["high"].values
    lows = df["low"].values
    n = len(df)
    k = window
    sh = np.full(n, False)
    sl = np.full(n, False)
    
    for i in range(k, n-k):
        if highs[i] == np.max(highs[i-k:i+k+1]):
            sh[i] = True
        if lows[i] == np.min(lows[i-k:i+k+1]):
            sl[i] = True
    
    out = df.copy()
    out["swing_high"] = sh
    out["swing_low"] = sl
    return out

def last_two_swing_idx(df: pd.DataFrame):
    """Get indices of last two swing highs and lows."""
    h_idx = list(df.index[df["swing_high"]])
    l_idx = list(df.index[df["swing_low"]])
    return h_idx[-2:], l_idx[-2:]

# ================= Fetch Data =================
def get_klines(symbol: str, interval=INTERVAL, limit=LIMIT) -> pd.DataFrame:
    """Fetch candlestick data from Binance API."""
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
    """Get 15m timeframe data."""
    return get_klines(symbol=symbol, interval="15m", limit=limit)

# ================= Trend Score =================
def score_trend(df: pd.DataFrame) -> dict:
    """Score the current trend based on multiple indicators."""
    last = df.iloc[-1]
    score = 0
    reasons = []

    # EMA alignment & price location (EMA50 vs EMA100)
    if last["ema50"] > last["ema100"]:
        score += 1
        reasons.append("EMA50 > EMA100")
    elif last["ema50"] < last["ema100"]:
        score -= 1
        reasons.append("EMA50 < EMA100")

    if last["close"] > last["ema50"]:
        score += 1
        reasons.append("Close > EMA50")
    elif last["close"] < last["ema50"]:
        score -= 1
        reasons.append("Close < EMA50")

    # RSI regime
    if last["rsi"] > 55:
        score += 1
        reasons.append("RSI > 55")
    elif last["rsi"] < 45:
        score -= 1
        reasons.append("RSI < 45")

    # MACD momentum
    if last["macd_hist"] > 0 and last["macd_line"] > last["macd_signal"]:
        score += 1
        reasons.append("MACD up")
    elif last["macd_hist"] < 0 and last["macd_line"] < last["macd_signal"]:
        score -= 1
        reasons.append("MACD down")

    # Volume vs VolMA20
    if last["volume"] > last["vol_ma20"]:
        score += (1 if last["close"] >= df.iloc[-2]["close"] else -1)

    # Trend classification
    if score >= 2 and last["close"] > last["ema50"] > last["ema100"]:
        label = "UPTREND"
    elif score <= -2 and last["close"] < last["ema50"] < last["ema100"]:
        label = "DOWNTREND"
    else:
        label = "SIDEWAYS / MIXED"

    return {
        "score": score, "label": label, "last_time": last["open_time"],
        "last_close": last["close"], "ema50": last["ema50"], "ema100": last["ema100"],
        "rsi": last["rsi"], "macd_line": last["macd_line"],
        "macd_signal": last["macd_signal"], "macd_hist": last["macd_hist"],
        "volume": last["volume"], "vol_ma20": last["vol_ma20"], "reasons": reasons
    }

# ================= Reversal Signals =================
def detect_bos(df: pd.DataFrame) -> List[Dict]:
    """Detect Break of Structure signals."""
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
    """Detect RSI divergence patterns."""
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
    """Detect engulfing candlestick patterns."""
    sig = []
    sub = df.tail(lookback+1).copy()
    for i in range(1,len(sub)):
        o1,c1 = sub.iloc[i-1]["open"], sub.iloc[i-1]["close"]
        o2,c2 = sub.iloc[i]["open"],  sub.iloc[i]["close"]
        # bullish engulfing near swing low
        is_bull = (c2>o2) and not (c1>o1) and (max(o2,c2)>=max(o1,c1)) and (min(o2,c2)<=min(o1,c1))
        near_sl = bool(sub.iloc[i]["swing_low"]) or bool(sub.iloc[i-1]["swing_low"])
        if is_bull and near_sl:
            sig.append({"type":"Engulfing","side":"bullish","at":sub.iloc[i]["open_time"],
                        "price":float(c2), "note":"Bullish engulfing @ swing-low"})
        # bearish engulfing near swing high
        is_bear = (c2<o2) and not (c1<o1) and (max(o2,c2)>=max(o1,c1)) and (min(o2,c2)<=min(o1,c1))
        near_sh = bool(sub.iloc[i]["swing_high"]) or bool(sub.iloc[i-1]["swing_high"])
        if is_bear and near_sh:
            sig.append({"type":"Engulfing","side":"bearish","at":sub.iloc[i]["open_time"],
                        "price":float(c2), "note":"Bearish engulfing @ swing-high"})
    return sig

def detect_ema_cross(df: pd.DataFrame, within:int=20)->List[Dict]:
    """Detect EMA crossover signals."""
    sig=[]
    e50=df["ema50"].values
    e100=df["ema100"].values
    n=len(df)
    start=max(1,n-within)
    for i in range(start,n):
        if (e50[i-1]<=e100[i-1]) and (e50[i]>e100[i]):
            sig.append({"type":"EMA Cross","side":"bullish","at":df.iloc[i]["open_time"],
                        "price":float(df.iloc[i]["close"]), "note":"Golden cross (50>100)"})
        if (e50[i-1]>=e100[i-1]) and (e50[i]<e100[i]):
            sig.append({"type":"EMA Cross","side":"bearish","at":df.iloc[i]["open_time"],
                        "price":float(df.iloc[i]["close"]), "note":"Death cross (50<100)"})
    return sig

def detect_macd_cross(df: pd.DataFrame, within:int=20)->List[Dict]:
    """Detect MACD crossover signals."""
    sig=[]
    n=len(df)
    start=max(1,n-within)
    ml=df["macd_line"].values
    sg=df["macd_signal"].values
    for i in range(start,n):
        if (ml[i-1]<=sg[i-1]) and (ml[i]>sg[i]):
            sig.append({"type":"MACD Cross","side":"bullish","at":df.iloc[i]["open_time"],
                        "value":float(ml[i]), "note":"MACD up-cross"})
        if (ml[i-1]>=sg[i-1]) and (ml[i]<sg[i]):
            sig.append({"type":"MACD Cross","side":"bearish","at":df.iloc[i]["open_time"],
                        "value":float(ml[i]), "note":"MACD down-cross"})
    return sig

def collect_reversal_signals(df: pd.DataFrame) -> List[Dict]:
    """Collect all reversal signals."""
    sig = []
    sig += detect_bos(df)
    sig += detect_rsi_divergence(df)
    sig += detect_engulfing(df, lookback=min(10, len(df)-2))
    sig += detect_ema_cross(df, within=min(20, len(df)-1))
    sig += detect_macd_cross(df, within=min(20, len(df)-1))
    return sorted(sig, key=lambda s: s["at"])

# ================= Plotting =================
def plot_price(df: pd.DataFrame, signals: List[Dict], save_path: str, interval: str = "3m", symbol: str = "BTCUSDT"):
    """Plot price chart with indicators and signals."""
    plt.figure()
    plt.plot(df["open_time"], df["close"], linewidth=1.0, label="Close")
    plt.plot(df["open_time"], df["ema50"], linewidth=1.0, label="EMA50")
    plt.plot(df["open_time"], df["ema100"], linewidth=1.0, label="EMA100")

    # Plot signals
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
                xs=[pts[0][0], pts[1][0]]
                ys=[pts[0][1], pts[1][1]]
                plt.plot(xs, ys, linewidth=1.0)
                m = "v" if s["side"]=="bearish" else "^"
                plt.scatter([xs[-1]],[ys[-1]], marker=m, s=60)
                plt.annotate("Div", (xs[-1],ys[-1]), xytext=(6,10), textcoords="offset points", fontsize=8)

    plt.title(f"{symbol} – {interval} Price with EMA50/EMA100 (+Signals)")
    plt.xlabel("Time (Asia/Ho_Chi_Minh)")
    plt.ylabel("Price")
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M', tz=TZ))
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

# ================= Messaging =================
def send_telegram_message(message: str, max_retries: int = 3) -> bool:
    """Send message to Telegram."""
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
    """Send photo to Telegram."""
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

# ================= SL/TP & Helpers =================
def compute_sl_tp(dfp: pd.DataFrame, side: str, symbol: str = "BTCUSDT") -> dict:
    """Enhanced SL/TP computation with ATR buffer and multiple TP levels."""
    try:
        # Get symbol-specific configuration
        config = SYMBOL_SPECIFIC_RR.get(symbol, {
            "min_rr": 1.5, "atr_sl_mult": 1.5, "atr_tp_mult": 3.0
        })
        
        h_idx, l_idx = last_two_swing_idx(dfp)
        entry = float(dfp.iloc[-1]["close"])
        
        # Calculate ATR for dynamic SL/TP
        atr_series = calculate_atr(dfp)
        current_atr = atr_series.iloc[-1] if not atr_series.isna().iloc[-1] else (entry * 0.02)
        
        sl = tp1 = tp2 = tp3 = None
        strategy_used = "Enhanced_Swing_ATR"
        confidence = 0.7
        
        if side == "bullish":
            # Enhanced SL calculation with ATR buffer
            if l_idx:
                swing_low = float(dfp.loc[l_idx[-1], "low"])
                atr_buffer = current_atr * 0.5  # ATR buffer
                sl = swing_low - atr_buffer
            else:
                # Fallback to ATR-based SL
                sl = entry - (current_atr * config["atr_sl_mult"])
                strategy_used = "ATR_Based"
                confidence = 0.6
            
            # Enhanced TP calculation - multiple levels
            if h_idx:
                swing_high = float(dfp.loc[h_idx[-1], "high"])
                tp1 = swing_high  # Primary target at swing high
                
                # Calculate swing range for extensions
                if l_idx:
                    swing_range = swing_high - float(dfp.loc[l_idx[-1], "low"])
                    tp2 = swing_high + (swing_range * 0.618)  # Fibonacci extension
                    tp3 = swing_high + (swing_range * 1.0)    # 100% extension
                else:
                    tp2 = entry + (current_atr * config["atr_tp_mult"])
                    tp3 = entry + (current_atr * config["atr_tp_mult"] * 1.5)
            else:
                # Fallback to ATR-based TP
                tp1 = entry + (current_atr * config["atr_tp_mult"])
                tp2 = entry + (current_atr * config["atr_tp_mult"] * 1.2)
                tp3 = entry + (current_atr * config["atr_tp_mult"] * 1.5)
                strategy_used = "ATR_Based"
        
        else:  # bearish
            # Enhanced SL calculation with ATR buffer
            if h_idx:
                swing_high = float(dfp.loc[h_idx[-1], "high"])
                atr_buffer = current_atr * 0.5
                sl = swing_high + atr_buffer
            else:
                sl = entry + (current_atr * config["atr_sl_mult"])
                strategy_used = "ATR_Based"
                confidence = 0.6
            
            # Enhanced TP calculation - multiple levels
            if l_idx:
                swing_low = float(dfp.loc[l_idx[-1], "low"])
                tp1 = swing_low
                
                if h_idx:
                    swing_range = float(dfp.loc[h_idx[-1], "high"]) - swing_low
                    tp2 = swing_low - (swing_range * 0.618)
                    tp3 = swing_low - (swing_range * 1.0)
                else:
                    tp2 = entry - (current_atr * config["atr_tp_mult"])
                    tp3 = entry - (current_atr * config["atr_tp_mult"] * 1.5)
            else:
                tp1 = entry - (current_atr * config["atr_tp_mult"])
                tp2 = entry - (current_atr * config["atr_tp_mult"] * 1.2)
                tp3 = entry - (current_atr * config["atr_tp_mult"] * 1.5)
                strategy_used = "ATR_Based"
        
        # Calculate R:R ratio
        rr = None
        if sl is not None and tp1 is not None:
            risk = abs(entry - sl)
            reward = abs(tp1 - entry)
            rr = reward / risk if risk > 0 else None
        
        # Volume and volatility analysis for confidence adjustment
        volume_ratio = dfp.iloc[-1]["volume"] / dfp["volume"].tail(20).mean()
        if volume_ratio > VOLUME_THRESHOLD:
            confidence *= 1.1  # Boost confidence with high volume
        elif volume_ratio < 0.8:
            confidence *= 0.9  # Reduce confidence with low volume
        
        # Volatility adjustment
        avg_atr = atr_series.tail(20).mean()
        volatility_ratio = current_atr / avg_atr if avg_atr > 0 else 1.0
        if volatility_ratio > VOLATILITY_THRESHOLD:
            confidence *= 0.8  # Reduce confidence in high volatility
        
        return {
            "entry": entry,
            "sl": sl,
            "tp1": tp1,
            "tp2": tp2,
            "tp3": tp3,
            "rr": rr,
            "strategy": strategy_used,
            "confidence": confidence,
            "atr": current_atr,
            "volume_ratio": volume_ratio,
            "volatility_ratio": volatility_ratio
        }
        
    except Exception as e:
        print(f"Error in enhanced SL/TP calculation: {e}")
        # Fallback to simple calculation
        h_idx, l_idx = last_two_swing_idx(dfp)
        entry = float(dfp.iloc[-1]["close"])
        
        sl = tp1 = None
        if side == "bullish":
            if l_idx: sl = float(dfp.loc[l_idx[-1], "low"])
            if h_idx: tp1 = float(dfp.loc[h_idx[-1], "high"])
        else:
            if h_idx: sl = float(dfp.loc[h_idx[-1], "high"])
            if l_idx: tp1 = float(dfp.loc[l_idx[-1], "low"])
        
        rr = None
        if sl is not None and tp1 is not None:
            if side == "bullish" and entry > sl:
                rr = (tp1 - entry) / (entry - sl)
            elif side == "bearish" and sl > entry:
                rr = (entry - tp1) / (sl - entry)
        
        return {
            "entry": entry, "sl": sl, "tp1": tp1, "tp2": None, "tp3": None,
            "rr": rr, "strategy": "Fallback_Simple", "confidence": 0.5,
            "atr": None, "volume_ratio": 1.0, "volatility_ratio": 1.0
        }

def aligned_with_15m(side: str, label15: str) -> Tuple[bool, str]:
    """Determine alignment with 15m frame; return with descriptive tag."""
    if label15 == "UPTREND" and side == "bullish":
        return True, "WITH 15m trend"
    if label15 == "DOWNTREND" and side == "bearish":
        return True, "WITH 15m trend"
    if label15 == "SIDEWAYS / MIXED":
        return True, "15m SIDEWAYS"
    return False, "COUNTER-TREND 15m"

def fmt(n: Optional[float], digits=2) -> str:
    """Format number with specified digits."""
    return "—" if n is None else f"{n:.{digits}f}"

# ================= Main Processing =================
def process_symbol(symbol: str, first_run: bool, last_signal_ids: dict) -> Tuple[bool, str]:
    """Process a single symbol to detect and report trading signals.
    
    Args:
        symbol: The trading pair symbol
        first_run: Whether this is the first iteration
        last_signal_ids: Dict tracking the last signal sent for each symbol
        
    Returns:
        Tuple of (success, message)
    """
    try:
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
        df15["rsi"] = rsi(df15["close"], 14)
        macd_line15, macd_signal15, macd_hist15 = macd(df15["close"], 12, 26, 9)
        df15["macd_line"] = macd_line15
        df15["macd_signal"] = macd_signal15
        df15["macd_hist"] = macd_hist15
        df15["vol_ma20"] = df15["volume"].rolling(20).mean()
        df15 = swing_points(df15, window=3)
        dfp15 = df15.tail(120).copy()
        result15 = score_trend(dfp15)

        # ===== Console log =====
        print(f"\n=== {symbol} Trend Detector ===")
        print(f"Time: {result3['last_time'].strftime('%Y-%m-%d %H:%M:%S %Z')}")
        print(f"3m Close: {result3['last_close']:.2f} | EMA50: {result3['ema50']:.2f} | EMA100: {result3['ema100']:.2f}")
        print(f"3m Score: {result3['score']}  =>  Trend: {result3['label']}")
        print(f"15m Trend: {result15['label']}")

        # Plot 15m chart
        signals15 = collect_reversal_signals(dfp15)
        plot_price(dfp15, signals15, save_paths["price_15m"], interval="15m", symbol=symbol)

        if signals:
            latest = signals[-1]
            signal_id = f"{symbol}_{latest['type']}_{latest['side']}_{latest['at']}"
            
            if first_run or signal_id != last_signal_ids.get(symbol):
                side = latest['side']
                rr_data = compute_sl_tp(dfp, side, symbol)
                
                # Get symbol-specific minimum R:R
                symbol_config = SYMBOL_SPECIFIC_RR.get(symbol, {"min_rr": 1.5})
                symbol_min_rr = symbol_config["min_rr"]
                
                ok_15m, tag_15m = aligned_with_15m(side, result15['label'])
                mtf_ok = (not MTF_CONFIRM) or ok_15m
                rr_ok = (rr_data["rr"] is not None) and (rr_data["rr"] >= symbol_min_rr)
                
                # Additional quality filters
                confidence_ok = rr_data["confidence"] >= 0.5
                
                print(f"📊 {symbol} Signal Check:")
                rr_text = f"{rr_data['rr']:.2f}" if rr_data['rr'] is not None else "N/A"
                print(f"   R:R: {rr_text} (min: {symbol_min_rr})")
                print(f"   15m: {ok_15m} | Confidence: {rr_data['confidence']:.2f}")
                print(f"   Volume: {rr_data['volume_ratio']:.2f}x | Volatility: {rr_data['volatility_ratio']:.2f}x")
                
                if mtf_ok and rr_ok and confidence_ok:
                    signal_time = latest['at'].strftime('%Y-%m-%d %H:%M:%S')
                    
                    # Build TP levels text
                    tp_text = f"🟢 TP1: {fmt(rr_data['tp1'])}"
                    if rr_data['tp2']:
                        tp_text += f" | TP2: {fmt(rr_data['tp2'])}"
                    if rr_data['tp3']:
                        tp_text += f" | TP3: {fmt(rr_data['tp3'])}"
                    
                    # Market condition indicators
                    market_indicators = []
                    if rr_data['volume_ratio'] > VOLUME_THRESHOLD:
                        market_indicators.append("🔥 High Volume")
                    elif rr_data['volume_ratio'] < 0.8:
                        market_indicators.append("⚠️ Low Volume")
                    
                    if rr_data['volatility_ratio'] > VOLATILITY_THRESHOLD:
                        market_indicators.append("📈 High Volatility")
                    
                    market_text = " | ".join(market_indicators) if market_indicators else "📊 Normal Conditions"
                    
                    msg_parts = [
                        f"<b>🚨 {symbol} {INTERVAL}</b> — {latest['type']} ({side.upper()})",
                        f"⏰ Time: {signal_time}",
                        f"💰 Entry: {fmt(rr_data['entry'])} | 🔴 SL: {fmt(rr_data['sl'])}",
                        tp_text,
                        f"📊 R:R: <b>{fmt(rr_data['rr'], 2)}</b> | Strategy: {rr_data['strategy']}",
                        f"📈 Trend 3m: {result3['label']} | 15m: {result15['label']} ({tag_15m})",
                        f"🎯 Confidence: {rr_data['confidence']:.1%} | {market_text}",
                        f"📋 Close: {result3['last_close']:.2f} | EMA: {result3['ema50']:.2f}/{result3['ema100']:.2f} | RSI: {result3['rsi']:.1f}"
                    ]
                    
                    # Add ATR info if available
                    if rr_data['atr']:
                        atr_percent = (rr_data['atr'] / rr_data['entry']) * 100
                        msg_parts.append(f"📊 ATR: {rr_data['atr']:.4f} ({atr_percent:.2f}%)")
                    
                    msg = "\n".join(msg_parts)
                    
                    print(f"✅ {symbol}: Sending signal (R:R={rr_data['rr']:.2f}, Confidence={rr_data['confidence']:.2f})")
                    
                    send_telegram_message(msg)
                    if SEND_IMAGES:
                        send_telegram_photo(save_paths["price_3m"], f"{symbol} Price 3m")
                        send_telegram_photo(save_paths["price_15m"], f"{symbol} Price 15m")
                    
                    last_signal_ids[symbol] = signal_id
                else:
                    filter_reasons = []
                    if not rr_ok: 
                        rr_text = f"{rr_data['rr']:.2f}" if rr_data['rr'] is not None else "N/A"
                        filter_reasons.append(f"R:R {rr_text} < {symbol_min_rr}")
                    if not mtf_ok: filter_reasons.append("15m misalignment")
                    if not confidence_ok: filter_reasons.append(f"Low confidence {rr_data['confidence']:.2f}")
                    
                    print(f"❌ {symbol}: Signal filtered - {', '.join(filter_reasons)}")
        
        return True, "Success"
    except Exception as e:
        print(f"[ERROR] Processing {symbol}: {e}")
        error_counts[symbol] += 1
        return False, str(e)

def main():
    """Main loop that processes all configured symbols."""
    last_signal_ids = {symbol: None for symbol in SYMBOLS}
    first_run = True
    
    print("\nStarting Crypto Signal Detector...")
    
    # Send startup notification
    try:
        if os.path.exists('start_server.png'):
            send_telegram_photo('start_server.png', "Start server detector ....")
    except Exception as e:
        print(f"Error sending startup photo: {e}")
    
    while True:
        try:
            for symbol in SYMBOLS:
                success, message = process_symbol(symbol, first_run, last_signal_ids)
                if success:
                    time.sleep(SYMBOL_DELAY)  # Delay only after successful processing
                else:
                    print(f"Failed to process {symbol}: {message}")
            
            first_run = False
            time.sleep(LOOP_SLEEP_SECONDS)
            
        except KeyboardInterrupt:
            print("\nShutting down gracefully...")
            break
        except Exception as e:
            print(f"[CRITICAL ERROR] {e}")
            time.sleep(RETRY_DELAY)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nDetected Ctrl+C, shutting down gracefully...")
    except Exception as e:
        print(f"\nFatal error: {e}")
        sys.exit(1)
