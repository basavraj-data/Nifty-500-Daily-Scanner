#!/usr/bin/env python3
"""
Nifty 500 Daily Scanner  —  EMA(61, HLC/3) + RSI(60)
-----------------------------------------------------
Entry : Close > EMA(61) of HLC/3   AND   RSI(60) > 60   (checked ~3:00 PM IST)
Exit  : Close crosses below EMA(61) of HLC/3
Alert : New entries, exits and open positions are sent to Telegram.

Open positions are remembered in positions.json so a stock is only reported
as an "entry" once, and is reported as an "exit" when it closes below the EMA.

Env variables:
  TELEGRAM_BOT_TOKEN   bot token from @BotFather
  TELEGRAM_CHAT_ID     your chat id (or a group/channel id)
  FRESH_ONLY=1         optional: only count an entry if the condition became
                       true TODAY (was false yesterday). Default 0.
  FORCE=1              optional: run even if today's candle isn't found
                       (useful for testing on weekends/holidays)
"""
import html
import io
import json
import os
import sys
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
import yfinance as yf

# ---------------- settings ----------------
EMA_LEN = 61
RSI_LEN = 60
RSI_LEVEL = 60
HISTORY = "2y"          # enough bars for EMA/RSI to settle
CHUNK = 100             # tickers per yfinance request

IST = ZoneInfo("Asia/Kolkata")
FRESH_ONLY = os.getenv("FRESH_ONLY", "0") == "1"
FORCE = os.getenv("FORCE", "0") == "1"

BASE = os.path.dirname(os.path.abspath(__file__))
STATE_FILE = os.path.join(BASE, "positions.json")
SYMBOLS_CACHE = os.path.join(BASE, "nifty500_symbols.csv")
LOG_FILE = os.path.join(BASE, "signals_log.csv")

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"}


# ---------------- symbols ----------------
def get_symbols():
    urls = [
        "https://www.niftyindices.com/IndexConstituent/ind_nifty500list.csv",
        "https://nsearchives.nseindia.com/content/indices/ind_nifty500list.csv",
        "https://archives.nseindia.com/content/indices/ind_nifty500list.csv",
    ]
    for u in urls:
        try:
            r = requests.get(u, headers=UA, timeout=20)
            r.raise_for_status()
            df = pd.read_csv(io.StringIO(r.text))
            df.columns = [c.strip() for c in df.columns]
            if "Series" in df.columns:                      # cash segment only
                df = df[df["Series"].astype(str).str.strip() == "EQ"]
            syms = df["Symbol"].astype(str).str.strip().tolist()
            if len(syms) >= 400:
                pd.DataFrame({"Symbol": syms}).to_csv(SYMBOLS_CACHE, index=False)
                print(f"Loaded {len(syms)} symbols from {u}")
                return syms
        except Exception as e:
            print(f"Symbol list failed from {u}: {e}")
    if os.path.exists(SYMBOLS_CACHE):
        syms = pd.read_csv(SYMBOLS_CACHE)["Symbol"].astype(str).tolist()
        print(f"Using cached symbol list ({len(syms)})")
        return syms
    sys.exit("Could not get the Nifty 500 list and no cache exists.")


# ---------------- data ----------------
def download(symbols):
    tickers = [s + ".NS" for s in symbols]
    frames = {}
    for i in range(0, len(tickers), CHUNK):
        chunk = tickers[i:i + CHUNK]
        if len(chunk) == 1:          # keep multi-ticker column layout
            chunk = chunk + ["^NSEI"]
        try:
            data = yf.download(chunk, period=HISTORY, interval="1d",
                               group_by="ticker", auto_adjust=False,
                               threads=True, progress=False)
        except Exception as e:
            print(f"Download error for chunk {i}: {e}")
            continue
        for t in chunk:
            if t == "^NSEI" and "^NSEI" not in tickers:
                continue
            try:
                df = data[t][["Open", "High", "Low", "Close"]].dropna()
                if len(df):
                    frames[t[:-3]] = df
            except Exception:
                pass
    print(f"Got data for {len(frames)}/{len(symbols)} symbols")
    return frames


# ---------------- indicators ----------------
def ema(series, n):
    return series.ewm(span=n, adjust=False).mean()


def rsi(close, n):
    """Wilder's RSI (same smoothing as TradingView / most Indian platforms)."""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    avg_loss = loss.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    rs = avg_gain / avg_loss
    out = 100 - 100 / (1 + rs)
    return out.where(avg_loss != 0, 100.0)


# ---------------- scan ----------------
def scan(frames, positions, today):
    entries, exits, holding = [], [], []
    for sym, df in frames.items():
        if len(df) < EMA_LEN + RSI_LEN + 2:
            continue
        hlc3 = (df["High"] + df["Low"] + df["Close"]) / 3
        e = ema(hlc3, EMA_LEN)
        r = rsi(df["Close"], RSI_LEN)

        close, ev, rv = float(df["Close"].iloc[-1]), float(e.iloc[-1]), float(r.iloc[-1])
        if np.isnan(ev) or np.isnan(rv):
            continue
        cond_today = close > ev and rv > RSI_LEVEL
        cond_yday = (df["Close"].iloc[-2] > e.iloc[-2]) and (r.iloc[-2] > RSI_LEVEL)
        row = {"symbol": sym, "price": round(close, 2), "ema": round(ev, 2),
               "rsi": round(rv, 1), "dist": round((close / ev - 1) * 100, 2)}

        if sym in positions:
            pos = positions[sym]
            row["entry_date"] = pos["entry_date"]
            row["entry_price"] = pos["entry_price"]
            row["pnl"] = round((close / pos["entry_price"] - 1) * 100, 2)
            if close < ev:
                exits.append(row)
            else:
                holding.append(row)
        elif cond_today and (not FRESH_ONLY or not cond_yday):
            entries.append(row)

    for row in exits:
        positions.pop(row["symbol"], None)
    for row in entries:
        positions[row["symbol"]] = {"entry_date": today, "entry_price": row["price"]}

    entries.sort(key=lambda x: -x["rsi"])
    exits.sort(key=lambda x: x["pnl"])
    holding.sort(key=lambda x: -x["pnl"])
    return entries, exits, holding


# ---------------- output ----------------
def build_message(entries, exits, holding, now):
    h = html.escape
    lines = [f"<b>📊 Nifty 500 Scanner</b>",
             f"{now:%d %b %Y, %I:%M %p} IST",
             f"EMA({EMA_LEN}, HLC/3) + RSI({RSI_LEN}) &gt; {RSI_LEVEL}", ""]

    lines.append(f"<b>🟢 NEW ENTRIES ({len(entries)})</b>")
    if entries:
        for x in entries:
            lines.append(f"• <b>{h(x['symbol'])}</b>  ₹{x['price']}  | RSI {x['rsi']} "
                         f"| {x['dist']:+.1f}% vs EMA")
    else:
        lines.append("None today")
    lines.append("")

    lines.append(f"<b>🔴 EXITS – closed below EMA ({len(exits)})</b>")
    if exits:
        for x in exits:
            lines.append(f"• <b>{h(x['symbol'])}</b>  ₹{x['price']}  | EMA ₹{x['ema']} "
                         f"| in {x['entry_date']} @ ₹{x['entry_price']} | {x['pnl']:+.1f}%")
    else:
        lines.append("None today")
    lines.append("")

    lines.append(f"<b>📌 Open positions: {len(holding)}</b>")
    for x in holding:
        lines.append(f"{h(x['symbol'])} {x['pnl']:+.1f}%")
    return "\n".join(lines)


def send_telegram(text):
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat = os.getenv("TELEGRAM_CHAT_ID")
    if not token or not chat:
        print("Telegram not configured — message below:\n")
        print(text)
        return
    # Telegram limit is 4096 chars; split on line boundaries
    parts, cur = [], ""
    for line in text.split("\n"):
        if len(cur) + len(line) + 1 > 3900:
            parts.append(cur)
            cur = ""
        cur += line + "\n"
    parts.append(cur)
    for p in parts:
        r = requests.post(f"https://api.telegram.org/bot{token}/sendMessage",
                          data={"chat_id": chat, "text": p, "parse_mode": "HTML",
                                "disable_web_page_preview": True}, timeout=20)
        if not r.ok:
            print("Telegram error:", r.text)


def append_log(entries, exits, today):
    rows = [{"date": today, "signal": "ENTRY", **x} for x in entries] + \
           [{"date": today, "signal": "EXIT", **x} for x in exits]
    if not rows:
        return
    df = pd.DataFrame(rows)
    df.to_csv(LOG_FILE, mode="a", header=not os.path.exists(LOG_FILE), index=False)


def load_positions():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE) as f:
            return json.load(f)
    return {}


def main():
    now = datetime.now(IST)
    today = now.strftime("%Y-%m-%d")
    positions = load_positions()

    symbols = sorted(set(get_symbols()) | set(positions))   # keep tracking held stocks
    frames = download(symbols)

    # Skip market holidays: today's candle must exist for most stocks
    have_today = sum(1 for df in frames.values() if df.index[-1].strftime("%Y-%m-%d") == today)
    if have_today < 0.5 * max(len(frames), 1) and not FORCE:
        print(f"No candle for {today} (holiday/weekend?). Skipping.")
        return
    # Drop stocks whose latest candle isn't today (suspended etc.)
    if not FORCE:
        frames = {s: d for s, d in frames.items() if d.index[-1].strftime("%Y-%m-%d") == today}

    entries, exits, holding = scan(frames, positions, today)

    with open(STATE_FILE, "w") as f:
        json.dump(positions, f, indent=1, sort_keys=True)
    append_log(entries, exits, today)
    send_telegram(build_message(entries, exits, holding, now))
    print(f"Entries {len(entries)} | Exits {len(exits)} | Open {len(holding)}")


if __name__ == "__main__":
    main()
