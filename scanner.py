#!/usr/bin/env python3
"""
Nifty 500 Daily Scanner  —  EMA(61, HLC/3) + RSI(14) crossing above 60
-----------------------------------------------------
Entry : RSI(14) crosses above 60 today (yesterday <= 60, today > 60)
        AND daily Close > EMA(61) of HLC/3            (checked ~3:00 PM IST)
Exit  : Close crosses below EMA(61) of HLC/3
Alert : New entries, exits and open positions are sent to Telegram.

Open positions are remembered in positions.json so a stock is only reported
as an "entry" once, and is reported as an "exit" when it closes below the EMA.

Env variables:
  TELEGRAM_BOT_TOKEN   bot token from @BotFather
  TELEGRAM_CHAT_ID     your chat id (or a group/channel id)
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
RSI_LEN = 14
RSI_LEVEL = 60
HISTORY = "2y"          # enough bars for EMA/RSI to settle
CHUNK = 100             # tickers per yfinance request

IST = ZoneInfo("Asia/Kolkata")
FORCE = os.getenv("FORCE", "0") == "1"

BASE = os.path.dirname(os.path.abspath(__file__))
STATE_FILE = os.path.join(BASE, "positions.json")
SYMBOLS_CACHE = os.path.join(BASE, "nifty500_symbols.csv")
LOG_FILE = os.path.join(BASE, "signals_log.csv")

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"}


# ---------------- symbols ----------------
UPLOADED_LIST = os.path.join(BASE, "ind_nifty500list.csv")   # optional: file you upload yourself


def _parse_list(text):
    df = pd.read_csv(io.StringIO(text))
    df.columns = [c.strip() for c in df.columns]
    if "Series" in df.columns:                              # cash segment only
        df = df[df["Series"].astype(str).str.strip() == "EQ"]
    return df["Symbol"].astype(str).str.strip().tolist()


def get_symbols():
    # 1) Try NSE / niftyindices (refreshes the list when index changes)
    s = requests.Session()
    s.headers.update({**UA, "Accept": "text/csv,*/*", "Referer": "https://www.nseindia.com/"})
    try:
        s.get("https://www.nseindia.com", timeout=15)       # picks up cookies NSE expects
    except Exception:
        pass
    urls = [
        "https://nsearchives.nseindia.com/content/indices/ind_nifty500list.csv",
        "https://www.niftyindices.com/IndexConstituent/ind_nifty500list.csv",
        "https://archives.nseindia.com/content/indices/ind_nifty500list.csv",
    ]
    for u in urls:
        try:
            r = s.get(u, timeout=20)
            r.raise_for_status()
            syms = _parse_list(r.text)
            if len(syms) >= 400:
                pd.DataFrame({"Symbol": syms}).to_csv(SYMBOLS_CACHE, index=False)
                print(f"Loaded {len(syms)} symbols from {u}")
                return syms
        except Exception as e:
            print(f"Symbol list failed from {u}: {e}")

    # 2) Fall back to a file in the repo (uploaded by you, or saved by an earlier run)
    for path in (UPLOADED_LIST, SYMBOLS_CACHE):
        if os.path.exists(path):
            with open(path, encoding="utf-8-sig") as f:
                syms = _parse_list(f.read())
            print(f"Using symbol list from {os.path.basename(path)} ({len(syms)})")
            return syms

    msg = ("⚠️ Nifty 500 scanner: could not download the Nifty 500 list from NSE "
           "(NSE blocks cloud servers). Download ind_nifty500list.csv from "
           "niftyindices.com and upload it to the root of your GitHub repo.")
    send_telegram(msg)
    sys.exit(msg)


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
        rv_prev = float(r.iloc[-2])
        rsi_cross_up = rv > RSI_LEVEL and rv_prev <= RSI_LEVEL     # RSI crossed above 60 today
        entry_signal = rsi_cross_up and close > ev
        row = {"symbol": sym, "price": round(close, 2), "ema": round(ev, 2),
               "rsi": round(rv, 1), "rsi_prev": round(rv_prev, 1), "dist": round((close / ev - 1) * 100, 2)}

        if sym in positions:
            pos = positions[sym]
            row["entry_date"] = pos["entry_date"]
            row["entry_price"] = pos["entry_price"]
            row["pnl"] = round((close / pos["entry_price"] - 1) * 100, 2)
            if close < ev:
                exits.append(row)
            else:
                holding.append(row)
        elif entry_signal:
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
             f"Entry: RSI({RSI_LEN}) crossed above {RSI_LEVEL} + Close &gt; EMA({EMA_LEN}, HLC/3)",
             f"Exit: Close &lt; EMA({EMA_LEN}, HLC/3)", ""]

    lines.append(f"<b>🟢 NEW ENTRIES ({len(entries)})</b>")
    if entries:
        for x in entries:
            lines.append(f"• <b>{h(x['symbol'])}</b>  ₹{x['price']}  | RSI {x['rsi_prev']}→{x['rsi']} "
                         f"| EMA ₹{x['ema']} ({x['dist']:+.1f}%)")
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

    lines.append(f"<b>📌 OPEN POSITIONS – still above EMA ({len(holding)})</b>")
    if holding:
        for x in holding:
            lines.append(f"• <b>{h(x['symbol'])}</b>  ₹{x['price']}  | RSI {x['rsi']} "
                         f"| in {x['entry_date']} @ ₹{x['entry_price']} | {x['pnl']:+.1f}%")
    else:
        lines.append("None")
    return "\n".join(lines)


def send_telegram(text):
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat = os.getenv("TELEGRAM_CHAT_ID")
    print(text)
    if not token or not chat:
        print("::error::Telegram secrets TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID are missing.")
        return False
    # Telegram limit is 4096 chars; split on line boundaries
    parts, cur = [], ""
    for line in text.split("\n"):
        if len(cur) + len(line) + 1 > 3900:
            parts.append(cur)
            cur = ""
        cur += line + "\n"
    parts.append(cur)
    # TELEGRAM_CHAT_ID may hold several IDs separated by commas, e.g. "111,222,333"
    chat_ids = [c.strip() for c in chat.replace(";", ",").split(",") if c.strip()]
    delivered = 0
    for cid in chat_ids:
        ok = True
        for p in parts:
            try:
                r = requests.post(f"https://api.telegram.org/bot{token.strip()}/sendMessage",
                                  data={"chat_id": cid, "text": p, "parse_mode": "HTML",
                                        "disable_web_page_preview": True}, timeout=20)
                if not r.ok:
                    print(f"::warning::Telegram error for chat {cid}:", r.text)
                    ok = False
            except Exception as e:
                print(f"::warning::Telegram error for chat {cid}:", e)
                ok = False
        if ok:
            delivered += 1
            print(f"✅ Telegram message sent to chat {cid}")
    print(f"Delivered to {delivered}/{len(chat_ids)} chats")
    # Succeed if at least one person got it (so one blocked user can't stop the scanner)
    return delivered > 0


def append_log(entries, exits, today):
    rows = [{"date": today, "signal": "ENTRY", **x} for x in entries] + \
           [{"date": today, "signal": "EXIT", **x} for x in exits]
    if not rows:
        return
    df = pd.DataFrame(rows)
    df.to_csv(LOG_FILE, mode="a", header=not os.path.exists(LOG_FILE), index=False)


def send_google_sheet(entries, exits, holding, now):
    """Post today's results to the Google Sheet web app (optional)."""
    url = os.getenv("GSHEET_WEBHOOK_URL", "").strip()
    if not url:
        print("Google Sheet not configured (GSHEET_WEBHOOK_URL missing) — skipped.")
        return
    payload = {"secret": os.getenv("GSHEET_SECRET", "").strip(),
               "date": now.strftime("%Y-%m-%d"), "time": now.strftime("%H:%M"),
               "entries": entries, "exits": exits, "open": holding}
    try:
        r = requests.post(url, data=json.dumps(payload, default=float),
                          headers={"Content-Type": "application/json"}, timeout=60)
        try:
            res = r.json()
        except Exception:
            res = {"ok": False, "error": f"HTTP {r.status_code}: {r.text[:200]}"}
        if res.get("ok"):
            print("✅ Google Sheet updated")
        else:
            print("::warning::Google Sheet update failed:", res.get("error"))
    except Exception as e:
        print("::warning::Google Sheet update failed:", e)


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
    if len(frames) < 100:
        msg = (f"⚠️ Nifty 500 scanner: Yahoo Finance returned data for only "
               f"{len(frames)} stocks. Will retry at the next run.")
        send_telegram(msg)
        sys.exit(msg)

    # Skip market holidays: today's candle must exist for most stocks
    have_today = sum(1 for df in frames.values() if df.index[-1].strftime("%Y-%m-%d") == today)
    if have_today < 0.5 * max(len(frames), 1) and not FORCE:
        print(f"No candle for {today} (holiday/weekend?). Skipping.")
        return
    # Drop stocks whose latest candle isn't today (suspended etc.)
    if not FORCE:
        frames = {s: d for s, d in frames.items() if d.index[-1].strftime("%Y-%m-%d") == today}

    entries, exits, holding = scan(frames, positions, today)
    print(f"Entries {len(entries)} | Exits {len(exits)} | Open {len(holding)}")

    # Only save if Telegram delivered, so no alert is ever lost
    if not send_telegram(build_message(entries, exits, holding, now)):
        sys.exit("Telegram message NOT delivered — positions not saved; alerts will repeat next run.")
    with open(STATE_FILE, "w") as f:
        json.dump(positions, f, indent=1, sort_keys=True)
    append_log(entries, exits, today)
    send_google_sheet(entries, exits, holding, now)


if __name__ == "__main__":
    main()
