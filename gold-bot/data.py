"""Download XAUUSD prices from Dukascopy and store 1-minute bid/ask candles.

Dukascopy serves LZMA-compressed .bi5 files; MM in the URL is zero-based
(January = 00) and XAUUSD prices are stored * 1000.

- Daily minute candles (2 requests per day, used by default):
    .../XAUUSD/YYYY/MM/DD/{BID,ASK}_candles_min_1.bi5
  24-byte records: seconds into the day (uint32), open, close, low, high
  (uint32), volume (float32).
- Hourly ticks (fallback for recent days that have no candle file yet):
    .../XAUUSD/YYYY/MM/DD/HHh_ticks.bi5
  20-byte records: ms into the hour, ask, bid (uint32), ask/bid volume (float32).

Dukascopy rate-limits aggressively, so requests run one at a time with a
pause, and a 429 backs off for up to several minutes.

Usage:
    python data.py --start 2025-10-27 --end 2026-10-27
"""

from __future__ import annotations

import argparse
import lzma
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

BASE = "https://datafeed.dukascopy.com/datafeed/XAUUSD/{y}/{m:02d}/{d:02d}/"
POINT = 1000.0
DATA_DIR = Path(__file__).parent / "data"
PAUSE = 1.0  # seconds between requests

CANDLE = np.dtype([("t", ">u4"), ("o", ">u4"), ("c", ">u4"), ("l", ">u4"), ("h", ">u4"), ("v", ">f4")])
TICK = np.dtype([("ms", ">u4"), ("ask", ">u4"), ("bid", ">u4"), ("askv", ">f4"), ("bidv", ">f4")])

_session = requests.Session()


def _url(day: date, name: str) -> str:
    return BASE.format(y=day.year, m=day.month - 1, d=day.day) + name


def fetch(url: str) -> bytes | None:
    """GET a .bi5 file; None if it does not exist. Retries with long back-off."""
    for attempt in range(10):
        time.sleep(PAUSE)
        try:
            r = _session.get(url, timeout=30)
            if r.status_code == 404:
                return None
            r.raise_for_status()
            return lzma.decompress(r.content) if r.content else b""
        except requests.RequestException:
            if attempt == 9:
                raise
            time.sleep(min(15 * 2 ** attempt, 300))  # rate limit (429) or dropped connection
    return None


def fetch_day_candles(day: date) -> pd.DataFrame | None:
    """1-minute bid/ask candles for one UTC day; None if Dukascopy has no file."""
    sides = {}
    for side in ("BID", "ASK"):
        raw = fetch(_url(day, f"{side}_candles_min_1.bi5"))
        if raw is None:
            return None
        a = np.frombuffer(raw, dtype=CANDLE)
        idx = pd.Timestamp(day, tz="UTC") + pd.to_timedelta(a["t"].astype(np.int64), unit="s")
        s = side.lower()
        sides[s] = pd.DataFrame({f"{s}_open": a["o"] / POINT, f"{s}_high": a["h"] / POINT,
                                 f"{s}_low": a["l"] / POINT, f"{s}_close": a["c"] / POINT,
                                 f"{s}_vol": a["v"]}, index=idx)
    bars = sides["bid"].join(sides["ask"], how="inner")
    # Dukascopy fills closed-market minutes with flat zero-volume candles
    bars["ticks"] = (bars["bid_vol"] + bars["ask_vol"] > 0).astype(int)
    bars = bars[bars["ticks"] > 0].drop(columns=["bid_vol", "ask_vol"])
    return bars


def fetch_hour_ticks(day: date, hour: int) -> pd.DataFrame:
    raw = fetch(_url(day, f"{hour:02d}h_ticks.bi5"))
    if not raw:
        return pd.DataFrame()
    a = np.frombuffer(raw, dtype=TICK)
    base = pd.Timestamp(datetime(day.year, day.month, day.day, hour, tzinfo=timezone.utc))
    idx = base + pd.to_timedelta(a["ms"].astype(np.int64), unit="ms")
    return pd.DataFrame({"ask": a["ask"] / POINT, "bid": a["bid"] / POINT}, index=idx)


def ticks_to_bars(ticks: pd.DataFrame, rule: str = "1min") -> pd.DataFrame:
    """Aggregate ticks into bid and ask OHLC candles plus tick count."""
    if ticks.empty:
        return pd.DataFrame()
    bid = ticks["bid"].resample(rule).ohlc().add_prefix("bid_")
    ask = ticks["ask"].resample(rule).ohlc().add_prefix("ask_")
    n = ticks["bid"].resample(rule).count().rename("ticks")
    bars = pd.concat([bid, ask, n], axis=1)
    return bars[bars["ticks"] > 0]


def day_file(day: date) -> Path:
    return DATA_DIR / f"XAUUSD_{day:%Y-%m-%d}_1m.csv.gz"


def download_day(day: date) -> Path | None:
    path = day_file(day)
    if path.exists():
        return path
    if day.weekday() == 5:  # Saturday: market closed
        return None
    bars = fetch_day_candles(day)
    if bars is None:
        parts = [p for p in (fetch_hour_ticks(day, h) for h in range(24)) if not p.empty]
        bars = ticks_to_bars(pd.concat(parts)) if parts else pd.DataFrame()
    DATA_DIR.mkdir(exist_ok=True)
    bars.to_csv(path, compression="gzip")
    return path


def download(start: date, end: date) -> None:
    days = [start + timedelta(d) for d in range((end - start).days + 1)]
    for i, day in enumerate(days, 1):
        download_day(day)
        if i % 20 == 0 or i == len(days):
            print(f"  {i}/{len(days)} days", flush=True)


def load_bars(start: str | None = None, end: str | None = None, rule: str = "1min") -> pd.DataFrame:
    """Load cached 1-minute bars, optionally resampled (e.g. '5min')."""
    files = sorted(DATA_DIR.glob("XAUUSD_*_1m.csv.gz"))
    frames = [pd.read_csv(f, index_col=0, parse_dates=True) for f in files]
    frames = [f for f in frames if not f.empty]
    if not frames:
        raise FileNotFoundError("No data cached; run `python data.py` first.")
    bars = pd.concat(frames).sort_index()
    bars = bars[~bars.index.duplicated()]
    if start:
        bars = bars[bars.index >= pd.Timestamp(start, tz="UTC")]
    if end:
        bars = bars[bars.index < pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1)]
    if rule != "1min":
        bars = resample(bars, rule)
    return bars


def resample(bars: pd.DataFrame, rule: str) -> pd.DataFrame:
    agg = {}
    for side in ("bid", "ask"):
        agg.update({f"{side}_open": "first", f"{side}_high": "max",
                    f"{side}_low": "min", f"{side}_close": "last"})
    agg["ticks"] = "sum"
    out = bars.resample(rule).agg(agg)
    return out[out["ticks"] > 0]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--start", default=(date.today() - timedelta(days=365)).isoformat())
    p.add_argument("--end", default=(date.today() - timedelta(days=1)).isoformat())
    a = p.parse_args()
    print(f"Downloading XAUUSD {a.start} .. {a.end}")
    download(date.fromisoformat(a.start), date.fromisoformat(a.end))


if __name__ == "__main__":
    main()
