"""Download XAUUSD tick data from Dukascopy and build bid/ask candles.

Dukascopy serves one LZMA-compressed .bi5 file per hour:
    https://datafeed.dukascopy.com/datafeed/XAUUSD/YYYY/MM/DD/HHh_ticks.bi5
where MM is zero-based (January = 00). Each tick is a 20-byte big-endian
record: ms offset in the hour (uint32), ask (uint32), bid (uint32),
ask volume (float32), bid volume (float32). XAUUSD prices are stored * 1000.

Usage:
    python data.py --start 2025-10-01 --end 2026-10-28
"""

from __future__ import annotations

import argparse
import lzma
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

URL = "https://datafeed.dukascopy.com/datafeed/XAUUSD/{y}/{m:02d}/{d:02d}/{h:02d}h_ticks.bi5"
POINT = 1000.0
DATA_DIR = Path(__file__).parent / "data"


def fetch_hour(session: requests.Session, day: date, hour: int) -> pd.DataFrame:
    url = URL.format(y=day.year, m=day.month - 1, d=day.day, h=hour)
    for attempt in range(4):
        try:
            r = session.get(url, timeout=30)
            if r.status_code == 404:
                return pd.DataFrame()
            r.raise_for_status()
            break
        except requests.RequestException:
            if attempt == 3:
                raise
    if not r.content:
        return pd.DataFrame()
    raw = lzma.decompress(r.content)
    arr = np.frombuffer(raw, dtype=np.dtype([("ms", ">u4"), ("ask", ">u4"), ("bid", ">u4"),
                                             ("askv", ">f4"), ("bidv", ">f4")]))
    base = datetime(day.year, day.month, day.day, hour, tzinfo=timezone.utc)
    idx = pd.to_datetime(base) + pd.to_timedelta(arr["ms"].astype(np.int64), unit="ms")
    return pd.DataFrame({"ask": arr["ask"] / POINT, "bid": arr["bid"] / POINT}, index=idx)


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
    with requests.Session() as s:
        parts = [fetch_hour(s, day, h) for h in range(24)]
    parts = [p for p in parts if not p.empty]
    bars = ticks_to_bars(pd.concat(parts)) if parts else pd.DataFrame()
    DATA_DIR.mkdir(exist_ok=True)
    bars.to_csv(path, compression="gzip")
    return path


def download(start: date, end: date, workers: int = 8) -> None:
    days = [start + timedelta(d) for d in range((end - start).days + 1)]
    with ThreadPoolExecutor(workers) as ex:
        for i, _ in enumerate(ex.map(download_day, days), 1):
            if i % 20 == 0 or i == len(days):
                print(f"  {i}/{len(days)} days")


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
