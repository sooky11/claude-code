"""Candidate scalping strategies for XAUUSD on 5-minute bars.

Each strategy is `fn(bars, **params) -> Signals` and has a small parameter
grid in GRIDS. All indicators use mid prices and only data up to the bar's
close, so signals are free of look-ahead.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from engine import Signals

SESSION_START_UTC = 7    # London open
SESSION_END_UTC = 20     # flat by 20:00 UTC (before the daily break)


def mid(bars: pd.DataFrame, f: str) -> pd.Series:
    return (bars[f"bid_{f}"] + bars[f"ask_{f}"]) / 2


def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def atr(bars: pd.DataFrame, n: int = 14) -> pd.Series:
    h, l, c = mid(bars, "high"), mid(bars, "low"), mid(bars, "close")
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def rsi(s: pd.Series, n: int) -> pd.Series:
    d = s.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def entry_window(bars: pd.DataFrame) -> np.ndarray:
    """New entries allowed from London open until one hour before session end."""
    h = bars.index.hour
    return (h >= SESSION_START_UTC) & (h < SESSION_END_UTC - 1)


def flat_mask(bars: pd.DataFrame, bar_minutes: int = 5) -> np.ndarray:
    """Bars at whose close any open position must be closed."""
    idx = bars.index
    end = (idx + pd.Timedelta(minutes=bar_minutes)).hour >= SESSION_END_UTC
    gap = np.append((idx[1:] - idx[:-1]) > pd.Timedelta(minutes=30), True)
    return np.asarray(end | gap)


def _finish(bars, long_, short_, sl_atr, tp_atr, max_bars) -> Signals:
    a = atr(bars).to_numpy()
    ok = entry_window(bars)
    side = np.where(long_ & ok, 1, np.where(short_ & ok, -1, 0))
    side[:200] = 0  # indicator warm-up
    return Signals(side=side, sl_dist=a * sl_atr, tp_dist=a * tp_atr, max_bars=max_bars)


def ema_rsi_pullback(bars, trend_n=100, rsi_n=7, rsi_lvl=30, sl_atr=1.5, tp_atr=1.5, max_bars=24):
    """Trade with the EMA trend after RSI returns from an extreme."""
    c = mid(bars, "close")
    e = ema(c, trend_n)
    r = rsi(c, rsi_n)
    up = (c > e) & (e > e.shift(5))
    dn = (c < e) & (e < e.shift(5))
    long_ = up & (r.shift() < rsi_lvl) & (r >= rsi_lvl)
    short_ = dn & (r.shift() > 100 - rsi_lvl) & (r <= 100 - rsi_lvl)
    return _finish(bars, long_.to_numpy(), short_.to_numpy(), sl_atr, tp_atr, max_bars)


def opening_range_breakout(bars, range_min=30, sl_atr=1.5, tp_atr=2.0, max_bars=24):
    """Break of the first `range_min` minutes of the London (07:00) and NY (13:30) sessions."""
    c = mid(bars, "close").to_numpy()
    h, l = mid(bars, "high").to_numpy(), mid(bars, "low").to_numpy()
    idx = bars.index
    minutes = idx.hour * 60 + idx.minute
    day = idx.normalize()
    long_ = np.zeros(len(bars), bool)
    short_ = np.zeros(len(bars), bool)
    for open_min, close_min in ((7 * 60, 12 * 60), (13 * 60 + 30, 18 * 60)):
        in_range = (minutes >= open_min) & (minutes < open_min + range_min)
        after = (minutes >= open_min + range_min) & (minutes < close_min)
        df = pd.DataFrame({"day": day, "h": np.where(in_range, h, np.nan),
                           "l": np.where(in_range, l, np.nan)})
        hi = df.groupby("day")["h"].transform("max").to_numpy()
        lo = df.groupby("day")["l"].transform("min").to_numpy()
        brk_up = after & (c > hi)
        brk_dn = after & (c < lo)
        # only the first breakout per session per day
        first_up = pd.Series(brk_up).groupby(day).cumsum().to_numpy() == 1
        first_dn = pd.Series(brk_dn).groupby(day).cumsum().to_numpy() == 1
        long_ |= brk_up & first_up
        short_ |= brk_dn & first_dn
    return _finish(bars, long_, short_, sl_atr, tp_atr, max_bars)


def bollinger_reversion(bars, bb_n=20, bb_k=2.5, sl_atr=1.5, tp_atr=1.5, max_bars=24):
    """Fade a close outside the band once price closes back inside."""
    c = mid(bars, "close")
    m = c.rolling(bb_n).mean()
    sd = c.rolling(bb_n).std()
    upper, lower = m + bb_k * sd, m - bb_k * sd
    long_ = (c.shift() < lower.shift()) & (c >= lower)
    short_ = (c.shift() > upper.shift()) & (c <= upper)
    return _finish(bars, long_.to_numpy(), short_.to_numpy(), sl_atr, tp_atr, max_bars)


def donchian_breakout(bars, don_n=24, trend_n=100, sl_atr=1.5, tp_atr=2.0, max_bars=24):
    """Close beyond the prior N-bar high/low in the direction of the EMA trend."""
    c = mid(bars, "close")
    hh = mid(bars, "high").rolling(don_n).max().shift()
    ll = mid(bars, "low").rolling(don_n).min().shift()
    e = ema(c, trend_n)
    long_ = (c > hh) & (c.shift() <= hh) & (c > e)
    short_ = (c < ll) & (c.shift() >= ll) & (c < e)
    return _finish(bars, long_.to_numpy(), short_.to_numpy(), sl_atr, tp_atr, max_bars)


STRATEGIES = {
    "ema_rsi_pullback": ema_rsi_pullback,
    "opening_range_breakout": opening_range_breakout,
    "bollinger_reversion": bollinger_reversion,
    "donchian_breakout": donchian_breakout,
}

_EXITS = {"sl_atr": [1.0, 1.5, 2.0], "tp_atr": [1.0, 1.5, 2.0, 3.0], "max_bars": [12, 24, 48]}

GRIDS = {
    "ema_rsi_pullback": {"trend_n": [50, 100, 200], "rsi_n": [5, 7, 14], "rsi_lvl": [20, 25, 30], **_EXITS},
    "opening_range_breakout": {"range_min": [15, 30, 60], **_EXITS},
    "bollinger_reversion": {"bb_n": [20, 40], "bb_k": [2.0, 2.5, 3.0], **_EXITS},
    "donchian_breakout": {"don_n": [12, 24, 48], "trend_n": [50, 200], **_EXITS},
}
