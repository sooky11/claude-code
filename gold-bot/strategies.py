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
ENTRY_HOURS = (SESSION_START_UTC, SESSION_END_UTC - 1)  # [start, end) UTC hours for new entries


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
    """New entries allowed within ENTRY_HOURS (default: London open until 19:00 UTC)."""
    h = bars.index.hour
    return (h >= ENTRY_HOURS[0]) & (h < ENTRY_HOURS[1])


def flat_mask(bars: pd.DataFrame, bar_minutes: int = 5) -> np.ndarray:
    """Bars at whose close any open position must be closed."""
    idx = bars.index
    end = (idx + pd.Timedelta(minutes=bar_minutes)).hour >= SESSION_END_UTC
    gap = np.append((idx[1:] - idx[:-1]) > pd.Timedelta(minutes=max(30, 2 * bar_minutes)), True)
    return np.asarray(end | gap)


def bar_delta(bars: pd.DataFrame) -> pd.Timedelta:
    return pd.Series(bars.index[1:200] - bars.index[:199]).min()


def crosses(t0, t1, hh: int, mm: int) -> np.ndarray:
    """Bars that start before hh:mm and end at or after it (local times t0/t1)."""
    x = hh * 60 + mm
    return np.asarray(((t0.hour * 60 + t0.minute) < x) & ((t1.hour * 60 + t1.minute) >= x)
                      | ((t1.hour * 60 + t1.minute) < (t0.hour * 60 + t0.minute)) & (x > t0.hour * 60 + t0.minute))


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


def session_drift(bars, mode="both", long_start=23, long_hours=3, short_start=13, short_hours=6,
                  sl_atr=10.0, tp_atr=1000.0):
    """Time-of-day drift: gold tends to rise in Asian hours and fall in US hours.

    Long from `long_start` UTC for `long_hours`, short from `short_start` UTC for
    `short_hours`; exits on time or on a protective ATR stop. 5-minute bars.
    """
    idx = bars.index
    step = bar_delta(bars)
    nxt = idx + step
    at = lambda h: (nxt.hour == h % 24) & (nxt.minute == 0)  # bar closing right at h:00
    long_ = at(long_start) if mode in ("both", "long") else np.zeros(len(bars), bool)
    short_ = at(short_start) if mode in ("both", "short") else np.zeros(len(bars), bool)
    a = atr(bars).to_numpy()
    side = np.where(long_, 1, np.where(short_, -1, 0))
    side[:200] = 0
    hold = (np.where(side > 0, long_hours, short_hours) * (pd.Timedelta(hours=1) / step)).astype(int)
    # one Signals object needs one max_bars: encode per-trade hold via flat bars
    flat = np.append((idx[1:] - idx[:-1]) > max(pd.Timedelta(minutes=30), 2 * step), True)
    exit_bars = np.zeros(len(bars), bool)
    for i in np.flatnonzero(side):
        j = i + hold[i]
        if j < len(bars):
            exit_bars[j] = True
    return Signals(side=side, sl_dist=a * sl_atr, tp_dist=a * tp_atr, max_bars=10_000,
                   flat=flat | exit_bars)


def trend_dip(bars, trend_days=20, rsi_n=14, rsi_lvl=35, sl_atr=2.0, tp_atr=3.0, max_bars=24):
    """Buy dips when the multi-day trend is up (sell rallies when it is down).

    Trend = close above a `trend_days` moving average of daily closes that is
    itself rising. Entry when RSI comes back from `rsi_lvl` (or 100 - rsi_lvl).
    """
    c = mid(bars, "close")
    daily = c.resample("1D").last().dropna()
    ma = daily.rolling(trend_days).mean()
    up_d = (daily > ma) & (ma > ma.shift(5))
    dn_d = (daily < ma) & (ma < ma.shift(5))
    # yesterday's daily state applies to today's bars (no look-ahead)
    day = c.index.normalize()
    up = up_d.shift(1).reindex(day).fillna(False).to_numpy()
    dn = dn_d.shift(1).reindex(day).fillna(False).to_numpy()
    r = rsi(c, rsi_n)
    long_ = up & ((r.shift() < rsi_lvl) & (r >= rsi_lvl)).to_numpy()
    short_ = dn & ((r.shift() > 100 - rsi_lvl) & (r <= 100 - rsi_lvl)).to_numpy()
    return _finish(bars, long_, short_, sl_atr, tp_atr, max_bars)


def ny_fade(bars, ref="london", entry="09:30", exit_="15:55", min_move_atr=0.0, sl_atr=10.0, tp_atr=1000.0):
    """Fade the move made before the New York session.

    At `entry` New York time, compare price with the reference point (London
    open 03:00 NY, or Globex open 18:00 NY the evening before) and trade the
    opposite direction until `exit_` NY time, with a protective ATR stop.
    """
    c = mid(bars, "close")
    ny = bars.index.tz_convert("America/New_York")
    nxt = ny + bar_delta(bars)
    hh, mm = map(int, entry.split(":"))
    xh, xm = map(int, exit_.split(":"))
    sig_bar = (nxt.hour == hh) & (nxt.minute == mm)       # bar closing at entry time
    exit_bar = crosses(ny, nxt, xh, xm)
    # reference price: close of the bar ending at the reference time (same NY trading day)
    rh, roll = (3, 0) if ref == "london" else (18, 1)
    ref_bar = (nxt.hour == rh) & (nxt.minute == 0)
    trade_day = (ny + pd.Timedelta(hours=6)).normalize()   # 18:00 NY starts the next trading day
    ref_px = pd.Series(np.where(ref_bar, c.to_numpy(), np.nan), index=bars.index)
    ref_px = ref_px.groupby(trade_day.tz_localize(None)).ffill()
    move = c - ref_px
    a = atr(bars)
    big = (move.abs() > min_move_atr * a).to_numpy()
    long_ = sig_bar & big & (move < 0).to_numpy()
    short_ = sig_bar & big & (move > 0).to_numpy()
    side = np.where(long_, 1, np.where(short_, -1, 0))
    side[:200] = 0
    flat = np.append((bars.index[1:] - bars.index[:-1]) > max(pd.Timedelta(minutes=30), 2 * bar_delta(bars)), True)
    return Signals(side=side, sl_dist=a.to_numpy() * sl_atr, tp_dist=a.to_numpy() * tp_atr,
                   max_bars=10_000, flat=flat | exit_bar)


def _first_per_day(mask: np.ndarray, day) -> np.ndarray:
    return mask & (pd.Series(mask).groupby(np.asarray(day)).cumsum().to_numpy() == 1)


def asian_range(bars, range_end=7, trade_until=11, mode="break", sl_atr=1.5, tp_atr=2.0, max_bars=24):
    """Asian-session range (00:00 to `range_end` London time); at London, trade the
    first break of it (`break`) or the first failed break back inside (`fade`)."""
    c = mid(bars, "close").to_numpy()
    h, l = mid(bars, "high").to_numpy(), mid(bars, "low").to_numpy()
    ldn = bars.index.tz_convert("Europe/London")
    day = ldn.normalize().tz_localize(None)
    in_rng = ldn.hour < range_end
    after = (ldn.hour >= range_end) & (ldn.hour < trade_until)
    df = pd.DataFrame({"day": day, "h": np.where(in_rng, h, np.nan), "l": np.where(in_rng, l, np.nan)})
    hi = df.groupby("day")["h"].transform("max").to_numpy()
    lo = df.groupby("day")["l"].transform("min").to_numpy()
    prev_c = np.roll(c, 1)
    if mode == "break":
        long_, short_ = after & (c > hi), after & (c < lo)
    else:  # previous close was outside the range, this close is back inside
        long_ = after & (prev_c < lo) & (c >= lo)
        short_ = after & (prev_c > hi) & (c <= hi)
    return _finish(bars, _first_per_day(long_, day), _first_per_day(short_, day), sl_atr, tp_atr, max_bars)


def prev_day_levels(bars, mode="fade", sl_atr=1.5, tp_atr=2.0, max_bars=24):
    """Yesterday's high/low (trading day rolls at 17:00 New York): trade the first
    break (`break`) or the first failed break back inside (`fade`)."""
    c = mid(bars, "close")
    ny = bars.index.tz_convert("America/New_York")
    day = (ny + pd.Timedelta(hours=7)).normalize().tz_localize(None)
    g = pd.DataFrame({"h": mid(bars, "high").to_numpy(), "l": mid(bars, "low").to_numpy(), "day": day})
    dh, dl = g.groupby("day")["h"].max(), g.groupby("day")["l"].min()
    ph = dh.shift(1).reindex(day).to_numpy()
    pl = dl.shift(1).reindex(day).to_numpy()
    cv = c.to_numpy()
    prev_c = np.roll(cv, 1)
    if mode == "break":
        long_ = (cv > ph) & (prev_c <= ph)
        short_ = (cv < pl) & (prev_c >= pl)
    else:
        long_ = (prev_c < pl) & (cv >= pl)
        short_ = (prev_c > ph) & (cv <= ph)
    return _finish(bars, _first_per_day(long_, day), _first_per_day(short_, day), sl_atr, tp_atr, max_bars)


def vol_breakout(bars, open_ref="globex", k=0.5, exit_="15:55", sl_atr=10.0, tp_atr=1000.0):
    """Volatility breakout: enter when price moves k x yesterday's range away from
    today's open (Globex 18:00 NY, or London 03:00 NY); hold until `exit_` NY time."""
    c = mid(bars, "close")
    ny = bars.index.tz_convert("America/New_York")
    nxt = ny + bar_delta(bars)
    day = (ny + pd.Timedelta(hours=6)).normalize().tz_localize(None)  # trading day starts 18:00 NY
    g = pd.DataFrame({"h": mid(bars, "high").to_numpy(), "l": mid(bars, "low").to_numpy(),
                      "o": mid(bars, "open").to_numpy(), "day": day})
    rng = (g.groupby("day")["h"].max() - g.groupby("day")["l"].min()).shift(1).reindex(day).to_numpy()
    if open_ref == "globex":
        op = g.groupby("day")["o"].transform("first").to_numpy()
        started = np.ones(len(bars), bool)
    else:
        is_open = (ny.hour == 3) & (ny.minute == 0)
        op = pd.Series(np.where(is_open, g["o"], np.nan)).groupby(day).ffill().to_numpy()
        started = ~np.isnan(op)
    xh, xm = map(int, exit_.split(":"))
    before_exit = (ny.hour * 60 + ny.minute) < (xh * 60 + xm - 60)  # no entries in the last hour
    cv = c.to_numpy()
    prev_c = np.roll(cv, 1)
    up, dn = op + k * rng, op - k * rng
    long_ = started & before_exit & (cv > up) & (prev_c <= up)
    short_ = started & before_exit & (cv < dn) & (prev_c >= dn)
    long_, short_ = _first_per_day(long_, day), _first_per_day(short_, day)
    a = atr(bars).to_numpy()
    side = np.where(long_, 1, np.where(short_, -1, 0))
    side[:300] = 0
    exit_bar = crosses(ny, nxt, xh, xm)
    flat = np.append((bars.index[1:] - bars.index[:-1]) > max(pd.Timedelta(minutes=30), 2 * bar_delta(bars)), True)
    return Signals(side=side, sl_dist=a * sl_atr, tp_dist=a * tp_atr, max_bars=10_000, flat=flat | exit_bar)


def tsmom(bars, lookback=50, hold=5, mode="both", sl_atr=3.0, tp_atr=1000.0):
    """Time-series momentum on daily bars: go with the sign of the `lookback`-day
    return, hold `hold` days (ATR stop), re-decide after each exit."""
    c = mid(bars, "close")
    ret = c / c.shift(lookback) - 1
    long_ = (ret > 0).to_numpy()
    short_ = (ret < 0).to_numpy() if mode == "both" else np.zeros(len(bars), bool)
    side = np.where(long_, 1, np.where(short_, -1, 0))
    side[: max(lookback, 20)] = 0
    a = atr(bars).to_numpy()
    flat = np.append((bars.index[1:] - bars.index[:-1]) > pd.Timedelta(days=5), True)
    return Signals(side=side, sl_dist=a * sl_atr, tp_dist=a * tp_atr, max_bars=hold, flat=flat)


STRATEGIES = {
    "ema_rsi_pullback": ema_rsi_pullback,
    "opening_range_breakout": opening_range_breakout,
    "bollinger_reversion": bollinger_reversion,
    "donchian_breakout": donchian_breakout,
    "session_drift": session_drift,
    "trend_dip": trend_dip,
    "ny_fade": ny_fade,
    "asian_range": asian_range,
    "prev_day_levels": prev_day_levels,
    "vol_breakout": vol_breakout,
    "tsmom": tsmom,
}

_EXITS = {"sl_atr": [1.0, 1.5, 2.0], "tp_atr": [1.0, 1.5, 2.0, 3.0], "max_bars": [12, 24, 48]}

GRIDS = {
    "ema_rsi_pullback": {"trend_n": [50, 100, 200], "rsi_n": [5, 7, 14], "rsi_lvl": [20, 25, 30], **_EXITS},
    "opening_range_breakout": {"range_min": [15, 30, 60], **_EXITS},
    "bollinger_reversion": {"bb_n": [20, 40], "bb_k": [2.0, 2.5, 3.0], **_EXITS},
    "donchian_breakout": {"don_n": [12, 24, 48], "trend_n": [50, 200], **_EXITS},
    "session_drift": {"mode": ["long", "short", "both"], "long_start": [23, 0], "long_hours": [2, 3, 4, 7],
                      "short_start": [13, 14], "short_hours": [3, 5, 6], "sl_atr": [5.0, 10.0, 20.0],
                      "tp_atr": [1000.0]},
    "ny_fade": {"ref": ["london", "globex"], "entry": ["09:30", "10:00", "10:30"],
                "exit_": ["12:00", "14:00", "15:55"], "min_move_atr": [0.0, 3.0, 6.0],
                "sl_atr": [5.0, 10.0, 20.0], "tp_atr": [1000.0]},
    "asian_range": {"range_end": [6, 7, 8], "trade_until": [10, 12], "mode": ["break", "fade"], **_EXITS},
    "prev_day_levels": {"mode": ["break", "fade"], **_EXITS},
    "vol_breakout": {"open_ref": ["globex", "london"], "k": [0.2, 0.3, 0.5, 0.7], "exit_": ["12:00", "15:55"],
                     "sl_atr": [5.0, 10.0, 20.0], "tp_atr": [1000.0]},
    "tsmom": {"lookback": [10, 20, 50, 100, 200], "hold": [1, 5, 10, 20], "mode": ["long", "both"],
              "sl_atr": [2.0, 3.0, 5.0], "tp_atr": [1000.0]},
    "trend_dip": {"trend_days": [10, 20, 50], "rsi_n": [7, 14], "rsi_lvl": [25, 30, 35, 40], **_EXITS},
}
