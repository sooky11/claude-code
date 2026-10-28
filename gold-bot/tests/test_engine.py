import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engine import Costs, Signals, run, stats  # noqa: E402

SPREAD = 0.30


def make_bars(mids, highs=None, lows=None):
    """Bars with open=close=mid unless highs/lows given; constant spread."""
    n = len(mids)
    mids = np.asarray(mids, float)
    highs = mids if highs is None else np.asarray(highs, float)
    lows = mids if lows is None else np.asarray(lows, float)
    idx = pd.date_range("2026-06-01 08:00", periods=n, freq="5min", tz="UTC")
    d = {}
    for side, off in (("bid", -SPREAD / 2), ("ask", SPREAD / 2)):
        d[f"{side}_open"] = mids + off
        d[f"{side}_high"] = highs + off
        d[f"{side}_low"] = lows + off
        d[f"{side}_close"] = mids + off
    d["ticks"] = 1
    return pd.DataFrame(d, index=idx)


def one_signal(n, at, side, sl=2.0, tp=3.0, max_bars=24):
    s = np.zeros(n, int)
    s[at] = side
    return Signals(side=s, sl_dist=np.full(n, sl), tp_dist=np.full(n, tp), max_bars=max_bars)


def test_long_hits_take_profit_with_spread():
    # entry at bar 1 open ask = 2000.15; TP = 2003.15 needs bid high >= 2003.15
    bars = make_bars([2000] * 5, highs=[2000, 2000, 2004, 2000, 2000])
    t = run(bars, one_signal(5, 0, 1), np.zeros(5, bool))
    assert len(t) == 1
    assert t.iloc[0]["reason"] == "tp"
    assert t.iloc[0]["pnl"] == pytest.approx(3.0)
    assert t.iloc[0]["r"] == pytest.approx(1.5)


def test_stop_first_when_both_hit_in_same_bar():
    bars = make_bars([2000] * 4, highs=[2000, 2000, 2010, 2000], lows=[2000, 2000, 1990, 2000])
    t = run(bars, one_signal(4, 0, 1), np.zeros(4, bool))
    assert t.iloc[0]["reason"] == "sl"
    assert t.iloc[0]["pnl"] == pytest.approx(-2.0)


def test_short_stop_gap_fills_at_open():
    # short entry at bid 1999.85, SL 2001.85; bar 2 opens (ask) at 2005.15 -> gap fill
    bars = make_bars([2000, 2000, 2005, 2005])
    t = run(bars, one_signal(4, 0, -1), np.zeros(4, bool))
    assert t.iloc[0]["reason"] == "sl"
    assert t.iloc[0]["exit"] == pytest.approx(2005.15)


def test_time_exit_pays_spread():
    bars = make_bars([2000] * 10)
    t = run(bars, one_signal(10, 0, 1, max_bars=3), np.zeros(10, bool))
    assert t.iloc[0]["reason"] == "time"
    assert t.iloc[0]["bars"] == 3
    assert t.iloc[0]["pnl"] == pytest.approx(-SPREAD)


def test_flat_mask_closes_position_and_blocks_entry():
    bars = make_bars([2000] * 10)
    flat = np.zeros(10, bool)
    flat[2] = True
    t = run(bars, one_signal(10, 0, 1), flat)
    assert t.iloc[0]["reason"] == "flat"
    assert t.iloc[0]["exit_time"] == bars.index[2]
    assert run(bars, one_signal(10, 2, 1), flat).empty


def test_no_entry_on_last_bar_and_one_position_at_a_time():
    bars = make_bars([2000] * 6)
    s = Signals(side=np.ones(6, int), sl_dist=np.full(6, 2.0), tp_dist=np.full(6, 3.0), max_bars=2)
    t = run(bars, s, np.zeros(6, bool))
    # trades: entry 1 exit 2, entry 4 exit 5; never overlapping
    assert list(t["bars"]) == [2, 2]
    assert (t["entry_time"].iloc[1:].values > t["exit_time"].iloc[:-1].values).all()


def test_costs_spread_mult_and_slippage():
    bars = make_bars([2000] * 10)
    t = run(bars, one_signal(10, 0, 1, max_bars=3), np.zeros(10, bool),
            Costs(spread_mult=2.0, slippage=0.05, commission=0.1))
    assert t.iloc[0]["pnl"] == pytest.approx(-(2 * SPREAD + 0.1 + 0.1))


def test_stats():
    t = pd.DataFrame({"r": [1.0, -1.0, 2.0, -0.5], "pnl": [1, -1, 2, -0.5]})
    s = stats(t)
    assert s["trades"] == 4
    assert s["pf"] == pytest.approx(3.0 / 1.5)
    assert s["total_r"] == pytest.approx(1.5)
    assert s["max_dd_r"] == pytest.approx(1.0)
