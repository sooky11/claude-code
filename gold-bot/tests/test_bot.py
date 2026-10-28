import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bot import Params, SwingBot  # noqa: E402

SPREAD = 0.4


def daily(closes, lows=None, opens=None, start="2026-01-01"):
    closes = np.asarray(closes, float)
    lows = closes - 1 if lows is None else np.asarray(lows, float)
    opens = closes if opens is None else np.asarray(opens, float)
    idx = pd.date_range(start, periods=len(closes), freq="D", tz="UTC")
    d = {}
    for side, off in (("bid", -SPREAD / 2), ("ask", SPREAD / 2)):
        d[f"{side}_open"] = opens + off
        d[f"{side}_high"] = np.maximum(opens, closes) + 1 + off
        d[f"{side}_low"] = lows + off
        d[f"{side}_close"] = closes + off
    d["ticks"] = 1
    return pd.DataFrame(d, index=idx)


def make_bot(tmp_path, **kw):
    return SwingBot(Params(lookback=5, hold=3, sl_atr=2.0, risk_pct=1.0, start_equity=10_000, **kw),
                    state_dir=tmp_path)


def test_buys_on_uptrend_at_next_open_and_sizes_by_risk(tmp_path):
    bars = daily(np.arange(100, 131, 1.0), opens=np.arange(100, 131, 1.0) - 0.5)
    bot = make_bot(tmp_path)
    bot.step(bars.iloc[:30])
    assert bot.state["position"] is None and bot.state["pending"] is not None
    bot.step(bars)
    pos = bot.state["position"]
    assert pos is not None
    assert pos["entry"] == pytest.approx(129.5 + SPREAD / 2)  # ask open of the next day
    # 1% of 10,000 = $100 risk; units = floor(100 / risk distance)
    assert pos["units"] == pytest.approx(int(100 / pos["risk"] * 100) / 100)
    assert pos["stop"] == pytest.approx(pos["entry"] - pos["risk"], abs=1e-3)


def test_no_buy_in_downtrend(tmp_path):
    bot = make_bot(tmp_path)
    bot.step(daily(np.arange(130, 100, -1.0)))
    assert bot.state["position"] is None
    assert "NIET kopen" in bot.state["log"][-1]


def test_time_exit_after_hold_days_and_idempotent(tmp_path):
    up = np.arange(100, 130, 1.0)
    bars = daily(np.concatenate([up, [130, 131, 132, 133]]))
    bot = make_bot(tmp_path)
    bot.step(bars.iloc[:30])
    bot.step(bars.iloc[:31])
    assert bot.state["position"] is not None
    bot.step(bars)
    t = bot.trades()
    assert len(t) == 1 and t.iloc[0]["reason"] == "tijd"
    assert t.iloc[0]["exit"] == pytest.approx(132 - SPREAD / 2)  # 3rd day held, counting the entry day
    # same data again -> nothing happens
    assert bot.step(bars) == []
    assert len(bot.trades()) == 1


def test_stop_and_gap_through_stop(tmp_path):
    up = np.arange(100, 130, 1.0)
    bot = make_bot(tmp_path)
    up2 = np.concatenate([up, [130]])
    bot.step(daily(up))
    bot.step(daily(up2))
    stop = bot.state["position"]["stop"]
    # next day opens far below the stop
    crash = daily(np.concatenate([up2, [110]]), lows=np.concatenate([up2 - 1, [105]]),
                  opens=np.concatenate([up2, [111]]))
    bot.step(crash)
    t = bot.trades()
    assert t.iloc[0]["reason"] == "stop"
    assert t.iloc[0]["exit"] == pytest.approx(min(stop, 111 - SPREAD / 2))
    assert bot.state["equity"] == pytest.approx(10_000 + t.iloc[0]["pnl"])


def test_state_survives_restart(tmp_path):
    bot = make_bot(tmp_path)
    bot.step(daily(np.arange(100, 130, 1.0)))
    bot.step(daily(np.arange(100, 131, 1.0)))
    again = make_bot(tmp_path)
    assert again.state["position"] is not None
    assert again.state["position"] == bot.state["position"]
