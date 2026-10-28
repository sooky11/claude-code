"""Gold swing bot: daily time-series momentum, long only.

Rule (same as strategies.tsmom, the best candidate of the research):
- After each completed day: if the close is above the close `lookback` days
  ago and there is no position -> buy.
- Protective stop at entry - sl_atr x ATR(14, daily).
- Exit after `hold` completed days, or earlier when the stop is hit.

Modes:
- paper        : everything is simulated locally (default).
- demo-orders  : also sends real orders to the OANDA account (use a practice account).

Usage:
    python bot.py --once      # process the latest completed day(s)
    python bot.py --loop      # check every hour
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

import data
from strategies import atr, mid

STATE_DIR = Path(__file__).parent / "state"


@dataclass
class Params:
    lookback: int = 20
    hold: int = 20
    sl_atr: float = 2.0
    risk_pct: float = 1.0          # % of equity risked per trade (distance to stop)
    start_equity: float = 10_000.0


def daily_from_dukascopy(days: int = 400) -> pd.DataFrame:
    """Completed UTC days: hourly history plus minute files for the current month."""
    today = datetime.now(timezone.utc).date()
    first_of_month = today.replace(day=1)
    data.download_hourly(today - timedelta(days=days), first_of_month - timedelta(days=1))
    d = first_of_month
    while d < today:
        data.download_day(d)
        d += timedelta(days=1)
    hourly = data.load_hourly((today - timedelta(days=days)).isoformat())
    hourly = hourly[hourly.index < pd.Timestamp(first_of_month, tz="UTC")]
    minute = data.load_bars(first_of_month.isoformat()) if d > first_of_month else pd.DataFrame()
    bars = pd.concat([hourly, data.resample(minute, "1h")]) if not minute.empty else hourly
    daily = data.resample(bars, "1D")
    return daily[daily.index < pd.Timestamp(today, tz="UTC")]


def load_daily(source: str = "auto") -> tuple[pd.DataFrame, str]:
    from oanda import Oanda
    o = Oanda()
    if source in ("auto", "oanda") and o.configured:
        return o.candles("D", 400), "OANDA"
    if source == "oanda":
        raise RuntimeError("OANDA_API_KEY is not set")
    return daily_from_dukascopy(), "Dukascopy"


class SwingBot:
    def __init__(self, params: Params | None = None, mode: str = "paper", state_dir: Path = STATE_DIR):
        self.p = params or Params()
        self.mode = mode
        self.dir = state_dir
        self.dir.mkdir(parents=True, exist_ok=True)
        self.state_file = self.dir / "state.json"
        self.trades_file = self.dir / "trades.csv"
        self.state = self._load()

    # ---------------------------------------------------------------- state
    def _load(self) -> dict:
        if self.state_file.exists():
            return json.loads(self.state_file.read_text())
        return {"position": None, "pending": None, "last_day": None, "equity": self.p.start_equity, "log": [],
                "params": asdict(self.p)}

    def save(self) -> None:
        self.state["log"] = self.state["log"][-300:]
        self.state_file.write_text(json.dumps(self.state, indent=2, default=str))

    def log(self, day, msg: str) -> None:
        stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")
        self.state["log"].append(f"{stamp} UTC | dag {pd.Timestamp(day):%Y-%m-%d} | {msg}")

    def trades(self) -> pd.DataFrame:
        if not self.trades_file.exists():
            return pd.DataFrame(columns=["entry_time", "exit_time", "entry", "exit", "units", "pnl", "r", "reason"])
        return pd.read_csv(self.trades_file, parse_dates=["entry_time", "exit_time"])

    def _record(self, pos: dict, exit_time, exit_px: float, reason: str) -> None:
        pnl = (exit_px - pos["entry"]) * pos["units"]
        r = (exit_px - pos["entry"]) / pos["risk"] if pos["risk"] else 0.0
        row = pd.DataFrame([{"entry_time": pos["entry_time"], "exit_time": exit_time, "entry": pos["entry"],
                             "exit": exit_px, "units": pos["units"], "pnl": round(pnl, 2), "r": round(r, 3),
                             "reason": reason}])
        row.to_csv(self.trades_file, mode="a", header=not self.trades_file.exists(), index=False)
        self.state["equity"] = round(self.state["equity"] + pnl, 2)

    # ---------------------------------------------------------------- logic
    def size(self, risk: float) -> float:
        """Ounces so that hitting the stop costs risk_pct of equity. Paper trading
        allows 0.01 oz; real OANDA orders need whole ounces (rounded down)."""
        oz = self.state["equity"] * self.p.risk_pct / 100 / risk
        return float(math.floor(oz)) if self.mode == "demo-orders" else math.floor(oz * 100) / 100

    def signal(self, daily: pd.DataFrame) -> dict:
        """Signal on the last completed day."""
        c = mid(daily, "close")
        a = atr(daily)
        if len(c) <= self.p.lookback + 14:
            return {"buy": False, "reason": "te weinig data"}
        now, past = c.iloc[-1], c.iloc[-1 - self.p.lookback]
        buy = bool(now > past)
        why = (f"slot {now:.2f} {'>' if buy else '<='} slot {self.p.lookback} dagen terug {past:.2f}"
               f" -> {'KOPEN' if buy else 'NIET kopen'}")
        return {"buy": buy, "close": float(now), "past": float(past), "atr": float(a.iloc[-1]), "reason": why}

    def step(self, daily: pd.DataFrame, broker=None) -> list[str]:
        """Process completed days not seen before. Safe to call repeatedly."""
        last_seen = pd.Timestamp(self.state["last_day"]) if self.state["last_day"] else None
        new_days = daily.index if last_seen is None else daily.index[daily.index > last_seen]
        if len(new_days) == 0:
            return []
        if last_seen is None:  # first run: start from the latest day only
            new_days = new_days[-1:]
        before = len(self.state["log"])

        for day in new_days:
            bar = daily.loc[day]
            pend = self.state.get("pending")
            if pend is not None:  # paper: fill yesterday's signal at today's open, like the backtest
                entry = float(bar["ask_open"])
                units = self.size(pend["risk"])
                self.state["position"] = {"entry_time": str(day), "entry": entry,
                                          "stop": round(entry - pend["risk"], 3), "risk": pend["risk"],
                                          "units": units, "days_held": 0, "trade_id": None}
                self.state["pending"] = None
                self.log(day, f"GEKOCHT {units} oz @ {entry:.2f} (opening), stop {entry - pend['risk']:.2f}")
            pos = self.state["position"]
            exited = False
            if pos is not None:
                pos["days_held"] += 1
                if bar["bid_low"] <= pos["stop"]:
                    px = min(pos["stop"], bar["bid_open"])  # a gap through the stop fills at the open
                    exited, reason = True, "stop"
                elif pos["days_held"] >= self.p.hold:
                    px, exited, reason = float(bar["bid_close"]), True, "tijd"
                if exited:
                    if self.mode == "demo-orders" and broker is not None and pos.get("trade_id"):
                        if reason == "tijd":
                            broker.close_trade(pos["trade_id"])
                    self._record(pos, day, float(px), reason)
                    self.log(day, f"VERKOCHT {pos['units']} oz @ {px:.2f} ({reason}), "
                                  f"resultaat ${(px - pos['entry']) * pos['units']:+.2f}")
                    self.state["position"] = None
                else:
                    self.log(day, f"positie dag {pos['days_held']}/{self.p.hold}, slot {bar['bid_close']:.2f}, "
                                  f"stop {pos['stop']:.2f}")

            if self.state["position"] is None and self.state.get("pending") is None and not exited:
                sig = self.signal(daily.loc[:day])
                if sig["buy"]:
                    risk = self.p.sl_atr * sig["atr"]
                    units = self.size(risk)
                    if units <= 0:
                        self.log(day, f"{sig['reason']}, maar de positie is te klein bij dit risico "
                                      f"(stop ${risk:.0f}/oz)")
                    elif self.mode == "demo-orders" and broker is not None:
                        # live: market order right after the daily close (fills around the next open)
                        bid, ask = broker.price()
                        res = broker.market_order(units, round(ask - risk, 3))
                        fill = res.get("orderFillTransaction", {})
                        entry = float(fill.get("price", ask))
                        self.state["position"] = {"entry_time": str(day), "entry": entry,
                                                  "stop": round(entry - risk, 3), "risk": risk, "units": units,
                                                  "days_held": 0,
                                                  "trade_id": fill.get("tradeOpened", {}).get("tradeID")}
                        self.log(day, f"{sig['reason']}: GEKOCHT {units} oz @ {entry:.2f} (OANDA)")
                    else:
                        self.state["pending"] = {"signal_day": str(day), "risk": risk}
                        self.log(day, f"{sig['reason']}: koop bij de volgende opening")
                else:
                    self.log(day, sig["reason"])
            self.state["last_day"] = str(day)

        self.save()
        return self.state["log"][before:]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--loop", action="store_true")
    ap.add_argument("--mode", choices=["paper", "demo-orders"], default="paper")
    ap.add_argument("--source", choices=["auto", "oanda", "dukascopy"], default="auto")
    a = ap.parse_args()
    from oanda import Oanda
    broker = Oanda() if a.mode == "demo-orders" else None
    bot = SwingBot(mode=a.mode)
    while True:
        daily, src = load_daily(a.source)
        for line in bot.step(daily, broker) or [f"geen nieuwe dag (laatste: {bot.state['last_day']}, bron {src})"]:
            print(line, flush=True)
        if not a.loop:
            break
        time.sleep(3600)


if __name__ == "__main__":
    main()
