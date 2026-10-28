"""Minimal OANDA v20 client for XAU_USD (practice account by default).

Credentials come from the environment or gold-bot/.env:
    OANDA_API_KEY=...
    OANDA_ACCOUNT_ID=101-004-1234567-001
    OANDA_ENV=practice        # or "live" (not recommended)
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import requests

INSTRUMENT = "XAU_USD"
HOSTS = {"practice": "https://api-fxpractice.oanda.com", "live": "https://api-fxtrade.oanda.com"}


def _load_env() -> None:
    env = Path(__file__).parent / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


class Oanda:
    def __init__(self) -> None:
        _load_env()
        self.key = os.environ.get("OANDA_API_KEY", "")
        self.account = os.environ.get("OANDA_ACCOUNT_ID", "")
        self.host = HOSTS[os.environ.get("OANDA_ENV", "practice")]
        self.s = requests.Session()
        self.s.headers["Authorization"] = f"Bearer {self.key}"

    @property
    def configured(self) -> bool:
        return bool(self.key)

    def _get(self, path: str, **params) -> dict:
        r = self.s.get(self.host + path, params=params, timeout=30)
        r.raise_for_status()
        return r.json()

    def _post(self, path: str, body: dict) -> dict:
        r = self.s.post(self.host + path, json=body, timeout=30)
        r.raise_for_status()
        return r.json()

    def candles(self, granularity: str = "D", count: int = 300) -> pd.DataFrame:
        """Completed bid/ask candles in the same column layout as data.py.
        Daily candles align to 17:00 New York (OANDA default)."""
        js = self._get(f"/v3/instruments/{INSTRUMENT}/candles", granularity=granularity,
                       count=count, price="BA")
        rows = [(c["time"], *(float(c[s][f]) for s in ("bid", "ask") for f in "ohlc"), c["volume"])
                for c in js["candles"] if c["complete"]]
        cols = [f"{s}_{f}" for s in ("bid", "ask") for f in ("open", "high", "low", "close")]
        df = pd.DataFrame(rows, columns=["time", *cols, "ticks"])
        df.index = pd.to_datetime(df.pop("time"), utc=True)
        return df

    def price(self) -> tuple[float, float]:
        js = self._get(f"/v3/accounts/{self.account}/pricing", instruments=INSTRUMENT)
        p = js["prices"][0]
        return float(p["bids"][0]["price"]), float(p["asks"][0]["price"])

    def account_summary(self) -> dict:
        return self._get(f"/v3/accounts/{self.account}/summary")["account"]

    def market_order(self, units: float, stop_loss: float) -> dict:
        """Market order; units in ounces (positive = buy). Stop is attached to the trade."""
        body = {"order": {"type": "MARKET", "instrument": INSTRUMENT, "units": f"{units:.0f}",
                          "timeInForce": "FOK", "positionFill": "DEFAULT",
                          "stopLossOnFill": {"price": f"{stop_loss:.3f}"}}}
        return self._post(f"/v3/accounts/{self.account}/orders", body)

    def close_trade(self, trade_id: str) -> dict:
        r = self.s.put(f"{self.host}/v3/accounts/{self.account}/trades/{trade_id}/close", timeout=30)
        r.raise_for_status()
        return r.json()

    def open_trades(self) -> list[dict]:
        return self._get(f"/v3/accounts/{self.account}/openTrades")["trades"]
