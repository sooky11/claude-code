"""Bar-by-bar backtest engine with bid/ask fills.

Rules (chosen to avoid look-ahead and to be conservative):
- A signal is computed on the close of bar i; the order fills at the open of
  bar i+1 (long at ask, short at bid).
- Longs exit on bid prices, shorts on ask prices.
- If stop loss and take profit are both inside one bar, the stop is assumed
  to be hit first. A gap through the stop fills at the bar open.
- Positions are closed at the close of a bar flagged in `flat`
  (session end, daily break, weekend).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Costs:
    spread_mult: float = 1.0   # 1.5 = spread 50% wider than recorded
    slippage: float = 0.0      # $/oz lost on entry and on exit
    commission: float = 0.0    # $/oz per round trip


@dataclass
class Signals:
    side: np.ndarray      # +1 long, -1 short, 0 none (per bar, on close)
    sl_dist: np.ndarray   # stop distance in $/oz
    tp_dist: np.ndarray   # target distance in $/oz
    max_bars: int = 24    # time stop in bars


def apply_costs(bars: pd.DataFrame, costs: Costs) -> dict[str, np.ndarray]:
    out = {}
    for f in ("open", "high", "low", "close"):
        bid = bars[f"bid_{f}"].to_numpy(float)
        ask = bars[f"ask_{f}"].to_numpy(float)
        mid, half = (bid + ask) / 2, (ask - bid) / 2 * costs.spread_mult
        out[f"bid_{f}"], out[f"ask_{f}"] = mid - half, mid + half
    return out


def run(bars: pd.DataFrame, sig: Signals, flat: np.ndarray, costs: Costs = Costs()) -> pd.DataFrame:
    p = apply_costs(bars, costs)
    bo, bh, bl, bc = p["bid_open"], p["bid_high"], p["bid_low"], p["bid_close"]
    ao, ah, al, ac = p["ask_open"], p["ask_high"], p["ask_low"], p["ask_close"]
    times = bars.index
    n = len(bars)
    side_arr = sig.side
    candidates = np.flatnonzero(side_arr != 0)
    trades = []
    next_free = 0  # first bar index at which a new signal may be taken

    for i in candidates:
        if i < next_free or i + 1 >= n or flat[i]:
            continue
        side = int(side_arr[i])
        sl_d, tp_d = float(sig.sl_dist[i]), float(sig.tp_dist[i])
        if not (sl_d > 0 and tp_d > 0):
            continue
        e = i + 1
        if side > 0:
            entry = ao[e] + costs.slippage
            sl, tp = entry - sl_d, entry + tp_d
        else:
            entry = bo[e] - costs.slippage
            sl, tp = entry + sl_d, entry - tp_d

        exit_px, reason, j = None, None, e
        for j in range(e, n):
            if side > 0:
                if bl[j] <= sl:
                    exit_px, reason = min(sl, bo[j]) if j > e else sl, "sl"
                elif bh[j] >= tp:
                    exit_px, reason = tp, "tp"
            else:
                if ah[j] >= sl:
                    exit_px, reason = max(sl, ao[j]) if j > e else sl, "sl"
                elif al[j] <= tp:
                    exit_px, reason = tp, "tp"
            if reason is None and (j - e + 1 >= sig.max_bars or flat[j] or j == n - 1):
                exit_px = bc[j] if side > 0 else ac[j]
                reason = "time" if j - e + 1 >= sig.max_bars else "flat"
            if reason is not None:
                break

        exit_px -= side * costs.slippage
        pnl = side * (exit_px - entry) - costs.commission
        trades.append((times[e], times[j], side, entry, exit_px, sl_d, pnl, pnl / sl_d, reason, j - e + 1))
        next_free = j + 1  # one position at a time; no re-entry on the exit bar

    return pd.DataFrame(trades, columns=["entry_time", "exit_time", "side", "entry", "exit",
                                         "risk", "pnl", "r", "reason", "bars"])


def stats(trades: pd.DataFrame, risk_pct: float = 0.005) -> dict:
    """Summary metrics. P&L is per 1 oz; R = P&L / stop distance."""
    if trades.empty:
        return dict(trades=0, winrate=0.0, pf=0.0, exp_r=0.0, total_r=0.0, max_dd_r=0.0,
                    pnl=0.0, ret_pct=0.0)
    r = trades["r"].to_numpy()
    wins, losses = r[r > 0].sum(), -r[r < 0].sum()
    cum = np.cumsum(r)
    dd = np.max(np.maximum.accumulate(np.concatenate([[0], cum]))[1:] - cum)
    equity = np.prod(1 + risk_pct * r)
    return dict(
        trades=len(r),
        winrate=float((r > 0).mean()),
        pf=float(wins / losses) if losses > 0 else float("inf"),
        exp_r=float(r.mean()),
        total_r=float(r.sum()),
        max_dd_r=float(max(dd, 0)),
        pnl=float(trades["pnl"].sum()),
        ret_pct=float((equity - 1) * 100),
    )
