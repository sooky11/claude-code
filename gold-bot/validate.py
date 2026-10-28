"""Run one strategy with fixed parameters on a period and report the checks.

    python validate.py trend_dip '{"trend_days": 20, ...}' --rule 1h --source hourly \
        --start 2024-10-27 --end 2025-10-26
"""

from __future__ import annotations

import argparse
import json

import pandas as pd

import data
import research
from engine import Costs, run, stats
from strategies import STRATEGIES, flat_mask


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("strategy")
    ap.add_argument("params")
    ap.add_argument("--rule", default="1h")
    ap.add_argument("--source", choices=["minute", "hourly"], default="hourly")
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--spread-mult", type=float, default=1.0)
    a = ap.parse_args()

    if a.source == "hourly":
        bars = data.load_hourly(a.start, a.end)
        if a.rule != "1h":
            bars = data.resample(bars, a.rule)
    else:
        bars = data.load_bars(a.start, a.end, rule=a.rule)
    flat = flat_mask(bars, bar_minutes=int(pd.Timedelta(a.rule).total_seconds() // 60))
    fn = STRATEGIES[a.strategy]
    p = json.loads(a.params)
    t = run(bars, fn(bars, **p), flat, Costs(spread_mult=a.spread_mult))
    s = stats(t)
    stress = stats(run(bars, fn(bars, **p), flat, Costs(spread_mult=a.spread_mult * 1.5)))
    mc = research.monte_carlo(t["r"].to_numpy())
    bd = research.breakdown(t)
    months = bd.get("by_month", {})
    print(f"{a.strategy} {a.start}..{a.end}: {research._fmt(s)}")
    print(f"  spread x1.5: {research._fmt(stress)}")
    print(f"  bootstrap P(loss) {mc['p_loss']:.1%}  positive months {sum(v['sum'] > 0 for v in months.values())}/{len(months)}"
          f"  without best 5: {bd.get('total_r_without_top5', 0):+.1f}R")
    print("  by year:", t.groupby(t.entry_time.dt.year)["r"].sum().round(1).to_dict() if len(t) else {})


if __name__ == "__main__":
    main()
