"""Strategy research: walk-forward, holdout, sensitivity, cost stress, Monte Carlo.

Every parameter combination is backtested once over the full history (signals
are causal, so a trade's outcome does not depend on the window). Windows are
then evaluated by filtering trades on entry time.

Usage:
    python research.py                 # uses cached data in ./data
    python research.py --holdout-days 182 --report report.html
"""

from __future__ import annotations

import argparse
import base64
import io
import itertools
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import data
from engine import Costs, run, stats
import strategies
from strategies import GRIDS, STRATEGIES, flat_mask

MIN_TRAIN_TRADES = 30

CRITERIA = {
    "min_trades": 100,          # holdout trades
    "min_pf_holdout": 1.15,
    "min_pf_walkforward": 1.10,
    "max_p_loss": 0.10,         # bootstrap probability that the holdout is a loss
    "min_positive_months": 4,   # of the 6 holdout months
    "stress_spread": 1.5,       # must still make money with 50% wider spread
    "min_positive_neighbours": 0.5,
}


def combos(grid: dict) -> list[dict]:
    keys = list(grid)
    return [dict(zip(keys, v)) for v in itertools.product(*grid.values())]


def neighbours(grid: dict, p: dict) -> list[dict]:
    """Parameter sets one grid step away in a single dimension."""
    out = []
    for k, vals in grid.items():
        i = vals.index(p[k])
        for j in (i - 1, i + 1):
            if 0 <= j < len(vals):
                out.append({**p, k: vals[j]})
    return out


def key(p: dict) -> tuple:
    return tuple(sorted(p.items()))


def run_all(name: str, bars: pd.DataFrame, flat: np.ndarray, costs: Costs = Costs()) -> dict:
    fn = STRATEGIES[name]
    return {key(p): run(bars, fn(bars, **p), flat, costs) for p in combos(GRIDS[name])}


def window(trades: pd.DataFrame, start, end) -> pd.DataFrame:
    t = trades["entry_time"]
    return trades[(t >= start) & (t < end)]


def pick_robust(name: str, results: dict, start, end) -> tuple[dict, float]:
    """Best parameter set by score averaged with its grid neighbours (plateau, not peak)."""
    grid = GRIDS[name]
    raw = {}
    for k, tr in results.items():
        w = window(tr, start, end)
        raw[k] = w["r"].sum() if len(w) >= MIN_TRAIN_TRADES else -1e9
    best, best_score = None, -np.inf
    for p in combos(grid):
        vals = [raw[key(p)]] + [raw[key(q)] for q in neighbours(grid, p)]
        s = float(np.mean(vals))
        if s > best_score:
            best, best_score = p, s
    return best, best_score


def walk_forward(name: str, results: dict, start, end, train_months: int = 3):
    months = pd.date_range(start.normalize().replace(day=1), end, freq="MS", tz="UTC")
    oos, picks = [], []
    for i in range(train_months, len(months)):
        tr_start, te_start = months[i - train_months], months[i]
        te_end = months[i + 1] if i + 1 < len(months) else end
        p, _ = pick_robust(name, results, tr_start, te_start)
        oos.append(window(results[key(p)], te_start, te_end))
        picks.append({"test_month": te_start.strftime("%Y-%m"), **p})
    trades = pd.concat(oos) if oos else pd.DataFrame(columns=["r"])
    return trades, picks


def monte_carlo(r: np.ndarray, n: int = 2000, seed: int = 1) -> dict:
    rng = np.random.default_rng(seed)
    if len(r) == 0:
        return {"p_loss": 1.0, "dd_median": 0.0, "dd_p95": 0.0}
    boot = rng.choice(r, size=(n, len(r)), replace=True).sum(axis=1)
    dds = []
    for _ in range(n):
        cum = np.cumsum(rng.permutation(r))
        dds.append(np.max(np.maximum.accumulate(np.concatenate([[0], cum]))[1:] - cum))
    return {"p_loss": float((boot <= 0).mean()), "dd_median": float(np.median(dds)),
            "dd_p95": float(np.percentile(dds, 95))}


def breakdown(trades: pd.DataFrame) -> dict:
    if trades.empty:
        return {}
    t = trades.copy()
    t["month"] = t["entry_time"].dt.strftime("%Y-%m")
    hour = t["entry_time"].dt.hour + t["entry_time"].dt.minute / 60
    t["session"] = np.where(hour < 13.5, "London", "New York")
    t["dir"] = np.where(t["side"] > 0, "long", "short")
    r_sorted = np.sort(t["r"].to_numpy())
    return {
        "by_month": t.groupby("month")["r"].agg(["count", "sum"]).round(2).to_dict("index"),
        "by_session": t.groupby("session")["r"].agg(["count", "sum"]).round(2).to_dict("index"),
        "by_dir": t.groupby("dir")["r"].agg(["count", "sum"]).round(2).to_dict("index"),
        "by_reason": t.groupby("reason")["r"].agg(["count", "sum"]).round(2).to_dict("index"),
        "total_r_without_top5": float(r_sorted[:-5].sum()) if len(r_sorted) > 5 else 0.0,
    }


def sensitivity(name: str, results: dict, p: dict, start, end) -> pd.DataFrame:
    g = GRIDS[name]
    rows = {}
    for sl in g["sl_atr"]:
        rows[sl] = {tp: window(results[key({**p, "sl_atr": sl, "tp_atr": tp})], start, end)["r"].sum()
                    for tp in g["tp_atr"]}
    return pd.DataFrame(rows).T.rename_axis("sl_atr \\ tp_atr")


def evaluate(name: str, bars: pd.DataFrame, flat: np.ndarray, holdout_start) -> dict:
    start, end = bars.index[0], bars.index[-1] + pd.Timedelta(days=1)
    print(f"[{name}] backtesting {len(combos(GRIDS[name]))} parameter sets ...")
    results = run_all(name, bars, flat)

    wf_trades, wf_picks = walk_forward(name, results, start, end)
    wf_hold = window(wf_trades, holdout_start, end)

    params, _ = pick_robust(name, results, start, holdout_start)
    train = window(results[key(params)], start, holdout_start)
    hold = window(results[key(params)], holdout_start, end)

    fn = STRATEGIES[name]
    stress = {}
    for label, c in {"spread x1.5": Costs(spread_mult=1.5), "spread x2": Costs(spread_mult=2.0),
                     "slippage $0.10": Costs(slippage=0.10)}.items():
        stress[label] = stats(window(run(bars, fn(bars, **params), flat, c), holdout_start, end))

    sens = sensitivity(name, results, params, holdout_start, end)
    hb = breakdown(hold)
    mc = monte_carlo(hold["r"].to_numpy())
    s_hold, s_wf = stats(hold), stats(wf_hold)
    months_pos = sum(v["sum"] > 0 for v in hb.get("by_month", {}).values())

    checks = {
        f"holdout trades >= {CRITERIA['min_trades']}": s_hold["trades"] >= CRITERIA["min_trades"],
        f"holdout PF >= {CRITERIA['min_pf_holdout']}": s_hold["pf"] >= CRITERIA["min_pf_holdout"],
        f"walk-forward PF (last 6m) >= {CRITERIA['min_pf_walkforward']}":
            s_wf["pf"] >= CRITERIA["min_pf_walkforward"],
        f"profitable with spread x{CRITERIA['stress_spread']}": stress["spread x1.5"]["total_r"] > 0,
        f"bootstrap P(loss) < {CRITERIA['max_p_loss']:.0%}": mc["p_loss"] < CRITERIA["max_p_loss"],
        f">= {CRITERIA['min_positive_months']} positive months": months_pos >= CRITERIA["min_positive_months"],
        "profitable without best 5 trades": hb.get("total_r_without_top5", 0) > 0,
        f">= {CRITERIA['min_positive_neighbours']:.0%} of SL/TP grid profitable":
            (sens.to_numpy() > 0).mean() >= CRITERIA["min_positive_neighbours"],
    }
    return {
        "name": name, "params": params, "train": stats(train), "holdout": s_hold,
        "walk_forward": s_wf, "walk_forward_all": stats(wf_trades), "wf_picks": wf_picks,
        "stress": stress, "monte_carlo": mc, "breakdown": hb, "sensitivity": sens,
        "checks": {c: bool(ok) for c, ok in checks.items()}, "passed": bool(all(checks.values())),
        "_hold_trades": hold, "_wf_trades": wf_hold,
    }


# ---------------------------------------------------------------- report

def _png(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def _equity_png(res: dict) -> str:
    fig, ax = plt.subplots(figsize=(8, 3))
    for label, t in (("holdout (fixed params)", res["_hold_trades"]),
                     ("walk-forward OOS", res["_wf_trades"])):
        if not t.empty:
            ax.plot(t["exit_time"], t["r"].cumsum(), label=label)
    ax.axhline(0, color="grey", lw=0.6)
    ax.set_ylabel("cumulative R")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.3)
    return _png(fig)


def _fmt(s: dict) -> str:
    return (f"{s['trades']} trades · winrate {s['winrate']:.0%} · PF {s['pf']:.2f} · "
            f"exp {s['exp_r']:+.2f}R · total {s['total_r']:+.1f}R · maxDD {s['max_dd_r']:.1f}R · "
            f"{s['pnl']:+.0f} $/oz · {s['ret_pct']:+.1f}% @0.5% risk")


def write_report(results: list[dict], bars: pd.DataFrame, holdout_start, path: Path, setup: str) -> None:
    parts = [f"""<!doctype html><html lang="nl"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>XAUUSD strategy research</title>
<style>body{{font:14px/1.5 system-ui,sans-serif;max-width:960px;margin:24px auto;padding:0 16px;color:#222}}
table{{border-collapse:collapse;margin:8px 0}}td,th{{border:1px solid #ddd;padding:3px 8px;text-align:right}}
.pass{{color:#0a7a2f}}.fail{{color:#b3261e}}img{{max-width:100%}}section{{border-top:2px solid #eee;margin-top:24px}}</style>
</head><body><h1>XAUUSD scalping: strategy research</h1>
<p>Data: Dukascopy XAUUSD, {bars.index[0]:%Y-%m-%d} – {bars.index[-1]:%Y-%m-%d}, {setup} bid/ask bars
({len(bars):,} bars). Holdout (last 6 months, never used to pick parameters): from {holdout_start:%Y-%m-%d}.
Fills at next bar open at bid/ask; real spread included; SL first if SL and TP hit in one bar.</p>
<h2>Summary</h2><table><tr><th>strategy</th><th>holdout</th><th>walk-forward (last 6m)</th><th>verdict</th></tr>"""]
    for r in results:
        v = '<b class="pass">PASS</b>' if r["passed"] else '<b class="fail">FAIL</b>'
        parts.append(f"<tr><td style='text-align:left'>{r['name']}</td><td>{_fmt(r['holdout'])}</td>"
                     f"<td>{_fmt(r['walk_forward'])}</td><td>{v}</td></tr>")
    parts.append("</table>")
    for r in results:
        checks = "".join(f"<li class='{'pass' if ok else 'fail'}'>{'✔' if ok else '✘'} {c}</li>"
                         for c, ok in r["checks"].items())
        stress = "".join(f"<tr><td style='text-align:left'>{k}</td><td>{_fmt(s)}</td></tr>"
                         for k, s in r["stress"].items())
        bm = pd.DataFrame(r["breakdown"].get("by_month", {})).T.to_html() if r["breakdown"] else ""
        bd = pd.DataFrame({**r["breakdown"].get("by_session", {}),
                           **r["breakdown"].get("by_dir", {})}).T.to_html() if r["breakdown"] else ""
        mc = r["monte_carlo"]
        parts.append(f"""<section><h2>{r['name']}</h2>
<p><b>Parameters</b> (chosen on data before the holdout): <code>{json.dumps(r['params'])}</code></p>
<p>Train: {_fmt(r['train'])}<br>Holdout: {_fmt(r['holdout'])}<br>
Walk-forward OOS (all months): {_fmt(r['walk_forward_all'])}</p>
<ul>{checks}</ul>
<img src="data:image/png;base64,{_equity_png(r)}">
<h3>Cost stress (holdout)</h3><table>{stress}</table>
<h3>Monte Carlo (holdout trades)</h3><p>Bootstrap P(loss) {mc['p_loss']:.1%} ·
max drawdown median {mc['dd_median']:.1f}R, 95th percentile {mc['dd_p95']:.1f}R ·
total without best 5 trades {r['breakdown'].get('total_r_without_top5', 0):+.1f}R</p>
<h3>SL × TP sensitivity (holdout total R)</h3>{r['sensitivity'].round(1).to_html()}
<h3>Per month (holdout)</h3>{bm}<h3>Session / direction (holdout)</h3>{bd}
<h3>Walk-forward parameter picks</h3>{pd.DataFrame(r['wf_picks']).to_html(index=False)}
</section>""")
    parts.append("</body></html>")
    path.write_text("\n".join(parts))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout-days", type=int, default=182)
    ap.add_argument("--report", default=str(Path(__file__).parent / "report.html"))
    ap.add_argument("--strategies", nargs="*", default=list(STRATEGIES))
    ap.add_argument("--rule", default="5min", help="bar size: 5min, 15min, 1h")
    ap.add_argument("--start", default="2025-10-27", help="first day of the research period")
    ap.add_argument("--end", default=None, help="last day of the research period")
    ap.add_argument("--entry-hours", default="7-19", help="UTC hours for new entries, e.g. 13-16")
    a = ap.parse_args()

    strategies.ENTRY_HOURS = tuple(int(h) for h in a.entry_hours.split("-"))
    bars = data.load_bars(a.start, a.end, rule=a.rule)
    flat = flat_mask(bars, bar_minutes=int(pd.Timedelta(a.rule).total_seconds() // 60))
    holdout_start = (bars.index[-1] - pd.Timedelta(days=a.holdout_days)).normalize()
    results = [evaluate(n, bars, flat, holdout_start) for n in a.strategies]
    write_report(results, bars, holdout_start, Path(a.report), f"{a.rule}, entries {a.entry_hours} UTC,")

    summary = {r["name"]: {"passed": r["passed"], "params": r["params"], "holdout": r["holdout"],
                           "walk_forward": r["walk_forward"], "checks": r["checks"],
                           "monte_carlo": r["monte_carlo"]} for r in results}
    Path(a.report).with_suffix(".json").write_text(json.dumps(summary, indent=2, default=str))
    for r in results:
        print(f"{r['name']:<24} {'PASS' if r['passed'] else 'FAIL'}  holdout: {_fmt(r['holdout'])}")
    print(f"Report: {a.report}")


if __name__ == "__main__":
    main()
