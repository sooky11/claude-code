"""Simple interface for the gold swing bot.

    streamlit run app.py
"""

from __future__ import annotations

from dataclasses import asdict

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import bot as botmod
from bot import Params, SwingBot
from engine import Costs, run, stats
from oanda import Oanda
from strategies import mid, tsmom

st.set_page_config(page_title="Goud swing bot", page_icon="🪙", layout="wide")

WARNING = ("**Let op:** deze strategie is niet bewezen winstgevend. Uit het onderzoek (16 jaar data) blijkt dat "
           "de winst vooral komt uit het feit dat goud steeg; willekeurig long instappen deed het ongeveer even "
           "goed. Het voordeel: in slechte jaren verloor hij minder dan gewoon vasthouden. "
           "Test eerst op een demo-account, gebruik geen geld dat je niet kunt missen.")


@st.cache_data(ttl=3600, show_spinner="Koersdata laden ...")
def load_history(source: str) -> tuple[pd.DataFrame, str]:
    if source == "OANDA":
        return Oanda().candles("D", 5000), "OANDA"
    return botmod.daily_from_dukascopy(days=7000), "Dukascopy"


# ------------------------------------------------------------------ sidebar
st.sidebar.title("🪙 Goud swing bot")
st.sidebar.caption("XAU/USD · dagelijks · alleen long")
o = Oanda()
source = st.sidebar.radio("Databron", ["OANDA", "Dukascopy"] if o.configured else ["Dukascopy"],
                          help="Zonder OANDA_API_KEY wordt gratis Dukascopy-data gebruikt.")
st.sidebar.subheader("Instellingen")
lookback = st.sidebar.slider("Trend: vergelijk met slot van ... dagen terug", 5, 100, 20)
hold = st.sidebar.slider("Vasthouden (dagen)", 1, 60, 20)
sl_atr = st.sidebar.slider("Stop (× ATR)", 1.0, 5.0, 2.0, 0.5)
risk_pct = st.sidebar.slider("Risico per trade (% van account)", 0.25, 3.0, 1.0, 0.25)
start_equity = st.sidebar.number_input("Startkapitaal ($)", 1000, 1_000_000, 10_000, 1000)
params = Params(lookback=lookback, hold=hold, sl_atr=sl_atr, risk_pct=risk_pct, start_equity=float(start_equity))
if (lookback, hold, sl_atr) != (20, 20, 2.0):
    st.sidebar.warning("Afwijkend van de geteste standaard (20 / 20 / 2.0).")

try:
    daily, src = load_history(source)
except Exception as e:  # network or credentials
    st.error(f"Kon geen koersdata laden ({source}): {e}")
    st.stop()

tab_over, tab_bt, tab_live = st.tabs(["📊 Overzicht", "🧪 Backtest", "🔴 Live demo"])

# ------------------------------------------------------------------ overview
with tab_over:
    st.warning(WARNING)
    b = SwingBot(params)
    sig = b.signal(daily)
    pos = b.state.get("position")
    last = daily.index[-1]
    price = float(mid(daily, "close").iloc[-1])
    c1, c2, c3, c4 = st.columns(4)
    if pos:
        status = "IN POSITIE"
    elif b.state.get("pending"):
        status = "KOOPT BIJ OPENING"
    else:
        status = "KOPEN" if sig["buy"] else "WACHTEN"
    c1.metric("Signaal", status)
    c2.metric("Goudprijs (laatste slot)", f"${price:,.2f}", help=f"Dag {last:%Y-%m-%d}, bron {src}")
    c3.metric("Account (papier)", f"${b.state['equity']:,.2f}",
              f"{(b.state['equity'] / params.start_equity - 1) * 100:+.1f}%")
    c4.metric("Laatst verwerkte dag", b.state["last_day"][:10] if b.state["last_day"] else "nog niet")
    st.write(f"**Waarom:** {sig['reason']}")
    if pos:
        unreal = (float(daily['bid_close'].iloc[-1]) - pos["entry"]) * pos["units"]
        st.info(f"Open positie: {pos['units']} oz gekocht @ ${pos['entry']:,.2f} op {pos['entry_time'][:10]} · "
                f"stop ${pos['stop']:,.2f} · dag {pos['days_held']}/{params.hold} · "
                f"ongerealiseerd ${unreal:+,.2f}")
    fig = go.Figure(go.Scatter(x=daily.index[-250:], y=mid(daily, "close").iloc[-250:], name="Goud",
                               line=dict(color="#c9a227")))
    ref = mid(daily, "close").shift(lookback).iloc[-250:]
    fig.add_trace(go.Scatter(x=ref.index, y=ref, name=f"Slot {lookback} dagen terug",
                             line=dict(color="#888", dash="dot")))
    if pos:
        fig.add_hline(y=pos["stop"], line=dict(color="#c0392b", dash="dash"), annotation_text="stop")
        fig.add_hline(y=pos["entry"], line=dict(color="#2e86c1", dash="dash"), annotation_text="instap")
    fig.update_layout(height=380, margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h"),
                      title="Goudprijs afgelopen jaar · kopen als de gouden lijn boven de grijze ligt")
    st.plotly_chart(fig, use_container_width=True)
    st.subheader("Logboek")
    st.code("\n".join(reversed(b.state["log"][-15:])) or "Nog geen acties. Ga naar 'Live demo' en start de bot.")

# ------------------------------------------------------------------ backtest
with tab_bt:
    st.caption("Zelfde regels als de bot, met echte bid/ask-spread. Instap op de opening na het signaal; "
               "raakt de koers de stop, dan wordt verkocht op de stop (of lager bij een gat).")
    years = sorted(set(daily.index.year))
    y0, y1 = st.select_slider("Periode", options=years, value=(years[0], years[-1]))
    d = daily[(daily.index.year >= y0) & (daily.index.year <= y1)]
    spread_x = st.radio("Spread", [1.0, 1.5, 2.0], horizontal=True,
                        format_func=lambda x: "normaal" if x == 1 else f"×{x} (stresstest)")
    sig_bt = tsmom(d, lookback=lookback, hold=hold, mode="long", sl_atr=sl_atr)
    t = run(d, sig_bt, np.zeros(len(d), bool), Costs(spread_mult=spread_x))
    s = stats(t, risk_pct=risk_pct / 100)
    eq = pd.Series(start_equity * np.cumprod(1 + risk_pct / 100 * t["r"].to_numpy()), index=t["exit_time"])
    m = mid(d, "close")
    bh = start_equity * m / m.iloc[0]
    k = st.columns(5)
    k[0].metric("Eindkapitaal", f"${(eq.iloc[-1] if len(eq) else start_equity):,.0f}", f"{s['ret_pct']:+.1f}%")
    k[1].metric("Gewoon goud vasthouden", f"${bh.iloc[-1]:,.0f}", f"{(bh.iloc[-1] / start_equity - 1) * 100:+.1f}%")
    k[2].metric("Trades", s["trades"], f"{s['winrate']:.0%} winst")
    k[3].metric("Profit factor", f"{s['pf']:.2f}", help="Winst / verlies. Boven 1 = winstgevend.")
    k[4].metric("Grootste daling", f"{s['max_dd_r'] * risk_pct:.1f}%", help="Max. drawdown (benadering).")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=eq.index, y=eq, name="Bot", line=dict(color="#2e86c1")))
    fig.add_trace(go.Scatter(x=bh.index, y=bh, name="Goud vasthouden (100% belegd)",
                             line=dict(color="#c9a227", dash="dot")))
    fig.update_layout(height=380, margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h"),
                      yaxis_title="$", title="Kapitaal")
    st.plotly_chart(fig, use_container_width=True)
    st.caption(f"Let op: de bot zit maar een deel van de tijd in de markt en riskeert {risk_pct}% per trade; "
               "goud vasthouden is 100% belegd en dus veel riskanter.")
    if len(t):
        c1, c2 = st.columns([1, 2])
        per_year = t.groupby(t["entry_time"].dt.year.rename("jaar")).agg(
            trades=("r", "size"), R=("r", "sum"), **{"$/oz": ("pnl", "sum")}).round(1)
        c1.write("**Per jaar** (R = winst in eenheden risico)")
        c1.dataframe(per_year, use_container_width=True)
        c2.write("**Trades**")
        tt = t[["entry_time", "exit_time", "entry", "exit", "pnl", "r", "reason"]].copy()
        tt.columns = ["instap", "uitstap", "instapprijs", "uitstapprijs", "$/oz", "R", "reden"]
        tt["instap"], tt["uitstap"] = tt["instap"].dt.date, tt["uitstap"].dt.date
        c2.dataframe(tt.iloc[::-1].round(2), use_container_width=True, height=320)
    with st.expander("Is dit beter dan toeval?"):
        st.write("Vergelijk met 100 keer willekeurig long instappen (zelfde vasthoudtijd en stop).")
        if st.button("Vergelijk met toeval"):
            rng = np.random.default_rng(0)
            res = []
            for _ in range(100):
                sg = tsmom(d, lookback=lookback, hold=hold, mode="long", sl_atr=sl_atr)
                sg.side = np.where(rng.random(len(d)) < 0.5, 1, 0)
                sg.side[: lookback + 20] = 0
                res.append(run(d, sg, np.zeros(len(d), bool))["pnl"].sum())
            share = float(np.mean(np.array(res) >= t["pnl"].sum()))
            st.write(f"Bot: **{t['pnl'].sum():+,.0f} $/oz** · toeval mediaan: **{np.median(res):+,.0f} $/oz** · "
                     f"toeval was {share:.0%} van de keren even goed of beter.")
            st.write("✅ Duidelijk beter dan toeval." if share < 0.05 else
                     "⚠️ Niet duidelijk beter dan toeval: de winst komt vooral uit de richting van goud.")

# ------------------------------------------------------------------ live
with tab_live:
    st.warning(WARNING)
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Verbinding")
        if o.configured:
            try:
                acc = o.account_summary()
                st.success(f"OANDA ({o.host.split('//')[1].split('.')[0]}) verbonden · saldo "
                           f"{float(acc['balance']):,.2f} {acc['currency']}")
            except Exception as e:
                st.error(f"OANDA-sleutel gevonden, maar verbinding mislukt: {e}")
        else:
            st.info("Geen OANDA-sleutel: de bot handelt op papier met gratis Dukascopy-data. Zet "
                    "`OANDA_API_KEY` en `OANDA_ACCOUNT_ID` in `gold-bot/.env` voor live demo-koersen en -orders.")
        mode = st.radio("Modus", ["paper", "demo-orders"], horizontal=True,
                        format_func=lambda m: "Papier (alleen bijhouden)" if m == "paper"
                        else "Echte orders op OANDA-demo",
                        disabled=not o.configured)
    with c2:
        st.subheader("Bot draaien")
        b = SwingBot(params, mode=mode)
        st.write(f"Laatst verwerkte dag: **{(b.state['last_day'] or 'nog niet')[:10]}** · "
                 f"nieuwste beschikbare dag: **{daily.index[-1]:%Y-%m-%d}**")
        if st.button("▶️ Voer stap nu uit", type="primary"):
            if mode == "demo-orders" and not o.configured:
                st.error("Geen OANDA-sleutel.")
            else:
                load_history.clear()
                fresh, _ = load_history(source)
                new = b.step(fresh, o if mode == "demo-orders" else None)
                st.success("\n".join(new) if new else "Geen nieuwe dag om te verwerken.")
        auto = st.toggle("Automatisch elk uur controleren (zolang deze pagina open is)")
        st.caption("Voor echt doorlopend draaien: `python bot.py --loop` in een terminal.")

    if auto:
        @st.fragment(run_every="1h")
        def _auto():
            load_history.clear()
            fresh, _ = load_history(source)
            new = SwingBot(params, mode=mode).step(fresh, o if mode == "demo-orders" else None)
            st.caption(f"Automatische controle {pd.Timestamp.now(tz='UTC'):%H:%M} UTC: "
                       + ("; ".join(new) if new else "niets nieuws"))
        _auto()

    st.subheader("Gesloten trades")
    tr = SwingBot(params).trades()
    if tr.empty:
        st.write("Nog geen gesloten trades.")
    else:
        st.dataframe(tr.iloc[::-1], use_container_width=True)
        start = tr["entry_time"].min()
        live_r = tr["r"].sum()
        sim = run(daily, tsmom(daily, lookback=lookback, hold=hold, mode="long", sl_atr=sl_atr),
                  np.zeros(len(daily), bool))
        sim = sim[sim["entry_time"] >= start]
        st.write(f"**Live vs backtest over dezelfde periode** (vanaf {start:%Y-%m-%d}): "
                 f"live {live_r:+.2f}R · backtest {sim['r'].sum():+.2f}R")
    with st.expander("Instellingen van de bot"):
        st.json(asdict(params))
