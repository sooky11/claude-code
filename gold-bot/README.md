# Gold (XAU/USD) scalping bot

Status: **fase A, strategie-onderzoek.** De bot en de live paper-trading
interface (OANDA demo) worden pas gebouwd als een strategie de tests hieronder
doorstaat.

## Installeren

```bash
cd gold-bot
pip install -r requirements.txt
```

## 1. Data downloaden (Dukascopy, gratis)

```bash
python data.py --start 2025-10-01 --end 2026-10-27
```

Hiermee download je XAUUSD tick-data per uur en sla je die op als 1-minuut
bid/ask-candles in `data/`.

## 2. Strategie-onderzoek

```bash
python research.py
```

Dit test 4 simpele strategieën op 5-minuut candles (EMA+RSI pullback,
opening-range breakout, Bollinger mean-reversion, Donchian breakout) en
schrijft `report.html` en `report.json`.

Hoe er getest wordt:

- **Fills:** signaal op de slotkoers, instap op de open van de volgende candle
  tegen bid/ask, met de echte spread. Worden SL en TP in dezelfde candle
  geraakt, dan telt de stop.
- **Holdout:** de laatste 6 maanden worden nooit gebruikt om parameters te
  kiezen. Die worden gekozen op de periode daarvoor, als "plateau" (gemiddelde
  van de buren in het grid), niet als losse piek.
- **Walk-forward:** steeds 3 maanden optimaliseren en de volgende maand testen.
- **Stresstest:** spread ×1.5 en ×2, en $0.10 slippage.
- **Monte Carlo:** met bootstrap de kans op verlies en de spreiding van de
  drawdown.
- **Spreiding:** per maand, per sessie, long/short, en het resultaat zonder de
  5 beste trades.

Een strategie slaagt alleen als ze **alle** criteria in `research.CRITERIA`
haalt. Op willekeurige data (ruis) faalt elke strategie, zoals het hoort.
Eén strategie haalde daar toevallig wel PF 1.19. Daarom is één goed cijfer
niet genoeg.

## Tests

```bash
python -m pytest -q tests
```

> Een backtest is geen garantie voor de toekomst. Test eerst live op een
> demo-account voordat je echt geld inzet.
