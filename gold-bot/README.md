# Goud (XAU/USD) swing bot

Een simpele bot voor goud met een interface om hem te backtesten en live
(op papier of op een OANDA-demo) te volgen.

> **Eerlijke samenvatting:** deze strategie is **niet bewezen winstgevend**.
> Lees "Wat het onderzoek liet zien" hieronder voordat je hem gebruikt. Test
> eerst op een demo-account.

## Snel starten

```bash
cd gold-bot
pip install -r requirements.txt
streamlit run app.py            # opent de interface in je browser
```

De eerste keer haalt de app ongeveer 19 jaar koersdata op bij Dukascopy. Dat
duurt een paar minuten. Daarna staat alles in de cache in `data/`.

### Tabs

- **Overzicht:** het huidige signaal (KOPEN / WACHTEN / IN POSITIE), waarom,
  de grafiek, een open positie met de stop, en het logboek.
- **Backtest:** kies een periode en instellingen. Je ziet het kapitaal tegenover
  gewoon goud vasthouden, de resultaten per jaar, alle trades en een vergelijking
  met willekeurig instappen.
- **Live demo:** voer een stap uit of laat de bot elk uur controleren. Je ziet
  de gesloten trades en hoe live zich verhoudt tot de backtest.

### Bot zonder interface

```bash
python bot.py --once     # verwerk de laatste afgesloten dag
python bot.py --loop     # elk uur controleren (laat draaien)
```

De status staat in `state/state.json` en de trades in `state/trades.csv`.

## De regels

Elke afgesloten dag:

1. **Kopen** als de slotkoers hoger is dan die van 20 dagen terug, en je hebt
   nog geen positie. Je koopt op de opening van de volgende dag.
2. **Stop** op de instapprijs min 2 × de gemiddelde dagbeweging (ATR 14).
3. **Verkopen** na 20 dagen, of eerder als de stop geraakt wordt.
4. **Grootte:** zoveel ounce dat het raken van de stop 1% van je account kost.

Er wordt alleen long gehandeld. Short verloor in het onderzoek.

## OANDA-demo koppelen (optioneel)

1. Maak een gratis **demo-account** (practice) aan bij OANDA.
2. Ga naar **Manage API Access** en genereer een sleutel.
3. Maak `gold-bot/.env` aan (dit bestand wordt nooit naar git gestuurd):
   ```
   OANDA_API_KEY=jouw-sleutel
   OANDA_ACCOUNT_ID=101-004-xxxxxxx-001
   ```
4. Herstart de app. Je kunt dan OANDA als databron kiezen en in "Live demo"
   overschakelen naar **Echte orders op OANDA-demo**.

Zonder sleutel handelt de bot op papier met de gratis Dukascopy-data.

## Wat het onderzoek liet zien

Het doel was eerst een winstgevende **scalping**-bot. Daarvoor heb ik
13 strategiefamilies getest, met echte bid/ask-spread, een holdout, een
walk-forward-test, een stresstest op de kosten en een Monte Carlo-analyse:

- **Scalping en daytrading (5 min – 1 uur):** op 1 jaar minuutdata en op 16 jaar
  uurdata (2008–2024) had **geen enkele** strategie een betrouwbare voorsprong.
  De profit factor lag tussen 0,84 en 1,11. De spread (± $0,65) is ongeveer 12%
  van een gemiddelde 5-minuutbeweging.
- **Swing (deze bot):** 2008–2022 PF 1,40; 2022–24 PF 2,36; 2024–25 +$384 per
  ounce; 2025–26 ± $0. **Maar** willekeurig long instappen met dezelfde
  vasthoudtijd en stop deed het ongeveer even goed. De winst komt dus vooral
  doordat goud steeg. Het pluspunt: in slechte jaren (zoals 2013) verloor hij
  veel minder dan gewoon vasthouden.

Conclusie: zie de bot als een gedisciplineerde manier om met beperkt risico
long te zitten in goud, niet als een geldmachine. Daalt goud langdurig, dan
verliest hij ook.

## Bestanden

| Bestand | Wat |
|---|---|
| `app.py` | interface (Streamlit) |
| `bot.py` | de swing bot: signaal, positie, stop, logboek |
| `oanda.py` | koppeling met OANDA (koersen, orders) |
| `data.py` | koersdata van Dukascopy (minuut, uur) of OANDA |
| `engine.py` | backtest-motor (bid/ask, stop gaat voor) |
| `strategies.py` | alle onderzochte strategieën |
| `research.py`, `validate.py` | het strategie-onderzoek en de tests |

## Tests

```bash
python -m pytest -q tests
```

De bot is ook dag voor dag over 2008–2026 nagespeeld. Daarbij leverde hij
dezelfde 258 trades op als de backtest.
