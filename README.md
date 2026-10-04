# Fantasy Trade Analyzer

Regression models that find buy-low and sell-high trade targets for fantasy football, built on 2026 NFL season data from [nflverse](https://github.com/nflverse/nflverse-data).

## The models

All models are ordinary least squares on season-to-date aggregates (players with 3+ games, PPR scoring). Each answers a different question:

1. **Opportunity model** — *Is he scoring like someone with his workload should?*
   Position-specific: WR/TE expected PPR from targets + receiving air yards; RB expected PPR from targets + carries. The gap between actual and expected PPR is the buy-low / sell-high signal.
2. **Red-zone expected TDs** — *Are his scoring chances turning into touchdowns?*
   Expected TDs from red-zone targets, red-zone carries, and goal-line touches (inside the 10), computed from play-by-play data. A negative gap (few actual TDs vs many chances) means TD regression is coming — TDs are the streakiest stat in football.
3. **Air-yards conversion** — *Is he catching what he's thrown?*
   Expected receiving yards from air yards (how deep his targets are). Underperformers may be victims of bad QB play or drops rather than bad opportunity.
4. **Trend** — *Is he heating up or cooling off?*
   Last-two-week average PPR minus earlier-week average. Catches momentum the season totals hide.

The strongest signal is stacking them: a player who's negative on opportunity *and* TD regression *and* trending up is the best buy-low there is.

**Injuries:** players listed as Out / Doubtful / IR on the latest nflverse injury report are excluded from all recommendations; Questionable players are flagged.

## Run it

```bash
pip install -r requirements.txt
python fantasy_model.py
```

Data downloads automatically from nflverse on first run (~10 MB) into `data/`. Re-run any week for fresh numbers.

## Sample output (2026 Weeks 1–4)

**Buy-low gaps (actual vs expected PPR):** KC Concepcion −20.5, Quentin Johnston −17.3, DK Metcalf −16.7, Quinshon Judkins −14.4

**Due for TDs (actual vs expected):** Emanuel Wilson 0 vs 2.4, Trey McBride 2 vs 3.6, J.K. Dobbins / Tony Pollard / Travis Etienne / Jaylen Warren all 0 TDs on real red-zone usage

**Sell-high (scoring above opportunity):** Jaxon Smith-Njigba +37.9, CeeDee Lamb +25.7, Jahmyr Gibbs +23.7, Derrick Henry +23.4

## Notes

- Assumes PPR scoring; adjust the scoring formula in `fantasy_model.py` for half-PPR or standard.
- Four weeks is a small sample — treat early-season gaps as leads, not verdicts. Check matchups and roles before trading.
- Companion dashboard (private web app) refreshes these numbers weekly.
