# Overview #
A practical, Sleeper-integrated toolkit that builds your own season projections, ranks players by VORP with risk bands, and runs a live draft assistant GUI that reacts to every pick in real time (mock drafts or your actual league). It supports mock draft support, roster-aware recommendations, and configurable strategy knobs.

# Features #

## Sleeper integration ##
- Pull league settings and scoring
- Track picks in mock or live drafts
- Map Sleeper player metadata to ADP

## Data pipeline ##
- Clean FantasyPros ADP and merge with Sleeper players
- Build custom projections from your league’s history + (optionally) Sleeper projections for the upcoming year

## Projection engine ##
- Empirical per-player distributions from past games
- Optional blending with Sleeper weekly projections (PPR/your scoring)
- Monte Carlo simulation of the season to get player projections (e.g., 1,000 sims × 14 weeks) → mean, p10, p90, std, VORP

## Live Draft Assistant GUI ##
- Real-time suggestions with risk bars (p10–p90)
- Roster-aware ranking, position need weighting, and ADP reach penalty
- Strategy controls (early QB/TE caps, defer K/DEF, need weights, etc.)
- Tooltips with variance (CV%) and simulated finish distribution

## Mock Draft CLI ##
- Test the end-to-end flow with Sleeper’s mock draft rooms

# Requirements #
- Python: 3.9-3.12
- Packages:
```bash
pip install pandas numpy requests scipy sv-ttk
```
