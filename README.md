# Overview #
A practical, Sleeper-integrated toolkit that builds your own season projections, ranks players by VORP with risk bands, and runs a live draft assistant GUI that reacts to every pick in real time (mock drafts or your actual league). It supports mock draft support, roster-aware recommendations, and configurable strategy knobs.

# Features #

## Sleeper integration ##
- Pull league settings and scoring
- Track picks in mock or live drafts
- Map Sleeper player metadata to ADP
<img width="432" height="567" alt="image" src="https://github.com/user-attachments/assets/82b556b2-4614-4a9e-b3a0-4494a7867816" />

## Data pipeline ##
- Clean FantasyPros ADP and merge with Sleeper players
- Build custom projections from your league’s history + (optionally) Sleeper projections for the upcoming year

## Projection engine ##
- Empirical per-player distributions from past games
- Optional blending with Sleeper weekly projections (PPR/your scoring)
- Monte Carlo simulation of the season to get player projections (e.g., 1,000 sims × 14 weeks) → mean, p10, p90, std, VORP
<img width="760" height="708" alt="image" src="https://github.com/user-attachments/assets/d3061a67-2966-48ec-9b14-a88a0d2bcbb4" />

## Live Draft Assistant GUI ##
- Real-time suggestions with risk bars (p10–p90)
- Roster-aware ranking, position need weighting, and ADP reach penalty
- Strategy controls (early QB/TE caps, defer K/DEF, need weights, etc.)
- Tooltips with variance (CV%) and simulated finish distribution
<img width="682" height="393" alt="image" src="https://github.com/user-attachments/assets/145e9074-95c3-4817-bc29-de3f96eb4214" />
<img width="682" height="336" alt="image" src="https://github.com/user-attachments/assets/255e08a9-174a-4366-9e7f-4b92ef11d280" />

## Mock Draft CLI ##
- Test the end-to-end flow with Sleeper’s mock draft rooms
<img width="1683" height="96" alt="image" src="https://github.com/user-attachments/assets/3ab63551-65e9-4182-9fdb-0e76a1b0da31" />

# Requirements #
- Python: 3.9-3.12
- Packages:
```bash
pip install pandas numpy requests scipy sv-ttk
```

# Running the Project #
