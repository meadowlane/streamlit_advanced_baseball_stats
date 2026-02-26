# mlb-splits — Project Plan

## Overview
A Streamlit web app that surfaces advanced Statcast-friendly batting and pitching splits for MLB players and teams. The primary focus is on xwOBA/wOBA, K%, BB%, HardHit%, and Barrel%, with percentile context and a built-in glossary.

---

## Architecture

```
mlb-splits/
├── app.py                  # Streamlit entry point
├── data/
│   └── fetcher.py          # pybaseball / Statcast data retrieval
├── stats/
│   └── splits.py           # Split calculations (L/R, home/away, month, etc.)
│   └── percentiles.py      # League-wide percentile ranking logic
├── ui/
│   └── components.py       # Reusable Streamlit widgets (stat cards, bar charts)
│   └── glossary.py         # Glossary definitions + percentile explainer text
├── tests/
│   └── test_splits.py
│   └── test_percentiles.py
├── requirements.txt
├── PLAN.md
└── TODO.md
```

---

## Data Source
- **pybaseball** — wraps Baseball Savant (Statcast) and FanGraphs endpoints.
  - `statcast_batter()` / `statcast_pitcher()` for raw Statcast data.
  - `batting_stats()` / `pitching_stats()` for season-level FanGraphs data (includes xwOBA, K%, BB%, etc.).
- Data is cached in-session via `@st.cache_data` to avoid repeated API calls.

---

## Core Stats (Phase 1)
| Stat | Source | Description |
|------|--------|-------------|
| wOBA | FanGraphs | Weighted on-base average |
| xwOBA | Statcast | Expected wOBA based on exit velo + launch angle |
| K% | FanGraphs | Strikeout rate |
| BB% | FanGraphs | Walk rate |
| HardHit% | Statcast | % of batted balls with exit velocity ≥ 95 mph |
| Barrel% | Statcast | % of batted balls classified as barrels |

---

## Splits (Phase 1)
- vs. LHP / vs. RHP
- Home / Away
- By month (April–September)

## Splits (Phase 2)
- By count (ahead / behind / even)
- vs. specific teams
- RISP vs. bases empty

---

## UI Layout

### Sidebar
- Player search (autocomplete by name)
- Season selector (current year + 3 prior)
- Split type selector (vs. hand, home/away, monthly)
- Batter / Pitcher toggle

### Main Panel
1. **Player header** — name, team, position, headshot (if available)
2. **Stat summary cards** — wOBA, xwOBA, K%, BB%, HardHit%, Barrel% with color-coded percentile badges
3. **Split table** — filterable, sortable DataTable
4. **Percentile bar chart** — horizontal bars vs. league average
5. **Glossary expander** — definitions + "how to read percentiles" section

---

## Percentile Color Scale
- 90–100: red (elite)
- 70–89: orange
- 50–69: yellow
- 30–49: blue
- 0–29: gray (below average)

(Mirrors Baseball Savant's color convention.)

---

## Tech Stack
| Tool | Purpose |
|------|---------|
| Python 3.11+ | Runtime |
| Streamlit | UI framework |
| pybaseball | Data retrieval |
| pandas | Data manipulation |
| plotly | Interactive charts |
| pytest | Testing |

---

## Out of Scope (v1)
- User accounts / saved searches
- Real-time pitch-by-pitch data
- Mobile-optimized layout
- Database persistence
