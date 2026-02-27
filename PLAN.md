# MLB Splits — Current Plan (Pre-Publish)

## Scope (Current MVP)
This repo currently ships a **batter-only** Streamlit app for Statcast-based split analysis.  
Pitcher support is intentionally gated off in UI (`FEATURE_PITCHERS = False`) and deferred to the next release.

Implemented user-facing capabilities:
- Season player lookup and selection
- Dynamic filter builder (inning, pitcher hand, home/away, month, count)
- Active filter summary + filtered sample size display
- Extensible stat system via `StatSpec` registry
- User-controlled stat visibility (sidebar checkbox grid)
- Comparison mode (Player A vs Player B) with delta tiles
- Percentile ranking chart + split tables
- Rule-based NL query parser that can set season/players/filters/stats

## Architecture At A Glance
### Data flow
1. Sidebar + NL query form update `st.session_state` (`season`, players, filters, selected stats, comparison mode).
2. `data/fetcher.py` loads season batting leaderboard + batter Statcast events via `pybaseball` (cached with `st.cache_data`).
3. `stats/filters.py` prepares raw Statcast once per `(player_id, season, player_type)` and applies parsed filters.
4. `stats/splits.py` computes stats and split tables; sample sizes are derived from filtered events.
5. `stats/percentiles.py` builds league distributions from season data and computes percentiles/color tiers.
6. `ui/components.py` renders cards, percentile chart, and split tables; `ui/glossary.py` renders glossary content.

### Key modules
- `app.py`: Streamlit entrypoint and orchestration
- `data/fetcher.py`: data retrieval + ID resolution
- `stats/filters.py`: filter registry, row translation, dataframe prep/cache, filter application
- `stats/splits.py`: stat registry (`StatSpec`), stat math, split generation
- `stats/percentiles.py`: percentile engine + color tiers
- `stats/nl_query.py`: rules-based NL parsing into app intent
- `ui/components.py`: reusable UI renderers
- `tests/`: unit tests for fetch, filters, splits, percentiles, components, NL parser

## Non-Goals (Current MVP)
- Pitcher workflow in UI
- Persistent user accounts or saved queries
- Database-backed storage
- Real-time/pitch-by-pitch live updates

## Publish Readiness Notes
- Entrypoint: `app.py`
- Dependencies: `requirements.txt`
- Test coverage includes parser/filter/split/stat components and integration-adjacent logic
- App behavior relies on external upstream data sources via `pybaseball`; empty/no-data paths are handled in UI
