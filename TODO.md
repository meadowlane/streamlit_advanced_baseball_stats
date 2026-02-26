# mlb-splits — TODO

Legend: [ ] pending | [x] done | [-] skipped/deferred

---

## Phase 0 — Project Setup
- [x] Create `requirements.txt` (streamlit, pybaseball, pandas, plotly, pytest)
- [x] Create `app.py` skeleton (title, sidebar stub, main panel stub)
- [x] Verify `streamlit run app.py` launches without errors
- [x] Set up `tests/` directory with a placeholder test
- [x] Verify `pytest` runs cleanly

---

## Phase 1 — Data Layer
- [x] `data/fetcher.py`: fetch season batting stats via pybaseball (`batting_stats`)
- [x] `data/fetcher.py`: fetch Statcast batter data (`statcast_batter`)
- [x] Add `@st.cache_data` wrappers to both fetch functions
- [x] Write unit tests for fetcher (mock network calls)
- [x] Confirm the 6 core stats are present in returned DataFrames

---

## Phase 2 — Splits Logic
- [x] `stats/splits.py`: compute vs-LHP / vs-RHP splits from Statcast data
- [x] `stats/splits.py`: compute home / away splits
- [x] `stats/splits.py`: compute monthly splits (group by `game_date` month)
- [x] Write unit tests for each split function with fixture data

---

## Phase 3 — Percentile Engine
- [x] `stats/percentiles.py`: load league-wide distribution for each core stat
- [x] `stats/percentiles.py`: `get_percentile(stat, value, year)` function
- [x] Decide color tier per percentile range (matches Savant scale)
- [x] Write unit tests for percentile edge cases (min/max, NaN handling)

---

## Phase 4 — UI Components
- [x] `ui/components.py`: stat card widget (label, value, percentile badge)
- [x] `ui/components.py`: percentile bar chart (plotly horizontal bars)
- [x] `ui/components.py`: split DataTable with sortable columns
- [x] Player header section (name, team, position)

---

## Phase 5 — Glossary & Explainer
- [x] `ui/glossary.py`: write definitions for all 6 core stats
- [x] `ui/glossary.py`: write "How to read percentiles" explainer (color scale key)
- [x] Wire glossary into a Streamlit `st.expander` in the main panel

---

## Phase 6 — Sidebar & Search
- [x] Sidebar: player name search with autocomplete (selectbox or text_input + filter)
- [x] Sidebar: season selector (current year and 3 prior)
- [x] Sidebar: split type radio (vs. hand / home-away / monthly)
- [x] Sidebar: batter / pitcher toggle

---

## Phase 7 — Integration & Polish
- [x] Wire sidebar inputs → data fetch → splits → percentiles → UI render
- [x] Add loading spinners (`st.spinner`) during data fetch
- [x] Handle empty / no-data states gracefully
- [x] Add page config (title, favicon, layout="wide")
- [x] Manual smoke test: search 3 known players, verify numbers

---

## Phase 8 — Testing & Cleanup
- [x] Full `pytest` run — all tests green
- [x] Remove debug print statements
- [x] Review for hardcoded magic numbers → constants
- [x] Final check: `streamlit run app.py` end-to-end clean

---

## Deferred / Phase 2 Features
- [ ] Count-state splits (ahead/behind/even)
- [ ] vs. specific opponent splits
- [ ] RISP splits
- [ ] Pitcher mode (mirror of batter stats)
- [ ] Export to CSV button
