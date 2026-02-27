# MLB Splits — TODO

Legend: `[ ]` pending, `[x]` done

## Phase 0 — Batter MVP (Completed)
- [x] Streamlit app scaffold + cached pybaseball data layer
- [x] Core stat computation and split tables (hand, home/away, monthly)
- [x] Percentile engine + color tiers + glossary
- [x] Dynamic filter builder (`filter_rows`) with all current filter types
- [x] Active filter summary and sample-size display (`Pitches`, `Balls in play`, `Approx. PA`)
- [x] Extensible stat registry (`StatSpec`) and stat-selection UI
- [x] Comparison mode (A vs B) with side-by-side tiles and delta tiles
- [x] Rule-based NL query parsing + parsed intent preview
- [x] Batter-only UI gating (`FEATURE_PITCHERS = False`)
- [x] Unit tests for fetch, filters, splits, percentiles, UI helpers, and NL parser

## Phase 1 — Next Release (Must-Haves: Pitcher Mode)
- [ ] Add pitcher leaderboard + Statcast fetch path to `data/fetcher.py` with parity caching behavior.
- [ ] Define pitcher stat requirements (v1 set) and add pitcher `StatSpec` compute functions in `stats/splits.py`.
- [ ] Implement pitcher split definitions and labels (mirror batter UX where valid).
- [ ] Re-enable player type selector behind feature flag; route all downstream logic by selected type.
- [ ] Add pitcher-aware validations and empty-state messaging in `app.py`.
- [ ] Expand tests to cover pitcher fetch/filters/stats/splits and mixed mode regression.

## Phase 2 — Must-Haves After Pitchers
- [ ] Add explicit NL disambiguation UI when name fragments match multiple players (user chooses exact player).
- [ ] Add integration-style tests for NL query -> session state application in `app.py`.
- [ ] Add export action for currently visible split table (`CSV`) with selected stat columns.
- [ ] Add deployment metadata checks (optional `runtime.txt`, pinned Streamlit settings) and document release checklist.

## Nice-to-Haves
- [ ] Add saved query presets (session-local first, persistent storage later).
- [ ] Add richer query phrases in parser (team/opponent, custom month ranges).
- [ ] Add optional chart interaction toggles (hover on/off, compact labels) in sidebar.
- [ ] Add visual screenshot assets and a docs page for common workflows.
