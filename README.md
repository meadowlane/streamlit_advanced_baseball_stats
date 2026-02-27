# MLB Splits

MLB Splits is a Streamlit app for exploring **advanced batter performance** using Statcast-aligned metrics and split views. It combines season-level context (percentiles vs league) with pitch/event-level filtering so you can answer questions like “how does this hitter perform late in games vs lefties?”

The app is currently a **batter-only MVP** focused on speed, clarity, and practical exploration. It supports side-by-side player comparison, customizable stat views, and a rules-based natural-language query box that can pre-populate players, filters, season, and stats.

## Key Features
- Six core stats: `wOBA`, `xwOBA`, `K%`, `BB%`, `HardHit%`, `Barrel%`
- Split tables: vs handedness, home/away, by month
- Dynamic filters:
  - inning range
  - pitcher handedness
  - home/away
  - month
  - count (`2-1`, `full count`, `3 balls any strikes`, etc.)
- Active filter summary and filtered sample sizes
- Stat selection UI (choose which stats appear in tiles/charts/tables)
- Comparison mode (Player A vs Player B) with per-stat deltas
- NL query parser with intent preview and warnings
- Cached data fetch and preprocessing for responsive reruns

## Screenshots
### Main Dashboard
_Add screenshot here_

### Comparison Mode
_Add screenshot here_

### NL Query Parsing
_Add screenshot here_

## Quickstart (Local)
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Data Sources & Performance Notes
- Data is loaded via [`pybaseball`](https://github.com/jldbc/pybaseball), which wraps Baseball Savant and FanGraphs endpoints.
- External upstream availability/latency can affect load times and completeness.
- The app uses Streamlit caching (`st.cache_data`) plus prepared-data memoization to reduce repeated network calls and dataframe preprocessing.

## How Filters Work
- The sidebar filter builder stores rows in `st.session_state["filter_rows"]`.
- Rows are converted to typed filters and applied before stat/split computation.
- If multiple rows set the same filter type, **last row wins**.
- Filter summary text is shown near Season Stats so active scope is always visible.

## NL Query Examples
Try these in **Type a query…**:
- `Gunnar Henderson inning 7-9 vs LHP`
- `Gunnar Henderson vs Dylan Beavers inning 7+ at home May`
- `compare Gunnar Henderson and Dylan Beavers 2025 full count`
- `Pete Alonso vs Gunnar Henderson 2024`
- `Aaron Judge on the road 2-1 count`
- `Mike Trout barrel% xwOBA 2025`
- `Gunnar Henderson vs righties June`

## Testing
Run all tests:
```bash
pytest -q
```

## Deployment (Streamlit Community Cloud)
1. Push this repo to GitHub.
2. In Streamlit Community Cloud, create a new app from that repo.
3. Set the entrypoint to `app.py`.
4. Ensure dependencies install from `requirements.txt`.
5. (Optional) Add `runtime.txt` only if you want to pin Python version for deployment consistency.

## Roadmap
- Next release: pitcher mode (fetch, stats, splits, and UI mode toggle)
- Follow-up: stronger NL disambiguation UX and export workflows

## License
License: **TBD** (no `LICENSE` file is currently present).
