import streamlit as st

from data.fetcher import (
    get_batting_stats,
    get_mlbam_id,
    get_player_row,
    get_statcast_batter,
)
from stats.percentiles import (
    build_league_distributions,
    get_all_color_tiers,
    get_all_percentiles,
)
from stats.nl_query import extract_last_year, parse_nl_query
from stats.filters import (
    FILTER_REGISTRY,
    apply_filters,
    get_prepared_df_cached,
    rows_to_split_filters,
    summarize_filter_rows,
)
from stats.splits import STAT_REGISTRY, _compute_stats, get_sample_sizes, get_splits
from ui.components import (
    percentile_bar_chart,
    player_header,
    split_table,
    stat_card,
    stat_cards_row,
)
from ui.glossary import render_glossary

st.set_page_config(
    page_title="MLB Splits",
    page_icon="⚾",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SEASONS = [2025, 2024, 2023, 2022]
FEATURE_PITCHERS = False
DEFAULT_PLAYER_TYPE = "Batter"

SPLIT_TYPE_MAP = {
    "vs. Handedness (L/R)": "hand",
    "Home / Away":          "home_away",
    "By Month":             "monthly",
}

CORE_STATS = ["wOBA", "xwOBA", "K%", "BB%", "HardHit%", "Barrel%"]
_PREPARED_DF_CACHE_KEY = "_prepared_df_cache"


# ---------------------------------------------------------------------------
# Filter UI helpers
# ---------------------------------------------------------------------------

# Ordered label list and reverse lookup, derived from FILTER_REGISTRY so the
# sidebar dropdown always stays in sync with the backend registry.
_REGISTRY_LABELS = [spec.label for spec in FILTER_REGISTRY.values()]
_LABEL_TO_KEY    = {spec.label: spec.key for spec in FILTER_REGISTRY.values()}

_MONTH_LABELS = ["Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct"]
_MONTH_TO_INT = {"Apr": 4, "May": 5, "Jun": 6, "Jul": 7, "Aug": 8, "Sep": 9, "Oct": 10}
_INT_TO_MONTH = {v: k for k, v in _MONTH_TO_INT.items()}

_BALLS_OPTS   = ["any", "0", "1", "2", "3"]
_STRIKES_OPTS = ["any", "0", "1", "2"]
_DELTA_DECIMALS = {"wOBA": 3, "xwOBA": 3, "K%": 1, "BB%": 1, "HardHit%": 1, "Barrel%": 1}
_GRID_COLS_PER_ROW = 3
_DELTA_TILE_CSS = """
<style>
.delta-card {{
    border: 1px solid rgba(255, 255, 255, 0.12);
    border-radius: 10px;
    padding: 14px 10px 12px;
    text-align: center;
    background: #1e2029;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.4);
}}
.delta-label {{
    font-size: 11px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    color: rgba(255, 255, 255, 0.55);
    margin-bottom: 6px;
}}
.delta-value {{
    font-size: 26px;
    font-weight: 800;
    line-height: 1.1;
}}
</style>
"""
_DELTA_TILE_HTML = """
<div class="delta-card">
  <div class="delta-label">{label}</div>
  <div class="delta-value" style="color:{value_color};">{value}</div>
</div>
"""


def _make_type_change_cb(row_id: str):
    """Return an on_change callback that resets params when filter type changes."""
    def _cb():
        new_label = st.session_state[f"filter_{row_id}_type"]
        new_key   = _LABEL_TO_KEY[new_label]
        for r in st.session_state["filter_rows"]:
            if r["id"] == row_id:
                if r["filter_type"] != new_key:
                    r["filter_type"] = new_key
                    r["params"]      = FILTER_REGISTRY[new_key].default_params.copy()
                break
    return _cb


def _sample_size_text(sample_sizes: dict[str, int | None]) -> str:
    parts = [f"Pitches: {sample_sizes['N_pitches']:,}"]
    if sample_sizes["N_BIP"] is not None:
        parts.append(f"Balls in play: {sample_sizes['N_BIP']:,}")
    if sample_sizes["approx_PA"] is not None:
        parts.append(f"Approx. PA: {sample_sizes['approx_PA']:,}")
    return " | ".join(parts)


def _delta_text(stat: str, a_val: float | None, b_val: float | None) -> str:
    if a_val is None or b_val is None:
        return "—"
    decimals = _DELTA_DECIMALS.get(stat, 3)
    return f"{(a_val - b_val):+.{decimals}f}"


def _delta_value(a_val: float | None, b_val: float | None) -> float | None:
    if a_val is None or b_val is None:
        return None
    return a_val - b_val


def _chunk_stats(stats_order: list[str], chunk_size: int = _GRID_COLS_PER_ROW) -> list[list[str]]:
    return [stats_order[i:i + chunk_size] for i in range(0, len(stats_order), chunk_size)]


def _render_player_stat_grid(
    stat_values: dict[str, float | None],
    percentiles: dict[str, float],
    color_tiers: dict[str, dict[str, str]],
    stats_order: list[str],
) -> None:
    if not stats_order:
        st.info("No stats selected.")
        return

    for stat_row in _chunk_stats(stats_order):
        row_cols = st.columns(_GRID_COLS_PER_ROW)
        for col_idx in range(_GRID_COLS_PER_ROW):
            with row_cols[col_idx]:
                if col_idx < len(stat_row):
                    stat = stat_row[col_idx]
                    stat_card(
                        label=stat,
                        value=stat_values.get(stat),
                        percentile=percentiles.get(stat),
                        color_tier=color_tiers.get(stat, {"hex": "#95A5A6"}),
                    )
                else:
                    st.empty()


def _render_delta_stat_grid(
    stats_order: list[str],
    player_stats_a: dict[str, float | None],
    player_stats_b: dict[str, float | None],
) -> None:
    if not stats_order:
        st.info("No stats selected.")
        return

    for stat_row in _chunk_stats(stats_order):
        row_cols = st.columns(_GRID_COLS_PER_ROW)
        for col_idx in range(_GRID_COLS_PER_ROW):
            with row_cols[col_idx]:
                if col_idx < len(stat_row):
                    stat = stat_row[col_idx]
                    delta_val = _delta_value(player_stats_a.get(stat), player_stats_b.get(stat))
                    delta_str = _delta_text(stat, player_stats_a.get(stat), player_stats_b.get(stat))
                    value_color = "#95A5A6"
                    if delta_val is not None:
                        if delta_val > 0:
                            value_color = "#2ECC71"
                        elif delta_val < 0:
                            value_color = "#E74C3C"

                    st.markdown(
                        _DELTA_TILE_CSS
                        + _DELTA_TILE_HTML.format(label=stat, value=delta_str, value_color=value_color),
                        unsafe_allow_html=True,
                    )
                else:
                    st.empty()


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("⚾ MLB Splits")
    st.divider()

    if "season_selectbox" not in st.session_state:
        st.session_state["season_selectbox"] = SEASONS[0]

    with st.form("nl_query_form"):
        st.text_input("Type a query…", key="nl_query_input")
        nl_run = st.form_submit_button("Run", use_container_width=True)

    if nl_run:
        raw_query = st.session_state.get("nl_query_input", "").strip()
        if not raw_query:
            st.session_state["_nl_parsed_preview"] = {
                "raw_query": "",
                "cleaned_query": "",
                "player_a": None,
                "player_b": None,
                "comparison_mode": False,
                "season": None,
                "selected_stats": [],
                "filter_rows": [],
                "warnings": ["Query is empty. Enter a player-oriented query and try again."],
            }
            st.rerun()

        parsed_year = extract_last_year(raw_query)
        lookup_season = parsed_year if parsed_year in SEASONS else st.session_state["season_selectbox"]

        with st.spinner("Parsing query…"):
            lookup_df = get_batting_stats(lookup_season)

        lookup_player_names = sorted(
            lookup_df["Name"].dropna().unique().tolist(),
            key=lambda n: n.split()[-1],
        )
        parsed_intent = parse_nl_query(
            raw_query,
            lookup_player_names,
            valid_seasons=set(SEASONS),
            allowed_stats=CORE_STATS,
        )

        st.session_state["_nl_parsed_preview"] = parsed_intent

        parsed_season = parsed_intent.get("season")
        if parsed_season in SEASONS:
            st.session_state["season_selectbox"] = parsed_season

        parsed_player_a = parsed_intent.get("player_a")
        if parsed_player_a is not None:
            st.session_state["player_selectbox"] = parsed_player_a

        parsed_comparison = bool(parsed_intent.get("comparison_mode"))
        parsed_player_b = parsed_intent.get("player_b")
        if parsed_comparison:
            st.session_state["comparison_mode"] = True
            st.session_state["player_b_selectbox"] = parsed_player_b
        else:
            st.session_state["comparison_mode"] = False
            st.session_state["player_b_selectbox"] = None

        parsed_rows = parsed_intent.get("filter_rows", [])
        st.session_state["filter_rows"] = parsed_rows
        st.session_state["_filter_next_id"] = len(parsed_rows)

        parsed_stats = parsed_intent.get("selected_stats", [])
        if parsed_stats:
            for stat in CORE_STATS:
                st.session_state[f"stat_show_{stat}"] = stat in parsed_stats
            st.session_state["selected_stats_requested"] = parsed_stats

        st.rerun()

    if "_nl_parsed_preview" in st.session_state:
        with st.expander("Parsed intent preview", expanded=False):
            preview = st.session_state["_nl_parsed_preview"]
            if preview.get("warnings"):
                st.warning(" | ".join(preview["warnings"]))
            st.json(preview)

    # Batter-only MVP: keep pitcher code paths in place but unreachable in UI.
    player_type = DEFAULT_PLAYER_TYPE
    if FEATURE_PITCHERS:
        player_type = st.radio("Player type", ["Batter", "Pitcher"], horizontal=True)

    if st.session_state["season_selectbox"] not in SEASONS:
        st.session_state["season_selectbox"] = SEASONS[0]
    season = st.selectbox("Season", options=SEASONS, key="season_selectbox")

    split_type_label = st.radio("Split", list(SPLIT_TYPE_MAP.keys()))
    split_type = SPLIT_TYPE_MAP[split_type_label]
    comparison_mode = st.checkbox("Comparison mode", key="comparison_mode")

    if "selected_stats_requested" not in st.session_state:
        st.session_state["selected_stats_requested"] = CORE_STATS.copy()

    with st.expander("Stats to show", expanded=False):
        for stat in CORE_STATS:
            k = f"stat_show_{stat}"
            if k not in st.session_state:
                st.session_state[k] = stat in st.session_state["selected_stats_requested"]

        btn_col1, btn_col2, btn_col3 = st.columns(3)
        with btn_col1:
            if st.button("Reset", key="stats_reset_defaults", use_container_width=True):
                for stat in CORE_STATS:
                    st.session_state[f"stat_show_{stat}"] = True
                st.rerun()
        with btn_col2:
            if st.button("All", key="stats_select_all", use_container_width=True):
                for stat in CORE_STATS:
                    st.session_state[f"stat_show_{stat}"] = True
                st.rerun()
        with btn_col3:
            if st.button("None", key="stats_select_none", use_container_width=True):
                for stat in CORE_STATS:
                    st.session_state[f"stat_show_{stat}"] = False
                st.rerun()

        selected_count = sum(
            1 for stat in CORE_STATS if st.session_state.get(f"stat_show_{stat}", False)
        )
        st.caption(f"Showing {selected_count} stats")

        cols = st.columns(2)
        for i, stat in enumerate(CORE_STATS):
            with cols[i % 2]:
                st.checkbox(stat, key=f"stat_show_{stat}")

    selected_stats_requested = [
        stat for stat in CORE_STATS
        if st.session_state.get(f"stat_show_{stat}", False)
    ]
    st.session_state["selected_stats_requested"] = selected_stats_requested

    st.divider()

    with st.spinner("Loading player list…"):
        season_df = get_batting_stats(season)

    player_names = sorted(season_df["Name"].dropna().unique().tolist(), key=lambda n: n.split()[-1])

    # Preserve the selected player across season changes.
    # If they didn't qualify this season (< 50 PA), inject their name so the
    # widget keeps showing it; a warning is surfaced below and in the main area.
    #
    # Clear logic: the button sets a pending flag; we honour it here, *before*
    # the widget is instantiated, which is the only point Streamlit allows
    # writing to a widget's own session-state key.
    if st.session_state.pop("_clear_player_pending", False):
        st.session_state["player_selectbox"] = None

    _prev_player = st.session_state.get("player_selectbox")
    _player_not_in_season = _prev_player is not None and _prev_player not in player_names
    if _player_not_in_season:
        player_names = [_prev_player] + player_names

    selected_name = st.selectbox(
        "Player",
        options=player_names,
        index=None,
        placeholder="Type to search…",
        key="player_selectbox",
    )

    if _player_not_in_season and selected_name == _prev_player:
        st.warning(f"No qualifying data for {selected_name} in {season}.")

    if selected_name is not None:
        if st.button("Clear player", use_container_width=True):
            st.session_state["_clear_player_pending"] = True
            st.rerun()

    selected_name_b = None
    if comparison_mode:
        if st.session_state.pop("_clear_player_b_pending", False):
            st.session_state["player_b_selectbox"] = None

        player_b_names = [n for n in player_names if n != selected_name]

        _prev_player_b = st.session_state.get("player_b_selectbox")
        _player_b_not_in_season = (
            _prev_player_b is not None and _prev_player_b not in player_b_names
        )
        if _player_b_not_in_season:
            player_b_names = [_prev_player_b] + player_b_names

        selected_name_b = st.selectbox(
            "Player B",
            options=player_b_names,
            index=None,
            placeholder="Type to search…",
            key="player_b_selectbox",
        )

        if _player_b_not_in_season and selected_name_b == _prev_player_b:
            st.warning(f"No qualifying data for {selected_name_b} in {season}.")

        if selected_name_b is not None:
            if st.button("Clear player B", use_container_width=True):
                st.session_state["_clear_player_b_pending"] = True
                st.rerun()

    # --- filter row session state init ---
    if "filter_rows" not in st.session_state:
        st.session_state["filter_rows"] = []
    if "_filter_next_id" not in st.session_state:
        st.session_state["_filter_next_id"] = 0

    with st.expander("Filters"):
        if st.button("+ Add filter", key="filter_add"):
            new_id = f"f{st.session_state['_filter_next_id']}"
            st.session_state["_filter_next_id"] += 1
            st.session_state["filter_rows"].append({
                "id":          new_id,
                "filter_type": "inning",
                "params":      FILTER_REGISTRY["inning"].default_params.copy(),
            })
            st.rerun()

        rows = st.session_state["filter_rows"]

        if not rows:
            st.caption("No filters active.")

        for i, row in enumerate(rows):
            rid = row["id"]
            col_lbl, col_rm = st.columns([5, 1])

            with col_lbl:
                type_key = f"filter_{rid}_type"
                if type_key not in st.session_state:
                    st.session_state[type_key] = FILTER_REGISTRY[row["filter_type"]].label
                st.selectbox(
                    "Filter type",
                    options=_REGISTRY_LABELS,
                    key=type_key,
                    label_visibility="collapsed",
                    on_change=_make_type_change_cb(rid),
                )

            with col_rm:
                if st.button("×", key=f"filter_{rid}_remove",
                             help="Remove this filter"):
                    st.session_state["filter_rows"].pop(i)
                    st.rerun()

            # Render param widgets for the current filter type.
            # Each widget key is namespaced by row id so removing a row never
            # scrambles another row's widget state.
            ft = row["filter_type"]

            if ft == "inning":
                k = f"filter_{rid}_range"
                if k not in st.session_state:
                    st.session_state[k] = (row["params"].get("min", 1),
                                           row["params"].get("max", 9))
                rng = st.slider(
                    "Inning range", 1, 9,
                    key=k, label_visibility="collapsed",
                )
                row["params"]["min"] = rng[0]
                row["params"]["max"] = rng[1]

            elif ft == "pitcher_hand":
                k = f"filter_{rid}_hand"
                if k not in st.session_state:
                    st.session_state[k] = row["params"].get("hand", "R")
                hand = st.radio(
                    "Pitcher hand", ["L", "R"],
                    key=k, horizontal=True, label_visibility="collapsed",
                )
                row["params"]["hand"] = hand

            elif ft == "home_away":
                k = f"filter_{rid}_side"
                if k not in st.session_state:
                    st.session_state[k] = row["params"].get("side", "home")
                side = st.radio(
                    "Home / Away", ["home", "away"],
                    key=k, horizontal=True, label_visibility="collapsed",
                )
                row["params"]["side"] = side

            elif ft == "month":
                k = f"filter_{rid}_month"
                default_label = _INT_TO_MONTH.get(row["params"].get("month", 4), "Apr")
                if k not in st.session_state:
                    st.session_state[k] = default_label
                month_label = st.selectbox(
                    "Month", _MONTH_LABELS,
                    key=k, label_visibility="collapsed",
                )
                row["params"]["month"] = _MONTH_TO_INT[month_label]

            elif ft == "count":
                kb = f"filter_{rid}_balls"
                ks = f"filter_{rid}_strikes"
                if kb not in st.session_state:
                    bv = row["params"].get("balls")
                    st.session_state[kb] = "any" if bv is None else str(bv)
                if ks not in st.session_state:
                    sv = row["params"].get("strikes")
                    st.session_state[ks] = "any" if sv is None else str(sv)
                cb_col, cs_col = st.columns(2)
                with cb_col:
                    bv = st.selectbox("Balls", _BALLS_OPTS, key=kb)
                with cs_col:
                    sv = st.selectbox("Strikes", _STRIKES_OPTS, key=ks)
                row["params"]["balls"]   = None if bv == "any" else int(bv)
                row["params"]["strikes"] = None if sv == "any" else int(sv)

            if i < len(rows) - 1:
                st.divider()

    active_filter_summary = summarize_filter_rows(st.session_state["filter_rows"])
    st.caption(f"Active: {active_filter_summary}")

    filters = rows_to_split_filters(st.session_state["filter_rows"])

    st.divider()
    st.caption("Data via pybaseball / Baseball Savant")


# ---------------------------------------------------------------------------
# Landing state — no player selected
# ---------------------------------------------------------------------------

if selected_name is None:
    st.header("⚾ MLB Splits")
    st.markdown(
        "Search for a batter in the sidebar to view advanced Statcast stats "
        "and splits by pitcher handedness, home/away, or month."
    )
    render_glossary()
    st.stop()

if comparison_mode and selected_name_b is None:
    st.info("Select **Player B** in the sidebar to compare two players.")
    render_glossary()
    st.stop()


# ---------------------------------------------------------------------------
# Resolve player metadata
# ---------------------------------------------------------------------------

player_row = get_player_row(season_df, selected_name)
if player_row is None:
    st.warning(
        f"{selected_name} has no qualifying data in the {season} season "
        "(fewer than 50 plate appearances or season not yet available). "
        "Select a different season or use **Clear player** in the sidebar."
    )
    st.stop()

fg_id = int(player_row["IDfg"])
team  = str(player_row.get("Team", "—"))

with st.spinner("Resolving player ID…"):
    mlbam_id = get_mlbam_id(fg_id, player_name=selected_name)

if mlbam_id is None:
    st.error(f"Could not resolve a Statcast ID for {selected_name}.")
    st.stop()

team_b = None
mlbam_id_b = None
if comparison_mode:
    player_row_b = get_player_row(season_df, selected_name_b)
    if player_row_b is None:
        st.warning(
            f"{selected_name_b} has no qualifying data in the {season} season "
            "(fewer than 50 plate appearances or season not yet available). "
            "Select a different season or use **Clear player B** in the sidebar."
        )
        st.stop()

    fg_id_b = int(player_row_b["IDfg"])
    team_b = str(player_row_b.get("Team", "—"))

    with st.spinner("Resolving Player B ID…"):
        mlbam_id_b = get_mlbam_id(fg_id_b, player_name=selected_name_b)

    if mlbam_id_b is None:
        st.error(f"Could not resolve a Statcast ID for {selected_name_b}.")
        st.stop()


# ---------------------------------------------------------------------------
# Fetch Statcast data + apply filters
# (must precede stat cards so Season Stats reflect active filters)
# ---------------------------------------------------------------------------

with st.spinner(f"Loading {season} Statcast data for {selected_name}…"):
    raw_statcast_df = get_statcast_batter(mlbam_id, season)

prepared_cache = st.session_state.setdefault(_PREPARED_DF_CACHE_KEY, {})
prepare_cache_key = (int(mlbam_id), int(season), str(player_type))
statcast_df = get_prepared_df_cached(
    raw_statcast_df,
    prepared_cache,
    prepare_cache_key,
    log_fn=print,
)

filtered_df = apply_filters(statcast_df, filters)
sample_sizes = get_sample_sizes(filtered_df)

statcast_df_b = None
filtered_df_b = None
sample_sizes_b = None
if comparison_mode and mlbam_id_b is not None:
    with st.spinner(f"Loading {season} Statcast data for {selected_name_b}…"):
        raw_statcast_df_b = get_statcast_batter(mlbam_id_b, season)
    prepare_cache_key_b = (int(mlbam_id_b), int(season), str(player_type))
    statcast_df_b = get_prepared_df_cached(
        raw_statcast_df_b,
        prepared_cache,
        prepare_cache_key_b,
        log_fn=print,
    )
    filtered_df_b = apply_filters(statcast_df_b, filters)
    sample_sizes_b = get_sample_sizes(filtered_df_b)

missing_stat_requirements: dict[str, list[str]] = {}
selected_stats: list[str] = []
validation_frames: list[tuple[str, object]] = [(selected_name, statcast_df)]
if comparison_mode and statcast_df_b is not None:
    validation_frames.append((selected_name_b, statcast_df_b))

for stat in selected_stats_requested:
    spec = STAT_REGISTRY.get(stat)
    if spec is None:
        continue

    missing_notes: list[str] = []
    for player_label, df_for_player in validation_frames:
        missing_cols = [col for col in spec.required_cols if col not in df_for_player.columns]
        if missing_cols:
            missing_notes.append(f"{player_label}: {', '.join(missing_cols)}")

    if missing_notes:
        missing_stat_requirements[stat] = missing_notes
        continue
    selected_stats.append(stat)

if missing_stat_requirements:
    missing_text = ", ".join(
        f"{stat} ({'; '.join(notes)})"
        for stat, notes in missing_stat_requirements.items()
    )
    st.warning(f"Some selected stats were skipped due to missing data columns: {missing_text}.")

# ---------------------------------------------------------------------------
# Build player stats from filtered Statcast data.
# Distributions use the full-season league (correct reference population) so
# the percentile chart shows where this filtered performance ranks overall.
# ---------------------------------------------------------------------------

_raw         = _compute_stats(filtered_df)
player_stats = {stat: _raw.get(stat) for stat in selected_stats}

distributions = build_league_distributions(season_df)
percentiles   = get_all_percentiles(player_stats, distributions)
color_tiers   = get_all_color_tiers(percentiles)

player_stats_b = None
percentiles_b = None
color_tiers_b = None
if comparison_mode and filtered_df_b is not None:
    _raw_b = _compute_stats(filtered_df_b)
    player_stats_b = {stat: _raw_b.get(stat) for stat in selected_stats}
    percentiles_b = get_all_percentiles(player_stats_b, distributions)
    color_tiers_b = get_all_color_tiers(percentiles_b)


# ---------------------------------------------------------------------------
# Player header
# ---------------------------------------------------------------------------

if comparison_mode and team_b is not None:
    st.subheader(f"{selected_name} vs {selected_name_b}")
    st.caption(f"{team} vs {team_b} · {season} · {player_type}")
else:
    player_header(selected_name, team, season, player_type)

if player_type == "Pitcher":
    st.warning("Pitcher splits are coming in a future update. Showing batter stats.")

st.divider()


# ---------------------------------------------------------------------------
# Season stat cards
# ---------------------------------------------------------------------------

if active_filter_summary == "No filters (full season data)":
    st.caption(active_filter_summary)
else:
    st.caption(f"Active filters: {active_filter_summary}")

st.subheader("Season Stats")
if comparison_mode and player_stats_b is not None and percentiles_b is not None and color_tiers_b is not None:
    col_a, col_b, col_delta = st.columns(3)
    with col_a:
        st.markdown(f"**{selected_name}**")
        st.caption(_sample_size_text(sample_sizes))
        _render_player_stat_grid(player_stats, percentiles, color_tiers, selected_stats)
    with col_b:
        st.markdown(f"**{selected_name_b}**")
        st.caption(_sample_size_text(sample_sizes_b))
        _render_player_stat_grid(player_stats_b, percentiles_b, color_tiers_b, selected_stats)
    with col_delta:
        st.markdown(f"**Difference ({selected_name} minus {selected_name_b})**")
        st.caption(
            f"Positive means {selected_name} is higher; negative means {selected_name_b} is higher."
        )
        _render_delta_stat_grid(selected_stats, player_stats, player_stats_b)
else:
    st.caption(f"Sample size: {_sample_size_text(sample_sizes)}")
    stat_cards_row(player_stats, percentiles, color_tiers, stats_order=selected_stats)

st.divider()


# ---------------------------------------------------------------------------
# Percentile bar chart
# ---------------------------------------------------------------------------

st.subheader("Percentile Rankings vs. League")
if comparison_mode and percentiles_b is not None and color_tiers_b is not None and player_stats_b is not None:
    st.caption(f"Player A: {selected_name}")
    percentile_bar_chart(percentiles, color_tiers, player_stats, stats_order=selected_stats)
    st.caption(f"Player B: {selected_name_b}")
    percentile_bar_chart(percentiles_b, color_tiers_b, player_stats_b, stats_order=selected_stats)
else:
    percentile_bar_chart(percentiles, color_tiers, player_stats, stats_order=selected_stats)

st.divider()


# ---------------------------------------------------------------------------
# Split table
# ---------------------------------------------------------------------------

st.subheader(f"Splits: {split_type_label}")

if comparison_mode and statcast_df_b is not None and filtered_df_b is not None:
    left_split_col, right_split_col = st.columns(2)

    with left_split_col:
        st.caption(f"Player A: {selected_name}")
        if statcast_df.empty:
            st.warning(
                f"No Statcast data found for {selected_name} in {season}. "
                "They may not have had enough plate appearances or the season data "
                "may not yet be available."
            )
        else:
            splits_df = get_splits(filtered_df, split_type)
            if splits_df.empty or splits_df["PA"].sum() == 0:
                st.info("No plate appearances found for the selected split.")
            else:
                split_cols = ["Split", "PA"] + [s for s in selected_stats if s in splits_df.columns]
                split_table(splits_df[split_cols])

    with right_split_col:
        st.caption(f"Player B: {selected_name_b}")
        if statcast_df_b.empty:
            st.warning(
                f"No Statcast data found for {selected_name_b} in {season}. "
                "They may not have had enough plate appearances or the season data "
                "may not yet be available."
            )
        else:
            splits_df_b = get_splits(filtered_df_b, split_type)
            if splits_df_b.empty or splits_df_b["PA"].sum() == 0:
                st.info("No plate appearances found for the selected split.")
            else:
                split_cols_b = ["Split", "PA"] + [s for s in selected_stats if s in splits_df_b.columns]
                split_table(splits_df_b[split_cols_b])
else:
    if statcast_df.empty:
        st.warning(
            f"No Statcast data found for {selected_name} in {season}. "
            "They may not have had enough plate appearances or the season data "
            "may not yet be available."
        )
    else:
        splits_df = get_splits(filtered_df, split_type)
        if splits_df.empty or splits_df["PA"].sum() == 0:
            st.info("No plate appearances found for the selected split.")
        else:
            split_cols = ["Split", "PA"] + [s for s in selected_stats if s in splits_df.columns]
            split_table(splits_df[split_cols])

st.divider()


# ---------------------------------------------------------------------------
# Glossary
# ---------------------------------------------------------------------------

render_glossary()
