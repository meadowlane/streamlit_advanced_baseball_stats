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
from stats.filters import FILTER_REGISTRY, apply_filters, rows_to_split_filters
from stats.splits import _compute_stats, get_splits
from ui.components import (
    percentile_bar_chart,
    player_header,
    split_table,
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

SPLIT_TYPE_MAP = {
    "vs. Handedness (L/R)": "hand",
    "Home / Away":          "home_away",
    "By Month":             "monthly",
}

CORE_STATS = ["wOBA", "xwOBA", "K%", "BB%", "HardHit%", "Barrel%"]


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


def _filter_summary(rows: list[dict]) -> str | None:
    """Return a compact comma-joined summary of active filter rows, or None."""
    parts = []
    for row in rows:
        ft = row["filter_type"]
        p  = row["params"]
        if ft == "inning":
            parts.append(f"Inning {p.get('min', 1)}–{p.get('max', 9)}")
        elif ft == "pitcher_hand":
            parts.append("vs LHP" if p.get("hand") == "L" else "vs RHP")
        elif ft == "home_away":
            parts.append(p.get("side", "home").capitalize())
        elif ft == "month":
            parts.append(_INT_TO_MONTH.get(p.get("month", 4), "?"))
        elif ft == "count":
            b, s = p.get("balls"), p.get("strikes")
            if b is not None and s is not None:
                parts.append(f"{b}–{s} count")
            elif b is not None:
                parts.append(f"{b}-ball count")
            elif s is not None:
                parts.append(f"{s}-strike count")
    return ", ".join(parts) if parts else None


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("⚾ MLB Splits")
    st.divider()

    player_type = st.radio("Player type", ["Batter", "Pitcher"], horizontal=True)

    season = st.selectbox("Season", options=SEASONS, index=0)

    split_type_label = st.radio("Split", list(SPLIT_TYPE_MAP.keys()))
    split_type = SPLIT_TYPE_MAP[split_type_label]

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

    _summary = _filter_summary(st.session_state["filter_rows"])
    if _summary:
        st.caption(f"Active: {_summary}")

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


# ---------------------------------------------------------------------------
# Fetch Statcast data + apply filters
# (must precede stat cards so Season Stats reflect active filters)
# ---------------------------------------------------------------------------

with st.spinner(f"Loading {season} Statcast data for {selected_name}…"):
    statcast_df = get_statcast_batter(mlbam_id, season)

filtered_df = apply_filters(statcast_df, filters)

# ---------------------------------------------------------------------------
# Build player stats from filtered Statcast data.
# Distributions use the full-season league (correct reference population) so
# the percentile chart shows where this filtered performance ranks overall.
# ---------------------------------------------------------------------------

_raw         = _compute_stats(filtered_df)
player_stats = {stat: _raw.get(stat) for stat in CORE_STATS}

distributions = build_league_distributions(season_df)
percentiles   = get_all_percentiles(player_stats, distributions)
color_tiers   = get_all_color_tiers(percentiles)


# ---------------------------------------------------------------------------
# Player header
# ---------------------------------------------------------------------------

player_header(selected_name, team, season, player_type)

if player_type == "Pitcher":
    st.warning("Pitcher splits are coming in a future update. Showing batter stats.")

st.divider()


# ---------------------------------------------------------------------------
# Season stat cards
# ---------------------------------------------------------------------------

st.subheader("Season Stats")
stat_cards_row(player_stats, percentiles, color_tiers)

st.divider()


# ---------------------------------------------------------------------------
# Percentile bar chart
# ---------------------------------------------------------------------------

st.subheader("Percentile Rankings vs. League")
percentile_bar_chart(percentiles, color_tiers, player_stats)

st.divider()


# ---------------------------------------------------------------------------
# Split table
# ---------------------------------------------------------------------------

st.subheader(f"Splits: {split_type_label}")

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
        split_table(splits_df)

st.divider()


# ---------------------------------------------------------------------------
# Glossary
# ---------------------------------------------------------------------------

render_glossary()
