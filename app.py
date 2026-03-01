import datetime as dt

import pandas as pd
import streamlit as st

from data.fetcher import (
    get_batting_stats,
    get_fg_batting_wrc_plus,
    get_pitching_stats,
    get_mlbam_id,
    get_player_row,
    get_statcast_batter,
    get_statcast_pitcher,
)
from stats.percentiles import (
    build_league_distributions,
    build_pitcher_league_distributions,
    get_all_color_tiers,
    get_all_percentiles,
)
from stats.nl_query import extract_last_year, parse_nl_query
from stats.filters import (
    FILTER_REGISTRY,
    SplitFilters,
    apply_filters,
    get_prepared_df_cached,
    rows_to_split_filters,
    summarize_filter_rows,
)
from stats.splits import (
    STAT_REGISTRY,
    _compute_all_pitcher_stats,
    _compute_stats,
    compute_pitch_arsenal,
    get_pitcher_splits,
    get_sample_sizes,
    get_splits,
    get_trend_stats,
)
from ui.components import (
    percentile_bar_chart,
    player_header,
    render_pitch_arsenal,
    render_trend_custom,
    render_trend_dashboard,
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

STATCAST_START_YEAR = 2015
CURRENT_YEAR = dt.date.today().year
STATCAST_SEASONS = list(range(CURRENT_YEAR, STATCAST_START_YEAR - 1, -1))
SEASONS = STATCAST_SEASONS
DEFAULT_SEASON = max(STATCAST_START_YEAR, CURRENT_YEAR - 1)
FEATURE_PITCHERS = True
FEATURE_PITCH_ZONE = True
DEFAULT_PLAYER_TYPE = "Batter"

SPLIT_TYPE_MAP = {
    "vs. Handedness (L/R)": "hand",
    "Home / Away":          "home_away",
    "By Month":             "monthly",
}

CORE_STATS = ["wOBA", "xwOBA", "K%", "BB%", "HardHit%", "Barrel%"]
BATTER_EXTRA_STATS = ["wRC+"]
TRADITIONAL_STATS = ["AVG", "OBP", "SLG", "OPS"]
PITCHER_CORE_STATS = ["wOBA", "xwOBA", "K%", "BB%", "K-BB%", "CSW%"]
PITCHER_SECONDARY_STATS = ["HardHit%", "Barrel%", "GB%", "Whiff%", "FirstStrike%"]
PITCHER_EXTRA_STATS = [stat for stat in (PITCHER_CORE_STATS + PITCHER_SECONDARY_STATS) if stat not in CORE_STATS]
_DEFAULT_STATS = BATTER_EXTRA_STATS + CORE_STATS  # stats shown by default; Reset restores this
_PREPARED_DF_CACHE_KEY = "_prepared_df_cache"
_PITCHER_STAT_LABELS = {"wOBA": "wOBA Allowed", "xwOBA": "xwOBA Allowed"}


# ---------------------------------------------------------------------------
# Filter UI helpers
# ---------------------------------------------------------------------------

# Ordered label list and reverse lookup, derived from FILTER_REGISTRY so the
# sidebar dropdown always stays in sync with the backend registry.
_REGISTRY_LABELS = [spec.label for spec in FILTER_REGISTRY.values()]
_LABEL_TO_KEY    = {spec.label: spec.key for spec in FILTER_REGISTRY.values()}
_LABEL_TO_KEY["Batter hand"] = "batter_hand"

_MONTH_LABELS = ["Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct"]
_MONTH_TO_INT = {"Apr": 4, "May": 5, "Jun": 6, "Jul": 7, "Aug": 8, "Sep": 9, "Oct": 10}
_INT_TO_MONTH = {v: k for k, v in _MONTH_TO_INT.items()}

_BALLS_OPTS   = ["any", "0", "1", "2", "3"]
_STRIKES_OPTS = ["any", "0", "1", "2"]
_DELTA_DECIMALS = {
    "wOBA": 3,
    "xwOBA": 3,
    "K%": 1,
    "BB%": 1,
    "HardHit%": 1,
    "Barrel%": 1,
    "GB%": 1,
    "CSW%": 1,
    "Whiff%": 1,
    "FirstStrike%": 1,
    "K-BB%": 1,
    "wRC+": 0,
    "AVG": 3,
    "OBP": 3,
    "SLG": 3,
    "OPS": 3,
}
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


def _available_filter_keys(player_type: str) -> list[str]:
    excluded = {"batter_hand"} if player_type == "Batter" else {"pitcher_hand"}
    return [k for k in FILTER_REGISTRY if k not in excluded]


def _sidebar_filter_label(filter_key: str, player_type: str) -> str:
    if player_type == "Pitcher" and filter_key == "batter_hand":
        return "Batter hand"
    return FILTER_REGISTRY[filter_key].label


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


def _appearance_label(player_type: str) -> str:
    return "Approx. BF" if player_type == "Pitcher" else "Approx. PA"


def _name_team_key(name: object, team: object) -> str:
    name_str = str(name).strip().lower() if name is not None else ""
    team_str = str(team).strip().upper() if team is not None else ""
    return f"{name_str}|{team_str}"


def _as_optional_float(value: object) -> float | None:
    try:
        parsed = pd.to_numeric(value, errors="coerce")
    except Exception:
        return None
    if pd.isna(parsed):
        return None
    return float(parsed)


def _merge_batter_wrc_plus(base_df: pd.DataFrame, season: int, context: str) -> pd.DataFrame:
    """Left-merge cached FanGraphs wRC+ data into a one-season batting table."""
    out = base_df.copy()
    if "wRC+" not in out.columns:
        out["wRC+"] = pd.NA
    if out.empty:
        return out

    wrc_df = get_fg_batting_wrc_plus(season)
    if wrc_df.empty:
        out["wRC+"] = pd.to_numeric(out["wRC+"], errors="coerce")
        return out

    out["__season"] = int(season)
    created_name_key = "name_team_key" not in out.columns
    if created_name_key:
        out["name_team_key"] = [_name_team_key(name, team) for name, team in zip(out["Name"], out["Team"], strict=False)]

    work = wrc_df.copy()
    work["__season"] = pd.to_numeric(work["season"], errors="coerce").fillna(int(season)).astype(int)

    merge_keys: list[str]
    if (
        "key_mlbam" in out.columns
        and "key_mlbam" in work.columns
        and out["key_mlbam"].notna().any()
        and work["key_mlbam"].notna().any()
    ):
        out["key_mlbam"] = pd.to_numeric(out["key_mlbam"], errors="coerce").astype("Int64")
        work["key_mlbam"] = pd.to_numeric(work["key_mlbam"], errors="coerce").astype("Int64")
        merge_keys = ["__season", "key_mlbam"]
    elif "IDfg" in out.columns and "IDfg" in work.columns:
        out["IDfg"] = pd.to_numeric(out["IDfg"], errors="coerce")
        work["IDfg"] = pd.to_numeric(work["IDfg"], errors="coerce")
        merge_keys = ["__season", "IDfg"]
    else:
        merge_keys = ["__season", "name_team_key"]
        duplicate_keys = int(work["name_team_key"].duplicated(keep=False).sum())
        if duplicate_keys > 0:
            print(
                f"[wRC+] warning: {context} ({season}) has {duplicate_keys} duplicate name/team keys; "
                "fallback merge may be ambiguous."
            )

    work = work[merge_keys + ["wRC+"]].drop_duplicates(subset=merge_keys, keep="first")
    merged = out.merge(work.rename(columns={"wRC+": "__wRC_plus_fg"}), how="left", on=merge_keys)
    merged["wRC+"] = pd.to_numeric(merged["wRC+"], errors="coerce")
    merged["__wRC_plus_fg"] = pd.to_numeric(merged["__wRC_plus_fg"], errors="coerce")
    merged["wRC+"] = merged["wRC+"].fillna(merged["__wRC_plus_fg"])

    unmatched = int(merged["wRC+"].isna().sum())
    if 0 < unmatched < len(merged):
        print(f"[wRC+] warning: {context} ({season}) unmatched rows after merge: {unmatched}/{len(merged)}.")

    drop_cols = ["__season", "__wRC_plus_fg"]
    if created_name_key:
        drop_cols.append("name_team_key")
    return merged.drop(columns=drop_cols, errors="ignore")


def _sample_size_text(sample_sizes: dict[str, int | None], player_type: str) -> str:
    parts = [f"Pitches: {sample_sizes['N_pitches']:,}"]
    if sample_sizes["N_BIP"] is not None:
        parts.append(f"Balls in play: {sample_sizes['N_BIP']:,}")
    if sample_sizes["approx_PA"] is not None:
        parts.append(f"{_appearance_label(player_type)}: {sample_sizes['approx_PA']:,}")
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
    label_overrides: dict[str, str] | None = None,
) -> None:
    if not stats_order:
        st.info("No stats selected.")
        return

    label_map = label_overrides or {}
    for stat_row in _chunk_stats(stats_order):
        row_cols = st.columns(_GRID_COLS_PER_ROW)
        for col_idx in range(_GRID_COLS_PER_ROW):
            with row_cols[col_idx]:
                if col_idx < len(stat_row):
                    stat = stat_row[col_idx]
                    stat_card(
                        stat_key=stat,
                        label=label_map.get(stat, stat),
                        value=stat_values.get(stat),
                        percentile=percentiles.get(stat, float("nan")),
                        color_tier=color_tiers.get(stat, {"hex": "#95A5A6"}),
                    )
                else:
                    st.empty()


def _render_delta_stat_grid(
    stats_order: list[str],
    player_stats_a: dict[str, float | None],
    player_stats_b: dict[str, float | None],
    label_overrides: dict[str, str] | None = None,
) -> None:
    if not stats_order:
        st.info("No stats selected.")
        return

    label_map = label_overrides or {}
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
                        + _DELTA_TILE_HTML.format(label=label_map.get(stat, stat), value=delta_str, value_color=value_color),
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

    if st.session_state.get("season_a") not in STATCAST_SEASONS:
        if "season_a" in st.session_state:
            st.warning(
                f"Season A {st.session_state['season_a']} is outside supported Statcast seasons "
                f"({STATCAST_START_YEAR}-{CURRENT_YEAR}); reset to {DEFAULT_SEASON}."
            )
        st.session_state["season_a"] = DEFAULT_SEASON
    if st.session_state.get("season_b") not in STATCAST_SEASONS:
        if "season_b" in st.session_state:
            st.warning(
                f"Season B {st.session_state['season_b']} is outside supported Statcast seasons "
                f"({STATCAST_START_YEAR}-{CURRENT_YEAR}); reset to {st.session_state['season_a']}."
            )
        st.session_state["season_b"] = st.session_state["season_a"]
    if "link_seasons" not in st.session_state:
        st.session_state["link_seasons"] = True
    debug_param = str(st.query_params.get("debug", "")).strip().lower()
    experimental_param = str(st.query_params.get("experimental", "")).strip().lower()
    query_authed = (
        debug_param in {"1", "true", "yes", "on"}
        or experimental_param in {"1", "true", "yes", "on"}
    )
    if "enable_query_input" not in st.session_state:
        st.session_state["enable_query_input"] = query_authed
    elif query_authed:
        st.session_state["enable_query_input"] = True

    # Player-type mode selector (feature-flagged).
    player_type = DEFAULT_PLAYER_TYPE
    if FEATURE_PITCHERS:
        if "player_type" not in st.session_state:
            st.session_state["player_type"] = DEFAULT_PLAYER_TYPE
        player_type = st.radio("Player type", ["Batter", "Pitcher"], horizontal=True, key="player_type")

    _prev_player_type = st.session_state.get("_prev_player_type")
    if _prev_player_type is None:
        st.session_state["_prev_player_type"] = player_type
    elif _prev_player_type != player_type:
        st.session_state["_prev_player_type"] = player_type
        st.session_state["player_selectbox"] = None
        st.session_state["player_b_selectbox"] = None
        st.session_state.pop("selected_stats_requested", None)
        st.session_state.pop("trend_custom_stats", None)
        for stat in CORE_STATS + BATTER_EXTRA_STATS + PITCHER_EXTRA_STATS:
            st.session_state.pop(f"stat_show_{stat}", None)
        st.session_state["filter_rows"] = [
            row for row in st.session_state.get("filter_rows", [])
            if row.get("filter_type") in _available_filter_keys(player_type)
        ]
        st.rerun()

    if st.session_state["enable_query_input"] and player_type == "Pitcher":
        st.caption("NL query is currently available for batter mode only.")
    elif st.session_state["enable_query_input"]:
        with st.form("nl_query_form"):
            st.text_area("Type a query…", key="nl_query_input", height=100)
            nl_run = st.form_submit_button("Run", width="stretch")

        if nl_run:
            raw_query = st.session_state.get("nl_query_input", "").strip()
            if not raw_query:
                st.session_state["_nl_parsed_preview"] = {
                    "raw_query": "",
                    "cleaned_query": "",
                    "player_a": None,
                    "player_b": None,
                    "comparison_mode": False,
                    "season_a": None,
                    "season_b": None,
                    "link_seasons": True,
                    "season": None,
                    "selected_stats": [],
                    "filter_rows": [],
                    "warnings": ["Query is empty. Enter a player-oriented query and try again."],
                }
                st.rerun()

            parsed_year = extract_last_year(raw_query)
            lookup_season = parsed_year if parsed_year in SEASONS else st.session_state["season_a"]

            with st.spinner("Parsing query…"):
                lookup_df = (
                    get_batting_stats(lookup_season)
                    if player_type == "Batter"
                    else get_pitching_stats(lookup_season)
                )

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

            parsed_season_a = parsed_intent.get("season_a")
            parsed_season_b = parsed_intent.get("season_b")
            parsed_link_seasons = parsed_intent.get("link_seasons")
            if parsed_season_a in SEASONS:
                st.session_state["season_a"] = parsed_season_a
            if parsed_season_b in SEASONS:
                st.session_state["season_b"] = parsed_season_b
            if isinstance(parsed_link_seasons, bool):
                st.session_state["link_seasons"] = parsed_link_seasons

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
                st.session_state["link_seasons"] = True

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

    # ── Season & split ────────────────────────────────────────────────────────
    if st.session_state["season_a"] not in SEASONS:
        st.session_state["season_a"] = DEFAULT_SEASON
    season_a = st.selectbox("Season", options=SEASONS, key="season_a")
    if st.session_state.get("link_seasons", True):
        st.session_state["season_b"] = season_a

    st.divider()

    # ── Player list load ──────────────────────────────────────────────────────
    with st.spinner("Loading player list…"):
        season_df = (
            get_batting_stats(season_a)
            if player_type == "Batter"
            else get_pitching_stats(season_a)
        )
    if player_type == "Batter":
        season_df = _merge_batter_wrc_plus(season_df, season_a, context="Player A")
    if season_df.empty and season_df.attrs.get("warning"):
        st.warning(str(season_df.attrs["warning"]))

    player_names = sorted(season_df["Name"].dropna().unique().tolist(), key=lambda n: n.split()[-1])
    season_df_b_fg = season_df  # default; overridden below for unlinked cross-year mode

    # ── Player A ──────────────────────────────────────────────────────────────
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
        st.warning(f"No MLB Statcast data for {selected_name} in {season_a}.")

    if selected_name is not None:
        if st.button("Clear player", width="stretch"):
            st.session_state["_clear_player_pending"] = True
            st.rerun()

    # ── Comparison mode + Player B ────────────────────────────────────────────
    comparison_mode = st.checkbox("Comparison mode", key="comparison_mode")

    selected_name_b = None
    if comparison_mode:
        # Per-side season state — allows cross-year comparison (e.g. 2025 A vs 2024 B).
        link_seasons = st.checkbox(
            "Link seasons",
            key="link_seasons",
            help="Unlink to compare players across different seasons",
        )
        if link_seasons:
            st.session_state["season_b"] = season_a
        else:
            if st.session_state.get("season_b") not in STATCAST_SEASONS:
                st.session_state["season_b"] = season_a
            st.selectbox("Season B", STATCAST_SEASONS, key="season_b")

        season_b = st.session_state["season_b"]

        if st.session_state.pop("_clear_player_b_pending", False):
            st.session_state["player_b_selectbox"] = None

        # Load Player B's FG data for the correct season (may differ from A when unlinked).
        if not link_seasons and season_b != season_a:
            with st.spinner("Loading Player B player list…"):
                season_df_b_fg = (
                    get_batting_stats(season_b)
                    if player_type == "Batter"
                    else get_pitching_stats(season_b)
                )
                if player_type == "Batter":
                    season_df_b_fg = _merge_batter_wrc_plus(season_df_b_fg, season_b, context="Player B")
            if season_df_b_fg.empty and season_df_b_fg.attrs.get("warning"):
                st.warning(str(season_df_b_fg.attrs["warning"]))
            player_b_names = sorted(
                season_df_b_fg["Name"].dropna().unique().tolist(),
                key=lambda n: n.split()[-1],
            )
            player_b_names = [n for n in player_b_names if n != selected_name]
        else:
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
            st.warning(f"No MLB Statcast data for {selected_name_b} in {season_b}.")

        if selected_name_b is not None:
            if st.button("Clear player B", width="stretch"):
                st.session_state["_clear_player_b_pending"] = True
                st.rerun()
    else:
        link_seasons = True
        st.session_state["link_seasons"] = True
        st.session_state["season_b"] = season_a
        season_b = season_a

    if player_type == "Batter":
        with st.expander("Options", expanded=False):
            show_trad = st.toggle(
                "Show traditional slash stats (AVG/OBP/SLG/OPS)",
                value=False,
                key="show_traditional_batting_stats",
            )
            _ = show_trad

    # ── Stats to show ─────────────────────────────────────────────────────────
    show_traditional_batting_stats = bool(st.session_state.get("show_traditional_batting_stats", False))
    batter_stats_catalog = BATTER_EXTRA_STATS + CORE_STATS
    if show_traditional_batting_stats:
        batter_stats_catalog = batter_stats_catalog + TRADITIONAL_STATS

    stats_catalog = (
        PITCHER_CORE_STATS + PITCHER_SECONDARY_STATS
        if player_type == "Pitcher"
        else batter_stats_catalog
    )
    stats_selection_key = "selected_stats_requested"
    if stats_selection_key not in st.session_state:
        st.session_state[stats_selection_key] = (
            (PITCHER_CORE_STATS + PITCHER_SECONDARY_STATS)
            if player_type == "Pitcher"
            else _DEFAULT_STATS.copy()
        )

    if player_type == "Batter":
        prev_toggle_key = "show_traditional_batting_stats_prev"
        if prev_toggle_key not in st.session_state:
            st.session_state[prev_toggle_key] = show_traditional_batting_stats
        prev_show_traditional = bool(st.session_state.get(prev_toggle_key, show_traditional_batting_stats))
        selected_stats_state = list(st.session_state.get(stats_selection_key, _DEFAULT_STATS.copy()))

        if show_traditional_batting_stats and not prev_show_traditional:
            selected_stats_state = list(dict.fromkeys(selected_stats_state + TRADITIONAL_STATS))
            st.session_state[stats_selection_key] = selected_stats_state
            for stat in TRADITIONAL_STATS:
                st.session_state[f"stat_show_{stat}"] = True
        elif (not show_traditional_batting_stats) and prev_show_traditional:
            selected_stats_state = [stat for stat in selected_stats_state if stat not in TRADITIONAL_STATS]
            st.session_state[stats_selection_key] = selected_stats_state
            for stat in TRADITIONAL_STATS:
                st.session_state[f"stat_show_{stat}"] = False

        st.session_state[prev_toggle_key] = show_traditional_batting_stats

    _selected_count = sum(
        1 for stat in stats_catalog if st.session_state.get(f"stat_show_{stat}", True)
    )
    with st.expander(f"Stats to show ({_selected_count}/{len(stats_catalog)})", expanded=False):
        for stat in stats_catalog:
            k = f"stat_show_{stat}"
            if k not in st.session_state:
                st.session_state[k] = stat in st.session_state[stats_selection_key]

        btn_col1, btn_col2, btn_col3 = st.columns(3)
        with btn_col1:
            if st.button("Reset", key="stats_reset_defaults", width="stretch",
                         help="Restore default stats"):
                default_stats = (
                    PITCHER_CORE_STATS + PITCHER_SECONDARY_STATS
                    if player_type == "Pitcher"
                    else _DEFAULT_STATS
                )
                for stat in stats_catalog:
                    st.session_state[f"stat_show_{stat}"] = stat in default_stats
                st.rerun()
        with btn_col2:
            if st.button("All", key="stats_select_all", width="stretch",
                         help="Enable all stats"):
                for stat in stats_catalog:
                    st.session_state[f"stat_show_{stat}"] = True
                st.rerun()
        with btn_col3:
            if st.button("None", key="stats_select_none", width="stretch"):
                for stat in stats_catalog:
                    st.session_state[f"stat_show_{stat}"] = False
                st.rerun()

        cols = st.columns(2)
        for i, stat in enumerate(stats_catalog):
            with cols[i % 2]:
                st.checkbox(stat, key=f"stat_show_{stat}")

    selected_stats_requested = [
        stat for stat in stats_catalog
        if st.session_state.get(f"stat_show_{stat}", False)
    ]
    st.session_state[stats_selection_key] = selected_stats_requested

    # ── Filters ───────────────────────────────────────────────────────────────
    # --- filter row session state init ---
    if "filter_rows" not in st.session_state:
        st.session_state["filter_rows"] = []
    if "_filter_next_id" not in st.session_state:
        st.session_state["_filter_next_id"] = 0

    available_filter_keys = _available_filter_keys(player_type)
    available_filter_labels = [_sidebar_filter_label(k, player_type) for k in available_filter_keys]

    # Guard against cross-mode carryover (e.g., pitcher_hand row in pitcher mode).
    filtered_rows = [
        row for row in st.session_state["filter_rows"]
        if row.get("filter_type") in available_filter_keys
    ]
    if len(filtered_rows) != len(st.session_state["filter_rows"]):
        st.session_state["filter_rows"] = filtered_rows

    _n_filters = len(st.session_state["filter_rows"])
    _filter_label = f"Filters ({_n_filters} active)" if _n_filters > 0 else "Filters"
    with st.expander(_filter_label, expanded=False):
        if st.button("+ Add filter", key="filter_add"):
            new_id = f"f{st.session_state['_filter_next_id']}"
            st.session_state["_filter_next_id"] += 1
            st.session_state["filter_rows"].append({
                "id":          new_id,
                "filter_type": "inning" if "inning" in available_filter_keys else available_filter_keys[0],
                "params":      FILTER_REGISTRY["inning"].default_params.copy(),
            })
            if st.session_state["filter_rows"][-1]["filter_type"] != "inning":
                chosen = st.session_state["filter_rows"][-1]["filter_type"]
                st.session_state["filter_rows"][-1]["params"] = FILTER_REGISTRY[chosen].default_params.copy()
            st.rerun()

        rows = st.session_state["filter_rows"]

        if not rows:
            st.caption("No filters active.")

        for i, row in enumerate(rows):
            rid = row["id"]
            col_lbl, col_rm = st.columns([5, 1])

            with col_lbl:
                type_key = f"filter_{rid}_type"
                if row["filter_type"] not in available_filter_keys:
                    row["filter_type"] = available_filter_keys[0]
                    row["params"] = FILTER_REGISTRY[row["filter_type"]].default_params.copy()
                current_type_label = _sidebar_filter_label(row["filter_type"], player_type)
                if type_key not in st.session_state:
                    st.session_state[type_key] = current_type_label
                if st.session_state[type_key] not in available_filter_labels:
                    st.session_state[type_key] = current_type_label
                st.selectbox(
                    "Filter type",
                    options=available_filter_labels,
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
                    "Pitcher hand",
                    ["L", "R"],
                    key=k, horizontal=True, label_visibility="collapsed",
                )
                row["params"]["hand"] = hand

            elif ft == "batter_hand":
                k = f"filter_{rid}_batter_hand"
                if k not in st.session_state:
                    st.session_state[k] = row["params"].get("hand", "R")
                hand = st.radio(
                    "Batter hand",
                    ["L", "R"],
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
    if player_type == "Pitcher":
        active_filter_summary = active_filter_summary.replace("Pitcher handedness", "Batter hand")
        active_filter_summary = active_filter_summary.replace("Batter handedness", "Batter hand")
    st.caption(f"Active: {active_filter_summary}")

    filters = rows_to_split_filters(st.session_state["filter_rows"])

    with st.expander("Stat Reference", expanded=False):
        render_glossary(mode="compact", player_type=player_type)

    st.divider()
    st.caption("Data via pybaseball / Baseball Savant")

    with st.expander("Experimental", expanded=False):
        st.checkbox("Enable query input", key="enable_query_input")


# ---------------------------------------------------------------------------
# Landing state — no player selected
# ---------------------------------------------------------------------------

if selected_name is None:
    st.header("⚾ MLB Splits")
    if season_df.empty and season_df.attrs.get("warning"):
        st.warning(str(season_df.attrs["warning"]))
    st.markdown(
        "Search for a player in the sidebar to view advanced Statcast stats "
        "and splits by handedness, home/away, or month."
    )
    render_glossary()
    st.stop()

_comparison_incomplete = comparison_mode and selected_name_b is None


# ---------------------------------------------------------------------------
# Resolve player metadata
# ---------------------------------------------------------------------------

player_row = get_player_row(season_df, selected_name)
if player_row is None:
    st.warning(
        f"No MLB Statcast data for {selected_name} in {season_a}. "
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
player_row_b = None
if comparison_mode and selected_name_b is not None:
    player_row_b = get_player_row(season_df_b_fg, selected_name_b)
    if player_row_b is None:
        st.warning(
            f"No MLB Statcast data for {selected_name_b} in {season_b}. "
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

with st.spinner(f"Loading {season_a} Statcast data for {selected_name}…"):
    fetch_statcast_fn = get_statcast_batter if player_type == "Batter" else get_statcast_pitcher
    raw_statcast_df = fetch_statcast_fn(mlbam_id, season_a)

prepared_cache = st.session_state.setdefault(_PREPARED_DF_CACHE_KEY, {})
prepare_cache_key = (int(mlbam_id), int(season_a), str(player_type))
statcast_df = get_prepared_df_cached(
    raw_statcast_df,
    prepared_cache,
    prepare_cache_key,
    log_fn=print,
)

filtered_df = apply_filters(statcast_df, filters, pitcher_perspective=(player_type == "Pitcher"))
sample_sizes = get_sample_sizes(filtered_df)

statcast_df_b = None
filtered_df_b = None
sample_sizes_b = None
if comparison_mode and mlbam_id_b is not None:
    with st.spinner(f"Loading {season_b} Statcast data for {selected_name_b}…"):
        raw_statcast_df_b = fetch_statcast_fn(mlbam_id_b, season_b)
    prepare_cache_key_b = (int(mlbam_id_b), int(season_b), str(player_type))
    statcast_df_b = get_prepared_df_cached(
        raw_statcast_df_b,
        prepared_cache,
        prepare_cache_key_b,
        log_fn=print,
    )
    filtered_df_b = apply_filters(statcast_df_b, filters, pitcher_perspective=(player_type == "Pitcher"))
    sample_sizes_b = get_sample_sizes(filtered_df_b)

missing_stat_requirements: dict[str, list[str]] = {}
selected_stats: list[str] = []
validation_frames: list[tuple[str, object]] = [(selected_name, statcast_df)]
if comparison_mode and statcast_df_b is not None:
    validation_frames.append((selected_name_b, statcast_df_b))

for stat in selected_stats_requested:
    if stat in TRADITIONAL_STATS and player_type == "Batter":
        selected_stats.append(stat)
        continue

    if stat == "wRC+" and player_type == "Batter":
        selected_stats.append(stat)
        continue

    if stat in {"K-BB%", "CSW%", "Whiff%", "FirstStrike%"} and player_type == "Pitcher":
        selected_stats.append(stat)
        continue

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

_raw = _compute_all_pitcher_stats(filtered_df) if player_type == "Pitcher" else _compute_stats(filtered_df)
if player_type == "Batter":
    _raw["wRC+"] = _as_optional_float(player_row.get("wRC+"))
    for stat in TRADITIONAL_STATS:
        _raw[stat] = _as_optional_float(player_row.get(stat))
player_stats = {stat: _raw.get(stat) for stat in selected_stats}

distributions = (
    build_pitcher_league_distributions(season_df)
    if player_type == "Pitcher"
    else build_league_distributions(season_df)
)
if player_type == "Batter" and "wRC+" in season_df.columns:
    wrc_values = pd.to_numeric(season_df["wRC+"], errors="coerce").dropna().to_numpy(dtype=float)
    if len(wrc_values) > 0:
        distributions["wRC+"] = wrc_values
if player_type == "Batter":
    for stat in TRADITIONAL_STATS:
        if stat in season_df.columns:
            stat_values = pd.to_numeric(season_df[stat], errors="coerce").dropna().to_numpy(dtype=float)
            if len(stat_values) > 0:
                distributions[stat] = stat_values
percentiles   = get_all_percentiles(player_stats, distributions, player_type=player_type)
color_tiers   = get_all_color_tiers(percentiles)

player_stats_b = None
percentiles_b = None
color_tiers_b = None
if comparison_mode and filtered_df_b is not None:
    _raw_b = _compute_all_pitcher_stats(filtered_df_b) if player_type == "Pitcher" else _compute_stats(filtered_df_b)
    if player_type == "Batter" and player_row_b is not None:
        _raw_b["wRC+"] = _as_optional_float(player_row_b.get("wRC+"))
        for stat in TRADITIONAL_STATS:
            _raw_b[stat] = _as_optional_float(player_row_b.get(stat))
    player_stats_b = {stat: _raw_b.get(stat) for stat in selected_stats}
    # When seasons differ, percentile Player B against their own year's league population.
    distributions_b = (
        build_pitcher_league_distributions(season_df_b_fg)
        if player_type == "Pitcher"
        else (
            build_league_distributions(season_df_b_fg)
            if season_b != season_a
            else distributions
        )
    )
    if player_type == "Batter" and "wRC+" in season_df_b_fg.columns:
        wrc_values_b = pd.to_numeric(season_df_b_fg["wRC+"], errors="coerce").dropna().to_numpy(dtype=float)
        if len(wrc_values_b) > 0:
            distributions_b["wRC+"] = wrc_values_b
    if player_type == "Batter":
        for stat in TRADITIONAL_STATS:
            if stat in season_df_b_fg.columns:
                stat_values_b = pd.to_numeric(season_df_b_fg[stat], errors="coerce").dropna().to_numpy(dtype=float)
                if len(stat_values_b) > 0:
                    distributions_b[stat] = stat_values_b
    percentiles_b = get_all_percentiles(player_stats_b, distributions_b, player_type=player_type)
    color_tiers_b = get_all_color_tiers(percentiles_b)


# ---------------------------------------------------------------------------
# Player header
# ---------------------------------------------------------------------------

if comparison_mode and team_b is not None:
    if season_a != season_b:
        st.subheader(f"{selected_name} ({season_a}) vs {selected_name_b} ({season_b})")
        st.caption(f"{team} vs {team_b} · {player_type}")
    else:
        st.subheader(f"{selected_name} vs {selected_name_b}")
        st.caption(f"{team} vs {team_b} · {season_a} · {player_type}")
else:
    player_header(selected_name, team, season_a, player_type)

if _comparison_incomplete:
    st.info("⬅ Select **Player B** in the sidebar to compare two players.")

st.divider()


# ---------------------------------------------------------------------------
# Season stat cards
# ---------------------------------------------------------------------------

if active_filter_summary == "No filters (full season data)":
    st.caption(active_filter_summary)
else:
    st.caption(f"Active filters: {active_filter_summary}")

st.subheader("Season Stats")
pitcher_label_overrides = _PITCHER_STAT_LABELS if player_type == "Pitcher" else None
if comparison_mode and player_stats_b is not None and percentiles_b is not None and color_tiers_b is not None:
    col_a, col_b, col_delta = st.columns(3)
    with col_a:
        st.markdown(f"**{selected_name}**")
        st.caption(_sample_size_text(sample_sizes, player_type))
        _render_player_stat_grid(
            player_stats,
            percentiles,
            color_tiers,
            selected_stats,
            label_overrides=pitcher_label_overrides,
        )
    with col_b:
        st.markdown(f"**{selected_name_b}**")
        st.caption(_sample_size_text(sample_sizes_b, player_type))
        _render_player_stat_grid(
            player_stats_b,
            percentiles_b,
            color_tiers_b,
            selected_stats,
            label_overrides=pitcher_label_overrides,
        )
    with col_delta:
        if season_a != season_b:
            st.markdown(f"**Difference ({selected_name} {season_a} minus {selected_name_b} {season_b})**")
        else:
            st.markdown(f"**Difference ({selected_name} minus {selected_name_b})**")
        st.caption(
            f"Positive means {selected_name} is higher; negative means {selected_name_b} is higher."
        )
        _render_delta_stat_grid(
            selected_stats,
            player_stats,
            player_stats_b,
            label_overrides=pitcher_label_overrides,
        )
else:
    st.caption(f"Sample size: {_sample_size_text(sample_sizes, player_type)}")
    if player_type == "Pitcher":
        core_card_order = ["wOBA", "xwOBA", "K-BB%", "K%", "BB%", "CSW%"]
        contact_order = ["HardHit%", "Barrel%", "GB%"]
        command_order = ["Whiff%", "FirstStrike%"]
        core_stats = [s for s in core_card_order if s in selected_stats]
        contact_stats = [s for s in contact_order if s in selected_stats]
        command_stats = [s for s in command_order if s in selected_stats]

        if core_stats:
            st.markdown("**Core Pitching Stats**")
            stat_cards_row(
                player_stats,
                percentiles,
                color_tiers,
                stats_order=core_stats,
                cols_per_row=3,
                label_overrides=_PITCHER_STAT_LABELS,
            )
            st.caption("wOBA/xwOBA allowed per BF result · K%, BB%, K-BB% per BF · CSW% per pitch")

        if contact_stats:
            st.markdown("**Contact & Batted Ball Profile**")
            stat_cards_row(
                player_stats,
                percentiles,
                color_tiers,
                stats_order=contact_stats,
                cols_per_row=3,
                label_overrides=_PITCHER_STAT_LABELS,
            )
            st.caption("per batted ball event")

        if command_stats:
            st.markdown("**Pitch Command Detail**")
            stat_cards_row(
                player_stats,
                percentiles,
                color_tiers,
                stats_order=command_stats,
                cols_per_row=2,
                label_overrides=_PITCHER_STAT_LABELS,
            )
            st.caption("Whiff% per swing · First-Strike% per 0-0 first pitch")

        if not (core_stats or contact_stats or command_stats):
            st.info("No stats selected.")
    else:
        stat_cards_row(player_stats, percentiles, color_tiers, stats_order=selected_stats)

st.divider()


# ---------------------------------------------------------------------------
# Pitch arsenal
# ---------------------------------------------------------------------------

if player_type == "Pitcher":
    if comparison_mode and filtered_df_b is not None:
        col_a_arsenal, col_b_arsenal = st.columns(2)
        with col_a_arsenal:
            st.caption(f"Player A: {selected_name}")
            render_pitch_arsenal(compute_pitch_arsenal(filtered_df))
        with col_b_arsenal:
            st.caption(f"Player B: {selected_name_b}")
            render_pitch_arsenal(compute_pitch_arsenal(filtered_df_b))
    else:
        render_pitch_arsenal(compute_pitch_arsenal(filtered_df))

if FEATURE_PITCH_ZONE:
    from ui.components import render_pitch_zone_chart

    _pitch_zone_role = "pitcher" if player_type == "Pitcher" else "batter"
    render_pitch_zone_chart(filtered_df, role=_pitch_zone_role)

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

st.caption("Split type = table rows. Filters = data pool for each row.")
if st.session_state.get("split_type_label") not in SPLIT_TYPE_MAP:
    st.session_state["split_type_label"] = next(iter(SPLIT_TYPE_MAP))
st.radio(
    "Split by",
    list(SPLIT_TYPE_MAP.keys()),
    key="split_type_label",
    help="Sets the rows in the splits table. Use Filters to narrow which pitches are counted for all rows.",
    horizontal=True,
)
split_type_label = st.session_state["split_type_label"]
split_type = SPLIT_TYPE_MAP[split_type_label]
st.subheader(f"Splits: {split_type_label}")

if comparison_mode and statcast_df_b is not None and filtered_df_b is not None:
    left_split_col, right_split_col = st.columns(2)

    with left_split_col:
        st.caption(f"Player A: {selected_name}")
        if statcast_df.empty:
            st.warning(
                f"No Statcast data found for {selected_name} in {season_a}. "
                "They may not have had enough plate appearances or the season data "
                "may not yet be available."
            )
        else:
            splits_df = (
                get_pitcher_splits(filtered_df, split_type)
                if player_type == "Pitcher"
                else get_splits(filtered_df, split_type)
            )
            if splits_df.empty or splits_df["PA"].sum() == 0:
                st.info(f"No {'batters faced' if player_type == 'Pitcher' else 'plate appearances'} found for the selected split.")
            else:
                split_cols = ["Split", "PA"] + [s for s in selected_stats if s in splits_df.columns]
                split_view = splits_df[split_cols]
                if player_type == "Pitcher":
                    split_view = split_view.rename(columns=_PITCHER_STAT_LABELS)
                split_table(split_view)

    with right_split_col:
        st.caption(f"Player B: {selected_name_b}")
        if statcast_df_b.empty:
            st.warning(
                f"No Statcast data found for {selected_name_b} in {season_b}. "
                "They may not have had enough plate appearances or the season data "
                "may not yet be available."
            )
        else:
            splits_df_b = (
                get_pitcher_splits(filtered_df_b, split_type)
                if player_type == "Pitcher"
                else get_splits(filtered_df_b, split_type)
            )
            if splits_df_b.empty or splits_df_b["PA"].sum() == 0:
                st.info(f"No {'batters faced' if player_type == 'Pitcher' else 'plate appearances'} found for the selected split.")
            else:
                split_cols_b = ["Split", "PA"] + [s for s in selected_stats if s in splits_df_b.columns]
                split_view_b = splits_df_b[split_cols_b]
                if player_type == "Pitcher":
                    split_view_b = split_view_b.rename(columns=_PITCHER_STAT_LABELS)
                split_table(split_view_b)
else:
    if statcast_df.empty:
        st.warning(
            f"No Statcast data found for {selected_name} in {season_a}. "
            "They may not have had enough plate appearances or the season data "
            "may not yet be available."
        )
    else:
        splits_df = (
            get_pitcher_splits(filtered_df, split_type)
            if player_type == "Pitcher"
            else get_splits(filtered_df, split_type)
        )
        if splits_df.empty or splits_df["PA"].sum() == 0:
            st.info(f"No {'batters faced' if player_type == 'Pitcher' else 'plate appearances'} found for the selected split.")
        else:
            split_cols = ["Split", "PA"] + [s for s in selected_stats if s in splits_df.columns]
            split_view = splits_df[split_cols]
            if player_type == "Pitcher":
                split_view = split_view.rename(columns=_PITCHER_STAT_LABELS)
            split_table(split_view)

st.divider()


# ---------------------------------------------------------------------------
# Trend by Year
# ---------------------------------------------------------------------------

with st.expander("Player Trend by Year", expanded=False):
    trend_seasons_a = sorted(s for s in STATCAST_SEASONS if s <= season_a)
    trend_seasons_b = sorted(s for s in STATCAST_SEASONS if s <= season_b)
    trend_control_max = max(season_a, season_b) if comparison_mode else season_a
    trend_seasons_control = sorted(s for s in STATCAST_SEASONS if s <= trend_control_max)
    available_trend_stats = selected_stats if selected_stats else list(STAT_REGISTRY.keys())

    trend_year_min = trend_seasons_control[0]
    trend_year_max = trend_seasons_control[-1]
    trend_default_start = max(trend_year_min, trend_year_max - 6)
    if "trend_year_range" not in st.session_state:
        st.session_state["trend_year_range"] = (trend_default_start, trend_year_max)
    else:
        existing_start, existing_end = st.session_state["trend_year_range"]
        clamped_start = min(max(int(existing_start), trend_year_min), trend_year_max)
        clamped_end = min(max(int(existing_end), trend_year_min), trend_year_max)
        if clamped_start > clamped_end:
            clamped_start = clamped_end
        st.session_state["trend_year_range"] = (clamped_start, clamped_end)
    trend_year_cols = st.columns([5, 1])
    with trend_year_cols[1]:
        all_years_clicked = st.button("All years", key="trend_all_years_btn")
    if all_years_clicked:
        st.session_state["trend_year_range"] = (trend_year_min, trend_year_max)
    with trend_year_cols[0]:
        trend_year_range = st.slider(
            "Year range",
            min_value=trend_year_min,
            max_value=trend_year_max,
            step=1,
            key="trend_year_range",
        )

    apply_trend_filters = st.checkbox(
        "Apply current filters to each year",
        value=False,
        key="trend_apply_filters",
    )

    # Guard: only fetch/render when user explicitly requests it.
    # This prevents N Statcast API calls on every rerun while the expander is collapsed.
    # The context key captures the current player+season combo; when it changes the
    # "Load" button re-appears so stale data is never shown silently.
    _trend_ctx = (
        int(mlbam_id),
        int(season_a),
        int(mlbam_id_b) if (comparison_mode and mlbam_id_b is not None) else None,
        int(season_b) if comparison_mode else int(season_a),
    )
    _loaded_trend_ctx = st.session_state.get("_trend_loaded_key")

    if _trend_ctx != _loaded_trend_ctx:
        st.info("Click **Load trend data** to fetch yearly stats.")
        if st.button("Load trend data", key="trend_load_btn"):
            st.session_state["_trend_loaded_key"] = _trend_ctx
            st.rerun()
    else:
        trend_filters = filters if apply_trend_filters else SplitFilters()
        with st.spinner("Loading trend data…"):
            trend_data_a = get_trend_stats(
                mlbam_id,
                trend_seasons_a,
                player_type,
                trend_filters,
                get_statcast_batter if player_type == "Batter" else get_statcast_pitcher,
                prepared_cache,
            )
            trend_data_b_trend = None
            if comparison_mode and mlbam_id_b is not None:
                trend_data_b_trend = get_trend_stats(
                    mlbam_id_b,
                    trend_seasons_b,
                    player_type,
                    trend_filters,
                    get_statcast_batter if player_type == "Batter" else get_statcast_pitcher,
                    prepared_cache,
                )

        trend_tab_dash, trend_tab_custom = st.tabs(["Dashboard", "Custom"])

        with trend_tab_dash:
            render_trend_dashboard(
                trend_data_a=trend_data_a,
                trend_data_b=trend_data_b_trend,
                player_label_a=selected_name,
                player_label_b=selected_name_b if comparison_mode else None,
                year_range=trend_year_range,
                player_type=player_type,
                apply_filters_to_each_year=apply_trend_filters,
                active_filter_summary=active_filter_summary,
            )

        with trend_tab_custom:
            if "trend_custom_stats" in st.session_state:
                st.session_state["trend_custom_stats"] = [
                    stat for stat in st.session_state["trend_custom_stats"]
                    if stat in available_trend_stats
                ]
            selected_custom_stats = st.multiselect(
                "Select stats to plot (max 4)",
                options=available_trend_stats,
                default=available_trend_stats[:2],
                key="trend_custom_stats",
            )
            render_trend_custom(
                trend_data_a=trend_data_a,
                trend_data_b=trend_data_b_trend,
                player_label_a=selected_name,
                player_label_b=selected_name_b if comparison_mode else None,
                year_range=trend_year_range,
                player_type=player_type,
                available_stats=available_trend_stats,
                selected_custom_stats=selected_custom_stats,
                apply_filters_to_each_year=apply_trend_filters,
                active_filter_summary=active_filter_summary,
            )

st.divider()
