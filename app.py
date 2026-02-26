import math

import streamlit as st

from data.fetcher import (
    get_batting_stats,
    get_mlbam_id,
    get_player_row,
    get_statcast_batter,
)
from stats.percentiles import (
    PROPORTION_STATS,
    build_league_distributions,
    get_all_color_tiers,
    get_all_percentiles,
)
from stats.splits import get_splits
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

    player_names = sorted(season_df["Name"].dropna().unique().tolist())

    selected_name = st.selectbox(
        "Player",
        options=player_names,
        index=None,
        placeholder="Type to search…",
    )

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
    st.error(f"Could not find {selected_name} in the {season} season data.")
    st.stop()

fg_id = int(player_row["IDfg"])
team  = str(player_row.get("Team", "—"))

with st.spinner("Resolving player ID…"):
    mlbam_id = get_mlbam_id(fg_id)

if mlbam_id is None:
    st.error(f"Could not resolve a Statcast ID for {selected_name}.")
    st.stop()


# ---------------------------------------------------------------------------
# Build season-level stats for percentile engine
# Convert FanGraphs proportion stats (0–1) to display scale (0–100)
# ---------------------------------------------------------------------------

player_stats: dict[str, float | None] = {}
for stat in CORE_STATS:
    val = player_row.get(stat)
    if val is not None and not (isinstance(val, float) and math.isnan(float(val))):
        player_stats[stat] = float(val) * 100.0 if stat in PROPORTION_STATS else float(val)
    else:
        player_stats[stat] = None

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
# Split table — fetch Statcast data
# ---------------------------------------------------------------------------

st.subheader(f"Splits: {split_type_label}")

with st.spinner(f"Loading {season} Statcast data for {selected_name}…"):
    statcast_df = get_statcast_batter(mlbam_id, season)

if statcast_df.empty:
    st.warning(
        f"No Statcast data found for {selected_name} in {season}. "
        "They may not have had enough plate appearances or the season data "
        "may not yet be available."
    )
else:
    splits_df = get_splits(statcast_df, split_type)
    if splits_df.empty or splits_df["PA"].sum() == 0:
        st.info("No plate appearances found for the selected split.")
    else:
        split_table(splits_df)

st.divider()


# ---------------------------------------------------------------------------
# Glossary
# ---------------------------------------------------------------------------

render_glossary()
