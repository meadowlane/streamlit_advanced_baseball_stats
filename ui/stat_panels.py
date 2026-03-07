"""Stat cards, tables, and header UI helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go  # type: ignore[import-untyped]
import streamlit as st
from streamlit.elements.lib.column_types import ColumnConfig

from stats.percentiles import LOWER_IS_BETTER

# Display format per stat: (format_string, suffix)
_STAT_FORMAT: dict[str, tuple[str, str]] = {
    "wOBA": (".3f", ""),
    "xwOBA": (".3f", ""),
    "ERA": (".2f", ""),
    "FIP": (".2f", ""),
    "xFIP": (".2f", ""),
    "SIERA": (".2f", ""),
    "xERA": (".2f", ""),
    "K%": (".1f", "%"),
    "BB%": (".1f", "%"),
    "FB%": (".1f", "%"),
    "HardHit%": (".1f", "%"),
    "Barrel%": (".1f", "%"),
    "wRC+": (".0f", ""),
    "AVG": (".3f", ""),
    "OBP": (".3f", ""),
    "SLG": (".3f", ""),
    "OPS": (".3f", ""),
    "GB%": (".1f", "%"),
    "CSW%": (".1f", "%"),
    "Whiff%": (".1f", "%"),
    "FirstStrike%": (".1f", "%"),
    "K-BB%": (".1f", "%"),
    "EV": (".1f", ""),
    "LA": (".1f", ""),
    "FBv": (".1f", ""),
    "HR": (".0f", ""),
    "RBI": (".0f", ""),
    "SO": (".0f", ""),
    "Stuff+": (".0f", ""),
    "Location+": (".0f", ""),
    "Pitching+": (".0f", ""),
}

_SPLIT_TABLE_HELP: dict[str, str] = {
    "PA": "Sample size for the split. Plate appearances for hitters; batters faced for pitchers.",
    "wOBA": "Weighted On-Base Average for the split. Higher is better for hitters; lower is better for pitchers.",
    "wOBA Allowed": "Weighted On-Base Average allowed by pitchers in this split. Lower is better for pitchers.",
    "xwOBA": "Expected wOBA based on contact quality plus K/BB inputs. Higher is better for hitters; lower is better for pitchers.",
    "xwOBA Allowed": "Expected wOBA allowed by pitchers in this split. Lower is better for pitchers.",
    "K%": "Strikeout rate in this split. Higher is better for pitchers; lower is better for hitters.",
    "BB%": "Walk rate in this split. Higher is better for hitters; lower is better for pitchers.",
    "K-BB%": "Strikeout rate minus walk rate. Higher is better for pitchers.",
    "CSW%": "Called Strikes + Whiffs rate per pitch. Higher is generally better for pitchers.",
    "Whiff%": "Swing-and-miss rate per swing. Higher is generally better for pitchers.",
    "FirstStrike%": "First-pitch strike rate at 0-0 counts. Higher is generally better for pitchers.",
    "HardHit%": "Share of batted balls at 95+ mph. Higher is better for hitters; lower is better for pitchers.",
    "Barrel%": "Share of batted balls classified as barrels. Higher is better for hitters; lower is better for pitchers.",
    "GB%": "Ground-ball rate on balls in play. Often better higher for pitchers; context-dependent for hitters.",
}

_SPLIT_TABLE_FORMAT: dict[str, ColumnConfig | str | None] = {
    "PA": st.column_config.NumberColumn(
        "PA", format="%d", help=_SPLIT_TABLE_HELP.get("PA", None)
    ),
    "wOBA": st.column_config.NumberColumn(
        "wOBA", format="%.3f", help=_SPLIT_TABLE_HELP.get("wOBA", None)
    ),
    "wOBA Allowed": st.column_config.NumberColumn(
        "wOBA Allowed", format="%.3f", help=_SPLIT_TABLE_HELP.get("wOBA Allowed", None)
    ),
    "xwOBA": st.column_config.NumberColumn(
        "xwOBA", format="%.3f", help=_SPLIT_TABLE_HELP.get("xwOBA", None)
    ),
    "xwOBA Allowed": st.column_config.NumberColumn(
        "xwOBA Allowed",
        format="%.3f",
        help=_SPLIT_TABLE_HELP.get("xwOBA Allowed", None),
    ),
    "K%": st.column_config.NumberColumn(
        "K%", format="%.1f%%", help=_SPLIT_TABLE_HELP.get("K%", None)
    ),
    "BB%": st.column_config.NumberColumn(
        "BB%", format="%.1f%%", help=_SPLIT_TABLE_HELP.get("BB%", None)
    ),
    "HardHit%": st.column_config.NumberColumn(
        "HardHit%", format="%.1f%%", help=_SPLIT_TABLE_HELP.get("HardHit%", None)
    ),
    "Barrel%": st.column_config.NumberColumn(
        "Barrel%", format="%.1f%%", help=_SPLIT_TABLE_HELP.get("Barrel%", None)
    ),
    "GB%": st.column_config.NumberColumn(
        "GB%", format="%.1f%%", help=_SPLIT_TABLE_HELP.get("GB%", None)
    ),
    "K-BB%": st.column_config.NumberColumn(
        "K-BB%", format="%.1f%%", help=_SPLIT_TABLE_HELP.get("K-BB%", None)
    ),
    "CSW%": st.column_config.NumberColumn(
        "CSW%", format="%.1f%%", help=_SPLIT_TABLE_HELP.get("CSW%", None)
    ),
    "Whiff%": st.column_config.NumberColumn(
        "Whiff%", format="%.1f%%", help=_SPLIT_TABLE_HELP.get("Whiff%", None)
    ),
    "FirstStrike%": st.column_config.NumberColumn(
        "FirstStrike%",
        format="%.1f%%",
        help=_SPLIT_TABLE_HELP.get("FirstStrike%", None),
    ),
}

_ARSENAL_TABLE_HELP: dict[str, str] = {
    "Pitch": "Pitch type classification from Statcast for this arsenal row.",
    "N": "Number of pitches of this type in the selected sample.",
    "Usage%": "Share of all pitches thrown that were this pitch type.",
    "Velo": "Average velocity for this pitch type in miles per hour (mph).",
    "Spin": "Average spin rate for this pitch type in revolutions per minute (RPM).",
    "CSW%": "Called Strikes + Whiffs rate for this pitch type per pitch.",
    "Whiff%": "Swing-and-miss rate for this pitch type per swing.",
}

_ORDERED_STATS = ["wOBA", "xwOBA", "K%", "BB%", "HardHit%", "Barrel%"]


def format_stat_value(stat: str, value: float | None) -> str:
    """Return a display string for a stat value (e.g. '.380', '22.1%')."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "—"
    fmt, suffix = _STAT_FORMAT.get(stat, (".3f", ""))
    result = f"{value:{fmt}}{suffix}"
    if stat in {"wOBA", "xwOBA"} and result.startswith("0."):
        result = result[1:]
    return result


def format_percentile(percentile: float) -> str:
    """Return a compact percentile string (e.g. '87th') or '—' for NaN."""
    if percentile is None or (isinstance(percentile, float) and np.isnan(percentile)):
        return "—"
    return f"{int(round(percentile))}th"


def build_chart_df(
    percentiles: dict[str, float],
    color_tiers: dict[str, dict[str, str]],
    stat_values: dict[str, float | None],
    stats_order: list[str] | None = None,
) -> pd.DataFrame:
    """Build the DataFrame fed to the percentile bar chart."""
    order = stats_order if stats_order is not None else _ORDERED_STATS
    rows = []
    for stat in order:
        pct = percentiles.get(stat, np.nan)
        color = color_tiers.get(stat, {}).get("hex", "#95A5A6")
        val_str = format_stat_value(stat, stat_values.get(stat))
        direction = "↓ lower is better" if stat in LOWER_IS_BETTER else ""
        rows.append(
            {
                "stat": stat,
                "percentile": pct if not np.isnan(pct) else 0.0,
                "color": color,
                "label": f"{val_str}  ({format_percentile(pct)})",
                "direction": direction,
            }
        )
    return pd.DataFrame(rows)


_CARD_CSS = """
<style>
.stat-card {{
    border: 1px solid rgba(255, 255, 255, 0.12);
    border-radius: 10px;
    padding: 14px 10px 12px;
    text-align: center;
    background: #1e2029;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.4);
}}
.stat-label {{
    font-size: 11px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    color: rgba(255, 255, 255, 0.55);
    margin-bottom: 6px;
}}
.stat-value {{
    font-size: 26px;
    font-weight: 800;
    color: #f0f0f0;
    line-height: 1.1;
    margin-bottom: 8px;
}}
.stat-badge {{
    display: inline-block;
    border-radius: 99px;
    padding: 2px 10px;
    font-size: 11px;
    font-weight: 700;
    color: #fff;
    background-color: {hex};
}}
.stat-card-compact {{
    padding: 10px 8px 9px;
}}
.stat-card-compact .stat-label {{
    font-size: 10px;
    margin-bottom: 4px;
}}
.stat-card-compact .stat-value {{
    font-size: 24px;
    margin-bottom: 6px;
}}
</style>
"""

_CARD_HTML = """
<div class="{card_class}">
  <div class="stat-label">{label}</div>
  <div class="stat-value">{value}</div>
  {badge_html}
</div>
"""


def stat_card(
    stat_key: str,
    label: str,
    value: float | None,
    percentile: float,
    color_tier: dict[str, str],
    compact: bool = False,
) -> None:
    """Render a single stat card with a colored percentile badge."""
    hex_color = color_tier.get("hex", "#95A5A6")
    val_str = format_stat_value(stat_key, value)
    pct_str = format_percentile(percentile)
    badge_html = (
        ""
        if pct_str == "—"
        else f'<div class="stat-badge" style="background-color:{hex_color};">{pct_str}</div>'
    )
    card_class = "stat-card stat-card-compact" if compact else "stat-card"
    st.markdown(
        _CARD_CSS.format(hex=hex_color)
        + _CARD_HTML.format(
            card_class=card_class,
            label=label,
            value=val_str,
            badge_html=badge_html,
        ),
        unsafe_allow_html=True,
    )


def stat_cards_row(
    stat_values: dict[str, float | None],
    percentiles: dict[str, float],
    color_tiers: dict[str, dict[str, str]],
    stats_order: list[str] | None = None,
    cols_per_row: int | None = None,
    label_overrides: dict[str, str] | None = None,
    compact: bool = False,
) -> None:
    """Render stat cards in a single row or wrapped rows when requested."""
    order = stats_order if stats_order is not None else _ORDERED_STATS
    if not order:
        st.info("No stats selected.")
        return

    if cols_per_row is None:
        cols = st.columns(len(order))
        for col, stat in zip(cols, order):
            with col:
                stat_card(
                    stat_key=stat,
                    label=(label_overrides or {}).get(stat, stat),
                    value=stat_values.get(stat),
                    percentile=percentiles.get(stat, np.nan),
                    color_tier=color_tiers.get(stat, {"hex": "#95A5A6"}),
                    compact=compact,
                )
        return

    row_width = max(1, int(cols_per_row))
    label_map = label_overrides or {}
    for start in range(0, len(order), row_width):
        if start > 0:
            st.markdown('<div style="height:6px;"></div>', unsafe_allow_html=True)
        row_stats = order[start : start + row_width]
        cols = st.columns(row_width)
        for idx in range(row_width):
            with cols[idx]:
                if idx < len(row_stats):
                    stat = row_stats[idx]
                    stat_card(
                        stat_key=stat,
                        label=label_map.get(stat, stat),
                        value=stat_values.get(stat),
                        percentile=percentiles.get(stat, np.nan),
                        color_tier=color_tiers.get(stat, {"hex": "#95A5A6"}),
                        compact=compact,
                    )
                else:
                    st.empty()


def percentile_bar_chart(
    percentiles: dict[str, float],
    color_tiers: dict[str, dict[str, str]],
    stat_values: dict[str, float | None],
    stats_order: list[str] | None = None,
) -> None:
    """Render a horizontal Plotly bar chart of percentile ranks."""
    df = build_chart_df(percentiles, color_tiers, stat_values, stats_order=stats_order)

    if df.empty:
        st.info("No stats selected.")
        return

    fig = go.Figure()

    for _, row in df.iterrows():
        fig.add_trace(
            go.Bar(
                x=[row["percentile"]],
                y=[row["stat"]],
                orientation="h",
                marker_color=row["color"],
                text=[row["label"]],
                textposition="outside",
                cliponaxis=False,
                showlegend=False,
                hoverinfo="skip",
            )
        )

    fig.update_layout(
        template="plotly_dark",
        xaxis=dict(
            range=[0, 115],
            title="Percentile rank",
            tickvals=[0, 25, 50, 75, 100],
            showgrid=True,
            gridcolor="rgba(128,128,128,0.2)",
        ),
        yaxis=dict(autorange="reversed", tickfont=dict(size=13, family="monospace")),
        height=280,
        margin=dict(l=20, r=10, t=10, b=40),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        bargap=0.35,
    )
    fig.add_vline(
        x=50, line_dash="dot", line_color="rgba(128,128,128,0.5)", line_width=1
    )

    st.plotly_chart(
        fig,
        width="stretch",
        config={"displayModeBar": False},
    )


def split_table(df: pd.DataFrame) -> None:
    """Render the split results DataFrame with formatted columns."""
    if df.empty:
        st.info("No split data available.")
        return

    col_config = {
        col: cfg for col, cfg in _SPLIT_TABLE_FORMAT.items() if col in df.columns
    }

    st.dataframe(
        df,
        width="stretch",
        hide_index=True,
        column_config=col_config,
    )


def render_pitch_arsenal(arsenal_df: pd.DataFrame) -> None:
    """Render a pitcher pitch-arsenal summary table."""
    st.subheader("Pitch Arsenal")
    st.caption("min. 25 pitches per type")
    with st.expander("Show pitch mix", expanded=True):
        if arsenal_df.empty:
            st.info("No pitch arsenal data available for this selection.")
            return

        st.dataframe(
            arsenal_df,
            width="stretch",
            hide_index=True,
            column_config={
                "Pitch": st.column_config.TextColumn(
                    "Pitch", help=_ARSENAL_TABLE_HELP.get("Pitch", None)
                ),
                "N": st.column_config.NumberColumn(
                    "N", format="%d", help=_ARSENAL_TABLE_HELP.get("N", None)
                ),
                "Usage%": st.column_config.NumberColumn(
                    "Usage%",
                    format="%.1f%%",
                    help=_ARSENAL_TABLE_HELP.get("Usage%", None),
                ),
                "Velo": st.column_config.NumberColumn(
                    "Velo", format="%.1f", help=_ARSENAL_TABLE_HELP.get("Velo", None)
                ),
                "Spin": st.column_config.NumberColumn(
                    "Spin", format="%d", help=_ARSENAL_TABLE_HELP.get("Spin", None)
                ),
                "CSW%": st.column_config.NumberColumn(
                    "CSW%", format="%.1f%%", help=_ARSENAL_TABLE_HELP.get("CSW%", None)
                ),
                "Whiff%": st.column_config.NumberColumn(
                    "Whiff%",
                    format="%.1f%%",
                    help=_ARSENAL_TABLE_HELP.get("Whiff%", None),
                ),
            },
        )


def player_header(name: str, team: str, season: int, player_type: str) -> None:
    """Render a compact player header (name · team · season · type)."""
    col_text, col_spacer = st.columns([3, 1])
    del col_spacer
    with col_text:
        st.subheader(name)
        st.caption(f"{team} · {season} · {player_type}")
