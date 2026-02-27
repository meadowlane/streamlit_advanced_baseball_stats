"""Reusable Streamlit UI components for mlb-splits.

All public functions render directly into the current Streamlit context.
Data transformation helpers (format_stat_value, build_chart_df) are kept
pure so they can be tested without a Streamlit runtime.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from stats.percentiles import CORE_STATS, LOWER_IS_BETTER

# ---------------------------------------------------------------------------
# Formatting helpers (pure — no Streamlit calls)
# ---------------------------------------------------------------------------

# Display format per stat: (format_string, suffix)
_STAT_FORMAT: dict[str, tuple[str, str]] = {
    "wOBA":      (".3f", ""),
    "xwOBA":     (".3f", ""),
    "K%":        (".1f", "%"),
    "BB%":       (".1f", "%"),
    "HardHit%":  (".1f", "%"),
    "Barrel%":   (".1f", "%"),
}

# Human-readable labels for split table column headers
_SPLIT_TABLE_FORMAT: dict[str, st.column_config.Column] = {
    "PA":        st.column_config.NumberColumn("PA",       format="%d"),
    "wOBA":      st.column_config.NumberColumn("wOBA",     format="%.3f"),
    "xwOBA":     st.column_config.NumberColumn("xwOBA",    format="%.3f"),
    "K%":        st.column_config.NumberColumn("K%",       format="%.1f%%"),
    "BB%":       st.column_config.NumberColumn("BB%",      format="%.1f%%"),
    "HardHit%":  st.column_config.NumberColumn("HardHit%", format="%.1f%%"),
    "Barrel%":   st.column_config.NumberColumn("Barrel%",  format="%.1f%%"),
}

# Ordered display sequence for the 6 core stats
_ORDERED_STATS = ["wOBA", "xwOBA", "K%", "BB%", "HardHit%", "Barrel%"]


def format_stat_value(stat: str, value: float | None) -> str:
    """Return a display string for a stat value (e.g. '.380', '22.1%').

    wOBA and xwOBA follow Baseball Savant convention: no leading zero ('.380').
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "—"
    fmt, suffix = _STAT_FORMAT.get(stat, (".3f", ""))
    result = f"{value:{fmt}}{suffix}"
    # Strip leading zero for proportion-scale stats (wOBA, xwOBA)
    if suffix == "" and result.startswith("0."):
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
    """Build the DataFrame fed to the percentile bar chart.

    Returns columns: stat, percentile, color, label, direction_note.
    Pure function — no Streamlit calls.
    """
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


# ---------------------------------------------------------------------------
# Stat cards
# ---------------------------------------------------------------------------

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
</style>
"""

_CARD_HTML = """
<div class="stat-card">
  <div class="stat-label">{label}</div>
  <div class="stat-value">{value}</div>
  <div class="stat-badge" style="background-color:{hex};">{pct_str}</div>
</div>
"""


def stat_card(
    label: str,
    value: float | None,
    percentile: float,
    color_tier: dict[str, str],
) -> None:
    """Render a single stat card with a colored percentile badge."""
    hex_color = color_tier.get("hex", "#95A5A6")
    val_str = format_stat_value(label, value)
    pct_str = format_percentile(percentile)
    st.markdown(
        _CARD_CSS.format(hex=hex_color)
        + _CARD_HTML.format(label=label, value=val_str, hex=hex_color, pct_str=pct_str),
        unsafe_allow_html=True,
    )


def stat_cards_row(
    stat_values: dict[str, float | None],
    percentiles: dict[str, float],
    color_tiers: dict[str, dict[str, str]],
    stats_order: list[str] | None = None,
) -> None:
    """Render all 6 core stat cards in a single row of columns."""
    order = stats_order if stats_order is not None else _ORDERED_STATS
    if not order:
        st.info("No stats selected.")
        return
    cols = st.columns(len(order))
    for col, stat in zip(cols, order):
        with col:
            stat_card(
                label=stat,
                value=stat_values.get(stat),
                percentile=percentiles.get(stat, np.nan),
                color_tier=color_tiers.get(stat, {"hex": "#95A5A6"}),
            )


# ---------------------------------------------------------------------------
# Percentile bar chart
# ---------------------------------------------------------------------------

def percentile_bar_chart(
    percentiles: dict[str, float],
    color_tiers: dict[str, dict[str, str]],
    stat_values: dict[str, float | None],
    stats_order: list[str] | None = None,
) -> None:
    """Render a horizontal Plotly bar chart of all 6 percentile ranks."""
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
        xaxis=dict(
            range=[0, 115],  # extra room for outside text labels
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

    # League-average reference line at 50th
    fig.add_vline(x=50, line_dash="dot", line_color="rgba(128,128,128,0.5)", line_width=1)

    st.plotly_chart(
        fig,
        use_container_width=True,
        config={"displayModeBar": False},
    )


# ---------------------------------------------------------------------------
# Split DataTable
# ---------------------------------------------------------------------------

def split_table(df: pd.DataFrame) -> None:
    """Render the split results DataFrame with formatted columns."""
    if df.empty:
        st.info("No split data available.")
        return

    # Build column_config only for columns that are present
    col_config = {
        col: cfg
        for col, cfg in _SPLIT_TABLE_FORMAT.items()
        if col in df.columns
    }

    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        column_config=col_config,
    )


# ---------------------------------------------------------------------------
# Player header
# ---------------------------------------------------------------------------

def player_header(name: str, team: str, season: int, player_type: str) -> None:
    """Render a compact player header (name · team · season · type)."""
    col_text, col_spacer = st.columns([3, 1])
    with col_text:
        st.subheader(name)
        st.caption(f"{team} · {season} · {player_type}")
