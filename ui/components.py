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
from stats.splits import STAT_REGISTRY

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
    "GB%":       (".1f", "%"),
    "CSW%":      (".1f", "%"),
    "Whiff%":    (".1f", "%"),
    "FirstStrike%": (".1f", "%"),
    "K-BB%":     (".1f", "%"),
}

# Human-readable labels for split table column headers
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

_SPLIT_TABLE_FORMAT: dict[str, st.column_config.Column] = {
    "PA":        st.column_config.NumberColumn("PA",       format="%d", help=_SPLIT_TABLE_HELP.get("PA", None)),
    "wOBA":      st.column_config.NumberColumn("wOBA",     format="%.3f", help=_SPLIT_TABLE_HELP.get("wOBA", None)),
    "wOBA Allowed": st.column_config.NumberColumn("wOBA Allowed", format="%.3f", help=_SPLIT_TABLE_HELP.get("wOBA Allowed", None)),
    "xwOBA":     st.column_config.NumberColumn("xwOBA",    format="%.3f", help=_SPLIT_TABLE_HELP.get("xwOBA", None)),
    "xwOBA Allowed": st.column_config.NumberColumn("xwOBA Allowed", format="%.3f", help=_SPLIT_TABLE_HELP.get("xwOBA Allowed", None)),
    "K%":        st.column_config.NumberColumn("K%",       format="%.1f%%", help=_SPLIT_TABLE_HELP.get("K%", None)),
    "BB%":       st.column_config.NumberColumn("BB%",      format="%.1f%%", help=_SPLIT_TABLE_HELP.get("BB%", None)),
    "HardHit%":  st.column_config.NumberColumn("HardHit%", format="%.1f%%", help=_SPLIT_TABLE_HELP.get("HardHit%", None)),
    "Barrel%":   st.column_config.NumberColumn("Barrel%",  format="%.1f%%", help=_SPLIT_TABLE_HELP.get("Barrel%", None)),
    "GB%":       st.column_config.NumberColumn("GB%",      format="%.1f%%", help=_SPLIT_TABLE_HELP.get("GB%", None)),
    "K-BB%":     st.column_config.NumberColumn("K-BB%",    format="%.1f%%", help=_SPLIT_TABLE_HELP.get("K-BB%", None)),
    "CSW%":      st.column_config.NumberColumn("CSW%",     format="%.1f%%", help=_SPLIT_TABLE_HELP.get("CSW%", None)),
    "Whiff%":    st.column_config.NumberColumn("Whiff%",   format="%.1f%%", help=_SPLIT_TABLE_HELP.get("Whiff%", None)),
    "FirstStrike%": st.column_config.NumberColumn("FirstStrike%", format="%.1f%%", help=_SPLIT_TABLE_HELP.get("FirstStrike%", None)),
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

# Ordered display sequence for the 6 core stats
_ORDERED_STATS = ["wOBA", "xwOBA", "K%", "BB%", "HardHit%", "Barrel%"]

# Pitch zone chart constants
ZONE_X_BOUNDS = (-0.8333, 0.8333)  # feet
ZONE_Z_FALLBACK = (1.5, 3.5)
_PITCH_TYPE_COLORS: dict[str, str] = {
    "FF": "#1f77b4",
    "SI": "#ff7f0e",
    "SL": "#2ca02c",
    "CH": "#d62728",
    "CU": "#9467bd",
    "KC": "#8c564b",
    "FC": "#e377c2",
    "FS": "#7f7f7f",
    "ST": "#bcbd22",
}
_PLOTLY_QUAL_PALETTE = [
    "#636EFA",
    "#EF553B",
    "#00CC96",
    "#AB63FA",
    "#FFA15A",
    "#19D3F3",
    "#FF6692",
    "#B6E880",
    "#FF97FF",
    "#FECB52",
]
_OUTCOME_SYMBOLS: dict[str, str] = {
    "Ball": "circle-open",
    "Called Strike": "circle",
    "Swinging Strike": "x",
    "Foul": "triangle-up",
    "In Play": "square",
    "Other": "diamond",
}
_OUTCOME_ORDER = ["Ball", "Called Strike", "Swinging Strike", "Foul", "In Play", "Other"]
_OUTCOME_LEGEND_HTML = (
    '<p style="font-size:12px;color:rgba(255,255,255,0.60);margin:2px 0 6px 0;">'
    "○ Ball &nbsp;·&nbsp; ● Called Strike &nbsp;·&nbsp; ✕ Swinging Strike"
    " &nbsp;·&nbsp; ▲ Foul &nbsp;·&nbsp; ■ In Play"
    "</p>"
)


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
    stat_key: str,
    label: str,
    value: float | None,
    percentile: float,
    color_tier: dict[str, str],
) -> None:
    """Render a single stat card with a colored percentile badge."""
    hex_color = color_tier.get("hex", "#95A5A6")
    # Always format by internal stat key; display label may be an override.
    val_str = format_stat_value(stat_key, value)
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
    cols_per_row: int | None = None,
    label_overrides: dict[str, str] | None = None,
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
                )
        return

    row_width = max(1, int(cols_per_row))
    label_map = label_overrides or {}
    for start in range(0, len(order), row_width):
        row_stats = order[start:start + row_width]
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
                    )
                else:
                    st.empty()


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
        template="plotly_dark",
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
        width="stretch",
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
                "Pitch": st.column_config.TextColumn("Pitch", help=_ARSENAL_TABLE_HELP.get("Pitch", None)),
                "N": st.column_config.NumberColumn("N", format="%d", help=_ARSENAL_TABLE_HELP.get("N", None)),
                "Usage%": st.column_config.NumberColumn("Usage%", format="%.1f%%", help=_ARSENAL_TABLE_HELP.get("Usage%", None)),
                "Velo": st.column_config.NumberColumn("Velo", format="%.1f", help=_ARSENAL_TABLE_HELP.get("Velo", None)),
                "Spin": st.column_config.NumberColumn("Spin", format="%d", help=_ARSENAL_TABLE_HELP.get("Spin", None)),
                "CSW%": st.column_config.NumberColumn("CSW%", format="%.1f%%", help=_ARSENAL_TABLE_HELP.get("CSW%", None)),
                "Whiff%": st.column_config.NumberColumn("Whiff%", format="%.1f%%", help=_ARSENAL_TABLE_HELP.get("Whiff%", None)),
            },
        )


# ---------------------------------------------------------------------------
# Pitch zone chart
# ---------------------------------------------------------------------------

def _zone_bounds(df: pd.DataFrame) -> tuple[float, float, float, float]:
    """Return strike-zone bounds as (x_min, x_max, z_bot, z_top)."""
    x_min, x_max = ZONE_X_BOUNDS
    z_bot, z_top = ZONE_Z_FALLBACK

    if "sz_bot" in df.columns:
        sz_bot = pd.to_numeric(df["sz_bot"], errors="coerce").dropna()
        if not sz_bot.empty:
            z_bot = float(sz_bot.median())

    if "sz_top" in df.columns:
        sz_top = pd.to_numeric(df["sz_top"], errors="coerce").dropna()
        if not sz_top.empty:
            z_top = float(sz_top.median())

    if z_top <= z_bot:
        z_bot, z_top = ZONE_Z_FALLBACK

    return float(x_min), float(x_max), float(z_bot), float(z_top)


def _add_zone_shapes(
    fig: go.Figure,
    x_min: float,
    x_max: float,
    z_bot: float,
    z_top: float,
    grid: int = 3,
    draw_grid: bool = True,
) -> None:
    """Add strike-zone outer box and per-cell rectangle grid to figure shapes."""
    fig.add_shape(
        type="rect",
        x0=x_min,
        x1=x_max,
        y0=z_bot,
        y1=z_top,
        line=dict(width=2, color="rgba(200,200,200,0.75)"),
        fillcolor="rgba(0,0,0,0)",
    )

    if not draw_grid:
        return

    cell_w = (x_max - x_min) / float(grid)
    cell_h = (z_top - z_bot) / float(grid)

    for row in range(grid):
        for col in range(grid):
            cx0 = x_min + col * cell_w
            cx1 = cx0 + cell_w
            cy0 = z_bot + row * cell_h
            cy1 = cy0 + cell_h
            fig.add_shape(
                type="rect",
                x0=cx0,
                x1=cx1,
                y0=cy0,
                y1=cy1,
                line=dict(width=0.6, color="rgba(100,100,100,0.22)"),
                fillcolor="rgba(0,0,0,0)",
            )


def _compute_zone_histogram(
    df: pd.DataFrame,
    x_min: float,
    x_max: float,
    z_bot: float,
    z_top: float,
    bins: int = 20,
    normalize: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute 2D histogram for plate_x/plate_z within the strike-zone bounds."""
    if df.empty:
        return np.zeros((bins, bins)), np.linspace(x_min, x_max, bins + 1), np.linspace(z_bot, z_top, bins + 1)

    in_bounds = df[
        (df["plate_x"] >= x_min)
        & (df["plate_x"] <= x_max)
        & (df["plate_z"] >= z_bot)
        & (df["plate_z"] <= z_top)
    ].copy()
    if in_bounds.empty:
        return np.zeros((bins, bins)), np.linspace(x_min, x_max, bins + 1), np.linspace(z_bot, z_top, bins + 1)

    hist, x_edges, z_edges = np.histogram2d(
        in_bounds["plate_x"].to_numpy(),
        in_bounds["plate_z"].to_numpy(),
        bins=[bins, bins],
        range=[[x_min, x_max], [z_bot, z_top]],
    )
    if normalize and hist.sum() > 0:
        hist = hist / hist.sum()
    return hist, x_edges, z_edges


def _stable_downsample(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """
    Deterministically downsample to n rows in a way stable across reruns.

    Uses the best available sort-key set before selecting evenly spaced rows.
    """
    n_int = max(1, int(n))
    priority_cols = [
        ["game_pk", "at_bat_number", "pitch_number"],
        ["game_pk", "inning", "balls", "strikes"],
        ["game_date", "inning", "balls", "strikes"],
    ]

    chosen_cols: list[str] | None = None
    for cols in priority_cols:
        if all(col in df.columns for col in cols):
            chosen_cols = cols
            break

    if chosen_cols is not None:
        ordered = df.sort_values(by=chosen_cols, ascending=True, kind="mergesort").copy()
    else:
        ordered = df.sort_index().copy()

    if len(ordered) <= n_int:
        return ordered

    idx = np.linspace(0, len(ordered) - 1, n_int, dtype=int)
    return ordered.iloc[idx].copy()


def _add_zone_heatmap_layer(
    fig: go.Figure,
    hist: np.ndarray,
    x_edges: np.ndarray,
    z_edges: np.ndarray,
    normalize: bool = False,
    show_scale: bool = False,
) -> None:
    """Add a subtle heatmap layer behind scatter traces."""
    x_centers = (x_edges[:-1] + x_edges[1:]) / 2.0
    z_centers = (z_edges[:-1] + z_edges[1:]) / 2.0
    fig.add_trace(
        go.Heatmap(
            x=x_centers,
            y=z_centers,
            z=hist.T,
            colorscale="Blues",
            opacity=0.20,
            showscale=show_scale,
            colorbar=dict(
                title="Density %" if normalize else "Count",
                ticksuffix="%" if normalize else "",
                tickformat=".1%" if normalize else None,
                len=0.55,
            ),
            hovertemplate=(
                "Plate X: %{x:.2f} ft<br>"
                "Plate Z: %{y:.2f} ft<br>"
                + ("Density: %{z:.2%}" if normalize else "Count: %{z:.0f}")
                + "<extra></extra>"
            ),
        )
    )


def _bucket_outcome(description: str) -> str:
    """Map Statcast pitch description to a coarse outcome group."""
    desc = str(description or "").strip().lower()
    if desc == "called_strike":
        return "Called Strike"
    if desc in {"swinging_strike", "swinging_strike_blocked", "missed_bunt"}:
        return "Swinging Strike"
    if desc in {"foul", "foul_tip", "foul_bunt"}:
        return "Foul"
    if desc in {"ball", "blocked_ball", "pitchout"}:
        return "Ball"
    if desc.startswith("hit_into_play"):
        return "In Play"
    return "Other"


def _ball_strike_bucket(outcome_bucket: str) -> str:
    """Map 6-way outcome bucket to Ball/Strike/Unknown."""
    if outcome_bucket == "Ball":
        return "Ball"
    if outcome_bucket in {"Called Strike", "Swinging Strike", "Foul", "In Play"}:
        return "Strike"
    return "Unknown"


def _derive_outcome_fields(description: str) -> tuple[str, str]:
    """Return (outcome_bucket, bs_bucket) for a pitch description."""
    outcome_bucket = _bucket_outcome(description)
    return outcome_bucket, _ball_strike_bucket(outcome_bucket)


def _build_zone_chart(
    df: pd.DataFrame,
    max_pitches: int = 500,
    grid: int = 3,
    pitch_types: list[str] | None = None,
    encode_outcomes: bool = False,
    selected_outcomes: list[str] | None = None,
    show_heatmap: bool = False,
    heatmap_bins: int = 20,
    heatmap_normalize: bool = False,
    heatmap_show_scale: bool = False,
    show_grid_overlay: bool = True,
    view_mode: str = "Zoom to zone",
) -> go.Figure:
    """Build pitch-location scatter plot with strike-zone grid overlay."""
    work = df.copy()

    if pitch_types is not None:
        if "pitch_type" not in work.columns:
            return go.Figure()
        work = work[work["pitch_type"].astype(str).isin([str(p) for p in pitch_types])].copy()

    if "pitch_type" in work.columns:
        work["pitch_type"] = work["pitch_type"].fillna("UNK").astype(str)
    else:
        work["pitch_type"] = "UNK"
    if "description" in work.columns:
        outcomes = work["description"].fillna("").map(_derive_outcome_fields)
    else:
        outcomes = pd.Series([_derive_outcome_fields("")] * len(work), index=work.index)
    work["outcome_bucket"] = outcomes.map(lambda x: x[0])
    work["bs_bucket"] = outcomes.map(lambda x: x[1])
    if selected_outcomes is not None:
        allowed = {str(outcome) for outcome in selected_outcomes}
        work = work[work["outcome_bucket"].isin(allowed)].copy()
    work["plate_x"] = pd.to_numeric(work["plate_x"], errors="coerce")
    work["plate_z"] = pd.to_numeric(work["plate_z"], errors="coerce")
    work = work.dropna(subset=["plate_x", "plate_z"]).copy()
    if work.empty:
        return go.Figure()

    sample_n = min(int(max_pitches), len(work))
    work = _stable_downsample(work, sample_n)

    x_min, x_max, z_bot, z_top = _zone_bounds(work)
    fig = go.Figure()
    if show_heatmap:
        hist, x_edges, z_edges = _compute_zone_histogram(
            work,
            x_min=x_min,
            x_max=x_max,
            z_bot=z_bot,
            z_top=z_top,
            bins=int(heatmap_bins),
            normalize=bool(heatmap_normalize),
        )
        _add_zone_heatmap_layer(
            fig,
            hist,
            x_edges,
            z_edges,
            normalize=bool(heatmap_normalize),
            show_scale=bool(heatmap_show_scale),
        )

    pitch_order = sorted(work["pitch_type"].unique())
    fallback_types = [pt for pt in pitch_order if pt not in _PITCH_TYPE_COLORS]
    fallback_map = {
        pt: _PLOTLY_QUAL_PALETTE[idx % len(_PLOTLY_QUAL_PALETTE)]
        for idx, pt in enumerate(fallback_types)
    }

    for pitch_type in pitch_order:
        pitch_df = work[work["pitch_type"] == pitch_type].copy()
        if pitch_df.empty:
            continue

        pitch_color = _PITCH_TYPE_COLORS.get(pitch_type, fallback_map.get(pitch_type, "#9aa0a6"))
        if encode_outcomes:
            fig.add_trace(
                go.Scattergl(
                    x=[None],
                    y=[None],
                    mode="markers",
                    name=pitch_type,
                    legendgroup=f"pt:{pitch_type}",
                    showlegend=True,
                    marker=dict(
                        size=10,
                        symbol="circle",
                        color=pitch_color,
                        opacity=0.75,
                    ),
                    hoverinfo="skip",
                )
            )

        custom_cols: list[np.ndarray] = []
        hover_lines = [
            f"Pitch: {pitch_type}",
            "Ball/Strike: %{customdata[0]}",
            "Outcome: %{customdata[1]}",
            "x: %{x:.2f} ft",
            "z: %{y:.2f} ft",
        ]
        custom_idx = 2

        custom_cols.append(pitch_df["bs_bucket"].fillna("Unknown").astype(str).to_numpy())
        custom_cols.append(pitch_df["outcome_bucket"].fillna("Other").astype(str).to_numpy())

        if "release_speed" in pitch_df.columns:
            velo = pd.to_numeric(pitch_df["release_speed"], errors="coerce")
            custom_cols.append(np.where(velo.notna(), np.round(velo, 1).astype(str), "—"))
            hover_lines.append(f"Velo: %{{customdata[{custom_idx}]}} mph")
            custom_idx += 1

        if "description" in pitch_df.columns:
            desc = pitch_df["description"].fillna("—").astype(str).to_numpy()
            custom_cols.append(desc)
            hover_lines.append(f"Desc: %{{customdata[{custom_idx}]}}")
            custom_idx += 1

        if "events" in pitch_df.columns:
            events = pitch_df["events"].fillna("—").astype(str).to_numpy()
            custom_cols.append(events)
            hover_lines.append(f"Event: %{{customdata[{custom_idx}]}}")
            custom_idx += 1

        if "balls" in pitch_df.columns and "strikes" in pitch_df.columns:
            balls = pd.to_numeric(pitch_df["balls"], errors="coerce")
            strikes = pd.to_numeric(pitch_df["strikes"], errors="coerce")
            count_vals = np.where(
                balls.notna() & strikes.notna(),
                balls.astype("Int64").astype(str) + "-" + strikes.astype("Int64").astype(str),
                "—",
            )
            custom_cols.append(count_vals)
            hover_lines.append(f"Count: %{{customdata[{custom_idx}]}}")

        customdata = np.column_stack(custom_cols) if custom_cols else None
        marker_symbols = (
            pitch_df["outcome_bucket"].map(_OUTCOME_SYMBOLS).fillna("diamond").tolist()
            if encode_outcomes
            else "circle"
        )
        fig.add_trace(
            go.Scattergl(
                x=pitch_df["plate_x"],
                y=pitch_df["plate_z"],
                mode="markers",
                name=pitch_type,
                legendgroup=f"pt:{pitch_type}",
                showlegend=not encode_outcomes,
                marker=dict(
                    size=7,
                    symbol=marker_symbols,
                    color=pitch_color,
                    opacity=0.75,
                ),
                customdata=customdata,
                hovertemplate="<br>".join(hover_lines) + "<extra></extra>",
            )
        )

    _add_zone_shapes(fig, x_min, x_max, z_bot, z_top, grid=grid, draw_grid=bool(show_grid_overlay))

    if view_mode == "Zoom to zone":
        x_range = [x_min - 0.6, x_max + 0.6]
        y_range = [z_bot - 0.4, z_top + 0.4]
    else:
        x_data_min = float(work["plate_x"].min())
        x_data_max = float(work["plate_x"].max())
        y_data_min = float(work["plate_z"].min())
        y_data_max = float(work["plate_z"].max())
        x_range = [min(x_data_min - 0.3, x_min - 0.25), max(x_data_max + 0.3, x_max + 0.25)]
        y_range = [min(y_data_min - 0.3, z_bot - 0.3), max(y_data_max + 0.3, z_top + 0.3)]

    fig.update_layout(
        template="plotly_dark",
        height=480,
        showlegend=True,
        legend=dict(
            bgcolor="rgba(0,0,0,0.55)",
            bordercolor="rgba(255,255,255,0.25)",
            borderwidth=1,
            font=dict(size=12),
            itemsizing="constant",
            x=1.01,
            y=0.99,
            xanchor="left",
            yanchor="top",
        ),
        margin=dict(l=0, r=0, t=20, b=10),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(
            title="Plate X (ft)",
            range=x_range,
            zeroline=False,
            showgrid=True,
            gridcolor="rgba(128,128,128,0.15)",
        ),
        yaxis=dict(
            title="Plate Z (ft)",
            range=y_range,
            zeroline=False,
            showgrid=True,
            gridcolor="rgba(128,128,128,0.15)",
            scaleanchor="x",
            scaleratio=1,
        ),
    )

    return fig


def render_pitch_movement_chart(
    df: pd.DataFrame,
    max_pitches: int,
    pitch_types: list[str] | None = None,
    perspective: str = "Pitcher view",
    mirror_lhp: bool = False,
    vertical_axis_mode: str = "Break direction (down = more drop)",
) -> None:
    """Render horizontal/vertical movement scatter (pfx_x vs pfx_z)."""
    if "pfx_x" not in df.columns or "pfx_z" not in df.columns:
        st.info("Movement data not available.")
        return

    work = df.copy()
    if pitch_types is not None and "pitch_type" in work.columns:
        work = work[work["pitch_type"].astype(str).isin([str(p) for p in pitch_types])].copy()

    work["pfx_x"] = pd.to_numeric(work["pfx_x"], errors="coerce")
    work["pfx_z"] = pd.to_numeric(work["pfx_z"], errors="coerce")
    work = work.dropna(subset=["pfx_x", "pfx_z"]).copy()
    if work.empty:
        st.info("Movement data not available.")
        return

    sample_n = min(int(max_pitches), len(work))
    work = work.sample(n=sample_n, random_state=42).copy()
    if "pitch_type" in work.columns:
        work["pitch_type"] = work["pitch_type"].fillna("UNK").astype(str)
    else:
        work["pitch_type"] = "UNK"
    if "description" in work.columns:
        outcomes = work["description"].fillna("").map(_derive_outcome_fields)
    else:
        outcomes = pd.Series([_derive_outcome_fields("")] * len(work), index=work.index)
    work["outcome_bucket"] = outcomes.map(lambda x: x[0])
    work["bs_bucket"] = outcomes.map(lambda x: x[1])

    x_vals = work["pfx_x"].to_numpy(copy=True)
    y_vals = work["pfx_z"].to_numpy(copy=True)
    if perspective == "Pitcher view":
        x_vals = -x_vals

    if mirror_lhp and "p_throws" in work.columns:
        throws = work["p_throws"].fillna("").astype(str).str.upper().to_numpy()
        signs = np.where(throws == "L", -1.0, 1.0)
        x_vals = x_vals * signs

    if vertical_axis_mode == "Break direction (down = more drop)":
        y_vals = -y_vals

    work["movement_x"] = x_vals
    work["movement_z"] = y_vals

    pitch_order = sorted(work["pitch_type"].unique())
    fallback_types = [pt for pt in pitch_order if pt not in _PITCH_TYPE_COLORS]
    fallback_map = {
        pt: _PLOTLY_QUAL_PALETTE[idx % len(_PLOTLY_QUAL_PALETTE)]
        for idx, pt in enumerate(fallback_types)
    }

    fig = go.Figure()
    for pitch_type in pitch_order:
        trace_df = work[work["pitch_type"] == pitch_type].copy()
        if trace_df.empty:
            continue

        custom_cols: list[np.ndarray] = []
        hover_lines = [
            f"Pitch: {pitch_type}",
            "Ball/Strike: %{customdata[0]}",
            "Outcome: %{customdata[1]}",
            "Break X: %{x:.2f} ft",
            "Break Z: %{y:.2f} ft",
        ]
        custom_idx = 2

        custom_cols.append(trace_df["bs_bucket"].fillna("Unknown").astype(str).to_numpy())
        custom_cols.append(trace_df["outcome_bucket"].fillna("Other").astype(str).to_numpy())

        if "release_speed" in trace_df.columns:
            velo = pd.to_numeric(trace_df["release_speed"], errors="coerce")
            custom_cols.append(np.where(velo.notna(), np.round(velo, 1).astype(str), "—"))
            hover_lines.append(f"Velo: %{{customdata[{custom_idx}]}} mph")
            custom_idx += 1

        if "description" in trace_df.columns:
            desc = trace_df["description"].fillna("—").astype(str).to_numpy()
            custom_cols.append(desc)
            hover_lines.append(f"Desc: %{{customdata[{custom_idx}]}}")
            custom_idx += 1

        if "events" in trace_df.columns:
            events = trace_df["events"].fillna("—").astype(str).to_numpy()
            custom_cols.append(events)
            hover_lines.append(f"Event: %{{customdata[{custom_idx}]}}")

        customdata = np.column_stack(custom_cols) if custom_cols else None
        fig.add_trace(
            go.Scattergl(
                x=trace_df["movement_x"],
                y=trace_df["movement_z"],
                mode="markers",
                name=pitch_type,
                marker=dict(
                    size=7,
                    symbol="circle",
                    color=_PITCH_TYPE_COLORS.get(pitch_type, fallback_map.get(pitch_type, "#9aa0a6")),
                    opacity=0.75,
                ),
                customdata=customdata,
                hovertemplate="<br>".join(hover_lines) + "<extra></extra>",
            )
        )

    x_pad = 0.5
    y_pad = 0.5
    x_min = float(work["movement_x"].min()) - x_pad
    x_max = float(work["movement_x"].max()) + x_pad
    y_min = float(work["movement_z"].min()) - y_pad
    y_max = float(work["movement_z"].max()) + y_pad
    x_label = (
        "Horizontal Break (pitcher view, ft)"
        if perspective == "Pitcher view"
        else "Horizontal Break (Statcast view, ft)"
    )
    y_label = (
        "Vertical Break (induced/Statcast, ft)"
        if vertical_axis_mode == "Induced (Statcast)"
        else "Vertical Break (pitcher view; down = more drop, ft)"
    )
    fig.update_layout(
        template="plotly_dark",
        height=480,
        showlegend=True,
        legend=dict(
            bgcolor="rgba(0,0,0,0.55)",
            bordercolor="rgba(255,255,255,0.25)",
            borderwidth=1,
            font=dict(size=12),
            itemsizing="constant",
            x=1.01,
            y=0.99,
            xanchor="left",
            yanchor="top",
        ),
        margin=dict(l=0, r=0, t=20, b=10),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(
            title=x_label,
            range=[x_min, x_max],
            showgrid=True,
            gridcolor="rgba(128,128,128,0.15)",
            zeroline=False,
        ),
        yaxis=dict(
            title=y_label,
            range=[y_min, y_max],
            showgrid=True,
            gridcolor="rgba(128,128,128,0.15)",
            zeroline=False,
        ),
        shapes=[
            dict(
                type="line",
                x0=0,
                x1=0,
                y0=y_min,
                y1=y_max,
                line=dict(color="rgba(180,180,180,0.55)", width=1.5, dash="dot"),
            ),
            dict(
                type="line",
                x0=x_min,
                x1=x_max,
                y0=0,
                y1=0,
                line=dict(color="rgba(180,180,180,0.55)", width=1.5, dash="dot"),
            ),
        ],
    )
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})


PITCH_PRESETS: dict[str, list[str] | None] = {
    "All": None,
    "Fastballs": ["FF", "SI", "FC"],
    "Breaking": ["SL", "CU", "KC", "SV", "ST", "CS"],
    "Offspeed": ["CH", "FS", "FO"],
    "Putaway": ["SL", "CU", "CH", "ST", "FS"],
}


def render_pitch_zone_chart(df: pd.DataFrame) -> None:
    """Render interactive pitch-location chart with strike-zone overlay."""
    with st.expander("Pitch Location", expanded=False):
        if df.empty:
            st.info("No pitch data available for this selection.")
            return

        viz_df = df.copy()
        has_stand = "stand" in df.columns
        if "pitch_type" in viz_df.columns:
            pitch_type_options = sorted(viz_df["pitch_type"].dropna().astype(str).unique().tolist())
        else:
            pitch_type_options = []

        preset_col, hand_col = st.columns([2.5, 2.0])
        with preset_col:
            pitch_preset = st.radio(
                "Pitch group",
                options=list(PITCH_PRESETS.keys()),
                horizontal=True,
                key="pitch_zone_preset",
            )
        with hand_col:
            st.caption("Batter hand")
            batter_hand = st.radio(
                "",
                options=["All", "vs LHB", "vs RHB"],
                horizontal=True,
                key="pitch_zone_batter_hand",
                label_visibility="collapsed",
            )

        prev_preset = st.session_state.get("_pitch_zone_prev_preset")
        if prev_preset != pitch_preset:
            st.session_state["_pitch_zone_prev_preset"] = pitch_preset
            implied = PITCH_PRESETS.get(pitch_preset)
            if implied is not None:
                st.session_state["pitch_zone_pitch_types_advanced"] = [
                    pitch_type for pitch_type in implied if pitch_type in pitch_type_options
                ]

        with st.expander("Options", expanded=False):
            max_pitches = st.slider(
                "Max pitches",
                min_value=100,
                max_value=1000,
                value=500,
                step=100,
                key="pitch_zone_max_pitches",
            )
            include_switch = True
            if has_stand and batter_hand != "All":
                include_switch = st.checkbox(
                    "Include switch hitters",
                    value=True,
                    key="pitch_zone_include_switch",
                )

            st.markdown("**Zone Map**")
            encode_outcomes = st.checkbox(
                "Encode outcomes (symbols)",
                value=False,
                key="pitch_loc_encode_outcomes",
            )
            if encode_outcomes:
                selected_outcomes = st.multiselect(
                    "Outcomes",
                    options=_OUTCOME_ORDER,
                    default=_OUTCOME_ORDER,
                    key="pitch_loc_outcomes",
                )
            else:
                selected_outcomes = None

            grid_label = st.radio(
                "Grid",
                options=["3×3", "5×5"],
                horizontal=True,
                key="pitch_zone_grid",
            )
            grid = 3 if grid_label == "3×3" else 5

            view_mode = st.radio(
                "View",
                options=["Zoom to zone", "Show all pitches"],
                horizontal=True,
                key="pitch_loc_zone_view",
            )
            show_heatmap = st.checkbox("Show heatmap", value=False, key="pitch_zone_show_heatmap")
            normalize_heatmap = st.checkbox(
                "Normalize to %",
                value=False,
                key="pitch_zone_heatmap_normalize",
                disabled=not show_heatmap,
            )
            heatmap_bins = st.slider(
                "Heatmap resolution",
                min_value=10,
                max_value=40,
                value=20,
                step=1,
                key="pitch_zone_heatmap_bins",
                disabled=not show_heatmap,
            )
            heatmap_show_scale = st.checkbox(
                "Show heatmap scale",
                value=False,
                key="pitch_zone_heatmap_show_scale",
                disabled=not show_heatmap,
            )
            prev_heatmap = st.session_state.get("_pitch_loc_prev_show_heatmap")
            if prev_heatmap is None or bool(prev_heatmap) != bool(show_heatmap):
                st.session_state["pitch_loc_show_grid_overlay"] = not bool(show_heatmap)
                st.session_state["_pitch_loc_prev_show_heatmap"] = bool(show_heatmap)
            show_grid_overlay = st.checkbox(
                "Show grid overlay",
                key="pitch_loc_show_grid_overlay",
            )

            st.markdown("**Movement**")
            movement_perspective = st.radio(
                "Movement perspective",
                options=["Pitcher view", "Catcher/Statcast view"],
                horizontal=True,
                key="pitch_movement_perspective",
            )
            default_vertical_mode = (
                "Break direction (down = more drop)"
                if movement_perspective == "Pitcher view"
                else "Induced (Statcast)"
            )
            prev_perspective = st.session_state.get("_pitch_movement_prev_perspective")
            if prev_perspective != movement_perspective:
                st.session_state["pitch_movement_vertical_axis"] = default_vertical_mode
                st.session_state["_pitch_movement_prev_perspective"] = movement_perspective

            vertical_axis_mode = st.radio(
                "Vertical axis",
                options=["Induced (Statcast)", "Break direction (down = more drop)"],
                horizontal=True,
                key="pitch_movement_vertical_axis",
            )
            has_throws = "p_throws" in viz_df.columns and viz_df["p_throws"].notna().any()
            if has_throws:
                mirror_lhp = st.checkbox(
                    "Mirror LHP to RHP view (arm-side consistent)",
                    value=False,
                    key="pitch_movement_mirror_lhp",
                )
            else:
                mirror_lhp = False

            st.markdown("**Pitch types (advanced)**")
            if "pitch_zone_pitch_types_advanced" not in st.session_state:
                st.session_state["pitch_zone_pitch_types_advanced"] = pitch_type_options
            else:
                st.session_state["pitch_zone_pitch_types_advanced"] = [
                    pitch_type
                    for pitch_type in st.session_state["pitch_zone_pitch_types_advanced"]
                    if pitch_type in pitch_type_options
                ]
            selected_pitch_types = st.multiselect(
                "Pitch type",
                options=pitch_type_options,
                key="pitch_zone_pitch_types_advanced",
            )

        filtered_df = viz_df.copy()

        if has_stand:
            stand_vals = filtered_df["stand"].fillna("").astype(str).str.upper()
            if batter_hand == "vs LHB":
                keep = {"L", "S"} if include_switch else {"L"}
                filtered_df = filtered_df[stand_vals.isin(keep)].copy()
            elif batter_hand == "vs RHB":
                keep = {"R", "S"} if include_switch else {"R"}
                filtered_df = filtered_df[stand_vals.isin(keep)].copy()

        advanced_override = (
            bool(pitch_type_options)
            and set(selected_pitch_types) != set(pitch_type_options)
            and "pitch_type" in filtered_df.columns
        )
        if advanced_override:
            filtered_df = filtered_df[
                filtered_df["pitch_type"].astype(str).isin([str(p) for p in selected_pitch_types])
            ].copy()
        else:
            preset_types = PITCH_PRESETS[pitch_preset]
            if preset_types is not None and "pitch_type" in filtered_df.columns:
                available_types = set(filtered_df["pitch_type"].dropna().astype(str).unique().tolist())
                preset_match = sorted(available_types.intersection(set(preset_types)))
                if preset_match:
                    filtered_df = filtered_df[filtered_df["pitch_type"].astype(str).isin(preset_match)].copy()

        tab_zone, tab_movement = st.tabs(["Zone Map", "Movement"])
        with tab_zone:
            if encode_outcomes:
                st.markdown(_OUTCOME_LEGEND_HTML, unsafe_allow_html=True)

            if show_heatmap and max_pitches < 300:
                st.caption("ℹ Heatmap is most meaningful with ≥ 300 pitches. Increase Max pitches in Options.")

            if "plate_x" not in df.columns or "plate_z" not in df.columns:
                st.info("Pitch location columns are unavailable for this selection.")
            else:
                valid_zone_df = filtered_df.dropna(subset=["plate_x", "plate_z"])
                if valid_zone_df.empty:
                    st.info("No pitch locations available after filtering.")
                else:
                    fig = _build_zone_chart(
                        df=valid_zone_df,
                        max_pitches=max_pitches,
                        grid=grid,
                        pitch_types=None,
                        encode_outcomes=encode_outcomes,
                        selected_outcomes=selected_outcomes if encode_outcomes else None,
                        show_heatmap=show_heatmap,
                        heatmap_bins=heatmap_bins,
                        heatmap_normalize=normalize_heatmap,
                        heatmap_show_scale=heatmap_show_scale,
                        show_grid_overlay=show_grid_overlay,
                        view_mode=view_mode,
                    )
                    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

        with tab_movement:
            render_pitch_movement_chart(
                df=filtered_df,
                max_pitches=max_pitches,
                pitch_types=None,
                perspective=movement_perspective,
                mirror_lhp=mirror_lhp,
                vertical_axis_mode=vertical_axis_mode,
            )


# ---------------------------------------------------------------------------
# Trend line chart
# ---------------------------------------------------------------------------

_COLOR_TREND_A = "#4FC3F7"
_COLOR_TREND_B = "#FF8A65"
_TREND_TIDY_COLUMNS = ["year", "stat_key", "value", "n_pitches", "approx_pa", "n_bip"]
TREND_LOW_SAMPLE_PA = 50
_PITCHER_ONLY_PCT_STATS = {"CSW%", "Whiff%", "FirstStrike%", "K-BB%"}
_TREND_DASHBOARD_METRIC_COLORS = [_COLOR_TREND_A, "#FFB74D"]
_TREND_CUSTOM_COLORS = ["#A5D6A7", "#CE93D8", "#FFCC80", "#80DEEA"]
_TREND_MARKER_SYMBOLS = {
    "wOBA": "circle",
    "xwOBA": "square",
    "K%": "diamond",
    "BB%": "triangle-up",
    "HardHit%": "x",
    "Barrel%": "triangle-down",
    "K-BB%": "diamond-open",
    "CSW%": "cross",
    "Whiff%": "star",
    "FirstStrike%": "hexagon",
    "GB%": "pentagon",
}

def _build_trend_tidy_df(trend_data: list[dict], stats_order: list[str]) -> pd.DataFrame:
    rows: list[dict] = []
    for season_row in trend_data:
        year = season_row.get("season", season_row.get("year"))
        if year is None:
            continue

        n_pitches = season_row.get("n_pitches", season_row.get("N_pitches"))
        n_bip = season_row.get("n_bip", season_row.get("N_BIP"))
        approx_pa = season_row.get("approx_pa", season_row.get("approx_PA", season_row.get("PA")))

        # Accept already-tidy rows from optimized paths.
        if "stat_key" in season_row and "value" in season_row:
            stat_key = season_row.get("stat_key")
            if stat_key is None:
                continue
            if stats_order and stat_key not in stats_order:
                continue
            rows.append(
                {
                    "year": int(year),
                    "stat_key": stat_key,
                    "value": season_row.get("value"),
                    "n_pitches": n_pitches,
                    "n_bip": n_bip,
                    "approx_pa": approx_pa,
                }
            )
            continue

        for stat_key in stats_order:
            rows.append(
                {
                    "year": int(year),
                    "stat_key": stat_key,
                    "value": season_row.get(stat_key),
                    "n_pitches": n_pitches,
                    "n_bip": n_bip,
                    "approx_pa": approx_pa,
                }
            )

    return pd.DataFrame(rows, columns=_TREND_TIDY_COLUMNS)


def _trend_value_format(stat_key: str) -> str:
    return "%.3f" if stat_key in {"wOBA", "xwOBA"} else "%.1f%%"


def _trend_stat_label(stat_key: str) -> str:
    spec = STAT_REGISTRY.get(stat_key)
    return spec.label if spec is not None else stat_key


def _stat_formatter(stat_key: str) -> str:
    """Return formatter key used for trend chart axis/hover formatting."""
    spec = STAT_REGISTRY.get(stat_key)
    if spec is not None and spec.formatter in {"decimal_3", "pct_1"}:
        return spec.formatter
    if stat_key in _PITCHER_ONLY_PCT_STATS:
        return "pct_1"
    return "pct_1" if stat_key.endswith("%") else "decimal_3"


def _stats_share_scale(stat_keys: list[str]) -> bool:
    if not stat_keys:
        return True
    return len({_stat_formatter(key) for key in stat_keys}) == 1


def _filter_real_data_rows(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    n_pitches = pd.to_numeric(df["n_pitches"], errors="coerce")
    value = pd.to_numeric(df["value"], errors="coerce")
    return df[(n_pitches > 0) & value.notna()].copy()


def _trend_year_ticks(year_range: tuple[int, int]) -> tuple[list[int], list[str]]:
    year_start, year_end = int(year_range[0]), int(year_range[1])
    years = list(range(year_start, year_end + 1))
    return years, ["2020*" if int(y) == 2020 else str(int(y)) for y in years]


def _trend_hover_value_template(stat_key: str) -> str:
    if _stat_formatter(stat_key) == "decimal_3":
        return "%{y:.3f}"
    return "%{y:.1f}%"


def _trend_counts_for_hover(series: pd.Series) -> np.ndarray:
    vals = pd.to_numeric(series, errors="coerce")
    return np.array(
        [str(int(v)) if pd.notna(v) else "—" for v in vals],
        dtype=object,
    )


def _trend_stat_df(
    tidy_df: pd.DataFrame,
    stat_key: str,
    year_range: tuple[int, int],
) -> pd.DataFrame:
    if tidy_df.empty:
        return tidy_df.copy()
    year_start, year_end = int(year_range[0]), int(year_range[1])
    out = tidy_df[
        (tidy_df["stat_key"] == stat_key)
        & (tidy_df["year"] >= year_start)
        & (tidy_df["year"] <= year_end)
    ].copy()
    out = _filter_real_data_rows(out)
    return out.sort_values("year")


def _trend_title_suffix(year_range: tuple[int, int]) -> str:
    year_start, year_end = int(year_range[0]), int(year_range[1])
    if year_start == year_end:
        return f"{year_start}"
    return f"{year_start}-{year_end}"


def _render_trend_filter_caption(
    apply_filters_to_each_year: bool,
    active_filter_summary: str | None,
) -> None:
    if apply_filters_to_each_year and active_filter_summary and "No filters" not in active_filter_summary:
        st.caption(f"Filters applied to trend: {active_filter_summary}")
    elif not apply_filters_to_each_year:
        st.caption("Filters applied to trend: Off (full-season data each year).")


def _add_trend_traces(
    fig: go.Figure,
    stat_df: pd.DataFrame,
    player_label: str,
    stat_key: str,
    color: str,
    dash: str,
    player_type: str,
) -> bool:
    """Add normal + low-sample traces for one player/stat pair.

    Returns True when low-sample rows were plotted.
    """
    if stat_df.empty:
        return False

    del player_type  # Reserved for future hover-label customization.

    work = stat_df.sort_values("year").copy()
    approx_pa = pd.to_numeric(work["approx_pa"], errors="coerce")
    low_sample_mask = approx_pa.notna() & (approx_pa < TREND_LOW_SAMPLE_PA)
    normal_df = work[~low_sample_mask].copy()
    low_df = work[low_sample_mask].copy()

    stat_label = _trend_stat_label(stat_key)
    value_template = _trend_hover_value_template(stat_key)
    hovertemplate = (
        f"{player_label}<br>"
        "Year: %{x}<br>"
        f"{stat_label}: {value_template}<br>"
        "PA/BF: %{customdata[0]}<br>"
        "Pitches: %{customdata[1]}"
        "<extra></extra>"
    )
    marker_symbol = _TREND_MARKER_SYMBOLS.get(stat_key, "circle")
    trace_name = f"{player_label} · {stat_label}"

    if not normal_df.empty:
        fig.add_trace(
            go.Scatter(
                x=normal_df["year"],
                y=normal_df["value"],
                mode="lines+markers",
                name=trace_name,
                line=dict(color=color, width=2, dash=dash),
                marker=dict(color=color, size=8, symbol=marker_symbol),
                customdata=np.column_stack(
                    (
                        _trend_counts_for_hover(normal_df["approx_pa"]),
                        _trend_counts_for_hover(normal_df["n_pitches"]),
                    )
                ),
                hovertemplate=hovertemplate,
                connectgaps=False,
            )
        )

    if not low_df.empty:
        fig.add_trace(
            go.Scatter(
                x=low_df["year"],
                y=low_df["value"],
                mode="markers",
                name=trace_name,
                marker=dict(
                    color=color,
                    size=10,
                    symbol="circle-open",
                    line=dict(color=color, width=2),
                ),
                customdata=np.column_stack(
                    (
                        _trend_counts_for_hover(low_df["approx_pa"]),
                        _trend_counts_for_hover(low_df["n_pitches"]),
                    )
                ),
                hovertemplate=hovertemplate,
                connectgaps=False,
                showlegend=False,
            )
        )

    return not low_df.empty


def _build_single_stat_chart(
    stat_df_a: pd.DataFrame,
    stat_df_b: pd.DataFrame,
    stat_key: str,
    player_label_a: str,
    player_label_b: str | None,
    year_range: tuple[int, int],
    player_type: str,
    height: int = 340,
) -> go.Figure:
    fig = go.Figure()
    has_low_sample = False

    if not stat_df_a.empty:
        has_low_sample = _add_trend_traces(
            fig, stat_df_a, player_label_a, stat_key, _COLOR_TREND_A, "solid", player_type
        ) or has_low_sample
    if not stat_df_b.empty and player_label_b:
        has_low_sample = _add_trend_traces(
            fig, stat_df_b, player_label_b, stat_key, _COLOR_TREND_B, "dash", player_type
        ) or has_low_sample

    years, tick_text = _trend_year_ticks(year_range)
    stat_label = _trend_stat_label(stat_key)
    title_suffix = _trend_title_suffix(year_range)
    if not stat_df_b.empty and player_label_b:
        title_text = f"{player_label_a} vs {player_label_b} — {stat_label} by year ({title_suffix})"
    else:
        title_text = f"{player_label_a} — {stat_label} by year ({title_suffix})"

    fig.update_xaxes(
        tickvals=years,
        ticktext=tick_text,
        tickfont=dict(size=11),
        showgrid=True,
        gridcolor="rgba(128,128,128,0.15)",
        title="Year",
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor="rgba(128,128,128,0.15)",
        tickfont=dict(size=10),
        zeroline=False,
        title=stat_label,
        tickformat=".3f" if _stat_formatter(stat_key) == "decimal_3" else ".1f",
    )
    fig.update_layout(
        template="plotly_dark",
        height=height,
        margin=dict(l=10, r=10, t=20, b=48),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        showlegend=True,
        hovermode="closest",
        title=title_text,
    )
    if has_low_sample:
        fig.add_annotation(
            x=0,
            y=-0.26,
            xref="paper",
            yref="paper",
            showarrow=False,
            text="Open markers = PA < 50 (small sample)",
            font=dict(size=11, color="rgba(255,255,255,0.75)"),
            xanchor="left",
        )
    return fig


def _build_overlay_chart(
    tidy_df_a: pd.DataFrame,
    tidy_df_b: pd.DataFrame,
    stat_keys: list[str],
    player_label_a: str,
    player_label_b: str | None,
    year_range: tuple[int, int],
    player_type: str,
    title_text: str,
    color_mode: str,
    height: int = 340,
) -> go.Figure | None:
    fig = go.Figure()
    has_data = False
    has_low_sample = False
    has_player_b = player_label_b is not None and not tidy_df_b.empty

    for idx, stat_key in enumerate(stat_keys):
        stat_df_a = _trend_stat_df(tidy_df_a, stat_key, year_range)
        stat_df_b = _trend_stat_df(tidy_df_b, stat_key, year_range) if has_player_b else tidy_df_b.iloc[0:0]

        if color_mode == "stat":
            stat_color = _TREND_CUSTOM_COLORS[idx % len(_TREND_CUSTOM_COLORS)]
            color_a = stat_color
            color_b = stat_color
        elif color_mode == "dashboard_metric":
            stat_color = _TREND_DASHBOARD_METRIC_COLORS[idx % len(_TREND_DASHBOARD_METRIC_COLORS)]
            color_a = stat_color
            color_b = stat_color
        else:
            color_a = _COLOR_TREND_A
            color_b = _COLOR_TREND_B

        if not stat_df_a.empty:
            has_data = True
            has_low_sample = _add_trend_traces(
                fig, stat_df_a, player_label_a, stat_key, color_a, "solid", player_type
            ) or has_low_sample
        if has_player_b and not stat_df_b.empty and player_label_b:
            has_data = True
            has_low_sample = _add_trend_traces(
                fig, stat_df_b, player_label_b, stat_key, color_b, "dash", player_type
            ) or has_low_sample

    if not has_data:
        return None

    years, tick_text = _trend_year_ticks(year_range)
    scale_stat = stat_keys[0]
    fig.update_xaxes(
        tickvals=years,
        ticktext=tick_text,
        tickfont=dict(size=11),
        showgrid=True,
        gridcolor="rgba(128,128,128,0.15)",
        title="Year",
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor="rgba(128,128,128,0.15)",
        tickfont=dict(size=10),
        zeroline=False,
        tickformat=".3f" if _stat_formatter(scale_stat) == "decimal_3" else ".1f",
    )
    fig.update_layout(
        template="plotly_dark",
        height=height,
        margin=dict(l=10, r=10, t=20, b=48),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        showlegend=True,
        hovermode="x unified",
        title=title_text,
    )
    if has_low_sample:
        fig.add_annotation(
            x=0,
            y=-0.26,
            xref="paper",
            yref="paper",
            showarrow=False,
            text="Open markers = PA < 50 (small sample)",
            font=dict(size=11, color="rgba(255,255,255,0.75)"),
            xanchor="left",
        )
    return fig


def _render_trend_data_table(
    tidy_df_a: pd.DataFrame,
    tidy_df_b: pd.DataFrame,
    stat_keys: list[str],
    player_label_a: str,
    player_label_b: str | None,
    player_type: str,
) -> None:
    del player_type  # Currently presented as unified "PA/BF" in this table.
    if not stat_keys:
        return

    def _player_parts(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        if df.empty:
            return pd.DataFrame(columns=stat_keys), pd.Series(dtype="Float64")
        subset = df[df["stat_key"].isin(stat_keys)].copy()
        if subset.empty:
            return pd.DataFrame(columns=stat_keys), pd.Series(dtype="Float64")
        subset["year"] = pd.to_numeric(subset["year"], errors="coerce").astype("Int64")
        subset = subset[subset["year"].notna()].copy()
        subset["year"] = subset["year"].astype(int)
        value_pivot = subset.pivot_table(index="year", columns="stat_key", values="value", aggfunc="first")
        value_pivot = value_pivot.reindex(columns=stat_keys)
        pa_by_year = pd.to_numeric(subset.groupby("year", sort=True)["approx_pa"].first(), errors="coerce")
        return value_pivot, pa_by_year

    values_a, pa_a = _player_parts(tidy_df_a)
    has_player_b = player_label_b is not None
    values_b, pa_b = _player_parts(tidy_df_b) if has_player_b else (pd.DataFrame(columns=stat_keys), pd.Series(dtype="Float64"))

    year_index = sorted(
        set(values_a.index.tolist())
        | set(values_b.index.tolist())
        | set(pa_a.index.tolist())
        | set(pa_b.index.tolist())
    )
    if not year_index:
        st.info("No trend data available for selected stat(s).")
        return

    table_df = pd.DataFrame({"year": year_index})
    table_df["year"] = pd.to_numeric(table_df["year"], errors="coerce").astype("Int64")
    column_config: dict[str, st.column_config.Column] = {
        "year": st.column_config.NumberColumn("Year", format="%d"),
    }

    if has_player_b and player_label_b is not None:
        pa_col_a = f"{player_label_a} PA/BF"
        pa_col_b = f"{player_label_b} PA/BF"
        table_df[pa_col_a] = pa_a.reindex(year_index).values
        table_df[pa_col_b] = pa_b.reindex(year_index).values
        table_df[pa_col_a] = pd.to_numeric(table_df[pa_col_a], errors="coerce").astype("Int64")
        table_df[pa_col_b] = pd.to_numeric(table_df[pa_col_b], errors="coerce").astype("Int64")
        column_config[pa_col_a] = st.column_config.NumberColumn(pa_col_a, format="%d")
        column_config[pa_col_b] = st.column_config.NumberColumn(pa_col_b, format="%d")

        for stat_key in stat_keys:
            stat_label = _trend_stat_label(stat_key)
            col_a = f"{player_label_a} {stat_label}"
            col_b = f"{player_label_b} {stat_label}"
            table_df[col_a] = values_a[stat_key].reindex(year_index).values if stat_key in values_a.columns else np.nan
            table_df[col_b] = values_b[stat_key].reindex(year_index).values if stat_key in values_b.columns else np.nan
            stat_fmt = _trend_value_format(stat_key)
            column_config[col_a] = st.column_config.NumberColumn(col_a, format=stat_fmt)
            column_config[col_b] = st.column_config.NumberColumn(col_b, format=stat_fmt)
    else:
        table_df["PA/BF"] = pa_a.reindex(year_index).values
        table_df["PA/BF"] = pd.to_numeric(table_df["PA/BF"], errors="coerce").astype("Int64")
        column_config["PA/BF"] = st.column_config.NumberColumn("PA/BF", format="%d")

        for stat_key in stat_keys:
            stat_label = _trend_stat_label(stat_key)
            table_df[stat_label] = values_a[stat_key].reindex(year_index).values if stat_key in values_a.columns else np.nan
            column_config[stat_label] = st.column_config.NumberColumn(stat_label, format=_trend_value_format(stat_key))

    st.dataframe(
        table_df,
        width="stretch",
        hide_index=True,
        column_config=column_config,
    )


def render_trend_dashboard(
    trend_data_a: list[dict],
    trend_data_b: list[dict] | None,
    player_label_a: str,
    player_label_b: str | None,
    year_range: tuple[int, int],
    player_type: str,
    apply_filters_to_each_year: bool,
    active_filter_summary: str | None,
) -> None:
    _render_trend_filter_caption(apply_filters_to_each_year, active_filter_summary)

    is_pitcher = str(player_type).strip().lower() == "pitcher"
    if is_pitcher:
        dashboard_groups = [
            ["K-BB%", "CSW%"],
            ["Whiff%", "FirstStrike%"],
            ["HardHit%", "Barrel%"],
            ["wOBA", "xwOBA"],
        ]
    else:
        dashboard_groups = [
            ["K%", "BB%"],
            ["HardHit%", "Barrel%"],
            ["wOBA", "xwOBA"],
        ]

    stats_needed: list[str] = []
    for group in dashboard_groups:
        for stat in group:
            if stat not in stats_needed:
                stats_needed.append(stat)

    tidy_df_a = _build_trend_tidy_df(trend_data_a, stats_needed)
    tidy_df_b = _build_trend_tidy_df(trend_data_b or [], stats_needed)

    title_suffix = _trend_title_suffix(year_range)

    def _group_title(stat_keys: list[str]) -> str:
        stat_text = " + ".join(_trend_stat_label(s) for s in stat_keys)
        if player_label_b and not tidy_df_b.empty:
            return f"{player_label_a} vs {player_label_b} — {stat_text} by year ({title_suffix})"
        return f"{player_label_a} — {stat_text} by year ({title_suffix})"

    def _render_group(stat_keys: list[str], height: int = 340) -> None:
        fig = _build_overlay_chart(
            tidy_df_a=tidy_df_a,
            tidy_df_b=tidy_df_b,
            stat_keys=stat_keys,
            player_label_a=player_label_a,
            player_label_b=player_label_b,
            year_range=year_range,
            player_type=player_type,
            title_text=_group_title(stat_keys),
            color_mode="dashboard_metric",
            height=height,
        )
        if fig is None:
            st.info(f"No trend data available for {' + '.join(stat_keys)}.")
            return
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

    row_1 = dashboard_groups[:2]
    col_left, col_right = st.columns(2)
    with col_left:
        _render_group(row_1[0])
    with col_right:
        _render_group(row_1[1])

    for group in dashboard_groups[2:]:
        _render_group(group)


def render_trend_custom(
    trend_data_a: list[dict],
    trend_data_b: list[dict] | None,
    player_label_a: str,
    player_label_b: str | None,
    year_range: tuple[int, int],
    player_type: str,
    available_stats: list[str],
    selected_custom_stats: list[str],
    apply_filters_to_each_year: bool,
    active_filter_summary: str | None,
) -> None:
    _render_trend_filter_caption(apply_filters_to_each_year, active_filter_summary)

    selected = [s for s in selected_custom_stats if s in available_stats]
    if len(selected) > 4:
        st.warning("Select up to 4 stats. Showing the first 4 selections.")
        selected = selected[:4]
    if not selected:
        st.info("Select at least one stat.")
        return

    tidy_df_a = _build_trend_tidy_df(trend_data_a, available_stats)
    tidy_df_b = _build_trend_tidy_df(trend_data_b or [], available_stats)

    if len(selected) == 1:
        stat_key = selected[0]
        stat_df_a = _trend_stat_df(tidy_df_a, stat_key, year_range)
        stat_df_b = _trend_stat_df(tidy_df_b, stat_key, year_range)
        if stat_df_a.empty and stat_df_b.empty:
            st.info("No trend data available for that stat/year range.")
        else:
            fig = _build_single_stat_chart(
                stat_df_a=stat_df_a,
                stat_df_b=stat_df_b,
                stat_key=stat_key,
                player_label_a=player_label_a,
                player_label_b=player_label_b,
                year_range=year_range,
                player_type=player_type,
            )
            st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    elif _stats_share_scale(selected):
        title_suffix = _trend_title_suffix(year_range)
        stat_text = " + ".join(_trend_stat_label(s) for s in selected)
        if player_label_b and not tidy_df_b.empty:
            title_text = f"{player_label_a} vs {player_label_b} — {stat_text} by year ({title_suffix})"
        else:
            title_text = f"{player_label_a} — {stat_text} by year ({title_suffix})"

        fig = _build_overlay_chart(
            tidy_df_a=tidy_df_a,
            tidy_df_b=tidy_df_b,
            stat_keys=selected,
            player_label_a=player_label_a,
            player_label_b=player_label_b,
            year_range=year_range,
            player_type=player_type,
            title_text=title_text,
            color_mode="stat",
            height=340,
        )
        if fig is None:
            st.info("No trend data available for selected stat(s)/year range.")
        else:
            st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    else:
        cols_per_row = min(len(selected), 2)
        for start in range(0, len(selected), cols_per_row):
            row_stats = selected[start:start + cols_per_row]
            cols = st.columns(cols_per_row)
            for idx in range(cols_per_row):
                with cols[idx]:
                    if idx >= len(row_stats):
                        st.empty()
                        continue
                    stat_key = row_stats[idx]
                    stat_df_a = _trend_stat_df(tidy_df_a, stat_key, year_range)
                    stat_df_b = _trend_stat_df(tidy_df_b, stat_key, year_range)
                    if stat_df_a.empty and stat_df_b.empty:
                        st.info(f"No trend data for {stat_key}.")
                        continue
                    fig = _build_single_stat_chart(
                        stat_df_a=stat_df_a,
                        stat_df_b=stat_df_b,
                        stat_key=stat_key,
                        player_label_a=player_label_a,
                        player_label_b=player_label_b,
                        year_range=year_range,
                        player_type=player_type,
                        height=280,
                    )
                    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

    year_start, year_end = int(year_range[0]), int(year_range[1])
    table_df_a = _filter_real_data_rows(
        tidy_df_a[
            (tidy_df_a["year"] >= year_start)
            & (tidy_df_a["year"] <= year_end)
            & (tidy_df_a["stat_key"].isin(selected))
        ].copy()
    )
    table_df_b = _filter_real_data_rows(
        tidy_df_b[
            (tidy_df_b["year"] >= year_start)
            & (tidy_df_b["year"] <= year_end)
            & (tidy_df_b["stat_key"].isin(selected))
        ].copy()
    )
    _render_trend_data_table(
        tidy_df_a=table_df_a,
        tidy_df_b=table_df_b,
        stat_keys=selected,
        player_label_a=player_label_a,
        player_label_b=player_label_b,
        player_type=player_type,
    )


def render_trend_section(
    trend_data_a: list[dict],
    selected_stats: list[str],
    selected_stat: str,
    year_range: tuple[int, int],
    player_label_a: str,
    trend_data_b: list[dict] | None = None,
    player_label_b: str | None = None,
    apply_filters_to_each_year: bool = True,
    active_filter_summary: str | None = None,
) -> None:
    """Render single-stat yearly trend chart + compact table."""
    if not selected_stats:
        st.info("No stats selected for trend chart.")
        return

    tidy_df_a = _build_trend_tidy_df(trend_data_a, selected_stats)
    tidy_df_b = (
        _build_trend_tidy_df(trend_data_b, selected_stats)
        if trend_data_b is not None and player_label_b is not None
        else _build_trend_tidy_df([], selected_stats)
    )
    if tidy_df_a.empty and tidy_df_b.empty:
        st.info("No trend data available.")
        return

    if selected_stat not in selected_stats:
        selected_stat = selected_stats[0]
    stat_label = _trend_stat_label(selected_stat)

    year_start, year_end = year_range
    stat_df_a = tidy_df_a[
        (tidy_df_a["stat_key"] == selected_stat)
        & (tidy_df_a["year"] >= int(year_start))
        & (tidy_df_a["year"] <= int(year_end))
    ].copy()
    stat_df_b = tidy_df_b[
        (tidy_df_b["stat_key"] == selected_stat)
        & (tidy_df_b["year"] >= int(year_start))
        & (tidy_df_b["year"] <= int(year_end))
    ].copy()
    stat_df_a = _filter_real_data_rows(stat_df_a)
    stat_df_b = _filter_real_data_rows(stat_df_b)

    if stat_df_a.empty and stat_df_b.empty:
        st.info("No trend data available for that stat/year range.")
        return

    stat_df_a = stat_df_a.sort_values("year")
    stat_df_b = stat_df_b.sort_values("year")

    if apply_filters_to_each_year and active_filter_summary and "No filters" not in active_filter_summary:
        st.caption(f"Filters applied to trend: {active_filter_summary}")
    elif not apply_filters_to_each_year:
        st.caption("Filters applied to trend: Off (full-season data each year).")

    def _hover_text(df: pd.DataFrame, label: str) -> list[str]:
        hover_rows: list[str] = []
        for _, row in df.iterrows():
            hover_parts = [
                f"{label}",
                f"Year: {int(row['year'])}",
                f"{stat_label}: {format_stat_value(selected_stat, row['value'])}",
                f"N_pitches: {int(row['n_pitches']) if pd.notna(row['n_pitches']) else '—'}",
                f"Approx PA: {int(row['approx_pa']) if pd.notna(row['approx_pa']) else '—'}",
            ]
            if pd.notna(row["n_bip"]):
                hover_parts.append(f"N_BIP: {int(row['n_bip'])}")
            hover_rows.append("<br>".join(hover_parts) + "<extra></extra>")
        return hover_rows

    fig = go.Figure()
    title_suffix = (
        f"{int(year_start)}-{int(year_end)}"
        if int(year_start) != int(year_end)
        else f"{int(year_start)}"
    )
    if not stat_df_b.empty and player_label_b is not None:
        title_text = f"{player_label_a} vs {player_label_b} — {stat_label} by year ({title_suffix})"
    else:
        title_text = f"{player_label_a} — {stat_label} by year ({title_suffix})"

    if not stat_df_a.empty:
        fig.add_trace(
            go.Scatter(
                x=stat_df_a["year"],
                y=stat_df_a["value"],
                mode="lines+markers",
                name=player_label_a,
                line=dict(color=_COLOR_TREND_A, width=2),
                marker=dict(color=_COLOR_TREND_A, size=8),
                hovertext=_hover_text(stat_df_a, player_label_a),
                hoverinfo="text",
            )
        )
    if not stat_df_b.empty and player_label_b is not None:
        fig.add_trace(
            go.Scatter(
                x=stat_df_b["year"],
                y=stat_df_b["value"],
                mode="lines+markers",
                name=player_label_b,
                line=dict(color=_COLOR_TREND_B, width=2),
                marker=dict(color=_COLOR_TREND_B, size=8),
                hovertext=_hover_text(stat_df_b, player_label_b),
                hoverinfo="text",
            )
        )

    years = sorted(
        set(stat_df_a["year"].tolist())
        | set(stat_df_b["year"].tolist())
    )
    tick_text = ["2020*" if int(y) == 2020 else str(int(y)) for y in years]
    fig.update_xaxes(
        tickvals=years,
        ticktext=tick_text,
        tickfont=dict(size=11),
        showgrid=True,
        gridcolor="rgba(128,128,128,0.15)",
        title="Year",
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor="rgba(128,128,128,0.15)",
        tickfont=dict(size=10),
        zeroline=False,
        title=stat_label,
    )
    fig.update_layout(
        template="plotly_dark",
        height=340,
        margin=dict(l=10, r=10, t=20, b=40),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        showlegend=(not stat_df_b.empty),
        hovermode="closest",
        title=title_text,
    )
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

    if not stat_df_b.empty and player_label_b is not None:
        a_table = stat_df_a[["year", "value", "n_pitches", "approx_pa"]].rename(
            columns={
                "value": "A_value",
                "n_pitches": "A_n_pitches",
                "approx_pa": "A_approx_pa",
            }
        )
        b_table = stat_df_b[["year", "value", "n_pitches", "approx_pa"]].rename(
            columns={
                "value": "B_value",
                "n_pitches": "B_n_pitches",
                "approx_pa": "B_approx_pa",
            }
        )
        table_df = pd.merge(a_table, b_table, on="year", how="outer").sort_values("year")
        table_df["diff(A-B)"] = table_df["A_value"] - table_df["B_value"]
        table_df = table_df[
            ["year", "A_value", "B_value", "diff(A-B)", "A_n_pitches", "B_n_pitches", "A_approx_pa", "B_approx_pa"]
        ]
        table_df["year"] = table_df["year"].astype("Int64")
        table_df["A_n_pitches"] = pd.to_numeric(table_df["A_n_pitches"], errors="coerce").astype("Int64")
        table_df["B_n_pitches"] = pd.to_numeric(table_df["B_n_pitches"], errors="coerce").astype("Int64")
        table_df["A_approx_pa"] = pd.to_numeric(table_df["A_approx_pa"], errors="coerce").astype("Int64")
        table_df["B_approx_pa"] = pd.to_numeric(table_df["B_approx_pa"], errors="coerce").astype("Int64")

        if table_df["A_approx_pa"].isna().all() and table_df["B_approx_pa"].isna().all():
            table_df = table_df.drop(columns=["A_approx_pa", "B_approx_pa"])

        column_config: dict[str, st.column_config.Column] = {
            "year": st.column_config.NumberColumn("year", format="%d"),
            "A_value": st.column_config.NumberColumn(f"{player_label_a} {stat_label}", format=_trend_value_format(selected_stat)),
            "B_value": st.column_config.NumberColumn(f"{player_label_b} {stat_label}", format=_trend_value_format(selected_stat)),
            "diff(A-B)": st.column_config.NumberColumn("diff(A-B)", format=_trend_value_format(selected_stat)),
            "A_n_pitches": st.column_config.NumberColumn("A_n_pitches", format="%d"),
            "B_n_pitches": st.column_config.NumberColumn("B_n_pitches", format="%d"),
        }
        if "A_approx_pa" in table_df.columns:
            column_config["A_approx_pa"] = st.column_config.NumberColumn("A_approx_pa", format="%d")
        if "B_approx_pa" in table_df.columns:
            column_config["B_approx_pa"] = st.column_config.NumberColumn("B_approx_pa", format="%d")

        st.dataframe(
            table_df,
            width="stretch",
            hide_index=True,
            column_config=column_config,
        )
    else:
        table_df = stat_df_a[["year", "value", "n_pitches", "n_bip", "approx_pa"]].copy()
        table_df["year"] = table_df["year"].astype("Int64")
        table_df["n_pitches"] = pd.to_numeric(table_df["n_pitches"], errors="coerce").astype("Int64")
        table_df["n_bip"] = pd.to_numeric(table_df["n_bip"], errors="coerce").astype("Int64")
        table_df["approx_pa"] = pd.to_numeric(table_df["approx_pa"], errors="coerce").astype("Int64")

        st.dataframe(
            table_df,
            width="stretch",
            hide_index=True,
            column_config={
                "year": st.column_config.NumberColumn("year", format="%d"),
                "value": st.column_config.NumberColumn(stat_label, format=_trend_value_format(selected_stat)),
                "n_pitches": st.column_config.NumberColumn("n_pitches", format="%d"),
                "n_bip": st.column_config.NumberColumn("n_bip", format="%d"),
                "approx_pa": st.column_config.NumberColumn("approx_pa", format="%d"),
            },
        )

    if 2020 in set(years):
        st.caption("* 2020 was a 60-game season.")


# ---------------------------------------------------------------------------
# Player header
# ---------------------------------------------------------------------------

def player_header(name: str, team: str, season: int, player_type: str) -> None:
    """Render a compact player header (name · team · season · type)."""
    col_text, col_spacer = st.columns([3, 1])
    with col_text:
        st.subheader(name)
        st.caption(f"{team} · {season} · {player_type}")
