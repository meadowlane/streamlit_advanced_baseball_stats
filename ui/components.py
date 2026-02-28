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
