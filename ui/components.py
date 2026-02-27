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
_SPLIT_TABLE_FORMAT: dict[str, st.column_config.Column] = {
    "PA":        st.column_config.NumberColumn("PA",       format="%d"),
    "wOBA":      st.column_config.NumberColumn("wOBA",     format="%.3f"),
    "xwOBA":     st.column_config.NumberColumn("xwOBA",    format="%.3f"),
    "K%":        st.column_config.NumberColumn("K%",       format="%.1f%%"),
    "BB%":       st.column_config.NumberColumn("BB%",      format="%.1f%%"),
    "HardHit%":  st.column_config.NumberColumn("HardHit%", format="%.1f%%"),
    "Barrel%":   st.column_config.NumberColumn("Barrel%",  format="%.1f%%"),
    "GB%":       st.column_config.NumberColumn("GB%",      format="%.1f%%"),
    "K-BB%":     st.column_config.NumberColumn("K-BB%",    format="%.1f%%"),
    "CSW%":      st.column_config.NumberColumn("CSW%",     format="%.1f%%"),
    "Whiff%":    st.column_config.NumberColumn("Whiff%",   format="%.1f%%"),
    "FirstStrike%": st.column_config.NumberColumn("FirstStrike%", format="%.1f%%"),
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
    if arsenal_df.empty:
        st.info("No pitch arsenal data available for this selection.")
        return

    st.dataframe(
        arsenal_df,
        width="stretch",
        hide_index=True,
        column_config={
            "Pitch": st.column_config.TextColumn("Pitch"),
            "N": st.column_config.NumberColumn("N", format="%d"),
            "Usage%": st.column_config.NumberColumn("Usage%", format="%.1f%%"),
            "Velo": st.column_config.NumberColumn("Velo", format="%.1f"),
            "Spin": st.column_config.NumberColumn("Spin", format="%d"),
            "CSW%": st.column_config.NumberColumn("CSW%", format="%.1f%%"),
            "Whiff%": st.column_config.NumberColumn("Whiff%", format="%.1f%%"),
        },
    )


# ---------------------------------------------------------------------------
# Trend line chart
# ---------------------------------------------------------------------------

_COLOR_TREND_A = "#4FC3F7"
_COLOR_TREND_B = "#FF8A65"
_TREND_TIDY_COLUMNS = ["year", "stat_key", "value", "n_pitches", "approx_pa", "n_bip"]

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


def _filter_real_data_rows(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    n_pitches = pd.to_numeric(df["n_pitches"], errors="coerce")
    value = pd.to_numeric(df["value"], errors="coerce")
    return df[(n_pitches > 0) & value.notna()].copy()


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
