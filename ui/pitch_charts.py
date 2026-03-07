"""Pitch location and movement chart helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go  # type: ignore[import-untyped]
import streamlit as st

from stats.filters import SplitFilters

ZONE_X_BOUNDS = (-0.8333, 0.8333)
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
    "HBP": "star",
    "Other": "diamond",
}
_OUTCOME_ORDER = [
    "Ball",
    "Called Strike",
    "Swinging Strike",
    "Foul",
    "In Play",
    "HBP",
    "Other",
]
_OUTCOME_LEGEND_HTML = (
    '<p style="font-size:12px;color:rgba(255,255,255,0.60);margin:2px 0 6px 0;">'
    "○ Ball &nbsp;·&nbsp; ● Called Strike &nbsp;·&nbsp; ✕ Swinging Strike"
    " &nbsp;·&nbsp; ▲ Foul &nbsp;·&nbsp; ■ In Play &nbsp;·&nbsp; ★ HBP"
    "</p>"
)


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
        return (
            np.zeros((bins, bins)),
            np.linspace(x_min, x_max, bins + 1),
            np.linspace(z_bot, z_top, bins + 1),
        )

    in_bounds = df[
        (df["plate_x"] >= x_min)
        & (df["plate_x"] <= x_max)
        & (df["plate_z"] >= z_bot)
        & (df["plate_z"] <= z_top)
    ].copy()
    if in_bounds.empty:
        return (
            np.zeros((bins, bins)),
            np.linspace(x_min, x_max, bins + 1),
            np.linspace(z_bot, z_top, bins + 1),
        )

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
        ordered = df.sort_values(
            by=chosen_cols, ascending=True, kind="mergesort"
        ).copy()
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
    if desc == "hit_by_pitch":
        return "HBP"
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
        work = work[
            work["pitch_type"].astype(str).isin([str(p) for p in pitch_types])
        ].copy()

    if "pitch_type" in work.columns:
        work["pitch_type"] = work["pitch_type"].fillna("UNK").astype(str)
    else:
        work["pitch_type"] = "UNK"
    if "description" in work.columns:
        work["outcome_bucket"] = work["description"].fillna("").map(_bucket_outcome)
    else:
        work["outcome_bucket"] = "Other"
    work["outcome_bucket"] = work["outcome_bucket"].astype(str).str.strip()
    work["bs_bucket"] = work["outcome_bucket"].map(_ball_strike_bucket)
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

        pitch_color = _PITCH_TYPE_COLORS.get(
            pitch_type, fallback_map.get(pitch_type, "#9aa0a6")
        )
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

        custom_cols.append(
            pitch_df["bs_bucket"].fillna("Unknown").astype(str).to_numpy()
        )
        custom_cols.append(
            pitch_df["outcome_bucket"].fillna("Other").astype(str).to_numpy()
        )

        if "release_speed" in pitch_df.columns:
            velo = pd.to_numeric(pitch_df["release_speed"], errors="coerce")
            custom_cols.append(
                np.where(velo.notna(), np.round(velo, 1).astype(str), "—")
            )
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
                balls.astype("Int64").astype(str)
                + "-"
                + strikes.astype("Int64").astype(str),
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
        if (
            __debug__
            and encode_outcomes
            and (pitch_df["outcome_bucket"] == "Ball").any()
        ):
            assert "circle-open" in marker_symbols
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
                    line=dict(width=1.5, color=pitch_color),
                ),
                customdata=customdata,
                hovertemplate="<br>".join(hover_lines) + "<extra></extra>",
            )
        )

    _add_zone_shapes(
        fig, x_min, x_max, z_bot, z_top, grid=grid, draw_grid=bool(show_grid_overlay)
    )

    if view_mode == "Zoom to zone":
        x_range = [x_min - 0.6, x_max + 0.6]
        y_range = [z_bot - 0.4, z_top + 0.4]
    else:
        x_data_min = float(work["plate_x"].min())
        x_data_max = float(work["plate_x"].max())
        y_data_min = float(work["plate_z"].min())
        y_data_max = float(work["plate_z"].max())
        x_range = [
            min(x_data_min - 0.3, x_min - 0.25),
            max(x_data_max + 0.3, x_max + 0.25),
        ]
        y_range = [
            min(y_data_min - 0.3, z_bot - 0.3),
            max(y_data_max + 0.3, z_top + 0.3),
        ]

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
        work = work[
            work["pitch_type"].astype(str).isin([str(p) for p in pitch_types])
        ].copy()

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
        work["outcome_bucket"] = work["description"].fillna("").map(_bucket_outcome)
    else:
        work["outcome_bucket"] = "Other"
    work["outcome_bucket"] = work["outcome_bucket"].astype(str).str.strip()
    work["bs_bucket"] = work["outcome_bucket"].map(_ball_strike_bucket)

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

        custom_cols.append(
            trace_df["bs_bucket"].fillna("Unknown").astype(str).to_numpy()
        )
        custom_cols.append(
            trace_df["outcome_bucket"].fillna("Other").astype(str).to_numpy()
        )

        if "release_speed" in trace_df.columns:
            velo = pd.to_numeric(trace_df["release_speed"], errors="coerce")
            custom_cols.append(
                np.where(velo.notna(), np.round(velo, 1).astype(str), "—")
            )
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
                    color=_PITCH_TYPE_COLORS.get(
                        pitch_type, fallback_map.get(pitch_type, "#9aa0a6")
                    ),
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


def _pitch_zone_hand_conflict_note(
    role: str,
    active_filters: SplitFilters | None,
    hand_filter: str,
) -> str | None:
    """Return a warning when local hand filters contradict sidebar filters."""
    if active_filters is None or hand_filter == "All":
        return None

    if role == "batter":
        if active_filters.pitcher_hand is None:
            return None
        sidebar_label = "vs LHP" if active_filters.pitcher_hand == "L" else "vs RHP"
        filter_label = "Pitcher hand"
    else:
        if active_filters.batter_hand is None:
            return None
        sidebar_label = "vs LHB" if active_filters.batter_hand == "L" else "vs RHB"
        filter_label = "Batter hand"

    if hand_filter == sidebar_label:
        return None
    return (
        f"Pitch Location {filter_label} filter conflicts with the sidebar "
        f"{filter_label.lower()} filter ({sidebar_label}). Clear one filter to see data."
    )


def render_pitch_zone_chart(
    df: pd.DataFrame,
    role: str = "pitcher",
    active_filters: SplitFilters | None = None,
) -> None:
    """Render interactive pitch-location chart with strike-zone overlay."""
    with st.expander("Pitch Location", expanded=False):
        if df.empty:
            st.info("No pitch data available for this selection.")
            return

        is_batter_role = role == "batter"
        viz_df = df.copy()
        has_stand = "stand" in df.columns
        if "pitch_type" in viz_df.columns:
            pitch_type_options = sorted(
                viz_df["pitch_type"].dropna().astype(str).unique().tolist()
            )
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
            if is_batter_role:
                hand_caption = "Pitcher hand"
                hand_options = ["All", "vs LHP", "vs RHP"]
                hand_key = "pitch_zone_pitcher_hand"
            else:
                hand_caption = "Batter hand"
                hand_options = ["All", "vs LHB", "vs RHB"]
                hand_key = "pitch_zone_batter_hand"
            st.caption(hand_caption)
            hand_filter = st.radio(
                hand_caption,
                options=hand_options,
                horizontal=True,
                key=hand_key,
                label_visibility="collapsed",
            )

        prev_preset = st.session_state.get("_pitch_zone_prev_preset")
        if prev_preset != pitch_preset:
            st.session_state["_pitch_zone_prev_preset"] = pitch_preset
            implied = PITCH_PRESETS.get(pitch_preset)
            if implied is not None:
                st.session_state["pitch_zone_pitch_types_advanced"] = [
                    pitch_type
                    for pitch_type in implied
                    if pitch_type in pitch_type_options
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
            if (not is_batter_role) and has_stand and hand_filter != "All":
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
            show_heatmap = st.checkbox(
                "Show heatmap", value=False, key="pitch_zone_show_heatmap"
            )
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
                st.session_state["_pitch_movement_prev_perspective"] = (
                    movement_perspective
                )

            vertical_axis_mode = st.radio(
                "Vertical axis",
                options=["Induced (Statcast)", "Break direction (down = more drop)"],
                horizontal=True,
                key="pitch_movement_vertical_axis",
            )
            has_throws = (
                "p_throws" in viz_df.columns and viz_df["p_throws"].notna().any()
            )
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
                    for pitch_type in st.session_state[
                        "pitch_zone_pitch_types_advanced"
                    ]
                    if pitch_type in pitch_type_options
                ]
            selected_pitch_types = st.multiselect(
                "Pitch type",
                options=pitch_type_options,
                key="pitch_zone_pitch_types_advanced",
            )

        conflict_note = _pitch_zone_hand_conflict_note(role, active_filters, hand_filter)
        if conflict_note:
            st.warning(conflict_note)
            return

        filtered_df = viz_df.copy()

        if is_batter_role:
            has_pitcher_throws = "p_throws" in filtered_df.columns
            if has_pitcher_throws:
                throws_vals = filtered_df["p_throws"].fillna("").astype(str).str.upper()
                if hand_filter == "vs LHP":
                    filtered_df = filtered_df[throws_vals == "L"].copy()
                elif hand_filter == "vs RHP":
                    filtered_df = filtered_df[throws_vals == "R"].copy()
        else:
            if has_stand:
                stand_vals = filtered_df["stand"].fillna("").astype(str).str.upper()
                if hand_filter == "vs LHB":
                    keep = {"L", "S"} if include_switch else {"L"}
                    filtered_df = filtered_df[stand_vals.isin(keep)].copy()
                elif hand_filter == "vs RHB":
                    keep = {"R", "S"} if include_switch else {"R"}
                    filtered_df = filtered_df[stand_vals.isin(keep)].copy()

        advanced_override = (
            bool(pitch_type_options)
            and set(selected_pitch_types) != set(pitch_type_options)
            and "pitch_type" in filtered_df.columns
        )
        if advanced_override:
            filtered_df = filtered_df[
                filtered_df["pitch_type"]
                .astype(str)
                .isin([str(p) for p in selected_pitch_types])
            ].copy()
        else:
            preset_types = PITCH_PRESETS[pitch_preset]
            if preset_types is not None and "pitch_type" in filtered_df.columns:
                available_types = set(
                    filtered_df["pitch_type"].dropna().astype(str).unique().tolist()
                )
                preset_match = sorted(available_types.intersection(set(preset_types)))
                if preset_match:
                    filtered_df = filtered_df[
                        filtered_df["pitch_type"].astype(str).isin(preset_match)
                    ].copy()

        tab_zone, tab_movement = st.tabs(["Zone Map", "Movement"])
        with tab_zone:
            if encode_outcomes:
                st.markdown(_OUTCOME_LEGEND_HTML, unsafe_allow_html=True)

            if show_heatmap and max_pitches < 300:
                st.caption(
                    "ℹ Heatmap is most meaningful with ≥ 300 pitches. Increase Max pitches in Options."
                )

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
                        selected_outcomes=selected_outcomes
                        if encode_outcomes
                        else None,
                        show_heatmap=show_heatmap,
                        heatmap_bins=heatmap_bins,
                        heatmap_normalize=normalize_heatmap,
                        heatmap_show_scale=heatmap_show_scale,
                        show_grid_overlay=show_grid_overlay,
                        view_mode=view_mode,
                    )
                    st.plotly_chart(
                        fig, width="stretch", config={"displayModeBar": False}
                    )

        with tab_movement:
            render_pitch_movement_chart(
                df=filtered_df,
                max_pitches=max_pitches,
                pitch_types=None,
                perspective=movement_perspective,
                mirror_lhp=mirror_lhp,
                vertical_axis_mode=vertical_axis_mode,
            )
