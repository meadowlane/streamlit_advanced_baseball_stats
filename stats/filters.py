"""Filter configuration and application for Statcast DataFrames.

SplitFilters holds the optional pre-filter parameters that narrow the raw
pitch-level data before it is grouped by a split function.  All fields are
Optional so callers only specify what they need; None means "no restriction".

apply_filters() is the single entry point that consumes a SplitFilters and
returns a narrowed DataFrame.  It is intentionally orthogonal to get_splits():
call it first, then pass the result to get_splits().

FilterSpec / FILTER_REGISTRY describe the available filter types for the
dynamic filter builder UI (Phase 2).  rows_to_split_filters() converts the
session-state row list produced by that UI into a SplitFilters instance.

Month filtering note
--------------------
``prepare_df()`` derives ``_month`` from ``game_date`` once after fetch/cache.
The month filter uses that derived column.  For backward compatibility,
``apply_filters()`` can still fall back to deriving month from ``game_date``
when ``_month`` is absent.

home_away encoding note
-----------------------
Statcast ``inning_topbot`` is ``"Bot"`` when the home team is batting (bottom
half) and ``"Top"`` when the visiting team is batting (top half).  The
``home_away`` filter maps ``"home"`` → ``"Bot"`` and ``"away"`` → ``"Top"``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

import pandas as pd


# ---------------------------------------------------------------------------
# FilterSpec registry
# ---------------------------------------------------------------------------

@dataclass
class FilterSpec:
    """Metadata for a single filter type used by the dynamic filter builder.

    Attributes
    ----------
    key : str
        Unique identifier; used as the ``"filter_type"`` value in row dicts.
    label : str
        Human-readable name shown in the sidebar UI.
    required_cols : list[str]
        DataFrame columns that must be present for this filter to operate.
    default_params : dict
        Initial ``"params"`` dict used when a new row of this type is created.
    """

    key: str
    label: str
    required_cols: list[str]
    default_params: dict


FILTER_REGISTRY: dict[str, FilterSpec] = {
    "inning": FilterSpec(
        key="inning",
        label="Inning range",
        required_cols=["inning"],
        default_params={"min": 1, "max": 9},
    ),
    "pitcher_hand": FilterSpec(
        key="pitcher_hand",
        label="Pitcher handedness",
        required_cols=["p_throws"],
        default_params={"hand": "R"},
    ),
    "batter_hand": FilterSpec(
        key="batter_hand",
        label="Batter handedness",
        required_cols=["stand"],
        default_params={"hand": "R"},
    ),
    "home_away": FilterSpec(
        key="home_away",
        label="Home / Away",
        required_cols=["inning_topbot"],
        # "Bot" = home team bats (bottom half); "Top" = away team bats.
        default_params={"side": "home"},
    ),
    "month": FilterSpec(
        key="month",
        label="Month",
        required_cols=["game_date"],
        # Month is derived in prepare_df via game_date -> _month.
        default_params={"month": 4},
    ),
    "count": FilterSpec(
        key="count",
        label="Count",
        required_cols=["balls", "strikes"],
        # Either or both of balls/strikes may be None (= any value).
        default_params={"balls": None, "strikes": None},
    ),
}

_MONTH_TO_LABEL = {4: "Apr", 5: "May", 6: "Jun", 7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct"}
_MONTH_COL = "_month"


# ---------------------------------------------------------------------------
# SplitFilters
# ---------------------------------------------------------------------------

@dataclass
class SplitFilters:
    """Parameters used to pre-filter a Statcast DataFrame before split computation.

    All fields default to None (no filter applied).

    Attributes
    ----------
    inning_min : int | None
        Keep only pitches from this inning onward (inclusive).
    inning_max : int | None
        Keep only pitches up to this inning (inclusive).
    pitcher_hand : "L" | "R" | None
        Keep only pitches thrown by a left- or right-handed pitcher.
        Maps to the Statcast ``p_throws`` column.
    batter_hand : "L" | "R" | None
        Keep only pitches to left- or right-handed batters.
        Maps to the Statcast ``stand`` column.
    home_away : "home" | "away" | None
        Keep only pitches from home plate appearances (batter's team is home)
        or away plate appearances.  Maps to ``inning_topbot``: ``"Bot"`` = home
        bats, ``"Top"`` = away bats.
    month : int | None
        Keep only pitches from games in this calendar month (1–12).
        Uses the derived ``_month`` column (created by ``prepare_df``).
    balls : int | None
        Keep only pitches where the ball count equals this value (0–3).
    strikes : int | None
        Keep only pitches where the strike count equals this value (0–2).
    """

    inning_min:   int | None = None
    inning_max:   int | None = None
    pitcher_hand: Literal["L", "R"] | None = None
    batter_hand:  Literal["L", "R"] | None = None
    home_away:    Literal["home", "away"] | None = None
    month:        int | None = None
    balls:        int | None = None
    strikes:      int | None = None


# ---------------------------------------------------------------------------
# rows_to_split_filters
# ---------------------------------------------------------------------------

def rows_to_split_filters(rows: list[dict]) -> SplitFilters:
    """Translate a session-state filter-row list into a SplitFilters instance.

    Each row dict must have the shape::

        {
            "id":          str,   # unique per row; used for widget keys
            "filter_type": str,   # key into FILTER_REGISTRY
            "params":      dict,  # type-specific parameter values
        }

    When multiple rows set the same ``SplitFilters`` field (e.g. two ``"inning"``
    rows), the **last row wins** — its values overwrite any earlier ones.
    Unknown ``filter_type`` values are silently ignored.

    Parameters
    ----------
    rows : list[dict]
        Ordered list of filter-row dicts from ``st.session_state["filter_rows"]``.

    Returns
    -------
    SplitFilters
        Assembled filter configuration ready for ``apply_filters``.
    """
    kwargs: dict = {}
    for row in rows:
        ft = row.get("filter_type")
        if ft is None:
            continue  # skip malformed rows missing the required key
        p  = row.get("params", {})

        if ft == "inning":
            kwargs["inning_min"] = p.get("min")
            kwargs["inning_max"] = p.get("max")
        elif ft == "pitcher_hand":
            kwargs["pitcher_hand"] = p.get("hand")
        elif ft == "batter_hand":
            kwargs["batter_hand"] = p.get("hand")
        elif ft == "home_away":
            kwargs["home_away"] = p.get("side")
        elif ft == "month":
            kwargs["month"] = p.get("month")
        elif ft == "count":
            # Always assign both fields so a later row that resets balls or
            # strikes to None correctly overrides an earlier row's value.
            kwargs["balls"]   = p.get("balls")
            kwargs["strikes"] = p.get("strikes")
        # Unknown filter types are silently ignored.

    return SplitFilters(**kwargs)


def prepare_df(df: pd.DataFrame) -> pd.DataFrame:
    """Return a prepared copy of *df* with normalized dtypes/derived columns.

    - ``game_date`` coerced to datetime (invalid values become NaT)
    - derived ``_month`` column added once from ``game_date``
    - ``inning`` coerced to numeric if present (invalid values become NaN)
    """
    out = df.copy()

    if "game_date" in out.columns:
        out["game_date"] = pd.to_datetime(out["game_date"], errors="coerce")
        out[_MONTH_COL] = out["game_date"].dt.month

    if "inning" in out.columns:
        out["inning"] = pd.to_numeric(out["inning"], errors="coerce")

    return out


def get_prepared_df_cached(
    df: pd.DataFrame,
    cache: dict[tuple[int, int, str], pd.DataFrame],
    cache_key: tuple[int, int, str],
    log_fn: Callable[[str], None] | None = None,
) -> pd.DataFrame:
    """Return a memoized prepared DataFrame keyed by (player_id, season, type)."""
    if cache_key in cache:
        if log_fn is not None:
            log_fn(f"[prepare_df] cache hit: {cache_key}")
        return cache[cache_key]

    if log_fn is not None:
        log_fn(f"[prepare_df] cache miss: {cache_key}")

    prepared = prepare_df(df)
    cache[cache_key] = prepared
    return prepared


def summarize_filter_rows(rows: list[dict]) -> str:
    """Return a compact human-readable summary for dynamic filter rows."""
    if not rows:
        return "No filters (full season data)"

    parts: list[str] = []
    for row in rows:
        ft = row.get("filter_type")
        spec = FILTER_REGISTRY.get(ft)
        if spec is None:
            continue

        p = row.get("params", {})
        label = spec.label

        if ft == "inning":
            inning_min = p.get("min", spec.default_params.get("min", 1))
            inning_max = p.get("max", spec.default_params.get("max", 9))
            parts.append(f"{label}: {inning_min}-{inning_max}")
        elif ft == "pitcher_hand":
            hand = p.get("hand", spec.default_params.get("hand", "R"))
            parts.append(f"{label}: {hand}")
        elif ft == "batter_hand":
            hand = p.get("hand", spec.default_params.get("hand", "R"))
            parts.append(f"{label}: {hand}")
        elif ft == "home_away":
            side = str(p.get("side", spec.default_params.get("side", "home"))).capitalize()
            parts.append(f"{label}: {side}")
        elif ft == "month":
            month_int = p.get("month", spec.default_params.get("month", 4))
            month_label = _MONTH_TO_LABEL.get(month_int, str(month_int))
            parts.append(f"{label}: {month_label}")
        elif ft == "count":
            balls = p.get("balls")
            strikes = p.get("strikes")
            if balls is None and strikes is None:
                value = "Any"
            elif balls is None:
                value = f"strikes {strikes}"
            elif strikes is None:
                value = f"balls {balls}"
            else:
                value = f"{balls}-{strikes}"
            parts.append(f"{label}: {value}")

    return ", ".join(parts) if parts else "No filters (full season data)"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _col_ok(df: pd.DataFrame, col: str, filter_name: str) -> bool:
    """Return True if *col* is present in *df*.

    Missing-column policy (mirrors the existing inning-filter behaviour):

    * Inside a live Streamlit session: emit ``st.warning`` and return ``False``
      so the caller skips this individual filter without crashing the UI.
    * Outside a Streamlit session: raise ``ValueError``.

    Unlike the original single-filter implementation that returned the whole
    DataFrame early, this helper skips only the offending filter so that
    remaining active filters are still applied.
    """
    if col in df.columns:
        return True

    msg = (
        f"apply_filters: '{filter_name}' filter is active but the DataFrame "
        f"has no '{col}' column."
    )
    try:
        import streamlit.runtime  # noqa: PLC0415
        if streamlit.runtime.exists():
            import streamlit as st  # noqa: PLC0415
            st.warning(msg)
            return False
    except Exception:
        pass
    raise ValueError(msg)


# ---------------------------------------------------------------------------
# Filter application
# ---------------------------------------------------------------------------

def apply_filters(
    df: pd.DataFrame,
    filters: SplitFilters,
    pitcher_perspective: bool = False,
) -> pd.DataFrame:
    """Return a copy of *df* narrowed to the rows matching all active filters.

    A filter field is "active" when it is not None.  All active filters are
    AND-combined.  If no fields are active the original DataFrame is returned
    unchanged (no copy, no allocation).

    Month filtering uses the derived ``_month`` column created by
    ``prepare_df``.  For backward compatibility, if ``_month`` is absent but
    ``game_date`` is present, month is derived on the fly.

    Parameters
    ----------
    df : pd.DataFrame
        Raw Statcast pitch-level DataFrame as returned by get_statcast_batter.
    filters : SplitFilters
        Filter configuration.  None fields are skipped.
    pitcher_perspective : bool
        When ``True``, interpret ``home_away`` from the pitcher's perspective:
        ``home`` → ``inning_topbot == "Top"`` and ``away`` → ``"Bot"``.

    Raises
    ------
    ValueError
        When a filter is active but its required column is absent and the call
        is made outside a live Streamlit session.  Inside Streamlit a warning
        is shown and that individual filter is skipped.
    """
    # Fast path — avoid any allocation when nothing is active.
    if not any([
        filters.inning_min   is not None,
        filters.inning_max   is not None,
        filters.pitcher_hand is not None,
        filters.batter_hand  is not None,
        filters.home_away    is not None,
        filters.month        is not None,
        filters.balls        is not None,
        filters.strikes      is not None,
    ]):
        return df

    # --- inning ----------------------------------------------------------
    if filters.inning_min is not None or filters.inning_max is not None:
        if _col_ok(df, "inning", "inning"):
            if filters.inning_min is not None:
                df = df[df["inning"] >= filters.inning_min]
            if filters.inning_max is not None:
                df = df[df["inning"] <= filters.inning_max]

    # --- pitcher handedness ----------------------------------------------
    if filters.pitcher_hand is not None:
        if _col_ok(df, "p_throws", "pitcher_hand"):
            df = df[df["p_throws"] == filters.pitcher_hand]

    # --- batter handedness -----------------------------------------------
    if filters.batter_hand is not None:
        if _col_ok(df, "stand", "batter_hand"):
            df = df[df["stand"] == filters.batter_hand]

    # --- home / away -----------------------------------------------------
    if filters.home_away is not None:
        if _col_ok(df, "inning_topbot", "home_away"):
            if pitcher_perspective:
                topbot = "Top" if filters.home_away == "home" else "Bot"
            else:
                topbot = "Bot" if filters.home_away == "home" else "Top"
            df = df[df["inning_topbot"] == topbot]

    # --- month (prefer precomputed _month; fallback to game_date) --------
    if filters.month is not None:
        if _MONTH_COL in df.columns:
            df = df[df[_MONTH_COL] == filters.month]
        elif _col_ok(df, "game_date", "month"):
            df = df[pd.to_datetime(df["game_date"], errors="coerce").dt.month == filters.month]

    # --- count: balls ----------------------------------------------------
    if filters.balls is not None:
        if _col_ok(df, "balls", "balls"):
            df = df[df["balls"] == filters.balls]

    # --- count: strikes --------------------------------------------------
    if filters.strikes is not None:
        if _col_ok(df, "strikes", "strikes"):
            df = df[df["strikes"] == filters.strikes]

    return df
