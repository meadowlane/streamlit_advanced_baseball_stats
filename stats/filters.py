"""Filter configuration and application for Statcast DataFrames.

SplitFilters holds the optional pre-filter parameters that narrow the raw
pitch-level data before it is grouped by a split function.  All fields are
Optional so callers only specify what they need; None means "no restriction".

apply_filters() is the single entry point that consumes a SplitFilters and
returns a narrowed DataFrame.  It is intentionally orthogonal to get_splits():
call it first, then pass the result to get_splits().
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


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
    """

    inning_min: int | None = None
    inning_max: int | None = None


# ---------------------------------------------------------------------------
# Filter application
# ---------------------------------------------------------------------------

def apply_filters(df: pd.DataFrame, filters: SplitFilters) -> pd.DataFrame:
    """Return a copy of *df* narrowed to the rows matching all active filters.

    A filter field is "active" when it is not None.  If no fields are active
    the original DataFrame is returned unchanged (no copy, no allocation).

    Parameters
    ----------
    df : pd.DataFrame
        Raw Statcast pitch-level DataFrame as returned by get_statcast_batter.
    filters : SplitFilters
        Filter configuration.  None fields are skipped.

    Raises
    ------
    ValueError
        When an inning filter is active but ``df`` has no ``"inning"`` column
        and the call is made outside a live Streamlit session.  Inside a
        Streamlit session a warning is shown instead so the UI stays responsive.
    """
    inning_active = filters.inning_min is not None or filters.inning_max is not None

    if not inning_active:
        return df  # fast path — no allocation

    if "inning" not in df.columns:
        msg = (
            "apply_filters: an inning filter is set but the DataFrame has no "
            "'inning' column.  Ensure 'inning' is present in STATCAST_KEEP_COLS."
        )
        try:
            import streamlit.runtime
            if streamlit.runtime.exists():
                import streamlit as st
                st.warning(msg)
                return df
        except Exception:
            pass
        raise ValueError(msg)

    if filters.inning_min is not None:
        df = df[df["inning"] >= filters.inning_min]
    if filters.inning_max is not None:
        df = df[df["inning"] <= filters.inning_max]
    return df
