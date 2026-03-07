"""Game type scope filtering for the stat verification harness.

Statcast pitch-level data includes a ``game_type`` column that identifies
each game as regular season, postseason, spring training, or exhibition.
Without explicit filtering, stats computed from Statcast rows will include
postseason and spring games — inflating PA, BF, and all rate denominators
relative to what FanGraphs, MLB API, and Baseball Reference report (which
are regular-season only).

Codes
-----
R  — Regular season
F  — Wild Card / First Round
D  — Division Series (ALDS / NLDS)
L  — League Championship Series (ALCS / NLCS)
W  — World Series
C  — Championship (historical / some eras)
P  — Generic Playoff (historical data in some seasons)
S  — Spring Training
E  — Exhibition
A  — All-Star Game

Source capability matrix
------------------------
- FanGraphs, MLB API, Baseball Reference: regular season only
- App / Statcast event data: can be filtered to any scope

Usage
-----
::

    from tools.verification.game_scope import filter_by_scope, pa_breakdown_by_game_type

    sc_df = _fetch_statcast_batter(player.mlbam_id, year)
    breakdown = pa_breakdown_by_game_type(sc_df)   # before filtering
    sc_df = filter_by_scope(sc_df, "regular")       # then filter
    computed = _compute_stats(sc_df, player_type="Batter")
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

# ---------------------------------------------------------------------------
# Game type code sets
# ---------------------------------------------------------------------------

#: Regular season only.
REGULAR_GAME_TYPES: frozenset[str] = frozenset(["R"])

#: All postseason codes.  Detailed series codes vary by era / league.
POSTSEASON_GAME_TYPES: frozenset[str] = frozenset(["F", "D", "L", "W", "C", "P"])

#: Spring training and exhibition (not counted in any standard season).
SPRING_GAME_TYPES: frozenset[str] = frozenset(["S", "E", "A"])

#: Every known game type — union of the three sets above.
ALL_GAME_TYPES: frozenset[str] = (
    REGULAR_GAME_TYPES | POSTSEASON_GAME_TYPES | SPRING_GAME_TYPES
)

# ---------------------------------------------------------------------------
# Source capability matrix
# ---------------------------------------------------------------------------

#: Scopes supported by each verification source.
#: Sources that don't support the requested scope are skipped (not failed).
SOURCE_SCOPE_SUPPORT: dict[str, frozenset[str]] = {
    "app": frozenset(["regular", "postseason", "all"]),
    "statcast": frozenset(["regular", "postseason", "all"]),
    "fangraphs": frozenset(["regular"]),
    "mlb_api": frozenset(["regular"]),
    "baseball_ref": frozenset(["regular"]),
}

# ---------------------------------------------------------------------------
# Filter utility
# ---------------------------------------------------------------------------


def filter_by_scope(df: "pd.DataFrame", scope: str) -> "pd.DataFrame":
    """Return only rows matching the requested game-type scope.

    Parameters
    ----------
    df:
        Raw Statcast event DataFrame.  Must contain a ``game_type`` column.
        If the column is absent the DataFrame is returned unchanged — the
        caller is responsible for noting this limitation.
    scope:
        ``"regular"``, ``"postseason"``, or ``"all"``.

    Returns
    -------
    Filtered DataFrame (copy).  ``"all"`` returns a copy of the full input.
    """
    if "game_type" not in df.columns:
        return df.copy()

    if scope == "regular":
        return df[df["game_type"].isin(REGULAR_GAME_TYPES)].copy()
    elif scope == "postseason":
        return df[df["game_type"].isin(POSTSEASON_GAME_TYPES)].copy()
    else:  # "all"
        return df.copy()


def pa_breakdown_by_game_type(df: "pd.DataFrame") -> dict[str, int]:
    """Return PA-outcome event counts grouped by ``game_type``.

    Counts rows where ``events`` is a plate-appearance event (as defined by
    ``stats.splits.PA_EVENTS``), grouped by ``game_type``.  Returns an empty
    dict when either column is missing from *df*.

    Typical output for a batter who played in the postseason::

        {"R": 713, "F": 15, "D": 22, "L": 64}

    Call this on the **unfiltered** DataFrame (before :func:`filter_by_scope`)
    so the breakdown shows what was excluded.
    """
    try:
        from stats.splits import PA_EVENTS  # type: ignore[import-untyped]
    except ImportError:
        return {}

    if "game_type" not in df.columns or "events" not in df.columns:
        return {}

    pa_mask = df["events"].notna() & df["events"].isin(PA_EVENTS)
    pa_df = df[pa_mask]
    counts = pa_df["game_type"].value_counts()
    return {str(k): int(v) for k, v in counts.items()}
