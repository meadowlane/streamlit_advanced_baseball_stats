"""Stat mapping table: canonical definitions, tolerances, and verifiability.

Every stat the app displays has one ``StatMapping`` entry.  The mapping drives:
- which sources can verify the stat (``verifiable_sources``)
- how close values must be to PASS (``tolerance``)
- whether the stat is "fully verifiable" (≥3 independent sources)
- fallback strategy when verification is impossible

Tolerance conventions
---------------------
* ``"exact"``     — counting stats (HR, BB, SO, …); any diff → FAIL
* ``"abs"``       — absolute difference ≤ ``value`` → PASS
* ``"warn_only"`` — provider-specific stat; emit NON_VERIFIABLE, never FAIL
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


# ---------------------------------------------------------------------------
# Tolerance dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StatTolerance:
    """Defines the acceptable discrepancy for one stat."""

    kind: Literal["exact", "abs", "warn_only"]
    value: float = 0.0  # used only for kind="abs"

    def check(
        self,
        our_val: float | None,
        src_val: float | None,
    ) -> Literal["PASS", "FAIL", "WARN"]:
        """Return a simple two-value verdict (PASS / FAIL).

        Callers apply the multi-source WARN logic on top of individual
        check results — see :func:`tools.verification.comparison.compare_stat`.
        """
        if our_val is None or src_val is None:
            return "PASS"  # cannot compare → skip, not fail
        if self.kind == "warn_only":
            return "PASS"  # provider-specific; classified as NON_VERIFIABLE upstream
        if self.kind == "exact":
            return "PASS" if our_val == src_val else "FAIL"
        # abs
        return "PASS" if abs(our_val - src_val) <= self.value else "FAIL"


# ---------------------------------------------------------------------------
# StatMapping dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StatMapping:
    """Full description of one displayed stat for the verification harness."""

    key: str
    canonical_def: str
    # Sources that *can* supply this stat.  Order matters: first = primary.
    verifiable_sources: list[str]
    tolerance: StatTolerance
    # True when ≥3 independent external sources are available.
    verifiable: bool
    # For Statcast/FG-proprietary stats.
    non_verifiable_reason: str | None = None
    # What to verify instead (inputs, recompute from formula, …).
    fallback: str | None = None
    # Notes on known definition differences between providers.
    provider_notes: str | None = None


# ---------------------------------------------------------------------------
# Tolerance table
# ---------------------------------------------------------------------------

TOLERANCES: dict[str, StatTolerance] = {
    # ----- Counting stats (exact) -------------------------------------------
    "PA": StatTolerance("exact"),
    "H": StatTolerance("exact"),
    "HR": StatTolerance("exact"),
    "BB": StatTolerance("exact"),
    "SO": StatTolerance("exact"),
    "HBP": StatTolerance("exact"),
    "W": StatTolerance("exact"),
    "L": StatTolerance("exact"),
    # ----- Traditional rate stats (3 dp) ------------------------------------
    "AVG": StatTolerance("abs", 0.001),
    "OBP": StatTolerance("abs", 0.001),
    "SLG": StatTolerance("abs", 0.001),
    "OPS": StatTolerance("abs", 0.002),
    # ----- Advanced rate stats (3 dp) ---------------------------------------
    "wOBA": StatTolerance("abs", 0.003),
    "xwOBA": StatTolerance("abs", 0.003),
    # ----- Pitcher ERA-like stats -------------------------------------------
    "ERA": StatTolerance("abs", 0.01),
    "FIP": StatTolerance("abs", 0.02),
    "xERA": StatTolerance("abs", 0.05),
    # ----- Percentage stats (0-100 scale) -----------------------------------
    "K%": StatTolerance("abs", 0.5),
    "BB%": StatTolerance("abs", 0.5),
    "K-BB%": StatTolerance("abs", 1.0),
    "HardHit%": StatTolerance("abs", 1.0),
    "Barrel%": StatTolerance("abs", 0.5),
    "GB%": StatTolerance("abs", 1.0),
    "FB%": StatTolerance("abs", 1.0),
    "CSW%": StatTolerance("abs", 0.5),
    "Whiff%": StatTolerance("abs", 1.0),
    "FirstStrike%": StatTolerance("abs", 1.0),
    # ----- Index stats (integer) --------------------------------------------
    "wRC+": StatTolerance("abs", 2.0),
    # ----- Velocity (mph) ---------------------------------------------------
    "FBv": StatTolerance("abs", 0.2),
    # ----- Provider-specific (never FAIL) -----------------------------------
    "xFIP": StatTolerance("warn_only"),
    "SIERA": StatTolerance("warn_only"),
    "Stuff+": StatTolerance("warn_only"),
    "Location+": StatTolerance("warn_only"),
    "Pitching+": StatTolerance("warn_only"),
}

# ---------------------------------------------------------------------------
# Full stat map
# ---------------------------------------------------------------------------

# Source name identifiers (must match BaseSource.source_name implementations)
_APP = "app"
_FG = "fangraphs"
_SC = "statcast"
_MLB = "mlb_api"
_BREF = "baseball_ref"

STAT_MAP: dict[str, StatMapping] = {
    # ------------------------------------------------------------------
    # Traditional counting stats
    # ------------------------------------------------------------------
    "PA": StatMapping(
        key="PA",
        canonical_def="Plate appearances",
        verifiable_sources=[_FG, _SC, _MLB, _BREF],
        tolerance=TOLERANCES["PA"],
        verifiable=True,
    ),
    "H": StatMapping(
        key="H",
        canonical_def="Hits",
        verifiable_sources=[_FG, _MLB, _BREF],
        tolerance=TOLERANCES["H"],
        verifiable=True,
    ),
    "HR": StatMapping(
        key="HR",
        canonical_def="Home runs",
        verifiable_sources=[_FG, _MLB, _BREF],
        tolerance=TOLERANCES["HR"],
        verifiable=True,
    ),
    "BB": StatMapping(
        key="BB",
        canonical_def="Walks (bases on balls)",
        verifiable_sources=[_FG, _SC, _MLB, _BREF],
        tolerance=TOLERANCES["BB"],
        verifiable=True,
    ),
    "SO": StatMapping(
        key="SO",
        canonical_def="Strikeouts",
        verifiable_sources=[_FG, _SC, _MLB, _BREF],
        tolerance=TOLERANCES["SO"],
        verifiable=True,
    ),
    "HBP": StatMapping(
        key="HBP",
        canonical_def="Hit by pitch",
        verifiable_sources=[_FG, _MLB, _BREF],
        tolerance=TOLERANCES["HBP"],
        verifiable=True,
    ),
    "W": StatMapping(
        key="W",
        canonical_def="Wins (pitcher)",
        verifiable_sources=[_FG, _MLB, _BREF],
        tolerance=TOLERANCES["W"],
        verifiable=True,
    ),
    "L": StatMapping(
        key="L",
        canonical_def="Losses (pitcher)",
        verifiable_sources=[_FG, _MLB, _BREF],
        tolerance=TOLERANCES["L"],
        verifiable=True,
    ),
    # ------------------------------------------------------------------
    # Traditional rate stats
    # ------------------------------------------------------------------
    "AVG": StatMapping(
        key="AVG",
        canonical_def="Batting average (H / AB)",
        verifiable_sources=[_FG, _MLB, _BREF],
        tolerance=TOLERANCES["AVG"],
        verifiable=True,
    ),
    "OBP": StatMapping(
        key="OBP",
        canonical_def="On-base percentage ((H+BB+HBP) / (AB+BB+HBP+SF))",
        verifiable_sources=[_FG, _MLB, _BREF],
        tolerance=TOLERANCES["OBP"],
        verifiable=True,
    ),
    "SLG": StatMapping(
        key="SLG",
        canonical_def="Slugging percentage (TB / AB)",
        verifiable_sources=[_FG, _MLB, _BREF],
        tolerance=TOLERANCES["SLG"],
        verifiable=True,
    ),
    "OPS": StatMapping(
        key="OPS",
        canonical_def="On-base plus slugging (OBP + SLG)",
        verifiable_sources=[_FG, _MLB, _BREF],
        tolerance=TOLERANCES["OPS"],
        verifiable=True,
    ),
    # ------------------------------------------------------------------
    # Advanced rate stats
    # ------------------------------------------------------------------
    "wOBA": StatMapping(
        key="wOBA",
        canonical_def=(
            "Weighted on-base average — linear weights applied to PA outcomes. "
            "App uses 2024 FanGraphs constants: walk=0.690, HBP=0.722, "
            "1B=0.888, 2B=1.271, 3B=1.616, HR=2.101."
        ),
        verifiable_sources=[_FG, _SC, _BREF],
        tolerance=TOLERANCES["wOBA"],
        verifiable=True,
        provider_notes=(
            "FG publishes wOBA computed from Statcast.  Our in-app wOBA also uses "
            "Statcast raw data with the same weights.  BRef wOBA uses slightly "
            "different season constants and may differ by ±0.005."
        ),
    ),
    "xwOBA": StatMapping(
        key="xwOBA",
        canonical_def=(
            "Expected wOBA — mean of Baseball Savant's estimated_woba_using_speedangle. "
            "Note: the app computes this as the mean over all PA rows with a non-null "
            "xwOBA value (batted balls only); FanGraphs uses a run-expectancy model "
            "that includes strikeouts and walks."
        ),
        verifiable_sources=[_FG, _SC],
        tolerance=TOLERANCES["xwOBA"],
        verifiable=False,
        non_verifiable_reason=(
            "Only 2 independent sources: FanGraphs (same Statcast feed) and our own "
            "Statcast recompute.  Definition differs: FG includes xwOBA for all PA; "
            "app averages only over batted-ball rows."
        ),
        fallback=(
            "Compare FG xwOBA to app xwOBA with wider tolerance (±0.010).  "
            "Flag definitional difference in report."
        ),
    ),
    # ------------------------------------------------------------------
    # Percentage stats (0-100 scale in app)
    # ------------------------------------------------------------------
    "K%": StatMapping(
        key="K%",
        canonical_def="Strikeout rate: strikeouts / plate appearances × 100",
        verifiable_sources=[_FG, _SC, _MLB, _BREF],
        tolerance=TOLERANCES["K%"],
        verifiable=True,
        provider_notes="FG stores as 0-1 fraction; normalised to 0-100 before compare.",
    ),
    "BB%": StatMapping(
        key="BB%",
        canonical_def="Walk rate: walks / plate appearances × 100",
        verifiable_sources=[_FG, _SC, _MLB, _BREF],
        tolerance=TOLERANCES["BB%"],
        verifiable=True,
        provider_notes="FG stores as 0-1 fraction; normalised to 0-100 before compare.",
    ),
    "K-BB%": StatMapping(
        key="K-BB%",
        canonical_def="K% minus BB% (pitcher efficiency metric)",
        verifiable_sources=[_FG, _SC, _BREF],
        tolerance=TOLERANCES["K-BB%"],
        verifiable=True,
    ),
    "HardHit%": StatMapping(
        key="HardHit%",
        canonical_def="Hard-hit rate: batted balls with exit velocity ≥ 95 mph / total BIP × 100",
        verifiable_sources=[_FG, _SC],
        tolerance=TOLERANCES["HardHit%"],
        verifiable=False,
        non_verifiable_reason=(
            "Only FanGraphs and Statcast recompute available.  The 95 mph threshold "
            "and BIP definition are Statcast-proprietary."
        ),
        fallback="Compare FG HardHit% column to app value.  Check BIP count matches.",
    ),
    "Barrel%": StatMapping(
        key="Barrel%",
        canonical_def=(
            "Barrel rate: events with launch_speed_angle == 6 / total BIP × 100.  "
            "Barrel classification is Statcast-proprietary (speed+angle combo)."
        ),
        verifiable_sources=[_FG, _SC],
        tolerance=TOLERANCES["Barrel%"],
        verifiable=False,
        non_verifiable_reason=(
            "Only FanGraphs and Statcast recompute available.  "
            "Barrel is a Statcast-proprietary classification."
        ),
        fallback="Compare FG Barrel% column to app value.",
    ),
    "GB%": StatMapping(
        key="GB%",
        canonical_def="Ground-ball rate: ground balls / total BIP × 100",
        verifiable_sources=[_FG, _SC, _BREF],
        tolerance=TOLERANCES["GB%"],
        verifiable=True,
        provider_notes=(
            "BRef GB% uses the same Statcast classification but may differ slightly "
            "due to timing of data publication."
        ),
    ),
    "FB%": StatMapping(
        key="FB%",
        canonical_def="Fly-ball rate: fly balls / total BIP × 100",
        verifiable_sources=[_FG, _SC, _BREF],
        tolerance=TOLERANCES["FB%"],
        verifiable=True,
    ),
    # ------------------------------------------------------------------
    # Pitcher ERA-like / run-prevention stats
    # ------------------------------------------------------------------
    "ERA": StatMapping(
        key="ERA",
        canonical_def="Earned run average: (ER × 9) / IP",
        verifiable_sources=[_FG, _MLB, _BREF],
        tolerance=TOLERANCES["ERA"],
        verifiable=True,
    ),
    "FIP": StatMapping(
        key="FIP",
        canonical_def=(
            "Fielding independent pitching: (13×HR + 3×BB − 2×SO) / IP + FIP_constant.  "
            "FIP constant varies by season (typically 3.1–3.2)."
        ),
        verifiable_sources=[_FG, _BREF],
        tolerance=TOLERANCES["FIP"],
        verifiable=False,
        non_verifiable_reason=(
            "Only FanGraphs and Baseball Reference provide FIP.  "
            "MLB Stats API does not.  The FIP constant differs slightly between "
            "providers because they may use different league-wide ER environments."
        ),
        fallback=(
            "Verify inputs (HR, BB, SO, IP) across 3 sources.  "
            "Recompute FIP with FG's published season constant and compare."
        ),
    ),
    "xFIP": StatMapping(
        key="xFIP",
        canonical_def=(
            "Expected FIP: substitutes actual HR with lgHR/FB × pitcher FB count.  "
            "FanGraphs-proprietary — uses their park-factor-adjusted HR rate."
        ),
        verifiable_sources=[_FG],
        tolerance=TOLERANCES["xFIP"],
        verifiable=False,
        non_verifiable_reason="Only FanGraphs publishes xFIP.  Formula requires FG's internal lgHR rate.",
        fallback=(
            "Verify inputs (BB, SO, IP, FB%) across multiple sources.  "
            "Cannot independently recompute xFIP without FG's HR/FB constants."
        ),
    ),
    "SIERA": StatMapping(
        key="SIERA",
        canonical_def=(
            "Skill-interactive ERA — FanGraphs-proprietary regression formula using "
            "K%, BB%, ground-ball rate, and non-linear interaction terms."
        ),
        verifiable_sources=[_FG],
        tolerance=TOLERANCES["SIERA"],
        verifiable=False,
        non_verifiable_reason="Only FanGraphs publishes SIERA.  Formula is proprietary.",
        fallback="Verify inputs (K%, BB%, GB%) across multiple sources.",
    ),
    "xERA": StatMapping(
        key="xERA",
        canonical_def=(
            "Expected ERA based on xwOBA — Statcast/Baseball Savant metric "
            "that converts xwOBA against into an ERA scale."
        ),
        verifiable_sources=[_FG, _SC],
        tolerance=TOLERANCES["xERA"],
        verifiable=False,
        non_verifiable_reason=(
            "Only FanGraphs (passthrough from Statcast) and our Statcast recompute "
            "available.  BRef and MLB API do not publish xERA."
        ),
        fallback="Compare FG xERA passthrough to app value.",
    ),
    # ------------------------------------------------------------------
    # Pitch-level metrics
    # ------------------------------------------------------------------
    "CSW%": StatMapping(
        key="CSW%",
        canonical_def=(
            "Called strikes + whiffs per pitch: "
            "(called_strike + swinging_strike + swinging_strike_blocked + foul_tip) "
            "/ total_pitches × 100"
        ),
        verifiable_sources=[_FG, _SC],
        tolerance=TOLERANCES["CSW%"],
        verifiable=False,
        non_verifiable_reason=(
            "Only FanGraphs (same Statcast source) and our Statcast recompute.  "
            "MLB Stats API and BRef do not publish CSW%."
        ),
        fallback="Compare FG CSW% to app recompute from raw Statcast.",
    ),
    "Whiff%": StatMapping(
        key="Whiff%",
        canonical_def="Swing-and-miss rate: swinging strikes / total swings × 100",
        verifiable_sources=[_FG, _SC],
        tolerance=TOLERANCES["Whiff%"],
        verifiable=False,
        non_verifiable_reason="Only FanGraphs and Statcast recompute available.",
        fallback="Compare FG Whiff% to app recompute.",
        provider_notes=(
            "FanGraphs Whiff% denominator = swings (foul + whiff + in-play).  "
            "Confirm description-set matches SWING_DESCRIPTIONS in stats/splits.py."
        ),
    ),
    "FirstStrike%": StatMapping(
        key="FirstStrike%",
        canonical_def=(
            "First-pitch strike rate: CSW-eligible descriptions on 0-0 count / "
            "total 0-0 pitches × 100"
        ),
        verifiable_sources=[_FG, _SC],
        tolerance=TOLERANCES["FirstStrike%"],
        verifiable=False,
        non_verifiable_reason="Only FanGraphs (F-Strike%) and Statcast recompute available.",
        fallback="Compare FG F-Strike% column to app value.",
        provider_notes="FG column is 'F-Strike%'; map to FirstStrike% key before compare.",
    ),
    "FBv": StatMapping(
        key="FBv",
        canonical_def="Mean four-seam / fastball velocity (mph) over the season",
        verifiable_sources=[_FG, _SC],
        tolerance=TOLERANCES["FBv"],
        verifiable=False,
        non_verifiable_reason="Only FanGraphs and Statcast pitch-data recompute available.",
        fallback="Compare FG FBv to mean release_speed on FF/FA pitch types in Statcast.",
    ),
    # ------------------------------------------------------------------
    # Index / proprietary stats
    # ------------------------------------------------------------------
    "wRC+": StatMapping(
        key="wRC+",
        canonical_def=(
            "Weighted runs created plus — park and league adjusted offensive value.  "
            "100 = average; formula uses FG's wOBA scale, league wOBA, park factors, "
            "and run environment.  FanGraphs-defined."
        ),
        verifiable_sources=[_FG],
        tolerance=TOLERANCES["wRC+"],
        verifiable=False,
        non_verifiable_reason=(
            "wRC+ is a FanGraphs-proprietary metric.  Park factors and wOBA scale "
            "are FG-specific.  No other provider publishes wRC+."
        ),
        fallback=(
            "Verify inputs (PA, BB, HBP, 1B, 2B, 3B, HR) across 3 sources.  "
            "If inputs match, the wRC+ discrepancy is due to park factors or "
            "league wOBA differences — classify as WARN not FAIL."
        ),
    ),
    "Stuff+": StatMapping(
        key="Stuff+",
        canonical_def=(
            "FanGraphs Stuff+ — proprietary model rating pitch quality based on "
            "velocity, movement, and release.  100 = average."
        ),
        verifiable_sources=[_FG],
        tolerance=TOLERANCES["Stuff+"],
        verifiable=False,
        non_verifiable_reason="Proprietary FanGraphs model; no other provider publishes this.",
        fallback="Verify that the FG value matches what the app fetches (passthrough check).",
    ),
    "Location+": StatMapping(
        key="Location+",
        canonical_def="FanGraphs Location+ — proprietary model for pitch location quality.",
        verifiable_sources=[_FG],
        tolerance=TOLERANCES["Location+"],
        verifiable=False,
        non_verifiable_reason="Proprietary FanGraphs model.",
        fallback="Verify passthrough from FG.",
    ),
    "Pitching+": StatMapping(
        key="Pitching+",
        canonical_def=(
            "FanGraphs Pitching+ — composite of Stuff+ and Location+.  "
            "Proprietary model.  100 = average."
        ),
        verifiable_sources=[_FG],
        tolerance=TOLERANCES["Pitching+"],
        verifiable=False,
        non_verifiable_reason="Proprietary FanGraphs model.",
        fallback="Verify passthrough from FG.",
    ),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def get_verifiable_stats(player_type: str) -> list[str]:
    """Return stat keys for which ``verifiable=True`` for the given player type."""
    batter_verifiable = {
        "PA", "H", "HR", "BB", "SO", "HBP",
        "AVG", "OBP", "SLG", "OPS",
        "wOBA", "K%", "BB%", "GB%",
    }
    pitcher_verifiable = {
        "W", "L", "SO", "BB", "HR",
        "ERA", "K%", "BB%", "K-BB%", "GB%",
    }
    if player_type.lower() == "pitcher":
        keys = pitcher_verifiable
    else:
        keys = batter_verifiable
    return [k for k in STAT_MAP if STAT_MAP[k].verifiable and k in keys]


def get_partial_stats(player_type: str) -> list[str]:
    """Return stat keys that are partially verifiable (2 sources, not 3)."""
    if player_type.lower() == "pitcher":
        return ["FIP", "xERA", "CSW%", "Whiff%", "FirstStrike%", "FBv", "xwOBA"]
    return ["xwOBA", "HardHit%", "Barrel%", "xERA"]


def get_non_verifiable_stats() -> list[str]:
    """Return stat keys with ``verifiable=False`` that have only 1 source."""
    return [k for k, v in STAT_MAP.items() if not v.verifiable]
