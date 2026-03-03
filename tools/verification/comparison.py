"""Tolerance-based stat comparison engine.

Takes our app's stat dict and dicts from 1-3 external sources, applies
per-stat tolerance rules, and returns structured :class:`StatComparison`
objects with PASS / FAIL / WARN / SKIP / NON_VERIFIABLE verdicts.

Verdict rules
-------------
* **PASS**          : |our_val - src_val| ≤ tolerance for ALL available sources.
* **FAIL**          : exceeds tolerance for ANY primary source.
* **WARN**          : 2/3 sources agree but 1 outlier — not FAIL for that source.
* **SKIP**          : stat cannot be compared (None on one or both sides, or
                      below minimum sample threshold).
* **NON_VERIFIABLE**: stat has ``verifiable=False`` in STAT_MAP; always emitted
                      with reason + fallback strategy.

When all three sources disagree with *each other* as well, FAIL is emitted
with a note explaining that a definition mismatch is likely.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Literal

from tools.verification.stat_map import STAT_MAP, StatMapping


Verdict = Literal["PASS", "FAIL", "WARN", "SKIP", "NON_VERIFIABLE"]

# Minimum samples — comparisons skip below these thresholds
MIN_BATTER_PA = 30
MIN_PITCHER_BF = 30
MIN_PITCHER_IP = 5.0


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class StatComparison:
    """Comparison result for one stat for one player-season."""

    stat: str
    our_value: float | int | None
    source_values: dict[str, float | int | None] = field(default_factory=dict)
    abs_diffs: dict[str, float | None] = field(default_factory=dict)
    rel_diffs: dict[str, float | None] = field(default_factory=dict)
    verdict: Verdict = "SKIP"
    note: str | None = None
    non_verifiable_reason: str | None = None
    fallback: str | None = None


# ---------------------------------------------------------------------------
# Core comparison logic
# ---------------------------------------------------------------------------


def _rel_diff(our: float, src: float) -> float | None:
    """Return relative difference as a fraction.  Returns None when src == 0."""
    if src == 0.0:
        return None
    return (our - src) / abs(src)


def compare_stat(
    stat: str,
    our_val: float | int | None,
    source_vals: dict[str, float | int | None],
) -> StatComparison:
    """Build a :class:`StatComparison` for one stat.

    Parameters
    ----------
    stat:
        Canonical stat key (e.g. ``"wOBA"``).
    our_val:
        The value produced by the app pipeline (from ``AppSource``).
    source_vals:
        ``{source_name: value}`` dict from all external sources.
    """
    mapping: StatMapping | None = STAT_MAP.get(stat)

    # --- Non-verifiable ---
    if mapping is not None and not mapping.verifiable:
        return StatComparison(
            stat=stat,
            our_value=our_val,
            source_values=source_vals,
            verdict="NON_VERIFIABLE",
            non_verifiable_reason=mapping.non_verifiable_reason,
            fallback=mapping.fallback,
        )

    # --- Skip if our value is missing ---
    if our_val is None:
        return StatComparison(
            stat=stat,
            our_value=None,
            source_values=source_vals,
            verdict="SKIP",
            note="App value is None — cannot compare",
        )

    # --- Compute per-source diffs ---
    tolerance = STAT_MAP[stat].tolerance if mapping else None
    abs_diffs: dict[str, float | None] = {}
    rel_diffs: dict[str, float | None] = {}
    per_source_pass: dict[str, bool] = {}

    for src_name, src_val in source_vals.items():
        if src_val is None:
            abs_diffs[src_name] = None
            rel_diffs[src_name] = None
            per_source_pass[src_name] = True  # can't fail what we can't measure
            continue
        try:
            a = float(our_val)
            b = float(src_val)
        except (TypeError, ValueError):
            abs_diffs[src_name] = None
            rel_diffs[src_name] = None
            per_source_pass[src_name] = True
            continue

        ad = abs(a - b)
        abs_diffs[src_name] = round(ad, 6)
        rel_diffs[src_name] = _rel_diff(a, b)

        if tolerance is not None:
            result = tolerance.check(a, b)
            per_source_pass[src_name] = result == "PASS"
        else:
            per_source_pass[src_name] = True  # no tolerance defined → skip

    # --- Determine verdict ---
    verdicts = list(per_source_pass.values())
    n_fail = sum(1 for v in verdicts if not v)
    n_total = len(verdicts)

    if n_total == 0:
        verdict: Verdict = "SKIP"
        note: str | None = "No source values available"
    elif n_fail == 0:
        verdict = "PASS"
        note = None
    elif n_fail == n_total and n_total >= 2:
        # All sources fail — check if sources agree with each other
        non_null_src = {k: v for k, v in source_vals.items() if v is not None}
        if _sources_agree_with_each_other(non_null_src, tolerance):
            verdict = "FAIL"
            note = (
                f"All {n_total} sources agree with each other but differ from our value.  "
                "Likely a computation or field-mapping bug in the app pipeline."
            )
        else:
            verdict = "FAIL"
            note = (
                f"All {n_total} sources disagree (sources also disagree with each other).  "
                "Possible definition mismatch, park-factor difference, or mid-season data artifact.  "
                "Investigate each source's denominator and sample cutoff."
            )
    elif n_fail == 1 and n_total >= 3:
        # 2/3 agree → WARN
        failing_src = next(k for k, v in per_source_pass.items() if not v)
        verdict = "WARN"
        note = (
            f"2/{n_total} sources match; outlier: {failing_src!r}.  "
            "May reflect a minor definition difference for this source."
        )
    elif n_fail >= 1:
        verdict = "FAIL"
        note = None

    return StatComparison(
        stat=stat,
        our_value=our_val,
        source_values=source_vals,
        abs_diffs=abs_diffs,
        rel_diffs=rel_diffs,
        verdict=verdict,
        note=note,
    )


def _sources_agree_with_each_other(
    vals: dict[str, Any],
    tolerance: Any,
) -> bool:
    """Return True if all provided source values are within tolerance of each other."""
    values = [v for v in vals.values() if v is not None]
    if len(values) < 2:
        return True
    for i in range(len(values)):
        for j in range(i + 1, len(values)):
            if tolerance is None:
                continue
            result = tolerance.check(float(values[i]), float(values[j]))
            if result == "FAIL":
                return False
    return True


# ---------------------------------------------------------------------------
# Multi-stat batch comparison
# ---------------------------------------------------------------------------


def compare_all_stats(
    our_stats: dict[str, Any],
    source_dicts: dict[str, dict[str, Any]],
    stats_to_check: list[str] | None = None,
    sample_pa: int | None = None,
    sample_ip: float | None = None,
    player_type: str = "batter",
) -> list[StatComparison]:
    """Compare all stats between the app and external sources.

    Parameters
    ----------
    our_stats:
        Full stat dict from ``AppSource``.
    source_dicts:
        ``{source_name: stat_dict}`` for each external source.
    stats_to_check:
        Subset of stat keys to check.  ``None`` means all keys found in
        ``our_stats`` or in any source dict.
    sample_pa:
        Plate appearances (batters) or batters faced (pitchers).
        Used to emit SKIP verdicts for low-sample players.
    sample_ip:
        Innings pitched (decimal).  For the IP threshold check.
    player_type:
        ``"batter"`` or ``"pitcher"``.
    """
    # Determine which stats to check
    all_keys: set[str] = set(our_stats.keys())
    for d in source_dicts.values():
        all_keys.update(d.keys())
    if stats_to_check is not None:
        all_keys = all_keys.intersection(stats_to_check)

    results: list[StatComparison] = []

    for stat in sorted(all_keys):
        # --- Sample-size SKIP ---
        if player_type == "batter":
            if sample_pa is not None and sample_pa < MIN_BATTER_PA:
                results.append(StatComparison(
                    stat=stat,
                    our_value=our_stats.get(stat),
                    verdict="SKIP",
                    note=f"Sample too small: PA={sample_pa} < {MIN_BATTER_PA}",
                ))
                continue
        else:
            if sample_ip is not None and sample_ip < MIN_PITCHER_IP:
                results.append(StatComparison(
                    stat=stat,
                    our_value=our_stats.get(stat),
                    verdict="SKIP",
                    note=f"Sample too small: IP={sample_ip:.1f} < {MIN_PITCHER_IP}",
                ))
                continue

        our_val = our_stats.get(stat)
        source_vals = {name: d.get(stat) for name, d in source_dicts.items()}

        results.append(compare_stat(stat, our_val, source_vals))

    return results
