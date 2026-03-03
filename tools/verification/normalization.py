"""Value normalization utilities for the stat verification harness.

All external source values must be normalised to the same conventions the app
uses before comparison:

* Percentage stats (K%, BB%, HardHit%, …) → **0-100 scale**
* Rate stats (AVG, OBP, SLG, wOBA, …) → **decimal** (e.g. 0.350)
* Innings pitched (strings like "157.1") → **decimal innings**
  (1 recorded digit after the decimal = thirds: 157.1 → 157.333…)
* Counting stats (HR, BB, …) → **integer**
"""

from __future__ import annotations

import math
import re
import unicodedata
from typing import Any


# ---------------------------------------------------------------------------
# Stats stored as 0-100 in the app (percentage scale)
# ---------------------------------------------------------------------------

_PCT_100_STATS = frozenset(
    [
        "K%",
        "BB%",
        "K-BB%",
        "HardHit%",
        "Barrel%",
        "GB%",
        "FB%",
        "CSW%",
        "Whiff%",
        "FirstStrike%",
        "F-Strike%",
    ]
)

# Sources that publish percentage stats as 0-1 fractions.
# FG stores K%, BB% etc. as 0-1; MLB API and BRef typically return 0-100.
_SOURCES_USE_FRACTION: frozenset[str] = frozenset(["fangraphs"])


def normalize_pct(
    value: float | None,
    *,
    source_name: str = "",
) -> float | None:
    """Convert a percentage value to the 0-100 scale the app uses.

    Heuristic: if the raw value is < 1.5 **and** the source is known to use
    the 0-1 fraction convention, multiply by 100.  Otherwise return as-is.
    """
    if value is None or math.isnan(value):
        return None
    if source_name in _SOURCES_USE_FRACTION and value < 1.5:
        return round(value * 100.0, 4)
    return value


def normalize_ip(ip_value: str | float | None) -> float | None:
    """Convert innings-pitched to a decimal (IP) value.

    Baseball records IP as X.Y where Y ∈ {0, 1, 2} representing full thirds
    of an inning.  For example::

        normalize_ip("157.1")  # → 157.3333…
        normalize_ip("200.2")  # → 200.6666…
        normalize_ip(9)        # → 9.0

    Returns ``None`` for None / unparseable inputs.
    """
    if ip_value is None:
        return None
    if isinstance(ip_value, (int, float)):
        if math.isnan(float(ip_value)):
            return None
        ip_value = str(ip_value)

    ip_value = str(ip_value).strip()
    if not ip_value or ip_value in ("", "nan", "None"):
        return None

    # Accept both "." and "," as decimal separator
    ip_value = ip_value.replace(",", ".")
    match = re.match(r"^(\d+)(?:\.(\d))?$", ip_value)
    if not match:
        try:
            return float(ip_value)  # already a proper decimal
        except ValueError:
            return None

    full_innings = int(match.group(1))
    thirds = int(match.group(2)) if match.group(2) else 0
    if thirds not in (0, 1, 2):
        # Unexpected — treat the digit after the decimal as 10ths
        return full_innings + thirds / 10.0
    return full_innings + thirds / 3.0


def normalize_avg(value: str | float | None) -> float | None:
    """Parse batting average strings like '.322' or '0.322' → float."""
    if value is None:
        return None
    if isinstance(value, float):
        return None if math.isnan(value) else value
    s = str(value).strip()
    if s in ("", "nan", "None", ".---", "---"):
        return None
    try:
        return float(s)
    except ValueError:
        return None


def normalize_count(value: Any) -> int | None:
    """Convert a value to an integer counting stat."""
    if value is None:
        return None
    try:
        f = float(value)
        if math.isnan(f):
            return None
        return int(round(f))
    except (TypeError, ValueError):
        return None


def normalize_float(value: Any) -> float | None:
    """Convert to float, returning None for NaN / unparseable values."""
    if value is None:
        return None
    try:
        f = float(value)
        return None if math.isnan(f) else f
    except (TypeError, ValueError):
        return None


def normalize_player_name(name: str) -> str:
    """Normalise a player name to lowercase ASCII for fuzzy matching.

    Handles:
    - "Judge, Aaron" → "aaron judge"
    - "Shohei Ohtani" → "shohei ohtani"
    - Accented characters: "García" → "garcia"
    """
    # Flip "Last, First" to "First Last"
    if "," in name:
        parts = name.split(",", 1)
        name = f"{parts[1].strip()} {parts[0].strip()}"
    # Normalise accents
    nfkd = unicodedata.normalize("NFKD", name)
    ascii_name = nfkd.encode("ascii", "ignore").decode("ascii")
    return ascii_name.lower().strip()


# ---------------------------------------------------------------------------
# Dispatch normalizer — converts a raw source value to app convention
# ---------------------------------------------------------------------------

# Counting stat keys
_COUNTING_STATS = frozenset(["PA", "H", "HR", "BB", "SO", "HBP", "W", "L"])

# Rate stats stored as decimal in the app
_DECIMAL_STATS = frozenset(
    ["AVG", "OBP", "SLG", "OPS", "wOBA", "xwOBA", "ERA", "FIP", "xFIP", "SIERA", "xERA"]
)


def normalize_stat(
    key: str,
    value: Any,
    source_name: str = "",
) -> float | int | None:
    """Normalize one stat value from a source into app-compatible format.

    Parameters
    ----------
    key:
        Canonical stat key (e.g. ``"K%"``, ``"wOBA"``).
    value:
        Raw value from the source dict (may be str, float, int, None).
    source_name:
        The ``BaseSource.source_name`` of the providing source, used to
        apply source-specific scale corrections (e.g. FG 0-1 fractions).
    """
    if key == "IP":
        return normalize_ip(value)
    if key in _COUNTING_STATS:
        return normalize_count(value)
    if key in _PCT_100_STATS:
        f = normalize_float(value)
        if f is None:
            return None
        return normalize_pct(f, source_name=source_name)
    if key in _DECIMAL_STATS:
        return normalize_avg(value)  # handles string ".322" and floats
    if key == "FBv":
        return normalize_float(value)
    # wRC+, Stuff+, Location+, Pitching+ — round to nearest int
    if key in ("wRC+", "Stuff+", "Location+", "Pitching+"):
        f = normalize_float(value)
        return None if f is None else int(round(f))
    # Default: return as float
    return normalize_float(value)
