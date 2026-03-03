"""JSON fixture persistence for offline / CI test runs.

Fixtures allow the test suite to run without network access.  Each fixture is
a JSON file stored under ``tests/verification_fixtures/{source}/`` keyed by
``{player_type}_{mlbam_id}_{year}.json``.

Workflow
--------
1. **Record** (live fetch):
   ``python -m tools.verify_stats ... --record-fixtures``
   → fetches live data from all sources and writes JSON files.

2. **Replay** (offline / CI):
   ``python -m tools.verify_stats ... --offline``
   or ``pytest tests/test_stat_verification.py`` (uses offline by default)
   → loads from JSON files; raises ``FixtureNotFoundError`` if missing.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Fixture directory — relative to the project root
# ---------------------------------------------------------------------------

# fixtures.py lives at: <project_root>/tools/verification/fixtures.py
# parents[0] = tools/verification/
# parents[1] = tools/
# parents[2] = <project_root>  ← correct
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_FIXTURE_DIR = _PROJECT_ROOT / "tests" / "verification_fixtures"

# Module-level fixture dir; override via set_fixture_root() or --fixture-root CLI flag
FIXTURE_DIR: Path = _DEFAULT_FIXTURE_DIR


def set_fixture_root(path: str | Path) -> None:
    """Override the fixture root directory (e.g. from ``--fixture-root`` CLI flag).

    Must be called before any :func:`fixture_path` / :func:`save_fixture` /
    :func:`load_fixture` calls take effect.
    """
    global FIXTURE_DIR
    FIXTURE_DIR = Path(path).resolve()


class FixtureNotFoundError(FileNotFoundError):
    """Raised when an expected fixture file is missing during offline mode."""


def fixture_path(
    source: str,
    player_type: str,
    mlbam_id: int,
    year: int,
) -> Path:
    """Return the Path for one fixture file.

    Example: ``tests/verification_fixtures/fangraphs/batter_592450_2024.json``
    """
    return FIXTURE_DIR / source / f"{player_type}_{mlbam_id}_{year}.json"


def fixture_exists(
    source: str,
    player_type: str,
    mlbam_id: int,
    year: int,
) -> bool:
    """Return True when the fixture file exists on disk."""
    return fixture_path(source, player_type, mlbam_id, year).is_file()


def save_fixture(
    source: str,
    player_type: str,
    mlbam_id: int,
    year: int,
    data: dict[str, Any],
) -> Path:
    """Serialise *data* to the fixture file, creating parent dirs as needed.

    Parameters
    ----------
    data:
        The normalised stat dict returned by a source adapter.

    Returns the path written.
    """
    path = fixture_path(source, player_type, mlbam_id, year)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, default=_json_serializer)
    return path


def load_fixture(
    source: str,
    player_type: str,
    mlbam_id: int,
    year: int,
) -> dict[str, Any]:
    """Load and return a previously recorded fixture.

    Raises
    ------
    FixtureNotFoundError
        When the fixture file does not exist.
    """
    path = fixture_path(source, player_type, mlbam_id, year)
    if not path.is_file():
        raise FixtureNotFoundError(
            f"Fixture not found: {path}\n"
            f"Run with --record-fixtures to generate it:\n"
            f"  python -m tools.verify_stats --record-fixtures ..."
        )
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def _json_serializer(obj: Any) -> Any:
    """Custom JSON serializer for types not natively supported by json."""
    import math
    if isinstance(obj, float) and math.isnan(obj):
        return None
    if hasattr(obj, "item"):  # numpy scalars
        return obj.item()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")
