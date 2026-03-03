"""Abstract base class and shared types for all verification source adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

#: Valid game-type scope identifiers.
VALID_SCOPES = frozenset(["regular", "postseason", "all"])


@dataclass
class PlayerIdentity:
    """Holds cross-source IDs for a single player.

    Attributes
    ----------
    name:
        Display name ("Aaron Judge").
    mlbam_id:
        MLBAM (MLB Advanced Media) integer ID — used by Statcast and the
        official MLB Stats API.
    fg_id:
        FanGraphs integer player ID (IDfg column).
    bref_id:
        Baseball-Reference slug, e.g. ``"judgea01"``.  Optional; used only
        by the BRef source adapter.
    """

    name: str
    mlbam_id: int
    fg_id: int | None = None
    bref_id: str | None = None


class SourceError(Exception):
    """Raised when a source adapter cannot fetch or parse data."""


class BaseSource(ABC):
    """Common interface every verification source adapter must implement.

    Adapters must be safe to call in offline mode when ``offline=True`` is
    passed to :meth:`get_batter_season` / :meth:`get_pitcher_season`.  In
    that mode they should raise :class:`SourceError` rather than making any
    network requests.

    Return values
    -------------
    Both ``get_batter_season`` and ``get_pitcher_season`` return a plain
    ``dict[str, Any]`` keyed by the **canonical stat names** used throughout
    the harness (see :mod:`tools.verification.stat_map`).  Missing stats are
    simply omitted from the dict — callers must handle ``KeyError`` /
    ``dict.get`` defensively.
    """

    @property
    @abstractmethod
    def source_name(self) -> str:
        """Short, unique identifier for this source, e.g. ``"fangraphs"``."""

    @property
    def supported_scopes(self) -> frozenset[str]:
        """Game-type scopes this source can produce data for.

        Defaults to regular-season only.  Override in sources that support
        ``"postseason"`` and ``"all"`` (i.e., event-based sources that can
        filter their raw data by ``game_type``).
        """
        return frozenset(["regular"])

    @abstractmethod
    def get_batter_season(
        self,
        player: PlayerIdentity,
        year: int,
        *,
        game_type: str = "regular",
        offline: bool = False,
    ) -> dict[str, Any]:
        """Return a stat dict for one batter's full season.

        Parameters
        ----------
        game_type:
            ``"regular"`` (default), ``"postseason"``, or ``"all"``.
            Raise :class:`SourceError` if the requested scope is not in
            :attr:`supported_scopes`.

        Values must already be normalised to the conventions used by the app:
        - K%, BB%, HardHit%, Barrel%, GB% etc. in **0-100 scale**
        - wOBA, xwOBA, AVG, OBP, SLG, OPS in **decimal** (e.g. 0.350)
        - Counting stats (HR, BB, SO, PA) as **integers**
        """

    @abstractmethod
    def get_pitcher_season(
        self,
        player: PlayerIdentity,
        year: int,
        *,
        game_type: str = "regular",
        offline: bool = False,
    ) -> dict[str, Any]:
        """Return a stat dict for one pitcher's full season.

        Same normalisation conventions as :meth:`get_batter_season`.
        """
