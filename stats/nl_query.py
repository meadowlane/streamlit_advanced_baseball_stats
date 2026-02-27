"""Rule-based natural-language query parser for app session state updates.

This module is intentionally pure (no Streamlit calls) so parser behavior is
unit-testable and predictable.
"""

from __future__ import annotations

from difflib import SequenceMatcher
import re
from typing import Any, Sequence

from stats.filters import FILTER_REGISTRY

# Ordered stat keys used across app UI.
CORE_STATS: list[str] = ["wOBA", "xwOBA", "K%", "BB%", "HardHit%", "Barrel%"]

_YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")
_COMPARISON_HINT_RE = re.compile(r"\b(compare|vs|versus|against)\b", re.IGNORECASE)
_COMPARE_AND_RE = re.compile(r"^\s*compare\s+(?P<a>.+?)\s+and\s+(?P<b>.+?)\s*$", re.IGNORECASE)
_VS_RE = re.compile(r"^\s*(?P<a>.+?)\s+(?:vs|versus|against)\s+(?P<b>.+?)\s*$", re.IGNORECASE)
_GLOBAL_SEASON_MODIFIER_RES = [
    re.compile(r"\b(?:in|for)\s+((?:19|20)\d{2})\b", re.IGNORECASE),
    re.compile(r"\bseason\s+((?:19|20)\d{2})\b", re.IGNORECASE),
    re.compile(r"\b((?:19|20)\d{2})\b", re.IGNORECASE),
]
_TRAILING_FRAGMENT_STOPWORDS = {"in", "season", "year", "the", "a", "an", "of", "trend", "to"}

_TREND_KEYWORDS_RE = re.compile(r"\b(trend|career\s+arc)\b", re.IGNORECASE)
_LAST_N_YEARS_RE = re.compile(
    r"\b(?:over\s+(?:the\s+)?)?last\s+(\d+)\s+(?:years?|seasons?)\b",
    re.IGNORECASE,
)
_YEAR_RANGE_RE = re.compile(
    r"\b((?:19|20)\d{2})\s*(?:[-\u2013]|to)\s*((?:19|20)\d{2})\b",
    re.IGNORECASE,
)
_TREND_MODIFIER_STRIP_RES = [
    re.compile(r"\bcareer\s+arc\b", re.IGNORECASE),
    re.compile(r"\btrend\b", re.IGNORECASE),
    re.compile(r"\b(?:over\s+(?:the\s+)?)?last\s+\d+\s+(?:years?|seasons?)\b", re.IGNORECASE),
]

_MONTH_NAME_TO_INT = {
    "january": 1,
    "jan": 1,
    "february": 2,
    "feb": 2,
    "march": 3,
    "mar": 3,
    "april": 4,
    "apr": 4,
    "may": 5,
    "june": 6,
    "jun": 6,
    "july": 7,
    "jul": 7,
    "august": 8,
    "aug": 8,
    "september": 9,
    "sep": 9,
    "sept": 9,
    "october": 10,
    "oct": 10,
    "november": 11,
    "nov": 11,
    "december": 12,
    "dec": 12,
}
_MONTH_RE = re.compile(
    r"\b("
    + "|".join(sorted(_MONTH_NAME_TO_INT.keys(), key=len, reverse=True))
    + r")\b",
    re.IGNORECASE,
)

_LEFT_HAND_RE = re.compile(
    r"\b(?:vs|versus|against)\s+"
    r"(?:lhp|left(?:[-\s]?hand(?:ed|ers?)?|ies?))\b",
    re.IGNORECASE,
)
_RIGHT_HAND_RE = re.compile(
    r"\b(?:vs|versus|against)\s+"
    r"(?:rhp|right(?:[-\s]?hand(?:ed|ers?)?|ies?))\b",
    re.IGNORECASE,
)

_INNING_RANGE_RE = re.compile(r"\binnings?\s*([1-9])\s*[-–]\s*([1-9])\b", re.IGNORECASE)
_INNING_THROUGH_RE = re.compile(
    r"\b([1-9])(?:st|nd|rd|th)\s+through\s+([1-9])(?:st|nd|rd|th)\b",
    re.IGNORECASE,
)
_INNING_PLUS_RES = [
    re.compile(r"\binnings?\s*([1-9])\+", re.IGNORECASE),
    re.compile(r"\b([1-9])(?:st|nd|rd|th)\s+inning\+", re.IGNORECASE),
    re.compile(r"\binning\s*>=\s*([1-9])\b", re.IGNORECASE),
    re.compile(r"\b([1-9])(?:st|nd|rd|th)\s+inning\s+and\s+later\b", re.IGNORECASE),
]

_HOME_RE = re.compile(r"\b(?:at home|home(?!\s*run))\b", re.IGNORECASE)
_AWAY_RE = re.compile(r"\b(?:on the road|away)\b", re.IGNORECASE)

_COUNT_PAIR_RE = re.compile(r"\b([0-3])\s*-\s*([0-2])\s*count\b", re.IGNORECASE)
_FULL_COUNT_RE = re.compile(r"\bfull\s+count\b", re.IGNORECASE)
_BALLS_ANY_STRIKES_RE = re.compile(r"\b([0-3])\s+balls?\s+any\s+strikes?\b", re.IGNORECASE)
_ANY_BALLS_RE = re.compile(r"\bany\s+balls?\b", re.IGNORECASE)
_ANY_STRIKES_RE = re.compile(r"\bany\s+strikes?\b", re.IGNORECASE)
_BALLS_RE = re.compile(r"\b([0-3])\s+balls?\b", re.IGNORECASE)
_STRIKES_RE = re.compile(r"\b([0-2])\s+strikes?\b", re.IGNORECASE)

_STAT_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("xwOBA", re.compile(r"\bxwoba\b", re.IGNORECASE)),
    ("wOBA", re.compile(r"\bwoba\b", re.IGNORECASE)),
    ("K%", re.compile(r"\bk%\b|\bstrikeout rate\b|\bk rate\b", re.IGNORECASE)),
    ("BB%", re.compile(r"\bbb%\b|\bwalk rate\b|\bbb rate\b", re.IGNORECASE)),
    (
        "HardHit%",
        re.compile(r"\bhard[-\s]?hit%?\b|\bhard[-\s]?hit rate\b", re.IGNORECASE),
    ),
    ("Barrel%", re.compile(r"\bbarrel%?\b|\bbarrel rate\b", re.IGNORECASE)),
]

_FILLER_PREFIX_RE = re.compile(
    r"^\s*(?:show|find|give me|stats for|stat line for|for|please)\s+",
    re.IGNORECASE,
)


def extract_last_year(text: str) -> int | None:
    """Return the last 4-digit year in *text*, or None."""
    matches = list(_YEAR_RE.finditer(text or ""))
    if not matches:
        return None
    return int(matches[-1].group(0))


def parse_nl_query(
    query: str,
    player_names: Sequence[str],
    *,
    valid_seasons: set[int] | None = None,
    allowed_stats: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Parse a free-text query into app-ready intent fields.

    Output keys:
        raw_query, cleaned_query, player_a_fragment, player_b_fragment,
        player_a, player_b, comparison_mode, season_a, season_b,
        link_seasons, season, selected_stats, filter_rows, warnings
    """
    text = (query or "").strip()
    warnings: list[str] = []
    parsed_years = [int(m.group(0)) for m in _YEAR_RE.finditer(text)]

    cleaned_query = _remove_global_modifiers(text)

    spans: list[tuple[int, int]] = []
    stats_allowed = list(allowed_stats) if allowed_stats is not None else CORE_STATS
    selected_stats, stat_spans = _parse_stats(text, stats_allowed)
    spans.extend(stat_spans)

    filter_values, filter_positions, filter_spans, filter_warnings = _parse_filters(text)
    warnings.extend(filter_warnings)
    spans.extend(filter_spans)

    player_text = _strip_spans(text, spans)
    player_text = _clean_player_fragment(player_text)

    comparison_mode = False
    player_a_fragment: str | None = None
    player_b_fragment: str | None = None
    player_a: str | None = None
    player_b: str | None = None
    season_a: int | None = None
    season_b: int | None = None
    link_seasons = True

    comp_a_frag, comp_b_frag = _extract_comparison_fragments(player_text)
    comparison_intent = bool(comp_a_frag and comp_b_frag) or bool(_COMPARISON_HINT_RE.search(player_text))
    if comparison_intent and comp_a_frag is not None and comp_b_frag is not None:
        comparison_mode = True

        frag_year_a, cleaned_frag_a = _extract_fragment_season(comp_a_frag, valid_seasons, warnings)
        frag_year_b, cleaned_frag_b = _extract_fragment_season(comp_b_frag, valid_seasons, warnings)
        player_a_fragment = cleaned_frag_a
        player_b_fragment = cleaned_frag_b

        resolved_a, warn_a, _ = _resolve_player_name(cleaned_frag_a, player_names)
        resolved_b, warn_b, _ = _resolve_player_name(cleaned_frag_b, player_names)

        if resolved_a is None:
            warnings.append(
                f"Could not resolve Player A from '{cleaned_frag_a}' (try full name)."
            )
        elif warn_a:
            warnings.append(warn_a)

        if resolved_b is None:
            warnings.append(
                f"Could not resolve Player B from '{cleaned_frag_b}' (try full name)."
            )
        elif warn_b:
            warnings.append(warn_b)

        player_a = resolved_a
        player_b = resolved_b

        if frag_year_a is not None and frag_year_b is not None:
            season_a = frag_year_a
            season_b = frag_year_b
            link_seasons = season_a == season_b
        elif frag_year_a is not None or frag_year_b is not None:
            same_year = frag_year_a if frag_year_a is not None else frag_year_b
            season_a = same_year
            season_b = same_year
            link_seasons = True
        elif len(parsed_years) >= 2:
            season_a = _normalize_parsed_season(parsed_years[0], valid_seasons, warnings)
            season_b = _normalize_parsed_season(parsed_years[1], valid_seasons, warnings)
            if season_a is not None and season_b is not None:
                link_seasons = season_a == season_b
            elif season_a is not None or season_b is not None:
                same_year = season_a if season_a is not None else season_b
                season_a = same_year
                season_b = same_year
                link_seasons = True
        elif len(parsed_years) == 1:
            single_year = _normalize_parsed_season(parsed_years[0], valid_seasons, warnings)
            season_a = single_year
            season_b = single_year
            link_seasons = True

        if resolved_a is not None and resolved_b is not None and resolved_a == resolved_b:
            warnings.append("Comparison players resolved to the same player; pick two distinct names.")
        elif resolved_a is None and resolved_b is None:
            warnings.append(
                "Could not parse both comparison players; interpreted as a single-player query."
            )
            comparison_mode = False
            player_b = None
            player_b_fragment = None
            player_a_fragment = player_text if player_text else None
            single_year, cleaned_single_frag = _extract_fragment_season(
                player_a_fragment or "",
                valid_seasons,
                warnings,
            )
            player_a_fragment = cleaned_single_frag
            player_a, warn_single, _ = _resolve_player_name(cleaned_single_frag, player_names)
            if warn_single:
                warnings.append(warn_single)
            if single_year is not None:
                season_a = single_year
                season_b = single_year
                link_seasons = True
            else:
                season_a = None
                season_b = None
                link_seasons = True
    elif comparison_intent:
        warnings.append(
            "Could not parse both comparison players; interpreted as a single-player query."
        )
        player_a_fragment = player_text if player_text else None
        single_year, cleaned_single_frag = _extract_fragment_season(
            player_a_fragment or "",
            valid_seasons,
            warnings,
        )
        player_a_fragment = cleaned_single_frag
        player_a, warn_single, _ = _resolve_player_name(cleaned_single_frag, player_names)
        if warn_single:
            warnings.append(warn_single)
        if single_year is not None:
            season_a = single_year
            season_b = single_year
            link_seasons = True
    else:
        player_a_fragment = player_text if player_text else None
        single_year, cleaned_single_frag = _extract_fragment_season(
            player_a_fragment or "",
            valid_seasons,
            warnings,
        )
        player_a_fragment = cleaned_single_frag
        player_a, warn_single, _ = _resolve_player_name(cleaned_single_frag, player_names)
        if warn_single:
            warnings.append(warn_single)
        if single_year is not None:
            season_a = single_year
            season_b = single_year
            link_seasons = True

    if season_a is None and season_b is None and parsed_years:
        fallback_year = _normalize_parsed_season(parsed_years[-1], valid_seasons, warnings)
        season_a = fallback_year
        season_b = fallback_year
        link_seasons = True

    filter_rows = _build_filter_rows(filter_values, filter_positions, warnings)
    season = season_a

    return {
        "raw_query": text,
        "cleaned_query": cleaned_query,
        "player_a_fragment": player_a_fragment,
        "player_b_fragment": player_b_fragment,
        "player_a": player_a,
        "player_b": player_b,
        "comparison_mode": comparison_mode,
        "season_a": season_a,
        "season_b": season_b,
        "link_seasons": link_seasons,
        "season": season,
        "selected_stats": selected_stats,
        "filter_rows": filter_rows,
        "warnings": warnings,
    }


def _parse_stats(text: str, allowed_stats: Sequence[str]) -> tuple[list[str], list[tuple[int, int]]]:
    spans: list[tuple[int, int]] = []
    found: set[str] = set()
    allowed_set = set(allowed_stats)
    for stat, pattern in _STAT_PATTERNS:
        if stat not in allowed_set:
            continue
        for m in pattern.finditer(text):
            found.add(stat)
            spans.append((m.start(), m.end()))
    selected = [stat for stat in allowed_stats if stat in found]
    return selected, spans


def _parse_filters(
    text: str,
) -> tuple[dict[str, dict[str, Any]], dict[str, int], list[tuple[int, int]], list[str]]:
    filters: dict[str, dict[str, Any]] = {}
    positions: dict[str, int] = {}
    spans: list[tuple[int, int]] = []
    warnings: list[str] = []

    def _set_filter(filter_type: str, params: dict[str, Any], start: int, end: int) -> None:
        filters[filter_type] = params
        positions[filter_type] = start
        spans.append((start, end))

    for m in _LEFT_HAND_RE.finditer(text):
        _set_filter("pitcher_hand", {"hand": "L"}, m.start(), m.end())
    for m in _RIGHT_HAND_RE.finditer(text):
        _set_filter("pitcher_hand", {"hand": "R"}, m.start(), m.end())

    for m in _INNING_RANGE_RE.finditer(text):
        _set_filter(
            "inning",
            {"min": int(m.group(1)), "max": int(m.group(2))},
            m.start(),
            m.end(),
        )
    for m in _INNING_THROUGH_RE.finditer(text):
        _set_filter(
            "inning",
            {"min": int(m.group(1)), "max": int(m.group(2))},
            m.start(),
            m.end(),
        )
    for pattern in _INNING_PLUS_RES:
        for m in pattern.finditer(text):
            inning_min = int(m.group(1))
            _set_filter(
                "inning",
                {"min": inning_min, "max": 9},
                m.start(),
                m.end(),
            )

    for m in _HOME_RE.finditer(text):
        _set_filter("home_away", {"side": "home"}, m.start(), m.end())
    for m in _AWAY_RE.finditer(text):
        _set_filter("home_away", {"side": "away"}, m.start(), m.end())

    for m in _MONTH_RE.finditer(text):
        month_value = _MONTH_NAME_TO_INT[m.group(1).lower()]
        _set_filter("month", {"month": month_value}, m.start(), m.end())

    count_events: list[tuple[int, int, str, int | None, int | None]] = []

    for m in _FULL_COUNT_RE.finditer(text):
        count_events.append((m.start(), m.end(), "full_count", 3, 2))
    for m in _COUNT_PAIR_RE.finditer(text):
        count_events.append((m.start(), m.end(), "pair", int(m.group(1)), int(m.group(2))))
    for m in _BALLS_ANY_STRIKES_RE.finditer(text):
        count_events.append((m.start(), m.end(), "balls_any_strikes", int(m.group(1)), None))
    for m in _ANY_BALLS_RE.finditer(text):
        count_events.append((m.start(), m.end(), "any_balls", None, None))
    for m in _ANY_STRIKES_RE.finditer(text):
        count_events.append((m.start(), m.end(), "any_strikes", None, None))
    for m in _BALLS_RE.finditer(text):
        count_events.append((m.start(), m.end(), "balls", int(m.group(1)), None))
    for m in _STRIKES_RE.finditer(text):
        count_events.append((m.start(), m.end(), "strikes", None, int(m.group(1))))

    if count_events:
        count_events.sort(key=lambda x: (x[0], x[1]))
        balls: int | None = None
        strikes: int | None = None
        last_start = 0
        last_end = 0
        for start, end, event_type, balls_val, strikes_val in count_events:
            last_start = start
            last_end = end
            spans.append((start, end))
            if event_type in {"full_count", "pair"}:
                balls = balls_val
                strikes = strikes_val
            elif event_type == "balls_any_strikes":
                balls = balls_val
                strikes = None
            elif event_type == "any_balls":
                balls = None
            elif event_type == "any_strikes":
                strikes = None
            elif event_type == "balls":
                balls = balls_val
            elif event_type == "strikes":
                strikes = strikes_val

        if balls is not None or strikes is not None:
            _set_filter(
                "count",
                {"balls": balls, "strikes": strikes},
                last_start,
                last_end,
            )

    # Basic sanity warning for unsupported filters accidentally introduced by parsing.
    unknown_types = [k for k in filters if k not in FILTER_REGISTRY]
    for filter_type in unknown_types:
        warnings.append(f"Dropped unknown filter type: {filter_type}.")
        filters.pop(filter_type, None)
        positions.pop(filter_type, None)

    return filters, positions, spans, warnings


def _build_filter_rows(
    filters: dict[str, dict[str, Any]],
    positions: dict[str, int],
    warnings: list[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    ordered_types = sorted(filters.keys(), key=lambda key: positions.get(key, 0))
    for idx, filter_type in enumerate(ordered_types):
        params = filters[filter_type]
        valid, validation_warning = _validate_filter(filter_type, params)
        if not valid:
            if validation_warning:
                warnings.append(validation_warning)
            continue
        rows.append(
            {
                "id": f"nl{idx}",
                "filter_type": filter_type,
                "params": params,
            }
        )
    return rows


def _validate_filter(filter_type: str, params: dict[str, Any]) -> tuple[bool, str | None]:
    if filter_type == "pitcher_hand":
        hand = params.get("hand")
        if hand not in {"L", "R"}:
            return False, f"Dropped pitcher_hand filter with invalid hand: {hand!r}."
        return True, None

    if filter_type == "inning":
        inning_min = params.get("min")
        inning_max = params.get("max")
        if not isinstance(inning_min, int) or not isinstance(inning_max, int):
            return False, "Dropped inning filter with non-integer bounds."
        if inning_min < 1 or inning_min > 9 or inning_max < 1 or inning_max > 9:
            return False, "Dropped inning filter outside valid range 1-9."
        if inning_min > inning_max:
            return False, "Dropped inning filter where min > max."
        return True, None

    if filter_type == "home_away":
        side = params.get("side")
        if side not in {"home", "away"}:
            return False, f"Dropped home_away filter with invalid side: {side!r}."
        return True, None

    if filter_type == "month":
        month = params.get("month")
        if not isinstance(month, int) or month < 1 or month > 12:
            return False, "Dropped month filter outside valid range 1-12."
        return True, None

    if filter_type == "count":
        balls = params.get("balls")
        strikes = params.get("strikes")
        if balls is None and strikes is None:
            return False, "Dropped count filter with both balls and strikes set to any."
        if balls is not None and (not isinstance(balls, int) or balls < 0 or balls > 3):
            return False, "Dropped count filter with invalid balls value."
        if strikes is not None and (not isinstance(strikes, int) or strikes < 0 or strikes > 2):
            return False, "Dropped count filter with invalid strikes value."
        return True, None

    return False, f"Dropped unknown filter type: {filter_type}."


def _extract_comparison_fragments(text: str) -> tuple[str | None, str | None]:
    if not text:
        return None, None

    match_compare = _COMPARE_AND_RE.match(text)
    if match_compare:
        return (
            _clean_player_fragment(match_compare.group("a")),
            _clean_player_fragment(match_compare.group("b")),
        )

    match_vs = _VS_RE.match(text)
    if match_vs:
        return (
            _clean_player_fragment(match_vs.group("a")),
            _clean_player_fragment(match_vs.group("b")),
        )

    return None, None


def _resolve_player_name(
    fragment: str,
    player_names: Sequence[str],
) -> tuple[str | None, str | None, bool]:
    text = _normalize_name(fragment)
    if not text:
        return None, "Could not detect a player name in query.", False

    normalized_to_original: dict[str, str] = {
        _normalize_name(name): name for name in player_names
    }

    exact = normalized_to_original.get(text)
    if exact is not None:
        return exact, None, False

    tokens = [tok for tok in text.split() if tok]
    partial_candidates: list[str] = []
    if tokens:
        for name in player_names:
            n_name = _normalize_name(name)
            if all(tok in n_name for tok in tokens):
                partial_candidates.append(name)

    if len(partial_candidates) == 1:
        return partial_candidates[0], None, False

    if len(partial_candidates) > 1:
        ranked = sorted(
            partial_candidates,
            key=lambda name: SequenceMatcher(None, text, _normalize_name(name)).ratio(),
            reverse=True,
        )
        chosen = ranked[0]
        warning = (
            f"Player fragment '{fragment}' matched multiple players; "
            f"using '{chosen}'."
        )
        return chosen, warning, True

    fuzzy_ranked = sorted(
        (
            (name, SequenceMatcher(None, text, _normalize_name(name)).ratio())
            for name in player_names
        ),
        key=lambda item: item[1],
        reverse=True,
    )
    if fuzzy_ranked and fuzzy_ranked[0][1] >= 0.8:
        chosen = fuzzy_ranked[0][0]
        warning = f"Resolved player fragment '{fragment}' to '{chosen}'."
        return chosen, warning, False

    return None, f"Could not resolve player name from '{fragment}'.", False


def _strip_spans(text: str, spans: Sequence[tuple[int, int]]) -> str:
    if not spans:
        return text

    merged: list[tuple[int, int]] = []
    for start, end in sorted(spans):
        if start >= end:
            continue
        if not merged or start > merged[-1][1]:
            merged.append((start, end))
        else:
            prev_start, prev_end = merged[-1]
            merged[-1] = (prev_start, max(prev_end, end))

    out_parts: list[str] = []
    cursor = 0
    for start, end in merged:
        out_parts.append(text[cursor:start])
        out_parts.append(" ")
        cursor = end
    out_parts.append(text[cursor:])
    stripped = "".join(out_parts)
    stripped = re.sub(r"\s+", " ", stripped).strip()
    return stripped


def _clean_player_fragment(text: str) -> str:
    cleaned = text or ""
    prev = None
    while prev != cleaned:
        prev = cleaned
        cleaned = _FILLER_PREFIX_RE.sub("", cleaned).strip()
    cleaned = cleaned.strip(" ,.;:-")
    cleaned = re.sub(r"\s+", " ", cleaned)
    tokens = cleaned.split()
    while tokens:
        normalized = re.sub(r"[^a-z0-9]+", "", tokens[-1].lower())
        if normalized in _TRAILING_FRAGMENT_STOPWORDS:
            tokens.pop()
            continue
        break
    cleaned = " ".join(tokens).strip()
    return cleaned


def _normalize_name(text: str) -> str:
    cleaned = re.sub(r"[^a-z0-9 ]+", " ", (text or "").lower())
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _normalize_parsed_season(
    season: int | None,
    valid_seasons: set[int] | None,
    warnings: list[str],
) -> int | None:
    if season is None:
        return None
    if valid_seasons is not None and season not in valid_seasons:
        warnings.append(f"Parsed season {season} is not in available seasons; season unchanged.")
        return None
    return season


def _extract_fragment_season(
    fragment: str,
    valid_seasons: set[int] | None,
    warnings: list[str],
) -> tuple[int | None, str]:
    years = [int(m.group(0)) for m in _YEAR_RE.finditer(fragment or "")]
    parsed_season = _normalize_parsed_season(years[-1], valid_seasons, warnings) if years else None
    cleaned = _YEAR_RE.sub(" ", fragment or "")
    return parsed_season, _clean_player_fragment(cleaned)


def _remove_global_modifiers(text: str) -> str:
    cleaned = text or ""
    for pattern in _GLOBAL_SEASON_MODIFIER_RES:
        cleaned = pattern.sub(" ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned
