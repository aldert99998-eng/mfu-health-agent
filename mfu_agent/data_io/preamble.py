"""Single source of truth for SQL/comment preamble detection and stripping.

Input files (Zabbix exports, ad-hoc CSVs) sometimes start with the SQL query
that produced them, SQL comments, or BOM bytes. Three different call sites
historically implemented slightly different versions of the same logic with
diverging keyword sets — that caused a file to be accepted by one parser and
rejected by another. This module unifies the behaviour.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable

logger = logging.getLogger(__name__)

SQL_PREAMBLE_KEYWORDS: frozenset[str] = frozenset({
    "SELECT",
    "UNION",
    "FROM",
    "WHERE",
    "INNER",
    "LEFT",
    "RIGHT",
    "JOIN",
    "INSERT",
    "UPDATE",
    "CREATE",
    "WITH",
})

_TOKEN_SPLIT = re.compile(r"[\s(,;]+")


def _line_is_preamble(stripped: str) -> bool:
    if not stripped:
        return True
    if stripped.startswith("--"):
        return True
    tokens = set(_TOKEN_SPLIT.split(stripped.upper()[:300]))
    return bool(tokens & SQL_PREAMBLE_KEYWORDS)


def strip_sql_preamble_text(text: str) -> tuple[str, int]:
    """Strip BOM + leading SQL/comment lines from text.

    Returns ``(cleaned_text, skipped_line_count)``. ``skipped_line_count == 0``
    means no preamble was detected and ``cleaned_text`` is the input verbatim
    (minus any leading BOM).
    """
    if text.startswith("\ufeff"):
        text = text[1:]

    lines = text.split("\n")
    start = 0
    for i, line in enumerate(lines):
        if _line_is_preamble(line.strip()):
            start = i + 1
            continue
        break

    if start == 0:
        return text, 0
    return "\n".join(lines[start:]), start


def strip_sql_preamble_bytes(raw: bytes) -> bytes:
    """Byte-level wrapper: decode, strip, re-encode as UTF-8.

    Returns ``raw`` unchanged if decoding fails or no preamble is present,
    so this is safe to call unconditionally before writing to disk.
    """
    for enc in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            text = raw.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    else:
        return raw

    cleaned, skipped = strip_sql_preamble_text(text)
    if skipped == 0:
        return raw
    logger.info("SQL-преамбула: пропущено %d строк перед записью", skipped)
    return cleaned.encode("utf-8")


def has_sql_preamble_in_columns(col_names: Iterable[str]) -> bool:
    """Detect the case where pandas parsed SQL query text as a single
    column name row. Used to decide whether to re-parse after stripping.
    """
    col_text = " ".join(str(c) for c in col_names).upper()
    tokens = set(_TOKEN_SPLIT.split(col_text[:500]))
    return bool(tokens & SQL_PREAMBLE_KEYWORDS)
