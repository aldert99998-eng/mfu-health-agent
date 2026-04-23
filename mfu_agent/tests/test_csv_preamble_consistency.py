"""All three preamble-stripping call sites must agree.

Historically the page-level byte stripper, the CSVParser text stripper, and
the normalizer's post-parse stripper used three different SQL keyword sets.
A file with `WITH` / `RIGHT` / `UPDATE` would get stripped by one and left
intact by another. This test guards the single-source-of-truth module.
"""

from __future__ import annotations

import pandas as pd
import pytest

from data_io.preamble import (
    SQL_PREAMBLE_KEYWORDS,
    has_sql_preamble_in_columns,
    strip_sql_preamble_bytes,
    strip_sql_preamble_text,
)


_SAMPLE = """-- Exported 2026-04-20
SELECT h.hostid, h.host FROM hosts h
WHERE h.status = 0;

device_id,timestamp,error_code
DEV001,2026-04-15T10:00:00Z,75-530-00
DEV002,2026-04-15T10:05:00Z,72-535-00
"""


def test_text_and_bytes_agree_on_sample() -> None:
    cleaned_text, skipped = strip_sql_preamble_text(_SAMPLE)
    assert skipped > 0
    assert cleaned_text.startswith("device_id,timestamp,error_code")

    cleaned_bytes = strip_sql_preamble_bytes(_SAMPLE.encode("utf-8"))
    assert cleaned_bytes.decode("utf-8") == cleaned_text


def test_csvparser_and_normalizer_use_same_keywords() -> None:
    """CSVParser._skip_preamble and normalizer._strip_sql_lines delegate
    to strip_sql_preamble_text; both must skip identical lines."""
    from data_io.parsers import CSVParser

    parser_cleaned = CSVParser._skip_preamble(_SAMPLE)
    direct_cleaned, _ = strip_sql_preamble_text(_SAMPLE)
    assert parser_cleaned == direct_cleaned


@pytest.mark.parametrize(
    "keyword",
    sorted(SQL_PREAMBLE_KEYWORDS),
)
def test_every_keyword_triggers_stripping(keyword: str) -> None:
    sample = f"{keyword} x FROM y\ndevice_id,ts\nDEV,2026-01-01\n"
    cleaned, skipped = strip_sql_preamble_text(sample)
    assert skipped >= 1, f"keyword {keyword!r} was not detected as preamble"
    assert cleaned.startswith("device_id")


def test_no_preamble_is_passthrough() -> None:
    sample = "device_id,ts\nDEV,2026-01-01\n"
    cleaned, skipped = strip_sql_preamble_text(sample)
    assert skipped == 0
    assert cleaned == sample


def test_bom_is_stripped_even_without_preamble() -> None:
    sample = "\ufeffdevice_id,ts\nDEV,2026-01-01\n"
    cleaned, _ = strip_sql_preamble_text(sample)
    assert not cleaned.startswith("\ufeff")


def test_has_sql_preamble_in_columns_matches_keyword() -> None:
    cols = ["SELECT h.hostid AS id"]
    assert has_sql_preamble_in_columns(cols)

    real_cols = ["device_id", "timestamp", "error_code"]
    assert not has_sql_preamble_in_columns(real_cols)


def test_dataframe_with_sql_header_is_detected() -> None:
    """End-to-end: normalizer's detector flags a malformed DataFrame."""
    from data_io.normalizer import _has_sql_preamble

    bad = pd.DataFrame([[1]], columns=["SELECT h.host FROM hosts"])
    good = pd.DataFrame([[1]], columns=["device_id"])
    assert _has_sql_preamble(bad)
    assert not _has_sql_preamble(good)
