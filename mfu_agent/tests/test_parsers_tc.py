"""QA test cases TC-E-001 through TC-E-015 for data_io/parsers.py."""

import json
import os
import sys
import tempfile
from pathlib import Path

# Ensure mfu_agent is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from data_io.parsers import (
    CSVParser,
    EmptyFileError,
    FormatDetector,
    InvalidFileFormatError,
    JSONParser,
    MalformedFileError,
    XLSXParser,
    parse_file,
)
from data_io.models import FileFormat

RESULTS: list[str] = []


def report(tc: str, passed: bool, evidence: str) -> None:
    status = "PASS" if passed else "FAIL"
    line = f"{tc} | {status} | {evidence}"
    RESULTS.append(line)
    print(line)


# ── TC-E-001 ─────────────────────────────────────────────────────────────────
def test_e001():
    tc = "TC-E-001"
    try:
        with tempfile.NamedTemporaryFile(suffix=".csv", mode="w", delete=False, encoding="utf-8") as f:
            f.write("device_id,timestamp,error_code,toner_level\n")
            for i in range(5):
                f.write(f"D{i},2024-01-0{i+1},E00{i},{90-i*10}\n")
            path = Path(f.name)

        df = parse_file(path)
        ok = (
            len(df) == 5
            and list(df.columns) == ["device_id", "timestamp", "error_code", "toner_level"]
            and all(df[c].dtype == object for c in df.columns)  # str dtype
        )
        report(tc, ok, f"rows={len(df)}, cols={list(df.columns)}, all_str={ok}")
    except Exception as exc:
        report(tc, False, f"Exception: {exc}")
    finally:
        path.unlink(missing_ok=True)


# ── TC-E-002 ─────────────────────────────────────────────────────────────────
def test_e002():
    tc = "TC-E-002"
    try:
        from openpyxl import Workbook
        with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as f:
            path = Path(f.name)

        wb = Workbook()
        ws1 = wb.active
        ws1.title = "errors"
        ws1.append(["device_id", "error_code"])
        for i in range(3):
            ws1.append([f"D{i}", f"E{i}"])

        ws2 = wb.create_sheet("resources")
        ws2.append(["device_id", "toner"])
        for i in range(2):
            ws2.append([f"D{i}", f"{80+i}"])

        wb.save(path)
        wb.close()

        df = parse_file(path)
        # Expect 3 + 2 = 5 rows total
        ok = len(df) == 5
        report(tc, ok, f"rows={len(df)}, expected=5 (3+2 sheets concatenated)")
    except Exception as exc:
        report(tc, False, f"Exception: {exc}")
    finally:
        path.unlink(missing_ok=True)


# ── TC-E-003 ─────────────────────────────────────────────────────────────────
def test_e003():
    tc = "TC-E-003"
    try:
        data = [{"id": "D1", "ts": "2024-01-01"}, {"id": "D2", "ts": "2024-01-02"}]
        with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False, encoding="utf-8") as f:
            json.dump(data, f)
            path = Path(f.name)

        df = parse_file(path)
        ok = len(df) == 2
        report(tc, ok, f"rows={len(df)}, expected=2")
    except Exception as exc:
        report(tc, False, f"Exception: {exc}")
    finally:
        path.unlink(missing_ok=True)


# ── TC-E-004 ─────────────────────────────────────────────────────────────────
def test_e004():
    tc = "TC-E-004"
    try:
        lines = [
            json.dumps({"id": f"D{i}", "val": i}) for i in range(3)
        ]
        with tempfile.NamedTemporaryFile(suffix=".jsonl", mode="w", delete=False, encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
            path = Path(f.name)

        df = parse_file(path)
        ok = len(df) == 3
        report(tc, ok, f"rows={len(df)}, expected=3")
    except Exception as exc:
        report(tc, False, f"Exception: {exc}")
    finally:
        path.unlink(missing_ok=True)


# ── TC-E-005 ─────────────────────────────────────────────────────────────────
def test_e005():
    tc = "TC-E-005"
    try:
        with tempfile.NamedTemporaryFile(suffix=".dat", mode="w", delete=False, encoding="utf-8") as f:
            f.write("a,b,c\n1,2,3\n4,5,6\n")
            path = Path(f.name)

        fmt = FormatDetector.detect(path)
        df = parse_file(path)
        ok = fmt == FileFormat.CSV and len(df) == 2
        report(tc, ok, f"detected={fmt}, rows={len(df)}")
    except Exception as exc:
        report(tc, False, f"Exception: {exc}")
    finally:
        path.unlink(missing_ok=True)


# ── TC-E-006 ─────────────────────────────────────────────────────────────────
def test_e006():
    tc = "TC-E-006"
    try:
        with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as f:
            f.write(os.urandom(256))
            path = Path(f.name)

        try:
            parse_file(path)
            report(tc, False, "No exception raised for binary file with .xlsx extension")
        except InvalidFileFormatError as exc:
            report(tc, True, f"InvalidFileFormatError raised: {exc}")
        except Exception as exc:
            # Accept MalformedFileError too if magic bytes pass but content is garbage
            report(tc, False, f"Wrong exception type: {type(exc).__name__}: {exc}")
    finally:
        path.unlink(missing_ok=True)


# ── TC-E-007 ─────────────────────────────────────────────────────────────────
def test_e007():
    tc = "TC-E-007"
    try:
        delimiters = [(",", "comma"), (";", "semicolon"), ("\t", "tab")]
        all_ok = True
        details = []
        for delim, name in delimiters:
            with tempfile.NamedTemporaryFile(suffix=".csv", mode="w", delete=False, encoding="utf-8") as f:
                f.write(f"a{delim}b{delim}c\n")
                f.write(f"1{delim}2{delim}3\n")
                f.write(f"4{delim}5{delim}6\n")
                path = Path(f.name)
            try:
                df = parse_file(path)
                ok = len(df) == 2 and len(df.columns) == 3
                details.append(f"{name}={'OK' if ok else 'FAIL'}")
                if not ok:
                    all_ok = False
            except Exception as exc:
                details.append(f"{name}=ERR({exc})")
                all_ok = False
            finally:
                path.unlink(missing_ok=True)

        report(tc, all_ok, "; ".join(details))
    except Exception as exc:
        report(tc, False, f"Exception: {exc}")


# ── TC-E-009 ─────────────────────────────────────────────────────────────────
def test_e009():
    tc = "TC-E-009"
    try:
        cyrillic_text = "Привет"
        encodings = [
            ("utf-8", "UTF-8"),
            ("utf-8-sig", "UTF-8 BOM"),
            ("cp1251", "Windows-1251"),
        ]
        all_ok = True
        details = []
        for enc, label in encodings:
            content = f"name,value\n{cyrillic_text},42\n"
            raw = content.encode(enc)
            with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
                f.write(raw)
                path = Path(f.name)
            try:
                df = parse_file(path)
                cell_val = df.iloc[0, 0]
                ok = cyrillic_text in cell_val
                details.append(f"{label}={'OK' if ok else 'FAIL('+cell_val+')'}")
                if not ok:
                    all_ok = False
            except Exception as exc:
                details.append(f"{label}=ERR({exc})")
                all_ok = False
            finally:
                path.unlink(missing_ok=True)

        report(tc, all_ok, "; ".join(details))
    except Exception as exc:
        report(tc, False, f"Exception: {exc}")


# ── TC-E-010 ─────────────────────────────────────────────────────────────────
def test_e010():
    tc = "TC-E-010"
    try:
        # Malformed CSV with unclosed quote
        content = 'a,b,c\n"unclosed,value,here\n1,2,3\n'
        with tempfile.NamedTemporaryFile(suffix=".csv", mode="w", delete=False, encoding="utf-8") as f:
            f.write(content)
            path = Path(f.name)

        try:
            df = parse_file(path)
            # pandas may still parse it but with warnings/wrong shape
            # If it parses without error, check if data is mangled
            report(tc, False, f"No exception raised, df shape={df.shape}")
        except (MalformedFileError, Exception) as exc:
            is_malformed = isinstance(exc, MalformedFileError)
            report(tc, True, f"{'MalformedFileError' if is_malformed else type(exc).__name__} raised")
    finally:
        path.unlink(missing_ok=True)


# ── TC-E-011 ─────────────────────────────────────────────────────────────────
def test_e011():
    tc = "TC-E-011"
    try:
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            path = Path(f.name)
            # Write nothing — 0 bytes

        try:
            parse_file(path)
            report(tc, False, "No exception for empty file")
        except EmptyFileError:
            report(tc, True, "EmptyFileError raised")
        except Exception as exc:
            report(tc, False, f"Wrong exception: {type(exc).__name__}: {exc}")
    finally:
        path.unlink(missing_ok=True)


# ── TC-E-012 ─────────────────────────────────────────────────────────────────
def test_e012():
    tc = "TC-E-012"
    try:
        with tempfile.NamedTemporaryFile(suffix=".csv", mode="w", delete=False, encoding="utf-8") as f:
            f.write("a,b,c\n")  # headers only
            path = Path(f.name)

        try:
            parse_file(path)
            report(tc, False, "No exception for headers-only file")
        except EmptyFileError:
            report(tc, True, "EmptyFileError raised for headers-only CSV")
        except Exception as exc:
            report(tc, False, f"Wrong exception: {type(exc).__name__}: {exc}")
    finally:
        path.unlink(missing_ok=True)


# ── TC-E-013 ─────────────────────────────────────────────────────────────────
def test_e013():
    tc = "TC-E-013"
    try:
        from data_io.normalizer import ingest_file
        from data_io.factor_store import FactorStore

        with tempfile.NamedTemporaryFile(suffix=".csv", mode="w", delete=False, encoding="utf-8") as f:
            f.write("timestamp,error_code,toner_level\n")
            f.write("2024-01-01,E001,90\n")
            path = Path(f.name)

        store = FactorStore()
        result = ingest_file(path, store)
        has_device_id_mention = any("device_id" in e for e in result.errors)
        ok = not result.success and has_device_id_mention
        report(tc, ok, f"success={result.success}, errors mention device_id={has_device_id_mention}")
    except Exception as exc:
        report(tc, False, f"Exception: {exc}")
    finally:
        path.unlink(missing_ok=True)


# ── TC-E-014 ─────────────────────────────────────────────────────────────────
def test_e014():
    tc = "TC-E-014"
    try:
        with tempfile.NamedTemporaryFile(suffix=".csv", mode="w", delete=False, encoding="utf-8") as f:
            f.write("A,B,A\n")
            f.write("1,2,3\n")
            f.write("4,5,6\n")
            path = Path(f.name)

        df = parse_file(path)
        cols = list(df.columns)
        # Expect duplicate "A" renamed: pandas default or custom logic
        has_renamed = any("A" in c and c != "A" for c in cols)
        ok = len(cols) == 3 and has_renamed
        report(tc, ok, f"columns={cols}")
    except Exception as exc:
        report(tc, False, f"Exception: {exc}")
    finally:
        path.unlink(missing_ok=True)


# ── TC-E-015 ─────────────────────────────────────────────────────────────────
def test_e015():
    tc = "TC-E-015"
    try:
        long_str = "X" * 15000
        with tempfile.NamedTemporaryFile(suffix=".csv", mode="w", delete=False, encoding="utf-8") as f:
            f.write("data\n")
            f.write(f"{long_str}\n")
            path = Path(f.name)

        df = parse_file(path)
        cell = df.iloc[0, 0]
        ok = len(cell) == 15000
        report(tc, ok, f"cell_len={len(cell)}, expected=15000")
    except Exception as exc:
        report(tc, False, f"Exception: {exc}")
    finally:
        path.unlink(missing_ok=True)


# ── Run all ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    test_e001()
    test_e002()
    test_e003()
    test_e004()
    test_e005()
    test_e006()
    test_e007()
    test_e009()
    test_e010()
    test_e011()
    test_e012()
    test_e013()
    test_e014()
    test_e015()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for r in RESULTS:
        print(r)
