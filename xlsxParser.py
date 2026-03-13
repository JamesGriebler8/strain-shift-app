"""
xlsxParser.py
=============
Parser for ARES/DHR data exported as Excel (.xlsx / .xls) files.

Each sheet = one step. Layout:
    Row 0  : step name  (e.g. "Arbitrary Wave - 5")
    Row 1  : column headers
    Row 2  : units
    Row 3+ : numeric data

Public interface is identical to txtParser.py.

Performance
-----------
Bypasses openpyxl. Instead:
  1. Opens the .xlsx zip archive once, reads all sheet XMLs into memory.
  2. Parses each sheet XML with stdlib xml.etree.ElementTree, using cell
     references (e.g. "C3") to correctly place sparse cells.
  3. Builds numpy arrays directly, skipping pandas type inference.

~2.5x faster than openpyxl read_only mode on large files.

Notes
-----
- Strain exported in fractional units. Use AnalysisConfig(percent_strain=False).
- Non-step sheets skipped automatically (first cell must match "<Type> - <N>").
- Sparse rows are placed by column reference, not sequential padding, so
  blank cells (e.g. a missing Strain unit) land at the correct column index.
"""

from __future__ import annotations

import re
import warnings
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

_NS   = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
_RNS  = "http://schemas.openxmlformats.org/package/2006/relationships"
_RONS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
_STEP_RE = re.compile(r"^(.+?)\s*-\s*(\d+)$")
_COL_RE  = re.compile(r"([A-Z]+)")


def _col_index(ref: str) -> int:
    """Convert the column-letter portion of a cell ref (e.g. 'C3') to a 0-based index."""
    m = _COL_RE.match(ref)
    if not m:
        return 0
    idx = 0
    for ch in m.group(1):
        idx = idx * 26 + (ord(ch) - ord("A") + 1)
    return idx - 1


def _parse_shared_strings(zf: zipfile.ZipFile) -> list:
    try:
        xml_bytes = zf.read("xl/sharedStrings.xml")
    except KeyError:
        return []
    tree = ET.fromstring(xml_bytes)
    return ["".join(si.itertext()) for si in tree.findall(f"{{{_NS}}}si")]


def _sheet_order(zf: zipfile.ZipFile) -> list:
    """Return [(sheet_name, zip_path_relative_to_xl/), ...]."""
    wb_tree   = ET.fromstring(zf.read("xl/workbook.xml"))
    rels_tree = ET.fromstring(zf.read("xl/_rels/workbook.xml.rels"))
    rid_to_target = {
        r.get("Id"): r.get("Target")
        for r in rels_tree.findall(f"{{{_RNS}}}Relationship")
    }
    sheets_el = wb_tree.find(f"{{{_NS}}}sheets")
    return [
        (s.get("name"), rid_to_target[s.get(f"{{{_RONS}}}id")])
        for s in sheets_el
    ]


def _parse_sheet_xml(xml_bytes: bytes, shared: list) -> list:
    """
    Parse a worksheet XML blob into a list of rows.

    Uses each cell's column reference (e.g. the 'C' in 'C3') to place values
    at the correct index, so sparse rows with missing middle cells are handled
    correctly — a missing cell becomes None rather than collapsing the row.
    """
    tree = ET.fromstring(xml_bytes)
    rows = []
    for row_el in tree.iter(f"{{{_NS}}}row"):
        cells: dict = {}
        for c in row_el:
            ref = c.get("r", "")
            idx = _col_index(ref) if ref else len(cells)
            t    = c.get("t", "")
            v_el = c.find(f"{{{_NS}}}v")
            if v_el is None or v_el.text is None:
                cells[idx] = None
            elif t == "s":
                cells[idx] = shared[int(v_el.text)]
            else:
                try:
                    cells[idx] = float(v_el.text)
                except ValueError:
                    cells[idx] = v_el.text
        if cells:
            max_col = max(cells.keys())
            rows.append([cells.get(i) for i in range(max_col + 1)])
    return rows


def _to_float(v) -> float:
    if v is None:
        return float("nan")
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def _pad(row: list, ncols: int) -> list:
    """Pad or trim a row to exactly ncols entries."""
    if len(row) < ncols:
        return row + [None] * (ncols - len(row))
    return row[:ncols]


class Step:
    """
    One measurement step from a single Excel sheet.

    Attributes
    ----------
    name : str
    step_type : str
    number : int
    df : pd.DataFrame
    units : dict[str, str]
    columns : list[str]
    """

    def __init__(self, sheet_name: str, rows: list) -> None:
        """
        Parameters
        ----------
        sheet_name : str
        rows : list of lists
            Step-name row already stripped by File.
            rows[0] = headers, rows[1] = units, rows[2:] = numeric data.
        """
        m = _STEP_RE.match(sheet_name)
        if m is None:
            raise ValueError(f"Sheet '{sheet_name}' does not match '<StepType> - <N>'.")
        self.name      = sheet_name
        self.step_type = m.group(1).strip()
        self.number    = int(m.group(2))

        raw_headers = rows[0] if rows else []
        ncols = len(raw_headers)
        headers = [
            str(v) if v is not None else f"col_{i}"
            for i, v in enumerate(raw_headers)
        ]

        # Units row: pad to ncols so every column gets an entry (may be "")
        units_row = _pad(rows[1] if len(rows) > 1 else [], ncols)
        self.units = {
            col: (str(u) if u is not None else "")
            for col, u in zip(headers, units_row)
        }

        data_rows = rows[2:] if len(rows) > 2 else []
        arr = np.array(
            [[_to_float(v) for v in _pad(row, ncols)] for row in data_rows],
            dtype=np.float64,
        )
        if arr.size:
            arr = arr[~np.all(np.isnan(arr), axis=1)]

        self.df      = pd.DataFrame(arr, columns=headers)
        self.columns = headers

    def __repr__(self):
        return (f"Step(name={self.name!r}, step_type={self.step_type!r}, "
                f"number={self.number}, rows={len(self.df)})")


class File:
    """
    Parsed ARES/DHR Excel export.

    Attributes
    ----------
    filepath : str
    filename : str
    steps : List[Step]
    """

    def __init__(self, filepath: str) -> None:
        self.filepath = str(Path(filepath).resolve())
        self.filename = Path(filepath).name

        # Read everything from the zip in one sequential pass, then close it.
        # All sheet XMLs are held in memory before parsing begins.
        with zipfile.ZipFile(filepath) as zf:
            shared   = _parse_shared_strings(zf)
            order    = _sheet_order(zf)
            raw_xmls = {name: zf.read(f"xl/{path}") for name, path in order}

        self.steps: List[Step] = []
        for sheet_name, _ in order:
            rows = _parse_sheet_xml(raw_xmls[sheet_name], shared)
            if not rows:
                continue
            cell00 = str(rows[0][0]).strip() if rows[0] and rows[0][0] is not None else ""
            if not _STEP_RE.match(cell00):
                continue
            try:
                self.steps.append(Step(sheet_name, rows[1:]))
            except Exception as exc:
                warnings.warn(f"Skipping sheet '{sheet_name}': {exc}", stacklevel=2)

    def get_steps_by_type(self, step_type: str = "") -> List[Step]:
        """Return steps whose step_type contains step_type (case-insensitive)."""
        needle = step_type.strip().lower()
        if not needle:
            return list(self.steps)
        return [s for s in self.steps if needle in s.step_type.lower()]

    def __repr__(self):
        return f"File(filename={self.filename!r}, steps={len(self.steps)})"


if __name__ == "__main__":
    import sys, time
    path = sys.argv[1] if len(sys.argv) > 1 else None
    if not path:
        print("Usage: python xlsxParser.py <file.xlsx>")
        sys.exit(1)

    t0 = time.perf_counter()
    f  = File(path)
    elapsed = time.perf_counter() - t0

    print(f"File    : {f.filename}")
    print(f"Steps   : {len(f.steps)}")
    print(f"Elapsed : {elapsed:.2f}s")

    counts: dict = {}
    for s in f.steps:
        counts[s.step_type] = counts.get(s.step_type, 0) + 1
    print("Step types:")
    for st, n in counts.items():
        print(f"  {st!r:30s} x {n}")

    arb = f.get_steps_by_type("Arbitrary Wave")
    if arb:
        ex = arb[0]
        print(f"\nExample : {ex}")
        print(f"Columns : {ex.columns}")
        print(f"Units   : {ex.units}")
        print(f"Head:\n{ex.df.head(3)}")