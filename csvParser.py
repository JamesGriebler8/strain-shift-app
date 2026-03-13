"""
csvParser.py
============
Parser for RheoCompass CSV exports, providing the same File / Step interface
as txtParser.py so that strain_shift_analysis.py works without modification.

CSV format
----------
The file contains one or more ``Result:`` blocks, each corresponding to one
step (e.g. "0.112Pa Forward 1", "0.112Pa Reverse 1").  Within each Result
block there are one or more numbered intervals, each with its own header:

    Result:,<step name>
    ...
    Interval and data points:,<interval_no>,<n_points>
    Interval data:,Point No.,Time,Interval Time,Shear Stress,Shear Strain
    ...
    ,,[s],[s],[Pa],[1]
    ,1,<t_global>,<t_interval>,<stress>,<strain>
    ...

Interval structure
------------------
Each Result block contains multiple intervals that share a continuous global
clock (``Time`` column) but whose ``Interval Time`` resets to ~0 at the start
of each interval.  This mirrors the MATLAB import_mcr_xls output where
``data_out(:,1,:)`` stores the interval number and subsequent columns store
the measured variables.

The MATLAB code identifies interval boundaries by detecting when the point
number resets, and measurement boundaries by detecting when global time goes
backwards.  This parser replicates that logic directly from the CSV headers.

Column mapping to the Step interface
-------------------------------------
    "Step time"     <- global Time offset to zero at the start of the Result
                       (monotonic across all intervals — safe for dt / FFT)
    "Stress"        <- Shear Stress  [Pa]
    "Strain"        <- Shear Strain  [1]
    "Interval"      <- 1-based interval number within the Result block
    "Interval Time" <- per-interval time (resets each interval) [s]
    "Time"          <- absolute instrument time [s]

The key fix vs. the raw CSV "Interval Time" column: that column resets to
~0 at each interval boundary, making it non-monotonic across intervals and
corrupting dt and FFT-based frequency detection in the analysis code.
"Step time" is always monotonic within a Result block.

Usage
-----
    from csvParser import File
    data = File("my_experiment.csv")
    all_steps  = data.steps                        # one Step per Result block
    fwd_steps  = data.get_steps_by_type("Forward") # substring match on name
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Step
# ---------------------------------------------------------------------------

class Step:
    """
    One Result block from a RheoCompass CSV export.

    Attributes
    ----------
    name : str
        The result name as it appears in the CSV (e.g. "0.112Pa Forward 1").
    step_type : str
        Same as name; callers filter by substring via get_steps_by_type.
    df : pd.DataFrame
        All data points from this step concatenated across intervals.
        Always contains:
            "Step time"     - time since start of this Result block [s], monotonic
            "Stress"        - shear stress [Pa]
            "Strain"        - shear strain [1]
            "Interval"      - 1-based interval number within this Result
            "Interval Time" - per-interval time (resets each interval) [s]
            "Time"          - absolute instrument time [s]
    columns : list[str]
        Convenience accessor for df.columns.
    """

    def __init__(self, name: str, df: pd.DataFrame) -> None:
        self.name: str = name
        self.step_type: str = name
        self.df: pd.DataFrame = df.reset_index(drop=True)

    @property
    def columns(self) -> List[str]:
        return list(self.df.columns)

    def __repr__(self) -> str:
        n_intervals = self.df["Interval"].nunique() if "Interval" in self.df else "?"
        return (f"Step(name={self.name!r}, "
                f"n_points={len(self.df)}, "
                f"n_intervals={n_intervals})")


# ---------------------------------------------------------------------------
# File
# ---------------------------------------------------------------------------

class File:
    """
    Parsed RheoCompass CSV export.

    Parameters
    ----------
    filepath : str or Path

    Attributes
    ----------
    filepath : str
    project : str
    test : str
    steps : list[Step]
        One Step per ``Result:`` block, in file order.
    """

    def __init__(self, filepath: str | Path) -> None:
        self.filepath: str = str(filepath)
        self.project: str = ""
        self.test: str = ""
        self.steps: List[Step] = []
        self._parse()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def get_steps_by_type(self, step_type: str) -> List[Step]:
        """
        Return all steps whose name contains *step_type* (case-insensitive).

        For this CSV format the step "type" is embedded in the result name,
        e.g. "Forward", "Reverse".  Pass any substring to filter.
        Passing an empty string returns all steps.
        """
        if not step_type:
            return list(self.steps)
        key = step_type.lower()
        return [s for s in self.steps if key in s.name.lower()]

    def __repr__(self) -> str:
        return (f"File(path={self.filepath!r}, "
                f"project={self.project!r}, "
                f"test={self.test!r}, "
                f"n_steps={len(self.steps)})")

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------

    def _parse(self) -> None:
        path = Path(self.filepath)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {self.filepath}")

        # Detect encoding from the BOM if present, then fall through common
        # RheoCompass encodings.  utf-16 must be tried first because its BOM
        # (FF FE) would be misread as valid latin-1 characters, producing
        # garbled null-padded text.  latin-1 is the catch-all final fallback
        # since it accepts any byte value without raising UnicodeDecodeError.
        _encodings = ["utf-16", "utf-8-sig", "utf-8", "cp1252", "latin-1"]
        lines = None
        for enc in _encodings:
            try:
                with open(path, "r", encoding=enc) as fh:
                    lines = [ln.rstrip("\r\n") for ln in fh]
                break
            except (UnicodeDecodeError, UnicodeError):
                continue
        if lines is None:
            with open(path, "r", encoding="utf-8", errors="replace") as fh:
                lines = [ln.rstrip("\r\n") for ln in fh]

        # Detect delimiter: tab-delimited and comma-delimited are both used
        # by RheoCompass depending on export settings and software version.
        # Sniff from the first non-empty line.
        _delim = ","
        for ln in lines:
            if ln.strip():
                _delim = "\t" if "\t" in ln else ","
                break

        i = 0
        n = len(lines)

        while i < n:
            line = lines[i]

            if line.startswith("Project:"):
                self.project = _csv_field(line, 1, delim=_delim)
                i += 1
                continue

            if line.startswith("Test:"):
                self.test = _csv_field(line, 1, delim=_delim)
                i += 1
                continue

            if line.startswith("Result:"):
                step_name = _csv_field(line, 1, delim=_delim)
                i += 1
                df, i = self._parse_result(lines, i, n, _delim)
                if df is not None:
                    self.steps.append(Step(step_name, df))
                continue

            i += 1

    def _parse_result(
        self, lines: List[str], start: int, n: int, delim: str = ","
    ) -> tuple[Optional[pd.DataFrame], int]:
        """
        Parse all interval blocks for one Result, returning a single
        concatenated DataFrame with a monotonic Step time column.
        """
        frames: List[pd.DataFrame] = []
        i = start
        interval_number = 0

        while i < n:
            line = lines[i]

            if line.startswith("Result:"):
                break

            if line.startswith("Interval and data points:"):
                parts = line.split(delim)
                try:
                    interval_number = int(parts[1])
                except (IndexError, ValueError):
                    interval_number += 1
                i += 1

                # Read "Interval data:" header line to build a column position map.
                # RheoCompass can export Shear Stress and Shear Strain in either
                # order depending on the template, so we must not assume position.
                #
                # Known header variants and their canonical names:
                #   "Shear Stress" / "Stress"   -> "Stress"
                #   "Shear Strain" / "Strain"   -> "Strain"
                #   "Time"                      -> "Time"
                #   "Interval Time"             -> "Interval Time"
                #   "Point No."                 -> "Point No."
                col_map: dict = {}   # canonical_name -> column_index
                if i < n and lines[i].startswith("Interval data:"):
                    header_parts = lines[i].split(delim)
                    _NAME_MAP = {
                        "shear stress":  "Stress",
                        "stress":        "Stress",
                        "shear strain":  "Strain",
                        "strain":        "Strain",
                        "time":          "Time",
                        "interval time": "Interval Time",
                        "point no.":     "Point No.",
                    }
                    for idx, h in enumerate(header_parts):
                        canonical = _NAME_MAP.get(h.strip().lower())
                        if canonical:
                            col_map[canonical] = idx
                    i += 1

                # Skip blank line(s)
                while i < n and lines[i].strip().strip(",") == "":
                    i += 1

                # Skip units line (e.g. ,,[s],[s],[Pa],[1])
                if i < n and re.match(r"^,*\[", lines[i]):
                    i += 1

                # Fall back to positional mapping if header parsing failed
                # (preserves compatibility with any unusual export formats)
                if not col_map:
                    col_map = {
                        "Point No.":     1,
                        "Time":          2,
                        "Interval Time": 3,
                        "Stress":        4,
                        "Strain":        5,
                    }

                # Collect data rows for this interval
                rows: List[dict] = []
                while i < n:
                    data_line = lines[i]
                    if (data_line.startswith("Interval and data points:")
                            or data_line.startswith("Result:")):
                        break
                    if data_line.strip().strip(",") == "":
                        i += 1
                        break
                    cols = data_line.split(delim)
                    required_cols = {"Time", "Stress", "Strain"}
                    if (all(k in col_map for k in required_cols)
                            and len(cols) > max(col_map[k] for k in required_cols)):
                        try:
                            rows.append({
                                "Interval":      interval_number,
                                "Point No.":     _to_float(cols[col_map["Point No."]]) if "Point No." in col_map else float("nan"),
                                "Time":          _to_float(cols[col_map["Time"]]),
                                "Interval Time": _to_float(cols[col_map["Interval Time"]]) if "Interval Time" in col_map else float("nan"),
                                "Stress":        _to_float(cols[col_map["Stress"]]),
                                "Strain":        _to_float(cols[col_map["Strain"]]),
                            })
                        except ValueError:
                            pass
                    i += 1

                if rows:
                    frames.append(pd.DataFrame(rows))
                continue

            i += 1

        if not frames:
            return None, i

        df = pd.concat(frames, ignore_index=True)

        # Build monotonic Step time: global Time offset to zero at the start
        # of this Result block.  The raw "Interval Time" column resets at each
        # interval boundary (non-monotonic), which would corrupt dt and FFT-
        # based frequency detection in the analysis code.  Using the global
        # Time column — which is continuous across intervals — and zeroing it
        # to the first point of this Result replicates the MATLAB behavior of
        # tracking time continuity across intervals within a measurement.
        t0 = df["Time"].iloc[0]
        df.insert(0, "Step time", df["Time"] - t0)

        return df, i


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _csv_field(line: str, index: int, default: str = "", delim: str = ",") -> str:
    parts = line.split(delim)
    try:
        return parts[index].strip()
    except IndexError:
        return default


def _to_float(s: str) -> float:
    return float(s.strip())


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python csvParser.py <path_to_csv>")
        sys.exit(1)

    f = File(sys.argv[1])
    print(f)
    print()

    for step in f.steps:
        print(f"  {step}")

    print()
    s0 = f.steps[0]
    t = s0.df["Step time"].values
    drops = [i for i in range(1, len(t)) if t[i] < t[i-1]]
    print(f"Step time monotonic: {len(drops) == 0}  (drops={len(drops)})")

    iv_counts = s0.df.groupby("Interval").size()
    print(f"\nInterval point counts (first step):")
    print(iv_counts.to_string())

    print("\nAround interval 1->2 boundary:")
    mask = s0.df["Interval"].isin([1, 2])
    sub = s0.df[mask][["Interval", "Step time", "Interval Time", "Stress", "Strain"]]
    print(sub.iloc[8:14].to_string())
    