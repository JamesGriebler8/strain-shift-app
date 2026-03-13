"""
txtParser.py
============
Parser for ARES tab-delimited .txt exports, providing the File / Step
interface used by strain_shift_analysis.py.

File format
-----------
A file-level header block (key<TAB>value lines) is followed by one or more
step blocks delimited by ``[step]``.  Each step block has the form:

    [step]
    <Step name> - <step number>
    <col1>\\t<col2>\\t...
    <unit1>\\t<unit2>\\t...
    <data rows...>

The step name (everything before the trailing " - <N>") is used as the
step type for filtering via ``get_steps_by_type``.

Column handling
---------------
Column names are used exactly as they appear in the file header row.
The units row is stored in ``step.units`` but is not part of ``step.df``.
Strain exported as % is NOT automatically converted here — leave that to
the caller (AnalysisConfig.percent_strain handles it in the analysis code).

Usage
-----
    from txtParser import File
    data = File("my_experiment.txt")
    arb_steps = data.get_steps_by_type("Arbitrary Wave")
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Step
# ---------------------------------------------------------------------------

class Step:
    """
    One step block from an ARES .txt export.

    Attributes
    ----------
    name : str
        Full step name including number, e.g. "Arbitrary Wave - 3".
    step_type : str
        Step type without the trailing number, e.g. "Arbitrary Wave".
    number : int
        The step sequence number extracted from the name.
    units : dict[str, str]
        Mapping of column name to unit string.
    df : pd.DataFrame
        All data rows for this step.  Column names match the file header.
    columns : list[str]
        Convenience accessor for df.columns.
    """

    def __init__(self, name: str, step_type: str, number: int,
                 df: pd.DataFrame, units: Dict[str, str]) -> None:
        self.name: str = name
        self.step_type: str = step_type
        self.number: int = number
        self.df: pd.DataFrame = df.reset_index(drop=True)
        self.units: Dict[str, str] = units

    @property
    def columns(self) -> List[str]:
        return list(self.df.columns)

    def __repr__(self) -> str:
        return (f"Step(name={self.name!r}, "
                f"n_points={len(self.df)}, "
                f"columns={self.columns})")


# ---------------------------------------------------------------------------
# File
# ---------------------------------------------------------------------------

class File:
    """
    Parsed ARES .txt export.

    Parameters
    ----------
    filepath : str or Path

    Attributes
    ----------
    filepath : str
    metadata : dict[str, str]
        File-level header fields (Filename, Instrument name, Operator, etc.).
    steps : list[Step]
        All steps in file order.
    """

    def __init__(self, filepath: str | Path) -> None:
        self.filepath: str = str(filepath)
        self.metadata: Dict[str, str] = {}
        self.steps: List[Step] = []
        self._parse()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def get_steps_by_type(self, step_type: str) -> List[Step]:
        """
        Return all steps whose type contains *step_type* (case-insensitive
        substring match on the step type string, i.e. the name without the
        trailing " - <N>").

        Examples
        --------
        >>> data.get_steps_by_type("Arbitrary Wave")
        >>> data.get_steps_by_type("Creep")
        """
        key = step_type.lower()
        return [s for s in self.steps if key in s.step_type.lower()]

    def __repr__(self) -> str:
        types = {}
        for s in self.steps:
            types[s.step_type] = types.get(s.step_type, 0) + 1
        type_summary = ", ".join(f"{t}×{n}" for t, n in types.items())
        return (f"File(path={self.filepath!r}, "
                f"n_steps={len(self.steps)}, [{type_summary}])")

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------

    def _parse(self) -> None:
        path = Path(self.filepath)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {self.filepath}")

        with open(path, "r", encoding="utf-8-sig") as fh:
            lines = [ln.rstrip("\r\n") for ln in fh]

        i = 0
        n = len(lines)

        while i < n:
            line = lines[i]

            if line == "[step]":
                step, i = self._parse_step(lines, i, n)
                if step is not None:
                    self.steps.append(step)
                continue

            # File-level metadata: "Key\tValue"
            if "\t" in line and not line.startswith("\t"):
                parts = line.split("\t", 1)
                self.metadata[parts[0].strip()] = parts[1].strip() if len(parts) > 1 else ""

            i += 1

    def _parse_step(self, lines: List[str], start: int, n: int
                    ) -> tuple[Optional[Step], int]:
        """
        Parse one step block beginning at the ``[step]`` line.
        Returns (Step, next_line_index).
        """
        i = start + 1  # move past [step]

        if i >= n:
            return None, i

        # ── Step name line ───────────────────────────────────────────
        name_line = lines[i].strip()
        i += 1

        # Parse "Type Name - N"  (number after the last " - ")
        step_type, number = _split_step_name(name_line)

        if i >= n:
            return None, i

        # ── Column header line ───────────────────────────────────────
        col_line = lines[i].strip()
        columns = col_line.split("\t")
        i += 1

        if i >= n:
            return None, i

        # ── Units line ───────────────────────────────────────────────
        unit_line = lines[i].strip()
        unit_values = unit_line.split("\t")
        units = {col: (unit_values[j] if j < len(unit_values) else "")
                 for j, col in enumerate(columns)}
        i += 1

        # ── Data rows ────────────────────────────────────────────────
        rows: List[List[float]] = []
        while i < n:
            line = lines[i]
            # A new step starts with [step]
            if line == "[step]":
                break
            # Skip blank lines
            if line.strip() == "":
                i += 1
                continue
            # Try to parse as numeric data
            parts = line.split("\t")
            try:
                row = [float(v) for v in parts]
                if len(row) == len(columns):
                    rows.append(row)
            except ValueError:
                # Non-numeric line (shouldn't normally appear mid-step)
                pass
            i += 1

        if not rows:
            return None, i

        df = pd.DataFrame(rows, columns=columns)
        step = Step(name=name_line, step_type=step_type,
                    number=number, df=df, units=units)
        return step, i


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _split_step_name(name: str) -> tuple[str, int]:
    """
    Split "Arbitrary Wave - 3" into ("Arbitrary Wave", 3).
    If the pattern doesn't match, return (name, 0).
    """
    # Split on the last occurrence of " - "
    idx = name.rfind(" - ")
    if idx == -1:
        return name.strip(), 0
    type_part = name[:idx].strip()
    num_part = name[idx + 3:].strip()
    try:
        return type_part, int(num_part)
    except ValueError:
        return name.strip(), 0


# ---------------------------------------------------------------------------
# Minimal self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python txtParser.py <path_to_txt>")
        sys.exit(1)

    f = File(sys.argv[1])
    print(f)
    print()
    for step in f.steps:
        print(f"  {step}")
    print()

    arb = f.get_steps_by_type("Arbitrary Wave")
    print(f"Arbitrary Wave steps: {len(arb)}")
    if arb:
        print(f"  First step columns : {arb[0].columns}")
        print(f"  First step units   : {arb[0].units}")
        print(f"  First 2 rows:\n{arb[0].df.head(2).to_string()}")