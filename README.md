# Strain Shift Analysis App

A local web app for analyzing oscillatory strain-shift rheology experiments. Supports TA Instruments (DHR) (tab-delimited `.txt`), RheoCompass/MCR (`.csv`), and DHR Excel exports (`.xlsx`).

NOTE: DHR works better and faster with .txt importer.

---

## Requirements

- Python 3.9+
- Anaconda or pip

## Install dependencies

```bash
pip install streamlit matplotlib pandas numpy scipy openpyxl
```

## Files needed (keep in the same folder)

```
app.py
strain_shift_analysis.py
txtParser.py
csvParser.py
xlsxParser.py
```

## Run

```bash
cd \Folder_Holding_all_files
streamlit run app.py
```

A browser window will open automatically at `http://localhost:8501`.

---

## Usage

1. Upload your data file (`.txt`, `.csv`, or `.xlsx`)
2. Select your instrument:
   - **DHR — TXT**: DHR tab-delimited export
   - **MCR — CSV**: Anton Paar RheoCompass export
   - **DHR — XLSX**: DHR Excel export
3. Set control mode (stress or strain), repeat tests per condition, and periods to average
4. Press **▶ Run analysis**
5. View results across the tabs — moduli vs strain, moduli vs stress, strain shift, diagnostics, and a full results table
6. Download results as CSV from the **Results table** tab

---

## Instrument notes

### DHR — TXT
- Strain exported as fractional — untick **"Strain exported as %"**
- Steps are identified as `Arbitrary Wave - N`

### MCR — CSV
- Strain exported as fractional (not %) — untick **"Strain exported as %"**
- Each Result block contains multiple intervals: pre-shear, oscillation cycles, and a recovery window — these are detected automatically from the `Interval` column
- Column order (Stress/Strain) is read from the file header and does not need to match any particular template

### DHR — XLSX
- Strain exported as fractional — untick **"Strain exported as %"**
- Each sheet is one step; non-data sheets (Details, Frequency sweep, Creep) are skipped automatically
- Large files (70+ sheets) take ~10 seconds to parse

---

## Analysis parameters

| Parameter | Description |
|---|---|
| Control mode | Whether stress or strain was the controlled variable |
| Repeat tests per condition | Number of forward/reverse pairs at each amplitude (typically 1 or 3) |
| Periods averaged for FT | Number of periods from the end of each step used for Fourier averaging (typically 3–5) |
| Strain exported as % | Tick for DHR/TXT; untick for MCR/CSV and DHR/XLSX |
| Step filter | Optional substring to filter step names (leave blank for default behavior) |

---

## Output quantities

| Column | Description |
|---|---|
| `Gprime [Pa]` | Storage modulus G' |
| `Gpprime [Pa]` | Loss modulus G'' |
| `Gpprime_fluid [Pa]` | Fluid contribution to G'' (strain-shift based) |
| `Gpprime_solid [Pa]` | Solid contribution to G'' (recoverable strain based) |
| `Strain Shift` | DC offset of strain waveform (indicator of yielding) |
| `Strain Amplitude` | Total strain amplitude |
| `Rec Strain Amplitude` | Recoverable strain amplitude |
| `Unrec Strain Amplitude` | Unrecoverable strain amplitude |
| `Phase Angle [rad]` | Total phase angle δ |
| `Rec Phase Angle [rad]` | Recoverable phase angle δ_rec |

---

## Notes

- Data never leaves your machine — the app runs entirely locally
- The app can be quit cleanly using the **⏹ Quit** button in the sidebar
- If you see a negative strain shift warning, this is expected for some materials and sign conventions — the absolute value is used automatically in fluid/solid modulus calculations
- For questions about the recovery rheology framework underlying this analysis, see the Rogers group publications on strain shift and yielding

---

## File structure

```
strain-shift-app/
├── app.py                    # Streamlit GUI
├── strain_shift_analysis.py  # Core analysis pipeline
├── txtParser.py              # DHR .txt parser
├── csvParser.py              # RheoCompass .csv parser
├── xlsxParser.py             # DHR .xlsx parser
└── README.md
```