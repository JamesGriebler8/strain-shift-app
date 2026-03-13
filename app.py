"""
app.py
======
Streamlit GUI for strain-shift rheology analysis.

Run with:
    streamlit run app.py

Requires strain_shift_analysis.py, txtParser.py, and csvParser.py
in the same directory.
"""

import io
import tempfile
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import streamlit as st

# ── Page config ────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Strain Shift Analysis",
    page_icon="📊",
    layout="wide",
)

# ── Minimal styling ─────────────────────────────────────────────────────────

st.markdown("""
<style>
    /* Tighten sidebar */
    section[data-testid="stSidebar"] { min-width: 300px; max-width: 340px; }
    /* Subtle card look for expanders */
    div[data-testid="stExpander"] { border: 1px solid #e0e0e0; border-radius: 6px; }
    /* Monospace for warnings */
    .warn-box {
        background: #fff8e1;
        border-left: 4px solid #f9a825;
        padding: 0.6em 1em;
        border-radius: 4px;
        font-family: monospace;
        font-size: 0.85em;
        white-space: pre-wrap;
    }
    .error-box {
        background: #ffebee;
        border-left: 4px solid #c62828;
        padding: 0.6em 1em;
        border-radius: 4px;
        font-size: 0.9em;
    }
</style>
""", unsafe_allow_html=True)


# ── Helpers ─────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def run_analysis(file_bytes: bytes, filename: str, instrument: str,
                 control_type: str, repeat_tests: int,
                 periods_for_ft: int, percent_strain: bool,
                 step_filter: str) -> tuple:
    """Full analysis pipeline (cached)."""
    from strain_shift_analysis import StrainShiftExperiment, AnalysisConfig, _load_file

    suffix = Path(filename).suffix
    warnings_caught = []

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    import sys
    old_stdout = sys.stdout
    sys.stdout = captured = io.StringIO()

    try:
        data = _load_file(tmp_path, instrument)
        cfg = AnalysisConfig(
            instrument=instrument,
            control_type=control_type,
            repeat_tests=repeat_tests,
            periods_for_ft=periods_for_ft,
            percent_strain=percent_strain,
            step_filter=step_filter,
        )
        exp = StrainShiftExperiment(data, cfg)
        exp.process()
        df = exp.summary()
    finally:
        sys.stdout = old_stdout
        Path(tmp_path).unlink(missing_ok=True)

    printed = captured.getvalue().strip()
    if printed:
        warnings_caught = [l for l in printed.splitlines() if l.strip()]

    return df, warnings_caught, exp


def make_figure(df: pd.DataFrame, x_col: str, x_label: str) -> plt.Figure:
    """All four moduli on one log-log plot."""
    fig, ax = plt.subplots(figsize=(7, 5))

    x = df[x_col].values
    series = [
        ("Gprime [Pa]",        "G'",             "o", "#1f77b4"),
        ("Gpprime [Pa]",       "G''",            "s", "#d62728"),
        ("Gpprime_fluid [Pa]", "G''$_{fluid}$",  "^", "#2ca02c"),
        ("Gpprime_solid [Pa]", "G''$_{solid}$",  "D", "#9467bd"),
    ]

    for col, label, marker, color in series:
        ax.plot(x, df[col].values, marker + "-", color=color,
                markersize=6, linewidth=1.6, label=label)

    ax.set_xlabel(x_label, fontsize=11)
    ax.set_ylabel("Modulus [Pa]", fontsize=11)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.xaxis.set_major_formatter(ticker.LogFormatterSciNotation())
    ax.yaxis.set_major_formatter(ticker.LogFormatterSciNotation())
    ax.grid(True, which="both", ls=":", alpha=0.4)
    ax.legend(fontsize=10)
    ax.set_title(f"Moduli vs {x_label}", fontsize=12, fontweight="bold")
    fig.tight_layout()
    return fig


def make_shift_figure(df: pd.DataFrame, x_col: str, x_label: str) -> plt.Figure:
    """Strain shift vs amplitude, log-log."""
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(df[x_col].values,
            abs(df["Strain Shift"].values),
            "s-", color="#e377c2", markersize=7, linewidth=1.8)
    ax.set_xlabel(x_label, fontsize=11)
    ax.set_ylabel("Strain Shift", fontsize=11)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.xaxis.set_major_formatter(ticker.LogFormatterSciNotation())
    ax.yaxis.set_major_formatter(ticker.LogFormatterSciNotation())
    ax.grid(True, which="both", ls=":", alpha=0.4)
    ax.set_title(f"Strain Shift vs {x_label}", fontsize=11, fontweight="bold")
    fig.tight_layout()
    return fig


def make_diagnostic_figure(exp) -> plt.Figure:
    """Strain shift per period for each paired test."""
    from strain_shift_analysis import AnalysisConfig

    cfg = exp.config
    n_ft = cfg.periods_for_ft

    fig, ax = plt.subplots(figsize=(8, 4))
    cmap = plt.get_cmap("tab10")

    for i, pair in enumerate(exp.pairs):
        # Recompute per-period strain shift from data_per_period
        n_periods = pair.n_periods
        shifts = np.array([
            (pair.data_per_period[:, 2, p].max() +
             pair.data_per_period[:, 2, p].min()) / 2
            for p in range(n_periods)
        ])
        label = f"Pair {i}"
        color = cmap(i % 10)
        ax.plot(np.arange(1, n_periods + 1), shifts,
                "o-", color=color, markersize=4, linewidth=1.2,
                alpha=0.7, label=label)
        # Shade the averaging window
        if n_periods >= n_ft:
            ax.axvspan(n_periods - n_ft + 0.5, n_periods + 0.5,
                       color=color, alpha=0.08)

    ax.set_xlabel("Period index", fontsize=10)
    ax.set_ylabel("Strain Shift (per period)", fontsize=10)
    ax.set_title("Diagnostic: Strain Shift Convergence", fontsize=11, fontweight="bold")
    ax.grid(True, ls=":", alpha=0.4)
    if len(exp.pairs) <= 12:
        ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    return fig


# ── Sidebar ─────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("⚙️ Settings")

    uploaded = st.file_uploader(
        "Upload data file",
        type=["txt", "csv", "xlsx", "xls"],
        help=".txt for ARES/DHR tab-delimited exports, .csv for RheoCompass (MCR), .xlsx/.xls for ARES/DHR Excel exports",
    )

    st.divider()

    INSTRUMENT_LABELS = ["DHR — TXT", "MCR — CSV", "DHR — XLSX"]
    INSTRUMENT_KEYS   = ["DHR",        "MCR",        "XLSX"]
    _inst_choice = st.radio(
        "Instrument / format",
        options=INSTRUMENT_LABELS,
        horizontal=False,
    )
    instrument = INSTRUMENT_KEYS[INSTRUMENT_LABELS.index(_inst_choice)]

    control_type = st.radio(
        "Control mode",
        options=["stress", "strain"],
        horizontal=True,
    )

    st.divider()
    st.subheader("Analysis parameters")

    repeat_tests = st.number_input(
        "Repeat tests per condition",
        min_value=1, max_value=10, value=3, step=1,
        help="How many forward/reverse pairs were repeated at each amplitude",
    )

    periods_for_ft = st.number_input(
        "Periods averaged for FT",
        min_value=1, max_value=20, value=3, step=1,
        help="Number of periods (from end of each test) used for Fourier averaging",
    )

    percent_strain = st.checkbox(
        "Strain exported as %",
        #value=(instrument != "XLSX"),
        help="Tick if the strain column is in percent (typical for ARES/DHR .txt). "
    )         
    st.divider()
    st.subheader("Advanced")

    step_filter = st.text_input(
        "Step filter (optional)",
        value="",
        help='Leave blank for auto (DHR → "Arbitrary Wave", MCR → all steps). '
             'Set to any substring to filter by step name.',
    )

    run_btn = st.button("▶  Run analysis", type="primary",
                        disabled=(uploaded is None))

    st.divider()
    if st.button("⏹  Quit", help="Shut down the Streamlit server"):
        st.info("Server shutting down…")
        import os, signal
        os.kill(os.getpid(), signal.SIGTERM)

# ── Main area ────────────────────────────────────────────────────────────────

st.title("Strain Shift Rheology Analysis")

if uploaded is None:
    st.info("Upload a data file in the sidebar to get started.")
    st.stop()

if not run_btn and "last_result" not in st.session_state:
    st.info("Configure settings in the sidebar, then press **Run analysis**.")
    st.stop()

# Run (or use cached result if settings haven't changed)
if run_btn:
    file_bytes = uploaded.read()
    progress = st.progress(0, text="Reading file…")
    status   = st.empty()
    try:
        # Stage 1 — parse file
        progress.progress(15, text="Parsing data file…")
        # We need the step count for the label; peek without caching
        import tempfile as _tf, sys as _sys
        from pathlib import Path as _Path
        from strain_shift_analysis import _load_file as _lf
        _suffix = _Path(uploaded.name).suffix
        with _tf.NamedTemporaryFile(suffix=_suffix, delete=False) as _tmp:
            _tmp.write(file_bytes)
            _tmp_path = _tmp.name
        try:
            _data = _lf(_tmp_path, instrument)
            _n = len(_data.get_steps_by_type("Arbitrary Wave" if instrument in ("DHR", "XLSX") else ""))
        finally:
            _Path(_tmp_path).unlink(missing_ok=True)

        progress.progress(40, text=f"File loaded — {_n} oscillation steps found. Running analysis…")

        # Stage 2 — full cached analysis
        df, warnings, exp = run_analysis(
            file_bytes, uploaded.name, instrument,
            control_type, int(repeat_tests), int(periods_for_ft),
            percent_strain, step_filter,
        )
        progress.progress(85, text="Building results…")
        st.session_state["last_result"] = (df, warnings, exp)
        progress.progress(100, text="Done!")
        progress.empty()
        status.empty()
    except Exception:
        progress.empty()
        status.empty()
        st.markdown(
            f'<div class="error-box"><strong>Analysis failed:</strong><br>'
            f'{traceback.format_exc()}</div>',
            unsafe_allow_html=True,
        )
        st.stop()

df, warnings, exp = st.session_state["last_result"]

# ── Warnings ────────────────────────────────────────────────────────────────

if warnings:
    with st.expander(f"⚠️  {len(warnings)} warning(s) — click to expand", expanded=True):
        for w in warnings:
            st.markdown(f'<div class="warn-box">{w}</div>', unsafe_allow_html=True)

# ── Data summary ─────────────────────────────────────────────────────────────

n_pairs = len(df)
st.success(
    f"✓ Loaded **{uploaded.name}** — "
    f"**{n_pairs}** paired condition{'s' if n_pairs != 1 else ''} found."
)

# ── Tabs ────────────────────────────────────────────────────────────────────

tab_strain, tab_stress, tab_shift, tab_diag, tab_data = st.tabs([
    "📈 Moduli vs Strain",
    "📈 Moduli vs Stress",
    "📉 Strain Shift",
    "🔬 Diagnostic",
    "📋 Results table",
])

with tab_strain:
    fig = make_figure(df, "Strain Amplitude", "Strain Amplitude")
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

with tab_stress:
    fig = make_figure(df, "Stress Amplitude [Pa]", "Stress Amplitude [Pa]")
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

with tab_shift:
    col1, col2 = st.columns(2)
    with col1:
        fig = make_shift_figure(df, "Stress Amplitude [Pa]", "Stress Amplitude [Pa]")
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
    with col2:
        fig = make_shift_figure(df, "Strain Amplitude", "Strain Amplitude")
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

with tab_diag:
    st.caption(
        "Per-period strain shift for each forward/reverse pair. "
        "Shaded region = averaging window used for final result."
    )
    fig = make_diagnostic_figure(exp)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

with tab_data:
    st.dataframe(df, use_container_width=True)

    csv_bytes = df.to_csv(index=False).encode()
    stem = Path(uploaded.name).stem
    st.download_button(
        label="⬇  Download CSV",
        data=csv_bytes,
        file_name=f"{stem}_analyzed.csv",
        mime="text/csv",
    )

# ── Footer ───────────────────────────────────────────────────────────────────

st.divider()
st.caption("Strain Shift Analysis · Dr. James J Griebler, updated March 13th 2026")