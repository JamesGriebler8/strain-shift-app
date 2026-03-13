"""
strain_shift_analysis.py
========================
Analysis of oscillatory strain-shift rheology experiments.

Works directly with data imported by txtParser.py (File / Step).

Workflow
--------
1. Identify paired Arbitrary Wave steps (forward + reverse, consecutive)
2. For each pair:
   a. Find the padding region (stress == 0) and compute baseline strain
   b. Auto-detect the oscillation frequency via FFT of the stress signal
   c. Average forward and reverse waveforms to cancel directional bias
   d. Trim to whole periods only, subtract baseline strain
3. Run per-period Fourier analysis to extract amplitudes, strain shift,
   and dynamic moduli (G', G'') or compliances (J', J'')
4. Average results over repeat tests and the last N periods

Usage
-----
    from txtParser import File
    from strain_shift_analysis import StrainShiftExperiment, AnalysisConfig

    data = File("my_experiment.txt")
    cfg  = AnalysisConfig(control_type="stress", repeat_tests=1, periods_for_ft=3)
    exp  = StrainShiftExperiment(data, cfg)
    exp.process()

    print(exp.summary())
    exp.export("results.npz")
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

def _load_file(filepath: str, instrument: str) -> "File":
    """
    Instantiate the correct parser based on instrument type.

    Parameters
    ----------
    filepath : str
        Path to the data file.
    instrument : str
        ``"MCR"``   → RheoCompass CSV export      (csvParser.File)
        ``"DHR"``   → ARES/DHR tab-delimited .txt (txtParser.File)
        ``"XLSX"``  → ARES/DHR Excel export       (xlsxParser.File)
    """
    instrument = instrument.upper()
    if instrument == "MCR":
        from csvParser import File
    elif instrument == "DHR":
        from txtParser import File
    elif instrument == "XLSX":
        from xlsxParser import File
    else:
        raise ValueError(
            f"Unknown instrument {instrument!r}. "
            "Expected 'MCR' (RheoCompass CSV), 'DHR' (ARES/DHR txt), "
            "or 'XLSX' (ARES/DHR Excel)."
        )
    return File(filepath)


# Re-export Step from whichever parser is available for type annotations.
try:
    from txtParser import Step
except ImportError:
    try:
        from csvParser import Step
    except ImportError:
        Step = object  # type: ignore


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class AnalysisConfig:
    """
    Parameters that control the analysis.

    Attributes
    ----------
    control_type : str
        'stress' or 'strain' — which quantity was controlled during the experiment.
    repeat_tests : int
        How many times the same amplitude/frequency condition was repeated (usually 1 or 3).
    periods_for_ft : int
        Number of periods (from the end of each test) to average over for the
        final Fourier results. Must be at least 2 fewer than the total number
        of available periods.
    percent_strain : bool
        True if strain is exported in % (typical for ARES/DHR). Will be
        converted to fractional automatically.
    """
    control_type: str = "stress"
    repeat_tests: int = 1
    periods_for_ft: int = 3
    percent_strain: bool = True
    recovery_points: int = 10
    """Number of points at the end of the recovery window to average for unrec strain."""
    instrument: str = "DHR"
    """
    Instrument / file format selector.
        ``"DHR"``  – ARES / DHR tab-delimited .txt export  (txtParser)
        ``"MCR"``  – Anton Paar RheoCompass CSV export     (csvParser)
    """
    step_filter: str = ""
    """
    Substring passed to ``get_steps_by_type()`` to select oscillation steps.
    Defaults to ``"Arbitrary Wave"`` for DHR and ``""`` (all steps) for MCR
    when left as an empty string.  Set explicitly to override.
    """


# =============================================================================
# Internal data structures
# =============================================================================

@dataclass
class ProcessedPair:
    """
    Holds the averaged, baseline-subtracted data for one forward/reverse pair,
    separated into individual periods.

    Attributes
    ----------
    pair_index : int
        0-based index of this pair among all pairs.
    forward_step : Step
    reverse_step : Step
    freq_hz : float
        Auto-detected oscillation frequency [Hz].
    ang_freq : float
        Angular frequency [rad/s].
    period_length : int
        Number of data points per period.
    skip_points : int
        Number of leading padding points removed.
    baseline_strain : float
        Mean strain during the padding region, subtracted from all strain values.
    data_per_period : np.ndarray
        Shape (period_length, 3, n_periods).
        Axis 1: [time, stress, strain]
    n_periods : int
        Number of whole periods extracted.
    """
    pair_index: int
    forward_step: Step
    reverse_step: Step
    freq_hz: float
    ang_freq: float
    period_length: int
    skip_points: int
    baseline_strain: float
    data_per_period: np.ndarray   # (period_length, 3, n_periods)
    n_periods: int
    unrec_strain: float
    """Unrecoverable strain = mean of last recovery_points of recovery window / 2."""


@dataclass
class PairedTestResult:
    """
    Final averaged result for one paired condition (one stress amplitude /
    frequency combination), after averaging over repeat tests and periods.

    All moduli in [Pa], strains in [fractional], stress in [Pa].
    """
    paired_index: int
    freq_hz: float
    avg_stress_amp: float
    avg_strain_amp: float
    avg_strain_shift: float       # DC offset of strain
    avg_unrec_strain: float       # unrecovered strain (half-period mean at end)
    avg_rec_strain_amp: float     # recoverable strain amplitude
    phase_angle: float            # total phase angle [rad]
    rec_phase_angle: float        # recoverable phase angle [rad]
    G_prime: float                # storage modulus [Pa]
    G_double_prime: float         # loss modulus [Pa]
    J_prime: float                # storage compliance [1/Pa]
    J_double_prime: float         # loss compliance [1/Pa]
    fluid_loss_modulus: float
    solid_loss_modulus: float
    fluid_loss_compliance: float
    solid_loss_compliance: float
    energy_stored: float
    energy_dissipated: float


# =============================================================================
# Main experiment class
# =============================================================================

class StrainShiftExperiment:
    """
    Full analysis pipeline for a strain-shift oscillatory rheology experiment.

    Parameters
    ----------
    ares_file : File
        Parsed data from txtParser.py.
    config : AnalysisConfig, optional
        Analysis parameters. Defaults to AnalysisConfig().
    """

    def __init__(self, ares_file: File, config: Optional[AnalysisConfig] = None):
        self.ares_file = ares_file
        self.config = config or AnalysisConfig()
        self.pairs: List[ProcessedPair] = []
        self.results: List[PairedTestResult] = []
        self._processed = False

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def process(self) -> None:
        """Run the full analysis pipeline."""
        # Resolve which step type to filter on
        _filter = self.config.step_filter
        if not _filter:
            _filter = "Arbitrary Wave" if self.config.instrument.upper() in ("DHR", "XLSX") else ""
        osc_steps = self.ares_file.get_steps_by_type(_filter)
        if len(osc_steps) < 2:
            raise ValueError(
                f"Need at least one forward/reverse pair of oscillation steps "
                f"(filter={_filter!r}), but found {len(osc_steps)}."
            )
        if len(osc_steps) % 2 != 0:
            warnings.warn(
                f"Odd number of oscillation steps ({len(osc_steps)}) with "
                f"filter={_filter!r}. Last step will be ignored."
            )

        # Process each consecutive forward/reverse pair
        for k in range(0, len(osc_steps) - 1, 2):
            pair = self._process_pair(k // 2, osc_steps[k], osc_steps[k + 1])
            self.pairs.append(pair)

        # Average over repeats and compute final moduli
        self._compute_results()
        self._processed = True

    def summary(self) -> pd.DataFrame:
        """Return a DataFrame of final per-paired-test results."""
        if not self._processed:
            raise RuntimeError("Call .process() before .summary()")
        rows = []
        for r in self.results:
            tan_d     = np.tan(r.phase_angle)
            tan_d_rec = np.tan(r.rec_phase_angle)
            rows.append({
                "Frequency [rad/s]":       r.freq_hz * 2 * np.pi,
                "Strain Amplitude":        r.avg_strain_amp,
                "Unrec Strain Amplitude":  r.avg_unrec_strain,
                "Rec Strain Amplitude":    r.avg_rec_strain_amp,
                "tan(delta)":              tan_d,
                "tan(delta_rec)":          tan_d_rec,
                "Gprime [Pa]":             r.G_prime,
                "Gpprime [Pa]":            r.G_double_prime,
                "Gpprime_fluid [Pa]":      r.fluid_loss_modulus,
                "Gpprime_solid [Pa]":      r.solid_loss_modulus,
                "Stress Amplitude [Pa]":   r.avg_stress_amp,
                "Strain Shift":            r.avg_strain_shift,
                "Phase Angle [rad]":       r.phase_angle,
                "Rec Phase Angle [rad]":   r.rec_phase_angle,
                "Jprime [1/Pa]":           r.J_prime,
                "Jpprime [1/Pa]":          r.J_double_prime,
                "Jprime_fluid [1/Pa]":     r.fluid_loss_compliance,
                "Jprime_solid [1/Pa]":     r.solid_loss_compliance,
                "Energy Stored [Pa]":      r.energy_stored,
                "Energy Dissipated [Pa/s]": r.energy_dissipated,
            })
        return pd.DataFrame(rows)

    def export(self, filepath: str = None) -> None:
        """
        Save results to a CSV file.

        Parameters
        ----------
        filepath : str, optional
            Output path. If not provided, saves next to the source .txt file
            with '_analyzed.csv' appended to the stem.
        """
        if not self._processed:
            raise RuntimeError("Call .process() before .export()")
        if filepath is None:
            src = Path(self.ares_file.filepath)
            filepath = str(src.parent / (src.stem + "_analyzed.csv"))
        df = self.summary()
        df.to_csv(filepath, index=False)
        print(f"Results saved to {filepath}")

    # ------------------------------------------------------------------
    # Step 1: process a single forward/reverse pair
    # ------------------------------------------------------------------

    def _process_pair(
        self, pair_index: int, fwd: Step, rev: Step
    ) -> ProcessedPair:
        fwd_arr = self._step_to_array(fwd)   # (N, 3): time, stress, strain
        rev_arr = self._step_to_array(rev)

        # ── Boundary detection ────────────────────────────────────────────
        # MCR files use structural intervals and never have exactly-zero stress,
        # so the stress-zero heuristic is unreliable.  Use the Interval column
        # instead: interval 1 = pre-shear (skip), intervals 2..N-1 = oscillation,
        # interval N = recovery.
        #
        # DHR/XLSX files have a zero-padded pre-oscillation region and a
        # trailing zero-stress recovery window, so the stress-zero heuristic
        # works correctly for those formats.
        is_mcr = self.config.instrument.upper() == "MCR"

        if is_mcr:
            skip_points, end_point_fwd, rec_strain_fwd = self._mcr_boundaries(fwd)
            _,           end_point_rev, rec_strain_rev = self._mcr_boundaries(rev)
            # Baseline: mean strain in interval 1 of forward step
            baseline_strain = float(np.mean(fwd_arr[:skip_points, 2]))
        else:
            skip_f = self._find_skip_points(fwd_arr[:, 1])
            skip_r = self._find_skip_points(rev_arr[:, 1])
            skip_points = max(skip_f, skip_r)
            baseline_strain = float(np.mean(fwd_arr[:skip_points, 2]))

        # Auto-detect frequency from the stress signal in the oscillation region
        dt = float(fwd_arr[1, 0] - fwd_arr[0, 0])
        if is_mcr:
            osc_stress = fwd_arr[skip_points:end_point_fwd, 1]
        else:
            osc_stress = fwd_arr[skip_points:, 1]
        freq_hz, period_length = self._detect_frequency(osc_stress, dt)
        ang_freq = 2 * np.pi * freq_hz

        # Align forward and reverse to the same length
        fwd_arr, rev_arr = self._align_lengths(fwd_arr, rev_arr)

        # Average forward and reverse waveforms
        avg = np.zeros_like(fwd_arr)
        avg[:, 0] = fwd_arr[:, 0]
        avg[:, 1] = (fwd_arr[:, 1] - rev_arr[:, 1]) / 2
        avg[:, 2] = (fwd_arr[:, 2] - rev_arr[:, 2]) / 2

        # Subtract baseline strain and convert % -> fraction if needed
        avg[:, 2] -= baseline_strain
        if self.config.percent_strain:
            avg[:, 2] /= 100.0

        # ── end_point: last row of oscillation data ───────────────────────
        if is_mcr:
            # Use the interval boundary directly; average fwd/rev end points
            end_point = (end_point_fwd + min(end_point_rev, len(avg))) // 2
        else:
            # DHR/XLSX: find last non-zero stress row
            last_nonzero = np.flatnonzero(avg[:, 1] != 0)
            end_point = int(last_nonzero[-1]) + 1 if len(last_nonzero) > 0 else len(avg)

        # Trim to whole periods after the padding, stopping at end_point
        signal = avg[skip_points:end_point]
        n_whole_periods = len(signal) // period_length
        signal = signal[: n_whole_periods * period_length]

        # Reshape to (period_length, 3, n_periods)
        data_per_period = signal.reshape(n_whole_periods, period_length, 3)
        data_per_period = data_per_period.transpose(1, 2, 0)

        # ── Unrecoverable strain ──────────────────────────────────────────
        n_rec = self.config.recovery_points
        if is_mcr:
            # Average the fwd/rev recovery strain (already in fractional units
            # for MCR — percent_strain=False is expected for this format)
            rec_strain = (rec_strain_fwd - rec_strain_rev) / 2
            recovery_strain = rec_strain
        else:
            recovery_strain = avg[end_point:, 2]

        if len(recovery_strain) >= n_rec:
            unrec_strain = float(np.mean(recovery_strain[-n_rec:])) / 2
        elif len(recovery_strain) > 0:
            unrec_strain = float(np.mean(recovery_strain)) / 2
        else:
            unrec_strain = 0.0

        return ProcessedPair(
            pair_index=pair_index,
            forward_step=fwd,
            reverse_step=rev,
            freq_hz=freq_hz,
            ang_freq=ang_freq,
            period_length=period_length,
            skip_points=skip_points,
            baseline_strain=baseline_strain,
            data_per_period=data_per_period,
            n_periods=n_whole_periods,
            unrec_strain=unrec_strain,
        )

    # ------------------------------------------------------------------
    # Step 2: Fourier analysis per period, then average over repeats
    # ------------------------------------------------------------------

    def _compute_results(self) -> None:
        cfg = self.config
        n_pairs = len(self.pairs)
        n_paired_tests = n_pairs // cfg.repeat_tests

        for pt_idx in range(n_paired_tests):
            repeat_indices = list(range(
                pt_idx * cfg.repeat_tests,
                (pt_idx + 1) * cfg.repeat_tests,
            ))

            all_strain_amp, all_stress_amp = [], []
            all_strain_shift, all_moduli, all_unrec = [], [], []

            for ri in repeat_indices:
                sa_ft, ta_ft, ss_ft, mod_ft = self._ft_analysis_all_periods(self.pairs[ri])
                all_strain_amp.append(sa_ft)
                all_stress_amp.append(ta_ft)
                all_strain_shift.append(ss_ft)
                all_moduli.append(mod_ft)
                all_unrec.append(self.pairs[ri].unrec_strain)

            n_ft = cfg.periods_for_ft

            def tail_mean(arrays):
                return float(np.mean([a[-n_ft:] for a in arrays]))

            avg_sa    = tail_mean(all_strain_amp)
            avg_ta    = tail_mean(all_stress_amp)
            avg_ss    = tail_mean(all_strain_shift)

            if avg_ss < 0:
                print(
                    f"Warning [paired test {pt_idx}]: average strain shift is "
                    f"negative ({avg_ss:.4e}). This may indicate a sign convention "
                    f"difference or waveform asymmetry. Absolute value will be used "
                    f"in fluid/solid loss modulus calculations."
                )
            avg_unrec = abs(avg_ss)#float(np.mean(all_unrec))
            avg_G     = float(np.mean([m[-n_ft:, 0] for m in all_moduli]))
            avg_Gd    = float(np.mean([m[-n_ft:, 1] for m in all_moduli]))

            ang_freq = self.pairs[repeat_indices[0]].ang_freq
            phi      = float(np.arctan2(avg_Gd, avg_G))
            phi_rec  = float(np.arctan2(
                avg_sa * np.sin(phi) - abs(avg_ss),
                avg_sa * np.cos(phi),
            ))
            rec_sa = (float(avg_sa * np.cos(phi) / np.cos(phi_rec))
                      if np.cos(phi_rec) != 0 else 0.0)

            denom           = avg_G ** 2 + avg_Gd ** 2
            J_prime         = avg_G  / denom if denom != 0 else 0.0
            J_dprime        = avg_Gd / denom if denom != 0 else 0.0
            fluid_loss_mod  = avg_ta * abs(avg_ss) / avg_sa ** 2 if avg_sa != 0 else 0.0
            solid_loss_mod  = (avg_ta * abs(rec_sa * np.sin(phi_rec)) / avg_sa ** 2
                               if avg_sa != 0 else 0.0)
            fluid_loss_comp = fluid_loss_mod  / denom if denom != 0 else 0.0
            solid_loss_comp = solid_loss_mod  / denom if denom != 0 else 0.0

            self.results.append(PairedTestResult(
                paired_index=pt_idx,
                freq_hz=self.pairs[repeat_indices[0]].freq_hz,
                avg_stress_amp=avg_ta,
                avg_strain_amp=avg_sa,
                avg_strain_shift=avg_ss,
                avg_unrec_strain=avg_unrec,
                avg_rec_strain_amp=rec_sa,
                phase_angle=phi,
                rec_phase_angle=phi_rec,
                G_prime=avg_G,
                G_double_prime=avg_Gd,
                J_prime=J_prime,
                J_double_prime=J_dprime,
                fluid_loss_modulus=fluid_loss_mod,
                solid_loss_modulus=solid_loss_mod,
                fluid_loss_compliance=fluid_loss_comp,
                solid_loss_compliance=solid_loss_comp,
                energy_stored=0.25 * avg_sa ** 2 * avg_G,
                energy_dissipated=0.50 * avg_sa ** 2 * avg_Gd * ang_freq,
            ))

    def _ft_analysis_all_periods(
        self, pair: ProcessedPair
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Run Fourier analysis on every period of a processed pair.

        Returns
        -------
        strain_amp_ft   : (n_periods,)
        stress_amp_ft   : (n_periods,)
        strain_shift_ft : (n_periods,)
        moduli          : (n_periods, 2)  [G', G''] or normalised [J', J'']
        """
        dpp = pair.data_per_period    # (period_length, 3, n_periods)
        n_periods = pair.n_periods
        n_pt = pair.period_length

        strain_amp_ft   = np.zeros(n_periods)
        stress_amp_ft   = np.zeros(n_periods)
        strain_shift_ft = np.zeros(n_periods)
        moduli          = np.zeros((n_periods, 2))

        for j in range(n_periods):
            stress_data = dpp[:, 1, j]
            strain_data = dpp[:, 2, j]

            strain_fft = np.fft.fft(strain_data)
            stress_fft = np.fft.fft(stress_data)
            n_osc = 1   # fundamental harmonic

            stress_amp = (2 / n_pt) * abs(stress_fft[n_osc])
            strain_amp = (2 / n_pt) * abs(strain_fft[n_osc])
            strain_amp_ft[j]   = strain_amp
            stress_amp_ft[j]   = stress_amp
            strain_shift_ft[j] = (strain_data.max() + strain_data.min()) / 2

            # Phase difference between stress and strain fundamentals
            phase_stress = np.angle(stress_fft[n_osc])
            phase_strain = np.angle(strain_fft[n_osc])
            delta = phase_stress - phase_strain   # stress leads strain by delta

            # G* resolved into storage and loss components
            G_star = stress_amp / strain_amp if strain_amp != 0 else 0.0
            if self.config.control_type == "strain":
                moduli[j] = [G_star * np.cos(delta), G_star * np.sin(delta)]
            else:
                # Stress-controlled: compute J* then invert
                J_star = strain_amp / stress_amp if stress_amp != 0 else 0.0
                J_prime  =  J_star * np.cos(delta)
                J_dprime = -J_star * np.sin(delta)
                denom = J_prime**2 + J_dprime**2
                if denom != 0:
                    moduli[j] = [J_prime / denom, -J_dprime / denom]

        return strain_amp_ft, stress_amp_ft, strain_shift_ft, moduli

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    # Maps known column name variants (lowercased) to the three canonical
    # names the analysis expects.  Add entries here if a new instrument
    # export template uses different labels.
    _COLUMN_ALIASES: dict = {
        # Time — note: "time" and "t" are intentionally omitted because MCR
        # files already have a canonical "Step time" column alongside a raw
        # "Time" column; mapping the latter would clobber the correct one.
        "step time":            "Step time",
        # Stress
        "stress":               "Stress",
        "shear stress":         "Stress",
        "tau":                  "Stress",
        "sigma":                "Stress",
        "oscillation stress":   "Stress",
        # Strain
        "strain":               "Strain",
        "shear strain":         "Strain",
        "gamma":                "Strain",
        "oscillation strain":   "Strain",
        "displacement":         "Strain",
    }

    @staticmethod
    def _normalise_columns(step: Step) -> pd.DataFrame:
        """
        Return step.df (or a renamed copy) with column names mapped to the
        canonical set: ``"Step time"``, ``"Stress"``, ``"Strain"``.

        Only columns whose lowercased, stripped name appears in
        ``_COLUMN_ALIASES`` are renamed; all others are left unchanged.
        A warning is issued if any renaming takes place so the caller is
        aware that non-standard labels were detected.
        """
        aliases = StrainShiftExperiment._COLUMN_ALIASES
        rename_map = {}
        for col in step.df.columns:
            canonical = aliases.get(col.strip().lower())
            if canonical and canonical != col:
                rename_map[col] = canonical
        if rename_map:
            warnings.warn(
                f"Step '{step.name}': non-standard column names detected — "
                f"renaming {rename_map} to canonical names for analysis.",
                stacklevel=3,
            )
            return step.df.rename(columns=rename_map)
        return step.df

    @staticmethod
    def _step_to_array(step: Step) -> np.ndarray:
        """
        Extract [Step time, Stress, Strain] columns as a (N, 3) numpy array.

        Column names are normalised against ``_COLUMN_ALIASES`` before the
        required-column check, so minor naming differences across instrument
        export templates are handled gracefully.  If a required column is
        still missing after normalisation, a descriptive error is raised that
        names the missing columns, lists what is available, and points to
        ``_COLUMN_ALIASES`` as the place to add a new mapping.
        """
        df = StrainShiftExperiment._normalise_columns(step)
        required = {"Step time", "Stress", "Strain"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(
                f"Step '{step.name}' is missing required columns: {missing}.\n"
                f"  Available columns : {list(df.columns)}\n"
                f"  If the column exists under a different name, add an entry "
                f"to StrainShiftExperiment._COLUMN_ALIASES."
            )
        return df[["Step time", "Stress", "Strain"]].to_numpy(dtype=float)

    @staticmethod
    def _mcr_boundaries(step: Step) -> Tuple[int, int, np.ndarray]:
        """
        Return (skip_points, end_point, recovery_strain) for an MCR step by
        reading the ``Interval`` column directly, rather than relying on the
        stress-zero heuristic (which doesn't work for MCR because the stress
        signal is never exactly zero).

        MCR interval structure
        ----------------------
        Interval 1        : pre-shear / equilibration — skip entirely.
        Intervals 2 to N-1: oscillation data — used for Fourier analysis.
        Interval N        : recovery window — strain tail used for unrec_strain.

        Parameters
        ----------
        step : Step
            A single MCR Step (must have an ``"Interval"`` column).

        Returns
        -------
        skip_points : int
            Index of the first row belonging to interval 2.
        end_point : int
            Index of the first row belonging to the last interval (recovery).
        recovery_strain : np.ndarray
            Strain values from the recovery interval.
        """
        df = step.df
        if "Interval" not in df.columns:
            raise ValueError(
                f"Step '{step.name}' has no 'Interval' column; "
                "cannot use MCR interval-based boundary detection."
            )
        intervals = sorted(df["Interval"].unique())
        if len(intervals) < 3:
            raise ValueError(
                f"Step '{step.name}' has only {len(intervals)} interval(s); "
                "expected at least 3 (pre-shear, oscillation×N, recovery)."
            )

        first_osc_iv  = intervals[1]   # interval 2
        recovery_iv   = intervals[-1]  # last interval

        skip_points = int(df[df["Interval"] == first_osc_iv].index[0])
        end_point   = int(df[df["Interval"] == recovery_iv].index[0])
        recovery_strain = df.loc[df["Interval"] == recovery_iv, "Strain"].values

        return skip_points, end_point, recovery_strain

    @staticmethod
    def _find_skip_points(stress: np.ndarray) -> int:
        """Return index of first non-zero stress value."""
        nonzero = np.flatnonzero(stress != 0.0)
        return int(nonzero[0]) if len(nonzero) > 0 else 0

    @staticmethod
    def _detect_frequency(stress_signal: np.ndarray, dt: float, min_period_s: float = 0.5) -> Tuple[float, int]:
        """
        Detect the oscillation frequency via autocorrelation of the stress signal.

        Autocorrelation is used instead of FFT argmax because the stress waveform
        is biphasic (goes positive then negative within one cycle). The FFT sees
        two bumps per true period and picks up the second harmonic, giving half
        the correct period. Autocorrelation finds the true repeat length directly.

        Parameters
        ----------
        stress_signal : np.ndarray
            Stress values with padding already removed.
        dt : float
            Sample spacing [s].
        min_period_s : float
            Minimum plausible period [s] to avoid picking up noise peaks.

        Returns
        -------
        freq_hz : float
        period_length : int
            Samples per period.
        """
        signal = stress_signal - np.mean(stress_signal)
        corr = np.correlate(signal, signal, mode='full')
        corr = corr[len(corr) // 2:]   # positive lags only
        corr = corr / corr[0]          # normalise to 1 at zero lag

        min_lag = max(1, int(min_period_s / dt))
        for i in range(min_lag, len(corr) - 1):
            if corr[i] > corr[i - 1] and corr[i] > corr[i + 1] and corr[i] > 0.3:
                freq_hz = 1.0 / (i * dt)
                return float(freq_hz), i

        # Fallback to FFT if autocorrelation finds nothing
        warnings.warn("Autocorrelation period detection failed, falling back to FFT.")
        freqs = np.fft.rfftfreq(len(stress_signal), d=dt)
        fft_mag = np.abs(np.fft.rfft(stress_signal))
        dominant_idx = int(np.argmax(fft_mag[1:])) + 1
        freq_hz = float(freqs[dominant_idx])
        return freq_hz, int(round(1.0 / freq_hz / dt))

    @staticmethod
    def _align_lengths(
        fwd: np.ndarray, rev: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Trim the longer array so both have the same number of rows."""
        diff = len(fwd) - len(rev)
        if diff > 0:
            fwd = fwd[:-diff]
        elif diff < 0:
            rev = rev[:diff]
        return fwd, rev


# =============================================================================
# File picker (Qt-based, works in Spyder)
# =============================================================================

def open_file_dialog(title="Select ARES .txt file", file_filter="Text files (*.txt);;All files (*.*)"):
    """
    Open a Qt file picker dialog. Works correctly in Spyder.

    Remembers the last selected file in _last_file.txt next to this script,
    and offers to reuse it so you don't have to re-navigate every run.

    Returns
    -------
    str or None
        Selected file path, or None if cancelled.
    """
    from qtpy.QtWidgets import QApplication, QFileDialog, QMessageBox
    from pathlib import Path
    import sys

    app = QApplication.instance() or QApplication(sys.argv)

    # Check for a saved last-used file
    cache = Path(__file__).parent / "_last_file.txt"
    last_path = None
    if cache.exists():
        candidate = cache.read_text().strip()
        if Path(candidate).exists():
            last_path = candidate

    # Ask whether to reuse it
    if last_path:
        msg = QMessageBox()
        msg.setWindowTitle("Load file")
        msg.setText(f"Use the last file?\n\n{last_path}")
        msg.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        msg.setDefaultButton(QMessageBox.Yes)
        if msg.exec_() == QMessageBox.Yes:
            return last_path

    # Otherwise open the picker
    path, _ = QFileDialog.getOpenFileName(None, title, last_path or "", file_filter)
    if path:
        cache.write_text(path)
        return path
    return None


# =============================================================================
# Quick demo
# =============================================================================

if __name__ == "__main__":
    import sys
    from pathlib import Path

    # Set instrument here: "DHR" for ARES/DHR txt, "MCR" for RheoCompass CSV
    INSTRUMENT = "MCR"

    path = open_file_dialog()
    if not path:
        print("No file selected.")
        sys.exit(0)

    print(f"Loading {path} (instrument={INSTRUMENT}) ...")
    data = _load_file(path, INSTRUMENT)

    cfg = AnalysisConfig(
        control_type="stress",
        repeat_tests=3,
        periods_for_ft=3,
        percent_strain=True,
        instrument=INSTRUMENT,
    )

    exp = StrainShiftExperiment(data, cfg)
    exp.process()

    print("\nStress amplitudes per paired test:")
    for r in exp.results:
        print(f"  Paired test {r.paired_index}: {r.avg_stress_amp:.4f} Pa")

    exp.export()