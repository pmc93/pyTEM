"""
read_tem_xyz.py - Reader for TEM Data Manager `.xyz` exports (TEM2Go / tTEM).

Parses the three things needed for modelling:
  * data       - the sounding table (dB/dt values and uncertainties per gate)
  * waveforms  - LM/HM transmitter current waveforms (time, amplitude)
  * gate_times - LM/HM processed gate open/center/close times [s]

Usage
-----
    from read_tem_xyz import read_tem_xyz

    tem = read_tem_xyz("2026_0701_165727_ChA.xyz")

    tem.waveforms["LM"]["time"]        # -> np.ndarray of ramp break-point times
    tem.waveforms["LM"]["amplitude"]   # -> np.ndarray of normalised current
    tem.gate_times["HM"]["center"]     # -> np.ndarray of HM gate centre times [s]
    tem.data                           # -> pandas.DataFrame with every record

    lm, hm = tem.low_moment(), tem.high_moment()   # split by moment
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class TEMData:
    """Container for a parsed TEM Data Manager `.xyz` file."""

    meta: dict = field(default_factory=dict)
    waveforms: dict = field(default_factory=dict)   # {"LM": {"time":.., "amplitude":..}, "HM": {...}}
    gate_times: dict = field(default_factory=dict)   # {"LM": {"open":.., "center":.., "close":..}, "HM": {...}}
    data: pd.DataFrame = field(default_factory=pd.DataFrame)

    def low_moment(self) -> pd.DataFrame:
        """Records acquired with the low moment (Moment == 0)."""
        return self.data[self.data["Moment"] == 0].reset_index(drop=True)

    def high_moment(self) -> pd.DataFrame:
        """Records acquired with the high moment (Moment == 1)."""
        return self.data[self.data["Moment"] == 1].reset_index(drop=True)

    def dbdt(self, moment: str) -> np.ndarray:
        """Stacked dB/dt matrix (n_records, n_gates) for 'LM' or 'HM'."""
        df = self.low_moment() if moment.upper() == "LM" else self.high_moment()
        n = len(self.gate_times[moment.upper()]["center"])
        cols = [f"dbdtDat{i:03d}" for i in range(1, n + 1)]
        return df[cols].to_numpy(dtype=float)

    def dbdt_std(self, moment: str) -> np.ndarray:
        """Stacked dB/dt uncertainty matrix (n_records, n_gates) for 'LM' or 'HM'.

        Values are fractions of the corresponding dB/dt reading (dbdtStdFUnit).
        """
        df = self.low_moment() if moment.upper() == "LM" else self.high_moment()
        n = len(self.gate_times[moment.upper()]["center"])
        cols = [f"dbdtStd{i:03d}" for i in range(1, n + 1)]
        return df[cols].to_numpy(dtype=float)

    def _meta_floats(self, key: str) -> list:
        """Parse a whitespace-separated numeric metadata value into a list."""
        return [float(t) for t in self.meta.get(key, "").split()]

    def snr(self, moment: str):
        """
        Per-gate signal-to-noise ratio of the stacked sounding.

        Returns
        -------
        (times, mean, sem, snr) : tuple of np.ndarray
            Gate centre times, mean dB/dt across records, standard error of the
            mean (from the scatter across records), and |mean| / sem.
        """
        m = moment.upper()
        t = self.gate_times[m]["center"]
        dbdt = self.dbdt(m)
        n_eff = np.sum(np.isfinite(dbdt), axis=0)
        mean = np.nanmean(dbdt, axis=0)
        sem = np.nanstd(dbdt, axis=0) / np.sqrt(np.maximum(n_eff, 1))
        with np.errstate(divide="ignore", invalid="ignore"):
            snr = np.abs(mean) / sem
        return t, mean, sem, snr

    def to_pytem(self, moment: str, record=None, tx_turns=None,
                 signed=False, min_noise=0.03):
        """
        Convert one moment ('LM' or 'HM') into keyword arguments for pyTEM.

        The returned dict plugs directly into ``pytem.invert`` and the
        ``pytem.fwd_square_*`` forward functions.  Conventions handled here:

          * gate centre times -> ``times`` [s]
          * dB/dt data are already normalised by Rx-coil area, so their unit is
            [V/m^2] == [T/s]; usable as ``obs_data`` with ``rx_area = 1``.
          * the waveform amplitude column is normalised (0..1); it is scaled to
            absolute ampere-turns (peak current * Tx turns) so that
            ``waveform_currents`` carries the full transmitter moment.
          * ``dbdtStd`` is a *fractional* uncertainty -> ``noise_std``.
          * loop size / Rx offset are read from the header to pick the geometry.

        Parameters
        ----------
        moment    : str    'LM' (low) or 'HM' (high).
        record    : int or None
            Index of a single sounding within the moment.  ``None`` (default)
            stacks all records of that moment (mean of dB/dt).
        tx_turns  : int or None
            Number of Tx turns.  ``None`` reads ``TxLoop_NTurns`` from the header.
        signed    : bool   Keep the measured sign (default takes abs value, as
                           pyTEM expects positive dB/dt).
        min_noise : float  Floor applied to the fractional noise (default 0.03).

        Returns
        -------
        dict
            Keys: ``times``, ``obs_data``, ``noise_std``, ``waveform_times``,
            ``waveform_currents``, ``geometry``, ``tx_size``, ``rx_x``,
            ``rx_y``, plus context keys ``gate_open``, ``gate_close``,
            ``tx_turns``, ``peak_current``, ``n_records``.
        """
        m = moment.upper()
        if m not in ("LM", "HM"):
            raise ValueError("moment must be 'LM' or 'HM'")

        gt = self.gate_times[m]
        times = gt["center"]

        df = self.low_moment() if m == "LM" else self.high_moment()
        dbdt = self.dbdt(m)
        std = self.dbdt_std(m)

        if record is not None:
            obs = dbdt[record]
            frac = std[record]
            peak_current = float(df["TxCurrent"].iloc[record])
            n_records = 1
        else:
            obs = np.nanmean(dbdt, axis=0)
            # Uncertainty of the stacked mean estimated from the empirical
            # scatter across records: standard error of the mean divided by the
            # mean magnitude gives a fractional noise. This reflects the true
            # data quality far better than the stored per-record fractions.
            n_eff = np.sum(np.isfinite(dbdt), axis=0)
            sem = np.nanstd(dbdt, axis=0) / np.sqrt(np.maximum(n_eff, 1))
            with np.errstate(divide="ignore", invalid="ignore"):
                frac = sem / np.abs(obs)
            peak_current = float(np.nanmedian(df["TxCurrent"]))
            n_records = int(df.shape[0])

        if not signed:
            obs = np.abs(obs)
        noise_std = np.clip(np.nan_to_num(frac, nan=min_noise), min_noise, None)

        # ---- Transmitter waveform in absolute ampere-turns ----
        if tx_turns is None:
            tx_turns = int(float(self.meta.get("TxLoop_NTurns", 1) or 1))
        wf_t = self.waveforms[m]["time"]
        wf_I = self.waveforms[m]["amplitude"] * peak_current * tx_turns

        # ---- Geometry from the header ----
        tx_len = self._meta_floats("TxLoop_XYLength")
        tx_side = tx_len[0] if tx_len else None
        rx_pos = self._meta_floats("RxCoil_XYZPos")
        rx_x = rx_pos[0] if rx_pos else 0.0
        rx_y = rx_pos[1] if len(rx_pos) > 1 else 0.0
        offset = abs(rx_x) > 1e-6 or abs(rx_y) > 1e-6
        geometry = "square_offset" if offset else "square_central"

        return {
            "times": times,
            "obs_data": obs,
            "noise_std": noise_std,
            "waveform_times": wf_t,
            "waveform_currents": wf_I,
            "geometry": geometry,
            "tx_size": tx_side,
            "rx_x": rx_x,
            "rx_y": rx_y,
            # --- context (not invert kwargs) ---
            "gate_open": gt.get("open"),
            "gate_close": gt.get("close"),
            "tx_turns": tx_turns,
            "peak_current": peak_current,
            "n_records": n_records,
        }


def _floats(text: str) -> np.ndarray:
    """Parse a whitespace-separated list of floats."""
    return np.array([float(tok) for tok in text.split()], dtype=float)


def read_tem_xyz(path: str) -> TEMData:
    """
    Read a TEM Data Manager `.xyz` file.

    Parameters
    ----------
    path : str
        Path to the `.xyz` file.

    Returns
    -------
    TEMData
        Parsed metadata, waveforms, gate times and the sounding DataFrame.
    """
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        lines = fh.readlines()

    tem = TEMData()
    tem.waveforms = {"LM": {}, "HM": {}}
    tem.gate_times = {"LM": {}, "HM": {}}

    header_idx = None
    for i, raw in enumerate(lines):
        line = raw.strip()
        if not line:
            continue

        # Section markers like "[RxTxSpecs]" carry no key/value.
        if line.startswith("[") and "]" in line:
            continue

        if "=" in line:
            key, val = line.split("=", 1)
            key, val = key.strip(), val.strip()

            if key.endswith("Waveform_Time"):
                tem.waveforms[key[:2]]["time"] = _floats(val)
            elif key.endswith("Waveform_Amplitude"):
                tem.waveforms[key[:2]]["amplitude"] = _floats(val)
            elif key.endswith("OpenTime"):
                tem.gate_times[key[:2]]["open"] = _floats(val)
            elif key.endswith("CenterTime"):
                tem.gate_times[key[:2]]["center"] = _floats(val)
            elif key.endswith("CloseTime"):
                tem.gate_times[key[:2]]["close"] = _floats(val)
            else:
                tem.meta[key] = val
            continue

        # First non key/value, non-section line beginning with "Date" is the
        # column header of the data table.
        if line.startswith("Date"):
            header_idx = i
            break

    if header_idx is None:
        raise ValueError(f"No data table header found in {path!r}")

    columns = lines[header_idx].split()
    tem.data = pd.read_csv(
        path,
        skiprows=header_idx + 1,
        sep=r"\s+",
        names=columns,
        na_values=["nan", "NaN"],
        engine="python",
    )

    return tem


if __name__ == "__main__":
    import sys

    fname = sys.argv[1] if len(sys.argv) > 1 else r"C:\Users\pamcl\Downloads\2026_0701_165727_ChA.xyz"
    tem = read_tem_xyz(fname)

    print(f"Instrument : {tem.meta.get('InstrumentType', '?')} "
          f"({tem.meta.get('InstrumentID', '?')})")
    print(f"Records    : {len(tem.data)}  "
          f"(LM={len(tem.low_moment())}, HM={len(tem.high_moment())})")
    for m in ("LM", "HM"):
        wt = tem.waveforms[m].get("time")
        gc = tem.gate_times[m].get("center")
        print(f"{m}: waveform pts={len(wt) if wt is not None else 0}, "
              f"gates={len(gc) if gc is not None else 0}, "
              f"gate range=[{gc[0]:.3e}, {gc[-1]:.3e}] s" if gc is not None else f"{m}: no gates")
    print("\nColumns:", list(tem.data.columns[:20]), "...")

    for m in ("LM", "HM"):
        kw = tem.to_pytem(m)
        print(f"\n=== pyTEM kwargs for {m} "
              f"(geometry={kw['geometry']}, tx_size={kw['tx_size']} m, "
              f"turns={kw['tx_turns']}, peak_I={kw['peak_current']} A, "
              f"rx=({kw['rx_x']}, {kw['rx_y']}) m) ===")
        print(f"  times            : {kw['times'].shape} "
              f"[{kw['times'][0]:.3e} .. {kw['times'][-1]:.3e}] s")
        print(f"  obs_data         : {kw['obs_data'].shape} "
              f"[{kw['obs_data'][0]:.3e} .. {kw['obs_data'][-1]:.3e}] T/s")
        print(f"  noise_std        : "
              f"[{kw['noise_std'].min():.3f} .. {kw['noise_std'].max():.3f}]")
        print(f"  waveform_times   : {kw['waveform_times'].shape} "
              f"[{kw['waveform_times'][0]:.3e} .. {kw['waveform_times'][-1]:.3e}] s")
        print(f"  waveform_currents: peak {np.max(kw['waveform_currents']):.1f} A-turns "
              f"(stacked over {kw['n_records']} records)")
