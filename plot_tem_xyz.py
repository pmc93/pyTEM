"""
plot_tem_xyz.py - Quick-look plots of a TEM Data Manager `.xyz` file.

Produces:
  1. Raw dB/dt decay curves (all records + stacked mean) for LM and HM.
  2. The transmitter waveforms (normalised current vs time) for LM and HM.

Usage
-----
    python plot_tem_xyz.py [path-to-.xyz]
"""

from __future__ import annotations

import sys

import numpy as np
import matplotlib.pyplot as plt

from read_tem_xyz import read_tem_xyz

MOMENT_COLOR = {"LM": "tab:blue", "HM": "tab:red"}


def plot_raw(tem, ax):
    """Plot every raw sounding (faint) plus the stacked mean per moment."""
    for m in ("LM", "HM"):
        t = tem.gate_times[m]["center"]
        dbdt = tem.dbdt(m)                      # (n_records, n_gates)
        color = MOMENT_COLOR[m]

        # Faint individual records (absolute value, log-log).
        for row in dbdt:
            ax.plot(t, np.abs(row), color=color, lw=0.4, alpha=0.15)

        # Stacked mean.
        mean = np.abs(np.nanmean(dbdt, axis=0))
        ax.plot(t, mean, color=color, lw=2.2, marker="o", ms=4,
                label=f"{m} mean ({dbdt.shape[0]} records)")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("dB/dt [V/m$^2$]")
    ax.set_title("Raw dB/dt decay")
    ax.grid(True, which="both", ls=":", alpha=0.5)
    ax.legend(fontsize=8)


def plot_data_uncertainty(tem, ax):
    """Plot the stacked mean with the measured fractional uncertainty band."""
    for m in ("LM", "HM"):
        t = tem.gate_times[m]["center"]
        dbdt = tem.dbdt(m)
        frac = tem.dbdt_std(m)
        color = MOMENT_COLOR[m]

        mean = np.abs(np.nanmean(dbdt, axis=0))
        rel = np.nanmedian(frac, axis=0)                # fractional std
        lo = mean * np.clip(1.0 - rel, 1e-3, None)
        hi = mean * (1.0 + rel)

        ax.fill_between(t, lo, hi, color=color, alpha=0.2)
        ax.plot(t, mean, color=color, lw=2.0, marker="o", ms=4, label=f"{m}")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("dB/dt [V/m$^2$]")
    ax.set_title("Stacked mean with uncertainty")
    ax.grid(True, which="both", ls=":", alpha=0.5)
    ax.legend(fontsize=8)


def plot_waveforms(tem, ax):
    """Plot the normalised transmitter current waveforms."""
    for m in ("LM", "HM"):
        wf = tem.waveforms[m]
        ax.plot(wf["time"] * 1e3, wf["amplitude"], marker="o", ms=4,
                color=MOMENT_COLOR[m], label=f"{m} waveform")
    ax.axvline(0.0, color="k", lw=0.8, ls="--", alpha=0.6)
    ax.set_xlabel("Time [ms]")
    ax.set_ylabel("Normalised current [-]")
    ax.set_title("Transmitter waveforms")
    ax.grid(True, ls=":", alpha=0.5)
    ax.legend(fontsize=8)


def main(path):
    tem = read_tem_xyz(path)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    plot_raw(tem, axes[0])
    plot_data_uncertainty(tem, axes[1])
    plot_waveforms(tem, axes[2])

    instr = tem.meta.get("InstrumentType", "TEM")
    fig.suptitle(f"{instr}  -  {path.split(chr(92))[-1]}", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    out = "tem_xyz_rawdata.png"
    fig.savefig(out, dpi=130)
    print(f"Saved {out}")
    plt.show()


if __name__ == "__main__":
    fname = sys.argv[1] if len(sys.argv) > 1 else r"C:\Users\pamcl\Downloads\2026_0701_165727_ChA.xyz"
    main(fname)
