import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

# ── Path setup ────────────────────────────────────────────────────────────────
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from pytem import fwd_circle_central, butterworth_filter, cascade_filter

# ── Page header ───────────────────────────────────────────────────────────────
st.title("🔊 System Filters ▶️")
st.header(":green[How the instrument's frequency response shapes dB/dt]")

st.markdown(
    r"""
    Real TEM receivers are **not perfectly flat** in the frequency domain. They
    have high-frequency roll-offs (low-pass behaviour) due to their analogue
    circuitry. This means that the recorded signal is a **filtered version** of
    the true earth response — the system's transfer function $H(\omega)$
    multiplies the frequency-domain field before the Fourier transform to the
    time domain:

    $$H_z^{meas}(\omega) = H(\omega)\cdot H_z^{sec}(\omega)$$

    Ignoring the system filter in the forward model leads to systematic errors
    in the inverted model, particularly at **early times** where the filter
    roll-off most strongly distorts the signal.

    pyTEM implements two filter types:
    - **Butterworth filter** — 1st or 2nd order LP/HP/BP.
    - **Cascade filter** — two cascaded 1st-order Butterworth LP (WalkTEM convention).
    """
)

with st.expander(":green[**Check your understanding — quiz**]"):
    col1, col2 = st.columns(2)
    with col1:
        q1 = st.radio(
            ":green[**At which times does the system filter have the greatest impact?**]",
            ["Early times", "Late times", "All times equally"],
            index=None,
        )
        if q1 == "Early times":
            st.success("Correct! High-frequency content is suppressed by the low-pass filter, which attenuates the rapidly varying early-time signal most strongly.")
        elif q1 is not None:
            st.error("Think about which time domain is associated with high frequencies — the Fourier transform relates high frequencies to early times.")

    with col2:
        q2 = st.radio(
            ":green[**Increasing the low-pass cutoff frequency f_high will ...**]",
            [
                "Suppress more of the high-frequency signal",
                "Allow more high-frequency signal through",
                "Have no effect on dB/dt",
            ],
            index=None,
        )
        if q2 == "Allow more high-frequency signal through":
            st.success("Correct! A higher cutoff lets more bandwidth pass, moving the filtered response closer to the unfiltered response.")
        elif q2 is not None:
            st.error("A higher cutoff frequency means the filter rolls off at a higher frequency, allowing more of the signal bandwidth through.")

# ── Controls ──────────────────────────────────────────────────────────────────
st.subheader(":green-background[Model & filter controls]", divider="green")

col_model, col_filter = st.columns(2)

with col_model:
    st.markdown("**Earth model** — last row is the half-space, leave its Thickness empty")
    _default_sf = pd.DataFrame({
        "Thickness (m)": [30.0, 80.0, None],
        "Resistivity (Ω·m)": [100.0, 10.0, 300.0],
    })
    _edited_sf = st.data_editor(
        _default_sf,
        column_config={
            "Thickness (m)": st.column_config.NumberColumn(
                min_value=0.1, max_value=10000.0, format="%.1f",
            ),
            "Resistivity (Ω·m)": st.column_config.NumberColumn(
                min_value=0.01, max_value=1e6, format="%.1f",
            ),
        },
        num_rows="dynamic",
        use_container_width=True,
        key="sf_model_editor",
    )
    _valid_sf = _edited_sf.dropna(subset=["Resistivity (Ω·m)"])
    _sf_thicknesses = _valid_sf["Thickness (m)"].dropna().tolist()
    _sf_resistivities = _valid_sf["Resistivity (Ω·m)"].tolist()
    if len(_sf_resistivities) < 1:
        st.warning("Add at least one layer.")
        st.stop()
    tx_r_sf = st.number_input("Loop radius (m)", min_value=1.0, max_value=500.0, value=50.0, step=5.0, key="sf_r")

with col_filter:
    st.markdown("**Filter type**")
    filter_type = st.selectbox(
        "Filter type",
        ["None (unfiltered)", "Butterworth 1st order", "Butterworth 2nd order", "Cascade (WalkTEM)"],
    )

    f_high = st.select_slider(
        "Low-pass cutoff f_high (Hz)",
        options=[1e3, 3e3, 1e4, 3e4, 1e5, 3e5, 1e6],
        value=3e4,
        format_func=lambda x: f"{x/1e3:.0f} kHz",
        key="sf_fhigh",
    )

    if filter_type in ("Butterworth 1st order", "Butterworth 2nd order"):
        use_hp = st.toggle("Add high-pass (low-cut)", value=False)
        f_low = None
        if use_hp:
            f_low = st.select_slider(
                "High-pass cutoff f_low (Hz)",
                options=[1.0, 10.0, 100.0, 1e3],
                value=10.0,
                format_func=lambda x: f"{x:.0f} Hz",
            )
    else:
        use_hp = False
        f_low = None

times_sf = np.logspace(-5, -2, 31)

@st.cache_data(show_spinner=False)
def run_with_filter(rho_t, thick_t, tx_r, times_t, filter_key):
    thicknesses = list(thick_t)
    resistivities = list(rho_t)
    times = np.array(times_t)

    dbdt_raw = fwd_circle_central(thicknesses, resistivities, tx_radius=tx_r, times=times)

    if filter_key == "None":
        return -dbdt_raw, -dbdt_raw
    elif filter_key.startswith("Butterworth1"):
        parts = filter_key.split("|")
        fh = float(parts[1])
        fl = float(parts[2]) if parts[2] != "None" else None
        H = butterworth_filter(f_low=fl, f_high=fh, order=1)
    elif filter_key.startswith("Butterworth2"):
        parts = filter_key.split("|")
        fh = float(parts[1])
        fl = float(parts[2]) if parts[2] != "None" else None
        H = butterworth_filter(f_low=fl, f_high=fh, order=2)
    elif filter_key.startswith("Cascade"):
        parts = filter_key.split("|")
        fh = float(parts[1])
        H = cascade_filter(fh)
    else:
        H = None

    dbdt_filt = fwd_circle_central(thicknesses, resistivities, tx_radius=tx_r, times=times,
                                   system_filter=H)
    return -dbdt_raw, -dbdt_filt


# Build a cache key that encodes all filter params
if filter_type == "None (unfiltered)":
    fkey = "None"
elif filter_type == "Butterworth 1st order":
    fl_val = f_low if use_hp else None
    fkey = f"Butterworth1|{f_high}|{fl_val}"
elif filter_type == "Butterworth 2nd order":
    fl_val = f_low if use_hp else None
    fkey = f"Butterworth2|{f_high}|{fl_val}"
else:
    fkey = f"Cascade|{f_high}"

with st.spinner("Computing …"):
    dbdt_unfilt, dbdt_filt = run_with_filter(
        tuple(_sf_resistivities), tuple(_sf_thicknesses), tx_r_sf, tuple(times_sf), fkey
    )

# ── Plots ──────────────────────────────────────────────────────────────────────
st.subheader(":green-background[Results]", divider="green")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

ax1 = axes[0]
ax1.loglog(times_sf * 1e3, dbdt_unfilt, "k--", lw=1.5, label="Unfiltered")
ax1.loglog(times_sf * 1e3, dbdt_filt, "g-o", ms=4, lw=1.5, label=filter_type)
ax1.set_xlabel("Time (ms)")
ax1.set_ylabel(r"$|\partial B_z / \partial t|$  (T/s)")
ax1.set_title("Effect of system filter on dB/dt")
ax1.grid(True, which="both", ls="--", alpha=0.4)
ax1.legend()

# Transfer function plot
ax2 = axes[1]
f_plot = np.logspace(1, 7, 500)
omega_plot = 2 * np.pi * f_plot

if filter_type == "None (unfiltered)":
    H_vals = np.ones(len(omega_plot))
elif filter_type == "Butterworth 1st order":
    fl_val = f_low if use_hp else None
    H_obj = butterworth_filter(f_low=fl_val, f_high=f_high, order=1)
    H_vals = np.abs(H_obj(omega_plot))
elif filter_type == "Butterworth 2nd order":
    fl_val = f_low if use_hp else None
    H_obj = butterworth_filter(f_low=fl_val, f_high=f_high, order=2)
    H_vals = np.abs(H_obj(omega_plot))
else:
    H_obj = cascade_filter(f_high)
    H_vals = np.abs(H_obj(omega_plot))

ax2.semilogx(f_plot / 1e3, 20 * np.log10(np.maximum(H_vals, 1e-10)),
             "g-", lw=2, label=filter_type)
ax2.axvline(f_high / 1e3, color="r", ls="--", alpha=0.6, label=f"f_high = {f_high/1e3:.0f} kHz")
ax2.set_xlabel("Frequency (kHz)")
ax2.set_ylabel("Gain (dB)")
ax2.set_title("Filter transfer function |H(f)|")
ax2.grid(True, which="both", ls="--", alpha=0.4)
ax2.set_ylim(-60, 5)
ax2.legend()

plt.tight_layout()
st.pyplot(fig)
plt.close(fig)

st.info(
    "Tip: zoom in on early times (use sliders on the time axis) to see how the "
    "filter attenuates the fast transients at t < 0.1 ms."
)
