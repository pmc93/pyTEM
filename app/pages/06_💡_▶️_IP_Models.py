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

from pytem import fwd_circle_central
from pytem.ip_models import pelton_res_rho, cole_cole_rho, tem_forward_ip

# ── Page header ───────────────────────────────────────────────────────────────
st.title("💡 IP Models ▶️")
st.header(":orange[Induced Polarisation effects in TEM data]")

st.markdown(
    r"""
    **Induced Polarisation (IP)** arises when the subsurface contains polarisable
    minerals (e.g. disseminated sulphides or clay particles). During the TEM
    transient, part of the energy is stored in the polarisation of these minerals
    rather than flowing as Ohm's-law current. When the electric field relaxes,
    this stored energy is released — generating a secondary response that can
    **reverse the sign** of the late-time dB/dt signal.

    IP effects are modelled by replacing the real DC resistivity $\rho_0$ with a
    **complex, frequency-dependent resistivity** $\rho(\omega)$. Two common models are:

    **Pelton et al. (1978) — resistivity formulation:**

    $$\rho(\omega) = \rho_0\!\left[1 - m\!\left(1 - \frac{1}{1 + (j\omega\tau)^c}\right)\right]$$

    **Cole-Cole (1941) — conductivity formulation:**

    $$\sigma(\omega) = \sigma_\infty + \frac{\sigma_0 - \sigma_\infty}{1 + (j\omega\tau)^c},
    \qquad \rho(\omega) = 1/\sigma(\omega)$$

    The key IP parameters are:
    - $m$ — chargeability (dimensionless, 0–1); controls the magnitude of the IP effect
    - $\tau$ — relaxation time (s); controls which time range shows the sign reversal
    - $c$ — frequency exponent (0–1); controls the breadth of the relaxation peak
    """
)

with st.expander(":green[**Check your understanding — quiz**]"):
    col1, col2 = st.columns(2)
    with col1:
        q1 = st.radio(
            ":orange[**What is the most visible sign of IP in a TEM decay curve?**]",
            [
                "Faster decay at early times",
                "Sign reversal (negative dBdt becoming positive) at late times",
                "A kink in the middle of the curve",
                "Oscillations in the time series",
            ],
            index=None,
        )
        if q1 == "Sign reversal (negative dBdt becoming positive) at late times":
            st.success("Correct! IP causes the delayed release of stored polarisation energy, which can produce a sign reversal in the late-time dB/dt.")
        elif q1 is not None:
            st.error("The characteristic signature is a sign reversal at late times — the dB/dt changes sign due to the IP secondary response.")

    with col2:
        q2 = st.radio(
            ":orange[**Increasing the chargeability m will ...**]",
            [
                "Shift the sign reversal to earlier times",
                "Make the IP effect stronger (larger sign reversal)",
                "Have no effect if τ is small",
                "Suppress the IP effect",
            ],
            index=None,
        )
        if q2 == "Make the IP effect stronger (larger sign reversal)":
            st.success("Correct! A larger chargeability means more energy is stored in polarisation and a more pronounced sign reversal.")
        elif q2 is not None:
            st.error("Chargeability m controls the amplitude of the IP effect. A larger m gives a stronger (more visible) IP signature.")

# ── Model ──────────────────────────────────────────────────────────────────────
st.subheader(":orange-background[Earth model with IP layer]", divider="orange")

col_m, col_ip = st.columns(2)

with col_m:
    st.markdown("**Background model** — last row is the half-space, leave its Thickness empty")
    _default_ip = pd.DataFrame({
        "Thickness (m)": [30.0, 60.0, None],
        "Resistivity (Ω·m)": [100.0, 50.0, 300.0],
    })
    _edited_ip = st.data_editor(
        _default_ip,
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
        key="ip_model_editor",
    )
    _valid_ip = _edited_ip.dropna(subset=["Resistivity (Ω·m)"])
    _ip_thicknesses = _valid_ip["Thickness (m)"].dropna().tolist()
    _ip_resistivities = _valid_ip["Resistivity (Ω·m)"].tolist()
    if len(_ip_resistivities) < 2:
        st.warning("Need at least 2 layers: the IP layer and a half-space.")
        st.stop()
    tx_r_ip = st.number_input("Loop radius (m)", min_value=1.0, max_value=500.0, value=50.0, step=5.0, key="ip_txr")
    st.caption("The IP effect is applied to the second-to-last layer (the IP layer).")

with col_ip:
    st.markdown("**IP parameters (Pelton or Cole-Cole)**")
    ip_model_type = st.selectbox("IP model", ["Pelton (resistivity)", "Cole-Cole (conductivity)"])
    m_ip = st.number_input("Chargeability m", min_value=0.0, max_value=0.99, value=0.5, step=0.05, key="ip_m")
    tau_log = st.number_input("log₁₀(τ / s)", min_value=-4.0, max_value=1.0, value=-2.0, step=0.1, key="ip_tau")
    c_ip = st.number_input("Frequency exponent c", min_value=0.1, max_value=1.0, value=0.5, step=0.05, key="ip_c")
    tau_ip = 10 ** tau_log

times_ip = np.logspace(-5, -2, 41)
thicknesses_ip = _ip_thicknesses
resistivities_ip = _ip_resistivities
# IP effect is applied to the second-to-last layer (index -2)
_n_ip_layers = len(resistivities_ip)

# ── Forward without IP ─────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def fwd_no_ip(thick_t, rho_t, tx_r, times_t):
    return fwd_circle_central(list(thick_t), list(rho_t), tx_radius=tx_r, times=np.array(times_t))


# ── Forward with IP (cannot cache easily due to lambda) ───────────────────────
@st.cache_data(show_spinner=False)
def fwd_with_ip_pelton(thick_t, rho_t, tx_r, times_t, m, tau, c):
    thicknesses = list(thick_t)
    resistivities = list(rho_t)
    times = np.array(times_t)
    n = len(resistivities)
    ip_funcs = [None] * n
    ip_funcs[-2] = lambda rho_0, omega, _m=m, _tau=tau, _c=c: pelton_res_rho(rho_0, _m, _tau, _c, omega)
    return tem_forward_ip(thicknesses, resistivities, tx_r, times, ip_funcs=ip_funcs)


@st.cache_data(show_spinner=False)
def fwd_with_ip_colecole(thick_t, rho_t, tx_r, times_t, m, tau, c):
    thicknesses = list(thick_t)
    resistivities = list(rho_t)
    times = np.array(times_t)
    rho_0 = rho_t[-2]
    sigma_0 = 1.0 / rho_0
    sigma_inf = sigma_0 / (1.0 - m)
    n = len(resistivities)
    ip_funcs = [None] * n
    ip_funcs[-2] = lambda _rho_0, omega, _s0=sigma_0, _sinf=sigma_inf, _tau=tau, _c=c: \
        cole_cole_rho(_rho_0, _s0, _sinf, _tau, _c, omega)
    return tem_forward_ip(thicknesses, resistivities, tx_r, times, ip_funcs=ip_funcs)


with st.spinner("Computing …"):
    dbdt_noip = fwd_no_ip(tuple(thicknesses_ip), tuple(resistivities_ip),
                          tx_r_ip, tuple(times_ip))
    if ip_model_type == "Pelton (resistivity)":
        dbdt_ip = fwd_with_ip_pelton(tuple(thicknesses_ip), tuple(resistivities_ip),
                                     tx_r_ip, tuple(times_ip), m_ip, tau_ip, c_ip)
    else:
        dbdt_ip = fwd_with_ip_colecole(tuple(thicknesses_ip), tuple(resistivities_ip),
                                       tx_r_ip, tuple(times_ip), m_ip, tau_ip, c_ip)

# ── Plots ──────────────────────────────────────────────────────────────────────
st.subheader(":orange-background[Results]", divider="orange")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Left: dB/dt with and without IP
ax1 = axes[0]
# For plotting, show absolute value and mark sign reversals
dbdt_ip_plot = -dbdt_ip      # flip for positive expected sign
dbdt_noip_plot = -dbdt_noip

pos_mask = dbdt_ip_plot > 0
neg_mask = ~pos_mask

ax1.loglog(times_ip[pos_mask] * 1e3, np.abs(dbdt_ip_plot[pos_mask]),
           "o", color="darkorange", ms=5, label="With IP (positive)")
if neg_mask.any():
    ax1.loglog(times_ip[neg_mask] * 1e3, np.abs(dbdt_ip_plot[neg_mask]),
               "^", color="firebrick", ms=5, label="With IP (sign-reversed!)")
ax1.loglog(times_ip * 1e3, np.abs(dbdt_noip_plot), "k--", lw=1.5, label="No IP")
ax1.set_xlabel("Time (ms)")
ax1.set_ylabel(r"$|\partial B_z / \partial t|$  (T/s)")
ax1.set_title("dB/dt: IP vs. no IP")
ax1.grid(True, which="both", ls="--", alpha=0.4)
ax1.legend()

# Right: frequency-dependent resistivity of the IP layer
freqs = np.logspace(-2, 6, 500)
omegas = 2 * np.pi * freqs
if ip_model_type == "Pelton (resistivity)":
    rho_complex = pelton_res_rho(rho_ip_layer, m_ip, tau_ip, c_ip, omegas)
else:
    sigma_0 = 1.0 / rho_ip_layer
    sigma_inf = sigma_0 / (1.0 - m_ip)
    rho_complex = cole_cole_rho(rho_ip_layer, sigma_0, sigma_inf, tau_ip, c_ip, omegas)

ax2 = axes[1]
ax2_twin = ax2.twinx()
ax2.semilogx(freqs, np.real(rho_complex), "darkorange", lw=2, label=r"Re[$\rho(\omega)$]")
ax2_twin.semilogx(freqs, -np.angle(rho_complex, deg=True), "steelblue", lw=2,
                   ls="--", label="Phase (°)")
ax2.set_xlabel("Frequency (Hz)")
ax2.set_ylabel(r"Real resistivity ($\Omega\cdot$m)", color="darkorange")
ax2_twin.set_ylabel("Phase angle (°)", color="steelblue")
ax2.set_title(f"IP layer complex resistivity ({ip_model_type})")
ax2.grid(True, which="both", ls="--", alpha=0.4)
lines1, labels1 = ax2.get_legend_handles_labels()
lines2, labels2 = ax2_twin.get_legend_handles_labels()
ax2.legend(lines1 + lines2, labels1 + labels2, loc="center left")

plt.tight_layout()
st.pyplot(fig)
plt.close(fig)

if neg_mask.any():
    st.warning(
        f"Sign reversal detected at {times_ip[neg_mask][0]*1e3:.2f} ms "
        "— triangles on the log-log plot mark these reversed-sign values. "
        "In practice the sign reversal is the key diagnostic for IP in TEM data."
    )
else:
    st.info("No sign reversal detected with these parameters. "
            "Try increasing the chargeability m or adjusting τ to match the gate time range.")
