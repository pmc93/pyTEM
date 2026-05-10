import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# -- Path setup ----------------------------------------------------------------
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from pytem import fwd_circle_central, getJ_ana
from ves import forward as ves_forward, jacobian as ves_jacobian

# -- Constants -----------------------------------------------------------------
_N_DATA = 20          # fixed number of data points for both methods
_RHO    = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000]


# -- Shared utilities ----------------------------------------------------------
def _model_ui(n):
    hdr = st.columns([2, 3, 4])
    hdr[0].caption("Layer")
    hdr[1].caption("Thickness (m)")
    hdr[2].caption("Resistivity (Ohm*m)")
    h_out, r_out = [], []
    defaults_rho = [100, 20, 300, 100, 50, 200]
    defaults_h   = [10, 30, 20, 15, 40]
    for i in range(n):
        c = st.columns([2, 3, 4])
        c[0].markdown(f"**{'Layer ' + str(i+1) if i < n-1 else 'Half-space'}**")
        if i < n - 1:
            h_def = defaults_h[i] if i < len(defaults_h) else 20
            h_out.append(float(c[1].slider(f"Thickness {i+1} (m)", 1, 500, h_def,
                                            key=f"jac_h{i}",
                                            label_visibility="collapsed")))
        rho_def = defaults_rho[i] if i < len(defaults_rho) else 100
        rho_def = min(_RHO, key=lambda x: abs(x - rho_def))
        r_out.append(float(c[2].select_slider(f"Resistivity {i+1} (Ohm*m)", _RHO, value=rho_def,
                                               key=f"jac_r{i}",
                                               label_visibility="collapsed")))
    return h_out, r_out


# -- Cached computations -------------------------------------------------------
@st.cache_data(show_spinner=False)
def _tem_jac(h_t, log_rho_t, tx_r, times_t):
    h       = list(h_t)
    log_rho = np.array(log_rho_t)
    times   = np.array(times_t)
    dbdt = fwd_circle_central(h, np.exp(log_rho).tolist(), tx_radius=tx_r, times=times)
    J    = getJ_ana(h, log_rho, tx_r, times,
                    geometry="circle_central",
                    hankel_filter="key_101", fourier_filter="key_101")
    return -dbdt, J


@st.cache_data(show_spinner=False)
def _ves_jac(ab2_t, rho_t, h_t, filt):
    ab2   = np.array(ab2_t)
    rho   = np.array(rho_t)
    h     = np.array(h_t)
    rhoap = ves_forward(ab2, rho, h, filt)
    J     = ves_jacobian(ab2, rho, h, filt)
    return rhoap, J


# -- Page ----------------------------------------------------------------------
st.title("Jacobian & Sensitivity")
st.header(":violet[Which data points are sensitive to which layers?]")

st.markdown(
    r"""
    **TEM** measures a transient electromagnetic signal that decays over time.
    Early time gates see shallow structure; late gates see deeper structure.
    The data are plotted as $|\partial B_z / \partial t|$ vs time on a log-log scale.

    **VES** injects current into the ground and measures voltage.
    Increasing electrode spacing AB/2 samples progressively greater depth.
    The data are plotted as apparent resistivity $\rho_a$ vs AB/2 on a log-log scale.

    The **Jacobian** $J_{ij}$ is the sensitivity of data point $i$ to layer $j$:
    a large $|J_{ij}|$ means that data point carries information about that layer.
    Red cells mean increasing the layer's resistivity raises the data value; blue means the opposite.
    The **column norm** (bar chart) is the total sensitivity across all data points for each layer
    -- a small bar means the layer is poorly constrained by the data.
    """
)

# -- Layout: model | TEM settings | VES settings -------------------------------
col_mod, col_tem_s, col_ves_s = st.columns([4, 2, 2])

with col_mod:
    st.subheader(":blue-background[Shared earth model]")
    n_layers = int(st.number_input("Number of layers", 2, 6, 3, key="jac_n"))
    thick, rho = _model_ui(n_layers)

with col_tem_s:
    st.subheader(":violet-background[TEM settings]")
    tx_area = st.number_input("Tx loop area (m²)", min_value=100, max_value=500000, value=1600, step=100, key="jac_tem_area")
    tx_r    = float(np.sqrt(tx_area / np.pi))
    t_min = st.slider("Earliest gate (10^x s)", -6.0, -4.0, -5.0, 0.25, key="jac_tem_tmin")
    t_max = st.slider("Latest gate (10^x s)",   -3.0, -1.0, -2.0, 0.25, key="jac_tem_tmax")
    filt_tem = "key_101"

with col_ves_s:
    st.subheader(":orange-background[VES settings]")
    ab2_min = st.slider("AB/2 minimum (m)",   1,    50,   1,   key="jac_ves_ab2min")
    ab2_max = st.slider("AB/2 maximum (m)",  50,  2000, 300,   key="jac_ves_ab2max")
    filt_ves = st.selectbox("Filter", ["gs7", "gs11", "gs22"], key="jac_ves_filt")

# -- Compute -------------------------------------------------------------------
times = np.logspace(t_min, t_max, _N_DATA)
ab2   = np.logspace(np.log10(ab2_min), np.log10(ab2_max), _N_DATA)
log_rho = np.log(np.array(rho))

with st.spinner("Computing Jacobians ..."):
    dbdt, J_tem          = _tem_jac(tuple(thick), tuple(log_rho.tolist()), tx_r, tuple(times.tolist()))
    rhoap, J_ves_raw     = _ves_jac(tuple(ab2.tolist()), tuple(rho), tuple(thick), filt_ves)

N = len(rho)
J_ves = J_ves_raw[:, :N] * np.array(rho)[None, :] / rhoap[:, None]   # relative Jacobian

# -- Plots ---------------------------------------------------------------------
layer_lbls = [f"L{i+1}\n{rho[i]:.0f}" for i in range(N)]

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
fig.subplots_adjust(hspace=0.45, wspace=0.35)

# -- TEM row --
ax = axes[0, 0]
ax.loglog(times * 1e3, dbdt, "o-", color="mediumpurple", ms=4, lw=1.5)
ax.set_xlabel("Time (ms)")
ax.set_ylabel(r"$|\partial B_z/\partial t|$ (A/m$^2$)")
ax.set_title("TEM - decay curve")
ax.grid(True, which="both", ls="--", alpha=0.4)

ax = axes[0, 1]
vmax = max(np.abs(J_tem).max(), 1e-9)
im = ax.imshow(J_tem, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
ax.set_xticks(range(N))
ax.set_xticklabels(layer_lbls, fontsize=8)
t_ticks = list(range(0, _N_DATA, max(1, _N_DATA // 5)))
ax.set_yticks(t_ticks)
ax.set_yticklabels([f"{times[k]*1e3:.2f} ms" for k in t_ticks], fontsize=8)
ax.set_xlabel("Layer")
ax.set_ylabel("Time gate")
ax.set_title("TEM Jacobian")
plt.colorbar(im, ax=ax)

ax = axes[0, 2]
ax.bar(layer_lbls, np.linalg.norm(J_tem, axis=0), color="mediumpurple", alpha=0.8)
ax.set_ylabel("Column norm")
ax.set_title("TEM - sensitivity per layer")
ax.grid(axis="y", ls="--", alpha=0.4)

# -- VES row --
ax = axes[1, 0]
_span_ves = np.log10(rhoap.max()) - np.log10(rhoap.min())
_ctr_ves  = (np.log10(rhoap.max()) + np.log10(rhoap.min())) / 2
if _span_ves < 2.5:
    ax.set_ylim(10 ** (_ctr_ves - 1.25), 10 ** (_ctr_ves + 1.25))
ax.loglog(ab2, rhoap, "o-", color="darkorange", ms=4, lw=1.5)
ax.set_xlabel("AB/2 (m)")
ax.set_ylabel("Apparent resistivity (Ohm.m)")
ax.set_title("VES - sounding curve")
ax.grid(True, which="both", ls="--", alpha=0.4)

ax = axes[1, 1]
vmax = max(np.abs(J_ves).max(), 1e-9)
im = ax.imshow(J_ves, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
ax.set_xticks(range(N))
ax.set_xticklabels(layer_lbls, fontsize=8)
ab2_ticks = list(range(0, _N_DATA, max(1, _N_DATA // 5)))
ax.set_yticks(ab2_ticks)
ax.set_yticklabels([f"{ab2[k]:.1f} m" for k in ab2_ticks], fontsize=8)
ax.set_xlabel("Layer")
ax.set_ylabel("AB/2 spacing")
ax.set_title("VES Jacobian (relative)")
plt.colorbar(im, ax=ax)

ax = axes[1, 2]
ax.bar(layer_lbls, np.linalg.norm(J_ves, axis=0), color="darkorange", alpha=0.8)
ax.set_ylabel("Column norm")
ax.set_title("VES - sensitivity per layer")
ax.grid(axis="y", ls="--", alpha=0.4)

st.pyplot(fig)
plt.close(fig)

st.caption(
    "Red = increasing resistivity raises the data value; blue = opposite. "
    "Column norm = total information a layer contributes across all data points."
)

with st.expander(":green[Check your understanding -- quiz]"):
    col1, col2 = st.columns(2)
    with col1:
        qa = st.radio(
            ":red[**Which TEM time gates are most sensitive to deep layers?**]",
            ["Early gates (short times)", "Late gates (long times)",
             "All gates equally", "Middle gates only"],
            index=None, key="jac_q1",
        )
        if qa == "Late gates (long times)":
            st.success("Correct! The EM field diffuses deeper over time, so late gates sample deeper structure.")
        elif qa is not None:
            st.error("Think about how the electromagnetic field diffuses into the ground over time.")
    with col2:
        qb = st.radio(
            ":red[**What does a small column norm mean for a layer?**]",
            ["The layer is very resistive",
             "The layer is poorly constrained by the data",
             "The layer is too thin to detect",
             "The inversion will converge faster"],
            index=None, key="jac_q2",
        )
        if qb == "The layer is poorly constrained by the data":
            st.success("Correct! A small column norm means few data points respond to that layer.")
        elif qb is not None:
            st.error("Column norm measures how much information the data carries about a given layer.")

with st.expander(":green[Check your understanding -- quiz]"):
    col1, col2 = st.columns(2)
    with col1:
        qa = st.radio(
            ":red[**Which TEM time gates are most sensitive to deep layers?**]",
            ["Early gates (short times)", "Late gates (long times)",
             "All gates equally", "Middle gates only"],
            index=None, key="jac_q1",
        )
        if qa == "Late gates (long times)":
            st.success("Correct! The electromagnetic signal diffuses deeper over time, so late gates sample deeper structure.")
        elif qa is not None:
            st.error("Think about how the EM field diffuses into the ground over time.")
    with col2:
        qb = st.radio(
            ":red[**What does a small column norm mean for a layer?**]",
            ["The layer is very resistive",
             "The layer is poorly constrained by the data",
             "The layer is too thin to detect",
             "The inversion will converge faster"],
            index=None, key="jac_q2",
        )
        if qb == "The layer is poorly constrained by the data":
            st.success("Correct! A small column norm means few data points respond to that layer, so it cannot be recovered reliably.")
        elif qb is not None:
            st.error("The column norm is a measure of how much information the data carries about a given layer.")

with st.expander(":green[Check your understanding -- quiz]"):
    col1, col2 = st.columns(2)
    with col1:
        qa = st.radio(
            ":red[**Which TEM time gates are most sensitive to deep layers?**]",
            ["Early gates (short times)", "Late gates (long times)",
             "All gates equally", "Middle gates only"],
            index=None, key="jac_q1",
        )
        if qa == "Late gates (long times)":
            st.success("Correct! The electromagnetic signal diffuses deeper over time, so late gates sample deeper structure.")
        elif qa is not None:
            st.error("Think about how the EM field diffuses into the ground over time.")
    with col2:
        qb = st.radio(
            ":red[**What does a small column norm mean for a layer?**]",
            ["The layer is very resistive",
             "The layer is poorly constrained by the data",
             "The layer is too thin to detect",
             "The inversion will converge faster"],
            index=None, key="jac_q2",
        )
        if qb == "The layer is poorly constrained by the data":
            st.success("Correct! A small column norm means few data points respond to that layer, so it cannot be recovered reliably.")
        elif qb is not None:
            st.error("The column norm is a measure of how much information the data carries about a given layer.")
