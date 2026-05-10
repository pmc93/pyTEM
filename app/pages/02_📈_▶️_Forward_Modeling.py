import os
import sys

import numpy as np
import matplotlib.pyplot as plt
# -- Matplotlib font sizes (mobile-friendly) --------------------------
plt.rcParams.update({
    "axes.labelsize":  14,
    "axes.titlesize":  15,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 11,
})

import streamlit as st

# ── Path setup ────────────────────────────────────────────────────────────────
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from pytem import fwd_circle_central
from ves import forward as ves_forward

# ── Shared utilities ──────────────────────────────────────────────────────────
_RHO = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000]


def _model_ui(prefix, n, def_rho, def_h):
    """Labelled slider per layer. Returns (thicknesses, resistivities)."""
    h_out, r_out = [], []
    for i in range(n):
        label = f"Layer {i+1}" if i < n - 1 else "Half-space"
        st.markdown(f"**{label}**")
        if i < n - 1:
            h_def = int(def_h[i]) if i < len(def_h) else 20
            h_out.append(float(st.slider(f"Thickness (m)", 1, 500, h_def,
                                         key=f"{prefix}_h{i}")))
        rho_def = min(_RHO, key=lambda x: abs(x - (def_rho[i] if i < len(def_rho) else 100)))
        r_out.append(float(st.select_slider(f"Resistivity (Ohm.m)", _RHO, value=rho_def,
                                            key=f"{prefix}_r{i}")))
    return h_out, r_out


def _stair(thick, rho):
    depths = [0.0] + list(np.cumsum(thick))
    bot = depths[-1] + max(depths[-1] * 0.3, 20.0)
    r_s, d_s = [], []
    for i, r in enumerate(rho):
        d_top = depths[i]
        d_bot = depths[i + 1] if i < len(thick) else bot
        r_s += [r, r]
        d_s += [d_top, d_bot]
    return r_s, d_s


# ── Cached forward calls ──────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def _tem_fwd(h_t, rho_t, r, times_t):
    return -fwd_circle_central(list(h_t), list(rho_t),
                               tx_radius=r, times=np.array(times_t))


@st.cache_data(show_spinner=False)
def _ves_fwd(ab2_t, rho_t, h_t, filt):
    return ves_forward(np.array(ab2_t), np.array(rho_t), np.array(h_t), filt)


# ── Page header ───────────────────────────────────────────────────────────────
st.header(":blue[Predicted response for a layered earth model]")
st.markdown(
    "Build a layered resistivity model and see the predicted sounding curve "
    "update in real time. Each tab is independent; you can explore different "
    "models for TEM and VES."
)

tab_tem, tab_ves = st.tabs(["🧲 TEM", "⚡️ VES"])

# ═══════════════════════════════════════════════════════════════════════════════
# TEM TAB
# ═══════════════════════════════════════════════════════════════════════════════
with tab_tem:
    st.subheader(":blue-background[TEM - dB/dt sounding]", divider="blue")

    st.markdown("**System Parameters**")
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        tx_area = st.number_input("Tx loop area (m²)", min_value=100, max_value=500000, value=1600, step=100, key="fwd_tem_area")
        tx_r = float(np.sqrt(tx_area / np.pi))
        n_t = int(st.number_input("Time gates", 5, 50, 25, key="fwd_tem_nt"))
    with col_s2:
        t_min = st.slider("Early time (10^x s)", -6.0, -4.0, -5.0, 0.25, key="fwd_tem_tmin")
        t_max = st.slider("Late time (10^x s)", -3.0, -1.0, -2.0, 0.25, key="fwd_tem_tmax")

    st.markdown("**Layer model**")
    n_tem = int(st.number_input("Number of layers", 2, 6, 3, key="fwd_tem_n"))
    t_thick, t_rho = _model_ui("fwd_tem", n_tem,
                                [100, 10, 300], [20, 50])

    times = np.logspace(t_min, t_max, n_t)

    st.button("🧮 Compute forward model", key="fwd_tem_btn", type="primary")

    try:
        with st.spinner("Computing …"):
            dbdt = _tem_fwd(tuple(t_thick), tuple(t_rho), tx_r, tuple(times.tolist()))

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        ax1.loglog(times * 1e3, dbdt, "o-", color="steelblue", ms=4, lw=1.5)
        ax1.set_xlabel("Time (ms)")
        ax1.set_ylabel(r"|dB/dt| [A/m$^2$]")
        ax1.grid(True, which="both", ls="--", alpha=0.4)

        rs, ds = _stair(t_thick, t_rho)
        _span_m = max(rs) / min(r for r in rs if r > 0)
        if _span_m < 10**2.5:
            _ctr_m = (max(rs) * min(r for r in rs if r > 0)) ** 0.5
            _mlo, _mhi = _ctr_m / 10**1.25, _ctr_m * 10**1.25
        else:
            _mlo, _mhi = min(r for r in rs if r > 0) * 0.8, max(rs) * 1.25
        ax2.semilogx(rs, ds, color="steelblue", lw=2)
        #ax2.fill_betweenx(ds, rs, alpha=0.15, color="steelblue")
        ax2.set_xlim(_mlo, _mhi)
        ax2.invert_yaxis()
        ax2.set_xlabel(r"Resistivity [Ohm.m]")
        ax2.set_ylabel("Depth [m]")
        ax2.grid(True, which="both", ls="--", alpha=0.4)

        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    except Exception as _e:
        st.warning(f"⚠️ Could not compute: {_e}. Adjust the sliders and click **🧮 Compute forward model**.")

# ═══════════════════════════════════════════════════════════════════════════════
# VES TAB
# ═══════════════════════════════════════════════════════════════════════════════
with tab_ves:
    st.subheader(":orange-background[VES - Apparent resistivity sounding]", divider="orange")

    st.markdown("**Survey Parameters**")
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        ab2_min = st.slider("AB/2 minimum (m)", 1, 30, 1, key="fwd_ves_ab2min")
        ab2_max = st.slider("AB/2 maximum (m)", 50, 2000, 300, key="fwd_ves_ab2max")
        n_ab2 = int(st.number_input("AB/2 points", 5, 60, 25, key="fwd_ves_nab2"))
        filt = "gs11"

    st.markdown("**Layer model**")
    n_ves = int(st.number_input("Number of layers", 2, 6, 3, key="fwd_ves_n"))
    v_thick, v_rho = _model_ui("fwd_ves", n_ves,
                                [100, 20, 200], [10, 30])

    ab2 = np.logspace(np.log10(ab2_min), np.log10(ab2_max), n_ab2)

    st.button("📊 Compute forward model", key="fwd_ves_btn", type="primary",
              help="Manually trigger computation (also updates automatically on slider change)")

    try:
        with st.spinner("Computing …"):
            rhoap = _ves_fwd(tuple(ab2.tolist()), tuple(v_rho), tuple(v_thick), filt)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        _span_v = np.log10(rhoap.max()) - np.log10(rhoap.min())
        if _span_v < 2.5:
            _ctr_v = (np.log10(rhoap.max()) + np.log10(rhoap.min())) / 2
            _vlo, _vhi = 10 ** (_ctr_v - 1.25), 10 ** (_ctr_v + 1.25)
        else:
            _vlo, _vhi = rhoap.min() * 0.8, rhoap.max() * 1.25
        ax1.loglog(ab2, rhoap, "o-", color="darkorange", ms=4, lw=1.5)
        ax1.set_ylim(_vlo, _vhi)
        ax1.set_xlabel(r"$AB/2$ [m]")
        ax1.set_ylabel("Apparent resistivity [Ohm.m]")
        ax1.grid(True, which="both", ls="--", alpha=0.4)

        rs, ds = _stair(v_thick, v_rho)
        _span_m = max(rs) / min(r for r in rs if r > 0)
        if _span_m < 10**2.5:
            _ctr_m = (max(rs) * min(r for r in rs if r > 0)) ** 0.5
            _mlo, _mhi = _ctr_m / 10**1.25, _ctr_m * 10**1.25
        else:
            _mlo, _mhi = min(r for r in rs if r > 0) * 0.8, max(rs) * 1.25
        ax2.semilogx(rs, ds, color="darkorange", lw=2)
        #ax2.fill_betweenx(ds, rs, alpha=0.15, color="darkorange")
        ax2.set_xlim(_mlo, _mhi)
        ax2.invert_yaxis()
        ax2.set_xlabel(r"Resistivity [Ohm.m]")
        ax2.set_ylabel("Depth [m]")
        ax2.grid(True, which="both", ls="--", alpha=0.4)

        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    except Exception as _e:
        st.warning(f"⚠️ Could not compute: {_e}. Adjust the sliders and click **📊 Compute forward model**.")
