import os
import sys

import numpy as np
import matplotlib.pyplot as plt
# -- Matplotlib font sizes (mobile-friendly) --------------------------
plt.rcParams.update({
    "font.size":       16,
    "axes.labelsize":  16,
    "axes.titlesize":  16,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 16,
})

import pandas as pd
import streamlit as st

# -- Path setup ----------------------------------------------------------------
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

_APP_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _APP_DIR not in sys.path:
    sys.path.insert(0, _APP_DIR)

from pytem import fwd_circle_central, invert as tem_invert, getJ_ana
from pytem.inversion import getR
from pytem.ip_models import tem_forward_ip, pelton_res_rho
from ves import forward as ves_forward, invert as ves_invert, jacobian as ves_jacobian, forward_ip as ves_forward_ip
from _shared import render_footer

# -- Page header ---------------------------------------------------------------
st.header(":red[Recover a resistivity model from noisy data]")

st.markdown(
    """
    This page runs a **synthetic inversion** in three steps: (1) set the model and
    system parameters and **generate clean data**, (2) **corrupt the data with
    noise**, and (3) **invert** the noisy data and compare the recovered model with
    the truth. The inversion minimises a weighted data misfit plus a smoothness
    regularisation term to find a plausible model that explains the data.
    """
)

# -- Shared staircase helper ---------------------------------------------------
def _stair(thick, rho, bottom=None, extra=200.0):
    depths = [0.0] + list(np.cumsum(thick))
    bot = bottom if bottom is not None else depths[-1] + extra
    r_s, d_s = [], []
    for i, r in enumerate(rho):
        d_top = depths[i]
        d_bot = depths[i + 1] if i < len(thick) else bot
        r_s += [r, r]
        d_s += [d_top, d_bot]
    return r_s, d_s


# Invalidate any stale cached results when the page version changes
_PAGE_VERSION = "v9"
if st.session_state.get("_inv_page_version") != _PAGE_VERSION:
    for _k in ("tem_inv_result", "ves_inv_result", "inv_result", "inv_clean", "inv_noisy"):
        st.session_state.pop(_k, None)
    st.cache_data.clear()
    st.session_state["_inv_page_version"] = _PAGE_VERSION

# ==============================================================================
# System properties
# ==============================================================================
st.subheader(':blue-background[System properties]', divider='blue')

col_tem_s, col_ves_s = st.columns(2)

with col_tem_s:
    st.markdown('**TEM**')
    tx_side     = st.number_input('Tx loop side length (m)', min_value=5.0, max_value=500.0, value=40.0, step=5.0, key='inv_tx_side')
    tx_r        = float(np.sqrt(tx_side ** 2 / np.pi))
    t_start_ms  = st.number_input('First gate (ms)', min_value=0.001, max_value=1.0,   value=0.01, format='%.3f', key='inv_t_start')
    t_end_ms    = st.number_input('Last gate (ms)',  min_value=1.0,   max_value=100.0, value=10.0, format='%.1f', key='inv_t_end')
    n_times     = int(st.number_input('Number of gates', min_value=5, max_value=60, value=20, step=2, key='inv_n_times'))
    times       = np.logspace(np.log10(t_start_ms * 1e-3), np.log10(t_end_ms * 1e-3), n_times)

with col_ves_s:
    st.markdown('**VES**')
    ab2_min     = st.number_input('AB/2 min (m)',  min_value=0.5,  max_value=100.0,  value=1.0,   step=0.5,  key='inv_ab2min')
    ab2_max     = st.number_input('AB/2 max (m)',  min_value=10.0, max_value=5000.0, value=300.0, step=10.0, key='inv_ab2max')
    n_ab2       = int(st.number_input('Number of electrode spacings', min_value=5, max_value=60, value=20, step=1, key='inv_nab2'))
    ab2         = np.logspace(np.log10(ab2_min), np.log10(ab2_max), n_ab2)

# ==============================================================================
# Model parameters
# ==============================================================================
st.subheader(':green-background[Model parameters]', divider='green')
st.markdown(
    'Define the **true earth model** used to generate synthetic data for **both** methods. '
    'TEM and VES are inverted independently and results are compared below.'
)

col_true, col_inv_s = st.columns([3, 1])
with col_true:
    st.caption('Last row is the half-space - leave its Thickness cell empty.')
    _default_true = pd.DataFrame({
            'Thickness (m)':       [10.0, 40.0, None],
            'Resistivity (Ohm.m)': [100.0, 10.0, 200.0],
        })
    _edited_true = st.data_editor(
        _default_true,
        column_config={
            'Thickness (m)':       st.column_config.NumberColumn(min_value=0.1, max_value=10000.0, format='%.1f'),
            'Resistivity (Ohm.m)': st.column_config.NumberColumn(min_value=0.01, max_value=1e6,    format='%.1f'),
        },
        num_rows='dynamic', use_container_width=True, key='inv_true_editor',
    )
    _valid_true = _edited_true.dropna(subset=['Resistivity (Ohm.m)'])
    true_thick = _valid_true['Thickness (m)'].dropna().tolist()
    true_rho   = _valid_true['Resistivity (Ohm.m)'].tolist()
    if len(true_rho) < 1:
        st.warning('Add at least one layer.')
        st.stop()

with col_inv_s:
    st.markdown('**Inversion settings**')
    start_rho = st.number_input('Starting resistivity (Ohm·m)', min_value=1.0, max_value=5000.0, value=100.0, step=10.0, key='inv_start_rho')
    st.caption('Regularisation and smoothing are set automatically from the data.')

maxit_tem = 15
maxit_ves = 15
# Fixed VES filter (gs11 â€” good balance between speed and accuracy)
_VES_FILTER = "gs11"

# ==============================================================================
# 1. Generate synthetic (clean) data
# ==============================================================================
st.subheader(":violet-background[1. Generate synthetic data]", divider="violet")
st.markdown(
    "Compute the **clean** forward response of the true model for both methods. "
    "No noise yet, this is the ideal data a perfect instrument would record."
)


@st.cache_data(show_spinner=False)
def _gen_clean(true_thick_t, true_rho_t, tx_r, times_t, ab2_t):
    true_thick = list(true_thick_t)
    true_rho   = list(true_rho_t)
    times = np.array(times_t)
    ab2   = np.array(ab2_t)
    dbdt_clean = -fwd_circle_central(true_thick, true_rho, tx_radius=tx_r, times=times)
    rhoa_clean = ves_forward(ab2, true_rho, true_thick, _VES_FILTER)
    return dbdt_clean, rhoa_clean


def _add_noise(dbdt_clean, times, b_coeff, rhoa_clean, ves_frac, seed=42):
    rng = np.random.default_rng(seed)
    # TEM: noise floor follows e = b * t^(-1/2)
    noise_std  = b_coeff * times ** (-0.5)
    dbdt_obs   = dbdt_clean + rng.normal(size=len(times)) * noise_std
    # VES: relative (log-normal) percentage error
    rhoa_sigma = ves_frac * rhoa_clean
    rhoa_obs   = rhoa_clean * np.exp(rng.normal(0.0, ves_frac, len(rhoa_clean)))
    return dbdt_obs, noise_std, rhoa_obs, rhoa_sigma


def _plot_data(clean, noisy):
    fig, (axt, axv) = plt.subplots(1, 2, figsize=(13, 5))
    _t = clean["times"]
    _a = clean["ab2"]
    axt.loglog(_t, np.abs(clean["dbdt_clean"]), "-", color="steelblue", lw=1.8, label="Clean")
    axv.loglog(_a, clean["rhoa_clean"], "-", color="darkorange", lw=1.8, label="Clean")
    if noisy is not None:
        axt.errorbar(_t, np.abs(noisy["dbdt_obs"]), yerr=noisy["noise_std"],
                     fmt="o", color="black", ms=4, ecolor="gray", elinewidth=1,
                     capsize=2, label="Noisy", zorder=4)
        axt.loglog(_t, noisy["noise_std"], "--", color="steelblue", lw=1.3,
                   label="Noise floor", zorder=3)
        axv.errorbar(_a, noisy["rhoa_obs"], yerr=noisy["rhoa_sigma"],
                     fmt="o", color="black", ms=4, ecolor="gray", elinewidth=1,
                     capsize=2, label="Noisy", zorder=4)
    axt.set_xlabel("Time [s]")
    axt.set_ylabel(r"|dB/dt| [V/m$^2$]")
    axt.set_title("TEM sounding")
    axt.grid(True, which="both", ls="--", alpha=0.4)
    axt.legend()
    axv.set_xlabel("AB/2 [m]")
    axv.set_ylabel("Apparent resistivity [Ohm.m]")
    axv.set_title("VES sounding")
    axv.grid(True, which="both", ls="--", alpha=0.4)
    axv.legend()
    fig.tight_layout()
    return fig


if st.button("Generate data", type="primary", key="inv_gen_btn"):
    _dbdt_clean, _rhoa_clean = _gen_clean(
        tuple(true_thick), tuple(true_rho), tx_r, tuple(times), tuple(ab2)
    )
    st.session_state["inv_clean"] = dict(
        times=times, ab2=ab2, dbdt_clean=_dbdt_clean, rhoa_clean=_rhoa_clean,
        true_thick=list(true_thick), true_rho=list(true_rho), tx_r=tx_r,
    )
    st.session_state.pop("inv_noisy", None)
    st.session_state.pop("inv_result", None)

if "inv_clean" not in st.session_state:
    st.info("Set the parameters above, then press **Generate data**.")
    render_footer()
    st.stop()

clean = st.session_state["inv_clean"]

# ==============================================================================
# 2. Corrupt the data with noise
# ==============================================================================
st.subheader(":violet-background[2. Corrupt the data with noise]", divider="violet")
st.markdown(
    "Real data are never clean. Add measurement noise to each sounding, then press "
    "**Add noise / update** to corrupt the synthetic data and redraw the plot."
)

col_n1, col_n2 = st.columns(2)
with col_n1:
    st.markdown(r"**TEM noise floor** &nbsp; $e = b\,t^{-1/2}$")
    log_b = st.slider("log\u2081\u2080(b)", -13.0, -8.0, -11.5, 0.1, key="inv_tem_b",
                      help="TEM noise standard deviation follows e = b\u00b7t^(-1/2), with t in seconds.")
    b_coeff = 10.0 ** log_b
    st.caption(f"b = {b_coeff:.2e}  \u2192  noise \u2248 {b_coeff * 1e-3 ** -0.5:.2e} V/m\u00b2 at 1 ms")
with col_n2:
    st.markdown("**VES noise** &nbsp; relative percentage")
    ves_pct = st.number_input("VES error (%)", min_value=1.0, max_value=30.0,
                              value=5.0, step=1.0, key="inv_ves_pct")
    ves_frac = ves_pct / 100.0

if st.button("Add noise / update", key="inv_noise_btn"):
    _dbdt_obs, _noise_std, _rhoa_obs, _rhoa_sigma = _add_noise(
        clean["dbdt_clean"], clean["times"], b_coeff,
        clean["rhoa_clean"], ves_frac,
    )
    st.session_state["inv_noisy"] = dict(
        dbdt_obs=_dbdt_obs, noise_std=_noise_std,
        rhoa_obs=_rhoa_obs, rhoa_sigma=_rhoa_sigma, ves_frac=ves_frac,
    )
    st.session_state.pop("inv_result", None)

noisy = st.session_state.get("inv_noisy")
st.pyplot(_plot_data(clean, noisy))
plt.close("all")
if noisy is None:
    st.caption("Currently showing **clean** data. Press **Add noise / update** to corrupt it.")

# ==============================================================================
# 3. Run the inversion
# ==============================================================================
st.subheader(":violet-background[3. Run the inversion]", divider="violet")


@st.cache_data(show_spinner=False)
def _run_inv(dbdt_obs_t, noise_std_t, times_t, tx_r,
             rhoa_obs_t, ab2_t, ves_frac, start_rho):
    dbdt_obs  = np.array(dbdt_obs_t)
    noise_std = np.array(noise_std_t)
    times     = np.array(times_t)
    rhoa_obs  = np.array(rhoa_obs_t)
    ab2       = np.array(ab2_t)

    # --- TEM inversion --------------------------------------------------------
    depths_tem  = np.logspace(np.log10(2), np.log10(250), 19)
    thick_tem   = np.diff(np.concatenate([[0.0], depths_tem])).tolist()   # 19
    log_rho_tem = np.log(np.full(20, start_rho))                          # 20
    res_tem = tem_invert(
        obs_data=dbdt_obs, thicknesses=thick_tem,
        log_resistivities=log_rho_tem, tx_size=tx_r, times=times,
        noise_std=noise_std, alpha_steps=10, maxit=50,
        max_noise_frac=0.0,
        transform="dlf", hankel_filter="key_101", fourier_filter="key_81",
        analytical_j=True,
    )
    dbdt_pred = -fwd_circle_central(
        thick_tem, res_tem["resistivities"].tolist(), tx_radius=tx_r, times=times,
    )

    # --- VES inversion --------------------------------------------------------
    depth_max = ab2[-1] / 3.0
    deps_ves  = np.logspace(np.log10(max(ab2[0] * 0.5, 0.5)), np.log10(max(depth_max, 1.0)), 15)
    thick_ves = np.diff(np.concatenate([[0.0], deps_ves])).tolist()
    rho0_ves  = np.full(16, start_rho)
    res_ves = ves_invert(
        ab2=ab2, rhoap_obs=rhoa_obs,
        resistivities=rho0_ves, thicknesses=thick_ves,
        regularization="auto", iter_max=50, filter_set=_VES_FILTER,
        fix_thicknesses=True, noise_frac=ves_frac,
    )

    return (
        dbdt_pred, thick_tem, res_tem["resistivities"], res_tem["rms_history"],
        res_ves["rhoap_pred"], thick_ves, res_ves["resistivities"], res_ves["rms_history"],
    )


if noisy is None:
    st.info("Add noise to the data first, then run the inversion.")
    render_footer()
    st.stop()

if st.button("Run inversions", type="primary", key="inv_run_btn"):
    with st.spinner("Running TEM and VES inversions..."):
        st.session_state["inv_result"] = _run_inv(
            tuple(noisy["dbdt_obs"]), tuple(noisy["noise_std"]), tuple(clean["times"]),
            clean["tx_r"], tuple(noisy["rhoa_obs"]), tuple(clean["ab2"]),
            noisy["ves_frac"], start_rho,
        )

if "inv_result" not in st.session_state:
    st.info("Press **Run inversions** to fit both datasets.")
    render_footer()
    st.stop()

(dbdt_pred, thick_tem_r, rho_tem_r, rms_tem,
 rhoap_pred, thick_ves_r, rho_ves_r, rms_ves) = st.session_state["inv_result"]

times      = clean["times"]
ab2        = clean["ab2"]
true_thick = clean["true_thick"]
true_rho   = clean["true_rho"]
dbdt_obs   = noisy["dbdt_obs"]
rhoap_obs  = noisy["rhoa_obs"]

# -- Metrics -------------------------------------------------------------------
col_m1, col_m2, col_m3, col_m4 = st.columns(4)
col_m1.metric("TEM final RMS", f"{rms_tem[-1]:.3f}" if rms_tem else "-", help="Target ~1.0")
col_m2.metric("TEM iterations", len(rms_tem))
_ves_rms_norm = (rms_ves[-1] / noisy["ves_frac"]) if rms_ves else None
col_m3.metric("VES final RMS", f"{_ves_rms_norm:.3f}" if _ves_rms_norm is not None else "-", help="Target ~1.0")
col_m4.metric("VES iterations", len(rms_ves))

# -- Results grid: 2 data fits + combined model --------------------------------
fig = plt.figure(figsize=(14, 13))
gs  = fig.add_gridspec(3, 2, hspace=0.42, wspace=0.3)
ax_tem_data = fig.add_subplot(gs[0, 0])
ax_ves_data = fig.add_subplot(gs[0, 1])
ax_model    = fig.add_subplot(gs[1:, :])

# TEM data fit
ax = ax_tem_data
ax.loglog(times, np.abs(dbdt_pred), color="steelblue", linestyle="-", lw=1.5, label="Predicted TEM", zorder=3)
ax.loglog(times, np.abs(dbdt_obs),  color="black", marker="o", linestyle="None", ms=4,   label="Observed TEM", zorder=4)
#ax.loglog(times, noisy["noise_std"], "--", color="steelblue", lw=1.2, label="Noise floor", zorder=2)
ax.set_xlabel("Time [s]")
ax.set_ylabel(r"|dB/dt| [V/m$^2$]")
ax.grid(True, which="both", ls="--", alpha=0.4)
ax.legend()

# VES data fit
ax = ax_ves_data
_rho_all = np.concatenate([rhoap_obs, rhoap_pred])
_span = np.log10(_rho_all.max()) - np.log10(_rho_all.min())
_ctr  = (np.log10(_rho_all.max()) + np.log10(_rho_all.min())) / 2
if _span < 2.5:
    _rlo, _rhi = 10 ** (_ctr - 1.25), 10 ** (_ctr + 1.25)
else:
    _rlo, _rhi = _rho_all.min() * 0.8, _rho_all.max() * 1.25
ax.loglog(ab2, rhoap_pred, color="darkorange", linestyle="-", lw=1.5, label="Predicted VES", zorder=3)
ax.loglog(ab2, rhoap_obs,  color="black", marker="o", linestyle="None", ms=4,   label="Observed VES", zorder=4)
ax.set_ylim(_rlo, _rhi)
ax.set_xlabel("AB/2 [m]")
ax.set_ylabel("Apparent resistivity [Ohm.m]")
ax.grid(True, which="both", ls="--", alpha=0.4)
ax.legend()

# Combined recovered model
ax = ax_model
_max_depth = max(float(np.sum(true_thick)), float(np.sum(thick_tem_r)), float(np.sum(thick_ves_r)))
r_true, d_true = _stair(true_thick, true_rho, bottom=_max_depth)
r_tem,  d_tem  = _stair(list(thick_tem_r), list(rho_tem_r), bottom=_max_depth)
r_ves,  d_ves  = _stair(list(thick_ves_r), list(rho_ves_r), bottom=_max_depth)
ax.semilogx(r_true, d_true, color="black", linestyle="--", lw=2.5, label="True model")
ax.semilogx(r_tem,  d_tem,  color="steelblue", linestyle="-", lw=2,   label="TEM recovered")
ax.semilogx(r_ves,  d_ves,  color="darkorange", linestyle="-", lw=2,   label="VES recovered")
ax.invert_yaxis()
_all_r = r_true + r_tem + r_ves
_ctr = (np.log10(min(_all_r)) + np.log10(max(_all_r))) / 2
_span = max(np.log10(max(_all_r)) - np.log10(min(_all_r)), 2.5)
ax.set_xlim(10 ** (_ctr - _span / 2), 10 ** (_ctr + _span / 2))
ax.set_xlabel("Resistivity [Ohm.m]")
ax.set_ylabel("Depth [m]")
ax.grid(True, which="both", ls="--", alpha=0.4)
ax.legend()
fig.tight_layout()

st.pyplot(fig)
plt.close(fig)

with st.expander("How to read these results"):
    st.markdown(
        """
        **Top panels - data fit.** Each plot overlays the *observed* data
        (points) with the *predicted* response of the recovered model (line).
        When the line threads through the scatter, the model explains the data
        to the noise level. The headline number is the **RMS misfit**: a value
        near **1.0** means the model fits the data about as well as the noise
        allows. Much above 1.0 means underfitting; well below 1.0 means the
        model is chasing noise.

        **Bottom panel - recovered models.** The dashed black line is the
        **true model**; the coloured lines are what TEM and VES each recovered.
        Things to look for:

        - **TEM (blue)** locks onto the **conductive** layer and its depth, but
          tends to smear resistive units, they barely affect the curve.
        - **VES (orange)** expresses **resistive** layers more sharply, but loses
          resolution with depth as electrode spacings grow.
        - **Smoothing** makes both recovered models gradational: sharp true
          boundaries appear as ramps, not steps.
        - **Non-uniqueness:** a different model can fit equally well (equivalence).
          Agreement between two independent methods is the strongest evidence
          that a feature is real, which is exactly why TEM and VES are combined.
        """
    )

render_footer()

