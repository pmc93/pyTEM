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

import pandas as pd
import streamlit as st

# -- Path setup ----------------------------------------------------------------
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from pytem import fwd_circle_central, invert as tem_invert, getJ_ana
from pytem.inversion import getR
from pytem.ip_models import tem_forward_ip, pelton_res_rho
from ves import forward as ves_forward, invert as ves_invert, jacobian as ves_jacobian, forward_ip as ves_forward_ip

# -- Page header ---------------------------------------------------------------
st.title("Inversion")
st.header(":red[Recover a resistivity model from noisy data]")

st.markdown(
    r"""
    This page runs a **synthetic inversion**: generate noisy data from a known
    **true model**, start from an **initial model**, and let the algorithm
    iterate until it converges.

    **The objective function** minimised at each iteration is:

$$\phi(\mathbf{m}) = \left\|\mathbf{W}\!\left(\mathbf{d}_{obs} - \mathbf{d}_{pred}(\mathbf{m})\right)\right\|^2 + \alpha\,\mathbf{m}^T\!\mathbf{R}\,\mathbf{m}$$

    where $\mathbf{W}$ is a noise-weighting matrix, $\mathbf{R}$ is a roughness
    regularisation matrix, and $\alpha$ is the regularisation parameter.
    """
)

with st.expander(":green[Check your understanding -- quiz]"):
    col1, col2 = st.columns(2)
    with col1:
        q1 = st.radio(
            ":red[**What does RMS = 1 mean in an inversion?**]",
            [
                "The model perfectly fits the data",
                "The data fit is consistent with the assumed noise level",
                "The regularisation is too strong",
                "The inversion has not converged",
            ],
            index=None,
        )
        if q1 == "The data fit is consistent with the assumed noise level":
            st.success("Correct! RMS = 1 means the residuals are on average equal to one standard deviation -- a statistically ideal fit.")
        elif q1 is not None:
            st.error("RMS = 1 is the target: the model fits the data to within the assumed noise, no more, no less.")
    with col2:
        q2 = st.radio(
            ":red[**What happens if the regularisation parameter is too large?**]",
            [
                "The model is too rough (overfits noise)",
                "The model is too smooth (underfits data)",
                "The algorithm diverges",
                "The computation becomes faster",
            ],
            index=None,
        )
        if q2 == "The model is too smooth (underfits data)":
            st.success("Correct! Large regularisation penalises model roughness so strongly that real structure is smoothed out.")
        elif q2 is not None:
            st.error("Large regularisation forces the model to be very smooth, even if the data demand more structure.")


# -- Shared staircase helper ---------------------------------------------------
def _stair(thick, rho, extra=200.0):
    depths = [0.0] + list(np.cumsum(thick))
    bot = depths[-1] + extra
    r_s, d_s = [], []
    for i, r in enumerate(rho):
        d_top = depths[i]
        d_bot = depths[i + 1] if i < len(thick) else bot
        r_s += [r, r]
        d_s += [d_top, d_bot]
    return r_s, d_s


# Invalidate any stale cached results when the page version changes
_PAGE_VERSION = "v8"
if st.session_state.get("_inv_page_version") != _PAGE_VERSION:
    for _k in ("tem_inv_result", "ves_inv_result", "joint_inv_result"):
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
    tx_area     = st.number_input('Tx loop area (m²)', min_value=100.0, max_value=1e6, value=1600.0, step=100.0, key='inv_tx_area')
    tx_r        = float(np.sqrt(tx_area / np.pi))
    t_start_ms  = st.number_input('First gate (ms)', min_value=0.001, max_value=1.0,   value=0.01, format='%.3f', key='inv_t_start')
    t_end_ms    = st.number_input('Last gate (ms)',  min_value=1.0,   max_value=100.0, value=10.0, format='%.1f', key='inv_t_end')
    n_times     = int(st.number_input('Number of gates', min_value=5, max_value=60, value=20, step=2, key='inv_n_times'))
    times       = np.logspace(np.log10(t_start_ms * 1e-3), np.log10(t_end_ms * 1e-3), n_times)
    noise_tem_pct = st.number_input('Measurement error (%)', min_value=1, max_value=30, value=5, step=1, key='inv_tem_noise')

with col_ves_s:
    st.markdown('**VES**')
    ab2_min     = st.number_input('AB/2 min (m)',  min_value=0.5,  max_value=100.0,  value=1.0,   step=0.5,  key='inv_ab2min')
    ab2_max     = st.number_input('AB/2 max (m)',  min_value=10.0, max_value=5000.0, value=300.0, step=10.0, key='inv_ab2max')
    n_ab2       = int(st.number_input('Number of electrode spacings', min_value=5, max_value=60, value=20, step=1, key='inv_nab2'))
    ab2         = np.logspace(np.log10(ab2_min), np.log10(ab2_max), n_ab2)
    noise_ves_pct = st.number_input('Measurement error (%)', min_value=0, max_value=30, value=5, step=1, key='inv_ves_noise')

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
# Cached inversion runner
# ==============================================================================
@st.cache_data(show_spinner=False)
def _run_inv(true_thick_t, true_rho_t,
             tx_r, times_t, noise_tem_frac,
             ab2_t, noise_ves_frac,
             start_rho, seed=42):
    true_thick = list(true_thick_t)
    true_rho   = list(true_rho_t)
    times      = np.array(times_t)
    ab2        = np.array(ab2_t)
    rng        = np.random.default_rng(seed)
    # --- TEM forward + noise --------------------------------------------------
    dbdt_fwd     = -fwd_circle_central(true_thick, true_rho, tx_radius=tx_r, times=times)
    noise_at_1ms = 1e-10
    noise_floor  = noise_at_1ms * (times / 1e-3) ** (-0.5)
    noise_std    = np.sqrt((noise_tem_frac * dbdt_fwd) ** 2 + noise_floor ** 2)
    dbdt_obs     = dbdt_fwd + rng.normal(size=len(times)) * noise_std

    depths_tem   = np.logspace(np.log10(2), np.log10(250), 19)
    thick_tem    = np.diff(np.concatenate([[0.0], depths_tem])).tolist()   # 19
    log_rho_tem  = np.log(np.full(20, start_rho))                          # 20

    res_tem = tem_invert(
        obs_data=dbdt_obs, thicknesses=thick_tem,
        log_resistivities=log_rho_tem, tx_radius=tx_r, times=times,
        noise_std=noise_std, alpha_steps=10, maxit=15,
        max_noise_frac=0.0,
        transform="dlf", hankel_filter="key_101", fourier_filter="key_81",
        analytical_j=True,
    )
    dbdt_pred = -fwd_circle_central(
        thick_tem, res_tem["resistivities"].tolist(), tx_radius=tx_r, times=times
    )

    # --- VES forward + noise --------------------------------------------------
    rhoap_true = ves_forward(ab2, true_rho, true_thick, _VES_FILTER)
    rhoap_obs  = rhoap_true * np.exp(rng.normal(0.0, noise_ves_frac, len(ab2)))

    depth_max  = ab2[-1] / 3.0
    deps_ves   = np.logspace(np.log10(max(ab2[0] * 0.5, 0.5)), np.log10(max(depth_max, 1.0)), 15)
    thick_ves  = np.diff(np.concatenate([[0.0], deps_ves])).tolist()
    rho0_ves   = np.full(16, start_rho)

    res_ves = ves_invert(
        ab2=ab2, rhoap_obs=rhoap_obs,
        resistivities=rho0_ves, thicknesses=thick_ves,
        regularization="auto", iter_max=15, filter_set=_VES_FILTER,
        fix_thicknesses=True, noise_frac=noise_ves_frac,
    )

    return (
        dbdt_obs, dbdt_fwd, dbdt_pred,
        thick_tem, res_tem["resistivities"], res_tem["rms_history"],
        rhoap_obs, rhoap_true, res_ves["rhoap_pred"],
        thick_ves, res_ves["resistivities"], res_ves["rms_history"],
    )

run_btn = st.button("Run both inversions", type="primary", key="inv_run_btn")

if run_btn or "joint_inv_result" in st.session_state:
    if run_btn:
        with st.spinner("Running TEM and VES inversions..."):
            _res = _run_inv(
                tuple(true_thick), tuple(true_rho),
                tx_r, tuple(times), noise_tem_pct / 100.0,
                tuple(ab2), noise_ves_pct / 100.0,
                start_rho,
            )
        st.session_state["joint_inv_result"] = _res
    else:
        _res = st.session_state["joint_inv_result"]

    (dbdt_obs, dbdt_fwd, dbdt_pred,
     thick_tem_r, rho_tem_r, rms_tem,
     rhoap_obs, rhoap_true, rhoap_pred,
     thick_ves_r, rho_ves_r, rms_ves) = _res

    # -- Metrics ---------------------------------------------------------------
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    col_m1.metric("TEM final RMS", f"{rms_tem[-1]:.3f}" if rms_tem else "-", help="Target ~1.0")
    col_m2.metric("TEM iterations", len(rms_tem))
    _ves_rms_norm = (rms_ves[-1] / (noise_ves_pct / 100.0)) if rms_ves else None
    col_m3.metric("VES final RMS", f"{_ves_rms_norm:.3f}" if _ves_rms_norm is not None else "-", help="Target ~1.0")
    col_m4.metric("VES iterations", len(rms_ves))

    # -- 2Ã—2 results grid: 2 data fits + combined model -------------------------
    fig = plt.figure(figsize=(14, 13))
    gs  = fig.add_gridspec(3, 2, hspace=0.42, wspace=0.3)
    ax_tem_data = fig.add_subplot(gs[0, 0])
    ax_ves_data = fig.add_subplot(gs[0, 1])
    ax_model    = fig.add_subplot(gs[1:, :])

    # TEM data fit
    ax = ax_tem_data
    ax.loglog(times * 1e3, dbdt_pred, "r-",  lw=1.5, label="Predicted", zorder=3)
    ax.loglog(times * 1e3, dbdt_obs,  "ko",  ms=4,   label="Observed (noisy)", zorder=4)
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel(r"$|\partial B_z/\partial t|$ (A/m$^2$)")
    ax.set_title("TEM - data fit")
    ax.grid(True, which="both", ls="--", alpha=0.4)
    ax.legend(fontsize=8)

    # VES data fit
    ax = ax_ves_data
    _rho_all = np.concatenate([rhoap_obs, rhoap_pred])
    _span = np.log10(_rho_all.max()) - np.log10(_rho_all.min())
    _ctr  = (np.log10(_rho_all.max()) + np.log10(_rho_all.min())) / 2
    if _span < 2.5:
        _rlo, _rhi = 10 ** (_ctr - 1.25), 10 ** (_ctr + 1.25)
    else:
        _rlo, _rhi = _rho_all.min() * 0.8, _rho_all.max() * 1.25
    ax.loglog(ab2, rhoap_pred, "b-",  lw=1.5, label="Predicted", zorder=3)
    ax.loglog(ab2, rhoap_obs,  "ko",  ms=4,   label="Observed (noisy)", zorder=4)
    ax.set_ylim(_rlo, _rhi)
    ax.set_xlabel("AB/2 (m)")
    ax.set_ylabel("Apparent resistivity (Ohm.m)")
    ax.set_title("VES - data fit")
    ax.grid(True, which="both", ls="--", alpha=0.4)
    ax.legend(fontsize=8)

    # Combined recovered model
    ax = ax_model
    r_true, d_true = _stair(true_thick, true_rho)
    r_tem,  d_tem  = _stair(list(thick_tem_r), list(rho_tem_r))
    r_ves,  d_ves  = _stair(list(thick_ves_r), list(rho_ves_r))
    ax.semilogx(r_true, d_true, "g--", lw=2.5, label="True model")
    ax.semilogx(r_tem,  d_tem,  "r-",  lw=2,   label="TEM recovered")
    ax.semilogx(r_ves,  d_ves,  "b-",  lw=2,   label="VES recovered")
    ax.invert_yaxis()
    ax.set_xlabel("Resistivity (Ohm.m)")
    ax.set_ylabel("Depth (m)")
    ax.set_title("Recovered model: TEM and VES")
    ax.grid(True, which="both", ls="--", alpha=0.4)
    ax.legend(fontsize=8)

    st.pyplot(fig)
    plt.close(fig)

else:
    st.info("Press **Run both inversions** to start.")

