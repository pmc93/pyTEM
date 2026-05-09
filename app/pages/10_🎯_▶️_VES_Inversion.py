import os
import sys
import copy

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

# ── Path setup ────────────────────────────────────────────────────────────────
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_VES_ROOT = os.path.join(_ROOT, "ves")
for _p in [_ROOT, _VES_ROOT]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from forward import SLB
from fitting import SLB_LSInv

# ── Page header ───────────────────────────────────────────────────────────────
st.title("🎯 VES Inversion ▶️")
st.header(":blue[Recover a resistivity model from apparent resistivity data]")

st.markdown(
    r"""
    This page runs a **synthetic inversion**: a known **true model** generates
    a synthetic sounding curve, Gaussian noise is added, then both the
    **Levenberg-Marquardt (LM)** and **SVD** methods invert the noisy data
    starting from the same initial model.

    The model update at each iteration is:

    $$\Delta\mathbf{m} = \left(\mathbf{J}^T\mathbf{J} + \mu\,\mathbf{I}\right)^{-1}\mathbf{J}^T\,\mathbf{d} \quad \text{(LM)}$$

    $$\Delta\mathbf{m} = \mathbf{V}\,\mathrm{diag}\!\left(\frac{s_k}{s_k+\mu}\right)\mathbf{U}^T\,\mathbf{d} \quad \text{(SVD)}$$

    The Jacobian $\mathbf{J}$ is built by finite differences with perturbation
    fraction $\varepsilon$, and $\mu$ is the damping parameter.
    """
)

with st.expander(":green[**Check your understanding — quiz**]"):
    col1, col2 = st.columns(2)
    with col1:
        q1 = st.radio(
            ":blue[**Why do both methods use a damping parameter μ?**]",
            [
                "To speed up forward model evaluation",
                "To prevent the model from leaving the starting value",
                "To stabilise the solution when J^T J is near-singular",
                "To enforce a smoothness constraint on resistivity",
            ],
            index=None,
        )
        if q1 == "To stabilise the solution when J^T J is near-singular":
            st.success("Correct! The μI term regularises the matrix inversion, preventing large unstable steps.")
        elif q1 is not None:
            st.error("The damping μ keeps (J^T J + μI) well-conditioned when the Jacobian is rank-deficient.")

    with col2:
        q2 = st.radio(
            ":blue[**What does SVD add compared to plain LM?**]",
            [
                "It runs the forward model fewer times",
                "It naturally tapers poorly constrained model directions",
                "It avoids computing the Jacobian entirely",
                "It guarantees a globally optimal solution",
            ],
            index=None,
        )
        if q2 == "It naturally tapers poorly constrained model directions":
            st.success("Correct! The s_k/(s_k+μ) factors suppress directions with small singular values (poor sensitivity).")
        elif q2 is not None:
            st.error("SVD exposes which combinations of model parameters are well or poorly constrained by the data.")

# ── True model ────────────────────────────────────────────────────────────────
st.subheader(":blue-background[True (synthetic) model]", divider="blue")
st.markdown(
    "Define the **true earth model** used to generate synthetic data. "
    "Gaussian noise is added at the level set on the right."
)

col_true, col_survey = st.columns([3, 2])
with col_true:
    st.caption("Last row is the half-space — leave its Thickness cell empty.")
    _default_true = pd.DataFrame({
        "Thickness (m)": [10.0, 30.0, None],
        "Resistivity (Ω·m)": [150.0, 20.0, 300.0],
    })
    _edited_true = st.data_editor(
        _default_true,
        column_config={
            "Thickness (m)": st.column_config.NumberColumn(
                min_value=0.1, max_value=5000.0, format="%.1f",
            ),
            "Resistivity (Ω·m)": st.column_config.NumberColumn(
                min_value=0.01, max_value=1e6, format="%.2f",
            ),
        },
        num_rows="dynamic",
        use_container_width=True,
        key="ves_inv_true",
    )
    _valid_true = _edited_true.dropna(subset=["Resistivity (Ω·m)"])
    true_thick = _valid_true["Thickness (m)"].dropna().tolist()
    true_rho = _valid_true["Resistivity (Ω·m)"].tolist()
    if len(true_rho) < 1:
        st.warning("Add at least one layer.")
        st.stop()
    if len(true_thick) != len(true_rho) - 1:
        st.warning("Number of thicknesses must equal number of resistivities minus 1.")
        st.stop()

with col_survey:
    st.markdown("**Survey settings**")
    ab2_min = st.number_input("AB/2 min (m)", min_value=0.5, max_value=100.0, value=1.0, step=0.5, key="ves_inv_ab2min")
    ab2_max = st.number_input("AB/2 max (m)", min_value=10.0, max_value=5000.0, value=200.0, step=10.0, key="ves_inv_ab2max")
    n_ab2 = int(st.number_input("Number of AB/2 points", min_value=5, max_value=80, value=20, step=5, key="ves_inv_nab2"))
    noise_pct = st.number_input("Noise level (%)", min_value=0, max_value=30, value=5, step=1, key="ves_inv_noise")
    filter_coeff = st.selectbox("Filter", ["guptasarma_7", "guptasarma_11", "guptasarma_22"], index=0, key="ves_inv_filter")

if ab2_min >= ab2_max:
    st.error("AB/2 minimum must be less than maximum.")
    st.stop()

ab2 = np.logspace(np.log10(ab2_min), np.log10(ab2_max), n_ab2)

# ── Starting model ────────────────────────────────────────────────────────────
st.subheader(":blue-background[Starting model]", divider="blue")
st.markdown(
    "Both inversions start from this same initial model. "
    "A homogeneous half-space with a few logarithmically spaced layers is typical."
)

col_s1, col_s2 = st.columns([3, 2])
with col_s1:
    st.caption("Last row is the half-space — leave its Thickness cell empty.")
    _default_start = pd.DataFrame({
        "Thickness (m)": [20.0, 50.0, None],
        "Resistivity (Ω·m)": [100.0, 100.0, 100.0],
    })
    _edited_start = st.data_editor(
        _default_start,
        column_config={
            "Thickness (m)": st.column_config.NumberColumn(
                min_value=0.1, max_value=5000.0, format="%.1f",
            ),
            "Resistivity (Ω·m)": st.column_config.NumberColumn(
                min_value=0.01, max_value=1e6, format="%.2f",
            ),
        },
        num_rows="dynamic",
        use_container_width=True,
        key="ves_inv_start",
    )
    _valid_start = _edited_start.dropna(subset=["Resistivity (Ω·m)"])
    start_thick = _valid_start["Thickness (m)"].dropna().tolist()
    start_rho = _valid_start["Resistivity (Ω·m)"].tolist()
    if len(start_rho) < 1:
        st.warning("Add at least one layer to the starting model.")
        st.stop()
    if len(start_thick) != len(start_rho) - 1:
        st.warning("Starting model: thicknesses must equal resistivities minus 1.")
        st.stop()

# ── Inversion settings ────────────────────────────────────────────────────────
st.subheader(":blue-background[Inversion settings]", divider="blue")

col_lm, col_svd = st.columns(2)
with col_lm:
    st.markdown("**Levenberg-Marquardt**")
    lm_damping = st.number_input("Damping μ", min_value=1e-6, max_value=10.0, value=0.01, format="%.4f", key="lm_damp")
    lm_epsilon = st.number_input("Perturbation ε", min_value=0.001, max_value=0.5, value=0.01, format="%.4f", key="lm_eps")
    lm_itermax = int(st.number_input("Max iterations", min_value=1, max_value=200, value=30, step=5, key="lm_iter"))
    lm_errmin = st.number_input("RMS stop threshold", min_value=0.001, max_value=50.0, value=1.0, format="%.3f", key="lm_err")

with col_svd:
    st.markdown("**SVD**")
    svd_damping = st.number_input("Damping μ", min_value=1e-6, max_value=10.0, value=0.01, format="%.4f", key="svd_damp")
    svd_epsilon = st.number_input("Perturbation ε", min_value=0.001, max_value=0.5, value=0.01, format="%.4f", key="svd_eps")
    svd_itermax = int(st.number_input("Max iterations", min_value=1, max_value=200, value=30, step=5, key="svd_iter"))
    svd_errmin = st.number_input("RMS stop threshold", min_value=0.001, max_value=50.0, value=1.0, format="%.3f", key="svd_err")

# ── Run button ────────────────────────────────────────────────────────────────
run_btn = st.button("▶️ Run both inversions", type="primary")

def _run_inversions(
    ab2, true_rho, true_thick, start_rho, start_thick,
    noise_pct, filter_coeff,
    lm_damping, lm_epsilon, lm_itermax, lm_errmin,
    svd_damping, svd_epsilon, svd_itermax, svd_errmin,
):
    # Generate synthetic observed data
    rng = np.random.default_rng(42)
    slb_true = SLB()
    rhoap_true = slb_true.run(ab2, np.ones_like(ab2), np.array(true_rho), np.array(true_thick), filter_coeff)
    noise = rng.normal(0.0, noise_pct / 100.0, size=len(ab2))
    rhoap_obs = rhoap_true * np.exp(noise)  # multiplicative log-normal noise

    # LM inversion
    inv_lm = SLB_LSInv()
    rho_lm, thick_lm = inv_lm.fit(
        ab2, rhoap_obs,
        np.array(start_rho, dtype=float), np.array(start_thick, dtype=float),
        damping=lm_damping, epsilon=lm_epsilon,
        err_min=lm_errmin, iter_max=lm_itermax,
        method="lm", filter_coeff=filter_coeff,
    )
    fig_lm_err = inv_lm.plot_err()
    err_hist_lm = list(fig_lm_err.axes[0].lines[0].get_ydata())
    plt.close(fig_lm_err)

    # SVD inversion (fresh starting model)
    inv_svd = SLB_LSInv()
    rho_svd, thick_svd = inv_svd.fit(
        ab2, rhoap_obs,
        np.array(start_rho, dtype=float), np.array(start_thick, dtype=float),
        damping=svd_damping, epsilon=svd_epsilon,
        err_min=svd_errmin, iter_max=svd_itermax,
        method="svd", filter_coeff=filter_coeff,
    )
    fig_svd_err = inv_svd.plot_err()
    err_hist_svd = list(fig_svd_err.axes[0].lines[0].get_ydata())
    plt.close(fig_svd_err)

    # Predicted curves for the recovered models
    slb = SLB()
    rhoap_lm = slb.run(ab2, rhoap_obs, rho_lm.copy(), thick_lm.copy(), filter_coeff)
    rhoap_svd = slb.run(ab2, rhoap_obs, rho_svd.copy(), thick_svd.copy(), filter_coeff)

    return (
        rhoap_obs, rhoap_true,
        rho_lm.copy(), thick_lm.copy(), rhoap_lm, err_hist_lm,
        rho_svd.copy(), thick_svd.copy(), rhoap_svd, err_hist_svd,
    )


if run_btn or "ves_inv_result" in st.session_state:
    if run_btn:
        with st.spinner("Running LM and SVD inversions …"):
            result = _run_inversions(
                ab2, true_rho, true_thick, start_rho, start_thick,
                noise_pct, filter_coeff,
                lm_damping, lm_epsilon, lm_itermax, lm_errmin,
                svd_damping, svd_epsilon, svd_itermax, svd_errmin,
            )
        st.session_state["ves_inv_result"] = result
        st.session_state["ves_inv_inputs"] = (
            true_rho, true_thick, start_rho, start_thick, ab2.tolist()
        )
    else:
        result = st.session_state["ves_inv_result"]
        true_rho, true_thick, start_rho, start_thick, ab2_list = st.session_state["ves_inv_inputs"]
        ab2 = np.array(ab2_list)

    (
        rhoap_obs, rhoap_true,
        rho_lm, thick_lm, rhoap_lm, err_hist_lm,
        rho_svd, thick_svd, rhoap_svd, err_hist_svd,
    ) = result

    # ── Metrics ───────────────────────────────────────────────────────────────
    st.subheader(":blue-background[Results]", divider="blue")

    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    col_m1.metric("LM — final RMS", f"{err_hist_lm[-1]:.3f}" if err_hist_lm else "—")
    col_m2.metric("LM — iterations", len(err_hist_lm))
    col_m3.metric("SVD — final RMS", f"{err_hist_svd[-1]:.3f}" if err_hist_svd else "—")
    col_m4.metric("SVD — iterations", len(err_hist_svd))

    # ── Helper: staircase model ───────────────────────────────────────────────
    def _stair(thicknesses, resistivities, extra=None):
        thicknesses = list(thicknesses)
        resistivities = list(resistivities)
        depths = [0.0] + list(np.cumsum(thicknesses))
        bot = depths[-1] + (extra if extra else max(depths[-1] * 0.3, 20.0))
        r_s, d_s = [], []
        for i, rho in enumerate(resistivities):
            d_top = depths[i]
            d_bot = depths[i + 1] if i < len(thicknesses) else bot
            r_s += [rho, rho]
            d_s += [d_top, d_bot]
        return r_s, d_s

    # ── Data fit plot ─────────────────────────────────────────────────────────
    fig1, ax1 = plt.subplots(figsize=(10, 5))
    ax1.loglog(ab2, rhoap_obs, "ko", ms=5, zorder=5, label="Observed (noisy)")
    ax1.loglog(ab2, rhoap_true, "g--", lw=2, label="True model")
    ax1.loglog(ab2, rhoap_lm, "b-", lw=1.8, label="LM predicted")
    ax1.loglog(ab2, rhoap_svd, "r-.", lw=1.8, label="SVD predicted")
    ax1.set_xlabel(r"$AB/2$ (m)")
    ax1.set_ylabel(r"$\rho_a$ ($\Omega\cdot$m)")
    ax1.set_title("Data fit")
    ax1.grid(True, which="both", ls="--", alpha=0.4)
    ax1.legend()
    plt.tight_layout()
    st.pyplot(fig1)
    plt.close(fig1)

    # ── Model comparison ──────────────────────────────────────────────────────
    fig2, ax2 = plt.subplots(figsize=(7, 6))
    r_t, d_t = _stair(true_thick, true_rho)
    ax2.semilogx(r_t, d_t, "g--", lw=2, label="True model")
    r_s0, d_s0 = _stair(start_thick, start_rho)
    ax2.semilogx(r_s0, d_s0, "k:", lw=1.5, label="Starting model")
    r_lm, d_lm = _stair(thick_lm, rho_lm)
    ax2.semilogx(r_lm, d_lm, "b-", lw=2, label="LM recovered")
    r_svd, d_svd = _stair(thick_svd, rho_svd)
    ax2.semilogx(r_svd, d_svd, "r-.", lw=2, label="SVD recovered")
    ax2.invert_yaxis()
    ax2.set_xlabel(r"Resistivity ($\Omega\cdot$m)")
    ax2.set_ylabel("Depth (m)")
    ax2.set_title("Model comparison")
    ax2.grid(True, which="both", ls="--", alpha=0.4)
    ax2.legend()
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close(fig2)

    # ── Convergence ───────────────────────────────────────────────────────────
    st.markdown("**Convergence**")
    col_c1, col_c2 = st.columns(2)

    with col_c1:
        fig3, ax3 = plt.subplots(figsize=(5, 3.5))
        ax3.semilogy(err_hist_lm, "b-o", ms=4)
        ax3.set_xlabel("Iteration")
        ax3.set_ylabel("RMS")
        ax3.set_title("LM convergence")
        ax3.grid(True, which="both", ls="--", alpha=0.4)
        plt.tight_layout()
        st.pyplot(fig3)
        plt.close(fig3)

    with col_c2:
        fig4, ax4 = plt.subplots(figsize=(5, 3.5))
        ax4.semilogy(err_hist_svd, "r-o", ms=4)
        ax4.set_xlabel("Iteration")
        ax4.set_ylabel("RMS")
        ax4.set_title("SVD convergence")
        ax4.grid(True, which="both", ls="--", alpha=0.4)
        plt.tight_layout()
        st.pyplot(fig4)
        plt.close(fig4)

    # ── Recovered model tables ────────────────────────────────────────────────
    with st.expander("Show recovered model parameters"):
        col_t1, col_t2 = st.columns(2)
        with col_t1:
            st.markdown("**LM recovered model**")
            lm_df = pd.DataFrame({
                "Thickness (m)": list(thick_lm) + [None],
                "Resistivity (Ω·m)": list(rho_lm),
            })
            st.dataframe(lm_df.style.format({"Thickness (m)": "{:.2f}", "Resistivity (Ω·m)": "{:.2f}"}),
                         use_container_width=True)
        with col_t2:
            st.markdown("**SVD recovered model**")
            svd_df = pd.DataFrame({
                "Thickness (m)": list(thick_svd) + [None],
                "Resistivity (Ω·m)": list(rho_svd),
            })
            st.dataframe(svd_df.style.format({"Thickness (m)": "{:.2f}", "Resistivity (Ω·m)": "{:.2f}"}),
                         use_container_width=True)

else:
    st.info("Press **▶️ Run both inversions** to start. Adjust the models and settings above first if you like.")
