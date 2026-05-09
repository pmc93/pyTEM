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

from pytem import fwd_circle_central, invert

# ── Page header ───────────────────────────────────────────────────────────────
st.title("🎯 Inversion ▶️")
st.header(":red[Recover a resistivity model from noisy TEM data]")

st.markdown(
    r"""
    This page runs a **synthetic inversion**: we generate noisy dB/dt data
    from a known **true model**, then start from an **initial (starting) model**
    and let the Gauss-Newton algorithm iterate until it converges.

    **The objective function** minimised at each iteration is:

    $$\phi(\mathbf{m}) =
    \underbrace{\left\|\mathbf{W}\!\left(\ln\mathbf{d}_{obs} -
    \ln\mathbf{d}_{pred}(\mathbf{m})\right)\right\|^2}_{\text{data misfit}}
    + \alpha\,\mathbf{m}^T\!\mathbf{R}\,\mathbf{m}$$

    where $\mathbf{W}$ is a data-weighting matrix (based on the noise level),
    $\mathbf{R}$ is a roughness (finite-difference) regularisation matrix,
    and $\alpha$ is the regularisation parameter. The algorithm automatically
    searches for $\alpha$ to target a normalised **RMS misfit of 1**.
    """
)

with st.expander(":green[**Check your understanding — quiz**]"):
    col1, col2 = st.columns(2)
    with col1:
        q1 = st.radio(
            ":red[**What does RMS = 1 mean in a TEM inversion?**]",
            [
                "The model perfectly fits the data",
                "The data fit is consistent with the assumed noise level",
                "The regularisation is too strong",
                "The inversion has not converged",
            ],
            index=None,
        )
        if q1 == "The data fit is consistent with the assumed noise level":
            st.success("Correct! An RMS of 1 means the residuals are on average equal to one standard deviation — a statistically ideal fit.")
        elif q1 is not None:
            st.error("RMS = 1 is the target: the model fits the data to within the assumed noise, no more, no less.")

    with col2:
        q2 = st.radio(
            ":red[**What happens if the regularisation parameter α is too large?**]",
            [
                "The model is too rough (overfits noise)",
                "The model is too smooth (underfits data)",
                "The algorithm diverges",
                "The computation becomes faster",
            ],
            index=None,
        )
        if q2 == "The model is too smooth (underfits data)":
            st.success("Correct! A large α penalises model roughness so strongly that real structure is smoothed out.")
        elif q2 is not None:
            st.error("A large α means the roughness penalty dominates, forcing the model to be very smooth even if the data demand structure.")

# ── True model controls ───────────────────────────────────────────────────────
st.subheader(":red-background[True (synthetic) model]", divider="red")
st.markdown(
    "Define the **true earth model** used to generate synthetic data. "
    "Gaussian noise is added at the level set below."
)

col_t, col_n = st.columns([3, 1])
with col_t:
    st.caption("Last row is the half-space — leave its Thickness cell empty.")
    _default_true = pd.DataFrame({
        "Thickness (m)": [30.0, 80.0, None],
        "Resistivity (Ω·m)": [100.0, 5.0, 300.0],
    })
    _edited_true = st.data_editor(
        _default_true,
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
        key="inv_true_editor",
    )
    _valid_true = _edited_true.dropna(subset=["Resistivity (Ω·m)"])
    true_thicknesses = _valid_true["Thickness (m)"].dropna().tolist()
    true_resistivities = _valid_true["Resistivity (Ω·m)"].tolist()
    if len(true_resistivities) < 1:
        st.warning("Add at least one layer.")
        st.stop()

with col_n:
    st.markdown("**Noise & geometry**")
    noise_pct = st.number_input("Noise level (%)", min_value=1, max_value=30, value=5, step=1, key="inv_noise")
    tx_r_inv = st.number_input("Loop radius (m)", min_value=1.0, max_value=500.0, value=50.0, step=5.0, key="inv_r")
    n_t_inv = int(st.number_input("Time gates", min_value=5, max_value=60, value=20, step=2, key="inv_nt"))

times_inv = np.logspace(-5, -2, n_t_inv)

# ── Starting model controls ───────────────────────────────────────────────────
st.subheader(":red-background[Starting model]", divider="red")
st.markdown(
    "The inversion starts from this model. "
    "A homogeneous half-space is a typical starting point."
)

col_s1, col_s2 = st.columns(2)
with col_s1:
    n_start_layers = int(st.number_input("Number of layers (starting)", min_value=2, max_value=10, value=4, step=1, key="inv_nstart"))
with col_s2:
    start_rho_uniform = st.number_input("Uniform starting resistivity (Ω·m)", min_value=1.0, max_value=5000.0, value=100.0, step=10.0, key="inv_start_rho")

# Build a logarithmically spaced starting model
# n_start_layers layers -> n_start_layers-1 thicknesses
start_depths = np.logspace(1, 2.5, n_start_layers - 1)
start_thicknesses = np.diff(np.concatenate([[0.0], start_depths])).tolist()
start_resistivities = [start_rho_uniform] * n_start_layers
start_log_rho = np.log(np.array(start_resistivities))

# ── Run inversion ─────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def run_inversion(true_thick_t, true_rho_t, start_thick_t, start_log_rho_t,
                  tx_r, times_t, noise_std_frac, seed=42):
    true_thick = list(true_thick_t)
    true_rho = list(true_rho_t)
    start_thick = list(start_thick_t)
    start_log_rho = np.array(start_log_rho_t)
    times = np.array(times_t)

    rng = np.random.default_rng(seed)
    dbdt_true = fwd_circle_central(true_thick, true_rho, tx_radius=tx_r, times=times)
    noise = rng.normal(0.0, noise_std_frac, size=len(times))
    dbdt_obs = dbdt_true * np.exp(noise)  # multiplicative noise in log space
    dbdt_obs = -dbdt_obs  # positive values

    result = invert(
        obs_data=dbdt_obs,
        thicknesses=start_thick,
        log_resistivities=start_log_rho,
        tx_radius=tx_r,
        times=times,
        noise_std=noise_std_frac,
        maxit=15,
        transform="dlf",
        hankel_filter="key_101",
        fourier_filter="key_101",
        analytical_j=True,
    )
    return dbdt_obs, -dbdt_true, result


run_btn = st.button("▶️ Run inversion", type="primary")

if run_btn or "inv_result" in st.session_state:
    if run_btn:
        with st.spinner("Running inversion …"):
            dbdt_obs, dbdt_true, result = run_inversion(
                tuple(true_thicknesses), tuple(true_resistivities),
                tuple(start_thicknesses), tuple(start_log_rho),
                tx_r_inv, tuple(times_inv), noise_pct / 100.0,
            )
        st.session_state["inv_result"] = (dbdt_obs, dbdt_true, result,
                                          true_thicknesses, true_resistivities,
                                          start_thicknesses, start_log_rho)
    else:
        (dbdt_obs, dbdt_true, result,
         true_thicknesses, true_resistivities,
         start_thicknesses, start_log_rho) = st.session_state["inv_result"]

    # ── Results ───────────────────────────────────────────────────────────────
    st.subheader(":red-background[Results]", divider="red")

    rho_recovered = result["resistivities"]
    rms_history = result.get("rms_history", [])
    rms_final = rms_history[-1] if rms_history else None
    n_iter = result.get("n_iter", "?")

    col_m1, col_m2 = st.columns(2)
    col_m1.metric("Final RMS", f"{rms_final:.3f}" if rms_final is not None else "—",
                  help="Target is 1.0")
    col_m2.metric("Iterations", n_iter)

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))

    # Left: data fit
    ax1 = axes[0]
    ax1.loglog(times_inv * 1e3, dbdt_obs, "ko", ms=5, label="Observed (noisy)", zorder=3)
    ax1.loglog(times_inv * 1e3, dbdt_true, "g--", lw=2, label="True model")
    # Compute predicted from final recovered model
    dbdt_pred = fwd_circle_central(
        result["thicknesses"].tolist(), rho_recovered.tolist(),
        tx_radius=tx_r_inv, times=times_inv,
    )
    ax1.loglog(times_inv * 1e3, -dbdt_pred, "r-", lw=1.5, label="Predicted (inverted)")
    ax1.set_xlabel("Time (ms)")
    ax1.set_ylabel(r"$|\partial B_z / \partial t|$  (T/s)")
    ax1.set_title("Data fit")
    ax1.grid(True, which="both", ls="--", alpha=0.4)
    ax1.legend()

    # Right: model comparison
    ax2 = axes[1]

    def _stair(thicknesses, resistivities, extra_depth=200.0):
        depths = [0.0] + list(np.cumsum(thicknesses))
        d_s, r_s = [0.0], []
        for i, r in enumerate(resistivities):
            if i < len(thicknesses):
                d_s.append(depths[i + 1])
                d_s.append(depths[i + 1])
            r_s.append(r)
            r_s.append(r)
        d_s.append(d_s[-1] + extra_depth)
        r_s.append(r_s[-1])
        return r_s, d_s

    r_true, d_true = _stair(true_thicknesses, true_resistivities)
    ax2.semilogx(r_true, d_true, "g--", lw=2, label="True model")

    r_start, d_start = _stair(start_thicknesses, np.exp(start_log_rho))
    ax2.semilogx(r_start, d_start, "k:", lw=1.5, label="Starting model")

    r_rec, d_rec = _stair(result["thicknesses"], rho_recovered)
    ax2.semilogx(r_rec, d_rec, "r-", lw=2, label="Recovered model")

    ax2.invert_yaxis()
    ax2.set_xlabel(r"Resistivity ($\Omega\cdot$m)")
    ax2.set_ylabel("Depth (m)")
    ax2.set_title("Model comparison")
    ax2.grid(True, which="both", ls="--", alpha=0.4)
    ax2.legend()

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # Convergence plot
    if "rms_history" in result and result["rms_history"]:
        fig2, ax3 = plt.subplots(figsize=(7, 3))
        ax3.semilogy(result["rms_history"], "o-", color="firebrick")
        ax3.axhline(1.0, color="k", ls="--", label="Target RMS = 1")
        ax3.set_xlabel("Gauss-Newton iteration")
        ax3.set_ylabel("Normalised RMS misfit")
        ax3.set_title("Convergence")
        ax3.grid(True, which="both", ls="--", alpha=0.4)
        ax3.legend()
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close(fig2)
else:
    st.info("Press **▶️ Run inversion** to start. Adjust the models above first if you like.")
