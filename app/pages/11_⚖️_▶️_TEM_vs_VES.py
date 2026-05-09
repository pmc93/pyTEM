import os
import sys

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

from pytem import fwd_circle_central, invert as tem_invert
from forward import SLB
from fitting import SLB_LSInv

# ── Page header ───────────────────────────────────────────────────────────────
st.title("⚖️ TEM vs VES — Method Comparison ▶️")
st.header(":violet[Invert the same earth model with two different geophysical methods]")

st.markdown(
    r"""
    This page inverts a **shared layered earth model** using both:
    - **TEM** — time-domain EM, dB/dt sounding
    - **VES** — DC resistivity, apparent resistivity sounding

    Both methods are sensitive to the subsurface **resistivity** structure, but
    they differ fundamentally in physics, depth sensitivity, and ability to
    resolve resistive vs conductive targets. Running both on the same synthetic
    model reveals these differences directly.
    """
)

# ── Theory comparison ─────────────────────────────────────────────────────────
st.subheader(":violet-background[Physics comparison]", divider="violet")

col_t1, col_t2 = st.columns(2)
with col_t1:
    st.markdown(
        r"""
        **TEM**

        The transmitter loop drives a step-off current that induces eddy
        currents diffusing downward through the earth. The secondary field
        recorded at the receiver decays as:

        $$\dot{B}_z(t) \propto \int_0^\infty r_{TE}(\lambda,\omega)\,\lambda\,J_1(\lambda a)\,e^{i\omega t}\,d\omega\,d\lambda$$

        - **Depth encoded in time**: early gates = shallow; late gates = deep
        - **Sensitive to conductors**: eddy currents concentrate in low-$\rho$ zones
        - **No ground contact** needed — inductive coupling
        - **Noise floor** limits late-time (deep) sensitivity
        """
    )
with col_t2:
    st.markdown(
        r"""
        **VES**

        Current injected at electrodes A, B flows through the earth; the
        voltage at M, N gives apparent resistivity via the geometric factor:

        $$\rho_a(r) = r^2 \int_0^\infty T(\lambda)\,J_1(\lambda r)\,\lambda\,d\lambda$$

        - **Depth encoded in AB/2**: larger spacing = deeper current path
        - **Sensitive to both** resistors and conductors equally
        - **Galvanic coupling** — electrodes must contact the ground
        - **Equivalence problem**: thin resistive layers can trade off against deeper structure
        """
    )

st.markdown(
    r"""
    | Property | TEM | VES |
    |---|---|---|
    | Source | Inductive (loop current) | Galvanic (electrode injection) |
    | Depth proxy | Time $t$ | Electrode spacing $AB/2$ |
    | Best for | Conductive targets | Resistive targets |
    | Ground contact | Not required | Required |
    | Typical depth | 10–300 m | 5–500 m |
    | Forward kernel | Wait TE recursion + Fourier/Hankel | DC kernel recursion + Hankel |
    | Inversion style | Gauss-Newton (pyTEM) | LM / SVD (ves) |
    """
)

with st.expander(":green[**Check your understanding — quiz**]"):
    col1, col2 = st.columns(2)
    with col1:
        q1 = st.radio(
            ":violet[**Which method is better at detecting a thin conductive clay layer?**]",
            ["TEM, because eddy currents concentrate in conductors",
             "VES, because DC current is unaffected by conductivity",
             "Both are equally sensitive",
             "Neither — you need seismic for this"],
            index=None,
        )
        if q1 == "TEM, because eddy currents concentrate in conductors":
            st.success("Correct! TEM eddy currents preferentially flow through conductive zones, making TEM highly sensitive to them.")
        elif q1 is not None:
            st.error("TEM is more sensitive to conductors because the induced eddy currents concentrate there.")

    with col2:
        q2 = st.radio(
            ":violet[**If the near-surface is frozen (very high resistivity), which method is safer to use?**]",
            ["VES — the high resistivity helps electrode contact",
             "TEM — inductive coupling bypasses the resistive surface",
             "Both work equally well on frozen ground",
             "Neither — both require conductive near-surface"],
            index=None,
        )
        if q2 == "TEM — inductive coupling bypasses the resistive surface":
            st.success("Correct! TEM loops don't need ground contact, so a frozen or dry resistive surface doesn't prevent the measurement.")
        elif q2 is not None:
            st.error("TEM is inductive — no ground contact needed. VES requires electrode stakes, which is difficult on frozen or dry ground.")

# ── Shared true model ─────────────────────────────────────────────────────────
st.subheader(":violet-background[Shared true earth model]", divider="violet")
st.markdown(
    "Define one layered earth model that is used as the ground truth for "
    "**both** the TEM and VES synthetic inversions below."
)

col_model, col_note = st.columns([3, 2])
with col_model:
    st.caption("Last row is the half-space — leave its Thickness cell empty.")
    _default_true = pd.DataFrame({
        "Thickness (m)": [15.0, 40.0, None],
        "Resistivity (Ω·m)": [120.0, 8.0, 400.0],
    })
    _edited = st.data_editor(
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
        key="cmp_true",
    )
    _valid = _edited.dropna(subset=["Resistivity (Ω·m)"])
    true_thick = _valid["Thickness (m)"].dropna().tolist()
    true_rho = _valid["Resistivity (Ω·m)"].tolist()
    if len(true_rho) < 1:
        st.warning("Add at least one layer.")
        st.stop()
    if len(true_thick) != len(true_rho) - 1:
        st.warning("Number of thicknesses must equal number of resistivities minus 1.")
        st.stop()

with col_note:
    st.markdown(
        r"""
        **Default model:** a conductive middle layer ($8\;\Omega\cdot$m) sandwiched
        between resistive layers. This is a classic groundwater / saline
        aquifer scenario — TEM is expected to resolve the conductor better than VES.
        """
    )

# ── Survey & inversion settings ───────────────────────────────────────────────
st.subheader(":violet-background[Survey & inversion settings]", divider="violet")

col_s1, col_s2 = st.columns(2)
with col_s1:
    st.markdown("**TEM settings**")
    tx_r = st.number_input("Loop radius (m)", min_value=5.0, max_value=500.0, value=40.0, step=5.0, key="cmp_txr")
    n_t = int(st.number_input("Time gates", min_value=5, max_value=60, value=20, step=5, key="cmp_nt"))
    tem_noise_pct = st.number_input("TEM noise (%)", min_value=1, max_value=30, value=5, step=1, key="cmp_temnoise")
    n_tem_layers = int(st.number_input("TEM starting layers", min_value=2, max_value=10, value=4, step=1, key="cmp_tml"))
    tem_start_rho = st.number_input("TEM uniform starting ρ (Ω·m)", min_value=1.0, max_value=5000.0, value=100.0, step=10.0, key="cmp_tmrho")

with col_s2:
    st.markdown("**VES settings**")
    ab2_min = st.number_input("AB/2 min (m)", min_value=0.5, max_value=100.0, value=1.0, step=0.5, key="cmp_ab2min")
    ab2_max = st.number_input("AB/2 max (m)", min_value=10.0, max_value=3000.0, value=200.0, step=10.0, key="cmp_ab2max")
    n_ab2 = int(st.number_input("VES AB/2 points", min_value=5, max_value=60, value=20, step=5, key="cmp_nab2"))
    ves_noise_pct = st.number_input("VES noise (%)", min_value=0, max_value=30, value=5, step=1, key="cmp_vesnoise")
    ves_filter = st.selectbox("VES filter", ["guptasarma_7", "guptasarma_11", "guptasarma_22"], index=0, key="cmp_vesfilter")

# ── Run button ────────────────────────────────────────────────────────────────
run_btn = st.button("▶️ Run both inversions", type="primary")


@st.cache_data(show_spinner=False)
def _run_tem(true_thick_t, true_rho_t, tx_r, n_t, noise_pct, n_layers, start_rho_uniform):
    true_thick = list(true_thick_t)
    true_rho = list(true_rho_t)
    times = np.logspace(-5, -2, n_t)

    rng = np.random.default_rng(42)
    dbdt_true = fwd_circle_central(true_thick, true_rho, tx_radius=tx_r, times=times)
    noise = rng.normal(0.0, noise_pct / 100.0, size=n_t)
    dbdt_obs = dbdt_true * np.exp(noise)

    # Build starting model
    start_depths = np.logspace(1, 2.5, n_layers - 1)
    start_thicknesses = np.diff(np.concatenate([[0.0], start_depths])).tolist()
    start_log_rho = np.log(np.full(n_layers, start_rho_uniform))

    result = tem_invert(
        obs_data=-dbdt_obs,
        thicknesses=start_thicknesses,
        log_resistivities=start_log_rho,
        tx_radius=tx_r,
        times=times,
        noise_std=noise_pct / 100.0,
        maxit=15,
        transform="dlf",
        hankel_filter="key_101",
        fourier_filter="key_101",
        analytical_j=True,
    )

    dbdt_pred = fwd_circle_central(
        result["thicknesses"].tolist(), result["resistivities"].tolist(),
        tx_radius=tx_r, times=times,
    )
    return (
        times,
        -dbdt_obs,
        -dbdt_true,
        -dbdt_pred,
        result["thicknesses"].tolist(),
        result["resistivities"].tolist(),
        result.get("rms_history", []),
        start_thicknesses,
        [start_rho_uniform] * n_layers,
    )


@st.cache_data(show_spinner=False)
def _run_ves(true_thick_t, true_rho_t, ab2_t, noise_pct, ves_filter):
    true_thick = np.array(true_thick_t)
    true_rho = np.array(true_rho_t)
    ab2 = np.array(ab2_t)
    n_layers = len(true_rho)

    rng = np.random.default_rng(42)
    slb = SLB()
    rhoap_true = slb.run(ab2, np.ones_like(ab2), true_rho, true_thick, ves_filter)
    noise = rng.normal(0.0, noise_pct / 100.0, size=len(ab2))
    rhoap_obs = rhoap_true * np.exp(noise)

    # Starting model: same number of layers as true, uniform resistivity
    start_rho = np.full(n_layers, float(np.median(true_rho)))
    start_thick = true_thick.copy()  # use true thicknesses as starting point

    inv = SLB_LSInv()
    rho_rec, thick_rec = inv.fit(
        ab2, rhoap_obs, start_rho.copy(), start_thick.copy(),
        damping=0.01, epsilon=0.01, err_min=0.5, iter_max=30,
        method="lm", filter_coeff=ves_filter,
    )

    fig_err = inv.plot_err()
    err_hist = list(fig_err.axes[0].lines[0].get_ydata())
    plt.close(fig_err)

    rhoap_pred = slb.run(ab2, rhoap_obs, rho_rec.copy(), thick_rec.copy(), ves_filter)

    return (
        rhoap_obs, rhoap_true, rhoap_pred,
        thick_rec.tolist(), rho_rec.tolist(),
        err_hist,
        start_thick.tolist(), start_rho.tolist(),
    )


if run_btn or "cmp_result" in st.session_state:
    if run_btn:
        ab2 = np.logspace(np.log10(ab2_min), np.log10(ab2_max), n_ab2)
        with st.spinner("Running TEM inversion …"):
            tem_res = _run_tem(
                tuple(true_thick), tuple(true_rho),
                tx_r, n_t, tem_noise_pct, n_tem_layers, tem_start_rho,
            )
        with st.spinner("Running VES inversion …"):
            ves_res = _run_ves(
                tuple(true_thick), tuple(true_rho),
                tuple(ab2.tolist()), ves_noise_pct, ves_filter,
            )
        st.session_state["cmp_result"] = (tem_res, ves_res, ab2.tolist(), true_thick, true_rho)
    else:
        tem_res, ves_res, ab2_list, true_thick, true_rho = st.session_state["cmp_result"]
        ab2 = np.array(ab2_list)

    (
        times, dbdt_obs, dbdt_true, dbdt_pred,
        tem_thick_rec, tem_rho_rec, tem_rms_hist,
        tem_start_thick, tem_start_rho,
    ) = tem_res
    (
        rhoap_obs, rhoap_true, rhoap_pred,
        ves_thick_rec, ves_rho_rec, ves_err_hist,
        ves_start_thick, ves_start_rho,
    ) = ves_res

    # ── Summary metrics ───────────────────────────────────────────────────────
    st.subheader(":violet-background[Results]", divider="violet")
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    col_m1.metric("TEM final RMS", f"{tem_rms_hist[-1]:.3f}" if tem_rms_hist else "—", help="Target ~1.0")
    col_m2.metric("TEM iterations", len(tem_rms_hist))
    col_m3.metric("VES final RMS", f"{ves_err_hist[-1]:.3f}" if ves_err_hist else "—")
    col_m4.metric("VES iterations", len(ves_err_hist))

    # ── Helper: staircase ─────────────────────────────────────────────────────
    def _stair(thicknesses, resistivities):
        thicknesses = list(thicknesses)
        resistivities = list(resistivities)
        depths = [0.0] + list(np.cumsum(thicknesses))
        extra = max(depths[-1] * 0.3, 20.0)
        bot = depths[-1] + extra
        r_s, d_s = [], []
        for i, rho in enumerate(resistivities):
            d_top = depths[i]
            d_bot = depths[i + 1] if i < len(thicknesses) else bot
            r_s += [rho, rho]
            d_s += [d_top, d_bot]
        return r_s, d_s

    # ── Data fit plots ────────────────────────────────────────────────────────
    st.markdown("**Data fit**")
    col_d1, col_d2 = st.columns(2)

    with col_d1:
        fig1, ax1 = plt.subplots(figsize=(5.5, 4))
        ax1.loglog(times * 1e3, dbdt_obs, "ko", ms=4, label="Observed (noisy)", zorder=3)
        ax1.loglog(times * 1e3, dbdt_true, "g--", lw=1.8, label="True")
        ax1.loglog(times * 1e3, dbdt_pred, "b-", lw=1.8, label="TEM recovered")
        ax1.set_xlabel("Time (ms)")
        ax1.set_ylabel(r"$|\partial B_z/\partial t|$ (T/s)")
        ax1.set_title("TEM data fit")
        ax1.grid(True, which="both", ls="--", alpha=0.4)
        ax1.legend(fontsize=8)
        plt.tight_layout()
        st.pyplot(fig1)
        plt.close(fig1)

    with col_d2:
        fig2, ax2 = plt.subplots(figsize=(5.5, 4))
        ax2.loglog(ab2, rhoap_obs, "ko", ms=4, label="Observed (noisy)", zorder=3)
        ax2.loglog(ab2, rhoap_true, "g--", lw=1.8, label="True")
        ax2.loglog(ab2, rhoap_pred, "r-", lw=1.8, label="VES recovered")
        ax2.set_xlabel(r"$AB/2$ (m)")
        ax2.set_ylabel(r"$\rho_a$ ($\Omega\cdot$m)")
        ax2.set_title("VES data fit")
        ax2.grid(True, which="both", ls="--", alpha=0.4)
        ax2.legend(fontsize=8)
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close(fig2)

    # ── Model comparison ──────────────────────────────────────────────────────
    st.markdown("**Recovered model comparison — both methods on one plot**")
    fig3, ax3 = plt.subplots(figsize=(8, 6))
    r_t, d_t = _stair(true_thick, true_rho)
    ax3.semilogx(r_t, d_t, "g--", lw=2.5, label="True model")
    r_ts, d_ts = _stair(tem_start_thick, tem_start_rho)
    ax3.semilogx(r_ts, d_ts, "k:", lw=1.5, label="TEM starting model")
    r_tem, d_tem = _stair(tem_thick_rec, tem_rho_rec)
    ax3.semilogx(r_tem, d_tem, "b-", lw=2, label="TEM recovered")
    r_ves, d_ves = _stair(ves_thick_rec, ves_rho_rec)
    ax3.semilogx(r_ves, d_ves, "r-.", lw=2, label="VES recovered")
    ax3.invert_yaxis()
    ax3.set_xlabel(r"Resistivity ($\Omega\cdot$m)")
    ax3.set_ylabel("Depth (m)")
    ax3.set_title("TEM vs VES — recovered models")
    ax3.grid(True, which="both", ls="--", alpha=0.4)
    ax3.legend()
    plt.tight_layout()
    st.pyplot(fig3)
    plt.close(fig3)

    # ── Convergence comparison ────────────────────────────────────────────────
    st.markdown("**Convergence**")
    col_c1, col_c2 = st.columns(2)

    with col_c1:
        fig4, ax4 = plt.subplots(figsize=(5, 3.5))
        ax4.semilogy(tem_rms_hist, "b-o", ms=4)
        if tem_rms_hist:
            ax4.axhline(1.0, color="k", ls="--", label="Target RMS = 1")
        ax4.set_xlabel("Iteration")
        ax4.set_ylabel("Normalised RMS")
        ax4.set_title("TEM convergence (Gauss-Newton)")
        ax4.grid(True, which="both", ls="--", alpha=0.4)
        ax4.legend(fontsize=8)
        plt.tight_layout()
        st.pyplot(fig4)
        plt.close(fig4)

    with col_c2:
        fig5, ax5 = plt.subplots(figsize=(5, 3.5))
        ax5.semilogy(ves_err_hist, "r-o", ms=4)
        ax5.set_xlabel("Iteration")
        ax5.set_ylabel("RMS")
        ax5.set_title("VES convergence (LM)")
        ax5.grid(True, which="both", ls="--", alpha=0.4)
        plt.tight_layout()
        st.pyplot(fig5)
        plt.close(fig5)

    # ── Interpretation notes ──────────────────────────────────────────────────
    st.subheader(":violet-background[Interpretation]", divider="violet")
    st.markdown(
        r"""
        **What to look for in the plots above:**

        - The **conductive middle layer** (low $\rho$) is the main target.
          TEM typically resolves it more sharply because eddy currents
          concentrate in conductors, producing a strong late-time signal.
        - VES resolves the conductor too, but the **equivalence principle**
          means the product $\rho \cdot h$ is better constrained than
          $\rho$ and $h$ individually — thin conductive layers can trade
          off in thickness vs resistivity.
        - The **resistive half-space** below is harder for TEM to image
          at depth (the signal decays quickly in resistive material), while
          VES can extend $AB/2$ as far as needed.
        - **Starting model dependence**: both methods are non-linear and
          may find different local minima depending on the starting model.
          Try changing the starting model settings to explore this.
        - **Noise level**: increase the noise to see where each method
          starts to break down — TEM late-time gates are most affected;
          VES large-$AB/2$ points are most affected.
        """
    )

else:
    st.info(
        "Press **▶️ Run both inversions** to compare TEM and VES on the shared true model above."
    )
