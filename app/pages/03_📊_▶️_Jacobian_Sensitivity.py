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

from pytem import fwd_circle_central, getJ_ana

# ── Page header ───────────────────────────────────────────────────────────────
st.title("📊 Jacobian & Sensitivity ▶️")
st.header(":violet[Understanding which time gates see which layers]")

st.markdown(
    r"""
    The **Jacobian** (or sensitivity matrix) $\mathbf{J}$ tells us how much each
    measured time gate $i$ would change if we perturbed the resistivity of layer $j$
    by a small amount. Working in log–log space keeps the entries dimensionless:

    $$J_{ij} = \frac{\partial \ln(-\dot{B}_i)}{\partial \ln \rho_j}$$

    A large $|J_{ij}|$ means gate $i$ is **very sensitive** to layer $j$; a value
    near zero means gate $i$ carries almost no information about that layer.

    The Jacobian is the cornerstone of the Gauss-Newton inversion — it is recomputed
    at every iteration to guide the model update toward a better fit.
    """
)

with st.expander(":green[**Check your understanding — quiz**]"):
    col1, col2 = st.columns(2)
    with col1:
        q1 = st.radio(
            ":violet[**Which time gates are most sensitive to deep layers?**]",
            ["Early times", "Late times", "All times equally", "It depends on loop size only"],
            index=None,
        )
        if q1 == "Late times":
            st.success("Correct! Eddy currents diffuse to depth over time, so late gates sample deeper structure.")
        elif q1 is not None:
            st.error("Think about where the eddy currents are at late times — they have diffused deep into the earth.")

    with col2:
        q2 = st.radio(
            ":violet[**What does a column of J near zero imply?**]",
            [
                "That layer has high resistivity",
                "No time gate is sensitive to that layer — it cannot be resolved",
                "The inversion has converged",
                "The layer is very thick",
            ],
            index=None,
        )
        if q2 == "No time gate is sensitive to that layer — it cannot be resolved":
            st.success("Correct! Layers with near-zero columns in J are poorly constrained by the data.")
        elif q2 is not None:
            st.error("A near-zero column means the data contain almost no information about that layer.")

# ── Model controls ────────────────────────────────────────────────────────────
st.subheader(":violet-background[Model & geometry]", divider="violet")

col_m, col_g = st.columns(2)

with col_m:
    st.markdown("**Layer model** — last row is the half-space, leave its Thickness empty")
    _default_j = pd.DataFrame({
        "Thickness (m)": [20.0, 60.0, None],
        "Resistivity (Ω·m)": [100.0, 10.0, 300.0],
    })
    _edited_j = st.data_editor(
        _default_j,
        column_config={
            "Thickness (m)": st.column_config.NumberColumn(
                min_value=0.1, max_value=10000.0, format="%.1f",
                help="Layer thickness in metres. Leave empty for the half-space.",
            ),
            "Resistivity (Ω·m)": st.column_config.NumberColumn(
                min_value=0.01, max_value=1e6, format="%.1f",
            ),
        },
        num_rows="dynamic",
        use_container_width=True,
        key="jac_model_editor",
    )
    _valid_j = _edited_j.dropna(subset=["Resistivity (Ω·m)"])
    thicknesses_j = _valid_j["Thickness (m)"].dropna().tolist()
    resistivities_j = _valid_j["Resistivity (Ω·m)"].tolist()
    n_layers = len(resistivities_j)
    if n_layers < 1:
        st.warning("Add at least one layer.")
        st.stop()

with col_g:
    st.markdown("**Loop geometry & time axis**")
    tx_r_j = st.number_input("Loop radius (m)", min_value=1.0, max_value=500.0, value=50.0, step=5.0, key="jac_r")
    t_min_j = st.number_input("Early time (10^ˣ s)", min_value=-7.0, max_value=-3.0, value=-5.0, step=0.25, key="jtmin")
    t_max_j = st.number_input("Late time (10^ˣ s)", min_value=-4.0, max_value=-1.0, value=-2.0, step=0.25, key="jtmax")
    n_t_j = int(st.number_input("Number of time gates", min_value=5, max_value=60, value=20, step=1, key="jnt"))

times_j = np.logspace(t_min_j, t_max_j, n_t_j)

# ── Compute Jacobian ──────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def compute_jacobian(thicknesses_t, log_res_t, tx_r, times_t):
    thicknesses = list(thicknesses_t)
    log_res = np.array(log_res_t)
    resistivities = np.exp(log_res).tolist()
    times = np.array(times_t)
    dbdt = fwd_circle_central(thicknesses, resistivities, tx_radius=tx_r, times=times)
    J = getJ_ana(
        thicknesses, log_res, tx_r, times,
        geometry="circle_central",
        hankel_filter="key_101", fourier_filter="key_101",
    )
    return -dbdt, J  # flip sign for plotting


log_res_j = np.log(resistivities_j)

with st.spinner("Computing Jacobian …"):
    dbdt_j, J = compute_jacobian(
        tuple(thicknesses_j), tuple(log_res_j), tx_r_j, tuple(times_j)
    )

# ── Plot ──────────────────────────────────────────────────────────────────────
st.subheader(":violet-background[Results]", divider="violet")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Left: dB/dt
ax1 = axes[0]
ax1.loglog(times_j * 1e3, dbdt_j, "o-", color="mediumpurple", ms=5, lw=1.5)
ax1.set_xlabel("Time (ms)")
ax1.set_ylabel(r"$|\partial B_z / \partial t|$  (T/s)")
ax1.set_title("Predicted decay curve")
ax1.grid(True, which="both", ls="--", alpha=0.4)

# Right: Jacobian heatmap
ax2 = axes[1]
layer_labels = [f"Layer {i+1}\n({resistivities_j[i]:.0f} Ω·m)" for i in range(n_layers)]
time_labels = [f"{t*1e3:.2f}" for t in times_j]

im = ax2.imshow(
    J, aspect="auto", cmap="RdBu_r",
    vmin=-np.abs(J).max(), vmax=np.abs(J).max(),
)
ax2.set_xticks(range(n_layers))
ax2.set_xticklabels(layer_labels, fontsize=8)
ax2.set_yticks(range(0, n_t_j, max(1, n_t_j // 8)))
ax2.set_yticklabels(
    [f"{times_j[k]*1e3:.2f} ms" for k in range(0, n_t_j, max(1, n_t_j // 8))],
    fontsize=8,
)
ax2.set_xlabel("Layer (model parameter)")
ax2.set_ylabel("Time gate")
ax2.set_title(r"Jacobian  $J_{ij} = \partial\ln|\dot{B}_i|/\partial\ln\rho_j$")
plt.colorbar(im, ax=ax2, label="Sensitivity")

plt.tight_layout()
st.pyplot(fig)
plt.close(fig)

# ── Column norms ──────────────────────────────────────────────────────────────
st.subheader(":violet-background[Column norms — total sensitivity per layer]", divider="violet")
col_norms = np.linalg.norm(J, axis=0)

fig3, ax3 = plt.subplots(figsize=(6, 3))
ax3.bar(layer_labels, col_norms, color="mediumpurple", alpha=0.8)
ax3.set_ylabel(r"$\|\mathbf{J}_{:j}\|_2$")
ax3.set_title("Total data sensitivity per layer")
ax3.grid(axis="y", ls="--", alpha=0.4)
plt.tight_layout()
st.pyplot(fig3)
plt.close(fig3)

st.markdown(
    """
    **Reading the plots:**
    - **Red** cells in the heatmap indicate that increasing that layer's resistivity
      increases the dB/dt at that time gate.
    - **Blue** cells indicate the opposite: increasing resistivity decreases dB/dt.
    - Layers with small column norms (short bars) are poorly resolved — the data
      provide little constraint on their resistivity.
    """
)
