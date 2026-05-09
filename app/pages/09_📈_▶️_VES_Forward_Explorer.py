import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

# ── Path setup ────────────────────────────────────────────────────────────────
# pages/ -> app/ -> pyTEM/ (project root)
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_VES_ROOT = os.path.join(_ROOT, "ves")
for _p in [_ROOT, _VES_ROOT]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from forward import SLB  # ves/forward/linear_filter.py

# ── Page header ───────────────────────────────────────────────────────────────
st.title("📈 VES Forward Explorer ▶️")
st.header(":blue[Compute the apparent resistivity sounding curve]")

st.markdown(
    r"""
    Define a **layered resistivity model** and a **Schlumberger AB/2 range**.
    The forward model computes the predicted **apparent resistivity** curve
    $\rho_a(AB/2)$ using the DC kernel recursion and Guptasarma linear filter.

    The model is defined from the surface downward: each layer has a
    **thickness** (m) and a **resistivity** ($\Omega\cdot$m).
    The deepest row is the **half-space** (leave Thickness empty).
    """
)

# ── Model data editor ─────────────────────────────────────────────────────────
st.subheader(":blue-background[Layered earth model]", divider="blue")

col_model, col_settings = st.columns([3, 2])

with col_model:
    st.caption("Last row is the half-space — leave its Thickness cell empty.")
    _default_model = pd.DataFrame({
        "Thickness (m)": [10.0, 30.0, None],
        "Resistivity (Ω·m)": [100.0, 20.0, 200.0],
    })
    _edited = st.data_editor(
        _default_model,
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
        key="ves_fwd_model",
    )
    _valid = _edited.dropna(subset=["Resistivity (Ω·m)"])
    thicknesses = _valid["Thickness (m)"].dropna().tolist()
    resistivities = _valid["Resistivity (Ω·m)"].tolist()
    if len(resistivities) < 1:
        st.warning("Add at least one layer.")
        st.stop()
    if len(thicknesses) != len(resistivities) - 1:
        st.warning(
            f"Model mismatch: {len(resistivities)} resistivity rows require "
            f"{len(resistivities) - 1} thickness values. "
            "Make sure the last (half-space) row has no thickness."
        )
        st.stop()

with col_settings:
    st.markdown("**Survey geometry**")
    ab2_min = st.number_input("AB/2 minimum (m)", min_value=0.5, max_value=100.0,
                               value=1.0, step=0.5, key="ab2_min")
    ab2_max = st.number_input("AB/2 maximum (m)", min_value=10.0, max_value=5000.0,
                               value=300.0, step=10.0, key="ab2_max")
    n_ab2 = int(st.number_input("Number of AB/2 points", min_value=5, max_value=100,
                                 value=25, step=5, key="n_ab2"))
    filter_coeff = st.selectbox(
        "Guptasarma filter",
        ["guptasarma_7", "guptasarma_11", "guptasarma_22"],
        index=0,
        key="ves_filter",
        help="7-point is fast; 22-point is most accurate.",
    )

    st.markdown("**Comparison model** (optional)")
    show_compare = st.checkbox("Add a second model to compare", key="ves_compare")

if ab2_min >= ab2_max:
    st.error("AB/2 minimum must be less than maximum.")
    st.stop()

ab2 = np.logspace(np.log10(ab2_min), np.log10(ab2_max), n_ab2)

# ── Optional comparison model ─────────────────────────────────────────────────
if show_compare:
    st.subheader(":blue-background[Comparison model]", divider="blue")
    st.caption("Define a second model to overlay on the same plot.")
    _default_cmp = pd.DataFrame({
        "Thickness (m)": [20.0, None],
        "Resistivity (Ω·m)": [50.0, 500.0],
    })
    _edited_cmp = st.data_editor(
        _default_cmp,
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
        key="ves_cmp_model",
    )
    _valid_cmp = _edited_cmp.dropna(subset=["Resistivity (Ω·m)"])
    cmp_thick = _valid_cmp["Thickness (m)"].dropna().tolist()
    cmp_rho = _valid_cmp["Resistivity (Ω·m)"].tolist()
    _compare_ok = (len(cmp_rho) >= 1) and (len(cmp_thick) == len(cmp_rho) - 1)
else:
    cmp_thick, cmp_rho, _compare_ok = [], [], False

# ── Forward computation ───────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def run_ves_forward(ab2_t, rhotr_t, thick_t, filter_coeff):
    ab2 = np.array(ab2_t)
    rhotr = np.array(rhotr_t)
    thick = np.array(thick_t)
    slb = SLB()
    # rhoap_obs is only used for RMS; pass ones as placeholder
    rhoap = slb.run(ab2, np.ones_like(ab2), rhotr, thick, filter_coeff)
    return rhoap


with st.spinner("Computing …"):
    rhoap = run_ves_forward(
        tuple(ab2.tolist()),
        tuple(resistivities),
        tuple(thicknesses),
        filter_coeff,
    )
    if show_compare and _compare_ok:
        rhoap_cmp = run_ves_forward(
            tuple(ab2.tolist()),
            tuple(cmp_rho),
            tuple(cmp_thick),
            filter_coeff,
        )
    else:
        rhoap_cmp = None

# ── Plots ─────────────────────────────────────────────────────────────────────
st.subheader(":blue-background[Results]", divider="blue")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Left: sounding curve
ax1 = axes[0]
ax1.loglog(ab2, rhoap, "b-o", ms=4, lw=1.5, label="Model 1")
if rhoap_cmp is not None:
    ax1.loglog(ab2, rhoap_cmp, "r--s", ms=4, lw=1.5, label="Model 2")
ax1.set_xlabel(r"$AB/2$ (m)")
ax1.set_ylabel(r"$\rho_a$ ($\Omega\cdot$m)")
ax1.set_title("Apparent resistivity sounding curve")
ax1.grid(True, which="both", ls="--", alpha=0.4)
ax1.legend()

# Right: earth model staircase
ax2 = axes[1]


def _stair_ves(thicknesses, resistivities, extra_depth_frac=0.2):
    """Build staircase (rho, depth) arrays for plotting."""
    depths = [0.0] + list(np.cumsum(thicknesses))
    max_depth = depths[-1] if len(thicknesses) > 0 else 10.0
    extra = max(max_depth * extra_depth_frac, 10.0)
    r_s, d_s = [], []
    for i, rho in enumerate(resistivities):
        d_top = depths[i]
        d_bot = depths[i + 1] if i < len(thicknesses) else depths[-1] + extra
        r_s += [rho, rho]
        d_s += [d_top, d_bot]
    return r_s, d_s


r_s, d_s = _stair_ves(thicknesses, resistivities)
ax2.semilogx(r_s, d_s, "b-", lw=2, label="Model 1")
if rhoap_cmp is not None and _compare_ok:
    r_sc, d_sc = _stair_ves(cmp_thick, cmp_rho)
    ax2.semilogx(r_sc, d_sc, "r--", lw=2, label="Model 2")

ax2.invert_yaxis()
ax2.set_xlabel(r"Resistivity ($\Omega\cdot$m)")
ax2.set_ylabel("Depth (m)")
ax2.set_title("Earth model")
ax2.grid(True, which="both", ls="--", alpha=0.4)
ax2.legend()

plt.tight_layout()
st.pyplot(fig)
plt.close(fig)

# ── Numeric table ─────────────────────────────────────────────────────────────
with st.expander("Show computed values"):
    out_df = pd.DataFrame({"AB/2 (m)": ab2, "ρ_a Model 1 (Ω·m)": rhoap})
    if rhoap_cmp is not None:
        out_df["ρ_a Model 2 (Ω·m)"] = rhoap_cmp
    st.dataframe(out_df.style.format("{:.3g}"), use_container_width=True)
