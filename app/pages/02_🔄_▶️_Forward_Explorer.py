import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

# ── Path setup ────────────────────────────────────────────────────────────────
# pages/ -> app/ -> pyTEM/ (project root that contains the pytem package)
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from pytem import fwd_circle_central, fwd_square_central

# ── Page header ───────────────────────────────────────────────────────────────
st.title("🔄 Forward Explorer ▶️")
st.header(":blue[Build a layered earth model and compute dB/dt]")

st.markdown(
    """
    Use the controls below to define a **layered resistivity model** and a
    **loop geometry**. The predicted $\\partial B_z / \\partial t$ response
    updates automatically whenever you change a value.

    The model is defined from the surface downward:
    each layer has a **thickness** (m) and a **resistivity** ($\\Omega\\cdot$m).
    The deepest layer is a **half-space** (infinite thickness).
    """
)

# ── Cached forward call ───────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def run_forward(thicknesses_tuple, resistivities_tuple, loop_type, loop_size, times_tuple):
    thicknesses = list(thicknesses_tuple)
    resistivities = list(resistivities_tuple)
    times = np.array(times_tuple)
    if loop_type == "Circle — central":
        dbdt = fwd_circle_central(thicknesses, resistivities, tx_radius=loop_size, times=times)
    else:
        dbdt = fwd_square_central(thicknesses, resistivities, side_length=loop_size, times=times)
    return -dbdt  # positive values for plotting (step-off gives negative dBdt)


# ── Sidebar: time axis ────────────────────────────────────────────────────────
st.sidebar.header("⏱ Time axis")
t_min_exp = st.sidebar.slider("Early time  (10^x s)", -6.0, -4.0, -5.0, 0.25)
t_max_exp = st.sidebar.slider("Late time   (10^x s)", -3.0, -1.0, -2.0, 0.25)
n_times = st.sidebar.slider("Number of time gates", 10, 60, 31, 1)
times = np.logspace(t_min_exp, t_max_exp, n_times)

# ── Sidebar: loop geometry ────────────────────────────────────────────────────
st.sidebar.header("🔲 Loop geometry")
loop_type = st.sidebar.selectbox(
    "Loop type",
    ["Circle — central", "Square — central"],
)
if loop_type == "Circle — central":
    loop_size = st.sidebar.number_input("Loop radius (m)", min_value=1.0, max_value=500.0, value=50.0, step=5.0)
    loop_label = f"Circle, r = {loop_size:.0f} m"
else:
    loop_size = st.sidebar.number_input("Loop side length (m)", min_value=1.0, max_value=500.0, value=100.0, step=5.0)
    loop_label = f"Square, L = {loop_size:.0f} m"

# ── Main area: model builder ──────────────────────────────────────────────────
st.subheader(":blue-background[Layer model]", divider="blue")

st.caption(
    "Edit the table directly — click any cell to change its value. "
    "Add rows with the ➕ button. "
    "The **last row** is always the half-space: leave its Thickness cell empty."
)

_default_m1 = pd.DataFrame({
    "Thickness (m)": [20.0, 50.0, None],
    "Resistivity (Ω·m)": [100.0, 10.0, 300.0],
})
_edited_m1 = st.data_editor(
    _default_m1,
    column_config={
        "Thickness (m)": st.column_config.NumberColumn(
            min_value=0.1, max_value=10000.0, format="%.1f",
            help="Layer thickness in metres. Leave empty for the half-space (last row).",
        ),
        "Resistivity (Ω·m)": st.column_config.NumberColumn(
            min_value=0.01, max_value=1e6, format="%.1f",
            help="DC resistivity in Ohm·m",
        ),
    },
    num_rows="dynamic",
    use_container_width=True,
    key="model1_editor",
)
_valid_m1 = _edited_m1.dropna(subset=["Resistivity (Ω·m)"])
thicknesses = _valid_m1["Thickness (m)"].dropna().tolist()
resistivities = _valid_m1["Resistivity (Ω·m)"].tolist()
depths = [0.0] + list(np.cumsum(thicknesses))
if len(resistivities) < 1:
    st.warning("Add at least one layer.")
    st.stop()

# ── Compute and plot ──────────────────────────────────────────────────────────
st.subheader(":blue-background[dB/dt response]", divider="blue")

with st.spinner("Computing forward response …"):
    dbdt_pos = run_forward(
        tuple(thicknesses), tuple(resistivities),
        loop_type, loop_size, tuple(times)
    )

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Left: dB/dt vs time
ax = axes[0]
ax.loglog(times * 1e3, dbdt_pos, "o-", color="steelblue", ms=4, lw=1.5,
          label=loop_label)
ax.set_xlabel("Time (ms)")
ax.set_ylabel(r"$|\partial B_z / \partial t|$  (T/s)")
ax.set_title("TEM decay curve")
ax.grid(True, which="both", ls="--", alpha=0.4)
ax.legend()

# Right: resistivity model as a step-plot
ax2 = axes[1]
depths_plot = np.array(depths + [depths[-1] + sum(thicknesses) * 0.5])
depths_plot[-1] = depths_plot[-2] + 200.0
rho_plot = resistivities
# Build staircase
depth_stair = [0.0]
rho_stair = []
for i, (d, r) in enumerate(zip(depths[1:], rho_plot)):
    depth_stair.append(d)
    depth_stair.append(d)
    rho_stair.append(rho_plot[i])
    rho_stair.append(rho_plot[i])
depth_stair.append(depth_stair[-1] + 200.0)
rho_stair.append(rho_plot[-1])
rho_stair.append(rho_plot[-1])

ax2.semilogx(rho_stair, depth_stair, "k-", lw=2)
ax2.fill_betweenx(depth_stair, rho_stair, alpha=0.15, color="steelblue")
ax2.invert_yaxis()
ax2.set_xlabel(r"Resistivity  ($\Omega\cdot$m)")
ax2.set_ylabel("Depth (m)")
ax2.set_title("Resistivity model")
ax2.grid(True, which="both", ls="--", alpha=0.4)

plt.tight_layout()
st.pyplot(fig)
plt.close(fig)

# ── Comparison: add a second model ────────────────────────────────────────────
st.subheader(":blue-background[Compare with a second model]", divider="blue")

show_compare = st.toggle("Enable second model for comparison", value=False)
if show_compare:
    st.caption("Edit Model 2 in the table below. Last row = half-space (leave Thickness empty).")
    _default_m2 = pd.DataFrame({
        "Thickness (m)": [20.0, 50.0, None],
        "Resistivity (Ω·m)": [50.0, 500.0, 20.0],
    })
    _edited_m2 = st.data_editor(
        _default_m2,
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
        key="model2_editor",
    )
    _valid_m2 = _edited_m2.dropna(subset=["Resistivity (Ω·m)"])
    thicknesses2 = _valid_m2["Thickness (m)"].dropna().tolist()
    resistivities2 = _valid_m2["Resistivity (Ω·m)"].tolist()

    with st.spinner("Computing second model …"):
        dbdt2 = run_forward(
            tuple(thicknesses2), tuple(resistivities2),
            loop_type, loop_size, tuple(times)
        )

    fig2, ax3 = plt.subplots(figsize=(8, 5))
    ax3.loglog(times * 1e3, dbdt_pos, "o-", color="steelblue", ms=4, lw=1.5, label="Model 1")
    ax3.loglog(times * 1e3, dbdt2, "s--", color="firebrick", ms=4, lw=1.5, label="Model 2")
    ax3.set_xlabel("Time (ms)")
    ax3.set_ylabel(r"$|\partial B_z / \partial t|$  (T/s)")
    ax3.set_title("Model comparison")
    ax3.grid(True, which="both", ls="--", alpha=0.4)
    ax3.legend()
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close(fig2)

# ── Interpretation hints ──────────────────────────────────────────────────────
with st.expander(":green[**Interpretation tips**]"):
    st.markdown(
        """
        - **Steep decay** at late times indicates a **resistive substratum** — the currents
          diffuse slowly and the signal drops off quickly.
        - **Slow, flat decay** at late times suggests a **conductive half-space** that sustains
          eddy currents for longer.
        - A **conductive layer sandwiched** between resistive layers creates a characteristic
          "bump" or shoulder on the decay curve because the eddy currents linger in the
          conductive layer.
        - Try setting a very conductive middle layer (< 5 Ω·m) and a resistive half-space
          to see this effect clearly.
        - The **loop size** controls the volume of ground sampled: a larger loop produces a
          stronger signal and reaches greater depth.
        """
    )
