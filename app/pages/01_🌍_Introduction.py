import streamlit as st

st.title("🌍 Introduction")
st.header(":blue[Two methods for mapping subsurface resistivity]")

st.markdown(
    """
    **TEM** (time-domain EM) and **VES** (vertical electrical sounding) are two
    complementary geophysical methods that both map the subsurface resistivity
    structure — but they work in fundamentally different ways.
    """
)

col1, col2 = st.columns(2)

with col1:
    st.subheader(":blue[⚡ TEM — Time-Domain EM]")
    st.markdown(
        r"""
        A large transmitter loop carries a steady current that is abruptly
        switched off. The collapsing magnetic field induces **eddy currents**
        that diffuse downward through the earth. A receiver coil records the
        decaying secondary field $\partial B_z / \partial t$.

        - **Source**: inductive — no ground contact needed
        - **Depth proxy**: time $t$ — early = shallow, late = deep
        - **Best for**: conductive targets (clay, saline water)
        - **Data**: dB/dt decay curve (V/m² or T/s)
        """
    )

with col2:
    st.subheader(":orange[📈 VES — Vertical Electrical Sounding]")
    st.markdown(
        r"""
        Current is injected via two electrodes (A, B); a second pair (M, N)
        measures the resulting voltage. Increasing the electrode spacing AB/2
        drives current deeper, sampling greater depth.

        - **Source**: galvanic — electrodes must contact the ground
        - **Depth proxy**: electrode spacing $AB/2$
        - **Best for**: resistive layers and general stratigraphy
        - **Data**: apparent resistivity curve $\rho_a(AB/2)$
        """
    )

st.divider()

st.subheader("Comparison")
st.markdown(
    """
    | Property | ⚡ TEM | 📈 VES |
    |---|---|---|
    | Source | Inductive loop | Galvanic electrodes |
    | Depth proxy | Time $t$ | Spacing $AB/2$ |
    | Ground contact | Not required | Required |
    | Best sensitivity | Conductive layers | Both — but resistors harder |
    | Key limitation | Noise floor at late times | Equivalence (thin resistors) |
    | Lateral footprint | ~loop area | ~AB/2 in each direction |
    | Model output | Resistivity + thickness | Resistivity + thickness |
    """
)

st.divider()

st.subheader("The forward problem")
st.markdown(
    r"""
    Both methods assume a **1-D layered earth**: horizontal layers with
    uniform resistivity $\rho_j$ and thickness $h_j$.

    **TEM** — Wait's EM recursion in the frequency domain, transformed to time
    via the Fourier digital linear filter (DLF):

$$\frac{\partial B_z}{\partial t}(t) \approx \frac{1}{t}\sum_k w_k^{(F)}\,\mu_0\,\mathrm{Im}\!\left[H_z^{sec}(\omega_k)\right]$$

    **VES** — DC Wait recursion upward through layers, transformed with the
    Hankel DLF (Guptasarma & Singh 1997):

$$\rho_a(r) \approx r^2 \sum_k w_k\,T\!\left(\frac{\lambda_k}{r}\right)$$

    where $T(\lambda)$ is the DC kernel evaluated at each filter node.
    """
)

st.divider()

st.subheader("Module structure")
st.markdown(
    """
    Navigate using the sidebar:

    | Page | Content |
    |---|---|
    | **📈 Forward Modeling ▶️** | Build a model; compute dB/dt (TEM) and apparent resistivity (VES) |
    | **📊 Jacobian & Sensitivity ▶️** | Sensitivity heatmap for TEM and VES side by side |
    | **🎯 Inversion ▶️** | Recover a model from noisy synthetic data (TEM and VES) |
    | ** IP Models ▶️** | Induced polarisation (Cole-Cole) in TEM |
    | **👉 About** | References and acknowledgements |
    """
)
