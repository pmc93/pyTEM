import streamlit as st

st.set_page_config(
    page_title="TEM vs VES",
    page_icon="🌍",
)

st.header(":blue[Two methods for mapping subsurface resistivity]")

st.markdown(
    """
    The **TEM** (transient electromagnetic) and **VES** (vertical electrical sounding) methods
    are two geophysical methods that map the subsurface resistivity
    structure. However, they work in fundamentally different ways.
    """
)

col1, col2 = st.columns(2)

with col1:
    st.subheader(":blue[🧲 TEM - Transient Electromagnetic Method]")
    st.markdown(
        r"""
        A large transmitter (Tx) loop carries a steady current that is abruptly
        switched off. The collapsing magnetic field induces **eddy currents**
        that diffuse downward through the earth. A receiver (Rx) coil records the
        decaying secondary field dB/dt.

        - **Source**: inductive — no ground contact needed
        - **Depth proxy**: time — early = shallow, late = deep
        - **Best for**: conductive targets (clay, saline water)
        - **Data**: dB/dt decay curve (V/m²)
        """
    )

with col2:
    st.subheader(":orange[⚡️ VES - Vertical Electrical Sounding Method]")
    st.markdown(
        r"""
        Current is injected via two electrodes (A, B); a second pair (M, N)
        measures the resulting voltage. Increasing the electrode spacing AB/2
        drives current deeper, sampling greater depth.

        - **Source**: galvanic — electrodes must contact the ground
        - **Depth proxy**: AB electrode spacing — small = shallow, large = deep
        - **Best for**: resistive layers and general stratigraphy
        - **Data**: apparent resistivity curve $\rho_a$(AB/2)
        """
    )

left_co, cent_co, last_co = st.columns((1, 3, 1))
with cent_co:
    st.image(
        "https://upload.wikimedia.org/wikipedia/commons/thumb/5/5b/"
        "Transient_electromagnetic_method.svg/800px-Transient_electromagnetic_method.svg.png",
        caption=(
            "Sketch of a central-loop TEM survey: the transmitter loop (Tx) carries a "
            "steady current that is switched off at t = 0, and the receiver (Rx) records "
            "the decaying secondary field dB/dt."
        ),
    )

st.divider()

st.subheader("Comparison")
st.markdown(
    """
    | Property | 🧲 TEM | ⚡️ VES |
    |---|---|---|
    | Source | Inductive loop | Galvanic electrodes |
    | Depth proxy | Time | AB electrode spacing |
    | Ground contact | Not required | Required |
    | Best sensitivity | Conductive layers | Both — but resistors harder |
    | Key limitation | Noise floor at late times | Equivalence (thin resistors) |
    | Lateral footprint | Tx loop area | AB electrode spacing |
    | Model output | Resistivity + thickness | Resistivity + thickness |
    """
)

st.subheader(':blue[Module overview]', divider="blue")

st.markdown(
    """
    :blue[The module is organised as follows:]
    - 📈 **Forward Modeling ▶️** — build a layered earth model and compute the predicted
      dB/dt (TEM) and apparent resistivity curve (VES) in real time.
    - 📊 **Jacobian & Sensitivity ▶️** — explore which data points are sensitive to which
      layers; compare TEM and VES sensitivity side by side.
    - 🎯 **Inversion ▶️** — run a synthetic inversion for TEM and VES
      and inspect how well each recovers the true model.
    -  **IP Models ▶️** — add induced-polarisation effects (Cole-Cole) to individual
      layers and observe the characteristic sign reversal in the TEM decay.
    - 👉 **About** — references and acknowledgements.
    """
)


