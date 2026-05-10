import streamlit as st

st.set_page_config(
    page_title="🌍 TEM vs VES",
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
    st.markdown("""
<div style="background-color:#1a3a5c; border-left:6px solid steelblue;
            border-radius:6px; padding:1rem 1.2rem;">
  <h4 style="color:steelblue; margin-top:0;">🧲 TEM — Transient Electromagnetic Method</h4>
  <p>A large transmitter (Tx) loop carries a steady current that is abruptly switched off.
  The collapsing magnetic field induces <b>eddy currents</b> that diffuse downward through
  the earth. A receiver (Rx) coil records the decaying secondary field dB/dt.</p>
  <ul style="margin-bottom:0;">
    <li><b>Source:</b> inductive — no ground contact needed</li>
    <li><b>Depth proxy:</b> time — early = shallow, late = deep</li>
    <li><b>Best for:</b> conductive targets (clay, saline water)</li>
    <li><b>Data:</b> dB/dt decay curve (V/m² or T/s)</li>
  </ul>
</div>
""", unsafe_allow_html=True)

with col2:
    st.markdown("""
<div style="background-color:#3d2200; border-left:6px solid darkorange;
            border-radius:6px; padding:1rem 1.2rem;">
  <h4 style="color:darkorange; margin-top:0;">⚡️ VES — Vertical Electrical Sounding Method</h4>
  <p>Current is injected via two electrodes (A, B); a second pair (M, N) measures the
  resulting voltage. Increasing the electrode spacing AB/2 drives current deeper,
  sampling greater depth.</p>
  <ul style="margin-bottom:0;">
    <li><b>Source:</b> galvanic — electrodes must contact the ground</li>
    <li><b>Depth proxy:</b> electrode spacing AB/2 — small = shallow, large = deep</li>
    <li><b>Best for:</b> resistive layers and general stratigraphy</li>
    <li><b>Data:</b> apparent resistivity curve ρₐ(AB/2)</li>
  </ul>
</div>
""", unsafe_allow_html=True)

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
    | Depth proxy | Time $t$ | Spacing $AB/2$ |
    | Ground contact | Not required | Required |
    | Best sensitivity | Conductive layers | Both — but resistors harder |
    | Key limitation | Noise floor at late times | Equivalence (thin resistors) |
    | Lateral footprint | ~loop area | ~AB/2 in each direction |
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
    - 🎯 **Inversion ▶️** — run a synthetic inversion for TEM (Gauss-Newton) and VES
      (Levenberg-Marquardt) and inspect how well each recovers the true model.
    -  **IP Models ▶️** — add induced-polarisation effects (Cole-Cole) to individual
      layers and observe the characteristic sign reversal in the TEM decay.
    - 👉 **About** — references and acknowledgements.
    """
)


