import streamlit as st

st.set_page_config(
    page_title="TEM vs VES — Resistivity Methods",
    page_icon="🌍",
)

st.title("TEM vs VES 🌍")
st.subheader("Comparing two methods for mapping subsurface resistivity")

st.header(':blue-background[Welcome 👋]')

st.markdown(
    """
    **TEM** (Time-Domain Electromagnetics) and **VES** (Vertical Electrical Sounding) are
    two complementary geophysical methods that both map subsurface resistivity — but they
    work in fundamentally different ways.

    **TEM** uses an inductive source (a large loop): a steady current is switched off, and
    the decaying secondary field **dB/dt** is recorded. No ground contact is needed.

    **VES** uses a galvanic source (electrodes): current is injected into the ground and
    the resulting voltage is measured. The apparent resistivity curve $\\rho_a(AB/2)$
    is interpreted to recover the 1-D resistivity structure.

    This interactive module lets you explore and compare both methods side by side.
    """
)

st.subheader(':blue[Module overview]', divider="blue")

st.markdown(
    """
    This interactive module covers the key concepts behind **TEM** and **VES** geophysics
    and the pyTEM / VES implementation. Pages marked with ▶️ contain live interactive
    applications — try adjusting the sliders and tables!

    :blue[The module is organised as follows:]

    - 🌍 **Introduction** — side-by-side overview of TEM and VES: physics, depth proxy,
      and the 1-D forward problem for both methods.
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

st.subheader(':blue[Navigation]', divider="blue")
st.markdown(
    """
    Use the **sidebar** on the left to jump to any section.
    Pages marked with ▶️ contain live interactive applications — try adjusting the sliders!

    | Page | Content |
    |------|---------|
    | 🌍 Introduction | TEM vs VES — physics, depth proxy, and the 1-D forward problem |
    | 📈 Forward Modeling ▶️ | Build a model; compute dB/dt (TEM) and apparent resistivity (VES) |
    | 📊 Jacobian & Sensitivity ▶️ | Sensitivity heatmap for TEM and VES side by side |
    | 🎯 Inversion ▶️ | Recover a model from noisy synthetic data (TEM and VES) |
    | 🔊 System Filters ▶️ | Instrument filter effects on the TEM response |
    | 💡 IP Models ▶️ | Induced polarisation (Cole-Cole) in TEM |
    | 👉 About | References and acknowledgements |
    """
)
