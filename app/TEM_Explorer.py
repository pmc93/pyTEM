import streamlit as st

st.set_page_config(
    page_title="pyTEM — TEM Explorer",
    page_icon="📡",
)

st.title("pyTEM — TEM Explorer 📡")

st.sidebar.success("☝️ Select a page above ☝️")

st.header(':blue-background[Welcome 👋]')

st.markdown(
    """
    **Time-Domain Electromagnetic (TEM)** surveys are one of the most widely used geophysical
    methods for mapping subsurface resistivity. A large transmitter loop is energised with a
    steady current; when the current is abruptly switched off, the collapse of the magnetic field
    induces eddy currents in the ground. These currents diffuse downward and outward over time,
    and the secondary magnetic field they produce is measured at a receiver. The recorded
    **dB/dt** decay curve carries information about the resistivity structure of the earth
    from the surface down to hundreds of metres.

    **pyTEM** is a pure-Python package for **1-D layered-earth TEM forward modelling and
    inversion**. It supports four loop geometries, three compute backends (NumPy / Numba / CuPy),
    and a full regularised Gauss-Newton inversion with optional induced-polarisation (IP)
    models and system-filter correction.
    """
)

st.subheader(':blue[Overview of the module]', divider="blue")

st.markdown(
    """
    This interactive module walks you through the key concepts behind TEM geophysics and
    the pyTEM implementation. It combines theory with live interactive applications so you
    can develop intuition for how the dB/dt signal depends on model parameters.

    :blue[The module is organised as follows:]

    - 📡 **TEM Theory** — electromagnetic fundamentals, Wait's recursion, and the
      digital linear filter (DLF) transforms used to compute the response.
    - 🔄 **Forward Explorer** — build your own layered resistivity model and see the
      predicted dB/dt curve update in real time as you move the sliders.
    - 📊 **Jacobian & Sensitivity** — explore which time gates are sensitive to which
      layers and understand how the sensitivity matrix drives the inversion.
    - 🎯 **Inversion** — run a synthetic Gauss-Newton inversion and inspect how well
      the algorithm recovers the true model from noisy data.
    - 🔊 **System Filters** — apply a Butterworth or cascade instrument filter and see
      how it modifies the high-frequency end of the dB/dt response.
    - 💡 **IP Models** — add induced-polarisation effects (Pelton / Cole-Cole) to
      individual layers and observe the characteristic sign reversal in the decay.
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

    | Page | Symbol | Content |
    |------|--------|---------|
    | TEM Theory | 📡 | Background physics and transforms |
    | Forward Explorer | 🔄 ▶️ | Build a model, compute dB/dt live |
    | Jacobian & Sensitivity | 📊 ▶️ | Sensitivity heatmap |
    | Inversion | 🎯 ▶️ | Run a synthetic inversion |
    | System Filters | 🔊 ▶️ | Instrument filter effects |
    | IP Models | 💡 ▶️ | Induced polarisation in TEM |
    | About | 👉 | References and acknowledgements |
    """
)
