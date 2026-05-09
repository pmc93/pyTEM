import streamlit as st

st.title("📡 TEM Theory")
st.header(":orange[Electromagnetic foundations of TEM sounding]")

# ── Introduction ─────────────────────────────────────────────────────────────
st.subheader(":orange-background[What is a TEM survey?]", divider="orange")
st.markdown(
    """
    A **Time-Domain Electromagnetic (TEM)** survey uses a large rectangular or
    circular transmitter loop laid flat on the ground. The loop carries a steady
    current for some time, then the current is **abruptly switched off** (a
    step-off transient). The collapse of the transmitter's magnetic field
    induces **eddy currents** in any conductive material below. These currents
    diffuse downward and outward — slower in resistive rock, faster in
    conductive sediments — and the secondary magnetic field they produce is
    recorded at a receiver (Rx) coil as a function of time.

    The recorded quantity is typically $\\partial B_z / \\partial t$ (written
    $\\dot{B}_z$ or dB/dt), the time derivative of the vertical component of
    the secondary magnetic flux density. Early times carry information about
    shallow layers; late times carry information about deeper layers.
    """
)

with st.expander(":green[**Check your understanding — initial quiz**]"):
    col1, col2 = st.columns(2)
    with col1:
        q1 = st.radio(
            ":orange[**What physical property does TEM primarily map?**]",
            ["Density", "Seismic velocity", "Electrical resistivity", "Magnetic susceptibility"],
            index=None,
        )
        if q1 == "Electrical resistivity":
            st.success("Correct! TEM is sensitive to the subsurface resistivity (or conductivity) structure.")
        elif q1 is not None:
            st.error("Not quite. TEM eddy currents are governed by Ohm's law — they flow more easily in conductive (low resistivity) material.")

        q2 = st.radio(
            ":orange[**Early-time TEM gates are sensitive to ...**]",
            ["Deep layers", "Shallow layers", "All layers equally", "The air layer above the surface"],
            index=None,
        )
        if q2 == "Shallow layers":
            st.success("Correct! At early times, the eddy current ring has not yet diffused far into the ground and samples shallow structure.")
        elif q2 is not None:
            st.error("Think about where the eddy currents are at very early times after switch-off — they start near the surface.")

    with col2:
        q3 = st.radio(
            ":orange[**Which receiver placement gives the cleanest signal from a horizontal loop?**]",
            ["At the centre of the transmitter loop", "Far outside the loop", "Vertically above the loop", "Buried beneath the loop"],
            index=None,
        )
        if q3 == "At the centre of the transmitter loop":
            st.success("Correct! The central-loop configuration places the Rx at the geometric centre where the primary field is most uniform.")
        elif q3 is not None:
            st.error("The central-loop configuration is the most common for 1-D sounding because the primary field is nearly uniform there.")

        q4 = st.radio(
            ":orange[**What happens to the dB/dt signal over time in a conductive halfspace?**]",
            ["It stays constant", "It decays monotonically", "It first rises then falls", "It oscillates"],
            index=None,
        )
        if q4 == "It decays monotonically":
            st.success("Correct! For a pure resistive/conductive earth (no IP) the step-off dB/dt decays monotonically in time.")
        elif q4 is not None:
            st.error("Without induced polarisation effects, the dB/dt decays monotonically — try the Forward Explorer to see this!")

# ── Wait's recursion ─────────────────────────────────────────────────────────
st.subheader(":orange-background[Step 1: TE Reflection Coefficient — Wait's Recursion]", divider="orange")
st.markdown(
    r"""
    For a **1-D horizontally layered earth**, the electromagnetic response can be
    decomposed into a **TE (Transverse Electric)** mode, which is the only mode
    excited by a horizontal loop (vertical magnetic dipole) source.

    The TE surface reflection coefficient $r_{TE}(\lambda, \omega)$ encodes the
    full subsurface response as a function of:
    - $\lambda$ — horizontal wavenumber (spatial frequency along the surface)
    - $\omega$ — angular frequency of the EM field

    **Vertical wavenumber** in layer $j$ (with conductivity $\sigma_j = 1/\rho_j$):

    $$\Gamma_j = \sqrt{\lambda^2 + j\omega\,\mu_0\,\sigma_j}$$

    **Wait's upward recursion** starts from the bottom half-space (no reflector
    below an infinite medium, so $\gamma_N = 0$) and propagates upward through
    each interface:

    $$\psi_{j+1} = \frac{\Gamma_j - \Gamma_{j+1}}{\Gamma_j + \Gamma_{j+1}}, \qquad
    \gamma_j = e^{-2\Gamma_j h_j}\,
    \frac{\gamma_{j+1} + \psi_{j+1}}{1 + \gamma_{j+1}\,\psi_{j+1}}$$

    where $h_j$ is the thickness of layer $j$. After propagating from the
    deepest layer to the surface, $r_{TE} = \gamma_1$ gives the **surface
    reflection coefficient** used in the subsequent Hankel transform.

    This recursion is implemented in `pytem/recursion.py` as
    `te_reflection_coeff`.
    """
)

# ── Hankel transform ──────────────────────────────────────────────────────────
st.subheader(":orange-background[Step 2: Frequency-Domain Field — Hankel J₁ DLF]", divider="orange")
st.markdown(
    r"""
    The **secondary vertical magnetic field** at the receiver requires integrating
    $r_{TE}$ over all horizontal wavenumbers, weighted by the loop geometry:

    $$H_z^{sec}(\omega) = \frac{1}{2}\int_0^\infty r_{TE}(\lambda,\omega)\,
    \lambda\,J_1(\lambda\,a)\,d\lambda$$

    where $a$ is the loop radius and $J_1$ is the Bessel function of order 1.

    This **Hankel transform** is evaluated numerically using **Key's (2009)
    Digital Linear Filter (DLF)** — a set of pre-optimised weights $w_k$ and
    base points $\lambda_k / a$:

    $$\int_0^\infty f(\lambda)\,d\lambda \;\approx\; \frac{1}{a}
    \sum_k w_k\, f\!\left(\frac{\lambda_k}{a}\right)$$

    pyTEM ships two Hankel filter lengths (101 and 201 points). The 201-point
    filter is more accurate; the 101-point filter is faster and sufficient for
    most applications.
    """
)

# ── Fourier transform ─────────────────────────────────────────────────────────
st.subheader(":orange-background[Step 3: Time-Domain Response — Fourier Sine DLF]", divider="orange")
st.markdown(
    r"""
    TEM data are recorded in the **time domain**. Converting
    $H_z^{sec}(\omega)$ to $\partial B_z/\partial t$ requires a **Fourier
    transform**. For a step-off transient (the current switches from $I$ to 0),
    the relevant transform is:

    $$\frac{\partial B_z}{\partial t}(t) =
    -\frac{\mu_0\,I}{\pi}\int_0^\infty
    \mathrm{Im}\!\left[H_z^{sec}(\omega)\right]\sin(\omega t)\,d\omega$$

    This is again replaced by a DLF weighted sum (Key 2009), evaluated at the
    Fourier sample points $\omega_k = f_k / t$:

    $$\frac{\partial B_z}{\partial t}(t) \approx
    \frac{1}{t}\sum_k w_k^{(F)}\,\mu_0\,
    \mathrm{Im}\!\left[H_z^{sec}(\omega_k)\right]$$

    pyTEM ships two Fourier filter lengths (81 and 101 points). An alternative
    **Euler/Stehfest** inverse Laplace transform is also available and useful
    for checking numerical accuracy.
    """
)

# ── Loop geometries ───────────────────────────────────────────────────────────
st.subheader(":orange-background[Loop Geometries Supported by pyTEM]", divider="orange")
st.markdown(
    """
    pyTEM implements four source-receiver configurations:

    | Function | Tx shape | Rx position | Notes |
    |----------|----------|-------------|-------|
    | `fwd_circle_central` | Circular | Centre of loop | Most common; symmetric |
    | `fwd_circle_offset` | Circular | Radial offset | For offset-loop surveys |
    | `fwd_square_central` | Square | Centre of loop | Square loops in the field |
    | `fwd_square_offset` | Square | Arbitrary (x, y) | Full 2-D offset |

    For square loops, the Hankel transform is replaced by a 2-D Gauss-Legendre
    quadrature over the loop perimeter, decomposed into vertical magnetic dipole
    (VMD) contributions from each element.

    The **equivalent-area circle approximation** $a_{\\text{eff}} = L / \\sqrt{\\pi}$
    (where $L$ is the square side length) is often used as a fast surrogate for
    a square loop in the circular forward function. It is accurate at late times
    but introduces small errors at early times.
    """
)

# ── Compute backends ──────────────────────────────────────────────────────────
st.subheader(":orange-background[Compute Backends]", divider="orange")
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown(
        """
        **NumPy (default)**
        - Pure Python + NumPy
        - No extra dependencies
        - Reference implementation
        - ~seconds per sounding
        """
    )
with col2:
    st.markdown(
        """
        **Numba (JIT)**
        - Compiled inner loops
        - ~10× faster than NumPy
        - Small first-call overhead
        - Enabled when `numba` installed
        """
    )
with col3:
    st.markdown(
        """
        **CuPy (GPU)**
        - CUDA GPU acceleration
        - ~100× faster than NumPy
        - Requires NVIDIA GPU + CUDA
        - Fastest for large surveys
        """
    )

st.info(
    "In this app, computations use whichever backend is available. "
    "The forward calls are cached so re-running with the same parameters "
    "is instantaneous."
)
