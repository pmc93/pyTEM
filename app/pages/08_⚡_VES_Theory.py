import streamlit as st

st.title("⚡ VES Theory")
st.header(":blue[DC Resistivity Sounding — fundamentals]")

# ── Introduction ──────────────────────────────────────────────────────────────
st.subheader(":blue-background[What is a VES survey?]", divider="blue")
st.markdown(
    r"""
    A **Vertical Electrical Sounding (VES)** survey uses a DC (direct current)
    source and four metal stakes pushed into the ground along a straight line.
    Two outer **current electrodes A and B** drive a steady current $I$ into the
    earth; two inner **potential electrodes M and N** measure the resulting
    voltage difference $\Delta V$.

    As the outer spacing $AB$ is widened the current penetrates progressively
    deeper, sampling a larger volume of the subsurface. Dividing the measured
    voltage by the current and scaling by the **geometric factor** $K$ of the
    electrode array yields the **apparent resistivity**:

    $$\rho_a = K \,\frac{\Delta V}{I}$$

    Plotting $\rho_a$ against the half-spacing $AB/2$ on a log-log scale
    produces a **sounding curve** whose shape encodes the layered resistivity
    structure — a highly resistive layer lifts the curve; a conductive layer
    depresses it.

    VES is widely used in groundwater exploration, soil mapping, environmental
    site investigations, and archaeological surveys.
    """
)

with st.expander(":green[**Check your understanding — initial quiz**]"):
    col1, col2 = st.columns(2)
    with col1:
        q1 = st.radio(
            ":blue[**What physical property does VES primarily map?**]",
            ["Seismic velocity", "Gravity", "Electrical resistivity", "Magnetic susceptibility"],
            index=None,
        )
        if q1 == "Electrical resistivity":
            st.success("Correct! The sounding curve is shaped entirely by the layered resistivity (or conductivity) structure.")
        elif q1 is not None:
            st.error("VES measures the apparent resistivity — the ratio of voltage to current scaled by the electrode geometry.")

        q2 = st.radio(
            ":blue[**What happens to the sounding curve when AB/2 is increased?**]",
            [
                "The signal samples shallower material",
                "The signal samples deeper material",
                "The geometry factor K decreases",
                "The current becomes AC",
            ],
            index=None,
        )
        if q2 == "The signal samples deeper material":
            st.success("Correct! Wider electrode spacings drive current deeper, so larger AB/2 carries information about deeper layers.")
        elif q2 is not None:
            st.error("Increasing AB/2 expands the current path depth — this is the basis of 'vertical sounding'.")

    with col2:
        q3 = st.radio(
            ":blue[**In the Schlumberger array, what is moved to extend the sounding?**]",
            ["Only the potential electrodes M, N", "Only the current electrodes A, B", "All four electrodes equally", "The transmitter coil"],
            index=None,
        )
        if q3 == "Only the current electrodes A, B":
            st.success("Correct! In Schlumberger geometry, M and N stay close together while A and B are progressively moved outward.")
        elif q3 is not None:
            st.error("The Schlumberger array keeps M and N nearly fixed and expands A and B to increase depth penetration.")

        q4 = st.radio(
            ":blue[**A deep conductive layer will cause the sounding curve to ...**]",
            ["Rise at large AB/2", "Show no change", "Drop at large AB/2", "Oscillate"],
            index=None,
        )
        if q4 == "Drop at large AB/2":
            st.success("Correct! A conductive (low resistivity) layer pulls ρ_a downward once AB/2 is large enough to sense it.")
        elif q4 is not None:
            st.error("A conductive layer provides a low-resistance path; as AB/2 grows to sample it, ρ_a decreases.")

# ── Schlumberger array ────────────────────────────────────────────────────────
st.subheader(":blue-background[The Schlumberger Electrode Array]", divider="blue")
st.markdown(
    r"""
    The **Schlumberger configuration** is the most common arrangement for VES.
    The four electrodes are collinear with a large ratio $AB/MN$:
    """
)
st.code(
    "A ——————————— M · N ——————————— B\n"
    "|<—— AB/2 ——>|<MN/2>|<—— AB/2 ——>|",
    language=None,
)
st.markdown(
    r"""
    The **geometric factor** for the Schlumberger array is:

    $$K = \frac{\pi\left[\left(\frac{AB}{2}\right)^2 - \left(\frac{MN}{2}\right)^2\right]}{MN} \approx \frac{\pi\left(\frac{AB}{2}\right)^2}{MN} \quad\text{when } AB \gg MN$$

    The **apparent resistivity** at each half-spacing $r = AB/2$ is:

    $$\rho_a(r) = K\,\frac{\Delta V}{I}$$

    In practice a sounding covers $AB/2$ from a few metres to several hundred
    metres, with readings at logarithmically spaced intervals. The M, N spacing
    is kept constant until the measured voltage drops too close to the noise
    floor, at which point M and N are also widened (introducing a small
    overlap). The result is a complete log-log apparent resistivity curve.
    """
)

# ── 1D forward problem ────────────────────────────────────────────────────────
st.subheader(":blue-background[Step 1: The Resistivity Kernel — DC Wait Recursion]", divider="blue")
st.markdown(
    r"""
    For a **horizontally layered 1-D earth** with $N$ layers (the last being an
    infinite half-space), the apparent resistivity for any collinear electrode
    array can be written as a **Hankel transform**:

    $$\rho_a(r) = r^2 \int_0^\infty T(\lambda)\,J_1(\lambda r)\,\lambda\,d\lambda$$

    where $r = AB/2$, $J_1$ is the Bessel function of order 1, and
    $T(\lambda)$ is the **resistivity transform function** (kernel), a
    function of the horizontal wavenumber $\lambda$.

    $T(\lambda)$ is computed by an **upward recursion** starting from the
    bottom half-space (layer $N$):

    $$T_N = \rho_N \qquad\text{(half-space seed)}$$

    Then propagating up through each interface from $j = N-1$ down to $j = 1$:

    $$T_j = \rho_j\,\frac{T_{j+1} + \rho_j\,\tanh(\lambda h_j)}{\rho_j + T_{j+1}\,\tanh(\lambda h_j)}$$

    where $h_j$ is the thickness of layer $j$ and $\rho_j$ its resistivity.
    The surface kernel $T(\lambda) = T_1$ is then used in the transform above.

    > **Note:** This is the DC ($\omega = 0$) analogue of Wait's TE recursion
    > used in TEM modelling. Setting the imaginary (inductive) term to zero
    > recovers this real-valued DC recursion.
    """
)

# ── Linear filter ─────────────────────────────────────────────────────────────
st.subheader(":blue-background[Step 2: Evaluating the Hankel Transform — Linear Filter]", divider="blue")
st.markdown(
    r"""
    The integral $\int_0^\infty T(\lambda)\,J_1(\lambda r)\,\lambda\,d\lambda$
    is evaluated numerically using the **Digital Linear Filter (DLF)** method
    (Ghosh 1971, refined by Guptasarma & Singh 1997).

    The key observation is that by substituting $\lambda = 10^{u}$ and
    $r = 10^{v}$, the convolution becomes a **log-domain cross-correlation**:

    $$\rho_a(r) \approx \sum_{k=1}^{K} \varphi_k \; T\!\left(10^{a_k - \log_{10} r}\right)$$

    where $\{\varphi_k, a_k\}$ are pre-optimised **filter coefficients** that
    depend only on the order of the filter:

    | Filter set | Points ($K$) | Accuracy | Speed |
    |---|---|---|---|
    | Guptasarma 7 | 7 | moderate | fastest |
    | Guptasarma 11 | 11 | good | fast |
    | Guptasarma 22 | 22 | high | slower |

    In practice, the 7-point filter is accurate enough for most exploration
    depths. The 22-point filter is preferred when very high accuracy is needed
    or when the model has thin, strongly contrasting layers.

    The recursion + filter together constitute the **complete 1-D VES forward
    model** used in this app (implemented in `ves/forward/linear_filter.py`).
    """
)

# ── Inversion overview ────────────────────────────────────────────────────────
st.subheader(":blue-background[Step 3: Inversion — Recovering the Layered Model]", divider="blue")
st.markdown(
    r"""
    **Inversion** finds a layered earth model $\mathbf{m}$ (resistivities and
    thicknesses) whose predicted curve $\rho_a^{pred}(\mathbf{m})$ matches the
    observed curve $\rho_a^{obs}$. The model vector has $2N - 1$ parameters:
    $N$ resistivities and $N - 1$ layer thicknesses (the half-space has no
    thickness).

    The misfit is measured by the **RMS error**:

    $$\text{RMS} = \sqrt{\frac{1}{M}\sum_{i=1}^{M}\left(\rho_{a,i}^{obs} - \rho_{a,i}^{pred}\right)^2}$$

    Two iterative methods are supported.
    """
)

col_lm, col_svd = st.columns(2)
with col_lm:
    st.markdown(
        r"""
        **Levenberg-Marquardt (LM)**

        At each iteration a **Jacobian** $\mathbf{J}$ (shape $M \times (2N-1)$)
        is built by finite differences. The model update is:

        $$\Delta\mathbf{m} = \left(\mathbf{J}^T\mathbf{J} + \mu\,\mathbf{I}\right)^{-1}\mathbf{J}^T\,\mathbf{d}$$

        where $\mathbf{d} = \rho_a^{obs} - \rho_a^{pred}$ is the residual and
        $\mu$ is the **damping parameter** that stabilises the inversion.
        """
    )
with col_svd:
    st.markdown(
        r"""
        **Singular Value Decomposition (SVD)**

        SVD decomposes $\mathbf{J} = \mathbf{U}\,\mathbf{S}\,\mathbf{V}^T$ and
        computes the update as:

        $$\Delta\mathbf{m} = \mathbf{V}\,\mathrm{diag}\!\left(\frac{s_k}{s_k + \mu}\right)\mathbf{U}^T\,\mathbf{d}$$

        The $s_k/(s_k + \mu)$ factors **taper small singular values**, providing
        a natural form of regularisation for poorly constrained parameters.
        """
    )

st.markdown(
    "Both methods iterate until the RMS drops below a threshold or a maximum "
    "number of iterations is reached."
)

# ── Depth of investigation ────────────────────────────────────────────────────
st.subheader(":blue-background[Depth of Investigation]", divider="blue")
st.markdown(
    r"""
    A useful rule of thumb for the **maximum depth of investigation** $z_{max}$
    in a VES survey is:

    $$z_{max} \approx \frac{AB/2}{3} \quad\text{to}\quad \frac{AB/2}{5}$$

    depending on the resistivity contrast. In a uniform half-space, the depth
    at which 50 % of the total signal originates is approximately $AB/4$.

    In practice, if you need to resolve a target at depth $z$, you should
    extend the survey to at least $AB/2 \approx 3z$ to ensure sufficient
    current penetration.

    | Target depth | Required AB/2 |
    |---|---|
    | 10 m | ~30 m |
    | 50 m | ~150 m |
    | 100 m | ~300 m |
    | 200 m | ~600 m |
    """
)

# ── Comparison with TEM ───────────────────────────────────────────────────────
st.subheader(":blue-background[VES vs TEM — when to use which?]", divider="blue")
col1, col2 = st.columns(2)
with col1:
    st.markdown(
        r"""
        **VES (DC resistivity)**
        - Galvanic coupling — electrodes must contact the ground
        - Sensitive to resistive AND conductive layers equally
        - Very resistive near-surface (e.g. frozen ground, dry sand) does not
          attenuate the signal
        - Relatively simple equipment
        - Slower: electrode stakes must be moved for each spacing
        - Lateral coverage limited per sounding
        """
    )
with col2:
    st.markdown(
        r"""
        **TEM (Time-Domain EM)**
        - Inductive coupling — no ground contact needed, works on ice/asphalt
        - More sensitive to **conductive** layers (poor contrast in resistive targets)
        - Fast: a single fixed loop covers an entire sounding automatically
        - Well-suited for large-scale surveys with many soundings
        - Signal decays rapidly at late times — depth limited by transmitter moment
        """
    )

with st.expander(":green[**Final quiz — pull it all together**]"):
    col1, col2 = st.columns(2)
    with col1:
        qa = st.radio(
            ":blue[**Which parameter encodes depth in a VES survey?**]",
            ["The frequency of the AC source", "The AB/2 electrode half-spacing",
             "The receiver coil area", "The MN voltage"],
            index=None,
        )
        if qa == "The AB/2 electrode half-spacing":
            st.success("Correct! Increasing AB/2 drives current deeper, so each AB/2 value samples a different depth range.")
        elif qa is not None:
            st.error("VES is a DC method — depth is controlled by the electrode spacing AB/2, not frequency.")

    with col2:
        qb = st.radio(
            ":blue[**The damping parameter μ in LM inversion is used to ...**]",
            [
                "Speed up the Jacobian computation",
                "Stabilise the matrix inversion when J^T J is near-singular",
                "Filter noise from the observed data",
                "Convert resistivity to conductivity",
            ],
            index=None,
        )
        if qb == "Stabilise the matrix inversion when J^T J is near-singular":
            st.success("Correct! The μI term keeps (J^T J + μI) well-conditioned, especially in early iterations.")
        elif qb is not None:
            st.error("The Marquardt damping μ adds a diagonal to J^T J, preventing instability due to near-zero singular values.")
