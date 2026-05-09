import streamlit as st

st.title("👉 About pyTEM")
st.header(":blue[References, acknowledgements, and how to run the app]")

st.subheader(":blue-background[How to launch the app]", divider="blue")
st.markdown(
    """
    From inside the `pyTEM/app/` directory, run:

    ```bash
    streamlit run TEM_Explorer.py
    ```

    The app requires the `pytem` package (located one level up in `pyTEM/pytem/`).
    The path is added automatically by each page — no `pip install` of pyTEM is needed.
    """
)

st.subheader(":blue-background[Dependencies]", divider="blue")
col1, col2 = st.columns(2)
with col1:
    st.markdown(
        """
        **Required:**
        - `streamlit`
        - `numpy`
        - `scipy`
        - `matplotlib`
        """
    )
with col2:
    st.markdown(
        """
        **Optional (speed):**
        - `numba` — JIT-compiled CPU kernels (~10× faster)
        - `cupy` — CUDA GPU kernels (~100× faster)
        """
    )

st.subheader(":blue-background[Package structure]", divider="blue")
st.markdown(
    """
    | Module | Contents |
    |--------|----------|
    | `pytem/recursion.py` | Wait's upward TE recursion + adjoint gradient |
    | `pytem/transform_weights.py` | DLF filter coefficients (Key 2009, 2012) + Euler weights |
    | `pytem/forward.py` | Public forward model API — four loop geometries |
    | `pytem/inversion.py` | Jacobian, regularised Gauss-Newton inversion |
    | `pytem/system_filter.py` | Butterworth and cascade instrument filters |
    | `pytem/waveform.py` | Piecewise-linear transmitter waveform convolution |
    | `pytem/ip_models.py` | Pelton, Cole-Cole, double-Pelton, MPA IP models |
    | `pytem/plotter.py` | Convenience plotting utilities |
    | `pytem/backends.py` | CuPy / CUDA detection |
    | `pytem/kernels_numba.py` | Numba JIT forward kernels |
    | `pytem/kernels_gpu.py` | CuPy GPU forward kernels |
    | `pytem/kernels_jacobian.py` | Numba + GPU analytical Jacobian kernels |
    """
)

st.subheader(":blue-background[References]", divider="blue")
st.markdown(
    """
    - **Key, K.** (2009). 1D inversion of multicomponent, multifrequency marine CSEM data:
      Methodology and synthetic studies for resolving thin resistive layers.
      *Geophysics*, **74**(2), F9-F20. DLF filter coefficients used in forward model.

    - **Wait, J. R.** (1954). Mutual coupling of loops lying on the ground.
      *Geophysics*, **19**(2), 290-296. Upward TE recursion.

    - **Ward, S. H., & Hohmann, G. W.** (1988). Electromagnetic theory for geophysical
      applications. In M. N. Nabighian (Ed.), *Electromagnetic Methods in Applied
      Geophysics*, Vol. 1. SEG. Layered-earth theory and analytical solutions.

    - **Constable, S. C., Parker, R. L., & Constable, C. G.** (1987). Occam's inversion:
      A practical algorithm for generating smooth models from EM sounding data.
      *Geophysics*, **52**(3), 289-300. Regularised inversion framework.

    - **Pelton, W. H., Ward, S. H., Hallof, P. G., Sill, W. R., & Nelson, P. H.** (1978).
      Mineral discrimination and removal of inductive coupling with multifrequency IP.
      *Geophysics*, **43**(3), 588-609. Pelton IP model.

    - **Cole, K. S., & Cole, R. H.** (1941). Dispersion and absorption in dielectrics.
      *Journal of Chemical Physics*, **9**, 341-351. Cole-Cole model.

    - **Fiandaca, G., Madsen, L. M., & Maurya, P. K.** (2018). Re-parameterisations of the
      Cole-Cole model for improved spectral inversion of induced polarisation data.
      *Near Surface Geophysics*, **16**(4), 385-399. MPA model used in pyTEM.

    - **Abate, J., & Whitt, W.** (1995). Numerical inversion of Laplace transforms of
      probability distributions. *ORSA Journal on Computing*, **7**(1), 36-43.
      Euler-Stehfest acceleration weights.
    """
)

st.subheader(":blue-background[Notebooks]", divider="blue")
st.markdown(
    """
    The `pyTEM/notebooks/` folder contains Jupyter notebooks that accompany this app:

    | Notebook | Contents |
    |----------|----------|
    | `1. pytem_fwd.ipynb` | Forward modelling walkthrough and validation |
    | `2. pytem_getJ.ipynb` | Jacobian computation and speed benchmarks |
    | `3. pytem_inv.ipynb` | Inversion tutorial |
    | `4. pytem_inv_with_geometries.ipynb` | Multi-geometry inversion examples |
    | `5. pytem_inversion_examples.ipynb` | End-to-end worked examples |
    """
)

st.subheader(":blue-background[About this module]", divider="blue")
st.markdown(
    """
    This interactive module accompanies the **pyTEM** Python package for
    1-D layered-earth Time-Domain Electromagnetic (TEM) modelling and inversion.
    It is built with [Streamlit](https://streamlit.io) and is structured similarly
    to the Groundwater Project interactive learning modules.

    The module is designed to help students and practitioners develop intuition for
    TEM sounding data — how the dB/dt response is shaped by the subsurface resistivity
    structure, how the Jacobian governs what the inversion can resolve, and how
    instrument filters and IP effects appear in field data.
    """
)

st.subheader(":blue-background[pyTEM package]", divider="blue")
st.markdown(
    """
    **pyTEM** implements:

    - 1-D forward modelling via Wait's upward TE recursion + digital linear filter (DLF) transforms
    - Four loop geometries: circular/square, central/offset
    - Three compute backends: NumPy, Numba JIT, and CuPy (CUDA GPU)
    - Analytical Jacobian via the adjoint recursion (Constable et al., 1987)
    - Regularised Gauss-Newton inversion with automatic regularisation search
    - Instrument system filters (Butterworth, cascade)
    - Transmitter waveform convolution
    - Induced polarisation models: Pelton, Cole-Cole, Double Pelton, MPA
    """
)

st.subheader(":blue-background[Key references]", divider="blue")
st.markdown(
    """
    **Digital linear filters:**
    - Key, K. (2009). 1D inversion of multicomponent, multifrequency marine CSEM data:
      Methodology and synthetic studies for resolving thin resistive layers.
      *Geophysics*, 74(2), F9–F20.
    - Key, K. (2012). Is the fast Hankel transform faster than quadrature?
      *Geophysics*, 77(3), F21–F30.

    **Layered-earth EM theory:**
    - Ward, S. H., & Hohmann, G. W. (1988). Electromagnetic theory for geophysical
      applications. In M. N. Nabighian (ed.), *Electromagnetic Methods in Applied
      Geophysics*, Vol. 1. SEG.

    **Wait's recursion:**
    - Wait, J. R. (1954). Mutual electromagnetic coupling of loops over a
      homogeneous ground. *Geophysics*, 19(2), 290–296.

    **Gauss-Newton inversion and Jacobian:**
    - Constable, S. C., Parker, R. L., & Constable, C. G. (1987). Occam's inversion:
      A practical algorithm for generating smooth models from electromagnetic sounding data.
      *Geophysics*, 52(3), 289–300.

    **IP models:**
    - Pelton, W. H., Ward, S. H., Hallof, P. G., Sill, W. R., & Nelson, P. H. (1978).
      Mineral discrimination and removal of inductive coupling with multifrequency IP.
      *Geophysics*, 43(3), 588–609.
    - Cole, K. S., & Cole, R. H. (1941). Dispersion and absorption in dielectrics.
      *Journal of Chemical Physics*, 9(4), 341–351.
    - Fiandaca, G., Ramm, J., Christiansen, A. V., Gazoty, A., Auken, E., &
      Binley, A. (2012). Resolving spectral information from time domain induced
      polarization data through 2-D inversion. *Geophysical Journal International*,
      192(2), 631–646.

    **Euler/Stehfest inverse Laplace transform:**
    - Abate, J., & Whitt, W. (1995). Numerical inversion of Laplace transforms of
      probability distributions. *ORSA Journal on Computing*, 7(1), 36–43.
    """
)

st.subheader(":blue-background[Acknowledgements]", divider="blue")
st.markdown(
    """
    This module was developed at the **Technical University of Denmark (DTU)**.

    The Streamlit multi-page app structure is inspired by the
    [Groundwater Project](https://gw-project.org) interactive learning modules.
    """
)

st.divider()
st.caption("pyTEM TEM Explorer | Built with Streamlit | DTU")
