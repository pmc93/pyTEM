"""
pyTEM: 1-D layered-earth TEM modelling.

Supports circular and square loop geometries, central and offset receivers,
DLF and Euler transforms, with NumPy / Numba / CuPy backends.
"""

from .transform_weights import MU0, HANKEL_FILTERS, FOURIER_FILTERS, EULER_PARAMS
from .backends import HAS_CUDA
from .kernels_numba import HAS_NUMBA
from .recursion import te_reflection_coeff, te_reflection_coeff_grad

from .forward import (
    fwd_circle_central,
    fwd_circle_offset,
    fwd_square_central,
    fwd_square_offset,
    fwd_analytical_central,
    fwd_analytical_offset,
)

from .waveform import convolve_waveform
from .system_filter import butterworth_filter, cascade_filter
from .euler import euler_invert
from .inversion import (getJ_ana, getJ_fd, getR, dbdt_to_apprho, getRMS,
                        getAlpha, getAlphas, invert)
from .plotter import plot_sounding, plot_model, plot_inversion

from .ip_models import (
    pelton_res_rho,
    cole_cole_rho,
    double_pelton_rho,
    get_m_taur_MPA,
    mpa_rho,
    tem_forward_ip,
)

__all__ = [
    # Constants & flags
    'MU0', 'HAS_CUDA', 'HAS_NUMBA',
    'HANKEL_FILTERS', 'FOURIER_FILTERS', 'EULER_PARAMS',
    # Core
    'te_reflection_coeff',
    'te_reflection_coeff_grad',
    # Forward models
    'fwd_circle_central',
    'fwd_circle_offset',
    'fwd_square_central',
    'fwd_square_offset',
    # Analytical
    'fwd_analytical_central',
    'fwd_analytical_offset',
    # Waveform & system filter
    'convolve_waveform',
    'butterworth_filter',
    'cascade_filter',
    # IP models
    'pelton_res_rho',
    'cole_cole_rho',
    'double_pelton_rho',
    'get_m_taur_MPA',
    'mpa_rho',
    'tem_forward_ip',
    # Euler (verification)
    'euler_invert',
    # Inversion
    'getJ_ana',
    'getJ_fd',
    'getR',
    'dbdt_to_apprho',
    'getRMS',
    'getAlpha',
    'getAlphas',
    'invert',
    # Plotting
    'plot_sounding',
    'plot_model',
    'plot_inversion'
]
