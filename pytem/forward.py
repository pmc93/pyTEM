"""
forward.py — TEM forward modelling functions and analytical solutions.

Internal helpers:
    _rx_scale, _resolve_filters, _resolve_gpu_filters,
    _apply_signal_scaling, _precompute_filter_dlf, _precompute_filter_euler,
    _vmd_greenfct, _build_circular_geometry, _build_square_geometry,
    _run_circular, _run_square

Forward functions:
    fwd_circle_central  — central-loop circular Tx, Rx at loop centre
    fwd_circle_offset   — circular Tx, Rx at radial offset
    fwd_square_central  — central square-loop Tx, Rx at loop centre
    fwd_square_offset   — square Tx, Rx at arbitrary (rx_x, rx_y)

Analytical solutions:
    fwd_analytical_central         — Ward & Hohmann eq 4.69a
    fwd_analytical_offset  — Ward & Hohmann eq 4.97

Adding a new geometry
---------------------
1. Write _build_<name>_geometry(...) to produce geometry arrays (distances
   and quadrature weights) describing source-receiver relationships.
2. Call _run_circular (J1 Hankel kernel) or _run_square (J0 VMD area integral).
3. Write a public fwd_<name>: cast inputs -> resolve filters -> precompute filter
   -> build geometry -> run kernel -> quadrature scale if needed
   -> _apply_signal_scaling -> scale by rx_fac.
"""

import numpy as np
from scipy.special import j0, erf

from .filters import MU0, HANKEL_FILTERS, FOURIER_FILTERS, EULER_PARAMS
from .backends import HAS_CUDA, GPU_HANKEL, GPU_FOURIER
from .recursion import te_reflection_coeff
from .kernels_numba import (
    HAS_NUMBA,
    _tem_circular_jit, _tem_square_jit,
    _tem_circular_euler_jit, _tem_square_euler_jit,
)

if HAS_CUDA:
    import cupy as cp
    from .kernels_gpu import (
        _tem_circular_gpu, _tem_square_gpu,
        _tem_circular_euler_gpu, _tem_square_euler_gpu,
    )


# ============================================================================
# Helper functions
# ============================================================================

def _rx_scale(rx_area=1.0, rx_turns=1):
    """Receiver effective area scaling factor (N * A)."""
    return float(rx_area) * float(rx_turns)


def _resolve_filters(hankel_filter, fourier_filter, transform, euler_order=11):
    """Look up filter coefficient arrays from registries."""
    h_base, h_j0, h_j1 = HANKEL_FILTERS[hankel_filter]
    if transform == 'dlf':
        f_base, f_sin, f_cos = FOURIER_FILTERS[fourier_filter]
        return h_base, h_j0, h_j1, f_base, f_sin, f_cos, None, None
    else:  # 'euler'
        e_eta, e_A = EULER_PARAMS[euler_order]
        return h_base, h_j0, h_j1, None, None, None, e_eta, e_A


def _resolve_gpu_filters(hankel_filter, fourier_filter, transform):
    """Look up GPU (CuPy) filter arrays from registries."""
    d_h_base, d_h_j0, d_h_j1 = GPU_HANKEL[hankel_filter]
    if transform == 'dlf':
        d_f_base, d_f_sin, d_f_cos = GPU_FOURIER[fourier_filter]
        return d_h_base, d_h_j0, d_h_j1, d_f_base, d_f_sin, d_f_cos
    else:
        return d_h_base, d_h_j0, d_h_j1, None, None, None


def _apply_signal_scaling(dbdt, current, signal, transform):
    """Apply current amplitude and step-off/step-on/impulse sign convention."""
    if transform == 'euler':
        dbdt *= current * float(signal)
    else:  # dlf
        if signal == 1:
            dbdt *= -current * 2.0 / np.pi
        else:
            dbdt *= current * 2.0 / np.pi
    return dbdt


def _precompute_filter_dlf(system_filter, times, f_base):
    """Pre-evaluate system_filter(omega) at all DLF Fourier frequencies."""
    n_t = len(times)
    n_f = len(f_base)
    fw = np.empty((n_t, n_f), dtype=np.complex128)
    for i in range(n_t):
        omega_row = f_base / times[i]
        fw[i] = system_filter(omega_row)
    return fw


def _precompute_filter_euler(system_filter, times, e_eta, e_A):
    """Pre-evaluate system_filter(omega) at all Euler/Bromwich points."""
    n_t = len(times)
    n_euler = len(e_eta)
    fw = np.empty((n_t, n_euler), dtype=np.complex128)
    for i in range(n_t):
        t = times[i]
        c = e_A / (2.0 * t)
        h_step = np.pi / t
        ks = np.arange(n_euler, dtype=float)
        s_vals = c + ks * h_step * 1j
        omega_vals = s_vals / 1j
        fw[i] = system_filter(omega_vals)
    return fw


def _vmd_greenfct(offset_dist, omega, thicknesses, resistivities, h_base, h_j0):
    """VMD Green's function G(offset_dist, omega) via J0 Hankel DLF.

    offset_dist : source-element to receiver distance [m]
    """
    lam = h_base / offset_dist
    r_te = te_reflection_coeff(lam, omega, thicknesses, resistivities)
    return np.dot(r_te * lam**2, h_j0) / offset_dist / (4.0 * np.pi)


# ============================================================================
# Geometry builders
# ============================================================================

def _build_circular_geometry(tx_radius, h_base, rx_offset=0.0):
    """
    Pre-compute per-wavenumber extra weights for a circular Tx loop.

    Central receiver (rx_offset=0): returns ones (standard J1 loop integral).
    Offset receiver: returns J0(lam * rx_offset) - the additional Bessel factor.
    """
    lam = h_base / float(tx_radius)
    if rx_offset == 0.0:
        return np.ones(len(lam), dtype=float)
    return j0(lam * float(rx_offset))


def _build_square_geometry(side_length, n_quad, rx_x=0.0, rx_y=0.0,
                           use_symmetry=True):
    """
    Build Gauss-Legendre quadrature nodes and weights for a square Tx loop.
    Returns (offset_dist_q, area_w).

    Central (use_symmetry=True):
        Integrates over one quadrant [0, L/2]^2, exploiting x<->y symmetry.
        The caller must multiply dbdt by 4 to account for all four quadrants.

    Offset (use_symmetry=False):
        Integrates over the full square [-L/2, L/2]^2.
        Distances clipped at 1e-6 m to avoid singularity when Rx is inside the loop.
    """
    L = float(side_length)
    hs = L / 2.0
    gl_nodes, gl_weights = np.polynomial.legendre.leggauss(n_quad)

    if use_symmetry:
        x_pts = hs / 2.0 * (1.0 + gl_nodes)
        w_pts = gl_weights * hs / 2.0
        offset_dist_q, area_w = [], []
        for i in range(n_quad):
            for jj in range(i, n_quad):
                w = w_pts[i] * w_pts[jj] * (2.0 if i != jj else 1.0)
                offset_dist_q.append(np.sqrt(x_pts[i]**2 + x_pts[jj]**2))
                area_w.append(w)
        return np.asarray(offset_dist_q, dtype=float), np.asarray(area_w, dtype=float)
    else:
        x_pts = hs * gl_nodes
        wx = hs * gl_weights
        xx, yy = np.meshgrid(x_pts, x_pts, indexing='xy')
        ww_x, ww_y = np.meshgrid(wx, wx, indexing='xy')
        offset_dist_q = np.sqrt((xx.ravel() - rx_x)**2 + (yy.ravel() - rx_y)**2)
        offset_dist_q = np.maximum(offset_dist_q, 1e-6)
        return offset_dist_q, (ww_x * ww_y).ravel()


# ============================================================================
# Kernel dispatchers
# ============================================================================

def _run_circular(times, thicknesses, resistivities,
                  tx_radius, extra_weights,
                  h_base, h_j1, f_base, f_sin, f_cos, e_eta, e_A,
                  filter_wt, system_filter, signal, transform,
                  use_numba, use_cuda, hankel_filter, fourier_filter):
    """
    Dispatch the circular-loop kernel to the best available backend.

    Returns dbdt before _apply_signal_scaling and rx_fac scaling.
    Backends tried in order: CUDA > Numba > pure Python.
    The pure Python path also handles the impulse response (signal=0).
    """
    a = float(tx_radius)

    # PATH 1: CUDA GPU
    if HAS_CUDA and use_cuda and signal in (-1, 1):
        d_h_base, d_h_j0, d_h_j1, d_f_base, d_f_sin, d_f_cos = \
            _resolve_gpu_filters(hankel_filter, fourier_filter, transform)
        d_extra = cp.asarray(extra_weights)
        if transform == 'euler':
            return _tem_circular_euler_gpu(
                times, thicknesses, resistivities, a,
                d_extra, d_h_base, d_h_j1,
                e_eta, e_A, filter_weights=filter_wt)
        else:
            return _tem_circular_gpu(
                times, thicknesses, resistivities, a,
                d_extra, d_h_base, d_h_j1, d_f_base, d_f_sin,
                filter_weights=filter_wt)

    # PATH 2: Numba JIT
    if HAS_NUMBA and use_numba and signal in (-1, 1):
        n_eval = len(f_base) if transform == 'dlf' else len(e_eta)
        fw = filter_wt if filter_wt is not None else \
            np.ones((len(times), n_eval), dtype=np.complex128)
        if transform == 'euler':
            return _tem_circular_euler_jit(
                times, thicknesses, resistivities,
                a, extra_weights, MU0, h_base, h_j1, e_eta, e_A, fw)
        else:
            return _tem_circular_jit(
                times, thicknesses, resistivities,
                a, extra_weights, MU0, h_base, h_j1, f_base, f_sin, fw)

    # PATH 3: Pure Python (also handles impulse, signal=0)
    lam = h_base / a

    def _hz_sec(omega):
        r_te = te_reflection_coeff(lam, omega, thicknesses, resistivities)
        hz = 0.5 * np.dot(r_te * lam * extra_weights, h_j1)
        if system_filter is not None:
            hz *= system_filter(omega)
        return hz

    dbdt = np.zeros(len(times))
    if transform == 'euler':
        n_eval = len(e_eta)
        ks = np.arange(n_eval, dtype=float)
        for i, t in enumerate(times):
            c = e_A / (2.0 * t)
            vals = np.array([np.real(MU0 * _hz_sec((c + k * np.pi / t * 1j) / 1j))
                             for k in range(n_eval)])
            dbdt[i] = np.exp(e_A / 2.0) / t * np.dot(e_eta * (-1.0)**ks, vals)
    else:
        for i, t in enumerate(times):
            omega_pts = f_base / t
            if signal in (-1, 1):
                spectral = np.array([MU0 * np.imag(_hz_sec(w)) for w in omega_pts])
                dbdt[i] = np.dot(spectral, f_sin) / t
            else:  # impulse (signal=0)
                spectral = np.array([np.real(-MU0 * 1j * w * _hz_sec(w))
                                     for w in omega_pts])
                dbdt[i] = np.dot(spectral, f_cos) / t
    return dbdt


def _run_square(times, thicknesses, resistivities,
                offset_dist_q, area_w,
                h_base, h_j0, f_base, f_sin, f_cos, e_eta, e_A,
                filter_wt, system_filter, signal, transform,
                use_numba, use_cuda, hankel_filter, fourier_filter):
    """
    Dispatch the square-loop (VMD area integral) kernel to the best backend.

    Returns dbdt before any quadrature scale factor, _apply_signal_scaling,
    and rx_fac. For fwd_square_central, multiply the result by 4 before
    calling _apply_signal_scaling to account for the one-quadrant integration.
    """
    # PATH 1: CUDA GPU
    if HAS_CUDA and use_cuda and signal in (-1, 1):
        d_h_base, d_h_j0, d_h_j1, d_f_base, d_f_sin, d_f_cos = \
            _resolve_gpu_filters(hankel_filter, fourier_filter, transform)
        if transform == 'euler':
            return _tem_square_euler_gpu(
                times, thicknesses, resistivities,
                offset_dist_q, area_w, d_h_base, d_h_j0,
                e_eta, e_A, filter_weights=filter_wt)
        else:
            return _tem_square_gpu(
                times, thicknesses, resistivities,
                offset_dist_q, area_w, d_h_base, d_h_j0,
                d_f_base, d_f_sin, filter_weights=filter_wt)

    # PATH 2: Numba JIT
    if HAS_NUMBA and use_numba and signal in (-1, 1):
        n_eval = len(f_base) if transform == 'dlf' else len(e_eta)
        fw = filter_wt if filter_wt is not None else \
            np.ones((len(times), n_eval), dtype=np.complex128)
        if transform == 'euler':
            return _tem_square_euler_jit(
                times, thicknesses, resistivities,
                offset_dist_q, area_w, MU0,
                h_base, h_j0, e_eta, e_A, fw)
        else:
            return _tem_square_jit(
                times, thicknesses, resistivities,
                offset_dist_q, area_w, MU0,
                h_base, h_j0, f_base, f_sin, fw)

    # PATH 3: Pure Python (also handles impulse, signal=0)
    def _hz_sec(omega):
        total = 0j
        for q in range(len(offset_dist_q)):
            total += area_w[q] * _vmd_greenfct(
                offset_dist_q[q], omega, thicknesses, resistivities, h_base, h_j0)
        if system_filter is not None:
            total *= system_filter(omega)
        return total

    dbdt = np.zeros(len(times))
    if transform == 'euler':
        n_eval = len(e_eta)
        ks = np.arange(n_eval, dtype=float)
        for i, t in enumerate(times):
            c = e_A / (2.0 * t)
            vals = np.array([np.real(MU0 * _hz_sec((c + k * np.pi / t * 1j) / 1j))
                             for k in range(n_eval)])
            dbdt[i] = np.exp(e_A / 2.0) / t * np.dot(e_eta * (-1.0)**ks, vals)
    else:
        for i, t in enumerate(times):
            omega_pts = f_base / t
            if signal in (-1, 1):
                spectral = np.array([MU0 * np.imag(_hz_sec(w)) for w in omega_pts])
                dbdt[i] = np.dot(spectral, f_sin) / t
            else:  # impulse (signal=0)
                spectral = np.array([np.real(-MU0 * 1j * w * _hz_sec(w))
                                     for w in omega_pts])
                dbdt[i] = np.dot(spectral, f_cos) / t
    return dbdt


# ============================================================================
# Public forward functions
# ============================================================================

def fwd_circle_central(thicknesses, resistivities, tx_radius, times,
                       current=1.0, signal=-1,
                       system_filter=None, use_numba=True, use_cuda=True,
                       rx_area=1.0, rx_turns=1,
                       hankel_filter='key_201', fourier_filter='key_101',
                       transform='dlf', euler_order=11):
    """
    Central-loop TEM forward response for a 1-D layered earth.

    Computes dBz/dt [V/m^2] at the centre of a circular Tx loop.
    Backends tried in order: CUDA > Numba > pure Python.
    The pure Python path also supports the impulse response (signal=0).

    Parameters
    ----------
    thicknesses   : array-like, shape (N-1,)  layer thicknesses [m]
    resistivities : array-like, shape (N,)    layer resistivities [Ohm.m]
    tx_radius     : float                     Tx loop radius [m]
    times         : array-like, shape (n_t,)  gate times [s]
    current       : float, default 1.0        Tx current [A]
    signal        : int, default -1           -1=step-off, +1=step-on, 0=impulse
    system_filter : callable or None          H(omega) -> complex
    use_numba     : bool, default True
    use_cuda      : bool, default True
    rx_area       : float, default 1.0        Rx area [m^2]
    rx_turns      : int, default 1            Rx turns
    hankel_filter : str, default 'key_201'
    fourier_filter: str, default 'key_81'
    transform     : str, default 'dlf'        'dlf' or 'euler'
    euler_order   : int, default 11

    Returns
    -------
    dbdt : ndarray, shape (n_t,)  dBz/dt [V/m^2]
    """
    thicknesses = np.asarray(thicknesses, dtype=float)
    resistivities = np.asarray(resistivities, dtype=float)
    times = np.asarray(times, dtype=float)
    transform = transform.lower()
    rx_fac = _rx_scale(rx_area, rx_turns)

    h_base, h_j0, h_j1, f_base, f_sin, f_cos, e_eta, e_A = \
        _resolve_filters(hankel_filter, fourier_filter, transform, euler_order)

    if system_filter is not None:
        filter_wt = _precompute_filter_dlf(system_filter, times, f_base) \
            if transform == 'dlf' else \
            _precompute_filter_euler(system_filter, times, e_eta, e_A)
    else:
        filter_wt = None

    extra_weights = _build_circular_geometry(tx_radius, h_base)
    dbdt = _run_circular(times, thicknesses, resistivities,
                         tx_radius, extra_weights,
                         h_base, h_j1, f_base, f_sin, f_cos, e_eta, e_A,
                         filter_wt, system_filter, signal, transform,
                         use_numba, use_cuda, hankel_filter, fourier_filter)
    _apply_signal_scaling(dbdt, current, signal, transform)
    return dbdt * rx_fac


def fwd_circle_offset(thicknesses, resistivities, tx_radius, rx_offset,
                      times, current=1.0, signal=-1,
                      system_filter=None, use_numba=True, use_cuda=True,
                      rx_area=1.0, rx_turns=1,
                      hankel_filter='key_201', fourier_filter='key_101',
                      transform='dlf', euler_order=11):
    """
    Offset-loop TEM forward response for a 1-D layered earth.

    Computes dBz/dt [V/m^2] at a receiver displaced by rx_offset [m]
    from the centre of a circular Tx loop.

    Parameters
    ----------
    thicknesses   : array-like, shape (N-1,)
    resistivities : array-like, shape (N,)
    tx_radius     : float   Tx loop radius [m]
    rx_offset     : float   Radial Tx-centre to Rx distance [m]
    times         : array-like, shape (n_t,)
    current, signal, system_filter, use_numba, use_cuda,
    rx_area, rx_turns, hankel_filter, fourier_filter,
    transform, euler_order : see fwd_circle_central

    Returns
    -------
    dbdt : ndarray, shape (n_t,)
    """
    thicknesses = np.asarray(thicknesses, dtype=float)
    resistivities = np.asarray(resistivities, dtype=float)
    times = np.asarray(times, dtype=float)
    transform = transform.lower()
    rx_fac = _rx_scale(rx_area, rx_turns)

    h_base, h_j0, h_j1, f_base, f_sin, f_cos, e_eta, e_A = \
        _resolve_filters(hankel_filter, fourier_filter, transform, euler_order)

    if system_filter is not None:
        filter_wt = _precompute_filter_dlf(system_filter, times, f_base) \
            if transform == 'dlf' else \
            _precompute_filter_euler(system_filter, times, e_eta, e_A)
    else:
        filter_wt = None

    extra_weights = _build_circular_geometry(tx_radius, h_base,
                                             rx_offset=float(rx_offset))
    dbdt = _run_circular(times, thicknesses, resistivities,
                         tx_radius, extra_weights,
                         h_base, h_j1, f_base, f_sin, f_cos, e_eta, e_A,
                         filter_wt, system_filter, signal, transform,
                         use_numba, use_cuda, hankel_filter, fourier_filter)
    _apply_signal_scaling(dbdt, current, signal, transform)
    return dbdt * rx_fac


def fwd_square_central(thicknesses, resistivities, side_length, times,
                       current=1.0, signal=-1,
                       system_filter=None, n_quad=5, use_numba=True,
                       use_cuda=True, use_symmetry=True,
                       rx_area=1.0, rx_turns=1,
                       hankel_filter='key_201', fourier_filter='key_101',
                       transform='dlf', euler_order=11):
    """
    Central square-loop TEM forward response for a 1-D layered earth.

    Models the square Tx loop as an area distribution of VMDs integrated by
    2-D Gauss-Legendre quadrature over one quadrant (exploiting x<->y symmetry).

    Parameters
    ----------
    thicknesses, resistivities, times, current, signal,
    system_filter, use_numba, use_cuda, rx_area, rx_turns,
    hankel_filter, fourier_filter, transform, euler_order : see fwd_circle_central
    side_length   : float  square Tx side length [m]
    n_quad        : int    GL quadrature points per axis (default 5)
    use_symmetry  : bool   exploit x<->y symmetry within quadrant (default True)

    Returns
    -------
    dbdt : ndarray, shape (n_t,)
    """
    thicknesses = np.asarray(thicknesses, dtype=float)
    resistivities = np.asarray(resistivities, dtype=float)
    times = np.asarray(times, dtype=float)
    transform = transform.lower()
    rx_fac = _rx_scale(rx_area, rx_turns)

    h_base, h_j0, h_j1, f_base, f_sin, f_cos, e_eta, e_A = \
        _resolve_filters(hankel_filter, fourier_filter, transform, euler_order)

    if system_filter is not None:
        filter_wt = _precompute_filter_dlf(system_filter, times, f_base) \
            if transform == 'dlf' else \
            _precompute_filter_euler(system_filter, times, e_eta, e_A)
    else:
        filter_wt = None

    offset_dist_q, area_w = _build_square_geometry(
        side_length, n_quad, use_symmetry=use_symmetry)
    dbdt = _run_square(times, thicknesses, resistivities,
                       offset_dist_q, area_w,
                       h_base, h_j0, f_base, f_sin, f_cos, e_eta, e_A,
                       filter_wt, system_filter, signal, transform,
                       use_numba, use_cuda, hankel_filter, fourier_filter)

    # One-quadrant integration: multiply by 4 to recover the full loop response.
    dbdt *= 4.0
    _apply_signal_scaling(dbdt, current, signal, transform)
    return dbdt * rx_fac


def fwd_square_offset(thicknesses, resistivities, side_length,
                      rx_x, rx_y, times,
                      current=1.0, signal=-1,
                      system_filter=None, n_quad=11, use_numba=True,
                      use_cuda=True, rx_area=1.0, rx_turns=1,
                      hankel_filter='key_201', fourier_filter='key_101',
                      transform='dlf', euler_order=11):
    """
    Square-loop TEM response at an arbitrary receiver position (rx_x, rx_y).

    Integrates over the full square [-L/2, L/2]^2 without symmetry reduction.

    Parameters
    ----------
    thicknesses, resistivities, times, current, signal,
    system_filter, use_numba, use_cuda, rx_area, rx_turns,
    hankel_filter, fourier_filter, transform, euler_order : see fwd_circle_central
    side_length : float  square Tx side length [m]
    rx_x        : float  Rx x-coordinate [m]
    rx_y        : float  Rx y-coordinate [m]
    n_quad      : int    GL quadrature points per axis (default 11)

    Returns
    -------
    dbdt : ndarray, shape (n_t,)
    """
    thicknesses = np.asarray(thicknesses, dtype=float)
    resistivities = np.asarray(resistivities, dtype=float)
    times = np.asarray(times, dtype=float)
    transform = transform.lower()
    rx_fac = _rx_scale(rx_area, rx_turns)

    h_base, h_j0, h_j1, f_base, f_sin, f_cos, e_eta, e_A = \
        _resolve_filters(hankel_filter, fourier_filter, transform, euler_order)

    if system_filter is not None:
        filter_wt = _precompute_filter_dlf(system_filter, times, f_base) \
            if transform == 'dlf' else \
            _precompute_filter_euler(system_filter, times, e_eta, e_A)
    else:
        filter_wt = None

    offset_dist_q, area_w = _build_square_geometry(
        side_length, n_quad, rx_x=float(rx_x), rx_y=float(rx_y),
        use_symmetry=False)
    dbdt = _run_square(times, thicknesses, resistivities,
                       offset_dist_q, area_w,
                       h_base, h_j0, f_base, f_sin, f_cos, e_eta, e_A,
                       filter_wt, system_filter, signal, transform,
                       use_numba, use_cuda, hankel_filter, fourier_filter)
    _apply_signal_scaling(dbdt, current, signal, transform)
    return dbdt * rx_fac


# ============================================================================
# Analytical half-space solutions
# ============================================================================

def fwd_analytical_central(resistivity, tx_radius, times, current=1.0):
    """Analytical dBz/dt at centre of a loop on a homogeneous half-space.
    Ward & Hohmann (1988), eq 4.69a.  Returns V/m^2."""
    times = np.asarray(times, dtype=float)
    sigma = 1.0 / resistivity
    a = float(tx_radius)

    theta = np.sqrt(MU0 * sigma / (4.0 * times))
    theta_a = theta * a
    dbdt = (-current / (sigma * a**3)) * (
        3.0 * erf(theta_a)
        - (2.0 / np.sqrt(np.pi)) * theta_a * (3.0 + 2.0 * theta_a**2)
        * np.exp(-theta_a**2)
    )
    return dbdt


def fwd_analytical_offset(resistivity, tx_radius, rx_offset, times,
                                   current=1.0):
    """Analytical dBz/dt at radial offset from a VMD on a homogeneous half-space.
    Ward & Hohmann (1988), eq 4.97 differentiated w.r.t. time.
    Valid in the far field (rx_offset >> tx_radius).  Returns V/m^2."""
    times = np.asarray(times, dtype=float)
    sigma = 1.0 / resistivity
    r = float(rx_offset)
    M = current * np.pi * float(tx_radius)**2

    theta = np.sqrt(MU0 * sigma / (4.0 * times))
    u = theta * r
    dbdt = (MU0 * M / (8.0 * np.pi * r**3 * times)) * (
        (9.0 / u**2) * erf(u)
        - (1.0 / np.sqrt(np.pi)) * (18.0 / u + 12.0 * u + 8.0 * u**3)
        * np.exp(-u**2)
    )
    return dbdt
