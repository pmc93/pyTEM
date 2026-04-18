"""
forward.py — TEM forward modelling functions and analytical solutions.

Contains:
  Helpers:
    _rx_scale, _resolve_filters, _resolve_gpu_filters,
    _apply_signal_scaling, _precompute_filter_dlf, _precompute_filter_euler,
    _vmd_greenfct

  Forward functions:
    tem_forward_circle        — central-loop circular Tx
    tem_forward_circle_offset — offset-loop circular Tx
    tem_forward_square       — central square-loop Tx
    tem_forward_square_offset — offset square-loop Tx

  Analytical solutions:
    halfspace_dbdt_analytic         — Ward & Hohmann eq 4.69a
    halfspace_dbdt_offset_analytic  — Ward & Hohmann eq 4.97
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


def _apply_signal_scaling(dbdt, current, signal, transform, quadrant_factor=1.0):
    """Apply current amplitude, signal type, and optional quadrant scaling."""
    if transform == 'euler':
        dbdt *= current * float(signal) * quadrant_factor
    else:  # dlf
        if signal == 1:
            dbdt *= -current * 2.0 / np.pi * quadrant_factor
        else:
            dbdt *= current * 2.0 / np.pi * quadrant_factor
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


def _vmd_greenfct(rho, omega, thicknesses, resistivities, h_base, h_j0):
    """VMD Green's function G(rho, omega) via J0 Hankel DLF."""
    lam = h_base / rho
    r_te = te_reflection_coeff(lam, omega, thicknesses, resistivities)
    integrand = r_te * lam**2
    hankel_val = np.dot(integrand, h_j0) / rho
    return hankel_val / (4.0 * np.pi)


# ============================================================================
# tem_forward_circle — central-loop circular Tx, Rx at loop centre
# ============================================================================

def tem_forward_circle(thicknesses, resistivities, tx_radius, times,
                current=1.0, signal=-1,
                system_filter=None, use_numba=True, use_cuda=True,
                rx_area=1.0, rx_turns=1,
                hankel_filter='key_201', fourier_filter='key_81',
                transform='dlf', euler_order=11):
    """
    Central-loop TEM forward response for a 1-D layered earth.

    Computes dBz/dt [V/m^2] at the centre of a circular Tx loop.
    Three execution paths are tried in priority order:
      1. CUDA/CuPy GPU  (fastest)
      2. Numba JIT       (fast)
      3. Pure Python      (supports impulse response)

    Parameters
    ----------
    thicknesses   : array-like, shape (N-1,)
    resistivities : array-like, shape (N,)
    tx_radius     : float  Tx loop radius [m]
    times         : array-like, shape (n_t,)
    current       : float, default 1.0
    signal        : int, default -1  (-1=step-off, +1=step-on, 0=impulse)
    system_filter : callable or None — H(omega) -> complex
    use_numba     : bool, default True
    use_cuda      : bool, default True
    rx_area       : float, default 1.0
    rx_turns      : int, default 1
    hankel_filter : str, default 'key_201'
    fourier_filter: str, default 'key_81'
    transform     : str, default 'dlf'
    euler_order   : int, default 11

    Returns
    -------
    dbdt : ndarray, shape (n_t,)
    """
    thicknesses = np.asarray(thicknesses, dtype=float)
    resistivities = np.asarray(resistivities, dtype=float)
    times = np.asarray(times, dtype=float)
    a = float(tx_radius)
    rx_fac = _rx_scale(rx_area, rx_turns)

    h_base, h_j0, h_j1, f_base, f_sin, f_cos, e_eta, e_A = \
        _resolve_filters(hankel_filter, fourier_filter, transform, euler_order)

    n_lam = len(h_base)

    # Pre-compute system filter at all evaluation frequencies
    if system_filter is not None:
        if transform == 'dlf':
            filter_wt = _precompute_filter_dlf(system_filter, times, f_base)
        else:
            filter_wt = _precompute_filter_euler(system_filter, times, e_eta, e_A)
    else:
        filter_wt = None

    # ---- PATH 1: CUDA GPU ----
    if HAS_CUDA and use_cuda and signal in (-1, 1):
        d_h_base, d_h_j0, d_h_j1, d_f_base, d_f_sin, d_f_cos = \
            _resolve_gpu_filters(hankel_filter, fourier_filter, transform)
        d_ones = cp.ones(len(d_h_base), dtype=cp.float64)
        if transform == 'euler':
            dbdt = _tem_circular_euler_gpu(
                times, thicknesses, resistivities, a,
                d_ones, d_h_base, d_h_j1,
                e_eta, e_A, filter_weights=filter_wt)
        else:
            dbdt = _tem_circular_gpu(
                times, thicknesses, resistivities, a,
                d_ones, d_h_base, d_h_j1, d_f_base, d_f_sin,
                filter_weights=filter_wt)
        _apply_signal_scaling(dbdt, current, signal, transform)
        dbdt *= rx_fac
        return dbdt

    # ---- PATH 2: Numba JIT ----
    if HAS_NUMBA and use_numba and signal in (-1, 1):
        extra_w = np.ones(n_lam, dtype=np.float64)
        if filter_wt is None:
            n_eval = len(f_base) if transform == 'dlf' else len(e_eta)
            fw = np.ones((len(times), n_eval), dtype=np.complex128)
        else:
            fw = filter_wt
        if transform == 'euler':
            dbdt = _tem_circular_euler_jit(
                times, thicknesses, resistivities,
                a, extra_w, MU0, h_base, h_j1, e_eta, e_A, fw)
        else:
            dbdt = _tem_circular_jit(
                times, thicknesses, resistivities,
                a, extra_w, MU0, h_base, h_j1, f_base, f_sin, fw)
        _apply_signal_scaling(dbdt, current, signal, transform)
        dbdt *= rx_fac
        return dbdt

    # ---- PATH 3: Pure Python (supports impulse) ----
    def _hz_sec(omega):
        lam = h_base / a
        r_te = te_reflection_coeff(lam, omega, thicknesses, resistivities)
        hankel = np.dot(r_te * lam, h_j1) / a
        hz = 0.5 * a * hankel
        if system_filter is not None:
            hz *= system_filter(omega)
        return hz

    dbdt = np.zeros(len(times))

    if transform == 'euler':
        n_eval = len(e_eta)
        for i, t in enumerate(times):
            c = e_A / (2.0 * t)
            h_step = np.pi / t
            vals = np.array([np.real(MU0 * _hz_sec((c + k * h_step * 1j) / 1j))
                             for k in range(n_eval)])
            signs = np.array([(-1.0)**k for k in range(n_eval)])
            dbdt[i] = np.exp(e_A / 2.0) / t * np.dot(e_eta * signs, vals)
        _apply_signal_scaling(dbdt, current, signal, transform)
    else:
        for i, t in enumerate(times):
            omega_pts = f_base / t
            if signal in (-1, 1):
                sig = np.array([MU0 * np.imag(_hz_sec(w)) for w in omega_pts])
                dbdt[i] = np.dot(sig, f_sin) / t
            else:  # impulse (signal == 0)
                sig = np.array([np.real(-MU0 * 1j * w * _hz_sec(w))
                                for w in omega_pts])
                dbdt[i] = np.dot(sig, f_cos) / t
        _apply_signal_scaling(dbdt, current, signal, transform)

    dbdt *= rx_fac
    return dbdt


# ============================================================================
# tem_forward_circle_offset — circular Tx loop, Rx at radial offset
# ============================================================================

def tem_forward_circle_offset(thicknesses, resistivities, tx_radius, rx_offset,
                       times, current=1.0, signal=-1,
                       system_filter=None, use_numba=True, use_cuda=True,
                       rx_area=1.0, rx_turns=1,
                       hankel_filter='key_201', fourier_filter='key_81',
                       transform='dlf', euler_order=11):
    """
    Offset-loop TEM forward response for a 1-D layered earth.

    Computes dBz/dt [V/m^2] at a receiver displaced radial distance
    `rx_offset` from the centre of a circular Tx loop.

    Parameters
    ----------
    thicknesses   : array-like, shape (N-1,)
    resistivities : array-like, shape (N,)
    tx_radius     : float
    rx_offset     : float  Radial distance from Tx centre to Rx [m]
    times         : array-like, shape (n_t,)
    current       : float, default 1.0
    signal        : int, default -1
    system_filter : callable or None
    use_numba     : bool, default True
    use_cuda      : bool, default True
    rx_area       : float, default 1.0
    rx_turns      : int, default 1
    hankel_filter : str, default 'key_201'
    fourier_filter: str, default 'key_81'
    transform     : str, default 'dlf'
    euler_order   : int, default 11

    Returns
    -------
    dbdt : ndarray, shape (n_t,)
    """
    thicknesses = np.asarray(thicknesses, dtype=float)
    resistivities = np.asarray(resistivities, dtype=float)
    times = np.asarray(times, dtype=float)
    a = float(tx_radius)
    rho = float(rx_offset)
    rx_fac = _rx_scale(rx_area, rx_turns)

    h_base, h_j0, h_j1, f_base, f_sin, f_cos, e_eta, e_A = \
        _resolve_filters(hankel_filter, fourier_filter, transform, euler_order)

    # Offset geometry: extra_weights = J0(lam_i * rho)
    lam = h_base / a
    j0_vals = j0(lam * rho)

    # Pre-compute system filter
    if system_filter is not None:
        if transform == 'dlf':
            filter_wt = _precompute_filter_dlf(system_filter, times, f_base)
        else:
            filter_wt = _precompute_filter_euler(system_filter, times, e_eta, e_A)
    else:
        filter_wt = None

    # ---- PATH 1: CUDA GPU ----
    if HAS_CUDA and use_cuda and signal in (-1, 1):
        d_h_base, d_h_j0, d_h_j1, d_f_base, d_f_sin, d_f_cos = \
            _resolve_gpu_filters(hankel_filter, fourier_filter, transform)
        d_j0 = cp.asarray(j0_vals)
        if transform == 'euler':
            dbdt = _tem_circular_euler_gpu(
                times, thicknesses, resistivities, a,
                d_j0, d_h_base, d_h_j1,
                e_eta, e_A, filter_weights=filter_wt)
        else:
            dbdt = _tem_circular_gpu(
                times, thicknesses, resistivities, a,
                d_j0, d_h_base, d_h_j1, d_f_base, d_f_sin,
                filter_weights=filter_wt)
        _apply_signal_scaling(dbdt, current, signal, transform)
        dbdt *= rx_fac
        return dbdt

    # ---- PATH 2: Numba JIT ----
    if HAS_NUMBA and use_numba and signal in (-1, 1):
        if filter_wt is None:
            n_eval = len(f_base) if transform == 'dlf' else len(e_eta)
            fw = np.ones((len(times), n_eval), dtype=np.complex128)
        else:
            fw = filter_wt
        if transform == 'euler':
            dbdt = _tem_circular_euler_jit(
                times, thicknesses, resistivities,
                a, j0_vals, MU0, h_base, h_j1, e_eta, e_A, fw)
        else:
            dbdt = _tem_circular_jit(
                times, thicknesses, resistivities,
                a, j0_vals, MU0, h_base, h_j1, f_base, f_sin, fw)
        _apply_signal_scaling(dbdt, current, signal, transform)
        dbdt *= rx_fac
        return dbdt

    # ---- PATH 3: Pure Python ----
    def _hz_sec(omega):
        lam_loc = h_base / a
        r_te = te_reflection_coeff(lam_loc, omega, thicknesses, resistivities)
        kernel = r_te * lam_loc * j0(lam_loc * rho)
        hankel = np.dot(kernel, h_j1) / a
        hz = 0.5 * a * hankel
        if system_filter is not None:
            hz *= system_filter(omega)
        return hz

    dbdt = np.zeros(len(times))

    if transform == 'euler':
        n_eval = len(e_eta)
        for i, t in enumerate(times):
            c = e_A / (2.0 * t)
            h_step = np.pi / t
            vals = np.array([np.real(MU0 * _hz_sec((c + k * h_step * 1j) / 1j))
                             for k in range(n_eval)])
            signs = np.array([(-1.0)**k for k in range(n_eval)])
            dbdt[i] = np.exp(e_A / 2.0) / t * np.dot(e_eta * signs, vals)
        _apply_signal_scaling(dbdt, current, signal, transform)
    else:
        for i, t in enumerate(times):
            omega_pts = f_base / t
            if signal in (-1, 1):
                sig = np.array([MU0 * np.imag(_hz_sec(w)) for w in omega_pts])
                dbdt[i] = np.dot(sig, f_sin) / t
            else:
                sig = np.array([np.real(-MU0 * 1j * w * _hz_sec(w))
                                for w in omega_pts])
                dbdt[i] = np.dot(sig, f_cos) / t
        _apply_signal_scaling(dbdt, current, signal, transform)

    dbdt *= rx_fac
    return dbdt


# ============================================================================
# tem_forward_square — central Rx at centre of square Tx loop
# ============================================================================

def tem_forward_square(thicknesses, resistivities, side_length, times,
                       current=1.0, signal=-1,
                       system_filter=None, n_quad=5, use_numba=True,
                       use_cuda=True, use_symmetry=True,
                       rx_area=1.0, rx_turns=1,
                       hankel_filter='key_201', fourier_filter='key_81',
                       transform='dlf', euler_order=11):
    """
    Central square-loop TEM forward response for a 1-D layered earth.

    Computes dBz/dt [V/m^2] at the centre of a square Tx loop.
    The loop is modelled as an area distribution of VMDs, with the area
    integral evaluated by 2D Gauss-Legendre quadrature.

    Parameters
    ----------
    thicknesses   : array-like, shape (N-1,)
    resistivities : array-like, shape (N,)
    side_length   : float
    times         : array-like, shape (n_t,)
    current       : float, default 1.0
    signal        : int, default -1
    system_filter : callable or None
    n_quad        : int, default 5
    use_numba     : bool, default True
    use_cuda      : bool, default True
    use_symmetry  : bool, default True
    rx_area       : float, default 1.0
    rx_turns      : int, default 1
    hankel_filter : str, default 'key_201'
    fourier_filter: str, default 'key_81'
    transform     : str, default 'dlf'
    euler_order   : int, default 11

    Returns
    -------
    dbdt : ndarray, shape (n_t,)
    """
    thicknesses = np.asarray(thicknesses, dtype=float)
    resistivities = np.asarray(resistivities, dtype=float)
    times = np.asarray(times, dtype=float)
    L = float(side_length)
    hs = L / 2.0
    rx_fac = _rx_scale(rx_area, rx_turns)
    transform = transform.lower()

    h_base, h_j0, h_j1, f_base, f_sin, f_cos, e_eta, e_A = \
        _resolve_filters(hankel_filter, fourier_filter, transform, euler_order)

    # --- 2D Gauss-Legendre quadrature over one quadrant [0, hs]x[0, hs] ---
    gl_nodes, gl_weights = np.polynomial.legendre.leggauss(n_quad)
    x_pts = hs / 2.0 * (1.0 + gl_nodes)
    w_pts = gl_weights * hs / 2.0

    if use_symmetry:
        rho_q = []
        area_w = []
        for i in range(n_quad):
            xi = x_pts[i]
            wi = w_pts[i]
            for jj in range(i, n_quad):
                yj = x_pts[jj]
                wj = w_pts[jj]
                w = wi * wj
                if i != jj:
                    w *= 2.0
                rho_q.append(np.sqrt(xi * xi + yj * yj))
                area_w.append(w)
        rho_q = np.asarray(rho_q, dtype=float)
        area_w = np.asarray(area_w, dtype=float)
    else:
        xx, yy = np.meshgrid(x_pts, x_pts)
        ww_x, ww_y = np.meshgrid(w_pts, w_pts)
        rho_q = np.sqrt(xx.ravel()**2 + yy.ravel()**2)
        area_w = (ww_x * ww_y).ravel()

    # Pre-compute system filter
    if system_filter is not None:
        if transform == 'dlf':
            filter_wt = _precompute_filter_dlf(system_filter, times, f_base)
        else:
            filter_wt = _precompute_filter_euler(system_filter, times, e_eta, e_A)
    else:
        filter_wt = None

    # ---- PATH 1: CUDA GPU ----
    if HAS_CUDA and use_cuda and signal in (-1, 1):
        d_h_base, d_h_j0, d_h_j1, d_f_base, d_f_sin, d_f_cos = \
            _resolve_gpu_filters(hankel_filter, fourier_filter, transform)
        if transform == 'euler':
            dbdt = _tem_square_euler_gpu(
                times, thicknesses, resistivities,
                rho_q, area_w, d_h_base, d_h_j0,
                e_eta, e_A, filter_weights=filter_wt)
        else:
            dbdt = _tem_square_gpu(
                times, thicknesses, resistivities,
                rho_q, area_w, d_h_base, d_h_j0,
                d_f_base, d_f_sin, filter_weights=filter_wt)
        _apply_signal_scaling(dbdt, current, signal, transform)
        dbdt *= rx_fac
        return dbdt

    # ---- PATH 2: Numba JIT ----
    if HAS_NUMBA and use_numba and signal in (-1, 1):
        if filter_wt is None:
            n_eval = len(f_base) if transform == 'dlf' else len(e_eta)
            fw = np.ones((len(times), n_eval), dtype=np.complex128)
        else:
            fw = filter_wt
        if transform == 'euler':
            dbdt = _tem_square_euler_jit(
                times, thicknesses, resistivities,
                rho_q, area_w, MU0,
                h_base, h_j0, e_eta, e_A, fw)
        else:
            dbdt = _tem_square_jit(
                times, thicknesses, resistivities,
                rho_q, area_w, MU0,
                h_base, h_j0, f_base, f_sin, fw)
        _apply_signal_scaling(dbdt, current, signal, transform)
        dbdt *= rx_fac
        return dbdt

    # ---- PATH 3: Pure Python ----
    def _hz_sec(omega):
        total = 0j
        for q in range(len(rho_q)):
            g = _vmd_greenfct(rho_q[q], omega, thicknesses, resistivities,
                              h_base, h_j0)
            total += area_w[q] * g
        hz = 4.0 * total
        if system_filter is not None:
            hz *= system_filter(omega)
        return hz

    dbdt = np.zeros(len(times))

    if transform == 'euler':
        n_eval = len(e_eta)
        for i, t in enumerate(times):
            c = e_A / (2.0 * t)
            h_step = np.pi / t
            vals = np.array([np.real(MU0 * _hz_sec((c + k * h_step * 1j) / 1j))
                             for k in range(n_eval)])
            signs = np.array([(-1.0)**k for k in range(n_eval)])
            dbdt[i] = np.exp(e_A / 2.0) / t * np.dot(e_eta * signs, vals)
        _apply_signal_scaling(dbdt, current, signal, transform)
    else:
        for i, t in enumerate(times):
            omega_pts = f_base / t
            if signal in (-1, 1):
                sig = np.array([MU0 * np.imag(_hz_sec(w)) for w in omega_pts])
                dbdt[i] = np.dot(sig, f_sin) / t
            else:
                sig = np.array([np.real(-MU0 * 1j * w * _hz_sec(w))
                                for w in omega_pts])
                dbdt[i] = np.dot(sig, f_cos) / t
        _apply_signal_scaling(dbdt, current, signal, transform)

    dbdt *= rx_fac
    return dbdt


# ============================================================================
# tem_forward_square_offset — Rx at arbitrary (rx_x, rx_y)
# ============================================================================

def tem_forward_square_offset(thicknesses, resistivities, side_length,
                              rx_x, rx_y, times,
                              current=1.0, signal=-1,
                              system_filter=None, n_quad=11, use_numba=True,
                              use_cuda=True, rx_area=1.0, rx_turns=1,
                              hankel_filter='key_201', fourier_filter='key_81',
                              transform='dlf', euler_order=11):
    """
    Square-loop TEM response at an arbitrary receiver position (rx_x, rx_y).

    Integrates over the full square [-L/2, L/2]^2 (no symmetry reduction).

    Parameters
    ----------
    thicknesses   : array-like, shape (N-1,)
    resistivities : array-like, shape (N,)
    side_length   : float
    rx_x          : float
    rx_y          : float
    times         : array-like, shape (n_t,)
    current       : float, default 1.0
    signal        : int, default -1
    system_filter : callable or None
    n_quad        : int, default 11
    use_numba     : bool, default True
    use_cuda      : bool, default True
    rx_area       : float, default 1.0
    rx_turns      : int, default 1
    hankel_filter : str, default 'key_201'
    fourier_filter: str, default 'key_81'
    transform     : str, default 'dlf'
    euler_order   : int, default 11

    Returns
    -------
    dbdt : ndarray, shape (n_t,)
    """
    thicknesses = np.asarray(thicknesses, dtype=float)
    resistivities = np.asarray(resistivities, dtype=float)
    times = np.asarray(times, dtype=float)
    L = float(side_length)
    hs = L / 2.0
    rx_fac = _rx_scale(rx_area, rx_turns)
    transform = transform.lower()

    h_base, h_j0, h_j1, f_base, f_sin, f_cos, e_eta, e_A = \
        _resolve_filters(hankel_filter, fourier_filter, transform, euler_order)

    # --- Full-square GL quadrature: [-hs, hs] x [-hs, hs] ---
    gl_nodes, gl_weights = np.polynomial.legendre.leggauss(n_quad)
    x_pts = hs * gl_nodes
    y_pts = hs * gl_nodes
    wx = hs * gl_weights
    wy = hs * gl_weights

    xx, yy = np.meshgrid(x_pts, y_pts, indexing="xy")
    ww_x, ww_y = np.meshgrid(wx, wy, indexing="xy")
    area_w = (ww_x * ww_y).ravel()

    rho_q = np.sqrt((xx.ravel() - rx_x)**2 + (yy.ravel() - rx_y)**2)
    rho_q = np.maximum(rho_q, 1e-6)

    # Pre-compute system filter
    if system_filter is not None:
        if transform == 'dlf':
            filter_wt = _precompute_filter_dlf(system_filter, times, f_base)
        else:
            filter_wt = _precompute_filter_euler(system_filter, times, e_eta, e_A)
    else:
        filter_wt = None

    # ---- PATH 1: CUDA GPU ----
    if HAS_CUDA and use_cuda and signal in (-1, 1):
        d_h_base, d_h_j0, d_h_j1, d_f_base, d_f_sin, d_f_cos = \
            _resolve_gpu_filters(hankel_filter, fourier_filter, transform)
        if transform == 'euler':
            dbdt = _tem_square_euler_gpu(
                times, thicknesses, resistivities,
                rho_q, area_w, d_h_base, d_h_j0,
                e_eta, e_A, filter_weights=filter_wt)
        else:
            dbdt = _tem_square_gpu(
                times, thicknesses, resistivities,
                rho_q, area_w, d_h_base, d_h_j0,
                d_f_base, d_f_sin, filter_weights=filter_wt)
        _apply_signal_scaling(dbdt, current, signal, transform,
                              quadrant_factor=0.25)
        dbdt *= rx_fac
        return dbdt

    # ---- PATH 2: Numba JIT ----
    if HAS_NUMBA and use_numba and signal in (-1, 1):
        if filter_wt is None:
            n_eval = len(f_base) if transform == 'dlf' else len(e_eta)
            fw = np.ones((len(times), n_eval), dtype=np.complex128)
        else:
            fw = filter_wt
        if transform == 'euler':
            dbdt = _tem_square_euler_jit(
                times, thicknesses, resistivities,
                rho_q, area_w, MU0,
                h_base, h_j0, e_eta, e_A, fw)
        else:
            dbdt = _tem_square_jit(
                times, thicknesses, resistivities,
                rho_q, area_w, MU0,
                h_base, h_j0, f_base, f_sin, fw)
        _apply_signal_scaling(dbdt, current, signal, transform,
                              quadrant_factor=0.25)
        dbdt *= rx_fac
        return dbdt

    # ---- PATH 3: Pure Python ----
    def _hz_sec(omega):
        total = 0j
        for q in range(len(rho_q)):
            total += area_w[q] * _vmd_greenfct(rho_q[q], omega, thicknesses,
                                                resistivities, h_base, h_j0)
        if system_filter is not None:
            total *= system_filter(omega)
        return total

    dbdt = np.zeros(len(times))

    if transform == 'euler':
        n_eval = len(e_eta)
        for i, t in enumerate(times):
            c = e_A / (2.0 * t)
            h_step = np.pi / t
            vals = np.array([np.real(MU0 * _hz_sec((c + k * h_step * 1j) / 1j))
                             for k in range(n_eval)])
            signs = np.array([(-1.0)**k for k in range(n_eval)])
            dbdt[i] = np.exp(e_A / 2.0) / t * np.dot(e_eta * signs, vals)
        _apply_signal_scaling(dbdt, current, signal, transform)
    else:
        for i, t in enumerate(times):
            omega_pts = f_base / t
            if signal in (-1, 1):
                sig = np.array([MU0 * np.imag(_hz_sec(w)) for w in omega_pts])
                dbdt[i] = np.dot(sig, f_sin) / t
            else:
                sig = np.array([np.real(-MU0 * 1j * w * _hz_sec(w))
                                for w in omega_pts])
                dbdt[i] = np.dot(sig, f_cos) / t
        _apply_signal_scaling(dbdt, current, signal, transform)

    return dbdt * rx_fac


# ============================================================================
# Analytical half-space solutions
# ============================================================================

def halfspace_dbdt_analytic(resistivity, tx_radius, times, current=1.0):
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


def halfspace_dbdt_offset_analytic(resistivity, tx_radius, rx_offset, times,
                                   current=1.0):
    """Analytical dBz/dt at offset r from a VMD on a homogeneous half-space.
    Ward & Hohmann (1988), eq 4.97 differentiated w.r.t. time.
    Valid in the far field (r >> a).  Returns V/m^2 (T/s)."""
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
