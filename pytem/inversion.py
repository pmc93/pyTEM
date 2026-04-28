"""
inversion.py â€” Inversion utilities for 1-D layered-earth TEM.

Contains:
  - getR               : roughness (smoothness) matrix
  - getJ               : finite-difference Jacobian
  - getJ_analytical    : analytical Jacobian (CUDA/Numba/NumPy)
  - dbdt_to_apprho     : dB/dt â†’ apparent resistivity
  - getRMS, getAlpha, getAlphas : inversion helpers
  - _gn_solve          : Gauss-Newton normal equations solver
  - _backtrack         : step-halving bound enforcement
  - _alpha_search      : log-spaced regularisation search with parabola backtrack
  - invert             : regularised Gauss-Newton inversion loop
"""

import time as _time_mod

import numpy as np

from .filters import MU0, HANKEL_FILTERS, FOURIER_FILTERS, EULER_PARAMS
from .backends import HAS_CUDA
from .kernels_numba import HAS_NUMBA
from .forward import (fwd_circle_central, fwd_square_central,
                      _precompute_filter_dlf, _precompute_filter_euler)

if HAS_NUMBA:
    from .kernels_jacobian import (
        _te_rte_grad_jit,
        _tem_circular_grad_jit, _tem_square_grad_jit,
        _tem_circular_grad_euler_jit, _tem_square_grad_euler_jit,
    )

if HAS_CUDA:
    import cupy as cp
    from .kernels_jacobian import (
        _te_reflection_coeff_grad_gpu,
        _tem_circular_grad_gpu, _tem_square_grad_gpu,
        _tem_circular_grad_euler_gpu, _tem_square_grad_euler_gpu,
    )


# ============================================================
# NumPy / CuPy batch function (array-level, xp-agnostic)
# ============================================================

def _te_grad_batch(lam, omegas, thick, rho, xp):
    """Batched TE gradient for M omegas. Works with numpy or cupy.
    lam: (K,) xp, omegas: (M,) xp, thick: cpu list/array, rho: (N,) xp
    Returns r_TE (M,K), dr_TE (N,M,K)
    """
    n_lay = len(rho)
    K = len(lam)
    M = len(omegas)
    sigma = 1.0 / xp.asarray(rho, dtype=xp.complex128)
    thick_f = [float(t) for t in np.asarray(thick)]
    sval = 1j * omegas
    lam2 = lam ** 2

    Gamma = xp.sqrt(lam2[None, None, :] + sval[None, :, None] * MU0 * sigma[:, None, None])
    dG = -sval[None, :, None] * MU0 * sigma[:, None, None] / (2.0 * Gamma)

    r = xp.zeros((M, K), dtype=xp.complex128)
    r_st = xp.empty((n_lay, M, K), dtype=xp.complex128)
    e_st = xp.empty((n_lay - 1, M, K), dtype=xp.complex128)
    r_st[n_lay - 1] = 0.0

    for j in range(n_lay - 2, -1, -1):
        ej = xp.exp(-2.0 * Gamma[j] * thick_f[j])
        e_st[j] = ej
        ps = (Gamma[j] - Gamma[j+1]) / (Gamma[j] + Gamma[j+1])
        r = ej * (r + ps) / (1.0 + r * ps)
        r_st[j] = r

    pa = (lam[None, :] - Gamma[0]) / (lam[None, :] + Gamma[0])
    da = 1.0 + r_st[0] * pa
    r_TE = (r_st[0] + pa) / da

    adj = (1.0 - pa ** 2) / da ** 2
    dpg0 = -2.0 * lam[None, :] / (lam[None, :] + Gamma[0]) ** 2
    drpa = (1.0 - r_st[0] ** 2) / da ** 2

    dr = xp.zeros((n_lay, M, K), dtype=xp.complex128)
    dr[0] += drpa * dpg0 * dG[0]

    for j in range(n_lay - 1):
        rb = r_st[j+1]; ej = e_st[j]
        gs = Gamma[j] + Gamma[j+1]
        ps = (Gamma[j] - Gamma[j+1]) / gs
        den = 1.0 + rb * ps; num = rb + ps
        drps = ej * (1.0 - rb**2) / den**2
        dej = -2.0 * thick_f[j] * ej
        dr[j]   += adj * (dej * num / den + drps * 2.0 * Gamma[j+1] / gs**2) * dG[j]
        dr[j+1] += adj * drps * (-2.0 * Gamma[j] / gs**2) * dG[j+1]
        adj = adj * ej * (1.0 - ps**2) / den**2

    return r_TE, dr

# ============================================================
# Numba JIT kernels live in kernels_jacobian.py (imported above).
# The functions imported are:
#   _te_rte_grad_jit              — single-omega adjoint recursion
#   _tem_circular_grad_jit/euler  — circle DLF / Euler (with filter_weights)
#   _tem_square_grad_jit/euler    — square DLF / Euler (with filter_weights)
# ============================================================

def getJ_ana(thicknesses, log_resistivities, tx_geom, times,
             geometry='circle_central',
             rx_offset=0.0, rx_y=0.0, n_quad=5, use_symmetry=True,
             use_numba=True, use_cuda=True,
             system_filter=None,
             transform='dlf', hankel_filter='key_101',
             fourier_filter='key_101', euler_order=11):
    """Analytical Jacobian  d(ln(-dBdt_i)) / d(ln rho_j)  for all loop geometries.

    Uses the adjoint Wait recursion: a single forward+backward pass per
    quadrature frequency yields gradients for all N layers simultaneously.

    Transform modes
    ---------------
    'dlf'   — Digital Linear Filter Fourier transform (default).
    'euler' — Euler-Stehfest inverse Laplace transform.  All backends
              (NumPy / Numba / CUDA) share the same adjoint recursion.

    Supported geometries
    --------------------
    'circle_central'  — Rx at centre of circular Tx loop (default)
    'circle_offset'   — Rx at radial offset rx_offset from circular Tx
    'square_central'  — Rx at centre of square Tx loop
    'square_offset'   — Rx at (rx_offset, rx_y) offset from square Tx centre

    Backend strategy
    ----------------
    NumPy   : All n_f frequencies batched into a single _te_grad_batch call
              per gate time so the inner Wait recursion runs in NumPy's C
              layer.  A Python for-loop over frequencies would add n_f
              function-call overheads; batching eliminates them entirely.
    Numba   : Scalar loops compiled to SIMD machine code via JIT; prange
              over gate times achieves CPU parallelism.  Tight scalar loops
              beat NumPy broadcasting inside JIT because large intermediate
              arrays exceed L2 cache for typical problem sizes.
    GPU     : Full (n_t, n_f, K) tensor batched in one CuPy operation to
              saturate GPU occupancy.  Per-frequency launches would leave
              most warps idle.

    System filter
    -------------
    system_filter is applied in the frequency domain before the imaginary
    (DLF) or real (Euler) part is taken.  Since H(omega) is independent of
    resistivity, it multiplies the gradient kernel identically:
        d/d(ln rho_j) [H * K] = H * dK/d(ln rho_j)

    Parameters
    ----------
    thicknesses       : (N-1,) layer thicknesses [m]
    log_resistivities : (N,)   ln(rho_j)
    tx_geom           : float  equivalent circle radius [m]
    times             : (n_t,) gate times [s]
    geometry          : str    loop geometry (default 'circle_central')
    rx_offset         : float  receiver radial / x-offset [m] (default 0.0)
    rx_y              : float  receiver y-offset for square_offset [m]
    n_quad            : int    Gauss-Legendre order for square geometries
    use_symmetry      : bool   exploit x<->y symmetry for square_central
    use_numba         : bool   Numba JIT backend (default True)
    use_cuda          : bool   CuPy GPU backend  (default True)
    system_filter     : callable or None  H(omega) -> complex (default None)
    transform         : 'dlf' or 'euler'
    hankel_filter     : str   (default 'key_101')
    fourier_filter    : str   (default 'key_101') — DLF only
    euler_order       : int   8, 11, 15, or 19 (default 11) — Euler only

    Returns
    -------
    J : (n_t, N) float64
    """
    from scipy.special import j0 as _j0

    thicknesses   = np.asarray(thicknesses, dtype=float)
    resistivities = np.exp(np.asarray(log_resistivities, dtype=float))
    times         = np.asarray(times, dtype=float)
    a             = float(tx_geom)
    n_lay         = len(resistivities)
    n_t           = len(times)

    h_base, h_j0, h_j1 = HANKEL_FILTERS[hankel_filter]
    _use_euler = (transform == 'euler')

    if _use_euler:
        e_eta, e_A = EULER_PARAMS[euler_order]
    else:
        f_base, f_sin, _ = FOURIER_FILTERS[fourier_filter]

    # ---- Geometry-specific pre-computation ----
    _is_circle = geometry in ('circle_central', 'circle_offset')

    if geometry == 'circle_central':
        lam      = h_base / a
        lam_kern = lam * h_j1

    elif geometry == 'circle_offset':
        lam      = h_base / a
        j0_vals  = _j0(lam * float(rx_offset))
        lam_kern = lam * j0_vals * h_j1

    elif geometry == 'square_central':
        side  = a * np.sqrt(np.pi)
        hs    = side / 2.0
        gl_nodes, gl_weights = np.polynomial.legendre.leggauss(n_quad)
        x_pts = hs / 2.0 * (1.0 + gl_nodes)
        w_pts = gl_weights * hs / 2.0
        if use_symmetry:
            rho_q_l, area_w_l = [], []
            for _i in range(n_quad):
                for _jj in range(_i, n_quad):
                    w = w_pts[_i] * w_pts[_jj]
                    if _i != _jj:
                        w *= 2.0
                    rho_q_l.append(np.sqrt(x_pts[_i]**2 + x_pts[_jj]**2))
                    area_w_l.append(w)
            rho_q  = np.array(rho_q_l)
            area_w = np.array(area_w_l)
        else:
            _xx, _yy = np.meshgrid(x_pts, x_pts)
            _wx, _wy = np.meshgrid(w_pts, w_pts)
            rho_q  = np.sqrt(_xx.ravel()**2 + _yy.ravel()**2)
            area_w = (_wx * _wy).ravel()
        quad_scale = 4.0

    elif geometry == 'square_offset':
        side  = a * np.sqrt(np.pi)
        hs    = side / 2.0
        gl_nodes, gl_weights = np.polynomial.legendre.leggauss(n_quad)
        x_pts = hs * gl_nodes
        wx    = hs * gl_weights
        _xx, _yy = np.meshgrid(x_pts, x_pts, indexing='xy')
        _wx, _wy = np.meshgrid(wx, wx, indexing='xy')
        rho_q  = np.sqrt((_xx.ravel() - float(rx_offset))**2 +
                         (_yy.ravel() - float(rx_y))**2)
        rho_q  = np.maximum(rho_q, 1e-6)
        area_w = (_wx * _wy).ravel()
        quad_scale = 1.0

    else:
        raise ValueError(
            f"Unknown geometry '{geometry}'. Choose from: "
            "'circle_central', 'circle_offset', 'square_central', 'square_offset'.")

    # ---- Pre-evaluate system filter at all transform frequencies ----
    # When system_filter is None, ones are used (no effect on the integrals).
    # For DLF:   filter_weights[i, k] = H(f_base[k] / times[i]),  shape (n_t, n_f).
    # For Euler: filter_weights[i, k] = H(omega_k(times[i])),      shape (n_t, n_eval).
    if _use_euler:
        n_eval = len(e_eta)
        filter_weights = (_precompute_filter_euler(system_filter, times, e_eta, e_A)
                          if system_filter is not None
                          else np.ones((n_t, n_eval), dtype=np.complex128))
    else:
        n_f = len(f_base)
        filter_weights = (_precompute_filter_dlf(system_filter, times, f_base)
                          if system_filter is not None
                          else np.ones((n_t, n_f), dtype=np.complex128))

    # ---- Backend dispatch: CUDA > Numba > NumPy ----
    _use_nb  = HAS_NUMBA and use_numba
    _use_gpu = HAS_CUDA  and use_cuda and not _use_nb

    if _use_nb:
        # Numba path — scalar loops JIT-compiled to SIMD + prange over gates.
        # filter_weights passed as complex128 array; Numba handles arithmetic natively.
        if _use_euler:
            if _is_circle:
                dbdt, J_raw = _tem_circular_grad_euler_jit(
                    times, thicknesses, resistivities, lam, lam_kern, MU0,
                    e_eta, e_A, filter_weights)
            else:
                dbdt, J_raw = _tem_square_grad_euler_jit(
                    times, thicknesses, resistivities,
                    rho_q, area_w, float(quad_scale),
                    h_base, h_j0, MU0, e_eta, e_A, filter_weights)
        else:
            if _is_circle:
                dbdt, J_raw = _tem_circular_grad_jit(
                    times, thicknesses, resistivities, lam, lam_kern, MU0,
                    f_base, f_sin, filter_weights)
            else:
                dbdt, J_raw = _tem_square_grad_jit(
                    times, thicknesses, resistivities,
                    rho_q, area_w, float(quad_scale),
                    h_base, h_j0, MU0, f_base, f_sin, filter_weights)

    elif _use_gpu:
        # GPU path — full (n_t, n_f, K) tensor batched in one CuPy operation.
        # d_filter_weights transferred to GPU; CuPy handles complex128 natively.
        d_h_base         = cp.asarray(h_base)
        d_filter_weights = cp.asarray(filter_weights)
        if _use_euler:
            if _is_circle:
                dbdt, J_raw = _tem_circular_grad_euler_gpu(
                    times, thicknesses, resistivities, a, lam_kern,
                    d_h_base, e_eta, e_A, d_filter_weights)
            else:
                d_h_j0 = cp.asarray(h_j0)
                dbdt, J_raw = _tem_square_grad_euler_gpu(
                    times, thicknesses, resistivities,
                    rho_q, area_w, float(quad_scale),
                    d_h_base, d_h_j0, e_eta, e_A, d_filter_weights)
        else:
            d_f_base = cp.asarray(f_base)
            d_f_sin  = cp.asarray(f_sin)
            if _is_circle:
                dbdt, J_raw = _tem_circular_grad_gpu(
                    times, thicknesses, resistivities, a, lam_kern,
                    d_h_base, d_f_base, d_f_sin, d_filter_weights)
            else:
                d_h_j0 = cp.asarray(h_j0)
                dbdt, J_raw = _tem_square_grad_gpu(
                    times, thicknesses, resistivities,
                    rho_q, area_w, float(quad_scale),
                    d_h_base, d_h_j0, d_f_base, d_f_sin, d_filter_weights)

    else:
        # NumPy path — all n_f frequencies batched into one _te_grad_batch call
        # per gate time.  _te_grad_batch(lam, omega_arr, thick, rho, np) returns
        # r_TE (M, K) and dr_TE (N, M, K), keeping inner loops in NumPy's C layer.
        dbdt  = np.zeros(n_t)
        J_raw = np.zeros((n_t, n_lay))

        if _use_euler:
            k_arr   = np.arange(len(e_eta), dtype=float)
            signs_k = (-1.0)**k_arr * e_eta                 # (n_eval,)
            for i, t in enumerate(times):
                c         = e_A / (2.0 * t)
                h_step    = np.pi / t
                omega_arr = k_arr * h_step - c * 1j          # (n_eval,) complex
                fw        = filter_weights[i]                 # (n_eval,) complex
                if _is_circle:
                    r_TE, dr_TE = _te_grad_batch(lam, omega_arr, thicknesses, resistivities, np)
                    # r_TE: (n_eval, K),  dr_TE: (N, n_eval, K)
                    hz_c  = 0.5 * (r_TE  * lam_kern[None, :]).sum(-1)         # (n_eval,)
                    dhz_c = 0.5 * (dr_TE * lam_kern[None, None, :]).sum(-1)   # (N, n_eval)
                else:
                    hz_c  = np.zeros(len(omega_arr), dtype=complex)
                    dhz_c = np.zeros((n_lay, len(omega_arr)), dtype=complex)
                    for q in range(len(rho_q)):
                        rq     = rho_q[q];  wq = area_w[q]
                        lam_q  = h_base / rq
                        kern_q = lam_q**2 * h_j0 / (rq * 4.0 * np.pi)
                        r_q, dr_q = _te_grad_batch(lam_q, omega_arr, thicknesses, resistivities, np)
                        hz_c  += wq * (r_q  * kern_q[None, :]).sum(-1)
                        dhz_c += wq * (dr_q * kern_q[None, None, :]).sum(-1)
                    hz_c  *= quad_scale
                    dhz_c *= quad_scale
                # Apply filter, then Euler dot product
                hz_acc  = MU0 * np.dot(signs_k, (hz_c  * fw).real)         # scalar
                dhz_acc = MU0 * ((dhz_c * fw[None, :]).real @ signs_k)     # (N,)
                prefac      = np.exp(e_A / 2.0) / t
                dbdt[i]     = -prefac * hz_acc
                J_raw[i, :] = -prefac * dhz_acc

        else:
            for i, t in enumerate(times):
                omega_arr = f_base / t      # (n_f,) — all frequencies at once
                fw        = filter_weights[i]  # (n_f,) complex
                if _is_circle:
                    r_TE, dr_TE = _te_grad_batch(lam, omega_arr, thicknesses, resistivities, np)
                    # r_TE: (n_f, K),  dr_TE: (N, n_f, K)
                    hz_c  = 0.5 * (r_TE  * lam_kern[None, :]).sum(-1)         # (n_f,)
                    dhz_c = 0.5 * (dr_TE * lam_kern[None, None, :]).sum(-1)   # (N, n_f)
                else:
                    hz_c  = np.zeros(len(omega_arr), dtype=complex)
                    dhz_c = np.zeros((n_lay, len(omega_arr)), dtype=complex)
                    for q in range(len(rho_q)):
                        rq     = rho_q[q];  wq = area_w[q]
                        lam_q  = h_base / rq
                        kern_q = lam_q**2 * h_j0 / (rq * 4.0 * np.pi)
                        r_q, dr_q = _te_grad_batch(lam_q, omega_arr, thicknesses, resistivities, np)
                        hz_c  += wq * (r_q  * kern_q[None, :]).sum(-1)
                        dhz_c += wq * (dr_q * kern_q[None, None, :]).sum(-1)
                    hz_c  *= quad_scale
                    dhz_c *= quad_scale
                # Apply filter, then Fourier dot product
                hz_im  = MU0 * (hz_c  * fw).imag           # (n_f,) real
                dhz_im = MU0 * (dhz_c * fw[None, :]).imag  # (N, n_f) real
                dbdt[i]     = np.dot(hz_im, f_sin) / t
                J_raw[i, :] = (dhz_im @ f_sin) / t

    # DLF: normalise by 2/pi.  Euler kernels incorporate exp(A/2)/t and the
    # step-off sign (-1) internally, so no further scaling is needed.
    if not _use_euler:
        scale  = 2.0 / np.pi
        dbdt  *= scale
        J_raw *= scale

    f0 = -dbdt
    if np.any(f0 <= 0):
        print(f"WARNING: {(f0 <= 0).sum()} non-positive dbdt values (zeroed in J)")
    J     = np.zeros((n_t, n_lay))
    valid = f0 > 0
    J[valid, :] = -J_raw[valid, :] / f0[valid, None]
    np.nan_to_num(J, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    return J



def dbdt_to_apprho(obs_data, tx_area, times):
    """Convert dB/dt to apparent resistivity.

    Parameters
    ----------
    obs_data : array_like
        Observed dB/dt values [T/s].
    tx_area  : float
        Transmitter moment (current Ã— area) [AÂ·mÂ²].
    times    : array_like
        Gate centre times [s].

    Returns
    -------
    rho_a : ndarray
        Apparent resistivity [OhmÂ·m].
    """
    M = tx_area
    term = (2 * MU0 * M) / (5 * times * obs_data)
    app_rho = (MU0 / (4 * np.pi * times)) * (term ** (2 / 3))
    return app_rho


def getRMS(obs_data, mod_data, obs_noise):
    """Root-mean-square misfit normalised by noise."""
    total_points = obs_data.size
    data_residual = (mod_data - obs_data) ** 2 / (obs_noise) ** 2
    rms = np.sqrt(np.sum(data_residual) / total_points)
    return rms


def getAlpha(alpha_start, step):
    """Log-spaced regularisation parameter for a given cooling step."""
    alpha_step = 1 / 9
    log_alpha = np.log10(alpha_start) - alpha_step * step
    alpha = 10 ** log_alpha
    return alpha


def getAlphas(alpha, thicknesses):
    """Depth-weighted regularisation vector."""
    thicknesses = np.asarray(thicknesses)
    tops = np.cumsum(np.concatenate(([0], thicknesses[:-1])))
    midpoints = tops + thicknesses / 2.0
    del_z = np.diff(midpoints)
    alpha_factor = np.empty(len(del_z) + 1, dtype=del_z.dtype)
    alpha_factor[0] = 1 / del_z[0]
    alpha_factor[1:-1] = 1 / del_z[:-1] + 1 / del_z[1:]
    alpha_factor[-1] = 1 / del_z[-1]
    return alpha * alpha_factor


def getR(resistivities, damp=1e-4):
    """First-order roughness (smoothness) matrix with optional damping."""
    n_params = resistivities.size
    D = np.zeros((n_params - 1, n_params))
    for k in range(n_params - 1):
        D[k, k] = -1.0
        D[k, k + 1] = 1.0
    R = D.T @ D + damp * np.eye(n_params)
    return R


def getJ_fd(thicknesses, log_resistivities, tx_geom, times,
            use_numba=False, use_cuda=True, eps=1e-4, fwd=fwd_circle_central,
            transform='dlf', hankel_filter='key_201', fourier_filter='key_101',
            euler_order=11):
    """Finite-difference Jacobian d(log(-dBdt))/d(ln rho)."""
    fwd_kw = dict(use_numba=use_numba, use_cuda=use_cuda, transform=transform,
                  hankel_filter=hankel_filter, fourier_filter=fourier_filter,
                  euler_order=euler_order)

    if fwd is fwd_square_central:
        f0 = -fwd(thicknesses=thicknesses, resistivities=np.exp(log_resistivities),
                  side_length=tx_geom, times=times, **fwd_kw)
    else:
        f0 = -fwd(thicknesses=thicknesses, resistivities=np.exp(log_resistivities),
                  tx_radius=tx_geom, times=times, **fwd_kw)

    bad_f0 = f0 <= 0
    if np.any(bad_f0):
        print(f"WARNING: f0 has {bad_f0.sum()} non-positive values at gate indices {np.where(bad_f0)[0]} (zeroed in J)")

    J = np.zeros((f0.size, log_resistivities.size))
    bad_count = 0
    for i in range(log_resistivities.size):
        perturbed = log_resistivities.copy()
        step = eps * max(1.0, abs(log_resistivities[i]))
        perturbed[i] += step
        if fwd is fwd_square_central:
            fi = -fwd(thicknesses=thicknesses, resistivities=np.exp(perturbed),
                      side_length=tx_geom, times=times, **fwd_kw)
        else:
            fi = -fwd(thicknesses=thicknesses, resistivities=np.exp(perturbed),
                      tx_radius=tx_geom, times=times, **fwd_kw)

        valid = (f0 > 0) & (fi > 0)
        if not np.all(valid):
            bad_count += 1
        J[valid, i] = (np.log(fi[valid]) - np.log(f0[valid])) / step

    if bad_count:
        print(f"WARNING: {bad_count}/{log_resistivities.size} perturbed models had non-positive values (zeroed in J)")

    return J

# ============================================================
# Regularised Gauss-Newton inversion helpers
# ============================================================

def _gn_solve(Jw, dw, R, alpha_vector, m):
    """Solve the weighted, regularised Gauss-Newton normal equations.

    Solves  (Jw^T Jw + diag(alpha) R) dm = Jw^T dw - diag(alpha) R m
    via least-squares (robust to mild rank deficiency).

    Parameters
    ----------
    Jw           : (n_d, N)  noise-weighted Jacobian
    dw           : (n_d,)    noise-weighted log-space residuals
    R            : (N, N)    roughness matrix from getR()
    alpha_vector : (N,)      per-layer regularisation weights from getAlphas()
    m            : (N,)      current log-resistivity model

    Returns
    -------
    dm : (N,) model update
    """
    AR  = np.diag(alpha_vector) @ R
    lhs = Jw.T @ Jw + AR
    rhs = Jw.T @ dw - AR @ m
    dm, _, _, _ = np.linalg.lstsq(lhs, rhs, rcond=1e-10)
    return dm


def _backtrack(m, delta, ln_rho_min, ln_rho_max):
    """Halve the step length until the trial model is within bounds.

    Parameters
    ----------
    m           : (N,) current log-resistivity model
    delta       : (N,) proposed step
    ln_rho_min  : float  lower bound in log-resistivity space
    ln_rho_max  : float  upper bound in log-resistivity space

    Returns
    -------
    trial : (N,) new model (clipped to bounds as a last resort)
    step  : float  accepted step length in [0, 1]
    """
    step = 1.0
    for _ in range(10):
        trial = m + step * delta
        if np.all(trial >= ln_rho_min) and np.all(trial <= ln_rho_max):
            return trial, step
        step *= 0.5
    return np.clip(m + step * delta, ln_rho_min, ln_rho_max), step


def _alpha_search(alpha_start, alpha_steps, Jw, dw, R, m,
                  thicknesses, fwd_fn, obs_data, w,
                  ln_rho_min, ln_rho_max, plot=False):
    """Log-spaced alpha search with parabola backtrack to RMS = 1.

    Tests ``alpha_steps`` regularisation strengths starting from
    ``alpha_start`` on a log-spaced ladder defined by ``getAlpha``, evaluates
    the RMS for each, fits a polynomial, and locates alpha* where RMS = 1.

    Parameters
    ----------
    alpha_start  : float      largest (strongest) regularisation to try
    alpha_steps  : int        number of alpha values on the ladder
    Jw           : (n_d, N)   noise-weighted Jacobian
    dw           : (n_d,)     noise-weighted log-space residuals
    R            : (N, N)     roughness matrix
    m            : (N,)       current log-resistivity model
    thicknesses  : (N,)       layer thicknesses for getAlphas() depth weighting
    fwd_fn       : callable   log_rho -> (n_t,) positive forward response
    obs_data     : (n_t,)     observed data (positive)
    w            : (n_t,)     noise weights (1 / noise_log)
    ln_rho_min   : float
    ln_rho_max   : float
    plot         : bool       show alpha-RMS diagnostic figure (default False)

    Returns
    -------
    alpha_hist : list of float    tested + parabola alpha values
    rms_hist   : list of float    corresponding RMS values
    delta_hist : list of ndarray  corresponding model deltas (m_trial - m)
    """
    alpha_hist, rms_hist, delta_hist = [], [], []

    for i in range(alpha_steps):
        alpha = getAlpha(alpha_start, step=i)
        avs   = getAlphas(alpha, thicknesses)
        delta = _gn_solve(Jw, dw, R, avs, m)
        trial, step = _backtrack(m, delta, ln_rho_min, ln_rho_max)
        mod   = fwd_fn(trial)
        valid = (obs_data > 0) & (mod > 0)
        d_res = np.log(obs_data[valid]) - np.log(mod[valid])
        rms   = np.sqrt(np.mean((w[valid] * d_res) ** 2))
        print(f"    alpha = {alpha:.3g},  RMS = {rms:.4f}"
              + (f"  (step = {step:.2f})" if step < 1.0 else ""))
        alpha_hist.append(alpha)
        rms_hist.append(rms)
        delta_hist.append(trial - m)

        if len(rms_hist) > 1 and rms > rms_hist[-2]:
            print("    RMS increased — stopping alpha search early.")
            break

    # Polynomial backtrack to find alpha* where RMS = 1
    x_data = np.log10(np.array(alpha_hist))
    y_data = np.array(rms_hist)
    deg    = min(2, len(x_data) - 1)
    parabola_alpha = None
    coeffs         = None

    if deg >= 1 and np.min(y_data) < 1.0:
        coeffs         = np.polyfit(x_data, y_data, deg)
        root_c         = coeffs.copy()
        root_c[-1]    -= 1.0
        roots          = np.roots(root_c)
        x_lo, x_hi     = x_data.min() - 1.0, x_data.max() + 1.0
        real_roots      = roots[np.abs(roots.imag) < 1e-10].real
        valid_roots     = real_roots[(real_roots >= x_lo) & (real_roots <= x_hi)]

        if valid_roots.size > 0:
            parabola_x     = float(valid_roots.max())
            parabola_alpha = 10.0 ** parabola_x
            avs_par        = getAlphas(parabola_alpha, thicknesses)
            delta_par      = _gn_solve(Jw, dw, R, avs_par, m)
            trial_par, step_par = _backtrack(m, delta_par, ln_rho_min, ln_rho_max)
            mod_par        = fwd_fn(trial_par)
            valid_par      = (obs_data > 0) & (mod_par > 0)
            d_par          = np.log(obs_data[valid_par]) - np.log(mod_par[valid_par])
            rms_par        = np.sqrt(np.mean((w[valid_par] * d_par) ** 2))
            print(f"    Parabola: alpha* = {parabola_alpha:.3g},  "
                  f"predicted RMS = 1.00,  actual RMS = {rms_par:.4f}"
                  + (f"  (step = {step_par:.2f})" if step_par < 1.0 else ""))
            alpha_hist.append(parabola_alpha)
            rms_hist.append(rms_par)
            delta_hist.append(trial_par - m)

    if plot and coeffs is not None:
        import matplotlib.pyplot as _plt
        fig, ax = _plt.subplots(figsize=(5, 3.5))
        x_fit = np.linspace(x_data.min() - 0.5, x_data.max() + 0.5, 300)
        y_fit = np.polyval(coeffs, x_fit)
        ax.plot(x_fit, y_fit, '-', color='C0', lw=1.5,
                label=f'Degree-{deg} fit')
        ax.plot(x_data, y_data, 'o', color='C1', zorder=5,
                label='Tested alphas')
        ax.axhline(1.0, color='k', ls='--', lw=1, label='RMS = 1 target')
        if parabola_alpha is not None:
            ax.axvline(np.log10(parabola_alpha), color='C2', ls='--', lw=1,
                       label=f'$\\alpha^* = {parabola_alpha:.3g}$')
            ax.plot(np.log10(parabola_alpha), 1.0, '*', color='C2',
                    markersize=12, zorder=6)
        ax.set_xlabel('$\\log_{{10}}(\\alpha)$')
        ax.set_ylabel('RMS')
        ax.set_title('Alpha search: parabola backtrack')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        _plt.show()

    return alpha_hist, rms_hist, delta_hist


# ============================================================
# Main inversion function
# ============================================================

def invert(obs_data, thicknesses, log_resistivities, tx_radius, times,
           alpha_start=None, alpha_steps=5, maxit=20, eps=1e-4,
           noise_std=0.02, use_numba=True, use_cuda=True,
           calc_sens=False, store_J=False,
           transform='euler', hankel_filter='key_101', fourier_filter='key_81',
           euler_order=11, rho_min=1e-1, rho_max=1e5, max_noise_frac=0.10,
           plot_alpha=False, analytical_j=True,
           system_filter=None,
           waveform_times=None, waveform_currents=None, n_step=200):
    """Regularised Gauss-Newton inversion for 1-D layered-earth TEM.

    Minimises  phi(m) = ||W (ln d_obs - ln d_pred(m))||^2 + alpha * m^T R m
    using iterative Gauss-Newton updates with a log-spaced alpha search and
    parabola backtrack to target RMS = 1.

    All optimisation is performed in log-resistivity space, so the forward
    function is always evaluated with ``resistivities = exp(m)``.

    System filter
    -------------
    Pass ``system_filter`` (a callable H(omega) -> complex array) to apply a
    frequency-domain instrument response inside the forward model and the
    analytical Jacobian.  When ``analytical_j=False``, the filter is applied
    inside ``fwd_circle_central`` automatically; when ``analytical_j=True``,
    it is passed to ``getJ_analytical``.

    Waveform convolution
    --------------------
    When ``waveform_times`` and ``waveform_currents`` are supplied, the step-off
    response is computed on a dense log-spaced time grid (``n_step`` points)
    and convolved with the piecewise-linear waveform using
    ``waveform.convolve_waveform``.  In this mode the Jacobian is always
    computed by finite differences (``analytical_j`` is ignored), because the
    linear convolution operator is automatically captured in the FD differences.

    Parameters
    ----------
    obs_data            : (n_t,) observed dBz/dt [T/s], positive values
    thicknesses         : (N,)   layer thicknesses [m]
    log_resistivities   : (N,)   initial ln(rho) [ln(Ohm.m)]
    tx_radius           : float  equivalent transmitter radius [m]
    times               : (n_t,) gate centre times [s]
    alpha_start         : float or None  starting regularisation strength
                          (auto-estimated from JTd if None)
    alpha_steps         : int    number of alpha values per outer iteration
    maxit               : int    maximum Gauss-Newton iterations
    eps                 : float  finite-difference step for FD Jacobian
    noise_std           : float or (n_t,)  fractional noise standard deviation
    use_numba           : bool   enable Numba JIT backend
    use_cuda            : bool   enable CuPy GPU backend
    calc_sens           : bool   compute parameter sensitivity at convergence
    store_J             : bool   store Jacobian at each iteration
    transform           : 'dlf' or 'euler'
    hankel_filter       : str    (default 'key_101')
    fourier_filter      : str    (default 'key_81')
    euler_order         : int    (default 11)
    rho_min             : float  lower resistivity bound [Ohm.m] (default 0.1)
    rho_max             : float  upper resistivity bound [Ohm.m] (default 1e5)
    max_noise_frac      : float  noise floor as a fraction of peak data
    plot_alpha          : bool   show alpha-RMS plot at each iteration
    analytical_j        : bool   use analytical Jacobian (ignored when waveform)
    system_filter       : callable or None  H(omega) -> complex
    waveform_times      : array-like or None  waveform break points [s]
    waveform_currents   : array-like or None  current at break points [A]
    n_step              : int    dense time-grid size for waveform convolution

    Returns
    -------
    result : dict with keys
        'log_resistivities' : (N,)         final ln(rho)
        'resistivities'     : (N,)         final rho [Ohm.m]
        'thicknesses'       : (N,)         layer thicknesses [m]
        'model_history'     : list of (N,) all models (initial + each iteration)
        'rms_history'       : list of float  RMS after each iteration
        'J_history'         : list of (n_t, N) or None
        'sensitivity'       : (N,) or None   column-norm of final J
        'times'             : (n_t,) gate times [s]
        'obs_data'          : (n_t,) observed data
        'n_iter'            : int   number of completed iterations
    """
    # ---- Parse inputs ----
    obs_data    = np.asarray(obs_data,          dtype=float)
    thicknesses = np.asarray(thicknesses,       dtype=float)
    m           = np.asarray(log_resistivities, dtype=float).copy()
    times       = np.asarray(times,             dtype=float)

    ln_rho_min = np.log(float(rho_min))
    ln_rho_max = np.log(float(rho_max))

    # ---- Noise weighting ----
    if np.isscalar(noise_std):
        noise_abs = noise_std * np.abs(obs_data)
    else:
        noise_abs = np.asarray(noise_std, dtype=float)
    # Floor: noise cannot be smaller than max_noise_frac * peak |obs|
    noise_abs = np.maximum(noise_abs, max_noise_frac * np.abs(obs_data).max())
    # Log-space noise: sigma_log_i = sigma_i / |d_i|
    noise_log = noise_abs / np.abs(obs_data)
    w         = 1.0 / noise_log   # weight per datum in log space

    # ---- Waveform convolution setup ----
    _use_waveform = (waveform_times is not None and waveform_currents is not None)
    if _use_waveform:
        from .waveform import convolve_waveform as _convolve
        _wf_t  = np.asarray(waveform_times,    dtype=float)
        _wf_I  = np.asarray(waveform_currents, dtype=float)
        _step_t = np.logspace(
            np.log10(times.min() * 1e-2),
            np.log10(times.max() * 10.0),
            n_step,
        )


    _fwd_kw = dict(
        tx_radius=float(tx_radius),
        use_numba=use_numba,
        use_cuda=use_cuda,
        system_filter=system_filter,
        transform=transform,
        hankel_filter=hankel_filter,
        fourier_filter=fourier_filter,
        euler_order=euler_order,
    )

    # ---- Forward model closure ----
    def _forward_response(log_rho):
        res = np.exp(log_rho)
        if _use_waveform:
            step_resp = -fwd_circle_central(
                thicknesses=thicknesses, resistivities=res,
                times=_step_t, **_fwd_kw)
            return _convolve(_step_t, step_resp, _wf_t, _wf_I, times)
        return -fwd_circle_central(
            thicknesses=thicknesses, resistivities=res,
            times=times, **_fwd_kw)

    # ---- Jacobian closure ----
    def _build_jacobian(log_rho):
        if analytical_j and not _use_waveform:
            # Pure analytical Jacobian — no waveform.
            return getJ_analytical(
                thicknesses=thicknesses,
                log_resistivities=log_rho,
                tx_geom=float(tx_radius),
                times=times,
                use_numba=use_numba,
                use_cuda=False,
                system_filter=system_filter,
                transform=transform,
                hankel_filter=hankel_filter,
                fourier_filter=fourier_filter,
                euler_order=euler_order,
            )

        if analytical_j and _use_waveform:
            # Analytical Jacobian with waveform convolution.
            #
            # The convolution G_i = conv(F, w)_i is linear in F, so:
            #   dG_i/d(ln rho_j) = conv(dF/d(ln rho_j), w)_i
            #
            # getJ_analytical returns J_anal[k,j] = d ln(F(t_k)) / d ln(rho_j)
            # so the absolute gradient is: dF(t_k)/d(ln rho_j) = J_anal[k,j] * F(t_k)
            #
            # Then the log-space Jacobian of the convolved response:
            #   J_conv[i,j] = dG_i/d(ln rho_j) / G_i
            res       = np.exp(log_rho)
            step_resp = -fwd_circle_central(
                thicknesses=thicknesses, resistivities=res,
                times=_step_t, **_fwd_kw)
            J_anal = getJ_analytical(
                thicknesses=thicknesses,
                log_resistivities=log_rho,
                tx_geom=float(tx_radius),
                times=_step_t,
                use_numba=use_numba,
                use_cuda=False,
                system_filter=system_filter,
                transform=transform,
                hankel_filter=hankel_filter,
                fourier_filter=fourier_filter,
                euler_order=euler_order,
            )  # (n_step, N)
            G = _convolve(_step_t, step_resp, _wf_t, _wf_I, times)  # (n_t,)
            J_conv = np.zeros((len(times), log_rho.size))
            valid_g = G > 0
            for j in range(log_rho.size):
                dF_j = J_anal[:, j] * step_resp          # absolute gradient on dense grid
                dG_j = _convolve(_step_t, dF_j, _wf_t, _wf_I, times)
                J_conv[valid_g, j] = dG_j[valid_g] / G[valid_g]
            return J_conv

        # Finite-difference Jacobian: waveform convolution is included
        # automatically because _forward_response handles it.
        f0 = _forward_response(log_rho)
        J  = np.zeros((f0.size, log_rho.size))
        for i in range(log_rho.size):
            pert    = log_rho.copy()
            h       = eps * max(1.0, abs(log_rho[i]))
            pert[i] += h
            fi      = _forward_response(pert)
            valid   = (f0 > 0) & (fi > 0)
            J[valid, i] = (np.log(fi[valid]) - np.log(f0[valid])) / h
        return J

    # ---- Initial alpha heuristic ----
    print("Building initial Jacobian...")
    t0 = _time_mod.time()
    J0 = _build_jacobian(m)
    d0 = _forward_response(m)
    print(f"  done ({_time_mod.time() - t0:.1f} s)")

    valid0  = (obs_data > 0) & (d0 > 0)
    res0    = np.zeros(len(obs_data))
    res0[valid0] = np.log(obs_data[valid0]) - np.log(d0[valid0])
    Jw0     = J0 * w[:, None]
    dw0     = res0 * w

    if alpha_start is None:
        alpha_start = 10.0 ** np.ceil(
            np.log10(np.linalg.norm(Jw0.T @ dw0, np.inf) + 1e-30))
    print(f"alpha_start = {alpha_start:.3g}")

    # ---- Gauss-Newton loop ----
    rms_history   = []
    model_history = [m.copy()]
    J_history     = [J0.copy()] if store_J else []
    J_cur         = J0

    t_loop = _time_mod.time()

    for it in range(maxit):
        d_pred = _forward_response(m)
        valid  = (obs_data > 0) & (d_pred > 0)

        if not np.any(valid):
            print("WARNING: No valid data — stopping.")
            break

        res_log         = np.zeros(len(obs_data))
        res_log[valid]  = np.log(obs_data[valid]) - np.log(d_pred[valid])
        rms = np.sqrt(np.mean((w[valid] * res_log[valid]) ** 2))
        rms_history.append(rms)

        elapsed = _time_mod.time() - t_loop
        print(f"Iteration {it + 1:>3d}:  RMS = {rms:.4f}  ({elapsed:.1f} s)")

        if rms <= 1.0:
            print("  RMS <= 1 — converged.")
            break

        if it > 0:
            J_cur = _build_jacobian(m)
            if store_J:
                J_history.append(J_cur.copy())

        R  = getR(m)
        Jw = J_cur * w[:, None]
        dw = res_log * w

        alpha_h, rms_h, delta_h = _alpha_search(
            alpha_start, alpha_steps,
            Jw, dw, R, m,
            thicknesses, _forward_response, obs_data, w,
            ln_rho_min, ln_rho_max,
            plot=plot_alpha,
        )

        # Select model update closest to RMS = 1 from below; fall back to min.
        rms_arr = np.array(rms_h)
        below   = rms_arr[rms_arr <= 1.0]
        if below.size > 0:
            best_idx = int(np.where(rms_arr == below.max())[0][-1])
        else:
            best_idx = int(np.argmin(rms_arr))

        m = np.clip(m + delta_h[best_idx], ln_rho_min, ln_rho_max)
        model_history.append(m.copy())

    # ---- Optional sensitivity ----
    sensitivity = None
    if calc_sens:
        Jf          = _build_jacobian(m)
        sensitivity = np.sqrt(np.sum(Jf ** 2, axis=0))

    return {
        'log_resistivities': m,
        'resistivities':     np.exp(m),
        'thicknesses':       thicknesses,
        'model_history':     model_history,
        'rms_history':       rms_history,
        'J_history':         J_history if store_J else None,
        'sensitivity':       sensitivity,
        'times':             times,
        'obs_data':          obs_data,
        'n_iter':            len(rms_history),
    }

