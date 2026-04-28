"""
kernels_jacobian.py — Numba JIT and CuPy GPU kernels for the analytical Jacobian.

Implements the adjoint Wait recursion: a single forward+backward pass per
(omega, lambda) point yields d(r_TE)/d(ln rho_j) for all N layers at once.

Backend rationale
-----------------
Numba (scalar loops, JIT-compiled to SIMD + prange over gate times)
    Tight scalar loops compile to native SIMD machine code and prange over
    gate times parallelises across CPU cores.  Allocating large intermediate
    arrays (n_f, K) inside a JIT kernel carries heap overhead and the working
    set would exceed L2 cache for typical problem sizes (30 layers, 101-pt
    Hankel filter), so a per-frequency inner loop is preferred over
    NumPy-style broadcasting.

GPU / CuPy (full (n_t, n_f, K) tensor batched in a single CuPy operation)
    CUDA throughput depends on keeping many warps in flight simultaneously.
    Batching all gate times and all filter frequencies into one (n_t, n_f, K)
    tensor maximises occupancy.  Per-frequency Python loops would serialise
    kernel launches and leave most warps idle.

System filter
-------------
A system filter H(omega) is applied by multiplying the complex kernel value
*before* taking the imaginary (DLF) or real (Euler) part.  Since H(omega)
does not depend on resistivity, the same weight applies identically to the
forward kernel and to every layer's gradient kernel:

    d/d(ln rho_j) [ H(omega) * K(omega, rho) ] = H(omega) * dK/d(ln rho_j)

Callers pre-evaluate filter_weights[n_t, n_eval] = H(omega_ki) using
_precompute_filter_dlf / _precompute_filter_euler from forward.py.
When no filter is needed, pass np.ones((n_t, n_eval), dtype=np.complex128).
"""

import numpy as np
from .transform_weights import MU0
from .backends import HAS_CUDA

try:
    import numba as nb
    HAS_NUMBA = True
    _NB_OPTS = {'nogil': True, 'cache': True}
except ImportError:
    HAS_NUMBA = False


# ============================================================================
# Numba JIT kernels
# ============================================================================
#
# Scalar loops compiled to native SIMD machine code via Numba JIT.
# prange over gate times achieves parallelism across CPU cores.
# Per-frequency inner loops beat NumPy broadcasting inside JIT because:
#   1. Heap allocation of (n_f, K) intermediates carries overhead in JIT kernels
#   2. Large tensors exceed L2 cache for typical problem sizes
#
# _te_rte_grad_jit handles both real (DLF) and complex (Euler) omega
# natively — the same JIT function is called by both DLF and Euler kernels.
# ============================================================================

if HAS_NUMBA:

    @nb.njit(**_NB_OPTS)
    def _te_rte_grad_jit(lam, omega, thicknesses, resistivities, mu0):
        """Wait recursion + adjoint gradient — single omega, scalar loops (Numba JIT).

        Returns r_te (K,) and dr_te (N, K) = d(r_TE)/d(ln rho_j) for all layers.
        Handles both real (DLF) and complex (Euler) omega natively.
        """
        n_lay = len(resistivities)
        n_lam = len(lam)
        sval  = 1j * omega

        Gamma  = np.empty((n_lay, n_lam), dtype=np.complex128)
        dGamma = np.empty((n_lay, n_lam), dtype=np.complex128)
        for j in range(n_lay):
            sigma_j = 1.0 / resistivities[j]
            prod    = sval * mu0 * sigma_j
            for m in range(n_lam):
                g            = np.sqrt(lam[m]**2 + prod)
                Gamma[j, m]  = g
                dGamma[j, m] = -prod / (2.0 * g)

        r_store   = np.zeros((n_lay, n_lam), dtype=np.complex128)
        exp_store = np.zeros((n_lay - 1, n_lam), dtype=np.complex128)
        r = np.zeros(n_lam, dtype=np.complex128)
        r_store[n_lay - 1] = r

        for j in range(n_lay - 2, -1, -1):
            h     = thicknesses[j]
            r_new = np.empty(n_lam, dtype=np.complex128)
            for m in range(n_lam):
                ej              = np.exp(-2.0 * Gamma[j, m] * h)
                exp_store[j, m] = ej
                psi             = (Gamma[j, m] - Gamma[j+1, m]) / (Gamma[j, m] + Gamma[j+1, m])
                r_new[m]        = ej * (r[m] + psi) / (1.0 + r[m] * psi)
            r = r_new
            r_store[j] = r

        r_te    = np.empty(n_lam, dtype=np.complex128)
        psi_air = np.empty(n_lam, dtype=np.complex128)
        for m in range(n_lam):
            pa         = (lam[m] - Gamma[0, m]) / (lam[m] + Gamma[0, m])
            psi_air[m] = pa
            r_te[m]    = (r_store[0, m] + pa) / (1.0 + r_store[0, m] * pa)

        dr_te = np.zeros((n_lay, n_lam), dtype=np.complex128)
        adj   = np.empty(n_lam, dtype=np.complex128)
        for m in range(n_lam):
            pa        = psi_air[m]
            r0        = r_store[0, m]
            d_air     = 1.0 + r0 * pa
            adj[m]    = (1.0 - pa**2)  / d_air**2
            dpsi_dphi = (1.0 - r0**2)  / d_air**2
            dpsi_dG0  = -2.0 * lam[m] / (lam[m] + Gamma[0, m])**2
            dr_te[0, m] += dpsi_dphi * dpsi_dG0 * dGamma[0, m]

        for j in range(n_lay - 1):
            for m in range(n_lam):
                rb   = r_store[j+1, m];  ej   = exp_store[j, m]
                Gj   = Gamma[j, m];      Gjp1 = Gamma[j+1, m]
                sG   = Gj + Gjp1;        psi  = (Gj - Gjp1) / sG
                num  = rb + psi;         den  = 1.0 + rb * psi
                dpsi_dGj   =  2.0 * Gjp1 / sG**2
                dpsi_dGjp1 = -2.0 * Gj   / sG**2
                dr_dpsi    = ej * (1.0 - rb**2) / den**2
                dr_dGj     = (-2.0 * thicknesses[j] * ej) * num / den + dr_dpsi * dpsi_dGj
                dr_dGjp1   = dr_dpsi * dpsi_dGjp1
                dr_drbelow = ej * (1.0 - psi**2) / den**2
                dr_te[j,   m] += adj[m] * dr_dGj   * dGamma[j,   m]
                dr_te[j+1, m] += adj[m] * dr_dGjp1 * dGamma[j+1, m]
                adj[m] = adj[m] * dr_drbelow

        return r_te, dr_te

    @nb.njit(**_NB_OPTS)
    def _tem_circular_grad_jit(times, thicknesses, resistivities,
                               lam, lam_hj1, mu0,
                               fourier_base, fourier_weights,
                               filter_weights):
        """Circle dBz/dt + analytical Jacobian (Numba JIT, DLF).

        Works for circle_central and circle_offset — only lam_hj1 differs:
            circle_central : lam_hj1 = lam * h_j1
            circle_offset  : lam_hj1 = lam * J0(lam * rx_offset) * h_j1

        filter_weights : (n_t, n_f) complex128
            Pre-evaluated system filter H(omega_ki).
            Pass np.ones((n_t, n_f), complex) when no filter is needed.
            H(omega) multiplies the complex kernel before the imaginary part
            is taken; since H is independent of resistivity, it applies
            equally to the forward response and every layer's gradient.

        Returns dbdt (n_t,) and J_raw (n_t, N) before log-log conversion.
        """
        n_t   = len(times)
        n_f   = len(fourier_base)
        n_lam = len(lam)
        n_lay = len(resistivities)
        dbdt  = np.zeros(n_t)
        J_raw = np.zeros((n_t, n_lay))

        for i in range(n_t):
            t       = times[i]
            hz_acc  = 0.0
            dhz_acc = np.zeros(n_lay)
            for k in range(n_f):
                omega = fourier_base[k] / t
                fw    = filter_weights[i, k]   # H(omega) for this gate/frequency
                r_te, dr_te = _te_rte_grad_jit(
                    lam, omega, thicknesses, resistivities, mu0)
                hz_c = 0.0 + 0.0j
                for m in range(n_lam):
                    hz_c += r_te[m] * lam_hj1[m]
                hz_c *= 0.5 * fw            # apply filter before taking imaginary part
                hz_acc += mu0 * hz_c.imag * fourier_weights[k]
                for j in range(n_lay):
                    dhz_c = 0.0 + 0.0j
                    for m in range(n_lam):
                        dhz_c += dr_te[j, m] * lam_hj1[m]
                    dhz_c      *= 0.5 * fw  # same H(omega): independent of resistivity
                    dhz_acc[j] += mu0 * dhz_c.imag * fourier_weights[k]
            dbdt[i]     = hz_acc  / t
            J_raw[i, :] = dhz_acc / t
        return dbdt, J_raw

    @nb.njit(**_NB_OPTS)
    def _tem_square_grad_jit(times, thicknesses, resistivities,
                             rho_q, area_w, quad_scale,
                             h_base, h_j0, mu0,
                             fourier_base, fourier_weights,
                             filter_weights):
        """Square-loop dBz/dt + analytical Jacobian (Numba JIT, DLF).

        Works for square_central (one-quadrant GL with quad_scale=4.0) and
        square_offset (full-square GL with quad_scale=1.0).

        filter_weights : (n_t, n_f) complex128 — see _tem_circular_grad_jit.
        Returns dbdt (n_t,) and J_raw (n_t, N) before log-log conversion.
        """
        n_t   = len(times)
        n_f   = len(fourier_base)
        n_q   = len(rho_q)
        n_lam = len(h_base)
        n_lay = len(resistivities)
        _4pi  = 4.0 * np.pi
        dbdt  = np.zeros(n_t)
        J_raw = np.zeros((n_t, n_lay))

        lam_q  = np.empty(n_lam, dtype=np.float64)
        kern_q = np.empty(n_lam, dtype=np.float64)

        for i in range(n_t):
            t       = times[i]
            hz_acc  = 0.0
            dhz_acc = np.zeros(n_lay)
            for k in range(n_f):
                omega = fourier_base[k] / t
                fw    = filter_weights[i, k]   # H(omega)
                hz_f  = 0.0 + 0.0j
                dhz_f = np.zeros(n_lay, dtype=np.complex128)
                for q in range(n_q):
                    rq = rho_q[q];  wq = area_w[q]
                    for m in range(n_lam):
                        lm        = h_base[m] / rq
                        lam_q[m]  = lm
                        kern_q[m] = lm * lm * h_j0[m] / (rq * _4pi)
                    r_te_q, dr_te_q = _te_rte_grad_jit(
                        lam_q, omega, thicknesses, resistivities, mu0)
                    hz_c = 0.0 + 0.0j
                    for m in range(n_lam):
                        hz_c += r_te_q[m] * kern_q[m]
                    hz_f += wq * hz_c
                    for j in range(n_lay):
                        dhz_c = 0.0 + 0.0j
                        for m in range(n_lam):
                            dhz_c += dr_te_q[j, m] * kern_q[m]
                        dhz_f[j] += wq * dhz_c
                # Apply quad_scale and filter, then accumulate
                hz_f  *= quad_scale * fw   # H(omega) independent of resistivity
                dhz_f *= quad_scale * fw
                hz_acc += mu0 * hz_f.imag * fourier_weights[k]
                for j in range(n_lay):
                    dhz_acc[j] += mu0 * dhz_f[j].imag * fourier_weights[k]
            dbdt[i]     = hz_acc  / t
            J_raw[i, :] = dhz_acc / t
        return dbdt, J_raw

    @nb.njit(**_NB_OPTS)
    def _tem_circular_grad_euler_jit(times, thicknesses, resistivities,
                                     lam, lam_hj1, mu0, e_eta, e_A,
                                     filter_weights):
        """Circle dBz/dt + analytical Jacobian (Numba JIT, Euler–Stehfest).

        Uses complex Bromwich frequencies omega_k = k*pi/t - (A/2t)*i.
        _te_rte_grad_jit handles complex omega natively — no separate
        implementation is needed; the recursion is analytic in omega.

        filter_weights : (n_t, n_eval) complex128
            H(omega_k) for each gate and Euler term.
            Pass np.ones((n_t, n_eval), complex) when no filter is needed.
            Applied before taking the real part (Euler accumulation).

        Returns dbdt (n_t,) and J_raw (n_t, N) with step-off sign applied.
        """
        n_t    = len(times)
        n_eval = len(e_eta)
        n_lam  = len(lam)
        n_lay  = len(resistivities)
        dbdt   = np.zeros(n_t)
        J_raw  = np.zeros((n_t, n_lay))

        for i in range(n_t):
            t      = times[i]
            c      = e_A / (2.0 * t)
            h_step = np.pi / t
            hz_acc  = 0.0
            dhz_acc = np.zeros(n_lay)
            for k in range(n_eval):
                omega  = k * h_step - c * 1j   # complex Bromwich frequency
                sign_k = (-1.0)**k * e_eta[k]
                fw     = filter_weights[i, k]  # H(omega_k)
                r_te, dr_te = _te_rte_grad_jit(
                    lam, omega, thicknesses, resistivities, mu0)
                hz_c = 0.0 + 0.0j
                for m in range(n_lam):
                    hz_c += r_te[m] * lam_hj1[m]
                hz_c *= 0.5 * fw            # apply filter before taking real part
                hz_acc += sign_k * mu0 * hz_c.real
                for j in range(n_lay):
                    dhz_c = 0.0 + 0.0j
                    for m in range(n_lam):
                        dhz_c += dr_te[j, m] * lam_hj1[m]
                    dhz_c      *= 0.5 * fw  # H(omega) independent of resistivity
                    dhz_acc[j] += sign_k * mu0 * dhz_c.real
            prefac      = np.exp(e_A / 2.0) / t
            dbdt[i]     = -prefac * hz_acc    # -1 for step-off
            J_raw[i, :] = -prefac * dhz_acc
        return dbdt, J_raw

    @nb.njit(**_NB_OPTS)
    def _tem_square_grad_euler_jit(times, thicknesses, resistivities,
                                   rho_q, area_w, quad_scale,
                                   h_base, h_j0, mu0, e_eta, e_A,
                                   filter_weights):
        """Square-loop dBz/dt + analytical Jacobian (Numba JIT, Euler–Stehfest).

        filter_weights : (n_t, n_eval) complex128 — see _tem_circular_grad_euler_jit.
        Returns dbdt (n_t,) and J_raw (n_t, N) with step-off sign applied.
        """
        n_t    = len(times)
        n_eval = len(e_eta)
        n_q    = len(rho_q)
        n_lam  = len(h_base)
        n_lay  = len(resistivities)
        _4pi   = 4.0 * np.pi
        dbdt   = np.zeros(n_t)
        J_raw  = np.zeros((n_t, n_lay))

        lam_q  = np.empty(n_lam, dtype=np.float64)
        kern_q = np.empty(n_lam, dtype=np.float64)

        for i in range(n_t):
            t      = times[i]
            c      = e_A / (2.0 * t)
            h_step = np.pi / t
            hz_acc  = 0.0
            dhz_acc = np.zeros(n_lay)
            for k in range(n_eval):
                omega  = k * h_step - c * 1j
                sign_k = (-1.0)**k * e_eta[k]
                fw     = filter_weights[i, k]   # H(omega_k)
                hz_f  = 0.0 + 0.0j
                dhz_f = np.zeros(n_lay, dtype=np.complex128)
                for q in range(n_q):
                    rq = rho_q[q];  wq = area_w[q]
                    for m in range(n_lam):
                        lm        = h_base[m] / rq
                        lam_q[m]  = lm
                        kern_q[m] = lm * lm * h_j0[m] / (rq * _4pi)
                    r_te_q, dr_te_q = _te_rte_grad_jit(
                        lam_q, omega, thicknesses, resistivities, mu0)
                    hz_c = 0.0 + 0.0j
                    for m in range(n_lam):
                        hz_c += r_te_q[m] * kern_q[m]
                    hz_f += wq * hz_c
                    for j in range(n_lay):
                        dhz_c = 0.0 + 0.0j
                        for m in range(n_lam):
                            dhz_c += dr_te_q[j, m] * kern_q[m]
                        dhz_f[j] += wq * dhz_c
                hz_f  *= quad_scale * fw   # H(omega) independent of resistivity
                dhz_f *= quad_scale * fw
                hz_acc  += sign_k * mu0 * hz_f.real
                for j in range(n_lay):
                    dhz_acc[j] += sign_k * mu0 * dhz_f[j].real
            prefac      = np.exp(e_A / 2.0) / t
            dbdt[i]     = -prefac * hz_acc
            J_raw[i, :] = -prefac * dhz_acc
        return dbdt, J_raw


# ============================================================================
# CuPy GPU kernels
# ============================================================================
#
# The full (n_t, n_f, K) frequency × wavenumber tensor is processed in a
# single batched CuPy operation.  CUDA throughput depends on keeping many
# warps in flight simultaneously; the large (n_t, n_f, K) batch saturates
# GPU occupancy.  Per-frequency Python loops would serialise kernel launches
# and leave most warps idle.
#
# System filter: d_filter_weights (n_t, n_f) cupy complex128, pre-evaluated
# on the GPU.  For DLF: (hz * d_filter_weights).imag  before the f_sin dot.
# For Euler: (hz * d_filter_weights).real  before summing Euler coefficients.
# ============================================================================

if HAS_CUDA:
    import cupy as cp

    def _te_reflection_coeff_grad_gpu(d_lam, omega_2d, d_thicknesses, d_resistivities):
        """Batched TE gradient on GPU — (n_t, n_f, K) tensor.

        Parameters
        ----------
        d_lam           : (K,)        cupy float64
        omega_2d        : (n_t, n_f)  cupy complex128 — real or complex omega
        d_thicknesses   : (N-1,)      cupy float64
        d_resistivities : (N,)        cupy float64

        Returns
        -------
        r_TE  : (n_t, n_f, K)    cupy complex128
        dr_TE : (N, n_t, n_f, K) cupy complex128
        """
        n_lay    = len(d_resistivities)
        n_t, n_f = omega_2d.shape
        K        = len(d_lam)
        d_sigma  = 1.0 / d_resistivities                             # (N,)
        sval     = 1j * omega_2d[:, :, None]                         # (n_t, n_f, 1)

        Gamma = cp.sqrt(
            d_lam[None, None, :]**2
            + sval * MU0 * d_sigma[:, None, None, None])             # (N, n_t, n_f, K)

        sval_4d       = 1j * omega_2d[None, :, :, None]              # (1, n_t, n_f, 1)
        dGamma_dlnrho = (-sval_4d * MU0
                         * d_sigma[:, None, None, None]
                         / (2.0 * Gamma))                             # (N, n_t, n_f, K)

        r_store   = cp.zeros((n_lay, n_t, n_f, K), dtype=cp.complex128)
        exp_store = cp.zeros((n_lay - 1, n_t, n_f, K), dtype=cp.complex128)
        r         = cp.zeros((n_t, n_f, K), dtype=cp.complex128)
        r_store[n_lay - 1] = r

        for j in range(n_lay - 2, -1, -1):
            psi          = (Gamma[j] - Gamma[j+1]) / (Gamma[j] + Gamma[j+1])
            exp_j        = cp.exp(-2.0 * Gamma[j] * d_thicknesses[j])
            exp_store[j] = exp_j
            r            = exp_j * (r + psi) / (1.0 + r * psi)
            r_store[j]   = r

        psi_air   = (d_lam[None, None, :] - Gamma[0]) / (d_lam[None, None, :] + Gamma[0])
        r_TE      = (r_store[0] + psi_air) / (1.0 + r_store[0] * psi_air)

        denom_air      = 1.0 + r_store[0] * psi_air
        dr_TE_dr0      = (1.0 - psi_air**2)    / denom_air**2
        dr_TE_dpsi_air = (1.0 - r_store[0]**2) / denom_air**2
        dpsi_air_dG0   = (-2.0 * d_lam[None, None, :]
                          / (d_lam[None, None, :] + Gamma[0])**2)

        dr_TE_all = cp.zeros((n_lay, n_t, n_f, K), dtype=cp.complex128)
        dr_TE_all[0] += dr_TE_dpsi_air * dpsi_air_dG0 * dGamma_dlnrho[0]
        adj = dr_TE_dr0.copy()

        for j in range(n_lay - 1):
            r_below    = r_store[j+1];  exp_j      = exp_store[j]
            psi_j      = (Gamma[j] - Gamma[j+1]) / (Gamma[j] + Gamma[j+1])
            numer      = r_below + psi_j;  denom = 1.0 + r_below * psi_j
            dpsi_dGj   =  2.0 * Gamma[j+1] / (Gamma[j] + Gamma[j+1])**2
            dpsi_dGjp1 = -2.0 * Gamma[j]   / (Gamma[j] + Gamma[j+1])**2
            dr_dpsi    = exp_j * (1.0 - r_below**2) / denom**2
            dr_dGj     = (-2.0 * d_thicknesses[j] * exp_j * numer / denom
                          + dr_dpsi * dpsi_dGj)
            dr_dGjp1   = dr_dpsi * dpsi_dGjp1
            dr_drbelow = exp_j * (1.0 - psi_j**2) / denom**2
            dr_TE_all[j]   += adj * dr_dGj   * dGamma_dlnrho[j]
            dr_TE_all[j+1] += adj * dr_dGjp1 * dGamma_dlnrho[j+1]
            adj = adj * dr_drbelow

        return r_TE, dr_TE_all

    def _tem_circular_grad_gpu(times, thicknesses, resistivities, tx_radius,
                               lam_hj1, d_h_base, d_f_base, d_f_sin,
                               d_filter_weights):
        """Circle dBz/dt + analytical Jacobian on GPU (DLF).

        Works for circle_central and circle_offset via lam_hj1.

        d_filter_weights : (n_t, n_f) cupy complex128
            H(omega_ki) pre-evaluated on GPU.
            Pass cp.ones((n_t, n_f), dtype=cp.complex128) when no filter is needed.

        Returns (dbdt, J_raw) as NumPy arrays, shapes (n_t,) and (n_t, N).
        """
        d_times   = cp.asarray(times)
        d_thick   = cp.asarray(thicknesses, dtype=cp.float64)
        d_rho     = cp.asarray(resistivities, dtype=cp.float64)
        d_lam     = d_h_base / float(tx_radius)
        d_lam_hj1 = cp.asarray(lam_hj1)
        omega_2d  = d_f_base[None, :] / d_times[:, None]   # (n_t, n_f) real

        r_te, dr_te = _te_reflection_coeff_grad_gpu(d_lam, omega_2d, d_thick, d_rho)
        # r_te: (n_t, n_f, K),  dr_te: (N, n_t, n_f, K)

        hz  = 0.5 * cp.sum(r_te  * d_lam_hj1[None, None, :],           axis=2)  # (n_t, n_f)
        dhz = 0.5 * cp.sum(dr_te * d_lam_hj1[None, None, None, :], axis=3)      # (N, n_t, n_f)

        # Apply system filter: H(omega) multiplies complex Hz before imaginary part
        sig  = MU0 * (hz  * d_filter_weights).imag              # (n_t, n_f)
        dsig = MU0 * (dhz * d_filter_weights[None, :, :]).imag  # (N, n_t, n_f)

        dbdt  = cp.sum(sig  * d_f_sin[None, :],           axis=1) / d_times          # (n_t,)
        J_raw = (cp.sum(dsig * d_f_sin[None, None, :], axis=2) / d_times[None, :]).T  # (n_t, N)
        return cp.asnumpy(dbdt), cp.asnumpy(J_raw)

    def _tem_square_grad_gpu(times, thicknesses, resistivities,
                             rho_q, area_w, quad_scale,
                             d_h_base, d_h_j0, d_f_base, d_f_sin,
                             d_filter_weights):
        """Square-loop dBz/dt + analytical Jacobian on GPU (DLF).

        Loops over n_q quadrature points; each iteration runs the full
        (n_t, n_f, K) adjoint on the GPU then accumulates with area weight.

        d_filter_weights : (n_t, n_f) cupy complex128 — see _tem_circular_grad_gpu.
        Returns (dbdt, J_raw) as NumPy arrays, shapes (n_t,) and (n_t, N).
        """
        d_times  = cp.asarray(times)
        d_thick  = cp.asarray(thicknesses, dtype=cp.float64)
        d_rho    = cp.asarray(resistivities, dtype=cp.float64)
        n_t      = len(times)
        n_lay    = len(resistivities)
        omega_2d = d_f_base[None, :] / d_times[:, None]   # (n_t, n_f)
        _4pi     = 4.0 * np.pi

        d_dbdt = cp.zeros(n_t,          dtype=cp.float64)
        d_Jraw = cp.zeros((n_t, n_lay), dtype=cp.float64)

        for q in range(len(rho_q)):
            rq       = float(rho_q[q]);  wq = float(area_w[q])
            d_lam_q  = d_h_base / rq
            d_kern_q = d_lam_q**2 * d_h_j0 / (rq * _4pi)

            r_te, dr_te = _te_reflection_coeff_grad_gpu(
                d_lam_q, omega_2d, d_thick, d_rho)
            # r_te: (n_t, n_f, K),  dr_te: (N, n_t, n_f, K)

            hz  = cp.sum(r_te  * d_kern_q[None, None, :],           axis=2)  # (n_t, n_f)
            dhz = cp.sum(dr_te * d_kern_q[None, None, None, :], axis=3)      # (N, n_t, n_f)

            # Apply system filter before imaginary part
            sig  = MU0 * (hz  * d_filter_weights).imag              # (n_t, n_f)
            dsig = MU0 * (dhz * d_filter_weights[None, :, :]).imag  # (N, n_t, n_f)

            d_dbdt += wq * cp.sum(sig  * d_f_sin[None, :],           axis=1) / d_times
            d_Jraw += wq * (cp.sum(dsig * d_f_sin[None, None, :], axis=2) / d_times[None, :]).T

        d_dbdt *= quad_scale
        d_Jraw *= quad_scale
        return cp.asnumpy(d_dbdt), cp.asnumpy(d_Jraw)

    def _tem_circular_grad_euler_gpu(times, thicknesses, resistivities, tx_radius,
                                     lam_hj1, d_h_base, e_eta, e_A,
                                     d_filter_weights):
        """Circle dBz/dt + analytical Jacobian on GPU (Euler–Stehfest).

        _te_reflection_coeff_grad_gpu handles complex omega_2d natively, so
        no separate Euler kernel is needed — the same GPU adjoint recursion
        works for both real (DLF) and complex (Euler) frequencies.

        d_filter_weights : (n_t, n_eval) cupy complex128
            H(omega_k) for each gate and Euler term.
            Applied before taking the real part (Euler accumulation).

        Returns (dbdt, J_raw) as NumPy arrays, shapes (n_t,) and (n_t, N).
        """
        d_times   = cp.asarray(times)
        d_thick   = cp.asarray(thicknesses, dtype=cp.float64)
        d_rho     = cp.asarray(resistivities, dtype=cp.float64)
        d_lam     = d_h_base / float(tx_radius)
        d_lam_hj1 = cp.asarray(lam_hj1)
        n_eval    = len(e_eta)
        k_arr     = cp.arange(n_eval, dtype=cp.float64)
        c_vals    = float(e_A) / (2.0 * d_times)
        h_vals    = cp.full(len(times), np.pi) / d_times
        omega_2d  = k_arr[None, :] * h_vals[:, None] - c_vals[:, None] * 1j  # (n_t, n_eval)

        r_te, dr_te = _te_reflection_coeff_grad_gpu(d_lam, omega_2d, d_thick, d_rho)
        # r_te: (n_t, n_eval, K),  dr_te: (N, n_t, n_eval, K)

        hz  = 0.5 * cp.sum(r_te  * d_lam_hj1[None, None, :],           axis=2)  # (n_t, n_eval)
        dhz = 0.5 * cp.sum(dr_te * d_lam_hj1[None, None, None, :], axis=3)      # (N, n_t, n_eval)

        signs_k = cp.asarray((-1.0)**np.arange(n_eval) * e_eta)   # (n_eval,)

        # Apply system filter before taking real part, then dot with Euler coefficients
        hz_filt  = hz  * d_filter_weights                    # (n_t, n_eval)
        dhz_filt = dhz * d_filter_weights[None, :, :]        # (N, n_t, n_eval)

        hz_acc  = MU0 * cp.sum(signs_k[None, :]       * hz_filt.real,  axis=1)  # (n_t,)
        dhz_acc = MU0 * cp.sum(signs_k[None, None, :] * dhz_filt.real, axis=2)  # (N, n_t)

        prefac = cp.exp(float(e_A) / 2.0) / d_times              # (n_t,)
        dbdt   = -prefac * hz_acc                                  # step-off sign
        J_raw  = (-prefac[None, :] * dhz_acc).T                   # (n_t, N)
        return cp.asnumpy(dbdt), cp.asnumpy(J_raw)

    def _tem_square_grad_euler_gpu(times, thicknesses, resistivities,
                                   rho_q, area_w, quad_scale,
                                   d_h_base, d_h_j0, e_eta, e_A,
                                   d_filter_weights):
        """Square-loop dBz/dt + analytical Jacobian on GPU (Euler–Stehfest).

        d_filter_weights : (n_t, n_eval) cupy complex128 — see _tem_circular_grad_euler_gpu.
        Returns (dbdt, J_raw) as NumPy arrays, shapes (n_t,) and (n_t, N).
        """
        d_times  = cp.asarray(times)
        d_thick  = cp.asarray(thicknesses, dtype=cp.float64)
        d_rho    = cp.asarray(resistivities, dtype=cp.float64)
        n_t      = len(times)
        n_lay    = len(resistivities)
        n_eval   = len(e_eta)
        _4pi     = 4.0 * np.pi
        k_arr    = cp.arange(n_eval, dtype=cp.float64)
        c_vals   = float(e_A) / (2.0 * d_times)
        h_vals   = cp.full(n_t, np.pi) / d_times
        omega_2d = k_arr[None, :] * h_vals[:, None] - c_vals[:, None] * 1j  # (n_t, n_eval)
        signs_k  = cp.asarray((-1.0)**np.arange(n_eval) * e_eta)

        d_hz_acc  = cp.zeros(n_t,          dtype=cp.float64)
        d_dhz_acc = cp.zeros((n_t, n_lay), dtype=cp.float64)

        for q in range(len(rho_q)):
            rq       = float(rho_q[q]);  wq = float(area_w[q])
            d_lam_q  = d_h_base / rq
            d_kern_q = d_lam_q**2 * d_h_j0 / (rq * _4pi)

            r_te, dr_te = _te_reflection_coeff_grad_gpu(d_lam_q, omega_2d, d_thick, d_rho)
            # r_te: (n_t, n_eval, K),  dr_te: (N, n_t, n_eval, K)

            hz  = cp.sum(r_te  * d_kern_q[None, None, :],           axis=2)  # (n_t, n_eval)
            dhz = cp.sum(dr_te * d_kern_q[None, None, None, :], axis=3)      # (N, n_t, n_eval)

            # Apply system filter before taking real part
            hz_filt  = hz  * d_filter_weights                 # (n_t, n_eval)
            dhz_filt = dhz * d_filter_weights[None, :, :]    # (N, n_t, n_eval)

            d_hz_acc  += wq * MU0 * cp.sum(signs_k[None, :]       * hz_filt.real,  axis=1)
            d_dhz_acc += wq * (MU0 * cp.sum(signs_k[None, None, :] * dhz_filt.real, axis=2)).T

        d_hz_acc  *= quad_scale
        d_dhz_acc *= quad_scale
        prefac     = cp.exp(float(e_A) / 2.0) / d_times
        d_dbdt     = -prefac * d_hz_acc
        d_Jraw     = -(prefac[:, None] * d_dhz_acc)
        return cp.asnumpy(d_dbdt), cp.asnumpy(d_Jraw)
