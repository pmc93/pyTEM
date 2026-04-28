"""
kernels_gpu.py — CuPy/CUDA GPU kernels for TEM forward modelling.

Contains:
  - _te_reflection_coeff_gpu  : Wait recursion, batched complex
  - _tem_circular_gpu         : central/offset circular loop (Fourier DLF)
  - _tem_square_gpu           : square loop (Fourier DLF)
  - _tem_circular_euler_gpu   : central/offset (Euler acceleration)
  - _tem_square_euler_gpu     : square loop (Euler acceleration)

All kernels accept an optional filter_weights parameter (n_t, n_eval)
complex128. When provided, Hz(omega) is multiplied by H(omega) before
the final transform. Pass None when no system filter is needed.
"""

from .filters import MU0
from .backends import HAS_CUDA

if HAS_CUDA:
    import cupy as cp
    import numpy as np

    def _te_reflection_coeff_gpu(d_lam, omega_2d, d_thicknesses, d_resistivities):
        """
        GPU-batched TE reflection coefficient via Wait recursion (complex).
        Works with both real omega (DLF) and complex omega (Euler/Bromwich).
        """
        n_lay = len(d_resistivities)
        sigma = 1.0 / d_resistivities
        sval = 1j * omega_2d

        lam2 = d_lam ** 2
        prod = (MU0 * sigma)[:, None, None] * sval[None, :, :]
        Gamma = cp.sqrt(lam2[None, None, None, :] + prod[:, :, :, None])

        r = cp.zeros((*omega_2d.shape, len(d_lam)), dtype=cp.complex128)
        for j in range(n_lay - 2, -1, -1):
            psi = (Gamma[j] - Gamma[j + 1]) / (Gamma[j] + Gamma[j + 1])
            exp_term = cp.exp(-2.0 * Gamma[j] * d_thicknesses[j])
            r = exp_term * (r + psi) / (1.0 + r * psi)

        psi_air = (d_lam[None, None, :] - Gamma[0]) / (d_lam[None, None, :] + Gamma[0])
        r_TE = (r + psi_air) / (1.0 + r * psi_air)
        return r_TE

    # ------------------------------------------------------------------
    # Fourier DLF GPU — circular (central + offset unified)
    # ------------------------------------------------------------------
    def _tem_circular_gpu(times, thicknesses, resistivities, tx_radius,
                          extra_weights, d_h_base, d_h_j1, d_f_base, d_f_sin,
                          filter_weights=None):
        """Circular-loop dBz/dt fully on GPU (Fourier DLF)."""
        d_times = cp.asarray(times)
        d_thick = cp.asarray(thicknesses, dtype=cp.float64)
        d_rho = cp.asarray(resistivities, dtype=cp.float64)
        a = float(tx_radius)

        d_lam = d_h_base / a
        omega_2d = d_f_base[None, :] / d_times[:, None]
        r_te = _te_reflection_coeff_gpu(d_lam, omega_2d, d_thick, d_rho)

        kernel = r_te * d_lam[None, None, :] * extra_weights[None, None, :]
        hz = 0.5 * cp.sum(kernel * d_h_j1[None, None, :], axis=2)

        if filter_weights is not None:
            hz = hz * cp.asarray(filter_weights)

        sig = MU0 * hz.imag
        dbdt = cp.sum(sig * d_f_sin[None, :], axis=1) / d_times
        return cp.asnumpy(dbdt)

    # ------------------------------------------------------------------
    # Fourier DLF GPU — square (VMD area integral)
    # ------------------------------------------------------------------
    def _tem_square_gpu(times, thicknesses, resistivities, offset_dist_q, area_w,
                        d_h_base, d_h_j0, d_f_base, d_f_sin,
                        filter_weights=None):
        """Square-loop dBz/dt fully on GPU (Fourier DLF + VMD area integral)."""
        n_q = len(offset_dist_q)
        d_times = cp.asarray(times)
        d_thick = cp.asarray(thicknesses, dtype=cp.float64)
        d_rho_lay = cp.asarray(resistivities, dtype=cp.float64)
        d_offset_dist_q = cp.asarray(offset_dist_q)
        d_area_w = cp.asarray(area_w)

        omega_2d = d_f_base[None, :] / d_times[:, None]
        n_f = len(d_f_base)
        hz_total = cp.zeros((len(times), n_f), dtype=cp.complex128)

        for q in range(n_q):
            dist = d_offset_dist_q[q]
            d_lam = d_h_base / dist
            r_te = _te_reflection_coeff_gpu(d_lam, omega_2d, d_thick, d_rho_lay)

            kernel = r_te * (d_lam ** 2)[None, None, :]
            g = cp.sum(kernel * d_h_j0[None, None, :], axis=2) / dist / (4.0 * cp.pi)
            hz_total += float(d_area_w[q]) * g

        if filter_weights is not None:
            hz_total = hz_total * cp.asarray(filter_weights)

        sig = MU0 * hz_total.imag
        dbdt = cp.sum(sig * d_f_sin[None, :], axis=1) / d_times
        return cp.asnumpy(dbdt)

    # ------------------------------------------------------------------
    # Euler GPU — circular (central + offset unified)
    # ------------------------------------------------------------------
    def _tem_circular_euler_gpu(times, thicknesses, resistivities, tx_radius,
                                extra_weights, d_h_base, d_h_j1,
                                euler_eta, euler_A, filter_weights=None):
        """Circular-loop dBz/dt on GPU via Euler-accelerated Bromwich inversion."""
        d_times = cp.asarray(times)
        d_thick = cp.asarray(thicknesses, dtype=cp.float64)
        d_rho = cp.asarray(resistivities, dtype=cp.float64)
        a = float(tx_radius)

        d_lam = d_h_base / a
        n_euler = len(euler_eta)
        d_eta = cp.asarray(euler_eta)
        half_A = euler_A / 2.0

        ks = cp.arange(n_euler, dtype=cp.float64)
        c = half_A / d_times
        h = cp.pi / d_times
        s_2d = c[:, None] + ks[None, :] * h[:, None] * 1j
        omega_2d = s_2d / 1j

        r_te = _te_reflection_coeff_gpu(d_lam, omega_2d, d_thick, d_rho)

        kernel = r_te * d_lam[None, None, :] * extra_weights[None, None, :]
        hz = 0.5 * cp.sum(kernel * d_h_j1[None, None, :], axis=2)

        if filter_weights is not None:
            hz = hz * cp.asarray(filter_weights)

        fvals = (MU0 * hz).real
        signs = cp.array([(-1.0)**k for k in range(n_euler)])
        dbdt = cp.exp(half_A) / d_times * cp.sum(d_eta[None, :] * signs[None, :] * fvals, axis=1)

        return cp.asnumpy(dbdt)

    # ------------------------------------------------------------------
    # Euler GPU — square (VMD area integral)
    # ------------------------------------------------------------------
    def _tem_square_euler_gpu(times, thicknesses, resistivities, offset_dist_q, area_w,
                              d_h_base, d_h_j0,
                              euler_eta, euler_A, filter_weights=None):
        """Square-loop dBz/dt on GPU via Euler-accelerated Bromwich + VMD integral."""
        n_q = len(offset_dist_q)
        d_times = cp.asarray(times)
        d_thick = cp.asarray(thicknesses, dtype=cp.float64)
        d_rho_lay = cp.asarray(resistivities, dtype=cp.float64)
        n_euler = len(euler_eta)
        d_eta = cp.asarray(euler_eta)
        half_A = euler_A / 2.0

        ks = cp.arange(n_euler, dtype=cp.float64)
        c = half_A / d_times
        h = cp.pi / d_times
        s_2d = c[:, None] + ks[None, :] * h[:, None] * 1j
        omega_2d = s_2d / 1j

        n_t = len(times)
        hz_total = cp.zeros((n_t, n_euler), dtype=cp.complex128)

        for q in range(n_q):
            dist = float(offset_dist_q[q])
            d_lam = d_h_base / dist
            r_te = _te_reflection_coeff_gpu(d_lam, omega_2d, d_thick, d_rho_lay)
            kernel = r_te * (d_lam ** 2)[None, None, :]
            g = cp.sum(kernel * d_h_j0[None, None, :], axis=2) / dist / (4.0 * cp.pi)
            hz_total += float(area_w[q]) * g

        if filter_weights is not None:
            hz_total = hz_total * cp.asarray(filter_weights)

        fvals = (MU0 * hz_total).real
        signs = cp.array([(-1.0)**k for k in range(n_euler)])
        dbdt = cp.exp(half_A) / d_times * cp.sum(d_eta[None, :] * signs[None, :] * fvals, axis=1)

        return cp.asnumpy(dbdt)
