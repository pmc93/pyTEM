"""
kernels_numba.py — Numba JIT kernels for TEM forward modelling.

Contains:
  - _te_rte_jit          : Wait recursion (complex), scalar loops
  - _tem_circular_jit    : central/offset circular loop (Fourier DLF)
  - _tem_square_jit      : square loop via VMD area integral (Fourier DLF)
  - _tem_circular_euler_jit : central/offset (Euler acceleration)
  - _tem_square_euler_jit   : square loop (Euler acceleration)

All kernels accept a filter_weights array (n_t, n_eval) complex128.
When no system filter is needed, pass np.ones((n_t, n_eval), complex128).
"""

import numpy as np

try:
    import numba as nb
    HAS_NUMBA = True
    _NB_OPTS = {'nogil': True, 'cache': True}
except ImportError:
    HAS_NUMBA = False

if HAS_NUMBA:

    @nb.njit(**_NB_OPTS)
    def _te_rte_jit(lam, omega, thicknesses, resistivities, mu0):
        """JIT-compiled TE reflection coefficient (Wait recursion)."""
        n_lay = len(resistivities)
        n_lam = len(lam)
        sval = 1j * omega

        Gamma = np.empty((n_lay, n_lam), dtype=np.complex128)
        for j in range(n_lay):
            sigma_j = 1.0 / resistivities[j]
            prod = sval * mu0 * sigma_j
            for m in range(n_lam):
                Gamma[j, m] = np.sqrt(lam[m]**2 + prod)

        r = np.zeros(n_lam, dtype=np.complex128)
        for j in range(n_lay - 2, -1, -1):
            h = thicknesses[j]
            for m in range(n_lam):
                psi = (Gamma[j, m] - Gamma[j + 1, m]) / (Gamma[j, m] + Gamma[j + 1, m])
                exp_term = np.exp(-2.0 * Gamma[j, m] * h)
                r[m] = exp_term * (r[m] + psi) / (1.0 + r[m] * psi)

        r_te = np.empty(n_lam, dtype=np.complex128)
        for m in range(n_lam):
            psi_air = (lam[m] - Gamma[0, m]) / (lam[m] + Gamma[0, m])
            r_te[m] = (r[m] + psi_air) / (1.0 + r[m] * psi_air)
        return r_te

    # ------------------------------------------------------------------
    # Circular loop: central + offset (Fourier DLF)
    # ------------------------------------------------------------------
    @nb.njit(**_NB_OPTS)
    def _tem_circular_jit(times, thicknesses, resistivities,
                          tx_radius, extra_weights, mu0,
                          hankel_base, hankel_j1,
                          fourier_base, fourier_weights,
                          filter_weights):
        """Circular-loop dBz/dt via fused Numba loops (Fourier DLF)."""
        n_t = len(times)
        n_f = len(fourier_base)
        a = tx_radius
        lam = hankel_base / a
        n_lam = len(lam)
        dbdt = np.empty(n_t)

        for i in range(n_t):
            t = times[i]
            accum = 0.0
            for k in range(n_f):
                omega = fourier_base[k] / t
                r_te = _te_rte_jit(lam, omega, thicknesses,
                                   resistivities, mu0)
                hz_c = 0.0 + 0.0j
                for m in range(n_lam):
                    hz_c += r_te[m] * lam[m] * hankel_j1[m] * extra_weights[m]
                hz_c *= 0.5
                hz_im = (hz_c * filter_weights[i, k]).imag
                accum += mu0 * hz_im * fourier_weights[k]
            dbdt[i] = accum / t
        return dbdt

    # ------------------------------------------------------------------
    # Square loop: VMD area integral (Fourier DLF)
    # ------------------------------------------------------------------
    @nb.njit(**_NB_OPTS)
    def _tem_square_jit(times, thicknesses, resistivities,
                        rho_q, area_w, mu0,
                        hankel_base, hankel_j0,
                        fourier_base, fourier_weights,
                        filter_weights):
        """Square-loop dBz/dt via VMD area integral (Fourier DLF, Numba)."""
        n_t = len(times)
        n_f = len(fourier_base)
        n_q = len(rho_q)
        n_lam = len(hankel_base)
        dbdt = np.empty(n_t)

        for i in range(n_t):
            t = times[i]
            accum_t = 0.0
            for k in range(n_f):
                omega = fourier_base[k] / t
                hz_c = 0.0 + 0.0j

                for q in range(n_q):
                    rho = rho_q[q]
                    lam = np.empty(n_lam)
                    for m in range(n_lam):
                        lam[m] = hankel_base[m] / rho

                    r_te = _te_rte_jit(lam, omega, thicknesses,
                                       resistivities, mu0)

                    g_c = 0.0 + 0.0j
                    for m in range(n_lam):
                        lm = lam[m]
                        g_c += r_te[m] * (lm * lm) * hankel_j0[m]
                    g_c = g_c / rho / (4.0 * np.pi)
                    hz_c += area_w[q] * g_c

                hz_c *= 4.0
                hz_im = (hz_c * filter_weights[i, k]).imag
                accum_t += mu0 * hz_im * fourier_weights[k]

            dbdt[i] = accum_t / t
        return dbdt

    # ------------------------------------------------------------------
    # Circular loop: central + offset (Euler acceleration)
    # ------------------------------------------------------------------
    @nb.njit(**_NB_OPTS)
    def _tem_circular_euler_jit(times, thicknesses, resistivities,
                                tx_radius, extra_weights, mu0,
                                hankel_base, hankel_j1,
                                euler_eta, euler_A,
                                filter_weights):
        """Circular-loop dBz/dt via Euler-accelerated Bromwich inversion."""
        n_t = len(times)
        n_euler = len(euler_eta)
        a = tx_radius
        n_lam = len(hankel_base)
        lam = np.empty(n_lam)
        for m in range(n_lam):
            lam[m] = hankel_base[m] / a
        half_A = euler_A / 2.0
        pi_val = np.pi
        dbdt = np.empty(n_t)

        for i in range(n_t):
            t = times[i]
            c = half_A / t
            h = pi_val / t

            d = 0.0
            for k in range(n_euler):
                s = c + k * h * 1j
                omega = s / 1j
                r_te = _te_rte_jit(lam, omega, thicknesses,
                                   resistivities, mu0)
                hz = 0.0 + 0.0j
                for m in range(n_lam):
                    hz += r_te[m] * lam[m] * hankel_j1[m] * extra_weights[m]
                hz *= 0.5
                hz *= filter_weights[i, k]
                fval = (mu0 * hz).real
                sign = 1.0 if k % 2 == 0 else -1.0
                d += euler_eta[k] * sign * fval

            dbdt[i] = np.exp(half_A) / t * d
        return dbdt

    # ------------------------------------------------------------------
    # Square loop: VMD area integral (Euler acceleration)
    # ------------------------------------------------------------------
    @nb.njit(**_NB_OPTS)
    def _tem_square_euler_jit(times, thicknesses, resistivities,
                              rho_q, area_w, mu0,
                              hankel_base, hankel_j0,
                              euler_eta, euler_A,
                              filter_weights):
        """Square-loop dBz/dt via Euler-accelerated Bromwich + VMD integral."""
        n_t = len(times)
        n_euler = len(euler_eta)
        n_q = len(rho_q)
        n_lam = len(hankel_base)
        half_A = euler_A / 2.0
        pi_val = np.pi
        dbdt = np.empty(n_t)

        for i in range(n_t):
            t = times[i]
            c = half_A / t
            h = pi_val / t

            d = 0.0
            for k in range(n_euler):
                s = c + k * h * 1j
                omega = s / 1j
                hz = 0.0 + 0.0j

                for q in range(n_q):
                    rho = rho_q[q]
                    lam = np.empty(n_lam)
                    for m in range(n_lam):
                        lam[m] = hankel_base[m] / rho

                    r_te = _te_rte_jit(lam, omega, thicknesses,
                                       resistivities, mu0)
                    g = 0.0 + 0.0j
                    for m in range(n_lam):
                        lm = lam[m]
                        g += r_te[m] * (lm * lm) * hankel_j0[m]
                    g = g / rho / (4.0 * pi_val)
                    hz += area_w[q] * g

                hz *= 4.0
                hz *= filter_weights[i, k]
                fval = (mu0 * hz).real
                sign = 1.0 if k % 2 == 0 else -1.0
                d += euler_eta[k] * sign * fval

            dbdt[i] = np.exp(half_A) / t * d
        return dbdt
