"""
inversion.py — Inversion utilities for 1-D layered-earth TEM.

Contains:
  - getR               : roughness (smoothness) matrix
  - getJ               : finite-difference Jacobian
  - getJ_analytical    : analytical Jacobian (CUDA/Numba/NumPy)
  - dbdt_to_apprho     : dB/dt → apparent resistivity
  - getRMS, getAlpha, getAlphas : inversion helpers
"""

import numpy as np

from .filters import MU0, HANKEL_FILTERS, FOURIER_FILTERS
from .backends import HAS_CUDA
from .kernels_numba import HAS_NUMBA
from .forward import tem_forward_circle, tem_forward_square

try:
    import numba
except ImportError:
    numba = None


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
# Numba JIT kernels
# ============================================================

if numba is not None:

    @numba.njit(cache=True)
    def _te_grad_one(lam_c, lam2, omega, thick, sigma, MU0_v):
        """Single-omega TE gradient. lam_c: complex(K,), lam2: float(K,)."""
        n_lay = len(sigma)
        K = len(lam_c)
        sval = 1j * omega

        Gamma = np.empty((n_lay, K), dtype=np.complex128)
        dG_a = np.empty((n_lay, K), dtype=np.complex128)
        for j in range(n_lay):
            sv = sval * MU0_v * sigma[j]
            for k in range(K):
                Gamma[j, k] = np.sqrt(lam2[k] + sv)
                dG_a[j, k] = -sv / (2.0 * Gamma[j, k])

        r = np.zeros(K, dtype=np.complex128)
        r_st = np.zeros((n_lay, K), dtype=np.complex128)
        e_st = np.zeros((n_lay - 1, K), dtype=np.complex128)

        for j in range(n_lay - 2, -1, -1):
            tj = thick[j]
            for k in range(K):
                e_st[j, k] = np.exp(-2.0 * Gamma[j, k] * tj)
                gs = Gamma[j, k] + Gamma[j+1, k]
                ps = (Gamma[j, k] - Gamma[j+1, k]) / gs
                r[k] = e_st[j, k] * (r[k] + ps) / (1.0 + r[k] * ps)
            for k in range(K):
                r_st[j, k] = r[k]

        rTE = np.empty(K, dtype=np.complex128)
        adj = np.empty(K, dtype=np.complex128)
        dr = np.zeros((n_lay, K), dtype=np.complex128)

        for k in range(K):
            pa = (lam_c[k] - Gamma[0, k]) / (lam_c[k] + Gamma[0, k])
            da = 1.0 + r_st[0, k] * pa
            rTE[k] = (r_st[0, k] + pa) / da
            adj[k] = (1.0 - pa * pa) / (da * da)
            dpg = -2.0 * lam_c[k] / ((lam_c[k] + Gamma[0, k]) ** 2)
            drpa = (1.0 - r_st[0, k] * r_st[0, k]) / (da * da)
            dr[0, k] += drpa * dpg * dG_a[0, k]

        for j in range(n_lay - 1):
            tj = thick[j]
            for k in range(K):
                rb = r_st[j+1, k]; ej = e_st[j, k]
                gs = Gamma[j, k] + Gamma[j+1, k]
                ps = (Gamma[j, k] - Gamma[j+1, k]) / gs
                den = 1.0 + rb * ps; num = rb + ps
                drps = ej * (1.0 - rb * rb) / (den * den)
                dej = -2.0 * tj * ej
                dr[j, k]   += adj[k] * (dej * num / den + drps * 2.0 * Gamma[j+1, k] / (gs * gs)) * dG_a[j, k]
                dr[j+1, k] += adj[k] * drps * (-2.0 * Gamma[j, k] / (gs * gs)) * dG_a[j+1, k]
                adj[k] *= ej * (1.0 - ps * ps) / (den * den)

        return rTE, dr

    @numba.njit(parallel=True, cache=True)
    def _jac_numba_circ(h_base, h_j1, f_base, f_sin, times,
                         thick, sigma, a, MU0_v):
        K = len(h_base); n_f = len(f_base); n_t = len(times); n_lay = len(sigma)
        lam_c = np.empty(K, dtype=np.complex128)
        lam2 = np.empty(K, dtype=np.float64)
        lam_h = np.empty(K, dtype=np.float64)
        for k in range(K):
            v = h_base[k] / a
            lam_c[k] = v + 0j
            lam2[k] = v * v
            lam_h[k] = v * h_j1[k]

        dbdt = np.zeros(n_t)
        d_dbdt = np.zeros((n_lay, n_t))

        for i in numba.prange(n_t):
            for kf in range(n_f):
                rTE, drTE = _te_grad_one(lam_c, lam2, f_base[kf] / times[i],
                                          thick, sigma, MU0_v)
                hz_im = 0.0
                for k in range(K):
                    hz_im += (rTE[k] * lam_h[k]).imag
                c = 0.5 * MU0_v * f_sin[kf] / times[i]
                dbdt[i] += hz_im * c
                for j in range(n_lay):
                    dh = 0.0
                    for k in range(K):
                        dh += (drTE[j, k] * lam_h[k]).imag
                    d_dbdt[j, i] += dh * c

        return dbdt, d_dbdt

    @numba.njit(parallel=True, cache=True)
    def _jac_numba_sq(h_base, h_j0, f_base, f_sin, times,
                       thick, sigma, rho_q, area_w, MU0_v):
        K = len(h_base); n_f = len(f_base); n_t = len(times)
        n_lay = len(sigma); n_q = len(rho_q)

        all_lam_c = np.empty((n_q, K), dtype=np.complex128)
        all_lam2 = np.empty((n_q, K), dtype=np.float64)
        all_l2h = np.empty((n_q, K), dtype=np.float64)
        for q in range(n_q):
            for k in range(K):
                v = h_base[k] / rho_q[q]
                all_lam_c[q, k] = v + 0j
                all_lam2[q, k] = v * v
                all_l2h[q, k] = v * v * h_j0[k]

        inv_4pi = 1.0 / (4.0 * np.pi)
        dbdt = np.zeros(n_t)
        d_dbdt = np.zeros((n_lay, n_t))

        for i in numba.prange(n_t):
            for kf in range(n_f):
                omega = f_base[kf] / times[i]
                hz_im = 0.0
                dhz_im = np.zeros(n_lay)

                for q in range(n_q):
                    rTE, drTE = _te_grad_one(all_lam_c[q], all_lam2[q], omega,
                                              thick, sigma, MU0_v)
                    g_im = 0.0
                    for k in range(K):
                        g_im += (rTE[k] * all_l2h[q, k]).imag
                    wf = area_w[q] * inv_4pi / rho_q[q]
                    hz_im += wf * g_im

                    for j in range(n_lay):
                        dg_im = 0.0
                        for k in range(K):
                            dg_im += (drTE[j, k] * all_l2h[q, k]).imag
                        dhz_im[j] += wf * dg_im

                c = 4.0 * MU0_v * f_sin[kf] / times[i]
                dbdt[i] += hz_im * c
                for j in range(n_lay):
                    d_dbdt[j, i] += dhz_im[j] * c

        return dbdt, d_dbdt


# ============================================================
# Dispatcher: CUDA > Numba > NumPy
# ============================================================

def getJ_analytical(thicknesses, log_resistivities, tx_geom, times,
                    fwd=tem_forward_circle, use_cuda=True, use_numba=True,
                    hankel_filter='key_101', fourier_filter='key_101'):
    """Analytical Jacobian d(log(-dbdt))/d(ln rho).
    Three paths: CUDA/CuPy > Numba JIT > NumPy. DLF step-off only.
    """
    thicknesses = np.asarray(thicknesses, dtype=float)
    resistivities = np.exp(np.asarray(log_resistivities, dtype=float))
    times = np.asarray(times, dtype=float)
    n_lay = len(resistivities); n_t = len(times)
    h_base, h_j0, h_j1 = HANKEL_FILTERS[hankel_filter]
    f_base, f_sin, f_cos = FOURIER_FILTERS[fourier_filter]
    n_f = len(f_base)

    # Quadrature for square loop
    if fwd is tem_forward_square:
        L = float(tx_geom); hs = L / 2.0; n_quad = 5
        gl_n, gl_w = np.polynomial.legendre.leggauss(n_quad)
        x_pts = hs / 2.0 * (1.0 + gl_n); w_pts = gl_w * hs / 2.0
        rho_q_l, area_w_l = [], []
        for ii in range(n_quad):
            for jj in range(ii, n_quad):
                w = w_pts[ii] * w_pts[jj]
                if ii != jj: w *= 2.0
                rho_q_l.append(np.sqrt(x_pts[ii]**2 + x_pts[jj]**2))
                area_w_l.append(w)
        rho_q = np.asarray(rho_q_l); area_w = np.asarray(area_w_l)

    # ---- PATH 1: CUDA GPU ----
    if HAS_CUDA and use_cuda:
        import cupy as cp
        d_rho = cp.asarray(resistivities)
        d_f_sin = cp.asarray(f_sin)
        d_times = cp.asarray(times)
        all_om = (f_base[None, :] / times[:, None]).ravel()
        d_om = cp.asarray(all_om); M = len(all_om)

        if fwd is tem_forward_square:
            d_h_j0 = cp.asarray(h_j0)
            hz = cp.zeros(M, dtype=cp.complex128)
            dhz = cp.zeros((n_lay, M), dtype=cp.complex128)
            for q in range(len(rho_q)):
                d_lam = cp.asarray(h_base / rho_q[q])
                l2h = d_lam ** 2 * d_h_j0
                rte, drte = _te_grad_batch(d_lam, d_om, thicknesses, d_rho, cp)
                wf = float(area_w[q]) / float(rho_q[q]) / (4.0 * np.pi)
                hz  += wf * cp.sum(rte * l2h[None, :], axis=1)
                dhz += wf * cp.sum(drte * l2h[None, None, :], axis=2)
            hz *= 4.0; dhz *= 4.0
        else:
            a = float(tx_geom)
            d_lam = cp.asarray(h_base / a)
            l_h = d_lam * cp.asarray(h_j1)
            rte, drte = _te_grad_batch(d_lam, d_om, thicknesses, d_rho, cp)
            hz  = 0.5 * cp.sum(rte * l_h[None, :], axis=1)
            dhz = 0.5 * cp.sum(drte * l_h[None, None, :], axis=2)

        sig = (MU0 * cp.imag(hz)).reshape(n_t, n_f)
        dsig = (MU0 * cp.imag(dhz)).reshape(n_lay, n_t, n_f)
        dbdt = cp.asnumpy(cp.sum(sig * d_f_sin[None, :], axis=1) / d_times) * (2.0 / np.pi)
        d_dbdt = cp.asnumpy(
            cp.sum(dsig * d_f_sin[None, None, :], axis=2) / d_times[None, :]
        ) * (2.0 / np.pi)

    # ---- PATH 2: Numba JIT ----
    elif HAS_NUMBA and use_numba:
        sigma_c = (1.0 / resistivities).astype(np.complex128)
        if fwd is tem_forward_square:
            dbdt, d_dbdt = _jac_numba_sq(h_base, h_j0, f_base, f_sin, times,
                                          thicknesses, sigma_c, rho_q, area_w, MU0)
        else:
            dbdt, d_dbdt = _jac_numba_circ(h_base, h_j1, f_base, f_sin, times,
                                            thicknesses, sigma_c, float(tx_geom), MU0)
        dbdt *= (2.0 / np.pi); d_dbdt *= (2.0 / np.pi)

    # ---- PATH 3: Pure NumPy ----
    else:
        dbdt = np.zeros(n_t); d_dbdt = np.zeros((n_lay, n_t))
        if fwd is tem_forward_square:
            for i in range(n_t):
                omegas = f_base / times[i]
                hz = np.zeros(n_f, dtype=complex); dhz = np.zeros((n_lay, n_f), dtype=complex)
                for q in range(len(rho_q)):
                    lam = h_base / rho_q[q]; lam2 = lam ** 2
                    rte, drte = _te_grad_batch(lam, omegas, thicknesses, resistivities, np)
                    hz  += area_w[q] * np.sum(rte * lam2[None, :] * h_j0[None, :], axis=1) / rho_q[q] / (4.0 * np.pi)
                    dhz += area_w[q] * np.sum(drte * lam2[None, None, :] * h_j0[None, None, :], axis=2) / rho_q[q] / (4.0 * np.pi)
                hz *= 4.0; dhz *= 4.0
                sig = MU0 * np.imag(hz); dsig = MU0 * np.imag(dhz)
                dbdt[i] = np.dot(sig, f_sin) / times[i]
                d_dbdt[:, i] = np.sum(dsig * f_sin[None, :], axis=1) / times[i]
        else:
            a = float(tx_geom); lam = h_base / a
            for i in range(n_t):
                omegas = f_base / times[i]
                rte, drte = _te_grad_batch(lam, omegas, thicknesses, resistivities, np)
                hz = 0.5 * np.sum(rte * lam[None, :] * h_j1[None, :], axis=1)
                dhz = 0.5 * np.sum(drte * lam[None, None, :] * h_j1[None, None, :], axis=2)
                sig = MU0 * np.imag(hz); dsig = MU0 * np.imag(dhz)
                dbdt[i] = np.dot(sig, f_sin) / times[i]
                d_dbdt[:, i] = np.sum(dsig * f_sin[None, :], axis=1) / times[i]
        dbdt *= (2.0 / np.pi); d_dbdt *= (2.0 / np.pi)

    # Log-space Jacobian
    f0 = -dbdt
    bad = f0 <= 0
    if np.any(bad):
        print(f"WARNING: {bad.sum()} non-positive values (zeroed in J)")
    J = np.zeros((n_t, n_lay))
    valid = f0 > 0
    for j in range(n_lay):
        J[valid, j] = d_dbdt[j, valid] / dbdt[valid]
    return J


def dbdt_to_apprho(obs_data, tx_area, times):
    """Convert dB/dt to apparent resistivity.

    Parameters
    ----------
    obs_data : array_like
        Observed dB/dt values [T/s].
    tx_area  : float
        Transmitter moment (current × area) [A·m²].
    times    : array_like
        Gate centre times [s].

    Returns
    -------
    rho_a : ndarray
        Apparent resistivity [Ohm·m].
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


def getJ(thicknesses, log_resistivities, tx_geom, times,
         use_numba=False, use_cuda=True, eps=1e-4, fwd=tem_forward_circle,
         transform='dlf', hankel_filter='key_201', fourier_filter='key_81',
         euler_order=11):
    """Finite-difference Jacobian d(log(-dBdt))/d(ln rho)."""
    fwd_kw = dict(use_numba=use_numba, use_cuda=use_cuda, transform=transform,
                  hankel_filter=hankel_filter, fourier_filter=fourier_filter,
                  euler_order=euler_order)

    if fwd is tem_forward_square:
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
        if fwd is tem_forward_square:
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
