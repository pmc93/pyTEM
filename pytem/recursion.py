"""
recursion.py — TE reflection coefficient via Wait recursion (NumPy).
"""

import numpy as np
from .transform_weights import MU0


def te_reflection_coeff(lam, omega, thicknesses, resistivities):
    """
    TE-mode surface reflection coefficient for a layered isotropic earth
    using the Wait (1954) upward recursion.  Complex arithmetic.

    Parameters
    ----------
    lam           : (K,)   horizontal wavenumbers [1/m]
    omega         : float or complex  angular frequency [rad/s]
    thicknesses   : (N-1,) layer thicknesses [m]
    resistivities : (N,)   layer resistivities [Ohm.m] (may be complex for IP)

    Returns
    -------
    r_TE : (K,) complex128 — TE surface reflection coefficient
    """
    n_lay = len(resistivities)
    sval = 1j * omega
    resistivities = np.asarray(resistivities, dtype=complex)

    sigma = 1.0 / resistivities
    Gamma = np.sqrt(lam[None, :]**2 + (sval * MU0 * sigma)[:, None])

    r = np.zeros(len(lam), dtype=complex)
    for j in range(n_lay - 2, -1, -1):
        psi = (Gamma[j] - Gamma[j + 1]) / (Gamma[j] + Gamma[j + 1])
        r = np.exp(-2.0 * Gamma[j] * thicknesses[j]) * (r + psi) / (1.0 + r * psi)

    psi_air = (lam - Gamma[0]) / (lam + Gamma[0])
    r_TE = (r + psi_air) / (1.0 + r * psi_air)
    return r_TE


def te_reflection_coeff_grad(lam, omega, thicknesses, resistivities):
    """
    TE reflection coefficient AND its gradient w.r.t. log(resistivity).

    Returns both r_TE and dr_TE/d(ln rho_j) for every layer j,
    computed in a single forward + backward pass through the Wait recursion.

    Parameters
    ----------
    lam           : (K,)   horizontal wavenumbers [1/m]
    omega         : float or complex  angular frequency [rad/s]
    thicknesses   : (N-1,) layer thicknesses [m]
    resistivities : (N,)   layer resistivities [Ohm.m]

    Returns
    -------
    r_TE     : (K,)    complex128 — TE surface reflection coefficient
    dr_TE    : (N, K)  complex128 — d(r_TE) / d(ln rho_j)
    """
    n_lay = len(resistivities)
    K = len(lam)
    sval = 1j * omega
    resistivities = np.asarray(resistivities, dtype=complex)

    sigma = 1.0 / resistivities
    lam2 = lam ** 2
    Gamma = np.sqrt(lam2[None, :] + (sval * MU0 * sigma)[:, None])  # (N, K)

    # dGamma_j / d(ln rho_j) = -sval*MU0*sigma_j / (2*Gamma_j)  *  (-rho_j)
    #   since d(sigma)/d(ln rho) = -sigma, so d(Gamma^2)/d(ln rho) = -sval*MU0*sigma
    #   => dGamma/d(ln rho) = -sval*MU0*sigma / (2*Gamma)
    dGamma_dlnrho = -sval * MU0 * sigma[:, None] / (2.0 * Gamma)  # (N, K)

    # --- Forward pass: store intermediate r_j and exp_j ---
    r_store = np.zeros((n_lay, K), dtype=complex)
    exp_store = np.zeros((n_lay - 1, K), dtype=complex)

    r = np.zeros(K, dtype=complex)  # r_{N} = 0 (half-space bottom)
    r_store[n_lay - 1] = r

    for j in range(n_lay - 2, -1, -1):
        exp_j = np.exp(-2.0 * Gamma[j] * thicknesses[j])
        exp_store[j] = exp_j
        psi = (Gamma[j] - Gamma[j + 1]) / (Gamma[j] + Gamma[j + 1])
        r = exp_j * (r + psi) / (1.0 + r * psi)
        r_store[j] = r

    # Air interface
    psi_air = (lam - Gamma[0]) / (lam + Gamma[0])
    r_TE = (r_store[0] + psi_air) / (1.0 + r_store[0] * psi_air)

    # --- Backward pass: accumulate dr_TE/d(ln rho_j) ---
    # Chain rule: dr_TE/d(param) = (dr_TE/dr_0) * (dr_0/d(param))
    # where dr_TE/dr_0 comes from the air interface.

    # Air interface derivative: dr_TE/dr_0
    denom_air = (1.0 + r_store[0] * psi_air)
    dr_TE_dr0 = (1.0 - psi_air ** 2) / denom_air ** 2

    # Air interface derivative: dr_TE/dGamma_0 (through psi_air)
    dpsi_air_dG0 = -2.0 * lam / (lam + Gamma[0]) ** 2
    dr_TE_dpsi_air = (1.0 - r_store[0] ** 2) / denom_air ** 2

    dr_TE_all = np.zeros((n_lay, K), dtype=complex)

    # For each layer j, we need dr_0/dr_j and dr_j/d(ln rho_j)
    # We propagate dr_0/dr_j backwards: dr_0/dr_{j} = dr_0/dr_{j-1} * dr_{j-1}/dr_j

    # Start: sensitivity of r_TE to r_0 (already computed as dr_TE_dr0)
    # Then for each j from 0 upward, compute local derivatives

    for j in range(n_lay - 1):
        # r_j = exp_j * (r_{j+1} + psi_j) / (1 + r_{j+1} * psi_j)
        # where psi_j = (Gamma_j - Gamma_{j+1}) / (Gamma_j + Gamma_{j+1})

        r_below = r_store[j + 1] if j + 1 < n_lay else np.zeros(K, dtype=complex)
        exp_j = exp_store[j]
        psi_j = (Gamma[j] - Gamma[j + 1]) / (Gamma[j] + Gamma[j + 1])
        numer = r_below + psi_j
        denom = 1.0 + r_below * psi_j

        # dpsi_j / dGamma_j
        dpsi_dGj = 2.0 * Gamma[j + 1] / (Gamma[j] + Gamma[j + 1]) ** 2
        # dpsi_j / dGamma_{j+1}
        dpsi_dGjp1 = -2.0 * Gamma[j] / (Gamma[j] + Gamma[j + 1]) ** 2

        # dr_j/dpsi_j
        dr_dpsi = exp_j * (1.0 - r_below ** 2) / denom ** 2

        # dr_j/dGamma_j (through exp and psi)
        dexp_dGj = -2.0 * thicknesses[j] * exp_j
        dr_dGj = dexp_dGj * numer / denom + dr_dpsi * dpsi_dGj

        # dr_j/dGamma_{j+1} (through psi only)
        dr_dGjp1 = dr_dpsi * dpsi_dGjp1

        # dr_j/dr_{j+1}
        dr_drbelow = exp_j * (1.0 - psi_j ** 2) / denom ** 2

        # Now accumulate: we need d(r_TE)/d(ln rho_j)
        # For layer j: Gamma_j depends on rho_j
        #   d(r_TE)/d(ln rho_j) += chain_to_r_j * dr_j/dGamma_j * dGamma_j/d(ln rho_j)
        #                        (and also through Gamma_j's effect on r_{j-1}, etc.)

        # We'll use a different strategy: propagate adjoint from r_TE backward.

    # --- Cleaner adjoint approach ---
    # Let adj[j] = dr_TE / dr_j  (adjoint sensitivity of r_TE to the recursion variable at layer j)
    # adj[0] = dr_TE/dr_0  (from air interface)
    # For j >= 0: adj[j+1] = adj[j] * dr_j/dr_{j+1}
    # And the contribution to ln(rho_j) comes from Gamma_j appearing in layers j and j-1.

    dr_TE_all = np.zeros((n_lay, K), dtype=complex)

    adj = dr_TE_dr0.copy()  # dr_TE / dr_0

    # Layer 0 also contributes through the air interface psi_air
    dr_TE_all[0] += dr_TE_dpsi_air * dpsi_air_dG0 * dGamma_dlnrho[0]

    for j in range(n_lay - 1):
        r_below = r_store[j + 1]
        exp_j = exp_store[j]
        psi_j = (Gamma[j] - Gamma[j + 1]) / (Gamma[j] + Gamma[j + 1])
        numer = r_below + psi_j
        denom = 1.0 + r_below * psi_j

        dpsi_dGj = 2.0 * Gamma[j + 1] / (Gamma[j] + Gamma[j + 1]) ** 2
        dpsi_dGjp1 = -2.0 * Gamma[j] / (Gamma[j] + Gamma[j + 1]) ** 2

        dr_dpsi = exp_j * (1.0 - r_below ** 2) / denom ** 2
        dexp_dGj = -2.0 * thicknesses[j] * exp_j
        dr_dGj = dexp_dGj * numer / denom + dr_dpsi * dpsi_dGj
        dr_dGjp1 = dr_dpsi * dpsi_dGjp1
        dr_drbelow = exp_j * (1.0 - psi_j ** 2) / denom ** 2

        # Contribution of Gamma_j to r_j (through exp and psi_j)
        dr_TE_all[j] += adj * dr_dGj * dGamma_dlnrho[j]
        # Contribution of Gamma_{j+1} to r_j (through psi_j)
        dr_TE_all[j + 1] += adj * dr_dGjp1 * dGamma_dlnrho[j + 1]

        # Propagate adjoint down
        adj = adj * dr_drbelow

    return r_TE, dr_TE_all
