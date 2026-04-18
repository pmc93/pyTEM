"""
euler.py — Standalone Euler acceleration for inverse Laplace transform.

This is a reference/verification implementation (Abate & Whitt, 1995).
The production Euler path in the forward functions uses precomputed
eta/A weights from EULER_PARAMS (see kernels_numba).
"""

import numpy as np


def euler_invert(F, t, N=15, A=18.4):
    """
    Inverse Laplace transform via Euler acceleration (Abate & Whitt 1995).

    Evaluates F(s) at 2N+1 complex points along Re(s) = A/(2t),
    then accelerates the alternating partial sums with binomial weights.

    Parameters
    ----------
    F : callable  F(s) -> complex, the Laplace-domain function
    t : float     time at which to evaluate the inverse
    N : int       half-order (default 15 — 31 evaluations, ~8 digits)
    A : float     Bromwich abscissa parameter (default 18.4)

    Returns
    -------
    f(t) : float — the time-domain value
    """
    c = A / (2.0 * t)
    h = np.pi / t

    # Function values along the Bromwich line
    vals = np.zeros(2 * N + 1)
    for k in range(2 * N + 1):
        s = c + k * h * 1j
        vals[k] = np.real(F(s))

    # Partial sums of the alternating series
    partial = np.zeros(2 * N + 1)
    partial[0] = vals[0] / 2.0
    for k in range(1, 2 * N + 1):
        partial[k] = partial[k - 1] + (-1)**k * vals[k]

    # Euler acceleration: binomial average of partial[N:2N+1]
    d = 0.0
    binom = 1.0
    for j in range(N + 1):
        d += binom * partial[N + j]
        if j < N:
            binom *= float(N - j) / float(j + 1)
    d /= 2.0**N

    return np.exp(A / 2.0) / t * d
