"""
waveform.py — Waveform convolution for piecewise-linear transmitter waveforms.

Contains:
  convolve_waveform  — public API (dispatches to Numba JIT when available)
  _log_interp_scalar — Numba JIT log-time interpolation
  _convolve_waveform_jit — Numba JIT inner loops
"""

import numpy as np
from .kernels_numba import HAS_NUMBA

if HAS_NUMBA:
    import numba as nb
    _NB_OPTS = {'nogil': True, 'cache': True}

    @nb.njit(**_NB_OPTS)
    def _log_interp_scalar(log_t, log_st, sr):
        """Linear interpolation of step_response at a single log(t) value."""
        n = len(log_st)
        if log_t <= log_st[0]:
            return sr[0]
        if log_t >= log_st[n - 1]:
            return sr[n - 1]
        lo = 0
        hi = n - 1
        while hi - lo > 1:
            mid = (lo + hi) >> 1
            if log_st[mid] <= log_t:
                lo = mid
            else:
                hi = mid
        frac = (log_t - log_st[lo]) / (log_st[hi] - log_st[lo])
        return sr[lo] + frac * (sr[hi] - sr[lo])

    @nb.njit(**_NB_OPTS)
    def _convolve_waveform_jit(gate_times, wf_t, wf_I, log_st, sr,
                               gl_nodes, gl_weights):
        """Numba-compiled waveform convolution core."""
        n_gates = len(gate_times)
        n_seg = len(wf_t) - 1
        n_quad = len(gl_nodes)
        result = np.zeros(n_gates)

        for seg in range(n_seg):
            dt_seg = wf_t[seg + 1] - wf_t[seg]
            if abs(dt_seg) < 1e-30:
                continue
            slope = (wf_I[seg + 1] - wf_I[seg]) / dt_seg
            if abs(slope) < 1e-30:
                continue

            mid = 0.5 * (wf_t[seg + 1] + wf_t[seg])
            half = 0.5 * dt_seg

            for j in range(n_gates):
                tg = gate_times[j]
                accum = 0.0
                for q in range(n_quad):
                    tau = mid + half * gl_nodes[q]
                    t_eval = tg - tau
                    val = _log_interp_scalar(np.log(t_eval), log_st, sr)
                    accum += gl_weights[q] * val
                result[j] += -slope * half * accum

        return result


def convolve_waveform(step_times, step_response, waveform_times,
                      waveform_currents, gate_times, n_quad=8):
    """
    Convolve a step response with a piecewise-linear transmitter waveform.

    Computes V(t) = -integral (dI/dtau) * S(t - tau) dtau using
    Gauss-Legendre quadrature per waveform segment.

    Parameters
    ----------
    step_times       : array-like  Times at which the step response is known [s].
    step_response    : array-like  Step response values.
    waveform_times   : array-like  Break points of the piecewise-linear waveform [s].
    waveform_currents: array-like  Current at each break point [A].
    gate_times       : array-like  Output measurement gate centre times [s].
    n_quad           : int         Gauss-Legendre order per segment (default 8).

    Returns
    -------
    result : ndarray, shape (n_gates,)
    """
    gate_times = np.asarray(gate_times, dtype=float)
    wf_t = np.asarray(waveform_times, dtype=float)
    wf_I = np.asarray(waveform_currents, dtype=float)

    log_st = np.log(np.asarray(step_times, dtype=float))
    sr = np.asarray(step_response, dtype=float)

    gl_nodes, gl_weights = np.polynomial.legendre.leggauss(n_quad)

    if HAS_NUMBA:
        return _convolve_waveform_jit(gate_times, wf_t, wf_I,
                                      log_st, sr, gl_nodes, gl_weights)

    # Pure Python fallback
    def step_interp(t):
        return np.interp(np.log(t), log_st, sr)

    result = np.zeros(len(gate_times))

    for k in range(len(wf_t) - 1):
        dt_seg = wf_t[k + 1] - wf_t[k]
        if abs(dt_seg) < 1e-30:
            continue
        slope = (wf_I[k + 1] - wf_I[k]) / dt_seg
        if abs(slope) < 1e-30:
            continue

        mid = 0.5 * (wf_t[k + 1] + wf_t[k])
        half = 0.5 * dt_seg
        tau_pts = mid + half * gl_nodes
        w_sc = half * gl_weights

        for j, tg in enumerate(gate_times):
            t_eval = tg - tau_pts
            result[j] += -slope * np.dot(w_sc, step_interp(t_eval))

    return result
