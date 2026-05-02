"""
waveform.py — Waveform convolution for piecewise-linear transmitter waveforms.

Contains:
  setup_waveform         — precompute quadrature structure (empymod-style)
  convolve_waveform      — public API (dispatches to Numba JIT when available)
  _log_interp_scalar     — Numba JIT log-time interpolation
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
                    if t_eval <= 0.0:
                        continue
                    val = _log_interp_scalar(np.log(t_eval), log_st, sr)
                    accum += gl_weights[q] * val
                result[j] += -slope * half * accum

        return result


def setup_waveform(gate_times, waveform_times, waveform_currents, n_quad=8):
    """
    Precompute the quadrature structure for waveform convolution.

    Follows the empymod pattern: deduplicate all GL quadrature times across
    every (gate, segment) pair once at setup, so the forward model only needs
    to evaluate the step response at the returned ``comp_times`` array on each
    inversion iteration.

    Parameters
    ----------
    gate_times        : array-like  Measurement gate centre times [s].
    waveform_times    : array-like  Break points of the piecewise-linear waveform [s].
    waveform_currents : array-like  Current at each break point [A].
    n_quad            : int         Gauss-Legendre order per segment (default 8).

    Returns
    -------
    comp_times : ndarray, shape (n_unique,)
        Deduplicated set of times at which the step response must be evaluated.
    apply_waveform : callable
        ``apply_waveform(step_resp)`` where *step_resp* is a 1-D array of
        shape ``(n_unique,)`` or a 2-D array ``(n_unique, N)`` (e.g. Jacobian
        columns).  Returns the convolved result of shape ``(n_gates,)`` or
        ``(n_gates, N)``.
    """
    gate_times = np.asarray(gate_times, dtype=float)
    wf_t = np.asarray(waveform_times, dtype=float)
    wf_I = np.asarray(waveform_currents, dtype=float)

    gl_nodes, gl_weights = np.polynomial.legendre.leggauss(n_quad)

    dt = np.diff(wf_t)
    dIdt = np.diff(wf_I) / dt

    act = np.abs(dIdt) > 1e-30
    t0 = wf_t[:-1][act]
    t1 = wf_t[1:][act]
    slope = dIdt[act]
    n_gates = gate_times.size
    n_active = t0.size

    if n_active == 0:
        # No ramp segments: waveform is constant — return zeros.
        def apply_waveform(step_resp):
            sr = np.asarray(step_resp)
            if sr.ndim == 1:
                return np.zeros(n_gates)
            return np.zeros((n_gates, sr.shape[1]))
        return np.array([1.0]), apply_waveform

    # Delays from each gate to segment endpoints: (n_gates, n_active)
    ta = gate_times[:, None] - t0[None, :]   # gate - segment_start
    tb = gate_times[:, None] - t1[None, :]   # gate - segment_end

    valid = ta > 0.0                          # gate must be after segment start
    tb = np.where(tb < 0.0, 0.0, tb)         # clamp: gate within segment -> tb=0

    # GL quadrature delay times: (n_gates, n_active, n_quad)
    comp_time = (0.5 * (tb - ta))[:, :, None] * gl_nodes \
              + (0.5 * (ta + tb))[:, :, None]

    # Per-(gate,segment) weight = Jacobian_of_interval_transform * slope
    seg_w = np.where(valid, 0.5 * (tb - ta) * slope[None, :], 0.0)  # (n_gates, n_active)

    # Mask out invalid (gate before segment start, or non-positive delay)
    valid3 = valid[:, :, None] & (comp_time > 0.0)   # (n_gates, n_active, n_quad)

    # Replace invalid times with zero so np.unique works cleanly
    comp_time_safe = np.where(valid3, comp_time, 0.0)

    # Deduplicate — same approach as empymod
    comp_times_flat, map_time = np.unique(comp_time_safe[valid3], return_inverse=True)

    # Clamp to a minimum time so the forward model is never called at
    # unphysically early times (empymod uses the same guard).
    _min_t = gate_times.min() * 1e-2
    comp_times_flat = np.where(comp_times_flat < _min_t, _min_t, comp_times_flat)

    # Build a weight matrix W of shape (n_gates, n_unique) once at setup.
    # apply_waveform then reduces to a single matmul: W @ step_resp.
    j_idx, k_idx, q_idx = np.where(valid3)
    entry_weights = seg_w[j_idx, k_idx] * gl_weights[q_idx]  # (n_valid,)
    W = np.zeros((n_gates, len(comp_times_flat)))
    np.add.at(W, (j_idx, map_time), entry_weights)

    def apply_waveform(step_resp):
        """Compute convolved response from step response at comp_times."""
        sr = np.asarray(step_resp)
        return W @ sr

    return comp_times_flat, apply_waveform


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

    # Vectorised NumPy fallback — empymod-style (Key, DIPOLE1D).
    # Reference: empymod.utils.check_waveform
    #
    # Strategy: pre-compute all GL quadrature delay times for every
    # (gate, segment) pair at once, deduplicate with np.unique, do a
    # single batched np.interp, then reassemble the weighted sum.

    dt = np.diff(wf_t)
    dIdt = np.diff(wf_I) / dt

    # Keep only segments with a non-zero current ramp.
    act = np.abs(dIdt) > 1e-30
    t0 = wf_t[:-1][act]       # segment start times
    t1 = wf_t[1:][act]        # segment end times
    slope = dIdt[act]

    if t0.size == 0:
        return np.zeros(gate_times.size)

    # Delays from each gate time to the segment endpoints: (n_gates, n_seg)
    #   ta = gate - t0  >=0 when gate is after segment start
    #   tb = gate - t1  clipped to 0 when gate is within the segment
    ta = gate_times[:, None] - t0[None, :]
    tb = gate_times[:, None] - t1[None, :]

    # A segment only contributes to gate j if the segment has already started.
    valid = ta > 0.0                          # (n_gates, n_seg)

    # Truncate: if gate is still within the segment, cap the upper delay at 0.
    tb = np.where(tb < 0.0, 0.0, tb)

    # GL quadrature delay points for every (gate, segment, node): (n_gates, n_seg, n_quad)
    # Change of interval from [-1,1] to [ta, tb]:
    #   delay = (tb-ta)/2 * g_x + (ta+tb)/2
    comp_time = (0.5 * (tb - ta))[:, :, None] * gl_nodes \
              + (0.5 * (ta + tb))[:, :, None]

    # Per-(gate,segment) integration weight: Jacobian * slope.
    # Jacobian = (tb-ta)/2  (<0 since ta>tb), so the sign already gives -slope*half.
    seg_w = np.where(valid, 0.5 * (tb - ta) * slope[None, :], 0.0)

    # Mask invalid entries (gate before segment, or non-positive delay).
    valid3 = valid[:, :, None] & (comp_time > 0.0)   # (n_gates, n_seg, n_quad)

    # Flatten, deduplicate, interpolate in one vectorised pass.
    t_flat = comp_time.ravel()
    v_flat = valid3.ravel()

    resp_flat = np.zeros(t_flat.size)
    if v_flat.any():
        t_good = t_flat[v_flat]
        uniq_t, inv = np.unique(t_good, return_inverse=True)
        resp_flat[v_flat] = np.interp(np.log(uniq_t), log_st, sr)[inv]

    resp = resp_flat.reshape(comp_time.shape)   # (n_gates, n_seg, n_quad)

    # Weighted sum: result[j] = Σ_{k,q}  seg_w[j,k] * gl_weights[q] * resp[j,k,q]
    return np.einsum('jk,q,jkq->j', seg_w, gl_weights, resp)
