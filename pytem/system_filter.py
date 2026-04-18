"""
system_filter.py — Butterworth bandpass and cascade system transfer functions.

Contains:
  butterworth_filter  — 1st or 2nd order Butterworth LP/HP/BP
  cascade_filter      — WalkTEM convention (two cascaded 1st-order LP)
"""

import numpy as np


def butterworth_filter(f_low=None, f_high=None, order=1):
    """
    Create a Butterworth bandpass system transfer function.

    Parameters
    ----------
    f_low  : low-cut frequency [Hz] (high-pass), or None for no high-pass
    f_high : high-cut frequency [Hz] (low-pass), or None for no low-pass
    order  : filter order (1 or 2)

    Returns
    -------
    H : callable  omega -> complex transfer function
    """
    def H(omega):
        s = 1j * omega
        result = np.ones_like(s)
        # Low-pass (high-cut)
        if f_high is not None:
            wc = 2 * np.pi * f_high
            if order == 1:
                result *= wc / (s + wc)
            elif order == 2:
                result *= wc**2 / (s**2 + np.sqrt(2) * wc * s + wc**2)
        # High-pass (low-cut)
        if f_low is not None:
            wc = 2 * np.pi * f_low
            if order == 1:
                result *= s / (s + wc)
            elif order == 2:
                result *= s**2 / (s**2 + np.sqrt(2) * wc * s + wc**2)
        return result
    return H


def cascade_filter(filtfreq):
    """Two cascaded 1st-order Butterworth LP (WalkTEM convention)."""
    H1 = butterworth_filter(f_high=filtfreq, order=1)
    H2 = butterworth_filter(f_high=3e5, order=1)
    return lambda omega: H1(omega) * H2(omega)
