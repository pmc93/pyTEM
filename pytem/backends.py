"""
backends.py — CuPy/CUDA detection and GPU filter array transfer.

Provides:
  - HAS_CUDA flag
  - GPU_HANKEL, GPU_FOURIER registries (CuPy arrays on device)
"""

import numpy as np

from .filters import (
    _HANKEL_BASE_201, _HANKEL_J0_201, _HANKEL_J1_201,
    _HANKEL_BASE_101, _HANKEL_J0_101, _HANKEL_J1_101,
    _FOURIER_BASE_81, _FOURIER_SIN_81, _FOURIER_COS_81,
    _FOURIER_BASE_101, _FOURIER_SIN_101, _FOURIER_COS_101,
)

try:
    import cupy as cp
    HAS_CUDA = cp.cuda.is_available()
except ImportError:
    HAS_CUDA = False

GPU_HANKEL = {}
GPU_FOURIER = {}

if HAS_CUDA:
    import cupy as cp

    _GPU_DEVICE = cp.cuda.Device(0)
    _GPU_DEVICE.use()

    # 201-pt Hankel
    _d_HANKEL_BASE_201 = cp.asarray(_HANKEL_BASE_201)
    _d_HANKEL_J0_201   = cp.asarray(_HANKEL_J0_201, dtype=cp.complex128)
    _d_HANKEL_J1_201   = cp.asarray(_HANKEL_J1_201, dtype=cp.complex128)

    # 101-pt Hankel
    _d_HANKEL_BASE_101 = cp.asarray(_HANKEL_BASE_101)
    _d_HANKEL_J0_101   = cp.asarray(_HANKEL_J0_101, dtype=cp.complex128)
    _d_HANKEL_J1_101   = cp.asarray(_HANKEL_J1_101, dtype=cp.complex128)

    # 81-pt Fourier
    _d_FOURIER_BASE_81 = cp.asarray(_FOURIER_BASE_81)
    _d_FOURIER_SIN_81  = cp.asarray(_FOURIER_SIN_81)
    _d_FOURIER_COS_81  = cp.asarray(_FOURIER_COS_81)

    # 101-pt Fourier
    _d_FOURIER_BASE_101 = cp.asarray(_FOURIER_BASE_101)
    _d_FOURIER_SIN_101  = cp.asarray(_FOURIER_SIN_101)
    _d_FOURIER_COS_101  = cp.asarray(_FOURIER_COS_101)

    GPU_HANKEL = {
        'key_201': (_d_HANKEL_BASE_201, _d_HANKEL_J0_201, _d_HANKEL_J1_201),
        'key_101': (_d_HANKEL_BASE_101, _d_HANKEL_J0_101, _d_HANKEL_J1_101),
    }
    GPU_FOURIER = {
        'key_81':  (_d_FOURIER_BASE_81,  _d_FOURIER_SIN_81,  _d_FOURIER_COS_81),
        'key_101': (_d_FOURIER_BASE_101, _d_FOURIER_SIN_101, _d_FOURIER_COS_101),
    }
