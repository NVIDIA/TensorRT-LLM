"""Kernels-free NVFP4 typing helpers."""

from __future__ import annotations

from enum import Enum


class ActScale(str, Enum):
    """Activation per-tensor decode-scale mode for ``NVFP4Linear``."""

    DYNAMIC = "dynamic"  # Per-forward amax / GLOBAL_RANGE.
    FIXED_1 = "fixed_1"  # Constant 1.0; alpha equals weight_scale_2.
    STATIC = "static"  # Calibrated checkpoint input_scale; alpha = input_scale * weight_scale_2.
