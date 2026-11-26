# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CSA2 persistent cache formats and their power-of-two E4M3 scales.

The PrimTS kernels' host code imports only this, not ``params.py``, so the PrimTS precompile
workers (``fmha/_prims_ts_worker.py``) can compile CSA2 kernels without TensorRT-LLM's runtime.
``params.py`` re-exports every name.
"""

from __future__ import annotations

import functools
import math
import os
from typing import Literal

SwaCacheFormat = Literal["swa", "swa_fp8"]
# Rows PrimTS reads for MAIN: persistent 288-byte NVFP4 records, or 512-byte
# unit-scale E4M3 rows of a context-phase bank decoded from them.
PrimsTSMainFormat = Literal["main", "main_fp8"]


def resolve_swa_cache_format(kv_cache_dtype: str = "auto") -> SwaCacheFormat:
    """Resolve persistent SWA encoding independently of the compute staging dtype."""
    if kv_cache_dtype not in ("auto", "bfloat16", "fp8", "fp8_ds_mla"):
        raise ValueError("CSA2 KV cache dtype must be auto, bfloat16, fp8 or fp8_ds_mla")
    return "swa_fp8" if kv_cache_dtype == "fp8" else "swa"


# The largest SWA magnitude swa_fp8 rows hold; it sets their power-of-two storage scale.
SWA_FP8_AMAX_ENV = "TRTLLM_CSA2_SWA_FP8_AMAX"
# The largest query magnitude PrimTS's E4M3 Q holds; it sets Q's power-of-two scale.
Q_FP8_AMAX_ENV = "TRTLLM_CSA2_Q_FP8_AMAX"
_E4M3_MAX = 448.0
# Native context staging spends at most three binades (448 * 2**3 covers the main pool),
# so a larger scale would saturate SWA rows there.
_SWA_FP8_SCALE_EXPONENTS = range(-8, 4)
_Q_FP8_SCALE_EXPONENTS = range(-8, 9)


def _amax_scale_exponent(env: str, exponents: range, what: str) -> int:
    """The smallest ``e`` in ``exponents`` with 448 * 2**e >= the amax in ``env`` (default 448)."""
    value = os.environ.get(env, str(_E4M3_MAX))
    try:
        amax = float(value)
    except ValueError as error:
        raise ValueError(f"{env} must be a positive finite float, got {value!r}") from error
    if not math.isfinite(amax) or amax <= 0:
        raise ValueError(f"{env} must be a positive finite float, got {value!r}")
    exponent = math.ceil(math.log2(amax / _E4M3_MAX))
    if exponent not in exponents:
        lo, hi = exponents[0], exponents[-1]
        raise ValueError(
            f"{env}={value} needs a {what} scale of 2**{exponent}; supported "
            f"scales are 2**{lo} to 2**{hi} (amax up to {_E4M3_MAX * 2.0**hi:g})"
        )
    return exponent


@functools.cache
def swa_fp8_scale_exponent() -> int:
    """Exponent ``e`` of the swa_fp8 SWA records' storage scale: a row stores E4M3(x / 2**e).

    ``e`` is the smallest exponent whose E4M3 range, 448 * 2**e, covers the magnitude in
    ``TRTLLM_CSA2_SWA_FP8_AMAX`` (default 448: unit scale, ``e = 0``). Every reader applies
    the same scale, so all processes that share a cache, such as the context and generation
    workers of a disaggregated deployment, need the same value; it is fixed per process.
    """
    return _amax_scale_exponent(SWA_FP8_AMAX_ENV, _SWA_FP8_SCALE_EXPONENTS, "swa_fp8")


@functools.cache
def q_fp8_scale_exponent() -> int:
    """Exponent ``e`` of the scale of PrimTS's E4M3 Q: it reads E4M3(q / 2**e).

    ``e`` is the smallest exponent whose E4M3 range covers ``TRTLLM_CSA2_Q_FP8_AMAX``
    (default 448: unit scale, ``e = 0``); PrimTS multiplies Q.K back by ``2**e``. Q is
    rounded every forward, so only this process reads it at this scale.
    """
    return _amax_scale_exponent(Q_FP8_AMAX_ENV, _Q_FP8_SCALE_EXPONENTS, "PrimTS Q")


def swa_scale_exponent(swa_cache_format: str) -> int:
    """Storage-scale exponent of a SWA format's rows: swa_fp8 rows hold E4M3(x / 2**e) with
    ``e = swa_fp8_scale_exponent()``; scaled ``swa`` records carry their own group scales."""
    return swa_fp8_scale_exponent() if swa_cache_format == "swa_fp8" else 0
