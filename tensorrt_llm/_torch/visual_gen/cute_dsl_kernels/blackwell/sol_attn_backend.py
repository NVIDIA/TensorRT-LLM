# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Shape/dtype guard and dense-fallback wrapper around the Sol-Attn kernel.

Adapted from upstream's ``techniques/sparse_backends/sol_attn_backend.py`` at
the pin in ``sol_attn/THIRD_PARTY_NOTICES.md``, which records exactly which
subset is carried and how this version deliberately diverges. Check that file
before re-syncing against upstream.

The kernel-facing API accepts contiguous BF16
``[batch, tokens, heads, 128]`` Q/K/V, ``tau``, ``thresh_type``,
``kv_splits``, and an optional exact KV sink range.

TRT-LLM's dispatch path (``attention_backend/cute_dsl/sol_attn.py``,
``SolAttention``) consumes exactly two names from this module:
``_run_sol_attn_bthd`` and ``sol_attn_supported``. The dense-prefix decision
lives there too, keyed off the normalized timestep forward kwarg.

CuTe DSL imports and compilation are deferred to first use. Calls the kernel
cannot serve -- wrong shape, dtype, or an architecture with no kernel --
delegate to dense attention rather than failing, and increment
``_SOL_STATS["dense_fallback_calls"]`` so the degradation is countable. Set
``SOL_ATTN_STRICT=1`` to raise instead of falling back.
"""

from __future__ import annotations

import functools
import os
from typing import Callable, Optional

import torch

from tensorrt_llm.logger import logger

HEAD_DIM = 128
DEFAULT_TAU = 1.0
DEFAULT_THRESH_TYPE = "diag"
_DEFAULT_SCALE = HEAD_DIM**-0.5


@functools.lru_cache(maxsize=1)
def _load_sol_attn() -> Callable:
    """Import the kernel package's public entry point.

    Deferred rather than done at module scope because importing it pulls in
    the CuTe DSL, which is expensive and not needed unless Sol-Attn is the
    selected backend.
    """

    from .sol_attn import sol_attn

    return sol_attn


# Architectures with a Sol-Attn CuTe kernel. Kept in sync with
# ``sol_attn/interface.py::_CUTE_BACKENDS``; duplicated here so the eligibility
# check does not have to import the CuTe DSL.
SUPPORTED_ARCHS = frozenset({(10, 0)})


def sol_attn_ineligible_reason(q) -> Optional[str]:
    """Why ``q`` cannot use the CuTe kernel, or None if it can.

    Returns a human-readable reason so the caller can say *why* it fell back,
    rather than degrading silently -- an unsupported architecture or head_dim
    otherwise shows up only as absent speedup.
    """
    try:
        import torch
    except Exception:  # pragma: no cover - torch is a runtime dependency
        return "torch is unavailable"
    if not (hasattr(q, "is_cuda") and q.is_cuda):
        return "q is not a CUDA tensor"
    if q.ndim != 4:
        return f"q must be 4-D [B, S, H, D], got ndim={q.ndim}"
    if q.shape[-1] != HEAD_DIM:
        return f"head_dim must be {HEAD_DIM}, got {q.shape[-1]}"
    if q.dtype != torch.bfloat16:
        return f"dtype must be bfloat16, got {q.dtype}"
    try:
        arch = tuple(torch.cuda.get_device_capability(q.device))
    except Exception as exc:
        return f"could not query device capability: {exc}"
    if arch not in SUPPORTED_ARCHS:
        return f"no Sol-Attn kernel for SM{arch[0]}{arch[1]}; supported: " + ", ".join(
            f"SM{a}{b}" for a, b in sorted(SUPPORTED_ARCHS)
        )
    return None


def sol_attn_supported(q) -> bool:
    """Whether ``q`` is eligible for a Sol-Attn CuTe kernel."""

    return sol_attn_ineligible_reason(q) is None


@functools.lru_cache(maxsize=1)
def _cute_runtime_available() -> bool:
    """Whether model dispatch can use one of the optional CuTe kernels."""

    try:
        import cuda.bindings.driver  # noqa: F401
        import cutlass.cute  # noqa: F401
    except ImportError:
        return False
    return True


def _resolve_kv_splits(q, kv_splits: int | str | None) -> int:
    """Resolve the integration-only ``auto`` policy to the public integer API.

    ``auto`` is always 1 here: kv_splits=2/4 was an SM90-only path, and this
    build ships SM100 kernels only.
    """

    if kv_splits in (None, "auto"):
        return 1
    return int(kv_splits)


def _strict() -> bool:
    """Whether SOL_ATTN_STRICT=1 asks us to raise instead of degrading."""

    return os.environ.get("SOL_ATTN_STRICT", "0") == "1"


def _dense_bthd(q, k, v):
    return torch.nn.functional.scaled_dot_product_attention(
        q.transpose(1, 2),
        k.transpose(1, 2),
        v.transpose(1, 2),
    ).transpose(1, 2)


# Opaque to Dynamo, like every other CuTe DSL launch boundary here (see
# cute_dsl/fmha.py, video_sparse_attention/interface.py). Otherwise Dynamo
# traces into the CuTe DSL JIT builder and retraces on every call: near two
# orders of magnitude slower on B200 (2496.9 s mean denoise without it), and
# silently, as if compile just didn't help.
@torch.compiler.disable
def _run_sol_attn_bthd(
    q,
    k,
    v,
    *,
    tau: float = DEFAULT_TAU,
    thresh_type: str = DEFAULT_THRESH_TYPE,
    kv_splits: int | str | None = "auto",
    sink_start: int | None = None,
    sink_tokens: int = 0,
    dense_fn: Callable | None = None,
):
    """Run Sol-Attn on contiguous BTHD tensors, with a safe dense fallback."""

    q0, k0, v0 = q.contiguous(), k.contiguous(), v.contiguous()

    def dense():
        _SOL_STATS["dense_fallback_calls"] += 1
        if dense_fn is not None:
            return dense_fn(q0, k0, v0)
        return _dense_bthd(q0, k0, v0)

    reason = sol_attn_ineligible_reason(q0)
    if reason is None and (k0.shape != q0.shape or v0.shape != q0.shape):
        reason = f"k/v shape must match q {tuple(q0.shape)}"
    if reason is None and (k0.dtype != q0.dtype or v0.dtype != q0.dtype):
        reason = f"k/v dtype must match q {q0.dtype}"
    if reason is not None:
        # Same strictness contract as the kernel-exception path below: this is
        # the arm that silently turns Sol-Attn into a no-op for a whole run
        # (wrong arch, head_dim, or dtype), so it must be visible.
        if _strict():
            raise RuntimeError(f"[sol-attn] cannot run the CuTe kernel: {reason}")
        logger.warning_once(
            f"[sol-attn] falling back to dense attention: {reason}. Sol-Attn will not "
            "accelerate this run. Set SOL_ATTN_STRICT=1 to raise instead.",
            key=("sol_attn_ineligible", reason),
        )
        return dense()

    try:
        kernel = _load_sol_attn()
        out = kernel(
            q0,
            k0,
            v0,
            tau=float(tau),
            thresh_type=str(thresh_type),
            kv_splits=_resolve_kv_splits(q0, kv_splits),
            sink_start=sink_start,
            sink_tokens=int(sink_tokens),
        )
        _SOL_STATS["kernel_calls"] += 1
        return out
    except Exception as exc:
        if _strict():
            raise
        logger.warning_once(
            f"[sol-attn] kernel raised {type(exc).__name__}: {exc}; falling back to dense "
            "attention for this call. Set SOL_ATTN_STRICT=1 to raise instead of silently falling "
            "back.",
            key=(type(exc).__name__, str(exc)),
        )
        return dense()


# Lightweight run-validation counters. `kernel_calls` is the census used to
# prove the CuTe kernel actually ran: because forward() falls back to dense
# SDPA on any kernel exception, a run that "works" but never increments this
# was silently dense. Set SOL_ATTN_STRICT=1 to raise instead of falling back.
_SOL_STATS = {"kernel_calls": 0, "dense_fallback_calls": 0}


def reset_sol_attn_stats() -> None:
    """Zero the counters, e.g. after an untimed warmup generation."""

    for key in _SOL_STATS:
        _SOL_STATS[key] = 0


def get_sol_attn_stats() -> dict[str, int]:
    """Return the run-validation counters."""

    return {key: int(value) for key, value in _SOL_STATS.items()}
