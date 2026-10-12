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
the pin in ``sol_attn/THIRD_PARTY_NOTICES.md``, which records which subset is
carried. Check that file before re-syncing against upstream.

Deliberate divergences from upstream, which a re-sync must preserve rather
than overwrite:

* ``logger.warning_once`` replaces ``print()``, so fallbacks are suppressible
  and routed through this repository's logger.
* ``_SOL_STATS["dense_fallback_calls"]`` makes a silently-degraded run
  countable rather than only visible on stderr.
* ``sol_attn_ineligible_reason()`` names the specific reason (architecture,
  head_dim, dtype) instead of returning one boolean.
* Kernel exceptions propagate. Upstream catches them and silently reruns the
  call with torch SDPA; a failed CuTe launch can leave the device in a bad
  state, so recovery is not attempted here. Only inputs known up front to be
  unservable (shape, dtype, architecture) are routed to dense attention, and
  ``TRTLLM_SOL_ATTN_STRICT=1`` turns even that into an error.
* Dense paths route to ``cute_dsl_fmha_fwd`` through ``dense_fn``. Upstream
  falls back to torch SDPA; staying inside the configured backend is what lets
  a ``backend: CUTEDSL`` A/B isolate sparsity rather than also swapping the
  dense kernel.
* ``@torch.compiler.disable`` guards the launch boundary; see the comment on
  ``_run_sol_attn_bthd`` for why, and for the ``torch.library.custom_op``
  alternative upstream uses.

The kernel-facing API accepts contiguous BF16
``[batch, tokens, heads, 128]`` Q/K/V, ``tau``, ``thresh_type``,
``kv_splits``, and an optional exact KV sink range.

TRT-LLM's dispatch path (``attention_backend/sparse/sol/backend.py``,
``SOLCuTeDSLAttention``) consumes exactly two names from this module:
``_run_sol_attn_bthd`` and ``sol_attn_supported``. The dense-prefix decision
lives there too, keyed off the normalized timestep forward kwarg.

CuTe DSL imports and compilation are deferred to first use. Calls the kernel
is known not to serve -- wrong shape, dtype, or an architecture with no kernel
-- are delegated to dense attention before any launch, logged once, and
counted in ``_SOL_STATS["dense_fallback_calls"]``. Set
``TRTLLM_SOL_ATTN_STRICT=1`` to raise instead. Errors raised by the kernel
itself are not caught.
"""

from __future__ import annotations

import functools
import os
from typing import Callable, Optional

import torch

from tensorrt_llm._utils import get_sm_version
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


# SM versions with a Sol-Attn CuTe kernel, in ``get_sm_version()`` form
# (major * 10 + minor). Kept in sync with ``sol_attn/interface.py::_CUTE_BACKENDS``;
# duplicated here so the eligibility check does not have to import the CuTe DSL.
SUPPORTED_ARCHS = frozenset({100, 103})


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
    sm = get_sm_version()
    if sm not in SUPPORTED_ARCHS:
        return f"no Sol-Attn kernel for SM{sm}; supported: " + ", ".join(
            f"SM{v}" for v in sorted(SUPPORTED_ARCHS)
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


def _strict() -> bool:
    """Whether TRTLLM_SOL_ATTN_STRICT=1 asks us to raise on an unservable input."""

    return os.environ.get("TRTLLM_SOL_ATTN_STRICT", "0") == "1"


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
#
# Upstream instead wraps the same call in a `torch.library.custom_op` with a
# `register_fake`, which keeps the kernel in the compiled graph as an opaque
# node rather than breaking the graph at it. That is the better end state --
# it removes the per-layer graph break -- and is tracked as a follow-up. It is
# not done here because `torch.compiler.disable` is the convention every other
# CuTe DSL entry point in this repository already follows, and is what the
# reported measurements were taken with.
@torch.compiler.disable
def _run_sol_attn_bthd(
    q,
    k,
    v,
    *,
    tau: float = DEFAULT_TAU,
    thresh_type: str = DEFAULT_THRESH_TYPE,
    kv_splits: int = 1,
    sink_start: int | None = None,
    sink_tokens: int = 0,
    dense_fn: Callable | None = None,
):
    """Run Sol-Attn on contiguous BTHD tensors.

    Inputs the kernel is known not to serve go to dense attention up front;
    errors raised by the kernel itself propagate.
    """

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
        # This is the arm that silently turns Sol-Attn into a no-op for a whole
        # run (wrong arch, head_dim, or dtype), so it must be visible.
        if _strict():
            raise RuntimeError(f"[sol-attn] cannot run the CuTe kernel: {reason}")
        logger.warning_once(
            f"[sol-attn] falling back to dense attention: {reason}. Sol-Attn will not "
            "accelerate this run. Set TRTLLM_SOL_ATTN_STRICT=1 to raise instead.",
            key=("sol_attn_ineligible", reason),
        )
        return dense()

    kernel = _load_sol_attn()
    out = kernel(
        q0,
        k0,
        v0,
        tau=float(tau),
        thresh_type=str(thresh_type),
        # Only 1 split exists on the shipped sm100/sm103 kernels (2/4 was an
        # SM90-only path), so this is not a user-facing knob; the kernel
        # interface rejects anything else.
        kv_splits=int(kv_splits),
        sink_start=sink_start,
        sink_tokens=int(sink_tokens),
    )
    _SOL_STATS["kernel_calls"] += 1
    return out


# Lightweight run-validation counters. `kernel_calls` is the census used to
# prove the CuTe kernel actually ran; `dense_fallback_calls` counts calls the
# kernel was known not to serve. Set TRTLLM_SOL_ATTN_STRICT=1 to raise instead
# of falling back on those.
_SOL_STATS = {"kernel_calls": 0, "dense_fallback_calls": 0}


def reset_sol_attn_stats() -> None:
    """Zero the counters, e.g. after an untimed warmup generation."""

    for key in _SOL_STATS:
        _SOL_STATS[key] = 0


def get_sol_attn_stats() -> dict[str, int]:
    """Return the run-validation counters."""

    return {key: int(value) for key, value in _SOL_STATS.items()}
