# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The PyTorch paths the gate projection can take today, as benchmark candidates.

``MiniMaxM3Gate`` holds an FP32 router weight and is handed BF16 hidden states.
For anything wider than a handful of tokens that call lands in
``F.linear(hidden_states.to(torch.float32), weight)``, which is two kernels: a
cast that writes an FP32 copy of the whole activation, then a cuBLAS GEMM that
reads it back. The cast exists only to satisfy cuBLAS' same-dtype requirement,
and it is the larger of the two memory transactions.

That GEMM runs on the TF32 tensor cores, which is the part worth sitting with.
The weight is kept in FP32 to preserve the router's numerics, and then the
multiply rounds it to ten mantissa bits anyway -- so the FP32 activation is
being materialised to feed a kernel that does not use the precision. The
comparison baseline is therefore the TF32 pair, and a replacement is only
interesting if it is both faster and no worse than ``1e-3`` relative.

Also here are two BF16 cuBLAS calls that are not drop-in replacements. They are
speed references: ``bf16 cublas n=128`` is what the tensor cores can do if the
weight is simply truncated, and ``bf16 cublas n=256`` is the same with the
weight split into a high and a low BF16 term, which is the amount of work an
accurate tensor-core path has to do. Nothing can beat the first; the second is
the honest target.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass, field
from typing import Callable, ContextManager, Iterator

import torch


@dataclass(frozen=True)
class Candidate:
    name: str
    build: Callable[[torch.Tensor, torch.Tensor], Callable[[], torch.Tensor]]
    #: Entered around both graph capture and replay. cuBLAS picks its kernel on
    #: the host at capture time, so a flag like ``allow_tf32`` has to be set
    #: then, not inside the timed region.
    context: Callable[[], ContextManager] = field(default=contextlib.nullcontext)
    #: Candidates that only make sense for a subset of the sweep.
    max_tokens: int | None = None


@contextlib.contextmanager
def _tf32(enabled: bool) -> Iterator[None]:
    previous = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = enabled
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous


def _cast_then_linear(x: torch.Tensor, w: torch.Tensor) -> Callable[[], torch.Tensor]:
    return lambda: torch.nn.functional.linear(x.to(torch.float32), w)


def _bf16_truncated(x: torch.Tensor, w: torch.Tensor) -> Callable[[], torch.Tensor]:
    w_bf16 = w.to(torch.bfloat16)
    return lambda: torch.nn.functional.linear(x, w_bf16).to(torch.float32)


def _bf16_split(x: torch.Tensor, w: torch.Tensor) -> Callable[[], torch.Tensor]:
    """Weight as ``hi + lo``, both BF16, stacked into one N=256 GEMM.

    Splitting the FP32 weight into two BF16 terms recovers about sixteen
    mantissa bits, comfortably past TF32's ten, and the two halves ride in a
    single GEMM as extra columns rather than a second launch. The split itself
    is hoisted out of the timed region because in production the router weight
    is a parameter: it would be split once at load.

    Output stays BF16 here, which is not usable as-is; this candidate is a
    speed reference for the tensor-core work an accurate path must do, so its
    accuracy column is expected to be poor.
    """
    w_hi = w.to(torch.bfloat16)
    w_lo = (w - w_hi.to(torch.float32)).to(torch.bfloat16)
    w_cat = torch.cat([w_hi, w_lo], dim=0).contiguous()
    n = w.shape[0]

    def run() -> torch.Tensor:
        y = torch.nn.functional.linear(x, w_cat).to(torch.float32)
        return y[:, :n] + y[:, n:]

    return run


def _precast_linear(x: torch.Tensor, w: torch.Tensor) -> Callable[[], torch.Tensor]:
    """The GEMM alone, with the cast hoisted out of the timed region.

    Not a candidate -- an FP32 activation cannot appear for free. It splits the
    baseline into its two halves, so the cast and the kernel choice can be
    blamed separately.
    """
    x_f32 = x.to(torch.float32)
    return lambda: torch.nn.functional.linear(x_f32, w)


def _tf32_split(x: torch.Tensor, w: torch.Tensor) -> Callable[[], torch.Tensor]:
    """Two TF32 terms for the weight, cast hoisted, as an N=256 GEMM.

    TF32 keeps ten mantissa bits, so two terms cover about twenty-one and the
    weight stops being the limiting error. Reference for what an accurate
    TF32-tensor-core path costs, against the BF16 equivalent below.
    """
    x_f32 = x.to(torch.float32)
    hi = (w.view(torch.int32) & -0x2000).view(torch.float32)  # truncate to TF32
    lo = w - hi
    w_cat = torch.cat([hi, lo], dim=0).contiguous()
    n = w.shape[0]

    def run() -> torch.Tensor:
        y = torch.nn.functional.linear(x_f32, w_cat)
        return y[:, :n] + y[:, n:]

    return run


def torch_candidates() -> list[Candidate]:
    return [
        # What runs today. The router GEMM lands on the TF32 tensor cores, so
        # the FP32 weight buys nothing at the multiply: cuBLAS rounds it to ten
        # mantissa bits on the way in. The cast is still paid in full, because
        # the tensor cores want a 32-bit activation either way.
        Candidate("cast + cublas tf32", _cast_then_linear, lambda: _tf32(True)),
        # The same call forced off the tensor cores, i.e. what the FP32 weight
        # would cost if its mantissa were actually respected. Here to show what
        # the current path is trading away, not as a live alternative.
        Candidate("cast + cublas fp32", _cast_then_linear, lambda: _tf32(False)),
        # Lower bound on runtime: one BF16 tensor-core pass, no cast at all.
        Candidate("bf16 cublas n=128", _bf16_truncated),
        # Lower bound for an accurate tensor-core path: two BF16 passes' worth.
        Candidate("bf16 cublas n=256", _bf16_split),
    ]


def reference_candidates() -> list[Candidate]:
    """Not achievable, but they bound the design space. See ``--refs``."""
    return [
        Candidate("[ref] gemm only, fp32", _precast_linear, lambda: _tf32(False)),
        Candidate("[ref] gemm only, tf32", _precast_linear, lambda: _tf32(True)),
        Candidate("[ref] gemm only, tf32 n=256", _tf32_split, lambda: _tf32(True)),
    ]


def triton_gemv_candidate() -> Candidate | None:
    """The existing decode GEMV, for the narrow end of the sweep.

    It falls back to ``cast + cublas`` past ``MAX_GEMV_TOKENS`` tokens, so
    registering it beyond that would just duplicate a column -- and the default
    sweep starts above that line, which is the whole reason the TF32 pair is
    the baseline. Reach it with ``--tokens 1 8 16``.
    """
    try:
        from ._repo_import import import_bare

        module = import_bare("tensorrt_llm._torch.modules.fp32_router_gemm")
    except ImportError:
        return None
    MAX_GEMV_TOKENS = module.MAX_GEMV_TOKENS
    fp32_router_gemm = module.fp32_router_gemm

    return Candidate(
        "triton gemv (existing)",
        lambda x, w: (lambda: fp32_router_gemm(x, w)),
        max_tokens=MAX_GEMV_TOKENS,
    )
