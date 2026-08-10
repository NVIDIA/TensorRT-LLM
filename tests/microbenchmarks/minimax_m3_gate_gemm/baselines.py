# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The PyTorch paths the gate projection can take today, as benchmark candidates.

The baseline is `F.linear(hidden_states.to(torch.float32), weight)`, which is two
kernels: a cast that writes an FP32 copy of the whole activation, then a cuBLAS
GEMM that reads it back. The cast is the larger of the two memory transactions
and exists only to satisfy cuBLAS' same-dtype requirement, and the GEMM it feeds
lands on the TF32 tensor cores, which round the FP32 weight to ten mantissa bits
regardless. A replacement is interesting only if it is both faster and no worse
than 1e-3 relative.

Two BF16 cuBLAS calls are here as speed references rather than drop-in
replacements. `bf16 cublas n=128` is what the tensor cores manage with the weight
truncated, which nothing can beat, and `bf16 cublas n=256` is the same with a
high and a low BF16 term, which is the work an accurate tensor-core path must do.
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
    #: the host at capture time, so a flag like `allow_tf32` has to be set then,
    #: not inside the timed region.
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
    """Weight as hi plus lo, both BF16, stacked into one N=256 GEMM.

    Two BF16 terms recover about sixteen mantissa bits, past TF32's ten, and both
    ride in a single GEMM as extra columns rather than a second launch. The split
    is hoisted out of the timed region because the router weight is a parameter
    and would be split once at load.

    The output stays BF16, which is not usable as-is. This is a speed reference
    for the tensor-core work an accurate path must do, so its accuracy column is
    expected to be poor.
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

    Not a candidate, since an FP32 activation cannot appear for free. It splits
    the baseline in two so the cast and the kernel choice can be blamed
    separately.
    """
    x_f32 = x.to(torch.float32)
    return lambda: torch.nn.functional.linear(x_f32, w)


def _tf32_split(x: torch.Tensor, w: torch.Tensor) -> Callable[[], torch.Tensor]:
    """Two TF32 terms for the weight, cast hoisted, as an N=256 GEMM.

    TF32 keeps ten mantissa bits, so two terms cover about twenty-one and the
    weight stops being the limiting error. Reference for what an accurate TF32
    tensor-core path costs, against the BF16 equivalent.
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
        # What runs today, and the baseline the table compares against.
        Candidate("cast + cublas tf32", _cast_then_linear, lambda: _tf32(True)),
        # The same call forced off the tensor cores, which is what the FP32
        # weight would cost with its mantissa respected. Not a live alternative.
        Candidate("cast + cublas fp32", _cast_then_linear, lambda: _tf32(False)),
        # Lower bound on runtime: one BF16 tensor-core pass, no cast at all.
        Candidate("bf16 cublas n=128", _bf16_truncated),
        # Lower bound for an accurate tensor-core path: two BF16 passes' worth.
        Candidate("bf16 cublas n=256", _bf16_split),
    ]


def reference_candidates() -> list[Candidate]:
    """Not achievable, but they bound the design space. Enabled by --refs."""
    return [
        Candidate("[ref] gemm only, fp32", _precast_linear, lambda: _tf32(False)),
        Candidate("[ref] gemm only, tf32", _precast_linear, lambda: _tf32(True)),
        Candidate("[ref] gemm only, tf32 n=256", _tf32_split, lambda: _tf32(True)),
    ]


def triton_gemv_candidate() -> Candidate | None:
    """The decode GEMV, for the narrow end of the sweep.

    It falls back to the cast plus cuBLAS pair past `MAX_GEMV_TOKENS` tokens, so
    registering it beyond that would duplicate a column. The default sweep starts
    above that line, so reach it with --tokens 1 8 16.
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
