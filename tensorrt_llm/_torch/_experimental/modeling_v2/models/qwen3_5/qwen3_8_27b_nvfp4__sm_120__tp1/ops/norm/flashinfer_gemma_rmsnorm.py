# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Gemma-style RMS normalization (weight applied as ``1 + w``) via the flashinfer kernel."""

from __future__ import annotations

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*
from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper


class FlashinferGemmaRmsnorm(OpWrapper):
    """`x / sqrt(mean(x^2, -1) + eps) * (1 + weight)`, one op call, new tensor out.

    The reduction, the `1 + w` and the product are all fp32 inside the kernel;
    only the store rounds to `x.dtype` -- so `(1 + w)` is not first rounded to
    bf16. `x` is 2-D `[T, H]` or 3-D `[B, heads, D]` (a separate per-head
    kernel), normalized over the last dim; `x` is not mutated.

    Fusion boundary: normalization and the biased weight only. No residual add
    (the trtllm op `flashinfer_gemma_fused_add_rmsnorm` does that), no quantization;
    the plain-`w` form is `flashinfer_rmsnorm`.
    """

    ARCHS = frozenset({Arch.SM_120})

    # The first layer's input_layernorm, which has no residual to add, on the
    # contiguous [T, 5120] embedding output.
    CELLS: tuple[Cell, ...] = (
        Cell(
            why="layer-0 input norm, decode 1 token: contiguous [1, 5120]",
            spec=dict(shape=(1, 5120), dtype=torch.bfloat16, eps=1e-6),
        ),
        Cell(
            why="layer-0 input norm, prefill 8192 tokens: contiguous [8192, 5120]",
            spec=dict(shape=(8192, 5120), dtype=torch.bfloat16, eps=1e-6),
        ),
    )

    note = """
    Metadata consumed: none. Registered only when flashinfer is importable.
    Certified path: flashinfer's CuTe DSL `rmsnorm_cute` kernel for 2-D bf16
    `x` (`weight_bias=1.0`), contiguous `[1, 5120]` and `[8192, 5120]`. Not
    certified: strided rows, other widths, other row counts, the 3-D per-head
    kernel (`qk_rmsnorm_cute`) and fp16/fp32 operands. The CUDA-JIT fallback
    (`FLASHINFER_USE_CUDA_NORM=1`, or a DSL that cannot target the device) is a
    different implementation and is not certified. PDL (`TRTLLM_ENABLE_PDL`,
    default on) changes scheduling, not results.

    Loud (measured), so not guarded: 1-D and 4-D `x`, `x.stride(-1) != 1`, a weight of
    the wrong length, dtype or device, and a CPU `x` raise ValueError;
    float64/int8/uint8 raise KeyError from the DSL dtype table. float8_e4m3fn
    and float8_e5m2 are in that table and return a plausible fp8 tensor --
    `is_valid` refuses them.

    Numerics: default tolerance holds for bf16, nothing loosened; the
    largest deviation measured is one bf16 ulp (0.03125) at 8192 rows of 5120.
    The `+ 1` is measured: against a plain-`w` reference the output is ~4 away,
    so the cells tell the two apart.

    Hard limits
      T * H  contiguous inputs above 2^31 - 1 elements route to the strided
      (int64-offset) kernel  boundary 2^31  read from the dispatch, not probed
    """

    def __call__(self, x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
        return torch.ops.trtllm.flashinfer_gemma_rmsnorm(x, weight, eps)

    def reference(self, x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
        """fp32 throughout, `(1 + w)` formed in fp32, one rounding at the end."""
        xf = x.float()
        normed = xf * torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + eps)
        return (normed * (1.0 + weight.float())).to(x.dtype)

    def is_valid(self, x: torch.Tensor, weight: torch.Tensor, eps: float) -> None:
        assert x.dtype not in (torch.float8_e4m3fn, torch.float8_e5m2), (
            f"{x.dtype} is accepted by the kernel's dtype table and returns a "
            "plausible fp8 result; dequantize before normalizing"
        )


flashinfer_gemma_rmsnorm = FlashinferGemmaRmsnorm()
