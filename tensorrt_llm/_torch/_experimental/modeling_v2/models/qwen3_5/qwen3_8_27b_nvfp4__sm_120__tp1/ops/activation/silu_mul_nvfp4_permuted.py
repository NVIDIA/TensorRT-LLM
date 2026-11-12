# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SwiGLU + NVFP4 quantize of down_proj's input, from a permuted gate_up output.

Below the rows where gate_up runs the SwiGLU-epilogue GEMM, gate_up is a plain
NVFP4 GEMM over the load-time permuted weight, so its BF16 output columns are in
``gate_up_swiglu_row_permutation`` order. One Triton kernel reads gate/up pairs
back through that permutation, applies ``silu(gate) * up`` rounded to BF16 and
quantizes to NVFP4 with down_proj's static global scale -- bitwise
``silu_and_mul`` + ``fp4_quantize`` on the unpermuted output, scale padding
included. Rows 5-32 run a decode kernel (one thread per 16-output scale block),
every other row count the base kernel (with PDL while its grid fits one wave).
Kernels in ``_silu_mul_nvfp4``.
"""

from __future__ import annotations

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper
from tensorrt_llm._torch._experimental.modeling_v2.catalog.quantization.fp4_quantize import (
    fp4_quantize,
)

from ..quantization._nvfp4_recipes import gate_up_swiglu_row_permutation
from ._silu_mul_nvfp4 import silu_mul_nvfp4_permuted as _launch

_SF_VEC = 16


def _cell(why: str, m: int) -> Cell:
    return Cell(why=why, spec=dict(m=m, d=17408))


class SiluMulNvfp4Permuted(OpWrapper):
    """``fp4_quantize(bf16(silu(gate) * up), global_scale, 16)`` with swizzled scales,
    where ``[gate | up] = x[:, argsort(perm)]`` and ``perm`` =
    ``gate_up_swiglu_row_permutation(2d)``. ``x`` BF16 ``[M, 2d]``; returns ``(fp4 [M,
    d/2] uint8, sf uint8 [ceil128(M) * ceil4(d/16)])``, padding rows of ``sf`` zero."""

    ARCHS = frozenset({Arch.SM_120})

    # Qwen3.8-27B MLP (intermediate 17408) at rows 1-40 (the epilogue GEMM takes 41 and
    # up), and at every row count while the AutoTuner tunes (the unfused gate_up path).
    CELLS: tuple[Cell, ...] = (
        _cell("M=1: base kernel, PDL", 1),
        _cell("M=4: base kernel's last decode row, PDL", 4),
        _cell("M=5: first decode-kernel row", 5),
        _cell("M=16: decode kernel", 16),
        _cell("M=32: last decode-kernel row", 32),
        _cell("M=33: base kernel above one wave, no PDL", 33),
        _cell("M=40: last row below the SwiGLU-epilogue GEMM", 40),
        _cell("M=2048: the AutoTuner warmup's unfused prefill", 2048),
    )

    note = """
    Metadata consumed: none. `global_scale` is down_proj's activation global scale
    (fp32 `[1]`).

    Preconditions: `x` BF16 `[M, 2d]` with unit inner stride (any row stride), `d` a
    multiple of 64, columns in `gate_up_swiglu_row_permutation(2d)` order (a gate_up
    computed on the permuted weight).

    Numerics: exact -- bitwise `silu_and_mul` (Triton `a * sigmoid(a) * b`, BF16) then
    `fp4_quantize(.., global_scale, 16)` with swizzled scales on the unpermuted columns,
    both kernels. Against `reference` (torch's sigmoid) only the BF16 intermediate's
    last bit can differ, and with it a block's scale or a code.

    PDL: the base kernel launches with PDL when TRTLLM_ENABLE_PDL is on and its grid
    (`(M + 1) x ceil(d / 512)` programs) is smaller than the SM count; it releases its
    dependents at once and waits before every access. The decode kernel has no PDL.
    """

    def __call__(
        self, x: torch.Tensor, global_scale: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return _launch(x, global_scale)

    def reference(
        self, x: torch.Tensor, global_scale: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Un-permute, ``bf16(gate * sigmoid(gate) * up)`` in fp32, `fp4_quantize`'s
        native-torch NVFP4 with swizzled scales."""
        n = x.shape[1]
        d = n // 2
        inv = torch.argsort(gate_up_swiglu_row_permutation(n).to(x.device))
        h = x[:, inv].float()
        gate, up = h[:, :d], h[:, d:]
        act = (gate * torch.sigmoid(gate) * up).to(torch.bfloat16)
        return fp4_quantize.reference(act, global_scale, _SF_VEC, False, True)

    def is_valid(self, x: torch.Tensor, global_scale: torch.Tensor) -> None:
        # The kernels load global_scale[0] only.
        assert global_scale.numel() == 1, "global_scale must hold exactly one element"
        assert global_scale.dtype == torch.float32, "global_scale is read as fp32"


silu_mul_nvfp4_permuted = SiluMulNvfp4Permuted()
