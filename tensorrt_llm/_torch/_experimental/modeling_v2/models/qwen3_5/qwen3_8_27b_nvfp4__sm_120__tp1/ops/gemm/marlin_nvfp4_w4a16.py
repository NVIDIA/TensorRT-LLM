# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""W4A16 NVFP4 projection on the target's Marlin GEMM (``native_ops.marlin_nvfp4_w4a16_gemm``,
built from the target's ``csrc/marlin``, a copy of TensorRT-LLM's dense Marlin NVFP4 GEMM
with this target's decode tuning).

BF16 activations against a Marlin-packed NVFP4 weight: the decode copies of this
checkpoint's FP8 per-tensor projections (GDN in_proj_qkvz and out_proj, attention
qkv_proj and o_proj; built at load by ``_nvfp4_recipes.fp8_to_nvfp4_values`` +
``marlin_pack``) and the NVFP4 lm_head repacked as stored
(``_nvfp4_recipes.lm_head_marlin``). The weight streams once per call, which is
what decode rows pay for; the activation is never quantized.

The call selects two things the kernel cannot infer:

* ``workspace_mode`` -- 0: split-K partial sums chained through the BF16 output, a
  persistent per-stream lock workspace, PDL launch; 2: the same with the split-K
  partials reduced in FP32, so every output is rounded to BF16 once;
* ``launch_hint`` -- 0: the stock launch-shape dispatcher; 1: the Qwen3.8 launch
  tables for (N, K) in {(16384, 5120), (14336, 5120), (5120, 6144)} on 188-SM parts
  (rows <= 8 a full-K grid, rows 9-16 and 17-32 per-shape tables, the in-proj
  shapes prefetching their first weight stages into L2 before the PDL wait). Any
  other shape or SM count takes the stock dispatcher under either hint.
"""

from __future__ import annotations

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import (
    Arch,
    Cell,
    OpWrapper,
    true_fp32_matmul,
)

from ... import native_ops
from ..quantization._nvfp4_reference import dequantize_marlin, unpack_marlin_scales

#: Output columns the reference dequantizes at a time (the lm_head weight is 1.27e9
#: values; 8192 columns of it are 168 MB in fp32).
_REF_CHUNK_N = 8192


def _cell(why: str, n: int, k: int, m: int, workspace_mode: int, launch_hint: int) -> Cell:
    return Cell(
        why=why, spec=dict(n=n, k=k, m=m, workspace_mode=workspace_mode, launch_hint=launch_hint)
    )


class MarlinNvfp4W4a16(OpWrapper):
    """``bf16(x @ dequant(W).T)`` for BF16 ``x`` ``[M, size_k]`` and a Marlin NVFP4 weight.

    ``dequant(W) = e2m1 * sf * g`` with ``sf`` the E4M3 block scale (one per 16 K) and
    ``g`` the global scale the BF16 ``weight_global_scale`` carries (times 2^119);
    blocks whose E4M3 scale is subnormal dequantize to zero. Returns a fresh
    ``[M, size_n]`` BF16 tensor.
    """

    ARCHS = frozenset({Arch.SM_120})

    # Qwen3.8-27B tp1 decode rows that reach Marlin (the dispatch table): GDN in_proj_qkvz
    # (16384, 5120) and the (5120, 6144) out_proj / o_proj at rows 4-32 (rows 1-3 take the
    # GEMV), attention qkv_proj (14336, 5120) at 3-64 (1-2 take the GEMV); mode 0 up to 16
    # rows, mode 2 above. The lm_head (248320, 5120) at rows 1-55, mode 0, stock launch.
    CELLS: tuple[Cell, ...] = (
        _cell("qkvz M=4: first Marlin row, full-K grid, pre-wait prefetch", 16384, 5120, 4, 0, 1),
        _cell("qkvz M=8: last full-K row", 16384, 5120, 8, 0, 1),
        _cell("qkvz M=16: rows 9-16 table, last BF16-chain row", 16384, 5120, 16, 0, 1),
        _cell("qkvz M=17: rows 17-32 table, FP32 reduce", 16384, 5120, 17, 2, 1),
        _cell("qkvz M=32: the qkvz Marlin cap", 16384, 5120, 32, 2, 1),
        _cell("qkv M=3: first Marlin row of qkv_proj", 14336, 5120, 3, 0, 1),
        _cell("qkv M=8: full-K grid, 112 CTAs", 14336, 5120, 8, 0, 1),
        _cell("qkv M=16: rows 9-16 table", 14336, 5120, 16, 0, 1),
        _cell("qkv M=17: rows 17-32 table, FP32 reduce", 14336, 5120, 17, 2, 1),
        _cell("qkv M=33: generic launch above the tables, FP32 reduce", 14336, 5120, 33, 2, 1),
        _cell("qkv M=64: the qkv Marlin cap", 14336, 5120, 64, 2, 1),
        _cell("out_proj/o_proj M=4: exact full-K replay of the split-K chain", 5120, 6144, 4, 0, 1),
        _cell("out_proj/o_proj M=8: last full-K row", 5120, 6144, 8, 0, 1),
        _cell("out_proj/o_proj M=16: rows 9-16 table", 5120, 6144, 16, 0, 1),
        _cell("out_proj/o_proj M=17: rows 17-32 table, FP32 reduce", 5120, 6144, 17, 2, 1),
        _cell("out_proj/o_proj M=32: the (5120, 6144) Marlin cap", 5120, 6144, 32, 2, 1),
        _cell("lm_head M=1: decode, stock launch", 248320, 5120, 1, 0, 0),
        _cell("lm_head M=55: last Marlin lm_head row", 248320, 5120, 55, 0, 0),
    )

    note = """
    Metadata consumed: none. The weight triple and the padded sizes come from
    `_nvfp4_recipes.marlin_pack` / `lm_head_marlin` (`MarlinNvfp4Weight`).

    Preconditions: `x` BF16, dense row-major `[M, size_k]`; `weight` int32
    `[size_k/16, 2*size_n]`, `weight_scale` `[size_k/16, size_n]`, `weight_global_scale`
    BF16 `[1]`; size_n a multiple of 128, size_k of 64.

    Numerics: Marlin dequantizes to BF16 through an S0E5M3 scale, which flushes blocks
    with a subnormal E4M3 scale (< 2^-6) to zero -- `reference` models it. Tensor-core
    BF16 products accumulate in fp32. Mode 0 rounds each split-K partial to BF16 as it
    chains through the output (a few BF16 roundings per output); mode 2 reduces the
    partials in fp32 (one rounding). The full-K launches of `launch_hint=1` replay the
    split-K chain of the launch they replace bitwise for (5120, 6144); for the in-proj
    shapes they round each column once.

    PDL and workspace: modes 0 and 2 launch with PDL (subject to TRTLLM_ENABLE_PDL) on
    a zeroed lock workspace kept per stream; every slot reads zero again after the
    call. Under `launch_hint=1` the qkv/qkvz launches prefetch their first weight stages
    into L2 before `griddepcontrol.wait`, so a producer launched before this call may
    still be writing `x` when the kernel starts -- `x` is read only after the wait.

    Hard limits
      M        the target calls rows <= 64 (FP8 copies) / <= 55 (lm_head); larger M
               runs further 64-row passes        not certified above 64
      tables   only on 188-SM parts and the three (N, K) above; elsewhere stock
    """

    def __call__(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
        weight_global_scale: torch.Tensor,
        size_n: int,
        size_k: int,
        workspace_mode: int,
        launch_hint: int,
    ) -> torch.Tensor:
        return native_ops.marlin_nvfp4_w4a16_gemm(
            x,
            weight,
            weight_scale,
            weight_global_scale,
            size_n,
            size_k,
            workspace_mode,
            launch_hint,
        )

    def reference(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
        weight_global_scale: torch.Tensor,
        size_n: int,
        size_k: int,
        workspace_mode: int,
        launch_hint: int,
    ) -> torch.Tensor:
        """Unpack the Marlin layout, dequantize exactly (subnormal-scale blocks zeroed),
        one fp32 product (TF32 off), one rounding. `workspace_mode` and `launch_hint`
        change summation order only and are ignored."""
        scales = unpack_marlin_scales(weight_scale, size_k, size_n)
        out = torch.empty((x.shape[0], size_n), dtype=torch.bfloat16, device=x.device)
        xf = x.float()
        with true_fp32_matmul():
            for n0 in range(0, size_n, _REF_CHUNK_N):
                n1 = min(size_n, n0 + _REF_CHUNK_N)
                w = dequantize_marlin(
                    weight, weight_scale, weight_global_scale, size_k, size_n, n0, n1, scales
                )
                out[:, n0:n1] = (xf @ w.t()).to(torch.bfloat16)
        return out

    def is_valid(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
        weight_global_scale: torch.Tensor,
        size_n: int,
        size_k: int,
        workspace_mode: int,
        launch_hint: int,
    ) -> None:
        # The op reads `x` as dense [M, size_k] from its data pointer and the weight as
        # [size_k/16, 2*size_n] tiles; it checks neither strides nor that the sizes match
        # the tensors, so either mistake returns a plausible wrong product.
        assert x.dim() == 2 and x.is_contiguous(), "x must be a dense row-major [M, K] tensor"
        assert x.shape[1] == size_k, f"x has {x.shape[1]} columns, size_k is {size_k}"
        assert weight.numel() == size_k // 16 * size_n * 2, (
            "weight is not a Marlin [size_k/16, 2*size_n] int32 tensor for these sizes"
        )
        assert weight_scale.numel() == size_k // 16 * size_n, (
            "weight_scale is not a [size_k/16, size_n] Marlin scale tensor for these sizes"
        )
        assert weight_global_scale.dtype == torch.bfloat16, (
            "weight_global_scale is read as BF16 (nvfp4_marlin_process_global_scale)"
        )


marlin_nvfp4_w4a16 = MarlinNvfp4W4a16()
