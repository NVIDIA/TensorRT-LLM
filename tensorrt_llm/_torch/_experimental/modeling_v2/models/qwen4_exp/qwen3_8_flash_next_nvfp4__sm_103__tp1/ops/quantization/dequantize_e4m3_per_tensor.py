# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Per-tensor FP8 dequantization: float8_e4m3fn -> bf16/fp16/fp32, multiplied by one scale."""

from __future__ import annotations

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — loads the extension registering torch.ops.tensorrt_llm.*
from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

#: The PLE n-gram table's per-tensor dequantization scale in this checkpoint.
PLE_SCALE = 0.00019931793212890625

#: Scale values the whole-domain cells sweep: the checkpoint's, unity, a power
#: of two, an arbitrary one, and the smallest e4m3 subnormal.
_DOMAIN_SCALES = (PLE_SCALE, 1.0, 0.5, 0.037, 2.0**-9)


def _ple(tokens: int) -> dict:
    """`_ple_block`'s call: 16 gathered 160-wide rows per token, one bf16 scale."""
    return dict(shape=(tokens, 16, 160), dtype=torch.bfloat16, scales=(PLE_SCALE,), source="gather")


class DequantizeE4m3PerTensor(OpWrapper):
    """`out = to(scales.dtype)(float(input) * float(scales[0]))`, one rounding.

    Wraps `torch.ops.tensorrt_llm.dequantize_e4m3_per_tensor` -- the
    `tensorrt_llm` namespace; there is no `trtllm::` alias.

    It MULTIPLIES: pass a checkpoint's `weight_scale` straight, never
    reciprocated (the inverse, `static_quantize_e4m3_per_tensor`, divides). The
    output dtype follows `scales`, not `input`: a bf16 scale yields bf16 with no
    cast. Shape is preserved exactly. `scales` has `input`'s rank with every dim
    1 (`[1, 1, 1]` for the PLE's rank-3 gather).

    Fusion boundary: scale-and-widen only. The table gather before it and the
    flatten to `[T, 2560]` after it (a view) are the caller's.
    """

    ARCHS = frozenset({Arch.SM_103})

    CELLS: tuple[Cell, ...] = (
        Cell(why="_ple_block decode: [1, 16, 160] gathered rows, bf16 scale", spec=_ple(1)),
        Cell(why="_ple_block at 8 tokens", spec=_ple(8)),
        Cell(why="_ple_block prefill: [4096, 16, 160]", spec=_ple(4096)),
        Cell(
            why="every e4m3 byte into bf16, five scale values, NaN codes included",
            spec=dict(
                shape=(1, 256), dtype=torch.bfloat16, scales=_DOMAIN_SCALES, source="all_codes"
            ),
        ),
        Cell(
            why="every e4m3 byte into fp16",
            spec=dict(
                shape=(1, 256), dtype=torch.float16, scales=_DOMAIN_SCALES, source="all_codes"
            ),
        ),
        Cell(
            why="every e4m3 byte into fp32",
            spec=dict(
                shape=(1, 256), dtype=torch.float32, scales=_DOMAIN_SCALES, source="all_codes"
            ),
        ),
        Cell(
            why="rank 2: the scale's rank tracks the input's",
            spec=dict(shape=(4, 8), dtype=torch.bfloat16, scales=(PLE_SCALE,), source="randn"),
        ),
        Cell(
            why="rank 5: the deepest rank run",
            spec=dict(
                shape=(2, 2, 2, 4, 8), dtype=torch.bfloat16, scales=(PLE_SCALE,), source="randn"
            ),
        ),
        Cell(
            why="1 x 1: far below the fixed 1024-block grid",
            spec=dict(shape=(1, 1), dtype=torch.bfloat16, scales=(PLE_SCALE,), source="randn"),
        ),
        Cell(
            why="3 x 7: odd shape, no width or alignment rule",
            spec=dict(shape=(3, 7), dtype=torch.bfloat16, scales=(PLE_SCALE,), source="randn"),
        ),
        Cell(
            why="[2^20, 160]: 167,772,160 elements through the grid-stride loop",
            spec=dict(
                shape=(1 << 20, 160), dtype=torch.bfloat16, scales=(PLE_SCALE,), source="randn"
            ),
        ),
        Cell(
            why="scale as a one-element slice at a nonzero storage offset",
            spec=dict(
                shape=(8, 160),
                dtype=torch.bfloat16,
                scales=(PLE_SCALE,),
                source="randn",
                scale_slice=True,
            ),
        ),
        Cell(
            why="input at a non-16-byte-aligned address: same kernel",
            spec=dict(
                shape=(64, 160),
                dtype=torch.bfloat16,
                scales=(PLE_SCALE, 1 / 3, 7.3e-3),
                source="unaligned",
            ),
        ),
    )

    note = """
    Metadata consumed: none; stateless. One kernel per call,
    `scaleMatrix<QuantizeMode 1, ..., __nv_fp8_e4m3>` at a fixed 1024-block
    grid with a grid-stride loop; there is no vectorized variant, so alignment
    is not an axis.

    Preconditions, all enforced loudly by the op (`symmetric_dequantize_per_
    tensor` in thop/fp8Op.cpp): `input` float8_e4m3fn (e5m2, int8, bf16 raise),
    CUDA, contiguous, rank >= 2, non-empty (zero rows or columns raise -- an
    empty batch must branch first); `scales` CUDA, bf16 / fp16 / fp32 (fp64
    raises), of `input`'s rank with every dim 1.

    Degenerate scale values are accepted and computed exactly: 0 zeroes every
    lane (signs kept, no NaN), a negative scale flips signs, +inf sends finite
    lanes to +-inf and zero lanes to NaN, NaN makes everything NaN. Overflow is
    inf, not a clamp (fp16 scale 1000 sends 448 to inf). NaN codes stay NaN.
    `448 * PLE_SCALE` reads back as bf16 0.08935546875.

    CUDA-graph capture is certified: replay picks up a new `input` and a new
    `scales` value written in place (the scale is read from memory at replay).

    Hard limits
      numel  probed to 167,772,160; the int32 boundary 2^31 not probed

    Not certified: `TRTLLM_ENABLE_PDL=1`, `scales` on a different device than
    `input`.
    """

    def __call__(self, input: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
        return torch.ops.tensorrt_llm.dequantize_e4m3_per_tensor(input, scales)

    def reference(self, input: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
        """One fp32 multiply, one conversion to the scales' dtype."""
        return (input.float() * scales.float().flatten()[0]).to(scales.dtype)


dequantize_e4m3_per_tensor = DequantizeE4m3PerTensor()
