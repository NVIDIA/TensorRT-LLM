# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi K3.1 MegaMoE input on SM107: a chunk's rows quantized to NVFP4 straight into its bucket."""

from __future__ import annotations

import torch

from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper

from ...kernels.moe.rubin_megamoe_quantize import BLOCKS_PER_PROGRAM, SF_VEC
from ...kernels.moe.rubin_megamoe_quantize import quantize_into_bucket as _launch


def _cell(why: str, rows: int, bucket: int, **spec) -> Cell:
    return Cell(why=why, spec=dict(rows=rows, bucket=bucket, **spec))


class RubinMegamoeQuantize(OpWrapper):
    """``fp4_quantize(x, global_scale, 16, False, False)`` written into a MegaMoE bucket.

    ``x`` BF16 ``[n, hidden]`` (``1 <= n <= T``) and ``global_scale`` FP32 ``[1]``; the bucket's
    ``activation`` U8 ``[T, hidden / 2]`` and ``activation_sf`` U8 ``[T, hidden / 16]`` take the
    NVFP4 values and the linear E4M3 block scales of the ``n`` rows in rows ``[0, n)``, the bytes
    ``fp4_quantize`` returns; its ``topk_idx`` int32 / ``topk_weights`` FP32 ``[T, 16]`` take -1 /
    0 in rows ``[n, T)``, the routing of no row. Nothing else is written; one launch on the current
    stream. Returns nothing.

    With the routing's rows written by ``rubin_route_topk_out``, the bucket then holds what the
    launcher's staging would have put there, and ``rubin_megamoe_in_bucket`` launches on it.
    """

    ARCHS = frozenset({Arch.SM_107})

    CELLS: tuple[Cell, ...] = (
        _cell("DEP16 generation b12 k7: 96 rows into bucket 128, input scale 1", 96, 128),
        _cell("a full bucket: 128 rows, no routing tail", 128, 128),
        _cell("one row into the smallest bucket", 1, 32),
        _cell("40 rows into bucket 64 at an input scale other than 1", 40, 64, scale=0.37),
        _cell(
            "blocks of zeros, tiny and large magnitudes in one row: zero and saturated scales",
            24,
            32,
            mode="edges",
        ),
        _cell("generation with context rows: 300 rows into bucket 384", 300, 384),
    )

    note = """
    The kernel reads ``x`` and the four buffers as dense rows (``topk_weights`` as the bucket
    holds it, ``top_k`` wide) and checks nothing; ``hidden`` is a multiple of 512 (the K3.1
    latent, 3584, is). It writes the routing tail on every call, so what the bucket holds past
    the rows never depends on an earlier call or on the order CUDA graphs were captured in. A
    block with a NaN is not served.
    """

    def __call__(
        self,
        x: torch.Tensor,
        global_scale: torch.Tensor,
        activation: torch.Tensor,
        activation_sf: torch.Tensor,
        topk_idx: torch.Tensor,
        topk_weights: torch.Tensor,
    ) -> None:
        _launch(x, global_scale, activation, activation_sf, topk_idx, topk_weights)

    def reference(
        self,
        x: torch.Tensor,
        global_scale: torch.Tensor,
        activation: torch.Tensor,
        activation_sf: torch.Tensor,
        topk_idx: torch.Tensor,
        topk_weights: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``(values U8 [n, hidden / 2], scales U8 [n, hidden / 16])``: what rows ``[0, n)`` of
        ``activation`` / ``activation_sf`` receive, as the ``fp4_quantize`` entry's reference
        computes it (linear scales). The four buffers are not read; the routing tail (-1 / 0) is a
        constant the test checks itself."""
        # Imported here: the catalog entry loads the engine's op library, which only a test needs.
        from tensorrt_llm._torch.modeling_v2.catalog.quantization.fp4_quantize import (  # noqa: E501
            fp4_quantize,
        )

        values, scales = fp4_quantize.reference(x, global_scale, SF_VEC, False, False)
        return values, scales.view(x.shape[0], x.shape[1] // SF_VEC)

    def is_valid(
        self,
        x: torch.Tensor,
        global_scale: torch.Tensor,
        activation: torch.Tensor,
        activation_sf: torch.Tensor,
        topk_idx: torch.Tensor,
        topk_weights: torch.Tensor,
    ) -> None:
        """The kernel writes raw rows: a misshaped, mistyped or strided buffer is written wrong."""
        n, hidden = x.shape
        t = topk_idx.shape[0]
        assert x.dtype == torch.bfloat16 and x.is_contiguous(), "x must be dense BF16 rows"
        assert hidden % (SF_VEC * BLOCKS_PER_PROGRAM) == 0, (
            f"hidden {hidden} is not a multiple of 512"
        )
        assert 1 <= n <= t, f"{n} rows for a bucket of {t}"
        assert global_scale.numel() == 1 and global_scale.dtype == torch.float32, "one FP32 scale"
        assert tuple(activation.shape) == (t, hidden // 2) and activation.dtype == torch.uint8
        assert tuple(activation_sf.shape) == (t, hidden // SF_VEC), (
            f"linear block scales [{t}, {hidden // SF_VEC}], got {tuple(activation_sf.shape)}"
        )
        assert activation_sf.dtype == torch.uint8
        assert topk_idx.dtype == torch.int32 and topk_weights.dtype == torch.float32
        assert topk_weights.shape == topk_idx.shape, (topk_weights.shape, topk_idx.shape)
        for name, b in (("activation", activation), ("activation_sf", activation_sf),
                        ("topk_idx", topk_idx), ("topk_weights", topk_weights)):  # fmt: skip
            assert b.is_contiguous(), f"{name} must be dense rows"
            assert b.device == x.device, f"{name} is on another device"
        assert activation.data_ptr() % 4 == 0, "activation is written as 32-bit words"


rubin_megamoe_quantize = RubinMegamoeQuantize()
