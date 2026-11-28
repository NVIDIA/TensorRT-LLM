# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""`hc_post_collapse_norm` of the tensor-parallel sum of the previous sub-layer's partial output, the one-shot
all-reduce (and, for the MoE, the top-k combine) done inside the same launch."""

from __future__ import annotations

from typing import Optional

import torch

from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper
from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.residual.hc_post_collapse_norm import (
    hc_post_collapse_norm,
)


class HcAllreducePostCollapseNorm(OpWrapper):
    """`(x_partial, residual, post_prev, comb_prev, pre_mix, norm_weight, norm_eps, allreduce)` ->
    `(residual_cur, h)` as `hc_post_collapse_norm(sum over ranks of x_partial, ...)`.

    `x_partial` is this rank's partial sum `[n, d]` bf16, or the engine's `MoeUnfinalized` (the routed expert
    rows before the top-k combine plus the shared-expert partial), which the kernel finalizes in fp32 as it
    reduces. `allreduce` is the engine's Lamport workspace of the group (`LamportAllReduce` over the IPC
    workspace inside one tray, `MnnvlLamportAllReduce` over the fabric across trays), built by the target once
    at load time (`mhc_lamport_allreduce`); the kernel advances the same flags and triple buffer as the engine's
    one-shot all-reduce, so the two interleave. The sum is fp32 in rank order, rounded once to bf16, so the result
    is bit-equal to all-reducing and then `hc_post_collapse_norm`.

    With an fp32 `nvfp4_scale [1]` (the MoE's input global scale, as stock's FFN boundary takes it) a third element:
    `(fp4 [n, d // 2] uint8, scales [n * d // 16] uint8)`, `h` quantized as `fp4_quantize(h, nvfp4_scale, 16, False,
    False)` (row-major scales), written by the same launch.
    """

    ARCHS = frozenset({Arch.SM_103})
    CELLS = (
        Cell(
            why="decode rows 6, bf16 partial, 4 ranks of one tray",
            spec=dict(n=6, d=5120, world=4, moe=False),
        ),
        Cell(why="decode rows 24, bf16 partial", spec=dict(n=24, d=5120, world=4, moe=False)),
        Cell(
            why="decode rows 24, MoE rows before the top-k combine",
            spec=dict(n=24, d=5120, world=4, moe=True),
        ),
    )
    note = """
    A collective: every rank of the group must call it in the same order, and it shares the engine's one-shot
    workspace with `AllReduce`, so calls interleave only with that. Verified by the multi-rank test (needs
    `world` GPUs). `reference` needs a way to sum over the group: the test attaches `reference_sum` to the
    workspace object.
    """

    def __call__(
        self,
        x_partial,
        residual: torch.Tensor,
        post_prev: torch.Tensor,
        comb_prev: torch.Tensor,
        pre_mix: torch.Tensor,
        norm_weight: torch.Tensor,
        norm_eps: float,
        allreduce,
        nvfp4_scale: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        from tensorrt_llm._torch.modules.mhc.mhc_lowm import mhc_allreduce_post_collapse_norm

        return mhc_allreduce_post_collapse_norm(
            x_partial,
            residual,
            post_prev,
            comb_prev,
            pre_mix,
            norm_weight,
            norm_eps,
            allreduce,
            nvfp4_scale=nvfp4_scale,
        )

    def reference(
        self,
        x_partial,
        residual: torch.Tensor,
        post_prev: torch.Tensor,
        comb_prev: torch.Tensor,
        pre_mix: torch.Tensor,
        norm_weight: torch.Tensor,
        norm_eps: float,
        allreduce,
        nvfp4_scale: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """The group's sum (`allreduce.reference_sum`, finalizing a `MoeUnfinalized` first), then the split boundary."""
        part = x_partial.finalize() if hasattr(x_partial, "finalize") else x_partial
        out = hc_post_collapse_norm.reference(
            allreduce.reference_sum(part),
            residual,
            post_prev,
            comb_prev,
            pre_mix,
            norm_weight,
            norm_eps,
        )
        if nvfp4_scale is None:
            return out
        fp4, scales = torch.ops.trtllm.fp4_quantize(out[1], nvfp4_scale, 16, False, False)
        return (*out, (fp4, scales.reshape(-1)))

    def is_valid(
        self,
        x_partial,
        residual: torch.Tensor,
        post_prev: torch.Tensor,
        comb_prev: torch.Tensor,
        pre_mix: torch.Tensor,
        norm_weight: torch.Tensor,
        norm_eps: float,
        allreduce,
        nvfp4_scale: Optional[torch.Tensor] = None,
    ) -> None:
        part = x_partial.shared if hasattr(x_partial, "finalize") else x_partial
        hc_post_collapse_norm.is_valid(
            part, residual, post_prev, comb_prev, pre_mix, norm_weight, norm_eps
        )
        assert post_prev is not None, (
            "the all-reduce is of a sub-layer output: there is a post update to apply"
        )
        assert allreduce.nranks in (2, 4, 8) and part.is_contiguous()
        assert nvfp4_scale is None or (
            nvfp4_scale.dtype == torch.float32 and nvfp4_scale.numel() == 1
        ), "nvfp4_scale is the fp32 [1] global scale of the NVFP4 copy"
        if hasattr(x_partial, "finalize"):
            m = x_partial
            assert (
                m.fc2.dtype == torch.bfloat16
                and m.perm.dtype == torch.int32
                and m.perm.is_contiguous()
            )
            assert m.scales.shape == m.perm.shape and m.scales.is_contiguous()
            assert all(t.data_ptr() % 16 == 0 for t in (m.fc2, m.scales, m.perm, m.shared)), (
                "the kernel reads the MoE operands as 16-byte aligned tensors "
                "(a permutation viewed from a routing table needs the table's rows aligned)"
            )


hc_allreduce_post_collapse_norm = HcAllreducePostCollapseNorm()
