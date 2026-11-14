# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The tensor-parallel all-reduce of a decode step's partial rows fused with the mHC boundary after it.

One cluster of 16 CTAs runs TensorRT-LLM's Lamport one-shot all-reduce on the TP group's all-reduce
workspace, so its calls interleave with the library's one-shot all-reduce on that workspace, and then
the boundary between two mHC-wrapped blocks: the previous block's post-mapping into the residual
streams and the next block's pre-mapping (mixer, Sinkhorn-normalized stream combination, RMS-normalized
layer input). It takes decode steps of at most 8 rows in place of the library all-reduce and
``mHC.fused_hc``. The CTA's slice of the mixer weights is loaded before the grid-dependency wait, and the 16 CTAs
add their partial mixer sums over distributed shared memory in a fixed order. The kernel is in
``csrc/mhcArBoundary.cu``.
"""

from __future__ import annotations

from typing import Sequence

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import (
    Arch,
    Cell,
    OpWrapper,
    true_fp32_matmul,
)
from tensorrt_llm._torch.flashinfer_utils import get_env_enable_pdl

from ... import native_ops
from ..mhc.mhc_big_fuse import sinkhorn

WORLD = 4
MAX_ROWS = 8
MULT = 4
HIDDEN = 4096


class MhcArBoundary(OpWrapper):
    """The mHC boundary after a block whose output each of the 4 ranks holds a partial of.

    ``partial`` is this rank's ``[M, 4096]`` BF16 share of the block output; their sum over the group,
    rounded to BF16 like the library's one-shot all-reduce, is the block output ``x``. ``residual``
    ``[M, 4, 4096]`` BF16 holds the streams, ``post`` ``[M, 4]`` and ``comb`` ``[M, 4, 4]`` FP32 the
    previous block's coefficients; ``fn``, ``scale`` and ``base`` are the next block's mixer and
    ``norm_weight`` its input RMSNorm. ``workspace`` is the group's all-reduce workspace and ``rank``
    this rank in it.

    Returns ``(residual_out, post_out, comb_out, layer_input)``: the streams ``post * x + comb^T
    streams`` (BF16), the next block's ``post`` and Sinkhorn-normalized ``comb`` (FP32, ``[M, 4]`` and
    ``[M, 4, 4]``) and its RMS-normalized layer input (BF16) -- what the library all-reduce followed by
    ``mHC.fused_hc(..., norm_weight=...)`` returns, ``post`` without its trailing unit dimension.
    """

    ARCHS = frozenset({Arch.SM_100})

    # GLM-5.3-Flash at tp4: hc_mult 4, hidden 4096, 20 Sinkhorn iterations.
    _GLM = dict(world=WORLD, mult=MULT, hidden=HIDDEN, sinkhorn_iters=20)
    CELLS: tuple[Cell, ...] = (
        Cell(why="decode of one request without MTP", spec=dict(rows=1, **_GLM)),
        Cell(why="MTP-3 verify of one request", spec=dict(rows=4, **_GLM)),
        Cell(why="the kernel's 8-row bound", spec=dict(rows=8, **_GLM)),
    )

    note = """
    The kernel is compiled for 4 ranks, hc_mult 4 and hidden 4096 and holds at most 8 rows. The model
    sends it the steps the decode MoE kernel takes (at most 4 rows); larger steps take the library
    all-reduce and ``mHC.fused_hc``. It reads and rotates the workspace's
    Lamport state (receive buffers, flag, counter, clear size), so its calls and the library one-shot
    all-reduce's on that workspace must be ordered on one stream. The mixer products run on TF32
    tensor cores (the BF16 streams are exact in TF32) and the layer input's RMS comes from the 4 x 4
    Gram matrix of the new streams, so no second reduction over the hidden dimension is needed.
    """

    def __call__(
        self,
        partial: torch.Tensor,
        workspace: torch.Tensor,
        rank: int,
        residual: torch.Tensor,
        post: torch.Tensor,
        comb: torch.Tensor,
        fn: torch.Tensor,
        scale: torch.Tensor,
        base: torch.Tensor,
        norm_weight: torch.Tensor,
        sinkhorn_iters: int,
        norm_eps: float,
        hc_eps: float,
        sinkhorn_eps: float,
        post_mult: float,
        layer_input_eps: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        rows = partial.shape[0]
        residual_out = torch.empty_like(residual)
        post_out = torch.empty((rows, MULT), dtype=torch.float32, device=partial.device)
        comb_out = torch.empty((rows, MULT, MULT), dtype=torch.float32, device=partial.device)
        layer_input = torch.empty_like(partial)
        native_ops.mhc_ar_boundary(
            partial,
            workspace,
            rank,
            residual,
            post,
            comb,
            fn,
            scale,
            base,
            norm_weight,
            sinkhorn_iters,
            norm_eps,
            hc_eps,
            sinkhorn_eps,
            post_mult,
            layer_input_eps,
            residual_out,
            post_out,
            comb_out,
            layer_input,
            get_env_enable_pdl(),
        )
        return residual_out, post_out, comb_out, layer_input

    def reference(
        self,
        partial: Sequence[torch.Tensor],
        workspace: torch.Tensor,
        rank: int,
        residual: torch.Tensor,
        post: torch.Tensor,
        comb: torch.Tensor,
        fn: torch.Tensor,
        scale: torch.Tensor,
        base: torch.Tensor,
        norm_weight: torch.Tensor,
        sinkhorn_iters: int,
        norm_eps: float,
        hc_eps: float,
        sinkhorn_eps: float,
        post_mult: float,
        layer_input_eps: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """The same outputs in FP32 torch. A collective: ``partial`` is the group's partials in rank
        order, not this rank's one; ``workspace`` and ``rank`` are ignored.

        Rounded to BF16 where the op stores a tensor: the all-reduce result and the new streams (the
        next block reads the stored streams); everything else stays FP32, with FP32 matmuls. The
        collapsed layer input is normalized in FP32: the kernel takes its RMS from the Gram matrix of
        the new streams and never stores the row, so unlike the unfused path (the library all-reduce,
        then ``mHC.fused_hc``) it is not rounded to BF16 before the RMSNorm. That moves the layer
        input by under 2 ulp per element (0.8 ulp RMS) against the unfused path.
        """
        rows = residual.shape[0]
        with true_fp32_matmul():
            x = torch.stack([p.float() for p in partial]).sum(0).to(torch.bfloat16).float()
            streams = residual.float().reshape(rows, MULT, HIDDEN)
            post_prev = post.float().reshape(rows, MULT, 1)
            comb_prev = comb.float().reshape(rows, MULT, MULT)
            new = (x.unsqueeze(1) * post_prev + torch.bmm(comb_prev.mT, streams)).to(torch.bfloat16)
            flat = new.float().reshape(rows, MULT * HIDDEN)
            rstd = torch.rsqrt(flat.square().mean(-1, keepdim=True) + norm_eps)
            mixes = flat @ fn.float().T * rstd
        scales = torch.cat(
            [scale[0].expand(MULT), scale[1].expand(MULT), scale[2].expand(MULT * MULT)]
        ).float()
        mixes = mixes * scales + base.float()
        pre = torch.sigmoid(mixes[:, :MULT]) + hc_eps
        post_out = torch.sigmoid(mixes[:, MULT : 2 * MULT]) * post_mult
        comb_out = sinkhorn(
            mixes[:, 2 * MULT :].reshape(rows, MULT, MULT), sinkhorn_iters, sinkhorn_eps
        )
        collapsed = (new.float() * pre.unsqueeze(-1)).sum(1)
        layer_input = collapsed * torch.rsqrt(
            collapsed.square().mean(-1, keepdim=True) + layer_input_eps
        )
        return new, post_out, comb_out, (layer_input * norm_weight.float()).to(torch.bfloat16)

    def is_valid(
        self,
        partial: torch.Tensor,
        workspace: torch.Tensor,
        rank: int,
        residual: torch.Tensor,
        post: torch.Tensor,
        comb: torch.Tensor,
        fn: torch.Tensor,
        scale: torch.Tensor,
        base: torch.Tensor,
        norm_weight: torch.Tensor,
        sinkhorn_iters: int,
        norm_eps: float,
        hc_eps: float,
        sinkhorn_eps: float,
        post_mult: float,
        layer_input_eps: float,
    ) -> None:
        rows, hidden = partial.shape
        assert 1 <= rows <= MAX_ROWS, f"{rows} rows; at most {MAX_ROWS}"
        assert hidden == HIDDEN and residual.shape == (rows, MULT, HIDDEN), (
            "the kernel is compiled for hc_mult 4 and hidden 4096"
        )
        assert workspace.numel() == 3 * WORLD + 2 and 0 <= rank < WORLD, (
            f"the kernel is compiled for {WORLD} ranks (a workspace of {3 * WORLD + 2} pointers)"
        )
        for t in (partial, residual, norm_weight):
            assert t.dtype == torch.bfloat16 and t.is_contiguous()
        for t in (post, comb, fn, scale, base):
            assert t.dtype == torch.float32 and t.is_contiguous()
        assert post.numel() == rows * MULT and comb.numel() == rows * MULT * MULT
        assert fn.shape == ((2 + MULT) * MULT, MULT * HIDDEN)


mhc_ar_boundary = MhcArBoundary()
