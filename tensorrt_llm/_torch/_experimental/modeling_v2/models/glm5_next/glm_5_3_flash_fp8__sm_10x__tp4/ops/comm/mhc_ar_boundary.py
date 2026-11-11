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

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper
from tensorrt_llm._torch.flashinfer_utils import get_env_enable_pdl

from ... import native_ops

WORLD = 4
MAX_ROWS = 8
MULT = 4
HIDDEN = 4096


def _sinkhorn(logits: torch.Tensor, iters: int, eps: float) -> torch.Tensor:
    """Softmax over the rows, a column normalization, then ``iters - 1`` row / column pairs."""
    comb = logits.softmax(-1) + eps
    comb = comb / (comb.sum(-2, keepdim=True) + eps)
    for _ in range(iters - 1):
        comb = comb / (comb.sum(-1, keepdim=True) + eps)
        comb = comb / (comb.sum(-2, keepdim=True) + eps)
    return comb


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

    ARCHS = frozenset({Arch.SM_100, Arch.SM_103})

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
    ):
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
    ):
        """The same outputs in FP32 torch. A collective: ``partial`` is the group's partials in rank
        order, not this rank's one; ``workspace`` and ``rank`` are ignored.

        The all-reduce result and the new streams are rounded to BF16 where the op's outputs are
        (the next block reads the stored streams); everything else stays FP32 until the outputs.
        """
        rows = residual.shape[0]
        x = torch.stack([p.float() for p in partial]).sum(0).to(torch.bfloat16).float()
        streams = residual.float().reshape(rows, MULT, HIDDEN)
        post_prev = post.float().reshape(rows, MULT, 1)
        comb_prev = comb.float().reshape(rows, MULT, MULT)
        new = (x.unsqueeze(1) * post_prev + torch.bmm(comb_prev.mT, streams)).to(torch.bfloat16)
        flat = new.float().reshape(rows, MULT * HIDDEN)
        mixes = flat @ fn.float().T * torch.rsqrt(flat.square().mean(-1, keepdim=True) + norm_eps)
        scales = torch.cat(
            [scale[0].expand(MULT), scale[1].expand(MULT), scale[2].expand(MULT * MULT)]
        ).float()
        mixes = mixes * scales + base.float()
        pre = torch.sigmoid(mixes[:, :MULT]) + hc_eps
        post_out = torch.sigmoid(mixes[:, MULT : 2 * MULT]) * post_mult
        comb_out = _sinkhorn(
            mixes[:, 2 * MULT :].reshape(rows, MULT, MULT), sinkhorn_iters, sinkhorn_eps
        )
        collapsed = (new.float() * pre.unsqueeze(-1)).sum(1)
        layer_input = collapsed * torch.rsqrt(
            collapsed.square().mean(-1, keepdim=True) + layer_input_eps
        )
        return new, post_out, comb_out, (layer_input * norm_weight.float()).to(torch.bfloat16)

    def is_valid(
        self, partial, workspace, rank, residual, post, comb, fn, scale, base, norm_weight, **kwargs
    ):
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
