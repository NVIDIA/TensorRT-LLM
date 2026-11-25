# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""One lagged Hyper-Connections boundary in a single launch: the previous sub-layer's post update, this
sub-layer's mixes and Sinkhorn, the collapse with the previous `pre`, and the RMSNorm."""

from __future__ import annotations

from typing import Optional

import torch

from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper
from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.residual.hc_post_mapping import (
    hc_post_mapping,
)
from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.residual.hc_pre_mapping import (
    hc_pre_mapping,
)

# The single-pass kernel takes up to this many rows with a post update (4096 without).
MAX_ROWS_WITH_POST = 1023
MAX_ROWS_NO_POST = 4096


class HcBoundary(OpWrapper):
    """`(x_prev | None, residual [n, hc, d], post_prev, comb_prev, pre_mix [n, hc], ...)` ->
    `(residual_cur [n, hc, d], pre [n, hc], post [n, hc], comb [n, hc, hc], h [n, d])`.

    The engine's CuTe DSL single-pass mHC boundary (`mhc_boundary`). With `x_prev` (the previous sub-layer's
    output, already summed over the tensor-parallel group) the previous sub-layer's post update runs first:

        residual_cur = hc_post_mapping(x_prev, residual, post_prev, comb_prev)

    and without it `residual_cur` is `residual` itself. Then, on `residual_cur`:

        pre, post, comb, h = hc_pre_mapping(residual_cur, pre_mix, ...)

    i.e. this sub-layer's own mixes, and `h = rmsnorm(collapse(residual_cur, pre_mix))` collapsed with the
    *previous* `pre`. `pre`, `post`, `comb` are fp32, `h` and `residual_cur` bf16.

    `boundary_weight` is `prepare_weight(hc_fn)`, a load-time rearrangement of `hc_fn` into the kernel's MMA
    fragments (TF32 products reduced in a fixed order); the caller keeps it next to `hc_fn`.
    """

    ARCHS = frozenset({Arch.SM_103})
    CELLS = (
        Cell(why="decode rows 6 with the previous post update", spec=dict(n=6, d=5120, post=True)),
        Cell(
            why="decode rows 18 with the previous post update", spec=dict(n=18, d=5120, post=True)
        ),
        Cell(
            why="decode rows 96 with the previous post update", spec=dict(n=96, d=5120, post=True)
        ),
        Cell(
            why="decode rows 1023: the largest batch with a post update",
            spec=dict(n=1023, d=5120, post=True),
        ),
        Cell(
            why="the first boundary of a step (no post update to absorb)",
            spec=dict(n=6, d=5120, post=False),
        ),
        Cell(
            why="a collapse-only boundary of a prefill tile", spec=dict(n=4096, d=5120, post=False)
        ),
    )
    note = """
    The mixer product is the kernel's own: the weight is rounded to TF32 and split into two bf16 halves, the
    activations are bf16, products accumulate in fp32 in a fixed order, so the result is deterministic and
    equal on every rank of a replicated stream. Sinkhorn and the RMS statistics are fp32 in-kernel. The
    kernels are JIT-compiled per (rows tile, post update) on first use, so a CUDA-graph capture must be
    preceded by an eager run of the same row count (`available` says whether it is). The op allocates only
    its outputs and syncs nothing on the host. The MXFP8 / NVFP4 quantized copies of `h` that the kernel can
    also write are not wrapped.
    """

    @staticmethod
    def prepare_weight(hc_fn: torch.Tensor) -> torch.Tensor:
        """Load-time: `hc_fn [24, 4 * d]` fp32 -> the kernel's weight fragments."""
        from tensorrt_llm._torch.kernels.blackwell.mhc import mhc_boundary

        return mhc_boundary.prepare_boundary_weight(hc_fn)

    @staticmethod
    def supported(hidden: int) -> bool:
        """Whether this device and hidden size run the kernel at all."""
        from tensorrt_llm._torch.kernels.blackwell.mhc import mhc_boundary

        return torch.cuda.get_device_capability() in (
            (10, 0),
            (10, 3),
            (10, 7),
        ) and mhc_boundary.MhcBoundaryKernel.supports(hidden)

    @staticmethod
    def available(
        n: int,
        hidden: int,
        with_post: bool,
        sinkhorn_iters: int,
        deferred: bool = False,
        mxfp8: bool = False,
    ) -> bool:
        """Whether a call over `n` rows can be made now: in range and, inside a CUDA-graph capture, compiled
        (`deferred`: the boundary without Sinkhorn iterations plus `mhc_sinkhorn`;
        `mxfp8`: the variant with the MXFP8 layer input)."""
        from tensorrt_llm._torch.kernels.blackwell.mhc import mhc_boundary

        if not HcBoundary.supported(hidden):
            return False
        if not 0 < n <= (MAX_ROWS_WITH_POST if with_post else MAX_ROWS_NO_POST):
            return False
        return not torch.cuda.is_current_stream_capturing() or mhc_boundary.compiled(
            hidden, with_post, sinkhorn_iters, n, deferred, mxfp8
        )

    def __call__(
        self,
        x_prev: Optional[torch.Tensor],
        residual: torch.Tensor,
        post_prev: Optional[torch.Tensor],
        comb_prev: Optional[torch.Tensor],
        pre_mix: torch.Tensor,
        hc_fn: torch.Tensor,
        boundary_weight: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
        norm_weight: torch.Tensor,
        hc_mult: int,
        sinkhorn_iters: int,
        rms_eps: float,
        norm_eps: float,
        hc_eps: float,
        post_mult: float,
        mxfp8: bool = False,
        sinkhorn_side: Optional[tuple] = None,
    ) -> tuple:
        """-> (residual, pre, post, comb, layer input); with `mxfp8` also the layer input as `mxfp8_quantize(h, True)`
        would give it (E4M3 rows, swizzled UE8M0 scales), written by the same launch (None past the single-pass range).

        `sinkhorn_side = (stream, fork event, done event)`: the boundary runs without Sinkhorn iterations (its `comb` is
        the row softmax) and the engine's `mhc_sinkhorn` finishes `comb` in place on `stream`, as the engine's
        multi-stream boundary does: everything else is ready on return, `comb` once `done` is waited on."""
        from tensorrt_llm._torch.kernels.blackwell.mhc import mhc_boundary

        n, _, d = residual.shape
        res, pre, post, comb, h, h_mx = mhc_boundary.mhc_boundary(
            x_prev,
            residual,
            post_prev,
            None if comb_prev is None else comb_prev.view(n, hc_mult * hc_mult),
            pre_mix,
            boundary_weight,
            hc_scale,
            hc_base,
            norm_weight,
            rms_eps,
            hc_eps,
            hc_eps,
            post_mult,
            norm_eps,
            0 if sinkhorn_side is not None else sinkhorn_iters,
            mxfp8,
        )
        if sinkhorn_side is not None:
            stream, fork, done = sinkhorn_side
            fork.record()
            with torch.cuda.stream(stream):
                fork.wait()
                mhc_boundary.mhc_sinkhorn(comb, hc_eps, sinkhorn_iters)
                done.record()
        out = (res, pre, post, comb.view(n, hc_mult, hc_mult), h)
        return out + (h_mx,) if mxfp8 else out

    def reference(
        self,
        x_prev: Optional[torch.Tensor],
        residual: torch.Tensor,
        post_prev: Optional[torch.Tensor],
        comb_prev: Optional[torch.Tensor],
        pre_mix: torch.Tensor,
        hc_fn: torch.Tensor,
        boundary_weight: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
        norm_weight: torch.Tensor,
        hc_mult: int,
        sinkhorn_iters: int,
        rms_eps: float,
        norm_eps: float,
        hc_eps: float,
        post_mult: float,
        mxfp8: bool = False,
        sinkhorn_side: Optional[tuple] = None,
    ) -> tuple:
        """The two entries it fuses, composed: `hc_post_mapping` then `hc_pre_mapping` (both float64 references)."""
        cur = (
            residual
            if x_prev is None
            else hc_post_mapping.reference(x_prev, residual, post_prev, comb_prev)
        )
        pre, post, comb, h = hc_pre_mapping.reference(
            cur, pre_mix, hc_fn, hc_scale, hc_base, norm_weight, hc_mult, sinkhorn_iters, rms_eps, norm_eps,
            hc_eps, post_mult,
        )  # fmt: skip
        if mxfp8:
            return cur, pre, post, comb, h, torch.ops.trtllm.mxfp8_quantize(h.contiguous(), True)
        return cur, pre, post, comb, h

    def is_valid(
        self,
        x_prev: Optional[torch.Tensor],
        residual: torch.Tensor,
        post_prev: Optional[torch.Tensor],
        comb_prev: Optional[torch.Tensor],
        pre_mix: torch.Tensor,
        hc_fn: torch.Tensor,
        boundary_weight: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
        norm_weight: torch.Tensor,
        hc_mult: int,
        sinkhorn_iters: int,
        rms_eps: float,
        norm_eps: float,
        hc_eps: float,
        post_mult: float,
        mxfp8: bool = False,
        sinkhorn_side: Optional[tuple] = None,
    ) -> None:
        assert residual.dim() == 3 and residual.shape[1] == hc_mult == 4, "residual is [n, 4, d]"
        n, _, d = residual.shape
        assert residual.dtype == torch.bfloat16 and residual.is_contiguous() and residual.is_cuda
        assert d % 512 == 0, (
            "the kernel splits the hidden size over a cluster of 16 slices of 32-wide chunks"
        )
        with_post = x_prev is not None
        assert (post_prev is None) == (comb_prev is None) == (not with_post), (
            "x_prev, post_prev, comb_prev go together"
        )
        assert 0 < n <= (MAX_ROWS_WITH_POST if with_post else MAX_ROWS_NO_POST), (
            "rows outside the single-pass range"
        )
        tensors = [residual, pre_mix, norm_weight]
        if with_post:
            assert x_prev.shape == (n, d) and x_prev.dtype == torch.bfloat16
            assert post_prev.shape == (n, hc_mult) and post_prev.dtype == torch.float32
            assert comb_prev.shape == (n, hc_mult, hc_mult) and comb_prev.dtype == torch.float32
            tensors += [x_prev, post_prev, comb_prev]
        assert all(t.is_contiguous() and t.data_ptr() % 16 == 0 for t in tensors), (
            "16-byte aligned, contiguous"
        )
        assert pre_mix.shape == (n, hc_mult) and pre_mix.dtype == torch.float32
        assert norm_weight.shape == (d,) and norm_weight.dtype == torch.bfloat16
        mix = (2 + hc_mult) * hc_mult
        assert hc_fn.shape == (mix, hc_mult * d) and hc_fn.dtype == torch.float32
        assert hc_scale.shape == (3,) and hc_scale.dtype == torch.float32
        assert hc_base.shape == (mix,) and hc_base.dtype == torch.float32
        assert boundary_weight.dtype == torch.int32, "boundary_weight is prepare_weight(hc_fn)"
        assert sinkhorn_iters >= 1 and post_mult == 2.0, (
            "the reference entries fix the post scale at 2"
        )


hc_boundary = HcBoundary()
