# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Hyper-Connections pre-mapping of one sub-layer in two launches: mixes, Sinkhorn, collapse, RMSNorm."""

from __future__ import annotations

import torch

from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper
from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.residual.hc_mixes import hc_mixes
from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.residual.hc_pre import hc_pre
from tensorrt_llm._torch.modules.mhc.mhc_cuda import mhc_pre_mapping_fused


class HcPreMapping(OpWrapper):
    """`x [n, hc, d]`, the previous sub-layer's `pre_mix [n, hc]` -> `(pre, post, comb, h)`.

    The engine's lagged mHC boundary (`mhc_gemm_sqrsum_fma` or the DeepGEMM TF32 split-K GEMM, then
    `mhc_big_fuse`), equal to the composition of three plain-torch entries:

        pre, post, comb = hc_mixes(x, ...)                  # this sub-layer's own mixes
        h = rmsnorm(hc_pre(x, pre_mix), norm_weight)        # collapse with the *previous* `pre`

    `pre`, `post` `[n, hc]` and `comb [n, hc, hc]` are fp32 (`pre` is carried to the next sub-layer),
    `h [n, d]` is bf16 and already normalized.
    """

    ARCHS = frozenset({Arch.SM_103})
    CELLS = (
        Cell(
            why="decode rows 6: one request, a 5-token verify step plus the sampled token",
            spec=dict(n=6, hc=4, d=5120),
        ),
        Cell(why="decode rows 18: three requests of 6 rows", spec=dict(n=18, hc=4, d=5120)),
        Cell(why="decode rows 96", spec=dict(n=96, hc=4, d=5120)),
        Cell(
            why="decode rows 1280: the largest graph batch (320 requests x 4 rows)",
            spec=dict(n=1280, hc=4, d=5120),
        ),
        Cell(why="prefill tile: 8192 tokens of a chunk", spec=dict(n=8192, hc=4, d=5120)),
        Cell(why="an attention-DP rank with no rows", spec=dict(n=0, hc=4, d=5120)),
    )
    note = """
    The mixer GEMM runs on CUDA cores in fp32 (exact fp32 accumulation) or, where the engine's
    autotuner picks it, as a DeepGEMM TF32 split-K product; the Sinkhorn loop and the RMS
    statistic are fp32 in-kernel. The collapse is rounded to bf16 before the norm, like the
    stock `hc_pre` + `flashinfer_rmsnorm` pair. The op allocates its outputs and the GEMM
    accumulators with `torch.empty` (the graph pool under capture) and syncs nothing on the host;
    the autotuner's choice is looked up, not measured, outside its tuning context.
    """

    def __call__(
        self,
        x: torch.Tensor,
        pre_mix: torch.Tensor,
        hc_fn: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
        norm_weight: torch.Tensor,
        hc_mult: int,
        sinkhorn_iters: int,
        rms_eps: float,
        norm_eps: float,
        hc_eps: float,
        post_mult: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        n, _, d = x.shape
        if n == 0:  # an attention-DP rank without rows: the kernels do not take an empty grid
            return (
                pre_mix.new_empty(0, hc_mult),
                pre_mix.new_empty(0, hc_mult),
                pre_mix.new_empty(0, hc_mult, hc_mult),
                x.new_empty(0, d),
            )
        post, comb, h, pre = mhc_pre_mapping_fused(
            x.view(n, hc_mult * d),
            hc_fn,
            x,
            hc_mult,
            hc_scale,
            hc_base,
            d,
            rms_eps,
            hc_eps,
            hc_eps,
            post_mult,
            sinkhorn_iters,
            norm_weight=norm_weight,
            norm_eps=norm_eps,
            pre_mix_ext=pre_mix,
        )
        return pre, post, comb.view(n, hc_mult, hc_mult), h

    def reference(
        self,
        x: torch.Tensor,
        pre_mix: torch.Tensor,
        hc_fn: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
        norm_weight: torch.Tensor,
        hc_mult: int,
        sinkhorn_iters: int,
        rms_eps: float,
        norm_eps: float,
        hc_eps: float,
        post_mult: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """The plain-torch entries composed, in float64 (`post_mult` is the entries' fixed 2)."""
        assert post_mult == 2.0, "hc_mixes fixes the post scale at 2"
        if x.shape[0] == 0:
            e = pre_mix.new_empty
            return e(0, hc_mult), e(0, hc_mult), e(0, hc_mult, hc_mult), x.new_empty(0, x.shape[2])
        pre, post, comb = hc_mixes.reference(
            x, hc_fn, hc_scale, hc_base, hc_mult, sinkhorn_iters, rms_eps, hc_eps
        )
        li = hc_pre.reference(x, pre_mix).double()
        rs = torch.rsqrt(li.square().mean(-1, keepdim=True) + norm_eps)
        h = (li * rs * norm_weight.double()).to(x.dtype)
        return pre, post, comb, h

    def is_valid(
        self,
        x: torch.Tensor,
        pre_mix: torch.Tensor,
        hc_fn: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
        norm_weight: torch.Tensor,
        hc_mult: int,
        sinkhorn_iters: int,
        rms_eps: float,
        norm_eps: float,
        hc_eps: float,
        post_mult: float,
    ) -> None:
        assert x.dim() == 3 and x.shape[1] == hc_mult, "x is [n, hc_mult, d]"
        n, _, d = x.shape
        assert x.dtype == torch.bfloat16 and x.is_contiguous() and x.is_cuda, (
            "x: contiguous bf16 on the GPU"
        )
        assert d % 8 == 0 and x.data_ptr() % 16 == 0, "the kernels load 16-byte bf16 vectors"
        mix = (2 + hc_mult) * hc_mult
        assert (
            hc_fn.shape == (mix, hc_mult * d)
            and hc_fn.dtype == torch.float32
            and hc_fn.is_contiguous()
        )
        assert hc_scale.shape == (3,) and hc_scale.dtype == torch.float32
        assert hc_base.shape == (mix,) and hc_base.dtype == torch.float32
        assert (
            pre_mix.shape == (n, hc_mult)
            and pre_mix.dtype == torch.float32
            and pre_mix.is_contiguous()
        )
        assert (
            norm_weight.shape == (d,)
            and norm_weight.dtype == torch.bfloat16
            and norm_weight.is_contiguous()
        )
        assert hc_mult == 4, "the fused kernels are instantiated for 4 residual copies"
        assert sinkhorn_iters >= 1, "zero iterations would still run the first normalization"
        assert post_mult == 2.0, "the reference entries fix the post scale at 2"


hc_pre_mapping = HcPreMapping()
