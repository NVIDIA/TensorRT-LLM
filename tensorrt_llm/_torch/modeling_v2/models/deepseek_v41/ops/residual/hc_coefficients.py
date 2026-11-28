# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The mixes of a Hyper-Connections boundary alone (pre, post, Sinkhorn comb), for a side stream."""

from __future__ import annotations

import torch

from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper
from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.residual.hc_mixes import hc_mixes


class HcCoefficients(OpWrapper):
    """`residual [n, hc, d]` -> `(pre [n, hc], post [n, hc], comb [n, hc, hc])`, fp32.

    The coefficient half of a split boundary: the engine's mixer GEMM + sum of squares and `mhc_big_fuse`
    (sigmoid, softmax, Sinkhorn), with the collapse and the layer-input write pointed at a tiny stand-in so no
    residual is read twice. Equal to the plain-torch `hc_mixes` entry. Nothing here depends on the
    boundary's collapse, so the target runs it on a side stream while the sub-layer computes.
    """

    ARCHS = frozenset({Arch.SM_103})
    CELLS = (
        Cell(why="decode rows 6", spec=dict(n=6, d=5120)),
        Cell(why="decode rows 24", spec=dict(n=24, d=5120)),
        Cell(why="decode rows 64: the largest split boundary", spec=dict(n=64, d=5120)),
    )
    note = """
    The product is the engine's (exact fp32 on CUDA cores, or its DeepGEMM TF32 split-K tactic when the
    autotuner picked it), so the band is the `hc_pre_mapping` one. The stand-in buffers are the engine's
    process-wide scratch, grown only outside a CUDA-graph capture: a capture must be preceded by an eager call
    of at least as many rows.
    """

    def __call__(
        self,
        residual: torch.Tensor,
        hc_fn: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
        hc_mult: int,
        sinkhorn_iters: int,
        rms_eps: float,
        hc_eps: float,
        post_mult: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        from tensorrt_llm._torch.modules.mhc.hyper_connection import (
            _coefficient_scratch_pre,
            _coefficient_scratch_residual,
        )
        from tensorrt_llm._torch.modules.mhc.mhc_cuda import mhc_pre_mapping_fused

        n, _, d = residual.shape
        ones = _coefficient_scratch_pre(n, hc_mult, residual.device)
        standin = _coefficient_scratch_residual(n, hc_mult, residual.device)
        post, comb, _, pre = mhc_pre_mapping_fused(
            residual.view(n, hc_mult * d),
            hc_fn,
            residual,
            hc_mult,
            hc_scale,
            hc_base,
            d,
            rms_eps,
            hc_eps,
            hc_eps,
            post_mult,
            sinkhorn_iters,
            pre_mix_ext=ones,
            bigfuse_residual=standin,
        )
        return pre.view(n, hc_mult), post.view(n, hc_mult), comb.view(n, hc_mult, hc_mult)

    def reference(
        self,
        residual: torch.Tensor,
        hc_fn: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
        hc_mult: int,
        sinkhorn_iters: int,
        rms_eps: float,
        hc_eps: float,
        post_mult: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        assert post_mult == 2.0, "hc_mixes fixes the post scale at 2"
        return hc_mixes.reference(
            residual, hc_fn, hc_scale, hc_base, hc_mult, sinkhorn_iters, rms_eps, hc_eps
        )

    def is_valid(
        self,
        residual: torch.Tensor,
        hc_fn: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
        hc_mult: int,
        sinkhorn_iters: int,
        rms_eps: float,
        hc_eps: float,
        post_mult: float,
    ) -> None:
        assert residual.dim() == 3 and residual.shape[1] == hc_mult == 4, "residual is [n, 4, d]"
        n, _, d = residual.shape
        assert 0 < n, "the kernels do not take an empty grid"
        assert residual.dtype == torch.bfloat16 and residual.is_contiguous() and residual.is_cuda
        assert d % 8 == 0 and residual.data_ptr() % 16 == 0
        mix = (2 + hc_mult) * hc_mult
        assert (
            hc_fn.shape == (mix, hc_mult * d)
            and hc_fn.dtype == torch.float32
            and hc_fn.is_contiguous()
        )
        assert hc_scale.shape == (3,) and hc_scale.dtype == torch.float32
        assert hc_base.shape == (mix,) and hc_base.dtype == torch.float32
        assert sinkhorn_iters >= 1 and post_mult == 2.0


hc_coefficients = HcCoefficients()


FMA_MAX_ROWS = 8  # the engine's `FMA_COEFF_MAX_ROWS` (`mhc_coefficient_fma.MAX_ROWS`)


def hc_fma_fits(num_tokens: int, hidden: int, mult: int) -> bool:
    """Whether a boundary of `num_tokens` rows takes `hc_coefficients_fma` (a setup-free predicate, shapes only):
    up to 8 rows, 4 copies, a width the kernel splits into 256-column CTAs, an sm_100 / sm_103 device. The engine also
    requires the stream to be replicated over the TP group (attention TP), which is the caller's to know.
    """
    from tensorrt_llm._torch.kernels.blackwell.mhc.mhc_coefficient_fma import (
        MhcCoefficientFmaKernel,
    )

    dev = torch.cuda.is_available() and torch.cuda.get_device_capability() in ((10, 0), (10, 3))
    return (
        dev
        and 0 < num_tokens <= FMA_MAX_ROWS
        and mult == 4
        and MhcCoefficientFmaKernel.supports(hidden)
    )


def hc_fma_state(hc_fn: torch.Tensor, hidden: int, sinkhorn_iters: int) -> tuple:
    """`(weight, counter)` of `hc_coefficients_fma` for one boundary, a setup helper called once at load time (never
    under graph capture): `hc_fn` rounded to TF32 as DeepGEMM's `hc_prenorm` GEMM rounds it, a zeroed arrival ticket
    that only this boundary's launches share (they must not overlap), and every row count's kernel compiled now.
    """
    from tensorrt_llm._torch.kernels.blackwell.mhc import mhc_coefficient_fma as fma

    fma.compile_all(hidden, sinkhorn_iters)
    return fma.prepare_fma_weight(hc_fn), fma.fma_counter(hc_fn.device)


class HcCoefficientsFma(OpWrapper):
    """`hc_coefficients` of up to 8 rows in one CUDA-core launch (the engine's `MhcCoefficientFmaKernel`, the
    built-in model's coefficients of small TP-replicated boundaries, #22360): a CTA per 256 residual columns forms the
    24-wide mixer product and the sum of squares with FP32 FMAs, the last CTA to arrive sums the partials in CTA
    order and runs the RMS scale, sigmoids and Sinkhorn. Runs at the direct decode's 58% shared-memory carveout, so
    its CTAs share SMs with the attention it overlaps. With `release_after_wait` it lets its dependents launch from
    its dependency wait on (a consumer that reads the coefficients only after its own wait, a split boundary).

    `weight` / `counter` are `hc_fma_state`'s; the other arguments are `hc_coefficients`'s.
    """

    ARCHS = frozenset({Arch.SM_103})
    CELLS = (
        Cell(why="tep8 decode rows 6", spec=dict(n=6, d=5120)),
        Cell(why="one row", spec=dict(n=1, d=5120)),
        Cell(why="eight rows: the kernel's edge", spec=dict(n=8, d=5120)),
        Cell(
            why="tep8 decode rows 6, releasing dependents after its wait",
            spec=dict(n=6, d=5120, release=True),
        ),
    )
    note = """
    FP32 products of the TF32-rounded weight with the BF16 residual (as DeepGEMM's TF32 GEMM forms them) in another
    summation order: within a few 1e-5 relative of `hc_coefficients` (the engine measured 8.2e-6 / 6.7e-6 / 2.6e-5 for
    pre / post / comb), so the `hc_coefficients` band against the float64 reference holds.
    """

    def __call__(
        self,
        residual: torch.Tensor,
        weight: torch.Tensor,
        counter: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
        hc_mult: int,
        sinkhorn_iters: int,
        rms_eps: float,
        hc_eps: float,
        post_mult: float,
        release_after_wait: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        from tensorrt_llm._torch.kernels.blackwell.mhc.mhc_coefficient_fma import (
            mhc_coefficients_fma,
        )

        n, _, d = residual.shape
        pre, post, comb = mhc_coefficients_fma(
            residual.view(n, hc_mult * d), weight, hc_scale, hc_base, rms_eps, hc_eps, hc_eps, post_mult,
            sinkhorn_iters, counter, release_after_wait=release_after_wait,
        )  # fmt: skip
        return pre.view(n, hc_mult), post.view(n, hc_mult), comb.view(n, hc_mult, hc_mult)

    def reference(
        self,
        residual: torch.Tensor,
        weight: torch.Tensor,
        counter: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
        hc_mult: int,
        sinkhorn_iters: int,
        rms_eps: float,
        hc_eps: float,
        post_mult: float,
        release_after_wait: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """`hc_mixes`' float64 reference on the (TF32-rounded) weight."""
        assert post_mult == 2.0, "hc_mixes fixes the post scale at 2"
        return hc_mixes.reference(
            residual, weight, hc_scale, hc_base, hc_mult, sinkhorn_iters, rms_eps, hc_eps
        )

    def is_valid(
        self,
        residual: torch.Tensor,
        weight: torch.Tensor,
        counter: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
        hc_mult: int,
        sinkhorn_iters: int,
        rms_eps: float,
        hc_eps: float,
        post_mult: float,
        release_after_wait: bool = False,
    ) -> None:
        hc_coefficients.is_valid(
            residual, weight, hc_scale, hc_base, hc_mult, sinkhorn_iters, rms_eps, hc_eps, post_mult
        )
        n, _, d = residual.shape
        assert hc_fma_fits(n, d, hc_mult), "1..8 rows, 4 copies, 256-column CTAs"
        assert counter.dtype == torch.int32 and counter.numel() == 1, "hc_fma_state's ticket"


hc_coefficients_fma = HcCoefficientsFma()
