# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The coefficient half of an attention-DP split boundary in one launch: the engine's CuTe DSL coefficient kernels
(`mHC.lagged_coefficients`, `kernels/blackwell/mhc/mhc_boundary.py`, `mhc_coefficient_tc.py`)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper
from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.residual.hc_coefficients import (
    hc_coefficients,
)

# Row ranges of the engine's kernels (`mHC.lagged_coefficients` on SM100 / SM103, a boundary that is not TP-replicated):
SMALL_MIN_ROWS = (
    17  # above `_PRE_MAPPING_DG_ONLY_MAX_M`: the small-batch kernel (exact fp32 products) ..
)
SMALL_MAX_ROWS = 64  # .. up to `LOWM_MAX_TOKENS`
MMA_MAX_ROWS = 1023  # the mma.sync kernel (`MMA_SYNC_MAX_TOKENS`) ..
TC_MIN_ROWS = 512  # .. below the tcgen05 kernel's first row count (`TC_MIN_ROWS`)
TC_MAX_ROWS = 4096  # the split boundary's last row count (`split_max_tokens(mega_mhc=False)`)


@dataclass
class HcCoefficientState:
    """Per boundary (one per (layer, sub-layer)): the kernels' arrival counters, which they leave zeroed, so two
    launches sharing one must not overlap; and the tcgen05 kernel's weight (`prepare_coefficient_weight(hc_fn)`, BF16
    `[48, 4 * d]`), built on the first eager call of 512 rows or more, as the engine builds it."""

    counters: torch.Tensor
    tc_weight: Optional[torch.Tensor] = None
    tc_counters: Optional[torch.Tensor] = None


def _kernels():
    from tensorrt_llm._torch.kernels.blackwell.mhc import mhc_boundary, mhc_coefficient_tc

    return mhc_boundary, mhc_coefficient_tc


class HcLaggedCoefficients(OpWrapper):
    """`(residual [n, hc, d], state, hc_fn, boundary_weight, ...)` -> `(pre [n, hc], post [n, hc], comb [n, hc, hc])`,
    fp32: the mixes of a boundary from its bf16 residual (the `hc_coefficients` contract) in one launch.

    The engine's choice for an attention-DP boundary of `n` rows (`mHC.lagged_coefficients`, not TP-replicated):

    * 17 .. 64 rows: the small-batch kernel (`mhc_small_coefficients`): fp32 mixer weight, exact fp32 products;
    * 65 .. 511 rows: the mma.sync kernel (`mhc_coefficients`) over `boundary_weight` (`hc_boundary.prepare_weight`):
      the weight rounded to TF32 and split into two bf16 halves, bf16 products in a fixed order, the same
      coefficients as the single-pass `hc_boundary` of that residual;
    * 512 .. 4096 rows: the tcgen05 kernel (`mhc_coefficients_tc`) over `state.tc_weight`, the same split weight;
    * any other row count (or 17 .. 64 rows under a capture before the small-batch kernel is compiled): the two-launch
      `hc_coefficients`.

    `prepare_state` builds a boundary's state at load time; `ready(n, ...)` says whether this launch runs one kernel
    (under a CUDA-graph capture only once the kernel is compiled, by the eager warm-up of the same shape).
    """

    ARCHS = frozenset({Arch.SM_103})
    CELLS = (
        Cell(why="dep32 s8 decode, 48 rows: the small-batch kernel", spec=dict(n=48, d=5120)),
        Cell(why="17 rows: the small-batch kernel's first row count", spec=dict(n=17, d=5120)),
        Cell(why="64 rows: the small-batch kernel's last row count", spec=dict(n=64, d=5120)),
        Cell(why="dep32 c7600 decode, 96 rows: the mma.sync kernel", spec=dict(n=96, d=5120)),
        Cell(why="dep32 batch 25, 150 rows: the mma.sync kernel", spec=dict(n=150, d=5120)),
        Cell(
            why="dep8 c7000 decode, 496 rows: the mma.sync kernel's largest dep8 rung",
            spec=dict(n=496, d=5120),
        ),
        Cell(why="512 rows: the tcgen05 kernel's first row count", spec=dict(n=512, d=5120)),
        Cell(why="dep8 batch 320, 1280 rows: the tcgen05 kernel", spec=dict(n=1280, d=5120)),
        Cell(why="16 rows: the two-launch path", spec=dict(n=16, d=5120)),
    )
    note = """
    `boundary_weight` and `state.tc_weight` are rounded copies of `hc_fn`: a weight reload must rebuild them (the
    targets build them once after loading). The kernels allocate their outputs and an fp32 partials scratch; the
    counters are per boundary because a launch leaves them zeroed only when it finishes.
    """

    @staticmethod
    def prepare_state(device: torch.device) -> HcCoefficientState:
        """Load-time: the counters of one boundary (the tcgen05 weight follows on first use)."""
        mhc_boundary, _ = _kernels()
        return HcCoefficientState(counters=mhc_boundary.coefficient_counters(device))

    @staticmethod
    def kernel(n: int, state: HcCoefficientState) -> Optional[str]:
        """`"small"`, `"mma"`, `"tc"` or None (the two-launch path) for `n` rows."""
        if SMALL_MIN_ROWS <= n <= SMALL_MAX_ROWS:
            return "small"
        if TC_MIN_ROWS <= n <= TC_MAX_ROWS and state.tc_weight is not None:
            return "tc"
        if SMALL_MAX_ROWS < n <= MMA_MAX_ROWS:
            return "mma"
        return None

    def ready(
        self,
        n: int,
        hidden: int,
        sinkhorn_iters: int,
        state: HcCoefficientState,
        hc_fn: torch.Tensor,
    ) -> bool:
        """Whether `__call__` of `n` rows runs one kernel now. Outside a capture it also builds the tcgen05 weight from
        512 rows on (`mHC.coefficient_kernel_ready`); inside one the kernel must already be compiled."""
        mhc_boundary, tc = _kernels()
        capturing = torch.cuda.is_current_stream_capturing()
        if TC_MIN_ROWS <= n <= TC_MAX_ROWS and state.tc_weight is None and not capturing:
            if tc.MhcCoefficientTcKernel.supports(hidden):
                state.tc_weight = mhc_boundary.prepare_coefficient_weight(hc_fn)
                state.tc_counters = tc.coefficient_counters(hc_fn.device)
        kind = self.kernel(n, state)
        if kind is None:
            return False
        if kind == "small" and not mhc_boundary.MhcSmallCoefficientKernel.supports(hidden):
            return False
        if not capturing:
            return True
        if kind == "small":
            return mhc_boundary.small_coefficients_compiled(hidden, sinkhorn_iters)
        if kind == "tc":
            return tc.compiled(hidden, sinkhorn_iters)
        return mhc_boundary.coefficients_compiled(hidden, sinkhorn_iters)

    def __call__(
        self,
        residual: torch.Tensor,
        state: HcCoefficientState,
        hc_fn: torch.Tensor,
        boundary_weight: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
        hc_mult: int,
        sinkhorn_iters: int,
        rms_eps: float,
        hc_eps: float,
        post_mult: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mhc_boundary, tc = _kernels()
        n, mult, d = residual.shape
        args = (hc_scale, hc_base, rms_eps, hc_eps, hc_eps, post_mult, sinkhorn_iters)
        kind = HcLaggedCoefficients.kernel(n, state)
        if kind == "small" and not (
            mhc_boundary.MhcSmallCoefficientKernel.supports(d)
            and (
                not torch.cuda.is_current_stream_capturing()
                or mhc_boundary.small_coefficients_compiled(d, sinkhorn_iters)
            )
        ):
            kind = None  # as the engine: a capture never compiles; the two-launch path stands in
        if kind == "small":
            pre, post, comb = mhc_boundary.mhc_small_coefficients(
                residual, hc_fn, *args, state.counters
            )
        elif kind == "tc":
            pre, post, comb = tc.mhc_coefficients_tc(
                residual, state.tc_weight, *args, state.tc_counters
            )
        elif kind == "mma":
            pre, post, comb = mhc_boundary.mhc_coefficients(
                residual, boundary_weight, *args, state.counters
            )
        else:
            return hc_coefficients(
                residual,
                hc_fn,
                hc_scale,
                hc_base,
                hc_mult,
                sinkhorn_iters,
                rms_eps,
                hc_eps,
                post_mult,
            )
        return pre.view(n, mult), post.view(n, mult), comb.view(n, mult, mult)

    def reference(
        self,
        residual: torch.Tensor,
        state: HcCoefficientState,
        hc_fn: torch.Tensor,
        boundary_weight: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
        hc_mult: int,
        sinkhorn_iters: int,
        rms_eps: float,
        hc_eps: float,
        post_mult: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return hc_coefficients.reference(
            residual, hc_fn, hc_scale, hc_base, hc_mult, sinkhorn_iters, rms_eps, hc_eps, post_mult
        )

    def is_valid(
        self,
        residual: torch.Tensor,
        state: HcCoefficientState,
        hc_fn: torch.Tensor,
        boundary_weight: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
        hc_mult: int,
        sinkhorn_iters: int,
        rms_eps: float,
        hc_eps: float,
        post_mult: float,
    ) -> None:
        hc_coefficients.is_valid(
            residual, hc_fn, hc_scale, hc_base, hc_mult, sinkhorn_iters, rms_eps, hc_eps, post_mult
        )
        n = residual.shape[0]
        assert n <= TC_MAX_ROWS, "the split boundary's rows"
        assert state.counters.dtype == torch.int32 and state.counters.device == residual.device


hc_lagged_coefficients = HcLaggedCoefficients()
