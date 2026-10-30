# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Owner-local PDL roles for the persistent dense GEMM.

In the decode attention chain, the persistent GEMMs that use this module (the
k_b and v_b batched GEMMs and the o_proj GEMM) are both PDL consumers and
producers.  A :class:`PersistentPdlRole` sets the compile-time PDL flags of
``dense_gemm_persistent.py``:

* ``wait_for_prior`` marks the launch as a programmatic dependent launch and
  waits for the prior grid before the first k-block;
* ``prewait_tma_operand="a"`` loads swapped operand A (the immutable weight)
  before that wait, and swapped operand B (the predecessor-owned activation)
  after it;
* ``signal_next`` signals the next grid after the TMA-store tail and TMEM
  cleanup.

The GEMM's TMA-store epilogue writes a ``Float8E4M3FN`` C (the k_b batched
GEMM writing the FP8 q-base, ``emit_fp8_q_base=True``) through an exact
FP32 -> BF16 -> E4M3 word-packed path; every other C dtype takes the library
epilogue.  Callers must opt into one explicit :class:`PersistentPdlRole`.
"""

from dataclasses import dataclass
from typing import Literal

import cutlass
import cutlass.cute as cute
import torch
from cutlass import testing
from cutlass.cute.runtime import make_fake_stream

from . import _swapped_dense_gemm_persistent as _swappeddensegemmpersiste

CompiledPersistentGemm = _swappeddensegemmpersiste.CompiledPersistentGemm
GemmSchedule = _swappeddensegemmpersiste.GemmSchedule
_DENSE_GEMM = _swappeddensegemmpersiste._DENSE_GEMM
_GlmPersistentDenseGemmKernel = _swappeddensegemmpersiste._GlmPersistentDenseGemmKernel
_make_runtime_arguments = _swappeddensegemmpersiste._make_runtime_arguments
_problem_shape_and_majors = _swappeddensegemmpersiste._problem_shape_and_majors


@dataclass(frozen=True)
class PersistentPdlRole:
    """Compile-time consumer/producer role for one persistent GEMM owner."""

    wait_for_prior: bool
    signal_next: bool
    prewait_tma_operand: Literal["none", "a"] = "none"

    def __post_init__(self) -> None:
        if not self.wait_for_prior and self.prewait_tma_operand != "none":
            raise ValueError("a producer-only role cannot preload a PDL operand")
        if not self.wait_for_prior and not self.signal_next:
            raise ValueError("a PDL role must wait for or signal another grid")


WAIT_SIGNAL_PRELOAD_WEIGHT_ROLE = PersistentPdlRole(True, True, "a")


def make_pdl_chain_kernel(
    schedule: GemmSchedule,
    role: PersistentPdlRole,
) -> _GlmPersistentDenseGemmKernel:
    """Create a persistent GEMM kernel carrying one immutable PDL role."""
    if schedule.orientation != "swapped":
        raise ValueError("PDL-chain persistent GEMM requires swapped orientation")
    return _GlmPersistentDenseGemmKernel(
        cutlass.Float32,
        schedule.use_2cta_instrs,
        schedule.mma_tiler_mn,
        schedule.cluster_shape_mn,
        schedule.use_tma_store,
        schedule.swizzle_size,
        schedule.raster_along,
        wait_for_prior_pdl=role.wait_for_prior,
        signal_next_pdl=role.signal_next,
        prewait_tma_operand=role.prewait_tma_operand,
    )


def compile_pdl_chain_persistent_gemm(
    input_tensor: torch.Tensor,
    weight: torch.Tensor,
    output: torch.Tensor,
    schedule: GemmSchedule,
    *,
    pdl_role: PersistentPdlRole,
    kernel_name_prefix: str | None = None,
) -> CompiledPersistentGemm:
    """Compile a persistent GEMM with an owner-local PDL role at the call's M.

    The compiled persistent object is row-agnostic (``mark_dynamic_layout`` on
    the row dimension; it was verified bit-identical to, and as fast as, a
    static-M compile at every decode M), so any positive row count
    compiles; the projection catalogs decide at which M a PDL role is used
    (``ProjectionAbi.pdl_enabled_m``).
    """
    active_m, input_features = input_tensor.shape
    output_features, weight_k = weight.shape
    if active_m <= 0:
        raise ValueError(f"PDL-chain persistent GEMM requires M > 0, got {active_m}")
    if weight_k != input_features or output.shape != (active_m, output_features):
        raise ValueError(
            "incompatible PDL-chain projection shapes: "
            f"input={input_tensor.shape}, weight={weight.shape}, output={output.shape}"
        )
    mnkl, majors = _problem_shape_and_majors(
        input_tensor,
        weight,
        schedule.orientation,
    )
    a, b, c = _make_runtime_arguments(
        input_tensor,
        weight,
        output,
        schedule.orientation,
    )
    cluster_size = schedule.cluster_shape_mn[0] * schedule.cluster_shape_mn[1]
    max_active_clusters = _DENSE_GEMM.utils.HardwareInfo().get_max_active_clusters(cluster_size)
    gemm = make_pdl_chain_kernel(schedule, pdl_role)
    if not gemm.can_implement(
        mnkl,
        a.element_type,
        b.element_type,
        c.element_type,
        *majors,
    ):
        raise testing.CantImplementError(
            "PDL-chain projection cannot implement "
            f"mnkl={mnkl}, majors={majors}, schedule={schedule}"
        )
    if kernel_name_prefix is not None:
        gemm.kernel.set_name_prefix(
            f"{kernel_name_prefix}_pdl_chain",
            remove_cutlass_symbol=True,
        )
    compiled_bmm = cute.compile(
        _DENSE_GEMM.bmm,
        gemm,
        a,
        b,
        c,
        max_active_clusters,
        make_fake_stream(),
        lambda value: value,
    )
    return CompiledPersistentGemm(
        input_tensor.shape,
        weight.shape,
        output.shape,
        schedule.orientation,
        compiled_bmm,
    )


def compile_pdl_wait_signal_persistent_gemm(
    input_tensor: torch.Tensor,
    weight: torch.Tensor,
    output: torch.Tensor,
    schedule: GemmSchedule,
    *,
    kernel_name_prefix: str | None = None,
) -> CompiledPersistentGemm:
    """Compile a PDL-chain GEMM with weight preload, activation wait, and signal."""
    return compile_pdl_chain_persistent_gemm(
        input_tensor,
        weight,
        output,
        schedule,
        pdl_role=WAIT_SIGNAL_PRELOAD_WEIGHT_ROLE,
        kernel_name_prefix=kernel_name_prefix,
    )
