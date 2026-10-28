# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Low-M BF16 projection on the target's split-K CuTe DSL GEMM, into a caller-owned output."""

from __future__ import annotations

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import (
    Arch,
    Cell,
    OpWrapper,
    true_fp32_matmul,
)
from tensorrt_llm._torch.flashinfer_utils import get_env_enable_pdl

from ._low_m_splitk import device_limits, operands_overlap, run_splitk_dense, splitk_default_tactic


class GlmLowMGemm(OpWrapper):
    """`out = x @ weight.T`, BF16 in and out, FP32 accumulation rounded once; returns `out`.

    One launch of the low-M split-K kernel at the shape's `splitk_default_tactic`: each CTA of a
    cluster accumulates an exact K slice in FP32, the cluster reduces through DSMEM in a fixed
    order, and the owner CTA rounds and stores. The decode step's plain BF16 projections of at
    most 32 rows take it: the shared expert's gate_up and down, the LM head on this rank's
    vocabulary shard, and MTP draft step 0's eh_proj on this rank's input slice.
    """

    ARCHS = frozenset({Arch.SM_107})

    # GLM-5.2 at tp16: hidden 6144, shared-expert intermediate 2048 / 16 = 128 (gate_up fused,
    # 256 rows), vocabulary 154880 / 16 = 9680, eh_proj input 2 * 6144 / 16 = 768. A verify
    # step runs 8 rows (MTP K=7); the MTP draft steps run the shared expert and the LM head on
    # 1 row, draft step 0's eh_proj on 8.
    CELLS: tuple[Cell, ...] = (
        Cell(
            why="shared expert gate_up, verify step: 8 x 6144 -> 256, split-K 8 (32 CTAs)",
            spec=dict(m=8, k=6144, n=256),
        ),
        Cell(
            why="shared expert gate_up, MTP draft step: 1 row",
            spec=dict(m=1, k=6144, n=256),
        ),
        Cell(
            why="shared expert down, verify step: 8 x 128 -> 6144, one K tile, no split (96 CTAs)",
            spec=dict(m=8, k=128, n=6144),
        ),
        Cell(
            why="shared expert down, MTP draft step: 1 row",
            spec=dict(m=1, k=128, n=6144),
        ),
        Cell(
            why="LM head vocabulary shard, verify step: 8 x 6144 -> 9680, partial last tile",
            spec=dict(m=8, k=6144, n=9680),
        ),
        Cell(
            why="LM head vocabulary shard, MTP draft step: 1 row",
            spec=dict(m=1, k=6144, n=9680),
        ),
        Cell(
            why="eh_proj, MTP draft step 0: 8 x 768 (row-parallel slice) -> 6144, split-K 2",
            spec=dict(m=8, k=768, n=6144),
        ),
    )

    note = """
    The tactic is a function of (M, N, K) and the device's SM count and opt-in shared memory
    (`splitk_default_tactic`); the launch reserves the full shared-memory capacity, one CTA per
    SM. PDL follows `TRTLLM_ENABLE_PDL` (default on): the weight is streamed before the grid
    dependency wait, `x` is read after it, and the kernel releases its dependents once its weight
    loads are issued.

    Loud, so not guarded: M outside [1, 32], K not a multiple of 128, mismatched shapes, dtypes
    or devices, operands that are neither row- nor column-major, pointers not 32-byte aligned
    (ValueError before the launch). The first call of a (shape class, tactic) compiles the
    kernel; make it before CUDA-graph capture.
    """

    def __call__(self, x: torch.Tensor, weight: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
        """x [M, K] BF16 row-major; weight [N, K] BF16 (the stored weight); out [M, N] BF16."""
        m, k = x.shape
        n = weight.shape[0]
        sm_count, smem_capacity = device_limits(x.device.index)
        tactic = splitk_default_tactic(m, n, k, sm_count=sm_count, smem_capacity=smem_capacity)
        return run_splitk_dense(x, weight.t(), None, out, get_env_enable_pdl(), tactic)

    def reference(self, x: torch.Tensor, weight: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
        """The product the op writes into `out`: FP32-accumulated (TF32 off), rounded once.

        `out` is named to mirror `__call__` and not read.
        """
        with true_fp32_matmul():
            return (x.float() @ weight.float().t()).to(x.dtype)

    def is_valid(self, x: torch.Tensor, weight: torch.Tensor, out: torch.Tensor) -> None:
        # The owner CTA of a tile stores its rows while other clusters still stream x and the
        # weight; an `out` that shares memory with either is overwritten under their reads and
        # the result is silently wrong.
        for operand, name in ((x, "x"), (weight, "weight")):
            assert not operands_overlap(out, operand), f"out must not share memory with {name}"


low_m_gemm = GlmLowMGemm()
