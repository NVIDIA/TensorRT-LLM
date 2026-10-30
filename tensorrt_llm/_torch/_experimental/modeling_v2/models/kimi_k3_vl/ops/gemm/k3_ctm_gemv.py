# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi K3 decode GEMV x @ weight^T for M <= 8 bf16 tokens via the CTM (CuTe DSL) kernel."""

import torch

import tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.kernels.k3_ctm_gemv.op  # noqa: F401  (registers torch.ops.trtllm.k3_ctm_gemv*)
from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import (
    Arch,
    Cell,
    OpWrapper,
    true_fp32_matmul,
)


class K3CtmGemv(OpWrapper):
    """Return `bf16(x @ weight.T)`, fp32-accumulated, in one k3_ctm_gemv call."""

    ARCHS = frozenset({Arch.SM_100, Arch.SM_103})
    CELLS: tuple[Cell, ...] = (
        Cell(
            why="MLA o_proj, all K resident in one CTA",
            spec=dict(name="mla_o_proj_s1", n=7168, k=768, split=1, push=False, ms=range(1, 9)),
        ),
        Cell(
            why="MLA o_proj, two-CTA split-K reduction",
            spec=dict(name="mla_o_proj_s2", n=7168, k=768, split=2, push=False, ms=range(1, 9)),
        ),
        Cell(
            why="drafter o_proj, three resident k-tiles",
            spec=dict(name="drafter_o_proj", n=7168, k=384, split=1, push=True, ms=range(1, 9)),
        ),
        Cell(
            why="synthetic drafter o_proj, two resident k-tiles",
            spec=dict(
                name="drafter_o_proj_synthetic", n=7168, k=256, split=1, push=True, ms=range(1, 9)
            ),
        ),
    )

    note = """
    Each cell covers M=1..8 with trigger_early=True, x~N(0,1) and
    weight~0.03*N(0,1), rounded to bf16. Split 4 and other admitted shapes
    are not certified. Inputs are read-only; every output element is written
    in a new contiguous tensor. Bias, activation, residual, all-reduce and
    quantization remain with the caller.

    K is partitioned into 128-column tiles. Rank r accumulates tiles
    r, r+split, ...; the first K/128 % split ranks take one more tile.
    The owning rank adds fp32 partials in rank order and rounds once to bf16.
    Push selects DSMEM stores plus release-arrive or 16-byte st.async stores
    completing the owner's barrier by bytes; it has no effect at split 1.
    Rows past M enter the MMA as zeros and are not stored. Repeated calls and
    each token's bits are independent of M, push, trigger_early and PDL,
    but can differ with split and from cuBLAS's summation order.
    The grid is (N/128)*split CTAs of 256 threads, clustered when split>1.

    No attention metadata, KV cache or module state is consumed. The current
    CUDA stream on x's device is used. The result-neutral process compile
    cache is keyed by (N, K, split, trigger_early, push, PDL); M is runtime.
    A new key requires an eager CuTe DSL compile before graph capture, or
    raises RuntimeError. cutlass and cuda.bindings must be importable.

    TRTLLM_ENABLE_PDL is read every call: unset or 1 enables PDL. Weight TMA
    loads use L2 evict-first and precede griddepcontrol.wait; weight must not
    be written by preceding in-flight work. The x read waits. With
    trigger_early=True, dependents are released after weight loads are issued
    and must wait for this grid before reading y; otherwise they launch at
    grid completion. Non-PDL consumers, including torch, wait normally.
    The tests also cover split-2 transport/trigger equivalence, warmed graph
    replay at M=1,8 and the op's own loud unsupported-call refusals.
    """

    def __call__(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        trigger_early: bool = True,
        split: int = 1,
        push: bool = False,
    ) -> torch.Tensor:
        return torch.ops.trtllm.k3_ctm_gemv(
            x, weight, trigger_early=trigger_early, split=split, push=push
        )

    def reference(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        trigger_early: bool = True,
        split: int = 1,
        push: bool = False,
    ) -> torch.Tensor:
        with true_fp32_matmul():
            return (x.float() @ weight.float().t()).bfloat16()

    def is_valid(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        trigger_early: bool = True,
        split: int = 1,
        push: bool = False,
    ) -> None:
        assert x.data_ptr() % 16 == 0, "x must start at a 16-byte-aligned address"
        assert weight.data_ptr() % 16 == 0, "weight must start at a 16-byte-aligned address"
        assert weight.device == x.device, "weight must be on x's device"


k3_ctm_gemv = K3CtmGemv()
