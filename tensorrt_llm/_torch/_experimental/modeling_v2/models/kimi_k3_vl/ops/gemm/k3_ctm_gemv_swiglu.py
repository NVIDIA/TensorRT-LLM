# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SwiGLU down projection silu_and_mul(gu) @ weight^T for M <= 8 tokens via the CTM (CuTe DSL) kernel."""

import torch
import torch.nn.functional as F

import tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.kernels.k3_ctm_gemv.op  # noqa: F401  (registers torch.ops.trtllm.k3_ctm_gemv*)
from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import (
    Arch,
    Cell,
    OpWrapper,
    true_fp32_matmul,
)


class K3CtmGemvSwiglu(OpWrapper):
    """Return `bf16(silu_and_mul(gu) @ weight.T)`, gate columns first, in one k3_ctm_gemv_swiglu call."""

    ARCHS = frozenset({Arch.SM_100, Arch.SM_103})
    CELLS: tuple[Cell, ...] = (
        Cell(
            why="drafter down, seven k-tiles split unevenly as four and three",
            spec=dict(name="drafter_down", n=7168, k=896, split=2, push=True, ms=range(1, 9)),
        ),
        Cell(
            why="synthetic drafter down, six k-tiles split evenly",
            spec=dict(
                name="drafter_down_synthetic", n=7168, k=768, split=2, push=True, ms=range(1, 9)
            ),
        ),
    )

    note = """
    Each cell covers M=1..8 with trigger_early=True, gu~2*N(0,1) and
    weight~0.02*N(0,1), rounded to bf16. Splits 1 and 4 and other admitted
    shapes are not certified. The caller produces gate_up, with gate first.
    The activation (g*sigmoid(g))*u is evaluated in fp32 using IEEE division
    and full-precision exp and rounded once to bf16 inside the kernel,
    never materialized in memory. F.silu can differ in the last bf16 place
    because its fp32 operations differ. The GEMV then accumulates in fp32
    and rounds once. No bias, all-reduce or quantization is included.
    Inputs are read-only and every element of the new contiguous output
    is written.

    Rank r accumulates 128-column k-tiles r, r+split, ...; the first
    K/128%split ranks take one more. The owning rank adds fp32 partials
    in rank order. Push selects 16-byte st.async stores completing the
    owner's barrier by bytes instead of DSMEM stores plus release-arrive.
    Rows past M enter as zeros and are not stored. Repeated calls and token
    rows are bit-identical across M, push, trigger_early and PDL; split can
    change the last bf16 place. silu_and_mul plus cuBLAS can differ in both
    activation and accumulation order. The grid is (N/128)*split CTAs of
    256 threads, clustered when split>1.

    No attention metadata, KV cache or module state is consumed. The current
    CUDA stream is used. The result-neutral process compile cache is keyed
    by (N, K, split, trigger_early, push, PDL); M is runtime. Warm every key
    eagerly before graph capture or the first call raises RuntimeError.
    cutlass and cuda.bindings must be importable.

    TRTLLM_ENABLE_PDL is read every call: unset or 1 enables PDL. Whole
    weight TMA loads with L2 evict-first precede griddepcontrol.wait, so
    weight must not be written by preceding in-flight work; gu may be.
    trigger_early releases dependents after weight loads are issued, and
    they must wait before reading y. False defers them to grid completion;
    non-PDL consumers wait normally. Tests retain transport/trigger
    equivalence, warmed drafter-down graph replay at M=1,8, and loud
    unsupported-call refusals.
    """

    def __call__(
        self,
        gu: torch.Tensor,
        weight: torch.Tensor,
        trigger_early: bool = True,
        split: int = 2,
        push: bool = False,
    ) -> torch.Tensor:
        return torch.ops.trtllm.k3_ctm_gemv_swiglu(
            gu, weight, trigger_early=trigger_early, split=split, push=push
        )

    def reference(
        self,
        gu: torch.Tensor,
        weight: torch.Tensor,
        trigger_early: bool = True,
        split: int = 2,
        push: bool = False,
    ) -> torch.Tensor:
        """silu_and_mul's fp32 arithmetic and one bf16 rounding (gate columns first)."""
        k = gu.shape[1] // 2
        act = (F.silu(gu[:, :k].float()) * gu[:, k:].float()).bfloat16()
        with true_fp32_matmul():
            return (act.float() @ weight.float().t()).bfloat16()

    def is_valid(
        self,
        gu: torch.Tensor,
        weight: torch.Tensor,
        trigger_early: bool = True,
        split: int = 2,
        push: bool = False,
    ) -> None:
        assert gu.data_ptr() % 16 == 0, "gu must start at a 16-byte-aligned address"
        assert weight.data_ptr() % 16 == 0, "weight must start at a 16-byte-aligned address"
        assert weight.device == gu.device, "weight must be on gu's device"


k3_ctm_gemv_swiglu = K3CtmGemvSwiglu()
