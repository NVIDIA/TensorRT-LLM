# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi K3 long-K decode GEMV x @ weight^T for M <= 8 tokens via the split-K CTM (CuTe DSL) kernel."""

import torch

import tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.kernels.k3_ctm_gemv.op  # noqa: F401  (registers torch.ops.trtllm.k3_ctm_gemv*)
from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import (
    Arch,
    Cell,
    OpWrapper,
    true_fp32_matmul,
)


class K3CtmGemvLong(OpWrapper):
    """Return `bf16(x @ weight.T)`, columns >= `sig_col0` (if >= 0) as `bf16(sigmoid(.))`, in one call."""

    ARCHS = frozenset({Arch.SM_100, Arch.SM_103})
    CELLS: tuple[Cell, ...] = (
        Cell(
            why="MLA fused W_a/W_g, sigmoid gate columns and split 6",
            spec=dict(
                name="mla_qkv_a_gate",
                n=2880,
                k=7168,
                sig_col0=2112,
                split=6,
                ring=6,
                push=True,
                ms=range(1, 9),
            ),
        ),
        Cell(
            why="dense MLP gate_up, split 4 requires a five-stage ring",
            spec=dict(
                name="dense_gate_up",
                n=4224,
                k=7168,
                sig_col0=-1,
                split=4,
                ring=5,
                push=False,
                ms=range(1, 9),
            ),
        ),
        Cell(
            why="dense MLP down, final k-tile is a 64-column half",
            spec=dict(
                name="dense_down",
                n=7168,
                k=2112,
                sig_col0=-1,
                split=2,
                ring=6,
                push=False,
                ms=range(1, 9),
            ),
        ),
        Cell(
            why="drafter qkv, four output blocks with split 8",
            spec=dict(
                name="drafter_qkv",
                n=512,
                k=7168,
                sig_col0=-1,
                split=8,
                ring=6,
                push=True,
                ms=range(1, 9),
            ),
        ),
        Cell(
            why="drafter gate_up, fourteen output blocks with split 8",
            spec=dict(
                name="drafter_gate_up",
                n=1792,
                k=7168,
                sig_col0=-1,
                split=8,
                ring=6,
                push=True,
                ms=range(1, 9),
            ),
        ),
        Cell(
            why="synthetic drafter gate_up, twelve output blocks",
            spec=dict(
                name="drafter_gate_up_synthetic",
                n=1536,
                k=7168,
                sig_col0=-1,
                split=8,
                ring=6,
                push=True,
                ms=range(1, 9),
            ),
        ),
        Cell(
            why="KDA q/k/v/g/f_a/b, final output block has eight rows",
            spec=dict(
                name="kda_qkvg",
                n=3208,
                k=7168,
                sig_col0=-1,
                split=5,
                ring=6,
                push=True,
                ms=range(1, 9),
            ),
        ),
    )

    note = """
    Each cell covers M=1..8 with trigger_early=True, x~N(0,1) and
    weight~0.02*N(0,1), rounded to bf16. Other admitted configurations are
    not certified. Inputs are read-only and the contiguous output is new.
    Bias, other activations, residual, all-reduce and quantization are outside.

    Rank r accumulates 128-column k-tiles r, r+split, ...; a final half-tile
    is zero-filled by TMA. The first ceil(K/128)%split ranks take one extra.
    Each 32-row quarter q is owned by rank q (q//2 at split 2); its fp32
    partials are added in rank order before one bf16 rounding. Sigmoid uses
    that rounded bf16 value in fp32 with IEEE division and full-precision
    exp, then rounds again. Ring and transport do not change the order:
    push uses 16-byte st.async stores, otherwise DSMEM plus release-arrive.
    Rows past M and N are not stored. Repeated calls and token rows are
    bit-identical across M, ring, push, trigger_early and PDL; split can
    change the last bf16 place. Wide GEMV agrees at the same split; cuBLAS
    may differ in summation order. The grid is ceil(N/128)*split CTAs of
    256 threads in clusters of split, without a one-wave limit.

    No attention metadata, KV cache or module state is consumed. The current
    CUDA stream is used. The result-neutral process compile cache is keyed
    by (N, K, split, ring, trigger_early, push, PDL); M and sig_col0 are
    runtime. Warm every key eagerly before graph capture or the first call
    raises RuntimeError. cutlass and cuda.bindings must be importable.

    TRTLLM_ENABLE_PDL is read every call: unset or 1 enables PDL. Filling
    the weight ring via TMA with L2 evict-first and prefetching remaining
    k-tiles precede griddepcontrol.wait. Weight must not be written by
    preceding in-flight work; x may be. trigger_early releases dependents
    after initial weight traffic is issued; they must wait before reading y.
    False defers them until grid completion; non-PDL consumers wait normally.
    Tests retain schedule-flag equivalence, warmed graph replay at MLA and
    dense-down M=1,8, and the op's own loud unsupported-call refusals.
    """

    def __call__(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        sig_col0: int = -1,
        split: int = 6,
        ring: int = 5,
        trigger_early: bool = True,
        push: bool = False,
    ) -> torch.Tensor:
        return torch.ops.trtllm.k3_ctm_gemv_long(
            x,
            weight,
            sig_col0=sig_col0,
            split=split,
            ring=ring,
            trigger_early=trigger_early,
            push=push,
        )

    def reference(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        sig_col0: int = -1,
        split: int = 6,
        ring: int = 5,
        trigger_early: bool = True,
        push: bool = False,
    ) -> torch.Tensor:
        with true_fp32_matmul():
            out = (x.float() @ weight.float().t()).bfloat16()
        if sig_col0 >= 0:
            out[:, sig_col0:] = out[:, sig_col0:].float().sigmoid().bfloat16()
        return out

    def is_valid(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        sig_col0: int = -1,
        split: int = 6,
        ring: int = 5,
        trigger_early: bool = True,
        push: bool = False,
    ) -> None:
        assert x.data_ptr() % 16 == 0, "x must start at a 16-byte-aligned address"
        assert weight.data_ptr() % 16 == 0, "weight must start at a 16-byte-aligned address"
        assert weight.device == x.device, "weight must be on x's device"


k3_ctm_gemv_long = K3CtmGemvLong()
