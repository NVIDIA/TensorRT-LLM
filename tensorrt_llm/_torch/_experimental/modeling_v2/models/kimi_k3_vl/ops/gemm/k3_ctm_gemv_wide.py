# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi K3 wide decode GEMV x @ weight^T for M <= 64 tokens via the split-K CTM (CuTe DSL) kernel."""

import torch

import tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.kernels.k3_ctm_gemv.op  # noqa: F401  (registers torch.ops.trtllm.k3_ctm_gemv*)
from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import (
    Arch,
    Cell,
    OpWrapper,
    true_fp32_matmul,
)


class K3CtmGemvWide(OpWrapper):
    """Return `x @ weight.T` in bf16 (columns >= `sig_col0` as sigmoid) or fp32, in one call."""

    ARCHS = frozenset({Arch.SM_100, Arch.SM_103})
    CELLS: tuple[Cell, ...] = (
        Cell(
            why="KDA q/k/v/g/f_a/b, final output block has eight rows",
            spec=dict(
                name="kda_qkvg", n=3208, k=7168, sig_col0=-1, out_fp32=False, ms=range(1, 65)
            ),
        ),
        Cell(
            why="MLA fused W_a/W_g, sigmoid gate columns",
            spec=dict(
                name="mla_qkv_a_gate",
                n=2880,
                k=7168,
                sig_col0=2112,
                out_fp32=False,
                ms=range(1, 65),
            ),
        ),
        Cell(
            why="KDA/MLA o_proj, six k-tiles",
            spec=dict(name="o_proj", n=7168, k=768, sig_col0=-1, out_fp32=False, ms=range(1, 65)),
        ),
        Cell(
            why="MoE head latent-down slice and router rows, fp32 output",
            spec=dict(name="moe_head", n=280, k=7168, sig_col0=-1, out_fp32=True, ms=range(1, 65)),
        ),
        Cell(
            why="MoE tail, five k-tiles",
            spec=dict(name="moe_tail", n=7168, k=640, sig_col0=-1, out_fp32=False, ms=range(1, 65)),
        ),
        Cell(
            why="shared expert gate_up, six output blocks",
            spec=dict(
                name="shared_gate_up",
                n=768,
                k=7168,
                sig_col0=-1,
                out_fp32=False,
                ms=range(1, 65),
            ),
        ),
        Cell(
            why="dense MLP gate_up, thirty-three output blocks",
            spec=dict(
                name="dense_gate_up",
                n=4224,
                k=7168,
                sig_col0=-1,
                out_fp32=False,
                ms=range(1, 65),
            ),
        ),
        Cell(
            why="dense MLP down, final k-tile is a 64-column half",
            spec=dict(
                name="dense_down", n=7168, k=2112, sig_col0=-1, out_fp32=False, ms=range(1, 65)
            ),
        ),
        Cell(
            why="drafter qkv, four output blocks",
            spec=dict(
                name="drafter_qkv", n=512, k=7168, sig_col0=-1, out_fp32=False, ms=range(1, 65)
            ),
        ),
        Cell(
            why="drafter gate_up, fourteen output blocks",
            spec=dict(
                name="drafter_gate_up",
                n=1792,
                k=7168,
                sig_col0=-1,
                out_fp32=False,
                ms=range(1, 65),
            ),
        ),
        Cell(
            why="drafter o_proj, three k-tiles",
            spec=dict(
                name="drafter_o_proj",
                n=7168,
                k=384,
                sig_col0=-1,
                out_fp32=False,
                ms=range(1, 65),
            ),
        ),
    )

    note = """
    Each cell covers M=1..64, all token tiles, with x~N(0,1) and
    weight~0.02*N(0,1), rounded to bf16. Other admitted shapes are not
    certified. The inputs are read-only and the output is new and contiguous.
    Bias, other activations, residual, all-reduce and quantization are outside.

    The token tile is the smallest of 16,32,64 holding M. wide_config picks
    the largest split in (8,7,6,5,4,2) with enough k-tiles whose
    ceil(N/128)*split CTAs fit the device's SM count, then the deepest weight
    ring fitting shared memory beside an activation ring of up to 3 stages.
    At 148/152 SMs the listed cells use splits 5,6,2,8,2,8,4,2,8,8,2.
    Rank r accumulates 128-column k-tiles r,r+split,... with a zero-filled
    final half-tile. The first ceil(K/128)%split ranks take one more.
    Partials travel by 16-byte st.async stores; each rank reduces its tokens'
    partials in rank order in fp32. Bf16 output rounds once, then sigmoid
    columns apply fp32 sigmoid to that rounded value and round again.
    Fp32 output has no rounding or sigmoid. Long GEMV agrees at equal split;
    repeated calls and token rows are independent of M and PDL, but device
    SM count can change split and thus bits. cuBLAS sums in another order.
    The grid is ceil(N/128)*split CTAs of 256 threads, in clusters of split.

    No attention metadata, KV cache or module state is consumed. The current
    CUDA stream and device SM count are used. The result-neutral process
    compile cache is keyed by (N, K, split, ring, activation ring, token tile,
    out_fp32, PDL); M within a tile and sig_col0 are runtime. Warm every key
    eagerly before graph capture or the first call raises RuntimeError.
    cutlass and cuda.bindings must be importable.

    TRTLLM_ENABLE_PDL is read every call: unset or 1 enables PDL. Weight
    ring TMA loads with L2 evict-first and remaining k-tile prefetches precede
    griddepcontrol.wait, so weight must not be written by preceding
    in-flight work; x may be. Dependents are always released after initial
    weight traffic and must wait before reading y. Non-PDL consumers wait
    normally. Tests retain equal-split long-GEMV comparisons, warmed graph
    replay at MLA/MoE-head M=1,64, and the op's own loud refusals, including
    misalignment, sigmoid with fp32, and unsupported dimensions.
    """

    def __call__(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        sig_col0: int = -1,
        out_fp32: bool = False,
    ) -> torch.Tensor:
        return torch.ops.trtllm.k3_ctm_gemv_wide(x, weight, sig_col0=sig_col0, out_fp32=out_fp32)

    def reference(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        sig_col0: int = -1,
        out_fp32: bool = False,
    ) -> torch.Tensor:
        with true_fp32_matmul():
            out = x.float() @ weight.float().t()
        if out_fp32:
            return out
        out = out.bfloat16()
        if sig_col0 >= 0:
            out[:, sig_col0:] = out[:, sig_col0:].float().sigmoid().bfloat16()
        return out

    def is_valid(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        sig_col0: int = -1,
        out_fp32: bool = False,
    ) -> None:
        assert weight.device == x.device, "weight must be on x's device"


k3_ctm_gemv_wide = K3CtmGemvWide()
