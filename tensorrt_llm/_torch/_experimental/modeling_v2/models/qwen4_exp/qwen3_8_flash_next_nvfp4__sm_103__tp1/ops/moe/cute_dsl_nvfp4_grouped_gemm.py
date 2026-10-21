# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CuteDSL NVFP4 FC2 of the routed experts: the plain grouped GEMM over the permuted
FC1 output, one bf16 row per permuted row, in one trtllm call."""

from __future__ import annotations

from typing import NamedTuple

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*
from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

from .cute_dsl_nvfp4_gather_grouped_gemm_act_fusion import _dequant, unswizzle


def _ckpt(
    tokens: int,
    tile_size: int,
    mma_n: int = 128,
    cluster_n: int = 1,
    dist: str = "router",
    gsf: str = "calibrated",
    zero_tokens: int = 0,
) -> dict:
    """The checkpoint's routed experts: 512 experts, top-10, hidden 2560, I 640, all
    local (tp1), bf16 out, 16-element e4m3 block scales.

    `mma_n` / `cluster_n` select the autotuner tactic: MMA tile `(tile_size,
    mma_n)`, cluster `(tile_size // 128, cluster_n)`; (128, 1) is also the
    fallback when the tuner has no entry. `dist` names how the test draws the
    expert ids: `router` = each row's top-10 of uniform scores. `gsf` is the FC1
    `global_sf` that quantized this op's input, against the SwiGLU output's
    magnitude: `calibrated` = the output maximum the test calibrates to maps to
    the e4m3 maximum, as the checkpoint's per-layer `1 / down_proj.input_scale`
    does; `subnormal` = 2^-12 of it, so input block scales fall into e4m3
    subnormals and zero. `zero_tokens` makes that many leading tokens' rows all
    zero, as FC1 returns them for an all-zero token.
    """
    return dict(
        tokens=tokens,
        tile_size=tile_size,
        mma_n=mma_n,
        cluster_n=cluster_n,
        dist=dist,
        gsf=gsf,
        zero_tokens=zero_tokens,
        num_experts=512,
        top_k=10,
        hidden=2560,
        intermediate=640,
    )


class Fc2Expected(NamedTuple):
    """What `reference` returns: the op's output, plus what the test's compare reads.

    `out` is laid out exactly as the op returns it; rows the op does not write
    hold zeros. `rows` marks the rows the op writes -- always the prefix of
    `num_non_exiting_tiles * tile_size` rows. `exact` is `alpha[e] * sum_k a_k
    w_k` on those rows, before any rounding; `magnitude` is `|alpha[e]| * sum_k
    |a_k w_k|`, the scale every fp32 rounding of the accumulation is relative to.
    """

    out: torch.Tensor
    rows: torch.Tensor
    exact: torch.Tensor
    magnitude: torch.Tensor


class CuteDslNvfp4GroupedGemm(OpWrapper):
    """`torch.ops.trtllm.cute_dsl_nvfp4_grouped_gemm_blackwell`: FC2 of the routed
    experts on the CuteDSL MoE path, one kernel launch.

    For every row `p` of the first `num_non_exiting_tiles` tiles (tile `i = p //
    tile_size`):

        e       = tile_idx_to_group_idx[i]                       the local expert
        acc[p]  = deq(input[p]) @ deq(weight[e]).T                [H], fp32
        out[p]  = bf16(alpha[e] * acc[p])                         fp32 product

    where `deq` is e2m1 times its e4m3 block scale. Every row of a live tile is
    computed, padding rows included; no row is gathered, weighted or summed.

    Arguments, as the CuteDSL MoE passes them (`run_moe_nvfp4_impl`,
    `_torch/moe/fused_moe/fused_moe_cute_dsl.py`, its non-fused-finalize branch):

    - `input` `[M, I/2]` float4_e2m1fn_x2 and `input_scale` `[M * I/16]` uint8:
      FC1's output exactly as the CuteDSL FC1
      (`cute_dsl_nvfp4_gather_grouped_gemm_act_fusion_blackwell`) returns it --
      one row per PERMUTED row, the block scales 128x4-swizzled over `[M,
      I/16]`. `M = permuted_idx_to_expanded_idx.numel()` of the sort.
    - `weight` `[E, H, I/2]` float4_e2m1fn_x2: `down_proj` per expert in the
      checkpoint's row order. No 32-row shuffle (that is trtllm-gen's layout).
    - `weight_scale` `[E, H, I/16]` uint8: its e4m3 block scales, 128x4-swizzled
      per expert (`block_scale_interleave`).
    - `alpha` `[E]` fp32: per expert, `max(down_proj.input_scale) * weight_scale_2[e]`
      -- the inverse of FC1's `global_sf` times the expert's weight scale.
    - `tile_idx_to_group_idx` `[M / tile_size]` int32 and `num_non_exiting_tiles`
      `[1]` int32: moe_sort's `tile_idx_to_expert_idx` and
      `num_non_exiting_tiles` from the sort FC1 ran on, with tile_tokens_dim ==
      `tile_size`.
    - `num_experts`, `top_k`, `num_local_experts`, `local_expert_offset`: read only
      by the autotuner's shape buckets and cache key; the kernel takes the
      expert count from `weight.size(0)` and the expert from the tile.
    - `output_dtype`: bf16, the only one the runner builds.

    Returns `[M, H]` bfloat16, one row per permuted row. The pair `(t, k)` is row
    `expanded_idx_to_permuted_idx[t, k]`; rows outside the live tiles are not
    written (see `note`).

    Fusion boundary: the grouped GEMM, the per-expert alpha and the bf16
    conversion happen inside. The rows arrive already permuted (FC1 gathered
    them); the routing-weighted top-k combine is the caller's
    (`moe_unpermute_inplace`). Routing weights are not read here.
    """

    ARCHS = frozenset({Arch.SM_103})

    CELLS: tuple[Cell, ...] = (
        Cell(
            why="tile 128: decode, 1 request without MTP; the fallback tactic",
            spec=_ckpt(1, 128),
        ),
        Cell(
            why="tile 128: MMA N 256 and cluster N 2 tactic, 8192-row max_num_tokens prefill",
            spec=_ckpt(8192, 128, mma_n=256, cluster_n=2),
        ),
        Cell(
            why="tile 128: tiny FC1 global_sf and 7 all-zero token rows; subnormal and zero "
            "input scales",
            spec=_ckpt(2048, 128, gsf="subnormal", zero_tokens=7),
        ),
    )

    note = """
    Path certified: the CuTe DSL kernel `Sm100BlockScaledContiguousGroupedGemmKernel`
    (`_torch/moe/kernels/blackwell/blockscaled_contiguous_grouped_gemm.py`),
    built by `Sm100BlockScaledContiguousGroupedGemmRunner`
    (`_torch/custom_ops/cute_dsl_custom_ops.py`), which JIT-compiles it with
    `cute.compile` on the first call per (tile, tactic) and keeps it for the
    process. The op is registered only when the CuTe DSL imports
    (`IS_CUTLASS_DSL_AVAILABLE`), and the runner accepts SM 100 and 103 only.

    Metadata consumed: none beyond the arguments. Stateless: allocates its output
    with torch.empty on every call, on the current stream, with no host
    synchronization; `num_non_exiting_tiles` is read on the device, so it
    captures into a CUDA graph together with moe_sort and FC1.

    Call sites: the engine's CuteDSL MoE (`run_moe_nvfp4_impl`, when the fused
    finalize is off) and the target's routed-expert step, both followed by
    `moe_unpermute_inplace`; eager for prefill and mixed batches (T up to
    max_num_tokens 8192) and captured for decode (T = requests x MTP verify
    width, up to 32 x 4 = 128). Tile 128 is the target's and the cells' tile;
    256 is certified through the per-pair invariance below.

    Numerics: each written element equals the reference's `out` bit for bit on
    every cell (and, at bring-up, on the real FC1 output at 8192 tokens and on the
    checkpoint's own down_proj experts). Operands whose block scales span a wider range can round a
    partial sum; the test's compare admits that at bf16 ties only.

    Tactics. The op asks the AutoTuner for an MMA tile of (tile_size, 128) or
    (tile_size, 256) and a cluster of (tile_size // 128, 1) or (tile_size //
    128, 2), and with no tuning entry falls back to ((tile_size, 128),
    (tile_size // 128, 1)); serving warmup tunes per row bucket and may pick any
    of the four. Every pair's row is bitwise identical across tile 128 / 256 x
    the four tactics x the fallback (the test checks it at 8192 rows), so cells
    driven at one tactic certify the others.

    Determinism. A (token, expert) pair's output row depends on that pair's input
    row and its expert alone: it is bitwise unchanged by moving it to another row
    of its expert's run (moe_sort orders them by atomics from T = 5), by repeating
    the call, and by cutting a batch to a prefix. This op does not make the MoE
    run-to-run nondeterministic; the fused-finalize variant
    (`cute_dsl_nvfp4_grouped_gemm_finalize_inplace_blackwell`) reduces across a
    token's experts in launch order and does.

    What is read and written: every row of the first num_non_exiting_tiles tiles
    is written; the pair rows hold their results and the padding rows of a live
    tile are computed from whatever the input holds there -- zero for FC1's
    output, which zeroes them. Rows at and past num_non_exiting_tiles * tile_size
    are never written: eager they hold whatever torch.empty returned, under a
    graph whatever the pool last held. Input rows past that point reach no written
    row. The combine must read only expanded_idx_to_permuted_idx's rows.

    Preconditions the caller owns:
    - `weight` / `weight_scale` in the layout the class docstring states, and
      `input_scale` swizzled. A wrong preparation of each runs without error,
      with finite output, and puts most of the pair rows' elements outside the
      test's gate: trtllm-gen's 32-row shuffle of the weight, input scales
      linear, one alpha for the whole layer.
    - `alpha` per expert, `max(down_proj.input_scale) * weight_scale_2[e]`, with
      the maximum the value FC1's `global_sf` inverts.
    - `tile_idx_to_group_idx` / `num_non_exiting_tiles` from the same sort as
      FC1's input, tile_tokens_dim == `tile_size`, and every expert index below
      `weight.size(0)`: nothing checks the values (a tile naming expert 512 of
      512 ran without error and came back all zeros).
    - Every tensor on the device. Nothing checks it: a host-resident `alpha` or
      tile table was read correctly on a coherent-memory Grace host, unmeasured
      where the GPU cannot dereference a host pointer.

    Refused by `is_valid`, each silently wrong unguarded: a non-contiguous input,
    input_scale, weight, weight_scale, alpha or tile_idx_to_group_idx (data_ptr()
    read as dense; finite, wrong rows).

    Loud (AssertionError from the runner before the launch): a tile_size other
    than the sort's, either way round; output_dtype float16; alpha bf16 or short
    of weight.size(0); weight_scale with fewer experts or typed float8_e4m3fn;
    input typed uint8; input_scale 2-D or short; scaling_vector_size 32;
    num_non_exiting_tiles of 2 elements; tile_idx_to_group_idx int64 or one tile
    short. tile_size 64: ValueError.

    Measured at bring-up and not re-driven by the test: the CUDA-graph capture
    with moe_sort and FC1, the determinism and read / write statements, the wrong
    preparations' distance from the gate, and the loud rejections. The test
    re-drives the cells, the tile / tactic invariance and the `input` contiguity
    refusal.

    Uncertified: output dtypes other than bf16 (the runner builds none); expert
    parallelism (the kernel reads neither local_expert_offset nor num_experts, so
    EP changes only the sort and the weight slice); other hidden / intermediate
    sizes; scaling_vector_size other than 16; other architectures.

    Hard limits (sm_103)
      input, input_scale, output, weight, weight_scale   past 2^31 elements /
                      bytes: exact (probed up to M = 53687296 rows and 20973
                      experts)
      tile index      int32 tables, within moe_sort's 2^31-pair limit; no index
                      passes through fp32
    """

    def __call__(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        input_scale: torch.Tensor,
        weight_scale: torch.Tensor,
        alpha: torch.Tensor,
        tile_idx_to_group_idx: torch.Tensor,
        num_non_exiting_tiles: torch.Tensor,
        num_experts: int,
        top_k: int,
        num_local_experts: int,
        local_expert_offset: int,
        tile_size: int,
        output_dtype: torch.dtype,
        scaling_vector_size: int = 16,
    ) -> torch.Tensor:
        return torch.ops.trtllm.cute_dsl_nvfp4_grouped_gemm_blackwell(
            input=input,
            weight=weight,
            input_scale=input_scale,
            weight_scale=weight_scale,
            alpha=alpha,
            tile_idx_to_group_idx=tile_idx_to_group_idx,
            num_non_exiting_tiles=num_non_exiting_tiles,
            num_experts=num_experts,
            top_k=top_k,
            num_local_experts=num_local_experts,
            local_expert_offset=local_expert_offset,
            tile_size=tile_size,
            output_dtype=output_dtype,
            scaling_vector_size=scaling_vector_size,
        )

    def reference(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        input_scale: torch.Tensor,
        weight_scale: torch.Tensor,
        alpha: torch.Tensor,
        tile_idx_to_group_idx: torch.Tensor,
        num_non_exiting_tiles: torch.Tensor,
        num_experts: int,
        top_k: int,
        num_local_experts: int,
        local_expert_offset: int,
        tile_size: int,
        output_dtype: torch.dtype,
        scaling_vector_size: int = 16,
    ) -> Fc2Expected:
        """Dequantize, GEMM in float64, scale by alpha in fp32, round once to bf16.

        Products of e2m1 and e4m3 values are exact, and float64 sums of 640 of
        them are exact to ~1e-16 relative, so `exact` carries no accumulation
        error of its own; `out` is that sum rounded to fp32, multiplied by the
        fp32 alpha, rounded to bf16 -- what an fp32-accumulating GEMM with this
        epilogue returns whenever its fp32 partial sums are exact. Rows whose
        input dequantizes to all zeros are zero without a GEMM.

        bf16 output only (anything else raises NotImplementedError).
        `num_experts`, `top_k`, `num_local_experts` and `local_expert_offset` do
        not enter the computation.
        """
        if output_dtype != torch.bfloat16:
            raise NotImplementedError("reference covers bf16 output only")
        vec = scaling_vector_size
        dev = input.device
        m = input.shape[0]
        hidden = weight.shape[1]
        k_blocks = input.shape[1] * 2 // vec
        n_tiles = int(num_non_exiting_tiles.reshape(-1)[0])
        span = n_tiles * tile_size

        rows = torch.zeros(m, dtype=torch.bool, device=dev)
        rows[:span] = True
        a = _dequant(
            input.view(torch.uint8)[:span],
            unswizzle(input_scale.view(torch.uint8), m, k_blocks)[:span],
            vec,
        )
        live = a.abs().amax(dim=1) > 0
        expert = tile_idx_to_group_idx[:n_tiles].long().repeat_interleave(tile_size)
        acc = torch.zeros(span, hidden, dtype=torch.float64, device=dev)
        mag = torch.zeros(span, hidden, dtype=torch.float64, device=dev)
        w_sf = weight_scale.view(torch.uint8).reshape(weight.shape[0], -1)
        for e in torch.unique(expert[live]).tolist():
            sel = torch.nonzero(live & (expert == e)).squeeze(1)
            w = _dequant(weight[e].view(torch.uint8), unswizzle(w_sf[e], hidden, k_blocks), vec)
            acc[sel] = a[sel] @ w.T
            mag[sel] = a[sel].abs() @ w.abs().T
        scale = alpha.float()[expert].unsqueeze(1)
        out = torch.zeros(m, hidden, dtype=torch.bfloat16, device=dev)
        out[:span] = (acc.float() * scale).to(torch.bfloat16)
        return Fc2Expected(
            out=out,
            rows=rows,
            exact=acc * scale.double(),
            magnitude=mag * scale.double().abs(),
        )

    def is_valid(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        input_scale: torch.Tensor,
        weight_scale: torch.Tensor,
        alpha: torch.Tensor,
        tile_idx_to_group_idx: torch.Tensor,
        num_non_exiting_tiles: torch.Tensor,
        num_experts: int,
        top_k: int,
        num_local_experts: int,
        local_expert_offset: int,
        tile_size: int,
        output_dtype: torch.dtype,
        scaling_vector_size: int = 16,
    ) -> None:
        # The runner hands data_ptr() to the kernel with dense row-major layouts
        # assumed; it checks dims, sizes and dtypes, never strides.
        for name, t in (
            ("input", input),
            ("input_scale", input_scale),
            ("weight", weight),
            ("weight_scale", weight_scale),
            ("alpha", alpha),
            ("tile_idx_to_group_idx", tile_idx_to_group_idx),
        ):
            assert t.is_contiguous(), (
                f"{name} must be contiguous; the kernel reads it from data_ptr() as a "
                "dense row-major buffer and silently reads the wrong elements"
            )


cute_dsl_nvfp4_grouped_gemm = CuteDslNvfp4GroupedGemm()
