# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-V4.1's routed experts over NVFP4 rows with the engine's CuTe-DSL grouped GEMMs, as stock's CUTEDSL MoE
backend runs them on Blackwell (`moe/fused_moe/cutedsl/grouped_gemm/nvfp4.py`, no expert all-to-all):

`moe_sort` (rows -> 128-row expert tiles) -> FC1 `cute_dsl_nvfp4_gather_grouped_gemm_act_fusion_zero_blackwell` (gather
+ grouped GEMM + clamped SwiGLU + NVFP4 requantize; its scheduler warp zeroes the output meanwhile, as stock's
all-to-all path does) -> FC2 `cute_dsl_nvfp4_grouped_gemm_finalize_inplace_blackwell`
(grouped GEMM + routing-weighted scatter-add into the zeroed output). One call takes any row count (no chunking).

Weights are stock's CuteDSL NVFP4 layout (`NVFP4CuteDslFusedMoEMethod`): FC1 is `[up | gate]` with rows regrouped as
`[up64, gate64, ...]` and its e4m3 block scales permuted alike then 128x4-swizzled; FC2 is the checkpoint's row order
with swizzled scales. `cute_dsl_moe_layout` builds them from the checkpoint form.
"""

from __future__ import annotations

from typing import Tuple

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 -- registers torch.ops.trtllm.*
from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper
from tensorrt_llm._torch.modeling_v2.catalog.moe.fp4_block_scale_moe_runner import (
    dequantize_nvfp4,
    quantize_nvfp4,
)

TILE = 128  # rows per expert tile of the grouped GEMMs (stock's tile_size)
SF_VEC = 16
_GROUP = 64  # FC1 rows per up / gate group (Blackwell kernels)


def _interleave_up_gate(x: torch.Tensor) -> torch.Tensor:
    """`[up | gate]` rows (dim 0, each half a multiple of 64) -> `[up64, gate64, up64, gate64, ...]`."""
    n = x.shape[0]
    assert n % (2 * _GROUP) == 0, n
    return (
        x.view(2, n // (2 * _GROUP), _GROUP, *x.shape[1:])
        .transpose(0, 1)
        .contiguous()
        .view(x.shape)
    )


def _swizzle(s: torch.Tensor) -> torch.Tensor:
    return torch.ops.trtllm.block_scale_interleave(s.contiguous()).reshape(s.shape)


def cute_dsl_moe_layout(up_w, up_s, gate_w, gate_s, down_w, down_s) -> Tuple[torch.Tensor, ...]:
    """One expert's kernel operands from its checkpoint form (a setup helper, not an entry).

    `up_w`/`gate_w` uint8 codes `[I, H/2]`, `up_s`/`gate_s` uint8 e4m3 scales `[I, H/16]`, `down_w` `[H, I/2]`,
    `down_s` `[H, I/16]` -> `(fc1_w [2I, H/2], fc1_s [2I, H/16], fc2_w [H, I/2], fc2_s [H, I/16])`, scales swizzled.
    """
    fc1_w = _interleave_up_gate(torch.cat([up_w, gate_w], 0))
    fc1_s = _swizzle(_interleave_up_gate(torch.cat([up_s, gate_s], 0)))
    return fc1_w, fc1_s, down_w.contiguous(), _swizzle(down_s)


def _moe_chain(
    x_fp4,
    x_sf,
    ids,
    weights,
    fc1_w,
    fc1_s,
    fc2_w,
    fc2_s,
    fc1_alpha,
    fc2_alpha,
    fc2_global,
    swiglu_limit: float,
    num_experts: int,
    local_offset: int,
    out,
    tile_size: int,
) -> torch.Tensor:
    """moe_sort -> FC1 (gather, SwiGLU, NVFP4 requantize, zeroing `out`) -> FC2 (weighted scatter-add into `out`)."""
    t, top_k = ids.shape
    e_local = fc1_w.shape[0]
    scales = weights if weights.dtype == torch.float32 else weights.float()
    (tile_to_expert, tile_to_mn_limit, _expanded_to_permuted, permuted_to_expanded, _padded,
     num_tiles) = torch.ops.trtllm.moe_sort(
        token_selected_experts=ids, token_final_scales=scales, num_experts=num_experts, top_k=top_k,
        local_expert_offset=local_offset, local_num_experts=e_local, tile_tokens_dim=tile_size,
    )  # fmt: skip
    if out is None:
        out = torch.empty(t, fc2_w.shape[1], dtype=torch.bfloat16, device=x_fp4.device)
    # FC1's scheduler warp zeroes the finalize output while FC1 runs (stock's `fc1_zero`), instead of a memset
    h, a_sf = torch.ops.trtllm.cute_dsl_nvfp4_gather_grouped_gemm_act_fusion_zero_blackwell(
        input=x_fp4.view(torch.float4_e2m1fn_x2), weight=fc1_w.view(torch.float4_e2m1fn_x2),
        input_scale=x_sf.view(torch.uint8), weight_scale=fc1_s.view(torch.uint8), alpha=fc1_alpha,
        tile_idx_to_group_idx=tile_to_expert, tile_idx_to_mn_limit=tile_to_mn_limit,
        permuted_idx_to_expanded_idx=permuted_to_expanded, num_non_exiting_tiles=num_tiles, global_sf=fc2_global,
        num_experts=num_experts, top_k=top_k, num_local_experts=e_local, local_expert_offset=local_offset,
        tile_size=tile_size, swiglu_limit_scalar=float(swiglu_limit), zero_out=out,
    )  # fmt: skip
    torch.ops.trtllm.cute_dsl_nvfp4_grouped_gemm_finalize_inplace_blackwell(
        input=h.view(torch.float4_e2m1fn_x2), weight=fc2_w.view(torch.float4_e2m1fn_x2),
        input_scale=a_sf.view(torch.uint8), weight_scale=fc2_s.view(torch.uint8), alpha=fc2_alpha, output=out,
        tile_idx_to_group_idx=tile_to_expert, tile_idx_to_mn_limit=tile_to_mn_limit,
        permuted_idx_to_expanded_idx=permuted_to_expanded, num_non_exiting_tiles=num_tiles,
        token_final_scales=scales, num_experts=num_experts, top_k=top_k, num_local_experts=e_local,
        local_expert_offset=local_offset, tile_size=tile_size, output_dtype=torch.bfloat16,
    )  # fmt: skip
    return out


class CuteDslMoeNvfp4(OpWrapper):
    """`out [T, H] bf16 = sum_k w[t, k] * expert_{ids[t, k]}(x[t])` over this rank's expert window.

    `x_fp4` uint8 `[T, H/2]` and `x_sf` uint8 `[T, H/16]` (row-major e4m3) are the rows quantized under the input
    global scale; `ids` int32 `[T, top_k]` global expert ids, `weights` `[T, top_k]` combine weights. Ids outside
    `[local_offset, local_offset + E_local)` contribute zero, so an expert-parallel split sums.
    `fc1_w`/`fc1_s`/`fc2_w`/`fc2_s` are `cute_dsl_moe_layout`'s per local expert (stacked on dim 0), `fc1_alpha` /
    `fc2_alpha` fp32 `[E_local]` the per-expert dequant scalars (input global scale x weight_scale_2), `fc2_global`
    fp32 `[1]` the FC1 output's NVFP4 global scale, `swiglu_limit` the checkpoint's limit in real units.

    `out`, when given, is a contiguous `[T, H]` bf16 buffer the result is written into (zeroed first, the finalize
    accumulates into it) and returned: a caller can place it in the expert all-to-all's combine region.
    `tile_size` is the rows per expert tile of both grouped GEMMs (stock's outer tactic: 128 runs 1-CTA tiles,
    256 the 2-CTA 256-row tiles on 2x1 clusters).
    """

    ARCHS = frozenset({Arch.SM_103})
    CELLS = (
        Cell(
            why="dep2 decode-sized step: 64 rows over a 192-expert window of 384, top-6, H 5120 / I 2304",
            spec=dict(
                tokens=64, experts=384, local=192, offset=192, top_k=6, hidden=5120, inter=2304
            ),
        ),
        Cell(
            why="dep2 context step: 8192 gathered rows, first window",
            spec=dict(
                tokens=8192, experts=384, local=192, offset=0, top_k=6, hidden=5120, inter=2304
            ),
        ),
        Cell(
            why="dep2 context step: 32768 gathered rows in one call (no chunking)",
            spec=dict(
                tokens=32768, experts=384, local=192, offset=192, top_k=6, hidden=5120, inter=2304
            ),
        ),
        Cell(
            why="dep2 context step on 256-row tiles (2-CTA, 2x1 clusters): 32768 rows",
            spec=dict(
                tokens=32768,
                experts=384,
                local=192,
                offset=0,
                top_k=6,
                hidden=5120,
                inter=2304,
                tile=256,
            ),
        ),
        Cell(
            why="256-row tiles at a small step (tiles mostly partial): 1024 rows",
            spec=dict(
                tokens=1024,
                experts=384,
                local=192,
                offset=192,
                top_k=6,
                hidden=5120,
                inter=2304,
                tile=256,
            ),
        ),
    )
    note = """
    FC1's SwiGLU output is requantized to NVFP4 under `fc2_global` before FC2, as the kernel does; the reference
    models that step (`quantize_nvfp4`). The gate is a relative RMS error against the fp32 reference.
    """

    def __call__(
        self,
        x_fp4,
        x_sf,
        ids,
        weights,
        fc1_w,
        fc1_s,
        fc2_w,
        fc2_s,
        fc1_alpha,
        fc2_alpha,
        fc2_global,
        swiglu_limit: float,
        num_experts: int,
        local_offset: int,
        out=None,
        tile_size: int = TILE,
    ) -> torch.Tensor:
        from tensorrt_llm._torch.autotuner import AutoTuner

        t, top_k = ids.shape
        if AutoTuner.get().is_tuning_mode and t:
            # Warm-up tuning (stock's `prime_inner_tactics`): the FC1 / FC2 tactics are tuned per tile size, so the
            # other tile size is tuned on these rows too, whichever one this step runs.
            other = 256 if tile_size == 128 else 128
            _moe_chain(
                x_fp4,
                x_sf,
                ids,
                weights,
                fc1_w,
                fc1_s,
                fc2_w,
                fc2_s,
                fc1_alpha,
                fc2_alpha,
                fc2_global,
                swiglu_limit,
                num_experts,
                local_offset,
                None,
                other,
            )
        return _moe_chain(
            x_fp4,
            x_sf,
            ids,
            weights,
            fc1_w,
            fc1_s,
            fc2_w,
            fc2_s,
            fc1_alpha,
            fc2_alpha,
            fc2_global,
            swiglu_limit,
            num_experts,
            local_offset,
            out,
            tile_size,
        )

    def reference(
        self,
        xq,
        ids,
        weights,
        up,
        gate,
        down,
        fc1_alpha,
        fc2_alpha,
        fc2_global: float,
        swiglu_limit: float,
        local_offset: int,
    ) -> torch.Tensor:
        """fp32 over dequantized operands. `xq` `[T, H]` is the input rows' NVFP4 value (codes x block scales, i.e.
        the rows times the input global scale), `up`/`gate`/`down` per local expert `(codes [N, K], e4m3 scales
        [N, K/16])` in checkpoint form. FC1 accumulators are dequantized by `fc1_alpha`, clamped (up to
        `[-limit, limit]`, gate to at most `limit`), SwiGLU'd, NVFP4-requantized under `fc2_global`, and FC2's
        accumulator is dequantized by `fc2_alpha` and weighted into its rows."""
        t, hdim = xq.shape[0], down[0][0].shape[0]
        out = torch.zeros(t, hdim, dtype=torch.float32, device=xq.device)
        for le in range(len(up)):
            tok, slot = (ids == local_offset + le).nonzero(as_tuple=True)
            if tok.numel() == 0:
                continue
            xe = xq[tok].float()
            a1 = float(fc1_alpha[le])
            u = ((xe @ dequantize_nvfp4(*up[le]).t()) * a1).clamp(
                min=-swiglu_limit, max=swiglu_limit
            )
            g = ((xe @ dequantize_nvfp4(*gate[le]).t()) * a1).clamp(max=swiglu_limit)
            act = quantize_nvfp4(u * g * torch.sigmoid(g), fc2_global)
            y = (act @ dequantize_nvfp4(*down[le]).t()) * float(fc2_alpha[le])
            out.index_add_(0, tok, y * weights[tok, slot].float().unsqueeze(1))
        return out

    def is_valid(
        self,
        x_fp4,
        x_sf,
        ids,
        weights,
        fc1_w,
        fc1_s,
        fc2_w,
        fc2_s,
        fc1_alpha,
        fc2_alpha,
        fc2_global,
        swiglu_limit: float,
        num_experts: int,
        local_offset: int,
        out=None,
        tile_size: int = TILE,
    ) -> None:
        assert tile_size in (128, 256), (
            "the Blackwell grouped GEMMs take 128-row (1-CTA) or 256-row (2-CTA) tiles"
        )
        t, h2 = x_fp4.shape
        e_local, n2, k2 = fc1_w.shape
        assert x_fp4.dtype == torch.uint8 and x_sf.shape == (t, 2 * h2 // SF_VEC) and k2 == h2
        assert ids.dtype == torch.int32 and ids.shape[0] == t and weights.shape == ids.shape
        assert n2 % (2 * _GROUP) == 0 and fc1_s.shape == (e_local, n2, 2 * h2 // SF_VEC)
        assert fc2_w.shape == (e_local, 2 * h2, n2 // 4) and fc2_s.shape == (
            e_local,
            2 * h2,
            n2 // 2 // SF_VEC,
        )
        assert fc1_alpha.shape == fc2_alpha.shape == (e_local,) and fc2_global.numel() == 1
        assert 0 <= local_offset and local_offset + e_local <= num_experts
        if out is not None:
            assert out.shape == (t, 2 * h2) and out.dtype == torch.bfloat16 and out.is_contiguous()


cute_dsl_moe_nvfp4 = CuteDslMoeNvfp4()
