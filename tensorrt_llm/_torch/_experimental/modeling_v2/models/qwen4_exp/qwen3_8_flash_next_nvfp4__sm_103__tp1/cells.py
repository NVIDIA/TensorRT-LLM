# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The cells of the catalog entries the Qwen3.8-Flash-Next NVFP4 sm_103 tp1 target calls.

Each entry's own cells are the op's: tile edges, row counts, the dtypes its
kernel takes. These are the shapes this target calls it with. The entry's
test drives both through `certified_cells`; nothing in the forward reads
them, so a change here reruns the entry tests and nothing of the target's.
"""

from __future__ import annotations

import torch

from ....catalog._op import Cell, OpWrapper
from ....catalog.attention.fused_qk_norm_rope import FusedQkNormRope
from ....catalog.attention.thop_attention import (
    _QWEN4_DENSE,
    _QWEN4_VISION,
    QM_FP8_KV_CACHE,
    QM_NVFP4_KV_CACHE,
    ThopAttention,
)
from ....catalog.moe.fp4_block_scale_moe_runner import Fp4BlockScaleMoeRunner
from ....catalog.quantization.fp4_quantize import Fp4Quantize

# qwen4_exp (Qwen3.8-Flash-Next) runs Gemma QK norm and an unscaled rope
# on the first 64 dims (32 pairs), base 1e7, at two geometries. Its
# mrope_section is [11, 11, 10], passed as sections 11 / 11. Positions are
# drawn uniformly below `max_position` (a test axis, not an op argument):
# the kernel's fast sincos loses phase as `position * freq` grows, so the
# position range is part of what a cell certifies.
_QWEN_ROPE = dict(
    rotary_dim=64,
    base=1e7,
    is_neox=True,
    is_qk_norm=True,
    use_gemma=True,
    max_position=32768,
)
_QWEN_MROPE = dict(use_mrope=True, mrope_section1=11, mrope_section2=11, **_QWEN_ROPE)
# Main attention (trunk and MTP): 24 query heads, 2 kv, head_dim 256.
_QWEN_MAIN = dict(num_heads_q=24, num_heads_k=2, num_heads_v=2, head_dim=256, **_QWEN_MROPE)
# The QSA index plane's query rope: 4 index heads plus the one k and one v
# slot of its scratch layout, over index_head_dim 128 (== 2 * rotary_dim,
# so the partial factor is not reapplied).
_QWEN_INDEX_HEADS = dict(num_heads_q=4, num_heads_k=1, num_heads_v=1, head_dim=128)
_QWEN_INDEX_MROPE = dict(**_QWEN_INDEX_HEADS, **_QWEN_MROPE)
_QWEN_INDEX_PLAIN = dict(**_QWEN_INDEX_HEADS, **_QWEN_ROPE)

# qwen4_exp's (Qwen3.8-Flash-Next) routed geometry: many small experts, a
# top_k past 8, an intermediate width of 640, all 512 experts local. Its
# router renormalizes the top-10 softmax weights to sum to 1 before the
# call (`routing="renormalized"`, a test axis rather than an op argument;
# the R1 cells draw uniform [0, 1) weights). Nothing is padded:
# H % 256, 2I % 128, H/16 % 4 and I/16 % 4 are all zero.
_QWEN4_EXP = dict(
    hidden=2560,
    intermediate=640,
    num_experts=512,
    top_k=10,
    local_expert_offset=0,
    local_num_experts=512,
    routing="renormalized",
)

#: The qwen4_exp full-attention call over an fp8-e4m3 pool at the production kv
#: scale 1.0 (fp32 [1] scale tensors, quant_mode 128): what the trunk layers and
#: the MTP draft layer pass when the engine serves an fp8 KV cache. Same chosen
#: pool sizes as `_QWEN4_DENSE`.
_QWEN4_FP8 = dict(_QWEN4_DENSE, quant_mode=QM_FP8_KV_CACHE, kv_scale=1.0)

#: The same call over an NVFP4 pool: e2m1 data and a second pool of e4m3 block
#: scales, fp32 [3] unit kv scales, `[1, 2, 2]` pool pointers, bf16 output --
#: what the trunk layers and the MTP draft layer pass when the engine serves an
#: NVFP4 KV cache. Same chosen pool sizes as `_QWEN4_DENSE`.
_QWEN4_NVFP4 = dict(_QWEN4_DENSE, quant_mode=QM_NVFP4_KV_CACHE, kv_scale=1.0)

#: `_QWEN4_DENSE` and `_QWEN4_FP8` for a verify step of up to 32 requests, the captured-batch
#: ceiling: one block-offset row per request and pages for 32 histories (the largest cell
#: needs 133 of the 144).
_QWEN4_DENSE_G32 = dict(_QWEN4_DENSE, max_num_requests=32, num_blocks=144)
_QWEN4_FP8_G32 = dict(_QWEN4_FP8, max_num_requests=32, num_blocks=144)


def _spread(batch: int, longest: int) -> tuple[int, ...]:
    """`batch` histories rising evenly from 1 to `longest` tokens: one-page, page-boundary
    and multi-page sequences in one verify step."""
    return tuple(1 + (longest - 1) * i // (batch - 1) for i in range(batch))


CELLS: dict[type[OpWrapper], tuple[Cell, ...]] = {
    FusedQkNormRope: (
        Cell(
            why="qwen4_exp main rope, decode: one token, Gemma norm and interleaved mRoPE "
            "with three distinct position rows",
            spec=dict(tokens=1, **_QWEN_MAIN),
        ),
        Cell(
            why="qwen4_exp main rope, 19 tokens: the size its wiring and saturation tests run at",
            spec=dict(tokens=19, **_QWEN_MAIN),
        ),
        Cell(
            why="qwen4_exp main rope, 8192-token prefill: the Gemma one-ulp tail, under "
            "GEMMA_PREFILL_ATOL",
            spec=dict(tokens=8192, gemma_prefill_band=True, **_QWEN_MAIN),
        ),
        # The main rope run in place on qwen4_exp's fused input projection, whose
        # row is [q | k | v | gate | index | pad] = 55 heads of 256: everything
        # after k is passed as v heads, which only sets the row stride.
        Cell(
            why="qwen4_exp fused-row main rope, 1 token: 29 v heads only set the row stride",
            spec=dict(tokens=1, **{**_QWEN_MAIN, "num_heads_v": 29}),
        ),
        Cell(
            why="qwen4_exp fused-row main rope, 128 tokens: decode at max concurrency",
            spec=dict(tokens=128, **{**_QWEN_MAIN, "num_heads_v": 29}),
        ),
        Cell(
            why="qwen4_exp fused-row main rope, 8192 tokens: prefill, under GEMMA_PREFILL_ATOL",
            spec=dict(tokens=8192, gemma_prefill_band=True, **{**_QWEN_MAIN, "num_heads_v": 29}),
        ),
        Cell(
            why="qwen4_exp index rope (mRoPE), 1 token: one k and one v head, the "
            "narrowest layout; 5 warps of an 8-warp block",
            spec=dict(tokens=1, **_QWEN_INDEX_MROPE),
        ),
        Cell(
            why="qwen4_exp index rope (mRoPE), 2 tokens: the second token's heads "
            "straddle a block boundary",
            spec=dict(tokens=2, **_QWEN_INDEX_MROPE),
        ),
        Cell(
            why="qwen4_exp index rope (mRoPE), 5 tokens: the last block runs one warp",
            spec=dict(tokens=5, **_QWEN_INDEX_MROPE),
        ),
        Cell(
            why="qwen4_exp index rope (mRoPE), 17 tokens: one past a power of two",
            spec=dict(tokens=17, **_QWEN_INDEX_MROPE),
        ),
        Cell(
            why="qwen4_exp index rope (mRoPE), 19 tokens: the size its wiring and "
            "saturation tests run at",
            spec=dict(tokens=19, **_QWEN_INDEX_MROPE),
        ),
        Cell(
            why="qwen4_exp index rope (mRoPE), 33 tokens: one past a power of two",
            spec=dict(tokens=33, **_QWEN_INDEX_MROPE),
        ),
        Cell(
            why="qwen4_exp index rope (mRoPE), 65 tokens: one past a power of two",
            spec=dict(tokens=65, **_QWEN_INDEX_MROPE),
        ),
        Cell(
            why="qwen4_exp index rope (mRoPE), 8192-token prefill: the index site sees the "
            "whole packed batch; under GEMMA_PREFILL_ATOL",
            spec=dict(tokens=8192, gemma_prefill_band=True, **_QWEN_INDEX_MROPE),
        ),
        Cell(
            why="qwen4_exp index rope (plain), decode: the form the index plane runs "
            "today, row 0 alone as a flat [T] position_ids",
            spec=dict(tokens=1, **_QWEN_INDEX_PLAIN),
        ),
        Cell(
            why="qwen4_exp index rope (plain), 19 tokens: an odd count across blocks",
            spec=dict(tokens=19, **_QWEN_INDEX_PLAIN),
        ),
        Cell(
            why="qwen4_exp index rope (plain), 8192-token prefill: under GEMMA_PREFILL_ATOL",
            spec=dict(tokens=8192, gemma_prefill_band=True, **_QWEN_INDEX_PLAIN),
        ),
    ),
    ThopAttention: (
        Cell(
            why="qwen4_exp prefill: GQA 24q/2kv d256, bf16 pool, paged context; "
            "a fresh 70-row sequence beside 33 rows over a 48-token cached prefix",
            spec=dict(
                _QWEN4_DENSE,
                cached=(0, 48),
                new=(70, 33),
                num_contexts=2,
                predicted_tokens_per_seq=1,
            ),
        ),
        Cell(
            why="qwen4_exp decode P=1: one row per generation sequence over "
            "page-crossing histories (70, 35)",
            spec=dict(
                _QWEN4_DENSE,
                cached=(70, 35),
                new=(1, 1),
                num_contexts=0,
                predicted_tokens_per_seq=1,
            ),
        ),
        Cell(
            why="qwen4_exp MTP verify P=2 (max_draft_len 1): draft row t attends "
            "[0, L-P+t], bottom-right causal within the block",
            spec=dict(
                _QWEN4_DENSE,
                cached=(70, 35),
                new=(2, 2),
                num_contexts=0,
                predicted_tokens_per_seq=2,
            ),
        ),
        Cell(
            why="qwen4_exp MTP verify P=3 (max_draft_len 2): a separate decode "
            "compile, P reaches the kernel as maxSeqLenQ",
            spec=dict(
                _QWEN4_DENSE,
                cached=(70, 35),
                new=(3, 3),
                num_contexts=0,
                predicted_tokens_per_seq=3,
            ),
        ),
        Cell(
            why="qwen4_exp MTP verify P=4 (max_draft_len 3)",
            spec=dict(
                _QWEN4_DENSE,
                cached=(70, 35),
                new=(4, 4),
                num_contexts=0,
                predicted_tokens_per_seq=4,
            ),
        ),
        *(
            Cell(
                why=f"qwen4_exp MTP verify P={p} (max_draft_len {p - 1})"
                + (": the target's ceiling" if p == 8 else ""),
                spec=dict(
                    _QWEN4_DENSE,
                    cached=(70, 35),
                    new=(p, p),
                    num_contexts=0,
                    predicted_tokens_per_seq=p,
                ),
            )
            for p in (5, 6, 7, 8)
        ),
        Cell(
            why="qwen4_exp P=8 G=1: one draft block from row 95, across the page edge at 96",
            spec=dict(
                _QWEN4_DENSE,
                cached=(95,),
                new=(8,),
                num_contexts=0,
                predicted_tokens_per_seq=8,
            ),
        ),
        # The trtllm-gen launch picks its generation kernel by batch size at P 5..8, so each P
        # is also driven at 32 requests and at one batch in 17..20, with histories of every
        # page shape in the step.
        *(
            Cell(
                why=f"qwen4_exp bf16 batch P={p} G=32: 32 blocks over histories 1..218, "
                f"{32 * p} q rows",
                spec=dict(
                    _QWEN4_DENSE_G32,
                    cached=_spread(32, 218),
                    new=(p,) * 32,
                    num_contexts=0,
                    predicted_tokens_per_seq=p,
                ),
            )
            for p in (5, 6, 7, 8)
        ),
        *(
            Cell(
                why=f"qwen4_exp bf16 batch P={p} G={g}: {g} blocks over histories 200..1, "
                f"{g * p} q rows",
                spec=dict(
                    _QWEN4_DENSE_G32,
                    cached=tuple(reversed(_spread(g, 200))),
                    new=(p,) * g,
                    num_contexts=0,
                    predicted_tokens_per_seq=p,
                ),
            )
            for p, g in ((5, 17), (6, 18), (7, 19), (8, 20))
        ),
        Cell(
            why="qwen4_exp mixed step: a fresh 20-row context ahead of two P=2 "
            "draft blocks, generation rows token-major after the context rows",
            spec=dict(
                _QWEN4_DENSE,
                cached=(0, 70, 35),
                new=(20, 2, 2),
                num_contexts=1,
                predicted_tokens_per_seq=2,
            ),
        ),
        Cell(
            why="qwen4_exp mixed step P=8: a fresh 20-row context ahead of two P=8 "
            "draft blocks over histories 70 and 35",
            spec=dict(
                _QWEN4_DENSE,
                cached=(0, 70, 35),
                new=(20, 8, 8),
                num_contexts=1,
                predicted_tokens_per_seq=8,
            ),
        ),
        Cell(
            why="qwen4_exp hybrid pool: row 47 of the 48-row two-pool mapping "
            "(pool 0, layer-in-pool 11), P=2 verify",
            spec=dict(
                _QWEN4_DENSE,
                pool_layout="hybrid48",
                local_layer_idx=47,
                cached=(64, 25),
                new=(2, 2),
                num_contexts=0,
                predicted_tokens_per_seq=2,
            ),
        ),
        Cell(
            why="qwen4_exp P=8 hybrid pool: row 47 of the 48-row two-pool mapping "
            "(pool 0, layer-in-pool 11), P=8 verify",
            spec=dict(
                _QWEN4_DENSE,
                pool_layout="hybrid48",
                local_layer_idx=47,
                cached=(64, 25),
                new=(8, 8),
                num_contexts=0,
                predicted_tokens_per_seq=8,
            ),
        ),
        Cell(
            why="qwen4_exp vision tower: no KV cache, MHA 16/16 d72, padding mask, "
            "frames of 27/9/41 rows each attending only to itself",
            spec=dict(_QWEN4_VISION, seg_lens=(27, 9, 41)),
        ),
        Cell(
            why="mmmu serving shape: one 5580-row frame, the 62x90 patch grid",
            spec=dict(_QWEN4_VISION, seg_lens=(5580,)),
        ),
        Cell(
            why="mmmu serving shape: one 8160-row frame, the largest single-segment call",
            spec=dict(_QWEN4_VISION, seg_lens=(8160,)),
        ),
        Cell(
            why="mmmu serving shape: 2204 rows in four frames",
            spec=dict(_QWEN4_VISION, seg_lens=(728, 728, 460, 288)),
        ),
        Cell(
            why="mmmu serving shape: 1892 rows in five frames, the most segments per call",
            spec=dict(_QWEN4_VISION, seg_lens=(480, 480, 400, 352, 180)),
        ),
        Cell(
            why="mmmu serving shape: 9008 rows in two frames",
            spec=dict(_QWEN4_VISION, seg_lens=(8000, 1008)),
        ),
        Cell(
            why="mmmu serving shape: 22144 rows in two frames, the largest call; its "
            "12544-row frame is the longest segment",
            spec=dict(_QWEN4_VISION, seg_lens=(12544, 9600)),
        ),
        # qwen4_exp's own geometry over an fp8 pool: the kv head count changes the compiled
        # decode kernel, so the 24/4/256 cells above do not cover it.
        Cell(
            why="qwen4_exp fp8 pool paged-context prefill, 24/2/256: a fresh 48-row "
            "sequence beside 33 rows over a 48-token e4m3 prefix",
            spec=dict(
                _QWEN4_FP8,
                cached=(0, 48),
                new=(48, 33),
                num_contexts=2,
                predicted_tokens_per_seq=1,
            ),
        ),
        Cell(
            why="qwen4_exp fp8 pool decode P=1, 24/2/256: histories 200 and 48 span "
            "pages and KV tiles",
            spec=dict(
                _QWEN4_FP8,
                cached=(200, 48),
                new=(1, 1),
                num_contexts=0,
                predicted_tokens_per_seq=1,
            ),
        ),
        *(
            Cell(
                why=f"qwen4_exp fp8 pool MTP verify P={p}, 24/2/256 (max_draft_len {p - 1})",
                spec=dict(
                    _QWEN4_FP8,
                    cached=(70, 35),
                    new=(p, p),
                    num_contexts=0,
                    predicted_tokens_per_seq=p,
                ),
            )
            for p in (2, 4, 5, 6, 7, 8)
        ),
        Cell(
            why="qwen4_exp fp8 pool mixed step P=3, 24/2/256: a 20-row context ahead "
            "of two P=3 blocks over histories 31 and 64",
            spec=dict(
                _QWEN4_FP8,
                cached=(0, 31, 64),
                new=(20, 3, 3),
                num_contexts=1,
                predicted_tokens_per_seq=3,
            ),
        ),
        Cell(
            why="qwen4_exp fp8 pool mixed step P=8, 24/2/256: a 20-row context ahead "
            "of two P=8 blocks over histories 31 and 64",
            spec=dict(
                _QWEN4_FP8,
                cached=(0, 31, 64),
                new=(20, 8, 8),
                num_contexts=1,
                predicted_tokens_per_seq=8,
            ),
        ),
        # KV stays within one 128-key tile (histories to 119 plus 8 rows), where the e4m3
        # probability model of `reference` is the kernel's own at any batch.
        Cell(
            why="qwen4_exp fp8 batch P=8 G=32: 32 blocks over histories 1..119, 256 q rows",
            spec=dict(
                _QWEN4_FP8_G32,
                cached=_spread(32, 119),
                new=(8,) * 32,
                num_contexts=0,
                predicted_tokens_per_seq=8,
            ),
        ),
        Cell(
            why="qwen4_exp fp8 hybrid pool P=2: row 47 of the 48-row two-pool mapping "
            "(pool 0, layer-in-pool 11) over e4m3 slabs",
            spec=dict(
                _QWEN4_FP8,
                pool_layout="hybrid48",
                local_layer_idx=47,
                cached=(64, 25),
                new=(2, 2),
                num_contexts=0,
                predicted_tokens_per_seq=2,
            ),
        ),
        Cell(
            why="qwen4_exp fp8 hybrid pool P=8: row 47 of the 48-row two-pool mapping "
            "(pool 0, layer-in-pool 11) over e4m3 slabs",
            spec=dict(
                _QWEN4_FP8,
                pool_layout="hybrid48",
                local_layer_idx=47,
                cached=(64, 25),
                new=(8, 8),
                num_contexts=0,
                predicted_tokens_per_seq=8,
            ),
        ),
        Cell(
            why="qwen4_exp nvfp4 pool paged-context prefill, 24/2/256: a fresh 48-row "
            "sequence beside 33 rows over a 48-token e2m1 prefix",
            spec=dict(
                _QWEN4_NVFP4,
                cached=(0, 48),
                new=(48, 33),
                num_contexts=2,
                predicted_tokens_per_seq=1,
            ),
        ),
        Cell(
            why="qwen4_exp nvfp4 pool decode P=1, 24/2/256: histories 200 and 48",
            spec=dict(
                _QWEN4_NVFP4,
                cached=(200, 48),
                new=(1, 1),
                num_contexts=0,
                predicted_tokens_per_seq=1,
            ),
        ),
        Cell(
            why="qwen4_exp nvfp4 pool MTP verify P=2, 24/2/256 (max_draft_len 1)",
            spec=dict(
                _QWEN4_NVFP4,
                cached=(70, 35),
                new=(2, 2),
                num_contexts=0,
                predicted_tokens_per_seq=2,
            ),
        ),
        Cell(
            why="qwen4_exp nvfp4 pool MTP verify P=4, 24/2/256 (max_draft_len 3)",
            spec=dict(
                _QWEN4_NVFP4,
                cached=(70, 35),
                new=(4, 4),
                num_contexts=0,
                predicted_tokens_per_seq=4,
            ),
        ),
        Cell(
            why="qwen4_exp nvfp4 pool mixed step P=1: a 20-row context ahead of two "
            "1-row sequences over histories 31 and 64",
            spec=dict(
                _QWEN4_NVFP4,
                cached=(0, 31, 64),
                new=(20, 1, 1),
                num_contexts=1,
                predicted_tokens_per_seq=1,
            ),
        ),
        Cell(
            why="qwen4_exp nvfp4 pool mixed step P=4: a 20-row context ahead of two 4-row blocks",
            spec=dict(
                _QWEN4_NVFP4,
                cached=(0, 31, 64),
                new=(20, 4, 4),
                num_contexts=1,
                predicted_tokens_per_seq=4,
            ),
        ),
    ),
    Fp4BlockScaleMoeRunner: (
        Cell(
            why="qwen4_exp decode: one token over 512 experts, top-10 renormalized, H 2560 / I 640",
            spec=dict(tokens=1, **_QWEN4_EXP),
        ),
        Cell(
            why="qwen4_exp at 4 rows: the last count the pre-routed path's static-block "
            "permutation kernel takes",
            spec=dict(tokens=4, **_QWEN4_EXP),
        ),
        Cell(
            why="qwen4_exp at 5 rows: the first count on the single-cluster permutation "
            "kernel (the dynamic-block one stops at 256 experts)",
            spec=dict(tokens=5, **_QWEN4_EXP),
        ),
        Cell(
            why="qwen4_exp prefill: 8192 rows, the single-cluster kernel's cap and the top "
            "of the certified column",
            spec=dict(tokens=8192, **_QWEN4_EXP),
        ),
    ),
    Fp4Quantize: (
        Cell(
            why="qwen3.8 decode: 1 token, hidden 2560, linear -- its routed-expert input",
            spec=dict(shape=(1, 2560), dtype=torch.bfloat16, vec=16, swizzled=False),
        ),
        Cell(
            why="qwen3.8, 7 rows at hidden 2560, linear: several rows, no row padding",
            spec=dict(shape=(7, 2560), dtype=torch.bfloat16, vec=16, swizzled=False),
        ),
        Cell(
            why="qwen3.8 prefill at max_num_tokens 8192, hidden 2560, linear",
            spec=dict(shape=(8192, 2560), dtype=torch.bfloat16, vec=16, swizzled=False),
        ),
    ),
}
