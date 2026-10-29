# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""NVFP4-weight / NVFP4-activation mixture-of-experts layer: caller-supplied
top-k + grouped FC1 GEMM + gated activation + NVFP4 requantization + grouped
FC2 GEMM + routing-weighted combine, in one trtllm-gen call."""

from __future__ import annotations

from typing import Optional, Tuple

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*
from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

#: NVFP4 block size: one e4m3 scale per 16 elements along K.
SF_VEC_SIZE = 16

#: Largest e2m1 magnitude, and the largest finite e4m3 magnitude. Their product
#: is the global scale a checkpoint's `input_scale` encodes.
E2M1_MAX = 6.0
E4M3_MAX = 448.0

#: `act_type` for SwiGlu, the only activation this entry certifies.
ACT_TYPE_SWIGLU = 0

#: The top of the certified token column at the R1 and qwen4_exp geometries:
#: trtllm's default `max_num_tokens`. A caller at those geometries whose own
#: token count exceeds it must chunk. The Mistral Large 4 cells certify 1..131072
#: at theirs (H 6144 / I 6144 / 128 experts / top-4 / 32-wide windows): the
#: tep4 cells 1..32768, the dep4 cells the gathered 32769..131072; and the
#: tp4 and tp8 cells 1..32768 at theirs (H 6144 / I 1536 and 768 / all 128
#: experts local / top-4); and the tep8 cells 1..32768 and the dep8 cells 1..32768 gathered
#: tokens per call at 16-wide windows.
MAX_CERTIFIED_TOKENS = 8192

#: The 16 e2m1 code points in code order: sign bit 3, exponent bits 2:1,
#: mantissa bit 0.
_E2M1_VALUES = (
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    6.0,
    -0.0,
    -0.5,
    -1.0,
    -1.5,
    -2.0,
    -3.0,
    -4.0,
    -6.0,
)

#: One expert stack's `(codes, block scales)`, unshuffled and unswizzled.
Operand = Tuple[torch.Tensor, torch.Tensor]


def _e2m1_rne(a: torch.Tensor) -> torch.Tensor:
    """Round to nearest even on the e2m1 grid, saturating at +-6.

    The grid step is 0.5 below 2, 1 below 4 and 2 above -- the binade step of
    e2m1's three exponents, the subnormal binade sharing the first normal one's.
    `torch.round` is banker's rounding, which is the format's ties-to-even-code
    rule: 0.25 -> 0, 0.75 -> 1, 1.25 -> 1, 1.75 -> 2, 2.5 -> 2, 3.5 -> 4,
    5.0 -> 4.
    """
    step = torch.where(a.abs() < 2.0, 0.5, torch.where(a.abs() < 4.0, 1.0, 2.0))
    return torch.sign(a) * torch.clamp(torch.round(a.abs() / step) * step, max=E2M1_MAX)


def quantize_nvfp4(x: torch.Tensor, global_scale: float) -> torch.Tensor:
    """NVFP4 quantize-dequantize of `x` under `global_scale`.

    Per 16 consecutive columns: `sf = e4m3(g * blockmax / 6)` and
    `data = e2m1(x * g / sf)`, returning `data * sf` -- the value a downstream
    block-scaled MMA multiplies, which is `g` times the reconstruction of `x`.
    This models the FC1 epilogue, which the entry's test pins bit-exactly.
    """
    rows, cols = x.shape
    blocks = x.reshape(rows, cols // SF_VEC_SIZE, SF_VEC_SIZE)
    sf = (
        (global_scale * blocks.abs().amax(dim=-1, keepdim=True) / E2M1_MAX)
        .clamp(max=E4M3_MAX)
        .to(torch.float8_e4m3fn)
        .float()
    )
    out_scale = torch.where(sf == 0, torch.zeros_like(sf), global_scale / sf)
    return (_e2m1_rne(blocks * out_scale) * sf).reshape(rows, cols)


def dequantize_nvfp4(codes: torch.Tensor, block_scales: torch.Tensor) -> torch.Tensor:
    """One expert's `[N, K]` fp32 weight from its codes and e4m3 block scales."""
    table = torch.tensor(_E2M1_VALUES, dtype=torch.float32, device=codes.device)
    return table[codes.long()] * block_scales.float().repeat_interleave(SF_VEC_SIZE, dim=1)


class Fp4BlockScaleMoeRunner(OpWrapper):
    """One NVFP4-weight MoE layer over NVFP4 (e2m1 + e4m3 block-scale)
    activations, entered pre-routed.

    The caller owns top-k: `topk_ids` carries global expert ids and
    `topk_weights` the combine weights, neither renormalized here. This rank
    answers for `[local_expert_offset, local_expert_offset + local_num_experts)`
    and contributes zero for every other id, so an expert-parallel split is N
    calls whose outputs sum -- which is what the deepseek dep4 target does;
    qwen4_exp makes one call over its whole expert stack, Mistral Large 4's tep4
    target one call per rank over its 32-wide window, Mistral Large 4's dep4
    target the same per-rank call over the four ranks' gathered tokens, and
    Mistral Large 4's dep8 target one call per rank over its 16-wide window for
    each chunk of at most 32768 of the eight ranks' gathered tokens, and its
    tep8 target one call per rank over its 16-wide window and all of the
    step's tokens.
    Mistral Large 4's tp4 and tp8 targets split the experts' width instead: one
    call per rank over all 128 experts, each cut to that rank's 1536-wide
    (tp4) or 768-wide (tp8) slice of its intermediate, and the ranks' outputs
    sum to the layer.

    With `do_finalize=True` the result is a one-element list holding the
    combined `[T, H]` bf16 rows, or an empty tensor when `output` was given.
    With `do_finalize=False` it is three tensors: per-slot expert rows, an
    unwritten scale buffer, and the expanded-index -> permuted-row map.
    """

    ARCHS = frozenset({Arch.SM_103})

    # DeepSeek-R1-0528's routed geometry. The dep4 target gathers four ranks'
    # tokens before the expert call and issues one call per 64-wide window, so
    # the window cells are the shape it runs and the full stack is the
    # single-rank one.
    _R1 = dict(hidden=7168, intermediate=2048, num_experts=256, top_k=8)

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

    # Mistral-Large-4-1T-A52B-NVFP4's routed experts at tep4: H 6144, I 6144,
    # top-4 of 128 experts renormalized (bf16 weights, as renorm_moe_routing_op
    # emits them), each rank answering for its 32-wide window. Every rank sees
    # all T tokens (no attention DP); decode is captured at 1..16 tokens and
    # context steps run eager up to max_num_tokens 32768, unchunked. Operands
    # carry the checkpoint's magnitudes: weight block scales e4m3 8..448,
    # per-expert weight_scale_2 across the checkpoint's whole range, activation
    # block scales over the whole e4m3 range (0, subnormals, 448), and the
    # `layer`'s own input scales in the three scalars. `tune_max_num_tokens` is
    # max_num_tokens, as the target passes it.
    _ML4 = dict(
        hidden=6144,
        intermediate=6144,
        num_experts=128,
        top_k=4,
        local_num_experts=32,
        routing="renormalized",
        tune_max_num_tokens=32768,
    )

    # The same routed experts at dep4 (attention DP, experts EP over 4 ranks):
    # each rank zero-pads its own tokens to T_max, the largest rank's count, and
    # gathers all four ranks' rows, so every rank's 32-wide window runs over the
    # same 4 x T_max tokens -- up to 4 x max_num_tokens 32768 = 131072, 524288
    # expanded slots -- and decode graphs capture 4 x 1/2/4/8/16 rows.
    # `tune_max_num_tokens` is 4 x max_num_tokens, as the dep4 target passes it.
    _ML4_131K = dict(_ML4, tune_max_num_tokens=131072)

    # The same routed experts at tp4 (experts tensor parallel over 4 ranks, no
    # expert parallelism): every rank holds all 128 experts of a layer, each cut
    # to its 1536-wide slice of the intermediate -- w1 and w3 by output rows
    # [1536r, 1536r + 1536), w2 by input columns, their block scales with them,
    # weight_scale_2 and input_scale whole -- so one call answers for every
    # expert at offset 0 and returns this rank's partial sum over its slice.
    # Every rank sees all T tokens; decode is captured at 1..16 tokens and
    # context steps run eager up to max_num_tokens 32768, unchunked.
    # `tune_max_num_tokens` is max_num_tokens, as the tp4 target passes it.
    _ML4_TP4 = dict(
        hidden=6144,
        intermediate=1536,
        num_experts=128,
        top_k=4,
        local_expert_offset=0,
        local_num_experts=128,
        routing="renormalized",
        tune_max_num_tokens=32768,
    )

    # The same routed experts over eight ranks, 16-wide windows: rank r answers
    # for the 16 whole experts [16r, 16r + 16). At tep8 (attention TP over 8
    # ranks, experts EP over 8) every rank sees all T tokens of the step, as at
    # tep4 -- decode captured at 1..16 tokens, context steps eager up to
    # max_num_tokens 32768, unchunked. At dep8 (attention DP) the call covers
    # the eight ranks' gathered tokens, 8 x T_max up to 262144, issued in calls
    # of at most 32768 gathered rows, so T per call is a multiple of 8 in
    # 8..32768; decode graphs capture 8 x 1..16 gathered rows. Both targets pass
    # `tune_max_num_tokens` 32768 (tep8's max_num_tokens, dep8's chunk). Cells
    # labelled "tep8 and dep8" are both targets' calls.
    _ML4_16 = dict(_ML4, local_num_experts=16)

    # The same routed experts at tp8 (experts tensor parallel over 8 ranks on
    # two nodes, no expert parallelism): every rank holds all 128 experts of a
    # layer, each cut to its 768-wide slice of the intermediate -- w1 and w3 by
    # output rows [768r, 768r + 768), w2 by input columns, their block scales
    # with them, weight_scale_2 and input_scale whole -- so one call answers for
    # every expert at offset 0 and returns this rank's partial sum over its
    # slice. Every rank sees all T tokens; decode is captured at 1..16 tokens
    # and context steps run eager up to max_num_tokens 32768, unchunked.
    # `tune_max_num_tokens` is max_num_tokens, as the tp8 target passes it.
    _ML4_TP8 = dict(
        hidden=6144,
        intermediate=768,
        num_experts=128,
        top_k=4,
        local_expert_offset=0,
        local_num_experts=128,
        routing="renormalized",
        tune_max_num_tokens=32768,
    )

    CELLS: tuple[Cell, ...] = (
        Cell(
            why="dep4's own call: the 64-wide expert window at offset 0, R1 geometry",
            spec=dict(tokens=64, local_expert_offset=0, local_num_experts=64, **_R1),
        ),
        Cell(
            why="a window at a non-zero offset -- the ids this rank must answer for move with it",
            spec=dict(tokens=64, local_expert_offset=192, local_num_experts=64, **_R1),
        ),
        Cell(
            why="the single-rank shape: all 256 experts local, and one token, the decode row count",
            spec=dict(tokens=1, local_expert_offset=0, local_num_experts=256, **_R1),
        ),
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
            why="qwen4_exp at 5 rows: the first count past the static-block permutation "
            "kernel, on the dynamic-block one (5..12 rows, at 512 experts too)",
            spec=dict(tokens=5, **_QWEN4_EXP),
        ),
        Cell(
            why="qwen4_exp prefill: 8192 rows, the single-cluster kernel's cap and the top "
            "of the certified column",
            spec=dict(tokens=8192, **_QWEN4_EXP),
        ),
        Cell(
            why="Mistral Large 4 tep4 decode graphs at 1..16 tokens, rank 0 window (0, 32), layer 1 "
            "scalars (FC2 global scale 6898.53, the checkpoint's largest): each captured, then "
            "replayed on new tokens and new routing",
            spec=dict(tokens=16, local_expert_offset=0, layer=1, graph=tuple(range(1, 17)), **_ML4),
        ),
        Cell(
            why="Mistral Large 4 tep4 decode, 16 tokens at layer 45's scalars (FC2 global scale "
            "1.36898, the checkpoint's smallest)",
            spec=dict(tokens=16, local_expert_offset=0, layer=45, **_ML4),
        ),
        Cell(
            why="Mistral Large 4 tep4 eager token counts to max_num_tokens 32768, rank 3 window "
            "(96, 32): every routing, tile and finalize kernel edge, each count bitwise the "
            "leading rows of the 32768-token call",
            spec=dict(
                tokens=32768,
                local_expert_offset=96,
                layer=45,
                sweep=tuple(
                    sorted(
                        {
                            *range(1, 257),
                            *(2**k + d for k in range(8, 15) for d in (-1, 0, 1)),
                            *(263, 264, 519, 520, 1031, 1032, 9215, 9216, 9217, 32767, 32768),
                        }
                    )
                ),
                **_ML4,
            ),
        ),
        Cell(
            why="Mistral Large 4 tep4 32768 tokens, rank 0 window (0, 32), layer 3 scalars (the "
            "largest input global scale, 9039.45): the cooperative routing kernel",
            spec=dict(tokens=32768, local_expert_offset=0, layer=3, **_ML4),
        ),
        Cell(
            why="Mistral Large 4 tep4 32768 tokens, rank 1 window (32, 32), layer 65 scalars (the "
            "smallest input global scale, 23.0112): the FC2 input far past its calibrated "
            "range, where its block scales saturate at 448",
            spec=dict(tokens=32768, local_expert_offset=32, layer=65, **_ML4),
        ),
        Cell(
            why="Mistral Large 4 tep4 32768 tokens, rank 2 window (64, 32), layer 71 scalars",
            spec=dict(tokens=32768, local_expert_offset=64, layer=71, **_ML4),
        ),
        Cell(
            why="Mistral Large 4 tep4 32768 tokens with every slot in window (0, 32): 131072 "
            "permuted rows on one rank, all 32768 tokens on local expert 31",
            spec=dict(tokens=32768, local_expert_offset=0, layer=1, skew="one_window", **_ML4),
        ),
        Cell(
            why="Mistral Large 4 dep4 rank 0 window (0, 32) at 131072 gathered tokens (4 x "
            "max_num_tokens 32768, 524288 expanded slots), layer 3 scalars (the largest input "
            "global scale, 9039.45)",
            spec=dict(tokens=131072, local_expert_offset=0, layer=3, **_ML4_131K),
        ),
        Cell(
            why="Mistral Large 4 dep4 rank 1 window (32, 32) at 131072 gathered tokens, layer 65 "
            "scalars (the smallest input global scale, 23.0112): the FC2 input far past its "
            "calibrated range",
            spec=dict(tokens=131072, local_expert_offset=32, layer=65, **_ML4_131K),
        ),
        Cell(
            why="Mistral Large 4 dep4 rank 2 window (64, 32) at 131072 gathered tokens, layer 71 "
            "scalars",
            spec=dict(tokens=131072, local_expert_offset=64, layer=71, **_ML4_131K),
            # Its 16-ulp element gate holds at this draw (12.0 ulp) and not at every draw:
            # another seed lands two of 805M elements at 16.4 ulp, RMS unchanged at 0.75.
            seed=1,
        ),
        Cell(
            why="Mistral Large 4 dep4 rank 3 window (96, 32) sweep of gathered token counts "
            "32769..131072, layer 45 scalars: each count bitwise the leading rows of the "
            "131072-token call, on both sides of 2^16 and at 2^17 - 1",
            spec=dict(
                tokens=131072,
                local_expert_offset=96,
                layer=45,
                sweep=(32769, 65535, 65536, 65537, 98304, 131071, 131072),
                **_ML4_131K,
            ),
        ),
        Cell(
            why="Mistral Large 4 dep4 one window at 131072 gathered tokens: all 524288 slots in "
            "window (0, 32), all 131072 tokens on local expert 31 -- per-slot rows to 524287 "
            "and beyond, past the 2^31-byte (row 174763) and 2^31-element (row 349526) marks "
            "of the bf16 FC2 output",
            spec=dict(
                tokens=131072, local_expert_offset=0, layer=1, skew="one_window", **_ML4_131K
            ),
        ),
        Cell(
            why="Mistral Large 4 dep4 padding skew at 131072 gathered tokens: one rank at 32768 "
            "tokens and three at 1, so 98301 all-zero padding rows ride each of experts 0..3 "
            "at 0.25 in window (0, 32), and come back +0.0",
            spec=dict(
                tokens=131072,
                local_expert_offset=0,
                layer=1,
                skew="padded_zero_rows",
                rank_tokens=(32768, 1, 1, 1),
                **_ML4_131K,
            ),
        ),
        Cell(
            why="Mistral Large 4 dep4 decode graphs at 4/8/16/32/64 gathered tokens (4 ranks x "
            "1/2/4/8/16), rank 0 window (0, 32), layer 1 scalars: each captured, then replayed "
            "on new tokens and new routing",
            spec=dict(
                tokens=64, local_expert_offset=0, layer=1, graph=(4, 8, 16, 32, 64), **_ML4_131K
            ),
        ),
        Cell(
            why="Mistral Large 4 tp4 decode graphs at 1..16 tokens, all 128 experts at their 1536-wide "
            "slice, layer 1 scalars (FC2 global scale 6898.53, the checkpoint's largest): each "
            "captured, then replayed on new tokens and new routing",
            spec=dict(tokens=16, layer=1, graph=tuple(range(1, 17)), **_ML4_TP4),
        ),
        Cell(
            why="Mistral Large 4 tp4 decode, 16 tokens at layer 45's scalars (FC2 global scale "
            "1.36898, the checkpoint's smallest), all 128 experts at their 1536-wide slice",
            spec=dict(tokens=16, layer=45, **_ML4_TP4),
        ),
        Cell(
            why="Mistral Large 4 tp4 eager token counts to max_num_tokens 32768, all 128 experts at "
            "their 1536-wide slice, layer 45: every routing, tile and finalize kernel edge (the "
            "token tile 8 to 287 tokens, then 16, 32, 64, 128 and 256 from 288, 544, 1056, 2080 "
            "and 4128), each count bitwise the leading rows of the 32768-token call",
            spec=dict(
                tokens=32768,
                layer=45,
                sweep=tuple(
                    sorted(
                        {
                            *range(1, 258),
                            *(2**k + d for k in range(9, 15) for d in (-1, 0, 1)),
                            *(edge + d for edge in (288, 544, 1056, 2080, 4128) for d in (-1, 0)),
                            *(9215, 9216, 9217, 32767, 32768),
                        }
                    )
                ),
                **_ML4_TP4,
            ),
        ),
        Cell(
            why="Mistral Large 4 tp4 32768 tokens, layer 3 scalars (the largest input global scale, "
            "9039.45): 131072 slots over all 128 experts at their 1536-wide slice, the "
            "cooperative routing kernel",
            spec=dict(tokens=32768, layer=3, **_ML4_TP4),
        ),
        Cell(
            why="Mistral Large 4 tp4 32768 tokens, layer 65 scalars (the smallest input global scale, "
            "23.0112): the FC2 input far past its calibrated range, where its block scales "
            "saturate at 448",
            spec=dict(tokens=32768, layer=65, **_ML4_TP4),
        ),
        Cell(
            why="Mistral Large 4 tp4 32768 tokens, layer 71 scalars, all 128 experts at their "
            "1536-wide slice",
            spec=dict(tokens=32768, layer=71, **_ML4_TP4),
        ),
        Cell(
            why="Mistral Large 4 tp4 32768 tokens with every token's four slots on experts 124..127: "
            "32768 rows on each of four experts whose FC1 codes start past 2^31 e2m1 elements",
            spec=dict(tokens=32768, layer=1, skew="top_experts", **_ML4_TP4),
        ),
        Cell(
            why="Mistral Large 4 tp4 four width slices at 32768 tokens, layer 45: the four ranks' "
            "partials, summed in fp32 and as a bf16 running sum in rank order, against the "
            "whole-width (I 6144) layer; one slice left out or one run twice lands outside",
            spec=dict(tokens=32768, layer=45, width_slices=4, **_ML4_TP4),
        ),
        Cell(
            why="Mistral Large 4 dep8 decode graphs at 8..128 gathered tokens (8 ranks x 1..16), "
            "rank 0 window (0, 16), layer 1 scalars: each captured, then replayed on new "
            "tokens and new routing",
            spec=dict(
                tokens=128,
                local_expert_offset=0,
                layer=1,
                graph=tuple(range(8, 129, 8)),
                **_ML4_16,
            ),
        ),
        Cell(
            why="Mistral Large 4 dep8 rank 7 decode graphs at 8..128 gathered tokens, window (112, "
            "16), layer 1 scalars: each captured at offset 112, then replayed on new tokens and "
            "new routing with slots inside the window and outside it",
            spec=dict(
                tokens=128,
                local_expert_offset=112,
                layer=1,
                graph=tuple(range(8, 129, 8)),
                **_ML4_16,
            ),
        ),
        Cell(
            why="Mistral Large 4 dep8 128 gathered tokens eager at layer 45's scalars (FC2 global "
            "scale 1.36898, the checkpoint's smallest), window (0, 16)",
            spec=dict(tokens=128, local_expert_offset=0, layer=45, **_ML4_16),
        ),
        Cell(
            why="Mistral Large 4 dep8 eager gathered token counts to the 32768-row chunk, rank 7 "
            "window (112, 16): every count 1..256 and each routing, tile and finalize kernel "
            "edge above it, each bitwise the leading rows of the 32768-token call",
            spec=dict(
                tokens=32768,
                local_expert_offset=112,
                layer=45,
                sweep_range=(1, 256),
                # 2^k - 1, 2^k, 2^k + 1 to 2^14 + 1, and the tile and top
                # edges past 256, ascending, no count twice.
                sweep_also=tuple(
                    sorted(
                        (
                            *(2**k + d for k in range(9, 15) for d in (-1, 0, 1)),
                            *(259, 260, 515, 516, 9215, 9216, 9217, 32767, 32768),
                        )
                    )
                ),
                **_ML4_16,
            ),
        ),
        Cell(
            why="Mistral Large 4 tep8 and dep8 rank 0 window (0, 16), 32768 tokens (a tep8 step, a "
            "whole dep8 chunk), layer 3 scalars (the largest input global scale, 9039.45): the "
            "cooperative routing kernel",
            spec=dict(tokens=32768, local_expert_offset=0, layer=3, **_ML4_16),
        ),
        Cell(
            why="Mistral Large 4 tep8 and dep8 rank 1 window (16, 16), 32768 tokens, layer 65 "
            "scalars (the smallest input global scale, 23.0112): the FC2 input far past its "
            "calibrated range, where its block scales saturate at 448",
            spec=dict(tokens=32768, local_expert_offset=16, layer=65, **_ML4_16),
        ),
        Cell(
            why="Mistral Large 4 dep8 rank 4 window (64, 16), a whole 32768-token chunk, layer 71 "
            "scalars",
            spec=dict(tokens=32768, local_expert_offset=64, layer=71, **_ML4_16),
        ),
        Cell(
            why="Mistral Large 4 tep8 and dep8, one window takes all 32768 tokens: every slot in "
            "window (0, 16), 131072 permuted rows over 16 experts, all 32768 tokens on local "
            "expert 15",
            spec=dict(tokens=32768, local_expert_offset=0, layer=1, skew="one_window", **_ML4_16),
        ),
        Cell(
            why="Mistral Large 4 dep8 padding skew, chunk 1 of 262144 gathered tokens (one rank at "
            "32768, seven at 1): one live row and 32767 all-zero padding rows, whose 131068 "
            "slots ride experts 0..3 at 0.25 in window (0, 16) and come back +0.0",
            spec=dict(
                tokens=32768,
                local_expert_offset=0,
                layer=1,
                skew="padded_zero_rows",
                rank_tokens=(32768, 1, 1, 1, 1, 1, 1, 1),
                chunk=1,
                **_ML4_16,
            ),
        ),
        Cell(
            why="Mistral Large 4 tep8 and dep8 eight windows over 32768 tokens, layer 45 scalars: "
            "the eight ranks' partials, summed in fp32 and as a bf16 running sum in rank "
            "order, against the 128-expert layer",
            spec=dict(tokens=32768, local_expert_offset=0, layer=45, windows=8, **_ML4_16),
        ),
        Cell(
            why="Mistral Large 4 tep8 decode graphs at 1..16 tokens, rank 0 window (0, 16), layer 1 "
            "scalars (FC2 global scale 6898.53, the checkpoint's largest): each captured, then "
            "replayed on new tokens and new routing",
            spec=dict(
                tokens=16, local_expert_offset=0, layer=1, graph=tuple(range(1, 17)), **_ML4_16
            ),
        ),
        Cell(
            why="Mistral Large 4 tep8 rank 7 decode graphs at 1..16 tokens, window (112, 16), layer 1 "
            "scalars: each captured at offset 112, then replayed on new tokens and new routing "
            "with slots inside the window and outside it",
            spec=dict(
                tokens=16, local_expert_offset=112, layer=1, graph=tuple(range(1, 17)), **_ML4_16
            ),
        ),
        Cell(
            why="Mistral Large 4 tep8 decode, 16 tokens at layer 45's scalars (FC2 global scale "
            "1.36898, the checkpoint's smallest), window (0, 16)",
            spec=dict(tokens=16, local_expert_offset=0, layer=45, **_ML4_16),
        ),
        Cell(
            why="Mistral Large 4 tep8 eager token counts to max_num_tokens 32768, rank 7 window "
            "(112, 16): every routing, tile and finalize kernel edge at 16 local experts, each "
            "count bitwise the leading rows of the 32768-token call",
            spec=dict(
                tokens=32768,
                local_expert_offset=112,
                layer=45,
                # Every count to 2^8 + 1, then 2^k +- 1 to 2^14 + 1, and the
                # tile and top edges, no count twice.
                sweep=tuple(
                    sorted(
                        {
                            *range(1, 258),
                            *(2**k + d for k in range(9, 15) for d in (-1, 0, 1)),
                            *(259, 260, 515, 516, 9215, 9216, 9217, 32767, 32768),
                        }
                    )
                ),
                **_ML4_16,
            ),
        ),
        Cell(
            why="Mistral Large 4 tep8 32768 tokens, rank 3 window (48, 16), layer 71 scalars",
            spec=dict(tokens=32768, local_expert_offset=48, layer=71, **_ML4_16),
        ),
        Cell(
            why="Mistral Large 4 tp8 decode graphs at 1..16 tokens, all 128 experts at their 768-wide "
            "slice, layer 1 scalars (FC2 global scale 6898.53, the checkpoint's largest): each "
            "captured, then replayed on new tokens and new routing",
            spec=dict(tokens=16, layer=1, graph=tuple(range(1, 17)), **_ML4_TP8),
        ),
        Cell(
            why="Mistral Large 4 tp8 decode, 16 tokens at layer 45's scalars (FC2 global scale "
            "1.36898, the checkpoint's smallest), all 128 experts at their 768-wide slice",
            spec=dict(tokens=16, layer=45, **_ML4_TP8),
        ),
        Cell(
            why="Mistral Large 4 tp8 eager token counts to max_num_tokens 32768, all 128 experts at "
            "their 768-wide slice, layer 45: every routing, tile and finalize kernel edge (the "
            "token tile 8 to 287 tokens, then 16, 32, 64, 128 and 256 from 288, 544, 1056, 2080 "
            "and 4128), each count bitwise the leading rows of the 32768-token call",
            spec=dict(
                tokens=32768,
                layer=45,
                sweep=tuple(
                    sorted(
                        {
                            *range(1, 258),
                            *(2**k + d for k in range(9, 15) for d in (-1, 0, 1)),
                            *(edge + d for edge in (288, 544, 1056, 2080, 4128) for d in (-1, 0)),
                            *(9215, 9216, 9217, 32767, 32768),
                        }
                    )
                ),
                **_ML4_TP8,
            ),
        ),
        Cell(
            why="Mistral Large 4 tp8 32768 tokens, layer 3 scalars (the largest input global scale, "
            "9039.45): 131072 slots over all 128 experts at their 768-wide slice, the "
            "cooperative routing kernel",
            spec=dict(tokens=32768, layer=3, **_ML4_TP8),
        ),
        Cell(
            why="Mistral Large 4 tp8 32768 tokens, layer 65 scalars (the smallest input global scale, "
            "23.0112): the FC2 input far past its calibrated range, where its block scales "
            "saturate at 448",
            spec=dict(tokens=32768, layer=65, **_ML4_TP8),
        ),
        Cell(
            why="Mistral Large 4 tp8 32768 tokens, layer 71 scalars, all 128 experts at their "
            "768-wide slice",
            spec=dict(tokens=32768, layer=71, **_ML4_TP8),
        ),
        Cell(
            why="Mistral Large 4 tp8 32768 tokens with every token's four slots on experts 124..127: "
            "32768 rows on each of four experts, the most rows one call puts on an expert",
            spec=dict(tokens=32768, layer=1, skew="top_experts", **_ML4_TP8),
        ),
        Cell(
            why="Mistral Large 4 tp8 eight width slices at 32768 tokens, layer 45: the eight ranks' "
            "partials, summed in fp32 and as a bf16 running sum in rank order, against the "
            "whole-width (I 6144) layer; one slice left out or one run twice lands outside",
            spec=dict(tokens=32768, layer=45, width_slices=8, **_ML4_TP8),
        ),
    )

    note = """
    Certified pre-routed only: `routing_logits` is `None` at every cell, so the
    kernel's own routing is not exercised and `routing_method_type`, `n_group`,
    `topk_group` and `routed_scaling_factor` are inert -- they are passed
    because the signature takes them, not because they do anything here.

    That holds at 512 experts / top-10 too, where it would matter if it were
    read: which `(num_experts, top_k)` pairs the routing kernels are compiled
    for is a per-policy tier list, and a pair with no tier is a logged error,
    not a raise. The pre-routed path substitutes its own preprocess/postprocess
    pair before dispatch, and `routing_method_type` 0, 2, 4, 5, 6 were measured
    bitwise identical to 1 there. It picks its permutation kernel by row count
    instead, read back from the launches at 128 experts / top-4 and at 512 /
    top-10: a static-block kernel up to 4 rows, a dynamic-block one for 5..12,
    a single-cluster one for 13..8192, and from 8193 an expert-count init plus
    a cooperative kernel (driven to 131072 rows at 128 experts).

    `act_type` 0 (SwiGlu) only. `gemm1_bias`, `gemm1_alpha`, `gemm1_beta`,
    `gemm1_clamp_limit` and `gemm2_bias` are `None`. `hidden_states_scale` is
    the linear (unswizzled) e4m3 block-scale layout, one scale per 16 elements
    of each row.

    At 512 experts only the whole stack is certified: no expert-parallel window
    of it, `do_finalize=False` or a caller-supplied `output` was measured there.

    `tune_max_num_tokens` is an autotuner bucket hint and was measured inert;
    it is not a token count. `T` is `hidden_states.shape[0]`, and 8192 is the
    top of the certified column at the R1 and qwen4_exp geometries (131072 at
    Mistral Large 4's, below). A warm autotuner cache was measured bitwise
    inert against a cold one at the R1 stack, its windows and the 512-expert
    stack; `local_num_experts` separates the tuner's cache entries, so each
    layout warms its own.

    The caller owns the weight layout entirely: FC1 and FC2 weights are
    pre-shuffled and their scales additionally 128x4 swizzled. Nothing here
    checks that, and getting it wrong is 300-3400 bf16 ulp with no error -- the
    entry's test builds the layout twice, once in torch and once through the
    trtllm preprocessing ops, and compares them before either reaches the op.

    Mistral Large 4 tep4 (H 6144, I 6144, top-4 of 128, windows of 32), on the
    cold fallback tactic. Kernels by token count, read back from the
    launches: permutation as above, cooperative from 8193 to 32768; both
    GEMMs take the token tile `nextPowerOfTwo(int(T * top_k /
    local_num_experts))` clamped to 8..256 -- 8 up to 71 tokens, 16 for
    72..135, 32 for 136..263, 64 for 264..519, 128 for 520..1031, 256 from
    1032; finalize -- scalar up to 49 tokens, vector loads from 50. Rows are
    batch invariant on that path: every count of the sweep cell, across all
    of those switches, is bitwise the leading rows of one 32768-token call. A
    CUDA graph captured at 1..16 tokens replays bitwise equal to eager on new
    tokens and new routing, the permutation kernels inside the graph.

    The tuned path. Every valid tactic of the op's tunable runner was run at
    1..16 tokens and at each autotuner bucket 32..8192 and 32768. From 256
    tokens up every one is bitwise the fallback. Below that, a few tile-8
    configs give other bits on some draws (8 of 216 at 11 tokens; 24 of 216
    at 32 and 64; 24 of 224 at 128), each inside the test's gate. So the
    cells certify whichever tactic warm-up keeps, while bitwise batch
    invariance holds on the cold path only. Pass `tune_max_num_tokens` =
    max_num_tokens (32768), as the tep4 target does: the tuner then adds a
    32768 bucket and maps 8193..32768 onto it. Warming at 32768 tokens wrote
    15 cache entries in 14.7 s and left the result bitwise unchanged at every
    count checked. Certified with TLLM_USE_FINE_GRAINED_SYNC unset (the
    engine's default, `use_fine_grained_sync` False), so the runner takes
    the non-fine-grained cubins.

    Mistral Large 4 dep4: the same windows over the four ranks' gathered tokens,
    4 x T_max up to 131072 (524288 expanded slots), on the cold fallback
    tactic. From 8193 to 131072 the launches read back do not change: the
    expert-count init plus the cooperative routing kernel, both GEMMs at
    token tile 256, the vector-load finalize -- under uniform routing at
    8193, 32768, 32769, 65536 and 131072, and at 131072 with every slot in
    one window and under the padding skew. Rows stay batch invariant: each
    sweep count is bitwise the leading rows of the 131072-token call. Decode
    graphs at 4..64 gathered tokens replay bitwise equal to eager. A padding
    row is `fp4_quantize` of an all-zero row -- every code and every block
    scale 0, read back from that op -- routed to experts 0..3 at 0.25, and
    comes back +0.0; a step carries up to 98301 of them on window (0, 32).
    The four windows over one gathered token set answer every slot exactly
    once: their partials, summed, pass the test's gate against the summed
    window references, while leaving one window out or running one twice
    lands 118-175 rms ulp out.
    The dep4 target passes `tune_max_num_tokens` = 4 x max_num_tokens =
    131072 and `use_dp` False, so the tuner's buckets are 1..8192 and 131072,
    and 8193..131072 map onto 131072. The op offers 8 valid tactics there
    (tiles 128 and 256), 8 at 8193 and 65537 too, and every one is
    bitwise the fallback -- uniform, one-window and padded alike. Warming at
    131072 tokens wrote 15 cache entries in 12.3 s and left the result
    bitwise unchanged at 8193, 32768, 65537 and 131072 in all three.

    Mistral Large 4 tep8 (H 6144, I 6144, top-4 of 128, windows of 16, every rank
    over all T tokens), on the cold fallback tactic. Kernels by token count,
    read back from the launches at every count of the tep8 sweep cell: the
    permutation and finalize kernels switch at the counts they do at tep4
    (their choice reads the token count, not the window:
    `runPostTopKPipeline` in `kernels/moe/trtllmGen/routing/
    RoutingFromTopKIds.cu`), and both GEMMs take the token tile at 16 local
    experts -- 8 up to 35 tokens, 16 for 36..67, 32 for 68..131, 64 for
    132..259, 128 for 260..515, 256 from 516. With every slot in one window
    at 32768 tokens the call launches what it does under uniform routing:
    the expert-count init plus the cooperative kernel, tile 256, the
    vector-load finalize. Rows stay batch invariant: every count of the
    sweep cell is bitwise the leading rows of one 32768-token call, and
    decode graphs at 1..16 tokens replay bitwise equal to eager at their own
    window -- captured on rank 0's (0, 16) and, after those in the same
    process, on rank 7's (112, 16). The offset reaches the kernels as a
    per-call int (`run_moe`'s `local_expert_offset`; the C++ runner object
    `FP4BlockScaleMoERunner.get_runner` caches is keyed on `act_type`
    alone, so every offset shares it), and a graph keeps the one it was
    captured at: the window-112 replays' data run at offset 0 land
    237-24805 rms ulp from `reference`. At decode counts a 16-wide window
    has few live rows, so one FC1 value the kernel requantizes to the
    neighbouring e2m1 code or e4m3 block scale -- read out of the kernel,
    each such value within 5.24e-5 relative of the rounding midpoint --
    moves a step's relative RMS against `reference` by several bf16 ulp
    (4.98 at one live row), so the test gates the decode graph replays on
    the element figure alone (its `_compare`). The eight windows over one
    32768-token step answer every slot exactly once: their
    partials, summed in fp32 and as a bf16 running sum in rank order, pass
    the test's gate against `reference` over all 128 experts at once, while
    leaving one window out or running one twice lands 84-119 rms ulp out.
    How the tep8 target sums them is its closing all-reduce's
    (`comm/cute_dsl_allreduce_add_rmsnorm`).
    The tuned path at tep8: the op offers 216 valid tactics at 1..35 tokens
    (tiles 8, 16 and 32), 224 at 36 and 64, 142 at 128, 80 at 256, 16 at 512
    and 8 (tiles 128 and 256) from 1024 to 32768. Every one is bitwise the
    fallback except 24 tile-8 configs at 9 and 11 tokens on the test's
    draws, each inside the test's gate; at 32768 tokens with every slot in
    one window all 8 are bitwise the fallback. Warming at 32768 tokens with
    `tune_max_num_tokens` 32768 wrote 15 cache entries in 8.5 s and left the
    result bitwise unchanged at 40 counts from 1 to 32768 and on the
    one-window draw.

    Mistral Large 4 dep8 (H 6144, I 6144, top-4 of 128, windows of 16): rank r
    calls its window (16r, 16) over the eight ranks' gathered tokens in
    calls of at most 32768 of them, on the cold fallback tactic. Kernels by
    token count, read back from the launches at each side of every switch:
    the permutation and finalize kernels switch at the counts they do at
    tep4's 32-wide windows, and both GEMMs take the token tile at 16 local
    experts -- 8 up to 35 tokens, 16
    for 36..67, 32 for 68..131, 64 for 132..259, 128 for 260..515, 256 from
    516. A 32768-token call with every slot in one window, and the padding
    chunk below, launch what a uniform one does: the expert-count init plus
    the cooperative kernel, tile 256, the vector-load finalize. Rows stay
    batch invariant: every count 1..256 and every edge above it to 32768 is
    bitwise the leading rows of one 32768-token call, so each chunk the
    target issues (a multiple of 8, a step's last chunk the remainder) is
    the leading rows of a certified call. Decode graphs at 8..128 gathered
    tokens (8 ranks x 1..16) replay bitwise equal to eager, captured on rank
    0's window (0, 16) and, after those in the same process, on rank 7's
    (112, 16). The offset reaches the kernels as a per-call int, and a graph
    keeps the one it was captured at: the window-112 replays' data run at
    offset 0 land 269-493 rms ulp from `reference`. With one rank at 32768
    tokens and seven at 1, chunks 1..7 of the 262144 gathered rows each hold
    one live row and 32767 padding rows, whose 131068 slots ride experts
    0..3 of window (0, 16) at 0.25 and come back +0.0. The eight windows
    over one 32768-token chunk answer every slot exactly once: their
    partials, summed in fp32 and as a bf16 running sum in rank order, pass
    the test's gate against `reference` over all 128 experts at once, while
    leaving one window out or running one twice lands 83-118 rms ulp out.
    How the dep8 target sums them is `comm/reducescatter`'s.
    The dep8 target passes `tune_max_num_tokens` = its chunk, 32768, and
    `use_dp` False, so the tuner's buckets are 1..8192 and 32768, and a call
    runs the tactic of the power of two at or below its count (8193..32768
    that of 32768). Every tactic valid at that bucket was run at each decode
    count 8..128 (216 at buckets 8..32, 224 at 64, 142 at 128), every valid
    one at 256..8192 and 32768 (80, 16, then 8), and at 32768 on the
    one-window and padding chunks: all bitwise the fallback except 24 tile-8
    configs of bucket 64 at 88 tokens on the test's draw, each inside the
    test's gate. Warming at 32768 tokens wrote 15 cache entries in 8.5 s
    (tile 8 kept at buckets 1..64, 16 at 128, 32 at 256, 64 at 512, 128 at
    1024..8192, 256 at 32768) and left the result bitwise unchanged at 46
    calls: 1..16, 24..128 step 8, 255 to 32768, and the one-window and
    padding chunks. CUDA graphs captured on that warm cache at 8..128
    gathered tokens replayed bitwise equal to warm eager, each inside the
    test's gate (a probe, not a cell).

    Operands, as a Mistral Large 4 loader builds them per local expert e (global
    id = local_expert_offset + e) from layer N's `experts.<id>.{w1,w2,w3}`:
    w1 is the gate, w3 the up and w2 the down projection; `.weight` is U8
    `[out, in/2]`, two e2m1 per byte (the op reads the even element from the
    low nibble), and `.weight_scale` F8_E4M3 `[out, in/16]`, linear.
      gemm1_weights[e]       rows of cat([w3.weight, w1.weight]) gathered by
                             fc1_perm: [12288, 3072] uint8
      gemm1_weights_scale[e] the same gather of cat([w3.weight_scale,
                             w1.weight_scale]), then the 128x4 swizzle:
                             [12288, 384] float8_e4m3fn
      gemm2_weights[e]       w2.weight rows gathered by blk32_perm: [6144, 3072]
      gemm2_weights_scale[e] the same gather of w2.weight_scale, then the
                             swizzle: [6144, 384]
    `blk32_perm` is the 32-row shuffle: within each block of 32 rows, source
    row 4u + v lands at row 8v + u. `fc1_perm` first interleaves the halves
    (row 2i is up row i, row 2i + 1 gate row i), then applies `blk32_perm`.
    `torch.ops.trtllm.shuffle_matrix(rows, perm)` is the row gather and
    `torch.ops.trtllm.block_scale_interleave` the swizzle; the test proves
    both equal to the torch recipe at these shapes, with no padding. The
    scalars are fp32, per local expert, from the layer's single input_scale
    per projection: output1_scale_gate_scalar = input_scale(w1) *
    weight_scale_2(w1); output1_scale_scalar = that / input_scale(w2);
    output2_scale_scalar = input_scale(w2) * weight_scale_2(w2). w1 and w3
    share both values on every expert. `hidden_states` and
    `hidden_states_scale` are `fp4_quantize(x, 1 / input_scale(w1), 16,
    False, False)`. The cells draw the operands at the checkpoint's own
    magnitudes. Weight block scales are 8..448 and activation block scales
    span the whole e4m3 range. weight_scale_2 covers its extremes over all
    9088 experts. The input scales are those of layers 1, 3, 45, 65 and 71:
    the extremes of both global scales, and the last layer. At layer 65's
    scalars the FC2 input runs far past its calibrated range -- on three of
    four experts sampled, 91-100 % of its block scales saturate at 448; on
    the 16-wide window's layer-65 cell's draws over its 16 experts, 84-100 %
    on 11 and 1.4-17.8 % on the five with the smallest w1 weight_scale_2 (tep8
    and dep8 runs) -- and kernel and reference still agree.

    Hard limits (sm_103, Mistral Large 4 tep4, 32768 tokens)
      gemm1_weights  local expert e starts at e2m1 element e * 2I * H
                     (75,497,472 per expert); experts 29..31 start past 2^31
                     elements, 28 straddles it. The tensor is 1,207,959,552
                     bytes. probed: each of 29, 30, 31 gives the same bits
                     moved to slot 1 -- passed
      permuted rows  allocated (32 + (4T - 32) // tile) * tile = 139,008 at
                     T 32768, tile 256; gemm2 output 139,008 x 6144 bf16 =
                     854,065,152 elements, 1,708,130,304 bytes; rows x 12,288
                     bytes reaches 2^31 at 174,763 rows -- not reached.
                     probed at the envelope's top: all 131,072 slots in one
                     window, 32,768 of them on one expert -- passed
      gemm1 output   the NVFP4 FC2 input, 3072 bytes a row: 427,032,576
                     bytes at 139,008 rows -- below 2^31
      the rest       T * top_k = 131,072 expanded slots; hidden_states
                     100,663,296 bytes; output 201,326,592 elements -- all
                     below 2^31. No index passes through fp32: the tile choice
                     truncates T * top_k / local_num_experts, 4096 at the top
      T              certified 1..32768 at this geometry by the tep4 cells,
                     to 131072 by the dep4 ones (below)

    Hard limits (sm_103, Mistral Large 4 dep4, 131072 gathered tokens)
      permuted rows  allocated (32 + (4T - 32) // tile) * tile = 532,224 at
                     T 131072, tile 256; the bf16 FC2 output is 3,269,984,256
                     elements, 6,539,968,512 bytes. A row's byte offset
                     passes 2^31 at row 174,763 and its element offset at
                     row 349,526. probed: the one-window cell's draw (all
                     524,288 slots in window (0, 32)) uses rows to 528,639 --
                     351,398 slots at or past row 174,763, 178,702 at or past
                     349,526 -- and a padding-skew draw rows to 429,280; both
                     pass the gate, and the per-slot rows (`do_finalize=False`)
                     gathered and weighted in torch equal the finalized output
                     bitwise -- passed. A control that reads the rows past
                     either mark from that many rows earlier picks up
                     unwritten tile-padding rows: NaN, which the gate
                     refuses. Both finalize kernels (`finalizeKernel`
                     and `finalizeKernelVecLoad` in
                     `kernels/moe/trtllmGen/DevKernel.cu`) form the row offset
                     in 64 bits
      gemm1 output   532,224 rows x 3072 bytes = 1,635,012,608 -- below 2^31
      the rest       T * top_k = 524,288 expanded slots (int32 indices);
                     hidden_states 402,653,184 bytes; output 805,306,368
                     elements -- all below 2^31. No index passes through
                     fp32: the tile choice truncates T * top_k /
                     local_num_experts, 16384 at the top
      T              certified 1..131072 at this geometry. The cooperative
                     routing kernel takes T <= (SMs - 8) * 128 * 64 / top_k =
                     294,912 on sm_103; above it the routing switches path --
                     computed, not probed

    Mistral Large 4 tp4 (H 6144, I 1536 -- each expert's slice of its 6144-wide
    intermediate -- top-4 of 128, all 128 experts local at offset 0), on the
    cold fallback tactic. Kernels by token count, read back from the launches
    (the GEMM tile at every count of the tp4 sweep cell, the permutation and
    finalize kernels on both sides of each switch): permutation as above, the
    static block kernel to 4 tokens, the dynamic one for 5..12, the
    single-cluster one for 13..8192, the expert-count init plus the
    cooperative kernel from 8193 to 32768; both GEMMs take the token tile
    `nextPowerOfTwo(int(T * top_k / local_num_experts))`, T / 32 here, clamped
    to 8..256 -- 8 up to 287 tokens, 16 for 288..543, 32 for 544..1055, 64 for
    1056..2079, 128 for 2080..4127, 256 from 4128; finalize -- scalar up to 49
    tokens, vector loads from 50, as at tep4 (that switch reads the hidden
    width and T, not the expert count). Rows are batch invariant on that path:
    every count of the sweep cell is bitwise the leading rows of one
    32768-token call. A CUDA graph captured at 1..16 tokens replays bitwise
    equal to eager on new tokens and new routing.

    The tuned path at tp4. The op's tunable runner offers 160 valid tactics
    (tiles 8, 16 and 32) at every count read up to 287 tokens, 164 at 288 and
    512, 106 at 1024, 60 at 2048, 12 at 4096 and 8 (tiles 128 and 256) at
    8192, 8193, 16384 and 32768. Every one was run at 1..16 tokens and at each
    autotuner bucket 32..8192 and 32768, and every one is bitwise the fallback
    there. The target passes `tune_max_num_tokens` = max_num_tokens = 32768
    and `use_dp` False, so the buckets are 1..8192 and 32768, and 8193..32768
    map onto 32768. Warming at 32768 tokens wrote 15 cache entries in 13.9 s
    and left the result bitwise unchanged at 33 counts from 1 to 32768.

    The four width slices. Rank r's call -- the same tokens, routing and
    scalars, its own slice of every expert -- returns the partial sum over
    intermediate columns [1536r, 1536r + 1536). The FC2-input requantization
    runs in 16-element blocks under the layer's one global scale and 1536 =
    96 x 16, so no block straddles a slice boundary: the four partials,
    summed in fp32 and as a bf16 running sum in rank order, pass the test's
    gate against the whole-width (I 6144) reference, while one slice left out
    lands 128 rms ulp out and slice 0 run twice 181. The element figure at the
    slice is a max over a narrower contraction, so the test gates elements
    at 32 ulp there (16 at the whole width) and RMS at 2 as everywhere but the
    decode graph replays; its
    `_compare` has the derivation and the measurements. How the tp4 target
    sums the partials is its closing all-reduce's.

    Operands at tp4, as a loader builds them for rank r from layer N's
    `experts.<e>.{w1,w2,w3}`, e = 0..127 (the slice is cut first, then laid
    out: the layout is a function of the slice's row count):
      gemm1_weights[e]       rows [1536r, 1536r + 1536) of w3.weight, then
                             the same rows of w1.weight, gathered by
                             fc1_perm: [3072, 3072] uint8
      gemm1_weights_scale[e] the same rows of the two weight_scale tensors,
                             the same gather, then the 128x4 swizzle:
                             [3072, 384] float8_e4m3fn
      gemm2_weights[e]       w2.weight byte columns [768r, 768r + 768)
                             (inputs [1536r, 1536r + 1536)), rows gathered
                             by blk32_perm: [6144, 768]
      gemm2_weights_scale[e] w2.weight_scale columns [96r, 96r + 96), the
                             same gather, then the swizzle: [6144, 96]
    3072 and 6144 rows are multiples of 128 and 384 and 96 scale columns
    multiples of 4, so the swizzle pads nothing; the test proves
    `block_scale_interleave` equal to the torch recipe at both scale shapes.
    weight_scale_2 and input_scale are per expert and per layer, unsplit, so
    the three scalars are tep4's, for all 128 experts. The cells draw the
    slices at tep4's magnitudes and the same five layers' input scales; at
    layer 65's the FC2 input saturates as at tep4 (99-100 % of the block
    scales at 448 on three of four experts sampled).

    Hard limits (sm_103, Mistral Large 4 tp4, 32768 tokens)
      gemm1_weights  local expert e starts at e2m1 element e * 2I * H
                     (18,874,368 per expert at I 1536): expert 113 straddles
                     2^31 elements, 114..127 start past it. The tensor is
                     1,207,959,552 bytes. probed: each of 113..127 gives the
                     same bits moved to slot 1, and a cell puts all 32768
                     tokens on each of 124..127 -- passed
      permuted rows  allocated (128 + (4T - 128) // tile) * tile = 163,584 at
                     T 32768, tile 256 -- read back as the per-slot rows of
                     `do_finalize=False` at 1, 16, 287, 288, 4128, 8193 and
                     32768 tokens, each the formula's; the bf16 gemm2 output is
                     1,005,060,096 elements, 2,010,120,192 bytes. A row's byte
                     offset passes 2^31 at row 174,763 -- not reached, 11,179
                     rows short (a uniform draw used rows to 147,645, the
                     top_experts skew to 131,071)
      gemm1 output   the NVFP4 FC2 input, 768 bytes a row: 125,632,512 bytes
                     at 163,584 rows -- below 2^31
      the rest       T * top_k = 131,072 expanded slots; gemm2_weights
                     1,207,959,552 e2m1 elements; hidden_states 100,663,296
                     bytes; output 201,326,592 elements -- all below 2^31. No
                     index passes through fp32: the tile choice truncates
                     T * top_k / local_num_experts, 1024 at the top
      T              certified 1..32768 at this geometry by the tp4 cells

    Mistral Large 4 tp8 (H 6144, I 768 -- each expert's slice of its 6144-wide
    intermediate -- top-4 of 128, all 128 experts local at offset 0), on the
    cold fallback tactic. Kernels by token count, read back from the launches
    (the GEMM tile at every count of the tp8 sweep cell, the permutation and
    finalize kernels on both sides of each switch): permutation as above, the
    static block kernel to 4 tokens, the dynamic one for 5..12, the
    single-cluster one for 13..8192, the expert-count init plus the
    cooperative kernel from 8193 to 32768; both GEMMs take the token tile
    `nextPowerOfTwo(int(T * top_k / local_num_experts))`, T / 32 here, clamped
    to 8..256 -- 8 up to 287 tokens, 16 for 288..543, 32 for 544..1055, 64 for
    1056..2079, 128 for 2080..4127, 256 from 4128 (the expert count sets
    them, not the width); finalize -- scalar up to 49 tokens, vector loads
    from 50. Rows are batch invariant on that path: every count of the sweep
    cell is bitwise the leading rows of one 32768-token call. A CUDA graph
    captured at 1..16 tokens replays bitwise equal to eager on new tokens and
    new routing.

    The tuned path at tp8. The op's tunable runner offers 152 valid tactics
    (tiles 8, 16 and 32) at every count read up to 287 tokens, 160 for
    288..543 (tile 64 joins), 107 for 544..1055, 61 for 1056..2079, 13 for
    2080..4127 and 5 (tiles 128 and 256) from 4128 to 32768. Every one was
    run at 1..16 tokens and at each autotuner bucket 32..8192 and 32768, and
    every one is bitwise the fallback there. The target passes
    `tune_max_num_tokens` = max_num_tokens = 32768 and `use_dp` False, so the
    buckets are 1..8192 and 32768, and 8193..32768 map onto 32768. Warming at
    32768 tokens wrote 15 cache entries in 10.6 s and left the result bitwise
    unchanged at 33 counts from 1 to 32768.

    The eight width slices. Rank r's call -- the same tokens, routing and
    scalars, its own slice of every expert -- returns the partial sum over
    intermediate columns [768r, 768r + 768). The FC2-input requantization
    runs in 16-element blocks under the layer's one global scale and 768 =
    48 x 16, so no block straddles a slice boundary: the eight partials,
    summed in fp32 and as a bf16 running sum in rank order, pass the test's
    gate against the whole-width (I 6144) reference, while one slice left out
    lands 90 rms ulp out and slice 0 run twice 128. The element figure at the
    slice is a max over a narrower contraction, so the test gates elements
    at 16 * sqrt(8) = 45.25 ulp there (16 at the whole width) and RMS at 2 as
    everywhere but the decode graph replays; its `_compare` has the derivation
    and the measurements. How
    the tp8 target sums the partials is its closing all-reduce's.

    Decode at tp8 gates few rows: a step of T tokens is T live rows of 4 x 768
    FC1 outputs, so one FC1 output that kernel and reference round to
    neighbouring e2m1 codes (or e4m3 block scales) at a rounding midpoint
    moves the relative RMS figure of the whole call. Over 5120 fresh decode
    draws a correct kernel exceeded the test's 2 ulp RMS gate on 21 (max
    7.05) with the element figure at or below 22.61; each of those differs
    from an fp64 model only at FC1 rounding midpoints (`_compare` has the
    protocol). The test gates the decode graph replays on the element figure
    alone.

    Operands at tp8, as a loader builds them for rank r from layer N's
    `experts.<e>.{w1,w2,w3}`, e = 0..127 (the slice is cut first, then laid
    out: the layout is a function of the slice's row count):
      gemm1_weights[e]       rows [768r, 768r + 768) of w3.weight, then
                             the same rows of w1.weight, gathered by
                             fc1_perm: [1536, 3072] uint8
      gemm1_weights_scale[e] the same rows of the two weight_scale tensors,
                             the same gather, then the 128x4 swizzle:
                             [1536, 384] float8_e4m3fn
      gemm2_weights[e]       w2.weight byte columns [384r, 384r + 384)
                             (inputs [768r, 768r + 768)), rows gathered by
                             blk32_perm: [6144, 384]
      gemm2_weights_scale[e] w2.weight_scale columns [48r, 48r + 48), the
                             same gather, then the swizzle: [6144, 48]
    1536 and 6144 rows are multiples of 128 and 48 scale columns a multiple
    of 4, so the swizzle pads nothing; the test proves
    `block_scale_interleave` equal to the torch recipe at both scale shapes.
    weight_scale_2 and input_scale are per expert and per layer, unsplit, so
    the three scalars are tep4's, for all 128 experts. The cells draw the
    slices at tep4's magnitudes and the same five layers' input scales; at
    layer 65's the FC2 input saturates as at tep4 (99.8-100 % of the block
    scales at 448 on two of four experts sampled, 1.4 and 16.6 % on the
    other two).

    Hard limits (sm_103, Mistral Large 4 tp8, 32768 tokens)
      gemm1_weights  local expert e starts at e2m1 element e * 2I * H
                     (9,437,184 per expert at I 768): expert 127 starts at
                     1,198,522,368 and the tensor holds 1,207,959,552
                     elements (603,979,776 bytes) -- below 2^31, computed,
                     not probed; a cell puts all 32768 tokens on each of
                     experts 124..127
      permuted rows  allocated (128 + (4T - 128) // tile) * tile = 163,584 at
                     T 32768, tile 256 -- read back as the per-slot rows of
                     `do_finalize=False` at 1, 16, 287, 288, 4128, 8193 and
                     32768 tokens, each the formula's; the bf16 gemm2 output is
                     1,005,060,096 elements, 2,010,120,192 bytes. A row's byte
                     offset passes 2^31 at row 174,763 -- not reached, 11,179
                     rows short (a uniform draw used rows to 147,645, the
                     top_experts skew to 131,071)
      gemm1 output   the NVFP4 FC2 input, 384 bytes a row: 62,816,256 bytes
                     at 163,584 rows -- below 2^31
      the rest       T * top_k = 131,072 expanded slots; gemm2_weights
                     603,979,776 e2m1 elements; hidden_states 100,663,296
                     bytes; output 201,326,592 elements -- all below 2^31. No
                     index passes through fp32: the tile choice truncates
                     T * top_k / local_num_experts, 1024 at the top
      T              certified 1..32768 at this geometry by the tp8 cells

    Hard limits (sm_103, Mistral Large 4 tep8, 16-wide windows, 32768 tokens)
      gemm1_weights  local expert e starts at e2m1 element e * 2I * H
                     (75,497,472 per expert); the last, 15, at 1,132,462,080
                     -- below 2^31, not reached. The tensor is 603,979,776
                     bytes
      permuted rows  allocated (16 + (4T - 16) // tile) * tile = 134,912 at
                     T 32768, tile 256 -- read back (`do_finalize=False`) at
                     20 counts from 1 to 32768 and with every slot in one
                     window; gemm2 output 134,912 x 6144 bf16 = 828,899,328
                     elements, 1,657,798,656 bytes; rows x 12,288 bytes
                     reaches 2^31 at 174,763 rows -- not reached. The
                     one-window draw's slots reach row 133,375
      gemm1 output   134,912 rows x 3072 bytes = 414,449,664 -- below 2^31
      the rest       as at tep4: 131,072 expanded slots, hidden_states
                     100,663,296 bytes, output 201,326,592 elements. No index
                     passes through fp32: the tile choice truncates T * top_k
                     / local_num_experts, 8192 at the top
      T              certified 1..32768 at this geometry by the tep8 cells

    Hard limits (sm_103, Mistral Large 4 dep8, 16-wide windows, 32768-token calls)
      gemm1_weights  local expert e starts at e2m1 element e * 2I * H
                     (75,497,472 per expert); the last, 15, at 1,132,462,080
                     -- below 2^31, not reached. The tensor is 603,979,776
                     bytes
      permuted rows  allocated (16 + (4T - 16) // tile) * tile = 134,912 at
                     T 32768, tile 256 -- read back (`do_finalize=False`) at
                     the 20 kernel-edge counts and on the one-window and
                     padding chunks; the bf16 FC2 output is 828,899,328
                     elements, 1,657,798,656 bytes. A row's byte offset
                     passes 2^31 at row 174,763 -- not reached: the slots'
                     rows, read back, reach 132,607 on the one-window chunk
                     and 131,072 on the padding chunk. A call of 65536 tokens
                     with every slot in one window would allocate 265,984
                     rows, past it (computed); the 32768-token chunk keeps
                     every call below it
      gemm1 output   134,912 rows x 3072 bytes = 414,449,664 -- below 2^31
      the rest       T * top_k = 131,072 expanded slots; hidden_states
                     100,663,296 bytes; output 201,326,592 elements -- all
                     below 2^31. No index passes through fp32: the tile choice
                     truncates T * top_k / local_num_experts, 8192 at the top
      T              certified 1..32768 per call at this geometry by the dep8
                     cells; a gathered set past 32768 is the caller's to cut

    Not certified at Mistral Large 4: `do_finalize=False` and `output=` as a
    caller's mode (the dep4, tep8 and dep8 tests read per-slot rows only to
    locate and count them), the in-kernel routing path,
    TLLM_USE_FINE_GRAINED_SYNC=1, more than 131072 tokens (more than 32768
    at 16-wide windows), gathered graph batches other than 4/8/16/32/64 at
    32-wide windows and 8..128 step 8 at 16-wide ones, tep8 and dep8 decode
    graphs on windows (16, 16) to (96, 16) (those on 0 and 112 are cells),
    graphs captured on a tuned tactic as cells (the graph cells capture the
    cold fallback; every tactic was run eager, and probed at dep8's 8..128 on
    window (0, 16): each replay bitwise warm eager, inside the gate), and the
    windows in separate processes -- dep4's four, tep8's and dep8's eight:
    the op holds no cross-rank state, each rank's call is this one at its own
    offset, and the windows' partials were summed in one process, in fp32 and
    as a bf16 running sum in rank order. How the dep4 and dep8 combines sum
    them is `comm/reducescatter`'s; how the tep8 target's does,
    `comm/cute_dsl_allreduce_add_rmsnorm`'s. At tp4 and tp8: more than 32768
    tokens, graph batches past 16, a width slice other than 1536 and 768
    (moe_tensor_parallel_size other than 4 and 8), and the four or eight width
    slices in as many processes -- they were summed in one process, as the
    windows were.
    """

    def __call__(
        self,
        routing_logits: Optional[torch.Tensor],
        routing_bias: Optional[torch.Tensor],
        hidden_states: torch.Tensor,
        hidden_states_scale: torch.Tensor,
        gemm1_weights: torch.Tensor,
        gemm1_weights_scale: torch.Tensor,
        gemm1_bias: Optional[torch.Tensor],
        gemm1_alpha: Optional[torch.Tensor],
        gemm1_beta: Optional[torch.Tensor],
        gemm1_clamp_limit: Optional[torch.Tensor],
        gemm2_weights: torch.Tensor,
        gemm2_weights_scale: torch.Tensor,
        gemm2_bias: Optional[torch.Tensor],
        output1_scale_scalar: torch.Tensor,
        output1_scale_gate_scalar: torch.Tensor,
        output2_scale_scalar: torch.Tensor,
        num_experts: int,
        top_k: int,
        n_group: Optional[int],
        topk_group: Optional[int],
        intermediate_size: int,
        local_expert_offset: int,
        local_num_experts: int,
        routed_scaling_factor: Optional[float],
        routing_method_type: int,
        do_finalize: bool,
        act_type: int = 0,
        topk_weights: Optional[torch.Tensor] = None,
        topk_ids: Optional[torch.Tensor] = None,
        output: Optional[torch.Tensor] = None,
        tune_max_num_tokens: int = 8192,
        use_dp: bool = False,
    ) -> list[torch.Tensor]:
        return torch.ops.trtllm.fp4_block_scale_moe_runner(
            routing_logits,
            routing_bias,
            hidden_states,
            hidden_states_scale,
            gemm1_weights,
            gemm1_weights_scale,
            gemm1_bias,
            gemm1_alpha,
            gemm1_beta,
            gemm1_clamp_limit,
            gemm2_weights,
            gemm2_weights_scale,
            gemm2_bias,
            output1_scale_scalar,
            output1_scale_gate_scalar,
            output2_scale_scalar,
            num_experts,
            top_k,
            n_group,
            topk_group,
            intermediate_size,
            local_expert_offset,
            local_num_experts,
            routed_scaling_factor,
            routing_method_type,
            do_finalize,
            act_type,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            output=output,
            tune_max_num_tokens=tune_max_num_tokens,
            use_dp=use_dp,
        )

    def reference(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        up: Operand,
        gate: Operand,
        down: Operand,
        fc2_global_scale: float,
        local_expert_offset: int = 0,
        local_num_experts: Optional[int] = None,
    ) -> torch.Tensor:
        """The layer, in plain torch over dequantized weights, fp32 throughout.

        The documented exception to mirroring: `__call__` takes the *prepared*
        operands -- packed e2m1 codes, shuffled rows, swizzled scales -- and a
        reference given those would be re-deriving the layout rather than the
        computation. So this takes the per-expert `(codes, scales)` pairs the
        layout was built from, which is where a target's checkpoint starts too,
        plus the routing this call was entered with.

        `up`, `gate` and `down` are each `(codes[E, M, K], scales[E, M, K/16])`
        in their unshuffled, unswizzled form, one e2m1 code (0..15) per weight
        element. `scales` may be fp32 with the expert's global weight scale
        (`weight_scale_2`) already multiplied in: the weight is dequantized as
        `code * scale`, so a checkpoint whose experts carry their own global
        scale is referenced exactly that way, and `hidden_states` is then the
        activation with its own global scale undone (`code * scale *
        input_scale`). `topk_ids` carries *global* ids;
        this rank contributes only for the ids its window owns, and every other
        slot contributes zero -- which is what makes an expert-parallel split
        summable.

        The FC1 activation is requantized to NVFP4 under `fc2_global_scale`
        before FC2, because the kernel does: the FC1 epilogue emits `g2 * act`
        as NVFP4 and FC2's alpha divides it back out. Skipping that step is a
        reference that disagrees with a correct kernel by ~23 ulp RMS, so it is
        not an optional refinement.
        """
        num_tokens, hidden = hidden_states.shape
        up_c, up_s = up
        gt_c, gt_s = gate
        dn_c, dn_s = down
        local_num_experts = up_c.shape[0] if local_num_experts is None else local_num_experts
        out = torch.zeros(num_tokens, hidden, dtype=torch.float32, device=hidden_states.device)
        for local_e in range(local_num_experts):
            tok, slot = (topk_ids == local_expert_offset + local_e).nonzero(as_tuple=True)
            if tok.numel() == 0:
                continue
            xe = hidden_states[tok]
            u = xe @ dequantize_nvfp4(up_c[local_e], up_s[local_e]).t()
            g = xe @ dequantize_nvfp4(gt_c[local_e], gt_s[local_e]).t()
            act = u * g * torch.sigmoid(g)
            act = quantize_nvfp4(act, fc2_global_scale) / fc2_global_scale
            y = act @ dequantize_nvfp4(dn_c[local_e], dn_s[local_e]).t()
            out.index_add_(0, tok, y * topk_weights[tok, slot].float().unsqueeze(1))
        return out

    def is_valid(
        self,
        routing_logits: Optional[torch.Tensor],
        routing_bias: Optional[torch.Tensor],
        hidden_states: torch.Tensor,
        hidden_states_scale: torch.Tensor,
        gemm1_weights: torch.Tensor,
        gemm1_weights_scale: torch.Tensor,
        gemm1_bias: Optional[torch.Tensor],
        gemm1_alpha: Optional[torch.Tensor],
        gemm1_beta: Optional[torch.Tensor],
        gemm1_clamp_limit: Optional[torch.Tensor],
        gemm2_weights: torch.Tensor,
        gemm2_weights_scale: torch.Tensor,
        gemm2_bias: Optional[torch.Tensor],
        output1_scale_scalar: torch.Tensor,
        output1_scale_gate_scalar: torch.Tensor,
        output2_scale_scalar: torch.Tensor,
        num_experts: int,
        top_k: int,
        n_group: Optional[int],
        topk_group: Optional[int],
        intermediate_size: int,
        local_expert_offset: int,
        local_num_experts: int,
        routed_scaling_factor: Optional[float],
        routing_method_type: int,
        do_finalize: bool,
        act_type: int = 0,
        topk_weights: Optional[torch.Tensor] = None,
        topk_ids: Optional[torch.Tensor] = None,
        output: Optional[torch.Tensor] = None,
        tune_max_num_tokens: int = 8192,
        use_dp: bool = False,
    ) -> None:
        """The two ways this op takes an input and answers it wrongly.

        Everything else a caller can get wrong here the op rejects loudly: a
        mismatched `intermediate_size`, a `top_k` outside `(0, num_experts)`, a
        hidden width that disagrees between the two weight operands. Those are
        not restated -- swapping their errors for ours would only make them
        worse.
        """
        # The kernel takes raw data pointers and assumes a dense row-major
        # layout for every tensor. A strided view is accepted without complaint
        # and silently reads -- or writes -- the wrong elements. Observed on
        # this machine for hidden_states, topk_weights and the weight operands.
        for name, tensor in (
            ("routing_logits", routing_logits),
            ("routing_bias", routing_bias),
            ("hidden_states", hidden_states),
            ("hidden_states_scale", hidden_states_scale),
            ("gemm1_weights", gemm1_weights),
            ("gemm1_weights_scale", gemm1_weights_scale),
            ("gemm1_bias", gemm1_bias),
            ("gemm1_alpha", gemm1_alpha),
            ("gemm1_beta", gemm1_beta),
            ("gemm1_clamp_limit", gemm1_clamp_limit),
            ("gemm2_weights", gemm2_weights),
            ("gemm2_weights_scale", gemm2_weights_scale),
            ("gemm2_bias", gemm2_bias),
            ("output1_scale_scalar", output1_scale_scalar),
            ("output1_scale_gate_scalar", output1_scale_gate_scalar),
            ("output2_scale_scalar", output2_scale_scalar),
            ("topk_weights", topk_weights),
            ("topk_ids", topk_ids),
            ("output", output),
        ):
            if tensor is not None:
                assert tensor.is_contiguous(), (
                    f"{name} must be contiguous; a strided view is read as if dense "
                    "and silently produces wrong results"
                )

        # With the block-scale layout this entry contracts -- row shuffle plus
        # 128x4 scale swizzle -- a hidden size that is only a multiple of 128
        # makes the kernel read the FC1 weight scales for the wrong blocks.
        # Measured wrong at hidden 384 / 640 / 896: no error, 300-3400 bf16 ulp.
        assert gemm1_weights.shape[-1] * 2 % 256 == 0, (
            "hidden size (gemm1_weights.shape[-1] * 2) must be a multiple of 256; "
            "the prepared block-scale layout is silently misread otherwise"
        )


fp4_block_scale_moe_runner = Fp4BlockScaleMoeRunner()
