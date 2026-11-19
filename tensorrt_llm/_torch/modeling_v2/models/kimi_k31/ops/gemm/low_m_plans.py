# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Measured low-M GEMM plans for the Kimi K3.1 SM 10.3 perf-target layouts (``ops/gemm/low_m_linear``).

Measured on GB300 and reviewed by hand: the comments in the tables record the rows decided beyond the rules below. The
plans are measured tables, not a runtime autotune: a choice fixes the kernel and with it the FP32 summation order, so
every rank and every run of a layout must take the same one, and the TEP16, TEP8 and DEP16 plans are held to the bits
of the path they replace (the general target's GEMMs at TEP16 and DEP16, TEP8's planless path).

One plan per generation layout: TEP32 (attention TP 32), TEP16 (attention TP 16), TEP8 (attention TP 8) and DEP16
(attention DP, every projection replicated). A plan maps a projection's ``(N, K)`` per rank to ``((M upper bound,
choice), ...)``; the choices are ``kernels/gemm/low_m_gemm``'s. Rows per step = requests x the DFlash2 verify width.

How the rows were chosen: every projection shape of the layout, at 1-64 (TEP32), 1-128 (TEP16) and 1-224 (DEP16)
rows (every 8 rows over 40-80 / 40-96 at TEP16 / DEP16), timed on GB300 for cuBLAS (``F.linear`` on the same,
possibly packed, weight) and every tactic of the direct, split-K and TGV kernels: cold weights (a different weight
copy per call, L2 flushed per replay), CUDA-graph replay, PDL on, each call timed back to back and behind a
producer kernel. A kernel took an M when the mean of the two times beat cuBLAS by at least 4% and its
producer-behind time was no more than 5% above cuBLAS's (its prologue is exposed there); a run of M kept its kernel
when it won by 5% on average. The M between two measured points takes the larger point's choice. Measured and left
on cuBLAS: the 7168 x 384 TEP16 shared-expert down projection (faster back to back, slower behind a producer) and
the TEP32 shared-expert down projection (K 192, which no low-M kernel tiles). The TEP32 / TEP16 ``o_proj`` is not
planned: those layouts run it with its all-reduce in one kernel.

Shapes (per rank): KDA ``[q|k|v|g]`` 4 x 12288 / tp x 7168; KDA / MLA ``o_proj`` 7168 x 12288 / tp; the packed
DSA hidden projection ``[q_a | kv_a | wk | weights_proj | g_proj]`` 2272 + 12288 / tp x 7168 (2272 without the
gate); MLA ``g_proj`` and the shared-expert ``gate_up`` 12288 / tp x 7168; the shared-expert ``down`` 7168 x
6144 / tp; MLA ``q_b`` 18432 / tp x 1536; the indexer ``wq_b`` 4096 x 1536; the MoE latent down 3584 x 7168 and
up 7168 x 3584 (replicated at every layout); the KDA gate pair ``[f_a | b]`` 128 + 96 / tp (padded to 8) x 7168.
"""

from __future__ import annotations

from typing import Optional

from tensorrt_llm._torch.modeling_v2.models.kimi_k31.kernels.gemm.low_m_gemm import LowMPlan


def _d(block_size: int, outputs_per_block: int) -> tuple:
    return ("direct", block_size, outputs_per_block)


def _s(mma_m: int, mma_n: int, split_k: int, ab_stages: int) -> tuple:
    return ("splitk", mma_m, mma_n, split_k, ab_stages)


def _sc(mma_m: int, mma_n: int, split_k: int, ab_stages: int) -> tuple:
    return ("splitkc", mma_m, mma_n, split_k, ab_stages)


def _t(cta_m: int, cta_n: int, ab_stages: int, use_2cta: bool) -> tuple:
    return ("tgv", cta_m, cta_n, ab_stages, use_2cta)


def _c(cta_n: int, ab_stages: int, *bounds: int) -> tuple:
    return ("tgvchunk", cta_n, ab_stages) + bounds


def _cm(cta_m: int, cta_n: int, ab_stages: int, *bounds: int) -> tuple:
    return ("tgvchunkm", cta_m, cta_n, ab_stages) + bounds


_DEFAULT = ("default",)

# Shapes every layout shares (replicated projections): the indexer wq_b, the MoE latent down, and the DFlash2
# drafter's block-convolution kernel projection (2 sides x 2 taps x 448 groups), packed [q_a | kv_a] and
# all-layer context latent projection (5 x 576).
_WQ_B = ((1, _d(64, 4)), (8, _t(64, 16, 12, True)), (16, _DEFAULT), (32, _t(64, 32, 11, True)))
_LATENT_DOWN = (
    (1, _d(128, 1)),
    (8, _s(64, 8, 2, 12)),
    (16, _s(64, 16, 2, 11)),
    (32, _s(64, 32, 2, 9)),
)
_D_CONVPROJ = (
    (1, _d(128, 1)),
    (8, _s(64, 8, 4, 10)),
    (16, _s(64, 16, 4, 10)),
    (32, _s(64, 32, 4, 8)),
)
_D_QA_KVA = _D_CONVPROJ
_D_CTX_KV = (
    (1, _d(128, 1)),
    (8, _s(128, 8, 4, 6)),
    (16, _s(64, 16, 2, 11)),
    (32, _s(64, 32, 2, 9)),
)

K31_LOW_M_PLAN_TEP32: LowMPlan = {
    # KDA [q | k | v | g]
    (1536, 7168): (
        (1, _d(128, 1)),
        (8, _s(64, 8, 4, 10)),
        (16, _s(64, 16, 4, 10)),
        (32, _s(64, 32, 4, 8)),
        (64, _t(64, 16, 12, True)),
    ),
    # packed DSA hidden projection with the output gate
    (2656, 7168): (
        (1, _d(128, 2)),
        (8, _s(128, 8, 4, 6)),
        (16, _s(64, 16, 2, 11)),
        (32, _s(64, 32, 2, 9)),
        (48, _DEFAULT),
        (64, _t(64, 32, 11, True)),
    ),
    # packed DSA hidden projection without the gate
    (2272, 7168): (
        (1, _d(128, 1)),
        (8, _s(64, 8, 4, 10)),
        (16, _s(64, 16, 4, 8)),
        (32, _s(64, 32, 2, 9)),
        (64, _t(64, 32, 11, True)),
    ),
    # MLA g_proj and the shared-expert gate_up. At 33-64 rows cuBLAS splits K in 4 even chunks with its reduce, which
    # the chunked split-K kernel gives bit for bit in one launch (the shared expert runs outside the MoE front above 16
    # rows). GB300, cold weights, us back to back / behind a producer: 5.6 / 5.5-5.6 -> 4.0-4.2 / 5.1-5.3.
    (384, 7168): ((1, _d(128, 1)), (5, _d(64, 1)), (32, _s(64, 8, 4, 10)), (64, _sc(64, 16, 4, 8))),
    # MLA q_b
    (576, 1536): (
        (1, _d(192, 2)),
        (5, _d(96, 1)),
        (8, _d(64, 1)),
        (24, _s(64, 8, 4, 10)),
        (32, _s(64, 8, 3, 10)),
        (64, _t(64, 16, 12, True)),
    ),
    # indexer wq_b
    (4096, 1536): _WQ_B,
    # MoE latent down / up
    (3584, 7168): _LATENT_DOWN,
    (7168, 3584): (
        (1, _d(64, 2)),
        (5, _t(64, 16, 12, True)),
        (8, _DEFAULT),
        (16, _t(64, 16, 12, True)),
    ),
    # drafter block-convolution kernel projection
    (1792, 7168): _D_CONVPROJ,
    # drafter packed [q_a | kv_a], and the dense layer-0 gate / up (2 x 33792 / 32): the same shape
    (2112, 7168): _D_QA_KVA,
    # drafter gate / up (2 x 14336 / 32)
    (896, 7168): (
        (1, _d(128, 1)),
        (8, _s(64, 8, 4, 10)),
        (16, _s(64, 16, 4, 10)),
        (32, _s(64, 16, 4, 8)),
    ),
    # drafter context latents of all 5 layers
    (2880, 7168): _D_CTX_KV,
}

# TEP16 up to 32 rows: every projection the target runs there gives the general target's bits (F.linear, or cublas_mm
# for the indexer wq_b and k31_fused_a_gemm for the latent down / up): a TGV tactic where cuBLAS does not split K, a
# split-K kernel where its split matches cuBLAS's, else cuBLAS itself. Not the packed DSA hidden projections, which
# the general target runs as separate GEMMs.
K31_LOW_M_PLAN_TEP16: LowMPlan = {
    # KDA [q | k | v | g]. Up to 32 rows cuBLAS splits K in 3 front-heavy chunks (19 / 19 / 18 K tiles), which the
    # chunked split-K kernel reproduces in one launch (cuBLAS: the split GEMM and a reduce kernel, neither with
    # programmatic dependent launch). GB300, cold weights, us per KDA layer: 1-8 rows 10.3-10.4 -> 8.3-8.5 (8-token
    # tile), 9-16 10.4-10.8 -> 8.9-9.1 (16-token tile); 17-32 rows 10.8-11.0 -> 11.2-11.5 (32-token tile) alone, but in
    # the model (TEP16, 120k context, 4 requests of width-5 verify = 20 rows) -0.2 to -0.5 ms per step. Above 32 rows
    # the rows are timed in the verify projection pair as the target issues it (beside the [f_a | b] -> f_b gate pair on
    # its stream) and are bitwise whatever they pick: every TGV tactic gives the same bits, and cuBLAS's kernel at
    # 113-192 rows has no split-K and gives them too (its 97-112 rows split K: they stay on cuBLAS). Above 128 rows the
    # 2-CTA tiles of 80 / 96 tokens split the rows into two even token tiles (FlashInfer's table stops at 64 / 128;
    # low_m_gemm compiles them). GB300, us per KDA layer: 33-48 rows 14.9-15.3 (TGV 64 x 16) -> 10.9 (64 x 32); 65-96
    # 14.6-14.8 (64 x 32) -> 13.0-13.2 (64 x 64); 113-128 14.9-15.0 (cuBLAS) -> 13.6-13.9; 129-160 15.0-15.3 (64 x 128)
    # -> 13.8-14.1 (64 x 80); 161-192 15.4-15.9 -> 15.1-15.6 (64 x 96).
    (3072, 7168): (
        (8, _sc(128, 8, 3, 6)),
        (16, _sc(128, 16, 3, 5)),
        (32, _sc(128, 32, 3, 4)),
        (64, _t(64, 32, 11, True)),
        (96, _t(64, 64, 9, True)),
        (112, _DEFAULT),
        (128, _t(64, 64, 9, True)),
        (160, _t(64, 80, 8, True)),
        (192, _t(64, 96, 8, True)),
    ),
    # packed DSA hidden projection with the output gate
    (3040, 7168): (
        (1, _d(128, 1)),
        (8, _s(128, 8, 4, 6)),
        (10, _s(64, 16, 2, 9)),
        (16, _s(64, 16, 2, 11)),
        (32, _s(64, 32, 2, 9)),
        (40, _t(64, 32, 11, True)),
        (48, _t(64, 16, 12, True)),
        (96, _t(64, 32, 11, True)),
    ),
    # packed DSA hidden projection without the gate
    (2272, 7168): (
        (1, _d(128, 1)),
        (5, _s(64, 16, 4, 8)),
        (8, _s(64, 8, 4, 10)),
        (16, _s(64, 16, 4, 8)),
        (24, _s(64, 32, 2, 9)),
        (32, _s(64, 32, 4, 8)),
        (48, _t(64, 32, 11, True)),
        (64, _t(64, 16, 12, True)),
        (128, _t(64, 32, 11, True)),
    ),
    # MLA g_proj and the shared-expert gate_up (above 32 rows the shared expert's are in K31_LOW_M_PLAN_TEP16_SHARED)
    (768, 7168): (
        (16, _s(64, 8, 4, 10)),
        (32, _s(64, 16, 4, 8)),
        (64, _DEFAULT),
        (80, _t(64, 16, 12, True)),
    ),
    # MLA q_b
    (1152, 1536): (
        (80, _t(64, 16, 12, True)),
        (96, _t(64, 16, 8, True)),
    ),
    # indexer wq_b
    (4096, 1536): ((8, _t(64, 16, 12, True)), (16, _DEFAULT), (32, _t(64, 32, 11, True))),
    # MoE latent down / up
    (3584, 7168): ((8, _s(64, 8, 2, 12)),) + _LATENT_DOWN[2:],
    (7168, 3584): (
        (1, _t(64, 16, 12, True)),
        (8, _s(64, 8, 1, 12)),
        (16, _t(64, 16, 12, True)),
        (64, _DEFAULT),
        (80, _t(64, 128, 7, True)),
    ),
    # drafter block-convolution kernel projection, and the drafter gate / up (2 x 14336 / 16): the same shape
    (1792, 7168): _D_CONVPROJ + ((80, _t(64, 16, 12, True)),),
    # drafter packed [q_a | kv_a]
    (2112, 7168): _D_QA_KVA + ((48, _t(64, 16, 12, True)), (80, _t(64, 32, 11, True))),
    # drafter context latents of all 5 layers
    (2880, 7168): _D_CTX_KV
    + ((48, _t(64, 16, 12, True)), (64, _DEFAULT), (80, _t(64, 32, 11, True))),
    # dense layer-0 gate / up (2 x 33792 / 16)
    (4224, 7168): (
        (5, _s(64, 8, 2, 12)),
        (8, _s(64, 8, 2, 10)),
        (10, _s(64, 16, 2, 11)),
        (16, _s(64, 16, 2, 9)),
        (32, _s(64, 32, 2, 9)),
    ),
    # KDA gate pair [f_a | b] (128 + 6 heads, padded to 136 rows), beside [q | k | v | g] on the verify's aux stream
    # (beta and the f_b input, so it gates the verify recurrence). At 65-128 rows cuBLAS splits K in 4 even chunks of
    # 14 K tiles (a split-K GEMM on 64 CTAs, then splitKreduce), which the chunked split-K reproduces in one launch of
    # 36-48 CTAs on wide token tiles; at 33-64 rows it splits K at 64-wide bounds (5 chunks), which no plan kernel
    # tiles. Timed in the verify projection pair with the [q | k | v | g] GEMM's CTAs placed first, as its PDL launch
    # behind the attention-residual mix places them in the model (GB300, cold weights, CUDA graph, us per KDA layer,
    # cuBLAS -> plan): 65-96 rows 14.5-14.7 -> 12.7-12.9 (24-token tiles), 113-128 rows 14.5 -> 13.1-13.2 (32-token
    # tiles); behind a producer kernel 22.3-22.4 -> 20.6-20.9. At 97-112 rows [q | k | v | g] runs on cuBLAS, which has
    # no PDL head start, so the pair runs in issue order (gates first), where the plan kernel is 0.5-0.8 slower: those
    # rows stay on cuBLAS.
    (136, 7168): (
        (64, _DEFAULT),
        (96, _sc(64, 24, 4, 8)),
        (112, _DEFAULT),
        (128, _sc(64, 32, 4, 8)),
    ),
}

# TEP16 shared expert: the TEP16 plan with its own gate_up rows. At 33-64 rows cuBLAS splits the 768 x 7168 gate_up's K
# (K tiles (19, 38) at 33-48 rows, (14, 28, 42) at 49-64) into a split-K GEMM and a reduce kernel; the chunked TGV sums
# K in the same chunks in one launch, so it gives the same bits. Alone it is 0.6-1.7 us slower than cuBLAS's two
# kernels, but on the shared expert's stream, beside the routed experts, the shorter chain of launches ends sooner
# (GB300 TEP16, 120k context, 5 verify rows, ms per decode step: 50 rows -0.21, 60 rows -0.24, 35 / 40 rows 0). MLA
# g_proj, the same shape on the attention path, keeps the TEP16 plan's cuBLAS rows.
K31_LOW_M_PLAN_TEP16_SHARED: LowMPlan = {
    **K31_LOW_M_PLAN_TEP16,
    (768, 7168): (
        (16, _s(64, 8, 4, 10)),
        (32, _s(64, 16, 4, 8)),
        (48, _c(16, 10, 19, 38)),
        (64, _c(16, 10, 14, 28, 42)),
        (80, _t(64, 16, 12, True)),
    ),
}

# TEP8 up to 32 rows: only the shapes where a kernel gives the bits of the op TEP8 runs without a plan (F.linear, or
# k31_fused_a_gemm for the MoE latent down) and beats it cold: a split-K kernel with cuBLAS's split (the chunked one
# where cuBLAS chunks K front-heavy). Every row 1-32 was checked for bits. GB300, cold weights, us per call, cuBLAS ->
# plan at 1-8 / 9-16 / 17-32 rows: MLA g_proj and shared gate_up 9.2-9.4 -> 6.7, 8.1-9.4 -> 7.0, 7.8 -> 7.1; packed DSA
# hidden 12.0-12.8 -> 11.0-11.2, 12.2-12.4 -> 11.3-11.4, 12.6 -> 11.7-11.9; MoE latent down 11.6-11.9 -> 10.6-10.7,
# 11.7-12.0 -> 10.8-11.0, 12.0-12.1 -> 11.3-11.5. MLA q_b and the indexer wq_b at 2-8 rows (TGV / split-K kernels that
# accumulate K as cuBLAS does there): q_b 5.2-5.8 -> 3.8-3.9, wq_b 5.4-6.1 -> 4.3-4.4. At 1 row and above 8 rows, and
# for the latent up and the shared down, cuBLAS stays (no bitwise kernel is faster). KDA [q|k|v|g] takes one TGV launch
# in place of cuBLAS's split-K GEMM and reduce, not bitwise (cuBLAS splits K): 16.6-17.3 -> 15.4-15.8 us at 5-30 rows.
# At 33-64 rows it takes the chunked TGV with cuBLAS's own split there (3 chunks of 19 / 19 / 18 K tiles), bitwise to
# cuBLAS: beside the gate pair as the verify issues them, 34.7-38.0 -> 19.2-20.0 us per layer (alone 18.9-21.7 ->
# 17.6-18.7). The DFlash2 drafter has its own plan at TEP8 (K31_LOW_M_DRAFT_PLAN_TEP8).
K31_LOW_M_PLAN_TEP8: LowMPlan = {
    # KDA [q | k | v | g] (49152 / 8 x 7168)
    (6144, 7168): (
        (8, _t(64, 16, 12, True)),
        (32, _t(64, 32, 11, True)),
        (48, _c(64, 7, 19, 38)),
        (64, _c(64, 6, 19, 38)),
    ),
    # MLA g_proj and the shared-expert gate_up (12288 / 8 x 7168): cuBLAS splits K in 3 front-heavy chunks
    (1536, 7168): ((8, _sc(64, 8, 3, 10)), (16, _sc(64, 16, 3, 10)), (32, _sc(64, 32, 3, 8))),
    # packed DSA hidden projection with the output gate (2272 + 12288 / 8)
    (3808, 7168): ((8, _s(64, 8, 2, 10)), (16, _s(64, 16, 2, 9)), (32, _s(64, 32, 2, 9))),
    # MoE latent down
    (3584, 7168): ((8, _s(64, 8, 2, 10)), (16, _s(64, 16, 2, 9)), (32, _s(64, 32, 2, 9))),
    # MLA q_b (12 heads x 192 per rank)
    (2304, 1536): (
        (1, _DEFAULT),
        (2, _t(64, 8, 12, False)),
        (7, _t(64, 8, 10, False)),
        (8, _t(64, 8, 8, False)),
    ),
    # DSA indexer wq_b (64 heads x 64, replicated)
    (4096, 1536): (
        (1, _DEFAULT),
        (2, _s(64, 8, 1, 12)),
        (3, _t(64, 8, 12, False)),
        (4, _s(64, 8, 1, 12)),
        (5, _t(64, 8, 12, False)),
        (6, _s(64, 8, 1, 12)),
        (8, _t(64, 8, 12, False)),
    ),
}

# The DFlash2 drafter's plan at TEP8. The drafter is not held to the planless bits, so every row takes the fastest
# kernel, including its gate / up and q_b, whose shapes the target also runs (the MoE latent down, the MLA q_b): the
# target's lookups read K31_LOW_M_PLAN_TEP8 only. GB300, cold weights, us per call at 5 / 10 / 20-30 rows, cuBLAS ->
# plan: kernel projection 8.1 -> 6.9 / 8.1 -> 7.2 / 8.9-9.0 -> 8.1-8.3; [q_a | kv_a] 8.7 -> 7.7 / 8.6 -> 7.8 /
# 9.7 -> 8.5-8.6; context latents 10.1 -> 9.2 / 10.2 -> 9.4 / 10.4 -> 9.9; gate / up 11.7 -> 10.6 / 11.8 -> 10.9 /
# 12.0-12.2 -> 11.4-11.5; q_b 5.3 -> 3.9 at 5 rows. Above 32 rows the other TEP layouts' rows (the same shapes).
K31_LOW_M_DRAFT_PLAN_TEP8: LowMPlan = {
    # block-convolution kernel projection (2 sides x 2 taps x 448 groups)
    (1792, 7168): (
        (2, _d(128, 2)),
        (5, _d(64, 2)),
        (16, _s(64, 16, 4, 8)),
        (32, _s(64, 32, 4, 8)),
        (80, _t(64, 16, 12, True)),
    ),
    # packed [q_a | kv_a]
    (2112, 7168): (
        (2, _d(128, 2)),
        (3, _d(64, 4)),
        (8, _s(64, 8, 4, 10)),
        (16, _s(64, 16, 4, 8)),
        (32, _s(64, 16, 2, 11)),
        (48, _t(64, 16, 12, True)),
        (80, _t(64, 32, 11, True)),
    ),
    # context latents of all 5 layers (the accepted tokens' rows)
    (2880, 7168): (
        (2, _d(128, 2)),
        (3, _d(64, 4)),
        (8, _sc(64, 8, 3, 10)),
        (16, _sc(64, 16, 3, 10)),
        (32, _sc(64, 32, 3, 8)),
        (48, _t(64, 16, 12, True)),
        (64, _DEFAULT),
        (80, _t(64, 32, 11, True)),
    ),
    # gate / up (2 x 14336 / 8)
    (3584, 7168): (
        (2, _d(128, 2)),
        (4, _d(64, 4)),
        (8, _s(64, 8, 2, 10)),
        (16, _s(64, 16, 2, 9)),
        (32, _s(64, 32, 2, 9)),
    ),
    # q_b (12 heads x 192 per rank): cuBLAS is the fastest from 9 rows
    (2304, 1536): ((3, _d(64, 1)), (5, _d(64, 2)), (8, _t(64, 8, 8, False))),
}

# DEP16 (attention DP, every projection replicated), held to the general target's bits: up to 80 rows, every projection
# the general target runs as one GEMM gives the general target's bits (F.linear, cublas_mm for the indexer wq_b,
# k31_fused_a_gemm for the latent down / up): a TGV tactic where cuBLAS does not split K, a split-K kernel with cuBLAS's
# split where it does, else cuBLAS itself. cuBLAS splits the o_proj K (96 tiles) in 4 even chunks up to 64 rows and in 2
# above, the KDA [q|k|v|g] K (56 tiles) in 7 even chunks at 1-4 rows and in 3 front-heavy ones at 5-8; the chunked TGV
# reproduces those (the pair-sharded KDA projection's half width splits K the same way). Not held: the packed DSA hidden
# projections (the general target runs them as separate GEMMs) and the drafter's projections. GB300, cold weights, us
# per call (cuBLAS in parentheses): o_proj at 8 / 32 / 64 / 80 rows 27.3 / 28.1 / 29.6 / 30.2 (29.3 / 30.4 / 34.4 /
# 35.4); KDA [q|k|v|g] at 1-4 rows 103-107 (111-115). The shared-expert gate_up (cuBLAS: 7 even chunks at 1-8 rows, 3
# front-heavy ones at 9-32) runs the chunked TGV on 128-row weight tiles, as fast as the 128-row TGV tile: 27.7-28.0
# us at 1-16 rows, 28.4-28.6 at 17-32 (cuBLAS 30.2-32.8).
K31_LOW_M_PLAN_DEP16: LowMPlan = {
    # KDA [q | k | v | g]: plan rows up to 32 rows also give the pair-sharded projection its kernels
    (49152, 7168): (
        (2, _c(32, 10, 8, 16, 24, 32, 40, 48)),
        (4, _c(16, 10, 8, 16, 24, 32, 40, 48)),
        (8, _sc(128, 8, 3, 6)),
        (16, _t(64, 16, 12, True)),
        (32, _t(128, 32, 6, True)),
    ),
    # KDA / MLA o_proj
    (7168, 12288): (
        (16, _c(16, 10, 24, 48, 72)),
        (32, _c(32, 10, 24, 48, 72)),
        (64, _c(64, 8, 24, 48, 72)),
        (80, _c(80, 8, 48)),
    ),
    # packed DSA hidden projection with the output gate
    (14560, 7168): ((1, _d(128, 2)), (16, _t(128, 16, 6, True))),
    # packed DSA hidden projection without the gate
    (2272, 7168): (
        (1, _d(128, 1)),
        (8, _s(128, 8, 4, 6)),
        (16, _s(64, 16, 4, 8)),
        (32, _s(64, 32, 2, 9)),
        (64, _t(64, 16, 12, True)),
        (128, _t(64, 32, 11, True)),
    ),
    # MLA g_proj and the shared-expert gate_up: the chunked TGV on 128-row weight tiles with cuBLAS's 7-way and
    # 3-way splits
    (12288, 7168): (
        (8, _cm(128, 16, 6, 8, 16, 24, 32, 40, 48)),
        (16, _cm(128, 16, 6, 19, 38)),
        (32, _cm(128, 32, 6, 19, 38)),
    ),
    # shared-expert down: cuBLAS has no split-K here; a TGV tile at 1 row
    (7168, 6144): ((1, _t(64, 16, 12, True)),),
    # MLA q_b
    (18432, 1536): ((1, _s(128, 16, 1, 6)),),
    # indexer wq_b
    (4096, 1536): (
        (1, _t(64, 8, 12, False)),
        (8, _t(64, 16, 12, True)),
        (16, _DEFAULT),
        (64, _t(64, 32, 11, True)),
    ),
    # MoE latent down (k31_fused_a_gemm's 2-way split from 1 row) / up
    (3584, 7168): ((8, _s(64, 8, 2, 12)),) + _LATENT_DOWN[2:],
    (7168, 3584): (
        (1, _s(64, 8, 1, 12)),
        (8, _DEFAULT),
        (16, _t(64, 16, 12, True)),
        (64, _DEFAULT),
        (80, _t(64, 128, 7, True)),
    ),
    # drafter block-convolution kernel projection, packed [q_a | kv_a], context latents of all 5 layers
    (1792, 7168): _D_CONVPROJ + ((80, _t(64, 16, 12, True)), (112, _t(64, 32, 11, True))),
    (2112, 7168): (
        (1, _d(128, 1)),
        (8, _s(64, 8, 4, 10)),
        (16, _s(64, 16, 4, 8)),
        (32, _s(64, 32, 4, 8)),
        (112, _t(64, 32, 11, True)),
    ),
    (2880, 7168): _D_CTX_KV + ((96, _t(64, 32, 11, True)), (112, _t(64, 64, 9, True))),
    # drafter down projection (all 14336 columns: attention DP replicates the drafter)
    (7168, 14336): (
        (16, _t(64, 16, 12, True)),
        (32, _t(64, 32, 11, True)),
        (64, _t(64, 64, 9, True)),
        (112, _t(64, 128, 7, True)),
    ),
    # dense layer-0 gate / up (2 x 33792, replicated)
    (67584, 7168): ((16, _DEFAULT), (32, _t(64, 32, 11, True)), (64, _t(64, 64, 9, True))),
}

#: The plans by layout name.
K31_LOW_M_PLANS: dict[str, LowMPlan] = {
    "tep32": K31_LOW_M_PLAN_TEP32,
    "tep16": K31_LOW_M_PLAN_TEP16,
    "tep16_shared": K31_LOW_M_PLAN_TEP16_SHARED,
    "tep8": K31_LOW_M_PLAN_TEP8,
    "tep8_draft": K31_LOW_M_DRAFT_PLAN_TEP8,
    "dep16": K31_LOW_M_PLAN_DEP16,
}


def k31_low_m_plan(tp_size: int, enable_attention_dp: bool) -> Optional[LowMPlan]:
    """The plan of a generation layout: attention DP (any size), attention TP 32, 16 or 8; None otherwise."""
    if enable_attention_dp:
        return K31_LOW_M_PLAN_DEP16
    return {32: K31_LOW_M_PLAN_TEP32, 16: K31_LOW_M_PLAN_TEP16, 8: K31_LOW_M_PLAN_TEP8}.get(tp_size)


def k31_low_m_shared_expert_plan(tp_size: int, enable_attention_dp: bool) -> Optional[LowMPlan]:
    """The shared expert's plan: its own at attention TP 16, elsewhere the layout's plan."""
    if not enable_attention_dp and tp_size == 16:
        return K31_LOW_M_PLAN_TEP16_SHARED
    return k31_low_m_plan(tp_size, enable_attention_dp)


def k31_low_m_draft_plan(tp_size: int, enable_attention_dp: bool) -> Optional[LowMPlan]:
    """The DFlash2 drafter's plan: its own at attention TP 8, elsewhere the layout's plan."""
    if not enable_attention_dp and tp_size == 8:
        return K31_LOW_M_DRAFT_PLAN_TEP8
    return k31_low_m_plan(tp_size, enable_attention_dp)
