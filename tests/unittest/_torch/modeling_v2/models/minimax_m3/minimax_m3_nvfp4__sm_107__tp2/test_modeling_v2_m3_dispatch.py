# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The tp2 (Rubin) decode target's fast-path table (``modeling.py``: the row ranges and the selection functions): its
row ranges against the catalog cells that certify them, and the selection for every decode row count.

Every range's covered shapes are listed in ``ENABLED``; a range not listed covers nothing.

CPU only: nothing here launches a kernel.
"""

import inspect
from types import SimpleNamespace

import pytest

from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_nvfp4__sm_107__tp2 import (  # noqa: E501
    modeling as dispatch,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_nvfp4__sm_107__tp2.ops import (  # noqa: E501
    m3_draft_step as draft_step_kernel,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_nvfp4__sm_107__tp2.ops import (  # noqa: E501
    triton_sparse_decode as tsd,
)

ROWS = range(1, dispatch.TINY_MAX_TOKENS + 2)  # every decode row count and the first one past it


@pytest.fixture(autouse=True)
def _policy():
    """Every test starts outside require mode with nothing logged, and leaves it so."""
    dispatch.set_require(False)
    dispatch.reset_logged()
    yield
    dispatch.set_require(False)
    dispatch.reset_logged()


@pytest.mark.parametrize("fast_path", dispatch.FAST_PATHS, ids=lambda r: r.path)
def test_every_covered_shape_is_a_certified_cell(fast_path) -> None:
    uncertified = fast_path.keys - fast_path.certified()
    assert not uncertified, (
        f"{fast_path.path} covers {sorted(uncertified, key=str)} but "
        f"{type(fast_path.entry).__name__}.CELLS certifies none of them: add a cell or narrow the range"
    )


def test_the_fast_path_names_are_unique() -> None:
    names = [r.path for r in dispatch.FAST_PATHS]
    assert len(names) == len(set(names))


def test_the_decode_target_takes_the_gen_workers_batch() -> None:
    """Up to eight requests verifying 1 + 3 drafts (32 rows) take the decode target; a wider step is the prefill
    target's. Every fast path still keeps its own rows (the batch-1/2 target's TINY_MAX_TOKENS unless raised)."""
    assert dispatch.TINY_MAX_TOKENS == 8
    rows = range(1, dispatch.DECODE_MAX_TOKENS + 2)
    if dispatch.B8_DECODE:
        assert dispatch.DECODE_MAX_TOKENS == 32
    elif dispatch.B4_DECODE:
        assert dispatch.DECODE_MAX_TOKENS == 16
    else:
        assert dispatch.DECODE_MAX_TOKENS == dispatch.TINY_MAX_TOKENS
    expected = [True] * dispatch.DECODE_MAX_TOKENS + [False]
    assert [dispatch.is_decode_step(n) for n in rows] == expected


# ------------------------------------------------------------------ the cells turned on


def test_the_moe_engine_takes_every_decode_step() -> None:
    """The MoE lane's m3_moe engine (TP2 width 1536, shared expert folded in) at 1..8 rows, unless killed; its sum
    pushes into the multicast boundary all-reduce at the engine's rows while that all-reduce is on."""
    expected = set() if dispatch.KILL_MOE_ENGINE else set(range(1, 9))
    assert set(dispatch.MOE_ENGINE.keys) == expected
    for rows in ROWS:
        if rows in expected:
            want = dispatch.MOE_ENGINE.path
        elif dispatch.MC_FINALIZE.covers(rows):  # B3: past the engine, the multicast finalize
            want = dispatch.MC_FINALIZE.path
        else:
            want = None
        assert dispatch.moe_path(rows) == want, rows
    push_rows = dispatch.MOE_ENGINE_MAX_TOKENS if dispatch.MC_AR_TP2 else 0
    assert dispatch.MOE_PUSH_MAX_TOKENS == push_rows
    assert set(dispatch.MOE_PUSH.keys) == set(range(1, push_rows + 1))


# ------------------------------------------------------------------ on: the sparse layers' block selection


def test_the_index_scorer_takes_one_to_forty_eight_requests_with_their_schedules() -> None:
    # W-IDX1 (9..48 requests) is parked: the default covers 1..8; its schedules below stay certified for the knob.
    assert dispatch.INDEX_SCORE.keys == frozenset(range(1, 9))
    assert dispatch.index_score_schedule(1) == (1536, True)
    assert dispatch.index_score_schedule(2) == (768, True)
    assert dispatch.index_score_schedule(3) == (424, True)
    assert dispatch.index_score_schedule(4) == (318, True)
    for requests in range(5, 9):
        assert dispatch.index_score_schedule(requests) == (1272 // requests, True)
    # W-IDX1: 9..48 requests in 7 split buckets, about 1272 CTAs in total, early issue
    assert len({dispatch.index_score_schedule(r)[0] for r in range(9, 49)}) == 7
    for requests in range(9, 49):
        split, early = dispatch.index_score_schedule(requests)
        assert early and requests * split <= 1272 + split
    # every covered request count has a split (the graph captures it)
    assert set(dispatch.INDEX_SCORE.keys) <= set(dispatch.INDEX_SCORE_SPLIT_K)


def test_the_long_select_takes_every_key_width_up_to_its_maximum() -> None:
    assert dispatch.LONG_SELECT.keys == frozenset(dispatch.LONG_SELECT_KEY_WIDTHS)
    assert (dispatch.LONG_SELECT_MIN_BLOCKS, dispatch.LONG_SELECT_MAX_BLOCKS) == (129, 9216)


def test_the_384_thread_select_takes_every_key_width_with_its_launch_key() -> None:
    assert dispatch.SELECT_LONG_DSL is True
    assert dispatch.LONG_SELECT_DSL.keys == frozenset(dispatch.LONG_SELECT_DSL_KEY_WIDTHS)
    keys = dispatch.long_select_dsl_keys()
    assert [key[1:] for key in keys] == [(False, dispatch.SELECT_LONG_TRIGGER_PHASE)]


@pytest.mark.parametrize("num_blocks", [129, 1294, 5888, 9216])
def test_a_long_select_decline_inside_its_range_is_a_miss(num_blocks) -> None:
    """The kernel declining operands its cells certify must not fall back silently under require."""
    assert dispatch.long_select_covers(num_blocks)
    # Outside require: logged once, the in-tree selector runs.
    dispatch.long_select_fallback(num_blocks, 8)
    dispatch.set_require(True)
    with pytest.raises(dispatch.FastPathUnavailable, match="long_select"):
        dispatch.long_select_fallback(num_blocks, 8)


@pytest.mark.parametrize("num_blocks", [1, 16, 128, 9217, 20000])
def test_rows_outside_the_long_select_range_are_uncovered_even_under_require(num_blocks) -> None:
    """Short rows and rows past LONG_SELECT_MAX_BLOCKS take the in-tree selector in every mode (long contexts must not
    fail under require)."""
    assert not dispatch.long_select_covers(num_blocks)
    dispatch.set_require(True)
    dispatch.long_select_fallback(num_blocks, 8)


def test_taken_logs_without_raising_under_require() -> None:
    dispatch.set_require(True)
    dispatch.taken(dispatch.INDEX_SCORE.path, 1)
    dispatch.taken(dispatch.LONG_SELECT.path, 8)


def test_an_index_score_cell_of_another_schedule_certifies_nothing() -> None:
    spec = dict(seq_lens=(165521,), qlen=4, heads=2, split_k=512, early=True)
    assert tuple(dispatch.INDEX_SCORE.certifies(spec)) == ()
    assert tuple(dispatch.INDEX_SCORE.certifies(dict(spec, split_k=1536, heads=1))) == ()
    assert tuple(dispatch.INDEX_SCORE.certifies(dict(spec, split_k=1536))) == (1,)


#: Every TP2 MXFP8 projection (N, K): sparse-layer QKV + index, dense QKV, o_proj, shared gate_up / down, dense
#: gate_up / down.
TP2_PROJECTIONS = (
    (4992, 6144),
    (4608, 6144),
    (6144, 4096),
    (3072, 6144),
    (6144, 1536),
    (12288, 6144),
    (6144, 6144),
)
_DECODE_ROWS = range(1, dispatch.TINY_MAX_TOKENS + 1)
# Requests of a fused spec-tail step (B4 spec tail: up to the target's steps).
_SPEC_REQS = range(1, dispatch.SPEC_TAIL_MAX_REQUESTS + 1)
# W2: the fused verify's requests (up to WIDE_MAX_TOKENS rows, unless WIDE_SPEC_TAIL is killed).
_VERIFY_REQS = range(1, dispatch.WIDE_VERIFY_MAX_REQUESTS + 1)
#: B2: the GEMVs' rows at batch 4 (1..16 unless MX_GEMV_B4 is killed). The bf16 GEMV declines dense gate_up / down at
#: 9..16 rows (24 k-tiles per split at MMA width 16, past its in-kernel quantize limit of 16).
_GEMV_HI = (
    (dispatch.DECODE_MAX_TOKENS if dispatch.B8_MX_GEMV_32 else dispatch.B4_MAX_TOKENS)
    if dispatch.MX_GEMV_B4
    else dispatch.TINY_MAX_TOKENS
)
_BF16_WIDE_DECLINES = frozenset({(12288, 6144), (6144, 6144)})


#: Batch 8: at 17..32 rows (MMA width 32) the bf16 GEMV's in-kernel quantize serves only shared down (6 k-tiles per
#: split); the other projections keep their 16-row reach.
_BF16_B8_SERVES = frozenset({(6144, 1536)})


#: Step B (GEMM-AR 794929): dense gate_up / down and o_proj keep the 16-row GEMV even with B8_MX_GEMV_32.
_GEMV_B4_ROWS_ONLY = frozenset({(12288, 6144), (6144, 6144), (6144, 4096)})


def _mx_hi(nk) -> int:
    if nk in _GEMV_B4_ROWS_ONLY:
        return min(_GEMV_HI, 16)
    # W-G1: the four projections it raises reach their served rows up to 48 (dispatch._GEMV_W48_ROWS).
    return max(dispatch._GEMV_W48_ROWS.get(nk, ()), default=_GEMV_HI)


def _bf16_hi(nk) -> int:
    if nk in _BF16_WIDE_DECLINES:
        return dispatch.TINY_MAX_TOKENS
    if _GEMV_HI > 16 and nk not in _BF16_B8_SERVES:
        return 16
    return _GEMV_HI


#: Fast paths a lane has turned on, each with the shapes it covers (the rest of the skeleton stays off).
ENABLED = {
    # GEMM-AR (agg G-d): the tensor-core router of an unfused MoE at 9..48 rows, as the agg canon runs it.
    "m3_router_tc": frozenset(
        range(dispatch.ROUTER_TC_GEN_MIN_ROWS, dispatch.ROUTER_TC_GEN_MAX_ROWS + 1)
    ),
    # SpecTail: sampled (temperature + top-p) verify of 1-2 requests at draft length 3 (TP2 shard kernel).
    "m3_sampled_accept": frozenset(
        (3, n) for n in range(1, dispatch.SAMPLED_VERIFY_MAX_REQUESTS + 1)
    ),
    # SpecTail: greedy verify (shard argmax + accept) and the fused draft loop of 1-2 requests at draft length 3,
    # and that loop's draft-input rows (1 / 2 later-pass rows, 4 / 8 first-pass rows).
    "m3_shard_accept": frozenset((3, n) for n in _VERIFY_REQS),
    "m3_draft_step": frozenset((3, n) for n in _SPEC_REQS),
    "m3_draft_input": frozenset(r for n in _SPEC_REQS for r in (n, 4 * n)),
    # SpecTail: the drafter's BF16 linears on FlashInfer's low-M kernels (TP8 port) at the certified b1 rows.
    "m3_draft_lowm": frozenset(
        {
            ((4608, 12288), 1),
            ((4608, 12288), 4),
            ((6144, 4096), 1),
            ((6144, 4096), 4),
            ((18432, 6144), 1),
            ((6144, 9216), 1),
            ((6144, 9216), 4),
            ((6144, 18432), 1),
        }
        # Batch 4: the 16-row draft pass 0 (o, down, fc), unless DRAFT_LOWM_MAX_ROWS is set back to 4.
        | (
            {((6144, 4096), 16), ((6144, 9216), 16), ((6144, 18432), 16)}
            if dispatch.DRAFT_LOWM_MAX_ROWS >= 16
            else set()
        )
        # Batch 8: the 8-row passes' o and the 32-row pass 0's down, unless DRAFT_LOWM_MAX_ROWS is set lower.
        | ({((6144, 4096), 8)} if dispatch.DRAFT_LOWM_MAX_ROWS >= 8 else set())
        | ({((6144, 9216), 32)} if dispatch.DRAFT_LOWM_MAX_ROWS >= 32 else set())
    ),
    # SpecTail: the fused first drafter inputs of 1-2 fused-verify requests, the capture norm and the chunk norm at
    # their 4 / 8 verify rows (own ranges and kills since the TP2 fast-path audit).
    "m3_fused_first_inputs": frozenset(_SPEC_REQS),
    "m3_capture_norm": frozenset(4 * n for n in _SPEC_REQS),
    "m3_chunk_norm": frozenset(4 * n for n in _SPEC_REQS),
    # MSA: the long-context sparse decode with the MXFP8 o_proj hand-off, every decode step, and the sparse layers'
    # NVFP4 QKV + indexer producer at the TP2 heads.
    "m3_sparse_decode_lc": frozenset({"mxfp8"}),
    "m3_producer": frozenset({"nvfp4_tp2"}),
    "sparse_decode_producer": frozenset({"per_row"}),
    # MoE: the m3_moe engine (TP2 width 1536, shared expert folded in) at 1..8 rows, unless killed.
    "m3_moe_engine": frozenset() if dispatch.KILL_MOE_ENGINE else frozenset(range(1, 9)),
    # Indexer: the index scorer at 1-8 requests, the long-row select at every key width (129..9216 blocks).
    "m3_index_score": frozenset(range(1, dispatch.INDEX_SCORE_MAX_REQUESTS + 1)),
    "long_select": frozenset({2, 4, 9}),
    "long_select_dsl": frozenset({4, 11, 24}),
    # LongAttn: the dense layers' full-context decode of 1-2 requests, 1 + 3 verify rows or one draft-pass row.
    "m3_fullctx_decode": frozenset({(4, 1), (1, 1), (4, 2), (1, 2)}),
    # MoE (uGPU): the die-placed target head at 4 / 8 / 12 / 16 verify rows, unless killed.
    "m3_ugpu_head": frozenset() if dispatch.KILL_UGPU_HEAD else frozenset({4, 8, 12, 16}),
    # GEMM/AR (+ B2): the decode MXFP8 GEMVs at every TP2 projection and decode row, on an MXFP8 and a bf16
    # activation, raised to 16 rows where the kernel serves them.
    "m3_mx_gemv": frozenset((nk, m) for nk in TP2_PROJECTIONS for m in range(1, _mx_hi(nk) + 1))
    if dispatch.MX_GEMV_TP2
    else frozenset(),
    "m3_mx_gemv_bf16": frozenset(
        (nk, m) for nk in TP2_PROJECTIONS for m in range(1, _bf16_hi(nk) + 1)
    )
    if dispatch.MX_GEMV_TP2
    else frozenset(),
    # GEMM/AR: the multicast boundary all-reduce, the o_proj's push into it, the fused embedding + first RMSNorm.
    # (+ B3: 9..16 rows on the 16-token bucket, unless B4_MC_AR is killed.)
    "mc_allreduce_norm": frozenset(range(1, dispatch.PLAN_MC_AR_MAX_TOKENS + 1))
    if dispatch.MC_AR_TP2
    else frozenset(),
    # (+ B6: 9..16 rows into the 16-token bucket, unless B4_ATTN_PUSH is killed; the o_proj GEMV stops at 16.)
    "attn_push": frozenset(
        range(1, min(dispatch.PLAN_MC_AR_MAX_TOKENS, dispatch.B4_MAX_TOKENS) + 1)
        if dispatch.B4_ATTN_PUSH
        else _DECODE_ROWS
    )
    if (dispatch.MC_AR_TP2 and dispatch.ATTN_PUSH_TP2)
    else frozenset(),
    "dense_push": frozenset(_DECODE_ROWS)
    if (dispatch.MC_AR_TP2 and dispatch.DENSE_PUSH_TP2)
    else frozenset(),
    # B3: the MoE's multicast finalize + all-reduce + norm at 9..16 rows (past the engine), unless killed.
    "mc_finalize_allreduce_norm": frozenset(
        range(dispatch.MC_FINALIZE_MIN_TOKENS, dispatch.PLAN_MC_AR_MAX_TOKENS + 1)
    )
    if dispatch.B4_MC_FINALIZE
    else frozenset(),
    # MoE: the engine's push into the multicast boundary all-reduce, at the engine's rows.
    "moe_push": frozenset(range(1, dispatch.MOE_PUSH_MAX_TOKENS + 1)),
    # (+ B5: 9..16 rows, unless B4_EMBED_FIRSTNORM is killed; 17..32 keep the stock layer 0.)
    "embed_firstnorm": frozenset(
        range(1, dispatch.B4_MAX_TOKENS + 1) if dispatch.B4_EMBED_FIRSTNORM else _DECODE_ROWS
    )
    if dispatch.EMBED_FIRSTNORM_TP2
    else frozenset(),
    # SpecTail: the fused draft loop's all-reduce boundaries (1..8 rows) on the multicast all-reduce (opts 16).
    "m3_draft_mc_allreduce_norm": frozenset(range(1, dispatch.DRAFT_MC_AR_MAX_ROWS + 1)),
    # LongAttn: the Eagle3 drafter layer's generation passes of 1-2 requests on the same recipes.
    "m3_drafter_fullctx": frozenset({(4, 1), (1, 1), (4, 2), (1, 2)}),
}


@pytest.mark.parametrize("fast_path", dispatch.FAST_PATHS, ids=lambda r: r.path)
def test_skeleton_no_fast_path_covers_a_shape(fast_path) -> None:
    expected = ENABLED.get(fast_path.path, frozenset())
    assert fast_path.keys == expected, f"{fast_path.path} covers {sorted(fast_path.keys, key=str)}"


def test_the_sparse_decode_writes_the_mxfp8_handoff() -> None:
    """The sparse decode (LC2, the Triton merge) writes the MXFP8 o_proj hand-off; the range keys the target's
    SPARSE_DECODE_LC / SPARSE_DECODE_PRODUCER cover are the kernel module's."""
    assert tsd.MERGE_MXFP8_HANDOFF is True
    assert dispatch.SPARSE_DECODE_LC.keys is tsd.LC_KEYS
    assert dispatch.SPARSE_DECODE_PRODUCER.keys is tsd.PRODUCER_KEYS


def test_sparse_decode_lc_covers_the_decode_steps_with_the_handoff() -> None:
    """Every decode step (at most DECODE_MAX_TOKENS rows) writes the MXFP8 o_proj hand-off, so the long-context
    sparse decode needs only its "mxfp8" variant, on the stock log-domain recurrence and without pre-wait staging."""
    assert dispatch.SPARSE_DECODE_LC.keys == frozenset({"mxfp8"})
    assert tsd.MERGE_MXFP8_MIN_TOKENS <= 1 and dispatch.DECODE_MAX_TOKENS <= tsd.MergeMxfp8.MAX_ROWS
    assert tsd.LC_PREWAIT is False


def test_the_dense_layers_take_the_fullctx_decode_at_the_gen_workers_shapes() -> None:
    """One or two requests, 1 + 3 verify rows or one draft-pass row, each at its own pinned recipe."""
    assert dispatch.FULLCTX_DECODE_ENABLED
    assert dispatch.FULLCTX_DECODE.keys == {(4, 1), (1, 1), (4, 2), (1, 2)}
    assert set(dispatch.FULLCTX_DECODE_RECIPES) == set(dispatch.FULLCTX_DECODE.keys)
    for recipe in dispatch.FULLCTX_DECODE_RECIPES.values():
        assert 2 <= recipe["splits"] <= 128  # the standalone reducer's range


def test_the_fullctx_entry_cells_run_at_the_forward_plan_capacity() -> None:
    from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_nvfp4__sm_107__tp2.ops.attention import (  # noqa: E501
        m3_fullctx_decode as entry,
    )

    assert entry.MAX_KV_LEN == dispatch.FULLCTX_DECODE_MAX_KV_LEN


def test_the_fullctx_capacity_covers_the_agentx_cache_bound() -> None:
    """The forward compares the cache's length bound (max_seq_len plus the speculative rows it reserves) with the plan
    capacity; at the TP2 AgentX max_seq_len of 1 << 20 with up to 8 draft rows the dense layers must still be taken."""
    assert dispatch.FULLCTX_DECODE_MAX_KV_LEN >= (1 << 20) + 8


def test_the_drafter_takes_the_fullctx_decode_at_the_same_shapes() -> None:
    assert dispatch.DRAFTER_FULLCTX_ENABLED
    assert dispatch.DRAFTER_FULLCTX.keys == dispatch.FULLCTX_DECODE.keys


@pytest.mark.parametrize("rows", range(1, 50))
def test_the_tensor_core_router_takes_9_to_48_rows(rows) -> None:
    """agg G-d: the tensor-core router of an unfused MoE at 9..48 rows, as the agg canon runs it; 1..8 rows stay on
    the MoE engine's fused router (the batch-1/2 target's) and 49+ on the fp32 router."""
    assert dispatch.ROUTER_TC_GEN.covers(rows) == (9 <= rows <= 48)


# ------------------------------------------------------------------ landed: the decode MXFP8 GEMVs (GEMM/AR lane)


def test_the_gemv_rows_follow_the_kill_constant() -> None:
    expected = set(TP2_PROJECTIONS) if dispatch.MX_GEMV_TP2 else set()
    assert set(dispatch.MX_GEMV_ROWS) == expected
    assert set(dispatch.MX_GEMV_BF16_MAX_TOKENS) == expected


@pytest.mark.parametrize("rows", range(1, dispatch.TINY_MAX_TOKENS + 1))
@pytest.mark.parametrize("nk", TP2_PROJECTIONS, ids=str)
def test_every_projection_takes_the_gemv_at_every_decode_row(nk, rows) -> None:
    """MXFP8 activation (a producer quantized it) -> the GEMV; bf16 -> the GEMV quantizing in its prologue (staged:
    the stock decode-graph GEMMs while MX_GEMV_TP2 is off)."""
    on = dispatch.MX_GEMV_TP2
    assert dispatch.linear_path(*nk, rows, prequant=True, low_m_gemm=True) == (
        dispatch.MX_GEMV_PATH if on else dispatch.PREQUANT_GEMM_PATH
    )
    assert dispatch.linear_path(*nk, rows, prequant=False, low_m_gemm=True) == (
        dispatch.MX_GEMV_BF16_PATH if on else dispatch.GENERIC_GEMM_PATH
    )


@pytest.mark.parametrize("nk", TP2_PROJECTIONS, ids=str)
def test_wider_than_the_decode_step_is_the_generic_gemm(nk) -> None:
    # Past the decode step and past every GEMV row it serves (W-G1 raises four projections to 48).
    rows = max(dispatch.DECODE_MAX_TOKENS, _mx_hi(nk)) + 1
    assert (
        dispatch.linear_path(*nk, rows, prequant=True, low_m_gemm=True)
        == dispatch.PREQUANT_GEMM_PATH
    )
    assert (
        dispatch.linear_path(*nk, rows, prequant=False, low_m_gemm=True)
        == dispatch.GENERIC_GEMM_PATH
    )


def test_an_unknown_projection_is_the_generic_gemm() -> None:
    assert (
        dispatch.linear_path(256, 512, 1, prequant=True, low_m_gemm=True)
        == dispatch.PREQUANT_GEMM_PATH
    )
    assert (
        dispatch.linear_path(256, 512, 1, prequant=False, low_m_gemm=True)
        == dispatch.GENERIC_GEMM_PATH
    )


def test_taken_logs_once_per_path_and_rows(monkeypatch) -> None:
    lines = []
    monkeypatch.setattr(dispatch.logger, "info", lines.append)
    for _ in range(3):
        dispatch.taken(dispatch.MX_GEMV_PATH, 4, "[6144, 4096] mxfp8 in")
    dispatch.taken(dispatch.MX_GEMV_PATH, 8, "[6144, 4096] mxfp8 in")
    assert len(lines) == 2 and all("m3_mx_gemv taken at" in line for line in lines)


def test_skeleton_the_dense_decode_split_is_the_built_in_one() -> None:
    assert dict(dispatch.DENSE_DECODE_MAX_SEQ_LEN_BY_BATCH) == {}


def test_skeleton_speculation_runs_the_in_tree_loop() -> None:
    # SpecTail: fused greedy / sampled verify and the fused draft loop up to the target's steps (B4 spec tail; the
    # batch-1/2 rows with B4_SPEC_TAIL off; BF16 head; no coded heads).
    reqs = dispatch.SPEC_TAIL_MAX_TOKENS // (dispatch.SHARD_ACCEPT_DRAFT_LEN + 1)
    assert dispatch.SPEC_TAIL_MAX_TOKENS == (
        dispatch.DECODE_MAX_TOKENS if dispatch.B4_SPEC_TAIL else dispatch.TINY_MAX_TOKENS
    )
    # W2: the fused verify alone widens to WIDE_MAX_TOKENS rows (WIDE_SPEC_TAIL off keeps SPEC_TAIL_MAX_TOKENS).
    verify = (
        dispatch.WIDE_MAX_TOKENS if dispatch.WIDE_SPEC_TAIL else dispatch.SPEC_TAIL_MAX_TOKENS
    ) // (dispatch.SHARD_ACCEPT_DRAFT_LEN + 1)
    assert dispatch.WIDE_VERIFY_MAX_REQUESTS == verify
    assert dispatch.SHARD_ACCEPT_MAX_REQUESTS == verify
    # The sampled verify stays at SPEC_TAIL_MAX_TOKENS rows (past it, the in-tree sampler).
    assert dispatch.SAMPLED_VERIFY_MAX_REQUESTS == reqs
    assert dispatch.DRAFT_STEP_MAX_REQUESTS == min(reqs, 8)
    assert dispatch.FC_EARLY_MAX_ROWS == 0
    assert not draft_step_kernel.HEAD_CODED


def test_a_miss_raises_only_under_require() -> None:
    dispatch.missed("some_path", 1, "a precondition")
    dispatch.set_require(True)
    with pytest.raises(dispatch.FastPathUnavailable, match="some_path"):
        dispatch.missed("some_path", 1, "a precondition")


def test_skeleton_first_drafter_inputs_are_in_tree() -> None:
    # SpecTail turned the sampled verify on (SAMPLED_VERIFY_MAX_REQUESTS 2, test_skeleton_speculation_...).
    assert dispatch.FIRST_DRAFTER_INPUTS_MAX_REQUESTS == 0


@pytest.mark.parametrize(
    "rows, capture_normed, taken",
    [
        (4, False, True),
        (8, False, True),
        (8, True, True),
        # 3 / 4 requests: the B4 spec tail's rows (in-tree with B4_SPEC_TAIL off).
        (12, False, dispatch.B4_SPEC_TAIL),
        (12, True, dispatch.B4_SPEC_TAIL),
        (16, True, dispatch.B4_SPEC_TAIL),
    ],
)
def test_the_fused_first_inputs_take_only_their_covered_rows(rows, capture_normed, taken) -> None:
    """The fused first drafter inputs (and the fc fork) take a fused-verify step only at the requests
    FUSED_FIRST_INPUTS covers and, unless the captures applied the fc_norm, the rows CHUNK_NORM covers; anything else
    runs the in-tree first drafter inputs."""

    worker = object.__new__(dispatch.M3Eagle3Worker)
    worker._m3_capture_norm = (None, 1e-6) if capture_normed else None
    meta = SimpleNamespace(runtime_draft_len=dispatch.SHARD_ACCEPT_DRAFT_LEN)
    assert worker._m3_first_inputs_covered(rows, meta) is taken


# ------------------------------------------------------- landed: the multicast boundary all-reduce (GEMM/AR lane)


def test_both_boundaries_take_the_multicast_all_reduce() -> None:
    """One 8-token bucket for the whole decode step; the MoE boundary publishes the next QKV's MXFP8 rows (killed:
    every boundary on the built-in all-reduce while MC_AR_TP2 is off)."""
    on = dispatch.MC_AR_TP2
    tiny = dispatch.DECODE_PLANS[0]
    assert (tiny.name, tiny.max_tokens) == ("tiny", dispatch.TINY_MAX_TOKENS)
    assert (tiny.attn_mc_ar, tiny.moe_mc_ar, tiny.qkv_prequant) == (on, on, on), tiny
    # B6: the o_proj push follows the boundary's rows up to the o_proj GEMV's 16, unless B4_ATTN_PUSH is killed.
    push_rows = (
        min(dispatch.PLAN_MC_AR_MAX_TOKENS, dispatch.B4_MAX_TOKENS)
        if dispatch.B4_ATTN_PUSH
        else dispatch.TINY_MAX_TOKENS
    )
    assert dispatch.ATTN_PUSH_MAX_TOKENS == (push_rows if on and dispatch.ATTN_PUSH_TP2 else 0)
    assert not dispatch.B4_ATTN_PUSH or dispatch.B4_MC_AR
    assert dispatch.ATTN_PUSH_MAX_TOKENS <= max(
        r for (nk, r) in dispatch.MX_GEMV.keys if nk == dispatch.O_PROJ
    )
    assert dispatch.DENSE_PUSH_MAX_TOKENS == (
        dispatch.TINY_MAX_TOKENS if on and dispatch.DENSE_PUSH_TP2 else 0
    )
    for rows in _DECODE_ROWS:
        assert dispatch.plan_for(rows) is dispatch.DECODE_PLANS[0]
    # The MoE pushes only from its engine, at the engine's rows.
    assert dispatch.MOE_PUSH_MAX_TOKENS == (dispatch.MOE_ENGINE_MAX_TOKENS if on else 0)


def test_the_generic_boundaries_take_the_multicast_all_reduce() -> None:
    """Prefill, mixed steps and the Eagle3 capture layers: the multicast all-reduce's 48-token bucket, alone and folded,
    and the MoE finalize folded into it, as the agg canon runs them."""
    rows = 48 if dispatch.MC_AR_TP2 else 0
    assert dispatch.GENERIC_MC_AR_MAX_TOKENS == rows
    assert dispatch.GENERIC_MC_AR_FOLDED_MAX_TOKENS == rows
    assert dispatch.GENERIC_MC_FINALIZE_MAX_TOKENS == rows


def test_the_tensor_core_router_reaches_48_rows() -> None:
    assert dispatch.ROUTER_TC_GEN_MAX_ROWS == 48


def test_the_sampled_verify_owner_walk_is_one_token_by_default() -> None:
    assert dispatch.m3_sampled_accept_kernel.ILP_WALK is False


def test_the_fused_embedding_rows_follow_the_kill_constant() -> None:
    # B5: 9..16 rows too (not 17..32), unless B4_EMBED_FIRSTNORM is killed.
    rows = dispatch.B4_MAX_TOKENS if dispatch.B4_EMBED_FIRSTNORM else dispatch.TINY_MAX_TOKENS
    assert dispatch.EMBED_FIRSTNORM_MAX_TOKENS == (rows if dispatch.EMBED_FIRSTNORM_TP2 else 0)


# ------------------------------------------------------- landed: MX-TAC (pinned decode-graph MXFP8 GEMM tactic)


def test_the_decode_mxfp8_decision_is_pinned_to_cute_dsl() -> None:
    from tensorrt_llm._torch.custom_ops.torch_custom_ops import (
        FlashInferMXFP8GemmRunner,
        MXFP8QuantizeRunner,
    )

    assert dispatch.MXFP8_DECODE_QUANT_TACTIC == MXFP8QuantizeRunner.CUTE_DSL
    assert dispatch.MXFP8_DECODE_GEMM_TACTIC == FlashInferMXFP8GemmRunner.CUTE_DSL


@pytest.mark.parametrize("target", ["DecodeTarget", "PrefillTarget"])
def test_the_moe_forward_reports_the_engine_taken(target) -> None:
    """The engagement evidence comes from the forward (not the engine build's warm call)."""

    src = inspect.getsource(getattr(dispatch, target)._moe)
    assert src.index("taken(MOE_ENGINE.path") < src.index("m3_moe_engine(h.contiguous()")


# ------------------------------------------------------------------ the drafter's low-M linears (TP8 port)


def test_draft_lowm_covers_exactly_the_certified_rows_of_its_tactic_table() -> None:
    from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_nvfp4__sm_107__tp2.ops.gemm import (  # noqa: E501
        m3_draft_lowm as entry,
    )

    assert set(entry.TACTICS) == set(entry.ROWS) == set(dispatch.DRAFT_LOWM_ROWS)
    assert dispatch.DRAFT_LOWM.keys == frozenset(
        (nk, m)
        for nk, rows in entry.ROWS.items()
        for m in rows
        if m <= dispatch.DRAFT_LOWM_MAX_ROWS
    )
    for nk, table in entry.TACTICS.items():
        rows = [row[0] for row in table]
        assert rows == sorted(rows) and len(set(rows)) == len(rows), nk
        for m in entry.ROWS[nk]:
            assert entry.tactic_for(*nk, m) is not None, (nk, m)


def test_prebuild_draft_head_is_idempotent_and_keeps_replaced_copies(monkeypatch) -> None:
    """A repeated cache_derived_state with an unchanged head weight keeps the built (placed) copy; a changed weight
    rebuilds it and keeps the old copy alive, so a CUDA graph captured earlier never reads unmapped memory."""

    import torch

    built = []

    def fake_prepare(lm_head, tp_rank):
        head = SimpleNamespace(weight=lm_head.weight)
        built.append(head)
        return head

    monkeypatch.setattr(draft_step_kernel, "prepare_head", fake_prepare)
    worker = object.__new__(dispatch.M3Eagle3Worker)
    worker._m3_draft_head = None
    worker._m3_draft_head_version = None
    worker._m3_draft_head_graveyard = []
    worker.mapping = SimpleNamespace(tp_rank=0)
    monkeypatch.setattr(worker, "_m3_drafter_supported", lambda dm: True, raising=False)
    model = SimpleNamespace(lm_head=SimpleNamespace(weight=torch.zeros(4, 4)))

    assert worker.prebuild_draft_head(model) and len(built) == 1
    first = worker._m3_draft_head
    assert worker.prebuild_draft_head(model) and len(built) == 1 and worker._m3_draft_head is first
    model.lm_head.weight.add_(1.0)  # new values: version bump
    assert worker.prebuild_draft_head(model) and len(built) == 2
    assert worker._m3_draft_head is built[1] and worker._m3_draft_head_graveyard == [first]


# ----------------------------------------------------------------------- batch 4 (this target): the b16 plan


def test_the_batch_8_steps_take_the_b16_and_b32_plans() -> None:
    """9..16-row steps run the "b16" plan (16-token bucket) and 17..32-row steps the "b32" plan (48-token bucket), both
    on the multicast all-reduce (B4_MC_AR) with no MXFP8 copy for the next QKV; 1..8 rows keep the "tiny" plan."""
    if not dispatch.B4_DECODE:
        assert len(dispatch.DECODE_PLANS) == 1
        return
    names = [p.name for p in dispatch.DECODE_PLANS]
    assert names == (["tiny", "b16", "b32"] if dispatch.B8_DECODE else ["tiny", "b16"])
    b16 = dispatch.DECODE_PLANS[1]
    assert b16.max_tokens == dispatch.B4_MAX_TOKENS == 16
    assert dispatch.DECODE_MAX_TOKENS == (32 if dispatch.B8_DECODE else 16)
    b3 = dispatch.B4_MC_AR
    for plan in dispatch.DECODE_PLANS[1:]:
        quant = dispatch.B8_QKV_PREQUANT if plan.name == "b32" else dispatch.B4_QKV_PREQUANT
        assert (plan.attn_mc_ar, plan.moe_mc_ar, plan.qkv_prequant) == (b3, b3, quant)
    assert dispatch.PLAN_MC_AR_MAX_TOKENS == (
        dispatch.DECODE_MAX_TOKENS if b3 else dispatch.TINY_MAX_TOKENS
    )
    for rows in range(1, dispatch.DECODE_MAX_TOKENS + 1):
        want = "tiny" if rows <= dispatch.TINY_MAX_TOKENS else "b16" if rows <= 16 else "b32"
        assert dispatch.plan_for(rows).name == want, rows


def test_the_batch_4_gemv_rows() -> None:
    """B2: the decode GEMVs serve 1..16 rows (kill MX_GEMV_B4: 1..8), the bf16 one except dense gate_up / down."""
    if not dispatch.MX_GEMV_TP2:
        return
    for nk, rows in dispatch.MX_GEMV_ROWS.items():
        assert rows == frozenset(range(1, _mx_hi(nk) + 1)), nk
        assert dispatch.MX_GEMV_BF16_MAX_TOKENS[nk] == _bf16_hi(nk), nk


@pytest.mark.parametrize("nk", sorted(_BF16_WIDE_DECLINES), ids=str)
def test_the_bf16_gemv_declines_take_the_generic_gemm_at_batch_4(nk) -> None:
    """Dense gate_up / down on a bf16 activation: the GEMV at 1..8 rows, the generic GEMM at 9..16 (the MXFP8 GEMV
    still serves them on a prequantized activation)."""
    if not dispatch.MX_GEMV_TP2:
        return
    for rows in range(1, dispatch.DECODE_MAX_TOKENS + 1):
        bf16 = dispatch.linear_path(*nk, rows, prequant=False, low_m_gemm=False)
        mx = dispatch.linear_path(*nk, rows, prequant=True, low_m_gemm=False)
        assert bf16 == (
            dispatch.MX_GEMV_BF16_PATH
            if rows <= dispatch.TINY_MAX_TOKENS
            else dispatch.GENERIC_GEMM_PATH
        ), rows
        assert mx == (
            dispatch.MX_GEMV_PATH if rows <= _mx_hi(nk) else dispatch.PREQUANT_GEMM_PATH
        ), rows


def test_no_other_fast_path_is_raised_past_the_batch_1_2_rows() -> None:
    """Only the paths this target raises for 16 rows (the GEMVs, B3's multicast all-reduce) cover more than
    TINY_MAX_TOKENS rows; every other row-keyed path declines a 9..16-row step to stock inside the decode target."""
    raised = {
        dispatch.MX_GEMV.path,
        dispatch.MX_GEMV_BF16.path,
        dispatch.MC_ALLREDUCE.path,
        dispatch.MC_FINALIZE.path,
        dispatch.ROUTER_TC_GEN.path,  # agg G-d: 9..48 rows
    }
    # SpecTail TR-5b: B5's fused embedding and B6's o_proj push past 8 rows, unless killed.
    if dispatch.B4_EMBED_FIRSTNORM:
        raised.add(dispatch.EMBED_FIRSTNORM.path)
    if dispatch.B4_ATTN_PUSH:
        raised.add(dispatch.ATTN_PUSH.path)
    if dispatch.B4_SPEC_TAIL:
        # The B4 spec tail: the fused verify's and draft loop's row-keyed paths up to SPEC_TAIL_MAX_TOKENS.
        raised |= {
            dispatch.DRAFT_INPUT.path,
            dispatch.CAPTURE_NORM.path,
            dispatch.CHUNK_NORM.path,
        }
    # Keys that are not row counts (both selects' key widths; the index scorer's request counts, W-IDX1), or rows
    # already certified past 8 in the batch-1/2 target (the die-split head's 12 / 16-row cells, reached only with the
    # fused verify).
    not_rows = {
        dispatch.LONG_SELECT.path,
        dispatch.LONG_SELECT_DSL.path,
        dispatch.UGPU_HEAD.path,
        dispatch.INDEX_SCORE.path,
    }
    for path in dispatch.FAST_PATHS:
        if path.path in raised or path.path in not_rows:
            continue
        for key in path.keys:
            if isinstance(key, int):
                assert key <= dispatch.TINY_MAX_TOKENS, (path.path, key)


def test_the_batch_4_moe_finalizes_on_the_multicast_kernel() -> None:
    """1..8 rows keep the MoE engine; 9..16 rows (past it) take the multicast finalize + all-reduce + norm on the
    16-token bucket, or the generic MoE when B4_MC_FINALIZE is killed; nothing past the decode cap."""
    for rows in range(1, dispatch.TINY_MAX_TOKENS + 1):
        assert dispatch.moe_path(rows) == (
            dispatch.MOE_ENGINE.path if not dispatch.KILL_MOE_ENGINE else None
        ), rows
    for rows in range(dispatch.TINY_MAX_TOKENS + 1, dispatch.DECODE_MAX_TOKENS + 1):
        expected = dispatch.MC_FINALIZE.path if dispatch.B4_MC_FINALIZE else None
        assert dispatch.moe_path(rows) == expected, rows
    assert dispatch.moe_path(dispatch.DECODE_MAX_TOKENS + 1) is None


# ------------------------------------------------------------------ W-G1: the MXFP8 GEMV at 33..48 rows (agg wide
# decode)

_W48 = (dispatch.QKV_SPARSE, dispatch.QKV, dispatch.SHARED_GATE_UP, dispatch.SHARED_DOWN)


def test_w48_rows_follow_the_kill_constant() -> None:
    """WIDE_MX_GEMV_48 raises the four projections it wins on to their served rows past 32 (at most 48); o_proj and the
    dense MLP keep theirs."""
    on = (
        dispatch.MX_GEMV_TP2
        and dispatch.WIDE_MX_GEMV_48
        and dispatch.GEMV_MAX_TOKENS == dispatch.DECODE_MAX_TOKENS
    )
    for nk in _W48:
        rows = dispatch._GEMV_W48_ROWS.get(nk, frozenset())
        if not on:
            assert not rows, nk
            continue
        assert rows and min(rows) == dispatch.GEMV_MAX_TOKENS + 1, nk
        assert max(rows) <= dispatch.WIDE_GEMV_MAX_TOKENS, nk
        assert rows == frozenset(range(min(rows), max(rows) + 1)), nk
        assert rows <= dispatch.MX_GEMV_ROWS[nk], nk
    for nk in (dispatch.O_PROJ, dispatch.DENSE_GATE_UP, dispatch.DENSE_DOWN):
        assert nk not in dispatch._GEMV_W48_ROWS and nk not in dispatch.MX_GEMV_QUANT_ROWS, nk


@pytest.mark.parametrize("nk", _W48, ids=str)
def test_w48_bf16_activation_takes_quantize_plus_gemv(nk) -> None:
    """A bf16 activation on rows the bf16 GEMV does not serve but the MXFP8 GEMV does: linear_path returns the GEMV
    (M3MXFP8LinearMethod.apply quantizes with the decode-graph quantize first); rows the bf16 GEMV serves keep it; rows
    past the MXFP8 GEMV's reach keep the generic GEMM."""
    if not (dispatch.MX_GEMV_TP2 and dispatch.WIDE_MX_GEMV_48):
        return
    bf16_hi = dispatch.MX_GEMV_BF16_MAX_TOKENS.get(nk, 0)
    for rows in range(1, 65):
        path = dispatch.linear_path(*nk, rows, prequant=False, low_m_gemm=False)
        if rows <= bf16_hi:
            assert path == dispatch.MX_GEMV_BF16_PATH, rows
        elif rows in dispatch.MX_GEMV_ROWS[nk]:
            assert rows in dispatch.MX_GEMV_QUANT_ROWS[nk], rows
            assert path == dispatch.MX_GEMV_PATH, rows
        else:
            assert path == dispatch.GENERIC_GEMM_PATH, rows
        if rows in dispatch.MX_GEMV_ROWS[nk]:
            assert (
                dispatch.linear_path(*nk, rows, prequant=True, low_m_gemm=False)
                == dispatch.MX_GEMV_PATH
            ), rows


def test_w48_rows_are_certified_by_the_gemv_cells() -> None:
    """Every (projection, rows) the dispatch can send to the MXFP8 GEMV is one of m3_mx_gemv's CELLS (served_cells to
    48)."""
    cells = {(c.spec["nk"], c.spec["m"]) for c in dispatch.m3_mx_gemv.CELLS}
    for nk, rows in dispatch.MX_GEMV_ROWS.items():
        for m in rows:
            assert (nk, m) in cells, (nk, m)
