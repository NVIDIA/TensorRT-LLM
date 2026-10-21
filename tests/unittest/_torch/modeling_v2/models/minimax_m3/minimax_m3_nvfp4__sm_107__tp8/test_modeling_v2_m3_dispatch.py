# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The tp8 (Rubin, two trays) decode target's fast-path table (``modeling.py``: the row ranges and the selection
functions): its row ranges against the catalog cells that certify them, and the selection for every decode row count.

Every range's covered shapes are listed in ``ENABLED``; a range not listed covers nothing.

CPU only: nothing here launches a kernel.
"""

import inspect
from types import SimpleNamespace

import pytest

from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_nvfp4__sm_107__tp8 import (  # noqa: E501
    modeling as dispatch,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_nvfp4__sm_107__tp8.ops import (  # noqa: E501
    m3_draft_step as draft_step_kernel,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_nvfp4__sm_107__tp8.ops import (  # noqa: E501
    triton_sparse_decode as tsd,
)

ROWS = range(1, dispatch.DECODE_MAX_TOKENS + 2)  # every decode row count and the first one past it


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
    """Two requests verifying 1 + 3 drafts; a wider step is the prefill target's."""
    assert dispatch.DECODE_MAX_TOKENS == 8
    assert [dispatch.is_decode_step(n) for n in ROWS] == [True] * 8 + [False]


# ------------------------------------------------------------------ the cells turned on


def test_the_moe_engine_takes_every_decode_step() -> None:
    """The MoE lane's m3_moe engine (TP8 width 384, shared expert folded in) at 1..8 rows, unless killed; its sum
    pushes into the multicast boundary all-reduce at the engine's rows while that all-reduce is on."""
    expected = set() if dispatch.KILL_MOE_ENGINE else set(range(1, 9))
    assert set(dispatch.MOE_ENGINE.keys) == expected
    for rows in ROWS:
        want = dispatch.MOE_ENGINE.path if rows in expected else None
        assert dispatch.moe_path(rows) == want, rows
    push_rows = dispatch.MOE_ENGINE_MAX_TOKENS if dispatch.MC_AR_TP2 else 0
    assert dispatch.MOE_PUSH_MAX_TOKENS == push_rows
    assert set(dispatch.MOE_PUSH.keys) == set(range(1, push_rows + 1))


# ------------------------------------------------------------------ on: the sparse layers' block selection


def test_the_index_scorer_takes_one_and_two_requests_with_their_schedules() -> None:
    assert dispatch.INDEX_SCORE.keys == frozenset({1, 2})
    assert dispatch.index_score_schedule(1) == (1272, True)
    assert dispatch.index_score_schedule(2) == (768, True)
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
    spec = dict(seq_lens=(165521,), qlen=4, heads=1, split_k=512, early=True)
    assert tuple(dispatch.INDEX_SCORE.certifies(spec)) == ()
    assert tuple(dispatch.INDEX_SCORE.certifies(dict(spec, split_k=1272, heads=2))) == ()
    assert tuple(dispatch.INDEX_SCORE.certifies(dict(spec, split_k=1272))) == (1,)


#: Every TP8 per-rank MXFP8 projection (N, K): sparse-layer QKV + index, dense QKV, o_proj, shared gate_up / down,
#: dense gate_up / down.
TP8_PROJECTIONS = (
    (1536, 6144),
    (1280, 6144),
    (6144, 1024),
    (768, 6144),
    (6144, 384),
    (3072, 6144),
    (6144, 1536),
)
_DECODE_ROWS = range(1, dispatch.DECODE_MAX_TOKENS + 1)

#: Fast paths a lane has turned on, each with the shapes it covers (the rest of the skeleton stays off).
ENABLED = {
    # SpecTail: sampled (temperature + top-p) verify of 1-2 requests at draft length 3 (the shard kernel).
    "m3_sampled_accept": frozenset({(3, 1), (3, 2)}),
    # SpecTail: greedy verify (shard argmax + accept) and the fused draft loop of 1-2 requests at draft length 3,
    # and that loop's draft-input rows (1 / 2 later-pass rows, 4 / 8 first-pass rows).
    "m3_shard_accept": frozenset({(3, 1), (3, 2)}),
    "m3_draft_step": frozenset({(3, 1), (3, 2)}),
    "m3_draft_input": frozenset({1, 2, 4, 8}),
    # SpecTail: the drafter's BF16 linears on FlashInfer's low-M kernels (TP8 port) at the certified b1 rows.
    "m3_draft_lowm": frozenset(
        {
            ((1280, 12288), 1),
            ((1280, 12288), 4),
            ((6144, 1024), 1),
            ((6144, 1024), 4),
            ((4608, 6144), 1),
            ((4608, 6144), 4),
            ((6144, 2304), 1),
            ((6144, 2304), 4),
            ((6144, 18432), 1),
        }
    ),
    # SpecTail: the fused first drafter inputs of 1-2 fused-verify requests, the capture norm and the chunk norm at
    # their 4 / 8 verify rows (own ranges and kills since the TP2 fast-path audit).
    "m3_fused_first_inputs": frozenset({1, 2}),
    "m3_capture_norm": frozenset({4, 8}),
    "m3_chunk_norm": frozenset({4, 8}),
    # MSA: the one-launch long-context sparse decode at the TP8 heads (pre-wait up to its rows, plain above), the
    # identity-selection producer, and the sparse layers' NVFP4 QKV + indexer producer at the TP8 heads.
    "m3_sparse_decode_lc8": frozenset({"plain", "prewait"}),
    "m3_producer": frozenset({"nvfp4_tp8"}),
    "sparse_decode_producer": frozenset({"multi_query", "per_row"}),
    # MoE: the m3_moe engine (TP8 width 384, shared expert folded in) at 1..8 rows, unless killed.
    "m3_moe_engine": frozenset() if dispatch.KILL_MOE_ENGINE else frozenset(range(1, 9)),
    # The TF32 tensor-core router of 9..16-row pure generation steps on the generic layer (above the engine's rows).
    "m3_router_tc": frozenset(range(9, 17)),
    # Indexer: the index scorer at 1-2 requests, the long-row select at every key width (129..9216 blocks).
    "m3_index_score": frozenset({1, 2}),
    "long_select": frozenset({2, 4, 9}),
    "long_select_dsl": frozenset({4, 11, 24}),
    # LongAttn: the dense layers' full-context decode of 1-2 requests, 1 + 3 verify rows or one draft-pass row.
    "m3_fullctx_decode": frozenset({(4, 1), (1, 1), (4, 2), (1, 2)}),
    # GEMM/AR: the decode MXFP8 GEMVs at every TP8 projection and decode row, on an MXFP8 and a bf16 activation.
    "m3_mx_gemv": frozenset((nk, m) for nk in TP8_PROJECTIONS for m in _DECODE_ROWS)
    if dispatch.MX_GEMV_TP2
    else frozenset(),
    "m3_mx_gemv_bf16": frozenset((nk, m) for nk in TP8_PROJECTIONS for m in _DECODE_ROWS)
    if dispatch.MX_GEMV_TP2
    else frozenset(),
    # GEMM/AR: the multicast boundary all-reduce, the o_proj's push into it, the fused embedding + first RMSNorm.
    "mc_allreduce_norm": frozenset(_DECODE_ROWS) if dispatch.MC_AR_TP2 else frozenset(),
    "attn_push": frozenset(_DECODE_ROWS)
    if (dispatch.MC_AR_TP2 and dispatch.ATTN_PUSH_TP2)
    else frozenset(),
    "dense_push": frozenset(_DECODE_ROWS)
    if (dispatch.MC_AR_TP2 and dispatch.DENSE_PUSH_TP2)
    else frozenset(),
    # MoE: the engine's push into the multicast boundary all-reduce, at the engine's rows.
    "moe_push": frozenset(range(1, dispatch.MOE_PUSH_MAX_TOKENS + 1)),
    "embed_firstnorm": frozenset(_DECODE_ROWS) if dispatch.EMBED_FIRSTNORM_TP2 else frozenset(),
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
    """The sparse decode (LC8, the Triton merge) writes the MXFP8 o_proj hand-off; the range keys the target's
    SPARSE_DECODE_LC8 / SPARSE_DECODE_PRODUCER cover are the kernel module's."""
    assert tsd.MERGE_MXFP8_HANDOFF is True
    assert dispatch.SPARSE_DECODE_LC8.keys is tsd.LC8_KEYS
    assert dispatch.SPARSE_DECODE_PRODUCER.keys is tsd.PRODUCER_KEYS


def test_sparse_decode_lc8_covers_the_decode_steps_with_the_handoff() -> None:
    """Every decode step (at most DECODE_MAX_TOKENS rows) writes the MXFP8 o_proj hand-off; the one-launch decode
    stages its pre-wait reads up to m3_attn_lc8.PREWAIT_MAX_ROWS rows and merges once."""
    from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_nvfp4__sm_107__tp8.ops import (  # noqa: E501
        m3_attn_lc8,
    )

    assert dispatch.SPARSE_DECODE_LC8.keys == frozenset({"plain", "prewait"})
    assert tsd.MERGE_MXFP8_MIN_TOKENS <= 1 and dispatch.DECODE_MAX_TOKENS <= tsd.MergeMxfp8.MAX_ROWS
    assert (
        m3_attn_lc8.PREWAIT_MAX_ROWS == 8 and m3_attn_lc8.MERGE_ONCE and m3_attn_lc8.ONE_DIV_MERGE
    )


def test_the_dense_layers_take_the_fullctx_decode_at_the_gen_workers_shapes() -> None:
    """One or two requests, 1 + 3 verify rows or one draft-pass row, each at its own pinned recipe."""
    assert dispatch.FULLCTX_DECODE_ENABLED
    assert dispatch.FULLCTX_DECODE.keys == {(4, 1), (1, 1), (4, 2), (1, 2)}
    assert set(dispatch.FULLCTX_DECODE_RECIPES) == set(dispatch.FULLCTX_DECODE.keys)
    for recipe in dispatch.FULLCTX_DECODE_RECIPES.values():
        assert 2 <= recipe["splits"] <= 128  # the standalone reducer's range


def test_the_fullctx_entry_cells_run_at_the_forward_plan_capacity() -> None:
    from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_nvfp4__sm_107__tp8.ops.attention import (  # noqa: E501
        m3_fullctx_decode as entry,
    )

    assert entry.MAX_KV_LEN == dispatch.FULLCTX_DECODE_MAX_KV_LEN


def test_the_fullctx_capacity_covers_the_agentx_cache_bound() -> None:
    """The forward compares the cache's length bound (max_seq_len plus the speculative rows it reserves) with the plan
    capacity; at the AgentX max_seq_len of 1 << 20 with up to 8 draft rows the dense layers must still be taken."""
    assert dispatch.FULLCTX_DECODE_MAX_KV_LEN >= (1 << 20) + 8


def test_the_drafter_takes_the_fullctx_decode_at_the_same_shapes() -> None:
    assert dispatch.DRAFTER_FULLCTX_ENABLED
    assert dispatch.DRAFTER_FULLCTX.keys == dispatch.FULLCTX_DECODE.keys


# ------------------------------------------------------------------ landed: the decode MXFP8 GEMVs (GEMM/AR lane)


def test_the_gemv_rows_follow_the_kill_constant() -> None:
    expected = set(TP8_PROJECTIONS) if dispatch.MX_GEMV_TP2 else set()
    assert set(dispatch.MX_GEMV_ROWS) == expected
    assert set(dispatch.MX_GEMV_BF16_MAX_TOKENS) == expected


@pytest.mark.parametrize("rows", range(1, dispatch.DECODE_MAX_TOKENS + 1))
@pytest.mark.parametrize("nk", TP8_PROJECTIONS, ids=str)
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


@pytest.mark.parametrize("nk", TP8_PROJECTIONS, ids=str)
def test_wider_than_the_decode_step_is_the_generic_gemm(nk) -> None:
    rows = dispatch.DECODE_MAX_TOKENS + 1
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
    # SpecTail: fused greedy / sampled verify and the fused draft loop at b1/b2 (BF16 head; no coded heads).
    assert dispatch.SHARD_ACCEPT_MAX_REQUESTS == 2
    assert dispatch.SAMPLED_VERIFY_MAX_REQUESTS == 2
    assert dispatch.DRAFT_STEP_MAX_REQUESTS == 2
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
        (12, False, False),
        (12, True, False),
        (16, True, False),
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
    assert tuple(p.max_tokens for p in dispatch.DECODE_PLANS) == (dispatch.DECODE_MAX_TOKENS,)
    for plan in dispatch.DECODE_PLANS:
        assert (plan.attn_mc_ar, plan.moe_mc_ar, plan.qkv_prequant) == (on, on, on), plan
    assert dispatch.ATTN_PUSH_MAX_TOKENS == (
        dispatch.DECODE_MAX_TOKENS if on and dispatch.ATTN_PUSH_TP2 else 0
    )
    assert dispatch.DENSE_PUSH_MAX_TOKENS == (
        dispatch.DECODE_MAX_TOKENS if on and dispatch.DENSE_PUSH_TP2 else 0
    )
    for rows in _DECODE_ROWS:
        assert dispatch.plan_for(rows) is dispatch.DECODE_PLANS[0]
    # The MoE pushes only from its engine, at the engine's rows.
    assert dispatch.MOE_PUSH_MAX_TOKENS == (dispatch.MOE_ENGINE_MAX_TOKENS if on else 0)


def test_the_generic_boundaries_take_the_multicast_all_reduce() -> None:
    """Prefill, mixed steps and the Eagle3 capture layers: the same certified rows, alone and folded."""
    rows = dispatch.DECODE_MAX_TOKENS if dispatch.MC_AR_TP2 else 0
    assert dispatch.GENERIC_MC_AR_MAX_TOKENS == rows
    assert dispatch.GENERIC_MC_AR_FOLDED_MAX_TOKENS == rows


# ------------------------------------------------------------------ landed: the fused embedding + first RMSNorm


def test_the_fused_embedding_rows_follow_the_kill_constant() -> None:
    assert dispatch.EMBED_FIRSTNORM_MAX_TOKENS == (
        dispatch.DECODE_MAX_TOKENS if dispatch.EMBED_FIRSTNORM_TP2 else 0
    )


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
    from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_nvfp4__sm_107__tp8.ops.gemm import (  # noqa: E501
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
