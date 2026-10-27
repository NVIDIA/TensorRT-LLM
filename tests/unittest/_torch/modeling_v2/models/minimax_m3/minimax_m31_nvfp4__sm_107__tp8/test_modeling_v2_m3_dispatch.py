# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The tp8 (Rubin, two trays) decode target's fast-path table (``modeling.py``: the row ranges and the selection
functions): its row ranges against the catalog cells that certify them, and the selection for every decode row count.

Every range's covered shapes are listed in ``ENABLED``; a range not listed covers nothing.

CPU only: nothing here launches a kernel.
"""

import inspect
import os
import subprocess
import sys
from types import SimpleNamespace

import pytest

from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m31_nvfp4__sm_107__tp8 import (  # noqa: E501
    modeling as dispatch,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m31_nvfp4__sm_107__tp8.kernels import (  # noqa: E501
    m3_draft_step as draft_step_kernel,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m31_nvfp4__sm_107__tp8.kernels import (  # noqa: E501
    triton_sparse_decode as tsd,
)

ROWS = range(1, dispatch.DECODE_STEP_MAX + 2)  # every decode row count and the first one past it


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
    """The TP2-lineage paths' 8 rows (2 requests verifying 1 + 3 drafts); R1VR8 certifies the ops to 32 rows (8
    requests, K3 c8) and VR-BANDS raises the step bound to 64 (K7 c8); a wider step is the prefill target's."""
    assert dispatch.TP2_ROWS == 8
    assert dispatch.DECODE_MAX_TOKENS == _STEP_MAX
    assert dispatch.DECODE_STEP_MAX == (64 if dispatch.VR_BANDS else _STEP_MAX)
    assert [dispatch.is_decode_step(n) for n in ROWS] == [True] * dispatch.DECODE_STEP_MAX + [False]


# ------------------------------------------------------------------ the cells turned on


def test_the_moe_engine_takes_every_decode_step() -> None:
    """The m3_moe engine (TP8 width 384, shared expert folded in) at 1..8 rows, unless killed; its sum
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
# R1VR8: the step bound is 32 rows; the TP2-lineage paths keep their 8 certified rows;
# on an MXFP8 activation the GEMV serves the QKV and both gate_up projections up to 32 rows and the o_proj up to 16
# (the downs, bf16 input: 8). TRTLLM_M31_DISABLE_R1VR8=1: every range is 8 rows.
_STEP_MAX = 32 if dispatch.R1VR8 else 8
_DECODE_ROWS = range(1, 8 + 1)  # the TP2-lineage rows
_STEP_ROWS = range(1, _STEP_MAX + 1)
_GEMV_MAX = dict.fromkeys(TP8_PROJECTIONS, 8)
# Past the decode bound (VR_GEMV48 / LINEAR_CUBLASLT): the generation steps of 33..64 rows.
_WIDE_ROWS = range(max(33, dispatch.DECODE_MAX_TOKENS + 1), 64 + 1)
_GEMV48_ROWS = range(max(33, dispatch.DECODE_MAX_TOKENS + 1), 48 + 1)
_GEMV48_SHAPES = (
    (1536, 6144),
    (1280, 6144),
    (768, 6144),
)  # sparse / dense-attention QKV, shared gate_up
_CUBLASLT_SHAPES = (
    (1536, 6144),
    (6144, 1024),
    (768, 6144),
    (6144, 384),
    (3072, 6144),
    (6144, 1536),
)
if dispatch.R1VR8:
    _GEMV_MAX.update(
        {(1536, 6144): 32, (1280, 6144): 32, (768, 6144): 32, (3072, 6144): 32, (6144, 1024): 16}
    )

#: Fast paths turned on, each with the shapes it covers (the rest of the skeleton stays off).
ENABLED = {
    # Sampled (temperature + top-p) verify of 1-2 requests at draft length 3 (the shard kernel).
    "m3_sampled_accept": frozenset({(3, 1), (3, 2)}),
    # Greedy verify (shard argmax + accept) and the fused draft loop of 1-2 requests at draft length 3,
    # and that loop's draft-input rows (1 / 2 later-pass rows, 4 / 8 first-pass rows).
    "m3_shard_accept": frozenset({(3, 1), (3, 2)}),
    "m3_draft_step": frozenset({(3, 1), (3, 2)}),
    "m3_draft_input": frozenset({1, 2, 4, 8}),
    # The drafter's BF16 linears on FlashInfer's low-M kernels (TP8 port) at the certified b1 rows.
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
    # The fused first drafter inputs of 1-2 fused-verify requests, the capture norm and the chunk norm at
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
    "m3_router_tc": frozenset(range(9, (48 if dispatch.ROUTER48 else 16) + 1)),
    # The index scorer at 1-2 requests, the long-row select at every key width (129..9216 blocks).
    "m3_index_score": frozenset({1, 2}),
    "long_select": frozenset({2, 4, 9}),
    "long_select_dsl": frozenset({4, 11, 24}),
    # The dense layers' full-context decode of 1-2 requests, 1 + 3 verify rows or one draft-pass row.
    "m3_fullctx_decode": frozenset({(4, 1), (1, 1), (4, 2), (1, 2)}),
    # The decode MXFP8 GEMVs at every TP8 projection and decode row, on an MXFP8 and a bf16 activation.
    "m3_mx_gemv": frozenset((nk, m) for nk in TP8_PROJECTIONS for m in range(1, _GEMV_MAX[nk] + 1))
    if dispatch.MX_GEMV_TP2
    else frozenset(),
    "m3_mx_gemv_bf16": frozenset((nk, m) for nk in TP8_PROJECTIONS for m in _DECODE_ROWS)
    if dispatch.MX_GEMV_TP2
    else frozenset(),
    # Past the decode bound, the GEMV (VR_GEMV48) on the QKV and shared gate_up projections at 33..48 rows and
    # cuBLASLt's MXFP8 GEMM (LINEAR_CUBLASLT) on its six projections at every other row up to 64.
    "m3_mx_gemv48": frozenset((nk, m) for nk in _GEMV48_SHAPES for m in _GEMV48_ROWS)
    if (dispatch.MX_GEMV_TP2 and dispatch.VR_GEMV48)
    else frozenset(),
    "m3_mxfp8_cublaslt": frozenset(
        (nk, m)
        for nk in _CUBLASLT_SHAPES
        for m in _WIDE_ROWS
        if not (
            dispatch.MX_GEMV_TP2
            and dispatch.VR_GEMV48
            and nk in _GEMV48_SHAPES
            and m in _GEMV48_ROWS
        )
    )
    if dispatch.LINEAR_CUBLASLT
    else frozenset(),
    # The multicast boundary all-reduce, the o_proj's push into it, the fused embedding + first RMSNorm.
    "mc_allreduce_norm": frozenset(_STEP_ROWS) if dispatch.MC_AR_TP2 else frozenset(),
    "attn_push": frozenset(_DECODE_ROWS)
    if (dispatch.MC_AR_TP2 and dispatch.ATTN_PUSH_TP2)
    else frozenset(),
    "dense_push": frozenset(_DECODE_ROWS)
    if (dispatch.MC_AR_TP2 and dispatch.DENSE_PUSH_TP2)
    else frozenset(),
    # MoE: the engine's push into the multicast boundary all-reduce, at the engine's rows.
    "moe_push": frozenset(range(1, dispatch.MOE_PUSH_MAX_TOKENS + 1)),
    "embed_firstnorm": frozenset(_DECODE_ROWS) if dispatch.EMBED_FIRSTNORM_TP2 else frozenset(),
    # The fused draft loop's all-reduce boundaries (1..8 rows) on the multicast all-reduce (opts 16).
    "m3_draft_mc_allreduce_norm": frozenset(range(1, dispatch.DRAFT_MC_AR_MAX_ROWS + 1)),
    # The Eagle3 drafter layer's generation passes of 1-2 requests on the same recipes.
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
    from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m31_nvfp4__sm_107__tp8.kernels import (  # noqa: E501
        m3_attn_lc8,
    )

    assert dispatch.SPARSE_DECODE_LC8.keys == frozenset({"plain", "prewait"})
    assert tsd.MERGE_MXFP8_MIN_TOKENS <= 1 and dispatch.DECODE_MAX_TOKENS <= tsd.MergeMxfp8.MAX_ROWS
    assert (
        m3_attn_lc8.PREWAIT_MAX_ROWS == 8
        and m3_attn_lc8.MERGE_ONCE
        and not m3_attn_lc8.ONE_DIV_MERGE
    )
    # ONE_DIV is default off (TRTLLM_M31_ENABLE_LC8_ONEDIV=1 opts in); when on, up to 8 rows only (
    # TRTLLM_M31_DISABLE_ONEDIV_CAP=1 keeps it at every row count).
    assert m3_attn_lc8.ONE_DIV_MAX_ROWS == 8
    assert m3_attn_lc8.one_div_applies(
        8
    ) is m3_attn_lc8.ONE_DIV_MERGE and not m3_attn_lc8.one_div_applies(9)


def test_the_dense_layers_take_the_fullctx_decode_at_the_gen_workers_shapes() -> None:
    """One or two requests, 1 + 3 verify rows or one draft-pass row, each at its own pinned recipe."""
    assert dispatch.FULLCTX_DECODE_ENABLED
    assert dispatch.FULLCTX_DECODE.keys == {(4, 1), (1, 1), (4, 2), (1, 2)}
    assert set(dispatch.FULLCTX_DECODE_RECIPES) == set(dispatch.FULLCTX_DECODE.keys)
    for recipe in dispatch.FULLCTX_DECODE_RECIPES.values():
        assert 2 <= recipe["splits"] <= 128  # the standalone reducer's range


def test_the_fullctx_entry_cells_run_at_the_forward_plan_capacity() -> None:
    from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m31_nvfp4__sm_107__tp8.ops.attention import (  # noqa: E501
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


# ------------------------------------------------------------------ landed: the decode MXFP8 GEMVs


def test_the_gemv_rows_follow_the_kill_constant() -> None:
    expected = set(TP8_PROJECTIONS) if dispatch.MX_GEMV_TP2 else set()
    assert set(dispatch.MX_GEMV_ROWS) == expected
    assert set(dispatch.MX_GEMV_BF16_MAX_TOKENS) == expected


@pytest.mark.parametrize("rows", _DECODE_ROWS)
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


@pytest.mark.parametrize("rows", [*_WIDE_ROWS, 65])
@pytest.mark.parametrize("nk", TP8_PROJECTIONS, ids=str)
def test_wider_than_the_decode_step(nk, rows) -> None:
    """Past the decode bound: the GEMV on GEMV48_SHAPES at 33..48 rows, cuBLASLt's MXFP8 GEMM on its shapes at the
    other rows up to 64, else (and past 64) FlashInfer's GEMM; the same choice for either activation."""
    gemv48 = (
        (dispatch.MX_GEMV_TP2 and dispatch.VR_GEMV48)
        and nk in _GEMV48_SHAPES
        and rows in _GEMV48_ROWS
    )
    cublaslt = (
        dispatch.LINEAR_CUBLASLT and nk in _CUBLASLT_SHAPES and rows in _WIDE_ROWS and not gemv48
    )
    for prequant, generic in (
        (True, dispatch.PREQUANT_GEMM_PATH),
        (False, dispatch.GENERIC_GEMM_PATH),
    ):
        expected = (
            dispatch.GEMV48_PATH
            if gemv48
            else (dispatch.CUBLASLT_GEMM_PATH if cublaslt else generic)
        )
        assert dispatch.linear_path(*nk, rows, prequant=prequant, low_m_gemm=True) == expected


@pytest.mark.parametrize(
    "switch, expected",
    [
        # Without the GEMV past the bound, cuBLASLt takes those rows too; without cuBLASLt, FlashInfer's GEMM does.
        ("TRTLLM_M31_DISABLE_VR_GEMV48", "0 m3_mxfp8_cublaslt m3_mxfp8_cublaslt"),
        ("TRTLLM_M31_DISABLE_LINEAR_CUBLASLT", "0 m3_mx_gemv48 flashinfer_mxfp8_gemm"),
    ],
)
def test_wide_kill_switches(switch, expected) -> None:
    """Each switch, read at import, removes its rows (printed: the killed table's size, the 40-row QKV_SPARSE path,
    the 56-row QKV_SPARSE path on an MXFP8 activation)."""
    table = "GEMV48_ROWS" if "GEMV48" in switch else "LINEAR_CUBLASLT_ROWS"
    code = (
        f"from {dispatch.__name__} import {table} as r, linear_path as p\n"
        "print(len(r), p(1536, 6144, 40, True, True), p(1536, 6144, 56, True, True))"
    )
    env = {**os.environ, switch: "1"}
    out = subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True
    )
    assert " ".join(out.stdout.split()[-3:]) == expected


def test_router48_kill_switch() -> None:
    """TRTLLM_M31_DISABLE_ROUTER48=1, read at import, keeps the tensor-core router at its 16-row range."""
    code = (
        f"from {dispatch.__name__} import ROUTER_TC_GEN as r\n"
        "print(min(r.keys), max(r.keys), r.covers(32))"
    )
    env = {**os.environ, "TRTLLM_M31_DISABLE_ROUTER48": "1"}
    out = subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True
    )
    assert out.stdout.split()[-3:] == ["9", "16", "False"]
    assert max(dispatch.ROUTER_TC_GEN.keys) == (48 if dispatch.ROUTER48 else 16)


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
    # Fused greedy / sampled verify and the fused draft loop at b1/b2 (BF16 head; no coded heads).
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
    # The sampled verify is on (SAMPLED_VERIFY_MAX_REQUESTS 2, test_skeleton_speculation_...).
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


# ------------------------------------------------------- landed: the multicast boundary all-reduce


def test_both_boundaries_take_the_multicast_all_reduce() -> None:
    """The 8-token plan for the TP2 rows (R1VR8 adds the 16- and 32-row plans); every plan's boundaries take the
    multicast all-reduce and the MoE boundary publishes the next QKV's MXFP8 rows (killed: every boundary on the
    built-in all-reduce while MC_AR_TP2 is off)."""
    on = dispatch.MC_AR_TP2
    assert tuple(p.max_tokens for p in dispatch.DECODE_PLANS) == (
        (8, 16, 32) if dispatch.R1VR8 else (8,)
    ) + ((64,) if dispatch.VR_BANDS else ())
    for plan in dispatch.DECODE_PLANS:
        want = (False, False, False) if plan.name == "wide" else (on, on, on)
        assert (plan.attn_mc_ar, plan.moe_mc_ar, plan.qkv_prequant) == want, plan
    assert dispatch.ATTN_PUSH_MAX_TOKENS == (8 if on and dispatch.ATTN_PUSH_TP2 else 0)
    assert dispatch.DENSE_PUSH_MAX_TOKENS == (8 if on and dispatch.DENSE_PUSH_TP2 else 0)
    for rows in _DECODE_ROWS:
        assert dispatch.plan_for(rows) is dispatch.DECODE_PLANS[0]
    # The MoE pushes only from its engine, at the engine's rows.
    assert dispatch.MOE_PUSH_MAX_TOKENS == (dispatch.MOE_ENGINE_MAX_TOKENS if on else 0)


def test_the_generic_boundaries_take_the_multicast_all_reduce() -> None:
    """Prefill, mixed steps and the Eagle3 capture layers: the same certified rows, alone and folded."""
    rows = 8 if dispatch.MC_AR_TP2 else 0
    assert dispatch.GENERIC_MC_AR_MAX_TOKENS == rows
    assert dispatch.GENERIC_MC_AR_FOLDED_MAX_TOKENS == rows


# ------------------------------------------------------------------ landed: the fused embedding + first RMSNorm


def test_the_fused_embedding_rows_follow_the_kill_constant() -> None:
    assert dispatch.EMBED_FIRSTNORM_MAX_TOKENS == (8 if dispatch.EMBED_FIRSTNORM_TP2 else 0)


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
    from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m31_nvfp4__sm_107__tp8.ops.gemm import (  # noqa: E501
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


def _runtime(fused: bool, indexer: str = "fp8", kv: str = "fp8"):
    """A model config with the runtime fields ``_validate_runtime`` reads."""
    sparse = SimpleNamespace(
        algorithm="minimax_m3",
        implementation="msa",
        fuse_qkv_index_projection=fused,
        indexer_kv_dtype=indexer,
    )
    mode = SimpleNamespace(
        has_fp8_kv_cache=lambda: kv == "fp8", has_fp4_kv_cache=lambda: kv == "nvfp4"
    )
    return SimpleNamespace(
        sparse_attention_config=sparse, quant_config=SimpleNamespace(quant_mode=mode)
    )


@pytest.mark.parametrize("fused", [True, False], ids=["five_way", "unfused"])
def test_the_target_takes_both_projection_layouts(fused) -> None:
    """The packed five-way projection and the separate QKV + MXFP8 index projections
    (fuse_qkv_index_projection=False) are both this target's; a BF16 indexer or a BF16 KV cache is not."""
    dispatch._validate_runtime(_runtime(fused))
    dispatch._validate_runtime(_runtime(fused, kv="nvfp4"))
    with pytest.raises(ValueError, match="indexer_kv_dtype='fp8'"):
        dispatch._validate_runtime(_runtime(fused, indexer="bf16"))
    with pytest.raises(ValueError, match="FP8 or NVFP4 KV cache"):
        dispatch._validate_runtime(_runtime(fused, kv="bf16"))
