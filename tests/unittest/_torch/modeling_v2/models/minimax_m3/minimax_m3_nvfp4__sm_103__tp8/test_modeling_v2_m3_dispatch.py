# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The tp8 decode target's dispatch table (``dispatch.py``): its row ranges against the catalog cells that certify
them and the kernels' own limits, the selection for every decode row count, and the miss / uncovered policy.

CPU only: nothing here launches a kernel.
"""

import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_nvfp4__sm_103__tp8 import (  # noqa: E501
    dispatch,
    layers,
    linears,
    spec_worker,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_nvfp4__sm_103__tp8.kernels import (  # noqa: E501
    m3_attn_lc8,
    m3_draft_lowm,
    m3_draft_step,
    m3_fused_score_select,
    m3_mc_allreduce,
    m3_shard_accept,
    triton_sparse_decode,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_nvfp4__sm_103__tp8.kernels.mx_gemv import (  # noqa: E501
    op as mx_gemv,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_nvfp4__sm_103__tp8.modeling import (  # noqa: E501
    DecodeTarget,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_nvfp4__sm_103__tp8.ops.attention import (  # noqa: E501
    m3_fused_score_select as fused_score_select_entry,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_nvfp4__sm_103__tp8.ops.attention import (  # noqa: E501
    m3_sparse_decode_lc8 as lc8_entry,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_nvfp4__sm_103__tp8.ops.gemm import (  # noqa: E501
    m3_draft_lowm as draft_lowm_entry,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_nvfp4__sm_103__tp8.ops.gemm.m3_mx_gemv import (  # noqa: E501
    DENSE_DOWN,
    DENSE_GATE_UP,
    O_PROJ,
    QKV,
    QKV_SPARSE,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_nvfp4__sm_103__tp8.ops.spec import (  # noqa: E501
    m3_draft_step as draft_step_entry,
)
from tensorrt_llm._torch.modules.gated_mlp import GatedMLP
from tensorrt_llm._torch.modules.linear import TensorParallelMode

_TARGET = (
    "tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_nvfp4__sm_103__tp8"
)

PROJECTIONS = (QKV_SPARSE, QKV, O_PROJ, DENSE_GATE_UP, DENSE_DOWN)
ROWS = range(1, dispatch.DECODE_MAX_TOKENS + 2)  # every decode row count and the first one past it

# The Eagle3 drafter's BF16 linears at TP8 (per-rank qkv, o_proj, gate_up, down shards and the replicated fc).
DRAFTER_LINEARS = ((1280, 12288), (6144, 1024), (4608, 6144), (6144, 2304), (6144, 18432))


@pytest.fixture(autouse=True)
def _policy():
    """Every test starts outside require mode with nothing logged, and leaves it so."""
    dispatch.set_require(False)
    dispatch.reset_logged()
    yield
    dispatch.set_require(False)
    dispatch.reset_logged()


class _Log:
    def __init__(self):
        self.info, self.warning = [], []


@pytest.fixture
def log(monkeypatch) -> _Log:
    """The dispatch's log lines, by level."""
    rec = _Log()
    monkeypatch.setattr(dispatch.logger, "info", rec.info.append)
    monkeypatch.setattr(dispatch.logger, "warning", rec.warning.append)
    return rec


# ------------------------------------------------------------------ the ranges against their certificates


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


def test_the_removed_tp4_paths_are_gone() -> None:
    """The tp8 target carries none of the tp4 target's coded heads, low-M MXFP8 GEMM, 9..13-row MoE engine or
    attention megakernels: no fast path, range constant or kernel module of theirs is left to select."""
    names = {r.path for r in dispatch.FAST_PATHS}
    assert not names & {
        "m3_target_head",
        "mxfp8_low_m_gemm",
        "m3_moe16_engine",
        "attn_mega",
        "attn_c1",
        "fused_decode_attn",
    }
    for constant in (
        "TARGET_HEAD",
        "TARGET_HEAD_MAX_ROWS",
        "DRAFT_HEAD_CODED",
        "LOW_M_GEMM",
        "LOW_M_GEMM_PATH",
        "LOW_M_GEMM_MAX_TOKENS",
        "MOE16_ENGINE",
        "MOE16_MIN_TOKENS",
        "MOE16_MAX_TOKENS",
        "ATTN_MEGA",
        "ATTN_C1",
        "ATTN_C1_QUERY_LEN",
        "FUSED_DECODE_ATTN",
        "FUSED_ATTN_MAX_QUERY_LEN",
        "FUSED_ATTN_MAX_TOKENS",
        "SHARED_GATE_UP",
        "SHARED_DOWN",
    ):
        assert not hasattr(dispatch, constant), constant
    for module in (
        "kernels.m3_target_head",
        "kernels.mxfp8_low_m_gemm",
        "kernels.m3_attn_mega",
        "kernels.m3_attn_mega_c1",
        "kernels.m3_fused_attn",
        "kernels.m3_attn_lc",
        "ops.gemm.m3_target_head",
        "ops.gemm.mxfp8_low_m_gemm",
        "ops.moe.m3_moe16_engine",
        "ops.attention.m3_attn_mega",
        "ops.attention.m3_attn_c1",
        "ops.attention.m3_fused_decode_attn",
        "ops.attention.m3_sparse_decode_lc",
    ):
        assert importlib.util.find_spec(f"{_TARGET}.{module}") is None, module
    for method in ("prebuild_target_head", "_m3_coded_head"):
        assert not hasattr(spec_worker.M3Eagle3Worker, method), method
    for method in ("moe16_engine", "_build_moe16_engine"):
        assert not hasattr(layers.M3MoE, method), method


# ------------------------------------------------------------------ the ranges against the kernels' own limits


@pytest.mark.parametrize("nk", PROJECTIONS, ids=str)
def test_gemv_rows_are_the_rows_the_kernel_serves(nk) -> None:
    n, k = nk
    served = {m for m in range(1, mx_gemv.MAX_TOKENS + 1) if mx_gemv.serves(n, k, m, False)}
    assert dispatch.MX_GEMV_ROWS[nk] == served
    served_bf16 = [m for m in range(1, mx_gemv.MAX_TOKENS + 1) if mx_gemv.serves(n, k, m, True)]
    assert served_bf16 == list(range(1, dispatch.MX_GEMV_BF16_MAX_TOKENS[nk] + 1))


def test_the_gemv_tables_are_the_tp8_projections() -> None:
    assert set(dispatch.MX_GEMV_ROWS) == set(dispatch.MX_GEMV_BF16_MAX_TOKENS) == set(PROJECTIONS)
    assert PROJECTIONS == ((1536, 6144), (1280, 6144), (6144, 1024), (3072, 6144), (6144, 1536))
    for n, k in PROJECTIONS:
        assert mx_gemv.get_config(n, k) is not None, (n, k)


@pytest.mark.parametrize("nk,push", [((768, 6144), None), ((6144, 384), "moe")], ids=str)
def test_the_shared_expert_projections_take_no_gemv(nk, push) -> None:
    """At TP8 the shared expert (768 x 6144 gate_up, 6144 x 384 down) runs inside the MoE engine on decode rows and on
    the generic GEMM otherwise: the GEMV has no config for it and the dispatch never routes a call there."""
    assert mx_gemv.get_config(*nk) is None
    module = SimpleNamespace(weight=torch.zeros(1, 1).expand(*nk), m3_push=push, tp_size=8)
    assert linears.gemv_variants(module) == []


def test_the_plans_are_the_all_reduce_buckets() -> None:
    assert tuple(p.max_tokens for p in dispatch.DECODE_PLANS) == m3_mc_allreduce.BUCKETS
    assert dispatch.DECODE_MAX_TOKENS == m3_mc_allreduce.MAX_TOKENS == mx_gemv.MAX_TOKENS


def test_the_thresholds_are_the_kernels() -> None:
    assert dispatch.SPARSE_DECODE_MQ_MIN_TOKENS == triton_sparse_decode.MQ_MIN_TOKENS
    assert (
        dispatch.MOE_ENGINE_MAX_TOKENS < dispatch.ATTN_PUSH_MAX_TOKENS <= dispatch.DECODE_MAX_TOKENS
    )
    assert dispatch.MC_FINALIZE_MIN_TOKENS == dispatch.MOE_ENGINE_MAX_TOKENS + 1
    verify_rows = dispatch.SHARD_ACCEPT_MAX_REQUESTS * (dispatch.SHARD_ACCEPT_DRAFT_LEN + 1)
    assert verify_rows == dispatch.DECODE_MAX_TOKENS == m3_shard_accept.SHARD_ROWS
    assert dispatch.FUSED_SCORE_SELECT_MAX_REQUESTS == fused_score_select_entry.MAX_REQUESTS == 2
    assert dispatch.INDEX_SCORE_EARLY_MAX_REQUESTS == 0
    assert dispatch.FC_EARLY_MAX_ROWS == 0
    assert dispatch.LC8_PREWAIT_MAX_ROWS == 8
    assert dict(dispatch.DRAFT_DECODE_MAX_SEQ_LEN_BY_BATCH) == {1: 16384}
    assert dict(dispatch.DENSE_DECODE_MAX_SEQ_LEN_BY_BATCH) == {1: 16384}


def test_no_fast_path_quantizes_a_bf16_head_or_the_drafter() -> None:
    """The target carries the exact-numerics configuration only: the lm_head and the Eagle3 drafter run at the
    checkpoint's precision, so no fast path (nor a range constant) of an FP8 / NVFP4 / pruned head or of a quantized
    drafter projection is left to select."""
    names = {r.path for r in dispatch.FAST_PATHS}
    assert not names & {
        "fp8_lm_head",
        "draft_head",
        "m3_draft_gemv",
        "m3_draft_in_gemv",
        "m3_gemv_ar",
    }
    for constant in (
        "FP8_LM_HEAD",
        "FP8_LM_HEAD_MAX_TOKENS",
        "DRAFT_HEAD",
        "DRAFT_HEAD_MAX_TOKENS",
    ):
        assert not hasattr(dispatch, constant), constant


# ------------------------------------------------------------------ selection, every decode row count


def _expected_moe(rows: int):
    if rows <= 8:
        return "m3_moe_engine"
    if rows <= 48:
        return "mc_finalize_allreduce_norm"
    return None


def _expected_router(rows: int):
    if rows == 1:
        return "m3_router_gemv"
    if rows <= 48:
        return "m3_router_tc"
    return None


@pytest.mark.parametrize("rows", ROWS)
def test_selection(rows) -> None:
    assert dispatch.is_decode_step(rows) == (rows <= 48)
    if rows <= 48:
        assert dispatch.plan_for(rows).name == (
            "tiny" if rows <= 8 else "small" if rows <= 16 else "verify"
        )
    else:
        with pytest.raises(AssertionError, match="no decode plan"):
            dispatch.plan_for(rows)
    assert dispatch.moe_path(rows) == _expected_moe(rows)
    assert dispatch.router_path(rows) == _expected_router(rows)
    assert dispatch.EMBED_FIRSTNORM.covers(rows) == (rows <= 8)
    assert dispatch.ATTN_PUSH.covers(rows) == (rows <= 16)
    assert dispatch.MOE_PUSH.covers(rows) == (rows <= 8)
    assert dispatch.MC_ALLREDUCE.covers(rows) == (rows <= 48)
    assert dispatch.FUSED_SCORE_SELECT.covers(rows) == (rows <= 2)


@pytest.mark.parametrize("rows", ROWS)
def test_linear_selection(rows) -> None:
    bf16_max = {
        QKV_SPARSE: 16,
        QKV: 16,
        O_PROJ: 48,
        DENSE_GATE_UP: 16,
        DENSE_DOWN: 32,
    }
    for nk in PROJECTIONS:
        assert dispatch.linear_path(*nk, rows, True) == (
            "m3_mx_gemv" if rows <= 48 else "flashinfer_mxfp8_gemm"
        )
        assert dispatch.linear_path(*nk, rows, False) == (
            "m3_mx_gemv_bf16" if rows <= bf16_max[nk] else "mxfp8_gemm"
        )
    # A projection outside the table (the shared expert's, which runs in the MoE engine) takes the generic GEMM.
    for nk in ((768, 6144), (6144, 384), (4096, 4096)):
        assert dispatch.linear_path(*nk, rows, True) == "flashinfer_mxfp8_gemm"
        assert dispatch.linear_path(*nk, rows, False) == "mxfp8_gemm"


def test_draft_step_selection() -> None:
    """The fused draft loop takes the configured draft length at 1..8 requests (one head row per request, the
    kernel's limit); its draft input every pass's rows: the first pass's 4 per request, then one per request."""
    assert (
        dispatch.DRAFT_STEP_MAX_REQUESTS
        == draft_step_entry.MAX_TOKENS
        == m3_draft_step.MAX_TOKENS
        == 8
    )
    assert dispatch.DRAFT_STEP_DRAFT_LEN == dispatch.SHARD_ACCEPT_DRAFT_LEN
    assert draft_step_entry.WORLD == 8 and draft_step_entry.DRAFT_SHARD * 8 == 200064
    for draft_len in range(1, 6):
        for nreq in range(1, 14):
            covered = draft_len == 3 and nreq <= 8
            assert dispatch.DRAFT_STEP.covers((draft_len, nreq)) == covered
    assert dispatch.DRAFT_INPUT.keys == frozenset(range(1, 9)) | {12, 16, 20, 24, 28, 32}


def test_verify_selection() -> None:
    for draft_len in range(1, 6):
        for nreq in range(1, 14):
            covered = draft_len == 3 and nreq * (draft_len + 1) <= 48
            assert dispatch.SHARD_ACCEPT.covers((draft_len, nreq)) == covered


def test_the_index_scorer_runs_its_prefetching_schedule_only() -> None:
    """Spans of at most two requests score in the fused score + select; the scorer covers only the prefetching
    kernel (no one-request early-issue schedule is ever selected)."""
    assert dispatch.INDEX_SCORE.keys == frozenset({"prefetch"})
    assert not dispatch.INDEX_SCORE.covers("early")
    for cell in dispatch.INDEX_SCORE.entry.CELLS:
        assert not cell.spec["early"]
        assert len(cell.spec["seq_lens"]) > dispatch.FUSED_SCORE_SELECT_MAX_REQUESTS


def test_the_drafter_decode_split_is_capped_at_one_request() -> None:
    assert dispatch.DRAFT_DECODE_MAX_SEQ_LEN_BY_BATCH[1] == 16384
    assert 2 not in dispatch.DRAFT_DECODE_MAX_SEQ_LEN_BY_BATCH


# ------------------------------------------------------------------ the long-context sparse decode (lc8)


def test_lc8_variants_are_the_prewait_threshold() -> None:
    assert dispatch.SPARSE_DECODE_LC8.keys == frozenset({"prewait", "plain"})
    for cell in lc8_entry.m3_sparse_decode_lc8.CELLS:
        rows = len(cell.spec["seq_lens"]) * cell.spec["decode_query_len"]
        if cell.spec["prewait"]:
            assert rows <= dispatch.LC8_PREWAIT_MAX_ROWS, cell.why


def _lc8_operands(total_q: int):
    q = torch.zeros(total_q, lc8_entry.NUM_HEADS, lc8_entry.HEAD_DIM, dtype=torch.float8_e4m3fn)
    pages = torch.zeros(1, 1, lc8_entry.PAGE, lc8_entry.HEAD_DIM // 2, dtype=torch.uint8)
    scales = torch.zeros(1, 1, lc8_entry.PAGE, lc8_entry.HEAD_DIM // 16, dtype=torch.uint8)
    topk = torch.full((1, total_q, 16), -1, dtype=torch.int32)
    out = torch.zeros(total_q, lc8_entry.NUM_HEADS, lc8_entry.HEAD_DIM, dtype=torch.bfloat16)
    return q, pages, scales, topk, out


@pytest.mark.parametrize("total_q", [1, 4, 7, 8, 9, 16, 32, 48])
def test_lc8_prewait_follows_the_row_threshold(monkeypatch, total_q) -> None:
    """The call site stages before the grid wait exactly up to ``LC8_PREWAIT_MAX_ROWS`` query rows."""
    calls = []
    monkeypatch.setattr(m3_attn_lc8, "m3_sparse_decode_lc8", lambda *args: calls.append(args))
    q, pages, scales, topk, out = _lc8_operands(total_q)
    assert m3_attn_lc8.lc8_applies(q, pages, pages, topk, scales, scales, out)
    m3_attn_lc8.m3_lc8_attn(
        q,
        pages,
        pages,
        topk,
        torch.zeros(1, 1, dtype=torch.int32),
        torch.zeros(1, dtype=torch.int32),
        sm_scale=0.1,
        output=out,
        decode_query_len=1,
        k_block_scale=scales,
        v_block_scale=scales,
        k_global_scale=torch.ones(1),
        v_global_scale=torch.ones(1),
    )
    (args,) = calls
    assert args[-1] == (total_q <= dispatch.LC8_PREWAIT_MAX_ROWS)
    assert args[13] == lc8_entry.WIDTH // 128 == 8


def test_lc8_takes_the_tp8_geometry_only() -> None:
    q, pages, scales, topk, out = _lc8_operands(4)
    assert m3_attn_lc8.lc8_applies(q, pages, pages, topk, scales, scales, out)
    q16 = torch.zeros(4, 16, 128, dtype=torch.float8_e4m3fn)
    out16 = torch.zeros(4, 16, 128, dtype=torch.bfloat16)
    assert not m3_attn_lc8.lc8_applies(q16, pages, pages, topk, scales, scales, out16)


# ------------------------------------------------------------------ require mode and fallback logging


def test_a_miss_raises_under_require() -> None:
    dispatch.set_require(True)
    with pytest.raises(dispatch.FastPathUnavailable, match=r"'m3_moe_engine'.* 5 rows.*not built"):
        dispatch.missed("m3_moe_engine", 5, "the engine is not built")


def test_a_miss_logs_once_per_reason_otherwise(log) -> None:
    for _ in range(3):
        dispatch.missed("m3_moe_engine", 5, "the engine is not built")
    dispatch.missed("m3_moe_engine", 6, "the engine failed to build")
    assert len(log.warning) == 2
    assert "m3_moe_engine at 5 rows: the engine is not built" in log.warning[0]


def test_uncovered_never_raises_and_logs_once(log) -> None:
    dispatch.set_require(True)
    for _ in range(3):
        dispatch.uncovered("m3_fused_score_select", "more than 2 requests: scorer + selector")
    assert len(log.info) == 1 and not log.warning


class _Core:
    """The parts of the text model ``DecodeTarget._embed`` reads: no embedding owners built."""

    def __init__(self):
        self._embed_owners = {}

    @staticmethod
    def embed_tokens(input_ids: torch.Tensor) -> torch.Tensor:
        return input_ids.to(torch.float32)[:, None].expand(-1, 4)


@pytest.mark.parametrize("rows", (1, 8))
def test_embedding_without_its_owner_raises_under_require(rows) -> None:
    dispatch.set_require(True)
    with pytest.raises(
        dispatch.FastPathUnavailable, match=rf"'embed_firstnorm'.* {rows} rows.*owner"
    ):
        DecodeTarget(_Core())._embed(torch.arange(rows, dtype=torch.int32), rows)


def test_embedding_without_its_owner_falls_back_and_logs_once(log) -> None:
    target = DecodeTarget(_Core())
    for _ in range(2):
        hidden, residual = target._embed(torch.arange(3, dtype=torch.int32), 3)
    assert residual is None and hidden.shape == (3, 4)
    assert len(log.warning) == 1 and "embed_firstnorm at 3 rows" in log.warning[0]


def test_embedding_past_its_rows_is_uncovered_in_every_mode(log) -> None:
    dispatch.set_require(True)
    hidden, residual = DecodeTarget(_Core())._embed(torch.arange(9, dtype=torch.int32), 9)
    assert residual is None and hidden.shape == (9, 4)
    assert len(log.info) == 1 and "embed_firstnorm" in log.info[0] and not log.warning


def test_a_push_without_its_workspace_raises_under_require(monkeypatch) -> None:
    monkeypatch.setattr(m3_mc_allreduce, "get_workspace", lambda mapping, hidden: None)
    dispatch.set_require(True)
    with pytest.raises(dispatch.FastPathUnavailable, match=r"'attn_push'.* 4 rows"):
        DecodeTarget(_Core())._attn_push_target(_Layer(), 4, 6144)


def test_a_push_without_its_workspace_falls_back_and_logs_once(monkeypatch, log) -> None:
    monkeypatch.setattr(m3_mc_allreduce, "get_workspace", lambda mapping, hidden: None)
    target = DecodeTarget(_Core())
    assert target._attn_push_target(_Layer(), 4, 6144) is None
    assert target._moe_push_targets(_Layer(), 4, 6144) is None
    assert target._attn_push_target(_Layer(), 4, 6144) is None
    assert len(log.warning) == 2
    assert "attn_push at 4 rows" in log.warning[0] and "moe_push at 4 rows" in log.warning[1]


class _Layer:
    """The parts of a decoder layer the push targets read."""

    mapping = None


def test_every_dispatch_name_the_target_uses_exists() -> None:
    """A fallback message or selection that names a missing ``dispatch.X`` fails only when that branch runs."""
    target_dir = Path(dispatch.__file__).parent
    missing = []
    for path in sorted(target_dir.rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text())):
            if (
                isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Name)
                and node.value.id == "dispatch"
                and not hasattr(dispatch, node.attr)
            ):
                missing.append(
                    f"{path.relative_to(target_dir)}:{node.lineno}: dispatch.{node.attr}"
                )
    assert not missing, missing


# ------------------------------------------------------------------ the fused index score + top-16 select


class _FssConfig:
    num_index_heads = 1
    sparse_index_dim = 128
    num_kv_heads = 1
    init_blocks = 0
    local_blocks = 1


def _fss_layer():
    """What ``M3Attention._fused_score_select`` reads of its layer."""
    return SimpleNamespace(attn=SimpleNamespace(m3_config=_FssConfig()), layer_idx=5)


def _fss_metadata(rows: int, qlen: int, max_tiles: int = 8192, n_valid: bool = True):
    """A decode span's metadata as the fused score + select reads it (CPU stand-ins, never launched)."""
    return SimpleNamespace(
        msa_idx_k_cache=lambda layer_idx: torch.zeros(4, 1, 128, 128, dtype=torch.float8_e4m3fn),
        msa_worst_case_max_k_tiles=max_tiles,
        msa_n_valid_blocks=torch.ones(rows * qlen, dtype=torch.int32) if n_valid else None,
        msa_block_table=torch.zeros(rows, 4, dtype=torch.int32),
        msa_seq_lens_cuda=torch.full((rows,), 300, dtype=torch.int32),
    )


def _fss(rows: int, qlen: int = 4, dtype=torch.float8_e4m3fn, **md):
    idx_q = torch.zeros(rows * qlen, 128, dtype=dtype)
    return layers.M3Attention._fused_score_select(
        _fss_layer(), idx_q, _fss_metadata(rows, qlen, **md), rows, qlen, rows * qlen
    )


def test_fused_score_select_covers_one_and_two_requests() -> None:
    assert dispatch.FUSED_SCORE_SELECT.keys == frozenset({1, 2})
    assert {c.spec["batch"] for c in dispatch.FUSED_SCORE_SELECT.entry.CELLS} == {1, 2}


@pytest.mark.parametrize(
    "kwargs,reason",
    [
        (dict(dtype=torch.bfloat16), "not FP8"),
        (dict(qlen=5), "not the kernel's geometry"),
        (dict(max_tiles=16384), "do not fit the stage ring"),
        (dict(n_valid=False), "not one decode span"),
    ],
    ids=["bf16_index", "qlen5", "ring", "no_valid_counts"],
)
@pytest.mark.parametrize("rows", [1, 2])
def test_a_covered_span_failing_a_precondition_raises_under_require(rows, kwargs, reason) -> None:
    dispatch.set_require(True)
    with pytest.raises(
        dispatch.FastPathUnavailable, match=rf"'m3_fused_score_select'.* {rows} rows.*{reason}"
    ):
        _fss(rows, **kwargs)


def test_a_covered_span_failing_a_precondition_falls_back_and_logs_once(log) -> None:
    for _ in range(2):
        assert _fss(2, dtype=torch.bfloat16) is None
    assert len(log.warning) == 1 and "m3_fused_score_select at 2 rows" in log.warning[0]


def test_a_capture_without_the_select_scratch_raises_under_require(monkeypatch) -> None:
    monkeypatch.setattr(m3_fused_score_select, "buffers", lambda device, max_tiles: None)
    dispatch.set_require(True)
    with pytest.raises(
        dispatch.FastPathUnavailable, match=r"'m3_fused_score_select'.* 1 rows.*scratch.*captured"
    ):
        _fss(1)


@pytest.mark.parametrize("rows", [3, 8])
def test_a_span_past_two_requests_is_uncovered_in_every_mode(rows, log) -> None:
    dispatch.set_require(True)
    assert _fss(rows) is None
    assert len(log.info) == 1 and "m3_fused_score_select" in log.info[0] and not log.warning


# ------------------------------------------------------------------ the drafter's low-M linears


def test_draft_lowm_keys_are_the_tactic_rows() -> None:
    tactics = draft_lowm_entry.TACTICS
    assert set(tactics) == set(DRAFTER_LINEARS) == set(dispatch.DRAFT_LOWM_ROWS)
    expected = frozenset(
        (nk, m) for nk, table in tactics.items() for m in range(1, table[-1][0] + 1)
    )
    assert dispatch.DRAFT_LOWM.keys == expected
    for nk, table in tactics.items():
        rows = [entry[0] for entry in table]
        assert rows == sorted(rows) and len(set(rows)) == len(rows), nk
        for m in range(1, table[-1][0] + 1):
            assert draft_lowm_entry.tactic_for(*nk, m) is not None, (nk, m)
        assert draft_lowm_entry.tactic_for(*nk, table[-1][0] + 1) is None, nk


def _lowm_operands(nk, m: int):
    """An [N, K] BF16 weight (a broadcast view: the dispatch reads its shape and dtype only) and an [m, K] input."""
    n, k = nk
    weight = torch.zeros(1, 1, dtype=torch.bfloat16).expand(n, k)
    return weight, torch.zeros(m, k, dtype=torch.bfloat16)


def _draft_linear(weight, x, bias=None):
    return m3_draft_lowm.draft_linear(weight, x, bias)


@pytest.mark.parametrize("nk", DRAFTER_LINEARS, ids=str)
def test_draft_linear_without_its_prebuilt_kernel_raises_under_require(monkeypatch, nk) -> None:
    monkeypatch.setattr(m3_draft_lowm, "_BUILT", set())
    weight, x = _lowm_operands(nk, 1)
    dispatch.set_require(True)
    with pytest.raises(
        dispatch.FastPathUnavailable, match=r"'m3_draft_lowm'.* 1 rows.*not compiled before warmup"
    ):
        _draft_linear(weight, x)


def test_draft_linear_without_its_prebuilt_kernel_falls_back_and_logs_once(
    monkeypatch, log
) -> None:
    monkeypatch.setattr(m3_draft_lowm, "_BUILT", set())
    weight, x = _lowm_operands((6144, 1024), 2)
    for _ in range(2):
        assert _draft_linear(weight, x) is None
    assert len(log.warning) == 1 and "m3_draft_lowm at 2 rows" in log.warning[0]


def test_draft_linear_on_a_non_bf16_activation_is_a_miss() -> None:
    weight, x = _lowm_operands((6144, 1024), 1)
    dispatch.set_require(True)
    with pytest.raises(dispatch.FastPathUnavailable, match=r"'m3_draft_lowm'.*not BF16"):
        _draft_linear(weight, x.float())


def test_draft_linear_on_a_strided_activation_keeps_the_in_tree_gemm(log) -> None:
    """The first pass's fc may read a column window of the captured hidden states: uncovered, never raised."""
    weight, x = _lowm_operands((6144, 1024), 2)
    wide = x.new_zeros(2, 2048)
    wide[:, :1024] = x
    assert not wide[:, :1024].is_contiguous()
    dispatch.set_require(True)
    assert _draft_linear(weight, wide[:, :1024]) is None
    assert len(log.info) == 1 and not log.warning


def test_draft_linear_past_its_rows_is_uncovered_in_every_mode(monkeypatch, log) -> None:
    monkeypatch.setattr(m3_draft_lowm, "_BUILT", set())
    dispatch.set_require(True)
    weight, x = _lowm_operands((6144, 1024), 3)  # the o_proj table ends at 2 rows
    assert _draft_linear(weight, x) is None
    weight, x = _lowm_operands((1280, 12288), 9)
    assert _draft_linear(weight, x) is None
    weight, x = _lowm_operands((4096, 1024), 1)  # not a drafter shape
    assert _draft_linear(weight, x) is None
    assert len(log.info) == 3 and not log.warning


def test_draft_linear_with_a_bias_is_not_taken(monkeypatch, log) -> None:
    monkeypatch.setattr(m3_draft_lowm, "_BUILT", set())
    dispatch.set_require(True)
    weight, x = _lowm_operands((6144, 1024), 1)
    assert _draft_linear(weight, x, torch.zeros(6144, dtype=torch.bfloat16)) is None
    assert _draft_linear(weight, x[None]) is None
    assert not log.info and not log.warning


def test_prebuild_drafter_linears_caps_the_drafter_attention(monkeypatch) -> None:
    """The drafter attention gets ``DRAFT_DECODE_MAX_SEQ_LEN_BY_BATCH`` and every drafter linear goes to the
    low-M method before warmup; without a drafter nothing is built."""
    installed = []
    monkeypatch.setattr(m3_draft_lowm, "install", lambda dm: installed.append(dm) or 5)
    monkeypatch.setattr(m3_draft_lowm, "prebuild", lambda dm: 17)
    worker = spec_worker.M3Eagle3Worker.__new__(spec_worker.M3Eagle3Worker)
    assert worker.prebuild_drafter_linears(None) == 0 and not installed
    attn = SimpleNamespace()
    drafter = SimpleNamespace(
        model=SimpleNamespace(midlayer=SimpleNamespace(self_attn=SimpleNamespace(attn=attn)))
    )
    assert worker.prebuild_drafter_linears(drafter) == 17
    assert installed == [drafter]
    assert attn.dense_decode_max_seq_len_by_batch is dispatch.DRAFT_DECODE_MAX_SEQ_LEN_BY_BATCH


# ------------------------------------------------------------------ the fused draft loop's preconditions

_HEAD_ROWS = 64
_HIDDEN = 32


def _drafter(head_weight=None):
    """A drafter with only what M3Eagle3Worker._m3_drafter_supported reads: one plain Eagle3 layer with a gated MLP,
    a whole-table embedding, all-reduced o_proj and down_proj and a vocab-sharded BF16 head (TP8)."""
    mlp = GatedMLP.__new__(GatedMLP)
    torch.nn.Module.__init__(mlp)
    mlp.down_proj = SimpleNamespace(all_reduce=object())
    layer = SimpleNamespace(
        _next_layer_regular=False,
        mlp=mlp,
        self_attn=SimpleNamespace(o_proj=SimpleNamespace(all_reduce=object())),
    )
    model = SimpleNamespace(
        num_layers=1,
        _use_mla=False,
        _eh_proj_before_attn=False,
        midlayer=layer,
        embed_tokens=SimpleNamespace(tp_size=1),
    )
    weight = (
        head_weight
        if head_weight is not None
        else torch.zeros((_HEAD_ROWS, _HIDDEN), dtype=torch.bfloat16)
    )
    lm_head = SimpleNamespace(
        weight=weight, bias=None, tp_mode=TensorParallelMode.COLUMN, padding_size=0, tp_size=8
    )
    return SimpleNamespace(model=model, lm_head=lm_head)


def _worker():
    """An M3Eagle3Worker past the step's verify: the fused verify left the first-pass gather rows and the fused
    first draft inputs; nothing else of the in-tree worker is read."""
    worker = spec_worker.M3Eagle3Worker.__new__(spec_worker.M3Eagle3Worker)
    worker.mapping = SimpleNamespace(tp_size=8, tp_rank=1, enable_attention_dp=False)
    worker._d2t = None
    worker._m3_draft_head = None
    worker._m3_first_gather_ids = torch.zeros(1, dtype=torch.int64)
    worker._m3_fused_first_inputs = True
    return worker


_ATTN_MD = SimpleNamespace(kv_lens_cuda=None)
_SPEC_MD = SimpleNamespace(runtime_draft_len=3)


@pytest.fixture
def no_gpu_draft_head(monkeypatch):
    """The fused head's CUDA-only lookups replaced: a 148-SM device, and a built exchange workspace."""
    monkeypatch.setattr(m3_draft_step, "_num_sms", lambda: 148)
    monkeypatch.setattr(m3_draft_step, "_WS", object())


def test_draft_loop_without_the_prebuilt_head_raises_under_require(no_gpu_draft_head) -> None:
    dispatch.set_require(True)
    with pytest.raises(
        dispatch.FastPathUnavailable, match=r"'m3_draft_step'.* 4 rows.*prebuild_draft_head"
    ):
        _worker()._m3_fused_loop_ok(_ATTN_MD, _SPEC_MD, 4, _drafter())


def test_draft_loop_without_the_prebuilt_head_falls_back_and_logs_once(
    no_gpu_draft_head, log
) -> None:
    worker = _worker()
    for _ in range(2):
        assert not worker._m3_fused_loop_ok(_ATTN_MD, _SPEC_MD, 4, _drafter())
    assert len(log.warning) == 1 and "m3_draft_step at 4 rows" in log.warning[0]


def test_draft_loop_with_a_head_built_for_another_weight_raises_under_require(
    no_gpu_draft_head,
) -> None:
    worker = _worker()
    assert worker.prebuild_draft_head(_drafter())
    dispatch.set_require(True)
    with pytest.raises(dispatch.FastPathUnavailable, match=r"'m3_draft_step'"):
        worker._m3_fused_loop_ok(_ATTN_MD, _SPEC_MD, 2, _drafter())


def test_draft_loop_takes_a_covered_step_once_the_head_is_built(no_gpu_draft_head) -> None:
    dispatch.set_require(True)
    drafter = _drafter()
    worker = _worker()
    assert worker.prebuild_draft_head(drafter)
    head = worker._m3_draft_head
    assert head.weight is drafter.lm_head.weight and head.vocab_offset == _HEAD_ROWS
    assert torch.equal(draft_step_entry.from_fragments(head.wf), drafter.lm_head.weight)
    for nreq in range(1, 9):
        assert worker._m3_fused_loop_ok(_ATTN_MD, _SPEC_MD, nreq, drafter)


def test_draft_loop_streams_the_bf16_fragments_only(no_gpu_draft_head) -> None:
    """The tp8 draft head is the fragment-ordered BF16 copy at every K (no coded copy is built, whatever K), it
    decodes to the weight bit for bit, and the op refuses a coded stream."""
    dispatch.set_require(True)
    g = torch.Generator().manual_seed(11)
    weight = (torch.randn((_HEAD_ROWS, 128), generator=g) * 0.02).to(torch.bfloat16)
    drafter = _drafter(weight)
    worker = _worker()
    assert worker.prebuild_draft_head(drafter)
    head = worker._m3_draft_head
    assert not hasattr(head, "coded") and not hasattr(head, "exc_off")
    assert head.weight is weight and head.vocab_offset == _HEAD_ROWS
    decoded = draft_step_entry.head_weight(head.wf, 128)
    assert torch.equal(decoded.view(torch.int16), weight.view(torch.int16))
    with pytest.raises(NotImplementedError, match="BF16 fragments only"):
        draft_step_entry.head_weight(head.wf, 128, torch.zeros(5, dtype=torch.int32))
    assert all(not c.spec["coded"] for c in draft_step_entry.m3_draft_step.CELLS)
    for nreq in range(1, 9):
        assert worker._m3_fused_loop_ok(_ATTN_MD, _SPEC_MD, nreq, drafter)


def test_draft_loop_past_its_requests_is_uncovered_in_every_mode(no_gpu_draft_head, log) -> None:
    dispatch.set_require(True)
    drafter = _drafter()
    worker = _worker()
    assert worker.prebuild_draft_head(drafter)
    assert not worker._m3_fused_loop_ok(_ATTN_MD, _SPEC_MD, 9, drafter)
    assert not worker._m3_fused_loop_ok(_ATTN_MD, SimpleNamespace(runtime_draft_len=2), 4, drafter)
    assert len(log.info) == 2 and not log.warning


def test_draft_loop_needs_the_fused_first_inputs(no_gpu_draft_head, log) -> None:
    """A step whose first draft inputs came from the in-tree path (or that the fused verify did not prepare) drafts
    with the in-tree loop, silently: it is not a draft-step shape."""
    dispatch.set_require(True)
    drafter = _drafter()
    worker = _worker()
    assert worker.prebuild_draft_head(drafter)
    worker._m3_fused_first_inputs = False
    assert not worker._m3_fused_loop_ok(_ATTN_MD, _SPEC_MD, 4, drafter)
    worker._m3_fused_first_inputs, worker._m3_first_gather_ids = True, None
    assert not worker._m3_fused_loop_ok(_ATTN_MD, _SPEC_MD, 4, drafter)
    assert not log.info and not log.warning


def test_an_unsupported_drafter_is_uncovered(no_gpu_draft_head, log) -> None:
    dispatch.set_require(True)
    worker = _worker()
    padded = _drafter()
    padded.lm_head.padding_size = 16
    two_layers = _drafter()
    two_layers.model.num_layers = 2
    for drafter in (padded, two_layers):
        assert not worker.prebuild_draft_head(drafter)
        assert not worker._m3_fused_loop_ok(_ATTN_MD, _SPEC_MD, 4, drafter)
    assert log.info and not log.warning


def test_a_capture_without_the_exchange_workspace_raises_under_require(monkeypatch) -> None:
    monkeypatch.setattr(m3_draft_step, "_num_sms", lambda: 148)
    monkeypatch.setattr(m3_draft_step, "_WS", None)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    drafter = _drafter()
    worker = _worker()
    assert worker.prebuild_draft_head(drafter)
    dispatch.set_require(True)
    with pytest.raises(dispatch.FastPathUnavailable, match=r"'m3_draft_step'.*workspace.*captured"):
        worker._m3_fused_loop_ok(_ATTN_MD, _SPEC_MD, 4, drafter)


# ------------------------------------------------------------------ the fc fork (never early at tp8)


class _CaptureMetadata:
    def __init__(self, nreq: int):
        self.layers_to_capture = (1, 29, 56)
        self.num_capture_tokens = nreq * 4
        self.runtime_draft_len = 3
        self.captured = []

    def maybe_capture_hidden_states(self, layer_id, hidden_states, residual) -> None:
        self.captured.append(layer_id)


def _fc_early_worker(monkeypatch, eligible: bool = True):
    worker = spec_worker.M3Eagle3Worker.__new__(spec_worker.M3Eagle3Worker)
    worker._m3_capture_norm = None
    worker._m3_fc_norm_w = None
    worker._m3_fc_early_drafter = None
    worker._m3_fc_early_stream = None
    forks = []
    monkeypatch.setattr(worker, "tail_eligible", lambda *args: eligible)

    def fork(md, num_tokens, draft_model):
        forks.append((num_tokens, draft_model))
        return f"stream-{len(forks)}"

    monkeypatch.setattr(worker, "_m3_fc_fork", fork)
    return worker, forks


@pytest.mark.parametrize("nreq,eligible", [(1, True), (2, True), (3, True), (8, True), (1, False)])
def test_no_early_fork_at_tp8(monkeypatch, nreq, eligible) -> None:
    """``FC_EARLY_MAX_ROWS`` is 0: no verify (not even one request's 4 rows) forks the drafter's fc at the last
    captured layer; every captured layer still captures and the fc forks beside the verify head instead."""
    worker, forks = _fc_early_worker(monkeypatch, eligible)
    md = _CaptureMetadata(nreq)
    worker._m3_fc_early_stream = "stale"
    worker.arm_capture_norm(SimpleNamespace(num_seqs=nreq), md, object(), object())
    assert worker._m3_fc_early_drafter is None
    for layer_id in md.layers_to_capture:
        worker.capture_hidden_states(md, layer_id, None)
    worker.capture_hidden_states(md, spec_worker.FINAL_HIDDEN_LAYER_ID, None)
    assert md.captured == [1, 29, 56, spec_worker.FINAL_HIDDEN_LAYER_ID]
    assert not forks and worker._m3_fc_early_stream is None


def test_no_early_fork_with_a_final_state_capture(monkeypatch) -> None:
    """The final-state column is written after the model returns, so captures that include it keep the fc at its
    stock place instead of forking at the last captured decoder layer."""
    worker, forks = _fc_early_worker(monkeypatch)
    md = _CaptureMetadata(1)
    md.layers_to_capture = (spec_worker.FINAL_HIDDEN_LAYER_ID, 29, 56)
    worker.arm_capture_norm(SimpleNamespace(num_seqs=1), md, object(), object())
    for layer_id in (29, 56, spec_worker.FINAL_HIDDEN_LAYER_ID):
        worker.capture_hidden_states(md, layer_id, None)
    assert not forks and worker._m3_fc_early_stream is None


# ------------------------------------------------------------------ the NVFP4 producer's coverage


def test_the_producer_covers_the_tp8_nvfp4_layers_only() -> None:
    """``dispatch.PRODUCER`` covers a sparse layer on an NVFP4 cache at the TP8 heads (8 query, one KV and one
    index head); elsewhere the built-in producers run without a miss."""
    covered = layers.producer_pw_covered
    assert covered(True, 8, 1, 1)
    assert not covered(False, 8, 1, 1)
    assert not covered(True, 16, 1, 1)
    assert not covered(True, 8, 2, 2)
    assert dispatch.PRODUCER.keys == frozenset({"nvfp4_tp8"})
    assert dispatch.PRODUCER.covers("nvfp4_tp8")
    assert not dispatch.PRODUCER.covers("nvfp4_tp4")


class _FakeMsaAttn:
    indexer_kv_dtype = "fp8"


def _producer_layer(monkeypatch, nvfp4: bool = True, heads: int = 8):
    """What the producer dispatch reads before its first precondition: a fused-projection MSA sparse layer."""
    monkeypatch.setattr(layers, "MiniMaxM3MsaSparseAttention", _FakeMsaAttn)
    return SimpleNamespace(
        enable_fused_qkv_index_projection=True,
        attn=_FakeMsaAttn(),
        main_kv_is_fp8=not nvfp4,
        main_kv_is_nvfp4=nvfp4,
        num_heads=heads,
        num_key_value_heads=1,
        sparse_num_index_heads=1,
        pos_embd_params=None,
    )


def _produce(layer):
    # The packed TP8 projection: 8 query heads, K, V, index Q and index K of 128 each.
    packed = torch.zeros(4, 12 * 128, dtype=torch.bfloat16)
    return layers.M3Attention._fused_fp8_qkv_indexer_norm_rope_kv_insert(
        layer, packed, None, SimpleNamespace()
    )


def test_a_covered_producer_layer_missing_a_precondition_raises_under_require(monkeypatch) -> None:
    """A TP8 NVFP4 sparse layer whose step lacks a producer precondition (here the position ids) is a miss."""
    dispatch.set_require(True)
    with pytest.raises(
        dispatch.FastPathUnavailable, match=r"'m3_producer' covers this step at 4 rows"
    ):
        _produce(_producer_layer(monkeypatch))


def test_a_covered_producer_layer_without_the_fused_projection_raises_under_require(
    monkeypatch,
) -> None:
    layer = _producer_layer(monkeypatch)
    layer.enable_fused_qkv_index_projection = False
    dispatch.set_require(True)
    with pytest.raises(dispatch.FastPathUnavailable, match="fuse_qkv_index_projection is disabled"):
        layers.M3Attention._check_producer_projection(layer)


def test_an_uncovered_layer_without_the_fused_projection_is_not_a_miss(monkeypatch) -> None:
    layer = _producer_layer(monkeypatch, nvfp4=False)
    layer.enable_fused_qkv_index_projection = False
    dispatch.set_require(True)
    layers.M3Attention._check_producer_projection(layer)


def test_a_covered_producer_miss_falls_back_and_logs_once(monkeypatch, log) -> None:
    layer = _producer_layer(monkeypatch)
    assert _produce(layer) is None and _produce(layer) is None
    assert len(log.warning) == 1 and "m3_producer at 4 rows" in log.warning[0]


@pytest.mark.parametrize("nvfp4,heads", [(False, 8), (True, 16)], ids=["fp8_cache", "tp4_heads"])
def test_an_uncovered_producer_layer_falls_back_silently(monkeypatch, log, nvfp4, heads) -> None:
    dispatch.set_require(True)
    assert _produce(_producer_layer(monkeypatch, nvfp4, heads)) is None
    assert not log.warning


def test_a_covered_producer_layer_with_the_wrong_pool_layout_is_not_supported() -> None:
    """The pool-layout check the covered branch takes before the native fallback (then a miss)."""
    pool = torch.zeros(4, 2, 1, 128, 32, dtype=torch.uint8)
    scales = torch.zeros(4, 2, 1, 128, 8, dtype=torch.uint8)
    assert not layers.producer_pw_supported(8, 1, 1, pool, scales)
    good = torch.zeros(4, 2, 1, 128, 64, dtype=torch.uint8)
    assert layers.producer_pw_supported(8, 1, 1, good, scales)
    assert not layers.producer_pw_supported(16, 1, 1, good, scales)
    assert not layers.producer_pw_supported(
        8, 1, 1, good, scales.transpose(3, 4).contiguous().transpose(3, 4)
    )
