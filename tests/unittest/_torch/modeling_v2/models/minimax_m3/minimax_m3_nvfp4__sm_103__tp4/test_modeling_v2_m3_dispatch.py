# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The decode target's dispatch table (``modeling.py``, imported here as ``dispatch``): its row ranges against the
catalog cells that certify them and the kernels' own limits, the selection for every decode row count, and the miss
/ uncovered policy.

CPU only: nothing here launches a kernel.
"""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_nvfp4__sm_103__tp4 import (  # noqa: E501
    modeling as dispatch,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_nvfp4__sm_103__tp4.ops import (  # noqa: E501
    m3_attn_mega_c1,
    m3_draft_step,
    m3_fused_attn,
    m3_mc_allreduce,
    m3_shard_accept,
    m3_target_head,
    triton_sparse_decode,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_nvfp4__sm_103__tp4.ops import (  # noqa: E501
    mxfp8_low_m_gemm as low_m,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_nvfp4__sm_103__tp4.ops.gemm import (  # noqa: E501
    m3_target_head as target_head_entry,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_nvfp4__sm_103__tp4.ops.gemm import (  # noqa: E501
    mxfp8_low_m_gemm as low_m_entry,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_nvfp4__sm_103__tp4.ops.gemm.m3_mx_gemv import (  # noqa: E501
    DENSE_DOWN,
    DENSE_GATE_UP,
    O_PROJ,
    QKV,
    QKV_SPARSE,
    SHARED_DOWN,
    SHARED_GATE_UP,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_nvfp4__sm_103__tp4.ops.mx_gemv import (  # noqa: E501
    op as mx_gemv,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_nvfp4__sm_103__tp4.ops.spec import (  # noqa: E501
    m3_draft_step as draft_step_entry,
)
from tensorrt_llm._torch.modules.linear import TensorParallelMode

PROJECTIONS = (QKV_SPARSE, QKV, O_PROJ, SHARED_GATE_UP, SHARED_DOWN, DENSE_GATE_UP, DENSE_DOWN)
ROWS = range(1, dispatch.DECODE_MAX_TOKENS + 2)  # every decode row count and the first one past it


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


# ------------------------------------------------------------------ the ranges against the kernels' own limits


@pytest.mark.parametrize("nk", PROJECTIONS, ids=str)
def test_gemv_rows_are_the_rows_the_kernel_serves(nk) -> None:
    n, k = nk
    served = {m for m in range(1, mx_gemv.MAX_TOKENS + 1) if mx_gemv.serves(n, k, m, False)}
    assert dispatch.MX_GEMV_ROWS[nk] == served
    served_bf16 = [m for m in range(1, mx_gemv.MAX_TOKENS + 1) if mx_gemv.serves(n, k, m, True)]
    assert served_bf16 == list(range(1, dispatch.MX_GEMV_BF16_MAX_TOKENS[nk] + 1))


@pytest.mark.parametrize("nk", sorted(dispatch.LOW_M_GEMM_MAX_TOKENS), ids=str)
def test_low_m_gemm_rows_are_its_configuration_limit(nk) -> None:
    warps, cluster_k, max_m, row_tiles = low_m.get_config(*nk)
    assert dispatch.LOW_M_GEMM_MAX_TOKENS[nk] == max_m
    for m in range(1, max_m + 1):
        assert low_m._smem_bytes(m, nk[1], warps, cluster_k, row_tiles) <= low_m._MAX_DYNAMIC_SMEM


def test_the_low_m_gemm_has_a_range_for_every_configuration() -> None:
    assert set(dispatch.LOW_M_GEMM_MAX_TOKENS) == set(low_m._CONFIGS)


def test_c1_takes_exactly_its_range() -> None:
    for qlen in range(1, dispatch.FUSED_ATTN_MAX_QUERY_LEN + 1):
        for reqs in range(1, dispatch.FUSED_ATTN_MAX_TOKENS + 1):
            assert m3_attn_mega_c1.c1_applies(qlen, reqs) == dispatch.ATTN_C1.covers((qlen, reqs))


def test_every_fused_attention_shape_has_a_compiled_geometry() -> None:
    for qlen, reqs in dispatch.FUSED_DECODE_ATTN.keys:
        assert (qlen, *m3_fused_attn.default_geometry(qlen, reqs)) in m3_fused_attn.GEOMETRIES


def test_the_plans_are_the_all_reduce_buckets() -> None:
    assert tuple(p.max_tokens for p in dispatch.DECODE_PLANS) == m3_mc_allreduce.BUCKETS
    assert dispatch.DECODE_MAX_TOKENS == m3_mc_allreduce.MAX_TOKENS == mx_gemv.MAX_TOKENS


def test_the_thresholds_are_the_kernels() -> None:
    assert dispatch.SPARSE_DECODE_MQ_MIN_TOKENS == triton_sparse_decode.MQ_MIN_TOKENS
    assert dispatch.MOE16_MAX_TOKENS < dispatch.ATTN_PUSH_MAX_TOKENS <= dispatch.DECODE_MAX_TOKENS
    verify_rows = dispatch.SHARD_ACCEPT_MAX_REQUESTS * (dispatch.SHARD_ACCEPT_DRAFT_LEN + 1)
    assert verify_rows == dispatch.DECODE_MAX_TOKENS == m3_shard_accept.SHARD_ROWS


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
    if rows <= 13:
        return "m3_moe16_engine"
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


@pytest.mark.parametrize("rows", ROWS)
def test_linear_selection(rows) -> None:
    bf16_max = {
        QKV_SPARSE: 16,
        QKV: 16,
        O_PROJ: 32,
        SHARED_GATE_UP: 16,
        SHARED_DOWN: 16,
        DENSE_GATE_UP: 8,
        DENSE_DOWN: 16,
    }
    for nk in PROJECTIONS:
        prequant_gemv = rows <= 48 and not (nk == SHARED_DOWN and 17 <= rows <= 32)
        assert dispatch.linear_path(*nk, rows, True, True) == (
            "m3_mx_gemv" if prequant_gemv else "flashinfer_mxfp8_gemm"
        )
        for low_m_gemm in (False, True):
            assert dispatch.linear_path(*nk, rows, False, low_m_gemm) == (
                "m3_mx_gemv_bf16" if rows <= bf16_max[nk] else "mxfp8_gemm"
            )
    # A projection outside the table takes the generic GEMM at every row count.
    assert dispatch.linear_path(4096, 4096, rows, True, True) == "flashinfer_mxfp8_gemm"
    assert dispatch.linear_path(4096, 4096, rows, False, True) == "mxfp8_gemm"


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


def test_target_head_selection() -> None:
    """The coded target head takes every verify of at most 8 rows (one or two requests at draft length 3); the early
    fc fork the same verifies."""
    assert dispatch.TARGET_HEAD_MAX_ROWS == 8 == dispatch.FC_EARLY_MAX_ROWS
    for rows in ROWS:
        assert dispatch.TARGET_HEAD.covers(rows) == (rows <= 8)


@pytest.mark.parametrize("rows", ROWS)
def test_attention_shapes(rows) -> None:
    for qlen in range(1, 9):
        if rows % qlen:
            continue
        reqs = rows // qlen
        assert dispatch.FUSED_DECODE_ATTN.covers((qlen, reqs)) == (rows <= 16)
        assert dispatch.ATTN_MEGA.covers((qlen, reqs)) == (qlen == 1 and reqs == 1)
        assert dispatch.ATTN_C1.covers((qlen, reqs)) == (qlen == 6 and reqs == 1)


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


# ------------------------------------------------------------------ the phase split


def test_the_core_selects_the_target_on_the_context_count_alone() -> None:
    """Every pure generation step is decode's, whatever its width; a step with context rows (a mixed one too) is
    prefill's. Nothing but ``num_contexts`` is read: the metadata stand-in carries no other field."""
    core = SimpleNamespace(_prefill="prefill", _decode="decode")
    select = dispatch.MiniMaxM3Tp4ModelingV2._select_target
    assert select(core, SimpleNamespace(num_contexts=0)) == "decode"
    assert select(core, SimpleNamespace(num_contexts=1)) == "prefill"
    assert select(core, SimpleNamespace(num_contexts=7)) == "prefill"


_DECODE_FIELDS = (
    "num_tokens",
    "num_seqs",
    "kv_cache_manager",
    "msa_idx_k_cache",
    "msa_out_cache_loc",
    "msa_decode_span",
    "msa_block_table",
    "msa_seq_lens_cuda",
    "_msa_short_ok",
    "msa_short_table",
    "msa_proxy_max_score_view",
    "msa_worst_case_max_k_tiles",
    "msa_n_valid_blocks",
)


def _decode_metadata(num_tokens: int) -> SimpleNamespace:
    """The decode projection's fields and nothing else (no ``num_contexts`` / ``num_ctx_tokens``)."""
    md = SimpleNamespace(**{name: object() for name in _DECODE_FIELDS})
    md.num_tokens = num_tokens
    return md


def test_the_decode_projection_states_the_phase_and_reads_no_count_back() -> None:
    md = _decode_metadata(4)
    step = dispatch.DecodeTarget(SimpleNamespace()).step_args(md)
    assert step["num_contexts"] == 0 and step["num_ctx_tokens"] == 0
    assert step["metadata"] is md and step["msa_short_ok"] is md._msa_short_ok


@pytest.mark.parametrize(
    "rows,embeds", [(49, False), (64, False), (4, True)], ids=["49_rows", "64_rows", "embeds"]
)
def test_a_wide_generation_step_runs_every_layer_generically_in_decode(
    monkeypatch, rows, embeds
) -> None:
    """Past the decode kernels' rows (or with embeddings) the decode target runs its own generic layer for every
    layer, with its own projection, and never the decode-kernel layer."""
    layers, seen = 3, []
    core = SimpleNamespace(num_layers=layers, w={"embed": None})
    monkeypatch.setattr(dispatch, "m3_embedding", lambda ids, table: ("embedded", int(ids.numel())))

    def generic(self, i, position_ids, hidden, residual, step, spec_metadata, capture):
        seen.append((i, step["num_contexts"], step["metadata"]))
        return ("hidden", i), ("residual", i)

    def kernels(self, *args):
        raise AssertionError("the decode-kernel layer ran on a wide step")

    monkeypatch.setattr(dispatch.DecodeTarget, "_generic_layer", generic)
    monkeypatch.setattr(dispatch.DecodeTarget, "_layer", kernels)
    md = _decode_metadata(rows)
    ids = torch.arange(rows, dtype=torch.int32)
    out = dispatch.DecodeTarget(core).forward(
        md, None if embeds else ids, None, ("embeds",) if embeds else None, None, None
    )
    assert out == ("hidden", layers - 1)
    assert seen == [(i, 0, md) for i in range(layers)]


def test_uncovered_never_raises_and_logs_once(log) -> None:
    dispatch.set_require(True)
    for _ in range(3):
        dispatch.uncovered("fused_decode_attn", "a request beyond 16 KV blocks")
    assert len(log.info) == 1 and not log.warning


class _Core:
    """The parts of the core ``DecodeTarget``'s embedding and push targets read: no embedding owners built."""

    def __init__(self):
        self.embed_owners = {}
        self.w = {"embed": torch.arange(64 * 4, dtype=torch.float32).view(64, 4)}
        self.mapping = None
        self.hidden = 6144


@pytest.mark.parametrize("rows", (1, 8))
def test_embedding_without_its_owner_raises_under_require(rows) -> None:
    dispatch.set_require(True)
    with pytest.raises(
        dispatch.FastPathUnavailable, match=rf"'embed_firstnorm'.* {rows} rows.*owner"
    ):
        dispatch.DecodeTarget(_Core())._embed(torch.arange(rows, dtype=torch.int32), rows)


def test_embedding_without_its_owner_falls_back_and_logs_once(log) -> None:
    target = dispatch.DecodeTarget(_Core())
    for _ in range(2):
        hidden, residual = target._embed(torch.arange(3, dtype=torch.int32), 3)
    assert residual is None and hidden.shape == (3, 4)
    assert len(log.warning) == 1 and "embed_firstnorm at 3 rows" in log.warning[0]


def test_embedding_past_its_rows_is_uncovered_in_every_mode(log) -> None:
    dispatch.set_require(True)
    hidden, residual = dispatch.DecodeTarget(_Core())._embed(torch.arange(9, dtype=torch.int32), 9)
    assert residual is None and hidden.shape == (9, 4)
    assert len(log.info) == 1 and "embed_firstnorm" in log.info[0] and not log.warning


def test_a_push_without_its_workspace_raises_under_require(monkeypatch) -> None:
    monkeypatch.setattr(m3_mc_allreduce, "get_workspace", lambda mapping, hidden: None)
    dispatch.set_require(True)
    with pytest.raises(dispatch.FastPathUnavailable, match=r"'attn_push'.* 4 rows"):
        dispatch.DecodeTarget(_Core())._attn_push_target(4)


def test_a_push_without_its_workspace_falls_back_and_logs_once(monkeypatch, log) -> None:
    monkeypatch.setattr(m3_mc_allreduce, "get_workspace", lambda mapping, hidden: None)
    target = dispatch.DecodeTarget(_Core())
    assert target._attn_push_target(4) is None
    assert target._moe_push_targets(4) is None
    assert target._attn_push_target(4) is None
    assert len(log.warning) == 2
    assert "attn_push at 4 rows" in log.warning[0] and "moe_push at 4 rows" in log.warning[1]


def test_the_target_has_no_dispatch_module_left() -> None:
    """The dispatch tables live in ``modeling.py``: no module of the target still reaches a ``dispatch`` module."""
    target_dir = Path(dispatch.__file__).parent
    assert not (target_dir / "dispatch.py").exists()
    stale = []
    for path in sorted(target_dir.rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ImportFrom) and any(a.name == "dispatch" for a in node.names):
                stale.append(f"{path.relative_to(target_dir)}:{node.lineno}")
            if (
                isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Name)
                and node.value.id == "dispatch"
            ):
                stale.append(f"{path.relative_to(target_dir)}:{node.lineno}: dispatch.{node.attr}")
    assert not stale, stale


def test_low_m_gemm_cells_follow_its_configurations() -> None:
    for (n, k), (warps, cluster_k, row_tiles, max_m) in low_m_entry.CONFIGS.items():
        assert low_m.get_config(n, k) == (warps, cluster_k, max_m, row_tiles)
        assert dispatch.LOW_M_GEMM_MAX_TOKENS[(n, k)] == max_m


# ------------------------------------------------------------------ the fused draft loop's preconditions

_HEAD_ROWS = 64
_HIDDEN = 32


def _drafter(head_weight=None):
    """A drafter with only what M3Eagle3Worker._m3_drafter_supported reads: its vocab-sharded BF16 head (the
    drafter's one-layer shape is validated when ``M3Eagle3Draft`` is built)."""
    weight = (
        head_weight
        if head_weight is not None
        else torch.zeros((_HEAD_ROWS, _HIDDEN), dtype=torch.bfloat16)
    )
    lm_head = SimpleNamespace(
        weight=weight, bias=None, tp_mode=TensorParallelMode.COLUMN, padding_size=0, tp_size=4
    )
    return SimpleNamespace(lm_head=lm_head)


def _worker():
    """An M3Eagle3Worker past the step's verify: the fused verify left the first-pass gather rows and the fused
    first draft inputs; nothing else of the in-tree worker is read."""
    worker = dispatch.M3Eagle3Worker.__new__(dispatch.M3Eagle3Worker)
    worker.mapping = SimpleNamespace(tp_size=4, tp_rank=1, enable_attention_dp=False)
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
    with pytest.raises(dispatch.FastPathUnavailable, match=r"'m3_draft_step'.* 4 rows.*prebuild"):
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


def test_draft_loop_streams_the_coded_head_once_built(no_gpu_draft_head) -> None:
    """A drafter head whose K the coding covers is prebuilt in its 12-bit coding (decoding to the weight bit for bit)
    and serves every covered step; one whose coding fails its round trip keeps the BF16 fragments (fail closed)."""
    dispatch.set_require(True)
    g = torch.Generator().manual_seed(11)
    weight = (torch.randn((_HEAD_ROWS, 128), generator=g) * 0.02).to(torch.bfloat16)
    drafter = _drafter(weight)
    worker = _worker()
    assert worker.prebuild_draft_head(drafter)
    head = worker._m3_draft_head
    assert head.coded and head.weight is weight and head.vocab_offset == _HEAD_ROWS
    decoded = draft_step_entry.head_weight(
        head.wf, 128, head.exc_off, head.exc_pos, head.exc_w, head.exp_base
    )
    assert torch.equal(decoded.view(torch.int16), weight.view(torch.int16))
    for nreq in range(1, 9):
        assert worker._m3_fused_loop_ok(_ATTN_MD, _SPEC_MD, nreq, drafter)


def test_draft_head_that_fails_its_round_trip_keeps_the_bf16_fragments(
    no_gpu_draft_head, monkeypatch
) -> None:
    real = target_head_entry.encode_chunked

    def corrupt(weight, *args, **kwargs):
        packed, base, exc_off, exc_pos, exc_w = real(weight)
        packed = packed.clone()
        packed[5] ^= 2
        return packed, base, exc_off, exc_pos, exc_w

    monkeypatch.setattr(m3_draft_step, "encode_chunked", corrupt)
    dispatch.set_require(True)
    weight = torch.ones((_HEAD_ROWS, 128), dtype=torch.bfloat16)
    drafter = _drafter(weight)
    worker = _worker()
    assert worker.prebuild_draft_head(drafter)
    head = worker._m3_draft_head
    assert not head.coded and head.exc_off is None
    assert torch.equal(draft_step_entry.from_fragments(head.wf), weight)
    assert worker._m3_fused_loop_ok(_ATTN_MD, _SPEC_MD, 4, drafter)


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
    for drafter in (padded,):
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


# ------------------------------------------------------------------ the coded target head's preconditions

_TARGET_ROWS = 64
_TARGET_K = 128


def _target_lm_head(weight=None):
    if weight is None:
        g = torch.Generator().manual_seed(7)
        weight = (torch.randn(_TARGET_ROWS, _TARGET_K, generator=g) * 0.02).to(torch.bfloat16)
    return SimpleNamespace(
        weight=weight, bias=None, tp_mode=TensorParallelMode.COLUMN, padding_size=0, tp_size=4
    )


@pytest.fixture
def no_gpu_target_head(monkeypatch):
    monkeypatch.setattr(m3_target_head, "_num_sms", lambda: 148)


def _target_worker():
    worker = dispatch.M3Eagle3Worker.__new__(dispatch.M3Eagle3Worker)
    worker._m3_target_head = None
    return worker


def test_target_head_without_the_prebuilt_copy_raises_under_require() -> None:
    dispatch.set_require(True)
    with pytest.raises(dispatch.FastPathUnavailable, match=r"'m3_target_head'.* 4 rows.*prebuild"):
        _target_worker()._m3_coded_head(_target_lm_head(), 4)


def test_target_head_without_the_prebuilt_copy_falls_back_and_logs_once(log) -> None:
    worker = _target_worker()
    for _ in range(2):
        assert worker._m3_coded_head(_target_lm_head(), 8) is None
    assert len(log.warning) == 1 and "m3_target_head at 8 rows" in log.warning[0]


def test_target_head_takes_a_covered_verify_once_built(no_gpu_target_head) -> None:
    """The prebuilt copy decodes to the head's weight bit for bit and serves every covered row count; a copy built
    for another weight is a miss."""
    dispatch.set_require(True)
    lm_head = _target_lm_head()
    worker = _target_worker()
    assert worker.prebuild_target_head(lm_head)
    head = worker._m3_target_head
    assert head.coded and head.weight is lm_head.weight and head.grid == 148
    decoded = target_head_entry.decode(
        head.w, head.base, head.exc_off, head.exc_pos, head.exc_w, _TARGET_ROWS, _TARGET_K
    )
    assert torch.equal(decoded.view(torch.int16), lm_head.weight.view(torch.int16))
    for rows in range(1, 9):
        assert worker._m3_coded_head(lm_head, rows) is head
    with pytest.raises(dispatch.FastPathUnavailable, match=r"'m3_target_head'"):
        worker._m3_coded_head(_target_lm_head(lm_head.weight.clone()), 4)


def test_target_head_past_its_rows_is_uncovered_in_every_mode(no_gpu_target_head, log) -> None:
    dispatch.set_require(True)
    lm_head = _target_lm_head()
    worker = _target_worker()
    assert worker.prebuild_target_head(lm_head)
    for rows in (9, 12, 32, 48):
        assert worker._m3_coded_head(lm_head, rows) is None
    assert log.info and not log.warning


def test_target_head_that_fails_its_round_trip_keeps_the_bf16_gemm(
    no_gpu_target_head, monkeypatch, log
) -> None:
    """Fail closed: a coding that does not decode to the weight bit for bit is dropped at prebuild (a warning), and the
    verify runs the in-tree GEMM without a miss, in every mode."""
    real = target_head_entry.encode_chunked

    def corrupt(weight, *args, **kwargs):
        packed, base, exc_off, exc_pos, exc_w = real(weight)
        packed = packed.clone()
        packed[3] ^= 1
        return packed, base, exc_off, exc_pos, exc_w

    monkeypatch.setattr(m3_target_head, "encode_chunked", corrupt)
    dispatch.set_require(True)
    lm_head = _target_lm_head()
    worker = _target_worker()
    assert not worker.prebuild_target_head(lm_head)
    assert not worker._m3_target_head.coded
    assert len(log.warning) == 1 and "bit for bit" in log.warning[0]
    assert worker._m3_coded_head(lm_head, 4) is None
    assert any("round trip" in line for line in log.info)


def test_an_unsupported_target_head_is_not_coded(no_gpu_target_head) -> None:
    worker = _target_worker()
    padded = _target_lm_head()
    padded.padding_size = 16
    narrow = _target_lm_head(torch.zeros((_TARGET_ROWS, 96), dtype=torch.bfloat16))
    for lm_head in (padded, narrow):
        assert not worker.prebuild_target_head(lm_head)
        assert worker._m3_target_head is None


# ------------------------------------------------------------------ the early fc fork


class _CaptureMetadata:
    def __init__(self, nreq: int):
        self.layers_to_capture = (1, 29, 56)
        self.num_capture_tokens = nreq * 4
        self.runtime_draft_len = 3
        self.captured = []

    def maybe_capture_hidden_states(self, layer_id, hidden_states, residual) -> None:
        self.captured.append(layer_id)


def _fc_early_worker(monkeypatch, eligible: bool = True):
    worker = dispatch.M3Eagle3Worker.__new__(dispatch.M3Eagle3Worker)
    worker._m3_capture_norm = None
    worker._m3_fc_norm_w = None
    worker._m3_fc_early_drafter = None
    worker._m3_fc_early_stream = None
    forks = []
    monkeypatch.setattr(worker, "tail_eligible", lambda *args: eligible)

    def fork(md, num_tokens, draft_model, after=None):
        forks.append((num_tokens, draft_model, after))
        return f"stream-{len(forks)}"

    monkeypatch.setattr(worker, "_m3_fc_fork", fork)
    return worker, forks


@pytest.mark.parametrize("nreq", [1, 2])
def test_the_fc_is_forked_at_the_last_captured_layer(monkeypatch, nreq) -> None:
    """A fused-verify step of at most 8 verify rows forks the drafter's fc once, right after the last captured layer
    (every captured layer still captures), and the fork is left for the verify to join."""
    worker, forks = _fc_early_worker(monkeypatch)
    md = _CaptureMetadata(nreq)
    drafter = object()
    worker.arm_capture_norm(SimpleNamespace(num_seqs=nreq), md, object(), drafter)
    for layer_id in md.layers_to_capture[:-1]:
        worker.capture_hidden_states(md, layer_id, None)
        assert not forks
    worker.capture_hidden_states(md, md.layers_to_capture[-1], None)
    worker.capture_hidden_states(md, dispatch.FINAL_HIDDEN_LAYER_ID, None)
    assert md.captured == [1, 29, 56, dispatch.FINAL_HIDDEN_LAYER_ID]
    assert forks == [(4 * nreq, drafter, None)]
    assert worker._m3_fc_early_stream == "stream-1" and worker._m3_fc_early_drafter is None


def test_no_early_fork_with_a_final_state_capture(monkeypatch) -> None:
    """The final-state column is written after the model returns, so captures that include it keep the fc at its
    stock place instead of forking at the last captured decoder layer."""
    worker, forks = _fc_early_worker(monkeypatch)
    md = _CaptureMetadata(1)
    md.layers_to_capture = (dispatch.FINAL_HIDDEN_LAYER_ID, 29, 56)
    worker.arm_capture_norm(SimpleNamespace(num_seqs=1), md, object(), object())
    for layer_id in (29, 56, dispatch.FINAL_HIDDEN_LAYER_ID):
        worker.capture_hidden_states(md, layer_id, None)
    assert not forks and worker._m3_fc_early_stream is None


@pytest.mark.parametrize("nreq,eligible", [(3, True), (8, True), (1, False)])
def test_no_early_fork_past_its_rows_or_off_the_fused_verify(monkeypatch, nreq, eligible) -> None:
    worker, forks = _fc_early_worker(monkeypatch, eligible)
    md = _CaptureMetadata(nreq)
    worker._m3_fc_early_stream = "stale"
    worker.arm_capture_norm(SimpleNamespace(num_seqs=nreq), md, object(), object())
    for layer_id in md.layers_to_capture:
        worker.capture_hidden_states(md, layer_id, None)
    assert not forks and worker._m3_fc_early_stream is None


# ------------------------------------------------------------------ the NVFP4 producer's coverage


def _producer_core(nvfp4: bool = True, heads: int = 16):
    """What the producer dispatch reads off the core before its first precondition."""
    return SimpleNamespace(main_kv_is_nvfp4=nvfp4, heads=heads, kv_heads=1, index_heads=1)


def test_the_producer_covers_the_tp4_nvfp4_layers_only() -> None:
    """``PRODUCER`` covers a sparse layer on an NVFP4 cache at the TP4 heads; elsewhere the generic producer runs
    without a miss."""
    covered = dispatch._producer_covered
    assert covered(_producer_core())
    assert not covered(_producer_core(nvfp4=False))
    assert not covered(_producer_core(heads=8))
    assert not covered(SimpleNamespace(main_kv_is_nvfp4=True, heads=16, kv_heads=2, index_heads=2))
    assert dispatch.PRODUCER.covers("nvfp4_tp4")


def _produce(core):
    packed = torch.zeros(4, 20 * 128, dtype=torch.bfloat16)
    return dispatch._sparse_qkv_producer(core, 3, packed, None, None, None, None)


def test_a_covered_producer_layer_missing_a_precondition_raises_under_require() -> None:
    """A TP4 NVFP4 sparse layer whose step lacks a producer precondition (here the position ids) is a miss."""
    dispatch.set_require(True)
    with pytest.raises(
        dispatch.FastPathUnavailable, match=r"'m3_producer' covers this step at 4 rows"
    ):
        _produce(_producer_core())


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
    """The packed five-way projection and the separate QKV + BF16 index projections
    (fuse_qkv_index_projection=False) are both this target's; a BF16 indexer or a BF16 KV cache is not."""
    dispatch._validate_runtime(_runtime(fused))
    dispatch._validate_runtime(_runtime(fused, kv="nvfp4"))
    with pytest.raises(ValueError, match="indexer_kv_dtype='fp8'"):
        dispatch._validate_runtime(_runtime(fused, indexer="bf16"))
    with pytest.raises(ValueError, match="FP8 or NVFP4 KV cache"):
        dispatch._validate_runtime(_runtime(fused, kv="bf16"))


@pytest.mark.parametrize("target", ["PrefillTarget", "DecodeTarget"])
def test_an_unfused_layer_the_producer_covers_is_a_miss(target) -> None:
    """As the built-in target: a layer the NVFP4 producer covers needs the five-way projection, so the unfused path is
    a miss there (raised under require) before anything runs, in either target."""
    dispatch.set_require(True)
    attention = getattr(dispatch, target)(_producer_core())._unfused_sparse_attention
    with pytest.raises(dispatch.FastPathUnavailable, match="fuse_qkv_index_projection is disabled"):
        if target == "PrefillTarget":
            attention(3, None, None, None)
        else:
            attention(3, None, None, None, None)


def test_a_covered_producer_miss_falls_back_and_logs_once(log) -> None:
    core = _producer_core()
    assert _produce(core) is None and _produce(core) is None
    assert len(log.warning) == 1 and "m3_producer at 4 rows" in log.warning[0]


@pytest.mark.parametrize("nvfp4,heads", [(False, 16), (True, 8)], ids=["fp8_cache", "tp8_heads"])
def test_an_uncovered_producer_layer_falls_back_silently(log, nvfp4, heads) -> None:
    dispatch.set_require(True)
    assert _produce(_producer_core(nvfp4, heads)) is None
    assert not log.warning


def test_a_covered_producer_layer_with_the_wrong_pool_layout_is_not_supported() -> None:
    """The pool-layout check the covered branch takes before the generic producer (then a miss)."""
    pool = torch.zeros(4, 2, 1, 128, 32, dtype=torch.uint8)
    scales = torch.zeros(4, 2, 1, 128, 8, dtype=torch.uint8)
    assert not dispatch._producer_layout(pool, scales)
    good = torch.zeros(4, 2, 1, 128, 64, dtype=torch.uint8)
    assert dispatch._producer_layout(good, scales)
    assert not dispatch._producer_layout(good, scales.transpose(3, 4).contiguous().transpose(3, 4))


# ------------------------------------------------------------------ the shell's speculative plumbing


def test_the_shell_installs_guided_decoding_on_the_worker() -> None:
    """With speculation the guided decoder goes to the worker (the in-tree one-engine shell's delegation); without a
    worker the shell reports it cannot take one."""
    shell_cls = dispatch.ModelingV2MiniMaxM3Nvfp4Sm103Tp4
    shell = shell_cls.__new__(shell_cls)
    torch.nn.Module.__init__(shell)
    installed = []
    shell.spec_worker = SimpleNamespace(set_guided_decoder=lambda g: installed.append(g) or True)
    decoder = object()
    assert shell.set_guided_decoder(decoder) and installed == [decoder]
    shell.spec_worker = None
    assert shell.set_guided_decoder(decoder) is False
