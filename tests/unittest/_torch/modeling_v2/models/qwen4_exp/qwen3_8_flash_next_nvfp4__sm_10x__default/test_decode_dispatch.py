# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Which of the family's decode ops a generation step past one rank takes.

* **Index scores** (`_decode_index_scores`): the visible-split kernel up to
  `_INDEX_GROUP_MIN_ROWS - 1` rows, request groups of `p` rows from there, the CuTe group kernel
  from `_INDEX_SCORES_CUTE_MIN_GROUPS` groups where it serves them.
* **Sparse GQA** (`_sparse_gqa_decode`): at an attention-DP rank under `DP_GQA_BITMAP_ENV` the
  trunk's bitmap-union kernel from its floor where it takes the call; the request-grouped CuTe
  kernel from
  `_GROUPED_DECODE_MIN_REQUESTS` requests (an attention-DP rank's `DP_GQA_MIN_REQUESTS_ENV`),
  the per-row split-K one where it serves the call
  (and at a tep4 rank under `TEP_FAST_DECODE_ENV` its e4m3 shard up to 16 rows), None (the
  caller's `_sparse_gqa`) otherwise and over an NVFP4 pool.
* **Selection** (`_selection`): the cluster launch, then the one-CTA launch, then the two
  launches apart; a prefill step's context parts ask the one-CTA op for its context launch, and
  an attention-DP rank's generation rows take the cluster up to `DP_TOPK_CLUSTER_ROWS_ENV`.

The ops are replaced by recorders; no kernel runs.
"""

import ast
import inspect
from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_10x__default import (  # noqa: E501
    modeling as M,
)

pytestmark = pytest.mark.cpu_only

CORE = SimpleNamespace(
    compress_ratio=4, softmax_scale=0.0625, token_topk=2048, block_topk=512, tep_fast=False
)


def _scores_calls(monkeypatch, cute_serves: bool):
    calls = []
    monkeypatch.setattr(M, "_index_groups_cute_serves", lambda q, cache, rpr: cute_serves)
    monkeypatch.setattr(
        M, "index_scores_groups_cute", lambda *args: calls.append(("cute", args[-1])) or "cute"
    )

    def triton_op(**kw):
        calls.append(("triton", kw["rows_per_request"]))
        assert kw["only_visible_blocks"] and not kw["context_rows"] and kw["wide_split"]
        return "triton"

    monkeypatch.setattr(M, "qsa_paged_index_scores", triton_op)
    return calls


@pytest.mark.parametrize(
    "rows, p, serves, expected",
    [
        (4, 4, True, ("triton", 1)),  # one request: below the grouping's rows
        (16, 4, True, ("triton", 1)),  # 4 requests x 4 rows: still below
        (20, 4, False, ("triton", 4)),  # grouped on the Triton kernel
        (21, 3, True, ("triton", 3)),  # 7 groups: below the CuTe kernel's groups
        (24, 3, True, ("cute", 3)),  # 8 groups of 3
        (24, 3, False, ("triton", 3)),  # the CuTe kernel declines
        (64, 1, True, ("triton", 1)),  # one row a request: never grouped
    ],
)
def test_index_scores_dispatch(monkeypatch, rows, p, serves, expected):
    calls = _scores_calls(monkeypatch, serves)
    q = torch.zeros(rows, 4, 128, dtype=torch.bfloat16)
    logical = torch.zeros(rows, dtype=torch.int64)
    req = torch.zeros(rows, dtype=torch.int32)
    M._decode_index_scores(CORE, q, None, None, logical, req, 32, p)
    assert calls == [expected]


@pytest.mark.parametrize(
    "tep_fast, rows, p, expected",
    [
        (True, 8, 4, ("triton", 4)),  # tep4, 2 requests: grouped (`_TEP_INDEX_GROUP_ROWS`)
        (True, 4, 4, ("triton", 1)),
        (True, 12, 4, ("triton", 1)),  # the visible split
        (True, 16, 4, ("triton", 1)),
        (True, 8, 1, ("triton", 1)),  # one row a request: never grouped
        (False, 8, 4, ("triton", 1)),  # another layout
    ],
)
def test_index_scores_tep4_rows(monkeypatch, tep_fast, rows, p, expected):
    calls = _scores_calls(monkeypatch, True)
    core = SimpleNamespace(**{**vars(CORE), "tep_fast": tep_fast})
    q = torch.zeros(rows, 4, 128, dtype=torch.bfloat16)
    logical = torch.zeros(rows, dtype=torch.int64)
    M._decode_index_scores(
        core, q, None, None, logical, torch.zeros(rows, dtype=torch.int32), 32, p
    )
    assert calls == [expected]


@pytest.mark.parametrize(
    "switch, behind, capturing, rows, p, expected",
    [
        (True, True, True, 4, 4, ("prefetch", 4)),  # 1 request
        (True, True, True, 8, 4, ("prefetch", 4)),  # 2 requests
        (True, True, True, 12, 4, ("triton", 1)),  # past the measured rows
        (True, True, True, 8, 1, ("triton", 1)),  # one row a request: no verify step
        (True, False, True, 8, 4, ("triton", 4)),  # not behind the prologue (the draft layer)
        (True, True, False, 8, 4, ("triton", 4)),  # an eager step
        (False, True, True, 8, 4, ("triton", 4)),  # the switch off
    ],
)
def test_index_scores_tep4_prefetch(monkeypatch, switch, behind, capturing, rows, p, expected):
    """`TEP_INDEX_PREFETCH_ENV`: a captured verify step's trunk scores behind the prologue at
    `_TEP_INDEX_PREFETCH_ROWS` take the prefetching request-group launch."""
    calls = []

    def scores(**kw):
        tag = "prefetch" if kw.get("prefetch") else "triton"
        if kw.get("prefetch"):
            assert kw["use_pdl"] and kw["only_visible_blocks"] and not kw["context_rows"]
        calls.append((tag, kw["rows_per_request"]))
        return tag

    monkeypatch.setattr(M, "qsa_paged_index_scores", scores)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: capturing)
    core = SimpleNamespace(**{**vars(CORE), "tep_fast": True, "tep_index_prefetch": switch})
    q = torch.zeros(rows, 4, 128, dtype=torch.bfloat16)
    logical = torch.zeros(rows, dtype=torch.int64)
    req = torch.zeros(rows, dtype=torch.int32)
    M._decode_index_scores(core, q, None, None, logical, req, 32, p, behind_prologue=behind)
    assert calls == [expected]


@pytest.mark.parametrize(
    "env, tep_fast, pdl, expected, warns",
    [
        (None, True, True, False, False),
        ("1", True, True, True, False),
        ("1", False, True, False, True),  # another layout
        ("1", True, False, False, True),  # TRTLLM_ENABLE_PDL=0
    ],
)
def test_tep_index_prefetch_switch(monkeypatch, env, tep_fast, pdl, expected, warns):
    warned = []
    if env is None:
        monkeypatch.delenv(M.TEP_INDEX_PREFETCH_ENV, raising=False)
    else:
        monkeypatch.setenv(M.TEP_INDEX_PREFETCH_ENV, env)
    monkeypatch.setattr(M, "_INDEX_SCORES_PDL", pdl)
    monkeypatch.setattr(M.logger, "warning", warned.append)
    assert M._tep_index_prefetch(SimpleNamespace(tep_fast=tep_fast)) is expected
    assert bool(warned) is warns


def test_index_scores_cute_needs_its_index_dtypes(monkeypatch):
    """The CuTe kernel reads int64 positions and int32 request indices; other dtypes keep the
    Triton kernel."""
    calls = _scores_calls(monkeypatch, True)
    q = torch.zeros(24, 4, 128, dtype=torch.bfloat16)
    M._decode_index_scores(
        CORE, q, None, None, torch.zeros(24, dtype=torch.int32), torch.zeros(24, dtype=torch.int32),
        32, 3,
    )  # fmt: skip
    assert calls == [("triton", 3)]


class _GqaOp:
    """A recorder in place of a CuTe GQA op: `serves` answers `taken` (and keeps the
    `min_groups` it was asked with), a call records `(name, gated, rows_per_request or None)`."""

    def __init__(self, name: str, taken: bool, calls: list):
        self.name, self.taken, self.calls = name, taken, calls
        self.asked = []
        self.kwargs = []

    def serves(self, *args, **kwargs) -> bool:
        self.asked.append(kwargs.get("min_groups"))
        return self.taken

    def __call__(self, *args, **kwargs):
        self.kwargs.append(kwargs)
        rpr = args[-2] if self.name == "grouped" else None
        self.calls.append((self.name, args[-1] is not None, rpr))
        return self.name


def _gqa(monkeypatch, grouped: bool, splitk: bool) -> list:
    calls = []
    monkeypatch.setattr(M, "qsa_sparse_gqa_grouped_cute", _GqaOp("grouped", grouped, calls))
    monkeypatch.setattr(M, "qsa_sparse_gqa_splitk_cute", _GqaOp("splitk", splitk, calls))
    return calls


@pytest.mark.parametrize(
    "rows, p, grouped, splitk, expected",
    [
        (128, 4, True, True, ("grouped", True, 4)),  # 32 requests of 4
        (124, 4, True, True, ("splitk", True, None)),  # 31 requests: below the grouped kernel
        (128, 1, True, True, ("splitk", True, None)),  # one row a request
        (128, 4, False, True, ("splitk", True, None)),
        (128, 4, False, False, None),  # neither serves: the caller's `_sparse_gqa`
    ],
)
def test_sparse_gqa_dispatch(monkeypatch, rows, p, grouped, splitk, expected):
    calls = _gqa(monkeypatch, grouped, splitk)
    kv = SimpleNamespace(k=None, v=None, k_scale=None)
    gate = torch.zeros(rows, 1)
    out = M._sparse_gqa_decode(CORE, kv, torch.zeros(rows, 24, 256), None, None, None, 32, gate, p)
    if expected is None:
        assert out is None and calls == []
    else:
        assert out == expected[0] and calls == [expected]


def test_sparse_gqa_nvfp4_pool_keeps_its_kernels(monkeypatch):
    calls = _gqa(monkeypatch, True, True)
    kv = SimpleNamespace(k=None, v=None, k_scale=torch.ones(1))
    assert (
        M._sparse_gqa_decode(CORE, kv, torch.zeros(128, 24, 256), None, None, None, 32, None, 4)
        is None
    )
    assert calls == []


@pytest.mark.parametrize(
    "tep_fast, rows, dtype, selected, expected",
    [
        (True, 4, torch.float8_e4m3fn, 2051, "splitk"),  # one request of four rows
        (True, 16, torch.float8_e4m3fn, 2051, "splitk"),  # four requests
        (True, 20, torch.float8_e4m3fn, 2051, None),  # past the shard's rows
        (True, 16, torch.bfloat16, 2051, None),  # the op's own table decides a bf16 pool
        (True, 16, torch.float8_e4m3fn, 64, None),  # a selection of one tile
        (False, 4, torch.float8_e4m3fn, 2051, None),  # not a tep4 rank, or the switch off
    ],
)
def test_sparse_gqa_tep4_shard_on_splitk(monkeypatch, tep_fast, rows, dtype, selected, expected):
    """A tep4 rank's 6 q heads on its one e4m3 KV head take the split-K kernel up to 16 rows,
    which the op's own table (the FP8 target's routing) leaves out, and name that reach so the
    first call compiles every plan up to 16 rows."""
    calls = _gqa(monkeypatch, False, False)
    splitk = M.qsa_sparse_gqa_splitk_cute
    core = SimpleNamespace(**{**vars(CORE), "tep_fast": tep_fast})
    k = torch.zeros(8, 1, 32, 256, dtype=dtype)
    kv = SimpleNamespace(k=k, v=k, k_scale=None)
    q = torch.zeros(rows, 6, 256, dtype=torch.bfloat16)
    sel = torch.zeros(rows, selected, dtype=torch.int32)
    out = M._sparse_gqa_decode(core, kv, q, None, sel, None, 32, torch.zeros(rows, 1), 4)
    if expected is None:
        assert out is None and calls == []
    else:
        assert out == expected and calls == [("splitk", True, None)]
        assert splitk.kwargs == [{"max_rows": 16}]


@pytest.mark.parametrize(
    "switch, rows, supports, expected",
    [
        (True, 4, True, "e4m3"),  # the shard's rows on the e4m3 entry
        (True, 16, True, "e4m3"),
        (True, 20, True, None),  # past the shard's rows: neither kernel
        (True, 16, False, "splitk"),  # the entry declines the operands: the split-K kernel
        (False, 16, True, "splitk"),  # the switch off
    ],
)
def test_sparse_gqa_tep4_shard_on_e4m3(monkeypatch, switch, rows, supports, expected):
    """`TEP_GQA_E4M3_ENV`: the rows the shared split-K serves at the shard go to the e4m3 entry
    where it takes the operands."""
    calls = _gqa(monkeypatch, False, False)
    monkeypatch.setattr(M, "_e4m3_gqa_supports", lambda *a: supports)
    monkeypatch.setattr(
        M,
        "qsa_paged_sparse_gqa_e4m3_decode",
        lambda *a: calls.append(("e4m3", a[-1] is not None, None)) or "e4m3",
    )
    core = SimpleNamespace(**{**vars(CORE), "tep_fast": True, "tep_gqa_e4m3": switch})
    k = torch.zeros(8, 1, 32, 256, dtype=torch.float8_e4m3fn)
    kv = SimpleNamespace(k=k, v=k, k_scale=None)
    q = torch.zeros(rows, 6, 256, dtype=torch.bfloat16)
    sel = torch.zeros(rows, 2051, dtype=torch.int32)
    out = M._sparse_gqa_decode(core, kv, q, None, sel, None, 32, torch.zeros(rows, 1), 4)
    if expected is None:
        assert out is None and calls == []
    else:
        assert out == expected and calls == [(expected, True, None)]


@pytest.mark.parametrize(
    "env, tep_fast, sm, expected, warns",
    [
        (None, True, (10, 3), False, False),
        ("1", True, (10, 3), True, False),
        ("1", False, (10, 3), False, True),  # another layout
        ("1", True, (10, 0), False, True),  # another arch
    ],
)
def test_tep_gqa_e4m3_switch(monkeypatch, env, tep_fast, sm, expected, warns):
    warned = []
    if env is None:
        monkeypatch.delenv(M.TEP_GQA_E4M3_ENV, raising=False)
    else:
        monkeypatch.setenv(M.TEP_GQA_E4M3_ENV, env)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: sm)
    monkeypatch.setattr(M.logger, "warning", warned.append)
    core = SimpleNamespace(tep_fast=tep_fast)
    assert M._tep_gqa_e4m3(core, SimpleNamespace(type="cuda")) is expected
    assert bool(warned) is warns


@pytest.mark.parametrize(
    "env, tep_fast, sm, check, expected, warns",
    [
        (None, True, (10, 3), [], True, False),  # on by default
        ("0", True, (10, 3), [], False, False),
        (None, False, (10, 3), [], False, False),  # another layout: silently off
        (None, True, (10, 0), [], False, False),  # another arch
        (None, True, (10, 3), ["K/V pool at 4 rows"], False, True),  # the check declines
        (None, True, (10, 3), RuntimeError("no compile"), False, True),
    ],
)
def test_tep_attn_inplace_switch(monkeypatch, env, tep_fast, sm, check, expected, warns):
    """`TEP_ATTN_INPLACE_ENV`: a tep4 rank with its fast forms on SM103 takes the in-place
    prologue once its self-check finds the bytes equal; a failing or raising check declines on
    this rank alone, with one warning."""
    warned = []
    if env is None:
        monkeypatch.delenv(M.TEP_ATTN_INPLACE_ENV, raising=False)
    else:
        monkeypatch.setenv(M.TEP_ATTN_INPLACE_ENV, env)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: sm)
    monkeypatch.setattr(M.logger, "warning", warned.append)
    monkeypatch.setattr(M.logger, "info", lambda *a: None)

    def run_check(core, device):
        if isinstance(check, Exception):
            raise check
        return check

    monkeypatch.setattr(M, "_tep_attn_inplace_check", run_check)
    core = SimpleNamespace(tep_fast=tep_fast, mapping=SimpleNamespace(tp_rank=1))
    assert M._tep_attn_inplace_build(core, SimpleNamespace(type="cuda")) is expected
    assert bool(warned) is warns


def test_tep_attn_inplace_reads_the_query_heads_off_the_projection():
    """The in-place form feeds the sparse GQA the projection's q heads (no packed copy) and
    engages only on a few rows over an e4m3 pool, behind the same rows table."""
    source = inspect.getsource(M.DecodeTarget._attention_block)
    assert "_TEP_ATTN_INPLACE_MAX_ROWS" in source and "torch.float8_e4m3fn" in source
    assert "_attn_in_heads(core, proj)[0] if inplace else _packed_q(core, qkv)" in source
    assert M._TEP_ATTN_INPLACE_MAX_ROWS == 4


@pytest.mark.parametrize(
    "cluster, cta, expected",
    [(True, True, "cluster"), (False, True, "cta"), (False, False, "apart")],
)
def test_selection_order(monkeypatch, cluster, cta, expected):
    taken = []
    monkeypatch.setattr(M, "_topk_expand_cluster_supports", lambda scores, max_rows: cluster)
    monkeypatch.setattr(M, "_topk_expand_supports", lambda scores: cta)
    monkeypatch.setattr(M, "qsa_topk_expand_cluster", lambda *a, **k: taken.append("cluster"))
    monkeypatch.setattr(M, "qsa_topk_expand_decode", lambda *a, **k: taken.append("cta"))
    monkeypatch.setattr(M, "_select", lambda *a, **k: taken.append("apart"))
    monkeypatch.setattr(M, "qsa_expand_block_indices", lambda *a, **k: torch.zeros(1, 1))
    out = torch.zeros(1, 1)
    M._selection(CORE, torch.zeros(1, 8), None, None, None, None, None, out)
    assert taken == [expected]


@pytest.mark.parametrize("context", [False, True])
def test_selection_forwards_context_to_the_one_cta_op(monkeypatch, context):
    seen = []
    monkeypatch.setattr(M, "_topk_expand_cluster_supports", lambda scores, max_rows: False)
    monkeypatch.setattr(M, "_topk_expand_supports", lambda scores: True)
    monkeypatch.setattr(M, "qsa_topk_expand_decode", lambda *a, **k: seen.append(k["context"]))
    out = torch.zeros(1, 1)
    M._selection(CORE, torch.zeros(1, 8), None, None, None, None, None, out, context=context)
    assert seen == [context]


def test_score_and_select_marks_context_parts(monkeypatch):
    """Each part of a prefill step selects with its own `context_rows`: a context part takes
    the op's context launch, a part of generation rows the decode launch."""
    seen = []
    monkeypatch.setattr(M, "qsa_paged_index_scores", lambda **kw: torch.zeros(kw["q"].shape[0], 8))
    monkeypatch.setattr(
        M, "_selection", lambda *a, context=False, **k: seen.append((a[1].shape[0], context))
    )
    rows = 310
    t = torch.zeros(rows, dtype=torch.int64)
    block_table = torch.zeros(2, 4, dtype=torch.int32)
    parts = [(0, 300, True), (300, rows, False)]
    M._score_and_select(
        CORE, None, torch.zeros(rows, 4), block_table, t, t, 32, parts, t, t, t, t, t, None
    )
    assert seen == [(300, True), (10, False)]


@pytest.mark.parametrize(
    "rows, floor, expected",
    [
        (8, 2, ("grouped", True, 4)),  # 2 requests of 4 at a floor of 2
        (4, 2, ("splitk", True, None)),  # one request: below any floor
        (28, 8, ("splitk", True, None)),  # 7 requests below a floor of 8
        (32, 8, ("grouped", True, 4)),
        (32, 32, ("splitk", True, None)),  # the row table's 32 (the switch unset)
    ],
)
def test_sparse_gqa_dp_min_requests(monkeypatch, rows, floor, expected):
    """`DP_GQA_MIN_REQUESTS_ENV` moves the grouped kernel's first request count; the op is asked
    for that many groups."""
    calls = []
    grouped = _GqaOp("grouped", True, calls)
    monkeypatch.setattr(M, "qsa_sparse_gqa_grouped_cute", grouped)
    monkeypatch.setattr(M, "qsa_sparse_gqa_splitk_cute", _GqaOp("splitk", True, calls))
    core = SimpleNamespace(**{**vars(CORE), "dp_gqa_min_requests": floor})
    kv = SimpleNamespace(k=None, v=None, k_scale=None)
    gate = torch.zeros(rows, 1)
    out = M._sparse_gqa_decode(core, kv, torch.zeros(rows, 24, 256), None, None, None, 32, gate, 4)
    assert out == expected[0] and calls == [expected]
    # The grouped op is asked (at the floor) exactly when the row-count guard lets it be: a
    # grouped launch never skips its `serves`, and a guarded count never reaches it.
    assert grouped.asked == ([floor] if expected[0] == "grouped" else [])


@pytest.mark.parametrize(
    "env, dp_size, expected",
    [
        (None, 4, 32),
        ("2", 4, 2),
        ("8", 8, 8),
        ("32", 8, 32),
        ("2", 1, 32),  # no attention DP: warns, keeps the row table
        ("1", 4, ValueError),
        ("33", 4, ValueError),
        ("eight", 4, ValueError),
    ],
)
def test_dp_gqa_min_requests_switch(monkeypatch, env, dp_size, expected):
    if env is None:
        monkeypatch.delenv(M.DP_GQA_MIN_REQUESTS_ENV, raising=False)
    else:
        monkeypatch.setenv(M.DP_GQA_MIN_REQUESTS_ENV, env)
    core = SimpleNamespace(dp_size=dp_size)
    if expected is ValueError:
        with pytest.raises(ValueError):
            M._dp_gqa_min_requests(core)
    else:
        assert M._dp_gqa_min_requests(core) == expected


@pytest.mark.parametrize(
    "rows_env, context, asked",
    [
        (None, False, 37),  # the op's own rows
        (256, False, 256),  # generation rows up to the switch's rows
        (256, True, 37),  # a context part keeps the op's own rows
    ],
)
def test_selection_cluster_rows(monkeypatch, rows_env, context, asked):
    seen = []
    monkeypatch.setattr(
        M, "_topk_expand_cluster_supports", lambda scores, max_rows: seen.append(max_rows)
    )
    monkeypatch.setattr(M, "_topk_expand_supports", lambda scores: True)
    monkeypatch.setattr(M, "qsa_topk_expand_decode", lambda *a, **k: None)
    extra = {} if rows_env is None else {"dp_topk_cluster_rows": rows_env}
    core = SimpleNamespace(**{**vars(CORE), **extra})
    out = torch.zeros(1, 1)
    M._selection(core, torch.zeros(1, 8), None, None, None, None, None, out, context=context)
    assert seen == [asked]


@pytest.mark.parametrize(
    "env, dp_size, expected",
    [
        (None, 4, 37),
        ("76", 8, 76),
        ("256", 4, 256),
        ("76", 1, 37),  # no attention DP: warns, keeps the op's rows
        ("37", 4, ValueError),  # not past the op's own rows
        ("257", 4, ValueError),
        ("all", 4, ValueError),
    ],
)
def test_dp_topk_cluster_rows_switch(monkeypatch, env, dp_size, expected):
    if env is None:
        monkeypatch.delenv(M.DP_TOPK_CLUSTER_ROWS_ENV, raising=False)
    else:
        monkeypatch.setenv(M.DP_TOPK_CLUSTER_ROWS_ENV, env)
    core = SimpleNamespace(dp_size=dp_size)
    if expected is ValueError:
        with pytest.raises(ValueError):
            M._dp_topk_cluster_rows(core)
    else:
        assert M._dp_topk_cluster_rows(core) == expected


def _bitmap_case(monkeypatch, takes: bool):
    """Recorders for the bitmap-union kernel (`fp8_sparse_gqa_verify_grouped`, its plan) and the
    row table's two kernels."""
    calls = _gqa(monkeypatch, True, True)
    asked = []

    def plan(q, k, bt, sel, tpb, p, min_groups=None):
        asked.append(min_groups)
        return takes

    def bitmap(*args, output_gate=None, min_groups=None):
        calls.append(("bitmap", output_gate is not None, args[-1], min_groups))
        return "bitmap"

    monkeypatch.setattr(M, "_bitmap_takes_grouped", plan)
    monkeypatch.setattr(M, "fp8_sparse_gqa_verify_grouped", bitmap)
    return calls, asked


def _bitmap_operands(rows, dtype=torch.float8_e4m3fn, gate_cols=24 * 256):
    k = torch.zeros(8, 2, 32, 256, dtype=dtype)
    kv = SimpleNamespace(k=k, v=k, k_scale=None)
    q = torch.zeros(rows, 24, 256, dtype=torch.bfloat16)
    sel = torch.zeros(rows, 2051, dtype=torch.int32)
    req = torch.zeros(rows, dtype=torch.int32)
    bt = torch.zeros(4, 16, dtype=torch.int32)
    gate = torch.zeros(rows, gate_cols, dtype=torch.bfloat16)
    return kv, q, bt, sel, req, gate


@pytest.mark.parametrize(
    "rows, floor, trunk, takes, expected",
    [
        (32, 8, True, True, ("bitmap", True, 4, 8)),  # 8 requests of 4 at the front's floor
        (8, 2, True, True, ("bitmap", True, 4, 2)),  # 2 requests at a floor of 2 (caps 3-7)
        (28, 8, True, True, ("splitk", True, None)),  # 7 requests: below the floor
        (128, 32, True, True, ("bitmap", True, 4, 8)),  # a floor past the op's asks it for 8
        (32, 8, False, True, ("splitk", True, None)),  # the draft layer keeps the row table
        (32, 8, True, False, ("splitk", True, None)),  # the kernel declines (plan, arch)
        (32, 0, True, True, ("splitk", True, None)),  # the switch off
    ],
)
def test_sparse_gqa_bitmap_dispatch(monkeypatch, rows, floor, trunk, takes, expected):
    """`DP_GQA_BITMAP_ENV`: the trunk's verify step takes the bitmap-union kernel from its floor
    of requests where the kernel's plan takes the call; below it, off the trunk and where it
    declines, the row table decides as before (here: split-K)."""
    calls, asked = _bitmap_case(monkeypatch, takes)
    core = SimpleNamespace(**{**vars(CORE), "dp_gqa_bitmap": floor, "dp_gqa_min_requests": 32})
    kv, q, bt, sel, req, gate = _bitmap_operands(rows)
    out = M._sparse_gqa_decode(core, kv, q, bt, sel, req, 32, gate, 4, trunk=trunk)
    assert out == expected[0] and calls == [expected]
    if expected[0] == "bitmap" or (floor and trunk and rows >= floor * 4):
        assert asked == [min(floor, 8)]


@pytest.mark.parametrize(
    "why, mutate",
    [
        ("a bf16 pool", "bf16"),
        ("one row a request", "p1"),
        ("no gate", "nogate"),
        ("a gate of other columns", "gatecols"),
        ("a strided selection", "selstride"),
    ],
)
def test_sparse_gqa_bitmap_declines_layouts_its_entry_refuses(monkeypatch, why, mutate):
    calls, asked = _bitmap_case(monkeypatch, True)
    core = SimpleNamespace(**{**vars(CORE), "dp_gqa_bitmap": 8, "dp_gqa_min_requests": 32})
    kv, q, bt, sel, req, gate = _bitmap_operands(32)
    p = 4
    if mutate == "bf16":
        kv.k = kv.k.bfloat16()
    elif mutate == "p1":
        p = 1
    elif mutate == "nogate":
        gate = None
    elif mutate == "gatecols":
        gate = torch.zeros(32, 16 * 256, dtype=torch.bfloat16)
    elif mutate == "selstride":
        sel = torch.zeros(32, 4102, dtype=torch.int32)[:, :2051]
    M._sparse_gqa_decode(core, kv, q, bt, sel, req, 32, gate, p, trunk=True)
    assert all(c[0] != "bitmap" for c in calls), why
    assert asked == []


@pytest.mark.parametrize(
    "bitmap_env, min_env, dp_size, expected",
    [
        (None, None, 8, 0),  # off by default
        ("1", None, 8, 8),  # the front's floor
        ("1", "2", 8, 2),  # DP_GQA_MIN_REQUESTS moves it (caps 3-7)
        ("1", "16", 4, 16),
        ("0", "2", 8, 0),
        ("1", None, 1, 0),  # no attention DP: warns, off
        ("on", None, 4, 8),
        ("maybe", None, 4, ValueError),
    ],
)
def test_dp_gqa_bitmap_switch(monkeypatch, bitmap_env, min_env, dp_size, expected):
    for env, value in ((M.DP_GQA_BITMAP_ENV, bitmap_env), (M.DP_GQA_MIN_REQUESTS_ENV, min_env)):
        if value is None:
            monkeypatch.delenv(env, raising=False)
        else:
            monkeypatch.setenv(env, value)
    core = SimpleNamespace(dp_size=dp_size)
    if expected is ValueError:
        with pytest.raises(ValueError):
            M._dp_gqa_bitmap(core)
        return
    core.dp_gqa_min_requests = M._dp_gqa_min_requests(core)
    assert M._dp_gqa_bitmap(core) == expected


@pytest.mark.parametrize(
    "rows, draft, expected",
    [
        (37, False, {}),  # clusters of 4: the op's own placement
        (38, False, dict(solo_blocks=8192, pairs=0)),  # pairs of CTAs, short rows alone
        (76, False, dict(solo_blocks=8192, pairs=0)),
        (77, False, dict(solo_blocks=8192, pairs=74)),  # past one wave: the row map
        (104, False, dict(solo_blocks=8192, pairs=74)),
        (105, False, dict(solo_blocks=32768, pairs=0)),  # rows of up to 32768 blocks alone
        (256, False, dict(solo_blocks=32768, pairs=0)),
        (38, True, {}),  # draft: clusters of 2 as the op places them
        (39, True, dict(solo_blocks=32768, pairs=0)),
        (64, True, dict(solo_blocks=32768, pairs=0)),
        (65, True, {}),
    ],
)
def test_topk_placement_rows(monkeypatch, rows, draft, expected):
    monkeypatch.setattr(M, "_sm_count", lambda index: 148)
    assert M._topk_placement(torch.zeros(rows, 8), draft) == expected


@pytest.mark.parametrize(
    "rows_env, draft, rows, knobs",
    [
        (None, False, 20, {}),  # the switch unset: nothing past the op's own rows
        (256, False, 20, {}),  # clusters of 4 or 8: no placement
        (256, False, 96, dict(solo_blocks=8192, pairs=74)),
        (256, True, 48, dict(solo_blocks=32768, pairs=0)),
    ],
)
def test_selection_passes_the_placement(monkeypatch, rows_env, draft, rows, knobs):
    seen = []
    monkeypatch.setattr(M, "_sm_count", lambda index: 148)
    monkeypatch.setattr(M, "_topk_expand_cluster_supports", lambda scores, max_rows: True)
    monkeypatch.setattr(M, "qsa_topk_expand_cluster", lambda *a, **k: seen.append(k))
    extra = {} if rows_env is None else {"dp_topk_cluster_rows": rows_env}
    core = SimpleNamespace(**{**vars(CORE), **extra})
    out = torch.zeros(rows, 1)
    M._selection(core, torch.zeros(rows, 8), None, None, None, None, None, out, draft=draft)
    assert seen == [dict(out=out, **knobs)]


def _fp8_recorders(monkeypatch):
    calls = []
    monkeypatch.setattr(
        M, "qsa_index_scores_fp8", lambda *a: calls.append(("fp8", a[-3], a[-2], a[-1])) or "fp8"
    )
    monkeypatch.setattr(
        M,
        "qsa_paged_index_scores",
        lambda **kw: calls.append(("bf16", kw["context_rows"])) or "bf16",
    )
    monkeypatch.setattr(M, "index_scores_groups_cute", lambda *a: calls.append(("cute",)) or "cute")
    monkeypatch.setattr(M, "_index_groups_cute_serves", lambda q, cache, rpr: True)
    monkeypatch.setattr(
        M, "index_anchors_to_fp8", lambda *a: calls.append(("convert", a[-2], a[-1]))
    )
    return calls


@pytest.mark.parametrize("fp8", [False, True])
@pytest.mark.parametrize("rows, p", [(4, 4), (32, 4), (24, 3), (8, 1)])
def test_index_key_fp8_decode_scores(monkeypatch, fp8, rows, p):
    """`INDEX_KEY_FP8_ENV`: every generation step's scores on the e4m3 kernel with the step's
    rows a request, never the bf16 Triton or CuTe kernels."""
    calls = _fp8_recorders(monkeypatch)
    core = SimpleNamespace(**{**vars(CORE), "index_key_fp8": fp8})
    t = torch.zeros(rows, dtype=torch.int64)
    M._decode_index_scores(core, torch.zeros(rows, 4, 128), None, None, t, t.int(), 32, p)
    if fp8:
        assert calls == [("fp8", True, False, p)]
    else:
        assert calls and calls[0][0] in ("bf16", "cute")


def test_index_key_fp8_keeps_the_tep4_prefetch_off(monkeypatch):
    """`TEP_INDEX_PREFETCH_ENV` with `INDEX_KEY_FP8_ENV`: the prefetching kernel reads bf16
    anchors, so a captured trunk call behind the prologue goes to the e4m3 kernel instead."""
    calls = _fp8_recorders(monkeypatch)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    core = SimpleNamespace(
        **{**vars(CORE), "tep_fast": True, "tep_index_prefetch": True, "index_key_fp8": True}
    )
    t = torch.zeros(8, dtype=torch.int64)
    q = torch.zeros(8, 4, 128)
    M._decode_index_scores(core, q, None, None, t, t.int(), 32, 4, behind_prologue=True)
    assert calls == [("fp8", True, False, 4)]


@pytest.mark.parametrize("fp8", [False, True])
def test_index_key_fp8_reader_and_writer_helpers(monkeypatch, fp8):
    calls = _fp8_recorders(monkeypatch)
    core = SimpleNamespace(**{**vars(CORE), "index_key_fp8": fp8})
    kw = dict(
        q=torch.zeros(300, 4, 128), index_cache=None, block_table=None, query_positions=None,
        request_indices=None, tokens_per_block=32, compress_ratio=4, only_visible_blocks=True,
        context_rows=True, rows_per_request=1,
    )  # fmt: skip
    M._index_scores(core, **kw)
    M._index_anchors_fp8(core, None, None, None, None, 32)
    if fp8:
        assert calls == [("fp8", True, True, 1), ("convert", 32, 4)]
    else:
        assert calls == [("bf16", True)]


def test_index_key_fp8_switch(monkeypatch):
    monkeypatch.delenv(M.INDEX_KEY_FP8_ENV, raising=False)
    assert M._switch(M.INDEX_KEY_FP8_ENV) is False
    monkeypatch.setenv(M.INDEX_KEY_FP8_ENV, "1")
    assert M._switch(M.INDEX_KEY_FP8_ENV) is True


#: The launches that write a QSA index anchor (the group compression and the two fused
#: prologues, which compress at a group's last token), and the bf16-key score kernels.
_ANCHOR_WRITERS = frozenset(
    {"qsa_prefill_compress", "one_rank_attn_decode_prologue", "qwen4_exp_attn_decode_prologue"}
)
_BF16_SCORE_KERNELS = frozenset({"qsa_paged_index_scores", "index_scores_groups_cute"})


def _calls_by_function():
    """`(qualified name, called names)` of every function of the modeling source."""
    found = []

    def visit(node, prefix):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.ClassDef):
                visit(child, f"{prefix}{child.name}.")
            elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                names = []
                for sub in ast.walk(child):
                    if isinstance(sub, ast.Call):
                        f = sub.func
                        names.append(f.id if isinstance(f, ast.Name) else getattr(f, "attr", None))
                found.append((prefix + child.name, names))

    visit(ast.parse(inspect.getsource(M)), "")
    return found


#: Build self-checks that run the anchor writers on a scratch bf16 plane no step reads.
_SCRATCH_ANCHOR_WRITERS = frozenset({"_tep_attn_inplace_check"})


def test_index_key_fp8_every_anchor_writer_converts():
    """Under `INDEX_KEY_FP8_ENV` a bf16 anchor left in the plane is read as e4m3 bytes: every
    function that launches an anchor writer converts behind each launch (`_index_anchors_fp8`),
    so a writer added without its conversion fails here."""
    writers = 0
    for name, calls in _calls_by_function():
        if name in _SCRATCH_ANCHOR_WRITERS:
            continue
        launched = sum(c in _ANCHOR_WRITERS for c in calls)
        writers += launched
        assert calls.count("_index_anchors_fp8") == launched, (
            f"{name}: {launched} anchor writer launches, "
            f"{calls.count('_index_anchors_fp8')} conversions"
        )
    assert writers >= 6, "the writers were renamed: update _ANCHOR_WRITERS"


def test_index_key_fp8_every_score_call_is_routed():
    """Every bf16-key score kernel call sits in the two routers that take the e4m3 kernel under
    `INDEX_KEY_FP8_ENV` (`_index_scores`, `_decode_index_scores`)."""
    callers = {
        name for name, calls in _calls_by_function() if any(c in _BF16_SCORE_KERNELS for c in calls)
    }
    assert callers == {"_index_scores", "_decode_index_scores"}
