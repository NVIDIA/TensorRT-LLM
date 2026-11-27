# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Which of the family's decode ops a generation step past one rank takes.

* **Index scores** (`_decode_index_scores`): the visible-split kernel up to
  `_INDEX_GROUP_MIN_ROWS - 1` rows, request groups of `p` rows from there, the CuTe group kernel
  from `_INDEX_SCORES_CUTE_MIN_GROUPS` groups where it serves them.
* **Sparse GQA** (`_sparse_gqa_decode`): the request-grouped CuTe kernel from
  `_GROUPED_DECODE_MIN_REQUESTS` requests (an attention-DP rank's `DP_GQA_MIN_REQUESTS_ENV`),
  the per-row split-K one where it serves the call
  (and at a tep4 rank under `TEP_FAST_DECODE_ENV` its e4m3 shard up to 16 rows), None (the
  caller's `_sparse_gqa`) otherwise and over an NVFP4 pool.
* **Selection** (`_selection`): the cluster launch, then the one-CTA launch, then the two
  launches apart; a prefill step's context parts ask the one-CTA op for its context launch, and
  an attention-DP rank's generation rows take the cluster up to `DP_TOPK_CLUSTER_ROWS_ENV`.

The ops are replaced by recorders; no kernel runs.
"""

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
