# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Which of the family's decode ops a generation step past one rank takes.

* **Index scores** (`_decode_index_scores`): the visible-split kernel up to
  `_INDEX_GROUP_MIN_ROWS - 1` rows, request groups of `p` rows from there, the CuTe group kernel
  from `_INDEX_SCORES_CUTE_MIN_GROUPS` groups where it serves them.
* **Sparse GQA** (`_sparse_gqa_decode`): the request-grouped CuTe kernel from
  `_GROUPED_DECODE_MIN_REQUESTS` requests, the per-row split-K one where it serves the call,
  None (the caller's `_sparse_gqa`) otherwise and over an NVFP4 pool.
* **Selection** (`_selection`): the cluster launch, then the one-CTA launch, then the two
  launches apart; a prefill step's context parts ask the one-CTA op for its context launch.

The ops are replaced by recorders; no kernel runs.
"""

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_10x__default import (  # noqa: E501
    modeling as M,
)

pytestmark = pytest.mark.cpu_only

CORE = SimpleNamespace(compress_ratio=4, softmax_scale=0.0625, token_topk=2048, block_topk=512)


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
    """A recorder in place of a CuTe GQA op: `serves` answers `taken`, a call records
    `(name, gated, rows_per_request or None)`."""

    def __init__(self, name: str, taken: bool, calls: list):
        self.name, self.taken, self.calls = name, taken, calls

    def serves(self, *args) -> bool:
        return self.taken

    def __call__(self, *args):
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
    "cluster, cta, expected",
    [(True, True, "cluster"), (False, True, "cta"), (False, False, "apart")],
)
def test_selection_order(monkeypatch, cluster, cta, expected):
    taken = []
    monkeypatch.setattr(M, "_topk_expand_cluster_supports", lambda scores: cluster)
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
    monkeypatch.setattr(M, "_topk_expand_cluster_supports", lambda scores, *a: False)
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
