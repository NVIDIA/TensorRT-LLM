# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DEP2CTX_IDX_SCORER and DEP2CTX_TSEL_ROWS on the dep2ctx target (CPU).

- score() hands the select a zero-copy [H, pages, T] view of the scorer's row-major [T * H, pitch] buffer (one index
  head per KV head), equal to the amax / permute / contiguous copy; more index heads per KV head keep the copy;
- the step cache: one .tolist() per step, one plan per distinct lengths (the old one released first), one block-table
  gather per step and page table;
- declines: capture, dtypes, non-contiguous or misaligned index-Q, rows off bottom-right, > 8192 pages; a decline
  releases the kept plan; prewarm() runs one small plan + score and leaves the kept plan;
- tile_select.select takes head-major scores only (the tile op); tile_select.select_rows takes row-major ones (the rows
  op), whose op follows its switch.

Registering the scorer on the MSA indexer needs the indexer's prefill-scorer hooks, which this engine does not have.
"""

import gc
import weakref
from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2._router_index import NoTarget
from tensorrt_llm._torch.attention.backends.sparse.minimax_m3 import msa_indexer

try:
    from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_1_flash_nvfp4__sm_107__dep2ctx import (  # noqa: E501
        idx_scorer,
        tile_select,
    )
except NoTarget as exc:
    pytest.skip(str(exc), allow_module_level=True)

pytestmark = pytest.mark.cpu_only

PAGE, D = 128, 128


@pytest.fixture(autouse=True)
def _cpu_capture_answer(monkeypatch):
    """A CPU-only torch raises from torch.cuda.is_current_stream_capturing; a CUDA build answers False outside a
    capture, which is what these host-side tests need (tests that model a capture patch it themselves)."""
    if not torch.cuda.is_available():
        monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)


@pytest.fixture(autouse=True)
def _fresh_state(monkeypatch):
    monkeypatch.setattr(idx_scorer, "_STATE", {})
    monkeypatch.setattr(tile_select, "_STATE", {})


# ---- a CPU stand-in for the Q8KV8 wrapper ------------------------------------------------------------------------


def _pitch(pages):
    return -(-pages // 4) * 4


def _cols(pages):
    return pages if pages % 4 < 2 else _pitch(pages)


class _Wrapper:
    """BatchPrefillIndexerQ8KV8Wrapper's contract in fp32: plan() from the lengths and block table, run_scores() the raw
    page maxima of q.k into row t * H + h of a [T * H, pitch] buffer (sliced to its score columns), historical pages
    only (the local page and later ones keep the buffer's contents: NaN here)."""

    plans = 0
    previous = None  # a weak reference to the last planned wrapper

    def __init__(self, num_heads):
        self.num_heads = num_heads

    def plan(self, cu_q, seq_lens, block_table, *, total_q, max_seqlen_q, max_seqlen_k):
        assert idx_scorer._STATE.get("wrapper") is None, "the old plan was not released first"
        # No reference at all (a local one included) may keep the old plan's score buffer alive here.
        old = _Wrapper.previous() if _Wrapper.previous is not None else None
        assert old is None or old is self, "the old wrapper is still alive when the new one plans"
        del old
        _Wrapper.previous = weakref.ref(self)
        _Wrapper.plans += 1
        self.cu_q, self.seq_lens, self.block_table = cu_q.tolist(), seq_lens.tolist(), block_table
        self.total_q = total_q
        pages = -(-max_seqlen_k // PAGE)
        self.scores = torch.full((total_q * self.num_heads, _pitch(pages)), float("nan"))[
            :, : _cols(pages)
        ]

    def run_scores(self, q, k):
        H = self.num_heads
        qf, kf = q.float(), k.float()
        for r, kv in enumerate(self.seq_lens):
            q0, q1 = self.cu_q[r], self.cu_q[r + 1]
            for t in range(q0, q1):
                pos = kv - (q1 - q0) + (t - q0)
                for p in range(pos // PAGE):  # historical pages: all their tokens precede pos
                    page = int(self.block_table[r, p])
                    s = (qf[t] @ kf[page].T).amax(dim=1)  # [H]
                    self.scores[t * H : (t + 1) * H, p] = s
        return self.scores


def _fake_module(monkeypatch):
    monkeypatch.setattr(
        idx_scorer, "_module", lambda: SimpleNamespace(BatchPrefillIndexerQ8KV8Wrapper=_Wrapper)
    )
    _Wrapper.plans = 0
    _Wrapper.previous = None


def _inputs(qo, prefix, heads=4, seed=0, shuffle=True):
    g = torch.Generator().manual_seed(seed)
    qo_cpu = torch.tensor(qo, dtype=torch.int32)
    kv_cpu = qo_cpu + torch.tensor(prefix, dtype=torch.int32)
    pages = int(sum(-(-int(x) // PAGE) for x in kv_cpu))
    # Few distinct values: ties between pages are frequent, so the (key, block) tie rule is exercised.
    q = torch.randint(-2, 3, (sum(qo), heads, D), generator=g).float().to(torch.float8_e4m3fn)
    k = torch.randint(-2, 3, (pages, 1, PAGE, D), generator=g).float().to(torch.float8_e4m3fn)
    kv_indices = (torch.randperm(pages, generator=g) if shuffle else torch.arange(pages)).to(
        torch.int32
    )
    return q, k, qo_cpu, kv_cpu, kv_cpu - qo_cpu, kv_indices


# ---- the zero-copy view ----------------------------------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_group_one_scores_are_a_zero_copy_view_equal_to_the_old_transpose(monkeypatch) -> None:
    _fake_module(monkeypatch)
    q, k, qo, kv, off, kv_indices = _inputs([300], [4096 + 77])
    out = idx_scorer.score(q, k, qo, kv, off, kv_indices, num_kv_heads=4)
    buf = idx_scorer._STATE["wrapper"].scores  # [T * H, cols] slice of [T * H, pitch]
    pitch, cols = buf.stride(0), buf.shape[1]
    assert out.shape == (4, cols, 300) and out.stride() == (pitch, 1, 4 * pitch)
    assert out.data_ptr() == buf.data_ptr()
    old = buf.view(300, 4, 1, -1).amax(dim=2).permute(1, 2, 0).contiguous()  # the copy form
    assert torch.equal(out.nan_to_num(nan=-7.0), old.nan_to_num(nan=-7.0))


def test_more_heads_per_kv_head_keep_the_reduced_copy(monkeypatch) -> None:
    _fake_module(monkeypatch)
    q, k, qo, kv, off, kv_indices = _inputs([64], [1024], heads=4)
    out = idx_scorer.score(q, k, qo, kv, off, kv_indices, num_kv_heads=2)
    raw = idx_scorer._STATE["wrapper"].scores.view(64, 4, -1).permute(1, 2, 0)
    ref = msa_indexer._group_max_reduce(raw, SimpleNamespace(num_index_heads=4, num_kv_heads=2))
    assert out.shape == (2, raw.shape[1], 64) and out.is_contiguous()
    assert torch.equal(out.nan_to_num(nan=-7.0), ref.nan_to_num(nan=-7.0))


# ---- the step cache --------------------------------------------------------------------------------------------


class _Lens:
    """A host length tensor that counts its .tolist() calls."""

    def __init__(self, values):
        self.t = torch.tensor(values, dtype=torch.int32)
        self.calls = 0

    def tolist(self):
        self.calls += 1
        return self.t.tolist()


def test_step_cache_one_tolist_one_plan_one_gather_per_step(monkeypatch) -> None:
    _fake_module(monkeypatch)
    q, k, qo, kv, off, kv_indices = _inputs([256, 100], [2048, 900])
    lens = [_Lens(qo.tolist()), _Lens(kv.tolist()), _Lens(off.tolist())]
    for _ in range(5):  # five layers of one step
        assert idx_scorer.score(q, k, *lens, kv_indices, num_kv_heads=4) is not None
    assert [x.calls for x in lens] == [1, 1, 1] and _Wrapper.plans == 1
    w = idx_scorer._STATE["wrapper"]
    # Same step, same page table object: the block table is not gathered again (prepare() is its only writer).
    gathered = w.block_table.clone()
    kv_indices.copy_(kv_indices.flip(0))
    idx_scorer.score(q, k, *lens, kv_indices, num_kv_heads=4)
    assert torch.equal(w.block_table, gathered)
    # A new page table object in the same step: gathered once.
    fresh = kv_indices.clone()
    idx_scorer.score(q, k, *lens, fresh, num_kv_heads=4)
    assert not torch.equal(w.block_table, gathered) and _Wrapper.plans == 1
    # The next step: new length objects with equal values keep the plan, read the lengths once and re-gather.
    nxt = [_Lens(qo.tolist()), _Lens(kv.tolist()), _Lens(off.tolist())]
    fresh.copy_(torch.arange(fresh.numel(), dtype=torch.int32))
    for _ in range(3):
        idx_scorer.score(q, k, *nxt, fresh, num_kv_heads=4)
    assert [x.calls for x in nxt] == [1, 1, 1] and _Wrapper.plans == 1
    assert idx_scorer._STATE["wrapper"] is w and int(w.block_table[0, 1]) == 1
    # New values: one new plan, after the old one was released and dead (_Wrapper.plan asserts both; this test holds
    # no reference of its own to the old wrapper meanwhile).
    old = weakref.ref(w)
    del w
    gc.collect()
    q2, k2, qo2, kv2, off2, kv_indices2 = _inputs([256, 100], [4096, 900])
    idx_scorer.score(q2, k2, qo2, kv2, off2, kv_indices2, num_kv_heads=4)
    assert _Wrapper.plans == 2 and old() is None


# ---- declines --------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "case",
    [
        "capture",
        "bf16 q",
        "bf16 k",
        "non-contiguous q",
        "misaligned q",
        "offset",
        "pages",
        "lengths",
    ],
)
def test_declines(monkeypatch, case) -> None:
    _fake_module(monkeypatch)
    q, k, qo, kv, off, kv_indices = _inputs([256], [2048])
    if case == "capture":
        monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    elif case == "bf16 q":
        q = q.to(torch.bfloat16)
    elif case == "bf16 k":
        k = k.to(torch.bfloat16)
    elif case == "non-contiguous q":
        q = torch.cat([q, q], dim=2)[:, :, :D]
        assert not q.is_contiguous()
    elif case == "misaligned q":
        flat = torch.zeros(q.numel() + 16, dtype=torch.float8_e4m3fn)
        base = (16 - flat.data_ptr() % 16) % 16
        q = flat[base + 1 : base + 1 + q.numel()].view(q.shape)
        assert q.is_contiguous() and q.data_ptr() % 16
    elif case == "offset":
        off = off + 1
    elif case == "pages":
        kv = qo + idx_scorer.MAX_PAGES * PAGE  # 8193 pages in the row
        off = kv - qo
    elif case == "lengths":
        qo = torch.tensor([255], dtype=torch.int32)
        off = kv - qo
    assert idx_scorer.score(q, k, qo, kv, off, kv_indices, num_kv_heads=4) is None
    assert _Wrapper.plans == 0


def test_at_the_page_bound_it_scores(monkeypatch) -> None:
    _fake_module(monkeypatch)
    monkeypatch.setattr(_Wrapper, "run_scores", lambda self, q, k: self.scores)
    q, k, qo, kv, off, kv_indices = _inputs([4], [64])
    kv = qo + idx_scorer.MAX_PAGES * PAGE - 4  # exactly 8192 pages
    big = torch.arange(idx_scorer.MAX_PAGES, dtype=torch.int32)
    assert idx_scorer.score(q, k, qo, kv, kv - qo, big, num_kv_heads=4) is not None


def test_a_decline_releases_the_kept_plan(monkeypatch) -> None:
    _fake_module(monkeypatch)
    q, k, qo, kv, off, kv_indices = _inputs([256], [2048])
    assert idx_scorer.score(q, k, qo, kv, off, kv_indices, num_kv_heads=4) is not None
    assert idx_scorer._STATE["wrapper"] is not None
    bad = qo + idx_scorer.MAX_PAGES * PAGE
    assert idx_scorer.score(q, k, qo, bad, bad - qo, kv_indices, num_kv_heads=4) is None
    assert all(idx_scorer._STATE[key] is None for key in ("wrapper", "key", "gather", "table_src"))
    # The next taken step plans anew.
    assert idx_scorer.score(q, k, qo, kv, off, kv_indices, num_kv_heads=4) is not None
    assert _Wrapper.plans == 2


def test_prewarm_runs_one_small_step_and_keeps_the_plan(monkeypatch) -> None:
    calls = []

    class Wrapper:
        def __init__(self, num_heads):
            calls.append(("init", num_heads))

        def plan(self, cu_q, seq_lens, block_table, *, total_q, max_seqlen_q, max_seqlen_k):
            calls.append(
                ("plan", cu_q.tolist(), seq_lens.tolist(), block_table.tolist(), total_q,
                 max_seqlen_q, max_seqlen_k)
            )  # fmt: skip
            assert cu_q.dtype == seq_lens.dtype == block_table.dtype == torch.int32

        def run_scores(self, q, k):
            calls.append(("run", tuple(q.shape), q.dtype, tuple(k.shape), k.dtype))

    monkeypatch.setattr(
        idx_scorer, "_module", lambda: SimpleNamespace(BatchPrefillIndexerQ8KV8Wrapper=Wrapper)
    )
    kept = object()
    idx_scorer._STATE["wrapper"] = kept
    idx_scorer.prewarm(4, device=torch.device("cpu"))
    assert calls == [
        ("init", 4),
        ("plan", [0, PAGE], [2 * PAGE], [[0, 1]], PAGE, PAGE, 2 * PAGE),
        ("run", (PAGE, 4, D), torch.float8_e4m3fn, (2, PAGE, D), torch.float8_e4m3fn),
    ]
    assert idx_scorer._STATE["wrapper"] is kept


def test_prewarm_without_the_module_or_cuda_does_nothing(monkeypatch) -> None:
    built = []
    module = SimpleNamespace(BatchPrefillIndexerQ8KV8Wrapper=lambda **kw: built.append(kw))
    monkeypatch.setattr(idx_scorer, "_module", lambda: None)
    idx_scorer.prewarm(4, device=torch.device("cpu"))
    monkeypatch.setattr(idx_scorer, "_module", lambda: module)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    idx_scorer.prewarm(4)
    assert built == []


@pytest.mark.parametrize("rows_on", [True, False])
def test_rows_op_follows_its_switch(monkeypatch, rows_on) -> None:
    built = object()
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a: (10, 7))
    monkeypatch.setattr(
        tile_select,
        "family_ops",
        lambda: SimpleNamespace(minimax_m31_select_prefill_tile_rows=built),
    )
    monkeypatch.setattr(tile_select, "TSEL_ROWS", rows_on)
    assert tile_select._rows_op() is (built if rows_on else None)


# ---- tile_select routing ---------------------------------------------------------------------------------------------


def _routes(monkeypatch, rows_op=True):
    calls = []

    def op(name):
        def run(scores, nvb, out, init, local):
            calls.append((name, scores.stride()))
            out.fill_(0)

        return run

    monkeypatch.setattr(tile_select, "_op", lambda: op("tile"))
    monkeypatch.setattr(tile_select, "_rows_op", lambda: op("rows") if rows_op else None)
    return calls


def test_tsel_routes_by_layout(monkeypatch) -> None:
    calls = _routes(monkeypatch)
    nvb = torch.ones(300, dtype=torch.int32)
    head_major = torch.randn(4, 40, 300)
    row_major = torch.randn(300, 4, 40).permute(1, 2, 0)
    neither = torch.randn(40, 4, 300, 2)[..., 0].permute(1, 0, 2)  # strides (600, 2400, 2)
    # select (every indexer select): head-major only.
    assert tile_select.select(head_major, nvb, 0, 1) is not None
    assert tile_select.select(row_major, nvb, 0, 1) is None
    assert tile_select.select(neither, nvb, 0, 1) is None
    # select_rows (the prefill scorer's selector only): row-major only.
    assert tile_select.select_rows(row_major, nvb, 0, 1) is not None
    assert tile_select.select_rows(head_major, nvb, 0, 1) is None
    assert tile_select.select_rows(neither, nvb, 0, 1) is None
    assert [c[0] for c in calls] == ["tile", "rows"] and calls[1][1] == (40, 1, 160)
    # The unchanged gates on both: too few rows, too many blocks, not fp32.
    for fn, few, many, half in (
        (tile_select.select, torch.randn(4, 40, 255), torch.randn(4, 5857, 300), head_major.half()),
        (
            tile_select.select_rows,
            torch.randn(255, 4, 40).permute(1, 2, 0),
            torch.randn(1, 4, 5857).permute(1, 2, 0).expand(4, 5857, 300),
            row_major.to(torch.bfloat16),
        ),
    ):
        assert fn(few, nvb, 0, 1) is None and fn(many, nvb, 0, 1) is None
        assert fn(half, nvb, 0, 1) is None
    assert len(calls) == 2


def test_tsel_rows_switch_keeps_the_builtin_select_for_row_major(monkeypatch) -> None:
    calls = _routes(monkeypatch, rows_op=False)
    nvb = torch.ones(300, dtype=torch.int32)
    assert tile_select.select_rows(torch.randn(300, 4, 40).permute(1, 2, 0), nvb, 0, 1) is None
    assert (
        tile_select.select(torch.randn(4, 40, 300), nvb, 0, 1) is not None and calls[0][0] == "tile"
    )
