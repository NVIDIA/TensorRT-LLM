# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""helix8's drafter copies as zero-padded MNNVL all-reduces, and the agreement's census, on CPU.

``draft_pairs_zero_padded`` gathers the selector's top-32 ``(global id, value)`` pairs and
``draft_agree_zero_padded`` broadcasts rank 0's draft tokens through ``ZeroPaddedReduce`` on the
target head's FP32 pair exchange; ``draft_agree_check`` counts the rows whose local drafts
differed from rank 0's. All three are logged generation constants: the two copies on, the
census off.

From helix8's modeling.py (read from the source, run over stand-ins; nothing is imported from
the target), with one process playing the 8 ranks and a recording all-reduce that sums the
ranks' buffers as the MNNVL kernels do (FP32, a -0.0 written as +0.0):

* the pairs gather equals the all-gather's rank-order layout bit for bit up to the sign of zero,
  through ``_dflash2_global_top_k``'s sharded branch too, whose ids, values and order stay the
  full-vocab top-32's; unbound, or at rows it was not built for, the branch calls the all-gather;
* the agreement gives every rank rank 0's drafts and the draft columns of ``next_new_tokens``;
  unbound, or at another draft length, it is the engine's;
* ``_bind_zero_padded_copies`` binds each copy only under its constant, on an FP32 MNNVL
  exchange whose group order is the worker's, with the census's totals under the check;
* the census counts real and padding rows apart and reads them out every 200 steps;
* the INFO lines are the README's.
"""

from __future__ import annotations

import ast
import itertools
from types import SimpleNamespace

import pytest
import torch

__extra_import_path__ = ["."]
from test_rubin_helix8_draft_shard import (  # noqa: E402
    SHARD,
    TOP_K,
    TP,
    VOCAB,
    _equal,
    _logits,
    _merge_top32,
    _shard_top32,
)
from test_rubin_helix8_draft_window import (  # noqa: E402
    _MODELING,
    _README,
    _constants,
    _exec,
    _tree,
)

pytestmark = pytest.mark.cpu_only

K = 5
_SWITCHES = {
    "draft_pairs_zero_padded": "_DRAFT_PAIRS_ZERO_PADDED",
    "draft_agree_zero_padded": "_DRAFT_AGREE_ZERO_PADDED",
    "draft_agree_check": "_DRAFT_AGREE_CHECK",
}
_METHODS = (
    "_bind_zero_padded_copies",
    "_agree_draft_tokens",
    "_count_agreement",
    "_report_draft_agreement",
    "_refresh_ctx_block_tables",
    "_dflash2_global_top_k",
)


class _Recording:
    """An all-reduce that keeps each rank's input; ``reduce`` sums them as the kernels do."""

    def __init__(self):
        self.inputs = []

    def __call__(self, padded):
        self.inputs.append(padded.clone())
        return padded

    def reduce(self):
        total = torch.zeros_like(self.inputs[0], dtype=torch.float32)
        for x in self.inputs:
            total += torch.where(x == 0, torch.zeros_like(x), x).float()
        return total.to(self.inputs[0].dtype)


class _Engine:
    """The engine worker's agreement and table refresh, recorded."""

    def _agree_draft_tokens(self, outputs):
        self.calls.append("engine_agree")
        outputs["next_draft_tokens"] = outputs["next_draft_tokens"].clone()

    def _refresh_ctx_block_tables(self, attn_metadata, num_seqs, request_ids=None):
        self.calls.append("engine_refresh")
        return True


class _Log:
    def __init__(self):
        self.info_lines, self.warnings = [], []

    def info_once(self, msg, key):
        if msg not in self.info_lines:
            self.info_lines.append(msg)

    def info(self, msg):
        self.info_lines.append(msg)

    def warning_once(self, msg, key):
        self.warnings.append(msg)


def _namespace(**switches) -> dict:
    """The module's constants, ``ZeroPaddedReduce`` and the worker's methods (in a class over
    ``_Engine``) with stand-ins for the kernels and the all-gather; ``switches`` override the
    module aliases."""
    tree = _tree(_MODELING)
    ns = _constants(tree)
    for key, alias in _SWITCHES.items():
        ns[alias] = switches.get(key, ns[alias])
    gathered = []

    def allgather_module(tensor, mapping, dim, sizes):
        gathered.append(tensor.clone())
        return torch.cat([tensor] * TP, dim=-1)

    ns.update(
        Any=object,
        Optional=__import__("typing").Optional,
        logger=_Log(),
        k31_dflash2_shard_top32=_shard_top32,
        k31_dflash2_merge_top32=_merge_top32,
        allgather_module=allgather_module,
        _gathered=gathered,
    )
    ns.update(_exec(ns, tree, "_pairs_in_id_order"))
    # draft_index_pairs (shipped on): the selection kernel's id-order pairs, whose reference is
    # _pairs_in_id_order of the shard's top-32.
    ns["k31_dflash2_shard_index_pairs"] = lambda logits, offset: ns["_pairs_in_id_order"](
        _shard_top32(logits, TOP_K, offset, True)
    )
    (cls,) = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "ZeroPaddedReduce"]
    (worker,) = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "DFlashWorker"]
    methods = {n.name: n for n in worker.body if isinstance(n, ast.FunctionDef)}
    body = "\n".join(
        "    " + line for m in _METHODS for line in ast.unparse(methods[m]).splitlines()
    )
    source = ast.unparse(cls) + f"\n\nclass Worker(_Engine):\n{body}\n"
    ns["_Engine"] = _Engine
    exec(compile(source, str(_MODELING), "exec"), ns)
    return ns


def _worker(ns, rank=0, agreement=None, pairs=None, count=None):
    w = ns["Worker"]()
    w.calls = []
    w._vocab_shards, w._vocab_shard, w.mapping = TP, rank, None
    w._draft_logits_are_sharded = lambda logits, md: logits.shape[-1] < VOCAB
    w._pairs_gather, w._draft_agreement = pairs, agreement
    w._agree_count, w._agree_pending, w._agree_steps = count, None, 0
    w._batch_to_slot_buffer, w._dummy_slot = None, 99
    return w


def _special(values):
    flat = values.view(-1)
    flat[1:7:2] = torch.tensor([-0.0, float("-inf"), 1e-40])
    return values


def _same_bits_up_to_zero_sign(a, b):
    a = torch.where(a == 0, torch.zeros_like(a), a)
    b = torch.where(b == 0, torch.zeros_like(b), b)
    return torch.equal(a.view(torch.int32), b.view(torch.int32))


def test_the_switches_are_logged_generation_constants_as_shipped():
    tree = _tree(_MODELING)
    ns = _constants(tree)
    generation = ns["_CONSTANTS"]["generation"]
    ns.update(_exec(ns, tree, "constants_line"))
    line = ns["constants_line"]("generation")
    for key, alias in _SWITCHES.items():
        shipped = key != "draft_agree_check"
        assert generation[key] is shipped, key
        assert ns[alias] is shipped, alias
        assert f"{key}={shipped}" in line, key
    assert ns["_DRAFT_AGREEMENT_WIDTH"] // 4 >= 512  # a wide block clears the earlier buffer


@pytest.mark.parametrize("rows", [5, 10, 45, 55, 80])
def test_pairs_gather_is_the_all_gather(rows):
    ns = _namespace()
    zpr, width = ns["ZeroPaddedReduce"], 2 * TOP_K
    recording = _Recording()
    ranks = [
        zpr(recording, r * width, width, torch.zeros(80, TP * width, dtype=torch.float32))
        for r in range(TP)
    ]
    g = torch.Generator().manual_seed(rows)
    shares = []
    for r in range(TP):
        share = torch.randn(rows, width, generator=g)
        share[:, 0::2] = torch.randint(
            r * SHARD, (r + 1) * SHARD, (rows, TOP_K), generator=g
        ).float()
        shares.append(_special(share))
    for rank, share in zip(ranks, shares):
        assert rank.fits(rows, width)
        assert rank(share, rows).shape == (rows, TP * width)
    assert _same_bits_up_to_zero_sign(recording.reduce(), torch.cat(shares, dim=1))
    for r, rank in enumerate(ranks):  # only the rank's own columns were ever written
        others = torch.ones(TP * width, dtype=torch.bool)
        others[r * width : (r + 1) * width] = False
        assert not rank.buffer[:, others].any()


def test_agreement_broadcasts_rank_zero_drafts():
    ns = _namespace()
    recording = _Recording()
    ranks = [
        ns["ZeroPaddedReduce"](
            recording, 0 if r == 0 else None, K, torch.zeros(16, 2048, dtype=torch.float32)
        )
        for r in range(TP)
    ]
    drafts = [torch.randint(0, VOCAB, (9, K), dtype=torch.int32) for _ in range(TP)]
    drafts[0][0, 0] = VOCAB - 1
    for rank, mine in zip(ranks, drafts):
        rank(mine, 9)
    assert torch.equal(recording.reduce()[:, :K].to(torch.int32), drafts[0])
    assert all(not rank.buffer.any() for rank in ranks[1:])
    for rows, cols in ((9, K), (16, K), (17, K), (0, K), (9, K - 1), (9, K + 1)):
        assert len({rank.fits(rows, cols) for rank in ranks}) == 1


def test_build_refuses_what_it_cannot_serve(monkeypatch):
    ns = _namespace()
    zpr = ns["ZeroPaddedReduce"]
    mnnvl = SimpleNamespace(
        dtype=torch.float32,
        mapping=SimpleNamespace(tp_size=TP),
        get_required_workspace_size=lambda rows, width, world, dtype: rows * width * world * 4,
    )
    ns["get_or_scale_allreduce_mnnvl_workspace"] = lambda m, d: {
        "buffer_size_bytes": 80 * 512 * TP * 4
    }
    cpu = torch.device("cpu")
    assert zpr.build(SimpleNamespace(), 0, 64, 512, 80, torch.float32, cpu) is None
    ar = SimpleNamespace(mnnvl_allreduce=mnnvl)
    assert zpr.build(ar, 0, 64, 512, 80, torch.bfloat16, cpu) is None
    assert zpr.build(ar, 0, 64, 512, 81, torch.float32, cpu) is None
    built = zpr.build(ar, 7 * 64, 64, 512, 80, torch.float32, cpu)
    assert built.buffer.shape == (80, 512) and not built.buffer.any()
    assert zpr.build(ar, None, K, 7, 4, torch.float32, cpu).buffer.shape == (4, 8)
    with pytest.raises(ValueError):
        zpr.build(ar, 4, K, 7, 4, torch.float32, cpu)


@pytest.mark.parametrize("bound", [False, True])
def test_the_sharded_branch_gathers_through_the_zero_padded_copy(bound):
    """Every rank's branch returns the full-vocab top-32 (ids, values, order) whether the pairs
    go through the zero-padded all-reduce or the all-gather. The rows are the shard test's tie
    kinds but its signed-zero row: a -0.0 pair value arrives as +0.0 through the all-reduce."""
    ns = _namespace()
    n = 2  # 10 rows: the tie kinds before "subnormal"
    logits = _logits(n, K, seed=20261118)
    shards = [logits[..., r * SHARD : (r + 1) * SHARD].contiguous() for r in range(TP)]
    full = _worker(ns)
    full._vocab_shards = 1
    want_ids, want_unary, _ = full._dflash2_global_top_k(logits.float(), None, TOP_K, VOCAB)
    recording = _Recording()
    workers = []
    for r in range(TP):
        pairs = None
        if bound:
            pairs = ns["ZeroPaddedReduce"](
                recording, r * 2 * TOP_K, 2 * TOP_K, torch.zeros(16 * K, TP * 2 * TOP_K)
            )
        workers.append(_worker(ns, r, pairs=pairs))
    # Pass 1 stages every rank's pairs; pass 2 hands each rank the reduced buffer.
    for w, shard in zip(workers, shards):
        w._dflash2_global_top_k(shard, None, TOP_K, VOCAB)
    if bound:
        reduced = recording.reduce()
        for w in workers:
            w._pairs_gather.all_reduce = lambda buf, reduced=reduced: reduced[: buf.shape[0]]
        assert not ns["_gathered"]
    else:
        everyone = torch.cat(ns["_gathered"], dim=-1)
        ns["allgather_module"] = lambda tensor, mapping, dim, sizes: everyone
    for rank, (w, shard) in enumerate(zip(workers, shards)):
        ids, unary, block = w._dflash2_global_top_k(shard, None, TOP_K, VOCAB)
        assert _equal(ids, want_ids), f"rank {rank}: ids or their order differ"
        assert _equal(unary, want_unary), f"rank {rank}: values differ"
        assert block.dtype == torch.float32 and block.shape == (n, K, VOCAB)


def test_rows_past_the_buffer_take_the_all_gather():
    ns = _namespace()
    pairs = ns["ZeroPaddedReduce"](_Recording(), 0, 2 * TOP_K, torch.zeros(2 * K, TP * 2 * TOP_K))
    w = _worker(ns, 0, pairs=pairs)
    logits = _logits(3, K, seed=3)[..., :SHARD].contiguous()
    w._dflash2_global_top_k(logits, None, TOP_K, VOCAB)
    assert len(ns["_gathered"]) == 1 and not pairs.buffer.any()


def _outputs(rows, seed):
    g = torch.Generator().manual_seed(seed)
    drafts = torch.randint(0, VOCAB, (rows, K), dtype=torch.int32, generator=g)
    new_tokens = torch.cat([torch.arange(rows, dtype=torch.int32)[:, None], drafts], dim=1)
    return {"next_draft_tokens": drafts, "next_new_tokens": new_tokens}


def test_the_agreement_override():
    ns = _namespace()
    rank0 = _outputs(4, 0)
    agreed = rank0["next_draft_tokens"].clone()
    reduce_to = torch.zeros(16, 2048)
    reduce_to[:4, :K] = agreed.float()
    zpr = ns["ZeroPaddedReduce"](
        lambda buf: reduce_to[: buf.shape[0]], None, K, torch.zeros(16, 2048)
    )
    w = _worker(ns, 3, agreement=zpr)
    mine = _outputs(4, 1)
    w._agree_draft_tokens(mine)
    assert torch.equal(mine["next_draft_tokens"], agreed)
    assert mine["next_draft_tokens"].dtype == torch.int32
    assert torch.equal(mine["next_new_tokens"][:, 1:], agreed)
    assert torch.equal(mine["next_new_tokens"][:, 0], torch.arange(4, dtype=torch.int32))
    assert w.calls == []
    # Unbound, or a draft length the buffer was not built for: the engine's all-gather.
    for agreement, k in ((None, K), (zpr, K - 1)):
        w = _worker(ns, 3, agreement=agreement)
        out = _outputs(4, 2)
        out["next_draft_tokens"] = out["next_draft_tokens"][:, :k]
        out["next_new_tokens"] = out["next_new_tokens"][:, : k + 1]
        w._agree_draft_tokens(out)
        assert w.calls == ["engine_agree"]


@pytest.mark.parametrize("zero_padded", [False, True])
def test_the_census_counts_real_and_padding_rows_apart(zero_padded, monkeypatch):
    ns = _namespace()
    rows = 6
    local = _outputs(rows, 5)
    agreed = local["next_draft_tokens"].clone()
    agreed[1, 0] += 1  # a real row that differed
    agreed[5, 4] += 1  # a padding row that differed
    count = torch.zeros(4, dtype=torch.int64)
    zpr = None
    if zero_padded:
        reduce_to = torch.zeros(8, 2048)
        reduce_to[:rows, :K] = agreed.float()
        zpr = ns["ZeroPaddedReduce"](
            lambda buf: reduce_to[: buf.shape[0]], None, K, torch.zeros(8, 2048)
        )
    w = _worker(ns, 2, agreement=zpr, count=count)
    w._batch_to_slot_buffer = SimpleNamespace(tensor=torch.tensor([3, 0, 7, 1, 99, 99, 5]))

    def engine(self, outputs):
        outputs["next_draft_tokens"] = agreed.clone()

    monkeypatch.setattr(_Engine, "_agree_draft_tokens", engine)
    w._agree_draft_tokens(local)
    assert count.tolist() == [1, 4, 1, 2]


def test_no_census_without_the_check():
    ns = _namespace()
    w = _worker(ns, 1)
    out = _outputs(3, 7)
    w._agree_draft_tokens(out)
    assert w.calls == ["engine_agree"] and w._agree_count is None
    assert w._refresh_ctx_block_tables(None, 3) is True
    assert w.calls == ["engine_agree", "engine_refresh"] and w._agree_pending is None


class _Event:
    def __init__(self):
        self.done = False

    def record(self):
        pass

    def query(self):
        return self.done


def test_the_census_reads_out_every_period_without_a_sync():
    ns = _namespace()
    events = []

    def event():
        events.append(_Event())
        return events[-1]

    ns["prefer_pinned"] = lambda: False
    ns["torch"] = SimpleNamespace(
        cuda=SimpleNamespace(is_current_stream_capturing=lambda: False, Event=event),
        empty=lambda *a, **k: torch.empty(*a, dtype=k["dtype"]),
        int64=torch.int64,
    )
    count = torch.tensor([0, 30, 0, 6])
    w = _worker(ns, 4, count=count)
    period = ns["_DRAFT_AGREE_PERIOD"]
    w._report_draft_agreement()  # step 1: a copy is put in flight
    assert len(events) == 1 and w._agree_pending is not None
    count.copy_(torch.tensor([1, 31, 0, 6]))  # the device moves on; the copy holds step 1's
    for _ in range(period - 1):
        w._report_draft_agreement()  # not complete yet: nothing logged, nothing new in flight
    assert len(events) == 1 and not ns["logger"].info_lines
    events[0].done = True
    w._report_draft_agreement()  # step period + 1: logs step 1's totals, puts the next copy out
    assert ns["logger"].info_lines == [ns["_DRAFT_AGREE_LINE"].format(4, 0, 30, 0, 6, 1)]
    assert len(events) == 2 and not ns["logger"].warnings
    events[1].done = True
    w._report_draft_agreement()
    assert ns["logger"].info_lines[-1] == ns["_DRAFT_AGREE_LINE"].format(4, 1, 31, 0, 6, period + 1)
    assert len(ns["logger"].warnings) == 1


def _required(rows, width, world, dtype):
    """``MNNVLAllReduce.get_required_workspace_size`` at the default 1 MiB one-shot threshold."""
    one_shot = rows * width * world * dtype.itemsize
    if one_shot <= 1 << 20:
        return one_shot
    return 2 * -(-rows // world) * world * width * dtype.itemsize


class _Exchange:
    def __init__(self, rank, dtype=torch.float32):
        mnnvl = SimpleNamespace(
            dtype=dtype, mapping=SimpleNamespace(tp_size=TP), get_required_workspace_size=_required
        )
        self.all_reduce = SimpleNamespace(mnnvl_allreduce=mnnvl)
        self.tp_size, self.tp_rank = TP, rank


def _bindable(ns, rank, exchange_rank=None, **attrs):
    w = _worker(ns, rank)
    w._ctx_len = torch.zeros(1)
    w.max_draft_len = K
    w.agree_draft_tokens_across_tp = True
    exchange = _Exchange(rank if exchange_rank is None else exchange_rank)
    w.target_vocab_parallel_head = SimpleNamespace(_exchange=exchange, vocab_size=VOCAB)
    w._draft_agreement_mapping = lambda: SimpleNamespace(tp_size=TP, tp_rank=rank)
    for k, v in attrs.items():
        setattr(w, k, v)
    return w


_DRAFTER = SimpleNamespace(
    context_kv_only=False,
    has_candidate_selector=True,
    candidate_selector=SimpleNamespace(top_k=TOP_K),
)


@pytest.mark.parametrize("pairs,agree,check", list(itertools.product((False, True), repeat=3)))
def test_each_copy_binds_under_its_switch(pairs, agree, check):
    ns = _namespace(
        draft_pairs_zero_padded=pairs, draft_agree_zero_padded=agree, draft_agree_check=check
    )
    ns["get_or_scale_allreduce_mnnvl_workspace"] = lambda m, d: {"buffer_size_bytes": 1 << 20}
    for rank in (0, 5):
        w = _bindable(ns, rank)
        w._bind_zero_padded_copies(_DRAFTER, 16)
        assert (w._agree_count is not None) is check
        if not pairs:
            assert w._pairs_gather is None
        else:
            g = w._pairs_gather
            assert (g.offset, g.columns, tuple(g.buffer.shape)) == (rank * 64, 64, (16 * K, 512))
            assert g.buffer.dtype == torch.float32
        if not agree:
            assert w._draft_agreement is None
        else:
            a = w._draft_agreement
            assert (a.offset, a.columns, tuple(a.buffer.shape)) == (
                0 if rank == 0 else None,
                K,
                (16, ns["_DRAFT_AGREEMENT_WIDTH"]),
            )
    lines = ns["logger"].info_lines
    assert any("top-32 pairs gather" in line for line in lines) is pairs
    assert any("draft-token agreement" in line for line in lines) is agree


def test_nothing_binds_off_the_exchange_or_its_rank_order():
    ns = _namespace(draft_pairs_zero_padded=True, draft_agree_zero_padded=True)
    ns["get_or_scale_allreduce_mnnvl_workspace"] = lambda m, d: {"buffer_size_bytes": 1 << 20}
    cases = [
        _bindable(ns, 2, exchange_rank=3),  # the exchange's group order is not the worker's
        _bindable(ns, 2, target_vocab_parallel_head=None),  # no vocab-parallel selection
    ]
    small = _bindable(ns, 2)
    ns["get_or_scale_allreduce_mnnvl_workspace"] = lambda m, d: {"buffer_size_bytes": 4096}
    for w in cases:
        w._bind_zero_padded_copies(_DRAFTER, 16)
        assert w._pairs_gather is None and w._draft_agreement is None
    small._bind_zero_padded_copies(_DRAFTER, 16)  # a workspace too small for its rows
    assert small._pairs_gather is None and small._draft_agreement is None
    ctx_only = _bindable(ns, 2)
    ctx_only._bind_zero_padded_copies(
        SimpleNamespace(**{**vars(_DRAFTER), "context_kv_only": True}), 16
    )
    assert ctx_only._pairs_gather is None and ctx_only._draft_agreement is None


def test_the_info_lines_are_the_readmes():
    ns = _constants(_tree(_MODELING))
    readme = _README.read_text()
    for what, rows, cols in (("top-32 pairs gather", 80, 512), ("draft-token agreement", 16, 2048)):
        assert f"\n{ns['_DRAFT_ZERO_PADDED_LINE'].format(what, rows, cols)}\n" in readme
    assert f"\n{ns['_DRAFT_AGREE_LINE'].format(0, 0, 8640, 0, 960, 1001)}\n" in readme
