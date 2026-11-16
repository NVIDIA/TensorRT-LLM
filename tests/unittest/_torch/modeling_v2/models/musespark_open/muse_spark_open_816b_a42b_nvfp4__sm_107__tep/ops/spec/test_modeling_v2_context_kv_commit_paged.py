# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for ``context_kv_commit_paged`` (fast path ``paged_context_commit``).

The commit over a paged drafter context must leave ``_ctx_len`` and the
lengths the drafter reads bit for bit as ``DFlashWorker._commit_context`` does
-- the add, the whole-tensor clamp, the dummy slot's zero, the context base
(``TRTLLM_DFLASH_WINDOWED_CTX_ARENA``), the allocation limit -- and, where the
unified cache's check fails, end the process with a CUDA error as the built-in's
device assert does.
"""

import subprocess
import sys
import textwrap
import types

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_107__tep.kernels.context_kv_fork import (  # noqa: E501
    fork as fork_mod,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_107__tep.kernels.verify_fold import (  # noqa: E501
    fold as fold_mod,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_107__tep.ops.spec import (  # noqa: E501
    context_kv_commit_paged as mod,
)
from tensorrt_llm._torch.speculative.dflash import DFlashWorker

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")

PAGE = 32
BLOCK = 8
MAX_CTX = 4096


class _Worker:
    """The worker state and the built-in methods ``_commit_context`` runs, borrowed from ``DFlashWorker``."""

    _commit_context = DFlashWorker._commit_context
    _advance_ctx_len = DFlashWorker._advance_ctx_len
    _empty_dummy_slot = DFlashWorker._empty_dummy_slot
    _settle_context = DFlashWorker._settle_context
    _drafter_context_lengths = DFlashWorker._drafter_context_lengths
    _check_block_within_allocation = staticmethod(DFlashWorker._check_block_within_allocation)

    def __init__(self, ctx_len, block_counts, unified, ctx_base=None):
        self._ctx_len = ctx_len
        self._max_ctx = MAX_CTX
        self._dummy_slot = ctx_len.numel() - 1
        self._ctx_base = ctx_base
        self._ctx_keep = False
        self._ctx_paged = True
        self._ctx_block_tables = None if block_counts is None else object()
        self._ctx_block_counts = block_counts
        self._ctx_page_size = PAGE
        self._compute_block_size = BLOCK
        self._ctx_unified_cache = unified


def _case(gens, seed, limit=True, counts_dtype=torch.int32, dummies=1, base=False):
    g = torch.Generator(device="cuda").manual_seed(seed)
    num_slots = gens + 6
    dummy = num_slots - 1
    ctx_len = torch.randint(0, MAX_CTX - 4, (num_slots,), generator=g, device="cuda")
    ctx_len[: num_slots // 4] = MAX_CTX - torch.randint(
        0, 3, (num_slots // 4,), generator=g, device="cuda"
    )
    ctx_len[dummy] = 0
    slots = torch.randperm(num_slots - 1, generator=g, device="cuda")[:gens].contiguous()
    if dummies and gens > 1:
        # CUDA-graph padding rows: the dummy slot, shared.
        slots[-dummies:] = dummy
    counts = torch.randint(1, 9, (gens,), generator=g, device="cuda").to(counts_dtype)
    rows = torch.arange(gens, device="cuda")
    ctx_base = None
    if base:
        # A base at or below each slot's length; the dummy's is stale (the commit zeroes it).
        ctx_base = (ctx_len * torch.rand(num_slots, generator=g, device="cuda")).long()
        ctx_base[dummy] = 7
    block_counts = None
    if limit:
        # Enough pages for every request's advanced length plus a block.
        need = (ctx_len[slots] + counts.long() + BLOCK + PAGE - 1) // PAGE
        block_counts = torch.zeros(gens + 4, dtype=torch.int64, device="cuda")
        block_counts[:gens] = need + torch.randint(0, 3, (gens,), generator=g, device="cuda")
    return ctx_len, ctx_base, slots, counts, rows, block_counts, dummy


@pytest.mark.parametrize("base", [False, True], ids=["no-base", "base"])
@pytest.mark.parametrize("counts_dtype", [torch.int32, torch.int64], ids=["i32", "i64"])
@pytest.mark.parametrize("limit", [True, False], ids=["limit", "no-limit"])
@pytest.mark.parametrize("gens", [1, 3, 16, mod.MAX_REQUESTS])
def test_the_commit_is_the_builtin_method(gens, limit, counts_dtype, base):
    ctx_len, ctx_base, slots, counts, rows, block_counts, dummy = _case(
        gens, gens, limit, counts_dtype, base=base
    )
    want_len = ctx_len.clone()
    want_base = None if ctx_base is None else ctx_base.clone()
    worker = _Worker(want_len, block_counts, unified=True, ctx_base=want_base)
    want = worker._commit_context(slots, counts.long(), rows)
    out = torch.full((gens,), -5, dtype=torch.int64, device="cuda")
    args = (ctx_len, ctx_base, slots, counts, rows, block_counts, PAGE, BLOCK, MAX_CTX, dummy, True)
    mod.context_kv_commit_paged.is_valid(*args, out)
    mod.context_kv_commit_paged(*args, out)
    torch.cuda.synchronize()
    assert torch.equal(ctx_len, want_len)
    assert (ctx_base is None) or torch.equal(ctx_base, want_base)
    assert torch.equal(out, want)


def test_the_reference_is_the_builtin_method():
    ctx_len, ctx_base, slots, counts, rows, block_counts, dummy = _case(16, 4, base=True)
    want_len, want_base = ctx_len.clone(), ctx_base.clone()
    worker = _Worker(want_len, block_counts, unified=True, ctx_base=want_base)
    want = worker._commit_context(slots, counts.long(), rows)
    out = torch.empty(16, dtype=torch.int64, device="cuda")
    mod.context_kv_commit_paged.reference(
        ctx_len, ctx_base, slots, counts, rows, block_counts, PAGE, BLOCK, MAX_CTX, dummy, True, out
    )
    assert torch.equal(ctx_len, want_len) and torch.equal(ctx_base, want_base)
    assert torch.equal(out, want)


def test_a_length_past_its_allocation_is_clamped_without_the_check():
    ctx_len, ctx_base, slots, counts, rows, block_counts, dummy = _case(4, 9, dummies=0)
    block_counts[0] = 0
    want_len = ctx_len.clone()
    want = _Worker(want_len, block_counts, unified=False)._commit_context(
        slots, counts.long(), rows
    )
    out = torch.empty(4, dtype=torch.int64, device="cuda")
    mod.context_kv_commit_paged(
        ctx_len, None, slots, counts, rows, block_counts, PAGE, BLOCK, MAX_CTX, dummy, False, out
    )
    torch.cuda.synchronize()
    assert out[0].item() == 0 and torch.equal(out, want) and torch.equal(ctx_len, want_len)


@pytest.mark.parametrize("fused", [True, False], ids=["fused", "builtin"])
def test_a_length_past_its_allocation_ends_the_process_under_the_check(fused):
    script = textwrap.dedent(
        f"""
        import importlib
        import torch
        mod = importlib.import_module({mod.__name__!r})
        ctx_len = torch.tensor([100, 200, 0], dtype=torch.int64, device="cuda")
        slots = torch.tensor([0, 1], dtype=torch.int64, device="cuda")
        counts = torch.tensor([3, 3], dtype=torch.int32, device="cuda")
        rows = torch.tensor([0, 1], dtype=torch.int64, device="cuda")
        block_counts = torch.tensor([1, 64], dtype=torch.int64, device="cuda")
        out = torch.empty(2, dtype=torch.int64, device="cuda")
        op = mod.context_kv_commit_paged if {fused} else mod.context_kv_commit_paged.reference
        op(ctx_len, None, slots, counts, rows, block_counts, {PAGE}, {BLOCK}, {MAX_CTX}, 2, True, out)
        torch.cuda.synchronize()
        print("NO ERROR", out.tolist())
        """
    )
    done = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=600
    )
    assert done.returncode != 0, done.stdout
    assert "NO ERROR" not in done.stdout
    assert "CUDA" in done.stderr or "cuda" in done.stderr, done.stderr[-2000:]


def test_the_fork_takes_the_commit_only_under_the_fast_path():
    ctx_len, ctx_base, slots, counts, rows, block_counts, dummy = _case(3, 2)
    worker = _Worker(ctx_len, block_counts, unified=True)
    fork = types.SimpleNamespace(paged_commit_on=False)
    takes = fork_mod.ContextKvFork._commits_paged
    assert not takes(fork, worker, slots, counts)
    fork.paged_commit_on = True
    assert takes(fork, worker, slots, counts)
    # The windowed arena's base (TRTLLM_DFLASH_WINDOWED_CTX_ARENA) is taken too.
    worker._ctx_base = torch.zeros_like(ctx_len)
    assert takes(fork, worker, slots, counts)
    worker._ctx_base = torch.zeros(ctx_len.numel(), dtype=torch.int32, device="cuda")
    assert not takes(fork, worker, slots, counts)


def test_a_worker_class_with_its_own_commit_is_refused():
    class OwnCommit(DFlashWorker):
        def _drafter_context_lengths(self, slots, rows):
            return super()._drafter_context_lengths(slots, rows)

    worker = OwnCommit.__new__(OwnCommit)
    reason = fork_mod.paged_commit_unsupported_reason(worker)
    assert reason is not None and "_drafter_context_lengths" in reason
    assert fork_mod.paged_commit_unsupported_reason(DFlashWorker.__new__(DFlashWorker)) is None


# --------------------------------------------------------------------------
# The commit's launch writing the drafter preamble too (``PagedPreamble``): on a forked folded
# step it must leave everything bit for bit as the commit followed by the drafter forward's own
# preamble (windowed_block_decode) and the prepare's ``_seq_lens_cuda`` fill.
# --------------------------------------------------------------------------
def _qpos(gens, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    qpos = torch.randint(0, MAX_CTX, (gens, BLOCK), generator=g, device="cuda")
    # A few positions past int32: the casts must wrap alike.
    qpos.view(-1)[::7] += 3 << 31
    return qpos


def _preamble(gens, batch_idx, qpos, seq_lens, value=BLOCK):
    outs = (
        torch.full((n,), -9, dtype=torch.int32, device="cuda")
        for n in (gens, gens, gens, gens * qpos.shape[1])
    )
    return mod.PagedPreamble(batch_idx, qpos, seq_lens, value, *outs)


def _separate_preamble(out, batch_idx, qpos, seq_lens, value=BLOCK):
    """What follows the commit when its launch does not write the preamble: windowed_block_decode's
    statements (``block_size`` the query block) and the prepare's ``_seq_lens_cuda`` fill."""
    gens = out.shape[0]
    cache_seqlens_i32 = out[:gens].to(torch.int32)
    seq_lens_after = cache_seqlens_i32 + qpos.shape[1]
    cache_batch_idx_i32 = batch_idx.to(torch.int32)
    query_positions_flat_i32 = qpos.reshape(-1).to(torch.int32)
    seq_lens[:gens] = value
    return cache_seqlens_i32, seq_lens_after, cache_batch_idx_i32, query_positions_flat_i32


@pytest.mark.parametrize("seq_dtype", [torch.int32, torch.int64], ids=["seq-i32", "seq-i64"])
@pytest.mark.parametrize("base", [False, True], ids=["no-base", "base"])
@pytest.mark.parametrize("limit", [True, False], ids=["limit", "no-limit"])
@pytest.mark.parametrize("gens", [1, 3, 16, mod.MAX_REQUESTS])
def test_the_commit_with_the_preamble_is_the_commit_then_the_preamble(gens, limit, base, seq_dtype):
    ctx_len, ctx_base, slots, counts, rows, block_counts, dummy = _case(
        gens, 100 + gens, limit, torch.int32, dummies=min(2, gens - 1), base=base
    )
    qpos = _qpos(gens, gens)
    # The drafter's ctx_cache_batch_idx (_drafter_inputs): the rows with block tables, else the slots.
    batch_idx = rows if limit else slots
    start_len = ctx_len.clone()
    start_base = None if ctx_base is None else ctx_base.clone()
    want_len = ctx_len.clone()
    want_base = None if ctx_base is None else ctx_base.clone()
    want = torch.full((gens,), -5, dtype=torch.int64, device="cuda")
    want_seq = torch.full((gens + 5,), -3, dtype=seq_dtype, device="cuda")
    tail = (slots, counts, rows, block_counts, PAGE, BLOCK, MAX_CTX, dummy, True)
    mod.context_kv_commit_paged(want_len, want_base, *tail, want)
    want_i32 = _separate_preamble(want, batch_idx, qpos, want_seq)

    out = torch.full((gens,), -5, dtype=torch.int64, device="cuda")
    seq_lens = torch.full((gens + 5,), -3, dtype=seq_dtype, device="cuda")
    pre = _preamble(gens, batch_idx, qpos, seq_lens)
    mod.context_kv_commit_paged.is_valid(ctx_len, ctx_base, *tail, out, pre)
    mod.context_kv_commit_paged(ctx_len, ctx_base, *tail, out, preamble=pre)
    torch.cuda.synchronize()
    assert torch.equal(ctx_len, want_len)
    assert (ctx_base is None) or torch.equal(ctx_base, want_base)
    assert torch.equal(out, want)
    assert torch.equal(seq_lens, want_seq)
    for got, expected in zip(pre.i32s, want_i32):
        assert got.dtype == torch.int32 and torch.equal(got, expected)

    # The reference with the preamble gives the same.
    ref_len = start_len.clone()
    ref_base = None if start_base is None else start_base.clone()
    ref_out = torch.empty_like(out)
    ref_seq = torch.full_like(seq_lens, -3)
    ref_pre = _preamble(gens, batch_idx, qpos, ref_seq)
    mod.context_kv_commit_paged.reference(ref_len, ref_base, *tail, ref_out, ref_pre)
    assert torch.equal(ref_len, want_len) and torch.equal(ref_out, want)
    assert (ref_base is None) or torch.equal(ref_base, want_base)
    assert torch.equal(ref_seq, want_seq)
    assert all(torch.equal(a, b) for a, b in zip(ref_pre.i32s, want_i32))


def test_the_preamble_takes_an_int32_batch_index():
    ctx_len, _, slots, counts, rows, block_counts, dummy = _case(5, 77)
    qpos = _qpos(5, 77)
    tail = (slots, counts, rows, block_counts, PAGE, BLOCK, MAX_CTX, dummy, True)
    want = torch.empty(5, dtype=torch.int64, device="cuda")
    want_seq = torch.zeros(8, dtype=torch.int32, device="cuda")
    want_len = ctx_len.clone()
    mod.context_kv_commit_paged(want_len, None, *tail, want)
    want_i32 = _separate_preamble(want, rows.to(torch.int32), qpos, want_seq)
    out = torch.empty_like(want)
    seq_lens = torch.zeros_like(want_seq)
    pre = _preamble(5, rows.to(torch.int32), qpos, seq_lens)
    mod.context_kv_commit_paged(ctx_len, None, *tail, out, preamble=pre)
    torch.cuda.synchronize()
    assert (
        torch.equal(out, want)
        and torch.equal(seq_lens, want_seq)
        and torch.equal(ctx_len, want_len)
    )
    assert all(torch.equal(a, b) for a, b in zip(pre.i32s, want_i32))


def test_the_preamble_shapes_are_checked():
    ctx_len, _, slots, counts, rows, block_counts, dummy = _case(4, 5)
    tail = (slots, counts, rows, block_counts, PAGE, BLOCK, MAX_CTX, dummy, True)
    out = torch.empty(4, dtype=torch.int64, device="cuda")
    seq_lens = torch.zeros(4, dtype=torch.int32, device="cuda")
    good = _preamble(4, rows, _qpos(4, 5), seq_lens)
    mod.context_kv_commit_paged.is_valid(ctx_len, None, *tail, out, good)
    bad = (
        good._replace(query_positions=good.query_positions[:3]),
        good._replace(query_positions=good.query_positions.to(torch.int32)),
        good._replace(batch_idx=rows[:3]),
        good._replace(seq_lens=seq_lens[:3]),
        good._replace(seq_lens=seq_lens.float()),
        good._replace(num_ctx_i32=good.num_ctx_i32.long()),
        good._replace(query_positions_i32=good.query_positions_i32[:-1]),
    )
    for pre in bad:
        with pytest.raises(ValueError, match="preamble"):
            mod.context_kv_commit_paged.is_valid(ctx_len, None, *tail, out, pre)


@pytest.mark.parametrize("base", [False, True], ids=["no-base", "base"])
@pytest.mark.parametrize("limit", [True, False], ids=["limit", "no-limit"])
def test_the_fork_commit_with_the_preamble_is_its_commit_then_the_preamble(limit, base):
    """``ContextKvFork.commit_paged_preamble`` (the verify fold's forked step) against
    ``ContextKvFork.commit`` (paged_context_commit) followed by the drafter forward's preamble and the
    prepare's fill; and the launch counts the fork logs."""
    gens = 12
    state = _case(gens, 900 + limit + 2 * base, limit, torch.int32, dummies=2, base=base)
    ctx_len, ctx_base, slots, counts, rows, block_counts, dummy = state
    qpos = _qpos(gens, 900)
    results = []
    for fused in (False, True):
        worker = _Worker(
            ctx_len.clone(), block_counts, unified=True,
            ctx_base=None if ctx_base is None else ctx_base.clone(),
        )  # fmt: skip
        fork = fork_mod.ContextKvFork(worker, None, "verify")
        fork.paged_commit_on = True
        launched = fork_mod.Launched(
            torch.cuda.current_stream(), None, gens, BLOCK - 1, "verify", None
        )
        seq_lens = torch.full((gens + 4,), -3, dtype=torch.int32, device="cuda")
        # The drafter's ctx_cache_batch_idx, as _drafter_inputs gives it.
        batch_idx = fork.rows[:gens] if limit else slots
        if fused:
            num_ctx, i32s = fork.commit_paged_preamble(
                launched, slots, counts, batch_idx, qpos, seq_lens, BLOCK, BLOCK
            )
            fork.join(launched)
        else:
            num_ctx = fork.commit(launched, slots, counts)
            fork.join(launched)
            i32s = _separate_preamble(num_ctx, batch_idx, qpos, seq_lens)
        torch.cuda.synchronize()
        results.append((worker._ctx_len, worker._ctx_base, num_ctx, seq_lens, i32s))
        assert fork.paged_commits == {"launches": 1, "with preamble": int(fused)}
    (len_a, base_a, num_a, seq_a, i32_a), (len_b, base_b, num_b, seq_b, i32_b) = results
    assert torch.equal(len_a, len_b) and torch.equal(num_a, num_b) and torch.equal(seq_a, seq_b)
    assert (base_a is None) or torch.equal(base_a, base_b)
    assert all(torch.equal(a, b) for a, b in zip(i32_a, i32_b))


def test_the_fork_refuses_the_preamble_commit_off_the_fast_path():
    ctx_len, _, slots, counts, rows, block_counts, dummy = _case(3, 2)
    worker = _Worker(ctx_len, block_counts, unified=True)
    fork = fork_mod.ContextKvFork(worker, None, "verify")
    launched = fork_mod.Launched(torch.cuda.current_stream(), None, 3, BLOCK - 1, "verify", None)
    seq_lens = torch.zeros(3, dtype=torch.int32, device="cuda")
    with pytest.raises(RuntimeError, match="paged_context_commit"):
        fork.commit_paged_preamble(
            launched, slots, counts, rows, _qpos(3, 2), seq_lens, BLOCK, BLOCK
        )


def test_the_commit_with_the_preamble_replays_in_a_graph_and_timing(capsys):
    """Captured in a CUDA graph the launch replays the eager results; a timing print of the one launch
    vs the commit followed by the five preamble / fill launches (in graphs) at 8 and 16 requests."""
    lines = []
    for gens in (8, 16):
        ctx_len, ctx_base, slots, counts, rows, block_counts, dummy = _case(
            gens, 40 + gens, base=True
        )
        # Ten calls per replay advance the lengths ten times: allocations the check never trips on.
        block_counts.fill_(1 << 30)
        qpos = _qpos(gens, gens)
        tail = (slots, counts, rows, block_counts, PAGE, BLOCK, MAX_CTX, dummy, True)
        keep_len, keep_base = ctx_len.clone(), ctx_base.clone()
        seq_lens = torch.zeros(gens + 4, dtype=torch.int32, device="cuda")
        out = torch.empty(gens, dtype=torch.int64, device="cuda")
        pre = _preamble(gens, rows, qpos, seq_lens)

        def fused():
            mod.context_kv_commit_paged(ctx_len, ctx_base, *tail, out, preamble=pre)

        def separate():
            mod.context_kv_commit_paged(ctx_len, ctx_base, *tail, out)
            _separate_preamble(out, rows, qpos, seq_lens)

        def reset():
            ctx_len.copy_(keep_len)
            ctx_base.copy_(keep_base)
            out.fill_(-1)
            seq_lens.fill_(0)
            for t in pre.i32s:
                t.fill_(-1)

        # Eager vs one replay, from the same starting state.
        fused()
        torch.cuda.synchronize()
        eager = (
            ctx_len.clone(),
            ctx_base.clone(),
            out.clone(),
            seq_lens.clone(),
            [t.clone() for t in pre.i32s],
        )
        reset()
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            fused()
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            fused()
        reset()
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(ctx_len, eager[0]) and torch.equal(ctx_base, eager[1])
        assert torch.equal(out, eager[2]) and torch.equal(seq_lens, eager[3])
        assert all(torch.equal(a, b) for a, b in zip(pre.i32s, eager[4]))

        res = {}
        for name, fn in (("one launch", fused), ("commit + preamble", separate)):
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                fn()
            torch.cuda.current_stream().wait_stream(side)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                for _ in range(10):
                    fn()
            times = []
            for _ in range(11):
                ctx_len.copy_(keep_len)
                ctx_base.copy_(keep_base)
                start, end = (
                    torch.cuda.Event(enable_timing=True),
                    torch.cuda.Event(enable_timing=True),
                )
                start.record()
                graph.replay()
                end.record()
                torch.cuda.synchronize()
                times.append(start.elapsed_time(end) * 1e3 / 10)
            res[name] = sorted(times)[5]
        lines.append(
            f"[paged commit + preamble] {gens:2d} requests: one launch {res['one launch']:6.2f} us, "
            f"commit + 5 preamble launches {res['commit + preamble']:6.2f} us per step "
            f"({res['one launch'] - res['commit + preamble']:+.2f})"
        )
    with capsys.disabled():
        print("\n" + "\n".join(lines), flush=True)


# feat/musespark-open-agentx's entry (context_kv_commit_paged_preamble, a PagedPreamble over
# the same launch) and its fold refusal over pages.
@pytest.mark.parametrize("seq_dtype", [torch.int32, torch.int64], ids=["seq-i32", "seq-i64"])
@pytest.mark.parametrize("batch", ["rows", "slots-i32"])
@pytest.mark.parametrize("base", [False, True], ids=["no-base", "base"])
@pytest.mark.parametrize("limit", [True, False], ids=["limit", "no-limit"])
@pytest.mark.parametrize("gens", [1, 3, mod.MAX_REQUESTS])
def test_the_feat_preamble_entry_is_the_commit_and_the_forward_s_preamble(
    gens, limit, base, batch, seq_dtype
):
    ctx_len, ctx_base, slots, counts, rows, block_counts, dummy = _case(
        gens, 3 * gens, limit, base=base
    )
    # The forward's ctx_cache_batch_idx: the batch rows over the manager's block tables, the
    # slots over the worker's own pages (any integer dtype).
    batch_idx = rows if batch == "rows" else slots.to(torch.int32)
    g = torch.Generator(device="cuda").manual_seed(gens)
    query_positions = torch.randint(0, MAX_CTX, (gens, BLOCK), generator=g, device="cuda")
    seq_lens = torch.full((gens + 3,), -9, dtype=seq_dtype, device="cuda")
    want_len = ctx_len.clone()
    want_base = None if ctx_base is None else ctx_base.clone()
    want = _Worker(want_len, block_counts, unified=True, ctx_base=want_base)._commit_context(
        slots, counts.long(), rows
    )
    # windowed_block_decode's own derivations and the prepare's _seq_lens_cuda fill.
    want_i32s = (
        want.to(torch.int32),
        want.to(torch.int32) + BLOCK,
        batch_idx.to(torch.int32),
        query_positions.reshape(-1).to(torch.int32),
    )
    want_seq_lens = seq_lens.clone()
    want_seq_lens[:gens] = BLOCK - 1
    out = torch.full((gens,), -5, dtype=torch.int64, device="cuda")
    i32s = tuple(torch.full_like(t, -3) for t in want_i32s)
    mod.context_kv_commit_paged_preamble(
        ctx_len,
        ctx_base,
        slots,
        counts,
        rows,
        block_counts,
        PAGE,
        BLOCK,
        MAX_CTX,
        dummy,
        True,
        out,
        batch_idx,
        query_positions,
        seq_lens,
        BLOCK - 1,
        BLOCK,
        i32s,
    )
    torch.cuda.synchronize()
    assert torch.equal(ctx_len, want_len)
    assert (ctx_base is None) or torch.equal(ctx_base, want_base)
    assert torch.equal(out, want)
    for got, expected in zip(i32s, want_i32s):
        assert got.dtype == torch.int32 and torch.equal(got, expected)
    assert torch.equal(seq_lens, want_seq_lens)


def _fold_inputs(gens):
    fold = types.SimpleNamespace(
        query_positions=torch.zeros(gens, BLOCK, dtype=torch.int64, device="cuda")
    )
    attn_metadata = types.SimpleNamespace(
        _seq_lens_cuda=torch.zeros(gens + 2, dtype=torch.int32, device="cuda")
    )
    return fold, attn_metadata


def test_the_fold_takes_the_preamble_over_pages_only_where_the_paged_commit_runs():
    ctx_len, _, slots, counts, rows, block_counts, _ = _case(3, 2)
    worker = _Worker(ctx_len, block_counts, unified=True)
    fork = types.SimpleNamespace(paged_commit_on=True)
    fork._commits_paged = types.MethodType(fork_mod.ContextKvFork._commits_paged, fork)
    fold, attn_metadata = _fold_inputs(3)

    def refusal(batch_idx=rows):
        return fold_mod._preamble_refusal(
            worker, fork, fold, attn_metadata, slots, counts, batch_idx, 3, BLOCK
        )

    assert refusal() is None
    assert refusal(slots.to(torch.int32)) is None
    assert "batch index" in refusal(torch.arange(6, device="cuda")[::2])
    fork.paged_commit_on = False
    assert "context_kv_commit_paged" in refusal()
    fork.paged_commit_on = True
    worker._ctx_base = torch.zeros_like(ctx_len)
    assert "windowed" in refusal()
    # The dense full-length arena: the slots index it.
    worker._ctx_base, worker._ctx_paged, worker._ctx_block_tables = None, False, None
    assert refusal(slots) is None
