# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for ``context_kv_commit_paged`` (fast path ``paged_context_commit``).

The commit over a paged drafter context must leave ``_ctx_len`` and the
lengths the drafter reads bit for bit as ``DFlashWorker._commit_context`` does
-- the add, the whole-tensor clamp, the dummy slot's zero, the context base
(``TRTLLM_DFLASH_WINDOWED_CTX_ARENA``), the allocation limit -- and, where the
unified cache's check fails, end the process with a CUDA error as the built-in's
device assert does. ``context_kv_commit_paged_preamble`` must do the same and write the
drafter preamble as ``windowed_block_decode`` and the prepare derive it, and the verify fold
must take it over pages only where the fork's commit is this kernel.
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


@pytest.mark.parametrize("seq_dtype", [torch.int32, torch.int64], ids=["seq-i32", "seq-i64"])
@pytest.mark.parametrize("batch", ["rows", "slots-i32"])
@pytest.mark.parametrize("base", [False, True], ids=["no-base", "base"])
@pytest.mark.parametrize("limit", [True, False], ids=["limit", "no-limit"])
@pytest.mark.parametrize("gens", [1, 3, mod.MAX_REQUESTS])
def test_the_preamble_commit_is_the_commit_and_the_forward_s_preamble(
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
