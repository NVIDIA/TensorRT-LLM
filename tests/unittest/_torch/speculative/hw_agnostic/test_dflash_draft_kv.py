# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Separate-draft-KV policy follows the DFlash backend's storage ownership.

TRTLLM borrows the managed paged pool. VANILLA and FA4 own private context
buffers and must not allocate a redundant manager or be gated on its capacity.
"""

import pytest

from tensorrt_llm._torch.speculative.interface import should_use_separate_draft_kv_cache
from tensorrt_llm.llmapi import DFlashDecodingConfig, EagleDecodingConfig

pytestmark = pytest.mark.cpu_only


@pytest.mark.parametrize(
    ("backend", "uses_managed_pool"),
    [("VANILLA", False), ("FA4", False), ("TRTLLM", True)],
)
def test_dflash_draft_cache_matches_backend_storage(backend: str, uses_managed_pool: bool) -> None:
    config = DFlashDecodingConfig(max_draft_len=7, attention_backend=backend)
    assert config.spec_dec_mode.use_one_engine()
    assert not config._use_shared_kv_cache
    assert config._allow_separate_draft_kv_cache
    assert should_use_separate_draft_kv_cache(config) is uses_managed_pool


def test_default_dflash_has_no_redundant_mirror_or_target_draft_layers() -> None:
    from tensorrt_llm._torch.speculative.utils import get_num_spec_layers

    config = DFlashDecodingConfig(max_draft_len=7)
    assert config.attention_backend == "VANILLA"
    assert should_use_separate_draft_kv_cache(config) is False
    assert get_num_spec_layers(config) == 0


@pytest.mark.parametrize("backend", ["VANILLA", "FA4", "TRTLLM"])
def test_dflash_explicit_cache_opt_out_is_preserved(backend: str) -> None:
    config = DFlashDecodingConfig(max_draft_len=7, attention_backend=backend)
    config._allow_separate_draft_kv_cache = False
    assert should_use_separate_draft_kv_cache(config) is False


def test_eagle3_one_model_still_gets_a_separate_draft_kv_cache():
    """The DFlash exemption must not leak into other one-engine drafters."""
    config = EagleDecodingConfig(max_draft_len=3, speculative_model="/tmp/eagle3-draft")
    assert config.spec_dec_mode.is_eagle3_one_model()
    assert should_use_separate_draft_kv_cache(config) is True


@pytest.mark.parametrize(
    ("bound_to_pool", "reuse_addressable", "reads_rebased_view", "cur", "first_pos", "expected"),
    [
        # Managed pool: the block table is indexed by target position, so a
        # tail that reuse let the target start at first_pos is stored there.
        (True, False, True, 0, 8192, 8192),
        (True, True, True, 0, 8192, 8192),
        # A pool paired with the target's reuse holds the prefix below the
        # tail, so the tail goes on top of it even for a worker that reads
        # from position 0 (the Kimi K3 workers).
        (True, True, False, 0, 8192, 8192),
        # An unpaired pool read from position 0 numbers the tail from 0.
        (True, False, False, 0, 8192, 0),
        # A continuing chunk keeps the slot's stored length.
        (True, False, True, 4096, 4096, 4096),
        # Private arena: every request numbers from 0.
        (False, False, True, 0, 8192, 0),
        (False, False, True, 0, 0, 0),
    ],
)
def test_dflash_ctx_store_start_follows_the_pool_numbering(
    bound_to_pool: bool,
    reuse_addressable: bool,
    reads_rebased_view: bool,
    cur: int,
    first_pos: int,
    expected: int,
) -> None:
    from tensorrt_llm._torch.speculative.dflash import DFlashWorker

    worker = object.__new__(DFlashWorker)
    worker._ctx_block_tables = object() if bound_to_pool else None
    worker._ctx_reuse_addressable = reuse_addressable
    worker._ctx_reads_rebased_view = reads_rebased_view
    assert worker._ctx_store_start(cur, first_pos) == expected


@pytest.mark.parametrize(
    ("cur", "start", "reuse_addressable", "expected"),
    [
        # Unpaired pool, tail stored at a reused prefix's end: the pages below
        # it hold no drafter K/V and are hidden from the block decode.
        (0, 8192, False, 128),
        # A start inside a page rounds the base up; the tail tokens in that
        # page stay stored but unread.
        (0, 8200, False, 129),
        # A continuing chunk numbers from the stored length: no base change.
        (4096, 4096, False, None),
        # A pool paired with the target's reuse holds the prefix's drafter K/V.
        (0, 8192, True, None),
    ],
)
def test_dflash_ctx_base_for_store(
    cur: int, start: int, reuse_addressable: bool, expected: int | None
) -> None:
    from tensorrt_llm._torch.speculative.dflash import DFlashWorker

    worker = object.__new__(DFlashWorker)
    worker._ctx_page_size = 64
    worker._ctx_reuse_addressable = reuse_addressable
    assert worker._ctx_base_for_store(cur, start) == expected


def test_dflash_rebased_ctx_rows_hide_the_reused_prefix() -> None:
    """A tail stored after a reused prefix on an unpaired pool. The block decode
    reads the slot's row rebased to the tail and a context length shortened by
    the prefix, so the never-written prefix pages (poisoned here) are not within
    reach, and every tail position still resolves to the page and offset the
    store wrote it to."""
    import torch

    from tensorrt_llm._torch.speculative.dflash import dflash_rebased_ctx_rows

    page_size, first_pos, tail = 4, 24, 5
    row = torch.tensor([[7, 3, 11, 0, 5, 9, 13, 2, 6, 1, 15, 4, 8, 10, 12, 14]], dtype=torch.int32)
    pool = torch.zeros(16, page_size)
    pool[row[0, : first_pos // page_size].long()] = float("nan")
    for pos in range(first_pos, first_pos + tail):
        pool[row[0, pos // page_size], pos % page_size] = pos

    base_pages = torch.tensor([first_pos // page_size])
    view = dflash_rebased_ctx_rows(row, base_pages)
    num_ctx = first_pos + tail - base_pages.item() * page_size
    assert view.shape == row.shape
    readable = view[0, : -(-num_ctx // page_size)].long()
    assert torch.isfinite(pool[readable]).all()
    for pos in range(num_ctx):
        assert pool[view[0, pos // page_size], pos % page_size] == first_pos + pos
    # A slot without a base reads its row unchanged.
    assert torch.equal(dflash_rebased_ctx_rows(row, torch.zeros(1, dtype=torch.long)), row)


@pytest.mark.parametrize(
    ("first_pos", "tail"),
    [
        # A reused prefix ending inside a 64-token page with a one-token tail:
        # the base rounds up to the next page and the context stays shorter
        # than it for 55 steps.
        (8200, 1),
        # One token into a page: the first write offsets overlap the stored
        # tail, which sits below the base and is never read.
        (8193, 1),
        # A page-aligned prefix end is never short of its base.
        (8192, 1),
    ],
)
def test_dflash_rebased_ctx_view_writes_the_block_into_an_owned_page(
    first_pos: int, tail: int
) -> None:
    """A tail stored after a reused prefix on an unpaired pool, with the mirror
    funded the way the manager funds it: pages up to the one holding the
    context plus the draft tokens, the window below it, nothing past it. The
    block decode writes the block's own K/V at rebased positions
    num_ctx .. num_ctx + block_size through the view it reads, so at every
    context length those positions must resolve to a funded entry, including
    the steps on which the context is still shorter than the rounded-up base
    and the decode reads nothing. Entry ``base`` is unfunded on those steps and
    holds 0, the pool page another request owns."""
    import torch

    from tensorrt_llm._torch.speculative.dflash import DFlashWorker, dflash_rebased_ctx_view

    page_size, block_size, draft_len, window_pages = 64, 3, 3, 64
    worker = object.__new__(DFlashWorker)
    worker._ctx_page_size = page_size
    worker._ctx_reuse_addressable = False
    base = worker._ctx_base_for_store(0, first_pos)
    assert base == -(-first_pos // page_size)
    base_pages = torch.tensor([base])

    def funded_row(ctx_len: int) -> torch.Tensor:
        last = (ctx_len + draft_len - 1) // page_size
        row = torch.zeros(1, base + 8, dtype=torch.int32)
        row[0, last + 1 - window_pages : last + 1] = (
            torch.arange(last + 1 - window_pages, last + 1, dtype=torch.int32) + 1000
        )
        return row

    blind_steps = 0
    for ctx_len in range(first_pos + tail, (base + 1) * page_size):
        row = funded_row(ctx_len)
        view, num_ctx = dflash_rebased_ctx_view(row, base_pages, torch.tensor([ctx_len]), page_size)
        assert view.shape == row.shape
        blind = ctx_len < base * page_size
        blind_steps += blind
        assert num_ctx.item() == max(0, ctx_len - base * page_size)
        if blind:
            assert view[0, 0] == row[0, ctx_len // page_size]
            if (ctx_len + draft_len - 1) // page_size < base:
                # Entry ``base`` is funded only once the draft tokens reach it.
                assert row[0, base] == 0
        for pos in range(num_ctx.item() + block_size):
            entry = view[0, pos // page_size]
            assert entry >= 1000, (ctx_len, pos)
            if pos < num_ctx.item():
                # A read position p is absolute position base * page_size + p.
                assert entry == row[0, (base * page_size + pos) // page_size]
    assert blind_steps == max(0, base * page_size - first_pos - tail)


def test_dflash_write_ctx_len_zero_length_reset_wins_over_a_pending_base() -> None:
    """_store_prefill_context records a chunk's base pages before the overflow
    check that may reset the same slot to length 0 in the same call. The reset
    takes precedence: the slot ends with length 0 and base 0 on the host
    mirrors and the device tensors alike, while a slot that was not reset keeps
    its new length and base."""
    import torch

    from tensorrt_llm._torch.speculative.dflash import DFlashWorker

    worker = object.__new__(DFlashWorker)
    worker._ctx_len_host = [0, 512, 64]
    worker._ctx_base_host = [0, 8, 1]
    worker._ctx_len = torch.tensor(worker._ctx_len_host)
    worker._ctx_base_pages = torch.tensor(worker._ctx_base_host)

    # Slot 1: base recorded, then reset by the overflow check. Slot 2: a
    # continuing chunk with a new base.
    worker._write_ctx_len({1: 0, 2: 96}, {1: 128, 2: 2})

    assert worker._ctx_len_host == [0, 0, 96]
    assert worker._ctx_base_host == [0, 0, 2]
    assert worker._ctx_len.tolist() == [0, 0, 96]
    assert worker._ctx_base_pages.tolist() == [0, 0, 2]


@pytest.mark.parametrize(
    ("bound_to_pool", "page_aligned", "reuse_addressable", "position", "expected"),
    [
        # A disaggregated generation row: the slot starts empty at the first
        # position of the page holding its position and reads from there,
        # paired with reuse or not (the mirror of such a request never looked
        # a prefix up).
        (True, True, False, 8200, (8192, 128)),
        (True, True, True, 8200, (8192, 128)),
        (True, True, False, 8192, (8192, 128)),
        # Clamped to the drafter's context ceiling, base included.
        (True, True, False, 1 << 20, (65536, 1024)),
        # A pool window shorter than a page may already have released that
        # page: the slot starts at the position, reading from the next page.
        (True, False, False, 8200, (8200, 129)),
        # The private arena is slot-addressed and starts at 0.
        (False, True, False, 8200, (0, None)),
    ],
)
def test_dflash_generation_slot_seed_is_absolute_with_a_base(
    bound_to_pool: bool,
    page_aligned: bool,
    reuse_addressable: bool,
    position: int,
    expected: tuple,
) -> None:
    """A generation row that gets a fresh slot (a disaggregated generation
    worker's first step, or a request whose slot was evicted) starts in the
    coordinates of its draft mirror, at a page boundary no further than a page
    below its absolute position, with every page below the base hidden from
    the block decode."""
    from tensorrt_llm._torch.speculative.dflash import DFlashWorker

    worker = object.__new__(DFlashWorker)
    worker._ctx_block_tables = object() if bound_to_pool else None
    worker._ctx_reuse_addressable = reuse_addressable
    worker._ctx_seed_page_aligned = page_aligned
    worker._ctx_page_size = 64
    worker._max_ctx = 65536
    updates, bases = {}, {}
    worker._seed_generation_slot(3, position, updates, bases)
    assert (updates[3], bases.get(3)) == expected


def test_dflash_assign_slot_resets_the_base_of_a_reused_slot() -> None:
    """A slot freed and handed out again inside one batched update starts
    with base 0, not the previous occupant's base."""
    from collections import deque

    from tensorrt_llm._torch.speculative.dflash import DFlashWorker

    worker = object.__new__(DFlashWorker)
    worker._req_to_slot = {7: 0}
    worker._free_slots = deque()
    updates, bases = {}, {}
    assert worker._assign_slot(8, reset=True, updates=updates, bases=bases) is None
    assert worker._assign_slot(7, reset=True, updates=updates, bases=bases) == 0
    assert (updates, bases) == ({0: 0}, {0: 0})


def test_dflash_rollback_restores_the_base_pages() -> None:
    """A failed or warmup forward rolls the slot lengths back to the
    pre-forward snapshot, and the base pages with them, on the device and the
    host mirror; a base seeded during the forward does not survive."""
    import torch

    from tensorrt_llm._torch.speculative.dflash import DFlashWorker

    worker = object.__new__(DFlashWorker)
    worker._ctx_len_host = [0, 512, 64]
    worker._ctx_base_host = [0, 8, 0]
    worker._ctx_len = torch.tensor(worker._ctx_len_host)
    worker._ctx_base_pages = torch.tensor(worker._ctx_base_host)
    worker._req_ctx_pos = {}
    worker._saved_ctx_len = worker._ctx_len.clone()
    worker._saved_ctx_len_host = list(worker._ctx_len_host)
    worker._saved_ctx_base = worker._ctx_base_pages.clone()
    worker._saved_ctx_base_host = list(worker._ctx_base_host)
    worker._saved_req_ctx_pos = {}

    worker._write_ctx_len({1: 0, 2: 8200}, {2: 129})
    worker._restore_ctx_len_device()
    worker._restore_ctx_len_host()

    assert worker._ctx_len.tolist() == [0, 512, 64]
    assert worker._ctx_base_pages.tolist() == [0, 8, 0]
    assert worker._ctx_len_host == [0, 512, 64]
    assert worker._ctx_base_host == [0, 8, 0]


@pytest.mark.parametrize(
    ("windows", "expected"),
    [
        (None, True),
        ([None, None], True),
        ([None, 4096], True),
        ([128, 4096], True),
        ([None, 64], False),
    ],
)
def test_dflash_windows_span_a_page(windows: list | None, expected: bool) -> None:
    """A full-attention layer (window None) keeps every page; only a window
    shorter than a page rules out the page-aligned generation seed."""
    from tensorrt_llm._torch.speculative.dflash import dflash_windows_span_a_page

    assert dflash_windows_span_a_page(windows, 128) is expected
