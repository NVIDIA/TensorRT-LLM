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
    ("bound_to_pool", "cur", "first_pos", "expected"),
    [
        # Managed pool: the block table is indexed by target position, so a
        # tail that reuse let the target start at first_pos is stored there.
        (True, 0, 8192, 8192),
        # A continuing chunk keeps the slot's stored length.
        (True, 4096, 4096, 4096),
        # Private arena: every request numbers from 0.
        (False, 0, 8192, 0),
        (False, 0, 0, 0),
    ],
)
def test_dflash_ctx_store_start_follows_the_pool_numbering(
    bound_to_pool: bool, cur: int, first_pos: int, expected: int
) -> None:
    from tensorrt_llm._torch.speculative.dflash import DFlashWorker

    worker = object.__new__(DFlashWorker)
    worker._ctx_block_tables = object() if bound_to_pool else None
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
