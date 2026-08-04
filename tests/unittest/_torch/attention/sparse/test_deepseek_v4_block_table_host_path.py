# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Host-path tests for the DeepSeek-V4 block-table preparation.

These cover two behaviours that are easy to break and expensive to notice:

1. ``_get_copy_index_cached`` must return the same mapping the uncached
   ``IndexMapper.get_copy_index`` call would have produced, must recompute when
   the request set changes (including in-place mutation of a reused list), and
   must be invalidated at the start of every step.
2. The block-table destination buffers are only partially overwritten by the
   copy that follows, so the untouched padding must still carry
   ``BAD_PAGE_INDEX``. Padded CUDA-graph token slots can index those rows
   through ``req_idx_per_token``, so leaving stale values there is a silent
   correctness bug rather than a crash.
"""

import torch

from tensorrt_llm._torch.attention_backend.sparse.deepseek_v4.cache_manager import (
    DeepseekV4CacheManager, )
from tensorrt_llm.runtime.kv_cache_manager_v2._common import BAD_PAGE_INDEX


class _RecordingIndexMapper:
    """Stands in for the C++ IndexMapper, counting how often it is walked."""

    def __init__(self):
        self.calls = 0

    def get_copy_index(self, request_ids, num_contexts, beam_width):
        self.calls += 1
        # Mirror the real mapper's contract closely enough for these tests: one
        # row per request, deterministic in the arguments.
        return torch.tensor(
            [(r * 10 + num_contexts + beam_width) for r in request_ids],
            dtype=torch.int32,
        )


def _manager_with_mapper():
    mgr = DeepseekV4CacheManager.__new__(DeepseekV4CacheManager)
    mgr.index_mapper = _RecordingIndexMapper()
    return mgr


def test_copy_index_memo_reuses_one_walk_per_step():
    mgr = _manager_with_mapper()
    request_ids = [3, 1, 4, 1, 5]

    mgr._copy_idx_memo_key = None
    first = mgr._get_copy_index_cached(request_ids, 2, 1)
    again = mgr._get_copy_index_cached(request_ids, 2, 1)
    third = mgr._get_copy_index_cached(request_ids, 2, 1)

    assert mgr.index_mapper.calls == 1, "memo should collapse three walks into one"
    torch.testing.assert_close(first, again, rtol=0, atol=0)
    torch.testing.assert_close(first, third, rtol=0, atol=0)


def test_copy_index_memo_matches_uncached_result():
    """The memo must not change the value, only how often it is computed."""
    cached_mgr = _manager_with_mapper()
    plain_mgr = _manager_with_mapper()
    request_ids = [7, 2, 9]

    cached_mgr._copy_idx_memo_key = None
    got = cached_mgr._get_copy_index_cached(request_ids, 1, 1)
    expected = plain_mgr.index_mapper.get_copy_index(request_ids, 1, 1)

    torch.testing.assert_close(got, expected, rtol=0, atol=0)


def test_copy_index_memo_recomputes_on_changed_arguments():
    mgr = _manager_with_mapper()
    mgr._copy_idx_memo_key = None

    mgr._get_copy_index_cached([1, 2], 0, 1)
    assert mgr.index_mapper.calls == 1
    # different request set
    mgr._get_copy_index_cached([1, 3], 0, 1)
    assert mgr.index_mapper.calls == 2
    # different num_contexts
    mgr._get_copy_index_cached([1, 3], 1, 1)
    assert mgr.index_mapper.calls == 3
    # different beam width
    mgr._get_copy_index_cached([1, 3], 1, 2)
    assert mgr.index_mapper.calls == 4


def test_copy_index_memo_keys_on_contents_not_identity():
    """The caller may reuse one list object and mutate it between steps."""
    mgr = _manager_with_mapper()
    reused = [1, 2, 3]

    mgr._copy_idx_memo_key = None
    before = mgr._get_copy_index_cached(reused, 0, 1).clone()
    assert mgr.index_mapper.calls == 1

    reused[1] = 99  # same object, different contents
    after = mgr._get_copy_index_cached(reused, 0, 1)

    assert mgr.index_mapper.calls == 2, "must not serve a stale mapping"
    assert not torch.equal(before, after)


def test_copy_index_memo_is_reset_each_step():
    """Resetting the key is what makes the memo safe across steps."""
    mgr = _manager_with_mapper()
    request_ids = [4, 5]

    mgr._copy_idx_memo_key = None
    mgr._get_copy_index_cached(request_ids, 0, 1)
    assert mgr.index_mapper.calls == 1

    # start of the next step
    mgr._copy_idx_memo_key = None
    mgr._get_copy_index_cached(request_ids, 0, 1)
    assert mgr.index_mapper.calls == 2


def test_sliding_block_table_padding_keeps_bad_page_index():
    """Only [:_num_tables] is overwritten; the tail must stay BAD_PAGE_INDEX."""
    layers, types, capacity, max_blocks, num_tables = 2, 5, 9, 4, 3
    src = torch.arange(layers * types * num_tables * max_blocks,
                       dtype=torch.int32).reshape(layers, types, num_tables,
                                                 max_blocks)

    # Reference: the original code filled the whole tensor first.
    reference = torch.empty((layers, types, capacity, max_blocks),
                            dtype=torch.int32)
    reference.fill_(BAD_PAGE_INDEX)
    reference[:, :, :num_tables, :].copy_(src)

    # Optimized: pre-poison everything, fill only the tail, then copy the head.
    optimized = torch.full((layers, types, capacity, max_blocks),
                           -999999,
                           dtype=torch.int32)
    if num_tables < optimized.size(2):
        optimized[:, :, num_tables:, :].fill_(BAD_PAGE_INDEX)
    optimized[:, :, :num_tables, :].copy_(src)

    torch.testing.assert_close(optimized, reference, rtol=0, atol=0)
    assert not (optimized == -999999).any(), "head was not fully overwritten"


def test_sliding_block_table_padding_when_full():
    """With num_tables == capacity there is no tail to fill."""
    layers, types, capacity, max_blocks = 1, 2, 6, 3
    src = torch.arange(layers * types * capacity * max_blocks,
                       dtype=torch.int32).reshape(layers, types, capacity,
                                                  max_blocks)

    reference = torch.empty((layers, types, capacity, max_blocks),
                            dtype=torch.int32)
    reference.fill_(BAD_PAGE_INDEX)
    reference[:, :, :capacity, :].copy_(src)

    optimized = torch.full((layers, types, capacity, max_blocks),
                           -999999,
                           dtype=torch.int32)
    if capacity < optimized.size(2):  # false: guard must skip the fill
        optimized[:, :, capacity:, :].fill_(BAD_PAGE_INDEX)
    optimized[:, :, :capacity, :].copy_(src)

    torch.testing.assert_close(optimized, reference, rtol=0, atol=0)


def test_block_offsets_padding_keeps_bad_page_index():
    """copy_batch_block_offsets writes only beam 0 of [:_num_tables]."""
    layers, capacity, beams, max_blocks, num_tables = 2, 7, 2, 3, 4
    src = torch.arange(layers * num_tables * max_blocks,
                       dtype=torch.int32).reshape(layers, num_tables,
                                                  max_blocks)

    reference = torch.empty((layers, capacity, beams, max_blocks),
                            dtype=torch.int32)
    reference.fill_(BAD_PAGE_INDEX)
    reference[:, :num_tables, 0, :].copy_(src)

    optimized = torch.full((layers, capacity, beams, max_blocks),
                           -999999,
                           dtype=torch.int32)
    if num_tables < optimized.size(1):
        optimized[:, num_tables:, :, :].fill_(BAD_PAGE_INDEX)
    optimized[:, :num_tables, 1:, :].fill_(BAD_PAGE_INDEX)
    optimized[:, :num_tables, 0, :].copy_(src)

    torch.testing.assert_close(optimized, reference, rtol=0, atol=0)
    assert not (optimized == -999999).any(), "some region was left unwritten"
