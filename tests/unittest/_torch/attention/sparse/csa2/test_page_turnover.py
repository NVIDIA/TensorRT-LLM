# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""KV cache v2 page turnover on the DeepSeek-V4.1 CSA2 layout.

Admits and retires generation requests in shuffled order, with and without prefix reuse, and
checks that every live request keeps exactly the pages it was given and that the pool and host
memory return to their baseline.
"""

import ctypes
import random
from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.attention.backends.sparse.csa2.cache_manager import CSA2CacheManager
from tensorrt_llm._torch.attention.backends.sparse.csa2.params import CSA2Layout
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest, LlmRequestState
from tensorrt_llm.bindings import DataType, SamplingConfig
from tensorrt_llm.bindings.internal.batch_manager import CacheType
from tensorrt_llm.llmapi.llm_args import BlockReuseConfig, KvCacheConfig
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.runtime.kv_cache_manager_v2 import BAD_PAGE_INDEX, GPU_LEVEL

skip_pre_blackwell = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() < (10, 0),
    reason="This test requires Blackwell or newer",
)

TOKENS_PER_BLOCK = 128
PREFIX_BLOCKS = 8


class _MallInfo2(ctypes.Structure):
    _fields_ = [
        (name, ctypes.c_size_t)
        for name in (
            "arena",
            "ordblks",
            "smblks",
            "hblks",
            "hblkhd",
            "usmblks",
            "fsmblks",
            "uordblks",
            "fordblks",
            "keepcost",
        )
    ]


def _heap_in_use() -> int:
    mallinfo2 = ctypes.CDLL(None).mallinfo2
    mallinfo2.restype = _MallInfo2
    info = mallinfo2()
    return info.uordblks + info.hblkhd


def _manager(block_reuse: bool) -> CSA2CacheManager:
    # The DeepSeek-V4.1 attention layout.
    layout = CSA2Layout(
        (0, 0) + (2,) * 18 + (1,) * 20,
        (2, 8, 14, 20),
        (2, 8, 14, 20, 24, 28, 32, 36),
        candidate_source_layer_id=20,
    )
    return CSA2CacheManager(
        KvCacheConfig(
            max_gpu_total_bytes=6 << 30,
            host_cache_size=0,
            enable_block_reuse=block_reuse,
            block_reuse_config=BlockReuseConfig(policy="all_reusable"),
            dtype="fp8",
        ),
        CacheType.SELFKONLY,
        num_layers=40,
        tokens_per_block=TOKENS_PER_BLOCK,
        max_seq_len=64 * 1024,
        max_batch_size=8,
        max_num_tokens=1024,
        mapping=Mapping(),
        dtype=DataType.BF16,
        vocab_size=129280,
        layout=layout,
        is_disagg=True,
    )


def _gen_init(manager: CSA2CacheManager, request_id: int, tokens: list[int]) -> LlmRequest:
    request = LlmRequest(
        request_id=request_id,
        max_new_tokens=64,
        input_tokens=tokens,
        sampling_config=SamplingConfig(),
        is_streaming=False,
    )
    request.state = LlmRequestState.DISAGG_GENERATION_INIT
    assert manager.prepare_disagg_gen_init(request)
    return request


def _publish_prompt(manager: CSA2CacheManager, request: LlmRequest) -> None:
    """Commit the received prompt for reuse, as the GEN worker does after its transfer."""
    request.context_current_position = request.prompt_len
    manager.try_commit_blocks(request)
    assert manager.kv_cache_map[request.py_request_id].num_committed_tokens > 0


def _tables(manager: CSA2CacheManager, request_id: int) -> list[list[int]]:
    cache = manager.kv_cache_map[request_id]
    tables = []
    for group in range(len(manager.impl.layer_grouping)):
        table = list(cache.get_base_page_indices(group)[: cache.num_blocks])
        # The published table names the pages the request actually holds.
        pages = list(cache.get_aggregated_page_indices(group))
        assert all(index in (BAD_PAGE_INDEX, page) for index, page in zip(table, pages))
        tables.append(table)
    return tables


def _check_live(manager: CSA2CacheManager, live: dict[int, list[list[int]]], shared_blocks: int):
    """Live requests read the pages they were given; only a reused prefix is shared."""
    owners = {}
    for request_id, tables in live.items():
        assert _tables(manager, request_id) == tables
        for group, pages in enumerate(tables):
            for ordinal, page in enumerate(pages):
                if page == BAD_PAGE_INDEX:
                    continue
                owner = owners.setdefault((group, page), (request_id, ordinal))
                assert owner == (request_id, ordinal) or (
                    ordinal == owner[1] and ordinal < shared_blocks
                )


def _free(manager: CSA2CacheManager) -> list[int]:
    return [stats.free for stats in manager.impl.get_storage_statistics(GPU_LEVEL)]


@skip_pre_blackwell
@pytest.mark.parametrize("block_reuse", [False, True], ids=["gen", "prefix_reuse"])
def test_shuffled_turnover_keeps_pages_and_returns_to_baseline(block_reuse: bool) -> None:
    rng = random.Random(7)
    manager = _manager(block_reuse)
    shared_prefix = list(range(1000, 1000 + PREFIX_BLOCKS * TOKENS_PER_BLOCK))
    shared_blocks = PREFIX_BLOCKS if block_reuse else 0
    live: dict[int, list[list[int]]] = {}
    requests: dict[int, LlmRequest] = {}
    heap_after_round = []
    try:
        torch.cuda.synchronize()
        free_before = _free(manager)
        next_id = 0
        for round_index in range(4):
            for _ in range(8):
                length = rng.randint(64, 400) * TOKENS_PER_BLOCK + rng.randint(0, TOKENS_PER_BLOCK)
                tail = [rng.randint(2, 120000) for _ in range(length)]
                tokens = shared_prefix + tail if block_reuse else tail
                requests[next_id] = _gen_init(manager, next_id, tokens)
                live[next_id] = _tables(manager, next_id)
                next_id += 1
            if block_reuse:
                # A later request reuses the prefix a live one published; the shuffled release
                # below frees the publisher before or after it.
                _publish_prompt(manager, requests[min(live)])
                tail = [rng.randint(2, 120000) for _ in range(300)]
                requests[next_id] = _gen_init(manager, next_id, shared_prefix + tail)
                assert manager.kv_cache_map[next_id].num_committed_tokens > 0
                live[next_id] = _tables(manager, next_id)
                next_id += 1
            _check_live(manager, live, shared_blocks)

            order = list(live)
            rng.shuffle(order)
            for position, request_id in enumerate(order):
                manager.free_resources(requests.pop(request_id))
                del live[request_id]
                if position == len(order) // 2:
                    _check_live(manager, live, shared_blocks)
            torch.cuda.synchronize()
            if block_reuse:
                manager.reset_reuse_state()
            assert _free(manager) == free_before, f"round {round_index}"
            heap_after_round.append(_heap_in_use())
        # Retired requests' page objects are freed with them.
        assert heap_after_round[-1] - heap_after_round[0] < 1 << 20, heap_after_round
    finally:
        for request in requests.values():
            manager.free_resources(request)
        manager.shutdown()


@skip_pre_blackwell
def test_page_held_by_another_request_outlives_its_owner() -> None:
    manager = _manager(block_reuse=True)
    prefix = list(range(5000, 5000 + 4 * TOKENS_PER_BLOCK))
    try:
        torch.cuda.synchronize()
        free_before = _free(manager)
        owner = _gen_init(manager, 1, prefix + [7, 8, 9])
        _publish_prompt(manager, owner)
        reader = _gen_init(manager, 2, prefix + [11, 12])
        assert manager.kv_cache_map[2].num_committed_tokens == len(prefix)
        reader_tables = _tables(manager, 2)
        manager.free_resources(owner)
        # The reader still holds the prefix pages its owner allocated.
        assert _tables(manager, 2) == reader_tables
        manager.free_resources(reader)
        torch.cuda.synchronize()
        manager.reset_reuse_state()
        assert _free(manager) == free_before
    finally:
        for request_id in list(manager.kv_cache_map):
            manager.free_resources(SimpleNamespace(py_request_id=request_id))
        manager.shutdown()
