# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""History/stream integration and read coverage; KVCM storage remains a double."""

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from tensorrt_llm._torch.attention.backends.sparse.deepseek_v4.cache_manager import (
    DeepseekV4CacheManager,
)
from tensorrt_llm._torch.attention.backends.sparse.deepseek_v4.kernels import (
    check_sparse_read_table,
    deepseek_v4_local_to_global_indices,
)
from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import BlockReusePolicy
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequestState
from tensorrt_llm.runtime.kv_cache_manager_v2 import PageIndexMode

from .test_deepseek_v4_offload_attention import _attention_case, _consume
from .test_deepseek_v4_offload_metadata import _manager, _metadata

_requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")


class _HistoryCache:
    def __init__(self, tokens_per_block: int, cuda_stream: int) -> None:
        self.id, self.beam_width, self.num_blocks = 20, 1, 3
        self.capacity, self.history_length = 3 * tokens_per_block, 0
        self.cuda_stream, self.is_active = cuda_stream, True
        self.pages = [0, 7, 8]
        self.calls: list[tuple[int | None, int | None]] = []
        self.fail_resize = False

    def resize(self, capacity: int | None, history_length: int | None) -> bool:
        self.calls.append((capacity, history_length))
        if self.fail_resize:
            return False
        if capacity is not None:
            self.capacity = capacity
        if history_length is not None:
            self.history_length = history_length
        return True


def _history_case(
    tokens_per_block: int = 128, *, stream: torch.cuda.Stream | None = None
) -> tuple[DeepseekV4CacheManager, _HistoryCache, SimpleNamespace, SimpleNamespace]:
    manager = _manager(tokens_per_block)
    manager._stream = stream or Mock(cuda_stream=19, device=None)
    manager.block_reuse_policy = BlockReusePolicy.ALL_REUSABLE
    manager.conversation_manager = None
    manager._allocated_draft_lens = {}
    cache = _HistoryCache(tokens_per_block, manager._stream.cuda_stream)
    manager.kv_cache_map = {20: cache}
    request = SimpleNamespace(
        py_request_id=20,
        context_current_position=tokens_per_block - 1,
        context_remaining_length=0,
        is_dummy_request=False,
        py_num_accepted_draft_tokens=0,
        py_rewind_len=0,
        state=LlmRequestState.GENERATION_IN_PROGRESS,
        max_beam_num_tokens=tokens_per_block,
    )
    batch = SimpleNamespace(context_requests=[request], generation_requests=[])
    return manager, cache, request, batch


@pytest.mark.cpu_only
@pytest.mark.parametrize("tokens_per_block", [128, 256])
def test_completed_history_advances_without_reuse_or_capacity_growth(tokens_per_block: int) -> None:
    manager, cache, request, batch = _history_case(tokens_per_block)
    caller_stream = object()
    with (
        patch("torch.cuda.current_stream", return_value=caller_stream),
        patch("torch.cuda.is_current_stream_capturing", return_value=False),
    ):
        manager.update_context_resources(batch)
        assert cache.history_length == tokens_per_block - 1
        assert cache.calls == [(None, tokens_per_block - 1)]
        batch.context_requests, batch.generation_requests = [], [request]
        for completed in (
            tokens_per_block,
            tokens_per_block + 1,
            2 * tokens_per_block,
            2 * tokens_per_block + 1,
        ):
            # Sampling has appended one output token; that token has no KV yet.
            request.max_beam_num_tokens = completed + 1
            manager.update_resources(batch)
            assert cache.history_length == completed
            assert cache.calls[-1] == (3 * tokens_per_block, completed)
        assert manager._stream.wait_stream.call_count == 5
        manager._stream.wait_stream.assert_called_with(caller_stream)
        assert len(cache.calls) == 5


@pytest.mark.cpu_only
@pytest.mark.parametrize("context", [False, True])
@pytest.mark.parametrize(
    "problem", ["resize_failure", "wrong_stream", "capture", "suspended", "removed"]
)
def test_history_update_rejects_failures_and_skips_inactive(context: bool, problem: str) -> None:
    manager, cache, request, batch = _history_case()
    update = manager.update_context_resources if context else manager.update_resources
    if not context:
        batch.context_requests, batch.generation_requests = [], [request]
    if problem == "resize_failure":
        cache.fail_resize = True
    elif problem == "wrong_stream":
        cache.cuda_stream += 1
    elif problem == "suspended":
        cache.is_active = False
    elif problem == "removed":
        manager.kv_cache_map.clear()
    with (
        patch("torch.cuda.current_stream"),
        patch("torch.cuda.is_current_stream_capturing", return_value=problem == "capture"),
    ):
        if problem in ("suspended", "removed"):
            update(batch)
            manager._stream.wait_stream.assert_not_called()
        else:
            with pytest.raises((ValueError, RuntimeError)):
                update(batch)
    assert cache.history_length == 0
    assert len(cache.calls) == (1 if problem == "resize_failure" else 0)


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "policy", ["enable_block_reuse", "kv_compression_manages_history", "_has_cp_helix", "is_draft"]
)
def test_offload_rejects_other_history_policies(policy: str) -> None:
    manager, cache, _, batch = _history_case()
    setattr(manager, policy, True)
    with pytest.raises(NotImplementedError, match="Sparse offload requires"):
        manager.update_context_resources(batch)
    assert not cache.calls


@pytest.mark.cpu_only
def test_descriptor_uses_common_pool_origin_and_layer_relative_bound() -> None:
    manager = _manager()
    descriptors = manager.get_sparse_offload_descriptors()
    first, second = descriptors[2], descriptors[4]
    assert first.page_index_upper_bound == 303
    assert second.page_index_upper_bound == 297
    role = first.buffer_id.role
    assert manager.impl.get_mem_pool_base_address(
        41, role, PageIndexMode.SHARED
    ) != manager.impl.get_mem_pool_base_address(19, role, PageIndexMode.SHARED)
    manager.impl.get_page_index_upper_bound = Mock(return_value=0)
    with pytest.raises(ValueError, match="physical-page bound"):
        manager.get_sparse_offload_descriptors()


def _cuda(value: list[int] | list[list[int]]) -> torch.Tensor:
    return torch.tensor(value, dtype=torch.int32, device="cuda")


@_requires_cuda
@pytest.mark.parametrize(
    "problem", ["none", "missing", "bound", "length", "ordinal", "request", "active", "padding"]
)
def test_read_coverage_detects_required_pages_without_host_reads(problem: str) -> None:
    topk = _cuda([[0, 31, -1, -1], [0, 31, 32, 63], [0] * 4, [0] * 4])
    requests, active, lengths = _cuda([1, 0, 2, 999]), _cuda([2]), _cuda([64] * 4)
    read = _cuda([[6, 8, 10], [2, -1, 4], [-1] * 3, [-1] * 3])
    if problem == "missing":
        read[0, 1] = -1
    elif problem == "bound":
        read[0, 1] = 16
    elif problem == "length":
        topk[1, 0] = 64
    elif problem == "ordinal":
        topk[1, 0], lengths[0] = 128, 160
    elif problem == "request":
        requests[0] = 999
    elif problem == "active":
        active.fill_(5)
    elif problem == "padding":
        active.zero_()  # Stale, invalid queries cannot invalidate an empty replay.
    valid = _cuda([77])
    check_sparse_read_table(topk, requests, active, lengths, read, 32, 16, valid)
    assert valid.item() == int(problem in ("none", "padding"))
    # Status is rewritten, rather than sticky from the previous failed step.
    active.zero_()
    check_sparse_read_table(topk, requests, active, lengths, read, 32, 16, valid)
    torch._assert_async(valid)
    assert valid.item() == 1


@_requires_cuda
def test_decode_padding_masks_both_pools_before_table_access() -> None:
    active = _cuda([1])
    args = dict(
        req_id=_cuda([0, 0, 999, -1]),
        block_table_swa=_cuda([[2], [3], [4], [5]]),
        swa_local_indices=_cuda([[0, 1]] * 4),
        swa_pool_base_ptr=0,
        swa_buffer_ptr=0,
        tokens_per_block=128,
        token_stride=32,
        block_table_compressed=_cuda([[6], [7], [8], [9]]),
        compressed_local_indices=_cuda([[0, 1]] * 4),
        compress_ratio=4,
        num_compressed_indices=2,
        active_request_count=active,
    )
    indices = deepseek_v4_local_to_global_indices(**args)
    torch.testing.assert_close(indices[0], _cuda([256, 257, 192, 193]))
    assert (indices[1:] == -1).all()


@_requires_cuda
def test_missing_fetch_result_is_checked_before_index_conversion() -> None:
    case = _attention_case()
    case.manager.impl.fetch_sparse_pages = lambda **kwargs: kwargs["out"].fill_(-1)

    # Avoid poisoning the suite's CUDA context with a deliberate device assert.
    # Exercise the production gate, but inspect its device status in this test.
    def check(valid: torch.Tensor, message: str) -> None:
        if not valid.item():
            raise RuntimeError(message)

    with (
        patch("torch._assert_async", side_effect=check),
        patch(
            "tensorrt_llm._torch.attention.backends.sparse.deepseek_v4.backend.deepseek_v4_local_to_global_indices"
        ) as convert,
    ):
        with pytest.raises(RuntimeError, match="missing/out-of-bounds KV page"):
            _consume(case, case.backends[0])
        convert.assert_not_called()


@_requires_cuda
@pytest.mark.parametrize("tokens_per_block", [128, 256])
def test_history_demotion_waits_for_all_layer_writers_and_readers(tokens_per_block: int) -> None:
    cache_stream, model_stream = torch.cuda.Stream(), torch.cuda.Stream()
    manager, cache, request, batch = _history_case(tokens_per_block, stream=cache_stream)
    state = _metadata(manager).sparse_offload_state
    write = torch.empty((4, 6), dtype=torch.int32, device="cuda")
    p = tokens_per_block // 4
    # Two CSA layers write different data to the same logical history page.
    source = torch.zeros((2, p, 16), dtype=torch.bfloat16, device="cuda")
    host = torch.empty(source.shape, dtype=source.dtype, pin_memory=True)
    read_before_recycle = torch.empty_like(source)
    fetched = torch.empty_like(source)
    model_stream.wait_stream(torch.cuda.current_stream())
    resize = cache.resize

    def demote(capacity: int | None, history: int | None) -> bool:
        previous = cache.history_length
        result = resize(capacity, history)
        if previous < tokens_per_block <= cache.history_length:
            with torch.cuda.stream(cache_stream):
                host.copy_(source, non_blocking=True)
                source.fill_(-99)  # Simulate reusing the released resident slots.
        return result

    cache.resize = demote
    with torch.cuda.stream(model_stream):
        source[0, :-1].fill_(1)
        source[1, :-1].fill_(2)
        manager.update_context_resources(batch)
        manager.prepare_sparse_offload(state, [20], write, beam_width=1)
    model_stream.synchronize()
    assert cache.history_length == tokens_per_block - 1
    assert state.history_blocks[0].item() == 0
    assert write[0, 0].item() == 0

    events = []
    producers = [torch.cuda.Stream(), torch.cuda.Stream()]
    for layer, producer in enumerate(producers):
        producer.wait_stream(model_stream)
        with torch.cuda.stream(producer):
            torch.cuda._sleep(2_000_000 * (layer + 1))
            source[layer, -1].fill_(layer + 3)
            events.append(producer.record_event())
    batch.context_requests, batch.generation_requests = [], [request]
    request.max_beam_num_tokens = tokens_per_block + 1
    with torch.cuda.stream(model_stream):
        for event in events:
            model_stream.wait_event(event)
        read_before_recycle.copy_(source)
        manager.update_resources(batch)
        manager.prepare_sparse_offload(state, [20], write, beam_width=1)
        fetched.copy_(host, non_blocking=True)
    model_stream.synchronize()
    assert cache.history_length == tokens_per_block
    assert state.history_blocks[0].item() == 1
    assert write[0, 0].item() == -1
    assert (source == -99).all()
    expected = torch.empty_like(source)
    expected[0, :-1], expected[1, :-1] = 1, 2
    expected[0, -1], expected[1, -1] = 3, 4
    torch.testing.assert_close(read_before_recycle, expected, rtol=0, atol=0)
    torch.testing.assert_close(fetched, expected, rtol=0, atol=0)
