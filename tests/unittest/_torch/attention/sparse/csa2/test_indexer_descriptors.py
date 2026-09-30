# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Decode descriptors preserve live pages, masking and graph-owned storage."""

from __future__ import annotations

from contextlib import nullcontext
from copy import copy
from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest
import torch

from tensorrt_llm._torch.attention.backends.sparse.csa2.metadata import CSA2TrtllmMetadata
from tensorrt_llm._torch.attention.backends.sparse.csa2.quantization import INDEX_PAGE_ROWS
from tensorrt_llm._torch.attention.backends.trtllm import TrtllmAttentionMetadata

from . import test_metadata
from .test_metadata import _prepare_decode_metadata

if TYPE_CHECKING:
    from tensorrt_llm._torch.attention.backends.sparse.csa2.cache_manager import CSA2CacheManager
    from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest

manager_requests = test_metadata.manager_requests


def _inputs(
    query_count: int,
    request_count: int,
    source_pages: int,
    page_capacity: int,
    step: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Move page holes and request identities without changing captured tensor shapes."""
    table = [
        [
            -1 if (request + page + step) % 4 == 1 else 3 * request + page + 7 * step
            for page in range(source_pages)
        ]
        for request in range(request_count)
    ]
    table[0][0] = 0  # Physical page zero is allocated, not a hole.
    if step % 2 and request_count > 1:
        table[-1] = [-1] * source_pages
    requests = [
        (query + step) % request_count if query < request_count else request_count + query
        for query in range(query_count)
    ]
    if step % 2:
        requests[-1] = -1
        if query_count > 2:
            requests[1] = request_count
    boundaries = (
        0,
        1,
        INDEX_PAGE_ROWS - 1,
        INDEX_PAGE_ROWS,
        INDEX_PAGE_ROWS + 1,
        page_capacity * INDEX_PAGE_ROWS - 1,
        page_capacity * INDEX_PAGE_ROWS,
        page_capacity * INDEX_PAGE_ROWS + 1,
    )
    visible = [boundaries[(query + step) % len(boundaries)] for query in range(query_count)]
    return (
        torch.tensor(table, dtype=torch.int32),
        torch.tensor(requests, dtype=torch.int64),
        torch.tensor(visible, dtype=torch.int64),
    )


def _reference(
    table: torch.Tensor,
    token_requests: torch.Tensor,
    decode_visible: torch.Tensor,
    context_requests: int,
    pages_per_source_page: int,
    page_capacity: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Scalar logical-page oracle, independent of tensor expansion and gather operations."""
    table_rows = table.tolist()
    blocks, contexts, visible_rows, valid_rows = [], [], [], []
    for token_request, length in zip(token_requests.tolist(), decode_visible.tolist(), strict=True):
        request = token_request - context_requests
        request_valid = 0 <= request < len(table_rows)
        visible = length if request_valid else 0
        row_blocks, row_valid = [], []
        for native_page in range(page_capacity):
            source_page, offset = divmod(native_page, pages_per_source_page)
            allocated = (
                request_valid
                and source_page < len(table_rows[request])
                and table_rows[request][source_page] >= 0
            )
            row_blocks.append(
                table_rows[request][source_page] * pages_per_source_page + offset
                if allocated
                else 0
            )
            row_valid.extend(
                allocated and native_page * INDEX_PAGE_ROWS + row < visible
                for row in range(INDEX_PAGE_ROWS)
            )
        blocks.append(row_blocks)
        contexts.append([max(visible, 1)])
        visible_rows.append(visible)
        valid_rows.append(row_valid)
    return (
        torch.tensor(blocks, dtype=torch.int32),
        torch.tensor(contexts, dtype=torch.int32),
        torch.tensor(visible_rows, dtype=torch.int32),
        torch.tensor(valid_rows, dtype=torch.bool),
    )


def _output_storage(query_count: int, page_capacity: int) -> tuple[torch.Tensor, ...]:
    # Spare rows catch accidental writes outside the active descriptor views.
    return (
        torch.full((query_count + 2, page_capacity), -123, dtype=torch.int32, device="cuda"),
        torch.full((query_count + 2, 1), -123, dtype=torch.int32, device="cuda"),
        torch.full((query_count + 2,), -123, dtype=torch.int32, device="cuda"),
        torch.zeros(
            (query_count + 2, page_capacity * INDEX_PAGE_ROWS), dtype=torch.bool, device="cuda"
        ),
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(
    "query_count,request_count,source_pages,page_capacity,pages_per_source_page",
    [
        pytest.param(1, 1, 1, 1, 1, id="single"),
        pytest.param(8, 5, 3, 2, 1, id="truncate-source-pages"),
        pytest.param(16, 9, 3, 8, 2, id="pad-native-capacity"),
        pytest.param(32, 12, 3, 9, 4, id="expanded-truncated-pages"),
    ],
)
@torch.inference_mode()
def test_indexer_descriptors_follow_live_graph_inputs(
    query_count: int,
    request_count: int,
    source_pages: int,
    page_capacity: int,
    pages_per_source_page: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Replay changed contents against eager and scalar references with exact integer parity."""
    monkeypatch.setattr("tensorrt_llm._torch.utils.is_piecewise_running", lambda: False)
    eager = CSA2TrtllmMetadata._fill_indexer_descriptors
    captured = CSA2TrtllmMetadata._fill_indexer_descriptors
    host_inputs = _inputs(query_count, request_count, source_pages, page_capacity, 0)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        inputs = tuple(tensor.cuda() for tensor in host_inputs)
        storage = _output_storage(query_count, page_capacity)
        outputs = tuple(tensor[:query_count] for tensor in storage)
        eager_outputs = tuple(torch.empty_like(tensor) for tensor in outputs)
        for _ in range(3):
            captured(*inputs, 0, pages_per_source_page, INDEX_PAGE_ROWS, *outputs)
    stream.synchronize()
    pointers = tuple(tensor.data_ptr() for tensor in (*inputs, *outputs))
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        captured(*inputs, 0, pages_per_source_page, INDEX_PAGE_ROWS, *outputs)

    previous = None
    changed = False
    for step in range(1, 6):
        host_inputs = _inputs(query_count, request_count, source_pages, page_capacity, step)
        expected = _reference(*host_inputs, 0, pages_per_source_page, page_capacity)
        with torch.cuda.stream(stream):
            for tensor, updated in zip(inputs, host_inputs, strict=True):
                tensor.copy_(updated)
            eager(*inputs, 0, pages_per_source_page, INDEX_PAGE_ROWS, *eager_outputs)
            # Every active element must be rewritten on each replay, including
            # invalid requests and masked holes; previous outputs cannot pass.
            for tensor in outputs:
                tensor.fill_(True if tensor.dtype == torch.bool else -777)
            graph.replay()
        stream.synchronize()
        actual = tuple(tensor.cpu() for tensor in outputs)
        for reference, eager_output, output in zip(expected, eager_outputs, actual, strict=True):
            torch.testing.assert_close(eager_output.cpu(), reference, rtol=0, atol=0)
            torch.testing.assert_close(output, reference, rtol=0, atol=0)
        assert pointers == tuple(tensor.data_ptr() for tensor in (*inputs, *outputs))
        for tensor in storage[:-1]:
            assert torch.all(tensor[query_count:] == -123).item()
        assert not storage[-1][query_count:].any().item()
        if previous is not None:
            changed |= any(
                not torch.equal(left, right) for left, right in zip(actual, previous, strict=True)
            )
        previous = actual
    assert changed, "The graph must observe changing descriptor inputs"


@pytest.mark.cpu_only
def test_graph_clone_post_init_clears_decode_deferral_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metadata = object.__new__(CSA2TrtllmMetadata)
    metadata._csa2_deferred_decode_outputs = True
    clone = copy(metadata)
    monkeypatch.setattr(TrtllmAttentionMetadata, "__post_init__", lambda self: None)
    clone.__post_init__()
    assert not clone._csa2_deferred_decode_outputs
    assert metadata._csa2_deferred_decode_outputs


@pytest.mark.cpu_only
@pytest.mark.parametrize("entrypoint", ["prepare", "prepare_csa2"])
def test_failed_prepare_clears_decode_deferral_state(
    monkeypatch: pytest.MonkeyPatch, entrypoint: str
) -> None:
    metadata = object.__new__(CSA2TrtllmMetadata)
    metadata._csa2_deferred_decode_outputs = True
    metadata.kv_cache_manager = SimpleNamespace()
    metadata.is_cuda_graph = False
    metadata.kv_lens_cuda = torch.empty(0)

    def fail_prepare() -> None:
        raise ValueError("injected prepare failure")

    monkeypatch.setattr(torch.cuda, "device", lambda _: nullcontext())
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", fail_prepare)
    error = ValueError if entrypoint == "prepare" else TypeError
    message = "injected prepare failure" if entrypoint == "prepare" else "CSA2CacheManager"
    with pytest.raises(error, match=message):
        getattr(metadata, entrypoint)()
    assert not metadata._csa2_deferred_decode_outputs


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@torch.inference_mode()
def test_selected_decode_prepare_controls_indexer_dispatch(
    manager_requests: tuple[CSA2CacheManager, list[LlmRequest]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager, requests = manager_requests
    base = CSA2TrtllmMetadata(max_num_requests=2, max_num_tokens=1024, kv_cache_manager=manager)
    # A shallow graph clone must discard any eligibility inherited from its source.
    base._csa2_deferred_decode_outputs = True
    metadata = base.create_cuda_graph_metadata(2)
    assert not metadata._csa2_deferred_decode_outputs
    assert base._csa2_deferred_decode_outputs
    starts = [127, 128]
    calls = []
    eager = CSA2TrtllmMetadata._fill_indexer_descriptors

    def record_refresh(*args: torch.Tensor | int) -> None:
        calls.append(metadata._csa2_deferred_decode_outputs)
        return eager(*args)

    # The same fused dispatcher serves every preparation mode. Graph parity
    # is checked above; isolate per-owner reuse and preparation state here.
    monkeypatch.setattr(metadata, "_fill_indexer_descriptors", record_refresh)
    monkeypatch.setattr("tensorrt_llm._torch.utils.is_piecewise_running", lambda: True)
    monkeypatch.setattr(
        "tensorrt_llm.deep_gemm.get_paged_mqa_logits_metadata",
        lambda lengths, page_size, sm_count: torch.zeros(
            (1,), dtype=torch.int32, device=lengths.device
        ),
    )

    _prepare_decode_metadata(metadata, requests, starts, deferred=True)
    assert metadata._csa2_deferred_decode_outputs
    assert not metadata._csa2_defer_decode_outputs
    metadata.on_update_kv_lens()
    assert metadata.prepare_indexer(1) is metadata
    assert calls == [True]
    # Layers 1 and 2 share an owner. One serial refresh serves both layers.
    metadata.prepare_indexer(1)
    metadata.prepare_indexer(2)
    assert calls == [True]

    _prepare_decode_metadata(metadata, requests, starts)
    assert not metadata._csa2_deferred_decode_outputs
    metadata.on_update_kv_lens()
    metadata.prepare_indexer(1)
    assert calls == [True, False]

    _prepare_decode_metadata(metadata, requests, starts, deferred=True)
    assert metadata._csa2_deferred_decode_outputs
    metadata.set_source_batch([1, 1], starts)
    _prepare_decode_metadata(metadata, requests, starts, deferred=True)
    assert not hasattr(metadata, "_csa2_source_batch")
    assert not metadata._csa2_deferred_decode_outputs
    metadata.on_update_kv_lens()
    metadata.prepare_indexer(1)
    assert calls == [True, False, False]
