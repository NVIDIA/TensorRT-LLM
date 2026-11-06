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

"""Tests for DeepSeek-V4 sparse offload kernels."""

import pytest
import torch

from tensorrt_llm._torch.attention.backends.sparse.deepseek_v4.kernels import (
    check_sparse_read_table,
    deepseek_v4_local_to_global_indices,
    merge_sparse_read_table,
    prepare_sparse_write_table,
    select_sparse_history_pages,
)

_requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")


# Sparse history selection and read-table kernels.


def _cuda_tensor(value: torch.Tensor | list, strided: bool = False) -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=torch.int32, device="cuda")
    if not strided:
        return tensor
    storage = torch.full(
        tuple(2 * n for n in tensor.shape), 12345, dtype=torch.int32, device="cuda"
    )
    view = storage[tuple(slice(None, None, 2) for _ in tensor.shape)]
    view.copy_(tensor)
    return view


def _selection_reference(
    topk: torch.Tensor,
    request_rows: torch.Tensor,
    active_count: int,
    lengths: torch.Tensor,
    raw: torch.Tensor,
    history: torch.Tensor,
    tokens_per_page: int,
    width: int,
) -> torch.Tensor:
    result = torch.full((raw.shape[0], width), -1, dtype=torch.int32)
    for query, request in enumerate(request_rows.tolist()):
        if query >= active_count or not 0 <= request < min(active_count, raw.shape[0]):
            continue
        pages = set()
        for token in topk[query].tolist():
            if 0 <= token < lengths[request]:
                ordinal = token // tokens_per_page
                if ordinal < min(history[request], raw.shape[1]) and raw[request, ordinal] >= 0:
                    pages.add(ordinal)
        ordered = sorted(pages)
        result[request, : len(ordered)] = torch.tensor(ordered, dtype=torch.int32)
    return result


def _merge_reference(
    fetched: torch.Tensor,
    write: torch.Tensor,
    history: torch.Tensor,
    active_count: int,
    scale: int,
) -> torch.Tensor:
    result = torch.full_like(write, -1)
    for request in range(min(active_count, write.shape[0])):
        for ordinal in range(write.shape[1]):
            if ordinal < history[request]:
                page = int(fetched[request, ordinal]) * scale
            else:
                page = int(write[request, ordinal])
            if 0 <= page <= torch.iinfo(torch.int32).max:
                result[request, ordinal] = page
    return result


@_requires_cuda
@pytest.mark.parametrize("tokens_per_block", [128, 256])
@pytest.mark.parametrize("strided", [False, True])
def test_select_history_pages_boundaries(tokens_per_block: int, strided: bool) -> None:
    p = tokens_per_block // 4
    original_history = torch.tensor(
        [
            tokens_per_block - 1,
            tokens_per_block,
            tokens_per_block + 1,
            3 * tokens_per_block + 7,
            2 * tokens_per_block,
            4 * tokens_per_block,
            10000,
            10000,
        ],
        dtype=torch.int32,
    )
    history = original_history // tokens_per_block
    lengths = (original_history + 1) // 4
    raw = torch.arange(8 * 6, dtype=torch.int32).reshape(8, 6) + 17
    raw[1, 0] = 0  # Host slot zero must not be mistaken for an absent page.
    raw[2, 0] = -1
    raw[3, 1] = -1
    by_request = [
        [0, p - 1, p, -1],
        [p - 1, 0, 0, p, -1],
        [0, p - 1, p, -1],
        [2 * p + 5, 0, p + 1, p + 1, 3 * p, 3 * p + 2, -1],
        [p - 1, p, 2 * p - 1, 2 * p, 0, -2],
        [3 * p + 1, 2 * p, p, 0, 0, 3 * p - 1, 4 * p, -1],
    ]
    request_rows = torch.tensor([5, 2, 0, 4, 1, 3, 0, 999], dtype=torch.int32)
    topk = torch.full((8, 16), -1, dtype=torch.int32)
    for query, request in enumerate(request_rows[:6].tolist()):
        tokens = by_request[request]
        topk[query, : len(tokens)] = torch.tensor(tokens, dtype=torch.int32)
    topk[6:] = 0  # Stale graph padding, including a duplicate live request ID.
    topk_gpu = _cuda_tensor(topk, strided)
    raw_gpu = _cuda_tensor(raw, strided)
    out = _cuda_tensor(torch.full((8, 9), 999, dtype=torch.int32), strided)
    active_count = _cuda_tensor([6])
    args = (
        topk_gpu,
        _cuda_tensor(request_rows, strided),
        active_count,
        _cuda_tensor(lengths, strided),
        raw_gpu,
        _cuda_tensor(history, strided),
        p,
        out,
    )

    select_sparse_history_pages(*args)

    expected = _selection_reference(topk, request_rows, 6, lengths, raw, history, p, 9)
    torch.testing.assert_close(out.cpu(), expected, rtol=0, atol=0)
    assert out[1, 0].item() == 0
    assert (out[0] == -1).all()  # A newly completed page is still the resident tail.
    torch.testing.assert_close(topk_gpu.cpu(), topk, rtol=0, atol=0)
    torch.testing.assert_close(raw_gpu.cpu(), raw, rtol=0, atol=0)

    active_count.zero_()
    select_sparse_history_pages(*args)
    assert (out == -1).all()


@_requires_cuda
@pytest.mark.parametrize("tokens_per_page", [32, 64])
@pytest.mark.parametrize(("topk_size", "max_blocks"), [(1, 1), (7, 3), (513, 1100), (1024, 257)])
def test_select_history_pages_scattered(
    tokens_per_page: int, topk_size: int, max_blocks: int
) -> None:
    generator = torch.Generator().manual_seed(42)
    tokens = (torch.arange(topk_size) % max_blocks) * tokens_per_page
    tokens += torch.arange(topk_size) % tokens_per_page
    topk = torch.stack([tokens[torch.randperm(topk_size, generator=generator)] for _ in range(3)])
    topk = topk.to(torch.int32)
    raw = torch.arange(3 * max_blocks, dtype=torch.int32).reshape(3, max_blocks)
    history = torch.tensor([max_blocks, max_blocks // 2, 0], dtype=torch.int32)
    lengths = torch.full((3,), max_blocks * tokens_per_page, dtype=torch.int32)
    requests = torch.tensor([2, 0, 1], dtype=torch.int32)
    width = min(topk_size, max_blocks)
    out = torch.empty((3, width), dtype=torch.int32, device="cuda")
    args = (
        _cuda_tensor(topk),
        _cuda_tensor(requests),
        _cuda_tensor([3]),
        _cuda_tensor(lengths),
        _cuda_tensor(raw),
        _cuda_tensor(history),
        tokens_per_page,
    )

    select_sparse_history_pages(*args, out)

    expected = _selection_reference(
        topk, requests, 3, lengths, raw, history, tokens_per_page, width
    )
    torch.testing.assert_close(out.cpu(), expected, rtol=0, atol=0)
    with pytest.raises(ValueError, match="selection must not be truncated"):
        select_sparse_history_pages(*args, out[:, : width - 1])


@_requires_cuda
def test_select_history_pages_masks_invalid_request_ids() -> None:
    out = torch.full((4, 1), 999, dtype=torch.int32, device="cuda")
    select_sparse_history_pages(
        _cuda_tensor([[0], [0], [0], [0]]),
        _cuda_tensor([-1, 999, 1, 0]),
        _cuda_tensor([4]),
        _cuda_tensor([32] * 4),
        _cuda_tensor([[0], [1], [2], [3]]),
        _cuda_tensor([1] * 4),
        32,
        out,
    )
    torch.testing.assert_close(out.cpu(), torch.tensor([[0], [0], [-1], [-1]], dtype=torch.int32))


@_requires_cuda
@pytest.mark.parametrize(("batch", "max_blocks", "topk"), [(0, 3, 7), (2, 0, 7), (2, 3, 0)])
def test_sparse_offload_empty_shapes(batch: int, max_blocks: int, topk: int) -> None:
    raw = torch.empty((batch, max_blocks), dtype=torch.int32, device="cuda")
    out = torch.full((batch, max(1, min(topk, max_blocks))), 999, dtype=torch.int32, device="cuda")
    history = _cuda_tensor([0] * batch)
    active_count = _cuda_tensor([0])
    select_sparse_history_pages(
        torch.empty((batch, topk), dtype=torch.int32, device="cuda"),
        _cuda_tensor(list(range(batch))),
        active_count,
        _cuda_tensor([0] * batch),
        raw,
        history,
        32,
        out,
    )
    assert (out == -1).all()
    read = torch.full_like(raw, 999)
    merge_sparse_read_table(raw, raw, history, active_count, read, fetched_page_scale=1)
    assert (read == -1).all()


@_requires_cuda
@pytest.mark.parametrize("scale", [1, 4])
@pytest.mark.parametrize("strided", [False, True])
def test_merge_sparse_read_table(scale: int, strided: bool) -> None:
    generator = torch.Generator().manual_seed(17)
    fetched = torch.randint(0, 4000, (5, 513), generator=generator, dtype=torch.int32)
    write = torch.randint(0, 4000, (5, 513), generator=generator, dtype=torch.int32)
    fetched[:, ::17] = -1
    fetched[1, 0] = 0
    fetched[2, 1] = 1 << 30  # Scaling must not wrap this to a valid page zero.
    write[:, ::11] = -1
    write[0, 0] = 0
    history = torch.tensor([0, 1, 257, 513, 2], dtype=torch.int32)
    fetched_gpu = _cuda_tensor(fetched, strided)
    write_gpu = _cuda_tensor(write, strided)
    out = _cuda_tensor(torch.full_like(write, 999), strided)

    merge_sparse_read_table(
        fetched_gpu,
        write_gpu,
        _cuda_tensor(history, strided),
        _cuda_tensor([4]),
        out,
        fetched_page_scale=scale,
    )

    expected = _merge_reference(fetched, write, history, 4, scale)
    torch.testing.assert_close(out.cpu(), expected, rtol=0, atol=0)
    torch.testing.assert_close(fetched_gpu.cpu(), fetched, rtol=0, atol=0)
    torch.testing.assert_close(write_gpu.cpu(), write, rtol=0, atol=0)


@_requires_cuda
def test_sparse_offload_cuda_graph_replay() -> None:
    batch, max_blocks, topk_size, p = 3, 6, 8, 32
    topk = torch.zeros((batch, topk_size), dtype=torch.int32, device="cuda")
    requests = _cuda_tensor([2, 0, 1])
    active = _cuda_tensor([3])
    lengths = _cuda_tensor([128] * batch)
    history = _cuda_tensor([2, 0, 1])
    raw = torch.zeros((batch, max_blocks), dtype=torch.int32, device="cuda")
    fetched = torch.zeros_like(raw)
    write = torch.zeros_like(raw)
    selected = torch.empty_like(raw)
    read = torch.empty_like(raw)

    def run() -> None:
        select_sparse_history_pages(topk, requests, active, lengths, raw, history, p, selected)
        merge_sparse_read_table(fetched, write, history, active, read, fetched_page_scale=4)

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        run()
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        run()
    pointers = selected.data_ptr(), read.data_ptr()

    for n_active, request_ids, history_counts in (
        (3, [2, 0, 1], [2, 0, 1]),
        (1, [0, 0, 0], [3, 2, 1]),
        (0, [0, 0, 0], [3, 3, 3]),
        (3, [1, 2, 0], [0, 2, 3]),
    ):
        tokens_cpu = torch.tensor([[64, 0, 32, 95, 0, -1, 96, 127]] * batch, dtype=torch.int32)
        requests_cpu = torch.tensor(request_ids, dtype=torch.int32)
        history_cpu = torch.tensor(history_counts, dtype=torch.int32)
        lengths_cpu = torch.tensor([128, 97, 64], dtype=torch.int32)
        raw_cpu = torch.arange(batch * max_blocks, dtype=torch.int32).reshape(batch, max_blocks)
        raw_cpu[0, 1] = -1
        expected_selected = _selection_reference(
            tokens_cpu, requests_cpu, n_active, lengths_cpu, raw_cpu, history_cpu, p, max_blocks
        )
        # Emulate KVCM's mapping input to the merger. No transfer is implemented
        # by this test: the independently tested selector and merger are captured.
        fetched_cpu = torch.full_like(raw_cpu, -1)
        for request in range(n_active):
            for position, ordinal in enumerate(expected_selected[request].tolist()):
                if ordinal >= 0:
                    fetched_cpu[request, ordinal] = 100 + request * max_blocks + position
        write_cpu = raw_cpu + 500  # Deliberately populated history catches accidental fallback.
        for target, source in (
            (topk, tokens_cpu),
            (requests, requests_cpu),
            (active, torch.tensor([n_active])),
            (lengths, lengths_cpu),
            (history, history_cpu),
            (raw, raw_cpu),
            (fetched, fetched_cpu),
            (write, write_cpu),
        ):
            target.copy_(source)
        graph.replay()
        torch.testing.assert_close(selected.cpu(), expected_selected, rtol=0, atol=0)
        torch.testing.assert_close(
            read.cpu(),
            _merge_reference(fetched_cpu, write_cpu, history_cpu, n_active, 4),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(topk.cpu(), tokens_cpu, rtol=0, atol=0)
        torch.testing.assert_close(write.cpu(), write_cpu, rtol=0, atol=0)
        assert (selected.data_ptr(), read.data_ptr()) == pointers


@_requires_cuda
@pytest.mark.parametrize("compress_ratio", [1, 4, 128])
@pytest.mark.parametrize("split_extra", [False, True])
@pytest.mark.parametrize("mask_invalid_pages", [False, True])
def test_local_to_global_invalid_page_mask_is_opt_in(
    compress_ratio: int,
    split_extra: bool,
    mask_invalid_pages: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TRTLLM_ENABLE_PDL", "0")
    p = 128 // compress_ratio
    requests = [1, 0]
    swa_table = [[0, -1, 3], [-1, 2, 0]]
    compressed_table = [[160, -1, 36], [-1, 0, 164]]
    swa_tokens = [[0, 127, 128, 255, 256, -1, -7, 999]] * 2
    compressed_tokens = [[0, p - 1, p, 2 * p, 2 * p + p - 1, -1, -7, 999]] * 2
    # Omit the option for the default path to verify masking remains opt-in.
    mask_kwargs = {"mask_invalid_pages": True} if mask_invalid_pages else {}
    result = deepseek_v4_local_to_global_indices(
        req_id=_cuda_tensor(requests),
        block_table_swa=_cuda_tensor(swa_table),
        swa_local_indices=_cuda_tensor(swa_tokens),
        swa_pool_base_ptr=1 << 42,
        swa_buffer_ptr=(1 << 42) + 10000 * 1024,
        tokens_per_block=128,
        token_stride=1024,
        compressed_token_stride=512,
        block_table_compressed=_cuda_tensor(compressed_table) if compress_ratio > 1 else None,
        compressed_local_indices=_cuda_tensor(compressed_tokens) if compress_ratio > 1 else None,
        compress_pool_base_ptr=1 << 43,
        compressed_buffer_ptr=(1 << 43) + 20000 * 512,
        compress_ratio=compress_ratio,
        num_compressed_indices=8 if compress_ratio > 1 else 0,
        split_extra=split_extra,
        **mask_kwargs,
    )

    def reference(
        table: list[list[int]], tokens: list[list[int]], page_size: int, offset: int
    ) -> torch.Tensor:
        expected = torch.full((2, 8), -1, dtype=torch.int32)
        for query, request in enumerate(requests):
            for k, token in enumerate(tokens[query]):
                if 0 <= token < len(table[request]) * page_size:
                    page = table[request][token // page_size]
                    if page >= 0 or not mask_invalid_pages:
                        expected[query, k] = offset + page * page_size + token % page_size
        return expected

    expected_swa = reference(swa_table, swa_tokens, 128, 10000)
    expected_compressed = reference(compressed_table, compressed_tokens, p, 20000)
    if split_extra:
        swa, compressed = result
        torch.testing.assert_close(swa.cpu(), expected_swa, rtol=0, atol=0)
        if compress_ratio > 1:
            torch.testing.assert_close(compressed.cpu(), expected_compressed, rtol=0, atol=0)
        else:
            assert compressed is None
    else:
        expected = (
            torch.cat((expected_swa, expected_compressed), dim=1)
            if compress_ratio > 1
            else expected_swa
        )
        torch.testing.assert_close(result.cpu(), expected, rtol=0, atol=0)


@_requires_cuda
@pytest.mark.parametrize("scale", [1, 5])
def test_prepare_write_table_preserves_invalids_and_strides(scale):
    raw = torch.tensor(
        [[0, 2, -1, 1 << 30], [3, 0, 2, -9], [7, 8, 9, 10]], dtype=torch.int32, device="cuda"
    )
    storage = torch.full((6, 8), 999, dtype=torch.int32, device="cuda")
    out = storage[::2, ::2]
    history = torch.tensor([1, 2, 0], dtype=torch.int32, device="cuda")
    active = torch.tensor([2], dtype=torch.int32, device="cuda")
    prepare_sparse_write_table(raw, history, active, out, page_scale=scale)
    expected = [
        [-1, 2 * scale, -1, (1 << 30) if scale == 1 else -1],
        [-1, -1, 2 * scale, -1],
        [-1] * 4,
    ]
    torch.testing.assert_close(out.cpu(), torch.tensor(expected, dtype=torch.int32))
    assert (storage[1::2] == 999).all() and (storage[:, 1::2] == 999).all()


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
