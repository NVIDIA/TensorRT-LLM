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

"""Tests for compact sparse history selection and DeepSeek-V4 index conversion."""

import pytest
import torch

from tensorrt_llm._torch.attention.backends.sparse.deepseek_v4.kernels import (
    deepseek_v4_local_to_global_indices,
)
from tensorrt_llm._torch.attention.backends.sparse.offload_kernels import (
    select_sparse_history_units,
)

_requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")


@_requires_cuda
@pytest.mark.parametrize("strided", [False, True])
@pytest.mark.parametrize("active", [0, 2, 4])
@pytest.mark.parametrize("split_extra", [False, True])
@pytest.mark.parametrize("batch_rows", [[0, 1, 2, 3], [2, 0, 3, 1]])
def test_sparse_index_preparation_merges_history_and_writes(
    strided, active, split_extra, batch_rows
):
    # Request order differs from query order, and Batch rows need not follow
    # either. Some rows complete a unit, one has no newly completed row, and
    # another has no valid write destination.
    topk = _cuda_tensor([[1, -1, 0, 99], [32, 31, 0, -2], [0, -1, -1, -1], [0, 1, 2, -1]], strided)
    requests = _cuda_tensor([1, 0, 3, 2], strided)
    lengths = _cuda_tensor([33, 2, 3, 1], strided)
    # Watermarks, fetches and writes of requests 0..3, stored at their Batch rows.
    by_row = torch.tensor(batch_rows).argsort()
    history = _cuda_tensor(torch.tensor([131, 8, 11, 3])[by_row], strided)
    fetched = _cuda_tensor((torch.arange(4 * 3 * 4).reshape(4, 3, 4) + 100)[by_row], strided)[:, 1]
    writes = _cuda_tensor(
        torch.tensor([[7, 7, 50], [-1, -1, -1], [5, 5, 42], [-1, -1, -1]])[by_row], strided
    )[:, 1]
    kwargs = dict(
        req_id=requests,
        block_table_swa=_cuda_tensor([[2], [3], [5], [9]]),
        swa_local_indices=_cuda_tensor([[0, 1, -1, 3]] * 4),
        swa_pool_base_ptr=4096,
        swa_buffer_ptr=4096,
        tokens_per_block=4,
        token_stride=1024,
        compress_ratio=4,
        num_compressed_indices=4,
        compressed_local_indices=topk,
        compressed_fetched_indices=fetched,
        compressed_write_indices=writes,
        compressed_history_lengths=history,
        compressed_lengths=lengths,
        compressed_batch_rows=_cuda_tensor(batch_rows, strided),
        active_request_count=_cuda_tensor([active]),
        mask_invalid_pages=True,
        split_extra=split_extra,
    )
    out = deepseek_v4_local_to_global_indices(**kwargs)
    expected = torch.full((4, 8), -1, dtype=torch.int32)
    for query, request in enumerate(requests.cpu().tolist()):
        if query >= active or request >= active:
            continue
        # The SWA table and the lengths follow the request; sparse state, its row.
        page = [2, 3, 5, 9][request]
        row = batch_rows[request]
        expected[query, :4] = torch.tensor([page * 4, page * 4 + 1, -1, page * 4 + 3])
        for slot, unit in enumerate(topk.cpu().tolist()[query]):
            if not 0 <= unit < lengths[request].item():
                continue
            if unit < history[row].item() // 4:
                expected[query, 4 + slot] = fetched[row, slot].cpu()
            elif unit == history[row].item() // 4:
                expected[query, 4 + slot] = writes[row].cpu()
    if split_extra:
        torch.testing.assert_close(out[0].cpu(), expected[:, :4])
        torch.testing.assert_close(out[1].cpu(), expected[:, 4:])
    else:
        torch.testing.assert_close(out.cpu(), expected)


@_requires_cuda
def test_sparse_index_preparation_replays_with_new_watermarks_and_padding():
    # The graph has two query rows; native Batch retains four rows. Invalid
    # request IDs in graph padding must never index either cache's metadata.
    requests = _cuda_tensor([0, 999])
    topk = _cuda_tensor([[0, 1, 2, -1], [0, 1, 2, 3]])
    fetched = _cuda_tensor([[71, 29, 88, -1]] * 4)
    writes = _cuda_tensor([7, -1, -1, -1], strided=True)
    history = _cuda_tensor([7, 0, 0, 0])
    lengths = _cuda_tensor([2, 0])
    batch_rows = _cuda_tensor([0, -1, -1, -1])
    active = _cuda_tensor([1])
    kwargs = dict(
        req_id=requests,
        block_table_swa=_cuda_tensor([[2], [3]]),
        swa_local_indices=_cuda_tensor([[0, 1, -1, 3]] * 2),
        swa_pool_base_ptr=4096,
        swa_buffer_ptr=4096,
        tokens_per_block=4,
        token_stride=1024,
        compress_ratio=4,
        num_compressed_indices=4,
        compressed_local_indices=topk,
        compressed_fetched_indices=fetched,
        compressed_write_indices=writes,
        compressed_history_lengths=history,
        compressed_lengths=lengths,
        compressed_batch_rows=batch_rows,
        active_request_count=active,
        mask_invalid_pages=True,
    )
    deepseek_v4_local_to_global_indices(**kwargs)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = deepseek_v4_local_to_global_indices(**kwargs)
    for row, history_tokens, length, destination, active_count, compressed in (
        (0, 7, 2, 7, 1, [71, 7, -1, -1]),
        (0, 8, 2, -1, 1, [71, 29, -1, -1]),
        (0, 11, 3, 5, 1, [71, 29, 5, -1]),
        (0, 11, 3, -1, 1, [71, 29, -1, -1]),
        (0, 11, 3, 5, 0, [-1] * 4),
        # The request now sits in Batch row 2; the replay follows the new map.
        (2, 11, 3, 5, 1, [71, 29, 5, -1]),
    ):
        history.zero_()
        writes.fill_(-1)
        batch_rows[0] = row
        history[row] = history_tokens
        lengths[0] = length
        writes[row] = destination
        active[0] = active_count
        graph.replay()
        swa = [8, 9, -1, 11] if active_count else [-1] * 4
        torch.testing.assert_close(
            out.cpu(), torch.tensor([swa + compressed, [-1] * 8], dtype=torch.int32)
        )


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


def _selection_reference(topk, requests, batch_rows, active, lengths, history):
    out = torch.full((history.numel(), topk.shape[1]), -1, dtype=torch.int32)
    for query, request in enumerate(requests.tolist()):
        if query >= active or not 0 <= request < min(active, history.numel(), lengths.numel()):
            continue
        row = batch_rows[request].item()
        if not 0 <= row < history.numel():
            continue
        for k, unit in enumerate(topk[query].tolist()):
            if 0 <= unit < min(lengths[request], history[row] // 4):
                out[row, k] = unit
    return out


@_requires_cuda
@pytest.mark.parametrize("strided", [False, True])
@pytest.mark.parametrize("permuted", [False, True])
def test_select_history_units_uses_compressed_boundaries_and_preserves_slots(strided, permuted):
    # Requests 0..7 in Batch rows; when permuted, rows 4 and 6 are free.
    rows = [9, 2, 5, 0, 7, 1, 3, 8] if permuted else list(range(8))
    batch_rows = torch.tensor(rows + [-1, -1], dtype=torch.int32)
    request_history = torch.tensor([0, 3, 4, 7, 8, 127, 128, 129], dtype=torch.int32)
    history = torch.full((10,), 999, dtype=torch.int32)
    history[rows] = request_history
    lengths = (request_history + 1) // 4
    requests = torch.tensor([7, 4, 1, 6, 0, 5, 3, 2], dtype=torch.int32)
    topk = torch.tensor([[31, -1, 0, 32, 1, 30, -2, 1000]] * 8, dtype=torch.int32)
    out = _cuda_tensor(torch.full((10, 8), 999, dtype=torch.int32), strided)
    gpu_topk = _cuda_tensor(topk, strided)
    select_sparse_history_units(
        gpu_topk,
        _cuda_tensor(requests, strided),
        _cuda_tensor(batch_rows, strided),
        _cuda_tensor([8]),
        _cuda_tensor(lengths, strided),
        _cuda_tensor(history, strided),
        out,
        unit_tokens=4,
    )
    expected = _selection_reference(topk, requests, batch_rows, 8, lengths, history)
    torch.testing.assert_close(out.cpu(), expected)
    torch.testing.assert_close(gpu_topk.cpu(), topk)
    # History token 7 already makes compressed unit 0 eligible.
    assert out[rows[3], 2].item() == 0
    # Unit 1 is still in flight at history 7 / capacity 8.
    assert out[rows[3], 4].item() == -1
    assert out[rows[5], 0].item() == -1 and out[rows[6], 0].item() == 31
    # Full native Batch capacity is cleared, free rows included.
    free = sorted(set(range(10)) - set(rows))
    assert (out[free] == -1).all()


@_requires_cuda
@pytest.mark.parametrize("topk_size", [1, 7, 513, 2048])
def test_select_history_units_scattered(topk_size):
    generator = torch.Generator().manual_seed(42)
    topk = torch.stack(
        [torch.randperm(3000, generator=generator)[:topk_size] for _ in range(3)]
    ).to(torch.int32)
    topk[:, 1::5] = -1
    requests = torch.tensor([2, 0, 1], dtype=torch.int32)
    batch_rows = torch.tensor([3, 0, 4, -1, -1], dtype=torch.int32)
    history = torch.tensor([4127, 0, 0, 12000, 0], dtype=torch.int32)
    lengths = torch.tensor([3000, 1032, 1], dtype=torch.int32)
    out = torch.empty((5, topk_size), dtype=torch.int32, device="cuda")
    select_sparse_history_units(
        _cuda_tensor(topk),
        _cuda_tensor(requests),
        _cuda_tensor(batch_rows),
        _cuda_tensor([3]),
        _cuda_tensor(lengths),
        _cuda_tensor(history),
        out,
        unit_tokens=4,
    )
    torch.testing.assert_close(
        out.cpu(), _selection_reference(topk, requests, batch_rows, 3, lengths, history)
    )


@_requires_cuda
def test_select_history_units_masks_invalid_request_ids_rows_and_padding():
    # Queries 0 and 1 carry invalid request IDs, query 3 is graph padding, and
    # request 1 maps to no Batch row.
    out = torch.full((6, 2), 99, dtype=torch.int32, device="cuda")
    select_sparse_history_units(
        _cuda_tensor([[0, 1]] * 4),
        _cuda_tensor([-1, 999, 0, 0]),
        _cuda_tensor([5, -1, 2, -1, -1, -1]),
        _cuda_tensor([3]),
        _cuda_tensor([2] * 4),
        _cuda_tensor([8] * 6),
        out,
        unit_tokens=4,
    )
    torch.testing.assert_close(out[5], _cuda_tensor([0, 1]))
    assert (out[:5] == -1).all()
    out.fill_(99)
    select_sparse_history_units(
        _cuda_tensor([[0, 1]] * 2),
        _cuda_tensor([1, 0]),
        _cuda_tensor([5, -1, 2, -1, -1, -1]),
        _cuda_tensor([2]),
        _cuda_tensor([2] * 2),
        _cuda_tensor([8] * 6),
        out,
        unit_tokens=4,
    )
    torch.testing.assert_close(out[5], _cuda_tensor([0, 1]))
    assert (out[:5] == -1).all()


@_requires_cuda
@pytest.mark.parametrize(("batch", "queries", "topk"), [(4, 0, 8), (0, 0, 8), (4, 2, 0)])
def test_select_history_units_empty_shapes(batch, queries, topk):
    out = torch.full((batch, topk), 99, dtype=torch.int32, device="cuda")
    select_sparse_history_units(
        torch.empty((queries, topk), dtype=torch.int32, device="cuda"),
        torch.zeros(queries, dtype=torch.int32, device="cuda"),
        torch.full((batch,), -1, dtype=torch.int32, device="cuda"),
        _cuda_tensor([0]),
        torch.zeros(queries, dtype=torch.int32, device="cuda"),
        torch.zeros(batch, dtype=torch.int32, device="cuda"),
        out,
        unit_tokens=4,
    )
    assert (out == -1).all()


@_requires_cuda
@pytest.mark.parametrize(
    "invalid",
    ["dtype", "cpu", "width", "requests", "batch_rows", "history", "lengths", "unit"],
)
def test_select_history_units_rejects_invalid_layout(invalid):
    topk = _cuda_tensor([[0, 1], [1, 0]])
    requests, batch_rows, active, lengths, history = (
        _cuda_tensor([0, 1]),
        _cuda_tensor([0, 1, -1, -1]),
        _cuda_tensor([2]),
        _cuda_tensor([2, 2]),
        _cuda_tensor([8] * 4),
    )
    unit = 4
    if invalid == "dtype":
        topk = topk.to(torch.int64)
    elif invalid == "cpu":
        topk = topk.cpu()
    elif invalid == "width":
        topk = topk[:, :1]
    elif invalid == "requests":
        requests = requests[:1]
    elif invalid == "batch_rows":
        batch_rows = batch_rows[:2]
    elif invalid == "history":
        history = history[:2]
    elif invalid == "lengths":
        lengths = lengths[:1]
    else:
        unit = 0
    with pytest.raises(ValueError):
        select_sparse_history_units(
            topk,
            requests,
            batch_rows,
            active,
            lengths,
            history,
            torch.empty((4, 2), dtype=torch.int32, device="cuda"),
            unit_tokens=unit,
        )


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


def _cuda(value):
    return _cuda_tensor(value)


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
