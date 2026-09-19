# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DSA's shared logical selection path, independent of model weights and page tables."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from tensorrt_llm._torch.attention.backends.sparse.dsa.selection import (
    make_dsa_selection,
    select_dsa_topk,
)


@pytest.mark.parametrize("generation", [False, True])
@pytest.mark.parametrize("shared", [False, True])
def test_logical_boundary(generation: bool, shared: bool) -> None:
    positions = torch.tensor([[4, 1, 4, -1]], dtype=torch.int32)
    shared_topk = torch.full((2, 4), -1, dtype=torch.int32)
    if shared:
        shared_topk[1 if generation else 0].copy_(positions[0])
    metadata = SimpleNamespace(
        num_ctx_tokens=1,
        num_tokens=2,
        num_seqs=2,
        req_idx_per_token=torch.tensor([0, 1], dtype=torch.int32),
        kv_lens_cuda=torch.tensor([5, 9], dtype=torch.int32),
        shared_topk_indices=shared_topk,
        in_mtp_draft_loop=False,
    )
    indexer = Mock(mtp_index_share=False)
    indexer.forward_from_projected.return_value = positions
    q = torch.empty(0)
    intermediates = [torch.empty(0)]
    logical = select_dsa_topk(
        None if shared else indexer, q, metadata, intermediates, is_generation=generation
    )
    rows = torch.tensor([3, 0], dtype=torch.int32)
    generations = torch.tensor([12, 7], dtype=torch.uint64)
    selected = make_dsa_selection(logical, metadata, rows, generations, is_generation=generation)
    assert selected.positions.tolist() == [[4, 1, 4, -1]]
    assert selected.positions is logical
    context = selected.context
    assert (
        context.req_idx_per_token.data_ptr()
        == metadata.req_idx_per_token[1 if generation else 0 :].data_ptr()
    )
    assert context.kv_lens_cuda.data_ptr() == metadata.kv_lens_cuda.data_ptr()
    assert context.host_source_rows is rows
    assert context.host_source_generations is generations
    # Phase slicing preserves absolute batch indices; generation does not rebase them.
    assert context.req_idx_per_token.tolist() == [1 if generation else 0]
    metadata.kv_lens_cuda[1] = 10
    assert context.kv_lens_cuda.tolist() == [5, 10]
    assert shared_topk[1 if generation else 0].tolist() == positions[0].tolist()
    assert shared_topk[0 if generation else 1].tolist() == [-1] * 4
    if shared:
        indexer.forward_from_projected.assert_not_called()
    else:
        indexer.forward_from_projected.assert_called_once_with(
            metadata, q, intermediates, is_generation=generation
        )
        assert logical is positions


def test_mtp_shared_topk_is_not_overwritten() -> None:
    positions = torch.tensor([[2, 3]], dtype=torch.int32)
    shared = torch.tensor([[8, 9]], dtype=torch.int32)
    indexer = Mock(mtp_index_share=True)
    indexer.forward_from_projected.return_value = positions
    metadata = SimpleNamespace(
        num_ctx_tokens=0, num_tokens=1, shared_topk_indices=shared, in_mtp_draft_loop=True
    )
    assert select_dsa_topk(indexer, torch.empty(0), metadata, [], is_generation=True) is positions
    assert shared.tolist() == [[8, 9]]
