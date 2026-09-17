# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""CPU tests for GLM sparse attention metadata and cache layouts."""

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from tensorrt_llm._torch.attention.backends.interface import (
    AttentionForwardArgs,
    AttentionInputType,
)
from tensorrt_llm._torch.attention.backends.sparse.glm_kpool import (
    GlmKpoolSparseAttention,
    GlmKpoolSparseParams,
    latent_pool_rows,
    paged_slot_indices,
)
from tensorrt_llm._torch.attention.backends.sparse.glm_kpool.params import (
    GlmKpoolBackendForwardArgs,
)

pytestmark = pytest.mark.cpu_only


@pytest.fixture
def forward_case():
    backend = object.__new__(GlmKpoolSparseAttention)
    backend.num_heads = 2
    backend.head_dim = backend.kv_lora_rank = 4
    state = SimpleNamespace(
        latent_pool=torch.arange(64, dtype=torch.bfloat16).view(4, 4, 4),
        block_tables=torch.tensor([[3, 0], [2, 1], [0, 2]]),
        num_contexts=1,
        tokens_per_block=4,
    )
    backend._cache_state = Mock(return_value=state)
    backend._dispatch_sparse_core = Mock()
    return SimpleNamespace(
        backend=backend,
        state=state,
        q=torch.ones(2, 8, dtype=torch.bfloat16),
        indices=torch.tensor([[0, -1], [1, 2]], dtype=torch.int32),
        metadata=object(),
    )


@pytest.mark.parametrize("route", ["context", "generation", "verify"])
@pytest.mark.parametrize("supply_output", [False, True])
def test_forward_dispatches_latent_rows_and_preserves_output(forward_case, route, supply_output):
    case = forward_case
    context = route == "context"
    q = case.q.repeat_interleave(2, dim=0) if route == "verify" else case.q
    indices = case.indices.repeat_interleave(2, dim=0) if route == "verify" else case.indices
    core_output = torch.arange(q.shape[0] * 8, dtype=q.dtype).view(q.shape[0], 2, 4)
    case.backend._dispatch_sparse_core.return_value = core_output
    selection = GlmKpoolBackendForwardArgs(topk_rows=indices)
    output = torch.empty(q.shape[0], 8, dtype=q.dtype) if supply_output else None
    args = AttentionForwardArgs(
        attention_input_type=(
            AttentionInputType.context_only if context else AttentionInputType.generation_only
        ),
        sparse_backend_args=selection,
        output=output,
    )
    actual = case.backend.forward(q, None, None, case.metadata, args)
    torch.testing.assert_close(actual, core_output.flatten(1))
    if supply_output:
        assert actual is output
    case.backend._cache_state.assert_called_once_with(case.metadata)
    case.backend._dispatch_sparse_core.assert_called_once()
    dispatched_q, dispatched_k, dispatched_indices = (
        case.backend._dispatch_sparse_core.call_args.args
    )
    torch.testing.assert_close(dispatched_q, q.view(-1, 2, 4))
    torch.testing.assert_close(dispatched_k, case.state.latent_pool.view(-1, 1, 4))
    torch.testing.assert_close(dispatched_indices, indices)


@pytest.mark.parametrize(
    "invalid",
    [
        "selection",
        "legacy_indices",
        "v",
        "out_scale",
        "out_scale_sf",
        "output_sf",
        "shape",
        "dtype",
        "device",
    ],
)
def test_forward_rejects_invalid_arguments_before_cache_access(forward_case, invalid):
    case = forward_case
    args = AttentionForwardArgs(
        attention_input_type=AttentionInputType.context_only,
        sparse_backend_args=GlmKpoolBackendForwardArgs(topk_rows=case.indices),
    )
    v = None
    error, message = ValueError, "quantized attention output"
    if invalid == "selection":
        args.sparse_backend_args = None
        error, message = ValueError, "pool-expanded selection"
    elif invalid == "legacy_indices":
        args.sparse_backend_args.topk_indices = case.indices
        message = "not request-local topk_indices"
    elif invalid == "v":
        v = torch.zeros(1)
        message = "v must be None"
    elif invalid in ("out_scale", "out_scale_sf", "output_sf"):
        setattr(args, invalid, torch.tensor(1.0))
    else:
        args.output = torch.empty(
            2,
            9 if invalid == "shape" else 8,
            dtype=torch.float32 if invalid == "dtype" else case.q.dtype,
            device="meta" if invalid == "device" else "cpu",
        )
        message = "forward_args.output must be"
    with pytest.raises(error, match=message):
        case.backend.forward(case.q, None, v, case.metadata, args)
    case.backend._cache_state.assert_not_called()
    case.backend._dispatch_sparse_core.assert_not_called()


def test_forward_rejects_mixed_phase(forward_case):
    case = forward_case
    selection = GlmKpoolBackendForwardArgs(topk_rows=case.indices)
    args = AttentionForwardArgs(
        attention_input_type=AttentionInputType.mixed, sparse_backend_args=selection
    )
    with pytest.raises(ValueError, match="phase-explicit"):
        case.backend.forward(case.q, None, None, case.metadata, args)
    case.backend._dispatch_sparse_core.assert_not_called()


@pytest.mark.parametrize("context", [False, True])
def test_forward_rejects_explicit_latent_source(forward_case, context):
    case = forward_case
    args = AttentionForwardArgs(
        attention_input_type=(
            AttentionInputType.context_only if context else AttentionInputType.generation_only
        ),
        sparse_backend_args=GlmKpoolBackendForwardArgs(topk_rows=case.indices),
    )
    k = torch.zeros(3, 4, dtype=case.q.dtype)
    message = "k must be None"
    with pytest.raises(ValueError, match=message):
        case.backend.forward(case.q, k, None, case.metadata, args)
    case.backend._dispatch_sparse_core.assert_not_called()


@pytest.mark.parametrize("is_cuda_graph", [False, True])
def test_cache_state_uses_prepared_metadata_without_fallback(is_cuda_graph):
    backend = object.__new__(GlmKpoolSparseAttention)
    backend.layer_idx = 0
    latent = torch.zeros(4, 8, 1, 512, dtype=torch.bfloat16)
    index = torch.zeros(4, 8, 1, 384, dtype=torch.bfloat16)
    tables = torch.tensor([[2, 0], [3, 1]], dtype=torch.long)
    live_lengths = torch.tensor([5, 8], dtype=torch.int32)
    metadata = SimpleNamespace(
        kv_cache_manager=SimpleNamespace(
            tokens_per_block=8,
            get_latent_state_buffer=lambda _: latent,
            get_index_state_buffer=lambda _: index,
        ),
        seq_lens=torch.ones(2, dtype=torch.long),
        num_contexts=0,
        is_cuda_graph=is_cuda_graph,
        kv_lens_cuda=live_lengths,
        mamba_metadata=SimpleNamespace(glm_block_tables=tables),
    )
    state = backend._cache_state(metadata)
    assert state.block_tables.data_ptr() == tables.data_ptr()
    assert state.kv_lens.data_ptr() == live_lengths.data_ptr()
    torch.testing.assert_close(state.kv_lens, live_lengths)
    metadata.mamba_metadata.glm_block_tables = None
    with patch("torch.zeros", side_effect=AssertionError("must not allocate fallback tables")):
        with pytest.raises(RuntimeError, match="requires prepared glm_block_tables"):
            backend._cache_state(metadata)
    metadata.mamba_metadata.glm_block_tables = tables
    metadata.kv_lens_cuda = None
    with pytest.raises(ValueError, match="kv_lens_cuda"):
        backend._cache_state(metadata)


@pytest.mark.parametrize("heads", [16, 64])
def test_backend_output_keeps_flat_contract_without_copy(heads):
    storage = torch.randn(3, 64, 512)
    per_head = storage[:, :heads]
    output = GlmKpoolSparseAttention._finalize_output(None, per_head, None)
    assert output.shape == (3, heads * 512)
    assert output.untyped_storage().data_ptr() == storage.untyped_storage().data_ptr()
    torch.testing.assert_close(output.view(3, heads, 512), per_head)
    supplied = torch.empty(3, heads * 512)
    assert GlmKpoolSparseAttention._finalize_output(None, per_head, supplied) is supplied
    torch.testing.assert_close(supplied, output)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"index_kpool": 0},
        {"index_kpool": 3},
        {"index_topk": 0},
        {"index_topk": 7},
        {"index_always_select_tail": False},
    ],
)
def test_unsupported_pool_layout_fails_before_kernel_launch(kwargs):
    with pytest.raises(ValueError, match="glm_kpool requires"):
        GlmKpoolSparseParams(**kwargs)


def test_derived_geometry_follows_the_checkpoint_defaults():
    params = GlmKpoolSparseParams()
    assert params.algorithm == "glm_kpool"
    assert params.select_k == 2048 // 4
    # topk positions plus the always-visible incomplete tail (kpool - 1 rows).
    assert params.output_width == 2048 + 3
    # Padded to the FlashMLA top-k tile.
    assert params.kernel_output_width == 2112
    assert params.packed_state_dim == 2 * 128
    assert params.cache_row_dim == 3 * 128
    small = GlmKpoolSparseParams(index_topk=64, index_kpool=8)
    assert (small.select_k, small.output_width, small.kernel_output_width) == (8, 71, 128)


def test_paged_slot_indices_returns_page_and_offset_pairs():
    table = torch.tensor([[7, 2, 9], [4, 0, 1]])
    positions = torch.tensor([[0, 5, 16], [3, 8, 23]])
    page, offset = paged_slot_indices(table, positions, tokens_per_block=8)
    assert torch.equal(page, torch.tensor([[7, 7, 9], [4, 0, 1]]))
    assert torch.equal(offset, torch.tensor([[0, 5, 0], [3, 0, 7]]))


def test_latent_pool_rows_reinterprets_a_coalesced_pool_without_copying():
    dim, tpb, slots = 16, 4, 3
    # A wider shared storage: each slot holds this buffer's page plus another
    # buffer's payload, so the slot stride exceeds tpb * dim (V2 coalescing).
    storage = torch.arange(slots * 2 * tpb * dim, dtype=torch.float32).view(slots, 2 * tpb, dim)
    pool = storage[:, :tpb, :]
    rows, base_row, rows_per_slot = latent_pool_rows(pool)
    assert (base_row, rows_per_slot) == (0, 2 * tpb)
    assert rows.data_ptr() == pool.data_ptr()
    for slot in range(slots):
        for t in range(tpb):
            assert torch.equal(rows[base_row + slot * rows_per_slot + t, 0], pool[slot, t])
    # A storage offset that is a whole number of rows is folded into base_row.
    shifted = storage.view(-1)[2 * dim :].view(-1, dim)[: slots * tpb].view(slots, tpb, dim)
    _, base, per_slot = latent_pool_rows(shifted)
    assert (base, per_slot) == (2, tpb)


def test_latent_pool_rows_rejects_layouts_without_a_uniform_row_view():
    dim, tpb, slots = 16, 4, 3
    ragged = torch.zeros(slots, tpb, dim + 1)[..., :dim]  # rows not contiguous within a page
    with pytest.raises(ValueError, match="contiguous within a page"):
        latent_pool_rows(ragged)
    odd_stride = torch.zeros(slots * tpb * dim + slots * 3).as_strided(
        (slots, tpb, dim), (tpb * dim + 3, dim, 1)
    )
    with pytest.raises(ValueError, match="not multiples of dim"):
        latent_pool_rows(odd_stride)
