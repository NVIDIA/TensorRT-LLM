# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Read-only FP8 history verification at the shared sparse-prediction boundary."""

from __future__ import annotations

import functools
from types import ModuleType
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

    from tensorrt_llm._torch.attention.backends.interface import AttentionForwardArgs
    from tensorrt_llm._torch.attention.backends.sparse.deepseek_v4.backend import (
        DeepseekV4TrtllmAttention,
    )
    from tensorrt_llm._torch.attention.backends.sparse.deepseek_v4.metadata import (
        DeepseekV4TrtllmAttentionMetadata,
    )
    from tensorrt_llm._torch.attention.backends.sparse.offload import SparseOffloadCacheMixin
    from tensorrt_llm.runtime.kv_cache_manager_v2 import _KVCache

    from .sparse_offload_accuracy import SparseHistoryVerifier
else:
    from sparse_offload_accuracy import SparseHistoryVerifier

_verifiers: dict[tuple[int, int, int], SparseHistoryVerifier] = {}


def finish_request(
    manager: SparseOffloadCacheMixin, request_id: int, cache: _KVCache
) -> dict | None:
    """Remove one request's verifier and return its completed integrity evidence.

    Args:
        manager: Sparse offload manager that owns the released request.
        request_id: Manager-local request identifier.
        cache: Native cache identity associated with that request.
    """
    verifier = _verifiers.pop((id(manager), request_id, id(cache)), None)
    return None if verifier is None else verifier.metrics()


def _verify_generation(
    attention: DeepseekV4TrtllmAttention,
    metadata: DeepseekV4TrtllmAttentionMetadata,
    forward_args: AttentionForwardArgs,
    indices: torch.Tensor | None,
) -> None:
    import torch

    from tensorrt_llm._utils import TensorWrapper, convert_to_torch_tensor

    state = getattr(metadata, "sparse_offload_state", None)
    layer = attention.layer_idx
    if state is None or layer not in state.layers:
        return
    manager = metadata.kv_cache_manager
    members = manager._sparse_batch_members
    if not members:
        return
    label = f"layer {layer}, requests {metadata.request_ids}"
    assert state.prepared, f"{label}: sparse state is not prepared"
    assert state.batch is not None, f"{label}: sparse batch is missing"
    assert metadata.num_contexts == 0, f"{label}: expected decode-only metadata"
    assert attention.compress_ratio == 4, f"{label}: expected ratio-4 compression"
    assert not attention._uses_nvfp4_compress, f"{label}: expected FP8 compressed history"
    descriptor = state.layers[layer]
    data_index = descriptor.data_buffer_index
    buffer_id = descriptor.buffer_ids[data_index]
    pages = list(manager.impl.get_aggregated_pages([buffer_id]))
    assert len(pages) == 1, f"{label}: expected one host page group, got {len(pages)}"
    host = pages[0]
    assert len(host.buffers) == 1, f"{label}: expected one host buffer, got {len(host.buffers)}"
    assert host.buffers[0].id == buffer_id, (
        f"{label}: host buffer {host.buffers[0].id} differs from descriptor {buffer_id}"
    )
    assert host.buffers[0].expansion == 1, f"{label}: expected an unexpanded FP8 host buffer"
    units_per_page = manager.tokens_per_block // descriptor.unit_tokens
    assert int(host.size) % units_per_page == 0, (
        f"{label}: host row size {host.size} is not divisible by {units_per_page} units"
    )
    row_bytes = int(host.size) // units_per_page
    assert row_bytes == attention.head_dim, (
        f"{label}: host row bytes {row_bytes} differ from head dimension {attention.head_dim}"
    )
    native = forward_args.sparse_runtime_params
    assert int(native.aux_kv_cache_pool_ptr) == descriptor.arena_base_addresses[data_index], (
        f"{label}: attention pool does not match the offload descriptor"
    )
    assert indices is not None, f"{label}: sparse predictor returned no attention indices"
    indices = indices.detach().cpu()
    backend_args = forward_args.sparse_backend_args
    assert backend_args is not None, f"{label}: sparse backend arguments are missing"
    assert backend_args.topk_indices is not None, f"{label}: sparse selections are missing"
    selected = backend_args.topk_indices.detach().cpu()
    window = attention.sparse_attention_config.window_size
    extra = metadata.max_compressed_indices[4]
    assert indices.shape[1] == window + extra, (
        f"{label}: index width {indices.shape[1]} differs from window/history {window}/{extra}"
    )
    assert selected.shape[1] == extra, f"{label}: selection width {selected.shape[1]} != {extra}"
    assert indices.shape[0] == selected.shape[0] == metadata.num_tokens, (
        f"{label}: index/selection/token rows "
        f"{indices.shape[0]}/{selected.shape[0]}/{metadata.num_tokens} differ"
    )
    request_positions = metadata.req_idx_per_token.detach().cpu().flatten().tolist()
    token_positions = metadata.token_positions_cuda.detach().cpu().flatten().tolist()
    request_ids = metadata.request_ids
    batch_rows = state.batch_rows.detach().cpu().tolist()
    active_count = int(state.active_request_count.item())
    lengths = metadata.compressed_kv_lens_cuda[4].detach().cpu().flatten().tolist()
    arena = convert_to_torch_tensor(
        TensorWrapper(
            int(native.aux_kv_cache_pool_ptr),
            torch.uint8,
            (descriptor.arena_row_counts[data_index], row_bytes),
        )
    )
    consumer_pool = manager.get_sparse_compress_buffers(layer)[0]
    for query in range(metadata.num_tokens):
        request_position = int(request_positions[query])
        assert 0 <= request_position < active_count, (
            f"{label}: request position {request_position} outside {active_count} active requests"
        )
        request_id = int(request_ids[request_position])
        member = members[request_id]
        request_label = f"layer {layer}, request {request_id}, position {token_positions[query]}"
        assert int(batch_rows[request_position]) == int(member.row), (
            f"{request_label}: batch row {batch_rows[request_position]} != member row {member.row}"
        )
        history = int(member.cache.history_length)
        units = selected[query].to(torch.int64)
        slots = indices[query, window:].to(torch.int64)
        valid = (units >= 0) & (units < lengths[request_position])
        assert bool((slots[~valid] == -1).all()), (
            f"{request_label}: invalid logical units have non-padding slots"
        )
        assert bool((slots[valid] >= 0).all()), (
            f"{request_label}: valid logical units have negative slots"
        )
        committed = valid & (units < history // descriptor.unit_tokens)
        units = units[committed]
        slots = slots[committed]
        if units.numel() == 0:
            assert history < descriptor.unit_tokens, (
                f"{request_label}: no committed selection from {history} history tokens"
            )
            continue
        host_pages = list(member.cache.get_aggregated_page_indices(int(host.layer_group_id)))
        host_view = convert_to_torch_tensor(
            TensorWrapper(
                int(host.base),
                torch.uint8,
                (max(host_pages) + 1, units_per_page, row_bytes),
                (int(host.stride), row_bytes, 1),
            )
        )
        key = (id(manager), request_id, id(member.cache))
        verifier = _verifiers.get(key)
        if verifier is None:
            verifier = _verifiers[key] = SparseHistoryVerifier()
        verifier.check_tensors(
            layer,
            units,
            slots,
            host_view,
            host_pages,
            arena,
            consumer_pool=consumer_pool,
            token_position=int(token_positions[query]),
        )


def observe(module: ModuleType) -> None:
    """Observe the predictor shared by fallback and PrimTS attention consumers.

    Args:
        module: Imported DeepSeek V4 sparse backend module.
    """
    attention_cls = module.DeepseekV4TrtllmAttention
    predictor = attention_cls.sparse_attn_predict

    @functools.wraps(predictor)
    def observed_prediction(
        self: DeepseekV4TrtllmAttention,
        query: torch.Tensor,
        key: torch.Tensor | None,
        metadata: DeepseekV4TrtllmAttentionMetadata,
        forward_args: AttentionForwardArgs,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        indices, offsets = predictor(self, query, key, metadata, forward_args)
        _verify_generation(self, metadata, forward_args, indices)
        return indices, offsets

    attention_cls.sparse_attn_predict = observed_prediction
