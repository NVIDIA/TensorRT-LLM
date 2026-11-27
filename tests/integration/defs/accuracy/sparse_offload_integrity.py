# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Read-only FP8 sparse-history verification at the native attention boundary."""

from __future__ import annotations

import functools
import importlib.abc
import importlib.machinery
import sys
from collections.abc import Sequence
from types import ModuleType
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tensorrt_llm._torch.attention.backends.fmha.phased import FmhaParams
    from tensorrt_llm._torch.attention.backends.sparse.offload import SparseOffloadCacheMixin
    from tensorrt_llm.runtime.kv_cache_manager_v2 import _KVCache

    from .sparse_offload_accuracy import SparseHistoryVerifier
else:
    from sparse_offload_accuracy import SparseHistoryVerifier

_MODULE = "tensorrt_llm._torch.attention.backends.fmha.fallback"
_verifiers: dict[tuple[int, int, int], SparseHistoryVerifier] = {}


def finish_request(
    manager: SparseOffloadCacheMixin, request_id: int, cache: _KVCache
) -> dict | None:
    """Remove one request's verifier and return its completed integrity evidence."""
    verifier = _verifiers.pop((id(manager), request_id, id(cache)), None)
    return None if verifier is None else verifier.metrics()


def _verify_generation(params: FmhaParams) -> None:
    import torch

    from tensorrt_llm._utils import TensorWrapper, convert_to_torch_tensor

    metadata = params.meta
    state = getattr(metadata, "sparse_offload_state", None)
    layer = params.attn.layer_idx
    if state is None or layer not in state.layers:
        return
    manager = metadata.kv_cache_manager
    members = manager._sparse_batch_members
    if not members:
        return
    assert state.prepared and state.batch is not None
    assert params.token_offset == params.seq_offset == 0 and metadata.num_contexts == 0
    assert params.attn.compress_ratio == 4 and not params.attn._uses_nvfp4_compress
    descriptor = state.layers[layer]
    data_index = descriptor.data_buffer_index
    buffer_id = descriptor.buffer_ids[data_index]
    pages = list(manager.impl.get_aggregated_pages([buffer_id]))
    assert len(pages) == 1
    host = pages[0]
    assert len(host.buffers) == 1 and host.buffers[0].id == buffer_id
    assert host.buffers[0].expansion == 1
    units_per_page = manager.tokens_per_block // descriptor.unit_tokens
    assert int(host.size) % units_per_page == 0
    row_bytes = int(host.size) // units_per_page
    assert row_bytes == params.attn.head_dim
    native = params.fwd.sparse_runtime_params
    assert int(native.aux_kv_cache_pool_ptr) == descriptor.arena_base_addresses[data_index]
    indices = native.sparse_attn_indices.detach().cpu()
    selected = params.fwd.sparse_backend_args.topk_indices.detach().cpu()
    window = params.attn.sparse_attention_config.window_size
    extra = metadata.max_compressed_indices[4]
    assert indices.shape[1] == window + extra and selected.shape[1] == extra
    assert indices.shape[0] == selected.shape[0] == params.num_tokens
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
    for query in range(params.num_tokens):
        request_position = int(request_positions[query])
        assert 0 <= request_position < active_count
        request_id = int(request_ids[request_position])
        member = members[request_id]
        assert int(batch_rows[request_position]) == int(member.row)
        history = int(member.cache.history_length)
        units = selected[query].to(torch.int64)
        slots = indices[query, window:].to(torch.int64)
        valid = (units >= 0) & (units < lengths[request_position])
        assert bool((slots[~valid] == -1).all())
        assert bool((slots[valid] >= 0).all())
        committed = valid & (units < history // descriptor.unit_tokens)
        units = units[committed]
        slots = slots[committed]
        if units.numel() == 0:
            assert history < descriptor.unit_tokens
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
            token_position=int(token_positions[query]),
        )


def _observe(module: ModuleType) -> None:
    generation = module.thop.run_generation

    @functools.wraps(generation)
    def observed_generation(params: FmhaParams):
        _verify_generation(params)
        return generation(params)

    module.thop.run_generation = observed_generation


class _IntegrityLoader(importlib.abc.Loader):
    def __init__(self, loader: importlib.abc.Loader) -> None:
        self._loader = loader

    def create_module(self, spec: importlib.machinery.ModuleSpec) -> ModuleType | None:
        return self._loader.create_module(spec)

    def exec_module(self, module: ModuleType) -> None:
        self._loader.exec_module(module)
        _observe(module)


class _IntegrityFinder(importlib.abc.MetaPathFinder):
    def find_spec(
        self, fullname: str, path: Sequence[str] | None, target: ModuleType | None = None
    ) -> importlib.machinery.ModuleSpec | None:
        if fullname != _MODULE:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path, target)
        assert spec is not None and spec.loader is not None
        spec.loader = _IntegrityLoader(spec.loader)
        return spec


def install() -> None:
    """Install before attention imports, without initializing CUDA or MPI."""
    assert _MODULE not in sys.modules, "Integrity observer must precede attention imports"
    sys.meta_path.insert(0, _IntegrityFinder())
