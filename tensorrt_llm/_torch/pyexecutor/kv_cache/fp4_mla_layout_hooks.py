# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Hooks through which a KV cache manager V2 carries an FP4 MLA cache layout."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar, Iterable, List, Optional

import torch

from tensorrt_llm.runtime.kv_cache_manager_v2 import BufferConfig, DataRole

if TYPE_CHECKING:
    from tensorrt_llm._torch.attention.backends.fp4_mla.cache_layout import Fp4MlaCacheLayout
    from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest


class Fp4MlaLayoutHooks:
    """Forwards the layout hooks of a V2 manager to its FP4 MLA cache layout.

    A manager that stores an MLA latent cache as NVFP4 holds an
    ``Fp4MlaCacheLayout`` as ``fp4_mla_layout``. Without one, every hook is
    the next class's, so the manager behaves as if this class were absent.
    List it before ``KVCacheManagerV2`` in the bases.
    """

    fp4_mla_layout: Optional[Fp4MlaCacheLayout] = None
    # The hooks the layout takes over. A manager class that overrides one of
    # them after this class would shadow the layout.
    LAYOUT_HOOKS: ClassVar[tuple[str, ...]] = (
        "_get_buffer_roles_for_layer",
        "get_buffers",
        "get_layer_bytes_per_token",
        "_extra_buffers_per_layer",
        "_prepare_page_table_tensor",
        "_get_runtime_cache_size_layer_components",
        "_get_generation_request_capacity",
    )

    @property
    def fp4_mla_context_only_cache(self) -> bool:
        """Whether the manager keeps a single-copy FP4 MLA cache, which cannot execute generation."""
        layout = self.fp4_mla_layout
        return layout is not None and layout.fp4_mla_context_only_cache

    def _get_buffer_roles_for_layer(self, local_layer_idx: int) -> List[DataRole]:
        layout = self.fp4_mla_layout
        default = super()._get_buffer_roles_for_layer
        if layout is None:
            return default(local_layer_idx)
        return layout.buffer_roles_for_layer(local_layer_idx, default)

    def get_buffers(
        self, layer_idx: int, kv_layout: str = "NHD", **kwargs: Any
    ) -> Optional[torch.Tensor]:
        layout = self.fp4_mla_layout
        default = super().get_buffers
        if layout is None:
            return default(layer_idx, kv_layout, **kwargs)
        return layout.get_buffers(default, layer_idx, kv_layout, **kwargs)

    def get_layer_bytes_per_token(self, local_layer_idx: int, data_role: DataRole) -> int:
        layout = self.fp4_mla_layout
        default = super().get_layer_bytes_per_token
        if layout is None:
            return default(local_layer_idx, data_role)
        return layout.get_layer_bytes_per_token(local_layer_idx, data_role, default)

    def _extra_buffers_per_layer(
        self, *, tokens_per_block: int
    ) -> Optional[dict[int, List[BufferConfig]]]:
        layout = self.fp4_mla_layout
        if layout is None:
            return super()._extra_buffers_per_layer(tokens_per_block=tokens_per_block)
        return layout.extra_buffers_per_layer()

    def _prepare_page_table_tensor(self, index_mapper_capacity: int) -> None:
        layout = self.fp4_mla_layout
        if layout is None:
            super()._prepare_page_table_tensor(index_mapper_capacity)
            return
        layout.prepare_page_table_tensor(index_mapper_capacity)

    def _get_runtime_cache_size_layer_components(self) -> tuple[list[int], list[Optional[int]]]:
        layout = self.fp4_mla_layout
        if layout is None:
            return super()._get_runtime_cache_size_layer_components()
        return layout.runtime_cache_size_layer_components()

    def _get_generation_request_capacity(self) -> int:
        layout = self.fp4_mla_layout
        if layout is None:
            return super()._get_generation_request_capacity()
        return layout.generation_request_capacity()

    def rebuild_disagg_imported_cache(self, requests: Iterable[LlmRequest]) -> None:
        """Rebuild the cache state a generation worker derives from imported KV."""
        layout = self.fp4_mla_layout
        if layout is not None:
            layout.rebuild_disagg_imported_cache(requests)


__all__ = ["Fp4MlaLayoutHooks"]
