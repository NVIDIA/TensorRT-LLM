# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Common runtime pipeline for cold-page quantization."""

import copy
from typing import Sequence

from ...pyexecutor.resource_manager import DataType, KVCacheCompressionManager


class ColdPageQuantizationCompression(KVCacheCompressionManager):
    """Common codec registration and callbacks for cold-page quantizers."""

    uses_iteration_lifecycle = False
    provides_cold_page_codec = True

    def create_cold_page_codec(
        self,
        cache_config: object,
        *,
        runtime_dtype: DataType,
        pp_layers: Sequence[int],
        num_kv_heads_per_layer: Sequence[int],
        head_dim_per_layer: Sequence[int],
        is_draft: bool = False,
    ) -> object:
        """Create one callback instance with state isolated to this KVCM."""

        from tensorrt_llm.bindings.internal import kv_cache_compression as native

        callback = copy.copy(self)
        callback._initialize_codec(
            cache_config,
            runtime_dtype=runtime_dtype,
            pp_layers=pp_layers,
            num_kv_heads_per_layer=num_kv_heads_per_layer,
            head_dim_per_layer=head_dim_per_layer,
            is_draft=is_draft,
        )
        callback._lifecycle_metadata = []
        return native.create_python_cold_page_codec(callback)

    @property
    def layer_ids(self) -> tuple[int, ...]:
        return self._layer_ids

    def configure(self, lifecycles: Sequence[object]) -> Sequence[object]:
        """Resolve hot buffers and publish each lifecycle's cold-page size."""

        from tensorrt_llm.bindings.internal import kv_cache_compression as native

        self._lifecycle_metadata = [
            self._build_lifecycle_metadata(lifecycle) for lifecycle in lifecycles
        ]
        properties = []
        for metadata in self._lifecycle_metadata:
            lifecycle = native.ColdPageLifecycleProperties()
            lifecycle.cold_page_bytes = metadata.cold_page_bytes
            lifecycle.page_index_location = native.ColdPageIndexLocation.HOST
            properties.append(lifecycle)
        return properties

    def encode(
        self,
        lifecycle_index: int,
        cold_base: int,
        page_indices: int,
        num_pages: int,
        stream: int,
    ) -> None:
        self._invoke_kernel(
            "encode",
            self._lifecycle_metadata[lifecycle_index],
            cold_base,
            page_indices,
            num_pages,
            stream,
        )

    def decode(
        self,
        lifecycle_index: int,
        cold_base: int,
        page_indices: int,
        num_pages: int,
        stream: int,
    ) -> None:
        self._invoke_kernel(
            "decode",
            self._lifecycle_metadata[lifecycle_index],
            cold_base,
            page_indices,
            num_pages,
            stream,
        )

    def _initialize_codec(
        self,
        cache_config: object,
        *,
        runtime_dtype: DataType,
        pp_layers: Sequence[int],
        num_kv_heads_per_layer: Sequence[int],
        head_dim_per_layer: Sequence[int],
        is_draft: bool = False,
    ) -> None:
        """Build format-specific immutable state for one KVCM."""
        raise NotImplementedError

    def _build_lifecycle_metadata(self, lifecycle: object) -> object:
        """Resolve one KVCM lifecycle into format-specific launch metadata."""
        raise NotImplementedError

    def _invoke_kernel(
        self,
        operation: str,
        metadata: object,
        cold_base: int,
        page_indices: int,
        num_pages: int,
        stream: int,
    ) -> None:
        """Submit one complete codec batch to the format-specific kernel."""
        raise NotImplementedError
