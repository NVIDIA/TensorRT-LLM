# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Common runtime pipeline for cold-page quantization."""

from typing import Any, Sequence

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
        """Create one native codec with state isolated to this KVCM."""

        from tensorrt_llm.bindings.internal import kv_cache_compression as native

        codec_state = self.build_codec_state(
            cache_config,
            runtime_dtype=runtime_dtype,
            pp_layers=pp_layers,
            num_kv_heads_per_layer=num_kv_heads_per_layer,
            head_dim_per_layer=head_dim_per_layer,
            is_draft=is_draft,
        )
        return native.create_python_cold_page_codec(self, codec_state)

    def configure(self, codec_state: Any, lifecycles: Sequence[object]) -> Sequence[object]:
        """Resolve hot buffers and publish each lifecycle's cold-page size."""

        from tensorrt_llm.bindings.internal import kv_cache_compression as native

        codec_state.lifecycle_metadata = tuple(
            self.build_lifecycle_metadata(codec_state, lifecycle) for lifecycle in lifecycles
        )
        properties = []
        for metadata in codec_state.lifecycle_metadata:
            lifecycle = native.ColdPageLifecycleProperties()
            lifecycle.cold_page_bytes = metadata.cold_page_bytes
            lifecycle.page_index_location = native.ColdPageIndexLocation.HOST
            properties.append(lifecycle)
        return properties

    def build_codec_state(
        self,
        cache_config: object,
        *,
        runtime_dtype: DataType,
        pp_layers: Sequence[int],
        num_kv_heads_per_layer: Sequence[int],
        head_dim_per_layer: Sequence[int],
        is_draft: bool = False,
    ) -> object:
        """Build the format-specific state owned by one native codec."""
        raise NotImplementedError

    def build_lifecycle_metadata(self, codec_state: object, lifecycle: object) -> object:
        """Resolve one KVCM lifecycle into format-specific launch metadata."""
        raise NotImplementedError
