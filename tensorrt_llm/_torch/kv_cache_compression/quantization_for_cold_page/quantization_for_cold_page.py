# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Common runtime pipeline for cold-page quantization."""

from dataclasses import replace
from typing import Any, Optional, Sequence

from ...pyexecutor.resource_manager import DataType, KVCacheCompressionManager


class ColdPageQuantizationCompression(KVCacheCompressionManager):
    """Common codec registration and callbacks for cold-page quantizers."""

    uses_iteration_lifecycle = False
    provides_cold_page_codec = True

    @property
    def mixed_precision_high_window(self) -> Optional[tuple[int, int]]:
        """The first/last-N tokens the config keeps at the active precision."""
        return getattr(self.config, "mixed_precision_high_window", None)

    def create_cold_page_codec(
        self,
        cache_config: object,
        *,
        runtime_dtype: DataType,
        pp_layers: Sequence[int],
        num_kv_heads_per_layer: Sequence[int],
        head_dim_per_layer: Sequence[int],
        is_draft: bool = False,
        mixed_precision_high_layers: Sequence[int] = (),
    ) -> object:
        """Create one native codec with state isolated to this KVCM.

        The mixed-precision high layers stay out of the codec: the native codec migrates
        the lifecycles it does not own unchanged, which keeps those tokens at the active
        precision in the cold tiers.
        """

        from tensorrt_llm.bindings.internal import kv_cache_compression as native

        if mixed_precision_high_layers:
            excluded = {int(layer) for layer in mixed_precision_high_layers}
            cache_config = replace(
                cache_config,
                layers=[
                    layer for layer in cache_config.layers if int(layer.layer_id) not in excluded
                ],
            )
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
