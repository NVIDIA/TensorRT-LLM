# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Quantization policies for KVCM V2 cold pages."""

from typing import TYPE_CHECKING, Sequence

from ...pyexecutor.resource_manager import DataType, KVCacheCompressionManager
from .nvfp4 import Nvfp4ColdPagePolicy

if TYPE_CHECKING:
    from tensorrt_llm.llmapi.llm_args import ColdPageQuantizationCompressionConfig


class ColdPageQuantizationCompression(KVCacheCompressionManager):
    """Select and own the configured cold-page quantization policy."""

    uses_iteration_lifecycle = False
    provides_cold_page_codec = True

    def __init__(self, config: "ColdPageQuantizationCompressionConfig") -> None:
        super().__init__(config)
        policies = {"nvfp4": Nvfp4ColdPagePolicy}
        try:
            policy = policies[config.quant]
        except KeyError as error:
            raise NotImplementedError(
                f"Unsupported cold-page quantization format {config.quant!r}"
            ) from error
        self._policy = policy(config.scale_checkpoint_path)

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
        """Create the native codec selected by the quantization policy."""

        return self._policy.create_cold_page_codec(
            cache_config,
            runtime_dtype=runtime_dtype,
            pp_layers=pp_layers,
            num_kv_heads_per_layer=num_kv_heads_per_layer,
            head_dim_per_layer=head_dim_per_layer,
            is_draft=is_draft,
        )
