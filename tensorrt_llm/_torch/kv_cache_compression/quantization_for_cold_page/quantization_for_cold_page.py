# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Quantization policies for KVCM V2 cold pages."""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Sequence

from ...pyexecutor.resource_manager import DataType, KVCacheCompressionManager

if TYPE_CHECKING:
    from tensorrt_llm.llmapi.llm_args import ColdPageQuantizationCompressionConfig


class ColdPageQuantizationMethod(ABC):
    """Configured quantization method that creates one codec per KVCM."""

    @abstractmethod
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
        """Create a codec using this method's immutable calibration."""


class ColdPageCodecPolicy(ABC):
    """Per-KVCM Python callback contract used by the generic native codec."""

    @property
    @abstractmethod
    def layer_ids(self) -> tuple[int, ...]:
        """Layers transformed by this policy; other lifecycles stay lossless."""

    @abstractmethod
    def configure(self, lifecycles: Sequence[object]) -> Sequence[object]:
        """Resolve each owned lifecycle into immutable method metadata."""

    @abstractmethod
    def encode(
        self,
        lifecycle_index: int,
        cold_base: int,
        page_indices: int,
        num_pages: int,
        stream: int,
    ) -> None:
        """Submit one complete hot-to-cold Page batch."""

    @abstractmethod
    def decode(
        self,
        lifecycle_index: int,
        cold_base: int,
        page_indices: int,
        num_pages: int,
        stream: int,
    ) -> None:
        """Submit one complete cold-to-hot Page batch."""


class ColdPageQuantizationCompression(KVCacheCompressionManager):
    """Select and own the configured cold-page quantization method."""

    uses_iteration_lifecycle = False
    provides_cold_page_codec = True

    def __init__(self, config: "ColdPageQuantizationCompressionConfig") -> None:
        super().__init__(config)
        from .nvfp4 import Nvfp4ColdPageQuantization

        methods = {"nvfp4": Nvfp4ColdPageQuantization}
        try:
            method = methods[config.quant]
        except KeyError as error:
            raise NotImplementedError(
                f"Unsupported cold-page quantization format {config.quant!r}"
            ) from error
        self._method: ColdPageQuantizationMethod = method(config.scale_checkpoint_path)

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
        """Create the native codec selected by the quantization method."""

        return self._method.create_cold_page_codec(
            cache_config,
            runtime_dtype=runtime_dtype,
            pp_layers=pp_layers,
            num_kv_heads_per_layer=num_kv_heads_per_layer,
            head_dim_per_layer=head_dim_per_layer,
            is_draft=is_draft,
        )
