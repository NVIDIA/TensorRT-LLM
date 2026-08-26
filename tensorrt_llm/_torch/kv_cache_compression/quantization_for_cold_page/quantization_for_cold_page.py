# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Common construction pipeline for cold-page quantization."""

from typing import TYPE_CHECKING, Optional, Sequence

from tensorrt_llm._utils import is_sm_100f
from tensorrt_llm.logger import logger
from tensorrt_llm.quantization import QuantAlgo

from ...pyexecutor.resource_manager import DataType, KVCacheCompressionManager

if TYPE_CHECKING:
    from tensorrt_llm._torch.model_config import ModelConfig
    from tensorrt_llm.llmapi.llm_args import (
        ColdPageQuantizationCompressionConfig,
        KvCacheConfig,
        SpeculativeConfig,
    )

_NATIVE_KV_CACHE_EQUIVALENTS = {"nvfp4": QuantAlgo.NVFP4}


def validate_cold_page_quantization_compatibility(
    config: "ColdPageQuantizationCompressionConfig",
    kv_cache_config: "KvCacheConfig",
    spec_config: Optional["SpeculativeConfig"],
) -> None:
    """Validate a cold-page method that will participate in this executor."""

    from tensorrt_llm.runtime.kv_cache_manager_v2 import _BACKEND

    if _BACKEND == "python":
        raise ValueError("Cold-page quantization requires the C++ KVCacheManagerV2 backend")
    if kv_cache_config.enable_block_reuse and not config.supports_block_reuse():
        raise ValueError(
            f"KV-cache compression algorithm {config.algorithm!r} does not "
            "support KV-cache block reuse. Set "
            "KvCacheConfig.enable_block_reuse=False."
        )
    if spec_config is None:
        return
    if not config.supports_speculative_decoding():
        raise ValueError(
            f"KV-cache compression algorithm {config.algorithm!r} does not "
            "support speculative decoding with its current configuration"
        )
    mode = spec_config.spec_dec_mode
    if not (mode.is_eagle3_one_model() or mode.is_mtp_eagle_one_model()):
        raise ValueError(
            "Cold-page quantization supports speculative decoding only "
            f"with one-model MTP-EAGLE or EAGLE3, not {mode.name}"
        )


def create_cold_page_quantization_manager(
    config: "ColdPageQuantizationCompressionConfig",
    *,
    model_config: "ModelConfig",
    kv_cache_config: "KvCacheConfig",
    spec_config: Optional["SpeculativeConfig"],
    estimating_kv_cache: bool = False,
) -> Optional["ColdPageQuantizationCompression"]:
    """Select, validate, and construct one cold-page quantization method."""

    active_quant_algo = _NATIVE_KV_CACHE_EQUIVALENTS.get(config.quant)
    if active_quant_algo is None:
        raise NotImplementedError(f"Unsupported cold-page quantization format {config.quant!r}")
    if estimating_kv_cache:
        return None

    quant_config = model_config.quant_config
    if (
        quant_config is not None
        and getattr(quant_config, "kv_cache_quant_algo", None) == active_quant_algo
    ):
        logger.info(
            "Skipping cold-page %s quantization because the active KV cache "
            "already uses the same format; KVCM will migrate it losslessly.",
            config.quant.upper(),
        )
        return None

    validate_cold_page_quantization_compatibility(config, kv_cache_config, spec_config)

    from .nvfp4_quantization import Nvfp4ColdPageQuantizationCompression

    if not is_sm_100f():
        raise RuntimeError(
            "NVFP4 cold-page quantization requires an SM100-family device (SM100 or SM103)."
        )
    return Nvfp4ColdPageQuantizationCompression(config)


class ColdPageQuantizationCompression(KVCacheCompressionManager):
    """Base pipeline shared by storage-bound cold-page quantizers."""

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
        """Create one generic native codec around a format-specific policy."""

        from tensorrt_llm.bindings.internal import kv_cache_compression as native

        policy = self._create_cold_page_policy(
            cache_config,
            runtime_dtype=runtime_dtype,
            pp_layers=pp_layers,
            num_kv_heads_per_layer=num_kv_heads_per_layer,
            head_dim_per_layer=head_dim_per_layer,
            is_draft=is_draft,
        )
        return native.create_python_cold_page_codec(policy)

    def _create_cold_page_policy(
        self,
        cache_config: object,
        *,
        runtime_dtype: DataType,
        pp_layers: Sequence[int],
        num_kv_heads_per_layer: Sequence[int],
        head_dim_per_layer: Sequence[int],
        is_draft: bool = False,
    ) -> object:
        """Build the format-specific layout and callback policy."""
        raise NotImplementedError
