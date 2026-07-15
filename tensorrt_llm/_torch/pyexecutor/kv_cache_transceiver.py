# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod
from os import getenv
from typing import Any, Dict, List, Optional

import tensorrt_llm
from tensorrt_llm import logger
from tensorrt_llm._torch.distributed.communicator import Distributed
from tensorrt_llm.bindings import WorldConfig
from tensorrt_llm.llmapi.llm_args import (_CACHE_TRANSCEIVER_BACKEND_ENV_VARS,
                                          CacheTransceiverConfig)
from tensorrt_llm.mapping import Mapping

from .llm_request import LlmRequest
from .mamba_cache_manager import (BaseMambaCacheManager,
                                  CppMambaHybridCacheManager,
                                  MixedMambaHybridCacheManager)
from .resource_manager import KVCacheManager

CacheTransceiverCpp = tensorrt_llm.bindings.internal.batch_manager.CacheTransceiver
AttentionTypeCpp = tensorrt_llm.bindings.internal.batch_manager.AttentionType
CacheTransBufferManagerCpp = tensorrt_llm.bindings.internal.batch_manager.CacheTransBufferManager
BackendTypeCpp = tensorrt_llm.bindings.executor.CacheTransceiverBackendType

_DISAGG_INFLIGHT_CANCEL_ENABLED_ENV = "TRTLLM_DISAGG_ENABLE_INFLIGHT_CANCEL"
_NIXL_KVCACHE_BACKEND_ENV = "TRTLLM_NIXL_KVCACHE_BACKEND"
_DISABLE_KV_CACHE_TRANSFER_OVERLAP_ENV = "TRTLLM_DISABLE_KV_CACHE_TRANSFER_OVERLAP"
_DISAGG_LAYERWISE_ENV = "TRTLLM_DISAGG_LAYERWISE"
_TRY_ZCOPY_FOR_KV_CACHE_TRANSFER_ENV = "TRTLLM_TRY_ZCOPY_FOR_KVCACHE_TRANSFER"
_SUPPORTED_INFLIGHT_CANCEL_NIXL_BACKEND = "UCX"
_disagg_inflight_cancel_enabled_cache: Optional[bool] = None


def is_disagg_inflight_cancel_enabled() -> bool:
    """Return whether disaggregated in-flight KV transfer cancellation is enabled."""
    global _disagg_inflight_cancel_enabled_cache
    if _disagg_inflight_cancel_enabled_cache is None:
        _disagg_inflight_cancel_enabled_cache = (getenv(
            _DISAGG_INFLIGHT_CANCEL_ENABLED_ENV, "0") == "1")
        if _disagg_inflight_cancel_enabled_cache:
            logger.warning(
                f"{_DISAGG_INFLIGHT_CANCEL_ENABLED_ENV}=1: disagg KV "
                "transfer in-flight cancellation was requested. It is active "
                "only for cache transceivers that advertise support.")
    return _disagg_inflight_cancel_enabled_cache


def _is_disagg_inflight_cancel_config_supported(
        cache_transceiver_config: CacheTransceiverConfig) -> bool:
    runtime = cache_transceiver_config.transceiver_runtime or "CPP"
    nixl_backend = getenv(_NIXL_KVCACHE_BACKEND_ENV,
                          _SUPPORTED_INFLIGHT_CANCEL_NIXL_BACKEND)
    return (runtime == "CPP" and cache_transceiver_config.backend == "NIXL"
            and nixl_backend == _SUPPORTED_INFLIGHT_CANCEL_NIXL_BACKEND
            and cache_transceiver_config.kv_transfer_timeout_ms is not None
            and getenv(_DISABLE_KV_CACHE_TRANSFER_OVERLAP_ENV) != "1"
            and getenv(_DISAGG_LAYERWISE_ENV) != "1"
            and getenv(_TRY_ZCOPY_FOR_KV_CACHE_TRANSFER_ENV) != "1")


def _validate_disagg_inflight_cancel_config(
        cache_transceiver_config: CacheTransceiverConfig) -> None:
    if not is_disagg_inflight_cancel_enabled():
        return

    enabled_backend_env_vars = [
        env_name for env_name, _ in _CACHE_TRANSCEIVER_BACKEND_ENV_VARS
        if getenv(env_name) == "1"
    ]
    if (cache_transceiver_config.backend == "DEFAULT"
            and len(enabled_backend_env_vars) > 1):
        raise ValueError(
            f"{_DISAGG_INFLIGHT_CANCEL_ENABLED_ENV}=1 requires an "
            "unambiguous cache transceiver backend, but multiple legacy "
            f"backend selectors are enabled: {enabled_backend_env_vars}.")

    if _is_disagg_inflight_cancel_config_supported(cache_transceiver_config):
        return

    runtime = cache_transceiver_config.transceiver_runtime or "CPP"
    nixl_backend = getenv(_NIXL_KVCACHE_BACKEND_ENV,
                          _SUPPORTED_INFLIGHT_CANCEL_NIXL_BACKEND)
    disable_overlap = getenv(_DISABLE_KV_CACHE_TRANSFER_OVERLAP_ENV)
    layerwise = getenv(_DISAGG_LAYERWISE_ENV)
    try_zcopy = getenv(_TRY_ZCOPY_FOR_KV_CACHE_TRANSFER_ENV)
    raise ValueError(
        f"{_DISAGG_INFLIGHT_CANCEL_ENABLED_ENV}=1 is experimental and "
        "currently supported only with transceiver_runtime='CPP', "
        "backend='NIXL', the UCX NIXL backend, a finite "
        "kv_transfer_timeout_ms, asynchronous non-layer-wise transfer, and "
        "zero-copy disabled; got "
        f"transceiver_runtime={runtime!r}, "
        f"backend={cache_transceiver_config.backend!r}, "
        f"resolved_nixl_backend={nixl_backend!r}, "
        f"kv_transfer_timeout_ms={cache_transceiver_config.kv_transfer_timeout_ms!r}, "
        f"disable_transfer_overlap={disable_overlap!r}, "
        f"layerwise={layerwise!r}, try_zcopy={try_zcopy!r}.")


def mapping_to_world_config(mapping: Mapping) -> WorldConfig:

    return WorldConfig(tensor_parallelism=mapping.tp_size,
                       pipeline_parallelism=mapping.pp_size,
                       context_parallelism=mapping.cp_size,
                       rank=mapping.rank,
                       gpus_per_node=mapping.gpus_per_node,
                       device_ids=None,
                       enable_attention_dp=mapping.enable_attention_dp)


def create_kv_cache_transceiver(
        mapping: Mapping,
        dist: Distributed,
        kv_cache_manager: KVCacheManager,
        attention_type: AttentionTypeCpp,
        cache_transceiver_config: CacheTransceiverConfig,
        mamba_cache_manager: Optional[BaseMambaCacheManager] = None):
    if cache_transceiver_config is None or cache_transceiver_config.backend is None:
        logger.info("cache_transceiver is disabled")
        return None

    # "auto" is normally resolved against the model's preference at config
    # load time (ModelLoader.load_config_and_apply_defaults); paths that skip
    # that step (e.g. AutoDeploy) fall back to the C++ transceiver here. This
    # must run before any consumer of transceiver_runtime below (e.g. the
    # inflight-cancel validation, which treats non-CPP runtimes as
    # unsupported).
    if cache_transceiver_config.transceiver_runtime == "auto":
        cache_transceiver_config.transceiver_runtime = None

    if (cache_transceiver_config.transceiver_runtime != "PYTHON"
            and isinstance(mamba_cache_manager, MixedMambaHybridCacheManager)):
        raise ValueError(
            "MixedMambaHybridCacheManager requires the Python transceiver "
            "runtime in disaggregated serving.")

    _validate_disagg_inflight_cancel_config(cache_transceiver_config)

    if cache_transceiver_config.backend == "DEFAULT":
        # When cache_transceiver_config.backend is not set, fallback to env_vars settings
        # NIXL is the default backend for non hybrid models
        backend, env_var = cache_transceiver_config._resolve_default_backend()
        if env_var is not None:
            logger.warning(
                f"{env_var}=1 is set, but it's recommended to set cache_transceiver_config.backend in yaml config"
            )
        cache_transceiver_config.backend = backend

    if cache_transceiver_config.backend == "MPI":
        logger.warning(
            "MPI CacheTransceiver is deprecated, UCX or NIXL is recommended")
    elif cache_transceiver_config.backend == "UCX":
        logger.info(
            "Using UCX kv-cache transceiver. If your devices are not in the same domain, please consider setting "
            "UCX_CUDA_IPC_ENABLE_MNNVL=n, UCX_RNDV_SCHEME=put_zcopy and/or unset UCX_NET_DEVICES upon server "
            "hangs or lower-than-expected performance.")

    # Select transceiver implementation based on transceiver_runtime
    # transceiver_runtime == None or "CPP" -> use C++ transceiver (default)
    # transceiver_runtime == "PYTHON" -> use Python transceiver
    if cache_transceiver_config.transceiver_runtime == "PYTHON":
        # Python transceiver currently only supports NIXL and DEFAULT backend
        if cache_transceiver_config.backend not in ("DEFAULT", "NIXL"):
            raise ValueError(
                f"Python transceiver currently only supports NIXL or DEFAULT backend, "
                f"got {cache_transceiver_config.backend}. "
                f"Please use transceiver_runtime='CPP' for MPI, UCX, or MOONCAKE backends."
            )
        from tensorrt_llm._torch.disaggregation.transceiver import \
            KvCacheTransceiverV2
        logger.info("Using KvCacheTransceiverV2")
        return KvCacheTransceiverV2(mapping, dist, kv_cache_manager,
                                    cache_transceiver_config)

    # Default: use C++ transceiver (transceiver_runtime is None or "CPP")
    return BindKvCacheTransceiver(mapping, dist, kv_cache_manager,
                                  attention_type, cache_transceiver_config,
                                  mamba_cache_manager)


class KvCacheTransceiver(ABC):

    @property
    def requires_physical_drain_before_request_release(self) -> bool:
        """Whether request resources must remain owned until cancel drains."""
        return False

    @abstractmethod
    def respond_and_send_async(self, req: LlmRequest):
        raise NotImplementedError

    @abstractmethod
    def request_and_receive_sync(self, req: LlmRequest):
        raise NotImplementedError

    @abstractmethod
    def request_and_receive_async(self, req: LlmRequest):
        raise NotImplementedError

    @abstractmethod
    def check_context_transfer_status(self, at_least_request_num: int):
        raise NotImplementedError

    @abstractmethod
    def check_gen_transfer_status(self, at_least_request_num: int):
        raise NotImplementedError

    @abstractmethod
    def check_gen_transfer_complete(self):
        raise NotImplementedError

    @abstractmethod
    def cancel_request(self, req: LlmRequest):
        raise NotImplementedError

    def supports_inflight_request_cancellation(self) -> bool:
        return False

    def has_poisoned_transfer_buffer(self) -> bool:
        return False

    @abstractmethod
    def prepare_context_requests(self, requests: List[LlmRequest]):
        """
        Prepare the context request for the cache transceiver in generation-first mode.
        This method should set the context request state to DISAGG_CONTEXT_WAIT_SCHEDULER
        so that it won't be scheduled if the responding generation kvcache request is not
        yet received otherwise set it to CONTEXT_INIT.
        """
        ...

    @abstractmethod
    def get_disaggregated_params(self) -> Dict[str, Any]:
        """
        Return a dictionary form of DisaggregatedParams to be set in the generation request.
        The generation server will use it to get kvcache in generation-first mode.
        """
        ...

    def commit_blocks_for_reuse(self, req: LlmRequest) -> None:
        """Commit received KV blocks to the radix tree for prefix reuse. No-op by default."""

    def shutdown(self) -> Optional[bool]:
        """Shut down the transceiver and release registered resources.

        Returns:
            Lifecycle-capable implementations return ``True`` only after a
            complete drain and ``False`` while resources remain owned. Legacy
            implementations may return ``None``; callers must not interpret it
            as transport-quiescence evidence.
        """


class BindKvCacheTransceiver(KvCacheTransceiver):

    def __init__(self,
                 mapping: Mapping,
                 dist: Distributed,
                 kv_cache_manager: KVCacheManager,
                 attention_type: AttentionTypeCpp,
                 cache_transceiver_config: CacheTransceiverConfig,
                 mamba_cache_manager: Optional[BaseMambaCacheManager] = None):
        _validate_disagg_inflight_cancel_config(cache_transceiver_config)
        world_config = mapping_to_world_config(mapping)
        # Filter out mamba/recurrent state layers (kv_heads == 0) so that
        # CacheState::ModelConfig::mNbKvHeadsPerLayer only contains attention
        # layers — matching the factory path (modelConfig.getNumKvHeadsPerLayer()).
        # This is critical: splitKVCacheDispatch uses mNbKvHeadsPerLayer.size()
        # as the layer count for the CUDA kernel grid dimension.
        total_num_kv_heads_per_layer = [
            h for h in kv_cache_manager.total_num_kv_heads_per_layer if h > 0
        ]
        head_dim = kv_cache_manager.head_dim
        tokens_per_block = kv_cache_manager.tokens_per_block
        dtype = kv_cache_manager.dtype
        # Get the *attention* layer count per PP rank (C++ uses this as
        # mAttentionLayerNumPerPP).  For CppMambaHybridCacheManager the local
        # pp_layers list includes mamba layers (kv_heads == 0); those must be
        # excluded so the C++ buffer-size calculations stay correct.
        pp_layer_num = sum(1 for h in kv_cache_manager.num_kv_heads_per_layer
                           if h > 0)
        pp_layer_num_per_pp_rank = dist.pp_allgather(pp_layer_num)

        self.kv_transfer_timeout_ms = cache_transceiver_config.kv_transfer_timeout_ms
        self.kv_transfer_sender_future_timeout_ms = cache_transceiver_config.kv_transfer_sender_future_timeout_ms
        self.kv_transfer_poll_interval_ms = cache_transceiver_config.kv_transfer_poll_interval_ms
        self._supports_inflight_request_cancellation = (
            _is_disagg_inflight_cancel_config_supported(
                cache_transceiver_config))

        # Get RNN layer distribution if mamba_cache_manager is provided.
        rnn_layer_num_per_pp_rank = []
        if mamba_cache_manager is not None:
            if isinstance(mamba_cache_manager, CppMambaHybridCacheManager):
                # Unified pool path: RNN model config is in LinearAttentionMetadata,
                # C++ reads it from BlockManager during CacheTransceiver construction.
                rnn_layer_num_per_pp_rank = dist.pp_allgather(
                    mamba_cache_manager.local_num_mamba_layers)
            else:
                # MixedMambaHybridCacheManager with PythonMambaCacheManager.
                rnn_layer_num_per_pp_rank = dist.pp_allgather(
                    len(mamba_cache_manager._impl.mamba_layer_offsets))
                logger.info(
                    f"RNN state transfer enabled: rnn_layer_num_per_pp={rnn_layer_num_per_pp_rank}"
                )

        self.impl = CacheTransceiverCpp(kv_cache_manager.impl,
                                        total_num_kv_heads_per_layer, head_dim,
                                        tokens_per_block, world_config,
                                        pp_layer_num_per_pp_rank, dtype,
                                        attention_type,
                                        cache_transceiver_config._to_pybind(),
                                        rnn_layer_num_per_pp_rank)

    def respond_and_send_async(self, req: LlmRequest):
        return self.impl.respond_and_send_async(req)

    def request_and_receive_sync(self, req: LlmRequest):
        return self.impl.request_and_receive_sync(req)

    def request_and_receive_async(self, req: LlmRequest):
        return self.impl.request_and_receive_async(req)

    def check_context_transfer_status(self, at_least_request_num: int):
        return self.impl.check_context_transfer_status(at_least_request_num)

    def check_gen_transfer_status(self, at_least_request_num: int):
        return self.impl.check_gen_transfer_status(at_least_request_num)

    def check_gen_transfer_complete(self):
        return self.impl.check_gen_transfer_complete()

    def cancel_request(self, req: LlmRequest):
        return self.impl.cancel_request(req)

    def supports_inflight_request_cancellation(self) -> bool:
        return self._supports_inflight_request_cancellation

    def has_poisoned_transfer_buffer(self) -> bool:
        if not is_disagg_inflight_cancel_enabled():
            return False
        return self.impl.has_poisoned_transfer_buffer()

    def prepare_context_requests(self, requests: List[LlmRequest]):
        # not implemented, an empty placeholder to allow being invoked unconditionally
        ...

    def get_disaggregated_params(self):
        # Cpp kv cache transceiver will set the disaggregated params to context response
        # Only new py cache transceiver will support gen-first disagg
        return {}


class CacheTransBufferManager:

    def __init__(self, kv_cache_manager: KVCacheManager, max_num_tokens: int):
        self.impl = CacheTransBufferManagerCpp(kv_cache_manager.impl,
                                               max_num_tokens)

    @staticmethod
    def pre_alloc_buffer_size(
            kv_cache_size_bytes_per_token_per_window: dict[int, int],
            tokens_per_block: int,
            cache_transceiver_config: CacheTransceiverConfig):
        return CacheTransBufferManagerCpp.pre_alloc_buffer_size(
            kv_cache_size_bytes_per_token_per_window, tokens_per_block,
            cache_transceiver_config)
