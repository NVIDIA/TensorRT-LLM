# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from abc import ABC, abstractmethod
from os import getenv
from typing import Any, Dict, List, Optional

import tensorrt_llm
from tensorrt_llm import logger
from tensorrt_llm._torch.distributed.communicator import Distributed
from tensorrt_llm.bindings import WorldConfig
from tensorrt_llm.llmapi.llm_args import CacheTransceiverConfig
from tensorrt_llm.mapping import Mapping

from .llm_request import LlmRequest
from .mamba_cache_manager import (BaseMambaCacheManager,
                                  CppMambaHybridCacheManager)
from .resource_manager import KVCacheManager

CacheTransceiverCpp = tensorrt_llm.bindings.internal.batch_manager.CacheTransceiver
AttentionTypeCpp = tensorrt_llm.bindings.internal.batch_manager.AttentionType
CacheTransBufferManagerCpp = tensorrt_llm.bindings.internal.batch_manager.CacheTransBufferManager
BackendTypeCpp = tensorrt_llm.bindings.executor.CacheTransceiverBackendType

KV_CACHE_TRANSFER_CHUNK_SIZE_BLOCKS_ENV_VAR_NAME = \
    "TRTLLM_KVCACHE_TRANSFER_CHUNK_SIZE_BLOCKS"
KV_CACHE_TRANSFER_EARLY_RELEASE_ENV_VAR_NAME = \
    "TRTLLM_KVCACHE_TRANSFER_EARLY_RELEASE"


def get_kv_cache_transfer_chunk_size_blocks() -> Optional[int]:
    value = getenv(KV_CACHE_TRANSFER_CHUNK_SIZE_BLOCKS_ENV_VAR_NAME)
    if value is None:
        return None

    if not value or not value.isascii() or not value.isdigit():
        raise ValueError(
            f"{KV_CACHE_TRANSFER_CHUNK_SIZE_BLOCKS_ENV_VAR_NAME} must be a "
            f"non-negative decimal integer, got {value!r}.")

    try:
        chunk_size_blocks = int(value)
    except ValueError as error:
        raise ValueError(
            f"{KV_CACHE_TRANSFER_CHUNK_SIZE_BLOCKS_ENV_VAR_NAME} must be a "
            f"non-negative decimal integer, got {value!r}.") from error
    if chunk_size_blocks > 2**31 - 1:
        raise ValueError(
            f"{KV_CACHE_TRANSFER_CHUNK_SIZE_BLOCKS_ENV_VAR_NAME} exceeds "
            f"the supported maximum ({2**31 - 1}), got {value!r}.")
    return chunk_size_blocks or None


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
    chunk_size_blocks = get_kv_cache_transfer_chunk_size_blocks()
    early_release = getenv(KV_CACHE_TRANSFER_EARLY_RELEASE_ENV_VAR_NAME) == "1"
    if early_release and chunk_size_blocks is None:
        raise ValueError(
            f"{KV_CACHE_TRANSFER_EARLY_RELEASE_ENV_VAR_NAME}=1 requires a "
            f"positive {KV_CACHE_TRANSFER_CHUNK_SIZE_BLOCKS_ENV_VAR_NAME}.")
    if cache_transceiver_config is None or cache_transceiver_config.backend is None:
        if chunk_size_blocks is not None or early_release:
            raise ValueError(
                "Chunked KV-cache transfer requires an enabled C++ KV cache "
                "transceiver.")
        logger.info("cache_transceiver is disabled")
        return None

    if (cache_transceiver_config.transceiver_runtime == "PYTHON"
            and (chunk_size_blocks is not None or early_release)):
        raise NotImplementedError(
            "Chunked KV-cache transfer and early release are supported only "
            "by the C++ KV cache transceiver; set "
            "cache_transceiver_config.transceiver_runtime='CPP'.")

    if cache_transceiver_config.backend == "DEFAULT":
        # When cache_transceiver_config.backend is not set, fallback to env_vars settings
        # NIXL is the default backend for non hybrid models
        cache_transceiver_config.backend = "NIXL"
        # Ordered by priority
        env_vars = [
            ("TRTLLM_USE_NIXL_KVCACHE", "NIXL"),
            ("TRTLLM_USE_UCX_KVCACHE", "UCX"),
            ("TRTLLM_USE_MOONCAKE_KVCACHE", "MOONCAKE"),
            ("TRTLLM_USE_MPI_KVCACHE", "MPI"),
        ]
        for env_var, be_type in env_vars:
            if getenv(env_var) == "1":
                logger.warning(
                    f"{env_var}=1 is set, but it's recommended to set cache_transceiver_config.backend in yaml config"
                )
                cache_transceiver_config.backend = be_type
                break

    if cache_transceiver_config.backend == "MPI":
        logger.warning(
            "MPI CacheTransceiver is deprecated, UCX or NIXL is recommended")
    elif cache_transceiver_config.backend == "UCX":
        logger.info(
            f"Using UCX kv-cache transceiver. If your devices are not in the same domain, please consider setting "
            f"UCX_CUDA_IPC_ENABLE_MNNVL=n, UCX_RNDV_SCHEME=put_zcopy and/or unset UCX_NET_DEVICES upon server "
            f"hangs or lower-than-expected performance.")

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
    def supports_cpp_chunked_transfer(self) -> bool:
        """Whether this implementation uses the env-gated C++ chunk protocol."""
        return False

    @property
    def transfer_chunk_size_blocks(self) -> Optional[int]:
        """Chunk size actually selected by the transceiver implementation."""
        return None

    @property
    def transfer_early_release_enabled(self) -> bool:
        """Whether this transceiver actually owns exact early-release leases."""
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

    def shutdown(self):
        """Shut down the transceiver and release registered resources."""


class BindKvCacheTransceiver(KvCacheTransceiver):

    @property
    def supports_cpp_chunked_transfer(self) -> bool:
        return True

    @property
    def transfer_chunk_size_blocks(self) -> Optional[int]:
        return self.impl.transfer_chunk_size_blocks

    @property
    def transfer_early_release_enabled(self) -> bool:
        return self.impl.transfer_early_release_enabled

    def __init__(self,
                 mapping: Mapping,
                 dist: Distributed,
                 kv_cache_manager: KVCacheManager,
                 attention_type: AttentionTypeCpp,
                 cache_transceiver_config: CacheTransceiverConfig,
                 mamba_cache_manager: Optional[BaseMambaCacheManager] = None):
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

        # Get RNN state manager and layer distribution if mamba_cache_manager is provided.
        rnn_state_manager = None
        rnn_layer_num_per_pp_rank = []
        if mamba_cache_manager is not None:
            if isinstance(mamba_cache_manager, CppMambaHybridCacheManager):
                # Unified pool path: RNN model config is in LinearAttentionMetadata,
                # C++ reads it from BlockManager during CacheTransceiver construction.
                rnn_layer_num_per_pp_rank = dist.pp_allgather(
                    mamba_cache_manager.local_num_mamba_layers)
            else:
                rnn_state_manager = mamba_cache_manager._impl.mamba_impl
                # Get the number of local RNN layers and allgather across PP ranks
                rnn_local_layer_num = rnn_state_manager.get_num_local_layers()
                rnn_layer_num_per_pp_rank = dist.pp_allgather(
                    rnn_local_layer_num)
                logger.info(
                    f"RNN state transfer enabled: rnn_layer_num_per_pp={rnn_layer_num_per_pp_rank}"
                )

        self.impl = CacheTransceiverCpp(
            kv_cache_manager.impl, total_num_kv_heads_per_layer, head_dim,
            tokens_per_block, world_config,
            pp_layer_num_per_pp_rank, dtype, attention_type,
            cache_transceiver_config._to_pybind(), rnn_state_manager,
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

    def shutdown(self):
        # Destroy the C++ transceiver while the KV-cache manager and its pools
        # are still alive. Its destructor joins workers and closes any active
        # transfer leases.
        if self.impl is not None:
            self.impl = None

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
