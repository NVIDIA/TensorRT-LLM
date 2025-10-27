from abc import ABC, abstractmethod
from os import getenv

import tensorrt_llm
from tensorrt_llm import logger
from tensorrt_llm._torch.distributed.communicator import Distributed
from tensorrt_llm.bindings import WorldConfig
from tensorrt_llm.llmapi.llm_args import CacheTransceiverConfig
from tensorrt_llm.mapping import Mapping

from .llm_request import LlmRequest
from .resource_manager import KVCacheManager

CacheTransceiverCpp = tensorrt_llm.bindings.internal.batch_manager.CacheTransceiver
AttentionTypeCpp = tensorrt_llm.bindings.internal.batch_manager.AttentionType
CacheTransBufferManagerCpp = tensorrt_llm.bindings.internal.batch_manager.CacheTransBufferManager
BackendTypeCpp = tensorrt_llm.bindings.executor.CacheTransceiverBackendType


def mapping_to_world_config(mapping: Mapping) -> WorldConfig:

    return WorldConfig(tensor_parallelism=mapping.tp_size,
                       pipeline_parallelism=mapping.pp_size,
                       context_parallelism=mapping.cp_size,
                       rank=mapping.rank,
                       gpus_per_node=mapping.gpus_per_node,
                       device_ids=None,
                       enable_attention_dp=mapping.enable_attention_dp)


def create_kv_cache_transceiver(
        mapping: Mapping, dist: Distributed, kv_cache_manager: KVCacheManager,
        attention_type: AttentionTypeCpp,
        cache_transceiver_config: CacheTransceiverConfig):
    if cache_transceiver_config is None or cache_transceiver_config.backend is None:
        logger.info("cache_transceiver is disabled")
        return None

    if cache_transceiver_config.backend == "DEFAULT":
        # When cache_transceiver_config.backend is not set, fallback to env_vars settings
        # NIXL is the default backend
        cache_transceiver_config.backend = "NIXL"
        # Ordered by priority
        env_vars = [("TRTLLM_USE_UCX_KVCACHE", "UCX"),
                    ("TRTLLM_USE_MPI_KVCACHE", "MPI")]
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

    return BindKvCacheTransceiver(mapping, dist, kv_cache_manager,
                                  attention_type, cache_transceiver_config)


class KvCacheTransceiver(ABC):

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


class BindKvCacheTransceiver(KvCacheTransceiver):

    def __init__(self, mapping: Mapping, dist: Distributed,
                 kv_cache_manager: KVCacheManager,
                 attention_type: AttentionTypeCpp,
                 cache_transceiver_config: CacheTransceiverConfig):
        world_config = mapping_to_world_config(mapping)
        total_num_kv_heads_per_layer = kv_cache_manager.total_num_kv_heads_per_layer
        head_dim = kv_cache_manager.head_dim
        tokens_per_block = kv_cache_manager.tokens_per_block
        dtype = kv_cache_manager.dtype
        # get the layer num per pp rank, which is required by cache transceiver.
        pp_layer_num = len(kv_cache_manager.pp_layers)
        pp_layer_num_per_pp_rank = dist.pp_allgather(pp_layer_num)

        self.kv_transfer_timeout_ms = cache_transceiver_config.kv_transfer_timeout_ms
        self.impl = CacheTransceiverCpp(kv_cache_manager.impl,
                                        total_num_kv_heads_per_layer, head_dim,
                                        tokens_per_block, world_config,
                                        pp_layer_num_per_pp_rank, dtype,
                                        attention_type,
                                        cache_transceiver_config._to_pybind())

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
