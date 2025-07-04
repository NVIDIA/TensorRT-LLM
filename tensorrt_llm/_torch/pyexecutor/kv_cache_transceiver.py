from abc import ABC, abstractmethod
from os import getenv

import tensorrt_llm
from tensorrt_llm.bindings import WorldConfig
from tensorrt_llm.bindings.executor import CacheTransceiverConfig
from tensorrt_llm.mapping import Mapping

from .llm_request import LlmRequest
from .resource_manager import KVCacheManager

CacheTransceiverCpp = tensorrt_llm.bindings.internal.batch_manager.CacheTransceiver
CommTypeCpp = tensorrt_llm.bindings.internal.batch_manager.CommType
AttentionTypeCpp = tensorrt_llm.bindings.internal.batch_manager.AttentionType
CacheTransBufferManagerCpp = tensorrt_llm.bindings.internal.batch_manager.CacheTransBufferManager


def mapping_to_world_config(mapping: Mapping) -> WorldConfig:

    return WorldConfig(tensor_parallelism=mapping.tp_size,
                       pipeline_parallelism=mapping.pp_size,
                       context_parallelism=mapping.cp_size,
                       rank=mapping.rank,
                       gpus_per_node=mapping.gpus_per_node,
                       device_ids=None,
                       enable_attention_dp=mapping.enable_attention_dp)


def create_kv_cache_transceiver(
        mapping: Mapping, kv_cache_manager: KVCacheManager,
        attention_type: AttentionTypeCpp,
        cache_transceiver_config: CacheTransceiverConfig):

    comm_type = None
    if getenv("TRTLLM_USE_UCX_KVCACHE"):
        comm_type = CommTypeCpp.UCX
    elif getenv("TRTLLM_USE_NIXL_KVCACHE"):
        comm_type = CommTypeCpp.NIXL
    elif getenv("TRTLLM_USE_MPI_KVCACHE"):
        comm_type = CommTypeCpp.MPI

    cache_transceiver = None
    if comm_type is not None:
        cache_transceiver = BindKvCacheTransceiver(mapping, comm_type,
                                                   kv_cache_manager,
                                                   attention_type,
                                                   cache_transceiver_config)

    return cache_transceiver


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


class BindKvCacheTransceiver(KvCacheTransceiver):

    def __init__(self, mapping: Mapping, comm_type: CommTypeCpp,
                 kv_cache_manager: KVCacheManager,
                 attention_type: AttentionTypeCpp,
                 cache_transceiver_config: CacheTransceiverConfig):
        world_config = mapping_to_world_config(mapping)
        num_kv_heads_per_layer = kv_cache_manager.num_kv_heads_per_layer
        head_dim = kv_cache_manager.head_dim
        tokens_per_block = kv_cache_manager.tokens_per_block
        dtype = kv_cache_manager.dtype

        self.impl = CacheTransceiverCpp(kv_cache_manager.impl, comm_type,
                                        num_kv_heads_per_layer, head_dim,
                                        tokens_per_block, world_config, dtype,
                                        attention_type,
                                        cache_transceiver_config)

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


class CacheTransBufferManager:

    def __init__(self, kv_cache_manager: KVCacheManager, max_num_tokens: int):
        self.impl = CacheTransBufferManagerCpp(kv_cache_manager.impl,
                                               max_num_tokens)

    @staticmethod
    def pre_alloc_buffer_size(max_num_tokens: int,
                              kv_cache_size_per_token: int):
        return CacheTransBufferManagerCpp.pre_alloc_buffer_size(
            max_num_tokens) * kv_cache_size_per_token
