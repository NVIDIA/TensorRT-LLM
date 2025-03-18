import math
from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Dict, List, Optional, Union

import torch

import tensorrt_llm
import tensorrt_llm.bindings
from tensorrt_llm.bindings.BuildInfo import ENABLE_MULTI_DEVICE
from tensorrt_llm.sampling_params import SamplingParams

from ..._utils import nvtx_range
from ...logger import logger
from ...mapping import Mapping
from .llm_request import LlmRequest
from .scheduler import ScheduledRequests

if ENABLE_MULTI_DEVICE:
    from mpi4py import MPI

    from tensorrt_llm._utils import mpi_comm

KVCacheManagerCpp = tensorrt_llm.bindings.internal.batch_manager.KVCacheManager
KvCacheConfigCpp = tensorrt_llm.bindings.KvCacheConfig
CacheTypeCpp = tensorrt_llm.bindings.internal.batch_manager.CacheType
ModelConfig = tensorrt_llm.bindings.ModelConfig
DataType = tensorrt_llm.bindings.DataType
KVCacheEventManagerCpp = tensorrt_llm.bindings.internal.batch_manager.KVCacheEventManager
RequestList = list[LlmRequest]


def compute_page_count(token_count: int, tokens_per_page: int) -> int:
    return (token_count + tokens_per_page) // tokens_per_page


class BaseResourceManager(ABC):

    @abstractmethod
    def get_max_resource_count(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def get_needed_resource_to_completion(self, request: LlmRequest) -> int:
        raise NotImplementedError

    def prepare_resources(self, scheduled_batch: ScheduledRequests):
        pass

    def update_resources(self, scheduled_batch: ScheduledRequests):
        pass

    def free_resources(self, request: LlmRequest):
        pass

    def shutdown(self):
        pass


class DummyKvCacheManager(BaseResourceManager):

    def __init__(self,
                 block_count: int,
                 max_num_tokens: int,
                 block_size: int = 64):
        super(BaseResourceManager, self).__init__()
        self.block_count = block_count
        self.max_num_tokens = max_num_tokens
        self.block_size = block_size

    def get_max_resource_count(self) -> int:
        return self.block_count

    def get_needed_resource_to_completion(self, request: LlmRequest) -> int:
        max_new_tokens = request.max_new_tokens if request.max_new_tokens is not None else self.max_num_tokens - request.orig_prompt_len
        context_token_count = request.orig_prompt_len
        num_context_blocks = context_token_count // self.block_size
        remaining_tokens = context_token_count + max_new_tokens - num_context_blocks * self.block_size
        need_blocks = num_context_blocks + math.ceil(
            remaining_tokens / self.block_size)
        return need_blocks


class KVCacheManager(BaseResourceManager):

    def __init__(
        self,
        kv_cache_config: KvCacheConfigCpp,
        kv_cache_type: CacheTypeCpp,
        *,
        num_layers: int,
        num_heads: int,
        num_kv_heads: Union[int, List[Optional[int]]],
        head_dim: int,
        tokens_per_block: int,
        # Note that max_seq_len is not necessarily equal to kv_cache_config.num_tokens.
        # It's derived from the model's BuildConfig for consistency with the C++ backend.
        max_seq_len: int,
        max_batch_size: int,
        mapping: Mapping,
        dtype: DataType = DataType.HALF,
        # Some speculative decoding methods need to use different kv lengths for the
        # draft/target layers. Add extra tokens to haddle this issue.
        num_extra_kv_tokens: int = 0,
    ) -> None:
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.mapping = mapping
        self.dtype = dtype
        self.kv_cache_type = kv_cache_type

        tp_size = mapping.tp_size
        if mapping.enable_attention_dp:
            tp_size = 1

        if isinstance(num_kv_heads, int):
            self.num_kv_heads_per_layer = [
                (num_kv_heads + tp_size - 1) // tp_size
                for _ in range(num_layers)
            ]

        else:
            assert len(num_kv_heads) == self.num_layers

            self.num_kv_heads_per_layer = []
            for layer_idx, kv_head in enumerate(num_kv_heads):
                if kv_head is not None:
                    self.num_kv_heads_per_layer.append(
                        (kv_head + tp_size - 1) // tp_size)
                else:
                    self.num_kv_heads_per_layer.append(0)

        assert len(self.num_kv_heads_per_layer) > 0

        self.is_homongenous = all(val == self.num_kv_heads_per_layer[0]
                                  for val in self.num_kv_heads_per_layer[1:])

        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.tokens_per_block = tokens_per_block
        self.max_seq_len = max_seq_len
        self.max_batch_size = max_batch_size
        self.kv_factor = 1 if kv_cache_type == CacheTypeCpp.SELFKONLY else 2
        self.num_extra_kv_tokens = num_extra_kv_tokens
        self.event_buffer_max_size = kv_cache_config.event_buffer_max_size

        if kv_cache_config.max_attention_window is None:
            max_attention_window = max_seq_len
        else:
            max_attention_window = max(kv_cache_config.max_attention_window)

        sink_token_length = (kv_cache_config.sink_token_length
                             if kv_cache_config.sink_token_length is not None
                             else 0)

        self.blocks_in_primary_pool, self.blocks_in_secondary_pool = self.calculate_max_num_blocks(
            kv_cache_config,
            head_dim=head_dim,
            tokens_per_block=tokens_per_block,
            mapping=mapping,
            dtype=dtype,
            kv_factor=self.kv_factor,
        )

        max_atten_window_upper_bound = self.get_max_atten_window_upper_bound(
            blocks_in_primary_pool=self.blocks_in_primary_pool,
            tokens_per_block=tokens_per_block,
            max_beam_width=1,
            sink_token_len=sink_token_length,
            max_seq_len=max_seq_len)

        if max_attention_window > max_atten_window_upper_bound:
            logger.warning(
                f"maxAttentionWindow and maxSequenceLen are too large for at least one sequence to fit in kvCache. They are reduced to {max_atten_window_upper_bound}"
            )
            max_attention_window = max_atten_window_upper_bound
            self.max_seq_len = max_atten_window_upper_bound

        max_kv_cache_len = (max_attention_window if kv_cache_type
                            == CacheTypeCpp.SELF else self.max_seq_len)

        # Note that this stream is unused for now. Will be used for copying to host
        # when that feature is enabled.
        self._stream = torch.cuda.Stream()
        kwargs = {
            'num_kv_heads_per_layer': self.num_kv_heads_per_layer,
            'size_per_head': head_dim,
            'tokens_per_block': tokens_per_block,
            'blocks_in_primary_pool': self.blocks_in_primary_pool,
            'blocks_in_secondary_pool': self.blocks_in_secondary_pool,
            'max_num_sequences': max_batch_size,
            'max_beam_width': 1,  # TODO: more than 1 beam?
            'max_attention_window': max_kv_cache_len,
            'temporary_attention_window': 0,
            'sink_token_length': sink_token_length,
            'stream': self._stream.cuda_stream,
            'max_sequence_length': max_seq_len,
            'enable_block_reuse': kv_cache_config.enable_block_reuse,
            'onboard_blocks': kv_cache_config.onboard_blocks,
            'cache_type': kv_cache_type,
            'enable_partial_reuse': kv_cache_config.enable_partial_reuse,
            'copy_on_partial_reuse': kv_cache_config.copy_on_partial_reuse,
        }
        if self.event_buffer_max_size > 0:
            kwargs['event_manager'] = KVCacheEventManagerCpp(
                max_kv_event_entries=self.event_buffer_max_size)

        self.impl = KVCacheManagerCpp(**kwargs)

        self.impl.allocate_pools(dtype, False)
        self.kv_cache_pool_pointers = self.impl.get_block_pool_pointers()
        self.kv_cache_pool_mapping = self.impl.get_layer_to_pool_mapping()
        self.num_pools = self.impl.num_pools
        self.max_blocks_per_seq = self.impl.max_blocks_per_seq

        self.max_num_slots = max_batch_size

    def shutdown(self):
        self.impl.release_pools()

    @classmethod
    def from_model_config(cls,
                          model_config: ModelConfig,
                          kv_cache_config: KvCacheConfigCpp,
                          mapping: Mapping,
                          kv_cache_type: CacheTypeCpp = CacheTypeCpp.SELF,
                          dtype: DataType = DataType.HALF) -> "KVCacheManager":
        return cls(
            kv_cache_config,
            kv_cache_type,
            num_layers=model_config.num_attention_layers(mapping.pp_size),
            num_heads=model_config.num_heads,
            # NOTE: this preserves existing behavior in KV cache manager.
            # But we should change this to pass a list at some point.
            # We're assuming the KV cache is homogeneous here.
            num_kv_heads=model_config.num_kv_heads(0),
            head_dim=model_config.size_per_head,
            tokens_per_block=model_config.tokens_per_block,
            max_seq_len=model_config.max_seq_len,
            max_batch_size=model_config.max_batch_size,
            mapping=mapping,
            dtype=dtype)

    def get_max_resource_count(self) -> int:
        return self.impl.max_num_blocks

    def get_needed_resource_to_completion(self, request: LlmRequest) -> int:
        # TODO: the C++ implementation of this method can be used, but the
        # Python and C++ schedulers currently do not agree on what "needed
        # resource to completion" means. The C++ one excludes already allocated
        # blocks; the Python one includes them. This should be unified, but
        # the Python scheduler needs to be fixed.
        #
        # return self.impl.get_remaining_blocks_to_completion(request)
        context_token_count = request.orig_prompt_len
        num_context_blocks = context_token_count // self.tokens_per_block
        remaining_tokens = context_token_count + request.max_new_tokens - num_context_blocks * self.tokens_per_block
        need_blocks = num_context_blocks + math.ceil(
            remaining_tokens / self.tokens_per_block)
        return need_blocks

    def prepare_resources(self, scheduled_batch: ScheduledRequests):
        context_batch = scheduled_batch.context_requests
        generation_batch = scheduled_batch.generation_requests
        # allocate KV Cache
        for req in context_batch:
            req_beam_width = 1  # req.sampling_config.beam_width
            if 'cp_type' in self.mapping.cp_config and 'star_attention' == self.mapping.cp_config[
                    'cp_type']:
                if req.ctx_iters == 0:
                    seq_len = sum(
                        len(ctx_block) for ctx_block in req.ctx_blocks)
                    self.impl.add_sequence(
                        req.py_request_id,
                        seq_len + (len(req.query_id) if self.mapping.cp_rank
                                   == self.mapping.cp_size - 1 else 0),
                        req_beam_width, req)
            else:
                if req.is_first_context_chunk():
                    self.impl.add_sequence(req.py_request_id, req.prompt_len,
                                           req_beam_width, req)
                    for _ in range(self.num_extra_kv_tokens):
                        self.impl.add_token(req.py_request_id)
                    if req.py_draft_tokens is not None:
                        for _ in range(len(req.py_draft_tokens)):
                            self.impl.add_token(req.py_request_id)

        for req in generation_batch:
            self.impl.add_token(req.py_request_id)
            if req.py_draft_tokens is not None:
                for _ in range(len(req.py_draft_tokens)):
                    self.impl.add_token(req.py_request_id)

    def add_dummy_requests(
        self,
        request_ids: List[int],
        token_nums: Optional[List[int]] = None,
        is_gen: bool = False,
        max_num_draft_tokens: int = 0,
    ):
        beam_width = 1
        requests = []
        for i, req_id in enumerate(request_ids):
            sampling_params = SamplingParams()
            token_num = token_nums[
                i] if token_nums is not None else 1 + max_num_draft_tokens
            encoder_input_tokens = [
                1
            ] * token_num if self.impl.cross_kv else None
            # Using 1 instead of 0 prevents NaN during warmup in e.g. Deepseek
            req = LlmRequest(
                request_id=req_id,
                max_new_tokens=1,
                input_tokens=[1] * token_num,
                sampling_config=tensorrt_llm.bindings.SamplingConfig(
                    sampling_params._get_sampling_config()),
                is_streaming=False,
                encoder_input_tokens=encoder_input_tokens)
            req.paged_kv_block_ids = []
            self.impl.add_sequence(req_id, token_num, beam_width, req)
            if is_gen:
                req.state = tensorrt_llm.bindings.LlmRequestState.GENERATION_IN_PROGRESS
                req.prompt_len = token_num - 1 + max_num_draft_tokens
                req.py_prompt_len = req.prompt_len
                if max_num_draft_tokens > 0:
                    req.py_draft_tokens = [0] * max_num_draft_tokens
            requests.append(req)
        return requests

    def update_resources(self, scheduled_batch: ScheduledRequests):
        # rewind kv cache
        for request in scheduled_batch.generation_requests:
            if request.state != tensorrt_llm.bindings.LlmRequestState.GENERATION_COMPLETE:
                if request.py_rewind_len > 0:
                    self.rewind_kv_cache(request, request.py_rewind_len)

    def free_resources(self, request: LlmRequest):
        self.impl.remove_sequence(request.py_request_id, request)

    def calculate_max_num_blocks(self,
                                 kv_cache_config: KvCacheConfigCpp,
                                 head_dim: int,
                                 tokens_per_block: int,
                                 mapping: Mapping,
                                 dtype: DataType,
                                 kv_factor: int = 2):
        free_mem_fraction = (kv_cache_config.free_gpu_memory_fraction
                             if kv_cache_config.free_gpu_memory_fraction
                             is not None else 0.9)

        cache_size_per_token = kv_factor * sum(
            self.num_kv_heads_per_layer) * head_dim

        if dtype == DataType.FP8:
            kv_cache_dtype_bytes = 1
        elif dtype in (DataType.HALF, DataType.BF16):
            kv_cache_dtype_bytes = 2
        elif dtype == DataType.FLOAT:
            kv_cache_dtype_bytes = 4
        else:
            raise ValueError(f'Cannot support {dtype} KV cache.')

        cache_size_bytes_per_token = cache_size_per_token * kv_cache_dtype_bytes
        free_mem, total_mem = torch.cuda.mem_get_info()

        assert free_mem_fraction < 1.0, f"Invalid freeMemFraction, freeMemFraction {free_mem_fraction} must be smaller than 1.0"
        max_tokens = free_mem_fraction * free_mem / cache_size_bytes_per_token

        # If user specified a number of tokens
        if kv_cache_config.max_tokens is not None:
            # If user also specified a free gpu memory fraction, take the min
            if kv_cache_config.free_gpu_memory_fraction is not None:
                max_tokens = min(kv_cache_config.max_tokens, max_tokens)
                logger.warning(
                    f'Both free_gpu_memory_fraction and max_tokens are set (to {free_mem_fraction} and {kv_cache_config.max_tokens}, respectively). The smaller value will be used.'
                )
            else:
                max_tokens = kv_cache_config.max_tokens

        if mapping.world_size > 1:
            # make sure all ranks use same value for maxTokens
            max_tokens = mpi_comm().allreduce(max_tokens, op=MPI.MIN)

        # get number of blocks
        blocks_in_primary_pool = math.ceil(max_tokens / tokens_per_block)
        host_cache_size = kv_cache_config.host_cache_size if kv_cache_config.host_cache_size else 0
        max_tokens_secondary = host_cache_size / cache_size_bytes_per_token
        blocks_in_secondary_pool = max(
            0, int(max_tokens_secondary / tokens_per_block))
        return blocks_in_primary_pool, blocks_in_secondary_pool

    def get_max_atten_window_upper_bound(self, blocks_in_primary_pool,
                                         tokens_per_block, max_beam_width,
                                         sink_token_len,
                                         max_seq_len: Optional[int]):
        token_capacity = blocks_in_primary_pool * tokens_per_block
        max_blocks_per_seq = math.floor(token_capacity /
                                        (max_beam_width * tokens_per_block))
        assert max_blocks_per_seq > 0, "Impossibe to fit in any sequence in kvCache"

        max_token_num = max_blocks_per_seq * tokens_per_block
        sink_tokens_in_last_block = sink_token_len % tokens_per_block
        sink_bubble_len = 0 if sink_tokens_in_last_block == 0 else tokens_per_block - sink_tokens_in_last_block
        max_atten_window_upper_bound = max_token_num - sink_bubble_len
        if max_seq_len is not None and max_seq_len > max_atten_window_upper_bound and max_beam_width > 1:
            max_atten_window_upper_bound -= tokens_per_block
        assert max_atten_window_upper_bound > 0, "Impossibe to fit in any sequence in kvCache"
        return max_atten_window_upper_bound

    def get_cache_indices(self, request: LlmRequest) -> List[int]:
        result = self.impl.get_cache_block_ids(request.py_request_id)
        assert len(result) == 1
        return result[0]

    def get_batch_cache_indices(
        self,
        request_ids: List[int],
    ) -> Dict[int, List[int]]:
        result = self.impl.get_batch_cache_block_ids(request_ids)
        for i in range(len(result)):
            assert (len(result[i])) == 1
            result[i] = result[i][0]
        return result

    def get_num_free_blocks(self) -> int:
        return self.impl.get_kv_cache_stats().free_num_blocks

    def get_num_kv_blocks(self, num_tokens: int) -> int:
        return (num_tokens + self.tokens_per_block - 1) // self.tokens_per_block

    def get_buffers(self, layer_idx: int) -> Optional[torch.Tensor]:
        result = self.impl.get_primary_pool_data(layer_idx)
        return result.reshape(
            result.shape[0],
            self.kv_factor,
            self.tokens_per_block,
            self.num_kv_heads_per_layer[layer_idx],
            self.head_dim,
        )

    def flush_iteration_events(self):
        self.impl.flush_iteration_events()

    def get_latest_events(self, timeout_ms: Optional[float] = 0):
        return self.impl.get_latest_events(timeout_ms)

    def get_kv_cache_stats(self):
        return self.impl.get_kv_cache_stats()

    def rewind_kv_cache(self, request: LlmRequest, rewind_len: int):
        self.impl.rewind_kv_cache(request.py_request_id, rewind_len)


class BaseDraftTokenManager(BaseResourceManager):

    @abstractmethod
    def get_draft_tokens(self,
                         input_token_ids: List[List[int]]) -> List[List[int]]:
        """
        This method is intended to take a sequence of token ids (prompt + decoded so far)
        and produce draft tokens for each request. We should have
        len(get_draft_tokens(tokens)) == len(tokens), but each request's list of draft tokens
        may be arbitrarily long.

        You can produce the draft tokens in any manner that you want.
        """

    def prepare_resources(self, scheduled_batch: ScheduledRequests) -> None:
        input_tokens = []
        for request in scheduled_batch.generation_requests:
            input_tokens.append(request.get_tokens(0))

        if not input_tokens:
            return

        results = self.get_draft_tokens(input_tokens)
        for request, output in zip(scheduled_batch.generation_requests,
                                   results):
            request.py_draft_tokens = output

    def get_max_resource_count(self) -> int:
        return 0

    def get_needed_resource_to_completion(self, request: LlmRequest) -> int:
        return 0


class ResourceManager(object):

    def __init__(self, resource_managers: dict[str, BaseResourceManager]):
        self.resource_managers = OrderedDict(resource_managers)

    def __call__(self, name: str):
        return self.resource_managers[name]

    def register_resource_manager(self, name: str,
                                  resource_manager: BaseResourceManager):
        self.resource_managers[name] = resource_manager

    def get_resource_manager(self, name: str) -> BaseResourceManager:
        return self.resource_managers.get(name)

    @nvtx_range("prepare_resources")
    def prepare_resources(self, scheduled_batch: ScheduledRequests):
        for _, resource_manager in self.resource_managers.items():
            if hasattr(resource_manager, "prepare_resources"):
                resource_manager.prepare_resources(scheduled_batch)

    @nvtx_range("update_resources")
    def update_resources(self, scheduled_batch: ScheduledRequests):
        for _, resource_manager in self.resource_managers.items():
            if hasattr(resource_manager, "update_resources"):
                resource_manager.update_resources(scheduled_batch)

    def free_resources(self, request: LlmRequest):
        for _, resource_manager in reversed(self.resource_managers.items()):
            if hasattr(resource_manager, "free_resources"):
                resource_manager.free_resources(request)

    def reorder_pipeline(self, resource_manager_list: list[str]):
        assert set(resource_manager_list) == set(self.resource_managers.keys())
        for resource_manager in resource_manager_list:
            self.resource_managers.move_to_end(resource_manager)
