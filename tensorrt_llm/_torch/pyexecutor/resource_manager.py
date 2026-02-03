import copy
import enum
import math
from abc import ABC, abstractmethod
from collections import OrderedDict, defaultdict, deque
from typing import (TYPE_CHECKING, Dict, Iterable, List, Optional, Sequence,
                    Set, Tuple, Union)

import numpy as np
import torch

import tensorrt_llm
import tensorrt_llm.bindings
from tensorrt_llm._torch.distributed.communicator import Distributed, ReduceOp
from tensorrt_llm._utils import (TensorWrapper, convert_to_torch_tensor,
                                 get_size_in_bytes)
from tensorrt_llm.bindings.internal.batch_manager.kv_cache_manager_v2_utils import (
    IndexMapper, copy_batch_block_offsets_to_device)
from tensorrt_llm.bindings.internal.runtime import TaskLayerModuleConfig
from tensorrt_llm.llmapi.llm_args import (KvCacheConfig, PeftCacheConfig,
                                          PybindMirror)
from tensorrt_llm.lora_helper import LoraConfig
from tensorrt_llm.lora_manager import LoraManager, LoraModelConfig
from tensorrt_llm.math_utils import ceil_div
from tensorrt_llm.runtime import ModelConfig as ModelConfigPython
from tensorrt_llm.runtime.kv_cache_manager_v2 import (AttentionLayerConfig,
                                                      BufferConfig,
                                                      GpuCacheTierConfig,
                                                      HostCacheTierConfig)
from tensorrt_llm.runtime.kv_cache_manager_v2 import \
    KVCacheManager as KVCacheManagerPy
from tensorrt_llm.runtime.kv_cache_manager_v2 import \
    KVCacheManagerConfig as KVCacheManagerConfigPy
from tensorrt_llm.runtime.kv_cache_manager_v2 import (LayerId, TokenIdExt,
                                                      _KVCache)
from tensorrt_llm.runtime.kv_cache_manager_v2._common import GPU_LEVEL
from tensorrt_llm.runtime.kv_cache_manager_v2._config import DataRole
from tensorrt_llm.runtime.kv_cache_manager_v2._utils import (exact_div,
                                                             typed_range)
from tensorrt_llm.sampling_params import SamplingParams

from ..._utils import (binding_to_str_dtype, get_size_in_bytes, mpi_rank,
                       nvtx_range)
from ...logger import logger
from ...mapping import CpType, Mapping
from .kv_cache_connector import KvCacheConnectorManager
from .llm_request import (LlmRequest, LlmRequestState, SamplingConfig,
                          get_draft_token_length)
from .scheduler import ScheduledRequests

BufferManagerCpp = tensorrt_llm.bindings.internal.runtime.BufferManager
KVCacheManagerCpp = tensorrt_llm.bindings.internal.batch_manager.KVCacheManager
CacheTypeCpp = tensorrt_llm.bindings.internal.batch_manager.CacheType
ModelConfigCpp = tensorrt_llm.bindings.ModelConfig
DataType = tensorrt_llm.bindings.DataType
KVCacheEventManagerCpp = tensorrt_llm.bindings.internal.batch_manager.KVCacheEventManager
RequestList = list[LlmRequest]
PeftCacheManagerCpp = tensorrt_llm.bindings.internal.batch_manager.PeftCacheManager
WorldConfig = tensorrt_llm.bindings.WorldConfig

if TYPE_CHECKING:
    from tensorrt_llm._torch.attention_backend.interface import \
        AttentionMetadata

TempAttentionWindowInputs = tensorrt_llm.bindings.internal.batch_manager.TempAttentionWindowInputs
BlocksPerWindow = Dict[int, Tuple[
    int,
    int]]  # window_size -> (blocks_in_primary_pool, blocks_in_secondary_pool)


class ResourceManagerType(enum.Enum):
    KV_CACHE_MANAGER = "KV_CACHE_MANAGER"
    DRAFT_KV_CACHE_MANAGER = "DRAFT_KV_CACHE_MANAGER"
    PEFT_CACHE_MANAGER = "PEFT_CACHE_MANAGER"
    SEQ_SLOT_MANAGER = "SEQ_SLOT_MANAGER"
    SPEC_RESOURCE_MANAGER = "SPEC_RESOURCE_MANAGER"


class Role:
    KEY = DataRole("key")
    VALUE = DataRole("value")
    KEY_BLOCK_QUANT = DataRole("key_block_quant")
    VALUE_BLOCK_QUANT = DataRole("value_block_quant")
    ALL = DataRole("all")


def compute_page_count(token_count: int, tokens_per_page: int) -> int:
    return (token_count + tokens_per_page) // tokens_per_page


class BaseResourceManager(ABC):

    @abstractmethod
    def get_max_resource_count(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def get_needed_resource_to_completion(self, request: LlmRequest) -> int:
        raise NotImplementedError

    def add_dummy_requests(self, request_ids: List[int]):
        pass

    def prepare_resources(self, scheduled_batch: ScheduledRequests):
        pass

    def update_resources(self, scheduled_batch: ScheduledRequests):
        pass

    def free_resources(self, request: LlmRequest):
        pass

    def shutdown(self):
        pass


def get_pp_layers(
    num_layers: int,
    mapping: Mapping,
    spec_config: Optional["DecodingBaseConfig"] = None,
    layer_mask: Optional[List[bool]] = None,
) -> Tuple[List[int], int]:
    from ..speculative.utils import get_num_spec_layers

    total_num_layers = num_layers
    if layer_mask is not None:
        assert sum(layer_mask) == num_layers, (
            f"The number of enabled layers in layer_mask ({sum(layer_mask)}) "
            f"must match the number of layers ({num_layers}) "
            f"in KV cache manager, but get layer_mask: {layer_mask}")
        total_num_layers = len(layer_mask)
    pp_layers = mapping.pp_layers(total_num_layers)
    if layer_mask is not None:
        pp_layers = [i for i in pp_layers if layer_mask[i]]
    if spec_config is not None:
        num_spec_layers = get_num_spec_layers(spec_config)
        total_num_layers += num_spec_layers
        if mapping.is_last_pp_rank():
            pp_layers.extend(
                range(total_num_layers - num_spec_layers, total_num_layers))
    if len(pp_layers) == 0:
        # Don't support empty KV cache for now, provide at least 1 layer
        pp_layers.append(0)
    return pp_layers, total_num_layers


def request_context(is_draft: bool, scheduled_requests: ScheduledRequests):

    class RequestContext:

        def __init__(self, is_draft: bool,
                     scheduled_requests: ScheduledRequests):
            self.is_draft = is_draft
            self.scheduled_requests = scheduled_requests

        def __enter__(self):
            if not self.is_draft:
                return

            for req in self.scheduled_requests.all_requests():
                req.use_draft_model = True

        def __exit__(self, exc_type, exc_val, exc_tb):
            if not self.is_draft:
                return

            # Clean up the state
            for req in self.scheduled_requests.all_requests():
                req.use_draft_model = False

    return RequestContext(is_draft, scheduled_requests)


class KVCacheManager(BaseResourceManager):

    def __init__(
        self,
        kv_cache_config: KvCacheConfig,
        kv_cache_type: CacheTypeCpp,
        *,
        num_layers: int,
        num_kv_heads: Union[int, List[Optional[int]]],
        head_dim: int,
        tokens_per_block: int,
        # Note that max_seq_len is not necessarily equal to kv_cache_config.num_tokens.
        # It's derived from the model's BuildConfig for consistency with the C++ backend.
        max_seq_len: int,
        max_batch_size: int,
        mapping: Mapping,
        dtype: DataType = DataType.HALF,
        spec_config: Optional["DecodingBaseConfig"] = None,
        layer_mask: Optional[List[bool]] = None,
        max_num_tokens: int = 8192,
        model_config: Optional[ModelConfigCpp] = None,
        max_beam_width: int = 1,
        is_draft: bool = False,
        kv_connector_manager: Optional[KvCacheConnectorManager] = None,
        enable_indexer_k_cache: bool = False,
        indexer_k_cache_quant_block_size: int = 128,
        indexer_k_cache_index_head_dim: int = 0,
        is_estimating_kv_cache: bool = False,
        execution_stream: Optional[torch.cuda.Stream] = None,
        **kwargs,
    ) -> None:
        self.mapping = mapping
        self.dtype = dtype
        self.kv_cache_type = kv_cache_type
        self.pp_layers, self.num_layers = get_pp_layers(
            num_layers,
            mapping,
            spec_config=spec_config,
            layer_mask=layer_mask,
        )
        self.is_draft = is_draft
        self.num_local_layers = len(self.pp_layers)
        self.layer_offsets = {
            idx: offset
            for offset, idx in enumerate(self.pp_layers)
        }

        self.kv_connector_manager = kv_connector_manager

        tp_size = mapping.tp_size
        if mapping.enable_attention_dp:
            tp_size = 1

        if isinstance(num_kv_heads, int):
            self.num_kv_heads_per_layer = [
                (num_kv_heads + tp_size - 1) // tp_size
                for _ in range(self.num_local_layers)
            ]
            self.total_num_kv_heads_per_layer = [
                (num_kv_heads + tp_size - 1) // tp_size
                for _ in range(self.num_layers)
            ]
        else:
            assert len(num_kv_heads) == self.num_layers

            def append_to_kv_heads_per_layer(num_kv_heads_per_layer: List[int],
                                             kv_head: Optional[int]):
                if kv_head is not None:
                    num_kv_heads_per_layer.append(
                        (kv_head + tp_size - 1) // tp_size)
                else:
                    num_kv_heads_per_layer.append(0)

            self.num_kv_heads_per_layer = []
            if self.num_local_layers > 0:
                for i in self.pp_layers:
                    kv_head = num_kv_heads[i]
                    append_to_kv_heads_per_layer(self.num_kv_heads_per_layer,
                                                 kv_head)

            self.total_num_kv_heads_per_layer = []
            for i in range(self.num_layers):
                kv_head = num_kv_heads[i]
                append_to_kv_heads_per_layer(self.total_num_kv_heads_per_layer,
                                             kv_head)

        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.tokens_per_block = tokens_per_block
        self.max_seq_len = max_seq_len
        self.max_batch_size = max_batch_size
        self.kv_factor = 1 if kv_cache_type == CacheTypeCpp.SELFKONLY else 2
        # Some speculative decoding methods need to use different kv lengths for the
        # draft/target layers. Add extra tokens to handle this issue.
        # Import here to avoid circular imports
        from ..speculative import get_num_extra_kv_tokens
        self.num_extra_kv_tokens = get_num_extra_kv_tokens(spec_config)
        self.event_buffer_max_size = kv_cache_config.event_buffer_max_size
        self.attention_dp_events_gather_period_ms = kv_cache_config.attention_dp_events_gather_period_ms
        self.max_num_tokens = max_num_tokens
        self.max_draft_len = spec_config.max_draft_len if spec_config is not None else 0
        self.max_total_draft_tokens = spec_config.max_total_draft_tokens if spec_config is not None else 0

        # Determine max_attention_window_vec
        if kv_cache_config.max_attention_window is None:
            # Use max_seq_len as default max_attention_window
            self.max_attention_window_vec = [max_seq_len]
        else:
            self.max_attention_window_vec = kv_cache_config.max_attention_window.copy(
            )  # Make a copy to avoid modifying original
            # Clamp all window sizes to max_seq_len before calculating the
            # number of KV cache blocks. This prevents the KV cache pool from
            # being skewed by the largest window values.
            self.max_attention_window_vec = [
                min(max_seq_len, w) for w in self.max_attention_window_vec
            ]

        sink_token_length = (kv_cache_config.sink_token_length
                             if kv_cache_config.sink_token_length is not None
                             else 0)

        # Determine if this is VSWA (Variable Sliding Window Attention)
        self.is_vswa = len(set(self.max_attention_window_vec)) > 1

        # Calculate kv cache blocks for each window size
        # FIXME: flashinfer.py accesses kv_cache_manager.blocks_in_primary_pool
        # This dependency should be adjusted as it only covers the single window
        # case and not VSWA scheme.
        if is_estimating_kv_cache:
            # If this is an estimation dry run, we have already calculated the
            # max_tokens under _util.py::try_prepare_estimation
            # Since this is a dry run, assigning the same max_tokens capacity
            # to all window sizes as they are full attentions is enough.
            self.blocks_in_primary_pool = int(kv_cache_config.max_tokens //
                                              tokens_per_block)

            host_cache_size = kv_cache_config.host_cache_size if kv_cache_config.host_cache_size else 0
            max_tokens_secondary = host_cache_size // self.get_cache_bytes_per_token(
            )
            self.blocks_in_secondary_pool = int(max_tokens_secondary //
                                                tokens_per_block)

            blocks_per_window = {
                window_size:
                (self.blocks_in_primary_pool, self.blocks_in_secondary_pool)
                for window_size in set(self.max_attention_window_vec)
            }
            logger.info(
                f"[kv cache manager] Primary/secondary blocks for window sizes set to {blocks_per_window} for estimation dry run"
            )
        else:
            if self.is_vswa:
                # VSWA case: use C++ implementation for variable window sizes
                if model_config is None:
                    raise ValueError(
                        "model_config is required for VSWA (Variable Sliding Window Attention)"
                    )
                assert isinstance(
                    kv_cache_config, KvCacheConfig
                ), "calculate_max_num_blocks_from_cpp only accepts KvCacheConfig"
                blocks_per_window = self.calculate_max_num_blocks_from_cpp(
                    kv_cache_config=kv_cache_config,
                    model_config=model_config,
                    extra_cost_memory=0,
                )
            else:
                # Standard case: use original Python implementation
                self.blocks_in_primary_pool, self.blocks_in_secondary_pool = self.calculate_max_num_blocks(
                    kv_cache_config=kv_cache_config,
                    head_dim=head_dim,
                    tokens_per_block=tokens_per_block,
                    mapping=mapping,
                    dtype=dtype,
                    kv_factor=self.kv_factor,
                )
                blocks_per_window = {
                    self.max_attention_window_vec[0]:
                    (self.blocks_in_primary_pool, self.blocks_in_secondary_pool)
                }

        # Validate and adjust attention windows against their upper bounds if needed
        blocks_per_window, self.max_seq_len, self.max_attention_window_vec = self._validate_and_adjust_attention_windows(
            max_attention_window_vec=self.max_attention_window_vec,
            blocks_per_window=blocks_per_window,
            tokens_per_block=tokens_per_block,
            sink_token_length=sink_token_length,
            max_seq_len=self.max_seq_len,
            max_beam_width=max_beam_width,
        )

        if kv_cache_type != CacheTypeCpp.SELF:
            assert len(
                blocks_per_window
            ) == 1, "Only one window size is supported for non-self KV cache"
            # rewrite the attention window size in blocks_per_window
            memory_pools = blocks_per_window[self.max_attention_window_vec[0]]
            blocks_per_window = {self.max_seq_len: memory_pools}
            logger.info(
                f"Adjusted attention window size to {self.max_seq_len} in blocks_per_window"
            )

        # Set up temp_attention_window_inputs
        temp_attention_window_inputs = self._set_temp_attention_window_inputs()

        # Use the provided execution stream for proper synchronization with KVCacheTransferManager.
        # The execution stream is the stream where model forward kernels run, and KVCacheTransferManager
        # needs to synchronize with it for onboard/offload operations.
        # If no execution stream is provided, create a new one (for backward compatibility).
        self._stream = execution_stream if execution_stream is not None else torch.cuda.Stream(
        )
        logger.info(f"[KVCacheManager] execution_stream: {self._stream}")
        kwargs = {
            'num_kv_heads_per_layer': self.num_kv_heads_per_layer,
            'size_per_head': head_dim,
            'tokens_per_block': tokens_per_block,
            'blocks_per_window': blocks_per_window,
            'max_num_sequences': max_batch_size,
            'max_beam_width': max_beam_width,
            'max_attention_window_vec': self.max_attention_window_vec,
            'temp_attention_window_inputs': temp_attention_window_inputs,
            'dtype': dtype,
            'sink_token_length': sink_token_length,
            'stream': self._stream.cuda_stream,  # Pass to BufferManager
            'max_sequence_length': max_seq_len,
            'enable_block_reuse': kv_cache_config.enable_block_reuse,
            'onboard_blocks': kv_cache_config.onboard_blocks,
            'cache_type': kv_cache_type,
            'enable_partial_reuse': kv_cache_config.enable_partial_reuse,
            'copy_on_partial_reuse': kv_cache_config.copy_on_partial_reuse,
            'kv_connector_manager': self.kv_connector_manager,
            'enable_indexer_k_cache': enable_indexer_k_cache,
            'indexer_k_cache_quant_block_size':
            indexer_k_cache_quant_block_size,
            'indexer_k_cache_index_head_dim': indexer_k_cache_index_head_dim
        }

        if self.event_buffer_max_size > 0:
            if mapping.enable_attention_dp:
                kwargs['event_manager'] = KVCacheEventManagerCpp(
                    max_kv_event_entries=self.event_buffer_max_size,
                    attention_dp_rank=mapping.rank,
                    attention_dp_size=mapping.world_size,
                    attention_dp_events_gather_period_ms=self.
                    attention_dp_events_gather_period_ms,
                )
            elif mpi_rank() == 0:
                kwargs['event_manager'] = KVCacheEventManagerCpp(
                    max_kv_event_entries=self.event_buffer_max_size)

        self.impl = KVCacheManagerCpp(**kwargs)

        self.impl.allocate_pools(False)
        self.kv_cache_pool_pointers = self.impl.get_block_pool_pointers()
        kv_cache_block_scale_pool_pointers = self.impl.get_block_scale_pool_pointers(
        )
        if kv_cache_block_scale_pool_pointers.numel() > 0:
            self.kv_cache_pool_pointers = torch.stack([
                self.kv_cache_pool_pointers, kv_cache_block_scale_pool_pointers
            ],
                                                      dim=-1)

        self.kv_cache_pool_mapping = self.impl.get_layer_to_pool_mapping()
        self.num_pools = self.impl.num_pools
        self.max_blocks_per_seq = self.impl.max_blocks_per_seq
        self.enable_block_reuse = kv_cache_config.enable_block_reuse
        self.host_kv_cache_block_offsets = torch.empty(self.num_pools,
                                                       max_batch_size *
                                                       max_beam_width,
                                                       2,
                                                       self.max_blocks_per_seq,
                                                       dtype=torch.int32,
                                                       pin_memory=True,
                                                       device='cpu')

    def shutdown(self):
        self.impl.release_pools()

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
        with request_context(self.is_draft, scheduled_batch):
            context_batch = scheduled_batch.context_requests
            generation_batch = scheduled_batch.generation_requests

            # wait for all pending work to finish before launching offload/onboarding/partial copy
            self.impl.sync_transfer_manager_with_buffer_manager()

            # allocate KV Cache
            for req in context_batch:
                req_beam_width = req.sampling_config.beam_width
                if 'cp_type' in self.mapping.cp_config and CpType.STAR == self.mapping.cp_config[
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
                    if req.is_first_context_chunk and self._kv_connector_should_add_sequence(
                            req):
                        self.impl.add_sequence(req.py_request_id,
                                               req.prompt_len, req_beam_width,
                                               req)
                        for _ in range(self.num_extra_kv_tokens):
                            self.impl.add_token(req.py_request_id)
                        for _ in range(get_draft_token_length(req)):
                            self.impl.add_token(req.py_request_id)

                        if self.kv_connector_manager is not None:
                            block_ids = self.get_cache_indices(req)
                            self.kv_connector_manager.update_state_after_alloc(
                                req, block_ids)

            for req in generation_batch:
                if self.mapping.has_cp_helix():
                    # Distribute the decode blocks across CP ranks in a round-robin manner.
                    decode_block_id = (req.py_decoding_iter -
                                       1) // self.tokens_per_block
                    if decode_block_id % self.mapping.cp_size == self.mapping.cp_rank:
                        req.py_helix_is_inactive_rank = False
                        req.seqlen_this_rank_cp += 1
                    else:
                        req.py_helix_is_inactive_rank = True
                        # Skip allocating KV cache at decode for inactive helix ranks.
                        continue
                self.impl.add_token(req.py_request_id)
                for _ in range(get_draft_token_length(req)):
                    self.impl.add_token(req.py_request_id)

            # prefill and generation kernels wait for scheduled offload/onboard/partial copy work before launching
            self.impl.refresh_blocks()

        if self.kv_connector_manager is not None:
            self.kv_connector_manager.build_scheduler_output(
                scheduled_batch, self)

    def _kv_connector_should_add_sequence(self, request: LlmRequest) -> bool:
        return self.kv_connector_manager is None or self.kv_connector_manager.should_add_sequence(
            request)

    def add_dummy_requests(
        self,
        request_ids: List[int],
        # Note that token_nums should be past_kv_len + input_len (without
        # spec decoding). The draft tokens will be added in this function,
        # so we don't need to take care of it in the caller. When preparing
        # token_nums, we should not take the draft tokens into account, so
        # don't use the kv_cache_manager.max_seq_len, which includes both
        # extra tokens and draft tokens.
        token_nums: Optional[List[int]] = None,
        is_gen: bool = False,
        prepare_resource: bool = True,
        max_num_draft_tokens: int = 0,
        use_mrope: bool = False,
        max_beam_width: int = 1,
        # For capturable drafting loops. During normal inference, the draft model always
        # has enough KV cache space to fit all of our draft tokens. During warmup, however,
        # we need to make the KV cache manager aware that multiple autoregressive steps will
        # occur.
        num_extra_decoding_steps: int = 0,
    ):
        available_blocks = self.get_num_free_blocks()
        # No padding if not enough KV cache space
        if available_blocks < 1:
            return None

        beam_width = max_beam_width
        requests = []
        for i, req_id in enumerate(request_ids):
            # exact choice of n can be ignored for dummy requests
            sampling_params = SamplingParams(n=beam_width,
                                             best_of=beam_width,
                                             use_beam_search=beam_width > 1)
            # Here 1+max_num_draft_tokens is used to extend the prompt length to
            # a non-zero number to skip illegal memory access issue in MLA kernel
            # during warmup.
            token_num = token_nums[
                i] if token_nums is not None else 1 + max_num_draft_tokens
            encoder_input_tokens = [
                1
            ] * token_num if self.impl.cross_kv else None
            # Using 1 instead of 0 prevents NaN during warmup in e.g. Deepseek
            req = LlmRequest(request_id=req_id,
                             max_new_tokens=1,
                             input_tokens=[1] * token_num,
                             sampling_config=SamplingConfig(
                                 sampling_params._get_sampling_config()),
                             is_streaming=False,
                             encoder_input_tokens=encoder_input_tokens)
            req.is_dummy_request = True
            req.paged_kv_block_ids = []
            if prepare_resource:
                self.impl.add_sequence(req_id, token_num, beam_width, req)
                for _ in range(self.num_extra_kv_tokens):
                    self.impl.add_token(req_id)

                for _ in range(num_extra_decoding_steps):
                    self.impl.add_token(req_id)

            if is_gen:
                req.state = LlmRequestState.GENERATION_IN_PROGRESS
                req.prompt_len = token_num - 1
                req.py_prompt_len = req.prompt_len
                # Helix parallelism: each CP rank holds token_num tokens per sequence.
                # Since KV cache write for query token happens only on the active rank,
                # prompt_len is (token_num - 1) there and token_num on inactive ranks.
                if self.mapping.has_cp_helix():
                    # Mark only the last rank to be active for helix parallelism.
                    if self.mapping.cp_size - 1 == self.mapping.cp_rank:
                        req.py_helix_is_inactive_rank = False
                        req.prompt_len = token_num - 1
                        req.py_prompt_len = req.prompt_len
                        req.seqlen_this_rank_cp = req.prompt_len
                        req.total_input_len_cp = token_num * self.mapping.cp_size - 1
                    else:
                        req.py_helix_is_inactive_rank = True
                        req.prompt_len = token_num
                        req.py_prompt_len = req.prompt_len
                        req.seqlen_this_rank_cp = req.prompt_len
                        req.total_input_len_cp = token_num * self.mapping.cp_size - 1
                req.py_draft_tokens = [1] * max_num_draft_tokens
                if prepare_resource:
                    for _ in range(max_num_draft_tokens):
                        self.impl.add_token(req_id)

            # TODO: Planning to get dummy_data from each model. Before that, we need to add dummy mrop_config to the request here.
            if use_mrope:
                dummy_mrope_position_ids = torch.arange(
                    0, token_num, dtype=torch.int32).expand(3, 1, -1).clone()
                req.py_multimodal_data = {
                    "mrope_config": {
                        "mrope_position_ids": dummy_mrope_position_ids
                    }
                }
                if is_gen:
                    dummy_mrope_position_deltas = torch.zeros(
                        1, dtype=torch.int32).unsqueeze(0)
                    req.py_multimodal_data["mrope_config"][
                        "mrope_position_deltas"] = dummy_mrope_position_deltas
            requests.append(req)
        return requests

    def update_resources(self,
                         scheduled_batch: ScheduledRequests,
                         attn_metadata: "AttentionMetadata" = None,
                         kv_cache_dtype_byte_size: float = None):
        if not self.is_draft:
            self.update_kv_cache_draft_token_location(scheduled_batch,
                                                      attn_metadata,
                                                      kv_cache_dtype_byte_size)

        # Rewind KV cache for requests with rejected draft tokens.
        # Skip:
        # - GENERATION_COMPLETE: finished requests
        # - CONTEXT_INIT: requests whose state was reset after being paused with KV cache freed.
        #   With overlap scheduler, the scheduler pauses a request and frees KV cache at iteration N,
        #   while the previous batch (N-1) is still trying to update the KV cache after forward pass.
        for request in scheduled_batch.generation_requests:
            if request.state in (LlmRequestState.GENERATION_COMPLETE,
                                 LlmRequestState.CONTEXT_INIT):
                continue
            if request.py_rewind_len > 0:
                self.rewind_kv_cache(request, request.py_rewind_len)

        # For context requests, we store the blocks for reuse.
        for request in scheduled_batch.context_requests:
            self.impl.store_context_blocks(request)

    def locate_accepted_draft_tokens(self, requests: List[LlmRequest]):
        num_accepted_draft_tokens = []
        accepted_draft_tokens_indices = []
        rewind_draft_token_separate_adjustments = []
        # for context requests, the py_num_accepted_draft_tokens = 0, and py_num_accepted_draft_tokens_indices = []
        for seq in requests:
            num_accepted_draft_tokens.append(seq.py_num_accepted_draft_tokens)
            rewind_draft_token_separate_adjustments.append(
                seq.py_rewind_draft_token_separate_adjustment)
            accepted_draft_tokens_indices.extend(
                seq.py_num_accepted_draft_tokens_indices)
        batch_size = len(requests)
        num_accepted_draft_tokens_offset = torch.zeros(batch_size + 1,
                                                       dtype=torch.int32,
                                                       device='cuda')
        num_accepted_draft_tokens_offset[1:] = torch.cumsum(torch.tensor(
            num_accepted_draft_tokens, dtype=torch.int32),
                                                            dim=0)
        accepted_draft_tokens_indices = torch.tensor(
            accepted_draft_tokens_indices, dtype=torch.int32, device='cuda')
        rewind_draft_token_separate_adjustments = torch.tensor(
            rewind_draft_token_separate_adjustments,
            dtype=torch.int32,
            device='cuda')
        return num_accepted_draft_tokens_offset, accepted_draft_tokens_indices, rewind_draft_token_separate_adjustments

    def update_kv_cache_draft_token_location(self,
                                             scheduled_batch: ScheduledRequests,
                                             attn_metadata: "AttentionMetadata",
                                             kv_cache_dtype_byte_size: float):
        run_kv_cache_rellocation = False
        for request in scheduled_batch.generation_requests:
            if request.state != LlmRequestState.GENERATION_COMPLETE:
                if request.py_num_accepted_draft_tokens > 0 and len(
                        request.py_num_accepted_draft_tokens_indices) > 0:
                    run_kv_cache_rellocation = True
        if not run_kv_cache_rellocation:
            return
        requests = scheduled_batch.all_requests()
        accepted_draft_token_offsets, packed_accepted_draft_tokens_indices, rewind_draft_token_separate_adjustments = self.locate_accepted_draft_tokens(
            requests)
        past_key_value_lengths = attn_metadata.kv_lens_cuda[:len(requests)]
        if attn_metadata.kv_cache_block_offsets is not None and attn_metadata.host_kv_cache_pool_pointers is not None and attn_metadata.host_kv_cache_pool_mapping is not None:
            use_paged_kv_cache = True
        else:
            use_paged_kv_cache = False
        assert use_paged_kv_cache, "Only paged kv cache is supported"
        assert len(
            self.max_attention_window_vec
        ) == 1, "Currently, only one max attention window size is supported."

        if use_paged_kv_cache:
            torch.ops.tensorrt_llm.update_kv_cache_draft_token_location(
                accepted_draft_token_offsets,
                packed_accepted_draft_tokens_indices,
                past_key_value_lengths,
                True,
                self.num_layers,
                self.num_kv_heads,
                int(self.head_dim * kv_cache_dtype_byte_size),
                self.max_total_draft_tokens,
                self.max_attention_window_vec[0],
                rewind_draft_token_separate_adjustments,
                None,
                self.kv_cache_pool_pointers,
                attn_metadata.kv_cache_block_offsets,
                self.max_blocks_per_seq,
                self.tokens_per_block,
                None,
            )

    def free_resources(self, request: LlmRequest, pin_on_release: bool = False):
        return self.impl.remove_sequence(request.py_request_id, request,
                                         pin_on_release)

    def store_blocks_for_reuse(self,
                               request: LlmRequest,
                               pin_blocks: bool = False):
        return self.impl.store_blocks_for_reuse(request.py_request_id, request,
                                                pin_blocks)

    @staticmethod
    def calculate_scaling_factor_size_bytes(
            cache_size: int, quant_vector_size: int,
            scaling_factor_dtype: DataType) -> int:
        assert cache_size % quant_vector_size == 0, "NVFP4 cache size must be divisible by quant vector size"
        return get_size_in_bytes(cache_size // quant_vector_size,
                                 scaling_factor_dtype)

    # TODO: refactor get_cache_size_per_token and get_cache_bytes_per_token to use the same logic
    @staticmethod
    def get_cache_size_per_token(model_config: ModelConfigPython,
                                 mapping: Mapping, **kwargs):

        # get num key value heads
        config = model_config.pretrained_config
        num_key_value_heads = getattr(config, 'num_key_value_heads',
                                      config.num_attention_heads)
        if isinstance(num_key_value_heads, Iterable):
            num_key_value_heads = sum(num_key_value_heads) / len(
                num_key_value_heads)

        # get head dim
        mla = hasattr(config, "kv_lora_rank")
        if mla:
            head_dim = config.kv_lora_rank + config.qk_rope_head_dim
            kv_factor = 1
        else:
            tp_size = 1 if mapping.enable_attention_dp else mapping.tp_size
            head_dim = getattr(config, "head_dim", None)
            if not isinstance(head_dim, int):
                head_dim = config.hidden_size // config.num_attention_heads
            head_dim = head_dim * num_key_value_heads // tp_size
            kv_factor = 2

        # provide at least 1 layer to prevent division by zero cache size
        num_attention_layers = max(
            len(mapping.pp_layers(model_config.get_num_attention_layers())), 1)
        # K and V
        mem_per_token = kv_factor * num_attention_layers * head_dim
        # The data type bytes.
        quant_config = model_config.quant_config
        if quant_config is not None and quant_config.quant_mode.has_fp8_kv_cache(
        ):
            mem_per_token *= 1
        elif quant_config is not None and quant_config.quant_mode.has_fp4_kv_cache(
        ):
            # 1 bytes for 2 elements, and SFs (fp8) per 16 elements.
            mem_per_token = math.ceil(mem_per_token / 2) + math.ceil(
                mem_per_token / 16)
        else:
            # All other cases (fp16/bf16 kv cache), we need 2 bytes per token for K and V.
            assert quant_config is None or (
                not quant_config.quant_mode.has_kv_cache_quant()
            ), "Quantized kv cache is not expected"
            mem_per_token *= 2
        return mem_per_token

    def get_cache_bytes_per_token(self):
        cache_size_per_token = self.kv_factor * sum(
            self.num_kv_heads_per_layer) * self.head_dim

        if self.dtype not in (DataType.FP8, DataType.HALF, DataType.BF16,
                              DataType.FLOAT, DataType.NVFP4):
            raise ValueError(f'Cannot support {self.dtype} KV cache.')

        cache_size_bytes_per_token = get_size_in_bytes(cache_size_per_token,
                                                       self.dtype)
        if self.dtype == DataType.NVFP4:
            cache_size_bytes_per_token += self.calculate_scaling_factor_size_bytes(
                cache_size_per_token,
                quant_vector_size=16,
                scaling_factor_dtype=DataType.FP8)
        return cache_size_bytes_per_token

    def calculate_max_num_blocks(self,
                                 kv_cache_config: KvCacheConfig,
                                 head_dim: int,
                                 tokens_per_block: int,
                                 mapping: Mapping,
                                 dtype: DataType,
                                 kv_factor: int = 2):
        free_mem_fraction = (kv_cache_config.free_gpu_memory_fraction
                             if kv_cache_config.free_gpu_memory_fraction
                             is not None else 0.9)

        cache_size_bytes_per_token = self.get_cache_bytes_per_token()

        free_mem, total_mem = torch.cuda.mem_get_info()

        assert free_mem_fraction < 1.0, f"Invalid freeMemFraction, freeMemFraction {free_mem_fraction} must be smaller than 1.0"
        max_tokens = free_mem_fraction * free_mem / cache_size_bytes_per_token

        # If user specified a number of tokens
        if kv_cache_config.max_tokens is not None:
            # If user also specified a free gpu memory fraction, take the min
            if kv_cache_config.free_gpu_memory_fraction is not None:
                max_tokens = min(kv_cache_config.max_tokens, max_tokens)
                logger.warning(
                    f'Both free_gpu_memory_fraction and max_tokens are set (to {free_mem_fraction} and {max_tokens} with free memory {free_mem / (1 << 30)}GiB of total memory {total_mem / (1<<30)}GiB, respectively). The smaller value will be used.'
                )
            else:
                max_tokens = kv_cache_config.max_tokens
                logger.info(
                    f"max_tokens is set by kv_cache_config.max_tokens: {max_tokens}"
                )

        if mapping.world_size > 1:
            # make sure all ranks use same value for maxTokens
            dist = Distributed.get(mapping)
            max_tokens = dist.allreduce(
                max_tokens,
                op=ReduceOp.MIN,
            )

        # get number of blocks
        blocks_in_primary_pool = int(max_tokens // tokens_per_block)

        host_cache_size = kv_cache_config.host_cache_size if kv_cache_config.host_cache_size else 0
        max_tokens_secondary = host_cache_size // self.get_cache_bytes_per_token(
        )
        blocks_in_secondary_pool = int(max_tokens_secondary // tokens_per_block)

        return blocks_in_primary_pool, blocks_in_secondary_pool

    def get_max_atten_window_upper_bound(self, blocks_in_primary_pool,
                                         tokens_per_block, max_beam_width,
                                         sink_token_len,
                                         max_seq_len: Optional[int]):
        token_capacity = blocks_in_primary_pool * tokens_per_block
        max_blocks_per_seq = math.floor(token_capacity /
                                        (max_beam_width * tokens_per_block))
        assert max_blocks_per_seq > 0, "Impossible to fit in any sequence in kvCache"

        max_token_num = max_blocks_per_seq * tokens_per_block
        sink_tokens_in_last_block = sink_token_len % tokens_per_block
        sink_bubble_len = 0 if sink_tokens_in_last_block == 0 else tokens_per_block - sink_tokens_in_last_block
        max_atten_window_upper_bound = max_token_num - sink_bubble_len
        if max_seq_len is not None and max_seq_len > max_atten_window_upper_bound and max_beam_width > 1:
            max_atten_window_upper_bound -= tokens_per_block
        assert max_atten_window_upper_bound > 0, "Impossibe to fit in any sequence in kvCache"
        return max_atten_window_upper_bound

    def get_cache_indices(self,
                          request: LlmRequest,
                          window_size: Optional[int] = None) -> List[int]:
        if window_size is None:
            if len(self.max_attention_window_vec) > 1:
                raise ValueError("window_size must be provided for VSWA")
            window_size = self.max_attention_window_vec[0]

        result = self.impl.get_cache_block_ids(request.py_request_id,
                                               window_size)
        assert len(result) == 1
        return result[0]

    def unpin_blocks_by_id(self, kv_cache_block_id: int):
        self.impl.unpin_blocks_by_id(kv_cache_block_id)

    def get_last_block_id(self, request_id: int) -> int:
        return self.impl.get_last_block_id(request_id)

    def get_batch_cache_indices(
        self,
        request_ids: List[int],
        layer_idx: Optional[int] = None,
    ) -> List[List[int]]:
        if layer_idx is None:
            if len(self.max_attention_window_vec) > 1:
                raise ValueError("layer_idx must be provided for VSWA")
            window_size = self.max_attention_window_vec[0]
        else:
            layer_offset = self.layer_offsets[layer_idx]
            window_size = self.max_attention_window_vec[layer_offset % len(
                self.max_attention_window_vec)]

        result = self.impl.get_batch_cache_block_ids(request_ids, window_size)
        for i in range(len(result)):
            assert (len(result[i])) == 1
            result[i] = result[i][0]
        return result

    def get_num_free_blocks(self) -> int:
        if self.is_vswa:
            logger.info(
                f"For VSWA case, we return the minimum of the number of free blocks for each window size: {self.impl.get_kv_cache_stats().num_free_blocks_per_window_size}"
            )
            return min(self.impl.get_kv_cache_stats().
                       num_free_blocks_per_window_size.values())
        else:
            return self.impl.get_kv_cache_stats().free_num_blocks

    def get_num_kv_blocks(self, num_tokens: int) -> int:
        return (num_tokens + self.tokens_per_block - 1) // self.tokens_per_block

    def get_num_available_tokens(self,
                                 max_num_draft_tokens: int = 0,
                                 **kwargs) -> int:
        return (self.get_num_free_blocks() * self.tokens_per_block -
                self.num_extra_kv_tokens - max_num_draft_tokens)

    def get_buffers(self,
                    layer_idx: int,
                    kv_layout: str = "NHD") -> Optional[torch.Tensor]:
        ''' Slice KV tensor for a specified layer and reshape it.

        1. Slice:
            [max_num_pages, num_layers, kv_factor, page_size * num_kv_heads * head_dim] ->
            [max_num_pages, kv_factor, page_size * num_kv_heads * head_dim]

        2. Reshape:
            kv_layout = "NHD" -> [max_num_pages, kv_factor, page_size, num_kv_heads, head_dim]
            kv_layout = "HND" -> [max_num_pages, kv_factor, num_kv_heads, page_size, head_dim]

        Note that different attention backend/implementation can have different KV layouts,
        "kv_layout" should be set accordingly to avoid surprises.
        '''
        layer_offset = self.layer_offsets[layer_idx]
        result = self.impl.get_primary_pool_data(layer_offset)

        assert kv_layout in ["NHD",
                             "HND"], f"Unsupported kv_layout: {kv_layout}"
        if kv_layout == "NHD":
            return result.reshape(
                result.shape[0],
                self.kv_factor,
                self.tokens_per_block,
                self.num_kv_heads_per_layer[layer_offset],
                self.head_dim,
            )
        else:
            return result.reshape(
                result.shape[0],
                self.kv_factor,
                self.num_kv_heads_per_layer[layer_offset],
                self.tokens_per_block,
                self.head_dim,
            )

    def get_indexer_k_cache_pool_data(self, layer_idx: int) -> torch.Tensor:
        result = self.impl.get_indexer_k_cache_pool_data(layer_idx)
        return result.view(result.shape[0], -1)

    def check_invalid_values_in_kv_cache(self,
                                         fill_with_zero: bool = False) -> bool:
        some_checks_unavailable = False
        has_invalid_values = torch.tensor([False],
                                          dtype=torch.bool,
                                          device=torch.cuda.current_device())
        for layer_idx, layer_offset in self.layer_offsets.items():
            buffer = self.impl.get_primary_pool_data(layer_offset)
            # process in chunks of 256 pages to avoid OoM
            for i in range(0, buffer.shape[0], 256):
                buffer_slice = buffer[i:i + 256]
                try:
                    has_invalid_values.logical_or_(
                        torch.isnan(buffer_slice).any())
                    has_invalid_values.logical_or_(
                        torch.isinf(buffer_slice).any())
                except NotImplementedError:
                    some_checks_unavailable = True
            if fill_with_zero:
                buffer.zero_()
        torch.cuda.synchronize()

        if some_checks_unavailable:
            logger.warning(
                "`torch.isnan` or `torch.isinf` is not implemented for current kv cache dtype, related checks are skipped"
            )
        return bool(has_invalid_values)

    def get_unique_primary_pool(self) -> torch.Tensor:
        return self.impl.get_unique_primary_pool()

    def get_block_ids_per_seq(self, request_ids: List[int]) -> torch.Tensor:
        block_ids_per_seq = self.get_batch_cache_indices(request_ids)
        block_ids_per_seq_tensors = [
            torch.tensor(sublist, dtype=torch.int)
            for sublist in block_ids_per_seq
        ]
        padded_tensor = torch.nn.utils.rnn.pad_sequence(
            block_ids_per_seq_tensors, batch_first=True, padding_value=0)
        return padded_tensor

    def flush_iteration_events(self):
        self.impl.flush_iteration_events()

    def get_latest_events(self, timeout_ms: Optional[float] = 0):
        return self.impl.get_latest_events(timeout_ms)

    def get_kv_cache_stats(self):
        return self.impl.get_kv_cache_stats()

    def rewind_kv_cache(self, request: LlmRequest, rewind_len: int):
        self.impl.rewind_kv_cache(request.py_request_id, rewind_len)

    def _get_window_size_to_layers(self) -> dict[int, list[int]]:
        """
        Get the window size to layers mapping.
        The returned map has window sizes as keys and lists of layer indices as values.

        max_attention_window_vec is treated as a repeating pattern.
        """
        window_size_to_layers_map = defaultdict(list)

        if not self.max_attention_window_vec:
            # This case should ideally be prevented by earlier config validation.
            # If num_local_layers is 0, an empty map is fine.
            if self.num_local_layers > 0:
                raise Exception(
                    "max_attention_window_vec cannot be empty if there are local layers."
                )
            return {
            }  # Return an empty dict if no local layers or if somehow vec is empty and no layers.

        # Treat max_attention_window_vec as a repeating pattern.
        pattern_len = len(
            self.max_attention_window_vec
        )  # `sliding_window_pattern`, in HF config terms, e.g. https://huggingface.co/google/gemma-3-1b-it/blob/main/config.json#L32
        # early return if max_attention_window_vec is a single value(SWA)
        if pattern_len == 1:
            return {
                self.max_attention_window_vec[0]:
                list(range(self.num_local_layers))
            }
        for local_layer_idx in range(self.num_local_layers):
            window_size = self.max_attention_window_vec[local_layer_idx %
                                                        pattern_len]
            window_size_to_layers_map[window_size].append(local_layer_idx)
        return window_size_to_layers_map

    @staticmethod
    def adjust_window_sizes_for_vswa(
        window_size_to_layers: Dict[int, List[int]],
        max_attention_window_vec: List[int],
        kv_cache_config: KvCacheConfig,
        model_config: ModelConfigCpp,
        pool_memory_bytes: int,
        kv_factor: int,
        dtype: DataType,
        is_cross_attention: bool = False,
    ) -> Tuple[Dict[int, List[int]], List[int]]:

        assert is_cross_attention is False, 'Cross attention is not supported'

        max_tokens_from_config = kv_cache_config.max_tokens

        def calculate_cache_size_per_token(layers: Set[int]) -> int:
            # Same as BaseKVCacheManager::calculateCacheSizePerTokenForSingleWindowSize
            total_kv_heads = sum(model_config.num_kv_heads_per_layer[i]
                                 for i in layers)
            return total_kv_heads * kv_factor * model_config.head_size

        # Calculate the required memory bytes per sequence.
        required_mem_bytes_per_seq = 0
        for window_size in sorted(window_size_to_layers):
            layers = window_size_to_layers[window_size]
            cache_size_per_token = calculate_cache_size_per_token(layers)
            cache_size_bytes_per_token = get_size_in_bytes(
                cache_size_per_token, dtype)
            if dtype == DataType.NVFP4:
                cache_size_bytes_per_token += KVCacheManager.calculate_scaling_factor_size_bytes(
                    cache_size_per_token,
                    quant_vector_size=16,
                    scaling_factor_dtype=DataType.FP8)
            required_mem_bytes_per_seq += window_size * cache_size_bytes_per_token
        logger.debug(
            f'Required memory per sequence: {required_mem_bytes_per_seq} bytes')

        if required_mem_bytes_per_seq < pool_memory_bytes:
            # No need to adjust the window sizes.
            return (copy.deepcopy(window_size_to_layers),
                    max_attention_window_vec)

        logger.debug(
            f'Adjusting the window sizes {list(window_size_to_layers)} to fit '
            f'the memory {pool_memory_bytes} bytes.')
        adjusted_window_size_to_layers = {}

        remaining_mem_bytes = pool_memory_bytes
        remaining_layers = set(i for layers in window_size_to_layers.values()
                               for i in layers)

        accum_max_tokens = 0
        prev_window_size = 0
        adjusted_dict = {}
        adjusted_max_attention_window_vec = max_attention_window_vec.copy()

        for window_size in sorted(window_size_to_layers):
            layers = window_size_to_layers[window_size]
            if remaining_mem_bytes > 0 and remaining_layers:
                # Calculate cache size per token for remaining layers only
                cache_size_per_token = calculate_cache_size_per_token(
                    remaining_layers)
                cache_size_bytes_per_token = get_size_in_bytes(
                    cache_size_per_token, dtype)
                if dtype == DataType.NVFP4:
                    cache_size_bytes_per_token += KVCacheManager.calculate_scaling_factor_size_bytes(
                        cache_size_per_token,
                        quant_vector_size=16,
                        scaling_factor_dtype=DataType.FP8)
                logger.debug(
                    f'Cache size per token for {len(remaining_layers)} layers: '
                    f'{cache_size_bytes_per_token} bytes')
                # Calculate max tokens that can fit in this window with remaining memory.
                max_tokens_in_window = min(
                    remaining_mem_bytes // cache_size_bytes_per_token,
                    window_size - prev_window_size)
                remaining_mem_bytes -= max_tokens_in_window * cache_size_bytes_per_token
                accum_max_tokens += max_tokens_in_window
                logger.debug(f'Remaining memory: {remaining_mem_bytes} bytes')
                logger.debug(
                    f'Max token of window {window_size}: {accum_max_tokens}')

                if accum_max_tokens < window_size:
                    logger.debug(
                        f'Max tokens ({accum_max_tokens}) cannot fill the current window ({window_size}). '
                        f'The larger windows will have the same max tokens.')
                    remaining_mem_bytes = 0

                # Clamp the sequence length if provided explicitly.
                if max_tokens_from_config is not None:
                    accum_max_tokens = min(max_tokens_from_config,
                                           accum_max_tokens)
                    # If max tokens from config is reached, stop allocating
                    # more memory. Since the maximum number of tokens is
                    # already reached, for the remaining windows maxTokens
                    # will be set by the current value of accumMaxTokens.
                    if accum_max_tokens == max_tokens_from_config:
                        remaining_mem_bytes = 0

            if accum_max_tokens not in adjusted_window_size_to_layers:
                adjusted_window_size_to_layers[accum_max_tokens] = layers.copy()
            else:
                adjusted_window_size_to_layers[accum_max_tokens].extend(layers)
            adjusted_dict[window_size] = accum_max_tokens
            # also update adjusted_max_attention_window_vec
            adjusted_max_attention_window_vec = [
                adjusted_dict.get(v, v)
                for v in adjusted_max_attention_window_vec
            ]

            remaining_layers -= set(layers)
            prev_window_size = window_size

        return (adjusted_window_size_to_layers,
                adjusted_max_attention_window_vec)

    def calculate_max_num_blocks_from_cpp(
            self,
            kv_cache_config: KvCacheConfig,
            model_config: ModelConfigCpp,
            extra_cost_memory: int = 0) -> dict[int, tuple[int, int]]:
        """
        This function is a wrapper of KVCacheManagerCpp.calculate_max_num_blocks.
        The final goal is to switch to the C++ implementation of calculate_max_num_blocks.
        Currently, this function is added to support *ONLY* VSWA.

        Args:
            kv_cache_config: The KV cache configuration object.
            model_config: The model configuration object.
            extra_cost_memory: Extra memory in bytes to exclude from available memory.

        Returns:
            A dict of (max_attention_window, (blocks_in_primary_pool, blocks_in_secondary_pool)).
        """

        # VSWA on Torch backend has not supported the cross attention.
        is_cross_attention = False
        # check model config
        assert model_config.layer_types is not None, "layer_types have to be set correctly for VSWA"

        # Construct WorldConfig from self.mapping
        world_config_cpp = WorldConfig(
            tensor_parallelism=self.mapping.tp_size,
            pipeline_parallelism=self.mapping.pp_size,
            rank=self.mapping.rank,
            gpus_per_node=self.mapping.gpus_per_node,
            enable_attention_dp=self.mapping.enable_attention_dp)

        window_size_to_layers = self._get_window_size_to_layers()
        logger.debug(f"window_size_to_layers: {window_size_to_layers}")

        free_mem, total_mem = torch.cuda.mem_get_info()
        # Respect max_gpu_total_bytes if provided
        free_gpu_memory_fraction = kv_cache_config.free_gpu_memory_fraction if kv_cache_config.free_gpu_memory_fraction else 0.9
        self._primary_pool_memory_bytes = kv_cache_config.max_gpu_total_bytes if kv_cache_config.max_gpu_total_bytes > 0 else int(
            free_mem * free_gpu_memory_fraction)
        self._secondary_pool_memory_bytes = kv_cache_config.host_cache_size if kv_cache_config.host_cache_size else 0
        logger.debug(
            f"primary_pool_memory_bytes is set to {self._primary_pool_memory_bytes/1024**3}GB, \n"
            f"secondary_pool_memory_bytes is set to {self._secondary_pool_memory_bytes/1024**3}GB"
        )

        # Adjust the window sizes to fit the memory if even a single sequence
        # cannot fit in the memory.
        window_size_to_layers, max_attention_window_vec = self.adjust_window_sizes_for_vswa(
            window_size_to_layers=window_size_to_layers,
            max_attention_window_vec=self.max_attention_window_vec,
            model_config=model_config,
            kv_cache_config=kv_cache_config,
            pool_memory_bytes=self._primary_pool_memory_bytes,
            kv_factor=self.kv_factor,
            dtype=self.dtype,
            is_cross_attention=is_cross_attention,
        )
        self.max_attention_window_vec = max_attention_window_vec

        blocks_per_window = KVCacheManagerCpp.calculate_max_num_blocks(
            config=PybindMirror.maybe_to_pybind(kv_cache_config),
            # TODO: support cross attention
            is_cross_attention=is_cross_attention,
            dtype=self.dtype,
            model_config=model_config,
            world_config=world_config_cpp,
            window_size_to_layers=window_size_to_layers,
            allotted_primary_mem_bytes=self._primary_pool_memory_bytes,
            allotted_secondary_mem_bytes=self._secondary_pool_memory_bytes,
            extra_cost_memory=extra_cost_memory,
            kv_factor=self.kv_factor,
        )
        return blocks_per_window

    def _validate_and_adjust_attention_windows(
        self,
        max_attention_window_vec: List[int],
        blocks_per_window: BlocksPerWindow,
        tokens_per_block: int,
        sink_token_length: int,
        max_seq_len: int,
        max_beam_width: int,
    ) -> Tuple[BlocksPerWindow, int, List[int]]:
        """
        Validate and adjust attention windows against their upper bounds if needed.
        If there is no adjustment, the returned max_attention_window_vec will be the same as the input.

        Args:
            max_attention_window_vec: List of attention window sizes
            blocks_per_window: Dict mapping window size to (primary_blocks, secondary_blocks)
            tokens_per_block: Number of tokens per block
            sink_token_length: Length of sink tokens
            max_seq_len: Maximum sequence length

        Returns:
            Tuple of (adjusted_blocks_per_window, adjusted_max_seq_len, adjusted_max_attention_window_vec)
        """
        window_adjustments = {}
        # Validate each window size in blocks_per_window against its upper bound
        for window_size, (blocks_in_primary_pool,
                          _) in blocks_per_window.items():
            upper_bound = self.get_max_atten_window_upper_bound(
                blocks_in_primary_pool=blocks_in_primary_pool,
                tokens_per_block=tokens_per_block,
                max_beam_width=max_beam_width,
                sink_token_len=sink_token_length,
                max_seq_len=max_seq_len)
            if window_size > upper_bound:
                logger.warning(
                    f"Attention window size {window_size} exceeds upper bound {upper_bound} "
                    f"for available blocks. Reducing to {upper_bound}.")
                window_adjustments[window_size] = upper_bound
        # Apply adjustments to the window vector if any were needed
        if window_adjustments:
            adjusted_window_vec = [
                window_adjustments.get(window, window)
                for window in max_attention_window_vec
            ]
            logger.warning(
                f"Adjusted max_attention_window_vec to {adjusted_window_vec}")
            # update the window size in blocks_per_window if it is adjusted
            adjusted_blocks_per_window = {}
            for window_size, memory_pools in blocks_per_window.items():
                if window_size in window_adjustments:
                    adjusted_window_size = window_adjustments[window_size]
                    adjusted_blocks_per_window[
                        adjusted_window_size] = memory_pools
                    logger.warning(
                        f"Adjusted window size {window_size} to {adjusted_window_size} in blocks_per_window"
                    )
                else:
                    adjusted_blocks_per_window[window_size] = memory_pools
            # Update max_seq_len to the maximum of adjusted windows
            adjusted_max_seq_len = max(adjusted_window_vec)
            logger.warning(f"Adjusted max_seq_len to {adjusted_max_seq_len}")

            return adjusted_blocks_per_window, adjusted_max_seq_len, adjusted_window_vec
        else:
            return blocks_per_window, max_seq_len, max_attention_window_vec

    def pin_blocks(self, request_id: int):
        self.impl.pin_blocks(request_id)

    def _set_temp_attention_window_inputs(
            self) -> Optional[TempAttentionWindowInputs]:
        """
        Set up temp_attention_window_inputs for sliding window.
        """
        is_sliding_window = min(
            self.max_attention_window_vec) < self.max_seq_len
        if is_sliding_window:
            temp_attention_window_inputs = TempAttentionWindowInputs()
            temp_attention_window_inputs.paged_context_fmha = True
            temp_attention_window_inputs.max_input_len = self.max_seq_len - 1
            temp_attention_window_inputs.max_num_tokens = self.max_num_tokens
            return temp_attention_window_inputs
        else:
            return None

    def copy_batch_block_offsets(self, dst_tensor: torch.Tensor,
                                 request_ids: List[int], beam_width: int,
                                 num_context: int, num_seqs: int):
        self.impl.copy_batch_block_offsets(self.host_kv_cache_block_offsets,
                                           request_ids[:num_context], 1, 0)
        self.impl.copy_batch_block_offsets(self.host_kv_cache_block_offsets,
                                           request_ids[num_context:],
                                           beam_width, num_context)

        for pool_idx in range(self.host_kv_cache_block_offsets.shape[0]):
            dst_tensor[pool_idx, :num_seqs].copy_(
                self.host_kv_cache_block_offsets[pool_idx, :num_seqs],
                non_blocking=True)

    def reset_reuse_state(self):
        """Reset the reuse state of the KV cache manager."""
        self.impl.reset_reuse_state()


class KVCacheManagerV2(BaseResourceManager):

    def __init__(
        self,
        kv_cache_config: KvCacheConfig,
        kv_cache_type: CacheTypeCpp,
        *,
        num_layers: int,
        num_kv_heads: Union[int, List[Optional[int]]],
        head_dim: int,
        tokens_per_block: int,
        # Note that max_seq_len is not necessarily equal to kv_cache_config.num_tokens.
        # It's derived from the model's BuildConfig for consistency with the C++ backend.
        max_seq_len: int,
        max_batch_size: int,
        mapping: Mapping,
        dtype: DataType = DataType.HALF,
        spec_config=None,
        layer_mask: Optional[List[bool]] = None,
        vocab_size: int = None,
        max_num_tokens: int = 8192,
        model_config: Optional[ModelConfigCpp] = None,
        max_beam_width: int = 1,
        is_draft: bool = False,
        kv_connector_manager: Optional[KvCacheConnectorManager] = None,
        **kwargs,
    ) -> None:
        self.mapping = mapping
        self.dtype = dtype

        assert self.dtype != DataType.NVFP4, "NVFP4 is not supported for KVCacheManagerV2"
        assert kv_connector_manager is None, "kv_connector_manager is not supported for KVCacheManagerV2"
        assert max_beam_width == 1, "max_beam_width must be 1 for KVCacheManagerV2"

        self.kv_cache_type = kv_cache_type
        self.pp_layers, self.num_layers = get_pp_layers(
            num_layers,
            mapping,
            spec_config=spec_config,
            layer_mask=layer_mask,
        )
        self.is_draft = is_draft
        self.num_local_layers = len(self.pp_layers)
        self.layer_offsets = {
            idx: offset
            for offset, idx in enumerate(self.pp_layers)
        }
        self.max_beam_width = max_beam_width

        tp_size = mapping.tp_size
        if mapping.enable_attention_dp:
            tp_size = 1

        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.tokens_per_block = tokens_per_block
        self.max_seq_len = max_seq_len
        self.max_batch_size = max_batch_size
        self.kv_factor = 1 if kv_cache_type == CacheTypeCpp.SELFKONLY else 2
        from ..speculative import get_num_extra_kv_tokens
        self.num_extra_kv_tokens = get_num_extra_kv_tokens(spec_config)

        self.event_buffer_max_size = kv_cache_config.event_buffer_max_size

        assert self.event_buffer_max_size == 0, "event_buffer_max_size must be 0"

        # Determine max_attention_window_vec
        if kv_cache_config.max_attention_window is not None:

            self.max_attention_window_vec = kv_cache_config.max_attention_window.copy(
            )  # Make a copy to avoid modifying original
            # Clamp all window sizes to max_seq_len before calculating the
            # number of KV cache blocks. This prevents the KV cache pool from
            # being skewed by the largest window values.
            self.max_attention_window_vec = [
                min(max_seq_len, w) for w in self.max_attention_window_vec
            ]

            self.max_attention_window_vec = [
                None if w == max_seq_len else w
                for w in self.max_attention_window_vec
            ]

        else:
            self.max_attention_window_vec = [None]

        if isinstance(num_kv_heads, int):
            self.num_kv_heads_per_layer = [
                (num_kv_heads + tp_size - 1) // tp_size
                for _ in range(self.num_local_layers)
            ]
            self.total_num_kv_heads_per_layer = [
                (num_kv_heads + tp_size - 1) // tp_size
                for _ in range(self.num_layers)
            ]
        else:
            assert len(num_kv_heads) == self.num_layers

            def append_to_kv_heads_per_layer(num_kv_heads_per_layer: List[int],
                                             kv_head: Optional[int]):
                if kv_head is not None:
                    num_kv_heads_per_layer.append(
                        (kv_head + tp_size - 1) // tp_size)
                else:
                    num_kv_heads_per_layer.append(0)

            self.num_kv_heads_per_layer = []
            if self.num_local_layers > 0:
                for i in self.pp_layers:
                    kv_head = num_kv_heads[i]
                    append_to_kv_heads_per_layer(self.num_kv_heads_per_layer,
                                                 kv_head)

            self.total_num_kv_heads_per_layer = []
            for i in range(self.num_layers):
                kv_head = num_kv_heads[i]
                append_to_kv_heads_per_layer(self.total_num_kv_heads_per_layer,
                                             kv_head)

        self.is_vswa = len(set(self.max_attention_window_vec)) > 1

        self.kv_connector_manager = kv_connector_manager

        quota = float('inf')

        if kv_cache_config.max_tokens is not None:
            quota = int(
                ceil_div(
                    kv_cache_config.max_tokens *
                    self.get_cache_bytes_per_token(),
                    kv_cache_config.max_util_for_resume))
            if kv_cache_config.free_gpu_memory_fraction is not None:
                logger.warning(
                    f"Both max_tokens and free_gpu_memory_fraction are set to {kv_cache_config.max_tokens} and {kv_cache_config.free_gpu_memory_fraction}, the smaller value will be used."
                )
        if kv_cache_config.max_gpu_total_bytes is not None and kv_cache_config.max_gpu_total_bytes > 0:
            if quota > int(kv_cache_config.max_gpu_total_bytes):
                logger.warning(
                    f"max_gpu_total_bytes {kv_cache_config.max_gpu_total_bytes / (1 << 30)}GiB is smaller than the calculated quota {quota / (1 << 30)}GiB, clamping quota to {kv_cache_config.max_gpu_total_bytes / (1 << 30)}GiB"
                )
            quota = min(quota, int(kv_cache_config.max_gpu_total_bytes))

        assert quota != float(
            'inf'
        ), "Quota not set. Check kv_cache_config.max_tokens or kv_cache_config.max_gpu_total_bytes"
        logger.info(
            f"KV cache manager v2 device quota set to {quota / (1 << 30)}GiB")

        cache_tiers = [GpuCacheTierConfig(quota=quota)]
        if kv_cache_config.host_cache_size is not None and kv_cache_config.host_cache_size > 0:
            cache_tiers.append(
                HostCacheTierConfig(quota=kv_cache_config.host_cache_size))
            logger.info(
                f"KV cache manager v2 host cache quota set to {kv_cache_config.host_cache_size / (1 << 30)}GiB"
            )

        buffer_type = [Role.KEY]
        if kv_cache_type != CacheTypeCpp.SELFKONLY:
            buffer_type.append(Role.VALUE)

        config = KVCacheManagerConfigPy(
            tokens_per_block=tokens_per_block,
            vocab_size=vocab_size,
            cache_tiers=cache_tiers,
            max_util_for_resume=kv_cache_config.max_util_for_resume,
            layers=[
                AttentionLayerConfig(
                    layer_id=layer_id,
                    buffers=[
                        BufferConfig(
                            role=role,
                            size=self.get_cache_bytes_per_token(
                                local_layer_idx=layer_id, data_role=role) *
                            tokens_per_block,
                        ) for role in buffer_type
                    ],
                    sliding_window_size=self.max_attention_window_vec[
                        layer_id % len(self.max_attention_window_vec)],
                    num_sink_tokens=None,
                ) for layer_id in typed_range(LayerId(self.num_local_layers))
            ],
        )

        self.kv_cache_manager_py_config = config

        self.impl = KVCacheManagerPy(config)

        self.num_pools = len(self.impl.layer_grouping)

        self.layer_to_pool_mapping_dict: dict[int, int] = {
            layer_id: self.impl.get_layer_group_id(layer_id)
            for layer_id in typed_range(LayerId(self.num_local_layers))
        }

        self.kv_cache_pool_pointers = torch.tensor([[
            self.impl.get_mem_pool_base_address(
                self.impl.layer_grouping[pool_id][0], Role.KEY), 0
        ] for pool_id in range(self.num_pools)],
                                                   dtype=torch.int64,
                                                   device="cpu",
                                                   pin_memory=True)

        kv_cache_pool_mapping_list = []
        for layer_id in typed_range(LayerId(self.num_local_layers)):
            layer_group_id = self.impl.get_layer_group_id(layer_id)
            offset = exact_div(
                self.impl.get_mem_pool_base_address(layer_id, Role.KEY) -
                int(self.kv_cache_pool_pointers[layer_group_id][0]),
                self.get_cache_bytes_per_token(layer_id, Role.KEY) *
                self.kv_factor * self.tokens_per_block)
            kv_cache_pool_mapping_list.append([layer_group_id, offset])

        self.kv_cache_pool_mapping = torch.tensor(kv_cache_pool_mapping_list,
                                                  dtype=torch.int32,
                                                  device="cpu",
                                                  pin_memory=True)
        # Pad max_blocks_per_seq to next multiple of 4 for copy_block_offsets kernel
        self.max_blocks_per_seq = (max_seq_len + tokens_per_block -
                                   1) // tokens_per_block
        if self.max_blocks_per_seq % 4 != 0:
            self.max_blocks_per_seq = ((self.max_blocks_per_seq + 3) // 4) * 4

        self.kv_cache_map: dict[int, _KVCache] = {}

        max_num_tokens = self.get_num_available_tokens()

        if max_seq_len > max_num_tokens:
            logger.warning(
                f"max_seq_len {max_seq_len} is greater than max_num_tokens {max_num_tokens} that can be allocated in kv cache manager, setting max_seq_len to {max_num_tokens}"
            )
            self.max_seq_len = max_num_tokens

        self.enable_block_reuse = kv_cache_config.enable_block_reuse

        # Plus 1 for cuda graph dummy request
        self.index_mapper = IndexMapper(max_batch_size + 1, max_beam_width)

        self.host_kv_cache_block_offsets = torch.empty(
            self.num_pools,
            (max_batch_size + 1) * max_beam_width,
            2,  # key and value
            self.max_blocks_per_seq,
            dtype=torch.int32,
            pin_memory=True,
            device='cpu')

    @property
    def blocks_in_primary_pool(self) -> int:
        """
        Get the number of blocks in the primary pool.
        """
        return self.impl.get_page_index_upper_bound(0, Role.KEY)

    def get_buffers(self,
                    layer_idx: int,
                    kv_layout: str = "NHD") -> Optional[torch.Tensor]:
        layer_offset = self.layer_offsets[layer_idx]
        addr_key = self.impl.get_mem_pool_base_address(layer_offset, Role.KEY)
        if self.kv_cache_type != CacheTypeCpp.SELFKONLY:
            addr_value = self.impl.get_mem_pool_base_address(
                layer_offset, Role.VALUE)
            page_size_key = self.impl.get_page_stride(layer_offset, Role.KEY)
            page_size_value = self.impl.get_page_stride(layer_offset,
                                                        Role.VALUE)

            assert addr_key + page_size_value == addr_value and page_size_key == page_size_value

        assert kv_layout in ["NHD",
                             "HND"], f"Unsupported kv_layout: {kv_layout}"

        if kv_layout == "NHD":
            shape = [
                self.impl.get_page_index_upper_bound(layer_offset, Role.KEY) //
                self.kv_factor,
                self.kv_factor,
                self.tokens_per_block,
                self.num_kv_heads_per_layer[layer_offset],
                self.head_dim,
            ]
        else:
            shape = [
                self.impl.get_page_index_upper_bound(layer_offset, Role.KEY) //
                self.kv_factor,
                self.kv_factor,
                self.num_kv_heads_per_layer[layer_offset],
                self.tokens_per_block,
                self.head_dim,
            ]

        return convert_to_torch_tensor(
            TensorWrapper(
                addr_key,
                self.dtype,
                shape,
            ))

    def get_num_available_tokens(self,
                                 *,
                                 batch_size: int = 1,
                                 max_num_draft_tokens: int = 0) -> int:
        if max_num_draft_tokens > 0:
            raise ValueError(
                "max_num_draft_tokens is not supported for KVCacheManagerV2")
        return int(
            self.impl.clamp_max_seq_len_for_mem(batch_size) *
            self.kv_cache_manager_py_config.max_util_for_resume
        ) - self.num_extra_kv_tokens - max_num_draft_tokens

    def get_num_free_blocks(self) -> int:
        # NOTE This method is used to get the number of blocks in the primary pool not the FREE blocks.
        # However, since we only use this function when the kv cache manager is empty, so it is safe to do so.
        assert len(
            self.kv_cache_map
        ) == 0, "get_num_free_blocks is only used when the kv cache manager is empty"
        max_num_pages = max([
            self.impl.get_page_index_upper_bound(layer_id, Role.KEY)
            for layer_id in typed_range(LayerId(self.num_local_layers))
        ])
        return max_num_pages // self.kv_factor

    @nvtx_range("prepare_resources_kv_cache_manager_v2")
    def prepare_resources(self, scheduled_batch: ScheduledRequests):
        with request_context(self.is_draft, scheduled_batch):
            context_batch = scheduled_batch.context_requests
            generation_batch = scheduled_batch.generation_requests
            # allocate KV Cache
            for req in context_batch:
                beam_width = req.sampling_config.beam_width
                if 'cp_type' in self.mapping.cp_config and CpType.STAR == self.mapping.cp_config[
                        'cp_type']:
                    raise RuntimeError(
                        "Star attention is not supported for kv cache manager v2"
                    )
                else:
                    if req.is_first_context_chunk and self._kv_connector_should_add_sequence(
                            req):
                        # Last token cannot be recovered, so we don't include it in the input tokens to look up for the block that can be reused.
                        kv_cache = self._create_kv_cache(
                            req.py_request_id, req.lora_task_id,
                            req.get_tokens(0)[:-1]
                            if self.enable_block_reuse else None)
                        assert beam_width == 1, "Currently, KVCacheManagerV2 only supports beam width 1"
                        if not self.enable_block_reuse:
                            assert kv_cache.num_committed_tokens == 0
                            kv_cache.stop_committing()
                        else:
                            req.context_current_position = kv_cache.num_committed_tokens
                            chunk_size = req.context_chunk_size
                            if req.context_current_position + req.context_chunk_size < req.prompt_len:
                                floored_end_position = (
                                    req.context_current_position +
                                    req.context_chunk_size
                                ) // self.tokens_per_block * self.tokens_per_block
                                chunk_size = floored_end_position - req.context_current_position

                            req.context_chunk_size = min(
                                chunk_size,
                                req.prompt_len - req.context_current_position)

                        success = kv_cache.resume(
                            torch.cuda.current_stream().cuda_stream)
                        assert success

                        kv_cache.resize(req.prompt_len)

                        if self.kv_connector_manager is not None:
                            block_ids = self.get_cache_indices(req)
                            self.kv_connector_manager.update_state_after_alloc(
                                req, block_ids)

            for req in generation_batch:
                kv_cache = self.kv_cache_map[req.py_request_id]
                kv_cache.resize(kv_cache.capacity + 1)

        if self.kv_connector_manager is not None:
            self.kv_connector_manager.build_scheduler_output(
                scheduled_batch, self)

    def _kv_connector_should_add_sequence(self, request: LlmRequest) -> bool:
        return self.kv_connector_manager is None or self.kv_connector_manager.should_add_sequence(
            request)

    def get_kv_cache_stats(self):

        class KVCacheStatus:

            def __init__(self, allocated_bytes: int):
                self.allocated_bytes = allocated_bytes

        return KVCacheStatus(allocated_bytes=self.impl.get_quota(GPU_LEVEL))

    def add_dummy_requests(
            self,
            request_ids: List[int],
            # Note that token_nums should be past_kv_len + input_len (without
            # spec decoding). The draft tokens will be added in this function,
            # so we don't need to take care of it in the caller. When preparing
            # token_nums, we should not take the draft tokens into account, so
            # don't use the kv_cache_manager.max_seq_len, which includes both
            # extra tokens and draft tokens.
            token_nums: Optional[List[int]] = None,
            is_gen: bool = False,
            prepare_resource: bool = True,
            max_num_draft_tokens: int = 0,
            use_mrope: bool = False,
            max_beam_width: int = 1,
            num_extra_decoding_steps:
        int = 0,  # TODO: support num_extra_decoding_steps
    ):

        beam_width = max_beam_width
        requests = []
        for i, req_id in enumerate(request_ids):
            # exact choice of n can be ignored for dummy requests
            sampling_params = SamplingParams(n=beam_width,
                                             best_of=beam_width,
                                             use_beam_search=beam_width > 1)
            # Here 1+max_num_draft_tokens is used to extend the prompt length to
            # a non-zero number to skip illegal memory access issue in MLA kernel
            # during warmup.
            token_num = token_nums[
                i] if token_nums is not None else 1 + max_num_draft_tokens
            # TODO: support cross attention
            encoder_input_tokens = None
            # Using 1 instead of 0 prevents NaN during warmup in e.g. Deepseek
            input_tokens = [1 for _ in range(token_num)]
            req = LlmRequest(request_id=req_id,
                             max_new_tokens=1,
                             input_tokens=input_tokens,
                             sampling_config=SamplingConfig(
                                 sampling_params._get_sampling_config()),
                             is_streaming=False,
                             encoder_input_tokens=encoder_input_tokens)
            req.is_dummy_request = True
            req.paged_kv_block_ids = []
            if prepare_resource:
                kv_cache = self._create_kv_cache(req.py_request_id,
                                                 req.lora_task_id, input_tokens)
                assert kv_cache.num_committed_tokens == 0
                success = kv_cache.resume(
                    torch.cuda.current_stream().cuda_stream)
                if not success:
                    for r in requests:
                        self.free_resources(r)
                    self.free_resources(req)
                    return None
                kv_cache.stop_committing()
                kv_cache.resize(token_num)

            if is_gen:
                req.state = LlmRequestState.GENERATION_IN_PROGRESS
                req.prompt_len = token_num - 1
                req.py_prompt_len = req.prompt_len

            # TODO: Planning to get dummy_data from each model. Before that, we need to add dummy mrop_config to the request here.
            if use_mrope:
                dummy_mrope_position_ids = torch.arange(
                    0, token_num, dtype=torch.int32).expand(3, 1, -1).clone()
                req.py_multimodal_data = {
                    "mrope_config": {
                        "mrope_position_ids": dummy_mrope_position_ids
                    }
                }
                if is_gen:
                    dummy_mrope_position_deltas = torch.zeros(
                        1, dtype=torch.int32).unsqueeze(0)
                    req.py_multimodal_data["mrope_config"][
                        "mrope_position_deltas"] = dummy_mrope_position_deltas
            requests.append(req)

        return requests

    def free_resources(self, request: LlmRequest, pin_on_release: bool = False):
        kv_cache = self.kv_cache_map.pop(request.py_request_id)
        kv_cache.close()
        self.index_mapper.remove_sequence(request.py_request_id)

    def get_batch_cache_indices(self,
                                request_ids: List[int],
                                layer_id: int = 0) -> List[List[int]]:

        return self._get_batch_cache_indices_by_pool_id(
            request_ids,
            pool_id=self.layer_to_pool_mapping_dict[layer_id],
            is_kv_aggregate=True)

    def _get_batch_cache_indices_by_pool_id(
            self,
            request_ids: List[int],
            *,
            pool_id: int = 0,
            is_kv_aggregate: bool = True) -> List[List[int]]:

        if is_kv_aggregate:
            # Div by kv_factor to index kv cache with size [num_blocks, kv_factor, tokens_per_block, num_kv_heads, head_dim]
            div_factor = self.kv_factor
        else:
            div_factor = 1

        return [
            (np.asarray(self.kv_cache_map[req_id].get_page_indices(pool_id)) //
             div_factor).tolist() for req_id in request_ids
        ]

    def get_cache_bytes_per_token(
            self,
            local_layer_idx: Optional[int] = None,
            data_role: Role = Role.ALL):  # None means all layers/data_roles
        if self.dtype not in (
                DataType.FP8,
                DataType.HALF,
                DataType.BF16,
                DataType.FLOAT,
                DataType.NVFP4,
        ):
            raise ValueError(f"Cannot support {self.dtype} KV cache.")

        if data_role == Role.ALL:
            kv_factor = self.kv_factor
        elif data_role in [
                Role.KEY, Role.VALUE, Role.KEY_BLOCK_QUANT,
                Role.VALUE_BLOCK_QUANT
        ]:
            if data_role in [Role.KEY_BLOCK_QUANT, Role.VALUE_BLOCK_QUANT]:
                assert self.dtype == DataType.NVFP4, "NVFP4 is the only supported dtype for block quant data roles"
            if data_role == Role.VALUE:
                assert self.kv_cache_type != CacheTypeCpp.SELFKONLY, "SELFKONLY is the only supported cache type for value data role"
            kv_factor = 1
        else:
            raise ValueError(f"Invalid data role: {data_role}")

        if local_layer_idx is None:
            cache_size_per_token = (kv_factor *
                                    sum(self.num_kv_heads_per_layer) *
                                    self.head_dim)
        else:
            cache_size_per_token = (
                kv_factor * self.num_kv_heads_per_layer[local_layer_idx] *
                self.head_dim)

        cache_size_bytes_per_token = get_size_in_bytes(cache_size_per_token,
                                                       self.dtype)

        if data_role in [Role.KEY, Role.VALUE]:
            return cache_size_bytes_per_token

        quant_size_per_token = 0

        if self.dtype == DataType.NVFP4:
            quant_size_per_token = self.calculate_scaling_factor_size_bytes(
                cache_size_per_token,
                quant_vector_size=16,
                scaling_factor_dtype=DataType.FP8,
            )

        if data_role in [Role.KEY_BLOCK_QUANT, Role.VALUE_BLOCK_QUANT]:
            return quant_size_per_token

        return cache_size_bytes_per_token + quant_size_per_token

    @staticmethod
    def calculate_scaling_factor_size_bytes(
            cache_size: int, quant_vector_size: int,
            scaling_factor_dtype: DataType) -> int:
        assert cache_size % quant_vector_size == 0, "NVFP4 cache size must be divisible by quant vector size"
        return get_size_in_bytes(cache_size // quant_vector_size,
                                 scaling_factor_dtype)

    def check_invalid_values_in_kv_cache(self,
                                         fill_with_zero: bool = False) -> bool:
        some_checks_unavailable = False
        has_invalid_values = torch.tensor([False],
                                          dtype=torch.bool,
                                          device=torch.cuda.current_device())
        pool_handled = set()

        # Handle each layer from start to end to traverse the whole KV cache.
        for layer_id in typed_range(LayerId(self.num_local_layers)):
            pool_id = self.layer_to_pool_mapping_dict[layer_id]
            if pool_id in pool_handled:
                continue
            buffer = self.get_buffers(layer_id)
            # process in chunks of 256 pages to avoid OoM
            for i in range(0, buffer.shape[0], 256):
                buffer_slice = buffer[i:i + 256]
                try:
                    has_invalid_values.logical_or_(
                        torch.isnan(buffer_slice).any())
                    has_invalid_values.logical_or_(
                        torch.isinf(buffer_slice).any())
                except NotImplementedError:
                    some_checks_unavailable = True
            if fill_with_zero:
                buffer.zero_()
            pool_handled.add(pool_id)
        torch.cuda.synchronize()

        if some_checks_unavailable:
            logger.warning(
                "`torch.isnan` or `torch.isinf` is not implemented for current kv cache dtype, related checks are skipped"
            )
        return bool(has_invalid_values)

    def shutdown(self):
        for kv_cache in self.kv_cache_map.values():
            kv_cache.close()
        self.kv_cache_map.clear()
        self.impl.clear_reusable_blocks()

    def get_max_resource_count(self) -> int:
        # TODO: implement this
        return 1

    def get_needed_resource_to_completion(self, request: LlmRequest) -> int:
        # TODO: implement this
        # context_token_count = request.orig_prompt_len
        # num_context_blocks = context_token_count // self.tokens_per_block
        # remaining_tokens = context_token_count + request.max_new_tokens - num_context_blocks * self.tokens_per_block
        # need_blocks = num_context_blocks + math.ceil(
        #     remaining_tokens / self.tokens_per_block)
        # return need_blocks
        return 0

    # TODO: refactor get_cache_size_per_token and get_cache_bytes_per_token to use the same logic
    @staticmethod
    def get_cache_size_per_token(model_config: ModelConfigPython,
                                 mapping: Mapping, **kwargs):
        # get kv cache dtype bytes
        mem_per_token = 2
        quant_config = model_config.quant_config
        if quant_config is not None and quant_config.quant_mode.has_fp8_kv_cache(
        ):
            mem_per_token = 1

        # get num key value heads
        config = model_config.pretrained_config
        num_key_value_heads = getattr(config, 'num_key_value_heads',
                                      config.num_attention_heads)
        if isinstance(num_key_value_heads, Iterable):
            num_key_value_heads = sum(num_key_value_heads) / len(
                num_key_value_heads)

        # get head dim
        mla = hasattr(config, "kv_lora_rank")
        if mla:
            head_dim = config.kv_lora_rank + config.qk_rope_head_dim
            kv_factor = 1
        else:
            tp_size = 1 if mapping.enable_attention_dp else mapping.tp_size
            head_dim = getattr(config, "head_dim", None)
            if not isinstance(head_dim, int):
                head_dim = config.hidden_size // config.num_attention_heads
            head_dim = head_dim * num_key_value_heads // tp_size
            kv_factor = 2

        # provide at least 1 layer to prevent division by zero cache size
        num_attention_layers = max(
            len(mapping.pp_layers(model_config.get_num_attention_layers())), 1)
        mem_per_token *= num_attention_layers * head_dim

        # K and V
        mem_per_token *= kv_factor
        return mem_per_token

    def update_resources(self,
                         scheduled_batch: ScheduledRequests,
                         attn_metadata: "AttentionMetadata" = None,
                         kv_cache_dtype_byte_size: float = None):
        for req in scheduled_batch.context_requests:
            if req.py_request_id not in self.kv_cache_map:
                continue
            kv_cache = self.kv_cache_map[req.py_request_id]
            if self.enable_block_reuse and not req.is_dummy_request:
                if req.context_current_position > kv_cache.num_committed_tokens:
                    kv_cache.commit(
                        req.get_tokens(0)[kv_cache.num_committed_tokens:req.
                                          context_current_position])
                kv_cache.stop_committing()
            else:
                kv_cache.resize(None, req.context_current_position)

        for req in scheduled_batch.generation_requests:
            if req.py_request_id not in self.kv_cache_map:
                continue
            kv_cache = self.kv_cache_map[req.py_request_id]
            kv_cache.resize(None, req.max_beam_num_tokens - 1)

    def copy_batch_block_offsets(self, dst_tensor: torch.Tensor,
                                 request_ids: List[int], beam_width: int,
                                 num_contexts: int, num_seqs: int):
        assert beam_width == 1, "beam_width must be 1 for KVCacheManagerV2"

        copy_idx = self.index_mapper.get_copy_index(request_ids, num_contexts,
                                                    beam_width)
        assert copy_idx.shape[0] == num_seqs

        copy_batch_block_offsets_to_device(
            self.host_kv_cache_block_offsets, dst_tensor, copy_idx,
            self.kv_cache_type == CacheTypeCpp.SELFKONLY,
            torch.cuda.current_stream().cuda_stream)

    def _create_kv_cache(self, request_id: int, lora_task_id: int | None,
                         input_tokens: Sequence[TokenIdExt] | None):
        assert request_id not in self.kv_cache_map, f"KV cache for request {request_id} already exists"
        kv_cache = self.impl.create_kv_cache(lora_task_id, input_tokens)
        self.kv_cache_map[request_id] = kv_cache
        index = self.index_mapper.add_new_sequence(request_id)
        for i in range(self.max_beam_width):
            for pool_idx in range(self.num_pools):
                buffer: torch.Tensor = self.host_kv_cache_block_offsets[
                    pool_idx, index * self.max_beam_width + i, 0]
                kv_cache.set_page_index_buf(i, pool_idx,
                                            memoryview(buffer.numpy()))
        return kv_cache


class SlotManager:

    def __init__(self, max_num_requests: int):
        self.max_num_requests = max_num_requests
        self.slot_mapping = dict()
        self.free_slots = set(range(max_num_requests))

    def get_slot(self, request_id: int):
        return self.slot_mapping.get(request_id, None)

    def fill_slot_id_tensor(self, requests: List[LlmRequest],
                            slot_id_tensor: torch.Tensor):
        for i, request in enumerate(requests):
            slot_id = self.get_slot(request.request_id)
            if slot_id is not None:
                slot_id_tensor[i] = slot_id
            else:
                raise ValueError(f"Request {request.request_id} has no slot id")

    def add_slot(self, request_id: int):
        if request_id in self.slot_mapping:
            # CUDA graph dummy request could be added for different batches,
            # but we only need to reserve slot for it once.
            from .cuda_graph_runner import CUDA_GRAPH_DUMMY_REQUEST_ID
            assert request_id == CUDA_GRAPH_DUMMY_REQUEST_ID
            return self.slot_mapping[request_id]

        if len(self.free_slots) == 0:
            raise ValueError("No free slots")
        slot = self.free_slots.pop()
        self.slot_mapping[request_id] = slot
        return slot

    def remove_slot(self, request_id: int):
        if request_id in self.slot_mapping:
            slot = self.slot_mapping.pop(request_id)
            self.free_slots.add(slot)

    def shutdown(self):
        req_ids_list = list(self.slot_mapping.keys())
        for rid in req_ids_list:
            self.remove_slot(rid)
        assert len(self.slot_mapping) == 0 and len(
            self.free_slots) == self.max_num_requests


class BlockManager:

    def __init__(self, num_blocks: int, tokens_per_block: int):
        self.num_blocks = num_blocks
        self.tokens_per_block = tokens_per_block
        self.max_blocks_per_seq = self.num_blocks

        self.base_block_offsets = torch.arange(self.num_blocks,
                                               device="cpu",
                                               dtype=torch.int32)

        self.block_ids = dict()
        self.num_sequences = dict()
        self.free_blocks = deque(range(self.num_blocks))

    def add_tokens(self, request_id: int, num_tokens: int):
        if num_tokens > 0:
            if request_id not in self.block_ids:
                self.block_ids[request_id] = []
                self.num_sequences[request_id] = num_tokens
            else:
                self.num_sequences[request_id] += num_tokens
            block_count_needed = self.compute_block_count(
                self.num_sequences[request_id], self.tokens_per_block)
            if len(self.block_ids[request_id]) < block_count_needed:
                new_blocks = self._allocate_blocks(
                    block_count_needed - len(self.block_ids[request_id]))
                self.block_ids[request_id].extend(new_blocks)

    def copy_block_offsets(self, request_ids: List[int],
                           block_offsets: torch.Tensor) -> None:
        for i in range(len(request_ids)):
            block_ids = self.block_ids[request_ids[i]]
            block_num = len(block_ids)
            block_offsets[i, 0:block_num].copy_(
                self.base_block_offsets[torch.tensor(block_ids,
                                                     dtype=torch.int32,
                                                     device="cpu")])

    def compute_block_count(self, token_count: int,
                            tokens_per_page: int) -> int:
        return (token_count + tokens_per_page - 1) // tokens_per_page

    def free_resources(self, request: LlmRequest):
        request_id = request.py_request_id
        self._free_blocks(self.block_ids[request_id])
        del self.block_ids[request_id]
        del self.num_sequences[request_id]

    def rewind_cache(self, request: LlmRequest, rewind_len: int):
        if rewind_len == 0:
            return
        request_id = request.py_request_id
        self.num_sequences[request_id] -= rewind_len
        updated_token_num = max(self.num_sequences[request_id], 0)
        block_count_needed = self.compute_block_count(updated_token_num,
                                                      self.tokens_per_block)
        num_rewind_pages = len(self.block_ids[request_id]) - block_count_needed
        if num_rewind_pages > 0:
            self._free_blocks(self.block_ids[request_id][-num_rewind_pages:])
            self.block_ids[request_id] = self.block_ids[
                request_id][:-num_rewind_pages]
        return

    def _allocate_blocks(self, block_count: int) -> list:
        assert len(self.free_blocks) >= block_count, "Not enough blocks."
        blocks = [self.free_blocks.popleft() for _ in range(block_count)]
        return blocks

    def _free_blocks(self, block_list: list):
        self.free_blocks.extend(block_list)


class ResourceManager:

    def __init__(self, resource_managers: dict[ResourceManagerType,
                                               BaseResourceManager]):
        self.resource_managers = OrderedDict(resource_managers)

    def __call__(self, type: ResourceManagerType):
        return self.resource_managers[type]

    def register_resource_manager(self, type: ResourceManagerType,
                                  resource_manager: BaseResourceManager):
        self.resource_managers[type] = resource_manager

    def get_resource_manager(
            self, type: ResourceManagerType) -> Optional[BaseResourceManager]:
        return self.resource_managers.get(type)

    @nvtx_range("prepare_resources")
    def prepare_resources(self, scheduled_batch: ScheduledRequests):
        for _, resource_manager in self.resource_managers.items():
            if hasattr(resource_manager, "prepare_resources"):
                resource_manager.prepare_resources(scheduled_batch)

    @nvtx_range("update_resources")
    def update_resources(
        self,
        scheduled_batch: ScheduledRequests,
        attn_metadata: Optional["AttentionMetadata"] = None,
        kv_cache_dtype_byte_size: Optional[float] = None,
    ):
        for _, resource_manager in self.resource_managers.items():
            if hasattr(resource_manager, "update_resources"):
                if isinstance(resource_manager, KVCacheManager):
                    resource_manager.update_resources(scheduled_batch,
                                                      attn_metadata,
                                                      kv_cache_dtype_byte_size)
                else:
                    resource_manager.update_resources(scheduled_batch)

    def free_resources(self, request: LlmRequest):
        for resource_type, resource_manager in reversed(
                self.resource_managers.items()):
            if hasattr(resource_manager, "free_resources"):
                resource_manager.free_resources(request)

    def reorder_pipeline(self,
                         resource_manager_list: list[ResourceManagerType]):
        assert set(resource_manager_list) == set(self.resource_managers.keys())
        for resource_manager in resource_manager_list:
            self.resource_managers.move_to_end(resource_manager)


class PeftCacheManager(BaseResourceManager):

    def __init__(self,
                 peft_cache_config: PeftCacheConfig,
                 lora_config: LoraConfig,
                 model_config: ModelConfigCpp,
                 world_config: WorldConfig | None = None,
                 execution_stream: Optional[torch.cuda.Stream] = None):
        import tensorrt_llm.bindings as _tb

        peft_cache_config = peft_cache_config._to_pybind()

        peft_cache_manager_config = _tb.PeftCacheManagerConfig(
            num_host_module_layer=peft_cache_config.num_host_module_layer,
            num_device_module_layer=peft_cache_config.num_device_module_layer,
            optimal_adapter_size=peft_cache_config.optimal_adapter_size,
            max_adapter_size=peft_cache_config.max_adapter_size,
            num_put_workers=peft_cache_config.num_put_workers,
            num_ensure_workers=peft_cache_config.num_ensure_workers,
            num_copy_streams=peft_cache_config.num_copy_streams,
            max_pages_per_block_host=peft_cache_config.max_pages_per_block_host,
            max_pages_per_block_device=peft_cache_config.
            max_pages_per_block_device,
            device_cache_percent=peft_cache_config.device_cache_percent,
            host_cache_size=peft_cache_config.host_cache_size,
            lora_prefetch_dir=peft_cache_config.lora_prefetch_dir,
        )

        if world_config is None:
            world_config = _tb.WorldConfig()

        BufferManager = tensorrt_llm.bindings.internal.runtime.BufferManager
        buffer_manager_stream = execution_stream.cuda_stream if execution_stream is not None else torch.cuda.current_stream(
        ).cuda_stream
        buffer_manager = BufferManager(buffer_manager_stream, True)
        logger.info(
            f"[PeftCacheManager] buffer_manager_stream: {buffer_manager_stream}"
        )
        self.impl = PeftCacheManagerCpp(config=peft_cache_manager_config,
                                        model_config=model_config,
                                        world_config=world_config,
                                        buffer_manager=buffer_manager)
        self._lora_config = lora_config
        self._lora_model_config = LoraModelConfig(
            lora_config.lora_target_modules,
            lora_config.trtllm_modules_to_hf_modules, model_config.hidden_size,
            binding_to_str_dtype(model_config.data_type),
            lora_config.swap_gate_up_proj_lora_b_weight)
        mapping = Mapping(
            world_size=world_config.size,
            rank=world_config.rank,
            tp_size=world_config.tensor_parallelism,
            pp_size=world_config.pipeline_parallelism,
            gpus_per_node=world_config.gpus_per_node,
        )
        self._lora_manager = LoraManager(
            mapping=mapping,
            model_config=ModelConfigPython.from_model_config_cpp(model_config),
            cpp_peft_cache_manager=self.impl)

        self._batch_peft_table: Optional[Dict[int, list[
            TaskLayerModuleConfig]]] = None  # task_id -> layer-module-configs mapping for the current batch

    def get_lora_manager(self) -> LoraManager:
        return self._lora_manager

    def add_request_peft(self, request: LlmRequest):
        if request.lora_task_id is not None:
            is_task_cached = self.impl.is_task_cached(request.lora_task_id)
            if is_task_cached:
                # PeftCacheManager::addRequestPeft in CPP doesn't allow having only one of [config tensor, weights
                # tensor] without the other. Since there's no need for any of them when the LoRA adapter is already
                # cached, we can safely remove both from the request.
                request.remove_lora_tensors()
            elif request.lora_weights is None and request.py_lora_path:
                self._lora_manager.load_from_ckpt(
                    [request.py_lora_path],
                    model_config=self._lora_model_config,
                    uids=[request.lora_task_id],
                    ckpt_source=self._lora_config.lora_ckpt_source)
                request.lora_weights = self._lora_manager.cpp_lora_weights[
                    request.lora_task_id]

            # PeftCacheManager CPP implementation expects an extra dim at index 0
            if request.lora_weights is not None:
                request.lora_weights = request.lora_weights.unsqueeze(0)
            if request.lora_config is not None:
                request.lora_config = request.lora_config.unsqueeze(0)
        self.impl.add_request_peft(request, True)

    def ensure_batch(self,
                     context_batch: List[LlmRequest],
                     generation_batch: List[LlmRequest],
                     reset_gpu_cache: bool = False) -> List[LlmRequest]:
        return self.impl.ensure_batch(context_batch, generation_batch,
                                      reset_gpu_cache)

    def get_max_resource_count(self) -> int:
        return 0

    def get_needed_resource_to_completion(self, request: LlmRequest) -> int:
        return 0

    def prepare_resources(self, scheduled_batch: ScheduledRequests):
        context_batch = scheduled_batch.context_requests
        generation_batch = scheduled_batch.generation_requests
        for req in context_batch:
            self.add_request_peft(req)

        self._batch_peft_table, _ = self.impl.ensure_batch_map_task_id(
            context_batch, generation_batch, False)

    def update_resources(self, scheduled_batch: ScheduledRequests):
        pass

    def free_resources(self, request: LlmRequest):
        self.impl.mark_request_done(request)

    def shutdown(self):
        pass

    def get_and_reset_batch_peft_table(
            self) -> Dict[int, list[TaskLayerModuleConfig]]:
        batch_peft_table = self._batch_peft_table
        self._batch_peft_table = None
        return batch_peft_table

    def is_task_cached_device(self, task_id: int) -> bool:
        return self.impl.is_task_cached_device(task_id)
