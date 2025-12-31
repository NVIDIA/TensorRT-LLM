# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
import types
from collections import defaultdict
from dataclasses import dataclass
from types import MethodType, SimpleNamespace
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from strenum import StrEnum
from torch._prims_common import DeviceLikeType

from tensorrt_llm._torch.attention_backend.interface import AttentionRuntimeFeatures
from tensorrt_llm._torch.auto_deploy.utils._graph import get_input_embeddings, get_lm_head_weights
from tensorrt_llm._torch.models.modeling_speculative import Eagle3ForCausalLM
from tensorrt_llm._torch.pyexecutor._util import (
    _create_kv_cache_manager,
    get_decoding_mode,
    get_kv_cache_manager_cls,
)
from tensorrt_llm._torch.pyexecutor.cuda_graph_runner import CUDA_GRAPH_DUMMY_REQUEST_ID
from tensorrt_llm._torch.pyexecutor.guided_decoder import GuidedDecoder
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest, get_draft_token_length
from tensorrt_llm._torch.pyexecutor.py_executor_creator import get_guided_decoding_config
from tensorrt_llm._torch.pyexecutor.seq_slot_manager import SeqSlotManager
from tensorrt_llm._torch.speculative import get_spec_drafter
from tensorrt_llm._torch.speculative.eagle3 import Eagle3ResourceManager
from tensorrt_llm._utils import nvtx_range
from tensorrt_llm.llmapi.llm_args import (
    ContextChunkingPolicy,
    EagleDecodingConfig,
    LoadFormat,
    SamplerType,
    TorchLlmArgs,
)
from tensorrt_llm.llmapi.tokenizer import TokenizerBase

from ...._utils import mpi_rank, mpi_world_size
from ....bindings.internal.batch_manager import CacheType
from ....mapping import Mapping
from ...distributed import MPIDist
from ...pyexecutor.model_engine import ModelEngine, PyTorchModelEngine
from ...pyexecutor.py_executor import PyExecutor
from ...pyexecutor.resource_manager import (
    BaseResourceManager,
    KVCacheManager,
    ResourceManager,
    ResourceManagerType,
)
from ...pyexecutor.sampler import TorchSampler, TRTLLMSampler
from ...pyexecutor.scheduler import (
    BindCapacityScheduler,
    BindMicroBatchScheduler,
    RequestList,
    ScheduledRequests,
    SimpleScheduler,
)
from ..custom_ops.attention_interface import SequenceInfo
from ..distributed import common as dist
from ..llm_args import LlmArgs
from ..transform.optimizer import InferenceOptimizer
from ..utils.logger import ad_logger
from .interface import CachedSequenceInterface, GetInferenceModel


@dataclass
class ReportingInfo:
    print_log: bool = False
    enable_iter_perf_stats: bool = False
    enable_iter_req_stats: bool = False


class _CacheManagerWithFakePool(KVCacheManager):
    """We use the default KVCacheManager but with a fake pool by setting head_dim=0.

    The actual cache pools are managed by auto_deploy layerwise cache pools.
    """

    def __init__(
        self,
        kv_cache_config,
        num_blocks: int,
        tokens_per_block: int,
        max_seq_len: int,
        max_batch_size: int,
    ):
        self.num_blocks = num_blocks
        super().__init__(
            kv_cache_config=kv_cache_config,
            kv_cache_type=CacheType.SELF,
            num_layers=1,
            num_kv_heads=1,
            head_dim=0,
            tokens_per_block=tokens_per_block,
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
            mapping=Mapping(),
        )

    def calculate_max_num_blocks(
        self, kv_cache_config, head_dim, tokens_per_block, mapping, dtype, kv_factor
    ) -> Tuple[int, int]:
        """Calculate the maximum number of blocks needed for the cache."""
        # TODO: this is VERY hacky... Ideally, we want to compute the number of blocks
        # just like in the original implementation. However, let's wait for the layer-wise attention
        # implementation before over-optimizing the function here
        ad_logger.info("Using fake cache manager with head_dim=0 and num pages:", self.num_blocks)
        return self.num_blocks, 0


class ADHiddenStateManager(Eagle3ResourceManager):
    def __init__(
        self,
        cache_seq_interface: CachedSequenceInterface,
        config: EagleDecodingConfig,
        max_num_requests: int,
        max_seq_len: int,
        max_num_tokens: int,
    ):
        hidden_state_buffer = self._get_hidden_state_buffers(cache_seq_interface)[0]
        dtype = hidden_state_buffer.dtype
        hidden_size = hidden_state_buffer.shape[1]

        super().__init__(config, dtype, hidden_size, max_num_requests, max_seq_len, max_num_tokens)

        self.hidden_state_write_indices: torch.Tensor = torch.empty(
            max_num_tokens, dtype=torch.long, device="cuda"
        )

    def _get_hidden_state_buffers(
        self, cache_seq_interface: CachedSequenceInterface
    ) -> List[torch.Tensor]:
        hidden_state_buffers = []
        for name, tensor in cache_seq_interface.named_args.items():
            if "hidden_states_cache" in name:
                hidden_state_buffers.append(tensor)

        if not hidden_state_buffers:
            raise ValueError(
                "No hidden_state_buffers found in cache_seq_interface. Check if we are actually running Eagle3."
            )
        return hidden_state_buffers

    def prepare_hidden_states_capture(
        self, ordered_requests: RequestList, cache_seq_interface: CachedSequenceInterface
    ) -> None:
        """Prepare the hidden states for capture by establishing indices that the hidden states will be written to."""
        seq_lens = cache_seq_interface.info.seq_len
        num_tokens = sum(seq_lens)

        start_idx = 0
        hidden_states_write_indices = []
        for request, seq_len in zip(ordered_requests, seq_lens):
            request_id = request.request_id
            slot_id = self.slot_manager.get_slot(request_id)
            self.start_indices[slot_id] = start_idx
            hidden_states_write_indices.extend(range(start_idx, start_idx + seq_len))
            start_idx += max(seq_len, self.max_total_draft_tokens + 1)
            assert start_idx < self.hidden_states.shape[0], (
                f"start_idx {start_idx} exceeds hidden_states capacity {self.hidden_states.shape[0]}"
            )

        if len(hidden_states_write_indices) != num_tokens:
            raise ValueError(
                f"len(hidden_state_write_indices) ({len(hidden_states_write_indices)}) != num_tokens \
                ({num_tokens}). Check whether ordered_requests matches up with seq_lens."
            )

        hidden_state_write_indices_host = torch.tensor(
            hidden_states_write_indices, dtype=torch.long
        )

        self.hidden_state_write_indices[:num_tokens].copy_(
            hidden_state_write_indices_host, non_blocking=True
        )

    def capture_hidden_states(self, cache_seq_interface: CachedSequenceInterface) -> None:
        """Capture configured hidden states that have been written by the model,
        in a format that can be used by the draft model.
        """
        full_hidden_states = self._get_hidden_state_buffers(cache_seq_interface)
        if not full_hidden_states:
            return

        num_tokens = sum(cache_seq_interface.info.seq_len)

        hidden_states = [hidden_state[:num_tokens] for hidden_state in full_hidden_states]
        hidden_states = torch.cat(hidden_states, dim=1)
        hidden_states = hidden_states.to(dtype=self.dtype)

        token_idx = self.hidden_state_write_indices[:num_tokens]
        self.hidden_states[:, : hidden_states.shape[1]].index_copy_(0, token_idx, hidden_states)


def construct_draft_llm_args(
    ad_config: LlmArgs,
) -> TorchLlmArgs:
    """Construct a TorchLlmArgs for the draft model from AutoDeploy config.

    Args:
        ad_config: The AutoDeploy LLM configuration

    Returns:
        A TorchLlmArgs instance suitable for creating a PyTorchModelEngine
    """
    # Extract common fields as a dict
    common_fields = {
        "model": ad_config.model,
        "tokenizer": ad_config.tokenizer,
        "max_batch_size": ad_config.max_batch_size,
        "max_seq_len": ad_config.max_seq_len,
        "max_beam_width": ad_config.max_beam_width,
        "max_num_tokens": ad_config.max_num_tokens,
        "max_input_len": ad_config.max_input_len,
        "kv_cache_config": ad_config.kv_cache_config,
        "enable_chunked_prefill": ad_config.enable_chunked_prefill,
        "attn_backend": ad_config.attn_backend,
        "disable_overlap_scheduler": ad_config.disable_overlap_scheduler,
        "speculative_config": ad_config.speculative_config,
        "checkpoint_loader": getattr(ad_config, "draft_checkpoint_loader", None),
    }

    # Add other fields that may exist in ad_config
    optional_fields = [
        "dtype",
        "trust_remote_code",
        "sparse_attention_config",
        "lora_config",
        "scheduler_config",
        "garbage_collection_gen0_threshold",
        "skip_tokenizer_init",
        "tokenizer_mode",
        "revision",
        "tokenizer_revision",
        "tensor_parallel_size",
        "pipeline_parallel_size",
        "context_parallel_size",
        "gpus_per_node",
        "enable_lora",
        "guided_decoding_backend",
        "peft_cache_config",
        "cache_transceiver_config",
        "decoding_config",
    ]

    for field in optional_fields:
        if hasattr(ad_config, field):
            value = getattr(ad_config, field)
            if value is not None:  # Only add if not None
                common_fields[field] = value

    draft_llm_args = TorchLlmArgs(**common_fields)

    # Handle load_format separately
    if ad_config.speculative_config.load_format == "dummy":
        draft_llm_args.load_format = LoadFormat.DUMMY

    draft_llm_args.tensor_parallel_size = ad_config.world_size

    return draft_llm_args


def create_draft_kv_cache_manager_maybe(
    draft_model_engine: Optional[PyTorchModelEngine],
    ad_config: LlmArgs,
    dist_mapping: Mapping,
) -> Optional[KVCacheManager]:
    if draft_model_engine is None or not draft_model_engine.model.model_config.is_generation:
        return None

    # Get the appropriate KV cache manager class
    kv_cache_manager_cls = get_kv_cache_manager_cls(draft_model_engine.model.model_config)

    return _create_kv_cache_manager(
        model_engine=draft_model_engine,
        kv_cache_manager_cls=kv_cache_manager_cls,
        mapping=dist_mapping,
        kv_cache_config=ad_config.kv_cache_config,
        tokens_per_block=ad_config.attn_page_size,
        max_seq_len=ad_config.max_seq_len,
        max_batch_size=ad_config.max_batch_size,
        spec_config=ad_config.speculative_config,
        sparse_attn_config=ad_config.sparse_attention_config,
        max_num_tokens=ad_config.max_num_tokens,
        max_beam_width=ad_config.max_beam_width,
        kv_connector_manager=None,  # KV connector manager not used in AutoDeploy (no disagg support)
        estimating_kv_cache=False,
    )


def _round_up_to_closest(batch_sizes: List[int], bs: int) -> Optional[int]:
    """Return closest batch size larger or equal to bs."""
    if bs > max(batch_sizes, default=0):
        return None
    return min(batch_sizes, key=lambda x: (x < bs, abs(x - bs)), default=None)


def _generate_dummy_request(
    resource_manager: ResourceManager, request_id: int, **request_kwargs
) -> Optional[LlmRequest]:
    # get resource managers we want
    kv_cache_manager: KVCacheManager = resource_manager.get_resource_manager(
        ResourceManagerType.KV_CACHE_MANAGER
    )
    slot_manager: SeqSlotManager = resource_manager.get_resource_manager(
        ResourceManagerType.SEQ_SLOT_MANAGER
    )
    spec_res_mgr: Optional[BaseResourceManager] = resource_manager.get_resource_manager(
        ResourceManagerType.SPEC_RESOURCE_MANAGER
    )

    # check if we have a free slot available and free page available
    if not slot_manager.slot_manager.free_slots or kv_cache_manager.get_num_free_blocks() == 0:
        return None

    # generate a dummy request
    dummy_request = kv_cache_manager.add_dummy_requests([request_id], **request_kwargs)[0]
    dummy_request.is_cuda_graph_dummy = True

    # add to spec resource manager
    if spec_res_mgr:
        spec_res_mgr.add_dummy_requests([request_id])

    # TODO: https://github.com/NVIDIA/TensorRT-LLM/issues/9883 clean up this hack
    dummy_request.seq_slot = slot_manager.get_max_resource_count()
    dummy_request.py_seq_slot = dummy_request.seq_slot

    return dummy_request


def maybe_pad_for_cuda_graph(func):
    def wrapper(
        self: "ADEngine",
        scheduled_requests: ScheduledRequests,
        resource_manager: ResourceManager,
        *args,
        **kwargs,
    ):
        def _call_func():
            return func(self, scheduled_requests, resource_manager, *args, **kwargs)

        # check if we use cuda graph and we can run it
        if not (self.cuda_graph_used and scheduled_requests.can_run_cuda_graph):
            return _call_func()

        # generate a persistent dummy request right away to ensure we can reserve the necessary
        # resources (kv page and slot)
        if self.padding_dummy_request is None:
            self.padding_dummy_request = _generate_dummy_request(
                resource_manager,
                request_id=CUDA_GRAPH_DUMMY_REQUEST_ID,
                is_gen=True,
                max_num_draft_tokens=self.max_total_draft_tokens,
                use_mrope=False,
                max_beam_width=self.max_beam_width,
            )

        # check closest cuda graph batch size
        closest_cg_bs = _round_up_to_closest(
            self.cuda_graph_batch_sizes, scheduled_requests.batch_size
        )

        # check if we need to pad
        num_padding = closest_cg_bs - scheduled_requests.batch_size

        if num_padding <= 0:
            return _call_func()

        # check if we have a dummy request to use
        if self.padding_dummy_request is None:
            ad_logger.error("No CUDA graph padding possible due to missing dummy request.")
            return _call_func()

        # pad the scheduled requests with the dummy request
        scheduled_requests.generation_requests.extend([self.padding_dummy_request] * num_padding)

        ret = _call_func()

        # truncate requests to remove the dummy requests we added
        scheduled_requests.generation_requests = scheduled_requests.generation_requests[
            :-num_padding
        ]

        return ret

    return wrapper


class ADEngine(ModelEngine):
    """The AutoDeploy Engine (ADEngine) is the main engine interface to execute AutoDeploy models.

    It follows the ``ModelEngine`` abstractions and is responsible for building the ad-optimized
    model, converting TRT-LLM scheduled requests into ad-native (pytorch-native) inputs, running
    the model, and returning correctly formatted logits.
    """

    @property
    def _device(self) -> DeviceLikeType:
        return self.cache_seq_interface.device

    @classmethod
    def build_from_config(cls, ad_config: LlmArgs, mapping: Optional[Mapping] = None):
        """Build the ADEngine using the LlmArgs that gets passed through from the LLM."""

        max_batch_size = ad_config.max_batch_size
        max_seq_len = ad_config.max_seq_len
        attn_page_size = ad_config.attn_page_size
        max_num_tokens = ad_config.max_num_tokens

        # update device to contain the current default device if it's in cuda
        device = torch.device(ad_config.device)
        if device.type == "cuda" and device.index is None:
            device = torch.device(f"cuda:{torch.cuda.current_device()}")
        device = str(device)

        factory = ad_config.create_factory()

        # initialize seq info object
        seq_info = SequenceInfo(
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
            page_size=attn_page_size,
            max_num_tokens=max_num_tokens,
            vocab_size_padded=factory.vocab_size_padded,
        )
        reporting_info = ReportingInfo(
            print_log=False,
            enable_iter_perf_stats=ad_config.enable_iter_perf_stats,
            enable_iter_req_stats=ad_config.enable_iter_req_stats,
        )
        # TODO (lucaslie): consider how we move args around InferenceOptimizer.__init__,
        # ADEngine.__init__, and ADEngine.build_from_config. Seems a bit unnatural atm.

        # construct inference optimizer
        build_and_optimize = InferenceOptimizer(factory=factory, config=ad_config.transforms)

        # construct engine
        return cls(
            build_and_optimize,
            seq_info,
            device,
            ad_config=ad_config,
            mapping=mapping,
            reporting_info=reporting_info,
        )

    @torch.inference_mode()
    def __init__(
        self,
        get_inference_model: GetInferenceModel,
        seq_info: SequenceInfo,
        device: DeviceLikeType,
        ad_config: Optional[LlmArgs] = None,
        mapping: Optional[Mapping] = None,
        reporting_info: ReportingInfo = ReportingInfo(),
    ) -> None:
        """Initialize the engine with model and sequence information."""
        # NOTE (lucaslie): create a fake Namespace to satisfy PyExecutor requirements...
        # This is not correctly declared in the base ModelEngine class though...
        self.llm_args = SimpleNamespace()
        self.llm_args.print_iter_log = reporting_info.print_log
        self.llm_args.enable_iter_perf_stats = reporting_info.enable_iter_perf_stats
        self.llm_args.enable_iter_req_stats = reporting_info.enable_iter_req_stats
        self.llm_args.stream_interval = 1
        self.llm_args.attention_dp_config = None
        self.llm_args.batch_wait_timeout_ms = 0
        self.llm_args.batch_wait_timeout_iters = 0
        self.llm_args.batch_wait_max_tokens_ratio = 0.0
        self.llm_args.max_num_tokens = seq_info.max_num_tokens
        self.llm_args.max_seq_len = seq_info.max_seq_len
        self.iter_counter = 0
        self.iter_states = {}

        # NOTE (lucaslie): not a declared base member in the base class; required by PyExecutor...
        self.enable_attention_dp = False

        if ad_config is not None:
            self.max_beam_width = ad_config.max_beam_width
            self.spec_config = ad_config.speculative_config
            self._disable_overlap_scheduler = ad_config.disable_overlap_scheduler
        else:
            self.max_beam_width = 1
            self.spec_config = None
            self._disable_overlap_scheduler = False

        # check for max total draft tokens
        if self.spec_config is not None:
            self.max_total_draft_tokens = self.spec_config.max_total_draft_tokens
        else:
            self.max_total_draft_tokens = 0

        # TODO(govind): Enable overlap scheduler for speculation.
        assert self.spec_config is None or self._disable_overlap_scheduler, (
            "Overlap scheduler is not supported \
            for speculative decoding in AutoDeploy."
        )

        # For compatibility with PyTorchModelEngine utilities
        self.batch_size = seq_info.max_batch_size

        # construct cache sequence interface
        self.cache_seq_interface = CachedSequenceInterface(
            sequence_info=seq_info,
            device=device,
        )

        # build model
        self.model = get_inference_model(self.cache_seq_interface)
        # start fresh with fixed seed
        torch.manual_seed(42)

        # check cuda graph padding...
        # TODO: better mechanism to retrieve this information when we refactor LlmArgs
        if ad_config is None:
            self.cuda_graph_used = False
            self.cuda_graph_batch_sizes = []
        else:
            self.cuda_graph_used = ad_config.is_cuda_graph_enabled()
            self.cuda_graph_batch_sizes = ad_config.cuda_graph_batch_sizes

        # keep a reference for one dummy request around
        self.padding_dummy_request: Optional[LlmRequest] = None

        # Reuse _execute_logit_post_processors from PyTorchModelEngine
        self.mapping = mapping
        self._execute_logit_post_processors = types.MethodType(
            PyTorchModelEngine._execute_logit_post_processors, self
        )

    @nvtx_range("ad_prepare_inputs")
    def _prepare_inputs(
        self,
        scheduled_requests: ScheduledRequests,
        resource_manager: ResourceManager,
        new_tokens: Optional[torch.Tensor] = None,
        gather_context_logits: bool = False,
    ) -> None:
        """Prepare inputs for AD Model from scheduled requests."""
        # cache manager
        kv_cache_manager = resource_manager.get_resource_manager(
            ResourceManagerType.KV_CACHE_MANAGER
        )
        # resource manager for hidden state capture
        spec_resource_manager = resource_manager.get_resource_manager(
            ResourceManagerType.SPEC_RESOURCE_MANAGER
        )

        # requests in order of context, generate
        context_requests = scheduled_requests.context_requests
        extend_requests = [
            r for r in scheduled_requests.generation_requests if get_draft_token_length(r) > 0
        ]
        generation_requests = [
            r for r in scheduled_requests.generation_requests if get_draft_token_length(r) == 0
        ]
        gen_requests = extend_requests + generation_requests
        ordered_requests = context_requests + gen_requests
        # info to be extracted
        input_ids: List[List[int]] = []
        position_ids: List[List[int]] = []
        input_pos: List[int] = []
        seq_len: List[int] = []
        cu_seqlen: List[int] = [0]
        cache_loc: List[int] = []
        pages_per_seq: List[int] = []
        cu_num_pages: List[int] = [0]
        seq_len_with_cache: List[int] = []
        last_page_len: List[int] = []
        slot_idx: List[int] = []
        use_initial_states: List[bool] = []

        # gather indices are used to gather tokens in new_tokens into input_ids
        flat_gather_indices: List[int] = []
        mask_scatter_indices: List[int] = []
        extra_args: Dict[str, List[torch.Tensor]] = defaultdict(list)

        # gather indices for logits
        logits_gather_indices: List[int] = []

        page_size = self.cache_seq_interface.info.page_size
        dummy_token = -1
        num_ctx_requests = len(context_requests)
        num_ctx_tokens = 0
        num_generation_tokens = 0

        # look at context requests first
        for request in context_requests:
            # store input ids and pos of first token in sequence
            # NOTE: begin_compute > 0 indicates block reuse
            # NOTE: end_compute will be used in the future for chunked prefill
            all_prompt_tokens = request.get_tokens(0)
            begin_compute = request.context_current_position
            end_compute = begin_compute + request.context_chunk_size
            prompt_tokens = all_prompt_tokens[begin_compute:end_compute]
            num_ctx_tokens += len(prompt_tokens)

            input_ids.append(prompt_tokens)
            input_pos.append(begin_compute)

            seq_len.append(len(input_ids[-1]))
            cu_seqlen.append(cu_seqlen[-1] + seq_len[-1])

            request.py_batch_idx = request.seq_slot

            if gather_context_logits:
                logits_gather_indices.extend(range(cu_seqlen[-2], cu_seqlen[-1]))
            else:
                logits_gather_indices.append(cu_seqlen[-1] - 1)

            # get cache indices and truncate the number of blocks according to end_compute
            cache_indices = kv_cache_manager.get_cache_indices(request)
            num_active_blocks = kv_cache_manager.get_num_kv_blocks(end_compute)
            cache_loc.extend(cache_indices[:num_active_blocks])
            pages_per_seq.append(num_active_blocks)
            cu_num_pages.append(cu_num_pages[-1] + pages_per_seq[-1])
            seq_len_with_cache.append(input_pos[-1] + seq_len[-1])
            last_page_len.append((seq_len_with_cache[-1] - 1) % page_size + 1)

            position_ids.append(list(range(input_pos[-1], seq_len_with_cache[-1])))

            # store seq slot idx
            slot_idx.append(request.seq_slot)
            use_initial_states.append(input_pos[-1] > 0)

            # store extra arguments
            if request.py_multimodal_data is not None:
                for k, v in request.py_multimodal_data.items():
                    extra_args[k].append(v)

        def _use_overlap_scheduler(request) -> bool:
            """Check if we should use overlap scheduler behavior."""
            return (
                not self._disable_overlap_scheduler
                and new_tokens is not None
                and not request.is_dummy
                and request.py_batch_idx is not None
            )

        def _compute_num_tokens_seen(request) -> int:
            """Compute num_tokens_seen based on request. Note that we treat extend requests
            (corresponding to running target model in speculative decoding) differently from
            normal generation requests.
            """
            is_extend = get_draft_token_length(request) > 0

            if is_extend:
                return request.max_beam_num_tokens - 1
            else:
                # When overlap scheduler is disabled, or when new_tokens is not available,
                # we use the previous token count
                use_overlap = _use_overlap_scheduler(request)
                if use_overlap:
                    return request.max_beam_num_tokens
                else:
                    return request.max_beam_num_tokens - 1

        def _build_input_ids(request) -> Tuple[List[int], List[int], bool]:
            """Build input_ids and gather indices for a request.
            Gather indices are used to gather tokens from new_tokens into input_ids when we run the overlap scheduler.
            """
            is_extend = get_draft_token_length(request) > 0

            # Check if we should use overlap scheduler behavior
            use_overlap = _use_overlap_scheduler(request)

            if not use_overlap:
                # No overlap scheduler or dummy request
                if is_extend:
                    input_ids = [request.get_token(0, request.get_num_tokens(0) - 1)] + [
                        token for token in request.py_draft_tokens
                    ]
                else:
                    input_ids = [request.get_token(0, request.get_num_tokens(0) - 1)]
                gather_indices = []
            else:
                # Overlap scheduler enabled
                if is_extend:
                    gather_indices = [
                        x * new_tokens.shape[1] + request.py_batch_idx
                        for x in range(len(request.py_draft_tokens) + 1)
                    ]

                    dummy_draft_tokens = [dummy_token for _ in range(len(request.py_draft_tokens))]
                    input_ids = [dummy_token] + dummy_draft_tokens
                else:
                    gather_indices = [request.py_batch_idx]
                    input_ids = [dummy_token]

            return input_ids, gather_indices, use_overlap

        for request in gen_requests:
            num_tokens_seen = _compute_num_tokens_seen(request)
            input_ids_for_request, gather_indices_to_append, use_overlap = _build_input_ids(request)

            input_ids.append(input_ids_for_request)
            input_pos.append(num_tokens_seen)
            flat_gather_indices.extend(gather_indices_to_append)

            num_generation_tokens += 1 + get_draft_token_length(request)
            request.py_batch_idx = request.seq_slot
            slot_idx.append(request.seq_slot)
            use_initial_states.append(input_pos[-1] > 0)

            seq_len.append(len(input_ids[-1]))
            cu_seqlen.append(cu_seqlen[-1] + seq_len[-1])

            # for generate requests, we always keep all logits (target logits + draft logits)
            logits_gather_indices.extend(range(cu_seqlen[-2], cu_seqlen[-1]))

            if use_overlap:
                mask_scatter_indices.extend(list(range(cu_seqlen[-2], cu_seqlen[-1])))

            # get cache indices
            cache_indices = kv_cache_manager.get_cache_indices(request)
            cache_loc.extend(cache_indices)
            pages_per_seq.append(len(cache_indices))
            cu_num_pages.append(cu_num_pages[-1] + pages_per_seq[-1])
            seq_len_with_cache.append(input_pos[-1] + seq_len[-1])
            last_page_len.append((seq_len_with_cache[-1] - 1) % page_size + 1)

            position_ids.append(list(range(input_pos[-1], seq_len_with_cache[-1])))

        # check for logits_gather_info
        # we only need to gather in the following situation:
        # 1. there are context requests and
        # 2. we are not gathering context logits
        # In other cases (decode-only) or when we keep all logits, we do not need to gather.
        gather_required = len(context_requests) > 0 and not gather_context_logits
        logits_gather_info = [len(logits_gather_indices), int(gather_required)]

        # update the sequence info object now
        self.cache_seq_interface.info.nest_sequences(
            input_ids,
            position_ids=position_ids,
            seq_len=seq_len,
            input_pos=input_pos,
            cu_seqlen=cu_seqlen,
            cache_loc=cache_loc,
            pages_per_seq=pages_per_seq,
            cu_num_pages=cu_num_pages,
            seq_len_with_cache=seq_len_with_cache,
            last_page_len=last_page_len,
            slot_idx=slot_idx,
            use_initial_states=use_initial_states,
            logits_gather_indices=logits_gather_indices,
            logits_gather_info=logits_gather_info,
            _gather_idx=None if new_tokens is None else flat_gather_indices,
            _mask_scatter_indices=None if new_tokens is None else mask_scatter_indices,
            **extra_args,
        )
        # scatter the new tokens into the input_ids tensor if provided
        if new_tokens is not None:
            self.cache_seq_interface.info.rescatter_input_ids(new_tokens.flatten())

        self.cache_seq_interface.info.run_host_prepare_for_attention_forward()

        if spec_resource_manager is not None and isinstance(
            spec_resource_manager, ADHiddenStateManager
        ):
            spec_resource_manager.prepare_hidden_states_capture(
                ordered_requests, self.cache_seq_interface
            )

        self.iter_states["num_ctx_requests"] = num_ctx_requests
        self.iter_states["num_ctx_tokens"] = num_ctx_tokens
        # TODO: handle extend requests and draft requests for specdec
        self.iter_states["num_generation_tokens"] = num_generation_tokens

    @nvtx_range("ad_compute_logits")
    def _compute_logits(self) -> List[torch.Tensor]:
        # run the model
        logits: torch.Tensor = self.model(**self.cache_seq_interface.named_args)[0]
        logits = self.cache_seq_interface.info.maybe_gather_and_squeeze_logits(logits)

        # TRTLLMSampler expects float32 logits. PyTorchModelEngine always casts to float32 regardless.
        return logits.float()

    def get_max_num_sequences(self) -> int:
        """Maximum number of sequences supported by the engine."""
        return self.cache_seq_interface.info.max_batch_size

    @torch.inference_mode()
    @maybe_pad_for_cuda_graph
    def forward(
        self,
        scheduled_requests: ScheduledRequests,
        resource_manager: ResourceManager,
        new_tensors_device: Optional[torch.Tensor] = None,
        gather_context_logits: bool = False,
        cache_indirection_buffer: Optional[torch.Tensor] = None,
        num_accepted_tokens_device: Optional[torch.Tensor] = None,
    ):
        """Run forward from scheduled requests; main entrypoint that gets called by the executor."""
        # convert requests and store in sequence info object
        new_tokens = getattr(new_tensors_device, "new_tokens", None)
        self._prepare_inputs(
            scheduled_requests, resource_manager, new_tokens, gather_context_logits
        )
        self.iter_counter += 1

        outputs = {
            "logits": self._compute_logits(),
        }

        # save hidden states after running model.forward() in _compute_logits()
        spec_resource_manager = resource_manager.get_resource_manager(
            ResourceManagerType.SPEC_RESOURCE_MANAGER
        )
        if spec_resource_manager is not None and isinstance(
            spec_resource_manager, ADHiddenStateManager
        ):
            spec_resource_manager.capture_hidden_states(self.cache_seq_interface)

        if self.mapping is not None:
            self._execute_logit_post_processors(scheduled_requests, outputs)

        return outputs


def share_target_weights_with_draft(
    target_model_engine: "ADEngine", draft_model_engine: PyTorchModelEngine
):
    """
    Certain speculative decoding methods (e.g. Eagle3) require sharing the target model's embedding and lm_head weights
    with the draft model. This function does this sharing if necessary.
    """

    assert isinstance(draft_model_engine.model, Eagle3ForCausalLM), (
        f"Expected draft_model_engine.model to be Eagle3ForCausalLM, got {type(draft_model_engine.model)}"
    )

    def share_embedding_weights_with_draft(
        target_model_engine: "ADEngine", draft_model_engine: PyTorchModelEngine
    ):
        embedding_weight = get_input_embeddings(target_model_engine.model)

        world_size = mpi_world_size()
        assert world_size <= 1, f"This code assumes tp<=1. World size: {world_size}"

        # Note: This simple forward function implementation assumes tp=1.
        # TODO(govind): Handle the tp>1 case.
        def new_embedding_forward(self, input_ids):
            return F.embedding(input_ids, self.weight)

        if draft_model_engine.model.model.embed_tokens is None:
            submodule = torch.nn.Module()
            submodule.forward = MethodType(new_embedding_forward, submodule)
            submodule.weight = embedding_weight
            draft_model_engine.model.model.embed_tokens = submodule

    def share_lm_head_weights_with_draft(
        target_model_engine: "ADEngine", draft_model_engine: PyTorchModelEngine
    ):
        vocab_size = target_model_engine.cache_seq_interface.info.vocab_size_padded

        lm_head_weight = get_lm_head_weights(target_model_engine.model)

        assert lm_head_weight.shape[0] == vocab_size, (
            f"Expected lm_head weight first dimension to be vocab_size={vocab_size}, "
            f"but got shape {lm_head_weight.shape}"
        )

        if draft_model_engine.model.load_lm_head_from_target:
            draft_model_engine.model.lm_head.weight = lm_head_weight

    share_embedding_weights_with_draft(target_model_engine, draft_model_engine)
    share_lm_head_weights_with_draft(target_model_engine, draft_model_engine)


def create_draft_model_engine_maybe(
    ad_config: LlmArgs, target_engine: ADEngine, dist_mapping: Mapping, mpi_dist: MPIDist
) -> Optional[PyTorchModelEngine]:
    """Create a draft model engine for speculative decoding.

    Args:
        ad_config: The AutoDeploy LLM configuration
        engine: The target model engine (ADEngine)
        dist_mapping: The distributed mapping configuration
        mpi_dist: The MPI distribution object

    Returns:
        PyTorchModelEngine configured as a draft model, or None if not needed
    """
    spec_config = ad_config.speculative_config

    if spec_config is None or not spec_config.spec_dec_mode.has_draft_model():
        return None

    has_spec_drafter = spec_config.spec_dec_mode.has_spec_drafter()

    draft_spec_config = copy.copy(spec_config)

    kv_cache_config = ad_config.kv_cache_config

    attn_runtime_features = AttentionRuntimeFeatures(
        chunked_prefill=ad_config.enable_chunked_prefill,
        cache_reuse=kv_cache_config.enable_block_reuse,
        has_speculative_draft_tokens=has_spec_drafter,
        chunk_size=target_engine.llm_args.max_num_tokens,
    )

    # Construct TorchLlmArgs for the draft model
    draft_llm_args = construct_draft_llm_args(
        ad_config=ad_config,
    )

    # chain drafter is not supported currently for AutoDeploy.
    # TODO(govind): Do this when we want to optimize 2-model spec dec performance.
    drafting_loop_wrapper = None

    draft_model_engine = PyTorchModelEngine(
        model_path=draft_spec_config.speculative_model_dir,
        llm_args=draft_llm_args,
        mapping=dist_mapping,
        attn_runtime_features=attn_runtime_features,
        dist=mpi_dist,
        spec_config=draft_spec_config,
        is_draft_model=True,
        drafting_loop_wrapper=drafting_loop_wrapper,
    )

    if draft_spec_config.spec_dec_mode.is_eagle3():
        share_target_weights_with_draft(
            target_model_engine=target_engine, draft_model_engine=draft_model_engine
        )

    draft_model_engine.kv_cache_manager_key = ResourceManagerType.DRAFT_KV_CACHE_MANAGER

    return draft_model_engine


class TRTLLMSamplerModelConfig:
    def __init__(self, vocab_size_padded: int):
        self.config = SimpleNamespace()
        self.config.vocab_size = vocab_size_padded

        # Initialized to dummy values as they are not used in the C++ code underlying TRTLLMSampler.
        self.config.num_hidden_layers = 42
        self.config.hidden_size = 42
        self.config.num_attention_heads = 42


def instantiate_sampler(
    ad_config: LlmArgs,
    max_num_sequences: int,
    max_draft_len: int,
    max_total_draft_tokens: int,
    dist_mapping: Mapping,
    engine: ADEngine,
):
    if ad_config.sampler_type == SamplerType.TorchSampler:
        # search sampler with speculative decoding
        sampler_args = TorchSampler.Args(
            max_seq_len=ad_config.max_seq_len,
            max_draft_len=max_draft_len,
            max_total_draft_tokens=max_total_draft_tokens,
            max_num_sequences=max_num_sequences,
            max_beam_width=ad_config.max_beam_width,
            disable_overlap_scheduler=ad_config.disable_overlap_scheduler,
        )
        sampler = TorchSampler(sampler_args)

    elif ad_config.sampler_type == SamplerType.TRTLLMSampler:
        vocab_size_padded: int = engine.cache_seq_interface.info.vocab_size_padded
        sampler_model_config = TRTLLMSamplerModelConfig(vocab_size_padded)
        decoding_mode = get_decoding_mode(ad_config.decoding_config, ad_config.max_beam_width)
        sampler = TRTLLMSampler(
            model=sampler_model_config,
            model_dtype=torch.bfloat16,  # hardcoded as bfloat16; does not seem necessary in C++ code.
            mapping=dist_mapping,
            decoding_mode=decoding_mode,
            disable_overlap_scheduler=ad_config.disable_overlap_scheduler,
            max_seq_len=ad_config.max_seq_len,
            max_batch_size=ad_config.max_batch_size,
            max_beam_width=ad_config.max_beam_width,
            decoding_config=ad_config.decoding_config,
            kv_cache_config=ad_config.kv_cache_config,
        )
    else:
        raise ValueError(f"Sampler type {ad_config.sampler_type} is not supported.")

    return sampler


def create_autodeploy_executor(ad_config: LlmArgs, tokenizer: Optional[TokenizerBase] = None):
    """Create an AutoDeploy executor from the given configuration and tokenizer.
    The tokenizer is required for guided decoding.

    This is the entrypoint API to the _autodeploy backend.
    """
    # initialize process groups
    world_size = mpi_world_size()
    rank = mpi_rank()
    dist_mapping = Mapping(rank=rank, world_size=world_size, tp_size=world_size)
    mpi_dist = MPIDist(dist_mapping)
    ad_logger.set_rank(rank)
    torch.cuda.set_device(rank)
    port = mpi_dist.broadcast(dist.get_free_port())  # use MPI broadcast to pick a free port
    dist.initialize_or_skip(rank, world_size, port)
    # some config
    assert ad_config.max_beam_width <= 1, "_autodeploy + beam_search is not supported"

    max_num_sequences = ad_config.max_batch_size * dist_mapping.pp_size
    # some derivative properties
    max_draft_len = (
        0 if ad_config.speculative_config is None else ad_config.speculative_config.max_draft_len
    )
    max_total_draft_tokens = (
        0
        if ad_config.speculative_config is None
        else ad_config.speculative_config.max_total_draft_tokens
    )

    # initialize model engine
    engine = ADEngine.build_from_config(ad_config=ad_config, mapping=dist_mapping)

    spec_config = ad_config.speculative_config
    if spec_config is not None and not (
        spec_config.spec_dec_mode.is_draft_target() or spec_config.spec_dec_mode.is_eagle3()
    ):
        raise ValueError(
            "Currently, AutoDeploy only supports speculative decoding in draft target or eagle3 mode."
        )

    if spec_config is not None and ad_config.guided_decoding_backend is not None:
        raise ValueError(
            "Guided decoding is not currently supported for speculative decoding in AutoDeploy."
        )

    draft_model_engine = create_draft_model_engine_maybe(
        ad_config=ad_config, target_engine=engine, dist_mapping=dist_mapping, mpi_dist=mpi_dist
    )

    spec_resource_manager = (
        ADHiddenStateManager(
            cache_seq_interface=engine.cache_seq_interface,
            config=spec_config,
            max_num_requests=ad_config.max_batch_size,
            max_seq_len=engine.llm_args.max_seq_len,
            max_num_tokens=engine.llm_args.max_num_tokens,
        )
        if isinstance(spec_config, EagleDecodingConfig)
        else None
    )

    # check kvcache config for partial block reuse
    # TODO: copy_on_partial_reuse is not supported yet, see
    # https://github.com/NVIDIA/TensorRT-LLM/issues/7142 for more details.
    enable_block_reuse = ad_config.kv_cache_config.enable_block_reuse
    enable_partial_reuse = ad_config.kv_cache_config.enable_partial_reuse
    copy_on_partial_reuse = ad_config.kv_cache_config.copy_on_partial_reuse
    if enable_block_reuse and enable_partial_reuse and copy_on_partial_reuse:
        raise RuntimeError(
            f"partial block reuse with {copy_on_partial_reuse=} set to True is NOT supported"
            " in AutoDeploy. Please set it to False via the kv_cache_config.copy_on_partial_reuse "
            "field in tensorrt_llm._torch.auto_deploy.llm_args.LlmArgs."
        )

    # TODO: detect whether SSM layer is present in the model and raise an error or disable block
    # reuse with a warning --> see https://github.com/NVIDIA/TensorRT-LLM/issues/7142. For now, we
    # just emit a general warning.
    if enable_block_reuse:
        ad_logger.warning(
            f"{enable_block_reuse=} is enabled. Note that this is not supported for SSM layers and"
            " may lead to incorrect results if the model contains SSM layers."
        )

    # resource managers
    kv_cache_manager = _CacheManagerWithFakePool(
        ad_config.kv_cache_config,
        num_blocks=engine.cache_seq_interface.info.num_pages,
        tokens_per_block=ad_config.attn_page_size,
        max_seq_len=ad_config.max_seq_len,
        max_batch_size=ad_config.max_batch_size,
    )
    seq_slot_manager = SeqSlotManager(max_num_sequences=max_num_sequences)

    draft_kv_cache_manager = create_draft_kv_cache_manager_maybe(
        draft_model_engine, ad_config, dist_mapping
    )

    resource_manager = ResourceManager(
        {
            ResourceManagerType.KV_CACHE_MANAGER: kv_cache_manager,
            ResourceManagerType.DRAFT_KV_CACHE_MANAGER: draft_kv_cache_manager,
            ResourceManagerType.SEQ_SLOT_MANAGER: seq_slot_manager,
            ResourceManagerType.SPEC_RESOURCE_MANAGER: spec_resource_manager,
        }
    )
    resource_manager.resource_managers.move_to_end(ResourceManagerType.KV_CACHE_MANAGER, last=True)

    # TODO: consider passing through scheduler_config arguments here. Not doing this for now since
    # it requires correctly setting up the C++ pybind scheduler config from the LLMArgs and then
    # processing the arguments here...

    # Chunked prefill
    if ad_config.enable_chunked_prefill:
        chunk_unit_size = ad_config.attn_page_size
        chunking_policy = ContextChunkingPolicy.FIRST_COME_FIRST_SERVED
        ctx_chunk_config: Tuple[StrEnum, int] = (chunking_policy, chunk_unit_size)
    else:
        ctx_chunk_config = None

    # scheduling
    capacitor_scheduler = BindCapacityScheduler(
        max_num_requests=ad_config.max_batch_size,
        kv_cache_manager=kv_cache_manager.impl,
        peft_cache_manager=None,
    )
    mb_scheduler = BindMicroBatchScheduler(
        max_batch_size=ad_config.max_batch_size,
        max_num_tokens=engine.cache_seq_interface.info.max_num_tokens,
        ctx_chunk_config=ctx_chunk_config,
    )
    scheduler = SimpleScheduler(capacitor_scheduler, mb_scheduler)

    vocab_size_padded = engine.cache_seq_interface.info.vocab_size_padded
    sampler = instantiate_sampler(
        ad_config=ad_config,
        max_num_sequences=max_num_sequences,
        max_draft_len=max_draft_len,
        max_total_draft_tokens=max_total_draft_tokens,
        dist_mapping=dist_mapping,
        engine=engine,
    )

    # Guided (structured) decoding.
    guided_decoder = None
    if (
        (guided_decoding_backend := ad_config.guided_decoding_backend) is not None
    ) and dist_mapping.is_last_pp_rank():
        if vocab_size_padded is None:
            raise RuntimeError(
                "Could not determine the vocabulary size. Required for guided decoding."
            )
        guided_decoding_config = get_guided_decoding_config(
            guided_decoding_backend=guided_decoding_backend, tokenizer=tokenizer
        )
        guided_decoder = GuidedDecoder(
            guided_decoding_config=guided_decoding_config,
            max_num_sequences=ad_config.max_batch_size,
            vocab_size_padded=vocab_size_padded,
        )

    drafter = get_spec_drafter(
        model_engine=engine,
        draft_model_engine=draft_model_engine,
        sampler=sampler,
        spec_resource_manager=spec_resource_manager,
    )

    # creating the executor object
    py_executor = PyExecutor(
        resource_manager,
        scheduler,
        model_engine=engine,
        sampler=sampler,
        dist=mpi_dist,
        max_num_sequences=max_num_sequences,
        disable_overlap_scheduler=ad_config.disable_overlap_scheduler,
        max_input_len=ad_config.max_input_len,
        max_batch_size=ad_config.max_batch_size,
        max_draft_len=max_draft_len,
        max_total_draft_tokens=max_total_draft_tokens,
        max_beam_width=ad_config.max_beam_width,
        guided_decoder=guided_decoder,
        drafter=drafter,
        iter_stats_max_iterations=ad_config.iter_stats_max_iterations,
    )
    return py_executor
