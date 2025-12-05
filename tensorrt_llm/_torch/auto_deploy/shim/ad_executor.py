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
from collections import defaultdict
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple

import torch
from strenum import StrEnum
from torch._prims_common import DeviceLikeType

from tensorrt_llm._torch.attention_backend.interface import AttentionRuntimeFeatures
from tensorrt_llm._torch.pyexecutor._util import _create_kv_cache_manager, get_kv_cache_manager_cls
from tensorrt_llm._torch.pyexecutor.guided_decoder import GuidedDecoder
from tensorrt_llm._torch.pyexecutor.llm_request import get_draft_token_length
from tensorrt_llm._torch.pyexecutor.py_executor_creator import get_guided_decoding_config
from tensorrt_llm._torch.pyexecutor.seq_slot_manager import SeqSlotManager
from tensorrt_llm._torch.speculative import get_spec_drafter
from tensorrt_llm._utils import nvtx_range
from tensorrt_llm.llmapi.llm_args import (
    ContextChunkingPolicy,
    LoadFormat,
    SpeculativeConfig,
    TorchLlmArgs,
)
from tensorrt_llm.llmapi.tokenizer import TokenizerBase

from ...._utils import mpi_rank, mpi_world_size
from ....bindings.internal.batch_manager import CacheType
from ....mapping import Mapping
from ...distributed import MPIDist
from ...pyexecutor.model_engine import ModelEngine, PyTorchModelEngine
from ...pyexecutor.py_executor import PyExecutor
from ...pyexecutor.resource_manager import KVCacheManager, ResourceManager, ResourceManagerType
from ...pyexecutor.sampler import TorchSampler
from ...pyexecutor.scheduler import (
    BindCapacityScheduler,
    BindMicroBatchScheduler,
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
    def build_from_config(cls, ad_config: LlmArgs):
        """Build the ADEngine using the LlmArgs that gets passed through from the LLM."""

        max_batch_size = ad_config.max_batch_size
        max_seq_len = ad_config.max_seq_len
        attn_page_size = ad_config.attn_page_size
        max_num_tokens = ad_config.max_num_tokens
        max_beam_width = ad_config.max_beam_width

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
            chunk_size=factory.chunk_size,
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
            max_beam_width,
            ad_config.speculative_config,
            ad_config.disable_overlap_scheduler,
            reporting_info,
        )

    @torch.inference_mode()
    def __init__(
        self,
        get_inference_model: GetInferenceModel,
        seq_info: SequenceInfo,
        device: DeviceLikeType,
        max_beam_width: int = 1,
        spec_config: Optional[SpeculativeConfig] = None,
        disable_overlap_scheduler: bool = False,
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
        self.iter_counter = 0
        self.iter_states = {}
        self.llm_args.max_seq_len = seq_info.max_seq_len

        # NOTE (lucaslie): not a declared base member in the base class; required by PyExecutor...
        self.max_beam_width = max_beam_width
        self.enable_attention_dp = False
        self._disable_overlap_scheduler = disable_overlap_scheduler

        self.spec_config = spec_config

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

    @nvtx_range("ad_prepare_inputs")
    def _prepare_inputs(
        self,
        scheduled_requests: ScheduledRequests,
        resource_manager: ResourceManager,
        new_tokens: Optional[torch.Tensor] = None,
        gather_context_logits: bool = False,
    ):
        """Prepare inputs for AD Model from scheduled requests."""
        # cache manager
        kv_cache_manager = resource_manager.get_resource_manager(
            ResourceManagerType.KV_CACHE_MANAGER
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
        # info to be extracted
        input_ids: List[List[int]] = []
        input_pos: List[int] = []
        logits_gather_mask: List[bool] = []
        page_assignments: List[List[int]] = []
        slot_idx: List[int] = []

        # gather indices are used to gather tokens in new_tokens into input_ids
        flat_gather_indices: List[List[int]] = []
        extra_args: Dict[str, List[torch.Tensor]] = defaultdict(list)

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

            request.py_batch_idx = request.seq_slot
            logits_gather_mask.extend([gather_context_logits] * len(prompt_tokens))
            logits_gather_mask[-1] = True

            # get cache indices and truncate the number of blocks according to end_compute
            cache_indices = kv_cache_manager.get_cache_indices(request)
            num_active_blocks = kv_cache_manager.get_num_kv_blocks(end_compute)
            page_assignments.append(cache_indices[:num_active_blocks])

            # store seq slot idx
            slot_idx.append(request.seq_slot)

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

        def _build_input_ids(request) -> Tuple[List[int], List[int]]:
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

            return input_ids, gather_indices

        for request in gen_requests:
            num_tokens_seen = _compute_num_tokens_seen(request)
            input_ids_for_request, gather_indices_to_append = _build_input_ids(request)

            # return all logits
            logits_gather_mask.append(True)
            input_ids.append(input_ids_for_request)
            input_pos.append(num_tokens_seen)
            flat_gather_indices.extend(gather_indices_to_append)

            num_generation_tokens += 1 + get_draft_token_length(request)
            request.py_batch_idx = request.seq_slot
            slot_idx.append(request.seq_slot)

            # get cache indices
            cache_indices = kv_cache_manager.get_cache_indices(request)
            page_assignments.append(cache_indices)

        # Only pass logits_gather_mask if it's a registered arg (i.e., transform was applied)
        if "logits_gather_mask" in self.cache_seq_interface.info._registered_args:
            extra_args["logits_gather_mask"] = logits_gather_mask

        # update the sequence info object now
        self.cache_seq_interface.info.nest_sequences(
            input_ids,
            input_pos=input_pos,
            page_assignments=page_assignments,
            slot_idx=slot_idx,
            **extra_args,
        )
        # scatter the new tokens into the input_ids tensor if provided
        if new_tokens is not None:
            self.cache_seq_interface.info.rescatter_input_ids(
                ungathered_input_ids=new_tokens.flatten(),  # ensure it's flattened
                gather_idx=flat_gather_indices,
                scatter_ref=dummy_token,
            )

        self.iter_states["num_ctx_requests"] = num_ctx_requests
        self.iter_states["num_ctx_tokens"] = num_ctx_tokens
        # TODO: handle extend requests and draft requests for specdec
        self.iter_states["num_generation_tokens"] = num_generation_tokens

    @nvtx_range("ad_compute_logits")
    def _compute_logits(self) -> List[torch.Tensor]:
        # run the model
        logits: torch.Tensor = self.model(**self.cache_seq_interface.named_args)[0]
        return logits

    def get_max_num_sequences(self) -> int:
        """Maximum number of sequences supported by the engine."""
        return self.cache_seq_interface.info.max_batch_size

    @torch.inference_mode()
    def forward(
        self,
        scheduled_requests: ScheduledRequests,
        resource_manager: ResourceManager,
        new_tensors_device: Optional[torch.Tensor] = None,
        gather_context_logits: bool = False,
        cache_indirection_buffer: Optional[torch.Tensor] = None,
    ):
        """Run forward from scheduled requests; main entrypoint that gets called by the executor."""
        # convert requests and store in sequence info object
        new_tokens = getattr(new_tensors_device, "new_tokens", None)
        self._prepare_inputs(
            scheduled_requests, resource_manager, new_tokens, gather_context_logits
        )
        self.iter_counter += 1

        return {"logits": self._compute_logits()}


def create_draft_model_engine_maybe(
    ad_config: LlmArgs, engine, dist_mapping: Mapping, mpi_dist: MPIDist
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
        chunk_size=engine.llm_args.max_num_tokens,
    )

    # Construct TorchLlmArgs for the draft model
    draft_llm_args = construct_draft_llm_args(
        ad_config=ad_config,
    )

    draft_model_engine = PyTorchModelEngine(
        model_path=draft_spec_config.speculative_model_dir,
        llm_args=draft_llm_args,
        mapping=dist_mapping,
        attn_runtime_features=attn_runtime_features,
        dist=mpi_dist,
        spec_config=draft_spec_config,
        is_draft_model=True,
        drafting_loop_wrapper=None,
    )

    draft_model_engine.kv_cache_manager_key = ResourceManagerType.DRAFT_KV_CACHE_MANAGER

    return draft_model_engine


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
    engine = ADEngine.build_from_config(ad_config=ad_config)

    spec_config = ad_config.speculative_config
    if spec_config is not None and not spec_config.spec_dec_mode.is_draft_target():
        raise ValueError(
            "Currently, AutoDeploy only supports speculative decoding in draft target mode."
        )

    if spec_config is not None and ad_config.guided_decoding_backend is not None:
        raise ValueError(
            "Guided decoding is not currently supported for speculative decoding in AutoDeploy."
        )

    # Speculative resource manager not needed for DraftTargetDecoding.
    spec_resource_manager = None

    draft_model_engine = create_draft_model_engine_maybe(
        ad_config=ad_config, engine=engine, dist_mapping=dist_mapping, mpi_dist=mpi_dist
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

    # Guided (structured) decoding.
    guided_decoder = None
    if (
        (guided_decoding_backend := ad_config.guided_decoding_backend) is not None
    ) and dist_mapping.is_last_pp_rank():
        vocab_size_padded = engine.cache_seq_interface.info.vocab_size_padded
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
    )
    return py_executor
