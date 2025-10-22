from collections import defaultdict
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple

import torch
from strenum import StrEnum
from torch._prims_common import DeviceLikeType

from tensorrt_llm._torch.pyexecutor.seq_slot_manager import SeqSlotManager
from tensorrt_llm._utils import nvtx_range
from tensorrt_llm.llmapi.llm_args import ContextChunkingPolicy

from ...._utils import mpi_rank, mpi_world_size
from ....bindings.internal.batch_manager import CacheType
from ....mapping import Mapping
from ...distributed import MPIDist
from ...pyexecutor.model_engine import ModelEngine
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
from ..llm_args import AutoDeployConfig, LlmArgs
from ..transform.optimizer import InferenceOptimizer
from ..utils.logger import ad_logger
from .interface import CachedSequenceInterface, GetInferenceModel


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
    def build_from_config(cls, ad_config: AutoDeployConfig):
        """Build the ADEngine using the AutoDeployConfig that gets passed through from the LLM."""

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

        # initialize seq info object
        seq_info = SequenceInfo(
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
            page_size=attn_page_size,
            max_num_tokens=max_num_tokens,
        )

        factory = ad_config.create_factory()

        # TODO (lucaslie): consider how we move args around InferenceOptimizer.__init__,
        # ADEngine.__init__, and ADEngine.build_from_config. Seems a bit unnatural atm.

        # construct inference optimizer
        build_and_optimize = InferenceOptimizer(factory=factory, config=ad_config.transforms)

        # construct engine
        return cls(build_and_optimize, seq_info, device, max_beam_width)

    @torch.inference_mode()
    def __init__(
        self,
        get_inference_model: GetInferenceModel,
        seq_info: SequenceInfo,
        device: DeviceLikeType,
        max_beam_width: int = 1,
    ) -> None:
        """Initialize the engine with model and sequence information."""
        # NOTE (lucaslie): create a fake Namespace to satisfy PyExecutor requirements...
        # This is not correctly declared in the base ModelEngine class though...
        self.pytorch_backend_config = SimpleNamespace()
        self.pytorch_backend_config.print_iter_log = False
        self.pytorch_backend_config.enable_iter_perf_stats = False
        self.pytorch_backend_config.enable_iter_req_stats = False
        self.pytorch_backend_config.stream_interval = 1
        self.pytorch_backend_config.attention_dp_enable_balance = False
        self.pytorch_backend_config.attention_dp_time_out_iters = 50
        self.pytorch_backend_config.attention_dp_batching_wait_iters = 10
        self.pytorch_backend_config.batch_wait_timeout_ms = 0
        self.pytorch_backend_config.batch_wait_timeout_iters = 0
        self.pytorch_backend_config.batch_wait_max_tokens_ratio = 0.0
        self.pytorch_backend_config.max_num_tokens = seq_info.max_num_tokens
        self.iter_counter = 0

        # NOTE (lucaslie): not a declared base member in the base class; required by PyExecutor...
        self.max_beam_width = max_beam_width
        self.enable_attention_dp = False

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
    ) -> List[bool]:
        """Prepare inputs for AD Model from scheduled requests."""
        # cache manager
        kv_cache_manager = resource_manager.get_resource_manager(
            ResourceManagerType.KV_CACHE_MANAGER
        )

        # requests in order of context, generate
        context_requests = scheduled_requests.context_requests
        gen_requests = [r for r in scheduled_requests.generation_requests if not r.draft_tokens]

        # info to be extracted
        input_ids: List[List[int]] = []
        input_pos: List[int] = []
        last_logit_only: List[bool] = []
        page_assignments: List[List[int]] = []
        slot_idx: List[int] = []
        flat_gather_idx: List[int] = []
        extra_args: Dict[str, List[torch.Tensor]] = defaultdict(list)

        dummy_token = -1

        # look at context requests first
        for request in context_requests:
            # store input ids and pos of first token in sequence
            # NOTE: begin_compute > 0 indicates block reuse
            # NOTE: end_compute will be used in the future for chunked prefill
            all_prompt_tokens = request.get_tokens(0)
            begin_compute = request.context_current_position
            end_compute = begin_compute + request.context_chunk_size
            prompt_tokens = all_prompt_tokens[begin_compute:end_compute]

            input_ids.append(prompt_tokens)
            input_pos.append(begin_compute)

            request.py_batch_idx = request.seq_slot
            last_logit_only.append(True)

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

        # look at generate requests next
        # TODO: we should also handle extend requests (for speculative decoding) here
        for request in gen_requests:
            # new_tokens are provided when the overlap scheduler is enabled.
            if new_tokens is None or request.is_dummy or request.py_batch_idx is None:
                input_ids.append([request.get_token(0, request.get_num_tokens(0) - 1)])
                input_pos.append(request.max_beam_num_tokens - 1)
            else:
                input_ids.append([dummy_token])
                input_pos.append(request.max_beam_num_tokens)
                flat_gather_idx.append(request.py_batch_idx)

            request.py_batch_idx = request.seq_slot

            # store seq slot idx
            # TODO: double-check if this is correct for the overlap scheduler
            slot_idx.append(request.seq_slot)

            # return all logits
            last_logit_only.append(False)

            # get cache indices
            cache_indices = kv_cache_manager.get_cache_indices(request)
            page_assignments.append(cache_indices)

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
                gather_idx=flat_gather_idx,
                scatter_ref=dummy_token,
            )

        return last_logit_only

    @nvtx_range("ad_compute_logits")
    def _compute_logits(self) -> List[torch.Tensor]:
        # run the model
        logits: torch.Tensor = self.model(**self.cache_seq_interface.named_args)[0]

        # return a list of tensors
        return self.cache_seq_interface.info.unnest_sequences(logits)

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
        last_logit_only = self._prepare_inputs(scheduled_requests, resource_manager, new_tokens)

        # compute all logits
        logits = self._compute_logits()

        # gather+cat logits
        logits_flat = torch.cat(
            [ls_one_seq[-last_only:] for ls_one_seq, last_only in zip(logits, last_logit_only)],
            dim=0,
        )

        return {"logits": logits_flat}


def create_autodeploy_executor(ad_config: LlmArgs):
    """Create an AutoDeploy executor from the given configuration and checkpoint directory.

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
    msg = "pytorch_backend_config must be an AD LlmArgs object"
    assert isinstance(ad_config, LlmArgs), msg
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
    resource_manager = ResourceManager(
        {
            ResourceManagerType.KV_CACHE_MANAGER: kv_cache_manager,
            ResourceManagerType.SEQ_SLOT_MANAGER: seq_slot_manager,
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
    )
    sampler = TorchSampler(sampler_args)

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
    )
    return py_executor
