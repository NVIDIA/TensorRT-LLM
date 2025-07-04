from itertools import chain
from types import SimpleNamespace
from typing import List, Optional, Tuple

import torch
from torch._prims_common import DeviceLikeType

from tensorrt_llm._torch.pyexecutor.seq_slot_manager import SeqSlotManager
from tensorrt_llm._utils import nvtx_range

from ...._utils import mpi_rank, mpi_world_size
from ....bindings.executor import ExecutorConfig
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
from ..llm_args import LlmArgs
from ..transformations.transform import InferenceOptimizer
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
    def build_from_config(cls, ad_config: LlmArgs):
        """Build the ADEngine using the AD LlmArgs that gets passed through from the LLM."""

        max_batch_size = ad_config.max_batch_size
        max_seq_len = ad_config.max_seq_len
        attn_page_size = ad_config.attn_page_size
        max_num_tokens = ad_config.max_num_tokens
        ad_logger.info(f"{max_seq_len=}, {max_batch_size=}, {attn_page_size=}, {max_num_tokens=}")

        # initialize seq info object
        seq_info = SequenceInfo(
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
            page_size=attn_page_size,
            max_num_tokens=max_num_tokens,
        )

        # update device to contain the current default device if it's in cuda
        device = torch.device(ad_config.device)
        if device.type == "cuda" and device.index is None:
            device = torch.device(f"cuda:{torch.cuda.current_device()}")
        device = str(device)

        # construct inference optimizer
        build_and_optimize = InferenceOptimizer(
            factory=ad_config.create_factory(), ad_config=ad_config
        )

        # construct engine
        return cls(build_and_optimize, seq_info, device)

    @torch.inference_mode()
    def __init__(
        self,
        get_inference_model: GetInferenceModel,
        seq_info: SequenceInfo,
        device: DeviceLikeType,
    ) -> None:
        """Initialize the engine with model and sequence information."""
        # NOTE (lucaslie): create a fake Namespace to satisfy PyExecutor requirements...
        # This is not correctly declared in the base ModelEngine class though...
        self.pytorch_backend_config = SimpleNamespace()
        self.pytorch_backend_config.print_iter_log = False
        self.pytorch_backend_config.enable_iter_perf_stats = False
        self.pytorch_backend_config.enable_iter_req_stats = False
        self.pytorch_backend_config.stream_interval = 1
        self.iter_counter = 0

        # NOTE (lucaslie): not a declared base member in the base class; required by PyExecutor...
        self.enable_attention_dp = False

        # construct cache sequence interface
        self.cache_seq_interface = CachedSequenceInterface(
            sequence_info=seq_info,
            device=device,
        )

        # build model
        self.model = get_inference_model(self.cache_seq_interface)

        # start fresh with fixed seed
        torch.manual_seed(1234)

    @nvtx_range("ad_prepare_inputs")
    def _prepare_inputs(
        self, scheduled_requests: ScheduledRequests, resource_manager: ResourceManager
    ) -> bool:
        """Prepare inputs for AD Model from scheduled requests."""
        # cache manager
        kv_cache_manager = resource_manager.get_resource_manager(
            ResourceManagerType.KV_CACHE_MANAGER
        )

        # requests in order of context, extend (generate with draft), generate
        context_requests = scheduled_requests.context_requests
        extend_requests = [r for r in scheduled_requests.generation_requests if r.draft_tokens]
        gen_requests = [r for r in scheduled_requests.generation_requests if not r.draft_tokens]

        # info to be extracted
        input_ids: List[List[int]] = []
        input_pos: List[int] = []
        last_logit_only: List[bool] = []
        page_assignments: List[List[int]] = []

        # look at context requests first
        for request in context_requests:
            # store input ids and pos of first token in sequence
            input_ids.append(request.get_tokens(0))
            input_pos.append(request.context_current_position)

            # only return last logit
            last_logit_only.append(True)

        # look at extend+generate requests next
        for request in chain(extend_requests, gen_requests):
            # store input ids and pos of first token in sequence
            input_ids.append([request.get_token(0, request.get_num_tokens(0) - 1)])
            input_pos.append(request.max_beam_num_tokens - 1)

            # check for draft tokens
            if request.draft_tokens:
                input_ids[-1].extend([t for t in request.draft_tokens])

            # return all logits
            last_logit_only.append(False)

        # extract cache information for all requests
        for request in chain(context_requests, extend_requests, gen_requests):
            # get cache indices
            cache_indices = kv_cache_manager.get_cache_indices(request)
            page_assignments.append(cache_indices)

        # update the sequence info object now
        si = self.cache_seq_interface.info
        si.nest_sequences(input_ids)
        si.update_pos(input_pos, reset=True)
        si.assign_cache_loc(page_assignments)

        return last_logit_only

    def _compute_logits(self) -> List[torch.Tensor]:
        # run the model
        logits: torch.Tensor = self.model(*self.cache_seq_interface.args)[0]

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
        new_tokens_device: Optional[torch.Tensor] = None,
        gather_context_logits: bool = False,
    ):
        """Run forward from scheduled requests; main entrypoint that gets called by the executor."""
        # convert requests and store in sequence info object
        last_logit_only = self._prepare_inputs(scheduled_requests, resource_manager)

        # compute all logits
        logits = self._compute_logits()

        # gather+cat logits
        logits_flat = torch.cat(
            [ls_one_seq[-last_only:] for ls_one_seq, last_only in zip(logits, last_logit_only)],
            dim=0,
        )

        return {"logits": logits_flat}


def create_autodeploy_executor(executor_config: ExecutorConfig, checkpoint_dir: str = None):
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
    assert isinstance(executor_config.pytorch_backend_config, LlmArgs), msg
    ad_config: LlmArgs = executor_config.pytorch_backend_config

    max_num_sequences = ad_config.max_batch_size * dist_mapping.pp_size
    # some derivative properties
    max_draft_tokens = (
        0 if ad_config.speculative_config is None else ad_config.speculative_config.max_draft_tokens
    )

    # initialize model engine
    engine = ADEngine.build_from_config(ad_config=ad_config)

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

    # scheduling
    capacitor_scheduler = BindCapacityScheduler(ad_config.max_batch_size, kv_cache_manager.impl)
    mb_scheduler = BindMicroBatchScheduler(
        ad_config.max_batch_size, engine.cache_seq_interface.info.max_num_tokens
    )
    scheduler = SimpleScheduler(capacitor_scheduler, mb_scheduler)

    # search sampler with speculative decoding
    # TODO (lucaslie, fridah-nv): some models require mixed_sampler=True to have good outputs, see
    # https://github.com/NVIDIA/TensorRT-LLM/issues/5254
    # We should expose mixed_sample to our build_and_run_ad script so we can configure this
    # correctly for models as needed.
    sampler_args = TorchSampler.Args(
        max_seq_len=ad_config.max_seq_len,
        max_draft_tokens=max_draft_tokens,
        max_num_sequences=max_num_sequences,
        max_beam_width=executor_config.max_beam_width,
        mixed_sampler=ad_config.mixed_sampler,
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
        max_draft_tokens=max_draft_tokens,
    )
    return py_executor
