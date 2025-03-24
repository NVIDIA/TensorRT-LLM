from itertools import chain
from typing import Dict, List, Optional, Tuple

import torch
from torch._prims_common import DeviceLikeType

from ...._utils import mpi_rank, mpi_world_size
from ....bindings.executor import ExecutorConfig
from ....bindings.internal.batch_manager import CacheType
from ....mapping import Mapping
from ...pyexecutor.config import PyTorchConfig
from ...pyexecutor.decoder import TorchDecoder
from ...pyexecutor.distributed import MPIDist
from ...pyexecutor.model_engine import ModelEngine
from ...pyexecutor.py_executor import PyExecutor
from ...pyexecutor.resource_manager import KVCacheManager, ResourceManager
from ...pyexecutor.scheduler import (
    BindCapacityScheduler,
    BindMicroBatchScheduler,
    ScheduledRequests,
    SimpleScheduler,
)
from ..custom_ops.attention_interface import SequenceInfo
from ..distributed import common as dist
from ..models import ModelFactoryRegistry
from ..transformations.transform import InferenceOptimizer
from ..utils.logger import ad_logger
from .interface import AutoDeployConfig, CachedSequenceInterface, GetInferenceModel


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
        # TODO (lliebenwein): this is VERY hacky... Ideally, we want to compute the number of blocks
        # just like in the original implementation. However, let's wait for the layer-wise attention
        # implementation before over-optimizing the function here
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
    def build_from_config(
        cls,
        model: str,
        ad_config: AutoDeployConfig,
        seq_info: SequenceInfo,
        device: DeviceLikeType,
    ):
        """Build the ADEngine using the AutoDeployConfig that gets passed through from the LLM."""
        # construct model factory
        model_kwargs = {"max_position_embeddings": seq_info.max_seq_len, **ad_config.model_kwargs}
        factory = ModelFactoryRegistry.get("hf")(
            model=model,
            model_kwargs=model_kwargs,
            skip_loading_weights=ad_config.skip_loading_weights,
        )

        # construct inference optimizer
        # TODO (lliebenwein): let's split up the compile backend to separately handle cuda graph
        # and torch compile so we can follow the PyTorchConfig here and enable it separately.
        if ad_config.use_cuda_graph or ad_config.torch_compile_enabled:
            compile_backend = "torch-opt"
        else:
            compile_backend = "torch-simple"
        build_and_optimize = InferenceOptimizer(
            factory=factory, attn_backend=ad_config.attn_backend, compile_backend=compile_backend
        )

        # construct engine
        engine = cls(build_and_optimize, seq_info, device)
        engine.pytorch_backend_config = ad_config

        return engine

    @torch.inference_mode()
    def __init__(
        self,
        get_inference_model: GetInferenceModel,
        seq_info: SequenceInfo,
        device: DeviceLikeType,
    ) -> None:
        """Initialize the engine with model and sequence information."""
        # TODO: this seems to have been hastily added in !8301. Let's see if it gets reverted...
        # PyTorchEngine requirement which is not set in the base class though...
        self.iter_states: Dict[str, int] = {}
        self.pytorch_backend_config: Optional[PyTorchConfig] = None

        # TODO: not a declared base member in the base class (see !8291 for diff)
        self.enable_attention_dp = False

        # TODO: fix baseclass from this diff in !8360
        self.iter_counter = 0

        # construct cache sequence interface
        self.cache_seq_interface = CachedSequenceInterface(
            sequence_info=seq_info,
            device=device,
        )

        # build model
        self.model = get_inference_model(self.cache_seq_interface)

        # start fresh with fixed seed
        torch.manual_seed(1234)

    def _prepare_inputs(
        self, scheduled_requests: ScheduledRequests, resource_manager: ResourceManager
    ) -> bool:
        """Prepare inputs for AD Model from scheduled requests."""
        # cache manager
        kv_cache_manager = resource_manager.get_resource_manager("kv_cache_manager")

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

        # update iter_stats for PyExecutor
        num_ctx_requests = len(context_requests)
        self.iter_states["num_ctx_requests"] = num_ctx_requests
        self.iter_states["num_ctx_tokens"] = sum(len(ids) for ids in input_ids[:num_ctx_requests])
        self.iter_states["num_generation_tokens"] = len(gen_requests)

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


def create_autodeploy_executor(
    executor_config: ExecutorConfig, checkpoint_dir: str = None, engine_dir: str = None
):
    """Create an AutoDeploy executor from the given configuration and checkpoint directory.

    This is the entrypoint API to the autodeploy backend.
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
    if executor_config.pytorch_backend_config is None:
        executor_config.pytorch_backend_config = AutoDeployConfig(attn_backend="FlashInfer")

    max_batch_size = executor_config.max_batch_size
    max_seq_len = executor_config.max_seq_len
    tokens_per_block = executor_config.tokens_per_block
    max_num_tokens = executor_config.max_num_tokens
    ad_logger.info(f"{max_seq_len=}, {max_batch_size=}, {tokens_per_block=}, {max_num_tokens=}")

    # initialize model engine
    engine = ADEngine.build_from_config(
        model=checkpoint_dir,
        ad_config=executor_config.pytorch_backend_config,
        seq_info=SequenceInfo(
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
            page_size=tokens_per_block,
            max_num_tokens=max_num_tokens,
        ),
        device="cuda",
    )

    # resource managers
    kv_cache_manager = _CacheManagerWithFakePool(
        executor_config.kv_cache_config,
        num_blocks=engine.cache_seq_interface.info.num_pages,
        tokens_per_block=tokens_per_block,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )
    resource_manager = ResourceManager({"kv_cache_manager": kv_cache_manager})
    resource_manager.resource_managers.move_to_end("kv_cache_manager", last=True)

    # scheduling
    capacitor_scheduler = BindCapacityScheduler(max_batch_size, kv_cache_manager.impl)
    mb_scheduler = BindMicroBatchScheduler(
        max_batch_size, engine.cache_seq_interface.info.max_num_tokens
    )
    scheduler = SimpleScheduler(capacitor_scheduler, mb_scheduler)

    # search decoder with speculative decoding
    decoder = TorchDecoder(max_seq_len=max_seq_len)

    # creating the executor object
    py_config: PyTorchConfig = executor_config.pytorch_backend_config
    py_executor = PyExecutor(
        resource_manager,
        scheduler,
        model_engine=engine,
        decoder=decoder,
        dist=mpi_dist,
        enable_overlap_scheduler=py_config.enable_overlap_scheduler,
    )
    return py_executor
