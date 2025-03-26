import copy

import tensorrt_llm
import tensorrt_llm.bindings as tllm
from tensorrt_llm._torch.attention_backend.interface import \
    AttentionRuntimeFeatures
from tensorrt_llm._torch.pyexecutor.config import PyTorchConfig
from tensorrt_llm._torch.pyexecutor.decoder import (EarlyStopDecoder,
                                                    TorchDecoder,
                                                    TorchStarAttentionDecoder,
                                                    TRTLLMDecoder)
from tensorrt_llm._torch.pyexecutor.distributed import MPIDist
from tensorrt_llm._torch.pyexecutor.guided_decoder import \
    GuidedDecoderResourceManager
from tensorrt_llm._torch.pyexecutor.kv_cache_transceiver import (
    AttentionTypeCpp, create_kv_cache_transceiver)
from tensorrt_llm._torch.pyexecutor.model_engine import PyTorchModelEngine
from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor
from tensorrt_llm._torch.pyexecutor.resource_manager import ResourceManager
from tensorrt_llm._torch.pyexecutor.scheduler import (BindCapacityScheduler,
                                                      BindMicroBatchScheduler,
                                                      SimpleScheduler)
from tensorrt_llm._torch.speculative import (get_spec_decoder,
                                             get_spec_resource_manager)
from tensorrt_llm.bindings.executor import ContextChunkingPolicy, ExecutorConfig
from tensorrt_llm.bindings.internal.batch_manager import ContextChunkingConfig
from tensorrt_llm.logger import logger
from tensorrt_llm.mapping import Mapping

from ._util import (create_kv_cache_manager,
                    estimate_max_kv_cache_tokens_maybe_update_executor,
                    get_token_num_for_estimation, is_mla)


def create_py_executor(executor_config: ExecutorConfig,
                       checkpoint_dir: str = None,
                       engine_dir: str = None):
    if executor_config.pytorch_backend_config is None:
        executor_config.pytorch_backend_config = PyTorchConfig()

    pytorch_backend_config = executor_config.pytorch_backend_config
    spec_config = executor_config.speculative_config

    if executor_config.mapping is None:
        mapping = Mapping(world_size=tensorrt_llm.mpi_world_size(),
                          tp_size=tensorrt_llm.mpi_world_size(),
                          gpus_per_node=tensorrt_llm.default_gpus_per_node(),
                          rank=tensorrt_llm.mpi_rank())
    else:
        mapping = copy.deepcopy(executor_config.mapping)
        mapping.rank = tensorrt_llm.mpi_rank()

    if pytorch_backend_config.attn_backend in [
            "FLASHINFER", "FLASHINFER_STAR_ATTENTION"
    ]:
        # Workaround for flashinfer and star attention
        if executor_config.kv_cache_config.enable_block_reuse:
            logger.warning(
                f"Disabling block reuse for {pytorch_backend_config.attn_backend} backend"
            )
            executor_config.kv_cache_config.enable_block_reuse = False

    if pytorch_backend_config.attn_backend in [
            "FLASHINFER", "FLASHINFER_STAR_ATTENTION"
    ] and executor_config.enable_chunked_context:
        logger.warning(
            f"Disabling chunked context for {pytorch_backend_config.attn_backend} backend"
        )
        executor_config.enable_chunked_context = False

    if executor_config.max_num_tokens is None:
        executor_config.max_num_tokens = 8192
    dist = MPIDist(mapping=mapping)

    attn_runtime_features = AttentionRuntimeFeatures(
        chunked_prefill=executor_config.enable_chunked_context,
        cache_reuse=executor_config.kv_cache_config.enable_block_reuse,
    )

    model_engine = PyTorchModelEngine(
        checkpoint_dir,
        pytorch_backend_config,
        batch_size=executor_config.max_batch_size,
        max_num_tokens=executor_config.max_num_tokens,
        max_seq_len=executor_config.max_seq_len,
        mapping=mapping,
        attn_runtime_features=attn_runtime_features,
        dist=dist,
        spec_config=spec_config,
        guided_decoding_config=executor_config.guided_decoding_config,
    )
    # PyTorchModelEngine modifies these fields, update them to executor_config
    max_seq_len = model_engine.max_seq_len
    origin_seq_len = max_seq_len
    if pytorch_backend_config.enable_overlap_scheduler:
        max_seq_len = model_engine.max_seq_len + 1
        if spec_config is not None:
            max_seq_len += spec_config.max_draft_tokens

    if spec_config is not None:
        max_seq_len += spec_config.num_extra_kv_tokens
    executor_config.max_seq_len = max_seq_len
    executor_config.max_num_tokens = model_engine.max_num_tokens
    spec_config = model_engine.spec_config
    if not model_engine.model.model_config.is_generation:
        #NOTE: non-generation models do not have kv cache
        executor_config.pytorch_backend_config.use_kv_cache = False
    if executor_config.enable_chunked_context:
        chunk_unit_size = executor_config.tokens_per_block
        chunking_policy = (
            executor_config.scheduler_config.context_chunking_policy
            if executor_config.scheduler_config.context_chunking_policy
            is not None else ContextChunkingPolicy.FIRST_COME_FIRST_SERVED)
        ctx_chunk_config = ContextChunkingConfig(chunking_policy,
                                                 chunk_unit_size)
    else:
        ctx_chunk_config = None

    config = model_engine.model.model_config.pretrained_config

    kv_cache_manager = None
    use_kv_cache_manager = model_engine.model.model_config.is_generation and executor_config.pytorch_backend_config.use_kv_cache
    origin_executor_config = copy.deepcopy(executor_config)
    if use_kv_cache_manager:
        # Don't change kv_cache_config.max_tokens for CP because it will impact kv cache tokens and
        # it doesn't accept None to set its value.
        if 'cp_type' not in mapping.cp_config:
            executor_config.kv_cache_config.max_tokens = get_token_num_for_estimation(
                executor_config)

        kv_cache_manager = create_kv_cache_manager(executor_config, mapping,
                                                   model_engine)
        if model_engine.attn_metadata is not None and kv_cache_manager is not None:
            model_engine.attn_metadata.kv_cache_manager = kv_cache_manager

    resources = {
        "kv_cache_manager": kv_cache_manager
    } if kv_cache_manager is not None else {}

    if spec_config is not None:
        spec_resource_manager = get_spec_resource_manager(
            spec_config, model_engine.model.config, model_engine.batch_size * 2)
        spec_decoder = get_spec_decoder(max_seq_len=model_engine.max_seq_len,
                                        spec_config=spec_config)
        if spec_resource_manager is not None:
            resources["spec_resource_manager"] = spec_resource_manager
    else:
        spec_decoder = None

    if mapping.is_last_pp_rank(
    ) and executor_config.guided_decoding_config is not None:
        if spec_config is not None:
            raise ValueError(
                "Guided decoding does not support with speculative decoding.")
        resources[
            "guided_decoder_resource_manager"] = GuidedDecoderResourceManager(
                executor_config.max_batch_size)

    logger.info(
        f"max_seq_len={executor_config.max_seq_len}, max_num_requests={executor_config.max_batch_size}, max_num_tokens={executor_config.max_num_tokens}"
    )

    for key, value in pytorch_backend_config.extra_resource_managers.items():
        if key in resources:
            raise ValueError(
                f"Cannot overwrite existing resource manager {key}.")
        resources[key] = value

    resource_manager = ResourceManager(resources)
    # Make sure the kv cache manager is always invoked last as it could
    # depend on the results of other resource managers.
    if kv_cache_manager is not None:
        resource_manager.resource_managers.move_to_end("kv_cache_manager",
                                                       last=True)

    num_micro_batches = 1
    if mapping.has_pp:
        num_micro_batches = mapping.pp_size + pytorch_backend_config.enable_overlap_scheduler

    capacity_scheduler = BindCapacityScheduler(
        executor_config.max_batch_size,
        kv_cache_manager.impl if kv_cache_manager is not None else None,
        executor_config.scheduler_config.capacity_scheduler_policy,
        num_micro_batches=num_micro_batches)
    mb_scheduler = BindMicroBatchScheduler(executor_config.max_batch_size,
                                           executor_config.max_num_tokens,
                                           ctx_chunk_config)
    scheduler = SimpleScheduler(capacity_scheduler, mb_scheduler)
    attention_type = AttentionTypeCpp.MLA if is_mla(
        config) else AttentionTypeCpp.DEFAULT
    kv_cache_transceiver = create_kv_cache_transceiver(mapping,
                                                       kv_cache_manager,
                                                       attention_type)
    if mapping.cp_config.get('cp_type') == 'star_attention':
        assert pytorch_backend_config.attn_backend == "FLASHINFER_STAR_ATTENTION", "attention backend of star attention should be 'FLASHINFER_STAR_ATTENTION'"
        decoder = TorchStarAttentionDecoder(
            max_seq_len=model_engine.max_seq_len)
    elif spec_decoder is not None:
        decoder = spec_decoder
    elif pytorch_backend_config.enable_trtllm_decoder:
        decoder = TRTLLMDecoder(executor_config, model_engine.model,
                                model_engine.dtype, mapping,
                                tllm.executor.DecodingMode.TopKTopP())
    else:
        # NOTE: choose decoder based on model type
        if not model_engine.model.model_config.is_generation:
            decoder = EarlyStopDecoder()
        else:
            decoder = TorchDecoder(
                max_seq_len=model_engine.max_seq_len,
                mixed_decoder=pytorch_backend_config.mixed_decoder)
    py_executor = PyExecutor(resource_manager,
                             scheduler,
                             model_engine=model_engine,
                             decoder=decoder,
                             dist=dist,
                             enable_overlap_scheduler=pytorch_backend_config.
                             enable_overlap_scheduler,
                             max_batch_size=executor_config.max_batch_size,
                             kv_cache_transceiver=kv_cache_transceiver)

    if use_kv_cache_manager:
        estimate_max_kv_cache_tokens_maybe_update_executor(
            py_executor, model_engine, origin_executor_config, mapping,
            origin_seq_len, resources, ctx_chunk_config)

    return py_executor
