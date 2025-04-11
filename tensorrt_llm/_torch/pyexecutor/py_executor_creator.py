import copy

import tensorrt_llm
from tensorrt_llm.bindings.executor import ContextChunkingPolicy, ExecutorConfig
from tensorrt_llm.bindings.internal.batch_manager import ContextChunkingConfig
from tensorrt_llm.logger import logger
from tensorrt_llm.mapping import Mapping

from ..attention_backend.interface import AttentionRuntimeFeatures
from ..speculative import Eagle3Config
from ._util import (create_kv_cache_manager, create_py_executor_instance,
                    estimate_max_kv_cache_tokens, get_token_num_for_estimation,
                    is_mla)
from .config import PyTorchConfig
from .distributed import MPIDist
from .model_engine import DRAFT_KV_CACHE_MANAGER_KEY, PyTorchModelEngine


def create_py_executor(executor_config: ExecutorConfig,
                       checkpoint_dir: str = None,
                       engine_dir: str = None):
    if executor_config.pytorch_backend_config is None:
        executor_config.pytorch_backend_config = PyTorchConfig()

    pytorch_backend_config = executor_config.pytorch_backend_config

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

    spec_config = executor_config.speculative_config
    has_draft_model_engine = isinstance(spec_config, Eagle3Config)

    attn_runtime_features = AttentionRuntimeFeatures(
        chunked_prefill=executor_config.enable_chunked_context,
        cache_reuse=executor_config.kv_cache_config.enable_block_reuse,
        has_speculative_draft_tokens=has_draft_model_engine,
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

    draft_model_engine = None
    if has_draft_model_engine:
        draft_model_engine = PyTorchModelEngine(
            spec_config.eagle_weights_path,
            pytorch_backend_config,
            batch_size=executor_config.max_batch_size,
            max_num_tokens=executor_config.max_num_tokens,
            max_seq_len=executor_config.max_seq_len,
            mapping=mapping,
            attn_runtime_features=attn_runtime_features,
            dist=dist,
            spec_config=copy.copy(spec_config),
        )
        draft_model_engine.kv_cache_manager_key = DRAFT_KV_CACHE_MANAGER_KEY

    # PyTorchModelEngine modifies these fields, update them to executor_config
    max_seq_len = model_engine.max_seq_len
    origin_seq_len = max_seq_len
    if pytorch_backend_config.enable_overlap_scheduler:
        max_seq_len = model_engine.max_seq_len + 1
        if spec_config is not None:
            max_seq_len += spec_config.max_draft_tokens

    if spec_config is not None:
        max_seq_len += spec_config.num_extra_kv_tokens
        max_seq_len += spec_config.max_draft_tokens

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
    if is_mla(config):
        if model_engine.model.model_config.enable_flash_mla:
            executor_config.tokens_per_block = 64
            logger.info(
                f"Change tokens_per_block to: {executor_config.tokens_per_block} for using FlashMLA"
            )
        executor_config.kv_cache_config.enable_block_reuse = False
        executor_config.enable_chunked_context = False

    kv_cache_manager = None
    draft_kv_cache_manager = None
    origin_executor_config = copy.deepcopy(executor_config)
    if executor_config.pytorch_backend_config.use_kv_cache:
        if 'cp_type' not in mapping.cp_config:
            executor_config.kv_cache_config.max_tokens = get_token_num_for_estimation(
                executor_config, model_engine.model.model_config)
        kv_cache_manager = create_kv_cache_manager(model_engine, mapping,
                                                   executor_config)
        draft_kv_cache_manager = create_kv_cache_manager(
            draft_model_engine, mapping,
            executor_config) if draft_model_engine is not None else None

    # KVCacheManager modifies these fields, update them to executor_config
    if kv_cache_manager is not None:
        executor_config.max_seq_len = kv_cache_manager.max_seq_len

    py_executor = create_py_executor_instance(dist, kv_cache_manager,
                                              draft_kv_cache_manager, mapping,
                                              pytorch_backend_config,
                                              executor_config, ctx_chunk_config,
                                              model_engine, draft_model_engine,
                                              False)

    if executor_config.pytorch_backend_config.use_kv_cache:
        kv_cache_max_tokens = estimate_max_kv_cache_tokens(
            py_executor, model_engine, origin_executor_config, mapping,
            origin_seq_len, ctx_chunk_config, draft_model_engine)
        # This may be None if no max number tokens set and enable cp.
        if kv_cache_max_tokens is not None:
            executor_config.kv_cache_config.max_tokens = kv_cache_max_tokens

            kv_cache_manager = create_kv_cache_manager(model_engine, mapping,
                                                       executor_config)

            if model_engine.attn_metadata is not None and kv_cache_manager is not None:
                if pytorch_backend_config.use_cuda_graph:
                    model_engine._release_cuda_graphs()
                del model_engine.attn_metadata
                model_engine.attn_metadata = None

            if draft_model_engine is not None:
                draft_kv_cache_manager = create_kv_cache_manager(
                    draft_model_engine, mapping, executor_config)
                if draft_model_engine.attn_metadata is not None and draft_kv_cache_manager is not None:
                    if pytorch_backend_config.use_cuda_graph:
                        draft_model_engine._release_cuda_graphs()
                    del draft_model_engine.attn_metadata
                    draft_model_engine.attn_metadata = None

            py_executor = create_py_executor_instance(
                dist, kv_cache_manager, draft_kv_cache_manager, mapping,
                pytorch_backend_config, executor_config, ctx_chunk_config,
                model_engine, draft_model_engine, True)

    py_executor.start_worker()
    return py_executor
