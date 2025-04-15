import copy
from typing import Optional

import tensorrt_llm
from tensorrt_llm._utils import get_sm_version
from tensorrt_llm.bindings.executor import ContextChunkingPolicy, ExecutorConfig
from tensorrt_llm.bindings.internal.batch_manager import ContextChunkingConfig
from tensorrt_llm.logger import logger
from tensorrt_llm.lora_manager import LoraConfig
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.quantization import KV_CACHE_QUANT_ALGO_LIST

from ..attention_backend.interface import AttentionRuntimeFeatures
from ..distributed import MPIDist
from ..speculative import Eagle3Config, get_spec_resource_manager
from ._util import (create_kv_cache_manager, create_py_executor_instance,
                    estimate_max_kv_cache_tokens, get_token_num_for_estimation,
                    instantiate_sampler, is_mla)
from .config import PyTorchConfig
from .config_utils import is_mla
from .model_engine import (DRAFT_KV_CACHE_MANAGER_KEY, KV_CACHE_MANAGER_KEY,
                           PyTorchModelEngine)
from .py_executor import PyExecutor


def create_py_executor(executor_config: ExecutorConfig,
                       checkpoint_dir: str = None,
                       engine_dir: str = None,
                       lora_config: Optional[LoraConfig] = None) -> PyExecutor:
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

    if pytorch_backend_config.attn_backend == "FLASHINFER_STAR_ATTENTION" and executor_config.enable_chunked_context:
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
        lora_config=lora_config,
    )

    if has_draft_model_engine:
        draft_spec_config = copy.copy(spec_config)
        # The draft model won't have any draft tokens attached to
        # generation requests when we invoke it autoregressively
        draft_spec_config.max_draft_tokens = 0

        draft_model_engine = PyTorchModelEngine(
            spec_config.eagle_weights_path,
            pytorch_backend_config,
            batch_size=executor_config.max_batch_size,
            max_num_tokens=executor_config.max_num_tokens,
            max_seq_len=model_engine.max_seq_len,
            mapping=mapping,
            attn_runtime_features=attn_runtime_features,
            dist=dist,
            spec_config=draft_spec_config,
        )
        draft_model_engine.kv_cache_manager_key = DRAFT_KV_CACHE_MANAGER_KEY
        draft_model_engine.load_weights_from_target_model(model_engine.model)
    else:
        draft_model_engine = None

    # PyTorchModelEngine modifies these fields, update them to executor_config
    max_seq_len = model_engine.max_seq_len
    origin_seq_len = max_seq_len
    if not pytorch_backend_config.disable_overlap_scheduler:
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

        if executor_config.kv_cache_config.enable_block_reuse and not (
                get_sm_version() >= 90 and get_sm_version() <= 100):
            logger.warning(
                f"KV cache reuse for MLA only can be enabled on SM90/SM100, "
                f"disable enable_block_reuse for SM{get_sm_version()}")
            executor_config.kv_cache_config.enable_block_reuse = False

        kv_cache_quant_algo = model_engine.model.model_config.quant_config.kv_cache_quant_algo
        if executor_config.kv_cache_config.enable_block_reuse and kv_cache_quant_algo in KV_CACHE_QUANT_ALGO_LIST:
            logger.warning(
                f"KV cache reuse for MLA only can be enabled without KV cache quantization, "
                f"disable enable_block_reuse for KV cache quant algorithm: {kv_cache_quant_algo}"
            )
            executor_config.kv_cache_config.enable_block_reuse = False

        executor_config.enable_chunked_context = False

    sampler = instantiate_sampler(model_engine, executor_config,
                                  pytorch_backend_config, mapping)

    kv_cache_manager = None
    draft_kv_cache_manager = None
    resources = {}
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
        resources[KV_CACHE_MANAGER_KEY] = kv_cache_manager
        resources[DRAFT_KV_CACHE_MANAGER_KEY] = draft_kv_cache_manager

    # resource managers for speculative decoding
    if spec_config is not None:
        spec_resource_manager = get_spec_resource_manager(
            spec_config, model_engine.model.config, model_engine.batch_size * 2)
        if spec_resource_manager is not None:
            resources["spec_resource_manager"] = spec_resource_manager

    py_executor = create_py_executor_instance(dist, resources, mapping,
                                              pytorch_backend_config,
                                              executor_config, ctx_chunk_config,
                                              model_engine, draft_model_engine,
                                              False, sampler, lora_config)

    if executor_config.pytorch_backend_config.use_kv_cache and 'cp_type' not in mapping.cp_config:
        kv_cache_max_tokens = estimate_max_kv_cache_tokens(
            py_executor, model_engine, origin_executor_config, mapping,
            origin_seq_len, ctx_chunk_config, draft_model_engine)
        # This may be None if no max number tokens set and enable cp.
        if kv_cache_max_tokens is not None:
            del py_executor  # free before constructing new
            del kv_cache_manager  # free before constructing new

            executor_config.kv_cache_config.max_tokens = kv_cache_max_tokens

            kv_cache_manager = create_kv_cache_manager(model_engine, mapping,
                                                       executor_config)
            resources[KV_CACHE_MANAGER_KEY] = kv_cache_manager

            if model_engine.attn_metadata is not None:
                if pytorch_backend_config.use_cuda_graph:
                    model_engine._release_cuda_graphs()
                del model_engine.attn_metadata
                model_engine.attn_metadata = None

            if draft_model_engine is not None:
                del draft_kv_cache_manager  # free before constructing new
                draft_kv_cache_manager = create_kv_cache_manager(
                    draft_model_engine, mapping, executor_config)
                resources[DRAFT_KV_CACHE_MANAGER_KEY] = draft_kv_cache_manager
                if draft_model_engine.attn_metadata is not None:
                    if pytorch_backend_config.use_cuda_graph:
                        draft_model_engine._release_cuda_graphs()
                    del draft_model_engine.attn_metadata
                    draft_model_engine.attn_metadata = None

            py_executor = create_py_executor_instance(
                dist, resources, mapping, pytorch_backend_config,
                executor_config, ctx_chunk_config, model_engine,
                draft_model_engine, False, sampler, lora_config)

    py_executor.start_worker()
    return py_executor
