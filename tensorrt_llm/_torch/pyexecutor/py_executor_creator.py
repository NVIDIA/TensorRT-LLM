import copy

import tensorrt_llm
from tensorrt_llm._torch.attention_backend.interface import \
    AttentionRuntimeFeatures
from tensorrt_llm._torch.pyexecutor.config import PyTorchConfig
from tensorrt_llm._torch.pyexecutor.decoder import (TorchDecoder,
                                                    TorchStarAttentionDecoder)
from tensorrt_llm._torch.pyexecutor.distributed import MPIDist
from tensorrt_llm._torch.pyexecutor.kv_cache_transceiver import \
    create_kv_cache_transceiver
from tensorrt_llm._torch.pyexecutor.model_engine import PyTorchModelEngine
from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor
from tensorrt_llm._torch.pyexecutor.resource_manager import (KVCacheManager,
                                                             MLAKVCacheManager,
                                                             ResourceManager)
from tensorrt_llm._torch.pyexecutor.scheduler import (BindCapacityScheduler,
                                                      BindMicroBatchScheduler,
                                                      SimpleScheduler)
from tensorrt_llm._utils import str_dtype_to_binding, torch_dtype_to_str
from tensorrt_llm.bindings.executor import ContextChunkingPolicy, ExecutorConfig
from tensorrt_llm.bindings.internal.batch_manager import ContextChunkingConfig
from tensorrt_llm.logger import logger
from tensorrt_llm.mapping import Mapping

from ._util import estimate_max_kv_cache_tokens, is_mla


def create_py_executor(executor_config: ExecutorConfig,
                       checkpoint_dir: str = None,
                       engine_dir: str = None):
    if executor_config.pytorch_backend_config is None:
        executor_config.pytorch_backend_config = PyTorchConfig()

    pytorch_backend_config = executor_config.pytorch_backend_config

    if executor_config.mapping is None:
        mapping = Mapping(world_size=tensorrt_llm.mpi_world_size(),
                          tp_size=tensorrt_llm.mpi_world_size(),
                          rank=tensorrt_llm.mpi_rank())
    else:
        mapping = copy.deepcopy(executor_config.mapping)
        mapping.rank = tensorrt_llm.mpi_rank()
    if mapping.cp_config.get('cp_type') == 'star_attention':
        assert pytorch_backend_config.attn_backend == "FLASHINFER_STAR_ATTENTION", "attention backend of star attention should be 'FLASHINFER_STAR_ATTENTION'"
        decoder = TorchStarAttentionDecoder()
    else:
        decoder = TorchDecoder(
            mixed_decoder=pytorch_backend_config.mixed_decoder)

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
    )
    # PyTorchModelEngine modifies these fields, update them to executor_config
    max_seq_len = model_engine.max_seq_len + 1 if pytorch_backend_config.enable_overlap_scheduler else model_engine.max_seq_len
    executor_config.max_seq_len = max_seq_len
    executor_config.max_num_tokens = model_engine.max_num_tokens

    hidden_size = model_engine.model.config.hidden_size
    num_attention_heads = model_engine.model.config.num_attention_heads
    num_key_value_heads = getattr(model_engine.model.config,
                                  'num_key_value_heads', num_attention_heads)
    head_dim = hidden_size // num_attention_heads

    quant_config = model_engine.model.model_config.quant_config
    if quant_config is not None and quant_config.quant_mode.has_fp8_kv_cache():
        kv_cache_dtype = tensorrt_llm.bindings.DataType.FP8
    else:
        kv_cache_dtype = str_dtype_to_binding(
            torch_dtype_to_str(model_engine.dtype))

    kv_cache_max_tokens = estimate_max_kv_cache_tokens(model_engine,
                                                       executor_config, mapping)

    if kv_cache_max_tokens is not None:
        executor_config.kv_cache_config.max_tokens = kv_cache_max_tokens

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
        executor_config.kv_cache_config.enable_block_reuse = False
        executor_config.enable_chunked_context = False
        kv_cache_manager = MLAKVCacheManager(
            executor_config.kv_cache_config,
            tensorrt_llm.bindings.internal.batch_manager.CacheType.SELFKONLY,
            model_engine.model.config.num_hidden_layers,
            num_attention_heads,
            num_key_value_heads,
            head_dim,
            executor_config.tokens_per_block,
            executor_config.max_seq_len,
            executor_config.max_batch_size,
            mapping,
            dtype=kv_cache_dtype,
            kv_lora_rank=config.kv_lora_rank,
            qk_rope_head_dim=config.qk_rope_head_dim)
    else:
        num_hidden_layers = model_engine.model.config.num_hidden_layers
        # the number of layers using attention in Nemotron5 is lower from the number of hidden layers
        if model_engine.model.config.architectures[0] == "Nemotron5ForCausalLM":
            # attention layers are derived from configuration (hybrid_override_pattern)
            num_hidden_layers = model_engine.model.config.hybrid_override_pattern.count(
                "*")
        kv_cache_manager = KVCacheManager(
            executor_config.kv_cache_config,
            tensorrt_llm.bindings.internal.batch_manager.CacheType.SELF,
            num_hidden_layers,
            num_attention_heads,
            num_key_value_heads,
            head_dim,
            executor_config.tokens_per_block,
            executor_config.max_seq_len,
            executor_config.max_batch_size,
            mapping,
            dtype=kv_cache_dtype,
        )

    # KVCacheManager modifies these fields, update them to executor_config
    executor_config.max_seq_len = kv_cache_manager.max_seq_len

    resources = {"kv_cache_manager": kv_cache_manager}

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
    resource_manager.resource_managers.move_to_end("kv_cache_manager",
                                                   last=True)

    capacity_scheduler = BindCapacityScheduler(
        executor_config.max_batch_size, kv_cache_manager.impl,
        executor_config.scheduler_config.capacity_scheduler_policy)
    mb_scheduler = BindMicroBatchScheduler(executor_config.max_batch_size,
                                           executor_config.max_num_tokens,
                                           ctx_chunk_config)
    scheduler = SimpleScheduler(capacity_scheduler, mb_scheduler)
    kv_cache_transceiver = create_kv_cache_transceiver(mapping,
                                                       kv_cache_manager)

    py_executor = PyExecutor(resource_manager,
                             scheduler,
                             model_engine=model_engine,
                             decoder=decoder,
                             dist=dist,
                             enable_overlap_scheduler=pytorch_backend_config.
                             enable_overlap_scheduler,
                             kv_cache_transceiver=kv_cache_transceiver)
    return py_executor
