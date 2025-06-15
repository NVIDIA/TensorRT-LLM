import copy
import enum
from contextlib import contextmanager
from dataclasses import dataclass
from itertools import chain
from typing import Optional

import torch

import tensorrt_llm
from tensorrt_llm._utils import get_sm_version
from tensorrt_llm.bindings.executor import ContextChunkingPolicy, ExecutorConfig
from tensorrt_llm.bindings.internal.batch_manager import ContextChunkingConfig
from tensorrt_llm.logger import logger
from tensorrt_llm.lora_manager import LoraConfig
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.quantization import QuantAlgo

from ..attention_backend.interface import AttentionRuntimeFeatures
from ..distributed import MPIDist
from ..speculative import NGramConfig, get_spec_resource_manager
from ._util import (KvCacheCreator, create_py_executor_instance,
                    instantiate_sampler, is_mla)
from .config import PyTorchConfig
from .config_utils import is_mla
from .model_engine import DRAFT_KV_CACHE_MANAGER_KEY, PyTorchModelEngine
from .py_executor import PyExecutor


class _ExecutorCreationStage(enum.Enum):
    SAMPLER = "Sampler"
    INIT_KV_CACHE = "Initial KV cache (temporary for KV cache size estimation)"
    INIT_EXTRA_RESOURCES = "Additional executor resources (temporary for KV cache size estimation)"
    MODEL_EXTRA = "Model resources created during usage"
    EXTRA_RESOURCES = "Additional executor resources"
    KV_CACHE = "KV cache"
    MODEL_ENGINE_MAIN = "Model"
    MODEL_ENGINE_DRAFT = "Draft model for speculative decoding"


class _ExecutorMemoryMonitor():
    """Currently this focuses on tracking memory usage and related errors."""

    @dataclass(frozen=True)
    class _GpuMemoryUsageSample:
        creation_stage: _ExecutorCreationStage
        free_gpu_memory_bytes_pre: int
        free_gpu_memory_bytes_post: int

    def __init__(self):
        self._total_gpu_memory_bytes = torch.cuda.mem_get_info()[1]
        self._samples: list["_ExecutorMemoryMonitor._GpuMemoryUsageSample"] = []

    @staticmethod
    def _bytes_to_gib(bytes: int) -> float:
        return bytes / (1024)**3

    def _maybe_explain_if_oom(self, e: Exception, *,
                              current_stage: _ExecutorCreationStage,
                              free_gpu_memory_bytes_pre: int) -> Optional[str]:
        if isinstance(e, torch.OutOfMemoryError) or "out of memory" in str(e):
            msg = "Executor creation failed due to insufficient GPU memory."
        elif (isinstance(e, RuntimeError) and "Failed, NCCL error" in str(e)
              and "unhandled cuda error (run with NCCL_DEBUG=INFO for details)"
              in str(e)):
            msg = (
                "Executor creation failed with an error which might indicate "
                "insufficient GPU memory.")
        else:
            return None

        # how to reduce component memory usage
        tuning_knobs = {
            _ExecutorCreationStage.SAMPLER:
            "reduce max_seq_len and/or max_attention_window_size",
            _ExecutorCreationStage.KV_CACHE:
            "reduce free_gpu_memory_fraction",
            _ExecutorCreationStage.INIT_KV_CACHE:
            "reduce max_num_tokens",
            _ExecutorCreationStage.MODEL_ENGINE_MAIN:
            ("reduce max_num_tokens and/or shard the model weights across GPUs by enabling "
             "pipeline and/or tensor parallelism"),
            _ExecutorCreationStage.MODEL_ENGINE_DRAFT:
            ("reduce max_num_tokens and/or shard the model weights across GPUs by enabling "
             "pipeline and/or tensor parallelism"),
            _ExecutorCreationStage.INIT_EXTRA_RESOURCES:
            "reduce max_num_tokens",
            _ExecutorCreationStage.EXTRA_RESOURCES:
            "reduce max_num_tokens",
            _ExecutorCreationStage.MODEL_EXTRA:
            "reduce max_num_tokens",
        }

        msg = "\n".join([
            msg,
            "",
            f"The following component could not be created: {current_stage.value}",
            f"Total GPU memory (GiB): {self._bytes_to_gib(self._total_gpu_memory_bytes):.2f}",
            f"Free GPU memory before component creation attempt (GiB): {self._bytes_to_gib(free_gpu_memory_bytes_pre):.2f}",
            "",
            "Previously created components and free GPU memory before/after creation (GiB):",
            *((f"{sample.creation_stage.value}: "
               f"{self._bytes_to_gib(sample.free_gpu_memory_bytes_pre):.2f} / {self._bytes_to_gib(sample.free_gpu_memory_bytes_post):.2f}"
               ) for sample in self._samples),
            "",
            ("Please refer to the TensorRT-LLM documentation for information on how "
             "to control the memory usage through TensorRT-LLM configuration options. "
             "Possible options include:"),
            *(f"  {stage.value}: {tuning_knobs[stage]}"
              for stage in chain((sample.creation_stage
                                  for sample in self._samples), [current_stage])
              if stage in tuning_knobs),
        ])
        return msg

    @contextmanager
    def observe_creation_stage(self, current_stage: _ExecutorCreationStage):
        """Catches OOM and prints instructive message."""

        free_gpu_memory_bytes_pre = torch.cuda.mem_get_info()[0]

        try:
            yield
        except Exception as e:
            explanation = self._maybe_explain_if_oom(
                e,
                current_stage=current_stage,
                free_gpu_memory_bytes_pre=free_gpu_memory_bytes_pre)
            if explanation is None:
                raise  # not an OOM
            raise RuntimeError(explanation) from e
        else:
            free_gpu_memory_bytes_post = torch.cuda.mem_get_info()[0]
            self._samples.append(
                self._GpuMemoryUsageSample(
                    creation_stage=current_stage,
                    free_gpu_memory_bytes_pre=free_gpu_memory_bytes_pre,
                    free_gpu_memory_bytes_post=free_gpu_memory_bytes_post,
                ))


def _mangle_executor_config(executor_config: ExecutorConfig):
    if executor_config.pytorch_backend_config is None:
        executor_config.pytorch_backend_config = PyTorchConfig()
    pytorch_backend_config = executor_config.pytorch_backend_config

    if executor_config.max_num_tokens is None:
        executor_config.max_num_tokens = 8192

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


def _get_mapping(executor_config: ExecutorConfig) -> Mapping:
    if executor_config.mapping is None:
        mapping = Mapping(world_size=tensorrt_llm.mpi_world_size(),
                          tp_size=tensorrt_llm.mpi_world_size(),
                          gpus_per_node=tensorrt_llm.default_gpus_per_node(),
                          rank=tensorrt_llm.mpi_rank())
    else:
        mapping = copy.deepcopy(executor_config.mapping)
        mapping.rank = tensorrt_llm.mpi_rank()
    return mapping


def create_py_executor(executor_config: ExecutorConfig,
                       checkpoint_dir: str = None,
                       engine_dir: str = None,
                       lora_config: Optional[LoraConfig] = None) -> PyExecutor:
    _mangle_executor_config(executor_config)
    pytorch_backend_config = executor_config.pytorch_backend_config

    mapping = _get_mapping(executor_config)

    dist = MPIDist(mapping=mapping)

    spec_config = executor_config.speculative_config
    has_draft_model_engine = False
    if spec_config is not None:
        has_draft_model_engine = spec_config.spec_dec_mode.has_draft_model()
    has_ngram_drafter = isinstance(spec_config, NGramConfig)

    attn_runtime_features = AttentionRuntimeFeatures(
        chunked_prefill=executor_config.enable_chunked_context,
        cache_reuse=executor_config.kv_cache_config.enable_block_reuse,
        has_speculative_draft_tokens=has_draft_model_engine
        or has_ngram_drafter,
    )
    logger.info("ATTENTION RUNTIME FEATURES: ", attn_runtime_features)

    mem_monitor = _ExecutorMemoryMonitor()
    with mem_monitor.observe_creation_stage(
            _ExecutorCreationStage.MODEL_ENGINE_MAIN):
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
        with mem_monitor.observe_creation_stage(
                _ExecutorCreationStage.MODEL_ENGINE_DRAFT):
            draft_spec_config = copy.copy(spec_config)
            # The draft model won't have any draft tokens attached to
            # generation requests when we invoke it autoregressively
            draft_spec_config.max_draft_tokens = 0

            draft_model_engine = PyTorchModelEngine(
                spec_config.draft_model_path,
                pytorch_backend_config,
                batch_size=executor_config.max_batch_size,
                max_num_tokens=executor_config.max_num_tokens,
                # Note: The draft model engine will infer its own max_seq_len.
                # We'll stop drafting when we hit the max.
                max_seq_len=executor_config.max_seq_len,
                mapping=mapping,
                attn_runtime_features=attn_runtime_features,
                dist=dist,
                spec_config=draft_spec_config,
            )
            draft_model_engine.kv_cache_manager_key = DRAFT_KV_CACHE_MANAGER_KEY
            draft_model_engine.load_weights_from_target_model(
                model_engine.model)
    else:
        draft_model_engine = None

    # PyTorchModelEngine modifies these fields, update them to executor_config
    max_seq_len = model_engine.max_seq_len
    net_max_seq_len = max_seq_len
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

    if executor_config.enable_chunked_context:
        chunk_unit_size = executor_config.tokens_per_block
        chunking_policy = (
            executor_config.scheduler_config.context_chunking_policy
            if executor_config.scheduler_config.context_chunking_policy
            is not None else ContextChunkingPolicy.FIRST_COME_FIRST_SERVED)
        assert chunk_unit_size is not None, "chunk_unit_size must be set"
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
                f"KV cache reuse for MLA can only be enabled on SM90/SM100, "
                f"disable enable_block_reuse for SM{get_sm_version()}")
            executor_config.kv_cache_config.enable_block_reuse = False

        kv_cache_quant_algo = model_engine.model.model_config.quant_config.kv_cache_quant_algo
        if executor_config.kv_cache_config.enable_block_reuse and not (
                kv_cache_quant_algo is None or kv_cache_quant_algo
                == QuantAlgo.NO_QUANT or kv_cache_quant_algo == QuantAlgo.FP8):
            logger.warning(
                f"KV cache reuse for MLA can only be enabled without KV cache quantization or with FP8 quantization, "
                f"disable enable_block_reuse for KV cache quant algorithm: {kv_cache_quant_algo}"
            )
            executor_config.kv_cache_config.enable_block_reuse = False

        executor_config.enable_chunked_context = False

    with mem_monitor.observe_creation_stage(_ExecutorCreationStage.SAMPLER):
        sampler = instantiate_sampler(model_engine, executor_config,
                                      pytorch_backend_config, mapping)

    resources = {}
    estimating_kv_cache = False
    kv_cache_creator = None
    if model_engine.model.model_config.is_generation:
        #NOTE: non-generation models do not have kv cache
        kv_cache_creator = KvCacheCreator(executor_config=executor_config,
                                          model_engine=model_engine,
                                          draft_model_engine=draft_model_engine,
                                          mapping=mapping,
                                          net_max_seq_len=net_max_seq_len)
        estimating_kv_cache = kv_cache_creator.try_prepare_estimation()
        with mem_monitor.observe_creation_stage(
                _ExecutorCreationStage.INIT_KV_CACHE
                if estimating_kv_cache else _ExecutorCreationStage.KV_CACHE):
            kv_cache_creator.build_managers(resources)

    # resource managers for speculative decoding
    if spec_config is not None:
        spec_resource_manager = get_spec_resource_manager(
            spec_config, model_engine.model.config, model_engine.batch_size * 2)
        if spec_resource_manager is not None:
            resources["spec_resource_manager"] = spec_resource_manager

    with mem_monitor.observe_creation_stage(
            _ExecutorCreationStage.INIT_EXTRA_RESOURCES
            if estimating_kv_cache else _ExecutorCreationStage.EXTRA_RESOURCES):
        py_executor = create_py_executor_instance(
            dist, resources, mapping, pytorch_backend_config, executor_config,
            ctx_chunk_config, model_engine, draft_model_engine, False, sampler,
            lora_config)

    if estimating_kv_cache:
        assert kv_cache_creator is not None
        with mem_monitor.observe_creation_stage(
                _ExecutorCreationStage.MODEL_EXTRA):
            kv_cache_creator.estimate_max_tokens(py_executor)
        kv_cache_creator.teardown_managers(resources)
        del py_executor  # free before constructing new

        with mem_monitor.observe_creation_stage(
                _ExecutorCreationStage.KV_CACHE):
            # Before estimating KV cache size, a minimal KV cache has been allocated using
            # create_kv_cache_manager above, which caps executor_config.max_seq_len. Restoring
            # the original value before creating the final KV cache.
            executor_config.max_seq_len = max_seq_len
            kv_cache_creator.build_managers(resources)

            for eng in [model_engine, draft_model_engine]:
                if eng is None:
                    continue
                if eng.attn_metadata is not None:
                    if pytorch_backend_config.use_cuda_graph:
                        eng._release_cuda_graphs()
                    eng.attn_metadata = None

        with mem_monitor.observe_creation_stage(
                _ExecutorCreationStage.EXTRA_RESOURCES):
            py_executor = create_py_executor_instance(
                dist, resources, mapping, pytorch_backend_config,
                executor_config, ctx_chunk_config, model_engine,
                draft_model_engine, False, sampler, lora_config)

    py_executor.start_worker()
    return py_executor
