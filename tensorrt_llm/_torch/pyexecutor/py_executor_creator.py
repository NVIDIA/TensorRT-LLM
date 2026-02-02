import copy
import gc
import importlib
import os
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from itertools import chain
from typing import Optional, Tuple

import torch
from strenum import StrEnum

import tensorrt_llm
from tensorrt_llm._torch.pyexecutor.resource_manager import ResourceManagerType
from tensorrt_llm._utils import get_sm_version
from tensorrt_llm.llmapi.llm_args import (CapacitySchedulerPolicy,
                                          ContextChunkingPolicy,
                                          GuidedDecodingConfig, LoadFormat,
                                          TorchLlmArgs)
from tensorrt_llm.llmapi.tokenizer import (TokenizerBase,
                                           _llguidance_tokenizer_info,
                                           _xgrammar_tokenizer_info)
from tensorrt_llm.logger import logger
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.quantization import QuantAlgo
from tensorrt_llm.tools.layer_wise_benchmarks import get_calibrator

from ..attention_backend.interface import AttentionRuntimeFeatures
from ..attention_backend.trtllm import TrtllmAttention
from ..distributed import Distributed
from ..speculative import (get_num_extra_kv_tokens, get_spec_drafter,
                           get_spec_resource_manager)
from ..virtual_memory import ExecutorMemoryType, RestoreMode
from ..virtual_memory import scope as virtual_memory_scope
from ._util import (KvCacheCreator, _adjust_torch_mem_fraction,
                    create_py_executor_instance, instantiate_sampler, is_mla,
                    validate_feature_combination)
from .config_utils import is_mla
from .guided_decoder import CapturableGuidedDecoder, GuidedDecoder
from .kv_cache_connector import KvCacheConnectorManager
from .model_engine import PyTorchModelEngine
from .py_executor import PyExecutor


class _ExecutorMemoryMonitor:
    """Currently this focuses on tracking memory usage and related errors."""

    @dataclass(frozen=True)
    class _GpuMemoryUsageSample:
        creation_stage: ExecutorMemoryType
        free_gpu_memory_bytes_pre: int
        free_gpu_memory_bytes_post: int

    def __init__(self):
        self._total_gpu_memory_bytes = torch.cuda.mem_get_info()[1]
        self._samples: list["_ExecutorMemoryMonitor._GpuMemoryUsageSample"] = []

    @staticmethod
    def _bytes_to_gib(bytes: int) -> float:
        return bytes / (1024)**3

    memory_type_friendly_names = {
        ExecutorMemoryType.SAMPLER:
        "Sampler",
        ExecutorMemoryType.DRAFTER:
        "Drafter",
        ExecutorMemoryType.GUIDED_DECODER:
        "Guided Decoder",
        ExecutorMemoryType.SPEC_RESOURCES:
        "Speculative decoding resources",
        ExecutorMemoryType.INIT_KV_CACHE:
        "Initial KV Cache (temporary for KV cache size estimation)",
        ExecutorMemoryType.INIT_EXTRA_RESOURCES:
        "Additional executor resources (temporary for KV cache size estimation)",
        ExecutorMemoryType.MODEL_EXTRA:
        "Model resources created during usage",
        ExecutorMemoryType.EXTRA_RESOURCES:
        "Additional executor resources",
        ExecutorMemoryType.KV_CACHE:
        "KV cache",
        ExecutorMemoryType.MODEL_ENGINE_MAIN:
        "Model",
        ExecutorMemoryType.MODEL_ENGINE_DRAFT:
        "Draft model for speculative decoding",
    }

    # Suggestion to reduce component memory usage
    memory_type_tuning_suggestion = {
        ExecutorMemoryType.SAMPLER:
        "reduce max_seq_len and/or max_attention_window_size",
        ExecutorMemoryType.DRAFTER:
        "reduce max_seq_len and/or max_draft_len",
        ExecutorMemoryType.SPEC_RESOURCES:
        "reduce max_seq_len and/or max_batch_size",
        ExecutorMemoryType.KV_CACHE:
        "reduce free_gpu_memory_fraction",
        ExecutorMemoryType.INIT_KV_CACHE:
        "reduce max_num_tokens",
        ExecutorMemoryType.MODEL_ENGINE_MAIN:
        ("reduce max_num_tokens and/or shard the model weights across GPUs by enabling "
         "pipeline and/or tensor parallelism"),
        ExecutorMemoryType.MODEL_ENGINE_DRAFT:
        ("reduce max_num_tokens and/or shard the model weights across GPUs by enabling "
         "pipeline and/or tensor parallelism"),
        ExecutorMemoryType.INIT_EXTRA_RESOURCES:
        "reduce max_num_tokens",
        ExecutorMemoryType.EXTRA_RESOURCES:
        "reduce max_num_tokens",
        ExecutorMemoryType.MODEL_EXTRA:
        "reduce max_num_tokens",
    }

    def _maybe_explain_if_oom(self, e: Exception, *,
                              current_stage: ExecutorMemoryType,
                              free_gpu_memory_bytes_pre: int) -> Optional[str]:
        if isinstance(e, torch.OutOfMemoryError) or "out of memory" in str(e):
            msg = "Executor creation failed due to insufficient GPU memory."
        elif (isinstance(e, RuntimeError) and "Failed, NCCL error" in str(e)
              and "unhandled cuda error (run with NCCL_DEBUG=INFO for details)"
              in str(e)):
            msg = f"Executor creation failed with NCCL error: {str(e)}"
            return msg
        else:
            return None

        msg = "\n".join([
            msg,
            "",
            f"The following component could not be created: {self.memory_type_friendly_names[current_stage]}",
            f"Total GPU memory (GiB): {self._bytes_to_gib(self._total_gpu_memory_bytes):.2f}",
            f"Free GPU memory before component creation attempt (GiB): {self._bytes_to_gib(free_gpu_memory_bytes_pre):.2f}",
            "",
            "Previously created components and free GPU memory before/after creation (GiB):",
            *((f"{sample.creation_stage.value}: "
               f"{self._bytes_to_gib(sample.free_gpu_memory_bytes_pre):.2f} / {self._bytes_to_gib(sample.free_gpu_memory_bytes_post):.2f}"
               ) for sample in self._samples),
            "",
            ("Please refer to the TensorRT LLM documentation for information on how "
             "to control the memory usage through TensorRT LLM configuration options. "
             "Possible options include:"),
            *(f"  {stage.value}: {self.memory_type_tuning_suggestion[stage]}"
              for stage in chain((sample.creation_stage
                                  for sample in self._samples), [current_stage])
              if stage in self.memory_type_tuning_suggestion),
        ])
        return msg

    @contextmanager
    def observe_creation_stage(self, current_stage: ExecutorMemoryType):
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


def _get_mapping(_mapping: Mapping) -> Mapping:
    if _mapping is None:
        mapping = Mapping(world_size=tensorrt_llm.mpi_world_size(),
                          tp_size=tensorrt_llm.mpi_world_size(),
                          gpus_per_node=tensorrt_llm.default_gpus_per_node(),
                          rank=tensorrt_llm.mpi_rank())
    else:
        mapping = copy.deepcopy(_mapping)
        mapping.rank = tensorrt_llm.mpi_rank()
    return mapping


def update_sampler_max_seq_len(max_seq_len, sampler):
    # Originally, TRTLLMSampler is constructed with executor_config, but
    # _create_kv_cache_manager (via build_managers) may later overwrite executor_config.max_seq_len.
    # Because TRTLLMSampler.sample_async still needs the updated limit and executor_config is
    # deprecated inside TRTLLMSampler, keep TRTLLMSampler.max_seq_len updated with
    # with executor_config.max_seq_len.
    from .sampler import TRTLLMSampler

    if isinstance(sampler, TRTLLMSampler):
        assert hasattr(sampler, "max_seq_len")
        sampler.max_seq_len = max_seq_len


def get_guided_decoding_config(guided_decoding_backend: str,
                               tokenizer: Optional[TokenizerBase] = None):
    guided_decoding_config = None
    if guided_decoding_backend == 'xgrammar':
        assert tokenizer is not None
        guided_decoding_config = GuidedDecodingConfig(
            backend=GuidedDecodingConfig.GuidedDecodingBackend.XGRAMMAR,
            **_xgrammar_tokenizer_info(tokenizer))
    elif guided_decoding_backend == 'llguidance':
        assert tokenizer is not None
        guided_decoding_config = GuidedDecodingConfig(
            backend=GuidedDecodingConfig.GuidedDecodingBackend.LLGUIDANCE,
            **_llguidance_tokenizer_info(tokenizer))
    elif guided_decoding_backend is not None:
        raise ValueError(
            f"Unsupported guided decoding backend {guided_decoding_backend}")
    return guided_decoding_config


def create_py_executor(
    llm_args: TorchLlmArgs,
    checkpoint_dir: Optional[str] = None,
    tokenizer: Optional[TokenizerBase] = None,
    profiling_stage_data: Optional[dict] = None,
) -> PyExecutor:
    torch.cuda.set_per_process_memory_fraction(1.0)
    garbage_collection_gen0_threshold = llm_args.garbage_collection_gen0_threshold
    lora_config = llm_args.lora_config
    kv_connector_config = llm_args.kv_connector_config

    scheduler_config = llm_args.scheduler_config

    # Since peft_cache_config may be subject to change, avoid these changes propagate back
    # to llm_args.peft_cache_config
    peft_cache_config = copy.deepcopy(llm_args.peft_cache_config)

    assert llm_args.kv_cache_config, "Expect llm_args.kv_cache_config is not None"
    kv_cache_config = llm_args.kv_cache_config
    if os.getenv("FORCE_DETERMINISTIC", "0") == "1":
        # Disable KV cache reuse for deterministic mode
        kv_cache_config.enable_block_reuse = False
        kv_cache_config.enable_partial_reuse = False

    decoding_config = llm_args.decoding_config
    guided_decoding_config = get_guided_decoding_config(
        llm_args.guided_decoding_backend, tokenizer)

    mm_encoder_only = llm_args.mm_encoder_only
    enable_chunked_context = llm_args.enable_chunked_prefill

    (
        max_beam_width,
        max_num_tokens,
        max_seq_len,
        max_batch_size,
    ) = llm_args.get_runtime_sizes()

    tokens_per_block = kv_cache_config.tokens_per_block
    if llm_args.attn_backend == "VANILLA":
        tokens_per_block = max_num_tokens

    if llm_args.attn_backend in ["FLASHINFER", "FLASHINFER_STAR_ATTENTION"]:
        # Workaround for flashinfer and star attention
        if kv_cache_config.enable_block_reuse:
            logger.warning(
                f"Disabling block reuse for {llm_args.attn_backend} backend")
            kv_cache_config.enable_block_reuse = False

    if llm_args.attn_backend == "FLASHINFER_STAR_ATTENTION" and enable_chunked_context:
        logger.warning(
            f"Disabling chunked context for {llm_args.attn_backend} backend")
        enable_chunked_context = False

    spec_config = llm_args.speculative_config
    if spec_config is not None and spec_config.decoding_type == "AUTO":
        from tensorrt_llm._torch.speculative import suggest_spec_config
        spec_config = suggest_spec_config(max_batch_size)

    if not llm_args.disable_overlap_scheduler and spec_config is not None:
        if not spec_config.spec_dec_mode.support_overlap_scheduler():
            logger.warning(
                f"Disable overlap scheduler for speculation mode {spec_config.spec_dec_mode.name}"
            )
            llm_args.disable_overlap_scheduler = True

    if spec_config is not None and spec_config.spec_dec_mode.use_one_engine(
    ) and not spec_config.allow_advanced_sampling:
        logger.warning(
            f"Falling back to greedy decoding for {spec_config.decoding_type}. If you "
            "want to use non-greedy sampling, please set allow_advanced_sampling=True."
        )

    if mm_encoder_only:
        llm_args.mm_encoder_only = True
        llm_args.disable_overlap_scheduler = True

        # Disable overlap scheduler for multimodal encoder-only mode
        logger.warning(
            "Disabling overlap scheduler for multimodal encoder-only mode. "
            "The overlap scheduler is designed for generation models and is not needed "
            "when only processing vision encoder inputs.")

    mapping = _get_mapping(llm_args.parallel_config.to_mapping())
    dist = Distributed.get(mapping)

    vm_pools = {}
    enable_sleep = llm_args.enable_sleep

    cache_transceiver_config = llm_args.cache_transceiver_config

    has_draft_model_engine = False
    has_spec_drafter = False
    if spec_config is not None:
        has_draft_model_engine = spec_config.spec_dec_mode.has_draft_model()
        has_spec_drafter = spec_config.spec_dec_mode.has_spec_drafter()

    # chunk_unit_size may be changed to 64 when using flash mla
    attn_runtime_features = AttentionRuntimeFeatures(
        chunked_prefill=enable_chunked_context,
        cache_reuse=kv_cache_config.enable_block_reuse,
        has_speculative_draft_tokens=has_draft_model_engine or has_spec_drafter,
        chunk_size=max_num_tokens,
    )
    logger.info("ATTENTION RUNTIME FEATURES: ", attn_runtime_features)

    mem_monitor = _ExecutorMemoryMonitor()

    @contextmanager
    def allocation_scope(current_stage: ExecutorMemoryType,
                         restore_mode: RestoreMode):
        with mem_monitor.observe_creation_stage(current_stage):
            stage = current_stage.value
            if not enable_sleep or stage.startswith("_no_capture"):
                yield
            else:
                with virtual_memory_scope(stage, restore_mode) as memory_pool:
                    if stage in vm_pools:
                        del vm_pools[stage]
                    vm_pools[stage] = memory_pool
                    yield

    with allocation_scope(ExecutorMemoryType.MODEL_ENGINE_MAIN,
                          RestoreMode.PINNED):
        model_engine = PyTorchModelEngine(
            model_path=checkpoint_dir,
            llm_args=llm_args,
            mapping=mapping,
            attn_runtime_features=attn_runtime_features,
            dist=dist,
            spec_config=spec_config,
        )

    validate_feature_combination(llm_args, model_engine, llm_args.sampler_type)

    calibrator = get_calibrator()
    layer_wise_benchmarks_config = llm_args.layer_wise_benchmarks_config
    calibrator.init(layer_wise_benchmarks_config.calibration_mode,
                    layer_wise_benchmarks_config.calibration_file_path,
                    layer_wise_benchmarks_config.calibration_layer_indices,
                    mapping=mapping,
                    dist=dist)
    model_engine.model = calibrator.maybe_wrap_model(model_engine.model)

    if has_draft_model_engine:
        with allocation_scope(ExecutorMemoryType.MODEL_ENGINE_DRAFT,
                              RestoreMode.PINNED):
            draft_spec_config = copy.copy(spec_config)

            use_chain_drafter = (
                guided_decoding_config is None
                and draft_spec_config._allow_chain_drafter
                and draft_spec_config._allow_greedy_draft_tokens
                and llm_args.attn_backend == "TRTLLM"
                and draft_spec_config.draft_len_schedule is None)

            logger.debug(f"USE CHAIN DRAFTER: {use_chain_drafter}")
            if use_chain_drafter:

                def drafting_loop_wrapper(model):
                    from tensorrt_llm._torch.speculative.drafting_loops import (
                        LinearDraftingLoopWrapper, TreeDraftingLoopWrapper)
                    from tensorrt_llm.llmapi import EagleDecodingConfig

                    use_tree_drafter = isinstance(
                        draft_spec_config, EagleDecodingConfig
                    ) and not draft_spec_config.is_linear_tree

                    if use_tree_drafter:
                        return TreeDraftingLoopWrapper(
                            spec_config.max_draft_len,
                            spec_config.max_total_draft_tokens, max_batch_size,
                            model)
                    else:
                        return LinearDraftingLoopWrapper(
                            spec_config.max_draft_len,
                            spec_config.max_total_draft_tokens, model)
            else:
                drafting_loop_wrapper = None

            draft_llm_args = copy.copy(llm_args)
            if spec_config.load_format == "dummy":
                draft_llm_args.load_format = LoadFormat.DUMMY

            draft_model_engine = PyTorchModelEngine(
                model_path=spec_config.speculative_model,
                llm_args=draft_llm_args,
                mapping=mapping,
                attn_runtime_features=attn_runtime_features,
                dist=dist,
                spec_config=draft_spec_config,
                is_draft_model=True,
                drafting_loop_wrapper=drafting_loop_wrapper,
            )
            # For DeepseekV3 MTP, we need to set the num_hidden_layers to 1 for the draft model
            if spec_config.spec_dec_mode.is_mtp_eagle():
                draft_model_engine.model.model_config.pretrained_config.num_hidden_layers = 1
            draft_model_engine.load_weights_from_target_model(
                model_engine.model)
    else:
        draft_model_engine = None

    # TODO: Overlap scheduler is not supported for below cases:
    # 1. non-CDL is used
    # 2. non-TrtllmAttention attention backend is used
    if has_draft_model_engine and (not use_chain_drafter or not issubclass(
            draft_model_engine.attn_backend, TrtllmAttention)):
        logger.warning(
            "Overlap scheduler is not supported for non-CDL or non-TrtllmAttention backend."
        )
        llm_args.disable_overlap_scheduler = True

    # PyTorchModelEngine modifies these fields, update them
    model_engine_max_seq_len = model_engine.max_seq_len
    net_max_seq_len = model_engine_max_seq_len
    if not llm_args.disable_overlap_scheduler:
        model_engine_max_seq_len = model_engine.max_seq_len + 1
        if spec_config is not None:
            model_engine_max_seq_len += spec_config.max_total_draft_tokens

    if spec_config is not None:
        model_engine_max_seq_len += get_num_extra_kv_tokens(spec_config)
        model_engine_max_seq_len += spec_config.max_total_draft_tokens

    if has_draft_model_engine and not llm_args.disable_overlap_scheduler:
        logger.warning(
            "Overlap scheduler is enabled for two-model speculative decoding. Rejection sampling will fallback to greedy sampling."
        )

    max_seq_len = model_engine_max_seq_len
    max_num_tokens = model_engine.max_num_tokens
    sparse_attention_config = model_engine.sparse_attention_config

    config = model_engine.model.model_config.pretrained_config
    if is_mla(config):
        if model_engine.model.model_config.enable_flash_mla:
            tokens_per_block = 64
            logger.info(
                f"Change tokens_per_block to: {tokens_per_block} for using FlashMLA"
            )

        sm_version = get_sm_version()
        if kv_cache_config.enable_block_reuse and sm_version not in [
                90, 100, 103, 120
        ]:
            logger.warning(
                f"KV cache reuse for MLA can only be enabled on SM90/SM100/SM103/SM120, "
                f"disable enable_block_reuse for SM{sm_version}")
            kv_cache_config.enable_block_reuse = False

        kv_cache_quant_algo = model_engine.model.model_config.quant_config.kv_cache_quant_algo
        if kv_cache_config.enable_block_reuse and not (
                kv_cache_quant_algo is None or kv_cache_quant_algo
                == QuantAlgo.NO_QUANT or kv_cache_quant_algo == QuantAlgo.FP8):
            logger.warning(
                f"KV cache reuse for MLA can only be enabled without KV cache quantization or with FP8 quantization, "
                f"disable enable_block_reuse for KV cache quant algorithm: {kv_cache_quant_algo}"
            )
            kv_cache_config.enable_block_reuse = False
        if enable_chunked_context and sm_version not in [90, 100, 103, 120]:
            logger.warning(
                "Chunked Prefill for MLA can only be enabled on SM90/SM100/SM103/SM120, "
                f"disable enable_chunked_context for SM{sm_version}")
            enable_chunked_context = False
            model_engine.attn_runtime_features.chunked_prefill = False
            if draft_model_engine is not None:
                draft_model_engine.attn_runtime_features.chunked_prefill = False

    if enable_chunked_context:
        chunk_unit_size = tokens_per_block
        max_attention_window = kv_cache_config.max_attention_window
        if max_attention_window and model_engine_max_seq_len > min(
                max_attention_window):
            # maxKvStepSizeInFmha = 256
            chunk_unit_size = max(256, chunk_unit_size)
            logger.info(
                f"ChunkUnitSize is set to {chunk_unit_size} as sliding window attention is used."
            )
        chunking_policy = (scheduler_config.context_chunking_policy if
                           scheduler_config.context_chunking_policy is not None
                           else ContextChunkingPolicy.FIRST_COME_FIRST_SERVED)
        assert chunk_unit_size is not None, "chunk_unit_size must be set"
        ctx_chunk_config: Tuple[StrEnum,
                                int] = (chunking_policy, chunk_unit_size)
    else:
        ctx_chunk_config = None

    guided_decoder: Optional[GuidedDecoder] = None
    if guided_decoding_config is not None:
        with allocation_scope(ExecutorMemoryType.GUIDED_DECODER,
                              RestoreMode.PINNED):
            if mapping.is_last_pp_rank():
                kwargs = {
                    "guided_decoding_config": guided_decoding_config,
                    "max_num_sequences": max_batch_size,
                    "vocab_size_padded": model_engine.model.vocab_size_padded,
                    "rank": mapping.rank,
                }
                if spec_config is not None:
                    kwargs[
                        "max_num_draft_tokens"] = spec_config.max_total_draft_tokens

                if spec_config is None or spec_config.spec_dec_mode.support_guided_decoder(
                ):
                    # GuidedDecoder is applicable to non-speculative decoding and two-model speculative decoding.
                    guided_decoder = GuidedDecoder(**kwargs)
                elif spec_config.spec_dec_mode.support_capturable_guided_decoder(
                ):
                    # CapturableGuidedDecoder is applicable to one-model speculative decoding.
                    success = model_engine.set_guided_decoder(
                        CapturableGuidedDecoder(**kwargs))
                    if not success:
                        raise ValueError(
                            f"Failed to set guided decoder for speculative decoding mode: {spec_config.spec_dec_mode.name}."
                        )
                else:
                    raise ValueError(
                        f"Guided decoding is not supported for speculative decoding mode: {spec_config.spec_dec_mode.name}."
                    )

    with allocation_scope(ExecutorMemoryType.SAMPLER, RestoreMode.PINNED):
        sampler = instantiate_sampler(
            model_engine,
            llm_args,
            mapping,
            max_batch_size=max_batch_size,
            max_beam_width=max_beam_width,
            max_seq_len=max_seq_len,
            mm_encoder_only=mm_encoder_only,
            speculative_config=spec_config,
            decoding_config=decoding_config,
            kv_cache_config=kv_cache_config,
            disable_flashinfer_sampling=llm_args.disable_flashinfer_sampling,
        )
        logger.info(f"Using Sampler: {type(sampler).__name__}")

    if kv_connector_config is not None:
        logger.info(
            f"Initializing kv connector with config: {kv_connector_config}")

        if scheduler_config.capacity_scheduler_policy != CapacitySchedulerPolicy.GUARANTEED_NO_EVICT:
            raise NotImplementedError(
                "KV connector is only supported with guaranteed no evict scheduler policy."
            )
        try:
            module = importlib.import_module(
                kv_connector_config.connector_module)
            worker_cls = getattr(module,
                                 kv_connector_config.connector_worker_class)
            scheduler_cls = getattr(
                module, kv_connector_config.connector_scheduler_class)

            rank = tensorrt_llm.mpi_rank()
            # Some connector API implementations may need to establish out-of-band communication between the scheduler and workers.
            # In this case, the worker may be dependent on the scheduler, or vice-versa.
            # To deal with cases like this, we instantiate them both concurrently.
            with ThreadPoolExecutor(max_workers=2) as executor:
                connector_worker_task = executor.submit(worker_cls, llm_args)

                if scheduler_cls is not None and rank == 0:
                    connector_scheduler_task = executor.submit(
                        scheduler_cls, llm_args)
                    connector_scheduler = connector_scheduler_task.result()
                else:
                    connector_scheduler = None

                connector_worker = connector_worker_task.result()

            forward_pass_callable = connector_worker.register_forward_pass_callable(
            )
            if forward_pass_callable:
                model_engine.register_forward_pass_callable(
                    forward_pass_callable)

            kv_connector_manager = KvCacheConnectorManager(
                connector_worker, connector_scheduler)

        except Exception as e:
            logger.error(f"Error instantiating connector: {e}")
            raise e
    else:
        kv_connector_manager = None

    resources = {}
    estimating_kv_cache = False
    kv_cache_creator = None

    # Create the execution stream for model forward operations
    # for proper synchronization with KVCacheTransferManager's onboard/offload operations.
    execution_stream = torch.cuda.Stream()
    logger.info(
        f"[create_py_executor] Created execution_stream: {execution_stream}")

    if model_engine.model.model_config.is_generation:
        #NOTE: non-generation models do not have kv cache
        kv_cache_creator = KvCacheCreator(
            model_engine=model_engine,
            draft_model_engine=draft_model_engine,
            mapping=mapping,
            net_max_seq_len=net_max_seq_len,
            kv_connector_manager=kv_connector_manager,
            max_num_tokens=max_num_tokens,
            max_beam_width=max_beam_width,
            tokens_per_block=tokens_per_block,
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
            kv_cache_config=kv_cache_config,
            llm_args=llm_args,
            speculative_config=spec_config,
            profiling_stage_data=profiling_stage_data,
            sparse_attention_config=sparse_attention_config,
            execution_stream=execution_stream,
        )
        estimating_kv_cache = kv_cache_creator.try_prepare_estimation()
        with allocation_scope(
                ExecutorMemoryType.INIT_KV_CACHE if estimating_kv_cache else
                ExecutorMemoryType.KV_CACHE, RestoreMode.NONE):
            kv_cache_creator.build_managers(resources, estimating_kv_cache)
            # Originally, max_seq_len might be mutated inside build_managers as field of executor config.
            # Since now, we are changing kv_cache_creator._max_seq_len instead. Restore max_seq_len here.
            max_seq_len = kv_cache_creator._max_seq_len
            update_sampler_max_seq_len(max_seq_len, sampler)

    # Resource managers for speculative decoding
    # For user-specified drafters, use extra_resource_managers in PyTorchBackend config
    # to provide a resource manager if required.

    with allocation_scope(ExecutorMemoryType.SPEC_RESOURCES,
                          RestoreMode.PINNED):
        spec_resource_manager = get_spec_resource_manager(
            model_engine, draft_model_engine)
    if spec_resource_manager is not None:
        resources[
            ResourceManagerType.SPEC_RESOURCE_MANAGER] = spec_resource_manager

    # Drafter for speculative decoding
    with allocation_scope(ExecutorMemoryType.DRAFTER, RestoreMode.PINNED):
        drafter = get_spec_drafter(model_engine,
                                   draft_model_engine,
                                   sampler,
                                   spec_resource_manager=spec_resource_manager,
                                   guided_decoder=guided_decoder)

    with allocation_scope(
            ExecutorMemoryType.INIT_EXTRA_RESOURCES if estimating_kv_cache else
            ExecutorMemoryType.EXTRA_RESOURCES, RestoreMode.PINNED):
        py_executor = create_py_executor_instance(
            dist=dist,
            resources=resources,
            mapping=mapping,
            llm_args=llm_args,
            ctx_chunk_config=ctx_chunk_config,
            model_engine=model_engine,
            start_worker=False,
            sampler=sampler,
            drafter=drafter,
            guided_decoder=guided_decoder,
            lora_config=lora_config,
            garbage_collection_gen0_threshold=garbage_collection_gen0_threshold,
            kv_connector_manager=kv_connector_manager
            if not estimating_kv_cache else None,
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
            max_beam_width=max_beam_width,
            max_num_tokens=max_num_tokens,
            peft_cache_config=peft_cache_config,
            scheduler_config=scheduler_config,
            cache_transceiver_config=cache_transceiver_config,
            virtual_memory_pools=vm_pools if not estimating_kv_cache else None,
            execution_stream=execution_stream,
        )
        # Originally, peft_cache_config might be mutated inside
        # create_py_executor_instance. Restore it here.
        peft_cache_config = py_executor.peft_cache_config

    if estimating_kv_cache:
        assert kv_cache_creator is not None
        with allocation_scope(ExecutorMemoryType.MODEL_EXTRA,
                              RestoreMode.PINNED):
            kv_cache_creator.configure_kv_cache_capacity(py_executor)
        kv_cache_creator.teardown_managers(resources)
        del py_executor  # free before constructing new

        with allocation_scope(ExecutorMemoryType.KV_CACHE, RestoreMode.NONE):
            # Before estimating KV cache size, a minimal KV cache has been allocated using
            # create_kv_cache_manager above, which caps kv_cache_creator.max_seq_len. Restoring
            # the original value before creating the final KV cache.
            kv_cache_creator._max_seq_len = model_engine_max_seq_len
            kv_cache_creator.build_managers(resources, False)
            # Originally, max_seq_len might be mutated inside build_managers as field of executor config.
            # Since now, we are changing kv_cache_creator._max_seq_len instead. Restore max_seq_len here.
            max_seq_len = kv_cache_creator._max_seq_len
            update_sampler_max_seq_len(max_seq_len, sampler)

            for eng in [model_engine, draft_model_engine]:
                if eng is None:
                    continue
                if eng.attn_metadata is not None:
                    if llm_args.cuda_graph_config is not None:
                        eng._release_cuda_graphs()
                    eng.attn_metadata = None

        with allocation_scope(ExecutorMemoryType.EXTRA_RESOURCES,
                              RestoreMode.PINNED):

            # run gc.collect() to free memory of the previous py_executor, avoid cudaFree overlap with cuda graph capture
            gc.collect()
            py_executor = create_py_executor_instance(
                dist=dist,
                resources=resources,
                mapping=mapping,
                llm_args=llm_args,
                ctx_chunk_config=ctx_chunk_config,
                model_engine=model_engine,
                start_worker=False,
                sampler=sampler,
                drafter=drafter,
                guided_decoder=guided_decoder,
                lora_config=lora_config,
                garbage_collection_gen0_threshold=
                garbage_collection_gen0_threshold,
                kv_connector_manager=kv_connector_manager,
                max_seq_len=max_seq_len,
                max_batch_size=max_batch_size,
                max_beam_width=max_beam_width,
                max_num_tokens=max_num_tokens,
                peft_cache_config=peft_cache_config,
                scheduler_config=scheduler_config,
                cache_transceiver_config=cache_transceiver_config,
                virtual_memory_pools=vm_pools,
                execution_stream=execution_stream,
            )

    _adjust_torch_mem_fraction()

    if mapping.rank == 0:
        logger.info(f"LLM Args:\n{llm_args}")

    py_executor.start_worker()
    return py_executor
