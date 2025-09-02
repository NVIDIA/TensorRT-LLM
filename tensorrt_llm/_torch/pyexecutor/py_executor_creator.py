import copy
import enum
import importlib
import os
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from itertools import chain
from typing import Dict, Optional

import torch

import tensorrt_llm
from tensorrt_llm._torch.pyexecutor.resource_manager import ResourceManagerType
from tensorrt_llm._utils import get_sm_version
from tensorrt_llm.bindings.executor import (CapacitySchedulerPolicy,
                                            ContextChunkingPolicy)
from tensorrt_llm.bindings.executor import \
    GuidedDecodingConfig as _GuidedDecodingConfig
from tensorrt_llm.bindings.internal.batch_manager import ContextChunkingConfig
from tensorrt_llm.builder import BuildConfig
from tensorrt_llm.llmapi.llm_args import (EagleDecodingConfig,
                                          KvCacheConnectorConfig,
                                          MTPDecodingConfig, PybindMirror,
                                          TorchLlmArgs)
from tensorrt_llm.llmapi.tokenizer import (TokenizerBase,
                                           _llguidance_tokenizer_info,
                                           _xgrammar_tokenizer_info)
from tensorrt_llm.logger import logger
from tensorrt_llm.lora_helper import LoraConfig
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.quantization import QuantAlgo

from ..attention_backend.interface import AttentionRuntimeFeatures
from ..distributed import MPIDist
from ..speculative import (get_num_extra_kv_tokens, get_spec_drafter,
                           get_spec_resource_manager)
from ._util import (KvCacheCreator, _adjust_torch_mem_fraction,
                    create_py_executor_instance, instantiate_sampler, is_mla)
from .config import LoadFormat, PyTorchConfig, _construct_checkpoint_loader
from .config_utils import is_mla
from .guided_decoder import GuidedDecoder
from .kv_cache_connector import KvCacheConnectorManager
from .model_engine import PyTorchModelEngine
from .py_executor import PyExecutor
from .sampler import TorchSampler, TRTLLMSampler


class _ExecutorCreationStage(enum.Enum):
    SAMPLER = "Sampler"
    DRAFTER = "Drafter"
    GUIDED_DECODER = "Guided decoder"
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
            _ExecutorCreationStage.DRAFTER:
            "reduce max_seq_len and/or max_draft_len",
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


def validate_llm_args(llm_args, model_engine, sampler):
    # Validate the flags for features' combination
    def init_feature_status(llm_args) -> Dict[str, bool]:
        assert isinstance(
            llm_args, TorchLlmArgs
        ), "Expect TorchLlmArgs used for feature status validation."
        feature_list = [
            "overlap_scheduler",
            "cuda_graph",
            "attention_dp",
            "disaggregated_serving",
            "chunked_prefill",
            "mtp",
            "eagle3_one_model",
            "eagle3_two_model",
            "torch_sampler",
            "tllm_cpp_sampler",
            "kv_cache_reuse",
            "slide_window_attention",
            # "logits_post_processor",
            "guided_decoding",
        ]
        feature_status: Dict[str, bool] = dict.fromkeys(feature_list)
        feature_status[
            "overlap_scheduler"] = not llm_args.disable_overlap_scheduler
        feature_status["cuda_graph"] = llm_args.cuda_graph_config is not None
        feature_status["attention_dp"] = llm_args.enable_attention_dp
        feature_status[
            "disaggregated_serving"] = llm_args.cache_transceiver_config is not None
        feature_status["chunked_prefill"] = llm_args.enable_chunked_prefill
        feature_status["mtp"] = (
            isinstance(llm_args.speculative_config, MTPDecodingConfig)
            and llm_args.speculative_config.num_nextn_predict_layers > 0)
        feature_status["eagle3_one_model"] = (
            isinstance(llm_args.speculative_config, EagleDecodingConfig)
            and llm_args.speculative_config.eagle3_one_model)
        feature_status["eagle3_two_model"] = (
            isinstance(llm_args.speculative_config, EagleDecodingConfig)
            and not llm_args.speculative_config.eagle3_one_model)
        feature_status["torch_sampler"] = isinstance(sampler, TorchSampler)
        feature_status["tllm_cpp_sampler"] = isinstance(sampler, TRTLLMSampler)
        feature_status[
            "kv_cache_reuse"] = llm_args.kv_cache_config is not None and llm_args.kv_cache_config.enable_block_reuse
        feature_status["slide_window_attention"] = (
            hasattr(model_engine.model.model_config.pretrained_config,
                    "layer_types") and "sliding_attention"
            in model_engine.model.model_config.pretrained_config.layer_types)
        feature_status[
            "guided_decoding"] = llm_args.guided_decoding_backend is not None
        assert all(v is not None for v in feature_status.values()
                   ), "feature status has not been fully initialized."
        assert all(
            k in feature_list for k in
            feature_status.keys()), "unexpected feature type in feature_status."
        return feature_status

    feature_status: Dict[str, bool] = init_feature_status(llm_args)

    ERR_MSG_TMPL = "{feature1} and {feature2} enabled together is not supported yet."
    # Some combinations of features are unsupported; the implementation may,
    # however, implicitly cast arguments to make them appear compatible.
    # Enabling this flag suppresses that cast and forces an explicit error instead.
    force_error_on_implicit_cast = False
    if force_error_on_implicit_cast:
        if feature_status["overlap_scheduler"] and feature_status[
                "eagle3_two_model"]:
            # Overlap scheduler with eagle-3 two models is not supported.
            # Previously we will disable overlap scheduler with a warning
            # https://github.com/NVIDIA/TensorRT-LLM/blob/fe7dda834d6d49a9b654ba70a4f1a8a6aaf9c715/tensorrt_llm/_torch/pyexecutor/py_executor_creator.py#L174-L179
            # Prefer a explicitly error msg here:
            raise ValueError(
                ERR_MSG_TMPL.format(feature1="overlap_scheduler",
                                    feature2="eagle3_two_model"))
        if (feature_status["mtp"] or feature_status["eagle3_one_model"]
                or feature_status["eagle3_two_model"]):
            if feature_status["tllm_cpp_sampler"]:
                # Mtp with tllm_cpp_sampler is not supported.
                # Previously we will use torch_sampler implicitly
                # https://github.com/NVIDIA/TensorRT-LLM/blob/70e352a6f784f1cfd2affc074ac8d83e430abd9a/tensorrt_llm/_torch/pyexecutor/_util.py#L599
                # Prefer a explicitly error msg here:
                raise ValueError(
                    ERR_MSG_TMPL.format(feature1="speculative decoding",
                                        feature2="tllm_cpp_sampler"))
    if feature_status["mtp"] and feature_status["slide_window_attention"]:
        raise ValueError(
            ERR_MSG_TMPL.format(feature1="mtp",
                                feature2="slide_window_attention"))
    if feature_status["guided_decoding"]:
        if feature_status["mtp"]:
            raise ValueError(
                ERR_MSG_TMPL.format(feature1="mtp", feature2="guided_decoding"))
        if feature_status["eagle3_one_model"]:
            raise ValueError(
                ERR_MSG_TMPL.format(feature1="eagle3_one_model",
                                    feature2="guided_decoding"))


def create_py_executor(
    llm_args: TorchLlmArgs,
    checkpoint_dir: str = None,
    tokenizer: Optional[TokenizerBase] = None,
    lora_config: Optional[LoraConfig] = None,
    kv_connector_config: Optional[KvCacheConnectorConfig] = None,
) -> PyExecutor:

    garbage_collection_gen0_threshold = llm_args.garbage_collection_gen0_threshold
    pytorch_backend_config = llm_args.get_pytorch_backend_config()
    if pytorch_backend_config is None:
        pytorch_backend_config = PyTorchConfig()
    max_num_tokens = llm_args.max_num_tokens

    kv_cache_config = llm_args.kv_cache_config
    if kv_cache_config is not None:
        kv_cache_config = PybindMirror.maybe_to_pybind(llm_args.kv_cache_config)
    if os.getenv("FORCE_DETERMINISTIC", "0") == "1":
        # Disable KV cache reuse for deterministic mode
        kv_cache_config.enable_block_reuse = False
        kv_cache_config.enable_partial_reuse = False
    enable_chunked_context = llm_args.enable_chunked_prefill
    mm_encoder_only = llm_args.mm_encoder_only
    mapping = llm_args.parallel_config.to_mapping()
    cache_transceiver_config = llm_args.cache_transceiver_config
    if cache_transceiver_config is not None:
        cache_transceiver_config = PybindMirror.maybe_to_pybind(
            cache_transceiver_config)
    scheduler_config = PybindMirror.maybe_to_pybind(llm_args.scheduler_config)
    peft_cache_config = llm_args.peft_cache_config
    if peft_cache_config is not None:
        peft_cache_config = PybindMirror.maybe_to_pybind(peft_cache_config)
    max_beam_width = llm_args.max_beam_width
    max_batch_size = llm_args.max_batch_size
    spec_config = llm_args.speculative_config
    if spec_config is not None and spec_config.decoding_type == "AUTO":
        from tensorrt_llm._torch.speculative import suggest_spec_config
        spec_config = suggest_spec_config(max_batch_size)
    build_config = BuildConfig()
    tokens_per_block = build_config.plugin_config.tokens_per_block
    decoding_config = llm_args.decoding_config

    guided_decoding_config = None
    if llm_args.guided_decoding_backend == 'xgrammar':
        assert tokenizer is not None
        guided_decoding_config = _GuidedDecodingConfig(
            backend=_GuidedDecodingConfig.GuidedDecodingBackend.XGRAMMAR,
            **_xgrammar_tokenizer_info(tokenizer))
    elif llm_args.guided_decoding_backend == 'llguidance':
        assert tokenizer is not None
        guided_decoding_config = _GuidedDecodingConfig(
            backend=_GuidedDecodingConfig.GuidedDecodingBackend.LLGUIDANCE,
            **_llguidance_tokenizer_info(tokenizer))
    elif llm_args.guided_decoding_backend is not None:
        raise ValueError(
            f"Unsupported guided decoding backend {llm_args.guided_decoding_backend}"
        )

    assert llm_args.backend == "pytorch"
    checkpoint_format = llm_args.checkpoint_format
    checkpoint_loader = _construct_checkpoint_loader(llm_args.backend,
                                                     llm_args.checkpoint_loader,
                                                     checkpoint_format)

    max_seq_len = None
    if llm_args.max_seq_len is not None:
        max_seq_len = llm_args.max_seq_len

    if max_num_tokens is None:
        max_num_tokens = 8192
    if pytorch_backend_config.attn_backend in [
            "FLASHINFER", "FLASHINFER_STAR_ATTENTION"
    ]:
        # Workaround for flashinfer and star attention
        if kv_cache_config.enable_block_reuse:
            logger.warning(
                f"Disabling block reuse for {pytorch_backend_config.attn_backend} backend"
            )
            kv_cache_config.enable_block_reuse = False

    if pytorch_backend_config.attn_backend == "FLASHINFER_STAR_ATTENTION" and enable_chunked_context:
        logger.warning(
            f"Disabling chunked context for {pytorch_backend_config.attn_backend} backend"
        )
        enable_chunked_context = False

    if not pytorch_backend_config.disable_overlap_scheduler and spec_config is not None:
        if not spec_config.spec_dec_mode.support_overlap_scheduler():
            logger.warning(
                f"Disable overlap scheduler for speculation mode {spec_config.spec_dec_mode.name}"
            )
            pytorch_backend_config.disable_overlap_scheduler = True

    if mm_encoder_only:
        pytorch_backend_config.mm_encoder_only = True
        pytorch_backend_config.load_format = LoadFormat.VISION_ONLY
        # Disable overlap scheduler for multimodal encoder-only mode
        logger.warning(
            "Disabling overlap scheduler for multimodal encoder-only mode. "
            "The overlap scheduler is designed for generation models and is not needed "
            "when only processing vision encoder inputs.")
        pytorch_backend_config.disable_overlap_scheduler = True

    if mapping is None:
        mapping = Mapping(world_size=tensorrt_llm.mpi_world_size(),
                          tp_size=tensorrt_llm.mpi_world_size(),
                          gpus_per_node=tensorrt_llm.default_gpus_per_node(),
                          rank=tensorrt_llm.mpi_rank())
    else:
        mapping = copy.deepcopy(mapping)
        mapping.rank = tensorrt_llm.mpi_rank()

    dist = MPIDist(mapping=mapping)

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
    with mem_monitor.observe_creation_stage(
            _ExecutorCreationStage.MODEL_ENGINE_MAIN):
        model_engine = PyTorchModelEngine(
            model_path=checkpoint_dir,
            pytorch_backend_config=pytorch_backend_config,
            batch_size=max_batch_size,
            max_beam_width=max_beam_width,
            max_num_tokens=max_num_tokens,
            max_seq_len=max_seq_len,
            mapping=mapping,
            attn_runtime_features=attn_runtime_features,
            dist=dist,
            spec_config=spec_config,
            lora_config=lora_config,
            checkpoint_loader=checkpoint_loader,
        )

    if has_draft_model_engine:
        with mem_monitor.observe_creation_stage(
                _ExecutorCreationStage.MODEL_ENGINE_DRAFT):
            draft_spec_config = copy.copy(spec_config)
            draft_pytorch_backend_config = copy.copy(pytorch_backend_config)
            if spec_config.load_format == "dummy":
                draft_pytorch_backend_config.load_format = LoadFormat.DUMMY
            # The draft model won't have any draft tokens attached to
            # generation requests when we invoke it autoregressively
            draft_spec_config.max_draft_len = 0

            draft_model_engine = PyTorchModelEngine(
                model_path=spec_config.speculative_model_dir,
                pytorch_backend_config=draft_pytorch_backend_config,
                batch_size=max_batch_size,
                max_beam_width=max_beam_width,
                max_num_tokens=max_num_tokens,
                # Note: The draft model engine will infer its own max_seq_len.
                # We'll stop drafting when we hit the max.
                max_seq_len=max_seq_len,
                mapping=mapping,
                attn_runtime_features=attn_runtime_features,
                dist=dist,
                spec_config=draft_spec_config,
                checkpoint_loader=checkpoint_loader,
                is_draft_model=True,
            )
            draft_model_engine.kv_cache_manager_key = ResourceManagerType.DRAFT_KV_CACHE_MANAGER
            draft_model_engine.load_weights_from_target_model(
                model_engine.model)
    else:
        draft_model_engine = None

    # PyTorchModelEngine modifies these fields, update them
    model_engine_max_seq_len = model_engine.max_seq_len
    net_max_seq_len = model_engine_max_seq_len
    if not pytorch_backend_config.disable_overlap_scheduler:
        model_engine_max_seq_len = model_engine.max_seq_len + 1
        if spec_config is not None:
            model_engine_max_seq_len += spec_config.max_draft_len

    if spec_config is not None:
        model_engine_max_seq_len += get_num_extra_kv_tokens(spec_config)
        model_engine_max_seq_len += spec_config.max_draft_len

    max_seq_len = model_engine_max_seq_len

    max_num_tokens = model_engine.max_num_tokens

    config = model_engine.model.model_config.pretrained_config
    if is_mla(config):
        if model_engine.model.model_config.enable_flash_mla:
            tokens_per_block = 64
            logger.info(
                f"Change tokens_per_block to: {tokens_per_block} for using FlashMLA"
            )

        sm_version = get_sm_version()
        if kv_cache_config.enable_block_reuse and sm_version not in [
                90, 100, 120
        ]:
            logger.warning(
                f"KV cache reuse for MLA can only be enabled on SM90/SM100/SM120, "
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
        if enable_chunked_context and sm_version not in [90, 100]:
            logger.warning(
                "Chunked Prefill for MLA can only be enabled on SM90/SM100, "
                f"disable enable_chunked_context for SM{sm_version}")
            enable_chunked_context = False
            model_engine.attn_runtime_features.chunked_prefill = False
            if draft_model_engine is not None:
                draft_model_engine.attn_runtime_features.chunked_prefill = False

    if enable_chunked_context:
        chunk_unit_size = tokens_per_block
        max_attention_window = kv_cache_config.max_attention_window
        if max_attention_window and max_seq_len > min(max_attention_window):
            # maxKvStepSizeInFmha = 256
            chunk_unit_size = max(256, chunk_unit_size)
            logger.info(
                f"ChunkUnitSize is set to {chunk_unit_size} as sliding window attention is used."
            )
        chunking_policy = (scheduler_config.context_chunking_policy if
                           scheduler_config.context_chunking_policy is not None
                           else ContextChunkingPolicy.FIRST_COME_FIRST_SERVED)
        assert chunk_unit_size is not None, "chunk_unit_size must be set"
        ctx_chunk_config = ContextChunkingConfig(chunking_policy,
                                                 chunk_unit_size)
    else:
        ctx_chunk_config = None

    with mem_monitor.observe_creation_stage(
            _ExecutorCreationStage.GUIDED_DECODER):
        guided_decoder: Optional[GuidedDecoder] = None
        if guided_decoding_config is not None:
            if spec_config is not None and not has_spec_drafter:
                raise ValueError(
                    "Guided decoding is only supported with speculative decoding that has a dedicated drafter (two-model engine)."
                )
            if mapping.is_last_pp_rank():
                max_num_draft_tokens = 0
                if spec_config is not None:
                    max_num_draft_tokens = spec_config.max_draft_len
                guided_decoder = GuidedDecoder(
                    guided_decoding_config,
                    max_batch_size,
                    model_engine.model.vocab_size_padded,
                    max_num_draft_tokens=max_num_draft_tokens)

    with mem_monitor.observe_creation_stage(_ExecutorCreationStage.SAMPLER):
        sampler = instantiate_sampler(model_engine, pytorch_backend_config,
                                      mapping, mm_encoder_only, max_batch_size,
                                      decoding_config, max_beam_width,
                                      kv_cache_config, max_seq_len, spec_config)
        logger.info(f"Using Sampler: {type(sampler).__name__}")

    validate_llm_args(llm_args, model_engine, sampler)

    if kv_connector_config is not None:
        logger.info(
            f"Initializing kv connector with config: {kv_connector_config}")

        if pytorch_backend_config.use_cuda_graph:
            raise NotImplementedError(
                "CUDA graphs are not supported with KV connector hooks.")

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
                connector_worker_task = executor.submit(worker_cls)

                if scheduler_cls is not None and rank == 0:
                    connector_scheduler_task = executor.submit(
                        scheduler_cls, tokens_per_block)
                    connector_scheduler = connector_scheduler_task.result()
                else:
                    connector_scheduler = None

                connector_worker = connector_worker_task.result()

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
    if model_engine.model.model_config.is_generation:
        #NOTE: non-generation models do not have kv cache
        kv_cache_creator = KvCacheCreator(
            model_engine=model_engine,
            draft_model_engine=draft_model_engine,
            mapping=mapping,
            net_max_seq_len=net_max_seq_len,
            kv_connector_manager=kv_connector_manager,
            kv_cache_config=kv_cache_config,
            max_num_tokens=max_num_tokens,
            max_beam_width=max_beam_width,
            pytorch_backend_config=pytorch_backend_config,
            speculative_config=spec_config,
            tokens_per_block=tokens_per_block,
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size)
        estimating_kv_cache = kv_cache_creator.try_prepare_estimation()

        with mem_monitor.observe_creation_stage(
                _ExecutorCreationStage.INIT_KV_CACHE
                if estimating_kv_cache else _ExecutorCreationStage.KV_CACHE):
            kv_cache_creator.build_managers(resources, estimating_kv_cache)
            # restore max_seq_len which might be changed in build_managers
            max_seq_len = kv_cache_creator.max_seq_len

    # Resource managers for speculative decoding
    # For user-specified drafters, use extra_resource_managers in PyTorchBackend config
    # to provide a resource manager if required.
    spec_resource_manager = get_spec_resource_manager(model_engine,
                                                      draft_model_engine)
    if spec_resource_manager is not None:
        resources[
            ResourceManagerType.SPEC_RESOURCE_MANAGER] = spec_resource_manager

    # Drafter for speculative decoding
    with mem_monitor.observe_creation_stage(_ExecutorCreationStage.DRAFTER):
        drafter = get_spec_drafter(model_engine,
                                   draft_model_engine,
                                   sampler,
                                   spec_resource_manager=spec_resource_manager,
                                   guided_decoder=guided_decoder)

    with mem_monitor.observe_creation_stage(
            _ExecutorCreationStage.INIT_EXTRA_RESOURCES
            if estimating_kv_cache else _ExecutorCreationStage.EXTRA_RESOURCES):
        py_executor = create_py_executor_instance(
            dist=dist,
            resources=resources,
            mapping=mapping,
            pytorch_backend_config=pytorch_backend_config,
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
            max_num_tokens=max_num_tokens,
            max_beam_width=max_beam_width,
            peft_cache_config=peft_cache_config,
            scheduler_config=scheduler_config,
            cache_transceiver_config=cache_transceiver_config,
        )

    if estimating_kv_cache:
        assert kv_cache_creator is not None
        with mem_monitor.observe_creation_stage(
                _ExecutorCreationStage.MODEL_EXTRA):
            kv_cache_creator.configure_kv_cache_capacity(py_executor)
        kv_cache_creator.teardown_managers(resources)
        del py_executor  # free before constructing new

        with mem_monitor.observe_creation_stage(
                _ExecutorCreationStage.KV_CACHE):
            # Before estimating KV cache size, a minimal KV cache has been allocated using
            # create_kv_cache_manager above, which caps executor_config.max_seq_len. Restoring
            # the original value before creating the final KV cache.

            # update kv_cache_creator.max_seq_len which might be used inside build_managers
            kv_cache_creator.max_seq_len = model_engine_max_seq_len
            kv_cache_creator.build_managers(resources, False)
            # restore executor_config.max_seq_len which might be changed in build_managers
            max_seq_len = kv_cache_creator.max_seq_len

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
                dist=dist,
                resources=resources,
                mapping=mapping,
                pytorch_backend_config=pytorch_backend_config,
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
                max_num_tokens=max_num_tokens,
                max_beam_width=max_beam_width,
                peft_cache_config=peft_cache_config,
                scheduler_config=scheduler_config,
                cache_transceiver_config=cache_transceiver_config,
            )

    _adjust_torch_mem_fraction(pytorch_backend_config)

    py_executor.start_worker()
    return py_executor
