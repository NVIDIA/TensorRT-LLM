__all__ = [
    'LlmArgs',
    'LlmBuildStats',
    'ModelLoader',
    '_ModelRuntimeContext',
    '_ModelInfo',
    '_ParallelConfig',
    '_ModelFormatKind',
    '_ModelWrapper',
    'BatchingType',
    'ExecutorConfig',
    'SchedulerConfig',
    'KvCacheConfig',
    'KvCacheRetentionConfig',
    'LookaheadDecodingConfig',
    'MedusaDecodingConfig',
    'ContextChunkingPolicy',
    'CapacitySchedulerPolicy',
    'BuildConfig',
    'BuildCacheConfig',
    'QuantConfig',
    'CalibConfig',
    'CachedModelLoader',
    'ConfigArbitrateError',
    '_ConfigArbitrator',
]

import copy
import json
import math
import os
import shutil
import tempfile
import time
import weakref
from argparse import Namespace
from dataclasses import asdict, dataclass, field, fields
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

import torch
from tqdm import tqdm
from transformers import PreTrainedTokenizerBase

from .._utils import mpi_barrier, mpi_broadcast, mpi_rank, release_gc
from ..auto_parallel import AutoParallelConfig, infer_cluster_config
# yapf: disable
from ..bindings.executor import (BatchingType, CapacitySchedulerPolicy,
                                 ContextChunkingPolicy, DecodingConfig,
                                 DecodingMode, EagleConfig, ExecutorConfig,
                                 ExtendedRuntimePerfKnobConfig, KvCacheConfig,
                                 KvCacheRetentionConfig,
                                 LookaheadDecodingConfig, PeftCacheConfig,
                                 SchedulerConfig)
# yapf: enable
from ..builder import BuildConfig, Engine, EngineConfig, build
from ..logger import logger
from ..mapping import Mapping
from ..models.automodel import MODEL_MAP, AutoConfig, AutoModelForCausalLM
from ..models.modeling_utils import (PretrainedConfig, QuantAlgo, QuantConfig,
                                     SpeculativeDecodingMode)
from ..module import Module
from .build_cache import (BuildCache, BuildCacheConfig, CachedStage,
                          get_build_cache_config_from_env)
from .mpi_session import MPINodeState, MpiSession
from .tokenizer import (TokenizerBase, TransformersTokenizer, load_hf_tokenizer,
                        tokenizer_factory)
# TODO[chunweiy]: move the following symbols back to utils scope, and remove the following import
from .utils import (append_docstring, download_hf_model,
                    download_hf_pretrained_config, enable_llm_debug,
                    get_directory_size_in_gb, print_colored,
                    print_traceback_on_error)


@dataclass
class _ParallelConfig:
    ''' The model distribution configs for LLM.  '''
    tp_size: int = 1
    pp_size: int = 1
    cp_size: int = 1
    gpus_per_node: int = 8
    moe_tp_size: int = 1
    moe_ep_size: int = 1
    cp_config: dict = field(default_factory=dict)
    enable_attention_dp: bool = False
    auto_parallel: bool = False

    _world_size: int = field(default=1, init=False)
    _devices: Optional[List[int]] = field(default=None, init=False)

    @property
    def devices(self) -> List[int]:
        if self._devices is None:
            return list(range(self.world_size))
        return self._devices

    @devices.setter
    def devices(self, devices: List[int]):
        if len(devices) != self.world_size:
            raise ValueError(
                f"devices {devices} should have the same length as world_size {self.world_size}"
            )
        self._devices = devices

    @property
    def world_size(self) -> bool:

        if self.auto_parallel:
            if self.tp_size > 1 or self.pp_size > 1 or self.cp_size > 1:
                raise RuntimeError(
                    "manually TP and PP are not supported in auto parallel mode."
                )
            return self._world_size

        if self._world_size > 1:
            raise RuntimeError(
                "world_size > 1 is only supported in auto parallel mode.")
        return self.tp_size * self.pp_size * self.cp_size

    @property
    def world_size_per_node(self) -> int:
        world_size = self.world_size
        total_nodes = math.ceil(world_size / self.gpus_per_node)
        return world_size // total_nodes  #TODO is this right?

    @world_size.setter
    def world_size(self, world_size: int):
        if self.auto_parallel:
            self._world_size = world_size
        elif (not self.auto_parallel
              ) and world_size != self.tp_size * self.pp_size * self.cp_size:
            raise ValueError(
                f"world_size {world_size} should be equal to tp_size * pp_size {self.tp_size * self.pp_size * self.cp_size} "
            )

    @property
    def is_multi_gpu(self) -> bool:
        return self.world_size > 1

    def to_mapping(self) -> Mapping:
        return Mapping(world_size=self.world_size,
                       rank=mpi_rank(),
                       gpus_per_node=self.gpus_per_node,
                       tp_size=self.tp_size,
                       pp_size=self.pp_size,
                       cp_size=self.cp_size,
                       cp_config=self.cp_config,
                       enable_attention_dp=self.enable_attention_dp,
                       moe_tp_size=self.moe_tp_size,
                       moe_ep_size=self.moe_ep_size,
                       auto_parallel=self.auto_parallel)


@dataclass(slots=True)
class CalibConfig:
    """
    Calibration configuration.

    Args:
        device (Literal['cuda', 'cpu'], default='cuda'): The device to run calibration.
        calib_dataset (str, default='cnn_dailymail'): The name or local path of calibration dataset.
        calib_batches (int, default=512): The number of batches that the calibration runs.
        calib_batch_size (int, default=1): The batch size that the calibration runs.
        calib_max_seq_length (int, default=512): The maximum sequence length that the calibration runs.
        random_seed (int, default=1234): The random seed used for calibration.
        tokenizer_max_seq_length (int, default=2048): The maximum sequence length to initialize tokenizer for calibration.
    """
    device: Literal['cuda', 'cpu'] = 'cuda'
    calib_dataset: str = 'cnn_dailymail'
    calib_batches: int = 512
    calib_batch_size: int = 1
    calib_max_seq_length: int = 512
    random_seed: int = 1234
    tokenizer_max_seq_length: int = 2048

    @classmethod
    def from_dict(cls, config: dict):
        return cls(**config)

    def to_dict(self):
        return asdict(self)


class _ModelFormatKind(Enum):
    HF = 0
    TLLM_CKPT = 1
    TLLM_ENGINE = 2


@dataclass
class _ModelInfo:
    dtype: Optional[str] = None
    architecture: Optional[str] = None

    @property
    def model_name(self) -> str:
        if self.architecture is None:
            raise RuntimeError("The architecture is not set yet.")

        return self.architecture

    @classmethod
    def from_pretrained_config(cls, config: PretrainedConfig):
        return cls(dtype=config.dtype, architecture=config.architecture)

    @classmethod
    def from_builder_config_json(cls, config: dict):
        if 'version' in config:
            # The Dict format is { 'builder_config':..., 'plugin_config':...}
            dtype = config['plugin_config']['gpt_attention_plugin']
        else:
            dtype = config['pretrained_config']['dtype']

        return cls(dtype=dtype, architecture=config['builder_config']['name'])

    @classmethod
    def from_module(cls, module: Module):
        raise NotImplementedError()


@dataclass
class MedusaDecodingConfig:
    medusa_choices: Optional[List[List[int]]] = None
    num_medusa_heads: Optional[int] = None


@dataclass
class EagleDecodingConfig:
    eagle_choices: Optional[List[List[int]]] = None
    greedy_sampling: Optional[bool] = True
    posterior_threshold: Optional[float] = None
    use_dynamic_tree: Optional[bool] = False
    dynamic_tree_max_topK: Optional[int] = None
    num_eagle_layers: Optional[int] = None
    max_non_leaves_per_layer: Optional[int] = None


@dataclass
class _ModelWrapper:
    model: Union[str, Path]

    def __post_init__(self):
        if not self.model:
            raise ValueError("model should be provided.")
        assert isinstance(self.model,
                          (str, Path)), f"Invalid model: {self.model}"

        model_dir = Path(self.model)

        if model_dir.exists() and model_dir.is_dir():
            self.model = model_dir

    @property
    def is_hub_model(self) -> bool:
        return not self.is_local_model

    @property
    def is_local_model(self) -> bool:
        return isinstance(self.model, Path)

    @property
    def model_dir(self) -> Path:
        assert self.is_local_model, f"model_dir is only available for local model, {self.model}."
        return self.model

    @model_dir.setter
    def model_dir(self, model_dir: Union[str, Path]):
        model_dir = Path(model_dir)
        assert model_dir.exists() and model_dir.is_dir(
        ), f"model_dir is not a valid path, {model_dir}"
        self.model = model_dir

    @property
    def model_name(self) -> Union[str, Path]:
        return self.model if isinstance(self.model, str) else None


# The docstring for LlmArgs and LLM; will be appended to the two classes' apidocs.
LLMARGS_DOCSTRING = r"""
        model (str or Path): The model name or a local model directory.
            Note that if the value could be both a model name or a local model directory,
            the local model directory will be prioritized.

        tokenizer (str, Path, TokenizerBase, PreTrainedTokenizerBase, optional):
            The name or path of a HuggingFace Transformers tokenizer, or the loaded tokenizer.
            Defaults to None.

        tokenizer_mode (Literal['auto', 'slow']): The tokenizer mode.
            'auto' will use the fast tokenizer if available, and 'slow' will always use the slow tokenizer.
            The fast tokenizer is based on Huggingface's Rust library tokenizers, which achieves a significant speed-up compared to its slow counterpart.
            Defaults to 'auto'.

        skip_tokenizer_init (bool):
            If true, skip initialization of tokenizer and detokenizer.
            LLM.generate and LLM.generate_async will accept prompt token ids as input only.
            Defaults to False.

        trust_remote_code (bool): Whether to trust remote code when downloading model and tokenizer from Hugging Face. Defaults to False.

        tensor_parallel_size(int): The number of processes for tensor parallelism. Defaults to 1.

        dtype (str): The data type for the model weights and activations.
            Can be "float16", "bfloat16", "float32", or "auto". If "auto", the data type
            will be automatically inferred from the source model. If the source data type
            is "float32", it will be converted to "float16". Defaults to "auto".

        revision (str, optional): The revision of the model to use. Defaults to None.

        tokenizer_revision (str, optional): The revision of the tokenizer to use. Defaults to None.

        pipeline_parallel_size (int): The pipeline parallel size. Defaults to 1.

        context_parallel_size (int): The context parallel size. Defaults to 1.

        gpus_per_node (Optional[int]): The number of GPUs per node. Defaults to None for automatic configure.

        load_format (Literal['auto', 'dummy']): The format of the model weights to load.
            * 'auto' will try to load the weights from the provided checkpoint.
            * 'dummy' will initialize the weights with random values, which is mainly for profiling.
            Defaults to 'auto'.

        enable_tqdm (bool): Whether to display a progress bar during model building. Defaults to False.

        enable_lora (bool): Enable LoRA adapters. Defaults to False.

        max_lora_rank (int, optional): Maximum LoRA rank. If specified, it overrides `build_config.lora_config.max_lora_rank`. Defaults to None.

        max_loras (int): Maximum number of LoRA adapters to be stored in GPU memory. Defaults to 4.

        max_cpu_loras (int): Maximum number of LoRA adapters to be stored in CPU memory. Defaults to 4.

        enable_prompt_adapter (bool): Enable prompt adapters. Defaults to False.

        max_prompt_adapter_token (int): Maximum number of prompt adapter tokens. Defaults to 0.

        quant_config (QuantConfig, optional): The quantization configuration for the model. Defaults to None.

        calib_config (CalibConfig, optional): The calibration configuration for the model. Defaults to None.

        build_config (BuildConfig, optional)): The build configuration for the model. Defaults to None.

        kv_cache_config (KvCacheConfig, optional): The key-value cache configuration for the model. Defaults to None.

        enable_chunked_prefill (bool): Whether to enable chunked prefill. Defaults to False.

        decoding_config (DecodingConfig, optional): The decoding configuration for the model. Defaults to None.

        guided_decoding_backend (str, optional): The guided decoding backend, currently supports 'xgrammar'. Defaults to None.

        logits_post_processor_map (Dict[str, Callable], optional): A map of logit post-processing functions. Defaults to None.

        iter_stats_max_iterations (int, optional): The maximum number of iterations for iteration statistics. Defaults to None.

        request_stats_max_iterations (int, optional): The maximum number of iterations for request statistics. Defaults to None.

        workspace(str, optional): The directory to store intermediate files. Defaults to None.

        embedding_parallel_mode (str): The parallel mode for embeddings. Defaults to 'SHARDING_ALONG_VOCAB'.

        auto_parallel (bool): Enable auto parallel mode. Defaults to False.

        auto_parallel_world_size (int): The MPI world size for auto parallel. Defaults to 1.

        moe_tensor_parallel_size (int, optional): The tensor parallel size for MoE models's expert weights.

        moe_expert_parallel_size (int, optional): The expert parallel size for MoE models's expert weights.

        enable_attention_dp (bool, optional): Enable attention data parallel. Defaults to False.

        fast_build: (bool): Enable features for faster engine building.
            This may cause some performance degradation and is currently incompatible with int8/int4 quantization.
            Defaults to False.

        enable_build_cache (bool, BuildCacheConfig, optional): Whether to enable build caching for the model. Defaults to None.

        peft_cache_config (PeftCacheConfig, optional): The PEFT cache configuration for the model. Defaults to None.

        scheduler_config (SchedulerConfig, optional): The scheduler configuration for the model. Defaults to None.

        speculative_config (LookaheadDecodingConfig or other speculative configurations, optional): The speculative decoding configuration. Defaults to None.

        batching_type (BatchingType, optional): The batching type for the model. Defaults to None.

        normalize_log_probs (bool): Whether to normalize log probabilities for the model. Defaults to False.

        gather_generation_logits (bool): Enable gathering generation logits.

        max_batch_size (int, optional): The maximum batch size for runtime. Defaults to None.

        max_num_tokens (int, optional): The maximum number of tokens for runtime. Defaults to None.

        extended_runtime_perf_knob_config (ExtendedRuntimePerfKnobConfig, optional): The extended runtime performance knob configuration for the model. Defaults to None.

"""


@append_docstring(LLMARGS_DOCSTRING)
@dataclass
class LlmArgs:
    """The arguments for constructing a LLM instance.

    Parameters:
    """
    # Explicit arguments
    model: Union[str, Path]

    speculative_model: Optional[Union[str, Path]] = None

    tokenizer: Optional[Union[str, Path, TokenizerBase,
                              PreTrainedTokenizerBase]] = None

    tokenizer_mode: Literal['auto', 'slow'] = 'auto'

    skip_tokenizer_init: bool = False

    trust_remote_code: bool = False

    tensor_parallel_size: int = 1

    dtype: str = "auto"

    revision: Optional[str] = None

    tokenizer_revision: Optional[str] = None

    # Below are all remaining arguments
    pipeline_parallel_size: int = 1

    context_parallel_size: int = 1

    gpus_per_node: Optional[int] = None

    moe_tensor_parallel_size: Optional[int] = None

    moe_expert_parallel_size: Optional[int] = None

    enable_attention_dp: bool = False

    cp_config: Optional[dict] = field(default_factory=dict)

    auto_parallel: bool = False

    auto_parallel_world_size: int = 1

    load_format: Literal['auto', 'dummy'] = 'auto'

    enable_tqdm: bool = False

    # LoRA arguments
    enable_lora: bool = False

    max_lora_rank: Optional[int] = None

    max_loras: int = 4

    max_cpu_loras: int = 4

    # Prompt adapter arguments
    enable_prompt_adapter: bool = False

    max_prompt_adapter_token: int = 0

    # Quantization and calibration configurations
    quant_config: Optional[QuantConfig] = None

    calib_config: Optional[CalibConfig] = None

    # BuildConfig is introduced to give users a familiar interface to configure the model building.
    build_config: Optional[BuildConfig] = None

    # Several options from ExecutorConfig, expanded here for less hierarchy
    kv_cache_config: Optional[KvCacheConfig] = None

    enable_chunked_prefill: bool = False

    # TODO[enweiz]: this might affect medusa, and could be removed in the future for API consistency
    decoding_config: Optional[DecodingConfig] = None

    guided_decoding_backend: Optional[str] = None

    logits_post_processor_map: Optional[Dict[str, Callable]] = None

    iter_stats_max_iterations: Optional[int] = None

    request_stats_max_iterations: Optional[int] = None

    workspace: Optional[str] = None

    # A handful of options from PretrainedConfig
    embedding_parallel_mode: str = 'SHARDING_ALONG_VOCAB'

    fast_build: bool = False

    # Once set, the model will reuse the build_cache
    enable_build_cache: Union[BuildCacheConfig, bool] = False

    peft_cache_config: Optional[PeftCacheConfig] = None

    scheduler_config: Optional[SchedulerConfig] = None

    # Speculative decoding parameters
    speculative_config: Optional[Union[LookaheadDecodingConfig]] = None

    batching_type: Optional[BatchingType] = None

    normalize_log_probs: bool = False

    gather_generation_logits: bool = False

    extended_runtime_perf_knob_config: Optional[
        ExtendedRuntimePerfKnobConfig] = None

    # TODO: remove this option in the future
    use_runtime_defaults: bool = True

    max_batch_size: Optional[int] = None
    max_num_tokens: Optional[int] = None

    # backend to use
    backend: Optional[str] = None

    # private options
    _num_postprocess_workers: int = 0  # Number of postprocess worker processes
    _postprocess_tokenizer_dir: Optional[str] = None
    _postprocess_result_handler: Optional[Callable] = None

    def __post_init__(self):
        # TODO[chunweiy]: Enable this option in the future
        # Currently we want LLMAPI to be consistent with the lower APIs in the model building, thus disable this to avoid
        # magics.
        self.perform_config_arbitration = False

        if self.skip_tokenizer_init:
            self.tokenizer = None
        else:
            self.tokenizer = tokenizer_factory(
                self.tokenizer,
                trust_remote_code=self.trust_remote_code,
                use_fast=self.tokenizer_mode != 'slow')

        if torch.cuda.get_device_properties(0).major < 8:
            if self.dtype == 'auto':
                self.dtype = 'float16'
            if self.dtype == 'bfloat16':
                raise RuntimeError("Pre SM 80 GPUs do not support bfloat16")

        if self.gpus_per_node is None:
            logger.warning(
                f"Using default gpus_per_node: {torch.cuda.device_count()}")
            self.gpus_per_node = torch.cuda.device_count()
        assert self.gpus_per_node is not None

        if self.moe_tensor_parallel_size is None:
            self.moe_tensor_parallel_size = -1

        if self.moe_expert_parallel_size is None:
            self.moe_expert_parallel_size = -1

        self.parallel_config = _ParallelConfig(
            tp_size=self.tensor_parallel_size,
            pp_size=self.pipeline_parallel_size,
            cp_size=self.context_parallel_size,
            gpus_per_node=self.gpus_per_node,
            moe_tp_size=self.moe_tensor_parallel_size,
            moe_ep_size=self.moe_expert_parallel_size,
            enable_attention_dp=self.enable_attention_dp,
            cp_config=self.cp_config,
            auto_parallel=self.auto_parallel)
        if self.parallel_config.auto_parallel:
            self.parallel_config.world_size = self.auto_parallel_world_size

        self.auto_parallel_config = AutoParallelConfig(
            sharded_io_allowlist=[
                "past_key_value_\\d+",
                "present_key_value_\\d*",
            ],
            same_buffer_io={
                "past_key_value_(\\d+)": "present_key_value_\\1",
            },
            **infer_cluster_config(),
        )

        self.kv_cache_config = self.kv_cache_config or KvCacheConfig()

        self.scheduler_config = self.scheduler_config or SchedulerConfig()

        # This is used to hold th options for convert_checkpoint
        self._convert_checkpoint_options = {}

    @classmethod
    def from_kwargs(cls, **kwargs) -> "LlmArgs":
        LlmArgs._check_executor_config_options_consistency()
        ret = cls(**kwargs)
        ret.setup()
        return ret

    def to_dict(self):
        return dict(
            (field.name, getattr(self, field.name)) for field in fields(self))

    @staticmethod
    def _check_executor_config_options_consistency():
        # max_beam_width is not included since vague behavior due to lacking the support for dynamic beam width during
        # generation
        black_list = set(["max_beam_width"])
        executor_config_attrs = set(attr for attr in dir(ExecutorConfig)
                                    if not attr.startswith('_')
                                    and callable(getattr(ExecutorConfig, attr)))
        executor_config_attrs -= black_list
        llm_args_attr = set([f.name for f in fields(LlmArgs)])
        # NOTE: When cpp ExecutorConfig add new options, please add the new options into `_LlmArgs` with docs as well
        # ASK chunweiy for help if you are not sure about the new options.
        assert executor_config_attrs.issubset(
            llm_args_attr
        ), f"New options found in underlying ExecutorConfig: {llm_args_attr - executor_config_attrs}"

    def setup(self):
        ''' This method will setup the configs right before building the model.
        It will check the consistency of the configs and arbitrate the conflicts.
        '''

        assert isinstance(self.model,
                          (str, Path)), f"Invalid model: {self.model}"

        self._setup_embedding_parallel_mode()

        if self.enable_build_cache:
            self.enable_build_cache = BuildCacheConfig() if isinstance(
                self.enable_build_cache, bool) else self.enable_build_cache
            if not isinstance(self.enable_build_cache, BuildCacheConfig):
                raise ValueError(
                    f"Invalid build_cache_config: {self.enable_build_cache}")
        model_obj = _ModelWrapper(self.model)
        speculative_model_obj = _ModelWrapper(
            self.speculative_model
        ) if self.speculative_model is not None else None
        if model_obj.is_local_model and getattr(self, 'backend',
                                                None) != 'pytorch':
            # Load parallel_config from the engine.
            self.model_format = ModelLoader.get_model_format(self.model)

            if self.model_format is _ModelFormatKind.TLLM_ENGINE:
                if self.build_config is not None:
                    logger.warning(
                        "The build_config is ignored for model format of TLLM_ENGINE."
                    )
                self._load_config_from_engine(model_obj.model_dir)
                runtime_defaults = self._pretrained_config.runtime_defaults
                if self.use_runtime_defaults and runtime_defaults:
                    self.kv_cache_config.fill_empty_fields_from_runtime_defaults(
                        runtime_defaults)

            # Load parallel_config from the checkpoint.
            elif self.model_format is _ModelFormatKind.TLLM_CKPT:
                self._load_config_from_ckpt(model_obj.model_dir)
        else:
            self.model_format = _ModelFormatKind.HF

        if self.speculative_model and speculative_model_obj.is_local_model:
            self.speculative_model_format = _ModelFormatKind.HF

        self.quant_config = self.quant_config or QuantConfig()

        self.calib_config = self.calib_config or CalibConfig()

        # Note: max_batch_size and max_num_tokens in LlmArgs are for runtime,
        # which will be passed to the C++ Executor API, overwriting the values
        # from an built engine. In order to set build configuration, it is
        # recommended to use build_config instead.
        if self.build_config is not None:
            if self.max_batch_size and self.build_config.max_batch_size != self.max_batch_size:
                logger.warning(
                    f"Conflict detected in LlmArgs build_config.max_batch_size "
                    f"({self.build_config.max_batch_size}) != max_batch_size ({self.max_batch_size})."
                    f"The 'max_batch_size' specified in LlmArgs is ignored at "
                    f"engine build and will override at runtime.")
            if self.max_num_tokens and self.build_config.max_num_tokens != self.max_num_tokens:
                logger.warning(
                    f"Conflict detected in LlmArgs build_config.max_num_tokens "
                    f"({self.build_config.max_num_tokens}) != max_batch_size ({self.max_num_tokens})."
                    f"The 'max_num_tokens' specified in LlmArgs is ignored at "
                    f"engine build and will override at runtime.")
        else:
            self.build_config = BuildConfig()
            if self.max_batch_size:
                self.build_config.max_batch_size = self.max_batch_size
            if self.max_num_tokens:
                self.build_config.max_num_tokens = self.max_num_tokens

        # TODO(xiweny): remove the checker when manage weights support all data types
        if self.fast_build and (self.quant_config.quant_algo is QuantAlgo.FP8
                                or self.quant_config.quant_algo is None):
            self._update_plugin_config("manage_weights", True)

        if self.parallel_config._world_size == 1:
            self.build_config.plugin_config.nccl_plugin = None

        if self.enable_lora:
            self.build_config.plugin_config.lora_plugin = 'auto'
            if self.max_lora_rank is not None:
                self.build_config.lora_config.max_lora_rank = self.max_lora_rank

        if self.enable_prompt_adapter:
            self.build_config.max_prompt_embedding_table_size = self.max_prompt_adapter_token * self.build_config.max_batch_size

        if self.perform_config_arbitration:
            self._perform_config_arbitration()

        if self.speculative_config:
            if isinstance(self.speculative_config, LookaheadDecodingConfig):
                lookahead_config = self.speculative_config
                # Update the build config
                _, _, max_draft_tokens, _ = lookahead_config.calculate_speculative_resource(
                )
                self.build_config.speculative_decoding_mode = SpeculativeDecodingMode.LOOKAHEAD_DECODING
                if max_draft_tokens > self.build_config.max_draft_len:
                    self.build_config.max_draft_len = max_draft_tokens

                self.decoding_config = DecodingConfig(
                    decoding_mode=DecodingMode.Lookahead(),
                    lookahead_decoding_config=lookahead_config)
            elif isinstance(self.speculative_config, MedusaDecodingConfig):
                self.decoding_config = DecodingConfig(
                    decoding_mode=DecodingMode.Medusa(),
                    medusa_choices=self.speculative_config.medusa_choices)
            elif isinstance(self.speculative_config, EagleDecodingConfig):
                eagle_config = EagleConfig(
                    self.speculative_config.eagle_choices,
                    self.speculative_config.greedy_sampling,
                    self.speculative_config.posterior_threshold,
                    self.speculative_config.use_dynamic_tree,
                    self.speculative_config.dynamic_tree_max_topK)
                self.decoding_config = DecodingConfig(
                    decoding_mode=DecodingMode.Eagle(),
                    eagle_config=eagle_config)
            else:
                raise ValueError(f"Speculative config type not recognized")

    def _perform_config_arbitration(self):
        '''
        Arbitrate the configurations for the model building. The configs between different functional or performance
        features might be conflicted, and this method will arbitrate the conflicts and raise errors if necessary.
        '''
        self._config_arbitrator = _ConfigArbitrator()
        if self.build_config_mutable:
            if not self.build_config.max_num_tokens:
                self.build_config.max_num_tokens = 2048

            self._setup_enable_chunked_context()
            self._setup_enable_streaming_llm()
            self._setup_quant_config()

            if self.build_config.max_beam_width > 1:
                self._config_arbitrator.claim_func(
                    "beam_search (beam_width > 1)",
                    config_name="kv_cache_config",
                    enable_block_reuse=False)

        else:
            self._setup_build_config_into_config_arbitrator()

        self._setup_kv_cache_config()

        self._config_arbitrator(plugin_config=self.build_config.plugin_config,
                                kv_cache_config=self.kv_cache_config,
                                build_config=self.build_config)

        self._config_arbitrator = None

    @property
    def build_config_mutable(self) -> bool:
        return self.model_format is not _ModelFormatKind.TLLM_ENGINE

    def _update_plugin_config(self, key: str, value: Any):
        setattr(self.build_config.plugin_config, key, value)

    def _load_config_from_engine(self, engine_dir: Path):
        engine_config = EngineConfig.from_json_file(engine_dir / "config.json")
        self._pretrained_config = engine_config.pretrained_config
        self.build_config = engine_config.build_config

        # load and check parallel_config
        mapping = self._pretrained_config.mapping
        if self.parallel_config.tp_size not in (1, mapping.tp_size):
            raise ValueError(
                f"tp_size {self.parallel_config.tp_size} is not consistent with the engine's tp_size {mapping.tp_size}"
            )
        if self.parallel_config.pp_size not in (1, mapping.pp_size):
            raise ValueError(
                f"pp_size {self.parallel_config.pp_size} is not consistent with the engine's pp_size {mapping.pp_size}"
            )
        if self.parallel_config.cp_size not in (1, mapping.cp_size):
            raise ValueError(
                f"cp_size {self.parallel_config.cp_size} is not consistent with the engine's cp_size {mapping.cp_size}"
            )
        self.parallel_config = _ParallelConfig(
            tp_size=mapping.tp_size,
            pp_size=mapping.pp_size,
            cp_size=mapping.cp_size,
            gpus_per_node=mapping.gpus_per_node,
            moe_tp_size=mapping.moe_tp_size,
            moe_ep_size=mapping.moe_ep_size)

    def _load_config_from_ckpt(self, ckpt_dir: Path):
        pretrained_config = PretrainedConfig.from_json_file(ckpt_dir /
                                                            "config.json")
        tp_size = pretrained_config.mapping.tp_size
        pp_size = pretrained_config.mapping.pp_size
        cp_size = pretrained_config.mapping.cp_size
        moe_tp_size = pretrained_config.mapping.moe_tp_size
        moe_ep_size = pretrained_config.mapping.moe_ep_size
        world_size = pretrained_config.mapping.world_size
        gpus_per_node = pretrained_config.mapping.gpus_per_node
        # load parallel_config
        if self.parallel_config.tp_size != 1 and self.parallel_config.tp_size != tp_size:
            raise ValueError(
                f"tp_size {self.parallel_config.tp_size} is not consistent with the checkpoint's tp_size {tp_size}"
            )
        if self.parallel_config.pp_size != 1 and self.parallel_config.pp_size != pp_size:
            raise ValueError(
                f"pp_size {self.parallel_config.pp_size} is not consistent with the checkpoint's pp_size {pp_size}"
            )
        if self.parallel_config.cp_size != 1 and self.parallel_config.cp_size != cp_size:
            raise ValueError(
                f"cp_size {self.parallel_config.cp_size} is not consistent with the checkpoint's cp_size {cp_size}"
            )
        if (self.parallel_config.auto_parallel
                and self.parallel_config.world_size != 1 and world_size != 1):
            raise ValueError(
                f"auto parallel with world_size {self.parallel_config.world_size} does not support checkpoint with "
                "world_size {world_size} > 1")
        if not self.parallel_config.auto_parallel:
            self.parallel_config = _ParallelConfig(tp_size=tp_size,
                                                   pp_size=pp_size,
                                                   cp_size=cp_size,
                                                   gpus_per_node=gpus_per_node,
                                                   moe_tp_size=moe_tp_size,
                                                   moe_ep_size=moe_ep_size)

    def _setup_embedding_parallel_mode(self):
        if self.embedding_parallel_mode == 'NONE':
            self._convert_checkpoint_options['use_parallel_embedding'] = False
        elif self.embedding_parallel_mode == 'SHARDING_ALONG_VOCAB':
            self._convert_checkpoint_options['use_parallel_embedding'] = True
            self._convert_checkpoint_options['embedding_sharding_dim'] = 0
        elif self.embedding_parallel_mode == 'SHARDING_ALONG_HIDDEN':
            self._convert_checkpoint_options['use_parallel_embedding'] = True
            self._convert_checkpoint_options['embedding_sharding_dim'] = 1
        else:
            raise ValueError(
                f"Invalid embedding_parallel_mode: {self.llm_args.embedding_parallel_mode}"
            )

    def _setup_build_config_into_config_arbitrator(self):
        # Setup the ConfigArbitrator with the plugin_config, the runtime configs such as KvCacheConfig should not be
        # conflict with it.
        build_config = asdict(self.build_config)
        del build_config['plugin_config']

        self._config_arbitrator.setup("BuildConfig is readonly",
                                      config_name="build_config",
                                      **build_config)

        plugin_config = asdict(self.build_config.plugin_config)
        self._config_arbitrator.setup("PluginConfig is readonly",
                                      config_name="plugin_config",
                                      **plugin_config)

    def _setup_enable_chunked_context(self):

        def fallback():
            logger.warning(
                f"Disabling chunked context due to configuration conflict.")
            self.enable_chunked_prefill = False

        if self.enable_chunked_prefill:
            if self.build_config_mutable:
                self._config_arbitrator.claim_perf("chunked_context",
                                                   config_name="plugin_config",
                                                   use_paged_context_fmha=True,
                                                   fallback=fallback)

    def _setup_enable_streaming_llm(self):
        if self.build_config.plugin_config.streamingllm:
            self._validate_kv_cache_config()

            self._config_arbitrator.claim_func("streamingllm",
                                               config_name="plugin_config",
                                               streamingllm=True,
                                               use_paged_context_fmha=False)

            self._config_arbitrator.claim_func("streamingllm",
                                               config_name="kv_cache_config",
                                               enable_block_reuse=False)

    def _validate_kv_cache_config(self):
        if self.kv_cache_config is None:
            raise ValueError("KvCacheConfig is required for streaming LLM.")

        if self.kv_cache_config.max_attention_window is None:
            raise ValueError(
                "KvCacheConfig.max_attention_window should be set for streaming LLM."
            )
        if any(i <= 0 for i in self.kv_cache_config.max_attention_window):
            raise ValueError(
                "Elements in KvCacheConfig.max_attention_window should be greater than 0."
            )

        if self.kv_cache_config.sink_token_length is None:
            raise ValueError(
                "KvCacheConfig.sink_token_length should be set for streaming LLM."
            )
        if self.kv_cache_config.sink_token_length <= 0:
            raise ValueError(
                "KvCacheConfig.sink_token_length should be greater than 0.")

    def _setup_kv_cache_config(self):
        assert self.kv_cache_config is not None

        if self.kv_cache_config.enable_block_reuse:
            self._config_arbitrator.claim_func("enable_block_reuse",
                                               config_name="kv_cache_config",
                                               enable_block_reuse=True)
            self._config_arbitrator.claim_func("enable_block_reuse",
                                               config_name="plugin_config",
                                               use_paged_context_fmha=True)

    def _setup_quant_config(self):
        if self.quant_config.quant_algo is QuantAlgo.FP8:
            self._config_arbitrator.claim_func("fp8_quant",
                                               config_name="plugin_config",
                                               use_paged_context_fmha=False)

    def __setstate__(self, state):
        self.__dict__.update(state)

    def __getstate__(self):
        state = self.__dict__.copy()
        if '_config_arbitrator' in state:
            del state['_config_arbitrator']
        return state


class ConfigArbitrateError(Exception):
    ''' Exception raised when there is a conflict in configurations. '''

    def __init__(self, message):
        super().__init__(message)


class _ConfigArbitrator:
    ''' The ConfigArbitrator will arbitrate the options from different sources and raise errors if there are conflicts. '''

    def __init__(self):
        # Dict of configs, the format is {config_name: {option: value}}
        self.virtual_configs: Dict[str, Dict[str, Any]] = {}
        # The claims for functionalities, the format is {config_name: [(func_name, {option: value})]}
        self.func_claims: Dict[str, List[Tuple[str, dict]]] = {}
        # The claims for performances, the format is {perf_name: [(config_name, {option: value}, fallback)]},
        # the fallback is a callback function to be called when the performance is abandoned.
        self.perf_claims: Dict[str, List[Tuple[str, dict,
                                               Optional[Callable[[],
                                                                 None]]]]] = {}
        # Track where the option settings came from, this will be used for messages when encountered conflicts.
        # The format is {config_name: {option: error_information}}
        self.option_sources: Dict[str, Dict[str, str]] = {}

    def __call__(self, **configs) -> None:
        '''
        Args:
            configs: name to config instance for each config need to be arbitrated.
        '''
        self._arbitrate()

        # Apply the successfully arbitrated virtual configs to the real configs
        for name, config in configs.items():
            if name in self.virtual_configs:
                virtual_config = self.virtual_configs[name]
                for option, value in virtual_config.items():
                    setattr(config, option, value)

    def setup(self, info: str, config_name: str, **kwargs):
        ''' Setup with some pre-defined configs comes from environment such as GPU arch. '''
        config = self.virtual_configs.setdefault(config_name, {})
        option_sources = self.option_sources.setdefault(config_name, {})
        for option, value in kwargs.items():
            assert config.get(option, value) == value
            config[option] = value
            option_sources[option] = info

    def claim_func(self, func: str, config_name: str, **options):
        ''' Claim a functionality demanding with configs and options.
        The functionality should be fulfilled, or errors will be raised. '''

        claims = self.func_claims.setdefault(config_name, [])
        claims.append((func, options))

    def claim_perf(self,
                   perf: str,
                   config_name: str,
                   fallback: Optional[Callable[[], None]] = None,
                   **options):
        ''' Claim a performance demanding for configs and options.
        The performance could be abandoned if the demanding is not available.'''
        claims = self.perf_claims.setdefault(perf, [])
        claims.append((config_name, options, fallback))

    def _arbitrate(self):
        ''' Arbitrate the configs for all the functionalities and performances. '''

        # Resolve functionality claims
        for config_name, funcs in self.func_claims.items():
            virtual_config = self.virtual_configs.setdefault(config_name, {})
            option_sources = self.option_sources.setdefault(config_name, {})
            for func, options in funcs:
                for option, value in options.items():
                    if option in virtual_config:
                        if virtual_config[option] != value:
                            existing_func = option_sources[option]
                            raise ConfigArbitrateError(
                                f"Cannot set '{option}' to be '{value}' when enabling '{func}', "
                                f"since '{existing_func}' has set it to be '{virtual_config[option]}'."
                            )
                    else:
                        virtual_config[option] = value
                        # Track where the setting came from
                        option_sources[option] = func

        # copy for restore
        # Resolve performance claims
        for perf, options in self.perf_claims.items():
            option_sources = copy.copy(self.option_sources)
            virtual_configs = copy.copy(self.virtual_configs)
            restore = False
            for config_name, options, fallback in options:
                virtual_config = virtual_configs.setdefault(config_name, {})
                option_source = option_sources.setdefault(config_name, {})
                for option, value in options.items():
                    if option in virtual_config and virtual_config[
                            option] != value:
                        logger.warning(
                            f"Ignoring performance claim '{perf}' for option '{option}' due to conflict."
                        )
                        restore = True
                    else:
                        virtual_config[option] = value
                        option_source[option] = perf
                    if restore: break
                if restore:
                    if fallback: fallback()
                    break

            if not restore:
                self.option_sources = option_sources
                self.virtual_configs = virtual_configs


@dataclass
class _ModelRuntimeContext:
    ''' _ModelRuntimeContext holds the minimum runtime resources for running a model.
    It could be a runtime cache in MPI nodes.
    '''
    engine: Optional[Engine] = None
    mapping: Optional[Mapping] = None
    model_info: Optional[_ModelInfo] = None

    # This is only used when build-cache is enabled
    engine_path: Optional[str] = None

    @property
    def model_arch(self) -> str:
        # "LlaMACausalForLM" or "OPTForCausalLM" and so on
        return self.engine.config.pretrained_config['architecture']


class ModelLoader:
    ''' The ModelLoader is used to build an end-to-end model for a single-gpu.
    It accepts model name or a local model dir, and will download the model if necessary.
    '''

    def __init__(self,
                 llm_args: LlmArgs,
                 workspace: Optional[str | tempfile.TemporaryDirectory] = None,
                 llm_build_stats: Optional["LlmBuildStats"] = None):
        self.llm_args = llm_args
        self._workspace = workspace or tempfile.TemporaryDirectory()
        self.llm_build_stats = llm_build_stats or LlmBuildStats()

        assert self.llm_args.build_config
        self.build_config = self.llm_args.build_config

        self.model_obj = _ModelWrapper(self.llm_args.model)
        self.speculative_model_obj = _ModelWrapper(
            self.llm_args.speculative_model
        ) if self.llm_args.speculative_model is not None else None
        self.convert_checkpoint_options = self.llm_args._convert_checkpoint_options
        self.rank = mpi_rank()
        self.mapping = llm_args.parallel_config.to_mapping()

        self._build_pipeline = []

        # For model from hub, the _model_dir is None, and will updated once downloaded
        self._model_dir: Optional[
            Path] = self.model_obj.model_dir if self.model_obj.is_local_model else None

        self._speculative_model_dir: Optional[
            Path] = self.speculative_model_obj.model_dir if self.speculative_model_obj is not None and self.model_obj.is_local_model else None
        self._model_info: Optional[_ModelInfo] = None
        self._model_format = self.llm_args.model_format

        self.auto_parallel_config = AutoParallelConfig(
            world_size=llm_args.parallel_config.world_size if llm_args.
            parallel_config.auto_parallel else 1)
        default_config = self.llm_args.auto_parallel_config
        self.auto_parallel_config.set_defaults(
            cluster_key=default_config.cluster_key,
            cluster_info=default_config.cluster_info,
            same_buffer_io=default_config.same_buffer_io,
            sharded_io_allowlist=default_config.sharded_io_allowlist,
        )

        self._gather_build_steps()

    def _gather_build_steps(self):
        # Prepare the model processing pipeline
        if isinstance(self.llm_args.model, Module):
            # Build engine from user provided model
            self._build_pipeline.append(
                ("Build TensorRT-LLM engine",
                 self._build_engine_from_inmemory_model))
            return

        if (self.model_obj.is_hub_model
                and self._model_format is not _ModelFormatKind.TLLM_ENGINE) or (
                    self.speculative_model_obj
                    and self.speculative_model_obj.is_hub_model):
            # Download HF model if necessary
            if self.model_obj.model_name is None:
                raise ValueError(
                    "Either model_dir or model should be provided to ModelConfig."
                )
            self._build_pipeline.append(
                ("Downloading HF model", self._download_hf_model))

        if self._model_format is _ModelFormatKind.HF:
            # HF -> TRT checkpoints -> engine
            self._build_pipeline.append(
                ("Loading HF model to memory", self._load_model_from_hf))
            self._build_pipeline.append(
                ("Building TRT-LLM engine", self._build_engine))
        elif self._model_format is _ModelFormatKind.TLLM_CKPT:
            # TRT checkpoints -> engine
            self._build_pipeline.append(("Loading TRT checkpoints to memory",
                                         self._load_model_from_ckpt))
            self._build_pipeline.append(
                ("Build TRT-LLM engine", self._build_engine))
        elif self._model_format is _ModelFormatKind.TLLM_ENGINE:
            # Nothing need to do
            pass
        else:
            raise ValueError(f"Unknown model format {self._model_format}")

    class BuildPipeline:

        def __init__(self, enable_tqdm: bool, labels: List[str],
                     step_handlers: List[Callable],
                     llm_build_stats: "LlmBuildStats"):
            assert len(labels) == len(step_handlers)
            self.labels = labels
            self.step_handlers = step_handlers
            self.llm_build_stats = llm_build_stats

            self.to_log = mpi_rank() == 0
            self.counter = 0

            self.progress_bar = tqdm(
                total=len(labels)) if enable_tqdm and self.to_log else None

        def __call__(self):
            start_time = time.time()

            for i in range(len(self.labels)):
                self.step_forward()

            if self.to_log:
                if self.progress_bar:
                    self.progress_bar.close()
                else:
                    overall_latency = time.time() - start_time
                    print_colored("Loading model done.\n", 'bold_green')
                    print_colored(
                        'Total latency: {:.3f}s\n'.format(overall_latency),
                        'grey')

        def step_forward(self):
            n_steps = len(self.labels)

            label = self.labels[self.counter]

            # display step information
            if self.to_log:
                if self.progress_bar:
                    self.progress_bar.set_description(self.labels[self.counter])
                else:
                    print_colored("Loading Model: ")
                    print_colored(f"[{self.counter+1}/{n_steps}]\t",
                                  'bold_green')
                    print_colored(f"{label}\n")

            # execute the step
            start_time = time.time()
            self.step_handlers[self.counter]()
            # release resource after each step
            release_gc()

            if self.progress_bar:
                self.progress_bar.update(1)

            latency = time.time() - start_time
            if self.to_log and not self.progress_bar:
                print_colored("Time: {:.3f}s\n".format(latency), 'grey')

            self.llm_build_stats.build_steps_info.append((label, latency))

            self.counter += 1

    def __call__(self, engine_dir: Optional[Path] = None) -> Path:
        '''
        The engine_dir is the path to save the built engine.
        '''
        if self.llm_args.model_format is _ModelFormatKind.TLLM_ENGINE:
            return self.model_obj.model_dir

        if self.llm_args.parallel_config.is_multi_gpu:
            torch.cuda.set_device(self.rank)

        pipeline = ModelLoader.BuildPipeline(
            self.llm_args.enable_tqdm,
            [label for label, _ in self._build_pipeline],
            [handler for _, handler in self._build_pipeline],
            llm_build_stats=self.llm_build_stats,
        )
        pipeline()

        assert engine_dir

        runtime_context = _ModelRuntimeContext(
            engine=self._engine,
            mapping=self.mapping,
            model_info=self._model_info,
        )
        ModelLoader.save(runtime_context, self.model_obj.model_dir, engine_dir)
        return engine_dir

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        for attr_name in dir(self):
            if not callable(getattr(
                    self, attr_name)) and not attr_name.startswith("__"):
                if attr_name not in ('model_format', 'workspace'):
                    setattr(self, attr_name, None)

        release_gc()

    @property
    def workspace(self) -> str:
        return self._workspace

    @property
    def model_format(self) -> _ModelFormatKind:
        return self._model_format

    @staticmethod
    def save(
        model: _ModelRuntimeContext,
        model_dir: str,
        engine_dir: str,
    ):
        ''' Save the built engine on a single GPU to the given path. '''
        mapping = model.mapping
        rank = mapping.rank

        def copy_hf_tokenizer_data_to_engine_dir():
            # Copy the HF tokenizer stuff to the engine dir so that we can use the engine dir as a standalone model dir
            # supports end-to-end task.
            # This is only for HF model for now, not available for users' customized tokenizers.
            import shutil

            for name in os.listdir(model_dir):
                src = os.path.join(model_dir, name)
                dst = os.path.join(engine_dir, name)
                if name.startswith('tokenizer'):
                    src = os.path.realpath(src) if os.path.islink(src) else src
                    if os.path.isdir(src):
                        shutil.copytree(src, dst, dirs_exist_ok=True)
                    else:
                        shutil.copy2(src, dst)

        model.engine.save(engine_dir)
        if rank == 0:
            copy_hf_tokenizer_data_to_engine_dir()

    @staticmethod
    def get_model_format(model_dir: str) -> _ModelFormatKind:
        ''' Get the format of the model.  '''
        if not (Path(model_dir) / 'config.json').exists():
            raise ValueError(
                f"Failed to infer model format because no config.json exists in {model_dir}"
            )

        with open(Path(model_dir) / 'config.json') as f:
            config = json.load(f)

        try:
            if 'pretrained_config' in config and 'build_config' in config:
                model_format = _ModelFormatKind.TLLM_ENGINE
                EngineConfig.from_json_file(Path(model_dir) / 'config.json')
            elif 'architecture' in config and 'dtype' in config:
                model_format = _ModelFormatKind.TLLM_CKPT
                PretrainedConfig.from_checkpoint(model_dir)
            else:
                model_format = _ModelFormatKind.HF
                AutoConfig.from_hugging_face(model_dir)
        except Exception as e:
            raise ValueError(
                f"Inferred model format {model_format}, but failed to load config.json: {e}"
            )
        else:
            return model_format

    def _download_hf_model(self):
        ''' Download HF model from third-party model hub like www.modelscope.cn or huggingface.  '''
        model_dir = None
        speculative_model_dir = None
        # Only the rank0 are allowed to download model
        if mpi_rank() == 0:
            assert self._workspace is not None
            assert isinstance(self.model_obj.model_name, str)
            # this will download only once when multiple MPI processes are running

            model_dir = download_hf_model(self.model_obj.model_name,
                                          revision=self.llm_args.revision)
            print_colored(f"Downloaded model to {model_dir}\n", 'grey')
            if self.speculative_model_obj:
                speculative_model_dir = download_hf_model(
                    self.speculative_model_obj.model_name)
                print_colored(f"Downloaded model to {speculative_model_dir}\n",
                              'grey')
        # Make all the processes got the same model_dir
        self._model_dir = mpi_broadcast(model_dir, root=0)
        self.model_obj.model_dir = self._model_dir  # mark as a local model
        assert self.model_obj.is_local_model
        if self.speculative_model_obj:
            self._speculative_model_dir = mpi_broadcast(speculative_model_dir,
                                                        root=0)
            self.speculative_model_obj.model_dir = self._speculative_model_dir

            assert self.speculative_model_obj.is_local_model

    def _load_model_from_hf(self):
        ''' Load a TRT-LLM model from a HF model. '''
        assert self._model_dir is not None

        model_cls = AutoModelForCausalLM.get_trtllm_model_class(
            self._model_dir, self.llm_args.trust_remote_code,
            self.llm_args.decoding_config.decoding_mode
            if hasattr(self.llm_args, "speculative_model")
            and self.llm_args.speculative_model else None)

        # Update quant_config if it's ModelOpt quantized ckpt
        user_quant_config = self.llm_args.quant_config
        hf_quant_config_path = Path(self._model_dir) / "hf_quant_config.json"
        if hf_quant_config_path.exists():
            logger.info(
                f"Found {hf_quant_config_path}, pre-quantized checkpoints are used."
            )
            already_quantized = True
            with open(hf_quant_config_path, "r") as f:
                hf_quant_config = json.load(f)
                hf_quant_algo = hf_quant_config["quantization"].get(
                    "quant_algo")
                if hf_quant_algo == "FP8" and user_quant_config.quant_algo \
                        and user_quant_config.quant_algo != QuantAlgo.FP8:
                    raise ValueError(
                        f"Expecting quant_algo to be FP8, got {user_quant_config.quant_algo}."
                    )
                user_quant_config.quant_algo = hf_quant_algo
                logger.info(f"quant_algo is set to {hf_quant_algo}")

                hf_kv_cache_quant_algo = hf_quant_config["quantization"].get(
                    "kv_cache_quant_algo")
                if hf_kv_cache_quant_algo != user_quant_config.kv_cache_quant_algo:
                    if user_quant_config.kv_cache_quant_algo is None:
                        user_quant_config.kv_cache_quant_algo = hf_kv_cache_quant_algo
                        logger.info(
                            f"kv_cache_quant_algo is set to {hf_kv_cache_quant_algo}"
                        )
                    elif user_quant_config.kv_cache_quant_algo == QuantAlgo.FP8 and hf_kv_cache_quant_algo is None:
                        logger.warning(
                            f"User specified kv_cache_quant_algo {user_quant_config.kv_cache_quant_algo} "
                            f"will overwrite {hf_kv_cache_quant_algo} from {hf_quant_config_path}."
                        )
                    else:
                        raise ValueError(
                            f"User specified kv_cache_quant_algo {user_quant_config.kv_cache_quant_algo}, "
                            f"while it's {hf_kv_cache_quant_algo} in {hf_quant_config_path}."
                        )
        else:
            already_quantized = False

        # FP4 Gemm force to use plugin.
        if self.llm_args.quant_config.quant_mode.has_nvfp4():
            self.llm_args.build_config.plugin_config.gemm_plugin = "nvfp4"

        if self.llm_args.load_format == 'dummy':
            config = model_cls.config_class.from_hugging_face(
                str(self._model_dir),
                dtype=self.llm_args.dtype,
                mapping=self.mapping,
                quant_config=self.llm_args.quant_config,
                **self.convert_checkpoint_options,
            )
            self.model = model_cls(config)
        elif self.llm_args.quant_config.requires_calibration and not already_quantized:
            assert self.workspace is not None
            checkpoint_dir = f"{self.workspace}/quantized-checkpoint"
            if self.rank == 0:
                model_cls.quantize(
                    self._model_dir,
                    checkpoint_dir,
                    dtype=self.llm_args.dtype,
                    mapping=self.mapping,
                    quant_config=self.llm_args.quant_config,
                    **self.llm_args.calib_config.to_dict(),
                    trust_remote_code=self.llm_args.trust_remote_code,
                )
            if self.llm_args.parallel_config.is_multi_gpu:
                mpi_barrier()
            self.model = model_cls.from_checkpoint(checkpoint_dir,
                                                   rank=self.mapping.rank)
        else:
            self.model = model_cls.from_hugging_face(
                str(self._model_dir),
                dtype=self.llm_args.dtype,
                mapping=self.mapping,
                quant_config=self.llm_args.quant_config,
                load_model_on_cpu=
                True,  # TODO:TRTLLM-195 to enhance the weights loading memory usage and chose best location
                trust_remote_code=self.llm_args.trust_remote_code,
                speculative_model=self._speculative_model_dir,
                speculative_config=self.llm_args.speculative_config
                if not isinstance(self.llm_args.speculative_config,
                                  LookaheadDecodingConfig) else None,
                **self.convert_checkpoint_options,
            )

        self.pretrained_config = self.model.config
        self._model_info = _ModelInfo.from_pretrained_config(
            self.pretrained_config)

    def _load_model_from_ckpt(self):
        ''' Load a TRT-LLM model from checkpoint. '''
        self.pretrained_config = PretrainedConfig.from_json_file(
            os.path.join(self._model_dir, 'config.json'))
        self.pretrained_config.mapping = self.mapping

        #TODO: TRTLLM-1091, change the architecture in the checkpoint to TRT-LLM one, not HF one.
        architecture = self.pretrained_config.architecture
        assert architecture in MODEL_MAP, \
            f"Unsupported model architecture: {architecture}"
        model_cls = MODEL_MAP[architecture]
        if self.llm_args.load_format == 'dummy':
            self.model = model_cls(self.pretrained_config)
        else:
            self.model = model_cls.from_checkpoint(
                self._model_dir, config=self.pretrained_config)
        self._model_info = _ModelInfo.from_pretrained_config(
            self.pretrained_config)

        # load parallel embedding related options
        self.convert_checkpoint_options[
            'use_parallel_embedding'] = self.pretrained_config.use_parallel_embedding

    def _build_engine_from_inmemory_model(self):
        assert isinstance(self.llm_args.model, Module)
        self._model_info = _ModelInfo.from_module(self.model)

    def _build_engine(self):
        assert isinstance(
            self.build_config,
            BuildConfig), f"build_config is not set yet: {self.build_config}"

        # avoid the original build_config is modified, avoid the side effect
        copied_build_config = copy.deepcopy(self.build_config)

        copied_build_config.update(
            auto_parallel_config=self.auto_parallel_config)
        copied_build_config.update_kv_cache_type(self._model_info.architecture)
        if self.auto_parallel_config.enabled:
            self.model.config.mapping.rank = self.rank
        assert self.model is not None, "model is loaded yet."

        self._engine = build(self.model, copied_build_config)
        self.mapping = self.model.config.mapping

        # delete the model explicitly to free all the build-time resources
        self.model = None

    def _save_engine_for_runtime(self):
        '''
        Persist the engine to disk for the cpp runtime. Currently, the cpp runtime can accept an engine path,
        that requires the engine should always be saved to disk.

        This explicit saving will be removed in the future when the cpp runtime can accept the engine buffer directly.
        But this is necessary for a build cache, but it can be optimized to async IO.
        '''
        if self.build_cache_enabled:
            self._model_dir = self.engine_cache_stage.cache_dir
            self._model_format = _ModelFormatKind.TLLM_ENGINE
            return

    def _load_engine_buffer(self):
        # Load engine buffer from disk
        self._engine = Engine.from_dir(self._model_dir)

    @staticmethod
    def load_extra_build_configs_from_engine(
            model_dir: str) -> Optional[Namespace]:
        ''' Load the extra build configs from the engine directory, return None if model isn't an engine. '''
        if ModelLoader.get_model_format(
                model_dir) is not _ModelFormatKind.TLLM_ENGINE:
            return None

        with open(Path(model_dir) / "config.json", "r") as f:
            engine_config = json.load(f)

        build_config = engine_config['build_config']
        build_config.pop("plugin_config")
        return Namespace(**build_config)

    @staticmethod
    def load_hf_tokenizer(
            model_dir,
            trust_remote_code: bool = True,
            use_fast: bool = True) -> Optional[TransformersTokenizer]:
        if (tokenizer := load_hf_tokenizer(model_dir, trust_remote_code,
                                           use_fast)) is not None:
            return tokenizer
        else:
            logger.error(f"Failed to load tokenizer from {model_dir}")
            return None


class CachedModelLoader:
    '''
    The CacheModelLoader is used to build the model in both single or multi-gpu, with cache might be enabled.
    '''

    def __init__(
        self,
        llm_args: LlmArgs,
        llm_build_stats: weakref.ReferenceType["LlmBuildStats"],
        mpi_session: Optional[MpiSession] = None,
        workspace: Optional[str] = None,
    ):
        self.llm_args = llm_args
        self.mpi_session = mpi_session
        self._workspace = workspace or tempfile.TemporaryDirectory()
        self.llm_build_stats = llm_build_stats

        # This is used for build cache. To compute the cache key, a local HF model is required, it could be download
        # from HF model hub, so this helps to hold the path.
        self._hf_model_dir: Optional[Path] = None

    @property
    def workspace(self) -> Path:
        return Path(self._workspace.name) if isinstance(
            self._workspace, tempfile.TemporaryDirectory) else Path(
                self._workspace)

    def __call__(self) -> Tuple[Path, Union[Path, None]]:

        if self.llm_args.model_format is _ModelFormatKind.TLLM_ENGINE:
            return self.llm_args.model, None

        self.engine_cache_stage: Optional[CachedStage] = None

        self._hf_model_dir = None

        self.model_loader = ModelLoader(self.llm_args)

        if self.build_cache_enabled:
            print_colored("Build cache is enabled.\n", 'yellow')
            if self.model_loader.model_obj.is_hub_model:
                # This will download the config.json from HF model hub, this helps to create a PretrainedConfig for
                # cache key.
                self._hf_model_dir = download_hf_pretrained_config(
                    self.model_loader.model_obj.model_name,
                    revision=self.llm_args.revision)

            elif self.model_loader.model_obj.is_local_model:
                self._hf_model_dir = self.model_loader.model_obj.model_dir if self.llm_args.model_format is _ModelFormatKind.HF else None

            self.engine_cache_stage = self._get_engine_cache_stage()
            if self.engine_cache_stage.is_cached():
                self.llm_build_stats.cache_hitted = True
                print_colored(
                    f"Reusing cached engine in {self.engine_cache_stage.get_engine_path()}\n\n",
                    'grey')
                self.model_loader.model_obj.model_dir = self.engine_cache_stage.get_engine_path(
                )
                self.llm_build_stats.engine_dir = self.model_loader.model_obj.model_dir
                return self.llm_build_stats.engine_dir, self._hf_model_dir

        if (self.llm_args.backend is not None):
            if self.llm_args.backend != "pytorch":
                raise ValueError(
                    f'backend {self.llm_args.backend} is not supported.')

            if self.model_loader.model_obj.is_hub_model:
                hf_folder = download_hf_model(
                    self.model_loader.model_obj.model_name,
                    self.llm_args.revision)
                self._hf_model_dir = hf_folder
            else:
                self._hf_model_dir = self.model_loader.model_obj.model_dir

            if self.llm_args.quant_config.quant_algo is not None:
                logger.warning(
                    "QuantConfig for pytorch backend is ignored. You can load"
                    "quantized model with hf_quant_config.json directly.")
            return None, self._hf_model_dir

        return self._build_model(), self._hf_model_dir

    def get_engine_dir(self) -> Path:
        if self.llm_args.model_format is _ModelFormatKind.TLLM_ENGINE:
            return self.model_obj.model_dir

        # generate a new path for writing the engine
        if self.build_cache_enabled:
            cache_stage = self._get_engine_cache_stage()
            return cache_stage.get_engine_path()

        return self.workspace / "tmp.engine"

    @property
    def build_cache_enabled(self) -> bool:
        _enable_build_cache, _ = get_build_cache_config_from_env()

        return (self.llm_args.enable_build_cache or _enable_build_cache) and (
            self.llm_args.model_format is _ModelFormatKind.HF
        ) and not self.llm_args.parallel_config.auto_parallel

    def _get_engine_cache_stage(self) -> CachedStage:
        ''' Get the cache stage for engine building. '''
        build_cache = BuildCache(self.llm_args.enable_build_cache)

        assert self._hf_model_dir is not None, "HF model dir is required for cache key."

        def serialize(d) -> str:
            dic = asdict(d) if not isinstance(
                d, PretrainedConfig) else d.to_dict()
            return json.dumps(dic, sort_keys=True)

        parallel_config = self.llm_args.parallel_config

        force_rebuild = False
        if parallel_config.auto_parallel:
            force_rebuild = True
        if self.llm_args.model_format is not _ModelFormatKind.HF:
            force_rebuild = True

        return build_cache.get_engine_building_cache_stage(
            build_config=self.llm_args.build_config,
            model_path=self._hf_model_dir,
            force_rebuild=force_rebuild,
            # Other configs affecting the engine building
            parallel_config=serialize(parallel_config),
            pretrained_config=serialize(self.get_pretrained_config()),
            quant_config=serialize(self.llm_args.quant_config),
        )

    def get_pretrained_config(self) -> PretrainedConfig:
        ''' Get the PretrainedConfig for cache key.
        NOTE, this is not the HF model's config, but the TRT-LLM's config. We use this as a generic information for
        HF and other models. '''
        assert self._hf_model_dir is not None
        return AutoConfig.from_hugging_face(
            self._hf_model_dir,
            mapping=self.llm_args.parallel_config.to_mapping(),
            quant_config=self.llm_args.quant_config,
            dtype=self.llm_args.dtype)

    def _build_model(self) -> Path:
        model_format = self.llm_args.model_format

        def build_task(engine_dir: Path):
            if model_format is not _ModelFormatKind.TLLM_ENGINE:
                model_loader_kwargs = {
                    'llm_args': self.llm_args,
                    'workspace': str(self.workspace),
                    'llm_build_stats': self.llm_build_stats,
                }

                if self.llm_args.parallel_config.is_multi_gpu:
                    assert self.mpi_session
                    # The engine_dir:Path will be stored to MPINodeState.state
                    build_infos = self.mpi_session.submit_sync(
                        CachedModelLoader._node_build_task,
                        engine_dir=engine_dir,
                        **model_loader_kwargs)
                    self.llm_build_stats.build_steps_info = build_infos[0]

                else:  # single-gpu
                    with ModelLoader(**model_loader_kwargs) as model_loader:
                        model_loader(engine_dir=engine_dir)

                release_gc()

        has_storage = True
        if self.build_cache_enabled:
            try:
                # TODO[chunweiy]: Cover the case when the model is from HF model hub.
                if self.model_loader.model_obj.is_local_model:
                    # This is not perfect, but will make build-cache much more robust.
                    free_storage = self.engine_cache_stage.parent.free_storage_in_gb(
                    )
                    model_size = get_directory_size_in_gb(
                        self.model_loader.model_obj.model_dir)
                    require_size = model_size * 1.3
                    has_storage = free_storage >= require_size

                    if not has_storage:
                        print_colored(
                            f"Build cache is disabled since the cache storage is too small.\n ",
                            'yellow')
                        print_colored(
                            f"Free storage: {free_storage}GB, Required storage: {require_size}GB\n",
                            'grey')
            except ValueError:
                has_storage = False
            except Exception as e:
                logger.error(e)
                has_storage = False

            if enable_llm_debug():
                print_colored(f"Has cache storage: {has_storage}\n", 'yellow')

            if has_storage:
                with self.engine_cache_stage.write_guard() as engine_dir:
                    build_task(engine_dir)
                    self.llm_build_stats.cache_hitted = True

            else:
                print_colored(
                    "The cache directory is too small, build-cache is disabled.\n",
                    'grey')
                self.llm_build_stats.cache_hitted = False
                self.llm_build_stats.cache_info = "The cache root directory is too small."

        if not (has_storage and self.build_cache_enabled):
            build_task(self.get_engine_dir())

        return self.get_engine_dir()

    @print_traceback_on_error
    @staticmethod
    def _node_build_task(
        llm_args: LlmArgs,
        workspace: Optional[str | tempfile.TemporaryDirectory] = None,
        llm_build_stats: Optional['LlmBuildStats'] = None,
        engine_dir: Optional[Path] = None,
    ):
        if MPINodeState.is_initialized():
            raise RuntimeError("The MPI node is already initialized.")

        with ModelLoader(llm_args,
                         workspace=workspace,
                         llm_build_stats=llm_build_stats) as model_loader:
            model_loader(engine_dir=engine_dir)
            return model_loader.llm_build_stats.build_steps_info

    def save(self, engine_dir: Path):
        # copy the engine directory to the target directory
        shutil.copytree(self.get_engine_dir(), engine_dir)


@dataclass
class LlmBuildStats:
    ''' LlmBuildStats is the statistics for the LLM model building. '''
    # Whether the cache is hit for the engine
    cache_hitted: bool = False
    cache_info: Optional[str] = None

    model_from_hf_hub: bool = False

    local_model_dir: Optional[Path] = None

    # The path to the trt-llm engine
    engine_dir: Optional[Path] = None

    # The build steps information, including the step name and the latency in seconds.
    build_steps_info: List[Tuple[str, float]] = field(default_factory=list)
