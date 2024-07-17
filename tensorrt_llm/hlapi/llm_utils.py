__all__ = [
    'LlmArgs',
    'LlmBuildStats',
    'ModelLoader',
    '_ModelRuntimeContext',
    '_ModelInfo',
    '_ParallelConfig',
    '_ModelFormatKind',
    'BatchingType',
    'ExecutorConfig',
    'SchedulerConfig',
    'KvCacheConfig',
    'ContextChunkingPolicy',
    'CapacitySchedulerPolicy',
    'BuildConfig',
    'BuildCacheConfig',
    'QuantConfig',
    'CachedModelLoader',
    'ConfigArbitrateError',
    '_ConfigArbitrator',
]

import copy
import json
import os
import shutil
import tempfile
import time
from argparse import Namespace
from dataclasses import asdict, dataclass, field, fields
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import tensorrt as trt
import torch
from tqdm import tqdm
from transformers import PretrainedConfig as HfPretrainedConfig
from transformers import PreTrainedTokenizerBase

from tensorrt_llm.models.llama.config import LLaMAConfig

from .._utils import mpi_barrier, mpi_broadcast, mpi_rank, release_gc
from ..auto_parallel import AutoParallelConfig, infer_cluster_config
from ..bindings.executor import (BatchingType, CapacitySchedulerPolicy,
                                 ContextChunkingPolicy, DecodingConfig,
                                 ExecutorConfig, KvCacheConfig, PeftCacheConfig,
                                 SchedulerConfig)
from ..builder import BuildConfig, Engine, EngineConfig, build
from ..logger import logger
from ..mapping import Mapping
from ..models import MODEL_MAP
from ..models.modeling_utils import PretrainedConfig, QuantAlgo, QuantConfig
from ..module import Module
from .build_cache import (BuildCache, BuildCacheConfig, CachedStage,
                          get_build_cache_config_from_env)
from .mpi_session import MPINodeState, MpiSession
from .tokenizer import TokenizerBase, TransformersTokenizer, tokenizer_factory
# TODO[chunweiy]: move the following symbols back to utils scope, and remove the following import
from .utils import (GpuArch, download_hf_model, download_hf_pretrained_config,
                    file_with_glob_exists, file_with_suffix_exists,
                    print_colored, print_traceback_on_error)


@dataclass
class _ParallelConfig:
    ''' The model distribution configs for LLM.  '''
    tp_size: int = 1
    pp_size: int = 1
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
            if self.tp_size > 1 or self.pp_size > 1:
                raise RuntimeError(
                    "manually TP and PP are not supported in auto parallel mode."
                )
            return self._world_size

        if self._world_size > 1:
            raise RuntimeError(
                "world_size > 1 is only supported in auto parallel mode.")
        return self.tp_size * self.pp_size

    @world_size.setter
    def world_size(self, world_size: int):
        if self.auto_parallel:
            self._world_size = world_size
        elif (not self.auto_parallel
              ) and world_size != self.tp_size * self.pp_size:
            raise ValueError(
                f"world_size {world_size} should be equal to tp_size * pp_size {self.tp_size * self.pp_size} "
                "in non-auto_parallel mode.\n"
                "For non-auto-parallel mode, the world_size is not needed to set"
            )

    @property
    def is_multi_gpu(self) -> bool:
        return self.world_size > 1


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
class LlmArgs:
    '''
    The arguments for constructing a LLM instance.

    Parameters:
    model (str or Path): The model name or a local model directory.
        Note that if the value could be both a model name or a local model directory,
        the local model directory will be prioritized.

    parallel_config (_ParallelConfig): The parallel configuration for the model.
        Default is an empty _ParallelConfig instance.

    tokenizer (str, Path, TokenizerBase, PreTrainedTokenizerBase, optional):
        The name or path of a HuggingFace Transformers tokenizer, or the loaded tokenizer.
        Default is None.

    skip_tokenizer_init (bool):
        If true, skip initialization of tokenizer and detokenizer. LLM.generate and
        LLM.generate_async will accept prompt token ids as input only.

    tokenizer_revision (str, optional): The revision of the tokenizer to use.
        Default is None.

    dtype (str, default="auto"): The data type for the model weights and activations.
        Can be "float16", "bfloat16", "float32", or "auto". If "auto", the data type
        will be automatically inferred from the source model. If the source data type
        is "float32", it will be converted to "float16".

    revision (str, optional): The revision of the model to use.
        Default is None.

    build_config (BuildConfig, default=BuildConfig()): The build configuration for the model.
        Default is an empty BuildConfig instance.

    quant_config (QuantConfig, default=QuantConfig()): The quantization configuration for the model.
        Default is an empty QuantConfig instance.

    embedding_parallel_mode (str, default="SHARDING_ALONG_VOCAB"): The parallel mode for embeddings.

    share_embedding_table (bool, default=False): Whether to share the embedding table.

    kv_cache_config (KvCacheConfig, optional): The key-value cache configuration for the model.
        Default is None.

    peft_cache_config (PeftCacheConfig, optional): The PEFT cache configuration for the model.
        Default is None.

    decoding_config (DecodingConfig, optional): The decoding configuration for the model.
        Default is None.

    logits_post_processor_map (Dict[str, Callable], optional): A map of logit post-processing functions.
        Default is None.

    scheduler_config (SchedulerConfig, default=SchedulerConfig()): The scheduler configuration for the model.
        Default is an empty SchedulerConfig instance.

    normalize_log_probs (bool, default=False): Whether to normalize log probabilities for the model.

    iter_stats_max_iterations (int, optional): The maximum number of iterations for iteration statistics.
        Default is None.

    request_stats_max_iterations (int, optional): The maximum number of iterations for request statistics.
        Default is None.

    batching_type (BatchingType, optional): The batching type for the model.
        Default is None.

    enable_build_cache (bool or BuildCacheConfig, optional): Whether to enable build caching for the model.
        Default is None.

    enable_tqdm (bool, default=False): Whether to display a progress bar during model building.
    '''

    model: Union[str, Path]

    parallel_config: _ParallelConfig = field(default_factory=_ParallelConfig)

    tokenizer: Optional[Union[str, Path, TokenizerBase,
                              PreTrainedTokenizerBase]] = None

    skip_tokenizer_init: bool = False

    tokenizer_revision: Optional[str] = None

    dtype: str = "auto"

    revision: Optional[str] = None

    # BuildConfig is introduced to give users a familiar interface to configure the model building.
    build_config: Optional[BuildConfig] = None

    quant_config: QuantConfig = field(default_factory=QuantConfig)

    # A handful of options from PretrainedConfig
    embedding_parallel_mode: str = 'SHARDING_ALONG_VOCAB'

    share_embedding_table: bool = False

    # Several options from ExecutorConfig, expanded here for less hierarchy
    kv_cache_config: Optional[KvCacheConfig] = None

    peft_cache_config: Optional[PeftCacheConfig] = None

    # TODO[enweiz]: this might affect medusa, and could be removed in the future for API consistency
    decoding_config: Optional[DecodingConfig] = None

    logits_post_processor_map: Optional[Dict[str, Callable]] = None

    scheduler_config: SchedulerConfig = field(default_factory=SchedulerConfig)

    normalize_log_probs: bool = False

    iter_stats_max_iterations: Optional[int] = None

    request_stats_max_iterations: Optional[int] = None

    batching_type: Optional[BatchingType] = None

    # Once set, the model will reuse the build_cache
    enable_build_cache: Union[BuildCacheConfig, bool] = False

    # Display the model building progress bar
    enable_tqdm: bool = False

    def __post_init__(self):
        # NOTE: this is only for the compatibility with the old API, and will be removed in the future
        # chunked context is disabled by default, and it is recommended to keep it enabled.
        # The underlying implementation might disable it if it is not supported.
        self.enable_chunked_context: bool = False

        if self.skip_tokenizer_init:
            self.tokenizer = None
        else:
            self.tokenizer = tokenizer_factory(self.tokenizer)

        self._engine_config: Optional[EngineConfig] = None

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

        # This is used to hold th options for convert_checkpoint
        self._convert_checkpoint_options = {}

    @classmethod
    def from_kwargs(cls, **kwargs) -> "LlmArgs":
        LlmArgs._check_executor_config_options_consistency()

        parallel_config = _ParallelConfig(
            tp_size=kwargs.pop('tensor_parallel_size', 1),
            pp_size=kwargs.pop('pipeline_parallel_size', 1),
            auto_parallel=kwargs.pop('auto_parallel', False),
        )
        # world_size is only used for auto_parallel mode
        world_size = kwargs.pop('world_size', 1)
        if parallel_config.auto_parallel:
            parallel_config.world_size = world_size

        if devices := kwargs.pop('devices', None):
            parallel_config.devices = devices

        ret = cls(parallel_config=parallel_config, **kwargs)
        ret.setup()
        return ret

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

        self._check_model_or_model_dir()

        self._setup_embedding_parallel_mode()

        if self.enable_build_cache:
            self.enable_build_cache = BuildCacheConfig() if isinstance(
                self.enable_build_cache, bool) else self.enable_build_cache
            if not isinstance(self.enable_build_cache, BuildCacheConfig):
                raise ValueError(
                    f"Invalid build_cache_config: {self.enable_build_cache}")

        if self.is_local_model:
            # Load parallel_config from the engine.
            self.model_format = ModelLoader.get_model_format(self.model_dir)
            if self.model_format is _ModelFormatKind.TLLM_ENGINE:
                if self.build_config is not None:
                    logger.warning(
                        "The build_config is ignored for model format of TLLM_ENGINE."
                    )
                self._load_config_from_engine(Path(self.model_dir))

            # Load parallel_config from the checkpoint.
            elif self.model_format is _ModelFormatKind.TLLM_CKPT:
                self._load_config_from_ckpt(Path(self.model_dir))
        else:
            self.model_format = _ModelFormatKind.HF

        self.build_config = self.build_config or BuildConfig()

        self._config_arbitrator = _ConfigArbitrator()
        if self.build_config_mutable:
            if not self.build_config.max_num_tokens:
                self.build_config.max_num_tokens = 2048

            if not GpuArch.is_post_ampere():
                self._config_arbitrator.setup("pre-ampere not supported",
                                              config_name="plugin_config",
                                              use_paged_context_fmha=False)

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

    def _check_model_or_model_dir(self):
        if not self.model:
            raise ValueError("model should be provided.")
        assert isinstance(self.model,
                          (str, Path)), f"Invalid model: {self.model}"
        model_dir = Path(self.model)
        if model_dir.exists() and model_dir.is_dir():
            self.model = model_dir

    @property
    def is_local_model(self) -> bool:
        return isinstance(self.model, Path)

    @property
    def is_hub_model(self) -> bool:
        return not self.is_local_model

    @property
    def model_dir(self) -> Path:
        assert self.is_local_model
        return self.model

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
        self.parallel_config = _ParallelConfig(
            tp_size=mapping.tp_size,
            pp_size=mapping.pp_size,
        )

    def _load_config_from_ckpt(self, ckpt_dir: Path):
        pretrained_config = PretrainedConfig.from_json_file(ckpt_dir /
                                                            "config.json")
        tp_size = pretrained_config.mapping.tp_size
        pp_size = pretrained_config.mapping.pp_size
        world_size = pretrained_config.mapping.world_size

        # load parallel_config
        if self.parallel_config.tp_size != 1 and self.parallel_config.tp_size != tp_size:
            raise ValueError(
                f"tp_size {self.parallel_config.tp_size} is not consistent with the checkpoint's tp_size {tp_size}"
            )
        if self.parallel_config.pp_size != 1 and self.parallel_config.pp_size != pp_size:
            raise ValueError(
                f"pp_size {self.parallel_config.pp_size} is not consistent with the checkpoint's pp_size {pp_size}"
            )
        if (self.parallel_config.auto_parallel
                and self.parallel_config.world_size != 1 and world_size != 1):
            raise ValueError(
                f"auto parallel with world_size {self.parallel_config.world_size} does not support checkpoint with "
                "world_size {world_size} > 1")
        if not self.parallel_config.auto_parallel:
            self.parallel_config = _ParallelConfig(
                tp_size=tp_size,
                pp_size=pp_size,
            )

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

        self._convert_checkpoint_options[
            'share_embedding_table'] = self.share_embedding_table

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
            self.enable_chunked_context = False

        if self.enable_chunked_context:
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
        if self.kv_cache_config.max_attention_window <= 0:
            raise ValueError(
                "KvCacheConfig.max_attention_window should be greater than 0.")

        if self.kv_cache_config.sink_token_length is None:
            raise ValueError(
                "KvCacheConfig.sink_token_length should be set for streaming LLM."
            )
        if self.kv_cache_config.sink_token_length <= 0:
            raise ValueError(
                "KvCacheConfig.sink_token_length should be greater than 0.")

    def _setup_kv_cache_config(self):
        assert self.kv_cache_config is not None

        if not GpuArch.is_post_ampere():
            self._config_arbitrator.setup("pre-ampere not supported",
                                          config_name="kv_cache_config",
                                          enable_block_reuse=False)

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
    engine_buffer: Optional[trt.IHostMemory] = None
    tokenizer: Optional[TokenizerBase] = None
    # engine_config is only used for saving the engine to disk
    engine_config: Optional[Union[dict, EngineConfig]] = None
    mapping: Optional[Mapping] = None
    model_info: Optional[_ModelInfo] = None

    # This is only used when build-cache is enabled
    engine_path: Optional[str] = None

    @property
    def engine(self) -> trt.IHostMemory:
        assert self.engine_buffer is not None
        return self.engine_buffer

    @property
    def model_arch(self) -> str:
        # "LlaMACausalForLM" or "OPTForCausalLM" and so on
        return self.engine_config.pretrained_config['architecture']


class ModelLoader:
    ''' The ModelLoader is used to build an end-to-end model for a single-gpu.
    It accepts model name or a local model dir, and will download the model if necessary.
    '''

    def __init__(self,
                 llm_args: LlmArgs,
                 tokenizer: Optional[TokenizerBase],
                 workspace: Optional[str | tempfile.TemporaryDirectory] = None,
                 llm_build_stats: Optional["LlmBuildStats"] = None):
        self.llm_args = llm_args
        self.tokenizer = tokenizer
        self._workspace = workspace or tempfile.TemporaryDirectory()
        self.llm_build_stats = llm_build_stats or LlmBuildStats()

        assert self.llm_args.build_config
        self.build_config = self.llm_args.build_config

        self.convert_checkpoint_options = self.llm_args._convert_checkpoint_options
        self.rank = mpi_rank() if llm_args.parallel_config.is_multi_gpu else 0
        if llm_args.parallel_config.is_multi_gpu and not llm_args.parallel_config.auto_parallel:
            self.mapping = Mapping(
                tp_size=llm_args.parallel_config.tp_size,
                pp_size=llm_args.parallel_config.pp_size,
                rank=self.rank,
                world_size=llm_args.parallel_config.world_size,
            )
        else:
            self.mapping = Mapping()

        self._build_pipeline = []

        # For model from hub, the _model_dir is None, and will updated once downloaded
        self._model_dir: Optional[
            Path] = self.llm_args.model_dir if self.llm_args.is_local_model else None
        self._model_info: Optional[_ModelInfo] = None
        self._model_name = self.llm_args.model
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

        if self.llm_args.is_hub_model and self._model_format is not _ModelFormatKind.TLLM_ENGINE:
            # Download HF model if necessary
            if self.llm_args.model is None:
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
            return self.llm_args.model_dir

        if self.llm_args.parallel_config.is_multi_gpu:
            torch.cuda.set_device(self.rank)

        len(self._build_pipeline)
        to_log = self.rank == 0

        pipeline = ModelLoader.BuildPipeline(
            self.llm_args.enable_tqdm,
            [label for label, _ in self._build_pipeline],
            [handler for _, handler in self._build_pipeline],
            llm_build_stats=self.llm_build_stats,
        )
        pipeline()

        if not hasattr(self, '_engine_config'):
            raise RuntimeError("config is not loaded.")

        config = self._engine_config

        assert engine_dir

        runtime_context = _ModelRuntimeContext(
            tokenizer=self.tokenizer,
            engine_buffer=self._engine_buffer,
            engine_config=config,
            mapping=self.mapping,
            model_info=self._model_info,
        )
        ModelLoader.save(runtime_context, self.llm_args.model_dir, engine_dir)
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

        engine = Engine(config=model.engine_config, engine=model.engine)
        engine.save(engine_dir)
        if rank == 0:
            copy_hf_tokenizer_data_to_engine_dir()

    @staticmethod
    def get_model_format(model_dir: str) -> _ModelFormatKind:
        ''' Get the format of the model.  '''
        # TODO: migrate to detect version field in config.json after TRTLLM-256 finished
        if Path.exists(
                Path(model_dir) / 'config.json') and file_with_glob_exists(
                    model_dir, 'rank*.safetensors'):
            return _ModelFormatKind.TLLM_CKPT
        if (Path.exists(Path(model_dir) / 'config.json')
                and (file_with_suffix_exists(model_dir, '.bin')
                     or file_with_suffix_exists(model_dir, '.safetensors'))):
            return _ModelFormatKind.HF
        if Path.exists(
                Path(model_dir) / 'config.json') and file_with_suffix_exists(
                    model_dir, '.engine'):
            return _ModelFormatKind.TLLM_ENGINE
        raise ValueError(f"Unknown model format for {model_dir}")

    def _download_hf_model(self):
        ''' Download HF model from third-party model hub like www.modelscope.cn or huggingface.  '''
        model_dir = None
        # Only the rank0 are allowed to download model
        if mpi_rank() == 0:
            assert self._workspace is not None
            assert isinstance(self.llm_args.model, str)
            # this will download only once when multiple MPI processes are running
            model_dir = download_hf_model(self.llm_args.model,
                                          revision=self.llm_args.revision)
        # Make all the processes got the same model_dir
        self._model_dir = mpi_broadcast(model_dir, root=0)
        self.llm_args.model = Path(self._model_dir)  # mark as a local model
        print_colored(f"Downloaded model to {self._model_dir}\n", 'grey')

    def _load_model_from_hf(self):
        ''' Load a TRT-LLM model from a HF model. '''
        from ..models import LLaMAForCausalLM
        assert self._model_dir is not None

        import transformers
        _pretrained_config = transformers.PretrainedConfig.from_json_file(
            os.path.join(self._model_dir, 'config.json'))

        model_arch = _pretrained_config.architectures[0]

        # TODO[chunweiy]: add more models if ready
        model_mapping = dict(
            LlamaForCausalLM=LLaMAForCausalLM,
            MixtralForCausalLM=LLaMAForCausalLM,
        )
        if model_arch not in model_mapping:
            raise KeyError(
                f"Unsupported model architecture: {model_arch}, "
                f"only {', '.join(model_mapping.keys())} are supported now.")

        model_cls = model_mapping[model_arch]

        if self.llm_args.quant_config.quant_mode.has_any_quant():
            assert self.workspace is not None
            checkpoint_dir = f"{self.workspace}/quantized-checkpoint"
            if self.rank == 0:
                model_cls.quantize(
                    self._model_dir,
                    checkpoint_dir,
                    dtype=self.llm_args.dtype,
                    mapping=self.mapping,
                    quant_config=self.llm_args.quant_config,
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

        architecture = self.pretrained_config.architecture
        assert architecture in MODEL_MAP, \
            f"Unsupported model architecture: {architecture}"
        model_cls = MODEL_MAP[architecture]
        self.model = model_cls.from_checkpoint(self._model_dir,
                                               config=self.pretrained_config)
        self._model_info = _ModelInfo.from_pretrained_config(
            self.pretrained_config)

        # load embedding sharing related options
        self.convert_checkpoint_options[
            'share_embedding_table'] = self.pretrained_config.share_embedding_table
        self.convert_checkpoint_options[
            'use_parallel_embedding'] = self.pretrained_config.use_parallel_embedding

    def _build_engine_from_inmemory_model(self):
        assert isinstance(self.llm_args.model, Module)
        self._model_info = _ModelInfo.from_module(self.model)

    def _build_engine(self):

        self.build_config.update(auto_parallel_config=self.auto_parallel_config)
        if self.auto_parallel_config.enabled:
            self.model.config.mapping.rank = self.rank
        assert self.model is not None, "model is loaded yet."

        assert isinstance(
            self.build_config,
            BuildConfig), f"build_config is not set yet: {self.build_config}"

        engine = build(self.model, self.build_config)

        self._engine_buffer = engine.engine
        self._engine_config = engine.config
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
        engine = Engine.from_dir(self._model_dir)
        self._engine_buffer = engine.engine
        self._engine_config = engine.config

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
    def load_hf_tokenizer(model_dir) -> Optional[TransformersTokenizer]:
        try:
            return TransformersTokenizer.from_pretrained(model_dir,
                                                         legacy=False,
                                                         padding_side='left',
                                                         truncation_side='left',
                                                         trust_remote_code=True,
                                                         use_fast=True)
        except:
            return None


class CachedModelLoader:
    '''
    The CacheModelLoader is used to build the model in both single or multi-gpu, with cache might be enabled.
    '''

    def __init__(
        self,
        llm_args: LlmArgs,
        llm_build_stats: "LlmBuildStats",
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

    def __call__(self) -> Path:

        if self.llm_args.model_format is _ModelFormatKind.TLLM_ENGINE:
            # do nothing for engine input
            return self.llm_args.model_dir

        self.engine_cache_stage: Optional[CachedStage] = None

        if self.build_cache_enabled:
            if self.llm_args.is_hub_model:
                # This will download the config.json from HF model hub, this helps to create a PretrainedConfig for
                # cache key.
                self._hf_model_dir = download_hf_pretrained_config(
                    self.llm_args.model, revision=self.llm_args.revision)

            elif self.llm_args.is_local_model:
                self._hf_model_dir = self.llm_args.model_dir if self.llm_args.model_format is _ModelFormatKind.HF else None

            self.engine_cache_stage = self._get_engine_cache_stage()
            if self.engine_cache_stage.cache_hitted():
                print_colored(
                    f"Reusing cached engine in {self.engine_cache_stage.get_engine_path()}\n\n",
                    'grey')
                self.llm_build_stats.cache_hitted = True
                self.llm_args.model = self.engine_cache_stage.get_engine_path()
                self.llm_build_stats.engine_dir = self.llm_args.model_dir
                return self.llm_build_stats.engine_dir

        return self._build_model()

    def get_engine_dir(self) -> Path:
        if self.llm_args.model_format is _ModelFormatKind.TLLM_ENGINE:
            return self.llm_args.model_dir

        # generate a new path for writing the engine
        if self.build_cache_enabled:
            cache_stage = self._get_engine_cache_stage()
            return cache_stage.get_engine_path()

        return self.workspace / "tmp.engine"

    @property
    def build_cache_enabled(self) -> bool:
        _enable_build_cache, _ = get_build_cache_config_from_env()

        return (self.llm_args.enable_build_cache or _enable_build_cache) and (
            self.llm_args.model_format is _ModelFormatKind.HF)

    def _get_engine_cache_stage(self) -> CachedStage:
        '''
        Get the cache stage for engine building.
        '''
        build_cache = BuildCache(self.llm_args.enable_build_cache)

        assert self._hf_model_dir is not None, "HF model dir is required for cache key."
        dummy_build_config = CachedModelLoader.get_final_build_config(
            self.llm_args, self._hf_model_dir)

        return build_cache.get_engine_building_cache_stage(
            build_config=dummy_build_config,
            model_path=self._hf_model_dir,
            # for PretrainedConfig
            parallel_config=self.llm_args.parallel_config,
            # Other configs affecting the engine building
            quant_config=self.llm_args.quant_config)

    @staticmethod
    def get_final_build_config(llm_args: LlmArgs,
                               model_dir: Path) -> BuildConfig:
        '''
        Get the build_config for cache key. The tricky part is that, the build_config will be altered in `build()`,
        but we need a final version of build_config before `build()` is called for cache key.

        Args:
            llm_args: The LlmArgs for building the model.
            model_dir: The path to the local HF model.
        '''

        # This is only needed by BuildCache for cache key
        # The build() doesn't need the real model instance to get a updated BuildConig. What is really needed is the
        # dtype. That's why the model will be downloaded from HF if necessary to get the accurate dtype.

        # TODO[chunweiy]: Support more architectures
        hf_pretrained_config = HfPretrainedConfig.from_json_file(model_dir /
                                                                 "config.json")
        if "LlamaForCausalLM" not in hf_pretrained_config.architectures:
            raise ValueError(
                f"Unsupported model architecture: {hf_pretrained_config.architectures}"
            )

        pretrained_config = LLaMAConfig.from_hugging_face(
            model_dir,
            mapping=Mapping(world_size=llm_args.parallel_config.world_size,
                            tp_size=llm_args.parallel_config.tp_size,
                            pp_size=llm_args.parallel_config.pp_size),
            quant_config=llm_args.quant_config,
            dtype=llm_args.dtype)

        @dataclass
        class DummyModel:
            # This is only used for getting the updated BuildConfig from build() without actually loading the whole
            # pretrained model to save overhead and memory.
            config: LLaMAConfig

        # dry_run to get the updated build_config for cache key.  The build_config is modified within build(), so using
        # a build_config before build() is not correct for cache key, so we need to get the build_config after build()
        # in dry_run mode.
        dummy_model = DummyModel(pretrained_config)
        dummy_build_config = copy.copy(llm_args.build_config)
        dummy_build_config.dry_run = True
        updated_build_config = build(dummy_model,
                                     dummy_build_config,
                                     return_build_config=True)
        return updated_build_config

    def _build_model(self) -> Path:
        model_format = self.llm_args.model_format

        def build_task():
            if model_format is not _ModelFormatKind.TLLM_ENGINE:

                if self.llm_args.parallel_config.is_multi_gpu:
                    assert self.mpi_session
                    # The engine_dir:Path will be stored to MPINodeState.state
                    build_infos = self.mpi_session.submit_sync(
                        CachedModelLoader._node_build_task,
                        llm_args=self.llm_args,
                        tokenizer=self.llm_args.
                        tokenizer,  # TODO[chunweiy]: Use llm_args directly
                        dtype=self.llm_args.dtype,
                        engine_dir=self.get_engine_dir())
                    self.llm_build_stats.build_steps_info = build_infos[0]

                else:  # single-gpu

                    with ModelLoader(self.llm_args,
                                     tokenizer=self.llm_args.tokenizer,
                                     workspace=self.workspace.name,
                                     llm_build_stats=self.llm_build_stats
                                     ) as model_loader:

                        model_loader(self.get_engine_dir())

                release_gc()

        if self.build_cache_enabled:
            with self.engine_cache_stage.write_guard():
                build_task()
                return self.get_engine_dir()
        else:
            build_task()

        return self.get_engine_dir()

    @print_traceback_on_error
    @staticmethod
    def _node_build_task(
        llm_args: LlmArgs,
        tokenizer: Optional[TokenizerBase] = None,
        dtype: str = 'auto',
        engine_dir: Optional[Path] = None,
    ):
        if MPINodeState.is_initialized():
            raise RuntimeError("The MPI node is already initialized.")

        with ModelLoader(
                llm_args,
                tokenizer=tokenizer,
        ) as model_loader:
            model_loader(engine_dir=engine_dir)
            return model_loader.llm_build_stats.build_steps_info

    def save(self, engine_dir: Path):
        # copy the engine directory to the target directory
        shutil.copytree(self.get_engine_dir(), engine_dir)


@dataclass
class LlmBuildStats:
    ''' LlmBuildStats is the statistics for the LLM model building. '''
    # Whether the cache is hitted for the engine
    cache_hitted: bool = False

    model_from_hf_hub: bool = False

    local_model_dir: Optional[Path] = None

    # The path to the trt-llm engine
    engine_dir: Optional[Path] = None

    # The build steps information, including the step name and the latency in seconds.
    build_steps_info: List[Tuple[str, float]] = field(default_factory=list)
