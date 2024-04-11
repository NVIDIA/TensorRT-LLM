import json
import os
import shutil
import tempfile
import time
from argparse import Namespace
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union

import tensorrt as trt
import torch

import tensorrt_llm.bindings as tllm
from tensorrt_llm.bindings import KvCacheConfig, SchedulerPolicy

from .. import bindings as tllm
from .._utils import mpi_barrier, mpi_rank, release_gc
from ..auto_parallel.config import AutoParallelConfig, infer_cluster_key
from ..bindings import KvCacheConfig, SchedulerPolicy
from ..builder import BuildConfig, Engine, EngineConfig, PluginConfig, build
from ..executor import GenerationExecutor, GenerationResult
from ..logger import logger
from ..mapping import Mapping
from ..models.modeling_utils import PretrainedConfig, QuantConfig, load_model
from ..module import Module
from .mpi_session import MPINodeState, MpiSession
from .tokenizer import TokenizerBase, TransformersTokenizer
from .utils import (GenerationOutput, SamplingConfig, file_with_glob_exists,
                    file_with_suffix_exists, get_device_count, print_colored,
                    print_traceback_on_error)


@dataclass
class ParallelConfig:
    ''' The model distribution configs for LLM.  '''
    tp_size: int = 1
    pp_size: int = 1
    world_size: int = 1
    devices: List[int] = field(default_factory=list)
    auto_parallel: bool = False

    def get_devices(self) -> List[int]:
        ''' Get the devices for the model. '''
        return self.devices if self.devices else list(range(self.tp_size))


@dataclass
class ModelConfig:

    # ``model_dir`` helps to locate a local model, the format of the model is determined by the model file itself.
    # Either HF model, TensorRT-LLM checkpoints or TensorRT-LLM engine format is supported.
    model_dir: Optional[str] = None

    # ``model`` could either the model directory or a in-memory model.
    # If ``model`` specifies the model kind like "llama-7B", etc.  The model will be download automatically from third-party
    # model hub like www.modelscope.cn or huggingface
    model: Optional[Union[str, Module]] = None

    # ``parallel_config`` is used to specify the parallelism of the model.
    parallel_config: ParallelConfig = field(default_factory=ParallelConfig)

    # ``quant_config`` is used to specify the quantization mode of the model.
    quant_config: QuantConfig = field(default_factory=QuantConfig)

    # ``max_beam_width`` specifies the maximum beam width for beam search.
    max_beam_width: int = 1

    # ``plugin_config`` overwrites the underlying plugin config. Default values will be used if it's None.
    # This is not suggested to be used directly, ideally the HLAPI will deduce all of options automatically.
    plugin_config: Union[PluginConfig, Dict[str, Any], None] = None

    @property
    def is_multi_gpu(self) -> bool:
        if self.parallel_config.auto_parallel:
            return self.parallel_config.world_size > 1
        else:
            return self.parallel_config.tp_size > 1 or self.parallel_config.pp_size > 1

    @property
    def world_size(self) -> bool:
        if self.parallel_config.auto_parallel:
            if self.parallel_config.tp_size > 1 or self.parallel_config.pp_size > 1:
                raise RuntimeError(
                    "manually TP and PP are not supported in auto parallel mode."
                )
            return self.parallel_config.world_size

        if self.parallel_config.world_size > 1:
            raise RuntimeError(
                "world_size > 1 is only supported in auto parallel mode.")
        return self.parallel_config.tp_size * self.parallel_config.pp_size

    def _set_additional_options(self,
                                max_batch_size: Optional[int] = None,
                                max_input_len: Optional[int] = None,
                                max_output_len: Optional[int] = None,
                                max_num_tokens: Optional[int] = None):
        ''' This method is used to set the additional options for the workflow, only for testing and debugging.
        Note, it is not ready for production use, and may be deprecated in the future.

        Usage:

            config = ModelConfig(<model-path>)
            # set the additional options in one time
            config._set_additional_options(max_batch_size=32, max_input_len=1024)

            # it is also safe to set the options in one by one
            config._set_additional_options(max_batch_size=32)
            config._set_additional_options(max_input_len=32)
        '''
        if max_batch_size is not None:
            self._max_batch_size = max_batch_size
        if max_input_len is not None:
            self._max_input_len = max_input_len
        if max_output_len is not None:
            self._max_output_len = max_output_len
        if max_num_tokens is not None:
            self._max_num_tokens = max_num_tokens

    @property
    def max_batch_size(self) -> int:
        return self._max_batch_size

    @property
    def max_input_len(self) -> int:
        return self._max_input_len

    @property
    def max_output_len(self) -> int:
        return self._max_output_len

    @property
    def max_num_tokens(self) -> Optional[int]:
        return self._max_num_tokens

    def __post_init__(self):
        if not self.model_dir:
            raise ValueError("model_dir is required.")

        if self.model:
            raise NotImplementedError("model is not supported yet.")

        model_path = Path(self.model_dir)
        if not model_path.exists():
            raise ValueError(
                f"model_dir of path {self.model_dir} does not exist.")

        # The additional options, they are not suggested to configure directly, the HLAPI will deduce them.
        # And they might be removed in the future.
        self._max_batch_size: int = 128
        self._max_input_len: int = 512
        self._max_output_len: int = 200
        self._max_num_tokens: Optional[int] = 4096

        self._build_config: Optional[BuildConfig] = None
        self._engine_config: Optional[EngineConfig] = None

        self.auto_parallel_config = AutoParallelConfig(
            cluster_key=infer_cluster_key(
                allow_fallback=self.parallel_config.auto_parallel),
            sharded_io_allowlist=[
                "past_key_value_\\d+",
                "present_key_value_\\d*",
            ],
            same_buffer_io={
                "past_key_value_(\\d+)": "present_key_value_\\1",
            },
        )

        # Load parallel_config from the engine.
        self.model_format = ModelLoader.get_model_format(self.model_dir)
        if self.model_format is _ModelFormatKind.TLLM_ENGINE:
            self._load_config_from_engine(Path(self.model_dir))

        # Load parallel_config from the checkpoint.
        if ModelLoader.get_model_format(
                self.model_dir) is _ModelFormatKind.TLLM_CKPT:
            self._load_config_from_ckpt(Path(self.model_dir))

    def _update_plugin_config(self, key: str, value: Any):
        if key == 'use_paged_context_fmha':
            self._validate_gpu_for_paged_context(value)

        self.plugin_config = self.plugin_config or {}
        if isinstance(self.plugin_config, PluginConfig):
            setattr(self.plugin_config, key, value)
        else:
            self.plugin_config[key] = value

    def _validate_gpu_for_paged_context(self, value: bool):
        if value:
            devices = self.parallel_config.get_devices()
            if torch.cuda.get_device_properties(devices[0]).major < 8:
                raise ValueError(
                    "Paged context is only supported on post Volta GPUs")

    def _load_config_from_engine(self, engine_dir: Path):
        with open(engine_dir / "config.json") as f:
            engine_config = json.load(f)
            for config_key in ("pretrained_config", "build_config"):
                if config_key not in engine_config:
                    raise ValueError(
                        f"Invalid engine config found from {engine_dir}, "
                        "please use the corresponding version of trtllm-build to build the engine."
                    )

            pretrained_config = PretrainedConfig.from_dict(
                engine_config["pretrained_config"])
            build_config = BuildConfig.from_dict(engine_config["build_config"])

            # load build_config
            self.max_beam_width = build_config.max_beam_width
            self._set_additional_options(
                max_batch_size=build_config.max_batch_size,
                max_input_len=build_config.max_input_len,
                max_output_len=build_config.max_output_len,
                max_num_tokens=build_config.max_num_tokens)

            # load plugin_config
            self.plugin_config = build_config.plugin_config

            # load parallel_config
            mapping = pretrained_config.mapping
            if self.parallel_config.tp_size not in (1, mapping.tp_size):
                raise ValueError(
                    f"tp_size {self.parallel_config.tp_size} is not consistent with the engine's tp_size {mapping.tp_size}"
                )
            if self.parallel_config.pp_size not in (1, mapping.pp_size):
                raise ValueError(
                    f"pp_size {self.parallel_config.pp_size} is not consistent with the engine's pp_size {mapping.pp_size}"
                )
            self.parallel_config = ParallelConfig(
                tp_size=mapping.tp_size,
                pp_size=mapping.pp_size,
            )

            self._pretrined_config = pretrained_config
            self._build_config = build_config

    def _load_config_from_ckpt(self, ckpt_dir: Path):
        with open(ckpt_dir / "config.json") as f:
            ckpt_config = json.load(f)
            tp_size = ckpt_config["mapping"]["tp_size"]
            pp_size = ckpt_config["mapping"]["pp_size"]
            world_size = ckpt_config["mapping"]["world_size"]
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
                    and self.parallel_config.world_size != 1
                    and world_size != 1):
                raise ValueError(
                    f"auto parallel with world_size {self.parallel_config.world_size} does not support checkpoint with world_size {world_size} > 1"
                )
            if not self.parallel_config.auto_parallel:
                self.parallel_config = ParallelConfig(
                    tp_size=tp_size,
                    pp_size=pp_size,
                )


@dataclass(unsafe_hash=True)
class StreamingLLMParam:
    # TODO[chunweiy]: optimize the default value
    max_attention_window_size: int = 2048
    sink_token_length: int = 4


class LLM:
    '''
    An end-to-end runner for LLM tasks.

    Classical usage:

    config = ModelConfig(<model-path>)

    llm = LLM(config)
    llm.generate(["What is your name?"]) # => ["My name is Llama."]
    '''

    def __init__(self,
                 config: ModelConfig,
                 *,
                 tokenizer: Optional[TokenizerBase] = None,
                 kv_cache_config: Optional[KvCacheConfig] = None,
                 streaming_llm: Union[bool, StreamingLLMParam] = False,
                 async_engine_tmp_dir: Optional[str] = None,
                 **_additional_options: Any):
        '''
        Args:
            config: The model config for the model.
            tokenizer: User provided tokenizer, will override the default one if exists in the HF model or TRT-LLM engine.
            kv_cache_config: The config for the paged KV cache.
            enable_streaming_llm(bool): Whether to enable the streaming LLM mode.
            async_engine_tmp_dir: The temporary directory to save the async engine. Only for debugging.
            _additional_params: Additional options for the model. These options are unstable and are not suggested to be used directly.

        The _additional_params are not suggested to be used directly, ideally the HLAPI will deduce them.  They are used for debugging and testing, and may be removed in the future.
        The options includes:
            enable_trt_overlap(bool): Whether to enable the TRT overlap for the generation.
            normalize_log_probs(bool): Whether to normalize the log probabilities.
            use_custom_all_reduce(bool): Whether to use the custom all reduce for the multi-gpu case. Default is False.
            multi_block_mode(bool): Switch the optimization on multi-head attention optimization for long context decoding.
            enable_chunked_context(bool): Whether to enable the chunked context for the generation.
            scheduling_policy(SchedulerPolicy): The scheduling policy for the generation.
        '''

        self.config = config

        self._tokenizer = tokenizer
        self.async_engine_tmp_dir = async_engine_tmp_dir
        self.kv_cache_config = kv_cache_config
        # TODO[chunweiy]: add doc for enable_streaming_llm
        self.enable_streaming_llm = streaming_llm
        if self.enable_streaming_llm is True:
            self.enable_streaming_llm = StreamingLLMParam()

        self.mpi_session = None

        plugin_config_alterable = self.config.model_format is not _ModelFormatKind.TLLM_ENGINE

        # Read the additional options
        self.normalize_log_probs = _additional_options.pop(
            'normalize_log_probs', None)
        # TODO[chunweiy]: Turn on the custom all reduce by default later
        self.use_custom_all_reduce = _additional_options.pop(
            'use_custom_all_reduce', False if plugin_config_alterable else None)
        self.multi_block_mode = _additional_options.pop('multi_block_mode',
                                                        None)
        # Chunked context is enabled by default for performance
        self.enable_chunked_context = _additional_options.pop(
            'enable_chunked_context', True if plugin_config_alterable else None)
        self.enable_trt_overlap = _additional_options.pop(
            'enable_trt_overlap', None)
        self.scheduling_policy = _additional_options.pop(
            'scheduling_policy', SchedulerPolicy.GUARANTEED_NO_EVICT)
        if _additional_options:
            raise ValueError(f"Unknown options {_additional_options}")

        devices = self.config.parallel_config.get_devices()
        if torch.cuda.get_device_properties(devices[0]).major < 8:
            logger.info(
                f"Disable the chunked context on GPUs that predate the Volta architecture."
            )
            self.enable_chunked_context = False

        if self.config.is_multi_gpu:
            if get_device_count() < self.config.world_size:
                raise RuntimeError(
                    f"Only {get_device_count()} GPUs are available, but {self.config.world_size} are required."
                )

            logger.info(
                f'start MpiSession with {self.config.world_size} workers')
            self.mpi_session = MpiSession(n_workers=self.config.world_size)

        # Due to the gptManager can only accept a engine path, we need to save the engine to a directory
        self._engine_dir: Union[tempfile.TemporaryDirectory, str, Path,
                                None] = None
        self._executor: Optional[GenerationExecutor] = None
        self._workspace = tempfile.TemporaryDirectory("llm-workspace")

        self.runtime_context: Optional[_ModelRuntimeContext] = None

        # Update the dependency config if necessary
        # When got an engine, the plugin config are fixed, shouldn't be altered.
        if self.config.model_format is not _ModelFormatKind.TLLM_ENGINE:
            if self.kv_cache_config is not None:
                if self.kv_cache_config.enable_block_reuse:
                    logger.info(
                        f"Turn on `use_paged_context_fmha` due to enable_block_reuse"
                    )
                    self.config._update_plugin_config("use_paged_context_fmha",
                                                      True)
            if self.enable_chunked_context is not None:
                self.config._update_plugin_config("enable_chunked_context",
                                                  self.enable_chunked_context)
                if self.enable_chunked_context is True:
                    self.config._update_plugin_config("use_paged_context_fmha",
                                                      True)
            if self.multi_block_mode is not None:
                self.config._update_plugin_config("multi_block_mode",
                                                  self.multi_block_mode)
            if self.use_custom_all_reduce is not None:
                self.config._update_plugin_config("use_custom_all_reduce",
                                                  self.use_custom_all_reduce)
            if self.enable_streaming_llm:
                self.config._update_plugin_config("streamingllm", True)

                self.kv_cache_config = KvCacheConfig(
                ) if self.kv_cache_config is None else self.kv_cache_config
                self.kv_cache_config.max_attention_window = self.enable_streaming_llm.max_attention_window_size
                self.kv_cache_config.sink_token_length = self.enable_streaming_llm.sink_token_length

        self._build_model()

    def generate(
        self,
        prompts: Union[Iterable[str], Iterable[List[int]]],
        sampling_config: Optional[Union[SamplingConfig,
                                        List[SamplingConfig]]] = None
    ) -> Iterable[GenerationOutput]:
        ''' Generate the output for the given inputs.

        Args:
            prompts: The raw text or token ids to the model.
            sampling_config: The sampling config for the generation, a default one will be used if not provided.
        '''
        prompts = list(prompts)

        if sampling_config is None:
            sampling_config = self.get_default_sampling_config()

        results = self._executor.generate(
            prompts,
            sampling_config=sampling_config,
        )

        return results

    def generate_async(self,
                       prompt: Union[str, List[int]],
                       sampling_config: Optional[SamplingConfig] = None,
                       streaming: bool = False) -> GenerationResult:
        ''' Generate in asynchronuous mode.

        Args:
            prompt: The raw text or token ids to the model.
            sampling_config: The sampling config for the generation, a default one will be used if not provided.
            streaming: Whether to use the streaming mode for the generation.
        '''
        if sampling_config is None:
            sampling_config = self.get_default_sampling_config()
        self._generate_check_arguments([prompt], sampling_config)

        results = self._executor.generate_async(prompt,
                                                streaming=streaming,
                                                sampling_config=sampling_config)
        return results

    def _generate_check_arguments(self, prompts,
                                  sampling_config: SamplingConfig):
        if sampling_config is None:
            raise ValueError("The sampling_config should to be provided.")
        if sampling_config.top_k is not None or sampling_config.top_p is not None:
            raise ValueError("The top_k and top_p are not supported yet.")

        sampling_configs = [sampling_config] if isinstance(
            sampling_config, SamplingConfig) else sampling_config
        max_num_beams = max([sc.beam_width for sc in sampling_configs])
        if max_num_beams > self.config.max_beam_width:
            raise ValueError(
                f"num_beams is larger than the maximum in the built engine {max_num_beams} > {self.config.max_beam_width}"
            )
        if len(prompts) > self.config.max_batch_size:
            raise ValueError(
                f"Batch size {len(prompts)} is larger than the maximum in the built engine {self.config.max_batch_size}"
            )

        input_digits = False
        if isinstance(prompts[0], list):
            input_digits = True
        if input_digits and sum(
                len(prompt) for prompt in prompts) > self.config.max_num_tokens:
            raise ValueError(f"The total input length is too large")
        if not input_digits:
            if max(len(prompt.split())
                   for prompt in prompts) > self.config.max_input_len:
                raise ValueError(
                    f"Input length is larger than the maximum in the built engine"
                )

    @property
    def tokenizer(self) -> TokenizerBase:
        if self._tokenizer is not None:
            return self._tokenizer
        if self.runtime_context is not None:
            return self.runtime_context.tokenizer

    def save(self, engine_dir: str):
        ''' Save the built engine to the given path. '''
        logger.info(f"Save model to {engine_dir}")
        if self._engine_dir is None:
            raise RuntimeError("The engine is not built yet.")
        src_engine_dir = self._engine_dir.name if isinstance(
            self._engine_dir, tempfile.TemporaryDirectory) else self._engine_dir
        if src_engine_dir != engine_dir:
            shutil.copytree(src_engine_dir, engine_dir, dirs_exist_ok=True)

    def shutdown(self):
        if self._executor is not None:
            self._executor.shutdown()

        if self.mpi_session is not None:
            self.mpi_session.shutdown()
            self.mpi_session = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        del exc_value, traceback
        self.shutdown()
        return exc_type is not None

    def _save_engine(self, engine_dir: str):
        logger.info(f"Save model to {engine_dir}")

        if self.config.is_multi_gpu:
            if self._executor is not None:
                self._executor.shutdown()
            self.mpi_session.submit_sync(LLM._node_save_task, engine_dir,
                                         self.config.model_dir)
        else:
            ModelLoader.save(self.runtime_context,
                             self.config.model_dir,
                             engine_dir=engine_dir,
                             model_info=self.runtime_context.model_info)

    def get_default_sampling_config(self) -> Optional[SamplingConfig]:
        ''' Get the default sampling config for the model.
        You can override the options.
        '''
        tokenizer = self.tokenizer
        if tokenizer is None:
            try:
                tokenizer = ModelLoader.load_hf_tokenizer(self.config.model_dir)
            except:
                return None

        return SamplingConfig(
            end_id=tokenizer.eos_token_id,
            pad_id=tokenizer.eos_token_id
            if tokenizer.pad_token_id is None else tokenizer.pad_token_id,
        )

    def _build_model(self):
        model_format = ModelLoader.get_model_format(self.config.model_dir)

        self._engine_dir = self.config.model_dir

        def get_engine_dir():
            return self._engine_dir.name if isinstance(
                self._engine_dir,
                tempfile.TemporaryDirectory) else self._engine_dir

        if model_format is not _ModelFormatKind.TLLM_ENGINE:

            if self._executor is not None:
                self._executor.shutdown()

            self._engine_dir = self.async_engine_tmp_dir
            if self._engine_dir is None:
                self._engine_dir = tempfile.TemporaryDirectory()

            if self.config.is_multi_gpu:
                self.mpi_session.submit_sync(
                    LLM._node_build_task,
                    self.config,
                    self._tokenizer,
                    self._workspace.name,
                )
                self._save_engine(get_engine_dir())

                self.mpi_session.submit_sync(LLM._node_free_state_task)

            else:

                with ModelLoader(
                        self.config,
                        tokenizer=self._tokenizer,
                        workspace=self._workspace.name,
                ) as model_loader:

                    runtime_context = model_loader()

                    # TODO[chunweiy]: Make GptManager support in-memory engine-buffer to save disk loading latency
                    ModelLoader.save(runtime_context,
                                     self.config.model_dir,
                                     engine_dir=get_engine_dir(),
                                     model_info=runtime_context.model_info)

                    # Once saved, the engine_buffer is not needed anymore
                    del runtime_context

            release_gc()

        tokenizer = self.tokenizer
        if not isinstance(tokenizer, TokenizerBase):
            tokenizer = ModelLoader.load_hf_tokenizer(self.config.model_dir)

        executor_config = tllm.TrtGptModelOptionalParams()
        if self.kv_cache_config is not None:
            executor_config.kv_cache_config = self.kv_cache_config
        executor_config.enable_trt_overlap = self.enable_trt_overlap
        executor_config.normalize_log_probs = self.normalize_log_probs
        executor_config.enable_chunked_context = self.enable_chunked_context

        self._executor = GenerationExecutor.create(
            get_engine_dir(),
            tokenizer,
            max_beam_width=self.config.max_beam_width,
            executor_config=executor_config,
            executor_policy=self.scheduling_policy,
            model_world_size=self.config.world_size,
            mpi_session=self.mpi_session)

    @print_traceback_on_error
    @staticmethod
    def _node_build_task(config: ModelConfig,
                         tokenizer: Optional[TokenizerBase] = None,
                         workspace: Optional[str] = None) -> bool:
        if MPINodeState.is_initialized():
            raise RuntimeError("The MPI node is already initialized.")

        with ModelLoader(config, tokenizer=tokenizer,
                         workspace=workspace) as model_loader:
            runtime_context = model_loader()

        # Hold the model builder for later use
        MPINodeState.state = runtime_context
        return True

    @print_traceback_on_error
    @staticmethod
    def _node_save_task(engine_dir: str, model_dir: str):
        runtime_context: _ModelRuntimeContext = MPINodeState.state
        if not isinstance(runtime_context, _ModelRuntimeContext):
            raise RuntimeError("Model is not built yet.")

        ModelLoader.save(runtime_context,
                         model_dir,
                         engine_dir=engine_dir,
                         model_info=runtime_context.model_info)

    @print_traceback_on_error
    @staticmethod
    def _node_free_state_task():
        MPINodeState.state = None
        # release the large resource explicitly and immediately, since the following LLM pipeline may need a lot of memory
        release_gc()

    def __getstate__(self):
        raise RuntimeError("LLM object can not be pickled.")

    def __del__(self):
        self.shutdown()


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

    @property
    def engine(self) -> trt.IHostMemory:
        assert self.engine_buffer is not None
        return self.engine_buffer

    @property
    def model_structure(self) -> str:
        # "LlaMACausalForLM" or "OPTForCausalLM" and so on
        return self.engine_config.pretrained_config['architecture']


class ModelLoader:
    ''' The ModelLoader is used to build an end-to-end model from a model config.
    It will construct the runtime resources including engine, tokenizer, model runner etc for a single gpu.
    '''

    def __init__(self,
                 config: ModelConfig,
                 tokenizer: Optional[TokenizerBase],
                 workspace: Optional[str] = None):
        self.config = config
        self.tokenizer = tokenizer
        self.workspace = workspace
        self.rank = mpi_rank() if config.is_multi_gpu else 0
        if config.is_multi_gpu and not config.parallel_config.auto_parallel:
            self.mapping = Mapping(
                tp_size=config.parallel_config.tp_size,
                pp_size=config.parallel_config.pp_size,
                rank=self.rank,
                world_size=config.world_size,
            )
        else:
            self.mapping = Mapping()

        self._model_pipeline = []

        self._model_dir = self.config.model_dir
        self._model_info: Optional[_ModelInfo] = None
        self._model_name = self.config.model
        self.auto_parallel_config = AutoParallelConfig(
            world_size=config.parallel_config.world_size)
        default_config = self.config.auto_parallel_config
        self.auto_parallel_config.set_defaults(
            cluster_key=default_config.cluster_key,
            same_buffer_io=default_config.same_buffer_io,
            sharded_io_allowlist=default_config.sharded_io_allowlist,
        )

        # Prepare the model processing pipeline
        if isinstance(self.config.model, Module):
            ''' Build engine from user provided model '''
            self._model_pipeline.append(
                ("Build TensorRT-LLM engine",
                 self._build_engine_from_inmemory_model))
            return

        if self.config.model_dir is None:
            ''' Download HF model if necessary '''
            # TODO[chunweiy]: Support HF model download
            raise NotImplementedError()

        if self._model_dir is None:
            raise ValueError("The model_dir is not set yet.")
        self._model_format = ModelLoader.get_model_format(self._model_dir)

        if self._model_format is _ModelFormatKind.HF:
            ''' HF -> TRT checkpoints -> engine '''
            self._model_pipeline.append(
                ("Load HF model to memory", self._load_model_from_hf))
            self._model_pipeline.append(
                ("Build TRT-LLM engine", self._build_engine))
        elif self._model_format is _ModelFormatKind.TLLM_CKPT:
            ''' TRT checkpoints -> engine '''
            self._model_pipeline.append(
                ("Load TRT checkpoints to memory", self._load_model_from_ckpt))
            self._model_pipeline.append(
                ("Build TRT-LLM engine", self._build_engine))
        elif self._model_format is _ModelFormatKind.TLLM_ENGINE:
            ''' TFRT engine '''
            self._model_pipeline.append(
                ("Load TensorRT-LLM engine", self._load_engine_buffer))
        else:
            raise ValueError(f"Unknown model format {self._model_format}")

        if self.tokenizer is None:
            ''' Use the default tokenizer if no one is provided. '''
            self._model_pipeline.append(
                ("Initialize tokenizer", self._load_hf_tokenizer))

    def __call__(self) -> _ModelRuntimeContext:
        if self.config.is_multi_gpu:
            torch.cuda.set_device(self.rank)

        n_steps = len(self._model_pipeline)
        to_log = self.rank == 0

        overall_start_time = time.time()
        for off, (info, step) in enumerate(self._model_pipeline):
            if to_log:
                print_colored("Loading Model: ")
                print_colored(f"[{off+1}/{n_steps}]\t", 'bold_green')
                print_colored(f"{info}\n")

            start_time = time.time()
            step()
            latency = time.time() - start_time
            if to_log:
                print_colored("Time: {:.3f}s\n".format(latency), 'grey')

        overall_latency = time.time() - overall_start_time
        if to_log:
            print_colored("Loading model done.\n", 'bold_green')
            print_colored('Total latency: {:.3f}s\n'.format(overall_latency),
                          'grey')

        if self._engine_buffer is None:
            raise RuntimeError("The engine is not built yet.")

        if not hasattr(self, '_engine_config'):
            raise RuntimeError("config is not loaded.")

        config = self._engine_config

        return _ModelRuntimeContext(
            tokenizer=self.tokenizer,
            engine_buffer=self._engine_buffer,
            engine_config=config,
            mapping=self.mapping,
            model_info=self._model_info,
        )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        for attr_name in dir(self):
            if not callable(getattr(
                    self, attr_name)) and not attr_name.startswith("__"):
                if attr_name not in ('model_format', ):
                    setattr(self, attr_name, None)

        release_gc()

    @property
    def model_format(self) -> _ModelFormatKind:
        return self._model_format

    # TODO[tali]: Replace this with a lower-level API
    @staticmethod
    def save(
        model: _ModelRuntimeContext,
        model_dir: str,
        engine_dir: str,
        model_info: _ModelInfo,
    ):
        ''' Save the built engine on a single GPU to the given path. '''
        mapping = model.mapping
        rank = mapping.rank

        def copy_hf_tokenizer_data_to_engine_dir():
            # Copy the HF tokenizer stuff to the engine dir so that we can use the engine dir as a standalone model dir supports end-to-end task.
            # This is only for HF model for now, not available for users' customized tokenizers.
            import shutil
            for name in os.listdir(model_dir):
                src = os.path.join(model_dir, name)
                dst = os.path.join(engine_dir, name)
                if name.startswith('tokenizer'):
                    if os.path.isdir(src):
                        shutil.copytree(src, dst, dirs_exist_ok=True)
                    else:
                        shutil.copy2(src, dst)

        engine = Engine(config=model.engine_config, engine=model.engine)
        engine.save(engine_dir)
        if rank == 0 and isinstance(model.tokenizer, TransformersTokenizer):
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
        raise NotImplementedError()

    def _load_model_from_hf(self):
        ''' Load a TRT-LLM model from a HF model. '''
        from ..models import LLaMAForCausalLM
        assert self._model_dir is not None

        import transformers
        _pretrained_config = transformers.PretrainedConfig.from_json_file(
            os.path.join(self._model_dir, 'config.json'))

        model_arch = _pretrained_config.architectures[0]

        # TODO[chunweiy]: add more models if ready
        model2struct = dict(
            LlamaForCausalLM=LLaMAForCausalLM,
            MixtralForCausalLM=LLaMAForCausalLM,
        )
        if model_arch not in model2struct:
            raise KeyError(
                f"Unsupported model architecture: {model_arch}, "
                f"only {', '.join(model2struct.keys())} are supported now.")

        if self.config.quant_config.quant_mode.has_any_quant():
            assert self.workspace is not None
            checkpoint_dir = f"{self.workspace}/quantized-checkpoint"
            if self.rank == 0:
                model2struct[model_arch].quantize(
                    self._model_dir,
                    checkpoint_dir,
                    self.config.quant_config,
                    mapping=self.mapping,
                )
            if self.config.is_multi_gpu:
                mpi_barrier()
            self.model = model2struct[model_arch].from_checkpoint(
                checkpoint_dir, rank=self.mapping.rank)
        else:
            self.model = model2struct[model_arch].from_hugging_face(
                self._model_dir,
                mapping=self.mapping,
                quantization=self.config.quant_config,
                load_model_on_cpu=
                True,  # TODO:TRTLLM-195 to enhance the weights loading memory usage and chose best location
            )

        self.pretrained_config = self.model.config
        self._model_info = _ModelInfo.from_pretrained_config(
            self.pretrained_config)

    def _load_model_from_ckpt(self):
        ''' Load a TRT-LLM model from checkpoint. '''
        model_config = PretrainedConfig.from_json_file(
            os.path.join(self._model_dir, 'config.json'))
        model_config.mapping = self.mapping
        self.model = load_model(model_config, self._model_dir)
        self.pretrained_config = model_config
        self._model_info = _ModelInfo.from_pretrained_config(
            self.pretrained_config)

    def _build_engine_from_inmemory_model(self):
        assert isinstance(self.config.model, Module)
        self._model_info = _ModelInfo.from_module(self.model)

    def _build_engine(self):
        plugin_config = self.config.plugin_config
        if not isinstance(self.config.plugin_config, PluginConfig):
            plugin_config = self.model.default_plugin_config()
            # patch the additional options
            if self.config.plugin_config is not None:
                assert isinstance(self.config.plugin_config, dict)
                for k, v in self.config.plugin_config.items():
                    setattr(plugin_config, k, v)

        build_config = BuildConfig(
            max_input_len=self.config.max_input_len,
            max_output_len=self.config.max_output_len,
            max_batch_size=self.config.max_batch_size,
            max_beam_width=self.config.max_beam_width,
            max_num_tokens=self.config.max_num_tokens,
            strongly_typed=True,
            auto_parallel_config=self.auto_parallel_config,
            plugin_config=plugin_config,
        )
        if self.auto_parallel_config.enabled:
            self.model.config.mapping.rank = self.rank
        engine = build(self.model, build_config)

        self._engine_buffer = engine.engine
        self._engine_config = engine.config
        self.mapping = self.model.config.mapping

        # delete the model explicitly to free all the build-time resources
        self.model = None

    def _load_engine_buffer(self):
        # Load engine buffer from disk
        engine = Engine.from_dir(self._model_dir)
        self._engine_buffer = engine.engine
        self._engine_config = engine.config

    def _load_hf_tokenizer(self):
        if self._model_dir:
            self.tokenizer = ModelLoader.load_hf_tokenizer(self._model_dir)
            if self.tokenizer is None:
                logger.warning(
                    f"failed to load HuggingFace tokenizer from {self._model_dir}\n"
                    "You can also try to copy the tokenizer* files from HuggingFace model to the engine directory manually."
                )

    @staticmethod
    def load_extra_build_configs_from_engine(
            model_dir: str) -> Optional[Namespace]:
        ''' Load the extra build configs from the engine directory, return None if model isn't an engine. '''
        if ModelLoader.get_model_format(
                model_dir) is not _ModelFormatKind.TLLM_ENGINE:
            return None

        with open(Path(model_dir) / "config.json", "r") as f:
            engine_config = json.load(f)

        # TODO[chunweiy]: Remove the following if-check after the engine config is unified.
        if 'build_config' not in engine_config:
            return None
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
