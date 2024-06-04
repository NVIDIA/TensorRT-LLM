import json
import os
import shutil
import tempfile
import time
from argparse import Namespace
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, List, Optional, Union

import tensorrt as trt
import torch

from .. import bindings as tllm
from .._utils import mpi_barrier, mpi_rank, release_gc
from ..auto_parallel import AutoParallelConfig, infer_cluster_config
from ..bindings import KvCacheConfig
from ..bindings.executor import CapacitySchedulerPolicy
from ..builder import BuildConfig, Engine, EngineConfig, build
from ..executor import GenerationExecutor, GenerationResult
from ..logger import logger
from ..mapping import Mapping
from ..models import MODEL_MAP
from ..models.modeling_utils import PretrainedConfig, QuantAlgo, QuantConfig
from ..module import Module
from .mpi_session import (MpiCommSession, MPINodeState, MpiPoolSession,
                          MpiSession, external_mpi_comm_available)
from .tokenizer import TokenizerBase, TransformersTokenizer
from .utils import (GenerationOutput, GpuArch, OutputConfig, SamplingConfig,
                    download_hf_model, file_with_glob_exists,
                    file_with_suffix_exists, get_device_count, init_log_level,
                    print_colored, print_traceback_on_error)

init_log_level(
)  # This should be called before importing the following cpp-runtime modules

from ..bindings.executor import CapacitySchedulerPolicy
from ..builder import BuildConfig, Engine, EngineConfig, build
from ..executor import GenerationExecutor, GenerationResult


@dataclass
class ParallelConfig:
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
                f"world_size {world_size} should be equal to tp_size * pp_size {self.tp_size * self.pp_size} in non-auto_parallel mode.\n"
                "For non-auto-parallel mode, the world_size is not needed to set"
            )

    @property
    def is_multi_gpu(self) -> bool:
        return self.world_size > 1


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

    # ``build_config`` is used to specify the build options of the model.
    build_config: BuildConfig = field(
        default_factory=lambda: BuildConfig(max_num_tokens=1024),
        init=False,
        repr=False)

    def __post_init__(self):
        if not (self.model_dir or self.model):
            raise ValueError("Either model_dir or model should be provided.")
        if self.model_dir and self.model:
            raise ValueError(
                "Only one of model_dir or model should be provided, provided both."
            )

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

        if self.model_dir:
            model_path = Path(self.model_dir)
            if not model_path.exists():
                raise ValueError(
                    f"model_dir of path {self.model_dir} does not exist.")

            # Load parallel_config from the engine.
            self.model_format = ModelLoader.get_model_format(self.model_dir)
            if self.model_format is _ModelFormatKind.TLLM_ENGINE:
                self._load_config_from_engine(Path(self.model_dir))

            # Load parallel_config from the checkpoint.
            if self.model_format is _ModelFormatKind.TLLM_CKPT:
                self._load_config_from_ckpt(Path(self.model_dir))
        else:
            self.model_format = _ModelFormatKind.HF

    def _update_plugin_config(self, key: str, value: Any):
        if key == 'use_paged_context_fmha':
            self._validate_gpu_for_paged_context(value)

        setattr(self.build_config.plugin_config, key, value)

    def _validate_gpu_for_paged_context(self, value: bool):
        if value:
            devices = self.parallel_config.devices
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
            self.build_config = BuildConfig.from_dict(
                engine_config["build_config"])

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
                 dtype: str = 'auto',
                 kv_cache_config: Optional[KvCacheConfig] = None,
                 streaming_llm: Union[bool, StreamingLLMParam] = False,
                 async_engine_tmp_dir: Optional[str] = None,
                 **_additional_options: Any):
        '''
        Args:
            config (ModelConfig):
                The model config for the model.
            tokenizer (TokenizerBase):
                User provided tokenizer, will override the default one if exists in the HF model or TRT-LLM engine.
            dtype (str):
                The data type for the model weights and activations (non-quantized). You can
                (1) explicitly specify `float16`, `bfloat16` or `float32`; or
                (2) implicitly specify `auto` (default), then `dtype` will be automatically inferred from the source model. However, if the source `dtype` is `float32`, will use `float16` instead.
            kv_cache_config (KvCacheConfig):
                The config for the paged KV cache.
            streaming_llm (bool, StreamingLLMParam):
                Whether to enable the streaming LLM mode.
            async_engine_tmp_dir (str):
                The temporary directory to save the async engine. Only for debugging.
            _additional_params:
                Additional options for the model. These options are unstable and are not suggested to be used directly.

        The _additional_params are not suggested to be used directly, ideally the HLAPI will deduce them.  They are used for debugging and testing, and may be removed in the future.
        The options includes:
            normalize_log_probs (bool):
                Whether to normalize the log probabilities.
            enable_chunked_context (bool):
                Whether to enable the chunked context for the generation.
            capacity_scheduling_policy (CapacitySchedulerPolicy)
                The capacity scheduling policy for the generation.
            embedding_parallel_mode (str):
                The tensor parallelism mode for embedding module(s).
                'NONE' means parallelim disabled;
                'SHARDING_ALONG_VOCAB' means parallelism enabled with lookup table weight sharded along the vocab dimension;
                'SHARDING_ALONG_HIDDEN' means parallelism enabled with lookup table weight sharded along the hidden dimension.
            share_embedding_table (bool):
                Whether to share the weight between token embedding lookup table and lm_head.
        '''

        self.config = config

        self._tokenizer = tokenizer
        self.dtype = dtype
        self.async_engine_tmp_dir = async_engine_tmp_dir
        self.kv_cache_config = kv_cache_config
        # TODO[chunweiy]: add doc for enable_streaming_llm
        self.enable_streaming_llm = streaming_llm
        if self.enable_streaming_llm is True:
            self.enable_streaming_llm = StreamingLLMParam()

        self.mpi_session: Optional[MpiSession] = None

        plugin_config_alterable = self.config.model_format is not _ModelFormatKind.TLLM_ENGINE

        # Read the additional options
        self.normalize_log_probs = _additional_options.pop(
            'normalize_log_probs', True)
        # Chunked context is enabled by default for performance
        self.enable_chunked_context = _additional_options.pop(
            'enable_chunked_context', True if plugin_config_alterable else None)
        self.capacity_scheduling_policy = _additional_options.pop(
            'capacity_scheduling_policy',
            CapacitySchedulerPolicy.GUARANTEED_NO_EVICT)
        self.context_chunking_policy = _additional_options.pop(
            'context_chunking_policy', None)

        self._convert_checkpoint_options = {}
        # TODO: Move these options to ParallelConfig
        embedding_parallel_mode = _additional_options.pop(
            'embedding_parallel_mode', 'SHARDING_ALONG_VOCAB')
        if embedding_parallel_mode == 'NONE':
            self._convert_checkpoint_options['use_parallel_embedding'] = False
        elif embedding_parallel_mode == 'SHARDING_ALONG_VOCAB':
            self._convert_checkpoint_options['use_parallel_embedding'] = True
            self._convert_checkpoint_options['embedding_sharding_dim'] = 0
        elif embedding_parallel_mode == 'SHARDING_ALONG_HIDDEN':
            self._convert_checkpoint_options['use_parallel_embedding'] = True
            self._convert_checkpoint_options['embedding_sharding_dim'] = 1
        else:
            raise ValueError(
                f"Invalid embedding_parallel_mode: {embedding_parallel_mode}")
        self._convert_checkpoint_options[
            'share_embedding_table'] = _additional_options.pop(
                'share_embedding_table', False)

        if _additional_options:
            raise ValueError(f"Unknown options {_additional_options}")

        self.config.parallel_config.devices
        if not GpuArch.is_post_ampere():
            logger.info(
                f"Disable the chunked context on GPUs that predate the Volta architecture."
            )
            self.enable_chunked_context = False

        if self.config.parallel_config.is_multi_gpu:
            if get_device_count() < self.config.parallel_config.world_size:
                raise RuntimeError(
                    f"Only {get_device_count()} GPUs are available, but {self.config.parallel_config.world_size} are required."
                )

            logger.info(
                f'start MpiSession with {self.config.parallel_config.world_size} workers'
            )
            if not external_mpi_comm_available(
                    self.config.parallel_config.world_size):
                self.mpi_session = MpiPoolSession(
                    n_workers=self.config.parallel_config.world_size)
            else:
                self.mpi_session = MpiCommSession(
                    n_workers=self.config.parallel_config.world_size)

        # Due to the gptManager can only accept a engine path, we need to save the engine to a directory
        self._engine_dir: Union[tempfile.TemporaryDirectory, str, Path,
                                None] = None
        self._executor: Optional[GenerationExecutor] = None
        self._workspace = tempfile.TemporaryDirectory("llm-workspace")

        self.runtime_context: Optional[_ModelRuntimeContext] = None

        # Update the dependency config if necessary
        # When got an engine, the plugin config are fixed, shouldn't be altered.
        # TODO[chunweiy]: Refine the rules here and make them easy to be updated through versions
        # TODO[chunweiy]: Deal with the rules those depend on each other

        if self.config.model_format is not _ModelFormatKind.TLLM_ENGINE:
            if self.enable_streaming_llm:
                self.config._update_plugin_config("streamingllm", True)

                self.kv_cache_config = KvCacheConfig(
                ) if self.kv_cache_config is None else self.kv_cache_config
                self.kv_cache_config.max_attention_window = self.enable_streaming_llm.max_attention_window_size
                self.kv_cache_config.sink_token_length = self.enable_streaming_llm.sink_token_length

                # Turn off the conflict perf-optim strategies
                if self.kv_cache_config.enable_block_reuse:
                    logger.warning(
                        f"Disable KvCacheConfig.enable_block_reuse since it is conflict with StreamingLLM feature."
                    )
                    self.kv_cache_config.enable_block_reuse = False

                if self.enable_chunked_context:
                    logger.warning(
                        f"Disable Chunked Context since it is conflict with StreamingLLM feature."
                    )
                    self.enable_chunked_context = False

                self.config._update_plugin_config("use_paged_context_fmha",
                                                  False)

            if self.kv_cache_config is not None:
                if (not GpuArch.is_post_ampere()
                    ) and self.kv_cache_config.enable_block_reuse:
                    logger.warning(
                        f"Disable KvCacheConfig.enable_block_reuse since it is only supported on GPUs that postdate the Ampere architecture."
                    )
                    self.kv_cache_config.enable_block_reuse = False

                if self.kv_cache_config.enable_block_reuse:
                    if GpuArch.is_post_volta():
                        logger.info(
                            f"Turn on `use_paged_context_fmha` due to enable_block_reuse"
                        )
                        self.config._update_plugin_config(
                            "use_paged_context_fmha", True)
            if self.config.quant_config.quant_algo is QuantAlgo.FP8:
                self.enable_chunked_context = False
                self.config._update_plugin_config("use_paged_context_fmha",
                                                  False)
            if self.enable_chunked_context is not None:
                if self.enable_chunked_context is True:
                    assert GpuArch.is_post_ampere()
                    self.config._update_plugin_config("use_paged_context_fmha",
                                                      True)

        self._build_model()

    def generate(
        self,
        prompts: Union[Iterable[str], Iterable[List[int]]],
        sampling_config: Optional[Union[SamplingConfig,
                                        List[SamplingConfig]]] = None,
        output_config: Optional[OutputConfig] = None,
        bad_words: Optional[List[List[int]]] = None,
        stop_words: Optional[List[List[int]]] = None,
    ) -> Iterable[GenerationOutput]:
        ''' Generate the output for the given inputs.

        Args:
            prompts: The raw text or token ids to the model.
            sampling_config: The sampling config for the generation, a default one will be used if not provided.
        '''
        prompts = list(prompts)

        if sampling_config is None:
            sampling_config = self.get_default_sampling_config()
        else:
            _sampling_config = sampling_config if isinstance(
                sampling_config, list) else [sampling_config]
            for sc in _sampling_config:
                if sc.end_id is None:
                    if self.tokenizer is None:
                        raise ValueError(
                            f"end_id is required in the sampling_config if tokenizer is not provided."
                        )
                    sc.end_id = self.tokenizer.eos_token_id
                    sc.pad_id = self.tokenizer.eos_token_id

        output_config = output_config or OutputConfig()

        results = self._executor.generate(
            prompts,
            sampling_config=sampling_config,
            output_config=output_config,
            bad_words=bad_words,
            stop_words=stop_words,
        )

        return results

    def generate_async(
            self,
            prompt: Union[str, List[int]],
            sampling_config: Optional[SamplingConfig] = None,
            output_config: Optional[OutputConfig] = None,
            streaming: bool = False,
            bad_words: Optional[List[int]] = None,
            stop_words: Optional[List[int]] = None) -> GenerationResult:
        ''' Generate in asynchronuous mode.

        Args:
            prompt: The raw text or token ids to the model.
            sampling_config: The sampling config for the generation, a default one will be used if not provided.
            streaming: Whether to use the streaming mode for the generation.
        '''
        if sampling_config is None:
            sampling_config = self.get_default_sampling_config()
        self._generate_check_arguments([prompt], sampling_config)

        output_config = output_config or OutputConfig()

        results = self._executor.generate_async(
            prompt,
            streaming=streaming,
            sampling_config=sampling_config,
            output_config=output_config,
            stop_words=[stop_words] if stop_words is not None else None,
            bad_words=[bad_words] if bad_words is not None else None,
        )
        return results

    def _generate_check_arguments(self, prompts,
                                  sampling_config: SamplingConfig):
        if sampling_config is None:
            raise ValueError("The sampling_config should to be provided.")
        if sampling_config.top_k is not None or sampling_config.top_p is not None:
            raise ValueError("The top_k and top_p are not supported yet.")

        build_config = self.config.build_config

        sampling_configs = [sampling_config] if isinstance(
            sampling_config, SamplingConfig) else sampling_config
        max_num_beams = max([sc.beam_width for sc in sampling_configs])
        if max_num_beams > build_config.max_beam_width:
            raise ValueError(
                f"num_beams is larger than the maximum in the built engine {max_num_beams} > {build_config.max_beam_width}"
            )
        if len(prompts) > build_config.max_batch_size:
            raise ValueError(
                f"Batch size {len(prompts)} is larger than the maximum in the built engine {build_config.max_batch_size}"
            )

        input_digits = False
        if isinstance(prompts[0], list):
            input_digits = True
        if input_digits and sum(
                len(prompt)
                for prompt in prompts) > build_config.max_num_tokens:
            raise ValueError(f"The total input length is too large")
        if not input_digits:
            if max(len(prompt.split())
                   for prompt in prompts) > build_config.max_input_len:
                raise ValueError(
                    f"Input length is larger than the maximum in the built engine"
                )

    @property
    def tokenizer(self) -> TokenizerBase:
        if self._tokenizer is not None:
            return self._tokenizer
        if self.runtime_context is not None:
            return self.runtime_context.tokenizer

        try:
            self._tokenizer = ModelLoader.load_hf_tokenizer(
                self.config.model_dir)
        except:
            pass

        return self._tokenizer

    def save(self, engine_dir: str):
        ''' Save the built engine to the given path. '''
        logger.info(f"Save model to {engine_dir}")
        if self._engine_dir is None:
            raise RuntimeError("The engine is not built yet.")
        src_engine_dir = self._engine_dir.name if isinstance(
            self._engine_dir, tempfile.TemporaryDirectory) else self._engine_dir

        if os.path.abspath(src_engine_dir) != os.path.abspath(engine_dir):
            shutil.copytree(src_engine_dir, engine_dir, dirs_exist_ok=True)

    def shutdown(self):
        if hasattr(self, "_executor") and self._executor is not None:
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

        if self.config.parallel_config.is_multi_gpu:
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
        model_format = self.config.model_format
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

            if self.config.parallel_config.is_multi_gpu:
                self.mpi_session.submit_sync(
                    LLM._node_build_task,
                    self.config,
                    self._tokenizer,
                    self.dtype,
                    self._workspace.name,
                    build_config=self.config.build_config,
                    convert_checkpoint_options=self._convert_checkpoint_options,
                )
                self._save_engine(get_engine_dir())

                self.mpi_session.submit_sync(LLM._node_free_state_task)

            else:

                with ModelLoader(
                        self.config,
                        tokenizer=self._tokenizer,
                        dtype=self.dtype,
                        workspace=self._workspace.name,
                        build_config=self.config.build_config,
                        convert_checkpoint_options=self.
                        _convert_checkpoint_options,
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
        executor_config.normalize_log_probs = self.normalize_log_probs
        executor_config.enable_chunked_context = self.enable_chunked_context
        self._executor = GenerationExecutor.create(
            get_engine_dir(),
            tokenizer,
            max_beam_width=self.config.build_config.max_beam_width,
            executor_config=executor_config,
            scheduler_config=tllm.executor.SchedulerConfig(
                self.capacity_scheduling_policy, self.context_chunking_policy),
            model_world_size=self.config.parallel_config.world_size,
            mpi_session=self.mpi_session,
            executor_type=tllm.TrtGptModelType.InflightFusedBatching,
            reuse_mpi_comm=external_mpi_comm_available(
                self.config.parallel_config.world_size))

    @print_traceback_on_error
    @staticmethod
    def _node_build_task(
            config: ModelConfig,
            tokenizer: Optional[TokenizerBase] = None,
            dtype: str = 'auto',
            workspace: Optional[str] = None,
            build_config: Optional[BuildConfig] = None,
            convert_checkpoint_options: Optional[dict] = None) -> bool:
        if MPINodeState.is_initialized():
            raise RuntimeError("The MPI node is already initialized.")

        with ModelLoader(config,
                         tokenizer=tokenizer,
                         dtype=dtype,
                         workspace=workspace,
                         build_config=build_config,
                         convert_checkpoint_options=convert_checkpoint_options
                         ) as model_loader:
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
                 dtype: str = 'auto',
                 workspace: Optional[str] = None,
                 build_config: Optional[BuildConfig] = None,
                 convert_checkpoint_options: Optional[dict] = None):
        self.config = config
        self.tokenizer = tokenizer
        self.dtype = dtype
        self.workspace = workspace

        assert build_config
        self.build_config = build_config

        self.convert_checkpoint_options = {} if convert_checkpoint_options is None else convert_checkpoint_options
        self.rank = mpi_rank() if config.parallel_config.is_multi_gpu else 0
        if config.parallel_config.is_multi_gpu and not config.parallel_config.auto_parallel:
            self.mapping = Mapping(
                tp_size=config.parallel_config.tp_size,
                pp_size=config.parallel_config.pp_size,
                rank=self.rank,
                world_size=config.parallel_config.world_size,
            )
        else:
            self.mapping = Mapping()

        self._model_pipeline = []

        self._model_dir = self.config.model_dir
        self._model_info: Optional[_ModelInfo] = None
        self._model_name = self.config.model
        self.auto_parallel_config = AutoParallelConfig(
            world_size=config.parallel_config.world_size if config.
            parallel_config.auto_parallel else 1)
        default_config = self.config.auto_parallel_config
        self.auto_parallel_config.set_defaults(
            cluster_key=default_config.cluster_key,
            cluster_info=default_config.cluster_info,
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
            if self.config.model is None:
                raise ValueError(
                    "Either model_dir or model should be provided to ModelConfig."
                )
            self._model_pipeline.append(
                ("Downloading HF model", self._download_hf_model))

        self._model_format = self.config.model_format

        if self._model_format is _ModelFormatKind.HF:
            ''' HF -> TRT checkpoints -> engine '''
            self._model_pipeline.append(
                ("Loading HF model to memory", self._load_model_from_hf))
            self._model_pipeline.append(
                ("Building TRT-LLM engine", self._build_engine))
        elif self._model_format is _ModelFormatKind.TLLM_CKPT:
            ''' TRT checkpoints -> engine '''
            self._model_pipeline.append(("Loading TRT checkpoints to memory",
                                         self._load_model_from_ckpt))
            self._model_pipeline.append(
                ("Build TRT-LLM engine", self._build_engine))
        elif self._model_format is _ModelFormatKind.TLLM_ENGINE:
            ''' TFRT engine '''
            self._model_pipeline.append(
                ("Loading TensorRT-LLM engine", self._load_engine_buffer))
        else:
            raise ValueError(f"Unknown model format {self._model_format}")

        if self.tokenizer is None:
            ''' Use the default tokenizer if no one is provided. '''
            self._model_pipeline.append(
                ("Initialize tokenizer", self._load_hf_tokenizer))

    def __call__(self) -> _ModelRuntimeContext:
        if self.config.parallel_config.is_multi_gpu:
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
        assert self.workspace is not None
        assert isinstance(self.config.model, str)
        self._model_dir = download_hf_model(self.config.model)
        self.config.model_dir = self._model_dir
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
        model2struct = dict(
            LlamaForCausalLM=LLaMAForCausalLM,
            MixtralForCausalLM=LLaMAForCausalLM,
        )
        if model_arch not in model2struct:
            raise KeyError(
                f"Unsupported model architecture: {model_arch}, "
                f"only {', '.join(model2struct.keys())} are supported now.")

        model_cls = model2struct[model_arch]

        if self.config.quant_config.quant_mode.has_any_quant():
            assert self.workspace is not None
            checkpoint_dir = f"{self.workspace}/quantized-checkpoint"
            if self.rank == 0:
                model_cls.quantize(
                    self._model_dir,
                    checkpoint_dir,
                    dtype=self.dtype,
                    mapping=self.mapping,
                    quant_config=self.config.quant_config,
                )
            if self.config.parallel_config.is_multi_gpu:
                mpi_barrier()
            self.model = model_cls.from_checkpoint(checkpoint_dir,
                                                   rank=self.mapping.rank)
        else:
            self.model = model_cls.from_hugging_face(
                self._model_dir,
                dtype=self.dtype,
                mapping=self.mapping,
                quant_config=self.config.quant_config,
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

    def _build_engine_from_inmemory_model(self):
        assert isinstance(self.config.model, Module)
        self._model_info = _ModelInfo.from_module(self.model)

    def _build_engine(self):

        self.build_config.update(auto_parallel_config=self.auto_parallel_config)
        if self.auto_parallel_config.enabled:
            self.model.config.mapping.rank = self.rank
        engine = build(self.model, self.build_config)

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
