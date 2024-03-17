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

from .._utils import mpi_rank
from ..auto_parallel.config import AutoParallelConfig, infer_cluster_key
from ..builder import (BuildConfig, Engine, EngineConfig, PluginConfig,
                       QuantMode, build)
from ..executor import (GenerationExecutor, GenerationResult,
                        ParallelGenerationExecutor)
from ..logger import logger
from ..mapping import Mapping
from ..models.modeling_utils import PretrainedConfig
from ..module import Module
from ..runtime import SamplingConfig
from .mpi_session import MpiSession, NodeSession
from .tokenizer import TokenizerBase, TransformersTokenizer
from .utils import (GenerationOutput, file_with_suffix_exists, get_device_count,
                    print_colored, print_traceback_on_error, release_gc)


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


class QuantConfig:

    def __init__(self,
                 quant_mode: Optional[QuantMode] = None,
                 quantize_lm_head: bool = False):
        self._quant_mode = quant_mode or QuantMode(0)
        self.quantize_lm_head = quantize_lm_head

    @property
    def quant_mode(self) -> QuantMode:
        return self._quant_mode

    def set_int8_kv_cache(self):
        self._quant_mode = self._quant_mode.set_int8_kv_cache()

    def set_fp8_kv_cache(self):
        self._quant_mode = self._quant_mode.set_fp8_kv_cache()

    def set_fp8_qdq(self):
        self._quant_mode = self._quant_mode.set_fp8_qdq()

    def init_from_description(self,
                              quantize_weights=False,
                              quantize_activations=False,
                              per_token=False,
                              per_channel=False,
                              per_group=False,
                              use_int4_weights=False,
                              use_int8_kv_cache=False,
                              use_fp8_kv_cache=False,
                              use_fp8_qdq=False):
        self._quant_mode = QuantMode.from_description(
            quantize_weights=quantize_weights,
            quantize_activations=quantize_activations,
            per_token=per_token,
            per_channel=per_channel,
            per_group=per_group,
            use_int4_weights=use_int4_weights,
            use_int8_kv_cache=use_int8_kv_cache,
            use_fp8_kv_cache=use_fp8_kv_cache,
            use_fp8_qdq=use_fp8_qdq)

    def __getattribute__(self, name: str) -> Any:

        def dummy_getter(*args, **kwargs):
            return getattr(self.quant_mode, name)(*args, **kwargs)

        if name.startswith('has_'):
            return dummy_getter

        return super().__getattribute__(name)


@dataclass
class ModelConfig:

    # ``model_dir`` helps to locate a local model, the format of the model is determined by the model file itself.
    # Either HF model, TensorRT-LLM checkpoints or TensorRT-LLM engine format is supported.
    model_dir: Optional[str] = None

    # ``model`` could either the model directory or a in-memory model.
    # If ``model`` specifies the model kind like "llama-7B", etc.  The model will be download automatically from third-party
    # model hub like www.modelscope.cn or huggingface
    model: Optional[Union[str, Module]] = None

    parallel_config: ParallelConfig = field(default_factory=ParallelConfig)

    quant_config: QuantConfig = field(default_factory=lambda: QuantConfig())

    # Switch the optimization on multi-head attention optimization for long context decoding.
    multi_block_mode: bool = False

    # The maximum beam width for beam search.
    max_beam_width: int = 1

    # Overwrite the underlying plugin config. Default values will be used if it's None.
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
        else:
            if self.parallel_config.world_size > 1:
                raise RuntimeError(
                    "world_size > 1 is only supported in auto parallel mode.")
            return self.parallel_config.tp_size * self.parallel_config.pp_size

    # TODO[chunweiy]: To support loading options from the engine config

    def __post_init__(self):
        assert self.model_dir, "model_dir is required."
        if self.model:
            raise NotImplementedError("model is not supported yet.")

        # TODO[chunweiy]: unify the model_dir to Path
        if self.model_dir is not None and not Path.exists(Path(self.model_dir)):
            raise ValueError(
                f"model_dir of path {self.model_dir} does not exist.")

        # Load parallel_config from the engine.
        if ModelLoader.get_model_format(
                self.model_dir) is _ModelFormatKind.TLLM_ENGINE:
            with open(Path(self.model_dir) / "config.json", "r") as f:
                engine_config = json.load(f)
            # TODO[chunweiy]: Remove the following if-check after the engine config is unified.
            if "pretrained_config" in engine_config:
                mapping = engine_config["pretrained_config"]["mapping"]

                if self.parallel_config.tp_size != 1 and self.parallel_config.tp_size != mapping[
                        "tp_size"]:
                    logger.warning(
                        f"tp_size {self.parallel_config.tp_size} is not consistent with the engine's tp_size {mapping['tp_size']}"
                    )
                if self.parallel_config.pp_size != 1 and self.parallel_config.pp_size != mapping[
                        "pp_size"]:
                    logger.warning(
                        f"pp_size {self.parallel_config.pp_size} is not consistent with the engine's pp_size {mapping['pp_size']}"
                    )

                self.parallel_config = ParallelConfig(
                    tp_size=mapping["tp_size"],
                    pp_size=mapping["pp_size"],
                )

    def _update_plugin_config(self, key: str, value: Any):
        if key == 'use_paged_context_fmha' and value is True:
            devices = self.parallel_config.get_devices()
            assert torch.cuda.get_device_properties(
                devices[0]
            ).major >= 8, "Paged context is only supported on post Volta GPUs"

        if self.plugin_config is None:
            self.plugin_config = {}

        if isinstance(self.plugin_config, PluginConfig):
            setattr(self.plugin_config, key, value)
        elif isinstance(self.plugin_config, dict):
            self.plugin_config[key] = value


class DecodingMode(Enum):
    ''' The decoding mode for the generation. Just a Pythonic wrapper for the C++ one. '''
    none = 0
    top_k = 1
    top_p = 2
    top_k_top_p = 3
    beam_search = 4

    def to_cpp(self):
        values = {
            DecodingMode.none.value: tllm.DecodingMode.none(),
            DecodingMode.top_k.value: tllm.DecodingMode.top_k(),
            DecodingMode.top_p.value: tllm.DecodingMode.top_p(),
            DecodingMode.top_k_top_p.value: tllm.DecodingMode.top_k_top_p(),
            DecodingMode.beam_search.value: tllm.DecodingMode.beam_search(),
        }
        return values[self.value]


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
                 tokenizer: Optional[TokenizerBase] = None,
                 kv_cache_config: Optional[KvCacheConfig] = None,
                 enable_trt_overlap: bool = False,
                 normalize_log_probs: bool = False,
                 enable_chunked_context: bool = False,
                 decoding_mode: Optional[DecodingMode] = None,
                 scheduling_policy: SchedulerPolicy = SchedulerPolicy.
                 GUARANTEED_NO_EVICT,
                 async_engine_tmp_dir: Optional[str] = None):
        '''
        Args:
            config: The model config for the model.
            tokenizer: User provided tokenizer, will override the default one if exists in the HF model or TRT-LLM engine.
            kv_cache_config: The config for the paged KV cache.
            enable_trt_overlap: When set to true, GptManager partitions available requests into 2 'microbatches' that can be run concurrently to hide exposed CPU runtime.
                However, it may not give performance benefits when the size of the model is not big enough to overlap the host overhead, or when the number of requests is too small.
            normalize_log_probs: When set to true, the log probabilities are normalized to avoid numerical issues.
            enable_chunked_context: Controls whether to do chunked decoding.
            decoding_mode: The decoding mode for the generation.
            scheduling_policy: The scheduling policy for the generation.
            async_engine_tmp_dir: The temporary directory to save the async engine. Only for debugging.
        '''

        self.config = config

        self._tokenizer = tokenizer
        self.async_engine_tmp_dir = async_engine_tmp_dir
        self.kv_cache_config = kv_cache_config
        self.enable_trt_overlap = enable_trt_overlap
        self.normalize_log_probs = normalize_log_probs
        self.enable_chunked_context = enable_chunked_context
        self.decoding_mode = decoding_mode
        self.scheduling_policy = scheduling_policy

        # TODO[chunweiy]: Support more models and gpus

        self._extra_build_config = ModelLoader.load_extra_build_configs_from_engine(
            self.config.model_dir)
        if not self._extra_build_config:
            self._extra_build_config = ModelLoader.get_extra_build_configs(
                'llama7b', 'a100')
        self.mpi_session = None

        if self.config.is_multi_gpu:
            if get_device_count() < self.config.world_size:
                raise RuntimeError(
                    f"Only {get_device_count()} GPUs are available, but {self.config.world_size} are required."
                )

            logger.info(
                f'start MpiSession with {self.config.world_size} workers')
            self.mpi_session = MpiSession(n_workers=self.config.world_size)

        # Due to the gptManager can only accept a engine path, we need to save the engine to a directory
        self._engine_dir: Union[tempfile.TemporaryDirectory, str, Path] = None
        self._executor: Optional[GenerationExecutor] = None

        self.runtime_context: Optional[_ModelRuntimeContext] = None

        # Update the plugin config if necessary
        if self.kv_cache_config is not None:
            if self.kv_cache_config.enable_block_reuse:
                logger.info(
                    f"Turn on `use_paged_context_fmha` due to enable_block_reuse"
                )
                self.config._update_plugin_config("use_paged_context_fmha",
                                                  True)

        self._build_model()

    def generate(
        self,
        prompts: Union[Iterable[str], Iterable[List[int]]],
        sampling_config: Optional[SamplingConfig] = None
    ) -> Iterable[GenerationOutput]:
        ''' Generate the output for the given inputs.

        Args:
            prompts: The raw text or token ids to the model.
            sampling_config: The sampling config for the generation, a default one will be used if not provided.
        '''
        prompts = list(prompts)

        if sampling_config is None:
            sampling_config = self.get_default_sampling_config()
        self._generate_check_arguments(prompts, sampling_config)

        results = self._executor.generate(
            prompts,
            max_new_tokens=sampling_config.max_new_tokens,
            end_id=sampling_config.end_id,
            pad_id=sampling_config.pad_id)

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

        results = self._executor.generate_async(
            prompt,
            streaming=streaming,
            # TODO[chunweiy]: make executor support all the options in SamplingConfig
            max_new_tokens=sampling_config.max_new_tokens,
            end_id=sampling_config.end_id,
            pad_id=sampling_config.pad_id)
        return results

    def _generate_check_arguments(self, prompts, sampling_config):
        if sampling_config is None:
            raise ValueError("The sampling_config should to be provided.")
        if sampling_config.num_beams > self.config.max_beam_width:
            raise ValueError(
                f"num_beams is larger than the maximum in the built engine {sampling_config.num_beams} > {self.config.max_beam_width}"
            )
        if len(prompts) > self._extra_build_config.max_batch_size:
            raise ValueError(
                f"Batch size {len(prompts)} is larger than the maximum in the built engine {self._extra_build_config.max_batch_size}"
            )

        input_digits = False
        if isinstance(prompts[0], list):
            input_digits = True
        if input_digits and sum(len(prompt) for prompt in prompts
                                ) > self._extra_build_config.max_num_tokens:
            raise ValueError(f"The total input length is too large")

        if self.decoding_mode is DecodingMode.beam_search and sampling_config.num_beams < 1:
            raise ValueError(
                f"num_beams should be no less than 1 for beam search, but get {sampling_config.num_beams}"
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
        assert self._engine_dir is not None, "The engine is not built yet."
        src_engine_dir = self._engine_dir.name if isinstance(
            self._engine_dir, tempfile.TemporaryDirectory) else self._engine_dir
        if src_engine_dir != engine_dir:
            shutil.copytree(src_engine_dir, engine_dir, dirs_exist_ok=True)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.mpi_session is not None:
            self.mpi_session.shutdown()
            self.mpi_session = None
        if hasattr(self, "_executor") and self._executor is not None:
            self._executor.__exit__(exc_type, exc_value, traceback)
            self._executor = None

    def _save_engine(self, engine_dir: str):
        logger.info(f"Save model to {engine_dir}")

        if self.config.is_multi_gpu:
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
            output_sequence_lengths=True,
            return_dict=True)

    def _build_model(self):
        model_format = ModelLoader.get_model_format(self.config.model_dir)

        self._engine_dir = self.config.model_dir

        def get_engine_dir():
            return self._engine_dir.name if isinstance(
                self._engine_dir,
                tempfile.TemporaryDirectory) else self._engine_dir

        if model_format is not _ModelFormatKind.TLLM_ENGINE:

            self._engine_dir = self.async_engine_tmp_dir
            if self._engine_dir is None:
                self._engine_dir = tempfile.TemporaryDirectory()

            if self.config.is_multi_gpu:
                self.mpi_session.submit_sync(
                    LLM._node_build_task,
                    self.config,
                    self._tokenizer,
                )
                self._save_engine(get_engine_dir())

                self.mpi_session.submit_sync(LLM._node_free_state_task)

            else:

                with ModelLoader(self.config,
                                 tokenizer=self._tokenizer) as model_loader:

                    runtime_context = model_loader()

                    # TODO[chunweiy]: Make GptManager support in-memory engine-buffer to save disk loading lantenecy
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
        executor_config.decoding_mode = self.decoding_mode.to_cpp(
        ) if self.decoding_mode else None

        if self.config.is_multi_gpu:
            self._executor = ParallelGenerationExecutor(
                world_size=self.config.world_size,
                engine_dir=get_engine_dir(),
                tokenizer=tokenizer,
                max_beam_width=self.config.max_beam_width,
                executor_policy=self.scheduling_policy,
                executor_config=executor_config,
            )
        else:

            self._executor = GenerationExecutor(
                get_engine_dir(),
                tokenizer=tokenizer,
                max_beam_width=self.config.max_beam_width,
                executor_config=executor_config,
                executor_policy=self.scheduling_policy,
            )

    @print_traceback_on_error
    @staticmethod
    def _node_build_task(config: ModelConfig,
                         tokenizer: TokenizerBase = None) -> bool:
        assert not NodeSession.is_initialized()

        with ModelLoader(config, tokenizer=tokenizer) as model_loader:
            runtime_context = model_loader()

        # Hold the model builder for later use
        NodeSession.state = runtime_context
        return True

    @print_traceback_on_error
    @staticmethod
    def _node_save_task(engine_dir: str, model_dir: str):
        runtime_context: _ModelRuntimeContext = NodeSession.state
        assert isinstance(runtime_context,
                          _ModelRuntimeContext), "Model is not built yet."

        ModelLoader.save(runtime_context,
                         model_dir,
                         engine_dir=engine_dir,
                         model_info=runtime_context.model_info)

    @print_traceback_on_error
    @staticmethod
    def _node_free_state_task():
        NodeSession.state = None
        # release the large resource explicitly and immediately, since the following LLM pipeline may need a lot of memory
        release_gc()

    def __getstate__(self):
        raise RuntimeError("LLM object can not be pickled.")

    def __del__(self):
        self.__exit__(None, None, None)


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
        assert self.architecture is not None, "The architecture is not set yet."
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

    def __init__(self, config: ModelConfig, tokenizer: Optional[TokenizerBase]):
        self.config = config
        self.tokenizer = tokenizer
        self.rank = mpi_rank() if config.is_multi_gpu else 0
        if not config.is_multi_gpu:
            self.mapping = Mapping()
        elif config.parallel_config.auto_parallel:
            self.mapping = Mapping()
            self.mapping.rank = self.rank
        else:
            self.mapping = Mapping(
                tp_size=config.parallel_config.tp_size,
                pp_size=config.parallel_config.pp_size,
                rank=self.rank,
                world_size=config.world_size,
            )

        self._model_pipeline = []

        self._model_dir = self.config.model_dir
        self._model_info: Optional[_ModelInfo] = None
        self._model_name = self.config.model
        # TODO[chunweiy]: Support more models and gpus
        self._extra_build_config = ModelLoader.get_extra_build_configs(
            'llama7b', 'h100')
        self.auto_parallel_config = AutoParallelConfig(
            world_size=config.parallel_config.world_size)
        default_config = self._extra_build_config.auto_parallel_config
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

        assert self._model_dir is not None, "The model_dir is not set yet."
        self._model_format = ModelLoader.get_model_format(self._model_dir)

        if self._model_format is _ModelFormatKind.HF:
            ''' HF -> TFRT checkpoints -> engine '''
            self._model_pipeline.append(
                ("Load HF model to memory", self._load_model_from_hf))
            self._model_pipeline.append(
                ("Build TRT-LLM engine", self._build_engine))
        elif self._model_format is _ModelFormatKind.TLLM_CKPT:
            ''' TFRT checkpoints -> engine '''
            # TODO[chunweiy]: Support checkpoints when quantization is ready
            raise NotImplementedError()
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

        assert self._engine_buffer is not None, "The engine is not built yet."

        assert hasattr(self, '_engine_config'), "config is not loaded."
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
        # TODO[chunweiy]: Add checkpoint support
        if (Path.exists(Path(model_dir) / 'generation_config.json')
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
        ''' Build a TRT-LLM model from a HF model.  '''
        from ..models import LLaMAForCausalLM
        assert self._model_dir is not None

        import transformers
        _pretrained_config = transformers.PretrainedConfig.from_json_file(
            os.path.join(self._model_dir, 'config.json'))

        # TODO[chunweiy]: inspect from hf model/config
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

        self.model = model2struct[model_arch].from_hugging_face(
            self._model_dir,
            mapping=self.mapping,
            quant_mode=self.config.quant_config.quant_mode,
            quantize_lm_head=self.config.quant_config.quantize_lm_head,
        )
        self.pretrained_config = self.model.config
        self._model_info = _ModelInfo.from_pretrained_config(
            self.pretrained_config)

    def _build_engine_from_inmemory_model(self):
        assert isinstance(self.config.model, Module)
        self.model = self.config.model.from_hugging_face(
            self._model_dir,
            mapping=self.mapping,
            quant_mode=self.config.quant_config.quant_mode,
            quantize_lm_head=self.config.quant_config.quantize_lm_head,
        )
        self._model_info = _ModelInfo.from_module(self.model)

    def _build_engine(self):
        max_input_len = self._extra_build_config.max_input_len
        max_output_len = self._extra_build_config.max_output_len
        max_batch_size = self._extra_build_config.max_batch_size
        max_beam_width = self.config.max_beam_width
        max_num_tokens = self._extra_build_config.max_num_tokens

        plugin_config = self.config.plugin_config
        if not isinstance(self.config.plugin_config, PluginConfig):
            plugin_config = self.model.default_plugin_config()
            # patch the additional options
            if isinstance(self.config.plugin_config, dict):
                for k, v in self.config.plugin_config.items():
                    setattr(plugin_config, k, v)
        if self.config.multi_block_mode:
            plugin_config.multi_block_mode = True

        build_config = BuildConfig(
            max_input_len=max_input_len,
            max_output_len=max_output_len,
            max_batch_size=max_batch_size,
            max_beam_width=max_beam_width,
            max_num_tokens=max_num_tokens,
            strongly_typed=True,
            auto_parallel_config=self.auto_parallel_config,
            plugin_config=plugin_config,
        )
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
    def get_extra_build_configs(model: str, device: str):
        # This is a demo implementation for some the default values targeting at a matrix of model and GPU
        # TODO[chunweiy]: Add more default configs for models x devices

        @dataclass
        class ExtraBuildConfig:
            max_batch_size: int
            max_input_len: int
            max_output_len: int
            max_num_tokens: int
            auto_parallel_config: AutoParallelConfig = None

        auto_parallel_config = AutoParallelConfig(
            cluster_key=infer_cluster_key(),
            sharded_io_allowlist=[
                "past_key_value_\\d+",
                "present_key_value_\\d*",
            ],
            same_buffer_io={
                "past_key_value_(\\d+)": "present_key_value_\\1",
            },
        )

        llama7b_config = ExtraBuildConfig(max_batch_size=128,
                                          max_input_len=412,
                                          max_output_len=200,
                                          max_num_tokens=4096)
        llama7b_config.auto_parallel_config = auto_parallel_config

        # Default configs for some meta parameters concerning engine building are assigned here.
        # Ideally, runtime could adapt these settings and make them invisible to users.
        default_config: Dict[str, Dict[str, ExtraBuildConfig]] = {
            'llama7b': {
                'a30': llama7b_config,
                'a100': llama7b_config,
                'h100': llama7b_config,
            }
        }

        return default_config[model][device]

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
