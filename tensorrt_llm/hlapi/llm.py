import gc
import json
import os
import tempfile
import time
from concurrent.futures import as_completed
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union

import tensorrt as trt
import torch

from .._utils import mpi_rank, mpi_world_size
from ..builder import (BuildConfig, Builder, BuilderConfig, PluginConfig,
                       QuantMode)
from ..executor import GenerationExecutor, GenerationResult
from ..logger import logger
from ..mapping import Mapping
from ..models.modeling_utils import PretrainedConfig
from ..module import Module
from ..runtime import (GenerationSession, ModelRunner, SamplingConfig,
                       model_runner)
from .mpi_session import MpiSession, NodeSession
from .tokenizer import TokenizerBase, TransformersTokenizer
from .utils import (GenerationOutput, file_with_suffix_exists, get_device_count,
                    print_colored, print_traceback_on_error)


@dataclass
class ParallelConfig:
    ''' The model distribution configs for LLM.  '''
    tp_size: int = 1
    pp_size: int = 1
    devices: List[int] = field(default_factory=list)

    @property
    def world_size(self) -> int:
        return self.tp_size * self.pp_size


class QuantConfig:

    def __init__(self, quant_mode: Optional[QuantMode] = None):
        self._quant_mode = quant_mode or QuantMode(0)

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

    # Override the underlying plugin config. Default values will be used if it's None.
    plugin_config: Optional[PluginConfig] = None

    @property
    def is_multi_gpu(self) -> bool:
        return self.parallel_config.tp_size > 1

    def __post_init__(self):
        assert self.model_dir, "model_dir is required."
        if self.model:
            raise NotImplementedError("model is not supported yet.")


class LLM:
    '''
    An end-to-end runner for LLM tasks.

    Classical usage:

    config = ModelConfig("llama-7B")

    llm = LLM(config)
    llm.generate(["What is your name?"]) # => ["My name is Llama."]
    '''

    @dataclass
    class AdditionalOptions:
        kvcahe_free_gpu_memory_fraction: Optional[float] = None

        def get_valid_options(self) -> List[str]:
            return [
                x for x in self.__dict__
                if x != 'self' and not x.startswith('_')
            ]

    def __init__(self,
                 config: ModelConfig,
                 tokenizer: Optional[TokenizerBase] = None,
                 enable_tokenizer: bool = True,
                 dump_model_processing_summary: Optional[str] = None,
                 async_mode: bool = False,
                 async_engine_tmp_dir: Optional[str] = None,
                 **options):
        '''
        Args:
            config: The model config for the model.
            tokenizer: User provided tokenizer, will override the default one
            enable_tokenizer: Turn on the preprocessing and postprocessing with a tokenizer to make the llm pipeline takes texts as input and produces text as output.
            dump_model_processing_summary: Dump the summary of the model building into a log file.
            async_mode: Run the model in async mode.
            async_engine_tmp_dir: The temporary directory to save the async engine. Only for debugging.
        '''

        self.config = config

        self._tokenizer = tokenizer
        self.enable_tokenizer = enable_tokenizer
        self.dump_model_processing_summary = dump_model_processing_summary
        self.async_mode = async_mode
        self.async_engine_tmp_dir = async_engine_tmp_dir
        # TODO[chunweiy]: Support more models and gpus
        self._extra_build_config = ModelLoader.get_extra_build_configs(
            'llama7b', 'a100')

        if self.async_mode and self.config.is_multi_gpu:
            raise NotImplementedError(
                f"Async mode is not supported for multi-gpu yet. {self.config.parallel_config}"
            )

        if not self.async_mode and self.config.is_multi_gpu:
            import torch
            assert torch.cuda.is_available(), "No CUDA device is available."
            assert get_device_count() >= self.config.parallel_config.world_size, \
                f"Only {get_device_count()} CUDA devices are available, but {self.config.parallel_config.world_size} are required."

            logger.warning(
                f'start MpiSession with {self.config.parallel_config.tp_size} workers'
            )
            self.mpi_session = MpiSession(
                n_workers=self.config.parallel_config.tp_size)

        self._async_engine: Optional[GenerationExecutor] = None
        self._additional_options = LLM.AdditionalOptions()

        # set additional options for constructing the LLM pipeline
        valid_options = self._additional_options.get_valid_options()

        def set_option(key, value):
            if key in valid_options:
                logger.warning(
                    f"Additionl option is a preview feature, setting {key}={value}"
                )
                setattr(self._additional_options, key, value)
            else:
                raise ValueError(f"Invalid option {key}")

        for key, value in options.items():
            set_option(key, value)

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
        assert not self.async_mode, "Please use generate_async(...) instead on async mode"
        prompts = list(prompts)

        if sampling_config is None:
            sampling_config = self.get_default_sampling_config()
        assert sampling_config.num_beams == self._extra_build_config.max_beam_width, "Beam search is not supported yet."
        assert len(prompts) <= self._extra_build_config.max_batch_size, \
            "The batch size is too large, not supported yet"
        assert sum(len(prompt) for prompt in prompts) <= self._extra_build_config.max_num_tokens, \
            "The total input length is too large, not supported yet"

        if self.config.is_multi_gpu:
            return self._generate_sync_multi_gpu(prompts, sampling_config)
        else:
            return self._generate_sync(
                prompts,
                self.runtime_context,
                sampling_config,
                max_batch_size=self._extra_build_config.max_batch_size)

    def generate_async(self,
                       prompt: Union[str, List[int]],
                       sampling_config: Optional[SamplingConfig] = None,
                       streaming: bool = False) -> GenerationResult:
        ''' Generate in asynchronuous mode. '''
        assert self._async_engine, "The async engine is not built yet."

        sampling_config = sampling_config or self.get_default_sampling_config()
        assert sampling_config is not None
        assert sampling_config.num_beams == self._extra_build_config.max_beam_width, "Beam search is not supported yet."
        assert len(prompt) <= self._extra_build_config.max_num_tokens, \
            "The total input length is too large, not supported yet"

        assert isinstance(prompt, str), "Only support str prompt for now"
        results = self._async_engine.generate_async(
            prompt,
            streaming=streaming,
            # TODO[chunweiy]: make executor support all the options in SamplingConfig
            max_new_tokens=sampling_config.max_new_tokens)
        return results

    @property
    def tokenizer(self) -> TokenizerBase:
        if self._tokenizer:
            return self._tokenizer
        if hasattr(self, 'runtime_context'):
            return self.runtime_context.tokenizer

    def save(self, engine_dir: str):
        ''' Save the built engine to the given path.  '''
        # TODO[chunweiy]: fix issue here: save() requires the engine-buffer in memory even after engine loading, which consumes a lot of memory.
        logger.info(f"Save model to {engine_dir}")

        if self.config.is_multi_gpu:
            self.mpi_session.submit_sync(LLM._node_save_task, engine_dir,
                                         self.config.model_dir,
                                         self.config.parallel_config.pp_size,
                                         self.config.parallel_config.tp_size)
        else:
            ModelLoader.save(self.runtime_context,
                             self.config.model_dir,
                             engine_dir=engine_dir,
                             model_info=self.runtime_context.model_info)

    def get_default_sampling_config(self) -> Optional[SamplingConfig]:
        ''' Get the default sampling config for the model.
        You can override the options.
        '''
        assert self.enable_tokenizer, "Tokenizer is required to deduce the default sampling config"
        tokenizer = self.tokenizer
        if not tokenizer:
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

        def build_sync():
            if self.config.is_multi_gpu:
                futures = self.mpi_session.submit(
                    LLM._node_build_task, self.config, self.enable_tokenizer,
                    self.config.parallel_config.tp_size,
                    self.config.parallel_config.pp_size, self._tokenizer)
                res = []
                for future in as_completed(futures):
                    res.append(future.result())
                return bool(res)
            else:
                model_loader = ModelLoader(self.config,
                                           self.enable_tokenizer,
                                           tokenizer=self._tokenizer)
                self.runtime_context = model_loader()

                self._tokenizer = self.runtime_context.tokenizer

                self.default_sampling_config = self.get_default_sampling_config(
                ) if self.tokenizer else None

                return True

        # TODO[chunweiy]: Support multi-gpu build
        def build_async():
            engine_dir = self.async_engine_tmp_dir
            if engine_dir is None:
                temp_dir = tempfile.TemporaryDirectory()
                engine_dir = temp_dir.name

            model_format = ModelLoader.get_model_format(self.config.model_dir)
            if model_format is not _ModelFormatKind.TLLM_ENGINE:

                with ModelLoader(self.config,
                                 self.enable_tokenizer,
                                 tokenizer=self._tokenizer) as model_loader:

                    runtime_context = model_loader()
                    # runner is not needed for GptManager

                    # TODO[chunweiy]: Make GptManager support in-memory engine-buffer to save disk loading lantenecy
                    ModelLoader.save(runtime_context,
                                     self.config.model_dir,
                                     engine_dir=engine_dir,
                                     model_info=runtime_context.model_info)

                    # Once saved, the engine_buffer is not needed anymore
                    del runtime_context

                gc.collect()
                torch.cuda.empty_cache()

            tokenizer = self.tokenizer
            if not isinstance(tokenizer, TokenizerBase):
                tokenizer = ModelLoader.load_hf_tokenizer(self.config.model_dir)
            assert isinstance(tokenizer, TokenizerBase)

            import tensorrt_llm.bindings as tllm
            executor_config = tllm.TrtGptModelOptionalParams()
            if self._additional_options.kvcahe_free_gpu_memory_fraction is not None:
                executor_config.kv_cache_config.free_gpu_memory_fraction = self._additional_options.kvcahe_free_gpu_memory_fraction

            self._async_engine = GenerationExecutor(
                engine_dir,
                tokenizer=tokenizer,
                max_beam_width=self._extra_build_config.max_beam_width,
                executor_config=executor_config,
                # TODO[chunweiy]: Expose more options
            )

            return True

        if self.async_mode:
            return build_async()
        else:
            return build_sync()

    @print_traceback_on_error
    @staticmethod
    def _node_build_task(config: ModelConfig,
                         enable_tokenizer: bool,
                         tp_size: int,
                         pp_size: int,
                         tokenizer: TokenizerBase = None) -> bool:
        assert not NodeSession.is_initialized()
        mapping = Mapping(tp_size=tp_size,
                          pp_size=pp_size,
                          rank=mpi_rank(),
                          world_size=tp_size * pp_size)

        model_loader = ModelLoader(config,
                                   enable_tokenizer,
                                   tokenizer=tokenizer,
                                   mapping=mapping)
        runtime_context = model_loader()

        # Hold the model builder for later use
        NodeSession.state = runtime_context
        return True

    @print_traceback_on_error
    @staticmethod
    def _node_generation_task(prompts: Union[List[str], List[List[int]]],
                              sampling_config: Optional[SamplingConfig],
                              max_batch_size: int) -> List[GenerationOutput]:
        assert NodeSession.is_initialized(), "Model is not built yet."
        assert isinstance(NodeSession.state, _ModelRuntimeContext)
        model: _ModelRuntimeContext = NodeSession.state
        if sampling_config is None:
            sampling_config = SamplingConfig(
                end_id=model.tokenizer.eos_token_id,
                pad_id=model.tokenizer.eos_token_id
                if model.tokenizer.pad_token_id is None else
                model.tokenizer.pad_token_id,
                output_sequence_lengths=True,
                return_dict=True) if model.tokenizer else None

        return list(
            LLM._generate_sync(prompts, model, sampling_config, max_batch_size))

    @print_traceback_on_error
    @staticmethod
    def _node_save_task(engine_dir: str, model_dir: str, pp_size: int,
                        tp_size: int):
        runtime_context: _ModelRuntimeContext = NodeSession.state

        mapping = Mapping(world_size=mpi_world_size(),
                          rank=mpi_rank(),
                          tp_size=tp_size,
                          pp_size=pp_size)
        ModelLoader.save(runtime_context,
                         model_dir,
                         engine_dir=engine_dir,
                         mapping=mapping,
                         model_info=runtime_context.model_info)

    def __getstate__(self):
        raise RuntimeError("LLM object can not be pickled.")

    @staticmethod
    def _generate_sync(prompts, runtime_context: "_ModelRuntimeContext",
                       sampling_config,
                       max_batch_size: int) -> Iterable[GenerationOutput]:
        ''' Generate in sync mode on a single GPU.  '''
        assert sampling_config is not None, "The sampling_config need to be provided."
        if not prompts:
            return []
        assert runtime_context.runtime, "The model runner is not built yet."

        def generate_batch(batch_input_ids: List[torch.Tensor]):
            batch_input_ids = [
                torch.tensor(x, dtype=torch.int32) for x in batch_input_ids
            ]  # List[torch.Tensor(seq)]

            assert len(batch_input_ids) <= max_batch_size, \
                f"Can not run batch size larger than {max_batch_size}, got {len(batch_input_ids)}"
            outputs = runtime_context.runtime.generate(batch_input_ids,
                                                       sampling_config)

            output_ids = outputs['output_ids']
            sequence_lengths = outputs['sequence_lengths']

            batch_size, num_beams, max_len = output_ids.size()
            input_lengths = [x.size(0) for x in batch_input_ids]
            assert num_beams == 1, "Support beam search later"

            for batch_idx in range(batch_size):
                for beam in range(num_beams):
                    inputs = output_ids[batch_idx][
                        0][:input_lengths[batch_idx]].tolist()

                    output_begin = input_lengths[batch_idx]
                    output_end = sequence_lengths[batch_idx][beam]
                    outputs = output_ids[batch_idx][beam][
                        output_begin:output_end].tolist()

                    output_text = runtime_context.tokenizer.decode(
                        outputs) if runtime_context.tokenizer else None

                    # get a sequence for each prompt directly
                    output = GenerationOutput(text=output_text,
                                              token_ids=outputs
                                              # TODO[chunweiy]: fill the probs
                                              )
                    yield output

        tokenizer = runtime_context.tokenizer

        def batching_prompts(prompts):
            need_tokenize: bool = isinstance(prompts[0], str)

            if need_tokenize:
                assert tokenizer, "The tokenizer is not built or provided."

            def process_batch(batch):
                return tokenizer.batch_encode_plus(
                    batch)['input_ids'] if need_tokenize else batch

            batch = []
            for i, prompt in enumerate(prompts):
                batch.append(prompt)
                if len(batch) >= max_batch_size:
                    yield process_batch(batch)
                    batch = []
            if batch:
                yield process_batch(batch)

        for batch in batching_prompts(prompts):
            outs = generate_batch(batch)
            for o in outs:
                yield o

    def _generate_sync_multi_gpu(
        self, prompts, sampling_config: Optional[SamplingConfig]
    ) -> Iterable[GenerationOutput]:
        # TODO[chunweiy]: May merge this with the one gpu version later
        assert self.config.is_multi_gpu, "The model is not distributed."

        features = self.mpi_session.submit(
            LLM._node_generation_task, prompts, sampling_config,
            self._extra_build_config.max_batch_size)

        res = [feature.result() for feature in as_completed(features)]

        # TODO[chunweiy]: make sure that the root's output is always the first
        return res[0]


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
        # The Dict format is { 'builder_config':..., 'plugin_config':...}
        dtype = config['plugin_config']['gpt_attention_plugin']
        return cls(dtype=dtype, architecture=config['builder_config']['name'])

    @classmethod
    def from_module(cls, module: Module):
        raise NotImplementedError()


@dataclass
class _ModelRuntimeContext:
    ''' _ModelRuntimeContext holds the minimum runtime resources for running a model.
    It could be a runtime cache in MPI nodes.
    '''
    runtime: Optional[Union[ModelRunner, trt.IHostMemory]] = None
    tokenizer: Optional[TokenizerBase] = None
    # engine_config is only used for saving the engine to disk
    engine_config: Optional[Union[dict, BuildConfig]] = None
    model_info: Optional[_ModelInfo] = None

    @property
    def engine(self) -> trt.IHostMemory:
        assert self.runtime is not None, "The model runner is not built yet."
        return self.runtime.serialize_engine()

    @property
    def model_structure(self) -> str:
        # "llama" or "opt" and so on
        return self.engine_config['builder_config']['name'] if isinstance(
            self.engine_config, dict) else self.engine_config.name


class ModelLoader:
    ''' The ModelLoader is used to build an end-to-end model from a model config.
    It will construct the runtime resources including engine, tokenizer, model runner, etc.
    '''

    def __init__(self,
                 config: ModelConfig,
                 enable_tokenizer: bool,
                 tokenizer: Optional[TokenizerBase],
                 mapping: Optional[Mapping] = None):
        self.config = config
        self.enable_tokenizer = enable_tokenizer
        self.tokenizer = tokenizer
        self.mapping = mapping

        if self.config.is_multi_gpu:
            assert self.mapping is not None, "The mapping is not set yet."

        self._model_pipeline = []

        self._model_dir = self.config.model_dir
        self._model_info: Optional[_ModelInfo] = None
        self._model_name = self.config.model
        # TODO[chunweiy]: Support more models and gpus
        self._extra_build_config = ModelLoader.get_extra_build_configs(
            'llama7b', 'h100')

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
                ("Load HF model to memory", self._build_model_from_hf))
            self._model_pipeline.append(
                ("Build TRT-LLM engine", self._build_engine_and_model_runner))
        elif self._model_format is _ModelFormatKind.TLLM_CKPT:
            ''' TFRT checkpoints -> engine '''
            # TODO[chunweiy]: Support checkpoints when quantization is ready
            raise NotImplementedError()
        elif self._model_format is _ModelFormatKind.TLLM_ENGINE:
            ''' TFRT engine '''
            self._model_pipeline.append(
                ("Load TensorRT-LLM engine", self._load_model_runner))
        else:
            raise ValueError(f"Unknown model format {self._model_format}")

        if self.enable_tokenizer and not self.tokenizer:
            ''' Use the default tokenizer if user doesn't provide one '''
            self._model_pipeline.append(
                ("Initialize tokenizer", self._load_hf_tokenizer))

    def __call__(self) -> _ModelRuntimeContext:
        if self.config.is_multi_gpu:
            torch.cuda.set_device(self.mapping.rank)

        n_steps = len(self._model_pipeline)
        to_log = not self.config.is_multi_gpu or mpi_rank() == 0

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

        assert self.runner is not None, "The model runner is not built yet."

        assert hasattr(self, '_builder_config') or hasattr(
            self, '_engine_config'), "config is not loaded."
        config = self._engine_config if hasattr(
            self, '_engine_config') else self._builder_config

        return _ModelRuntimeContext(tokenizer=self.tokenizer,
                                    runtime=self.runner,
                                    engine_config=config,
                                    model_info=self._model_info)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        for attr_name in dir(self):
            if not callable(getattr(
                    self, attr_name)) and not attr_name.startswith("__"):
                if attr_name not in ('model_format', ):
                    setattr(self, attr_name, None)

        gc.collect()
        torch.cuda.empty_cache()

    @property
    def model_format(self) -> _ModelFormatKind:
        return self._model_format

    # TODO[tali]: Replace this with a lower-level API
    @staticmethod
    def save(model: _ModelRuntimeContext,
             model_dir: str,
             engine_dir: str,
             model_info: _ModelInfo,
             mapping=None):
        ''' Save the built engine to the given path. '''
        mapping = mapping or Mapping()
        rank = mapping.rank if mapping else 0

        def save_engine_to_dir(engine_dir):
            # TODO[chunweiy, tao]: Fix here. The self.module is del after the constructor, that's why the self.model.save is not used here.
            def get_engine_name(model, dtype, tp_size, pp_size, rank):
                if pp_size == 1:
                    return '{}_{}_tp{}_rank{}.engine'.format(
                        model, dtype, tp_size, rank)
                return '{}_{}_tp{}_pp{}_rank{}.engine'.format(
                    model, dtype, tp_size, pp_size, rank)

            engine_dir = Path(engine_dir)
            if not engine_dir.exists():
                engine_dir.mkdir()
            config_path = engine_dir / 'config.json'

            assert model.model_info is not None
            engine_path = engine_dir / get_engine_name(
                model.model_info.model_name, model_info.dtype, mapping.tp_size,
                mapping.pp_size, rank)
            builder = Builder()
            # write config.json
            if isinstance(model.engine_config, BuilderConfig):
                builder.save_config(model.engine_config, config_path)
            elif isinstance(model.engine_config, dict):
                with open(config_path, 'w') as f:
                    json.dump(model.engine_config, f)
            else:
                raise ValueError("wrong engine_config type")

            logger.debug(f"Saving engine to {engine_path}")
            with open(engine_path, 'wb') as f:
                assert isinstance(model.engine, trt.IHostMemory)
                f.write(model.engine)

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

        save_engine_to_dir(engine_dir)
        if isinstance(model.tokenizer, TransformersTokenizer):
            if mapping is None or mapping.rank == 0:
                copy_hf_tokenizer_data_to_engine_dir()

    @staticmethod
    def get_model_format(model_dir: str) -> _ModelFormatKind:
        ''' Tell the format of the model.  '''
        # TODO[chunweiy]: Add checkpoint support
        if Path.exists(Path(model_dir) /
                       'generation_config.json') and file_with_suffix_exists(
                           model_dir, '.bin'):
            return _ModelFormatKind.HF
        if Path.exists(
                Path(model_dir) / 'config.json') and file_with_suffix_exists(
                    model_dir, '.engine'):
            return _ModelFormatKind.TLLM_ENGINE
        raise ValueError(f"Unknown model format for {model_dir}")

    def _download_hf_model(self):
        ''' Download HF model from third-party model hub like www.modelscope.cn or huggingface.  '''
        raise NotImplementedError()

    def _build_model_from_hf(self):
        ''' Build a TRT-LLM model from a HF model.  '''
        from ..models import LLaMAForCausalLM
        assert self._model_dir is not None

        import transformers
        _pretrained_config = transformers.PretrainedConfig.from_json_file(
            os.path.join(self._model_dir, 'config.json'))

        # TODO[chunweiy]: inspect from hf model/config
        model_arch = _pretrained_config.architectures[0]
        assert 'llama' in model_arch.lower(), "Only LLaMA is supported now"

        # TODO[chunweiy]: add more models if ready
        model2struct = dict(LlamaForCausalLM=LLaMAForCausalLM)

        self.model = model2struct[model_arch].from_hugging_face(
            self._model_dir,
            mapping=self.mapping,
            quant_mode=self.config.quant_config.quant_mode)
        self.pretrained_config = self.model.config
        self._model_info = _ModelInfo.from_pretrained_config(
            self.pretrained_config)

    def _build_engine_from_inmemory_model(self):
        assert isinstance(self.config.model, Module)
        self.model = self.config.model.from_hugging_face(
            self._model_dir,
            mapping=self.mapping,
            quant_mode=self.config.quant_config.quant_mode)
        self._model_info = _ModelInfo.from_module(self.model)

    def _load_model_runner(self):
        ''' Load a model runner from a TRT-LLM engine. '''
        assert self._model_dir
        logger.info(f"Loading model runner from {self._model_dir}")

        self.runner = ModelRunner.from_dir(self._model_dir)
        self._engine = self.runner.session.runtime.engine
        with open(os.path.join(self._model_dir, 'config.json'), 'r') as f:
            self._engine_config: dict = json.load(f)
            self._model_info = _ModelInfo.from_builder_config_json(
                self._engine_config)

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
            max_beam_width: int

        llama7b_config = ExtraBuildConfig(max_batch_size=128,
                                          max_input_len=412,
                                          max_output_len=200,
                                          max_num_tokens=4096,
                                          max_beam_width=1)

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

    def _build_engine_and_model_runner(self):
        ''' Build TensorRT-LLM engine from an in-memory model.
        The model runner will be created.
        '''
        self._engine, self._builder_config = self.model.to_trt(
            batch_size=self._extra_build_config.max_batch_size,
            input_len=self._extra_build_config.max_input_len,
            output_len=self._extra_build_config.max_output_len,
            plugin_config=self.config.plugin_config,
            # override some settings for build_config
            max_beam_width=self._extra_build_config.max_beam_width,
            max_num_tokens=self._extra_build_config.max_num_tokens)

        # delete the model explicitly to free all the build-time resources
        del self.model

        # TODO [chunweiy]: Is this conversion necessary?
        model_config, other_config = model_runner._builder_to_model_config(
            self._builder_config.to_dict())
        max_batch_size = other_config.get('max_batch_size')
        max_input_len = other_config.get('max_input_len')
        max_output_len = other_config.get('max_output_len')
        max_beam_width = other_config.get('max_beam_width')
        runtime_mapping = self.mapping or Mapping()
        session = GenerationSession(model_config, self._engine, runtime_mapping)
        # TODO[chunweiy]: switch to model_runner_cpp, currently it lacks serialize_engine support
        self.runner = ModelRunner(session, max_batch_size, max_input_len,
                                  max_output_len, max_beam_width)

    def _load_hf_tokenizer(self):
        assert self._model_dir
        self.tokenizer = ModelLoader.load_hf_tokenizer(self._model_dir)

    @staticmethod
    def load_hf_tokenizer(model_dir):
        return TransformersTokenizer.from_pretrained(model_dir,
                                                     legacy=False,
                                                     padding_side='left',
                                                     truncation_side='left',
                                                     trust_remote_code=True,
                                                     use_fast=True)

    def _convert_hf_to_trtllm_checkpoints(self):
        '''
        Convert a HuggingFace model to a TensorRT-LLM checkpoints.
        The checkpoints will be cached in the cache directory.
        '''
        raise NotImplementedError()

    def _quantize(self):
        ''' Quantize a TensorRT-LLM checkpoints from a TensorRT-LLM checkpoints.
        The checkpoints will be cached in the cache directory.
        '''
        raise NotImplementedError()
