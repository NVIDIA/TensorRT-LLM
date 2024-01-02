import json
import os
import time
from concurrent.futures import as_completed
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Iterable, List, Optional, Union

import torch
from tqdm import tqdm
from transformers import AutoTokenizer

from tensorrt_llm import Mapping, Module, logger
from tensorrt_llm.builder import BuildConfig, BuilderConfig
from tensorrt_llm.runtime import (GenerationSession, ModelRunner,
                                  SamplingConfig, model_runner)
from tensorrt_llm.runtime.engine import EngineConfig

from .mpi_session import MpiSession, NodeSession, mpi_rank, mpi_size


@dataclass
class ParallelConfig:
    ''' The model distribution configs for LLM.  '''
    tp_size: int = 1
    pp_size: int = 1
    devices: List[int] = field(default_factory=list, init=False)

    def __post_init__(self):
        assert self.tp_size > 0, "tp_size should be larger than 0"
        assert self.pp_size > 0, "pp_size should be larger than 0"
        assert not self.devices or len(self.devices) == self.world_size

    @property
    def world_size(self) -> int:
        return self.tp_size * self.pp_size


@dataclass
class ModelConfig:
    ''' ModelConfig holds the options for a model.

    An example of the usage:
        # A llama-7B model
        config = ModelConfig('llama-7B')
        # optionally override the default options
        config.build_config.max_batch_size = 64
    '''

    # the options shared by all models

    # ``model`` could either the model directory or a in-memory model.
    # If ``model`` specifies the model kind like "llama-7B", etc.  The model will be download automatically from third-party
    # model hub like www.modelscope.cn or huggingface
    model: Optional[Union[str, Module]] = None

    # ``model_dir`` helps to locate a local model, the format of the model is determined by the model file itself.
    # Either HF model, TensorRT-LLM checkpoints or TensorRT-LLM engine format is supported.
    model_dir: Optional[str] = None

    # ``build_config`` contains the options for building the model.
    build_config = BuildConfig()

    # ``quant_config`` contains the options for quantizing the model.
    # quant_config: QuantMode = QuantMode()

    # ``parallel_config`` contains the options for distributed inference.
    parallel_config: ParallelConfig = ParallelConfig()

    def __post_init__(self):
        assert self.model or self.model_dir, "Either model or model_dir should be provided."

    @property
    def is_multi_gpu(self) -> bool:
        return self.parallel_config.tp_size > 1


class ModelFormatKind(Enum):
    HF = 0
    TLLM_CKPT = 1
    TLLM_ENGINE = 2


TokenIdsTy = List[int]


@dataclass
class GenerationOuptut:
    request_id: int = -1
    generate_pieces: List["GenerationPiece"] = field(default_factory=list)


@dataclass
class GenerationPiece:
    ''' The output of the generation.
    For normal text generation, there is only one GenerationPiece for a given input.
    For streaming generation, there could be multiple GenerationOutput each for a generated piece.
    '''
    index: int = 0
    text: str = ""
    token_ids: List[int] = field(default_factory=list)
    logprobs: List[float] = field(default_factory=list)


class TokenizerBase:
    ''' This is a protocol for the tokenizer. Users can implement their own tokenizer by inheriting this class.  '''

    @property
    def eos_token_id(self) -> int:
        ''' Return the id of the end of sentence token.  '''
        raise NotImplementedError()

    @property
    def pad_token_id(self) -> int:
        ''' Return the id of the padding token.  '''
        raise NotImplementedError()

    def encode(self, text: str) -> TokenIdsTy:
        ''' Encode the text to token ids.  '''
        raise NotImplementedError()

    def decode(self, token_ids: TokenIdsTy) -> str:
        ''' Decode the token ids to text.  '''
        raise NotImplementedError()

    def batch_encode_plus(self, texts: List[str]) -> dict:
        ''' Encode the batch of texts to token ids.  '''
        raise NotImplementedError()


class TransformersTokenizer(TokenizerBase):
    ''' A wrapper for the Transformers' tokenizer.
    This is the default tokenizer for LLM. '''

    @classmethod
    def from_pretrained(self, pretrained_model_dir: str, **kwargs):
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_dir,
                                                  **kwargs)
        return TransformersTokenizer(tokenizer)

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    @property
    def eos_token_id(self) -> int:
        return self.tokenizer.eos_token_id

    @property
    def pad_token_id(self) -> int:
        return self.tokenizer.pad_token_id

    def encode(self, text: str) -> TokenIdsTy:
        return self.tokenizer.encode(text)

    def decode(self, token_ids: TokenIdsTy) -> str:
        return self.tokenizer.decode(token_ids)

    def batch_encode_plus(self, texts: List[str]) -> dict:
        return self.tokenizer.batch_encode_plus(texts)


@dataclass
class LLM:
    '''
    An end-to-end runner for LLM tasks.

    Classical usage:

    config = ModelConfig("llama-7B")

    llm = LLM(config)
    llm("What is your name?") # => "My name is Llama."
    '''

    def __init__(self,
                 config: ModelConfig,
                 tokenizer: Optional[TokenizerBase] = None,
                 enable_tokenizer: bool = True,
                 disable_model_download: bool = False,
                 display_model_processing_summary: bool = False,
                 dump_model_processing_summary: Optional[str] = None):
        '''
        Args:
            config: The model config for the model.
            tokenizer: User provided tokenizer, will override the default one
            enable_tokenizer: Turn on the preprocessing and postprocessing with a tokenizer to make the llm pipeline takes texts as input and produces text as output.
            disable_model_download: Disable downloading the HF model from third-party model hub like www.modelscope.cn or huggingface.
            display_model_processing_summary: Display the summary of the model building.
            dump_model_processing_summary: Dump the summary of the model building into a log file.
        '''

        self.config = config
        self._tokenizer = tokenizer
        self.enable_tokenizer = enable_tokenizer
        self.disable_model_download = disable_model_download
        self.display_model_processing_summary = display_model_processing_summary
        self.dump_model_processing_summary = dump_model_processing_summary

        if self.config.is_multi_gpu:
            import torch
            assert torch.cuda.is_available(), "No CUDA device is available."
            assert torch.cuda.device_count() >= self.config.parallel_config.world_size, \
                f"Only {torch.cuda.device_count()} CUDA devices are available, but {self.config.parallel_config.world_size} are required."

            logger.warning(
                f'start MpiSession with {self.config.parallel_config.tp_size} workers'
            )
            self.mpi_session = MpiSession(
                n_workers=self.config.parallel_config.tp_size)

        self._build_model()

    def __call__(
        self,
        prompts: List[str] | List[TokenIdsTy],
        sampling_config: Optional[SamplingConfig] = None
    ) -> Iterable[GenerationOuptut]:
        ''' Generate the output for the given inputs.

        Args:
            prompts: The raw text or token ids to the model.
            sampling_config: The sampling config for the generation, a default one will be used if not provided.
        '''

        if self.config.is_multi_gpu:
            return self._generate_sync_multi_gpu(prompts, sampling_config)
        else:
            return self._generate_sync(prompts, self.runtime_stuff,
                                       sampling_config)

    def __getstate__(self):
        # Customize the members to be pickled
        # We should not pickle huge objects like tokenizer, since `self` maybe pickled and sent to MPI nodes each submit().
        state = self.__dict__.copy()

        def rm(attr):
            if attr in state:
                del state[attr]

        for key in list(state.keys()):
            if key == "_tokenizer":
                # User passed tokenizer should be distributed to MPI nodes to override the default one.
                continue
            if key.startswith('_'):
                del state[key]

        rm("runtime_stuff")
        rm("mpi_session")

        # TODO[chunweiy]: Disable config pickle later
        # rm_attr("config")

        return state

    def _generate_sync(self, prompts, runtime_stuff: "_ModelRuntimeStuff",
                       sampling_config) -> Iterable[GenerationOuptut]:
        ''' Generate in sync mode on a single GPU.  '''
        sampling_config = sampling_config or self.default_sampling_config
        assert sampling_config is not None, "The sampling_config need to be provided."
        build_config = self.config.build_config
        if not prompts:
            return []
        assert runtime_stuff.runner, "The model runner is not built yet."

        def generate_batch(batch_input_ids: List[torch.Tensor]):
            batch_input_ids = [
                torch.tensor(x, dtype=torch.int32) for x in batch_input_ids
            ]  # List[torch.Tensor(seq)]

            assert len(batch_input_ids) <= build_config.max_batch_size, \
                f"Can not run batch size larger than {build_config.max_batch_size}, got {len(batch_input_ids)}"
            outputs = runtime_stuff.runner.generate(batch_input_ids,
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

                    output_text = runtime_stuff.tokenizer.decode(
                        outputs) if runtime_stuff.tokenizer else None

                    # get a sequence for each prompt directly
                    piece = GenerationPiece(text=output_text, token_ids=outputs)
                    yield GenerationOuptut(generate_pieces=[piece])

        tokenizer = runtime_stuff.tokenizer

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
                if len(batch) >= build_config.max_batch_size:
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
    ) -> Iterable[GenerationOuptut]:
        # TODO[chunweiy]: May merge this with the one gpu version later
        assert self.config.is_multi_gpu, "The model is not distributed."

        features = self.mpi_session.submit(self._node_generation_task, prompts,
                                           sampling_config)

        res = [feature.result() for feature in as_completed(features)]

        # TODO[chunweiy]: make sure that the root's output is always the first
        return res[0]

    def _node_generation_task(
            self, prompts: List[str] | List[List[int]],
            sampling_config: Optional[SamplingConfig]
    ) -> List[GenerationOuptut]:
        assert NodeSession.is_initialized(), "Model is not built yet."
        assert isinstance(NodeSession.state, _ModelRuntimeStuff)
        model: _ModelRuntimeStuff = NodeSession.state
        if sampling_config is None:
            sampling_config = SamplingConfig(
                end_id=model.tokenizer.eos_token_id,
                pad_id=model.tokenizer.eos_token_id
                if model.tokenizer.pad_token_id is None else
                model.tokenizer.pad_token_id,
                output_sequence_lengths=True,
                return_dict=True) if model.tokenizer else None

        return list(self._generate_sync(prompts, model, sampling_config))

    @property
    def tokenizer(self) -> TokenizerBase:
        return self._tokenizer or self.runtime_stuff.tokenizer

    def save(self, engine_dir: str):
        ''' Save the built engine to the given path.  '''

        if self.config.is_multi_gpu:
            futures = self.mpi_session.submit(
                self._node_save_task, engine_dir, self.config.model_dir,
                self.config.parallel_config.pp_size,
                self.config.parallel_config.tp_size)
            for future in futures:
                future.result()
        else:
            _ModelBuilder.save(self.runtime_stuff,
                               self.config.model_dir,
                               engine_dir=engine_dir)

    def _node_save_task(self, engine_dir: str, model_dir: str, pp_size: int,
                        tp_size: int):
        runtime_stuff = NodeSession.state
        mapping = Mapping(world_size=mpi_size(),
                          rank=mpi_rank(),
                          tp_size=tp_size,
                          pp_size=pp_size)
        _ModelBuilder.save(runtime_stuff,
                           model_dir,
                           engine_dir=engine_dir,
                           mapping=mapping)

    def _build_model(self):

        if self.config.is_multi_gpu:
            futures = self.mpi_session.submit(
                self._node_build_task, self.config.parallel_config.tp_size,
                self.config.parallel_config.pp_size, self._tokenizer)
            res = []
            for future in as_completed(futures):
                res.append(future.result())
            return bool(res)
        else:
            model_builder = _ModelBuilder(self.config,
                                          self.enable_tokenizer,
                                          tokenizer=self._tokenizer)
            self.runtime_stuff = model_builder()

            self._tokenizer = self.runtime_stuff.tokenizer

            self.default_sampling_config = SamplingConfig(
                end_id=self.tokenizer.eos_token_id,
                pad_id=self.tokenizer.eos_token_id
                if self.tokenizer.pad_token_id is None else
                self.tokenizer.pad_token_id,
                output_sequence_lengths=True,
                return_dict=True) if self.tokenizer else None

            return True

    def _node_build_task(self,
                         tp_size: int,
                         pp_size: int,
                         tokenizer: TokenizerBase = None):
        assert not NodeSession.is_initialized()
        mapping = Mapping(tp_size=tp_size,
                          pp_size=pp_size,
                          rank=mpi_rank(),
                          world_size=tp_size * pp_size)

        model_builder = _ModelBuilder(self.config,
                                      self.enable_tokenizer,
                                      tokenizer=tokenizer,
                                      mapping=mapping)
        runtime_stuff = model_builder()

        # Hold the model builder for later use
        NodeSession.state = runtime_stuff


@dataclass
class _ModelRuntimeStuff:
    ''' _ModelRuntimeStuff holds the minimum runtime stuff for running a model.
    It will be hold as a runtime cache in MPI nodes in the multi-gpu mode.
    '''
    runner: Optional[ModelRunner] = None
    tokenizer: Optional[TokenizerBase] = None
    # engine is only used for saving the engine to disk
    engine: Optional[bytes] = None
    # engine_config is only used for saving the engine to disk
    engine_config: Optional[dict | BuildConfig | EngineConfig] = None

    @property
    def model_structure(self) -> str:
        # "llama" or "opt" and so on
        return self.engine_config['builder_config']['name'] if isinstance(
            self.engine_config, dict) else self.engine_config.name


class _ModelBuilder:
    ''' The model builder is used to build an end-to-end model pipeline from a model config.
    It will construct the runtime resources including engine, tokenizer, model runner, etc.
    '''

    def __init__(
        self,
        config: ModelConfig,
        enable_tokenizer: bool,
        tokenizer: Optional[TokenizerBase],
        mapping: Optional[Mapping] = None,
        display_model_processing_summary: bool = False,
    ):
        self.config = config
        self.enable_tokenizer = enable_tokenizer
        self.tokenizer = tokenizer
        self.mapping = mapping
        self.display_model_processing_summary = display_model_processing_summary

        self._model_pipeline = []

        # Prepare the model processing pipeline

        if isinstance(self.config.model, Module):
            ''' Build engine from user provided model '''
            raise NotImplementedError()
            self._model_pipeline.append(
                ("build_engine", self._build_engine_from_inmemory_model))
            return

        self._model_dir = self.config.model_dir
        self._model_name = self.config.model
        self._model_structure = None

        if self.config.model_dir is None:
            ''' Download HF model if necessary '''
            raise NotImplementedError()

            assert self.config.model is not None, "Either model_dir or model should be provided."
            self._model_pipeline.append(
                ("download_hf_model", self._download_hf_model))
            self._model_dir = self._cache_manager.hf_checkpoint_dir()

        self._model_format = self._get_model_format(self._model_dir)
        self._model_name = self._model_name or self._get_model_kind(
            self._model_dir)

        if self._model_format is ModelFormatKind.HF:
            ''' HF -> TFRT checkpoints -> engine '''
            self._model_pipeline.append(
                ("hf_to_trtllm", self._build_model_from_hf))
            self._model_pipeline.append(
                ("build_engine", self._build_engine_and_model_runner))
        elif self._model_format is ModelFormatKind.TLLM_CKPT:
            ''' TFRT checkpoints -> engine '''
            raise NotImplementedError()
        elif self._model_format is ModelFormatKind.TLLM_ENGINE:
            ''' TFRT engine '''
            self._model_pipeline.append(
                ("load_engine", self._load_model_runner))

        if self.enable_tokenizer and not self.tokenizer:
            ''' Use the default tokenizer if user doesn't provide one '''
            self._model_pipeline.append(
                ("init_tokenizer", self._init_default_tokenizer))

    def __call__(self) -> _ModelRuntimeStuff:
        if self.config.is_multi_gpu:
            torch.cuda.set_device(self.mapping.rank)

        for step_name, step in tqdm(self._model_pipeline,
                                    desc="Model preprocessing"):
            # Each step could have a separate progress bar
            # e.g. the download_hf_model step or the build_engine step which is time-consuming
            if self.config.is_multi_gpu:
                # TODO[chunweiy]: here is for debugging, remove later
                print(f"\n#rank-{mpi_rank()} Executing {step_name}")
            else:
                print(f"\nExecuting {step_name}")
            start_time = time.time()
            step()
            end_time = time.time()
        logger.warning(
            f"Finish executing step {step_name} in {end_time - start_time} seconds"
        )

        if self.display_model_processing_summary:
            self._display_summary()

        assert self._model_structure is not None, "The model structure is not set yet."
        assert self.runner is not None, "The model runner is not built yet."

        assert hasattr(self, '_builder_config') or hasattr(
            self, '_engine_config'), "config is not loaded."
        config = self._engine_config if hasattr(
            self, '_engine_config') else self._builder_config

        return _ModelRuntimeStuff(tokenizer=self.tokenizer,
                                  runner=self.runner,
                                  engine=self._engine,
                                  engine_config=config)

    @staticmethod
    def save(model: _ModelRuntimeStuff,
             model_dir: str,
             engine_dir: str,
             mapping=None):
        ''' Save the built engine to the given path.  '''
        from tensorrt_llm.builder import Builder
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

            # TODO[chunweiy]: get dtype from config
            engine_path = engine_dir / get_engine_name(
                model.model_structure, 'float16', mapping.tp_size,
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
                f.write(model.engine)

        def copy_hf_tokenizer_stuff_to_engine_dir():
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
                copy_hf_tokenizer_stuff_to_engine_dir()

    def _get_model_format(self, model_dir: str) -> ModelFormatKind:
        ''' Tell the format of the model.  '''
        # TODO[chunweiy]: refine this
        return ModelFormatKind.HF if Path.exists(
            Path(model_dir) /
            'generation_config.json') else ModelFormatKind.TLLM_ENGINE

    def _get_model_kind(self, model_dir: str) -> str:
        ''' Tell the kind of the model. e.g. "llama" '''
        # TODO[chunweiy]: refine this
        return 'llama'

    def _download_hf_model(self):
        ''' Download HF model from third-party model hub like www.modelscope.cn or huggingface.  '''
        raise NotImplementedError()

    def _build_model_from_hf(self):
        ''' Build a TRT-LLM model from a HF model.  '''
        from tensorrt_llm.models import LLaMAForCausalLM

        # TODO[chunweiy]: inspect from hf model/config
        model_structure = 'LLaMaForCausalLM'

        # TODO[chunweiy]: add more models
        model2struct = dict(LLaMaForCausalLM=LLaMAForCausalLM)

        self.model = model2struct[model_structure].from_hugging_face(
            self._model_dir, mapping=self.mapping)

    def _load_model_runner(self):
        ''' Load a model runner from a TRT-LLM engine.  '''
        assert self._model_dir
        logger.warning(f"Loading model runner from {self._model_dir}")

        self.runner = ModelRunner.from_dir(self._model_dir)
        self._engine = self.runner.session.runtime.engine
        with open(os.path.join(self._model_dir, 'config.json'), 'r') as f:
            self._engine_config = json.load(f)
            self._model_structure = self._engine_config['builder_config'][
                'name']

    def _build_engine_and_model_runner(self):
        ''' Build TensorRT-LLM engine from a in-memory model.
        The model runner will be created.
        '''
        from tensorrt_llm.mapping import Mapping

        self._engine, self._builder_config = self.model.to_trt(
            self.config.build_config.max_batch_size,
            self.config.build_config.max_input_len,
            self.config.build_config.max_output_len)
        self._model_structure = self._builder_config.name

        # TODO[chunweiy]: Fix this, plugin_config should not be None, which triggers OOTB mode
        # plugin_config=self.config.build_config.plugin_config)

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
        self.runner = ModelRunner(session, max_batch_size, max_input_len,
                                  max_output_len, max_beam_width)

    def _init_default_tokenizer(self):
        self.tokenizer = TransformersTokenizer.from_pretrained(
            self._model_dir,
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

    def _display_summary(self):
        ''' Display the summary of the model.
        The following information could be displayed:
        - model kind
        - quantization information
        - runtime setting information
        - cache information
        and so on.
        '''
        raise NotImplementedError()
