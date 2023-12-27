import logging
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Callable, ClassVar, Iterable, List, Optional, Tuple, Union

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from tensorrt_llm import Module, logger
from tensorrt_llm.builder import BuildConfig
from tensorrt_llm.runtime import (GenerationSession, ModelRunner,
                                  SamplingConfig, model_runner)


@dataclass
class ParallelConfig:
    ''' The model distribution configs for LLM.  '''
    tp_size: int = 1
    pp_size: int = 1
    devices: List[int] = field(default_factory=list, init=False)

    def __post_init__(self):
        # some check about the parameters
        pass


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
    #quant_config: QuantMode = QuantMode()

    # ``parallel_config`` contains the options for distributed inference.
    parallel_config: ParallelConfig = ParallelConfig()

    def __post_init__(self):
        assert self.model or self.model_dir, "Either model or model_dir should be provided."


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

    For performance issue, one can disable the tokenizer and postprocessing to make the llm takes input ids directly and
    return output ids directly.

    llm.disable_tokenizer()
    llm([32, 12, 32]) # => [32, 12, 32, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    '''
    config: ModelConfig

    # user provided tokenizer, will override the default one
    tokenizer: Optional[PreTrainedTokenizerBase] = None

    # Turn on the preprocessing and postprocessing with a tokenizer to make the llm pipeline takes texts as input and produces text as output.
    # If turned off, the llm pipeline will take token ids as input and produce token ids as output.
    enable_tokenizer: bool = True

    # Disable downloading the HF model from third-party model hub like www.modelscope.cn or huggingface.
    # Useful when network is not available and force to use a local model.
    disable_model_download: bool = False

    # Display the summary of the model building.
    display_model_processing_summary: bool = False

    # Dump the summary of the model building into a log file.
    dump_model_processing_summary: Optional[str] = None

    # ======================== runtime members =========================
    _model_pipeline: List[Tuple[str, Callable]] = field(default_factory=list,
                                                        init=False)

    # a cache manager is used to manage the cache of the model formats, like TensorRT-LLM checkpoints or engines.
    # _cache_manager: "CacheManager" = field(default=None, init=False)

    def __post_init__(self):

        # 1. Prepare the model processing pipeline
        if isinstance(self.config.model, Module):
            ''' Build engine from user provided model '''
            raise NotImplementedError()
            self._model_pipeline.append(
                ("build_engine", self._build_engine_from_inmemory_model))

        else:
            ''' Build engine from model_dir or downloaded HF model. '''
            self._model_dir = self.config.model_dir
            self._model_name = self.config.model

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

        # TODO[chunweiy]: Concerning quantization step, since the weight_only quantization is embedded in the engine building phase,
        # and the FP8 quantization is supported by AMMO which directly produce a TRT-LLM checkpoint.
        # It is vague whether quantization should be a separate step. But it can be added here as a step if necessary.

        # 2. Execute the model processing pipeline and display the progress and timing to keep users patient
        for step_name, step in tqdm(self._model_pipeline,
                                    desc="Model preprocessing"):
            # Each step could have a separate progress bar
            # e.g. the download_hf_model step or the build_engine step which is time-consuming
            print(f"\nExecuting {step_name}")
            start_time = time.time()
            step()
            end_time = time.time()
        logging.warning(
            f"Finish executing step {step_name} in {end_time - start_time} seconds"
        )

        # 3. The model preprocessing is finished, display some summary information for double check
        if self.display_model_processing_summary:
            self._display_summary()

        self.default_sampling_config = SamplingConfig(
            end_id=self.tokenizer.eos_token_id,
            pad_id=self.tokenizer.eos_token_id if self.tokenizer.pad_token_id is
            None else self.tokenizer.pad_token_id,
            output_sequence_lengths=True,
            return_dict=True) if self.tokenizer else None

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
        assert self.runner is not None, "The engine is not built yet."

        sampling_config = sampling_config or self.default_sampling_config
        assert sampling_config is not None, "The sampling_config need to be provided."

        return self._generate_sync(prompts, sampling_config)

    def _generate_sync(self, prompts,
                       sampling_config) -> Iterable[GenerationOuptut]:
        ''' Generate in sync mode on a single GPU.  '''
        if not prompts: return []
        assert self.runner, "The model runner is not built yet."

        need_tokenize: bool = isinstance(prompts[0], str)
        if need_tokenize:
            assert self.tokenizer, "The tokenizer is not built or provided."

        build_config = self.config.build_config

        def generate_batch(batch_input_ids: List[torch.Tensor]):
            batch_input_ids = [
                torch.tensor(x, dtype=torch.int32) for x in batch_input_ids
            ]  # List[torch.Tensor(seq)]

            assert len(batch_input_ids) <= build_config.max_batch_size, \
                f"Can not run batch size larger than {build_config.max_batch_size}, got {len(batch_input_ids)}"
            outputs = self.runner.generate(batch_input_ids, sampling_config)

            # parse and print output
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

                    output_text = self.tokenizer.decode(
                        outputs) if self.tokenizer else None

                    # get a sequence for each prompt directly
                    piece = GenerationPiece(text=output_text, token_ids=outputs)
                    yield GenerationOuptut(generate_pieces=[piece])

        def batching_prompts():
            process_batch = lambda batch: self.tokenizer.batch_encode_plus(
                batch)['input_ids'] if need_tokenize else batch
            batch = []
            for i, prompt in enumerate(prompts):
                batch.append(prompt)
                if len(batch) >= build_config.max_batch_size:
                    yield process_batch(batch)
                    batch = []
            if batch:
                yield process_batch(batch)

        for batch in batching_prompts():
            outs = generate_batch(batch)
            for o in outs:
                yield o

    def save(self, engine_dir: str):
        ''' Save the built engine to the given path.  '''
        from tensorrt_llm.builder import Builder

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

            # TODO[chunweiy]: refine this
            engine_path = engine_dir / get_engine_name(
                self._builder_config.name, 'float16', 1, 1, 0)
            builder = Builder()
            builder.save_config(self._builder_config, config_path)
            with open(engine_path, 'wb') as f:
                f.write(self._engine)

        def copy_hf_tokenizer_stuff_to_engine_dir():
            # Copy the HF tokenizer stuff to the engine dir so that we can use the engine dir as a standalone model dir supports end-to-end task.
            # This is only for HF model for now, not available for users' customized tokenizers.
            import shutil
            for name in os.listdir(self._model_dir):
                src = os.path.join(self._model_dir, name)
                dst = os.path.join(engine_dir, name)
                if name.startswith('tokenizer'):
                    if os.path.isdir(src):
                        shutil.copytree(src, dst, dirs_exist_ok=True)
                    else:
                        shutil.copy2(src, dst)

        save_engine_to_dir(engine_dir)
        if isinstance(self.tokenizer, TransformersTokenizer):
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
            self._model_dir)

    def _load_model_runner(self):
        ''' Load a model runner from a TRT-LLM checkpoints.  '''
        assert self._model_dir
        logger.warning(f"Loading model runner from {self._model_dir}")
        self.runner = ModelRunner.from_dir(self._model_dir)

    def _build_engine_and_model_runner(self):
        ''' Build TensorRT-LLM engine from a in-memory model.
        The model runner will be created.
        '''
        from tensorrt_llm.mapping import Mapping

        print('start build engine')
        # TODO[chunweiy]: Enhance this, the to_trt should describe what arguments it needs
        # TODO[chunweiy]: Is the builder_config necessary?
        self._engine, self._builder_config = self.model.to_trt(
            self.config.build_config.max_batch_size,
            self.config.build_config.max_input_len,
            self.config.build_config.max_output_len)
        # TODO[chunweiy]: Fix this.
        #plugin_config=self.config.build_config.plugin_config)

        # delete the model explicitly to free all the build-time resources
        del self.model

        # TODO [chunweiy]: Is this conversion necessary?
        model_config, other_config = model_runner._builder_to_model_config(
            self._builder_config.to_dict())
        world_size = other_config.get('world_size')
        tp_size = other_config.get('tp_size')
        pp_size = other_config.get('pp_size')
        assert world_size == tp_size == pp_size == 1, "Multi GPU support is not implemented yet"
        max_batch_size = other_config.get('max_batch_size')
        max_input_len = other_config.get('max_input_len')
        max_output_len = other_config.get('max_output_len')
        max_beam_width = other_config.get('max_beam_width')
        rank = 0  #TODO: should from some where in the runtime when supporting multi gpus
        runtime_mapping = Mapping(world_size=world_size,
                                  rank=rank,
                                  tp_size=tp_size,
                                  pp_size=pp_size)
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


@dataclass
class CacheManager:
    # TODO[chunweiy]: Add cache manager to manage the cache of the model formats, like TensorRT-LLM checkpoints or engines.
    cache_root: ClassVar[str] = "~/.cache/tensorrt-llm"

    def get_model_download_dir(self, model_id: str):
        return os.path.join(self.cache_root, model_id)
