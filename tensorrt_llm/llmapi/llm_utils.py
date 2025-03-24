import copy
import json
import os
import shutil
import tempfile
import time
import weakref
from argparse import Namespace
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, List, Optional, Tuple, Union

import torch
from tqdm import tqdm

from .._utils import (global_mpi_rank, mpi_barrier, mpi_broadcast, mpi_rank,
                      release_gc)
from ..auto_parallel import AutoParallelConfig
# yapf: disable
from ..bindings.executor import (BatchingType, CapacitySchedulerPolicy,
                                 ContextChunkingPolicy, ExecutorConfig,
                                 KvCacheRetentionConfig, SchedulerConfig)
# yapf: enable
from ..builder import BuildConfig, Engine, build
from ..logger import logger
from ..mapping import Mapping
from ..models.automodel import MODEL_MAP, AutoConfig, AutoModelForCausalLM
from ..models.modeling_utils import PretrainedConfig, QuantAlgo, QuantConfig
from ..module import Module
from .build_cache import (BuildCache, BuildCacheConfig, CachedStage,
                          get_build_cache_config_from_env)
from .llm_args import (CalibConfig, ConfigArbitrateError, EagleDecodingConfig,
                       KvCacheConfig, LlmArgs, LookaheadDecodingConfig,
                       MedusaDecodingConfig, MTPDecodingConfig,
                       _ConfigArbitrator, _ModelFormatKind, _ModelWrapper,
                       _ParallelConfig, get_model_format,
                       update_llm_args_with_extra_dict,
                       update_llm_args_with_extra_options)
from .mpi_session import MPINodeState, MpiSession
from .tokenizer import TransformersTokenizer, load_hf_tokenizer
# TODO[chunweiy]: move the following symbols back to utils scope, and remove the following import
from .utils import (download_hf_model, download_hf_pretrained_config,
                    enable_llm_debug, get_directory_size_in_gb, print_colored,
                    print_traceback_on_error)


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
        self.global_rank = global_mpi_rank()
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
            torch.cuda.set_device(self.global_rank % self.mapping.gpus_per_node)

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
        self.save(runtime_context, self.model_obj.model_dir, engine_dir)
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

    def save(
        self,
        model: _ModelRuntimeContext,
        model_dir: str,
        engine_dir: str,
    ):
        ''' Save the built engine on a single GPU to the given path. '''
        model.engine.save(engine_dir)
        if model.mapping.rank == 0:
            tokenizer = ModelLoader.load_hf_tokenizer(
                model_dir,
                trust_remote_code=self.llm_args.trust_remote_code,
                use_fast=self.llm_args.tokenizer_mode != 'slow')
            if tokenizer is not None:
                tokenizer.save_pretrained(engine_dir)

    @staticmethod
    def get_model_format(model_dir: str) -> _ModelFormatKind:
        ''' Get the format of the model.  '''
        return get_model_format(model_dir)

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
        elif self.llm_args.quant_config._requires_calibration and not already_quantized:
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
            logger.warning(f"Failed to load tokenizer from {model_dir}")
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
            return Path(self.llm_args.model), None

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
            if self.llm_args.backend not in ["pytorch", "autodeploy"]:
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
    'KvCacheRetentionConfig',
    'LookaheadDecodingConfig',
    'MedusaDecodingConfig',
    'MTPDecodingConfig',
    'ContextChunkingPolicy',
    'CapacitySchedulerPolicy',
    'BuildConfig',
    'BuildCacheConfig',
    'QuantConfig',
    'CalibConfig',
    'KvCacheConfig',
    'CachedModelLoader',
    'ConfigArbitrateError',
    '_ConfigArbitrator',
    'EagleDecodingConfig',
    'update_llm_args_with_extra_dict',
    'update_llm_args_with_extra_options',
]
