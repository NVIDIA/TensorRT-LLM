import copy
import json
import math
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field, fields
from enum import Enum, EnumMeta
from pathlib import Path
from typing import (Any, Callable, ClassVar, Dict, List, Literal, Optional,
                    Tuple, Union)

import torch
import yaml
from pydantic import BaseModel, Field, validator
from transformers import PreTrainedTokenizerBase

from .._utils import mpi_rank
from ..auto_parallel import AutoParallelConfig, infer_cluster_config
# yapf: disable
from ..bindings.executor import BatchingType
from ..bindings.executor import \
    CapacitySchedulerPolicy as _CapacitySchedulerPolicy
from ..bindings.executor import ContextChunkingPolicy as _ContextChunkingPolicy
from ..bindings.executor import DecodingConfig, DecodingMode
from ..bindings.executor import DynamicBatchConfig as _DynamicBatchConfig
from ..bindings.executor import (EagleConfig, ExecutorConfig,
                                 ExtendedRuntimePerfKnobConfig)
from ..bindings.executor import KvCacheConfig as _KvCacheConfig
from ..bindings.executor import \
    LookaheadDecodingConfig as _LookaheadDecodingConfig
from ..bindings.executor import PeftCacheConfig as _PeftCacheConfig
from ..bindings.executor import SchedulerConfig as _SchedulerConfig
# yapf: enable
from ..builder import BuildConfig, EngineConfig
from ..logger import logger
from ..mapping import Mapping
from ..models.automodel import AutoConfig
from ..models.modeling_utils import (PretrainedConfig, QuantAlgo, QuantConfig,
                                     SpeculativeDecodingMode)
from ..sampling_params import BatchedLogitsProcessor
from .build_cache import BuildCacheConfig
from .mpi_session import MpiSession
from .tokenizer import TokenizerBase, tokenizer_factory
# TODO[chunweiy]: move the following symbols back to utils scope, and remove the following import
from .utils import append_docstring


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
        device (Literal['cuda', 'cpu']): The device to run calibration. Defaults to 'cuda'.
        calib_dataset (str): The name or local path of calibration dataset. Defaults to 'cnn_dailymail'.
        calib_batches (int): The number of batches that the calibration runs. Defaults to 512.
        calib_batch_size (int): The batch size that the calibration runs. Defaults to 1.
        calib_max_seq_length (int): The maximum sequence length that the calibration runs. Defaults to 512.
        random_seed (int): The random seed used for calibration. Defaults to 1234.
        tokenizer_max_seq_length (int): The maximum sequence length to initialize tokenizer for calibration. Defaults to 2048.
    """
    device: Literal['cuda', 'cpu'] = 'cuda'
    calib_dataset: str = 'cnn_dailymail'
    calib_batches: int = 512
    calib_batch_size: int = 1
    calib_max_seq_length: int = 512
    random_seed: int = 1234
    tokenizer_max_seq_length: int = 2048

    @classmethod
    def from_dict(cls, config: dict) -> 'CalibConfig':
        """Create a CalibConfig instance from a dict.

        Args:
            config (dict): The dict used to create CalibConfig.

        Returns:
            tensorrt_llm.llmapi.CalibConfig: The CalibConfig created from dict.
        """
        return cls(**config)

    def to_dict(self) -> dict:
        """Dump a CalibConfig instance to a dict.

        Returns:
            dict: The dict dumped from CalibConfig.
        """
        return asdict(self)


class _ModelFormatKind(Enum):
    HF = 0
    TLLM_CKPT = 1
    TLLM_ENGINE = 2


class DecodingBaseConfig(BaseModel):
    max_draft_len: Optional[int] = None
    speculative_model: Optional[Union[str, Path]] = None

    @classmethod
    def from_dict(cls, data: dict):
        # dispatch to the correct decoding config
        decoding_type = data.get("decoding_type")
        config_classes = {
            "MTP": MTPDecodingConfig,
            "Medusa": MedusaDecodingConfig,
            "Eagle": EagleDecodingConfig,
            "Lookahead": LookaheadDecodingConfig
        }

        config_class = config_classes.get(decoding_type)
        if config_class is None:
            raise ValueError(f"Invalid decoding type: {decoding_type}")

        return config_class(**data)

    def _check_fields(self):
        pass


class MedusaDecodingConfig(DecodingBaseConfig):
    medusa_choices: Optional[List[List[int]]] = None
    num_medusa_heads: Optional[int] = None

    @classmethod
    def from_dict(cls, data: dict):
        return cls(**data)

    decoding_type: ClassVar[str] = "Medusa"


class EagleDecodingConfig(DecodingBaseConfig):
    eagle_choices: Optional[List[List[int]]] = None
    greedy_sampling: Optional[bool] = True
    posterior_threshold: Optional[float] = None
    use_dynamic_tree: Optional[bool] = False
    dynamic_tree_max_topK: Optional[int] = None
    num_eagle_layers: Optional[int] = None
    max_non_leaves_per_layer: Optional[int] = None
    pytorch_eagle_weights_path: Optional[str] = None

    @classmethod
    def from_dict(cls, data: dict):
        return cls(**data)

    decoding_type: ClassVar[str] = "Eagle"


class MTPDecodingConfig(DecodingBaseConfig):
    num_nextn_predict_layers: Optional[int] = 1

    @classmethod
    def from_dict(cls, data: dict):
        return cls(**data)

    decoding_type: ClassVar[str] = "MTP"


class PybindMirror(ABC):
    ''' A class containing the utilities for mirroring Python classes to
    pybinding classes.
    '''

    @abstractmethod
    def _to_pybind(self):
        pass

    @staticmethod
    def maybe_to_pybind(ins):
        if isinstance(
                ins,
                PybindMirror) or type(ins).__class__ == PybindMirrorEnumMeta:
            return ins._to_pybind()
        return ins

    @staticmethod
    def mirror_pybind_fields(pybind_class):
        """
        Class decorator that ensures Python class fields mirror those of a C++ class.

        Args:
            pybind_class: The C++ class whose fields should be mirrored

        Returns:
            A decorator function that validates field mirroring
        """

        def decorator(cls):
            assert issubclass(cls, BaseModel)
            # Get all non-private fields from the C++ class
            cpp_fields = PybindMirror.get_pybind_variable_fields(pybind_class)
            python_fields = set(cls.model_fields.keys())

            # Check if all C++ fields exist in the Python class
            for field in cpp_fields:
                if field not in python_fields:
                    raise ValueError(
                        f"Field {field} is not mirrored in Python class {cls.__name__} from C++ class {pybind_class.__name__}. Please update the class."
                    )

            # Return the original class
            return cls

        return decorator

    @staticmethod
    def get_pybind_enum_fields(pybind_class):
        ''' Get all the enum fields from the pybind class. '''
        return [
            f for f in pybind_class.__members__.keys()
            if not f.startswith('_') and not callable(getattr(pybind_class, f))
        ]

    @staticmethod
    def mirror_pybind_enum(pybind_class):
        ''' Mirror the enum fields from the pybind class to the Python class. '''

        def decorator(cls):
            assert issubclass(cls, Enum)
            cpp_fields = PybindMirror.get_pybind_enum_fields(pybind_class)
            python_fields = set(cls.__members__.keys())

            for field in cpp_fields:
                if field not in python_fields:
                    raise ValueError(
                        f"Field {field} is not mirrored in Python class {cls.__name__} from C++ class {pybind_class.__name__}. Please update the class."
                    )
            return cls

        return decorator

    @staticmethod
    def get_pybind_variable_fields(config_cls):
        ''' Get all the variable fields from the pybind class. '''
        return [
            f for f in dir(config_cls)
            if not f.startswith('_') and not callable(getattr(config_cls, f))
        ]

    @staticmethod
    def pybind_equals(obj0, obj1):
        ''' Check if two pybind objects are equal. '''
        assert type(obj0) is type(obj1)
        for field in PybindMirror.get_pybind_variable_fields(type(obj0)):
            if getattr(obj0, field) != getattr(obj1, field):
                return False
        return True


class PybindMirrorMeta(type(PybindMirror)):
    pass


class PybindMirrorEnumMeta(EnumMeta, PybindMirrorMeta):
    """
    Combined metaclass for Enum and PybindMirror.  This is crucial.
    """


@PybindMirror.mirror_pybind_enum(_CapacitySchedulerPolicy)
class CapacitySchedulerPolicy(str, Enum, metaclass=PybindMirrorEnumMeta):
    MAX_UTILIZATION = "MAX_UTILIZATION"
    GUARANTEED_NO_EVICT = "GUARANTEED_NO_EVICT"
    STATIC_BATCH = "STATIC_BATCH"

    def _to_pybind(self):
        return getattr(_CapacitySchedulerPolicy, self.value)


@PybindMirror.mirror_pybind_enum(_ContextChunkingPolicy)
class ContextChunkingPolicy(str, Enum, metaclass=PybindMirrorEnumMeta):
    ''' Context chunking policy. '''
    FIRST_COME_FIRST_SERVED = "FIRST_COME_FIRST_SERVED"
    EQUAL_PROGRESS = "EQUAL_PROGRESS"

    def _to_pybind(self):
        return getattr(_ContextChunkingPolicy, self.value)


@PybindMirror.mirror_pybind_fields(_DynamicBatchConfig)
class DynamicBatchConfig(BaseModel, PybindMirror):
    """Dynamic batch configuration.

    Controls how batch size and token limits are dynamically adjusted at runtime.
    """
    enable_batch_size_tuning: bool = Field(
        description="Controls if the batch size should be tuned dynamically")

    enable_max_num_tokens_tuning: bool = Field(
        description="Controls if the max num tokens should be tuned dynamically"
    )

    dynamic_batch_moving_average_window: int = Field(
        description=
        "The window size for moving average of input and output length which is used to calculate dynamic batch size and max num tokens"
    )

    def _to_pybind(self):
        return _DynamicBatchConfig(
            enable_batch_size_tuning=self.enable_batch_size_tuning,
            enable_max_num_tokens_tuning=self.enable_max_num_tokens_tuning,
            dynamic_batch_moving_average_window=self.
            dynamic_batch_moving_average_window)


@PybindMirror.mirror_pybind_fields(_SchedulerConfig)
class SchedulerConfig(BaseModel, PybindMirror):
    capacity_scheduler_policy: CapacitySchedulerPolicy = Field(
        default=CapacitySchedulerPolicy.GUARANTEED_NO_EVICT,
        description="The capacity scheduler policy to use")

    context_chunking_policy: Optional[ContextChunkingPolicy] = Field(
        default=None, description="The context chunking policy to use")

    dynamic_batch_config: Optional[DynamicBatchConfig] = Field(
        default=None, description="The dynamic batch config to use")

    def _to_pybind(self):
        return _SchedulerConfig(
            capacity_scheduler_policy=self.capacity_scheduler_policy._to_pybind(
            ),
            context_chunking_policy=self.context_chunking_policy._to_pybind()
            if self.context_chunking_policy else None,
            dynamic_batch_config=self.dynamic_batch_config._to_pybind()
            if self.dynamic_batch_config else None)


@PybindMirror.mirror_pybind_fields(_PeftCacheConfig)
class PeftCacheConfig(BaseModel, PybindMirror):
    """
    Configuration for the PEFT cache.
    """
    num_host_module_layer: int = Field(
        default=0,
        description=
        "number of max sized 1-layer 1-module adapterSize=1 sets of weights that can be stored in host cache"
    )
    num_device_module_layer: int = Field(
        default=0,
        description=
        "number of max sized 1-layer 1-module sets of weights that can be stored in host cache"
    )
    optimal_adapter_size: int = Field(
        default=
        8,  # There are tests to keep the default value consistent with the pybind default value
        description="optimal adapter size used to set page width")
    max_adapter_size: int = Field(
        default=64,
        description="max supported adapter size. Used to compute minimum")
    num_put_workers: int = Field(
        default=1,
        description=
        "number of worker threads used to put weights into host cache")
    num_ensure_workers: int = Field(
        default=1,
        description=
        "number of worker threads used to copy weights from host to device")
    num_copy_streams: int = Field(
        default=1,
        description="number of streams used to copy weights from host to device"
    )
    max_pages_per_block_host: int = Field(
        default=24,
        description="Number of cache pages per allocation block (host)")
    max_pages_per_block_device: int = Field(
        default=8,
        description="Number of cache pages per allocation block (device)")
    device_cache_percent: Optional[float] = Field(
        default=None,
        description="percent of memory after engine load to use for cache")
    host_cache_size: Optional[int] = Field(
        default=None, description="size in bytes to use for host cache")
    lora_prefetch_dir: Optional[str] = Field(
        default=None,
        description=
        "folder to store the LoRA weights we hope to load during engine initialization"
    )

    def _to_pybind(self):
        return _PeftCacheConfig(
            num_host_module_layer=self.num_host_module_layer,
            num_device_module_layer=self.num_device_module_layer,
            optimal_adapter_size=self.optimal_adapter_size,
            max_adapter_size=self.max_adapter_size,
            num_put_workers=self.num_put_workers,
            num_ensure_workers=self.num_ensure_workers,
            num_copy_streams=self.num_copy_streams,
            max_pages_per_block_host=self.max_pages_per_block_host,
            max_pages_per_block_device=self.max_pages_per_block_device,
            device_cache_percent=self.device_cache_percent,
            host_cache_size=self.host_cache_size,
            lora_prefetch_dir=self.lora_prefetch_dir)


@PybindMirror.mirror_pybind_fields(_LookaheadDecodingConfig)
class LookaheadDecodingConfig(DecodingBaseConfig, PybindMirror):
    """
    Configuration for lookahead speculative decoding.
    """

    max_window_size: int = Field(
        default=_LookaheadDecodingConfig.get_default_lookahead_decoding_window(
        ),
        description="Number of NGrams in lookahead branch per step.")
    max_ngram_size: int = Field(
        default=_LookaheadDecodingConfig.get_default_lookahead_decoding_ngram(),
        description="Number of tokens per NGram.")
    max_verification_set_size: int = Field(
        default=_LookaheadDecodingConfig.
        get_default_lookahead_decoding_verification_set(),
        description="Number of NGrams in verification branch per step.")

    @validator('max_window_size', 'max_ngram_size', 'max_verification_set_size')
    def validate_positive_values(cls, v):
        if v <= 0:
            raise ValueError(f"Value must be positive, got {v}")
        return v

    def __init__(self, **data):
        super().__init__(**data)
        self._check_fields()

    def calculate_speculative_resource(self):
        return _LookaheadDecodingConfig.calculate_speculative_resource_tuple(
            self.max_window_size, self.max_ngram_size,
            self.max_verification_set_size)

    @classmethod
    def from_dict(cls, data: dict):
        return cls(**data)

    def _to_pybind(self):
        return _LookaheadDecodingConfig(self.max_window_size,
                                        self.max_ngram_size,
                                        self.max_verification_set_size)

    decoding_type: ClassVar[str] = "Lookahead"


@PybindMirror.mirror_pybind_fields(_KvCacheConfig)
class KvCacheConfig(BaseModel, PybindMirror):
    """
    Configuration for the KV cache.
    """
    enable_block_reuse: bool = Field(
        default=True,
        description=
        "Controls if KV cache blocks can be reused for different requests.")
    max_tokens: Optional[int] = Field(
        default=None,
        description=
        "The maximum number of tokens that should be stored in the KV cache. If both `max_tokens` and `free_gpu_memory_fraction` are specified, memory corresponding to the minimum will be used."
    )
    max_attention_window: Optional[List[int]] = Field(
        default=None,
        description=
        "Size of the attention window for each sequence. Only the last tokens will be stored in the KV cache. If the number of elements in `max_attention_window` is less than the number of layers, `max_attention_window` will be repeated multiple times to the number of layers."
    )
    sink_token_length: Optional[int] = Field(
        default=None,
        description=
        "Number of sink tokens (tokens to always keep in attention window).")
    free_gpu_memory_fraction: Optional[float] = Field(
        default=None,
        description=
        "The fraction of GPU memory fraction that should be allocated for the KV cache. Default is 90%. If both `max_tokens` and `free_gpu_memory_fraction` are specified, memory corresponding to the minimum will be used."
    )
    host_cache_size: Optional[int] = Field(
        default=None,
        description=
        "Size of the host cache in bytes. If both `max_tokens` and `host_cache_size` are specified, memory corresponding to the minimum will be used."
    )
    onboard_blocks: bool = Field(
        default=True, description="Controls if blocks are onboarded.")
    cross_kv_cache_fraction: Optional[float] = Field(
        default=None,
        description=
        "The fraction of the KV Cache memory should be reserved for cross attention. If set to p, self attention will use 1-p of KV Cache memory and cross attention will use p of KV Cache memory. Default is 50%. Should only be set when using encoder-decoder model."
    )
    secondary_offload_min_priority: Optional[int] = Field(
        default=None,
        description=
        "Only blocks with priority > mSecondaryOfflineMinPriority can be offloaded to secondary memory."
    )
    event_buffer_max_size: int = Field(
        default=0,
        description=
        "Maximum size of the event buffer. If set to 0, the event buffer will not be used."
    )
    enable_partial_reuse: bool = Field(
        default=True,
        description=
        "Whether blocks that are only partially matched can be reused.")
    copy_on_partial_reuse: bool = Field(
        default=True,
        description=
        "Whether partially matched blocks that are in use can be reused after copying them."
    )

    def _to_pybind(self):
        return _KvCacheConfig(
            enable_block_reuse=self.enable_block_reuse,
            max_tokens=self.max_tokens,
            max_attention_window=self.max_attention_window,
            sink_token_length=self.sink_token_length,
            free_gpu_memory_fraction=self.free_gpu_memory_fraction,
            host_cache_size=self.host_cache_size,
            onboard_blocks=self.onboard_blocks,
            cross_kv_cache_fraction=self.cross_kv_cache_fraction,
            secondary_offload_min_priority=self.secondary_offload_min_priority,
            event_buffer_max_size=self.event_buffer_max_size,
            enable_partial_reuse=self.enable_partial_reuse,
            copy_on_partial_reuse=self.copy_on_partial_reuse)


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
LLMARGS_EXPLICIT_DOCSTRING = """
        model (str, pathlib.Path): The model name or a local model directory.
            Note that if the value could be both a model name or a local model directory, the local model directory will be prioritized.

        tokenizer (str, pathlib.Path, transformers.PreTrainedTokenizerBase, tensorrt_llm.llmapi.tokenizer.TokenizerBase, optional):
            The name or path of a HuggingFace Transformers tokenizer, or the loaded tokenizer. Defaults to None.

        tokenizer_mode (Literal['auto', 'slow']): The tokenizer mode. Defaults to 'auto'.
            'auto' will use the fast tokenizer if available, and 'slow' will always use the slow tokenizer.
            The fast tokenizer is based on Huggingface's Rust library tokenizers, which achieves a significant speed-up compared to its slow counterpart.

        skip_tokenizer_init (bool): Whether to skip initialization of tokenizer and detokenizer. Defaults to False.
            LLM.generate and LLM.generate_async will accept prompt token ids as input only.

        trust_remote_code (bool): Whether to trust remote code when downloading model and tokenizer from Hugging Face. Defaults to False.

        tensor_parallel_size(int): The number of processes for tensor parallelism. Defaults to 1.

        dtype (str): The data type for the model weights and activations. Defaults to "auto".
            Can be "float16", "bfloat16", "float32", or "auto". If "auto", the data type will be automatically inferred from the source model.
            If the source data type is "float32", it will be converted to "float16".

        revision (str, optional): The revision of the model to use. Defaults to None.

        tokenizer_revision (str, optional): The revision of the tokenizer to use. Defaults to None.
"""

LLMARGS_IMPLICIT_DOCSTRING = """
        pipeline_parallel_size(int): The number of processes for pipeline parallelism. Defaults to 1.

        context_parallel_size (int): The context parallel size. Defaults to 1.

        gpus_per_node (int, optional): The number of GPUs per node. None means automatic configure. Defaults to None.

        load_format (Literal['auto', 'dummy']): The format of the model weights to load. Defaults to 'auto'.
            * 'auto' will try to load the weights from the provided checkpoint.
            * 'dummy' will initialize the weights with random values, which is mainly for profiling.

        enable_tqdm (bool): Whether to display a progress bar during model building. Defaults to False.

        enable_lora (bool): Enable LoRA adapters. Defaults to False.

        max_lora_rank (int, optional): Maximum LoRA rank. If specified, it overrides `build_config.lora_config.max_lora_rank`. Defaults to None.

        max_loras (int): Maximum number of LoRA adapters to be stored in GPU memory. Defaults to 4.

        max_cpu_loras (int): Maximum number of LoRA adapters to be stored in CPU memory. Defaults to 4.

        enable_prompt_adapter (bool): Enable prompt adapters. Defaults to False.

        max_prompt_adapter_token (int): Maximum number of prompt adapter tokens. Defaults to 0.

        quant_config (tensorrt_llm.llmapi.QuantConfig, optional): The quantization configuration for the model. Defaults to None.

        calib_config (tensorrt_llm.llmapi.CalibConfig, optional): The calibration configuration for the model. Defaults to None.

        build_config (tensorrt_llm.llmapi.BuildConfig, optional): The build configuration for the model. Defaults to None.

        kv_cache_config (tensorrt_llm.llmapi.llm_args.KvCacheConfig, optional): The key-value cache configuration for the model. Defaults to None.

        enable_chunked_prefill (bool): Whether to enable chunked prefill. Defaults to False.

        guided_decoding_backend (str, optional): The guided decoding backend, currently supports 'xgrammar'. Defaults to None.

        batched_logits_processor (tensorrt_llm.sampling_params.BatchedLogitsProcessor, optional): The batched logits postprocessor callback. Defaults to None.
            The BatchedLogitsProcessor class is recommended for callback creation.

        iter_stats_max_iterations (int, optional): The maximum number of iterations for iteration statistics. Defaults to None.

        request_stats_max_iterations (int, optional): The maximum number of iterations for request statistics. Defaults to None.

        workspace (str, optional): The directory to store intermediate files. Defaults to None.

        embedding_parallel_mode (str): The parallel mode for embeddings. Defaults to 'SHARDING_ALONG_VOCAB'.

        auto_parallel (bool): Enable auto parallel mode. Defaults to False.

        auto_parallel_world_size (int): The MPI world size for auto parallel. Defaults to 1.

        moe_tensor_parallel_size (int, optional): The tensor parallel size for MoE models's expert weights. Defaults to None.

        moe_expert_parallel_size (int, optional): The expert parallel size for MoE models's expert weights. Defaults to None.

        enable_attention_dp (bool): Enable attention data parallel. Defaults to False.

        cp_config (dict, optional): Context parallel config. Defaults to None.

        fast_build (bool): Enable features for faster engine building. Defaults to False.
            This may cause some performance degradation and is currently incompatible with int8/int4 quantization.

        enable_build_cache (bool, tensorrt_llm.llmapi.BuildCacheConfig): Whether to enable build caching for the model. Defaults to False.

        peft_cache_config (tensorrt_llm.llmapi.llm_args.PeftCacheConfig, optional): The PEFT cache configuration for the model. Defaults to None.

        scheduler_config (tensorrt_llm.llmapi.llm_args.SchedulerConfig, optional): The scheduler configuration for the model. Defaults to None.

        speculative_config (tensorrt_llm.llmapi.llm_args.LookaheadDecodingConfig, tensorrt_llm.llmapi.MedusaDecodingConfig, tensorrt_llm.llmapi.EagleDecodingConfig, tensorrt_llm.llmapi.MTPDecodingConfig, optional): The speculative decoding configuration. Defaults to None.

        decoding_config (tensorrt_llm.bindings.executor.DecodingConfig, optional): The decoding configuration for the model. Defaults to None.

        batching_type (tensorrt_llm.bindings.executor.BatchingType, optional): The batching type for the model. Defaults to None.

        normalize_log_probs (bool): Whether to normalize log probabilities for the model. Defaults to False.

        gather_generation_logits (bool): Enable gathering generation logits. Defaults to False.

        max_input_len (int): The maximum input length allowed for the model. Defaults to 1024.

        max_seq_len (int): The maximum sequence length for generation. Defaults to None.

        max_beam_width (int): The maximum beam width used in beam search. Defaults to 1.

        max_batch_size (int, optional): The maximum batch size for runtime. Defaults to None.

        max_num_tokens (int, optional): The maximum number of tokens for runtime. Defaults to None.

        extended_runtime_perf_knob_config (tensorrt_llm.bindings.executor.ExtendedRuntimePerfKnobConfig, optional): The extended runtime performance knob configuration for the model. Defaults to None.

        backend (str, optional): The backend to use. None means TensorRT engine and C++ executor. Defaults to None.
"""


@append_docstring(LLMARGS_EXPLICIT_DOCSTRING + LLMARGS_IMPLICIT_DOCSTRING)
@dataclass
class LlmArgs:
    """The arguments for constructing a LLM instance.

    Args:
    """
    # Explicit arguments
    model: Union[str, Path]

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

    cp_config: Optional[dict] = None

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

    guided_decoding_backend: Optional[str] = None

    batched_logits_processor: Optional[BatchedLogitsProcessor] = None

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
    speculative_config: Optional[Union[LookaheadDecodingConfig,
                                       MedusaDecodingConfig,
                                       EagleDecodingConfig,
                                       MTPDecodingConfig]] = None

    decoding_config: Optional[DecodingConfig] = None

    batching_type: Optional[BatchingType] = None

    normalize_log_probs: bool = False

    gather_generation_logits: bool = False

    extended_runtime_perf_knob_config: Optional[
        ExtendedRuntimePerfKnobConfig] = None

    # TODO: remove this option in the future
    _use_runtime_defaults: bool = True

    # generation constraints
    max_input_len: int = 1024

    max_seq_len: int = None

    max_beam_width: int = 1

    max_batch_size: Optional[int] = None

    max_num_tokens: Optional[int] = None

    # backend to use
    backend: Optional[str] = None

    # Optional mpi session to use for this LLM instance
    _mpi_session: Optional[MpiSession] = None

    # private options
    _num_postprocess_workers: int = 0  # Number of postprocess worker processes
    _postprocess_tokenizer_dir: Optional[str] = None

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

        if self.cp_config is None:
            self.co_config = {}

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
    def from_kwargs(cls, **kwargs: Any) -> "LlmArgs":
        """Create `LlmArgs` instance from kwargs.

        Args:
            kwargs (Any): Arguments passed to `LlmArgs` constructor.

        Returns:
            tensorrt_llm.llmapi.llm_utils.LlmArgs: The `LlmArgs` instance.
        """
        kwargs = LlmArgs._maybe_update_config_for_consistency(dict(kwargs))
        ret = cls(**kwargs)
        ret._setup()
        return ret

    def to_dict(self) -> dict:
        """Dump `LlmArgs` instance to a dict.

        Returns:
            dict: The dict that contains all fields of the `LlmArgs` instance.
        """
        return dict(
            (field.name, getattr(self, field.name)) for field in fields(self))

    @staticmethod
    def _maybe_update_config_for_consistency(
            kwargs_dict: Dict[str, Any]) -> Dict[str, Any]:
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

        # ensure build_config and LlmArgs consistency
        if kwargs_dict.get("backend") != "pytorch" and kwargs_dict.get(
                "build_config"):
            # TODO: move this to _perform_config_arbitration() once it's default-on.
            for field_name in [
                    "max_input_len", "max_seq_len", "max_beam_width"
            ]:
                build_val = getattr(kwargs_dict["build_config"], field_name,
                                    None)
                llmargs_val = kwargs_dict.get(field_name) or getattr(
                    LlmArgs, field_name)

                if build_val != llmargs_val:
                    logger.warning(
                        f"Overriding LlmArgs.{field_name} ({llmargs_val}) with build_config.{field_name} ({build_val})."
                    )
                    kwargs_dict[field_name] = build_val

        return kwargs_dict

    def _setup(self):
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

        self.speculative_model = getattr(self.speculative_config,
                                         "speculative_model", None)
        speculative_model_obj = _ModelWrapper(
            self.speculative_model
        ) if self.speculative_model is not None else None
        if model_obj.is_local_model and self.backend not in [
                'pytorch', 'autodeploy'
        ]:
            # Load parallel_config from the engine.
            self.model_format = get_model_format(self.model)

            if self.model_format is _ModelFormatKind.TLLM_ENGINE:
                if self.build_config is not None:
                    logger.warning(
                        "The build_config is ignored for model format of TLLM_ENGINE."
                    )
                self._load_config_from_engine(model_obj.model_dir)
                runtime_defaults = self._pretrained_config.runtime_defaults
                if self._use_runtime_defaults and runtime_defaults:
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
                    lookahead_decoding_config=PybindMirror.maybe_to_pybind(
                        lookahead_config))
            elif isinstance(self.speculative_config, MedusaDecodingConfig):
                self.build_config.speculative_decoding_mode = SpeculativeDecodingMode.MEDUSA

                assert self.speculative_config.max_draft_len > 0
                self.build_config.max_draft_len = self.speculative_config.max_draft_len
                self.decoding_config = DecodingConfig(
                    decoding_mode=DecodingMode.Medusa(),
                    medusa_choices=self.speculative_config.medusa_choices)
            elif isinstance(self.speculative_config, EagleDecodingConfig):
                self.build_config.speculative_decoding_mode = SpeculativeDecodingMode.EAGLE
                assert self.speculative_config.max_draft_len > 0

                self.build_config.max_draft_len = self.speculative_config.max_draft_len

                if self.backend != 'pytorch':
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
                    from tensorrt_llm._torch.speculative import Eagle3Config
                    self.speculative_config = Eagle3Config(
                        max_draft_tokens=self.speculative_config.max_draft_len,
                        eagle_weights_path=self.speculative_config.
                        pytorch_eagle_weights_path)

            elif isinstance(self.speculative_config, MTPDecodingConfig):
                from tensorrt_llm._torch.speculative import MTPConfig
                self.speculative_config = MTPConfig(
                    num_nextn_predict_layers=self.speculative_config.
                    num_nextn_predict_layers,
                    max_batch_size=self.build_config.max_batch_size)
            else:
                raise ValueError(
                    f"Speculative config type not recognized: {self.speculative_config}"
                )
        else:
            self.decoding_config = None

    def _perform_config_arbitration(self):
        '''
        Arbitrate the configurations for the model building. The configs between different functional or performance
        features might be conflicted, and this method will arbitrate the conflicts and raise errors if necessary.
        '''
        self._config_arbitrator = _ConfigArbitrator()
        if self._build_config_mutable:
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
    def _build_config_mutable(self) -> bool:
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
            if self._build_config_mutable:
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


def update_llm_args_with_extra_dict(
        llm_args: Dict,
        llm_args_dict: Dict,
        extra_llm_api_options: Optional[str] = None) -> Dict:

    from .._torch.pyexecutor.config import PyTorchConfig
    field_mapping = {
        "quant_config": QuantConfig,
        "calib_config": CalibConfig,
        "build_config": BuildConfig,
        "kv_cache_config": KvCacheConfig,
        "decoding_config": DecodingConfig,
        "enable_build_cache": BuildCacheConfig,
        "peft_cache_config": PeftCacheConfig,
        "scheduler_config": SchedulerConfig,
        "speculative_config": DecodingBaseConfig,
        "batching_type": BatchingType,
        "extended_runtime_perf_knob_config": ExtendedRuntimePerfKnobConfig,
        "pytorch_backend_config": PyTorchConfig,
    }
    for field, field_type in field_mapping.items():
        if field in llm_args_dict:
            if field == "speculative_config":
                llm_args_dict[field] = field_type.from_dict(
                    llm_args_dict[field])
            else:
                llm_args_dict[field] = field_type(**llm_args_dict[field])
            extra_llm_str = f"because it's specified in {extra_llm_api_options}" if extra_llm_api_options else ""
            logger.warning(f"Overriding {field} {extra_llm_str}")

    llm_args = llm_args | llm_args_dict
    return llm_args


def update_llm_args_with_extra_options(llm_args: Dict,
                                       extra_llm_api_options: str) -> Dict:
    if extra_llm_api_options is not None:
        with open(extra_llm_api_options, 'r') as f:
            llm_args_dict = yaml.safe_load(f)
            llm_args = update_llm_args_with_extra_dict(llm_args, llm_args_dict,
                                                       extra_llm_api_options)
    return llm_args


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
