import copy
import json
import math
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, EnumMeta
from pathlib import Path
from typing import (TYPE_CHECKING, Any, ClassVar, Dict, List, Literal, Optional,
                    Union)

import torch
import yaml
from pydantic import (BaseModel, Field, PrivateAttr, field_validator,
                      model_validator)
from strenum import StrEnum
from transformers import PreTrainedTokenizerBase

from tensorrt_llm.lora_manager import (LoraConfig,
                                       get_default_trtllm_modules_to_hf_modules)

from .._utils import mpi_rank
from ..auto_parallel import AutoParallelConfig, infer_cluster_config

if TYPE_CHECKING:
    from tensorrt_llm._torch.pyexecutor.config import PyTorchConfig

# yapf: disable
# isort: off
from ..bindings.executor import (
                                 BatchingType as _BatchingType,
                                 CacheTransceiverConfig as _CacheTransceiverConfig,
                                 CapacitySchedulerPolicy as _CapacitySchedulerPolicy,
                                 ContextChunkingPolicy as _ContextChunkingPolicy,
                                 DecodingConfig,
                                 DecodingMode,
                                 DynamicBatchConfig as _DynamicBatchConfig,
                                 EagleConfig as _EagleConfig,
                                 ExecutorConfig as _ExecutorConfig,
                                 ExtendedRuntimePerfKnobConfig as _ExtendedRuntimePerfKnobConfig,
                                 KvCacheConfig as _KvCacheConfig,
                                 LookaheadDecodingConfig as _LookaheadDecodingConfig,
                                 PeftCacheConfig as _PeftCacheConfig,
                                 SchedulerConfig as _SchedulerConfig) # isort: skip
# isort: on

# yapf: enable
from ..builder import BuildConfig, EngineConfig
from ..logger import logger
from ..mapping import Mapping
from ..models.automodel import AutoConfig
from ..models.modeling_utils import (PretrainedConfig, QuantAlgo, QuantConfig,
                                     SpeculativeDecodingMode)
from ..sampling_params import BatchedLogitsProcessor
from .build_cache import BuildCacheConfig
from .tokenizer import TokenizerBase, tokenizer_factory
from .utils import (generate_api_docs_as_docstring, get_type_repr,
                    print_traceback_on_error)

# TODO[chunweiy]: move the following symbols back to utils scope, and remove the following import


@dataclass
class _ParallelConfig:
    ''' The model distribution configs for LLM.  '''
    tp_size: int = 1
    pp_size: int = 1
    cp_size: int = 1
    gpus_per_node: int = 8
    moe_cluster_size: int = 1
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
                       moe_cluster_size=self.moe_cluster_size,
                       moe_tp_size=self.moe_tp_size,
                       moe_ep_size=self.moe_ep_size,
                       auto_parallel=self.auto_parallel)


class CalibConfig(BaseModel):
    """
    Calibration configuration.
    """
    device: Literal['cuda',
                    'cpu'] = Field(default='cuda',
                                   description="The device to run calibration.")
    calib_dataset: str = Field(
        default='cnn_dailymail',
        description="The name or local path of calibration dataset.")
    calib_batches: int = Field(
        default=512,
        description="The number of batches that the calibration runs.")
    calib_batch_size: int = Field(
        default=1, description="The batch size that the calibration runs.")
    calib_max_seq_length: int = Field(
        default=512,
        description="The maximum sequence length that the calibration runs.")
    random_seed: int = Field(
        default=1234, description="The random seed used for calibration.")
    tokenizer_max_seq_length: int = Field(
        default=2048,
        description=
        "The maximum sequence length to initialize tokenizer for calibration.")

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
        return self.model_dump()


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
            "Lookahead": LookaheadDecodingConfig,
            "NGram": NGramDecodingConfig,
            "DraftTarget": DraftTargetDecodingConfig,
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
    pytorch_weights_path: Optional[str] = None
    eagle3_one_model: Optional[bool] = True

    @classmethod
    def from_dict(cls, data: dict):
        return cls(**data)

    decoding_type: ClassVar[str] = "Eagle"


class NGramDecodingConfig(DecodingBaseConfig):
    """
    Configuration for NGram drafter speculative decoding.

    Arguments:
        prompt_lookup_num_tokens: int
                The length maximum of draft tokens (can be understood as length maximum of output draft tokens).

        max_matching_ngram_size: int
            The length maximum of searching tokens (can be understood as length maximum of input tokens to search).

        is_keep_all: bool = True
            Whether to keep all candidate pattern-matches pairs, only one match is kept for each pattern if False.

        is_use_oldest: bool = True
            Whether to provide the oldest match when pattern is hit, the newest one is provided if False.

        is_public_pool: bool = True
            Whether to use a common pool for all requests, or the pool is private for each request if False.
    """

    prompt_lookup_num_tokens: int = 2
    max_matching_ngram_size: int = 4
    is_keep_all: bool = True
    is_use_oldest: bool = True
    is_public_pool: bool = True

    @classmethod
    def from_dict(cls, data: dict):
        return cls(**data)

    decoding_type: ClassVar[str] = "NGram"


class DraftTargetDecodingConfig(DecodingBaseConfig):
    pytorch_weights_path: Optional[str] = None

    @classmethod
    def from_dict(cls, data: dict):
        return cls(**data)

    decoding_type: ClassVar[str] = "DraftTarget"


class MTPDecodingConfig(DecodingBaseConfig):
    num_nextn_predict_layers: Optional[int] = 1
    use_relaxed_acceptance_for_thinking: Optional[bool] = False
    relaxed_topk: Optional[int] = 1
    relaxed_delta: Optional[float] = 0.

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


@PybindMirror.mirror_pybind_enum(_BatchingType)
class BatchingType(StrEnum, metaclass=PybindMirrorEnumMeta):
    STATIC = "STATIC"
    INFLIGHT = "INFLIGHT"

    def _to_pybind(self):
        return getattr(_BatchingType, self.value)


@PybindMirror.mirror_pybind_enum(_CapacitySchedulerPolicy)
class CapacitySchedulerPolicy(StrEnum, metaclass=PybindMirrorEnumMeta):
    MAX_UTILIZATION = "MAX_UTILIZATION"
    GUARANTEED_NO_EVICT = "GUARANTEED_NO_EVICT"
    STATIC_BATCH = "STATIC_BATCH"

    def _to_pybind(self):
        return getattr(_CapacitySchedulerPolicy, self.value)


@PybindMirror.mirror_pybind_enum(_ContextChunkingPolicy)
class ContextChunkingPolicy(StrEnum, metaclass=PybindMirrorEnumMeta):
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

    @field_validator('max_window_size', 'max_ngram_size',
                     'max_verification_set_size')
    @classmethod
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


@PybindMirror.mirror_pybind_fields(_ExtendedRuntimePerfKnobConfig)
class ExtendedRuntimePerfKnobConfig(BaseModel, PybindMirror):
    """
    Configuration for extended runtime performance knobs.
    """

    multi_block_mode: bool = Field(
        default=True, description="Whether to use multi-block mode.")

    enable_context_fmha_fp32_acc: bool = Field(
        default=False,
        description="Whether to enable context FMHA FP32 accumulation.")

    cuda_graph_mode: bool = Field(default=False,
                                  description="Whether to use CUDA graph mode.")

    cuda_graph_cache_size: int = Field(
        default=0,
        description=
        "Number of cuda graphs to be cached in the runtime. The larger the cache, the better the perf, but more GPU memory is consumed."
    )

    def _to_pybind(self):
        res = _ExtendedRuntimePerfKnobConfig(
            multi_block_mode=self.multi_block_mode,
            enable_context_fmha_fp32_acc=self.enable_context_fmha_fp32_acc)
        res.cuda_graph_mode = self.cuda_graph_mode
        res.cuda_graph_cache_size = self.cuda_graph_cache_size
        return res


@PybindMirror.mirror_pybind_fields(_CacheTransceiverConfig)
class CacheTransceiverConfig(BaseModel, PybindMirror):
    """
    Configuration for the cache transceiver.
    """
    max_num_tokens: Optional[int] = Field(
        default=None,
        description="The max number of tokens the transfer buffer can fit.")

    def _to_pybind(self):
        return _CacheTransceiverConfig(max_num_tokens=self.max_num_tokens)


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


class BaseLlmArgs(BaseModel):
    """
    Base class for both TorchLlmArgs and TrtLlmArgs. It contains all the arguments that are common to both.
    """
    model_config = {
        "arbitrary_types_allowed": True,
        "extra": "forbid",
    }

    # Explicit arguments
    model: Union[str, Path] = Field(
        description=
        "The path to the model checkpoint or the model name from the Hugging Face Hub."
    )

    tokenizer: Optional[Union[
        str, Path, TokenizerBase, PreTrainedTokenizerBase]] = Field(
            description=
            "The path to the tokenizer checkpoint or the tokenizer name from the Hugging Face Hub.",
            default=None)

    tokenizer_mode: Literal['auto', 'slow'] = Field(
        default='auto',
        description="The mode to initialize the tokenizer.",
        json_schema_extra={"type": "Literal['auto', 'slow']"})

    skip_tokenizer_init: bool = Field(
        default=False,
        description="Whether to skip the tokenizer initialization.")

    trust_remote_code: bool = Field(
        default=False, description="Whether to trust the remote code.")

    tensor_parallel_size: int = Field(default=1,
                                      description="The tensor parallel size.")

    dtype: str = Field(default="auto",
                       description="The data type to use for the model.")

    revision: Optional[str] = Field(
        default=None, description="The revision to use for the model.")

    tokenizer_revision: Optional[str] = Field(
        default=None, description="The revision to use for the tokenizer.")

    # Below are all remaining arguments

    pipeline_parallel_size: int = Field(
        default=1, description="The pipeline parallel size.")

    context_parallel_size: int = Field(default=1,
                                       description="The context parallel size.")

    gpus_per_node: Optional[int] = Field(
        default=None,
        description="The number of GPUs per node.",
        validate_default=True)

    moe_cluster_parallel_size: Optional[int] = Field(
        default=None,
        description="The cluster parallel size for MoE models's expert weights."
    )

    moe_tensor_parallel_size: Optional[int] = Field(
        default=None,
        description="The tensor parallel size for MoE models's expert weights.")

    moe_expert_parallel_size: Optional[int] = Field(
        default=None,
        description="The expert parallel size for MoE models's expert weights.")

    enable_attention_dp: bool = Field(
        default=False, description="Enable attention data parallel.")

    cp_config: Optional[dict] = Field(default_factory=dict,
                                      description="Context parallel config.")

    load_format: Literal['auto', 'dummy'] = Field(
        default='auto',
        description="The format to load the model.",
        json_schema_extra={"type": "Literal['auto', 'dummy']"})

    # LoRA arguments
    enable_lora: bool = Field(default=False, description="Enable LoRA.")

    max_lora_rank: Optional[int] = Field(
        default=None,
        description="The maximum LoRA rank.",
        deprecated="Use lora_config.max_lora_rank instead.")

    max_loras: int = Field(default=4,
                           description="The maximum number of LoRA.",
                           deprecated="Use lora_config.max_loras instead.")

    max_cpu_loras: int = Field(
        default=4,
        description="The maximum number of LoRA on CPU.",
        deprecated="Use lora_config.max_cpu_loras instead.")

    lora_config: Optional[LoraConfig] = Field(
        default=None, description="LoRA configuration for the model.")

    # Prompt adapter arguments
    enable_prompt_adapter: bool = Field(default=False,
                                        description="Enable prompt adapter.")

    max_prompt_adapter_token: int = Field(
        default=0, description="The maximum number of prompt adapter tokens.")

    # Quantization and calibration configurations
    quant_config: Optional[QuantConfig] = Field(
        default=None, description="Quantization config.", validate_default=True)

    # Several options from ExecutorConfig, expanded here for less hierarchy
    kv_cache_config: KvCacheConfig = Field(default_factory=KvCacheConfig,
                                           description="KV cache config.")

    enable_chunked_prefill: bool = Field(default=False,
                                         description="Enable chunked prefill.")

    guided_decoding_backend: Optional[str] = Field(
        default=None, description="Guided decoding backend.")

    batched_logits_processor: Optional[object] = Field(
        default=None,
        description="Batched logits processor.",
        json_schema_extra={
            "type": f"Optional[{get_type_repr(BatchedLogitsProcessor)}]"
        })

    iter_stats_max_iterations: Optional[int] = Field(
        default=None,
        description="The maximum number of iterations for iter stats.")

    request_stats_max_iterations: Optional[int] = Field(
        default=None,
        description="The maximum number of iterations for request stats.")

    # A handful of options from PretrainedConfig
    peft_cache_config: Optional[PeftCacheConfig] = Field(
        default=None, description="PEFT cache config.")

    scheduler_config: SchedulerConfig = Field(default_factory=SchedulerConfig,
                                              description="Scheduler config.")

    cache_transceiver_config: Optional[CacheTransceiverConfig] = Field(
        default=None, description="Cache transceiver config.")

    # Speculative decoding parameters
    speculative_config: Optional[
        Union[LookaheadDecodingConfig, MedusaDecodingConfig,
              EagleDecodingConfig, MTPDecodingConfig, NGramDecodingConfig,
              DraftTargetDecodingConfig]] = Field(
                  default=None, description="Speculative decoding config.")

    batching_type: Optional[BatchingType] = Field(default=None,
                                                  description="Batching type.")

    normalize_log_probs: bool = Field(
        default=False, description="Normalize log probabilities.")

    max_batch_size: Optional[int] = Field(default=None,
                                          description="The maximum batch size.")

    # generation constraints
    max_input_len: Optional[int] = Field(
        default=None, description="The maximum input length.")

    max_seq_len: Optional[int] = Field(
        default=None, description="The maximum sequence length.")

    max_beam_width: Optional[int] = Field(default=None,
                                          description="The maximum beam width.")

    max_num_tokens: Optional[int] = Field(
        default=None, description="The maximum number of tokens.")

    gather_generation_logits: bool = Field(
        default=False, description="Gather generation logits.")

    # private fields those are unstable and just for internal use
    num_postprocess_workers: int = Field(
        default=0,
        description=
        "The number of processes used for postprocessing the generated tokens, including detokenization."
    )

    postprocess_tokenizer_dir: Optional[str] = Field(
        default=None,
        description="The path to the tokenizer directory for postprocessing.")

    reasoning_parser: Optional[str] = Field(
        default=None,
        description="The parser to separate reasoning content from output.")

    garbage_collection_gen0_threshold: int = Field(
        default=20000,
        description=
        "Threshold for Python garbage collection of generation 0 objects."
        "Lower values trigger more frequent garbage collection.")

    # TODO[Superjomn]: To deprecate this config.
    decoding_config: Optional[object] = Field(
        default=None,
        description="The decoding config.",
        json_schema_extra={"type": "Optional[DecodingConfig]"},
        deprecated="Use speculative_config instead.",
    )

    mpi_session: Optional[object] = Field(
        default=None,
        description="The optional MPI session to use for this LLM instance.",
        json_schema_extra={"type": "Optional[MpiSession]"},
        exclude=True,
        alias="_mpi_session")

    backend: Optional[str] = Field(
        default=None,
        description="The backend to use for this LLM instance.",
        exclude_json_schema=True,  # hide from API references
        validate_default=True,
    )

    _parallel_config: Optional[object] = PrivateAttr(default=None)
    _model_format: Optional[_ModelFormatKind] = PrivateAttr(default=None)
    _speculative_model: Optional[str] = PrivateAttr(default=None)
    _speculative_model_format: Optional[_ModelFormatKind] = PrivateAttr(
        default=None)

    @property
    def parallel_config(self) -> _ParallelConfig:
        return self._parallel_config

    @property
    def model_format(self) -> _ModelFormatKind:
        return self._model_format

    @property
    def speculative_model(self) -> Optional[_ModelFormatKind]:
        return self._speculative_model

    @property
    def speculative_model_format(self) -> _ModelFormatKind:
        return self._speculative_model_format

    @classmethod
    def from_kwargs(cls, **kwargs: Any) -> "BaseLlmArgs":
        """Create `LlmArgs` instance from kwargs.

        Args:
            kwargs (Any): Arguments passed to `LlmArgs` constructor.

        Returns:
            tensorrt_llm.llmapi.llm_utils.BaseLlmArgs: The `BaseLlmArgs` instance.
        """
        kwargs = BaseLlmArgs._check_consistency(dict(kwargs))

        ret = cls(**kwargs)
        return ret

    def to_dict(self) -> dict:
        """Dump `LlmArgs` instance to a dict.

        Returns:
            dict: The dict that contains all fields of the `LlmArgs` instance.
        """
        model_dict = self.model_dump(mode='json')
        # TODO: the BuildConfig.to_dict and from_dict don't work well with pydantic
        model_dict['build_config'] = copy.deepcopy(self.build_config)
        return model_dict

    @staticmethod
    def _check_consistency(kwargs_dict: Dict[str, Any]) -> Dict[str, Any]:
        # max_beam_width is not included since vague behavior due to lacking the support for dynamic beam width during
        # generation
        black_list = set(["max_beam_width"])
        executor_config_attrs = set(
            attr for attr in dir(_ExecutorConfig) if not attr.startswith('_')
            and callable(getattr(_ExecutorConfig, attr)))
        executor_config_attrs -= black_list
        llm_args_attr = set(BaseLlmArgs.model_fields.keys())
        # NOTE: When cpp ExecutorConfig add new options, please add the new options into `LlmArgs` with docs as well
        # ASK chunweiy for help if you are not sure about the new options.
        assert executor_config_attrs.issubset(
            llm_args_attr
        ), f"New options found in underlying ExecutorConfig: {llm_args_attr - executor_config_attrs}"

        return kwargs_dict

    @field_validator("dtype")
    @classmethod
    def validate_dtype(cls, v, info):
        if torch.cuda.get_device_properties(0).major < 8:
            if v == 'auto':
                v = 'float16'
            if v == 'bfloat16':
                raise RuntimeError("Pre SM 80 GPUs do not support bfloat16")
        return v

    @field_validator("quant_config", mode='before')
    @classmethod
    def validate_quant_config(cls, v, info):
        if v is None:
            v = QuantConfig()
        return v

    @field_validator("gpus_per_node", mode='before')
    @classmethod
    def validate_gpus_per_node(cls, v, info):
        if v is None:
            logger.warning(
                f"Using default gpus_per_node: {torch.cuda.device_count()}")
            v = torch.cuda.device_count()
        return v

    @field_validator("model")
    @classmethod
    def validate_model(cls, v, info):
        if not isinstance(v, (str, Path)):
            raise ValueError(f"Invalid model: {v}")
        return v

    @model_validator(mode="after")
    def validate_parallel_config(self):
        if self.moe_cluster_parallel_size is None:
            self.moe_cluster_parallel_size = -1

        if self.moe_tensor_parallel_size is None:
            self.moe_tensor_parallel_size = -1

        if self.moe_expert_parallel_size is None:
            self.moe_expert_parallel_size = -1

        self._parallel_config = _ParallelConfig(
            tp_size=self.tensor_parallel_size,
            pp_size=self.pipeline_parallel_size,
            cp_size=self.context_parallel_size,
            gpus_per_node=self.gpus_per_node,
            moe_cluster_size=self.moe_cluster_parallel_size,
            moe_tp_size=self.moe_tensor_parallel_size,
            moe_ep_size=self.moe_expert_parallel_size,
            enable_attention_dp=self.enable_attention_dp,
            cp_config=self.cp_config)
        return self

    @model_validator(mode="after")
    def set_default_max_input_len(self):
        if self.max_input_len is None:
            self.max_input_len = 1024
        return self

    @model_validator(mode="after")
    def validate_and_init_tokenizer(self):
        """Initialize tokenizer based on configuration."""
        if self.skip_tokenizer_init:
            self.tokenizer = None
        else:
            self.tokenizer = tokenizer_factory(
                self.tokenizer,
                trust_remote_code=self.trust_remote_code,
                use_fast=self.tokenizer_mode != 'slow')
        return self

    @model_validator(mode="after")
    def validate_model_format_misc(self):
        '''
        Load the model format, and do the following:

        1. Load the build_config if got an engine.
        2. Load the parallel_config if got a checkpoint.
        '''
        model_obj = _ModelWrapper(self.model)

        if model_obj.is_local_model and self.backend not in [
                'pytorch', '_autodeploy'
        ]:
            # Load parallel_config from the engine.
            model_format = get_model_format(self.model)

            if model_format is _ModelFormatKind.TLLM_ENGINE:
                if self.build_config is not None:
                    logger.warning(
                        "The build_config is ignored for model format of TLLM_ENGINE."
                    )
                self._load_config_from_engine(model_obj.model_dir)
                runtime_defaults = self._pretrained_config.runtime_defaults
                if runtime_defaults:
                    self.kv_cache_config.fill_empty_fields_from_runtime_defaults(
                        runtime_defaults)

            # Load parallel_config from the checkpoint.
            elif model_format is _ModelFormatKind.TLLM_CKPT:
                # We need to create a temporary instance to call _load_config_from_ckpt
                self._load_config_from_ckpt(model_obj.model_dir)
        else:
            model_format = _ModelFormatKind.HF

        # Store the model format in the values
        self._model_format = model_format
        return self

    @model_validator(mode="after")
    def init_build_config(self):
        """
        Creating a default BuildConfig if none is provided
        """
        if self.build_config is None:
            kwargs = {}
            if self.max_batch_size:
                kwargs["max_batch_size"] = self.max_batch_size
            if self.max_num_tokens:
                kwargs["max_num_tokens"] = self.max_num_tokens
            if self.max_seq_len:
                kwargs["max_seq_len"] = self.max_seq_len
            if self.max_beam_width:
                kwargs["max_beam_width"] = self.max_beam_width
            if self.max_input_len:
                kwargs["max_input_len"] = self.max_input_len
            self.build_config = BuildConfig(**kwargs)

        assert isinstance(
            self.build_config, BuildConfig
        ), f"build_config is not initialized: {self.build_config}"
        return self

    @model_validator(mode="after")
    def set_runtime_knobs_from_build_config(self):
        # TODO: remove this after PyT become default to adapt PyT with build_config as input
        assert self.build_config is not None, "build_config is not initialized"
        if self.backend == "pytorch":
            if self.build_config:
                for key in [
                        "max_batch_size", "max_num_tokens", "max_seq_len",
                        "max_input_len", "max_beam_width"
                ]:
                    if getattr(self.build_config, key) is not None:
                        if (v := getattr(self, key,
                                         None)) is not None and v != getattr(
                                             self.build_config, key):
                            logger.warning(
                                f"overriding {key} from build_config")
                        setattr(self, key, getattr(self.build_config, key))

        return self

    @model_validator(mode="after")
    def validate_build_config_with_runtime_params(self):
        # Note: max_batch_size and max_num_tokens in LlmArgs are for runtime,
        # which will be passed to the C++ Executor API, overwriting the values
        # from an built engine. In order to set build configuration, it is
        # recommended to use build_config instead.
        assert isinstance(
            self.build_config, BuildConfig
        ), f"build_config is not initialized: {self.build_config}"

        if self.max_batch_size is not None:
            if self.max_batch_size > self.build_config.max_batch_size:
                raise ValueError(
                    f"max_batch_size [{self.max_batch_size}] is greater than build_config.max_batch_size [{self.build_config.max_batch_size}] in build_config"
                )
        if self.max_num_tokens is not None:
            if self.max_num_tokens > self.build_config.max_num_tokens:
                raise ValueError(
                    f"max_num_tokens [{self.max_num_tokens}] is greater than build_config.max_num_tokens [{self.build_config.max_num_tokens}] in build_config"
                )
        if self.max_seq_len is not None:
            if self.max_seq_len != self.build_config.max_seq_len:
                logger.warning(
                    f"max_seq_len [{self.max_seq_len}] is overridden by build_config.max_seq_len [{self.build_config.max_seq_len}] in build_config"
                )
        if self.max_beam_width is not None:
            if self.max_beam_width != self.build_config.max_beam_width:
                logger.warning(
                    f"max_beam_width [{self.max_beam_width}] is overridden by build_config.max_beam_width [{self.build_config.max_beam_width}] in build_config"
                )
        if self.max_input_len is not None:
            if self.max_input_len != self.build_config.max_input_len:
                logger.warning(
                    f"max_input_len [{self.max_input_len}] is overridden by build_config.max_input_len [{self.build_config.max_input_len}] in build_config"
                )

        return self

    @model_validator(mode="after")
    def validate_build_config_remaining(self):
        is_trt_llm_args = isinstance(self, TrtLlmArgs)

        # TODO: remove the checker when manage weights support all data types
        if is_trt_llm_args and self.fast_build and (self.quant_config.quant_algo
                                                    is QuantAlgo.FP8):
            self._update_plugin_config("manage_weights", True)

        if self.parallel_config._world_size == 1 and self.build_config:
            self.build_config.plugin_config.nccl_plugin = None

        if self.enable_lora and self.lora_config is None and self.backend != 'pytorch':
            self.build_config.plugin_config.lora_plugin = 'auto'
            if self.max_lora_rank is not None:
                self.build_config.lora_config.max_lora_rank = self.max_lora_rank

        if self.enable_prompt_adapter:
            self.build_config.max_prompt_embedding_table_size = self.max_prompt_adapter_token * self.build_config.max_batch_size

        if self.max_beam_width is None:
            if self.build_config:
                self.max_beam_width = self.build_config.max_beam_width
            else:
                self.max_beam_width = 1

        return self

    @model_validator(mode="after")
    def validate_speculative_config(self):
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

                if self.backend not in ['pytorch', '_autodeploy']:
                    eagle_config = _EagleConfig(
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
                        draft_model_path=self.speculative_config.
                        pytorch_weights_path,
                        eagle3_one_model=self.speculative_config.
                        eagle3_one_model)
            elif isinstance(self.speculative_config, NGramDecodingConfig):
                self.build_config.speculative_decoding_mode = SpeculativeDecodingMode.NGRAM
                assert self.backend in ['pytorch', '_autodeploy']
                assert self.speculative_config.prompt_lookup_num_tokens > 0 and self.speculative_config.max_matching_ngram_size > 0
                self.build_config.max_draft_len = self.speculative_config.max_draft_len
                from tensorrt_llm._torch.speculative import NGramConfig
                self.speculative_config = NGramConfig(
                    prompt_lookup_num_tokens=self.speculative_config.
                    prompt_lookup_num_tokens,
                    max_matching_ngram_size=self.speculative_config.
                    max_matching_ngram_size,
                    is_keep_all=self.speculative_config.is_keep_all,
                    is_use_oldest=self.speculative_config.is_use_oldest,
                    is_public_pool=self.speculative_config.is_public_pool,
                )
            elif isinstance(self.speculative_config, DraftTargetDecodingConfig):
                self.build_config.speculative_decoding_mode = SpeculativeDecodingMode.DRAFT_TOKENS_EXTERNAL
                assert self.backend == 'pytorch'
                assert self.speculative_config.max_draft_len > 0
                self.build_config.max_draft_len = self.speculative_config.max_draft_len
                from tensorrt_llm._torch.speculative import DraftTargetConfig
                self.speculative_config = DraftTargetConfig(
                    max_draft_tokens=self.speculative_config.max_draft_len,
                    draft_model_path=self.speculative_config.
                    pytorch_weights_path)
            elif isinstance(self.speculative_config, MTPDecodingConfig):
                from tensorrt_llm._torch.speculative import MTPConfig
                self.speculative_config = MTPConfig(
                    num_nextn_predict_layers=self.speculative_config.
                    num_nextn_predict_layers,
                    max_batch_size=self.build_config.max_batch_size,
                    use_relaxed_acceptance_for_thinking=self.speculative_config.
                    use_relaxed_acceptance_for_thinking,
                    relaxed_topk=self.speculative_config.relaxed_topk,
                    relaxed_delta=self.speculative_config.relaxed_delta)
            else:
                raise ValueError(
                    f"Speculative config type not recognized: {self.speculative_config}"
                )
        else:
            self.decoding_config = None

        self._speculative_model = getattr(self.speculative_config,
                                          "speculative_model", None)
        speculative_model_obj = _ModelWrapper(
            self._speculative_model
        ) if self._speculative_model is not None else None
        if self._speculative_model and speculative_model_obj.is_local_model:
            self._speculative_model_format = _ModelFormatKind.HF

        return self

    @model_validator(mode="after")
    def validate_lora_config_consistency(self):
        if self.lora_config:
            if self.max_lora_rank is not None:
                logger.warning(
                    "max_lora_rank is ignored when lora_config is provided.")
            if self.max_loras != self.lora_config.max_loras:
                logger.warning(
                    "max_loras is ignored when lora_config is provided.")
            if self.max_cpu_loras != self.lora_config.max_cpu_loras:
                logger.warning(
                    "max_cpu_loras is ignored when lora_config is provided.")

            if len(self.lora_config.lora_dir) == 0:
                # TODO [TRTLLM-5173]
                logger.warning(
                    "lora_dir is empty, so custom embedding or lm head will not be applied."
                )

        if self.enable_lora and self.lora_config is not None and self.backend in [
                'pytorch', '_autodeploy'
        ]:
            logger.warning(
                f"enable_lora is ignored when lora_config is provided for {self.backend} backend."
            )

        if self.lora_config is not None:
            if len(self.lora_config.lora_dir) == 0 and len(
                    self.lora_config.lora_target_modules) == 0:
                logger.warning(
                    "Both lora_dir and lora_target_modules are empty, so all LoRA modules will be expected. "
                    "This will lead to serious memory consumption. Please provide either lora_dir or lora_target_modules if this behavior is not what you expect."
                )
                default_trtllm_modules_to_hf_modules = get_default_trtllm_modules_to_hf_modules(
                )
                self.lora_config.lora_target_modules = list(
                    default_trtllm_modules_to_hf_modules.keys())
        return self

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
        self._parallel_config = _ParallelConfig(
            tp_size=mapping.tp_size,
            pp_size=mapping.pp_size,
            cp_size=mapping.cp_size,
            gpus_per_node=mapping.gpus_per_node,
            moe_cluster_size=mapping.moe_cluster_size,
            moe_tp_size=mapping.moe_tp_size,
            moe_ep_size=mapping.moe_ep_size)

    def _load_config_from_ckpt(self, ckpt_dir: Path):
        pretrained_config = PretrainedConfig.from_json_file(ckpt_dir /
                                                            "config.json")
        tp_size = pretrained_config.mapping.tp_size
        pp_size = pretrained_config.mapping.pp_size
        cp_size = pretrained_config.mapping.cp_size
        moe_cluster_size = pretrained_config.mapping.moe_cluster_size
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
            self._parallel_config = _ParallelConfig(
                tp_size=tp_size,
                pp_size=pp_size,
                cp_size=cp_size,
                gpus_per_node=gpus_per_node,
                moe_cluster_size=moe_cluster_size,
                moe_tp_size=moe_tp_size,
                moe_ep_size=moe_ep_size)


class TrtLlmArgs(BaseLlmArgs):

    auto_parallel: bool = Field(
        default=False,
        description="Enable auto parallel mode.",
        deprecated=
        "Use tensor_parallel_size/pipeline_parallel_size/xxx_parallel_size instead.",
    )

    auto_parallel_world_size: Optional[int] = Field(
        default=None,
        description="The world size for auto parallel mode.",
        deprecated=
        "Use tensor_parallel_size/pipeline_parallel_size/xxx_parallel_size instead.",
    )

    enable_tqdm: bool = Field(default=False,
                              description="Enable tqdm for progress bar.")

    workspace: Optional[str] = Field(default=None,
                                     description="The workspace for the model.")

    # Once set, the model will reuse the build_cache
    enable_build_cache: object = Field(
        default=False,
        description="Enable build cache.",
        json_schema_extra={
            "type": f"Union[{get_type_repr(BuildCacheConfig)}, bool]"
        })

    extended_runtime_perf_knob_config: Optional[
        ExtendedRuntimePerfKnobConfig] = Field(
            default=None, description="Extended runtime perf knob config.")

    calib_config: Optional[CalibConfig] = Field(
        default=None, description="Calibration config.", validate_default=True)

    embedding_parallel_mode: str = Field(
        default='SHARDING_ALONG_VOCAB',
        description="The embedding parallel mode.")

    fast_build: bool = Field(default=False, description="Enable fast build.")

    # BuildConfig is introduced to give users a familiar interface to configure the model building.
    build_config: Optional[object] = Field(
        default=None,
        description="Build config.",
        json_schema_extra={"type": f"Optional[{get_type_repr(BuildConfig)}]"})

    # Private attributes
    _auto_parallel_config: Optional[AutoParallelConfig] = PrivateAttr(
        default=None)
    # This is used to hold the options for convert_checkpoint
    _convert_checkpoint_options: Dict[str,
                                      Any] = PrivateAttr(default_factory=dict)

    @property
    def auto_parallel_config(self) -> AutoParallelConfig:
        return self._auto_parallel_config

    @field_validator('calib_config', mode='before')
    @classmethod
    def init_calib_config(cls, v):
        if v is None:
            return CalibConfig()
        return v

    @model_validator(mode="after")
    def setup_embedding_parallel_mode(self):
        if self.embedding_parallel_mode == 'NONE':
            self._convert_checkpoint_options['use_parallel_embedding'] = False
        elif self.embedding_parallel_mode == 'SHARDING_ALONG_VOCAB':
            self._convert_checkpoint_options['use_parallel_embedding'] = True
            self._convert_checkpoint_options['embedding_sharding_dim'] = 0
        elif self.embedding_parallel_mode == 'SHARDING_ALONG_HIDDEN':
            self._convert_checkpoint_options['use_parallel_embedding'] = True
            self._convert_checkpoint_options['embedding_sharding_dim'] = 1
        # No else clause needed since validation already happened
        return self

    @model_validator(mode="after")
    def validate_auto_parallel(self):
        self._auto_parallel_config = AutoParallelConfig(
            sharded_io_allowlist=[
                "past_key_value_\\d+",
                "present_key_value_\\d*",
            ],
            same_buffer_io={
                "past_key_value_(\\d+)": "present_key_value_\\1",
            },
            **infer_cluster_config(),
        )

        self.parallel_config.auto_parallel = self.auto_parallel

        if self.parallel_config.auto_parallel:
            self.parallel_config.world_size = self.auto_parallel_world_size

        return self

    @model_validator(mode="after")
    def validate_enable_build_cache(self):
        if not self.enable_build_cache:
            return self
        self.enable_build_cache = BuildCacheConfig() if isinstance(
            self.enable_build_cache, bool) else self.enable_build_cache
        if not isinstance(self.enable_build_cache, BuildCacheConfig):
            raise ValueError(
                f"Invalid build_cache_config: {self.enable_build_cache}")
        return self


LlmArgs = TrtLlmArgs

LLMARGS_EXPLICIT_DOCSTRING = generate_api_docs_as_docstring(LlmArgs,
                                                            indent=' ' * 4)


class LoadFormat(Enum):
    AUTO = 0
    # Initialize all weights randomly.
    DUMMY = 1


class TorchCompileConfig(BaseModel):
    """
    Configuration for torch.compile.
    """
    torch_compile_fullgraph: bool = Field(
        default=True,
        description="Enable full graph compilation in torch.compile.")

    torch_compile_inductor_enabled: bool = Field(
        default=False, description="Enable inductor backend in torch.compile.")

    torch_compile_piecewise_cuda_graph: bool = Field(
        default=False,
        description="Enable piecewise CUDA graph in torch.compile.")

    torch_compile_enable_userbuffers: bool = Field(
        default=True,
        description=
        "When torch compile is enabled, userbuffers is enabled by default.")


class TorchLlmArgs(BaseLlmArgs):

    # Just a dummy BuildConfig to allow code reuse with the TrtLlmArgs
    build_config: Optional[object] = Field(
        default=None,
        description="Build config.",
        exclude_from_json=True,
        json_schema_extra={"type": f"Optional[{get_type_repr(BuildConfig)}]"})

    # PyTorch backend specific configurations

    use_cuda_graph: bool = Field(
        default=False,
        description=
        "If true, use CUDA graphs for decoding. CUDA graphs are only created for the batch sizes in cuda_graph_batch_sizes, and are enabled for batches that consist of decoding requests *only* (the reason is that it's hard to capture a single graph with prefill requests since the input shapes are a function of the sequence lengths). Note that each CUDA graph can use up to 200 MB of extra memory."
    )

    cuda_graph_batch_sizes: Optional[List[int]] = Field(
        default=None,
        description="List of batch sizes to create CUDA graphs for.")

    cuda_graph_max_batch_size: int = Field(
        default=0, description="Maximum batch size for CUDA graphs.")

    cuda_graph_padding_enabled: bool = Field(
        default=False,
        description=
        "If true, batches are rounded up to the nearest cuda_graph_batch_size. This is usually a net win for performance."
    )

    disable_overlap_scheduler: bool = Field(
        default=False, description="Disable the overlap scheduler.")

    moe_max_num_tokens: Optional[int] = Field(
        default=None,
        description=
        "If set, at most moe_max_num_tokens tokens will be sent to torch.ops.trtllm.fused_moe at the same time. If the number of tokens exceeds moe_max_num_tokens, the input tensors will be split into chunks and a for loop will be used."
    )

    moe_load_balancer: Optional[Union[object, str]] = Field(
        default=None,
        description="Configuration for MoE load balancing.",
        json_schema_extra={"type": "Union[MoeLoadBalancerConfig, str]"})

    attn_backend: str = Field(default='TRTLLM',
                              description="Attention backend to use.")

    moe_backend: str = Field(default='CUTLASS',
                             description="MoE backend to use.")

    mixed_sampler: bool = Field(
        default=False,
        description=
        "If true, will iterate over sampling_params of each request and use the corresponding sampling strategy, e.g. top-k, top-p, etc."
    )

    enable_trtllm_sampler: bool = Field(
        default=False,
        description=
        "If true, will use the TRTLLM sampler instead of the PyTorch sampler. The TRTLLM sampler has a wide coverage of sampling strategies."
    )

    kv_cache_dtype: str = Field(default="auto",
                                description="Data type for KV cache.")

    enable_iter_perf_stats: bool = Field(
        default=False, description="Enable iteration performance statistics.")

    enable_iter_req_stats: bool = Field(
        default=False,
        description=
        "If true, enables per request stats per iteration. Must also set enable_iter_perf_stats to true to get request stats."
    )

    print_iter_log: bool = Field(default=False,
                                 description="Print iteration logs.")

    torch_compile_config: Optional[TorchCompileConfig] = Field(
        default=None, description="Torch compile config.")

    autotuner_enabled: bool = Field(
        default=True,
        description="Enable autotuner only when torch compile is enabled.")

    enable_layerwise_nvtx_marker: bool = Field(
        default=False, description="If true, enable layerwise nvtx marker.")

    load_format: Union[str, LoadFormat] = Field(
        default=LoadFormat.AUTO,
        description=
        "How to load the model weights. By default, detect the weight type from the model checkpoint."
    )

    enable_min_latency: bool = Field(
        default=False,
        description=
        "If true, enable min-latency mode. Currently only used for Llama4.",
    )

    # TODO: make this a per-request parameter
    stream_interval: int = Field(
        default=1,
        description=
        "The iteration interval to create responses under the streaming mode.",
    )

    # TODO: remove backend later
    @field_validator('backend', mode='before')
    def init_backend(cls, v):
        if v is None:
            return 'pytorch'
        return v

    @field_validator('load_format', mode='before')
    @classmethod
    def convert_load_format(cls, v):
        if isinstance(v, LoadFormat):
            return v
        load_format = v.upper()
        if load_format not in LoadFormat.__members__:
            raise ValueError(f"Invalid LoadFormat: {v}")
        return LoadFormat[load_format]

    # Extra resource managers to use in addition to the KV cache manager.
    # Each manager's prepare_resources method is called before the forward pass,
    # and update_resources() is called after the pass finishes. free_resources()
    # is called when a request finishes. The KV cache manager is guaranteed to
    # be invoked after all of these extra managers in all stages.
    _extra_resource_managers: Dict[str,
                                   object] = PrivateAttr(default_factory=dict, )

    @property
    def extra_resource_managers(self) -> Dict[str, object]:
        return self._extra_resource_managers

    @extra_resource_managers.setter
    def extra_resource_managers(self, value: Dict[str, object]) -> None:
        self._extra_resource_managers = value

    @model_validator(mode="after")
    def validate_moe_load_balancer(self):
        from .._torch.model_config import MoeLoadBalancerConfig
        if isinstance(self.moe_load_balancer, str):
            if not os.path.exists(self.moe_load_balancer):
                raise FileNotFoundError(
                    f"MoE load balancer config file not found: {self.moe_load_balancer}"
                )
            try:
                with open(self.moe_load_balancer) as f:
                    moe_load_balancer_config = yaml.safe_load(f)
                self.moe_load_balancer = MoeLoadBalancerConfig(
                    **moe_load_balancer_config)
            except Exception as e:
                raise ValueError(
                    f"Failed to load MoE load balancer config file: {self.moe_load_balancer}"
                ) from e
        return self

    # TODO: Remove this after the PyTorch backend is fully migrated to TorchLlmArgs from ExecutorConfig
    def get_pytorch_backend_config(self) -> "PyTorchConfig":
        from tensorrt_llm._torch.pyexecutor.config import PyTorchConfig

        return PyTorchConfig(
            extra_resource_managers=self.extra_resource_managers,
            use_cuda_graph=self.use_cuda_graph,
            cuda_graph_batch_sizes=self.cuda_graph_batch_sizes,
            cuda_graph_max_batch_size=self.cuda_graph_max_batch_size,
            cuda_graph_padding_enabled=self.cuda_graph_padding_enabled,
            disable_overlap_scheduler=self.disable_overlap_scheduler,
            moe_max_num_tokens=self.moe_max_num_tokens,
            moe_load_balancer=self.moe_load_balancer,
            attn_backend=self.attn_backend,
            moe_backend=self.moe_backend,
            mixed_sampler=self.mixed_sampler,
            enable_trtllm_sampler=self.enable_trtllm_sampler,
            kv_cache_dtype=self.kv_cache_dtype,
            enable_iter_perf_stats=self.enable_iter_perf_stats,
            enable_iter_req_stats=self.enable_iter_req_stats,
            print_iter_log=self.print_iter_log,
            torch_compile_enabled=bool(self.torch_compile_config is not None),
            torch_compile_fullgraph=self.torch_compile_config.
            torch_compile_fullgraph
            if self.torch_compile_config is not None else True,
            torch_compile_inductor_enabled=self.torch_compile_config.
            torch_compile_inductor_enabled
            if self.torch_compile_config is not None else False,
            torch_compile_piecewise_cuda_graph=self.torch_compile_config.
            torch_compile_piecewise_cuda_graph
            if self.torch_compile_config is not None else False,
            torch_compile_enable_userbuffers=self.torch_compile_config.
            torch_compile_enable_userbuffers
            if self.torch_compile_config is not None else True,
            autotuner_enabled=self.autotuner_enabled,
            enable_layerwise_nvtx_marker=self.enable_layerwise_nvtx_marker,
            load_format=self.load_format,
            enable_min_latency=self.enable_min_latency,
            stream_interval=self.stream_interval)

    @field_validator('cuda_graph_max_batch_size')
    @classmethod
    def validate_cuda_graph_max_batch_size(cls, v):
        """Validate cuda_graph_max_batch_size is non-negative."""
        if v < 0:
            raise ValueError("cuda_graph_max_batch_size must be non-negative")
        return v

    @model_validator(mode='after')
    def validate_cuda_graph_config(self) -> 'TorchLlmArgs':
        """Validate CUDA graph configuration.

        Ensures that:
        1. If cuda_graph_batch_sizes is provided, cuda_graph_max_batch_size must be 0
        2. If cuda_graph_batch_sizes is not provided, it is generated based on cuda_graph_max_batch_size
        3. If both are provided, cuda_graph_batch_sizes must match the generated values
        """
        if self.cuda_graph_batch_sizes:
            self.cuda_graph_batch_sizes = sorted(self.cuda_graph_batch_sizes)
            if self.cuda_graph_max_batch_size != 0:
                if self.cuda_graph_batch_sizes != self._generate_cuda_graph_batch_sizes(
                        self.cuda_graph_max_batch_size,
                        self.cuda_graph_padding_enabled):
                    raise ValueError(
                        "Please don't set both cuda_graph_batch_sizes "
                        "and cuda_graph_max_batch_size.\n"
                        f"cuda_graph_batch_sizes: {self.cuda_graph_batch_sizes}, "
                        f"cuda_graph_max_batch_size: {self.cuda_graph_max_batch_size}"
                    )
            else:
                self.cuda_graph_max_batch_size = max(
                    self.cuda_graph_batch_sizes)
        else:
            max_batch_size = self.cuda_graph_max_batch_size or 128
            generated_sizes = self._generate_cuda_graph_batch_sizes(
                max_batch_size, self.cuda_graph_padding_enabled)
            self.cuda_graph_batch_sizes = generated_sizes
            self.cuda_graph_max_batch_size = max_batch_size

        return self

    @staticmethod
    def _generate_cuda_graph_batch_sizes(max_batch_size: int,
                                         padding_enabled: bool) -> List[int]:
        """Generate a list of batch sizes for CUDA graphs.

        Args:
            max_batch_size: Maximum batch size to generate up to
            padding_enabled: Whether padding is enabled, which affects the batch size distribution

        Returns:
            List of batch sizes to create CUDA graphs for
        """
        if padding_enabled:
            batch_sizes = [1, 2, 4] + [i * 8 for i in range(1, 17)]
        else:
            batch_sizes = list(range(1, 32)) + [32, 64, 128]

        # Add powers of 2 up to max_batch_size
        batch_sizes += [
            2**i for i in range(8, math.floor(math.log(max_batch_size, 2)))
        ]

        # Filter and sort batch sizes
        batch_sizes = sorted(
            [size for size in batch_sizes if size <= max_batch_size])

        # Add max_batch_size if not already included
        if max_batch_size != batch_sizes[-1]:
            batch_sizes.append(max_batch_size)

        return batch_sizes


class _AutoDeployLlmArgs(TorchLlmArgs):
    """LLM arguments specifically for AutoDeploy backend.

    This class extends TorchLlmArgs with AutoDeploy-specific configuration options.
    AutoDeploy provides automatic deployment and optimization of language models
    with various attention backends and optimization strategies.
    """

    model_factory: Literal[
        "AutoModelForCausalLM", "AutoModelForImageTextToText"] = Field(
            default="AutoModelForCausalLM",
            description="The model factory to use for loading the model.",
        )

    model_kwargs: Dict[str, Any] = Field(
        default_factory=dict,
        description=
        "Extra kwargs for the model config class to customize the model config. "
        "These arguments take precedence over default values or config values in the model config "
        "file. Arguments are resolved in order: 1) Default values in model config class, 2) Values "
        "in model config file, 3) Values in model_kwargs. Note: if a kwarg doesn't exist in the "
        "model config class, it will be ignored.",
    )

    mla_backend: Literal["MultiHeadLatentAttention"] = Field(
        default="MultiHeadLatentAttention",
        description="The Multi-Head Latent Attention backend to use.",
    )

    skip_loading_weights: bool = Field(
        default=False,
        description=
        "Whether to skip loading model weights during initialization. "
        "If True, only the model architecture is loaded.",
    )

    free_mem_ratio: float = Field(
        default=0.8,
        description="The fraction of available memory to allocate for cache. "
        "Must be between 0.0 and 1.0.",
    )

    simple_shard_only: bool = Field(
        default=False,
        description=
        "If True, force simple sharding (all_gather) in tensor parallelism. "
        "If False, auto-detect and use column+row (all_reduce) sharding when possible.",
    )

    # TODO: Remove this field once tokens_per_block is properly passed through
    attn_page_size: int = Field(
        default=64,
        description=
        "Page size for attention (tokens_per_block). For TritonWithFlattenedInputs "
        "backend, this should equal max_seq_len. Temporary field until tokens_per_block gets "
        "properly passed through.",
    )

    checkpoint_device: Optional[str] = Field(
        default=None,
        description="Device on which to load the model checkpoint. "
        "Defaults to the same device as the rest of the pipeline.",
    )

    @field_validator("free_mem_ratio")
    @classmethod
    def validate_free_mem_ratio(cls, v):
        """Validate that free_mem_ratio is between 0.0 and 1.0."""
        if not 0.0 <= v <= 1.0:
            raise ValueError(
                f"free_mem_ratio must be between 0.0 and 1.0, got {v}")
        return v

    @print_traceback_on_error
    def model_post_init(self, __context):
        # Modify default values that differ from TorchLlmArgs
        new_defaults = {
            "max_batch_size": 8,
            "max_seq_len": 512,
            "attn_backend": "FlashInfer",
            # TODO: Remove this when overlap scheduler is supported (https://github.com/NVIDIA/TensorRT-LLM/issues/4364)
            "disable_overlap_scheduler": True,
        }
        for k, v_default in new_defaults.items():
            if k not in self.__pydantic_fields_set__:
                setattr(self, k, v_default)

        # NOTE: Only call super() after setting the default values since default values should be
        # set first.
        super().model_post_init(__context)

        # Handle attn_page_size for TritonWithFlattenedInputs backend
        if self.attn_backend == "TritonWithFlattenedInputs":
            self.attn_page_size = self.max_seq_len

        # Add max_position_embeddings to model_kwargs
        # TODO (lucaslie): this is more HF specific than a generic model_kwargs. Ideally, we can
        # move this to the HF model factory but we don't have access to max_seq_len there right now.
        self.model_kwargs["max_position_embeddings"] = min(
            self.max_seq_len,
            self.model_kwargs.get("max_position_embeddings", self.max_seq_len),
        )

    # TODO: Remove this after the PyTorch backend is fully migrated to TorchLlmArgs from ExecutorConfig
    def get_pytorch_backend_config(self) -> "_AutoDeployLlmArgs":
        """Return the _AutoDeployLlmArgs (self) object."""
        return self


def update_llm_args_with_extra_dict(
        llm_args: Dict,
        llm_args_dict: Dict,
        extra_llm_api_options: Optional[str] = None) -> Dict:

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
        "cache_transceiver_config": CacheTransceiverConfig,
        "lora_config": LoraConfig,
    }
    for field_name, field_type in field_mapping.items():
        if field_name in llm_args_dict:
            if field_name == "speculative_config":
                llm_args_dict[field_name] = field_type.from_dict(
                    llm_args_dict[field_name])
            else:
                llm_args_dict[field_name] = field_type(
                    **llm_args_dict[field_name])
            extra_llm_str = f"because it's specified in {extra_llm_api_options}" if extra_llm_api_options else ""
            logger.warning(f"Overriding {field_name} {extra_llm_str}")

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
