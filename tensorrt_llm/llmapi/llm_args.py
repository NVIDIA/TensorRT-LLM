import ast
import copy
import functools
import json
import math
import os
import types
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, EnumMeta
from pathlib import Path
from typing import (TYPE_CHECKING, Any, ClassVar, Dict, List, Literal, Optional,
                    Set, Tuple, Type, TypeAlias, TypeVar, Union, get_args,
                    get_origin)

import torch
import yaml
from pydantic import BaseModel
from pydantic import Field as PydanticField
from pydantic import PrivateAttr, field_validator, model_validator
from strenum import StrEnum
from transformers import PreTrainedTokenizerBase

from tensorrt_llm.lora_helper import (LoraConfig,
                                      get_default_trtllm_modules_to_hf_modules)

from .._utils import mpi_rank

if TYPE_CHECKING:
    from tensorrt_llm._torch.pyexecutor.config import PyTorchConfig

# yapf: disable
# isort: off
from ..bindings.executor import (BatchingType as _BatchingType,
                                 CacheTransceiverBackendType as _CacheTransceiverBackendType,
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
                                 SchedulerConfig as _SchedulerConfig,
                                 GuidedDecodingConfig as _GuidedDecodingConfig) # isort: skip
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
from .utils import generate_api_docs_as_docstring, get_type_repr

# TODO[chunweiy]: move the following symbols back to utils scope, and remove the following import

TypeBaseModel = TypeVar("T", bound=BaseModel)


def Field(default: Any = ...,
          *,
          status: Optional[Literal["prototype", "beta", "deprecated"]] = None,
          **kwargs: Any) -> Any:
    """Custom Field wrapper that adds status to json_schema_extra.

    Args:
        default: The default value for the field
        status: Optional status indicator that gets added to json_schema_extra.
            - None: Stable.
            - "beta": Recommended for use per the latest documentation.
            - "prototype": Not yet stable and subject to breaking changes; intended for experimentation only.
        **kwargs: All other arguments passed to the original Pydantic Field

    Returns:
        A Pydantic FieldInfo object with the status added to json_schema_extra if provided
    """

    if status is not None:
        json_schema_extra = kwargs.get('json_schema_extra', {})
        if isinstance(json_schema_extra, dict):
            json_schema_extra['status'] = status
        else:
            # If json_schema_extra is not a dict, create a new dict with the status
            json_schema_extra = {'status': status}
        kwargs['json_schema_extra'] = json_schema_extra

    return PydanticField(default, **kwargs)


class StrictBaseModel(BaseModel):
    """
    A base model that forbids arbitrary fields.
    """

    class Config:
        extra = "forbid"  # globally forbid arbitrary fields


class CudaGraphConfig(StrictBaseModel):
    """
    Configuration for CUDA graphs.
    """
    # List of batch sizes to create CUDA graphs for.
    batch_sizes: Optional[List[int]] = Field(
        default=None,
        description="List of batch sizes to create CUDA graphs for.")

    max_batch_size: int = Field(
        default=0, description="Maximum batch size for CUDA graphs.")

    enable_padding: bool = Field(
        default=False,
        description=
        "If true, batches are rounded up to the nearest cuda_graph_batch_size. This is usually a net win for performance."
    )

    @field_validator('max_batch_size')
    @classmethod
    def validate_cuda_graph_max_batch_size(cls, v):
        """Validate cuda_graph_config.max_batch_size is non-negative."""
        if v < 0:
            raise ValueError(
                "cuda_graph_config.max_batch_size must be non-negative")
        return v

    @staticmethod
    def _generate_cuda_graph_batch_sizes(max_batch_size: int,
                                         enable_padding: bool) -> List[int]:
        """Generate a list of batch sizes for CUDA graphs.

        Args:
            max_batch_size: Maximum batch size to generate up to
            enable_padding: Whether padding is enabled, which affects the batch size distribution

        Returns:
            List of batch sizes to create CUDA graphs for
        """
        if enable_padding:
            batch_sizes = [1, 2, 4] + [i * 8 for i in range(1, 17)]
        else:
            batch_sizes = list(range(1, 32)) + [32, 64, 128]

        # Add powers of 2 up to max_batch_size
        batch_sizes += [
            2**i for i in range(8, math.ceil(math.log(max_batch_size, 2)))
        ]

        # Filter and sort batch sizes
        batch_sizes = sorted(
            [size for size in batch_sizes if size <= max_batch_size])

        # Add max_batch_size if not already included
        if max_batch_size != batch_sizes[-1]:
            batch_sizes.append(max_batch_size)

        return batch_sizes


class BaseSparseAttentionConfig(StrictBaseModel):
    """
    Configuration for sparse attention.
    """

    @classmethod
    def from_dict(cls, data: dict):
        # dispatch to the correct sparse attention config
        config_classes = {
            "rocket": RocketSparseAttentionConfig,
            "dsa": DeepSeekSparseAttentionConfig,
        }

        algorithm = data.get("algorithm", None)
        if algorithm is None:
            raise ValueError(f"Sparse attention algorithm is required")

        config_class = config_classes.get(algorithm.lower())
        if config_class is None:
            raise ValueError(f"Invalid algorithm: {algorithm}")

        # Remove 'algorithm' before passing to subclass constructor
        # It's a ClassVar in subclasses, and used for dispatching to the correct subclass
        data = {k: v for k, v in data.items() if k != 'algorithm'}
        return config_class(**data)

    def _check_fields(self):
        pass

    def supports_backend(self, backend: str) -> bool:
        """
        Override if the speculation algorithm does not support
        a subset of the possible backends.
        """
        return True


class RocketSparseAttentionConfig(BaseSparseAttentionConfig):
    """
    Configuration for RocketKV sparse attention.
    """
    algorithm: ClassVar[str] = "rocket"
    window_size: Optional[int] = Field(
        default=None, description="The window size for snap KV.")
    kernel_size: Optional[int] = Field(
        default=None, description="The kernel size for snap KV.")
    topr: Optional[Union[int, float]] = Field(default=76, description="Top-r")
    topk: Optional[int] = Field(default=128, description="Top-k")
    prompt_budget: Optional[int] = Field(default=1266,
                                         description="Prompt budget")
    page_size: Optional[int] = Field(default=3, description="Page size")

    @classmethod
    def from_dict(cls, data: dict):
        return cls(**data)

    def supports_backend(self, backend: str) -> bool:
        return backend == "pytorch"


class DeepSeekSparseAttentionConfig(BaseSparseAttentionConfig):
    """
    Configuration for DeepSeek Sparse Attention.
    """
    algorithm: ClassVar[str] = "dsa"
    index_n_heads: Optional[int] = Field(
        default=None, description="The number of heads for the indexer.")
    index_head_dim: Optional[int] = Field(
        default=None, description="The dimension of the indexer heads.")
    index_topk: Optional[int] = Field(default=None,
                                      description="The topk for the indexer.")
    indexer_max_chunk_size: Optional[int] = Field(
        default=None, description="The maximum chunk size for the indexer.")

    @classmethod
    def from_dict(cls, data: dict):
        return cls(**data)

    def supports_backend(self, backend: str) -> bool:
        return backend == "pytorch"


class MoeConfig(StrictBaseModel):
    """
    Configuration for MoE.
    """
    backend: Literal["CUTLASS", "CUTEDSL", "WIDEEP", "TRTLLM", "DEEPGEMM",
                     "VANILLA",
                     "TRITON"] = Field(default='CUTLASS',
                                       description="MoE backend to use.")

    max_num_tokens: Optional[int] = Field(
        default=None,
        description=
        "If set, at most max_num_tokens tokens will be sent to torch.ops.trtllm.fused_moe at the same time. If the number of tokens exceeds max_num_tokens, the input tensors will be split into chunks and a for loop will be used."
    )

    load_balancer: Optional[Union[object, str]] = Field(
        default=None,
        description="Configuration for MoE load balancing.",
        json_schema_extra={"type": "Union[MoeLoadBalancerConfig, dict, str]"})

    disable_finalize_fusion: bool = Field(
        default=False,
        description=
        "Disable FC2+finalize kernel fusion in CUTLASS MoE backend. Setting this to True recovers deterministic numerical behavior with top-k > 2."
    )

    use_low_precision_moe_combine: bool = Field(
        default=False,
        description=
        "Use low precision combine in MoE operations (only for NVFP4 quantization). When enabled, uses lower precision for combining expert outputs to improve performance."
    )

    @classmethod
    def from_dict(cls, data: dict):
        return cls(**data)


class AttentionDpConfig(StrictBaseModel):
    """
    Configuration for attention DP.
    """
    enable_balance: bool = Field(default=False,
                                 description="Whether to enable balance.")
    timeout_iters: int = Field(
        default=50, description="The number of iterations to timeout.")
    batching_wait_iters: int = Field(
        default=10,
        description="The number of iterations to wait for batching.")

    @classmethod
    def from_dict(cls, data: dict):
        return cls(**data)


class _ParallelConfig(StrictBaseModel):
    """The model distribution configs for LLM."""
    tp_size: int = 1
    pp_size: int = 1
    cp_size: int = 1
    gpus_per_node: int = 8
    # Set default for MoE fields to -1 to trigger auto-calculation in Mapping
    moe_cluster_size: int = -1
    moe_tp_size: int = -1
    moe_ep_size: int = -1
    cp_config: dict = Field(default_factory=dict)
    enable_attention_dp: bool = False
    enable_lm_head_tp_in_adp: bool = False

    _devices: Optional[List[int]] = PrivateAttr(default=None)

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
    def world_size(self) -> int:
        return self.tp_size * self.pp_size * self.cp_size

    @property
    def world_size_per_node(self) -> int:
        world_size = self.world_size
        total_nodes = math.ceil(world_size / self.gpus_per_node)
        return world_size // total_nodes  #TODO is this right?

    @world_size.setter
    def world_size(self, world_size: int):
        if world_size != self.tp_size * self.pp_size * self.cp_size:
            raise ValueError(
                f"world_size {world_size} should be equal to tp_size * pp_size * cp_size {self.tp_size * self.pp_size * self.cp_size} "
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
                       enable_lm_head_tp_in_adp=self.enable_lm_head_tp_in_adp,
                       moe_cluster_size=self.moe_cluster_size,
                       moe_tp_size=self.moe_tp_size,
                       moe_ep_size=self.moe_ep_size)


class CalibConfig(StrictBaseModel):
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


class DecodingBaseConfig(StrictBaseModel):
    # The number of the drafter layers.
    max_draft_len: Optional[int] = None
    # The number of draft tokens in the draft tokens tree.
    # If it's a linear tree, each draft layer will only generate one draft token.
    # In this case, max_draft_len == max_total_draft_tokens.
    # If it's a static or dynamic tree, each draft layer may generate more than one draft token.
    # In this case, max_total_draft_tokens >= max_draft_len.
    max_total_draft_tokens: Optional[int] = None
    speculative_model_dir: Optional[Union[str, Path]] = None

    # PyTorch only.
    # When specified, speculation will be disabled at batch sizes above
    # this value. Otherwise, speculation will always be on.
    max_concurrency: Optional[int] = None

    load_format: Optional[str] = None
    # PyTorch only.
    # Rolling average window size (N) for acceptance length across completed requests.
    # If not set or set to 0, the feature is disabled.
    acceptance_window: Optional[int] = None
    # PyTorch only.
    # Threshold for average acceptance length; speculation will be disabled
    # permanently once the rolling average over the last N completed requests
    # (N = acceptance_window) drops below this value.
    acceptance_length_threshold: Optional[float] = None

    # Validate acceptance controls at field level so they run on model creation
    @field_validator('acceptance_window')
    @classmethod
    def _validate_acceptance_window(cls, v: Optional[int]):
        if v is None:
            return v
        if v < 0:
            raise ValueError(
                f"acceptance_window must be >= 0 (0 disables), got {v}")
        return v

    @field_validator('acceptance_length_threshold')
    @classmethod
    def _validate_acceptance_length_threshold(cls, v: Optional[float]):
        if v is None:
            return v
        if v < 0:
            raise ValueError(
                f"acceptance_length_threshold must be >= 0, got {v}")
        return v

    # If set, drafting is allowed to use chain drafter.
    _allow_chain_drafter: bool = PrivateAttr(True)
    # If set, drafting uses greedy sampling, irrespective of sampling parameters.
    _allow_greedy_draft_tokens: bool = PrivateAttr(True)

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
            "SaveState": SaveHiddenStatesDecodingConfig,
            "UserProvided": UserProvidedDecodingConfig,
            "AUTO": AutoDecodingConfig,
        }

        config_class = config_classes.get(decoding_type)
        if config_class is None:
            raise ValueError(f"Invalid decoding type: {decoding_type}")
        data.pop("decoding_type")

        return config_class(**data)

    def _check_fields(self):
        pass

    def supports_backend(self, backend: str) -> bool:
        """
        Override if the speculation algorithm does not support
        a subset of the possible backends.
        """
        return True

    def validate(self) -> None:
        """
        Do any additional error checking here.
        """

    @functools.cached_property
    def spec_dec_mode(self):
        # spec_dec_mode has more functionality than the raw decoding_mode string.
        # Use an alias for the import here to avoid name collisions with the one for the
        # TRT backend.
        from tensorrt_llm._torch.speculative.interface import \
            SpeculativeDecodingMode as TorchSpeculativeDecodingMode
        return TorchSpeculativeDecodingMode.from_string(
            self.decoding_type.upper())


class KvCacheConnectorConfig(StrictBaseModel):
    """
    Configuration for the KV Cache Connector.
    """
    connector_module: str = Field(
        ...,
        description=
        "The import path to the connector module. It will be imported with `importlib.import_module`."
    )
    connector_scheduler_class: str = Field(
        ..., description="The class name of the scheduler within the module.")
    connector_worker_class: str = Field(
        ..., description="The class name of the worker within the module.")


class MedusaDecodingConfig(DecodingBaseConfig):
    medusa_choices: Optional[List[List[int]]] = None
    num_medusa_heads: Optional[int] = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.max_total_draft_tokens = self.max_draft_len  # Current Medusa only support linear tree

    @classmethod
    def from_dict(cls, data: dict):
        return cls(**data)

    decoding_type: ClassVar[str] = "Medusa"

    def supports_backend(self, backend: str) -> bool:
        return backend not in ("pytorch", "_autodeploy")


class EagleDecodingConfig(DecodingBaseConfig):
    eagle_choices: Optional[List[List[int]]] = None
    greedy_sampling: Optional[bool] = True
    posterior_threshold: Optional[float] = None
    # Whether to use dynamic tree.
    use_dynamic_tree: Optional[bool] = False
    # The topK value for each layer when enable dynamic tree.
    dynamic_tree_max_topK: Optional[int] = None
    # The number of eagle layer. will not be used in pytorch flow, just for compatibility with TRT flow
    num_eagle_layers: Optional[int] = None
    # The number of non-leaves in each layer.
    max_non_leaves_per_layer: Optional[int] = None
    eagle3_one_model: Optional[bool] = True
    eagle3_layers_to_capture: Optional[Set[int]] = None

    def __init__(self, **kwargs):
        super().__init__()
        for attr_name, attr_value in kwargs.items():
            if attr_name == 'max_draft_len':
                self.num_eagle_layers = attr_value
                self.max_total_draft_tokens = attr_value  # If using linear-tree, the max_total_draft_tokens is the same as max_draft_len
            # Convert the data type of Eagle choice from str to List[List[int]]
            if attr_name == 'eagle_choices' and attr_value is not None:
                logger.warning(
                    "NOTE: The Draft token tree is still under development, PLEASE DO NOT USE IT !!!"
                )
                if not isinstance(attr_value, list):
                    if isinstance(attr_value, str):
                        attr_value = ast.literal_eval(
                            attr_value.replace(" ", ""))
                    else:
                        raise ValueError(
                            "Wrong eagle choices type. Eagle choices should be a List[List[int]] or a string like [[0], [1], [2], [0, 0], [0, 1]]."
                        )
            setattr(self, attr_name, attr_value)

        assert self.max_draft_len is not None, "max_draft_len is required for Eagle"

        # Static tree logic
        # Checks whether the input eagle choices is valid
        # and reset the max_draft_len and num_eagle_layers if necessary
        if self.eagle_choices is not None:
            # If eagle_choices is provided, use_dynamic_tree should not be used
            assert not self.use_dynamic_tree, "If eagle_choices is provided, use_dynamic_tree need to be False"

            # Get num_eagle_layers from eagle_choices
            num_eagle_layers_from_choices = self.check_eagle_choices()
            if num_eagle_layers_from_choices != self.num_eagle_layers:
                logger.warning(
                    f"Base on the input choices, reset the num_eagle_layers(max_draft_len) from {self.num_eagle_layers} to {num_eagle_layers_from_choices}"
                )
                self.num_eagle_layers = num_eagle_layers_from_choices
                self.max_draft_len = num_eagle_layers_from_choices

            # Each draft node has a path(choice) from the root to it.
            # So the number of choices also represents the number of max draft nodes.
            self.max_total_draft_tokens = len(self.eagle_choices)

        # Dynamic tree logic
        if self.use_dynamic_tree:
            assert self.eagle_choices is None, "If use_dynamic_tree is True, eagle_choices should be None"
            assert self.max_draft_len is not None and self.max_draft_len > 0, "max_draft_len should be provided, which indicates the number of drafter layers"
            assert self.dynamic_tree_max_topK is not None and self.dynamic_tree_max_topK > 0, "dynamic_tree_max_topK should be provided, which indicates the number of nodes to expand each time"
            assert self.max_total_draft_tokens is not None and self.max_total_draft_tokens > 0, "max_total_draft_tokens should be provided, which indicates the total nodes of the final draft tree. (exclude the root node)"

    @classmethod
    def from_dict(cls, data: dict):
        return cls(**data)

    decoding_type: ClassVar[str] = "Eagle"

    def validate(self) -> None:
        if self.speculative_model_dir is None:
            raise ValueError("Draft model must be provided for EAGLE")

    def check_eagle_choices(self):
        # 1) Check connectivity
        unique_choices = set(
            tuple(sub_choice)
            for sub_choice in self.eagle_choices)  # remove repeated choices
        self.eagle_choices = sorted([list(t) for t in unique_choices],
                                    key=lambda x: (len(x), x))  # sort choices
        for choice in self.eagle_choices:
            if len(choice) > 1:
                assert choice[
                    0:
                    -1] in self.eagle_choices, f"Error: choice {choice} is not connected"

        # 2) Get num_eagle_layers_from_choices
        num_eagle_layers_from_choices = max(
            len(choice) for choice in self.eagle_choices)

        return num_eagle_layers_from_choices

    @functools.cached_property
    def spec_dec_mode(self):
        from tensorrt_llm._torch.speculative.interface import \
            SpeculativeDecodingMode as TorchSpeculativeDecodingMode
        if self.eagle3_one_model:
            return TorchSpeculativeDecodingMode.EAGLE3_ONE_MODEL
        return TorchSpeculativeDecodingMode.EAGLE3

    @functools.cached_property
    def num_capture_layers(self) -> int:
        """
        Returns the number of layers to capture of the target model.
        If eagle3_layers_to_capture is not None, return the length of the set.
        Otherwise, assume Eagle3 base set and return 3.
        """
        if self.eagle3_layers_to_capture is not None:
            return len(self.eagle3_layers_to_capture)
        return 3

    @functools.cached_property
    def is_linear_tree(self) -> bool:
        if self.eagle_choices is None and self.use_dynamic_tree is False:
            return True
        return False


class SaveHiddenStatesDecodingConfig(DecodingBaseConfig):
    output_directory: str
    write_interval: int = 20
    file_prefix: str = "data"
    eagle3_layers_to_capture: Optional[Set[int]] = None

    max_total_draft_tokens: Optional[int] = Field(default=1, init=False)
    eagle_choices: Optional[List[List[int]]] = Field(default=None, init=False)

    def model_post_init(self, __context):
        self._last_hidden_in_save = True
        if self.eagle3_layers_to_capture is None:
            self._last_hidden_in_save = False
        elif -1 not in self.eagle3_layers_to_capture:
            self._last_hidden_in_save = False
            self.eagle3_layers_to_capture.add(-1)

    @classmethod
    def from_dict(cls, data: dict):
        return cls(**data)

    decoding_type: ClassVar[str] = "SaveState"

    def validate(self) -> None:
        if self.output_directory is None or not self.eagle3_layers_to_capture:
            raise ValueError(
                "Save directory and layers to capture must be provided")

    @functools.cached_property
    def spec_dec_mode(self):
        from tensorrt_llm._torch.speculative.interface import \
            SpeculativeDecodingMode as TorchSpeculativeDecodingMode
        return TorchSpeculativeDecodingMode.SAVE_HIDDEN_STATES

    @functools.cached_property
    def num_capture_layers(self):
        """
        Returns the number of layers to capture of the target model.
        If eagle3_layers_to_capture is not None, return the length of the set.
        Otherwise, assume Eagle3 base set and return 3 + 1 (for post norm last hidden state).
        """
        if self.eagle3_layers_to_capture is None:
            return 4
        return len(self.eagle3_layers_to_capture)


class UserProvidedDecodingConfig(DecodingBaseConfig):
    # Cannot use real type annotations due to circular imports
    drafter: object  # Type is Drafter
    resource_manager: object = None  # Type is Optional[ResourceManager]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.max_total_draft_tokens = self.max_draft_len  # Current UserProvided only support linear tree

    @classmethod
    def from_dict(cls, data: dict):
        return cls(**data)

    decoding_type: ClassVar[str] = "User_Provided"


class NGramDecodingConfig(DecodingBaseConfig):
    """
    Configuration for NGram drafter speculative decoding.

    Arguments:
        max_draft_len: int
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
    max_matching_ngram_size: int = 0
    is_keep_all: bool = True
    is_use_oldest: bool = True
    is_public_pool: bool = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.max_total_draft_tokens = self.max_draft_len  # Current NGram only support linear tree

    @classmethod
    def from_dict(cls, data: dict):
        return cls(**data)

    decoding_type: ClassVar[str] = "NGram"

    def supports_backend(self, backend: str) -> bool:
        return backend == "pytorch"


class DraftTargetDecodingConfig(DecodingBaseConfig):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.max_total_draft_tokens = self.max_draft_len  # Current DraftTarget only support linear tree

    @classmethod
    def from_dict(cls, data: dict):
        return cls(**data)

    decoding_type: ClassVar[str] = "Draft_Target"

    def supports_backend(self, backend: str) -> bool:
        return backend == "pytorch"


class MTPDecodingConfig(DecodingBaseConfig):
    num_nextn_predict_layers: int = 1
    use_relaxed_acceptance_for_thinking: bool = False
    relaxed_topk: int = 1
    relaxed_delta: float = 0.
    use_mtp_vanilla: bool = False
    mtp_eagle_one_model: bool = True

    # TODO: remove this after distinguishing `max_draft_len` and `num_nextn_predict_layers`
    # Now we need a flag when MTPDecodingConfig is updated by PyTorchModelEngine.
    num_nextn_predict_layers_from_model_config: int = 1

    # TODO: Hard code for DeepSeek R1
    # When encounter <think>, start thinking phase.
    # When encounter </think>, end thinking phase.
    # <think> [thinking phase] </think> [real output]
    BEGIN_THINKING_PHASE_TOKEN: int = 128798
    END_THINKING_PHASE_TOKEN: int = 128799

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if 'num_nextn_predict_layers' in kwargs:
            self.max_draft_len = kwargs['num_nextn_predict_layers']
            self.max_total_draft_tokens = kwargs[
                'num_nextn_predict_layers']  # Current MTP only support linear tree

    @classmethod
    def from_dict(cls, data: dict):
        out = cls(**data)
        out.max_draft_len = out.num_nextn_predict_layers
        out.max_total_draft_tokens = out.num_nextn_predict_layers  # Current MTP only support linear tree
        return out

    decoding_type: ClassVar[str] = "MTP"

    def supports_backend(self, backend: str) -> bool:
        return backend == "pytorch"

    @functools.cached_property
    def num_capture_layers(self) -> int:
        if not self.use_mtp_vanilla and not self.mtp_eagle_one_model:
            return 1
        return 0

    @functools.cached_property
    def spec_dec_mode(self):
        from tensorrt_llm._torch.speculative.interface import \
            SpeculativeDecodingMode as TorchSpeculativeDecodingMode
        if self.num_nextn_predict_layers_from_model_config == 1 and not self.use_mtp_vanilla and self.mtp_eagle_one_model:
            return TorchSpeculativeDecodingMode.MTP_EAGLE_ONE_MODEL
        elif self.num_nextn_predict_layers_from_model_config == 1 and not self.use_mtp_vanilla and not self.mtp_eagle_one_model:
            return TorchSpeculativeDecodingMode.MTP_EAGLE
        return TorchSpeculativeDecodingMode.MTP


class AutoDecodingConfig(DecodingBaseConfig):
    """
    Configuration for auto speculative decoding.

    This config will automatically select a good, draft-model free
    speculation algorithm with some heuristic.

    Attributes that are inherited from the base class are ignored.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.max_total_draft_tokens = self.max_draft_len  # Current Auto only support linear tree

    @classmethod
    def from_dict(cls, data: dict):
        return cls(**data)

    decoding_type: ClassVar[str] = "AUTO"

    def supports_backend(self, backend: str) -> bool:
        return backend == "pytorch"


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
            assert issubclass(cls, StrictBaseModel)
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

    @classmethod
    def from_pybind(cls: Type[TypeBaseModel],
                    pybind_instance: "PybindMirror") -> TypeBaseModel:
        """Construct an instance of the given class from the fields in the given
        pybind class instance.

        Args:
            cls: Type of the class to construct, must be a subclass of pydantic
                 BaseModel
            pybind_instance: Instance of the pybind class to construct from its
                             fields

        Notes:
            When a field value is None in the pybind class, but it's not
            optional and has a default value in the BaseModel class, it would
            get the default value defined in the BaseModel class.

        Returns:
            Instance of the given class, populated with the fields of the given
            pybind instance
        """  # noqa: D205
        assert issubclass(cls, BaseModel)

        # Some of the fields are optional in the C++ class but in python they aren't
        # optional and have a default value, so copy the value from C++ instance
        # only if it has a value, so otherwise the default value defined in the
        # python class would be set.
        def _is_optional_type(annotation: Any) -> bool:
            """Returns True if a type annotation represents an Optional type
            (Optional[X]) or a Union type that includes None (Union[X, Y, None]
            or X | Y | None).
            """  # noqa: D205
            origin = get_origin(annotation)
            args = get_args(annotation)

            # Union is for Optional[x]
            # UnionType is for the new | operation in Python 3.10+
            return (origin is Union
                    or origin is types.UnionType) and type(None) in args

        fields_non_optional_with_default_value_in_basemodel = {
            field_name
            for field_name, field_info in cls.model_fields.items()
            if not (_is_optional_type(field_info.annotation)
                    and field_info.is_required())
        }

        kwargs = {}
        cpp_fields = PybindMirror.get_pybind_variable_fields(
            type(pybind_instance))
        for field_name in cpp_fields:
            field_value = getattr(pybind_instance, field_name)
            if field_value is not None or field_name not in fields_non_optional_with_default_value_in_basemodel:
                kwargs[field_name] = field_value
        return cls(**kwargs)


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
class DynamicBatchConfig(StrictBaseModel, PybindMirror):
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
class SchedulerConfig(StrictBaseModel, PybindMirror):
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
class PeftCacheConfig(StrictBaseModel, PybindMirror):
    """
    Configuration for the PEFT cache.
    """
    num_host_module_layer: int = Field(
        default=0,
        description=
        "number of max sized 1-layer 1-module adapterSize=1 sets of weights that can be stored in host cache"
        ", affects host cache size and overrides value of host_cache_size")
    num_device_module_layer: int = Field(
        default=0,
        description=
        "number of max sized 1-layer 1-module sets of weights that can be stored in device cache"
        ", affects device cache size and overrides value of device_cache_percent"
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
    device_cache_percent: float = Field(
        default=0.02,
        description=
        "Proportion of free device memory after engine load to use for cache, as a fraction from 0 to 1"
    )
    host_cache_size: int = Field(
        default=1024**3, description="size in bytes to use for host cache")
    lora_prefetch_dir: Optional[str] = Field(
        default=None,
        description=
        "folder to store the LoRA weights we hope to load during engine initialization, currently not supported"
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
        self.max_total_draft_tokens = self.max_draft_len  # Current Lookahead only support linear tree
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

    def supports_backend(self, backend: str) -> bool:
        return backend not in ("pytorch", "_autodeploy")

    decoding_type: ClassVar[str] = "Lookahead"


SpeculativeConfig: TypeAlias = Optional[Union[
    DraftTargetDecodingConfig,
    EagleDecodingConfig,
    LookaheadDecodingConfig,
    MedusaDecodingConfig,
    MTPDecodingConfig,
    NGramDecodingConfig,
    UserProvidedDecodingConfig,
    SaveHiddenStatesDecodingConfig,
    AutoDecodingConfig,
]]

SparseAttentionConfig: TypeAlias = Union[
    RocketSparseAttentionConfig,
    DeepSeekSparseAttentionConfig,
]


@PybindMirror.mirror_pybind_fields(_KvCacheConfig)
class KvCacheConfig(StrictBaseModel, PybindMirror):
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
        default=0.9,
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
    attention_dp_events_gather_period_ms: int = Field(
        default=5,
        description=
        "The period in milliseconds to gather attention DP events across ranks."
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
    use_uvm: bool = Field(default=False,
                          description="Whether to use UVM for the KV cache.")
    max_gpu_total_bytes: int = Field(
        default=0,
        description=
        "The maximum size in bytes of GPU memory that can be allocated for the KV cache. If both `max_gpu_total_bytes` and `free_gpu_memory_fraction` are specified, memory corresponding to the minimum will be allocated."
    )

    # This is a pure python field, not a pybind field. It is only for the Pytorch backend.
    dtype: str = Field(default="auto",
                       description="The data type to use for the KV cache.")

    # This is a pure python field, not a pybind field. It is only for the Pytorch backend.
    mamba_ssm_cache_dtype: Literal[
        "auto", "float16", "bfloat16", "float32"] = Field(
            default="auto",
            description=
            "The data type to use for the Mamba SSM cache. If set to 'auto', the data type will be inferred from the model config."
        )

    tokens_per_block: int = Field(default=32,
                                  description="The number of tokens per block.")

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
            copy_on_partial_reuse=self.copy_on_partial_reuse,
            use_uvm=self.use_uvm,
            attention_dp_events_gather_period_ms=self.
            attention_dp_events_gather_period_ms,
            max_gpu_total_bytes=self.max_gpu_total_bytes)

    @field_validator('free_gpu_memory_fraction')
    @classmethod
    def validate_free_gpu_memory_fraction(cls, v: float):
        """Validates that the fraction is between 0.0 and 1.0."""
        if not 0 <= v <= 1:
            raise ValueError(
                "kv_cache_config.free_gpu_memory_fraction must be a float between 0 and 1"
            )
        return v

    @field_validator('max_gpu_total_bytes')
    @classmethod
    def validate_max_gpu_total_bytes(cls, v: int):
        if v < 0:
            raise ValueError(
                "kv_cache_config.max_gpu_total_bytes must be non-negative")
        return v

    @field_validator('max_attention_window')
    @classmethod
    def validate_max_attention_window(cls, v: Optional[List[int]]):
        # Allow unset
        if v is None:
            return v

        # Must be a non-empty list of positive integers
        if not isinstance(v, list) or len(v) == 0:
            raise ValueError(
                "kv_cache_config.max_attention_window must be a non-empty list of positive integers"
            )
        for i in v:
            if not isinstance(i, int):
                raise ValueError(
                    "kv_cache_config.max_attention_window must contain only integers"
                )
            if i <= 0:
                raise ValueError(
                    "kv_cache_config.max_attention_window values must be positive"
                )
        return v


@PybindMirror.mirror_pybind_fields(_ExtendedRuntimePerfKnobConfig)
class ExtendedRuntimePerfKnobConfig(StrictBaseModel, PybindMirror):
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
class CacheTransceiverConfig(StrictBaseModel, PybindMirror):
    """
    Configuration for the cache transceiver.
    """

    backend: Optional[Literal["DEFAULT", "UCX", "NIXL", "MPI"]] = Field(
        default=None,
        description=
        "The communication backend type to use for the cache transceiver.")

    max_tokens_in_buffer: Optional[int] = Field(
        default=None,
        description="The max number of tokens the transfer buffer can fit.")

    kv_transfer_timeout_ms: Optional[int] = Field(
        default=None,
        gt=0,
        description=
        "Timeout in milliseconds for KV cache transfer. Requests exceeding this timeout will be cancelled."
    )

    def _to_pybind(self):
        return _CacheTransceiverConfig(
            backend=_CacheTransceiverBackendType.from_string(self.backend),
            max_tokens_in_buffer=self.max_tokens_in_buffer,
            kv_transfer_timeout_ms=self.kv_transfer_timeout_ms)


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


class BaseLlmArgs(StrictBaseModel):
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
        status="beta",
        validate_default=True)

    moe_cluster_parallel_size: Optional[int] = Field(
        default=None,
        description="The cluster parallel size for MoE models's expert weights.",
        status="beta")

    moe_tensor_parallel_size: Optional[int] = Field(
        default=None,
        description="The tensor parallel size for MoE models's expert weights.")

    moe_expert_parallel_size: Optional[int] = Field(
        default=None,
        description="The expert parallel size for MoE models's expert weights.")

    enable_attention_dp: bool = Field(
        default=False,
        description="Enable attention data parallel.",
        status="beta")

    enable_lm_head_tp_in_adp: bool = Field(
        default=False,
        description="Enable LM head TP in attention dp.",
        status="prototype")

    cp_config: Optional[dict] = Field(default_factory=dict,
                                      description="Context parallel config.",
                                      status="prototype")

    load_format: Literal['auto', 'dummy'] = Field(
        default='auto',
        description="The format to load the model.",
        json_schema_extra={"type": "Literal['auto', 'dummy']"})

    fail_fast_on_attention_window_too_large: bool = Field(
        default=False,
        description=
        "Fail fast when attention window is too large to fit even a single sequence in the KV cache.",
        status="prototype")

    # LoRA arguments
    enable_lora: bool = Field(default=False, description="Enable LoRA.")

    lora_config: Optional[LoraConfig] = Field(
        default=None, description="LoRA configuration for the model.")

    # Several options from ExecutorConfig, expanded here for less hierarchy
    kv_cache_config: KvCacheConfig = Field(default_factory=KvCacheConfig,
                                           description="KV cache config.")

    enable_chunked_prefill: bool = Field(default=False,
                                         description="Enable chunked prefill.")

    guided_decoding_backend: Optional[Literal["xgrammar", "llguidance"]] = Field(
        default=None,
        description=
        "Guided decoding backend. llguidance is supported in PyTorch backend only."
    )

    batched_logits_processor: Optional[object] = Field(
        default=None,
        description="Batched logits processor.",
        json_schema_extra={
            "type": f"Optional[{get_type_repr(BatchedLogitsProcessor)}]"
        })

    iter_stats_max_iterations: Optional[int] = Field(
        default=None,
        description="The maximum number of iterations for iter stats.",
        status="prototype")

    request_stats_max_iterations: Optional[int] = Field(
        default=None,
        description="The maximum number of iterations for request stats.",
        status="prototype")

    # A handful of options from PretrainedConfig
    peft_cache_config: Optional[PeftCacheConfig] = Field(
        default=None, description="PEFT cache config.", status="prototype")

    scheduler_config: SchedulerConfig = Field(default_factory=SchedulerConfig,
                                              description="Scheduler config.",
                                              status="prototype")

    cache_transceiver_config: Optional[CacheTransceiverConfig] = Field(
        default=None,
        description="Cache transceiver config.",
        status="prototype")

    # Sparse attention config
    sparse_attention_config: Optional[SparseAttentionConfig] = Field(
        default=None,
        description="Sparse attention config.",
        status="prototype")

    # Speculative decoding parameters
    speculative_config: SpeculativeConfig = Field(
        default=None, description="Speculative decoding config.")

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
        default=8192, description="The maximum number of tokens.")

    gather_generation_logits: bool = Field(
        default=False,
        description="Gather generation logits.",
        status="prototype")

    # private fields those are unstable and just for internal use
    num_postprocess_workers: int = Field(
        default=0,
        description=
        "The number of processes used for postprocessing the generated tokens, including detokenization.",
        status="prototype")

    postprocess_tokenizer_dir: Optional[str] = Field(
        default=None,
        description="The path to the tokenizer directory for postprocessing.",
        status="prototype")

    reasoning_parser: Optional[str] = Field(
        default=None,
        description="The parser to separate reasoning content from output.",
        status="prototype")

    # TODO[Superjomn]: To deprecate this config.
    decoding_config: Optional[object] = Field(
        default=None,
        description="The decoding config.",
        json_schema_extra={
            "type": "Optional[tensorrt_llm.llmapi.llm_args.DecodingConfig]"
        },
        status="deprecated",
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
        status="deprecated",
    )

    return_perf_metrics: bool = Field(default=False,
                                      description="Return perf metrics.",
                                      status="prototype")

    orchestrator_type: Optional[Literal["rpc", "ray"]] = Field(
        default=None,
        description=
        "The orchestrator type to use. Defaults to None, which uses MPI.",
        status="prototype",
    )

    _parallel_config: Optional[_ParallelConfig] = PrivateAttr(default=None)
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
    def speculative_model_dir(self) -> Optional[_ModelFormatKind]:
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
            enable_lm_head_tp_in_adp=self.enable_lm_head_tp_in_adp,
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
            model_format = get_model_format(
                self.model, trust_remote_code=self.trust_remote_code)

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
        build_config = getattr(self, "build_config", None)
        if build_config is None:
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
        else:
            assert isinstance(
                build_config,
                BuildConfig), f"build_config is not initialized: {build_config}"
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
    def validate_runtime_args(self):
        if self.max_batch_size is not None and self.max_num_tokens is not None:
            if self.max_batch_size > self.max_num_tokens:
                logger.warning(
                    f"max_batch_size [{self.max_batch_size}] should be less than or equal to max_num_tokens [{self.max_num_tokens}]"
                )
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
                self.max_batch_size = self.build_config.max_batch_size
                logger.warning(
                    f"max_batch_size [{self.max_batch_size}] is overridden by build_config.max_batch_size [{self.build_config.max_batch_size}] in build_config"
                )
        if self.max_num_tokens is not None:
            if self.max_num_tokens > self.build_config.max_num_tokens:
                self.max_num_tokens = self.build_config.max_num_tokens
                logger.warning(
                    f"max_num_tokens [{self.max_num_tokens}] is overridden by build_config.max_num_tokens [{self.build_config.max_num_tokens}] in build_config"
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

        if self.parallel_config.world_size == 1 and self.build_config:
            self.build_config.plugin_config.nccl_plugin = None

        if self.enable_lora and self.backend != 'pytorch':
            self.build_config.plugin_config.lora_plugin = 'auto'
            if self.lora_config is not None:
                self.build_config.lora_config.max_lora_rank = self.lora_config.max_lora_rank

        if hasattr(self,
                   'enable_prompt_adapter') and self.enable_prompt_adapter:
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
            if not self.speculative_config.supports_backend(self.backend):
                raise ValueError(
                    f"Speculation type {self.speculative_config.decoding_type} does not "
                    f"support backend {self.backend}")

            # Below, we only need to set speculative_decoding_mode/decoding_config for speculation
            # on the TRT backend.
            if isinstance(self.speculative_config, LookaheadDecodingConfig):
                max_draft_len = self.speculative_config.calculate_speculative_resource(
                )[2]
                assert max_draft_len > 0
                self.build_config.speculative_decoding_mode = SpeculativeDecodingMode.LOOKAHEAD_DECODING
                self.build_config.max_draft_len = max(
                    self.build_config.max_draft_len, max_draft_len)
                self.decoding_config = DecodingConfig(
                    decoding_mode=DecodingMode.Lookahead(),
                    lookahead_decoding_config=PybindMirror.maybe_to_pybind(
                        self.speculative_config))

            elif isinstance(self.speculative_config, MedusaDecodingConfig):
                assert self.speculative_config.max_draft_len > 0
                self.build_config.speculative_decoding_mode = SpeculativeDecodingMode.MEDUSA
                self.build_config.max_draft_len = self.speculative_config.max_draft_len
                self.decoding_config = DecodingConfig(
                    decoding_mode=DecodingMode.Medusa(),
                    medusa_choices=self.speculative_config.medusa_choices)

            elif isinstance(self.speculative_config, EagleDecodingConfig):
                assert self.speculative_config.max_draft_len > 0
                assert self.speculative_config.speculative_model_dir is not None, "Path to EAGLE3 weights must be specified."
                self.build_config.max_draft_len = self.speculative_config.max_draft_len
                self.build_config.speculative_decoding_mode = SpeculativeDecodingMode.EAGLE
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

            elif isinstance(self.speculative_config, NGramDecodingConfig):
                assert self.backend in ['pytorch', '_autodeploy']
                assert self.speculative_config.max_draft_len > 0 and self.speculative_config.max_matching_ngram_size > 0
                self.build_config.speculative_decoding_mode = SpeculativeDecodingMode.NGRAM
                self.build_config.max_draft_len = self.speculative_config.max_draft_len

            elif isinstance(self.speculative_config, DraftTargetDecodingConfig):
                assert self.backend in ['pytorch']
                assert self.speculative_config.max_draft_len > 0
                assert self.speculative_config.speculative_model_dir is not None, "Path to draft model must be specified."
                self.build_config.speculative_decoding_mode = SpeculativeDecodingMode.DRAFT_TOKENS_EXTERNAL
                self.build_config.max_draft_len = self.speculative_config.max_draft_len

            elif isinstance(self.speculative_config, MTPDecodingConfig):
                assert self.speculative_config.num_nextn_predict_layers > 0
                self.speculative_config.max_draft_len = self.speculative_config.num_nextn_predict_layers

            elif isinstance(self.speculative_config,
                            UserProvidedDecodingConfig):
                assert self.backend in ['pytorch', '_autodeploy']
                self.build_config.speculative_decoding_mode = SpeculativeDecodingMode.USER_PROVIDED
                self.build_config.max_draft_len = self.speculative_config.max_draft_len

            elif isinstance(self.speculative_config, AutoDecodingConfig):
                assert self.backend in ['pytorch', '_autodeploy']
                self.build_config.speculative_decoding_mode = SpeculativeDecodingMode.AUTO
                self.build_config.max_draft_len = self.speculative_config.max_draft_len

            elif isinstance(self.speculative_config,
                            SaveHiddenStatesDecodingConfig):
                assert self.backend in ['pytorch']
                logger.warning(
                    "SaveHiddenStatesDecodingConfig is active, setting max_batch_size to 1, disabling overlap scheduler, and setting cuda_graph_config to None"
                )
                self.build_config.max_batch_size = 1
                self.max_batch_size = 1
                self.disable_overlap_scheduler = True
                self.cuda_graph_config = None
                self.build_config.speculative_decoding_mode = SpeculativeDecodingMode.SAVE_HIDDEN_STATES
                self.build_config.max_draft_len = 1
                self.speculative_config.max_draft_len = 1

            else:
                raise ValueError(
                    f"Unrecognized speculative config type {type(self.speculative_config)}"
                )

        else:
            self.decoding_config = None

        self._speculative_model = getattr(self.speculative_config,
                                          "speculative_model_dir", None)
        speculative_model_obj = _ModelWrapper(
            self._speculative_model
        ) if self._speculative_model is not None else None
        if self._speculative_model and speculative_model_obj.is_local_model:
            self._speculative_model_format = _ModelFormatKind.HF

        return self

    @model_validator(mode="after")
    def validate_lora_config_consistency(self):
        if self.lora_config:
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

    @model_validator(mode="after")
    def validate_peft_cache_config(self):
        if self.peft_cache_config is not None and self.peft_cache_config.lora_prefetch_dir is not None:
            raise ValueError(
                f"lora_prefetch_dir was set to '{self.peft_cache_config.lora_prefetch_dir}' "
                "while LoRA prefetch is not supported")
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
        self._parallel_config = _ParallelConfig(
            tp_size=tp_size,
            pp_size=pp_size,
            cp_size=cp_size,
            gpus_per_node=gpus_per_node,
            moe_cluster_size=moe_cluster_size,
            moe_tp_size=moe_tp_size,
            moe_ep_size=moe_ep_size)

    def get_runtime_sizes(self, ) -> Tuple[int, int, int, int]:
        return (
            self.max_beam_width,
            self.max_num_tokens,
            self.max_seq_len,
            self.max_batch_size,
        )


class TrtLlmArgs(BaseLlmArgs):
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

    # Quantization and calibration configurations
    quant_config: Optional[QuantConfig] = Field(
        default=None, description="Quantization config.", validate_default=True)

    embedding_parallel_mode: str = Field(
        default='SHARDING_ALONG_VOCAB',
        description="The embedding parallel mode.")

    fast_build: bool = Field(default=False, description="Enable fast build.")

    # BuildConfig is introduced to give users a familiar interface to configure the model building.
    build_config: Optional[object] = Field(
        default=None,
        description="Build config.",
        json_schema_extra={"type": f"Optional[{get_type_repr(BuildConfig)}]"})

    # Prompt adapter arguments
    enable_prompt_adapter: bool = Field(default=False,
                                        description="Enable prompt adapter.")

    max_prompt_adapter_token: int = Field(
        default=0, description="The maximum number of prompt adapter tokens.")

    batching_type: Optional[BatchingType] = Field(default=None,
                                                  description="Batching type.")

    normalize_log_probs: bool = Field(
        default=False, description="Normalize log probabilities.")

    # Private attributes
    # This is used to hold the options for convert_checkpoint
    _convert_checkpoint_options: Dict[str,
                                      Any] = PrivateAttr(default_factory=dict)

    @field_validator('calib_config', mode='before')
    @classmethod
    def init_calib_config(cls, v):
        if v is None:
            return CalibConfig()
        return v

    @field_validator("quant_config", mode='before')
    @classmethod
    def validate_quant_config(cls, v, info):
        if v is None:
            v = QuantConfig()
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
    def validate_enable_build_cache(self):
        if not self.enable_build_cache:
            return self
        self.enable_build_cache = BuildCacheConfig() if isinstance(
            self.enable_build_cache, bool) else self.enable_build_cache
        if not isinstance(self.enable_build_cache, BuildCacheConfig):
            raise ValueError(
                f"Invalid build_cache_config: {self.enable_build_cache}")
        return self

    @model_validator(mode="after")
    def validate_kv_cache_dtype(self):
        assert self.kv_cache_config.dtype == "auto", "KvCacheConfig.dtype is not supported by the TensorRT backend."
        return self


class LoadFormat(Enum):
    AUTO = 0
    # Initialize all weights randomly.
    DUMMY = 1
    # Only load the multimodal(vision) encoder weights
    VISION_ONLY = 2


class SamplerType(StrEnum):
    """Enum for sampler type options."""
    TRTLLMSampler = "TRTLLMSampler"
    TorchSampler = "TorchSampler"
    auto = "auto"


class TorchCompileConfig(StrictBaseModel):
    """
    Configuration for torch.compile.
    """
    enable_fullgraph: bool = Field(
        default=True,
        description="Enable full graph compilation in torch.compile.")

    enable_inductor: bool = Field(
        default=False, description="Enable inductor backend in torch.compile.")

    enable_piecewise_cuda_graph: bool = Field(
        default=False,
        description="Enable piecewise CUDA graph in torch.compile.")

    capture_num_tokens: Optional[List[int]] = Field(
        default=None,
        description=
        "List of num of tokens to capture the piecewise CUDA graph for. If not provided, the number of tokens will be the same as cuda_graph_config.batch_sizes."
    )

    @field_validator('capture_num_tokens')
    @classmethod
    def validate_capture_num_tokens(cls, v):
        if v is None:
            return v
        if any(t <= 0 for t in v):
            raise ValueError("capture_num_tokens must contain positive ints.")
        return sorted(set(v), reverse=True)

    enable_userbuffers: bool = Field(
        default=True,
        description=
        "When torch compile is enabled, userbuffers is enabled by default.")

    max_num_streams: int = Field(
        default=1,
        description=
        "The maximum number of CUDA streams to use for torch.compile.")

    @field_validator('max_num_streams')
    @classmethod
    def validate_torch_compile_max_num_streams(cls, v):
        """Validate torch_compile_config.max_num_streams >= 1."""
        if v < 1:
            raise ValueError(
                "torch_compile_config.max_num_streams must be >= 1")
        return v


class TorchLlmArgs(BaseLlmArgs):
    # Just a dummy BuildConfig to allow code reuse with the TrtLlmArgs
    build_config: Optional[object] = Field(
        default=None,
        description="Build config.",
        exclude_from_json=True,
        json_schema_extra={"type": f"Optional[{get_type_repr(BuildConfig)}]"},
        status="deprecated",
    )

    # PyTorch backend specific configurations
    garbage_collection_gen0_threshold: int = Field(
        default=20000,
        description=
        "Threshold for Python garbage collection of generation 0 objects."
        "Lower values trigger more frequent garbage collection.",
        status="beta")

    cuda_graph_config: Optional[CudaGraphConfig] = Field(
        default_factory=CudaGraphConfig,
        description="CUDA graph config.If true, use CUDA graphs for decoding. \
        CUDA graphs are only created for the batch sizes in cuda_graph_config.batch_sizes, \
        and are enabled for batches that consist of decoding requests *only* \
        (the reason is that it's hard to capture a single graph with prefill requests \
        since the input shapes are a function of the sequence lengths).\
         Note that each CUDA graph can use up to 200 MB of extra memory.",
        status="beta")

    attention_dp_config: Optional[AttentionDpConfig] = Field(
        default=None,
        description="Optimized load-balancing for the DP Attention scheduler.",
        status="beta")

    disable_overlap_scheduler: bool = Field(
        default=False,
        description="Disable the overlap scheduler.",
        status="beta")

    moe_config: MoeConfig = Field(default_factory=MoeConfig,
                                  description="MoE config.",
                                  status="beta")

    attn_backend: str = Field(default='TRTLLM',
                              description="Attention backend to use.",
                              status="beta")

    sampler_type: Union[str, SamplerType] = Field(
        default=SamplerType.auto,
        description=
        "The type of sampler to use. Options are TRTLLMSampler, TorchSampler or auto. Defaults to auto, which will use TorchSampler unless BeamSearch is requested.",
        status="beta")

    enable_iter_perf_stats: bool = Field(
        default=False,
        description="Enable iteration performance statistics.",
        status="prototype")

    enable_iter_req_stats: bool = Field(
        default=False,
        description=
        "If true, enables per request stats per iteration. Must also set enable_iter_perf_stats to true to get request stats.",
        status="prototype")

    print_iter_log: bool = Field(default=False,
                                 description="Print iteration logs.",
                                 status="beta")

    perf_metrics_max_requests: int = Field(
        default=0,
        description=
        "The maximum number of requests for perf metrics. Must also set request_perf_metrics to true to get perf metrics.",
        status="prototype")

    batch_wait_timeout_ms: float = Field(
        default=0,
        description=
        "If greater than 0, the request queue might wait up to batch_wait_timeout_ms to receive max_batch_size requests, if fewer than max_batch_size requests are currently available. If 0, no waiting occurs.",
        status="prototype")

    batch_wait_timeout_iters: int = Field(
        default=0,
        description=
        "Maximum number of iterations the scheduler will wait to accumulate new coming requests for improved GPU utilization efficiency. If greater than 0, the scheduler will delay batch processing to gather more requests up to the specified iteration limit. If 0, disables timeout-iters-based batching delays.",
        status="prototype")

    batch_wait_max_tokens_ratio: float = Field(
        default=0,
        description=
        "Token accumulation threshold ratio for batch scheduling optimization. If greater than 0, the scheduler will accumulate requests locally until the total token count reaches batch_wait_max_tokens_ratio * max_num_tokens. This mechanism enhances GPU utilization efficiency by ensuring adequate batch sizes.If 0 disables token-based batching delays.",
        status="prototype")

    torch_compile_config: Optional[TorchCompileConfig] = Field(
        default=None, description="Torch compile config.", status="prototype")

    enable_autotuner: bool = Field(
        default=True,
        description="Enable autotuner only when torch compile is enabled.",
        status="prototype")

    enable_layerwise_nvtx_marker: bool = Field(
        default=False,
        description="If true, enable layerwise nvtx marker.",
        status="beta")

    load_format: Union[str, LoadFormat] = Field(
        default=LoadFormat.AUTO,
        description=
        "How to load the model weights. By default, detect the weight type from the model checkpoint."
    )

    enable_min_latency: bool = Field(
        default=False,
        description=
        "If true, enable min-latency mode. Currently only used for Llama4.",
        status="beta",
    )

    # TODO: make this a per-request parameter
    stream_interval: int = Field(
        default=1,
        description=
        "The iteration interval to create responses under the streaming mode. "
        "Set this to a larger value when the batch size is large, which helps reduce the streaming overhead.",
    )

    force_dynamic_quantization: bool = Field(
        default=False,
        description="If true, force dynamic quantization. Defaults to False.",
        status="prototype",
    )

    allreduce_strategy: Optional[Literal[
        'AUTO', 'NCCL', 'UB', 'MINLATENCY', 'ONESHOT', 'TWOSHOT',
        'LOWPRECISION', 'MNNVL',
        'NCCL_SYMMETRIC']] = Field(default='AUTO',
                                   description="Allreduce strategy to use.",
                                   status="beta")
    checkpoint_loader: Optional[object] = Field(
        default=None,
        description=
        "The checkpoint loader to use for this LLM instance. You may use a custom checkpoint loader by subclassing "
        "`BaseCheckpointLoader` and providing an instance of the subclass here to load weights from a custom "
        "checkpoint format.\n"
        "If neither checkpoint_format nor checkpoint_loader are provided, checkpoint_format will be set to HF "
        "and the default HfCheckpointLoader will be used.\n"
        "If checkpoint_format and checkpoint_loader are both provided, checkpoint_loader will be ignored.",
        json_schema_extra={
            "type":
            "Optional[tensorrt_llm._torch.models.checkpoints.BaseCheckpointLoader]"
        },
        status="prototype",
    )

    checkpoint_format: Optional[str] = Field(
        default=None,
        description=
        "The format of the provided checkpoint. You may use a custom checkpoint format by subclassing "
        "`BaseCheckpointLoader` and registering it with `register_checkpoint_loader`.\n"
        "If neither checkpoint_format nor checkpoint_loader are provided, checkpoint_format will be set to HF "
        "and the default HfCheckpointLoader will be used.\n"
        "If checkpoint_format and checkpoint_loader are both provided, checkpoint_loader will be ignored.",
        status="prototype",
    )

    kv_connector_config: Optional[KvCacheConnectorConfig] = Field(
        default=None,
        description="The config for KV cache connector.",
        status="prototype",
    )

    mm_encoder_only: bool = Field(
        default=False,
        description=
        "Only load/execute the vision encoder part of the full model. Defaults to False.",
        status="prototype",
    )

    ray_worker_extension_cls: Optional[str] = Field(
        default=None,
        description="The full worker extension class name including module path."
        "Allows users to extend the functions of the RayGPUWorker class.",
        status="prototype")

    # PrivateVars
    _quant_config: Optional[QuantConfig] = PrivateAttr(default=None)

    @property
    def quant_config(self) -> QuantConfig:
        if self._quant_config is None:
            self._quant_config = QuantConfig()
        return self._quant_config

    @quant_config.setter
    def quant_config(self, value: QuantConfig):
        self._quant_config = value

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
    def validate_stream_interval(self):
        if self.stream_interval <= 0:
            raise ValueError(
                f"stream_interval must be positive, got {self.stream_interval}")
        return self

    @model_validator(mode="after")
    def validate_checkpoint_format(self):
        if self.checkpoint_format is not None and self.checkpoint_loader is not None:
            logger.warning(
                "checkpoint_format and checkpoint_loader are both provided, "
                "checkpoint_loader will be ignored.")
            self.checkpoint_loader = None

        if self.checkpoint_format is None and self.checkpoint_loader is None:
            logger.info(
                "neither checkpoint_format nor checkpoint_loader were provided, "
                "checkpoint_format will be set to HF.")
            self.checkpoint_format = "HF"

        return self

    @model_validator(mode="after")
    def validate_load_balancer(self) -> 'TorchLlmArgs':
        from .._torch import MoeLoadBalancerConfig
        if isinstance(self.moe_config.load_balancer, str):
            if not os.path.exists(self.moe_config.load_balancer):
                raise FileNotFoundError(
                    f"MoE load balancer config file not found: {self.moe_config.load_balancer}"
                )
            try:
                with open(self.moe_config.load_balancer) as f:
                    moe_load_balancer_config = yaml.safe_load(f)
                self.moe_config.load_balancer = MoeLoadBalancerConfig(
                    **moe_load_balancer_config)
            except Exception as e:
                raise ValueError(
                    f"Failed to load MoE load balancer config file: {self.moe_config.load_balancer}"
                ) from e
        elif isinstance(self.moe_config.load_balancer, dict):
            try:
                self.moe_config.load_balancer = MoeLoadBalancerConfig(
                    **self.moe_config.load_balancer)
            except Exception as e:
                raise ValueError(
                    f"Failed to load MoE load balancer config: {self.moe_config.load_balancer}"
                ) from e
        return self

    @model_validator(mode='after')
    def validate_cuda_graph_config(self) -> 'TorchLlmArgs':
        """Validate CUDA graph configuration.

        Ensures that:
        1. If cuda_graph_config.batch_sizes is provided, cuda_graph_config.max_batch_size must be 0
        2. If cuda_graph_config.batch_sizes is not provided, it is generated based on cuda_graph_config.max_batch_size
        3. If both are provided, cuda_graph_config.batch_sizes must match the generated values
        """
        if self.cuda_graph_config is None:
            return self

        config = self.cuda_graph_config

        if config.batch_sizes:
            config.batch_sizes = sorted(config.batch_sizes)
            if config.max_batch_size != 0:
                if config.batch_sizes != CudaGraphConfig._generate_cuda_graph_batch_sizes(
                        config.max_batch_size, config.enable_padding):
                    raise ValueError(
                        "Please don't set both cuda_graph_config.batch_sizes "
                        "and cuda_graph_config.max_batch_size.\n"
                        f"cuda_graph_config.batch_sizes: {self.cuda_graph_config.batch_sizes}, "
                        f"cuda_graph_config.max_batch_size: {self.cuda_graph_config.max_batch_size}"
                    )
            else:
                config.max_batch_size = max(config.batch_sizes)
        else:
            max_batch_size = config.max_batch_size or 128
            generated_sizes = CudaGraphConfig._generate_cuda_graph_batch_sizes(
                max_batch_size, config.enable_padding)
            config.batch_sizes = generated_sizes
            config.max_batch_size = max_batch_size

        return self

    @model_validator(mode='after')
    def sync_quant_config_with_kv_cache_config_dtype(self) -> 'TorchLlmArgs':
        if self.kv_cache_config is None:
            return self

        assert self.quant_config is not None
        if self.kv_cache_config.dtype == "auto":
            return self
        elif self.kv_cache_config.dtype == 'fp8':
            self.quant_config.kv_cache_quant_algo = QuantAlgo.FP8
        else:
            logger.warning(
                f"Cannot sync quant_config.kv_cache_quant_algo with kv_cache_config.dtype of {self.kv_cache_config.dtype}, "
                "please update the validator")

        return self

    def warn_on_unstable_feature_usage(self) -> 'TorchLlmArgs':
        """Warn on unstable feature usage."""
        set_fields = self.model_dump(exclude_unset=True).keys()

        for field_name in set_fields:
            field_info = self.model_fields.get(field_name)

            if not field_info or not field_info.json_schema_extra:
                continue

            status = field_info.json_schema_extra.get('status', None)

            if status in ('beta', 'prototype'):
                logger.warning(
                    f"The '{field_name}' knob is a '{status}' feature. "
                    "It is not recommended for production use and may change or be removed.",
                )

        return self

    @model_validator(mode='after')
    def validate_attention_dp_config(self) -> 'TorchLlmArgs':
        """Validate attention DP configuration.

        Ensures that:
        1. If attention_dp_config.enable_balance is true, attention_dp_config.batching_wait_iters must be greater or equal to 0
        2. If attention_dp_config.enable_balance is true, attention_dp_config.timeout_iters must be greater or equal to 0
        """
        if self.attention_dp_config is None:
            return self

        config = self.attention_dp_config
        if config.enable_balance:
            if config.batching_wait_iters < 0:
                raise ValueError(
                    "attention_dp_config.batching_wait_iters must be greater or equal to 0 when enable_balance is true"
                )
            if config.timeout_iters < 0:
                raise ValueError(
                    "attention_dp_config.timeout_iters must be greater or equal to 0 when enable_balance is true"
                )
        return self

    @model_validator(mode='after')
    def validate_batch_wait_timeout_ms(self) -> 'TorchLlmArgs':
        """Validate batch wait timeout."""
        if self.batch_wait_timeout_ms < 0:
            raise ValueError("batch_wait_timeout_ms must be greater than 0")
        return self

    @model_validator(mode='after')
    def validate_batch_wait_timeout_iters(self) -> 'TorchLlmArgs':
        if self.batch_wait_timeout_iters < 0:
            raise ValueError(
                f"batch_wait_timeout_iters must be >= 0, got {self.batch_wait_timeout_iters}"
            )
        return self

    @model_validator(mode='after')
    def validate_batch_wait_max_tokens_ratio(self) -> 'TorchLlmArgs':
        if self.batch_wait_max_tokens_ratio < 0 or self.batch_wait_max_tokens_ratio > 1:
            raise ValueError(
                f"batch_wait_max_tokens_ratio must be in range [0, 1], got {self.batch_wait_max_tokens_ratio}"
            )
        return self

    @model_validator(mode='after')
    def validate_ray_worker_extension_cls(self) -> 'TorchLlmArgs':
        if self.ray_worker_extension_cls is not None and self.orchestrator_type != "ray":
            raise ValueError(
                f"ray_worker_extension_cls is only supported with orchestrator_type='ray'"
            )
        return self

    def get_executor_config(
        self,
        _hf_model_dir: Optional[Path] = None,
        tokenizer: Optional[TokenizerBase] = None,
    ) -> _ExecutorConfig:
        executor_config = super().get_executor_config(_hf_model_dir, tokenizer)
        executor_config.mm_encoder_only = self.mm_encoder_only
        return executor_config

    # TODO: Remove this after the PyTorch backend is fully migrated to TorchLlmArgs from ExecutorConfig
    def get_pytorch_backend_config(self) -> "PyTorchConfig":
        from tensorrt_llm._torch.pyexecutor.config import PyTorchConfig

        return PyTorchConfig(
            extra_resource_managers=self.extra_resource_managers,
            use_cuda_graph=bool(self.cuda_graph_config is not None),
            cuda_graph_batch_sizes=self.cuda_graph_config.batch_sizes
            if self.cuda_graph_config else
            CudaGraphConfig.model_fields['batch_sizes'].default,
            cuda_graph_max_batch_size=self.cuda_graph_config.max_batch_size
            if self.cuda_graph_config else
            CudaGraphConfig.model_fields['max_batch_size'].default,
            cuda_graph_padding_enabled=self.cuda_graph_config.enable_padding
            if self.cuda_graph_config else
            CudaGraphConfig.model_fields['enable_padding'].default,
            disable_overlap_scheduler=self.disable_overlap_scheduler,
            moe_max_num_tokens=self.moe_config.max_num_tokens,
            moe_load_balancer=self.moe_config.load_balancer,
            attn_backend=self.attn_backend,
            moe_backend=self.moe_config.backend,
            use_low_precision_moe_combine=self.moe_config.
            use_low_precision_moe_combine,
            sampler_type=self.sampler_type,
            kv_cache_dtype=self.kv_cache_config.dtype,
            mamba_ssm_cache_dtype=self.kv_cache_config.mamba_ssm_cache_dtype,
            enable_iter_perf_stats=self.enable_iter_perf_stats,
            enable_iter_req_stats=self.enable_iter_req_stats,
            print_iter_log=self.print_iter_log,
            torch_compile_enabled=bool(self.torch_compile_config is not None),
            torch_compile_fullgraph=self.torch_compile_config.enable_fullgraph
            if self.torch_compile_config is not None else
            TorchCompileConfig.model_fields['enable_fullgraph'].default,
            torch_compile_inductor_enabled=self.torch_compile_config.
            enable_inductor if self.torch_compile_config is not None else
            TorchCompileConfig.model_fields['enable_inductor'].default,
            torch_compile_piecewise_cuda_graph=self.torch_compile_config.
            enable_piecewise_cuda_graph
            if self.torch_compile_config is not None else TorchCompileConfig.
            model_fields['enable_piecewise_cuda_graph'].default,
            torch_compile_piecewise_cuda_graph_num_tokens=self.
            torch_compile_config.capture_num_tokens
            if self.torch_compile_config is not None else
            TorchCompileConfig.model_fields['capture_num_tokens'].default,
            torch_compile_enable_userbuffers=self.torch_compile_config.
            enable_userbuffers if self.torch_compile_config is not None else
            TorchCompileConfig.model_fields['enable_userbuffers'].default,
            torch_compile_max_num_streams=self.torch_compile_config.
            max_num_streams if self.torch_compile_config is not None else
            TorchCompileConfig.model_fields['max_num_streams'].default,
            enable_autotuner=self.enable_autotuner,
            enable_layerwise_nvtx_marker=self.enable_layerwise_nvtx_marker,
            load_format=self.load_format,
            enable_min_latency=self.enable_min_latency,
            moe_disable_finalize_fusion=self.moe_config.disable_finalize_fusion,
            stream_interval=self.stream_interval,
            force_dynamic_quantization=self.force_dynamic_quantization,
            allreduce_strategy=self.allreduce_strategy,
            attention_dp_enable_balance=bool(
                self.attention_dp_config is not None
                and self.attention_dp_config.enable_balance),
            attention_dp_time_out_iters=self.attention_dp_config.timeout_iters
            if self.attention_dp_config is not None else
            AttentionDpConfig.model_fields['timeout_iters'].default,
            attention_dp_batching_wait_iters=self.attention_dp_config.
            batching_wait_iters if self.attention_dp_config is not None else
            AttentionDpConfig.model_fields['batching_wait_iters'].default,
            batch_wait_timeout_ms=self.batch_wait_timeout_ms,
            batch_wait_timeout_iters=self.batch_wait_timeout_iters,
            batch_wait_max_tokens_ratio=self.batch_wait_max_tokens_ratio,
        )


def update_llm_args_with_extra_dict(
        llm_args: Dict,
        llm_args_dict: Dict,
        extra_llm_api_options: Optional[str] = None) -> Dict:

    field_mapping = {
        "quant_config": QuantConfig,
        "calib_config": CalibConfig,
        "build_config": BuildConfig,
        "decoding_config": DecodingConfig,
        "enable_build_cache": BuildCacheConfig,
        "speculative_config": DecodingBaseConfig,
        "lora_config": LoraConfig,
        "moe_config": MoeConfig,
        "attention_dp_config": AttentionDpConfig,
        "sparse_attention_config": BaseSparseAttentionConfig,
    }
    for field_name, field_type in field_mapping.items():
        if field_name in llm_args_dict:
            # Some fields need to be converted manually.
            if field_name in [
                    "speculative_config", "build_config",
                    "sparse_attention_config"
            ]:
                llm_args_dict[field_name] = field_type.from_dict(
                    llm_args_dict[field_name])
            else:
                llm_args_dict[field_name] = field_type(
                    **llm_args_dict[field_name])
            extra_llm_str = f"because it's specified in {extra_llm_api_options}" if extra_llm_api_options else ""
            logger.warning(f"Overriding {field_name} {extra_llm_str}")

    llm_args = llm_args | llm_args_dict

    # For trtllm-bench or trtllm-serve, build_config may be passed for the PyTorch
    # backend, overwriting the knobs there since build_config always has the highest priority
    if "build_config" in llm_args:
        for key in [
                "max_batch_size",
                "max_num_tokens",
                "max_beam_width",
                "max_seq_len",
        ]:
            if key in llm_args_dict:
                logger.info(
                    f"Overriding {key} from build_config to {llm_args_dict[key]}"
                )
                setattr(llm_args["build_config"], key, llm_args_dict[key])

    return llm_args


def update_llm_args_with_extra_options(llm_args: Dict,
                                       extra_llm_api_options: str) -> Dict:
    if extra_llm_api_options is not None:
        with open(extra_llm_api_options, 'r') as f:
            llm_args_dict = yaml.safe_load(f)
            llm_args = update_llm_args_with_extra_dict(llm_args, llm_args_dict,
                                                       extra_llm_api_options)
    return llm_args


def get_model_format(model_dir: str,
                     trust_remote_code: bool = False) -> _ModelFormatKind:
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
            AutoConfig.from_hugging_face(model_dir,
                                         trust_remote_code=trust_remote_code)
    except Exception as e:
        raise ValueError(
            f"Inferred model format {model_format}, but failed to load config.json: {e}"
        )
    else:
        return model_format


LlmArgs = TorchLlmArgs

TRT_LLMARGS_EXPLICIT_DOCSTRING = generate_api_docs_as_docstring(TrtLlmArgs,
                                                                indent=' ' * 4)
TORCH_LLMARGS_EXPLICIT_DOCSTRING = generate_api_docs_as_docstring(TorchLlmArgs,
                                                                  indent=' ' *
                                                                  4)
