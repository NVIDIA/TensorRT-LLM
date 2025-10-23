from importlib.resources import files
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Type, Union

import torch
from pydantic import Field, PrivateAttr, ValidationInfo, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from tensorrt_llm.models.modeling_utils import QuantConfig

from ...llmapi.llm_args import BaseLlmArgs, BuildConfig, KvCacheConfig, _ParallelConfig
from ...llmapi.utils import get_type_repr
from .models import ModelFactory, ModelFactoryRegistry
from .utils._config import DynamicYamlMixInForSettings
from .utils.logger import ad_logger

PathLike = Union[str, Path]


def _get_config_dict() -> SettingsConfigDict:
    return SettingsConfigDict(
        arbitrary_types_allowed=True,
        extra="forbid",
        nested_model_default_partial_update=True,
    )


def _check_for_default_value_only(
    cls: Type[BaseSettings], value: Any, info: ValidationInfo, msg: str
) -> Any:
    """Check if the value is the default value for the field.

    If the value is not the default value, raise a ValueError.
    """
    field_name = info.field_name
    assert field_name is not None, "field_name should be set for validated field."
    if value != cls.model_fields[field_name].get_default(call_default_factory=True):
        raise ValueError(msg)
    return value


_TRANSFORMS_SHORTCUT_LOOKUP = {
    "attn_backend": ("insert_cached_attention.backend", "transformers_replace_cached_attn.backend"),
    "free_mem_ratio": ("resize_kv_cache.free_mem_ratio",),
    "compile_backend": ("compile_model.backend",),
    "cuda_graph_batch_sizes": ("compile_model.cuda_graph_batch_sizes",),
}


def _shortcut_description(description: str, shortcut: str) -> str:
    long_names_str = ", ".join([f"transforms.{k}" for k in _TRANSFORMS_SHORTCUT_LOOKUP[shortcut]])
    return f"{description} Alias for: {long_names_str}."


class AutoDeployConfig(DynamicYamlMixInForSettings, BaseSettings):
    """An argument class stripped down to AutoDeploy-specific configurations.

    This class be used as a drop-in replacement to simplify configuring the AutoDeploy backend and
    should be used in place of LlmArgs unless more advanced features are needed.

    It is compatible with AutoDeploy's LLM API (``tensorrt_llm._torch.auto_deploy.llm.LLM``) and
    exposes the full set of parameters used in AutoDeploy's ``InferenceOptimizer``.
    """

    model_config = _get_config_dict()

    ### MODEL AND TOKENIZER FACTORY ################################################################
    model: PathLike = Field(
        description="The path to the model checkpoint or the model name from the Hugging Face Hub."
    )

    model_factory: str = Field(
        default="AutoModelForCausalLM",
        description="The model factory to use for loading the model.",
    )

    model_kwargs: Dict[str, Any] = Field(
        default_factory=dict,
        description="Extra kwargs for the model config class to customize the model config. "
        "These arguments take precedence over default values or config values in the model config "
        "file. Arguments are resolved in order: 1) Default values in model config class, 2) Values "
        "in model config file, 3) Values in model_kwargs. Note: if a kwarg doesn't exist in the "
        "model config class, it will be ignored.",
    )

    skip_loading_weights: bool = Field(
        default=False,
        description="Whether to skip loading model weights during initialization. "
        "If True, only the model architecture is loaded.",
    )

    tokenizer: Optional[PathLike] = Field(
        description="The tokenizer",
        default=None,
        repr=False,
    )

    tokenizer_kwargs: Dict[str, Any] = Field(
        default_factory=dict,
        description="Extra kwargs for the tokenizer class to customize the tokenizer. Same as "
        "model_kwargs. For example, the default HF Llama tokenizer can be initialized with the "
        "arguments specified here: "
        "https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/tokenization_llama_fast.py#L127.",
    )

    skip_tokenizer_init: bool = Field(
        default=False, description="Whether to skip the tokenizer initialization."
    )

    ### RUNTIME FEATURES ###########################################################################
    disable_overlap_scheduler: bool = Field(
        default=False,
        description="Disable the overlap scheduler in trtllm runtime",
    )

    world_size: int = Field(
        default=1,
        ge=0,
        description="Choose from number of GPUs for Auto Sharding. A world size of 0 indicates that"
        " no processes are spawned and the model is run on a single GPU (only for ``demollm``).",
    )

    runtime: Literal["demollm", "trtllm"] = Field(default="trtllm")

    device: str = Field(default="cuda", description="The device to use for the model.", frozen=True)

    # TODO: see if we can just remove this field and use kv_cache_config.dtype instead?
    kv_cache_dtype: str = Field(
        default="auto",
        description="Data type for KV cache. This is a temporary field until kv_cache_dtype is "
        "supported in AutoDeploy.",
    )

    # NOTE: we do not support copy_on_partial_reuse in AutoDeploy yet
    # see https://github.com/NVIDIA/TensorRT-LLM/issues/7142
    kv_cache_config: KvCacheConfig = Field(
        default_factory=lambda **kwargs: KvCacheConfig(copy_on_partial_reuse=False, **kwargs),
        description="KV cache config.",
    )

    max_beam_width: int = Field(
        default=1,
        description="The maximum beam width. >1 is not supported by AutoDeploy.",
        frozen=True,
    )

    enable_chunked_prefill: bool = Field(default=False, description="Enable chunked prefill.")

    ### INFERENCE OPTIMIZER CONFIG #################################################################
    mode: Literal["graph", "transformers"] = Field(
        default="graph",
        description="The mode to use for the inference optimizer. Currently, we "
        "support only the 'graph' and 'transformers' modes, i.e., full-graph capture + optimization"
        "or transformers-only cached attention optimization.",
    )

    transforms: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict,
        description="A dictionary of transform configurations. The key is the transform name and "
        "the value is the transform configuration.",
    )

    ### SHORTCUTS FOR COMMON INFERENCE OPTIMIZER CONFIGS ###########################################
    attn_backend: str = Field(
        default="flashinfer",
        description=_shortcut_description("Attention backend to use.", "attn_backend"),
    )
    free_mem_ratio: float = Field(
        default=0.0,
        description=_shortcut_description(
            "The fraction of available memory to allocate for cache.", "free_mem_ratio"
        ),
    )
    compile_backend: str = Field(
        default="torch-compile",
        description=_shortcut_description(
            "The backend to use for compiling the model.", "compile_backend"
        ),
    )
    cuda_graph_batch_sizes: Optional[List[int]] = Field(
        default=None,
        description=_shortcut_description(
            "List of batch sizes for CUDA graph creation. If not provided, a heuristic will"
            " be used to determine the batch sizes.",
            "cuda_graph_batch_sizes",
        ),
    )

    ### SEQUENCE INTERFACE CONFIG ##################################################################
    max_input_len: int = Field(default=1024, description="The maximum input length.")
    max_num_tokens: Optional[int] = Field(default=None, description="The maximum number of tokens.")
    max_seq_len: int = Field(default=512, ge=1, description="The maximum sequence length.")
    max_batch_size: int = Field(default=8, ge=1, description="The maximum batch size.")
    attn_page_size: int = Field(
        default=64,
        ge=1,
        description="Page size for attention (tokens_per_block). For triton and torch "
        "backends, this should equal max_seq_len. Temporary field until tokens_per_block gets "
        "properly passed through.",
    )

    ### VALIDATION #################################################################################
    @model_validator(mode="after")
    # TODO: discuss what to do with this once we fully transition to the new inference optimizer
    def update_attn_page_size(self):
        # NOTE force attn_page_size to equal max_seq_len for triton backend
        if self.transforms.get("insert_cached_attention", {}).get("backend") in [
            "triton",
            "torch",
        ]:
            self.attn_page_size = self.max_seq_len
        # NOTE: (hg) For transformers mode. This is ugly.
        if self.transforms.get("transformers_replace_cached_attn", {}).get("backend") in [
            "triton",
            "torch",
        ]:
            self.attn_page_size = self.max_seq_len
        return self

    @field_validator("model_factory", mode="after")
    @classmethod
    def model_factory_exists(cls, value: str) -> str:
        if not ModelFactoryRegistry.has(value):
            raise ValueError(
                f"'{value}' does not exist in the model factory registry. Available values: "
                f"{ModelFactoryRegistry.entries()}."
            )

        return value

    @model_validator(mode="after")
    def update_transforms_with_shortcuts(self) -> Dict[str, Any]:
        """Synchronize the transforms config with the values from the defined shortcuts.

        NOTE: shortcut values always take precedence over the values in the transforms config.
        """
        for shortcut_key, transforms_keys in _TRANSFORMS_SHORTCUT_LOOKUP.items():
            for transform_key in transforms_keys:
                t_key, config_key = transform_key.split(".")
                if t_key not in self.transforms:
                    continue

                # first update the transforms config with the shortcut value
                if shortcut_key in self.model_fields_set:
                    self.transforms[t_key][config_key] = getattr(self, shortcut_key)
                # then update the shortcut field with the value from the transforms config to make
                # sure both fields are in sync
                setattr(self, shortcut_key, self.transforms[t_key][config_key])

        return self

    @field_validator("kv_cache_config", mode="after")
    @classmethod
    def validate_kv_cache_config(cls, kv_cache_config: KvCacheConfig) -> KvCacheConfig:
        if kv_cache_config.copy_on_partial_reuse:
            kv_cache_config.copy_on_partial_reuse = False
            ad_logger.warning(
                "copy_on_partial_reuse is not supported by AutoDeploy. Setting it to False."
            )
        return kv_cache_config

    ### UTILITY METHODS ############################################################################
    def create_factory(self) -> ModelFactory:
        """Create a model factory from the arguments."""

        # TODO (lucaslie): consider supporting Path objects in the model factory
        return ModelFactoryRegistry.get(self.model_factory)(
            model=str(self.model),
            model_kwargs=self.model_kwargs,
            tokenizer=None if self.tokenizer is None else str(self.tokenizer),
            tokenizer_kwargs=self.tokenizer_kwargs,
            skip_loading_weights=self.skip_loading_weights,
            max_seq_len=self.max_seq_len,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert the arguments to a dictionary."""
        return self.model_dump()

    def to_llm_kwargs(self) -> Dict[str, Any]:
        """Convert the arguments to a dictionary that can be used as kwargs for the LLM API."""
        kwargs = self.to_dict()

        # ensure we remove the mode and yaml_default fields since they otherwise may conflict each
        # other.
        if "mode" not in self.model_fields_set:
            kwargs.pop("mode")
        if "yaml_default" not in self.model_fields_set:
            kwargs.pop("yaml_default")
        return kwargs

    ### PRIVATE METHODS ############################################################################
    @classmethod
    def _get_yaml_default_from_mode(cls, mode: Optional[str]) -> Optional[str]:
        config_path = files("tensorrt_llm._torch.auto_deploy.config")
        mapping = {
            "graph": str(config_path / "default.yaml"),
            "transformers": str(config_path / "transformers.yaml"),
        }
        return mapping.get(mode)


class LlmArgs(AutoDeployConfig, BaseLlmArgs, BaseSettings):
    """LlmArgs config class for providing full expert configurability of the AutoDeploy backend.

    Specifically, this class extends AutoDeployConfig with all the fields from BaseLlmArgs for
    providing configurability beyond what is provided by AutoDeployConfig.

    Just like AutoDeployConfig, this class is compatible with AutoDeploy's LLM API
    (``tensorrt_llm._torch.auto_deploy.llm.LLM``) but provides greater configurability.

    NOTE: this class should only be used directly for advanced use cases. For most use cases,
    AutoDeployConfig should be used instead.

    NOTE: this class may expose redundant fields from BaseLlmArgs or fields that are ignored or
    have overlapping functionality with AutoDeployConfig. Please be careful when using this class.
    """

    model_config = _get_config_dict()

    build_config: Optional[object] = Field(
        default_factory=lambda: BuildConfig(),
        description="!!! DO NOT USE !!! Internal only; needed for BaseLlmArgs compatibility.",
        exclude_from_json=True,
        frozen=True,
        json_schema_extra={"type": f"Optional[{get_type_repr(BuildConfig)}]"},
        repr=False,
    )
    backend: Literal["_autodeploy"] = Field(
        default="_autodeploy",
        description="The backend to use for this LLM instance.",
        frozen=True,
    )
    gpus_per_node: int = Field(
        default=torch.cuda.device_count(),
        description="The number of GPUs per node.",
        frozen=True,
    )
    garbage_collection_gen0_threshold: int = Field(default=20000, description="See TorchLlmArgs.")

    _quant_config: Optional[QuantConfig] = PrivateAttr(default=None)

    @property
    def quant_config(self) -> QuantConfig:
        if self._quant_config is None:
            self._quant_config = QuantConfig()
        return self._quant_config

    @quant_config.setter
    def quant_config(self, value: QuantConfig):
        self._quant_config = value

    ### VALIDATION #################################################################################
    @field_validator("max_seq_len", mode="before")
    @classmethod
    def ensure_max_seq_len(cls, value: Any, info: ValidationInfo) -> Any:
        if value is None:
            # Fallback to the AutoDeployConfig default when not provided
            return AutoDeployConfig.model_fields["max_seq_len"].get_default(
                call_default_factory=True
            )
        return value

    @field_validator("build_config", mode="before")
    @classmethod
    def ensure_no_build_config(cls, value: Any, info: ValidationInfo) -> Any:
        msg = "build_config is not in use by AutoDeploy's LlmArgs"
        return _check_for_default_value_only(cls, value, info, msg)

    @field_validator(
        "tensor_parallel_size",
        "pipeline_parallel_size",
        "context_parallel_size",
        "moe_cluster_parallel_size",
        "moe_tensor_parallel_size",
        "moe_expert_parallel_size",
        "enable_attention_dp",
        "cp_config",
        mode="before",
    )
    @classmethod
    def ensure_no_custom_parallel_config(cls, value: Any, info: ValidationInfo) -> Any:
        msg = "AutoDeploy only supports parallelization via the `world_size` argument."
        return _check_for_default_value_only(cls, value, info, msg)

    @model_validator(mode="after")
    def validate_parallel_config(self):
        """Setup parallel config according to world_size.

        NOTE: AutoDeploy does *not* use parallel_config directly. It simply uses world_size and
        rank to automatically shard the model. This is just to ensure that other objects in the
        runtime that may read parallel_config can do so.
        """

        # Set tp_size = self.world_size so that _ParallelConfig.world_size will return the
        # correct value (computed as tp_size * pp_size * cp_size). This does not necessarily
        # mean that TP will actually be used.
        self._parallel_config = _ParallelConfig(
            tp_size=self.world_size, gpus_per_node=self.gpus_per_node
        )
        return self

    @model_validator(mode="after")
    def validate_and_init_tokenizer(self):
        """Skip tokenizer initialization in config. We do this in the AutoDeploy LLM class."""
        return self

    ### UTILITY METHODS ############################################################################
    # TODO: Remove this after the PyTorch backend is fully migrated to LlmArgs from ExecutorConfig
    def get_pytorch_backend_config(self) -> "LlmArgs":
        """Return the LlmArgs (self) object."""
        # TODO: can we just pass through self directly??
        return type(self)(**self.to_llm_kwargs())

    def to_dict(self) -> Dict:
        """Convert model to a dictionary such that cls(**self.to_dict()) == self."""
        self_dict = super().to_dict()
        self_dict.pop("build_config", None)
        self_dict.pop("mpi_session", None)
        return self_dict
