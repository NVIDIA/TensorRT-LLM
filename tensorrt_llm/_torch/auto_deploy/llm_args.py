from importlib.resources import files
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Type, Union

import torch
from pydantic import Field, ValidationInfo, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from ...llmapi.llm_args import (
    BuildConfig,
    EagleDecodingConfig,
    SamplerType,
    TorchLlmArgs,
    _ParallelConfig,
)
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
    "compile_backend": ("compile_model.backend",),
    "cuda_graph_batch_sizes": ("compile_model.cuda_graph_batch_sizes",),
}


def _shortcut_description(description: str, shortcut: str) -> str:
    long_names_str = ", ".join([f"transforms.{k}" for k in _TRANSFORMS_SHORTCUT_LOOKUP[shortcut]])
    return f"{description} Alias for: {long_names_str}."


class LlmArgs(DynamicYamlMixInForSettings, TorchLlmArgs, BaseSettings):
    """LlmArgs config class for providing full expert configurability of the AutoDeploy backend."""

    model_config = _get_config_dict()

    build_config: Optional[BuildConfig] = Field(
        default_factory=BuildConfig,
        description="!!! DO NOT USE !!! Internal only; needed for BaseLlmArgs compatibility.",
        exclude_from_json=True,
        frozen=True,
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

    @field_validator("max_seq_len", mode="before")
    @classmethod
    def ensure_max_seq_len(cls, value: Any, info: ValidationInfo) -> Any:
        # NOTE: the bass class's default value is `None`, which is incompatible with the validators
        # defined in this child class. This is problematic when e.g. TRTLLM serve explicitly passes
        # the bass class's default in.
        if value is None:
            # Fallback to the AutoDeployConfig default when not provided.
            return cls.model_fields["max_seq_len"].get_default(call_default_factory=True)
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
    def setup_hidden_state_capture(self):
        if self.speculative_config is None or not isinstance(
            self.speculative_config, EagleDecodingConfig
        ):
            return self

        self.transforms["detect_hidden_states_for_capture"]["capture_hidden_states"] = True
        self.transforms["detect_hidden_states_for_capture"]["eagle3_layers_to_capture"] = (
            self.speculative_config.eagle3_layers_to_capture
        )
        return self

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

    ## !! Remnants (fields and validators) from the now removed `AutoDeployConfig`.

    ### MODEL AND TOKENIZER FACTORY ################################################################
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

    tokenizer_kwargs: Dict[str, Any] = Field(
        default_factory=dict,
        description="Extra kwargs for the tokenizer class to customize the tokenizer. Same as "
        "model_kwargs. For example, the default HF Llama tokenizer can be initialized with the "
        "arguments specified here: "
        "https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/tokenization_llama_fast.py#L127.",
    )

    ### RUNTIME FEATURES ###########################################################################
    world_size: int = Field(
        default=1,
        ge=0,
        description="Choose from number of GPUs for Auto Sharding. A world size of 0 indicates that"
        " no processes are spawned and the model is run on a single GPU (only for ``demollm``).",
    )

    runtime: Literal["demollm", "trtllm"] = Field(default="trtllm")

    device: str = Field(default="cuda", description="The device to use for the model.", frozen=True)

    sampler_type: Union[str, SamplerType] = Field(
        default=SamplerType.TorchSampler,
        description="The type of sampler to use. Options are TRTLLMSampler or TorchSampler. Defaults to TorchSampler.",
    )

    max_beam_width: int = Field(
        default=1,
        description="The maximum beam width. >1 is not supported by AutoDeploy.",
        frozen=True,
    )

    draft_checkpoint_loader: Optional[object] = Field(
        default=None,
        description="The checkpoint loader to use for the draft model when using speculative decoding with two models.",
    )

    ### INFERENCE OPTIMIZER CONFIG #################################################################
    mode: Literal["graph", "transformers", "export_edgellm_onnx"] = Field(
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
    compile_backend: str = Field(
        default="torch-compile",
        description=_shortcut_description(
            "The backend to use for compiling the model.", "compile_backend"
        ),
    )
    # TODO(#9306): fold this into `CudaGraphConfig`.
    cuda_graph_batch_sizes: Optional[List[int]] = Field(
        default=None,
        description=_shortcut_description(
            "List of batch sizes for CUDA graph creation. If not provided, a heuristic will"
            " be used to determine the batch sizes.",
            "cuda_graph_batch_sizes",
        ),
    )

    ### SEQUENCE INTERFACE CONFIG ##################################################################
    max_seq_len: int = Field(default=512, ge=1, description="The maximum sequence length.")
    max_batch_size: int = Field(default=8, ge=1, description="The maximum batch size.")

    def model_dump(self, *args, **kwargs):
        """Convert the arguments to a dictionary that can be used as kwargs for the LLM API."""
        kwargs = super().model_dump(*args, **kwargs)

        # ensure we remove the mode and yaml_default fields since they otherwise may conflict each
        # other.
        if "mode" not in self.model_fields_set:
            kwargs.pop("mode", None)
        if "yaml_default" not in self.model_fields_set:
            kwargs.pop("yaml_default", None)

        # We never want these.
        kwargs.pop("build_config", None)
        kwargs.pop("mpi_session", None)

        return kwargs

    ### VALIDATION #################################################################################
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

    @model_validator(mode="after")
    def update_cuda_graph_batch_sizes(self):
        # if not set, use heuristic
        if self.cuda_graph_batch_sizes is None:
            cg_bs = {1, self.max_batch_size}
            # Only add batch sizes up to max_batch_size
            cg_bs.update(range(1, min(128, self.max_batch_size) + 1, 16))
            cg_bs.update(range(128, self.max_batch_size + 1, 128))
        else:
            cg_bs = [b for b in self.cuda_graph_batch_sizes if b <= self.max_batch_size]
        self.cuda_graph_batch_sizes = sorted(cg_bs, reverse=True)
        ad_logger.info(f"Using cuda_graph_batch_sizes: {self.cuda_graph_batch_sizes}")

        # ensure that the cuda_graph_batch_sizes are updated in the shortcut and transform config
        self.update_transforms_with_shortcuts()
        return self

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

    def is_cuda_graph_enabled(self) -> bool:
        return self.compile_backend in ["torch-cudagraph", "torch-opt"]

    ### PRIVATE METHODS ############################################################################
    @classmethod
    def _get_yaml_default_from_mode(cls, mode: Optional[str]) -> Optional[str]:
        config_path = files("tensorrt_llm._torch.auto_deploy.config")
        mapping = {
            "graph": str(config_path / "default.yaml"),
            "transformers": str(config_path / "transformers.yaml"),
            "export_edgellm_onnx": str(config_path / "export_edgellm_onnx.yaml"),
        }
        return mapping.get(mode)
