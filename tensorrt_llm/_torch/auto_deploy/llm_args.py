from importlib.resources import files
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Type, Union

import torch
from pydantic import Field, ValidationInfo, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from tensorrt_llm.mapping import Mapping

from ...llmapi.llm_args import (
    BuildConfig,
    EagleDecodingConfig,
    MTPDecodingConfig,
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

    @field_validator("max_beam_width", mode="after")
    @classmethod
    def ensure_no_beam_search(cls, value: Any) -> Any:
        if value is not None and value > 1:
            raise ValueError("AutoDeploy does not support beam search (max_beam_width > 1).")
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
        spec_config = self.speculative_config
        if spec_config is None:
            return self

        if isinstance(spec_config, MTPDecodingConfig):
            if not spec_config.mtp_eagle_one_model:
                return self
            if spec_config.use_mtp_vanilla:
                raise ValueError("mtp_eagle_one_model and use_mtp_vanilla cannot both be enabled")
            if spec_config.max_draft_len is None:
                raise ValueError(
                    "MTPDecodingConfig.max_draft_len must not be None when mtp_eagle_one_model is "
                    "enabled. Ensure num_nextn_predict_layers is set in the model config."
                )
            capture_layers = {-1}
            self.model_factory = "eagle_one_model"
        elif isinstance(spec_config, EagleDecodingConfig):
            if spec_config.max_draft_len is None:
                raise ValueError(
                    "EagleDecodingConfig.max_draft_len must not be None. "
                    "Provide a positive integer for max_draft_len."
                )
            capture_layers = spec_config.eagle3_layers_to_capture
            if spec_config.eagle3_one_model:
                self.model_factory = "eagle_one_model"
        else:
            return self

        self.transforms["detect_hidden_states_for_capture"]["enabled"] = True
        self.transforms["detect_hidden_states_for_capture"]["eagle3_layers_to_capture"] = (
            capture_layers
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

    speculative_model_kwargs: Dict[str, Any] = Field(
        default_factory=dict,
        description="Extra kwargs for the speculative (draft) model config class. Same semantics "
        "as model_kwargs but applied to the draft model when using one-model Eagle speculative "
        "decoding.",
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

    runtime: Literal["demollm", "trtllm"] = Field(
        default="trtllm",
        description="The runtime backend to use. 'trtllm' is a production-grade runtime optimized for "
        "high-performance inference. 'demollm' is a lightweight runtime for development and testing "
        "with a simplified scheduler and KV-cache manager for easier debugging.",
    )

    device: str = Field(default="cuda", description="The device to use for the model.", frozen=True)

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
    compile_backend: str = Field(
        default="torch-compile",
        description=_shortcut_description(
            "The backend to use for compiling the model.", "compile_backend"
        ),
    )

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
    def max_batch_size_must_be_at_least_that_of_cuda_graph_config(self):
        """If provided, the top-level `max_batch_size` must be at least that of cuda graph config."""
        if (
            self.cuda_graph_config is not None
            and (cg_mbs := self.cuda_graph_config.max_batch_size)
            and "compile_model" in self.transforms
        ):
            if self.max_batch_size < cg_mbs:
                raise ValueError(
                    f"The top-level `max_batch_size` ({self.max_batch_size}) must be greater than "
                    f"or equal to `cuda_graph_config.max_batch_size` ({cg_mbs})."
                )
        return self

    @model_validator(mode="after")
    def cap_max_batch_size_to_max_num_tokens(self):
        """Ensure max_batch_size does not exceed max_num_tokens.

        Since each sequence uses at least one token slot, max_batch_size cannot
        exceed max_num_tokens. When only max_num_tokens is explicitly set, we
        silently cap max_batch_size and warn. When both are explicitly set and
        incompatible, we raise an error.
        """
        if self.max_num_tokens is not None and self.max_batch_size > self.max_num_tokens:
            both_explicit = (
                "max_batch_size" in self.model_fields_set
                and "max_num_tokens" in self.model_fields_set
            )
            if both_explicit:
                raise ValueError(
                    f"max_batch_size ({self.max_batch_size}) cannot exceed "
                    f"max_num_tokens ({self.max_num_tokens}). Each sequence "
                    f"consumes at least one token slot."
                )
            ad_logger.warning(
                f"max_batch_size ({self.max_batch_size}) exceeds max_num_tokens "
                f"({self.max_num_tokens}). Capping max_batch_size to "
                f"{self.max_num_tokens}."
            )
            self.max_batch_size = self.max_num_tokens
        return self

    ### UTILITY METHODS ############################################################################
    def create_factory(self) -> ModelFactory:
        """Create a model factory from the arguments.

        Side effects:
            This method resolves `max_seq_len` when it has not been explicitly set by the user.
            The value is inferred from the model configuration via the factory and written back to
            `self.max_seq_len` so that all downstream consumers see the same value.
        """

        # TODO (lucaslie): consider supporting Path objects in the model factory
        factory = ModelFactoryRegistry.get(self.model_factory)(
            model=str(self.model),
            model_kwargs=self.model_kwargs,
            tokenizer=None if self.tokenizer is None else str(self.tokenizer),
            tokenizer_kwargs=self.tokenizer_kwargs,
            skip_loading_weights=self.skip_loading_weights,
            max_seq_len=self.max_seq_len,
            # Extra kwargs consumed by EagleOneModelFactory (ignored by others via **kwargs)
            speculative_config=self.speculative_config,
            speculative_model_kwargs=self.speculative_model_kwargs or None,
        )

        # The factory handles the logic internally for getting the `max_seq_len` if not provided
        # by the user.
        self.max_seq_len = factory.max_seq_len

        return factory

    def is_cuda_graph_enabled(self) -> bool:
        return self.compile_backend in ["torch-cudagraph", "torch-opt"]

    def init_mapping_from_config(self, rank: int, world_size: int) -> Mapping:
        sharding_config = self.transforms.get("detect_sharding", {})
        dist_mapping_config = sharding_config.get("dist_mapping", {})
        enable_attention_dp = sharding_config.get("enable_attention_dp", False)

        # Determine MoE parallelism dimensions
        if enable_attention_dp:
            # EP + TP 2D parallelism is currently NOT supported with attention-DP.
            # EP-only: experts sharded across GPUs, use all-to-all dispatch/combine
            moe_ep_size = self.world_size
            moe_tp_size = 1
            ad_logger.info(
                f"Attention-DP with EP-only MoE: moe_ep_size={moe_ep_size}, moe_tp_size={moe_tp_size}"
            )
        else:
            # No attention-DP: use dist_mapping config or defaults
            moe_tp_size = dist_mapping_config.get("moe_tp", 1)
            moe_ep_size = dist_mapping_config.get("moe_ep", self.world_size)

        # Create Mapping with proper distributed configuration
        try:
            mapping = Mapping(
                world_size=world_size,
                rank=rank,
                tp_size=dist_mapping_config.get("tp", self.world_size),
                moe_tp_size=moe_tp_size,
                moe_ep_size=moe_ep_size,
                moe_cluster_size=dist_mapping_config.get("moe_cluster", 1),
                enable_attention_dp=enable_attention_dp,
            )
        except ValueError as e:
            raise ValueError(
                f"Invalid parallel grid config: {e}. "
                f"Please check your dist_mapping configuration: {dist_mapping_config}"
            ) from e

        return mapping

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
