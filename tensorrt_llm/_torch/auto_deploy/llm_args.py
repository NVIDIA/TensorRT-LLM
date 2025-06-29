import json
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

import torch
from pydantic import Field, field_validator, model_validator

from ...llmapi.llm_args import BaseLlmArgs, BuildConfig, _ParallelConfig
from ...llmapi.utils import get_type_repr
from .models import ModelFactory, ModelFactoryRegistry


def _try_decode_dict_with_str_values(value: Dict[str, Any]) -> Dict[str, Any]:
    """Try to parse string values as JSON to convert to native types if possible."""
    for k, v in value.items():
        if isinstance(v, str):
            try:
                value[k] = json.loads(v)
            except json.JSONDecodeError:
                pass
    return value


class LlmArgs(BaseLlmArgs):
    """LLM arguments specifically for AutoDeploy backend.

    This class extends BaseLlmArgs with AutoDeploy-specific configuration options.
    AutoDeploy provides automatic deployment and optimization of language models
    with various attention backends and optimization strategies.
    """

    ### MODEL AND TOKENIZER FACTORY ################################################################
    model_factory: Literal["AutoModelForCausalLM", "AutoModelForImageTextToText"] = Field(
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

    checkpoint_device: Optional[str] = Field(
        default=None,
        description="Device on which to load the model checkpoint. "
        "Defaults to the same device as the rest of the pipeline.",
    )

    tokenizer: Optional[Union[str, Path]] = Field(
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

    ### RUNTIME FEATURES ###########################################################################
    disable_overlap_scheduler: bool = Field(
        default=True,
        description="Disable the overlap scheduler. This is a temporary field until the overlap "
        "scheduler is supported (https://github.com/NVIDIA/TensorRT-LLM/issues/4364).",
        frozen=True,
        repr=False,
    )

    mixed_sampler: bool = Field(
        default=False,
        description="If true, will iterate over sampling_params of each request and use the corresponding "
        "sampling strategy, e.g. top-k, top-p, etc.",
    )

    world_size: int = Field(
        default=1,
        ge=0,
        description="Choose from number of GPUs for Auto Sharding. A world size of 0 indicates that"
        " no processes are spawned and the model is run on a single GPU (only for ``demollm``).",
    )

    runtime: Literal["demollm", "trtllm"] = Field(default="trtllm")

    device: str = Field(default="cuda", description="The device to use for the model.", frozen=True)

    # INFERENCE OPTIMIZER CONFIG ###################################################################
    attn_backend: Literal["flashinfer", "triton"] = Field(
        default="flashinfer", description="Attention backend to use."
    )

    mla_backend: Literal["MultiHeadLatentAttention"] = Field(
        default="MultiHeadLatentAttention",
        description="The Multi-Head Latent Attention backend to use.",
    )

    free_mem_ratio: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="The fraction of available memory to allocate for cache.",
    )

    simple_shard_only: bool = Field(
        default=False,
        description="If True, force simple sharding (all_gather) in tensor parallelism. "
        "If False, auto-detect and use column+row (all_reduce) sharding when possible.",
    )

    compile_backend: Literal["torch-simple", "torch-compile", "torch-cudagraph", "torch-opt"] = (
        Field(
            default="torch-compile",
            description="The backend to use for compiling the model.",
        )
    )

    cuda_graph_batch_sizes: Optional[List[int]] = Field(
        default=None, description="List of batch sizes to create CUDA graphs for."
    )

    visualize: bool = Field(default=False, description="Whether to visualize the model graph.")

    ### SEQUENCE INTERFACE CONFIG ##################################################################
    max_seq_len: int = Field(default=512, ge=1, description="The maximum sequence length.")
    max_batch_size: int = Field(default=8, ge=1, description="The maximum batch size.")
    attn_page_size: int = Field(
        default=64,
        ge=1,
        description="Page size for attention (tokens_per_block). For triton "
        "backend, this should equal max_seq_len. Temporary field until tokens_per_block gets "
        "properly passed through.",
    )

    ### !!! DO NOT USE !!! #########################################################################
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

    ### VALIDATION #################################################################################
    @field_validator("build_config", mode="before")
    @classmethod
    def ensure_no_build_config(cls, value: Any) -> Any:
        if value is not None:
            raise ValueError("build_config is not used")
        return value

    @field_validator("model_kwargs", "tokenizer_kwargs", mode="after")
    @classmethod
    def validate_model_kwargs(cls, value: Dict[str, Any]) -> Dict[str, Any]:
        """Try to parse string values as JSON to convert to native types if possible."""
        return _try_decode_dict_with_str_values(value)

    @model_validator(mode="after")
    def validate_parallel_config(self):
        """Setup parallel config according to world_size.

        NOTE: AutoDeploy does *not* use parallel_config directly. It simply uses world_size and
        rank to automatically shard the model. This is just to ensure that other objects in the
        runtime that may read parallel_config can do so.
        """
        # setup parallel config
        self._parallel_config = _ParallelConfig(
            auto_parallel=True, gpus_per_node=self.gpus_per_node
        )
        self._parallel_config.world_size = self.world_size
        return self

    @model_validator(mode="after")
    def validate_and_init_tokenizer(self):
        """Skip tokenizer initialization in config. We do this in the AutoDeploy LLM class."""
        return self

    @model_validator(mode="after")
    def update_attn_page_size(self):
        # NOTE force attn_page_size to equal max_seq_len for triton backend
        if self.attn_backend == "triton":
            self.attn_page_size = self.max_seq_len
        return self

    ### UTILITY METHODS ############################################################################
    def create_factory(self) -> ModelFactory:
        """Create a model factory from the arguments."""

        return ModelFactoryRegistry.get(self.model_factory)(
            model=self.model,
            model_kwargs=self.model_kwargs,
            tokenizer=self.tokenizer,
            tokenizer_kwargs=self.tokenizer_kwargs,
            skip_loading_weights=self.skip_loading_weights,
            max_seq_len=self.max_seq_len,
        )

    # TODO: Remove this after the PyTorch backend is fully migrated to LlmArgs from ExecutorConfig
    def get_pytorch_backend_config(self) -> "LlmArgs":
        """Return the LlmArgs (self) object."""
        # TODO: can we just pass through self directly??
        return type(self)(**self.to_dict())

    def to_dict(self) -> Dict:
        """Convert model to a dictionary such that cls(**self.to_dict()) == self."""
        self_dict = dict(self)
        self_dict.pop("build_config")
        self_dict.pop("mpi_session")
        return self_dict
