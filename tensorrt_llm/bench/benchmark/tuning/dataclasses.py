from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Literal, Optional

import yaml
from huggingface_hub import snapshot_download
from pydantic import (AliasChoices, AliasPath, BaseModel, Field, computed_field,
                      field_validator, model_validator)
from transformers import AutoConfig

from tensorrt_llm.bench.benchmark.tuning.utils import get_model_config
from tensorrt_llm.bench.build.utils import get_safetensors_metadata


class BenchmarkSpecification(BaseModel):
    environment: BenchmarkEnvironment
    engine_dir: Optional[Path] = Field(
        default=None,
        description="The path to the engine to use for benchmarking.")
    scenario: Optional[ScenarioSpecification] = Field(
        default=None, description="The scenario to use for benchmarking.")
    world: Optional[WorldConfig] = Field(
        default=None, description="The world to use for benchmarking.")
    constraints: Optional[TuningConstraints] = Field(
        default=None,
        description="The tuning criteria to use for benchmarking.")

    def __post_init__(self) -> None:
        if self.checkpoint_path is None and self.scenario.backend != "tensorrt":
            self.checkpoint_path = snapshot_download(self.model)


class BenchmarkEnvironment(BaseModel):
    model: str = Field(default="",
                       description="The HF model name to use for benchmarking.")
    checkpoint_path: Optional[Path] = Field(
        default=None,
        description="The path to the checkpoint to use for benchmarking.")
    workspace: Path = Field(
        default="/tmp", description="The workspace to use for engine building.")

    @computed_field(
        description=
        "The type of model being used, derived from the model configuration.")
    def model_type(self) -> str:
        return get_model_config(self.model, self.checkpoint_path).model_type


class ScenarioSpecification(BaseModel):
    extra_llm_api_options: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="A dictionary of extra LLM API options to use for tuning.")
    concurrency: Optional[int] = Field(
        default=None, description="The number of concurrent requests to serve.")
    kv_cache_free_gpu_mem_fraction: Optional[float] = Field(
        default=.9,
        description=
        "The percentage of memory to use for KV Cache after model load.")
    backend: Optional[Literal["pytorch", "autodeploy", "trt"]] = Field(
        default="pytorch", description="The backend to use for benchmarking.")

    class Config:
        extra = "ignore"

    @field_validator("extra_llm_api_options", mode="before")
    def validate_extra_llm_api_options(self, v) -> Dict[str, Any]:
        if v is None:
            return None
        else:
            if isinstance(v, Path) or isinstance(v, str):
                p = Path(v)
                if not p.exists():
                    raise ValueError(
                        f"extra_llm_api_options file {p} does not exist.")
                with open(p, "r") as f:
                    v = yaml.safe_load(f)
            elif isinstance(v, dict):
                for key, _ in v.items():
                    if not isinstance(key, str):
                        raise ValueError(
                            f"extra_llm_api_options keys must be strings, got {type(key)}"
                        )
                return v
            else:
                raise ValueError(
                    f"extra_llm_api_options must be a dictionary or a path to a YAML file, got {type(v)}"
                )


class TuningConstraints(BaseModel):
    mode: Literal["build", "benchmark"] = Field(
        default="benchmark",
        description="The path tuning is being accessed from.")
    modality: Optional[Literal["text", "image", "video"]] = Field(
        default="text", description="The modality of the model being used.")
    dataset_path: Optional[str] = Field(
        default=None, description="The path to the dataset to use for tuning.")
    target_input_len: Optional[int] = Field(
        default=None, description="The target input length to use for tuning.")
    target_output_len: Optional[int] = Field(
        default=None, description="The target output length to use for tuning.")
    max_input_len: Optional[int] = Field(
        default=None, description="The maximum input length to use for tuning.")
    max_output_len: Optional[int] = Field(
        default=None,
        description="The maximum output length to use for tuning.")
    max_batch_size: Optional[int] = Field(
        default=None, description="The maximum batch size to use for tuning.")
    max_num_tokens: Optional[int] = Field(
        default=None,
        description="The maximum number of tokens to use for tuning.")
    max_seq_len: Optional[int] = Field(
        default=None,
        description="The maximum sequence length to use for tuning.")
    beam_width: Optional[int] = Field(
        default=None, description="The beam width to use for tuning.")

    class Config:
        extra = "ignore"

    @model_validator(mode="after")
    def validate_tuning_setup(self):
        build_options = [
            self.dataset_path, self.max_input_len, self.target_input_len
        ]
        if all(opt is None for opt in build_options):
            raise ValueError(
                "No engine build option is selected, please provide at least one option."
            )
        elif sum([bool(opt) for opt in build_options]) > 1:
            raise ValueError(
                "Multiple engine build options detected, please choose only one engine build option."
            )

        if not self.dataset_path and not self.max_input_len:
            raise ValueError("Unspecified max_input_len for engine build.")
        return self

    @model_validator(mode="after")
    def validate_max_input_len(self):
        if self.mode == "build":
            if self.max_input_len is None:
                raise ValueError(
                    "max_input_len is required when mode is 'build'")
        return self


class WorldConfig(BaseModel):
    tp: Optional[int] = Field(
        default=None,
        description="The tensor parallelism size to use for tuning.")
    pp: Optional[int] = Field(
        default=None,
        description="The pipeline parallelism size to use for tuning.")
    ep: Optional[int] = Field(
        default=None,
        description="The expert parallelism size to use for tuning.")
    cluster_size: Optional[int] = Field(
        default=None, description="The expert cluster size to use for tuning.")

    @computed_field
    def world_size(self) -> int:
        return self.tp * self.pp


class ModelConfig(BaseModel):
    """ Model specific configurations. The parameters are needed in engine
        setting calculation.
    """
    name: str
    model_type: str
    param_count: int
    num_hidden_layers: int = Field(validation_alias=AliasChoices(
        "num_hidden_layers",
        "n_layer",
        AliasPath("text_config", "num_hidden_layers"),
    ))
    num_attention_heads: int = Field(validation_alias=AliasChoices(
        "num_attention_heads",
        "n_head",
        AliasPath("text_config", "num_attention_heads"),
    ))
    num_key_value_heads: Optional[int] = Field(
        default=None,
        validation_alias=AliasChoices(
            "num_key_value_heads",
            "num_kv_heads",
            AliasPath("text_config", "num_key_value_heads"),
        ),
    )
    hidden_size: int = Field(validation_alias=AliasChoices(
        "hidden_size",
        "n_embd",
        AliasPath("text_config", "hidden_size"),
    ))
    head_size: Optional[int] = Field(default=None,
                                     validation_alias=AliasChoices(
                                         "head_size",
                                         "head_dim",
                                         AliasPath("text_config", "head_dim"),
                                     ))
    max_position_embeddings: Optional[int] = Field(
        default=None,
        validation_alias=AliasChoices(
            "max_position_embeddings",
            "n_positions",
            AliasPath("text_config", "max_position_embeddings"),
        ))
    dtype: Literal["float16", "bfloat16",
                   None] = Field(default="float16",
                                 validation_alias=AliasChoices(
                                     "dtype", "torch_dtype"))

    @model_validator(mode="after")
    def set_values_if_none(self):
        """ Set the values if cannot get values from HF config.json. """
        if not self.dtype:  # for GPT-J
            self.dtype = "float16"
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads
        if self.head_size is None:
            self.head_size = self.hidden_size // self.num_attention_heads
        return self

    @classmethod
    def get_param_count(cls, model_hf_name, hf_model_path):
        """ Read the parameter count from HF safetensor metadata. """
        if model_hf_name == "EleutherAI/gpt-j-6b":  # GPT-J repo doesn't use safetensor format.
            param_count = 6053381344
        else:
            model_name_or_path = hf_model_path or model_hf_name
            metadata = get_safetensors_metadata(model_name_or_path)
            param_count = sum(metadata.parameter_count.values())
        assert param_count, f"Can't get valid parameter count for model: {model_name_or_path}."

        return param_count

    @classmethod
    def from_hf(cls, model_hf_name, hf_model_path):
        model_name_or_path = hf_model_path or model_hf_name
        hf_config = AutoConfig.from_pretrained(
            model_name_or_path, trust_remote_code=True).to_dict()

        param_count = cls.get_param_count(model_hf_name, hf_model_path)

        return cls(name=model_hf_name, param_count=param_count, **hf_config)
