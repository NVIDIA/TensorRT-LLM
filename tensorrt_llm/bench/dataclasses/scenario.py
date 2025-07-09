from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Self, Union

import yaml
from huggingface_hub import snapshot_download
from pydantic import (AliasChoices, AliasPath, BaseModel, Field, PrivateAttr,
                      computed_field, field_validator, model_validator)
from transformers import AutoConfig

# isort: off
from tensorrt_llm._torch.model_config import ModelConfig as TllmModelConfig
from tensorrt_llm.bench.benchmark.tuning.utils import get_model_config
from tensorrt_llm.bench.build.utils import get_safetensors_metadata
from tensorrt_llm.bench.dataclasses.general import DatasetMetadata
from tensorrt_llm.bench.dataclasses.statistics import PercentileStats
from tensorrt_llm.logger import logger
# isort: on


class BatchingConfiguration(BaseModel):
    max_seq_len: Optional[int] = Field(
        default=None,
        description="The maximum sequence length to use for batch scheduling.")
    max_batch_size: Optional[int] = Field(
        description="The maximum batch size to use for batch scheduling.",
        default=None)
    max_num_tokens: Optional[int] = Field(
        description="The maximum number of tokens to use for batch scheduling.",
        default=None)


class LlmRuntimeSpecification(BaseModel):
    backend: Optional[Literal["pytorch", "_autodeploy", "tensorrt"]] = Field(
        default="pytorch", description="The backend to use for benchmarking.")
    beam_width: Optional[int] = Field(
        default=None, description="The beam width to use for benchmarking.")
    concurrency: Optional[int] = Field(
        default=None,
        description="The number of concurrent requests to benchmarking.")
    eos_id: Optional[int] = Field(
        default=-1,
        description="The end-of-sequence token to use for benchmarking.")
    extra_llm_api_options: Dict[str, Any] = Field(
        default_factory=dict,
        description=
        "A dictionary of extra LLM API options to use for benchmarking.")
    kv_cache_free_gpu_mem_fraction: Optional[float] = Field(
        default=.9,
        description=
        "The percentage of memory to use for KV Cache after model load.")
    pad_id: Optional[int] = Field(
        default=-1, description="The padding token to use for benchmarking.")

    class Config:
        extra = "ignore"

    @field_validator("backend", mode="before")
    @classmethod
    def validate_backend(cls, v) -> Optional[str]:
        if v is not None:
            return v.lower()
        return v

    @field_validator("extra_llm_api_options", mode="before")
    @classmethod
    def validate_extra_llm_api_options(cls, v) -> Union[Dict[str, Any], None]:
        if v is None:
            return dict()
        else:
            if isinstance(v, Path) or isinstance(v, str):
                p = Path(v)
                if not p.exists():
                    raise ValueError(
                        f"extra_llm_api_options file {p} does not exist.")
                with open(p, "r") as f:
                    v = yaml.safe_load(f)

                return v
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
    target_input_len: Optional[int] = Field(
        description="The target input length to use for tuning.")
    target_output_len: Optional[int] = Field(
        description="The target output length to use for tuning.")
    max_input_len: int = Field(
        description="The maximum input length to use for tuning.")
    max_output_len: int = Field(
        description="The maximum output length to use for tuning.")

    class Config:
        extra = "ignore"

    @model_validator(mode="after")
    def validate_target_input_output_len(self) -> Self:
        self.target_input_len = self.target_input_len or self.max_input_len
        self.target_output_len = self.target_output_len or self.max_output_len

        return self

    @classmethod
    def from_dataset_metadata(cls,
                              metadata: DatasetMetadata) -> "TuningConstraints":
        return cls(
            target_input_len=metadata.avg_isl,
            target_output_len=metadata.avg_osl,
            max_input_len=metadata.max_isl,
            max_output_len=metadata.max_osl,
        )


class WorldConfig(BaseModel):
    tp: int = Field(
        default=None,
        description="The tensor parallelism size to use for tuning.")
    pp: int = Field(
        default=None,
        description="The pipeline parallelism size to use for tuning.")
    ep: Optional[int] = Field(
        default=None,
        description="The expert parallelism size to use for tuning.")
    cluster_size: Optional[int] = Field(
        default=None, description="The expert cluster size to use for tuning.")
    gpus_per_node: Optional[int] = Field(
        default=None,
        description="The number of GPUs per node to use for tuning.")

    @computed_field
    def world_size(self) -> int:
        return int(self.tp * self.pp)


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
    def set_values_if_none(self) -> Self:
        """ Set the values if cannot get values from HF config.json. """
        if not self.dtype:  # for GPT-J
            self.dtype = "float16"
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads
        if self.head_size is None:
            self.head_size = self.hidden_size // self.num_attention_heads
        return self

    @classmethod
    def get_param_count(cls, model_hf_name, hf_model_path) -> int:
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
    def from_hf(cls, model_hf_name, hf_model_path) -> Self:
        model_name_or_path = hf_model_path or model_hf_name
        hf_config = AutoConfig.from_pretrained(
            model_name_or_path, trust_remote_code=True).to_dict()

        param_count = cls.get_param_count(model_hf_name, hf_model_path)

        return cls(name=model_hf_name, param_count=param_count, **hf_config)


class ReportingConfiguration(BaseModel):
    report_json: Optional[Path] = Field(
        default=None,
        description="The path to the report to use for benchmarking.")
    iteration_log: Optional[Path] = Field(
        default=None,
        description="The path to the iteration log to use for benchmarking.")
    output_json: Optional[Path] = Field(
        default=None,
        description="The path to the output to use for benchmarking.")


class ScenarioSpecification(BaseModel):
    target_scenario: Optional[str] = Field(
        default="throughput",
        description="The target scenario to use for benchmarking.")
    constraints: Optional[TuningConstraints] = Field(
        description="The tuning criteria to use for benchmarking.",
        default=None,
    )
    dataset_path: Path = Field(
        description="The path to the dataset to use for benchmarking.",
        alias=AliasChoices("dataset", "dataset_path"),
    )
    dataset_metadata: Optional[DatasetMetadata] = Field(
        description="The metadata of the dataset to use for benchmarking.",
        default=None,
    )
    num_requests: int = Field(
        description="The number of requests to use for benchmarking.",
        default=0,
        ge=0,
    )
    engine_dir: Optional[Path] = Field(
        default=None,
        description="The path to the engine to use for benchmarking.")
    environment: BenchmarkEnvironment = Field(
        description="The environment to use for benchmarking.", )
    modality: Optional[Literal["text", "image", "video"]] = Field(
        default="text", description="The modality of the model being used.")
    mode: Literal["build", "benchmark"] = Field(
        default="benchmark",
        description="The path tuning is being accessed from.")
    llm_config: LlmRuntimeSpecification = Field(
        description="The LLM runtime options to use for benchmarking.")
    batching_config: BatchingConfiguration = Field(
        description="The batching options to use for benchmarking.",
        default_factory=BatchingConfiguration,
    )
    reporting_config: ReportingConfiguration = Field(
        description="The reporting options to use for benchmarking.",
        default_factory=ReportingConfiguration,
    )
    world: WorldConfig = Field(description="The world to use for benchmarking.")

    _engine_config: Dict[str, Any] = PrivateAttr(default=dict)

    def __post_init__(self) -> None:
        if self.checkpoint_path is None and self.llm_config.backend != "tensorrt":
            self.checkpoint_path = snapshot_download(self.environment.model)

    class Config:
        validate_assignment = True

    @model_validator(mode="after")
    def validate_engine_dir(self) -> Self:
        engine_dir = bool(self.engine_dir is not None)
        trt_backend = bool(self.llm_config.backend == "tensorrt")
        engine_dir_exists = bool(self.engine_dir is not None
                                 and self.engine_dir.exists())

        # We aren't dealing with tensorrt engine for this instance. Return.
        if not engine_dir and not trt_backend:
            return self

        # We are dealing with tensorrt engine for this instance but backend is not tensorrt.
        if engine_dir and not trt_backend:
            raise RuntimeError(
                "Specifying an engine directory ('engine_dir')is required for running the TRT workflow."
            )

        # Engine directory does not exist.
        if not engine_dir_exists:
            raise ValueError(
                f"Engine directory ('engine_dir') does not exist: {self.engine_dir}"
            )

        return self

    @model_validator(mode="after")
    def validate_engine_config(self) -> Self:
        if self.engine_dir is None:
            return self

        with open(self.engine_dir / "config.json", "r") as f:
            self._engine_config = json.load(f)

        # Validate/update settings in relation to the engine config.
        engine_mapping = self._engine_config["pretrained_config"]["mapping"]
        build_config = self._engine_config["build_config"]

        engine_to_config_map = {
            "gpus_per_node": "gpus_per_node",
            "tp": "tp",
            "pp": "pp",
            "ep": "ep",
            "cluster_size": "cluster_size",
        }
        invalid_values: list[str] = []
        for engine_key, config_key in engine_to_config_map.items():
            if engine_key in engine_mapping:
                cfg_value = getattr(self.world, config_key)
                engine_value = engine_mapping[engine_key]
                value = cfg_value or engine_value
                setattr(self.world, config_key, value)

                if value != engine_value:
                    invalid_values.append(
                        f"{config_key}: Config: {cfg_value} -> Engine: {engine_value}"
                    )
        # Print out all the invalid values we encountered.
        if invalid_values:
            raise ValueError("Invalid values detected in the engine config: \n"
                             f"{'\n'.join(invalid_values)}")

        if self.dataset_metadata is None:
            logger.warning("Dataset metadata is not set, skipping max sequence length validation.")
        else:
            # The engine config has a max sequence length, so we need to validate
            # that the dataset max sequence length is less than the engine max
            # sequence length.
            max_seq_len = self.dataset_metadata.seq_len_stats.maximum
            eng_max_seq_len = build_config["max_seq_len"]
            if max_seq_len > eng_max_seq_len:
                raise RuntimeError(
                    f"Engine supports a max sequence of {eng_max_seq_len}. "
                    "Provided dataset contains a maximum sequence of "
                    f"{max_seq_len}. Please rebuild a new engine "
                    "to support this dataset.")

        return self

    @model_validator(mode="after")
    def validate_heuristic_constraints(self) -> Self:
        if self.constraints is None:
            return self

        if self.mode == "build":
            build_options = [
                self.dataset_path, self.constraints.max_input_len,
                self.constraints.target_input_len
            ]
            if all(opt is None for opt in build_options):
                raise ValueError(
                    "No engine build option is selected, please provide at least one option."
                )
            elif sum([bool(opt) for opt in build_options]) > 1:
                raise ValueError(
                    "Multiple engine build options detected, please choose only one engine build option."
                )

            if not self.dataset_path and self.constraints.max_input_len:
                raise ValueError("Unspecified max_input_len for engine build.")

        return self

    @property
    def checkpoint(self) -> str:
        return str(self.environment.checkpoint_path or self.environment.model)

    @model_validator(mode="after")
    def validate_max_seq_len(self) -> Self:
        if self.batching_config.max_seq_len is None and self.modality != "text":
            self.batching_config.max_seq_len = 4096
        return self

    def _format_number(self, value: float) -> str:
        """Format number to fit within 9 characters including decimal."""
        if value >= 100000:
            return f"{value:9.2e}".ljust(
                9
            )  # Scientific notation for large numbers, padded to 9 characters
        return f"{value:9.4f}".ljust(
            9)  # Fixed point for smaller numbers, padded to 9 characters

    def get_dataset_summary(self) -> str:
        if self.dataset_metadata is None:
            raise ValueError("Dataset metadata is not set.")

        form = self._format_number
        isl_stats: PercentileStats = self.dataset_metadata.isl_stats
        osl_stats: PercentileStats = self.dataset_metadata.osl_stats
        seq_len_stats: PercentileStats = self.dataset_metadata.seq_len_stats

        return (
            "\n===========================================================\n"
            "= DATASET DETAILS\n"
            "===========================================================\n"
            f"Dataset Path:         {self.dataset_path}\n"
            f"Number of Sequences:  {self.dataset_metadata.num_requests}\n"
            "\n-- Percentiles statistics ---------------------------------\n\n"
            "        Input              Output           Seq. Length\n"
            "-----------------------------------------------------------\n"
            f"MIN:  {form(isl_stats.minimum)}          {form(osl_stats.minimum)}          {form(seq_len_stats.minimum)}\n"
            f"MAX:  {form(isl_stats.maximum)}          {form(osl_stats.maximum)}          {form(seq_len_stats.maximum)}\n"
            f"AVG:  {form(isl_stats.average)}          {form(osl_stats.average)}          {form(seq_len_stats.average)}\n"
            f"P50:  {form(isl_stats.p50)}          {form(osl_stats.p50)}          {form(seq_len_stats.p50)}\n"
            f"P90:  {form(isl_stats.p90)}          {form(osl_stats.p90)}          {form(seq_len_stats.p90)}\n"
            f"P95:  {form(isl_stats.p95)}          {form(osl_stats.p95)}          {form(seq_len_stats.p95)}\n"
            f"P99:  {form(isl_stats.p99)}          {form(osl_stats.p99)}          {form(seq_len_stats.p99)}\n"
            "===========================================================\n")


class BenchmarkEnvironment(BaseModel):
    model: str = Field(default="",
                       description="The HF model name to use for benchmarking.")
    checkpoint_path: Optional[Path] = Field(
        default=None,
        description="The path to the checkpoint to use for benchmarking.")
    workspace: Path = Field(
        default="/tmp", description="The workspace to use for engine building.")

    @property
    def model_path(self) -> str:
        return str(self.checkpoint_path or self.model)

    @computed_field(
        description=
        "The type of model being used, derived from the model configuration.")
    def model_type(self) -> str:
        return self.get_bench_model_config().model_type

    def get_tllm_model_config(self) -> TllmModelConfig:
        return TllmModelConfig.from_pretrained(
            self.model_path,
            trust_remote_code=True
        )

    def get_bench_model_config(self) -> ModelConfig:
        return get_model_config(self.model, self.checkpoint_path)


