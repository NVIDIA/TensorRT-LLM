from __future__ import annotations

from dataclasses import dataclass
from importlib.util import find_spec
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

import yaml
from pydantic import (BaseModel, Field, PositiveFloat, field_validator,
                      model_validator)

import tensorrt_llm.bindings.executor as trtllm
from tensorrt_llm._torch.pyexecutor.config import PyTorchConfig
from tensorrt_llm.bench.dataclasses.enums import IFBSchedulingPolicy
from tensorrt_llm.builder import BuildConfig
from tensorrt_llm.llmapi.llm_utils import BuildCacheConfig, CalibConfig
from tensorrt_llm.logger import logger
from tensorrt_llm.models.modeling_utils import (QuantConfig,
                                                SpeculativeDecodingMode)

SPECULATIVE_MAP = {
    SpeculativeDecodingMode.NONE: lambda *args: None,
    SpeculativeDecodingMode.MEDUSA: trtllm.DecodingMode.Medusa,
}


class RuntimeConfig(BaseModel):
    model: str
    model_path: Optional[Path] = None
    engine_dir: Optional[Path] = None
    sw_version: str
    settings_config: ExecutorSettingsConfig
    world_config: ExecutorWorldConfig
    decoding_config: DecodingConfig
    performance_options: PerformanceOptions
    backend: Literal["pytorch", None] = None
    extra_llm_api_options: Optional[str] = None

    def _update_with_extra_options(self, llm_args: Dict) -> Dict:
        if self.extra_llm_api_options is not None:
            with open(self.extra_llm_api_options, 'r') as f:
                llm_args_dict = yaml.safe_load(f)

            field_mapping = {
                "quant_config": QuantConfig,
                "calib_config": CalibConfig,
                "build_config": BuildConfig,
                "kv_cache_config": trtllm.KvCacheConfig,
                "decoding_config": trtllm.DecodingConfig,
                "enable_build_cache": BuildCacheConfig,
                "peft_cache_config": trtllm.PeftCacheConfig,
                "scheduler_config": trtllm.SchedulerConfig,
                "speculative_config": trtllm.LookaheadDecodingConfig,
                "batching_type": trtllm.BatchingType,
                "extended_runtime_perf_knob_config":
                trtllm.ExtendedRuntimePerfKnobConfig,
                "pytorch_backend_config": PyTorchConfig,
            }
            for field, field_type in field_mapping.items():
                if field in llm_args_dict:
                    llm_args_dict[field] = field_type(**llm_args_dict[field])
                    logger.warning(
                        f"Overriding {field} because it's specified in {self.extra_llm_api_options}."
                    )

            llm_args = llm_args | llm_args_dict

        return llm_args

    def get_llm_args(self) -> Dict:
        model = self.engine_dir or self.model_path or self.model

        llm_args = {
            "scheduler_config":
            self.settings_config.get_scheduler_config(),
            "model":
            model,
            "skip_tokenizer_init":
            True,
            "pipeline_parallel_size":
            self.world_config.pp_size,
            "tensor_parallel_size":
            self.world_config.tp_size,
            "gpus_per_node":
            self.world_config.gpus_per_node,
            "moe_expert_parallel_size":
            self.world_config.ep_size,
            "trust_remote_code":
            True,
            "kv_cache_config":
            self.settings_config.get_kvcache_config(),
            "enable_chunked_prefill":
            self.settings_config.chunking,
            "extended_runtime_perf_knob_config":
            self.performance_options.get_perf_config(),
            "decoding_config":
            self.decoding_config.get_decoding_config(),
            "batching_type":
            trtllm.BatchingType.INFLIGHT,
            "max_batch_size":
            self.settings_config.max_batch_size,
            "max_num_tokens":
            self.settings_config.max_num_tokens,
        }

        if self.backend == "pytorch":
            llm_args["pytorch_backend_config"] = \
                self.performance_options.get_pytorch_perf_config()

        return self._update_with_extra_options(llm_args)

    @model_validator(mode="after")
    def validate_full_config(self) -> RuntimeConfig:
        # TODO: Check engine to make sure it can support Medusa.
        return self


@dataclass
class PerformanceOptions:
    cuda_graphs: bool = False
    multi_block_mode: bool = True
    cuda_graph_cache_size: int = 1000
    pytorch_config: Dict[str, Any] = Field(default_factory=dict)

    def get_perf_config(self) -> trtllm.ExtendedRuntimePerfKnobConfig:
        config = trtllm.ExtendedRuntimePerfKnobConfig()
        config.cuda_graph_mode = self.cuda_graphs
        config.multi_block_mode = self.multi_block_mode
        config.cuda_graph_cache_size = self.cuda_graph_cache_size

        return config

    def get_pytorch_perf_config(self) -> PyTorchConfig:
        return PyTorchConfig(**self.pytorch_config)


class DecodingConfig(BaseModel):
    medusa_choices: Optional[List[List[int]]] = None
    decoding_mode: SpeculativeDecodingMode = SpeculativeDecodingMode.NONE

    @field_validator("decoding_mode")
    @classmethod
    def decoding_mode_validator(
        cls, value: Union[str, int,
                          SpeculativeDecodingMode]) -> SpeculativeDecodingMode:
        return SpeculativeDecodingMode(value)

    @model_validator(mode="after")
    def validate_speculative_decoding(self) -> DecodingConfig:
        if self.medusa_choices and self.decoding_mode != SpeculativeDecodingMode.MEDUSA:
            raise RuntimeError(
                "Attempting to use set Medusa choices with a non-Medusa engine."
                " Verify that you are using a Medusa engine.")

        return self

    def get_decoding_config(self) -> trtllm.DecodingConfig:
        """Create a populated TRT-LLM DecodingConfig."""
        kwargs = {"decoding_mode": SPECULATIVE_MAP[self.decoding_mode]()}

        if self.medusa_choices is not None:
            kwargs["medusa_choices"] = self.medusa_choices

        return trtllm.DecodingConfig(**kwargs)


class ExecutorWorldConfig(BaseModel):
    pp_size: int = 1
    tp_size: int = 1
    world_size: int = 1
    gpus_per_node: int = 8
    leader_mode: bool = False
    ep_size: Optional[int] = None

    @model_validator(mode="after")
    def validate_world_size(self) -> ExecutorWorldConfig:
        parallel_world = self.pp_size * self.tp_size
        num_gpus = self.world_size * self.gpus_per_node
        valid_world = bool(num_gpus >= parallel_world)

        if not valid_world:
            raise ValueError(
                f"World configuration is invalid, TP * PP ({parallel_world})"
                "does not equal the total number of available GPUs"
                f"({num_gpus}).")

        return self

    def _get_tensorrt_llm_executor_worker_path(self) -> Path:
        module_path = find_spec("tensorrt_llm").loader.get_filename()
        exec_path = Path(module_path).parent / 'bin' / 'executorWorker'
        return exec_path.absolute()

    def get_parallel_config(self) -> trtllm.ParallelConfig:
        if self.leader_mode:
            comm_mode = trtllm.CommunicationMode.LEADER
            orchestrator_config = None
        else:
            comm_mode = trtllm.CommunicationMode.ORCHESTRATOR
            orchestrator_config = trtllm.OrchestratorConfig(
                True, str(self._get_tensorrt_llm_executor_worker_path()))

        return trtllm.ParallelConfig(
            trtllm.CommunicationType.MPI,
            comm_mode,
            orchestrator_config=orchestrator_config,
        )


class ExecutorSettingsConfig(BaseModel):
    chunking: bool = True
    scheduler_policy: IFBSchedulingPolicy = IFBSchedulingPolicy.MAX_UTILIZTION
    max_batch_size: int
    max_num_tokens: int
    kv_cache_percent: PositiveFloat = Field(default=.90, le=1.0)
    kv_cache_reuse: bool = False
    dynamic_max_batch_size: bool = True
    dynamic_max_num_tokens: bool = False  # Will enable after more validation.

    def get_dynamic_config(self) -> trtllm.DynamicBatchConfig:
        window_size = 128 if self.dynamic_max_batch_size else 0
        return trtllm.DynamicBatchConfig(
            self.dynamic_max_batch_size,
            self.dynamic_max_num_tokens,
            window_size,
        )

    def get_kvcache_config(self) -> trtllm.KvCacheConfig:
        return trtllm.KvCacheConfig(
            free_gpu_memory_fraction=self.kv_cache_percent,
            enable_block_reuse=False,
        )

    def get_scheduler_config(self) -> trtllm.SchedulerConfig:
        if self.chunking:
            return trtllm.SchedulerConfig(
                capacity_scheduler_policy=self.scheduler_policy.value,
                context_chunking_policy=trtllm.ContextChunkingPolicy.
                FIRST_COME_FIRST_SERVED,
                dynamic_batch_config=self.get_dynamic_config(),
            )
        else:
            return trtllm.SchedulerConfig(
                capacity_scheduler_policy=self.scheduler_policy.value,
                dynamic_batch_config=self.get_dynamic_config())
