from typing import Any, Dict

from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.pyexecutor.model_engine import validate_and_set_kv_cache_quant
from tensorrt_llm.bench.build.build import get_benchmark_engine_settings
from tensorrt_llm.bench.dataclasses.scenario import ScenarioSpecification
from tensorrt_llm.bench.tuning import DefaultLlmHeuristic
from tensorrt_llm.bench.tuning.utils import get_model_config


class PytMaxThroughputScenario(DefaultLlmHeuristic):
    """Maximum throughput heuristic tuning for the PyTorch backend."""

    @classmethod
    def get_settings(cls, scenario: ScenarioSpecification) -> Dict[str, Any]:
        # Collect reference information.
        # Configuration classes
        tllm_model_config = scenario.get_tllm_model_config()
        bench_model_config = scenario.get_bench_model_config()

        # World, dataset, and settings information.
        world = scenario.world
        dataset_metadata = scenario.dataset_metadata
        kv_cache_dtype = \
            scenario.llm_config.extra_llm_api_options.get(
                "kv_cache_dtype", "auto"
            )
        chunked_prefill = \
            scenario.llm_config.extra_llm_api_options.get(
                "enable_chunked_prefill", False
            )

        # Update the KV cache settings.
        validate_and_set_kv_cache_quant(tllm_model_config, kv_cache_dtype)

        # Find tuned parameters
        max_batch_size, max_num_tokens = get_benchmark_engine_settings(
            bench_model_config,
            tllm_model_config.quant_config,
            world.tp_size,
            world.pp_size,
            dataset_metadata.avg_isl,
            dataset_metadata.avg_osl,
        )

        # Update CUDA graph settings.
        cuda_graph_config = {
            "padding_enabled": True,
            "max_batch_size": max_batch_size,
        }

        # Get the initial settings from the parent class (absolute default settings)
        llm_args = super().get_settings(scenario)
        # Update scheduler settings for scheduling in the IFB scheduler.
        llm_args |= {
            "scheduler_config": {
                "capacity_scheduler_policy": "GUARANTEED_NO_EVICT",
                "context_chunking_policy": "FIRST_COME_FIRST_SERVED",
            },
            "cuda_graph_config": cuda_graph_config,
            "max_num_tokens": max_num_tokens,
            "enable_chunked_prefill": chunked_prefill,
            "kv_cache_dtype": kv_cache_dtype,
        }

        return llm_args


class TrtMaxThroughputScenario(DefaultLlmHeuristic):
    """Maximum throughput heuristic tuning for the TensorRT backend."""
    @classmethod
    def get_settings(cls, scenario: ScenarioSpecification) -> Dict[str, Any]:
        if scenario.mode == "benchmark":
            return cls.get_benchmark_settings(scenario)
        elif scenario.mode == "build":
            return cls.get_build_settings(scenario)
        else:
            raise ValueError(f"Invalid mode: {scenario.mode}")

    @classmethod
    def get_benchmark_settings(cls, scenario: ScenarioSpecification) -> Dict[str, Any]:
        llm_args = super().get_settings(scenario)
        llm_args |= {
            "scheduler_config":
            dict(
                capacity_scheduler_policy="MAX_UTILIZATION",
                dynamic_batch_config=dict(
                    enable_batch_size_tuning=True,
                    enable_max_num_tokens_tuning=True,
                    dynamic_batch_moving_average_window=128,
                ),
            ),
            "extended_runtime_perf_knob_config":
            dict(
                cuda_graph_mode=True,
                multi_block_mode=True,
                enable_context_fmha_fp32_acc=False,
            ),
            "enable_chunked_prefill": True,
        }
        return llm_args

    @classmethod
    def get_build_settings(cls, scenario: ScenarioSpecification) -> Dict[str, Any]:
        ...

