from typing import Any, Dict

from tensorrt_llm.bench.dataclasses.scenario import (ScenarioSpecification,
                                                     TuningConstraints,
                                                     WorldConfig)
from tensorrt_llm.bench.tuning.heuristics import DefaultScenario


class TrtMaxThroughputScenario(DefaultScenario):

    @classmethod
    def get_settings(cls, scenario: ScenarioSpecification, world: WorldConfig,
                     tuning: TuningConstraints) -> Dict[str, Any]:
        llm_args = super().get_settings(scenario, world, tuning)
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
                multi_block_mode=True,
                enable_context_fmha_fp32_acc=False,
            ),
        }
        return llm_args
