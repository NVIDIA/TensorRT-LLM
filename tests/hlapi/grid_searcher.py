#!/usr/bin/env python
import operator
import time
from functools import reduce
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from tensorrt_llm import ModelConfig, logger
from tensorrt_llm.hlapi import CapacitySchedulerPolicy, KvCacheConfig
from tensorrt_llm.hlapi._perf_evaluator import LLMPerfEvaluator
from tensorrt_llm.hlapi.utils import print_colored


class GridSearcher:
    ''' Test all the combinations of the options for the LLM().
    Just for experimenting, not for production use. '''

    class Config:
        model_config: ModelConfig
        llm_kwargs: Dict[str, Any]

    def __init__(self, prune_space_for_debug: int = 1e8):
        self.prune_space_for_debug = prune_space_for_debug
        self.latest_latency_per_case: Optional[float] = None

    def evaluate(self,
                 model_config: ModelConfig,
                 samples_path: Path,
                 report_dir: Path,
                 specified_configs: Optional[List[Config]] = None,
                 num_samples: int = -1,
                 skip_steps=0,
                 skip_configs: Optional[List[dict]] = None,
                 memory_monitor_interval: Optional[int] = None):
        # Most of the knobs are referenced from docs/source/performance/perf-best-practices.md
        if not report_dir.exists():
            report_dir.mkdir(parents=True, exist_ok=True)
        skip_configs = set([tuple(d.items()) for d in (skip_configs or [])])

        self.model_config = model_config
        space = specified_configs or self.generate_cases(self.tunable_space)

        print_colored("Tunable options: ", color="green")
        for key, value in self.tunable_space.items():
            print_colored(f"  - {key}: {value}\n", color="green")
        print_colored("\n")

        for no, llm_kwargs in enumerate(space):
            if no >= self.prune_space_for_debug:
                break
            if no < skip_steps:
                continue

            skip_configs = skip_configs or set()
            if tuple(llm_kwargs.items()) in skip_configs:
                continue

            def capacity_scheduling_policy_str(policy: CapacitySchedulerPolicy):
                if policy == CapacitySchedulerPolicy.GUARANTEED_NO_EVICT:
                    return "guaranteed_no_evict"
                elif policy == CapacitySchedulerPolicy.MAX_UTILIZATION:
                    return "max_utilization"
                else:
                    raise ValueError(f"Unknown policy {policy}")

            origin_llm_kwargs = llm_kwargs.copy()
            origin_llm_kwargs[
                "capacity_scheduling_policy"] = capacity_scheduling_policy_str(
                    origin_llm_kwargs["capacity_scheduling_policy"])

            kvcache = KvCacheConfig()
            kvcache.enable_block_reuse = llm_kwargs.pop('kvcache_reuse_blocks')

            print_colored(f"Testing ", color="green")
            print_colored(f"{no}/{self.space_size}", color="bold_red")
            print_colored(f" case with {origin_llm_kwargs}\n", color="green")
            if self.latest_latency_per_case is not None:
                print_colored(
                    f"Estimated remaining time: {self.latest_latency_per_case * (self.space_size - no):.2f} min\n"
                )

            _start_time = time.time()
            with LLMPerfEvaluator.create(
                    model_config,
                    samples_path,
                    num_samples=num_samples,
                    warmup=max(num_samples // 10, 10),
                    kv_cache_config=kvcache,
                    memory_monitor_interval=memory_monitor_interval,
                    **llm_kwargs) as perf_evaluator:

                report_path = report_dir / f"report_{no}.json"
                assert perf_evaluator

                report = perf_evaluator.run()
                report.display()
                report.save_json(report_path, config=origin_llm_kwargs)
            self.latest_latency_per_case = (time.time() -
                                            _start_time) / 60  # min

    @property
    def tunable_space(self):
        tunable_options = dict(
            multi_block_mode=[False, True],
            kvcache_reuse_blocks=[False, True],
            capacity_scheduling_policy=[
                CapacitySchedulerPolicy.GUARANTEED_NO_EVICT,
                CapacitySchedulerPolicy.MAX_UTILIZATION
            ],
            enable_chunked_context=[False, True],
        )
        if self.model_config.parallel_config.is_multi_gpu:
            tunable_options["use_custom_all_reduce"] = [False, True]

        self.space_size = reduce(operator.mul,
                                 [len(v) for v in tunable_options.values()], 1)
        self.space_size = min(self.space_size, self.prune_space_for_debug)

        return tunable_options

    def generate_cases(self, tunable_options) -> Iterable[Dict[str, Any]]:
        if self.prune_space_for_debug:
            logger.warning("Pruning the space for debugging purpose")

        options = list(self.tunable_space.items())

        def gen_configs(options, config: dict):
            if not options:
                yield config
                return

            key, values = options[0]

            for option in values:
                new_config = config.copy()
                new_config[key] = option

                yield from gen_configs(options[1:], new_config)

        for config in gen_configs(options, {}):
            llm_kwargs = config.copy()
            yield llm_kwargs
