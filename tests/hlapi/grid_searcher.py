from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from tensorrt_llm import ModelConfig, logger
from tensorrt_llm.hlapi import KvCacheConfig, SchedulerPolicy
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

        space = specified_configs or self.gen_space()

        for no, llm_kwargs in enumerate(space):
            if no > self.prune_space_for_debug:
                break
            if no < skip_steps:
                continue

            skip_configs = skip_configs or set()
            if tuple(llm_kwargs.items()) in skip_configs:
                continue

            def scheduleing_policy_str(policy: SchedulerPolicy):
                if policy == SchedulerPolicy.GUARANTEED_NO_EVICT:
                    return "guaranteed_no_evict"
                elif policy == SchedulerPolicy.MAX_UTILIZATION:
                    return "max_utilization"
                else:
                    raise ValueError(f"Unknown policy {policy}")

            origin_llm_kwargs = llm_kwargs.copy()
            origin_llm_kwargs["scheduling_policy"] = scheduleing_policy_str(
                origin_llm_kwargs["scheduling_policy"])

            kvcache = KvCacheConfig()
            kvcache.enable_block_reuse = llm_kwargs.pop('kvcache_reuse_blocks')

            print_colored(
                f"Running {no}th experiment with {origin_llm_kwargs}\n",
                color="green")

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

    def gen_space(self) -> Iterable[Dict[str, Any]]:

        tunable_options = dict(
            use_custom_all_reduce=[False, True],
            multi_block_mode=[False, True],
            kvcache_reuse_blocks=[False, True],
            scheduling_policy=[
                SchedulerPolicy.GUARANTEED_NO_EVICT,
                SchedulerPolicy.MAX_UTILIZATION
            ],
        )

        if self.prune_space_for_debug:
            logger.warning("Pruning the space for debugging purpose")

        options = list(tunable_options.items())

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
