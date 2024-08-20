import json
import os
import sys
import tempfile
import time
from pathlib import Path

from tensorrt_llm.hlapi import BuildConfig, KvCacheConfig
from tensorrt_llm.hlapi._perf_evaluator import (LLMPerfEvaluator,
                                                MemoryContinuousMonitorThread)

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.util import force_ampere

try:
    from test_llm import llama_model_path
except ImportError:
    from .test_llm import llama_model_path


def test_memory_thread():
    thread = MemoryContinuousMonitorThread(0.5)
    thread.start()
    time.sleep(3)
    thread.stop()
    print(thread.memory_samples)
    print('max', thread.memory_samples.get_max())
    print('min', thread.memory_samples.get_min())
    print('ave', thread.memory_samples.get_average())


def gen_fake_samples(samples_path: str, num_samples: int, sample_length: int):
    data = {
        "samples": [{
            "input_ids": [20] * sample_length,
            "output_len": sample_length
        } for _ in range(num_samples)]
    }
    with open(samples_path, "w") as f:
        json.dump(data, f)


@force_ampere
def test_perf_evaluator():
    with tempfile.TemporaryDirectory() as temp_dir:
        workspace = Path(temp_dir)
        samples_path = workspace / "data.json"
        gen_fake_samples(samples_path, 10, 5)

        # try to set some flags
        kvcache_config = KvCacheConfig(enable_block_reuse=True)

        build_config = BuildConfig()
        build_config.plugin_config._use_paged_context_fmha = True

        evaluator = LLMPerfEvaluator.create(
            model=llama_model_path,
            num_samples=10,
            samples_path=samples_path,
            warmup=10,
            kv_cache_config=kvcache_config,
            build_config=build_config,
        )
        assert evaluator
        report = evaluator.run()
        report.display()
        report.save_json(workspace / "report.json")


if __name__ == '__main__':
    test_perf_evaluator()
