import json
import os
import sys
import tempfile
import time
from pathlib import Path

from tensorrt_llm.hlapi._perf_evaluator import (LLMPerfEvaluator,
                                                MemoryContinuousMonitorThread)
from tensorrt_llm.hlapi.llm import KvCacheConfig, ModelConfig

try:
    from .grid_searcher import GridSearcher
except:
    from grid_searcher import GridSearcher

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.llm_data import llm_models_root
from utils.util import force_ampere, skip_pre_ampere


def get_model_path(model_name):
    return str(llm_models_root() / model_name)


llama_model_path = get_model_path("llama-models/llama-7b-hf")


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
    config = ModelConfig(llama_model_path)

    with tempfile.TemporaryDirectory() as temp_dir:
        workspace = Path(temp_dir)
        samples_path = workspace / "data.json"
        gen_fake_samples(samples_path, 10, 5)

        # try to set some flags
        kvcache_config = KvCacheConfig(enable_block_reuse=True)

        evaluator = LLMPerfEvaluator.create(
            config,
            num_samples=10,
            samples_path=samples_path,
            warmup=10,
            kv_cache_config=kvcache_config,
        )
        assert evaluator
        report = evaluator.run()
        report.display()
        report.save_json(workspace / "report.json")


@skip_pre_ampere
def test_grid_search_tester(sample_length: int = 16,
                            report_root: Path = Path("./")):
    with tempfile.TemporaryDirectory() as temp_dir:
        workspace = Path(temp_dir)
        samples_path = workspace / "data.json"
        gen_fake_samples(samples_path, 10, sample_length)

        grid_searcher = GridSearcher(prune_space_for_debug=1)

        report_path = workspace / "report.json"

        model_config = ModelConfig(llama_model_path)

        input_len = int(sample_length * 2)
        output_len = int(sample_length * 2)
        max_num_tokens = 1024
        model_config._set_additional_options(max_output_len=output_len,
                                             max_input_len=input_len,
                                             max_num_tokens=max_num_tokens)

        grid_searcher.evaluate(
            model_config=model_config,
            samples_path=samples_path,
            report_dir=report_path,
            memory_monitor_interval=1,
        )


if __name__ == '__main__':
    test_perf_evaluator()
    test_grid_search_tester()
