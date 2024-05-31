import json
import os
import sys
import tempfile
import time
from pathlib import Path

from tensorrt_llm.hlapi._perf_evaluator import (LLMPerfEvaluator,
                                                MemoryContinuousMonitorThread)
from tensorrt_llm.hlapi.llm import KvCacheConfig, ModelConfig

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.llm_data import llm_models_root
from utils.util import force_ampere


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


if __name__ == '__main__':
    test_perf_evaluator()
