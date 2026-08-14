import pytest

from tensorrt_llm import LLM
from tensorrt_llm.llmapi import KvCacheConfig

from ..conftest import llm_models_root
from .accuracy_core import MMLU, LlmapiAccuracyTestHarness

pytestmark = pytest.mark.ray


class TestLlama3_1_8BInstruct(LlmapiAccuracyTestHarness):
    MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
    MODEL_PATH = f"{llm_models_root()}/llama-3.1-model/Llama-3.1-8B-Instruct"

    @pytest.mark.skip_less_device(2)
    @pytest.mark.skip_less_device_memory(32000)
    def test_pp2_ray(self):
        kv_cache_config = KvCacheConfig(free_gpu_memory_fraction=0.6)

        with LLM(self.MODEL_PATH,
                 orchestrator_type="ray",
                 pipeline_parallel_size=2,
                 kv_cache_config=kv_cache_config) as llm:
            task = MMLU(self.MODEL_NAME)
            task.evaluate(llm)
