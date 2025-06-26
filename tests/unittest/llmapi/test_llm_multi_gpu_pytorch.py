import pytest

# isort: off
from .test_llm import tinyllama_logits_processor_test_harness
from tensorrt_llm.llmapi import KvCacheConfig
from .test_llm_pytorch import (llama_v2_13b_lora_test_harness,
                               llama_7b_multi_lora_test_harness)

# isort: on

global_kv_cache_config = KvCacheConfig(free_gpu_memory_fraction=0.4)


@pytest.mark.gpu4
def test_tinyllama_logits_processor_tp2pp2():
    tinyllama_logits_processor_test_harness(backend="pytorch",
                                            tensor_parallel_size=2,
                                            pipeline_parallel_size=2)


@pytest.mark.gpu2
@pytest.mark.part0
@pytest.mark.parametrize("tp_size, pp_size", [(1, 2), (2, 1)])
def test_tinyllama_logits_processor_2gpu(tp_size: int, pp_size: int):
    tinyllama_logits_processor_test_harness(backend="pytorch",
                                            tensor_parallel_size=tp_size,
                                            pipeline_parallel_size=pp_size)


@pytest.mark.gpu2
def test_llama_v2_13b_lora_tp2():
    llama_v2_13b_lora_test_harness(tensor_parallel_size=2,
                                   kv_cache_config=global_kv_cache_config)


@pytest.mark.gpu2
def test_llama_7b_multi_lora_tp2():
    llama_7b_multi_lora_test_harness(tensor_parallel_size=2,
                                     kv_cache_config=global_kv_cache_config)
