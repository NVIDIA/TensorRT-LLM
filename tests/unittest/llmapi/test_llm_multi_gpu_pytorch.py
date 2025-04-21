import pytest
from test_llm import (global_kvcache_config,
                      tinyllama_guided_decoding_test_harness)

# isort: off
# isort: on


@pytest.mark.gpu4
def test_tinyllama_guided_decoding_tp2pp2_pytorch():
    llm_kwargs = {'backend': 'pytorch'}
    tinyllama_guided_decoding_test_harness(
        tensor_parallel_size=2,
        pipeline_parallel_size=2,
        kv_cache_config=global_kvcache_config,
        **llm_kwargs)
