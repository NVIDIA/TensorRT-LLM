import pytest

# isort: off
from .test_llm import (global_kvcache_config,
                       tinyllama_guided_decoding_test_harness)
# isort: on


@pytest.mark.gpu4
def test_tinyllama_guided_decoding_tp2pp2():
    tinyllama_guided_decoding_test_harness(
        tensor_parallel_size=2,
        pipeline_parallel_size=2,
        kv_cache_config=global_kvcache_config,
        backend='pytorch')
