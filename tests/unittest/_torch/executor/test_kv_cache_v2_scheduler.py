import json
from pathlib import Path

import pytest
from utils.llm_data import llm_models_root

from tensorrt_llm import LLM, SamplingParams
from tensorrt_llm.llmapi import (
    CapacitySchedulerPolicy,
    CudaGraphConfig,
    KvCacheConfig,
    SchedulerConfig,
)


# A test case of mmlu_llama from lm_eval
@pytest.fixture(scope="module")
def test_case():
    with open(Path(__file__).parent / "test_overlap_scheduler_input.json") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def model_path():
    return llm_models_root() / "gpt_oss/gpt-oss-20b"


def create_llm(
    model_dir,
    disable_overlap_scheduler,
    sampler_type,
    env_overrides=None,
    kv_cache_config=None,
    scheduler_config=None,
):
    """Create LLM with specific overlap scheduler setting"""
    pytorch_config = dict(
        disable_overlap_scheduler=disable_overlap_scheduler, sampler_type=sampler_type
    )

    if kv_cache_config is None:
        kv_cache_config = KvCacheConfig(enable_block_reuse=False)

    llm_kwargs = dict(
        model=str(model_dir),
        tensor_parallel_size=1,
        trust_remote_code=True,
        enable_chunked_prefill=True,
        cuda_graph_config=CudaGraphConfig(),
        **pytorch_config,
        kv_cache_config=kv_cache_config,
        max_num_tokens=128,  # Only one request longer than max_num_tokens is required to test chunked prefill
        env_overrides=env_overrides,
    )

    if scheduler_config is not None:
        llm_kwargs["scheduler_config"] = scheduler_config

    return LLM(**llm_kwargs)


def test_kv_cache_v2_policy_consistency(model_path, test_case):
    """
    Test that KVCacheManagerV2 produces consistent outputs between
    GUARANTEED_NO_EVICT and MAX_UTILIZATION policies.
    """
    # Test configuration
    prompts = test_case["prompts"][:2]  # Use fewer prompts for faster test
    max_new_tokens = 50  # Shorter for faster test

    sampling_config = SamplingParams(
        max_tokens=max_new_tokens,
        temperature=0.0,  # Deterministic for comparison
        top_p=1.0,
        n=1,
        use_beam_search=False,
    )

    # KVCacheConfig for V2
    # host_kv_cache_size is needed for max utilization scheduling due to request eviction
    kv_cache_config = KvCacheConfig(
        free_gpu_memory_fraction=0.7,
        dtype="auto",
        use_kv_cache_manager_v2=True,
        enable_block_reuse=False,
        host_kv_cache_size=10737418240,
    )

    # Test with GUARANTEED_NO_EVICT
    scheduler_config_no_evict = SchedulerConfig(
        capacity_scheduler_policy=CapacitySchedulerPolicy.GUARANTEED_NO_EVICT
    )

    with create_llm(
        model_path,
        disable_overlap_scheduler=False,
        sampler_type="TorchSampler",
        kv_cache_config=kv_cache_config,
        scheduler_config=scheduler_config_no_evict,
    ) as llm:
        outputs_no_evict = llm.generate(prompts, sampling_params=sampling_config)
        texts_no_evict = [output.outputs[0].text for output in outputs_no_evict]

    # Test with MAX_UTILIZATION
    scheduler_config_max_util = SchedulerConfig(
        capacity_scheduler_policy=CapacitySchedulerPolicy.MAX_UTILIZATION
    )

    with create_llm(
        model_path,
        disable_overlap_scheduler=False,
        sampler_type="TorchSampler",
        kv_cache_config=kv_cache_config,
        scheduler_config=scheduler_config_max_util,
    ) as llm:
        outputs_max_util = llm.generate(prompts, sampling_params=sampling_config)
        texts_max_util = [output.outputs[0].text for output in outputs_max_util]

    # Verify outputs are consistent between policies
    for i, (no_evict, max_util) in enumerate(zip(texts_no_evict, texts_max_util)):
        assert no_evict == max_util, (
            f"Output mismatch at index {i}:\nNO_EVICT: {no_evict}\nMAX_UTIL: {max_util}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
