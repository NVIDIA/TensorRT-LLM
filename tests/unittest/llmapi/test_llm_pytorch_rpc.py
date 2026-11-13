# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json

import pytest
from utils.util import skip_ray

from tensorrt_llm import LLM
from tensorrt_llm.executor.rpc_proxy import GenerationExecutorRpcProxy
from tensorrt_llm.sampling_params import SamplingParams

from .llm_test_utils import global_kvcache_config, prompts
from .model_paths import llama_model_path


@skip_ray
@pytest.mark.parametrize("num_requests", [1, 5, 10])
def test_llm_rpc(num_requests: int):
    # TODO: remove the with-statement when shutdown hang issue is fixed
    with LLM(
        model=llama_model_path, kv_cache_config=global_kvcache_config, orchestrator_type="rpc"
    ) as llm:
        assert isinstance(llm._executor, GenerationExecutorRpcProxy)

        res = llm.generate(
            "Tell me a joke", sampling_params=SamplingParams(max_tokens=10, end_id=-1)
        )
        print(f"get result: {res}")

        assert len(res.outputs) == 1
        assert len(res.outputs[0].token_ids) == 10


@skip_ray
@pytest.mark.asyncio
async def test_llm_rpc_streaming():
    # TODO: remove the with-statement when shutdown hang issue is fixed
    with LLM(
        model=llama_model_path, kv_cache_config=global_kvcache_config, orchestrator_type="rpc"
    ) as llm:
        assert isinstance(llm._executor, GenerationExecutorRpcProxy)

        outputs = []
        async for output in llm.generate_async(
            "Tell me a joke",
            sampling_params=SamplingParams(max_tokens=10, end_id=-1),
            streaming=True,
        ):
            outputs.append(output.outputs[0].text)
        "".join(outputs)
        print(f"get result: {outputs}")


@skip_ray
def test_llm_rpc_get_stats():
    """Test that get_stats works with RPC orchestrator."""
    with LLM(
        model=llama_model_path,
        kv_cache_config=global_kvcache_config,
        enable_iter_perf_stats=True,
        orchestrator_type="rpc",
    ) as llm:
        assert isinstance(llm._executor, GenerationExecutorRpcProxy)

        # Generate some output to produce stats
        for output in llm.generate(prompts, sampling_params=SamplingParams(max_tokens=5)):
            print(output)

        stats = llm.get_stats(timeout=5)

        assert len(stats) > 0, "Should have at least one stats entry"
        # Stats should be JSON strings that can be parsed
        parsed = json.loads(stats[0]) if isinstance(stats[0], str) else stats[0]
        assert "iter" in parsed, "Stats should contain 'iter' field"
        assert "cpuMemUsage" in parsed, "Stats should contain 'cpuMemUsage' field"


@skip_ray
@pytest.mark.asyncio
async def test_llm_rpc_get_stats_async():
    """Test that get_stats_async works with RPC orchestrator."""
    import json

    with LLM(
        model=llama_model_path,
        kv_cache_config=global_kvcache_config,
        enable_iter_perf_stats=True,
        orchestrator_type="rpc",
    ) as llm:
        assert isinstance(llm._executor, GenerationExecutorRpcProxy)

        # Generate some output to produce stats
        async for output in llm.generate_async(
            prompts[0], sampling_params=SamplingParams(max_tokens=5)
        ):
            print(output)

        # Get stats via async API
        stats_result = llm.get_stats_async(timeout=2)

        # Should be able to iterate over results
        stats_count = 0
        async for stat in stats_result:
            parsed = json.loads(stat) if isinstance(stat, str) else stat
            assert "iter" in parsed, "Stats should contain 'iter' field"
            stats_count += 1
            if stats_count >= 1:
                break  # Just verify we can get at least one

        assert stats_count > 0, "Should have received at least one stat"
