# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import random
import time

import pytest
from utils.util import skip_ray

from tensorrt_llm import LLM
from tensorrt_llm.disaggregated_params import DisaggregatedParams
from tensorrt_llm.llmapi import CacheTransceiverConfig, KvCacheConfig
from tensorrt_llm.llmapi.llm_args import SchedulerConfig, WaitingQueuePolicy
from tensorrt_llm.sampling_params import SamplingParams

from .llm_test_utils import global_kvcache_config, global_kvcache_config_no_reuse
from .model_paths import llama_model_path


@pytest.mark.threadleak(enabled=False)
@skip_ray
@pytest.mark.parametrize("transceiver_runtime", ["PYTHON"])
def test_llm_context_only_timed_out(transceiver_runtime):
    tp_size = 1
    use_overlap = False
    enable_iter_req_stats = False

    llm_args_extra = {}

    llm_args_extra.update(
        dict(
            enable_iter_perf_stats=True,
            enable_iter_req_stats=enable_iter_req_stats,
            disable_overlap_scheduler=not use_overlap,
        )
    )

    backend = "NIXL"
    llm = LLM(
        model=llama_model_path,
        kv_cache_config=global_kvcache_config,
        tensor_parallel_size=tp_size,
        cache_transceiver_config=CacheTransceiverConfig(
            backend=backend, kv_transfer_timeout_ms=1000, transceiver_runtime=transceiver_runtime
        ),
        **llm_args_extra,
    )

    max_tokens = 1
    sampling_params = SamplingParams(max_tokens=max_tokens)

    disaggregated_params = DisaggregatedParams(request_type="context_only")

    prompts0 = [
        "What is your name?",
    ]
    prompts1 = [
        "Nvidia is awesome because",
    ]

    # Send context-only request
    for output in llm.generate(
        prompts1, sampling_params=sampling_params, disaggregated_params=disaggregated_params
    ):
        print(output)

    # Wait until the context-only request has allocated KV cache blocks
    max_retries = 10
    all_results = []
    for _ in range(max_retries):
        results = llm.get_stats(2)
        all_results.extend(results)
        if all_results and all_results[-1]["kvCacheStats"]["usedNumBlocks"] > 0:
            break
        time.sleep(1)
    else:
        pytest.fail(f"Context-only KV cache blocks not allocated after {max_retries} retries")
    results = all_results

    context_only_used_num_blocks = results[-1]["kvCacheStats"]["usedNumBlocks"]
    print(f"Context only used num blocks: {context_only_used_num_blocks}")

    # Sleep 5 seconds to allow context only request to time out
    time.sleep(5)

    # Send regular request
    for output in llm.generate(prompts0, sampling_params=sampling_params):
        print(output)

    # Wait until KV cache blocks are released (usedNumBlocks == 0)
    max_retries = 10
    all_results = []
    for _ in range(max_retries):
        results = llm.get_stats(2)
        all_results.extend(results)
        if all_results and all_results[-1]["kvCacheStats"]["usedNumBlocks"] == 0:
            break
        time.sleep(1)
    else:
        pytest.fail(f"KV cache blocks not released after {max_retries} retries")
    results = all_results

    final_used_num_blocks = results[-1]["kvCacheStats"]["usedNumBlocks"]
    assert final_used_num_blocks == 0


# A context-only request times out when exhausted KV cache prevents scheduling.
@pytest.mark.threadleak(enabled=False)
@skip_ray
@pytest.mark.parametrize("sender_future_timeout_ms", [100, 1000])
@pytest.mark.parametrize("backend", ["NIXL"])
@pytest.mark.parametrize("transceiver_runtime", ["PYTHON"])
def test_llm_context_only_timed_out_kv_cache_exhausted(
    sender_future_timeout_ms, backend, transceiver_runtime
):
    tp_size = 1
    use_overlap = False
    enable_iter_req_stats = False

    llm_args_extra = {}

    llm_args_extra.update(
        dict(
            enable_iter_perf_stats=True,
            enable_iter_req_stats=enable_iter_req_stats,
            disable_overlap_scheduler=not use_overlap,
        )
    )

    kv_cache_config = KvCacheConfig(
        free_gpu_memory_fraction=0.1, max_tokens=1000, enable_block_reuse=False
    )
    llm = LLM(
        model=llama_model_path,
        kv_cache_config=kv_cache_config,
        tensor_parallel_size=tp_size,
        cache_transceiver_config=CacheTransceiverConfig(
            backend=backend,
            kv_transfer_timeout_ms=1000,
            kv_transfer_sender_future_timeout_ms=sender_future_timeout_ms,
            transceiver_runtime=transceiver_runtime,
        ),
        **llm_args_extra,
    )

    max_tokens = 1
    sampling_params = SamplingParams(max_tokens=max_tokens)

    disaggregated_params = DisaggregatedParams(request_type="context_only")

    prompts0 = [
        "What is your name?",
    ]
    prompts1 = [
        "lorem ipsum dolor sit amet consectetur adipiscing elit "
        "sed do eiusmod tempor incididunt ut labore et dolore magna aliqua " * 10
    ]

    # Send context-only request
    for output in llm.generate(
        prompts1 * 10, sampling_params=sampling_params, disaggregated_params=disaggregated_params
    ):
        print(output)

    max_retries = 10
    all_results = []
    for _ in range(max_retries):
        results = llm.get_stats(2)
        all_results.extend(results)

    assert len(all_results) > 0

    context_only_used_num_blocks = all_results[-1]["kvCacheStats"]["usedNumBlocks"]
    print(f"Context only used num blocks: {context_only_used_num_blocks}")

    # Sleep 5 seconds to allow context only request to time out
    time.sleep(5)

    # Send regular request
    for output in llm.generate(prompts0, sampling_params=sampling_params):
        print(output)

    # Get number of allocated blocks
    results = llm.get_stats(2)
    assert len(results) == 1
    final_used_num_blocks = results[0]["kvCacheStats"]["usedNumBlocks"]

    assert final_used_num_blocks == 0


_DISAGG_CANCEL_REQUEST_TIMEOUT_SECONDS = 120.0


# Two iterations retain repeated cleanup coverage without making the L0 test
# carry stress-test exposure.
_DISAGG_CANCEL_TEST_ITERATIONS = 2


@pytest.mark.threadleak(enabled=False)
@skip_ray
# https://nvbugs/6608382: isolate the test from reused MPI worker state while
# the rare context-response stall is being root-caused.
@pytest.mark.private_mpi_session
@pytest.mark.timeout(600)
@pytest.mark.asyncio
@pytest.mark.parametrize("transceiver_runtime", ["PYTHON"])
async def test_llm_disagg_gen_cancelled(transceiver_runtime):
    tp_size = 1
    use_overlap = False
    enable_iter_req_stats = False

    llm_args_extra = {}

    llm_args_extra.update(
        dict(
            enable_iter_perf_stats=True,
            enable_iter_req_stats=enable_iter_req_stats,
            disable_overlap_scheduler=not use_overlap,
        )
    )

    backend = "NIXL"
    llm_ctx = LLM(
        model=llama_model_path,
        kv_cache_config=global_kvcache_config_no_reuse,
        tensor_parallel_size=tp_size,
        cache_transceiver_config=CacheTransceiverConfig(
            backend=backend, kv_transfer_timeout_ms=1000, transceiver_runtime=transceiver_runtime
        ),
        **llm_args_extra,
    )

    llm_gen = LLM(
        model=llama_model_path,
        kv_cache_config=global_kvcache_config_no_reuse,
        tensor_parallel_size=tp_size,
        cache_transceiver_config=CacheTransceiverConfig(
            backend=backend, kv_transfer_timeout_ms=1000, transceiver_runtime=transceiver_runtime
        ),
        **llm_args_extra,
    )

    try:
        prev_after_free_num_blocks = 0
        for iteration in range(_DISAGG_CANCEL_TEST_ITERATIONS):
            max_tokens = 1
            sampling_params = SamplingParams(max_tokens=max_tokens)
            disaggregated_params = DisaggregatedParams(request_type="context_only")

            prompt = [
                "lorem ipsum dolor sit amet consectetur adipiscing elit "
                "sed do eiusmod tempor incididunt ut labore et dolore magna aliqua " * 10
            ]
            # Send context-only request
            ctx_output = llm_ctx.generate_async(
                prompt[0],
                sampling_params=sampling_params,
                disaggregated_params=disaggregated_params,
            )
            ctx_output.result(timeout=_DISAGG_CANCEL_REQUEST_TIMEOUT_SECONDS)

            max_tokens = 10000
            sampling_params = SamplingParams(max_tokens=max_tokens, ignore_eos=True)
            disaggregated_params = ctx_output.disaggregated_params
            disaggregated_params.request_type = "generation_only"

            # Send gen-only request
            gen_output = llm_gen.generate_async(
                prompt[0],
                sampling_params=sampling_params,
                disaggregated_params=disaggregated_params,
            )

            # Sleep a little to have tokens generated, between 0.2 and 0.7 seconds
            sleep_time = random.uniform(0.2, 0.7)
            time.sleep(sleep_time)
            # Abort the generation request.
            gen_output.abort()
            result = await asyncio.wait_for(
                gen_output.aresult(), timeout=_DISAGG_CANCEL_REQUEST_TIMEOUT_SECONDS
            )
            num_output_tokens = len(result.outputs[0].token_ids)
            print(f"num output tokens: {num_output_tokens}")
            assert result.outputs[0].finish_reason == "cancelled"

            # Wait until KV cache blocks are released (usedNumBlocks == 0)
            time.sleep(1.0)
            max_retries = 10
            all_results = []
            for _ in range(max_retries):
                results = llm_gen.get_stats(2)
                print("len(results):", len(results))
                all_results.extend(results)
                if all_results and all_results[-1]["kvCacheStats"]["usedNumBlocks"] == 0:
                    break
                time.sleep(1)
            else:
                pytest.fail(f"KV cache blocks not released after {max_retries} retries")
            results = all_results

            after_used_num_blocks = results[-1]["kvCacheStats"]["usedNumBlocks"]
            assert after_used_num_blocks == 0

            after_free_num_blocks = results[-1]["kvCacheStats"]["freeNumBlocks"]
            # Check that number of free blocks stays the same
            if iteration > 0:
                assert after_free_num_blocks == prev_after_free_num_blocks

            # Check that number of free blocks stays the same
            prev_after_free_num_blocks = after_free_num_blocks
    finally:
        llm_ctx.shutdown()
        llm_gen.shutdown()


@pytest.mark.threadleak(enabled=False)
@skip_ray
@pytest.mark.timeout(60)
def test_priority_request_completes_before_low_priority():
    """High-priority request must be scheduled before a lower-priority one.

    Setup (max_batch_size=1, WaitingQueuePolicy.PRIORITY):
      - Request 1: no explicit priority (default 0.5), max_tokens=500  – long, ignore_eos
      - Request 2: no explicit priority (default 0.5), max_tokens=5   – short
      - Request 3: priority=0.9,                       max_tokens=5   – short + high priority

    All three requests are submitted before the executor has a chance to
    schedule any of them.  With a PRIORITY waiting queue the executor will
    pick Request 3 ahead of Request 2.  Therefore Request 3 must complete
    before Request 2 regardless of how long Request 1 takes.
    """
    llm = LLM(
        model=llama_model_path,
        max_batch_size=1,
        kv_cache_config=KvCacheConfig(enable_block_reuse=False),
        scheduler_config=SchedulerConfig(waiting_queue_policy=WaitingQueuePolicy.PRIORITY),
    )

    prompt = "A B C D E F G H I J"
    completion_order = []

    async def run(llm: LLM):
        # Submit all three requests before the event loop can schedule any of
        # them – they all land in the waiting queue simultaneously.
        # Request 1 uses ignore_eos so it keeps the batch slot busy for the
        # full max_tokens count rather than stopping at an early EOS token.
        out1 = llm.generate_async(
            prompt, SamplingParams(max_tokens=500, temperature=0, ignore_eos=True)
        )
        out2 = llm.generate_async(prompt, SamplingParams(max_tokens=5, temperature=0))
        out3 = llm.generate_async(prompt, SamplingParams(max_tokens=5, temperature=0), priority=0.9)

        async def await_and_record(output, req_id: int):
            await output.aresult()
            completion_order.append(req_id)

        await asyncio.wait_for(
            asyncio.gather(
                await_and_record(out1, 1),
                await_and_record(out2, 2),
                await_and_record(out3, 3),
            ),
            timeout=55,
        )

    asyncio.run(run(llm))
    del llm

    assert 2 in completion_order and 3 in completion_order, (
        f"Not all requests completed. Completion order: {completion_order}"
    )
    idx_req2 = completion_order.index(2)
    idx_req3 = completion_order.index(3)
    assert idx_req3 < idx_req2, (
        f"Request 3 (priority=0.9) should complete before Request 2 (no priority). "
        f"Completion order: {completion_order}"
    )


@pytest.mark.threadleak(enabled=False)
@skip_ray
@pytest.mark.asyncio
@pytest.mark.parametrize("transceiver_runtime", ["PYTHON"])
async def test_llm_disagg_streaming_gen_cancelled(transceiver_runtime):
    tp_size = 1
    use_overlap = False
    enable_iter_req_stats = False

    llm_args_extra = {}

    llm_args_extra.update(
        dict(
            enable_iter_perf_stats=True,
            enable_iter_req_stats=enable_iter_req_stats,
            disable_overlap_scheduler=not use_overlap,
        )
    )

    backend = "NIXL"
    llm_ctx = LLM(
        model=llama_model_path,
        kv_cache_config=global_kvcache_config_no_reuse,
        tensor_parallel_size=tp_size,
        cache_transceiver_config=CacheTransceiverConfig(
            backend=backend, kv_transfer_timeout_ms=1000, transceiver_runtime=transceiver_runtime
        ),
        **llm_args_extra,
    )

    llm_gen = LLM(
        model=llama_model_path,
        kv_cache_config=global_kvcache_config_no_reuse,
        tensor_parallel_size=tp_size,
        cache_transceiver_config=CacheTransceiverConfig(
            backend=backend, kv_transfer_timeout_ms=1000, transceiver_runtime=transceiver_runtime
        ),
        **llm_args_extra,
    )

    try:
        num_iterations = 10
        num_concurrent_requests = 20
        prev_after_free_num_blocks = 0
        for iter in range(num_iterations):
            max_tokens = 1
            sampling_params = SamplingParams(max_tokens=max_tokens)
            disaggregated_params = DisaggregatedParams(request_type="context_only")

            prompts = [
                "lorem ipsum dolor sit amet consectetur adipiscing elit "
                "sed do eiusmod tempor incididunt ut labore et dolore magna aliqua "
                * random.randint(5, 15)
                for _ in range(num_concurrent_requests)
            ]
            # Send context-only requests
            ctx_outputs = []
            for prompt in prompts:
                for output in llm_ctx.generate(
                    [prompt],
                    sampling_params=sampling_params,
                    disaggregated_params=disaggregated_params,
                ):
                    ctx_outputs.append(output)

            assert len(ctx_outputs) == num_concurrent_requests

            max_tokens = 300
            sampling_params = SamplingParams(max_tokens=max_tokens, ignore_eos=True)

            # Send multiple gen-only requests concurrently
            async def process_request(idx):
                disagg_params = ctx_outputs[idx].disaggregated_params
                disagg_params.request_type = "generation_only"

                tokens_out = 0
                # Ensure we always cancel early to avoid race conditions
                stop_after_tokens = random.randint(10, 50)
                finished_reason = None
                async for gen_output in llm_gen.generate_async(
                    prompts[idx],
                    sampling_params=sampling_params,
                    disaggregated_params=disagg_params,
                    streaming=True,
                ):
                    tokens_out += 1
                    finished_reason = gen_output.outputs[0].finish_reason
                    if tokens_out == stop_after_tokens:
                        gen_output.abort()

                return finished_reason, stop_after_tokens

            # Launch all requests concurrently
            import asyncio

            results = await asyncio.gather(
                *[process_request(i) for i in range(num_concurrent_requests)]
            )

            # Verify all requests were cancelled
            for finished_reason, stop_after_tokens in results:
                assert finished_reason is not None
                assert finished_reason == "cancelled"

            # Check that the number of free/used blocks is as expected
            time.sleep(1.0)
            max_retries = 10
            for _ in range(max_retries):
                stats_results = llm_gen.get_stats(2)
                print("len(stats_results):", len(stats_results))
                if len(stats_results) > 0:
                    break
                time.sleep(1)
            else:
                pytest.fail(f"Failed to get stats after {max_retries} retries")

            after_used_num_blocks = stats_results[-1]["kvCacheStats"]["usedNumBlocks"]
            assert after_used_num_blocks == 0

            after_free_num_blocks = stats_results[-1]["kvCacheStats"]["freeNumBlocks"]
            # Check that number of free blocks stays the same
            if iter > 0:
                assert after_free_num_blocks == prev_after_free_num_blocks

            # Check that number of free blocks stays the same
            prev_after_free_num_blocks = after_free_num_blocks
    finally:
        llm_ctx.shutdown()
        llm_gen.shutdown()
