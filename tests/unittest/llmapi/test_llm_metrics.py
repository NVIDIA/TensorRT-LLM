# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json

import pytest
from utils.util import skip_ray

from tensorrt_llm import LLM
from tensorrt_llm.llmapi import KvCacheConfig
from tensorrt_llm.metrics import MetricNames
from tensorrt_llm.sampling_params import SamplingParams

from .llm_stats_test_utils import llm_get_stats_async_test_harness, llm_get_stats_test_harness
from .llm_test_utils import global_kvcache_config, prompts
from .model_paths import get_model_path, llama_model_path


@pytest.mark.threadleak(enabled=False)
def test_llm_get_queued_stats():
    enable_iter_req_stats = True
    use_overlap = False
    tp_size = 1

    num_requests = 10
    repeated_prompts = ["A B C D E F G H I J K L M"] * num_requests

    llm_args_extra = {}
    sampling_args_extra = {}

    llm_args_extra.update(
        dict(
            enable_iter_perf_stats=True,
            enable_iter_req_stats=enable_iter_req_stats,
            disable_overlap_scheduler=not use_overlap,
        )
    )

    llm = LLM(
        model=llama_model_path,
        kv_cache_config=global_kvcache_config,
        tensor_parallel_size=tp_size,
        max_batch_size=1,
        **llm_args_extra,
    )

    max_tokens = 10
    sampling_params = SamplingParams(max_tokens=max_tokens, **sampling_args_extra)

    max_tries = 10
    has_queue_requests = False

    while not has_queue_requests and max_tries > 0:
        max_tries -= 1
        # Generate outputs, which will queue requests
        for output in llm.generate(repeated_prompts, sampling_params=sampling_params):
            print(output)

        results = llm.get_stats(2)

        for index, result in enumerate(results):
            if "requestStats" in result:
                for requestStat in result["requestStats"]:
                    if requestStat["stage"] == "QUEUED":
                        has_queue_requests = True
                        assert requestStat["numGeneratedTokens"] == 0

        if not has_queue_requests:
            print("No queued requests found, retrying...")
            asyncio.sleep(1)
        else:
            print("Found queued requests, breaking out of the loop.")

    assert has_queue_requests


@skip_ray
@pytest.mark.parametrize(
    "return_context_logits, use_overlap, enable_chunked_prefill, enable_iter_req_stats",
    [
        (False, False, False, True),
        (False, False, True, True),
        (False, True, False, True),
        (False, True, True, True),
    ],
)
def test_llm_get_stats(
    return_context_logits, use_overlap, enable_chunked_prefill, enable_iter_req_stats
):
    llm_get_stats_test_harness(
        tp_size=1,
        pp_size=1,
        return_context_logits=return_context_logits,
        pytorch_backend=True,
        use_overlap=use_overlap,
        enable_chunked_prefill=enable_chunked_prefill,
        enable_iter_req_stats=enable_iter_req_stats,
    )


@skip_ray
@pytest.mark.parametrize(
    "return_context_logits, use_overlap, enable_chunked_prefill, enable_iter_req_stats",
    [
        (False, False, False, True),
        (False, False, True, True),
        (False, True, False, True),
        (False, True, True, True),
    ],
)
def test_llm_get_stats_async(
    return_context_logits, use_overlap, enable_chunked_prefill, enable_iter_req_stats
):
    llm_get_stats_async_test_harness(
        tp_size=1,
        pp_size=1,
        return_context_logits=return_context_logits,
        pytorch_backend=True,
        use_overlap=use_overlap,
        enable_chunked_prefill=enable_chunked_prefill,
        enable_iter_req_stats=enable_iter_req_stats,
    )


@skip_ray
@pytest.mark.parametrize("use_overlap", [False, True])
def test_llm_get_stats_with_interval(use_overlap):
    """iter_perf_stats_interval samples records but keeps counter sums exact."""
    interval = 4
    stat_prompts = [
        "A B C",
        "Nvidia is awesome because",
        "The capital of France is",
        "Once upon a time",
    ] * 2
    with LLM(
        model=llama_model_path,
        kv_cache_config=global_kvcache_config,
        enable_iter_perf_stats=True,
        enable_iter_req_stats=True,
        iter_perf_stats_interval=interval,
        disable_overlap_scheduler=not use_overlap,
    ) as llm:
        # Different lengths so requests finish on different iterations.
        sampling_params = [
            SamplingParams(max_tokens=5 + 3 * i, end_id=-1) for i in range(len(stat_prompts))
        ]
        llm.generate(stat_prompts, sampling_params=sampling_params)

        records = []
        while True:
            batch = llm.get_stats(timeout=2)
            if not batch:
                break
            records.extend(json.loads(r) if isinstance(r, str) else r for r in batch)

    iter_records = [r for r in records if "iter" in r]
    assert iter_records, "expected sampled iteration stats records"
    # Every record is on a sampled iteration, except the single record
    # flushed when the last active request finishes between samples.
    off_interval = [r["iter"] for r in iter_records if r["iter"] % interval]
    assert len(off_interval) <= 1, off_interval
    if off_interval:
        assert off_interval[0] == max(r["iter"] for r in iter_records)
    # Counters from skipped iterations are folded into emitted records.
    assert sum(r["numCompletedRequests"] for r in iter_records) == len(stat_prompts)
    assert sum(r["numNewActiveRequests"] for r in iter_records) == len(stat_prompts)
    # Far fewer records than executor iterations: the sampled iterations
    # 0, interval, ..., plus the startup snapshot and the drain record.
    last_iter = max(r["iter"] for r in iter_records)
    assert len(iter_records) <= last_iter // interval + 3


@skip_ray
def test_llm_perf_metrics():
    with LLM(
        model=llama_model_path,
        kv_cache_config=global_kvcache_config.model_copy(update={"use_kv_cache_manager_v2": True}),
        return_perf_metrics=False,
    ) as llm:
        sampling_params = SamplingParams(max_tokens=10, return_perf_metrics=True)
        outputs = llm.generate(prompts, sampling_params)
        assert outputs[0].outputs[0].request_perf_metrics is not None

        perf_metrics = outputs[0].outputs[0].request_perf_metrics

        timing_metrics = perf_metrics.timing_metrics
        assert timing_metrics.arrival_time < timing_metrics.first_scheduled_time
        assert timing_metrics.first_scheduled_time < timing_metrics.first_token_time
        assert timing_metrics.first_token_time < timing_metrics.last_token_time

        kv_cache_metrics = perf_metrics.kv_cache_metrics
        assert kv_cache_metrics.num_total_allocated_blocks == 1
        assert kv_cache_metrics.num_new_allocated_blocks == 1
        assert kv_cache_metrics.num_reused_blocks == 0
        assert kv_cache_metrics.num_missed_blocks == 1
        assert kv_cache_metrics.kv_cache_hit_rate == 0

        assert perf_metrics.first_iter is not None
        assert perf_metrics.iter - perf_metrics.first_iter == sampling_params.max_tokens - 1
        assert perf_metrics.last_iter == perf_metrics.iter


@skip_ray
@pytest.mark.parametrize("attn_backend", ["TRTLLM", "FLASHINFER"])
def test_llm_prefix_cache_reuse(attn_backend):
    model_path = get_model_path("Qwen3/Qwen3-0.6B")
    prompt = "The future of AI is " * 20
    sampling_params = SamplingParams(temperature=0, max_tokens=5, return_perf_metrics=True)

    with LLM(
        model=model_path,
        attn_backend=attn_backend,
        kv_cache_config=KvCacheConfig(enable_block_reuse=True, use_kv_cache_manager_v2=True),
        cuda_graph_config=None,
        return_perf_metrics=False,
    ) as llm:
        cold_output = llm.generate(prompt, sampling_params).outputs[0]
        warm_output = llm.generate(prompt, sampling_params).outputs[0]

    cold_metrics = cold_output.request_perf_metrics
    warm_metrics = warm_output.request_perf_metrics
    assert cold_metrics is not None
    assert warm_metrics is not None
    assert cold_metrics.kv_cache_metrics.num_reused_blocks == 0
    assert warm_metrics.kv_cache_metrics.num_reused_blocks > 0
    assert cold_output.token_ids == warm_output.token_ids


@skip_ray
def test_llm_prometheus():
    test_prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]
    sampling_params = SamplingParams(max_tokens=10, temperature=0.8, top_p=0.95)
    llm = LLM(
        model=llama_model_path, return_perf_metrics=True, kv_cache_config=global_kvcache_config
    )
    for test_prompt in test_prompts:
        request_output = llm.generate(test_prompt, sampling_params)
        assert request_output.metrics_dict is not None
        assert MetricNames.REQUEST_QUEUE_TIME in request_output.metrics_dict
        assert MetricNames.TPOT in request_output.metrics_dict
        assert MetricNames.TTFT in request_output.metrics_dict
        assert MetricNames.E2E in request_output.metrics_dict
        assert request_output.outputs is not None
