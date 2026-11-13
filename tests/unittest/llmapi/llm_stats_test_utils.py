# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import contextlib
import time

import pytest
from utils.util import altered_env

from tensorrt_llm import LLM
from tensorrt_llm.sampling_params import SamplingParams

from .llm_test_utils import global_kvcache_config, prompts
from .model_paths import llama_model_path


def validate_stats(
    *,
    results,
    pytorch_backend,
    max_tokens,
    prompt_len,
    pp_size=1,
    use_overlap=False,
    enable_chunked_prefill=False,
    enable_iter_req_stats=False,
):
    assert results
    for iter, result in enumerate(results):
        ifbStats = result["inflightBatchingStats"]
        print(f"iter: {iter}, ifbStats: {ifbStats}")

    # Filter out the results where no requests are scheduled
    results = [r for r in results if r["inflightBatchingStats"]["numScheduledRequests"] > 0]

    context_iterations = 2 if enable_chunked_prefill else 1
    generation_iterations = max_tokens - 1
    assert len(results) == context_iterations + generation_iterations

    microbatch_id = 0
    for iter, result in enumerate(results):
        ifbStats = result["inflightBatchingStats"]

        if iter < context_iterations:
            assert ifbStats["numScheduledRequests"] == 1, f"iter: {iter}"
            assert ifbStats["numContextRequests"] == 1, f"iter: {iter}"
            assert ifbStats["numGenRequests"] == 0, f"iter: {iter}"
            assert result["numActiveRequests"] == 1, f"iter: {iter}"
            assert ifbStats["microBatchId"] == microbatch_id, f"iter: {iter}"
        elif iter < (context_iterations + generation_iterations):
            assert ifbStats["numScheduledRequests"] == 1, f"iter: {iter}"
            assert ifbStats["numContextRequests"] == 0, f"iter: {iter}"
            assert ifbStats["numGenRequests"] == 1, f"iter: {iter}"
            assert result["numActiveRequests"] == 1, f"iter: {iter}"
            assert ifbStats["microBatchId"] == microbatch_id, f"iter: {iter}"

        # In pipeline parallel mode, increment microbatch_id for each context iteration except the last one,
        # since the context chunks can be scheduled in each iteration.
        if pp_size > 1 and iter < context_iterations - 1:
            microbatch_id += 1

        if enable_iter_req_stats:
            assert "requestStats" in result, f"iter: {iter}"
            req_stats = result["requestStats"]
            assert len(req_stats) == 1, f"iter: {iter}"
            req_stat = req_stats[0]
            if iter < (context_iterations - 1):
                # If use_overlap, the stats are one iteration ahead
                assert (
                    req_stat["stage"] == "GENERATION_IN_PROGRESS"
                    if use_overlap
                    else "CONTEXT_IN_PROGRESS"
                ), f"iter: {iter}"
                assert req_stat["contextPrefillPosition"] == prompt_len if use_overlap else 32, (
                    f"iter: {iter}"
                )
                assert req_stat["numGeneratedTokens"] == 0, f"iter: {iter}"
            elif iter < (context_iterations - 1 + generation_iterations):
                assert req_stat["stage"] == "GENERATION_IN_PROGRESS", f"iter: {iter}"
                assert req_stat["contextPrefillPosition"] == prompt_len, f"iter: {iter}"
                assert req_stat["numGeneratedTokens"] == iter - (context_iterations - 1) + 1, (
                    f"iter: {iter}"
                )
            else:
                assert req_stat["stage"] == "GENERATION_COMPLETE", f"iter: {iter}"
                assert req_stat["contextPrefillPosition"] == prompt_len, f"iter: {iter}"
                assert req_stat["numGeneratedTokens"] == max_tokens, f"iter: {iter}"
            assert req_stat["scheduled"], f"iter: {iter}"

        expected_num_completed = 1 if iter == len(results) - 1 else 0

        # TODO: For some reason, with stats_async and TRT backend, numCompleted is 0 at first iteration
        if pytorch_backend:
            assert result["numCompletedRequests"] == expected_num_completed

            # Per-iteration request-aggregate fields populated by
            # PyExecutor._update_iter_stats inside inflightBatchingStats.
            # Assert presence (a missing key indicates a serializer or
            # RPC-path regression) and sane per-iteration values (a
            # zero-under-load value indicates a mis-wired populate block).
            new_aggregate_keys = (
                "numCtxKvTokens",
                "numGenKvTokens",
                "numQueuedContextRequests",
                "numQueuedCtxTokens",
                "numQueuedGenRequests",
                "numQueuedGenKvTokens",
                "numPausedKvTokens",
            )
            for k in new_aggregate_keys:
                assert k in ifbStats, f"iter {iter}: missing ifbStats key {k}"
                assert isinstance(ifbStats[k], int), (
                    f"iter {iter}: ifbStats key {k} not int (got {type(ifbStats[k])})"
                )
                assert ifbStats[k] >= 0, f"iter {iter}: ifbStats key {k} negative"

            if iter < context_iterations:
                # Prefill iteration: at least one scheduled context request
                # and nonzero numCtxTokens. numCtxTokens is sourced from
                # model_engine.iter_states after _forward_step for this
                # batch, so it is overlap-safe under every scheduler
                # configuration.
                assert ifbStats["numContextRequests"] >= 1, f"iter: {iter}"
                assert ifbStats["numGenRequests"] == 0, f"iter: {iter}"
                assert ifbStats["numCtxTokens"] > 0, f"iter: {iter}"
            else:
                # Generation iteration: at least one decode request with
                # nonzero total KV context length.
                assert ifbStats["numGenRequests"] >= 1, f"iter: {iter}"
                assert ifbStats["numGenKvTokens"] > 0, f"iter: {iter}"
                assert ifbStats["numContextRequests"] == 0, f"iter: {iter}"


def llm_get_stats_test_harness(
    tp_size: int = 1,
    pp_size: int = 1,
    return_context_logits: bool = False,
    pytorch_backend: bool = False,
    use_overlap: bool = False,
    enable_chunked_prefill: bool = False,
    enable_iter_req_stats: bool = False,
):
    if return_context_logits and pytorch_backend:
        pytest.skip("pytorch backend does not support context logits")

    if enable_iter_req_stats and not pytorch_backend:
        pytest.skip("enable_iter_req_stats not supported yet without pytorch backend")

    print("-------------")
    print("return_context_logits: ", return_context_logits)
    print("pytorch_backend: ", pytorch_backend)
    print("use_overlap: ", use_overlap)
    print("enable_chunked_prefill: ", enable_chunked_prefill)
    print("enable_iter_req_stats: ", enable_iter_req_stats)
    print("-------------")

    llm_args_extra = {}
    sampling_args_extra = {}
    if return_context_logits:
        llm_args_extra["gather_generation_logits"] = True
        sampling_args_extra["return_context_logits"] = True

    if enable_chunked_prefill:
        llm_args_extra["enable_chunked_prefill"] = True
        llm_args_extra["max_num_tokens"] = 32

    if pytorch_backend:
        llm_args_extra.update(
            dict(
                enable_iter_perf_stats=True,
                enable_iter_req_stats=enable_iter_req_stats,
                disable_overlap_scheduler=not use_overlap,
            )
        )

    # Since we need to check pp's internal states, we disable the async broadcast
    # to get a deterministic behavior.
    env_ctx = (
        altered_env(TLLM_PP_ASYNC_BROADCAST_SAMPLE_STATE="0")
        if pp_size > 1
        else contextlib.nullcontext()
    )

    with (
        env_ctx,
        LLM(
            model=llama_model_path,
            kv_cache_config=global_kvcache_config,
            tensor_parallel_size=tp_size,
            pipeline_parallel_size=pp_size,
            **llm_args_extra,
        ) as llm,
    ):
        max_tokens = 5
        sampling_params = SamplingParams(max_tokens=max_tokens, **sampling_args_extra)

        long_prompts = ["A B C D E F G H I J K L M N O P Q R S T U V W X Y Z " * 2]

        prompt_len = None
        for output in llm.generate(long_prompts, sampling_params=sampling_params):
            print(output)
            prompt_len = len(output.prompt_token_ids)

        time.sleep(2)
        results = llm.get_stats(2)

        validate_stats(
            results=results,
            pp_size=pp_size,
            pytorch_backend=pytorch_backend,
            prompt_len=prompt_len,
            max_tokens=max_tokens,
            use_overlap=use_overlap,
            enable_chunked_prefill=enable_chunked_prefill,
            enable_iter_req_stats=enable_iter_req_stats,
        )

        assert not llm.get_stats(0.5)

        # test that IterationResult()._done is properly set
        _ = llm.generate(prompts, sampling_params=sampling_params)
        assert llm.get_stats(2)


def llm_get_stats_async_test_harness(
    tp_size: int = 1,
    pp_size: int = 1,
    return_context_logits: bool = False,
    pytorch_backend: bool = False,
    use_overlap: bool = False,
    enable_chunked_prefill: bool = False,
    enable_iter_req_stats: bool = False,
):
    if return_context_logits and pytorch_backend:
        pytest.skip("pytorch backend does not support context logits")

    if enable_iter_req_stats and not pytorch_backend:
        pytest.skip("enable_iter_req_stats not supported yet without pytorch backend")

    print("-------------")
    print("return_context_logits: ", return_context_logits)
    print("pytorch_backend: ", pytorch_backend)
    print("use_overlap: ", use_overlap)
    print("enable_chunked_prefill: ", enable_chunked_prefill)
    print("enable_iter_req_stats: ", enable_iter_req_stats)
    print("-------------")

    llm_args_extra = {}
    sampling_args_extra = {}
    if return_context_logits:
        sampling_args_extra["return_context_logits"] = True

    if enable_chunked_prefill:
        llm_args_extra["enable_chunked_prefill"] = True
        llm_args_extra["max_num_tokens"] = 32

    if pytorch_backend:
        llm_args_extra.update(
            dict(
                enable_iter_perf_stats=True,
                enable_iter_req_stats=enable_iter_req_stats,
                disable_overlap_scheduler=not use_overlap,
            )
        )

    with LLM(
        model=llama_model_path,
        kv_cache_config=global_kvcache_config,
        tensor_parallel_size=tp_size,
        pipeline_parallel_size=pp_size,
        **llm_args_extra,
    ) as llm:
        max_tokens = 6
        sampling_params = SamplingParams(max_tokens=max_tokens, **sampling_args_extra)

        long_prompts = ["A B C D E F G H I J K L M N O P Q R S T U V W X Y Z " * 2]

        prompt_len = None

        async def task0():
            nonlocal prompt_len
            async for output in llm.generate_async(
                long_prompts[0], streaming=True, sampling_params=sampling_params
            ):
                print(output)
                prompt_len = len(output.prompt_token_ids)

        async def task1(repetition_index: int):
            results = []
            await asyncio.sleep(4)  # ensure there's stats to collect for the assertion
            async for stats in llm.get_stats_async(10):  # it will return immediately
                results.append(stats)

            assert results
            if not use_overlap:
                validate_stats(
                    results=results,
                    pp_size=pp_size,
                    pytorch_backend=pytorch_backend,
                    max_tokens=max_tokens,
                    prompt_len=prompt_len,
                    use_overlap=use_overlap,
                    # After the first repetition, context will be reused and there will be no chunking.
                    enable_chunked_prefill=enable_chunked_prefill
                    if repetition_index == 0
                    else False,
                    enable_iter_req_stats=enable_iter_req_stats,
                )

        async def main():
            for repetition_index in range(2):  # test recurrent usage
                await asyncio.gather(task0(), task1(repetition_index))

        asyncio.run(main())
