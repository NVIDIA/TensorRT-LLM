# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""End-to-end forward-error survival check on a real single-GPU engine.

Exercises the PyExecutor forward-error survival path (PR #19465) without
mocks: TLLM_TEST_INJECT_FORWARD_FAIL_STEPS makes `_forward_step` raise on
chosen non-warmup steps, and the test then requires that

1. every in-flight future receives a terminal response (error or result)
   -- a future that never resolves is exactly the CI hang signature
   (client blocked in GenerationResult.result -> _result_step), and
2. the engine keeps serving new requests after the survived failures.

Run on both executor loops: the CI hangs were seen with the non-overlap
loop (test_best_of_n) and the overlap loop (DeepSeekV3Lite chunked
prefill).

The injection variable is set ONLY in this test's MPI workers, via an
explicit ``MpiPoolSession(env_overrides=...)`` passed as ``_mpi_session``.
It is deliberately NOT set in the parent process environment: MPI workers
freeze the parent environment at spawn, and the test suite's automatic
session-reuse/prefetch layer keeps and pre-spawns worker pools across
tests in background threads. A parent-env variable set during this test
would be frozen into one of those shared pools and then injected into an
unrelated later test. Passing an external ``_mpi_session`` also bypasses
the reuse/prefetch seam entirely (the library only patches its own pool
construction), so this pool is private and destroyed here.

Not wired into any L0 test list; this is a diagnostic for GPU runs.
"""

import os

import pytest
from utils.llm_data import llm_models_root

from tensorrt_llm import LLM, SamplingParams
from tensorrt_llm._utils import mpi_disabled
from tensorrt_llm.executor.utils import RequestError
from tensorrt_llm.llmapi.llm_utils import KvCacheConfig
from tensorrt_llm.llmapi.mpi_session import MpiPoolSession

PROMPTS = [
    "Born in north-east France, Soyer trained as a",
    "The future of AI is",
]

# Per-future wait. Generous for TinyLlama; a hit means the terminal
# response was lost, not that the engine is slow.
FUTURE_TIMEOUT_S = 180


@pytest.mark.skipif(mpi_disabled(), reason="needs the MPI worker path to inject via env_overrides")
@pytest.mark.parametrize("overlap", [False, True], ids=["no_overlap", "overlap"])
@pytest.mark.threadleak(enabled=False)
def test_forward_error_survival_e2e(overlap):
    # Fail the 3rd and 9th non-warmup forward steps. The variable is frozen
    # into THIS pool's workers only (env_overrides); the parent process
    # environment is never touched, so no shared/reused worker pool can pick
    # it up. wait_shutdown so the workers exit before the next test.
    mpi_session = MpiPoolSession(
        n_workers=1,
        wait_shutdown=True,
        env_overrides={"TLLM_TEST_INJECT_FORWARD_FAIL_STEPS": "3,9"},
    )

    try:
        # Passing _mpi_session makes the LLM use this external pool and NOT
        # own it, so LLM shutdown leaves it alive; this test shuts it down in
        # the finally below.
        llm = LLM(
            model=os.path.join(llm_models_root(), "llama-models-v2", "TinyLlama-1.1B-Chat-v1.0"),
            kv_cache_config=KvCacheConfig(max_tokens=1000),
            max_batch_size=8,
            max_seq_len=64,
            disable_overlap_scheduler=not overlap,
            _mpi_session=mpi_session,
        )

        with llm:
            # Mirror test_best_of_n.py::test_async_n_outputs: 10 async
            # requests with n=3 (child requests), exceeding max_batch_size.
            sampling_params = SamplingParams(n=3, temperature=0.8, top_p=0.95)
            futures = []
            for _ in range(5):
                for prompt in PROMPTS:
                    futures.append(llm.generate_async(prompt, sampling_params))

            hung, errored, completed = [], 0, 0
            for idx, future in enumerate(futures):
                try:
                    future.result(timeout=FUTURE_TIMEOUT_S)
                    completed += 1
                except RequestError:
                    # Expected for requests in flight when the injected
                    # forward failure hit: batch-wide error responses.
                    errored += 1
                except TimeoutError:
                    hung.append(idx)

            assert not hung, (
                f"futures {hung} never received a terminal response within "
                f"{FUTURE_TIMEOUT_S}s (completed={completed}, "
                f"errored={errored}): the forward-error survival path lost "
                "responses"
            )
            assert errored >= 1, (
                f"fault injection never fired (completed={completed}); "
                "TLLM_TEST_INJECT_FORWARD_FAIL_STEPS did not reach the worker"
            )

            # The engine must still serve after the survived failures.
            post = llm.generate(PROMPTS, SamplingParams(max_tokens=8, temperature=0.0))
            for output in post:
                assert output.outputs[0].token_ids, (
                    "engine stopped producing tokens after a survived forward failure"
                )
    finally:
        mpi_session.shutdown()
