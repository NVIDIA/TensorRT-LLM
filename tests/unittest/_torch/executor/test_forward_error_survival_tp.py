# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Rank-local forward-failure survival under plain TP, on a real TP=2 engine.

Companion to test_forward_error_survival_e2e.py. There the injected failure
raises on every rank symmetrically; here TLLM_TEST_INJECT_FORWARD_FAIL_RANKS
makes only rank 1 raise, after its model forward returns (so no TP peer is
left inside the forward's collectives). Without TP-wide agreement on the
failure, the ranks diverge: rank 1 fails and frees every active request while
rank 0 keeps scheduling them and broadcasting the schedule, and rank 1's error
responses never reach the client (only rank 0 enqueues responses under
non-ADP TP). The observable signatures are futures that never resolve and/or a
dead worker group.

With the agreement collective in the executor loops, every rank joins the
failure: the client receives error responses for the in-flight requests (from
rank 0) and the engine keeps serving.

The injection variables are set only in this test's MPI workers via
``MpiPoolSession(env_overrides=...)``; see the companion test's docstring for
why the parent environment must stay untouched.

No L0 list names this file, but several stages collect the whole
unittest/_torch/executor directory: it runs on the multi-GPU stages
(e.g. DGX B300, GB300) and skips on single-GPU stages via the 2-GPU
guard below.
"""

import os
import time

import pytest
import torch
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

# Shared deadline for collecting every future. Generous for TinyLlama TP=2;
# hitting it means terminal responses were lost (the pre-fix desync
# signature), not that the engine is slow. Shared, so a hung engine costs one
# deadline rather than one per future.
COLLECT_DEADLINE_S = 240


@pytest.mark.skipif(mpi_disabled(), reason="needs the MPI worker path to inject via env_overrides")
# Single-GPU L0 stages collect this directory with markexpr "not cpu_only",
# which does not exclude gpu2. The gpu2 marker itself does skip there, but
# only via the pytest.mark.gpu2 monkeypatch in tests/unittest/utils/util.py
# (import-order dependent); this explicit guard does not rely on it.
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs 2 GPUs")
@pytest.mark.gpu2
@pytest.mark.parametrize("overlap", [False, True], ids=["no_overlap", "overlap"])
@pytest.mark.threadleak(enabled=False)
def test_forward_error_survival_tp_rank_local(overlap):
    # Rank 1 (never the response-owning rank 0) fails its 3rd non-warmup
    # forward step, after the forward returns. Both variables are frozen into
    # THIS pool's two workers only.
    mpi_session = MpiPoolSession(
        n_workers=2,
        wait_shutdown=True,
        env_overrides={
            "TLLM_TEST_INJECT_FORWARD_FAIL_STEPS": "3",
            "TLLM_TEST_INJECT_FORWARD_FAIL_RANKS": "1",
        },
    )

    try:
        llm = LLM(
            model=os.path.join(llm_models_root(), "llama-models-v2", "TinyLlama-1.1B-Chat-v1.0"),
            tensor_parallel_size=2,
            kv_cache_config=KvCacheConfig(max_tokens=1000),
            max_batch_size=8,
            max_seq_len=64,
            disable_overlap_scheduler=not overlap,
            _mpi_session=mpi_session,
        )

        with llm:
            sampling_params = SamplingParams(n=3, temperature=0.8, top_p=0.95)
            futures = []
            for _ in range(5):
                for prompt in PROMPTS:
                    futures.append(llm.generate_async(prompt, sampling_params))

            hung, errored, completed = [], 0, 0
            deadline = time.monotonic() + COLLECT_DEADLINE_S
            for idx, future in enumerate(futures):
                try:
                    future.result(timeout=max(1.0, deadline - time.monotonic()))
                    completed += 1
                except RequestError:
                    # Expected for requests in flight when rank 1's injected
                    # failure hit: the TP group agrees and rank 0 emits
                    # batch-wide error responses.
                    errored += 1
                except TimeoutError:
                    hung.append(idx)

            assert not hung, (
                f"futures {hung} never received a terminal response within "
                f"{COLLECT_DEADLINE_S}s (completed={completed}, "
                f"errored={errored}): a rank-local TP forward failure lost "
                "responses (ranks desynchronized)"
            )
            assert errored >= 1, (
                f"fault injection never fired (completed={completed}); "
                "TLLM_TEST_INJECT_FORWARD_FAIL_RANKS did not reach rank 1"
            )

            # The engine must still serve after the survived rank-local
            # failure -- and the TP group must still agree on schedules.
            post = llm.generate(PROMPTS, SamplingParams(max_tokens=8, temperature=0.0))
            for output in post:
                assert output.outputs[0].token_ids, (
                    "engine stopped producing tokens after a survived rank-local TP forward failure"
                )
    finally:
        mpi_session.shutdown()
