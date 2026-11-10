# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""TLLM_CUDA_GRAPH_SKIP_WARMUP_PASS=1 on a real model: same tokens, one capture pass, safe fallback."""

import datetime
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import pytest
import torch
from utils.llm_data import llm_models_root

from tensorrt_llm import LLM, SamplingParams
from tensorrt_llm._torch.moe.fused_moe import moe_comm_timeout_guard
from tensorrt_llm._torch.moe.fused_moe.moe_comm_timeout_guard import MoECommTimeoutGuard
from tensorrt_llm._torch.pyexecutor.cuda_graph_runner import CUDAGraphRunner, KeyType
from tensorrt_llm._torch.pyexecutor.engine.runners.decoder.graphs import DecoderCudaGraphs
from tensorrt_llm._torch.pyexecutor.executor_request_queue import (
    ExecutorRequestQueue,
    RequestQueueItem,
)
from tensorrt_llm._torch.pyexecutor.resource_manager import ResourceManager
from tensorrt_llm.executor.result import Logprob
from tensorrt_llm.llmapi import CudaGraphConfig, KvCacheConfig

# A flip needs a near tie: top two logprobs within two BF16 logit steps (1/8 at logits 16-32).
NEAR_TIE = 0.3
# The generated token's logprob may move by two BF16 logit steps between the runs.
LOGPROB_TOLERANCE = 0.3

PROMPTS = [
    "The capital of France is",
    "The president of the United States is",
    "Machine learning is",
    "The future of AI is",
    "What is the capital of Australia?",
    "Explain in one sentence why the sky is blue.",
    "Who wrote the book 'Pride and Prejudice'?",
    "List three U.S. national holidays.",
]


@dataclass(frozen=True)
class _Run:
    """What one LLM generated, and how its executors captured and replayed CUDA graphs."""

    tokens: list[list[int]]
    # Per prompt and generated token, the top two logprobs and the generated token's.
    logprobs: list[list[dict[int, Logprob]]]
    keys: set[KeyType]
    # The graphs the generation replayed.
    replayed: set[KeyType]
    # Per executor, the is_warmup_only flag of each CUDA graph pass.
    passes: list[list[bool]]
    # Per executor, the (stream capturing, MoE warmup timeouts) pairs its capture forwards saw.
    capture_timeouts: list[set[tuple[bool, bool]]]


def _generate(skip_warmup_pass: bool, move_workspace: bool = False) -> _Run:
    """Generate greedily; ``move_workspace`` regrows the shared workspace before the second
    capture of the single pass, as a smaller shape needing more workspace would."""
    passes: list[list[bool]] = []
    capture_timeouts: list[set[tuple[bool, bool]]] = []
    replayed: list[KeyType] = []
    moved: list[bool] = []
    generating: list[bool] = []
    # The number of requests the executor's first fetch while generating returned.
    first_fetch: list[int] = []
    guard = MoECommTimeoutGuard(environ={})
    warmup_and_capture = DecoderCudaGraphs.warmup_and_capture
    run_cuda_graph_warmup = DecoderCudaGraphs.run_cuda_graph_warmup
    capture = CUDAGraphRunner.capture
    replay = CUDAGraphRunner.replay
    get_from_request_queue = ExecutorRequestQueue.get_from_request_queue

    def record_executor(
        self: DecoderCudaGraphs, resource_manager: ResourceManager, **kwargs: bool
    ) -> None:
        """Start the records of one executor's CUDA graph capture."""
        passes.append([])
        capture_timeouts.append(set())
        warmup_and_capture(self, resource_manager, **kwargs)

    def record_pass(
        self: DecoderCudaGraphs, resource_manager: ResourceManager, **kwargs: bool
    ) -> None:
        """Record one pass; only the single pass leaves its prefill graphs to the caller."""
        passes[-1].append(self.cuda_graph_runner.is_warmup_only)
        if move_workspace and kwargs.get("capture_prefill") is False:
            moved.append(False)
        run_cuda_graph_warmup(self, resource_manager, **kwargs)

    def record_capture(
        self: CUDAGraphRunner,
        key: KeyType,
        forward_fn: Callable[[dict[str, Any]], Any],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Record the timeouts each capture forward sees, and move the workspace if asked."""
        if moved == [False] and self.graphs:
            captured = next(iter(self.graph_metadata.values()))["attn_metadata"]
            workspace = captured.cuda_graph_workspace
            workspace.resize_(workspace.numel() + (1 << 20))
            moved[0] = True

        def observed_forward(inputs: dict[str, Any]) -> Any:
            """Record whether this forward is captured and which MoE timeouts it runs with."""
            capture_timeouts[-1].add((torch.cuda.is_current_stream_capturing(), guard.in_warmup))
            return forward_fn(inputs)

        return capture(self, key, observed_forward, *args, **kwargs)

    def record_replay(self: CUDAGraphRunner, key: KeyType, *args: Any, **kwargs: Any) -> Any:
        """Record the graphs the generation replays."""
        if generating:
            replayed.append(key)
        return replay(self, key, *args, **kwargs)

    def fetch_prompts_together(
        self: ExecutorRequestQueue, timeout: datetime.timedelta | None
    ) -> list[RequestQueueItem]:
        """Schedule every prompt in the first batch, so each run steps through the same batches."""
        items = get_from_request_queue(self, timeout)
        if items and generating and not first_fetch:
            while len(items) < len(PROMPTS) and (
                more := get_from_request_queue(self, datetime.timedelta(seconds=10))
            ):
                items += more
            first_fetch.append(len(items))
        return items

    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("TLLM_WORKER_USE_SINGLE_PROCESS", "1")
        patch.delenv("TRTLLM_SKIP_KV_CACHE_ESTIMATION", raising=False)
        if skip_warmup_pass:
            patch.setenv("TLLM_CUDA_GRAPH_SKIP_WARMUP_PASS", "1")
        else:
            patch.delenv("TLLM_CUDA_GRAPH_SKIP_WARMUP_PASS", raising=False)
        patch.setattr(moe_comm_timeout_guard, "_DEFAULT_GUARD", guard)
        patch.setattr(DecoderCudaGraphs, "warmup_and_capture", record_executor)
        patch.setattr(DecoderCudaGraphs, "run_cuda_graph_warmup", record_pass)
        patch.setattr(CUDAGraphRunner, "capture", record_capture)
        patch.setattr(CUDAGraphRunner, "replay", record_replay)
        patch.setattr(ExecutorRequestQueue, "get_from_request_queue", fetch_prompts_together)
        try:
            with LLM(
                model=str(llm_models_root() / "llama-models-v2" / "TinyLlama-1.1B-Chat-v1.0"),
                max_batch_size=8,
                cuda_graph_config=CudaGraphConfig(batch_sizes=[1, 2, 4, 8], enable_padding=True),
                kv_cache_config=KvCacheConfig(free_gpu_memory_fraction=0.3),
            ) as llm:
                generating.append(True)
                # Requests finish one by one, so the decode batch replays every graph size.
                outputs = llm.generate(
                    PROMPTS,
                    [
                        SamplingParams(
                            max_tokens=8 + 3 * i, temperature=0, ignore_eos=True, logprobs=2
                        )
                        for i in range(len(PROMPTS))
                    ],
                )
                keys = set(llm._executor.engine.model_engine.cuda_graph_runner.graphs)
        finally:
            # The in-process executor lowers the PyTorch memory fraction for the whole process.
            torch.cuda.set_per_process_memory_fraction(1.0)
    assert moved == ([True] if move_workspace else [])
    assert first_fetch == [len(PROMPTS)]
    return _Run(
        tokens=[list(output.outputs[0].token_ids) for output in outputs],
        logprobs=[output.outputs[0].logprobs for output in outputs],
        keys=keys,
        replayed=set(replayed),
        passes=passes,
        capture_timeouts=capture_timeouts,
    )


def _assert_same_generation(run: _Run, two_pass_run: _Run) -> None:
    """The tokens match the two-pass run until one flips to the baseline's runner-up at a near
    tie; that prompt is compared no further."""
    for tokens, logprobs, ref_tokens, ref_logprobs in zip(
        run.tokens, run.logprobs, two_pass_run.tokens, two_pass_run.logprobs, strict=True
    ):
        assert len(tokens) == len(ref_tokens)
        for token, step, ref_token, ref_step in zip(
            tokens, logprobs, ref_tokens, ref_logprobs, strict=True
        ):
            if token != ref_token:
                runner_up = max(
                    (c for c in ref_step if c != ref_token), key=lambda c: ref_step[c].logprob
                )
                assert token == runner_up
                assert ref_step[ref_token].logprob - ref_step[runner_up].logprob < NEAR_TIE
                break
            assert abs(step[token].logprob - ref_step[token].logprob) < LOGPROB_TOLERANCE


@pytest.fixture(scope="module")
def two_pass_run() -> _Run:
    """The default two-pass capture that the single-pass runs are compared with."""
    run = _generate(skip_warmup_pass=False)
    # The KV-cache estimation executor and the final one.
    assert run.passes == [[True, False], [True, False]]
    return run


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_single_pass_capture_matches_two_passes(two_pass_run: _Run) -> None:
    """The final executor captures in one pass and generates the same tokens from the same graphs."""
    run = _generate(skip_warmup_pass=True)
    # The KV-cache estimation executor keeps both passes.
    assert run.passes == [[True, False], [False]]
    assert run.keys == two_pass_run.keys
    assert run.replayed == run.keys
    _assert_same_generation(run, two_pass_run)
    # Eager forwards keep the warmup MoE timeouts; each capture switches to the serving ones.
    assert run.capture_timeouts[-1] == {(False, True), (True, False)}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_single_pass_capture_falls_back_when_a_buffer_moves(two_pass_run: _Run) -> None:
    """A metadata tensor moved after an earlier capture sends the final executor back to two passes."""
    run = _generate(skip_warmup_pass=True, move_workspace=True)
    assert run.passes == [[True, False], [False, True, False]]
    assert run.keys == two_pass_run.keys
    assert run.replayed == run.keys
    _assert_same_generation(run, two_pass_run)
