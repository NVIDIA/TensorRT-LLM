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
"""Aggregated TP16/c1 accuracy and MTP coverage for the SM107 GLM target.

Run with 16 MPI ranks on Rubin; the pytest temporary directory must be visible
to every rank. A short GPU trace proves actual optimized decode and draft
execution after engine warmup, including CUDA graph replay. The full GSM8K
evaluation then runs without profiling.
"""

import json
import os
import time
from pathlib import Path

import pytest

from tensorrt_llm import LLM
from tensorrt_llm.evaluate import GSM8KInferenceX
from tensorrt_llm.llmapi import (
    CudaGraphConfig,
    DeepSeekSparseAttentionConfig,
    KvCacheConfig,
    MoeConfig,
    MTPDecodingConfig,
    SamplingParams,
)

from ...conftest import llm_models_root, skip_no_rubin
from ..accuracy_core import LlmapiAccuracyTestHarness, acceptance_length_from_iteration_stats

_TP_SIZE = 16
_NUM_SAMPLES = 1319
_MAX_DRAFT_LEN = 7
_ACCURACY_FLOOR = 90.0
# Full-corpus aggregated reference on SM107 (2026-10-20), with the same 95%
# retention used by the shared acceptance-length reference policy.
_AL_REFERENCE = 5.126054389634928
_AL_FLOOR = 0.95 * _AL_REFERENCE
_TASK_NAME = "gsm8k_inferencex"
_MODEL_PATH = f"{llm_models_root()}/GLM-5.2-NVFP4"
_DATASET_PATH = f"{llm_models_root()}/datasets/openai/gsm8k"
_RANK_ENV = {
    "TRTLLM_DSA_INDEXER_BF16": "1",
    "TLLM_SPEC_DECODE_FORCE_NUM_ACCEPTED_TOKENS": "0",
}


def _engine_kwargs() -> dict:
    speculative_config = MTPDecodingConfig(
        max_draft_len=_MAX_DRAFT_LEN,
        use_relaxed_acceptance_for_thinking=False,
    )
    # This is a Pydantic private attribute: pass the object, not model_dump().
    # The optimized MTP worker requires the target's cache instead of a separate
    # draft cache. Its CUDA kernels in the probe also verify this on the workers.
    speculative_config._allow_separate_draft_kv_cache = False
    return dict(
        tensor_parallel_size=_TP_SIZE,
        pipeline_parallel_size=1,
        moe_expert_parallel_size=1,
        enable_attention_dp=False,
        modeling_v2="require",
        max_batch_size=1,
        max_seq_len=16384,
        max_num_tokens=4096,
        enable_chunked_prefill=True,
        kv_cache_config=KvCacheConfig(
            dtype="fp8",
            tokens_per_block=64,
            enable_block_reuse=False,
            free_gpu_memory_fraction=0.8,
            max_tokens=32768,
            host_cache_size=0,
        ),
        cuda_graph_config=CudaGraphConfig(batch_sizes=[1], enable_padding=True),
        moe_config=MoeConfig(backend="TRTLLM", use_low_precision_moe_combine=True),
        sparse_attention_config=DeepSeekSparseAttentionConfig(
            enable_heuristic_topk=True,
            use_cute_dsl_paged_mqa_logits=False,
        ),
        speculative_config=speculative_config,
        enable_iter_perf_stats=True,
        iter_perf_stats_interval=1,
        max_stats_len=-1,
        env_overrides=dict(_RANK_ENV),
    )


def _assert_optimized_paths(llm: LLM, trace_dir: Path) -> None:
    """Observe GPU execution, which Python forward counters miss on graph replay."""
    trace_dir.mkdir(parents=True)
    llm.start_profile(
        output_dir=str(trace_dir),
        num_steps=4,
        start_step=1,
        activities=["GPU"],
    )
    try:
        output = llm.generate(
            "Explain step by step how to multiply two three-digit numbers.",
            SamplingParams(
                temperature=0.0,
                top_p=1.0,
                max_tokens=64,
                ignore_eos=True,
            ),
        )
        assert len(output.outputs[0].token_ids) == 64
    finally:
        llm.stop_profile()

    # These kernels are exclusive to the GLM optimized decode and MTP paths.
    # Requiring every TP rank prevents a route-selection or warmup-only pass.
    required_kernels = (
        "routingFusedRouterKernel",
        "draftEhprojTableKernel",
        "draftSampleGlueKernel",
    )
    traces = list(trace_dir.glob("trtllm-trace-*-rank-*.json"))
    assert len(traces) == _TP_SIZE, f"Expected {_TP_SIZE} GPU traces, found {traces}"
    for rank in range(_TP_SIZE):
        rank_traces = [path for path in traces if path.name.endswith(f"-rank-{rank}.json")]
        assert len(rank_traces) == 1, f"Missing or duplicate GPU trace for rank {rank}"
        trace = json.loads(rank_traces[0].read_text())
        names = [event["name"] for event in trace["traceEvents"] if event.get("cat") == "kernel"]
        for kernel in required_kernels:
            assert any(kernel in name for name in names), (
                f"Rank {rank} did not execute {kernel} after warmup; "
                "check optimized decode dispatch and shared-cache MTP dispatch"
            )


def _request_stats(llm: LLM, num_requests: int) -> list[dict]:
    # Responses can arrive before their final iteration statistics. Drain until
    # all requests are represented, including the probe before corpus scoring.
    stats = []
    completed = 0
    deadline = time.monotonic() + 10
    while completed < num_requests and time.monotonic() < deadline:
        batch = llm.get_stats(timeout=2)
        stats.extend(batch)
        completed += sum(stat["numCompletedRequests"] for stat in batch)
    assert completed == num_requests, f"Statistics cover {completed}/{num_requests} requests"
    return stats


def _full_accuracy_scores(result_path: Path) -> dict[str, float]:
    results = json.loads(result_path.read_text())
    counts = results["n-samples"][_TASK_NAME]
    assert counts["original"] == counts["effective"] == _NUM_SAMPLES, counts
    samples = results["samples"][_TASK_NAME]
    assert len({sample["doc_id"] for sample in samples}) == _NUM_SAMPLES
    scores = results["results"][_TASK_NAME]
    return {name: scores[f"exact_match,{name}"] for name in ("strict-match", "flexible-extract")}


class TestModelingV2Glm52Nvfp4Sm107Tp16(LlmapiAccuracyTestHarness):
    @skip_no_rubin
    @pytest.mark.skip_less_mpi_world_size(_TP_SIZE)
    def test_mtp7_gsm8k_and_acceptance(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
        # The generic accuracy harness's one-sample integration mode cannot
        # satisfy this full-corpus guard.
        assert os.environ.get("INTEGRATION_TEST") != "1", "This guard requires all 1319 samples"
        monkeypatch.setenv("TLLM_EVAL_MAX_IN_FLIGHT", "1")
        monkeypatch.setenv("TLLM_EVAL_PARTIAL_SCORES_EVERY", "100")
        for key, value in _RANK_ENV.items():
            monkeypatch.setenv(key, value)

        with LLM(_MODEL_PATH, **_engine_kwargs()) as llm:
            _assert_optimized_paths(llm, tmp_path / "optimized-path-traces")
            _request_stats(llm, 1)  # Exclude the coverage probe from corpus AL.
            evaluator = GSM8KInferenceX(
                dataset_path=_DATASET_PATH,
                num_samples=_NUM_SAMPLES,
                num_fewshot=5,
                random_seed=0,
                apply_chat_template=True,
                fewshot_as_multiturn=False,
                shuffle_dataset=False,
                fewshot_random_seed=1234,
                log_samples=True,
                output_path=str(tmp_path),
            )
            evaluator.evaluate(
                llm,
                sampling_params=SamplingParams(
                    temperature=0.0,
                    top_p=1.0,
                    max_tokens=12288,
                    truncate_prompt_tokens=4096,
                ),
                streaming=False,
                sampling_override=True,
                scores_filter="exact_match,strict-match",
            )
            stats = _request_stats(llm, _NUM_SAMPLES)

        spec_stats = [
            stat["specDecodingStats"]
            for stat in stats
            if stat.get("specDecodingStats") and stat["specDecodingStats"]["numDraftTokens"] > 0
        ]
        assert len(spec_stats) >= _NUM_SAMPLES, "Missing full-run speculative iteration statistics"
        assert all(stat["numRequestsWithDraftTokens"] == 1 for stat in spec_stats)
        acceptance_length = acceptance_length_from_iteration_stats(stats)
        scores = _full_accuracy_scores(tmp_path / f"samples_{_TASK_NAME}.json")
        metrics = dict(scores)
        metrics["acceptance_length"] = acceptance_length
        metrics["speculative_iterations"] = len(spec_stats)
        metrics["accepted_draft_tokens"] = sum(stat["numAcceptedTokens"] for stat in spec_stats)
        metrics["drafted_tokens"] = sum(stat["numDraftTokens"] for stat in spec_stats)
        metrics["speculative_request_steps"] = sum(
            stat["numRequestsWithDraftTokens"] for stat in spec_stats
        )
        metrics["completed_requests"] = sum(stat["numCompletedRequests"] for stat in stats)
        metrics["new_requests"] = sum(stat["numNewActiveRequests"] for stat in stats)
        (tmp_path / "glm_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
        print(f"[GLM TP16 c1] {json.dumps(metrics)}")
        for name, score in scores.items():
            assert _ACCURACY_FLOOR <= score <= 100.0, (
                f"{name}: {score:.3f}% is below the local {_ACCURACY_FLOOR}% floor"
            )
        assert _AL_FLOOR <= acceptance_length <= _MAX_DRAFT_LEN + 1, (
            f"MTP acceptance length {acceptance_length:.3f} is outside "
            f"[{_AL_FLOOR:.3f}, {_MAX_DRAFT_LEN + 1}]"
        )
