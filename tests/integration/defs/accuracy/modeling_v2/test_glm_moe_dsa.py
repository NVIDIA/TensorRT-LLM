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
"""Whole-model gates for the glm_moe_dsa modeling_v2 targets.

One file per model family, beside the other accuracy suites rather than inside
test_glm52.py: these gate a parallel implementation, and reading them next to
the built-in model's tests would invite treating one as a variant of the other.

Every test here passes ``modeling_v2="require"`` to ``LLM``. Under ``"auto"`` a
configuration that missed a target's criteria -- or fell outside its bounds --
would quietly fall back to the built-in implementation, pass, and report the
built-in's numbers as the target's.

``TestModelingV2Glm52Nvfp4Sm107Tp16``: aggregated TP16/c1 accuracy and MTP
coverage for the SM107 target. Run with 16 MPI ranks on Rubin; the pytest
temporary directory must be visible to every rank. A short GPU trace proves
actual optimized decode and draft execution after engine warmup, including
CUDA graph replay. The full GSM8K evaluation then runs without profiling.

``TestModelingV2Glm52Nvfp4Sm103Tep4``: identity and MTP1 GSM8K, paired in one
session, and MTP1 acceptance for the SM103 tep4 target.
"""

import json
import os
import time
from pathlib import Path

import pytest

from tensorrt_llm import LLM
from tensorrt_llm._torch._experimental.modeling_v2 import MODELING_V2_VALIDATE_ENV
from tensorrt_llm._utils import get_sm_version
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
from ..accuracy_core import (
    GSM8K,
    LlmapiAccuracyTestHarness,
    acceptance_length_from_iteration_stats,
    assert_acceptance_length,
    compute_acceptance_length,
)
from ._env import scoped_env_overrides

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
        with scoped_env_overrides(_RANK_ENV), LLM(_MODEL_PATH, **_engine_kwargs()) as llm:
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


# The targets assert their own SM at construction: certification is per GPU
# architecture, and a receipt from another one says nothing here.
skip_not_sm103 = pytest.mark.skipif(
    get_sm_version() != 103, reason="modeling_v2 targets in this batch are certified on sm_103 only"
)


class TestModelingV2Glm52Nvfp4Sm103Tep4(LlmapiAccuracyTestHarness):
    """glm-5.2-nvfp4 / sm_103 / tep4: identity, the mtp1 variant and its acceptance."""

    MODEL_NAME = "zai-org/GLM-5.2"
    MODEL_PATH = f"{llm_models_root()}/GLM-5.2-NVFP4"

    # The parallel topology the path's tep4 segment declares: attention heads
    # and experts both sharded four ways, attention DP off. Routing derives the
    # target *from* these; attention DP would be a different weight layout and
    # so a different target.
    TEP4 = dict(tensor_parallel_size=4, moe_expert_parallel_size=4, enable_attention_dp=False)

    # The engine bounds the target's kernels are compiled for. Routing does not
    # read them, so they are set here rather than left to the defaults:
    #   max_batch_size 32  the CUDA-graph capture ceiling the target was
    #                      validated at. The sparse-MLA decode kernel is
    #                      compiled for the engine's max_batch_size.
    #   max_num_tokens 8192  the routed-expert kernels are warmed for, and
    #                      chunk a step at, 8192 tokens.
    #   max_seq_len 8192   the decode indexer's column count is fixed from it,
    #                      and GSM8K never comes near it.
    BOUNDS = dict(
        max_batch_size=32,
        max_num_tokens=8192,
        max_seq_len=8192,
        cuda_graph_config=CudaGraphConfig(max_batch_size=32, enable_padding=True),
    )

    # The checkpoint carries one MTP layer, so a draft length of 1 resolves to
    # the one-model MTP-Eagle loop, the only draft path the target implements;
    # its specialised rope kernel is built for that two-token width.
    MTP1 = MTPDecodingConfig(max_draft_len=1)

    # The kv-cache fractions are run requirements rather than tuning choices.
    # The MTP module and its separate draft pool take several GiB per rank, and
    # an MTP engine on this checkpoint at the identity leg's 0.7 has been seen
    # to stall rather than finish.
    IDENTITY_KV = KvCacheConfig(free_gpu_memory_fraction=0.7)
    MTP1_KV = KvCacheConfig(free_gpu_memory_fraction=0.55)

    # The acceptance anchor. It names the target and variant, not a test
    # function, because that is what the number is a property of. Its value was
    # measured on stock modeling (modeling_v2="off") under the same
    # speculative_config, so the gate is not read against the target's own number.
    ACCEPTANCE_KEY = "ModelingV2Glm52Nvfp4Sm103Tep4::mtp1"

    # There is deliberately one test. It evaluates the identity config as its
    # first leg and the mtp1 config as its second, and ``task.evaluate``
    # asserts accuracy against the reference on the way past; the mtp1 leg also
    # reads its acceptance length. Separate cases would gate nothing new and
    # would each cost another engine boot of a 78-layer, 4-rank model.

    @skip_not_sm103
    @pytest.mark.skip_less_device(4)
    def test_gsm8k_identity_vs_mtp1_and_acceptance(self):
        """The identity accuracy gate, the gate on MTP not moving it, and acceptance.

        Turning MTP on must not move the answers.

        Rejection sampling holds the emitted distribution to the target
        model's, so the two scores should differ only by sampling noise.

        Both legs assert accuracy against the registered reference as they
        run -- this test is therefore the identity gate as well as the
        comparison.

        Both measurements are taken **in this one test** on purpose: a delta
        against a score recorded in some earlier session carries that
        session's variance into the judgement. Paired, the variance is common
        to both and cancels.

        The score is the harness default, with no ``scores_filter``. That is
        the protocol the registered GLM-5.2 references were set under and the
        one this target was validated with, so the anchor and the measurement
        read the same metric.

        Acceptance length is the only gate that can see a miscomputed draft
        layer. Rejection sampling makes a wrong draft path *slower*, not wrong:
        every draft is rejected, the text stays correct, and the accuracy gates
        pass. The mtp1 leg therefore records iteration stats and checks its
        acceptance length against ``ACCEPTANCE_KEY``.
        """
        task = GSM8K(self.MODEL_NAME)

        # One dict for both legs: the comparison only means anything if the two
        # engines were built under the same mode. It also asks each engine to
        # check its step contract on its first forward -- the contract is about
        # metadata that engine prepared, so the only place it can be checked
        # honestly is a run that prepared it; a synthesized metadata object
        # would be checking the synthesizer.
        modeling_v2 = dict(modeling_v2="require", env_overrides={MODELING_V2_VALIDATE_ENV: "1"})

        with (
            scoped_env_overrides(modeling_v2["env_overrides"]),
            LLM(
                self.MODEL_PATH,
                kv_cache_config=self.IDENTITY_KV,
                **modeling_v2,
                **self.TEP4,
                **self.BOUNDS,
            ) as llm,
        ):
            identity = task.evaluate(llm)

        with (
            scoped_env_overrides(modeling_v2["env_overrides"]),
            LLM(
                self.MODEL_PATH,
                speculative_config=self.MTP1,
                kv_cache_config=self.MTP1_KV,
                enable_iter_perf_stats=True,
                **modeling_v2,
                **self.TEP4,
                **self.BOUNDS,
            ) as llm,
        ):
            mtp1 = task.evaluate(llm)
            acceptance_length = compute_acceptance_length(llm)

        delta = mtp1 - identity
        print(f"[modeling_v2] gsm8k identity={identity:.4f} mtp1={mtp1:.4f} delta={delta:+.4f}")
        # 2 sigma at the ~0.6 stderr this benchmark reports at n=1319.
        assert abs(delta) < 1.2, (
            f"MTP moved gsm8k by {delta:+.4f} (identity={identity:.4f}, "
            f"mtp1={mtp1:.4f}); rejection sampling should have held the "
            f"distribution, so this is not sampling noise"
        )
        print(f"[AL] mtp1 acceptance_length = {acceptance_length:.3f}")
        assert_acceptance_length(self.ACCEPTANCE_KEY, acceptance_length)
