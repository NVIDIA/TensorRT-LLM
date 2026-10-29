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
"""Whole-model accuracy and MTP coverage for the GLM Rubin modeling_v2 targets.

Run each case with its 16, 8 or 4 MPI ranks on Rubin; the pytest temporary directory must be visible
to every rank. A short GPU trace proves actual optimized decode and draft
execution after engine warmup, including CUDA graph replay. The full GSM8K
evaluation then runs without profiling.
"""

import hashlib
import json
import os
import re
import time
from collections import Counter
from dataclasses import asdict, dataclass
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


@dataclass(frozen=True)
class AccuracyCase:
    name: str
    tp_size: int
    batch_size: int
    draft_len: int
    max_seq_len: int
    kv_max_tokens: int
    al_reference: float | None
    production_capacity: bool = False
    sampled_probe: bool = False


POINT01 = AccuracyCase("tp16_b1_k7", 16, 1, 7, 16384, 32768, _AL_REFERENCE)
# Full-corpus modeling_v2=off reference for TP4/BS3/K5 on SM107 (2026-10-28),
# using the same 95% acceptance-length retention policy as POINT01.
POINT02 = AccuracyCase(
    "tp4_b3_k5",
    4,
    3,
    5,
    1 << 20,
    (1 << 20) + 64 * 3,
    4.53227789271686,
    production_capacity=True,
    sampled_probe=True,
)


# Full-corpus modeling_v2=off reference for TP8/BS2/K3 on SM107 (2026-10-28).
POINT03 = AccuracyCase(
    "tp8_b2_k3",
    8,
    2,
    3,
    1 << 20,
    (1 << 20) + 64 * 2,
    3.539837056750006,
    production_capacity=True,
    sampled_probe=True,
)


POINT04 = AccuracyCase(
    "tp8_b1_k7",
    8,
    1,
    7,
    1 << 20,
    (1 << 20) + 64,
    5.100986127360856,
    production_capacity=True,
    sampled_probe=True,
)


def _rank_env(case: AccuracyCase) -> dict[str, str]:
    env = dict(_RANK_ENV)
    if case in (POINT02, POINT03, POINT04):
        env["TRTLLM_MOE_FINALIZE_CTA1"] = "1"
    return env


def _engine_kwargs(case: AccuracyCase = POINT01, *, modeling_v2: str = "require") -> dict:
    assert modeling_v2 in ("require", "off")
    speculative_config = MTPDecodingConfig(
        max_draft_len=case.draft_len,
        use_relaxed_acceptance_for_thinking=False,
    )
    # This is a Pydantic private attribute: pass the object, not model_dump().
    # The optimized MTP worker requires the target's cache instead of a separate
    # draft cache. Its CUDA kernels in the probe also verify this on the workers.
    speculative_config._allow_separate_draft_kv_cache = False
    return dict(
        tensor_parallel_size=case.tp_size,
        pipeline_parallel_size=1,
        moe_expert_parallel_size=1,
        enable_attention_dp=False,
        modeling_v2=modeling_v2,
        max_batch_size=case.batch_size,
        max_seq_len=case.max_seq_len,
        max_num_tokens=4096,
        enable_chunked_prefill=True,
        kv_cache_config=KvCacheConfig(
            dtype="fp8",
            tokens_per_block=64,
            enable_block_reuse=False,
            free_gpu_memory_fraction=0.8,
            max_tokens=case.kv_max_tokens,
            host_cache_size=0,
        ),
        cuda_graph_config=CudaGraphConfig(
            batch_sizes=list(range(1, case.batch_size + 1)), enable_padding=True
        ),
        moe_config=MoeConfig(backend="TRTLLM", use_low_precision_moe_combine=True),
        sparse_attention_config=DeepSeekSparseAttentionConfig(
            enable_heuristic_topk=True,
            use_cute_dsl_paged_mqa_logits=False,
        ),
        speculative_config=speculative_config,
        enable_iter_perf_stats=True,
        iter_perf_stats_interval=1,
        max_stats_len=-1,
        env_overrides=_rank_env(case),
    )


def _assert_optimized_paths(llm: LLM, trace_dir: Path, *, optimized: bool = True) -> None:
    """Observe GPU execution, which Python forward counters miss on graph replay."""
    trace_dir.mkdir(parents=True)
    if optimized:
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
        if optimized:
            llm.stop_profile()

    if not optimized:
        return

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


def _write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n")


def _speculative_stats(stats: list[dict]) -> list[dict]:
    return [
        stat["specDecodingStats"]
        for stat in stats
        if stat.get("specDecodingStats") and stat["specDecodingStats"]["numDraftTokens"] > 0
    ]


def _assert_point02_trace(trace_dir: Path, batch_size: int, *, sampled: bool) -> None:
    traces = list(trace_dir.glob("trtllm-trace-*-rank-*.json"))
    assert len(traces) == POINT02.tp_size, traces
    router = {
        1: "routingFusedRouterKernel<6>",
        2: "routingFusedRouterWideKernel<12>",
        3: "routingFusedRouterMmaKernel<18>",
    }[batch_size]
    required = [
        router,
        "glmSparseMlaFlashDecodeHeads16Kernel",
        "FP8MQALogitsKernel",
        "GvrMainKernel",
        "draftSampleGlueWideKernel",
        "_stash_last_accepted_rows_kernel",
    ]
    if batch_size == 1:
        required.append("lowMBf16MoeFc2Kernel")
    else:
        required.extend(["lowMBf16MoeMultiFc1Kernel", "lowMBf16MoeMultiFc2Kernel"])
    forbidden = ("dsaHistTopKKernel", "dsaHistShardMetaKernel", "draftEhprojTableKernel")
    evidence = {}
    for rank in range(POINT02.tp_size):
        files = [path for path in traces if path.name.endswith(f"-rank-{rank}.json")]
        assert len(files) == 1, f"Missing or duplicate trace for rank {rank}"
        names = sorted(
            {
                event["name"]
                for event in json.loads(files[0].read_text())["traceEvents"]
                if event.get("cat") == "kernel"
            }
        )
        # Profiler demangling may add whitespace inside template arguments.
        compact = [re.sub(r"\s+", "", name) for name in names]
        for kernel in required:
            assert any(kernel in name for name in compact), (
                f"Rank {rank}, batch {batch_size}: missing {kernel}"
            )
        for kernel in forbidden:
            assert not any(kernel in name for name in names), (
                f"Rank {rank}, batch {batch_size}: unexpected {kernel}"
            )
        assert any("glm_moe_dsa" in name and "reduceKernel" in name for name in names), (
            f"Rank {rank}: owned IPC reduction did not execute"
        )
        if sampled:
            for kernel in ("directPartialsKernel", "directDrawKernel"):
                assert any("glm_moe_dsa" in name and kernel in name for name in names), (
                    f"Rank {rank}: owned non-greedy {kernel} did not execute"
                )
        evidence[str(rank)] = names
    _write_json(trace_dir / "kernel-evidence.json", evidence)


def _assert_tp8_trace(
    trace_dir: Path, case: AccuracyCase, batch_size: int, *, sampled: bool
) -> None:
    assert case in (POINT03, POINT04), case
    traces = list(trace_dir.glob("trtllm-trace-*-rank-*.json"))
    assert len(traces) == case.tp_size, traces
    required = [
        f"routingFusedRouterKernel<{batch_size * (case.draft_len + 1)}>",
        "glmSparseMlaFlashDecodeKernel<8,",
        "FP8MQALogitsKernel",
        "GvrMainKernel",
        "gemm2K256Kernel",
        "mnnvlHopKernel<8,true>",
        "mnnvlMoeFinalizeHopKernel<8>",
        "draftEhprojTableKernel<8>",
        "draftSampleGlueKernel",
        "_stash_last_accepted_rows_kernel",
    ]
    if batch_size == 1:
        required.append("lowMBf16MoeFc2Kernel")
    else:
        required.extend(["lowMBf16MoeMultiFc1Kernel", "lowMBf16MoeMultiFc2Kernel"])
    if sampled:
        required.extend(["directPartialsKernel", "directDrawKernel"])
    forbidden = (
        "dsaHistTopKKernel",
        "dsaHistShardMetaKernel",
        "glmSparseMlaFlashDecodeHeads16Kernel",
        "draftSampleGlueWideKernel",
    )
    evidence = {}
    for rank in range(case.tp_size):
        files = [path for path in traces if path.name.endswith(f"-rank-{rank}.json")]
        assert len(files) == 1, f"Missing or duplicate trace for rank {rank}"
        names = sorted(
            {
                event["name"]
                for event in json.loads(files[0].read_text())["traceEvents"]
                if event.get("cat") == "kernel"
            }
        )
        compact = [re.sub(r"\s+", "", name) for name in names]
        for kernel in required:
            assert any(kernel in name for name in compact), (
                f"Rank {rank}, batch {batch_size}: missing {kernel}"
            )
        for kernel in forbidden:
            assert not any(kernel in name for name in compact), (
                f"Rank {rank}, batch {batch_size}: unexpected {kernel}"
            )
        evidence[str(rank)] = names
    _write_json(trace_dir / "kernel-evidence.json", evidence)


def _batch_probe(
    llm: LLM,
    case: AccuracyCase,
    output_dir: Path,
    batch_size: int,
    *,
    optimized: bool,
    sampled: bool = False,
) -> list[dict]:
    label = f"probe-b{batch_size}-{'sampled' if sampled else 'greedy'}"
    probe_dir = output_dir / label
    probe_dir.mkdir()
    # >2048 keys forces real indexer scoring/TopK rather than the short-context
    # dense bypass. Equal-length pretokenized prompts keep the requested batch resident.
    prompts = []
    for row in range(batch_size):
        tokens = llm.tokenizer.encode(
            f"Example {row + 1}. " + "Explain multiplication and show each arithmetic step. " * 1024
        )
        assert len(tokens) >= 3072
        prompts.append(list(tokens[:3072]))
    assert len({tuple(prompt) for prompt in prompts}) == batch_size
    trace_dir = probe_dir / "traces"
    if optimized:
        trace_dir.mkdir()
        # Skip chunked-prefill iterations, then observe several real decode
        # iterations including captured graphs, not initialization warmup.
        llm.start_profile(output_dir=str(trace_dir), num_steps=8, start_step=8, activities=["GPU"])
    try:
        outputs = llm.generate(
            prompts,
            SamplingParams(
                temperature=1.0 if sampled else 0.0,
                top_p=0.95 if sampled else 1.0,
                max_tokens=128,
                ignore_eos=True,
            ),
        )
    finally:
        if optimized:
            llm.stop_profile()
    stats = _request_stats(llm, batch_size)
    _write_json(probe_dir / "iteration-stats.json", stats)
    _write_json(probe_dir / "outputs.json", [list(out.outputs[0].token_ids) for out in outputs])
    assert len(outputs) == batch_size
    assert all(len(out.outputs[0].token_ids) == 128 for out in outputs)
    counts = Counter(stat["numRequestsWithDraftTokens"] for stat in _speculative_stats(stats))
    _write_json(probe_dir / "batch-counts.json", dict(counts))
    assert counts[batch_size] > 0, f"Requested batch {batch_size} was never executed: {counts}"
    if optimized:
        if case in (POINT03, POINT04):
            _assert_tp8_trace(trace_dir, case, batch_size, sampled=sampled)
        else:
            _assert_point02_trace(trace_dir, batch_size, sampled=sampled)
    return stats


def _capacity_evidence(llm: LLM, stats: list[dict], output_dir: Path, case: AccuracyCase) -> None:
    """Measure final leader state; other pure-TP ranks are inferred, not inspected."""
    capacity = llm.get_kv_cache_capacity()
    snapshots = [
        {
            key: stat[key]
            for key in (
                "kvCacheStats",
                "kvCacheIterationStats",
                "kvCacheIterationStatsByPoolGroup",
                "kvCacheIterationStatsByLifecycle",
            )
            if key in stat
        }
        for stat in stats
    ]
    snapshots = [snapshot for snapshot in snapshots if snapshot]
    windows = {
        int(lifecycle["windowSize"])
        for snapshot in snapshots
        for lifecycle in snapshot.get("kvCacheIterationStatsByLifecycle", {}).values()
        if lifecycle.get("kind") == "attention" and lifecycle.get("windowSize") is not None
    }
    evidence = dict(
        measured_rank=0,
        capacity=capacity,
        attention_windows=sorted(windows),
        snapshots=snapshots,
        other_rank_capacity="inferred from V2 distributed MIN and replicated pure-TP cache schema",
        route_evidence="Every TP rank is separately checked by GPU traces; traces do not measure capacity",
    )
    _write_json(output_dir / "capacity-evidence.json", evidence)
    assert capacity.get("tokensPerBlock") == 64 and capacity.get("maxNumBlocks", 0) > 0, capacity
    # GLM uses full-attention lifecycles. Their V2 windowSize is the final
    # manager.max_seq_len after profiling/capacity clamping, not llm.args.
    assert capacity.get("maxNumTokens", 0) >= case.max_seq_len, capacity
    assert windows and min(windows) >= case.max_seq_len, (
        f"Actual leader window below {case.max_seq_len}: {windows}"
    )
    assert any(snapshot.get("kvCacheIterationStatsByPoolGroup") for snapshot in snapshots), (
        "Missing physical pool-group capacity statistics"
    )


def run_accuracy_case(
    case: AccuracyCase, output_dir: Path, *, modeling_v2: str = "require"
) -> dict:
    """Shared protocol; a task-local stock runner may explicitly select ``off``.

    Repository tests always require ModelingV2 and a frozen independent AL
    reference. Stock records the reference without a candidate retention check.
    """
    assert modeling_v2 in ("require", "off")
    assert os.environ.get("INTEGRATION_TEST") != "1", "This guard requires all 1319 samples"
    if modeling_v2 == "require":
        assert case.al_reference is not None, (
            f"{case.name}: run and review the matched full1319 modeling_v2=off baseline, "
            "then freeze its independent AL reference before running the repository target test"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(
        output_dir / "accuracy-protocol.json",
        dict(
            case=asdict(case),
            modeling_v2=modeling_v2,
            num_samples=_NUM_SAMPLES,
            model_path=_MODEL_PATH,
            dataset_path=_DATASET_PATH,
            test_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            scoring_temperature=0.0,
            scoring_top_p=1.0,
            max_output_tokens=12288,
            max_prompt_tokens=4096,
            forced_accepted_tokens=0,
            rank_env=_rank_env(case),
        ),
    )
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setenv("TLLM_EVAL_MAX_IN_FLIGHT", str(case.batch_size))
        monkeypatch.setenv("TLLM_EVAL_PARTIAL_SCORES_EVERY", "100")
        for key, value in _rank_env(case).items():
            monkeypatch.setenv(key, value)
        with LLM(_MODEL_PATH, **_engine_kwargs(case, modeling_v2=modeling_v2)) as llm:
            if case.production_capacity:
                probe_stats = []
                probe_batches = {POINT02: (3, 1, 2), POINT03: (2, 1), POINT04: (1,)}
                for batch in probe_batches[case]:
                    probe_stats.extend(
                        _batch_probe(
                            llm, case, output_dir, batch, optimized=modeling_v2 == "require"
                        )
                    )
                if case.sampled_probe:
                    probe_stats.extend(
                        _batch_probe(
                            llm,
                            case,
                            output_dir,
                            case.batch_size,
                            optimized=modeling_v2 == "require",
                            sampled=True,
                        )
                    )
                _capacity_evidence(llm, probe_stats, output_dir, case)
            else:
                _assert_optimized_paths(
                    llm, output_dir / "optimized-path-traces", optimized=modeling_v2 == "require"
                )
                _write_json(output_dir / "probe-iteration-stats.json", _request_stats(llm, 1))
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
                output_path=str(output_dir),
            )
            evaluator.evaluate(
                llm,
                sampling_params=SamplingParams(
                    temperature=0.0, top_p=1.0, max_tokens=12288, truncate_prompt_tokens=4096
                ),
                streaming=False,
                sampling_override=True,
                scores_filter="exact_match,strict-match",
            )
            stats = _request_stats(llm, _NUM_SAMPLES)
    _write_json(output_dir / "corpus-iteration-stats.json", stats)
    spec_stats = _speculative_stats(stats)
    counts = Counter(stat["numRequestsWithDraftTokens"] for stat in spec_stats)
    assert len(spec_stats) >= (_NUM_SAMPLES + case.batch_size - 1) // case.batch_size, (
        "Missing full-run speculative iteration statistics"
    )
    assert all(1 <= count <= case.batch_size for count in counts), counts
    assert counts[case.batch_size] > 0, f"Corpus never executed the full batch: {counts}"
    acceptance_length = acceptance_length_from_iteration_stats(stats)
    scores = _full_accuracy_scores(output_dir / f"samples_{_TASK_NAME}.json")
    metrics = dict(scores)
    metrics.update(
        acceptance_length=acceptance_length,
        speculative_iterations=len(spec_stats),
        accepted_draft_tokens=sum(stat["numAcceptedTokens"] for stat in spec_stats),
        drafted_tokens=sum(stat["numDraftTokens"] for stat in spec_stats),
        speculative_request_steps=sum(stat["numRequestsWithDraftTokens"] for stat in spec_stats),
        speculative_batch_counts=dict(counts),
        completed_requests=sum(stat["numCompletedRequests"] for stat in stats),
        new_requests=sum(stat["numNewActiveRequests"] for stat in stats),
        modeling_v2=modeling_v2,
        al_reference=case.al_reference,
    )
    _write_json(output_dir / "glm_metrics.json", metrics)
    print(f"[GLM {case.name} {modeling_v2}] {json.dumps(metrics)}")
    assert metrics["completed_requests"] == metrics["new_requests"] == _NUM_SAMPLES, metrics
    for name, score in scores.items():
        assert _ACCURACY_FLOOR <= score <= 100.0, (
            f"{name}: {score:.3f}% is below the local {_ACCURACY_FLOOR}% floor"
        )
    floor = 0.95 * case.al_reference if modeling_v2 == "require" else 1.0
    assert floor <= acceptance_length <= case.draft_len + 1, (
        f"MTP AL {acceptance_length:.6f} is outside [{floor:.6f}, {case.draft_len + 1}]"
    )
    return metrics


class TestModelingV2Glm52Nvfp4Sm107LowLatency(LlmapiAccuracyTestHarness):
    @skip_no_rubin
    @pytest.mark.skip_less_mpi_world_size(16)
    def test_mtp7_gsm8k_and_acceptance(self, tmp_path: Path):
        run_accuracy_case(POINT01, tmp_path)

    @skip_no_rubin
    @pytest.mark.skip_less_mpi_world_size(4)
    def test_tp4_mtp5_gsm8k_and_acceptance(self, tmp_path: Path):
        run_accuracy_case(POINT02, tmp_path)

    @skip_no_rubin
    @pytest.mark.skip_less_mpi_world_size(8)
    def test_tp8_mtp3_gsm8k_and_acceptance(self, tmp_path: Path):
        run_accuracy_case(POINT03, tmp_path)

    @skip_no_rubin
    @pytest.mark.skip_less_mpi_world_size(8)
    def test_tp8_mtp7_gsm8k_and_acceptance(self, tmp_path: Path) -> None:
        run_accuracy_case(POINT04, tmp_path)
