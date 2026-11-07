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
"""GLM-5.2 NVFP4 Rubin high-throughput GSM8K and MTP acceptance checks."""

import json
import os
import time
from pathlib import Path

import pytest

from tensorrt_llm import LLM
from tensorrt_llm.evaluate import GSM8KInferenceX
from tensorrt_llm.evaluate.lm_eval import LmEvalWrapper
from tensorrt_llm.llmapi import (
    CudaGraphConfig,
    DeepSeekSparseAttentionConfig,
    KvCacheConfig,
    MoeConfig,
    MTPDecodingConfig,
    SamplingParams,
)

from .....conftest import llm_models_root, skip_no_rubin
from ....accuracy_core import LlmapiAccuracyTestHarness
from ..._env import scoped_env_overrides

_TP_SIZE = 16
_NUM_SAMPLES = 1319
_ACCURACY_FLOOR = 90.0
_TASK_NAME = "gsm8k_inferencex"
_MODEL_PATH = f"{llm_models_root()}/GLM-5.2-NVFP4"
_DATASET_PATH = f"{llm_models_root()}/datasets/openai/gsm8k"


def _full_accuracy_scores(result_path: Path) -> dict[str, float]:
    results = json.loads(result_path.read_text())
    counts = results["n-samples"][_TASK_NAME]
    assert counts["original"] == counts["effective"] == _NUM_SAMPLES, counts
    samples = results["samples"][_TASK_NAME]
    assert len({sample["doc_id"] for sample in samples}) == _NUM_SAMPLES
    scores = results["results"][_TASK_NAME]
    return {name: scores[f"exact_match,{name}"] for name in ("strict-match", "flexible-extract")}


_HIGH_THROUGHPUT_POINTS = (
    # Full-corpus GSM8K references use the final request counters, as below.
    pytest.param(7, 7, 5.088854684111103, id="p05_tp16_ep16_b7_k7"),
    pytest.param(10, 5, 4.525905448220165, id="p06_tp16_ep16_b10_k5"),
    pytest.param(20, 5, 4.530280420319592, id="p07_tp16_ep16_b20_k5"),
)
_HIGH_THROUGHPUT_ENV = {
    "TRTLLM_DSA_INDEXER_BF16": "1",
    "TLLM_SPEC_DECODE_FORCE_NUM_ACCEPTED_TOKENS": "0",
    "TLLM_METRICS_ALL_RANKS": "1",
    "TRTLLM_ENABLE_PDL": "1",
    "TRTLLM_MLA_Q_QUANT_GRID_USE_Q_LEN": "1",
}


def _high_throughput_engine_kwargs(batch_size: int, draft_len: int) -> dict:
    kwargs = dict(
        tensor_parallel_size=_TP_SIZE,
        pipeline_parallel_size=1,
        modeling_v2="require",
        max_num_tokens=4096,
        enable_chunked_prefill=True,
        enable_iter_perf_stats=True,
        iter_perf_stats_interval=1,
        max_stats_len=-1,
    )
    kwargs.update(
        moe_expert_parallel_size=16,
        enable_attention_dp=True,
        enable_lm_head_tp_in_adp=True,
        max_batch_size=batch_size,
        max_seq_len=1048576,
        cuda_graph_config=CudaGraphConfig(
            batch_sizes=list(range(1, batch_size + 1)), enable_padding=True
        ),
        kv_cache_config=KvCacheConfig(
            dtype="nvfp4",
            tokens_per_block=64,
            enable_block_reuse=False,
            enable_partial_reuse=False,
            free_gpu_memory_fraction=0.8,
            # Keep the 1M indexer envelope after MTP's reserved tokens are subtracted.
            max_tokens=1048576 + 64,
            host_cache_size=0,
        ),
        moe_config=MoeConfig(backend="MEGAMOE_CUTEDSL", use_low_precision_moe_combine=True),
        sparse_attention_config=DeepSeekSparseAttentionConfig(
            enable_heuristic_topk=True,
            use_cute_dsl_paged_mqa_logits=True,
        ),
        env_overrides=dict(_HIGH_THROUGHPUT_ENV),
    )
    kwargs["speculative_config"] = MTPDecodingConfig(
        max_draft_len=draft_len, use_relaxed_acceptance_for_thinking=False
    )
    kwargs["speculative_config"]._allow_separate_draft_kv_cache = False
    return kwargs


def _assert_high_throughput_paths(llm: LLM, batch_size: int, trace_dir: Path) -> list:
    """Exercise every ADP owner after capture, with enough requests to fill its batch."""
    prompts = [
        f"Exercise {index}: explain step by step how to multiply two three-digit numbers."
        for index in range(2 * _TP_SIZE * batch_size)
    ]
    trace_dir.mkdir(parents=True)
    llm.start_profile(output_dir=str(trace_dir), num_steps=8, start_step=8, activities=["GPU"])
    try:
        outputs = llm.generate(
            prompts,
            SamplingParams(temperature=0.0, top_p=1.0, max_tokens=128, ignore_eos=True),
            use_tqdm=False,
        )
        assert len(outputs) == len(prompts)
        assert all(len(output.outputs[0].token_ids) == 128 for output in outputs)
    finally:
        llm.stop_profile()
    traces = list(trace_dir.glob("trtllm-trace-*-rank-*.json"))
    assert len(traces) == _TP_SIZE, f"Expected {_TP_SIZE} GPU traces, found {traces}"
    for rank in range(_TP_SIZE):
        rank_traces = [path for path in traces if path.name.endswith(f"-rank-{rank}.json")]
        assert len(rank_traces) == 1, f"Missing or duplicate trace for rank {rank}"
        events = json.loads(rank_traces[0].read_text())["traceEvents"]
        names = [event["name"] for event in events if event.get("cat") == "kernel"]
        for kernel in (
            "publishOwnerMembersKernel",
            "glmRubinNoAuxTcFp4QuantKernel",
            "draftSampleGlueAdpKernel",
        ):
            assert any(kernel in name for name in names), f"Rank {rank} did not execute {kernel}"
    return outputs


def _request_speculation_records(outputs: list) -> list[dict]:
    """Retain each owner's final counters; ADP iteration speculation fields are broadcast."""
    records = []
    for output in outputs:
        assert output.finished and not getattr(output, "error", None)
        per_draft = list(output.per_pos_drafted or [0])
        per_accepted = list(output.per_pos_accepted or [0])
        assert all(isinstance(value, int) and value >= 0 for value in (*per_draft, *per_accepted))
        if output.spec_dec_totals is None:
            assert not any(per_draft) and not any(per_accepted)
            accepted, drafted = 0, 0
        else:
            accepted, drafted = output.spec_dec_totals
        assert sum(per_draft) == drafted and sum(per_accepted) == accepted
        assert all(
            a <= (per_draft[i] if i < len(per_draft) else 0) for i, a in enumerate(per_accepted)
        )
        records.append(
            dict(
                request_id=output.request_id,
                accepted_draft_tokens=accepted,
                drafted_tokens=drafted,
                per_pos_drafted=per_draft,
                per_pos_accepted=per_accepted,
            )
        )
    assert len({record["request_id"] for record in records}) == len(records)
    return records


def _request_acceptance_metrics(records: list[dict]) -> dict:
    accepted = sum(record["accepted_draft_tokens"] for record in records)
    drafted = sum(record["drafted_tokens"] for record in records)
    steps = sum(record["per_pos_drafted"][0] for record in records)
    assert 0 <= accepted <= drafted and steps > 0
    return dict(
        acceptance_length=1 + accepted / steps,
        accepted_draft_tokens=accepted,
        drafted_tokens=drafted,
        speculative_request_steps=steps,
        completed_requests=len(records),
    )


def _drain_adp_stats(llm: LLM) -> list[dict]:
    # Request futures already completed. Iteration completion counters belong
    # to rank zero and cannot be used to count the whole ADP corpus.
    stats = []
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        batch = llm.get_stats(timeout=0.5)
        if not batch:
            break
        for stat in batch:
            if "rank" not in stat:
                stat = {**stat, "rank": stat["attentionDpRank"]}
            stats.append(stat)
    return stats


class TestModelingV2Glm52Nvfp4Sm107HighThroughput(LlmapiAccuracyTestHarness):
    @skip_no_rubin
    @pytest.mark.skip_less_mpi_world_size(_TP_SIZE)
    @pytest.mark.parametrize("batch_size,draft_len,al_reference", _HIGH_THROUGHPUT_POINTS)
    def test_gsm8k_and_acceptance(
        self,
        batch_size: int,
        draft_len: int,
        al_reference: float,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        assert os.environ.get("INTEGRATION_TEST") != "1", "This guard requires all 1319 samples"
        monkeypatch.setenv("TLLM_EVAL_MAX_IN_FLIGHT", str(_TP_SIZE * batch_size))
        monkeypatch.setenv("TLLM_EVAL_PARTIAL_SCORES_EVERY", "100")
        monkeypatch.setenv("TLLM_EVAL_SPEC_STATS", "1")
        request_records = []
        original_log_spec_stats = LmEvalWrapper._log_spec_stats

        def capture_request_stats(wrapper, outputs):
            request_records.extend(_request_speculation_records(outputs))
            original_log_spec_stats(wrapper, outputs)

        monkeypatch.setattr(LmEvalWrapper, "_log_spec_stats", capture_request_stats)
        with (
            scoped_env_overrides(_HIGH_THROUGHPUT_ENV),
            LLM(_MODEL_PATH, **_high_throughput_engine_kwargs(batch_size, draft_len)) as llm,
        ):
            probe_outputs = _assert_high_throughput_paths(
                llm, batch_size, tmp_path / "optimized-path-traces"
            )
            probe_records = _request_speculation_records(probe_outputs)
            probe_stats = _drain_adp_stats(llm)
            (tmp_path / "probe-request-stats.json").write_text(json.dumps(probe_records) + "\n")
            (tmp_path / "probe-iteration-stats.json").write_text(json.dumps(probe_stats) + "\n")
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
                    temperature=0.0, top_p=1.0, max_tokens=12288, truncate_prompt_tokens=4096
                ),
                streaming=False,
                sampling_override=True,
                scores_filter="exact_match,strict-match",
            )
            stats = _drain_adp_stats(llm)

        assert len(request_records) == _NUM_SAMPLES
        assert len({record["request_id"] for record in request_records}) == _NUM_SAMPLES
        rank_ids = {stat["rank"] for stat in stats}
        assert rank_ids == set(range(_TP_SIZE)), f"Incomplete ADP statistics: {rank_ids}"
        for rank in range(_TP_SIZE):
            assert (
                max(
                    stat["inflightBatchingStats"]["numGenRequests"]
                    for stat in probe_stats
                    if stat["rank"] == rank
                )
                == batch_size
            )
        scores = _full_accuracy_scores(tmp_path / f"samples_{_TASK_NAME}.json")
        metrics = {
            **scores,
            "batch_size": batch_size,
            "draft_len": draft_len,
            "al_reference": al_reference,
            **_request_acceptance_metrics(request_records),
        }
        acceptance_length = metrics["acceptance_length"]
        (tmp_path / "request-speculation-stats.json").write_text(json.dumps(request_records) + "\n")
        (tmp_path / "glm_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
        (tmp_path / "iteration_stats.json").write_text(json.dumps(stats) + "\n")
        print(f"[GLM high throughput] {json.dumps(metrics)}")
        assert all(_ACCURACY_FLOOR <= score <= 100.0 for score in scores.values())
        floor = 0.95 * al_reference
        assert floor <= acceptance_length <= draft_len + 1, (
            f"MTP AL {acceptance_length:.6f} is outside [{floor:.6f}, {draft_len + 1}]"
        )
