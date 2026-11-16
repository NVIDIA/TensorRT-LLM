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

"""Opt-in, unseeded long-reasoning replay with per-request resource evidence."""

import csv
import hashlib
import json
import os
import re
import subprocess
import time
from collections.abc import Callable
from pathlib import Path
from typing import Literal

import allure
import pytest
import torch
from datasets import DownloadConfig, load_dataset

from tensorrt_llm import LLM
from tensorrt_llm.llmapi import (
    CudaGraphConfig,
    DFlashDecodingConfig,
    KvCacheConfig,
    MoeConfig,
    MTPDecodingConfig,
    SamplingParams,
)

from ..conftest import llm_models_root
from .accuracy_core import LlmapiAccuracyTestHarness

_QUESTION_SHA256 = "8b5b2cb731b5a7fb6556557d17fc7ef7fb607adfa5050f0d70418eb3b983605b"
_CHOICE_SHA256 = (
    "785c02288b2b75549aab508b45b2b22e733b17ce8a3a06d790e49b6254b95bca",
    "f7689c98847a29cbdcc4683d628a756625e1733dc790c2bbd00dc93fdb14d771",
    "c917c5c161417067ea969ffcb96f9e08c1f65539b38f245ed0c9990478993e89",
    "17329249a660fad19ecd888929409446903e0ef1f857fa545f8c6e9b12cbef6c",
)
_PROMPT_SHA256 = "e82b92e0ce216d02bc81db8b9573d656bfebb78b029d7fc74ed48010b327daa1"
_MAX_TOKENS = 131072
_REPEATS = 11
_MODE = Literal["baseline", "mtp3", "dflash7"]


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _replay_prompt(dataset_path: Path) -> str:
    """Reconstruct the exact replay prompt from the local GPQA cache."""
    assert dataset_path.is_dir(), f"Missing cached GPQA dataset: {dataset_path}"
    dataset = load_dataset(
        str(dataset_path),
        "gpqa_diamond",
        split="train",
        download_config=DownloadConfig(local_files_only=True),
    )
    for row in dataset:
        question = row["Question"]
        if _digest(question) != _QUESTION_SHA256:
            continue
        choices = {
            _digest(row[field]): row[field]
            for field in (
                "Correct Answer",
                "Incorrect Answer 1",
                "Incorrect Answer 2",
                "Incorrect Answer 3",
            )
        }
        ordered_choices = [choices[digest] for digest in _CHOICE_SHA256]
        prompt = (
            "Answer the following multiple choice question. The last line of your response "
            "should be in the following format: 'Answer: A/B/C/D' (e.g. 'Answer: A').\n\n"
            + question
            + "\n\n"
            + "\n".join(f"{letter}) {choice}" for letter, choice in zip("ABCD", ordered_choices))
        )
        assert _digest(prompt) == _PROMPT_SHA256, "Cached replay prompt fingerprint differs"
        return prompt
    pytest.fail("The cached GPQA Diamond dataset does not contain the replay question")


def _gpu_summary(samples_path: Path, elapsed: float) -> dict:
    """Summarize device-wide samples, including allocations in executor workers."""
    samples = []
    with samples_path.open(encoding="utf-8") as resource_file:
        for row in csv.reader(resource_file):
            assert len(row) == 3, f"Unexpected nvidia-smi sample: {row}"
            samples.append([float(value.strip()) for value in row])
    assert samples, "nvidia-smi did not produce any resource samples"
    mean_utilization = sum(row[1] for row in samples) / len(samples)
    return {
        "scope": "entire selected GPU, sampled once per second",
        "samples": len(samples),
        "sampled_peak_memory_mib": max(row[0] for row in samples),
        "mean_utilization_percent": mean_utilization,
        "mean_power_watts": sum(row[2] for row in samples) / len(samples),
        "allocated_gpu_seconds": elapsed,
        "estimated_busy_gpu_seconds": elapsed * mean_utilization / 100,
    }


@pytest.mark.timeout(43200)
class TestQwen36ReasoningReplay(LlmapiAccuracyTestHarness):
    @pytest.mark.parametrize("mode", ["baseline", "mtp3", "dflash7"])
    def test_seedless_replay(
        self,
        mode: _MODE,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        record_property: Callable[[str, object], None],
    ) -> None:
        """Measure eleven identical requests in each mode without a sampling seed.

        Cap hits and missing final answers are evidence for review. This replay
        measures one question repeatedly and does not estimate GPQA accuracy.
        Passing means the experiment completed, not that accuracy was verified.
        """
        assert "LLM_MODELS_ROOT" in os.environ, "Set LLM_MODELS_ROOT to the model cache"
        if "B200" not in torch.cuda.get_device_name(0):
            pytest.skip("This replay requires a B200 GPU")
        model_root = Path(llm_models_root())
        model_path = model_root / "Qwen3.6-35B-A3B-NVFP4"
        assert model_path.is_dir(), f"Missing target checkpoint: {model_path}"
        prompt = _replay_prompt(model_root / "datasets/gpqa")

        spec_config = None
        if mode == "mtp3":
            spec_config = MTPDecodingConfig(
                num_nextn_predict_layers=3, use_rejection_sampling=False
            )
        elif mode == "dflash7":
            draft_path = model_root / "Qwen3.6-35B-A3B-DFlash"
            assert draft_path.is_dir(), f"Missing DFlash checkpoint: {draft_path}"
            spec_config = DFlashDecodingConfig(speculative_model=str(draft_path), max_draft_len=7)

        monkeypatch.setenv("TRTLLM_USE_GDN_REPLAY", "0")
        gpu_id = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
        query = [
            "nvidia-smi",
            f"--id={gpu_id}",
            "--query-gpu=memory.used,utilization.gpu,power.draw",
            "--format=csv,noheader,nounits",
        ]
        subprocess.run(query, check=True, capture_output=True, text=True)
        metrics_path = tmp_path / f"{mode}-requests.jsonl"
        responses_path = tmp_path / f"{mode}-responses.jsonl"
        samples_path = tmp_path / f"{mode}-gpu.csv"
        request_metrics = []
        completed = False
        started = time.perf_counter()
        with samples_path.open("w", encoding="utf-8") as samples_file:
            monitor = subprocess.Popen(query + ["--loop=1"], stdout=samples_file)
            try:
                with LLM(
                    str(model_path),
                    trust_remote_code=True,
                    tensor_parallel_size=1,
                    moe_expert_parallel_size=1,
                    max_batch_size=16,
                    max_seq_len=262144,
                    max_num_tokens=8192,
                    enable_chunked_prefill=True,
                    cuda_graph_config=CudaGraphConfig(enable_padding=True, max_batch_size=16),
                    kv_cache_config=KvCacheConfig(
                        enable_block_reuse=False,
                        free_gpu_memory_fraction=0.9,
                        dtype="fp8",
                        mamba_ssm_cache_dtype="bfloat16",
                        use_kv_cache_manager_v2=False,
                    ),
                    moe_config=MoeConfig(backend="TRTLLM"),
                    speculative_config=spec_config,
                ) as llm:
                    formatted_prompt = llm.tokenizer.apply_chat_template(
                        [{"role": "user", "content": prompt}],
                        tokenize=True,
                        add_generation_prompt=True,
                        enable_thinking=True,
                    )
                    with (
                        metrics_path.open("w", encoding="utf-8") as metrics_file,
                        responses_path.open("w", encoding="utf-8") as responses_file,
                    ):
                        for index in range(_REPEATS):
                            request_started = time.perf_counter()
                            result = llm.generate_async(
                                formatted_prompt,
                                sampling_params=SamplingParams(
                                    temperature=1.0, top_p=0.95, max_tokens=_MAX_TOKENS
                                ),
                            ).result()
                            elapsed = time.perf_counter() - request_started
                            output = result.outputs[0]
                            finish_reason = str(output.finish_reason)
                            answer = re.search(r"Answer:\s*([A-D])\s*[.!]?\s*$", output.text)
                            metric = {
                                "mode": mode,
                                "request_index": index,
                                "elapsed_seconds": elapsed,
                                "output_tokens": len(output.token_ids),
                                "finish_reason": finish_reason,
                                "cap_hit": len(output.token_ids) >= _MAX_TOKENS,
                                "final_answer": answer.group(1) if answer else None,
                            }
                            request_metrics.append(metric)
                            print("REPLAY_REQUEST " + json.dumps(metric), flush=True)
                            metrics_file.write(json.dumps(metric) + "\n")
                            metrics_file.flush()
                            responses_file.write(json.dumps({**metric, "text": output.text}) + "\n")
                            responses_file.flush()
                            assert finish_reason in ("stop", "length"), metric
                    completed = True
            finally:
                monitor.terminate()
                try:
                    monitor.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    monitor.kill()
                    monitor.wait(timeout=10)
                elapsed = time.perf_counter() - started
                try:
                    resources = _gpu_summary(samples_path, elapsed)
                except (AssertionError, ValueError) as error:
                    resources = {"sampling_error": str(error)}
                summary = {
                    "mode": mode,
                    "experiment_completed": completed,
                    "prompt_sha256": _PROMPT_SHA256,
                    "gpu_name": torch.cuda.get_device_name(0),
                    "gpu_id": gpu_id,
                    "tensor_parallel_size": 1,
                    "request_concurrency": 1,
                    "requests": len(request_metrics),
                    "cap_hits": sum(metric["cap_hit"] for metric in request_metrics),
                    "missing_final_answers": sum(
                        metric["final_answer"] is None for metric in request_metrics
                    ),
                    "elapsed_seconds_including_load": elapsed,
                    "resources": resources,
                }
                print("REPLAY_SUMMARY " + json.dumps(summary), flush=True)
                record_property("replay_summary", json.dumps(summary))
                record_property("replay_requests", json.dumps(request_metrics))
                allure.attach(
                    json.dumps(summary, indent=2),
                    name=f"{mode}-summary",
                    attachment_type=allure.attachment_type.JSON,
                )
                for artifact in (samples_path, metrics_path, responses_path):
                    if artifact.is_file():
                        allure.attach.file(
                            str(artifact),
                            name=artifact.name,
                            attachment_type=allure.attachment_type.TEXT,
                        )
        assert len(request_metrics) == _REPEATS
