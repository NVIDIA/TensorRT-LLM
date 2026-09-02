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
"""Post-merge ModelExpress accuracy canaries.

One reference-backed accuracy task per qualified model family runs on an MX
receiver that starts from a metadata-only snapshot, so its weights can only
have arrived through P2P from the live donor. The receiver worker evaluates the
task itself (in its own subprocess, after self-checking its transfer logs) and
writes the score to JSON; this module only launches processes, loads the
accuracy reference YAMLs, and asserts the hypothesis-testing threshold. There is
no paired HF-baseline evaluation: the reference value *is* the baseline.
"""

from __future__ import annotations

import importlib.util
import json
import math
import os
import subprocess
import sys
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

import pytest
import yaml

from defs.model_express.mx_harness import (
    MX_ROLES,
    TRANSFER_TIER_KINDS,
    WORKER_PATH,
    MxE2ECase,
    MxRunLayout,
    archive_run_artifacts,
    assert_transfer_evidence,
    build_canonical_snapshot,
    build_metadata_only_snapshot,
    donor_session,
    load_payload,
    log_tail,
    require_mx_environment,
    resolve_model_path,
    run_worker,
    skip_or_fail,
    worker_command,
    worker_environment,
)
from tensorrt_llm._torch.weight_sharing import ArtifactIdentity, compare_weight_manifests

_REFERENCES_DIR = Path(__file__).resolve().parents[1] / "accuracy" / "references"
_TASK_REFERENCE_FILES = {"MMLU": "mmlu.yaml", "GSM8K": "gsm8k.yaml"}
# The spec keys `accuracy_core.AccuracyTask.evaluate` derives from `llm.args`,
# pinned to the qualified BF16 unquantized, non-speculative MX envelope.
_BF16_ACCURACY_SPECS = dict(
    dtype="auto",
    quant_algo=None,
    kv_cache_quant_algo=None,
    spec_dec_algo=None,
    extra_acc_spec=None,
)


@dataclass(frozen=True, kw_only=True)
class MxAccuracyCase(MxE2ECase):
    """One accuracy canary: an `MxE2ECase` plus the reference key and task."""

    model_name: str
    task: str
    extra_evaluator_kwargs: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class MxEvalSpec:
    """Everything the receiver worker needs to reproduce the reference evaluation."""

    task: str
    num_samples: int
    dataset_path: str
    max_input_len: int
    max_output_len: int
    max_batch_size: int | None = None
    kv_fraction: float = 0.6
    random_seed: int = 0
    apply_chat_template: bool = False
    system_prompt: str | None = None

    @classmethod
    def from_task(cls, task_cls, params, extra: Mapping[str, object]) -> "MxEvalSpec":
        evaluator_kwargs = dict(task_cls.EVALUATOR_KWARGS)
        return cls(
            task=task_cls.__name__,
            num_samples=params.num_samples,
            dataset_path=str(evaluator_kwargs["dataset_path"]),
            max_input_len=task_cls.MAX_INPUT_LEN,
            max_output_len=task_cls.MAX_OUTPUT_LEN,
            max_batch_size=getattr(task_cls, "MAX_BATCH_SIZE", None),
            random_seed=int(evaluator_kwargs.get("random_seed", 0)),
            apply_chat_template=bool(extra.get("apply_chat_template", False)),
            system_prompt=extra.get("system_prompt"),
        )

    def to_argv(self) -> list[str]:
        argv = [
            "--eval-task",
            self.task,
            "--eval-num-samples",
            str(self.num_samples),
            "--eval-dataset-path",
            self.dataset_path,
            "--eval-max-input-len",
            str(self.max_input_len),
            "--eval-max-output-len",
            str(self.max_output_len),
            "--eval-kv-fraction",
            str(self.kv_fraction),
            "--eval-random-seed",
            str(self.random_seed),
        ]
        if self.max_batch_size is not None:
            argv.extend(("--eval-max-batch-size", str(self.max_batch_size)))
        if self.apply_chat_template:
            argv.append("--eval-apply-chat-template")
        if self.system_prompt is not None:
            argv.extend(("--eval-system-prompt", self.system_prompt))
        return argv


_MX_ACCURACY_CASES = (
    pytest.param(
        MxAccuracyCase(
            model_env="TRTLLM_MX_LLAMA3_8B_MODEL",
            default_model_subdir="llama-models-v3/llama-v3-8b-instruct-hf",
            repository_cache_prefix="models--trtllm-mx-acc--llama3-tp1",
            tp_size=1,
            model_name="meta-llama/Meta-Llama-3-8B-Instruct",
            task="MMLU",
        ),
        id="llama3-8b-instruct-mmlu-tp1",
        marks=pytest.mark.skip_less_device(2),
    ),
    pytest.param(
        MxAccuracyCase(
            model_env="TRTLLM_MX_QWEN25_MODEL",
            default_model_subdir="Qwen2.5-7B-Instruct",
            repository_cache_prefix="models--trtllm-mx-acc--qwen25-tp1",
            tp_size=1,
            model_name="Qwen/Qwen2.5-7B-Instruct",
            task="MMLU",
        ),
        id="qwen2.5-7b-instruct-mmlu-tp1",
        marks=pytest.mark.skip_less_device(2),
    ),
    pytest.param(
        MxAccuracyCase(
            model_env="TRTLLM_MX_QWEN3_MODEL",
            default_model_subdir="Qwen3/Qwen3-8B",
            repository_cache_prefix="models--trtllm-mx-acc--qwen3-tp1",
            tp_size=1,
            model_name="Qwen3/Qwen3-8B",
            task="GSM8K",
        ),
        id="qwen3-8b-gsm8k-tp1",
        marks=pytest.mark.skip_less_device(2),
    ),
)
_CASES_BY_ID = {param.id: param.values[0] for param in _MX_ACCURACY_CASES}


def _accuracy_core():
    """Import `accuracy_core` lazily: its class attributes resolve `llm_models_root()` at import."""
    from defs.accuracy import accuracy_core

    return accuracy_core


def _hypothesis_params(case: MxAccuracyCase, task_cls):
    return task_cls(case.model_name).get_hypothesis_testing_params(**_BF16_ACCURACY_SPECS)


def _require_evaluator_prerequisites(task_cls) -> None:
    dataset_path = Path(task_cls.EVALUATOR_KWARGS["dataset_path"])
    if not dataset_path.is_dir():
        skip_or_fail(f"Accuracy dataset directory does not exist: {dataset_path}")
    from tensorrt_llm.evaluate.lm_eval import LmEvalEvaluator

    if issubclass(task_cls.EVALUATOR_CLS, LmEvalEvaluator):
        probe = subprocess.run(
            [
                sys.executable,
                "-c",
                "import importlib.metadata as m, lm_eval; "
                "print('LM_EVAL_VERSION=' + m.version('lm_eval'))",
            ],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        if probe.returncode != 0:
            detail = (probe.stdout + probe.stderr).strip()
            skip_or_fail(
                f"lm_eval is required for {task_cls.__name__} but is unavailable: {detail}"
            )
        print(f"MX accuracy preflight passed: {probe.stdout.strip()}")


def _load_worker_module():
    spec = importlib.util.spec_from_file_location("mx_e2e_worker_under_test", WORKER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _record_metrics(
    record_property, output_dir, case_id: str, metrics: Mapping[str, object]
) -> None:
    for key, value in metrics.items():
        record_property(f"mx_{key}", value)
    print("MX_ACCURACY_METRICS " + json.dumps(metrics, default=str), flush=True)
    if output_dir:
        destination = Path(output_dir) / "model_express_accuracy"
        destination.mkdir(parents=True, exist_ok=True)
        (destination / f"{case_id}.json").write_text(
            json.dumps(metrics, indent=2, default=str) + "\n", encoding="utf-8"
        )


def _assert_donor_receiver_manifests(case: MxAccuracyCase, manifest_dir: Path) -> None:
    """Donor and receiver manifests must agree at the transfer boundary and after finalization."""
    if not manifest_dir.is_dir():
        pytest.fail(f"No weight manifests were written: {manifest_dir} does not exist")
    # The accuracy canary has no HF baseline role, so only the MX roles are required.
    from tensorrt_llm._torch.weight_sharing import (
        WEIGHT_MANIFEST_FILE_PATTERN,
        load_weight_manifest,
    )

    manifests = {}
    for path in sorted(manifest_dir.iterdir()):
        match = WEIGHT_MANIFEST_FILE_PATTERN.match(path.name) if path.is_file() else None
        if match is None:
            pytest.fail(f"Unexpected entry in the weight manifest directory: {path}")
        manifests[(match["family"], match["role"], int(match["rank"]))] = load_weight_manifest(path)
    expected = {
        (family, role, rank)
        for family in ("final", "transfer")
        for role in MX_ROLES
        for rank in range(case.tp_size)
    }
    if set(manifests) != expected:
        pytest.fail(
            "Weight manifest set differs from the expected keys: "
            f"absent={sorted(expected - set(manifests))}, "
            f"extra={sorted(set(manifests) - expected)}"
        )
    for rank in range(case.tp_size):
        assert manifests[("final", "receiver", rank)].context.get("weights_preloaded") is True
        for family, kinds in (("transfer", TRANSFER_TIER_KINDS), ("final", ("param", "buffer"))):
            diff = compare_weight_manifests(
                manifests[(family, "donor", rank)],
                manifests[(family, "receiver", rank)],
                kinds=kinds,
                exempt_patterns=case.final_manifest_exempt_patterns if family == "final" else (),
            )
            if not diff.is_empty:
                pytest.fail(
                    diff.describe(f"donor@{family} rank{rank}", f"receiver@{family} rank{rank}")
                )


def test_mx_accuracy_cases_have_references() -> None:
    """CPU-only guard: every canary has a bare BF16 reference entry and a valid worker argv."""
    for case_id, case in _CASES_BY_ID.items():
        references = yaml.safe_load(
            (_REFERENCES_DIR / _TASK_REFERENCE_FILES[case.task]).read_text(encoding="utf-8")
        )
        entries = references.get(case.model_name)
        assert entries, f"{case_id}: no {case.task} reference entries for {case.model_name!r}"
        bare = [
            entry
            for entry in entries
            if set(entry) <= {"accuracy", "dtype"} and entry.get("dtype", "auto") == "auto"
        ]
        assert bare, f"{case_id}: no bare BF16 {case.task} reference entry for {case.model_name!r}"
        assert isinstance(bare[0]["accuracy"], (int, float))

    worker = _load_worker_module()
    spec = MxEvalSpec(
        task="GSM8K",
        num_samples=1319,
        dataset_path="/datasets/gsm8k",
        max_input_len=4096,
        max_output_len=256,
        max_batch_size=32,
    )
    argv = ["--role", "receiver", "--model", "/snapshot", "--tp-size", "1", "--output", "/o.json"]
    args = worker._parse_args(argv + ["--mx-url", "http://mx:8001"] + spec.to_argv())
    assert args.eval_task == "GSM8K"
    assert args.eval_num_samples == 1319
    assert args.eval_max_input_len == 4096 and args.eval_max_output_len == 256
    assert args.eval_max_batch_size == 32 and args.eval_kv_fraction == 0.6
    with pytest.raises(SystemExit):
        worker._parse_args(
            ["--role", "donor", "--model", "/m", "--tp-size", "1", "--output", "/o"]
            + spec.to_argv()
        )


@pytest.mark.parametrize("case", _MX_ACCURACY_CASES)
def test_mx_receiver_accuracy(
    case: MxAccuracyCase, tmp_path: Path, record_property, output_dir, request
) -> None:
    """Evaluate the reference task on an MX receiver and assert the hypothesis-testing threshold."""
    accuracy_core = _accuracy_core()
    task_cls = accuracy_core.get_accuracy_task(case.task)
    params = _hypothesis_params(case, task_cls)  # fails fast on a missing reference entry

    mx_url, gpu_ids = require_mx_environment(case.tp_size * 2)
    _require_evaluator_prerequisites(task_cls)
    model_path = resolve_model_path(case)
    timeout_s = int(os.environ.get("TRTLLM_MX_E2E_TIMEOUT_S", "1200"))
    layout = MxRunLayout(tmp_path)
    eval_spec = MxEvalSpec.from_task(task_cls, params, case.extra_evaluator_kwargs)

    donor_snapshot = build_canonical_snapshot(case, model_path, tmp_path)
    receiver_snapshot = build_metadata_only_snapshot(donor_snapshot, tmp_path)
    assert ArtifactIdentity.from_checkpoint(receiver_snapshot) == ArtifactIdentity.from_checkpoint(
        donor_snapshot
    )
    donor_gpu_ids = gpu_ids[: case.tp_size]
    receiver_gpu_ids = gpu_ids[case.tp_size :]

    try:
        with donor_session(
            command=worker_command(
                role="donor",
                model_path=donor_snapshot,
                tp_size=case.tp_size,
                output_path=layout.output("donor"),
                mx_url=mx_url,
                ready_file=layout.donor_ready,
                stop_file=layout.donor_stop,
                max_serve_seconds=timeout_s + 120,
            ),
            environment=worker_environment(
                donor_gpu_ids, layout.transfer_logs("donor"), layout.manifest_dir
            ),
            log_path=layout.log("donor"),
            ready_file=layout.donor_ready,
            stop_file=layout.donor_stop,
            timeout_s=timeout_s,
        ) as donor:
            run_worker(
                worker_command(
                    role="receiver",
                    model_path=receiver_snapshot,
                    tp_size=case.tp_size,
                    output_path=layout.output("receiver"),
                    mx_url=mx_url,
                    extra_args=eval_spec.to_argv(),
                ),
                worker_environment(
                    receiver_gpu_ids, layout.transfer_logs("receiver"), layout.manifest_dir
                ),
                layout.log("receiver"),
                timeout_s,
            )

        assert donor.returncode == 0, (
            f"MX donor exited with status {donor.returncode}\n{log_tail(layout.log('donor'))}"
        )
        donor_payload = load_payload(layout.output("donor"))
        receiver_payload = load_payload(layout.output("receiver"))
        assert donor_payload["server_query_timeout_s"] == 0

        assert_transfer_evidence(
            case, layout.log("receiver"), layout.transfer_logs("receiver"), receiver_payload
        )
        _assert_donor_receiver_manifests(case, layout.manifest_dir)

        assert receiver_payload["eval_task"] == case.task
        assert receiver_payload["eval_num_samples"] == params.num_samples
        score = receiver_payload["score"]
        assert isinstance(score, (int, float)) and math.isfinite(score), (
            f"MX receiver reported a non-finite score: {score!r}"
        )
        score = float(score)
        print(params.report(score))
        _record_metrics(
            record_property,
            output_dir,
            request.node.callspec.id,
            {
                "model_name": case.model_name,
                "eval_task": case.task,
                "eval_num_samples": params.num_samples,
                "tp_size": case.tp_size,
                "donor_load_seconds": donor_payload.get("load_seconds"),
                "receiver_load_seconds": receiver_payload.get("load_seconds"),
                "receiver_self_check_seconds": receiver_payload.get("self_check_seconds"),
                "eval_seconds": receiver_payload.get("eval_seconds"),
                "score": score,
                "threshold": params.threshold,
                "ref_accuracy": params.ref_accuracy,
            },
        )
        params.assert_passing(score)
    finally:
        archive_run_artifacts(output_dir, request.node.callspec.id, layout)
