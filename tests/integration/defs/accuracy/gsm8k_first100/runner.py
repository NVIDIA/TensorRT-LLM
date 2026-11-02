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
"""One explicitly triggered first-100 CI run through the public GSM8K evaluator.

Nothing is read, staged or launched before the explicit trigger is checked; without it the
run skips. Once triggered, every missing prerequisite (hardware, timeout, inputs, recipe,
criterion in regression mode) fails the run instead of skipping it.

The CI part verifies the staged inputs and the hardware, materializes the recipe and loads
the regression criterion; the public evaluator (``examples/llm-api/gsm8k_eval``, the code
behind ``examples/llm-api/eval_gsm8k.py``) then launches and owns ``trtllm-serve``, asks the
first 100 pinned questions with ``max_batch_size`` requests in flight, records every reply in
``results.jsonl``, stops the server and writes ``summary.json``. The results directory is
``<--output-dir>/nvfp4_serving_gsm8k_first100`` (a fresh temporary directory without
``--output-dir``); running again with the same directory and run identity resumes.
"""

from __future__ import annotations

import dataclasses
import importlib.metadata
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Callable, Mapping, NoReturn, Optional

import pytest

from . import deployment
from .public import protocol
from .public import runner as evaluator

RESULTS_DIRNAME = "nvfp4_serving_gsm8k_first100"
#: Seconds of the pytest timeout kept for teardown and the summary.
TEARDOWN_RESERVE_S = 300
SKIP_REASON = (
    f"explicit trigger only: set {deployment.GUARD_ENV}=1 to run the first-100 GSM8K "
    "serving qualification"
)


@dataclasses.dataclass
class RunDeps(evaluator.RunDeps):
    """The evaluator's process/HTTP/clock entry points plus the GPU query (faked in tests)."""

    query_gpus: Callable[[Mapping[str, str]], list[dict[str, str]]] = deployment.query_gpus


def _fail(message: str) -> NoReturn:
    pytest.fail(f"first-100 GSM8K qualification: {message}", pytrace=False)


def _enabled(environ: Mapping[str, str]) -> bool:
    try:
        return deployment.explicit_trigger(environ)
    except ValueError as error:
        _fail(str(error))


def require_hardware_when_enabled(
    environ: Mapping[str, str], deps: Optional[RunDeps] = None
) -> None:
    """Fail, never skip, an explicitly triggered run without four GB300 (SM103) GPUs."""
    if not _enabled(environ):
        return
    deps = deps or RunDeps()
    try:
        gpus = deps.query_gpus(environ)
    except (OSError, RuntimeError, subprocess.SubprocessError) as error:
        _fail(f"cannot identify the GPUs: {error}")
    problems = deployment.check_hardware(gpus)
    if problems:
        _fail("; ".join(problems))


def _results_dir(output_dir: Optional[str]) -> Path:
    if not output_dir:
        return Path(tempfile.mkdtemp(prefix=f"{RESULTS_DIRNAME}-"))
    path = Path(output_dir) / RESULTS_DIRNAME
    path.mkdir(parents=True, exist_ok=True)
    return path


def _package_version(name: str) -> Optional[str]:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _source_revision(repo: str) -> Optional[str]:
    try:
        result = subprocess.run(
            ["git", "-C", repo, "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def _content_identity(description: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "files": {name: entry["sha256"] for name, entry in description["files"].items()},
        "weights_inventory_sha256": description["weights"]["inventory_sha256"],
        "num_hidden_layers": description["num_hidden_layers"],
    }


def run_first100(
    *,
    environ: Mapping[str, str],
    models_root: Callable[[], str],
    repo_root: Callable[[], str],
    timeout_s: Optional[float],
    output_dir: Optional[str],
    deps: Optional[RunDeps] = None,
    manifest_path: Path = protocol.MANIFEST_PATH,
    criterion_path: Path = deployment.CRITERION_PATH,
) -> dict[str, Any]:
    """Run or resume the qualification and return its summary; skip unless triggered.

    Args:
        environ: The process environment (trigger, mode and staged input paths).
        models_root: Returns the models root; called only after the trigger check.
        repo_root: Returns the repository root holding the curated recipe.
        timeout_s: The test's pytest timeout in seconds; a finite value is required.
        output_dir: The pytest ``--output-dir``; results go to a subdirectory of it.
        deps: Process/HTTP/GPU/clock entry points; the real ones by default.
        manifest_path: The frozen item manifest of the public protocol.
        criterion_path: The installed regression criterion (regression mode only).
    """
    if not _enabled(environ):
        pytest.skip(SKIP_REASON)
    deps = deps or RunDeps()
    try:
        mode = deployment.run_mode(environ)
    except ValueError as error:
        _fail(str(error))
    if timeout_s is None or timeout_s <= TEARDOWN_RESERVE_S + 60:
        _fail(
            f"a finite pytest timeout above {TEARDOWN_RESERVE_S + 60} s is required (test-db "
            f"TIMEOUT or --timeout), got {timeout_s!r}"
        )
    manifest_sha256 = protocol.sha256_file(manifest_path)
    criterion = None
    if mode == "regression":
        try:
            criterion = deployment.load_criterion(criterion_path, manifest_sha256=manifest_sha256)
        except deployment.CriterionError as error:
            _fail(f"regression mode refused: {error}")

    try:
        inputs = deployment.resolve_inputs(environ, models_root())
        target = deployment.describe_checkpoint(
            inputs.target, deployment.TARGET_FILES, deployment.TARGET_OPTIONAL_FILES
        )
        drafter = deployment.describe_checkpoint(
            inputs.drafter, deployment.DRAFTER_FILES, deployment.DRAFTER_OPTIONAL_FILES
        )
        kv_scales = deployment.describe_kv_scales(
            inputs.kv_scales, target["num_hidden_layers"], drafter["num_hidden_layers"]
        )
        gpus = deps.query_gpus(environ)
    except (AssertionError, OSError, RuntimeError, ValueError, subprocess.SubprocessError) as error:
        _fail(f"inputs are not usable: {type(error).__name__}: {error}")
    hardware_problems = deployment.check_hardware(gpus)
    if hardware_problems:
        _fail("; ".join(hardware_problems))
    gpus = gpus[: deployment.REQUIRED_GPU_COUNT]

    try:
        recipe_text, recipe = deployment.materialize_recipe(
            os.path.join(repo_root(), deployment.RECIPE_RELATIVE_PATH), inputs.drafter
        )
    except (OSError, ValueError) as error:
        _fail(f"the recipe is not usable: {error}")

    identity_extra = {
        "recipe_template_sha256": recipe["template_sha256"],
        "serve_command": deployment.serve_command("<TARGET>", "<CONFIG>"),
        "server_environment": {
            **deployment.SERVER_ENV,
            deployment.KV_SCALES_SERVER_ENV: "<KV_SCALES>",
        },
        "target": _content_identity(target),
        "drafter": _content_identity(drafter),
        "kv_scales": kv_scales,
        "gpus": [[gpu["name"], gpu["compute_cap"]] for gpu in gpus],
        "tensorrt_llm": _package_version("tensorrt_llm"),
        "source_revision": _source_revision(repo_root()),
    }
    if criterion is not None:
        mismatched = deployment.criterion_mismatches(
            criterion,
            {
                "protocol_id": protocol.PROTOCOL_ID,
                "manifest_sha256": manifest_sha256,
                "request_fields": protocol.SAMPLING_FIELDS,
                **identity_extra,
            },
        )
        if mismatched:
            _fail(
                "regression mode refused: the installed criterion was set for a different "
                f"{', '.join(mismatched)}"
            )
    results_dir = _results_dir(output_dir)
    config_path = results_dir / "recipe.yaml"
    config_path.write_text(recipe_text, encoding="utf-8")
    managed = evaluator.ManagedServer(
        argv=tuple(deployment.serve_command(inputs.target, str(config_path))),
        env=deployment.server_environment(environ, inputs.kv_scales),
        gpu_ids=tuple(gpu["index"] for gpu in gpus),
        settings={"recipe": recipe},
    )
    try:
        summary = evaluator.run(
            dataset_dir=inputs.dataset_dir,
            output_dir=results_dir,
            num_questions=protocol.NUM_QUESTIONS,
            concurrency=recipe["max_batch_size"],
            time_budget_s=float(timeout_s) - TEARDOWN_RESERVE_S,
            managed=managed,
            max_seq_len=recipe["max_seq_len"],
            min_correct=criterion["min_correct"] if criterion is not None else None,
            identity_extra=identity_extra,
            attempt_extra={
                "mode": mode,
                "pytest_timeout_s": timeout_s,
                "staged_inputs": inputs.relative,
                "target": target,
                "drafter": drafter,
                "server_environment": {
                    **deployment.SERVER_ENV,
                    deployment.KV_SCALES_SERVER_ENV: inputs.relative[deployment.KV_SCALES_ENV],
                },
                "gpus": gpus,
            },
            manifest_path=manifest_path,
            deps=deps,
        )
    except evaluator.EvaluationError as error:
        _fail(str(error))
    problems_found = list(summary["verdict"]["problems"])
    if problems_found:
        _fail("; ".join(problems_found) + f" (results: {results_dir})")
    return summary
