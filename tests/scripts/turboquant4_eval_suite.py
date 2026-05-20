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
"""Run TurboQuant4 eval canaries and production gates.

This is an orchestration wrapper around the narrower TurboQuant4 scripts. It is
deliberately conservative: proxy checks can pass, but ``production_ready`` is
only true when the native CUDA smoke and both TensorRT-LLM evals actually run
and pass in the current environment.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

PRODUCTION_STEP_NAMES = {
    "native_smoke",
    "trtllm_context_ppl",
    "trtllm_generation_quality",
}


@dataclass
class Step:
    name: str
    category: str
    command: list[str]
    output_json: Path | None = None


@dataclass
class StepResult:
    name: str
    category: str
    command: list[str]
    passed: bool
    returncode: int
    duration_seconds: float
    output_json: str | None
    output: dict[str, Any] | None
    stdout_tail: str
    stderr_tail: str


def _tail(text: str, max_chars: int = 4000) -> str:
    if len(text) <= max_chars:
        return text
    return text[-max_chars:]


def _run_step(step: Step, env: dict[str, str]) -> StepResult:
    if step.output_json is not None and step.output_json.exists():
        step.output_json.unlink()

    start = time.monotonic()
    process = subprocess.run(
        step.command,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    duration = time.monotonic() - start
    output = None
    if step.output_json is not None and step.output_json.exists():
        with step.output_json.open() as f:
            output = json.load(f)
    passed = process.returncode == 0
    if step.output_json is not None:
        passed = passed and output is not None and output.get("passed") is True
    return StepResult(
        name=step.name,
        category=step.category,
        command=step.command,
        passed=passed,
        returncode=process.returncode,
        duration_seconds=duration,
        output_json=None if step.output_json is None else str(step.output_json),
        output=output,
        stdout_tail=_tail(process.stdout),
        stderr_tail=_tail(process.stderr),
    )


def _child_env() -> dict[str, str]:
    env = os.environ.copy()
    repo_root = Path(__file__).resolve().parents[2]
    existing_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        str(repo_root)
        if not existing_pythonpath
        else f"{repo_root}{os.pathsep}{existing_pythonpath}"
    )
    return env


def _production_ready(results: list[StepResult]) -> bool:
    passed_by_name = {result.name: result.passed for result in results}
    return all(passed_by_name.get(name, False) for name in PRODUCTION_STEP_NAMES)


def _step_summary(results: list[StepResult]) -> dict[str, Any]:
    return {
        "num_steps": len(results),
        "passed_steps": sum(result.passed for result in results),
        "failed_steps": sum(not result.passed for result in results),
        "production_steps_required": sorted(PRODUCTION_STEP_NAMES),
        "production_steps_passed": sorted(
            result.name
            for result in results
            if result.name in PRODUCTION_STEP_NAMES and result.passed
        ),
    }


def _environment_summary() -> dict[str, Any]:
    summary: dict[str, Any] = {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "nvidia_smi": shutil.which("nvidia-smi"),
    }
    try:
        import torch

        summary["torch_version"] = torch.__version__
        summary["torch_cuda_available"] = torch.cuda.is_available()
        summary["torch_cuda_device_count"] = torch.cuda.device_count()
        if torch.cuda.is_available():
            summary["torch_cuda_devices"] = [
                torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())
            ]
    except Exception as exc:  # noqa: BLE001 - diagnostic metadata only.
        summary["torch_error"] = repr(exc)
    return summary


def _add_model_args(command: list[str], args: argparse.Namespace) -> None:
    command.extend(["--model", args.model])
    if args.tokenizer is not None:
        command.extend(["--tokenizer", args.tokenizer])
    if args.trust_remote_code:
        command.append("--trust-remote-code")


def _add_optional_path_arg(command: list[str], name: str, value: Path | None) -> None:
    if value is not None:
        command.extend([name, str(value)])


def _add_optional_value_arg(command: list[str], name: str, value: int | float | None) -> None:
    if value is not None:
        command.extend([name, str(value)])


def _build_steps(args: argparse.Namespace) -> list[Step]:
    script_dir = Path(__file__).resolve().parent
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    steps = [
        Step(
            name="reference_smoke",
            category="reference",
            command=[sys.executable, str(script_dir / "turboquant4_smoke.py")],
        )
    ]
    if args.include_native_smoke or args.require_production:
        steps.append(
            Step(
                name="native_smoke",
                category="production",
                command=[sys.executable, str(script_dir / "turboquant4_smoke.py"), "--native"],
            )
        )

    if args.include_hf_proxy:
        hf_ppl_json = output_dir / "turboquant4_hf_ppl_eval.json"
        hf_ppl_cmd = [
            sys.executable,
            str(script_dir / "turboquant4_hf_ppl_eval.py"),
            "--device",
            args.hf_device,
            "--dtype",
            args.hf_dtype,
            "--max-tokens-per-text",
            str(args.max_tokens_per_text),
            "--kv-cache-quantization",
            args.hf_kv_cache_quantization,
            "--output-json",
            str(hf_ppl_json),
        ]
        _add_optional_path_arg(hf_ppl_cmd, "--text-file", args.text_file)
        _add_optional_value_arg(hf_ppl_cmd, "--limit-texts", args.limit_texts)
        _add_optional_value_arg(
            hf_ppl_cmd,
            "--long-context-repeats",
            args.long_context_repeats,
        )
        _add_optional_value_arg(
            hf_ppl_cmd,
            "--max-nll-delta-per-token",
            args.max_nll_delta_per_token,
        )
        _add_optional_value_arg(
            hf_ppl_cmd,
            "--max-relative-ppl-delta",
            args.max_relative_ppl_delta,
        )
        _add_model_args(hf_ppl_cmd, args)
        steps.append(
            Step(
                name="hf_proxy_ppl",
                category="proxy",
                command=hf_ppl_cmd,
                output_json=hf_ppl_json,
            )
        )

        hf_cache_json = output_dir / "turboquant4_hf_cache_eval.json"
        hf_cache_cmd = [
            sys.executable,
            str(script_dir / "turboquant4_hf_cache_eval.py"),
            "--device",
            args.hf_device,
            "--dtype",
            args.hf_dtype,
            "--max-new-tokens",
            str(args.max_new_tokens),
            "--kv-cache-quantization",
            args.hf_kv_cache_quantization,
            "--output-json",
            str(hf_cache_json),
        ]
        _add_optional_path_arg(hf_cache_cmd, "--prompt-file", args.prompt_file)
        _add_optional_value_arg(hf_cache_cmd, "--limit-prompts", args.limit_prompts)
        _add_optional_value_arg(
            hf_cache_cmd,
            "--long-context-repeats",
            args.long_context_repeats,
        )
        _add_optional_value_arg(
            hf_cache_cmd,
            "--min-exact-token-match-rate",
            args.min_exact_token_match_rate,
        )
        _add_optional_value_arg(
            hf_cache_cmd,
            "--min-token-prefix-agreement",
            args.min_token_prefix_agreement,
        )
        _add_optional_value_arg(
            hf_cache_cmd,
            "--max-reference-regressions",
            args.max_reference_regressions,
        )
        _add_optional_value_arg(
            hf_cache_cmd,
            "--max-decode-nll-delta-per-token",
            args.max_decode_nll_delta_per_token,
        )
        _add_model_args(hf_cache_cmd, args)
        steps.append(
            Step(
                name="hf_proxy_generation",
                category="proxy",
                command=hf_cache_cmd,
                output_json=hf_cache_json,
            )
        )

    if args.include_trtllm or args.require_production:
        trtllm_ppl_json = output_dir / "turboquant4_trtllm_ppl_eval.json"
        trtllm_ppl_cmd = [
            sys.executable,
            str(script_dir / "turboquant4_trtllm_ppl_eval.py"),
            "--dtype",
            args.trtllm_dtype,
            "--attn-backend",
            args.attn_backend,
            "--output-json",
            str(trtllm_ppl_json),
        ]
        _add_optional_path_arg(trtllm_ppl_cmd, "--text-file", args.text_file)
        _add_optional_value_arg(trtllm_ppl_cmd, "--limit-texts", args.limit_texts)
        _add_optional_value_arg(
            trtllm_ppl_cmd,
            "--long-context-repeats",
            args.long_context_repeats,
        )
        _add_optional_value_arg(
            trtllm_ppl_cmd,
            "--max-nll-delta-per-token",
            args.max_nll_delta_per_token,
        )
        _add_optional_value_arg(
            trtllm_ppl_cmd,
            "--max-relative-ppl-delta",
            args.max_relative_ppl_delta,
        )
        _add_model_args(trtllm_ppl_cmd, args)
        steps.append(
            Step(
                name="trtllm_context_ppl",
                category="production",
                command=trtllm_ppl_cmd,
                output_json=trtllm_ppl_json,
            )
        )

        trtllm_quality_json = output_dir / "turboquant4_quality_eval.json"
        trtllm_quality_cmd = [
            sys.executable,
            str(script_dir / "turboquant4_quality_eval.py"),
            "--dtype",
            args.trtllm_dtype,
            "--attn-backend",
            args.attn_backend,
            "--max-new-tokens",
            str(args.max_new_tokens),
            "--max-decode-nll-delta-per-token",
            str(args.max_decode_nll_delta_per_token),
            "--output-json",
            str(trtllm_quality_json),
        ]
        _add_optional_path_arg(trtllm_quality_cmd, "--prompt-file", args.prompt_file)
        _add_optional_value_arg(trtllm_quality_cmd, "--limit-prompts", args.limit_prompts)
        _add_optional_value_arg(
            trtllm_quality_cmd,
            "--long-context-repeats",
            args.long_context_repeats,
        )
        _add_optional_value_arg(
            trtllm_quality_cmd,
            "--min-exact-token-match-rate",
            args.min_exact_token_match_rate,
        )
        _add_optional_value_arg(
            trtllm_quality_cmd,
            "--min-token-prefix-agreement",
            args.min_token_prefix_agreement,
        )
        _add_optional_value_arg(
            trtllm_quality_cmd,
            "--max-reference-regressions",
            args.max_reference_regressions,
        )
        _add_model_args(trtllm_quality_cmd, args)
        steps.append(
            Step(
                name="trtllm_generation_quality",
                category="production",
                command=trtllm_quality_cmd,
                output_json=trtllm_quality_json,
            )
        )

    return steps


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="distilgpt2", help="HF model name or local path.")
    parser.add_argument("--tokenizer", default=None, help="Optional tokenizer name or path.")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("turboquant4_eval_suite"),
        help="Directory for child eval JSON files and the suite manifest.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Suite manifest path. Defaults to <output-dir>/suite.json.",
    )
    parser.add_argument("--include-hf-proxy", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--include-native-smoke",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--include-trtllm", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--require-production",
        action="store_true",
        help=(
            "Require native CUDA smoke and TensorRT-LLM evals to pass. This is "
            "the mode to use before claiming production readiness."
        ),
    )
    parser.add_argument(
        "--continue-on-failure",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Continue running later steps after an earlier step fails.",
    )
    parser.add_argument("--hf-device", default="cpu", help="HF proxy device: cpu, cuda, mps, auto.")
    parser.add_argument(
        "--hf-dtype",
        default="float32",
        choices=["float32", "float16", "bfloat16"],
    )
    parser.add_argument(
        "--hf-kv-cache-quantization",
        default="both",
        choices=["both", "keys-only", "values-only"],
        help=(
            "Which cached tensors to quantize in HF proxy evals. TensorRT-LLM "
            "production evals always test the integrated TurboQuant4 KV path."
        ),
    )
    parser.add_argument("--trtllm-dtype", default="float16")
    parser.add_argument("--attn-backend", default="TRTLLM")
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--max-tokens-per-text", type=int, default=96)
    parser.add_argument(
        "--long-context-repeats",
        type=int,
        default=None,
        help=(
            "Optional long-context repeat count forwarded to PPL and TRT-LLM "
            "generation evals. Child script defaults are used when unset."
        ),
    )
    parser.add_argument(
        "--text-file",
        type=Path,
        default=None,
        help="JSONL texts for PPL evals; forwarded to HF proxy and TRT-LLM PPL evals.",
    )
    parser.add_argument(
        "--prompt-file",
        type=Path,
        default=None,
        help="JSONL prompts for generation evals; forwarded to HF proxy and TRT-LLM generation evals.",
    )
    parser.add_argument("--limit-texts", type=int, default=None)
    parser.add_argument("--limit-prompts", type=int, default=None)
    parser.add_argument(
        "--max-nll-delta-per-token",
        type=float,
        default=None,
        help="Optional PPL eval threshold forwarded to child PPL evals.",
    )
    parser.add_argument(
        "--max-relative-ppl-delta",
        type=float,
        default=None,
        help="Optional PPL eval threshold forwarded to child PPL evals.",
    )
    parser.add_argument(
        "--min-exact-token-match-rate",
        type=float,
        default=None,
        help="Optional generation eval threshold forwarded to child generation evals.",
    )
    parser.add_argument(
        "--min-token-prefix-agreement",
        type=float,
        default=None,
        help="Optional generation eval threshold forwarded to child generation evals.",
    )
    parser.add_argument(
        "--max-reference-regressions",
        type=int,
        default=None,
        help="Optional generation eval threshold forwarded to child generation evals.",
    )
    parser.add_argument("--max-decode-nll-delta-per-token", type=float, default=0.05)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.max_new_tokens <= 0:
        raise ValueError("--max-new-tokens must be positive.")
    if args.max_tokens_per_text < 2:
        raise ValueError("--max-tokens-per-text must be at least 2.")
    if args.long_context_repeats is not None and args.long_context_repeats < 0:
        raise ValueError("--long-context-repeats must be non-negative when provided.")
    if args.limit_texts is not None and args.limit_texts <= 0:
        raise ValueError("--limit-texts must be positive when provided.")
    if args.limit_prompts is not None and args.limit_prompts <= 0:
        raise ValueError("--limit-prompts must be positive when provided.")
    if args.require_production:
        args.include_native_smoke = True
        args.include_trtllm = True

    suite_json = args.output_json or args.output_dir / "suite.json"
    env = _child_env()
    steps = _build_steps(args)
    results = []
    for step in steps:
        print(f"[turboquant4-suite] running {step.name}: {' '.join(step.command)}")
        result = _run_step(step, env)
        results.append(result)
        print(
            f"[turboquant4-suite] {step.name}: "
            f"{'passed' if result.passed else 'failed'} in {result.duration_seconds:.1f}s"
        )
        if not result.passed and not args.continue_on_failure:
            break

    production_ready = _production_ready(results)
    all_run_steps_passed = all(result.passed for result in results)
    passed = all_run_steps_passed and (production_ready if args.require_production else True)
    manifest = {
        "passed": passed,
        "production_ready": production_ready,
        "proxy_only": not any(result.name in PRODUCTION_STEP_NAMES for result in results),
        "require_production": args.require_production,
        "model": args.model,
        "hf_kv_cache_quantization": args.hf_kv_cache_quantization,
        "long_context_repeats": args.long_context_repeats,
        "environment": _environment_summary(),
        "summary": _step_summary(results),
        "steps": [asdict(result) for result in results],
    }
    suite_json.parent.mkdir(parents=True, exist_ok=True)
    with suite_json.open("w") as f:
        json.dump(manifest, f, indent=2)

    print(
        json.dumps(
            {
                "passed": passed,
                "production_ready": production_ready,
                "summary": manifest["summary"],
                "suite_json": str(suite_json),
            },
            indent=2,
        )
    )
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
