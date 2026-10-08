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
"""ROCm generation, serving, benchmarking and local RDNA4 correctness validation."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

from trtllm_profile import active_session, enable_from_argv

from .runtime import diagnostics
from .sampling import SamplingParams


def _positive(value: str) -> int:
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return result


def _model_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("model", nargs="?", help="HF model ID or local checkpoint directory")
    parser.add_argument("--model", dest="model_option", help="Alternative to the positional model")
    parser.add_argument("--tokenizer")
    parser.add_argument(
        "--device", default="cuda:0", help="HIP logical device; explicit cpu for reference testing"
    )
    parser.add_argument(
        "--dtype", choices=("auto", "float32", "float16", "bfloat16"), default="auto"
    )
    parser.add_argument("--kernels", choices=("torch", "hip"), default="torch")
    parser.add_argument("--attn-backend", choices=("sdpa", "eager", "hip"), default="sdpa")
    parser.add_argument("--max-batch-size", type=_positive, default=1)
    parser.add_argument("--max-seq-len", type=_positive)
    parser.add_argument("--revision")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument(
        "--profile", action="store_true", help="Record component, per-file and utilization reports"
    )
    parser.add_argument("--profile-output", help="Report filename prefix (requires --profile)")
    parser.add_argument(
        "--profile-interval", type=float, help="Resource sampling interval in seconds"
    )
    parser.add_argument(
        "--profile-max-ops", type=_positive, help="Maximum timed dispatch operations"
    )


def _sampling_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--max-tokens", type=_positive, default=64)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--top-p", type=float, default=1)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--stop", action="append")
    parser.add_argument("--ignore-eos", action="store_true")


def _llm(options: argparse.Namespace):
    from .llm import LLM

    model = options.model_option or options.model
    if not model:
        raise ValueError("Provide a model ID/path using the positional model or --model")
    if options.model_option and options.model and options.model_option != options.model:
        raise ValueError("Do not specify two different model sources")
    return LLM(
        model=model,
        tokenizer=options.tokenizer,
        device=options.device,
        dtype=options.dtype,
        max_batch_size=options.max_batch_size,
        max_seq_len=options.max_seq_len,
        attn_backend=options.attn_backend,
        kernels=options.kernels,
        revision=options.revision,
        trust_remote_code=options.trust_remote_code,
        local_files_only=options.local_files_only,
    )


def _params(options: argparse.Namespace) -> SamplingParams:
    return SamplingParams(
        max_tokens=options.max_tokens,
        temperature=options.temperature,
        top_p=options.top_p,
        top_k=options.top_k,
        seed=options.seed,
        stop=options.stop,
        ignore_eos=options.ignore_eos,
    )


def _write_json(output: str | None, report: dict) -> None:
    text = json.dumps(report, indent=2) + "\n"
    print(text)
    if output:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)


def _prompts(options: argparse.Namespace) -> list[str] | list[list[int]]:
    if options.dataset:
        prompts = []
        with Path(options.dataset).open() as file:
            for line_number, line in enumerate(file, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                prompt = row.get("prompt", row.get("input_ids"))
                if not isinstance(prompt, (str, list)):
                    raise ValueError(
                        f"Dataset line {line_number} requires prompt text or input_ids"
                    )
                prompts.append(prompt)
        if not prompts:
            raise ValueError("Dataset is empty")
        return prompts
    return options.prompt or ["The capital of France is"]


def _benchmark(options: argparse.Namespace) -> None:
    if options.warmup < 0:
        raise ValueError("warmup must be non-negative")
    prompts = _prompts(options)
    params = _params(options)
    with _llm(options) as engine:
        for _ in range(options.warmup):
            engine.generate(prompts, params)
        runs = []
        for _ in range(options.iterations):
            engine.generate(prompts, params)
            runs.append(dict(engine.last_stats))
        durations = sorted(run["wall_s"] for run in runs)
        position = 0.95 * (len(durations) - 1)
        lower = int(position)
        upper = min(lower + 1, len(durations) - 1)
        p95 = durations[lower] + (durations[upper] - durations[lower]) * (position - lower)
        wall = sum(durations)
        outputs = sum(run["output_tokens"] for run in runs)
        report = {
            "model": engine.model_id,
            "device": str(engine.device),
            "dtype": str(engine.dtype),
            "kernels": options.kernels,
            "mode": options.benchmark,
            "warmup_iterations_excluded": options.warmup,
            "measured_iterations": options.iterations,
            "requests_per_iteration": len(prompts),
            "wall_s": wall,
            "input_tokens": sum(run["input_tokens"] for run in runs),
            "output_tokens": outputs,
            "output_tokens_per_s": outputs / wall if wall else 0,
            "batch_latency_ms": {
                "min": durations[0] * 1000,
                "median": statistics.median(durations) * 1000,
                "p95": p95 * 1000,
            },
            "runs": runs,
            "semantics": (
                "Serial batches, including tokenize/transfer/generate/detokenize; "
                "not in-flight batching or TTFT"
            ),
        }
        _write_json(options.output, report)


def _serve(options: argparse.Namespace) -> None:
    import uvicorn

    from .server import create_app

    engine = _llm(options)
    uvicorn.run(
        create_app(engine, served_model_name=options.served_model_name),
        host=options.host,
        port=options.port,
        workers=1,
    )


def _generate(options: argparse.Namespace) -> None:
    with _llm(options) as engine:
        for result in engine.generate(options.prompt or ["Hello, my name is"], _params(options)):
            print(
                json.dumps(
                    {
                        "request_id": result.request_id,
                        "prompt": result.prompt,
                        "outputs": [
                            {
                                "text": output.text,
                                "token_ids": output.token_ids,
                                "finish_reason": output.finish_reason,
                            }
                            for output in result.outputs
                        ],
                    }
                )
            )
        print(json.dumps({"generation_stats": engine.last_stats}))


def _parser(command: str, program: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=program)
    _model_arguments(parser)
    if command in ("generate", "bench"):
        _sampling_arguments(parser)
        parser.add_argument("--prompt", action="append")
    if command == "bench":
        parser.add_argument("--benchmark", choices=("throughput", "latency"), default="throughput")
        parser.add_argument(
            "--dataset", help="JSONL with prompt text or input_ids; output length is --max-tokens"
        )
        parser.add_argument("--warmup", type=int, default=1)
        parser.add_argument("--iterations", type=_positive, default=3)
        parser.add_argument("--output", help="Write benchmark JSON")
    if command == "serve":
        parser.add_argument(
            "--host", default="127.0.0.1", help="Bind address; defaults to loopback only"
        )
        parser.add_argument("--port", type=_positive, default=8000)
        parser.add_argument("--served-model-name")
    return parser


def _run(command: str, argv: list[str] | None, program: str) -> None:
    arguments = sys.argv if argv is None else [program, *argv]
    profiler = enable_from_argv(arguments)
    parser = _parser(command, program)
    options = parser.parse_args(arguments[1:])
    if command == "bench" and options.model_option and options.model in ("throughput", "latency"):
        options.benchmark, options.model = options.model, None
    # Profiling modifiers without --profile are probably a user mistake.
    if not profiler and any(
        getattr(options, name, None)
        for name in (
            "profile_output",
            "profile_interval",
            "profile_max_ops",
        )
    ):
        parser.error("Profile options require --profile")
    try:
        {"generate": _generate, "bench": _benchmark, "serve": _serve}[command](options)
    except (ValueError, NotImplementedError, RuntimeError) as error:
        if profiler is not None:
            profiler.status = f"failed: {type(error).__name__}"
        parser.exit(1, f"{program}: {error}\n")
    else:
        if profiler is not None:
            profiler.status = "completed"
    finally:
        if profiler is not None:
            profiler.finish()


def serve_main() -> None:
    _run("serve", None, "trtllm-serve (ROCm)")


def bench_main() -> None:
    _run("bench", None, "trtllm-bench (ROCm)")


def main(argv: list[str] | None = None) -> None:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if not arguments or arguments[0] in ("--help", "-h"):
        print(
            "trtllm-rdna4 {doctor,generate,bench,serve,validate} [options] [--profile]\n"
            "Use '<command> --help' for command options. GPU devices use PyTorch's HIP 'cuda:N' namespace."
        )
        return
    command, tail = arguments[0], arguments[1:]
    if command in ("generate", "bench", "serve"):
        _run(command, tail, f"trtllm-rdna4 {command}")
    elif command == "doctor":
        parser = argparse.ArgumentParser(prog="trtllm-rdna4 doctor")
        parser.add_argument(
            "--allow-cpu", action="store_true", help="Do not fail if no RDNA4 GPU is available"
        )
        parser.parse_args(tail)
        report = diagnostics()
        print(json.dumps(report, indent=2))
        if not report["ready"] and "--allow-cpu" not in tail:
            raise SystemExit(1)
    elif command == "validate":
        from .validation import main as validate

        validate(tail)
    else:
        raise SystemExit(f"Unknown ROCm command: {command}")
    profiler = active_session()
    if profiler is not None:
        profiler.status = "completed"
        profiler.finish()


__all__ = ["main", "serve_main", "bench_main"]
