# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Compare generation logprobs with prefill scoring of the exact generated tokens.

Each request runs alone, with prefix reuse, chunked prefill, and speculation disabled.
The first response token comes from prefill; only subsequent response tokens exercise
decode. Both groups are reported separately. Repeated scoring of the frozen sequence
measures prefill repeatability, not decode determinism or batch invariance.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import os
import platform
import time
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import torch

    from tensorrt_llm.executor.result import Logprob

_THRESHOLDS = (0.001, 0.01, 0.05, 0.1)
_DEFAULT_PROMPTS = [
    {
        "id": "arithmetic",
        "prompt": "A shop has 12 boxes of pencils, with 8 pencils in each box. It sells 29 pencils. "
        "How many pencils remain? Explain your calculation step by step.\nAnswer:",
    },
    {
        "id": "code",
        "prompt": "Write a Python function that merges two sorted lists of integers. "
        "Explain how it handles duplicates and empty lists.\nAnswer:",
    },
    {
        "id": "explanation",
        "prompt": "Explain why the seasons change during the year. Distinguish the roles of "
        "Earth's axial tilt, its rotation, and its orbit around the Sun.\nAnswer:",
    },
]


def _logprobs_from_logits(logits: torch.Tensor, token_ids: list[int]) -> list[float]:
    """Normalize both execution paths with the same FP32 CPU implementation."""
    import torch

    if logits.ndim != 2 or logits.shape[0] != len(token_ids) or not token_ids:
        raise ValueError("Expected one logits row per response token")
    if min(token_ids) < 0 or max(token_ids) >= logits.shape[1]:
        raise ValueError("Response token is outside the returned logits vocabulary")
    logits_cpu = logits.detach().to(device="cpu", dtype=torch.float32)
    indices = torch.tensor(token_ids, dtype=torch.int64).unsqueeze(1)
    selected = torch.log_softmax(logits_cpu, dim=-1).gather(1, indices).flatten()
    if not torch.isfinite(selected).all():
        raise ValueError("Non-finite logprob computed from model logits")
    return selected.tolist()


def _extract_logprobs(
    entries: list[dict[int, Logprob]], token_ids: list[int], *, offset: int = 0
) -> list[float]:
    """Extract scored token probabilities without guessing missing or shifted entries."""
    if not token_ids or offset < 0 or len(entries) < offset + len(token_ids):
        raise ValueError("Missing logprob rows for the requested token span")
    values = []
    for position, token in enumerate(token_ids, start=offset):
        if token not in entries[position]:
            raise ValueError(f"Token {token} is missing from logprob row {position}")
        value = float(entries[position][token].logprob)
        if not math.isfinite(value):
            raise ValueError(f"Non-finite logprob at row {position}: {value}")
        values.append(value)
    return values


def _score_response(
    entries: list[dict[int, Logprob]], prompt_ids: list[int], response_ids: list[int]
) -> list[float]:
    """Row i scores input token i+1; the first response uses row prompt_length-1."""
    if not prompt_ids:
        raise ValueError("The prompt must contain at least one token")
    sequence_length = len(prompt_ids) + len(response_ids)
    # Some executors also expose the extra generated token's score in the last row.
    if len(entries) not in (sequence_length - 1, sequence_length):
        raise ValueError(
            f"Unexpected prompt logprob count: {len(entries)} for {sequence_length} tokens"
        )
    return _extract_logprobs(entries, response_ids, offset=len(prompt_ids) - 1)


def _compare(reference: list[float], observed: list[float]) -> dict:
    if not reference or len(reference) != len(observed):
        raise ValueError("Comparison requires nonempty, equally sized logprob sequences")
    delta = np.asarray(observed, dtype=np.float64) - np.asarray(reference, dtype=np.float64)
    if not np.isfinite(delta).all():
        raise ValueError("Comparison contains non-finite logprobs")
    absolute = np.abs(delta)
    return {
        "token_count": len(reference),
        "mean_signed_delta": float(delta.mean()),
        "mean_abs_delta": float(absolute.mean()),
        "p50_abs_delta": float(np.quantile(absolute, 0.5)),
        "p99_abs_delta": float(np.quantile(absolute, 0.99)),
        "max_abs_delta": float(absolute.max()),
        "exact_match_fraction": float((absolute == 0).mean()),
        "fraction_above": {str(t): float((absolute > t).mean()) for t in _THRESHOLDS},
    }


def _summarize(cases: list[dict]) -> dict:
    decode, replay, boundary_decode, boundary_replay, repeated, first = [], [], [], [], [], []
    for case in cases:
        generated = case["generation_logprobs"]
        scored = case["prefill_logprobs"][0]
        if len(generated) < 2 or len(scored) != len(generated):
            raise ValueError("Each case needs a prefill boundary and at least one decode token")
        decode.extend(generated[1:])
        replay.extend(scored[1:])
        boundary_decode.append(generated[0])
        boundary_replay.append(scored[0])
        for repeat in case["prefill_logprobs"][1:]:
            if len(repeat) != len(scored):
                raise ValueError("Prefill repeats have different token counts")
            first.extend(scored)
            repeated.extend(repeat)
    result = {
        "decode_vs_prefill": _compare(decode, replay),
        "prefill_boundary_vs_full_prefill": _compare(boundary_decode, boundary_replay),
    }
    if first:
        result["prefill_repeatability"] = _compare(first, repeated)
    return result


def _write_report(path: Path, report: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _load_prompts(path: Path | None) -> list[dict]:
    prompts = json.loads(path.read_text(encoding="utf-8")) if path else _DEFAULT_PROMPTS
    if not isinstance(prompts, list) or not prompts:
        raise ValueError("Prompts must be a nonempty JSON list")
    identifiers = set()
    for prompt in prompts:
        if not isinstance(prompt, dict) or not isinstance(prompt.get("id"), str):
            raise ValueError("Each prompt needs a string id")
        if prompt["id"] in identifiers:
            raise ValueError(f"Duplicate prompt id: {prompt['id']}")
        identifiers.add(prompt["id"])
        tokens = prompt.get("prompt_token_ids")
        if tokens is not None:
            if not isinstance(tokens, list) or not tokens:
                raise ValueError("prompt_token_ids must be a nonempty integer list")
            if any(type(token) is not int or token < 0 for token in tokens):
                raise ValueError("prompt_token_ids must contain nonnegative integers")
        elif not isinstance(prompt.get("prompt"), str) or not prompt["prompt"]:
            raise ValueError("Each prompt needs prompt text or prompt_token_ids")
    return prompts


def _engine_options(config: dict, max_sequence_length: int, tp_size: int) -> dict:
    """Keep the execution comparisons controlled while allowing backend overrides."""
    required = {
        "backend": "pytorch",
        "max_batch_size": 1,
        "enable_chunked_prefill": False,
        "disable_overlap_scheduler": True,
        "speculative_config": None,
    }
    for key, value in required.items():
        if key in config and config[key] != value:
            raise ValueError(f"This probe requires {key}={value!r}")
    options = {
        "tensor_parallel_size": tp_size,
        "max_seq_len": max_sequence_length,
        "max_num_tokens": max_sequence_length,
        "cuda_graph_config": None,
        "allreduce_strategy": "NCCL",
        **config,
        **required,
    }
    if (
        options["max_seq_len"] < max_sequence_length
        or options["max_num_tokens"] < max_sequence_length
    ):
        raise ValueError("max_seq_len and max_num_tokens must fit the complete scoring request")
    kv_options = {
        "free_gpu_memory_fraction": 0.3,
        "mamba_ssm_cache_dtype": "float32",
        **options.get("kv_cache_config", {}),
    }
    if kv_options.get("enable_block_reuse", False):
        raise ValueError("Prefill scoring requires kv_cache_config.enable_block_reuse=False")
    kv_options["enable_block_reuse"] = False
    options["kv_cache_config"] = kv_options
    return options


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", required=True, help="Hugging Face checkpoint directory or model ID"
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="JSON report, including exact token IDs"
    )
    parser.add_argument("--config", type=Path, help="JSON file of extra LLM constructor options")
    parser.add_argument(
        "--prompts", type=Path, help="JSON list of {id, prompt} or {id, prompt_token_ids}"
    )
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--prefill-repeats", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument(
        "--prompt-lengths",
        type=int,
        nargs="+",
        default=[128, 512],
        help="Repeat/truncate raw prompt tokens to these lengths; token IDs are saved in the report",
    )
    parser.add_argument(
        "--max-abs-dlogprob",
        type=float,
        help="Optional failure threshold for decode tokens; omitted means measurement only",
    )
    args = parser.parse_args()
    if args.max_tokens < 2 or args.prefill_repeats < 2 or args.tp_size < 1:
        parser.error("Need max-tokens >= 2, prefill-repeats >= 2, and tp-size >= 1")
    if any(length < 1 for length in args.prompt_lengths):
        parser.error("Prompt lengths must be positive")
    if not math.isfinite(args.temperature) or args.temperature < 0:
        parser.error("Temperature must be finite and nonnegative")
    if args.max_abs_dlogprob is not None and (
        not math.isfinite(args.max_abs_dlogprob) or args.max_abs_dlogprob < 0
    ):
        parser.error("The failure threshold must be finite and nonnegative")

    prompts = _load_prompts(args.prompts)
    config = json.loads(args.config.read_text(encoding="utf-8")) if args.config else {}
    if not isinstance(config, dict) or "model" in config:
        parser.error("Config must be a JSON object; specify the model through --model")
    options = _engine_options(config, max(args.prompt_lengths) + args.max_tokens + 1, args.tp_size)

    import torch

    from tensorrt_llm import LLM, SamplingParams

    report = {
        "schema_version": 1,
        "status": "running",
        "measurement": "prefill_decode_consistency",
        "normalization": "torch.log_softmax, float32, cpu, temperature=1",
        "model": args.model,
        "engine_options": options,
        "generation": {
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "seed": args.seed,
        },
        "prefill_repeats": args.prefill_repeats,
        "max_abs_dlogprob_threshold": args.max_abs_dlogprob,
        "environment": {
            "python": platform.python_version(),
            "tensorrt_llm": importlib.metadata.version("tensorrt_llm"),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
            "determinism_env": {
                key: os.environ.get(key)
                for key in (
                    "FORCE_DETERMINISTIC",
                    "FORCE_MOE_KERNEL_DETERMINISTIC",
                    "FORCE_ATTENTION_KERNEL_DETERMINISTIC",
                    "FORCE_ALL_REDUCE_DETERMINISTIC",
                )
            },
        },
        "cases": [],
    }
    _write_report(args.output, report)
    started = time.monotonic()
    with LLM(model=args.model, **options) as llm:
        for prompt in prompts:
            source_ids = prompt.get("prompt_token_ids")
            if source_ids is None:
                source_ids = llm.tokenizer.encode(prompt["prompt"], add_special_tokens=False)
            if not source_ids:
                raise ValueError(f"Prompt {prompt['id']} tokenized to an empty sequence")
            for length in args.prompt_lengths:
                prompt_ids = (source_ids * math.ceil(length / len(source_ids)))[:length]
                case_id = f"{prompt['id']}/length_{length}"
                print(f"Running {case_id}", flush=True)
                sampling = SamplingParams(
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    seed=args.seed + len(report["cases"]),
                    top_p=1.0,
                    top_k=None,
                    ignore_eos=True,
                    add_special_tokens=False,
                    logprobs=0,
                    logprobs_mode="raw",
                    return_generation_logits=True,
                )
                generated = llm.generate([prompt_ids], sampling_params=sampling, use_tqdm=False)[0]
                response_ids = list(generated.outputs[0].token_ids)
                if (
                    list(generated.prompt_token_ids) != prompt_ids
                    or len(response_ids) != args.max_tokens
                ):
                    raise ValueError(f"{case_id}: generation changed the prompt or stopped early")
                generation_reported_logprobs = _extract_logprobs(
                    generated.outputs[0].logprobs, response_ids
                )
                generation_logprobs = _logprobs_from_logits(
                    generated.outputs[0].generation_logits, response_ids
                )
                del generated
                sequence = prompt_ids + response_ids
                prefill_logprobs = []
                prefill_reported_logprobs = []
                for _ in range(args.prefill_repeats):
                    scored = llm.generate(
                        [sequence],
                        sampling_params=SamplingParams(
                            max_tokens=1,
                            temperature=0.0,
                            ignore_eos=True,
                            add_special_tokens=False,
                            prompt_logprobs=0,
                            return_context_logits=True,
                        ),
                        use_tqdm=False,
                    )[0]
                    if list(scored.prompt_token_ids) != sequence:
                        raise ValueError(f"{case_id}: scoring changed the frozen token sequence")
                    prefill_reported_logprobs.append(
                        _score_response(scored.outputs[0].prompt_logprobs, prompt_ids, response_ids)
                    )
                    start = len(prompt_ids) - 1
                    prefill_logprobs.append(
                        _logprobs_from_logits(
                            scored.context_logits[start : start + len(response_ids)], response_ids
                        )
                    )
                    del scored
                case = {
                    "id": case_id,
                    "prompt_token_ids": prompt_ids,
                    "response_token_ids": response_ids,
                    "generation_logprobs": generation_logprobs,
                    "prefill_logprobs": prefill_logprobs,
                    "generation_reported_logprobs": generation_reported_logprobs,
                    "prefill_reported_logprobs": prefill_reported_logprobs,
                    "generation_normalization_check": _compare(
                        generation_reported_logprobs, generation_logprobs
                    ),
                    "prefill_normalization_check": _compare(
                        prefill_reported_logprobs[0], prefill_logprobs[0]
                    ),
                    "decode_vs_prefill": _compare(generation_logprobs[1:], prefill_logprobs[0][1:]),
                }
                report["cases"].append(case)
                report["summary"] = _summarize(report["cases"])
                _write_report(args.output, report)
                print(json.dumps({"case": case_id, **case["decode_vs_prefill"]}), flush=True)

    worst_delta = report["summary"]["decode_vs_prefill"]["max_abs_delta"]
    exceeded = args.max_abs_dlogprob is not None and worst_delta > args.max_abs_dlogprob
    report["status"] = "threshold_exceeded" if exceeded else "completed"
    report["elapsed_seconds"] = time.monotonic() - started
    _write_report(args.output, report)
    print(json.dumps(report["summary"], indent=2), flush=True)
    if exceeded:
        raise SystemExit(f"Decode/prefill delta {worst_delta} exceeds {args.max_abs_dlogprob}")


if __name__ == "__main__":
    main()
