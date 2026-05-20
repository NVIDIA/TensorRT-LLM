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
"""Baseline-vs-TurboQuant4 generation parity eval.

This script runs the same deterministic prompt set through the PyTorch TRTLLM
attention backend twice:

* baseline KV cache, normally dtype="auto"
* TurboQuant4 KV cache, dtype="turboquant4"

It is intentionally focused on TurboQuant4 production-path risk rather than on
absolute benchmark scores. It checks whether TurboQuant4 preserves the standard
model's generated tokens/text, whether it regresses simple reference-answer
prompts that the baseline answered correctly, and how generated-token logprobs
move once decode starts reading the KV cache.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class EvalPrompt:
    name: str
    prompt: str
    reference_contains: str | None = None


@dataclass
class OutputRecord:
    text: str
    token_ids: list[int]
    finish_reason: str | None
    contains_reference: bool | None
    decode_token_count: int
    decode_nll_per_token: float | None


@dataclass
class PromptRecord:
    name: str
    prompt: str
    reference_contains: str | None
    baseline: OutputRecord
    turboquant4: OutputRecord
    exact_token_match: bool
    exact_text_match: bool
    token_prefix_agreement: float
    normalized_text_distance: float
    reference_regressed: bool


def _default_prompts(long_context_repeats: int) -> list[EvalPrompt]:
    filler = (
        "This paragraph is padding for a long-context retrieval check. "
        "It mentions ordinary facts about schedules, teams, and notebooks, "
        "but the important answer is given only in the final instruction. "
    )
    long_context = filler * long_context_repeats
    return [
        EvalPrompt(
            name="capital_france",
            prompt="Answer with one word. The capital of France is",
            reference_contains="Paris",
        ),
        EvalPrompt(
            name="simple_addition",
            prompt="Answer with only the number. 17 plus 25 equals",
            reference_contains="42",
        ),
        EvalPrompt(
            name="color_mixing",
            prompt="Answer with one color. Mixing red and blue paint makes",
            reference_contains="purple",
        ),
        EvalPrompt(
            name="short_instruction",
            prompt=(
                "Rewrite this sentence in past tense and output only the rewrite: "
                "The engineer checks the cache."
            ),
            reference_contains="checked",
        ),
        EvalPrompt(
            name="long_context_codename",
            prompt=(
                f"{long_context}\nRemember this exact code: ORION-17.\n"
                "Question: What exact code were you told to remember? "
                "Answer with only the code."
            ),
            reference_contains="ORION-17",
        ),
        EvalPrompt(
            name="long_context_city",
            prompt=(
                f"{long_context}\nThe hidden destination is Kyoto.\n"
                "Question: What is the hidden destination? Answer with one word."
            ),
            reference_contains="Kyoto",
        ),
    ]


def _load_prompt_file(path: Path) -> list[EvalPrompt]:
    prompts = []
    with path.open() as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            try:
                prompts.append(
                    EvalPrompt(
                        name=str(data["name"]),
                        prompt=str(data["prompt"]),
                        reference_contains=(
                            None
                            if data.get("reference_contains") is None
                            else str(data["reference_contains"])
                        ),
                    )
                )
            except KeyError as exc:
                raise ValueError(f"{path}:{line_number} is missing field {exc}") from exc
    return prompts


def _levenshtein_distance(left: str, right: str) -> int:
    if left == right:
        return 0
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for left_index, left_char in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_char in enumerate(right, start=1):
            current.append(
                min(
                    previous[right_index] + 1,
                    current[right_index - 1] + 1,
                    previous[right_index - 1] + (left_char != right_char),
                )
            )
        previous = current
    return previous[-1]


def _prefix_agreement(left: list[int], right: list[int]) -> float:
    denominator = max(len(left), len(right), 1)
    matches = 0
    for left_token, right_token in zip(left, right):
        if left_token != right_token:
            break
        matches += 1
    return matches / denominator


def _contains_reference(text: str, reference: str | None) -> bool | None:
    if reference is None:
        return None
    return reference.lower() in text.lower()


def _ensure_repo_root_on_path() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)


def _decode_nll_per_token(
    prompt_name: str, token_ids: list[int], logprobs: Any
) -> tuple[int, float | None]:
    if not logprobs:
        return 0, None
    if len(logprobs) < len(token_ids):
        raise ValueError(
            f"{prompt_name} returned {len(logprobs)} generated-token logprob "
            f"entries for {len(token_ids)} generated tokens."
        )

    # The first generated token is selected from prefill logits. Tokens after it
    # exercise decode attention over the KV cache.
    if len(token_ids) <= 1:
        return 0, None

    total_nll = 0.0
    decode_token_count = len(token_ids) - 1
    for token_index, token_id in enumerate(token_ids[1:], start=1):
        token_logprobs = logprobs[token_index]
        logprob = token_logprobs.get(token_id)
        if logprob is None:
            raise ValueError(
                f"{prompt_name} generated logprobs are missing generated token "
                f"{token_id} at decode position {token_index}."
            )
        total_nll -= float(logprob.logprob)
    return decode_token_count, total_nll / decode_token_count


def _make_llm(args: argparse.Namespace, kv_cache_dtype: str):
    _ensure_repo_root_on_path()
    from tensorrt_llm import LLM
    from tensorrt_llm.llmapi import KvCacheConfig

    return LLM(
        args.model,
        backend="pytorch",
        tokenizer=args.tokenizer,
        tokenizer_mode=args.tokenizer_mode,
        skip_tokenizer_init=args.skip_tokenizer_init,
        trust_remote_code=args.trust_remote_code,
        dtype=args.dtype,
        load_format=args.load_format,
        attn_backend=args.attn_backend,
        kv_cache_config=KvCacheConfig(
            dtype=kv_cache_dtype,
            free_gpu_memory_fraction=args.free_gpu_memory_fraction,
            enable_block_reuse=False,
        ),
        cuda_graph_config=None,
        max_batch_size=args.max_batch_size,
        max_num_tokens=args.max_num_tokens,
        max_seq_len=args.max_seq_len,
        max_input_len=args.max_input_len,
        disable_overlap_scheduler=args.disable_overlap_scheduler,
    )


def _resolve_sampling_token_ids(args: argparse.Namespace) -> None:
    if args.end_id is not None:
        if args.pad_id is None:
            args.pad_id = args.end_id
        return

    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise ValueError(
            "--end-id is required when transformers is not available."
        ) from exc

    tokenizer_name = args.tokenizer or args.model
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name, trust_remote_code=args.trust_remote_code
        )
    except (OSError, ValueError) as exc:
        raise ValueError(
            "Could not resolve end_id from the tokenizer. Pass --end-id "
            "explicitly, and optionally --pad-id."
        ) from exc

    if tokenizer.eos_token_id is None:
        raise ValueError(
            "Tokenizer does not define eos_token_id. Pass --end-id explicitly."
        )

    args.end_id = int(tokenizer.eos_token_id)
    if args.pad_id is None:
        pad_token_id = tokenizer.pad_token_id
        args.pad_id = int(
            pad_token_id if pad_token_id is not None else args.end_id)


def _run_generation(
    args: argparse.Namespace, prompts: list[EvalPrompt], kv_cache_dtype: str
) -> list[OutputRecord]:
    from tensorrt_llm import SamplingParams

    sampling_params = SamplingParams(
        max_tokens=args.max_tokens,
        temperature=0.0,
        top_k=1,
        seed=args.seed,
        logprobs=args.logprobs,
        add_special_tokens=not args.skip_special_tokens_for_prompt,
        end_id=getattr(args, "end_id", None),
        pad_id=getattr(args, "pad_id", None),
    )
    with _make_llm(args, kv_cache_dtype) as llm:
        results = llm.generate(
            [prompt.prompt for prompt in prompts], sampling_params, use_tqdm=False
        )

    if not isinstance(results, list):
        results = [results]
    if len(results) != len(prompts):
        raise ValueError(f"Expected {len(prompts)} results, got {len(results)}.")

    records = []
    for prompt, result in zip(prompts, results):
        output = result.outputs[0]
        text = output.text or ""
        token_ids = list(output.token_ids or [])
        decode_token_count, decode_nll_per_token = _decode_nll_per_token(
            prompt.name, token_ids, output.logprobs
        )
        records.append(
            OutputRecord(
                text=text,
                token_ids=token_ids,
                finish_reason=output.finish_reason,
                contains_reference=_contains_reference(text, prompt.reference_contains),
                decode_token_count=decode_token_count,
                decode_nll_per_token=decode_nll_per_token,
            )
        )
    return records


def _release_cuda_memory() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
    gc.collect()


def _compare_outputs(
    prompts: list[EvalPrompt],
    baseline_outputs: list[OutputRecord],
    turbo_outputs: list[OutputRecord],
) -> tuple[list[PromptRecord], dict[str, Any]]:
    records = []
    for prompt, baseline, turbo in zip(prompts, baseline_outputs, turbo_outputs):
        max_text_len = max(len(baseline.text), len(turbo.text), 1)
        normalized_text_distance = _levenshtein_distance(baseline.text, turbo.text) / max_text_len
        reference_regressed = bool(
            baseline.contains_reference is True and turbo.contains_reference is False
        )
        records.append(
            PromptRecord(
                name=prompt.name,
                prompt=prompt.prompt,
                reference_contains=prompt.reference_contains,
                baseline=baseline,
                turboquant4=turbo,
                exact_token_match=baseline.token_ids == turbo.token_ids,
                exact_text_match=baseline.text == turbo.text,
                token_prefix_agreement=_prefix_agreement(baseline.token_ids, turbo.token_ids),
                normalized_text_distance=normalized_text_distance,
                reference_regressed=reference_regressed,
            )
        )

    count = len(records)
    exact_token_matches = sum(record.exact_token_match for record in records)
    exact_text_matches = sum(record.exact_text_match for record in records)
    reference_regressions = sum(record.reference_regressed for record in records)
    baseline_reference_correct = sum(
        record.baseline.contains_reference is True for record in records
    )
    turbo_reference_correct = sum(
        record.turboquant4.contains_reference is True for record in records
    )
    baseline_decode_nll, baseline_decode_tokens = _summarize_decode_nll(baseline_outputs)
    turbo_decode_nll, turbo_decode_tokens = _summarize_decode_nll(turbo_outputs)
    decode_nll_delta = (
        None
        if baseline_decode_nll is None or turbo_decode_nll is None
        else turbo_decode_nll - baseline_decode_nll
    )
    summary = {
        "num_prompts": count,
        "exact_token_match_rate": exact_token_matches / max(count, 1),
        "exact_text_match_rate": exact_text_matches / max(count, 1),
        "mean_token_prefix_agreement": sum(record.token_prefix_agreement for record in records)
        / max(count, 1),
        "mean_normalized_text_distance": sum(record.normalized_text_distance for record in records)
        / max(count, 1),
        "baseline_reference_correct": baseline_reference_correct,
        "turboquant4_reference_correct": turbo_reference_correct,
        "reference_regressions": reference_regressions,
        "baseline_decode_token_count": baseline_decode_tokens,
        "turboquant4_decode_token_count": turbo_decode_tokens,
        "baseline_decode_nll_per_token": baseline_decode_nll,
        "turboquant4_decode_nll_per_token": turbo_decode_nll,
        "decode_nll_delta_per_token": decode_nll_delta,
    }
    return records, summary


def _summarize_decode_nll(outputs: list[OutputRecord]) -> tuple[float | None, int]:
    total_nll = 0.0
    total_tokens = 0
    for output in outputs:
        if output.decode_nll_per_token is None:
            continue
        total_nll += output.decode_nll_per_token * output.decode_token_count
        total_tokens += output.decode_token_count
    if total_tokens == 0:
        return None, 0
    return total_nll / total_tokens, total_tokens


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="HF model name or local model directory.")
    parser.add_argument("--tokenizer", default=None, help="Optional tokenizer name or path.")
    parser.add_argument("--tokenizer-mode", default="auto", choices=["auto", "slow"])
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--load-format", default="auto", choices=["auto", "dummy"])
    parser.add_argument("--attn-backend", default="TRTLLM")
    parser.add_argument("--baseline-kv-cache-dtype", default="auto")
    parser.add_argument("--turbo-kv-cache-dtype", default="turboquant4")
    parser.add_argument("--max-tokens", "--max-new-tokens", dest="max_tokens", type=int, default=32)
    parser.add_argument(
        "--logprobs",
        type=int,
        default=1,
        help="Generated-token logprobs to request; set to 0 for sampled tokens only.",
    )
    parser.add_argument("--max-batch-size", type=int, default=8)
    parser.add_argument("--max-num-tokens", type=int, default=2048)
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--max-input-len", type=int, default=1800)
    parser.add_argument("--free-gpu-memory-fraction", type=float, default=0.45)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--end-id", type=int, default=None)
    parser.add_argument("--pad-id", type=int, default=None)
    parser.add_argument("--long-context-repeats", type=int, default=40)
    parser.add_argument(
        "--limit-prompts",
        type=int,
        default=None,
        help="Optionally run only the first N prompts as a quick canary.",
    )
    parser.add_argument("--prompt-file", type=Path, default=None)
    parser.add_argument("--output-json", type=Path, default=Path("turboquant4_quality_eval.json"))
    parser.add_argument("--skip-tokenizer-init", action="store_true")
    parser.add_argument("--skip-special-tokens-for-prompt", action="store_true")
    parser.add_argument("--disable-overlap-scheduler", action="store_true")
    parser.add_argument("--min-exact-token-match-rate", type=float, default=0.8)
    parser.add_argument("--min-token-prefix-agreement", type=float, default=0.9)
    parser.add_argument("--max-reference-regressions", type=int, default=0)
    parser.add_argument(
        "--max-decode-nll-delta-per-token",
        type=float,
        default=None,
        help="Optional failure threshold for TurboQuant4 decode NLL regression.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.max_tokens <= 0:
        raise ValueError("--max-tokens must be positive.")
    if args.logprobs < 0:
        raise ValueError("--logprobs must be non-negative.")
    prompts = (
        _load_prompt_file(args.prompt_file)
        if args.prompt_file is not None
        else _default_prompts(args.long_context_repeats)
    )
    if args.limit_prompts is not None:
        if args.limit_prompts <= 0:
            raise ValueError("--limit-prompts must be positive when provided.")
        prompts = prompts[: args.limit_prompts]
    if not prompts:
        raise ValueError("At least one eval prompt is required.")

    _resolve_sampling_token_ids(args)
    baseline_outputs = _run_generation(args, prompts, args.baseline_kv_cache_dtype)
    _release_cuda_memory()
    turbo_outputs = _run_generation(args, prompts, args.turbo_kv_cache_dtype)
    records, summary = _compare_outputs(prompts, baseline_outputs, turbo_outputs)

    thresholds = {
        "min_exact_token_match_rate": args.min_exact_token_match_rate,
        "min_token_prefix_agreement": args.min_token_prefix_agreement,
        "max_reference_regressions": args.max_reference_regressions,
        "max_decode_nll_delta_per_token": args.max_decode_nll_delta_per_token,
    }
    passed = (
        summary["exact_token_match_rate"] >= args.min_exact_token_match_rate
        and summary["mean_token_prefix_agreement"] >= args.min_token_prefix_agreement
        and summary["reference_regressions"] <= args.max_reference_regressions
    )
    if args.max_decode_nll_delta_per_token is not None:
        decode_nll_delta = summary["decode_nll_delta_per_token"]
        passed = (
            passed
            and decode_nll_delta is not None
            and decode_nll_delta <= args.max_decode_nll_delta_per_token
        )

    result = {
        "model": args.model,
        "baseline_kv_cache_dtype": args.baseline_kv_cache_dtype,
        "turbo_kv_cache_dtype": args.turbo_kv_cache_dtype,
        "dtype": args.dtype,
        "attn_backend": args.attn_backend,
        "long_context_repeats": args.long_context_repeats,
        "thresholds": thresholds,
        "passed": passed,
        "summary": summary,
        "records": [asdict(record) for record in records],
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w") as f:
        json.dump(result, f, indent=2)

    print(json.dumps({"passed": passed, "summary": summary}, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
