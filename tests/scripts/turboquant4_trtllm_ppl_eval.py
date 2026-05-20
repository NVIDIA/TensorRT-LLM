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
"""TensorRT-LLM baseline-vs-TurboQuant4 context perplexity canary.

This script scores prompt-token negative log likelihood through the LLM API
twice:

* baseline KV cache, normally dtype="auto"
* TurboQuant4 KV cache, dtype="turboquant4"

It uses prompt logprobs instead of generated-token parity so the comparison is
less sensitive to deterministic decoding tie breaks. This is useful as an
end-to-end LLM API canary, but prompt logprobs are mostly a prefill/context
metric. Use ``turboquant4_quality_eval.py`` for generated decode output that
reads the packed KV cache after the first generated token.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class EvalText:
    name: str
    text: str


@dataclass
class TextScore:
    name: str
    token_count: int
    baseline_nll: float
    turboquant4_nll: float
    baseline_ppl: float
    turboquant4_ppl: float
    nll_delta_per_token: float
    relative_ppl_delta: float


def _ensure_repo_root_on_path() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)


def _default_texts(long_context_repeats: int) -> list[EvalText]:
    filler = (
        "This paragraph is padding for a long-context cache quality check. "
        "It includes ordinary facts about deployment reviews, tensor shapes, "
        "and notebook labels, but the important tokens appear later. "
    )
    long_context = filler * long_context_repeats
    return [
        EvalText(
            name="technical_cache",
            text=(
                "TensorRT-LLM stores key and value tensors in a paged cache so "
                "that generation can reuse attention context across decode "
                "steps. Quantizing the cache saves memory, but the next-token "
                "distribution should remain close to the baseline model."
            ),
        ),
        EvalText(
            name="reasoning_short",
            text=(
                "A customer ordered seventeen blue notebooks and twenty-five "
                "green notebooks. The warehouse packed forty-two notebooks in "
                "total and shipped them before noon."
            ),
        ),
        EvalText(
            name="retrieval_short",
            text=(
                "The project codename is ORION-17. During the review, the "
                "engineer repeated that ORION-17 must appear exactly in the "
                "deployment checklist."
            ),
        ),
        EvalText(
            name="plain_language",
            text=(
                "Clear software changes are easy to review when the behavior is "
                "small, the tests explain the risk, and the implementation "
                "avoids unnecessary abstractions."
            ),
        ),
        EvalText(
            name="long_context_codename",
            text=(
                f"{long_context}"
                "Remember this exact code: ORION-17. "
                "The code ORION-17 is the only identifier that should be copied "
                "into the final checklist."
            ),
        ),
    ]


def _load_text_file(path: Path) -> list[EvalText]:
    texts = []
    with path.open() as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            try:
                texts.append(EvalText(name=str(data["name"]), text=str(data["text"])))
            except KeyError as exc:
                raise ValueError(f"{path}:{line_number} is missing field {exc}") from exc
    return texts


def _make_llm(args: argparse.Namespace, kv_cache_dtype: str):
    _ensure_repo_root_on_path()
    from tensorrt_llm import LLM
    from tensorrt_llm.llmapi import KvCacheConfig

    return LLM(
        args.model,
        backend="pytorch",
        tokenizer=args.tokenizer,
        tokenizer_mode=args.tokenizer_mode,
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


def _score_result(eval_text: EvalText, result: Any) -> tuple[float, int]:
    prompt_token_ids = list(result.prompt_token_ids)
    if len(prompt_token_ids) < 2:
        raise ValueError(f"{eval_text.name} must tokenize to at least two prompt tokens.")

    prompt_logprobs = result.outputs[0].prompt_logprobs
    target_token_ids = prompt_token_ids[1:]
    if len(prompt_logprobs) < len(target_token_ids):
        raise ValueError(
            f"{eval_text.name} returned {len(prompt_logprobs)} prompt logprob "
            f"entries for {len(target_token_ids)} scored prompt tokens."
        )

    total_nll = 0.0
    for token_index, token_id in enumerate(target_token_ids):
        token_logprobs = prompt_logprobs[token_index]
        logprob = token_logprobs.get(token_id)
        if logprob is None:
            raise ValueError(
                f"{eval_text.name} prompt logprobs are missing target token "
                f"{token_id} at scored position {token_index}."
            )
        total_nll -= float(logprob.logprob)
    return total_nll, len(target_token_ids)


def _run_scores(
    args: argparse.Namespace, texts: list[EvalText], kv_cache_dtype: str
) -> dict[str, tuple[float, int]]:
    _ensure_repo_root_on_path()
    from tensorrt_llm import SamplingParams

    sampling_params = SamplingParams(
        max_tokens=1,
        temperature=0.0,
        top_k=1,
        seed=args.seed,
        prompt_logprobs=args.prompt_logprobs,
        add_special_tokens=not args.skip_special_tokens_for_prompt,
    )
    with _make_llm(args, kv_cache_dtype) as llm:
        results = llm.generate([text.text for text in texts], sampling_params, use_tqdm=False)

    if not isinstance(results, list):
        results = [results]
    if len(results) != len(texts):
        raise ValueError(f"Expected {len(texts)} results, got {len(results)}.")

    return {text.name: _score_result(text, result) for text, result in zip(texts, results)}


def _release_cuda_memory() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
    gc.collect()


def _safe_exp(value: float) -> float:
    try:
        return math.exp(value)
    except OverflowError:
        return float("inf")


def _compare_scores(
    texts: list[EvalText],
    baseline_scores: dict[str, tuple[float, int]],
    turbo_scores: dict[str, tuple[float, int]],
) -> tuple[list[TextScore], dict[str, float | int]]:
    records = []
    total_baseline_nll = 0.0
    total_turbo_nll = 0.0
    total_tokens = 0
    for text in texts:
        baseline_nll, baseline_tokens = baseline_scores[text.name]
        turbo_nll, turbo_tokens = turbo_scores[text.name]
        if baseline_tokens != turbo_tokens:
            raise ValueError(
                f"{text.name} token count changed: baseline={baseline_tokens}, "
                f"turboquant4={turbo_tokens}."
            )
        baseline_nll_per_token = baseline_nll / baseline_tokens
        turbo_nll_per_token = turbo_nll / turbo_tokens
        baseline_ppl = _safe_exp(baseline_nll_per_token)
        turbo_ppl = _safe_exp(turbo_nll_per_token)
        relative_ppl_delta = (turbo_ppl - baseline_ppl) / baseline_ppl
        records.append(
            TextScore(
                name=text.name,
                token_count=baseline_tokens,
                baseline_nll=baseline_nll_per_token,
                turboquant4_nll=turbo_nll_per_token,
                baseline_ppl=baseline_ppl,
                turboquant4_ppl=turbo_ppl,
                nll_delta_per_token=turbo_nll_per_token - baseline_nll_per_token,
                relative_ppl_delta=relative_ppl_delta,
            )
        )
        total_baseline_nll += baseline_nll
        total_turbo_nll += turbo_nll
        total_tokens += baseline_tokens

    baseline_nll_per_token = total_baseline_nll / total_tokens
    turbo_nll_per_token = total_turbo_nll / total_tokens
    baseline_ppl = _safe_exp(baseline_nll_per_token)
    turbo_ppl = _safe_exp(turbo_nll_per_token)
    summary: dict[str, float | int] = {
        "num_texts": len(texts),
        "token_count": total_tokens,
        "baseline_nll": baseline_nll_per_token,
        "turboquant4_nll": turbo_nll_per_token,
        "nll_delta_per_token": turbo_nll_per_token - baseline_nll_per_token,
        "baseline_ppl": baseline_ppl,
        "turboquant4_ppl": turbo_ppl,
        "relative_ppl_delta": (turbo_ppl - baseline_ppl) / baseline_ppl,
        "max_text_nll_delta_per_token": max(record.nll_delta_per_token for record in records),
    }
    return records, summary


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
    parser.add_argument(
        "--prompt-logprobs",
        type=int,
        default=0,
        help="Prompt logprobs to request; 0 returns only the actual prompt token.",
    )
    parser.add_argument("--max-batch-size", type=int, default=8)
    parser.add_argument("--max-num-tokens", type=int, default=2048)
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--max-input-len", type=int, default=1800)
    parser.add_argument("--free-gpu-memory-fraction", type=float, default=0.45)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--long-context-repeats", type=int, default=40)
    parser.add_argument(
        "--limit-texts",
        type=int,
        default=None,
        help="Optionally run only the first N texts as a quick canary.",
    )
    parser.add_argument("--text-file", type=Path, default=None)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("turboquant4_trtllm_ppl_eval.json"),
    )
    parser.add_argument("--skip-special-tokens-for-prompt", action="store_true")
    parser.add_argument("--disable-overlap-scheduler", action="store_true")
    parser.add_argument("--max-relative-ppl-delta", type=float, default=0.05)
    parser.add_argument("--max-nll-delta-per-token", type=float, default=0.05)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.prompt_logprobs < 0:
        raise ValueError("--prompt-logprobs must be non-negative.")

    texts = (
        _load_text_file(args.text_file)
        if args.text_file is not None
        else _default_texts(args.long_context_repeats)
    )
    if args.limit_texts is not None:
        if args.limit_texts <= 0:
            raise ValueError("--limit-texts must be positive when provided.")
        texts = texts[: args.limit_texts]
    if not texts:
        raise ValueError("At least one eval text is required.")

    baseline_scores = _run_scores(args, texts, args.baseline_kv_cache_dtype)
    _release_cuda_memory()
    turbo_scores = _run_scores(args, texts, args.turbo_kv_cache_dtype)
    records, summary = _compare_scores(texts, baseline_scores, turbo_scores)

    thresholds = {
        "max_relative_ppl_delta": args.max_relative_ppl_delta,
        "max_nll_delta_per_token": args.max_nll_delta_per_token,
    }
    passed = (
        summary["relative_ppl_delta"] <= args.max_relative_ppl_delta
        and summary["nll_delta_per_token"] <= args.max_nll_delta_per_token
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
