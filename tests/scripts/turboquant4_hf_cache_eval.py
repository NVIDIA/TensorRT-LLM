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
"""Transformers baseline-vs-TurboQuant4 KV-cache quality canary.

This eval does not import ``tensorrt_llm``. It loads the TurboQuant4 Python
reference module by path, runs greedy generation with a HuggingFace causal LM,
and compares normal generation against a generation loop that applies the
TurboQuant4 quantize/dequantize transform once to each newly written cached K/V
position.

Use this as an algorithmic canary for output drift. It does not validate the
TensorRT-LLM packed-cache allocator, native CUDA ops, scheduler integration, or
serving path; those remain covered by ``turboquant4_quality_eval.py`` in a built
TensorRT-LLM environment.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch


@dataclass(frozen=True)
class EvalPrompt:
    name: str
    prompt: str
    reference_contains: str | None = None


@dataclass
class OutputRecord:
    text: str
    token_ids: list[int]
    contains_reference: bool | None
    decode_token_count: int
    decode_nll_per_token: float | None


@dataclass
class PromptRecord:
    name: str
    prompt: str
    reference_contains: str | None
    baseline: OutputRecord
    turboquant4_qdq: OutputRecord
    exact_token_match: bool
    exact_text_match: bool
    token_prefix_agreement: float
    normalized_text_distance: float
    reference_regressed: bool


def _load_turboquant4_module():
    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / "tensorrt_llm" / "_torch" / "modules" / "turboquant4.py"
    spec = importlib.util.spec_from_file_location("turboquant4_hf_eval_target", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _default_prompts(long_context_repeats: int) -> list[EvalPrompt]:
    prompts = [
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
            name="short_rewrite",
            prompt="Rewrite in past tense: The engineer checks the cache.",
            reference_contains="checked",
        ),
    ]
    if long_context_repeats > 0:
        filler = (
            "This paragraph is padding for a long-context retrieval check. "
            "It mentions ordinary facts about schedules, teams, and notebooks, "
            "but the important answer is given only in the final instruction. "
        )
        long_context = filler * long_context_repeats
        prompts.append(
            EvalPrompt(
                name="long_context_codename",
                prompt=(
                    f"{long_context}\nRemember this exact code: ORION-17.\n"
                    "Question: What exact code were you told to remember? "
                    "Answer with only the code."
                ),
                reference_contains="ORION-17",
            )
        )
    return prompts


def _load_prompt_file(path: Path) -> list[EvalPrompt]:
    prompts = []
    with path.open() as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            try:
                reference = data.get("reference_contains")
                prompts.append(
                    EvalPrompt(
                        name=str(data["name"]),
                        prompt=str(data["prompt"]),
                        reference_contains=None if reference is None else str(reference),
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


def _legacy_cache(past_key_values):
    if hasattr(past_key_values, "to_legacy_cache"):
        return past_key_values.to_legacy_cache()
    return past_key_values


def _quantize_cache_slice(tensor: torch.Tensor, start: int, turboquant4) -> torch.Tensor:
    if start >= tensor.shape[-2]:
        return tensor
    quantized = tensor.clone()
    quantized[..., start:, :] = turboquant4.turboquant4_quantize_dequantize(tensor[..., start:, :])
    return quantized


def _quantize_cache_slice_if_enabled(
    tensor: torch.Tensor,
    start: int,
    turboquant4,
    enabled: bool,
) -> torch.Tensor:
    if not enabled:
        return tensor
    return _quantize_cache_slice(tensor, start, turboquant4)


def _quantize_past_key_values(
    past_key_values,
    turboquant4,
    previous_seq_lens: list[int] | None,
    kv_cache_quantization: str,
) -> tuple[Any, list[int]]:
    quantize_keys = kv_cache_quantization in ("both", "keys-only")
    quantize_values = kv_cache_quantization in ("both", "values-only")
    new_seq_lens = []
    if hasattr(past_key_values, "layers"):
        for layer_index, layer_cache in enumerate(past_key_values.layers):
            if not getattr(layer_cache, "is_initialized", True):
                new_seq_lens.append(0)
                continue
            key = getattr(layer_cache, "keys", None)
            value = getattr(layer_cache, "values", None)
            if key is None or value is None:
                raise TypeError(
                    "TurboQuant4 HF cache eval cannot find keys/values on "
                    f"cache layer {layer_index}."
                )
            _validate_cache_head_dim(key, layer_index)
            start = previous_seq_lens[layer_index] if previous_seq_lens else 0
            layer_cache.keys = _quantize_cache_slice_if_enabled(
                key,
                start,
                turboquant4,
                quantize_keys,
            )
            layer_cache.values = _quantize_cache_slice_if_enabled(
                value,
                start,
                turboquant4,
                quantize_values,
            )
            new_seq_lens.append(layer_cache.keys.shape[-2])
        return past_key_values, new_seq_lens

    legacy_cache = _legacy_cache(past_key_values)
    if not isinstance(legacy_cache, (tuple, list)):
        raise TypeError(
            "TurboQuant4 HF cache eval expects tuple/list past_key_values or a "
            "cache object with layers or to_legacy_cache()."
        )

    quantized_layers = []
    for layer_index, layer_cache in enumerate(legacy_cache):
        if len(layer_cache) < 2:
            raise ValueError(f"Layer {layer_index} past_key_values entry has no K/V tensors.")
        key, value = layer_cache[:2]
        _validate_cache_head_dim(key, layer_index)
        start = previous_seq_lens[layer_index] if previous_seq_lens else 0
        quantized_key = _quantize_cache_slice_if_enabled(
            key,
            start,
            turboquant4,
            quantize_keys,
        )
        quantized_value = _quantize_cache_slice_if_enabled(
            value,
            start,
            turboquant4,
            quantize_values,
        )
        new_seq_lens.append(quantized_key.shape[-2])
        quantized_layers.append((quantized_key, quantized_value, *layer_cache[2:]))
    return tuple(quantized_layers), new_seq_lens


def _validate_cache_head_dim(key: torch.Tensor, layer_index: int) -> None:
    head_dim = key.shape[-1]
    if head_dim % 2 != 0 or (head_dim & (head_dim - 1)) != 0:
        raise ValueError(
            "TurboQuant4 HF cache eval requires even power-of-2 attention "
            f"head_dim, got {head_dim} in layer {layer_index}."
        )


@torch.no_grad()
def _greedy_generate(
    model,
    tokenizer,
    prompt: EvalPrompt,
    *,
    max_new_tokens: int,
    device: torch.device,
    turboquant4=None,
    kv_cache_quantization: str = "both",
) -> OutputRecord:
    encoded = tokenizer(prompt.prompt, return_tensors="pt")
    input_ids = encoded["input_ids"].to(device)
    generated = input_ids
    attention_mask = torch.ones_like(input_ids, device=device)
    past_key_values = None
    quantized_cache_seq_lens = None
    decode_nll = 0.0
    decode_token_count = 0

    for step in range(max_new_tokens):
        current_input_ids = generated if past_key_values is None else generated[:, -1:]
        outputs = model(
            input_ids=current_input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
            return_dict=True,
        )
        logits = outputs.logits[:, -1, :]
        next_token = logits.argmax(dim=-1, keepdim=True)
        if step > 0:
            token_logprob = torch.log_softmax(logits.float(), dim=-1).gather(-1, next_token)
            decode_nll -= float(token_logprob.item())
            decode_token_count += 1
        generated = torch.cat([generated, next_token], dim=-1)
        attention_mask = torch.ones_like(generated, device=device)
        past_key_values = outputs.past_key_values
        if turboquant4 is not None and step + 1 < max_new_tokens:
            past_key_values, quantized_cache_seq_lens = _quantize_past_key_values(
                past_key_values,
                turboquant4,
                quantized_cache_seq_lens,
                kv_cache_quantization,
            )

    prompt_len = input_ids.shape[-1]
    new_token_ids = generated[0, prompt_len:].detach().cpu().tolist()
    text = tokenizer.decode(new_token_ids, skip_special_tokens=True)
    return OutputRecord(
        text=text,
        token_ids=[int(token_id) for token_id in new_token_ids],
        contains_reference=_contains_reference(text, prompt.reference_contains),
        decode_token_count=decode_token_count,
        decode_nll_per_token=(decode_nll / decode_token_count if decode_token_count > 0 else None),
    )


def _compare_outputs(
    prompts: list[EvalPrompt],
    baseline_outputs: list[OutputRecord],
    turbo_outputs: list[OutputRecord],
) -> tuple[list[PromptRecord], dict[str, Any]]:
    records = []
    for prompt, baseline, turbo in zip(prompts, baseline_outputs, turbo_outputs, strict=True):
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
                turboquant4_qdq=turbo,
                exact_token_match=baseline.token_ids == turbo.token_ids,
                exact_text_match=baseline.text == turbo.text,
                token_prefix_agreement=_prefix_agreement(baseline.token_ids, turbo.token_ids),
                normalized_text_distance=normalized_text_distance,
                reference_regressed=reference_regressed,
            )
        )

    count = len(records)
    baseline_decode_nll = 0.0
    turbo_decode_nll = 0.0
    baseline_decode_token_count = 0
    turbo_decode_token_count = 0
    for record in records:
        baseline = record.baseline
        turbo = record.turboquant4_qdq
        if baseline.decode_nll_per_token is not None:
            baseline_decode_nll += baseline.decode_nll_per_token * baseline.decode_token_count
            baseline_decode_token_count += baseline.decode_token_count
        if turbo.decode_nll_per_token is not None:
            turbo_decode_nll += turbo.decode_nll_per_token * turbo.decode_token_count
            turbo_decode_token_count += turbo.decode_token_count

    decode_nll_delta_per_token = None
    if baseline_decode_token_count > 0 and turbo_decode_token_count > 0:
        decode_nll_delta_per_token = (
            turbo_decode_nll / turbo_decode_token_count
            - baseline_decode_nll / baseline_decode_token_count
        )

    summary = {
        "num_prompts": count,
        "exact_token_match_rate": sum(record.exact_token_match for record in records)
        / max(count, 1),
        "exact_text_match_rate": sum(record.exact_text_match for record in records) / max(count, 1),
        "mean_token_prefix_agreement": sum(record.token_prefix_agreement for record in records)
        / max(count, 1),
        "mean_normalized_text_distance": sum(record.normalized_text_distance for record in records)
        / max(count, 1),
        "baseline_reference_correct": sum(
            record.baseline.contains_reference is True for record in records
        ),
        "turboquant4_reference_correct": sum(
            record.turboquant4_qdq.contains_reference is True for record in records
        ),
        "reference_regressions": sum(record.reference_regressed for record in records),
        "baseline_decode_token_count": baseline_decode_token_count,
        "turboquant4_decode_token_count": turbo_decode_token_count,
        "decode_nll_delta_per_token": decode_nll_delta_per_token,
    }
    return records, summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="distilgpt2", help="HF causal LM name or path.")
    parser.add_argument("--tokenizer", default=None, help="Optional tokenizer name or path.")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--device", default="cpu", help="cpu, cuda, mps, or auto.")
    parser.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument(
        "--kv-cache-quantization",
        default="both",
        choices=["both", "keys-only", "values-only"],
        help=(
            "Which HF cached tensors to quantize in the TurboQuant4 proxy path. "
            "Use keys-only or values-only to localize decode quality regressions."
        ),
    )
    parser.add_argument("--long-context-repeats", type=int, default=0)
    parser.add_argument("--prompt-file", type=Path, default=None)
    parser.add_argument("--limit-prompts", type=int, default=None)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("turboquant4_hf_cache_eval.json"),
    )
    parser.add_argument("--min-exact-token-match-rate", type=float, default=0.5)
    parser.add_argument("--min-token-prefix-agreement", type=float, default=0.75)
    parser.add_argument("--max-reference-regressions", type=int, default=0)
    parser.add_argument("--max-decode-nll-delta-per-token", type=float, default=None)
    return parser.parse_args()


def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


def _resolve_dtype(name: str) -> torch.dtype:
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[name]


def main() -> None:
    args = _parse_args()
    if args.max_new_tokens <= 0:
        raise ValueError("--max-new-tokens must be positive.")
    if args.long_context_repeats < 0:
        raise ValueError("--long-context-repeats must be non-negative.")
    if args.limit_prompts is not None and args.limit_prompts <= 0:
        raise ValueError("--limit-prompts must be positive when provided.")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    turboquant4 = _load_turboquant4_module()
    prompts = (
        _load_prompt_file(args.prompt_file)
        if args.prompt_file
        else _default_prompts(args.long_context_repeats)
    )
    if args.limit_prompts is not None:
        prompts = prompts[: args.limit_prompts]
    if not prompts:
        raise ValueError("At least one eval prompt is required.")

    device = _resolve_device(args.device)
    dtype = _resolve_dtype(args.dtype)
    tokenizer_name = args.tokenizer or args.model
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        trust_remote_code=args.trust_remote_code,
    )
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        trust_remote_code=args.trust_remote_code,
    ).to(device)
    model.eval()

    baseline_outputs = [
        _greedy_generate(
            model,
            tokenizer,
            prompt,
            max_new_tokens=args.max_new_tokens,
            device=device,
        )
        for prompt in prompts
    ]
    turbo_outputs = [
        _greedy_generate(
            model,
            tokenizer,
            prompt,
            max_new_tokens=args.max_new_tokens,
            device=device,
            turboquant4=turboquant4,
            kv_cache_quantization=args.kv_cache_quantization,
        )
        for prompt in prompts
    ]
    records, summary = _compare_outputs(prompts, baseline_outputs, turbo_outputs)

    thresholds = {
        "min_exact_token_match_rate": args.min_exact_token_match_rate,
        "min_token_prefix_agreement": args.min_token_prefix_agreement,
        "max_reference_regressions": args.max_reference_regressions,
        "max_decode_nll_delta_per_token": args.max_decode_nll_delta_per_token,
    }
    decode_nll_passed = args.max_decode_nll_delta_per_token is None or (
        summary["decode_nll_delta_per_token"] is not None
        and summary["decode_nll_delta_per_token"] <= args.max_decode_nll_delta_per_token
    )
    passed = (
        summary["exact_token_match_rate"] >= args.min_exact_token_match_rate
        and summary["mean_token_prefix_agreement"] >= args.min_token_prefix_agreement
        and summary["reference_regressions"] <= args.max_reference_regressions
        and decode_nll_passed
    )
    result = {
        "model": args.model,
        "device": str(device),
        "dtype": args.dtype,
        "max_new_tokens": args.max_new_tokens,
        "kv_cache_quantization": args.kv_cache_quantization,
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
