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
"""Transformers baseline-vs-TurboQuant4 KV-cache perplexity canary.

This script does not import ``tensorrt_llm``. It loads the TurboQuant4 Python
reference module by path and scores texts with incremental cached decode. The
TurboQuant4 path applies the quantize/dequantize transform once to each newly
written K/V cache position before the next token is scored.

This is an algorithmic quality canary. It does not validate TensorRT-LLM native
CUDA ops, packed cache allocation, scheduler integration, or serving behavior.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


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


def _load_turboquant4_module():
    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / "tensorrt_llm" / "_torch" / "modules" / "turboquant4.py"
    spec = importlib.util.spec_from_file_location("turboquant4_hf_ppl_target", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _default_texts(long_context_repeats: int) -> list[EvalText]:
    texts = [
        EvalText(
            name="technical_cache",
            text=(
                "TensorRT-LLM stores key and value tensors in a paged cache so "
                "that generation can reuse attention context across decode "
                "steps. Quantizing the cache saves memory, but the logits should "
                "remain close to the baseline model."
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
                "small, the tests explain the risk, and the implementation avoids "
                "unnecessary abstractions."
            ),
        ),
    ]
    if long_context_repeats > 0:
        filler = (
            "This paragraph is padding for a long-context cache quality check. "
            "It includes ordinary facts about deployment reviews, tensor shapes, "
            "and notebook labels, but the important tokens appear later. "
        )
        long_context = filler * long_context_repeats
        texts.append(
            EvalText(
                name="long_context_codename",
                text=(
                    f"{long_context}"
                    "Remember this exact code: ORION-17. "
                    "The code ORION-17 is the only identifier that should be copied "
                    "into the final checklist."
                ),
            )
        )
    return texts


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


def _legacy_cache(past_key_values):
    if hasattr(past_key_values, "to_legacy_cache"):
        return past_key_values.to_legacy_cache()
    return past_key_values


def _validate_cache_head_dim(key: torch.Tensor, layer_index: int) -> None:
    head_dim = key.shape[-1]
    if head_dim % 2 != 0 or (head_dim & (head_dim - 1)) != 0:
        raise ValueError(
            "TurboQuant4 HF perplexity eval requires even power-of-2 attention "
            f"head_dim, got {head_dim} in layer {layer_index}."
        )


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
                    "TurboQuant4 HF perplexity eval cannot find keys/values "
                    f"on cache layer {layer_index}."
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
            "TurboQuant4 HF perplexity eval expects tuple/list past_key_values "
            "or a cache object with layers or to_legacy_cache()."
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


@torch.no_grad()
def _score_incremental(
    model,
    input_ids: torch.Tensor,
    turboquant4=None,
    kv_cache_quantization: str = "both",
) -> tuple[float, int]:
    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ValueError("TurboQuant4 HF perplexity eval expects a single tokenized text.")
    if input_ids.shape[1] < 2:
        raise ValueError("Each eval text must tokenize to at least two tokens.")

    past_key_values = None
    quantized_cache_seq_lens = None
    total_nll = 0.0
    token_count = input_ids.shape[1] - 1

    for token_idx in range(token_count):
        outputs = model(
            input_ids=input_ids[:, token_idx : token_idx + 1],
            past_key_values=past_key_values,
            use_cache=True,
            return_dict=True,
        )
        target = input_ids[:, token_idx + 1]
        total_nll += float(F.cross_entropy(outputs.logits[:, -1, :].float(), target))
        past_key_values = outputs.past_key_values
        if turboquant4 is not None and token_idx + 1 < token_count:
            past_key_values, quantized_cache_seq_lens = _quantize_past_key_values(
                past_key_values,
                turboquant4,
                quantized_cache_seq_lens,
                kv_cache_quantization,
            )

    return total_nll, token_count


def _safe_exp(value: float) -> float:
    if value > 100:
        return float("inf")
    return math.exp(value)


def _score_texts(
    args: argparse.Namespace, texts: list[EvalText]
) -> tuple[list[TextScore], dict[str, Any]]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    turboquant4 = _load_turboquant4_module()
    device = _resolve_device(args.device)
    dtype = _resolve_dtype(args.dtype)
    tokenizer_name = args.tokenizer or args.model
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        trust_remote_code=args.trust_remote_code,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        trust_remote_code=args.trust_remote_code,
    ).to(device)
    model.eval()

    records = []
    total_baseline_nll = 0.0
    total_turbo_nll = 0.0
    total_tokens = 0
    for text in texts:
        encoded = tokenizer(text.text, return_tensors="pt")
        input_ids = encoded["input_ids"][:, : args.max_tokens_per_text].to(device)
        baseline_nll, token_count = _score_incremental(model, input_ids)
        turbo_nll, turbo_token_count = _score_incremental(
            model,
            input_ids,
            turboquant4=turboquant4,
            kv_cache_quantization=args.kv_cache_quantization,
        )
        if token_count != turbo_token_count:
            raise RuntimeError("Baseline and TurboQuant4 token counts diverged.")
        baseline_avg_nll = baseline_nll / token_count
        turbo_avg_nll = turbo_nll / token_count
        baseline_ppl = _safe_exp(baseline_avg_nll)
        turbo_ppl = _safe_exp(turbo_avg_nll)
        relative_ppl_delta = (
            (turbo_ppl - baseline_ppl) / baseline_ppl
            if baseline_ppl and math.isfinite(baseline_ppl)
            else float("inf")
        )
        records.append(
            TextScore(
                name=text.name,
                token_count=token_count,
                baseline_nll=baseline_nll,
                turboquant4_nll=turbo_nll,
                baseline_ppl=baseline_ppl,
                turboquant4_ppl=turbo_ppl,
                nll_delta_per_token=turbo_avg_nll - baseline_avg_nll,
                relative_ppl_delta=relative_ppl_delta,
            )
        )
        total_baseline_nll += baseline_nll
        total_turbo_nll += turbo_nll
        total_tokens += token_count

    baseline_avg_nll = total_baseline_nll / total_tokens
    turbo_avg_nll = total_turbo_nll / total_tokens
    baseline_ppl = _safe_exp(baseline_avg_nll)
    turbo_ppl = _safe_exp(turbo_avg_nll)
    summary = {
        "num_texts": len(records),
        "token_count": total_tokens,
        "baseline_nll_per_token": baseline_avg_nll,
        "turboquant4_nll_per_token": turbo_avg_nll,
        "nll_delta_per_token": turbo_avg_nll - baseline_avg_nll,
        "baseline_ppl": baseline_ppl,
        "turboquant4_ppl": turbo_ppl,
        "relative_ppl_delta": (
            (turbo_ppl - baseline_ppl) / baseline_ppl
            if baseline_ppl and math.isfinite(baseline_ppl)
            else float("inf")
        ),
        "max_text_nll_delta_per_token": max(record.nll_delta_per_token for record in records),
    }
    return records, summary


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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="distilgpt2", help="HF causal LM name or path.")
    parser.add_argument("--tokenizer", default=None, help="Optional tokenizer name or path.")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--device", default="cpu", help="cpu, cuda, mps, or auto.")
    parser.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--max-tokens-per-text", type=int, default=96)
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
    parser.add_argument("--text-file", type=Path, default=None)
    parser.add_argument("--limit-texts", type=int, default=None)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("turboquant4_hf_ppl_eval.json"),
    )
    parser.add_argument("--max-nll-delta-per-token", type=float, default=0.05)
    parser.add_argument("--max-relative-ppl-delta", type=float, default=0.10)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.max_tokens_per_text < 2:
        raise ValueError("--max-tokens-per-text must be at least 2.")
    if args.long_context_repeats < 0:
        raise ValueError("--long-context-repeats must be non-negative.")
    if args.limit_texts is not None and args.limit_texts <= 0:
        raise ValueError("--limit-texts must be positive when provided.")

    texts = (
        _load_text_file(args.text_file)
        if args.text_file
        else _default_texts(args.long_context_repeats)
    )
    if args.limit_texts is not None:
        texts = texts[: args.limit_texts]
    if not texts:
        raise ValueError("At least one eval text is required.")

    records, summary = _score_texts(args, texts)
    thresholds = {
        "max_nll_delta_per_token": args.max_nll_delta_per_token,
        "max_relative_ppl_delta": args.max_relative_ppl_delta,
    }
    passed = (
        summary["nll_delta_per_token"] <= args.max_nll_delta_per_token
        and summary["relative_ppl_delta"] <= args.max_relative_ppl_delta
    )
    result = {
        "model": args.model,
        "device": str(_resolve_device(args.device)),
        "dtype": args.dtype,
        "max_tokens_per_text": args.max_tokens_per_text,
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
