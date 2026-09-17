# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""MiniMax-M3 VL input processor: text-only prompts take the opt-in prefix-tokenization cache.

The processor is built without a checkpoint (``__new__`` + the attributes ``call_with_text_prompt`` reads), with a
context-free stub tokenizer and a stub HF processor that tokenizes through it, so the test covers the wiring and the
eligibility rules rather than the tokenizer itself (``tests/unittest/inputs/test_prefix_token_cache.py`` does that).
"""

from __future__ import annotations

from typing import Any

import pytest
import torch

from tensorrt_llm._torch.models.modeling_minimaxm3_vl import MiniMaxM3VLInputProcessor
from tensorrt_llm.inputs.prefix_token_cache import ENABLE_ENV_VAR
from tensorrt_llm.sampling_params import SamplingParams

_MIN_CHARS = 16


class _Tokenizer:
    """One id per space-separated word, character offsets: the fast-tokenizer call contract the cache relies on."""

    is_fast = True

    def __call__(
        self, text: str, *, add_special_tokens: bool, return_offsets_mapping: bool = False
    ) -> dict[str, Any]:
        ids, offsets, pos = [], [], 0
        for word in text.split(" "):
            ids.append(sum(map(ord, word)) % 50_000 + 1)
            offsets.append((pos, pos + len(word)))
            pos += len(word) + 1
        out: dict[str, Any] = {"input_ids": ids}
        if return_offsets_mapping:
            out["offset_mapping"] = offsets
        return out

    def apply_chat_template(self, messages: list[dict[str, Any]], **kwargs: Any) -> str:
        return " ".join(str(m["content"]) for m in messages)


class _Processor:
    """Stands in for the HF ``MiniMaxVLProcessor``: tokenizes ``text`` through the tokenizer, counts its calls."""

    def __init__(self, tokenizer: _Tokenizer, extra_leading_ids: tuple[int, ...] = ()) -> None:
        self.tokenizer = tokenizer
        self._extra = list(extra_leading_ids)
        self.calls = 0

    def __call__(
        self,
        text: list[str],
        images: Any = None,
        videos: Any = None,
        return_tensors: str | None = None,
    ) -> dict[str, Any]:
        self.calls += 1
        ids = self._extra + self.tokenizer(text[0], add_special_tokens=False)["input_ids"]
        out: dict[str, Any] = {"input_ids": torch.tensor([ids], dtype=torch.int32)}
        if images:
            out["pixel_values"] = torch.zeros(1, 4, dtype=torch.float32)
            out["image_grid_thw"] = torch.tensor([[1, 2, 2]])
        return out


def _processor(
    monkeypatch: pytest.MonkeyPatch,
    *,
    enabled: bool = True,
    extra_leading_ids: tuple[int, ...] = (),
) -> MiniMaxM3VLInputProcessor:
    if enabled:
        monkeypatch.setenv(ENABLE_ENV_VAR, "1")
        monkeypatch.setenv("TLLM_PREFIX_TOKEN_CACHE_MIN_CHARS", str(_MIN_CHARS))
    else:
        monkeypatch.delenv(ENABLE_ENV_VAR, raising=False)
    ip = MiniMaxM3VLInputProcessor.__new__(MiniMaxM3VLInputProcessor)
    ip._tokenizer = _Tokenizer()
    ip._processor = _Processor(ip._tokenizer, extra_leading_ids)
    ip._dtype = torch.bfloat16
    ip._prefix_token_cache = ip._create_prefix_token_cache()
    return ip


def _turns(n: int) -> list[str]:
    """Chat-shaped prompts: each turn's prompt is the previous prompt (which ends with the assistant generation
    header) followed by the recorded reply, the next user message and the header again, so every prompt extends
    the previous one byte-for-byte, as rendered chat templates do."""
    words = [f"w{i}" for i in range(100 * n)]
    texts, history = [], ""
    for k in range(n):
        history += " ".join(words[100 * k : 100 * (k + 1)]) + " ]~b]ai\n"
        texts.append(history)
        history += " ok done ]~b]user\n "
    return texts


def test_text_only_prompts_use_the_cache_and_match_the_processor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ip = _processor(monkeypatch)
    assert ip._prefix_token_cache is not None
    probe_calls = ip._processor.calls  # one processor call from the equivalence probe
    reference = _Processor(_Tokenizer())
    params = SamplingParams(add_special_tokens=False)
    for k, text in enumerate(_turns(4)):
        ids, extra = ip.call_with_text_prompt({"prompt": text}, params)
        assert ids == reference(text=[text])["input_ids"][0].tolist(), k
        assert extra == {"multimodal_data": {}}
    assert ip._processor.calls == probe_calls, "text-only prompts must not reach the HF processor"
    assert ip._prefix_token_cache.hits == 3 and ip._prefix_token_cache.misses == 1


def test_disabled_by_default_takes_the_processor_path(monkeypatch: pytest.MonkeyPatch) -> None:
    ip = _processor(monkeypatch, enabled=False)
    assert ip._prefix_token_cache is None
    text = _turns(1)[0]
    ids, _ = ip.call_with_text_prompt({"prompt": text})
    assert ids == _Tokenizer()(text, add_special_tokens=False)["input_ids"]
    assert ip._processor.calls == 1


def test_processor_that_adds_tokens_disables_the_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    ip = _processor(monkeypatch, extra_leading_ids=(7,))
    assert ip._prefix_token_cache is None, (
        "a processor that prepends a BOS-like id is not equivalent to the tokenizer"
    )
    text = _turns(1)[0]
    ids, _ = ip.call_with_text_prompt({"prompt": text})
    assert ids[0] == 7


def test_image_requests_bypass_the_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    ip = _processor(monkeypatch)
    calls = ip._processor.calls
    ids, extra = ip.call_with_text_prompt(
        {"prompt": _turns(1)[0], "multi_modal_data": {"image": [object()]}}
    )
    assert ip._processor.calls == calls + 1
    assert "image" in extra["multimodal_data"]
    assert ip._prefix_token_cache.hits == 0 and ip._prefix_token_cache.misses == 0


@pytest.mark.parametrize(
    "params",
    [
        SamplingParams(add_special_tokens=True),
        SamplingParams(add_special_tokens=False, truncate_prompt_tokens=8),
    ],
)
def test_requests_that_add_special_tokens_or_truncate_bypass_the_cache(
    monkeypatch: pytest.MonkeyPatch, params: SamplingParams
) -> None:
    ip = _processor(monkeypatch)
    calls = ip._processor.calls
    ip.call_with_text_prompt({"prompt": _turns(1)[0]}, params)
    assert ip._processor.calls == calls + 1
    assert ip._prefix_token_cache.hits == 0 and ip._prefix_token_cache.misses == 0
