# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""MiniMax-M3 VL input processor with the prefix-tokenization cache.

The processor is built through ``create_input_processor``, as the LLM API does, with
a stub tokenizer and HF processor, so these tests cover the wiring rather than the
cache itself, which tests/unittest/inputs/test_prefix_token_cache.py covers.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest
import torch
import transformers

from tensorrt_llm._torch.models.modeling_minimaxm3_vl import get_minimax_m3_vl_input_processor_cls
from tensorrt_llm.inputs.registry import INPUT_PROCESSOR_REGISTRY, create_input_processor

pytestmark = pytest.mark.cpu_only


class _Tokenizer:
    """One id per space-separated word, prefixed by ``num_special_tokens`` ids."""

    is_fast = True

    def __init__(self, num_special_tokens: int = 0) -> None:
        self._num_special_tokens = num_special_tokens

    def __call__(
        self, text: str, add_special_tokens: bool = True, return_offsets_mapping: bool = False
    ) -> dict[str, Any]:
        ids, offsets, pos = [], [], 0
        for word in text.split(" "):
            ids.append(sum(map(ord, word)) % 50_000 + 1)
            offsets.append((pos, pos + len(word)))
            pos += len(word) + 1
        if add_special_tokens:
            ids = [1] * self._num_special_tokens + ids
        return {"input_ids": ids, "offset_mapping": offsets}

    def num_special_tokens_to_add(self) -> int:
        return self._num_special_tokens


class _HFProcessor:
    """Stands in for ``MiniMaxVLProcessor``: tokenizes ``text`` and counts its calls."""

    def __init__(self, tokenizer: _Tokenizer) -> None:
        self.tokenizer = tokenizer
        self.calls = 0

    def __call__(
        self, text: list[str], images: Any = None, videos: Any = None, return_tensors: Any = None
    ) -> dict[str, Any]:
        self.calls += 1
        out = {"input_ids": torch.tensor([self.tokenizer(text[0])["input_ids"]])}
        if images:
            out["pixel_values"] = torch.zeros(1, 4)
            out["image_grid_thw"] = torch.tensor([[1, 2, 2]])
        return out


def _create_input_processor(processor_cls: type, tokenizer: Any, enabled: bool) -> Any:
    model_cls = object()
    config = SimpleNamespace(image_token_index=200_025, video_token_index=200_026)
    with (
        patch(
            "tensorrt_llm._torch.model_config.ModelConfig.from_pretrained",
            return_value=SimpleNamespace(pretrained_config=config),
        ),
        patch("tensorrt_llm._torch.models.get_model_architecture", return_value=(model_cls, None)),
        patch.dict(
            INPUT_PROCESSOR_REGISTRY._input_processors_cls_by_model_type, {model_cls: processor_cls}
        ),
    ):
        return create_input_processor("unused", tokenizer, enable_tokenization_cache=enabled)


def _input_processor(
    monkeypatch: pytest.MonkeyPatch, enabled: bool = True, num_special_tokens: int = 0
) -> Any:
    monkeypatch.setenv("TLLM_PREFIX_TOKEN_CACHE_MIN_CHARS", "16")
    # Prompts are tokenized by the HF processor's tokenizer; the LLM tokenizer
    # only resolves the vision marker tokens.
    llm_tokenizer = SimpleNamespace(convert_tokens_to_ids=lambda token: 200_029)
    hf_processor = _HFProcessor(_Tokenizer(num_special_tokens))
    with patch.object(transformers.AutoProcessor, "from_pretrained", return_value=hf_processor):
        return _create_input_processor(
            get_minimax_m3_vl_input_processor_cls(), llm_tokenizer, enabled
        )


def _turns(n: int) -> list[str]:
    """Prompts that each extend the previous one."""
    return [" ".join(f"w{i}" for i in range(100 * k)) for k in range(1, n + 1)]


def test_text_only_prompts_use_the_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    ip = _input_processor(monkeypatch)
    for prompt in _turns(4):
        ids, extra = ip({"prompt": prompt}, None)
        assert ids == _Tokenizer()(prompt)["input_ids"]
        assert extra == {"multimodal_data": {}}
    assert ip.processor.calls == 0
    assert ip._prefix_token_cache.hits == 3


def test_cache_is_bypassed_when_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    ip = _input_processor(monkeypatch, enabled=False)
    assert ip._prefix_token_cache is None
    ip({"prompt": _turns(1)[0]}, None)
    assert ip.processor.calls == 1


def test_tokenizer_that_adds_special_tokens_disables_the_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ip = _input_processor(monkeypatch, num_special_tokens=1)
    assert ip._prefix_token_cache is None
    ids, _ = ip({"prompt": _turns(1)[0]}, None)
    assert ip.processor.calls == 1
    assert ids[0] == 1


def test_image_requests_bypass_the_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    ip = _input_processor(monkeypatch)
    _, extra = ip({"prompt": _turns(1)[0], "multi_modal_data": {"image": [object()]}}, None)
    assert ip.processor.calls == 1
    assert "image" in extra["multimodal_data"]
    assert ip._prefix_token_cache.hits == ip._prefix_token_cache.misses == 0


def test_other_input_processors_do_not_receive_the_flag() -> None:
    class _InputProcessor:
        def __init__(self, model_path: str, config: Any, tokenizer: Any, **kwargs: Any) -> None:
            self.kwargs = kwargs

    ip = _create_input_processor(_InputProcessor, None, enabled=True)
    assert "enable_tokenization_cache" not in ip.kwargs
