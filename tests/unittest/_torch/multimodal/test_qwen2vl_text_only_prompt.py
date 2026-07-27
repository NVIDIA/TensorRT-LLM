# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from tensorrt_llm._torch.models.modeling_qwen2vl import (
    Qwen2_5VLInputProcessorBase,
    Qwen2VLInputProcessorBase,
)
from tensorrt_llm._torch.models.modeling_qwen3vl import Qwen3VLInputProcessorBase

_PROMPT_TOKEN_IDS = [9001, 17, 42]


class _FakeTokenizer:
    def __init__(self, input_ids):
        self.input_ids = input_ids

    def __call__(self, prompt, return_tensors=None):
        assert return_tensors == "pt"
        return SimpleNamespace(input_ids=torch.tensor([self.input_ids]))


def _make_processor(processor_cls):
    # `_config` / `_processor` / `_dtype` are deliberately left unset: the
    # text-only branch reads nothing but the tokenizer, so a regression that
    # falls through into the multimodal branch fails loudly instead of
    # silently synthesizing multimodal data.
    processor = object.__new__(processor_cls)
    processor._tokenizer = _FakeTokenizer(_PROMPT_TOKEN_IDS)
    return processor


@pytest.mark.parametrize(
    "processor_cls",
    [
        pytest.param(Qwen2VLInputProcessorBase, id="qwen2-vl"),
        pytest.param(Qwen2_5VLInputProcessorBase, id="qwen2.5-vl"),
        pytest.param(Qwen3VLInputProcessorBase, id="qwen3-vl"),
    ],
)
@pytest.mark.parametrize(
    "inputs",
    [
        {"prompt": "a text prompt"},
        {"prompt": "a text prompt", "multi_modal_data": None},
        {"prompt": "a text prompt", "multi_modal_data": {}},
    ],
    ids=["absent", "none", "empty"],
)
def test_text_only_prompt_returns_no_extra_processed_inputs(processor_cls, inputs):
    """A prompt with no multimodal data emits no extra processed inputs at all."""
    processor = _make_processor(processor_cls)

    # Patched on the parametrized class so an inherited *or* overridden
    # implementation is intercepted (Qwen3-VL overrides `_preprocess`).
    with (
        patch.object(processor_cls, "get_mrope_config") as mrope_config,
        patch.object(processor_cls, "_preprocess") as preprocess,
    ):
        token_ids, extra = processor.call_with_text_prompt(inputs, None)

    # `extra is None` is the load-bearing assertion and must not be weakened:
    # it is what keeps the caller (`BaseLLM._preprocess`, llmapi/llm.py:835)
    # from building a `MultimodalParams` at all, which in turn keeps a
    # context-only request out of the mRoPE IPC re-registration at
    # _torch/pyexecutor/model_engine.py:4057-4068, whose `SharedTensorContainer`
    # exports are never rebuilt (and so leak) under disaggregated serving.
    assert extra is None
    assert token_ids == _PROMPT_TOKEN_IDS
    # Guards the two ways the fast path (modeling_qwen2vl.py:876-879) can
    # regress: recomputing a redundant mrope_config, or being deleted outright
    # so control falls through to `_preprocess` at modeling_qwen2vl.py:881.
    mrope_config.assert_not_called()
    preprocess.assert_not_called()


def test_multimodal_prompt_still_takes_the_full_path():
    """Complement check: the fast path did not over-reach past `if not mm_data`.

    This exercises modeling_qwen2vl.py:881-913, which the fix does not touch;
    it passes before and after, and is here only to pin that the gate at :876
    is still the sole thing the text-only path skips on.
    """
    processor = object.__new__(Qwen2VLInputProcessorBase)
    processor._tokenizer = _FakeTokenizer(_PROMPT_TOKEN_IDS)
    # Read by the real `get_rope_index` (modeling_qwen2vl.py:580-583) below.
    processor._config = SimpleNamespace(
        image_token_id=151655,
        video_token_id=151656,
        vision_start_token_id=151652,
        vision_config=SimpleNamespace(spatial_merge_size=2),
    )
    processed_inputs = {
        "input_ids": torch.tensor([[1, 2, 3]]),
        "attention_mask": torch.ones(1, 3, dtype=torch.long),
    }

    # Only the HF processor is stubbed; `get_mrope_config` runs for real so the
    # assertions below read computed values rather than a mock's return_value.
    with patch.object(Qwen2VLInputProcessorBase, "_preprocess", return_value=processed_inputs):
        token_ids, extra = processor.call_with_text_prompt(
            {"prompt": "an image prompt", "multi_modal_data": {"image": [object()]}},
            None,
        )

    assert token_ids == [1, 2, 3]
    mrope_config = extra["multimodal_data"]["mrope_config"]
    assert mrope_config["mrope_position_ids"].shape == (3, 1, 3)
    assert mrope_config["mrope_position_deltas"].shape == (1, 1)
