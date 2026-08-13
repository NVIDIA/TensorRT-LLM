# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Text-only prompts on the Qwen VL input processors carry no MRoPE metadata.

Without vision spans the M-RoPE coordinates degenerate to the scalar token
positions and the position delta is zero, so the processor emits no
`multimodal_data` at all and the model engine falls back to broadcasting the
scalar positions. Synthesizing an (3, 1, N) `mrope_position_ids` tensor per
request instead costs an O(seq_len) device allocation, and in disaggregated
serving the prefill worker re-registers it as a CUDA IPC handle no one reads.
"""

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.models.modeling_qwen2vl import (
    Qwen2_5VLInputProcessorBase,
    Qwen2VLInputProcessorBase,
)
from tensorrt_llm._torch.models.modeling_qwen3vl import Qwen3VLInputProcessorBase

_TOKEN_IDS = [151644, 872, 198, 9707, 151645]

PROCESSOR_CLASSES = [
    Qwen2VLInputProcessorBase,
    Qwen2_5VLInputProcessorBase,
    Qwen3VLInputProcessorBase,
]


class _FakeTokenizer:
    def __call__(self, prompt, return_tensors=None):
        assert prompt == "text prompt"
        assert return_tensors == "pt"
        return SimpleNamespace(input_ids=torch.tensor([_TOKEN_IDS]))


def _make_processor(processor_cls):
    processor = object.__new__(processor_cls)
    processor._tokenizer = _FakeTokenizer()

    def _fail(*args, **kwargs):
        raise AssertionError("get_mrope_config must not run for a text-only prompt")

    processor.get_mrope_config = _fail
    return processor


@pytest.mark.parametrize("processor_cls", PROCESSOR_CLASSES)
@pytest.mark.parametrize("mm_data", [None, {}], ids=["mm_data_none", "mm_data_empty"])
def test_text_only_prompt_emits_no_multimodal_data(processor_cls, mm_data):
    processor = _make_processor(processor_cls)

    token_ids, extra = processor.call_with_text_prompt(
        {"prompt": "text prompt", "multi_modal_data": mm_data},
        sampling_params=None,
    )

    assert token_ids == _TOKEN_IDS
    assert extra is None


@pytest.mark.parametrize("processor_cls", PROCESSOR_CLASSES)
def test_text_only_prompt_without_multi_modal_data_key(processor_cls):
    """`multi_modal_data` absent entirely takes the same path."""
    processor = _make_processor(processor_cls)

    token_ids, extra = processor.call_with_text_prompt(
        {"prompt": "text prompt"},
        sampling_params=None,
    )

    assert token_ids == _TOKEN_IDS
    assert extra is None
