# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""What the Kimi K2.5 / K3 / K3.1 VLM forward hands the language model.

The forward runs on a stand-in for the model that holds only what it touches, with the
MoonViT encoder replaced by fixed image embeddings, so the tests run on CPU.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from tensorrt_llm._torch.models import modeling_kimi_k25
from tensorrt_llm._torch.models.modeling_kimi_k3_vl import (
    KimiK3ForConditionalGeneration,
    KimiK31ForConditionalGeneration,
)
from tensorrt_llm._torch.models.modeling_kimi_k25 import KimiK25ForConditionalGeneration

pytestmark = pytest.mark.cpu_only

_PLACEHOLDER = 9
_HIDDEN = 4
_MODEL_CLASSES = [
    KimiK25ForConditionalGeneration,
    KimiK3ForConditionalGeneration,
    KimiK31ForConditionalGeneration,
]


def _forward_stub():
    """Minimal stand-in for the VLM: only what ``forward`` touches."""
    llm = MagicMock()
    llm.model.embed_tokens = torch.nn.Embedding(16, _HIDDEN)
    llm.forward.return_value = torch.zeros(1, 16)
    return SimpleNamespace(
        llm=llm,
        mm_encoder=SimpleNamespace(forward=None),
        mm_token_ids=torch.tensor([_PLACEHOLDER], dtype=torch.int32),
    )


@pytest.mark.parametrize("model_cls", _MODEL_CLASSES)
def test_fused_image_request_keeps_the_prompt_ids_for_the_drafter(model_cls, monkeypatch):
    """Fusion nulls input_ids, so orig_input_ids is the speculative drafter's only token source."""
    image_embeds = torch.full((2, _HIDDEN), 7.0)
    monkeypatch.setattr(
        modeling_kimi_k25, "get_multimodal_embeddings", lambda **kwargs: [image_embeds]
    )
    monkeypatch.setattr(modeling_kimi_k25, "find_input_mm_embeds", lambda embeds, params: embeds)
    model = _forward_stub()
    input_ids = torch.tensor([1, _PLACEHOLDER, _PLACEHOLDER, 2], dtype=torch.int32)

    model_cls.forward(
        model,
        SimpleNamespace(num_contexts=1, num_generations=0),
        input_ids=input_ids,
        multimodal_params=[object()],
        spec_metadata=object(),
    )

    args, kwargs = model.llm.forward.call_args
    _, fused_ids, _, inputs_embeds, _ = args
    assert fused_ids is None
    assert torch.equal(inputs_embeds[1:3], image_embeds)
    assert torch.equal(kwargs["orig_input_ids"], input_ids)


@pytest.mark.parametrize("model_cls", _MODEL_CLASSES)
def test_text_request_passes_the_same_ids_both_ways(model_cls):
    model = _forward_stub()
    input_ids = torch.tensor([1, 2, 3], dtype=torch.int32)

    model_cls.forward(
        model,
        SimpleNamespace(num_contexts=1, num_generations=0),
        input_ids=input_ids,
    )

    args, kwargs = model.llm.forward.call_args
    _, passed_ids, _, inputs_embeds, _ = args
    assert passed_ids is input_ids
    assert inputs_embeds is None
    assert kwargs["orig_input_ids"] is input_ids
