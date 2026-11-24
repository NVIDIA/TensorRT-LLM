# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""What the Kimi K3 / K3.1 VLM wrapper hands the language model, which requests it admits, and how the engine treats
it.

The forward and the admission check run on a stand-in for the model that holds only what they touch, with the
MoonViT encoder replaced by fixed image embeddings, so the tests run on CPU.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.input_processor import (
    K31InputProcessor,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kimi_k31_pro_mxfp4__sm_10x__default import (
    modeling as target_modeling,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kimi_k31_pro_mxfp4__sm_10x__default.modeling import (
    K31VLWrapper,
    KimiK31ForConditionalGeneration,
)
from tensorrt_llm._torch.pyexecutor.engine.multimodal import (
    is_multimodal,
    mm_item_scheduling_enabled,
)
from tensorrt_llm.llmapi.llm_args import MultimodalEncoderSchedulingPolicy

pytestmark = pytest.mark.cpu_only

_PLACEHOLDER = 9
_HIDDEN = 4
_MODEL_CLASSES = [
    K31VLWrapper,
    KimiK31ForConditionalGeneration,
]


def _forward_stub(model_cls=KimiK31ForConditionalGeneration):
    """Minimal stand-in for the VLM: only what ``forward`` and the admission check touch, with the model class's
    own local-tower property."""
    llm = MagicMock()
    llm.model.embed_tokens = torch.nn.Embedding(16, _HIDDEN)
    llm.forward.return_value = torch.zeros(1, 16)
    stub = type("_VLMStub", (), {"_local_tower": model_cls._local_tower})()
    stub.llm = llm
    stub.mm_encoder = SimpleNamespace(forward=None, max_image_grid=None)
    # A server that runs the vision encoder (no disable_mm_encoder / disaggregated encoder).
    stub._skip_mm_encoder = False
    return stub


@pytest.mark.parametrize("model_cls", _MODEL_CLASSES)
def test_fused_image_request_keeps_the_prompt_ids_for_the_drafter(model_cls, monkeypatch):
    """Fusion nulls input_ids, so orig_input_ids is the speculative drafter's only token source."""
    image_embeds = torch.full((2, _HIDDEN), 7.0)
    monkeypatch.setattr(
        target_modeling, "get_multimodal_embeddings", lambda **kwargs: [image_embeds]
    )
    monkeypatch.setattr(target_modeling, "find_input_mm_embeds", lambda embeds, params: embeds)
    model = _forward_stub(model_cls)
    input_ids = torch.tensor([1, _PLACEHOLDER, _PLACEHOLDER, 2], dtype=torch.int32)

    model_cls.forward(
        model,
        SimpleNamespace(num_contexts=1, num_generations=0, kv_cache_manager=None),
        input_ids=input_ids,
        multimodal_params=[object()],
        spec_metadata=object(),
        mm_token_indices=torch.tensor([1, 2]),
    )

    args, kwargs = model.llm.forward.call_args
    _, fused_ids, _, inputs_embeds, _ = args
    assert fused_ids is None
    assert torch.equal(inputs_embeds[1:3], image_embeds)
    assert torch.equal(kwargs["orig_input_ids"], input_ids)


@pytest.mark.parametrize("model_cls", _MODEL_CLASSES)
def test_text_request_passes_the_same_ids_both_ways(model_cls):
    model = _forward_stub(model_cls)
    input_ids = torch.tensor([1, 2, 3], dtype=torch.int32)

    model_cls.forward(
        model,
        SimpleNamespace(num_contexts=1, num_generations=0, kv_cache_manager=None),
        input_ids=input_ids,
    )

    args, kwargs = model.llm.forward.call_args
    _, passed_ids, _, inputs_embeds, _ = args
    assert passed_ids is input_ids
    assert inputs_embeds is None
    assert kwargs["orig_input_ids"] is input_ids


class _Tower:
    """Stand-in for the vision tower: counts the profiling encodes."""

    max_image_grid = (1, 4, 6)

    def __init__(self):
        self.encodes = 0

    def encode_largest_image(self):
        self.encodes += 1
        return torch.zeros(6, _HIDDEN)


def _metadata(num_contexts, estimating):
    return SimpleNamespace(
        num_contexts=num_contexts,
        kv_cache_manager=SimpleNamespace(is_estimating_kv_cache=estimating),
    )


def test_text_forward_runs_while_the_profiling_rows_are_held():
    """The rows stay alive through the text model's forward, as an image step keeps its rows."""
    model = _forward_stub()
    model.mm_encoder = _Tower()
    seen = []
    model.llm.forward.side_effect = lambda *args, **kwargs: seen.append(
        model.mm_encoder.encodes
    ) or torch.zeros(1)

    KimiK31ForConditionalGeneration.forward(
        model, _metadata(1, True), input_ids=torch.tensor([1, 2, 3], dtype=torch.int32)
    )
    assert seen == [1]


@pytest.mark.parametrize("model_cls", _MODEL_CLASSES)
def test_images_are_encoded_inline(model_cls):
    """The model is multimodal through its input processor, without item-level encoder scheduling."""
    llm_args = SimpleNamespace(
        disable_mm_encoder=False,
        pipeline_parallel_size=1,
        multimodal_config=SimpleNamespace(
            encoder_scheduling_policy=MultimodalEncoderSchedulingPolicy.DEFAULT
        ),
    )
    model = model_cls.__new__(model_cls)

    assert mm_item_scheduling_enabled(llm_args, model) is False
    assert is_multimodal(model, K31InputProcessor.__new__(K31InputProcessor)) is True


class _RejectingTower:
    def validate_multimodal_data(self, mm_data):
        raise ValueError("MoonViT cannot encode the image grid")


def _request(media=True, generation_only=False):
    """An admitted request as the executor's ``validate_request`` hook sees it."""
    mm_data = None
    if media:
        mm_data = {
            "image": {
                "pixel_values": torch.zeros(4, 3, 14, 14),
                "image_grid_thw": torch.tensor([[1, 2, 2]]),
            }
        }
    return SimpleNamespace(py_multimodal_data=mm_data, is_generation_only_request=generation_only)


def _without_a_local_tower(model, no_tower):
    if no_tower == "checkpoint_without_tower":
        model.mm_encoder = None
    else:
        model._skip_mm_encoder = True


def test_requests_are_validated_by_the_tower():
    model = _forward_stub()
    model.mm_encoder = _RejectingTower()
    with pytest.raises(ValueError, match="MoonViT cannot encode"):
        KimiK31ForConditionalGeneration.validate_request(model, _request())


@pytest.mark.parametrize("model_cls", _MODEL_CLASSES)
@pytest.mark.parametrize("no_tower", ["checkpoint_without_tower", "disable_mm_encoder"])
def test_image_requests_without_a_local_tower_are_refused_at_admission(model_cls, no_tower):
    """Without a local vision tower the executor's admission check refuses an image request, which fails that request
    alone; the forward of the batch's text requests runs as before."""
    model = _forward_stub(model_cls)
    _without_a_local_tower(model, no_tower)

    with pytest.raises(ValueError, match="runs no vision tower"):
        model_cls.validate_request(model, _request())
    model_cls.validate_request(model, _request(media=False))

    input_ids = torch.tensor([1, 2, 3], dtype=torch.int32)
    model_cls.forward(
        model,
        SimpleNamespace(num_contexts=1, num_generations=0, kv_cache_manager=None),
        input_ids=input_ids,
    )
    args, _ = model.llm.forward.call_args
    assert args[1] is input_ids


@pytest.mark.parametrize("model_cls", _MODEL_CLASSES)
@pytest.mark.parametrize("no_tower", ["checkpoint_without_tower", "disable_mm_encoder"])
def test_generation_only_image_requests_pass_without_a_local_tower(model_cls, no_tower):
    """The decode side of P/D disaggregation receives the context's KV and never encodes: a generation-only request
    with media is admitted on a server without a vision tower."""
    model = _forward_stub(model_cls)
    _without_a_local_tower(model, no_tower)

    model_cls.validate_request(model, _request(generation_only=True))
