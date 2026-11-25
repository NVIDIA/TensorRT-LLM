# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Construction, weight ownership and forward behavior of a V2 multimodal model."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn
from transformers import PretrainedConfig

from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.modeling_v2.multimodal import (
    MultimodalEncoderSpec,
    MultimodalModelForCausalLM,
)
from tensorrt_llm._torch.pyexecutor.engine.multimodal import _validate_multimodal_input
from tensorrt_llm.inputs.multimodal import (
    MULTIMODAL_ENCODER_ITEM_METADATA_KEY,
    MultimodalItemMetadata,
    MultimodalParams,
    MultimodalRuntimeData,
)
from tensorrt_llm.llmapi.llm_args import MultimodalConfig

pytestmark = pytest.mark.cpu_only

IMAGE_TOKEN = 7
PROMPT = torch.tensor([1, IMAGE_TOKEN, 2, IMAGE_TOKEN, 3], dtype=torch.int32)


class _TinyLlm(nn.Module):
    def __init__(self, model_config):
        super().__init__()
        self.model_config = model_config
        self.config = model_config.pretrained_config.text_config
        model_config.pretrained_config = self.config
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding.from_pretrained(torch.arange(8.0)[:, None])
        self.multimodal_token_ids = torch.tensor([IMAGE_TOKEN], dtype=torch.int32)

    def forward(self, **kwargs):
        return kwargs

    def load_weights(self, weights):
        self.loaded = weights


class _TinyTower(nn.Module):
    modality = "image"
    supports_encoder_cache = True

    def __init__(self, config, dtype):
        super().__init__()
        self.calls = 0

    def build_batched_input(self, params):
        return {"values": torch.cat([p.multimodal_data[self.modality]["values"] for p in params])}

    def forward(self, values):
        self.calls += 1
        return values

    def load_weights(self, weights):
        self.loaded = weights


class _TinyModel(MultimodalModelForCausalLM):
    llm_class = _TinyLlm
    mm_encoders = {"vision": MultimodalEncoderSpec(_TinyTower, "model.visual.", ("image",))}


class _InputIdsModel(_TinyModel):
    llm_requires_input_ids = True


def _model(cls=_TinyModel, disable=False):
    config = PretrainedConfig(
        text_config=PretrainedConfig(hidden_size=1, torch_dtype=torch.float32),
        vision_config=PretrainedConfig(),
    )
    return cls(
        ModelConfig(
            pretrained_config=config,
            disable_mm_encoder=disable,
            multimodal_config=MultimodalConfig(encoder_cache_max_bytes=4096),
        )
    )


def _params(*values, runtime=None):
    return MultimodalParams(
        multimodal_data={
            "image": {"values": torch.tensor(values, dtype=torch.float32)[:, None]},
            MULTIMODAL_ENCODER_ITEM_METADATA_KEY: MultimodalItemMetadata(
                [("image", 0)], output_embedding_lengths=[len(values)]
            ),
        },
        multimodal_runtime=runtime,
    )


def _forward(model, params, indices=(1, 3)):
    return model(
        SimpleNamespace(num_contexts=1, num_generations=0),
        PROMPT,
        multimodal_params=[params],
        mm_token_indices=torch.tensor(indices),
        text_token_indices=torch.tensor([0, 2, 4]),
    )


@pytest.mark.parametrize("disable", [False, True], ids=["tower", "disable_mm_encoder"])
def test_construction_and_checkpoint_ownership(disable, monkeypatch):
    if disable:
        monkeypatch.setattr(_TinyTower, "__init__", lambda *args: pytest.fail("constructed tower"))

    class Checkpoint(dict):
        def __getitem__(self, key):
            if disable and key.startswith("model.visual."):
                pytest.fail("read disabled tower weight")
            return super().__getitem__(key)

    model = _model(disable=disable)
    weights = Checkpoint({"model.visual.patch": 5, "model.layers.0.weight": 3})
    model.load_weights(weights)
    assert model.encoder_cache_active == (not disable)
    if disable:
        assert model.mm_encoder is None
    else:
        assert model.mm_encoder.loaded == {"model.visual.patch": 5}
    assert model.llm.loaded == {"model.layers.0.weight": 3}


def test_attached_embeddings_require_encoder_prefill_disaggregation(monkeypatch):
    model = _model(disable=True)
    data = {"multimodal_embedding": torch.ones(2, 1)}
    monkeypatch.setenv("TLLM_MULTIMODAL_DISAGGREGATED", "1")
    _validate_multimodal_input(model, data)
    monkeypatch.setenv("TLLM_MULTIMODAL_DISAGGREGATED", "0")
    with pytest.raises(ValueError, match="disaggregated"):
        _validate_multimodal_input(model, data)


def test_two_towers_restore_interleaved_prompt_order_across_requests():
    class Audio(_TinyTower):
        modality = "audio"

    class TwoTowers(_TinyModel):
        mm_encoders = {
            **_TinyModel.mm_encoders,
            "audio": MultimodalEncoderSpec(Audio, "audio.", ("audio",)),
        }

    model = _model(TwoTowers)
    params = _params(30, 40, 50)
    params.multimodal_data["audio"] = {"values": torch.tensor([[10.0], [20.0], [60.0]])}
    params.multimodal_data[MULTIMODAL_ENCODER_ITEM_METADATA_KEY] = MultimodalItemMetadata(
        [("audio", 0), ("image", 0), ("audio", 1), ("image", 1)],
        output_embedding_lengths=[2, 2, 1, 1],
    )
    embeds = model.encode_multimodal_inputs([params, _params(70)])
    assert embeds[:, 0].tolist() == [10, 20, 30, 40, 60, 50, 70]
    assert model.mm_encoders["vision"].calls == 1
    assert model.mm_encoders["audio"].calls == 1


def test_registration_requires_an_encoder_packing_hook():
    with pytest.raises(TypeError, match="build_batched_input"):

        class Incomplete(_TinyModel):
            mm_encoders = {"vision": MultimodalEncoderSpec(nn.Identity, "vision.", ("image",))}


def test_custom_forward_uses_shared_multimodal_preparation():
    class Custom(_TinyModel):
        def forward(self, input_ids, multimodal_params, num_context_requests=1):
            prepared = self.prepare_multimodal_embeds(
                input_ids=input_ids,
                positions=None,
                multimodal_params=multimodal_params,
                num_context_requests=num_context_requests,
            )
            return prepared[1][0] if prepared is not None else None

    class WithoutPreparation(Custom):
        def forward(self, input_ids, multimodal_params):
            return None

    model = _model(Custom)
    params = [_params(8, 9)]
    assert model(PROMPT, params).tolist() == [[8], [9]]
    with pytest.raises(RuntimeError, match="prepare_multimodal"):
        _model(WithoutPreparation)(PROMPT, params)


@pytest.mark.parametrize("cls", [_TinyModel, _InputIdsModel], ids=["fused", "input_ids"])
def test_embeddings_and_placeholder_positions_reach_language_model(cls):
    received = _forward(_model(cls), _params(8, 9))
    if cls.llm_requires_input_ids:
        assert received["input_ids"] is PROMPT and received["inputs_embeds"] is None
        assert received["mm_embeds"].tolist() == [[8], [9]]
        assert received["mm_token_indices"].tolist() == [1, 3]
    else:
        assert received["input_ids"] is None
        assert received["inputs_embeds"].tolist() == [[1], [8], [2], [9], [3]]


def test_a_chunk_hands_over_its_multimodal_embeddings():
    runtime = MultimodalRuntimeData(
        embed_mask_cumsum=torch.arange(1, 5), past_seen_token_num=2, chunk_end_pos=4
    )
    received = _forward(_model(_InputIdsModel), _params(6, 7, 8, 9, runtime=runtime))
    assert received["mm_embeds"].tolist() == [[8], [9]]


@pytest.mark.parametrize(
    "num_contexts, params", [(1, []), (0, [_params(8, 9)])], ids=["text_prefill", "generation"]
)
def test_a_step_without_media_runs_no_tower(num_contexts, params):
    model = _model()
    tokens = torch.tensor([3])
    received = model(
        SimpleNamespace(num_contexts=num_contexts, num_generations=1 - num_contexts),
        tokens,
        multimodal_params=params,
    )
    assert received["input_ids"] is tokens and received["inputs_embeds"] is None
    assert model.mm_encoder.calls == 0


def test_the_multimodal_token_count_must_match_the_placeholder_indices():
    with pytest.raises(AssertionError, match="2 multimodal tokens and 1 placeholder indices"):
        _forward(_model(_InputIdsModel), _params(8, 9), indices=(1,))
