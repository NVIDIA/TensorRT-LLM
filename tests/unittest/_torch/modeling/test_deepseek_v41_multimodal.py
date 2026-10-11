# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from transformers import PretrainedConfig

from tensorrt_llm._torch.configs.deepseek_v41 import DeepseekV41VisionConfig
from tensorrt_llm._torch.models.deepseek_v41_processor import DeepseekV41InputProcessor
from tensorrt_llm._torch.models.deepseek_v41_vision import (
    DeepseekV41Aligner,
    DeepseekV41VisionModel,
)
from tensorrt_llm._torch.models.modeling_deepseekv41 import DeepseekV41ForCausalLM
from tensorrt_llm._torch.models.modeling_deepseekv41_multimodal import (
    DeepseekV41ForConditionalGeneration,
)
from tensorrt_llm._torch.models.modeling_multimodal_mixin import PreparedLlmInputs
from tensorrt_llm.inputs import create_input_processor_with_hash

pytestmark = pytest.mark.cpu_only
_IMAGE_ID = 129264


@pytest.fixture
def vision_config() -> DeepseekV41VisionConfig:
    return DeepseekV41VisionConfig(
        num_hidden_layers=2,
        hidden_size=16,
        num_attention_heads=2,
        intermediate_size=12,
        patch_size=2,
        downsample_ratio=3,
        rms_norm_eps=1e-6,
    )


def _reference_rotary(x: torch.Tensor, height: int, width: int, theta: float) -> torch.Tensor:
    rotated = torch.empty_like(x)
    quarter, half = x.shape[-1] // 4, x.shape[-1] // 2
    for row in range(height):
        for column in range(width):
            token = row * width + column
            for axis, position in enumerate((row, column)):
                for frequency in range(quarter):
                    first_index = axis * quarter + frequency
                    second_index = first_index + half
                    angle = position / theta ** (frequency / quarter)
                    first, second = x[token, :, first_index], x[token, :, second_index]
                    rotated[token, :, first_index] = first * math.cos(angle) - second * math.sin(
                        angle
                    )
                    rotated[token, :, second_index] = second * math.cos(angle) + first * math.sin(
                        angle
                    )
    return rotated


def _reference_encoder(
    model: DeepseekV41VisionModel,
    aligner: DeepseekV41Aligner,
    config: DeepseekV41VisionConfig,
    patches: torch.Tensor,
    height: int,
    width: int,
) -> torch.Tensor:
    weights = model.state_dict()

    def norm(x: torch.Tensor, name: str) -> torch.Tensor:
        return (
            x * torch.rsqrt(x.square().mean(-1, keepdim=True) + config.rms_norm_eps) * weights[name]
        )

    hidden = F.linear(
        patches.flatten(1), weights["patch_embed.proj.weight"], weights["patch_embed.proj.bias"]
    )
    head_dim = config.hidden_size // config.num_attention_heads
    for index in range(config.num_hidden_layers):
        prefix = f"blocks.{index}."
        qkv = F.linear(
            norm(hidden, prefix + "norm1.weight"),
            weights[prefix + "attn.wqkv.weight"],
            weights[prefix + "attn.wqkv.bias"],
        )
        query, key, value = (
            part.reshape(height * width, config.num_attention_heads, head_dim)
            for part in qkv.chunk(3, dim=-1)
        )
        query = _reference_rotary(query, height, width, config.rope_theta).transpose(0, 1)
        key = _reference_rotary(key, height, width, config.rope_theta).transpose(0, 1)
        probabilities = (query @ key.transpose(-1, -2) / math.sqrt(head_dim)).softmax(-1)
        attended = (
            (probabilities @ value.transpose(0, 1)).transpose(0, 1).reshape(height * width, -1)
        )
        hidden = hidden + F.linear(
            attended, weights[prefix + "attn.wo.weight"], weights[prefix + "attn.wo.bias"]
        )
        gate, up = F.linear(
            norm(hidden, prefix + "norm2.weight"), weights[prefix + "mlp.w1.weight"]
        ).chunk(2, dim=-1)
        hidden = hidden + F.linear(gate * gate.sigmoid() * up, weights[prefix + "mlp.w2.weight"])
    features = norm(hidden, "norm.weight")
    ratio = config.downsample_ratio
    packed = torch.zeros(
        math.ceil(height / ratio) * math.ceil(width / ratio), config.hidden_size * ratio**2
    )
    for row in range(height):
        for column in range(width):
            output_row = row // ratio * math.ceil(width / ratio) + column // ratio
            for channel in range(config.hidden_size):
                output_column = channel * ratio**2 + row % ratio * ratio + column % ratio
                packed[output_row, output_column] = features[row * width + column, channel]
    projected = F.linear(packed, aligner.w1.weight, aligner.w1.bias)
    gelu = projected * 0.5 * (1 + torch.erf(projected / math.sqrt(2)))
    return F.linear(gelu, aligner.w2.weight, aligner.w2.bias)


def _wrapper() -> DeepseekV41ForConditionalGeneration:
    wrapper = DeepseekV41ForConditionalGeneration.__new__(DeepseekV41ForConditionalGeneration)
    nn.Module.__init__(wrapper)
    wrapper.config = SimpleNamespace(hidden_size=2)
    wrapper.image_token_id = _IMAGE_ID
    wrapper.llm = DeepseekV41ForCausalLM.__new__(DeepseekV41ForCausalLM)
    nn.Module.__init__(wrapper.llm)
    wrapper.llm.model = nn.Module()
    wrapper.llm.model.embed_tokens = nn.Embedding(4, 2)
    return wrapper


def test_processor_preserves_interleaved_text_and_image_payload() -> None:
    config = PretrainedConfig(
        text_config={"vocab_size": 129280},
        vision_config={
            "num_hidden_layers": 1,
            "patch_size": 2,
            "downsample_ratio": 3,
            "min_pixels": 0,
        },
        image_token_id=_IMAGE_ID,
    )
    tokenizer = Mock(unk_token_id=None, all_special_tokens=[], all_special_ids=[])
    tokenizer.convert_tokens_to_ids.return_value = _IMAGE_ID
    processor = create_input_processor_with_hash(
        DeepseekV41InputProcessor("unused", config, tokenizer)
    )
    pixels = np.arange(6 * 6 * 3, dtype=np.uint8).reshape(6, 6, 3)
    token_ids = [7, _IMAGE_ID, 8, _IMAGE_ID, 9]
    actual, extra = processor(
        {
            "prompt_token_ids": token_ids,
            "multi_modal_data": {"image": [Image.fromarray(pixels), Image.new("RGB", (14, 8))]},
        },
        None,
    )
    assert actual == [7] + [_IMAGE_ID] * 4 + [8] + [_IMAGE_ID] * 10 + [9]
    assert token_ids == [7, _IMAGE_ID, 8, _IMAGE_ID, 9]
    data = extra["multimodal_data"]
    image = data["image"]
    assert image["positions"] == [1, 6]
    assert image["num_tokens"] == data["multimodal_embedding_lengths"] == [4, 10]
    assert image["image_grid_hw"] == [(3, 3), (4, 7)]
    assert image["llm_grid_hw"] == [(1, 1), (2, 3)]
    assert image["token_types"][1].tolist() == [0, 1, 1, 1, 2, 1, 1, 1, 2, 3]
    expected = torch.from_numpy(pixels).permute(2, 0, 1).float() / 255
    expected = ((expected - 0.5) / 0.5).to(torch.bfloat16)
    expected = expected.reshape(3, 3, 2, 3, 2).permute(1, 3, 0, 2, 4).reshape(9, 3, 2, 2)
    torch.testing.assert_close(image["pixel_values"][0], expected, rtol=0, atol=0)
    assert extra["multimodal_input"] is not None
    tokenizer.encode.assert_not_called()
    tokenizer.decode.assert_not_called()


def test_complete_vision_and_aligner_match_reference(
    vision_config: DeepseekV41VisionConfig,
) -> None:
    torch.manual_seed(23)
    model = DeepseekV41VisionModel(vision_config).eval()
    aligner = DeepseekV41Aligner(vision_config, text_hidden_size=7).eval()
    patches = torch.randn(20, 3, 2, 2)
    norm_weights = {}
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if "norm" in name:
                parameter.copy_(torch.linspace(0.85317, 1.17439, parameter.numel()))
                norm_weights[name] = parameter.clone()
        actual = aligner(model(patches, 4, 5), 4, 5)
        expected = _reference_encoder(model, aligner, vision_config, patches, 4, 5)
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
    model.bfloat16()
    assert model.patch_embed.proj.weight.dtype == torch.bfloat16
    for name, parameter in model.named_parameters():
        if name in norm_weights:
            assert parameter.dtype == torch.float32
            torch.testing.assert_close(parameter, norm_weights[name], rtol=0, atol=0)
    assert model(patches.bfloat16(), 4, 5).dtype == torch.bfloat16


def test_wrapper_encodes_image_spans_and_tracks_active_positions(
    vision_config: DeepseekV41VisionConfig,
) -> None:
    wrapper = _wrapper()
    wrapper.vision = DeepseekV41VisionModel(vision_config).eval()
    wrapper.aligner = DeepseekV41Aligner(vision_config, 2).eval()
    wrapper.image_start = nn.Parameter(torch.tensor([-1.0, -2.0]))
    wrapper.image_end = nn.Parameter(torch.tensor([-3.0, -4.0]))
    wrapper.image_newline = nn.Parameter(torch.tensor([-5.0, -6.0]))
    pixels = torch.randn(20, 3, 2, 2)
    params = [
        SimpleNamespace(
            multimodal_data={
                "image": {
                    "pixel_values": [pixels],
                    "image_grid_hw": [(4, 5)],
                    "llm_grid_hw": [(2, 2)],
                    "num_tokens": [8],
                }
            }
        )
    ]
    output = wrapper.encode_multimodal_inputs(params)
    features = wrapper.aligner(wrapper.vision(pixels, 4, 5), 4, 5)
    torch.testing.assert_close(output[[1, 2, 4, 5]], features)
    torch.testing.assert_close(
        output[[0, 3, 6, 7]],
        torch.stack(
            (wrapper.image_start, wrapper.image_newline, wrapper.image_newline, wrapper.image_end)
        ),
    )
    ids = torch.tensor([_IMAGE_ID] * 10)
    kwargs = wrapper.get_language_model_extra_forward_kwargs(
        raw_input_ids=ids,
        position_ids=None,
        mm_inputs=PreparedLlmInputs(None, torch.zeros(10, 2)),
        multimodal_params=params,
        mm_token_indices=torch.arange(1, 9),
    )
    assert kwargs["orig_input_ids"] is ids
    assert kwargs["image_mask"].tolist() == [False] + [True] * 8 + [False]
    decode = wrapper.get_language_model_extra_forward_kwargs(
        raw_input_ids=ids[:1],
        position_ids=None,
        mm_inputs=PreparedLlmInputs(ids[:1], None),
        multimodal_params=params,
    )
    assert decode["image_mask"] is None
    wrapper.prepare_multimodal_inputs = Mock(return_value=PreparedLlmInputs(ids, None))
    wrapper.llm.forward = Mock(return_value=torch.empty(1, 4))
    for requests in ([object()], None):
        wrapper.forward(
            attn_metadata=SimpleNamespace(num_contexts=1, num_generations=0),
            input_ids=ids,
            **({"context_requests": requests} if requests is not None else {}),
        )
        assert wrapper.llm.forward.call_args.kwargs["context_requests"] is requests


def test_wrapper_request_lifecycle_preserves_image_history() -> None:
    wrapper = _wrapper()
    provider = Mock(config=SimpleNamespace(max_ngram_size=4))
    wrapper.llm.model.engram_hash_provider = provider
    wrapper.llm.model.use_engram = True
    # The third placeholder ID is text; only active image spans stop n-grams.
    request = SimpleNamespace(
        py_request_id=41,
        is_dummy=False,
        context_current_position=5,
        prompt_len=5,
        multimodal_positions=[1],
        multimodal_lengths=[3],
        get_tokens_range=Mock(return_value=[_IMAGE_ID] * 3),
    )
    scheduled = SimpleNamespace(
        context_requests=[], generation_requests=[request], all_requests=lambda: [request]
    )
    metadata = SimpleNamespace(kv_cache_manager=SimpleNamespace(max_seq_len=128))
    wrapper.prepare_request_inputs(scheduled, metadata, {41})
    provider.seed_context_history.assert_called_once_with(
        [41],
        {41: (5, [_IMAGE_ID] * 3)},
        max_seq_len=128,
        device=torch.device("cpu"),
        token_masks={41: [False, False, True]},
    )
    request.py_disaggregated_params = SimpleNamespace(
        multimodal_positions=request.multimodal_positions,
        multimodal_lengths=request.multimodal_lengths,
    )
    request.multimodal_positions = request.multimodal_lengths = None
    wrapper.prepare_disagg_generation_request(request)
    provider.queue_history_seed.assert_called_once_with(
        41, 2, [_IMAGE_ID] * 3, token_mask=[False, False, True]
    )
    wrapper.release_request_state(41)
    provider.release_request_state.assert_called_once_with(41)
