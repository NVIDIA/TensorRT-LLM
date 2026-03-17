# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for the Phi-4-reasoning-vision-15B custom model implementation."""

import json
import os
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import torch
from _model_test_utils import assert_rmse_close
from PIL import Image
from torch.export import Dim
from transformers import Siglip2VisionModel
from transformers.models.phi3.configuration_phi3 import Phi3Config
from transformers.models.siglip2.image_processing_siglip2 import Siglip2ImageProcessor

import tensorrt_llm._torch.auto_deploy.custom_ops  # noqa: F401
from tensorrt_llm._torch.auto_deploy.export import torch_export_to_gm
from tensorrt_llm._torch.auto_deploy.models.custom.modeling_phi4_visionr import (
    IMAGE_TOKEN_INDEX,
    Phi4VisionAttention,
    Phi4VisionDecoderLayer,
    Phi4VisionMLP,
    Phi4VisionRConfig,
    Phi4VisionRForConditionalGeneration,
    Phi4VisionRotaryEmbedding,
    Phi4VisionRProcessorWrapper,
    Phi4VisionRTokenizer,
    build_vision_projector,
)
from tensorrt_llm._torch.auto_deploy.models.hf import (
    AutoModelForCausalLMFactory,
    AutoModelForImageTextToTextFactory,
)
from tensorrt_llm._torch.auto_deploy.utils._graph import move_to_device

_BATCH_AND_SEQUENCE_TEST_CASES = ((2, 6), (1, 8))
_HF_MODEL_DIR = "models--microsoft--Phi-4-reasoning-vision-15B"


@pytest.fixture(scope="function", autouse=True)
def set_seed():
    torch.manual_seed(42)


def _create_small_config() -> Phi4VisionRConfig:
    return Phi4VisionRConfig(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        hidden_act="silu",
        max_position_embeddings=256,
        original_max_position_embeddings=256,
        rms_norm_eps=1e-5,
        rope_theta=10000.0,
        rope_scaling=None,
        partial_rotary_factor=1.0,
        attention_dropout=0.0,
        attention_bias=False,
        resid_pdrop=0.0,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        tie_word_embeddings=False,
        initializer_range=0.02,
        mm_vision_tower="google/siglip2-so400m-patch16-naflex",
        mm_projector_type="mlp2x_gelu",
        mm_hidden_size=32,
        min_num_patches=4,
        max_num_patches=16,
        tokenizer_model_max_length=256,
        vision_config={
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "patch_size": 16,
            "num_patches": 16,
        },
    )


def _create_hf_text_config():
    config = Phi3Config(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        hidden_act="silu",
        max_position_embeddings=256,
        original_max_position_embeddings=256,
        rms_norm_eps=1e-5,
        rope_theta=10000.0,
        rope_scaling=None,
        partial_rotary_factor=1.0,
        attention_dropout=0.0,
        attention_bias=False,
        resid_pdrop=0.0,
        pad_token_id=0,
    )
    config._attn_implementation = "eager"
    return config


class _RemoteLikePhi4VisionRConfig(Phi3Config):
    model_type = "phi4-siglip"


def _make_small_multimodal_inputs():
    processor = Siglip2ImageProcessor(patch_size=16, max_num_patches=16)
    image = Image.fromarray(np.full((32, 48, 3), 127, dtype=np.uint8))
    vision_inputs = processor(images=image, return_tensors="pt")
    input_ids = torch.tensor([[5, IMAGE_TOKEN_INDEX, 6, 7]], dtype=torch.long)
    return input_ids, vision_inputs


def _make_causal_mask(seq_len, device, dtype):
    mask = torch.full((seq_len, seq_len), torch.finfo(dtype).min, device=device, dtype=dtype)
    mask = torch.triu(mask, diagonal=1)
    return mask.unsqueeze(0).unsqueeze(0)


def _find_checkpoint_index() -> Path | None:
    hf_home = os.environ.get("HF_HOME")
    candidate_roots = [Path(".tmp/hf_home")]
    if hf_home:
        candidate_roots.insert(0, Path(hf_home))
    candidate_roots.append(Path.home() / ".cache" / "huggingface")

    for root in candidate_roots:
        snapshot_dir = root / "hub" / _HF_MODEL_DIR / "snapshots"
        if not snapshot_dir.exists():
            continue
        for index_path in sorted(snapshot_dir.glob("*/model.safetensors.index.json")):
            return index_path
    return None


def test_phi4_visionr_config_parses_flat_text_and_vision():
    config = _create_small_config()
    assert config.model_type == "phi4-siglip"
    assert config.text_config.model_type == "phi3"
    assert config.vision_config.model_type == "siglip2_vision_model"
    assert config.vision_config.patch_size == 16
    assert config.mm_projector_type == "mlp2x_gelu"


def test_phi4_visionr_text_forward_requires_position_ids():
    config = _create_small_config()
    model = Phi4VisionRForConditionalGeneration(config).model.eval()
    input_ids = torch.randint(0, config.text_config.vocab_size, (1, 4), dtype=torch.long)

    with pytest.raises(ValueError, match="position_ids must be provided"):
        model(input_ids=input_ids)


def test_phi4_visionr_accepts_flat_remote_like_config():
    remote_config = _RemoteLikePhi4VisionRConfig(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        hidden_act="silu",
        max_position_embeddings=256,
        original_max_position_embeddings=256,
        rms_norm_eps=1e-5,
        rope_theta=10000.0,
        rope_scaling=None,
        partial_rotary_factor=1.0,
        attention_dropout=0.0,
        attention_bias=False,
        resid_pdrop=0.0,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        tie_word_embeddings=False,
        initializer_range=0.02,
        mm_vision_tower="google/siglip2-so400m-patch16-naflex",
        mm_projector_type="mlp2x_gelu",
        mm_hidden_size=32,
        min_num_patches=4,
        max_num_patches=16,
        tokenizer_model_max_length=256,
        vision_config={
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "patch_size": 16,
            "num_patches": 16,
        },
    )

    model = Phi4VisionRForConditionalGeneration(remote_config).eval()
    assert model.model.config.model_type == "phi3"
    assert model.model.full_config.model_type == "phi4-siglip"
    assert model.model.full_config.text_config.hidden_size == 64


def test_phi4_visionr_is_registered_for_both_hf_factories():
    assert (
        AutoModelForCausalLMFactory._custom_model_mapping["Phi4VisionR"]
        is Phi4VisionRForConditionalGeneration
    )
    assert (
        AutoModelForImageTextToTextFactory._custom_model_mapping["Phi4VisionR"]
        is Phi4VisionRForConditionalGeneration
    )


def test_phi4_visionr_processor_wrapper_uses_tokenizer_chat_template():
    class FakeTokenizer:
        def __init__(self):
            self.chat_template = "{{ messages }}"
            self.calls = []

        def apply_chat_template(self, messages, **kwargs):
            self.calls.append(kwargs)
            if kwargs["tokenize"]:
                return {"input_ids": torch.tensor([[11, 12, 13]], dtype=torch.long)}
            return "formatted prompt"

    class FakeProcessor:
        def __init__(self, tokenizer):
            self.tokenizer = tokenizer

    wrapped_processor = Phi4VisionRProcessorWrapper(FakeProcessor(FakeTokenizer()))

    prompt_token_ids = wrapped_processor.apply_chat_template(
        [{"role": "user", "content": "hello"}],
        tokenize=True,
        return_dict=True,
    )

    assert torch.equal(
        prompt_token_ids["input_ids"], torch.tensor([[11, 12, 13]], dtype=torch.long)
    )
    assert len(wrapped_processor.tokenizer.tokenizer.calls) == 1


def test_phi4_visionr_tokenizer_wraps_prompt_with_chat_template():
    class FakeTokenizer:
        def __init__(self):
            self.chat_template = "{{ messages }}"
            self.calls = []
            self.encoded = []

        def apply_chat_template(self, messages, **kwargs):
            self.calls.append((messages, kwargs))
            return "formatted phi4 prompt"

        def encode(self, text, *args, **kwargs):
            self.encoded.append((text, kwargs))
            return [21, 22]

    tokenizer = Phi4VisionRTokenizer(FakeTokenizer())
    prompt_token_ids = tokenizer.encode("Where is the capital of Iceland?")

    assert prompt_token_ids == [21, 22]
    assert tokenizer.tokenizer.calls[0][0] == [
        {"role": "user", "content": "Where is the capital of Iceland?"}
    ]
    assert tokenizer.tokenizer.encoded[0][0] == "formatted phi4 prompt"


def test_phi4_visionr_state_dict_hierarchy_matches_checkpoint_layout():
    config = _create_small_config()
    model = Phi4VisionRForConditionalGeneration(config).eval()
    state_dict = model.state_dict()
    assert "model.embed_tokens.weight" in state_dict
    assert "model.layers.0.self_attn.qkv_proj.weight" in state_dict
    assert "model.norm.weight" in state_dict
    assert (
        "model.vision_tower.vision_tower.vision_model.embeddings.patch_embedding.weight"
        in state_dict
    )
    assert "model.mm_projector.0.weight" in state_dict


def test_phi4_visionr_state_dict_matches_real_checkpoint_index():
    index_path = _find_checkpoint_index()
    if index_path is None:
        pytest.skip("Missing HF checkpoint index in the local Hugging Face cache")

    weight_map = json.loads(index_path.read_text())["weight_map"]
    expected_keys = [
        "model.embed_tokens.weight",
        "model.layers.0.self_attn.qkv_proj.weight",
        "model.norm.weight",
        "model.vision_tower.vision_tower.vision_model.embeddings.patch_embedding.weight",
        "model.vision_tower.vision_tower.vision_model.encoder.layers.0.self_attn.q_proj.weight",
        "model.vision_tower.vision_tower.vision_model.post_layernorm.weight",
        "model.mm_projector.0.weight",
        "lm_head.weight",
    ]
    for key in expected_keys:
        assert key in weight_map, f"Expected checkpoint key missing from HF index: {key}"

    config = _create_small_config()
    model = Phi4VisionRForConditionalGeneration(config).eval()
    state_dict = model.state_dict()
    for key in expected_keys:
        assert key in state_dict, f"Expected model state_dict key missing: {key}"


@torch.no_grad()
def test_phi4_visionr_mlp_block_matches_hf_phi3():
    from transformers.models.phi3.modeling_phi3 import Phi3MLP

    device = "cpu"
    dtype = torch.float32
    config = _create_small_config().text_config
    hf_mlp = Phi3MLP(_create_hf_text_config()).to(device=device, dtype=dtype).eval()
    custom_mlp = Phi4VisionMLP(config).to(device=device, dtype=dtype).eval()
    custom_mlp.load_state_dict(hf_mlp.state_dict())

    hidden_states = torch.randn(2, 6, config.hidden_size, device=device, dtype=dtype)
    hf_out = hf_mlp(hidden_states)
    custom_out = custom_mlp(hidden_states)
    torch.testing.assert_close(custom_out, hf_out, rtol=1e-4, atol=1e-4)


@torch.no_grad()
def test_phi4_visionr_attention_block_matches_hf_phi3():
    from transformers.models.phi3.modeling_phi3 import Phi3Attention, Phi3RotaryEmbedding

    device = "cpu"
    dtype = torch.float32
    config = _create_small_config().text_config
    hf_config = _create_hf_text_config()

    hf_attention = Phi3Attention(hf_config, layer_idx=0).to(device=device, dtype=dtype).eval()
    custom_attention = (
        Phi4VisionAttention(config, layer_idx=0).to(device=device, dtype=dtype).eval()
    )
    custom_attention.load_state_dict(hf_attention.state_dict())

    rotary_dim = config.hidden_size // config.num_attention_heads
    hf_rotary = Phi3RotaryEmbedding(hf_config).to(device=device, dtype=dtype)
    custom_rotary = Phi4VisionRotaryEmbedding(
        rotary_dim,
        max_position_embeddings=config.max_position_embeddings,
        base=config.rope_theta,
    ).to(device=device, dtype=dtype)

    batch_size, seq_len = 2, 6
    hidden_states = torch.randn(batch_size, seq_len, config.hidden_size, device=device, dtype=dtype)
    position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)

    hf_position_embeddings = hf_rotary(hidden_states, position_ids)
    hf_out, _ = hf_attention(
        hidden_states=hidden_states,
        position_embeddings=hf_position_embeddings,
        attention_mask=_make_causal_mask(seq_len, device, dtype),
    )

    custom_out = custom_attention(hidden_states, position_ids, custom_rotary(hidden_states))
    assert_rmse_close(custom_out.float(), hf_out.float(), rmse_ratio_tol=0.10)


@torch.no_grad()
def test_phi4_visionr_decoder_layer_matches_hf_phi3():
    from transformers.models.phi3.modeling_phi3 import Phi3DecoderLayer, Phi3RotaryEmbedding

    device = "cpu"
    dtype = torch.float32
    config = _create_small_config().text_config
    hf_config = _create_hf_text_config()

    hf_layer = Phi3DecoderLayer(hf_config, layer_idx=0).to(device=device, dtype=dtype).eval()
    custom_layer = Phi4VisionDecoderLayer(config, layer_idx=0).to(device=device, dtype=dtype).eval()
    custom_layer.load_state_dict(hf_layer.state_dict())

    rotary_dim = config.hidden_size // config.num_attention_heads
    hf_rotary = Phi3RotaryEmbedding(hf_config).to(device=device, dtype=dtype)
    custom_rotary = Phi4VisionRotaryEmbedding(
        rotary_dim,
        max_position_embeddings=config.max_position_embeddings,
        base=config.rope_theta,
    ).to(device=device, dtype=dtype)

    batch_size, seq_len = 2, 6
    hidden_states = torch.randn(batch_size, seq_len, config.hidden_size, device=device, dtype=dtype)
    position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)

    hf_position_embeddings = hf_rotary(hidden_states, position_ids)
    hf_out = hf_layer(
        hidden_states=hidden_states,
        position_ids=position_ids,
        position_embeddings=hf_position_embeddings,
        attention_mask=_make_causal_mask(seq_len, device, dtype),
    )
    custom_out = custom_layer(hidden_states, position_ids, custom_rotary(hidden_states))
    assert_rmse_close(custom_out.float(), hf_out.float(), rmse_ratio_tol=0.05)


@pytest.mark.parametrize("B,S", _BATCH_AND_SEQUENCE_TEST_CASES)
@torch.no_grad()
def test_phi4_visionr_text_model_matches_hf_phi3(B, S):
    from transformers.models.phi3.modeling_phi3 import Phi3Model

    device = "cpu"
    dtype = torch.float32
    config = _create_small_config()
    hf_config = _create_hf_text_config()

    custom_model = Phi4VisionRForConditionalGeneration(config).model
    custom_model.to(device=device, dtype=dtype)
    custom_model.eval()

    hf_model = Phi3Model(hf_config).to(device=device, dtype=dtype)
    hf_model.eval()

    custom_model.load_state_dict(hf_model.state_dict(), strict=False)

    input_ids = torch.randint(0, config.text_config.vocab_size, (B, S), device=device)
    position_ids = torch.arange(S, device=device).unsqueeze(0).expand(B, -1)

    custom_out = custom_model(input_ids=input_ids, position_ids=position_ids).last_hidden_state
    hf_out = hf_model(input_ids=input_ids, position_ids=position_ids).last_hidden_state

    assert_rmse_close(custom_out.float(), hf_out.float(), rmse_ratio_tol=0.05)


@torch.no_grad()
def test_phi4_visionr_multimodal_forward_matches_reference():
    device = "cpu"
    dtype = torch.float32
    config = _create_small_config()
    model = Phi4VisionRForConditionalGeneration(config).to(device=device, dtype=dtype).eval()

    input_ids, vision_inputs = _make_small_multimodal_inputs()
    input_ids = input_ids.to(device)
    vision_inputs = {k: v.to(device) for k, v in vision_inputs.items()}

    actual = model(
        input_ids=input_ids,
        pixel_values=vision_inputs["pixel_values"],
        pixel_attention_mask=vision_inputs["pixel_attention_mask"],
        spatial_shapes=vision_inputs["spatial_shapes"],
    )

    hf_vision_config = deepcopy(config.vision_config)
    hf_vision_config._attn_implementation = "eager"
    hf_vision_model = Siglip2VisionModel(hf_vision_config).to(device=device, dtype=dtype).eval()
    hf_vision_model.load_state_dict(model.model.vision_tower.vision_tower.state_dict())

    ref_projector = build_vision_projector(config).to(device=device, dtype=dtype).eval()
    ref_projector.load_state_dict(model.model.mm_projector.state_dict())

    ref_lm_head = (
        torch.nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)
        .to(device=device, dtype=dtype)
        .eval()
    )
    ref_lm_head.load_state_dict(model.lm_head.state_dict())

    from transformers.models.phi3.modeling_phi3 import Phi3Model

    hf_text_model = Phi3Model(_create_hf_text_config()).to(device=device, dtype=dtype).eval()
    hf_text_model.load_state_dict(
        {
            key: value
            for key, value in model.model.state_dict().items()
            if key.startswith(("embed_tokens.", "layers.", "norm."))
        },
        strict=False,
    )

    vision_outputs = hf_vision_model(
        vision_inputs["pixel_values"],
        vision_inputs["pixel_attention_mask"],
        vision_inputs["spatial_shapes"],
        output_hidden_states=True,
    )
    image_features = vision_outputs.hidden_states[-2][0][
        vision_inputs["pixel_attention_mask"][0].bool()
    ]
    image_features = ref_projector(image_features)

    embed_tokens = hf_text_model.embed_tokens
    ref_inputs_embeds = torch.cat(
        [
            embed_tokens(input_ids[:, :1]).squeeze(0),
            image_features,
            embed_tokens(input_ids[:, 2:]).squeeze(0),
        ],
        dim=0,
    ).unsqueeze(0)
    ref_position_ids = torch.arange(
        ref_inputs_embeds.shape[1], dtype=torch.long, device=device
    ).unsqueeze(0)
    ref_hidden = hf_text_model(
        input_ids=None,
        inputs_embeds=ref_inputs_embeds,
        position_ids=ref_position_ids,
    ).last_hidden_state
    ref_logits = ref_lm_head(ref_hidden).float()

    assert_rmse_close(actual.logits.float(), ref_logits.float(), rmse_ratio_tol=0.05)


@torch.no_grad()
def test_phi4_visionr_text_model_can_be_exported():
    device = "cpu"
    dtype = torch.float32
    config = _create_small_config()
    model = Phi4VisionRForConditionalGeneration(config).model
    model.to(device=device, dtype=dtype)
    model.eval()

    batch_size, seq_len = 2, 8
    input_ids = torch.randint(
        0, config.text_config.vocab_size, (batch_size, seq_len), device=device
    )
    position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)

    dynamic_shapes = (
        {0: Dim.DYNAMIC, 1: Dim.DYNAMIC},
        {0: Dim.DYNAMIC, 1: Dim.DYNAMIC},
    )

    gm = torch_export_to_gm(
        model,
        args=tuple(),
        kwargs={"input_ids": input_ids, "position_ids": position_ids},
        dynamic_shapes=dynamic_shapes,
    )
    move_to_device(gm, device)

    eager_out = model(input_ids=input_ids, position_ids=position_ids).last_hidden_state
    exported_out = gm(input_ids=input_ids, position_ids=position_ids)["last_hidden_state"]

    torch.testing.assert_close(exported_out.float(), eager_out.float(), rtol=1e-4, atol=1e-4)

    input_ids_2 = torch.randint(0, config.text_config.vocab_size, (1, 4), device=device)
    position_ids_2 = torch.arange(4, device=device).unsqueeze(0)
    exported_out_2 = gm(input_ids=input_ids_2, position_ids=position_ids_2)["last_hidden_state"]

    assert exported_out_2.shape == (1, 4, config.text_config.hidden_size)
    assert torch.isfinite(exported_out_2).all()
