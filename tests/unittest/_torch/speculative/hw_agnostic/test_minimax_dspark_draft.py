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
"""MiniMax-M3 standalone DSpark drafter: config flattening and checkpoint
translation (CPU only; no drafter module is built here)."""

import json
import pathlib
from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models import modeling_dspark
from tensorrt_llm._torch.models.modeling_dspark import (
    _MINIMAX_DSPARK_GEMMA_NORM_SUFFIXES,
    GQADSparkForCausalLM,
    MiniMaxDSparkBackbone,
    MiniMaxDSparkForCausalLM,
    dequantize_mxfp8_weight,
    prepare_minimax_dspark_weights,
    remap_minimax_dspark_weight_name,
)
from tensorrt_llm._torch.models.modeling_utils import get_registered_model_class
from tensorrt_llm._torch.pyexecutor.config_utils import (
    MINIMAX_DSPARK_DRAFT_ARCH,
    build_minimax_dspark_draft_config,
    load_pretrained_config,
)
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.quantization import QuantAlgo

pytestmark = pytest.mark.cpu_only

# The shipped drafter config.json, reduced to the fields the loaders read. The
# backbone geometry sits under text_config; the dspark_* / dflash_config knobs
# are duplicated at both levels, as in the checkpoint.
_TEXT = {
    "architectures": ["DSparkMiniMaxDraftModel"],
    "dtype": "bfloat16",
    "hidden_size": 6144,
    "intermediate_size": 3072,
    "num_hidden_layers": 5,
    "num_attention_heads": 64,
    "num_key_value_heads": 4,
    "head_dim": 128,
    "vocab_size": 200064,
    "max_position_embeddings": 1048576,
    "rms_norm_eps": 1e-06,
    "use_gemma_norm": True,
    "rope_theta": 10000000,
    "rotary_dim": 64,
    "partial_rotary_factor": 0.5,
    "hidden_act": "swigluoai",
    "use_qk_norm": True,
    "dense_intermediate_size": 12288,
    "qk_norm_type": "per_head",
    "swiglu_alpha": 1.702,
    "swiglu_limit": 7.0,
    "sparse_attention_config": None,
    "dspark_block_size": 7,
    "dspark_markov_rank": 256,
    "dspark_markov_head_type": "vanilla",
    "dspark_noise_token_id": 200058,
    "dspark_use_dense_ffn": True,
    "dspark_ffn_hidden_size": 12288,
    "dspark_target_layer_ids": [3, 17, 31, 45, 59],
    "dspark_config": {
        "markov_rank": 256,
        "markov_head_type": "vanilla",
        "mask_token_id": 200058,
        "use_dense_ffn": True,
        "ffn_hidden_size": 12288,
    },
    "dflash_config": {
        "num_hidden_layers": 5,
        "num_target_layers": 60,
        "block_size": 7,
        "mask_token_id": 200058,
        "target_layer_ids": [3, 17, 31, 45, 59],
        "layer_types": ["sliding_attention"] * 5,
        "sliding_window": 4096,
    },
    "enable_confidence_head": True,
    "layer_types": ["sliding_attention"] * 5,
    "sliding_window": 4096,
    "model_type": "",
}

_CONFIG = {
    "transformers_version": "5.12.1",
    "architectures": ["DSparkMiniMaxDraftModel"],
    "dtype": "bfloat16",
    "model_type": "minimax_m3_vl",
    "text_config": _TEXT,
    "quantization_config": {
        "quant_method": "mxfp8",
        "weight_block_size": [1, 32],
        "ignored_layers": ["lm_head"],
    },
    "tie_word_embeddings": False,
    "dspark_block_size": 7,
    "dspark_target_layer_ids": [3, 17, 31, 45, 59],
    "dflash_config": dict(_TEXT["dflash_config"]),
    "hidden_size": 6144,
    "vocab_size": 200064,
    "num_hidden_layers": 5,
}


def test_flattened_config_exposes_backbone_and_drafter_fields() -> None:
    cfg = build_minimax_dspark_draft_config(_CONFIG)
    assert cfg.architectures == [MINIMAX_DSPARK_DRAFT_ARCH]
    assert cfg.model_type != "minimax_m3_vl"
    assert cfg.num_attention_heads == 64
    assert cfg.num_key_value_heads == 4
    assert cfg.head_dim == 128
    assert cfg.rotary_dim == 64 and cfg.partial_rotary_factor == 0.5
    assert cfg.dspark_target_layer_ids == [3, 17, 31, 45, 59]
    assert cfg.dspark_markov_rank == 256
    assert cfg.dflash_config["mask_token_id"] == 200058
    assert cfg.dflash_config["block_size"] == 7
    # The HF sliding window on 'sliding_attention' layers is translated to the
    # DFlash block-decode spelling.
    assert cfg.dflash_config["use_swa"] is True
    assert cfg.dflash_config["swa_window_size"] == 4096
    # Dequantized at load: the backbone is built unquantized.
    assert getattr(cfg, "quantization_config", None) is None
    assert not hasattr(cfg, "text_config")
    assert cfg.torch_dtype == torch.bfloat16


@pytest.mark.parametrize("checkpoint_use_swa", [None, True])
def test_flattened_config_honors_an_explicit_sliding_window_disable(
    checkpoint_use_swa: bool | None,
) -> None:
    """``use_sliding_window: false`` is the DFlash switch that disables the
    pooled-context window; the block-decode window must follow it, whether it
    would come from the HF ``sliding_attention`` spelling or from a
    ``dflash_config.use_swa`` the checkpoint already carries."""
    config = json.loads(json.dumps(_CONFIG))
    config["text_config"]["use_sliding_window"] = False
    if checkpoint_use_swa is not None:
        config["text_config"]["dflash_config"]["use_swa"] = checkpoint_use_swa
        config["text_config"]["dflash_config"]["swa_window_size"] = 4096
    cfg = build_minimax_dspark_draft_config(config)
    assert cfg.use_sliding_window is False
    assert cfg.dflash_config["use_swa"] is False
    assert "swa_window_size" not in cfg.dflash_config


def test_flattened_config_keeps_existing_dflash_window_spelling() -> None:
    config = json.loads(json.dumps(_CONFIG))
    config["text_config"]["dflash_config"]["use_swa"] = False
    cfg = build_minimax_dspark_draft_config(config)
    assert cfg.dflash_config["use_swa"] is False
    assert "swa_window_size" not in cfg.dflash_config


def test_load_pretrained_config_routes_by_architecture(tmp_path: pathlib.Path) -> None:
    (tmp_path / "config.json").write_text(json.dumps(_CONFIG))
    cfg = load_pretrained_config(str(tmp_path), trust_remote_code=False)
    assert cfg.architectures == [MINIMAX_DSPARK_DRAFT_ARCH]
    assert cfg.hidden_size == 6144 and cfg.num_hidden_layers == 5


def test_backbone_is_registered_under_the_checkpoint_architecture() -> None:
    assert get_registered_model_class(MINIMAX_DSPARK_DRAFT_ARCH) is MiniMaxDSparkBackbone
    assert issubclass(MiniMaxDSparkForCausalLM, GQADSparkForCausalLM)


@pytest.mark.parametrize(
    "name, expected",
    [
        (
            "language_model.model.dspark.layers.3.decoder_layer.self_attn.q_proj.weight",
            "layers.3.self_attn.q_proj.weight",
        ),
        (
            "language_model.model.dspark.layers.0.decoder_layer.mlp.down_proj.weight_scale_inv",
            "layers.0.mlp.down_proj.weight_scale_inv",
        ),
        ("language_model.model.dspark.fc.weight", "fc.weight"),
        ("language_model.model.dspark.hidden_norm.weight", "hidden_norm.weight"),
        ("language_model.model.dspark.final_norm.weight", "norm.weight"),
        (
            "language_model.model.dspark.markov_head.markov_w1.weight",
            "markov_head.markov_w1.weight",
        ),
        (
            "language_model.model.dspark.confidence_head.proj.bias",
            "confidence_head.proj.bias",
        ),
        ("lm_head.weight", "lm_head.weight"),
    ],
)
def test_weight_name_remap(name: str, expected: str) -> None:
    assert remap_minimax_dspark_weight_name(name) == expected


def test_mxfp8_dequantization_matches_per_block_reference() -> None:
    torch.manual_seed(0)
    out_features, in_features = 4, 96  # three 32-wide K blocks
    # Build finite E4M3 codes directly: dividing bf16-range samples by block scales as small
    # as 2**-13 would overflow E4M3 (max 448) into NaN and make the comparison meaningless.
    quantized = (torch.randn(out_features, in_features) * 3).to(torch.float8_e4m3fn)
    assert torch.isfinite(quantized.to(torch.float32)).all()
    # UE8M0 exponents around 127 (2**0).
    exponents = torch.randint(114, 131, (out_features, in_features // 32), dtype=torch.uint8)
    scale = torch.exp2(exponents.to(torch.float32) - 127).repeat_interleave(32, dim=1)
    expected = (quantized.to(torch.float32) * scale).to(torch.bfloat16)

    got = dequantize_mxfp8_weight(quantized, exponents, torch.bfloat16)
    assert got.dtype == torch.bfloat16
    torch.testing.assert_close(got, expected, rtol=0, atol=0)


def test_mxfp8_dequantization_reads_the_scale_exponent_bytes() -> None:
    """The scale is the UE8M0 code, whichever dtype safetensors hands back."""
    weight = torch.randn(4, 64).to(torch.float8_e4m3fn)
    scale = torch.tensor([[127, 128], [126, 130], [127, 127], [120, 135]], dtype=torch.uint8)
    expected = dequantize_mxfp8_weight(weight, scale, torch.bfloat16)
    e8m0 = getattr(torch, "float8_e8m0fnu", None)
    if e8m0 is not None:
        torch.testing.assert_close(
            dequantize_mxfp8_weight(weight, scale.view(e8m0), torch.bfloat16), expected
        )
    with pytest.raises(ValueError, match="UE8M0"):
        dequantize_mxfp8_weight(weight, scale.to(torch.float32), torch.bfloat16)


def test_minimax_drafter_refuses_a_quantized_draft_config() -> None:
    """A hf_quant_config.json next to the checkpoint would quantize the
    backbone the loader hands bf16 tensors to; the drafter refuses to build."""
    draft_config = SimpleNamespace(quant_config=QuantConfig(quant_algo=QuantAlgo.MXFP8))
    with pytest.raises(ValueError, match="hf_quant_config.json"):
        MiniMaxDSparkForCausalLM(draft_config)


def test_prepare_weights_dequantizes_and_folds_gemma_norms() -> None:
    prefix = "language_model.model.dspark."
    quantized = torch.ones(2, 64, dtype=torch.float8_e4m3fn)
    scale = torch.full((2, 2), 128, dtype=torch.uint8)  # 2**1 per block
    norm = torch.full((8,), 0.25, dtype=torch.bfloat16)
    weights = {
        prefix + "layers.0.decoder_layer.self_attn.k_proj.weight": quantized,
        prefix + "layers.0.decoder_layer.self_attn.k_proj.weight_scale_inv": scale,
        prefix + "layers.0.decoder_layer.self_attn.k_norm.weight": norm.clone(),
        prefix + "layers.0.decoder_layer.input_layernorm.weight": norm.clone(),
        prefix + "hidden_norm.weight": norm.clone(),
        prefix + "final_norm.weight": norm.clone(),
        prefix + "fc.weight": torch.zeros(8, 16, dtype=torch.bfloat16),
        prefix + "markov_head.markov_w1.weight": norm.clone(),
    }
    out = prepare_minimax_dspark_weights(weights, torch.bfloat16)

    assert set(out) == {
        "layers.0.self_attn.k_proj.weight",
        "layers.0.self_attn.k_norm.weight",
        "layers.0.input_layernorm.weight",
        "hidden_norm.weight",
        "norm.weight",
        "fc.weight",
        "markov_head.markov_w1.weight",
    }
    assert out["layers.0.self_attn.k_proj.weight"].dtype == torch.bfloat16
    torch.testing.assert_close(
        out["layers.0.self_attn.k_proj.weight"],
        torch.full((2, 64), 2.0, dtype=torch.bfloat16),
    )
    # Only the norms whose consumers read the raw weight are folded.
    for key in ("layers.0.self_attn.k_norm.weight", "hidden_norm.weight"):
        torch.testing.assert_close(out[key], torch.full((8,), 1.25, dtype=torch.bfloat16))
    # The backbone norms are Gemma modules (use_gemma=True) and keep the stored weight.
    for key in ("layers.0.input_layernorm.weight", "norm.weight"):
        torch.testing.assert_close(out[key], norm)
    # Head tensors are not norms and are passed through untouched.
    torch.testing.assert_close(out["markov_head.markov_w1.weight"], norm)


def test_backbone_norms_are_gemma_modules_and_only_raw_weight_norms_are_folded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The load-time ``1 + weight`` fold and the ``use_gemma`` flags must pair
    up: a norm is folded exactly when its module does not add the one itself.

    Attention, MLP and embedding construction allocate on CUDA (RoPE tables,
    backend buffers), so they are stubbed; the norms are built for real.
    """

    class _Stub(torch.nn.Module):
        def __init__(self, *args: object, **kwargs: object) -> None:
            super().__init__()

    monkeypatch.setattr(modeling_dspark, "MiniMaxDSparkAttention", _Stub)
    monkeypatch.setattr(modeling_dspark, "GatedMLP", _Stub)
    monkeypatch.setattr(modeling_dspark, "Embedding", _Stub)

    tiny = json.loads(json.dumps(_CONFIG))
    tiny["text_config"].update(hidden_size=64, num_hidden_layers=1)
    cfg = build_minimax_dspark_draft_config(tiny)
    cfg.torch_dtype = torch.bfloat16
    model = modeling_dspark.MiniMaxDSparkModel(ModelConfig(pretrained_config=cfg))
    layer = model.layers[0]

    for norm in (layer.input_layernorm, layer.post_attention_layernorm, model.norm):
        assert norm.use_gemma
        assert norm.weight.dtype == torch.bfloat16
    # q_norm / k_norm (plain RMSNorm in MiniMaxDSparkAttention) and the
    # wrapper-owned hidden_norm are the only folded weights.
    folded = {suffix.split(".")[0] for suffix in _MINIMAX_DSPARK_GEMMA_NORM_SUFFIXES}
    assert folded == {"q_norm", "k_norm", "hidden_norm"}
