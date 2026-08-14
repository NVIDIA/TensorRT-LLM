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
"""Qwen3.5/3.8 lm_head quantization decisions."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from tensorrt_llm._torch.models.modeling_qwen3_5 import (
    _lm_head_quant_enabled,
    _normalize_qwen35_exclude_modules,
    _normalize_qwen35_quant_config_dict,
)
from tensorrt_llm._torch.models.modeling_utils import DecoderModelForCausalLM
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.quantization.mode import QuantAlgo

NUM_HIDDEN_LAYERS = 64
# Qwen3.8-27B; divisible by every common tp_size, so COLUMN TP never pads.
VOCAB_SIZE = 248320

# A peer entry that must survive every lm_head decision unchanged.
PEER_KEY = "model.language_model.layers.0.self_attn.q_proj"
PEER_KEY_NORMALIZED = "model.layers.0.self_attn.q_proj"

# What llm-compressor's ``ignore`` list looks like for these checkpoints: never
# mentions lm_head (the checkpoint quantizes it), so the exclusion below is the
# one the normalizer adds.
CHECKPOINT_IGNORE = ["model.language_model.layers.0.linear_attn.in_proj_a"]

# The rowwise FP8 spelling used by compressed-tensors mixed-group checkpoints.
FP8_ALGOS = [QuantAlgo.FP8_PER_CHANNEL_PER_TOKEN]

BLACKWELL_SMS = [100, 103]
NON_BLACKWELL_SMS = [89, 90, 120]


def _model_config(
    lm_head_algo,
    *,
    tp_size: int = 1,
    vocab_size: int = VOCAB_SIZE,
    tie_word_embeddings: bool = False,
    enable_attention_dp: bool = False,
    enable_lm_head_tp_in_adp: bool = False,
) -> SimpleNamespace:
    """Build the minimal model_config the lm_head normalization pass reads."""
    quant_config_dict = {PEER_KEY: QuantConfig(quant_algo=QuantAlgo.FP8)}
    if lm_head_algo is not None:
        quant_config_dict["lm_head"] = QuantConfig(quant_algo=lm_head_algo)
    return SimpleNamespace(
        quant_config=QuantConfig(
            quant_algo=QuantAlgo.MIXED_PRECISION,
            kv_cache_quant_algo=QuantAlgo.FP8,
            exclude_modules=list(CHECKPOINT_IGNORE),
        ),
        quant_config_dict=quant_config_dict,
        pretrained_config=SimpleNamespace(
            num_hidden_layers=NUM_HIDDEN_LAYERS,
            vocab_size=vocab_size,
            tie_word_embeddings=tie_word_embeddings,
        ),
        mapping=SimpleNamespace(
            tp_size=tp_size,
            enable_attention_dp=enable_attention_dp,
            enable_lm_head_tp_in_adp=enable_lm_head_tp_in_adp,
        ),
    )


def _normalize(model_config) -> bool:
    """Run the pass both Qwen3.5 entry points run, and report the decision."""
    keep_lm_head_quant = _lm_head_quant_enabled(model_config)
    _normalize_qwen35_exclude_modules(model_config, keep_lm_head_quant=keep_lm_head_quant)
    _normalize_qwen35_quant_config_dict(model_config, keep_lm_head_quant=keep_lm_head_quant)
    return keep_lm_head_quant


def _assert_lm_head_quantized(model_config, expected_algo: QuantAlgo) -> None:
    """The normalized config must make the generic path build a quantized head."""
    assert model_config.quant_config_dict["lm_head"].quant_algo == expected_algo
    assert "lm_head" not in model_config.quant_config.exclude_modules
    resolved = DecoderModelForCausalLM._resolve_lm_head_quant_config(model_config)
    assert resolved is not None
    assert resolved.quant_algo == expected_algo


def _assert_lm_head_bf16(model_config) -> None:
    """The normalized config must leave LMHead unquantized."""
    assert "lm_head" not in model_config.quant_config_dict
    assert "lm_head" in model_config.quant_config.exclude_modules
    assert DecoderModelForCausalLM._resolve_lm_head_quant_config(model_config) is None


def _assert_peer_entry_intact(model_config) -> None:
    """lm_head handling must not disturb the rest of quant_config_dict."""
    peer = model_config.quant_config_dict[PEER_KEY_NORMALIZED]
    assert peer.quant_algo == QuantAlgo.FP8


# --- FP8 lm_head (compressed-tensors mixed-group checkpoints) ----------------
#
# The checkpoint stores lm_head.weight as e4m3 plus a per-channel
# lm_head.weight_scale and no packed-FP4 tensors, so nothing can dequantize it
# to bf16: the entry has to be kept, on any SM, with its algorithm untouched.


@pytest.mark.parametrize("sm_version", BLACKWELL_SMS + NON_BLACKWELL_SMS)
@pytest.mark.parametrize("algo", FP8_ALGOS)
def test_fp8_lm_head_is_retained_unchanged(algo, sm_version):
    model_config = _model_config(algo)

    with patch(
        "tensorrt_llm._torch.models.modeling_qwen3_5.get_sm_version", return_value=sm_version
    ):
        assert _normalize(model_config)

    # Kept on the checkpoint's own algorithm -- no NVFP4 promotion.
    _assert_lm_head_quantized(model_config, algo)
    _assert_peer_entry_intact(model_config)


@pytest.mark.parametrize("tp_size", [1, 2, 4, 8])
@pytest.mark.parametrize("algo", FP8_ALGOS)
def test_fp8_lm_head_retained_across_tp_sizes(algo, tp_size):
    # vocab_size divides every tp_size here, so COLUMN TP never pads the vocab
    # shard -- the case the quantized LMHead rejects at construction.
    model_config = _model_config(algo, tp_size=tp_size)

    with patch("tensorrt_llm._torch.models.modeling_qwen3_5.get_sm_version", return_value=100):
        assert _normalize(model_config)

    _assert_lm_head_quantized(model_config, algo)


# --- Unsupported FP8 configurations fail instead of dropping the scale ------


@pytest.mark.parametrize(
    "case,kwargs",
    [
        ("tied embeddings share the dense weight", {"tie_word_embeddings": True}),
        (
            "lm_head TP in attention-DP slices the raw weight",
            {"enable_attention_dp": True, "enable_lm_head_tp_in_adp": True},
        ),
        ("vocab does not divide tp_size (COLUMN TP pads)", {"vocab_size": 250, "tp_size": 4}),
    ],
)
@pytest.mark.parametrize("algo", FP8_ALGOS)
def test_unsupported_fp8_configuration_raises(algo, case, kwargs):
    model_config = _model_config(algo, **kwargs)

    with patch("tensorrt_llm._torch.models.modeling_qwen3_5.get_sm_version", return_value=100):
        with pytest.raises(ValueError, match="FP8 lm_head cannot fall back to bf16"):
            _normalize(model_config)


def test_fp8_lm_head_requires_vocab_size():
    model_config = _model_config(QuantAlgo.FP8_PER_CHANNEL_PER_TOKEN)
    del model_config.pretrained_config.vocab_size

    with patch("tensorrt_llm._torch.models.modeling_qwen3_5.get_sm_version", return_value=100):
        with pytest.raises(ValueError, match="vocab_size is required"):
            _normalize(model_config)


@pytest.mark.parametrize("algo", FP8_ALGOS + [QuantAlgo.W4A16_NVFP4])
def test_attention_dp_without_lm_head_tp_keeps_quantized_head(algo):
    model_config = _model_config(algo, enable_attention_dp=True)

    with patch("tensorrt_llm._torch.models.modeling_qwen3_5.get_sm_version", return_value=100):
        assert _normalize(model_config)

    expected_algo = QuantAlgo.NVFP4 if algo == QuantAlgo.W4A16_NVFP4 else algo
    _assert_lm_head_quantized(model_config, expected_algo)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"tie_word_embeddings": True},
        {"enable_attention_dp": True, "enable_lm_head_tp_in_adp": True},
        {"vocab_size": 250, "tp_size": 4},
    ],
)
def test_unsupported_w4a16_configuration_falls_back_to_bf16(kwargs):
    model_config = _model_config(QuantAlgo.W4A16_NVFP4, **kwargs)

    with patch("tensorrt_llm._torch.models.modeling_qwen3_5.get_sm_version", return_value=100):
        assert not _normalize(model_config)

    _assert_lm_head_bf16(model_config)
    _assert_peer_entry_intact(model_config)


@pytest.mark.parametrize("sm_version", BLACKWELL_SMS + NON_BLACKWELL_SMS)
@pytest.mark.parametrize("algo", [QuantAlgo.FP8, QuantAlgo.FP8_BLOCK_SCALES])
def test_unsupported_fp8_lm_head_algo_raises(algo, sm_version):
    model_config = _model_config(algo)

    with patch(
        "tensorrt_llm._torch.models.modeling_qwen3_5.get_sm_version", return_value=sm_version
    ):
        with pytest.raises(ValueError, match="FP8 lm_head algorithm"):
            _normalize(model_config)


@pytest.mark.parametrize("sm_version", BLACKWELL_SMS + NON_BLACKWELL_SMS)
@pytest.mark.parametrize("algo", [QuantAlgo.NVFP4, QuantAlgo.W4A16_AWQ, QuantAlgo.INT8])
def test_other_lm_head_algorithms_keep_existing_fallback(algo, sm_version):
    model_config = _model_config(algo)

    with patch(
        "tensorrt_llm._torch.models.modeling_qwen3_5.get_sm_version", return_value=sm_version
    ):
        assert not _normalize(model_config)

    _assert_lm_head_bf16(model_config)


@pytest.mark.parametrize("sm_version", BLACKWELL_SMS + NON_BLACKWELL_SMS)
def test_bf16_lm_head_checkpoint_is_untouched(sm_version):
    # No lm_head entry at all (checkpoint left it bf16): lm_head stays excluded
    # from quantization exactly as before.
    model_config = _model_config(None)

    with patch(
        "tensorrt_llm._torch.models.modeling_qwen3_5.get_sm_version", return_value=sm_version
    ):
        assert not _normalize(model_config)

    _assert_lm_head_bf16(model_config)
    _assert_peer_entry_intact(model_config)


# --- W4A16_NVFP4 lm_head (ModelOpt MIXED_PRECISION) must not regress --------


@pytest.mark.parametrize("sm_version", BLACKWELL_SMS)
def test_w4a16_nvfp4_lm_head_promoted_on_blackwell(sm_version):
    model_config = _model_config(QuantAlgo.W4A16_NVFP4)

    with patch(
        "tensorrt_llm._torch.models.modeling_qwen3_5.get_sm_version", return_value=sm_version
    ):
        assert _normalize(model_config)

    # Promoted to the NVFP4 (W4A4) Linear path, as before.
    _assert_lm_head_quantized(model_config, QuantAlgo.NVFP4)
    _assert_peer_entry_intact(model_config)


@pytest.mark.parametrize("sm_version", NON_BLACKWELL_SMS)
def test_w4a16_nvfp4_lm_head_falls_back_off_blackwell(sm_version):
    model_config = _model_config(QuantAlgo.W4A16_NVFP4)

    with patch(
        "tensorrt_llm._torch.models.modeling_qwen3_5.get_sm_version", return_value=sm_version
    ):
        assert not _normalize(model_config)

    # No W4A4 path here: the entry is dropped and the weight mapper
    # dequantizes the packed FP4 weight to bf16.
    _assert_lm_head_bf16(model_config)
