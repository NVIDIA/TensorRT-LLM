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
import re
from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models import modeling_dspark
from tensorrt_llm._torch.models.modeling_dflash import dspark_layer_window_size
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
    get_layer_attention_window,
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


_WINDOW_KEYS = (
    "layer_types",
    "sliding_window",
    "use_sliding_window",
    "use_swa",
    "swa_window_size",
)
_MIXED = ["sliding_attention", "full_attention"] * 2 + ["sliding_attention"]
_HF = {"layer_types": _MIXED, "sliding_window": 8}
_MIXED_WINDOWS = [8, None, 8, None, 8]
_UNBOUNDED = [None] * 5
_SOURCES = ("config", "text_config", "dflash_config", "text_config.dflash_config")


def _window_config(fields: dict[str, dict]) -> dict:
    """The published layout with every window field removed, then the given
    fields placed per source (``config``, ``text_config``, ``dflash_config``,
    ``text_config.dflash_config``)."""
    config = deepcopy(_CONFIG)
    sources = {
        "config": config,
        "text_config": config["text_config"],
        "dflash_config": config["dflash_config"],
        "text_config.dflash_config": config["text_config"]["dflash_config"],
    }
    for name, source in sources.items():
        for key in _WINDOW_KEYS:
            source.pop(key, None)
        source.update(deepcopy(fields.get(name, {})))
    if "dflash_config" in fields and "text_config.dflash_config" not in fields:
        config["text_config"].pop("dflash_config")
    return config


@pytest.mark.parametrize(
    "fields, context_windows, block_windows",
    [pytest.param({source: _HF}, _MIXED_WINDOWS, _MIXED_WINDOWS, id=source) for source in _SOURCES]
    + [
        pytest.param(
            {source: {"layer_types": schedule, "sliding_window": window} for source in _SOURCES},
            [window if kind == "sliding_attention" else None for kind in schedule],
            [window if kind == "sliding_attention" else None for kind in schedule],
            id=f"matching-{window}-{schedule[0]}",
        )
        for window in (1, 3, 4096, 8192)
        for schedule in (["sliding_attention"] * 5, ["full_attention"] * 5)
    ]
    + [
        pytest.param(
            {
                "config": {
                    **_HF,
                    **({"use_sliding_window": enabled} if location in ("top", "both") else {}),
                },
                "text_config": (
                    {"use_sliding_window": enabled} if location in ("text", "both") else {}
                ),
                "text_config.dflash_config": (
                    {"use_swa": True, "swa_window_size": 8} if native else {}
                ),
            },
            _MIXED_WINDOWS if enabled else _UNBOUNDED,
            _MIXED_WINDOWS if enabled else _UNBOUNDED,
            id=f"switch-{location}-{enabled}-{native}",
        )
        for location in ("top", "text", "both")
        for enabled in (False, True)
        for native in (False, True)
    ]
    + [
        pytest.param(
            {"config": _HF, "text_config": {key: None}},
            _MIXED_WINDOWS,
            _MIXED_WINDOWS,
            id=f"outer-survives-{key}",
        )
        for key in ("layer_types", "sliding_window")
    ]
    + [
        pytest.param(
            {
                "config": {**_HF, "use_sliding_window": False},
                "text_config": {"use_sliding_window": None},
            },
            _UNBOUNDED,
            _UNBOUNDED,
            id="outer-disable-survives-null",
        ),
        pytest.param(
            {
                "text_config": {"layer_types": None, "sliding_window": None},
                "text_config.dflash_config": _HF,
            },
            _MIXED_WINDOWS,
            _MIXED_WINDOWS,
            id="lift-through-null",
        ),
        pytest.param(
            {
                "text_config": _HF,
                "text_config.dflash_config": {
                    "layer_types": None,
                    "sliding_window": None,
                    "use_swa": None,
                    "swa_window_size": None,
                },
            },
            _MIXED_WINDOWS,
            _MIXED_WINDOWS,
            id="null-native-fields",
        ),
        pytest.param(
            {source: dict.fromkeys(_WINDOW_KEYS) for source in _SOURCES},
            _UNBOUNDED,
            _UNBOUNDED,
            id="all-window-fields-null",
        ),
        pytest.param(
            {"text_config": {"sliding_window": 8}},
            [8] * 5,
            _UNBOUNDED,
            id="hf-window-without-schedule",
        ),
        pytest.param(
            {"text_config.dflash_config": {"use_swa": True, "swa_window_size": 8}},
            _UNBOUNDED,
            [8] * 5,
            id="native-only",
        ),
        # Window fields declared only in the outer dflash_config survive the
        # text_config copy replacing that dict as a whole.
        pytest.param(
            {"dflash_config": _HF, "text_config.dflash_config": {}},
            _MIXED_WINDOWS,
            _MIXED_WINDOWS,
            id="outer-dflash-survives-text-dict",
        ),
        pytest.param(
            {"dflash_config": {**_HF, "use_swa": False}, "text_config.dflash_config": {}},
            _UNBOUNDED,
            _UNBOUNDED,
            id="outer-dflash-disable-survives-text-dict",
        ),
        # The operator switch written inside either dflash_config copy.
        pytest.param(
            {"text_config.dflash_config": {**_HF, "use_sliding_window": False}},
            _UNBOUNDED,
            _UNBOUNDED,
            id="nested-disable-text-dflash",
        ),
        pytest.param(
            {
                "dflash_config": {**_HF, "use_sliding_window": False},
                "text_config.dflash_config": {},
            },
            _UNBOUNDED,
            _UNBOUNDED,
            id="nested-disable-outer-dflash",
        ),
        # use_swa outside dflash_config is not a window declaration: nothing
        # reads it there, so it neither disables nor conflicts.
        pytest.param(
            {"config": {"use_swa": False}, "text_config": _HF},
            _MIXED_WINDOWS,
            _MIXED_WINDOWS,
            id="outer-use-swa-is-not-a-declaration",
        ),
        # An explicit use_swa without swa_window_size takes the HF window.
        pytest.param(
            {"text_config": _HF, "text_config.dflash_config": {"use_swa": True}},
            _MIXED_WINDOWS,
            _MIXED_WINDOWS,
            id="explicit-use-swa-takes-hf-window",
        ),
    ]
    + [
        pytest.param(
            {
                "text_config": {} if flag == "absent" else {"use_sliding_window": flag},
                "text_config.dflash_config": {**_HF, "use_swa": False},
            },
            _UNBOUNDED,
            _UNBOUNDED,
            id=f"native-disable-{flag}",
        )
        for flag in ("absent", None, False)
    ],
)
def test_flattened_config_accepts_window_layouts(
    fields: dict[str, dict],
    context_windows: list[int | None],
    block_windows: list[int | None],
) -> None:
    """Every layout the framework runs is accepted, and both window resolvers
    (context path and block decode) see the declared windows. The two
    helper-level results differ on purpose for the native-only and
    schedule-free HF cases: ``_resolve_block_attention`` applies the final
    precedence between them."""
    config = _window_config(fields)
    before = deepcopy(config)
    cfg = build_minimax_dspark_draft_config(config)
    dflash = cfg.dflash_config
    assert [
        get_layer_attention_window(cfg, i) for i in range(cfg.num_hidden_layers)
    ] == context_windows
    assert [
        dspark_layer_window_size(
            bool(dflash.get("use_swa", False)),
            int(dflash.get("swa_window_size", 0) or 0),
            getattr(cfg, "layer_types", None),
            i,
        )
        for i in range(cfg.num_hidden_layers)
    ] == [(-1, -1) if window is None else (window - 1, window - 1) for window in block_windows]
    assert config == before


@pytest.mark.parametrize(
    "fields, error",
    [
        pytest.param(
            {"text_config": {"layer_types": ["sliding_attention"] * count}},
            f"text_config.layer_types length={count} vs num_hidden_layers=5",
            id=f"schedule-length-{count}",
        )
        for count in (0, 1, 6)
    ]
    + [
        pytest.param(
            {"config": {"sliding_window": 8}, "text_config": {"sliding_window": 16}},
            "config.sliding_window=8 vs text_config.sliding_window=16",
            id="top-text-window",
        ),
        pytest.param(
            {
                "text_config": _HF,
                "dflash_config": {"sliding_window": 16},
                "text_config.dflash_config": {},
            },
            "text_config.sliding_window=8 vs dflash_config.sliding_window=16",
            id="shadowed-outer-dflash",
        ),
        pytest.param(
            {
                "dflash_config": {"sliding_window": 8},
                "text_config.dflash_config": {"sliding_window": 16},
            },
            "dflash_config.sliding_window=8 vs text_config.dflash_config.sliding_window=16",
            id="two-dflash-copies",
        ),
        pytest.param(
            {
                "config": {"layer_types": _MIXED},
                "text_config.dflash_config": {"layer_types": ["full_attention"] * 5},
            },
            "Conflicting MiniMax DSpark layer_types:",
            id="schedule-copies",
        ),
        pytest.param(
            {
                "config": {"use_sliding_window": False},
                "text_config": {"use_sliding_window": True},
            },
            "config.use_sliding_window=False vs text_config.use_sliding_window=True",
            id="switch-copies",
        ),
        pytest.param(
            {
                "dflash_config": {"use_swa": False},
                "text_config.dflash_config": {"use_swa": True},
            },
            "dflash_config.use_swa=False vs text_config.dflash_config.use_swa=True",
            id="native-switch-copies",
        ),
        pytest.param(
            {
                "dflash_config": {"swa_window_size": 8},
                "text_config.dflash_config": {"swa_window_size": 16},
            },
            "dflash_config.swa_window_size=8 vs text_config.dflash_config.swa_window_size=16",
            id="native-window-copies",
        ),
        pytest.param(
            {
                "text_config": _HF,
                "text_config.dflash_config": {"use_swa": True, "swa_window_size": 16},
            },
            "text_config.sliding_window=8 vs text_config.dflash_config.swa_window_size=16",
            id="hf-native-window",
        ),
        pytest.param(
            {"text_config.dflash_config": {"sliding_window": 8, "swa_window_size": 16}},
            "text_config.dflash_config.sliding_window=8 vs "
            "text_config.dflash_config.swa_window_size=16",
            id="hf-native-same-source",
        ),
    ]
    + [
        pytest.param(
            {
                source: {"use_sliding_window": True},
                "text_config.dflash_config": {"use_swa": False},
            },
            f"text_config.dflash_config.use_swa=False vs {source}.use_sliding_window=True",
            id=f"opposed-switches-{source}",
        )
        for source in ("config", "text_config")
    ],
)
def test_flattened_config_rejects_window_contradictions(
    fields: dict[str, dict], error: str
) -> None:
    """Only self-contradictory declarations are rejected, and the error names
    both sources."""
    config = _window_config(fields)
    before = deepcopy(config)
    with pytest.raises(ValueError, match=re.escape(error)):
        build_minimax_dspark_draft_config(config)
    assert config == before


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
    for source in (
        tiny["text_config"],
        tiny["dflash_config"],
        tiny["text_config"]["dflash_config"],
    ):
        source["layer_types"] = ["sliding_attention"]
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
