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

import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from torch import nn

from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.modeling_nemotron_h import (
    NemotronHMOE,
    NemotronHMTP,
    _remap_hf_quant_module_name,
    _with_replacement_mtp_quant_config,
)
from tensorrt_llm._torch.models.modeling_utils import DecoderModelForCausalLM
from tensorrt_llm._torch.utils import AuxStreamType
from tensorrt_llm.models.modeling_utils import QuantAlgo, QuantConfig


def _make_nemotron_h_moe_config(
    quant_config: QuantConfig, moe_backend: str = "CUTLASS"
) -> ModelConfig:
    return ModelConfig(
        pretrained_config=SimpleNamespace(
            hidden_size=16,
            intermediate_size=32,
            mlp_bias=False,
            moe_intermediate_size=64,
            moe_latent_size=None,
            n_group=1,
            n_routed_experts=4,
            n_shared_experts=0,
            num_experts_per_tok=1,
            routed_scaling_factor=1.0,
            topk_group=1,
            torch_dtype=torch.float16,
        ),
        moe_backend=moe_backend,
        quant_config=quant_config,
    )


def test_nemotron_h_moe_passes_w4a16_config_through_unchanged():
    """Every MoE backend resolves W4A16_NVFP4 itself, so the layer must not
    rewrite quant_algo on its way to create_moe."""
    quant_config = QuantConfig(
        quant_algo=QuantAlgo.W4A16_NVFP4, group_size=16, exclude_modules=["lm_head"]
    )
    model_config = _make_nemotron_h_moe_config(quant_config)
    captured = {}

    def fake_create_moe(**kwargs):
        captured.update(kwargs)
        return nn.Identity()

    with patch(
        "tensorrt_llm._torch.models.modeling_nemotron_h.create_moe", side_effect=fake_create_moe
    ):
        with patch("torch.cuda.Event", side_effect=lambda: object()):
            aux_stream_dict = {AuxStreamType.MoeShared: None}
            NemotronHMOE(model_config=model_config, layer_idx=1, aux_stream_dict=aux_stream_dict)

    effective = captured["override_quant_config"] or captured["model_config"].quant_config
    assert effective.quant_algo == QuantAlgo.W4A16_NVFP4
    assert effective.group_size == 16
    assert model_config.quant_config.quant_algo == QuantAlgo.W4A16_NVFP4


def test_nemotron_h_moe_uses_mixer_expert_layer_quant_config():
    global_quant_config = QuantConfig()
    layer_quant_config = QuantConfig(quant_algo=QuantAlgo.W4A16_NVFP4, group_size=16)
    model_config = _make_nemotron_h_moe_config(global_quant_config)
    model_config.quant_config_dict = {
        "model.layers.1.mixer.experts.0.up_proj": layer_quant_config,
    }
    captured = {}

    def fake_create_moe(**kwargs):
        captured.update(kwargs)
        return nn.Identity()

    with patch(
        "tensorrt_llm._torch.models.modeling_nemotron_h.create_moe",
        side_effect=fake_create_moe,
    ):
        with patch("torch.cuda.Event", side_effect=lambda: object()):
            NemotronHMOE(
                model_config=model_config,
                layer_idx=1,
                aux_stream_dict={AuxStreamType.MoeShared: None},
            )

    assert captured["override_quant_config"] is layer_quant_config


def test_nemotron_h_moe_exclusion_outranks_the_layer_quant_config():
    """An excluded experts module has to reach create_moe unquantized.

    The per-layer entry is applied first, so an exclusion that does not
    override it leaves the layer on the per-layer format and loads quantized
    weights the checkpoint left in bf16 -- wrong numerics rather than a
    failure. ``kv_cache_quant_algo`` is not part of what an expert exclusion
    turns off, so it has to survive.
    """
    global_quant_config = QuantConfig(
        quant_algo=QuantAlgo.W4A16_NVFP4,
        group_size=16,
        kv_cache_quant_algo=QuantAlgo.FP8,
        exclude_modules=["model.layers.1.mixer.experts"],
    )
    model_config = _make_nemotron_h_moe_config(global_quant_config)
    model_config.quant_config_dict = {
        "model.layers.1.mixer.experts.0.up_proj": QuantConfig(
            quant_algo=QuantAlgo.W4A16_NVFP4, group_size=16
        ),
    }
    captured = {}

    def fake_create_moe(**kwargs):
        captured.update(kwargs)
        return nn.Identity()

    with patch(
        "tensorrt_llm._torch.models.modeling_nemotron_h.create_moe",
        side_effect=fake_create_moe,
    ):
        with patch("torch.cuda.Event", side_effect=lambda: object()):
            NemotronHMOE(
                model_config=model_config,
                layer_idx=1,
                aux_stream_dict={AuxStreamType.MoeShared: None},
            )

    override = captured["override_quant_config"]
    assert override.quant_algo is None
    assert override.kv_cache_quant_algo == QuantAlgo.FP8


@pytest.mark.parametrize("suffix", ["", ".0.up_proj"])
def test_nemotron_h_moe_uses_module_prefix_for_mtp_sublayer_quant_config(suffix):
    """MTP sublayers live at model.layers.{N}.layers.{S}; the experts lookup
    has to follow that path rather than the decoder-layer default."""
    global_quant_config = QuantConfig(quant_algo=QuantAlgo.MIXED_PRECISION)
    layer_quant_config = QuantConfig(quant_algo=QuantAlgo.NVFP4, group_size=16)
    model_config = _make_nemotron_h_moe_config(global_quant_config)
    model_config.quant_config_dict = {
        f"model.layers.52.layers.1.mixer.experts{suffix}": layer_quant_config,
    }
    captured = {}

    def fake_create_moe(**kwargs):
        captured.update(kwargs)
        return nn.Identity()

    with patch(
        "tensorrt_llm._torch.models.modeling_nemotron_h.create_moe",
        side_effect=fake_create_moe,
    ):
        with patch("torch.cuda.Event", side_effect=lambda: object()):
            NemotronHMOE(
                model_config=model_config,
                layer_idx=52,
                aux_stream_dict={AuxStreamType.MoeShared: None},
                module_prefix="model.layers.52.layers.1",
            )

    assert captured["override_quant_config"] is layer_quant_config


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("backbone.layers.3.mixer.in_proj", "model.layers.3.mixer.in_proj"),
        ("backbone.layers.16*", "model.layers.16*"),
        ("mtp.layers.0.mixer.q_proj", "model.layers.52.layers.0.mixer.q_proj"),
        (
            "mtp.layers.1.mixer.experts.0.up_proj",
            "model.layers.52.layers.1.mixer.experts.0.up_proj",
        ),
        ("mtp.layers.1", "model.layers.52.layers.1"),
        ("mtp*", "model.layers.52"),
        ("mtp.*", "model.layers.52"),
        ("mtp", "model.layers.52"),
        ("lm_head", "lm_head"),
        ("mtp_head.weight", "mtp_head.weight"),
    ],
)
def test_remap_hf_quant_module_name(name, expected):
    assert _remap_hf_quant_module_name(name, num_hidden_layers=52) == expected


def test_remap_hf_quant_module_name_whole_head_exclusion_covers_sublayers():
    quant_config = QuantConfig(
        quant_algo=QuantAlgo.FP8,
        exclude_modules=[_remap_hf_quant_module_name("mtp*", num_hidden_layers=52)],
    )
    assert quant_config.is_module_excluded_from_quantization("model.layers.52.layers.0")
    assert quant_config.is_module_excluded_from_quantization(
        "model.layers.52.layers.1.mixer.experts"
    )
    assert not quant_config.is_module_excluded_from_quantization("model.layers.5.mixer")


def _build_mtp_capturing_sublayers(quant_config: QuantConfig) -> tuple[ModelConfig, list]:
    model_config = ModelConfig(
        pretrained_config=SimpleNamespace(
            mtp_hybrid_override_pattern="*E",
            torch_dtype=torch.bfloat16,
        ),
        moe_backend="CUTEDSL",
        quant_config=quant_config,
    )
    captured = []

    def fake_decoder_layer(**kwargs):
        captured.append(kwargs)
        return nn.Identity()

    with patch(
        "tensorrt_llm._torch.models.modeling_nemotron_h.NemotronHMTPDecoderLayer",
        side_effect=fake_decoder_layer,
    ):
        with patch(
            "tensorrt_llm._torch.models.modeling_nemotron_h.DeepseekV3MTPHead",
            side_effect=lambda model_config: nn.Identity(),
        ):
            with patch(
                "tensorrt_llm._torch.models.modeling_nemotron_h.get_sm_version",
                return_value=121,
            ):
                NemotronHMTP(
                    model_config=model_config,
                    layer_idx=52,
                    aux_stream_dict={},
                )
    assert len(captured) == 2
    return model_config, captured


def test_nemotron_h_mtp_sublayers_get_module_prefix_and_inherit_moe_backend():
    quant_config = QuantConfig(quant_algo=QuantAlgo.MIXED_PRECISION)
    model_config, captured = _build_mtp_capturing_sublayers(quant_config)

    for sublayer_idx, layer_kwargs in enumerate(captured):
        assert layer_kwargs["module_prefix"] == f"model.layers.52.layers.{sublayer_idx}"
        assert layer_kwargs["model_config"].moe_backend == model_config.moe_backend
    assert model_config.quant_config is quant_config


def test_nemotron_h_mtp_excluded_head_stays_unquantized():
    """modelopt writes ``mtp*`` for a head it left in bf16; after the rewrite
    that is the head module itself, which excludes every sublayer."""
    quant_config = QuantConfig(
        quant_algo=QuantAlgo.NVFP4,
        group_size=16,
        kv_cache_quant_algo=QuantAlgo.FP8,
        exclude_modules=["lm_head", "model.layers.52"],
    )
    _, captured = _build_mtp_capturing_sublayers(quant_config)

    for layer_kwargs in captured:
        sublayer_quant_config = layer_kwargs["model_config"].quant_config
        assert sublayer_quant_config.quant_algo is None
        assert sublayer_quant_config.kv_cache_quant_algo == QuantAlgo.FP8


def test_nemotron_h_mtp_quantized_head_inherits_checkpoint_quant_config():
    """A single-algo checkpoint that quantizes the MTP head must build the
    sublayers quantized, or the packed weights cannot be loaded."""
    quant_config = QuantConfig(
        quant_algo=QuantAlgo.NVFP4, group_size=16, exclude_modules=["lm_head"]
    )
    _, captured = _build_mtp_capturing_sublayers(quant_config)

    for layer_kwargs in captured:
        assert layer_kwargs["model_config"].quant_config is quant_config


@pytest.mark.cpu_only
@pytest.mark.parametrize("head_algo", ["FP8", "W4A16_NVFP4"])
def test_nemotron_replacement_quantization_isolates_mtp(tmp_path, head_algo):
    metadata = {
        "quant_algo": "MIXED_PRECISION",
        "quantized_layers": {
            "lm_head": {"quant_algo": head_algo, "group_size": 16},
            "mtp.layers.1.mixer.experts": {"quant_algo": "NVFP4", "group_size": 16},
        },
    }
    path = tmp_path / "hf_quant_config.json"
    path.write_text(json.dumps({"quantization": metadata}))
    config = _make_nemotron_h_moe_config(QuantConfig(exclude_modules=["lm_head"]))
    config.pretrained_config.num_hidden_layers = 2
    config.pretrained_config.num_nextn_predict_layers = 1
    config.spec_config = SimpleNamespace(uses_replacement_heads=True, speculative_model=tmp_path)
    stale = QuantConfig(quant_algo=QuantAlgo.FP8)
    original = {
        "model.layers.0.mixer.experts": stale,
        "model.layers.2.layers.0.mixer.q_proj": stale,
        "mtp.layers.0.mixer.q_proj": stale,
        "draft_model.lm_head": stale,
    }
    config.quant_config_dict = original
    config._frozen = True

    result = _with_replacement_mtp_quant_config(config)

    assert config.quant_config_dict is original and len(original) == 4
    assert result.extra_attrs is config.extra_attrs and result._frozen
    assert result.quant_config is config.quant_config
    assert set(result.quant_config_dict) == {
        "model.layers.0.mixer.experts",
        "model.layers.2.layers.1.mixer.experts",
        "draft_model.lm_head",
    }
    assert result.quant_config_dict["model.layers.0.mixer.experts"] is stale
    assert (
        result.quant_config_dict["model.layers.2.layers.1.mixer.experts"].quant_algo
        == QuantAlgo.NVFP4
    )
    assert result.quant_config_dict["draft_model.lm_head"].quant_algo == QuantAlgo(head_algo)

    config.quant_config.exclude_modules = ["model.layers.2.layers.1.mixer.experts"]
    with pytest.raises(ValueError, match="exclusion conflicts"):
        _with_replacement_mtp_quant_config(config)
    config.quant_config.exclude_modules = ["model.layers.2.layers.0.mixer.qkv_proj"]
    metadata["quantized_layers"]["mtp.layers.0.mixer.q_proj"] = {"quant_algo": "FP8"}
    path.write_text(json.dumps({"quantization": metadata}))
    with pytest.raises(ValueError, match="exclusion conflicts"):
        _with_replacement_mtp_quant_config(config)
    config.spec_config.uses_replacement_heads = False
    assert _with_replacement_mtp_quant_config(config) is config


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "target_kv_algo,replacement_kv_algo",
    [(QuantAlgo.FP8, None), (None, QuantAlgo.FP8), (QuantAlgo.FP8, QuantAlgo.FP8)],
)
def test_nemotron_replacement_quantization_inherits_target_kv_dtype(
    tmp_path, target_kv_algo, replacement_kv_algo
):
    metadata = {
        "quant_algo": "MIXED_PRECISION",
        "kv_cache_quant_algo": replacement_kv_algo,
        "quantized_layers": {
            "mtp.layers.0.mixer.q_proj": {"quant_algo": "FP8"},
            "lm_head": {"quant_algo": "W4A16_NVFP4", "group_size": 16},
        },
    }
    (tmp_path / "hf_quant_config.json").write_text(json.dumps({"quantization": metadata}))
    target_quant_config = QuantConfig(kv_cache_quant_algo=target_kv_algo)
    config = _make_nemotron_h_moe_config(target_quant_config)
    config.pretrained_config.num_hidden_layers = 2
    config.pretrained_config.num_nextn_predict_layers = 1
    config.spec_config = SimpleNamespace(uses_replacement_heads=True, speculative_model=tmp_path)
    target_entry = QuantConfig(quant_algo=QuantAlgo.FP8, kv_cache_quant_algo=target_kv_algo)
    config.quant_config_dict = {"model.layers.0.mixer.q_proj": target_entry}

    result = _with_replacement_mtp_quant_config(config)

    attention_config = result.quant_config_dict["model.layers.2.layers.0.mixer.q_proj"]
    head_config = result.quant_config_dict["draft_model.lm_head"]
    assert attention_config.kv_cache_quant_algo == target_kv_algo
    assert head_config.kv_cache_quant_algo == target_kv_algo
    assert attention_config.quant_algo == QuantAlgo.FP8
    assert head_config.quant_algo == QuantAlgo.W4A16_NVFP4
    assert result.quant_config is target_quant_config
    assert result.quant_config.kv_cache_quant_algo == target_kv_algo
    assert result.quant_config_dict["model.layers.0.mixer.q_proj"] is target_entry
    assert config.quant_config_dict == {"model.layers.0.mixer.q_proj": target_entry}


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "quant_algo,head_count,head_algo,exclusions,error",
    [
        pytest.param("FP8", 1, "FP8", [], "MIXED_PRECISION metadata", id="homogeneous"),
        pytest.param("MIXED_PRECISION", 2, "FP8", [], "supports one head", id="multiple-heads"),
        pytest.param(
            "MIXED_PRECISION", 1, "NVFP4", [], "NVFP4 LM heads are not supported", id="nvfp4-head"
        ),
        pytest.param(
            "MIXED_PRECISION",
            1,
            "FP8",
            ["mtp.layers.*.mixer.experts"],
            "Replacement quantization exclusion conflicts",
            id="replacement-exclusion",
        ),
    ],
)
def test_nemotron_replacement_quantization_rejects_unsupported_metadata(
    tmp_path, quant_algo, head_count, head_algo, exclusions, error
):
    metadata = {"quant_algo": quant_algo, "exclude_modules": exclusions}
    if quant_algo == "MIXED_PRECISION":
        metadata["quantized_layers"] = {
            "lm_head": {"quant_algo": head_algo, "group_size": 16},
            "mtp.layers.1.mixer.experts": {"quant_algo": "NVFP4", "group_size": 16},
        }
    (tmp_path / "hf_quant_config.json").write_text(json.dumps({"quantization": metadata}))
    config = _make_nemotron_h_moe_config(QuantConfig())
    config.pretrained_config.num_hidden_layers = 2
    config.pretrained_config.num_nextn_predict_layers = head_count
    config.spec_config = SimpleNamespace(uses_replacement_heads=True, speculative_model=tmp_path)

    with pytest.raises(ValueError, match=error):
        _with_replacement_mtp_quant_config(config)


@pytest.mark.cpu_only
@pytest.mark.parametrize("has_head_scale", [False, True])
def test_nemotron_replacement_preserves_homogeneous_target_head(
    tmp_path, monkeypatch, has_head_scale
):
    (tmp_path / "hf_quant_config.json").write_text(
        json.dumps(
            {
                "quantization": {
                    "quant_algo": "MIXED_PRECISION",
                    "quantized_layers": {"lm_head": {"quant_algo": "FP8"}},
                }
            }
        )
    )
    config = _make_nemotron_h_moe_config(QuantConfig(quant_algo=QuantAlgo.NVFP4, group_size=16))
    config.pretrained_config.num_hidden_layers = 2
    config.pretrained_config.num_nextn_predict_layers = 1
    config.spec_config = SimpleNamespace(uses_replacement_heads=True, speculative_model=tmp_path)
    monkeypatch.setattr(
        DecoderModelForCausalLM,
        "_checkpoint_has_lm_head_scale",
        staticmethod(lambda config: has_head_scale),
    )
    expected = DecoderModelForCausalLM._resolve_lm_head_quant_config(config)

    result = _with_replacement_mtp_quant_config(config)

    assert DecoderModelForCausalLM._resolve_lm_head_quant_config(result) is expected
    assert config.quant_config_dict is None
    assert result.quant_config_dict["draft_model.lm_head"].quant_algo == QuantAlgo.FP8


@pytest.mark.cpu_only
@pytest.mark.parametrize("target_algo", [QuantAlgo.NVFP4, QuantAlgo.FP8])
def test_nemotron_replacement_omitted_sublayers_stay_unquantized(target_algo: QuantAlgo) -> None:
    target_quant = QuantConfig(quant_algo=target_algo, kv_cache_quant_algo=QuantAlgo.FP8)
    config = _make_nemotron_h_moe_config(target_quant)
    config.spec_config = SimpleNamespace(uses_replacement_heads=True)
    mtp = object.__new__(NemotronHMTP)

    resolved = mtp._get_mtp_sublayer_quant_config(config, "model.layers.52.layers.0")

    assert resolved.quant_algo is None
    assert resolved.kv_cache_quant_algo == QuantAlgo.FP8
    assert config.quant_config is target_quant
    assert target_quant.quant_algo == target_algo
