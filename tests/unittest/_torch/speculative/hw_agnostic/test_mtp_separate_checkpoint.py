# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from types import SimpleNamespace

import torch

from tensorrt_llm._torch.models.checkpoints.hf.nemotron_h_weight_mapper import \
    NemotronHHfWeightMapper
from tensorrt_llm._torch.speculative.utils import (
    filter_mtp_checkpoint_weights, loads_mtp_from_speculative_model,
    select_mtp_checkpoint_weights, update_spec_config_from_model_config)
from tensorrt_llm.llmapi.llm_args import Eagle3DecodingConfig, MTPDecodingConfig


def test_needs_separate_draft_weights_for_mtp_with_speculative_model():
    cfg = MTPDecodingConfig(max_draft_len=3,
                            speculative_model="/path/to/mtp")
    assert cfg.needs_separate_draft_weights is True

    cfg_no_draft = MTPDecodingConfig(max_draft_len=3)
    assert cfg_no_draft.needs_separate_draft_weights is False


def test_needs_separate_draft_weights_still_true_for_eagle3():
    cfg = Eagle3DecodingConfig(max_draft_len=3,
                               speculative_model="/path/to/eagle3")
    assert cfg.needs_separate_draft_weights is True


def test_loads_mtp_from_speculative_model_helper():
    assert loads_mtp_from_speculative_model(
        MTPDecodingConfig(max_draft_len=3,
                          speculative_model="/path/to/mtp")) is True
    assert loads_mtp_from_speculative_model(
        MTPDecodingConfig(max_draft_len=3)) is False
    assert loads_mtp_from_speculative_model(None) is False


def test_filter_and_select_mtp_checkpoint_weights():
    weights = {
        "backbone.layers.0.mixer.weight": torch.ones(2),
        "mtp.layers.0.enorm.weight": torch.ones(4),
        "mtp.layers.1.norm.weight": torch.ones(4),
        "lm_head.weight": torch.ones(3),
    }
    filtered = filter_mtp_checkpoint_weights(weights)
    assert "backbone.layers.0.mixer.weight" in filtered
    assert "lm_head.weight" in filtered
    assert "mtp.layers.0.enorm.weight" not in filtered
    assert "mtp.layers.1.norm.weight" not in filtered

    selected = select_mtp_checkpoint_weights(weights)
    assert set(selected) == {
        "mtp.layers.0.enorm.weight",
        "mtp.layers.1.norm.weight",
    }


def test_update_spec_config_prefers_speculative_model_mtp_fields(tmp_path):
    mtp_dir = tmp_path / "mtp_heads"
    mtp_dir.mkdir()
    (mtp_dir / "config.json").write_text(
        json.dumps({
            "num_nextn_predict_layers": 1,
            "mtp_hybrid_override_pattern": "*E",
            "mtp_block_configs": [{
                "block_type": "moe",
                "num_experts": 8
            }],
        }))

    # Target has no MTP (or stale MTP metadata).
    model_config = SimpleNamespace(
        architectures=["NemotronHForCausalLM"],
        num_nextn_predict_layers=0,
        mtp_layers_block_type=None,
        mtp_block_configs=None,
    )
    spec_config = MTPDecodingConfig(max_draft_len=5,
                                    speculative_model=str(mtp_dir))

    update_spec_config_from_model_config(spec_config, model_config)

    assert spec_config.num_nextn_predict_layers == 1
    assert model_config.num_nextn_predict_layers == 1
    # Legacy pattern string is converted to the writable HF field.
    assert model_config.mtp_layers_block_type == ["attention", "moe"]
    assert model_config.mtp_block_configs == [{
        "block_type": "moe",
        "num_experts": 8
    }]
    # User-set max_draft_len is preserved for Eagle-style replay.
    assert spec_config.max_draft_len == 5


def test_update_spec_config_uses_mtp_layers_block_type_when_present(tmp_path):
    mtp_dir = tmp_path / "mtp_heads"
    mtp_dir.mkdir()
    (mtp_dir / "config.json").write_text(
        json.dumps({
            "num_nextn_predict_layers": 1,
            "mtp_layers_block_type": ["attention", "moe"],
        }))

    model_config = SimpleNamespace(
        architectures=["NemotronHForCausalLM"],
        num_nextn_predict_layers=0,
        mtp_layers_block_type=None,
    )
    spec_config = MTPDecodingConfig(max_draft_len=3,
                                    speculative_model=str(mtp_dir))
    update_spec_config_from_model_config(spec_config, model_config)
    assert model_config.mtp_layers_block_type == ["attention", "moe"]


def test_remap_preprocessed_mtp_weights_for_draft_model():
    from tensorrt_llm._torch.speculative.utils import \
        remap_preprocessed_mtp_weights_for_draft_model

    weights = {
        "model.layers.52.layers.0.enorm.weight": torch.ones(4),
        "model.layers.52.layers.1.norm.weight": torch.ones(4),
    }
    remapped = remap_preprocessed_mtp_weights_for_draft_model(
        weights, num_hidden_layers=52, num_mtp_layers=1)
    assert remapped == {
        "mtp_layers.0.layers.0.enorm.weight": weights[
            "model.layers.52.layers.0.enorm.weight"],
        "mtp_layers.0.layers.1.norm.weight": weights[
            "model.layers.52.layers.1.norm.weight"],
    }


def _make_one_engine_stub(spec_config):
    """A bare SpecDecOneEngineForCausalLM with just the module tree we need.

    ``__init__`` builds a whole target model, so construct the instance
    directly and register only the two aliases of a single MTP head: the
    target's ``model.layers[N]`` and ``draft_model.mtp_layers[0]``.
    """
    from tensorrt_llm._torch.models.modeling_speculative import \
        SpecDecOneEngineForCausalLM

    model = object.__new__(SpecDecOneEngineForCausalLM)
    torch.nn.Module.__init__(model)

    head = torch.nn.Module()
    inner = torch.nn.Module()
    inner.layers = torch.nn.ModuleList([torch.nn.Module(), head])
    model.model = inner
    draft = torch.nn.Module()
    draft.mtp_layers = torch.nn.ModuleList([head])
    model.draft_model = draft
    model.spec_config = spec_config
    return model


def _capture_parent_load_weights(monkeypatch) -> dict:
    """Intercept the base-class load to inspect the dispatch arguments."""
    from tensorrt_llm._torch.models.modeling_utils import \
        DecoderModelForCausalLM

    captured = {}

    def fake_load_weights(self,
                          weights,
                          weight_mapper=None,
                          skip_modules=(),
                          params_map=None,
                          allow_partial_loading=False):
        captured["skip_modules"] = list(skip_modules)
        captured["allow_partial_loading"] = allow_partial_loading
        captured["weights"] = weights

    monkeypatch.setattr(DecoderModelForCausalLM, "load_weights",
                        fake_load_weights)
    return captured


def test_mtp_head_module_names_covers_both_aliases():
    model = _make_one_engine_stub(
        MTPDecodingConfig(max_draft_len=3, speculative_model="/path/to/mtp"))
    assert set(model.mtp_head_module_names()) == {
        "model.layers.1",
        "draft_model.mtp_layers.0",
    }


def test_separate_mtp_target_load_skips_heads_without_partial_loading(
        monkeypatch):
    """The target load must never fall back to partial loading.

    ``allow_partial_loading=True`` suppresses ``process_weights_after_loading``
    on every quantized Linear/MoE it touches, which silently leaves the whole
    target model's quant scales uninitialized (garbage output). The MTP heads
    have to be excluded by module instead.
    """
    captured = _capture_parent_load_weights(monkeypatch)

    model = _make_one_engine_stub(
        MTPDecodingConfig(max_draft_len=3, speculative_model="/path/to/mtp"))
    model.load_weights(
        weights={
            "backbone.layers.0.norm.weight": torch.ones(4),
            "mtp.layers.0.enorm.weight": torch.ones(4),
        })

    assert captured["allow_partial_loading"] is False
    assert set(captured["skip_modules"]) == {
        "draft_model",
        "model.layers.1",
        "draft_model.mtp_layers.0",
    }
    assert "mtp.layers.0.enorm.weight" not in captured["weights"]
    assert "backbone.layers.0.norm.weight" in captured["weights"]


def test_embedded_mtp_target_load_is_unchanged(monkeypatch):
    captured = _capture_parent_load_weights(monkeypatch)

    model = _make_one_engine_stub(MTPDecodingConfig(max_draft_len=3))
    model.load_weights(weights={"mtp.layers.0.enorm.weight": torch.ones(4)})

    assert captured["skip_modules"] == ["draft_model"]
    assert captured["allow_partial_loading"] is False
    # Embedded heads still load from the target checkpoint.
    assert "mtp.layers.0.enorm.weight" in captured["weights"]


def test_nemotron_mapper_remaps_mtp_layers_keys():
    mapper = NemotronHHfWeightMapper()
    pretrained = SimpleNamespace(
        num_hidden_layers=52,
        mamba_head_dim=64,
        mamba_num_heads=8,
        n_groups=8,
        ssm_state_size=128,
        num_key_value_heads=2,
        tie_word_embeddings=False,
    )
    mapping = SimpleNamespace(enable_attention_dp=False,
                              tp_size=1,
                              tp_rank=0)
    model_config = SimpleNamespace(pretrained_config=pretrained,
                                   mapping=mapping,
                                   moe_backend="TRTLLM")
    mapper._config = model_config
    mapper._model = SimpleNamespace(model_config=model_config,
                                    config=pretrained)
    mapper._tp_size = 1

    weights = {
        "mtp.layers.0.enorm.weight": torch.ones(4),
        "mtp.layers.1.norm.weight": torch.ones(4),
        "backbone.layers.0.norm.weight": torch.ones(4),
    }
    remapped = mapper.preprocess_weights(weights)
    assert "model.layers.52.layers.0.enorm.weight" in remapped
    assert "model.layers.52.layers.1.norm.weight" in remapped
    assert "model.layers.0.norm.weight" in remapped
    assert "mtp.layers.0.enorm.weight" not in remapped
