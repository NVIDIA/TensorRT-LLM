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
