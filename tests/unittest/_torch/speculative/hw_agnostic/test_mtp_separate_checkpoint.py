# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from types import SimpleNamespace

import pytest
import torch
from transformers import PretrainedConfig

from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models import modeling_speculative
from tensorrt_llm._torch.models.checkpoints.hf.nemotron_h_weight_mapper import (
    NemotronHHfWeightMapper,
)
from tensorrt_llm._torch.speculative.utils import (
    filter_mtp_checkpoint_weights,
    resolve_mtp_checkpoint_source,
    select_mtp_checkpoint_weights,
    skip_modules_for_separate_mtp_checkpoint,
    update_spec_config_from_model_config,
    uses_mtp_head_checkpoint,
)
from tensorrt_llm.llmapi.llm_args import Eagle3DecodingConfig, MTPDecodingConfig


class _ExternalDraftModelTarget:
    build_mtp_draft_model_from_config = True


class _EmbeddedOrHeadReplacementTarget:
    pass


def _resolve_as_head_checkpoint(spec_config):
    pretrained_config = PretrainedConfig(architectures=["TargetModel"], num_nextn_predict_layers=1)
    update_spec_config_from_model_config(
        spec_config, pretrained_config, _EmbeddedOrHeadReplacementTarget
    )


def test_needs_separate_draft_weights_for_mtp_with_speculative_model():
    cfg = MTPDecodingConfig(max_draft_len=3, speculative_model="/path/to/mtp")
    _resolve_as_head_checkpoint(cfg)
    assert cfg.needs_separate_draft_weights is True

    cfg_no_draft = MTPDecodingConfig(max_draft_len=3)
    assert cfg_no_draft.needs_separate_draft_weights is False


def test_needs_separate_draft_weights_still_true_for_eagle3():
    cfg = Eagle3DecodingConfig(max_draft_len=3, speculative_model="/path/to/eagle3")
    assert cfg.needs_separate_draft_weights is True


def test_uses_mtp_head_checkpoint_helper():
    spec_config = MTPDecodingConfig(max_draft_len=3, speculative_model="/path/to/mtp")
    _resolve_as_head_checkpoint(spec_config)
    assert uses_mtp_head_checkpoint(spec_config) is True
    assert uses_mtp_head_checkpoint(MTPDecodingConfig(max_draft_len=3)) is False
    assert uses_mtp_head_checkpoint(None) is False


@pytest.mark.parametrize(
    ("speculative_model", "target_model_cls", "expected_checkpoint_type"),
    [
        (None, _EmbeddedOrHeadReplacementTarget, "embedded"),
        ("/path/to/mtp", _EmbeddedOrHeadReplacementTarget, "head_replacement"),
        ("/path/to/assistant", _ExternalDraftModelTarget, "external_draft_model"),
    ],
)
def test_mtp_checkpoint_type_selects_draft_model_constructor(
    monkeypatch, speculative_model, target_model_cls, expected_checkpoint_type
):
    spec_config = MTPDecodingConfig(max_draft_len=3, speculative_model=speculative_model)
    pretrained_config = PretrainedConfig(architectures=["TargetModel"], num_hidden_layers=52)
    model_config = ModelConfig(
        spec_config=spec_config,
        pretrained_config=pretrained_config,
    )
    update_spec_config_from_model_config(spec_config, pretrained_config, target_model_cls)

    external_draft_model = object()
    replacement_mtp_heads = object()
    monkeypatch.setattr(
        modeling_speculative.AutoModelForCausalLM,
        "from_config",
        lambda draft_config: external_draft_model,
    )
    monkeypatch.setattr(
        modeling_speculative,
        "MTPForCausalLM",
        lambda *args: replacement_mtp_heads,
    )

    draft_config = object() if expected_checkpoint_type == "external_draft_model" else None
    draft_model = modeling_speculative.get_draft_model(
        model_config, draft_config, object(), object()
    )

    if expected_checkpoint_type == "external_draft_model":
        assert spec_config.uses_external_draft_model
        assert not spec_config.uses_replacement_heads
        assert draft_model is external_draft_model
    else:
        assert not spec_config.uses_external_draft_model
        assert spec_config.uses_replacement_heads is (
            expected_checkpoint_type == "head_replacement"
        )
        assert draft_model is replacement_mtp_heads


def test_speculative_model_equal_to_target_keeps_embedded_mtp(tmp_path):
    """speculative_model == the target checkpoint is the pre-feature API usage."""
    target_dir = tmp_path / "target"
    target_dir.mkdir()

    cfg = MTPDecodingConfig(max_draft_len=3, speculative_model=str(target_dir))

    resolve_mtp_checkpoint_source(cfg, str(target_dir))
    assert uses_mtp_head_checkpoint(cfg) is False
    assert cfg.needs_separate_draft_weights is False
    # The user-provided value is left untouched.
    assert cfg.speculative_model == str(target_dir)


def test_speculative_model_equal_to_target_matches_equivalent_paths(tmp_path):
    target_dir = tmp_path / "target"
    target_dir.mkdir()
    link_dir = tmp_path / "target_link"
    link_dir.symlink_to(target_dir, target_is_directory=True)

    cfg = MTPDecodingConfig(max_draft_len=3, speculative_model=str(link_dir))
    resolve_mtp_checkpoint_source(cfg, str(target_dir) + "/")
    assert uses_mtp_head_checkpoint(cfg) is False


def test_separate_mtp_checkpoint_survives_resolution(tmp_path):
    target_dir = tmp_path / "target"
    target_dir.mkdir()
    mtp_dir = tmp_path / "mtp_heads"
    mtp_dir.mkdir()

    cfg = MTPDecodingConfig(max_draft_len=3, speculative_model=str(mtp_dir))
    resolve_mtp_checkpoint_source(cfg, str(target_dir))
    _resolve_as_head_checkpoint(cfg)
    assert uses_mtp_head_checkpoint(cfg) is True
    assert cfg.needs_separate_draft_weights is True


def test_resolution_does_not_affect_eagle3(tmp_path):
    """Eagle3 always loads its draft from speculative_model, same dir or not."""
    target_dir = tmp_path / "target"
    target_dir.mkdir()

    cfg = Eagle3DecodingConfig(max_draft_len=3, speculative_model=str(target_dir))
    resolve_mtp_checkpoint_source(cfg, str(target_dir))
    assert cfg.needs_separate_draft_weights is True


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
        json.dumps(
            {
                "num_nextn_predict_layers": 1,
                "mtp_hybrid_override_pattern": "*E",
                "mtp_block_configs": [{"block_type": "moe", "num_experts": 8}],
            }
        )
    )

    # Target has no MTP (or stale MTP metadata).
    model_config = SimpleNamespace(
        architectures=["NemotronHForCausalLM"],
        num_nextn_predict_layers=0,
        mtp_layers_block_type=None,
        mtp_block_configs=None,
    )
    spec_config = MTPDecodingConfig(max_draft_len=5, speculative_model=str(mtp_dir))

    update_spec_config_from_model_config(spec_config, model_config)

    assert spec_config.num_nextn_predict_layers == 1
    assert model_config.num_nextn_predict_layers == 1
    # Legacy pattern string is converted to the writable HF field.
    assert model_config.mtp_layers_block_type == ["attention", "moe"]
    assert model_config.mtp_block_configs == [{"block_type": "moe", "num_experts": 8}]
    # User-set max_draft_len is preserved for Eagle-style replay.
    assert spec_config.max_draft_len == 5


def test_update_spec_config_uses_mtp_layers_block_type_when_present(tmp_path):
    mtp_dir = tmp_path / "mtp_heads"
    mtp_dir.mkdir()
    (mtp_dir / "config.json").write_text(
        json.dumps(
            {
                "num_nextn_predict_layers": 1,
                "mtp_layers_block_type": ["attention", "moe"],
            }
        )
    )

    model_config = SimpleNamespace(
        architectures=["NemotronHForCausalLM"],
        num_nextn_predict_layers=0,
        mtp_layers_block_type=None,
    )
    spec_config = MTPDecodingConfig(max_draft_len=3, speculative_model=str(mtp_dir))
    update_spec_config_from_model_config(spec_config, model_config)
    assert model_config.mtp_layers_block_type == ["attention", "moe"]


def test_remap_preprocessed_mtp_weights_for_draft_model():
    from tensorrt_llm._torch.speculative.utils import remap_preprocessed_mtp_weights_for_draft_model

    weights = {
        "model.layers.52.layers.0.enorm.weight": torch.ones(4),
        "model.layers.52.layers.1.norm.weight": torch.ones(4),
    }
    remapped = remap_preprocessed_mtp_weights_for_draft_model(
        weights, num_hidden_layers=52, num_mtp_layers=1
    )
    assert remapped == {
        "mtp_layers.0.layers.0.enorm.weight": weights["model.layers.52.layers.0.enorm.weight"],
        "mtp_layers.0.layers.1.norm.weight": weights["model.layers.52.layers.1.norm.weight"],
    }


def test_skip_modules_for_separate_mtp_checkpoint_shared_head():
    # Nemotron-style: no shared_head tensors -> skip so strict load does not
    # demand an absent module.
    nemotron_keys = {
        "mtp_layers.0.layers.0.enorm.weight": torch.ones(4),
        "mtp_layers.0.layers.1.final_layernorm.weight": torch.ones(4),
    }
    assert skip_modules_for_separate_mtp_checkpoint(nemotron_keys) == ["shared_head"]

    # DeepSeek / Qwen / Exaone: shared_head.norm is present -> load it.
    deepseek_keys = {
        "mtp_layers.0.enorm.weight": torch.ones(4),
        "mtp_layers.0.shared_head.norm.weight": torch.ones(4),
    }
    assert skip_modules_for_separate_mtp_checkpoint(deepseek_keys) == []

    # Step3: shared_head also owns an output projection -> still load.
    step3_keys = {
        "mtp_layers.0.shared_head.norm.weight": torch.ones(4),
        "mtp_layers.0.shared_head.output.weight": torch.ones(4, 4),
    }
    assert skip_modules_for_separate_mtp_checkpoint(step3_keys) == []


def _make_one_engine_stub(spec_config, num_hidden_layers: int = 52):
    """A bare SpecDecOneEngineForCausalLM with just the module tree we need.

    ``__init__`` builds a whole target model, so construct the instance
    directly and register only the two aliases of a single MTP head: the
    target's ``model.layers[N]`` and ``draft_model.mtp_layers[0]``.
    """
    from tensorrt_llm._torch.models.modeling_speculative import SpecDecOneEngineForCausalLM

    _resolve_as_head_checkpoint(spec_config)

    class _OneEngineStub(SpecDecOneEngineForCausalLM):
        # The real ``config`` is a read-only property over
        # ``model_config.pretrained_config``, which this stub never builds.
        config = SimpleNamespace(num_hidden_layers=num_hidden_layers)

    model = object.__new__(_OneEngineStub)
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
    from tensorrt_llm._torch.models.modeling_utils import DecoderModelForCausalLM

    captured = {}

    def fake_load_weights(
        self,
        weights,
        weight_mapper=None,
        skip_modules=(),
        params_map=None,
        allow_partial_loading=False,
    ):
        captured["skip_modules"] = list(skip_modules)
        captured["allow_partial_loading"] = allow_partial_loading
        captured["weights"] = weights

    monkeypatch.setattr(DecoderModelForCausalLM, "load_weights", fake_load_weights)
    return captured


def test_mtp_head_module_names_covers_both_aliases():
    model = _make_one_engine_stub(
        MTPDecodingConfig(max_draft_len=3, speculative_model="/path/to/mtp")
    )
    assert set(model.mtp_head_module_names()) == {
        "model.layers.1",
        "draft_model.mtp_layers.0",
    }


def test_separate_mtp_target_load_skips_heads_without_partial_loading(monkeypatch):
    """The target load must never fall back to partial loading.

    ``allow_partial_loading=True`` suppresses ``process_weights_after_loading``
    on every quantized Linear/MoE it touches, which silently leaves the whole
    target model's quant scales uninitialized (garbage output). The MTP heads
    have to be excluded by module instead.
    """
    captured = _capture_parent_load_weights(monkeypatch)

    model = _make_one_engine_stub(
        MTPDecodingConfig(max_draft_len=3, speculative_model="/path/to/mtp")
    )
    model.load_weights(
        weights={
            "backbone.layers.0.norm.weight": torch.ones(4),
            "mtp.layers.0.enorm.weight": torch.ones(4),
        }
    )

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


def test_target_load_keeps_heads_when_speculative_model_is_target(monkeypatch, tmp_path):
    captured = _capture_parent_load_weights(monkeypatch)

    target_dir = tmp_path / "target"
    target_dir.mkdir()
    spec_config = MTPDecodingConfig(max_draft_len=3, speculative_model=str(target_dir))
    resolve_mtp_checkpoint_source(spec_config, str(target_dir))

    model = _make_one_engine_stub(spec_config)
    model.load_weights(weights={"mtp.layers.0.enorm.weight": torch.ones(4)})

    assert captured["skip_modules"] == ["draft_model"]
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
    mapping = SimpleNamespace(enable_attention_dp=False, tp_size=1, tp_rank=0)
    model_config = SimpleNamespace(
        pretrained_config=pretrained, mapping=mapping, moe_backend="TRTLLM"
    )
    mapper._config = model_config
    mapper._model = SimpleNamespace(model_config=model_config, config=pretrained)
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


def _nemotron_style_mtp_weights(*, include_shared_head: bool) -> dict:
    """Minimal remappable mtp.* tensors that satisfy the Nemotron required-suffix check."""
    weights = {
        "mtp.layers.0.enorm.weight": torch.ones(4),
        "mtp.layers.0.hnorm.weight": torch.ones(4),
        "mtp.layers.0.eh_proj.weight": torch.ones(4, 4),
        "mtp.layers.1.final_layernorm.weight": torch.ones(4),
    }
    if include_shared_head:
        weights["mtp.shared_head.norm.weight"] = torch.ones(4)
    return weights


class _PassthroughMtpMapper:
    """Nemotron-like remap: ``mtp.layers.{{i}}.*`` -> ``model.layers.{{N}}.layers.{{i}}.*``."""

    def __init__(self, num_hidden_layers: int):
        self._num_hidden_layers = num_hidden_layers

    def preprocess_weights(self, weights: dict) -> dict:
        out = {}
        for key, value in weights.items():
            if key.startswith("mtp.layers."):
                _, _, sublayer_idx, rest = key.split(".", 3)
                out[f"model.layers.{self._num_hidden_layers}.layers.{sublayer_idx}.{rest}"] = value
            elif key.startswith("mtp."):
                out[f"model.layers.{self._num_hidden_layers}.{key[len('mtp.') :]}"] = value
            else:
                out[key] = value
        return out


def test_separate_mtp_draft_load_skip_shared_head_scales(monkeypatch):
    """Draft load skips shared_head only when the remapped checkpoint omits it."""
    from tensorrt_llm._torch.models import modeling_utils

    captured = {}

    def fake_load_weights_impl_v2(
        model, weights, weight_mapper, skip_modules=(), allow_partial_loading=False, **kwargs
    ):
        captured["skip_modules"] = list(skip_modules)
        captured["allow_partial_loading"] = allow_partial_loading
        captured["weight_keys"] = set(weights)

    monkeypatch.setattr(modeling_utils, "_load_weights_impl_v2", fake_load_weights_impl_v2)

    spec_config = MTPDecodingConfig(max_draft_len=1, speculative_model="/path/to/mtp")
    model = _make_one_engine_stub(spec_config, num_hidden_layers=52)
    mapper = _PassthroughMtpMapper(num_hidden_layers=52)

    model.load_draft_weights(
        weights=_nemotron_style_mtp_weights(include_shared_head=False),
        weight_mapper=mapper,
    )
    assert captured["skip_modules"] == ["shared_head"]
    assert captured["allow_partial_loading"] is False
    assert not any("shared_head" in k for k in captured["weight_keys"])

    model.load_draft_weights(
        weights=_nemotron_style_mtp_weights(include_shared_head=True),
        weight_mapper=mapper,
    )
    assert captured["skip_modules"] == []
    assert "mtp_layers.0.shared_head.norm.weight" in captured["weight_keys"]
