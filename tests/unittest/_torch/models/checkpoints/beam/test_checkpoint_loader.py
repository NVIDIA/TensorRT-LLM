# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU contracts for model-delegated checkpoint loading and worker serialization."""

import builtins
import json
import os
import pickle
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from tensorrt_llm._torch.models.checkpoints.base_checkpoint_loader import BaseCheckpointLoader
from tensorrt_llm._torch.models.checkpoints.beam.checkpoint_loader import (
    BEAM_NVFP4_CHECKPOINT_FORMAT,
    KV_CACHE_SCALES_ENV,
    BeamNvfp4CheckpointLoader,
    read_kv_cache_scales,
)
from tensorrt_llm._torch.models.checkpoints.hf.checkpoint_loader import HfCheckpointLoader
from tensorrt_llm._torch.models.checkpoints.hf.config_loader import HfConfigLoader
from tensorrt_llm._torch.models.checkpoints.hf.weight_loader import HfWeightLoader
from tensorrt_llm._torch.models.checkpoints.hf.weight_mapper import HfWeightMapper
from tensorrt_llm.mapping import Mapping

pytestmark = pytest.mark.cpu_only


@pytest.fixture(autouse=True)
def loader_environment(monkeypatch):
    from tensorrt_llm._torch.distributed import Distributed

    monkeypatch.delenv(KV_CACHE_SCALES_ENV, raising=False)
    gather = Mock(allgather=lambda identity: [identity] * 4)
    monkeypatch.setattr(Distributed, "get", lambda mapping: gather)


def _model():
    return SimpleNamespace(load_checkpoint=Mock(), kv_cache_quant_algo=None, draft_model=None)


def test_registered_loader_delegates_without_importing_a_target(monkeypatch):
    original_import = builtins.__import__

    def reject_target_import(name, *args, **kwargs):
        assert "modeling_v2" not in name, f"Unexpected target import: {name}"
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", reject_target_import)
    loader = BaseCheckpointLoader.get(BEAM_NVFP4_CHECKPOINT_FORMAT)
    assert isinstance(loader, BeamNvfp4CheckpointLoader)
    assert not loader.is_weights_preloaded()
    model = _model()
    assert loader.load_weights("checkpoint", Mapping(), model=model) == {}
    model.load_checkpoint.assert_called_once_with("checkpoint", kv_cache_scales=None)
    assert loader.is_weights_preloaded()

    error = ValueError("checkpoint does not match selected model")
    model.load_checkpoint.side_effect = error
    with pytest.raises(ValueError) as caught:
        loader.load_weights("wrong-checkpoint", Mapping(), model=model)
    assert caught.value is error
    assert not loader.is_weights_preloaded()
    model.load_checkpoint.side_effect = None
    loader.load_weights("checkpoint", Mapping(), model=model)
    assert loader.is_weights_preloaded()


@pytest.mark.parametrize("model", [object(), SimpleNamespace(load_checkpoint=42)])
def test_missing_preload_interface_clears_previous_success(model):
    loader = BeamNvfp4CheckpointLoader()
    loader.load_weights("checkpoint", Mapping(), model=_model())
    with pytest.raises(TypeError, match="callable load_checkpoint"):
        loader.load_weights("checkpoint", Mapping(), model=model)
    assert not loader.is_weights_preloaded()


@pytest.mark.parametrize("custom", [False, True])
def test_worker_pickle_preserves_options_but_not_preloaded_state(custom):
    options = {}
    if custom:
        weight_loader = HfWeightLoader(partial_model_loading=True)
        config_loader = HfConfigLoader()
        weight_mapper = HfWeightMapper()
        weight_mapper._skip_modules = ["custom-module"]
        config_loader.test_setting = "custom-config-setting"
        options = dict(
            weight_loader=weight_loader, config_loader=config_loader, weight_mapper=weight_mapper
        )
    loader = BeamNvfp4CheckpointLoader(**options)
    loader.load_weights("checkpoint", Mapping(), model=_model())
    restored = pickle.loads(pickle.dumps(loader))
    assert loader.is_weights_preloaded()
    assert not restored.is_weights_preloaded()
    assert restored._target_dir is None
    assert type(restored.weight_loader) is HfWeightLoader
    assert type(restored.config_loader) is HfConfigLoader
    if custom:
        assert restored.weight_loader._partial_model_loading is True
        assert type(restored.weight_mapper) is HfWeightMapper
        assert restored.weight_mapper._skip_modules == ["custom-module"]
        assert restored.config_loader.test_setting == "custom-config-setting"
    else:
        assert restored.weight_mapper is None


def test_resolved_scale_table_survives_pickle_without_reading_environment(tmp_path, monkeypatch):
    from tensorrt_llm.quantization.mode import QuantAlgo

    path = tmp_path / "scales.json"
    path.write_text(json.dumps({"target": {"layers": 1, "scales": {"0": 0.5}}}))
    table = read_kv_cache_scales(str(path))
    loader = BeamNvfp4CheckpointLoader(table)
    model = _model()
    model.kv_cache_quant_algo = QuantAlgo.FP8
    loader.load_weights("checkpoint", Mapping(), model=model)
    model.load_checkpoint.assert_called_once_with("checkpoint", kv_cache_scales=table)
    monkeypatch.setenv(KV_CACHE_SCALES_ENV, str(tmp_path / "missing.json"))
    restored = pickle.loads(pickle.dumps(loader))
    assert restored.kv_cache_scales == table
    assert not restored.is_weights_preloaded()
    assert restored._target_dir is None


def test_drafter_load_still_uses_hf_after_target_preload(monkeypatch):
    weights = {"draft": object()}
    hf_load = Mock(return_value=weights)
    monkeypatch.setattr(HfCheckpointLoader, "load_weights", hf_load)
    loader = BeamNvfp4CheckpointLoader()
    mapping = Mapping()
    with pytest.raises(RuntimeError, match="follows the Beam target"):
        loader.load_weights("draft", mapping)
    loader.load_weights("target", mapping, model=_model())
    with pytest.raises(ValueError, match="target's checkpoint"):
        loader.load_weights(os.path.realpath("target"), mapping)
    hf_load.assert_not_called()
    assert loader.load_weights("draft", mapping) is weights
    hf_load.assert_called_once_with("draft", mapping=mapping)
    assert loader.is_weights_preloaded()


def test_failed_scale_validation_clears_previous_target_transaction():
    from tensorrt_llm.quantization.mode import QuantAlgo

    loader = BeamNvfp4CheckpointLoader()
    model = _model()
    loader.load_weights("target", Mapping(), model=model)
    model.kv_cache_quant_algo = QuantAlgo.FP8
    with pytest.raises(ValueError, match="needs the FP8 KV-cache scale table"):
        loader.load_weights("target", Mapping(), model=model)
    assert not loader.is_weights_preloaded()
    assert loader._target_dir is None
    model.load_checkpoint.assert_called_once()
    with pytest.raises(RuntimeError, match="follows the Beam target"):
        loader.load_weights("draft", Mapping())
