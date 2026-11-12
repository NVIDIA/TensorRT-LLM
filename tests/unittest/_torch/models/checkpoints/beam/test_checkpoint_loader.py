# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU contracts for model-delegated checkpoint loading and worker serialization."""

import builtins
import pickle
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from tensorrt_llm._torch.models.checkpoints.base_checkpoint_loader import BaseCheckpointLoader
from tensorrt_llm._torch.models.checkpoints.beam.checkpoint_loader import (
    BEAM_NVFP4_CHECKPOINT_FORMAT,
    BeamNvfp4CheckpointLoader,
)
from tensorrt_llm._torch.models.checkpoints.hf.config_loader import HfConfigLoader
from tensorrt_llm._torch.models.checkpoints.hf.weight_loader import HfWeightLoader
from tensorrt_llm._torch.models.checkpoints.hf.weight_mapper import HfWeightMapper
from tensorrt_llm.mapping import Mapping

pytestmark = pytest.mark.cpu_only


def test_registered_loader_delegates_without_importing_a_target(monkeypatch):
    original_import = builtins.__import__

    def reject_target_import(name, *args, **kwargs):
        assert "modeling_v2" not in name, f"Unexpected target import: {name}"
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", reject_target_import)
    loader = BaseCheckpointLoader.get(BEAM_NVFP4_CHECKPOINT_FORMAT)
    assert isinstance(loader, BeamNvfp4CheckpointLoader)
    assert not loader.is_weights_preloaded()
    model = SimpleNamespace(load_checkpoint=Mock())
    assert loader.load_weights("checkpoint", Mapping(), model=model) == {}
    model.load_checkpoint.assert_called_once_with("checkpoint")
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


@pytest.mark.parametrize("model", [None, object(), SimpleNamespace(load_checkpoint=42)])
def test_missing_preload_interface_clears_previous_success(model):
    loader = BeamNvfp4CheckpointLoader()
    loader.load_weights("checkpoint", Mapping(), model=SimpleNamespace(load_checkpoint=Mock()))
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
    loader.load_weights("checkpoint", Mapping(), model=SimpleNamespace(load_checkpoint=Mock()))
    restored = pickle.loads(pickle.dumps(loader))
    assert loader.is_weights_preloaded()
    assert not restored.is_weights_preloaded()
    assert type(restored.weight_loader) is HfWeightLoader
    assert type(restored.config_loader) is HfConfigLoader
    if custom:
        assert restored.weight_loader._partial_model_loading is True
        assert type(restored.weight_mapper) is HfWeightMapper
        assert restored.weight_mapper._skip_modules == ["custom-module"]
        assert restored.config_loader.test_setting == "custom-config-setting"
    else:
        assert restored.weight_mapper is None
