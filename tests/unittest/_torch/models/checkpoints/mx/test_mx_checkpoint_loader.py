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
"""Unit tests for the native TRT-LLM entrypoint into MX strategies."""

import logging
import sys
from dataclasses import replace
from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest

from tensorrt_llm._torch.models.checkpoints.auto_mapper import AutoCheckpointMapper
from tensorrt_llm._torch.models.checkpoints.base_checkpoint_loader import BaseCheckpointLoader
from tensorrt_llm._torch.models.checkpoints.hf.checkpoint_loader import HfCheckpointLoader
from tensorrt_llm._torch.models.checkpoints.hf.qwen3_next_weight_mapper import (
    Qwen3NextHfWeightMapper,
)
from tensorrt_llm._torch.models.checkpoints.hf.weight_mapper import HfWeightMapper
from tensorrt_llm._torch.models.checkpoints.mx import checkpoint_loader as checkpoint_loader_mod
from tensorrt_llm._torch.models.checkpoints.mx.checkpoint_loader import MXCheckpointLoader
from tensorrt_llm._torch.weight_sharing import (
    ARTIFACT_IDENTITY_FORMAT_VERSION,
    LLAMA_POST_TRANSFORM_LAYOUT_ABI_V1,
    SOURCE_IDENTITY_FORMAT_VERSION,
    ArtifactIdentity,
    SourceIdentity,
)

pytestmark = pytest.mark.cpu_only


def _source_identity() -> SourceIdentity:
    return SourceIdentity(
        format_version=SOURCE_IDENTITY_FORMAT_VERSION,
        artifact_identity=ArtifactIdentity(
            format_version=ARTIFACT_IDENTITY_FORMAT_VERSION,
            scheme="checkpoint_manifest_sha256",
            digest="0" * 64,
        ),
        model_fingerprint="model",
        quant_fingerprint="quant",
        backend_fingerprint="backend",
        parallel_fingerprint="parallel",
        rank=0,
        shard_fingerprint="shard",
        transform_abi_id=LLAMA_POST_TRANSFORM_LAYOUT_ABI_V1,
        model_name="meta-llama/Llama-3.1-8B-Instruct",
    )


def _loader(**kwargs):
    weight_loader = MagicMock()
    weight_loader.load_weights.return_value = {"disk": object()}
    config_loader = MagicMock()
    loader = MXCheckpointLoader(
        weight_loader=weight_loader,
        config_loader=config_loader,
        **kwargs,
    )
    return loader, weight_loader, config_loader


def _load_kwargs(**overrides):
    values = {
        "mapping": MagicMock(),
        "model": MagicMock(),
        "source_identity": _source_identity(),
        "model_config": MagicMock(),
        "load_config": MagicMock(),
        "allow_post_transform_weights": True,
        "prepare_post_transform_receiver": MagicMock(),
        "post_transform_protocol_version": 1,
    }
    values.update(overrides)
    return values


def _install_fake_mx(
    monkeypatch,
    *,
    p2p_succeeded,
    value,
    transform_protocol_version=1,
):
    instances = []

    class MxModelLoader:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.p2p_succeeded = p2p_succeeded
            self.transform_protocol_version = transform_protocol_version if p2p_succeeded else None
            self.publish_model = MagicMock()
            self.cleanup = MagicMock()
            instances.append(self)

        def load_model(self, model):
            self.model = model
            return value

    module = ModuleType("modelexpress.engines.trtllm")
    module.MxModelLoader = MxModelLoader
    monkeypatch.setitem(sys.modules, module.__name__, module)
    return instances


def test_construction_preserves_checkpoint_loader_contract():
    loader, _, _ = _loader(mx_server_url="mx:8001")

    assert isinstance(loader, HfCheckpointLoader)
    assert isinstance(loader, BaseCheckpointLoader)
    assert loader.checkpoint_format == "MX"
    assert loader._checkpoint_format == "MX"
    assert loader.mx_server_url == "mx:8001"
    assert not loader.is_weights_preloaded()


@pytest.mark.parametrize(
    ("effective_level", "expected_level"),
    ((logging.WARNING, logging.INFO), (logging.DEBUG, None)),
)
def test_transfer_log_dir_enables_info_records(monkeypatch, effective_level, expected_level):
    monkeypatch.setenv("MX_TRANSFER_LOG_DIR", "/tmp/mx-transfer-logs")
    mx_logger = MagicMock()
    mx_logger.getEffectiveLevel.return_value = effective_level

    with patch.object(checkpoint_loader_mod.logging, "getLogger", return_value=mx_logger):
        checkpoint_loader_mod._enable_mx_transfer_logging()

    if expected_level is None:
        mx_logger.setLevel.assert_not_called()
    else:
        mx_logger.setLevel.assert_called_once_with(expected_level)


def test_registered_under_mx_and_mapper_fallback_is_preserved():
    loader = BaseCheckpointLoader.get(
        checkpoint_format="MX",
        weight_loader=None,
        weight_mapper=None,
        config_loader=None,
        mx_server_url="mx:8001",
    )

    assert isinstance(loader, MXCheckpointLoader)
    assert isinstance(
        AutoCheckpointMapper.get("MX", "Qwen3NextForCausalLM"),
        Qwen3NextHfWeightMapper,
    )
    assert isinstance(
        AutoCheckpointMapper.get("MX", "UnknownArchitecture"),
        HfWeightMapper,
    )


def test_missing_mx_state_uses_native_hf_loader():
    loader, weight_loader, _ = _loader()
    kwargs = _load_kwargs()

    value = loader.load_weights("checkpoint", **kwargs)

    assert value == weight_loader.load_weights.return_value
    weight_loader.load_weights.assert_called_once_with(
        "checkpoint",
        mapping=kwargs["mapping"],
    )


def test_missing_trtllm_adapter_uses_native_hf_loader(monkeypatch):
    module = ModuleType("modelexpress.engines.trtllm")
    monkeypatch.setitem(sys.modules, module.__name__, module)
    loader, weight_loader, _ = _loader(mx_server_url="mx:8001")
    kwargs = _load_kwargs()

    value = loader.load_weights("checkpoint", **kwargs)

    assert value == weight_loader.load_weights.return_value
    weight_loader.load_weights.assert_called_once_with(
        "checkpoint",
        mapping=kwargs["mapping"],
    )


@pytest.mark.parametrize("fallback", ["missing-state", "missing-adapter"])
def test_native_fallback_releases_previous_mx_session(monkeypatch, fallback):
    instances = _install_fake_mx(
        monkeypatch,
        p2p_succeeded=False,
        value={"disk": object()},
    )
    loader, weight_loader, _ = _loader(mx_server_url="mx:8001")
    loader.load_weights("checkpoint", **_load_kwargs())
    previous_session = instances[0]

    kwargs = _load_kwargs()
    if fallback == "missing-state":
        kwargs["model"] = None
    else:
        monkeypatch.setitem(sys.modules, "modelexpress.engines.trtllm", ModuleType("empty"))

    assert loader.load_weights("checkpoint", **kwargs) == weight_loader.load_weights.return_value
    previous_session.cleanup.assert_called_once_with()
    assert loader._mx_loader is None

    loader.post_load_publish(
        MagicMock(),
        checkpoint_dir="checkpoint",
        weights_preloaded=False,
    )
    previous_session.publish_model.assert_not_called()


def test_trtllm_adapter_dependency_error_is_not_hidden(monkeypatch):
    def fail_import(_name):
        raise ModuleNotFoundError("No module named 'nixl'", name="nixl")

    monkeypatch.setattr(checkpoint_loader_mod, "import_module", fail_import)
    loader, _, _ = _loader(mx_server_url="mx:8001")

    with pytest.raises(ModuleNotFoundError, match="nixl"):
        loader.load_weights("checkpoint", **_load_kwargs())


def test_qualified_llama_delegates_to_shared_chain(monkeypatch):
    value = {}
    instances = _install_fake_mx(
        monkeypatch,
        p2p_succeeded=True,
        value=value,
    )
    loader, _, _ = _loader(mx_server_url="mx:8001")
    kwargs = _load_kwargs()

    assert loader.load_weights("checkpoint", **kwargs) is value

    session = instances[0]
    assert session.model is kwargs["model"]
    assert session.kwargs["checkpoint_loader"] is loader
    assert session.kwargs["checkpoint_dir"] == "checkpoint"
    assert session.kwargs["mapping"] is kwargs["mapping"]
    assert session.kwargs["source_identity"] is kwargs["source_identity"]
    assert session.kwargs["model_config"] is kwargs["model_config"]
    assert session.kwargs["load_config"] is kwargs["load_config"]
    assert session.kwargs["p2p_enabled"] is True
    assert session.kwargs["transform_protocol_version"] == 1
    assert loader.is_weights_preloaded()
    assert loader.is_post_transform_weights_preloaded()


def test_unqualified_model_keeps_rdma_unavailable(monkeypatch):
    instances = _install_fake_mx(
        monkeypatch,
        p2p_succeeded=False,
        value={"disk": object()},
    )
    loader, _, _ = _loader(mx_server_url="mx:8001")
    kwargs = _load_kwargs(
        allow_post_transform_weights=False,
        prepare_post_transform_receiver=None,
    )

    loader.load_weights("checkpoint", **kwargs)

    assert instances[0].kwargs["p2p_enabled"] is False
    assert not loader.is_weights_preloaded()
    assert not loader.is_post_transform_weights_preloaded()


def test_qualified_model_requires_receiver_preparation(monkeypatch):
    _install_fake_mx(monkeypatch, p2p_succeeded=True, value={})
    loader, _, _ = _loader(mx_server_url="mx:8001")

    with pytest.raises(
        RuntimeError,
        match="requires receiver structure preparation",
    ):
        loader.load_weights(
            "checkpoint",
            **_load_kwargs(prepare_post_transform_receiver=None),
        )


def test_qualified_model_requires_transform_protocol(monkeypatch):
    _install_fake_mx(monkeypatch, p2p_succeeded=True, value={})
    loader, _, _ = _loader(mx_server_url="mx:8001")

    with pytest.raises(
        RuntimeError,
        match="requires a transform protocol version",
    ):
        loader.load_weights(
            "checkpoint",
            **_load_kwargs(post_transform_protocol_version=None),
        )


@pytest.mark.parametrize(
    "source_identity",
    [
        replace(_source_identity(), format_version=SOURCE_IDENTITY_FORMAT_VERSION - 1),
        replace(_source_identity(), transform_abi_id=None),
    ],
    ids=["old-format", "missing-transform-abi"],
)
def test_qualified_model_requires_current_source_identity(monkeypatch, source_identity):
    _install_fake_mx(monkeypatch, p2p_succeeded=True, value={})
    loader, _, _ = _loader(mx_server_url="mx:8001")

    with pytest.raises(
        RuntimeError,
        match="current TRT-LLM SourceIdentity format and a transform-layout ABI",
    ):
        loader.load_weights(
            "checkpoint",
            **_load_kwargs(source_identity=source_identity),
        )


def test_incompatible_transfer_protocol_fails_closed(monkeypatch):
    _install_fake_mx(
        monkeypatch,
        p2p_succeeded=True,
        value={},
        transform_protocol_version=2,
    )
    loader, _, _ = _loader(mx_server_url="mx:8001")

    with pytest.raises(
        RuntimeError,
        match="compatible TRT-LLM transform protocol and SourceIdentity ABI",
    ):
        loader.load_weights("checkpoint", **_load_kwargs())

    assert not loader.is_weights_preloaded()
    assert not loader.is_post_transform_weights_preloaded()


def test_p2p_transfer_cannot_return_native_weights(monkeypatch):
    _install_fake_mx(
        monkeypatch,
        p2p_succeeded=True,
        value={"disk": object()},
    )
    loader, _, _ = _loader(mx_server_url="mx:8001")

    with pytest.raises(
        RuntimeError,
        match="MX P2P loading must not return native checkpoint weights",
    ):
        loader.load_weights("checkpoint", **_load_kwargs())

    assert not loader.is_weights_preloaded()
    assert not loader.is_post_transform_weights_preloaded()


def test_repeated_load_clears_cleaned_session_before_replacement(monkeypatch):
    instances = _install_fake_mx(
        monkeypatch,
        p2p_succeeded=False,
        value={"disk": object()},
    )
    loader, _, _ = _loader(mx_server_url="mx:8001")
    loader.load_weights("checkpoint", **_load_kwargs())
    first_session = instances[0]

    class FailingMxModelLoader:
        def __init__(self, **_kwargs):
            raise RuntimeError("construction failed")

    sys.modules["modelexpress.engines.trtllm"].MxModelLoader = FailingMxModelLoader

    with pytest.raises(RuntimeError, match="construction failed"):
        loader.load_weights("checkpoint", **_load_kwargs())

    first_session.cleanup.assert_called_once_with()
    assert loader._mx_loader is None

    loader.cleanup()
    first_session.cleanup.assert_called_once_with()


def test_p2p_receiver_republishes_after_trt_post_load(monkeypatch):
    instances = _install_fake_mx(
        monkeypatch,
        p2p_succeeded=True,
        value={},
    )
    loader, _, _ = _loader(mx_server_url="mx:8001")
    kwargs = _load_kwargs()
    model = kwargs["model"]
    loader.load_weights("checkpoint", **kwargs)
    instances[0].publish_model.assert_not_called()

    loader.post_load_publish(
        model,
        checkpoint_dir="checkpoint",
        weights_preloaded=True,
        source_identity=kwargs["source_identity"],
    )

    instances[0].publish_model.assert_called_once_with(model)


def test_native_source_publishes_after_trt_post_load(monkeypatch):
    instances = _install_fake_mx(
        monkeypatch,
        p2p_succeeded=False,
        value={"disk": object()},
    )
    loader, _, _ = _loader(mx_server_url="mx:8001")
    kwargs = _load_kwargs()
    model = kwargs["model"]
    loader.load_weights("checkpoint", **kwargs)
    instances[0].publish_model.assert_not_called()

    loader.post_load_publish(
        model,
        checkpoint_dir="checkpoint",
        weights_preloaded=False,
        source_identity=kwargs["source_identity"],
    )

    instances[0].publish_model.assert_called_once_with(model)


def test_cleanup_releases_mx_and_native_loader_resources(monkeypatch):
    instances = _install_fake_mx(
        monkeypatch,
        p2p_succeeded=False,
        value={"disk": object()},
    )
    loader, weight_loader, config_loader = _loader(mx_server_url="mx:8001")
    loader.load_weights("checkpoint", **_load_kwargs())

    loader.cleanup()

    instances[0].cleanup.assert_called_once_with()
    weight_loader.cleanup.assert_called_once_with()
    config_loader.cleanup.assert_called_once_with()
    assert loader._mx_loader is None


def test_cleanup_continues_when_mx_cleanup_fails(monkeypatch):
    instances = _install_fake_mx(
        monkeypatch,
        p2p_succeeded=False,
        value={"disk": object()},
    )
    loader, weight_loader, config_loader = _loader(mx_server_url="mx:8001")
    loader.load_weights("checkpoint", **_load_kwargs())
    instances[0].cleanup.side_effect = RuntimeError("cleanup failed")

    loader.cleanup()

    instances[0].cleanup.assert_called_once_with()
    weight_loader.cleanup.assert_called_once_with()
    config_loader.cleanup.assert_called_once_with()
    assert loader._mx_loader is None
