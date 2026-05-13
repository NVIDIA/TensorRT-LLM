# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the in-tree ModelExpress checkpoint loader shim."""

import sys
import types

import pytest

from tensorrt_llm._torch.models.checkpoints.base_checkpoint_loader import BaseCheckpointLoader
from tensorrt_llm._torch.models.checkpoints.mx.checkpoint_loader import MXCheckpointLoader


class FakeModelexpressMXCheckpointLoader:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


def _install_fake_modelexpress_loader(monkeypatch):
    fake_modelexpress = types.ModuleType("modelexpress")
    fake_engines = types.ModuleType("modelexpress.engines")
    fake_trtllm = types.ModuleType("modelexpress.engines.trtllm")
    fake_loader = types.ModuleType("modelexpress.engines.trtllm.loader")
    fake_loader.MXCheckpointLoader = FakeModelexpressMXCheckpointLoader

    fake_modelexpress.engines = fake_engines
    fake_engines.trtllm = fake_trtllm
    fake_trtllm.loader = fake_loader

    monkeypatch.setitem(sys.modules, "modelexpress", fake_modelexpress)
    monkeypatch.setitem(sys.modules, "modelexpress.engines", fake_engines)
    monkeypatch.setitem(sys.modules, "modelexpress.engines.trtllm", fake_trtllm)
    monkeypatch.setitem(sys.modules, "modelexpress.engines.trtllm.loader", fake_loader)


def test_shim_instantiates_modelexpress_loader(monkeypatch):
    _install_fake_modelexpress_loader(monkeypatch)

    loader = MXCheckpointLoader(mx_server_url="mx.example:8001")

    assert isinstance(loader, FakeModelexpressMXCheckpointLoader)
    assert loader.kwargs == {"mx_server_url": "mx.example:8001"}


def test_registry_instantiates_modelexpress_loader(monkeypatch):
    _install_fake_modelexpress_loader(monkeypatch)

    loader = BaseCheckpointLoader.get(
        checkpoint_format="modelexpress",
        mx_server_url="mx.example:8001",
        model_name="Qwen/Qwen2.5-7B",
    )

    assert isinstance(loader, FakeModelexpressMXCheckpointLoader)
    assert loader.kwargs == {
        "mx_server_url": "mx.example:8001",
        "model_name": "Qwen/Qwen2.5-7B",
    }


def test_shim_requires_modelexpress(monkeypatch):
    monkeypatch.setitem(sys.modules, "modelexpress", None)
    monkeypatch.setitem(sys.modules, "modelexpress.engines", None)
    monkeypatch.setitem(sys.modules, "modelexpress.engines.trtllm", None)
    monkeypatch.setitem(sys.modules, "modelexpress.engines.trtllm.loader", None)

    with pytest.raises(ImportError, match="requires the modelexpress Python package"):
        MXCheckpointLoader()
