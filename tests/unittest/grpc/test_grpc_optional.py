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
"""Optional-dependency and lifecycle tests for the SMG gRPC adapter.

These run correctly with or without the dependency installed: the "missing" case is
simulated so it is meaningful in every environment, and the "present" case is
guarded with ``importorskip``.
"""

import asyncio
import builtins
import importlib
import sys
import types
from unittest.mock import AsyncMock, MagicMock

import pytest


def test_smg_bindings_missing_gives_actionable_error(monkeypatch):
    """A missing 'smg-grpc-proto' yields an actionable install hint.

    Importing the SMG bindings must fail with a ``pip install
    "tensorrt_llm[grpc-smg]"`` hint rather than a bare ImportError.
    """
    real_import = builtins.__import__

    def import_without_smg(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "smg_grpc_proto.generated":
            raise ModuleNotFoundError("No module named 'smg_grpc_proto'", name="smg_grpc_proto")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.delitem(sys.modules, "smg_grpc_proto", raising=False)
    monkeypatch.delitem(sys.modules, "smg_grpc_proto.generated", raising=False)
    monkeypatch.delitem(sys.modules, "tensorrt_llm.grpc.smg.bindings", raising=False)
    monkeypatch.setattr(builtins, "__import__", import_without_smg)

    with pytest.raises(ModuleNotFoundError, match=r"tensorrt_llm\[grpc-smg\]") as exc_info:
        import tensorrt_llm.grpc.smg.bindings  # noqa: F401

    assert exc_info.value.name == "smg_grpc_proto"


def test_smg_bindings_preserves_unrelated_import_error(monkeypatch):
    """A transitive dependency failure is not rewritten as an install hint."""
    smg_package = types.ModuleType("smg_grpc_proto")
    smg_package.__path__ = []
    generated_package = types.ModuleType("smg_grpc_proto.generated")

    def missing_protobuf(_name):
        raise ModuleNotFoundError("No module named 'google.protobuf'", name="google.protobuf")

    generated_package.__getattr__ = missing_protobuf
    monkeypatch.setitem(sys.modules, "smg_grpc_proto", smg_package)
    monkeypatch.setitem(sys.modules, "smg_grpc_proto.generated", generated_package)
    monkeypatch.delitem(sys.modules, "tensorrt_llm.grpc.smg.bindings", raising=False)

    with pytest.raises(ModuleNotFoundError) as exc_info:
        importlib.import_module("tensorrt_llm.grpc.smg.bindings")

    assert exc_info.value.name == "google.protobuf"
    assert "tensorrt_llm[grpc-smg]" not in str(exc_info.value)


def test_smg_bindings_present_smoke():
    """When the dependency is installed, the bindings import cleanly.

    They must expose the pb2 modules the SMG adapter depends on.
    """
    pytest.importorskip(
        "smg_grpc_proto",
        reason="SMG gRPC adapter dependency not installed",
    )
    from tensorrt_llm.grpc.smg import bindings

    assert bindings.trtllm_service_pb2 is not None
    assert bindings.trtllm_service_pb2_grpc is not None


@pytest.mark.cpu_only
@pytest.mark.parametrize("failure_point", ["bind", "start"])
def test_smg_server_startup_failure_cleans_up(monkeypatch, failure_point):
    """Binding and startup failures stop both the gRPC server and LLM."""
    pytest.importorskip(
        "smg_grpc_proto",
        reason="SMG gRPC adapter dependency not installed",
    )
    from tensorrt_llm.grpc.smg import server as server_module

    llm = MagicMock()
    grpc_server = MagicMock()
    grpc_server.start = AsyncMock()
    grpc_server.stop = AsyncMock()

    if failure_point == "bind":
        grpc_server.add_insecure_port.side_effect = RuntimeError("bind failed")
    else:
        grpc_server.add_insecure_port.return_value = 8000
        grpc_server.start.side_effect = RuntimeError("start failed")

    monkeypatch.setitem(sys.modules, "grpc_reflection", None)
    monkeypatch.delitem(sys.modules, "grpc_reflection.v1alpha", raising=False)
    monkeypatch.setattr(server_module.uvloop, "run", asyncio.run)
    monkeypatch.setattr(server_module, "PyTorchLLM", MagicMock(return_value=llm))
    monkeypatch.setattr(server_module, "GrpcRequestManager", MagicMock())
    monkeypatch.setattr(server_module, "TrtllmServiceServicer", MagicMock())
    monkeypatch.setattr(server_module.grpc.aio, "server", MagicMock(return_value=grpc_server))
    monkeypatch.setattr(
        server_module.trtllm_service_pb2_grpc,
        "add_TrtllmServiceServicer_to_server",
        MagicMock(),
    )

    with pytest.raises(RuntimeError, match=f"{failure_point} failed"):
        server_module.launch_smg_server(
            host="127.0.0.1",
            port=8000,
            llm_args={"backend": "pytorch", "model": "test-model"},
        )

    grpc_server.stop.assert_awaited_once_with(grace=5.0)
    llm.shutdown.assert_called_once_with()
