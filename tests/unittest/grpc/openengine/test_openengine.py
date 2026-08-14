# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for the OpenEngine gRPC adapter."""

import asyncio
import importlib
import sys
import types
from pathlib import Path

import grpc
import pytest

openengine_pb2_grpc = pytest.importorskip(
    "openengine.v1.openengine_pb2_grpc",
    reason="OpenEngine bindings are installed only for the focused CI shard",
)
server_pb2 = pytest.importorskip(
    "openengine.v1.server_pb2",
    reason="OpenEngine bindings are installed only for the focused CI shard",
)


def test_openengine_stub_returns_unimplemented(monkeypatch: pytest.MonkeyPatch) -> None:
    """The real Control service responds with the stub contract."""
    repo_root = Path(__file__).resolve().parents[4]
    trtllm_package = types.ModuleType("tensorrt_llm")
    trtllm_package.__path__ = [str(repo_root / "tensorrt_llm")]
    grpc_package = types.ModuleType("tensorrt_llm.grpc")
    grpc_package.__path__ = [str(repo_root / "tensorrt_llm" / "grpc")]
    logger_module = types.ModuleType("tensorrt_llm.logger")
    logger_module.logger = types.SimpleNamespace(
        info=lambda *_args: None,
        warning=lambda *_args: None,
    )

    monkeypatch.setitem(sys.modules, "tensorrt_llm", trtllm_package)
    monkeypatch.setitem(sys.modules, "tensorrt_llm.grpc", grpc_package)
    monkeypatch.setitem(sys.modules, "tensorrt_llm.logger", logger_module)
    monkeypatch.delitem(sys.modules, "tensorrt_llm.grpc.openengine", raising=False)
    monkeypatch.delitem(sys.modules, "tensorrt_llm.grpc.openengine.server", raising=False)

    server_module = importlib.import_module("tensorrt_llm.grpc.openengine.server")

    async def exercise_server() -> None:
        server = server_module.OpenEngineServer(host="127.0.0.1", port=0)
        await server.start()
        channel = grpc.aio.insecure_channel(f"127.0.0.1:{server.port}")
        try:
            stub = openengine_pb2_grpc.ControlStub(channel)
            with pytest.raises(grpc.aio.AioRpcError) as error:
                await stub.GetServerInfo(server_pb2.GetServerInfoRequest(), timeout=5)
            assert error.value.code() == grpc.StatusCode.UNIMPLEMENTED
        finally:
            await channel.close()
            await server.stop(grace=0)

    asyncio.run(exercise_server())
