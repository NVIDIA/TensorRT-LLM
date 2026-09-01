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

"""Unit tests for the OpenEngine gRPC adapter.

The OpenEngine bindings resolve only from a custom index, so they are
installed on the CPU-Generic stages that own this test (see the gateway
install guard in jenkins/L0_Test.groovy). Tests that need the bindings are
guarded with ``importorskip`` so the module still collects cleanly in an
environment without them; the optional-dependency error path that every
environment can exercise lives in ``tests/unittest/grpc/test_grpc_optional.py``.
"""

import asyncio

import pytest

openengine_pb2_grpc = pytest.importorskip(
    "openengine.v1.openengine_pb2_grpc",
    reason='OpenEngine bindings not installed (pip install "tensorrt_llm[openengine]")',
)
server_pb2 = pytest.importorskip(
    "openengine.v1.server_pb2",
    reason='OpenEngine bindings not installed (pip install "tensorrt_llm[openengine]")',
)

import grpc  # noqa: E402

from tensorrt_llm.grpc.openengine.server import OpenEngineServer  # noqa: E402

pytestmark = pytest.mark.cpu_only


def test_format_bind_address_brackets_ipv6() -> None:
    """A bare IPv6 host is bracketed; IPv4 and already-bracketed hosts are not."""
    from tensorrt_llm.grpc.openengine.server import _format_bind_address

    assert _format_bind_address("127.0.0.1", 8000) == "127.0.0.1:8000"
    assert _format_bind_address("::1", 8000) == "[::1]:8000"
    assert _format_bind_address("[::1]", 8000) == "[::1]:8000"


def test_openengine_server_serves_the_stub_contract() -> None:
    """The server binds, attaches Control, answers, and shuts down cleanly.

    The servicer is the generated base, so UNIMPLEMENTED is its contract;
    what this exercises is TRT-LLM's own lifecycle around it -- port-zero
    resolution, startup, reachability, and graceful stop.
    """

    async def exercise_server() -> None:
        server = OpenEngineServer(host="127.0.0.1", port=0)
        # port=0 must be replaced by the kernel-assigned port, or nothing
        # downstream (including this test's channel) can reach the server.
        assert server.port != 0

        await server.start()
        channel = grpc.aio.insecure_channel(f"127.0.0.1:{server.port}")
        try:
            control = openengine_pb2_grpc.ControlStub(channel)
            with pytest.raises(grpc.aio.AioRpcError) as error:
                await control.GetServerInfo(server_pb2.GetServerInfoRequest(), timeout=5)
            assert error.value.code() == grpc.StatusCode.UNIMPLEMENTED
        finally:
            await channel.close()
            await server.stop(grace=0)

    asyncio.run(exercise_server())
