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
import gc
from types import SimpleNamespace

import pytest

grpc = pytest.importorskip(  # noqa: E402
    "grpc", reason='gRPC runtime not installed (pip install "grpcio>=1.67.1,<2")'
)

from tensorrt_llm.grpc.openengine.bindings import openengine_pb2_grpc, server_pb2  # noqa: E402
from tensorrt_llm.grpc.openengine.server import OpenEngineServer  # noqa: E402

# grpc.aio starts a `_poll_wrapper` daemon thread on server start and tears it
# down only once its internal state is released, which happens after the event
# loop closes -- later than pytest-threadleak's teardown snapshot. The thread is
# grpc's to own, not this test's, so exempt the module the same way the SMG
# adapter tests do.
pytestmark = [pytest.mark.cpu_only, pytest.mark.threadleak(enabled=False)]


def test_format_bind_address_brackets_ipv6() -> None:
    """A bare IPv6 host is bracketed; IPv4 and already-bracketed hosts are not."""
    from tensorrt_llm.grpc.openengine.server import _format_bind_address

    assert _format_bind_address("127.0.0.1", 8000) == "127.0.0.1:8000"
    assert _format_bind_address("::1", 8000) == "[::1]:8000"
    assert _format_bind_address("[::1]", 8000) == "[::1]:8000"


def test_openengine_server_serves_the_control_contract() -> None:
    """The server binds, attaches Control, answers, and shuts down cleanly.

    Exercises TRT-LLM's lifecycle around the servicer -- port-zero resolution,
    startup, reachability, graceful stop -- and that Control is wired in and
    answering rather than falling through to the generated base.
    """
    llm = SimpleNamespace(
        args=SimpleNamespace(
            max_seq_len=2048,
            max_batch_size=8,
            max_num_tokens=2048,
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            context_parallel_size=1,
            guided_decoding_backend=None,
            reasoning_parser=None,
            kv_cache_config=SimpleNamespace(tokens_per_block=32),
        ),
        llm_id="test-instance",
        tokenizer=object(),
        _check_health=lambda: True,
    )

    async def exercise_server() -> None:
        server = OpenEngineServer(host="127.0.0.1", port=0, llm=llm, model="test-model")
        # port=0 must be replaced by the kernel-assigned port, or nothing
        # downstream (including this test's channel) can reach the server.
        assert server.port != 0

        await server.start()
        with pytest.raises(RuntimeError, match="Failed to bind"):
            OpenEngineServer(host="127.0.0.1", port=server.port, llm=llm, model="test-model")
        channel = grpc.aio.insecure_channel(f"127.0.0.1:{server.port}")
        try:
            control = openengine_pb2_grpc.ControlStub(channel)
            info = await control.GetServerInfo(server_pb2.GetServerInfoRequest(), timeout=5)
            assert info.engine_name == "tensorrt_llm"
            assert info.schema_revision == 1
            assert info.minimum_client_revision == 1
            assert info.schema_release == "768a93c7b44e40f28c692ad0b471a8f2"
        finally:
            await channel.close()
            await server.stop(grace=0)

    asyncio.run(exercise_server())


def test_is_loopback_classifies_bind_hosts() -> None:
    """The listener is unauthenticated, so a non-loopback bind must be flagged."""
    from tensorrt_llm.grpc.openengine.server import _is_loopback

    assert _is_loopback("127.0.0.1")
    assert _is_loopback("::1")
    assert _is_loopback("[::1]")
    assert _is_loopback("localhost")
    assert not _is_loopback("0.0.0.0")
    assert not _is_loopback("10.0.0.7")
    # An unresolvable name is not assumed safe.
    assert not _is_loopback("some-host")


class _StopLaunch(Exception):
    """Ends launch_server once the server would start accepting requests."""


@pytest.mark.parametrize(
    ("value", "gc_enabled"),
    [("1", False), ("0", True), (None, True)],
)
def test_launch_server_disables_gc_only_when_requested(
    monkeypatch: pytest.MonkeyPatch, value: str | None, gc_enabled: bool
) -> None:
    """TRTLLM_SERVER_DISABLE_GC=1 turns cyclic GC off before serving, as trtllm-serve does."""
    import tensorrt_llm.grpc.openengine.server as oe_server

    if value is None:
        monkeypatch.delenv("TRTLLM_SERVER_DISABLE_GC", raising=False)
    else:
        monkeypatch.setenv("TRTLLM_SERVER_DISABLE_GC", value)
    seen = {}

    class _Llm:
        def __init__(self, **kwargs) -> None:
            del kwargs
            self.llm_id = "test-instance"

        def shutdown(self) -> None:
            pass

    class _Server:
        def __init__(self, **kwargs) -> None:
            del kwargs

        async def start(self) -> None:
            seen["gc_enabled"] = gc.isenabled()
            raise _StopLaunch

        async def stop(self) -> None:
            pass

    monkeypatch.setattr(oe_server, "PyTorchLLM", _Llm)
    monkeypatch.setattr(oe_server, "OpenEngineServer", _Server)
    gc_was_enabled = gc.isenabled()
    gc.enable()
    try:
        with pytest.raises(_StopLaunch):
            oe_server.launch_server("127.0.0.1", 0, {"backend": "pytorch", "model": "test-model"})
    finally:
        if gc_was_enabled:
            gc.enable()
        else:
            gc.disable()

    assert seen["gc_enabled"] is gc_enabled
