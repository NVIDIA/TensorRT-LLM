# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest import mock

import pytest
from google.protobuf.message import Message

grpc = pytest.importorskip("grpc")
pytest.importorskip("openengine.v1.openengine_pb2_grpc")

from openengine.v1 import (  # noqa: E402
    generation_pb2,
    kv_pb2,
    lifecycle_pb2,
    lora_pb2,
    model_pb2,
    openengine_pb2_grpc,
    server_pb2,
)

from tensorrt_llm.grpc.openengine import OpenEngineServer  # noqa: E402
from tensorrt_llm.grpc.openengine.server import _format_bind_address  # noqa: E402

_RPC_CASES = (
    ("Generate", openengine_pb2_grpc.InferenceStub, generation_pb2.GenerateRequest, True),
    ("GetServerInfo", openengine_pb2_grpc.ControlStub, server_pb2.GetServerInfoRequest, False),
    ("GetModelInfo", openengine_pb2_grpc.ControlStub, model_pb2.GetModelInfoRequest, False),
    ("GetLoad", openengine_pb2_grpc.ControlStub, server_pb2.GetLoadRequest, False),
    ("Health", openengine_pb2_grpc.ControlStub, lifecycle_pb2.HealthRequest, False),
    ("Abort", openengine_pb2_grpc.ControlStub, lifecycle_pb2.AbortRequest, False),
    ("LoadLora", openengine_pb2_grpc.ControlStub, lora_pb2.LoadLoraRequest, False),
    ("UnloadLora", openengine_pb2_grpc.ControlStub, lora_pb2.UnloadLoraRequest, False),
    ("ListLoras", openengine_pb2_grpc.ControlStub, lora_pb2.ListLorasRequest, False),
    ("GetKvEventSources", openengine_pb2_grpc.ControlStub, kv_pb2.GetKvEventSourcesRequest, False),
    ("SubscribeKvEvents", openengine_pb2_grpc.ControlStub, kv_pb2.SubscribeKvEventsRequest, True),
)


@pytest.mark.asyncio
@pytest.mark.parametrize("method_name,stub_type,request_type,is_stream", _RPC_CASES)
async def test_all_openengine_rpcs_are_unimplemented(
    method_name: str,
    stub_type: type[openengine_pb2_grpc.InferenceStub] | type[openengine_pb2_grpc.ControlStub],
    request_type: type[Message],
    is_stream: bool,
) -> None:
    server = OpenEngineServer(host="127.0.0.1", port=0)
    try:
        await server.start()
        async with grpc.aio.insecure_channel(f"127.0.0.1:{server.port}") as channel:
            rpc = getattr(stub_type(channel), method_name)
            call = rpc(request_type())
            with pytest.raises(grpc.aio.AioRpcError) as error:
                if is_stream:
                    await call.read()
                else:
                    await call
            assert error.value.code() == grpc.StatusCode.UNIMPLEMENTED
    finally:
        await server.stop(grace=0)


@pytest.mark.parametrize(
    ("host", "expected"),
    (
        ("127.0.0.1", "127.0.0.1:50051"),
        ("localhost", "localhost:50051"),
        ("::1", "[::1]:50051"),
        ("[::1]", "[::1]:50051"),
    ),
)
def test_format_bind_address(host: str, expected: str) -> None:
    assert _format_bind_address(host, 50051) == expected


def test_serve_cli_routes_openengine_protocol() -> None:
    from tensorrt_llm.commands.serve import main as serve_main

    with (
        mock.patch(
            "tensorrt_llm.commands.serve.get_is_diffusion_only_model",
            return_value=False,
        ),
        mock.patch("tensorrt_llm.commands.serve.device_count", return_value=1),
        mock.patch("tensorrt_llm.grpc.openengine.server.launch_server") as launch_server,
    ):
        serve_main(
            args=["dummy/model", "--grpc", "--grpc-protocol", "openengine"],
            standalone_mode=False,
        )

    launch_server.assert_called_once_with("localhost", 8000)
