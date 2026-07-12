# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Lifecycle wrapper for the optional OpenEngine sibling gRPC server."""

import grpc
from openengine.v1 import engine_pb2, openengine_pb2_grpc

from tensorrt_llm.inputs.multimodal import MultimodalServerConfig
from tensorrt_llm.logger import logger
from tensorrt_llm.serve.kv_event_fanout import KvEventFanout
from tensorrt_llm.serve.request_tracker import RequestTracker
from tensorrt_llm.serve.stats_fanout import StatsFanout

from ._schema_pin import OPENENGINE_COMMIT
from .servicer import OpenEngineServicer, schema_release

_MAX_MESSAGE_LENGTH = 256 * 1024 * 1024


def validate_schema_release(schema_release: str) -> str:
    """Require an immutable OpenEngine source identity before binding."""
    if schema_release == OPENENGINE_COMMIT:
        return schema_release
    raise RuntimeError(
        "OPENENGINE_SCHEMA_RELEASE must exactly match the pinned OPENENGINE_COMMIT "
        f"({OPENENGINE_COMMIT})"
    )


class OpenEngineServer:
    """An OpenEngine gRPC server which never owns or shuts down the LLM."""

    def __init__(
        self,
        llm: object,
        model: str,
        role: int,
        host: str,
        port: int,
        tracker: RequestTracker,
        media_config: MultimodalServerConfig | None = None,
        reasoning_parser: str | None = None,
        tool_parser: str | None = None,
        kv_event_fanout: KvEventFanout | None = None,
        stats_fanout: StatsFanout | None = None,
    ) -> None:
        validate_schema_release(schema_release())
        self.host = host
        self.port = port
        self._kv_event_fanout = kv_event_fanout
        self._stats_fanout = stats_fanout
        self._server = grpc.aio.server(
            options=[
                ("grpc.max_send_message_length", _MAX_MESSAGE_LENGTH),
                ("grpc.max_receive_message_length", _MAX_MESSAGE_LENGTH),
                ("grpc.keepalive_time_ms", 30_000),
                ("grpc.keepalive_timeout_ms", 10_000),
            ]
        )
        self.servicer = OpenEngineServicer(
            llm=llm,
            model=model,
            role=role,
            tracker=tracker,
            media_config=media_config,
            reasoning_parser=reasoning_parser,
            tool_parser=tool_parser,
            kv_event_fanout=kv_event_fanout,
            stats_fanout=stats_fanout,
            event_host=host,
            event_port=port,
        )
        openengine_pb2_grpc.add_OpenEngineServicer_to_server(self.servicer, self._server)
        bound = self._server.add_insecure_port(f"{host}:{port}")
        if bound == 0:
            raise RuntimeError(f"Failed to bind OpenEngine server to {host}:{port}")
        self.port = bound if port == 0 else port
        advertised_host = {
            "0.0.0.0": "127.0.0.1",
            "::": "::1",
            "[::]": "::1",
        }.get(host, host)
        self.servicer.event_host = advertised_host
        self.servicer.event_port = self.port

    async def start(self) -> None:
        if self._kv_event_fanout is not None:
            self._kv_event_fanout.start()
        if self._stats_fanout is not None:
            self._stats_fanout.start()
        try:
            await self._server.start()
        except BaseException:
            if self._kv_event_fanout is not None:
                await self._kv_event_fanout.stop()
            if self._stats_fanout is not None:
                await self._stats_fanout.stop()
            self.servicer.close()
            raise
        logger.info("OpenEngine sibling server started on %s:%d", self.host, self.port)

    async def stop(self, grace: float = 5.0) -> None:
        try:
            await self._server.stop(grace=grace)
        finally:
            self.servicer.close()
            if self._kv_event_fanout is not None:
                await self._kv_event_fanout.stop()
            if self._stats_fanout is not None:
                await self._stats_fanout.stop()
        logger.info("OpenEngine sibling server stopped")


def openengine_role(server_role: object | None) -> int:
    """Map TRT-LLM serve roles onto the supported OpenEngine milestone."""
    if server_role is None:
        return engine_pb2.ENGINE_ROLE_AGGREGATED
    name = getattr(server_role, "name", str(server_role)).upper()
    if name == "CONTEXT":
        return engine_pb2.ENGINE_ROLE_PREFILL
    if name == "GENERATION":
        return engine_pb2.ENGINE_ROLE_DECODE
    if name in ("MM_ENCODER", "VISUAL_GEN"):
        raise ValueError(f"OpenEngine does not support TRT-LLM server role {name!r}")
    raise ValueError(f"Unknown TRT-LLM server role {name!r}")


__all__ = ["OpenEngineServer", "openengine_role", "validate_schema_release"]
