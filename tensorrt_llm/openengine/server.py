# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Lifecycle wrapper for the optional OpenEngine sibling gRPC server."""

import importlib
import ipaddress

import grpc
from openengine.v1 import openengine_pb2_grpc, server_pb2

from tensorrt_llm.inputs.multimodal import MultimodalServerConfig
from tensorrt_llm.logger import logger
from tensorrt_llm.serve.kv_event_fanout import KvEventFanout
from tensorrt_llm.serve.request_tracker import RequestTracker
from tensorrt_llm.serve.stats_fanout import StatsFanout

from ._schema_pin import OPENENGINE_COMMIT
from .servicer import OpenEngineServicer, schema_release

_MAX_MESSAGE_LENGTH = 256 * 1024 * 1024
_IPV4_UNSPECIFIED = str(ipaddress.IPv4Address(0))


def validate_schema_release(schema_release: str) -> str:
    """Require an immutable OpenEngine source identity before binding."""
    if schema_release == OPENENGINE_COMMIT:
        return schema_release
    raise RuntimeError(
        "OPENENGINE_SCHEMA_RELEASE must exactly match the pinned OPENENGINE_COMMIT "
        f"({OPENENGINE_COMMIT})"
    )


def validate_runtime_dependencies(llm: object) -> None:
    """Fail startup when an advertised modality lacks its runtime decoder."""
    processor = getattr(llm, "input_processor", None)
    modalities: set[str] = set()
    for getter_name in (
        "get_openengine_modalities",
        "get_openengine_prefill_decode_modalities",
    ):
        getter = getattr(processor, getter_name, None)
        if callable(getter):
            modalities.update(getter())
    if "video" in modalities:
        try:
            importlib.import_module("cv2")
        except ImportError as exc:
            raise RuntimeError(
                "TensorRT-LLM OpenEngine advertises video input for this model, but "
                "OpenCV is not installed. Install the OpenEngine runtime dependencies "
                "with `python scripts/install_openengine.py` or "
                "`pip install 'tensorrt-llm[openengine]'`."
            ) from exc


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
        validate_runtime_dependencies(llm)
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
        openengine_pb2_grpc.add_InferenceServicer_to_server(self.servicer, self._server)
        openengine_pb2_grpc.add_ControlServicer_to_server(self.servicer, self._server)
        bound = self._server.add_insecure_port(f"{host}:{port}")
        if bound == 0:
            raise RuntimeError(f"Failed to bind OpenEngine server to {host}:{port}")
        self.port = bound if port == 0 else port
        advertised_host = {
            _IPV4_UNSPECIFIED: "127.0.0.1",
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
            if not await self.servicer.tracker.close_reapers(timeout=grace):
                logger.warning(
                    "OpenEngine request cleanup did not finish within %.1f seconds", grace
                )
            self.servicer.close()
            if self._kv_event_fanout is not None:
                await self._kv_event_fanout.stop()
            if self._stats_fanout is not None:
                await self._stats_fanout.stop()
        logger.info("OpenEngine sibling server stopped")


def openengine_role(server_role: object | None) -> int:
    """Map TRT-LLM serve roles onto the supported OpenEngine milestone."""
    if server_role is None:
        return server_pb2.ENGINE_ROLE_AGGREGATED
    name = getattr(server_role, "name", str(server_role)).upper()
    if name == "CONTEXT":
        return server_pb2.ENGINE_ROLE_PREFILL
    if name == "GENERATION":
        return server_pb2.ENGINE_ROLE_DECODE
    if name in ("MM_ENCODER", "VISUAL_GEN"):
        raise ValueError(f"OpenEngine does not support TRT-LLM server role {name!r}")
    raise ValueError(f"Unknown TRT-LLM server role {name!r}")


__all__ = [
    "OpenEngineServer",
    "openengine_role",
    "validate_runtime_dependencies",
    "validate_schema_release",
]
