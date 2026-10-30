# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""gRPC Health service with independent Watch clients."""

import asyncio
from collections.abc import AsyncIterator

import grpc
from grpc_health.v1 import health_pb2, health_pb2_grpc


class OpenEngineHealthServicer(health_pb2_grpc.HealthServicer):
    """Publish named health statuses to Check and all active Watch clients."""

    def __init__(self) -> None:
        self._statuses = {
            service: health_pb2.HealthCheckResponse.NOT_SERVING
            for service in ("", "inference", "liveness")
        }
        self._changed = asyncio.Condition()
        self._shutting_down = False

    async def Check(
        self, request: health_pb2.HealthCheckRequest, context: grpc.aio.ServicerContext
    ) -> health_pb2.HealthCheckResponse:
        status = self._statuses.get(request.service)
        if status is None:
            await context.abort(grpc.StatusCode.NOT_FOUND, "unknown service")
        return health_pb2.HealthCheckResponse(status=status)

    async def Watch(
        self, request: health_pb2.HealthCheckRequest, context: grpc.aio.ServicerContext
    ) -> AsyncIterator[health_pb2.HealthCheckResponse]:
        last_status = None
        while True:
            async with self._changed:
                status = self._statuses.get(
                    request.service, health_pb2.HealthCheckResponse.SERVICE_UNKNOWN
                )
                shutting_down = self._shutting_down
                if status == last_status:
                    if shutting_down:
                        return
                    await self._changed.wait()
                    continue
                last_status = status
            yield health_pb2.HealthCheckResponse(status=status)
            if shutting_down:
                return

    async def set(self, service: str, status: int) -> None:
        async with self._changed:
            if self._shutting_down or self._statuses.get(service) == status:
                return
            self._statuses[service] = status
            if service == "inference":
                self._statuses[""] = status
            self._changed.notify_all()

    async def enter_graceful_shutdown(self) -> None:
        async with self._changed:
            if self._shutting_down:
                return
            self._shutting_down = True
            self._statuses["inference"] = health_pb2.HealthCheckResponse.NOT_SERVING
            self._statuses[""] = health_pb2.HealthCheckResponse.NOT_SERVING
            self._changed.notify_all()
