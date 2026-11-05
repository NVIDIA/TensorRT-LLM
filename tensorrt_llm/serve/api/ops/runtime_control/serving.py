# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Runtime control: ``/release_memory``, ``/resume_memory`` and ``/memory_status``."""

import asyncio
from http import HTTPStatus
from typing import Any, List, Optional

from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse

from tensorrt_llm._torch.async_llm import AsyncLLM
from tensorrt_llm.serve._base.serving import BaseServing
from tensorrt_llm.serve.openai_protocol import RuntimeMemoryUpdateRequest
from tensorrt_llm.serve.runtime_control_auth import (
    RuntimeControlReplayCache, validate_runtime_control_request)


class RuntimeControlServing(BaseServing):
    """Memory checkpointing of the engine behind signed, replay-protected requests."""

    def __init__(
            self, *, model: str, generator: Any,
            runtime_control_api_key: Optional[str],
            runtime_control_replay_cache: RuntimeControlReplayCache) -> None:
        self.model = model
        self.generator = generator
        self._runtime_control_api_key = runtime_control_api_key
        self._runtime_control_replay_cache = runtime_control_replay_cache

    async def _require_runtime_control_auth(self, raw_request: Request) -> None:
        body = await raw_request.body()
        try:
            route = raw_request.scope.get("route")
            route_path = getattr(route, "path", None)
            if route_path is None:
                raise ValueError("Runtime control route is unavailable.")
            validate_runtime_control_request(
                self._runtime_control_api_key,
                raw_request.method,
                route_path,
                body,
                raw_request.headers,
                self._runtime_control_replay_cache,
            )
        except ValueError as error:
            raise HTTPException(
                status_code=HTTPStatus.UNAUTHORIZED,
                detail=str(error),
            ) from error

    @staticmethod
    def _runtime_control_error_response(error: Exception) -> JSONResponse:
        if isinstance(error, ValueError):
            status_code = HTTPStatus.BAD_REQUEST
        elif (isinstance(error, RuntimeError) and
              ("already parked" in str(error) or "while state is" in str(error)
               or "admission transition" in str(error))):
            status_code = HTTPStatus.CONFLICT
        else:
            status_code = HTTPStatus.INTERNAL_SERVER_ERROR
        return JSONResponse(status_code=status_code,
                            content={"error": {
                                "message": str(error)
                            }})

    async def _call_runtime_memory_method(self, method: str,
                                          tags: Optional[List[str]]) -> object:
        operation = getattr(self.generator, method)
        if isinstance(self.generator, AsyncLLM):
            return await operation(tags)
        return await asyncio.to_thread(operation, tags)

    async def runtime_release_memory(
        self,
        request: Optional[RuntimeMemoryUpdateRequest] = None,
    ) -> JSONResponse:
        try:
            if request is not None and request.tags == []:
                raise ValueError("Runtime memory tags must not be empty.")
            tags = None if request is None else request.tags
            await self._call_runtime_memory_method("release", tags)
        except Exception as error:
            return self._runtime_control_error_response(error)
        return JSONResponse(content={"status": "success"})

    async def runtime_resume_memory(
        self,
        request: Optional[RuntimeMemoryUpdateRequest] = None,
    ) -> JSONResponse:
        try:
            if request is not None and request.tags == []:
                raise ValueError("Runtime memory tags must not be empty.")
            tags = None if request is None else request.tags
            await self._call_runtime_memory_method("resume", tags)
        except Exception as error:
            return self._runtime_control_error_response(error)
        return JSONResponse(content={"status": "success"})

    async def runtime_memory_status(self) -> JSONResponse:
        try:
            if isinstance(self.generator, AsyncLLM):
                status = await self.generator.get_memory_status()
            else:
                status = await asyncio.to_thread(
                    self.generator.get_memory_status)
        except Exception as error:
            return JSONResponse(status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
                                content={"error": {
                                    "message": str(error)
                                }})
        return JSONResponse(content=status.model_dump(mode="json"))
