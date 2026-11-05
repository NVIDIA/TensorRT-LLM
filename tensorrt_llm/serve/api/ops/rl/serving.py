# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The RL control endpoints: ``/release_memory``, ``/resume_memory``, ``/update_weights``."""

from http import HTTPStatus
from typing import Any, Optional

from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse

from tensorrt_llm._torch.async_llm import AsyncLLM
from tensorrt_llm.serve._base.serving import BaseServing
from tensorrt_llm.serve.openai_protocol import (MemoryUpdateRequest,
                                                UpdateWeightsRequest)
from tensorrt_llm.serve.rl_control_auth import validate_rl_control_request


class RLServing(BaseServing):
    """Collective RPCs of an ``AsyncLLM`` behind the HMAC-signed RL control header."""

    def __init__(self, *, model: str, generator: Any,
                 rl_control_api_key: Optional[str]) -> None:
        self.model = model
        self.generator = generator
        self._rl_control_api_key = rl_control_api_key

    async def _require_rl_control_auth(self, raw_request: Request) -> None:
        body = await raw_request.body()
        try:
            validate_rl_control_request(self._rl_control_api_key, body,
                                        raw_request.headers)
        except ValueError as e:
            raise HTTPException(status_code=HTTPStatus.UNAUTHORIZED,
                                detail=str(e)) from e

    async def release_memory(self,
                             request: MemoryUpdateRequest) -> JSONResponse:
        assert isinstance(
            self.generator, AsyncLLM
        ), "/release_memory endpoint is only supported with AsyncLLM()"
        await self.generator.collective_rpc('sleep', args=(request.tags, ))
        return JSONResponse(content={"status": "success"})

    async def resume_memory(self, request: MemoryUpdateRequest) -> JSONResponse:
        assert isinstance(
            self.generator, AsyncLLM
        ), "/resume_memory endpoint is only supported with AsyncLLM()"
        await self.generator.collective_rpc('wakeup', args=(request.tags, ))
        return JSONResponse(content={"status": "success"})

    async def update_weights(self,
                             request: UpdateWeightsRequest) -> JSONResponse:
        assert isinstance(
            self.generator, AsyncLLM
        ), "/update_weights endpoint is only supported with AsyncLLM()"
        await self.generator.collective_rpc('update_weights',
                                            args=(request.weights, ))
        return JSONResponse(content={"status": "success"})
