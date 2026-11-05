# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``POST /start_profile`` and ``POST /stop_profile``."""

import asyncio
from typing import Any, Optional

from fastapi.responses import JSONResponse

from tensorrt_llm.logger import logger
from tensorrt_llm.serve._base.serving import BaseServing
from tensorrt_llm.serve.openai_protocol import StartProfileRequest


class ProfileServing(BaseServing):
    """Starts and stops the backend engine's runtime profiler."""

    def __init__(self, *, model: str, generator: Any) -> None:
        self.model = model
        self.generator = generator

    async def start_profile(
            self,
            request: Optional[StartProfileRequest] = None) -> JSONResponse:
        """Start runtime profiling in the backend engine.

        Request body (all optional): ``output_dir``, ``num_steps``,
        ``start_step``, ``activities``. See ``StartProfileRequest`` for
        descriptions.

        The backend ``PyExecutor.start_profile`` schedules the profile
        window and the broadcasted ``PROFILE_START_REQUEST_ID`` queue
        item wakes the executor loop, so no tickle-via-generation is
        needed on this side — the captured chrome trace stays free of
        synthetic single-token forward passes.

        The underlying ``GenerationExecutor.start_profile`` call may
        block (it waits for the worker subprocess to ack on the
        IPC-proxy path, up to ~60s). We run it on a worker thread via
        ``asyncio.to_thread`` so the FastAPI event loop stays
        responsive to other endpoints during that wait.
        """
        if request is None:
            request = StartProfileRequest()
        try:
            await asyncio.to_thread(
                self.generator.start_profile,
                output_dir=request.output_dir,
                num_steps=request.num_steps,
                start_step=request.start_step,
                activities=request.activities,
            )
        except RuntimeError as e:
            # ``PyExecutor.start_profile`` raises RuntimeError when a
            # profile window is already active or pending. Surface this
            # to the caller as 409 so they can distinguish it from a
            # generic backend failure (which keeps 500).
            msg = str(e)
            if "already in progress" in msg or "pending" in msg:
                logger.info(f"/start_profile rejected: {msg}")
                return JSONResponse(content={
                    "success": False,
                    "message": msg
                },
                                    status_code=409)
            logger.error(f"/start_profile failed: {e}")
            return JSONResponse(content={
                "success": False,
                "message": msg
            },
                                status_code=500)
        except (OSError, ValueError, TimeoutError) as e:
            # OSError: filesystem/IPC failures while preparing the
            # profile window. ValueError: schema-level rejections that
            # slipped past Pydantic. TimeoutError: ack-queue wait gave
            # up. Anything truly unexpected is allowed to propagate to
            # FastAPI's middleware so we get a real stack trace in the
            # server log instead of swallowing it as a generic 500.
            logger.error(f"/start_profile failed: {e}")
            return JSONResponse(content={
                "success": False,
                "message": str(e)
            },
                                status_code=500)

        return JSONResponse(content={"message": "Profiling started"})

    async def stop_profile(self) -> JSONResponse:
        """Stop any in-progress runtime profiling and flush traces.

        The backend ``PyExecutor.stop_profile`` schedules the stop and
        the broadcasted ``PROFILE_STOP_REQUEST_ID`` queue item wakes the
        executor loop. The call blocks until ``profile_step()`` has
        actually fired the stop, so by the time this handler returns 200
        the chrome trace is on disk. No synthetic ``generate_async([0])``
        tickle is submitted, so the captured trace is free of HTTP-layer
        events.

        The wait can take up to ~35s on the IPC-proxy path. We run the
        blocking call on a worker thread via ``asyncio.to_thread`` so
        the FastAPI event loop stays responsive to other endpoints
        (notably ``/health`` for liveness checks) during the flush.
        """
        try:
            await asyncio.to_thread(self.generator.stop_profile)
        except (RuntimeError, OSError, TimeoutError) as e:
            # RuntimeError: backend rejected the stop or broadcast
            # enqueue failed. OSError: filesystem error flushing the
            # trace. TimeoutError: ack-queue wait gave up. Truly
            # unexpected exceptions propagate to FastAPI so they show
            # up as a real stack trace rather than a swallowed 500.
            logger.error(f"/stop_profile failed: {e}")
            return JSONResponse(content={"error": str(e)}, status_code=500)

        return JSONResponse(content={"message": "Profiling stopped"})
