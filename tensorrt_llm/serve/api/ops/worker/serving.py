# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The worker endpoints of an LLM server.

``/health_generate``, ``/v1/data_transceiver_state``, ``/energy_metrics``,
``/steady_clock_offset``, ``/kv_cache_events`` and ``/server_info``.
"""

import asyncio
import time
import traceback
from typing import Annotated, Any, Optional, Protocol

from fastapi import Body, Request
from fastapi.responses import JSONResponse, Response

from tensorrt_llm._utils import (AdjustedSteadyClock,
                                 get_global_steady_clock_now_in_seconds)
from tensorrt_llm.logger import logger
from tensorrt_llm.runtime.kv_cache_hash import \
    get_effective_kv_cache_event_hash_algo
from tensorrt_llm.serve._base.serving import BaseServing
from tensorrt_llm.serve.multi_frontend import SharedClockOffset
from tensorrt_llm.serve.openai_protocol import ChatCompletionRequest
from tensorrt_llm.serve.render import RenderResources


class _ChatBackend(Protocol):
    """What ``health_generate`` calls on the chat serving object.

    A Protocol rather than an import: packages under ``serve/api/`` do not
    import each other, so the composition root injects the chat serving
    object.
    """

    async def openai_chat(self, request: ChatCompletionRequest,
                          raw_request: Request) -> Response:
        ...


class WorkerServing(BaseServing):
    """The worker endpoints, served from injected dependencies."""

    def __init__(self, *, model: str, generator: Any,
                 adjusted_steady_clock: AdjustedSteadyClock,
                 shared_clock_offset: Optional[SharedClockOffset],
                 chat_serving: _ChatBackend,
                 render_resources: RenderResources) -> None:
        self.model = model
        self.generator = generator
        self._adjusted_steady_clock = adjusted_steady_clock
        self._shared_clock_offset = shared_clock_offset
        self._chat_serving = chat_serving
        self._render_resources = render_resources
        # The composition root's lifespan hands the energy monitor over once
        # it exists.
        self.energy_monitor = None
        # get_server_info computes the rendering fingerprint once, then
        # caches it here.
        self._render_fingerprint = None

    async def data_transceiver_state(self) -> JSONResponse:
        """Return the serialized DataTransceiverState as base64-encoded JSON."""
        state = self.generator.get_data_transceiver_state()
        if not state:
            return JSONResponse(
                status_code=404,
                content={"error": "No transceiver state available"})
        import base64
        return JSONResponse(content={
            "data_transceiver_state":
            base64.b64encode(state).decode("utf-8")
        })

    async def health_generate(self, raw_request: Request) -> Response:
        """Health check that performs a minimal generation."""
        extra_args = {}
        if self.generator.args.max_beam_width > 1:
            extra_args = dict(
                use_beam_search=True,
                best_of=self.generator.args.max_beam_width,
                n=1,
            )
        try:
            # Create a minimal chat request
            health_request = ChatCompletionRequest(
                messages=[{
                    "role": "user",
                    "content": "hi"
                }],  # Minimal prompt (often > 1 token after tokenization)
                model=self.model,
                max_completion_tokens=1,  # Request only 1 token out
                stream=False,
                temperature=0.0,  # Deterministic output
                **extra_args,
            )

            # Call the chat completion logic
            response = await self._chat_serving.openai_chat(
                health_request, raw_request)

            # Check if the response indicates success (status code 200)
            if response.status_code == 200:
                return Response(status_code=200,
                                content="Generation health check OK")
            else:
                logger.error(
                    f"Health generate check failed with status code: {response.status_code}"
                )
                try:
                    # Attempt to get body for more details if possible
                    body = response.body if hasattr(
                        response, 'body') else await response.body()
                    logger.error(f"Health generate check response body: {body}")
                except Exception:
                    pass  # Ignore errors trying to get body details
                return Response(status_code=500,
                                content="Generation health check failed")

        except Exception as e:
            logger.error(f"Health generate check encountered exception: {e}")
            return Response(status_code=500,
                            content=f"Generation health check failed: {str(e)}")

    async def get_energy_metrics(self) -> JSONResponse:
        if self.energy_monitor is None:
            return JSONResponse(
                content={"error": "Energy monitoring is not available"},
                status_code=503)
        total_energy = self.energy_monitor.get_current_energy()
        if total_energy is None:
            return JSONResponse(content={"error": "Failed to read GPU energy"},
                                status_code=503)
        return JSONResponse(
            content={
                "total_energy_j": round(total_energy, 4),
                "query_time": time.perf_counter(),
            })

    async def set_steady_clock_offset(
            self, offset: Annotated[float, Body(embed=True)]) -> Response:
        self._adjusted_steady_clock.set_reference_offset(offset)
        logger.info(
            f"The steady clock offset between local and disagg server: {offset} second"
        )
        # Multi-frontend launcher: every frontend stamps perf metrics with it.
        if self._shared_clock_offset is not None:
            try:
                self._shared_clock_offset.publish(offset)
            except OSError as e:
                logger.error(
                    f"Failed to share the steady clock offset with the "
                    f"attached frontends: {e}")
                return Response(status_code=500, content=str(e))
        return Response(status_code=200)

    async def get_steady_clock_offset(self) -> JSONResponse:
        # The calibrated offset is applied to AdjustedSteadyClock, whose source
        # is the rank-adjusted global clock. Sample that same clock here so a
        # process-global offset is not counted twice in emitted metrics.
        receive_ts = get_global_steady_clock_now_in_seconds()
        await asyncio.sleep(0.2)
        transmit_ts = get_global_steady_clock_now_in_seconds()
        return JSONResponse(content={
            "receive_ts": receive_ts,
            "transmit_ts": transmit_ts
        })

    async def get_kv_cache_events(self) -> JSONResponse:
        events = []
        try:
            async for event in self.generator.get_kv_cache_events_async(0):
                events.append(event)
        except (IndexError, asyncio.QueueEmpty):
            # queue is empty, no more events
            pass
        return JSONResponse(content=events)

    async def _live_tokens_per_block(self) -> Optional[int]:
        """Return the runtime's effective KV block size, or None if unknown.

        The executor layer already turns RPC failures into an empty dict, so
        the only failure left to absorb here is ``encode_only``, which rejects
        the call outright. Generators without a KV cache (VisualGen) have no
        such method. Both mean "fall back to the configured value".
        """
        get_capacity = getattr(self.generator, "get_kv_cache_capacity", None)
        if get_capacity is None:
            return None
        try:
            # Off-loop: the RPC blocks, and this worker's own heartbeat task
            # shares this loop, so stalling it here can lapse its registration.
            capacity = await asyncio.to_thread(get_capacity)
        except RuntimeError as e:
            logger.debug(f"Could not read live tokens_per_block: {e}")
            return None
        return capacity.get("tokensPerBlock") or None

    async def get_server_info(self) -> JSONResponse:
        # Note: calling self.generator.disaggregated_params and startup_metrics below
        # may trigger an RPC sync call, blocking the server event loop. Since this server_info
        # is usually called only once before accepting requests, it's not a big concern.
        content = {"disaggregated_params": self.generator.disaggregated_params}
        # A router or caller checks this before trusting prompt token ids that
        # another process rendered; it is computed once, then reused.
        render_fingerprint = getattr(self, "_render_fingerprint", None)
        if render_fingerprint is None:
            try:
                # Hashing the vocabulary takes a moment; keep the event loop free.
                render_fingerprint = await asyncio.to_thread(
                    lambda: self._render_resources.fingerprint())
                self._render_fingerprint = render_fingerprint
            except Exception:
                logger.warning("Could not compute the rendering fingerprint:\n"
                               f"{traceback.format_exc()}")
        if render_fingerprint is not None:
            content["render_fingerprint"] = render_fingerprint
        args = getattr(self.generator, "args", None)
        if args is not None:
            if args.max_batch_size is not None:
                content["max_batch_size"] = args.max_batch_size
            kv_cache_config = args.kv_cache_config
            if kv_cache_config is not None:
                content[
                    "kv_cache_hash_algo"] = get_effective_kv_cache_event_hash_algo(
                        kv_cache_config.kv_cache_event_hash_algo,
                        kv_cache_config.use_kv_cache_manager_v2)
                if kv_cache_config.tokens_per_block is not None:
                    content[
                        "tokens_per_block"] = kv_cache_config.tokens_per_block
            # The runtime may override the configured block size (e.g. FlashMLA
            # forces 64) in the worker process, so args.kv_cache_config still
            # holds the pre-override value here. A kv-cache-aware router hashes
            # prompts in whatever block size this endpoint publishes, so a
            # stale value makes every block hash miss. Prefer the live value.
            live_tokens_per_block = await self._live_tokens_per_block()
            if live_tokens_per_block is not None:
                content["tokens_per_block"] = live_tokens_per_block
        content["startup_metrics"] = getattr(self.generator, "startup_metrics",
                                             {})
        return JSONResponse(content=content)
