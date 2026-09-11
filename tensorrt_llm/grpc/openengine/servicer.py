# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""OpenEngine Inference service adapter for the TensorRT-LLM LLM API."""

import asyncio
from collections.abc import AsyncIterator
from typing import Any

import grpc
from openengine.v1 import error_pb2, generation_pb2, openengine_pb2_grpc

from tensorrt_llm.executor.request import DEFAULT_REQUEST_PRIORITY
from tensorrt_llm.llmapi.llm import LLM
from tensorrt_llm.logger import logger

from .disagg import disaggregated_params_from_request
from .errors import AbortFailedError, UnsupportedFeatureError
from .formatting import _engine_error_response, _stop_texts
from .request_mapping import _input_from_request, _trace_headers, sampling_params_from_request
from .streaming import (
    RESPONSE_STALL_TIMEOUT_SECONDS,
    ActiveRequest,
    StallWatchdog,
    _format_result,
    _OpenEngineFormatState,
)


class OpenEngineInferenceServicer(openengine_pb2_grpc.InferenceServicer):
    """Translate OpenEngine generation streams to TensorRT-LLM requests."""

    def __init__(self, llm: LLM, model: str, kv_transfer_backend: str = "") -> None:
        self._llm = llm
        self._model = model
        # Informational label placed in the KvSessionRef of PrefillReady events so
        # a generation worker/orchestrator can see which KV transfer backend the
        # context worker uses. The actual transfer is driven by opaque_state.
        self._kv_transfer_backend = kv_transfer_backend
        self._guided_backend = llm.args.guided_decoding_backend
        self._active_requests: dict[str, Any] = {}

    def active_request_count(self) -> int:
        """Requests accepted and not yet completed."""
        return len(self._active_requests)

    def track_request(self, request_id: str, handle: Any) -> bool:
        """Register an in-flight request. False when the id is already active.

        Callers with no request_id of their own (the Control health probe) pass
        a synthetic one, so their engine work is still counted by GetLoad and
        reachable by Abort(all_requests).
        """
        return ActiveRequest(self._active_requests, request_id, handle).register()

    def untrack_request(self, request_id: str, handle: Any) -> None:
        """Drop the registration, but only if `handle` still owns it."""
        ActiveRequest(self._active_requests, request_id, handle).release()

    def abort_request_by_id(self, request_id: str) -> bool:
        """Abort one in-flight request. Returns False when it is not active.

        Does not pop the id: the Generate stream's ``finally`` owns cleanup, and
        removing it here would let a duplicate request_id past the ALREADY_EXISTS
        check while the original stream is still draining.

        Raises:
            AbortFailedError: the request is active but the engine refused to
                abort it, so it is still running.
        """
        handle = self._active_requests.get(request_id)
        if handle is None:
            return False
        try:
            handle.abort()
        except Exception as error:
            logger.warning(f"Failed to abort OpenEngine request {request_id}: {error}")
            raise AbortFailedError(request_id, error) from error
        return True

    def abort_all_requests(self) -> tuple[int, int]:
        """Abort every in-flight request; returns (aborted, failed) counts."""
        aborted = 0
        failed = 0
        for request_id in list(self._active_requests):
            try:
                if self.abort_request_by_id(request_id):
                    aborted += 1
            except AbortFailedError:
                failed += 1
        return aborted, failed

    async def Generate(
        self,
        request: generation_pb2.GenerateRequest,
        context: grpc.aio.ServicerContext,
    ) -> AsyncIterator[generation_pb2.GenerateResponse]:
        """Run a server-streaming OpenEngine generation request."""
        request_id = request.request_id
        try:
            if not request_id:
                raise ValueError("request_id must be non-empty")
            if not request.model:
                raise ValueError("model must be non-empty")
            # Like the HTTP /v1/completions single-model server, any non-empty
            # model name is accepted and served by the loaded model (no NOT_FOUND
            # on a mismatch) so the two transports behave consistently.
            if request_id in self._active_requests:
                await context.abort(
                    grpc.StatusCode.ALREADY_EXISTS,
                    f"request_id '{request_id}' is already active",
                )
                return
            if request.media:
                raise UnsupportedFeatureError("multimodal media is not supported")
            if request.lora_name:
                raise UnsupportedFeatureError("LoRA selection is not supported")
            if request.HasField("kv"):
                if request.kv.HasField("bypass_prefix_cache") and request.kv.bypass_prefix_cache:
                    raise UnsupportedFeatureError("prefix cache bypass is not supported")

            inputs = _input_from_request(request)
            sampling_params = sampling_params_from_request(request, self._guided_backend)
            trace_headers = _trace_headers(context)
            cache_salt = (
                request.kv.cache_salt
                if request.HasField("kv") and request.kv.HasField("cache_salt")
                else None
            )
            disaggregated_params = disaggregated_params_from_request(request)
        except UnsupportedFeatureError as error:
            await context.abort(grpc.StatusCode.UNIMPLEMENTED, str(error))
            return
        except ValueError as error:
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(error))
            return

        format_state = _OpenEngineFormatState(
            request_id=request_id,
            sampling_params=sampling_params,
            is_context_only=(
                disaggregated_params is not None
                and disaggregated_params.request_type == "context_only"
            ),
            kv_transfer_backend=self._kv_transfer_backend,
            stop_texts=_stop_texts(sampling_params),
            exclude_stop=not sampling_params.include_stop_str_in_output,
            tokenizer=self._llm.tokenizer,
        )

        try:
            result_handle = self._llm.generate_async(
                inputs=inputs,
                sampling_params=sampling_params,
                streaming=True,
                trace_headers=trace_headers,
                cache_salt=cache_salt,
                priority=DEFAULT_REQUEST_PRIORITY,
                disaggregated_params=disaggregated_params,
            )
        except ValueError as error:
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(error))
            return
        # A TypeError here is generate_async's signature drifting under us, not
        # a malformed request. Reporting it as INVALID_ARGUMENT would make a
        # router treat a totally broken worker as a fleet of bad clients: it
        # would not retry elsewhere, eject the engine, or alarm.
        except TypeError as error:
            logger.error(f"OpenEngine request {request_id} could not be submitted: {error}")
            await context.abort(grpc.StatusCode.INTERNAL, str(error))
            return
        # Broad by design: request submission can fail in engine-specific ways we
        # cannot enumerate here; any such failure must become a clean INTERNAL
        # status for the client rather than an unhandled servicer crash.
        except Exception as error:
            logger.error(f"Failed to submit OpenEngine request {request_id}: {error}")
            await context.abort(grpc.StatusCode.INTERNAL, str(error))
            return

        engine_terminal = False
        abort_requested = False
        registration = ActiveRequest(self._active_requests, request_id, result_handle)

        def abort_request(reason: str) -> None:
            nonlocal abort_requested
            if abort_requested or engine_terminal:
                return
            abort_requested = True
            try:
                result_handle.abort()
            # Broad by design: aborting is best-effort cleanup and must never
            # propagate out of this callback.
            except Exception as error:
                logger.warning(
                    f"Failed to abort OpenEngine request {request_id} ({reason}): {error}"
                )

        def on_stall() -> None:
            logger.warning(
                f"OpenEngine request {request_id} stopped consuming responses "
                f"for {RESPONSE_STALL_TIMEOUT_SECONDS:g} seconds"
            )
            abort_request("response consumer stalled")
            # The generator is suspended in `yield` and may never resume, so its
            # `finally` may never run. Release the registration here or the id
            # stays in flight forever: GetLoad over-reports it and the id is
            # permanently unusable.
            registration.release()

        watchdog = StallWatchdog(
            asyncio.get_running_loop(), RESPONSE_STALL_TIMEOUT_SECONDS, on_stall
        )

        context.add_done_callback(lambda _: abort_request("RPC completed before generation"))

        try:
            # Register inside the try so the finally below always releases the
            # id. There is no await between the duplicate-id check above and
            # here, so the asyncio single-thread atomicity that makes that check
            # correct still holds, while an exception before streaming can no
            # longer leak the id (which would make it permanently un-reusable,
            # since re-submits are rejected with ALREADY_EXISTS).
            registration.register()
            watchdog.start()
            async for result in result_handle:
                if context.cancelled():
                    abort_request("RPC cancelled")
                    return
                responses = _format_result(result, format_state)
                for resp in responses or ():
                    watchdog.pending()
                    yield resp
                    watchdog.delivered()
                    if watchdog.stalled:
                        engine_terminal = True
                        watchdog.pending()
                        yield _engine_error_response(
                            request_id,
                            "response consumer stalled",
                            result,
                            code=error_pb2.ERROR_CODE_OVERLOADED,
                            retryable=True,
                        )
                        return
                if result.error:
                    engine_terminal = True
                    return
                if result.finished:
                    engine_terminal = True
                    return

            # Defensive: the engine is expected to mark the request finished before
            # its result stream ends, so reaching here means it closed early (e.g. an
            # internal engine abort). Surface it as an error rather than silently
            # returning a truncated result to the client.
            engine_terminal = True
            watchdog.pending()
            yield _engine_error_response(
                request_id,
                "generation stream ended before all outputs finished",
                result_handle,
            )
        except asyncio.CancelledError:
            raise
        # Broad by design: this is the RPC boundary, so any engine/processing
        # failure after acceptance must be reported to the client as an error
        # event and must still trigger abort + cleanup in the finally below.
        except Exception as error:
            logger.error(f"OpenEngine request {request_id} failed after acceptance: {error}")
            abort_request("response processing failed")
            engine_terminal = True
            watchdog.pending()
            yield _engine_error_response(request_id, str(error), result_handle)
        finally:
            watchdog.close()
            if not engine_terminal:
                abort_request("response stream closed")
            registration.release()


__all__ = ["OpenEngineInferenceServicer"]
