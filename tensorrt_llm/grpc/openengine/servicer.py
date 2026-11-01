# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""OpenEngine Inference service adapter for the TensorRT-LLM LLM API."""

import asyncio
import threading
from collections.abc import AsyncIterator, Callable
from concurrent.futures import Executor
from contextlib import aclosing
from contextvars import copy_context
from functools import partial
from typing import Any

import grpc

from tensorrt_llm.executor.request import DEFAULT_REQUEST_PRIORITY
from tensorrt_llm.inputs.media_io import BaseMediaIO
from tensorrt_llm.llmapi.llm import LLM
from tensorrt_llm.logger import logger
from tensorrt_llm.scheduling_params import SchedulingParams

from .affinity import validate_parent_affinity
from .bindings import error_pb2, generation_pb2, openengine_pb2_grpc
from .coordinator import (
    CoordinationError,
    DuplicateRequestError,
    FrontendClient,
    InvalidRequestIdError,
    Reservation,
)
from .disagg import disaggregated_params_from_request
from .errors import AbortFailedError, MediaUnavailableError, UnsupportedFeatureError
from .formatting import _engine_error_response, _stop_texts
from .metrics import OpenEngineMetrics, RequestMetrics
from .request_mapping import (
    _load_prepared_input,
    _metadata_from_context,
    _prepare_input_from_request,
    _validate_media_placeholders,
    conversation_affinity_enabled,
    conversation_params_from_request,
    media_capabilities_for_model,
    sampling_params_from_request,
)
from .streaming import (
    RESPONSE_STALL_TIMEOUT_SECONDS,
    ActiveRequest,
    StallWatchdog,
    _format_result,
    _OpenEngineFormatState,
)


class _PreprocessingWork:
    """Own exports until submission, including results from abandoned worker threads."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._abandoned = False
        self._result = None

    @staticmethod
    def _release(result: Any) -> None:
        params = getattr(result, "multimodal_params", None)
        if params is not None:
            try:
                params.release_handles()
            except Exception as error:
                logger.error(f"Failed to reclaim abandoned preprocessing exports: {error}")

    def run(self, preprocess: Callable[[], Any]) -> Any:
        result = preprocess()
        with self._lock:
            if not self._abandoned:
                self._result = result
                return result
        self._release(result)
        return None

    def transfer(self) -> None:
        with self._lock:
            self._result = None

    def abandon(self) -> None:
        with self._lock:
            self._abandoned = True
            result, self._result = self._result, None
        self._release(result)


class _LocalRequest:
    """Abort-aware local request state spanning mapping and generation."""

    def __init__(self) -> None:
        self.mapping_task: asyncio.Task[Any] | None = None
        self.engine_handle: Any = None
        self.aborted = False
        self.preprocessing: _PreprocessingWork | None = None

    def bind_mapping(self, task: asyncio.Task[Any]) -> None:
        self.mapping_task = task
        if self.aborted:
            task.cancel()

    def clear_mapping(self, task: asyncio.Task[Any]) -> None:
        if self.mapping_task is task:
            self.mapping_task = None

    def bind_engine(self, handle: Any) -> None:
        self.engine_handle = handle
        if self.preprocessing is not None:
            self.preprocessing.transfer()
        if self.aborted:
            handle.abort()

    def abort(self) -> None:
        if self.engine_handle is not None:
            self.engine_handle.abort()
        if self.mapping_task is not None:
            self.mapping_task.cancel()
        self.aborted = True


class OpenEngineInferenceServicer(openengine_pb2_grpc.InferenceServicer):
    """Translate OpenEngine generation streams to TensorRT-LLM requests."""

    def __init__(
        self,
        llm: LLM,
        model: str,
        kv_transfer_backend: str = "",
        frontend: FrontendClient | None = None,
        subagent_affinity_auth_key: str | None = None,
        metrics: OpenEngineMetrics | None = None,
        input_executor: Executor | None = None,
        media_executor: Executor | None = None,
    ) -> None:
        self._frontend = frontend
        self._input_executor = input_executor
        self._media_executor = media_executor
        self._llm = llm
        self._model = model
        # Informational label placed in the KvSessionRef of PrefillReady events so
        # a generation worker/orchestrator can see which KV transfer backend the
        # context worker uses. The actual transfer is driven by opaque_state.
        self._kv_transfer_backend = kv_transfer_backend
        self._guided_backend = llm.args.guided_decoding_backend
        self._media_capabilities = media_capabilities_for_model(llm)
        self.subagent_affinity_auth_key = subagent_affinity_auth_key
        self._active_requests: dict[str, Any] = {}
        self._metrics = metrics

    @property
    def supported_media_modalities(self) -> frozenset[str]:
        """Cached media modalities."""
        return self._media_capabilities.modalities

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
        local_request = None
        registration = None
        reservation = None
        request_metrics = RequestMetrics(self._metrics) if self._metrics else None
        try:
            if not request.request_id:
                await context.abort(
                    grpc.StatusCode.INVALID_ARGUMENT, "request_id must be non-empty"
                )
            local_request = _LocalRequest()
            registration = ActiveRequest(self._active_requests, request.request_id, local_request)
            if not registration.register():
                await context.abort(
                    grpc.StatusCode.ALREADY_EXISTS,
                    f"request_id '{request.request_id}' is already active",
                )
            if self._frontend is not None:
                reservation = await self._frontend.reserve(request.request_id)
                reservation.bind(local_request)
            async with aclosing(
                self._generate(
                    request, context, local_request, registration, reservation, request_metrics
                )
            ) as responses:
                async for response in responses:
                    yield response
        except DuplicateRequestError as error:
            await context.abort(grpc.StatusCode.ALREADY_EXISTS, str(error))
        except InvalidRequestIdError as error:
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(error))
        except asyncio.CancelledError as error:
            if request_metrics is not None:
                request_metrics.observe_exception(error)
            raise
        except CoordinationError as error:
            if request_metrics is not None:
                request_metrics.observe_exception(error)
            await context.abort(grpc.StatusCode.UNAVAILABLE, str(error))
        finally:
            try:
                if local_request is not None and local_request.preprocessing is not None:
                    local_request.preprocessing.abandon()
                if registration is not None:
                    registration.release()
                if reservation is not None:
                    await asyncio.shield(reservation.release())
            except asyncio.CancelledError as error:
                if request_metrics is not None:
                    request_metrics.observe_exception(error)
                raise
            except CoordinationError as error:
                # Cleanup still fails the serving group through FrontendClient.
                # Preserve cancellation or an already selected RPC error.
                if not context.cancelled() and context.code() in (None, grpc.StatusCode.OK):
                    if request_metrics is not None:
                        request_metrics.observe_exception(error)
                    await context.abort(grpc.StatusCode.UNAVAILABLE, str(error))
            finally:
                if request_metrics is not None:
                    status = context.code()
                    remaining = context.time_remaining()
                    if (
                        status in (None, grpc.StatusCode.OK)
                        and remaining is not None
                        and remaining <= 0
                    ):
                        status = grpc.StatusCode.DEADLINE_EXCEEDED
                    request_metrics.finish(status)

    async def _generate(
        self,
        request: generation_pb2.GenerateRequest,
        context: grpc.aio.ServicerContext,
        local_request: _LocalRequest,
        registration: ActiveRequest,
        reservation: Reservation | None = None,
        request_metrics: RequestMetrics | None = None,
    ) -> AsyncIterator[generation_pb2.GenerateResponse]:
        """Run a server-streaming OpenEngine generation request."""
        request_id = request.request_id
        try:
            if not request.model:
                raise ValueError("model must be non-empty")
            # Like the HTTP /v1/completions single-model server, any non-empty
            # model name is accepted and served by the loaded model (no NOT_FOUND
            # on a mismatch) so the two transports behave consistently.
            if request.lora_name:
                raise UnsupportedFeatureError("LoRA selection is not supported")
            if request.HasField("kv"):
                if request.kv.HasField("bypass_prefix_cache") and request.kv.bypass_prefix_cache:
                    raise UnsupportedFeatureError("prefix cache bypass is not supported")

            prepared_input = _prepare_input_from_request(
                request, self._llm, self._media_capabilities
            )
            sampling_params = sampling_params_from_request(request, self._guided_backend)
            trace_headers, target_dp_rank, affinity_headers = _metadata_from_context(context)
            conversation_params = conversation_params_from_request(request)
            disaggregated_params = disaggregated_params_from_request(request)
            if request_metrics is not None:
                request_metrics.begin(disaggregated_params)
            parent_affinity_id = validate_parent_affinity(
                self.subagent_affinity_auth_key,
                request,
                None if conversation_params is None else conversation_params.conversation_id,
                None if disaggregated_params is None else disaggregated_params.request_type,
                affinity_headers,
            )
            if parent_affinity_id is not None and not conversation_affinity_enabled(self._llm):
                raise UnsupportedFeatureError(
                    "Subagent affinity requires conversation-aware attention DP"
                )
            if conversation_affinity_enabled(self._llm):
                if target_dp_rank is not None and (
                    conversation_params is None or not conversation_params.conversation_id.strip()
                ):
                    raise ValueError(
                        "A stable conversation ID is required with conversation affinity and a DP-rank hint"
                    )
            if target_dp_rank is not None:
                if sampling_params.n > 1:
                    raise ValueError(
                        "openengine-target-dp-rank does not support multiple output sequences"
                    )
                args = getattr(self._llm, "args", None)
                dp_size = (
                    max(1, int(getattr(args, "tensor_parallel_size", 1) or 1))
                    if getattr(args, "enable_attention_dp", False)
                    else 1
                )
                if target_dp_rank >= dp_size:
                    raise ValueError(
                        f"openengine-target-dp-rank must be in [0, {dp_size}), got {target_dp_rank}"
                    )
            scheduling_params = (
                SchedulingParams(
                    attention_dp_rank=target_dp_rank,
                    attention_dp_relax=target_dp_rank is None,
                    subagent_affinity_id=parent_affinity_id,
                )
                if target_dp_rank is not None or parent_affinity_id is not None
                else None
            )
            cache_salt = (
                request.kv.cache_salt
                if request.HasField("kv") and request.kv.HasField("cache_salt")
                else None
            )

            async def prepare_inputs() -> Any:
                loop = asyncio.get_running_loop()
                if prepared_input.media:
                    await loop.run_in_executor(
                        self._input_executor,
                        copy_context().run,
                        _validate_media_placeholders,
                        prepared_input,
                        self._llm,
                    )
                with BaseMediaIO.use_executor(self._media_executor):
                    inputs = await _load_prepared_input(prepared_input, self._llm)
                if prepared_input.media:
                    work = local_request.preprocessing = _PreprocessingWork()
                    return await loop.run_in_executor(
                        self._input_executor,
                        copy_context().run,
                        work.run,
                        partial(
                            self._llm.preprocess, inputs, sampling_params, disaggregated_params
                        ),
                    )
                return inputs

            mapping_task = asyncio.create_task(prepare_inputs())
            local_request.bind_mapping(mapping_task)
            try:
                # wait() reports child cancellation as completion; RPC cancellation
                # raises here instead, so the two paths remain distinguishable.
                await asyncio.wait((mapping_task,))
                if mapping_task.cancelled() and local_request.aborted:
                    await context.abort(
                        grpc.StatusCode.CANCELLED, "Request aborted before submission"
                    )
                    return
                try:
                    inputs = mapping_task.result()
                except (UnsupportedFeatureError, MediaUnavailableError, ValueError):
                    raise
                except Exception as error:
                    # Unexpected preparation failures are worker errors, not bad inputs.
                    if request_metrics is not None:
                        request_metrics.engine_failed()
                    logger.error(f"OpenEngine request {request_id} could not be prepared: {error}")
                    await context.abort(grpc.StatusCode.INTERNAL, str(error))
                    return
            except asyncio.CancelledError:
                mapping_task.cancel()
                await asyncio.gather(mapping_task, return_exceptions=True)
                raise
            finally:
                local_request.clear_mapping(mapping_task)
        except UnsupportedFeatureError as error:
            await context.abort(grpc.StatusCode.UNIMPLEMENTED, str(error))
            return
        except MediaUnavailableError as error:
            await context.abort(grpc.StatusCode.FAILED_PRECONDITION, str(error))
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

        if local_request.aborted or (reservation is not None and reservation.aborted):
            await context.abort(grpc.StatusCode.CANCELLED, "Request aborted before submission")

        try:
            result_handle = self._llm.generate_async(
                inputs=inputs,
                sampling_params=sampling_params,
                streaming=True,
                trace_headers=trace_headers,
                scheduling_params=scheduling_params,
                conversation_params=conversation_params,
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
            if request_metrics is not None:
                request_metrics.engine_failed()
            logger.error(f"OpenEngine request {request_id} could not be submitted: {error}")
            await context.abort(grpc.StatusCode.INTERNAL, str(error))
            return
        # Broad by design: request submission can fail in engine-specific ways we
        # cannot enumerate here; any such failure must become a clean INTERNAL
        # status for the client rather than an unhandled servicer crash.
        except Exception as error:
            if request_metrics is not None:
                request_metrics.engine_failed()
            logger.error(f"Failed to submit OpenEngine request {request_id}: {error}")
            await context.abort(grpc.StatusCode.INTERNAL, str(error))
            return

        local_request.bind_engine(result_handle)
        if request_metrics is not None:
            request_metrics.accepted()

        engine_terminal = False
        abort_requested = False

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
            if request_metrics is not None:
                request_metrics.stalled()
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
            if reservation is not None:
                reservation.release()

        watchdog = StallWatchdog(
            asyncio.get_running_loop(), RESPONSE_STALL_TIMEOUT_SECONDS, on_stall
        )

        # Task callbacks are released on completion. ServicerContext callbacks
        # retain the context in a reference cycle even after the RPC finishes.
        rpc_task = asyncio.current_task()
        assert rpc_task is not None

        def on_rpc_done(_: asyncio.Task) -> None:
            abort_request("RPC completed before generation")

        rpc_task.add_done_callback(on_rpc_done)

        try:
            watchdog.start()
            async for result in result_handle:
                if context.cancelled():
                    abort_request("RPC cancelled")
                    return
                responses = _format_result(result, format_state)
                for resp in responses or ():
                    if request_metrics is not None:
                        request_metrics.observe_response(resp)
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
                    if request_metrics is not None:
                        request_metrics.complete(
                            result, handoff_ready=bool(format_state.prefill_ready_sent)
                        )
                    engine_terminal = True
                    return

            # Defensive: the engine is expected to mark the request finished before
            # its result stream ends, so reaching here means it closed early (e.g. an
            # internal engine abort). Surface it as an error rather than silently
            # returning a truncated result to the client.
            engine_terminal = True
            response = _engine_error_response(
                request_id,
                "generation stream ended before all outputs finished",
                result_handle,
            )
            if request_metrics is not None:
                request_metrics.observe_response(response)
            watchdog.pending()
            yield response
        except asyncio.CancelledError:
            raise
        # Broad by design: this is the RPC boundary, so any engine/processing
        # failure after acceptance must be reported to the client as an error
        # event and must still trigger abort + cleanup in the finally below.
        except Exception as error:
            if request_metrics is not None:
                request_metrics.engine_failed()
            logger.error(f"OpenEngine request {request_id} failed after acceptance: {error}")
            abort_request("response processing failed")
            engine_terminal = True
            watchdog.pending()
            yield _engine_error_response(request_id, str(error), result_handle)
        finally:
            rpc_task.remove_done_callback(on_rpc_done)
            watchdog.close()
            if not engine_terminal:
                abort_request("response stream closed")
            registration.release()
            # A task callback already scheduled on the event loop can still
            # hold this cell. abort_request is a no-op after this cleanup.
            result_handle = None


__all__ = ["OpenEngineInferenceServicer"]
