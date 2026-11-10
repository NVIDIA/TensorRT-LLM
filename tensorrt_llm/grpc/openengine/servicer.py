# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""OpenEngine Inference service adapter for the TensorRT-LLM LLM API."""

import asyncio
import os
import threading
from collections.abc import AsyncIterator, Callable, Coroutine
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
from .capabilities import MediaCapabilities
from .coordinator import (
    CoordinationError,
    DuplicateRequestError,
    FrontendClient,
    InvalidRequestIdError,
    Reservation,
)
from .disagg import disaggregated_params_from_request
from .encoder import (
    EncoderCache,
    PrefillResultExports,
    encoder_role,
    release_encoder_result,
    validate_encoder,
)
from .errors import AbortFailedError, MediaUnavailableError, UnsupportedFeatureError
from .formatting import _engine_error_response, _stop_texts
from .metrics import OpenEngineMetrics, RequestMetrics
from .preparation import _decode_preparation, _prepared_decode_from_request
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

    def __init__(self, release: Callable[[Any], None] | None = None) -> None:
        self._release_result = release or self._release
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
        self._release_result(result)
        return None

    def transfer(self) -> None:
        with self._lock:
            self._result = None

    def abandon(self) -> None:
        with self._lock:
            self._abandoned = True
            result, self._result = self._result, None
        self._release_result(result)


class _PreparationLease:
    """Retain admission until cancelled executor work actually completes."""

    def __init__(self, admission: asyncio.Semaphore) -> None:
        self._admission = admission
        self._pending = 0
        self._closed = False
        self._released = False

    def track(self, future: asyncio.Future) -> None:
        self._pending += 1
        future.add_done_callback(self._completed)

    def _completed(self, future: asyncio.Future) -> None:
        if not future.cancelled():
            future.exception()
        self._pending -= 1
        self._release_if_idle()

    def close(self) -> None:
        self._closed = True
        self._release_if_idle()

    def _release_if_idle(self) -> None:
        if self._closed and not self._released and self._pending == 0:
            self._released = True
            self._admission.release()

    async def run(self, executor: Executor | None, fn: Callable, *args: Any) -> Any:
        future = asyncio.get_running_loop().run_in_executor(executor, copy_context().run, fn, *args)
        self.track(future)
        return await asyncio.shield(future)


class _LocalRequest:
    """Abort-aware local request state spanning mapping and generation."""

    def __init__(self) -> None:
        self.mapping_task: asyncio.Task[Any] | None = None
        self.engine_handle: Any = None
        self.aborted = False
        self.preprocessing: _PreprocessingWork | None = None
        self.decode_preparation: str | None = None

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
        max_media_preparations: int = 4,
    ) -> None:
        if type(max_media_preparations) is not int or max_media_preparations < 1:
            raise ValueError("max_media_preparations must be a positive integer")
        self._frontend = frontend
        self._input_executor = input_executor
        self._media_executor = media_executor
        self._media_admission = asyncio.Semaphore(max_media_preparations)
        self._llm = llm
        cache_directory = os.environ.get("TRTLLM_OPENENGINE_ENCODER_CACHE_DIR")
        self._encoder_cache = EncoderCache(cache_directory) if cache_directory else None
        encoder_transport = os.environ.get("TRTLLM_OPENENGINE_ENCODER_TRANSFER", "")
        if encoder_transport not in ("", "nixl"):
            raise ValueError("unsupported encoder transfer backend")
        if encoder_transport == "nixl":
            from .encoder_nixl import NixlEncoderCache

            self._encoder_cache = NixlEncoderCache(
                os.environ.get("TRTLLM_OPENENGINE_ENCODER_ENDPOINT", ""),
                os.environ.get("TRTLLM_OPENENGINE_ENCODER_MODEL_REVISION", ""),
            )
        self.encoder_role = encoder_role(llm, self._encoder_cache)
        if encoder_transport == "nixl" and (
            not self.encoder_role
            or (self.encoder_role == "encode" and not self._encoder_cache.endpoint)
        ):
            self._encoder_cache.close()
            raise ValueError(
                "NIXL encoder serving requires a supported role and a producer endpoint"
            )
        if getattr(llm.args, "mm_encoder_only", False) and self.encoder_role != "encode":
            raise ValueError(
                "encoder-only serving requires a shared cache, native MM disaggregation, and TP1"
            )
        self._model = model
        # Informational label placed in the KvSessionRef of PrefillReady events so
        # a generation worker/orchestrator can see which KV transfer backend the
        # context worker uses. The actual transfer is driven by opaque_state.
        self._kv_transfer_backend = kv_transfer_backend
        self._guided_backend = llm.args.guided_decoding_backend
        capabilities = media_capabilities_for_model(llm, encoder_handoff=bool(self.encoder_role))
        self._media_capabilities = MediaCapabilities(
            {
                name: template
                for name, template in capabilities.placeholder_templates.items()
                if not self.encoder_role or name == "image"
            }
        )
        self.subagent_affinity_auth_key = subagent_affinity_auth_key
        self._active_requests: dict[str, Any] = {}
        self._metrics = metrics
        self._encoder_cleanup_tasks: set[asyncio.Task] = set()
        self._encoder_cleanup_failed = False

    @property
    def supported_media_modalities(self) -> frozenset[str]:
        """Cached media modalities."""
        return self._media_capabilities.modalities

    def active_request_count(self) -> int:
        """Requests accepted and not yet completed."""
        return len(self._active_requests)

    async def _run_encoder_cleanup(self, operation: Coroutine[Any, Any, None]) -> None:
        cleanup = asyncio.create_task(operation)
        self._encoder_cleanup_tasks.add(cleanup)
        cleanup.add_done_callback(self._encoder_cleanup_tasks.discard)
        await asyncio.shield(cleanup)

    async def drain_encoder_cleanup(self) -> None:
        """Finish detached encoder cleanup before stopping the engine executors."""
        while self._encoder_cleanup_tasks:
            await asyncio.gather(*tuple(self._encoder_cleanup_tasks), return_exceptions=True)
        close = getattr(self._encoder_cache, "close", None)
        if close is not None:
            await asyncio.to_thread(close)

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
            if self._encoder_cleanup_failed:
                await context.abort(
                    grpc.StatusCode.UNAVAILABLE, "encoder cleanup failed; restart the engine"
                )
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

            encoder_action = validate_encoder(request, self.encoder_role)
            prepared_input = _prepare_input_from_request(
                request, self._llm, self._media_capabilities
            )
            sampling_params = sampling_params_from_request(request, self._guided_backend)
            trace_headers, target_dp_rank, affinity_headers = _metadata_from_context(context)
            conversation_params = conversation_params_from_request(request)
            disaggregated_params = disaggregated_params_from_request(request)
            if encoder_action and disaggregated_params is not None:
                if encoder_action != "consume" or disaggregated_params.request_type not in (
                    "context_only",
                    "context_and_generation",
                ):
                    raise UnsupportedFeatureError(
                        "encoder handoffs require a prefill or aggregated request"
                    )
                if disaggregated_params.request_type == "context_only" and not callable(
                    getattr(self._llm.input_processor, "capture_decode_state", None)
                ):
                    raise UnsupportedFeatureError(
                        "encoder prefill requires portable decode preparation"
                    )
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

            def prepare_with_decode_state(prepare: Callable[[], Any]) -> Any:
                capture_decode_state = getattr(
                    self._llm.input_processor, "capture_decode_state", None
                )
                if (
                    disaggregated_params is not None
                    and disaggregated_params.request_type == "context_only"
                    and callable(capture_decode_state)
                ):
                    with capture_decode_state() as capture:
                        result = prepare()
                    try:
                        local_request.decode_preparation = _decode_preparation(
                            request, self._llm, capture, result
                        )
                        if encoder_action == "consume" and local_request.decode_preparation is None:
                            raise UnsupportedFeatureError(
                                "encoder prefill requires portable decode preparation"
                            )
                    except Exception:
                        _PreprocessingWork._release(result)
                        raise
                    return result
                return prepare()

            async def prepare_inputs() -> Any:
                nonlocal disaggregated_params
                if not prepared_input.media:
                    return await _load_prepared_input(prepared_input, self._llm)
                await self._media_admission.acquire()
                lease = _PreparationLease(self._media_admission)
                try:
                    if encoder_action == "consume":
                        work = local_request.preprocessing = _PreprocessingWork()

                        def prepare_encoder_input() -> Any:
                            nonlocal disaggregated_params
                            inputs, disaggregated_params = self._encoder_cache.prepare(
                                request.encoder.handoff,
                                request,
                                self._llm,
                                sampling_params,
                                disaggregated_params,
                            )
                            return inputs

                        return await lease.run(
                            self._input_executor,
                            work.run,
                            partial(prepare_with_decode_state, prepare_encoder_input),
                        )
                    if (
                        disaggregated_params is not None
                        and disaggregated_params.request_type == "generation_only"
                    ):
                        work = local_request.preprocessing = _PreprocessingWork()
                        reused = await lease.run(
                            self._input_executor,
                            work.run,
                            partial(_prepared_decode_from_request, request, self._llm),
                        )
                        if reused is not None:
                            return reused
                    supports_token_id_mm_expansion = getattr(
                        self._llm.input_processor, "supports_token_id_mm_expansion", False
                    )
                    validated_ids = await lease.run(
                        self._input_executor,
                        partial(
                            _validate_media_placeholders,
                            add_special_tokens=(
                                sampling_params.add_special_tokens
                                and supports_token_id_mm_expansion
                            ),
                        ),
                        prepared_input,
                        self._llm,
                    )
                    with BaseMediaIO.use_executor(self._media_executor, on_submit=lease.track):
                        inputs = await _load_prepared_input(prepared_input, self._llm)
                    if (
                        validated_ids is not None
                        and isinstance(prepared_input.inputs, str)
                        and supports_token_id_mm_expansion
                    ):
                        inputs.pop("prompt")
                        inputs["prompt_token_ids"] = validated_ids
                    work = local_request.preprocessing = _PreprocessingWork()

                    return await lease.run(
                        self._input_executor,
                        work.run,
                        partial(
                            prepare_with_decode_state,
                            partial(
                                self._llm.preprocess, inputs, sampling_params, disaggregated_params
                            ),
                        ),
                    )
                finally:
                    lease.close()

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
            decode_preparation=local_request.decode_preparation,
        )
        prefill_exports = (
            PrefillResultExports(disaggregated_params) if format_state.is_context_only else None
        )
        prefill_may_export = format_state.is_context_only and (
            bool(prepared_input.media)
            or encoder_action == "consume"
            or getattr(inputs, "multimodal_params", None) is not None
        )

        async def release_prefill_result(result: Any) -> None:
            if prefill_exports is not None and prefill_exports.has_exports(result):
                if not await asyncio.to_thread(prefill_exports.release, result):
                    self._encoder_cleanup_failed = True

        if local_request.aborted or (reservation is not None and reservation.aborted):
            await context.abort(grpc.StatusCode.CANCELLED, "Request aborted before submission")

        try:
            if encoder_action == "encode":
                result_handle = self._llm.generate_async(
                    inputs=inputs, sampling_params=sampling_params
                )
            else:
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
                if prefill_exports is not None and prefill_exports.has_exports(result):
                    await self._run_encoder_cleanup(release_prefill_result(result))
                if context.cancelled():
                    if encoder_action == "encode":
                        await asyncio.to_thread(release_encoder_result, result)
                    abort_request("RPC cancelled")
                    return
                if encoder_action == "encode" and not result.error:
                    responses = []
                    if result.finished:
                        await self._media_admission.acquire()
                        lease = _PreparationLease(self._media_admission)

                        def discard_publication(reference: Any) -> None:
                            if reference is None:
                                return
                            try:
                                self._encoder_cache.discard(reference)
                            except Exception as error:
                                self._encoder_cleanup_failed = True
                                logger.error(f"Failed to reclaim abandoned encoder cache: {error}")

                        publication = _PreprocessingWork(discard_publication)
                        try:
                            cache = await lease.run(
                                self._input_executor,
                                publication.run,
                                partial(self._encoder_cache.publish, result, request, self._llm),
                            )
                            if context.cancelled():
                                return
                            responses.append(
                                generation_pb2.GenerateResponse(
                                    request_id=request_id,
                                    encode_ready=generation_pb2.EncodeReady(handoff=cache),
                                )
                            )
                            publication.transfer()
                        finally:
                            publication.abandon()
                            lease.close()
                else:
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
            if encoder_action == "encode" or (
                prefill_exports is not None
                and (
                    (abort_requested and prefill_may_export)
                    or prefill_exports.has_exports(result_handle)
                )
            ):

                async def reclaim_encoder(handle: Any) -> None:
                    try:
                        complete = getattr(handle, "aresult", None)
                        if (
                            abort_requested
                            and complete is not None
                            and (encoder_action == "encode" or prefill_may_export)
                        ):
                            await asyncio.wait_for(complete(), timeout=30)
                    except asyncio.TimeoutError:
                        self._encoder_cleanup_failed = True
                        logger.error("Native export cleanup timed out; refusing further requests")
                    except Exception:
                        # Native abort completion may raise after updating the result.
                        pass
                    finally:
                        if prefill_exports is None:
                            await asyncio.to_thread(release_encoder_result, handle)
                        else:
                            await release_prefill_result(handle)

                await self._run_encoder_cleanup(reclaim_encoder(result_handle))
            registration.release()
            # A task callback already scheduled on the event loop can still
            # hold this cell. abort_request is a no-op after this cleanup.
            result_handle = None


__all__ = ["OpenEngineInferenceServicer"]
