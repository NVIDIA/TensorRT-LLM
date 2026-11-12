# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Authenticated engine lifecycle routes, independent of the host restore driver."""

import asyncio
import threading
import time
from collections.abc import Awaitable, Callable
from concurrent.futures import CancelledError, Future
from typing import TYPE_CHECKING, TypeVar

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response
from starlette.types import ASGIApp, Receive, Scope, Send

from tensorrt_llm.executor.checkpoint import (
    AbortCheckpointRequest,
    ActivateRequest,
    CheckpointContext,
    CheckpointLifecycle,
    CheckpointPrepareNotDispatchedError,
    CheckpointProfile,
    CheckpointResult,
    CheckpointStep,
    GenerationValidation,
    PrepareCaptureRequest,
    RankCheckpointEvidence,
    ValidateRuntimeRequest,
    checkpoint_probe_sampling_params,
)
from tensorrt_llm.executor.request import GenerationRequest
from tensorrt_llm.executor.rpc.rpc_common import RPCCancelled, RPCError
from tensorrt_llm.llmapi.utils import StrictBaseModel
from tensorrt_llm.logger import logger
from tensorrt_llm.serve.anthropic_adapter import anthropic_error_response
from tensorrt_llm.serve.openai_protocol import ErrorResponse

if TYPE_CHECKING:
    from tensorrt_llm.llmapi.llm import LLM

RequestModel = TypeVar("RequestModel", bound=StrictBaseModel)


class ExecutorCheckpointBackend:
    """Use existing worker RPC and normal generation for lifecycle validation."""

    def __init__(self, llm: "LLM") -> None:
        """Bind the single HTTP frontend's existing executor.

        Args:
            llm: Running MPI LLM with native checkpoint support enabled.
        """
        self._llm = llm
        self.ingress = CheckpointIngress()
        self._profile: CheckpointProfile | None = None

    def _rpc(self, method: str, args: tuple, timeout: float) -> object:
        """Bound the caller's wait without treating a timeout as remote cancellation.

        Args:
            method: Explicitly supported worker method.
            args: Positional method arguments.
            timeout: Maximum wait in seconds.

        Returns:
            The rank-zero result, which can contain all-rank native evidence.

        Raises:
            RuntimeError: If the RPC transport closes or cancels the request.
            TimeoutError: If the caller's wait budget expires.
        """
        try:
            (future,) = self._llm._collective_rpc(method, args=args, non_block=True)
        except (RPCError, CancelledError) as error:
            raise RuntimeError("Checkpoint worker RPC is unavailable.") from error
        return self._await_rpc(future, timeout)

    def describe(self) -> CheckpointProfile:
        """Read the native, fixed-cohort profile before accepting control traffic.

        Returns:
            The resolved engine profile.
        """
        self._profile = CheckpointProfile.model_validate(self._rpc("checkpoint_profile", (), 30))
        return self._profile.model_copy(deep=True)

    @staticmethod
    def _await_rpc(future: Future, timeout: float) -> object:
        """Wait without interpreting a timeout as cancellation of native work."""
        try:
            return future.result(timeout=timeout)
        except (RPCCancelled, CancelledError) as error:
            raise RuntimeError("Checkpoint worker RPC is unavailable.") from error
        except RPCError as error:
            logger.error(f"Checkpoint worker operation failed: {error}")
            raise RuntimeError(
                "Checkpoint worker operation failed; inspect worker logs for details."
            ) from error

    def step(
        self,
        step: CheckpointStep,
        context: CheckpointContext,
        remaining_seconds: float,
        validation_token: str | None = None,
        source_session_id: str | None = None,
    ) -> list[RankCheckpointEvidence]:
        """Invoke one native all-rank operation within its remaining budget.

        Args:
            step: Native action to perform.
            context: Session-bound operation identity.
            remaining_seconds: Remaining controller budget.
            validation_token: Private one-use generation permit.
            source_session_id: Captured source identity for restore binding.

        Returns:
            Parsed participant evidence; the lifecycle validates completeness.

        Raises:
            CheckpointPrepareNotDispatchedError: If preparation times out before
                this synchronous caller can submit any native RPC.
            TimeoutError: If a dispatched RPC exceeds its wait budget.
        """
        deadline = time.monotonic() + remaining_seconds
        if step == CheckpointStep.PREPARE:
            # HTTP admission is already closed. Let accepted preprocessing and
            # streaming finish before the native queue rejects public submits.
            try:
                self.ingress.wait_drained(remaining_seconds)
            except TimeoutError as error:
                raise CheckpointPrepareNotDispatchedError(str(error)) from error
        remaining_seconds = deadline - time.monotonic()
        if remaining_seconds <= 0:
            if step == CheckpointStep.PREPARE:
                raise CheckpointPrepareNotDispatchedError(
                    "Checkpoint budget expired before native dispatch."
                )
            raise TimeoutError("Checkpoint budget expired before native dispatch.")
        args = (
            step.value,
            context.model_dump(mode="json"),
            remaining_seconds,
            validation_token,
            source_session_id,
        )
        if step == CheckpointStep.PREPARE:
            # Submission and cancellation share a lock; never hold it while waiting on RPC.
            with self.ingress.condition:
                if self.ingress.stopped:
                    raise CheckpointPrepareNotDispatchedError("Checkpoint preparation was stopped.")
                remaining_seconds = deadline - time.monotonic()
                if remaining_seconds <= 0:
                    raise CheckpointPrepareNotDispatchedError(
                        "Checkpoint budget expired before native dispatch."
                    )
                args = (args[0], args[1], remaining_seconds, *args[3:])
                try:
                    (future,) = self._llm._collective_rpc(
                        "checkpoint_step", args=args, non_block=True
                    )
                except (RPCError, CancelledError) as error:
                    raise RuntimeError("Checkpoint worker RPC is unavailable.") from error
            results = self._await_rpc(future, max(0, deadline - time.monotonic()))
        else:
            results = self._rpc("checkpoint_step", args, remaining_seconds)
        return [RankCheckpointEvidence.model_validate(item) for item in results]

    def validate_probe(self, validation: GenerationValidation) -> None:
        """Check the fixed profile before native restore or validation can start."""
        limits = self._profile.probe_limits if self._profile is not None else None
        if limits is None:
            raise RuntimeError("Checkpoint native profile does not provide probe limits.")
        vocab_size = limits.vocab_size
        if any(
            token >= vocab_size
            for token in validation.prompt_token_ids + validation.expected_token_ids
        ):
            raise ValueError("Checkpoint probe contains a token outside the model vocabulary.")
        prompt_length = len(validation.prompt_token_ids)
        if prompt_length > limits.max_input_len:
            raise ValueError("Checkpoint probe exceeds max_input_len.")
        if prompt_length + len(validation.expected_token_ids) > limits.max_seq_len:
            raise ValueError("Checkpoint probe exceeds max_seq_len.")
        if not limits.chunked_prefill and prompt_length > limits.max_num_tokens:
            raise ValueError("Checkpoint probe exceeds max_num_tokens without chunked prefill.")

    def generate(
        self, validation: GenerationValidation, token: str, remaining_seconds: float
    ) -> list[int]:
        """Run deterministic generation through normal submit and response handling.

        Args:
            validation: Tokenized prompt and the artifact's reference completion.
            token: Internal, one-use admission capability.
            remaining_seconds: Total generation budget.

        Returns:
            Actual generated token IDs, not a supplied readiness flag.
        """
        deadline = time.monotonic() + remaining_seconds
        sampling = self._llm._prepare_sampling_params(
            checkpoint_probe_sampling_params(len(validation.expected_token_ids))
        )
        request = GenerationRequest(validation.prompt_token_ids, sampling)
        request._checkpoint_validation_token = token
        result = self._llm._executor.submit(request)
        try:
            result.result(timeout=max(0, deadline - time.monotonic()))
            return list(result.outputs[0].token_ids)
        except TimeoutError:
            self._llm._executor.abort_request(result.request_id)
            raise

    def fail_closed(self, remaining_seconds: float = 0) -> None:
        """Request native ingress revocation without extending an expired operation.

        Args:
            remaining_seconds: Optional acknowledgement budget. Zero sends the
                request without waiting; HTTP admission remains closed either way.

        Raises:
            RuntimeError: If the transport cannot dispatch or acknowledge revocation.
            TimeoutError: If the acknowledgement budget expires.
        """
        with self.ingress.condition:
            self.ingress.stopped = True
            self.ingress.condition.notify_all()
        try:
            (future,) = self._llm._collective_rpc("checkpoint_fail_closed", non_block=True)
            if remaining_seconds > 0:
                future.result(timeout=remaining_seconds)
        except (RPCError, CancelledError) as error:
            raise RuntimeError("Checkpoint worker admission could not be revoked.") from error


class CheckpointIngress:
    """Track complete HTTP responses, including streaming and disconnect cleanup."""

    def __init__(self) -> None:
        """Start with no in-flight public handlers."""
        self.condition = threading.Condition()
        self.active = 0
        self.stopped = False

    def wait_drained(self, timeout: float) -> None:
        """Require all accepted public responses to finish within the budget.

        Args:
            timeout: Remaining capture operation time in seconds.

        Raises:
            TimeoutError: If a public handler or response remains active.
        """
        with self.condition:
            if not self.condition.wait_for(lambda: self.stopped or self.active == 0, timeout):
                raise TimeoutError("Public HTTP responses have not drained.")
            if self.stopped:
                raise CheckpointPrepareNotDispatchedError("Checkpoint preparation was stopped.")


class CheckpointAdmissionMiddleware:
    """Protect all public HTTP paths and retain their full ASGI lifetime."""

    def __init__(
        self,
        app: ASGIApp,
        lifecycle: CheckpointLifecycle,
        ingress: CheckpointIngress,
        passive_paths: frozenset[str],
        passive_prefixes: tuple[str, ...],
    ) -> None:
        """Bind the lifecycle gate and streaming-response tracker.

        Args:
            app: Next application in the ASGI chain.
            lifecycle: Engine lifecycle authority for admission.
            ingress: Counter shared with capture preparation.
            passive_paths: Server-owned GET/HEAD paths that need no admission.
            passive_prefixes: Mounted passive routes, matched at path boundaries.
        """
        self.app, self.lifecycle, self.ingress = app, lifecycle, ingress
        self.passive_paths = passive_paths
        self.passive_prefixes = passive_prefixes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Reject new public traffic while keeping authenticated controls reachable.

        Args:
            scope: ASGI connection metadata.
            receive: ASGI request input.
            send: ASGI response output.
        """
        path = scope.get("path", "")
        if scope["type"] == "http" and (
            path in ("/start_profile", "/stop_profile")
            or (path == "/v1/messages/batches" and scope["method"] == "POST")
        ):
            await self._error(path, "This operation is not supported in checkpoint mode.", 400)(
                scope, receive, send
            )
            return
        administrative = path.startswith("/checkpoint/") or (
            scope.get("method") in ("GET", "HEAD")
            and (
                path == "/ready"
                or path in self.passive_paths
                or any(
                    path == prefix or path.startswith(prefix + "/")
                    for prefix in self.passive_prefixes
                )
            )
        )
        if scope["type"] != "http" or administrative:
            await self.app(scope, receive, send)
            return
        with self.ingress.condition:
            admitted = self.lifecycle.admission_open()
            if admitted:
                self.ingress.active += 1
        if not admitted:
            response = self._error(path, "Checkpoint admission is closed.", 503)
            await response(scope, receive, send)
            return
        try:
            await self.app(scope, receive, send)
        finally:
            with self.ingress.condition:
                self.ingress.active -= 1
                self.ingress.condition.notify_all()

    @staticmethod
    def _error(path: str, message: str, status: int) -> JSONResponse:
        """Preserve the generation API's error envelope at the admission boundary."""
        if path.startswith("/v1/messages"):
            return anthropic_error_response(
                message, "overloaded_error" if status == 503 else "invalid_request_error", status
            )
        return JSONResponse(
            status_code=status,
            content=ErrorResponse(
                message=message,
                type="ServiceUnavailableError" if status == 503 else "BadRequestError",
                code=status,
            ).model_dump(),
        )


def register_checkpoint_routes(
    app: FastAPI,
    lifecycle: CheckpointLifecycle,
    authenticate: Callable[[Request], Awaitable[None]],
    ingress: CheckpointIngress,
    check_health: Callable[[], bool],
    passive_paths: frozenset[str],
    passive_prefixes: tuple[str, ...] = (),
) -> None:
    """Register authenticated operations and gate public traffic independently.

    Args:
        app: Single-frontend serving application.
        lifecycle: Persistent engine lifecycle owner.
        authenticate: Route/body-bound runtime-control authentication dependency.
        ingress: Full HTTP response tracker shared with native preparation.
        check_health: Existing engine liveness check, independent of admission.
        passive_paths: Read-only routes classified by the owning server.
        passive_prefixes: Read-only mounts classified by the owning server.
    """
    dependencies = [Depends(authenticate)]

    app.add_middleware(
        CheckpointAdmissionMiddleware,
        lifecycle=lifecycle,
        ingress=ingress,
        passive_paths=passive_paths,
        passive_prefixes=passive_prefixes,
    )

    async def invoke(
        operation: Callable[[RequestModel], CheckpointResult], request: RequestModel
    ) -> CheckpointResult | JSONResponse:
        """Run native work without blocking authenticated status requests.

        Args:
            operation: Validated lifecycle mutation to dispatch in a worker thread.
            request: Typed operation payload.

        Returns:
            Recorded lifecycle result, including terminal native failures.

        Raises:
            HTTPException: If identity, ordering or deadline rejects the mutation.
        """
        try:
            result = await asyncio.to_thread(operation, request)
        except ValueError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        except TimeoutError as error:
            raise HTTPException(status_code=504, detail=str(error)) from error
        except RuntimeError as error:
            raise HTTPException(status_code=500, detail=str(error)) from error
        if result.status == "failed":
            return JSONResponse(status_code=500, content=result.model_dump(mode="json"))
        return result

    @app.get("/ready")
    async def ready() -> Response:
        """Report traffic readiness without treating a parked engine as dead.

        Returns:
            HTTP 200 only while engine health and checkpoint admission both allow serving.
        """
        admitted = lifecycle.admission_open()
        return Response(status_code=200 if admitted and check_health() else 503)

    @app.get("/checkpoint/status", dependencies=dependencies)
    async def status() -> dict:
        """Read current state without asserting serving readiness from memory.

        Returns:
            JSON-compatible lifecycle status and latest operation.
        """
        return lifecycle.get_lifecycle_status().model_dump(mode="json")

    @app.post(
        "/checkpoint/prepare_capture", dependencies=dependencies, response_model=CheckpointResult
    )
    async def prepare_capture(request: PrepareCaptureRequest) -> CheckpointResult | JSONResponse:
        """Naturally drain the source and protect its resident state.

        Args:
            request: Source identity, version and capture budget.

        Returns:
            Recorded capture operation result.
        """
        return await invoke(lifecycle.prepare_capture, request)

    @app.post(
        "/checkpoint/validate_runtime", dependencies=dependencies, response_model=CheckpointResult
    )
    async def validate_runtime(request: ValidateRuntimeRequest) -> CheckpointResult | JSONResponse:
        """Validate a fresh restore attempt without opening public admission.

        Args:
            request: Fresh destination identity and deterministic private probe.

        Returns:
            Recorded runtime validation result.
        """
        return await invoke(lifecycle.validate_runtime, request)

    @app.post("/checkpoint/activate", dependencies=dependencies, response_model=CheckpointResult)
    async def activate(request: ActivateRequest) -> CheckpointResult | JSONResponse:
        """Require current runtime evidence and authenticated final host success.

        Args:
            request: Engine readiness reference and controller host attestation.

        Returns:
            Recorded all-rank activation result.
        """
        return await invoke(lifecycle.activate, request)

    @app.post(
        "/checkpoint/abort_operation", dependencies=dependencies, response_model=CheckpointResult
    )
    async def abort_operation(request: AbortCheckpointRequest) -> CheckpointResult | JSONResponse:
        """Stop pre-activation progress without authorizing resource reuse.

        Args:
            request: Current attempt identity and abort reason.

        Returns:
            Recorded abort result with host cleanup still unconfirmed.
        """
        return await invoke(lifecycle.abort_operation, request)
