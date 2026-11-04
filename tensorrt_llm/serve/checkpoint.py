# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Authenticated engine lifecycle routes, independent of the host restore driver."""

import asyncio
import threading
import time
from collections.abc import Awaitable, Callable
from concurrent.futures import CancelledError
from typing import TYPE_CHECKING

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

from tensorrt_llm.executor.checkpoint import (
    AbortCheckpointRequest,
    ActivateRequest,
    CheckpointContext,
    CheckpointLifecycle,
    CheckpointProfile,
    CheckpointResult,
    CheckpointStep,
    GenerationValidation,
    PrepareCaptureRequest,
    RankCheckpointEvidence,
    ValidateRuntimeRequest,
)
from tensorrt_llm.executor.request import GenerationRequest
from tensorrt_llm.executor.rpc.rpc_common import RPCError
from tensorrt_llm.sampling_params import SamplingParams

if TYPE_CHECKING:
    from tensorrt_llm.llmapi.llm import LLM


class ExecutorCheckpointBackend:
    """Use existing worker RPC and normal generation for lifecycle validation."""

    def __init__(self, llm: "LLM") -> None:
        """Bind the single HTTP frontend's existing executor.

        Args:
            llm: Running MPI LLM with native checkpoint support enabled.
        """
        self._llm = llm
        self.ingress = CheckpointIngress()

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
            return future.result(timeout=timeout)
        except (RPCError, CancelledError) as error:
            raise RuntimeError("Checkpoint worker RPC is unavailable.") from error

    def describe(self) -> CheckpointProfile:
        """Read the native, fixed-cohort profile before accepting control traffic.

        Returns:
            The resolved engine profile.
        """
        return CheckpointProfile.model_validate(self._rpc("checkpoint_profile", (), 30))

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
        """
        deadline = time.monotonic() + remaining_seconds
        if step == CheckpointStep.PREPARE:
            # HTTP admission is already closed. Let accepted preprocessing and
            # streaming finish before the native queue rejects public submits.
            self.ingress.wait_drained(remaining_seconds)
        remaining_seconds = deadline - time.monotonic()
        if remaining_seconds <= 0:
            raise TimeoutError("Checkpoint budget expired before native dispatch.")
        results = self._rpc(
            "checkpoint_step",
            (
                step.value,
                context.model_dump(mode="json"),
                remaining_seconds,
                validation_token,
                source_session_id,
            ),
            remaining_seconds,
        )
        return [RankCheckpointEvidence.model_validate(item) for item in results]

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
        sampling = SamplingParams(
            max_tokens=len(validation.expected_token_ids),
            temperature=0,
            ignore_eos=True,
            end_id=-1,
            pad_id=0,
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

    def wait_drained(self, timeout: float) -> None:
        """Require all accepted public responses to finish within the budget.

        Args:
            timeout: Remaining capture operation time in seconds.

        Raises:
            TimeoutError: If a public handler or response remains active.
        """
        with self.condition:
            if not self.condition.wait_for(lambda: self.active == 0, timeout):
                raise TimeoutError("Public HTTP responses have not drained.")


class CheckpointAdmissionMiddleware:
    """Protect all public HTTP paths and retain their full ASGI lifetime."""

    def __init__(
        self, app: ASGIApp, lifecycle: CheckpointLifecycle, ingress: CheckpointIngress
    ) -> None:
        """Bind the lifecycle gate and streaming-response tracker.

        Args:
            app: Next application in the ASGI chain.
            lifecycle: Engine lifecycle authority for admission.
            ingress: Counter shared with capture preparation.
        """
        self.app, self.lifecycle, self.ingress = app, lifecycle, ingress

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Reject new public traffic while keeping authenticated controls reachable.

        Args:
            scope: ASGI connection metadata.
            receive: ASGI request input.
            send: ASGI response output.
        """
        path = scope.get("path", "")
        administrative = path.startswith("/checkpoint/") or path in ("/memory_status", "/metrics")
        if scope["type"] != "http" or administrative:
            await self.app(scope, receive, send)
            return
        with self.ingress.condition:
            admitted = self.lifecycle.get_lifecycle_status().readiness.admission_open
            if admitted:
                self.ingress.active += 1
        if not admitted:
            response = JSONResponse(
                status_code=503, content={"detail": "Checkpoint admission is closed."}
            )
            await response(scope, receive, send)
            return
        try:
            await self.app(scope, receive, send)
        finally:
            with self.ingress.condition:
                self.ingress.active -= 1
                self.ingress.condition.notify_all()


def register_checkpoint_routes(
    app: FastAPI,
    lifecycle: CheckpointLifecycle,
    authenticate: Callable[[Request], Awaitable[None]],
    ingress: CheckpointIngress,
) -> None:
    """Register authenticated operations and gate public traffic independently.

    Args:
        app: Single-frontend serving application.
        lifecycle: Persistent engine lifecycle owner.
        authenticate: Route/body-bound runtime-control authentication dependency.
        ingress: Full HTTP response tracker shared with native preparation.
    """
    dependencies = [Depends(authenticate)]

    app.add_middleware(CheckpointAdmissionMiddleware, lifecycle=lifecycle, ingress=ingress)

    async def invoke(operation: Callable, request: object) -> CheckpointResult:
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
            return await asyncio.to_thread(operation, request)
        except ValueError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        except (RuntimeError, TimeoutError) as error:
            raise HTTPException(status_code=409, detail=str(error)) from error

    @app.get("/checkpoint/status", dependencies=dependencies)
    async def status() -> dict:
        """Read current state without asserting serving readiness from memory.

        Returns:
            JSON-compatible lifecycle status and latest operation.
        """
        return lifecycle.get_lifecycle_status().model_dump(mode="json")

    @app.post("/checkpoint/prepare_capture", dependencies=dependencies)
    async def prepare_capture(request: PrepareCaptureRequest) -> CheckpointResult:
        """Naturally drain the source and protect its resident state.

        Args:
            request: Source identity, version and capture budget.

        Returns:
            Recorded capture operation result.
        """
        return await invoke(lifecycle.prepare_capture, request)

    @app.post("/checkpoint/validate_runtime", dependencies=dependencies)
    async def validate_runtime(request: ValidateRuntimeRequest) -> CheckpointResult:
        """Validate a fresh restore attempt without opening public admission.

        Args:
            request: Fresh destination identity and deterministic private probe.

        Returns:
            Recorded runtime validation result.
        """
        return await invoke(lifecycle.validate_runtime, request)

    @app.post("/checkpoint/activate", dependencies=dependencies)
    async def activate(request: ActivateRequest) -> CheckpointResult:
        """Require current runtime evidence and authenticated final host success.

        Args:
            request: Engine readiness reference and controller host attestation.

        Returns:
            Recorded all-rank activation result.
        """
        return await invoke(lifecycle.activate, request)

    @app.post("/checkpoint/abort_operation", dependencies=dependencies)
    async def abort_operation(request: AbortCheckpointRequest) -> CheckpointResult:
        """Stop pre-activation progress without authorizing resource reuse.

        Args:
            request: Current attempt identity and abort reason.

        Returns:
            Recorded abort result with host cleanup still unconfirmed.
        """
        return await invoke(lifecycle.abort_operation, request)
