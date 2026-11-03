# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Test engines for the OpenEngine client tests.

``StubEngine`` is a scripted OpenEngine server, used to drive the client into
any state, including ones a real engine does not produce. ``trtllm_engine``
runs TensorRT-LLM's own OpenEngine server adapter in front of a fake ``LLM``.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from types import SimpleNamespace
from typing import Any

import grpc

from tensorrt_llm.grpc.openengine.bindings import (
    error_pb2,
    generation_pb2,
    lifecycle_pb2,
    model_pb2,
    openengine_pb2_grpc,
    server_pb2,
)

WAIT_SECONDS = 5.0

GenerateHandler = Callable[
    [generation_pb2.GenerateRequest, grpc.aio.ServicerContext],
    AsyncIterator[generation_pb2.GenerateResponse],
]
ControlHandler = Callable[[Any, grpc.aio.ServicerContext], Awaitable[Any]]


# -- Response builders ------------------------------------------------------


def usage(prompt_tokens: int = 3, completion_tokens: int = 2) -> generation_pb2.Usage:
    return generation_pb2.Usage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
    )


def token(
    index: int | None, *token_ids: int, with_usage: bool = False
) -> generation_pb2.GenerateResponse:
    output = generation_pb2.TokenOutput(
        tokens=[generation_pb2.TokenInfo(token_id=token_id) for token_id in token_ids]
    )
    if index is not None:
        output.output_index = index
    response = generation_pb2.GenerateResponse(token=output)
    if with_usage:
        response.usage.CopyFrom(usage())
    return response


def finished(
    index: int | None,
    reason: int = generation_pb2.FINISH_REASON_LENGTH,
    *,
    with_usage: bool = True,
) -> generation_pb2.GenerateResponse:
    event = generation_pb2.GenerationFinished(reason=reason)
    if index is not None:
        event.output_index = index
    response = generation_pb2.GenerateResponse(finished=event)
    if with_usage:
        response.usage.CopyFrom(usage())
    return response


def prompt() -> generation_pb2.GenerateResponse:
    return generation_pb2.GenerateResponse(prompt=generation_pb2.PromptOutput())


def prefill_ready() -> generation_pb2.GenerateResponse:
    return generation_pb2.GenerateResponse(
        prefill_ready=generation_pb2.PrefillReady(), usage=usage()
    )


def engine_error(
    code: int = error_pb2.ERROR_CODE_OVERLOADED, *, retryable: bool = True
) -> generation_pb2.GenerateResponse:
    return generation_pb2.GenerateResponse(
        error=error_pb2.EngineError(code=code, message="engine gave up", retryable=retryable),
        usage=usage(),
    )


class Status:
    """Script step: end the RPC with a non-OK status."""

    def __init__(self, code: grpc.StatusCode, details: str = "scripted failure") -> None:
        self.code = code
        self.details = details


class Hang:
    """Script step: stop responding until the RPC is cancelled."""


def script(*steps: Any) -> GenerateHandler:
    """A Generate handler that plays ``steps`` in order, then ends with OK.

    A step is a ``GenerateResponse`` to send, a :class:`Status` to abort with,
    or :class:`Hang`.
    """

    async def handler(
        request: generation_pb2.GenerateRequest, context: grpc.aio.ServicerContext
    ) -> AsyncIterator[generation_pb2.GenerateResponse]:
        for step in steps:
            if isinstance(step, Status):
                await context.abort(step.code, step.details)
            elif step is Hang or isinstance(step, Hang):
                await asyncio.Event().wait()
            else:
                yield step

    return handler


# -- Scripted engine ----------------------------------------------------------


def _server_info(**overrides: Any) -> server_pb2.ServerInfo:
    fields: dict[str, Any] = {
        "engine_name": "stub",
        "instance_id": "stub-1",
        "schema_revision": 1,
        "minimum_client_revision": 1,
    }
    fields.update(overrides)
    return server_pb2.ServerInfo(**fields)


class _StubInference(openengine_pb2_grpc.InferenceServicer):
    def __init__(self, engine: StubEngine) -> None:
        self._engine = engine

    async def Generate(
        self, request: generation_pb2.GenerateRequest, context: grpc.aio.ServicerContext
    ) -> AsyncIterator[generation_pb2.GenerateResponse]:
        engine = self._engine
        engine.requests.append(request)
        try:
            async for response in engine.generate(request, context):
                yield response
        except asyncio.CancelledError:
            engine.cancelled.append(request.request_id)
            engine.cancel_seen.set()
            raise


class _StubControl(openengine_pb2_grpc.ControlServicer):
    def __init__(self, engine: StubEngine) -> None:
        self._engine = engine

    async def _handle(self, rpc: str, request: Any, context: grpc.aio.ServicerContext) -> Any:
        self._engine.control_calls.append((rpc, request))
        handler = self._engine.control.get(rpc)
        if handler is None:
            await context.abort(grpc.StatusCode.UNIMPLEMENTED, f"{rpc} is not scripted")
        return await handler(request, context)

    async def GetServerInfo(
        self, request: server_pb2.GetServerInfoRequest, context: grpc.aio.ServicerContext
    ) -> server_pb2.ServerInfo:
        return await self._handle("GetServerInfo", request, context)

    async def GetModelInfo(
        self, request: model_pb2.GetModelInfoRequest, context: grpc.aio.ServicerContext
    ) -> model_pb2.ModelInfo:
        return await self._handle("GetModelInfo", request, context)

    async def GetLoad(
        self, request: server_pb2.GetLoadRequest, context: grpc.aio.ServicerContext
    ) -> server_pb2.LoadInfo:
        return await self._handle("GetLoad", request, context)

    async def Health(
        self, request: lifecycle_pb2.HealthRequest, context: grpc.aio.ServicerContext
    ) -> lifecycle_pb2.HealthResponse:
        return await self._handle("Health", request, context)

    async def Abort(
        self, request: lifecycle_pb2.AbortRequest, context: grpc.aio.ServicerContext
    ) -> lifecycle_pb2.AbortResponse:
        return await self._handle("Abort", request, context)


class StubEngine:
    """A scripted OpenEngine server on a loopback port.

    Args:
        generate: Generate handler; see :func:`script`.
        health_states: Health states returned in order; the last one repeats.
        server_info: Field overrides for the ``ServerInfo`` it reports.
        control: Per-RPC Control handlers, replacing the defaults.
    """

    def __init__(
        self,
        generate: GenerateHandler | None = None,
        *,
        health_states: Sequence[int] = (lifecycle_pb2.HEALTH_STATE_READY,),
        server_info: dict[str, Any] | None = None,
        control: dict[str, ControlHandler] | None = None,
    ) -> None:
        self.generate = generate or script(finished(0))
        self.requests: list[generation_pb2.GenerateRequest] = []
        self.cancelled: list[str] = []
        self.cancel_seen = asyncio.Event()
        self.control_calls: list[tuple[str, Any]] = []
        self._health_states = list(health_states)
        info = _server_info(**(server_info or {}))

        async def get_server_info(
            request: server_pb2.GetServerInfoRequest, context: grpc.aio.ServicerContext
        ) -> server_pb2.ServerInfo:
            return info

        async def get_model_info(
            request: model_pb2.GetModelInfoRequest, context: grpc.aio.ServicerContext
        ) -> model_pb2.ModelInfo:
            return model_pb2.ModelInfo(model_id="stub-model", served_model_name="stub-model")

        async def health(
            request: lifecycle_pb2.HealthRequest, context: grpc.aio.ServicerContext
        ) -> lifecycle_pb2.HealthResponse:
            state = (
                self._health_states.pop(0)
                if len(self._health_states) > 1
                else self._health_states[0]
            )
            return lifecycle_pb2.HealthResponse(state=state)

        async def abort(
            request: lifecycle_pb2.AbortRequest, context: grpc.aio.ServicerContext
        ) -> lifecycle_pb2.AbortResponse:
            return lifecycle_pb2.AbortResponse(status=lifecycle_pb2.ABORT_STATUS_ABORTED)

        self.control: dict[str, ControlHandler] = {
            "GetServerInfo": get_server_info,
            "GetModelInfo": get_model_info,
            "Health": health,
            "Abort": abort,
        }
        self.control.update(control or {})
        self._server: grpc.aio.Server | None = None
        self.port = 0

    @property
    def target(self) -> str:
        return f"127.0.0.1:{self.port}"

    async def __aenter__(self) -> StubEngine:
        self._server = grpc.aio.server()
        openengine_pb2_grpc.add_InferenceServicer_to_server(_StubInference(self), self._server)
        openengine_pb2_grpc.add_ControlServicer_to_server(_StubControl(self), self._server)
        self.port = self._server.add_insecure_port("127.0.0.1:0")
        await self._server.start()
        return self

    async def __aexit__(self, *exc_info: Any) -> None:
        assert self._server is not None
        await self._server.stop(grace=0)


# -- TensorRT-LLM's OpenEngine server with a fake LLM -----------------------


class FakeResultHandle:
    """Stands in for the streaming ``GenerationResult`` of one request.

    Emits one cumulative result per entry of ``steps``; the last one finishes
    with ``finish_reason``. With ``hang_after=k`` it stops after ``k`` steps
    until aborted, then finishes the output as cancelled, as the real engine
    does.
    """

    def __init__(
        self,
        prompt_token_ids: Sequence[int],
        steps: Sequence[Sequence[int]],
        *,
        finish_reason: str = "length",
        hang_after: int | None = None,
    ) -> None:
        self._steps = [list(step) for step in steps]
        self._finish_reason = finish_reason
        self._hang_after = hang_after
        self._aborted_event = asyncio.Event()
        self.aborted = False
        self.finished = False
        self.prompt_token_ids = list(prompt_token_ids)
        self.outputs: list[Any] = []
        self.cached_tokens = 0
        self.error = None

    def abort(self) -> None:
        self.aborted = True
        self._aborted_event.set()

    def _result(self, token_ids: list[int], finish_reason: str | None) -> SimpleNamespace:
        output = SimpleNamespace(
            index=0,
            token_ids=list(token_ids),
            text="",
            logprobs=[],
            prompt_logprobs=[],
            finish_reason=finish_reason,
            stop_reason=None,
        )
        self.outputs = [output]
        self.finished = finish_reason is not None
        return SimpleNamespace(
            prompt_token_ids=self.prompt_token_ids,
            outputs=[output],
            cached_tokens=0,
            error=None,
            finished=self.finished,
        )

    async def __aiter__(self) -> AsyncIterator[Any]:
        token_ids: list[int] = []
        for number, step in enumerate(self._steps, start=1):
            if self.aborted:
                break
            token_ids.extend(step)
            last = number == len(self._steps)
            yield self._result(token_ids, self._finish_reason if last else None)
            if last:
                return
            if self._hang_after is not None and number >= self._hang_after:
                await self._aborted_event.wait()
        yield self._result(token_ids, "cancelled")


class _FakeTokenizer:
    def convert_ids_to_tokens(
        self, token_ids: Sequence[int], skip_special_tokens: bool = False
    ) -> list[str]:
        return [f"t{token_id}" for token_id in token_ids]

    def decode(self, token_ids: Sequence[int], skip_special_tokens: bool = False) -> str:
        return "".join(f"t{token_id}" for token_id in token_ids)


class FakeLlm:
    """The parts of ``LLM`` the OpenEngine server adapter reads.

    Every Generate gets a new :class:`FakeResultHandle` built from
    ``handle_kwargs``; the handles are kept in ``handles`` in arrival order.
    """

    def __init__(self, **handle_kwargs: Any) -> None:
        self.args = SimpleNamespace(
            max_seq_len=2048,
            max_batch_size=8,
            max_num_tokens=2048,
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            context_parallel_size=1,
            guided_decoding_backend=None,
            reasoning_parser=None,
            kv_cache_config=SimpleNamespace(tokens_per_block=32),
        )
        self.llm_id = "fake-instance"
        self.tokenizer = _FakeTokenizer()
        self.handles: list[FakeResultHandle] = []
        self.handle_created = asyncio.Event()
        self._handle_kwargs = {"steps": [[11], [12]], **handle_kwargs}

    def _check_health(self) -> bool:
        return True

    def generate_async(self, inputs: Any, **kwargs: Any) -> FakeResultHandle:
        prompt_token_ids = inputs.get("prompt_token_ids", []) if isinstance(inputs, dict) else []
        handle = FakeResultHandle(prompt_token_ids, **self._handle_kwargs)
        self.handles.append(handle)
        self.handle_created.set()
        return handle


@contextlib.asynccontextmanager
async def trtllm_engine(llm: FakeLlm) -> AsyncIterator[Any]:
    """Run TensorRT-LLM's OpenEngine server for ``llm``; yields the server (``.port``)."""
    from tensorrt_llm.grpc.openengine import OpenEngineServer

    server = OpenEngineServer(host="127.0.0.1", port=0, llm=llm, model="test-model")
    await server.start()
    try:
        yield server
    finally:
        await server.stop(grace=0)


async def wait_until(predicate: Callable[[], bool], timeout: float = WAIT_SECONDS) -> None:
    """Poll ``predicate`` until it holds; fail after ``timeout`` seconds."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while not predicate():
        if loop.time() >= deadline:
            raise AssertionError(f"condition not met within {timeout}s")
        await asyncio.sleep(0.01)
