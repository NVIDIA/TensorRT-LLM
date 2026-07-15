# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Serve-specific IPC access to a generator.

The protocol uses one ROUTER socket on the generator side and one DEALER
socket per client. Only explicit request-output snapshots cross the process
boundary; the generator's queues, executor weak references, and tokenizer do
not.
"""

from __future__ import annotations

import asyncio
import dataclasses
import inspect
import os
import queue
import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Optional

import zmq

from tensorrt_llm.executor.ipc import ZeroMqQueue

MessageType = Literal[
    "health",
    "heartbeat",
    "disconnect",
    "alias",
    "cancel_alias",
    "call",
    "generate",
    "cancel",
    "credit",
    "ready",
    "result",
    "stream_start",
    "stream_data",
    "stream_end",
    "cancelled",
    "error",
    "closed",
]


class GeneratorIpcError(RuntimeError):
    """Base error raised by the generator IPC gateway."""


class GeneratorIpcClosedError(GeneratorIpcError):
    """Raised when the remote generator gateway closes."""


class GeneratorIpcRemoteError(GeneratorIpcError):
    """Raised when an operation fails in the generator process."""


@dataclass(slots=True)
class IpcEnvelope:
    """A routed protocol message."""

    request_id: str
    message_type: MessageType
    sequence: int
    payload: Any = None


@dataclass(slots=True)
class CompletionOutputSnapshot:
    """Serializable public state of one completion output."""

    index: int
    text: str = ""
    token_ids: Optional[list[int]] = field(default_factory=list)
    cumulative_logprob: Optional[float] = None
    logprobs: Any = field(default_factory=list)
    prompt_logprobs: Any = field(default_factory=list)
    finish_reason: Optional[str] = None
    stop_reason: Optional[int | str] = None
    generation_logits: Any = None
    additional_context_outputs: Any = None
    additional_generation_outputs: Any = None
    disaggregated_params: Any = None
    request_perf_metrics: Any = None
    postprocess_result: Any = None
    text_diff: str = ""
    token_ids_diff: list[int] = field(default_factory=list)
    logprobs_diff: Any = field(default_factory=list)
    last_text_len: int = 0

    @classmethod
    def capture(cls, output: Any) -> "CompletionOutputSnapshot":
        """Capture fields used by serve response post-processing."""
        return cls(
            index=output.index,
            text=getattr(output, "text", ""),
            token_ids=getattr(output, "token_ids", None),
            cumulative_logprob=getattr(output, "cumulative_logprob", None),
            logprobs=getattr(output, "logprobs", None),
            prompt_logprobs=getattr(output, "prompt_logprobs", None),
            finish_reason=getattr(output, "finish_reason", None),
            stop_reason=getattr(output, "stop_reason", None),
            generation_logits=getattr(output, "generation_logits", None),
            additional_context_outputs=getattr(output, "additional_context_outputs", None),
            additional_generation_outputs=getattr(output, "additional_generation_outputs", None),
            disaggregated_params=getattr(output, "disaggregated_params", None),
            request_perf_metrics=getattr(output, "request_perf_metrics", None),
            postprocess_result=getattr(output, "_postprocess_result", None),
            text_diff=getattr(output, "text_diff", ""),
            token_ids_diff=list(getattr(output, "token_ids_diff", []) or []),
            logprobs_diff=getattr(output, "logprobs_diff", None),
            last_text_len=getattr(output, "_last_text_len", 0),
        )


@dataclass(slots=True)
class RequestOutputSnapshot:
    """Serializable state of a RequestOutput-like promise."""

    request_id: int | str
    prompt_token_ids: list[int]
    outputs: list[CompletionOutputSnapshot]
    finished: bool
    error: Optional[str]
    prompt: Optional[str] = None
    context_logits: Any = None
    disaggregated_params: Any = None
    metrics_dict: dict[str, Any] = field(default_factory=dict)
    candidate_metrics: list[dict[str, Any]] = field(default_factory=list)
    time_breakdown_metrics: Any = None
    cached_tokens: int = 0
    avg_decoded_tokens_per_iter: Optional[float] = None

    @classmethod
    def capture(cls, output: Any) -> "RequestOutputSnapshot":
        """Capture a RequestOutput without its process-local implementation state."""
        return cls(
            request_id=output.request_id,
            prompt_token_ids=list(output.prompt_token_ids),
            outputs=[CompletionOutputSnapshot.capture(item) for item in output.outputs],
            finished=bool(output.finished),
            error=getattr(output, "error", None),
            prompt=getattr(output, "prompt", None),
            context_logits=getattr(output, "context_logits", None),
            disaggregated_params=getattr(output, "disaggregated_params", None),
            metrics_dict=dict(getattr(output, "metrics_dict", {}) or {}),
            candidate_metrics=list(getattr(output, "candidate_metrics", []) or []),
            time_breakdown_metrics=getattr(output, "time_breakdown_metrics", None),
            cached_tokens=getattr(output, "cached_tokens", 0),
            avg_decoded_tokens_per_iter=getattr(output, "avg_decoded_tokens_per_iter", None),
        )


@dataclass(slots=True)
class VisualGenOutputSnapshot:
    """Metadata and shared-file location for a VisualGen output."""

    request_id: int
    payload_path: Optional[str]
    frame_rate: Optional[float]
    audio_sample_rate: Optional[int]
    error: Optional[str]
    metrics: Any


def _snapshot_visual_output(output: Any) -> VisualGenOutputSnapshot:
    import torch

    media = {name: getattr(output, name, None) for name in ("image", "video", "audio")}
    payload_path = None
    if any(value is not None for value in media.values()):
        storage_path = Path(os.getenv("TRTLLM_MEDIA_STORAGE_PATH", "/tmp/trtllm_generated"))  # nosec B108
        storage_path.mkdir(exist_ok=True, parents=True)
        payload_path = str(storage_path / f"ipc_{uuid.uuid4().hex}.pt")
        torch.save(
            {
                name: value.detach().to("cpu") if value is not None else None
                for name, value in media.items()
            },
            payload_path,
        )
    return VisualGenOutputSnapshot(
        request_id=getattr(output, "request_id", -1),
        payload_path=payload_path,
        frame_rate=getattr(output, "frame_rate", None),
        audio_sample_rate=getattr(output, "audio_sample_rate", None),
        error=getattr(output, "error", None),
        metrics=getattr(output, "metrics", None),
    )


def _restore_visual_output(snapshot: VisualGenOutputSnapshot) -> Any:
    import torch

    from tensorrt_llm.visual_gen.output import VisualGenOutput

    media = {"image": None, "video": None, "audio": None}
    if snapshot.payload_path is not None:
        try:
            media = torch.load(snapshot.payload_path, map_location="cpu", weights_only=True)
        finally:
            try:
                os.unlink(snapshot.payload_path)
            except FileNotFoundError:
                pass
    return VisualGenOutput(
        request_id=snapshot.request_id,
        frame_rate=snapshot.frame_rate,
        audio_sample_rate=snapshot.audio_sample_rate,
        error=snapshot.error,
        metrics=snapshot.metrics,
        **media,
    )


def _prepare_unary_payload(result: Any) -> Any:
    from tensorrt_llm.visual_gen.output import VisualGenOutput

    if isinstance(result, VisualGenOutput):
        return _snapshot_visual_output(result)
    return result


class RemoteCompletionOutput:
    """Lightweight CompletionOutput-compatible snapshot."""

    def __init__(self, snapshot: CompletionOutputSnapshot) -> None:
        for item in dataclasses.fields(snapshot):
            setattr(self, item.name, getattr(snapshot, item.name))
        self._postprocess_result = snapshot.postprocess_result
        self._last_text_len = snapshot.last_text_len

    @property
    def length(self) -> int:
        """Return the generated token count."""
        return len(self.token_ids or [])


class RemoteRequestOutput:
    """Async iterator backed by snapshots from a remote generator."""

    def __init__(
        self,
        client: "GeneratorIpcClient",
        transport_request_id: str,
        response_queue: queue.Queue[IpcEnvelope | BaseException],
        snapshot: RequestOutputSnapshot,
    ) -> None:
        self._client = client
        self._transport_request_id = transport_request_id
        self._response_queue = response_queue
        self._aborted = False
        self._terminal = False
        self._restore(snapshot)

    def _restore(self, snapshot: RequestOutputSnapshot) -> None:
        for item in dataclasses.fields(snapshot):
            if item.name != "outputs":
                setattr(self, item.name, getattr(snapshot, item.name))
        self.outputs = [RemoteCompletionOutput(item) for item in snapshot.outputs]
        self.id = self.request_id
        self._done = self.finished

    def abort(self) -> None:
        """Cancel generation in the generator process."""
        if self._terminal or self._aborted:
            return
        self._aborted = True
        self._client._cancel(self._transport_request_id)

    def aborted(self) -> bool:
        """Return whether cancellation was requested."""
        return self._aborted

    def __aiter__(self) -> "RemoteRequestOutput":
        return self

    async def __anext__(self) -> "RemoteRequestOutput":
        if self._terminal:
            raise StopAsyncIteration
        item = await asyncio.to_thread(self._response_queue.get)
        if isinstance(item, BaseException):
            self._terminal = True
            raise item
        if item.message_type == "stream_data":
            self._restore(item.payload)
            await asyncio.to_thread(self._client._credit, self._transport_request_id)
            return self
        if item.message_type in ("stream_end", "cancelled"):
            if isinstance(item.payload, RequestOutputSnapshot):
                self._restore(item.payload)
            self._terminal = True
            raise StopAsyncIteration
        if item.message_type == "error":
            self._terminal = True
            raise GeneratorIpcRemoteError(_format_remote_error(item.payload))
        if item.message_type == "closed":
            self._terminal = True
            raise GeneratorIpcClosedError(str(item.payload))
        self._terminal = True
        raise GeneratorIpcError(f"Unexpected stream message: {item.message_type}")

    async def aresult(self) -> "RemoteRequestOutput":
        """Consume the stream and return its final snapshot."""
        async for _ in self:
            pass
        return self

    def __await__(self):
        return self.aresult().__await__()


class RemoteUnaryFuture:
    """Awaitable result for non-LLM generator calls such as VisualGen."""

    def __init__(
        self,
        client: "GeneratorIpcClient",
        transport_request_id: str,
        response_queue: queue.Queue[IpcEnvelope | BaseException],
    ) -> None:
        self._client = client
        self._transport_request_id = transport_request_id
        self._response_queue = response_queue

    def abort(self) -> None:
        self._client._cancel(self._transport_request_id)

    async def aresult(self) -> Any:
        item = await asyncio.to_thread(self._response_queue.get)
        if isinstance(item, BaseException):
            raise item
        if item.message_type == "result":
            if isinstance(item.payload, VisualGenOutputSnapshot):
                return _restore_visual_output(item.payload)
            return item.payload
        if item.message_type == "error":
            raise GeneratorIpcRemoteError(_format_remote_error(item.payload))
        if item.message_type == "closed":
            raise GeneratorIpcClosedError(str(item.payload))
        raise GeneratorIpcError(f"Unexpected unary generation response: {item.message_type}")

    def __await__(self):
        return self.aresult().__await__()


def _format_remote_error(payload: Any) -> str:
    if isinstance(payload, dict):
        return str(payload.get("message", payload))
    return str(payload)


class GeneratorIpcServer:
    """Expose an arbitrary generator over a ROUTER IPC endpoint."""

    def __init__(
        self,
        generator: Any,
        endpoint: str,
        *,
        hmac_key: Optional[bytes] = None,
        worker_timeout: float = 10.0,
        queue_size: int = 32,
    ) -> None:
        if not endpoint.startswith("ipc://"):
            raise ValueError("Generator IPC endpoint must use ipc://")
        if queue_size <= 0:
            raise ValueError("queue_size must be positive")
        if worker_timeout <= 0:
            raise ValueError("worker_timeout must be positive")
        self._generator = generator
        self._endpoint = endpoint
        self._hmac_key = hmac_key
        self._worker_timeout = worker_timeout
        self._queue_size = queue_size
        self._queue: Optional[ZeroMqQueue] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._started = threading.Event()
        self._startup_error: Optional[BaseException] = None
        self._stop_event: Optional[asyncio.Event] = None
        self._identities: set[bytes] = set()
        self._last_seen: dict[bytes, float] = {}
        self._active: dict[tuple[bytes, str], Any] = {}
        self._credits: dict[tuple[bytes, str], asyncio.BoundedSemaphore] = {}
        self._pump_tasks: dict[tuple[bytes, str], asyncio.Task[Any]] = {}
        self._aliases: dict[str, tuple[bytes, str]] = {}
        self._tasks: set[asyncio.Task[Any]] = set()

    @property
    def address(self) -> tuple[str, bytes]:
        """Return the endpoint and generated HMAC key after start."""
        if self._queue is None or self._queue.hmac_key is None:
            raise RuntimeError("Generator IPC server is not started")
        return self._endpoint, self._queue.hmac_key

    def start(self, timeout: float = 10.0) -> None:
        """Start the server on a background event-loop thread."""
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._thread_main,
            name="generator-ipc-server",
            daemon=True,
        )
        self._thread.start()
        if not self._started.wait(timeout):
            raise TimeoutError("Timed out starting generator IPC server")
        if self._startup_error is not None:
            raise GeneratorIpcError("Failed to start generator IPC server") from self._startup_error

    def _thread_main(self) -> None:
        try:
            asyncio.run(self._serve())
        except BaseException as error:
            self._startup_error = error
            self._started.set()

    async def _serve(self) -> None:
        self._loop = asyncio.get_running_loop()
        self._stop_event = asyncio.Event()
        try:
            self._queue = ZeroMqQueue(
                address=(self._endpoint, self._hmac_key),
                is_server=True,
                is_async=True,
                socket_type=zmq.ROUTER,
                name="generator_ipc_server",
            )
            self._queue.socket.setsockopt(zmq.SNDHWM, self._queue_size)
            self._queue.setup_lazily()
            self._started.set()
            while not self._stop_event.is_set():
                try:
                    envelope, identity = await self._queue.get_async_noblock(
                        timeout=0.1, return_identity=True
                    )
                except asyncio.TimeoutError:
                    self._expire_workers()
                    continue
                self._identities.add(identity)
                self._last_seen[identity] = time.monotonic()
                task = asyncio.create_task(self._dispatch(identity, envelope))
                self._tasks.add(task)
                task.add_done_callback(self._tasks.discard)
        finally:
            await self._shutdown_async()

    async def _dispatch(self, identity: bytes, envelope: IpcEnvelope) -> None:
        if not isinstance(envelope, IpcEnvelope):
            return
        try:
            if envelope.message_type == "health":
                await self._handle_health(identity, envelope)
            elif envelope.message_type == "heartbeat":
                await self._send(
                    identity,
                    IpcEnvelope("*", "ready", 0, {"ready": True}),
                )
            elif envelope.message_type == "disconnect":
                self._abort_identity(identity)
            elif envelope.message_type == "alias":
                key = (identity, envelope.payload["target_request_id"])
                if key in self._active:
                    self._aliases[str(envelope.payload["alias"])] = key
                await self._send(
                    identity,
                    IpcEnvelope(envelope.request_id, "result", 0, key in self._active),
                )
            elif envelope.message_type == "cancel_alias":
                key = self._aliases.pop(str(envelope.payload), None)
                if key is not None and key in self._active:
                    self._active[key].abort()
                    credits = self._credits.get(key)
                    if credits is not None and credits.locked():
                        credits.release()
                await self._send(
                    identity,
                    IpcEnvelope(envelope.request_id, "result", 0, key is not None),
                )
            elif envelope.message_type == "call":
                await self._handle_call(identity, envelope)
            elif envelope.message_type == "generate":
                await self._handle_generate(identity, envelope)
            elif envelope.message_type == "cancel":
                await self._handle_cancel(identity, envelope)
            elif envelope.message_type == "credit":
                self._handle_credit(identity, envelope)
            else:
                raise ValueError(f"Unsupported request type: {envelope.message_type}")
        except asyncio.CancelledError:
            raise
        except Exception as error:
            await self._send_error(identity, envelope.request_id, envelope.sequence, error)

    def _expire_workers(self) -> None:
        now = time.monotonic()
        expired = [
            identity
            for identity, last_seen in self._last_seen.items()
            if now - last_seen > self._worker_timeout
        ]
        for identity in expired:
            self._abort_identity(identity)

    def _abort_identity(self, identity: bytes) -> None:
        for key, promise in list(self._active.items()):
            if key[0] == identity:
                promise.abort()
                task = self._pump_tasks.pop(key, None)
                if task is not None:
                    task.cancel()
                self._active.pop(key, None)
                self._credits.pop(key, None)
        self._identities.discard(identity)
        self._last_seen.pop(identity, None)
        for alias, key in list(self._aliases.items()):
            if key[0] == identity:
                self._aliases.pop(alias, None)

    def _remove_aliases(self, key: tuple[bytes, str]) -> None:
        for alias, aliased_key in list(self._aliases.items()):
            if aliased_key == key:
                self._aliases.pop(alias, None)

    async def _handle_health(self, identity: bytes, envelope: IpcEnvelope) -> None:
        status = {"ready": True, "fatal_error": None}
        health_status = getattr(self._generator, "health_status", None)
        if health_status is not None:
            status = await self._invoke(health_status)
        else:
            health_check = getattr(self._generator, "_check_health", None)
            if health_check is not None:
                status["ready"] = bool(await self._invoke(health_check))
        await self._send(
            identity,
            IpcEnvelope(
                envelope.request_id,
                "ready",
                0,
                status,
            ),
        )

    async def _handle_call(self, identity: bytes, envelope: IpcEnvelope) -> None:
        method_name = envelope.payload["method"]
        if method_name.startswith("_"):
            raise ValueError("Private methods cannot be called over generator IPC")
        method = getattr(self._generator, method_name)
        result = await self._invoke(
            method,
            *envelope.payload.get("args", ()),
            **envelope.payload.get("kwargs", {}),
        )
        result = await asyncio.to_thread(_prepare_unary_payload, result)
        await self._send(identity, IpcEnvelope(envelope.request_id, "result", 0, result))

    async def _handle_generate(self, identity: bytes, envelope: IpcEnvelope) -> None:
        promise = self._generator.generate_async(
            *envelope.payload.get("args", ()),
            **envelope.payload.get("kwargs", {}),
        )
        key = (identity, envelope.request_id)
        self._active[key] = promise
        self._credits[key] = asyncio.BoundedSemaphore(envelope.payload["stream_window"])
        is_request_output = (
            hasattr(promise, "request_id")
            and hasattr(promise, "prompt_token_ids")
            and hasattr(promise, "outputs")
        )
        await self._send(
            identity,
            IpcEnvelope(
                envelope.request_id,
                "stream_start",
                0,
                RequestOutputSnapshot.capture(promise) if is_request_output else None,
            ),
        )
        pump = self._pump if is_request_output else self._pump_unary
        task = asyncio.create_task(pump(identity, envelope.request_id, promise))
        self._pump_tasks[key] = task
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _pump(self, identity: bytes, request_id: str, promise: Any) -> None:
        sequence = 1
        key = (identity, request_id)
        try:
            if hasattr(promise, "__aiter__"):
                async for output in promise:
                    await self._credits[key].acquire()
                    await self._send(
                        identity,
                        IpcEnvelope(
                            request_id,
                            "stream_data",
                            sequence,
                            RequestOutputSnapshot.capture(output),
                        ),
                    )
                    sequence += 1
            else:
                result = await promise.aresult()
                await self._credits[key].acquire()
                await self._send(
                    identity,
                    IpcEnvelope(
                        request_id,
                        "stream_data",
                        sequence,
                        RequestOutputSnapshot.capture(result),
                    ),
                )
                sequence += 1
            await self._send(
                identity,
                IpcEnvelope(
                    request_id,
                    "stream_end",
                    sequence,
                    RequestOutputSnapshot.capture(promise),
                ),
            )
        except asyncio.CancelledError:
            raise
        except Exception as error:
            await self._send_error(identity, request_id, sequence, error)
        finally:
            self._active.pop(key, None)
            self._credits.pop(key, None)
            self._pump_tasks.pop(key, None)
            self._remove_aliases(key)

    async def _pump_unary(self, identity: bytes, request_id: str, promise: Any) -> None:
        key = (identity, request_id)
        try:
            result = (
                await promise
                if inspect.isawaitable(promise)
                else await asyncio.to_thread(promise.result)
            )
            result = await asyncio.to_thread(_prepare_unary_payload, result)
            await self._send(
                identity,
                IpcEnvelope(request_id, "result", 1, result),
            )
        except asyncio.CancelledError:
            raise
        except Exception as error:
            await self._send_error(identity, request_id, 1, error)
        finally:
            self._active.pop(key, None)
            self._credits.pop(key, None)
            self._pump_tasks.pop(key, None)
            self._remove_aliases(key)

    async def _handle_cancel(self, identity: bytes, envelope: IpcEnvelope) -> None:
        promise = self._active.get((identity, envelope.request_id))
        if promise is not None:
            promise.abort()
            credits = self._credits.get((identity, envelope.request_id))
            if credits is not None and credits.locked():
                credits.release()

    def _handle_credit(self, identity: bytes, envelope: IpcEnvelope) -> None:
        credits = self._credits.get((identity, envelope.request_id))
        if credits is not None:
            credits.release()

    async def _invoke(self, method: Any, *args: Any, **kwargs: Any) -> Any:
        if inspect.iscoroutinefunction(method):
            return await method(*args, **kwargs)
        result = await asyncio.to_thread(method, *args, **kwargs)
        if inspect.isawaitable(result):
            return await result
        return result

    async def _send(self, identity: bytes, envelope: IpcEnvelope) -> None:
        if self._queue is not None:
            await self._queue.put_async(envelope, routing_id=identity)

    async def _send_error(
        self,
        identity: bytes,
        request_id: str,
        sequence: int,
        error: BaseException,
    ) -> None:
        await self._send(
            identity,
            IpcEnvelope(
                request_id,
                "error",
                sequence,
                {
                    "type": type(error).__name__,
                    "message": str(error),
                    "traceback": traceback.format_exc(),
                },
            ),
        )

    def close(self, timeout: float = 10.0) -> None:
        """Abort active generations and notify connected clients."""
        if self._thread is None:
            return
        if self._loop is not None and self._stop_event is not None:
            self._loop.call_soon_threadsafe(self._request_stop)
        if threading.current_thread() is not self._thread:
            self._thread.join(timeout)
        self._thread = None

    def _request_stop(self) -> None:
        for promise in list(self._active.values()):
            promise.abort()
        for task in list(self._tasks):
            task.cancel()
        if self._stop_event is not None:
            self._stop_event.set()

    async def _shutdown_async(self) -> None:
        current = asyncio.current_task()
        tasks = [task for task in self._tasks if task is not current]
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        for promise in list(self._active.values()):
            try:
                promise.abort()
            except Exception:
                pass
        for identity in self._identities:
            try:
                await self._send(
                    identity,
                    IpcEnvelope("*", "closed", 0, "Generator IPC server closed"),
                )
            except Exception:
                pass
        if self._queue is not None:
            self._queue.socket.setsockopt(zmq.LINGER, 100)
            self._queue.close()
            self._queue = None
        socket_path = self._endpoint.removeprefix("ipc://")
        try:
            os.unlink(socket_path)
        except FileNotFoundError:
            pass

    def __enter__(self) -> "GeneratorIpcServer":
        self.start()
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()


class GeneratorIpcClient:
    """Generator proxy with a single background DEALER reader."""

    def __init__(
        self,
        address: tuple[str, bytes],
        *,
        queue_size: int = 32,
        timeout: float = 10.0,
    ) -> None:
        endpoint, hmac_key = address
        if not endpoint.startswith("ipc://"):
            raise ValueError("Generator IPC endpoint must use ipc://")
        if queue_size <= 0:
            raise ValueError("queue_size must be positive")
        self._address = address
        self._queue_size = queue_size
        self._timeout = timeout
        self._outbound: queue.Queue[IpcEnvelope] = queue.Queue(maxsize=queue_size)
        self._pending: dict[str, queue.Queue[IpcEnvelope | BaseException]] = {}
        self._pending_lock = threading.Lock()
        self._closed = threading.Event()
        self._started = threading.Event()
        self._startup_error: Optional[BaseException] = None
        self._socket: Optional[ZeroMqQueue] = None
        self._sequences: dict[str, int] = {}
        self._last_received = time.monotonic()
        self._thread = threading.Thread(
            target=self._io_main,
            name="generator-ipc-client",
            daemon=True,
        )
        self._thread.start()
        if not self._started.wait(timeout):
            raise TimeoutError("Timed out starting generator IPC client")
        if self._startup_error is not None:
            raise GeneratorIpcError("Failed to start generator IPC client") from self._startup_error
        if not self.is_ready():
            self.close()
            raise GeneratorIpcError("Remote generator is not ready")

    def _io_main(self) -> None:
        try:
            self._socket = ZeroMqQueue(
                address=self._address,
                is_server=False,
                socket_type=zmq.DEALER,
                name="generator_ipc_client",
            )
            self._socket.socket.setsockopt(zmq.RCVHWM, self._queue_size)
            self._socket.setup_lazily()
            self._started.set()
            last_heartbeat = 0.0
            while not self._closed.is_set() or not self._outbound.empty():
                try:
                    while True:
                        self._socket.put(self._outbound.get_nowait())
                except queue.Empty:
                    pass
                if self._socket.poll(0.05):
                    self._last_received = time.monotonic()
                    self._route(self._socket.get())
                now = time.monotonic()
                if now - last_heartbeat >= 1.0:
                    self._socket.put(IpcEnvelope("*", "heartbeat", 0, None))
                    last_heartbeat = now
                if self._pending and now - self._last_received > self._timeout:
                    raise GeneratorIpcClosedError("Generator IPC heartbeat timed out")
        except BaseException as error:
            self._startup_error = error
            self._started.set()
            self._closed.set()
            self._fail_all(GeneratorIpcClosedError(f"Generator IPC reader stopped: {error}"))
        finally:
            if self._socket is not None:
                self._socket.socket.setsockopt(zmq.LINGER, 100)
                self._socket.close()
                self._socket = None

    def _route(self, envelope: IpcEnvelope) -> None:
        if envelope.message_type == "closed" and envelope.request_id == "*":
            self._fail_all(GeneratorIpcClosedError(str(envelope.payload)))
            return
        with self._pending_lock:
            response_queue = self._pending.get(envelope.request_id)
        if response_queue is None:
            return
        expected = self._sequences.get(envelope.request_id, envelope.sequence)
        if envelope.sequence != expected:
            self._fail_request(
                envelope.request_id,
                GeneratorIpcError(
                    f"Out-of-order response for {envelope.request_id}: "
                    f"expected {expected}, got {envelope.sequence}"
                ),
            )
            return
        self._sequences[envelope.request_id] = expected + 1
        try:
            response_queue.put_nowait(envelope)
        except queue.Full:
            self._cancel(envelope.request_id)
            while True:
                try:
                    response_queue.get_nowait()
                except queue.Empty:
                    break
            response_queue.put_nowait(
                GeneratorIpcError(f"Generator IPC response queue is full for {envelope.request_id}")
            )
            self._remove_pending(envelope.request_id)
            return
        if envelope.message_type in ("ready", "result", "error", "stream_end", "cancelled"):
            self._remove_pending(envelope.request_id)

    def _new_request(
        self, message_type: MessageType, payload: Any
    ) -> tuple[str, queue.Queue[IpcEnvelope | BaseException]]:
        if self._closed.is_set():
            raise GeneratorIpcClosedError("Generator IPC client is closed")
        request_id = uuid.uuid4().hex
        response_queue: queue.Queue[IpcEnvelope | BaseException] = queue.Queue(
            maxsize=self._queue_size + 1
        )
        with self._pending_lock:
            self._pending[request_id] = response_queue
            self._sequences[request_id] = 0
        try:
            self._outbound.put(
                IpcEnvelope(request_id, message_type, 0, payload), timeout=self._timeout
            )
        except queue.Full:
            self._remove_pending(request_id)
            raise TimeoutError("Generator IPC outbound queue is full")
        return request_id, response_queue

    def _wait(
        self,
        request_id: str,
        response_queue: queue.Queue[IpcEnvelope | BaseException],
    ) -> IpcEnvelope:
        try:
            item = response_queue.get(timeout=self._timeout)
        except queue.Empty:
            self._remove_pending(request_id)
            raise TimeoutError(f"Timed out waiting for generator IPC request {request_id}")
        if isinstance(item, BaseException):
            raise item
        if item.message_type == "error":
            raise GeneratorIpcRemoteError(_format_remote_error(item.payload))
        return item

    def health_status(self) -> dict[str, Any]:
        """Return the remote generator's health status."""
        request_id, response_queue = self._new_request("health", None)
        response = self._wait(request_id, response_queue)
        if response.message_type != "ready":
            raise GeneratorIpcError(f"Unexpected health response: {response.message_type}")
        return response.payload

    def is_ready(self) -> bool:
        """Perform a health/readiness handshake."""
        return bool(self.health_status()["ready"])

    def call(self, method: str, *args: Any, **kwargs: Any) -> Any:
        """Call a public generator method and wait for its result."""
        request_id, response_queue = self._new_request(
            "call",
            {"method": method, "args": args, "kwargs": kwargs},
        )
        response = self._wait(request_id, response_queue)
        if response.message_type != "result":
            raise GeneratorIpcError(f"Unexpected unary response: {response.message_type}")
        if isinstance(response.payload, VisualGenOutputSnapshot):
            return _restore_visual_output(response.payload)
        return response.payload

    async def call_async(self, method: str, *args: Any, **kwargs: Any) -> Any:
        """Call a public generator method without blocking the event loop."""
        return await asyncio.to_thread(self.call, method, *args, **kwargs)

    def generate_async(self, *args: Any, **kwargs: Any) -> RemoteRequestOutput | RemoteUnaryFuture:
        """Start remote generation and return its RequestOutput-like iterator."""
        request_id, response_queue = self._new_request(
            "generate",
            {
                "args": args,
                "kwargs": kwargs,
                "stream_window": self._queue_size,
            },
        )
        response = self._wait(request_id, response_queue)
        if response.message_type != "stream_start":
            raise GeneratorIpcError(f"Unexpected generation response: {response.message_type}")
        if response.payload is None:
            return RemoteUnaryFuture(self, request_id, response_queue)
        return RemoteRequestOutput(self, request_id, response_queue, response.payload)

    def _cancel(self, request_id: str) -> None:
        if self._closed.is_set():
            return
        sequence = self._sequences.get(request_id, 1)
        try:
            self._outbound.put_nowait(IpcEnvelope(request_id, "cancel", sequence, None))
        except queue.Full:
            self._fail_request(
                request_id,
                GeneratorIpcError("Could not enqueue generator cancellation"),
            )

    def _credit(self, request_id: str) -> None:
        if self._closed.is_set():
            return
        try:
            self._outbound.put(IpcEnvelope(request_id, "credit", 0, None), timeout=self._timeout)
        except queue.Full:
            self._fail_request(
                request_id,
                GeneratorIpcError("Could not return generator stream credit"),
            )

    def register_alias(self, alias: str, promise: RemoteUnaryFuture | RemoteRequestOutput) -> None:
        """Associate a cross-worker cancellation alias with a generation."""
        request_id, response_queue = self._new_request(
            "alias",
            {
                "alias": alias,
                "target_request_id": promise._transport_request_id,
            },
        )
        response = self._wait(request_id, response_queue)
        if not response.payload:
            raise GeneratorIpcError("Generation completed before its alias was registered")

    async def cancel_alias(self, alias: str) -> bool:
        """Cancel a generation registered by any frontend worker."""
        request_id, response_queue = self._new_request("cancel_alias", alias)
        response = await asyncio.to_thread(self._wait, request_id, response_queue)
        return bool(response.payload)

    def _fail_request(self, request_id: str, error: BaseException) -> None:
        with self._pending_lock:
            response_queue = self._pending.get(request_id)
        if response_queue is not None:
            while True:
                try:
                    response_queue.get_nowait()
                except queue.Empty:
                    break
            response_queue.put_nowait(error)
        self._remove_pending(request_id)

    def _fail_all(self, error: BaseException) -> None:
        with self._pending_lock:
            pending = list(self._pending.values())
            self._pending.clear()
            self._sequences.clear()
        for response_queue in pending:
            try:
                while True:
                    response_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                response_queue.put_nowait(error)
            except queue.Full:
                pass

    def _remove_pending(self, request_id: str) -> None:
        with self._pending_lock:
            self._pending.pop(request_id, None)
            self._sequences.pop(request_id, None)

    def close(self, timeout: float = 5.0) -> None:
        """Close the reader and fail all outstanding operations."""
        if not self._closed.is_set():
            try:
                self._outbound.put_nowait(IpcEnvelope("*", "disconnect", 0, None))
            except queue.Full:
                pass
            self._closed.set()
            self._fail_all(GeneratorIpcClosedError("Generator IPC client closed"))
        if threading.current_thread() is not self._thread:
            self._thread.join(timeout)

    def __enter__(self) -> "GeneratorIpcClient":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()
