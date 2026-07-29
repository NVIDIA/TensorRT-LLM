# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Dynamic batcher for encoder-only (embedding) serving.

Coalesces independent concurrent requests into a single batched `encode_fn` call,
mirroring NVIDIA Triton Inference Server's dynamic batcher: a configurable hold
window (`max_queue_delay`) and a maximum batch size, with a token budget and queue
backpressure. The actual model forward is injected as `encode_fn` so this module
carries no TensorRT-LLM / torch dependencies and is unit-testable on CPU.
"""

import asyncio
from dataclasses import dataclass
from typing import Any, Callable, List, Optional

# encode_fn maps a batch (list of token-id lists) to a list of per-item results,
# in the same order as the input.
EncodeFn = Callable[[List[List[int]]], List[Any]]


class QueueFullError(Exception):
    """Raised by submit() when the request queue is full (maps to HTTP 429)."""


class InputTooLongError(ValueError):
    """Raised by submit() when an input exceeds max_seq_len (maps to HTTP 400)."""


@dataclass
class _QueuedRequest:
    token_ids: List[int]
    future: asyncio.Future


class EncodeBatcher:
    """Async micro-batcher in front of a synchronous `encode_fn`.

    A single background worker drains a request queue and forms a batch, flushing
    it to `encode_fn` when any of three triggers fires: the batch reaches
    `max_batch_size`; adding the next request would exceed `max_num_tokens`; or
    the `max_queue_delay` hold window elapses. Results are routed back to each
    caller by input index. `encode_fn` runs in the default executor so the event
    loop is never blocked.

    There is exactly one worker by design (no `num_workers` knob). It is both
    sufficient and required: the GPU serializes forwards anyway, and the underlying
    `EncoderExecutor` runs `encode()` on the calling thread over shared,
    pre-allocated CUDA buffers and a single CUDA-graph runner — concurrent calls
    would race. Scale throughput via `max_batch_size` (bigger coalesced batches)
    or multiple single-GPU server instances, not more workers.
    """

    def __init__(
        self,
        encode_fn: EncodeFn,
        *,
        max_batch_size: int,
        max_queue_delay: float,
        max_queue_size: int,
        max_num_tokens: Optional[int] = None,
        max_seq_len: Optional[int] = None,
    ):
        self._encode_fn = encode_fn
        self._max_batch_size = max_batch_size
        self._max_queue_delay = max_queue_delay
        self._max_num_tokens = max_num_tokens
        self._max_seq_len = max_seq_len
        self._queue: "asyncio.Queue[_QueuedRequest]" = asyncio.Queue(maxsize=max_queue_size)
        self._worker: Optional[asyncio.Task] = None

    async def start(self) -> None:
        self._worker = asyncio.create_task(self._run())

    def is_alive(self) -> bool:
        """Whether the background worker is running.

        Returns False if the worker was never started or has exited (e.g. it
        crashed). A dead worker means every queued and future request hangs, so
        the server's health check reports unhealthy when this is False.
        """
        return self._worker is not None and not self._worker.done()

    def validate_input(self, token_ids: List[int]) -> None:
        """Validate a single input against the model's capacity.

        Raises InputTooLongError (HTTP 400) if the input exceeds `max_seq_len`
        (the model's positional limit) or, on its own, exceeds `max_num_tokens`
        (the engine's per-batch token budget) — either of which would make
        `encode_fn` reject the formed batch. Callers may invoke this before
        submitting a group of inputs so an oversize item fails fast.
        """
        if self._max_seq_len is not None and len(token_ids) > self._max_seq_len:
            raise InputTooLongError(
                f"Input length ({len(token_ids)}) exceeds max_seq_len ({self._max_seq_len})."
            )
        if self._max_num_tokens is not None and len(token_ids) > self._max_num_tokens:
            raise InputTooLongError(
                f"Input length ({len(token_ids)}) exceeds max_num_tokens ({self._max_num_tokens})."
            )

    async def submit(self, token_ids: List[int]) -> Any:
        """Enqueue one input and await its encoded result.

        Raises InputTooLongError (HTTP 400) if the input exceeds the model's
        capacity, and QueueFullError (HTTP 429) if the queue is full.
        """
        self.validate_input(token_ids)
        future: asyncio.Future = asyncio.get_running_loop().create_future()
        request = _QueuedRequest(token_ids=token_ids, future=future)
        try:
            self._queue.put_nowait(request)
        except asyncio.QueueFull:
            raise QueueFullError("Embedding request queue is full; retry later.")
        return await future

    async def _run(self) -> None:
        loop = asyncio.get_running_loop()
        # A request dequeued but held back because it would overflow the token
        # budget seeds the next batch instead of being dropped.
        pending: Optional[_QueuedRequest] = None
        while True:
            batch, pending = await self._collect_batch(loop, pending)
            await self._dispatch(loop, batch)

    async def _collect_batch(self, loop, pending):
        """Block for the first request, then coalesce more until a trigger fires.

        Returns the formed batch plus any request carried over to the next batch
        (an over-budget request that was dequeued but not added).
        """
        if pending is not None:
            batch = [pending]
        else:
            batch = [await self._queue.get()]
        num_tokens = len(batch[0].token_ids)
        deadline = loop.time() + self._max_queue_delay
        carry_over: Optional[_QueuedRequest] = None
        while len(batch) < self._max_batch_size:
            timeout = deadline - loop.time()
            if timeout <= 0:
                break
            try:
                request = await asyncio.wait_for(self._queue.get(), timeout)
            except asyncio.TimeoutError:
                break
            if (
                self._max_num_tokens is not None
                and num_tokens + len(request.token_ids) > self._max_num_tokens
            ):
                carry_over = request  # keeps the current batch within budget
                break
            batch.append(request)
            num_tokens += len(request.token_ids)
        return batch, carry_over

    async def _dispatch(self, loop, batch: List[_QueuedRequest]) -> None:
        """Run encode_fn on the batch and resolve each request's future."""
        token_ids_batch = [request.token_ids for request in batch]
        try:
            # encode_fn is synchronous and blocking (a GPU forward); run it in the
            # default executor so the event loop keeps serving requests.
            results = await loop.run_in_executor(None, self._encode_fn, token_ids_batch)
        except Exception as exc:  # noqa: BLE001 - fail only this batch
            # Isolate the failure to this batch; keep the worker alive.
            for request in batch:
                if not request.future.done():
                    request.future.set_exception(exc)
            return
        for request, result in zip(batch, results):
            # A request whose future was cancelled while queued (e.g. the client
            # disconnected) is already done; set_result() would raise
            # InvalidStateError and kill the sole worker, hanging every later
            # request. Skip those and only resolve futures still awaiting.
            if not request.future.done():
                request.future.set_result(result)

    async def shutdown(self) -> None:
        if self._worker is not None:
            self._worker.cancel()
            try:
                await self._worker
            except asyncio.CancelledError:
                pass
            self._worker = None
