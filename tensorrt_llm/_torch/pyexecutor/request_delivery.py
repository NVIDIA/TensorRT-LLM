# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import threading
from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING

from .llm_request import LlmResponse

if TYPE_CHECKING:
    from ray.actor import ActorHandle

__all__ = ["ResponseChannel"]


class ResponseChannel:
    """Local responses waiting for pickup and the condition their waiters sleep on."""

    def __init__(self) -> None:
        self.response_lock = threading.Lock()
        self.response_cv = threading.Condition(self.response_lock)
        self.responses: dict[int, list[LlmResponse]] = {}
        self.result_wait_queues: dict[int, ActorHandle | None] = {}

    def register(self, request_ids: Iterable[int], result_wait_queue: ActorHandle) -> None:
        """Forward the requests' future responses to a Ray result wait queue.

        Args:
            request_ids: The requests whose responses the queue receives.
            result_wait_queue: The Ray actor that ``publish`` hands each response to.
        """
        with self.response_cv:
            for request_id in request_ids:
                self.result_wait_queues[request_id] = result_wait_queue

    def release(self, request_id: int) -> None:
        """Stop forwarding a finished request's responses.

        Args:
            request_id: The request; an unregistered id is ignored.
        """
        self.result_wait_queues.pop(request_id, None)

    def clear(self) -> None:
        """Drop every stored response and every registered queue."""
        with self.response_cv:
            self.responses.clear()
            self.result_wait_queues.clear()

    def publish(self, responses: Iterable[tuple[int, LlmResponse]]) -> None:
        """Store responses for pickup, forward them to registered queues and wake the waiters.

        Args:
            responses: ``(request_id, response)`` pairs, in delivery order.
        """
        with self.response_cv:
            for req_id, resp in responses:
                if req_id in self.responses.keys():
                    self.responses[req_id].append(resp)
                else:
                    self.responses.update({req_id: [resp]})
                if (
                    type(resp) is LlmResponse
                    and req_id in self.result_wait_queues
                    and self.result_wait_queues[req_id] is not None
                ):
                    self.result_wait_queues[req_id].put_response.remote(resp.client_id, resp)
            self.response_cv.notify_all()

    def await_any(
        self, timeout: float | None, *, is_shutdown: Callable[[], bool]
    ) -> list[LlmResponse]:
        """Wait for any response, then take every stored response.

        Args:
            timeout: Seconds to wait; None waits until a response arrives or shutdown.
            is_shutdown: Whether the executor has shut down, which ends the wait early.

        Returns:
            Every stored response, or an empty list on timeout or shutdown.
        """

        def any_responses_ready() -> bool:
            return len(self.responses) > 0 or is_shutdown()

        responses = []
        with self.response_cv:
            self.response_cv.wait_for(any_responses_ready, timeout=timeout)
            for response in self.responses.values():
                responses += response
            self.responses = {}

        return responses

    def await_single(
        self,
        request_id: int,
        timeout: float | None,
        *,
        is_shutdown: Callable[[], bool],
        read_error: Callable[[], BaseException | None],
        mark_error_delivered: Callable[[], None],
    ) -> list[LlmResponse]:
        """Wait for one request's responses and take them.

        Args:
            request_id: The request.
            timeout: Seconds to wait; None waits until a response arrives or shutdown.
            is_shutdown: Whether the executor has shut down, which ends the wait early.
            read_error: The error that ended the executor loop, if any.
            mark_error_delivered: Called once that error is raised to this caller.

        Returns:
            The request's stored responses, or an empty list on timeout.

        Raises:
            RuntimeError: The executor shut down before the request had a response.
        """
        with self.response_cv:

            def key_has_response() -> bool:
                # Wake on shutdown too so that an event-loop crash
                # cannot trap callers here forever (nvbug 6038228).
                return request_id in self.responses or is_shutdown()

            self.response_cv.wait_for(key_has_response, timeout=timeout)
            if request_id in self.responses:
                return self.responses.pop(request_id)
            if is_shutdown():
                # The event-loop thread terminated before producing a
                # response for this request. Re-raise the original
                # exception (if any) so callers see a meaningful error
                # instead of hanging here or hitting a KeyError below.
                error = read_error()
                if error is not None:
                    # The caller is about to see the original exception, so
                    # the crash is reportable without killing the world.
                    mark_error_delivered()
                    raise RuntimeError(f"Event loop terminated with error: {error}") from error
                raise RuntimeError(
                    f"Event loop shut down before a response was received for request {request_id}."
                )
            # Timed out without shutdown: return [] like the other timeouts.
            return []
