import weakref
from queue import Queue
from typing import (TYPE_CHECKING, Callable, Optional, Union, Any, cast)

from tensorrt_llm._utils import nvtx_range_debug
from tensorrt_llm.llmapi.tracer import global_tracer
from tensorrt_llm.llmapi.utils import AsyncQueue
from tensorrt_llm.executor.utils import ErrorResponse, has_event_loop

if TYPE_CHECKING:
    from tensorrt_llm.executor import GenerationExecutor
from .request import MultimodalRequest, MultimodalResponse

__all__ = [
    "MultimodalResult",
]

class MultimodalResult:
    def __init__(
        self,
        mm_request: MultimodalRequest,
        background_error_handler: Optional[Callable] = None,
        executor: Optional["GenerationExecutor"] = None,
    ) -> None:
        self.request_id = mm_request.id  # abort_request is using request_id
        self._background_error_handler = background_error_handler
        self._done = False
        self._timeout = 2
        self._executor: Optional[weakref.ReferenceType[
            "GenerationExecutor"]] = weakref.ref(executor) if executor else None
        self._aborted = False
        self.multimodal_params = None
        if has_event_loop():
            self.aqueue = AsyncQueue()
            self.queue = self.aqueue.sync_q
        else:
            self.queue = Queue()
            self.aqueue = None

    def set_timeout(self, timeout: float) -> None:
        """Set the timeout for getting results."""
        self._timeout = timeout

    def mark_undone(self) -> None:
        """Should be called when new prompts are submitted."""
        self._done = False

    @property
    def finished(self) -> bool:
        return self._done

    async def _aresult_step(self) -> None:
        assert self.aqueue is not None, "The asyncio event loop was not present during initialization, so async operations are not available."
        response = await self.aqueue.get()
        global_tracer().log_instant("result_step.get")
        self._handle_response(response)

    async def aresult(self) -> "MultimodalResult":
        """Wait for the completion of the request, and return the result.

        Returns:
            MultimodalResult: The result object.
        """
        while not self._done:
            await self._aresult_step()
        return self

    def _result_step(self, timeout: Optional[float] = None) -> None:
        response = self.queue.get(timeout=timeout)
        self._handle_response(response)

    @nvtx_range_debug("handle_response",
                      color="red",
                      category="MultimodalResult")
    def _handle_response(self,
                         response: Union[MultimodalResponse, ErrorResponse]) -> None:
        if isinstance(response, MultimodalResponse):
            if response.has_error():
                if self._background_error_handler is not None and (
                        handler := self._background_error_handler()):
                    handler(response.error_msg)

            response_result = response.result
            self._done = response_result.is_final
            self.multimodal_params = response.get_params()

            if self._background_error_handler and (
                    handler := self._background_error_handler()):
                handler()
        elif isinstance(response, ErrorResponse):
            if self._background_error_handler is not None and (
                    handler := self._background_error_handler()):
                handler(response.error_msg)
            # TODO: we should not need to set done here; but proxy error_queue is always empty (?)
            # WAR: we set done to unblock the result.get()
            self._done = True
        else:
            raise ValueError(f"Unknown response type: {response}")

    def result(self, timeout: Optional[float] = None) -> "MultimodalResult":
        """Wait for the completion of the request, and return the result.

        Args:
            timeout (float, optional): Timeout. Defaults to None.

        Returns:
            tensorrt_llm.executor.result.GenerationResult: generation result.
        """
        while not self._done:
            self._result_step(timeout)
        return self

    def abort(self) -> None:
        """Abort the multimodal request."""
        assert self._executor is not None, "The executor is not set for this result."
        executor = self._executor()
        assert executor is not None, "The executor has been garbage collected."
        assert self.request_id is not None, "The request ID is not set."
        executor.abort_request(self.request_id)
        self._aborted = True

    def __await__(self):
        """Make the result awaitable."""
        return self.aresult().__await__()

    def __iter__(self):
        """Make the result iterable."""
        return self

    def __next__(self):
        """Get the next result."""
        if self._done:
            raise StopIteration
        return self.result()

    def __aiter__(self):
        """Make the result async iterable."""
        return self

    async def __anext__(self):
        """Get the next result asynchronously."""
        if self._done:
            raise StopAsyncIteration
        return await self.aresult()

    def _exception(self, timeout: Optional[float] = None) -> Optional[Exception]:
        """Get any exception that occurred during processing."""
        try:
            self._result_step(timeout)
        except Exception as e:
            return e
        return None

    def _repr_fields(self) -> dict[str, Any]:
        """Get fields for string representation."""
        return {
            "request_id": self.request_id,
            "done": self._done,
        }

    def __repr__(self) -> str:
        """Get string representation of the result."""
        fields = self._repr_fields()
        fields_str = ", ".join(f"{k}={v!r}" for k, v in fields.items())
        return f"{self.__class__.__name__}({fields_str})"

