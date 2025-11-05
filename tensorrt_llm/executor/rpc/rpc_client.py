import asyncio
import concurrent.futures
import threading
import uuid
from typing import Any, AsyncIterator, Callable, Optional

from ..._utils import nvtx_mark_debug
from ...llmapi.utils import enable_llmapi_debug, logger_debug
from ...logger import logger
from ..ipc import ZeroMqQueue
from .rpc_common import (RPCCancelled, RPCParams, RPCRequest, RPCResponse,
                         RPCStreamingError, RPCTimeout)


class RemoteCall:
    """Helper class to enable chained remote call syntax like client.method().remote()"""

    def __init__(self, client: 'RPCClient', method_name: str, *args,
                 **kwargs) -> None:
        self.client = client
        self.method_name = method_name
        self.args = args
        self.kwargs = kwargs

    def _prepare_and_call(self, timeout: Optional[float], need_response: bool,
                          mode: str, call_method: str) -> Any:
        """Common method to prepare RPC params and make the call.

        Args:
            timeout: Timeout for the RPC call
            need_response: Whether a response is expected
            mode: The RPC mode ("sync", "async", "future")
            call_method: The method name to call on the client

        Returns:
            The result of the client method call
        """
        rpc_params = RPCParams(timeout=timeout,
                               need_response=need_response,
                               mode=mode)
        self.kwargs["__rpc_params"] = rpc_params
        client_method = getattr(self.client, call_method)
        return client_method(self.method_name, *self.args, **self.kwargs)

    def remote(self,
               timeout: Optional[float] = None,
               need_response: bool = True) -> Any:
        """Synchronous remote call with optional RPC parameters.

        Args:
            timeout: Timeout for the RPC call
            need_response: Whether a response is expected

        Returns:
            The result of the client method call
        """
        return self._prepare_and_call(timeout, need_response, "sync",
                                      "_call_sync")

    def remote_async(self,
                     timeout: Optional[float] = None,
                     need_response: bool = True) -> Any:
        """Asynchronous remote call that returns a coroutine.

        Args:
            timeout: Timeout for the RPC call
            need_response: Whether a response is expected

        Returns:
            A coroutine that will yield the result of the client method call
        """
        return self._prepare_and_call(timeout, need_response, "async",
                                      "_call_async")

    def remote_future(self,
                      timeout: Optional[float] = None,
                      need_response: bool = True) -> concurrent.futures.Future:
        """Remote call that returns a Future object.

        Args:
            timeout: Timeout for the RPC call
            need_response: Whether a response is expected

        Returns:
            A Future object that can be used to retrieve the result of the client method call
        """
        return self._prepare_and_call(timeout, need_response, "future",
                                      "_call_future")

    def remote_streaming(self,
                         timeout: Optional[float] = None) -> AsyncIterator[Any]:
        """Remote call for streaming results.

        Args:
            timeout: Timeout for the RPC call

        Returns:
            An AsyncIterator that will yield the result of the client method call
        """
        # Streaming always needs a response
        return self._prepare_and_call(timeout, True, "async", "_call_streaming")


class RPCClient:
    """
    An RPC Client that connects to the RPCServer.
    """

    def __init__(self,
                 address: str,
                 hmac_key: Optional[bytes] = None,
                 timeout: Optional[float] = None,
                 num_workers: int = 4) -> None:
        '''
        Args:
            address: The ZMQ address to connect to.
            hmac_key: The HMAC key for encryption.
            timeout: The timeout (seconds) for RPC calls.
            num_workers: The number of workers for the RPC client.
        '''
        self._address = address
        self._timeout = timeout
        self._client_socket = ZeroMqQueue(address=(address, hmac_key),
                                          is_server=False,
                                          is_async=True,
                                          use_hmac_encryption=False)
        # Store futures directly without loop references
        self._pending_futures: dict[str, asyncio.Future] = {}
        # Use asyncio.Queue for streaming responses
        self._streaming_queues: dict[str, asyncio.Queue] = {}
        self._reader_task: Optional[asyncio.Task] = None
        # Keep executor for remote_future()
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=num_workers, thread_name_prefix="rpc_client_worker")

        self._server_stopped = False
        self._closed = False
        self._reader_lock = threading.Lock()

        logger_debug(f"RPC Client initialized. Connected to {self._address}")

    def shutdown_server(self) -> None:
        """Shutdown the server."""
        if self._server_stopped:
            return

        self._rpc_shutdown().remote()

        self._server_stopped = True

    def close(self) -> None:
        """Gracefully close the client, cleaning up background tasks."""
        if self._closed:
            return
        self._closed = True

        logger_debug("RPC Client closing")

        # Cancel the reader task if it exists
        if self._reader_task and not self._reader_task.done():
            self._reader_task.cancel()
            # Don't wait for the task here as it might be in a different event loop
            # The task will clean itself up when cancelled

        # Clean up the executor
        if self._executor:
            self._executor.shutdown(wait=True)
            self._executor = None

        # Close the socket
        if self._client_socket:
            self._client_socket.close()
            self._client_socket = None

        # Stop the event loop
        if self._loop and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._loop_thread:
            self._loop_thread.join(timeout=2.0)
            self._loop_thread = None

        if self._executor:
            self._executor.shutdown(wait=True)

        logger_debug("RPC Client closed")

    async def _response_reader(self) -> None:
        """Task to read responses from the socket and set results on futures."""
        logger_debug("Response reader started")

        try:
            while not self._closed:
                try:
                    response: RPCResponse = await self._client_socket.get_async(
                    )

                    logger_debug(f"RPC Client received response: {response}")
                    logger_debug(
                        f"Response request_id: {response.request_id}, is_streaming: {response.is_streaming}"
                    )

                    # Handle streaming responses
                    if response.is_streaming:
                        assert response.stream_status in [
                            'start', 'data', 'end', 'error'
                        ], f"Invalid stream status: {response.stream_status}"

                        queue = self._streaming_queues.get(response.request_id)
                        if queue:
                            await queue.put(response)
                            # Clean up if stream ended
                            if response.stream_status in ['end', 'error']:
                                self._streaming_queues.pop(
                                    response.request_id, None)
                    else:
                        # Handle regular responses
                        logger_debug(
                            f"Handling regular response for request_id: {response.request_id}"
                        )
                        future = self._pending_futures.pop(
                            response.request_id, None)
                        if future and not future.done():
                            if response.error is None:
                                logger_debug(
                                    f"RPC Client received response: request_id={response.request_id}, "
                                    f"is_streaming={response.is_streaming}, "
                                    f"pending_futures={len(self._pending_futures)}"
                                )
                                future.set_result(response.result)
                            else:
                                logger_debug(
                                    f"Setting exception for request_id: {response.request_id}, error: {response.error}"
                                )
                                future.set_exception(response.error)

                except asyncio.CancelledError:
                    logger_debug("Response reader cancelled")
                    break
                except Exception as e:
                    if self._closed:
                        break
                    logger.error(f"Exception in RPC response reader: {e}")
                    # Propagate exception to all pending futures
                    for future in list(self._pending_futures.values()):
                        if not future.done():
                            future.set_exception(e)
                    # Also signal error to streaming queues
                    for queue in list(self._streaming_queues.values()):
                        try:
                            await queue.put(
                                RPCResponse("", None, e, False, 0, 'error'))
                        except Exception:
                            pass
                    break

        finally:
            logger_debug("Response reader exiting gracefully")

    def _ensure_reader_task(self) -> None:
        """Ensure the response reader task is running."""
        with self._reader_lock:
            if self._reader_task is None or self._reader_task.done():
                try:
                    loop = asyncio.get_running_loop()
                    self._reader_task = loop.create_task(
                        self._response_reader())
                except RuntimeError:
                    # No running event loop, will be started when needed
                    pass

    async def _call_async(self, method_name: str, *args, **kwargs) -> Any:
        """Async version of RPC call.
        Args:
            method_name: Method name to call
            *args: Positional arguments
            **kwargs: Keyword arguments
            __rpc_params: RPCParams object containing RPC parameters.

        Returns:
            The result of the remote method call
        """
        if enable_llmapi_debug() or logger.level == 'debug':
            logger_debug(f"RPC client calling method: {method_name}")
        nvtx_mark_debug(f"RPC.async.{method_name}",
                        color="yellow",
                        category="RPC")
        if self._server_stopped:
            raise RPCCancelled("Server is shutting down, request cancelled")

        # Ensure reader task is running
        self._ensure_reader_task()

        rpc_params = kwargs.pop("__rpc_params", RPCParams())
        need_response = rpc_params.need_response
        timeout = rpc_params.timeout if rpc_params.timeout is not None else self._timeout

        request_id = uuid.uuid4().hex
        request = RPCRequest(request_id,
                             method_name=method_name,
                             args=args,
                             kwargs=kwargs,
                             need_response=need_response,
                             timeout=timeout)
        await self._client_socket.put_async(request)

        # Early return without waiting for response
        if not need_response:
            return None

        # Create future in the current event loop
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        self._pending_futures[request_id] = future

        logger_debug(
            f"RPC Client _call_async: Created future for request_id: {request_id}"
        )

        try:
            if timeout is None:
                res = await future
            else:
                res = await asyncio.wait_for(future, timeout)
            return res
        except RPCCancelled:
            self._server_stopped = True
            raise
        except asyncio.TimeoutError:
            self._pending_futures.pop(request_id, None)
            raise RPCTimeout(
                f"Request '{method_name}' timed out after {timeout}s")
        except Exception:
            raise

    def _call_sync(self, method_name: str, *args, **kwargs) -> Any:
        """Synchronous version of RPC call."""
        logger_debug(
            f"RPC Client calling method: {method_name} with args: {args} and kwargs: {kwargs}"
        )

        # Check if we're in an event loop
        try:
            asyncio.get_running_loop()

            # We're inside an event loop, we need to run in a thread to avoid deadlock
            def run_in_thread() -> Any:
                return asyncio.run(
                    self._call_async(method_name, *args, **kwargs))

            future = self._executor.submit(run_in_thread)
            return future.result()
        except RuntimeError:
            # No running event loop, we can use asyncio.run
            return asyncio.run(self._call_async(method_name, *args, **kwargs))

    def _call_future(self, name: str, *args,
                     **kwargs) -> concurrent.futures.Future:
        """
        Call a remote method and return a Future.

        Args:
            name: Method name to call
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            A Future object that can be used to retrieve the result
        """
        # Create a thread-safe future to bridge between asyncio and concurrent.futures
        thread_future = concurrent.futures.Future()

        async def _async_wrapper():
            try:
                result = await self._call_async(name, *args, **kwargs)
                thread_future.set_result(result)
            except Exception as e:
                thread_future.set_exception(e)

        try:
            # In an event loop, create the task
            asyncio.get_running_loop()
            asyncio.create_task(_async_wrapper())
        except RuntimeError:
            # No event loop, run in executor
            def _run_async_call():
                return asyncio.run(self._call_async(name, *args, **kwargs))

            return self._executor.submit(_run_async_call)

        return thread_future

    async def _call_streaming(self, name: str, *args,
                              **kwargs) -> AsyncIterator[Any]:
        """
        Call a remote async generator method and get streaming results.

        Args:
            name: Method name to call
            *args: Positional arguments
            **kwargs: Keyword arguments

        Yields:
            Results from the remote async generator
        """
        nvtx_mark_debug(f"RPC.streaming.{name}", color="red", category="RPC")

        if self._server_stopped:
            raise RPCCancelled("Server is shutting down, request cancelled")

        # Ensure reader task is running
        self._ensure_reader_task()

        rpc_params = kwargs.pop("__rpc_params", RPCParams())
        timeout = rpc_params.timeout if rpc_params.timeout is not None else self._timeout

        request_id = uuid.uuid4().hex
        # Use asyncio.Queue for streaming
        queue: asyncio.Queue = asyncio.Queue()
        self._streaming_queues[request_id] = queue

        try:
            # Send streaming request
            request = RPCRequest(request_id,
                                 name,
                                 args,
                                 kwargs,
                                 need_response=True,
                                 timeout=timeout,
                                 is_streaming=True)
            await self._client_socket.put_async(request)

            # Read streaming responses
            while True:
                if timeout is None:
                    response = await queue.get()
                else:
                    response = await asyncio.wait_for(queue.get(),
                                                      timeout=timeout)

                logger_debug(
                    f"RPC Client _call_streaming received [{response.stream_status}] response: {response}",
                    color="green")

                if response.stream_status == 'start':
                    # Start of stream
                    continue
                elif response.stream_status == 'data':
                    yield response.result
                elif response.stream_status == 'end':
                    # End of stream
                    break
                elif response.stream_status == 'error':
                    # Error in stream
                    if response.error:
                        raise response.error
                    else:
                        raise RPCStreamingError("Unknown streaming error")

        except asyncio.TimeoutError:
            raise RPCTimeout(
                f"Streaming request '{name}' timed out after {timeout}s")
        finally:
            # Clean up
            self._streaming_queues.pop(request_id, None)

    def get_server_attr(self, name: str) -> Any:
        """ Get the attribute of the RPC server.
        This is mainly used for testing. """
        return self._rpc_get_attr(name).remote()

    def __getattr__(self, name: str) -> Callable[..., RemoteCall]:
        """
        Magically handles calls to non-existent methods.
        Returns a callable that when invoked returns a RemoteCall instance.

        This enables the new syntax:
            client.method(args).remote()
            await client.method(args).remote_async()
            client.method(args).remote_future()
            async for x in client.method(args).remote_streaming()
        """
        logger_debug(f"RPC Client getting attribute: {name}")

        def method_caller(*args, **kwargs) -> RemoteCall:
            return RemoteCall(self, name, *args, **kwargs)

        return method_caller

    def __enter__(self) -> 'RPCClient':
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()
