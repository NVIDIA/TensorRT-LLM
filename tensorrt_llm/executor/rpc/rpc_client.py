import asyncio
import concurrent.futures
import threading
import uuid
from typing import Any, AsyncIterator, Dict, Optional

from ...llmapi.utils import AsyncQueue, _SyncQueue, logger_debug
from ...logger import logger
from ..ipc import ZeroMqQueue
from .rpc_common import (RPCCancelled, RPCParams, RPCRequest, RPCResponse,
                         RPCStreamingError, RPCTimeout)


class RemoteCall:
    """Helper class to enable chained remote call syntax like client.method().remote()"""

    def __init__(self, client: 'RPCClient', method_name: str, *args, **kwargs):
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
        """Synchronous remote call with optional RPC parameters."""
        return self._prepare_and_call(timeout, need_response, "sync",
                                      "_call_sync")

    def remote_async(self,
                     timeout: Optional[float] = None,
                     need_response: bool = True):
        """Asynchronous remote call that returns a coroutine."""
        return self._prepare_and_call(timeout, need_response, "async",
                                      "_call_async")

    def remote_future(self,
                      timeout: Optional[float] = None,
                      need_response: bool = True) -> concurrent.futures.Future:
        """Remote call that returns a Future object."""
        return self._prepare_and_call(timeout, need_response, "future",
                                      "_call_future")

    def remote_streaming(self,
                         timeout: Optional[float] = None) -> AsyncIterator[Any]:
        """Remote call for streaming results."""
        # Streaming always needs a response
        return self._prepare_and_call(timeout, True, "async", "_call_streaming")


class RPCClient:
    """
    An RPC Client that connects to the RPCServer.
    """

    def __init__(self,
                 address: str,
                 hmac_key=None,
                 timeout: Optional[float] = None,
                 num_workers: int = 4):
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
        self._pending_futures = {}
        # map request_id to the queue for streaming responses
        self._streaming_queues: Dict[str, AsyncQueue] = {}
        self._reader_task = None
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=num_workers, thread_name_prefix="rpc_client_worker")

        self._server_stopped = False
        self._closed = False
        self._stop_event = None
        self._loop = None
        self._loop_thread = None

        logger_debug(f"RPC Client initialized. Connected to {self._address}")

    def shutdown_server(self):
        """Shutdown the server."""
        if self._server_stopped:
            return

        self._rpc_shutdown().remote()

        self._server_stopped = True

    def close(self):
        """Gracefully close the client, cleaning up background tasks."""

        if self._closed:
            return
        # stop the main loop
        self._closed = True

        logger_debug("RPC Client closing")

        if self._stop_event and self._loop:
            # Use call_soon_threadsafe since set() is not a coroutine
            self._loop.call_soon_threadsafe(self._stop_event.set)

        if self._reader_task:
            try:
                self._reader_task.result(timeout=1.0)
            except concurrent.futures.TimeoutError:
                logger.warning(
                    "Reader task did not exit gracefully, cancelling")
                self._reader_task.cancel()
            except Exception as e:
                # Task might have already finished or been cancelled
                logger_debug(f"Reader task cleanup: {e}")
            self._reader_task = None

        if self._loop and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._loop_thread:
            self._loop_thread.join()
            self._loop_thread = None
        if self._executor:
            self._executor.shutdown(wait=True)

        if self._client_socket:
            self._client_socket.close()
            self._client_socket = None

        logger_debug("RPC Client closed")

    async def _response_reader(self):
        """Task to read responses from the socket and set results on futures."""
        logger_debug("Response reader started")

        while not self._stop_event.is_set():
            try:
                # Use wait_for with a short timeout to periodically check stop event
                try:
                    response: RPCResponse = await asyncio.wait_for(
                        self._client_socket.get_async(),
                        timeout=0.1  # Check stop event every 100ms
                    )
                except asyncio.TimeoutError:
                    # Timeout is expected - just check stop event and continue
                    continue

                logger_debug(f"RPC Client received response: {response}")
                logger_debug(
                    f"Response request_id: {response.request_id}, is_streaming: {response.is_streaming}"
                )
                logger_debug(
                    f"Pending futures: {list(self._pending_futures.keys())}")

                # Handle streaming responses
                if response.is_streaming:
                    assert response.stream_status in [
                        'start', 'data', 'end', 'error'
                    ], f"Invalid stream status: {response.stream_status}"
                    queue = self._streaming_queues.get(response.request_id)
                    if queue:
                        # put to the sync queue, as the current event loop is
                        # different from the one in call_async or call_streaming
                        assert isinstance(queue, AsyncQueue)
                        logger_debug(
                            f"RPC Client putting response to AsyncQueue: {response}"
                        )
                        queue.sync_q.put(response)
                        # Clean up if stream ended
                        if response.stream_status in ['end', 'error']:
                            self._streaming_queues.pop(response.request_id,
                                                       None)
                else:
                    # Handle regular responses
                    logger_debug(
                        f"Handling regular response for request_id: {response.request_id}"
                    )
                    if future_info := self._pending_futures.get(
                            response.request_id):
                        future, target_loop = future_info
                        logger_debug(
                            f"Found future for request_id: {response.request_id}, future done: {future.done()}"
                        )

                        if not future.done():
                            if response.error is None:
                                logger_debug(
                                    f"Setting result for request_id: {response.request_id}, result: {response.result}"
                                )
                                target_loop.call_soon_threadsafe(
                                    future.set_result, response.result)
                            else:
                                # Use the original RPCError from the response
                                logger_debug(
                                    f"Setting exception for request_id: {response.request_id}, error: {response.error}"
                                )
                                target_loop.call_soon_threadsafe(
                                    future.set_exception, response.error)
                    else:
                        logger_debug(
                            f"No future found for request_id: {response.request_id}"
                        )
                    self._pending_futures.pop(response.request_id, None)

            except asyncio.CancelledError:
                # Still handle cancellation for backward compatibility
                logger_debug("Response reader cancelled")
                break
            except Exception as e:
                logger.error(f"Exception in RPC response reader: {e}")
                # Propagate exception to all pending futures
                for (future, target_loop) in self._pending_futures.values():

                    if not future.done():
                        target_loop.call_soon_threadsafe(
                            future.set_exception, e)
                # Also signal error to streaming queues
                for queue in self._streaming_queues.values():
                    await queue.put(RPCResponse("", None, e, False, 0, 'error'))
                break

        logger_debug("Response reader exiting gracefully")
        self._reader_task = None

    def _start_response_reader_lazily(self):
        if self._reader_task is None or self._reader_task.done():
            # Ensure we have a persistent background loop
            self._ensure_event_loop()
            # Always create the reader task on the persistent loop
            future = asyncio.run_coroutine_threadsafe(self._response_reader(),
                                                      self._loop)
            # Store the concurrent.futures.Future
            self._reader_task = future

    async def _call_async(self, method_name, *args, **kwargs):
        """Async version of RPC call.
        Args:
            method_name: Method name to call
            *args: Positional arguments
            **kwargs: Keyword arguments
            __rpc_params: RPCParams object containing RPC parameters.

        Returns:
            The result of the remote method call
        """
        logger_debug(
            f"RPC client calling method: {method_name} with args: {args} and kwargs: {kwargs}"
        )
        if self._server_stopped:
            raise RPCCancelled("Server is shutting down, request cancelled")

        self._start_response_reader_lazily()
        rpc_params = kwargs.pop("__rpc_params", RPCParams())
        need_response = rpc_params.need_response
        timeout = rpc_params.timeout if rpc_params.timeout is not None else self._timeout

        request_id = uuid.uuid4().hex
        request = RPCRequest(request_id,
                             method_name,
                             args,
                             kwargs,
                             need_response,
                             timeout=timeout)
        logger_debug(f"RPC client sending request: {request}")
        await self._client_socket.put_async(request)

        if not need_response:
            return None

        loop = asyncio.get_running_loop()
        future = loop.create_future()
        logger_debug(
            f"RPC Client _call_async: Created future for request_id: {request_id} in loop: {id(loop)}"
        )
        self._pending_futures[request_id] = (future, loop)
        logger_debug(
            f"RPC Client _call_async: Stored future in pending_futures")

        try:
            # If timeout, the remote call should return a timeout error timely,
            # so we add 1 second to the timeout to ensure the client can get
            # that result.
            logger_debug(
                f"RPC Client _call_async: Awaiting future for request_id: {request_id}"
            )
            if timeout is None:
                res = await future
            else:
                # Add 1 second to the timeout to ensure the client can get
                res = await asyncio.wait_for(future, timeout)
            logger_debug(
                f"RPC Client _call_async: Got result for request_id: {request_id}: {res}"
            )
            return res
        except RPCCancelled:
            self._server_stopped = True
            raise
        except asyncio.TimeoutError:
            raise RPCTimeout(
                f"Request '{method_name}' timed out after {timeout}s")
        except Exception as e:
            raise e
        finally:
            self._pending_futures.pop(request_id, None)

    def _ensure_event_loop(self):
        """Ensure we have a running event loop in a background thread."""
        if self._loop is None or not self._loop.is_running():
            self._loop = asyncio.new_event_loop()

            def run_loop():
                asyncio.set_event_loop(self._loop)
                self._stop_event = asyncio.Event()
                self._loop.run_forever()

            self._loop_thread = threading.Thread(target=run_loop,
                                                 daemon=True,
                                                 name="rpc_client_loop")
            self._loop_thread.start()

            # Give the loop a moment to start
            import time
            time.sleep(0.1)

    def _call_sync(self, method_name, *args, **kwargs):
        """Synchronous version of RPC call."""
        logger_debug(
            f"RPC Client calling method: {method_name} with args: {args} and kwargs: {kwargs}"
        )
        self._ensure_event_loop()
        logger_debug(
            f"RPC Client _call_sync: Creating future for {method_name}")
        future = asyncio.run_coroutine_threadsafe(
            self._call_async(method_name, *args, **kwargs), self._loop)
        logger_debug(
            f"RPC Client _call_sync: Waiting for result of {method_name}")
        result = future.result()
        logger_debug(
            f"RPC Client _call_sync: Got result for {method_name}: {result}")
        return result

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

        def _async_to_sync():
            self._ensure_event_loop()
            future = asyncio.run_coroutine_threadsafe(
                self._call_async(name, *args, **kwargs), self._loop)
            return future.result()

        return self._executor.submit(_async_to_sync)

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
        if self._server_stopped:
            raise RPCCancelled("Server is shutting down, request cancelled")

        self._start_response_reader_lazily()
        rpc_params = kwargs.pop("__rpc_params", RPCParams())
        timeout = rpc_params.timeout if rpc_params.timeout is not None else self._timeout

        request_id = uuid.uuid4().hex
        # Use AsyncQueue to ensure proper cross-thread communication
        queue = AsyncQueue()
        # Recreate sync_q with the current running loop for proper cross-thread communication
        # This ensures the background _response_reader thread can properly notify this event loop
        queue._sync_q = _SyncQueue(queue, asyncio.get_running_loop())
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
                logger_debug(f"RPC Client _call_streaming waiting for response",
                             color="green")
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
                    logger_debug(
                        f"RPC Client _call_streaming received data: {response.result}",
                        color="green")
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

    def get_server_attr(self, name: str):
        """ Get the attribute of the RPC server.
        This is mainly used for testing. """
        return self._rpc_get_attr(name).remote()

    def __getattr__(self, name):
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

        def method_caller(*args, **kwargs):
            return RemoteCall(self, name, *args, **kwargs)

        return method_caller

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def __del__(self):
        self.close()
