import asyncio
import concurrent.futures
import threading
import time
import uuid
from typing import Any, AsyncIterator, Dict, Optional

import zmq

from tensorrt_llm._utils import (customized_gc_thresholds, nvtx_mark_debug,
                                 nvtx_range_debug)

from ...llmapi.utils import (AsyncQueue, _SyncQueue, enable_llmapi_debug,
                             logger_debug)
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
                                          use_hmac_encryption=False,
                                          socket_type=zmq.DEALER)
        self._pending_futures = {}
        # map request_id to the queue for streaming responses
        self._streaming_queues: Dict[str, AsyncQueue] = {}
        self._reader_task = None
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=num_workers, thread_name_prefix="rpc_client_worker")

        self._server_stopped = False
        self._closed = False
        self._loop = None
        self._loop_thread = None
        self._reader_asyncio_task = None  # Track the asyncio task for proper cancellation

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
        self._closed = True

        logger_debug("RPC Client closing")

        # Cancel the reader task first to avoid socket closure errors
        if self._reader_task and not self._reader_task.done():
            if self._loop and self._loop.is_running(
            ) and self._reader_asyncio_task:
                try:
                    # Cancel the asyncio task in its event loop
                    async def cancel_reader_task():
                        if self._reader_asyncio_task and not self._reader_asyncio_task.done(
                        ):
                            self._reader_asyncio_task.cancel()
                            try:
                                await self._reader_asyncio_task
                            except asyncio.CancelledError:
                                pass  # Expected

                    cancel_future = asyncio.run_coroutine_threadsafe(
                        cancel_reader_task(), self._loop)
                    cancel_future.result(timeout=2.0)
                    logger_debug("Reader task cancelled successfully")
                except concurrent.futures.TimeoutError:
                    logger.warning("Reader task did not exit gracefully")
                except Exception as e:
                    logger_debug(f"Reader task cleanup: {e}")
            self._reader_task = None
            self._reader_asyncio_task = None

        # Now close the socket after reader has stopped
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

    def _handle_streaming_response(self, response: RPCResponse):
        """Handle a streaming response by putting it in the appropriate queue.

        Args:
            response: The streaming response to handle
        """
        assert response.stream_status in [
            'start', 'data', 'end', 'error'
        ], f"Invalid stream status: {response.stream_status}"

        queue = self._streaming_queues.get(response.request_id)
        if queue:
            # put to the sync queue, as the current event loop is
            # different from the one in call_async or call_streaming
            assert isinstance(queue, AsyncQueue)
            if enable_llmapi_debug() or logger.level == 'debug':
                logger_debug(
                    f"RPC Client putting response to AsyncQueue: status={response.stream_status}, request_id={response.request_id}"
                )
            queue.sync_q.put(response)
            # Clean up if stream ended
            if response.stream_status in ['end', 'error']:
                self._streaming_queues.pop(response.request_id, None)

    def _handle_regular_response(self, response: RPCResponse):
        """Handle a regular (non-streaming) response by setting the future result.

        Args:
            response: The response to handle
        """
        if future_info := self._pending_futures.get(response.request_id):
            future, target_loop = future_info

            if not future.done():

                def safe_set_result():
                    """Safely set result on future, handling race conditions."""
                    try:
                        if not future.done():
                            if response.error is None:
                                future.set_result(response.result)
                            else:
                                future.set_exception(response.error)
                    except asyncio.InvalidStateError:
                        # Future was cancelled or completed between the check and set
                        # This is expected in high-load scenarios, just log and continue
                        if enable_llmapi_debug() or logger.level == 'debug':
                            logger_debug(
                                f"Future already done for request_id: {response.request_id}, skipping"
                            )

                if enable_llmapi_debug() or logger.level == 'debug':
                    if response.error is None:
                        logger_debug(
                            f"Setting result for request_id: {response.request_id}"
                        )
                    else:
                        logger_debug(
                            f"Setting exception for request_id: {response.request_id}, error: {response.error}"
                        )

                target_loop.call_soon_threadsafe(safe_set_result)
        else:
            if enable_llmapi_debug() or logger.level == 'debug':
                logger_debug(
                    f"No future found for request_id: {response.request_id}")

        self._pending_futures.pop(response.request_id, None)

    async def _handle_reader_exception(self, exception: Exception):
        """Propagate an exception to all pending futures and streaming queues.

        Args:
            exception: The exception to propagate
        """
        logger.error(f"Exception in RPC response reader: {exception}")

        # Propagate exception to all pending futures
        for (future, target_loop) in self._pending_futures.values():
            if not future.done():

                def safe_set_exception(f=future, exc=exception):
                    """Safely set exception on future, handling race conditions."""
                    try:
                        if not f.done():
                            f.set_exception(exc)
                    except asyncio.InvalidStateError:
                        # Future was cancelled or completed, this is fine
                        pass

                target_loop.call_soon_threadsafe(safe_set_exception)

        # Also signal error to streaming queues
        for queue in self._streaming_queues.values():
            await queue.put(RPCResponse("", None, exception, False, 0, 'error'))

    async def _wait_for_response(self) -> RPCResponse:
        """Wait for a response from the socket.

        Returns:
            RPCResponse from the server
        """
        # Directly await the socket - cancellation will be handled by task cancellation
        return await self._client_socket.get_async()

    async def _response_reader(self):
        """Task to read responses from the socket and set results on futures."""
        logger_debug("Response reader started")

        try:
            with customized_gc_thresholds(10000):
                while True:
                    with nvtx_range_debug("response_reader",
                                          color="cyan",
                                          category="RPC"):
                        try:
                            response = await self._wait_for_response()

                            nvtx_mark_debug(
                                f"RPC.response.{'streaming' if response.is_streaming else 'sync'}",
                                color="black",
                                category="RPC")

                            # Optimize: Check debug flag before expensive string operations
                            # This avoids holding GIL for f-string evaluation when debug is disabled
                            if enable_llmapi_debug() or logger.level == 'debug':
                                logger_debug(
                                    f"RPC Client received response: request_id={response.request_id}, "
                                    f"is_streaming={response.is_streaming}, "
                                    f"pending_futures={len(self._pending_futures)}"
                                )

                            with nvtx_range_debug("handle_response",
                                                  color="purple",
                                                  category="RPC"):
                                if response.is_streaming:
                                    self._handle_streaming_response(response)
                                else:
                                    self._handle_regular_response(response)

                        except Exception as e:
                            await self._handle_reader_exception(e)
                            break

        except asyncio.CancelledError:
            logger_debug("Response reader cancelled")
        finally:
            logger_debug("Response reader exiting gracefully")
            self._reader_task = None
            self._reader_asyncio_task = None

    def _start_response_reader_lazily(self):
        if self._reader_task is None or self._reader_task.done():
            # Ensure we have a persistent background loop
            self._ensure_event_loop()

            # Wrapper to track the asyncio task
            async def run_reader():
                self._reader_asyncio_task = asyncio.current_task()
                await self._response_reader()

            # Start the reader task on the persistent loop
            future = asyncio.run_coroutine_threadsafe(run_reader(), self._loop)
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
        if enable_llmapi_debug() or logger.level == 'debug':
            logger_debug(f"RPC client calling method: {method_name}")
        nvtx_mark_debug(f"RPC.async.{method_name}",
                        color="yellow",
                        category="RPC")
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
        await self._client_socket.put_async(request)

        if not need_response:
            return None

        loop = asyncio.get_running_loop()
        future = loop.create_future()
        self._pending_futures[request_id] = (future, loop)

        try:
            # If timeout, the remote call should return a timeout error timely,
            # so we add 1 second to the timeout to ensure the client can get
            # that result.
            if timeout is None:
                res = await future
            else:
                # Add 1 second to the timeout to ensure the client can get
                res = await asyncio.wait_for(future, timeout)
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

            # TODO: WAR. Remove after RPC shutdown is fixed.
            def custom_exception_handler(loop, context):
                exception = context.get('exception')
                message = context.get('message', '')

                if isinstance(exception,
                              asyncio.CancelledError) or "pending" in message:
                    logger.debug(f"Suppressed error during shutdown: {message}")
                    return

                loop.default_exception_handler(context)

            self._loop.set_exception_handler(custom_exception_handler)

            def run_loop():
                asyncio.set_event_loop(self._loop)
                self._loop.run_forever()

            self._loop_thread = threading.Thread(target=run_loop,
                                                 daemon=True,
                                                 name="rpc_client_loop")
            self._loop_thread.start()

            # Give the loop a moment to start
            time.sleep(0.1)

    def _call_sync(self, method_name, *args, **kwargs):
        """Synchronous version of RPC call."""
        if enable_llmapi_debug() or logger.level == 'debug':
            logger_debug(f"RPC Client calling method: {method_name}")
        nvtx_mark_debug(f"RPC.sync.{method_name}",
                        color="green",
                        category="RPC")
        self._ensure_event_loop()
        future = asyncio.run_coroutine_threadsafe(
            self._call_async(method_name, *args, **kwargs), self._loop)
        result = future.result()
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
        nvtx_mark_debug(f"RPC.future.{name}", color="blue", category="RPC")

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
        nvtx_mark_debug(f"RPC.streaming.{name}", color="red", category="RPC")

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
                if timeout is None:
                    response = await queue.get()
                else:
                    response = await asyncio.wait_for(queue.get(),
                                                      timeout=timeout)

                if enable_llmapi_debug() or logger.level == 'debug':
                    logger_debug(
                        f"RPC Client _call_streaming received [{response.stream_status}] response",
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
