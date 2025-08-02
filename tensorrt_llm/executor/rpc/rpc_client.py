import asyncio
import concurrent.futures
import threading
import uuid

from ...logger import logger
from ..ipc import ZeroMqQueue
from .rpc_common import RPCCancelled, RPCRequest, RPCResponse, RPCTimeout


class RPCClient:
    """
    An RPC Client that connects to the RPCServer.
    """

    def __init__(self,
                 address: str,
                 hmac_key=None,
                 timeout: float = 10,
                 num_workers: int = 4):
        '''
        Args:
            address: The ZMQ address to connect to.
            hmac_key: The HMAC key for encryption.
            timeout: The timeout (seconds) for RPC calls.
        '''
        self._address = address
        self._timeout = timeout
        self._client_socket = ZeroMqQueue(address=(address, hmac_key),
                                          is_server=False,
                                          is_async=True,
                                          use_hmac_encryption=False)
        self._pending_futures = {}
        self._reader_task = None
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=num_workers, thread_name_prefix="rpc_client")

        self._server_stopped = False
        self._loop = None
        self._loop_thread = None

        logger.debug(f"RPC Client initialized. Connected to {self._address}")

    def shutdown_server(self):
        """Shutdown the server."""
        if self._server_stopped:
            return

        self.call_sync("__rpc_shutdown")

        self._server_stopped = True

    def close(self):
        """Gracefully close the client, cleaning up background tasks."""
        if self._reader_task:
            self._reader_task.cancel()
            self._reader_task = None
        if self._loop and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._loop_thread:
            self._loop_thread.join()
            self._loop_thread = None
        if self._executor:
            self._executor.shutdown(wait=True)

    async def _response_reader(self):
        """Task to read responses from the socket and set results on futures."""

        while True:
            try:
                response: RPCResponse = await self._client_socket.get_async()
                logger.debug(f"RPC Client received response: {response}")
                future = self._pending_futures.get(response.request_id)
                if future and not future.done():
                    if response.error is None:
                        future.set_result(response.result)
                    else:
                        # Use the original RPCError from the response
                        future.set_exception(response.error)
                self._pending_futures.pop(response.request_id, None)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Exception in RPC response reader: {e}")
                # Propagate exception to all pending futures
                for future in self._pending_futures.values():
                    if not future.done():
                        future.set_exception(e)
                break

            await asyncio.sleep(0)

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

    async def _call_async(self, __rpc_method_name, *args, **kwargs):
        """Async version of RPC call.
        Args:
            __rpc_method_name: Method name to call
            *args: Positional arguments
            **kwargs: Keyword arguments
            __rpc_timeout: The timeout (seconds) for the RPC call.
            __rpc_need_response: Whether the RPC call needs a response.
                If set to False, the remote call will return immediately.

        Returns:
            The result of the remote method call
        """
        logger.debug(
            f"RPC client calling method: {__rpc_method_name} with args: {args} and kwargs: {kwargs}"
        )
        if self._server_stopped:
            raise RPCCancelled("Server is shutting down, request cancelled")

        self._start_response_reader_lazily()
        need_response = kwargs.pop("__rpc_need_response", True)
        timeout = kwargs.pop("__rpc_timeout", self._timeout)

        request_id = uuid.uuid4().hex
        logger.debug(f"RPC client sending request: {request_id}")
        request = RPCRequest(request_id,
                             __rpc_method_name,
                             args,
                             kwargs,
                             need_response,
                             timeout=timeout)
        logger.debug(f"RPC client sending request: {request}")
        await self._client_socket.put_async(request)

        if not need_response:
            return None

        loop = asyncio.get_running_loop()
        future = loop.create_future()
        self._pending_futures[request_id] = future

        try:
            # If timeout, the remote call should return a timeout error timely,
            # so we add 1 second to the timeout to ensure the client can get
            # that result.
            res = await asyncio.wait_for(future, timeout + 1)
            return res
        except RPCCancelled:
            self._server_stopped = True
            raise
        except asyncio.TimeoutError:
            raise RPCTimeout(
                f"Request '{__rpc_method_name}' timed out after {timeout}s")
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
                self._loop.run_forever()

            self._loop_thread = threading.Thread(target=run_loop, daemon=True)
            self._loop_thread.start()

            # Give the loop a moment to start
            import time
            time.sleep(0.1)

    def _call_sync(self, __rpc_method_name, *args, **kwargs):
        """Synchronous version of RPC call."""
        self._ensure_event_loop()
        future = asyncio.run_coroutine_threadsafe(
            self._call_async(__rpc_method_name, *args, **kwargs), self._loop)
        return future.result()

    def call_async(self, name: str, *args, **kwargs):
        """
        Call a remote method asynchronously.

        Args:
            name: Method name to call
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Coroutine that can be awaited

        Example:
            result = await client.call_async('remote_method', arg1, arg2, key=value)
        """
        return self._call_async(name, *args, **kwargs, __rpc_need_response=True)

    def call_future(self, name: str, *args,
                    **kwargs) -> concurrent.futures.Future:
        """
        Call a remote method and return a Future.

        Args:
            name: Method name to call
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            A Future object that can be used to retrieve the result

        Example:
            future = client.call_future('remote_method', arg1, arg2, key=value)
            result = future.result()  # blocks until complete
            # or
            future.add_done_callback(lambda f: print(f.result()))
        """

        def _async_to_sync():
            self._ensure_event_loop()
            future = asyncio.run_coroutine_threadsafe(
                self._call_async(name, *args, **kwargs), self._loop)
            return future.result()

        return self._executor.submit(_async_to_sync)

    def call_sync(self, name: str, *args, **kwargs):
        """
        Call a remote method synchronously (blocking).

        Args:
            name: Method name to call
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            The result of the remote method call

        Example:
            result = client.call_sync('remote_method', arg1, arg2, key=value)
        """
        return self._call_sync(name, *args, **kwargs)

    def get_server_attr(self, name: str):
        """ Get the attribute of the RPC server.
        This is mainly used for testing. """
        return self._call_sync("__rpc_get_attr", name, __rpc_timeout=10)

    def __getattr__(self, name):
        """
        Magically handles calls to non-existent methods.
        Returns a proxy object that supports multiple calling patterns.
        """

        class MethodProxy:

            def __init__(self, client, method_name):
                self.client = client
                self.method_name = method_name

            def __call__(self, *args, **kwargs):
                """Default synchronous call"""
                mode = kwargs.pop("__rpc_mode", "sync")
                if mode == "sync":
                    return self.client._call_sync(self.method_name, *args,
                                                  **kwargs)
                elif mode == "async":
                    return self.client._call_async(self.method_name, *args,
                                                   **kwargs)
                elif mode == "future":
                    return self.client.call_future(self.method_name, *args,
                                                   **kwargs)
                else:
                    raise ValueError(f"Invalid RPC mode: {mode}")

            def call_async(self, *args, **kwargs):
                """Async call - returns coroutine"""
                return self.client._call_async(self.method_name, *args,
                                               **kwargs)

            def call_future(self, *args, **kwargs) -> concurrent.futures.Future:
                """Future call - returns Future object"""
                return self.client.call_future(self.method_name, *args,
                                               **kwargs)

        return MethodProxy(self, name)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def __del__(self):
        self.close()
