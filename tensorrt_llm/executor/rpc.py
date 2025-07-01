import asyncio
import concurrent.futures
import queue
import threading
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any, NamedTuple, Optional

from ..llmapi.utils import ManagedThread
from ..logger import logger
from .ipc import ZeroMqQueue


# --- Custom Exceptions ---
class RPCError(Exception):
    """Custom exception for RPC-related errors raised on the client side."""


class RPCTimeout(RPCError):
    """Custom exception for when a client request times out."""


class RPCRequest(NamedTuple):
    request_id: str
    method_name: str
    args: tuple
    kwargs: dict
    need_response: bool = True
    timeout: float = 0.5


class RPCResponse(NamedTuple):
    request_id: str
    status: str
    result: Any


class RPCServer:
    """
    An RPC Server that listens for requests and executes them concurrently.
    """

    def __init__(self,
                 instance,
                 hmac_key=None,
                 num_workers: int = 1,
                 timeout: float = 0.5,
                 async_run_task: bool = False):
        """
        Initializes the server with an instance.

        Args:
            instance: The instance whose methods will be exposed via RPC.
            hmac_key (bytes, optional): HMAC key for encryption.
            num_workers (int): Number of worker threads.
            timeout (int): Timeout for RPC calls.
            async_run_task (bool): Whether to run the task asynchronously.
        """
        self._instance = instance
        self._hmac_key = hmac_key
        self._num_workers = num_workers
        self._address = None
        self._timeout = timeout
        self._client_socket = None

        # set the stop event to True, and all the workers will exit
        self._stop_event = threading.Event()

        self._functions = {"shutdown": self.shutdown}
        self._dispatcher_thread: Optional[ManagedThread] = None
        if async_run_task:
            self._executor = ThreadPoolExecutor(max_workers=num_workers,
                                                thread_name_prefix="rpc_worker")
        else:
            self._executor = None

        self._queue = None

        # Automatically register the instance
        self.register_instance(instance)

        logger.debug(f"RPC Server initialized with {num_workers} workers.")

    @property
    def address(self) -> str:
        assert self._client_socket is not None, "Client socket is not bound"
        return self._client_socket.address[0]

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.shutdown()

    def bind(self, address="tcp://*:5555"):
        """
        Bind the server to the specified address.

        Args:
            address (str): The ZMQ address to bind the client-facing socket.
        """
        self._address = address
        self._client_socket = ZeroMqQueue(address=(address, self._hmac_key),
                                          is_server=True,
                                          is_async=True,
                                          use_hmac_encryption=False)
        logger.info(f"RPC Server bound to {self._address}")

    def shutdown(self):
        """Internal method to trigger server shutdown."""
        logger.debug(
            "RPC Server shutdown signal received. Terminating server...")

        if self._dispatcher_thread and self._dispatcher_thread.is_alive():
            self._stop_event.set()
            self._dispatcher_thread.join()
            self._dispatcher_thread = None

        if self._executor:
            self._executor.shutdown(wait=False)

        if self._client_socket:
            self._client_socket.close()

        self._client_socket = None
        self._queue = None

    def register_function(self, func, name=None):
        """Exposes a single function to clients."""
        fname = name or func.__name__
        if fname in self._functions:
            logger.warning(
                f"Function '{fname}' is already registered. Overwriting.")
        self._functions[fname] = func
        logger.debug(f"Registered function: {fname}")

    def register_instance(self, instance):
        """Exposes all public methods of a class instance."""
        logger.debug(
            f"Registering instance of class: {instance.__class__.__name__}")
        for name in dir(instance):
            if not name.startswith('_'):
                attr = getattr(instance, name)
                if callable(attr):
                    self.register_function(attr, name)

    async def _dispatcher_routine(self, stop_event: threading.Event):
        assert self._client_socket is not None, "Client socket is not bound"
        assert self._queue is not None, "RPC queue is not initialized"

        while not stop_event.is_set():
            try:
                req: RPCRequest = await self._client_socket.get_async_noblock(
                    timeout=0.5)
                logger.debug(f"RPC dispatcher got request: {req}")
            except asyncio.TimeoutError:
                logger.debug("RPC dispatcher get request timeout")
                continue

            await self._queue.put(req)  # type: ignore

    async def _worker_routine(self, stop_event: threading.Event):
        """The routine executed by each worker thread."""
        assert self._client_socket is not None, "Client socket is not bound"
        assert self._queue is not None, "RPC queue is not initialized"

        while not stop_event.is_set():
            try:
                req: RPCRequest = await asyncio.wait_for(
                    self._queue.get(),  # type: ignore
                    timeout=self._timeout)
            except asyncio.TimeoutError:
                logger.debug("RPC worker get request timeout")
                continue

            if req.method_name in self._functions:
                try:
                    if self._executor is not None:
                        # Dispatch to worker thread and await result with timeout
                        loop = asyncio.get_running_loop()

                        # Create a wrapper function to handle keyword arguments
                        def call_with_kwargs():
                            return self._functions[req.method_name](
                                *req.args, **req.kwargs)

                        result = await asyncio.wait_for(loop.run_in_executor(
                            self._executor, call_with_kwargs),
                                                        timeout=req.timeout)
                    else:
                        # For synchronous execution, we need to run in executor to support timeout
                        loop = asyncio.get_running_loop()

                        # Create a wrapper function to handle keyword arguments
                        def call_with_kwargs():
                            return self._functions[req.method_name](
                                *req.args, **req.kwargs)

                        result = await asyncio.wait_for(loop.run_in_executor(
                            None, call_with_kwargs),
                                                        timeout=req.timeout)
                    response = RPCResponse(req.request_id, 'OK', result)
                except asyncio.TimeoutError:
                    response = RPCResponse(
                        req.request_id, 'ERROR',
                        f"Method '{req.method_name}' timed out after {req.timeout} seconds"
                    )
                except Exception:
                    tb = traceback.format_exc()
                    response = RPCResponse(req.request_id, 'ERROR', tb)
            else:
                response = RPCResponse(
                    req.request_id, 'ERROR',
                    f"Method '{req.method_name}' not found.")

            # Some tasks don't need response, e.g. submit_request or shutdown
            if req.need_response:
                await self._client_socket.put_async(response)

    def start(self):
        """Binds sockets, starts workers, and begins proxying messages."""
        if self._client_socket is None:
            raise RuntimeError(
                "Server must be bound to an address before starting. Call bind() first."
            )

        self._client_socket.setup_lazily()
        logger.info(f"RPC Server started and listening on {self._address}")

        async def tasks():
            self._queue = asyncio.Queue()
            await asyncio.gather(
                self._dispatcher_routine(self._stop_event), *[
                    self._worker_routine(self._stop_event)
                    for i in range(self._num_workers)
                ])

        def loop() -> bool:
            asyncio.run(tasks())
            return True  # ManagedThread

        error_queue = queue.Queue()
        self._dispatcher_thread = ManagedThread(task=loop,
                                                stop_event=self._stop_event,
                                                name="rpc_dispatcher_thread",
                                                error_queue=error_queue)
        self._dispatcher_thread.start()

        logger.info("RPC Server has started.")


Server = RPCServer


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
        logger.info(f"RPC Client initialized. Connected to {self._address}")

    def __del__(self):
        """Cleanup executor when client is destroyed."""
        self.close()

    def close(self):
        """Gracefully close the client, cleaning up background tasks."""
        if self._reader_task:
            self._reader_task.cancel()
            self._reader_task = None
        if self._executor:
            self._executor.shutdown(wait=False)

    async def _response_reader(self):
        """Task to read responses from the socket and set results on futures."""

        while True:
            try:
                response: RPCResponse = await self._client_socket.get_async()
                future = self._pending_futures.get(response.request_id)
                if future and not future.done():
                    if response.status == 'OK':
                        future.set_result(response.result)
                    elif response.status == 'ERROR':
                        # TODO: Maybe keep the original Error type?
                        future.set_exception(
                            RPCError(
                                f"Server-side exception:\n{response.result}"))
                    else:
                        future.set_exception(
                            RPCError(
                                f"Unknown response status: {response.status}"))
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

    async def _start_reader_if_needed(self):
        if self._reader_task is None or self._reader_task.done():
            loop = asyncio.get_running_loop()
            self._reader_task = loop.create_task(self._response_reader())

    async def _call_async(self, name, *args, **kwargs):
        """Async version of RPC call.
        Args:
            name: Method name to call
            *args: Positional arguments
            **kwargs: Keyword arguments
            __rpc_timeout: The timeout (seconds) for the RPC call.
            __rpc_need_response: Whether the RPC call needs a response.
                If set to False, the remote call will return immediately.

        Returns:
            The result of the remote method call
        """
        logger.debug(
            f"RPC client calling method: {name} with args: {args} and kwargs: {kwargs}"
        )
        await self._start_reader_if_needed()
        need_response = kwargs.pop("__rpc_need_response", True)
        timeout = kwargs.pop("__rpc_timeout", self._timeout)

        request_id = uuid.uuid4().hex
        logger.debug(f"RPC client sending request: {request_id}")
        request = RPCRequest(request_id,
                             name,
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
            return await asyncio.wait_for(future, timeout + 1)
        except asyncio.TimeoutError:
            raise RPCTimeout(f"Request '{name}' timed out after {timeout}s")
        finally:
            self._pending_futures.pop(request_id, None)

    def _call_sync(self, name, *args, **kwargs):
        """Synchronous version of RPC call."""
        return asyncio.run(self._call_async(name, *args, **kwargs))

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
            return asyncio.run(self._call_async(name, *args, **kwargs))

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
                return self.client._call_sync(self.method_name, *args, **kwargs)

            def call_async(self, *args, **kwargs):
                """Async call - returns coroutine"""
                return self.client._call_async(self.method_name, *args,
                                               **kwargs)

            def call_future(self, *args, **kwargs) -> concurrent.futures.Future:
                """Future call - returns Future object"""
                return self.client.call_future(self.method_name, *args,
                                               **kwargs)

        return MethodProxy(self, name)
