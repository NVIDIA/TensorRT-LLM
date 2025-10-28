import asyncio
import inspect
import queue
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import zmq

from ...llmapi.utils import ManagedThread, logger_debug
from ...logger import logger
from ..ipc import ZeroMqQueue
from .rpc_common import (RPCCancelled, RPCError, RPCRequest, RPCResponse,
                         RPCStreamingError, RPCTimeout)


class RPCServer:
    """
    An RPC Server that listens for requests and executes them concurrently.
    """

    def __init__(self,
                 instance,
                 hmac_key=None,
                 num_workers: int = 4,
                 timeout: float = 0.5,
                 async_run_task: bool = False):
        """
        Initializes the server with an instance.

        Args:
            instance: The instance whose methods will be exposed via RPC.
            hmac_key (bytes, optional): HMAC key for encryption.
            num_workers (int): Number of worker threads or worker tasks that help parallelize the task execution.
            timeout (int): Timeout for RPC calls.
            async_run_task (bool): Whether to run the task asynchronously.

        NOTE: make num_workers larger if there are some streaming tasks runs infinitely.
        """
        self._instance = instance
        self._hmac_key = hmac_key
        self._num_workers = num_workers
        self._address = None
        self._timeout = timeout
        self._client_socket = None

        # set the stop event to True, and all the workers will exit
        self._stop_event = threading.Event()

        self._num_pending_requests = 0

        self._functions = {
            "_rpc_shutdown": lambda: self.shutdown(is_remote_call=True),
            "_rpc_get_attr": lambda name: self.get_attr(name),
        }
        self._dispatcher_thread: Optional[ManagedThread] = None
        if async_run_task:
            # Increase thread pool size to avoid exhaustion with concurrent operations
            # Use 2x num_workers to handle both request processing and response handling
            self._executor = ThreadPoolExecutor(
                max_workers=num_workers * 2,
                thread_name_prefix="rpc_server_worker")
        else:
            self._executor = None

        self._queue = None

        # Automatically register the instance
        self.register_instance(instance)

        logger_debug(f"RPC Server initialized with {num_workers} workers.",
                     color="green")

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
                                          use_hmac_encryption=False,
                                          socket_type=zmq.ROUTER)
        logger.info(f"RPC Server bound to {self._address}")

    def shutdown(self, is_remote_call: bool = False):
        """Internal method to trigger server shutdown.

        Args:
            is_remote_call: Whether the shutdown is called by a remote call.
                This should be True when client.server_shutdown() is called.
        """
        # NOTE: shutdown is also a remote method, so it could be executed by
        # a thread in a worker executor thread

        if self._stop_event.is_set():
            return

        logger_debug(
            "RPC Server shutdown signal received. Terminating server immediately..."
        )

        # Set the stop event to True, this will trigger immediate shutdown
        self._stop_event.set()

        # Log pending requests that will be cancelled
        logger_debug(
            f"RPC Server shutdown: {self._num_pending_requests} pending requests will be cancelled"
        )

        # Clear the queue and send cancelled errors for all pending requests
        if self._queue is not None:
            self._cancel_pending_queue_requests()

        if not is_remote_call:
            # Block the thread until shutdown is finished

            # 1. Wait for the dispatcher thread to exit, so that no new requests are accepted
            logger_debug(f"RPC Server dispatcher thread joining")
            if self._dispatcher_thread:
                self._dispatcher_thread.join()
                self._dispatcher_thread = None
            logger_debug(f"RPC Server dispatcher thread joined")

            # 2. Shutdown the executor immediately without waiting for tasks
            if self._executor:
                self._executor.shutdown(wait=False)
                self._executor = None

            # 3. (Optionally) Close the client socket, this doesn't affect
            # anything since zmq client will not timeout even if the target is not available
            if self._client_socket:
                self._client_socket.close()
        else:
            # if the shutdown is called by a remote call, this method itself will
            # be executed in a executor thread, so we cannot join the dispatcher thread as
            # the dispatcher thread is awaiting for the shutdown result.
            logger_debug(
                f"RPC Server shutdown initiated: {self._num_pending_requests} pending requests will be cancelled"
            )

    def register_function(self, func, name=None):
        """Exposes a single function to clients."""
        fname = name or func.__name__
        if fname in self._functions:
            logger.warning(
                f"Function '{fname}' is already registered. Overwriting.")
        self._functions[fname] = func
        logger_debug(f"Registered function: {fname}")

    def register_instance(self, instance):
        """Exposes all public methods of a class instance."""
        logger_debug(
            f"Registering instance of class: {instance.__class__.__name__}")
        for name in dir(instance):
            if not name.startswith('_'):
                attr = getattr(instance, name)
                if callable(attr):
                    self.register_function(attr, name)

    def get_attr(self, name: str):
        """ Get the attribute of the RPC server.
        This is mainly used for testing. """
        return getattr(self, name)

    def _cancel_pending_queue_requests(self):
        """Cancel all pending requests in the queue."""

        async def cancel_requests():
            cancelled_count = 0
            while not self._queue.empty():
                try:
                    req: RPCRequest = self._queue.get_nowait()
                    cancelled_count += 1
                    await self._send_error_response(
                        req,
                        RPCCancelled(
                            "Server is shutting down, request cancelled"))
                except asyncio.QueueEmpty:
                    break
                except Exception as e:
                    logger.error(f"Error cancelling pending request: {e}")

            if cancelled_count > 0:
                logger_debug(
                    f"Cancelled {cancelled_count} pending requests in queue")

        # Run the cancellation in the event loop if available
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Schedule the coroutine to run in the existing event loop
                asyncio.create_task(cancel_requests())
            else:
                # Create a new event loop to run the coroutine
                asyncio.run(cancel_requests())
        except RuntimeError:
            # Event loop might not be available in some shutdown scenarios
            pass

    async def _dispatcher_routine(self, stop_event: threading.Event):
        assert self._client_socket is not None, "Client socket is not bound"
        assert self._queue is not None, "RPC queue is not initialized"

        # Once shutdown, the dispatcher will exit first, and the workers will
        # continue to process the pending requests.
        while not stop_event.is_set():
            try:
                req: RPCRequest = await self._client_socket.get_async_noblock(
                    timeout=0.5)
                logger_debug(f"RPC dispatcher got request: {req}")
            except asyncio.TimeoutError:
                await asyncio.sleep(0)
                continue
            except Exception as e:
                logger.error(f"RPC dispatcher caught an exception: {e}")
                logger.error(traceback.format_exc())
                continue

            await self._queue.put(req)  # type: ignore

            # shutdown methods depend on _num_pending_requests, so
            # they should not be counted
            if req.method_name not in ["_rpc_shutdown", "shutdown"]:
                self._num_pending_requests += 1
                logger_debug(
                    f"Dispatcher received request {req}, pending: {self._num_pending_requests}"
                )

    # TODO optimization: resolve the sequential scheduling for the remote calls
    # Suppose tons of submit remote call block the FIFO queue, and the later get_stats remote calls may be blocked
    # There could be two dispatch modes:
    # 1. (current) mix mode, share the same routine/pool
    # 2. (promising) stream mode, specific remote_call -> stream -> specific routine/pool
    #    - get_stats() - 1, remote_call -> dedicated queue -> dedicated routine/pool
    #    - submit() - 3 -> dedicated queue -> dedicated routine/pool
    # TODO potential optimization: for submit(), batch the ad-hoc requests in an interval like 5ms, reduce the IPC count
    async def _send_error_response(self, req: RPCRequest,
                                   error: Exception) -> None:
        """Send an error response for a request."""
        if not req.need_response:
            return

        if req.is_streaming:
            await self._client_socket.put_async(
                RPCResponse(
                    req.request_id,
                    None,
                    error,
                    is_streaming=
                    True,  # Important: mark as streaming so it gets routed correctly
                    stream_status='error'))
        else:
            await self._client_socket.put_async(
                RPCResponse(req.request_id, None, error))

    async def _handle_shutdown_request(self, req: RPCRequest) -> bool:
        """Handle a request during shutdown. Returns True if handled."""
        if not self._stop_event.is_set():
            return False

        # Allow shutdown methods to proceed
        if req.method_name in ["_rpc_shutdown", "shutdown"]:
            return False

        # Send cancellation error for all other requests
        await self._send_error_response(
            req, RPCCancelled("Server is shutting down, request cancelled"))

        # Decrement pending count
        self._num_pending_requests -= 1
        return True

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
                await asyncio.sleep(0)
                continue

            # Check if we should cancel due to shutdown
            if await self._handle_shutdown_request(req):
                continue

            # Check if the method exists
            if req.method_name not in self._functions:
                logger.error(
                    f"Method '{req.method_name}' not found in RPC server.")
                self._num_pending_requests -= 1

                error = RPCStreamingError if req.is_streaming else RPCError
                await self._send_error_response(
                    req,
                    error(
                        f"Method '{req.method_name}' not found in RPC server.",
                        traceback=traceback.format_exc()))
                continue

            func = self._functions[req.method_name]

            # Final shutdown check before processing
            if await self._handle_shutdown_request(req):
                continue

            # Process the request
            if req.is_streaming:
                if inspect.isasyncgenfunction(func):
                    await self._process_streaming_request(req)
                else:
                    # Non-streaming function called with streaming flag
                    await self._send_error_response(
                        req,
                        RPCStreamingError(
                            f"Method '{req.method_name}' is not a streaming function."
                        ))
            else:
                # Process regular request
                response = await self._process_request(req)

                # Send response if needed
                if req.need_response and response is not None:
                    logger_debug(
                        f"RPC Server sending response for request {req}, pending: {self._num_pending_requests}"
                    )
                    if await self._send_response(req, response):
                        logger_debug(
                            f"RPC Server sent response for request {req}")

            # Decrement pending count
            if req.method_name not in ["_rpc_shutdown", "shutdown"]:
                self._num_pending_requests -= 1

    def _calculate_adjusted_timeout(self,
                                    req: RPCRequest,
                                    is_streaming: bool = False) -> float:
        """Calculate adjusted timeout based on pending overhead.

        Args:
            req: The RPC request
            is_streaming: Whether this is for a streaming request

        Returns:
            The adjusted timeout value
        """
        adjusted_timeout = req.timeout
        if req.creation_timestamp is not None and req.timeout is not None and req.timeout > 0:
            pending_time = time.time() - req.creation_timestamp
            adjusted_timeout = max(0.1, req.timeout -
                                   pending_time)  # Keep at least 0.1s timeout
            if pending_time > 0.1:  # Only log if significant pending time
                method_type = "streaming " if is_streaming else ""
                logger_debug(
                    f"RPC Server adjusted timeout for {method_type}{req.method_name}: "
                    f"original={req.timeout}s, pending={pending_time:.3f}s, adjusted={adjusted_timeout:.3f}s"
                )
        return adjusted_timeout

    async def _process_request(self, req: RPCRequest) -> Optional[RPCResponse]:
        """Process a request. Returns None for streaming requests (handled separately)."""
        func = self._functions[req.method_name]

        # Calculate adjusted timeout based on pending overhead
        adjusted_timeout = self._calculate_adjusted_timeout(req)

        try:
            if inspect.iscoroutinefunction(func):
                # Execute async function directly in event loop, no need to run in executor due to the GIL
                logger_debug(
                    f"RPC Server running async task {req.method_name} in dispatcher"
                )
                result = await asyncio.wait_for(func(*req.args, **req.kwargs),
                                                timeout=adjusted_timeout)
            else:
                # Execute sync function in thread executor
                loop = asyncio.get_running_loop()

                def call_with_kwargs():
                    return func(*req.args, **req.kwargs)

                logger_debug(
                    f"RPC Server running async task {req.method_name} in worker"
                )
                # TODO: let num worker control the pool size
                result = await asyncio.wait_for(loop.run_in_executor(
                    self._executor, call_with_kwargs),
                                                timeout=adjusted_timeout)

            logger_debug(f"RPC Server returned result for request {req}")
            response = RPCResponse(req.request_id, result)

        except asyncio.TimeoutError:
            response = RPCResponse(
                req.request_id, None,
                RPCTimeout(
                    f"Method '{req.method_name}' timed out after {req.timeout} seconds",
                    traceback=traceback.format_exc()))

        except Exception as e:
            response = RPCResponse(
                req.request_id, None,
                RPCError(str(e), cause=e, traceback=traceback.format_exc()))

        return response

    async def _process_streaming_request(self, req: RPCRequest):
        """Process a streaming request by sending multiple responses."""
        func = self._functions[req.method_name]

        if not inspect.isasyncgenfunction(func):
            await self._client_socket.put_async(
                RPCResponse(
                    req.request_id,
                    None,
                    RPCStreamingError(
                        f"Method '{req.method_name}' is not an async generator.",
                        traceback=traceback.format_exc()),
                    is_streaming=
                    True,  # Important: mark as streaming so it gets routed correctly
                    # need to redirect the error to the client's streaming queue
                    stream_status='error'))
            return

        sequence_number = 0

        # Calculate adjusted timeout based on pending overhead
        adjusted_timeout = self._calculate_adjusted_timeout(req,
                                                            is_streaming=True)

        try:
            logger_debug(f"RPC Server running streaming task {req.method_name}")
            # Send start signal
            await self._client_socket.put_async(
                RPCResponse(req.request_id, None, None, True, sequence_number,
                            'start'))
            sequence_number += 1

            # Apply timeout to the entire streaming operation if specified
            if adjusted_timeout is not None and adjusted_timeout > 0:
                # Create a task for the async generator with timeout
                async def stream_with_timeout():
                    nonlocal sequence_number
                    async for result in func(*req.args, **req.kwargs):
                        # Check if shutdown was triggered
                        if self._stop_event.is_set():
                            raise RPCCancelled(
                                "Server is shutting down, streaming cancelled")

                        logger_debug(
                            f"RPC Server got data and ready to send result {result}"
                        )
                        response = RPCResponse(req.request_id, result, None,
                                               True, sequence_number, 'data')
                        if not await self._send_response(req, response):
                            # Stop streaming after a pickle error
                            return
                        sequence_number += 1

                # Use wait_for for timeout handling
                await asyncio.wait_for(stream_with_timeout(),
                                       timeout=adjusted_timeout)
            else:
                # No timeout specified, stream normally
                async for result in func(*req.args, **req.kwargs):
                    # Check if shutdown was triggered
                    if self._stop_event.is_set():
                        raise RPCCancelled(
                            "Server is shutting down, streaming cancelled")

                    logger_debug(
                        f"RPC Server got data and ready to send result {result}"
                    )
                    response = RPCResponse(req.request_id, result, None, True,
                                           sequence_number, 'data')
                    if not await self._send_response(req, response):
                        # Stop streaming after a pickle error
                        return
                    sequence_number += 1

            # Send end signal
            await self._client_socket.put_async(
                RPCResponse(req.request_id, None, None, True, sequence_number,
                            'end'))

        except RPCCancelled as e:
            # Server is shutting down, send cancelled error
            await self._client_socket.put_async(
                RPCResponse(req.request_id, None, e, True, sequence_number,
                            'error'))

        except asyncio.TimeoutError:
            await self._client_socket.put_async(
                RPCResponse(
                    req.request_id, None,
                    RPCTimeout(
                        f"Streaming method '{req.method_name}' timed out",
                        traceback=traceback.format_exc()), True,
                    sequence_number, 'error'))

        except Exception as e:
            response = RPCResponse(
                req.request_id, None,
                RPCStreamingError(str(e), traceback=traceback.format_exc()),
                True, sequence_number, 'error')
            await self._send_response(req, response)

    async def _send_response(self, req: RPCRequest,
                             response: RPCResponse) -> bool:
        """Safely sends a response, handling pickle errors."""
        try:
            await self._client_socket.put_async(response)
            return True
        except Exception as e:
            logger.error(
                f"Failed to pickle response for request {req.request_id}: {e}")
            error_msg = f"Failed to pickle response: {e}"
            if req.is_streaming:
                error_cls = RPCStreamingError
                # For streaming, we also need sequence number. The original response has it.
                sequence_number = response.sequence_number if response else None
                error_response = RPCResponse(
                    req.request_id,
                    None,
                    error_cls(error_msg, traceback=traceback.format_exc()),
                    is_streaming=True,
                    sequence_number=sequence_number,
                    stream_status='error')
            else:
                error_cls = RPCError
                error_response = RPCResponse(
                    req.request_id, None,
                    error_cls(error_msg, traceback=traceback.format_exc()))

            try:
                await self._client_socket.put_async(error_response)
            except Exception as e_inner:
                logger.error(
                    f"Failed to send error response for request {req.request_id}: {e_inner}"
                )
            return False

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
