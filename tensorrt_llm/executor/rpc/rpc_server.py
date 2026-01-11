import asyncio
import inspect
import os
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, List, Optional

import zmq

from ...llmapi.utils import logger_debug
from ...logger import logger
from ..ipc import ZeroMqQueue
from .rpc_common import (RPCCancelled, RPCError, RPCRequest, RPCResponse,
                         RPCStreamingError, RPCTimeout)


class RPCServer:
    """
    An RPC Server that listens for requests and executes them concurrently.
    """

    def __init__(self,
                 instance: Any,
                 hmac_key: Optional[bytes] = None,
                 num_workers: int = 4,
                 timeout: float = 0.5,
                 async_run_task: bool = False) -> None:
        """
        Initializes the server with an instance.

        Args:
            instance: The instance whose methods will be exposed via RPC.
            hmac_key (bytes, optional): HMAC key for encryption.
            num_workers (int): Number of worker threads or worker tasks that help parallelize the task execution.
            timeout (int): Timeout for RPC calls.
            async_run_task (bool): Whether to run the task asynchronously.

        NOTE: make num_workers larger or the remote() and remote_future() may
        be blocked by the thread pool.
        """
        self._instance = instance
        self._hmac_key = hmac_key
        self._num_workers = num_workers
        self._address = None
        self._timeout = timeout
        self._client_socket = None

        # Asyncio components
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._main_task: Optional[asyncio.Task] = None
        self._worker_tasks: List[asyncio.Task] = []
        self._shutdown_event: Optional[asyncio.Event] = None
        self._server_thread: Optional[threading.Thread] = None

        self._stop_event: threading.Event = threading.Event(
        )  # for thread-safe shutdown

        self._num_pending_requests = 0

        self._functions: Dict[str, Callable[..., Any]] = {
            # Some built-in methods for RPC server
            "_rpc_shutdown": lambda: self.shutdown(is_remote_call=True),
            "_rpc_get_attr": lambda name: self.get_attr(name),
        }

        if async_run_task:
            self._executor = ThreadPoolExecutor(
                max_workers=num_workers, thread_name_prefix="rpc_server_worker")
        else:
            self._executor = None

        self.register_instance(instance)

        logger_debug(
            f"[server] RPCServer initialized with {num_workers} workers.",
            color="green")

    @property
    def address(self) -> str:
        assert self._client_socket is not None, "Client socket is not bound"
        return self._client_socket.address[0]

    def __enter__(self) -> 'RPCServer':
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.shutdown()

    def bind(self, address: str = "tcp://*:5555") -> None:
        """
        Bind the server to the specified address.

        Args:
            address (str): The ZMQ address to bind the client-facing socket.
        """
        self._address = address

        # Check if PAIR mode is enabled via environment variable
        use_pair_mode = os.environ.get('TLLM_LLMAPI_ZMQ_PAIR', '0') != '0'
        socket_type = zmq.PAIR if use_pair_mode else zmq.ROUTER

        if use_pair_mode:
            logger_debug(
                "[server] Using zmq.PAIR socket type for RPC communication")

        self._client_socket = ZeroMqQueue(address=(address, self._hmac_key),
                                          is_server=True,
                                          is_async=True,
                                          use_hmac_encryption=self._hmac_key
                                          is not None,
                                          socket_type=socket_type,
                                          name="rpc_server")
        logger.info(f"RPCServer is bound to {self._address}")

    def shutdown(self, is_remote_call: bool = False) -> None:
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
            "[server] RPCServer is shutting down. Terminating server immediately..."
        )

        # Set the stop event to True, this will trigger immediate shutdown
        self._stop_event.set()

        # Log pending requests that will be cancelled
        logger_debug(
            f"[server] RPCServer is shutting down: {self._num_pending_requests} pending requests will be cancelled"
        )

        # Signal asyncio shutdown event if available
        if self._shutdown_event and self._loop:
            self._loop.call_soon_threadsafe(self._shutdown_event.set)

        if not is_remote_call:
            # Block the thread until shutdown is finished

            # 1. Cancel the main task gracefully which will trigger proper cleanup
            if self._main_task and not self._main_task.done():
                self._loop.call_soon_threadsafe(self._main_task.cancel)

            # 2. Wait for the server thread to exit (this will wait for proper cleanup)
            if self._server_thread and self._server_thread.is_alive():
                logger_debug(
                    "[server] RPCServer is waiting for server thread to exit")
                self._server_thread.join()
                self._server_thread = None
            logger_debug("[server] RPCServer thread joined")

            # 3. Shutdown the executor immediately without waiting for tasks
            if self._executor:
                self._executor.shutdown(wait=False)
                self._executor = None

            # 4. Close the client socket
            if self._client_socket:
                self._client_socket.close()
        else:
            # if the shutdown is called by a remote call, this method itself will
            # be executed in a executor thread, so we cannot join the server thread
            logger_debug(
                f"[server] RPC Server shutdown initiated: {self._num_pending_requests} pending requests will be cancelled"
            )

        logger_debug("[server] RPCServer is shutdown successfully",
                     color="yellow")

    def register_function(self,
                          func: Callable[..., Any],
                          name: Optional[str] = None) -> None:
        """Exposes a single function to clients.

        Args:
            func: The function to register.
            name: The name of the function. If not provided, the name of the function will be used.
        """
        fname = name or func.__name__
        if fname in self._functions:
            logger.warning(
                f"Function '{fname}' is already registered. Overwriting.")
        self._functions[fname] = func
        logger_debug(f"[server] Registered function: {fname}")

    def register_instance(self, instance: Any) -> None:
        """Exposes all public methods of a class instance.

        Args:
            instance: The instance to register.
        """
        logger_debug(
            f"[server] Registering instance of class: {instance.__class__.__name__}"
        )
        for name in dir(instance):
            if not name.startswith('_'):
                attr = getattr(instance, name)
                if callable(attr):
                    self.register_function(attr, name)

    def get_attr(self, name: str) -> Any:
        """ Get the attribute of the RPC server.

        Args:
            name: The name of the attribute to get.
        """
        return getattr(self, name)

    async def _drain_pending_requests(self) -> None:
        """Drain any remaining requests from the socket and send cancellation responses."""
        if self._client_socket is None:
            return

        logger_debug("[server] Draining pending requests after shutdown")
        drained_count = 0

        # Give a short window to drain any in-flight requests
        end_time = asyncio.get_event_loop().time() + 2

        while asyncio.get_event_loop().time() < end_time:
            try:
                req, routing_id = await asyncio.wait_for(
                    self._client_socket.get_async_noblock(return_identity=True),
                    timeout=2)
                req.routing_id = routing_id
                drained_count += 1
                logger_debug(f"[server] Draining request after shutdown: {req}")

                # Send cancellation response
                await self._send_error_response(
                    req,
                    RPCCancelled("Server is shutting down, request cancelled"))

            except asyncio.TimeoutError:
                # No more requests to drain
                break
            except Exception as e:
                logger.debug(f"Error draining request: {e}")
                break

        if drained_count > 0:
            logger_debug(
                f"[server] Drained {drained_count} requests after shutdown")

    async def _run_server(self) -> None:
        """Main server loop that handles incoming requests directly."""
        assert self._client_socket is not None, "Client socket is not bound"

        logger_debug("[server] RPC Server main loop started")

        # Create worker tasks
        for i in range(self._num_workers):
            task = asyncio.create_task(self._process_requests())
            self._worker_tasks.append(task)

        try:
            # Wait for all worker tasks to complete
            await asyncio.gather(*self._worker_tasks)
        except asyncio.CancelledError:
            logger_debug("[server] RPC Server main loop cancelled")
            # Cancel all worker tasks
            for task in self._worker_tasks:
                if not task.done():
                    task.cancel()
            # Wait for all tasks to finish cancellation
            await asyncio.gather(*self._worker_tasks, return_exceptions=True)
        except Exception as e:
            logger.error(f"RPC Server main loop error: {e}")
            logger.error(traceback.format_exc())
        finally:
            logger_debug("[server] RPC Server main loop exiting")

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
                    result=None,
                    error=error,
                    is_streaming=
                    True,  # Important: mark as streaming so it gets routed correctly
                    stream_status='error'),
                routing_id=req.routing_id)
            logger_debug(
                f"[server] Sent error response for request {req.request_id}",
                color="green")
        else:
            await self._client_socket.put_async(RPCResponse(req.request_id,
                                                            result=None,
                                                            error=error),
                                                routing_id=req.routing_id)
            logger_debug(
                f"[server] Sent error response for request {req.request_id}",
                color="green")

    async def _handle_shutdown_request(self, req: RPCRequest) -> bool:
        """Handle a request during shutdown. Returns True if handled."""
        if not self._shutdown_event.is_set():
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

    async def _process_requests(self) -> None:
        """Process incoming requests directly from the socket."""
        assert self._client_socket is not None, "Client socket is not bound"

        while not self._shutdown_event.is_set():
            try:
                #logger_debug(f"[server] Worker waiting for request", color="green")
                # Read request directly from socket with timeout
                req, routing_id = await asyncio.wait_for(
                    self._client_socket.get_async_noblock(return_identity=True),
                    timeout=2)
                req.routing_id = routing_id
                logger_debug(f"[server] Worker got request: {req}",
                             color="green")
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                logger_debug("[server] RPC worker cancelled")
                break
            except Exception as e:
                if self._shutdown_event.is_set():
                    break
                logger.error(f"RPC worker caught an exception: {e}")
                logger.error(traceback.format_exc())
                continue

            # shutdown methods depend on _num_pending_requests, so
            # they should not be counted
            if req.method_name not in ["_rpc_shutdown", "shutdown"]:
                self._num_pending_requests += 1
                logger_debug(
                    f"[server] Worker received request {req}, pending: {self._num_pending_requests}"
                )

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
                        f"[server] RPC Server sending response for request {req}, pending: {self._num_pending_requests}"
                    )
                    if await self._send_response(req, response):
                        logger_debug(
                            f"[server] RPC Server sent response for request {req}"
                        )

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
                    f"[server] RPC Server adjusted timeout for {method_type}{req.method_name}: "
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
                    f"[server] RPC Server running async task {req.method_name} in dispatcher"
                )
                result = await asyncio.wait_for(func(*req.args, **req.kwargs),
                                                timeout=adjusted_timeout)
            else:
                # Execute sync function in thread executor
                loop = asyncio.get_running_loop()

                def call_with_kwargs():
                    return func(*req.args, **req.kwargs)

                logger_debug(
                    f"[server] RPC Server running async task {req.method_name} in worker"
                )
                # TODO: let num worker control the pool size
                result = await asyncio.wait_for(loop.run_in_executor(
                    self._executor, call_with_kwargs),
                                                timeout=adjusted_timeout)

            response = RPCResponse(req.request_id, result=result)

        except asyncio.TimeoutError:
            response = RPCResponse(
                req.request_id,
                result=None,
                error=RPCTimeout(
                    f"Method '{req.method_name}' timed out after {req.timeout} seconds",
                    traceback=traceback.format_exc()))

        except Exception as e:
            response = RPCResponse(req.request_id,
                                   result=None,
                                   error=RPCError(
                                       str(e),
                                       cause=e,
                                       traceback=traceback.format_exc()))

        return response

    async def _process_streaming_request(self, req: RPCRequest) -> None:
        """Process a streaming request by sending multiple responses."""
        func = self._functions[req.method_name]

        if not inspect.isasyncgenfunction(func):
            await self._client_socket.put_async(RPCResponse(
                req.request_id,
                result=None,
                error=RPCStreamingError(
                    f"Method '{req.method_name}' is not an async generator.",
                    traceback=traceback.format_exc()),
                is_streaming=True,
                stream_status='error'),
                                                routing_id=req.routing_id)
            return

        chunk_index = 0

        adjusted_timeout: float = self._calculate_adjusted_timeout(
            req, is_streaming=True)

        try:
            logger_debug(
                f"[server] RPC Server running streaming task {req.method_name}")
            # Send start signal
            await self._client_socket.put_async(RPCResponse(
                req.request_id,
                result=None,
                error=None,
                is_streaming=True,
                chunk_index=chunk_index,
                stream_status='start'),
                                                routing_id=req.routing_id)
            logger_debug(
                f"[server] Sent start signal for request {req.request_id}",
                color="green")
            chunk_index += 1

            # Apply timeout to the entire streaming operation if specified
            if adjusted_timeout is not None and adjusted_timeout > 0:
                # Create a task for the async generator with timeout
                async def stream_with_timeout():
                    nonlocal chunk_index
                    async for result in func(*req.args, **req.kwargs):
                        if result is None or result == []:
                            # Skip None values or empty list to save bandwidth
                            # TODO[Superjomn]: add a flag to control this behavior
                            continue
                        # Check if shutdown was triggered
                        if self._shutdown_event.is_set():
                            raise RPCCancelled(
                                "Server is shutting down, streaming cancelled")

                        logger_debug(
                            f"[server] RPC Server got data and ready to send result {result}"
                        )
                        response = RPCResponse(req.request_id,
                                               result=result,
                                               error=None,
                                               is_streaming=True,
                                               chunk_index=chunk_index,
                                               stream_status='data')
                        if not await self._send_response(req, response):
                            # Stop streaming after a pickle error
                            return
                        logger_debug(
                            f"[server] Sent response for request {req.request_id}",
                            color="green")
                        chunk_index += 1

                # Use wait_for for timeout handling
                await asyncio.wait_for(stream_with_timeout(),
                                       timeout=adjusted_timeout)
            else:
                # No timeout specified, stream normally
                async for result in func(*req.args, **req.kwargs):
                    if result is None or result == []:
                        continue  # Skip None values or empty list
                    # Check if shutdown was triggered
                    if self._shutdown_event.is_set():
                        raise RPCCancelled(
                            "Server is shutting down, streaming cancelled")

                    logger_debug(
                        f"[server] RPC Server got data and ready to send result {result}"
                    )
                    response = RPCResponse(req.request_id,
                                           result=result,
                                           error=None,
                                           is_streaming=True,
                                           chunk_index=chunk_index,
                                           stream_status='data')
                    if not await self._send_response(req, response):
                        # Stop streaming after a pickle error
                        return
                    chunk_index += 1

            # Send end signal
            await self._client_socket.put_async(RPCResponse(
                req.request_id,
                result=None,
                error=None,
                is_streaming=True,
                chunk_index=chunk_index,
                stream_status='end'),
                                                routing_id=req.routing_id)
            logger_debug(
                f"[server] Sent end signal for request {req.request_id}",
                color="green")
        except RPCCancelled as e:
            # Server is shutting down, send cancelled error
            await self._client_socket.put_async(RPCResponse(
                req.request_id,
                result=None,
                error=e,
                is_streaming=True,
                chunk_index=chunk_index,
                stream_status='error'),
                                                routing_id=req.routing_id)
            logger_debug(
                f"[server] Sent error signal for request {req.request_id}",
                color="green")
        except asyncio.TimeoutError:
            await self._client_socket.put_async(RPCResponse(
                req.request_id,
                result=None,
                error=RPCTimeout(
                    f"Streaming method '{req.method_name}' timed out",
                    traceback=traceback.format_exc()),
                is_streaming=True,
                chunk_index=chunk_index,
                stream_status='error'),
                                                routing_id=req.routing_id)

        except Exception as e:
            response = RPCResponse(
                req.request_id,
                result=None,
                error=RPCStreamingError(str(e),
                                        traceback=traceback.format_exc()),
                is_streaming=True,
                chunk_index=chunk_index,
                stream_status='error')
            await self._send_response(req, response)

    async def _send_response(self, req: RPCRequest,
                             response: RPCResponse) -> bool:
        """Safely sends a response, handling pickle errors."""
        try:
            await self._client_socket.put_async(response,
                                                routing_id=req.routing_id)
            logger_debug(f"[server] Sent response for request {req.request_id}",
                         color="green")
            return True
        except Exception as e:
            logger.error(
                f"Failed to pickle response for request {req.request_id}: {e}")
            error_msg = f"Failed to pickle response: {e}"
            if req.is_streaming:
                error_cls = RPCStreamingError
                chunk_index = response.chunk_index if response else None
                error_response = RPCResponse(
                    req.request_id,
                    result=None,
                    error=error_cls(error_msg,
                                    traceback=traceback.format_exc()),
                    is_streaming=True,
                    chunk_index=chunk_index,
                    stream_status='error')
            else:
                error_cls = RPCError
                error_response = RPCResponse(
                    req.request_id,
                    result=None,
                    error=error_cls(error_msg,
                                    traceback=traceback.format_exc()))

            try:
                await self._client_socket.put_async(error_response,
                                                    routing_id=req.routing_id)
                logger_debug(
                    f"[server] Sent error response for request {req.request_id}",
                    color="green")
            except Exception as e_inner:
                logger.error(
                    f"Failed to send error response for request {req.request_id}: {e_inner}"
                )
            return False

    def start(self) -> None:
        """Binds sockets, starts workers, and begins proxying messages."""
        if self._client_socket is None:
            raise RuntimeError(
                "Server must be bound to an address before starting. Call bind() first."
            )

        self._client_socket.setup_lazily()
        logger.info(f"RPC Server started and listening on {self._address}")

        # Create and configure the event loop
        self._loop = asyncio.new_event_loop()

        self._shutdown_event = asyncio.Event()

        async def run_server():
            """Run the server until shutdown."""
            try:
                await self._run_server()
            except asyncio.CancelledError:
                logger_debug("[server] Server task cancelled")
            except Exception as e:
                logger.error(f"Server error: {e}")
                logger.error(traceback.format_exc())
            finally:
                # Cancel all worker tasks
                for task in self._worker_tasks:
                    if not task.done():
                        task.cancel()
                # Wait for all tasks to complete
                if self._worker_tasks:
                    await asyncio.gather(*self._worker_tasks,
                                         return_exceptions=True)

                # Drain any remaining requests and send cancellation responses
                await self._drain_pending_requests()

                logger_debug("[server] All server tasks completed")

        self._main_task = self._loop.create_task(run_server())

        def run_loop():
            asyncio.set_event_loop(self._loop)
            try:
                self._loop.run_until_complete(self._main_task)
            except RuntimeError as e:
                # This can happen if the event loop is stopped while futures are pending
                error_str = str(e)
                if "Event loop stopped before Future completed" in error_str:
                    # This is expected during shutdown - ignore it
                    logger.debug(
                        f"[server] Expected shutdown error: {error_str}")
                else:
                    # This is an unexpected RuntimeError - log full details
                    import traceback
                    logger.error(f"Event loop error: {error_str}")
                    logger.error(f"Traceback: {traceback.format_exc()}")
            except Exception as e:
                logger.error(f"Event loop error: {e}")
            finally:
                # Clean up any remaining tasks
                pending = asyncio.all_tasks(self._loop)
                for task in pending:
                    task.cancel()
                if pending:
                    try:
                        self._loop.run_until_complete(
                            asyncio.gather(*pending, return_exceptions=True))
                    except RuntimeError:
                        # Event loop might already be closed
                        pass
                self._loop.close()

        self._server_thread = threading.Thread(target=run_loop,
                                               name="rpc_server_thread",
                                               daemon=True)
        self._server_thread.start()

        logger.info("RPC Server has started.")


Server = RPCServer
