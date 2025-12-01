import asyncio
import hashlib
import hmac
import os
import pickle  # nosec B403
import time
import traceback
from queue import Queue
from typing import Any, Optional

import zmq
import zmq.asyncio

from tensorrt_llm.logger import logger

from .._utils import nvtx_mark, nvtx_range_debug
from ..llmapi.utils import (ManagedThread, enable_llm_debug, logger_debug,
                            print_colored)


class ZeroMqQueue:
    ''' A Queue-like container for IPC using ZeroMQ. '''

    socket_type_str = {
        zmq.PAIR: "PAIR",
        zmq.PULL: "PULL",
        zmq.PUSH: "PUSH",
        zmq.ROUTER: "ROUTER",
        zmq.DEALER: "DEALER",
    }

    def __init__(self,
                 address: Optional[tuple[str, Optional[bytes]]] = None,
                 *,
                 socket_type: int = zmq.PAIR,
                 is_server: bool,
                 is_async: bool = False,
                 name: Optional[str] = None,
                 use_hmac_encryption: bool = True):
        '''
        Parameters:
            address (tuple[str, Optional[bytes]], optional): The address (tcp-ip_port, hmac_auth_key) for the IPC. Defaults to None. If hmac_auth_key is None and use_hmac_encryption is False, the queue will not use HMAC encryption.
            socket_type (int): The type of socket to use. Defaults to zmq.PAIR.
            is_server (bool): Whether the current process is the server or the client.
            is_async (bool): Whether to use asyncio for the socket. Defaults to False.
            name (str, optional): The name of the queue. Defaults to None.
            use_hmac_encryption (bool): Whether to use HMAC encryption for pickled data. Defaults to True.
        '''

        self.socket_type = socket_type
        self.address_endpoint = address[
            0] if address is not None else "tcp://127.0.0.1:*"
        self.is_server = is_server
        self.context = zmq.Context() if not is_async else zmq.asyncio.Context()
        self.poller = None
        self.socket = None

        self._setup_done = False
        self.name = name
        self.socket = self.context.socket(socket_type)

        # For ROUTER sockets, track the last identity to enable replies. For now we assume there is only one client in our case.
        self._last_identity = None

        self.hmac_key = address[1] if address is not None else None
        self.use_hmac_encryption = use_hmac_encryption

        # Check HMAC key condition
        if self.use_hmac_encryption and not self.is_server and self.hmac_key is None:
            raise ValueError(
                "Client must receive HMAC key when encryption is enabled")
        elif not self.use_hmac_encryption and self.hmac_key is not None:
            raise ValueError(
                "Server and client should not receive HMAC key when encryption is disabled"
            )

        if (socket_type == zmq.PAIR and self.is_server
            ) or socket_type == zmq.PULL or socket_type == zmq.ROUTER:
            self.socket.bind(
                self.address_endpoint
            )  # Binds to the address and occupy a port immediately
            self.address_endpoint = self.socket.getsockopt(
                zmq.LAST_ENDPOINT).decode()
            logger_debug(
                f"Server [{name}] bound to {self.address_endpoint} in {self.socket_type_str[socket_type]}\n",
                "green")

            if self.use_hmac_encryption and not self.hmac_key:
                # Initialize HMAC key for pickle encryption
                logger.info(f"Generating a new HMAC key for server {self.name}")
                self.hmac_key = os.urandom(32)

            self.address = (self.address_endpoint, self.hmac_key)

    def setup_lazily(self):
        if self._setup_done:
            return
        self._setup_done = True

        if not self.is_server:
            logger_debug(
                f"Client [{self.name}] connecting to {self.address_endpoint} in {self.socket_type_str[self.socket_type]}\n",
                "green")
            self.socket.connect(self.address_endpoint)

        self.poller = zmq.Poller()
        self.poller.register(self.socket, zmq.POLLIN)

    def poll(self, timeout: int) -> bool:
        """
        Parameters:
            timeout (int): Timeout in seconds
        """
        self.setup_lazily()

        events = dict(self.poller.poll(timeout=timeout * 1000))
        if self.socket in events and events[self.socket] == zmq.POLLIN:
            return True
        else:
            return False

    def put(self, obj: Any):
        self.setup_lazily()
        with nvtx_range_debug("send", color="blue", category="IPC"):
            if self.use_hmac_encryption or self.socket_type == zmq.ROUTER:
                # Need manual serialization for encryption or ROUTER multipart
                data = self._prepare_data(obj)
                self._send_data(data)
            else:
                # Standard socket without encryption - use pyobj directly
                self.socket.send_pyobj(obj)

    def put_noblock(self,
                    obj: Any,
                    *,
                    retry: int = 1,
                    wait_time: float = 0.001):
        '''
        Put an object into the queue without blocking, and retry if the send fails.
        NOTE: It won't raise any error if the send fails.

        Parameters:
            obj (Any): The object to send.
            retry (int): The number of times to retry sending the object.
            wait_time (float): The time to wait before retrying.
        '''

        assert retry >= 0 and retry <= 10, "Retry must be between 0 and 10, adjust the wait_time if needed"

        self.setup_lazily()
        with nvtx_range_debug("send", color="blue", category="IPC"):

            data = self._prepare_data(obj)
            try:
                self._send_data(data, flags=zmq.NOBLOCK)
            except zmq.Again:
                if retry > 0:
                    time.sleep(wait_time)
                    self.put_noblock(obj, retry=retry - 1, wait_time=wait_time)
                else:
                    logger.error(f"Failed to send object: {obj}")

    async def put_async(self, obj: Any):
        self.setup_lazily()
        try:
            if self.use_hmac_encryption or self.socket_type == zmq.ROUTER:
                # Need manual serialization for encryption or ROUTER multipart
                data = self._prepare_data(obj)
                await self._send_data_async(data)
            else:
                # Standard socket without encryption
                await self.socket.send_pyobj(obj)
        except TypeError as e:
            logger.error(f"Cannot pickle {obj}")
            raise e
        except Exception as e:
            logger.error(f"Error sending object: {e}")
            logger.error(traceback.format_exc())
            raise e

        nvtx_mark("ipc.send", color="blue", category="IPC")

    async def put_async_noblock(self, obj: Any):
        self.setup_lazily()
        try:
            if self.use_hmac_encryption:
                data = pickle.dumps(obj)  # nosec B301
                signed_data = self._sign_data(data)
                await self.socket.send(signed_data, flags=zmq.NOBLOCK)
            else:
                await self.socket.send_pyobj(obj, flags=zmq.NOBLOCK)
        except Exception as e:
            logger.error(f"Error sending object: {e}")
            logger.error(traceback.format_exc())
            raise e

    def get(self) -> Any:
        self.setup_lazily()
        return self._recv_data()

    async def get_async(self) -> Any:
        self.setup_lazily()
        return await self._recv_data_async()

    async def get_async_noblock(self, timeout: float = 0.5) -> Any:
        return await asyncio.wait_for(self.get_async(), timeout)

    def close(self):
        if self.socket:
            self.socket.close()
            self.socket = None
        if self.context:
            self.context.term()
            self.context = None

    def _verify_hmac(self, data: bytes, actual_hmac: bytes) -> bool:
        """Verify the HMAC of received pickle data."""
        expected_hmac = hmac.new(self.hmac_key, data, hashlib.sha256).digest()
        return hmac.compare_digest(expected_hmac, actual_hmac)

    def _sign_data(self, data_before_encoding: bytes) -> bytes:
        """Generate HMAC for data."""
        hmac_signature = hmac.new(self.hmac_key, data_before_encoding,
                                  hashlib.sha256).digest()
        return data_before_encoding + hmac_signature

    def __del__(self):
        self.close()

    def _prepare_data(self, obj: Any) -> bytes:
        """Serialize object and optionally add HMAC signature."""
        data = pickle.dumps(obj)  # nosec B301
        if self.use_hmac_encryption:
            return self._sign_data(data)
        return data

    def _parse_data(self, data: bytes) -> Any:
        """Parse data and optionally verify HMAC signature."""
        if self.use_hmac_encryption:
            # Split data and HMAC
            message_data = data[:-32]
            actual_hmac = data[-32:]

            # Verify HMAC
            if not self._verify_hmac(message_data, actual_hmac):
                raise RuntimeError("HMAC verification failed")

            return pickle.loads(message_data)  # nosec B301
        else:
            return pickle.loads(data)  # nosec B301

    def _send_data(self, data: bytes, flags: int = 0):
        """Send data using appropriate API based on socket type."""
        if self.socket_type == zmq.ROUTER:
            if self._last_identity is None:
                raise ValueError("ROUTER socket requires identity")
            self.socket.send_multipart([self._last_identity, data], flags=flags)
        else:
            self.socket.send(data, flags=flags)

    async def _send_data_async(self, data: bytes):
        """Async version of _send_data."""
        if self.socket_type == zmq.ROUTER:
            if self._last_identity is None:
                raise ValueError("ROUTER socket requires identity")
            await self.socket.send_multipart([self._last_identity, data])
        else:
            await self.socket.send(data)

    def _recv_data(self) -> Any:
        """Receive data using appropriate API based on socket type."""
        if self.socket_type == zmq.ROUTER:
            identity, data = self.socket.recv_multipart()
            self._last_identity = identity  # Store for replies
            obj = self._parse_data(data)
            return obj
        else:
            if self.use_hmac_encryption:
                data = self.socket.recv()
                obj = self._parse_data(data)
            else:
                obj = self.socket.recv_pyobj()
            return obj

    async def _recv_data_async(self) -> Any:
        """Async version of _recv_data."""
        if self.socket_type == zmq.ROUTER:
            identity, data = await self.socket.recv_multipart()
            self._last_identity = identity  # Store for replies
            return self._parse_data(data)
        else:
            if self.use_hmac_encryption:
                data = await self.socket.recv()
                return self._parse_data(data)
            else:
                return await self.socket.recv_pyobj()

    def notify_with_retry(self, message, max_retries=5, timeout=1):
        """
        Notify with automatic retry on failure (for DEALER socket pattern).

        Args:
            message: Message to send
            max_retries: Maximum retry attempts (default: 5)
            timeout: Timeout in seconds for each attempt (default: 1)

        Returns:
            bool: True if acknowledgment received, False if failed after all retries
        """
        if self.socket_type != zmq.DEALER:
            raise ValueError(
                "notify_with_retry is only supported for DEALER socket for now")

        retry_count = 0

        while retry_count < max_retries:
            try:
                self.put(message)
                # Wait for ACK with timeout
                if self.poll(timeout):
                    self.get()
                    return True
                else:
                    retry_count += 1

            except Exception as e:
                logger.error(f"Failed to notify with retry: {e}")
                retry_count += 1

        return False


IpcQueue = ZeroMqQueue


class FusedIpcQueue:
    ''' A Queue-like container for IPC with optional message batched. '''

    def __init__(self,
                 address: Optional[tuple[str, Optional[bytes]]] = None,
                 *,
                 is_server: bool,
                 fuse_message=False,
                 fuse_size=100000,
                 error_queue=None,
                 queue_cls=ZeroMqQueue,
                 **kwargs):

        self.queue = queue_cls(address=address, is_server=is_server, **kwargs)
        self.fuse_message = fuse_message
        self.error_queue = error_queue
        self.fuse_size = fuse_size
        self._message_counter = 0
        self._obj_counter = 0
        self._send_thread = None
        self.sending_queue = Queue() if fuse_message else None

    def setup_sender(self):
        if not self.fuse_message or self._send_thread is not None:
            return

        def send_task():
            while True:
                qsize = self.sending_queue.qsize()
                if qsize > 0:
                    qsize = min(self.fuse_size, qsize)
                    self._obj_counter += qsize
                    message = [
                        self.sending_queue.get_nowait() for _ in range(qsize)
                    ]
                    self.queue.put(message)
                    self._message_counter += 1
                else:
                    time.sleep(0.001)

        self._send_thread = ManagedThread(send_task,
                                          name="fused_send_thread",
                                          error_queue=self.error_queue)
        self._send_thread.start()

    def put(self, obj: Any):
        self.setup_sender()
        if self.fuse_message:
            self.sending_queue.put_nowait(obj)
        else:
            batch = obj if isinstance(obj, list) else [obj]
            self.queue.put(batch)

    def get(self) -> Any:
        return self.queue.get()

    @property
    def address(self) -> tuple[str, Optional[bytes]]:
        return self.queue.address

    def __del__(self):
        self.close()

    def print_fuse_stats(self):
        if self._message_counter > 0:
            print_colored(
                f"IPCQueue: {self._message_counter} messages, {self._obj_counter} objects sent, average: {self._obj_counter/self._message_counter}.\n",
                "green")

    def close(self):
        self.queue.close()

        if self._send_thread is not None:
            self._send_thread.stop()
            self._send_thread.join()
            self._send_thread = None

        if enable_llm_debug():
            self.print_fuse_stats()
