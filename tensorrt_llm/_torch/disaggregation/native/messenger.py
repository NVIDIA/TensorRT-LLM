# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import time
from abc import ABC, abstractmethod
from threading import Event, Lock, Thread, current_thread
from typing import Callable, Optional

import zmq

from tensorrt_llm import logger
from tensorrt_llm._torch.disaggregation.native.utils import get_local_ip

_CONTROL_STOP = b"STOP"
_CONTROL_WAKE = b"WAKE"


class MessengerInterface(ABC):
    """
    Abstract base class for messenger implementations.
    """

    @abstractmethod
    def start(self) -> None:
        """
        Start the messenger service.
        """
        ...

    @abstractmethod
    def send(self, messages: list[bytes], recipient: Optional[bytes] = None) -> None:
        """
        Send messages to a recipient.
        :param messages: List of byte messages to send.
        :param recipient: Optional recipient identifier.
        """
        ...

    @abstractmethod
    def receive(self) -> list[bytes]:
        """
        Receive messages.
        :return: List of byte messages received.
        """
        ...

    @abstractmethod
    def start_listener(self, on_message: Callable[[list[bytes]], Optional[bool]]) -> None:
        """
        Start a listener thread to handle incoming messages.
        :param on_message: Callback function to process received messages.
        """
        ...

    @abstractmethod
    def stop(self) -> None:
        """
        Stop the messenger service.
        """
        ...

    @property
    @abstractmethod
    def endpoint(self) -> str:
        """
        Get the endpoint of the messenger.
        :return: Endpoint string.
        """
        ...


def decode_message(
    message: list[bytes], encoding: str = "ascii", err_mode: str = "strict"
) -> tuple:
    if not isinstance(message, list) or not all(isinstance(m, bytes) for m in message):
        raise ValueError("Input must be a list of bytes")
    return tuple(m.decode(encoding, errors=err_mode) for m in message)


class ZMQMessenger(MessengerInterface):
    SOCKET_MODES = {
        "ROUTER": zmq.ROUTER,  # Handles multiple connections and routes messages by address.
        "DEALER": zmq.DEALER,  # Load balances outgoing messages and receives replies fairly.
        "REQ": zmq.REQ,  # Sends requests and waits for replies (synchronous).
        "REP": zmq.REP,  # Receives requests and sends replies (synchronous).
    }

    def __init__(self, mode: str, endpoint: Optional[str] = None) -> None:
        if mode not in self.SOCKET_MODES:
            raise ValueError(
                f"Invalid mode '{mode}'. Allowed modes are {list(self.SOCKET_MODES.keys())}"
            )
        self._context = zmq.Context()
        self._mode = mode
        self._socket = self._context.socket(self.SOCKET_MODES[mode])
        self._endpoint: Optional[str] = None
        # Serialize state transitions separately from socket I/O.  stop() must
        # release _lock before joining the listener, because a listener
        # callback may be entering send().  _socket_io_lock then fences every
        # main-socket send/receive from close().
        self._lock = Lock()
        self._stop_lock = Lock()
        self._socket_io_lock = Lock()
        self._control_send_lock = Lock()
        self._io_waiters_lock = Lock()
        self._io_waiters = 0
        self._io_waiters_drained = Event()
        self._io_waiters_drained.set()
        self._closed = False
        self._closing = False
        self._stop_event = Event()
        self._listener_thread: Optional[Thread] = None
        self._initialize_control_sockets()

        if endpoint is None:
            if mode in ["DEALER", "REQ"]:
                raise ValueError("endpoint is required for DEALER/REQ modes")
            endpoint = f"tcp://{get_local_ip()}:*"

        if mode in ["ROUTER", "REP"]:
            self._socket.bind(endpoint)
            self._endpoint = self._socket.getsockopt_string(zmq.LAST_ENDPOINT)
        elif mode in ["DEALER", "REQ"]:
            self._socket.connect(endpoint)
            self._endpoint = endpoint

        logger.info(f"Initialized ZMQMessenger(mode={mode}, endpoint={self._endpoint})")

    def _initialize_control_sockets(self) -> None:
        self._control_socket = self._context.socket(zmq.PAIR)
        self._internal_socket = self._context.socket(zmq.PAIR)
        inproc_endpoint = "inproc://stop_listener"
        self._control_socket.bind(inproc_endpoint)
        self._internal_socket.connect(inproc_endpoint)

    def start(self) -> None:
        pass

    def _signal_listener(self, command: bytes) -> None:
        """Wake a blocking listener poll without ever blocking the caller."""
        try:
            with self._control_send_lock:
                listener = self._listener_thread
                with self._lock:
                    stopping = self._closing or self._closed
                if (
                    listener is None
                    or listener is current_thread()
                    or not listener.is_alive()
                    or (command == _CONTROL_WAKE and stopping)
                ):
                    return
                self._internal_socket.send(command, flags=zmq.DONTWAIT)
        except zmq.Again:
            # A queued control frame already wakes the poller. The listener
            # also observes _stop_event directly, so another frame is not
            # required for correctness.
            pass

    def _acquire_socket_io(self) -> None:
        """Wake the listener, yield its blocking poll, and acquire main-socket ownership."""
        with self._io_waiters_lock:
            self._io_waiters += 1
            self._io_waiters_drained.clear()
        try:
            self._signal_listener(_CONTROL_WAKE)
            self._socket_io_lock.acquire()
        finally:
            with self._io_waiters_lock:
                self._io_waiters -= 1
                if self._io_waiters == 0:
                    self._io_waiters_drained.set()

    def send(self, messages: list[bytes], recipient: Optional[bytes] = None) -> None:
        self._acquire_socket_io()
        try:
            with self._lock:
                if self._closing or self._closed:
                    raise RuntimeError("ZMQMessenger is stopping or closed")
            if recipient:
                self._socket.send_multipart([recipient] + messages)
            else:
                self._socket.send_multipart(messages)
        finally:
            self._socket_io_lock.release()

    def receive(self) -> list[bytes]:
        self._acquire_socket_io()
        try:
            with self._lock:
                if self._closing or self._closed:
                    raise RuntimeError("ZMQMessenger is stopping or closed")
            return self._socket.recv_multipart()
        finally:
            self._socket_io_lock.release()

    def start_listener(
        self,
        on_message: Callable[[list[bytes]], Optional[bool]],
        on_error: Optional[Callable[[Exception], None]] = None,
    ) -> None:
        assert self._mode in ["ROUTER", "REP"], (
            "Listener can only be started in ROUTER or REP modes"
        )

        def handle_listener_exceptions(
            exception: Exception, on_error: Optional[Callable[[Exception], None]]
        ) -> None:
            logger.error(f"Error in listener: {exception}")
            if on_error:
                on_error(exception)
            else:
                self._stop_event.set()

        def listener() -> None:
            poller = zmq.Poller()
            with self._socket_io_lock:
                poller.register(self._socket, zmq.POLLIN)
                poller.register(self._control_socket, zmq.POLLIN)

            while not self._stop_event.is_set():
                try:
                    messages = None
                    control_message = None
                    with self._socket_io_lock:
                        events = dict(poller.poll(timeout=100))
                        if self._control_socket in events:
                            control_message = self._control_socket.recv()
                        elif self._socket in events:
                            with self._lock:
                                stopping = self._closing or self._closed
                            if not stopping:
                                messages = self._socket.recv_multipart()
                    if control_message == _CONTROL_STOP:
                        self._stop_event.set()
                    elif control_message == _CONTROL_WAKE:
                        # Do not immediately reacquire the socket ahead of the
                        # send/receive thread that woke this poll.
                        while not self._stop_event.is_set() and not self._io_waiters_drained.wait(
                            timeout=0.1
                        ):
                            pass
                    elif control_message is not None:
                        logger.warning(
                            f"Ignoring unknown messenger control frame {control_message!r}"
                        )
                    elif messages is not None:
                        persist = on_message(messages)
                        if persist is False:
                            self._stop_event.set()
                except zmq.ZMQError as e:
                    handle_listener_exceptions(e, on_error)
                    break
                except Exception as e:
                    handle_listener_exceptions(e, on_error)
                    break

            self._stop_event.set()

        # Serialize listener creation with stop() so a thread cannot start
        # against sockets that teardown has already decided to close.
        with self._stop_lock:
            with self._lock:
                if self._closing or self._closed:
                    raise RuntimeError("ZMQMessenger is stopping or closed")
                if self._listener_thread and self._listener_thread.is_alive():
                    raise RuntimeError("Listener already running")
                logger.info(f"Starting Messenger listener thread for {self._endpoint}")
                self._listener_thread = Thread(target=listener, daemon=True)
                self._listener_thread.start()

    def stop(self, timeout: float = 5) -> None:
        def _close_socket(name: str, socket: zmq.Socket) -> Optional[str]:
            if socket.closed:
                return None

            try:
                socket.setsockopt(zmq.LINGER, 0)
            except Exception as e:
                # Do not close with an unknown linger policy.  Leaving the
                # socket open makes the failed teardown explicitly retryable.
                return f"{name} socket linger configuration failed: {e}"

            try:
                socket.close()
            except Exception as e:
                return f"{name} socket close failed: {e}"
            if not socket.closed:
                return f"{name} socket remained open after close"
            return None

        # A separate stop lock keeps retryable stop attempts single-threaded
        # without holding the state lock across listener.join().
        with self._stop_lock:
            timeout = max(float(timeout), 0.0)
            deadline = time.monotonic() + timeout
            with self._lock:
                if self._closed:
                    return
                self._closing = True
            logger.debug("Stopping ZMQMessenger...")

            self._stop_event.set()
            listener = self._listener_thread
            if listener and listener.is_alive() and listener is not current_thread():
                self._signal_listener(_CONTROL_STOP)
                listener.join(max(0.0, deadline - time.monotonic()))
                if listener.is_alive():
                    raise RuntimeError(
                        "ZMQMessenger listener thread did not terminate within timeout"
                    )

            # A send/receive admitted before _closing was set may still be in
            # progress.  Taking the same I/O lock waits for it to finish and
            # prevents any later operation from racing socket close.
            close_errors = []
            if not self._socket_io_lock.acquire(timeout=max(0.0, deadline - time.monotonic())):
                raise RuntimeError("ZMQMessenger main socket I/O did not quiesce within timeout")
            try:
                error = _close_socket("main", self._socket)
                if error is not None:
                    close_errors.append(error)
            finally:
                self._socket_io_lock.release()
            # Fence a waiter that observed the listener before join but had
            # not yet sent its wake frame. It rechecks liveness under this
            # lock, so no control-socket operation can race close.
            with self._control_send_lock:
                for name, socket in (
                    ("internal", self._internal_socket),
                    ("control", self._control_socket),
                ):
                    error = _close_socket(name, socket)
                    if error is not None:
                        close_errors.append(error)

            if close_errors:
                raise RuntimeError("Failed to stop ZMQMessenger: " + "; ".join(close_errors))

            try:
                if not self._context.closed:
                    self._context.term()
            except Exception as e:
                raise RuntimeError(
                    f"Failed to stop ZMQMessenger: context termination failed: {e}"
                ) from e
            with self._lock:
                self._closed = True

    @property
    def endpoint(self) -> str:
        assert self._endpoint is not None
        return self._endpoint

    def __enter__(self) -> "ZMQMessenger":
        return self

    def __exit__(
        self, exc_type: Optional[type], exc_val: Optional[BaseException], exc_tb: Optional
    ) -> None:
        self.stop()
