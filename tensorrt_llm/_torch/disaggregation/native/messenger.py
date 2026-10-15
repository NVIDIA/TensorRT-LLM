# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from abc import ABC, abstractmethod
from concurrent.futures import Future
from queue import Queue
from threading import Event, Lock, Thread, get_ident
from typing import Callable, Optional

import zmq

from tensorrt_llm import logger
from tensorrt_llm._torch.disaggregation.native.utils import get_local_ip


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
    LISTENER_MODES = {"ROUTER", "REP"}

    def __init__(self, mode: str, endpoint: Optional[str] = None) -> None:
        if mode not in self.SOCKET_MODES:
            raise ValueError(
                f"Invalid mode '{mode}'. Allowed modes are {list(self.SOCKET_MODES.keys())}"
            )
        # A context is thread-safe and owns a native I/O thread. Reuse the
        # process singleton while keeping every socket confined to its owning
        # thread; high-fanout disaggregated transfers can otherwise create
        # hundreds of contexts and native threads per process.
        self._context = zmq.Context.instance()
        self._mode = mode
        self._socket: Optional[zmq.Socket] = None
        self._socket_owner_thread_id: Optional[int] = None
        self._endpoint: Optional[str] = None
        self._lock = Lock()
        self._closed = False
        self._stop_event = Event()
        self._listener_registered = Event()
        self._listener_thread: Optional[Thread] = None
        self._on_message: Optional[Callable[[list[bytes]], Optional[bool]]] = None
        self._on_error: Optional[Callable[[Exception], None]] = None
        self._listener_commands: Queue[tuple[list[bytes], Optional[bytes], Future[None]]] = Queue()

        if endpoint is None:
            if mode in ["DEALER", "REQ"]:
                raise ValueError("endpoint is required for DEALER/REQ modes")
            endpoint = f"tcp://{get_local_ip()}:*"

        if mode in self.LISTENER_MODES:
            # A ZeroMQ socket must not migrate between threads.  Start its owner
            # immediately so bind, poll/recv/send, and close all happen there.
            ready: Future[str] = Future()
            self._listener_thread = Thread(
                target=self._listener_loop,
                args=(endpoint, ready),
                daemon=True,
                name=f"zmq_{mode.lower()}_owner",
            )
            self._listener_thread.start()
            self._endpoint = ready.result()
        else:
            self._socket_owner_thread_id = get_ident()
            self._socket = self._context.socket(self.SOCKET_MODES[mode])
            self._socket.connect(endpoint)
            self._endpoint = endpoint

        logger.info(f"Initialized ZMQMessenger(mode={mode}, endpoint={self._endpoint})")

    def start(self) -> None:
        pass

    def send(self, messages: list[bytes], recipient: Optional[bytes] = None) -> None:
        if self._mode in self.LISTENER_MODES:
            thread = self._listener_thread
            if thread is not None and thread.ident == get_ident():
                self._send_on_socket(messages, recipient)
                return

            result: Future[None] = Future()
            with self._lock:
                if self._closed or thread is None or not thread.is_alive():
                    raise RuntimeError("ZMQMessenger listener is not running")
                if self._on_message is None:
                    raise RuntimeError("ZMQMessenger listener has not been started")
                self._listener_commands.put((messages, recipient, result))
            result.result()
            return

        self._send_on_socket(messages, recipient)

    def _send_on_socket(self, messages: list[bytes], recipient: Optional[bytes] = None) -> None:
        self._assert_socket_owner()
        assert self._socket is not None
        frames = [recipient] + messages if recipient else messages
        self._socket.send_multipart(frames)

    def receive(self) -> list[bytes]:
        self._assert_socket_owner()
        assert self._socket is not None
        return self._socket.recv_multipart()

    def _assert_socket_owner(self) -> None:
        if self._socket_owner_thread_id != get_ident():
            raise RuntimeError("ZeroMQ socket accessed outside its owner thread")

    def start_listener(
        self,
        on_message: Callable[[list[bytes]], Optional[bool]],
        on_error: Optional[Callable[[Exception], None]] = None,
    ) -> None:
        assert self._mode in ["ROUTER", "REP"], (
            "Listener can only be started in ROUTER or REP modes"
        )
        with self._lock:
            if self._closed:
                raise RuntimeError("ZMQMessenger is closed")
            if self._listener_thread is None or not self._listener_thread.is_alive():
                raise RuntimeError("ZMQMessenger listener owner thread is not running")
            if self._on_message is not None:
                raise RuntimeError("Listener already running")
            self._on_message = on_message
            self._on_error = on_error
            self._listener_registered.set()
        logger.info(f"Started Messenger listener for {self._endpoint}")

    def _listener_loop(self, endpoint: str, ready: Future[str]) -> None:
        try:
            self._socket_owner_thread_id = get_ident()
            self._socket = self._context.socket(self.SOCKET_MODES[self._mode])
            self._socket.bind(endpoint)
            ready.set_result(self._socket.getsockopt_string(zmq.LAST_ENDPOINT))

            poller = zmq.Poller()
            poller.register(self._socket, zmq.POLLIN)
            while not self._stop_event.is_set():
                if not self._listener_registered.wait(timeout=0.1):
                    continue

                self._drain_listener_commands()
                events = dict(poller.poll(timeout=100))
                if self._socket in events:
                    messages = self.receive()
                    assert self._on_message is not None
                    persist = self._on_message(messages)
                    if persist is False:
                        self._stop_event.set()
                self._drain_listener_commands()
        except Exception as error:
            if not ready.done():
                ready.set_exception(error)
            else:
                logger.error(f"Error in listener: {error}")
                if self._on_error:
                    self._on_error(error)
        finally:
            self._stop_event.set()
            self._fail_listener_commands(RuntimeError("ZMQMessenger listener stopped"))
            self._close_socket_on_owner()

    def _drain_listener_commands(self) -> None:
        while not self._listener_commands.empty():
            messages, recipient, result = self._listener_commands.get()
            try:
                self._send_on_socket(messages, recipient)
            except Exception as error:
                result.set_exception(error)
            else:
                result.set_result(None)

    def _fail_listener_commands(self, error: Exception) -> None:
        while not self._listener_commands.empty():
            _, _, result = self._listener_commands.get()
            result.set_exception(error)

    def _close_socket_on_owner(self) -> None:
        assert self._mode in self.LISTENER_MODES
        self._assert_socket_owner()
        self._close_socket(self._socket)

    @staticmethod
    def _close_socket(socket: Optional[zmq.Socket]) -> None:
        if socket is None:
            return
        try:
            if not socket.closed:
                socket.setsockopt(zmq.LINGER, 0)
                socket.close()
        except Exception as error:
            logger.error(f"Error closing socket: {error}")

    def stop(self, timeout: int = 5) -> None:
        with self._lock:
            was_closed = self._closed
            self._closed = True
            self._stop_event.set()
            self._listener_registered.set()
            thread = self._listener_thread

        if self._mode in self.LISTENER_MODES:
            if thread is not None and thread.ident != get_ident():
                thread.join(timeout)
                if thread.is_alive():
                    logger.warning("Listener thread did not terminate within timeout")
            return

        if not was_closed:
            self._assert_socket_owner()
            self._close_socket(self._socket)
        # The process-wide context outlives every individual messenger.

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


class ZMQDealerPool:
    """Thread-safe multi-endpoint sender with one DEALER-socket owner thread."""

    def __init__(self) -> None:
        self._commands: Queue[tuple[str, list[bytes], Future[None]] | None] = Queue()
        self._lock = Lock()
        self._closed = False
        self._thread = Thread(target=self._run, daemon=True, name="zmq_dealer_pool")
        self._thread.start()

    def send(self, endpoint: str, messages: list[bytes]) -> None:
        """Send a multipart message without exposing a DEALER across threads."""
        result: Future[None] = Future()
        with self._lock:
            if self._closed:
                raise RuntimeError("ZMQDealerPool is closed")
            self._commands.put((endpoint, messages, result))
        result.result()

    def stop(self) -> None:
        """Drain queued sends, close sockets on their owner thread, and join it."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._commands.put(None)
        self._thread.join()

    def _run(self) -> None:
        dealers: dict[str, ZMQMessenger] = {}
        while True:
            command = self._commands.get()
            if command is None:
                break
            endpoint, messages, result = command
            try:
                dealer = dealers.get(endpoint)
                if dealer is None:
                    dealer = ZMQMessenger(mode="DEALER", endpoint=endpoint)
                    dealers[endpoint] = dealer
                dealer.send(messages)
            except Exception as error:
                result.set_exception(error)
            else:
                result.set_result(None)

        for endpoint, dealer in dealers.items():
            try:
                dealer.stop()
            except Exception as error:
                logger.warning(f"Failed to stop DEALER for endpoint {endpoint}: {error}")
