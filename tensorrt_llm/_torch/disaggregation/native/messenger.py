from abc import ABC, abstractmethod
from threading import Event, Lock, Thread
from typing import Callable, Optional

import zmq

from tensorrt_llm import logger
from tensorrt_llm._torch.disaggregation.native.utils import get_local_ip


class MessengerInterface(ABC):
    """
    Abstract base class for messenger implementations.
    """

    @abstractmethod
    def start(self):
        """
        Start the messenger service.
        """
        ...

    @abstractmethod
    def send(self, messages: list[bytes], recipient: Optional[bytes] = None):
        """
        Send messages to a recipient.
        :param messages: List of byte messages to send.
        :param recipient: Optional recipient identifier.
        """
        ...

    @abstractmethod
    def send_encoded(self, *messages, encoding: str = "ascii"):
        """
        Send messages after encoding them.
        :param messages: Messages to send.
        :param encoding: Encoding format.
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
    def start_listener(self, on_message: Callable[[list[bytes]], None]):
        """
        Start a listener thread to handle incoming messages.
        :param on_message: Callback function to process received messages.
        """
        ...

    @abstractmethod
    def stop(self):
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

    def __init__(self, mode: str, endpoint: Optional[str] = None):
        self._context = zmq.Context()
        self._mode = mode
        self._socket = self._context.socket(self.SOCKET_MODES[mode])
        self._endpoint: Optional[str] = None
        self._lock = Lock()
        self._closed = False
        self._stop_event = Event()
        self._listener_thread: Optional[Thread] = None
        self._initialize_control_sockets()

        if endpoint is None:
            endpoint = f"tcp://{get_local_ip()}:*"

        if mode in ["ROUTER", "REP"]:
            self._socket.bind(endpoint)
            self._endpoint = self._socket.getsockopt_string(zmq.LAST_ENDPOINT)
        elif mode in ["DEALER", "REQ"]:
            self._socket.connect(endpoint)
            self._endpoint = endpoint

        logger.info(f"Initialized ZMQMessenger(mode={mode}, endpoint={self._endpoint})")

    def _initialize_control_sockets(self):
        self._control_socket = self._context.socket(zmq.PAIR)
        self._internal_socket = self._context.socket(zmq.PAIR)
        inproc_endpoint = "inproc://stop_listener"
        self._control_socket.bind(inproc_endpoint)
        self._internal_socket.connect(inproc_endpoint)

    def start(self):
        pass

    def send(self, messages: list[bytes], recipient: Optional[bytes] = None):
        if recipient:
            self._socket.send_multipart([recipient] + messages)
        else:
            self._socket.send_multipart(messages)

    def send_encoded(self, *messages, encoding: str = "ascii"):
        encoded_messages = [str(message).encode(encoding) for message in messages]
        self.send(encoded_messages)

    def receive(self) -> list[bytes]:
        return self._socket.recv_multipart()

    def start_listener(
        self,
        on_message: Callable[[list[bytes]], None],
        on_error: Optional[Callable[[Exception], None]] = None,
    ):
        assert self._mode in ["ROUTER", "REP"], (
            "Listener can only be started in ROUTER or REP modes"
        )
        if self._listener_thread and self._listener_thread.is_alive():
            raise RuntimeError("Listener already running")

        def listener():
            poller = zmq.Poller()
            poller.register(self._socket, zmq.POLLIN)
            poller.register(self._control_socket, zmq.POLLIN)

            while not self._stop_event.is_set():
                events = dict(poller.poll(timeout=100))
                try:
                    if self._control_socket in events:
                        self._stop_event.set()
                    elif self._socket in events:
                        messages = self.receive()
                        try:
                            persist = on_message(messages)
                            if persist is False:
                                self._stop_event.set()
                        except Exception as e:
                            logger.error(f"Error in on_message callback: {e}")
                            if on_error:
                                on_error(e)
                            else:
                                self._stop_event.set()
                except zmq.ZMQError as e:
                    logger.error(f"ZMQ Error in listener: {e}")
                    if on_error:
                        on_error(e)
                    break
                except Exception as e:
                    logger.error(f"Error in listener: {e}")
                    if on_error:
                        on_error(e)
                    break

            self._stop_event.set()

        self._listener_thread = Thread(target=listener, daemon=True)
        self._listener_thread.start()

    def stop(self, timeout=5):
        def _close_socket(socket: zmq.Socket):
            try:
                if not socket.closed:
                    socket.close()
            except Exception as e:
                logger.error(f"Error closing socket: {e}")

        with self._lock:
            if self._closed:
                return
            self._closed = True
            logger.debug("Stopping ZMQMessenger...")

            self._stop_event.set()
            self._internal_socket.send(b"STOP")
            if self._listener_thread:
                self._listener_thread.join(timeout)
                if self._listener_thread.is_alive():
                    logger.warning("Listener thread did not terminate within timeout")

            _close_socket(self._socket)
            _close_socket(self._internal_socket)
            _close_socket(self._control_socket)

            try:
                if self._context.closed:
                    self._context.term()
            except Exception as e:
                logger.error(f"Error terminating ZMQ context: {e}")

    @property
    def endpoint(self) -> str:
        assert self._endpoint is not None
        return self._endpoint

    def __del__(self):
        self.stop()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
