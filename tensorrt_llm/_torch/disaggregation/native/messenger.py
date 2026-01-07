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


def decode_message(message: list[bytes], encoding: str = "ascii", errors: str = "strict") -> tuple:
    if not isinstance(message, list) or not all(isinstance(m, bytes) for m in message):
        raise ValueError("Input must be a list of bytes")
    return tuple(m.decode(encoding, errors=errors) for m in message)


class ZMQMessenger(MessengerInterface):
    def __init__(self, mode: str, endpoint: Optional[str] = f"tcp://{get_local_ip()}:*"):
        self._context = zmq.Context()
        self._mode = mode
        self._socket = self._initialize_socket(mode)
        self._closed = False
        self._listener_thread: Optional[Thread] = None
        self._stop_event = Event()
        self._lock = Lock()

        self._internal_socket = self._context.socket(zmq.PAIR)
        self._control_socket = self._context.socket(zmq.PAIR)
        inproc_endpoint = "inproc://stop_listener"
        self._control_socket.bind(inproc_endpoint)
        self._internal_socket.connect(inproc_endpoint)

        if endpoint:
            if mode in ["ROUTER", "REP"]:
                self._socket.bind(endpoint)
                self._endpoint = self._socket.getsockopt_string(zmq.LAST_ENDPOINT)
            elif mode in ["DEALER", "REQ"]:
                self._socket.connect(endpoint)
                self._endpoint = endpoint

        logger.debug(f"Initializing ZMQMessenger, mode={mode}, endpoint={self._endpoint}")

    def _initialize_socket(self, mode):
        if mode == "ROUTER":
            return self._context.socket(zmq.ROUTER)
        elif mode == "DEALER":
            return self._context.socket(zmq.DEALER)
        elif mode == "REQ":
            return self._context.socket(zmq.REQ)
        elif mode == "REP":
            return self._context.socket(zmq.REP)
        else:
            raise ValueError(f"Unsupported ZeroMQ socket mode: {mode}")

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

    def start_listener(self, on_message: Callable[[list[bytes]], None]):
        if self._listener_thread and self._listener_thread.is_alive():
            raise RuntimeError("Listener already running")

        def listener():
            poller = zmq.Poller()
            poller.register(self._socket, zmq.POLLIN)
            poller.register(self._control_socket, zmq.POLLIN)

            while not self._stop_event.is_set():
                events = dict(poller.poll())
                try:
                    if self._control_socket in events:
                        self._stop_event.set()
                    elif self._socket in events:
                        messages = self.receive()
                        persist = on_message(messages)
                        if persist is False:
                            self._stop_event.set()
                except zmq.ZMQError as e:
                    logger.error(f"ZMQ Error in listener: {e}")
                    continue
                except Exception as e:
                    logger.error(f"Error in listener: {e}")
                    continue

        self._listener_thread = Thread(target=listener, daemon=True)
        self._listener_thread.start()

    def stop(self, timeout=5):
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._stop_event.set()
            self._internal_socket.send(b"STOP")
            if self._listener_thread:
                self._listener_thread.join(timeout)
            self._socket.close()
            self._internal_socket.close()
            self._control_socket.close()
            self._context.term()

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
