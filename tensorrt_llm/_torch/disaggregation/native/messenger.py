import hashlib
import hmac
import struct
import time
from abc import ABC, abstractmethod
from collections import OrderedDict
from threading import Event, Lock, Thread
from typing import Callable, Mapping, NamedTuple, Optional

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
    def receive(self, timeout_ms: Optional[int] = None) -> list[bytes]:
        """
        Receive messages.
        :param timeout_ms: Maximum wait in milliseconds, or None to wait indefinitely.
        :raises TimeoutError: If no message arrives within the timeout.
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


class FrameAuthenticationError(PermissionError):
    """A signed control message failed verification."""


class SignedHeader(NamedTuple):
    key_id: int
    flow: int
    seq: int
    stream_id: bytes
    sender_id: bytes


class FrameSigner:
    """HMAC-SHA256 authentication for multipart control messages.

    A signed message is the payload frames followed by a header frame and a
    32-byte MAC frame: ``[*payload, header, mac]``. The MAC covers, each with a
    4-byte length prefix so frame boundaries cannot be shifted: the caller's
    context string, the header (key id, flow, seq, stream id, sender id), the
    recipient id, and every payload frame. ROUTER identity frames are added and
    consumed by ZMQ, never seen by the peer, and are therefore not signed.

    The signer holds only key material and the context string, so one instance
    can be shared by every socket of a component. Identities and flow are
    supplied per message by ``ZMQMessenger``.

    Args:
        keys: Accepted keys by key id (0-255). Verification accepts any of
            them, so a rotation can roll out "accept new" before "send new".
        active_key_id: Key id used for signing; must be present in ``keys``.
        context: Domain-separation string chosen by the caller. Two protocols
            sharing a key must use different contexts, otherwise a frame valid
            in one verifies in the other.
    """

    MAC_SIZE = 32
    STREAM_ID_SIZE = 16
    MIN_KEY_SIZE = 16
    # Flow is the socket direction, not a role: requests travel DEALER to
    # ROUTER and replies ROUTER to DEALER. Together with sender and recipient
    # ids it stops a frame from being reflected back the way it came.
    FLOW_TO_ROUTER = 0
    FLOW_TO_DEALER = 1

    _VERSION = 1
    _HEADER = struct.Struct("!BBBQ16sH")
    _LEN = struct.Struct("!I")

    def __init__(self, keys: Mapping[int, bytes], active_key_id: int, context: bytes) -> None:
        if not keys:
            raise ValueError("FrameSigner requires at least one key")
        for key_id, key in keys.items():
            if not 0 <= key_id <= 255:
                raise ValueError(f"Key id {key_id} is outside 0-255")
            if not isinstance(key, bytes) or len(key) < self.MIN_KEY_SIZE:
                raise ValueError(
                    f"Key {key_id} must be bytes of at least {self.MIN_KEY_SIZE} bytes"
                )
        if active_key_id not in keys:
            raise ValueError(f"Active key id {active_key_id} is not among the accepted keys")
        if not context:
            raise ValueError("FrameSigner requires a non-empty context")
        self._active_key_id = active_key_id
        self._context = bytes(context)
        # One HMAC state per key with the context already absorbed: each
        # message continues from a copy instead of re-keying the hash.
        self._mac_bases = {key_id: self._new_mac_base(key) for key_id, key in keys.items()}

    def __repr__(self) -> str:
        return (
            f"FrameSigner(key_ids={sorted(self._mac_bases)}, "
            f"active_key_id={self._active_key_id}, context={self._context!r})"
        )

    def sign(
        self,
        frames: list[bytes],
        *,
        sender_id: bytes,
        recipient_id: bytes,
        flow: int,
        seq: int = 0,
        stream_id: bytes = bytes(STREAM_ID_SIZE),
    ) -> list[bytes]:
        """Return the ``[header, mac]`` frames to append to ``frames``."""
        if len(stream_id) != self.STREAM_ID_SIZE:
            raise ValueError(f"stream_id must be {self.STREAM_ID_SIZE} bytes")
        header = (
            self._HEADER.pack(
                self._VERSION, self._active_key_id, flow, seq, stream_id, len(sender_id)
            )
            + sender_id
        )
        mac = self._mac(self._active_key_id, header, recipient_id, frames)
        return [header, mac]

    def verify(
        self, frames: list[bytes], *, recipient_id: bytes, flow: int
    ) -> tuple[list[bytes], SignedHeader]:
        """Verify ``[*payload, header, mac]`` and return the payload and header.

        Raises:
            FrameAuthenticationError: The message is malformed, uses an unknown
                key id, was sent on the other flow or to another recipient, or
                its MAC does not match.
        """
        if len(frames) < 2:
            raise FrameAuthenticationError("Message carries no authentication frames")
        payload, header, mac = frames[:-2], frames[-2], frames[-1]
        # Cheap structural checks first; the MAC is computed only for
        # well-formed headers that name a known key.
        if len(mac) != self.MAC_SIZE or len(header) < self._HEADER.size:
            raise FrameAuthenticationError("Malformed authentication frames")
        version, key_id, msg_flow, seq, stream_id, sender_len = self._HEADER.unpack_from(header)
        if version != self._VERSION or len(header) != self._HEADER.size + sender_len:
            raise FrameAuthenticationError("Malformed authentication header")
        if msg_flow != flow:
            raise FrameAuthenticationError("Message was sent on the other flow")
        if key_id not in self._mac_bases:
            raise FrameAuthenticationError(f"Unknown key id {key_id}")
        if not hmac.compare_digest(mac, self._mac(key_id, header, recipient_id, payload)):
            raise FrameAuthenticationError("MAC mismatch")
        sender_id = header[self._HEADER.size :]
        return list(payload), SignedHeader(key_id, msg_flow, seq, stream_id, sender_id)

    def _new_mac_base(self, key: bytes) -> hmac.HMAC:
        base = hmac.new(key, digestmod=hashlib.sha256)
        base.update(self._LEN.pack(len(self._context)))
        base.update(self._context)
        return base

    def _mac(self, key_id: int, header: bytes, recipient_id: bytes, frames: list[bytes]) -> bytes:
        mac = self._mac_bases[key_id].copy()
        for part in (header, recipient_id, *frames):
            mac.update(self._LEN.pack(len(part)))
            mac.update(part)
        return mac.digest()


class ZMQMessenger(MessengerInterface):
    """ZMQ socket wrapper used by the native transfer control plane.

    Messages are unauthenticated unless a ``signer`` is supplied. Passing
    ``signer=None`` is an explicit opt-out: never bind a signer-less ROUTER on a
    non-loopback address, because any host that can reach the port can then
    forge every control message.

    With a signer, only ROUTER and DEALER modes are supported. Every outgoing
    message gets ``[header, mac]`` appended, and every incoming message is
    verified and stripped before it reaches ``receive()`` callers or the
    listener callback, so handlers see the same frames as without a signer.

    Args:
        mode: One of ``SOCKET_MODES``.
        endpoint: Bind address for ROUTER/REP, connect address for DEALER/REQ.
        signer: Message authenticator, or None to send and accept plain frames.
        identity: This component's id in signed messages. A ROUTER defaults to
            its bound endpoint. A signed DEALER must be given the endpoint of
            its owner's ROUTER, which is how peers address replies to it.
    """

    # Bound on routing ids remembered for replies. Entries are added only
    # after a MAC verifies, so an unauthenticated sender cannot grow it.
    _MAX_PEER_IDS = 4096
    _AUTH_WARNING_INTERVAL_S = 10.0

    SOCKET_MODES = {
        "ROUTER": zmq.ROUTER,  # Handles multiple connections and routes messages by address.
        "DEALER": zmq.DEALER,  # Load balances outgoing messages and receives replies fairly.
        "REQ": zmq.REQ,  # Sends requests and waits for replies (synchronous).
        "REP": zmq.REP,  # Receives requests and sends replies (synchronous).
    }

    def __init__(
        self,
        mode: str,
        endpoint: Optional[str] = None,
        *,
        signer: Optional[FrameSigner],
        identity: Optional[str] = None,
    ) -> None:
        if mode not in self.SOCKET_MODES:
            raise ValueError(
                f"Invalid mode '{mode}'. Allowed modes are {list(self.SOCKET_MODES.keys())}"
            )
        if signer is not None:
            if mode not in ("ROUTER", "DEALER"):
                raise ValueError("Signed messaging supports only ROUTER and DEALER modes")
            if mode == "DEALER" and not identity:
                raise ValueError("A signed DEALER requires the identity of its owner")
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
            if mode in ["DEALER", "REQ"]:
                raise ValueError("endpoint is required for DEALER/REQ modes")
            endpoint = f"tcp://{get_local_ip()}:*"

        if mode in ["ROUTER", "REP"]:
            self._socket.bind(endpoint)
            self._endpoint = self._socket.getsockopt_string(zmq.LAST_ENDPOINT)
        elif mode in ["DEALER", "REQ"]:
            self._socket.connect(endpoint)
            self._endpoint = endpoint

        self._signer = signer
        self._identity = (identity or self._endpoint).encode("utf-8")
        self._peer_ids: "OrderedDict[bytes, bytes]" = OrderedDict()
        self._peer_ids_lock = Lock()
        self.auth_failures = 0
        self._last_auth_warning = float("-inf")

        logger.info(
            f"Initialized ZMQMessenger(mode={mode}, endpoint={self._endpoint}, "
            f"signed={signer is not None})"
        )

    def _initialize_control_sockets(self) -> None:
        self._control_socket = self._context.socket(zmq.PAIR)
        self._internal_socket = self._context.socket(zmq.PAIR)
        inproc_endpoint = "inproc://stop_listener"
        self._control_socket.bind(inproc_endpoint)
        self._internal_socket.connect(inproc_endpoint)

    def start(self) -> None:
        pass

    def send(self, messages: list[bytes], recipient: Optional[bytes] = None) -> None:
        if self._signer is None:
            if recipient:
                self._socket.send_multipart([recipient] + messages)
            else:
                self._socket.send_multipart(messages)
            return
        if self._mode == "ROUTER":
            # The routing id comes either as `recipient` or as the first frame.
            if recipient:
                routing_id, payload = recipient, list(messages)
            else:
                routing_id, payload = messages[0], list(messages[1:])
            with self._peer_ids_lock:
                peer_id = self._peer_ids.get(routing_id)
            if peer_id is None:
                # Replies go only to peers that already authenticated to us.
                raise FrameAuthenticationError("No authenticated peer for this routing id")
            auth = self._signer.sign(
                payload,
                sender_id=self._identity,
                recipient_id=peer_id,
                flow=FrameSigner.FLOW_TO_DEALER,
            )
            self._socket.send_multipart([routing_id, *payload, *auth])
        else:
            payload = ([recipient] if recipient else []) + list(messages)
            auth = self._signer.sign(
                payload,
                sender_id=self._identity,
                recipient_id=self._endpoint.encode("utf-8"),
                flow=FrameSigner.FLOW_TO_ROUTER,
            )
            self._socket.send_multipart([*payload, *auth])

    def receive(self, timeout_ms: Optional[int] = None) -> list[bytes]:
        """Receive one message; with a signer, verify and strip its auth frames.

        Raises:
            TimeoutError: No message arrived within ``timeout_ms``.
            FrameAuthenticationError: The message failed verification.
        """
        if timeout_ms is not None and not self._socket.poll(timeout_ms, zmq.POLLIN):
            raise TimeoutError("Timed out waiting for a peer response")
        frames = self._socket.recv_multipart()
        if self._signer is None:
            return frames
        return self._verify(frames)

    def _verify(self, frames: list[bytes]) -> list[bytes]:
        if self._mode == "ROUTER":
            if not frames:
                raise FrameAuthenticationError("Empty message")
            routing_id = frames[0]
            payload, header = self._signer.verify(
                frames[1:], recipient_id=self._identity, flow=FrameSigner.FLOW_TO_ROUTER
            )
            with self._peer_ids_lock:
                self._peer_ids[routing_id] = header.sender_id
                self._peer_ids.move_to_end(routing_id)
                while len(self._peer_ids) > self._MAX_PEER_IDS:
                    self._peer_ids.popitem(last=False)
            return [routing_id, *payload]
        payload, header = self._signer.verify(
            frames, recipient_id=self._identity, flow=FrameSigner.FLOW_TO_DEALER
        )
        # A DEALER talks to exactly one ROUTER: the reply must come from it.
        if header.sender_id != self._endpoint.encode("utf-8"):
            raise FrameAuthenticationError("Reply was not sent by the connected peer")
        return payload

    def _record_auth_failure(self, error: FrameAuthenticationError) -> None:
        # Drop the frame and keep listening: stopping here would let one forged
        # frame take the listener down. Never log frame contents or keys.
        self.auth_failures += 1
        now = time.monotonic()
        if now - self._last_auth_warning >= self._AUTH_WARNING_INTERVAL_S:
            self._last_auth_warning = now
            logger.warning(
                f"Dropped unauthenticated control message on {self._endpoint}: {error} "
                f"(total dropped: {self.auth_failures})"
            )

    def start_listener(
        self,
        on_message: Callable[[list[bytes]], Optional[bool]],
        on_error: Optional[Callable[[Exception], None]] = None,
    ) -> None:
        assert self._mode in ["ROUTER", "REP"], (
            "Listener can only be started in ROUTER or REP modes"
        )
        if self._listener_thread and self._listener_thread.is_alive():
            raise RuntimeError("Listener already running")

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
            poller.register(self._socket, zmq.POLLIN)
            poller.register(self._control_socket, zmq.POLLIN)

            while not self._stop_event.is_set():
                events = dict(poller.poll(timeout=100))
                try:
                    if self._control_socket in events:
                        self._stop_event.set()
                    elif self._socket in events:
                        try:
                            messages = self.receive()
                        except FrameAuthenticationError as e:
                            self._record_auth_failure(e)
                            continue
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

        logger.info(f"Starting Messenger listener thread for {self._endpoint}")
        self._listener_thread = Thread(target=listener, daemon=True)
        self._listener_thread.start()

    def stop(self, timeout: int = 5) -> None:
        def _close_socket(socket: zmq.Socket) -> None:
            try:
                if not socket.closed:
                    socket.setsockopt(zmq.LINGER, 0)
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
                self._internal_socket.send(b"STOP")
                self._listener_thread.join(timeout)
                if self._listener_thread.is_alive():
                    logger.warning("Listener thread did not terminate within timeout")

            _close_socket(self._socket)
            _close_socket(self._internal_socket)
            _close_socket(self._control_socket)

            try:
                if not self._context.closed:
                    self._context.term()
            except Exception as e:
                logger.error(f"Error terminating ZMQ context: {e}")

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
