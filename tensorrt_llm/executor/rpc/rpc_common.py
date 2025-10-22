import os
import tempfile
import time
import uuid
from dataclasses import dataclass
from typing import Any, Literal, NamedTuple, Optional


def get_unique_ipc_addr() -> str:
    """Generate a cryptographically unique IPC address using UUID."""
    # uuid.uuid4() generates a random, unique identifier
    unique_id = uuid.uuid4()
    temp_dir = tempfile.gettempdir()
    file_name = f"rpc_test_{unique_id}"
    full_path = os.path.join(temp_dir, file_name)
    return f"ipc://{full_path}"


class RPCParams(NamedTuple):
    """ Parameters for RPC calls. """

    # seconds to wait for the response
    timeout: Optional[float] = None

    # whether the client needs the response, if False, it will return immediately
    need_response: bool = True

    # mode for RPC calls: "sync", "async", or "future"
    mode: str = "sync"


# --- Custom Exceptions ---
class RPCError(Exception):
    """Custom exception for RPC-related errors raised on the client side.

    Args:
        message: The error message.
        cause: The original exception that caused this error.
        traceback: The traceback of the exception.
    """

    def __init__(self,
                 message: str,
                 cause: Optional[Exception] = None,
                 traceback: Optional[str] = None):
        super().__init__(message)
        self.cause = cause
        self.traceback = traceback


class RPCTimeout(RPCError):
    """Exception for when a request processing times out."""


class RPCCancelled(RPCError):
    """Exception for when a client request is cancelled.
    This happens when the server is shutting down and all the pending
    requests will be cancelled and return with this error.
    """


class RPCStreamingError(RPCError):
    """Exception for streaming-related errors."""


@dataclass
class RPCRequest:
    request_id: str
    method_name: str
    args: tuple
    kwargs: dict
    need_response: bool = True
    timeout: float = 0.5
    is_streaming: bool = False
    creation_timestamp: Optional[
        float] = None  # Unix timestamp when request was created

    def __post_init__(self):
        """Initialize creation_timestamp if not provided."""
        if self.creation_timestamp is None:
            self.creation_timestamp = time.time()


class RPCResponse(NamedTuple):
    request_id: str
    result: Any
    error: Optional[RPCError] = None
    is_streaming: bool = False  # True if more responses coming
    sequence_number: int = 0  # For ordering streaming responses
    stream_status: Literal['start', 'data', 'end', 'error'] = 'data'
