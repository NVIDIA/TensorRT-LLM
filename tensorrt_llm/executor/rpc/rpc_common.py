from typing import Any, NamedTuple, Optional


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


class RPCRequest(NamedTuple):
    request_id: str
    method_name: str
    args: tuple
    kwargs: dict
    need_response: bool = True
    timeout: float = 0.5


class RPCResponse(NamedTuple):
    request_id: str
    result: Any
    error: Optional[RPCError] = None
