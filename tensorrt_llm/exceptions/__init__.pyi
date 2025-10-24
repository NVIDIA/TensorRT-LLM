import enum


class RequestErrorCode(enum.Enum):
    UNKNOWN_ERROR = 0

    NETWORK_ERROR = 1000

UNKNOWN_ERROR: RequestErrorCode = RequestErrorCode.UNKNOWN_ERROR

NETWORK_ERROR: RequestErrorCode = RequestErrorCode.NETWORK_ERROR

class RequestSpecificException(Exception):
    request_id: None = None

    error_code: None = None
