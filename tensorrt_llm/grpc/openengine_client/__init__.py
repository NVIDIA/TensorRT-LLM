# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Async client for OpenEngine v1 servers.

The client speaks the OpenEngine wire protocol only: it connects, reads
ServerInfo and ModelInfo and checks the engine's schema revision, calls Health,
GetLoad and Abort, and streams Generate responses after validating them. It
sends and returns OpenEngine protobuf messages; mapping serving-API requests to
them is up to the caller.
"""

from .client import (
    MAX_MESSAGE_BYTES,
    SUPPORTED_SCHEMA_REVISION,
    AbortOutcome,
    OpenEngineClient,
    new_request_id,
)
from .errors import (
    OpenEngineClientError,
    OpenEngineEngineError,
    OpenEngineIncompatibleServerError,
    OpenEngineProtocolError,
    OpenEngineRpcError,
)
from .stream import GenerateStream

__all__ = [
    "MAX_MESSAGE_BYTES",
    "SUPPORTED_SCHEMA_REVISION",
    "AbortOutcome",
    "GenerateStream",
    "OpenEngineClient",
    "OpenEngineClientError",
    "OpenEngineEngineError",
    "OpenEngineIncompatibleServerError",
    "OpenEngineProtocolError",
    "OpenEngineRpcError",
    "new_request_id",
]
