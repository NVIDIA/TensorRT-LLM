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

"""Authentication for runtime-memory control endpoints."""

import hashlib
import hmac
import re
import secrets
import threading
import time
from typing import Mapping, Optional

RUNTIME_CONTROL_AUTH_HEADER = "x-trtllm-runtime-control-auth"
RUNTIME_CONTROL_NONCE_HEADER = "x-trtllm-runtime-control-nonce"
RUNTIME_CONTROL_TIMESTAMP_HEADER = "x-trtllm-runtime-control-timestamp"
RUNTIME_CONTROL_VALIDITY_WINDOW_SECONDS = 300
DEFAULT_RUNTIME_CONTROL_REPLAY_CACHE_CAPACITY = 4096
_SIGNATURE_PREFIX = "sha256="
_SIGNATURE_DOMAIN = b"trtllm-runtime-control-v1\n"
_NONCE_PATTERN = re.compile(r"[0-9a-f]{32}")


class RuntimeControlReplayCache:
    """Atomically retain accepted nonces for their full validity period."""

    def __init__(
        self,
        max_entries: int = DEFAULT_RUNTIME_CONTROL_REPLAY_CACHE_CAPACITY,
    ) -> None:
        if max_entries <= 0:
            raise ValueError("Runtime control replay cache capacity must be positive.")
        self._max_entries = max_entries
        self._expires_at_by_nonce: dict[str, int] = {}
        self._lock = threading.Lock()

    def claim(self, nonce: str, expires_at: int, current_time: int) -> None:
        """Claim a nonce or reject a duplicate without evicting live entries."""
        with self._lock:
            expired = [
                cached_nonce
                for cached_nonce, cached_expiry in self._expires_at_by_nonce.items()
                if cached_expiry < current_time
            ]
            for cached_nonce in expired:
                del self._expires_at_by_nonce[cached_nonce]

            if nonce in self._expires_at_by_nonce:
                raise ValueError("Runtime control request nonce has already been used.")
            if len(self._expires_at_by_nonce) >= self._max_entries:
                raise ValueError("Runtime control replay cache capacity exceeded.")
            self._expires_at_by_nonce[nonce] = expires_at


def _canonical_request(
    method: str,
    path: str,
    timestamp: str,
    nonce: str,
    body: bytes,
) -> bytes:
    canonical_method = method.upper()
    if not canonical_method.isascii() or not canonical_method.isalpha():
        raise ValueError("Invalid runtime control request authentication.")
    if (
        not path.startswith("/")
        or (len(path) > 1 and path.endswith("/"))
        or "?" in path
        or "#" in path
        or "\n" in path
    ):
        raise ValueError("Invalid runtime control request authentication.")
    if not timestamp.isascii() or not timestamp.isdecimal():
        raise ValueError("Invalid runtime control request authentication.")
    if _NONCE_PATTERN.fullmatch(nonce) is None:
        raise ValueError("Invalid runtime control request authentication.")

    return b"".join(
        (
            _SIGNATURE_DOMAIN,
            canonical_method.encode("ascii"),
            b"\n",
            path.encode("utf-8"),
            b"\n",
            timestamp.encode("ascii"),
            b"\n",
            nonce.encode("ascii"),
            b"\n",
            body,
        )
    )


def _sign_request(
    runtime_control_api_key: str,
    method: str,
    path: str,
    timestamp: str,
    nonce: str,
    body: bytes,
) -> str:
    payload = _canonical_request(method, path, timestamp, nonce, body)
    signature = hmac.new(
        runtime_control_api_key.encode("utf-8"), payload, hashlib.sha256
    ).hexdigest()
    return f"{_SIGNATURE_PREFIX}{signature}"


def build_runtime_control_auth_headers(
    runtime_control_api_key: str,
    method: str,
    path: str,
    body: bytes,
    *,
    timestamp: int | None = None,
    nonce: str | None = None,
) -> dict[str, str]:
    """Build authentication headers for an exact runtime-control request."""
    timestamp_value = str(int(time.time()) if timestamp is None else timestamp)
    nonce_value = secrets.token_hex(16) if nonce is None else nonce
    signature = _sign_request(
        runtime_control_api_key,
        method,
        path,
        timestamp_value,
        nonce_value,
        body,
    )
    return {
        RUNTIME_CONTROL_AUTH_HEADER: signature,
        RUNTIME_CONTROL_TIMESTAMP_HEADER: timestamp_value,
        RUNTIME_CONTROL_NONCE_HEADER: nonce_value,
    }


def validate_runtime_control_request(
    runtime_control_api_key: Optional[str],
    method: str,
    path: str,
    body: bytes,
    headers: Optional[Mapping[str, str]],
    replay_cache: RuntimeControlReplayCache,
    *,
    current_time: int | None = None,
) -> None:
    """Validate and atomically claim a signed runtime-control request."""
    if not runtime_control_api_key:
        raise ValueError("Runtime control endpoints are enabled but no API key is configured.")

    provided = None if headers is None else headers.get(RUNTIME_CONTROL_AUTH_HEADER)
    timestamp = None if headers is None else headers.get(RUNTIME_CONTROL_TIMESTAMP_HEADER)
    nonce = None if headers is None else headers.get(RUNTIME_CONTROL_NONCE_HEADER)
    if provided is None or timestamp is None or nonce is None:
        raise ValueError("Invalid runtime control request authentication.")

    expected = _sign_request(
        runtime_control_api_key,
        method,
        path,
        timestamp,
        nonce,
        body,
    )
    if not hmac.compare_digest(provided.encode("utf-8"), expected.encode("utf-8")):
        raise ValueError("Invalid runtime control request authentication.")

    now = int(time.time()) if current_time is None else current_time
    try:
        request_time = int(timestamp)
    except ValueError as error:
        raise ValueError("Invalid runtime control request timestamp.") from error
    if abs(now - request_time) > RUNTIME_CONTROL_VALIDITY_WINDOW_SECONDS:
        raise ValueError("Runtime control request timestamp is outside the validity window.")

    replay_cache.claim(
        nonce,
        request_time + RUNTIME_CONTROL_VALIDITY_WINDOW_SECONDS,
        now,
    )
