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
from typing import Mapping, Optional

RUNTIME_CONTROL_AUTH_HEADER = "x-trtllm-runtime-control-auth"
_SIGNATURE_PREFIX = "sha256="


def _sign_request(runtime_control_api_key: str, body: bytes) -> str:
    signature = hmac.new(runtime_control_api_key.encode("utf-8"), body, hashlib.sha256).hexdigest()
    return f"{_SIGNATURE_PREFIX}{signature}"


def build_runtime_control_auth_headers(
    runtime_control_api_key: str,
    body: bytes,
) -> dict[str, str]:
    """Build the authentication header for an exact HTTP request body."""
    return {RUNTIME_CONTROL_AUTH_HEADER: _sign_request(runtime_control_api_key, body)}


def validate_runtime_control_request(
    runtime_control_api_key: Optional[str],
    body: bytes,
    headers: Optional[Mapping[str, str]],
) -> None:
    """Validate an HMAC signature over the exact HTTP request body."""
    if not runtime_control_api_key:
        raise ValueError("Runtime control endpoints are enabled but no API key is configured.")
    expected = _sign_request(runtime_control_api_key, body)
    provided = None if headers is None else headers.get(RUNTIME_CONTROL_AUTH_HEADER)
    if provided is None or not hmac.compare_digest(
        provided.encode("utf-8"), expected.encode("utf-8")
    ):
        raise ValueError("Invalid runtime control request authentication.")
