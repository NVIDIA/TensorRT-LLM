# Copyright (c) 2026, NVIDIA CORPORATION.
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
"""Auth for the RL control-plane endpoints (/release_memory, /resume_memory, /update_weights).

Same HMAC-over-body scheme as disagg_auth.py, but always enforced (no
warn-and-allow fallback): these endpoints can free GPU memory or replace
model weights.
"""

import hashlib
import hmac
from typing import Mapping, Optional

RL_CONTROL_AUTH_HEADER = "x-trtllm-rl-control-auth"
_SIGNATURE_PREFIX = "sha256="


def _sign_request(rl_control_api_key: str, body: bytes) -> str:
    signature = hmac.new(rl_control_api_key.encode("utf-8"), body, hashlib.sha256).hexdigest()
    return f"{_SIGNATURE_PREFIX}{signature}"


def build_rl_control_auth_headers(rl_control_api_key: str, body: bytes) -> dict[str, str]:
    return {RL_CONTROL_AUTH_HEADER: _sign_request(rl_control_api_key, body)}


def validate_rl_control_request(
    rl_control_api_key: Optional[str],
    body: bytes,
    headers: Optional[Mapping[str, str]],
) -> None:
    if not rl_control_api_key:
        raise ValueError("RL control endpoints are enabled but no rl_control_api_key is configured")
    expected = _sign_request(rl_control_api_key, body)
    provided = None if headers is None else headers.get(RL_CONTROL_AUTH_HEADER)
    if provided is None or not hmac.compare_digest(provided, expected):
        raise ValueError("Invalid RL control request authentication")
