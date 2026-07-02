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

import hashlib
import hmac
import json
from typing import Mapping, Optional

INTERNAL_DISAGG_AUTH_HEADER = "x-trtllm-disagg-auth"


def request_requires_internal_disagg_auth(request) -> bool:
    params = getattr(request, "disaggregated_params", None)
    return params is not None and getattr(params, "ctx_info_endpoint", None) is not None


def _canonical_disagg_payload(request) -> bytes:
    params = getattr(request, "disaggregated_params", None)
    if params is None:
        return b"{}"

    payload = params.model_dump(mode="json", exclude_none=False)
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sign_internal_disagg_request(secret: str, request) -> str:
    return hmac.new(
        secret.encode("utf-8"), _canonical_disagg_payload(request), hashlib.sha256
    ).hexdigest()


def build_internal_disagg_auth_headers(secret: str, request) -> dict[str, str]:
    return {INTERNAL_DISAGG_AUTH_HEADER: sign_internal_disagg_request(secret, request)}


def verify_internal_disagg_request(
    secret: str, request, headers: Optional[Mapping[str, str]]
) -> bool:
    if headers is None:
        return False
    received_signature = headers.get(INTERNAL_DISAGG_AUTH_HEADER)
    if received_signature is None:
        return False
    expected_signature = sign_internal_disagg_request(secret, request)
    return hmac.compare_digest(received_signature, expected_signature)


def validate_internal_disagg_request(
    secret: Optional[str], request, headers: Optional[Mapping[str, str]]
) -> None:
    if not request_requires_internal_disagg_auth(request):
        return
    if not secret:
        raise ValueError("ctx_info_endpoint requires authenticated disaggregated proxy request")
    if not verify_internal_disagg_request(secret, request, headers):
        raise ValueError("Invalid internal disaggregated request authentication")
