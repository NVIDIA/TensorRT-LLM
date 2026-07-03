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

from tensorrt_llm.serve.openai_protocol import UCompletionRequest

INTERNAL_DISAGG_AUTH_HEADER = "x-trtllm-disagg-auth"
_SIGNATURE_PREFIX = "sha256="


def request_requires_internal_disagg_auth(request: UCompletionRequest) -> bool:
    disaggregated_params = getattr(request, "disaggregated_params", None)
    return (disaggregated_params is not None
            and disaggregated_params.encoded_opaque_state is not None)


def _auth_payload(request: UCompletionRequest) -> bytes:
    disaggregated_params = request.disaggregated_params
    payload = disaggregated_params.model_dump(mode="json",
                                              exclude_none=False)
    return json.dumps(payload,
                      sort_keys=True,
                      separators=(",", ":")).encode("utf-8")


def _sign_request(internal_disagg_auth_key: str,
                  request: UCompletionRequest) -> str:
    signature = hmac.new(internal_disagg_auth_key.encode("utf-8"),
                         _auth_payload(request),
                         hashlib.sha256).hexdigest()
    return f"{_SIGNATURE_PREFIX}{signature}"


def build_internal_disagg_auth_headers(
        internal_disagg_auth_key: Optional[str],
        request: UCompletionRequest) -> dict[str, str]:
    if not request_requires_internal_disagg_auth(request):
        return {}
    if not internal_disagg_auth_key:
        raise ValueError(
            "encoded_opaque_state requires authenticated disaggregated request forwarding"
        )
    return {
        INTERNAL_DISAGG_AUTH_HEADER:
        _sign_request(internal_disagg_auth_key, request)
    }


def validate_internal_disagg_request(
        internal_disagg_auth_key: Optional[str], request: UCompletionRequest,
        headers: Mapping[str, str]) -> None:
    if not request_requires_internal_disagg_auth(request):
        return
    if not internal_disagg_auth_key:
        raise ValueError(
            "encoded_opaque_state requires authenticated disaggregated request forwarding"
        )

    expected = _sign_request(internal_disagg_auth_key, request)
    provided = headers.get(INTERNAL_DISAGG_AUTH_HEADER)
    if provided is None or not hmac.compare_digest(provided, expected):
        raise ValueError("Invalid internal disaggregated request authentication")
