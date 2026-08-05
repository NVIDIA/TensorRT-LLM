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
import warnings
from typing import Any, Mapping, Optional

from tensorrt_llm.serve.openai_protocol import UCompletionRequest

INTERNAL_DISAGG_AUTH_HEADER = "x-trtllm-disagg-auth"
_SIGNATURE_PREFIX = "sha256="
_INTERNAL_DISAGG_AUTH_FIELDS = (
    "encoded_opaque_state",
    "ctx_info_endpoint",
    "logical_request_id",
    "prefill_artifact_id",
    "artifact_version",
    "handoff_attempt_uuid",
    "consumer_grant_id",
    "transfer_session_id",
    "generation_endpoint_name",
    "generation_endpoint_rank",
    "generation_endpoint_incarnation",
    "context_control_endpoint",
    "context_transceiver_lifecycle",
)
_INTERNAL_DISAGG_SIGNED_FIELDS = (
    *_INTERNAL_DISAGG_AUTH_FIELDS,
    "request_type",
    "ctx_dp_rank",
    "schedule_style",
)
_MISSING_AUTH_KEY_WARNING = (
    "Internal disaggregated authentication key is required for protected "
    "disaggregated request fields and lifecycle control routes. In a future "
    "release the requirement to use internal_request_auth_key will be "
    "enforced. Please update workflow accordingly."
)


def get_internal_disagg_auth_fields() -> tuple[str, ...]:
    return _INTERNAL_DISAGG_AUTH_FIELDS


def _warn_missing_auth_key() -> None:
    warnings.warn(_MISSING_AUTH_KEY_WARNING, FutureWarning, stacklevel=2)


def request_requires_internal_disagg_auth(request: UCompletionRequest) -> bool:
    disaggregated_params = getattr(request, "disaggregated_params", None)
    return disaggregated_params is not None and any(
        getattr(disaggregated_params, field_name) is not None
        for field_name in get_internal_disagg_auth_fields()
    )


def _canonical_ctx_info_endpoint(endpoint: Any) -> Any:
    if isinstance(endpoint, list):
        return endpoint[0] if endpoint else None
    return endpoint


def _canonical_json(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _auth_payload(request: UCompletionRequest) -> bytes:
    disaggregated_params = request.disaggregated_params
    serialized_params = disaggregated_params.model_dump(mode="json")
    payload = {
        field_name: _canonical_ctx_info_endpoint(value)
        if field_name == "ctx_info_endpoint"
        else value
        for field_name in _INTERNAL_DISAGG_SIGNED_FIELDS
        for value in [serialized_params.get(field_name)]
    }
    return _canonical_json(payload)


def _lifecycle_auth_payload(route: str, body: Mapping[str, Any]) -> bytes:
    return _canonical_json({"body": body, "route": route})


def _sign_request(internal_disagg_auth_key: str, request: UCompletionRequest) -> str:
    signature = hmac.new(
        internal_disagg_auth_key.encode("utf-8"), _auth_payload(request), hashlib.sha256
    ).hexdigest()
    return f"{_SIGNATURE_PREFIX}{signature}"


def _sign_lifecycle_control(
    internal_disagg_auth_key: str,
    route: str,
    body: Mapping[str, Any],
) -> str:
    signature = hmac.new(
        internal_disagg_auth_key.encode("utf-8"),
        _lifecycle_auth_payload(route, body),
        hashlib.sha256,
    ).hexdigest()
    return f"{_SIGNATURE_PREFIX}{signature}"


def build_internal_disagg_auth_headers(
    internal_disagg_auth_key: Optional[str],
    request: UCompletionRequest,
) -> dict[str, str]:
    if not request_requires_internal_disagg_auth(request):
        return {}
    if not internal_disagg_auth_key:
        _warn_missing_auth_key()
        return {}
    return {INTERNAL_DISAGG_AUTH_HEADER: _sign_request(internal_disagg_auth_key, request)}


def build_internal_disagg_lifecycle_auth_headers(
    internal_disagg_auth_key: Optional[str],
    route: str,
    body: Mapping[str, Any],
) -> dict[str, str]:
    if not internal_disagg_auth_key:
        _warn_missing_auth_key()
        return {}
    return {
        INTERNAL_DISAGG_AUTH_HEADER: _sign_lifecycle_control(
            internal_disagg_auth_key,
            route,
            body,
        )
    }


def validate_internal_disagg_request(
    internal_disagg_auth_key: Optional[str],
    request: UCompletionRequest,
    headers: Optional[Mapping[str, str]],
) -> None:
    if not request_requires_internal_disagg_auth(request):
        return
    if not internal_disagg_auth_key:
        _warn_missing_auth_key()
        return

    expected = _sign_request(internal_disagg_auth_key, request)
    provided = None if headers is None else headers.get(INTERNAL_DISAGG_AUTH_HEADER)
    if provided is None or not hmac.compare_digest(provided, expected):
        raise ValueError("Invalid internal disaggregated request authentication")


def validate_internal_disagg_lifecycle_request(
    internal_disagg_auth_key: Optional[str],
    route: str,
    body: Mapping[str, Any],
    headers: Optional[Mapping[str, str]],
) -> None:
    if not internal_disagg_auth_key:
        _warn_missing_auth_key()
        return

    expected = _sign_lifecycle_control(internal_disagg_auth_key, route, body)
    provided = None if headers is None else headers.get(INTERNAL_DISAGG_AUTH_HEADER)
    if provided is None or not hmac.compare_digest(provided, expected):
        raise ValueError("Invalid internal disaggregated lifecycle authentication")
