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

from tensorrt_llm.llmapi.disagg_utils import ServerRole
from tensorrt_llm.serve.conversation_id import (
    SUBAGENT_AFFINITY_HEADER,
    extract_subagent_affinity_id_from_headers,
    get_request_conversation_id,
    get_request_subagent_affinity_id,
)
from tensorrt_llm.serve.openai_protocol import UCompletionRequest

INTERNAL_DISAGG_AUTH_HEADER = "x-trtllm-disagg-auth"
SUBAGENT_AFFINITY_AUTH_HEADER = "x-trtllm-subagent-affinity-auth"
_SIGNATURE_PREFIX = "sha256="
_INTERNAL_DISAGG_AUTH_FIELDS = ("encoded_opaque_state", "ctx_info_endpoint")
_MISSING_AUTH_KEY_WARNING = (
    "Internal disaggregated authentication key is required for protected "
    "disaggregated request fields. In a future release the requirement to "
    "use internal_request_auth_key will be enforced. Please update workflow "
    "accordingly."
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


def _auth_payload(request: UCompletionRequest) -> bytes:
    disaggregated_params = request.disaggregated_params
    payload = {
        field_name: _canonical_ctx_info_endpoint(value)
        if field_name == "ctx_info_endpoint"
        else value
        for field_name in get_internal_disagg_auth_fields()
        for value in [getattr(disaggregated_params, field_name)]
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _sign_request(internal_disagg_auth_key: str, request: UCompletionRequest) -> str:
    signature = hmac.new(
        internal_disagg_auth_key.encode("utf-8"), _auth_payload(request), hashlib.sha256
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
    if provided is None or not hmac.compare_digest(
        provided.encode("utf-8"), expected.encode("utf-8")
    ):
        raise ValueError("Invalid internal disaggregated request authentication")


def _sign_subagent_affinity(
    key: str, request: UCompletionRequest, role: ServerRole, affinity_id: str
) -> str:
    params = request.disaggregated_params
    # Use a separate signature to preserve the existing KV-transfer protocol
    # during rolling upgrades. Bind the routing hint to its destination and
    # child's identity rather than issuing a reusable signature of the parent ID.
    payload = {
        "purpose": SUBAGENT_AFFINITY_HEADER,
        "role": role.name,
        "model": request.model,
        "conversation_id": get_request_conversation_id(request),
        "subagent_affinity_id": affinity_id,
        "request_type": None if params is None else params.request_type,
        "disagg_request_id": None if params is None else params.disagg_request_id,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    signature = hmac.new(key.encode("utf-8"), encoded, hashlib.sha256).hexdigest()
    return f"{_SIGNATURE_PREFIX}{signature}"


def build_subagent_affinity_headers(
    key: str | None, request: UCompletionRequest, role: ServerRole
) -> dict[str, str]:
    """Authenticate the edge's affinity hint for a context or generation worker."""
    affinity_id = get_request_subagent_affinity_id(request)
    affinity_id = None if affinity_id is None else affinity_id.strip()
    if not affinity_id or role not in (ServerRole.CONTEXT, ServerRole.GENERATION):
        return {}
    if not key:
        raise ValueError("Subagent affinity requires internal_request_auth_key")
    return {
        SUBAGENT_AFFINITY_HEADER: affinity_id,
        SUBAGENT_AFFINITY_AUTH_HEADER: _sign_subagent_affinity(key, request, role, affinity_id),
    }


def validate_subagent_affinity(
    key: str | None,
    request: UCompletionRequest,
    role: ServerRole | None,
    headers: Mapping[str, str] | None,
) -> str | None:
    """Authenticate a worker routing hint, inferring an absent role from the request."""
    if role is None and request.disaggregated_params is not None:
        # Standalone workers may serve CTX/GEN requests without --server_role.
        # This identifies which signature to verify; it does not authorize the hint.
        role = {
            "context_only": ServerRole.CONTEXT,
            "generation_only": ServerRole.GENERATION,
        }.get(request.disaggregated_params.request_type)
    if headers is None or role is None or role not in (ServerRole.CONTEXT, ServerRole.GENERATION):
        return None
    affinity_id = extract_subagent_affinity_id_from_headers(headers)
    if affinity_id is None:
        return None
    if not key:
        raise ValueError("Subagent affinity requires internal_request_auth_key")
    expected = _sign_subagent_affinity(key, request, role, affinity_id)
    lower_headers = {name.lower(): value for name, value in headers.items()}
    provided = lower_headers.get(SUBAGENT_AFFINITY_AUTH_HEADER)
    if provided is None or not hmac.compare_digest(
        provided.encode("utf-8"), expected.encode("utf-8")
    ):
        raise ValueError("Invalid internal subagent affinity authentication")
    return affinity_id
