# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Authenticate Dynamo's parent-session hint before attention-DP placement."""

import hashlib
import hmac
import json
from collections.abc import Mapping

from .bindings import generation_pb2

AFFINITY_ID_HEADER = "x-trtllm-subagent-affinity-id"
AFFINITY_AUTH_HEADER = "x-trtllm-subagent-affinity-auth"


def validate_parent_affinity(
    key: str | None,
    request: generation_pb2.GenerateRequest,
    conversation_id: str | None,
    request_type: str | None,
    headers: Mapping[str, str],
) -> str | None:
    """Return the signed parent ID for a context request, if one was supplied."""
    parent_id = headers.get(AFFINITY_ID_HEADER)
    signature = headers.get(AFFINITY_AUTH_HEADER)
    if parent_id is None and signature is None:
        return None
    if not key or not parent_id or not signature:
        raise ValueError("Subagent affinity requires a configured key and signed parent ID")
    if not parent_id.strip() or parent_id != parent_id.strip():
        raise ValueError("Subagent affinity parent ID must be non-empty and trimmed")
    if not conversation_id or not conversation_id.strip():
        raise ValueError("Subagent affinity requires a stable child conversation ID")
    if request_type != "context_only":
        raise ValueError("Subagent affinity is supported on context requests only")
    payload = {
        "purpose": AFFINITY_ID_HEADER,
        "model": request.model,
        "request_id": request.request_id,
        "conversation_id": conversation_id,
        "subagent_affinity_id": parent_id,
        "request_type": request_type,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )
    expected = "sha256=" + hmac.new(key.encode("utf-8"), encoded, hashlib.sha256).hexdigest()
    if not hmac.compare_digest(signature, expected):
        raise ValueError("Invalid internal subagent affinity authentication")
    return parent_id
