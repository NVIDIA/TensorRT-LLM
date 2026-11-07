# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""HMAC signatures for the records that servers register in a shared store."""

import hashlib
import hmac
import json
from typing import Any

_SIGNATURE_PREFIX = "sha256="
_SIGNATURE_HEX_LENGTH = 64  # SHA-256 hex digest


def sign_record(record_auth_key: str, key: str, value: Any) -> str:
    """Sign ``value`` as the record stored under ``key``; ``value`` must be JSON serializable."""
    payload = json.dumps(
        {"key": key, "value": value}, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    digest = hmac.new(record_auth_key.encode("utf-8"), payload, hashlib.sha256).hexdigest()
    return f"{_SIGNATURE_PREFIX}{digest}"


def _is_well_formed_signature(value: Any) -> bool:
    """Whether ``value`` has the form ``sign_record`` produces."""
    if not isinstance(value, str) or not value.startswith(_SIGNATURE_PREFIX):
        return False
    digest = value[len(_SIGNATURE_PREFIX) :]
    return len(digest) == _SIGNATURE_HEX_LENGTH and all(c in "0123456789abcdef" for c in digest)


def signature_matches(record_auth_key: str, key: str, value: Any, provided: Any) -> bool:
    """Whether ``provided`` is the signature of ``value`` as the record under ``key``."""
    return _is_well_formed_signature(provided) and hmac.compare_digest(
        provided, sign_record(record_auth_key, key, value)
    )
