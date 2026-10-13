# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Store keys for blob objects.

A private, versioned encoding of the lender's opaque object names, not a stable public format. An
incompatible change bumps ``KEY_FORMAT_VERSION`` so objects written under the old one miss instead
of being misread.
"""

from __future__ import annotations

KEY_FORMAT_VERSION = 1

_SEPARATOR = "/"


def encode_key(prefix: str, name: bytes) -> str:
    """``"<prefix>/v<version>/<hex of name>"``.

    Args:
        prefix: The deployment's namespace; non-empty, without ``/`` or whitespace.
        name: The lender's opaque object name; encoded whole, never parsed.
    """
    if not prefix or _SEPARATOR in prefix or any(c.isspace() for c in prefix):
        raise ValueError(f"bad key prefix {prefix!r}: non-empty, no '/' and no whitespace")
    if not name:
        raise ValueError("an object name is non-empty")
    return f"{prefix}{_SEPARATOR}v{KEY_FORMAT_VERSION}{_SEPARATOR}{name.hex()}"
