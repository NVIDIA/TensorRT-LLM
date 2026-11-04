# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Store keys for blob objects.

A private encoding of the lender's opaque object keys, not a stable public format. The key
carries ``FORMAT_VERSION``, which covers the key spelling and the payload layout alike: after an
incompatible change, objects written under the old version are never looked up, so they miss
instead of being misread.
"""

from __future__ import annotations

import re

FORMAT_VERSION = 1

_PREFIX = re.compile(r"[A-Za-z0-9_-]{1,64}")
_SEPARATOR = "/"


def check_key_prefix(prefix: str) -> None:
    """Raises ``ValueError`` unless ``prefix`` matches ``[A-Za-z0-9_-]{1,64}``."""
    if not _PREFIX.fullmatch(prefix):
        raise ValueError(f"bad key prefix {prefix!r}: expected [A-Za-z0-9_-]{{1,64}}")


def key_stem(prefix: str) -> str:
    """``"<prefix>/v<FORMAT_VERSION>/"``, how every key under ``prefix`` starts. A caller encoding
    many object keys checks the prefix once and appends each object key's hex to the stem.

    Args:
        prefix: As for ``encode_key``.
    """
    check_key_prefix(prefix)
    return f"{prefix}{_SEPARATOR}v{FORMAT_VERSION}{_SEPARATOR}"


def encode_key(prefix: str, object_key: bytes) -> str:
    """``"<prefix>/v<FORMAT_VERSION>/<hex of object_key>"``.

    Args:
        prefix: The store partition the objects live in, for example a deployment name; matches
            ``[A-Za-z0-9_-]{1,64}``.
        object_key: The lender's opaque object key; encoded whole, never parsed.
    """
    stem = key_stem(prefix)
    if not object_key:
        raise ValueError("an object key is non-empty")
    return stem + object_key.hex()
