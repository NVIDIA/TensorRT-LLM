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
"""Sign rendezvous store values with the Wide-EP deployment's shared secret."""

import hashlib
import hmac
import os
from datetime import timedelta
from typing import TYPE_CHECKING, List, Optional

from tensorrt_llm._wide_ep_fault_tolerance import SHARED_SECRET_ENV
from tensorrt_llm.math_utils import ceil_div

if TYPE_CHECKING:
    from torch.distributed import Store

_TAG_BYTES = hashlib.sha256().digest_size
_CHUNK_COUNT_BYTES = 4
# Values above this size are split into chunks of this size, half the 8 MiB value limit of torch's
# default (libuv) TCPStore server.
_CHUNK_BYTES = 4 << 20
SHARED_SECRET_BYTES = 32


class UnsignedValueError(ValueError):
    """A rendezvous store value is not signed with the deployment's shared secret."""


def parse_shared_secret(secret: str) -> bytes:
    """Parse the hex shared secret ``secret``.

    Raises:
        ValueError: ``secret`` is not hex or is shorter than ``SHARED_SECRET_BYTES``.
    """
    try:
        parsed = bytes.fromhex(secret)
    except ValueError:
        parsed = b""
    if len(parsed) < SHARED_SECRET_BYTES:
        raise ValueError(
            f"Wide-EP fault tolerance requires a shared secret of at least {SHARED_SECRET_BYTES} "
            f"bytes ({2 * SHARED_SECRET_BYTES} hex digits) in {SHARED_SECRET_ENV}"
        )
    return parsed


def shared_secret_digest(label: bytes) -> str:
    """Return the hex HMAC-SHA256 of ``label`` keyed by the shared secret in ``SHARED_SECRET_ENV``."""
    secret = parse_shared_secret(os.environ[SHARED_SECRET_ENV])
    return hmac.new(secret, label, hashlib.sha256).hexdigest()


class SignedStore:
    """A rendezvous store client that signs the values it sets and verifies those it gets.

    A value's key holds an HMAC-SHA256 tag, keyed by a digest of the shared secret, over the key, a
    chunk count and the value. A process without the secret cannot set a value that the deployment's
    processes accept. Counters (``add``), deletes and waits carry no tag, and an older value that
    the deployment set under a key verifies again if a process sets it back. A process that reaches
    the store's port can therefore stall or end the deployment, or make it act on an older record,
    as it can with an unsigned TCPStore: the store is meant for a trusted network.

    A value larger than ``_CHUNK_BYTES`` is set in chunks under ``<key>/chunks/<i>``, and ``key``
    holds their count and the value's SHA-256 digest in place of the value. The chunks are set
    before ``key``, so a process that waits for ``key`` finds them. A value under one key has a
    chunk count of 0.
    """

    def __init__(self, store: "Store", secret: Optional[str] = None):
        """Sign the values of ``store`` with ``secret`` (hex), by default ``SHARED_SECRET_ENV``.

        Raises:
            ValueError: ``secret`` is not a valid shared secret (:func:`parse_shared_secret`).
        """
        secret = os.environ[SHARED_SECRET_ENV] if secret is None else secret
        self._store = store
        self._key = hmac.new(
            parse_shared_secret(secret), b"rendezvous store values", hashlib.sha256
        ).digest()

    @property
    def host(self) -> str:
        return self._store.host

    @property
    def port(self) -> int:
        return self._store.port

    def set(self, key: str, value: bytes) -> None:
        value = bytes(value)
        chunks, head = 0, value
        if len(value) > _CHUNK_BYTES:
            chunks = ceil_div(len(value), _CHUNK_BYTES)
            for i in range(chunks):
                chunk = value[i * _CHUNK_BYTES : (i + 1) * _CHUNK_BYTES]
                self._store.set(_chunk_key(key, i), chunk)
            head = hashlib.sha256(value).digest()
        signed = chunks.to_bytes(_CHUNK_COUNT_BYTES, "big") + head
        self._store.set(key, self._tag(key, signed) + signed)

    def get(self, key: str) -> bytes:
        """Wait until ``key`` is set and return its value.

        Raises:
            UnsignedValueError: The value is not signed with the deployment's shared secret.
        """
        return self._value(key, self._store.get(key))

    def multi_get(self, keys: List[str]) -> List[bytes]:
        """Return the values for ``keys``, as :meth:`get` does."""
        return [self._value(k, v) for k, v in zip(keys, self._store.multi_get(keys))]

    def check(self, keys: List[str]) -> bool:
        return self._store.check(keys)

    def wait(self, keys: List[str], timeout: timedelta) -> None:
        self._store.wait(keys, timeout)

    def add(self, key: str, amount: int) -> int:
        return self._store.add(key, amount)

    def delete_key(self, key: str) -> bool:
        """Delete ``key`` and its chunks, and return whether ``key`` was set."""
        deleted = self._store.delete_key(key)
        chunk = 0
        while self._store.delete_key(_chunk_key(key, chunk)):
            chunk += 1
        return deleted

    def _tag(self, key: str, value: bytes) -> bytes:
        encoded = key.encode()
        message = len(encoded).to_bytes(4, "big") + encoded + value
        return hmac.new(self._key, message, hashlib.sha256).digest()

    def _value(self, key: str, stored: bytes) -> bytes:
        """Return the value of ``key`` from its ``stored`` bytes and its chunks, if it has any."""
        tag, signed = stored[:_TAG_BYTES], stored[_TAG_BYTES:]
        if not hmac.compare_digest(tag, self._tag(key, signed)):
            raise UnsignedValueError(
                f"Rendezvous store value {key!r} is not signed with this deployment's "
                f"{SHARED_SECRET_ENV}, so another process set it"
            )
        chunks = int.from_bytes(signed[:_CHUNK_COUNT_BYTES], "big")
        head = signed[_CHUNK_COUNT_BYTES:]
        if chunks == 0:
            return head
        value = b"".join(self._store.get(_chunk_key(key, i)) for i in range(chunks))
        if not hmac.compare_digest(hashlib.sha256(value).digest(), head):
            raise UnsignedValueError(
                f"The chunks of rendezvous store value {key!r} do not match its signed digest, so "
                "another process set them or the value changed while it was read"
            )
        return value


def _chunk_key(key: str, index: int) -> str:
    return f"{key}/chunks/{index}"
