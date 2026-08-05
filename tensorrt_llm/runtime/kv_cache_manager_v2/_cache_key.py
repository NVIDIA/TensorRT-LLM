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
"""Backend-neutral helpers for deriving KV-cache reuse keys.

These are the pure-Python reference implementation shared by both the Python
and C++ backends. They depend only on the light-weight, backend-neutral
``_common`` module (no CUDA / bindings), so they can be imported and re-exported
as public API regardless of the active backend.
"""

import hashlib
import itertools
from array import array
from typing import Iterable, Iterator, Sequence

from ._common import NDEBUG, TokenId, TokenIdExt

BlockKey = bytes
TokenBlock = list[TokenIdExt]


# id_offset is usually vocab_size
def gen_multimodal_cache_key_tokens(
    id_offset: int, multi_modal_data_digest: bytes, num_tokens: int, token_offset: int = 0
) -> list[TokenIdExt]:
    """Create synthetic tokens used only when building multimodal KV-cache keys.

    Item-local token 0 carries the content digest; later offsets use deterministic IDs above the vocab.
    """
    assert num_tokens > 0
    assert token_offset >= 0
    return [
        multi_modal_data_digest if token_offset + i == 0 else TokenId(id_offset + token_offset + i)
        for i in range(num_tokens)
    ]


class Hasher:
    # SECURITY INVARIANT: the block-key hash MUST stay cryptographically
    # collision-resistant and >= 256-bit. The radix tree is a globally shared,
    # cross-request/cross-tenant cache index; prefix matches are decided purely by
    # digest equality with NO re-check of the underlying tokens; and the hashed
    # input (tokens, the user-supplied cache_salt, multimodal content bytes) is
    # attacker-influenceable. A collision therefore silently reuses another
    # request's KV blocks (cross-request corruption / data leak), and cache_salt
    # tenant isolation relies entirely on this hash's collision resistance. Do NOT
    # swap in a non-cryptographic hash (xxHash, HighwayHash, ...) or truncate below
    # 256 bits without first adding a token-content equality check on match. The
    # C++ backend (blockRadixTree) mirrors this with SHA-256 (CSHA256).
    __slots__ = "_hasher"
    _hasher: "hashlib._Hash"

    def __init__(self, data: int | bytes | None | Sequence[int | bytes] = None) -> None:
        self._hasher = hashlib.sha256()
        if data is not None:
            self.update(data)

    # This function is perf-critical. Expect compromised code quality.
    def update(self, data: int | bytes | Sequence[int | bytes]) -> "Hasher":
        if type(data) is int:
            assert NDEBUG or (data >= 0 and data < (1 << 64))
            self._hasher.update(data.to_bytes(8, "little"))
        elif type(data) is bytes:
            self._hasher.update(data)
        else:
            # Hash the whole token block in one C call instead of one per token.
            # array("Q", data).tobytes() packs each int as 8 native-endian bytes;
            # all NVIDIA GPU host platforms (x86_64, aarch64/Grace) are little-endian
            # so this is byte-identical to the per-token to_bytes(8, "little") loop.
            # Falls back to that loop for multimodal blocks (which contain bytes items).
            try:
                self._hasher.update(array("Q", data).tobytes())  # type: ignore
            except (TypeError, OverflowError):
                for item in data:  # type: ignore
                    assert (
                        NDEBUG
                        or (type(item) is int and (0 <= item < (1 << 64)))
                        or type(item) is bytes
                    )
                    self._hasher.update(item.to_bytes(8, "little") if (type(item) is int) else item)  # type: ignore
        return self

    @property
    def digest(self) -> bytes:
        return self._hasher.digest()


def reuse_scope_to_bytes(reuse_scope: Iterable[int | None]) -> bytes:
    """Serialize a reuse scope to its reuse-namespace bytes.

    Backend-neutral: reads the scope's fields by iteration, so it works for both
    the pure-Python ``ReuseScope`` NamedTuple and the C++ binding without relying
    on a ``to_bytes()`` method. The layout mirrors the C++ ``emitReuseScopeBytes``:
    a mask byte (one bit per field, set when the field is present) followed by one
    little-endian ``uint64`` per present field (``signed=False``).
    """
    values = list(reuse_scope)
    mask = sum((value is not None) << i for i, value in enumerate(values))
    ret = mask.to_bytes((len(values) + 7) // 8, "little", signed=False)
    for value in values:
        if value is not None:
            ret += int(value).to_bytes(8, "little", signed=False)
    return ret


def sequence_to_blockchain_keys(
    tokens_per_block: int, reuse_scope: Iterable[int | None], tokens: Sequence[TokenIdExt]
) -> Iterator[tuple[TokenBlock, BlockKey]]:
    """Yield ``(token_block, key)`` pairs seeding a blockchain of KV-cache keys.

    The first pair is the root (``[]``, reuse-scope digest); each subsequent pair
    hashes one ``tokens_per_block`` chunk on top of the previous digest.
    """
    digest = Hasher(reuse_scope_to_bytes(reuse_scope)).digest
    yield [], digest
    iterator = iter(tokens)
    while True:
        token_block = list(itertools.islice(iterator, tokens_per_block))
        if not token_block:
            break
        digest = Hasher(digest).update(token_block).digest
        yield token_block, digest
