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

from collections.abc import Sequence
from typing import Optional

KV_CACHE_HASH_ALGO_AUTO = "auto"
KV_CACHE_HASH_ALGO_V1 = "v1_block_key"
KV_CACHE_HASH_ALGO_V2 = "v2_sha256"
KV_CACHE_HASH_ALGO_V2_SHA256_64 = "v2_sha256_64"
KV_CACHE_HASH_ALGO_DEFAULT = KV_CACHE_HASH_ALGO_V1

_UINT32_MASK = (1 << 32) - 1
_UINT64_MASK = (1 << 64) - 1
_HASH32_CONST = 0x45D9F3B
_HASH64_CONST_1 = 0xBF58476D1CE4E5B9
_HASH64_CONST_2 = 0x94D049BB133111EB
_HASH_COMBINE_CONST = 0x9E3779B9
_PARENT_HASH_CONST = 0xBF58476D1CE4E5B9


class NonTextTokenHashError(ValueError):
    """Raised when v1-compatible hashing receives a non-text token."""


def get_effective_kv_cache_event_hash_algo(hash_algo: str, use_kv_cache_manager_v2: bool) -> str:
    if hash_algo != KV_CACHE_HASH_ALGO_AUTO:
        return hash_algo
    return KV_CACHE_HASH_ALGO_DEFAULT


def truncate_sha256_hash_to_int64(block_hash: bytes) -> int:
    return int.from_bytes(block_hash[:8], "big", signed=False)


def get_cache_salt_id(cache_salt: str) -> int:
    """Return the cache salt id used by request handling and cache-aware routing."""
    from blake3 import blake3

    h = blake3(cache_salt.encode("utf-8")).digest(length=8)
    return int.from_bytes(h, "little", signed=False)


def hash_v1_block_key(
    tokens: Sequence[int],
    parent_hash: int = 0,
    lora_task_id: Optional[int] = None,
    cache_salt_id: Optional[int] = None,
) -> int:
    seed = (len(tokens) ^ ((parent_hash * _PARENT_HASH_CONST) & _UINT64_MASK)) & _UINT64_MASK
    if parent_hash == 0 and cache_salt_id is not None:
        seed = _hash64_mix(cache_salt_id, seed)
    for token in tokens:
        if type(token) is not int:
            raise NonTextTokenHashError("v1-compatible hashing only supports text tokens")
        seed = _hash32_mix(token, seed)
    if lora_task_id is not None:
        seed = _hash64_mix(lora_task_id, seed)
    return seed


def _hash32_mix(value: int, seed: int) -> int:
    value &= _UINT32_MASK
    value = (((value >> 16) ^ value) * _HASH32_CONST) & _UINT32_MASK
    value = (((value >> 16) ^ value) * _HASH32_CONST) & _UINT32_MASK
    value = ((value >> 16) ^ value) & _UINT32_MASK
    # In C++, value and _HASH_COMBINE_CONST are both 32-bit unsigned values,
    # so this part wraps to uint32_t before size_t terms are added.
    value = (value + _HASH_COMBINE_CONST) & _UINT32_MASK
    combined = (value + ((seed << 6) & _UINT64_MASK) + (seed >> 2)) & _UINT64_MASK
    return (seed ^ combined) & _UINT64_MASK


def _hash64_mix(value: int, seed: int) -> int:
    value &= _UINT64_MASK
    value = ((value ^ (value >> 30)) * _HASH64_CONST_1) & _UINT64_MASK
    value = ((value ^ (value >> 27)) * _HASH64_CONST_2) & _UINT64_MASK
    value = (value ^ (value >> 31)) & _UINT64_MASK
    combined = (
        value + _HASH_COMBINE_CONST + ((seed << 6) & _UINT64_MASK) + (seed >> 2)
    ) & _UINT64_MASK
    return (seed ^ combined) & _UINT64_MASK
