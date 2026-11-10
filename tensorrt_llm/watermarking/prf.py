# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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
"""Versioned, device-independent random function for text watermarking.

Version 1 hashes the context's uint32 token words with MurmurHash3 x86_32,
using seeds 0x9747b28c and 0x5bd1e995. The Philox4x32-10 counter is
``(candidate_token, context_hash_0, context_hash_1, domain)``; the key is the
low and high uint32 words of the configured uint64 secret. Output word zero
maps to the open unit interval as ``(word + 0.5) / 2**32``. Domain zero is
the primary watermark, one the secondary watermark, and two the mixture
choice (whose candidate word is the generated-token position).

This mapping is independent of PyTorch RNG versions and tensor parallel
rank. It is a pseudorandom watermark construction, not a cryptographic
authentication mechanism. The detector must use the same tokenizer and
configuration as generation.
"""

from collections.abc import Sequence

PRF_VERSION = 1
UINT32_MASK = (1 << 32) - 1
PHILOX_M0 = 0xD2511F53
PHILOX_M1 = 0xCD9E8D57
PHILOX_W0 = 0x9E3779B9
PHILOX_W1 = 0xBB67AE85
CONTEXT_HASH_SEEDS = (0x9747B28C, 0x5BD1E995)


def _rotate_left(value: int, amount: int) -> int:
    return ((value << amount) | (value >> (32 - amount))) & UINT32_MASK


def context_hash(context: Sequence[int]) -> tuple[int, int]:
    """Return two MurmurHash3 x86_32 hashes over uint32 context words."""
    result = []
    for seed in CONTEXT_HASH_SEEDS:
        value = seed
        for token in context:
            word = ((int(token) & UINT32_MASK) * 0xCC9E2D51) & UINT32_MASK
            word = (_rotate_left(word, 15) * 0x1B873593) & UINT32_MASK
            value = _rotate_left(value ^ word, 13)
            value = (value * 5 + 0xE6546B64) & UINT32_MASK
        value ^= 4 * len(context)
        value ^= value >> 16
        value = (value * 0x85EBCA6B) & UINT32_MASK
        value ^= value >> 13
        value = (value * 0xC2B2AE35) & UINT32_MASK
        value ^= value >> 16
        result.append(value)
    return result[0], result[1]


def philox4x32_10(
    counter: tuple[int, int, int, int], key: tuple[int, int]
) -> tuple[int, int, int, int]:
    """Evaluate Random123 Philox4x32 with ten rounds, using uint32 words."""
    c0, c1, c2, c3 = (word & UINT32_MASK for word in counter)
    k0, k1 = (word & UINT32_MASK for word in key)
    for _ in range(10):
        product0, product1 = PHILOX_M0 * c0, PHILOX_M1 * c2
        c0, c1, c2, c3 = (
            ((product1 >> 32) ^ c1 ^ k0) & UINT32_MASK,
            product1 & UINT32_MASK,
            ((product0 >> 32) ^ c3 ^ k1) & UINT32_MASK,
            product0 & UINT32_MASK,
        )
        k0 = (k0 + PHILOX_W0) & UINT32_MASK
        k1 = (k1 + PHILOX_W1) & UINT32_MASK
    return c0, c1, c2, c3


def token_uniform(key: int, context: Sequence[int], token: int, domain: int = 0) -> float:
    """Return the version-1 keyed uniform for one context and token."""
    hash0, hash1 = context_hash(context)
    word = philox4x32_10((int(token), hash0, hash1, domain), (key & UINT32_MASK, key >> 32))[0]
    return (word + 0.5) / (1 << 32)
