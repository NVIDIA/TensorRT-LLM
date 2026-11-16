# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Pseudo-random function shared by the watermark sampler and the detector.

Generation and detection must derive bit-identical random numbers, so the PRF is
spelled out here instead of delegating to a library generator:

* ``context_seed`` hashes the secret key and the preceding ``context_width``
  tokens into a 64-bit seed (a splitmix64-style mixer).
* ``philox_uniform`` turns ``(seed, token_id)`` into a uniform number in
  ``(0, 1)`` with Philox4x32-10 and takes the first output word of the block
  whose counter is the token id.

The numpy functions are the reference the detector uses. The torch functions
produce the same values on any device and feed the sampling kernel's seeds.
"""

from typing import Sequence

import numpy as np
import torch

__all__ = [
    "PAD_TOKEN",
    "secondary_key",
    "context_seed",
    "context_seeds",
    "context_seeds_torch",
    "philox_uniform",
    "philox_uniform_torch",
]

# Context position that precedes the start of the prompt (prompt shorter than the context).
PAD_TOKEN = 0xFFFFFFFF

_MASK32 = 0xFFFFFFFF
_MASK64 = (1 << 64) - 1

_PHILOX_M0 = 0xD2511F53
_PHILOX_M1 = 0xCD9E8D57
_PHILOX_W0 = 0x9E3779B9
_PHILOX_W1 = 0xBB67AE85
_PHILOX_ROUNDS = 10

_GOLDEN = 0x9E3779B97F4A7C15
_MIX_C1 = 0xBF58476D1CE4E5B9
_MIX_C2 = 0x94D049BB133111EB


_SECONDARY_SALT = 0xD1B54A32D192ED03


def secondary_key(key: int) -> int:
    """The second key of dual-key watermarking, derived from the master ``key``.

    Independent of ``key`` for all practical purposes (a mixed copy of it), so the two keys
    drive independent random streams for the same context.
    """
    with np.errstate(over="ignore"):
        return int(_mix64(np.array([(key & _MASK64) ^ _SECONDARY_SALT], dtype=np.uint64))[0])


def _signed64(value: int) -> int:
    """``value`` (in [0, 2**64)) as the int64 with the same bit pattern."""
    return value - (1 << 64) if value >= (1 << 63) else value


# ---------------------------------------------------------------------------
# numpy reference
# ---------------------------------------------------------------------------


def _mix64(x: np.ndarray) -> np.ndarray:
    x = x ^ (x >> np.uint64(30))
    x = x * np.uint64(_MIX_C1)
    x = x ^ (x >> np.uint64(27))
    x = x * np.uint64(_MIX_C2)
    return x ^ (x >> np.uint64(31))


def context_seeds(key: int, contexts: np.ndarray) -> np.ndarray:
    """Seeds for a batch of contexts.

    Args:
        key: secret key in [0, 2**64).
        contexts: ``[N, width]`` token ids; ``PAD_TOKEN`` marks positions before the prompt.

    Returns:
        ``[N]`` uint64 seeds.
    """
    contexts = np.asarray(contexts)
    seeds = np.full(contexts.shape[0], key & _MASK64, dtype=np.uint64)
    with np.errstate(over="ignore"):
        for column in range(contexts.shape[1]):
            token = (contexts[:, column].astype(np.int64) & _MASK32).astype(np.uint64)
            seeds = _mix64((seeds ^ token) + np.uint64(_GOLDEN))
    return seeds


def context_seed(key: int, context: Sequence[int]) -> int:
    """Seed of one context (a sequence of ``context_width`` token ids)."""
    return int(context_seeds(key, np.asarray(context, dtype=np.int64)[None, :])[0])


def philox_uniform(seed: "int | np.ndarray", token_ids: np.ndarray) -> np.ndarray:
    """Uniform numbers in (0, 1) for ``token_ids`` under ``seed``, as float32.

    ``seed`` is one integer for all tokens, or a uint64 array matching ``token_ids``.

    The value is ``(2 * (w >> 9) + 1) / 2**24`` for the first Philox4x32-10 output word ``w``,
    so it is exactly representable in float32 and never 0 or 1.
    """
    c0 = np.asarray(token_ids).astype(np.uint64) & np.uint64(_MASK32)
    c1 = np.zeros_like(c0)
    c2 = np.zeros_like(c0)
    c3 = np.zeros_like(c0)
    m32 = np.uint64(_MASK32)
    seed = np.asarray(seed, dtype=np.uint64)
    k0 = seed & m32
    k1 = (seed >> np.uint64(32)) & m32
    for round_idx in range(_PHILOX_ROUNDS):
        if round_idx:
            k0 = (k0 + np.uint64(_PHILOX_W0)) & m32
            k1 = (k1 + np.uint64(_PHILOX_W1)) & m32
        p0 = c0 * np.uint64(_PHILOX_M0)
        p1 = c2 * np.uint64(_PHILOX_M1)
        hi0, lo0 = p0 >> np.uint64(32), p0 & m32
        hi1, lo1 = p1 >> np.uint64(32), p1 & m32
        c0, c1, c2, c3 = hi1 ^ c1 ^ k0, lo1, hi0 ^ c3 ^ k1, lo0
    word = c0
    return ((2 * (word >> np.uint64(9)) + 1).astype(np.float64) / float(1 << 24)).astype(np.float32)


# ---------------------------------------------------------------------------
# torch (any device); int64 arithmetic wraps, so the 64-bit mixer matches numpy
# ---------------------------------------------------------------------------


def _lsr64(x: torch.Tensor, shift: int) -> torch.Tensor:
    """Logical right shift of an int64 tensor."""
    return (x >> shift) & ((1 << (64 - shift)) - 1)


def _mix64_torch(x: torch.Tensor) -> torch.Tensor:
    x = x ^ _lsr64(x, 30)
    x = x * _signed64(_MIX_C1)
    x = x ^ _lsr64(x, 27)
    x = x * _signed64(_MIX_C2)
    return x ^ _lsr64(x, 31)


def context_seeds_torch(key: int, contexts: torch.Tensor) -> torch.Tensor:
    """``context_seeds`` on a ``[N, width]`` integer tensor.

    Returns ``[N]`` int64 seeds holding the same bit patterns as the numpy uint64 result.
    """
    contexts = contexts.to(torch.int64)
    seeds = torch.full(
        (contexts.shape[0],), _signed64(key & _MASK64), dtype=torch.int64, device=contexts.device
    )
    for column in range(contexts.shape[1]):
        token = contexts[:, column] & _MASK32
        seeds = _mix64_torch((seeds ^ token) + _signed64(_GOLDEN))
    return seeds


def philox_uniform_torch(seeds: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
    """``philox_uniform`` for per-row seeds, on any device.

    ``seeds`` is ``[R]`` int64 and ``token_ids`` is ``[V]`` or ``[R, V]``; returns float32 of
    shape ``[R, V]``. Intended as a test oracle and for small inputs; the sampling kernel is
    the production path.
    """
    seeds = seeds.to(torch.int64).reshape(-1, 1)
    token_ids = token_ids.to(torch.int64)
    if token_ids.dim() == 1:
        token_ids = token_ids.unsqueeze(0).expand(seeds.shape[0], -1)
    c0 = token_ids & _MASK32
    c1 = torch.zeros_like(c0)
    c2 = torch.zeros_like(c0)
    c3 = torch.zeros_like(c0)
    k0 = seeds & _MASK32
    k1 = _lsr64(seeds, 32) & _MASK32
    for round_idx in range(_PHILOX_ROUNDS):
        if round_idx:
            k0 = (k0 + _PHILOX_W0) & _MASK32
            k1 = (k1 + _PHILOX_W1) & _MASK32
        p0 = c0 * _PHILOX_M0
        p1 = c2 * _PHILOX_M1
        hi0, lo0 = _lsr64(p0, 32), p0 & _MASK32
        hi1, lo1 = _lsr64(p1, 32), p1 & _MASK32
        c0, c1, c2, c3 = hi1 ^ c1 ^ k0, lo1, hi0 ^ c3 ^ k1, lo0
    word = c0
    return ((2 * (word >> 9) + 1).to(torch.float64) / float(1 << 24)).to(torch.float32)
