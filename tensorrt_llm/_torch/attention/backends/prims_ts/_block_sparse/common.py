# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Dependency-neutral semantic contract for PrimTS block-sparse attention."""

_FINE_KV_BLOCK_SIZES = (8, 16, 32)
_PREPARED_KV_ROUTE_SIZE = 128
_MAX_KV_ATOM_SIZE = 64
_SIGNED_INT32_MAX = (1 << 31) - 1
_BLOCK_SPARSE_Q_TILE_SIZES = (8, 16, 32, 64, 128)
_BLOCK_SPARSE_MAX_HEADS_Q_PER_KV = 32


def _validate_sparse_q_block_size(value: object) -> int:
    """Return a positive semantic Q block size representable by the ABI."""

    requirement = "q_block_size must be a positive Python integer"
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{requirement}, got {type(value).__name__}")
    if value <= 0:
        raise ValueError(requirement)
    if value > _SIGNED_INT32_MAX:
        raise OverflowError("q_block_size must fit in signed int32")
    return value


def _validate_sparse_kv_block_size(value: object) -> int:
    """Return a supported semantic sparse KV block size.

    Fine blocks map directly to KV atoms; coarse blocks must contain an exact
    number of 64-token atoms.
    Rejecting ``bool`` explicitly is necessary because it is a subclass of
    ``int`` in Python.
    """

    requirement = "kv_block_size must be 8, 16, 32, or a positive multiple of 64"
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(
            f"{requirement} expressed as a Python integer, got {type(value).__name__}"
        )
    if value not in _FINE_KV_BLOCK_SIZES and (value <= 0 or value % 64 != 0):
        raise ValueError(requirement)
    if value > _SIGNED_INT32_MAX:
        raise OverflowError("kv_block_size must fit in signed int32")
    return value


def _select_block_sparse_q_tile_size(
    *,
    q_block_size: int,
    heads_q_per_kv: int,
    kv_block_size: int,
) -> int:
    """Select the largest row-pure tile from grouped Q tokens and heads.

    One CTA belongs to one KV head and may group complete Q-head groups across
    several tokens. The selected tile therefore divides evenly into head
    groups, and its token span stays within one semantic sparse Q block.
    """

    q_block_size = _validate_sparse_q_block_size(q_block_size)
    kv_block_size = _validate_sparse_kv_block_size(kv_block_size)
    if isinstance(heads_q_per_kv, bool) or not isinstance(heads_q_per_kv, int):
        raise TypeError("heads_q_per_kv must be a positive integer")
    if heads_q_per_kv <= 0:
        raise ValueError("heads_q_per_kv must be positive")
    if heads_q_per_kv > _BLOCK_SPARSE_MAX_HEADS_Q_PER_KV:
        raise ValueError("block-sparse supports at most 32 Q heads per KV head")
    if heads_q_per_kv & (heads_q_per_kv - 1):
        raise ValueError("Q heads per KV head must be a power of two")

    max_q_tile_size = min(
        q_block_size * heads_q_per_kv,
        32 if kv_block_size < 64 else 128,
    )
    for q_tile_size in reversed(_BLOCK_SPARSE_Q_TILE_SIZES):
        if (
            q_tile_size <= max_q_tile_size
            and q_tile_size % heads_q_per_kv == 0
            and q_block_size % (q_tile_size // heads_q_per_kv) == 0
        ):
            return q_tile_size
    raise ValueError("block-sparse Q geometry has no row-pure Q tile")


def _block_sparse_kv_atom_size(kv_block_size: int) -> int:
    """Return the independently addressable fragment used in a prepared route.

    This is route-metadata granularity, not the TMA load tile. Coarse BSR
    blocks use KV64 fragments so either half can be addressed independently.
    Their primary TensorMap box is still KV128: B128/B256 (and every multiple
    of B128) naturally produce KV128 loads, while B64 and odd coarse block
    sizes use KV128 whenever the two route fragments are physically adjacent.
    """

    return min(
        _validate_sparse_kv_block_size(kv_block_size),
        _MAX_KV_ATOM_SIZE,
    )


def _prepared_kv_routes_are_block_aligned(
    kv_block_size: int,
    kv_route_size: int,
) -> bool:
    """Return whether each prepared route stays within one semantic BSR block."""

    return _validate_sparse_kv_block_size(kv_block_size) % kv_route_size == 0
