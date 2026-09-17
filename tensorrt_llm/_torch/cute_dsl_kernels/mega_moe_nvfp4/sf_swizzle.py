# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Device-side scaling-factor index swizzle for cute SFA/SFB atom layout.

cute's NVFP4 SF atom (32x4x4 swizzle) byte layout per 512-byte atom::

    layout: ((32, 4), (vec_size, 4)) : ((16, 4), (0, 1))   (byte units)
    --> for token ``t`` in the atom (0 <= t < 128), K-bank ``k`` in [0, 4):
        byte_in_atom(t, k) = (t % 32) * 16 + (t // 32) * 4 + k

Each atom holds 128 tokens x 4 K-banks (= 4 fp8 scale factors) = 512 byte.
Across atoms in K direction (`k_atom_idx >= 1`), atoms are placed
contiguously after the previous one (inner K-atom).  Across atoms in M
direction, the next M atom follows all K atoms of the previous M atom::

    atom_idx = row_block_idx * num_k_atoms + k_atom_idx
    atom_byte_start = atom_idx * 512

This file holds the device-side helpers that compute the linear Int32
position of a (token, K-atom) cell inside ``l1_sf_buffer`` such that the
cute mma side (via ``tile_atom_to_shape_SF``) reads back the exact bytes
dispatch wrote.
"""

from cutlass.cutlass_dsl import Int32, dsl_user_op

# Cute SF atom geometry (NVFP4 standard, 32x4x4 swizzle).
SF_ATOM_BLOCK_TOKENS: int = 128
"""Tokens covered by one M-direction SF atom (= 32 inner * 4 outer)."""

SF_ATOM_BYTES: int = 512
"""Bytes per atom (= 128 tokens * 4 K-banks * 1 byte/fp8)."""

SF_ATOM_INT32_PER_ATOM: int = 128
"""Int32 slots per atom (= 512 byte / 4 byte per Int32)."""


@dsl_user_op
def sf_atom_int32_offset(
    token_idx_in_pool_sf_axis,
    k_atom_idx,
    *,
    num_k_atoms: int,
    loc=None,
    ip=None,
):
    """Linear Int32 offset inside ``l1_sf_buffer`` for one (token, K-atom) cell.

    ``token_idx_in_pool_sf_axis`` is the token's M-axis index relative to the
    pool's SF axis start (must already include per-expert padding, i.e.
    ``expert_pool_block_offset * sf_block_m + token_idx_in_expert``).  The
    caller is responsible for keeping the per-expert offset atom-aligned (a
    multiple of ``SF_ATOM_BLOCK_TOKENS``), which falls out naturally when
    ``sf_block_m`` is itself a multiple of ``SF_ATOM_BLOCK_TOKENS``.

    ``k_atom_idx`` is the K-atom index (``0 <= k_atom_idx < num_k_atoms``);
    each K-atom holds one Int32 (= 4 fp8 K-bank SFs) per token.

    ``num_k_atoms`` is the per-token K-atom count (equal to
    ``sf_uint32_per_token`` on the dispatch side); declared as a kwarg-only
    Python ``int`` so it folds to a Constexpr at trace time.

    Returns the Int32-position to use as ``l1_sf_buffer_flat[<offset>]``.
    The Int32 store at that position covers 4 contiguous bytes inside the
    target atom's atom-inner byte layout.
    """
    t = Int32(token_idx_in_pool_sf_axis)
    row_block_idx = t // Int32(SF_ATOM_BLOCK_TOKENS)
    t_in_atom = t % Int32(SF_ATOM_BLOCK_TOKENS)
    # Atom outer (M-row block, K-atom interleaved).
    atom_idx = row_block_idx * Int32(num_k_atoms) + Int32(k_atom_idx)
    # Atom inner Int32 offset: byte (t%32)*16 + (t//32)*4 divided by 4
    #   -> (t%32)*4 + (t//32).  Each Int32 spans 4 consecutive K-bank bytes
    #   inside the atom (= one full Int32 worth of fp8 SF values).
    return (atom_idx * Int32(SF_ATOM_INT32_PER_ATOM) +
            (t_in_atom % Int32(32)) * Int32(4) + (t_in_atom // Int32(32)))
