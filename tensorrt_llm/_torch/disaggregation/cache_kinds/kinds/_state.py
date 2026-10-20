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
"""Rules shared by state units: one slot per request, holding every layer's state.

The slot holds the state at the end of the prompt, so a transfer moves it once, after the last
chunk. Roles declare how a layer's state reshards over the state grid (TP, TP x CP under helix, 1
under attention DP): sectioned (convolution state, each section sharded by channel), sharded by
head (SSM state), or replicated (side state, identical on every rank).
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Set, Tuple

import numpy as np

from tensorrt_llm._torch.disaggregation.base import CacheKind

from ..page_table import BufferDecl, PageTable, RegionKey
from ..spec import MatchedPair, PairContext, RegistrationRegion
from ..topology import STATE_TP, shard_grid

EMPTY = np.array([], dtype=np.int64)


def sectioned(section_bytes, nbytes: int, context: str) -> Tuple[BufferDecl, ...]:
    """Concatenated sections (convolution state ``[x | B | C]``), each sharded by byte on its own."""
    if sum(section_bytes) != nbytes:
        raise ValueError(
            f"{context}: section_bytes {list(section_bytes)} do not sum to the role's "
            f"{nbytes} bytes"
        )
    return tuple(
        BufferDecl(nbytes=int(s), domain=STATE_TP, axes=(("byte", int(s)),), split_axis="byte")
        for s in section_bytes
    )


def by_head(nbytes: int, bytes_per_head: int, context: str) -> Tuple[BufferDecl, ...]:
    """State sharded by head (SSM state ``[head, head_dim, d_state]``)."""
    if nbytes % bytes_per_head != 0:
        raise ValueError(
            f"{context}: {nbytes} bytes are not a multiple of bytes_per_head={bytes_per_head}"
        )
    heads = nbytes // bytes_per_head
    return (BufferDecl(nbytes=nbytes, domain=STATE_TP, axes=(("head", heads),), split_axis="head"),)


def whole(nbytes: int) -> Tuple[BufferDecl, ...]:
    """State every rank of the state grid holds in full: replicated side state, or a sharded role
    with no declared geometry, which a TP mismatch turns into a size mismatch instead of misplaced
    bytes."""
    return (BufferDecl(nbytes=nbytes, domain=STATE_TP),)


def units(slot: Optional[int]) -> np.ndarray:
    return EMPTY if slot is None else np.array([slot], dtype=np.int64)


def pair_units(
    src_units: np.ndarray, dst_units: np.ndarray, ctx: PairContext
) -> Tuple[np.ndarray, np.ndarray]:
    if src_units.size == 0 or dst_units.size == 0:
        return EMPTY, EMPTY
    return (
        np.asarray(src_units[:1], dtype=np.int64),
        np.asarray(dst_units[:1], dtype=np.int64),
    )


def transfer_bytes(region_bytes: int, units: np.ndarray) -> int:
    # One slot per request: each region moves once, whatever the unit count.
    return region_bytes if bool((units >= 0).any()) else 0


def registration_regions(page_table: PageTable, space_index: int) -> List[RegistrationRegion]:
    # Roles may share one allocation (interleaved, equal sizes) or be separate. Lowest base first,
    # so a covering region registers before roles that fall inside it.
    pools = sorted(
        (page_table.pools[p] for p in page_table.spaces[space_index].pools),
        key=lambda p: p.base_address,
    )
    return [
        RegistrationRegion(p.base_address, p.num_slots * p.slot_stride_bytes, True) for p in pools
    ]


def _keys(page_table: PageTable, kind: str) -> Set[RegionKey]:
    return {
        region.key
        for space in page_table.spaces
        if space.kind == kind and space.unit == CacheKind.STATE
        for region in space.regions
    }


def receiver_extra_bytes(
    sender_page_table: PageTable, receiver_page_table: PageTable, space_index: int, dst_units
) -> int:
    """Bytes landing in the receiver's slot: its own region bytes over regions the sender holds.

    Receiver-local, whatever the sender-side shard pairing, so it needs no simulation of which
    sender ranks write (RankInfoServer only describes rank 0).
    """
    if dst_units.size == 0:
        return 0
    space = receiver_page_table.spaces[space_index]
    sender_keys = _keys(sender_page_table, space.kind)
    return sum(region.nbytes for region in space.regions if region.key in sender_keys)


def validate_peer(kind: str, self_ri, peer_ri, pairs: Sequence[MatchedPair]) -> None:
    """Reject a peer whose recurrent layout cannot be resharded into ours.

    Analogue of the C++ ``rnnCacheFormatter inquireSupport`` gate. The core invariant is the
    *global* (grid-aggregated) state size: for a sharded buffer ``per_rank_bytes * grid`` is
    TP-invariant, so it must match even when the two sides' TP differ. A model that keeps a
    replicated full-size state while sharding over a grid > 1 violates it under heterogeneous TP --
    exactly where resharding would read past the slot -- and is rejected here.
    """
    for space, region, _, peer_region in pairs:
        if space.unit != CacheKind.STATE:
            continue
        role = f"role {region.role!r} (layer {region.layer})"
        for i, (a, b) in enumerate(zip(region.buffers, peer_region.buffers)):
            if a.split_axis is None:
                if a.nbytes != b.nbytes:
                    raise ValueError(
                        f"{kind}: replicated state size differs for {role} buffer {i} "
                        f"(local={a.nbytes}, peer={b.nbytes})"
                    )
                continue
            grid = shard_grid(a.domain, self_ri).size
            peer_grid = shard_grid(b.domain, peer_ri).size
            if a.nbytes * grid != b.nbytes * peer_grid:
                raise ValueError(
                    f"{kind}: global (TP-aggregated) size of {role} buffer {i} differs: local "
                    f"{a.nbytes} bytes/rank x state grid {grid} vs peer {b.nbytes} bytes/rank x "
                    f"state grid {peer_grid}. Per-rank state sizes are inconsistent with a sharded "
                    "layout across the two sides; either the state shape/dtype differs, or the "
                    "model keeps a replicated (non-sharded) per-rank recurrent state while "
                    "sharding over a grid > 1, which supports heterogeneous ctx/gen TP only with "
                    "attention-DP enabled on both sides."
                )
            if a.split_axis == "head" and a.unit_bytes(role) != b.unit_bytes(role):
                # TP-invariant head_dim * d_state * element_size: a different state shape or dtype.
                raise ValueError(
                    f"{kind}: bytes per head of {role} differ (local={a.unit_bytes(role)}, "
                    f"peer={b.unit_bytes(role)}); check head_dim / d_state / mamba_ssm_cache_dtype"
                )
