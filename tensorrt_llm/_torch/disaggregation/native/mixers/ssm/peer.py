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

from typing import Optional, Tuple

from tensorrt_llm._torch.disaggregation.native.rank_info import RankInfo
from tensorrt_llm._torch.disaggregation.resource.page import (
    CacheKind,
    KVCachePageTable,
    MambaLayerGroup,
    MapperKind,
)
from tensorrt_llm._torch.disaggregation.resource.utils import (
    find_replicated_role_mismatch,
    get_pool_view_global_layer_ids,
)


class MambaPolicy:
    """Recurrent-state compatibility checks between peers. Which bytes move, and which rank sends
    them, follows from the page layouts (``layout_mapping``)."""

    def __init__(self, self_rank_info: RankInfo):
        self._ri = self_rank_info

    @staticmethod
    def _find_mamba_layer_group(
        page_table: Optional[KVCachePageTable],
    ) -> Optional[MambaLayerGroup]:
        if page_table is None:
            return None
        return next(
            (lg for lg in page_table.layer_groups if lg.kind == CacheKind.STATE),
            None,
        )

    @staticmethod
    def validate_peer_compatible(
        self_ri: RankInfo,
        peer_ri: RankInfo,
        self_page_table: Optional[KVCachePageTable],
        peer_page_table: Optional[KVCachePageTable],
    ) -> None:
        """Validate recurrent-state (Mamba/KDA) layout compatibility with a peer.

        Analogue of the C++ ``rnnCacheFormatter inquireSupport`` gate for the
        V2 path: reject at peer-registration time instead of corrupting
        memory at transfer time. Raises ``ValueError`` naming the mismatched
        field.

        The core invariant checked is *global* (TP-aggregated) state size:
        for a TP-sharded state, ``per_rank_bytes * mamba_tp`` is
        TP-invariant, so it must match between peers even when their TP
        sizes differ. Kimi K3 KDA satisfies it: with attention-DP off the
        cache manager head-shards the state across tp_size (matching the
        model's head-sharded KDA compute), so heterogeneous ctx/gen TP
        passes; under attention-DP the state is replicated and ``_mamba_tp``
        is 1 on that side. A model that kept a replicated full-size per-rank
        state while reporting ``mamba_tp > 1`` would violate the invariant
        under heterogeneous TP — exactly the configuration where the
        TP-mismatch mappers would compute shard offsets past the end of the
        slot, silently corrupting the state — and is rejected here.
        """
        self_mlg = MambaPolicy._find_mamba_layer_group(self_page_table)
        peer_mlg = MambaPolicy._find_mamba_layer_group(peer_page_table)
        if self_mlg is None or peer_mlg is None:
            # Under pipeline parallelism each rank publishes only its own
            # stage's layers, so a hybrid model can pair a rank holding
            # recurrent layers with a peer stage holding none. The transfer
            # path intersects the two layer sets and
            # moves nothing when either side has no recurrent layers, so
            # there is nothing to validate for this pair.
            return

        # Layer sets are NOT required to match: with pipeline parallelism the
        # two sides may partition layers differently, and the transfer covers
        # exactly the intersection (empty intersection moves no state). The
        # invariants below are per-layer-slot quantities, uniform across a
        # model's recurrent layers, so they apply regardless of which layers
        # overlap.
        self_tp, _ = MambaPolicy._mamba_tp(self_ri)
        peer_tp, _ = MambaPolicy._mamba_tp(peer_ri)

        def _check_global(field: str, self_bytes: int, peer_bytes: int) -> None:
            if self_bytes * self_tp != peer_bytes * peer_tp:
                raise ValueError(
                    f"MambaPolicy.validate_peer_compatible: global (TP-aggregated) {field} "
                    f"differs: local {self_bytes} bytes/rank x mamba_tp={self_tp} vs "
                    f"peer {peer_bytes} bytes/rank x mamba_tp={peer_tp}. Per-rank state "
                    "sizes are inconsistent with a TP-sharded layout across the two "
                    "sides; either the state shape/dtype differs, or the model keeps a "
                    "replicated (non-TP-sharded) per-rank recurrent state while "
                    "reporting mamba_tp > 1, which supports heterogeneous ctx/gen TP "
                    "only with attention-DP enabled on both sides."
                )

        # Views pair up by pool_role, exactly as get_pool_mapping matches them.
        # A role present on only one side is a peer-declaration mismatch for
        # REPLICATED views (checked below); for sharded views it is left to
        # pool matching, which skips it.
        peer_views = {pv.pool_role: pv for pv in peer_mlg.pool_views}
        for self_pv in self_mlg.pool_views:
            peer_pv = peer_views.get(self_pv.pool_role)
            if peer_pv is None:
                continue
            role = sorted(self_pv.pool_role)
            if self_pv.mapper_kind != peer_pv.mapper_kind:
                raise ValueError(
                    f"MambaPolicy.validate_peer_compatible: mapper kind of role {role} differs "
                    f"(local={self_pv.mapper_kind.name}, peer={peer_pv.mapper_kind.name})"
                )
            if self_pv.bytes_per_layer is None or peer_pv.bytes_per_layer is None:
                continue

            if self_pv.mapper_kind == MapperKind.REPLICATED:
                # Identical bytes on every rank: sizes must match exactly.
                if self_pv.bytes_per_layer != peer_pv.bytes_per_layer:
                    raise ValueError(
                        "MambaPolicy.validate_peer_compatible: replicated state size differs "
                        f"for role {role} (local={self_pv.bytes_per_layer}, "
                        f"peer={peer_pv.bytes_per_layer})"
                    )
                continue

            _check_global(f"{role} slot_bytes", self_pv.bytes_per_layer, peer_pv.bytes_per_layer)

            if self_pv.mapper_kind == MapperKind.SECTIONED:
                if self_pv.section_bytes is not None and peer_pv.section_bytes is not None:
                    if len(self_pv.section_bytes) != len(peer_pv.section_bytes):
                        raise ValueError(
                            f"MambaPolicy.validate_peer_compatible: section count of role {role} "
                            f"differs (local={len(self_pv.section_bytes)}, "
                            f"peer={len(peer_pv.section_bytes)})"
                        )
                    for i, (s, p) in enumerate(zip(self_pv.section_bytes, peer_pv.section_bytes)):
                        _check_global(f"{role} section_bytes[{i}]", s, p)
            elif (
                self_pv.bytes_per_head is not None
                and peer_pv.bytes_per_head is not None
                and self_pv.bytes_per_head != peer_pv.bytes_per_head
            ):
                # TP-invariant: head_dim * d_state * element_size. A mismatch
                # means different state shape or SSM cache dtype.
                raise ValueError(
                    f"MambaPolicy.validate_peer_compatible: bytes_per_head of role {role} "
                    f"differs (local={self_pv.bytes_per_head}, peer={peer_pv.bytes_per_head}); "
                    "check head_dim / d_state / mamba_ssm_cache_dtype"
                )

        # Replicated side state (e.g. PLE) is copied whole per layer, so both
        # sides must declare the same roles on shared layers. A role missing
        # on one side would otherwise be dropped silently by pool matching.
        differing = find_replicated_role_mismatch(self_page_table, peer_page_table, CacheKind.STATE)
        if differing:
            raise ValueError(
                "MambaPolicy.validate_peer_compatible: replicated roles differ on "
                f"overlapping layers: {differing}"
            )

    @staticmethod
    def _mamba_tp(ri: RankInfo) -> Tuple[int, int]:
        """Return (mamba_effective_tp_size, mamba_effective_tp_rank).

        When attention_dp is enabled, mamba is not TP-sharded.
        """
        if ri.attention and ri.attention.enable_attention_dp:
            return 1, 0
        if ri.cp_size > 1:
            # Helix repurposes CP ranks as TP for the mamba layers, so the
            # shard grid is tp*cp with CP-minor flat rank (matches Mapping).
            return ri.tp_size * ri.cp_size, ri.tp_rank * ri.cp_size + ri.cp_rank
        return ri.tp_size, ri.tp_rank


def mamba_receiver_payload_bytes(
    sender_page_table: KVCachePageTable,
    receiver_page_table: KVCachePageTable,
    dst_slot: Optional[int],
) -> int:
    """Recurrent-state bytes that will land in the receiver's slot.

    Receiver-local invariant: regardless of the sender-side shard pairing,
    the slot receives exactly the receiver's own per-layer slot bytes over
    the overlapping mamba layers.  This avoids the RankInfoServer rank-0
    limitation that makes sender-side simulation return 0 for non-rank-0
    receivers.
    """
    if dst_slot is None:
        return 0
    sender_mlg = MambaPolicy._find_mamba_layer_group(sender_page_table)
    receiver_mlg = MambaPolicy._find_mamba_layer_group(receiver_page_table)
    if sender_mlg is None or receiver_mlg is None:
        return 0

    sender_views = {
        (pool_view.pool_role, pool_view.mapper_kind): pool_view
        for pool_view in sender_mlg.pool_views
    }
    total = 0
    for receiver_view in receiver_mlg.pool_views:
        sender_view = sender_views.get((receiver_view.pool_role, receiver_view.mapper_kind))
        if sender_view is None:
            continue
        sender_layers = set(get_pool_view_global_layer_ids(sender_view, sender_mlg))
        receiver_layers = set(get_pool_view_global_layer_ids(receiver_view, receiver_mlg))
        overlap = sender_layers & receiver_layers
        if not overlap:
            continue
        total += len(overlap) * receiver_view.bytes_per_layer
    return total
