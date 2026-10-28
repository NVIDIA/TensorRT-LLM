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

from typing import Optional

from tensorrt_llm import logger
from tensorrt_llm._torch.disaggregation.native.rank_info import RankInfo
from tensorrt_llm._torch.disaggregation.resource.page import CacheKind, KVCachePageTable, MapperKind
from tensorrt_llm._torch.disaggregation.resource.utils import find_replicated_role_mismatch


class AttentionPolicy:
    def __init__(self, self_rank_info: RankInfo):
        self._ri = self_rank_info

    def _tp_per_dp(self, ri: RankInfo) -> int:
        if getattr(ri.attention, "enable_attention_dp", False):
            return ri.tp_size // ri.dp_size
        return ri.tp_size

    def _fail_if(self, cond: bool, reason: str, **kv) -> bool:
        if not cond:
            return False
        details = ", ".join(f"{k}={v!r}" for k, v in kv.items())
        msg = f"AttentionPolicy: incompatible: {reason}" + (f"; {details}" if details else "")
        logger.warning("%s", msg)
        return True

    def _mismatch(self, field: str, local, peer) -> bool:
        return self._fail_if(
            local != peer, f"{field} mismatch", field=field, local=local, peer=peer
        )

    @staticmethod
    def _uses_exact_tpb_mapper(ri: RankInfo) -> bool:
        """NHD / replicated pools address bytes inside a block, so their
        geometry only lines up when both sides use the same tokens_per_block."""
        if ri.page_table is None:
            return False
        return any(
            pool_view.mapper_kind in (MapperKind.NHD, MapperKind.REPLICATED)
            for layer_group in ri.page_table.layer_groups
            for pool_view in getattr(layer_group, "pool_views", ())
        )

    def _tpb_check(self, local: int, peer: int, peer_ri: RankInfo) -> bool:
        if local == peer:
            return False
        if self._uses_exact_tpb_mapper(self._ri) or self._uses_exact_tpb_mapper(peer_ri):
            logger.warning(
                "AttentionPolicy: incompatible: tokens_per_block mismatch for "
                "NHD/replicated pools; local=%d peer=%d",
                local,
                peer,
            )
            return True
        larger, smaller = max(local, peer), min(local, peer)
        if larger % smaller != 0:
            logger.warning(
                "AttentionPolicy: incompatible: tokens_per_block not divisible; local=%d peer=%d",
                local,
                peer,
            )
            return True
        logger.warning(
            "AttentionPolicy: tokens_per_block mismatch (local=%d, peer=%d); "
            "KV transfer proceeds — ensure block boundaries align during transfer.",
            local,
            peer,
        )
        return False

    @staticmethod
    def validate_peer_compatible(
        self_page_table: Optional[KVCachePageTable],
        peer_page_table: Optional[KVCachePageTable],
    ) -> None:
        """Reject a peer whose PAGED groups declare different replicated roles.

        Pool matching drops a view with no counterpart silently, so an
        index-key side cache the peer never declares would leave the receiver
        holding zeroed state instead of raising. Raises ``ValueError``.
        """
        differing = find_replicated_role_mismatch(self_page_table, peer_page_table, CacheKind.PAGED)
        if differing:
            raise ValueError(
                "AttentionPolicy.validate_peer_compatible: replicated roles differ on "
                f"overlapping layers: {differing}"
            )

    def check_peer_compatible(self, peer_ri: RankInfo) -> bool:
        a = self._ri.attention
        b = peer_ri.attention

        return not (
            self._mismatch("is_mla", a.is_mla, b.is_mla)
            or self._fail_if(
                self._ri.cp_size != 1 and peer_ri.cp_size != 1,
                "cp_size must be 1 on at least one side (helix pairs a cp=1 "
                "context instance with a cp=N generation instance)",
                local=self._ri.cp_size,
                peer=peer_ri.cp_size,
            )
            or self._fail_if(
                (self._ri.cp_size != 1 or peer_ri.cp_size != 1)
                and a.tokens_per_block != b.tokens_per_block,
                "helix block-interleaved transfer requires equal "
                "tokens_per_block on both sides (block boundaries must "
                "coincide for [cp_rank::cp_size] ownership)",
                local=a.tokens_per_block,
                peer=b.tokens_per_block,
            )
            or self._mismatch("element_bytes", a.element_bytes, b.element_bytes)
            or self._fail_if(
                not self.head_match(peer_ri)[0]
                and not float(a.tokens_per_block * a.dims_per_head * a.element_bytes).is_integer(),
                "sub-byte head slicing is not byte-aligned",
                tokens_per_block=a.tokens_per_block,
                dims_per_head=a.dims_per_head,
                element_bytes=a.element_bytes,
            )
            or self._tpb_check(a.tokens_per_block, b.tokens_per_block, peer_ri)
            or self._mismatch("dims_per_head", a.dims_per_head, b.dims_per_head)
            or self._fail_if(
                a.is_mla and (a.kv_heads_per_rank != 1 or b.kv_heads_per_rank != 1),
                "MLA requires kv_heads_per_rank == 1 for both ranks",
                local=a.kv_heads_per_rank,
                peer=b.kv_heads_per_rank,
            )
        )

    def _head_factors(self, peer_ri: RankInfo) -> tuple[int, int]:
        self_tp = self._tp_per_dp(self._ri)
        peer_tp = self._tp_per_dp(peer_ri)
        a = self._ri.attention
        b = peer_ri.attention
        return a.kv_heads_per_rank * self_tp, b.kv_heads_per_rank * peer_tp

    def head_match(self, peer_ri: RankInfo) -> tuple[bool, bool]:
        factor_self, factor_peer = self._head_factors(peer_ri)
        is_dup_head = factor_self != factor_peer
        # Slot layout is compatible if kv_heads_per_rank matches (same slot size).
        head_match = (
            self._ri.attention.is_mla
            or self._ri.attention.kv_heads_per_rank == peer_ri.attention.kv_heads_per_rank
        )
        return head_match, is_dup_head

    def duplicate_head_factors(self, peer_ri: RankInfo) -> tuple[int, int]:
        # Head duplication only applies to attention KV.  In particular, do
        # not divide 0 / 0 for an attention-free hybrid PP stage; MambaPolicy
        # computes its TP mapping separately.
        if self._ri.attention.kv_heads_per_rank == 0 or peer_ri.attention.kv_heads_per_rank == 0:
            return 1, 1
        factor_self, factor_peer = self._head_factors(peer_ri)
        dup_head = max(1, factor_self // factor_peer)
        peer_dup_head = max(1, factor_peer // factor_self)
        return dup_head, peer_dup_head
