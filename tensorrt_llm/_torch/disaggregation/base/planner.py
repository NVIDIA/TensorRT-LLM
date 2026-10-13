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
"""The seam between the transfer framework and a model's KV-transfer plan.

The framework (``native/transfer.py``) moves bytes and tracks transfers. What
a model contributes splits in two, each behind its own interface:

* :class:`CacheDescriber` looks at one side only. Given this rank's page table
  and what a request holds, it says which rows (whole pool slots) exist for the
  request and where they are. It knows the model's cache layout, not the peer.
* :class:`TransferPlanner` looks at both sides. Given the rows each side
  described, it decides which bytes of this rank pair with which bytes of the
  peer: pool matching, head re-sharding, who writes.

:func:`plan_transfer` composes the two for one piece of one request. Both
sides of a transfer must run the same describer and the same planner
(``describer_id``, ``planner_id``): the framework enforces that and nothing
else about the plan.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, AbstractSet, List, Optional, Sequence, Tuple

import numpy as np

from .backend import CacheKind

if TYPE_CHECKING:
    from tensorrt_llm._torch.disaggregation.base.backend import Chunk, RecvReqInfo, TokenRange
    from tensorrt_llm._torch.disaggregation.native.rank_info import RankInfo
    from tensorrt_llm._torch.disaggregation.resource.page import KVCachePageTable


class PeerIncompatibleError(ValueError):
    """A peer cannot exchange KV with this rank. Fails the peer, never the worker."""


@dataclass
class PeerTargets:
    """Which peer ranks this rank exchanges with for one peer DP group."""

    ranks: List[int]
    # Every writer contributes an equal share of the receiver's region, so the
    # receiver may split one fan-in bounce reservation ``total // len(ranks)``.
    uniform_writers: bool = False


@dataclass
class TransferPlan:
    """Device addresses to copy, pairwise: ``sizes[i]`` bytes from ``src_ptrs[i]`` to ``dst_ptrs[i]``.

    An empty plan is valid: this rank writes nothing to this peer but still
    completes the handshake (a duplicate-head non-owner, a PP stage with no
    overlapping layers).
    """

    src_ptrs: np.ndarray
    dst_ptrs: np.ndarray
    sizes: np.ndarray

    @classmethod
    def empty(cls) -> "TransferPlan":
        return cls(
            src_ptrs=np.array([], dtype=np.int64),
            dst_ptrs=np.array([], dtype=np.int64),
            sizes=np.array([], dtype=np.int64),
        )


@dataclass(frozen=True)
class LayerGroupRows:
    """The rows one rank holds for a request in one pool view of one layer group.

    A row is a whole pool slot. Row ``i`` holds block ordinal ``block_ordinals[i]``
    (the block's position in the prompt) and starts at device address
    ``addresses[i]``; every row is ``row_bytes`` long. Ordinals this side does not
    hold are simply absent. ``span_blocks`` is how many ordinals the request
    covers on this side, so two sides can be checked to describe the same prompt.

    A STATE (recurrent) group holds one checkpoint for the request: one row,
    ordinal 0, at the slot's address. How a slot splits into layers and heads is
    the page table's business, which the planner's mappers read.
    """

    layer_group_index: int
    view_index: int
    kind: CacheKind
    block_ordinals: np.ndarray  # int64, strictly increasing
    addresses: np.ndarray  # int64, one per row
    row_bytes: int
    span_blocks: int

    def __len__(self) -> int:
        return int(self.block_ordinals.size)


class CacheDescriber(ABC):
    """How one rank's cache holds a request. One instance per rank; knows no peer.

    This is the single-sided half of planning: it reads the model's page table
    and a request's positional slot tables and produces :class:`LayerGroupRows`.
    The sender runs it twice per piece, over its own table and, from the
    receiver's wire message, over the receiver's.
    """

    #: Both sides must agree. The framework compares it before any registration.
    describer_id: str = ""

    @abstractmethod
    def rows(
        self,
        page_table: "KVCachePageTable",
        slot_table_per_layer_group: Sequence[np.ndarray],
        excluded_views: AbstractSet[Tuple[int, int]],
    ) -> List[LayerGroupRows]:
        """One entry per pool view the request may transfer, in page-table order.

        ``slot_table_per_layer_group[g]`` is positional: entry ``i`` is the pool slot
        holding block ordinal ``i``, or -1 where this side holds nothing (see
        ``Chunk``). Views named in ``excluded_views`` as ``(layer_group, view)`` get
        no entry at all.
        """


class TransferPlanner(ABC):
    """Model-owned KV transfer planning. One instance per rank, shared by Sender and Receiver.

    Threading is the framework's: ``pair`` runs where the sender builds write
    metadata; ``targets``, ``receiver_payload_bytes`` and ``check_peer`` run on
    the receiver's dispatch path and on both listeners. Keep per-peer caches
    keyed by ``(peer_ri.instance_name, peer_ri.instance_rank)`` and drop them
    in ``forget_peer``.
    """

    #: Both sides must agree. The framework compares it before ``check_peer``.
    planner_id: str = ""

    def describe(self) -> bytes:
        """Opaque self-description carried in ``RankInfo.planner_blob``; only this class decodes it."""
        return b""

    @abstractmethod
    def check_peer(self, self_ri: "RankInfo", peer_ri: "RankInfo") -> None:
        """Raise :class:`PeerIncompatibleError` with a field-level reason, else return."""

    @abstractmethod
    def targets(self, self_ri: "RankInfo", peer_ri: "RankInfo", peer_dp_rank: int) -> PeerTargets:
        """Peer flat ranks this rank exchanges with for the peer's DP group ``peer_dp_rank``."""

    @abstractmethod
    def pair(
        self,
        self_ri: "RankInfo",
        peer_ri: "RankInfo",
        self_rows: Sequence[LayerGroupRows],
        peer_rows: Sequence[LayerGroupRows],
        *,
        token_range: "TokenRange",
        prompt_len: int,
        dst_start_token: Optional[int],
    ) -> TransferPlan:
        """Sender side: the bytes this rank writes into ``peer_ri`` for one piece.

        ``self_rows`` is what this rank holds for the piece, ``peer_rows`` what the
        receiver holds, both as the describer produced them. ``token_range`` is the
        piece within the prompt of ``prompt_len`` tokens; ``dst_start_token`` is a
        lower bound below which the receiver accepts nothing, on top of the rows
        it already left out.
        """

    @abstractmethod
    def receiver_payload_bytes(
        self, self_ri: "RankInfo", peer_ri: "RankInfo", req_info: "RecvReqInfo"
    ) -> Optional[int]:
        """Receiver side: bytes beyond the paged blocks that ride the same write.

        ``None`` means the planner cannot size them; the receiver then disables
        its bounce reservation for this request.
        """

    def forget_peer(self, peer_ri: "RankInfo") -> None:
        """Drop any per-peer cache. Called by ``PeerRegistrar.unregister``."""
        return None


def plan_transfer(
    describer: CacheDescriber,
    planner: TransferPlanner,
    self_ri: "RankInfo",
    peer_ri: "RankInfo",
    chunk: "Chunk",
    req_info: "RecvReqInfo",
    prompt_len: int,
) -> TransferPlan:
    """The sender's plan for one piece: describe both sides, then pair them.

    The receiver's rows are rebuilt here from the slot tables in its wire
    message and its published page table.
    """
    self_rows = describer.rows(
        self_ri.page_table, chunk.block_ids_per_layer_groups, chunk.excluded_pool_views
    )
    peer_rows = describer.rows(peer_ri.page_table, req_info.block_ids_per_layer_groups, frozenset())
    return planner.pair(
        self_ri,
        peer_ri,
        self_rows,
        peer_rows,
        token_range=chunk.token_range,
        prompt_len=prompt_len,
        dst_start_token=req_info.dst_start_token,
    )
