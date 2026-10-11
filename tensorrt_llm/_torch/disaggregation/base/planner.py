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

The framework (``native/transfer.py``) moves bytes and tracks transfers. Which
bytes of this rank pair with which bytes of a peer rank is a property of the
model's cache layout and the two parallel topologies, so a model class may
supply its own :class:`TransferPlanner`; ``native/default_planner.py`` is the
one every model gets unless it overrides ``build_transfer_planner``.

Both sides of a transfer must run the same planner class (``planner_id``):
the framework enforces that and nothing else about the plan.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional

import numpy as np

if TYPE_CHECKING:
    from tensorrt_llm._torch.disaggregation.base.backend import Chunk, RecvReqInfo
    from tensorrt_llm._torch.disaggregation.native.rank_info import RankInfo


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


class TransferPlanner(ABC):
    """Model-owned KV transfer planning. One instance per rank, shared by Sender and Receiver.

    Threading is the framework's: ``plan`` runs where the sender builds write
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
    def plan(
        self,
        self_ri: "RankInfo",
        peer_ri: "RankInfo",
        chunk: "Chunk",
        req_info: "RecvReqInfo",
        prompt_len: int,
    ) -> TransferPlan:
        """Sender side: the bytes this rank writes into ``peer_ri`` for one chunk.

        ``chunk`` is this rank's description of what it holds for the piece (block
        tables per layer group, token range), ``req_info`` the receiver's description
        of where it lands (its block tables, a lower bound below which it accepts
        nothing). Both arrive whole so a planner honors every per-request fact the
        framework records, present or future, without a signature change.
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
