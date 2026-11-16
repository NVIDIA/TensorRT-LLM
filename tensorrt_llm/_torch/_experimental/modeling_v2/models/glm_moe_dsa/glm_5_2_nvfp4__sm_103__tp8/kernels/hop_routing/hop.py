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
"""Routing hop of hop_routing: the into-hop entry plus native top-8 routing.

``HopRoutingTransport`` wraps the ``attention_output_into_hop`` :class:`IntoHopTransport` of a
``post_attention_hop``. The producer side is the into-hop entry's unchanged (the attention tail
publishes into the same transport); the hop runs ``native_ops.post_attention_hop_routing``,
which returns the into-hop entry's normalized hidden, residual and router logits bit for bit plus
the native DeepSeek-V3 routing of every row (expert ids int32 ``[R, 8]`` and
weights FP32 ``[R, 8]`` holding the bf16-rounded native weights, descending
native key order). See
``glm_5_2_nvfp4__sm_103__tp8/csrc/postAttentionHopRouting.h`` for the
arithmetic, the gather protocol and the memory-order contract.

The gather buffers and the parity word are private to hop_routing and shared by every
layer that uses the same ``post_attention_hop`` (one TP group, one workspace); they are
allocated once, at model construction.
"""

from typing import Dict, Optional, Tuple

import torch

from ... import native_ops
from ...dispatch import log_capability

FEATURE = "hop_routing"
HIDDEN_SIZE = 6144
NUM_EXPERTS = 256
TOP_K = 8
MAX_ROWS = 8
NUM_RANKS = 8
ROUTED_SCALING = 2.5
GATHER_WORDS = 2 * MAX_ROWS * NUM_EXPERTS
STATE_WORDS = 4
NUM_STAMPS = 8
MAX_GRID_CTAS = 96
# The empty gather slot (0x80000000 as int32).
GATHER_SENTINEL = -(1 << 31)

_counts: Dict[str, int] = {}


def _count(key: str) -> None:
    _counts[key] = _counts.get(key, 0) + 1


def engagement_counts() -> Dict[str, int]:
    """Copy of the host-side call counters (eager calls and CUDA graph
    captures, not replays): ``hop_routing_hop`` counts routing-hop calls."""
    return dict(_counts)


def reset_engagement_counts() -> None:
    _counts.clear()


class _GatherState:
    """Gather buffers and parity of one ``post_attention_hop`` workspace."""

    def __init__(self, device: torch.device):
        self.gather = torch.full((GATHER_WORDS,), GATHER_SENTINEL, dtype=torch.int32, device=device)
        self.zstate = torch.zeros(STATE_WORDS, dtype=torch.int32, device=device)


def _gather_state(hop) -> _GatherState:
    state = getattr(hop, "_hop_routing_gather_state", None)
    if state is None:
        state = _GatherState(hop.flags.device)
        hop._hop_routing_gather_state = state
    return state


class HopRoutingTransport:
    """into-hop producer view plus the routing hop."""

    def __init__(self, into_hop):
        self.into_hop = into_hop
        self.hop = into_hop.hop
        self.rank = into_hop.rank
        self.flags = into_hop.flags
        self.mc_words = into_hop.mc_words
        state = _gather_state(self.hop)
        self.gather = state.gather
        self.zstate = state.zstate
        # Same schedule knobs as the into-hop entry (the hop arithmetic and protocol are the
        # into-hop entry's).
        self.early_poll = into_hop.early_poll
        self.router_load = into_hop.router_load

    @classmethod
    def create(cls, into_hop) -> Optional["HopRoutingTransport"]:
        """The routing hop of ``into_hop``'s hop, or None (into-hop path).

        Rank-uniform: ``into_hop`` exists on all ranks of the group or on none.
        """
        reason = "attention_output_into_hop transport unavailable" if into_hop is None else ""
        log_capability(FEATURE, "routing_hop", not reason, reason)
        return None if reason else cls(into_hop)

    def can_run(
        self,
        residual: torch.Tensor,
        norm_weight: torch.Tensor,
        router_weight: Optional[torch.Tensor],
        routing_bias: Optional[torch.Tensor],
    ) -> bool:
        """Host-only, rank-uniform eligibility of one call: the into-hop entry's, with router
        logits, and a contiguous bf16 ``[256]`` router bias on the device."""
        if router_weight is None or routing_bias is None:
            return False
        if (
            routing_bias.dtype != torch.bfloat16
            or tuple(routing_bias.shape) != (NUM_EXPERTS,)
            or not routing_bias.is_contiguous()
            or routing_bias.device != residual.device
        ):
            return False
        return self.into_hop.can_run(residual, norm_weight, router_weight)

    def __call__(
        self,
        residual: torch.Tensor,
        norm_weight: torch.Tensor,
        eps: float,
        router_weight: torch.Tensor,
        routing_bias: torch.Tensor,
        logits_in: Optional[torch.Tensor] = None,
        stamps: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run the routing hop on the partials the producer already published.

        Returns ``(normed, residual_out, router_logits, routing)`` where
        ``routing`` is int32 ``[2, R, 8]``: ``routing[0]`` the expert ids and
        ``routing[1]`` the FP32 bits of the weights (see :func:`split_routing`).
        ``logits_in`` (tests only) replaces the logits the routing reads.
        """
        _count("hop_routing_hop")
        normed, residual_out, logits, routing = native_ops.post_attention_hop_routing(
            residual.contiguous(),
            norm_weight.contiguous(),
            float(eps),
            router_weight,
            routing_bias,
            self.hop.uc_buffer,
            self.flags,
            self.gather,
            self.zstate,
            self.rank,
            NUM_RANKS,
            self.early_poll,
            self.router_load,
            ROUTED_SCALING,
            logits_in,
            stamps,
        )
        return normed, residual_out, logits, routing


def split_routing(routing: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """``(ids int32 [R, 8], weights fp32 [R, 8])`` views of a routing tensor."""
    return routing[0], routing[1].view(torch.float32)


__all__ = [
    "GATHER_SENTINEL",
    "NUM_STAMPS",
    "MAX_GRID_CTAS",
    "HopRoutingTransport",
    "engagement_counts",
    "reset_engagement_counts",
    "split_routing",
]
