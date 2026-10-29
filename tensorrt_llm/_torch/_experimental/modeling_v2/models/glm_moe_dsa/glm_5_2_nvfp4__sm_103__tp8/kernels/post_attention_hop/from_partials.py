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
"""into-hop entry of the post_attention_hop (``attention_output_into_hop``).

Reachable only when ``attention_output_into_hop`` is selected: the attention tail's o_proj kernel
writes its rank-local partial directly into the current Lamport buffer of the
post_attention_hop workspace (multicast, sanitized like post_attention_hop's own publish), and the
hop then runs ``trtllm::glm_moe_dsa_post_attention_hop_from_partials``, which takes every rank's
partial from that transport. Its normalized hidden, residual and router logits are
bit-identical to the post_attention_hop consuming the ordinary partial (see
``cpp/tensorrt_llm/kernels/glmMoeDsa/postAttentionHopFromPartials.h`` for the
arithmetic and the memory-order contract with the producer).

``IntoHopTransport`` exposes what the producer needs (the multicast view of the
workspace, the protocol state and the rank) and runs the hop. It never
changes the post_attention_hop object or its behaviour; both kinds of calls share the
workspace and its buffer rotation. Its eligibility check is the post_attention_hop's
per-call check without the hop's fallback counting (the layer evaluates the
post_attention_hop itself when the into-hop path is not taken).

Host-side counters (:func:`into_hop_engagement_counts`) record into-hop calls (eager
runs and CUDA graph captures, not replays). An into-hop call replaces a post_attention_hop
call, so the post-attention hop calls of a run are the post_attention_hop's ``engaged``
count plus ``into_hop_engaged``.
"""

from typing import Dict, Optional, Tuple

import torch

from ...dispatch import log_capability
from .hop import HIDDEN_SIZE, LAMPORT_BUFFER_BYTES, NUM_RANKS, WORKSPACE_BYTES, PostAttnHop

FEATURE = "attention_output_into_hop"
_OP_NAME = "glm_moe_dsa_post_attention_hop_from_partials"
LAMPORT_BUFFER_WORDS = LAMPORT_BUFFER_BYTES // 4

# Host-side call counters; they count Python calls (eager runs and CUDA graph
# captures), not graph replays.
_counts: Dict[str, int] = {}


def _count(key: str) -> None:
    _counts[key] = _counts.get(key, 0) + 1


def into_hop_engagement_counts() -> Dict[str, int]:
    """Copy of the into-hop call counters.

    Keys: ``into_hop_engaged`` (into-hop calls) and ``into_hop_engaged_router`` (into-hop entry
    calls that also produced router logits).
    """
    return dict(_counts)


def reset_into_hop_engagement_counts() -> None:
    """Clear the into-hop call counters."""
    _counts.clear()


def op_available() -> bool:
    """Whether ``trtllm::glm_moe_dsa_post_attention_hop_from_partials`` is loaded."""
    return hasattr(torch.ops.trtllm, _OP_NAME)


def _agreed_op_reason(hop: PostAttnHop) -> str:
    """Why no rank of ``hop``'s group may use the into-hop entry ("" when every rank may).

    The op's availability is a fact of each rank's build. It is agreed once
    per hop over the hop's TP communicator, so a rank without the op cannot
    leave its peers publishing into a transport whose hop it never runs.
    """
    cached = getattr(hop, "_into_hop_op_reason", None)
    if cached is not None:
        return cached
    local = "" if op_available() else f"trtllm::{_OP_NAME} is not loaded"
    comm = getattr(hop, "comm", None)
    agreed = comm is None or comm.allreduce(0 if local else 1) == comm.Get_size()
    reason = local or ("" if agreed else f"trtllm::{_OP_NAME} is not loaded on another rank")
    hop._into_hop_op_reason = reason
    return reason


class IntoHopTransport:
    """Producer view of one post_attention_hop workspace plus the transport-input hop."""

    def __init__(self, hop: PostAttnHop):
        self.hop = hop
        self.rank = hop.rank
        self.flags = hop.flags
        # Multicast view of the whole workspace as 32-bit words: a store to
        # word i reaches word i of every rank's unicast buffer.
        self.mc_words = hop.handle.get_mc_buffer((WORKSPACE_BYTES // 4,), torch.int32, 0)
        # Poll the transport only after griddepcontrol.wait: the hop's CTAs
        # can be resident next to the producer's, and polling before then
        # competes with the running o_proj for the SM's issue slots.
        self.early_poll = False
        # Where the hop loads its router-weight fragments (0: after the
        # barrier that ends the rank sum; 1: right after the rank sum; 2:
        # right before the MMAs). Same outputs. 1 issues the loads early
        # enough that their latency overlaps the RMS tree and the cluster
        # exchange, while a hop CTA still fits next to a CTA of the
        # publishing o_proj.
        self.router_load = 1

    @classmethod
    def create(cls, hop: Optional[PostAttnHop]) -> Optional["IntoHopTransport"]:
        """Return the transport of ``hop``, or None (the native path, or the
        post_attention_hop without the into-hop transport).

        Rank-uniform: ``hop`` exists on all ranks of the group or on none
        (its creation is agreed), and the availability of the into-hop op is agreed
        over the hop's TP group on the first call for that hop, which every
        rank makes at the same point of model construction.
        """
        reason = "post_attention_hop hop unavailable" if hop is None else _agreed_op_reason(hop)
        log_capability(FEATURE, "into_hop_transport", not reason, reason)
        return None if reason else cls(hop)

    def can_run(
        self,
        residual: torch.Tensor,
        norm_weight: torch.Tensor,
        router_weight: Optional[torch.Tensor],
    ) -> bool:
        """Same eligibility as the post_attention_hop for this call (rank-uniform facts,
        no device work). Not counted as a post_attention_hop fallback: when the into-hop path is
        not taken, the layer runs the post_attention_hop's own check."""
        return self.hop.call_ineligibility(residual, residual, norm_weight, router_weight) is None

    def __call__(
        self,
        residual: torch.Tensor,
        norm_weight: torch.Tensor,
        eps: float,
        router_weight: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Run the hop on the partials the producer already published."""
        _count("into_hop_engaged")
        if router_weight is not None:
            _count("into_hop_engaged_router")
        normed, residual_out, logits = getattr(torch.ops.trtllm, _OP_NAME)(
            residual.contiguous(),
            norm_weight.contiguous(),
            float(eps),
            router_weight,
            self.hop.uc_buffer,
            self.flags,
            self.rank,
            NUM_RANKS,
            self.early_poll,
            self.router_load,
        )
        return normed, residual_out, (logits if router_weight is not None else None)


def _register_fake() -> None:
    if not op_available():
        return
    try:

        @torch.library.register_fake(f"trtllm::{_OP_NAME}")
        def _(
            residual,
            gamma,
            eps,
            router_weight,
            uc_buffer,
            flags,
            rank,
            nranks,
            early_poll,
            router_load=1,
        ):
            rows = residual.numel() // HIDDEN_SIZE
            logits = (
                residual.new_empty((rows, 256), dtype=torch.float32)
                if router_weight is not None
                else residual.new_empty((0,), dtype=torch.float32)
            )
            return [torch.empty_like(residual), torch.empty_like(residual), logits]
    except RuntimeError:
        # Already registered by an earlier import in this process.
        pass


_register_fake()

__all__ = [
    "LAMPORT_BUFFER_WORDS",
    "IntoHopTransport",
    "op_available",
    "reset_into_hop_engagement_counts",
    "into_hop_engagement_counts",
]
