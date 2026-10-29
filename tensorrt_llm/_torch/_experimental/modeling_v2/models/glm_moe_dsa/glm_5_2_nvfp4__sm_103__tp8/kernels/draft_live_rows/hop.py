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
"""Live-row post-attention hop of the MTP layer (draft_live_rows).

At MTP draft step 0 the layer runs ``B * (K + 1)`` rows (every draft input
token of every request), but after the layer only the last accepted row of
each request is read: the draft LM head and the next step consume
``hidden_states[gather_ids]``, with ``gather_ids`` written on the device by
draft_loop's accept kernel. Attention needs every row (it appends their KV), while
everything after the post-attention hop is per row. :class:`LiveRowHop`
therefore replaces the post_attention_hop of that call by a fork that reads ``gather_ids``
after its programmatic dependency wait and reduces, adds the residual,
normalizes and computes router logits for the ``B`` live rows only; the MoE
and the post-MoE all-reduce that follow then run on ``B`` rows.

Every per-row quantity is computed with the post_attention_hop's arithmetic (fixed rank
order, the native RMS tree, the router MMA with the 8 slices summed in a
fixed order), so live row ``r`` equals row ``gather_ids[r]`` of the post_attention_hop
over all rows, bit for bit. The fork runs on the post_attention_hop's workspace and
device-side protocol state; post_attention_hop and draft_live_rows calls may follow each other in
any order on the stream that issues the TP group's hops.

Eligibility is decided from rank-uniform host facts only (the selection, the
post_attention_hop's agreed SM budget, shapes and dtypes), before any device work. The
one rank-local fact, whether this process carries the operator, is agreed
over the post_attention_hop's TP communicator when the live-row hop is built: it exists on
every rank of the group or on none.
"""

from typing import Any, Dict, Optional, Tuple

import torch

from tensorrt_llm.logger import logger

from ... import native_ops
from ...dispatch import log_capability

FEATURE = "draft_live_rows"
HIDDEN_SIZE = 6144
NUM_RANKS = 8
NUM_EXPERTS = 256
MAX_ROWS = 8
# The post_attention_hop workspace: [3 Lamport buffers][8 rows][8 ranks][6144] bf16.
WORKSPACE_BYTES = 3 * MAX_ROWS * NUM_RANKS * HIDDEN_SIZE * 2
# CTAs per cluster and the router layout's expert / row groups (kernel).
CLUSTER_CTAS = 8
ROUTER_EXPERT_GROUPS = 4
ROUTER_MAX_ROW_GROUPS = 3

# Host-side call counters. They count Python calls (eager runs and CUDA graph
# captures), not graph replays.
_counts: Dict[str, int] = {}


def _count(key: str) -> None:
    _counts[key] = _counts.get(key, 0) + 1


def engagement_counts() -> Dict[str, int]:
    """Copy of the host-side call counters.

    Keys: ``engaged`` (live-row hop calls), ``declined`` and
    ``declined:<reason>`` (armed calls that ran the post_attention_hop instead),
    ``step0_armed`` and ``step0_compacted`` (draft step-0 calls armed by the
    draft loop and those whose MTP layer returned the live rows).
    """
    return dict(_counts)


def reset_engagement_counts() -> None:
    """Clear the host-side call counters."""
    _counts.clear()


def required_sms(rows: int) -> int:
    """SMs one live-row hop of ``rows`` live rows needs (every CTA co-resident)."""
    return CLUSTER_CTAS * max(rows, ROUTER_EXPERT_GROUPS * min(rows, ROUTER_MAX_ROW_GROUPS))


class LiveRowHop:
    """The live-row hop on the workspace of a post_attention_hop (one per TP group)."""

    # id(post_attention_hop) -> (post_attention_hop, live-row hop or None, reason): one decision per
    # post_attention_hop.
    _instances: Dict[int, Tuple[Any, Optional["LiveRowHop"], str]] = {}

    @classmethod
    def for_post_attn_hop(cls, post_attn_hop: Any) -> Optional["LiveRowHop"]:
        """Return the live-row hop that shares ``post_attn_hop``'s workspace, or None.

        Must be called at the same point of model construction on every rank
        of the post_attention_hop's TP group. The post_attention_hop exists on every rank or on
        none; whether the operator is loaded is agreed over the post_attention_hop's
        communicator the first time a post_attention_hop is seen (one collective, also on
        a rank without the operator), so either every rank gets the live-row
        hop or every rank keeps the post_attention_hop. Later calls for the same post_attention_hop
        return the recorded decision without communicating.
        """
        if post_attn_hop is None or not getattr(post_attn_hop, "enabled", False):
            log_capability(
                FEATURE, "live_row_hop:no post_attention_hop", False, "no post_attention_hop hop"
            )
            return None
        key = id(post_attn_hop)
        entry = cls._instances.get(key)
        if entry is None or entry[0] is not post_attn_hop:
            hop, reason = cls._agreed(post_attn_hop)
            entry = (post_attn_hop, hop, reason)
            cls._instances[key] = entry
            if hop is None:
                logger.warning(
                    f"[glm_moe_dsa] draft_live_rows live-row hop disabled on every rank: {reason}"
                )
        _, hop, reason = entry
        log_capability(FEATURE, "live_row_hop", hop is not None, reason)
        return hop

    @classmethod
    def _agreed(cls, post_attn_hop: Any) -> Tuple[Optional["LiveRowHop"], str]:
        """Build the live-row hop if every rank of the post_attention_hop's group carries the operator."""
        comm = getattr(post_attn_hop, "comm", None)
        if comm is None:
            # An enabled post_attention_hop always has its communicator; without one there is no
            # group to agree in.
            return None, "post_attention_hop hop has no communicator"
        return cls(post_attn_hop), ""

    def __init__(self, post_attn_hop: Any):
        self.w2 = post_attn_hop
        self.rank = post_attn_hop.rank
        self.uc_buffer = post_attn_hop.uc_buffer
        self.mc_ptr = post_attn_hop.mc_ptr
        self.flags = post_attn_hop.flags
        self.min_sm_count = post_attn_hop.min_sm_count

    def call_ineligibility(
        self,
        hidden_states: torch.Tensor,
        row_ids: Optional[torch.Tensor],
        router_weight: Optional[torch.Tensor],
    ) -> Optional[str]:
        """Reason this armed call must run the post_attention_hop instead, or None (host only).

        The caller has already established that the post_attention_hop serves the call
        (same input, residual and norm weight), so only the live-row specific
        facts are checked here.
        """
        if router_weight is None:
            return "no router weight"
        if row_ids is None:
            return "no live rows"
        if (
            not isinstance(row_ids, torch.Tensor)
            or row_ids.dtype != torch.int64
            or row_ids.dim() != 1
            or not row_ids.is_contiguous()
            or row_ids.device != hidden_states.device
        ):
            return "live-row ids layout"
        input_rows = hidden_states.numel() // HIDDEN_SIZE
        rows = row_ids.shape[0]
        if not 1 <= rows <= input_rows or input_rows > MAX_ROWS:
            return f"live rows {rows} of {input_rows}"
        if required_sms(rows) > self.min_sm_count:
            return "SM count"
        return None

    def can_run(
        self,
        hidden_states: torch.Tensor,
        row_ids: Optional[torch.Tensor],
        router_weight: Optional[torch.Tensor],
    ) -> bool:
        """Whether this armed call can use the live-row hop (rank-uniform, no device work)."""
        reason = self.call_ineligibility(hidden_states, row_ids, router_weight)
        if reason is not None:
            _count("declined")
            _count(f"declined:{reason}")
            log_capability(FEATURE, f"live_row_hop_call:{reason}", False, reason)
            return False
        log_capability(FEATURE, "live_row_hop_call", True)
        return True

    def __call__(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        norm_weight: torch.Tensor,
        eps: float,
        router_weight: torch.Tensor,
        row_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run the hop; returns ``(normed, residual_out, router_logits)`` of the live rows.

        Callers check :meth:`can_run` first.
        """
        _count("engaged")
        return native_ops.draft_live_row_hop(
            hidden_states.contiguous(),
            residual.contiguous(),
            row_ids,
            norm_weight.contiguous(),
            float(eps),
            router_weight,
            self.uc_buffer,
            self.mc_ptr,
            self.flags,
            self.rank,
            NUM_RANKS,
        )
