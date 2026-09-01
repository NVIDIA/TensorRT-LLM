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
"""Device-side verify-window selection: the in-graph prologue.

The host planner decides windows from a confidence snapshot that is one
iteration old (the batch reshuffles across the lag, and the read itself must
not sync). This module ranks the block that is about to be verified by its
OWN confidence instead: everything from the slot-indexed gather to the
per-token row maps is a pure tensor function, so it can run at the head of
the captured graph, after the previous step's draft wrote the confidence
buffer and before this step's verify attention reads the layout.

The split of responsibilities follows the paper's dual-timescale design:

* the CAPACITY -- ``(padded_bs, bucket, budget)`` -- stays host-decided from
  the lagged snapshot, because it is the CUDA-graph key and the attention-DP
  agreement payload, both of which must exist before launch;
* the RANKING -- which requests win the budget -- is computed here from
  fresh confidence, at zero staleness.

Nothing in this module raises on data: a captured graph cannot branch.
Feasibility (the bucket bounds) is checkable from host-known scalars before
launch; per-row staleness degrades to the neutral row (verify everything for
that request), mirroring the host planner's fail-open semantics.
"""

from dataclasses import dataclass
from typing import Callable, Optional

import torch

from .dspark_ragged import build_qo_indptr, build_row_maps_device, fill_bucket_device
from .dspark_schedule import (
    NEUTRAL_CONFIDENCE_LOGIT,
    DSparkScheduleConfig,
    compute_survival,
    schedule_verify_lens_topk,
)

__all__ = [
    "DeviceWindowResult",
    "gather_packed_draft_tokens",
    "select_windows_device",
]


@dataclass
class DeviceWindowResult:
    """Everything the ragged layout consumers need, all device-resident.

    Attributes:
        verify_lens: ``[padded_bs]`` int32 token windows (bonus included),
            summing to exactly ``graph_num_tokens``; pad rows carry their
            fill.
        qo_indptr: ``[padded_bs + 1]`` int32 exclusive prefix sum.
        req_idx: ``[graph_num_tokens]`` int64 owning-row per packed token.
        kv_correction: ``[graph_num_tokens]`` int32; composed with a
            ``kv_lens`` gather it yields each token's KV extent
            (``refresh_ragged_row_kv_lens``).
    """

    verify_lens: torch.Tensor
    qo_indptr: torch.Tensor
    req_idx: torch.Tensor
    kv_correction: torch.Tensor


def gather_packed_draft_tokens(
    *,
    next_draft_tokens: torch.Tensor,
    batch_slots: torch.Tensor,
    verify_lens: torch.Tensor,
    qo_indptr: torch.Tensor,
    num_real: int,
    total_draft_tokens: int,
) -> torch.Tensor:
    """Gather the device-selected real draft rows, excluding bonus tokens.

    ``verify_lens`` and ``qo_indptr`` describe token windows that include one
    bonus/anchor per request.  The persistent draft buffer contains drafts
    only, packed request-major.  Construct exactly ``total_draft_tokens``
    owners so full-batch/full-K layouts never need a one-past-the-end discard
    slot for anchors.
    """
    if total_draft_tokens < 0:
        raise ValueError("total_draft_tokens must be non-negative")
    if total_draft_tokens == 0:
        return next_draft_tokens.new_empty((0,))
    device = verify_lens.device
    rows = torch.arange(num_real, device=device, dtype=torch.long)
    draft_counts = verify_lens[:num_real].to(torch.long) - 1
    owners = torch.repeat_interleave(rows, draft_counts, output_size=total_draft_tokens)
    # Removing one bonus from every preceding request converts the token
    # prefix into a draft-only prefix.
    draft_qo = qo_indptr[: num_real + 1].to(torch.long) - torch.arange(
        num_real + 1, device=device, dtype=torch.long
    )
    flat = torch.arange(total_draft_tokens, device=device, dtype=torch.long)
    offsets = flat - draft_qo[owners]
    slots = batch_slots[:num_real].to(torch.long)[owners]
    return next_draft_tokens[slots, offsets]


def select_windows_device(
    *,
    confidence_logits: torch.Tensor,
    slot_idx: torch.Tensor,
    num_real: torch.Tensor,
    budget: torch.Tensor,
    graph_num_tokens: int,
    cfg: DSparkScheduleConfig,
    apply_calibration: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    stamp: Optional[torch.Tensor] = None,
    expected_stamp: Optional[torch.Tensor] = None,
    pad_len: Optional[int] = None,
) -> DeviceWindowResult:
    """Rank the batch by fresh confidence and pack it into the agreed bucket.

    Mirrors the host chain ``_gather_rows -> apply_calibration ->
    compute_survival -> schedule_verify_lens_topk -> fill_bucket`` with the
    lag removed: ``confidence_logits`` is the LIVE slot-indexed buffer the
    previous draft pass scattered into, not a staged snapshot.

    Args:
        confidence_logits: ``[num_slots, K]`` raw logits, slot-indexed.
        slot_idx: ``[padded_bs]`` int64 buffer row per batch position;
            entries at or beyond ``num_real`` are never read meaningfully
            (point them at any valid row, e.g. 0).
        num_real: 0-d integer tensor; rows past it are padding.
        budget: 0-d integer tensor, verify tokens above the floor -- the
            host-decided (lagged) capacity knob, written before launch.
        graph_num_tokens: the captured token bucket (capture constant).
        cfg: scheduling bounds; ``resolved_max_verify_len + 1`` is the
            per-row token ceiling.
        apply_calibration: logits -> probabilities; sigmoid when None.
        stamp: optional ``[num_slots]`` last-writer stamps; with
            ``expected_stamp`` (0-d), rows whose stamp mismatches fall back
            to the neutral row (full survival: verify everything), the same
            fail-open the host planner applies to unknown slots.
        pad_len: when given, pad rows carry EXACTLY this many tokens and all
            fill slack goes to real rows -- the host fit's published split,
            which its pre-launch copy widths depend on (see
            :func:`fill_bucket_device`). The caller must clamp ``budget`` so
            the real rows can absorb ``graph_num_tokens - n_pad * pad_len``.

    Returns:
        :class:`DeviceWindowResult`; every tensor is derived without a host
        sync, so the call is capture-safe end to end.
    """
    if confidence_logits.dim() != 2:
        raise ValueError(
            f"confidence_logits must be [num_slots, K], got {tuple(confidence_logits.shape)}"
        )
    device = confidence_logits.device
    padded_bs = slot_idx.numel()

    selected = confidence_logits.index_select(0, slot_idx.to(torch.long))
    if stamp is not None:
        if expected_stamp is None:
            raise ValueError("stamp requires expected_stamp")
        stale = stamp.index_select(0, slot_idx.to(torch.long)) != expected_stamp
        selected = torch.where(
            stale.unsqueeze(1),
            torch.full_like(selected, NEUTRAL_CONFIDENCE_LOGIT),
            selected,
        )

    calibrate = apply_calibration or torch.sigmoid
    survival = compute_survival(calibrate(selected))
    # Pad rows must win no budget: zero survival sits below survival_eps, so
    # the ranking never selects them and their windows stay at the floor
    # until the fill tops them up.
    is_real = torch.arange(padded_bs, device=device) < num_real
    survival = torch.where(is_real.unsqueeze(1), survival, torch.zeros_like(survival))

    scheduled = schedule_verify_lens_topk(survival=survival, budget=budget, cfg=cfg)
    # The scheduler counts drafted positions; the token window adds the
    # bonus/anchor, matching every host fill_bucket callsite.
    token_lens = scheduled + 1
    max_token_len = int(cfg.resolved_max_verify_len) + 1
    filled = fill_bucket_device(
        token_lens,
        num_real=num_real,
        graph_num_tokens=graph_num_tokens,
        max_verify_len=max_token_len,
        pad_fill=pad_len,
    )
    req_idx, correction = build_row_maps_device(filled, graph_num_tokens=graph_num_tokens)
    return DeviceWindowResult(
        verify_lens=filled,
        qo_indptr=build_qo_indptr(filled),
        req_idx=req_idx,
        kv_correction=correction,
    )
