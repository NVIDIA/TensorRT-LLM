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
"""Device-side verify-window selection for the pre-replay prologue.

The host planner decides windows from a confidence snapshot that is one
iteration old (the batch reshuffles across the lag, and the read itself must
not sync). This module ranks the block that is about to be verified by its
OWN confidence instead: everything from the slot-indexed gather to the
per-token row maps is device-resident.  The production exact-tier caller runs
it immediately before replay, after the previous step's draft wrote the
confidence buffer and before this step's verify attention reads the layout.
The tensor-control fallback remains suitable for capture by a caller whose
graph key owns all of its replay-varying controls.

The split of responsibilities follows the paper's dual-timescale design:

* the CAPACITY -- ``(padded_bs, bucket, budget)`` -- stays host-decided from
  the lagged snapshot, because it is the CUDA-graph key and the attention-DP
  agreement payload, both of which must exist before launch;
* the RANKING -- which requests win the budget -- is computed here from
  fresh confidence, at zero staleness.

Per-row staleness degrades to the neutral row (verify everything for that
request), mirroring the host planner's fail-open semantics.  Exact-tier
feasibility is checked from host-known scalars before the fused launch; an
unsupported fused shape falls back to the established tensor schedule/fill.
"""

from dataclasses import dataclass
from typing import Callable, Optional, Union

import torch

from .dspark_ragged import build_qo_indptr, build_row_maps_device, fill_bucket_device
from .dspark_schedule import (
    NEUTRAL_CONFIDENCE_LOGIT,
    DSparkScheduleConfig,
    compute_survival,
    schedule_verify_lens_topk,
    schedule_verify_lens_topk_fused_fill,
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
    num_real: Union[int, torch.Tensor],
    budget: Union[int, torch.Tensor],
    graph_num_tokens: int,
    cfg: DSparkScheduleConfig,
    apply_calibration: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    stamp: Optional[torch.Tensor] = None,
    expected_stamp: Optional[torch.Tensor] = None,
    pad_len: Optional[int] = None,
    use_fused_exact: bool = False,
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
        num_real: Python integer in the production pre-replay prologue, or a
            0-d integer tensor for a device-owned caller; rows past it are
            padding.
        budget: Python integer in the production pre-replay prologue, or a
            0-d integer tensor paired with tensor ``num_real``; verify tokens
            above the floor -- the host-decided (lagged) capacity knob.
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
        use_fused_exact: enable the policy-neutral Triton fast path.
            Production sets this only for shapes compiled successfully during
            warmup and when its independent default-off switch is enabled;
            false uses the established tensor schedule/fill implementation.

    Returns:
        :class:`DeviceWindowResult`; every output tensor stays device-resident.
        The Python-control exact path is intended for the pre-replay prologue,
        while the tensor-control fallback may be captured by a compatible
        caller.
    """
    if confidence_logits.dim() != 2:
        raise ValueError(
            f"confidence_logits must be [num_slots, K], got {tuple(confidence_logits.shape)}"
        )
    device = confidence_logits.device
    padded_bs = slot_idx.numel()
    controls_are_tensors = isinstance(num_real, torch.Tensor)
    if controls_are_tensors != isinstance(budget, torch.Tensor):
        raise TypeError("num_real and budget must both be Python ints or both be tensors")
    if controls_are_tensors:
        if num_real.dim() != 0 or budget.dim() != 0:
            raise ValueError("tensor num_real and budget must both be 0-d")
        # Narrow integer controls can overflow while multiplying by K or while
        # casting the verification budget; production scalar controls are ordinary
        # Python ints, and a device-owned caller must use a safe width.
        integer_dtypes = {torch.int32, torch.int64}
        if num_real.dtype not in integer_dtypes or budget.dtype not in integer_dtypes:
            raise TypeError("tensor num_real and budget must both have integer dtype")
        if num_real.device != device or budget.device != device:
            raise ValueError("tensor num_real and budget must share confidence_logits.device")

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
    # Pad rows must win no policy budget. Zero is below the configured
    # numerical epsilon, preserving the established tensor scheduler exactly.
    is_real = torch.arange(padded_bs, device=device) < num_real
    survival = torch.where(is_real.unsqueeze(1), survival, torch.zeros_like(survival))

    max_token_len = int(cfg.resolved_max_verify_len) + 1
    fused_exact = (
        use_fused_exact
        and pad_len is not None
        and not controls_are_tensors
        and padded_bs <= 256
        and cfg.survival_eps > 0.0
    )
    if fused_exact:
        filled = schedule_verify_lens_topk_fused_fill(
            survival=survival,
            budget=int(budget),
            num_real=int(num_real),
            pad_len=int(pad_len),
            cfg=cfg,
            graph_num_tokens=graph_num_tokens,
        )
    else:
        scheduled = schedule_verify_lens_topk(survival=survival, budget=budget, cfg=cfg)
        # The scheduler counts drafted positions; the token window adds the
        # bonus/anchor, matching every host fill_bucket callsite.
        token_lens = scheduled + 1
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
