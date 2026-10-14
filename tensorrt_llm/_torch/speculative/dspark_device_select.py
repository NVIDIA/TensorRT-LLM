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
import triton
import triton.language as tl

from .dspark_schedule import (
    NEUTRAL_CONFIDENCE_LOGIT,
    DSparkFusedScheduleError,
    DSparkScheduleConfig,
    compute_survival,
    schedule_verify_lens_topk,
    schedule_verify_lens_topk_fused_fill,
)
from .ragged_helpers import build_qo_indptr, build_row_maps_device, fill_bucket_device

__all__ = [
    "DeviceWindowResult",
    "DeviceWindowWorkspace",
    "DSparkCompactLayoutError",
    "gather_packed_draft_tokens",
    "materialize_compact_layout",
    "select_windows_device",
]


class DSparkCompactLayoutError(DSparkFusedScheduleError):
    """Synchronous optional-layout failure with snapshot recovery state."""

    def __init__(self, message: str, *, past_seen_valid: bool) -> None:
        super().__init__(message)
        self.past_seen_valid = past_seen_valid


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
    workspace: Optional["DeviceWindowWorkspace"] = None


@dataclass
class DeviceWindowWorkspace:
    """Stable-address scratch for the fused compact-layout prologue.

    A workspace is owned by one model engine and sliced to the active graph
    shape.  Keeping these tensors alive eliminates steady-state scheduler,
    prefix-sum, row-map, and position-snapshot allocations.  The tensor
    fallback intentionally does not use this object, preserving it as an
    independent oracle.
    """

    verify_lens: torch.Tensor
    qo_indptr: torch.Tensor
    req_idx: torch.Tensor
    kv_correction: torch.Tensor
    past_seen: torch.Tensor

    @classmethod
    def allocate(
        cls,
        *,
        max_rows: int,
        max_tokens: int,
        device: Union[str, torch.device],
    ) -> "DeviceWindowWorkspace":
        if max_rows < 0 or max_tokens < 0:
            raise ValueError("workspace capacities must be non-negative")
        return cls(
            verify_lens=torch.empty(max_rows, dtype=torch.int32, device=device),
            qo_indptr=torch.empty(max_rows + 1, dtype=torch.int32, device=device),
            req_idx=torch.empty(max_tokens, dtype=torch.int64, device=device),
            kv_correction=torch.empty(max_tokens, dtype=torch.int32, device=device),
            past_seen=torch.empty(max_rows, dtype=torch.int32, device=device),
        )

    def validate(self, *, device: torch.device, num_rows: int, num_tokens: int) -> None:
        tensors = (
            ("verify_lens", self.verify_lens, torch.int32, num_rows),
            ("qo_indptr", self.qo_indptr, torch.int32, num_rows + 1),
            ("req_idx", self.req_idx, torch.int64, num_tokens),
            ("kv_correction", self.kv_correction, torch.int32, num_tokens),
            ("past_seen", self.past_seen, torch.int32, num_rows),
        )
        for name, tensor, dtype, capacity in tensors:
            if tensor.device != device:
                raise ValueError(f"workspace.{name} must be on {device}")
            if tensor.dtype != dtype:
                raise TypeError(f"workspace.{name} must have dtype {dtype}, got {tensor.dtype}")
            if tensor.dim() != 1 or tensor.numel() < capacity:
                raise ValueError(
                    f"workspace.{name} needs capacity {capacity}, got shape {tuple(tensor.shape)}"
                )
            if not tensor.is_contiguous():
                raise ValueError(f"workspace.{name} must be contiguous")


@triton.jit
def _materialize_row_maps_kernel(
    verify_lens_ptr,
    qo_indptr_ptr,
    req_idx_ptr,
    correction_ptr,
    MAX_TOKEN_LEN: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, MAX_TOKEN_LEN)
    length = tl.load(verify_lens_ptr + row)
    token = tl.load(qo_indptr_ptr + row) + offsets
    mask = offsets < length
    tl.store(req_idx_ptr + token, row, mask=mask)
    tl.store(correction_ptr + token, offsets - length + 1, mask=mask)


def _build_row_maps_workspace(
    *,
    workspace: DeviceWindowWorkspace,
    num_rows: int,
    graph_num_tokens: int,
    max_token_len: int,
) -> None:
    verify_lens = workspace.verify_lens[:num_rows]
    qo_indptr = workspace.qo_indptr[: num_rows + 1]
    qo_indptr[0].zero_()
    if num_rows == 0:
        return
    torch.cumsum(verify_lens, dim=0, out=qo_indptr[1:])
    if verify_lens.is_cuda:
        try:
            _materialize_row_maps_kernel[(num_rows,)](
                verify_lens,
                qo_indptr,
                workspace.req_idx,
                workspace.kv_correction,
                MAX_TOKEN_LEN=triton.next_power_of_2(max_token_len),
            )
        except Exception as exc:
            raise DSparkFusedScheduleError("DSpark fused row-map materialization failed") from exc
        return

    # CPU is a hardware-independent differential-test path.  Production CUDA
    # never enters the allocation-heavy tensor oracle below.
    req_idx, correction = build_row_maps_device(verify_lens, graph_num_tokens=graph_num_tokens)
    workspace.req_idx[:graph_num_tokens].copy_(req_idx)
    workspace.kv_correction[:graph_num_tokens].copy_(correction)


@triton.jit(do_not_specialize=["num_rows"])
def _snapshot_past_seen_kernel(
    old_qo_ptr,
    position_ids_ptr,
    past_seen_ptr,
    num_rows,
    BLOCK: tl.constexpr,
):
    rows = tl.arange(0, BLOCK)
    mask = rows < num_rows
    starts = tl.load(old_qo_ptr + rows, mask=mask, other=0)
    past_seen = tl.load(position_ids_ptr + starts, mask=mask, other=0)
    tl.store(past_seen_ptr + rows, past_seen, mask=mask)


@triton.jit(
    do_not_specialize=[
        "new_tokens_stride_token",
        "new_tokens_stride_slot",
        "next_draft_stride_slot",
        "next_draft_stride_token",
        "num_tokens",
        "num_real",
        "real_tokens",
    ]
)
def _materialize_compact_tokens_kernel(
    req_idx_ptr,
    qo_indptr_ptr,
    past_seen_ptr,
    batch_slots_ptr,
    new_tokens_ptr,
    new_tokens_lens_ptr,
    next_draft_tokens_ptr,
    input_ids_ptr,
    position_ids_ptr,
    previous_pos_indices_ptr,
    previous_pos_id_offsets_ptr,
    draft_tokens_ptr,
    new_tokens_stride_token,
    new_tokens_stride_slot,
    next_draft_stride_slot,
    next_draft_stride_token,
    num_tokens,
    num_real,
    real_tokens,
    BLOCK: tl.constexpr,
):
    token = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    token_mask = token < num_tokens
    row = tl.load(req_idx_ptr + token, mask=token_mask, other=0).to(tl.int32)
    row_start = tl.load(qo_indptr_ptr + row, mask=token_mask, other=0)
    offset = token - row_start
    owner = tl.minimum(row, num_real - 1)
    slot = tl.load(batch_slots_ptr + owner, mask=token_mask, other=0)

    new_token = tl.load(
        new_tokens_ptr + offset * new_tokens_stride_token + slot * new_tokens_stride_slot,
        mask=token_mask,
        other=0,
    )
    tl.store(input_ids_ptr + token, new_token, mask=token_mask)
    past_seen = tl.load(past_seen_ptr + row, mask=token_mask, other=0)
    tl.store(position_ids_ptr + token, past_seen + offset, mask=token_mask)

    real_mask = token < real_tokens
    tl.store(previous_pos_indices_ptr + token, slot, mask=real_mask)
    new_len = tl.load(new_tokens_lens_ptr + slot, mask=real_mask, other=0)
    tl.store(previous_pos_id_offsets_ptr + token, new_len, mask=real_mask)

    draft_mask = token_mask & (row < num_real) & (offset > 0)
    draft_token = tl.load(
        next_draft_tokens_ptr
        + slot * next_draft_stride_slot
        + (offset - 1) * next_draft_stride_token,
        mask=draft_mask,
        other=0,
    )
    draft_index = row_start - row + offset - 1
    tl.store(draft_tokens_ptr + draft_index, draft_token, mask=draft_mask)


@triton.jit(do_not_specialize=["num_rows"])
def _commit_compact_rows_kernel(
    new_lens_ptr,
    new_qo_ptr,
    old_lens_ptr,
    old_qo_ptr,
    kv_lens_ptr,
    previous_kv_offsets_ptr,
    num_rows,
    BLOCK: tl.constexpr,
):
    rows = tl.arange(0, BLOCK)
    row_mask = rows < num_rows
    new_lens = tl.load(new_lens_ptr + rows, mask=row_mask, other=0)
    old_lens = tl.load(old_lens_ptr + rows, mask=row_mask, other=0)
    delta = new_lens - old_lens
    kv_lens = tl.load(kv_lens_ptr + rows, mask=row_mask, other=0)
    previous_offsets = tl.load(previous_kv_offsets_ptr + rows, mask=row_mask, other=0)
    tl.store(kv_lens_ptr + rows, kv_lens + 2 * delta, mask=row_mask)
    tl.store(previous_kv_offsets_ptr + rows, previous_offsets - delta, mask=row_mask)
    tl.store(old_lens_ptr + rows, new_lens, mask=row_mask)
    qo_mask = rows <= num_rows
    new_qo = tl.load(new_qo_ptr + rows, mask=qo_mask, other=0)
    tl.store(old_qo_ptr + rows, new_qo, mask=qo_mask)


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


def materialize_compact_layout(
    *,
    result: DeviceWindowResult,
    old_verify_lens: torch.Tensor,
    old_qo_indptr: torch.Tensor,
    batch_slots: torch.Tensor,
    new_tokens: torch.Tensor,
    new_tokens_lens: torch.Tensor,
    next_draft_tokens: torch.Tensor,
    input_ids: torch.Tensor,
    position_ids: torch.Tensor,
    previous_pos_indices: torch.Tensor,
    previous_pos_id_offsets: torch.Tensor,
    draft_tokens: torch.Tensor,
    kv_lens: torch.Tensor,
    previous_kv_lens_offsets: torch.Tensor,
    num_real: int,
    real_tokens: int,
) -> None:
    """Materialize a selected compact layout without per-step temporaries.

    CUDA snapshots row positions, writes token/position/overlap/draft buffers,
    and commits the row delta with three allocation-free Triton launches.  A
    CPU implementation mirrors the former tensor code for differential tests;
    it is deliberately not a second optimized production path.

    The row commit is last.  Therefore a synchronous failure in either earlier
    launch leaves the staged lens/qo/KV state intact and lets the caller rerun
    the established tensor fallback safely.
    """
    workspace = result.workspace
    if workspace is None:
        raise ValueError("compact materialization requires a workspace-backed result")
    num_rows = result.verify_lens.numel()
    num_tokens = result.req_idx.numel()
    num_real = int(num_real)
    real_tokens = int(real_tokens)
    if not 0 < num_real <= num_rows:
        raise ValueError(f"num_real must be in [1, {num_rows}], got {num_real}")
    if not 0 <= real_tokens <= num_tokens:
        raise ValueError(f"real_tokens must be in [0, {num_tokens}], got {real_tokens}")
    device = result.verify_lens.device
    workspace.validate(device=device, num_rows=num_rows, num_tokens=num_tokens)

    one_dimensional = (
        ("old_verify_lens", old_verify_lens, num_rows),
        ("old_qo_indptr", old_qo_indptr, num_rows + 1),
        ("batch_slots", batch_slots, num_real),
        ("new_tokens_lens", new_tokens_lens, 1),
        ("input_ids", input_ids, num_tokens),
        ("position_ids", position_ids, num_tokens),
        ("previous_pos_indices", previous_pos_indices, real_tokens),
        ("previous_pos_id_offsets", previous_pos_id_offsets, real_tokens),
        ("draft_tokens", draft_tokens, real_tokens - num_real),
        ("kv_lens", kv_lens, num_rows),
        ("previous_kv_lens_offsets", previous_kv_lens_offsets, num_rows),
    )
    for name, tensor, capacity in one_dimensional:
        if tensor.device != device:
            raise ValueError(f"{name} must be on {device}")
        if tensor.dim() != 1 or tensor.numel() < capacity:
            raise ValueError(f"{name} needs capacity {capacity}, got shape {tuple(tensor.shape)}")
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous")
    if new_tokens.device != device or new_tokens.dim() not in (2, 3):
        raise ValueError("new_tokens must be a 2-D or 3-D tensor on the layout device")
    if new_tokens.dim() == 3 and new_tokens.shape[2] != 1:
        raise ValueError("new_tokens trailing beam dimension must be one")
    if next_draft_tokens.device != device or next_draft_tokens.dim() != 2:
        raise ValueError("next_draft_tokens must be a 2-D tensor on the layout device")
    if old_verify_lens.dtype != torch.int32 or old_qo_indptr.dtype != torch.int32:
        raise TypeError("old verify_lens and qo_indptr must have dtype torch.int32")
    if position_ids.dtype != torch.int32 or workspace.past_seen.dtype != position_ids.dtype:
        raise TypeError("position_ids and workspace.past_seen must have dtype torch.int32")
    real_draft = real_tokens - num_real
    if real_draft < 0:
        raise ValueError("real_tokens must include one bonus token per real row")

    if result.verify_lens.is_cuda:
        row_block = triton.next_power_of_2(num_rows + 1)
        token_block = 256
        try:
            _snapshot_past_seen_kernel[(1,)](
                old_qo_indptr,
                position_ids,
                workspace.past_seen,
                num_rows,
                BLOCK=row_block,
            )
        except Exception as exc:
            raise DSparkCompactLayoutError(
                "DSpark compact-layout position snapshot failed", past_seen_valid=False
            ) from exc
        try:
            _materialize_compact_tokens_kernel[(triton.cdiv(num_tokens, token_block),)](
                result.req_idx,
                result.qo_indptr,
                workspace.past_seen,
                batch_slots,
                new_tokens,
                new_tokens_lens,
                next_draft_tokens,
                input_ids,
                position_ids,
                previous_pos_indices,
                previous_pos_id_offsets,
                draft_tokens,
                new_tokens.stride(0),
                new_tokens.stride(1),
                next_draft_tokens.stride(0),
                next_draft_tokens.stride(1),
                num_tokens,
                num_real,
                real_tokens,
                BLOCK=token_block,
            )
            _commit_compact_rows_kernel[(1,)](
                result.verify_lens,
                result.qo_indptr,
                old_verify_lens,
                old_qo_indptr,
                kv_lens,
                previous_kv_lens_offsets,
                num_rows,
                BLOCK=row_block,
            )
        except Exception as exc:
            raise DSparkCompactLayoutError(
                "DSpark fused compact-layout materialization failed", past_seen_valid=True
            ) from exc
        return

    # Hardware-independent oracle for exact differential coverage.
    split_lens = old_verify_lens[:num_rows].clone()
    split_qo = old_qo_indptr[: num_rows + 1].to(torch.long)
    past_seen = position_ids[split_qo[:-1]].clone()
    workspace.past_seen[:num_rows].copy_(past_seen)
    past_seen = workspace.past_seen[:num_rows]
    req_idx = result.req_idx
    flat = torch.arange(num_tokens, device=device)
    offset = flat - result.qo_indptr.to(torch.long)[req_idx]
    prev_slots = batch_slots[:num_real].to(torch.long)
    slots_tok = prev_slots[req_idx.clamp(max=num_real - 1)]
    input_ids[:num_tokens].copy_(
        new_tokens.transpose(0, 1)[slots_tok, offset].flatten().to(input_ids.dtype)
    )
    position_ids[:num_tokens].copy_(past_seen[req_idx] + offset.to(past_seen.dtype))
    previous_pos_indices[:real_tokens].copy_(slots_tok[:real_tokens].to(previous_pos_indices.dtype))
    previous_pos_id_offsets[:real_tokens].copy_(
        new_tokens_lens[slots_tok[:real_tokens]].to(previous_pos_id_offsets.dtype)
    )
    if real_draft > 0:
        draft_tokens[:real_draft].copy_(
            gather_packed_draft_tokens(
                next_draft_tokens=next_draft_tokens,
                batch_slots=prev_slots,
                verify_lens=result.verify_lens,
                qo_indptr=result.qo_indptr,
                num_real=num_real,
                total_draft_tokens=real_draft,
            ).to(draft_tokens.dtype)
        )
    window_delta = result.verify_lens - split_lens
    kv_lens[:num_rows] += 2 * window_delta
    previous_kv_lens_offsets[:num_rows] -= window_delta.to(previous_kv_lens_offsets.dtype)
    old_verify_lens[:num_rows].copy_(result.verify_lens)
    old_qo_indptr[: num_rows + 1].copy_(result.qo_indptr)


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
    workspace: Optional[DeviceWindowWorkspace] = None,
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
        workspace: reusable output storage for the fused exact path. It is
            ignored by the tensor fallback so that path remains an independent
            allocation-owning oracle.

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
    max_token_len = int(cfg.resolved_max_verify_len) + 1
    fused_exact = (
        use_fused_exact
        and pad_len is not None
        and not controls_are_tensors
        and padded_bs <= 256
        and cfg.survival_eps > 0.0
    )
    if fused_exact:
        if workspace is not None:
            workspace.validate(device=device, num_rows=padded_bs, num_tokens=graph_num_tokens)
        filled = schedule_verify_lens_topk_fused_fill(
            survival=survival,
            budget=int(budget),
            num_real=int(num_real),
            pad_len=int(pad_len),
            cfg=cfg,
            graph_num_tokens=graph_num_tokens,
            out=None if workspace is None else workspace.verify_lens,
        )
    else:
        # Pad rows must win no policy budget. Zero is below the configured
        # numerical epsilon, preserving the established tensor scheduler
        # exactly. The fused scheduler masks them internally and avoids this
        # arange/where temporary in production.
        is_real = torch.arange(padded_bs, device=device) < num_real
        survival = torch.where(is_real.unsqueeze(1), survival, torch.zeros_like(survival))
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
    if fused_exact and workspace is not None:
        _build_row_maps_workspace(
            workspace=workspace,
            num_rows=padded_bs,
            graph_num_tokens=graph_num_tokens,
            max_token_len=max_token_len,
        )
        return DeviceWindowResult(
            verify_lens=workspace.verify_lens[:padded_bs],
            qo_indptr=workspace.qo_indptr[: padded_bs + 1],
            req_idx=workspace.req_idx[:graph_num_tokens],
            kv_correction=workspace.kv_correction[:graph_num_tokens],
            workspace=workspace,
        )
    req_idx, correction = build_row_maps_device(filled, graph_num_tokens=graph_num_tokens)
    return DeviceWindowResult(
        verify_lens=filled,
        qo_indptr=build_qo_indptr(filled),
        req_idx=req_idx,
        kv_correction=correction,
    )
