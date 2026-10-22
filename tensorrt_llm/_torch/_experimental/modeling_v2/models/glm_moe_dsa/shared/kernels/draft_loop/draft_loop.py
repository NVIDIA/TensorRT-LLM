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
"""Fused bookkeeping of the MTP one-model draft loop (draft_loop).

``DraftLoopOps`` is created once per one-model MTP forward. Its host-side
eligibility check runs before any device work; when it returns None the
caller keeps the native code everywhere. Otherwise it replaces, with the same
results:

* acceptance (strict, optionally with the forced-acceptance override) and the
  draft-0 input ids and gather rows: one launch after the native target
  sampler, instead of about 26 small kernels;
* the step-0 last-accepted-row gather of the MTP hidden states: one launch;
* the per-step position/KV-length/DSA-metadata update (``draft_step``);
* the post-loop draft-token stack and ``next_new_tokens``: one launch.
"""

from typing import List, Optional, Tuple

import torch
import triton
import triton.language as tl

from . import draft_step as _draft_step
from . import verify as _verify

_MAX_DRAFT_LEN = 8
_MAX_BATCH = 64
_NUM_WARPS = 4


@triton.jit
def _accept_kernel(
    target_ptr,  # int32/int64 [B * (K + 1)] sampled target tokens
    draft_ptr,  # int32 [B * K] previous draft tokens
    batch_idx_ptr,  # int32 [>= B] spec_metadata.batch_indices_cuda
    counter_ptr,  # int64 [1] forced-acceptance draw counter (FORCE_MODE 2)
    pool_ptr,  # float32 [POOL] forced-acceptance uniform pool (FORCE_MODE 2)
    accepted_ptr,  # int32 [B, K + 1] (out)
    num_accepted_ptr,  # int32 [B] (out)
    input_ids_ptr,  # int32 [B * (K + 1)] draft-0 input ids (out)
    gather_ptr,  # int64 [B] step-0 last-accepted rows (out)
    frac,  # float32 fractional forced acceptance
    counter_stride,
    slot_stride,
    pool_mask,
    B: tl.constexpr,
    K: tl.constexpr,
    FORCE_MODE: tl.constexpr,
    BASE_TOTAL: tl.constexpr,
    MAX_TOTAL: tl.constexpr,
    BLOCK_B: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    offs_b = tl.arange(0, BLOCK_B)
    offs_k = tl.arange(0, BLOCK_K)
    mask_b = offs_b < B
    row_mask = mask_b[:, None] & (offs_k[None, :] < K + 1)
    flat = offs_b[:, None] * (K + 1) + offs_k[None, :]
    target = tl.load(target_ptr + flat, mask=row_mask, other=0)
    target32 = target.to(tl.int32)
    tl.store(accepted_ptr + flat, target32, mask=row_mask)
    tl.store(input_ids_ptr + flat, target32, mask=row_mask)

    draft_mask = mask_b[:, None] & (offs_k[None, :] < K)
    draft = tl.load(draft_ptr + offs_b[:, None] * K + offs_k[None, :], mask=draft_mask, other=0)
    mismatch = draft_mask & (draft.to(tl.int64) != target.to(tl.int64))
    # Leading matches == sum(cumprod(draft == target)).
    leading = tl.min(tl.where(mismatch, offs_k[None, :], K), 1)
    num_accepted = 1 + leading

    if FORCE_MODE == 1:
        num_accepted = tl.full((BLOCK_B,), BASE_TOTAL, tl.int32)
    elif FORCE_MODE == 2:
        counter = tl.load(counter_ptr) + 1
        tl.store(counter_ptr, counter)
        index = (counter * counter_stride + offs_b.to(tl.int64) * slot_stride) & pool_mask
        draw = tl.load(pool_ptr + index, mask=mask_b, other=1.0)
        extra = (draw < frac).to(tl.int32)
        num_accepted = tl.minimum(BASE_TOTAL + extra, MAX_TOTAL)
    tl.store(num_accepted_ptr + offs_b, num_accepted, mask=mask_b)

    batch_idx = tl.load(batch_idx_ptr + offs_b, mask=mask_b, other=0)
    rows = (batch_idx * (K + 1)).to(tl.int64) + num_accepted.to(tl.int64) - 1
    tl.store(gather_ptr + offs_b, rows, mask=mask_b)


@triton.jit
def _gather_rows_kernel(
    src_ptr,  # [num_src_rows, H]
    rows_ptr,  # int64 [B]
    out_ptr,  # [B, H] (out)
    hidden,
    src_row_stride,
    BLOCK_H: tl.constexpr,
):
    b = tl.program_id(0)
    cols = tl.program_id(1) * BLOCK_H + tl.arange(0, BLOCK_H)
    mask = cols < hidden
    row = tl.load(rows_ptr + b)
    values = tl.load(src_ptr + row * src_row_stride + cols, mask=mask)
    tl.store(out_ptr + b.to(tl.int64) * hidden + cols, values, mask=mask)


@triton.jit
def _finalize_kernel(
    d0,
    d1,
    d2,
    d3,
    d4,
    d5,
    d6,
    d7,  # int32 [B] draft tokens of steps 0..K-1
    accepted_ptr,  # int32 [B, ACC_COLS]
    num_accepted_ptr,  # int32 [B]
    batch_idx_ptr,  # int32 [>= B]
    next_draft_ptr,  # int32 [B, K] (out)
    next_new_ptr,  # int32 [B, K + 1] (out)
    B: tl.constexpr,
    K: tl.constexpr,
    ACC_COLS: tl.constexpr,
    BLOCK_B: tl.constexpr,
):
    offs_b = tl.arange(0, BLOCK_B)
    mask = offs_b < B
    batch_idx = tl.load(batch_idx_ptr + offs_b, mask=mask, other=0).to(tl.int64)
    last = tl.load(num_accepted_ptr + offs_b, mask=mask, other=1).to(tl.int64) - 1
    bonus = tl.load(accepted_ptr + batch_idx * ACC_COLS + last, mask=mask, other=0)
    tl.store(next_new_ptr + offs_b * (K + 1), bonus, mask=mask)
    for k in tl.static_range(K):
        if k == 0:
            tok = tl.load(d0 + offs_b, mask=mask)
        elif k == 1:
            tok = tl.load(d1 + offs_b, mask=mask)
        elif k == 2:
            tok = tl.load(d2 + offs_b, mask=mask)
        elif k == 3:
            tok = tl.load(d3 + offs_b, mask=mask)
        elif k == 4:
            tok = tl.load(d4 + offs_b, mask=mask)
        elif k == 5:
            tok = tl.load(d5 + offs_b, mask=mask)
        elif k == 6:
            tok = tl.load(d6 + offs_b, mask=mask)
        else:
            tok = tl.load(d7 + offs_b, mask=mask)
        tl.store(next_draft_ptr + offs_b * K + k, tok, mask=mask)
        tl.store(next_new_ptr + offs_b * (K + 1) + 1 + k, tok, mask=mask)


def gather_rows(src: torch.Tensor, rows: torch.Tensor) -> torch.Tensor:
    """``src[rows]`` for a 2-D contiguous ``src`` and int64 ``rows`` (one launch)."""
    num_rows = rows.shape[0]
    hidden = src.shape[1]
    out = torch.empty((num_rows, hidden), dtype=src.dtype, device=src.device)
    block_h = 2048
    _gather_rows_kernel[(num_rows, triton.cdiv(hidden, block_h))](
        src, rows, out, hidden, src.stride(0), BLOCK_H=block_h, num_warps=_NUM_WARPS
    )
    return out


class DraftLoopOps:
    """draft_loop operations for one one-model MTP forward (see module docstring)."""

    def __init__(self, worker, attn_metadata, spec_metadata):
        self._worker = worker
        self._attn_metadata = attn_metadata
        self._spec_metadata = spec_metadata
        self.batch_size = attn_metadata.num_seqs
        self.draft_len = spec_metadata.runtime_draft_len
        self.input_ids0: Optional[torch.Tensor] = None
        self.gather_ids: Optional[torch.Tensor] = None
        self.accepted_tokens: Optional[torch.Tensor] = None
        self.num_accepted_tokens: Optional[torch.Tensor] = None
        self.next_new_tokens: Optional[torch.Tensor] = None
        self.step_engaged = True

    # ------------------------------------------------------------------ setup
    @staticmethod
    def ineligibility(worker, attn_metadata, spec_metadata) -> Optional[str]:
        """Return why this forward must stay native, or None (host only)."""
        if not getattr(worker, "is_mtp_eagle", False):
            return "not MTP Eagle one-model"
        if getattr(worker, "_uses_external_shared_target_kv", False):
            return "external shared-target KV draft loop"
        if worker.guided_decoder is not None or worker.sa_enhancer is not None:
            return "guided decoding or SA enhancer"
        if getattr(worker, "_auxiliary_state_handlers", None):
            return "auxiliary speculative state handlers"
        if getattr(worker.spec_config, "use_relaxed_acceptance_for_thinking", False):
            return "relaxed acceptance"
        if getattr(worker, "use_dynamic_tree", False):
            return "dynamic tree"
        if getattr(spec_metadata, "enable_penalty", False):
            return "occurrence penalties"
        if worker._can_use_rejection_sampling(spec_metadata):
            return "rejection sampling"
        if spec_metadata.wants_advanced_draft_sampling:
            return "advanced draft sampling"
        if attn_metadata.num_contexts != 0:
            return "context rows present"
        batch_size = attn_metadata.num_seqs
        draft_len = spec_metadata.runtime_draft_len
        if not 0 < draft_len <= _MAX_DRAFT_LEN or not 0 < batch_size <= _MAX_BATCH:
            return "batch or draft length outside the fused domain"
        draft_tokens = spec_metadata.draft_tokens
        if (
            draft_tokens is None
            or draft_tokens.dtype != torch.int32
            or draft_tokens.numel() != batch_size * draft_len
            or not draft_tokens.is_contiguous()
        ):
            return "draft-token buffer layout"
        if spec_metadata.batch_indices_cuda.dtype != torch.int32:
            return "batch-index dtype"
        model_config = worker.model_config
        if model_config is not None and model_config.mapping.enable_lm_head_tp_in_adp:
            return "LM-head TP in attention DP"
        return None

    # ------------------------------------------------------------- acceptance
    def accept(self, logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Native target sampling, then fused strict acceptance and draft-0 prep.

        Returns ``(accepted_tokens [B, K+1] int32, num_accepted_tokens [B] int32)``
        exactly as ``_sample_and_accept_draft_tokens_base`` (with the forced
        acceptance override) returns them; also prepares the draft-0 input ids
        and the step-0 gather rows.
        """
        worker = self._worker
        spec_metadata = self._spec_metadata
        batch_size, draft_len = self.batch_size, self.draft_len
        if logits.dim() == 1:
            logits = logits.unsqueeze(0)
        device = logits.device
        target_tokens = worker._sample_tokens_for_batch(logits, spec_metadata, 0, batch_size)

        force_mode, base_total, max_total, frac = 0, 0, draft_len + 1, 0.0
        counter, pool = spec_metadata.batch_indices_cuda, spec_metadata.batch_indices_cuda
        force = worker.force_num_accepted_tokens
        is_warmup = spec_metadata.is_cuda_graph and not torch.cuda.is_current_stream_capturing()
        if force != 0.0 and not is_warmup:
            int_part = int(force)
            frac = force - int_part
            base_total = min(int_part + 1, max_total)
            if frac > 0.0 and base_total < max_total:
                worker._ensure_force_accept_rng_state(device)
                force_mode = 2
                counter = worker._force_accept_rng_counter
                pool = worker._force_accept_rng_pool
            else:
                force_mode = 1

        from tensorrt_llm._torch.speculative import interface as _iface

        accepted = torch.empty((batch_size, draft_len + 1), dtype=torch.int32, device=device)
        num_accepted = torch.empty((batch_size,), dtype=torch.int32, device=device)
        self.input_ids0 = torch.empty(
            (batch_size * (draft_len + 1),), dtype=torch.int32, device=device
        )
        self.gather_ids = torch.empty((batch_size,), dtype=torch.int64, device=device)
        _accept_kernel[(1,)](
            target_tokens,
            spec_metadata.draft_tokens,
            spec_metadata.batch_indices_cuda,
            counter,
            pool,
            accepted,
            num_accepted,
            self.input_ids0,
            self.gather_ids,
            float(frac),
            _iface._FORCE_ACCEPT_RNG_COUNTER_STRIDE,
            _iface._FORCE_ACCEPT_RNG_SLOT_STRIDE,
            _iface._FORCE_ACCEPT_RNG_POOL_SIZE - 1,
            B=batch_size,
            K=draft_len,
            FORCE_MODE=force_mode,
            BASE_TOTAL=base_total,
            MAX_TOTAL=max_total,
            BLOCK_B=max(2, triton.next_power_of_2(batch_size)),
            BLOCK_K=max(2, triton.next_power_of_2(draft_len + 1)),
            # One warp: every lane reads the draw counter before it is stored.
            num_warps=1,
        )
        self.accepted_tokens, self.num_accepted_tokens = accepted, num_accepted
        if _verify.verify_enabled():
            _verify.verify_accept(
                worker,
                logits,
                spec_metadata,
                batch_size,
                draft_len,
                accepted,
                num_accepted,
                self.gather_ids,
                self.input_ids0,
            )
        return accepted, num_accepted

    # ----------------------------------------------------------- draft loop
    def first_drafter_inputs(self, position_ids, hidden_states) -> dict:
        """Inputs of draft step 0 (``prepare_1st_drafter_inputs`` without context rows)."""
        return {
            "input_ids": self.input_ids0,
            "position_ids": position_ids,
            "hidden_states": hidden_states,
            "attn_metadata": self._attn_metadata,
            "spec_metadata": self._spec_metadata,
        }

    def step0_rows(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """``(gather_ids, hidden_states[gather_ids])`` for draft step 0."""
        if hidden_states.dim() != 2 or not hidden_states.is_contiguous():
            from ... import dispatch

            desc = (tuple(hidden_states.shape), tuple(hidden_states.stride()))
            dispatch.log_capability("draft_loop", f"rows:{desc}", False, f"hidden {desc}")
            return self.gather_ids, hidden_states[self.gather_ids]
        rows = gather_rows(hidden_states, self.gather_ids)
        if _verify.verify_enabled():
            _verify.verify_rows(hidden_states, self.gather_ids, rows)
        return self.gather_ids, rows

    def step_update(
        self, attn_metadata, step: int, num_accepted_tokens, position_ids, gather_ids
    ) -> Optional["_draft_step.DraftStepResult"]:
        """Fused per-step metadata update, or None for the native update."""
        if not self.step_engaged:
            return None

        def run_draft_loop():
            return _draft_step.draft_step_update(
                attn_metadata,
                step,
                self.batch_size,
                0,
                self.draft_len,
                num_accepted_tokens,
                position_ids,
                gather_ids,
            )

        from ... import dispatch

        reason = _draft_step.step_ineligibility(attn_metadata, self.batch_size, 0)
        dispatch.log_capability(
            "draft_loop", f"draft_step:{reason or 'engaged'}", reason is None, reason or ""
        )
        if _verify.verify_enabled() and reason is None:
            result = _verify.verified_step_update(
                run_draft_loop,
                attn_metadata,
                step,
                self.batch_size,
                self.draft_len,
                num_accepted_tokens,
                position_ids,
                gather_ids,
            )
        else:
            result = run_draft_loop()
        if result is None:
            # The metadata state is per forward: keep the remaining steps native.
            self.step_engaged = False
        return result

    def finalize(self, draft_tokens: List[torch.Tensor]) -> Optional[torch.Tensor]:
        """Stack the draft tokens and build ``next_new_tokens`` in one launch.

        Returns ``next_draft_tokens [B, K]`` (``next_new_tokens [B, K+1]`` is kept
        in ``self.next_new_tokens``), or None when the native code must run.
        """
        batch_size, draft_len = self.batch_size, self.draft_len
        accepted_tokens, num_accepted_tokens = self.accepted_tokens, self.num_accepted_tokens
        if accepted_tokens is None:
            return None
        if len(draft_tokens) != draft_len or any(
            t.dtype != torch.int32 or t.shape != (batch_size,) or not t.is_contiguous()
            for t in draft_tokens
        ):
            from ... import dispatch

            desc = [(str(t.dtype), tuple(t.shape), t.is_contiguous()) for t in draft_tokens[:1]]
            dispatch.log_capability("draft_loop", f"finalize:{desc}", False, f"draft tokens {desc}")
            return None
        if accepted_tokens.dtype != torch.int32 or not accepted_tokens.is_contiguous():
            return None
        device = accepted_tokens.device
        next_draft = torch.empty((batch_size, draft_len), dtype=torch.int32, device=device)
        next_new = torch.empty((batch_size, draft_len + 1), dtype=torch.int32, device=device)
        ptrs = list(draft_tokens) + [draft_tokens[0]] * (_MAX_DRAFT_LEN - draft_len)
        _finalize_kernel[(1,)](
            *ptrs,
            accepted_tokens,
            num_accepted_tokens,
            self._spec_metadata.batch_indices_cuda,
            next_draft,
            next_new,
            B=batch_size,
            K=draft_len,
            ACC_COLS=accepted_tokens.shape[1],
            BLOCK_B=max(2, triton.next_power_of_2(batch_size)),
            num_warps=1,
        )
        self.next_new_tokens = next_new
        if _verify.verify_enabled():
            _verify.verify_finalize(
                self._worker,
                draft_tokens,
                accepted_tokens,
                num_accepted_tokens,
                self._spec_metadata.batch_indices_cuda,
                batch_size,
                next_draft,
                next_new,
            )
        return next_draft


def make_draft_loop_ops(worker, attn_metadata, spec_metadata) -> Optional[DraftLoopOps]:
    """Return the draft_loop ops for this forward, or None to keep the native path."""
    from ... import dispatch

    if _verify.verify_enabled():
        _verify.COUNTERS.ensure_pool(
            attn_metadata.kv_lens_cuda.device if hasattr(attn_metadata, "kv_lens_cuda") else "cuda"
        )
        _verify.COUNTERS.report()
    reason = DraftLoopOps.ineligibility(worker, attn_metadata, spec_metadata)
    dispatch.log_capability(
        "draft_loop", f"draft_loop:{reason or 'engaged'}", reason is None, reason or ""
    )
    if reason is not None:
        return None
    return DraftLoopOps(worker, attn_metadata, spec_metadata)
