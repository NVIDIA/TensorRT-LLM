# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The decode step's input glue as single native launches (RuntimeHooks items), shared by the M3.1 targets.

Family-level copies of what the sm_107 TP targets install from their own modules (tp8 ops/spec/m3_step_prologue,
ops/spec/m3_spec_offsets, kernels/msa_kv_update, kernels/m3_spec_tail.step_gather and M3DSparkStepFold); every op is
compiled into TensorRT-LLM (no Triton) and writes exactly what the engine's own lines write:

* ``step_prologue``: ``RuntimeHooks.pack_prologue_op``, the in-graph step-input prologue (host copies, device copies,
  header zeroed, epoch bumped) in one launch instead of two scatter launches and a header fill;
* ``spec_offsets``: ``RuntimeHooks.spec_offsets_op``, the overlap scheduler's speculative step correction
  (positions += offsets, kv_lens += offsets) in one launch instead of two adds;
* ``msa_kv_update``: ``MsaRuntimePolicy.kv_update``, the MSA KV-length correction in one launch instead of the
  metadata's ~23 one-CTA torch kernels;
* ``M3DSparkStepFold``: ``RuntimeHooks.spec_step_fold`` for the in-tree DSpark worker, the engine's previous-batch
  input gathers in one ``step_gather`` launch (the sampler keeps its own slot stores).

Each callable returns False, having written nothing, for an operand outside what its kernel covers; the caller then
runs its own lines. A target installs them only when ``ops_available()``. ``M3DSparkStepFold`` keeps the TP8 class's
name and gather but none of its forward-tail staging (the DSpark forward does not store), so it uploads nothing.
"""

from __future__ import annotations

from typing import Optional

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.native import family_ops
from tensorrt_llm.logger import logger

_ENGAGED: set = set()


def _engaged(item: str, line: str) -> None:
    if item not in _ENGAGED:
        _ENGAGED.add(item)
        logger.info(f"[M3 host] {line}")


_OPS: dict = {}

#: The four ops, by hook.
OP_NAMES = {
    "prologue": "minimax_m3_sm107_tp8_step_prologue",
    "offsets": "minimax_m3_sm107_tp8_spec_offsets",
    "kv_update": "minimax_m3_msa_kv_update",
    "gather": "minimax_m3_step_gather",
}


#: Where each hook's op lives: the family library carries its fixed copy of the MSA KV-length correction,
#: TensorRT-LLM the other three.
_LOOKUPS = {
    "prologue": lambda: getattr(torch.ops.trtllm, "minimax_m3_sm107_tp8_step_prologue", None),
    "offsets": lambda: getattr(torch.ops.trtllm, "minimax_m3_sm107_tp8_spec_offsets", None),
    "kv_update": lambda: getattr(family_ops(), "minimax_m3_msa_kv_update", None),
    "gather": lambda: getattr(torch.ops.trtllm, "minimax_m3_step_gather", None),
}


def _op(hook: str):
    """The op of ``hook`` (a key of OP_NAMES) when this build carries it, else None (looked up once)."""
    if hook not in _OPS:
        _OPS[hook] = _LOOKUPS[hook]()
    return _OPS[hook]


def ops_available() -> bool:
    """Whether this build carries all four ops (the sm_107 TensorRT-LLM builds)."""
    return all(_op(hook) is not None for hook in OP_NAMES)


def step_prologue(window: torch.Tensor, epoch: Optional[torch.Tensor], max_desc: int) -> None:
    """``window`` = the pack's device window (an int64 header [host copies, device copies], then one (dst, src, bytes)
    descriptor per copy, host copies first). Writes up to ``max_desc`` copies of each kind, host before device, then
    zeroes the header; with ``epoch`` increments ``epoch[0]`` once (trtllm::minimax_m3_sm107_tp8_step_prologue)."""
    _op("prologue")(window, epoch, max_desc)
    _engaged("prologue", "step-input prologue: one native launch (pack_prologue_op)")


def spec_offsets(
    pos: torch.Tensor, pos_off: torch.Tensor, kv: torch.Tensor, kv_off: torch.Tensor
) -> bool:
    """``pos += pos_off`` and ``kv += kv_off`` (int32 contiguous 1-D views, equal lengths per pair) in one launch
    (trtllm::minimax_m3_sm107_tp8_spec_offsets): the same wrapping int32 adds the two ATen ops do."""
    ts = (pos, pos_off, kv, kv_off)
    if (
        any(t.dtype != torch.int32 or t.dim() != 1 or t.stride(0) != 1 or not t.is_cuda for t in ts)
        or pos.numel() != pos_off.numel()
        or kv.numel() != kv_off.numel()
    ):
        return False
    _op("offsets")(pos, pos_off, kv, kv_off)
    _engaged("offsets", "spec step correction: one native launch (spec_offsets_op)")
    return True


def msa_kv_update(
    kv_lens: torch.Tensor,
    kv_lens_staged: torch.Tensor,
    seq_lens: torch.Tensor,
    q_batch_row: torch.Tensor,
    q_intra: torch.Tensor,
    block_table: torch.Tensor,
    tokens_per_block: int,
    batch: int,
    total_q: int,
    msa_seq_lens: torch.Tensor,
    cu_kv_lens: Optional[torch.Tensor],
    out_cache_loc: torch.Tensor,
    n_valid_blocks: Optional[torch.Tensor],
) -> bool:
    """``MsaKvUpdate`` (mv2_minimax_m3::minimax_m3_msa_kv_update; int32): msa_seq_lens[b] = min(kv_lens[b], staged[b]),
    cu_kv_lens its exclusive prefix sum, out_cache_loc / n_valid_blocks of each query token through block_table, as
    ``MiniMaxM3MsaSparseAttentionMetadata.on_update_kv_lens`` derives them, at any page size (the MSA pool's is 128).
    The request index of each token and its block-table column are not bounds-checked: the metadata's staging keeps
    them in range. False for other operands (the TP8 target's checks)."""
    ts = [
        kv_lens,
        kv_lens_staged,
        seq_lens,
        q_batch_row,
        q_intra,
        block_table,
        msa_seq_lens,
        out_cache_loc,
    ]
    ts += [t for t in (cu_kv_lens, n_valid_blocks) if t is not None]
    if (
        any(t.dtype != torch.int32 or not t.is_cuda for t in ts)
        or block_table.dim() != 2
        or block_table.stride(1) != 1
    ):
        return False
    if any(not t.is_contiguous() for t in ts if t is not block_table):
        return False
    if (
        kv_lens.numel() < batch
        or kv_lens_staged.numel() < batch
        or seq_lens.numel() < batch
        or msa_seq_lens.numel() < batch
        or q_batch_row.numel() < total_q
        or q_intra.numel() < total_q
        or out_cache_loc.numel() < total_q
        or (cu_kv_lens is not None and cu_kv_lens.numel() < batch + 1)
        or (n_valid_blocks is not None and n_valid_blocks.numel() < total_q)
    ):
        return False
    _op("kv_update")(
        kv_lens,
        kv_lens_staged,
        seq_lens,
        q_batch_row,
        q_intra,
        block_table,
        int(tokens_per_block),
        int(batch),
        int(total_q),
        msa_seq_lens,
        cu_kv_lens,
        out_cache_loc,
        n_valid_blocks,
    )
    _engaged("kv_update", "MSA KV-length correction: one native launch (msa kv_update)")
    return True


def step_gather(
    new_tok, next_draft, lens, slots, ppi, P, T, D, ids, ids0, draft, d0, pos_off, kv_off, b0
) -> bool:
    """The engine's previous-batch gathers in one launch (trtllm::minimax_m3_step_gather; int32):
    ids[ids0 + p*T + t] = new_tok[t, slot_p]; draft[d0 + p*D + d] = next_draft[slot_p, d]; pos_off[i] = lens[ppi[i -
    b0*T]] inside the previous rows' tokens, else 0; kv_off[i] = lens[slot_{i-b0}] - T inside the previous rows, else 0
    (every element of both offset buffers is written). False for other operands."""
    ts = (new_tok, next_draft, lens, slots, ppi, ids, draft, pos_off, kv_off)
    if (
        any(t.dtype != torch.int32 or not t.is_cuda for t in ts)
        or next_draft.dim() != 2
        or new_tok.dim() < 2
    ):
        return False
    if (new_tok.dim() == 3 and new_tok.shape[2] != 1) or not (
        lens.is_contiguous()
        and ids.is_contiguous()
        and draft.is_contiguous()
        and pos_off.is_contiguous()
        and kv_off.is_contiguous()
    ):
        return False
    _op("gather")(
        new_tok, next_draft, lens, slots, ppi, P, T, D, ids, ids0, draft, d0, pos_off, kv_off, b0
    )
    return True


class M3DSparkStepFold:
    """``RuntimeHooks.spec_step_fold`` for the in-tree DSpark worker (the ``SpecStepFold`` protocol): only the engine's
    previous-batch gathers fold, into one ``step_gather`` launch. The DSpark forward never stages the next step's
    inputs and the sampler keeps its own slot stores, so ``begin_step`` reports nothing staged (no device arming, no
    upload), ``sampler_rows_stored`` is False and ``scatter`` declines. ``taken`` / ``declined`` count the gathers."""

    #: Read by a target shell that stores in its forward; the DSpark forward does not.
    in_forward = False

    def __init__(self) -> None:
        self.store = None
        self.step_inputs: Optional[tuple] = None
        self.taken = 0
        self.declined = 0

    def bind_engine_buffers(
        self, input_ids, draft_tokens, position_id_offsets, kv_lens_offsets, max_batch_size
    ) -> None:
        self.step_inputs = (input_ids, draft_tokens, position_id_offsets, kv_lens_offsets)

    def bind_sampler_store(self, store: object) -> None:
        self.store = store

    def begin_step(self, seq_slots, num_rows: int) -> bool:
        return False

    def gather(
        self,
        new_tokens,
        next_draft_tokens,
        new_tokens_lens,
        previous_slots,
        previous_pos_indices,
        previous_batch_len: int,
        tokens_per_gen_step: int,
        draft_token_buffer_width: int,
        num_tokens: int,
        num_draft_tokens: int,
        first_previous_row: int,
    ) -> bool:
        if self.step_inputs is None:
            return False
        input_ids, draft_tokens, position_id_offsets, kv_lens_offsets = self.step_inputs
        if not step_gather(
            new_tokens,
            next_draft_tokens,
            new_tokens_lens,
            previous_slots,
            previous_pos_indices,
            previous_batch_len,
            tokens_per_gen_step,
            draft_token_buffer_width,
            input_ids,
            num_tokens,
            draft_tokens,
            num_draft_tokens,
            position_id_offsets,
            kv_lens_offsets,
            first_previous_row,
        ):
            self.declined += 1
            return False
        self.taken += 1
        _engaged(
            "gather", "DSpark step-input gather: one native launch (spec_step_fold step_gather)"
        )
        return True

    def sampler_rows_stored(self, outputs: dict, num_skip: int, seq_slots: list) -> bool:
        return False

    def scatter(
        self, seq_slots, new_tokens, next_new_tokens, new_tokens_lens, next_draft_tokens
    ) -> bool:
        return False
