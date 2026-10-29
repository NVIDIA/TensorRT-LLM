# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""One MTP draft step after the vocabulary-sharded draft LM head, in one kernel, via glm_draft_sample_glue."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*
from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

from ._draft_step_state import (
    assert_state_args_valid,
    eh_proj_input_window,
    global_argmax,
    next_step_state,
)


def _step(**overrides) -> dict:
    spec = dict(
        rows=1,
        draft_len=7,
        step=6,
        vocab_shard=9680,
        logits="random",
        glue=False,
        positions=None,
        fill=False,
        kv_update=1,
        dsa=True,
        next_new=False,
    )
    spec.update(overrides)
    return spec


class GlmDraftSampleGlue(OpWrapper):
    """The greedy draft token of every row, and optionally the next step's inputs and state, in one CTA.

    Collective over the TP group (`group_size` ranks, the workspace's group): each rank takes the argmax of its
    logits shard `[rows, vocab_shard]` (rank r holds tokens `[r * vocab_shard, (r + 1) * vocab_shard)`), the ranks
    exchange the (index, value) pairs through the MNNVL workspace `exchange_buffer` / `exchange_flags`, and every
    rank writes the global argmax, in torch.argmax's order, into column `draft_step` of `draft_tokens`. Then, each
    part skipped when its output is None:
    - `glue_out`: this rank's window `[col_lo, col_hi)` of `cat([enorm(embed_table[token]), hnorm(hidden)], -1)`,
      the next step's eh_proj input (FlashInfer's CuTe RMSNorm arithmetic), contiguous;
    - `positions_out[r] = positions_in[gather_ids[r] if gather_ids is not None else r] + 1`;
    - `seq_lens` filled with 1 when `fill_seq_lens`; `kv_lens` advanced as `kv_update` says (0 none, 1 `+ 1`,
      2 `- (draft_len - rewind_accepted)`);
    - the DSA decode metadata of the next step from those lengths (`req_idx_per_token`, the two slot mappings, the
      two generation indptrs; `dsa_layout` = [tokens, token lanes per request, tokens per block, data bytes per
      token, scale bytes per token, bytes per block, scale offset in a block]);
    - `next_new_tokens[r] = [accepted_tokens[batch_indices[r], num_accepted[r] - 1], draft tokens...]`, the
      output rows after the last draft step.

    Under PDL the kernel reads only the norm weights before its grid-dependency wait and triggers its dependents
    right after it. The workspace serves this op only (each call empties the buffer of the previous one).
    """

    ARCHS = frozenset({Arch.SM_107})

    # spec: rows (requests), draft_len (K), step, vocab_shard (the per-rank LM-head shard: 154880 / 16 at tp16),
    # logits ("random", or "tied": every rank's maximum equal), glue (the eh_proj input window, this rank's
    # 2 * 6144 / tp columns), positions (None, "gather" or "plain"), fill (seq_lens), kv_update, dsa (metadata),
    # next_new (output rows). At c1 the token table serves draft steps 0-5 and this op the last one (step 6):
    # token, KV + 1, DSA metadata and the next-new-tokens rows. Above two requests per rank (no token table) it
    # serves every step, with the eh_proj input window.
    CELLS: tuple[Cell, ...] = (
        Cell(
            why="c1 last draft step: token, KV + 1, DSA metadata, next-new-tokens rows",
            spec=_step(next_new=True),
        ),
        Cell(
            why="c1 last draft step, every rank's maximum equal: rank 0's index wins",
            spec=_step(logits="tied", next_new=True),
        ),
        Cell(
            why="step 0 without the token table: window, gathered positions, fill, rewind",
            spec=_step(rows=4, step=0, glue=True, positions="gather", fill=True, kv_update=2),
        ),
        Cell(
            why="a middle step without the token table: window, positions + 1, KV + 1",
            spec=_step(rows=4, step=3, glue=True, positions="plain"),
        ),
        Cell(
            why="the most rows one launch serves (8 requests), last step",
            spec=_step(rows=8, next_new=True),
        ),
    )

    note = """
    The workspace comes from `DraftExchangeWorkspace` (one per op, built outside capture). Ranks must call in
    lockstep with the same `rows`; a missing rank leaves the others polling.

    The vocabulary shards must be equal and in rank order, which an unpadded vocabulary-parallel LM head gives
    (154880 = 16 x 9680).
    """

    def __call__(
        self,
        logits: torch.Tensor,
        exchange_buffer: torch.Tensor,
        exchange_flags: torch.Tensor,
        group_size: int,
        draft_tokens: torch.Tensor,
        draft_step: int,
        embed_table: Optional[torch.Tensor],
        hidden: Optional[torch.Tensor],
        enorm_weight: Optional[torch.Tensor],
        hnorm_weight: Optional[torch.Tensor],
        eps: float,
        col_lo: int,
        col_hi: int,
        glue_out: Optional[torch.Tensor],
        positions_in: Optional[torch.Tensor],
        positions_out: Optional[torch.Tensor],
        gather_ids: Optional[torch.Tensor],
        seq_lens: Optional[torch.Tensor],
        fill_seq_lens: bool,
        kv_lens: Optional[torch.Tensor],
        kv_update: int,
        rewind_accepted: Optional[torch.Tensor],
        draft_len: int,
        block_offsets: Optional[torch.Tensor],
        dsa_layout: List[int],
        req_idx_per_token: Optional[torch.Tensor],
        slot_mapping_fp8: Optional[torch.Tensor],
        slot_mapping_scale: Optional[torch.Tensor],
        gen_kv_indptr: Optional[torch.Tensor],
        gen_cached_token_indptr: Optional[torch.Tensor],
        accepted_tokens: Optional[torch.Tensor],
        batch_indices: Optional[torch.Tensor],
        num_accepted: Optional[torch.Tensor],
        next_new_tokens: Optional[torch.Tensor],
    ) -> None:
        torch.ops.trtllm.glm_draft_sample_glue(
            logits,
            exchange_buffer,
            exchange_flags,
            group_size,
            draft_tokens,
            draft_step,
            embed_table,
            hidden,
            enorm_weight,
            hnorm_weight,
            eps,
            col_lo,
            col_hi,
            glue_out,
            positions_in,
            positions_out,
            gather_ids,
            seq_lens,
            fill_seq_lens,
            kv_lens,
            kv_update,
            rewind_accepted,
            draft_len,
            block_offsets,
            dsa_layout,
            req_idx_per_token,
            slot_mapping_fp8,
            slot_mapping_scale,
            gen_kv_indptr,
            gen_cached_token_indptr,
            accepted_tokens,
            batch_indices,
            num_accepted,
            next_new_tokens,
        )

    def reference(
        self,
        logits: Sequence[torch.Tensor],
        exchange_buffer: torch.Tensor,
        exchange_flags: torch.Tensor,
        group_size: int,
        draft_tokens: torch.Tensor,
        draft_step: int,
        embed_table: Optional[torch.Tensor],
        hidden: Optional[torch.Tensor],
        enorm_weight: Optional[torch.Tensor],
        hnorm_weight: Optional[torch.Tensor],
        eps: float,
        col_lo: int,
        col_hi: int,
        glue_out: Optional[torch.Tensor],
        positions_in: Optional[torch.Tensor],
        positions_out: Optional[torch.Tensor],
        gather_ids: Optional[torch.Tensor],
        seq_lens: Optional[torch.Tensor],
        fill_seq_lens: bool,
        kv_lens: Optional[torch.Tensor],
        kv_update: int,
        rewind_accepted: Optional[torch.Tensor],
        draft_len: int,
        block_offsets: Optional[torch.Tensor],
        dsa_layout: List[int],
        req_idx_per_token: Optional[torch.Tensor],
        slot_mapping_fp8: Optional[torch.Tensor],
        slot_mapping_scale: Optional[torch.Tensor],
        gen_kv_indptr: Optional[torch.Tensor],
        gen_cached_token_indptr: Optional[torch.Tensor],
        accepted_tokens: Optional[torch.Tensor],
        batch_indices: Optional[torch.Tensor],
        num_accepted: Optional[torch.Tensor],
        next_new_tokens: Optional[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """What the op leaves in each buffer it writes, by argument name.

        The documented exception to mirroring `__call__`: the token is a function of every rank's logits, so
        `logits` is the group's shards in rank order. Every other argument is the calling rank's, and the buffers
        the op updates in place are read as they were before the call. The exchange workspace is not consulted.
        """
        rows = logits[0].shape[0]
        tokens = global_argmax(logits)
        expected_draft = draft_tokens.clone()
        expected_draft[:rows, draft_step] = tokens
        out = {"draft_tokens": expected_draft}
        if glue_out is not None:
            out["glue_out"] = eh_proj_input_window(
                tokens, embed_table, hidden, enorm_weight, hnorm_weight, eps, col_lo, col_hi
            )
        out.update(
            next_step_state(
                rows,
                positions_in,
                positions_out,
                gather_ids,
                seq_lens,
                fill_seq_lens,
                kv_lens,
                kv_update,
                rewind_accepted,
                draft_len,
                block_offsets,
                dsa_layout,
                req_idx_per_token,
                slot_mapping_fp8,
                slot_mapping_scale,
                gen_kv_indptr,
                gen_cached_token_indptr,
            )
        )
        if next_new_tokens is not None:
            expected = next_new_tokens.clone().view(rows, -1)
            last = accepted_tokens[batch_indices[:rows].long(), num_accepted[:rows].long() - 1]
            expected[:, 0] = last
            expected[:, 1:] = expected_draft[:rows]
            out["next_new_tokens"] = expected.view_as(next_new_tokens)
        return out

    def is_valid(
        self,
        logits: torch.Tensor,
        exchange_buffer: torch.Tensor,
        exchange_flags: torch.Tensor,
        group_size: int,
        draft_tokens: torch.Tensor,
        draft_step: int,
        embed_table: Optional[torch.Tensor],
        hidden: Optional[torch.Tensor],
        enorm_weight: Optional[torch.Tensor],
        hnorm_weight: Optional[torch.Tensor],
        eps: float,
        col_lo: int,
        col_hi: int,
        glue_out: Optional[torch.Tensor],
        positions_in: Optional[torch.Tensor],
        positions_out: Optional[torch.Tensor],
        gather_ids: Optional[torch.Tensor],
        seq_lens: Optional[torch.Tensor],
        fill_seq_lens: bool,
        kv_lens: Optional[torch.Tensor],
        kv_update: int,
        rewind_accepted: Optional[torch.Tensor],
        draft_len: int,
        block_offsets: Optional[torch.Tensor],
        dsa_layout: List[int],
        req_idx_per_token: Optional[torch.Tensor],
        slot_mapping_fp8: Optional[torch.Tensor],
        slot_mapping_scale: Optional[torch.Tensor],
        gen_kv_indptr: Optional[torch.Tensor],
        gen_cached_token_indptr: Optional[torch.Tensor],
        accepted_tokens: Optional[torch.Tensor],
        batch_indices: Optional[torch.Tensor],
        num_accepted: Optional[torch.Tensor],
        next_new_tokens: Optional[torch.Tensor],
    ) -> None:
        rows = logits.shape[0]
        assert_state_args_valid(
            rows,
            draft_tokens,
            positions_in,
            positions_out,
            gather_ids,
            kv_update,
            draft_len,
            dsa_layout,
            req_idx_per_token,
        )
        if next_new_tokens is not None:
            # The kernel writes row r at offset r * (draft_len + 1) and reads
            # accepted_tokens[batch_indices[r], num_accepted[r] - 1] unbounded.
            assert tuple(next_new_tokens.shape) == (rows, draft_tokens.shape[1] + 1), (
                "next_new_tokens must be exactly [rows, draft_len + 1]"
            )
            index = batch_indices[:rows].long()
            count = num_accepted[:rows].long()
            assert bool(((index >= 0) & (index < accepted_tokens.shape[0])).all()), (
                "batch_indices must index accepted_tokens"
            )
            assert bool(((count >= 1) & (count <= accepted_tokens.shape[1])).all()), (
                "num_accepted must lie in [1, accepted_tokens.shape[1]]"
            )


glm_draft_sample_glue = GlmDraftSampleGlue()
