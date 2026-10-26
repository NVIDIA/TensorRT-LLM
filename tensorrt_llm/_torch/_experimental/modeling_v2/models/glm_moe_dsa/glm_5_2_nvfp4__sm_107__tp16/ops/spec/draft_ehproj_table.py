# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""An MTP draft step that also produces the next step's eh_proj output, via glm_draft_ehproj_table."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*
from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

from ._draft_step_state import assert_state_args_valid, global_argmax, next_step_state


def _step(**overrides) -> dict:
    spec = dict(
        rows=1,
        draft_len=7,
        step=1,
        vocab_shard=9680,
        logits="random",
        positions="plain",
        fill=False,
        kv_update=1,
        dsa=True,
    )
    spec.update(overrides)
    return spec


class GlmDraftEhprojTable(OpWrapper):
    """The greedy draft token of every row and the next step's eh_proj output with its all-reduce and norm.

    eh_proj is row-parallel over `cat([enorm(embed(t)), hnorm(h)], -1)`, so its output splits into
    `W[:, :H] @ enorm(embed(t))`, which depends on the token only, and `W[:, H:] @ hnorm(h)`, which is known before
    the draft LM head. `token_table` holds the first half for the tokens of this rank's LM-head shard
    (`[vocab_shard, H]` BF16, one rounding of the FP32 product); `hidden_half` is this rank's FP32 share
    `[rows, H / group_size]` of the second half (columns `[rank * H / tp, (rank + 1) * H / tp)`).

    Collective over the TP group, in one 8-CTA cluster per rank: each rank finds its logits shard's argmax and
    publishes, through the MNNVL workspace `exchange_buffer` / `exchange_flags`, the (index, value) packet, the
    token-table row of its candidate and its hidden-half share. Every rank then takes the global argmax in
    torch.argmax's order into column `draft_step` of `draft_tokens`, writes
    `residual_out = x = bf16(token_table_of_winner[token] + hidden_half)` (one FP32 add) and
    `normed_out = bf16((x * rsqrt(mean(x^2) + eps)) * norm_weight)`: what eh_proj with its all-reduce,
    input_layernorm and a zero residual returns. The next step's position ids, lengths and DSA decode metadata are
    written as `glm_draft_sample_glue` writes them.

    Under PDL the kernel reads only the norm weight before its grid-dependency wait and triggers its dependents
    right after it; the hidden-half GEMM on the other stream must be joined by an event before the launch.
    """

    ARCHS = frozenset({Arch.SM_107})

    # spec: rows (requests: 1 at c1, at most 2 where the target builds the table), draft_len (K), step,
    # vocab_shard (154880 / 16 at tp16), logits ("random" or "tied"), positions ("gather" on step 0, else
    # "plain"), fill, kv_update, dsa. At c1 the op serves draft steps 0-5: step 0 turns the 8 verified rows into
    # one (gathered positions, seq_lens filled with 1, KV rewound by the rejected drafts), steps 1-5 advance by one.
    CELLS: tuple[Cell, ...] = (
        Cell(
            why="c1 draft step 0: gathered positions, seq_lens fill, KV rewind, DSA metadata",
            spec=_step(step=0, positions="gather", fill=True, kv_update=2),
        ),
        Cell(
            why="c1 draft steps 1-5: positions + 1, KV + 1, DSA metadata",
            spec=_step(step=3),
        ),
        Cell(
            why="c1 middle step, every rank's maximum equal: rank 0's row wins",
            spec=_step(step=2, logits="tied"),
        ),
        Cell(
            why="two requests, the most rows the target builds the table for",
            spec=_step(rows=2, step=1),
        ),
    )

    note = """
    Serves 2 to 16 ranks whose shares of the 6144 hidden columns are whole 8-column vectors; at tp16 a CTA's 768
    columns span two ranks' 384-column shares, a mapping a 4-rank run does not reach. The workspace comes from
    `DraftExchangeWorkspace` (one per op, built outside capture).

    `normed_out` differs from eh_proj + all-reduce + RMSNorm in the order of the norm's sum of squares (CTA
    slices in order); `residual_out` differs from eh_proj's tp partial sums rounded to BF16 before their reduction:
    here only the token half is rounded before x. Both feed the draft only.
    """

    def __call__(
        self,
        logits: torch.Tensor,
        exchange_buffer: torch.Tensor,
        exchange_flags: torch.Tensor,
        group_size: int,
        draft_tokens: torch.Tensor,
        draft_step: int,
        hidden_half: torch.Tensor,
        token_table: torch.Tensor,
        norm_weight: torch.Tensor,
        eps: float,
        normed_out: torch.Tensor,
        residual_out: torch.Tensor,
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
    ) -> None:
        torch.ops.trtllm.glm_draft_ehproj_table(
            logits,
            exchange_buffer,
            exchange_flags,
            group_size,
            draft_tokens,
            draft_step,
            hidden_half,
            token_table,
            norm_weight,
            eps,
            normed_out,
            residual_out,
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

    def reference(
        self,
        logits: Sequence[torch.Tensor],
        exchange_buffer: torch.Tensor,
        exchange_flags: torch.Tensor,
        group_size: int,
        draft_tokens: torch.Tensor,
        draft_step: int,
        hidden_half: Sequence[torch.Tensor],
        token_table: Sequence[torch.Tensor],
        norm_weight: torch.Tensor,
        eps: float,
        normed_out: torch.Tensor,
        residual_out: torch.Tensor,
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
    ) -> Dict[str, torch.Tensor]:
        """What the op leaves in each buffer it writes, by argument name.

        The documented exception to mirroring `__call__`: `logits`, `hidden_half` and `token_table` are the
        group's, in rank order, because every rank's output depends on all of them. Every other argument is the
        calling rank's, read as it was before the call. The exchange workspace is not consulted.
        """
        rows = logits[0].shape[0]
        vocab_shard = logits[0].shape[1]
        tokens = global_argmax(logits)
        expected_draft = draft_tokens.clone()
        expected_draft[:rows, draft_step] = tokens
        token_half = torch.stack(
            [
                token_table[int(t) // vocab_shard][int(t) % vocab_shard].float()
                for t in tokens.tolist()
            ]
        )
        x = (token_half + torch.cat([h.float() for h in hidden_half], dim=-1)).to(torch.bfloat16)
        rstd = torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + eps)
        normed = ((x.float() * rstd) * norm_weight.float()).to(torch.bfloat16)
        out = {"draft_tokens": expected_draft, "residual_out": x, "normed_out": normed}
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
        return out

    def is_valid(
        self,
        logits: torch.Tensor,
        exchange_buffer: torch.Tensor,
        exchange_flags: torch.Tensor,
        group_size: int,
        draft_tokens: torch.Tensor,
        draft_step: int,
        hidden_half: torch.Tensor,
        token_table: torch.Tensor,
        norm_weight: torch.Tensor,
        eps: float,
        normed_out: torch.Tensor,
        residual_out: torch.Tensor,
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
    ) -> None:
        assert_state_args_valid(
            logits.shape[0],
            draft_tokens,
            positions_in,
            positions_out,
            gather_ids,
            kv_update,
            draft_len,
            dsa_layout,
            req_idx_per_token,
        )

        # x is stored row by row while the cluster still reads the token table for the later rows, and the norm
        # after the last one: an output overlapping an input, or the other output, replaces values the kernel has
        # yet to read or to return.
        def overlaps(a: torch.Tensor, b: torch.Tensor) -> bool:
            return a.data_ptr() < b.data_ptr() + b.nbytes and b.data_ptr() < a.data_ptr() + a.nbytes

        for name, t in (("normed_out", normed_out), ("residual_out", residual_out)):
            assert not any(overlaps(t, other) for other in (logits, hidden_half, token_table)), (
                f"{name} must not overlap the op's inputs"
            )
        assert not overlaps(normed_out, residual_out), (
            "normed_out and residual_out must be separate buffers"
        )


glm_draft_ehproj_table = GlmDraftEhprojTable()
