# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The DSpark vanilla-Markov draft chain over a vocab-sharded draft head, with the TP-gathered greedy draft tokens,
the next step's input tokens and the folded KV-length rewind, in one collective kernel over the TP group's MNNVL
workspace (``markov_workspace``)."""

from typing import Optional, Sequence, Tuple

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.kimi_k3_mxfp4__sm_100__tp16_moetp4ep4.kernels.k3_markov import (  # noqa: F401, E501 — registers the op
    op as _k3_markov_op,
)


class K3Markov(OpWrapper):
    """Return ``(corrected, tokens, next_new)`` for this rank's block logits ``base`` [B, K, S] (bf16 or fp32), the
    rows ``shard_offset .. shard_offset + S`` of the draft vocabulary: position k adds the bias
    ``bf16(markov_w2_shard . markov_w1[prev_k])`` to ``base[:, k]`` (fp32 ``corrected`` [B, K, S]), where prev_0 is
    ``first_prev`` and prev_k (k > 0) is the previous position's token; ``tokens`` [B, K] int32 is each position's
    first maximum over the whole vocabulary (every rank's shard; NaN never wins); ``next_new`` [B, K + 1] int32 is
    ``[accepted[accepted_rows[b], num_accepted[b] - 1], tokens[b, :]]``. With ``kv_lens`` and ``rewind`` it also sets
    ``kv_lens[rewind_first + b] = max(kv_lens[rewind_first + b] - rewind[b], 0)`` in place. Every rank of the
    workspace's TP group makes the same calls in the same order, with the same replicated arguments."""

    ARCHS = frozenset({Arch.SM_100, Arch.SM_103})
    CELLS: tuple[Cell, ...] = (
        Cell(
            why="one decode request, DSpark block of 7, bf16 head shard of TP16 (S 10240, the exchange of 16 ranks)",
            spec=dict(
                batch=1, block=7, shard="tp16", dtype=torch.bfloat16, rewind=False, contexts=0
            ),
        ),
        Cell(
            why="four generation requests after one context request: num_accepted / accepted_rows as [1:] slices "
            "and the rewind from rewind_first 1, as the worker passes them in a mixed step",
            spec=dict(
                batch=4, block=7, shard="tp16", dtype=torch.bfloat16, rewind=True, contexts=1
            ),
        ),
        Cell(
            why="eight requests (the target's decode maximum) with the folded KV-length rewind (clamped, negative "
            "and exact-to-zero amounts)",
            spec=dict(
                batch=8, block=7, shard="tp16", dtype=torch.bfloat16, rewind=True, contexts=0
            ),
        ),
        Cell(
            why="eight requests on fp32 base logits, two first anchors outside the vocabulary (zero bias)",
            spec=dict(
                batch=8,
                block=7,
                shard="tp16",
                dtype=torch.float32,
                rewind=False,
                contexts=0,
                anchors=True,
            ),
        ),
        Cell(
            why="eight requests with exact ties within a shard and across ranks: the lowest vocabulary index wins",
            spec=dict(
                batch=8,
                block=7,
                shard="tp16",
                dtype=torch.bfloat16,
                rewind=False,
                contexts=0,
                ties=True,
            ),
        ),
        Cell(
            why="shorter block of 3 positions, four requests",
            spec=dict(
                batch=4, block=3, shard="tp16", dtype=torch.bfloat16, rewind=True, contexts=0
            ),
        ),
        Cell(
            why="the whole vocabulary over the group (S 163840 / W, one slot per rank), eight requests",
            spec=dict(
                batch=8, block=7, shard="full", dtype=torch.bfloat16, rewind=True, contexts=0
            ),
        ),
    )
    note = """
    workspace is markov_workspace(mapping, push_copies) of the kernel's op
    module: Lamport buffers behind one multicast mapping per (mapping,
    push_copies), allocated collectively on first use, outside CUDA-graph
    capture. Every rank of the TP group must make the same calls in the same
    order on it, with the same B, K, S and the same replicated arguments
    (first_prev, markov_w1, accepted, num_accepted, accepted_rows, rewind);
    nothing detects a rank that diverges, and a different B, K or S can hang.
    shard_offset must be rank * S (the shards tile the vocabulary in rank
    order). push_copies > 1 only emulates a larger group's exchange in tests.

    Each (grid, rows, S, K, B, slots, push_copies, dtype, PDL) compiles on
    its first call, which must be eager; under capture it raises. Every CTA
    spins on its peers' entries, so the whole grid (at most 152 CTAs and
    the SM count) must be resident: no concurrent kernel may hold SMs, and
    no MPS or green-context SM cap may sit below the grid.

    The bias is bit-exact with the C++ dspark_markov_chain: per lane an FMA
    chain over 8 columns from +0, the 32 lane sums paired at distances 16,
    8, 4, 2, 1, one bf16 rounding, added to the fp32 base. reference states
    that arithmetic, and the matrix compares corrected (fp32 bits), tokens,
    next_new and kv_lens bit for bit on every rank. The built-in DSpark
    drafter's F.linear chain (``dspark_markov_step_bias``) sums in another
    order; the kernel test bounds the difference by one bf16 ulp of the bias
    plus the fp32 accumulation error.

    Out-of-range accepted_rows or num_accepted read outside accepted; is_valid
    rejects them (num_accepted 0 wraps to the last column, as torch indexing).
    """

    def __call__(
        self,
        base: torch.Tensor,
        first_prev: torch.Tensor,
        markov_w1: torch.Tensor,
        markov_w2_shard: torch.Tensor,
        shard_offset: int,
        accepted: torch.Tensor,
        num_accepted: torch.Tensor,
        accepted_rows: torch.Tensor,
        workspace: dict,
        kv_lens: Optional[torch.Tensor] = None,
        rewind: Optional[torch.Tensor] = None,
        rewind_first: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if kv_lens is None or rewind is None:
            kv_lens, rewind = workspace["no_rewind"], workspace["flags"][:0]
        return torch.ops.trtllm.k3_markov(
            base,
            first_prev,
            markov_w1,
            markov_w2_shard,
            int(shard_offset),
            workspace["uc"],
            workspace["mc"],
            workspace["flags"],
            workspace["rank"],
            workspace["slots"],
            workspace["push_copies"],
            workspace["buf_words"],
            accepted,
            num_accepted,
            accepted_rows,
            kv_lens,
            rewind,
            int(rewind_first),
        )

    def reference(
        self,
        bases_in_rank_order: Sequence[torch.Tensor],
        first_prev: torch.Tensor,
        markov_w1: torch.Tensor,
        markov_w2_shards_in_rank_order: Sequence[torch.Tensor],
        accepted: torch.Tensor,
        num_accepted: torch.Tensor,
        accepted_rows: torch.Tensor,
        kv_lens: Optional[torch.Tensor] = None,
        rewind: Optional[torch.Tensor] = None,
        rewind_first: int = 0,
    ) -> Tuple[list, torch.Tensor, torch.Tensor]:
        """A collective: takes every rank's ``base`` and ``markov_w2_shard`` in rank order (rank r's shard starts at
        the sum of the earlier shards' S, its ``shard_offset``) and the replicated arguments, and returns
        ``(corrected per rank in rank order, tokens, next_new)``; applies the rewind to ``kv_lens`` in place.
        ``workspace`` and ``shard_offset`` are implied by the rank order and not taken.

        The bias is the C++ ``dspark_markov_chain`` arithmetic: per lane l a chain over columns 8 l .. 8 l + 7 from +0
        (the products of bf16 values are exact in fp32, so fp32 adds equal FMAs), the 32 lane sums added in pairs at
        lane distances 16, 8, 4, 2, 1, the sum rounded to bf16 and added to the fp32 base; an anchor outside
        [0, V) adds zero. The token is the TP-gathered greedy sampler's: each rank's first maximum (NaN as -inf),
        then the first maximum over the ranks."""
        vocab = markov_w1.shape[0]
        batch, block = bases_in_rank_order[0].shape[:2]
        offsets, start = [], 0
        for w2s in markov_w2_shards_in_rank_order:
            offsets.append(start)
            start += w2s.shape[0]
        prev = first_prev.long()
        corrected = [[] for _ in bases_in_rank_order]
        tokens = []
        for k in range(block):
            valid = (prev >= 0) & (prev < vocab)
            w1r = markov_w1[prev.clamp(0, vocab - 1)].float().view(batch, 1, 32, 8)
            values, indices = [], []
            for r, (base, w2s) in enumerate(
                zip(bases_in_rank_order, markov_w2_shards_in_rank_order)
            ):
                shard = w2s.shape[0]
                products = w2s.float().view(1, shard, 32, 8) * w1r
                lanes = torch.zeros(batch, shard, 32, dtype=torch.float32, device=base.device)
                for j in range(8):
                    lanes = lanes + products[..., j]
                for half in (16, 8, 4, 2, 1):
                    lanes = lanes[..., :half] + lanes[..., half : 2 * half]
                bias = lanes[..., 0].bfloat16().float()
                bias = torch.where(valid[:, None], bias, torch.zeros_like(bias))
                step = base[:, k].float() + bias
                corrected[r].append(step)
                value, index = torch.max(step.masked_fill(step.isnan(), float("-inf")), dim=-1)
                values.append(value)
                indices.append(index + offsets[r])
            best = torch.argmax(torch.stack(values, dim=-1), dim=-1, keepdim=True)
            token = torch.gather(torch.stack(indices, dim=-1), -1, best).squeeze(-1)
            tokens.append(token.to(torch.int32))
            prev = token.long()
        tokens = torch.stack(tokens, dim=1)
        first = accepted[accepted_rows[:batch].long(), num_accepted[:batch].long() - 1].unsqueeze(1)
        next_new = torch.cat([first.to(torch.int32), tokens], dim=1)
        if kv_lens is not None and rewind is not None and rewind.numel() > 0:
            rows = slice(rewind_first, rewind_first + rewind.numel())
            kv_lens[rows] = (kv_lens[rows] - rewind).clamp_min(0)
        return [torch.stack(c, dim=1) for c in corrected], tokens, next_new

    def is_valid(
        self,
        base: torch.Tensor,
        first_prev: torch.Tensor,
        markov_w1: torch.Tensor,
        markov_w2_shard: torch.Tensor,
        shard_offset: int,
        accepted: torch.Tensor,
        num_accepted: torch.Tensor,
        accepted_rows: torch.Tensor,
        workspace: dict,
        kv_lens: Optional[torch.Tensor] = None,
        rewind: Optional[torch.Tensor] = None,
        rewind_first: int = 0,
    ) -> None:
        batch, _, shard = base.shape
        if not 0 <= shard_offset <= markov_w1.shape[0] - shard:
            raise ValueError(
                f"k3_markov: shard rows {shard_offset}..{shard_offset + shard} leave the vocabulary "
                f"({markov_w1.shape[0]}): those tokens would anchor the next position with a zero bias"
            )
        rows, width = accepted.shape
        picked_rows = accepted_rows[:batch]
        picked_counts = num_accepted[:batch]
        if bool(((picked_rows < 0) | (picked_rows >= rows)).any()):
            raise ValueError(f"k3_markov: accepted_rows must index accepted's {rows} rows")
        if bool(((picked_counts < 0) | (picked_counts > width)).any()):
            raise ValueError(f"k3_markov: num_accepted must lie in [0, {width}] (accepted's width)")


k3_markov = K3Markov()
