# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The DSpark draft block's fused glue on SM107: its attention over the paged context window
plus the query block, the latent and query rope tails, and the recurrent Markov head's
per-step input, gate and biased-logit argmax. Each entry's `reference` is the torch chain the
target ran before it."""

from __future__ import annotations

import torch
import triton

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

from . import _kernels

#: The draft latent: 512 content dims and the 64-wide rope key, the MLA latent of the target.
LAT = 512
PE = 64
#: Query rows per CTA of the attention: the smallest row count `tl.dot` takes.
_ATTN_ROWS = 16
#: Key tiles of 64. The 2048-key window splits 32 ways (one tile each) for up to 4 sequences and
#: 8 ways (four tiles each) above, plus one split for the query block's own keys; every split
#: writes fp32 partials the combine merges. Measured on VR200 at 1/4/8/16 sequences: 10.1,
#: 12.7, 15.8 and 21.7 us; 16 splits throughout took 15.9, 16.4, 25.8 and 61.3 us.
_ATTN_BLOCK_T = 64
_ATTN_SPLITS_FEW = 32
_ATTN_SPLITS_MANY = 8
_ATTN_FEW_SEQUENCES = 4
_ATTN_WARPS = 4
#: The Markov logit add and argmax read the vocabulary in chunks of this many columns.
_ARGMAX_BLOCK = 4096


def _rotate(x: torch.Tensor, table: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    """The draft's interleaved rope over a `[cos | sin]` table row per position."""
    cos, sin = table[positions.long()].chunk(2, dim=-1)
    if x.ndim == positions.ndim + 2:
        cos, sin = cos.unsqueeze(-2), sin.unsqueeze(-2)
    even, odd = x.float()[..., 0::2], x.float()[..., 1::2]
    return torch.stack((even * cos - odd * sin, even * sin + odd * cos), -1).flatten(-2).to(x.dtype)


class TritonDraftAttention(OpWrapper):
    """`latent = softmax(query @ keys^T * scale) @ keys[..., :512]` over each sequence's keys:
    the last `window` rows of its paged context (rows at or past its length masked) followed
    by the `block` keys of its own query block.

    `query` is `[batch, block, heads, 576]`, `pool` a layer's `[pages, page_size, 576]` view of
    the paged draft context, `page_table[table_rows[b]]` sequence `b`'s pages, `lengths[b]`
    its context length, `block_kv` `[batch, block, 576]`. Returns `[batch, block, heads, 512]`
    bf16.

    Arithmetic: the scores round to bf16 before the fp32 scale, as the bf16 score einsum does;
    the softmax is an online one over the splits, merged in fp32, and the probabilities enter the
    value product as bf16 unnormalized, so the output differs from the reference's normalized
    bf16 probabilities by bf16 rounding.
    """

    ARCHS = frozenset({Arch.SM_107})
    note = """
    The window is a multiple of 16 * 64 keys; heads 16 (tp4) or 8 (tp8); the pool's last dim
    contiguous. A masked key loads as zero, so pool rows past a sequence's length may hold any
    bits, NaN included.
    """
    CELLS = tuple(
        Cell(
            why=f"tp{tp} draft block of {batch} sequence(s), contexts {lengths}",
            spec=dict(batch=batch, heads=64 // tp, block=3, window=2048, page=64, lengths=lengths),
        )
        for tp, batch, lengths in (
            (4, 1, (17800,)),
            (4, 2, (5, 4096)),
            (4, 4, (2048, 2049, 300000, 1)),
            (4, 8, (2048,) * 8),
            (8, 1, (2047,)),
            (8, 16, tuple(range(1000, 33000, 2000))),
        )
    )

    def __call__(
        self,
        query: torch.Tensor,
        pool: torch.Tensor,
        page_table: torch.Tensor,
        table_rows: torch.Tensor,
        lengths: torch.Tensor,
        block_kv: torch.Tensor,
        scale: float,
        window: int,
    ) -> torch.Tensor:
        batch, block, heads, _ = query.shape
        qrows = block * heads
        page = pool.shape[1]
        nsplit = _ATTN_SPLITS_FEW if batch <= _ATTN_FEW_SEQUENCES else _ATTN_SPLITS_MANY
        splits = nsplit + 1
        partial_max = query.new_empty((batch, splits, qrows), dtype=torch.float32)
        partial_sum = torch.empty_like(partial_max)
        partial_out = query.new_empty((batch, splits, qrows, LAT), dtype=torch.float32)
        out = query.new_empty((batch, block, heads, LAT))
        grid = (batch, splits, triton.cdiv(qrows, _ATTN_ROWS))
        _kernels.draft_attention_partial_kernel[grid](
            query, pool, pool.stride(0), pool.stride(1), page_table, page_table.stride(0),
            table_rows, lengths, block_kv, partial_max, partial_sum, partial_out, float(scale),
            page_table.shape[1] * page - 1,
            PAGE=page, WINDOW=window, BLOCK=block, QROWS=qrows, NSPLIT=nsplit,
            BLOCK_T=_ATTN_BLOCK_T, ROWS=_ATTN_ROWS, LAT=LAT, PE=PE, num_warps=_ATTN_WARPS,
            enable_fp_fusion=False,
        )  # fmt: skip
        _kernels.draft_attention_combine_kernel[(batch * qrows,)](
            partial_max, partial_sum, partial_out, out, QROWS=qrows, SPLITS=splits, LAT=LAT,
            num_warps=4,
        )  # fmt: skip
        return out

    def reference(
        self,
        query: torch.Tensor,
        pool: torch.Tensor,
        page_table: torch.Tensor,
        table_rows: torch.Tensor,
        lengths: torch.Tensor,
        block_kv: torch.Tensor,
        scale: float,
        window: int,
    ) -> torch.Tensor:
        batch, block = query.shape[:2]
        page = pool.shape[1]
        lengths = lengths[:batch].long()
        offsets = torch.arange(window, device=query.device)
        positions = (lengths - window).clamp_min(0)[:, None] + offsets[None, :]
        valid = positions < lengths[:, None]
        table = page_table.index_select(0, table_rows.long()).long()
        safe = positions.clamp(max=table.shape[1] * page - 1)
        pages = table.gather(1, safe // page).clamp_min(0)
        context = pool[pages, safe % page].masked_fill(~valid.unsqueeze(-1), 0)
        keys = torch.cat((context, block_kv), dim=1)
        key_valid = torch.cat(
            (valid, torch.ones(batch, block, dtype=torch.bool, device=valid.device)), dim=1
        )
        scores = torch.einsum("bshd,btd->bsht", query, keys).float() * scale
        scores = scores.masked_fill(~key_valid[:, None, None, :], float("-inf"))
        probabilities = torch.softmax(scores, dim=-1).to(keys.dtype)
        return torch.einsum("bsht,btc->bshc", probabilities, keys[..., :LAT])


class TritonLatentTail(OpWrapper):
    """For every row and layer `l`: `out[:, l] = [rmsnorm(kv_l[:512]) * weight[l] |
    rope(kv_l[512:])]` with `kv_l = kv[:, l * 576:(l + 1) * 576]`; with `q` and `query`, also
    `query[r, h, 512:] = rope(q[r, h, nope:])` per head. The rope reads the row's position in
    the `[positions, 64]` `[cos | sin]` table.

    The rope is bit-identical to the torch chain; the RMSNorm reduces in fp32 in its own order,
    so it can differ from flashinfer's by one bf16 ulp."""

    ARCHS = frozenset({Arch.SM_107})
    #: The target calls it one layer at a time: a draft layer's tail with the query rope into a
    #: `[batch, block, 576]` buffer, or one of the 3 layers of the context features into its
    #: slice of the `[rows, 3, 1, 576]` buffer, over a decode step's accepted tokens or a
    #: prefill step.
    CELLS = tuple(
        Cell(
            why=f"{what}: {rows} rows",
            spec=dict(rows=rows, heads=heads, with_q=heads is not None),
        )
        for what, rows, heads in (
            ("tp4 draft layer, 1 sequence", 3, 16),
            ("tp4 draft layer, 16 sequences", 48, 16),
            ("tp8 draft layer, 4 sequences", 12, 8),
            ("context features layer, 1 sequence", 4, None),
            ("context features layer, 16 sequences", 64, None),
            ("context features layer, a 16384-token prefill step", 16384, None),
        )
    )

    def __call__(
        self,
        kv: torch.Tensor,
        weight: torch.Tensor,
        eps: float,
        positions: torch.Tensor,
        table: torch.Tensor,
        out: torch.Tensor,
        q: torch.Tensor | None = None,
        query: torch.Tensor | None = None,
        nope: int = 128,
    ) -> torch.Tensor:
        rows, layers = kv.shape[0], weight.shape[0]
        out_rows = out.view(rows, layers, LAT + PE)
        has_q = q is not None
        heads = query.shape[-2] if has_q else 1
        _kernels.latent_tail_kernel[(rows, layers)](
            kv, kv.stride(0), LAT + PE, weight, out_rows, out_rows.stride(0), out_rows.stride(1),
            q if has_q else kv, q.stride(0) if has_q else 0, query if has_q else out_rows,
            heads * (LAT + PE), positions, table, float(eps),
            HEADS=heads, NOPE=nope, QK=nope + PE, LAT=LAT, PE=PE, HAS_Q=has_q, num_warps=4,
            enable_fp_fusion=False,
        )  # fmt: skip
        return out

    def reference(
        self,
        kv: torch.Tensor,
        weight: torch.Tensor,
        eps: float,
        positions: torch.Tensor,
        table: torch.Tensor,
        out: torch.Tensor,
        q: torch.Tensor | None = None,
        query: torch.Tensor | None = None,
        nope: int = 128,
    ) -> torch.Tensor:
        rows, layers = kv.shape[0], weight.shape[0]
        out_rows = out.view(rows, layers, LAT + PE)
        positions = positions.reshape(-1)
        for layer in range(layers):
            lat = kv[:, layer * (LAT + PE) : (layer + 1) * (LAT + PE)]
            xf = lat[:, :LAT].float()
            normed = xf * torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + eps)
            out_rows[:, layer, :LAT] = (normed * weight[layer].float()).to(kv.dtype)
            out_rows[:, layer, LAT:] = _rotate(lat[:, LAT:], table, positions)
        if q is not None:
            heads = query.shape[-2]
            pe = q.view(rows, heads, -1)[..., nope:]
            query.view(rows, heads, LAT + PE)[..., LAT:] = _rotate(pe, table, positions)
        return out


class TritonMarkovInput(OpWrapper):
    """The recurrent Markov head's joint input for one step: `[state | embedding(prev) | hidden]`,
    the embedding zero for a token id off the vocabulary and the state zero when `state` is
    None (the chain's first step). Copies only, so bit-identical to the torch chain."""

    ARCHS = frozenset({Arch.SM_107})
    CELLS = tuple(
        Cell(
            why=f"{batch} sequence(s), {'first' if first else 'later'} step",
            spec=dict(batch=batch, first=first),
        )
        for batch in (1, 4, 16)
        for first in (True, False)
    )

    def __call__(
        self,
        previous: torch.Tensor,
        embedding: torch.Tensor,
        state: torch.Tensor | None,
        hidden: torch.Tensor,
    ) -> torch.Tensor:
        batch, rank = previous.shape[0], embedding.shape[1]
        out = hidden.new_empty((batch, 2 * rank + hidden.shape[1]))
        _kernels.markov_input_kernel[(batch,)](
            previous, embedding, embedding.shape[0], state if state is not None else out,
            hidden, hidden.stride(0), out,
            HAS_STATE=state is not None, RANK=rank, HIDDEN=hidden.shape[1], HID_BLOCK=2048,
            num_warps=4,
        )  # fmt: skip
        return out

    def reference(
        self,
        previous: torch.Tensor,
        embedding: torch.Tensor,
        state: torch.Tensor | None,
        hidden: torch.Tensor,
    ) -> torch.Tensor:
        valid = (previous >= 0) & (previous < embedding.shape[0])
        rows = torch.nn.functional.embedding(
            previous.clamp(0, embedding.shape[0] - 1), embedding
        ).masked_fill(~valid.unsqueeze(-1), 0)
        if state is None:
            state = torch.zeros_like(rows)
        return torch.cat((state, rows, hidden), dim=-1)


class TritonMarkovGate(OpWrapper):
    """One recurrent step after the joint projection: `gate, cand, emit = joint.chunk(3)`,
    `state = sigmoid(gate) * state + (1 - sigmoid(gate)) * tanh(cand)` and `tanh(emit)`,
    every op rounded to bf16 as the torch chain rounds it. Returns `(state, tanh(emit))`."""

    ARCHS = frozenset({Arch.SM_107})
    CELLS = TritonMarkovInput.CELLS

    def __call__(
        self, joint: torch.Tensor, state: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, rank = joint.shape[0], joint.shape[1] // 3
        new_state = joint.new_empty((batch, rank))
        emit = torch.empty_like(new_state)
        _kernels.markov_gate_kernel[(batch,)](
            joint, state if state is not None else new_state, new_state, emit,
            HAS_STATE=state is not None, RANK=rank, num_warps=4, enable_fp_fusion=False,
        )  # fmt: skip
        return new_state, emit

    def reference(
        self, joint: torch.Tensor, state: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        gate, candidate, emitted = joint.chunk(3, dim=-1)
        if state is None:
            state = torch.zeros_like(gate)
        gate = torch.sigmoid(gate)
        return gate * state + (1 - gate) * torch.tanh(candidate), torch.tanh(emitted)


class TritonMarkovLogits(OpWrapper):
    """`out = base + bias` (fp32 base, bf16 bias) written in place into `out`, and the row's
    argmax as int64 with torch.argmax's first-index tie-break. Bit-identical sums; the argmax
    is exact for rows without NaN."""

    ARCHS = frozenset({Arch.SM_107})
    CELLS = tuple(
        Cell(why=f"{batch} sequence(s) over the 204800 vocabulary", spec=dict(batch=batch))
        for batch in (1, 4, 16)
    )

    def __call__(self, base: torch.Tensor, bias: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
        batch, vocab = bias.shape
        chunks = triton.cdiv(vocab, _ARGMAX_BLOCK)
        part_max = bias.new_empty((batch, chunks), dtype=torch.float32)
        part_idx = bias.new_empty((batch, chunks), dtype=torch.int32)
        tokens = bias.new_empty((batch,), dtype=torch.int64)
        _kernels.markov_logits_kernel[(batch, chunks)](
            base, base.stride(0), bias, out, out.stride(0), part_max, part_idx, vocab,
            CHUNKS=chunks, BLOCK_V=_ARGMAX_BLOCK, num_warps=8,
        )  # fmt: skip
        _kernels.argmax_finish_kernel[(batch,)](
            part_max, part_idx, tokens, CHUNKS=chunks, CHUNKS_P2=triton.next_power_of_2(chunks),
            num_warps=1,
        )  # fmt: skip
        return tokens

    def reference(self, base: torch.Tensor, bias: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
        out.copy_(base + bias)
        return torch.argmax(out, dim=-1)


triton_draft_attention = TritonDraftAttention()
triton_latent_tail = TritonLatentTail()
triton_markov_input = TritonMarkovInput()
triton_markov_gate = TritonMarkovGate()
triton_markov_logits = TritonMarkovLogits()
