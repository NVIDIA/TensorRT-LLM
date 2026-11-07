# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DSA sparse MLA generation attention at 4 or 8 query heads per rank, with the v_b BMM folded in."""

from __future__ import annotations

import math
from typing import Optional

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*
from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import (
    Arch,
    Cell,
    OpWrapper,
    true_fp32_matmul,
)

#: Query heads per rank: GLM-5.2's 64 MLA heads over 16 ranks.
NUM_HEADS = 4
TP8_NUM_HEADS = 8
LATENT_NUM_HEADS = 16
LATENT_MAX_TOKENS = 22
#: The latent row: kv_lora_rank 512 followed by qk_rope_head_dim 64.
HEAD_DIM_QK = 576
KV_LORA_RANK = 512
V_HEAD_DIM = 256
#: The most selected tokens per query token the kernel reads.
MAX_TOP_K = 2048
#: The kernel splits the top-k slots over 16 CTAs of 128 slots; each split exponentiates
#: against its own max.
_SPLITS = 16
_SLOTS_PER_SPLIT = MAX_TOP_K // _SPLITS
#: P is quantized to E4M3 after this scale, as the TRTLLM-Gen generation kernels feed their
#: second GEMM.
_P_SCALE = 448.0

#: GLM-5.2's KV pool at TP16: 79 attention layers per rank (78 decoder layers and the MTP
#: layer, at local layer 78), one FP8 page of 64 tokens per layer and pool slot.
_GLM_POOL = dict(tokens_per_block=64, num_layers=79, top_k=MAX_TOP_K)
#: 1 / sqrt(qk_nope_head_dim + qk_rope_head_dim) = 1 / sqrt(192 + 64).
GLM_SOFTMAX_SCALE = 1.0 / math.sqrt(256.0)


def _cell(**spec) -> dict:
    """A cell spec over GLM-5.2's pool geometry.

    ``kv_lens`` holds each sequence's KV length with this step's ``seq_len_q`` tokens; ``remap``
    passes positions and the block table (else global pool rows); ``layer_offset`` is the pool
    layer; ``holes`` leaves masked entries inside the rows' valid range.
    """
    return {**_GLM_POOL, "gather_before_wait": False, "holes": False, **spec}


class GlmSparseMlaFlashDecode(OpWrapper):
    """Sparse MLA flash decode over an FP8 paged latent pool, v_b folded into the output.

    Wraps ``torch.ops.trtllm.glm_sparse_mla_flash_decode``. Query token ``t`` (row ``t`` of
    ``q``) belongs to sequence ``s = t // seq_len_q`` and attends to the first
    ``n_t = min(seq_lens_kv[s] - (seq_len_q - 1 - t % seq_len_q), top_k)`` slots of
    ``indices[t]``; a negative entry is masked. With ``block_table`` a slot holds a position
    ``p`` of the sequence, read at pool row::

        block_table[s, p // tokens_per_block] * page_index_scale * tokens_per_block
            + p % tokens_per_block + layer_offset * tokens_per_block

    (``convert_req_index_to_global``'s rows; a block past the table's width or a negative
    entry masks the slot). Without it, a slot holds the pool row. Every head ``h`` then gets
    the softmax over the selected rows of ``(q[t, h] . row) * mla_bmm1_scale[1]`` (base 2),
    applied to the rows' first 512 values and scaled by ``mla_bmm2_scale[0]``, rounded to BF16,
    and multiplied by ``v_b_proj[h]``: ``output[t, h * 256 + v]``. A row with nothing selected
    returns zeros.

    The probabilities enter the second product as E4M3: each split of 128 slots exponentiates
    against its own max, and ``exp2(s - max) * 448`` is rounded to E4M3, while the
    normalizing sums add the unrounded values. ``reference`` states that rounding.

    Fusion boundary: the index remap, the gather of the selected rows, the softmax, P.V, the
    16-way merge and the v_b product are inside. Selecting the tokens, the KV append, the
    query's projection, RoPE and quantization, and the scales are not.
    """

    ARCHS = frozenset({Arch.SM_107})

    CELLS: tuple[Cell, ...] = (
        Cell(
            why="verify, full-indexer layer: 8 rows, the layer's own top-k remapped in-kernel",
            spec=_cell(seq_len_q=8, kv_lens=(20_011,), remap=True, layer_offset=6),
        ),
        Cell(
            why="verify, shared-indexer layer: the leader's top-k, old rows gathered before wait",
            spec=_cell(
                seq_len_q=8, kv_lens=(20_011,), remap=True, layer_offset=7, gather_before_wait=True
            ),
        ),
        Cell(
            why="MTP draft step 0: 8 rows at the MTP layer, the last layer of the pool",
            spec=_cell(seq_len_q=8, kv_lens=(20_018,), remap=True, layer_offset=78),
        ),
        Cell(
            why="MTP draft steps 1-6: one row on global pool rows (reused top-k remapped aside)",
            spec=_cell(seq_len_q=1, kv_lens=(20_019,), remap=False, layer_offset=78),
        ),
        Cell(
            why="short context: fewer positions than the top-k, -1 padding and masked holes",
            spec=_cell(
                seq_len_q=8,
                kv_lens=(1_500,),
                remap=True,
                layer_offset=4,
                gather_before_wait=True,
                holes=True,
            ),
        ),
        Cell(
            why="two sequences: per-sequence block-table rows and KV lengths (max_batch_size 2)",
            spec=_cell(
                seq_len_q=8,
                kv_lens=(9_000, 17_003),
                remap=True,
                layer_offset=11,
                gather_before_wait=True,
            ),
        ),
        *(
            Cell(
                why=f"TP8 folded flash: {seq_len_q} query rows x {len(kv_lens)} requests, early={early}",
                spec=_cell(
                    heads=TP8_NUM_HEADS,
                    seq_len_q=seq_len_q,
                    kv_lens=kv_lens,
                    remap=remap,
                    layer_offset=7,
                    gather_before_wait=early,
                    holes=True,
                ),
            )
            for seq_len_q, kv_lens, remap, early in (
                (1, (1500,), False, False),
                (1, (9000, 17003), True, True),
                (4, (1500,), True, False),
                (4, (9000, 17003), True, True),
                (8, (1500,), True, False),
                (8, (20011,), True, False),
                (8, (20011,), True, True),
            )
        ),
    )

    note = """
    Metadata consumed: none -- every input is an argument. What the target passes:
      - `q`: the [T, 4 or 8, 576] E4M3 query that q_b's fused epilogue writes (absorbed q_nope,
        roped q_pe); `seq_len_q` the query tokens per sequence (8 on verify and MTP draft
        step 0, 1 on draft steps 1-6).
      - `kv_pool`: the DSA cache manager's `get_unique_primary_pool()`, the base that
        `host_kv_cache_pool_pointers[pool, 0]` names and `convert_req_index_to_global`'s
        rows count from.
      - With `block_table` (layers that run the indexer, and the shared layers after one):
        `indices` the [T, 2048] local top-k positions (the full layer's own; on a shared
        layer its full layer's), `block_table` the DSA metadata's `block_table[num_contexts:
        num_seqs]` (pool slots, -1 past each sequence), `(page_index_scale, layer_offset)`
        `get_primary_pool_page_index_params(local_layer)`, `tokens_per_block` the manager's.
        Without it (draft steps 1-6): `indices` are global pool rows.
      - `seq_lens_kv`: the generation sequences' KV lengths with this step's tokens
        (`kv_lens_cuda_runtime[num_contexts:num_seqs]`).
      - `mla_bmm1_scale` / `mla_bmm2_scale`: the buffers `glm_mla_kv_half_append` fills.

    Masked slots load pool row 0 and multiply it by P = 0: a NaN in that row reaches the
    output. Outputs are bitwise deterministic. `output` rows past `q.size(0)` are untouched.

    PDL: the kernel waits on its grid dependency before any global read, except with
    `gather_before_wait`: then the indices, the block table, the KV lengths and the pool rows
    of positions below `seq_lens_kv - seq_len_q` are read before the wait, so their writers
    must have completed before this grid launches (step-start host copies, earlier steps,
    and an earlier layer's top-k that some kernel ahead in the stream waited on). A layer
    whose own top-k writes `indices` must not set it. Dependents may launch once the partial
    outputs are sent; they must wait on their grid dependency before reading `output`.

    The test also replays the shared-layer cell from a captured CUDA graph (the op allocates
    nothing and syncs nothing) and runs both ops on a DSACacheManagerV2 pool against
    `convert_req_index_to_global`.

    Not certified: `top_k` below 2048, more than two sequences, a ragged batch (present it
    token-major: `seq_len_q` 1 with per-row lengths and tables).
    """

    def __call__(
        self,
        q: torch.Tensor,
        kv_pool: torch.Tensor,
        indices: torch.Tensor,
        seq_lens_kv: torch.Tensor,
        mla_bmm1_scale: torch.Tensor,
        mla_bmm2_scale: torch.Tensor,
        v_b_proj: torch.Tensor,
        output: torch.Tensor,
        seq_len_q: int,
        top_k: int,
        block_table: Optional[torch.Tensor] = None,
        tokens_per_block: int = 0,
        page_index_scale: int = 0,
        layer_offset: int = 0,
        gather_before_wait: bool = False,
    ) -> torch.Tensor:
        torch.ops.trtllm.glm_sparse_mla_flash_decode(
            q,
            kv_pool,
            indices,
            seq_lens_kv,
            mla_bmm1_scale,
            mla_bmm2_scale,
            v_b_proj,
            output,
            seq_len_q,
            top_k,
            block_table,
            tokens_per_block,
            page_index_scale,
            layer_offset,
            gather_before_wait,
        )
        return output

    def reference(
        self,
        q: torch.Tensor,
        kv_pool: torch.Tensor,
        indices: torch.Tensor,
        seq_lens_kv: torch.Tensor,
        mla_bmm1_scale: torch.Tensor,
        mla_bmm2_scale: torch.Tensor,
        v_b_proj: torch.Tensor,
        output: torch.Tensor,
        seq_len_q: int,
        top_k: int,
        block_table: Optional[torch.Tensor] = None,
        tokens_per_block: int = 0,
        page_index_scale: int = 0,
        layer_offset: int = 0,
        gather_before_wait: bool = False,
    ) -> torch.Tensor:
        """The [T, 1024] BF16 rows the op writes into ``output[:T]``.

        fp32 throughout, with the kernel's three roundings stated: P to E4M3 at scale 448 per
        128-slot split (against the split's max, the sums over the unrounded P), the merged
        attention output to BF16 before the v_b product, and the product to BF16. ``output``
        and ``gather_before_wait`` are ignored: one is where the op writes, the other when it
        reads.
        """
        num_tokens = q.shape[0]
        device = q.device
        pool_rows = kv_pool.view(torch.uint8).reshape(-1, HEAD_DIM_QK)

        slots = torch.full((num_tokens, MAX_TOP_K), -1, dtype=torch.int64, device=device)
        slots[:, :top_k] = indices[:, :top_k].long()
        token = torch.arange(num_tokens, device=device)
        seq = token // seq_len_q
        num_valid = torch.clamp(
            seq_lens_kv[seq].long() - (seq_len_q - 1 - token % seq_len_q), max=top_k
        )
        if block_table is not None:
            position = slots.clamp(min=0)
            block = position // tokens_per_block
            width = block_table.shape[1]
            page = block_table[seq[:, None], block.clamp(max=width - 1)].long()
            row = (
                page * (page_index_scale * tokens_per_block)
                + position % tokens_per_block
                + layer_offset * tokens_per_block
            )
            resident = (slots >= 0) & (block < width) & (page >= 0)
        else:
            row = slots
            resident = slots >= 0
        valid = resident & (torch.arange(MAX_TOP_K, device=device)[None, :] < num_valid[:, None])
        row = torch.where(valid, row, torch.zeros_like(row))

        keys = pool_rows[row].view(torch.float8_e4m3fn).float()
        queries = q.view(torch.float8_e4m3fn).float()
        with true_fp32_matmul():
            scores = torch.einsum("thd,tjd->thj", queries, keys) * mla_bmm1_scale[1].float()
        scores = scores.masked_fill(~valid[:, None, :], float("-inf"))
        scores = scores.view(num_tokens, q.shape[1], _SPLITS, _SLOTS_PER_SPLIT)
        split_max = scores.amax(dim=-1)
        empty = split_max == float("-inf")
        p = torch.exp2(scores - torch.where(empty, 0.0, split_max)[..., None])
        split_sum = p.sum(dim=-1)
        p_e4m3 = (p * _P_SCALE).to(torch.float8_e4m3fn).float()
        values = keys[..., :KV_LORA_RANK].reshape(
            num_tokens, _SPLITS, _SLOTS_PER_SPLIT, KV_LORA_RANK
        )
        with true_fp32_matmul():
            split_out = torch.einsum("thrj,trjd->thrd", p_e4m3, values)
        merged_max = split_max.amax(dim=-1, keepdim=True)
        alpha = torch.where(empty, 0.0, torch.exp2(split_max - merged_max))
        merged_sum = (alpha * split_sum).sum(dim=-1)[..., None]
        merged = (alpha[..., None] * split_out).sum(dim=-2)
        attn = torch.where(
            merged_sum > 0,
            merged / (merged_sum * _P_SCALE) * mla_bmm2_scale.reshape(-1)[0].float(),
            0.0,
        )
        attn = attn.to(torch.bfloat16).float()
        with true_fp32_matmul():
            out = torch.einsum("thd,hvd->thv", attn, v_b_proj.float())
        return out.reshape(num_tokens, q.shape[1] * V_HEAD_DIM).to(torch.bfloat16)

    def is_valid(
        self,
        q: torch.Tensor,
        kv_pool: torch.Tensor,
        indices: torch.Tensor,
        seq_lens_kv: torch.Tensor,
        mla_bmm1_scale: torch.Tensor,
        mla_bmm2_scale: torch.Tensor,
        v_b_proj: torch.Tensor,
        output: torch.Tensor,
        seq_len_q: int,
        top_k: int,
        block_table: Optional[torch.Tensor] = None,
        tokens_per_block: int = 0,
        page_index_scale: int = 0,
        layer_offset: int = 0,
        gather_before_wait: bool = False,
    ) -> None:
        """The index and table contents the op reads without checking.

        The op checks every shape, dtype and stride loudly; what it trusts is the contents of
        ``indices``, ``seq_lens_kv`` and ``block_table``. ``q``, the scales, ``v_b_proj``,
        ``output`` and ``gather_before_wait`` are not read here.
        """
        num_tokens = indices.shape[0]
        device = indices.device
        num_pool_rows = kv_pool.numel() // HEAD_DIM_QK
        slots = indices[:, :top_k].long()
        token = torch.arange(num_tokens, device=device)
        seq = token // seq_len_q
        kv_len = seq_lens_kv[seq].long()
        num_valid = torch.clamp(kv_len - (seq_len_q - 1 - token % seq_len_q), max=top_k)
        unread = torch.arange(slots.shape[1], device=device)[None, :] >= num_valid[:, None]

        # Only the first n_t slots of row t are read: a selection packed elsewhere (a valid
        # entry behind them) silently drops tokens.
        assert not ((slots >= 0) & unread).any(), (
            "indices holds a selected token past the row's valid count "
            "min(kv_len - (seq_len_q - 1 - t), top_k); the op never reads it"
        )
        selected = (slots >= 0) & ~unread
        if block_table is None:
            # Global rows are read as given: one past the pool reads other memory.
            assert not (selected & (slots >= num_pool_rows)).any(), (
                f"indices names a pool row past the pool's {num_pool_rows} rows"
            )
            return

        # The positions are the query token's own context: a later position reads a row this
        # step has not appended, or another request's.
        own_position = kv_len - seq_len_q + token % seq_len_q
        assert not (selected & (slots > own_position[:, None])).any(), (
            "indices selects a position past the query token's own: the row is not this "
            "token's context"
        )
        # The layer's rows of a slot lie inside the slot: an offset at or past the page scale
        # reads the next slot's pages.
        assert 0 <= layer_offset < page_index_scale, (
            f"layer_offset {layer_offset} must be below page_index_scale {page_index_scale}"
        )
        # Every block of a sequence's KV must be mapped: a narrower table or a -1 inside the
        # sequence masks live tokens instead of failing.
        num_seqs = num_tokens // seq_len_q
        lens = seq_lens_kv[:num_seqs].long()
        blocks = (lens + tokens_per_block - 1) // tokens_per_block
        width = block_table.shape[1]
        assert bool((blocks <= width).all()), (
            f"block_table has {width} columns; a sequence needs {int(blocks.max())}"
        )
        live = torch.arange(width, device=device)[None, :] < blocks[:, None]
        table = block_table[:num_seqs].long()
        assert not (live & (table < 0)).any(), "block_table has a -1 inside a sequence's KV"
        if bool(live.any()):
            last_page = int(table[live].max()) * page_index_scale + layer_offset
            end_row = (last_page + 1) * tokens_per_block
            assert end_row <= num_pool_rows, (
                f"block_table maps rows up to {end_row}, past the pool's {num_pool_rows} rows"
            )


sparse_mla_flash_decode = GlmSparseMlaFlashDecode()


class GlmSparseMlaFlashDecodeLatent(OpWrapper):
    """TP4 sparse MLA: sixteen query heads, explicit BF16 latent output [T,16,512].

    For up to eight query tokens the top-k is split over sixteen CTAs; for nine to
    twenty-two, over eight CTAs. The v_b projection is a separate consumer. With
    trigger_after_wait=True that consumer must stage only its weights before its
    own dependency wait and read this output after the wait. The default is False.
    Index, cache and gather-before-wait semantics match the folded four-head op.
    """

    ARCHS = frozenset({Arch.SM_107})
    CELLS: tuple[Cell, ...] = (
        *(
            Cell(
                why=f"TP4 draft batch {batch}: sixteen latent heads, global rows",
                spec=_cell(seq_len_q=1, kv_lens=(4099,) * batch, remap=False, layer_offset=78),
            )
            for batch in (1, 2, 3)
        ),
        *(
            Cell(
                why=f"TP4 verify batch {batch}: {6 * batch} queries, remapped pool rows",
                spec=_cell(
                    seq_len_q=6,
                    kv_lens=(4101,) * batch,
                    remap=True,
                    layer_offset=6,
                    gather_before_wait=batch == 2,
                ),
            )
            for batch in (1, 2, 3)
        ),
        Cell(
            why="TP4 eight-query boundary: sixteen CTA splits, masked short context",
            spec=_cell(seq_len_q=8, kv_lens=(1500,), remap=True, layer_offset=4, holes=True),
        ),
        Cell(
            why="TP4 nine-query boundary: eight CTA splits",
            spec=_cell(seq_len_q=9, kv_lens=(4101,), remap=True, layer_offset=4),
        ),
        Cell(
            why="TP4 twenty-two-query ceiling: eight CTA splits",
            spec=_cell(seq_len_q=22, kv_lens=(4101,), remap=True, layer_offset=4),
        ),
    )

    def __call__(
        self,
        q: torch.Tensor,
        kv_pool: torch.Tensor,
        indices: torch.Tensor,
        seq_lens_kv: torch.Tensor,
        mla_bmm1_scale: torch.Tensor,
        mla_bmm2_scale: torch.Tensor,
        output: torch.Tensor,
        seq_len_q: int,
        top_k: int,
        block_table: Optional[torch.Tensor] = None,
        tokens_per_block: int = 0,
        page_index_scale: int = 0,
        layer_offset: int = 0,
        gather_before_wait: bool = False,
        trigger_after_wait: bool = False,
    ) -> torch.Tensor:
        torch.ops.trtllm.glm_sparse_mla_flash_decode_latent(
            q,
            kv_pool,
            indices,
            seq_lens_kv,
            mla_bmm1_scale,
            mla_bmm2_scale,
            output,
            seq_len_q,
            top_k,
            block_table,
            tokens_per_block,
            page_index_scale,
            layer_offset,
            gather_before_wait,
            trigger_after_wait,
        )
        return output

    def reference(
        self,
        q: torch.Tensor,
        kv_pool: torch.Tensor,
        indices: torch.Tensor,
        seq_lens_kv: torch.Tensor,
        mla_bmm1_scale: torch.Tensor,
        mla_bmm2_scale: torch.Tensor,
        output: torch.Tensor,
        seq_len_q: int,
        top_k: int,
        block_table: Optional[torch.Tensor] = None,
        tokens_per_block: int = 0,
        page_index_scale: int = 0,
        layer_offset: int = 0,
        gather_before_wait: bool = False,
        trigger_after_wait: bool = False,
    ) -> torch.Tensor:
        """Selected-row attention with the kernel's split-local E4M3 probability rounding."""
        num_tokens = q.shape[0]
        splits = 16 if num_tokens <= 8 else 8
        slots_per_split = MAX_TOP_K // splits
        device = q.device
        pool_rows = kv_pool.view(torch.uint8).reshape(-1, HEAD_DIM_QK)

        slots = torch.full((num_tokens, MAX_TOP_K), -1, dtype=torch.int64, device=device)
        slots[:, :top_k] = indices[:, :top_k].long()
        token = torch.arange(num_tokens, device=device)
        seq = token // seq_len_q
        num_valid = torch.clamp(
            seq_lens_kv[seq].long() - (seq_len_q - 1 - token % seq_len_q), max=top_k
        )
        if block_table is not None:
            position = slots.clamp(min=0)
            block = position // tokens_per_block
            width = block_table.shape[1]
            page = block_table[seq[:, None], block.clamp(max=width - 1)].long()
            row = (
                page * (page_index_scale * tokens_per_block)
                + position % tokens_per_block
                + layer_offset * tokens_per_block
            )
            resident = (slots >= 0) & (block < width) & (page >= 0)
        else:
            row = slots
            resident = slots >= 0
        valid = resident & (torch.arange(MAX_TOP_K, device=device)[None, :] < num_valid[:, None])
        row = torch.where(valid, row, torch.zeros_like(row))

        keys = pool_rows[row].view(torch.float8_e4m3fn).float()
        queries = q.view(torch.float8_e4m3fn).float()
        with true_fp32_matmul():
            scores = torch.einsum("thd,tjd->thj", queries, keys) * mla_bmm1_scale[1].float()
        scores = scores.masked_fill(~valid[:, None, :], float("-inf"))
        scores = scores.view(num_tokens, LATENT_NUM_HEADS, splits, slots_per_split)
        split_max = scores.amax(dim=-1)
        empty = split_max == float("-inf")
        p = torch.exp2(scores - torch.where(empty, 0.0, split_max)[..., None])
        split_sum = p.sum(dim=-1)
        p_e4m3 = (p * _P_SCALE).to(torch.float8_e4m3fn).float()
        values = keys[..., :KV_LORA_RANK].reshape(num_tokens, splits, slots_per_split, KV_LORA_RANK)
        with true_fp32_matmul():
            split_out = torch.einsum("thrj,trjd->thrd", p_e4m3, values)
        merged_max = split_max.amax(dim=-1, keepdim=True)
        alpha = torch.where(empty, 0.0, torch.exp2(split_max - merged_max))
        merged_sum = (alpha * split_sum).sum(dim=-1)[..., None]
        merged = (alpha[..., None] * split_out).sum(dim=-2)
        attn = torch.where(
            merged_sum > 0,
            merged / (merged_sum * _P_SCALE) * mla_bmm2_scale.reshape(-1)[0].float(),
            0.0,
        )
        return attn.to(torch.bfloat16)

    def is_valid(
        self,
        q: torch.Tensor,
        kv_pool: torch.Tensor,
        indices: torch.Tensor,
        seq_lens_kv: torch.Tensor,
        mla_bmm1_scale: torch.Tensor,
        mla_bmm2_scale: torch.Tensor,
        output: torch.Tensor,
        seq_len_q: int,
        top_k: int,
        block_table: Optional[torch.Tensor] = None,
        tokens_per_block: int = 0,
        page_index_scale: int = 0,
        layer_offset: int = 0,
        gather_before_wait: bool = False,
        trigger_after_wait: bool = False,
    ) -> None:
        # Both ABIs read the same sparse metadata; this validator does not read v_b.
        GlmSparseMlaFlashDecode.is_valid(
            self,
            q,
            kv_pool,
            indices,
            seq_lens_kv,
            mla_bmm1_scale,
            mla_bmm2_scale,
            None,
            output,
            seq_len_q,
            top_k,
            block_table,
            tokens_per_block,
            page_index_scale,
            layer_offset,
            gather_before_wait,
        )


sparse_mla_flash_decode_latent = GlmSparseMlaFlashDecodeLatent()
