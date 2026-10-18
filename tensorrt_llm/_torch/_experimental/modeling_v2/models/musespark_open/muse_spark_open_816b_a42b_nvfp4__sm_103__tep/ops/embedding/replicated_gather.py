# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Token embedding from a replicated table, with the bits of the vocab-parallel embedding.

The built-in ``Embedding`` of this target is vocab-parallel (``COLUMN``, TP8):
each rank holds 25,256 of the 202,048 rows, masks the ids it does not own
(seven small kernels), gathers, zero-fills the rows it does not own and the
eight partial rows are summed by an MNNVL one-shot all-reduce. That
all-reduce first replaces every -0.0 of a rank's payload by +0.0 (the Lamport
protocol's empty-slot sentinel) and then adds the eight payloads in fp32 from
+0.0 in rank order, rounding once to bf16. With one owner row and seven +0.0
rows the sum is the owner's value exactly, except that -0.0 becomes +0.0, and an
id no rank owns gives a row of +0.0.

This entry produces the same rows from the whole table held on every rank: one
row gather per id, -0.0 written as +0.0, and a +0.0 row for an id outside the
table. Integer moves only, so the output equals the sharded path bit for bit
whenever the all-reduce's fp32 add keeps the table's values -- which holds for
every normal and zero bf16 value; a subnormal would depend on whether that
all-reduce kernel was built with flush-to-zero, so the feature that installs
this entry refuses a table that holds one (``replicated_embedding.py``).
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

#: Columns per program: 8 bf16 values (16 bytes) per thread at 4 warps.
BLOCK = 1024

#: Shards the reference splits the table into (the target's TP width).
REFERENCE_SHARDS = 8


@triton.jit
def _replicated_embedding_gather_kernel(
    table_ptr,
    ids_ptr,
    out_ptr,
    vocab,
    HIDDEN: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    chunk = tl.program_id(1)
    token = tl.load(ids_ptr + row).to(tl.int64)
    valid = (token >= 0) & (token < vocab)
    cols = chunk * BLOCK + tl.arange(0, BLOCK)
    in_row = cols < HIDDEN
    source = tl.where(valid, token, 0) * HIDDEN
    # bf16 moved as its 16 bits: an id outside the table reads nothing (+0.0),
    # and a zero of either sign is written as +0.0.
    bits = tl.load(table_ptr + source + cols, mask=in_row & valid, other=0)
    magnitude = bits.to(tl.int32) & 0x7FFF
    bits = tl.where(magnitude == 0, tl.zeros_like(bits), bits)
    tl.store(out_ptr + row.to(tl.int64) * HIDDEN + cols, bits, mask=in_row)


class ReplicatedEmbeddingGather(OpWrapper):
    """``out[..., :] = table[ids[...]]`` with -0.0 as +0.0 and +0.0 rows for ids outside the table.

    ``table`` is the whole ``[vocab, hidden]`` bf16 embedding (contiguous),
    ``ids`` any integer tensor; the output is ``[*ids.shape, hidden]`` bf16.
    """

    ARCHS = frozenset({Arch.SM_103})

    CELLS: tuple[Cell, ...] = (
        Cell(
            "the target's decode step: 8 int32 token ids",
            spec=dict(shape=(8,), dtype="int32", ids="valid", table="normal"),
        ),
        Cell(
            "the DFlash bonus and mask rows: 2 int64 ids, as the built-in drafter prep embeds them",
            spec=dict(shape=(2,), dtype="int64", ids="valid", table="normal"),
        ),
        Cell(
            "the DFlash noise block: [1, 8] int64 ids, the bonus token then 7 mask ids",
            spec=dict(shape=(1, 8), dtype="int64", ids="noise", table="normal"),
        ),
        Cell(
            "signed zeros: every -0.0 of the table comes out +0.0, as the all-reduce sanitizes it",
            spec=dict(shape=(8,), dtype="int32", ids="valid", table="signed_zeros"),
        ),
        Cell(
            "ids no shard owns (-1, vocab, larger) give +0.0 rows, as every rank masks them",
            spec=dict(shape=(6,), dtype="int64", ids="invalid", table="normal"),
        ),
        Cell(
            "shard edges: the first and last row of every rank's shard",
            spec=dict(shape=(16,), dtype="int64", ids="edges", table="normal"),
        ),
    )

    note = """
    Metadata consumed: none. Grid (ids, ceil(hidden / 1024)), 4 warps, one
    16-byte load and store per thread; no programmatic-dependent-launch
    attributes (each launch starts after its predecessor completes and its
    successor sees all of its stores).

    Call sites: ``replicated_embedding.ReplicatedEmbedding.forward``
    (``replicated_embedding``: the target's decode embedding and the drafter's
    bonus / mask embedding, which share the module) and the DFlash noise block
    of ``verify_fold`` with ``replicated_embedding`` (``spec_loop_fold.py``).

    Numerics: integer moves, bitwise the vocab-parallel embedding + MNNVL
    one-shot all-reduce for every table without subnormal values (the
    reference states the sharded computation).

    Not certified: torch.compile. CUDA-graph capture and replay are exercised
    by the test.
    """

    def __call__(self, table: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
        vocab, hidden = table.shape
        out = torch.empty((*ids.shape, hidden), dtype=table.dtype, device=table.device)
        rows = ids.numel()
        if rows == 0:
            return out
        _replicated_embedding_gather_kernel[(rows, triton.cdiv(hidden, BLOCK))](
            table.view(torch.int16),
            ids.reshape(-1),
            out.view(torch.int16),
            vocab,
            HIDDEN=hidden,
            BLOCK=BLOCK,
            num_warps=4,
        )
        return out

    def reference(self, table: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
        """The built-in sharded computation: masked shard lookups summed as the all-reduce sums them.

        The table is split into ``REFERENCE_SHARDS`` row shards as the built-in
        ``Embedding`` splits it; each shard masks the ids it does not own,
        looks the rest up and zero-fills the masked rows
        (``pre_comm_embedding_ops``); each partial has its -0.0 replaced by
        +0.0 and the partials are added in fp32 from +0.0 in rank order, then
        rounded once to bf16 (the one-shot all-reduce).
        """
        vocab, hidden = table.shape
        width = -(-vocab // REFERENCE_SHARDS)
        flat = ids.reshape(-1).long()
        total = torch.zeros((flat.numel(), hidden), dtype=torch.float32, device=table.device)
        for rank in range(REFERENCE_SHARDS):
            start, end = rank * width, min((rank + 1) * width, vocab)
            owned = (flat >= start) & (flat < end)
            local = torch.where(owned, flat - start, torch.zeros_like(flat))
            shard = table[start:end]
            partial = shard[local.clamp(max=max(end - start - 1, 0))] if end > start else None
            if partial is None:
                continue
            partial = partial.masked_fill(~owned.unsqueeze(-1), 0)
            bits = partial.view(torch.int16)
            partial = torch.where((bits.int() & 0x7FFF) == 0, torch.zeros_like(partial), partial)
            total = total + partial.float()
        return total.to(table.dtype).reshape(*ids.shape, hidden)

    def is_valid(self, table: torch.Tensor, ids: torch.Tensor) -> None:
        assert table.dim() == 2 and table.is_contiguous(), (
            "table must be a contiguous [vocab, hidden]"
        )
        assert table.dtype == torch.bfloat16, f"table must be bf16, got {table.dtype}"
        assert not ids.dtype.is_floating_point and ids.dtype != torch.bool, "ids must be integers"
        assert ids.is_contiguous(), "ids must be contiguous"
        assert ids.device == table.device, "ids and table must be on one device"


replicated_embedding_gather = ReplicatedEmbeddingGather()
