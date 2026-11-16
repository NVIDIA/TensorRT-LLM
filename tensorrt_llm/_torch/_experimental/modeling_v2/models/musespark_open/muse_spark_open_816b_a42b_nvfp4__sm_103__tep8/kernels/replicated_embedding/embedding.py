# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fast path ``replicated_embedding``: the token embedding of small calls from a replicated table.

The built-in token embedding is vocab-parallel over the eight TP ranks
(``modules/embedding.py``, ``COLUMN`` with ``gather_output``): every call masks
the ids a rank does not own (seven elementwise kernels), gathers from the
rank's 25,256-row shard, zero-fills the rows it does not own and sums the
eight partial rows with an all-reduce. On a decode step that is ten launches
and a cross-rank synchronization before the first layer can start, and the
DFlash worker pays the same again for the drafter's bonus and mask rows
(``DFlashWorker.prepare_1st_drafter_inputs``; the drafter shares the target's
``embed_tokens`` module).

With this fast path every rank also holds the whole ``[202048, 8192]`` bf16
table (3.08 GiB more per rank, which the KV cache pool does not get), built
once after the weights load by all-gathering the shards, and a call of at most
``MAX_ROWS`` ids is one row gather (``ops/embedding/replicated_gather.py``).
The rows are the sharded path's bits: the all-reduce writes -0.0 as +0.0 and
adds one owner row to seven +0.0 rows in fp32, which the gather reproduces;
the install refuses a table holding a subnormal value, the one case where the
all-reduce kernel's build flags (flush-to-zero) would decide the bits, or a
NaN or infinity (the fp32 sum canonicalizes a NaN's sign and payload).

Larger calls (long prefill chunks) run the built-in path; a prefill chunk of
at most ``MAX_ROWS`` ids takes the gather, with the same bits (the sharded
partials are one owner row and zero rows, whatever all-reduce sums them).
Every rank sees the same row count for a call, so every rank takes the same
branch and the all-reduce call sequence stays identical across ranks.
"""

from __future__ import annotations

from typing import Optional

import torch

from tensorrt_llm._torch.distributed import allgather
from tensorrt_llm._torch.modules.embedding import Embedding
from tensorrt_llm._torch.modules.linear import TensorParallelMode
from tensorrt_llm.logger import logger
from tensorrt_llm.mapping import Mapping

from ... import dispatch as _tp_agreement
from ...ops.embedding.replicated_gather import replicated_embedding_gather

#: Largest call (number of ids) served from the replicated table: the target
#: rows of a decode step of 64 requests (K + 1 = 8 each, 512), with room for
#: one more row per request (the drafter's bonus and mask ids, the DFlash
#: noise block). The gather has no row limit of its own.
MAX_ROWS = 576

#: Shard rows all-gathered per collective while the table is built.
GATHER_ROWS = 2048

#: Attribute of the re-classed module that holds the whole table.
TABLE_ATTR = "_replicated_table"


class ReplicatedEmbedding(Embedding):
    """The built-in vocab-parallel ``Embedding`` that serves small calls from a whole table.

    Never constructed: ``install`` re-classes the built-in module and sets
    ``TABLE_ATTR``. The parameters (the rank's shard) stay as they are and
    serve every other call.
    """

    def forward(self, input):
        table = self.__dict__.get(TABLE_ATTR)
        if (
            table is not None
            and isinstance(input, torch.Tensor)
            and input.device == table.device
            and not input.dtype.is_floating_point
            and input.dtype != torch.bool
            and 1 <= input.numel() <= MAX_ROWS
        ):
            return replicated_embedding_gather(table, input.contiguous())
        return super().forward(input)


def table_of(module) -> Optional[torch.Tensor]:
    """The replicated table of an installed ``ReplicatedEmbedding``, or None."""
    if isinstance(module, ReplicatedEmbedding):
        return module.__dict__.get(TABLE_ATTR)
    return None


def unsupported_reason(embedding, mapping: Mapping) -> Optional[str]:
    """Why ``embedding`` is not the vocab-parallel layout the replicated table reproduces, or None."""
    if type(embedding) not in (Embedding, ReplicatedEmbedding):
        return f"the token embedding is {type(embedding).__name__}, not the built-in Embedding"
    if mapping is None or mapping.tp_size <= 1:
        return "no tensor parallelism: the built-in embedding is already one gather"
    if getattr(mapping, "enable_attention_dp", False):
        return "attention DP"
    if getattr(embedding, "tp_mode", None) != TensorParallelMode.COLUMN:
        return (
            f"the embedding is {getattr(embedding, 'tp_mode', None)}, not vocab-parallel (COLUMN)"
        )
    if getattr(embedding, "tp_size", 1) != mapping.tp_size:
        return "the embedding's TP size differs from the mapping's"
    weight = getattr(embedding, "weight", None)
    if not isinstance(weight, torch.Tensor) or weight.dtype != torch.bfloat16 or weight.dim() != 2:
        return "the embedding weight is not a 2-D bf16 tensor"
    if getattr(embedding, "padding_size", 0) != 0:
        return "the vocabulary is padded across the shards"
    width = weight.shape[0]
    if width * mapping.tp_size != embedding.num_embeddings:
        return (
            f"shards of {width} rows x {mapping.tp_size} ranks != vocab {embedding.num_embeddings}"
        )
    if (embedding.vocab_start_index, embedding.vocab_end_index) != (
        mapping.tp_rank * width,
        (mapping.tp_rank + 1) * width,
    ):
        return "this rank's shard is not rows [rank * width, (rank + 1) * width)"
    return None


def _gather_table(
    shard: torch.Tensor, mapping: Mapping, out: Optional[torch.Tensor]
) -> torch.Tensor:
    """The whole table from every rank's shard (collective over the TP group), into ``out`` if given."""
    width, hidden = shard.shape
    world = mapping.tp_size
    table = (
        out
        if out is not None
        else torch.empty((width * world, hidden), dtype=shard.dtype, device=shard.device)
    )
    for begin in range(0, width, GATHER_ROWS):
        part = shard[begin : begin + GATHER_ROWS].contiguous()
        rows = part.shape[0]
        gathered = allgather(part, mapping, dim=0)
        for rank in range(world):
            table[rank * width + begin : rank * width + begin + rows].copy_(
                gathered[rank * rows : (rank + 1) * rows]
            )
        del gathered
    return table


def _value_census(table: torch.Tensor) -> tuple[int, int, int]:
    """(subnormal values, non-finite values, -0.0 values) in a bf16 table, counted in row chunks."""
    subnormal = non_finite = negative_zero = 0
    bits = table.view(torch.int16)
    for begin in range(0, bits.shape[0], 4 * GATHER_ROWS):
        chunk = bits[begin : begin + 4 * GATHER_ROWS].int() & 0xFFFF
        exponent = chunk & 0x7F80
        subnormal += int(((exponent == 0) & ((chunk & 0x007F) != 0)).sum().item())
        non_finite += int((exponent == 0x7F80).sum().item())
        negative_zero += int((chunk == 0x8000).sum().item())
    return subnormal, non_finite, negative_zero


def install(text_model, mapping: Mapping) -> None:
    """Build the replicated table and re-class ``text_model.embed_tokens``; collective over TP.

    Every rank must call this at the same point, after the weights are final
    and before any CUDA graph is captured (the shell's ``load_weights``). A
    repeated call (a reload) refills the existing table in place.

    Raises:
        ValueError: if the embedding is not the vocab-parallel layout this
            reproduces (``unsupported_reason``), if the gathered table does not
            hold this rank's shard where it should, or if it holds a subnormal,
            NaN or infinite value. Raised alike on every rank: each check's
            outcome is agreed over the TP group (``tp_agreement``; the layout
            before the table's gather, the table after it), so a rank whose
            layout or rows differ makes every rank refuse instead of leaving its
            peers in the next collective.
    """
    embedding = text_model.embed_tokens
    reason = unsupported_reason(embedding, mapping)
    reason = _tp_agreement.agreed_refusal(
        mapping,
        None
        if reason is None
        else f"[musespark_open] replicated_embedding cannot run here: {reason}",
    )
    if reason is not None:
        raise ValueError(reason)
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    free_before = torch.cuda.mem_get_info()[0]
    shard = embedding.weight.detach()
    previous = table_of(embedding)
    table = _gather_table(shard, mapping, previous)
    width = shard.shape[0]
    mine = table[mapping.tp_rank * width : (mapping.tp_rank + 1) * width]
    subnormal, non_finite, negative_zero = _value_census(table)
    if not torch.equal(mine.view(torch.int16), shard.view(torch.int16)):
        reason = "the gathered table does not hold this rank's shard at its rows"
    elif subnormal:
        reason = (
            f"the embedding holds {subnormal} subnormal values, whose bits after the built-in "
            "all-reduce depend on its flush-to-zero build"
        )
    elif non_finite:
        reason = (
            f"the embedding holds {non_finite} NaN or infinite values (the built-in "
            "all-reduce's fp32 sum turns a NaN into the hardware's canonical NaN, which a row "
            "gather does not reproduce)"
        )
    else:
        reason = None
    reason = _tp_agreement.agreed_refusal(
        mapping,
        None
        if reason is None
        else f"[musespark_open] replicated_embedding cannot run here: {reason}",
    )
    if reason is not None:
        raise ValueError(reason)
    embedding.__dict__[TABLE_ATTR] = table
    changed = type(embedding) is not ReplicatedEmbedding
    embedding.__class__ = ReplicatedEmbedding
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    added = free_before - torch.cuda.mem_get_info()[0]
    if changed or previous is None:
        logger.info(
            f"[musespark_open] replicated_embedding: replicated token embedding {tuple(table.shape)} bf16 for calls "
            f"of at most {MAX_ROWS} ids (+{table.numel() * table.element_size() / 2**30:.2f} GiB "
            f"table, device +{added / 2**30:.2f} GiB; {negative_zero} -0.0 entries written as "
            "+0.0, no subnormal or non-finite values)"
        )
