# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Gather every rank's rows into one tensor, via the trtllm allgather op."""

from __future__ import annotations

from typing import List, Optional, Sequence

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*
from tensorrt_llm._utils import mpi_rank

from .._op import Arch, Cell, OpWrapper

# Mistral Large 4 NVFP4 at dep4: the token gather before the router of every MoE
# layer (1..71). Each of the four ranks passes its post-attention-norm rows
# zero-padded to T_max = max(attn_metadata.all_rank_num_tokens), sizes=None, on
# the MPI-session communicator a dep4 world launched under mpirun reaches.
_DEP4_GATHER = dict(
    trailing=(6144,),
    dtype=torch.bfloat16,
    sizes=None,
    world_size=4,
    process_group=None,
)
# The decode batches the dep4 engine captures per rank (padded), and the
# gathers one captured decode step issues.
_DEP4_GRAPH_ROWS = (1, 2, 4, 8, 16)
_DEP4_MOE_LAYERS = 71
# Mistral Large 4 NVFP4 at tep4 with its 1M envelope (max_seq_len 1048576, chunked
# prefill, max_num_tokens 32768): the q-split's selection gather. Once a step's
# tc context rows reach the engine's q_split_threshold (8192), rank r of the TP
# group selects for rows [tc * r // 4, tc * (r + 1) // 4) only, and after the
# chunk loop every selecting indexer layer gathers the four shares' int32 top-k
# rows (index_topk 2048 wide) into rows [0, tc) of the step's selection buffer:
# sizes = q_sizes from the engine's `_split_prefill_queries`, uneven by at most
# one row, on the MPI-session communicator the mpirun-launched tep4 world
# reaches. Context steps are never captured, so every call is eager.
_TEP4_SPLIT = dict(
    trailing=(2048,),
    dtype=torch.int32,
    world_size=4,
    process_group=None,
    q_split=True,
)
_TEP4_SELECTING_LAYERS = 44
# Mistral Large 4 NVFP4 at tp4 with the same envelope makes the same call: its
# attention and indexer are sharded as at tep4 (the indexer replicated), so its
# PrefillTarget._indexer gathers the same q_sizes shares over the same TP group
# [0, 1, 2, 3] once per selecting layer, and the cells stating `q_split` serve
# both targets. The tp4 cells add the remaining split remainders, top-k
# selection payloads (key positions to 1048575 with -1 tails) and the target's
# copy-back into its selection buffer at every swept tc.
_TP4_TOP_POSITION = 1048575

# Mistral Large 4 NVFP4 at dep8: the same gather over the eight attention-DP ranks
# of two GB300 nodes of four (one NVL72 segment, so NCCL forms an MNNVL clique
# of 8), group [0..7] in all_rank_num_tokens order. Each rank passes its rows
# zero-padded to T_max, sizes=None, on the MPI-session communicator an
# eight-rank world launched under mpirun across both nodes reaches. `nodes` is
# the number of hosts the ranks span.
_DEP8_GATHER = dict(
    trailing=(6144,),
    dtype=torch.bfloat16,
    sizes=None,
    world_size=8,
    process_group=None,
    nodes=2,
)
# The decode batches a dep8 engine captures per rank -- the gates' padded
# 1/2/4/8/16 and the LLM API's unpadded default at max_batch_size 16, every
# count 1..16 -- and the gathers one captured decode step issues.
_DEP8_GRAPH_ROWS = tuple(range(1, 17))
_DEP8_MOE_LAYERS = 71
# Mistral Large 4 NVFP4 at tep8 with its 1M envelope (max_seq_len 1048576, chunked
# prefill, max_num_tokens 32768): the q-split's selection gather. Once a step's
# tc context rows reach the engine's q_split_threshold (8192), rank r of the
# eight-rank TP group selects for rows [tc * r // 8, tc * (r + 1) // 8) only,
# and after the chunk loop every selecting indexer layer gathers the eight
# shares' int32 top-k rows (index_topk 2048 wide) into rows [0, tc) of the
# step's selection buffer: sizes = q_sizes from the engine's
# `_split_prefill_queries`, uneven by at most one row, on the MPI-session
# communicator the mn_mpirun-launched tep8 world reaches, over two nodes of
# four (one MNNVL clique of 8). Context steps are never captured, so every
# call is eager.
_TEP8_SPLIT = dict(
    trailing=(2048,),
    dtype=torch.int32,
    world_size=8,
    nodes=2,
    process_group=None,
    q_split=True,
)
_TEP8_SELECTING_LAYERS = 44


class Allgather(OpWrapper):
    """Gather `input` from every rank in `group`, concatenated along dim 0.

    `sizes` gives each rank's `input.shape[0]` in ascending rank order, or is
    `None` when every rank holds the same number of rows. The result is a
    freshly allocated tensor on every rank; `input` is left untouched.
    """

    ARCHS = frozenset({Arch.SM_103})

    # deepseek-r1 and Mistral Large 4 NVFP4 at dep4 and dep8 pass `sizes=None`: they
    # pad every rank to a common height first. Their row counts span what an
    # attention-DP rank feeds its expert-parallel MoE call, from one decode token
    # to a whole prefill step. Mistral Large 4 NVFP4 at tep4, at tp4 and at tep8
    # passes the ragged form: its q-split gathers each rank's share of a context
    # step's top-k rows with `sizes` = the engine's q_sizes (the cells stating
    # `q_split`). The first three cells drove the `_pg` symbol (a boxed c10d
    # group); every cell stating `process_group=None` drove the MPI-session
    # symbol, at the world size it states.
    CELLS: tuple[Cell, ...] = (
        Cell(
            why="decode: one row per rank, hidden 7168",
            spec=dict(rows=1, trailing=(7168,), dtype=torch.bfloat16, sizes=None),
        ),
        Cell(
            why="a small padded batch, the common case between decode and prefill",
            spec=dict(rows=32, trailing=(7168,), dtype=torch.bfloat16, sizes=None),
        ),
        Cell(
            why="prefill chunk height",
            spec=dict(rows=2048, trailing=(7168,), dtype=torch.bfloat16, sizes=None),
        ),
        *(
            Cell(
                why=f"dep4 MoE gather, decode graph at {rows} row(s) per rank: captured with "
                f"{_DEP4_MOE_LAYERS} gathers (MoE layers 1..71) beside the other captured "
                "batches in one pool, after an eager warmup of the group; every replay on new "
                "data bitwise the concatenation",
                spec=dict(
                    rows=rows,
                    graph_rows=_DEP4_GRAPH_ROWS,
                    sites=_DEP4_MOE_LAYERS,
                    **_DEP4_GATHER,
                ),
            )
            for rows in _DEP4_GRAPH_ROWS
        ),
        Cell(
            why="dep4 MoE gather, eager rows every value 1..2048: every protocol and "
            "channel-count switch NCCL's ring makes at this width (LL through 154 rows, LL128 "
            "155..990, SIMPLE from 991), each bitwise the concatenation",
            spec=dict(rows=2048, sweep_rows=(1, 2048), sweep_step=1, **_DEP4_GATHER),
        ),
        Cell(
            why="dep4 MoE gather, eager rows 2049..32768, one NCCL tuning bucket (ring, "
            "SIMPLE, 32 channels): a stride of 61 rows plus every 2^k - 1, 2^k and 2^k + 1 up "
            "to max_num_tokens 32768 (a 131072 x 6144 result), each bitwise the concatenation",
            spec=dict(
                rows=32768,
                sweep_rows=(2049, 32768),
                sweep_step=61,
                sweep_also=(4095, 4096, 4097, 8191, 8192, 8193, 16383, 16384, 16385, 32767),
                **_DEP4_GATHER,
            ),
        ),
        Cell(
            why="dep4 MoE gather at max_num_tokens with the padding skew: one rank's 32768 "
            "rows of arbitrary bf16 bit patterns (NaN, inf, -0.0, subnormals) and three "
            "ranks' single rows zero-padded to 32768 move bitwise and repeat bitwise; the "
            "input is left untouched and the output is fresh",
            spec=dict(rows=32768, payload="bits", valid_rows=(32768, 1, 1, 1), **_DEP4_GATHER),
        ),
        *(
            Cell(
                why=f"dep8 MoE gather, decode graph at {rows} row(s) per rank, 8 ranks over two "
                f"nodes: captured with {_DEP8_MOE_LAYERS} gathers (MoE layers 1..71) beside the "
                "other captured batches 1..16 in one pool, after an eager warmup of the group; "
                "every replay on new data bitwise the concatenation",
                spec=dict(
                    rows=rows,
                    graph_rows=_DEP8_GRAPH_ROWS,
                    sites=_DEP8_MOE_LAYERS,
                    **_DEP8_GATHER,
                ),
            )
            for rows in _DEP8_GRAPH_ROWS
        ),
        Cell(
            why="dep8 MoE gather, eager rows every value 1..2048, 8 ranks over two nodes: "
            "every protocol and channel-count switch NCCL's ring makes at this width (LL "
            "through 96 rows, LL128 97..1190, SIMPLE from 1191), each bitwise the "
            "concatenation",
            spec=dict(rows=2048, sweep_rows=(1, 2048), sweep_step=1, **_DEP8_GATHER),
        ),
        Cell(
            why="dep8 MoE gather, eager rows 2049..32768, 8 ranks over two nodes, one NCCL "
            "tuning bucket (ring, SIMPLE, 32 channels): a stride of 61 rows plus every "
            "2^k - 1, 2^k and 2^k + 1 and rows 21845/21846/21847, where the result passes 2^31 "
            "bytes, up to max_num_tokens 32768 (a 262144 x 6144 result, 3,221,225,472 bytes), "
            "each bitwise the concatenation",
            spec=dict(
                rows=32768,
                sweep_rows=(2049, 32768),
                sweep_step=61,
                sweep_also=(
                    4095,
                    4096,
                    4097,
                    8191,
                    8192,
                    8193,
                    16383,
                    16384,
                    16385,
                    21845,
                    21846,
                    21847,
                    32767,
                ),
                **_DEP8_GATHER,
            ),
        ),
        Cell(
            why="dep8 MoE gather at max_num_tokens with the padding skew over two nodes: one "
            "rank's 32768 rows of arbitrary bf16 bit patterns (NaN, inf, -0.0, subnormals) and "
            "seven ranks' single rows zero-padded to 32768 move bitwise and repeat bitwise; the "
            "input is left untouched and the output is fresh",
            spec=dict(
                rows=32768, payload="bits", valid_rows=(32768, 1, 1, 1, 1, 1, 1, 1), **_DEP8_GATHER
            ),
        ),
        Cell(
            why="tep4 q-split gather at tc 8192, the engine's q_split_threshold: four 2048-row "
            "int32 shares of [8192, 2048] selections (uniform q_sizes, so one ncclAllGather), "
            "each a view at its offset in the rank's selection buffer; three calls on new "
            "payloads, each bitwise the concatenation, the input untouched, the output fresh",
            spec=dict(rows=2048, sizes=[2048, 2048, 2048, 2048], tokens=8192, **_TEP4_SPLIT),
        ),
        Cell(
            why="tep4 q-split gather at tc 8193, the first ragged split: q_sizes [2048, 2048, "
            "2048, 2049], four ncclBroadcasts in one NCCL group, each rank's share a view at "
            "its offset in the selection buffer; three calls on new payloads, each bitwise "
            "the concatenation, the input untouched, the output fresh",
            spec=dict(rows=2049, sizes=[2048, 2048, 2048, 2049], tokens=8193, **_TEP4_SPLIT),
        ),
        Cell(
            why="tep4 q-split gather at tc 32767, the largest ragged split: q_sizes [8191, 8192, "
            "8192, 8192], a [32767, 2048] int32 result; three calls on new payloads, each "
            "bitwise the concatenation, the input untouched, the output fresh",
            spec=dict(rows=8192, sizes=[8191, 8192, 8192, 8192], tokens=32767, **_TEP4_SPLIT),
        ),
        Cell(
            why="tep4 q-split gather at max_num_tokens, tc 32768: four 8192-row shares of "
            "arbitrary int32 bit patterns move bitwise and repeat bitwise over four calls into "
            "four distinct fresh tensors; the input is left untouched",
            spec=dict(
                rows=8192,
                sizes=[8192, 8192, 8192, 8192],
                tokens=32768,
                payload="bits",
                **_TEP4_SPLIT,
            ),
        ),
        Cell(
            why="tep4 q-split gather, eager tc 8192..32768: a stride of 61 plus 8193..8195 (each "
            "uneven pattern), 16383..16385 and 32767, each rank's input its q_sizes share, the "
            "sizes the engine's _split_prefill_queries gives at every tc; one NCCL tuning "
            "bucket (ring, SIMPLE, 32 channels) on both paths; each bitwise the concatenation",
            spec=dict(
                rows=8192,
                sweep_tokens=(8192, 32768),
                sweep_step=61,
                sweep_also=(8193, 8194, 8195, 16383, 16384, 16385, 32767),
                **_TEP4_SPLIT,
            ),
        ),
        Cell(
            why="tep4 q-split step at max_num_tokens: 44 gathers at tc 32768 (one per selecting "
            "indexer layer, in layer order, no synchronization between), each copied into rows "
            "[0, tc) of the sentinel-filled selection buffer its input is a view of, as the "
            "target does; every layer's rows bitwise the concatenation, none left unwritten",
            spec=dict(
                rows=8192,
                sizes=[8192, 8192, 8192, 8192],
                tokens=32768,
                sites=_TEP4_SELECTING_LAYERS,
                **_TEP4_SPLIT,
            ),
        ),
        Cell(
            why="tep4 q-split step, ragged: the last 32751-row chunk of the envelope's "
            "1048559-token prompt (31 x 32768 + 32751), q_sizes [8187, 8188, 8188, 8188], 44 "
            "gathers in layer order into the sentinel-filled selection buffer; every layer's "
            "rows bitwise the concatenation, none left unwritten",
            spec=dict(
                rows=8188,
                sizes=[8187, 8188, 8188, 8188],
                tokens=32751,
                sites=_TEP4_SELECTING_LAYERS,
                **_TEP4_SPLIT,
            ),
        ),
        Cell(
            why="tp4 q-split gather at tc 8194, the second remainder: q_sizes [2048, 2049, 2048, "
            "2049], four ncclBroadcasts in one NCCL group, each rank's share a view at its "
            "offset in the selection buffer; three calls on new payloads, each bitwise the "
            "concatenation, the input untouched, the output fresh",
            spec=dict(rows=2049, sizes=[2048, 2049, 2048, 2049], tokens=8194, **_TEP4_SPLIT),
        ),
        Cell(
            why="tp4 q-split gather at tc 8195, the third remainder: q_sizes [2048, 2049, 2049, "
            "2049], four ncclBroadcasts in one NCCL group, each rank's share a view at its "
            "offset in the selection buffer; three calls on new payloads, each bitwise the "
            "concatenation, the input untouched, the output fresh",
            spec=dict(rows=2049, sizes=[2048, 2049, 2049, 2049], tokens=8195, **_TEP4_SPLIT),
        ),
        Cell(
            why="tp4 q-split step at max_num_tokens on top-k selections: 44 gathers at tc 32768 "
            "in layer order, each layer's rows key positions 0..1048575 with -1 tails (rows "
            "that see fewer than 2048 keys, rows that see none), copied into rows [0, tc) of "
            "the sentinel-filled selection buffer; every layer's rows bitwise the "
            "concatenation, none left unwritten",
            spec=dict(
                rows=8192,
                sizes=[8192, 8192, 8192, 8192],
                tokens=32768,
                sites=_TEP4_SELECTING_LAYERS,
                payload="indices",
                top_position=_TP4_TOP_POSITION,
                **_TEP4_SPLIT,
            ),
        ),
        Cell(
            why="tp4 q-split copy-back, eager totals 8192..32768 at a stride of 61 and 32768, "
            "sizes the floor split at each: every call's top-k selections (key positions to "
            "1048575, -1 tails) staged as each rank's share in its sentinel-filled selection "
            "buffer, gathered and copied into rows [0, tc) as PrefillTarget._indexer does; "
            "each tc's rows bitwise the concatenation, none left unwritten",
            spec=dict(
                rows=8192,
                sizes="floor_split",
                sweep_totals=(8192, 32768),
                sweep_step=61,
                payload="indices",
                top_position=_TP4_TOP_POSITION,
                **_TEP4_SPLIT,
            ),
        ),
        Cell(
            why="tep8 q-split gather at tc 8192, the engine's q_split_threshold: eight 1024-row "
            "int32 shares of [8192, 2048] selections over two nodes (uniform q_sizes, so one "
            "ncclAllGather), each a view at its offset in the rank's selection buffer; three "
            "calls on new payloads, each bitwise the concatenation, the input untouched, the "
            "output fresh",
            spec=dict(rows=1024, sizes=[1024] * 8, tokens=8192, **_TEP8_SPLIT),
        ),
        Cell(
            why="tep8 q-split gather at tc 8193, the first ragged split: q_sizes [1024 x 7, "
            "1025], eight ncclBroadcasts in one NCCL group over two nodes, each rank's share a "
            "view at its offset in the selection buffer; three calls on new payloads, each "
            "bitwise the concatenation, the input untouched, the output fresh",
            spec=dict(rows=1025, sizes=[1024] * 7 + [1025], tokens=8193, **_TEP8_SPLIT),
        ),
        Cell(
            why="tep8 q-split gather at tc 32736, the V2 scheduler's chunk beside 15 decode "
            "rows: q_sizes [4092] * 8 (uniform), a [32736, 2048] int32 result over two nodes; "
            "three calls on new payloads, each bitwise the concatenation, the input untouched, "
            "the output fresh",
            spec=dict(rows=4092, sizes=[4092] * 8, tokens=32736, **_TEP8_SPLIT),
        ),
        Cell(
            why="tep8 q-split gather at tc 32767, the largest ragged split: q_sizes [4095, "
            "4096 x 7], a [32767, 2048] int32 result over two nodes; three calls on new "
            "payloads, each bitwise the concatenation, the input untouched, the output fresh",
            spec=dict(rows=4096, sizes=[4095] + [4096] * 7, tokens=32767, **_TEP8_SPLIT),
        ),
        Cell(
            why="tep8 q-split gather at max_num_tokens, tc 32768: eight 4096-row shares of "
            "arbitrary int32 bit patterns over two nodes move bitwise and repeat bitwise over "
            "four calls into four distinct fresh tensors; the input is left untouched",
            spec=dict(
                rows=4096,
                sizes=[4096] * 8,
                tokens=32768,
                payload="bits",
                **_TEP8_SPLIT,
            ),
        ),
        Cell(
            why="tep8 q-split gather, eager tc 8192..32768 over two nodes: a stride of 61 plus "
            "8193..8199 (every uneven pattern at eight ranks), 16383..16385 and 32767, each "
            "rank's input its q_sizes share, the sizes the engine's _split_prefill_queries "
            "gives at every tc; each bitwise the concatenation",
            spec=dict(
                rows=4096,
                sweep_tokens=(8192, 32768),
                sweep_step=61,
                sweep_also=(8193, 8194, 8195, 8196, 8197, 8198, 8199, 16383, 16384, 16385, 32767),
                **_TEP8_SPLIT,
            ),
        ),
        Cell(
            why="tep8 q-split gather at NCCL's protocol switch: every tc 14273..14288 over two "
            "nodes, whose largest share is 1785 rows (RING, LL128 through tc 14280) or 1786 "
            "(RING, SIMPLE from tc 14281), NCCL tuning each call on 8 x its largest share's "
            "bytes; each bitwise the concatenation",
            spec=dict(rows=1786, sweep_tokens=(14273, 14288), sweep_step=1, **_TEP8_SPLIT),
        ),
        Cell(
            why="tep8 q-split step at max_num_tokens: 44 gathers at tc 32768 over two nodes "
            "(one per selecting indexer layer, in layer order, no synchronization between), each "
            "copied into rows [0, tc) of the sentinel-filled selection buffer its input is a "
            "view of, as the target does; every layer's rows bitwise the concatenation, none "
            "left unwritten",
            spec=dict(
                rows=4096,
                sizes=[4096] * 8,
                tokens=32768,
                sites=_TEP8_SELECTING_LAYERS,
                **_TEP8_SPLIT,
            ),
        ),
        Cell(
            why="tep8 q-split step, ragged: the last 32751-row chunk of the envelope's "
            "1048559-token prompt (31 x 32768 + 32751), q_sizes [4093, 4094 x 7], 44 gathers "
            "in layer order over two nodes into the sentinel-filled selection buffer; every "
            "layer's rows bitwise the concatenation, none left unwritten",
            spec=dict(
                rows=4094,
                sizes=[4093] + [4094] * 7,
                tokens=32751,
                sites=_TEP8_SELECTING_LAYERS,
                **_TEP8_SPLIT,
            ),
        ),
    )

    note = """
    Two communicator flavours, one per symbol (see `__call__`). The cells that
    state `process_group=None` drove `torch.ops.trtllm.allgather` on the
    MPI-session communicator: at `world_size=4` four ranks of one node started
    by mpirun, as a dep4, tep4 or tp4 world is; at `world_size=8` eight ranks
    over two GB300 nodes of four (`nodes=2`, one NVL72 segment, one MNNVL
    clique of 8) started by the session's MPI launcher across both nodes, as a
    dep8 or tep8 world is. `getComm` (`cpp/tensorrt_llm/common/opUtils.cpp`)
    builds one NCCL communicator per `group` set at that set's first call,
    through the MPI session, and caches it for every op that names the same set
    (the matrix captures this entry and reducescatter on it in one graph); it
    sets NCCL_RUNTIME_CONNECT=0 and NCCL_GRAPH_REGISTER=0 where they are unset.
    With `sizes=None` the call is one ncclAllGather of `input.numel()` elements
    per rank. With a `sizes` list the result has `sum(sizes)` rows: when every
    entry is equal it is that same one ncclAllGather, and otherwise one
    ncclBroadcast per rank of the group inside one NCCL group, the i-th rank
    (ascending) sending `sizes[i]` rows from its input to row `sum(sizes[:i])`
    of every rank's result. The cells without the key drove `allgather_pg` on a
    boxed c10d group.

    Preconditions on the MPI-session symbol: every rank of `group` makes the
    same calls in the same order at the same shape. Calls pair by position on
    the communicator; the op matrix's call-order check measures, on the `_pg`
    symbol, that a mispairing at equal byte counts returns wrong data silently.
    The group's first call builds the communicator, so it is made eagerly: the
    graph cells warm the group before capture, as the engine's warmup does.
    Capturing a group's first call is not certified on this symbol. With a
    `sizes` list, `sizes[i]` must be the i-th rank's `input.shape[0]`: the op
    takes the row count from `sizes`, never from the input. Measured at tc
    8193 on four ranks: a rank whose input was one row short of its entry sent
    the row after its input in memory; one row long, its last row was dropped;
    with every entry 2049 and every input 2048 rows, the blocks landed at
    2048-row offsets and the result's last four rows were never written.
    Measured again at tc 8193 on eight ranks over two nodes, q_sizes [1024 x 7,
    1025]: the last rank's input one row short of its entry sent the row after
    its input in memory; rank 0's input one row long, its last row was dropped;
    with every entry 1025 and every input 1024 rows, the blocks landed at
    1024-row offsets and the result's last eight rows were never written. Both
    times every rank got the same wrong result, and nothing raised. All silent,
    so `is_valid` refuses a `sizes` entry that is not this rank's input height
    on this symbol (the rank from the MPI session).

    Measured on 4 sm_103 ranks at width 6144 bf16 (NCCL's TUNING log, every
    rows to 2048 and every 8th to 32768): the RING algorithm at every row
    count, LL through 154 rows, LL128 from 155, SIMPLE from 991, with the
    channel count moving between 3 and 32 below 621 rows and 32 from there on;
    every rank made the same choice on every call, and the captured batches
    1/2/4/8/16 get eager's algorithm, protocol and channels. The op moves
    bytes: every rows 1..32768 was driven once on the MPI-session symbol and
    each result was the concatenation bit for bit (47.9 s, past one cell's
    budget, so the cells keep every value to 2048 and a stride above it).

    Measured on 8 sm_103 ranks over two GB300 nodes at width 6144 bf16: the
    eight ranks form one MNNVL clique (NCCL reports nNodes 1 and localRanks 8,
    and connects every ring link P2P/MNNVL). Rank 0's TUNING log (NCCL logs
    the choice on rank 0 only), every rows to 2048 and every 8th to 32768: the
    RING algorithm at every row count -- NVLS is available on the devices and
    never chosen for this op -- LL through 96 rows, LL128 from 97, SIMPLE from
    1191, with the channel count moving between 3 and 32 below 311 rows and 32
    from there on; the captured batches 1..16 get eager's algorithm, protocol
    and channels. Every rows 1..32768 was driven once on the MPI-session symbol
    at world 8 and each result was the concatenation bit for bit on all eight
    ranks, a sweep past one cell's budget, so the dep8 cells too keep every
    value to 2048 and a stride above it.

    The tep4 q-split gather (the cells stating `q_split` at `world_size=4`),
    measured on 4 sm_103 ranks at width 2048 int32 (8 KiB rows), NCCL 2.31. Its
    sizes are what the engine's `_split_prefill_queries` (`dsa/indexer.py`)
    gives tp 4: checked against it at every tc 1..32768, unsplit below 8192 and
    [(r + 1) * tc // 4 - r * tc // 4] from there on, so q_sizes are equal when
    4 divides tc and uneven by one row otherwise. NCCL's TUNING log over 420
    split sizes (tc 8192..8199, a stride of 61 to 32768, the edges): RING,
    SIMPLE, 32 channels at every one, on every rank. The uniform ones run the
    ncclAllGather, and NCCL tunes the ragged ones' grouped broadcasts as one
    AllGather of the total bytes. The op moves bytes: every tc 8192..32768 was
    driven once on the MPI-session symbol, each rank's input its share of a
    fresh payload, and each result was the concatenation bit for bit (24577
    calls, 10.3 s, past one cell's budget, so the sweep cell keeps a stride of
    61 and the edges). A split step's 44 gathers, issued in layer order with no
    synchronization between them and each copied into rows [0, tc) of the
    buffer its input is a view of, left every layer's rows bitwise right and
    no row unwritten, at tc 32768 and 32751. Warm, a ragged split's gather
    costs more than the uniform one beside it.

    The tp4 target makes the same call (its indexer replicated, its TP group
    the same four ranks, its split the same: `_split_prefill_queries` agrees on
    its Mapping too, moe_tp_size 4). Measured again on a tp4 session (4 sm_103
    ranks, NCCL 2.31.2) in the target's own form -- each rank's share staged in
    a sentinel-filled selection buffer, gathered, and the result copied into
    rows [0, tc) of that buffer -- on top-k shaped payloads (key positions
    0..1048575 in a row's first slots, -1 in the rest: rows with a -1 tail,
    rows of -1 only, rows of 2048 positions): every tc 8192..32768 once, a
    fresh payload per call, each bitwise the shares' concatenation with no row
    unwritten (24577 calls, 19.2 s, past one cell's budget, so the copy-back
    cell keeps a stride of 61); a step's 44 such gathers in layer order left
    every layer right at tc 32768, 32751 and 8195; NCCL's TUNING log over the
    same 420 tc chose RING, SIMPLE, 32 channels on every rank. Warm, a gather
    at a ragged remainder costs more than the uniform one beside it.

    The tep8 q-split gather (the cells stating `q_split` at `world_size=8`),
    measured on 8 sm_103 ranks over two GB300 nodes of four (one MNNVL clique
    of 8) at width 2048 int32 (8 KiB rows). Its sizes are what the engine's
    `_split_prefill_queries` (`dsa/indexer.py`) gives the tep8 Mapping (tp 8,
    moe_ep 8, attention DP off): checked against it at every tc 1..32768,
    unsplit below 8192 and [(r + 1) * tc // 8 - r * tc // 8] from there on, so
    q_sizes are equal when 8 divides tc and uneven by one row otherwise. NCCL's
    TUNING log over 502 calls (tc 8192..8199, a stride of 61 to 32768, the
    edges, every tc 14224..14304), read call by call: NCCL tunes a call when
    its group ends, on 8 x the largest share's bytes, and chose RING with 32
    channels at every one, on every rank, with the LL128 protocol through tc
    14280 (largest share 1785 rows) and SIMPLE from tc 14281 (1786 rows), for
    the uniform ncclAllGather and the ragged grouped ncclBroadcasts alike. The
    op moves bytes: every tc 8192..32768 was driven once on the MPI-session
    symbol, each rank's input its share of a fresh payload, and each result was
    the concatenation bit for bit on all eight ranks (24577 calls, 11.0 s, past
    one cell's budget, so one cell keeps a stride of 61 and the edges and
    another every tc across the protocol switch). A split step's 44 gathers,
    issued in layer order with no synchronization between them and each copied
    into rows [0, tc) of the buffer its input is a view of, left every layer's
    rows bitwise right and no row unwritten, at tc 32768, 32751 and 8193. Warm,
    a ragged split's gather costs more than a uniform one beside it.

    The ragged form and dtypes other than bf16 and int32 also work on the
    `_pg` symbol, which the op matrix exercises; only the q-split's ragged form
    is a cell. Not certified on the MPI-session symbol: a ragged `sizes` other
    than the q-split's (int32 at width 2048, summing to 8192..32768 over four
    ranks of one node or eight ranks on two nodes, uneven by at most one row),
    other dtypes and widths, a group other than all four ranks of one node or
    all eight ranks of two nodes, other world sizes and node counts, rows above
    32768 (per rank in the uniform form, in all in the q-split's), and a ragged
    gather captured in a CUDA graph.

    CUDA graph behaviour on the `_pg` symbol -- capture at every engine batch
    size, replays surviving eager calls of other shapes, the captured sizes
    vector being held -- is not a cell axis: it is about the op across capture
    and replay rather than about an input configuration, and the matrix carries
    it as its own named checks. On the MPI-session symbol the graph cells above
    are the dep4 and dep8 decode graphs themselves.

    Hard limits (sm_103), the uniform form at width 6144 bf16, world 4
      output bytes   4 * rows * 12288 vs 2^31   reached at rows 43691
                     not reached by dep4 (32768: 1,610,612,736 bytes)   not probed
      output numel   4 * rows * 6144 vs 2^31    reached at rows 87382
                     not reached   not probed
      NCCL count     input.numel() as size_t; no index passes through fp32

    Hard limits (sm_103), the uniform form at width 6144 bf16, world 8 over two nodes
      output bytes   8 * rows * 12288 vs 2^31   reached at rows 21846, inside dep8
                     (32768: 3,221,225,472 bytes)   probed at every rows 21845..32768
                     once, and kept as cells at 21845, 21846, 21847, a stride of 61
                     and 32767/32768   passed, bitwise on all eight ranks
      output numel   8 * rows * 6144 vs 2^31    reached at rows 43691
                     not reached by dep8 (32768: 1,610,612,736)   not probed
      NCCL count     input.numel() as size_t; no index passes through fp32

    Hard limits (sm_103), the q-split form at width 2048 int32, world 4
      output bytes   tc * 8192 vs 2^31          reached at tc 262144
                     not reached by tep4 or tp4 (32768: 268,435,456 bytes)
                     not probed
      output numel   tc * 2048 vs 2^31          reached at tc 1048576
                     not reached (32768: 2^26 elements)   not probed
      sum(sizes)     summed into an int         wraps at 2^31 rows   not reached
      NCCL count     per root numel_base * sizes[i] as size_t, numel_base (the
                     trailing elements, 2048) multiplied into an int; the row
                     offset is torch's int64; no index passes through fp32

    Hard limits (sm_103), the q-split form at width 2048 int32, world 8
      output bytes   tc * 8192 vs 2^31          reached at tc 262144
                     not reached by tep8 (32768: 268,435,456 bytes)   not probed
      output numel   tc * 2048 vs 2^31          reached at tc 1048576
                     not reached   not probed
      sum(sizes)     summed into an int         wraps at 2^31 rows   not reached
      NCCL count     per root numel_base * sizes[i] as size_t, numel_base (the
                     trailing elements, 2048) multiplied into an int; the row
                     offset is torch's int64; no index passes through fp32
    """

    def __call__(
        self,
        input: torch.Tensor,
        sizes: Optional[List[int]],
        group: List[int],
        process_group: Optional[object] = None,
    ) -> torch.Tensor:
        """`process_group` selects the communicator flavor, mirroring the
        product router in `_torch/distributed/ops.py`: None runs the
        MPI-session symbol -- the one a dep4, tep4, tp4, dep8 or tep8 serving
        world calls -- and a boxed c10d ProcessGroup runs the `_pg` symbol the
        non-MPI backends use.
        """
        if process_group is None:
            return torch.ops.trtllm.allgather(input, sizes, group)
        return torch.ops.trtllm.allgather_pg(input, sizes, group, process_group)

    def reference(
        self,
        per_rank_inputs: Sequence[torch.Tensor],
        sizes: Optional[List[int]] = None,
        group: Optional[List[int]] = None,
    ) -> torch.Tensor:
        """The concatenation the collective is supposed to make.

        The documented exception to mirroring `__call__`: a collective's output
        is not a function of what the calling rank holds, so a reference given
        only `input` could not state anything at all. It takes the group's
        inputs in ascending rank order instead. The matrix builds that list on
        one rank from the seed convention every rank shares, which is what keeps
        the reference arithmetic rather than a second collective.
        """
        return torch.cat(list(per_rank_inputs), dim=0)

    def is_valid(
        self,
        input: torch.Tensor,
        sizes: Optional[List[int]],
        group: List[int],
        process_group: Optional[object] = None,
    ) -> None:
        # Each guard is for a violation measured to end in silent corruption or
        # a process kill rather than an error.
        assert input.dim() >= 1, (
            "input must have at least one dimension; a 0-d tensor segfaults inside "
            "AllgatherOp::run_list"
        )
        assert input.is_contiguous(), (
            "input must be contiguous; the op reads it as packed memory and a "
            "strided view is silently gathered from the wrong elements"
        )
        assert sizes is None or len(sizes) == len(group), (
            f"len(sizes)={len(sizes) if sizes is not None else None} must equal "
            f"len(group)={len(group)}; the op sizes its output from `sizes` alone, "
            "so a short list silently drops the trailing ranks and a long one "
            "appends uninitialized rows"
        )
        if process_group is None and sizes is not None and mpi_rank() in group:
            mine = sorted(set(group)).index(mpi_rank())
            assert sizes[mine] == input.shape[0], (
                f"sizes[{mine}]={sizes[mine]} must equal this rank's input.shape[0]="
                f"{input.shape[0]}; the MPI-session op sends sizes[{mine}] rows from the "
                "input whatever its height, so a shorter input is read past its end, a "
                "longer one is cut, and equal sizes pack the blocks at the inputs' offsets"
            )


allgather = Allgather()
