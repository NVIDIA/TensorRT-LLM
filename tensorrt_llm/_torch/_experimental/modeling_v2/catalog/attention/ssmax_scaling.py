# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SSMax (scalable-softmax) query scaling, ``q[t, h, :] * (s[h] * log(pos[t] + 1) + b[h])``,
in the reference implementation's bf16 rounding: the MLA query of all 72 layers and the DSA
indexer query of the 44 layers that carry an indexer. The cells cover both Mistral Large 4
checkpoints' s and b, every decode batch 1..16 a CUDA graph is captured at, and positions up
to the NVFP4 target's max_seq_len of 32768 beside the FP8 target's 9216. The MLA cells drive
one rank's 16 of the 64 heads (tep4), one rank's 8 (tep8 and tp8, which shard attention
alike) and all 64 heads on one rank (dep4, attention data parallel, where every rank holds
every head and the unsliced [64] s and b). At tep4's and tp4's 16 MLA and 32 indexer heads per
rank (tensor parallel 4 at tp4: the same slices as at tep4) they also cover the checkpoint's
own context, max_seq_len 1048576 with chunked prefill: decode rows at positions up to 1048575,
chunk rows behind a cached prefix, packed chunks of several requests beside decode rows, and
every position 0..1048575, on drawn heads and on the checkpoint's own heads whose factor
crosses zero. At tep8's 8 MLA heads per rank (each rank r's [8] slice s[8 r : 8 r + 8],
b[8 r : 8 r + 8]) and the 32 replicated indexer heads they also cover the checkpoint's own
context, max_seq_len 1048576 with chunked prefill: decode rows at positions up to 1048575,
chunk rows behind a cached prefix, mixed and packed steps of several requests' chunks beside
decode rows, and every position 0..1048575, on drawn heads and on the checkpoint's own heads
whose factor crosses zero. At dep4's 64 MLA and 32 indexer heads they also cover the
checkpoint's own context, max_seq_len 1048576 with chunked prefill: decode rows at positions
up to 1048575, chunk rows behind a cached prefix, and a whole 1048576-token prompt as its 32
chunks, every position 0..1048575.

Sourcing tier 4, composed from native torch: no engine kernel computes it.
Rejected at the cheaper rungs, each searched on this build: tier 1, ``torch.ops.trtllm``
registers no op that scales a query per (token, head) by a log-position affine
(``mla_rope_generation`` takes one scalar ``q_scaling``; ``fused_qk_norm_rope`` and
``fused_qk_norm_rope_to_fp8`` carry one scalar ``attention_factor``, nothing per position;
``llama4_fp8_bf16_gemm`` folds Llama-4's attention temperature, one scalar with a floor of
the position, into one FP8 GEMM); tier 2, no callable under ``tensorrt_llm/_torch/modules``
or ``_torch/attention`` (the nearest relative, ``MLA._attention_scaling`` in
``_torch/attention/mla.py``, is that same Llama-4 temperature written inline in fp32, with no
per-head terms); tier 3, FlashInfer's ``rope`` module only rotates, and no FlashInfer callable
carries a per-token query scale. Correct and slow by design: seven elementwise launches,
which a tuning pass may fold into the rope or quantization kernel beside it.
"""

from __future__ import annotations

import torch

from .._op import Arch, Cell, OpWrapper

#: Per-rank query heads at tep4: the MLA's 64 split four ways, the indexer's 32 replicated.
#: At dep4 (attention data parallel) every rank holds all MLA_PARAM_HEADS of the MLA; at
#: tep8 and tp8 each of the eight ranks holds MLA_TEP8_HEADS of them.
MLA_HEADS = 16
MLA_TEP8_HEADS = 8
TEP8_RANKS = 8
MLA_PARAM_HEADS = 64
INDEXER_HEADS = 32
HEAD_DIM = 128

#: max_seq_len, which bounds every position: the NVFP4 targets serve 32768 (max_num_tokens
#: 32768 per rank, no chunked prefill, so one context step holds a whole 32768-token
#: prompt); the FP8 target's 9216 cells are kept as they were.
MAX_SEQ_LEN = 32768
FP8_MAX_SEQ_LEN = 9216

#: The checkpoint's own context (max_position_embeddings), which the tep4, tp4, dep4 and tep8
#: targets serve with chunked prefill: a step still carries at most MAX_NUM_TOKENS rows per
#: rank, a chunk's rows sit at positions cached .. cached + n - 1 behind its request's cached
#: prefix, and decode rows reach position LONG_MAX_SEQ_LEN - 1, where log(pos + 1) is 13.86
#: and a bf16 step of it is 0.0625.
LONG_MAX_SEQ_LEN = 1048576
MAX_NUM_TOKENS = 32768

#: MLA layer 0's s and b on tp4 rank 1 (heads 16..31 of 64), as the NVFP4 checkpoint stores
#: them (bf16, read on disk). Every s is negative, and two heads' factors cross zero below
#: LONG_MAX_SEQ_LEN, the only two of the checkpoint's 72 x 64 MLA heads that do: local head 4
#: (head 20, s -0.0693359, b 0.914063) near position 531308 and local head 10 (head 26,
#: s -0.078125, b 0.90625) near 109097. No indexer head crosses below 1048576.
NVFP4_MLA_LAYER0_RANK1 = (
    (
        -0.0135498046875, -0.036865234375, -0.0289306640625, -0.05322265625,
        -0.0693359375, -0.039306640625, -0.0106201171875, -0.026123046875,
        -0.025390625, -0.052490234375, -0.078125, -0.04052734375,
        -0.0400390625, -0.0322265625, -0.0301513671875, -0.05810546875,
    ),
    (
        0.9609375, 0.9453125, 0.9453125, 0.93359375, 0.9140625, 0.9453125, 0.96875, 0.96875,
        0.94921875, 0.921875, 0.90625, 0.9609375, 0.953125, 0.95703125, 0.9453125, 0.92578125,
    ),
)  # fmt: skip

#: MLA layer 0's s and b on tep8 ranks 2 and 3 (heads 16..23 and 24..31 of 64, the two halves
#: of NVFP4_MLA_LAYER0_RANK1), as the NVFP4 checkpoint stores them (bf16, read on disk). Every
#: s is negative, and two heads' factors cross zero below LONG_MAX_SEQ_LEN, the only two of
#: the checkpoint's 72 x 64 MLA heads that do, one on each slice: rank 2's local head 4 (head
#: 20, s -0.0693359, b 0.914063) near position 531308 and rank 3's local head 2 (head 26,
#: s -0.078125, b 0.90625) near 109097. No indexer head crosses below 1048576.
NVFP4_MLA_LAYER0_TEP8_RANK2 = (
    (
        -0.0135498046875, -0.036865234375, -0.0289306640625, -0.05322265625,
        -0.0693359375, -0.039306640625, -0.0106201171875, -0.026123046875,
    ),
    (0.9609375, 0.9453125, 0.9453125, 0.93359375, 0.9140625, 0.9453125, 0.96875, 0.96875),
)  # fmt: skip
NVFP4_MLA_LAYER0_TEP8_RANK3 = (
    (
        -0.025390625, -0.052490234375, -0.078125, -0.04052734375,
        -0.0400390625, -0.0322265625, -0.0301513671875, -0.05810546875,
    ),
    (0.94921875, 0.921875, 0.90625, 0.9609375, 0.953125, 0.95703125, 0.9453125, 0.92578125),
)  # fmt: skip

#: Decode batches a CUDA graph is captured at: the gates' padded 1/2/4/8/16, then the LLM
#: API default's unpadded batches between them (the engine cuts its 1..32 to 1..16 at
#: max_batch_size 16). On the MLA at tep4, each batch drives one rank's slice of the 64
#: heads; at dep4, a rank's own batch drives all 64.
GRAPH_BATCHES_PADDED = ((1, 0), (2, 1), (4, 2), (8, 3), (16, 3))
GRAPH_BATCHES_UNPADDED = tuple((n, n % 4) for n in (3, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15))

#: The draw ranges hold both Mistral Large 4 checkpoints' s and b over every layer that carries
#: them. NVFP4 (read on disk, bf16): MLA (72 layers x 64 heads) s in [-0.078125, 0.574219],
#: b in [0.902344, 1.57031]; indexer (44 layers x 32 heads) s in [-0.0437012, 0.0664062],
#: b in [0.976562, 1.04688]. FP8, rounded outward: MLA s in [-0.0767, 0.574], b in
#: [0.902, 1.578]; indexer s in [-0.0486, 0.0703], b in [0.973, 1.055].
_BF16 = dict(
    head_dim=HEAD_DIM, dtype=torch.bfloat16, positions_dtype=torch.int32, q="normal", corners=False
)
_MLA = dict(
    _BF16,
    site="mla",
    heads=MLA_HEADS,
    param_heads=MLA_PARAM_HEADS,
    s_range=(-0.08, 0.58),
    b_range=(0.9, 1.6),
)
_INDEXER = dict(
    _BF16,
    site="indexer",
    heads=INDEXER_HEADS,
    param_heads=INDEXER_HEADS,
    rank=0,
    s_range=(-0.05, 0.075),
    b_range=(0.97, 1.06),
)
#: The MLA at dep4: rank 0 of a 64-head draw holding all 64, so s and b are the whole
#: parameter, with the four corners on heads 0..3.
_MLA_DEP4 = dict(
    _MLA, heads=MLA_PARAM_HEADS, rank=0, max_seq_len=MAX_SEQ_LEN, corners=True, graph=False
)
#: The MLA at tep8 and tp8: rank r's 8 heads, 8 r .. 8 r + 7 of a 64-head draw, so s and b are views
#: at offset 8 r into the parameter, with the four corners on four of the slice's heads.
_MLA_TEP8 = dict(_MLA, heads=MLA_TEP8_HEADS, max_seq_len=MAX_SEQ_LEN, corners=True)
#: dep4 at the checkpoint's own context: the MLA's 64 heads as above, and the indexer's 32
#: (replicated, the same at every parallel layout), with the corners, at max_seq_len 1048576.
_MLA_DEP4_1M = dict(_MLA_DEP4, max_seq_len=LONG_MAX_SEQ_LEN)
_INDEXER_1M = dict(_INDEXER, max_seq_len=LONG_MAX_SEQ_LEN, corners=True, graph=False)
#: The same tep8 slices at the checkpoint's own context, max_seq_len 1048576.
_MLA_TEP8_1M = dict(_MLA_TEP8, max_seq_len=LONG_MAX_SEQ_LEN)


class SSMaxScaling(OpWrapper):
    """``q * (s * log(positions + 1) + b)`` per (token, head), a new tensor in `q`'s dtype.

    `q` is ``[T, H, D]``, `positions` the step's ``[T]`` position ids (int32, as the engine
    hands them over), `s` and `b` ``[H]``: this rank's slice of the MLA's 64 per-head
    parameters (heads ``16 r .. 16 r + 15`` on rank ``r`` at tep4, ``8 r .. 8 r + 7`` at
    tep8 and tp8; all 64, unsliced, under attention data parallel at dep4), or the indexer's
    32, replicated.

    The rounding is the specification, not an implementation detail: the log is taken in
    fp32 and rounded to `q`'s dtype, then ``* s`` and ``+ b`` are each rounded to it, then
    ``q * scale`` is. Those are the reference implementation's unfused steps, which its
    fused kernels reproduce bit for bit, and these are the same torch calls in the same
    order. An fp32 scale with no intermediate rounding is closer to the closed form and is
    not this op (the test measures how far apart they land).

    Fusion boundary: the log of the positions and the scaling. Everything around the query
    is the caller's. The reference implementation scales the MLA query after its rope (on
    the rope's bf16 output; on NoPE layers, the whole 128-wide query), and the indexer
    query before its rope and fp8 quantization. A caller whose rope runs inside a later op
    gets the same product with the rounding at a different step.
    """

    ARCHS = frozenset({Arch.SM_103})

    # spec: q is [tokens, heads, head_dim]; positions are each context's 0..n-1 in
    # `contexts` order, then `decode_rows` rows at random positions in 1..max_seq_len - 1
    # (the first at max_seq_len - 1). s and b are drawn over [s_range] / [b_range] for
    # `param_heads` heads and sliced to rank `rank`'s `heads`, so a rank > 0 slice is a
    # view at an offset. `corners` pins four of the slice's heads, (4 * rank) % heads ..
    # + 3 (4 * rank .. 4 * rank + 3 at 16 heads: no two ranks' at the same heads; at 8
    # heads, the first four on even ranks and the last four on odd ones), to the four
    # (s, b) corners of the ranges: the largest factor and the deepest cancellation the
    # ranges allow at the cell's positions.
    # `q` "normal" is 4 * N(0, 1); "wide" draws every bf16 binade, subnormals included.
    # `graph` captures the call in a CUDA graph and replays it on new q and positions
    # written into the captured inputs. `cached` (one per context, 0 when absent) starts
    # each context at its cached prefix, at positions cached .. cached + n - 1, as a chunk
    # after a request's first runs. `position_sweep` (lo, hi) drives every position
    # lo .. hi - 1, `tokens` rows per call, each call on a fresh q. `slice_values` (s, b)
    # replaces the draw on the rank's slice with those per-head values. `chunks` drives a
    # whole prompt of chunks * tokens positions as that many calls of `tokens` rows, chunk k
    # at positions k * tokens onward, each call on a fresh q.
    CELLS: tuple[Cell, ...] = (
        tuple(
            Cell(
                why=f"MLA decode, graph batch {n:>2}, rank {r}'s heads: "
                "captured, replayed on new q and positions",
                spec=dict(
                    _MLA,
                    max_seq_len=FP8_MAX_SEQ_LEN,
                    rank=r,
                    tokens=n,
                    contexts=(),
                    decode_rows=n,
                    graph=True,
                ),
            )
            for n, r in GRAPH_BATCHES_PADDED + GRAPH_BATCHES_UNPADDED
        )
        + tuple(
            Cell(
                why=f"indexer decode, graph batch {n:>2}: "
                "captured, replayed on new q and positions",
                spec=dict(
                    _INDEXER,
                    max_seq_len=FP8_MAX_SEQ_LEN,
                    tokens=n,
                    contexts=(),
                    decode_rows=n,
                    graph=True,
                ),
            )
            for n, _ in GRAPH_BATCHES_PADDED + GRAPH_BATCHES_UNPADDED
        )
        + (
            Cell(
                why="MLA context of 9216 rows at positions 0..9215: "
                "every position the FP8 target's envelope holds, rank 1's heads",
                spec=dict(
                    _MLA,
                    max_seq_len=FP8_MAX_SEQ_LEN,
                    rank=1,
                    tokens=9216,
                    contexts=(9216,),
                    decode_rows=0,
                    graph=False,
                ),
            ),
            Cell(
                why="indexer context of 9216 rows at positions 0..9215: "
                "every position the FP8 target's envelope holds",
                spec=dict(
                    _INDEXER,
                    max_seq_len=FP8_MAX_SEQ_LEN,
                    tokens=9216,
                    contexts=(9216,),
                    decode_rows=0,
                    graph=False,
                ),
            ),
            Cell(
                why="MLA mixed step, 3 contexts + 13 decode rows (16 requests, 9216 rows): "
                "positions are not the row index",
                spec=dict(
                    _MLA,
                    max_seq_len=FP8_MAX_SEQ_LEN,
                    rank=2,
                    tokens=9216,
                    contexts=(5000, 3000, 1203),
                    decode_rows=13,
                    graph=False,
                ),
            ),
            Cell(
                why="stress, s in [-1, 1] and b in [-2, 2], q over every bf16 binade: "
                "cancellation, overflow, subnormals",
                spec=dict(
                    _MLA,
                    max_seq_len=FP8_MAX_SEQ_LEN,
                    rank=3,
                    s_range=(-1.0, 1.0),
                    b_range=(-2.0, 2.0),
                    q="wide",
                    tokens=9216,
                    contexts=(9216,),
                    decode_rows=0,
                    graph=False,
                ),
            ),
        )
        # The NVFP4 target's envelope, max_seq_len = max_num_tokens = 32768 with no chunked
        # prefill: positions up to 32767 (log 10.4, so the MLA factor reaches 7.53 at the
        # checkpoint's largest s and b), one context step of up to 32768 rows.
        + tuple(
            Cell(
                why=f"MLA decode 32K, graph batch {n:>2}, rank {r}'s heads: positions up to "
                "32767, the range corners, captured, replayed on new q and positions",
                spec=dict(
                    _MLA,
                    max_seq_len=MAX_SEQ_LEN,
                    corners=True,
                    rank=r,
                    tokens=n,
                    contexts=(),
                    decode_rows=n,
                    graph=True,
                ),
            )
            for n, r in GRAPH_BATCHES_PADDED + GRAPH_BATCHES_UNPADDED
        )
        + tuple(
            Cell(
                why=f"indexer decode 32K, graph batch {n:>2}: positions up to 32767, "
                "the range corners, captured, replayed on new q and positions",
                spec=dict(
                    _INDEXER,
                    max_seq_len=MAX_SEQ_LEN,
                    corners=True,
                    tokens=n,
                    contexts=(),
                    decode_rows=n,
                    graph=True,
                ),
            )
            for n, _ in GRAPH_BATCHES_PADDED + GRAPH_BATCHES_UNPADDED
        )
        + (
            Cell(
                why="MLA context of 32768 rows at positions 0..32767: every position the "
                "NVFP4 target's envelope holds, one whole-prompt step, rank 1's heads, the corners",
                spec=dict(
                    _MLA,
                    max_seq_len=MAX_SEQ_LEN,
                    corners=True,
                    rank=1,
                    tokens=32768,
                    contexts=(32768,),
                    decode_rows=0,
                    graph=False,
                ),
            ),
            Cell(
                why="indexer context of 32768 rows at positions 0..32767: every position "
                "the NVFP4 target's envelope holds, one whole-prompt step, the corners",
                spec=dict(
                    _INDEXER,
                    max_seq_len=MAX_SEQ_LEN,
                    corners=True,
                    tokens=32768,
                    contexts=(32768,),
                    decode_rows=0,
                    graph=False,
                ),
            ),
            Cell(
                why="MLA mixed step of 32768 rows, 3 contexts + 13 decode rows (16 requests, "
                "max_num_tokens): positions are not the row index, decode rows up to 32767",
                spec=dict(
                    _MLA,
                    max_seq_len=MAX_SEQ_LEN,
                    corners=True,
                    rank=2,
                    tokens=32768,
                    contexts=(17000, 11000, 4755),
                    decode_rows=13,
                    graph=False,
                ),
            ),
            Cell(
                why="stress at 32768 rows, s in [-1, 1] and b in [-2, 2], q over every bf16 "
                "binade, positions 0..32767: cancellation, overflow, subnormals",
                spec=dict(
                    _MLA,
                    max_seq_len=MAX_SEQ_LEN,
                    corners=True,
                    rank=3,
                    s_range=(-1.0, 1.0),
                    b_range=(-2.0, 2.0),
                    q="wide",
                    tokens=32768,
                    contexts=(32768,),
                    decode_rows=0,
                    graph=False,
                ),
            ),
        )
        # dep4, attention data parallel: each rank runs its own requests with all 64 MLA heads
        # and the unsliced [64] s and b, over the same per-rank envelope (positions up to
        # 32767, steps of up to 32768 rows, decode batches 1..16 captured).
        + tuple(
            Cell(
                why=f"MLA dep4 decode 32K, graph batch {n:>2}, all 64 heads: positions up to "
                "32767, the range corners, captured, replayed on new q and positions",
                spec=dict(_MLA_DEP4, tokens=n, contexts=(), decode_rows=n, graph=True),
            )
            for n, _ in GRAPH_BATCHES_PADDED + GRAPH_BATCHES_UNPADDED
        )
        + (
            Cell(
                why="MLA dep4 context of 32768 rows, all 64 heads, positions 0..32767: every "
                "position of a rank's envelope in one whole-prompt step, the corners",
                spec=dict(_MLA_DEP4, tokens=32768, contexts=(32768,), decode_rows=0),
            ),
            Cell(
                why="MLA dep4 mixed step of 32768 rows, all 64 heads, 3 contexts + 13 decode "
                "rows (16 requests, max_num_tokens): positions are not the row index",
                spec=dict(_MLA_DEP4, tokens=32768, contexts=(17000, 11000, 4755), decode_rows=13),
            ),
            # `sweep_rows` (lo, hi, step) ranges and `sweep_pow2` (2^k - 1, 2^k, 2^k + 1)
            # name step sizes T, each driven as its own call on the context's leading T rows
            # and held bitwise to them; `large` puts a cell past the pre-merge budget.
            Cell(
                why="MLA dep4 step sizes T 1..4096, the top 1024 and a stride of 61 between, "
                "all 64 heads: each T bitwise the leading T rows of the 32768-row context "
                "call, every last-block tail of every launch",
                spec=dict(
                    _MLA_DEP4,
                    tokens=32768,
                    contexts=(32768,),
                    decode_rows=0,
                    sweep_rows=((1, 4096, 1), (4097, 31744, 61), (31745, 32768, 1)),
                    sweep_pow2=True,
                ),
            ),
            Cell(
                why="MLA dep4 every step size T 1..32768, all 64 heads (large, ~14 s): each T "
                "bitwise the leading T rows of the 32768-row context call",
                spec=dict(
                    _MLA_DEP4,
                    tokens=32768,
                    contexts=(32768,),
                    decode_rows=0,
                    sweep_rows=((1, 32768, 1),),
                    large=True,
                ),
            ),
        )
        # tep8 and tp8 (the "MLA tep8" cells; the two targets shard attention alike), the 64
        # heads over eight ranks with attention data parallel off: each rank holds 8 heads and
        # the [8] slices s[8 r : 8 r + 8], b[8 r : 8 r + 8], over the same per-rank envelope
        # (positions up to 32767, steps of up to 32768 rows, decode batches 1..16 captured).
        # The graph batches cycle the rank over 0..7, so every slice is driven.
        + tuple(
            Cell(
                why=f"MLA tep8 decode 32K, graph batch {n:>2}, rank {n % TEP8_RANKS}'s 8 heads: "
                "positions up to 32767, the range corners, captured, replayed on new q and "
                "positions",
                spec=dict(
                    _MLA_TEP8,
                    rank=n % TEP8_RANKS,
                    tokens=n,
                    contexts=(),
                    decode_rows=n,
                    graph=True,
                ),
            )
            for n in range(1, 17)
        )
        + (
            Cell(
                why="MLA tep8 context of 32768 rows at positions 0..32767, rank 1's 8 heads: "
                "every position of a rank's envelope in one whole-prompt step, the corners",
                spec=dict(
                    _MLA_TEP8, rank=1, tokens=32768, contexts=(32768,), decode_rows=0, graph=False
                ),
            ),
            Cell(
                why="MLA tep8 mixed step of 32768 rows, rank 2's 8 heads, 3 contexts + 13 "
                "decode rows (16 requests, max_num_tokens): positions are not the row index",
                spec=dict(
                    _MLA_TEP8,
                    rank=2,
                    tokens=32768,
                    contexts=(17000, 11000, 4755),
                    decode_rows=13,
                    graph=False,
                ),
            ),
            Cell(
                why="MLA tep8 last slice, rank 7's heads 56..63, a 32768-row context at "
                "positions 0..32767, the corners on heads 60..63; step sizes T 1..4096, the "
                "top 1024 and a stride of 61 between, each bitwise the leading T rows of that "
                "call: every last-block tail of every launch at 8 heads",
                spec=dict(
                    _MLA_TEP8,
                    rank=7,
                    tokens=32768,
                    contexts=(32768,),
                    decode_rows=0,
                    graph=False,
                    sweep_rows=((1, 4096, 1), (4097, 31744, 61), (31745, 32768, 1)),
                    sweep_pow2=True,
                ),
            ),
        )
        # tep4 and tp4 at the checkpoint's own context, max_seq_len 1048576 with chunked
        # prefill (max_num_tokens 32768), on the 16 MLA heads of a rank and the 32 replicated
        # indexer heads: decode rows up to position 1048575 (log 13.86, so the MLA factor
        # reaches 9.53 at the checkpoint's largest s and b), and steps of up to 32768 rows
        # whose contexts start at their cached prefix.
        + tuple(
            Cell(
                why=f"MLA decode 1M, graph batch {n:>2}, rank {r}'s heads: positions up to "
                "1048575, the range corners, captured, replayed on new q and positions",
                spec=dict(
                    _MLA,
                    max_seq_len=LONG_MAX_SEQ_LEN,
                    corners=True,
                    rank=r,
                    tokens=n,
                    contexts=(),
                    decode_rows=n,
                    graph=True,
                ),
            )
            for n, r in GRAPH_BATCHES_PADDED + GRAPH_BATCHES_UNPADDED
        )
        + tuple(
            Cell(
                why=f"indexer decode 1M, graph batch {n:>2}: positions up to 1048575, "
                "the range corners, captured, replayed on new q and positions",
                spec=dict(
                    _INDEXER,
                    max_seq_len=LONG_MAX_SEQ_LEN,
                    corners=True,
                    tokens=n,
                    contexts=(),
                    decode_rows=n,
                    graph=True,
                ),
            )
            for n, _ in GRAPH_BATCHES_PADDED + GRAPH_BATCHES_UNPADDED
        )
        + (
            Cell(
                why="MLA 1M last chunk, 32768 rows behind a 1015808-token cached prefix: "
                "positions 1015808..1048575, rank 1's heads, the corners",
                spec=dict(
                    _MLA,
                    max_seq_len=LONG_MAX_SEQ_LEN,
                    corners=True,
                    rank=1,
                    tokens=MAX_NUM_TOKENS,
                    contexts=(MAX_NUM_TOKENS,),
                    cached=(LONG_MAX_SEQ_LEN - MAX_NUM_TOKENS,),
                    decode_rows=0,
                    graph=False,
                ),
            ),
            Cell(
                why="indexer 1M last chunk, 32768 rows behind a 1015808-token cached prefix: "
                "positions 1015808..1048575, the corners",
                spec=dict(
                    _INDEXER,
                    max_seq_len=LONG_MAX_SEQ_LEN,
                    corners=True,
                    tokens=MAX_NUM_TOKENS,
                    contexts=(MAX_NUM_TOKENS,),
                    cached=(LONG_MAX_SEQ_LEN - MAX_NUM_TOKENS,),
                    decode_rows=0,
                    graph=False,
                ),
            ),
            Cell(
                why="MLA 1M mixed step of 32768 rows, 3 chunk tails + 13 decode rows: chunks "
                "at positions 1031576.., 500000.. and 0.., decode rows up to 1048575",
                spec=dict(
                    _MLA,
                    max_seq_len=LONG_MAX_SEQ_LEN,
                    corners=True,
                    rank=2,
                    tokens=MAX_NUM_TOKENS,
                    contexts=(17000, 11000, 4755),
                    cached=(LONG_MAX_SEQ_LEN - 17000, 500000, 0),
                    decode_rows=13,
                    graph=False,
                ),
            ),
            Cell(
                why="indexer 1M mixed step of 32768 rows, 3 chunk tails + 13 decode rows: "
                "chunks at positions 1031576.., 500000.. and 0.., decode rows up to 1048575",
                spec=dict(
                    _INDEXER,
                    max_seq_len=LONG_MAX_SEQ_LEN,
                    corners=True,
                    tokens=MAX_NUM_TOKENS,
                    contexts=(17000, 11000, 4755),
                    cached=(LONG_MAX_SEQ_LEN - 17000, 500000, 0),
                    decode_rows=13,
                    graph=False,
                ),
            ),
            Cell(
                why="MLA 1M packed step of 32768 rows, 4 chunks + 13 decode rows: chunks of "
                "16384, 8192, 4096 and 4083 rows at positions 1032192.., 524288.., 98304.. "
                "and 0.., rank 3's heads, decode rows up to 1048575",
                spec=dict(
                    _MLA,
                    max_seq_len=LONG_MAX_SEQ_LEN,
                    corners=True,
                    rank=3,
                    tokens=MAX_NUM_TOKENS,
                    contexts=(16384, 8192, 4096, 4083),
                    cached=(LONG_MAX_SEQ_LEN - 16384, 524288, 98304, 0),
                    decode_rows=13,
                    graph=False,
                ),
            ),
            Cell(
                why="indexer 1M packed step of 32768 rows, 4 chunks + 13 decode rows: chunks "
                "of 16384, 8192, 4096 and 4083 rows at positions 1032192.., 524288.., "
                "98304.. and 0.., decode rows up to 1048575",
                spec=dict(
                    _INDEXER,
                    max_seq_len=LONG_MAX_SEQ_LEN,
                    corners=True,
                    tokens=MAX_NUM_TOKENS,
                    contexts=(16384, 8192, 4096, 4083),
                    cached=(LONG_MAX_SEQ_LEN - 16384, 524288, 98304, 0),
                    decode_rows=13,
                    graph=False,
                ),
            ),
            Cell(
                why="stress at the 1M last chunk, s in [-1, 1] and b in [-2, 2], q over every "
                "bf16 binade, positions 1015808..1048575: cancellation, overflow, subnormals",
                spec=dict(
                    _MLA,
                    max_seq_len=LONG_MAX_SEQ_LEN,
                    corners=True,
                    rank=3,
                    s_range=(-1.0, 1.0),
                    b_range=(-2.0, 2.0),
                    q="wide",
                    tokens=MAX_NUM_TOKENS,
                    contexts=(MAX_NUM_TOKENS,),
                    cached=(LONG_MAX_SEQ_LEN - MAX_NUM_TOKENS,),
                    decode_rows=0,
                    graph=False,
                ),
            ),
            Cell(
                why="MLA every position 0..1048575 in 32 chunk calls of 32768 rows, rank 0's "
                "heads, the corners: the device log rounds to the correctly rounded log's "
                "bf16 at every position below 2^20",
                spec=dict(
                    _MLA,
                    max_seq_len=LONG_MAX_SEQ_LEN,
                    corners=True,
                    rank=0,
                    tokens=MAX_NUM_TOKENS,
                    contexts=(MAX_NUM_TOKENS,),
                    decode_rows=0,
                    graph=False,
                    position_sweep=(0, LONG_MAX_SEQ_LEN),
                ),
            ),
            Cell(
                why="indexer every position 0..1048575 in 32 chunk calls of 32768 rows, the "
                "corners: the device log rounds to the correctly rounded log's bf16 at every "
                "position below 2^20",
                spec=dict(
                    _INDEXER,
                    max_seq_len=LONG_MAX_SEQ_LEN,
                    corners=True,
                    tokens=MAX_NUM_TOKENS,
                    contexts=(MAX_NUM_TOKENS,),
                    decode_rows=0,
                    graph=False,
                    position_sweep=(0, LONG_MAX_SEQ_LEN),
                ),
            ),
            Cell(
                why="MLA layer 0 on rank 1, the NVFP4 checkpoint's own s and b, every position "
                "0..1048575 in 32 chunk calls: two heads' factors cross zero and end negative",
                spec=dict(
                    _MLA,
                    max_seq_len=LONG_MAX_SEQ_LEN,
                    rank=1,
                    tokens=MAX_NUM_TOKENS,
                    contexts=(MAX_NUM_TOKENS,),
                    decode_rows=0,
                    graph=False,
                    position_sweep=(0, LONG_MAX_SEQ_LEN),
                    slice_values=NVFP4_MLA_LAYER0_RANK1,
                ),
            ),
        )
        # dep4 at the checkpoint's own context, max_seq_len 1048576 with chunked prefill
        # (max_num_tokens 32768 per rank): all 64 MLA heads with the unsliced [64] s and b, and
        # the indexer's 32, at decode rows up to position 1048575 (log 13.86, so the MLA factor
        # reaches 9.53 at the checkpoint's largest s and b) and steps of up to 32768 rows whose
        # contexts start at their cached prefix. The indexer's 32 heads are replicated, the same
        # at every parallel layout, so dep4's indexer decode, last-chunk and mixed-step cells
        # are the tep4 and tp4 ones above (the same why and spec).
        + tuple(
            Cell(
                why=f"MLA dep4 decode 1M, graph batch {n:>2}, all 64 heads: positions up to "
                "1048575, the range corners, captured, replayed on new q and positions",
                spec=dict(_MLA_DEP4_1M, tokens=n, contexts=(), decode_rows=n, graph=True),
            )
            for n, _ in GRAPH_BATCHES_PADDED + GRAPH_BATCHES_UNPADDED
        )
        + (
            Cell(
                why="MLA dep4 1M last chunk, 32768 rows behind a 1015808-token cached prefix, "
                "all 64 heads: positions 1015808..1048575, the corners",
                spec=dict(
                    _MLA_DEP4_1M,
                    tokens=MAX_NUM_TOKENS,
                    contexts=(MAX_NUM_TOKENS,),
                    cached=(LONG_MAX_SEQ_LEN - MAX_NUM_TOKENS,),
                    decode_rows=0,
                ),
            ),
            Cell(
                why="MLA dep4 1M mixed step of 32768 rows, all 64 heads: a 32755-row chunk at "
                "positions 1015808..1048562 behind its cached prefix, 13 decode rows up to 1048575",
                spec=dict(
                    _MLA_DEP4_1M,
                    tokens=MAX_NUM_TOKENS,
                    contexts=(MAX_NUM_TOKENS - 13,),
                    cached=(LONG_MAX_SEQ_LEN - MAX_NUM_TOKENS,),
                    decode_rows=13,
                ),
            ),
            Cell(
                why="MLA dep4 1M mixed step, 3 chunk tails + 13 decode rows (32768 rows, all 64 "
                "heads): chunks at positions 1031576.., 500000.. and 0.., decode rows up to "
                "1048575",
                spec=dict(
                    _MLA_DEP4_1M,
                    tokens=MAX_NUM_TOKENS,
                    contexts=(17000, 11000, 4755),
                    cached=(LONG_MAX_SEQ_LEN - 17000, 500000, 0),
                    decode_rows=13,
                ),
            ),
            Cell(
                why="stress at the dep4 1M last chunk, all 64 heads, s in [-1, 1] and b in "
                "[-2, 2], q over every bf16 binade, positions 1015808..1048575: cancellation, "
                "overflow, subnormals",
                spec=dict(
                    _MLA_DEP4_1M,
                    s_range=(-1.0, 1.0),
                    b_range=(-2.0, 2.0),
                    q="wide",
                    tokens=MAX_NUM_TOKENS,
                    contexts=(MAX_NUM_TOKENS,),
                    cached=(LONG_MAX_SEQ_LEN - MAX_NUM_TOKENS,),
                    decode_rows=0,
                ),
            ),
            Cell(
                why="MLA dep4 1M prompt as its 32 chunks of 32768 rows, all 64 heads, the "
                "corners: every position 0..1048575, the device log rounding to the correctly "
                "rounded log's bf16 at each",
                spec=dict(
                    _MLA_DEP4_1M,
                    tokens=MAX_NUM_TOKENS,
                    contexts=(MAX_NUM_TOKENS,),
                    chunks=LONG_MAX_SEQ_LEN // MAX_NUM_TOKENS,
                    decode_rows=0,
                ),
            ),
            Cell(
                why="indexer 1M prompt as its 32 chunks of 32768 rows, the corners: every "
                "position 0..1048575, the device log rounding to the correctly rounded log's "
                "bf16 at each",
                spec=dict(
                    _INDEXER_1M,
                    tokens=MAX_NUM_TOKENS,
                    contexts=(MAX_NUM_TOKENS,),
                    chunks=LONG_MAX_SEQ_LEN // MAX_NUM_TOKENS,
                    decode_rows=0,
                ),
            ),
        )
        # tep8 at the checkpoint's own context, max_seq_len 1048576 with chunked prefill
        # (max_num_tokens 32768), on a rank's 8 MLA heads (rank r's [8] slices of s and b) and
        # the 32 replicated indexer heads: decode rows up to position 1048575 (log 13.86, so
        # the MLA factor reaches 9.53 at the checkpoint's largest s and b), and steps of up to
        # 32768 rows whose contexts start at their cached prefix. The graph batches cycle the
        # rank over 0..7, so every slice is driven at decode. The indexer's 32 heads are
        # replicated, the same at every parallel layout, so tep8's indexer decode, last-chunk,
        # mixed-step, packed-step and every-position cells are the tep4 and tp4 ones above (the
        # same why and spec).
        + tuple(
            Cell(
                why=f"MLA tep8 decode 1M, graph batch {n:>2}, rank {n % TEP8_RANKS}'s 8 heads: "
                "positions up to 1048575, the range corners, captured, replayed on new q and "
                "positions",
                spec=dict(
                    _MLA_TEP8_1M,
                    rank=n % TEP8_RANKS,
                    tokens=n,
                    contexts=(),
                    decode_rows=n,
                    graph=True,
                ),
            )
            for n in range(1, 17)
        )
        + (
            Cell(
                why="MLA tep8 1M last chunk, 32768 rows behind a 1015808-token cached prefix: "
                "positions 1015808..1048575, rank 1's 8 heads, the corners",
                spec=dict(
                    _MLA_TEP8_1M,
                    rank=1,
                    tokens=MAX_NUM_TOKENS,
                    contexts=(MAX_NUM_TOKENS,),
                    cached=(LONG_MAX_SEQ_LEN - MAX_NUM_TOKENS,),
                    decode_rows=0,
                    graph=False,
                ),
            ),
            Cell(
                why="MLA tep8 1M mixed step of 32768 rows, rank 2's 8 heads, 3 chunk tails + 13 "
                "decode rows: chunks at positions 1031576.., 500000.. and 0.., decode rows up "
                "to 1048575",
                spec=dict(
                    _MLA_TEP8_1M,
                    rank=2,
                    tokens=MAX_NUM_TOKENS,
                    contexts=(17000, 11000, 4755),
                    cached=(LONG_MAX_SEQ_LEN - 17000, 500000, 0),
                    decode_rows=13,
                    graph=False,
                ),
            ),
            Cell(
                why="MLA tep8 1M packed step of 32768 rows, rank 3's 8 heads, 4 chunks + 13 "
                "decode rows: chunks of 16384, 8192, 4096 and 4083 rows at positions "
                "1032192.., 524288.., 98304.. and 0.., decode rows up to 1048575",
                spec=dict(
                    _MLA_TEP8_1M,
                    rank=3,
                    tokens=MAX_NUM_TOKENS,
                    contexts=(16384, 8192, 4096, 4083),
                    cached=(LONG_MAX_SEQ_LEN - 16384, 524288, 98304, 0),
                    decode_rows=13,
                    graph=False,
                ),
            ),
            Cell(
                why="MLA tep8 stress at the 1M last chunk, rank 5's 8 heads, s in [-1, 1] and b "
                "in [-2, 2], q over every bf16 binade, positions 1015808..1048575: "
                "cancellation, overflow, subnormals",
                spec=dict(
                    _MLA_TEP8_1M,
                    rank=5,
                    s_range=(-1.0, 1.0),
                    b_range=(-2.0, 2.0),
                    q="wide",
                    tokens=MAX_NUM_TOKENS,
                    contexts=(MAX_NUM_TOKENS,),
                    cached=(LONG_MAX_SEQ_LEN - MAX_NUM_TOKENS,),
                    decode_rows=0,
                    graph=False,
                ),
            ),
            Cell(
                why="MLA tep8 every position 0..1048575 in 32 chunk calls of 32768 rows, rank "
                "7's 8 heads, the corners: the device log rounds to the correctly rounded "
                "log's bf16 at every position below 2^20",
                spec=dict(
                    _MLA_TEP8_1M,
                    rank=7,
                    tokens=MAX_NUM_TOKENS,
                    contexts=(MAX_NUM_TOKENS,),
                    decode_rows=0,
                    graph=False,
                    position_sweep=(0, LONG_MAX_SEQ_LEN),
                ),
            ),
            Cell(
                why="MLA tep8 layer 0 on rank 2, the NVFP4 checkpoint's own s and b on heads "
                "16..23, every position 0..1048575 in 32 chunk calls: head 20 (local head 4) "
                "crosses zero and ends negative",
                spec=dict(
                    _MLA_TEP8_1M,
                    corners=False,
                    rank=2,
                    tokens=MAX_NUM_TOKENS,
                    contexts=(MAX_NUM_TOKENS,),
                    decode_rows=0,
                    graph=False,
                    position_sweep=(0, LONG_MAX_SEQ_LEN),
                    slice_values=NVFP4_MLA_LAYER0_TEP8_RANK2,
                ),
            ),
            Cell(
                why="MLA tep8 layer 0 on rank 3, the NVFP4 checkpoint's own s and b on heads "
                "24..31, every position 0..1048575 in 32 chunk calls: head 26 (local head 2) "
                "crosses zero and ends negative",
                spec=dict(
                    _MLA_TEP8_1M,
                    corners=False,
                    rank=3,
                    tokens=MAX_NUM_TOKENS,
                    contexts=(MAX_NUM_TOKENS,),
                    decode_rows=0,
                    graph=False,
                    position_sweep=(0, LONG_MAX_SEQ_LEN),
                    slice_values=NVFP4_MLA_LAYER0_TEP8_RANK3,
                ),
            ),
        )
    )

    note = """
    Metadata consumed: the step's position ids, flattened to [T] (`position_ids`); nothing
    else. Out-of-place: `q`, `positions`, `s` and `b` are not mutated. Seven launches:
    positions to fp32, + 1, log, round to q's dtype, * s, + b, q * scale. `s` and `b` pass
    through `.to(q.dtype)`, which is free when they are already bf16 (the checkpoint's
    dtype); an fp32 s or b holding bf16 values gives the bf16 result bitwise (tested).
    `positions` is converted to fp32 by name, so a bf16 default dtype never reaches the
    log (tested). No host synchronization: a CUDA-graph replay on new q and positions is
    bitwise equal to the eager call at every captured batch 1..16, at 16 and 32 heads, at
    all 64 (dep4) and at 8 (tep8 and tp8, every rank's slice), and at tep8's 8 MLA and 32
    indexer heads at decode positions up to 1048575 (max_seq_len 1048576). Under capture the
    intermediates come from the graph's pool.

    Cost, measured on sm_103: at T <= 16 an eager call is launch-bound and a CUDA-graph
    replay of the call is cheaper than the eager call; at T 32768 the two cost about the
    same.

    Launches, profiled on sm_103 at 64 heads for every T 1..32768: seven per call, 128
    threads each, none with a grid-stride loop. positions to fp32 is an unrolled copy over
    [T], grid ceil(T / 512); + 1, the log and the rounding to bf16 are vectorized over [T],
    grid ceil(T / 1024); * s and + b run the offset-calculator kernel over [T, H, 1], grid
    ceil(T * H / 512) (+ b takes the vectorized kernel at T = 1 and the offset-calculator
    one from T = 2, the only kernel that changes with T); q * scale runs the
    offset-calculator kernel, grid T * H * D / 512. What T moves is each grid and its last
    block's tail, and T 1..4096 holds every tail of every launch: the dep4 step-size cell
    drives those, the top 1024 and a stride between, each bitwise the leading rows of the
    32768-row call, and its large cell drives every T 1..32768 the same way. At tep8's 8
    heads, profiled the same way: the same seven kernels and blocks; * s and + b take grid
    ceil(T / 64), and q * scale 2 T blocks, a row's 1024 elements filling two blocks whole,
    so that launch has no partial block; + b still changes kernel at T = 2, and nothing
    else changes with T. Every tail repeats with a period of 1024 in T, so T 1..4096 holds
    them all: the tep8 last-slice cell drives those, the top 1024 and a stride between,
    each bitwise the leading rows of its 32768-row call, and every T 1..32768 at 8 heads
    was driven the same way once (measured, 0 of 32768 step sizes differ; not a cell).

    Numerics: bitwise equal to `reference`, the specified bf16 chain, over every cell.
    The chain itself sits up to 2.8 bf16 ulp from the closed form at the MLA cells' drawn
    s and b over positions 0..9215, and up to 15.5 ulp over positions 0..32767 with the
    ranges' corners pinned (measured), where the cancellation corner (s -0.08, b 0.9)
    shrinks the factor to 0.068 at position 32767 (computed); both inside the bound the
    test derives. An fp32 scale with no intermediate rounding lands within 0.5 ulp of the
    closed form and differs from this op in about a third of the elements (37.4% at 9216
    rows, 36.4% at 32768, 35.5% at 32768 x 64 heads, measured): a replacement kernel has
    to reproduce the roundings, not the math. At 64 heads, with the corners on heads 0..3,
    the chain sits up to 15.5 bf16 ulp from the closed form over positions 0..32767, as at
    16 (measured), and so it does at 8 heads with the corners on four of a rank's 8 (15.494
    at rank 2's slice, 15.531 at rank 7's, where an fp32 scale rounded once differs from the
    op in 43.6% and 39.1% of the elements; measured). The device's fp32 log (CUDA logf, not
    correctly rounded) differs from the correctly rounded log in the last fp32 bit at 935 of
    the positions 0..32767 and at 39331 of the positions 0..1048575, and rounds to the same
    bf16 at every one of them: the context cells drive every position below 32768 one by one,
    and the six sweep cells (two at 16 MLA heads, the indexer's, and three at tep8's 8 MLA
    heads) and dep4's two 32-chunk cells every position below 2^20, where the test also holds
    the two logs' bf16 equal directly. Below 32768 they cannot part: logf is within 1 fp32
    ulp of the exact log, and the exact log(pos + 1) of every position 1..32767 lies at least
    1.748 fp32 ulp from a bf16 rounding midpoint (measured; the nearest is position 6510).
    Above it there is no such margin: 38 of the positions 32768..1048575 lie within 1 fp32
    ulp of a midpoint, the nearest 0.0458 ulp from it (position 1028641, in the last chunk),
    so their agreement is measured, not implied, and the sweep and 32-chunk cells are what
    hold it. Measured beyond every envelope, not a cell: the first position where the two
    round apart is 1695942, and 5 of the positions below 2^24 do.

    At the NVFP4 target's envelope the factor reaches 7.53125 at position 32767 with the
    checkpoint's largest s and b (0.574219, 1.57031; the closed form is 7.5406). Its
    largest error is at the checkpoint's deepest cancellation, s -0.078125 with b
    0.902344, where the factor falls to 0.0898 at position 32767: there the chain's factor
    lands up to 8.997 bf16 ulp (of the closed form's factor) from it over positions
    0..32767 (measured), because s * log(pos + 1) is rounded at its own magnitude, about
    nine times the factor's. That is what the chain specifies. (That s and b are the
    corner of the checkpoint's ranges; no one head holds both: the head with s -0.078125,
    MLA layer 0 head 26, has b 0.90625 and a closed-form factor of 0.0940 at position
    32767, computed.)

    At tep4, tp4 and tep8 over max_seq_len 1048576, the closed-form factor at position
    1048575 reaches 9.53067 on the checkpoint's largest head (MLA layer 19 head 15, rank 0's
    local head 15 at tep4 and tp4 and tep8 rank 1's local head 7, which holds both range
    tops, s 0.574219 and b 1.57031; read on disk). log(pos + 1) is 13.86 there and a bf16
    step of it is 0.0625: the chain's log takes two values over the last chunk
    1015808..1048575 (13.8125 and 13.875), and log(pos + 1) and log(pos + 2) round to
    different bf16 at only 276 of the positions 0..1048575, 56 of them above 32767 and one
    in the last chunk (position 1028640; measured). A position off by one therefore leaves
    this op's output bitwise unchanged at most far positions, where neither gate of the test
    can see it; the cells' near positions can. On the tep8 last chunk (rank 1's 8 heads) it
    moves one row of 32768, the one at 1028640 (or 1028641 for log(pos) in place of log(pos
    + 1)), and stays inside the closed form's bound (0.680 of it), so neither gate sees it on
    the other rows. There, s and b swapped land at 1723 times the bound and rank 2's 8 heads
    on rank 1 at 94.2 times it, and the rounding variants move 9.0% to 43.2% of the elements
    (measured). Over every position 0..1048575 and every head of the NVFP4 checkpoint, the
    chain's factor lands within 0.0558 of the closed form's on the MLA (0.0411 over
    0..32767) and within 0.00793 on the indexer, and within 2.249 bf16 ulp of it wherever the
    factor is at least 0.25 (measured).

    Two MLA heads of the NVFP4 checkpoint, and no other of its 72 x 64, have a factor that
    crosses zero below 1048576, both on layer 0: head 20 (s -0.0693359, b 0.914063) near
    position 531308 and head 26 (s -0.078125, b 0.90625) near 109097 (computed from the
    closed form). Both are in tp4 rank 1's slice (`NVFP4_MLA_LAYER0_RANK1`, local heads 4
    and 10), and one is in each of tep8 ranks 2 and 3's slices (`NVFP4_MLA_LAYER0_TEP8_RANK2`,
    rank 2's local head 4; `NVFP4_MLA_LAYER0_TEP8_RANK3`, rank 3's local head 2). There the
    chain's factor is exactly 0, a zeroed query, over 33359 positions (517233..550591) and
    6992 positions (108418..115409), never has the opposite sign to the closed form, and
    ends at -0.046875 and -0.179688 at position 1048575 (the closed form -0.0471377 and
    -0.176792; measured). That is what the chain specifies: the checkpoint-slice cells, tp4
    rank 1's and tep8 ranks 2 and 3's, drive both crossings at every position and the test
    holds those runs. No indexer head crosses below 1048576 (the earliest near position
    7.2e9, computed). The ranges' cancellation corner crosses zero too (s -0.080078, b
    0.898438 in bf16, near position 74570), its chain factor stepping from +2^-8 to -2^-8
    where the bf16 log steps from 11.1875 to 11.25, with no exactly-zero value between
    (computed; the MLA corner sweeps, at 16 heads and at tep8's 8, zero no query, measured):
    the MLA corner cells past it (the last chunks, the packed and mixed steps, the MLA
    sweeps through the crossing, at 16 heads and at tep8's 8) drive the negative factor, and
    the test's closed-form bound, which scales with |s * log(pos + 1)| and not with the
    factor, holds there.

    At dep4 over max_seq_len 1048576 every rank holds all 64 MLA heads, so every MLA head
    above, the two whose factor crosses zero included, is on every rank. At position
    1048575 the chain gives 9.5625 on the largest head (the closed form 9.53067), and the
    indexer's closed-form factor reaches 1.90887 (layer 1 head 19). One of the 276
    positions where log(pos + 1) and log(pos + 2) round to different bf16 lies in the last
    chunk: over the MLA's 1M last chunk at 64 heads, either off-by-one moves one row of the
    32768, at position 1028640 or 1028641, and stays inside the closed form's bound
    (tested). On the indexer the chain's factor lands within 1.645 bf16 ulp of the closed
    form's wherever the factor is at least 0.25 (measured). Over dep4's 1M cells the output
    lands at most 0.965 of the test's bound from the closed form (measured).

    The cells' range corners over positions 0..1048575, the factor alone (measured at dep4):
    on the MLA (s -0.080078 or 0.578125, b 0.898438 or 1.60156 in bf16) the chain's factor
    lands within 0.0571 of the closed form's and within 2.969 bf16 ulp of it wherever the
    factor is at least 0.25, at most 0.8005 of the test's bound on the factor (dC at
    `_closed_form_ratio`; 0.3881 over the last chunk); on the indexer (s -0.050049 or
    0.075195, b 0.96875 or 1.0625) within 0.0138, 1.746 ulp and 0.8538 of dC. At position
    1048575 the MLA corners' factor is 9.625 at the top corner (closed form 9.61608) and
    -0.210938 at the cancellation corner (closed form -0.211681). That corner (s -0.080078,
    b 0.898438) crosses zero near position 74570 (computed): the chain's factor is never 0
    there, and is -0.00390625 at the 57 positions 74514..74570, where the closed form is
    positive and at most 6.06e-05 (measured). The test's bound scales with
    |s * log(pos + 1)|, not with the factor, so it holds through the crossing, and dep4's MLA
    32-chunk cell drives it there, as the MLA sweep cells (at 16 heads and at tep8's 8) do.

    Rows are independent: a padded row cannot reach a live one. A negative position
    gives a non-finite scale on its own row only (-1: log 0 = -inf, an infinite scale;
    -2: nan).

    `positions` must be [T] and `s`, `b` [H]: `is_valid` refuses a q that is not 3-D,
    positions of any other shape, and s or b of any other shape. These broadcast silently
    in the op: s or b of length 1 scales every head by head 0's terms; positions of length
    1 scales every row at position 0; T = 1 with positions of length P returns P rows;
    when T == H, s and b as [H, 1] scale row t by head t's terms, and a 2-D q [T, H * D]
    returns [T, H, H * D]. Mismatches the broadcast cannot absorb raise from torch: an
    unsliced [64] s or b against 16 heads raises `RuntimeError: The size of tensor a (16)
    must match the size of tensor b (64) at non-singleton dimension 1`, and a rank's
    16-head slice against all 64 heads (a tep4 slice at dep4) raises `RuntimeError: The
    size of tensor a (64) must match the size of tensor b (16) at non-singleton dimension
    1`. At tep8's 8 heads, the unsliced [64] s or b raises `RuntimeError: The size of
    tensor a (8) must match the size of tensor b (64) at non-singleton dimension 1`, and a
    tep4 rank's 16-head slice `RuntimeError: The size of tensor a (8) must match the size
    of tensor b (16) at non-singleton dimension 1`; s and b as [8, 1] pair rows with heads
    silently at decode batch 8, a captured batch where T == H (all tested).

    Hard limits (sm_103), computed at the widest envelope the cells cover: positions <=
    1048575 (tep4, tp4, dep4 and tep8 at max_seq_len 1048576 with chunked prefill; 32767 at
    the other NVFP4 targets) and T <= 32768 per step (max_num_tokens 32768 per rank, chunked
    prefill or not); none is reached, none was probed:
      positions -> fp32   exact below 2^24             boundary 2^24 + 1        not reached
                          (positions + 1 <= 2^20)
      q.numel()           elementwise TensorIterator   boundary 2^31 elements   not reached
                          (32768 x 64 x 128 = 268435456 at dep4; 32768 x 32 x 128 =
                          134217728 on the indexer, 32768 x 16 x 128 = 67108864 on the
                          tp4 MLA, 32768 x 8 x 128 = 33554432 on the tep8 MLA)
      row t of q          t * H * D, H * D = 1024      boundary 2^31 / 8192 =   not reached
                          (MLA at tep8), 2048 (MLA     262144 rows at dep4,     (T <= 32768)
                          at tep4 and tp4), 4096       2^31 / 1024 = 2097152
                          (indexer), 8192 (MLA, dep4)  at tep8 (2^20 rows at
                                                       16 heads)
      grid of q * scale   T * H * D / 512 blocks       boundary 2^31 - 1        not reached
                          (524288 at 32768 x 64)
      positions           int32 values, no table       no pool or slot index    -
    The op holds no KV pool, page or slot table, so the pool size the engine allocates
    does not reach it, and the context length does not size any operand: a 1048576-token
    prompt reaches it as chunks of at most 32768 rows. One call over every position of a
    1M request would be [1048576, 16, 128] at 16 heads, exactly 2^31 elements, and
    [1048576, 32, 128] on the indexer, 2^32 elements, past the 2^31 boundary; no step of
    the envelope is either, and the sweep cells drive every position in 32768-row calls.

    Not certified: a q dtype other than bf16 (each step would round to that dtype instead);
    int64 positions (tested equal to int32 bitwise, in no cell); head_dim other than 128,
    heads other than 8, 16, 32 and 64; at 8 and 64 heads, s and b outside the
    checkpoint-shaped draw ranges and q over every bf16 binade below position 1015808 (the
    stress cells run at 16 heads, and at 64 and at 8 over dep4's and tep8's 1M last chunk);
    positions above 1048575 (the 1M cells at 16 MLA heads are tep4's and tp4's, at all 64
    dep4's, at 8 tep8's, and the indexer's serve all four); steps above 32768 rows; decode
    batches above 16; a non-contiguous q; torch.compile.
    """

    def __call__(
        self, q: torch.Tensor, positions: torch.Tensor, s: torch.Tensor, b: torch.Tensor
    ) -> torch.Tensor:
        log_pos = torch.log(positions.to(torch.float32) + 1).to(q.dtype)
        scale = log_pos[:, None, None] * s.to(q.dtype)[:, None] + b.to(q.dtype)[:, None]
        return q * scale

    def reference(
        self, q: torch.Tensor, positions: torch.Tensor, s: torch.Tensor, b: torch.Tensor
    ) -> torch.Tensor:
        """The specified chain, each step evaluated in fp64 and rounded to fp32, then to
        `q.dtype`: ``lp = rnd(log(pos + 1))``, ``m = rnd(lp * s)``, ``scale = rnd(m + b)``,
        ``out = rnd(q * scale)``.

        fp64 holds a product of two bf16 values exactly, and a sum of two to far less than
        half a bf16 ulp of the larger, so the two roundings land where torch's bf16
        arithmetic (fp32 math, rounded to bf16) lands: this is the chain, not an
        approximation of it. Its fp32 log is the correctly rounded one; the device's is
        not, and the test states why the two agree once rounded to bf16.
        """
        dt = q.dtype

        def rnd(x: torch.Tensor) -> torch.Tensor:
            return x.to(torch.float32).to(dt).double()

        lp = rnd(torch.log(positions.double() + 1))
        m = rnd(lp[:, None] * rnd(s.double())[None, :])
        scale = rnd(m + rnd(b.double())[None, :])
        return (q.double() * scale[:, :, None]).to(torch.float32).to(dt)

    def is_valid(
        self, q: torch.Tensor, positions: torch.Tensor, s: torch.Tensor, b: torch.Tensor
    ) -> None:
        if q.dim() != 3:
            raise ValueError(
                f"ssmax_scaling: q must be [T, H, D], got shape {tuple(q.shape)}; a 2-D q "
                "broadcasts against the [T, H, 1] scale and pairs rows with heads when T == H"
            )
        tokens, heads = q.shape[0], q.shape[1]
        if tuple(positions.shape) != (tokens,):
            raise ValueError(
                f"ssmax_scaling: positions must be [T] = [{tokens}], got shape "
                f"{tuple(positions.shape)}; a length-1 positions scales every row at one "
                "position, and a q of one row is broadcast to every position"
            )
        for name, t in (("s", s), ("b", b)):
            if tuple(t.shape) != (heads,):
                raise ValueError(
                    f"ssmax_scaling: {name} must be [H] = [{heads}], this rank's heads, got "
                    f"shape {tuple(t.shape)}; a length-1 {name} is broadcast over every head, "
                    f"and an [H, 1] {name} pairs rows with heads when T == H"
                )


ssmax_scaling = SSMaxScaling()
