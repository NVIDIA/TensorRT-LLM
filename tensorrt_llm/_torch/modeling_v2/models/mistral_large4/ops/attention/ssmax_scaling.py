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

from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper

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
#: deployments serve with chunked prefill: a step still carries at most MAX_NUM_TOKENS rows per
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
        Cell(
            why="MLA 16h (tep4/tp4) context 32K",
            spec=dict(
                _MLA,
                tokens=MAX_SEQ_LEN,
                rank=1,
                max_seq_len=MAX_SEQ_LEN,
                contexts=(MAX_SEQ_LEN,),
                decode_rows=0,
                corners=True,
                graph=False,
            ),
        ),
        Cell(
            why="MLA 64h (dep4) context 32K",
            spec=dict(
                _MLA_DEP4,
                tokens=MAX_SEQ_LEN,
                contexts=(MAX_SEQ_LEN,),
                decode_rows=0,
            ),
        ),
        Cell(
            why="MLA 8h (tep8/tp8) 1M last chunk behind cached prefix",
            spec=dict(
                _MLA_TEP8_1M,
                tokens=MAX_NUM_TOKENS,
                rank=1,
                contexts=(MAX_NUM_TOKENS,),
                cached=(LONG_MAX_SEQ_LEN - MAX_NUM_TOKENS,),
                decode_rows=0,
                graph=False,
            ),
        ),
        Cell(
            why="MLA 16h decode graph batch 1 (captured)",
            spec=dict(
                _MLA,
                tokens=1,
                rank=0,
                max_seq_len=LONG_MAX_SEQ_LEN,
                contexts=(),
                decode_rows=1,
                corners=True,
                graph=True,
            ),
        ),
        Cell(
            why="MLA 16h decode graph batch 16 (captured)",
            spec=dict(
                _MLA,
                tokens=16,
                rank=3,
                max_seq_len=LONG_MAX_SEQ_LEN,
                contexts=(),
                decode_rows=16,
                corners=True,
                graph=True,
            ),
        ),
        Cell(
            why="indexer 32h context 32K",
            spec=dict(
                _INDEXER,
                tokens=MAX_SEQ_LEN,
                max_seq_len=MAX_SEQ_LEN,
                contexts=(MAX_SEQ_LEN,),
                decode_rows=0,
                corners=True,
                graph=False,
            ),
        ),
        Cell(
            why="indexer 32h decode graph batch 16 (captured)",
            spec=dict(
                _INDEXER,
                tokens=16,
                max_seq_len=LONG_MAX_SEQ_LEN,
                contexts=(),
                decode_rows=16,
                corners=True,
                graph=True,
            ),
        ),
        Cell(
            why="MLA 16h mixed step: 3 cached chunks + 13 decode rows",
            spec=dict(
                _MLA,
                tokens=32768,
                rank=2,
                max_seq_len=LONG_MAX_SEQ_LEN,
                contexts=(16384, 8192, 8179),
                cached=(1032192, 524288, 0),
                decode_rows=13,
                corners=True,
                graph=False,
            ),
        ),
        Cell(
            why="stress: s in [-1, 1], b in [-2, 2], q over all bf16 binades",
            spec=dict(
                _MLA,
                tokens=4096,
                rank=0,
                max_seq_len=LONG_MAX_SEQ_LEN,
                contexts=(4096,),
                cached=(LONG_MAX_SEQ_LEN - 4096,),
                decode_rows=0,
                s_range=(-1.0, 1.0),
                b_range=(-2.0, 2.0),
                q="wide",
                corners=True,
                graph=False,
            ),
        ),
    )

    note = """
    SSMax query scaling `q * (s * log(pos + 1) + b)` in bf16 rounding for MLA (8/16/64 heads)
    and DSA indexer (32 heads) queries up to 1M positions. `is_valid` refuses silent broadcast
    when `s` or `b` does not match `q.shape[1]`.
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
