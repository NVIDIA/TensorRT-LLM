# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""General matmul with optional fused bias via the trtllm cuBLASLt gemm op.

This target's copy of the catalog's `gemm/cublas_mm`, certified over every bf16
projection the Qwen3.8-Flash-Next-FP8 unified target makes."""

from __future__ import annotations

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*
from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import (
    Arch,
    Cell,
    OpWrapper,
    true_fp32_matmul,
)

#: Rows of the zero-padded `[down; block_inject; 0]` hyper-connection first-projection
#: weight (`down` 320 + `block_inject` 4 = 324 real rows, padded to a multiple of 128) and
#: the columns of it the projection multiplies (324 rounded up to 32).
HC_FIRST_ROWS, HC_FIRST_LIVE, HC_FIRST_COLS = 384, 324, 352

#: tep4's five trunk projections `(name, K, N, MTP-3 rows, MTP-6/7 rows)` and the rows their
#: cells run beyond 128. A verify step puts depth + 1 rows a request through the trunk over
#: the captured batches: MTP-3 160..512 (step 32, batches 40..128), past 512 MTP-6 560..896
#: (step 56, batches 80..128) and MTP-7 576..1024 (step 64, batches 72..128). cuBLASLt picks
#: the kernel per exact M: one row per kernel it runs over those rows for that shape
#: (measured; `note` lists the row runs), then the 8192-row prefill chunk, the largest row
#: the trunk sees at tep4. The draft layer's o_proj and shared expert are three of these
#: shapes, at the trunk's rows.
TEP4_TRUNK = (
    (
        "attn in-proj tp4/tep4 4352 wide",
        2560,
        4352,
        (160, 192, 224, 256, 288, 320, 384, 416, 512),
        (560, 576, 616, 672, 704, 784, 840, 1024),
    ),
    (
        "o_proj / GDN out_proj tp4, K 1536",
        1536,
        2560,
        (160, 224, 288, 320, 384, 416, 512),
        (576, 616, 672, 704, 784, 832, 1024),
    ),
    (
        "GDN in_proj tp4 4120 wide",
        2560,
        4120,
        (160, 192, 224, 256, 288, 320, 384, 416, 512),
        (560, 576, 616, 672, 704, 784, 832, 840, 1024),
    ),
    (
        "shared expert gate|up|logit 1288 wide",
        2560,
        1288,
        (160, 224, 288, 320, 384, 416, 512),
        (576, 616, 672, 704, 784, 832, 840, 896, 1024),
    ),
    (
        "shared expert down K 640",
        640,
        2560,
        (160, 224, 288, 320, 384, 416, 512),
        (576, 616, 672, 704, 784, 832, 1024),
    ),
)

#: The draft layer's tep4 projections on depth 7's first draft step, `(name, K, N, rows a
#: request, rows)`: 8 rows a request, and `fc_h` over the trunk's 4 hyper-connection streams
#: 32, past their 512-row cells over the captured batches (72..128; `fc_h` 24..128, up to
#: 4096 rows). One row per kernel no other cell of the shape runs there (measured; `note`
#: lists the runs), and for the three shapes no larger cell holds, the 8192-row draft
#: prefill chunk (a context chunk's draft-extend step).
TEP4_DRAFT = (
    ("MTP fc_e 2560 -> 2560", 2560, 2560, 8, (576, 640, 704, 832, 960)),
    ("MTP fc_h over 4 streams", 2560, 2560, 32, (1280, 1536, 1792, 2048, 2304, 2816, 3584)),
    ("MTP q|gate tep4 (6 heads) 3072 wide", 2560, 3072, 8, (640, 704, 832, 1024, 8192)),
    ("MTP k / v tep4 (1 kv head) 256 wide", 2560, 256, 8, (640, 8192)),
    ("MTP index q|k 640 wide", 2560, 640, 8, (640, 704, 768, 832, 960, 1024, 8192)),
)

#: The decode batches every attention-TP member captures (runtime_facts.yaml).
CAPTURED_BATCHES = (1, 2, 4, 8, *range(16, 129, 8))

#: Rows of a prefill chunk at max_num_tokens.
PREFILL_CHUNK = 8192


def _proj(m: int, k: int, n: int, out_dtype=None, bias: bool = False, **extra) -> dict:
    """One bf16 projection: `[m, k] @ weight.t()` with `weight` a row-major `[n, k]`."""
    return dict(m=m, k=k, n=n, dtype=torch.bfloat16, bias=bias, out_dtype=out_dtype, **extra)


def _verify_site(m: int) -> str:
    """The tep4 trunk step that runs `m` rows: a prefill chunk, an MTP-3 verify (4 rows a
    request) up to 512 rows, past it the MTP-7 and MTP-6 verifies (8 and 7 rows a request)
    of the captured batches that make `m`."""
    if m == PREFILL_CHUNK:
        return "a max_num_tokens prefill chunk"
    if m <= 512:
        return f"MTP-3 verify of batch {m // 4}"
    sites = [
        f"MTP-{d} verify of batch {m // (d + 1)}"
        for d in (7, 6)
        if m % (d + 1) == 0 and m // (d + 1) in CAPTURED_BATCHES
    ]
    assert sites, f"no captured batch's MTP-6 or MTP-7 verify runs {m} rows"
    return " / ".join(sites)


def _verify_rows(name: str, k: int, n: int, m: int) -> Cell:
    """One tep4 trunk projection at `m` verify rows, or at a prefill chunk."""
    return Cell(why=f"{name}, {m} rows: {_verify_site(m)}", spec=_proj(m, k, n))


def _draft_rows(name: str, k: int, n: int, per_request: int, m: int) -> Cell:
    """One tep4 draft-layer projection on depth 7's first draft step (`per_request` rows a
    request), or on a prefill chunk's draft-extend step."""
    if m == PREFILL_CHUNK:
        site = "a max_num_tokens draft prefill chunk"
    else:
        assert m % per_request == 0 and m // per_request in CAPTURED_BATCHES, (m, per_request)
        site = f"MTP-7 first draft step of batch {m // per_request}"
    return Cell(why=f"{name}, {m} rows: {site}", spec=_proj(m, k, n))


def _hc_first(m: int) -> dict:
    """The fp32 HC first-projection logits: `HC_FIRST_COLS` columns of the padded weight's
    `.t()`, a column window whose column stride is still K."""
    return _proj(m, 10240, HC_FIRST_COLS, torch.float32, rows=HC_FIRST_ROWS, live=HC_FIRST_LIVE)


class CublasMm(OpWrapper):
    """`mat_a @ mat_b (+ bias)`, fp32-accumulated, one cublas_mm call.

    `mat_b` is read column-major, which is what a stored weight's `.t()` already
    is -- the targets pass `weight.t()` and no transpose kernel runs.
    """

    ARCHS = frozenset({Arch.SM_103})

    # The catalog entry's own cell, then every bf16 projection of the unified target
    # (hidden 2560): the tp2 attention input projection swept over the row counts a rank
    # sees (decode batch 1, the largest captured decode batch 128, an MTP-7 verify of 128
    # requests, a max_num_tokens prefill chunk), every other shape at 128 rows unless its
    # site only runs prefill, the draft layer's projections at an MTP-3 verify's 512 rows,
    # tep4's five trunk projections at the MTP-3, MTP-6 and MTP-7 verify rows past 128 and
    # at a prefill chunk, tep4's draft layer on MTP-7's first draft step past 512 rows and
    # at a draft prefill chunk, the vision tower's biased linears at 5580 patches (1395
    # after its 2x2 merge), and the envelope's largest call: the draft fc_h over 4 streams
    # of an 8192-row chunk.
    # Per-member widths: tp1 and the attention-DP members hold every head, tp2/tep2 half,
    # tp4/tep4 a quarter.
    CELLS: tuple[Cell, ...] = (
        Cell(
            why="decode-shaped: one token, the shape every projection sees per step",
            spec=dict(m=1, k=7168, n=1536, dtype=torch.bfloat16, bias=False, out_dtype=None),
        ),
        # Attention fused input projection [q|gate, k, v, index q|k, pad]: 14080 / 7424 / 4352.
        Cell(why="attn in-proj tp2 7424 wide: decode batch 1", spec=_proj(1, 2560, 7424)),
        Cell(why="attn in-proj tp2 7424 wide: decode batch 128", spec=_proj(128, 2560, 7424)),
        Cell(
            why="attn in-proj tp2 7424 wide: MTP-7 verify 1024 rows", spec=_proj(1024, 2560, 7424)
        ),
        Cell(
            why="attn in-proj tp2 7424 wide: prefill chunk 8192 rows", spec=_proj(8192, 2560, 7424)
        ),
        Cell(why="attn in-proj tp1/dep 14080 wide, 128 rows", spec=_proj(128, 2560, 14080)),
        Cell(why="attn in-proj tp4/tep4 4352 wide, 128 rows", spec=_proj(128, 2560, 4352)),
        # Attention o_proj and GDN out_proj share K: 24 q heads x 256 = 48 v heads x 128.
        Cell(why="o_proj / GDN out_proj tp1/dep, K 6144, 128 rows", spec=_proj(128, 6144, 2560)),
        Cell(why="o_proj / GDN out_proj tp2, K 3072, 128 rows", spec=_proj(128, 3072, 2560)),
        Cell(why="o_proj / GDN out_proj tp4, K 1536, 128 rows", spec=_proj(128, 1536, 2560)),
        Cell(why="o_proj / GDN out_proj tp4, K 1536, decode batch 1", spec=_proj(1, 1536, 2560)),
        # GDN input projection [q|k, v, z, a, b]: 16480 / 8240 / 4120.
        Cell(why="GDN in_proj tp1/dep 16480 wide, 128 rows", spec=_proj(128, 2560, 16480)),
        Cell(why="GDN in_proj tp2 8240 wide, 128 rows", spec=_proj(128, 2560, 8240)),
        Cell(why="GDN in_proj tp4 4120 wide, 128 rows", spec=_proj(128, 2560, 4120)),
        # MoE router (replicated, 512 experts) and the shared expert.
        Cell(why="MoE router 512 experts: decode batch 1", spec=_proj(1, 2560, 512)),
        Cell(why="MoE router 512 experts: 128 rows", spec=_proj(128, 2560, 512)),
        Cell(why="MoE router 512 experts: prefill chunk 8192 rows", spec=_proj(8192, 2560, 512)),
        Cell(why="shared expert gate|up|logit 1288 wide, 128 rows", spec=_proj(128, 2560, 1288)),
        Cell(why="shared expert down K 640, 128 rows", spec=_proj(128, 640, 2560)),
        # HC first projection: fp32 logits over a column window of the padded weight.
        Cell(why="HC first fp32 logits, 352-col window: 1 row", spec=_hc_first(1)),
        Cell(why="HC first fp32 logits, 352-col window: 128 rows", spec=_hc_first(128)),
        Cell(why="HC first fp32 logits, 352-col window: 8192 rows", spec=_hc_first(8192)),
        # The captured decode batches between 1 and 128: tp1/tp2/tep2/tp4/tep4 capture
        # [1, 2, 4, 8, 16, 24, ..., 128], dep2 up to 64 rows a rank and dep4 up to 32. Each
        # decode shape at one of them between 16 and 64 rows; `note` lists which captured
        # rows each shape's cells hold (the kernel is picked per exact M).
        Cell(why="attn in-proj tp2 7424 wide: decode batch 16", spec=_proj(16, 2560, 7424)),
        Cell(why="attn in-proj tp4/tep4 4352 wide: decode batch 24", spec=_proj(24, 2560, 4352)),
        Cell(
            why="attn in-proj tp1/dep 14080 wide: dep4's largest batch 32",
            spec=_proj(32, 2560, 14080),
        ),
        Cell(
            why="o_proj / GDN out_proj tp1/dep, K 6144: decode batch 16", spec=_proj(16, 6144, 2560)
        ),
        Cell(why="o_proj / GDN out_proj tp2, K 3072: decode batch 32", spec=_proj(32, 3072, 2560)),
        Cell(why="o_proj / GDN out_proj tp4, K 1536: decode batch 48", spec=_proj(48, 1536, 2560)),
        Cell(why="GDN in_proj tp1/dep 16480 wide: decode batch 32", spec=_proj(32, 2560, 16480)),
        Cell(why="GDN in_proj tp2 8240 wide: dep2's largest batch 64", spec=_proj(64, 2560, 8240)),
        Cell(why="GDN in_proj tp4 4120 wide: decode batch 64", spec=_proj(64, 2560, 4120)),
        Cell(why="MoE router 512 experts: decode batch 40", spec=_proj(40, 2560, 512)),
        Cell(
            why="shared expert gate|up|logit 1288 wide: decode batch 56", spec=_proj(56, 2560, 1288)
        ),
        Cell(why="shared expert down K 640: decode batch 24", spec=_proj(24, 640, 2560)),
        Cell(why="HC first fp32 logits, 352-col window: 32 rows", spec=_hc_first(32)),
        # tep4's trunk under MTP-3: each of its five projections at one verify row per kernel
        # cuBLASLt runs over 160..512 rows (`TEP4_TRUNK`), and at an 8192-row prefill chunk.
        *(
            _verify_rows(name, k, n, m)
            for name, k, n, rows, _ in TEP4_TRUNK
            for m in (*rows, PREFILL_CHUNK)
        ),
        # ... and under MTP-6 / MTP-7: one verify row per kernel over 560..1024 rows.
        *(_verify_rows(name, k, n, m) for name, k, n, _, deep in TEP4_TRUNK for m in deep),
        # PLE key / value projections; the draft layer's fc_e has the value's shape.
        Cell(why="PLE key_proj 2560 -> 10240, 128 rows", spec=_proj(128, 2560, 10240)),
        Cell(why="PLE value_proj / MTP fc_e 2560 -> 2560, 128 rows", spec=_proj(128, 2560, 2560)),
        # The MTP draft layer at an MTP-3 verify of 128 requests (512 rows).
        Cell(why="MTP fc_h over 4 streams of 128 rows: 512 rows", spec=_proj(512, 2560, 2560)),
        Cell(why="MTP q|gate tep4 (6 heads) 3072 wide, 512 rows", spec=_proj(512, 2560, 3072)),
        Cell(why="MTP q|gate dep4 (24 heads) 12288 wide, 512 rows", spec=_proj(512, 2560, 12288)),
        Cell(why="MTP k / v tep4 (1 kv head) 256 wide, 512 rows", spec=_proj(512, 2560, 256)),
        Cell(why="MTP k / v dep4 (2 kv heads) 512 wide, 512 rows", spec=_proj(512, 2560, 512)),
        Cell(why="MTP index q|k 640 wide, 512 rows", spec=_proj(512, 2560, 640)),
        # The draft layer on depth 7's first draft step past those 512-row cells
        # (`TEP4_DRAFT`), and the draft-only shapes at an 8192-row draft prefill chunk.
        *(
            _draft_rows(name, k, n, per_request, m)
            for name, k, n, per_request, rows in TEP4_DRAFT
            for m in rows
        ),
        Cell(
            why="envelope: MTP fc_h over 4 streams of an 8192-row chunk, 32768 rows",
            spec=_proj(32768, 2560, 2560),
        ),
        # Vision tower (eager, biased), 5580 patches merged 2x2 into 1395 rows.
        Cell(
            why="vision qkv 1152 -> 3456 with bias, 5580 rows",
            spec=_proj(5580, 1152, 3456, bias=True),
        ),
        Cell(
            why="vision attn proj 1152 -> 1152 with bias, 5580 rows",
            spec=_proj(5580, 1152, 1152, bias=True),
        ),
        Cell(
            why="vision mlp fc1 1152 -> 4304 with bias, 5580 rows",
            spec=_proj(5580, 1152, 4304, bias=True),
        ),
        Cell(
            why="vision mlp fc2 4304 -> 1152 with bias, 5580 rows",
            spec=_proj(5580, 4304, 1152, bias=True),
        ),
        Cell(
            why="vision merger fc1 4608 -> 4608 with bias, 1395 rows",
            spec=_proj(1395, 4608, 4608, bias=True),
        ),
        Cell(
            why="vision merger fc2 4608 -> 2560 with bias, 1395 rows",
            spec=_proj(1395, 4608, 2560, bias=True),
        ),
    )

    note = """
    Source: compiled-in C++, `torch.ops.trtllm.cublas_mm` from
    `cpp/tensorrt_llm/thop/cublasScaledMM.cpp` (`cublas_mm` -> `cublas_mm_out` ->
    `cublas_gemm_caller`), one cuBLASLt matmul; nothing is JIT-compiled. Metadata
    consumed: none.

    Layout. The caller fixes both leading dimensions to K (`lda = ldb = K`) and
    the op checks only the unit strides (`mat_a.stride(1) == 1`,
    `mat_b.stride(0) == 1`). So `mat_b` may be a column window of a weight's
    `.t()` -- the HC first projection passes the first 352 of the padded
    `[384, 10240]` weight's columns, column stride still K, and is certified so
    -- but a K-slice of one (`weight.t()[:K']`, column stride the full K) or a
    row-strided `mat_a` is read as dense and answers a different product with no
    error. `is_valid` refuses both; the test holds the K-slice's silent miss.

    Kernel choice: bf16 in, bf16 or fp32 out, looks up `cublas_lut::bf16_algo_list`
    by `(max(next_pow2(M), 8), K, N)` (`find_special_algo`). That list holds only
    K 7168 / N 256, none of this target's shapes, so every call here passes no
    algo and cuBLASLt's heuristic picks the kernel for the exact M -- and it
    changes kernel inside a power-of-two bucket. Measured with the profiler over
    the 19 captured decode batches 1..128 (not re-driven by the test): the 13
    decode shapes run 52 distinct nvjet kernels between them, the 14080-wide
    attention input projection nine (rows 1..8, 16, 24..32, 40..48, 56..64,
    72..80, 88..96, 104..112, 120..128 each run their own). Over tep4's MTP-3
    verify rows 160..512 (step 32) the five tep4 trunk shapes switch as often:
    the attention and GDN input projections nine kernels each, the other three
    seven, all but one new to their shape (the shared gate|up runs its 40..64
    kernel at 160..192). Over the MTP-6 / MTP-7 verify rows past 512 (560, 576,
    616, 640, 672, 704, 728, 768, 784, 832, 840, 896, 960, 1024) they run eight
    (attention input), seven (o_proj / out_proj, shared down) and nine (GDN
    input, shared gate|up) kernels, every one new to its shape. On depth 7's
    first draft step the draft layer's own shapes run 13 kernels over the
    2560 x 2560 fc_e (576..1024) and fc_h (768..4096) rows, five over q|gate
    3072, two over k / v 256 and seven over index q|k 640, all but four new to
    their shape (fc_h at 3840 runs the 32768-row cell's kernel; q|gate, k / v
    and index q|k at 576, and k / v at 960..1024, their 512-row cells'). The
    choice is a function of the shape and M alone: repeated on fresh operands it
    does not move, and the five trunk shapes' choices at 1..512 and 8192 rows,
    profiled in two separate runs, matched. A cell certifies its own M; the
    kernel it runs is the only one it speaks for.

    Workspace: one per (device, stream), allocated on the first call on a stream
    and reused after. The first call on a capture stream therefore allocates it
    inside the capture. Captured and replayed on a fresh input, the call writes
    exactly the eager call's bits -- at decode batch 1 and 128 over the attention
    and GDN projections, the router and the fp32 HC logits, at an MTP-3 and an
    MTP-7 verify of batch 128 (512 and 1024 rows) over tep4's five trunk
    projections, at the draft layer's q|gate and index q|k on MTP-7's first draft
    step of batch 128 (1024 rows), and with the shared expert's two GEMMs forked
    onto a side stream and joined back (the test holds twenty-two such cases).

    Measured on GB300 (sm_103): every cell within one bf16 rounding flip of the
    correctly rounded fp32 product -- at worst 1.94 ulp of the row max
    element-wise and 0.051 ulp RMS, against the test's 4 / 0.5 gate (the decode
    cells at 16..64 rows at most 1.40 / 0.036; tep4's verify cells at 160..512
    rows 1.60 / 0.033, at 560..1024 rows 1.86 / 0.031, its 8192-row chunks
    1.86 / 0.030; the draft layer's cells past 512 rows 1.82 / 0.035); the fp32
    HC logits at most 0.006 / 0.003. Bias (vision) is added in the fp32 epilogue
    before the store, as `reference` does.

    Decode rows held, per shape (captured batches; a row is held when a cell of
    that shape runs the kernel the row selects, as measured above):
      attn in-proj 14080 (tp1/dep)   24..32, 120..128
      attn in-proj 7424 (tp2/tep2)   1..8, 16, 104..128
      attn in-proj 4352 (tp4/tep4)   1..8, 24..32, 104..128
      o_proj / out_proj K 6144       16, 120..128
      o_proj / out_proj K 3072       2..32, 104..128
      o_proj / out_proj K 1536       1..8, 40..64, 104..128
      GDN in_proj 16480              24..32, 104..128
      GDN in_proj 8240               56..64, 104..128
      GDN in_proj 4120               40..64, 104..128
      router 512                     1..64, 104..128
      shared expert gate|up 1288     40..64, 104..128
      shared expert down K 640       24..32, 104..128
      HC first fp32 window 352       1..32, 104..128
    An MTP-3 verify of batches 1..32 runs 4..128 rows and an MTP-7 verify of
    batches 1..16 8..128 rows, captured decode rows all, and follows this list
    (MTP-7 of batches 24..64 runs 192..512 rows, MTP-3's below). tep4's verify
    rows past 128 are all held for its five trunk shapes; per shape, the rows
    each kernel runs, held by the cell at that row or at the row in parentheses:
      attn in-proj 4352    160, 192, 224, 256, 288, 320, 352..384 (384), 416,
                           448..512 (512)
      o_proj / out K 1536  160..192 (160), 224..256 (224), 288, 320,
                           352..384 (384), 416..448 (416), 480..512 (512)
      GDN in_proj 4120     the same runs and cells as attn in-proj 4352
      shared gate|up 1288  160..192 (160), 224..256 (224), 288, 320,
                           352..384 (384), 416..448 (416), 480..512 (512)
      shared down K 640    as o_proj / out K 1536
    and each of the five at an 8192-row prefill chunk. Past 512 rows, the MTP-6
    (7 rows a request, batches 80..128) and MTP-7 (8, batches 72..128) verify
    rows, 896 being both, likewise:
      attn in-proj 4352    560, 576, 616..640 (616), 672, 704..768 (704), 784,
                           832 (1024), 840, 896..1024 (1024)
      o_proj / out K 1536  560..576 (576), 616..640 (616), 672, 704..768 (704),
                           784, 832..896 (832), 960..1024 (1024)
      GDN in_proj 4120     560, 576, 616..640 (616), 672, 704..768 (704), 784,
                           832, 840, 896 (832), 960..1024 (1024)
      shared gate|up 1288  560..576 (576), 616..640 (616), 672, 704..768 (704),
                           784, 832, 840, 896, 960..1024 (1024)
      shared down K 640    as o_proj / out K 1536
    The draft layer's first draft step at depth 7 (8 rows a request, fc_h 32)
    past its 512-row cells, over the captured batches 72..128 (fc_h 24..128):
      fc_e 2560 -> 2560    576, 640, 704..768 (704), 832..896 (832),
                           960..1024 (960)
      fc_h, the same shape 768 (704), 1024 (960), 1280, 1536, 1792, 2048, 2304,
                           2560 (1280), 2816, 3072 (2048), 3328 (1792), 3584,
                           3840 (32768), 4096 (2816)
      q|gate 3072          576 (512), 640, 704..768 (704), 832..960 (832), 1024
      k / v 256            576 (512), 640..896 (640), 960..1024 (512)
      index q|k 640        576 (512), 640, 704, 768, 832..896 (832), 960, 1024
    and q|gate, k / v and index q|k at an 8192-row draft prefill chunk. Its
    o_proj and shared expert are the trunk's K 1536 / 1288 / K 640 shapes at the
    trunk's rows, held above; the trunk's PLE value projection (2560 -> 2560) at
    its MTP-7 verify rows past 512 is fc_e's shape at fc_e's rows.
    A held row other than a cell's own M runs a certified kernel at a row count
    the test does not drive.

    `output_buffer_kind` and `group` are the op's multi-stream and collective
    arguments. This target leaves them at their defaults, so neither is a cell
    axis and neither is certified here.

    Not certified: every captured decode row outside its shape's held list above
    (each runs a kernel no cell of that shape runs); any other exact M -- verify
    rows past 128 of every shape but tep4's five trunk projections and the PLE
    value projection's MTP-7 rows past 512 (the router, the HC first window, the
    PLE key projection; tp1 / tp2 shapes under MTP, but for the 1024-row 7424
    cell); tep4's MTP-6 verify rows of 512 or fewer that are neither decode nor
    MTP-3 rows (7, 14, 28, 168, 280, 336, 392, 504: not profiled); the verify rows of
    the depths runtime_facts.yaml does not record at tep4 (2, 3, 5 or 6 rows a
    request, up to 768); the draft layer at rows its cells do not hold (its
    later draft steps at one row a request, fc_h at four; its first step at
    depth 7 up to 512 rows, and at every other depth); prefill chunks other than
    8192 (a context chunk's tail, a mixed context and verify batch), dep's
    gathered rows; vision patch counts other than 5580 / 1395; fp16 or fp32
    inputs; the registered fake; capture at other shapes or rows; sm_100.
    """

    def __call__(
        self,
        mat_a: torch.Tensor,
        mat_b: torch.Tensor,
        bias: torch.Tensor | None = None,
        out_dtype: torch.dtype | None = None,
        output_buffer_kind: int = 0,
        group: list[int] | None = None,
    ) -> torch.Tensor:
        return torch.ops.trtllm.cublas_mm(mat_a, mat_b, bias, out_dtype, output_buffer_kind, group)

    def reference(
        self,
        mat_a: torch.Tensor,
        mat_b: torch.Tensor,
        bias: torch.Tensor | None = None,
        out_dtype: torch.dtype | None = None,
        output_buffer_kind: int = 0,
        group: list[int] | None = None,
    ) -> torch.Tensor:
        """fp32-accumulated, TF32 off -- see `true_fp32_matmul`."""
        with true_fp32_matmul():
            ref = mat_a.float() @ mat_b.float()
        if bias is not None:
            ref = ref + bias.float()
        return ref.to(out_dtype if out_dtype is not None else mat_a.dtype)

    def is_valid(
        self,
        mat_a: torch.Tensor,
        mat_b: torch.Tensor,
        bias: torch.Tensor | None = None,
        out_dtype: torch.dtype | None = None,
        output_buffer_kind: int = 0,
        group: list[int] | None = None,
    ) -> None:
        # The kernel reads mat_a as dense row-major and mat_b as dense
        # column-major; other layouts produce silently wrong results.
        assert mat_a.is_contiguous(), "mat_a must be dense row-major [M, K]"
        assert mat_b.stride(0) == 1 and mat_b.stride(1) == mat_b.shape[0], (
            "mat_b must be dense column-major [K, N] (e.g. weight.t())"
        )
        if bias is not None:
            out_dt = out_dtype if out_dtype is not None else mat_a.dtype
            # A bias in the wrong dtype or shape is accepted by the op and
            # produces silently wrong results; with fp32 inputs the bias is
            # accepted but silently ignored.
            assert mat_a.dtype != torch.float32, "bias is silently ignored for fp32 inputs"
            assert bias.dtype == out_dt, "bias dtype must equal the output dtype"
            assert bias.shape == (mat_b.shape[1],) and bias.is_contiguous(), (
                "bias must be a contiguous [N] tensor"
            )


cublas_mm = CublasMm()
