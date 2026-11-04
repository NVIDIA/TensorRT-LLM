# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Dynamic 1x128 FP8 quantization of a bf16 activation, with packed UE8M0 block scales.

One power-of-two scale per 1x128 block, packed four to an int32 in the MN-major layout
DeepGEMM's FP8 GEMM reads.
"""

from __future__ import annotations

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*

from .._op import Arch, Cell, OpWrapper

#: Largest finite e4m3 magnitude.
E4M3_MAX = 448.0

#: The kernel's amax floor, compared in fp32: an all-zero block takes the scale 2**-42.
AMAX_FLOOR = 1e-10

#: A UE8M0 byte is a bare fp32 exponent: byte `b` encodes 2**(b - 127).
UE8M0_BIAS = 127

#: The UE8M0 cast saturates at the largest finite exponent, 2**127 (byte 254). Only an
#: infinite amax reaches it: the largest bf16 over 448 is below 2**120.
UE8M0_MAX_EXP = 127

#: Scale bytes per int32 word: the word in column `c` holds K blocks 4c .. 4c + 3.
BLOCKS_PER_WORD = 4


def scale_leading_dim(m: int) -> int:
    """The scale's column stride: M rounded up to a multiple of 4 (16-byte TMA alignment)."""
    return (m + 3) // 4 * 4


def block_exponents(input: torch.Tensor) -> torch.Tensor:
    """`e [M, ceil(K/128)]` int64: block `b` of row `t` is quantized with the scale `2**e`.

    `e = ceil(log2(amax / 448))`, exactly: `amax` is the block's largest `|x|` with NaN
    skipped (a short last block counts only its own columns), floored at `AMAX_FLOOR`, and
    `e` saturates at `UE8M0_MAX_EXP`. `frexp` of the fp64 quotient finds the ceiling without
    a `log2`, so a block whose amax is exactly 448 * 2**j takes `e = j`.
    """
    m, k = input.shape
    nb = (k + 127) // 128
    mag = input.float().abs()
    mag = torch.where(mag.isnan(), torch.zeros_like(mag), mag)
    mag = torch.nn.functional.pad(mag, (0, nb * 128 - k))
    amax = mag.reshape(m, nb, 128).amax(dim=2).clamp(min=AMAX_FLOOR)
    mant, exp = torch.frexp(amax.double() / E4M3_MAX)
    e = torch.where(mant == 0.5, exp - 1, exp).long()
    return torch.where(amax.isinf(), UE8M0_MAX_EXP, e).clamp(max=UE8M0_MAX_EXP)


#: The widths Mistral Large 4's FP8 128x128 block-scale linears quantize at tep4, and their
#: sites (the same per-rank shapes in the FP8 and the NVFP4 checkpoints): hidden 6144 feeds
#: wq, the indexer's wq_b and the dense and shared-expert gate_up; the dense layer-0 down
#: reads 6144 too (TP4 of 24576); wo reads 2048 (16 heads x 128 per rank); the shared-expert
#: down reads 1536 (TP4 of 6144).
_SITES = {
    6144: "wq, indexer wq_b, gate_up, dense down",
    2048: "wo",
    1536: "shared-expert down",
}

#: The widths Mistral Large 4 adds at dep4. Attention DP keeps all 64 heads and the dense
#: layer-0 MLP whole on every rank, each over the rank's own tokens: wo reads the whole
#: 64 x 128 attention output and the dense layer-0 down the whole SwiGLU output of
#: intermediate 24576. dep4's other sites (wq, indexer wq_b, the dense and shared-expert
#: gate_up, the shared-expert down) read hidden 6144.
_DEP4_SITES = {
    8192: "wo at dep4, all 64 heads",
    24576: "dense down at dep4, replicated",
}

#: The widths Mistral Large 4 adds at tep8 and tp8, which shard everything but the routed experts
#: alike. Tensor parallel 8 (attention DP off) slices eight ways: wo reads this rank's 8 of the
#: 64 heads x 128, the dense layer-0 down 24576 / 8 and the shared-expert down 6144 / 8. 768 is
#: six blocks: its second scale word holds two, and its bytes 2 and 3 are the empty slots. The
#: two targets' other sites (wq, indexer wq_b, the dense and shared-expert gate_up) read hidden
#: 6144.
_TEP8_SITES = {
    1024: "wo at tep8 and tp8, 8 heads",
    3072: "dense down at tep8 and tp8, 24576 / 8",
    768: "shared-expert down at tep8 and tp8, 6144 / 8",
}

#: Every width with cells, and its sites.
_WIDTHS = {**_SITES, **_DEP4_SITES, **_TEP8_SITES}

#: The CUDA-graph decode batches (max_batch_size 16, one row per request): the gates
#: capture 1/2/4/8/16 padded; the LLM API's unpadded default captures every batch 1..16.
_GRAPH_BATCHES = tuple(range(1, 17))
_PADDED_BATCHES = (1, 2, 4, 8, 16)

#: max_num_tokens of the NVFP4 target: chunked prefill is off, so one context step carries
#: up to 32768 rows (one whole 32K prompt, or up to 16 packed ones).
_MAX_TOKENS = 32768

#: max_num_tokens of the FP8 Mistral Large 4 target, which calls this op at the same widths.
_FP8_TARGET_MAX_TOKENS = 9216

#: The eager row counts each dep4 width is driven at, up to max_num_tokens (per rank under
#: attention DP), and what each one reaches.
_DEP4_EAGER_ROWS = (
    (3, "inside one 4-row pad unit, scale stride 4"),
    (5, "spills into a second pad unit, scale stride 8"),
    (6, "two zeroed pad slots per scale column"),
    (1023, "M % 4 == 3, one short prompt"),
    (4097, "M % 4 == 1 past 4096 rows"),
    (9215, "M % 4 == 3 past 8192 rows"),
    (9216, "M % 4 == 0 past 8192 rows"),
    (16385, "M % 4 == 1 past 16384 rows"),
    (_MAX_TOKENS - 1, "one row under max_num_tokens, a pad slot per scale column"),
    (_MAX_TOKENS, "max_num_tokens, one whole 32K prompt on the rank in one step"),
)

#: The eager row counts each tep8 and tp8 width is driven at, up to max_num_tokens (every rank sees
#: every token of the step), and what each one reaches.
_TEP8_EAGER_ROWS = (
    (3, "inside one 4-row pad unit, scale stride 4"),
    (5, "spills into a second pad unit, scale stride 8"),
    (1023, "M % 4 == 3, one short prompt"),
    (4097, "M % 4 == 1 past 4096 rows"),
    (_MAX_TOKENS - 1, "one row under max_num_tokens, a pad slot per scale column"),
    (_MAX_TOKENS, "max_num_tokens, one whole 32K prompt in one step"),
)


class Fp8Quantize1x128PackedUe8m0(OpWrapper):
    """Quantize a bf16 `[M, K]` activation to e4m3, one power-of-two scale per 1x128 row block.

    Per row `t` and block `b` of 128 consecutive columns: `s = 2**ceil(log2(amax / 448))`,
    `amax = max |x[t, block]|` floored at 1e-10, and `codes[t, block] = e4m3(x / s)`. As `s`
    is a power of two at or above `amax / 448`, the division is exact and every code lies in
    +-448: e4m3's round-to-nearest-even is the only rounding. Returns `(codes, scale)`:

    * `codes`: `[M, K]` float8_e4m3fn, contiguous -- a leading-row view of an allocation
      padded to a multiple of 4 rows.
    * `scale`: `[M, ceil(K / 512)]` int32 with strides `(1, align(M, 4))`, the MN-major
      layout `tensorrt_llm.deep_gemm.fp8_gemm_nt` takes for the activation scale (what
      `tensorrt_llm.deep_gemm.get_mn_major_tma_aligned_packed_ue8m0_tensor` makes of fp32
      power-of-two scales). Byte `j` (little-endian) of `scale[t, c]` is the biased exponent
      `e + 127` of K block `4c + j`; the scale is `2**(byte - 127)`, and byte slots past the
      last block hold 0. The tensor is not dense: its storage holds
      `(cols - 1) * align(M, 4) + M` int32s, so read it through its own strides (index it,
      or `.contiguous()`); an `as_strided` over the full `cols * align(M, 4)` rectangle is
      out of bounds whenever `M % 4 != 0`. The padding slots inside the storage hold 0.

    The pair feeds `torch.ops.trtllm.fp8_prequantized_swap_ab_gemm(codes, scale, weight,
    weight_scale, ..., disable_ue8m0_cast=True)`. `torch.ops.trtllm.fp8_swap_ab_gemm` makes
    this same call itself on the sm_100 family before its GEMM, when its autotuner picks the
    CUDA quantize tactic and `TRTLLM_FUSED_FP8_QUANT_PACK` is at its default `1`
    (`_fp8_quantize_1x128_ue8m0` in `_torch/custom_ops/torch_custom_ops.py`).

    The call passes `use_r128c4_layout=False`. The op's schema default, True, is the sm_107
    CuTe layout: a 1-D uint8 R128c4 buffer that `fp8_prequantized_swap_ab_gemm` refuses
    ("dimensionality of sizes (1) must match dimensionality of strides (2)"), so this entry
    does not expose the flag.

    Fusion boundary: the amax reduction, the UE8M0 rounding, the e4m3 cast and the scale
    packing are inside; nothing before (norm, SwiGLU) or after (the GEMM) is fused.
    """

    ARCHS = frozenset({Arch.SM_103})

    CELLS: tuple[Cell, ...] = tuple(
        Cell(
            why=f"graph batch {m}, K {k}: "
            f"{'padded' if m in _PADDED_BATCHES else 'unpadded default'} capture, replayed on "
            f"new rows ({_WIDTHS[k]})",
            spec=dict(m=m, k=k, graph=True),
        )
        for k in _WIDTHS
        for m in _GRAPH_BATCHES
    ) + (
        Cell(
            why="eager M 3: inside one 4-row pad unit, scale stride 4",
            spec=dict(m=3, k=6144, graph=False),
        ),
        Cell(
            why="eager M 5: spills into a second pad unit, scale stride 8",
            spec=dict(m=5, k=2048, graph=False),
        ),
        Cell(
            why="eager M 6: two zeroed pad slots per scale column",
            spec=dict(m=6, k=1536, graph=False),
        ),
        Cell(
            why="eager M 1023: a prefill chunk, M % 4 == 3",
            spec=dict(m=1023, k=6144, graph=False),
        ),
        Cell(
            why="eager M 4097: M % 4 == 1 past 4096 rows",
            spec=dict(m=4097, k=1536, graph=False),
        ),
        Cell(
            why="eager M 9215: one row under the FP8 target's max_num_tokens",
            spec=dict(m=_FP8_TARGET_MAX_TOKENS - 1, k=2048, graph=False),
        ),
        *(
            Cell(
                why=f"eager M 9216 at K {k}: the FP8 target's max_num_tokens ({_SITES[k]})",
                spec=dict(m=_FP8_TARGET_MAX_TOKENS, k=k, graph=False),
            )
            for k in _SITES
        ),
        Cell(
            why="eager M 16385: M % 4 == 1 past 16384 rows",
            spec=dict(m=16385, k=1536, graph=False),
        ),
        *(
            Cell(
                why=f"eager M 32767 at K {k}: one row under max_num_tokens, a pad slot per "
                "scale column",
                spec=dict(m=_MAX_TOKENS - 1, k=k, graph=False),
            )
            for k in _SITES
        ),
        *(
            Cell(
                why=f"eager M 32768 at K {k}: max_num_tokens, one whole 32K prompt in one "
                f"step ({_SITES[k]})",
                spec=dict(m=_MAX_TOKENS, k=k, graph=False),
            )
            for k in _SITES
        ),
        *(
            Cell(
                why=f"eager M {m} at K {k}: {what} ({_DEP4_SITES[k]})",
                spec=dict(m=m, k=k, graph=False),
            )
            for k in _DEP4_SITES
            for m, what in _DEP4_EAGER_ROWS
        ),
        *(
            Cell(
                why=f"eager M {m} at K {k}: {what} ({_TEP8_SITES[k]})",
                spec=dict(m=m, k=k, graph=False),
            )
            for k in _TEP8_SITES
            for m, what in _TEP8_EAGER_ROWS
        ),
    )

    note = """
    Metadata consumed: none. Runs on the current stream. The launch uses programmatic
    dependent launch unless `TRTLLM_ENABLE_PDL=0` (`getEnvEnablePDL` in
    `cpp/tensorrt_llm/common/envUtils.cpp` turns it on from sm_90); the codes and scales
    were measured byte-identical with it off (again at K 1024, 3072, 768 and 6144, at eight
    M from 1 to 32768).

    Preconditions, each refused loudly by the op (the test drives every one): `input` bf16
    ("Input matrix dtype must be BF16."), 2-D ("input must be a matrix"; flatten [B, S, H]
    yourself), contiguous ("self must be contiguous" -- a column slice of a fused
    projection is refused, not read through its stride), on CUDA, and `K % 16 == 0`
    ("self.sizes()[1] must be a multiple of 16, but got ..."). `M == 0` raises as well:
    "The specified pointer resides on host memory and is not registered with any CUDA
    device." (inferred: the empty scale buffer has no device pointer to wrap), so a caller
    with no rows skips the call. The op also checks for an sm_100-family device ("currently
    only supports SM100-family GPUs"), read in `thop/fp8Quantize.cpp`, not probed.

    Alignment: the kernel loads the input in 32-byte vectors (read in
    `fp8_blockscale_quant_packed.cu`). A contiguous view 8 elements (16 bytes) into its
    allocation was measured correct; one 1 element in faults with "CUDA error: misaligned
    address", which leaves the CUDA context unusable. A fresh allocation, or a row slice of
    a contiguous `[*, K]` tensor with `K % 16 == 0`, is always aligned.

    Edge behaviour, measured, and reproduced by `reference`: an all-zero block takes byte 85
    (2**-42, the floor) and zero codes; NaN is skipped by the amax and its own code is NaN
    (0x7f); an inf saturates its block to byte 254 (2**127), its own code to +-448 and every
    finite code in the block to +-0; -0.0 keeps its sign (0x80).

    A K that is not a multiple of 512 ends in a partial scale word; K 768 (six blocks, two
    words) is the one with cells. In each row's second word the lanes past K load zeros, store
    no codes, and have their bytes masked out of the word (read in
    `fp8_blockscale_quant_packed.cu`), so bytes 2 and 3 of `scale[:, 1]` are 0. Measured over
    an output allocation filled with 0xFF first: the empty bytes and the pad slots read 0,
    while the code rows past M, which the kernel never writes, kept 0xFF.

    Not certified: K outside {6144, 2048, 1536, 8192, 24576, 1024, 3072, 768} -- the op
    takes any `K % 16 == 0`, with a short last block and zero byte slots past it, but no
    cell holds a K with a short last block, and of the K whose last scale word is partial
    only 768 has cells; M above 32768 (two probes past it, below); the
    `use_r128c4_layout=True` layout, which this entry never requests.

    Hard limits (sm_103). Read in `fp8_blockscale_quant_packed.cu` and `thop/fp8Quantize.cpp`:
    the kernel forms the row index, the input and code offsets `row * K + col` and the
    scale offset `col * align(M, 4) + row` in int64; `M`, `K`, `align(M, 4)` (the scale
    stride) and the grid are int32. One warp per row and 512 columns (one scale word), four
    rows per block; K sets only grid x, one block column per scale word (12 at K 6144, 16 at
    K 8192, 48 at K 24576; 2 at K 1024 and 768, 6 at K 3072); the row blocks fold into grid
    z past 65535 blocks (M > 262140).
      row * K         input and code offsets, int64
                      boundary 2^31 elements: 2^31 / K = 349525.3 rows at K 6144,
                      1048576 at K 2048, 1398101.3 at K 1536, 262144 at K 8192,
                      87381.3 at K 24576, 2097152 at K 1024, 699050.7 at K 3072 and
                      2796202.7 at K 768; the envelope's largest is
                      32768 * 24576 = 805,306,368 elements (tep8's and tp8's 32768 *
                      3072 = 100,663,296), not reached
                      probed at M 349526, K 6144 (2,147,487,744 elements, grid z 2) and at
                      M 87382, K 24576 (2,147,500,032 elements, grid (48, 21846, 1)): passed.
                      Every row bitwise equal to `reference`; the rows straddling element
                      2^31 (and at K 6144 the grid-z seam) bitwise equal to the op on those
                      rows alone. Probes outside the envelope, not cells
      col * align(M, 4) + row
                      scale word offset, int64. At M 32768 the largest is
                      11 * 32768 + 32767 = 393,215 (K 6144), 131,071 (K 2048),
                      98,303 (K 1536), 524,287 (K 8192), 1,572,863 (K 24576), 65,535
                      (K 1024 and 768) and 196,607 (K 3072). In int32 it would first wrap
                      at M 178,956,969 (K 6144), 536,870,913 (K 2048), 715,827,881
                      (K 1536), 134,217,729 (K 8192), 44,739,241 (K 24576), 1,073,741,825
                      (K 1024 and 768) and 357,913,941 (K 3072): not reached. Probed at
                      M 349526, K 6144: largest offset 4,194,333, and at M 87382, K 24576:
                      largest offset 4,194,429, passed (above)
      align(M, 4)     the int32 scale stride: wraps for M > 2^31 - 4, which the op's
                      "M must be within int32" check (read, not probed) lets through for
                      M 2^31 - 3 .. 2^31 - 1. Computed, not probed: past any envelope
    """

    def __call__(self, input: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.ops.trtllm.fp8_quantize_1x128_packed_ue8m0(input, use_r128c4_layout=False)

    def reference(self, input: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """`(codes, scale)` from the exact power-of-two scales, laid out as the op lays them out."""
        m, k = input.shape
        e = block_exponents(input)
        nb = e.shape[1]
        cols = (nb + BLOCKS_PER_WORD - 1) // BLOCKS_PER_WORD
        s = torch.ldexp(torch.ones_like(e, dtype=torch.float64), e)
        q = (input.double() / s.repeat_interleave(128, dim=1)[:, :k]).float()
        codes = q.clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn)
        byte = torch.nn.functional.pad(e + UE8M0_BIAS, (0, cols * BLOCKS_PER_WORD - nb))
        words = byte.to(torch.uint8).reshape(m, cols, BLOCKS_PER_WORD).view(torch.int32)
        scale = torch.empty_strided(
            (m, cols), (1, scale_leading_dim(m)), dtype=torch.int32, device=input.device
        )
        scale.copy_(words.squeeze(-1))
        return codes, scale


fp8_quantize_1x128_packed_ue8m0 = Fp8Quantize1x128PackedUe8m0()
