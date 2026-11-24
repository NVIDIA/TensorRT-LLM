# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""ModelOpt MXFP8 W8A8 linear on sm_107: the BF16 activation quantized to MXFP8, then the dense GEMM.

The ModelOpt export stores the KDA projections and the shared experts as E4M3 weights with one
UE8M0 scale per 32 values of a row: the "K32" scales, a row-major ``[N, K / 32]`` byte matrix,
which the weight tables relay at load to the flat R128c4 buffer the GEMM reads. The activation
takes the same recipe at run time (``rubin_mxfp8_quantize``, bitwise the catalog ``mxfp8_quantize``
with swizzled scales), and ``rubin_mxfp8_gemm`` multiplies the two.
"""

from __future__ import annotations

import functools

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

from .rubin_mxfp8_gemm import (
    SF_VEC,
    mxfp8_scales_to_r128c4,
    quantize_mxfp8_reference,
    rubin_mxfp8_gemm,
)

#: The GEMM writes 16-byte BF16 rows, so N is a multiple of 8.
N_MULTIPLE = 8


def _pad_up(value: int, multiple: int) -> int:
    return (value + multiple - 1) // multiple * multiple


def mxfp8_weight_scales_shape(n: int, k: int) -> tuple[int]:
    """The relaid scales' shape for an ``[n, k]`` weight: ``(pad128(n) * pad4(k / 32),)``."""
    return (_pad_up(n, 128) * _pad_up(k // SF_VEC, 4),)


@functools.cache
def _quantize():
    """The MXFP8 activation quantization (R128c4 scales); imported on first use (it needs the GPU
    stack)."""
    from .rubin_mxfp8_quantize import rubin_mxfp8_quantize

    return rubin_mxfp8_quantize


#: Rows up to which the shared low-M GEMM (``fp8_block_scale_low_m_gemm``, 1x32 weight scales)
#: replaces quantize + dense GEMM: at 256 rows it is 0.80-0.96x of them on the K3.1 projections.
LOW_M_GEMM_MAX_M = 128
#: Weight shapes ``(N, K)`` that keep the mid-M GEMM past ``LOW_M_GEMM_MAX_M``, up to these rows
#: (sm_107, 144 rows: the helix8 verify step b24 k5). TP8 KDA stack on a producer's MXFP8: 7.99
#: vs 8.90 us (BF16 in about even, 10.3-10.8 vs 10.7 us); TP8 shared [gate | up] BF16 in: 7.55 vs
#: 8.31 us. Both entries move together, so the prequantized one stays bitwise the BF16 one.
MID_M_GEMM_MAX_M = {(6288, 7168): 160, (1536, 7168): 160}


def low_m_gemm_max_m(n: int, k: int) -> int:
    """Rows up to which both entries run the shared low-M / mid-M GEMM for an ``[n, k]`` weight."""
    return max(LOW_M_GEMM_MAX_M, MID_M_GEMM_MAX_M.get((n, k), 0))


@functools.cache
def _low_m():
    """The shared low-M / mid-M FP8 GEMM, or None where it does not run (no CUDA device, or one
    the kernel does not support)."""
    if not torch.cuda.is_available():
        return None
    from tensorrt_llm._torch.kernels.blackwell import fp8_block_scale_low_m_gemm as low_m

    return low_m if low_m._is_supported_device(torch.cuda.current_device()) else None


def _mnk(m: int, n: int, k: int, why: str) -> Cell:
    return Cell(why=why, spec=dict(m=m, n=n, k=k))


class RubinMxfp8Linear(OpWrapper):
    """``input [M, K] x weight [N, K]^T -> [M, N]`` BF16 for an MXFP8 weight and a BF16 input.

    ``weight`` is E4M3. ``weight_scale`` is its relaid scales: the flat R128c4 buffer (the bytes
    of the engine's ``block_scale_interleave``), ``pad128(N) * pad4(K / 32)`` UE8M0 bytes. The
    input is quantized with one power-of-two scale per 32 values, the smallest at or above
    ``amax / 448``; the product accumulates in FP32 and rounds once to BF16.
    """

    ARCHS: frozenset[Arch] = frozenset()

    # The calls of the specialized targets. The KDA stack is [q | k | v | g] (96 heads x 128 each,
    # split by the attention TP size), f_a (128 rows, whole) and b (one row per head), zero rows
    # to a multiple of 8. The shared experts are a TP8 share on TEP8 and whole elsewhere.
    CELLS: tuple[Cell, ...] = (
        _mnk(1, 6288, 7168, "KDA stack at TP8 (6144 + 128 + 12 rows, padded to 6288), one row"),
        _mnk(8, 6288, 7168, "KDA stack at TP8, the TEP8 verify step b1 k7 (8 rows)"),
        _mnk(96, 6288, 7168, "KDA stack at TP8, the helix8 verify step b16 k5 (96 rows)"),
        _mnk(144, 6288, 7168, "KDA stack at TP8, the helix8 verify step b24 k5 (mid-M, 144 rows)"),
        _mnk(2954, 6288, 7168, "KDA stack at TP8, a TEP8 prefill chunk (last M tile partial)"),
        _mnk(96, 49376, 7168, "KDA stack at DEP16 (49152 + 128 + 96 rows), verify b12 k7"),
        _mnk(2954, 49376, 7168, "KDA stack at DEP16, a prefill chunk"),
        _mnk(8, 7168, 1536, "KDA o_proj at TP8: K = 12 heads x 128 (48 scale blocks), 8 rows"),
        _mnk(96, 7168, 1536, "KDA o_proj at TP8, the helix8 verify step (96 rows)"),
        _mnk(2954, 7168, 1536, "KDA o_proj at TP8, a TEP8 prefill chunk"),
        _mnk(96, 7168, 12288, "KDA o_proj at DEP16: K = 96 heads x 128, 96 rows"),
        _mnk(2954, 7168, 12288, "KDA o_proj at DEP16, a prefill chunk"),
        _mnk(8, 1536, 7168, "shared experts [gate | up] at TP8 (2 x 768 rows), 8 rows"),
        _mnk(144, 1536, 7168, "shared experts [gate | up] at TP8, helix8 b24 k5 (mid-M, 144 rows)"),
        _mnk(8, 7168, 768, "shared experts down at TP8: K = 768 (24 scale blocks), 8 rows"),
        _mnk(2954, 1536, 7168, "shared experts [gate | up] at TP8, a TEP8 prefill chunk"),
        _mnk(96, 12288, 7168, "shared experts [gate | up] whole (2 x 6144 rows), 96 rows"),
        _mnk(96, 7168, 6144, "shared experts down whole: K = 6144, 96 rows"),
        _mnk(2954, 12288, 7168, "shared experts [gate | up] whole, a DEP16 prefill chunk"),
        _mnk(96, 3584, 7168, "dep16 latent_down under moe_latent_mxfp8 (pinned), 96 rows"),
        _mnk(96, 28672, 7168, "dep16 drafter MLP gate_up under draft_mxfp8 (pinned), 96 rows"),
        _mnk(96, 7168, 14336, "dep16 drafter MLP down under draft_mxfp8: K = 14336, 96 rows"),
    )

    note = """
    Up to LOW_M_GEMM_MAX_M rows (a MID_M_GEMM_MAX_M shape: up to its rows; N a multiple of 16,
    neither pinned, reserve_sms nor dense): one call of the shared low-M GEMM, which
    quantizes the activation inside the kernel (mid-M rows: its own quantization kernel chained
    with PDL) and reads the 1x32 weight scales; same recipe, FP32 accumulation in another order.
    Otherwise two kernels: rubin_mxfp8_quantize (one CuTe DSL kernel, swizzled scales, bitwise the
    catalog mxfp8_quantize) and the dense block-scaled GEMM.
    The weight scales are the flat R128c4 buffer the GEMM reads. A 2-D scale is refused: a
    checkpoint's row-major [N, K / 32] matrix has the zero-padded R128c4 matrix's shape whenever
    N % 128 == 0 and K / 32 % 4 == 0 (o_proj, the shared experts). Relay it at load.
    dep16's MXFP8 levers (MLA g_proj and o_proj: the shared [gate | up] and KDA o_proj shapes;
    latent_down; the drafter's o_proj, gate_up and down) pass pinned=True: rubin_mxfp8_gemm's
    split-K-free runner and pinned 96-row tactics.
    N must be a multiple of 8, so a projection with fewer rows per rank (the KDA b_proj at TP8 has
    12) runs stacked with other projections. The first call per device compiles; see
    rubin_mxfp8_gemm for the CUDA graph and autotuner notes, and for ``reserve_sms`` (SMs kept out
    of the GEMM's grid; the output is the same). ``dense=True`` keeps a call on the dense GEMM
    at every row count: dep16's shared [gate | up], which reserves SMs on graph steps only, so
    that its graph and in-order steps write the same bytes.
    """

    def __call__(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
        pinned: bool = False,
        reserve_sms: int = 0,
        *,
        dense: bool = False,
    ) -> torch.Tensor:
        m, n = input.shape[0], weight.shape[0]
        # Pinned tactics and reserved SMs are rubin_mxfp8_gemm's: those calls stay on it, as does
        # a ``dense`` call (one whose other launches reserve SMs: the same bytes with or without).
        shared = not (pinned or reserve_sms or dense) and input.is_cuda
        rows = low_m_gemm_max_m(n, input.shape[1])
        low_m = _low_m() if shared and m <= rows and n % 16 == 0 else None
        if low_m is not None:
            out = torch.empty((m, n), dtype=torch.bfloat16, device=input.device)
            return low_m.run_low_m_gemm(input, weight, weight_scale, out)
        act, act_scale = _quantize()(input)
        return rubin_mxfp8_gemm(
            act, weight, act_scale, weight_scale, pinned=pinned, reserve_sms=reserve_sms
        )

    def reference(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
        pinned: bool = False,
        reserve_sms: int = 0,
        *,
        dense: bool = False,
    ) -> torch.Tensor:
        """The input quantized as ``mxfp8_quantize`` does, then ``rubin_mxfp8_gemm.reference``;
        ``reserve_sms`` and ``dense`` do not enter it."""
        values, scales = quantize_mxfp8_reference(input)
        return rubin_mxfp8_gemm.reference(
            values, weight, mxfp8_scales_to_r128c4(scales), weight_scale
        )

    def is_valid(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
        pinned: bool = False,
        reserve_sms: int = 0,
        *,
        dense: bool = False,
    ) -> None:
        assert isinstance(pinned, bool) and isinstance(dense, bool), "pinned and dense are flags"
        # The quantization reads K in whole blocks and the GEMM runner checks only scale sizes;
        # each of these reads as a plausible wrong result.
        assert input.dtype == torch.bfloat16 and input.dim() == 2, "a BF16 [M, K] input"
        assert input.is_contiguous(), "a contiguous input; the quantization reads it flat"
        assert weight.dtype == torch.float8_e4m3fn and weight.dim() == 2, "an E4M3 [N, K] weight"
        assert weight.shape[1] == input.shape[1], "input [M, K] x weight [N, K]"
        assert input.shape[1] % SF_VEC == 0, "K must be a whole number of 32-value scale blocks"
        assert weight.shape[0] % N_MULTIPLE == 0, (
            "N must be a multiple of 8 (16-byte BF16 output rows)"
        )
        assert weight.is_contiguous(), "a contiguous K-major weight"
        # On most real weights a checkpoint's [N, K / 32] has the padded R128c4 matrix's shape,
        # so only the flat form marks relaid scales.
        assert weight_scale.dim() == 1, (
            "flat R128c4 weight scales; a 2-D [N, K / 32] matrix is the "
            "checkpoint's row-major order"
        )
        assert weight_scale.dtype == torch.uint8 and weight_scale.is_contiguous(), (
            "contiguous UE8M0 scale bytes"
        )
        assert tuple(weight_scale.shape) == mxfp8_weight_scales_shape(*weight.shape), (
            "pad128(N) * pad4(K / 32) bytes: this weight's relaid scales"
        )
        for t in (weight, weight_scale):
            assert t.data_ptr() % 16 == 0, "16-byte aligned weight and scales"
        assert input.is_cuda and input.device.index == torch.cuda.current_device(), (
            "input on the current CUDA device"
        )


rubin_mxfp8_linear = RubinMxfp8Linear()


def mxfp8_act_layout(m: int, n: int, k: int) -> tuple[int, str]:
    """``(sf_rows, layout)`` of the MXFP8 activation scales ``rubin_mxfp8_linear_prequantized``
    reads for ``[m, k] x [n, k]^T``: the low-M / mid-M GEMM's ``"mid"`` layout over its token
    tile, or the dense GEMM's ``"r128c4"`` beyond ``LOW_M_GEMM_MAX_M`` rows (also where a
    ``MID_M_GEMM_MAX_M`` shape keeps the mid-M GEMM, which then reads R128c4)."""
    low_m = _low_m() if m <= LOW_M_GEMM_MAX_M and n % 16 == 0 else None
    if low_m is None:
        return -(-m // 128) * 128, "r128c4"
    if m <= low_m.low_m_max_for(n, k):
        return 128, "mid"
    return -(-low_m.mid_mma_n_for(m) // 128) * 128, "mid"


def mxfp8_quantizes_in_gemm(m: int, n: int, k: int) -> bool:
    """Whether ``rubin_mxfp8_linear`` (not pinned, reserve_sms or dense) quantizes a BF16
    ``[m, k]`` input for an ``[n, k]`` weight inside its GEMM: the shared low-M kernel, up to its
    low-M rows (``low_m_max_for``). Otherwise a quantization kernel runs ahead of the GEMM (the
    mid-M GEMM's own, or ``rubin_mxfp8_quantize`` ahead of the dense GEMM), which a producer's
    MXFP8 form saves through ``rubin_mxfp8_linear_prequantized``."""
    low_m = _low_m() if m <= low_m_gemm_max_m(n, k) and n % 16 == 0 else None
    return low_m is not None and m <= low_m.low_m_max_for(n, k)


class RubinMxfp8LinearPrequantized(OpWrapper):
    """``rubin_mxfp8_linear`` on an activation its producer already quantized to MXFP8.

    ``act`` is E4M3 ``[M, K]``, ``act_scale`` its UE8M0 1x32 scales in the layout
    ``mxfp8_act_layout(M, N, K)`` names; ``weight`` / ``weight_scale`` as for ``rubin_mxfp8_linear``.
    Bitwise ``rubin_mxfp8_linear`` on the BF16 rows the pair was quantized from.
    """

    ARCHS: frozenset[Arch] = frozenset()

    CELLS: tuple[Cell, ...] = (
        _mnk(1, 7168, 768, "shared experts down at TP8, one row (low-M, prequantized)"),
        _mnk(8, 7168, 768, "shared experts down at TP8, the TEP8 verify step (8 rows)"),
        _mnk(96, 7168, 768, "shared experts down at TP8, 96 rows (mid-M)"),
        _mnk(8, 7168, 6144, "shared experts down whole, 8 rows"),
        _mnk(96, 7168, 6144, "shared experts down whole, 96 rows"),
        _mnk(300, 7168, 6144, "shared experts down whole, a prefill chunk (dense GEMM)"),
        _mnk(12, 7168, 1536, "KDA o_proj at TP8, helix8 b2 k5, the verify kernel's MXFP8 (low-M)"),
        _mnk(96, 7168, 1536, "KDA o_proj at TP8, helix8 b16 k5, the verify kernel's MXFP8 (mid-M)"),
        _mnk(144, 6288, 7168, "KDA stack at TP8, helix8 b24 k5, the mix's MXFP8 (mid-M, R128c4)"),
        _mnk(32, 7168, 12288, "KDA o_proj whole, 32 rows, the verify kernel's MXFP8"),
        _mnk(96, 7168, 12288, "KDA o_proj whole, 96 rows, the verify kernel's MXFP8 (mid-M)"),
    )

    note = """
    Up to LOW_M_GEMM_MAX_M rows (a MID_M_GEMM_MAX_M shape: up to its rows): the shared low-M GEMM
    without its quantization warps (run_prequantized; mid-M kernel above its cut-over, reading
    the R128c4 scales past 128 rows). Beyond: rubin_mxfp8_gemm on the R128c4 scales. No
    quantization kernel either way.
    """

    def __call__(
        self,
        act: torch.Tensor,
        act_scale: torch.Tensor,
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
    ) -> torch.Tensor:
        m, n = act.shape[0], weight.shape[0]
        rows = low_m_gemm_max_m(n, act.shape[1])
        low_m = _low_m() if m <= rows and n % 16 == 0 else None
        if low_m is not None:
            out = torch.empty((m, n), dtype=torch.bfloat16, device=act.device)
            # Up to 128 rows the two scale orders are the same bytes.
            return low_m.run_prequantized(
                act, act_scale, weight, weight_scale, out, act_r128c4=True
            )
        return rubin_mxfp8_gemm(act, weight, act_scale, weight_scale)

    def reference(
        self,
        act: torch.Tensor,
        act_scale: torch.Tensor,
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
    ) -> torch.Tensor:
        """Both operands dequantized exactly (the activation from its layout), FP32 product."""
        m, k = act.shape
        sf_rows, layout = mxfp8_act_layout(m, weight.shape[0], k)
        tiles = act_scale.view(-1)[: sf_rows * k // 32]
        if layout == "mid":
            tiles = tiles.view(k // 128, sf_rows // 128, 32, 4, 4).permute(1, 0, 2, 3, 4)
            act_scale = tiles.reshape(-1).contiguous()
        return rubin_mxfp8_gemm.reference(act, weight, act_scale, weight_scale)

    def is_valid(
        self,
        act: torch.Tensor,
        act_scale: torch.Tensor,
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
    ) -> None:
        assert act.dtype == weight.dtype == torch.float8_e4m3fn, "E4M3 activation and weight"
        assert act.dim() == 2 and act.shape[1] == weight.shape[1], "act [M, K] x weight [N, K]"
        assert act.is_contiguous() and weight.is_contiguous(), "contiguous K-major operands"
        assert act_scale.dtype == weight_scale.dtype == torch.uint8, "UE8M0 scale bytes"
        assert act_scale.dim() == 1 and weight_scale.dim() == 1, "flat scale buffers"
        sf_rows, _ = mxfp8_act_layout(act.shape[0], weight.shape[0], act.shape[1])
        assert act_scale.numel() >= sf_rows * act.shape[1] // 32, "act_scale covers sf_rows"
        assert tuple(weight_scale.shape) == mxfp8_weight_scales_shape(*weight.shape), (
            "pad128(N) * pad4(K / 32) bytes: this weight's relaid scales"
        )


rubin_mxfp8_linear_prequantized = RubinMxfp8LinearPrequantized()
