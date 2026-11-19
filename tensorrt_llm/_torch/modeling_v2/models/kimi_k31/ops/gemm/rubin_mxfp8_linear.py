# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""ModelOpt MXFP8 W8A8 linear: the BF16 activation quantized to MXFP8, then the dense GEMM.

The ModelOpt export stores the KDA projections and the shared experts as E4M3 weights with one
UE8M0 scale per 32 values of a row: the "K32" scales, a row-major ``[N, K / 32]`` byte matrix,
which the weight tables relay at load to the flat R128c4 buffer the GEMM reads. The activation
takes the same recipe at run time (the catalog ``mxfp8_quantize``, swizzled scales), and
``rubin_mxfp8_gemm`` multiplies the two on sm_107. On sm_103 the engine's MXFP8 GEMM
(``trtllm::mxfp8_mxfp8_gemm``, the CUTLASS block-scaled kernel of the SM10x family) multiplies the
same operands (``_sm103_linear``): R128c4 is the engine's swizzled scale order too.
"""

from __future__ import annotations

import functools

import torch

from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper
from tensorrt_llm._utils import get_sm_version

from .rubin_mxfp8_gemm import (
    SF_VEC,
    _alpha,
    mxfp8_scales_to_r128c4,
    quantize_mxfp8_reference,
    rubin_mxfp8_gemm,
)

#: The GEMM writes 16-byte BF16 rows, so N is a multiple of 8.
N_MULTIPLE = 8
#: The engine's GEMM on sm_103 takes N in multiples of 32 (stack zero rows up to it).
N_MULTIPLE_SM103 = 32


def _pad_up(value: int, multiple: int) -> int:
    return (value + multiple - 1) // multiple * multiple


def mxfp8_weight_scales_shape(n: int, k: int) -> tuple[int]:
    """The relaid scales' shape for an ``[n, k]`` weight: ``(pad128(n) * pad4(k / 32),)``."""
    return (_pad_up(n, 128) * _pad_up(k // SF_VEC, 4),)


def _sm103_linear(input: torch.Tensor, weight: torch.Tensor, weight_scale: torch.Tensor):
    """The sm_103 GEMM of :class:`RubinMxfp8Linear`: the same quantization, then the engine's MXFP8
    GEMM on the same operands (through its autotuned op while the engine autotunes, which registers
    the tactic the plain op reads afterwards). An empty input returns an empty output."""
    if input.shape[0] == 0:
        return input.new_empty((0, weight.shape[0]), dtype=torch.bfloat16)
    act, act_scale = _quantize()(input, True, SF_VEC)
    from tensorrt_llm._torch.autotuner import AutoTuner

    tuning = not torch.compiler.is_compiling() and AutoTuner.get().is_tuning_mode
    gemm = (
        torch.ops.trtllm.mxfp8_mxfp8_gemm_autotuned if tuning else torch.ops.trtllm.mxfp8_mxfp8_gemm
    )
    return gemm(act, act_scale, weight, weight_scale, _alpha(input.device), torch.bfloat16)


@functools.cache
def _quantize():
    """The catalog MXFP8 quantization; imported on first use (it needs the GPU stack)."""
    from tensorrt_llm._torch.modeling_v2.catalog.quantization.mxfp8_quantize import (  # noqa: E501
        mxfp8_quantize,
    )

    return mxfp8_quantize


def _mnk(m: int, n: int, k: int, why: str) -> Cell:
    return Cell(why=why, spec=dict(m=m, n=n, k=k))


class RubinMxfp8Linear(OpWrapper):
    """``input [M, K] x weight [N, K]^T -> [M, N]`` BF16 for an MXFP8 weight and a BF16 input.

    ``weight`` is E4M3. ``weight_scale`` is its relaid scales: the flat R128c4 buffer (the bytes
    of the engine's ``block_scale_interleave``), ``pad128(N) * pad4(K / 32)`` UE8M0 bytes. The
    input is quantized with one power-of-two scale per 32 values, the smallest at or above
    ``amax / 448``; the product accumulates in FP32 and rounds once to BF16. sm_107 runs
    ``rubin_mxfp8_gemm``, sm_103 the engine's MXFP8 GEMM; ``pinned`` and ``reserve_sms`` are
    sm_107's (that GEMM's tactics and grid).
    """

    ARCHS: frozenset[Arch] = frozenset({Arch.SM_103, Arch.SM_107})

    # The calls of the specialized targets. The KDA stack is [q | k | v | g] (96 heads x 128 each,
    # split by the attention TP size), f_a (128 rows, whole) and b (one row per head), zero rows
    # to a multiple of 8. The shared experts are a TP8 share on TEP8 and whole elsewhere.
    CELLS: tuple[Cell, ...] = (
        _mnk(1, 6288, 7168, "KDA stack at TP8 (6144 + 128 + 12 rows, padded to 6288), one row"),
        _mnk(8, 6288, 7168, "KDA stack at TP8, the TEP8 verify step b1 k7 (8 rows)"),
        _mnk(96, 6288, 7168, "KDA stack at TP8, the helix8 verify step b16 k5 (96 rows)"),
        _mnk(2954, 6288, 7168, "KDA stack at TP8, a TEP8 prefill chunk (last M tile partial)"),
        _mnk(96, 49376, 7168, "KDA stack at DEP16 (49152 + 128 + 96 rows), verify b12 k7"),
        _mnk(2954, 49376, 7168, "KDA stack at DEP16, a prefill chunk"),
        _mnk(8, 7168, 1536, "KDA o_proj at TP8: K = 12 heads x 128 (48 scale blocks), 8 rows"),
        _mnk(96, 7168, 1536, "KDA o_proj at TP8, the helix8 verify step (96 rows)"),
        _mnk(2954, 7168, 1536, "KDA o_proj at TP8, a TEP8 prefill chunk"),
        _mnk(96, 7168, 12288, "KDA o_proj at DEP16: K = 96 heads x 128, 96 rows"),
        _mnk(2954, 7168, 12288, "KDA o_proj at DEP16, a prefill chunk"),
        _mnk(8, 1536, 7168, "shared experts [gate | up] at TP8 (2 x 768 rows), 8 rows"),
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
    Two kernels: mxfp8_quantize (one CUDA kernel, swizzled scales) and the dense block-scaled GEMM.
    The quantizer is a generic call: the catalog entry, whose ARCHS do not list sm_107. On sm_107
    only this entry's cells drive it.
    The weight scales are the flat R128c4 buffer the GEMM reads. A 2-D scale is refused: a
    checkpoint's row-major [N, K / 32] matrix has the zero-padded R128c4 matrix's shape whenever
    N % 128 == 0 and K / 32 % 4 == 0 (o_proj, the shared experts). Relay it at load.
    dep16's MXFP8 levers (MLA g_proj and o_proj: the shared [gate | up] and KDA o_proj shapes;
    latent_down; the drafter's o_proj, gate_up and down) pass pinned=True: rubin_mxfp8_gemm's
    split-K-free runner and pinned 96-row tactics.
    N must be a multiple of 8, so a projection with fewer rows per rank (the KDA b_proj at TP8 has
    12) runs stacked with other projections. The first call per device compiles; see
    rubin_mxfp8_gemm for the CUDA graph and autotuner notes, and for ``reserve_sms`` (SMs kept out
    of the GEMM's grid; the output is the same).
    The cells were driven on sm_107 and sm_103. sm_103 (the NVFP4 default target) runs the engine's
    MXFP8 GEMM on the same quantization and scales, N a multiple of 32.
    """

    def __call__(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
        pinned: bool = False,
        reserve_sms: int = 0,
    ) -> torch.Tensor:
        if get_sm_version() == 103:
            return _sm103_linear(input, weight, weight_scale)
        act, act_scale = _quantize()(input, True, SF_VEC)
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
    ) -> torch.Tensor:
        """The input quantized as ``mxfp8_quantize`` does, then ``rubin_mxfp8_gemm.reference``;
        ``reserve_sms`` does not enter it."""
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
    ) -> None:
        assert isinstance(pinned, bool), "pinned is a flag"
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
        if get_sm_version() == 103:
            assert weight.shape[0] % N_MULTIPLE_SM103 == 0, "N must be a multiple of 32 on sm_103"
            assert not pinned and reserve_sms == 0, "pinned and reserve_sms are sm_107's"
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
