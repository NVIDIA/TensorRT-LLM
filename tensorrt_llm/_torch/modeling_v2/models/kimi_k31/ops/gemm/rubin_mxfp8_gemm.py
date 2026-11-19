# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Dense MXFP8 GEMM on sm_107: E4M3 operands, one UE8M0 scale per 32 values of a row.

The kernel and its runner are the family copy of the engine's SM107 dense block-scaled GEMM
(``kernels/gemm/rubin_dense_blockscaled_gemm.py``) with the Kimi K3.1 tuning: 16-row tuning
buckets up to 192 rows, a pruned tactic list, and split-K summed in FP32 planes and rounded once.
``__call__`` asks the autotuner for a tactic under this family's own key and runs the runner;
``pinned=True`` (dep16's MXFP8 levers) asks with a split-K-free runner and runs a pinned tactic at
the 96-row verify bucket. Nothing here is registered as a torch op.

The scales are read in the R128c4 order (rows in tiles of 128, scale columns in groups of 4). The
helpers below convert a row-major ``[rows, K / 32]`` byte matrix (a checkpoint's
``weight_scale``) to it and back, and quantize a BF16 matrix the way ``mxfp8_quantize`` does.
They run on CPU too.
"""

from __future__ import annotations

import functools

import torch

from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper, true_fp32_matmul

#: Values per UE8M0 scale along K (the "K32" of the scale layout).
SF_VEC = 32
#: The autotuner key. The engine's MXFP8 GEMM tunes under ``trtllm::cute_dsl_mxfp8_gemm_rubin``
#: with its own runner classes, so the two never share a cache entry.
TUNING_KEY = "kimi_k31::rubin_mxfp8_gemm"
_E4M3_MAX = 448.0
#: ``pinned=True``: the rows of the pinned tactics, the tuning bucket of a dep16 generation server's
#: 96-row verify step (12 requests x 8 rows); other row counts take the autotuner's pick among the
#: split-K-free tactics.
PINNED_MIN_M, PINNED_MAX_M = 81, 96
#: ``pinned=True``: the 81-96 row tactic of each weight shape ``(N, K)`` of dep16's MXFP8
#: levers. None splits K: every split-K-free tactic adds the K tiles into one FP32 accumulator in
#: the same order, so the bits do not depend on the tactic, and no server start can pick a split-K
#: tactic for these GEMMs. Each is the fastest split-K-free tactic of the K3 runner's profiled set
#: and the (4, 1) cluster tiles at 96 rows on sm_107 (CUPTI, L2 cold).
PINNED_TACTICS: dict[tuple[int, int], tuple] = {
    # MLA g_proj (96 heads x 128 rows), 10.98 us
    (12288, 7168): ("base", (256, 128, 128), (256, 128, 64), (2, 1), True, False, "static", "m", 1),
    # MLA and drafter o_proj, K = 96 heads x 128, 13.06 us
    (7168, 12288): ("base", (256, 64, 128), (256, 64, 64), (4, 1), True, False, "static", "m", 1),
    # the latent MoE's latent_down (3584 rows), 8.10 us
    (3584, 7168): (
        "base",
        (256, 64, 128),
        (256, 64, 64),
        (2, 1),
        True,
        False,
        "clc_dynamic",
        "n",
        1,
    ),
    # the drafter's MLP gate_up (2 x 14336 rows), 19.87 us
    (28672, 7168): ("base", (512, 128, 128), (256, 128, 64), (2, 1), True, False, "static", "n", 1),
    # the drafter's MLP down, K = 14336, 14.53 us
    (7168, 14336): ("base", (256, 64, 128), (256, 64, 64), (4, 1), True, False, "static", "m", 1),
}


def _pad_up(value: int, multiple: int) -> int:
    return (value + multiple - 1) // multiple * multiple


def mxfp8_scales_to_r128c4(scales: torch.Tensor) -> torch.Tensor:
    """Row-major ``[rows, K / 32]`` UE8M0 bytes -> the flat R128c4 buffer the kernel reads.

    Rows pad to 128 and columns to 4 with zero bytes. These are the bytes of the engine's
    ``block_scale_interleave`` and of ``mxfp8_quantize``'s swizzled scales.
    """
    rows, cols = scales.shape
    padded_rows, padded_cols = _pad_up(rows, 128), _pad_up(cols, 4)
    padded = scales.new_zeros(padded_rows, padded_cols)
    padded[:rows, :cols] = scales
    tiles = padded.view(padded_rows // 128, 4, 32, padded_cols // 4, 4)
    return tiles.permute(0, 3, 2, 1, 4).reshape(-1)


def mxfp8_scales_from_r128c4(flat: torch.Tensor, rows: int, k: int) -> torch.Tensor:
    """The inverse: the ``[rows, k / 32]`` scale bytes of an R128c4 buffer."""
    cols = k // SF_VEC
    padded_rows, padded_cols = _pad_up(rows, 128), _pad_up(cols, 4)
    tiles = flat.reshape(padded_rows // 128, padded_cols // 4, 32, 4, 4)
    return tiles.permute(0, 3, 2, 1, 4).reshape(padded_rows, padded_cols)[:rows, :cols]


def _pow2(byte: torch.Tensor) -> torch.Tensor:
    """``2 ** (byte - 127)`` in FP32, exactly (byte 0 is the subnormal 2^-127)."""
    bits = byte.to(torch.int32) << 23
    return torch.where(byte > 0, bits.view(torch.float32), 2.0**-127)


def quantize_mxfp8_reference(input: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """MXFP8 of a ``[rows, K]`` matrix: ``(E4M3 values [rows, K], scale bytes [rows, K / 32])``.

    The scale of each 32 values is the smallest power of two at or above ``amax / 448``, as the
    catalog ``mxfp8_quantize`` computes it; the scales come back row-major, not R128c4.
    """
    rows, k = input.shape
    blocks = input.float().reshape(rows, k // SF_VEC, SF_VEC)
    bits = (blocks.abs().amax(dim=-1) / _E4M3_MAX).view(torch.int32)
    byte = (((bits >> 23) & 0xFF) + ((bits & 0x7FFFFF) > 0).to(torch.int32)).clamp(0, 254)
    values = (blocks / _pow2(byte).unsqueeze(-1)).reshape(rows, k).to(torch.float8_e4m3fn)
    return values, byte.to(torch.uint8)


def dequantize_mxfp8(values: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """FP32 ``[rows, K]`` of E4M3 ``values`` and their R128c4 ``scales``; exact."""
    rows, k = values.shape
    byte = mxfp8_scales_from_r128c4(scales, rows, k)
    blocks = values.float().reshape(rows, k // SF_VEC, SF_VEC) * _pow2(byte).unsqueeze(-1)
    return blocks.reshape(rows, k)


_ALPHA: dict[torch.device, torch.Tensor] = {}


def _alpha(device: torch.device) -> torch.Tensor:
    """The FP32 one the kernel reads by pointer, made once per device outside graph capture."""
    alpha = _ALPHA.get(device)
    if alpha is None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "rubin_mxfp8_gemm: run one eager call on this device before CUDA graph capture"
            )
        alpha = _ALPHA[device] = torch.ones((), dtype=torch.float32, device=device)
    return alpha


@functools.cache
def _runner():
    """The autotuner and the family runner; imported on first use (they need the GPU stack)."""
    from tensorrt_llm._torch.autotuner import AutoTuner

    from ...kernels.gemm.rubin_dense_blockscaled_gemm import RubinMxfp8GemmRunner

    return AutoTuner, RubinMxfp8GemmRunner()


def _tuning_config(autotuner, runner):
    """The runner's tuning config for this process's tuning group.

    Where the ranks serve the same requests (TP without attention DP, and Helix CP), they merge
    their measurements into one tactic (``replicated_tuning_config``): the projections they run
    on the same rows through the same weights (Helix CP; under TP the KDA projection's replicated
    f_a columns) stay bit-identical across ranks, and a sharded projection runs one tactic on
    every shard. Under attention DP each rank multiplies its own rows and tunes alone
    (``tuning_config``).
    """
    mapping = getattr(autotuner, "mapping", None)
    if mapping is not None and (
        mapping.has_cp_helix() or (mapping.tp_size > 1 and mapping.tp_group_shares_requests)
    ):
        return runner.replicated_tuning_config
    return runner.tuning_config


@functools.cache
def _split_k_free_runner():
    """The family runner without split-K tactics (``pinned=True``): ``max_split_k`` is part of the
    runner's autotuner cache key, so no entry tuned with split-K is ever taken."""
    from ...kernels.gemm.rubin_dense_blockscaled_gemm import RubinMxfp8GemmRunner

    return RubinMxfp8GemmRunner(max_split_k=1)


def _mnk(m: int, n: int, k: int, why: str) -> Cell:
    return Cell(why=why, spec=dict(m=m, n=n, k=k))


class RubinMxfp8Gemm(OpWrapper):
    """``input [M, K] x weight [N, K]^T -> [M, N]`` BF16, both operands MXFP8.

    ``input`` and ``weight`` are E4M3. ``input_scale`` and ``weight_scale`` are the flat R128c4
    UE8M0 buffers (``pad_up(rows, 128) * pad_up(K / 32, 4)`` bytes each): ``mxfp8_quantize``'s
    swizzled scales for an activation, ``mxfp8_scales_to_r128c4`` of a checkpoint's row-major
    ``weight_scale``. Every 32 values of a row are scaled by ``2 ** (byte - 127)``; the kernel
    accumulates in FP32 and rounds once to BF16.
    """

    ARCHS = frozenset({Arch.SM_107})

    CELLS: tuple[Cell, ...] = (
        _mnk(1, 128, 7168, "one row, N one 128-row scale tile (the KDA f_a projection)"),
        _mnk(37, 96, 2048, "M and N both off the 128-row scale tile: padded scale rows"),
        _mnk(8, 7168, 1536, "K of 48 scale blocks (12 heads x 128): a TP8 KDA o_proj"),
        _mnk(96, 6144, 7168, "96 rows, the DEP16 and helix8 verify steps"),
        _mnk(300, 1024, 4096, "three 128-row M tiles, the last one partial"),
        _mnk(2954, 7168, 6144, "a prefill chunk through the shared experts' down projection"),
    )

    note = """
    The weights this entry serves are the ModelOpt export's MXFP8 projections, stored E4M3 with one
    scale per 32 values of a row. Large and small M both run this kernel: the family's low-M GEMMs
    read 128x128 block scales only. The first call per device and per tile compiles and makes the
    FP32 one the kernel reads by pointer, so it must run outside CUDA graph capture. Outside an
    autotuning pass the autotuner returns its cached tactic, or the runner's fallback tile. Ranks
    that serve the same requests (TP, Helix CP) merge their tuning into one tactic, so their
    replicated projections stay bit-identical; under attention DP each rank tunes alone. All
    tactics but split-K accumulate K in the same order; split-K sums FP32 planes in a fixed order.
    ``pinned=True`` asks the autotuner with a split-K-free runner, and at PINNED_MIN_M to
    PINNED_MAX_M rows runs the tactic PINNED_TACTICS holds for the weight shape (still asking, so a
    tuning pass at 96 rows tunes the smaller buckets). The cells cover K a multiple of 128 and N a
    multiple of 8.
    ``reserve_sms`` keeps that many SMs out of the GEMM's persistent grid (the static scheduler
    over fewer clusters), for a GEMM that runs beside a kernel which needs them from its start;
    the tactic is the one the autotuner returns either way, so the output is the same.
    """

    def __call__(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        input_scale: torch.Tensor,
        weight_scale: torch.Tensor,
        pinned: bool = False,
        reserve_sms: int = 0,
    ) -> torch.Tensor:
        tuner, runner = _runner()
        if pinned:
            runner = _split_k_free_runner()
        inputs = [input, weight, input_scale, weight_scale, _alpha(input.device)]
        autotuner = tuner.get()
        _, tactic = autotuner.choose_one(
            TUNING_KEY, [runner], _tuning_config(autotuner, runner), inputs
        )
        if pinned and PINNED_MIN_M <= input.shape[0] <= PINNED_MAX_M:
            tactic = PINNED_TACTICS.get((weight.shape[0], input.shape[1]), tactic)
        if reserve_sms > 0:
            return runner(inputs, tactic=tactic, reserve_sms=reserve_sms)
        return runner(inputs, tactic=tactic)

    def reference(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        input_scale: torch.Tensor,
        weight_scale: torch.Tensor,
        pinned: bool = False,
        reserve_sms: int = 0,
    ) -> torch.Tensor:
        """Both operands dequantized exactly, an FP32 product (TF32 off), one rounding to BF16; the
        tactic (``pinned``) and ``reserve_sms`` (where the kernel runs) do not enter it."""
        a = dequantize_mxfp8(input, input_scale)
        b = dequantize_mxfp8(weight, weight_scale)
        with true_fp32_matmul():
            return (a @ b.t()).to(torch.bfloat16)

    def is_valid(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        input_scale: torch.Tensor,
        weight_scale: torch.Tensor,
        pinned: bool = False,
        reserve_sms: int = 0,
    ) -> None:
        assert isinstance(pinned, bool), "pinned is a flag"
        # The runner turns the four tensors into raw pointers of the dtypes it was built for and
        # checks only the scale sizes, so each of these reads as a plausible wrong result.
        assert input.dtype == weight.dtype == torch.float8_e4m3fn, "E4M3 input and weight"
        assert input_scale.dtype == weight_scale.dtype == torch.uint8, "UE8M0 scale bytes"
        assert input.dim() == 2 and weight.dim() == 2 and input.shape[1] == weight.shape[1], (
            "input [M, K] x weight [N, K]"
        )
        assert input.shape[1] % SF_VEC == 0, "K must be a whole number of 32-value scale blocks"
        assert weight.shape[0] % 8 == 0, "N must be a multiple of 8 (16-byte BF16 output rows)"
        assert input.is_contiguous() and weight.is_contiguous(), "contiguous K-major operands"
        assert input_scale.dim() == 1 and weight_scale.dim() == 1, (
            "flat R128c4 scales; a 2-D [rows, K / 32] matrix is the row-major order"
        )
        for t in (input, weight, input_scale, weight_scale):
            assert t.data_ptr() % 16 == 0, "16-byte aligned operands and scales"
        # The runner allocates the output on the current device.
        assert input.is_cuda and input.device.index == torch.cuda.current_device(), (
            "input on the current CUDA device"
        )


rubin_mxfp8_gemm = RubinMxfp8Gemm()
