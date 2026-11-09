# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""BF16 ``y = x @ weight.T`` for at most 32 rows on the low-M split-K kernel, at a fixed tactic.

The built-in low-M dispatcher (``_torch/modules/low_m_gemm.py``) races the
flashinfer CuTe-DSL split-K and direct kernels against cuBLAS per shape on
each rank and caches the winner per rank; ranks can and do settle on
different kernels for one shape, and plain ``nn.Linear`` / ``F.linear`` call
sites never reach it. This entry is the split-K kernel the dispatcher would
launch, with the tactic chosen by the caller: one tactic per shape, the same
on every rank, measured once per target GPU and TP size (``SPLITK_TACTICS_BY_SM_TP``).
The kernel is this target's copy (``ops/gemm/splitk_gemm.py``) of flashinfer's
``dense_bf16_gemm_sm100_splitk``, with its env-gated changes.

The kernel accumulates in fp32 and rounds once; its K-split differs from
cuBLAS's, so results differ from ``F.linear`` in the last bf16 place on a few
elements -- the same distance from the fp64 product as cuBLAS.
"""

from __future__ import annotations

from ...sm107_defaults import env as _sm107_env
from ...sm107_defaults import sm103_tep8_f11 as _sm103_tep8_f11

import os

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import (
    Arch,
    OpWrapper,
    true_fp32_matmul,
)

#: The split-K tactic ``(mma_m, mma_n, split_k, ab_stages)`` per ``(M, K, N)`` of a
#: tep8 drafter (two trays) on sm103, chosen on GB300 with cold weights and PDL on;
#: each one beats cuBLAS at its shape. Shapes not listed are not taken by this op's callers.
SPLITK_TACTICS: dict[tuple[int, int, int], tuple[int, int, int, int]] = {
    # DFlash 2 convolution coefficient projection (replicated).
    (8, 8192, 2048): (64, 8, 4, 10),
    # drafter qkv_proj per rank, and the fused context K/V projection.
    (8, 8192, 1280): (64, 16, 4, 9),
    # drafter o_proj per rank.
    (8, 1024, 8192): (128, 8, 1, 6),
    # drafter down_proj per rank.
    (8, 3584, 8192): (64, 8, 1, 12),
    # drafter gate_up_proj per rank.
    (8, 8192, 7168): (128, 8, 2, 6),
    # DFlash fc, one TP8 column shard (``drafter_fc_shard``).
    (8, 40960, 1024): (64, 8, 4, 12),
}

#: The same shapes on sm107 (212 SMs), each tactic chosen there the same way.
SM107_SPLITK_TACTICS: dict[tuple[int, int, int], tuple[int, int, int, int]] = {
    (8, 8192, 2048): (64, 8, 4, 10),
    (8, 8192, 1280): (64, 8, 4, 12),
    (8, 1024, 8192): (64, 8, 1, 10),
    (8, 3584, 8192): (64, 8, 1, 12),
    (8, 8192, 7168): (64, 16, 1, 9),
    (8, 40960, 1024): (64, 8, 4, 12),
}

#: Opt-in: ``MUSESPARK_OPEN_TEP8_DRAFTER_WIDE_GEMMS=1`` adds the tep8 sm107 drafter's 2..4-request blocks
#: (16, 24 and 32 rows), and with ``MUSESPARK_OPEN_TEP8_WIDE64_SPLITK=1`` (the split-K kernel's 64-wide tiles)
#: its 5..8-request blocks (40..64 rows), which otherwise run cuBLAS (nvjet, plus a cublasLt splitKreduce launch on four
#: of the five per-layer projections: 20-23 reduce launches per step on the TEP8 bs4/bs8 critical path). Each shape keeps
#: the ``split_k`` of its 8-row cell, so every row of a block gets the bits it gets in a one-request step (flashinfer's
#: split-K sums the same K slices in the same order whatever the tile widths); against cuBLAS a row differs in the last
#: BF16 place on some elements, as the 8-row cells already do (o_proj and down_proj, split_k 1: equal). Tile widths
#: and stages: the fastest measured behind an early-releasing predecessor
#: with PDL, which is what the drafter's predecessors are (hop, block convolution, window-attention merge, SiLU-mul
#: all release their dependents early: the 8-row split-K GEMMs start 1.5-4.7 us before them in TEP8 bs1 traces).
TEP8_DRAFTER_WIDE_ENV = "MUSESPARK_OPEN_TEP8_DRAFTER_WIDE_GEMMS"
_TEP8_WIDE64_ENV = "MUSESPARK_OPEN_TEP8_WIDE64_SPLITK"
#: (mma_m, mma_n, split_k, ab_stages) per row count and site: the fastest split-K tile behind an early-releasing
#: predecessor with PDL, at the 8-row cell's split_k.
_TP8_DRAFTER_WIDE_SHAPES = {"conv_coef": (8192, 2048), "qkv": (8192, 1280), "o_proj": (1024, 8192),
                            "down": (3584, 8192), "gate_up": (8192, 7168)}  # qkv: also the context K/V
_TP8_DRAFTER_WIDE_TILES = {
    16: {"conv_coef": (64, 16, 4, 10), "qkv": (64, 16, 4, 10), "o_proj": (64, 16, 1, 11), "down": (64, 16, 1, 11), "gate_up": (64, 16, 1, 11)},
    24: {"conv_coef": (64, 32, 4, 8), "qkv": (64, 32, 4, 8), "o_proj": (64, 32, 1, 9), "down": (64, 32, 1, 9), "gate_up": (64, 32, 1, 9)},
    32: {"conv_coef": (64, 32, 4, 8), "qkv": (64, 32, 4, 8), "o_proj": (64, 32, 1, 9), "down": (64, 32, 1, 9), "gate_up": (64, 32, 1, 9)},
    40: {"conv_coef": (64, 64, 4, 5), "qkv": (64, 64, 4, 5), "o_proj": (64, 64, 1, 7), "down": (64, 64, 1, 7), "gate_up": (64, 64, 1, 7)},
    48: {"conv_coef": (64, 64, 4, 5), "qkv": (64, 64, 4, 5), "o_proj": (64, 64, 1, 7), "down": (64, 64, 1, 7), "gate_up": (64, 64, 1, 7)},
    56: {"conv_coef": (64, 64, 4, 5), "qkv": (64, 64, 4, 5), "o_proj": (64, 64, 1, 7), "down": (64, 64, 1, 7), "gate_up": (64, 64, 1, 7)},
    64: {"conv_coef": (64, 64, 4, 5), "qkv": (64, 64, 4, 5), "o_proj": (64, 64, 1, 7), "down": (64, 64, 1, 7), "gate_up": (64, 64, 1, 7)},
}
SM107_TP8_DRAFTER_WIDE_TACTICS: dict[tuple[int, int, int], tuple[int, int, int, int]] = {
    (_rows, *_TP8_DRAFTER_WIDE_SHAPES[_site]): _tactic
    for _rows, _sites in _TP8_DRAFTER_WIDE_TILES.items()
    for _site, _tactic in _sites.items()
}
#: Opt-in: ``MUSESPARK_OPEN_TEP8_FC_SHARD_WIDE`` (``spec_worker``'s TP8 fc shard above 8 rows) adds the
#: DFlash fc's TP8 column shard ``[rows, 40960] @ [1024, 40960].T`` at 16..32 rows, and with
#: ``MUSESPARK_OPEN_TEP8_WIDE64_SPLITK=1`` (the split-K kernel's M <= 64) 40..64 rows; other row counts run
#: ``F.linear`` on the shard. Each keeps the 8-row cell's split_k (4): a row's bits are its one-request bits, and at 24
#: and 32 rows they are also the replicated cuBLAS GEMM's (its nvjet 160x64 kernel splits K the same 4 ways; elsewhere
#: 99.4-99.6 % of elements equal, the rest one BF16 ulp). Widths: the fastest behind a plain predecessor (the side
#: chain's torch kernels), 16 rows per tile up to 32 rows, 32 above.
TEP8_FC_SHARD_WIDE_ENV = "MUSESPARK_OPEN_TEP8_FC_SHARD_WIDE"
SM107_TP8_FC_WIDE_TACTICS: dict[tuple[int, int, int], tuple[int, int, int, int]] = {
    (_rows, 40960, 1024): (64, 16, 4, 10) if _rows <= 32 else (64, 32, 4, 8) for _rows in (16, 24, 32, 40, 48, 56, 64)
}
TEP8_WIDE64 = _sm107_env(_TEP8_WIDE64_ENV, "0") == "1"
TEP8_DRAFTER_WIDE = _sm107_env(TEP8_DRAFTER_WIDE_ENV, "0") == "1"
TEP8_FC_WIDE = _sm107_env(TEP8_FC_SHARD_WIDE_ENV, "0").strip() not in ("", "0")
TEP8_DRAFTER_WIDE64 = TEP8_DRAFTER_WIDE and TEP8_WIDE64
if TEP8_DRAFTER_WIDE:
    SM107_SPLITK_TACTICS.update(
        {key: tactic for key, tactic in SM107_TP8_DRAFTER_WIDE_TACTICS.items() if key[0] <= 32 or TEP8_WIDE64}
    )
if TEP8_FC_WIDE:
    SM107_SPLITK_TACTICS.update(
        {key: tactic for key, tactic in SM107_TP8_FC_WIDE_TACTICS.items() if key[0] <= 32 or TEP8_WIDE64}
    )


def _sm103_tp8_wide(tactic: tuple, shape: tuple) -> tuple:
    """The sm103 form of an sm107 TP8 wide drafter tactic: the ``split_k`` of sm103's own 8-row cell
    (``SPLITK_TACTICS``; only gate_up differs: 2 on sm103, 1 on sm107), and the deepest of the sm107 stage count and
    what GB300's 232448 B opt-in holds at that split (no oversized shared memory on sm103; stages change no bits)."""
    mma_m, mma_n, split_k, stages = tactic
    split_k = SPLITK_TACTICS[(8, *shape)][2]

    def smem(st: int) -> int:  # splitk_gemm._smem_bytes, mirrored (no CuTe import at module load)
        ab = -(-mma_m * 128 * 2 * st // 1024) * 1024 + mma_n * 128 * 2 * st + 2 * st * 8 + 3 * 8 + 4
        if split_k == 1:
            return ab
        return -(-(-(-ab // 128) * 128 + (split_k - 1) * mma_m * mma_n * 4) // 8) * 8 + 8

    while stages > 1 and smem(stages) > 232448:
        stages -= 1
    return (mma_m, mma_n, split_k, stages)


#: sm_103 TEP8 under ``MUSESPARK_OPEN_SM103_TEP8_F11=1`` (``sm107_defaults``): the sm107 wide drafter / fc-shard
#: tactics in their sm103 form (``_sm103_tp8_wide``), added to sm103's TP8 table by the same switches.
SM103_TP8_DRAFTER_WIDE_TACTICS: dict[tuple[int, int, int], tuple[int, int, int, int]] = {
    key: _sm103_tp8_wide(tactic, key[1:]) for key, tactic in SM107_TP8_DRAFTER_WIDE_TACTICS.items()
}
SM103_TP8_FC_WIDE_TACTICS: dict[tuple[int, int, int], tuple[int, int, int, int]] = {
    key: _sm103_tp8_wide(tactic, key[1:]) for key, tactic in SM107_TP8_FC_WIDE_TACTICS.items()
}
if _sm103_tep8_f11():
    if TEP8_DRAFTER_WIDE:
        SPLITK_TACTICS.update(
            {key: tactic for key, tactic in SM103_TP8_DRAFTER_WIDE_TACTICS.items() if key[0] <= 32 or TEP8_WIDE64}
        )
    if TEP8_FC_WIDE:
        SPLITK_TACTICS.update(
            {key: tactic for key, tactic in SM103_TP8_FC_WIDE_TACTICS.items() if key[0] <= 32 or TEP8_WIDE64}
        )

#: The same for a tep4 drafter (one tray) on sm103: every TP-sharded projection is
#: twice the tep8 one along its sharded dimension; the replicated convolution
#: coefficient projection is the tep8 one. The fastest split-K tactic of each shape in
#: chains of cold weights with PDL on; it beats cuBLAS on the qkv, gate_up and
#: convolution shapes, ties it on down_proj and trails it slightly on o_proj in
#: isolation, where ``drafter_pdl_chain``'s early weight load is what it is for. Besides the
#: one-request block (8 rows) the table lists the blocks of 2..4 requests (16, 24 and 32
#: rows), each the fastest split-K tactic at that row count on GB300's 152 SMs over cold
#: weights with PDL on; every one is a single wave of at most 128 CTAs.
SPLITK_TACTICS_TP4: dict[tuple[int, int, int], tuple[int, int, int, int]] = {
    # DFlash 2 convolution coefficient projection (replicated).
    (8, 8192, 2048): (64, 8, 4, 10),
    # drafter qkv_proj per rank, and the fused context K/V projection.
    (8, 8192, 2560): (64, 8, 2, 11),
    # drafter o_proj per rank.
    (8, 2048, 8192): (64, 8, 1, 11),
    # drafter down_proj per rank.
    (8, 7168, 8192): (64, 8, 1, 11),
    # drafter gate_up_proj per rank.
    (8, 8192, 14336): (128, 8, 1, 6),
    (16, 8192, 2048): (64, 16, 4, 8),
    (16, 8192, 2560): (64, 16, 2, 10),
    (16, 2048, 8192): (64, 16, 1, 10),
    (16, 7168, 8192): (64, 16, 1, 10),
    (16, 8192, 14336): (128, 16, 1, 6),
    (24, 8192, 2048): (64, 16, 2, 11),
    (24, 8192, 2560): (64, 32, 2, 9),
    (24, 2048, 8192): (64, 32, 1, 9),
    (24, 7168, 8192): (64, 32, 1, 9),
    (24, 8192, 14336): (128, 32, 1, 5),
    (32, 8192, 2048): (64, 16, 2, 11),
    (32, 8192, 2560): (64, 32, 2, 9),
    (32, 2048, 8192): (64, 32, 1, 9),
    (32, 7168, 8192): (64, 32, 1, 9),
    (32, 8192, 14336): (128, 32, 1, 5),
}

#: The tep4 drafter shapes on sm107 (212 SMs), each the fastest split-K tactic there in
#: the same chains; every one beats cuBLAS at its shape. Besides the one-request block
#: (8 rows) the table lists the blocks of 2..4 requests (16, 24 and 32 rows), each the
#: fastest split-K tactic at that row count over cold weights with PDL on.
SM107_SPLITK_TACTICS_TP4: dict[tuple[int, int, int], tuple[int, int, int, int]] = {
    (8, 8192, 2048): (64, 8, 4, 10),
    (8, 8192, 2560): (64, 8, 4, 12),
    (8, 2048, 8192): (128, 16, 2, 6),
    (8, 7168, 8192): (64, 8, 2, 10),
    (8, 8192, 14336): (64, 8, 2, 11),
    (16, 8192, 2048): (64, 16, 4, 10),
    (16, 8192, 2560): (64, 16, 4, 10),
    (16, 2048, 8192): (64, 16, 1, 11),
    (16, 7168, 8192): (64, 32, 1, 8),
    (16, 8192, 14336): (128, 16, 1, 6),
    (24, 8192, 2048): (64, 32, 4, 8),
    (24, 8192, 2560): (64, 32, 4, 8),
    (24, 2048, 8192): (64, 32, 1, 9),
    (24, 7168, 8192): (64, 32, 1, 9),
    (24, 8192, 14336): (128, 32, 1, 5),
    (32, 8192, 2048): (64, 32, 4, 8),
    (32, 8192, 2560): (64, 32, 4, 8),
    (32, 2048, 8192): (64, 32, 1, 9),
    (32, 7168, 8192): (64, 32, 2, 9),
    (32, 8192, 14336): (64, 32, 2, 8),
}

#: The measured table of each ``(compute capability, drafter TP size)``.
SPLITK_TACTICS_BY_SM_TP: dict[
    tuple[tuple[int, int], int], dict[tuple[int, int, int], tuple[int, int, int, int]]
] = {
    ((10, 3), 8): SPLITK_TACTICS,
    ((10, 3), 4): SPLITK_TACTICS_TP4,
    ((10, 7), 8): SM107_SPLITK_TACTICS,
    ((10, 7), 4): SM107_SPLITK_TACTICS_TP4,
}

#: The kernel's row limit (the dispatcher's ``_MAX_M``); opt-in: 64 with the TP8 drafter's or fc shard's
#: 40..64-row entries (``MUSESPARK_OPEN_TEP8_WIDE64_SPLITK=1``, which needs the split-K kernel's 64-wide
#: tiles). Shapes no table lists still run the built-in GEMM.
MAX_M = 64 if TEP8_WIDE64 and (TEP8_DRAFTER_WIDE or TEP8_FC_WIDE) else 32

_DEVICE_CAPABILITY: tuple[int, int] | None = None


def device_capability() -> tuple[int, int]:
    """This process's CUDA device compute capability, read once.

    On a host without CUDA, where no kernel launches, the target's own sm103.
    """
    global _DEVICE_CAPABILITY
    if _DEVICE_CAPABILITY is None:
        _DEVICE_CAPABILITY = (
            tuple(torch.cuda.get_device_capability()) if torch.cuda.is_available() else (10, 3)
        )
    return _DEVICE_CAPABILITY


def splitk_tactics(
    tp_size: int = 8, capability: tuple[int, int] | None = None
) -> dict[tuple[int, int, int], tuple[int, int, int, int]]:
    """The tactic table of a drafter on a TP group of ``tp_size`` on ``capability``.

    ``capability`` defaults to this process's device (``device_capability``).

    Raises:
        ValueError: if ``SPLITK_TACTICS_BY_SM_TP`` has no table for the pair.
    """
    key = (tuple(device_capability() if capability is None else capability), tp_size)
    table = SPLITK_TACTICS_BY_SM_TP.get(key)
    if table is None:
        raise ValueError(
            f"fixed_low_m_gemm: no split-K tactics for SM {key[0]} at TP {key[1]} "
            f"(supported: {sorted(SPLITK_TACTICS_BY_SM_TP)})"
        )
    return table


def tactic_for(m: int, k: int, n: int, tactics: dict | None = None):
    """The fixed tactic for this shape in ``tactics`` (a ``splitk_tactics`` table; the
    device's TP8 one when None), or None when the shape is not covered."""
    return (splitk_tactics() if tactics is None else tactics).get((m, k, n))


def takes(x: torch.Tensor, weight: torch.Tensor) -> bool:
    """Whether the kernel takes these operands (else the caller runs the built-in GEMM)."""
    return (
        x.dim() == 2
        and weight.dim() == 2
        and x.dtype == weight.dtype == torch.bfloat16
        and x.is_cuda
        and x.device == weight.device
        and x.is_contiguous()
        and weight.is_contiguous()
        and 1 <= x.shape[0] <= MAX_M
        and x.shape[1] == weight.shape[1]
        and x.shape[1] % 128 == 0
        and x.data_ptr() % 32 == 0
        and weight.data_ptr() % 32 == 0
        and not torch.is_grad_enabled()
    )


class FixedLowMGemm(OpWrapper):
    """``x [M, K] @ weight [N, K].T -> [M, N]`` bf16, on the CuTe-DSL split-K kernel at ``tactic``."""

    ARCHS = frozenset({Arch.SM_103})

    note = """
    Metadata consumed: none. Calls
    ``ops/gemm/splitk_gemm.run_splitk_dense`` (flashinfer's ``dense_bf16_gemm_sm100_splitk``, moved here)
    (CuTe-DSL, JIT-compiled per tactic and shape on first use; the first call
    must not be inside a CUDA-graph capture) with the weight as the ``[K, N]``
    column-major view; both operands are passed detached (no autograd).
    Allocates the output per call.

    Call sites (``drafter_fixed_gemms`` / ``drafter_fc_shard``, ``drafter_worker.py``): the
    drafter's convolution coefficient projections, its context K/V projection,
    its TP-sharded ``qkv`` / ``o_proj`` / ``gate_up`` / ``down`` projections, and the
    DFlash ``fc`` shard -- each only at the shapes in the table of the drafter's
    TP size (``splitk_tactics``).

    Numerics: fp32 accumulation, one rounding to bf16; the K-split order is
    the tactic's. Within one bf16 ulp of the fp32 ``reference`` per element
    (the test's band); not bitwise ``F.linear``.

    Hard limits
      M <= 32, K % 128 == 0, bf16, contiguous, 32-byte aligned operands
      (``takes``), no bias.

    Not certified: torch.compile. CUDA-graph capture and replay are exercised
    by the test.
    """

    def __call__(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        tactic: tuple[int, int, int, int],
        enable_pdl: bool,
    ) -> torch.Tensor:
        from .splitk_gemm import (
            SplitKTactic,
            run_splitk_dense,
        )

        out = torch.empty((x.shape[0], weight.shape[0]), dtype=x.dtype, device=x.device)
        # DLPack refuses tensors that require grad; a trainable nn.Linear
        # weight still does under inference mode.
        return run_splitk_dense(
            x.detach(), weight.detach().t(), None, out, enable_pdl, SplitKTactic(*tactic)
        )

    def reference(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        tactic: tuple[int, int, int, int],
        enable_pdl: bool,
    ) -> torch.Tensor:
        """fp32 product, one rounding to bf16. ``tactic`` and ``enable_pdl`` are ignored."""
        with true_fp32_matmul():
            return (x.float() @ weight.float().t()).to(x.dtype)

    def is_valid(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        tactic: tuple[int, int, int, int],
        enable_pdl: bool,
    ) -> None:
        assert takes(x, weight), (
            "operands must be bf16, contiguous, 32-byte aligned, x [M <= 32, K] with K % 128 == 0 "
            f"matching weight [N, K]; got x {tuple(x.shape)} {x.dtype} weight "
            f"{tuple(weight.shape)} {weight.dtype}"
        )
        assert len(tuple(tactic)) == 4, (
            f"tactic must be (mma_m, mma_n, split_k, ab_stages), got {tactic}"
        )


fixed_low_m_gemm = FixedLowMGemm()
