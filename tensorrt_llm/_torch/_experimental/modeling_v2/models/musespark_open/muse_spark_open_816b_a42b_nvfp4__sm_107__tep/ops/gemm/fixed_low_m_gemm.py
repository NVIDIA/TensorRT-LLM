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
``dense_bf16_gemm_sm100_splitk``, with the changes ``splitk_gemm.configure`` sets.

The kernel accumulates in fp32 and rounds once; its K-split differs from
cuBLAS's, so results differ from ``F.linear`` in the last bf16 place on a few
elements -- the same distance from the fp64 product as cuBLAS.
"""

from __future__ import annotations

import functools

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import (
    Arch,
    OpWrapper,
    true_fp32_matmul,
)

#: The split-K tactic ``(mma_m, mma_n, split_k, ab_stages)`` per ``(M, K, N)`` of a
#: tep8 drafter (two trays) on sm107 (212 SMs), chosen with cold weights and PDL on.
#: Shapes not listed are not taken by this op's callers.
SM107_SPLITK_TACTICS: dict[tuple[int, int, int], tuple[int, int, int, int]] = {
    # DFlash 2 convolution coefficient projection (replicated).
    (8, 8192, 2048): (64, 8, 4, 10),
    # drafter qkv_proj per rank, and the fused context K/V projection.
    (8, 8192, 1280): (64, 8, 4, 12),
    # drafter o_proj per rank.
    (8, 1024, 8192): (64, 8, 1, 10),
    # drafter down_proj per rank.
    (8, 3584, 8192): (64, 8, 1, 12),
    # drafter gate_up_proj per rank.
    (8, 8192, 7168): (64, 16, 1, 9),
    # DFlash fc, one TP8 column shard (``drafter_fc_shard``).
    (8, 40960, 1024): (64, 8, 4, 12),
}

#: ``tep8_drafter_wide`` (``configure``) adds the tep8 sm107 drafter's 2..4-request blocks (16, 24
#: and 32 rows), and with ``tep8_wide64`` (the split-K kernel's 64-wide tiles) its 5..8-request
#: blocks (40..64 rows), which otherwise run cuBLAS (nvjet, plus a cublasLt splitKreduce launch on
#: four of the five per-layer projections: 20-23 reduce launches per step on the TEP8 bs4/bs8
#: critical path). Each shape keeps the ``split_k`` of its 8-row cell, so every row of a block gets
#: the bits it gets in a one-request step (the split-K kernel sums the same K slices in the same
#: order whatever the tile widths); against cuBLAS a row differs in the last BF16 place on some
#: elements, as the 8-row cells already do (o_proj and down_proj, split_k 1: equal in isolated
#: tests). Tile widths and stages: the fastest measured behind an early-releasing predecessor with
#: PDL, which is what the drafter's predecessors are (hop, block convolution, window-attention
#: merge, SiLU-mul all release their dependents early: the 8-row split-K GEMMs start 1.5-4.7 us
#: before them in TEP8 bs1 traces).
#: (mma_m, mma_n, split_k, ab_stages) per row count and site: the fastest split-K tile behind an
#: early-releasing predecessor with PDL, measured on sm107 at the 8-row cell's split_k.
_TP8_DRAFTER_WIDE_SHAPES = {
    "conv_coef": (8192, 2048),
    "qkv": (8192, 1280),
    "o_proj": (1024, 8192),
    "down": (3584, 8192),
    "gate_up": (8192, 7168),
}  # qkv: also the context K/V
_TP8_DRAFTER_WIDE_TILES = {
    16: {
        "conv_coef": (64, 16, 4, 10),
        "qkv": (64, 16, 4, 10),
        "o_proj": (64, 16, 1, 11),
        "down": (64, 16, 1, 11),
        "gate_up": (64, 16, 1, 11),
    },
    24: {
        "conv_coef": (64, 32, 4, 8),
        "qkv": (64, 32, 4, 8),
        "o_proj": (64, 32, 1, 9),
        "down": (64, 32, 1, 9),
        "gate_up": (64, 32, 1, 9),
    },
    32: {
        "conv_coef": (64, 32, 4, 8),
        "qkv": (64, 32, 4, 8),
        "o_proj": (64, 32, 1, 9),
        "down": (64, 32, 1, 9),
        "gate_up": (64, 32, 1, 9),
    },
    40: {
        "conv_coef": (64, 64, 4, 5),
        "qkv": (64, 64, 4, 5),
        "o_proj": (64, 64, 1, 7),
        "down": (64, 64, 1, 7),
        "gate_up": (64, 64, 1, 7),
    },
    48: {
        "conv_coef": (64, 64, 4, 5),
        "qkv": (64, 64, 4, 5),
        "o_proj": (64, 64, 1, 7),
        "down": (64, 64, 1, 7),
        "gate_up": (64, 64, 1, 7),
    },
    56: {
        "conv_coef": (64, 64, 4, 5),
        "qkv": (64, 64, 4, 5),
        "o_proj": (64, 64, 1, 7),
        "down": (64, 64, 1, 7),
        "gate_up": (64, 64, 1, 7),
    },
    64: {
        "conv_coef": (64, 64, 4, 5),
        "qkv": (64, 64, 4, 5),
        "o_proj": (64, 64, 1, 7),
        "down": (64, 64, 1, 7),
        "gate_up": (64, 64, 1, 7),
    },
}
SM107_TP8_DRAFTER_WIDE_TACTICS: dict[tuple[int, int, int], tuple[int, int, int, int]] = {
    (_rows, *_TP8_DRAFTER_WIDE_SHAPES[_site]): _tactic
    for _rows, _sites in _TP8_DRAFTER_WIDE_TILES.items()
    for _site, _tactic in _sites.items()
}
#: ``tep8_fc_wide`` (``configure``; ``spec_worker``'s TP8 fc shard above 8 rows) adds the DFlash
#: fc's TP8 column shard ``[rows, 40960] @ [1024, 40960].T`` at 16..32 rows, and with
#: ``tep8_wide64`` (the split-K kernel's M <= 64) 40..64 rows; other row counts run ``F.linear`` on
#: the shard. Each keeps the 8-row cell's split_k (4): a row's bits are its one-request bits, and
#: at 24 and 32 rows they are also the replicated cuBLAS GEMM's (its nvjet 160x64 kernel splits K
#: the same 4 ways; elsewhere 99.4-99.6 % of elements equal, the rest one BF16 ulp). Widths: the
#: fastest behind a plain predecessor (the side chain's torch kernels), 16 rows per tile up to 32
#: rows, 32 above.
SM107_TP8_FC_WIDE_TACTICS: dict[tuple[int, int, int], tuple[int, int, int, int]] = {
    (_rows, 40960, 1024): (64, 16, 4, 10) if _rows <= 32 else (64, 32, 4, 8)
    for _rows in (16, 24, 32, 40, 48, 56, 64)
}
#: ``tep8_drafter_max_rows`` (``configure``; ``tep8_drafter_wide_gemms__MAX_ROWS``): the most rows
#: of a TP8 drafter GEMM on the fixed split-K tactics, 64 by default (above it, 9..16 requests at
#: K = 7, cuBLAS: nvjet split-K plus a cublasLt splitKreduce launch on the convolution coefficient
#: and qkv projections at 72..128 rows, and on gate_up at 96, 16-21 reduce launches per step on the
#: drafter's critical path). Above 64, up to ``TEP8_DRAFTER_WIDE_LIMIT``, the calls of the
#: ``TEP8_DRAFTER_WIDE_SITES`` sites run in ONE launch of the site's 64-row tactic on the
#: ``TEP8_DRAFTER_WIDE_TILE`` tile (``tep8_drafter_wide_tile``, output columns x rows; stages lowered
#: to fit at launch, ``one_launch_tactic``), any token tiles past the first on the grid's second
#: dimension (``splitk_gemm.run_splitk_dense_tiles``, as the TP8 QKV's
#: ``tep8_wide64_splitk__MAX_ROWS``): the 64-row tactic keeps its 8-row cell's split_k, which fixes
#: every element's K slices and their cluster-rank reduction order whatever the tile, so every row
#: gets the bits it gets in a one-request step.
TEP8_DRAFTER_MAX_ROWS = 64
TEP8_DRAFTER_WIDE_LIMIT = 128
#: WIDE_TILE's names -> (mma_m, mma_n); ``64x64`` is the 64-row tactic's own tile, ``64x128`` (the
#: default) takes all rows in one token tile: the 64-row launch's CTA count, each weight slab read once.
TEP8_DRAFTER_WIDE_TILES = {"64x64": (64, 64), "128x64": (128, 64), "64x128": (64, 128)}
TEP8_DRAFTER_WIDE_TILE = (64, 128)
#: The sites of the 65..MAX_ROWS-row launch (``_TP8_DRAFTER_WIDE_SHAPES`` names); the others keep
#: cuBLAS above 64 rows. ``qkv`` covers the fused context K/V projection too (the same shape).
TEP8_DRAFTER_WIDE_SITES: tuple[str, ...] = ("conv_coef", "qkv")
#: The row count a 65..MAX_ROWS-row call's tactic is the one of (and is validated at).
ONE_LAUNCH_ROWS = 64
#: The settings ``configure`` sets (all off: the one-request sm107 TP8 table above as measured).
TEP8_WIDE64 = False
TEP8_DRAFTER_WIDE = False
TEP8_FC_WIDE = False
TEP8_DRAFTER_WIDE64 = False
#: ``tep8_drafter_deep_stages`` (``configure``; ``tep8_drafter_wide_gemms__DEEP_STAGES``): the 16..64-row TP8
#: drafter entries at deeper A/B pipelines, ``(rows, site) -> ab_stages``; tile and split_k stay the table's,
#: so every output keeps its bits (stages change no K slice and no summation order). The wide tactics'
#: stages were the deepest that fit the 232 KB opt-in shared memory (5 stages at a 64x64 split-4 tile, 7 at
#: split 1); the split-K kernel's oversized mode (``splitk_gemm.configure(oversized_smem=True)``, 327 KB on
#: sm107) fits 8 / 10. Measured on VR (vr_gst_bench: chains of cold weights behind a PDL predecessor, us
#: per call at 64 rows): conv_coef 6.30 -> 5.69, qkv 5.64 -> 4.80, o_proj 3.57 -> 3.45, down 8.28 -> 7.93,
#: gate_up 14.45 -> 13.44 (-3.5 us per drafter layer); 48 rows alike; 16 / 32 rows -0.2..-0.5 us on qkv /
#: down / gate_up, the other sites as listed.
_TP8_DRAFTER_DEEP_STAGES: dict[int, dict[str, int]] = {
    16: {"qkv": 12, "down": 14, "gate_up": 14},
    24: {"conv_coef": 10, "qkv": 12, "down": 13, "gate_up": 13},
    32: {"conv_coef": 10, "qkv": 12, "down": 13, "gate_up": 13},
    **{
        rows: {"conv_coef": 8, "qkv": 8, "o_proj": 8, "down": 10, "gate_up": 10}
        for rows in (40, 48, 56, 64)
    },
}
TEP8_DRAFTER_DEEP_STAGES = False
#: ``SM107_SPLITK_TACTICS`` as measured, before ``configure`` adds the wide entries.
_SM107_SPLITK_TACTICS_BASE = dict(SM107_SPLITK_TACTICS)

#: The same for a tep4 drafter (one tray) on sm107 (212 SMs): every TP-sharded projection is
#: twice the tep8 one along its sharded dimension; the replicated convolution coefficient
#: projection is the tep8 one. Each the fastest split-K tactic of its shape in chains of cold
#: weights with PDL on; every one beats cuBLAS at its shape. Besides the one-request block
#: (8 rows) the table lists the blocks of 2..4 requests (16, 24 and 32 rows), each the
#: fastest split-K tactic at that row count over cold weights with PDL on.
SM107_SPLITK_TACTICS_TP4: dict[tuple[int, int, int], tuple[int, int, int, int]] = {
    # DFlash 2 convolution coefficient projection (replicated).
    (8, 8192, 2048): (64, 8, 4, 10),
    # drafter qkv_proj per rank, and the fused context K/V projection.
    (8, 8192, 2560): (64, 8, 4, 12),
    # drafter o_proj per rank.
    (8, 2048, 8192): (128, 16, 2, 6),
    # drafter down_proj per rank.
    (8, 7168, 8192): (64, 8, 2, 10),
    # drafter gate_up_proj per rank.
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
    ((10, 7), 8): SM107_SPLITK_TACTICS,
    ((10, 7), 4): SM107_SPLITK_TACTICS_TP4,
}

#: The kernel's row limit (the dispatcher's ``_MAX_M``): 64 with the TP8 drafter's or fc shard's
#: 40..64-row entries (``tep8_wide64``, which needs the split-K kernel's 64-wide tiles), as
#: ``configure`` sets it. Shapes no table lists still run the built-in GEMM.
MAX_M = 32


def configure(
    *,
    tep8_wide64: bool = False,
    tep8_drafter_wide: bool = False,
    tep8_fc_wide: bool = False,
    tep8_drafter_max_rows: int = 64,
    tep8_drafter_wide_tile: str = "64x128",
    tep8_drafter_wide_sites: tuple[str, ...] = ("conv_coef", "qkv"),
    tep8_drafter_deep_stages: bool = False,
) -> None:
    """Set the wide entries of the sm107 TP8 tactic table and the kernel's row limit.

    ``lever_settings.apply`` sets them from the model's plan before any install (the drafter
    installs read the table then). The table is updated in place: ``SPLITK_TACTICS_BY_SM_TP``
    holds it.

    Args:
        tep8_wide64: also the 40..64-row entries of the two below, and ``MAX_M`` 64 with either
            (needs the split-K kernel's ``wide64``).
        tep8_drafter_wide: the TP8 drafter's 16..32-row blocks (``SM107_TP8_DRAFTER_WIDE_TACTICS``).
        tep8_fc_wide: the DFlash fc shard's 16..32-row calls (``SM107_TP8_FC_WIDE_TACTICS``).
        tep8_drafter_max_rows: ``TEP8_DRAFTER_MAX_ROWS``, 64..``TEP8_DRAFTER_WIDE_LIMIT``; above
            64 (needs ``tep8_wide64`` and ``tep8_drafter_wide``) the 65..max_rows-row calls of
            ``tep8_drafter_wide_sites`` take the site's 64-row tactic on
            ``tep8_drafter_wide_tile``, launched once (``FixedLowMGemm``), and ``MAX_M`` is
            max_rows.
        tep8_drafter_wide_tile: ``TEP8_DRAFTER_WIDE_TILE``, a name of ``TEP8_DRAFTER_WIDE_TILES``.
        tep8_drafter_wide_sites: ``TEP8_DRAFTER_WIDE_SITES``, a non-empty subset of
            ``_TP8_DRAFTER_WIDE_SHAPES``.
        tep8_drafter_deep_stages: ``TEP8_DRAFTER_DEEP_STAGES``: the listed 16..64-row drafter entries
            at the ``_TP8_DRAFTER_DEEP_STAGES`` stage counts (needs ``tep8_drafter_wide``; the 40..64-row
            ones exist with ``tep8_wide64``). ``deep_stage_tactics`` checks they fit at install.

    Raises:
        ValueError: for a ``tep8_drafter_max_rows`` outside 64..``TEP8_DRAFTER_WIDE_LIMIT``, or
            above 64 without ``tep8_wide64`` and ``tep8_drafter_wide``, an unknown tile, an
            empty or unknown site list, or deep stages without ``tep8_drafter_wide``; nothing is
            set then.
    """
    global TEP8_WIDE64, TEP8_DRAFTER_WIDE, TEP8_FC_WIDE, TEP8_DRAFTER_WIDE64, MAX_M
    global TEP8_DRAFTER_MAX_ROWS, TEP8_DRAFTER_WIDE_TILE, TEP8_DRAFTER_WIDE_SITES
    global TEP8_DRAFTER_DEEP_STAGES
    if tep8_drafter_deep_stages and not tep8_drafter_wide:
        raise ValueError(
            "fixed_low_m_gemm: tep8_drafter_deep_stages (tep8_drafter_wide_gemms__DEEP_STAGES) needs "
            "tep8_drafter_wide (the 16..64-row drafter entries it deepens)"
        )
    max_rows = int(tep8_drafter_max_rows)
    if not ONE_LAUNCH_ROWS <= max_rows <= TEP8_DRAFTER_WIDE_LIMIT:
        raise ValueError(
            f"fixed_low_m_gemm: tep8_drafter_max_rows must be in {ONE_LAUNCH_ROWS}.."
            f"{TEP8_DRAFTER_WIDE_LIMIT}, got {tep8_drafter_max_rows}"
        )
    if max_rows > ONE_LAUNCH_ROWS and not (tep8_wide64 and tep8_drafter_wide):
        raise ValueError(
            f"fixed_low_m_gemm: tep8_drafter_max_rows above {ONE_LAUNCH_ROWS} needs tep8_wide64 "
            "and tep8_drafter_wide (the 64-row drafter tactics it launches)"
        )
    if tep8_drafter_wide_tile not in TEP8_DRAFTER_WIDE_TILES:
        raise ValueError(
            f"fixed_low_m_gemm: tep8_drafter_wide_tile must be one of "
            f"{list(TEP8_DRAFTER_WIDE_TILES)}, got {tep8_drafter_wide_tile!r}"
        )
    sites = tuple(tep8_drafter_wide_sites)
    unknown = sorted(set(sites) - set(_TP8_DRAFTER_WIDE_SHAPES))
    if not sites or unknown:
        raise ValueError(
            f"fixed_low_m_gemm: tep8_drafter_wide_sites must be a non-empty subset of "
            f"{list(_TP8_DRAFTER_WIDE_SHAPES)}, got {list(sites)}"
        )
    TEP8_DRAFTER_MAX_ROWS = max_rows
    TEP8_DRAFTER_WIDE_TILE = TEP8_DRAFTER_WIDE_TILES[tep8_drafter_wide_tile]
    TEP8_DRAFTER_WIDE_SITES = sites
    TEP8_WIDE64 = bool(tep8_wide64)
    TEP8_DRAFTER_WIDE = bool(tep8_drafter_wide)
    TEP8_FC_WIDE = bool(tep8_fc_wide)
    TEP8_DRAFTER_WIDE64 = TEP8_DRAFTER_WIDE and TEP8_WIDE64
    SM107_SPLITK_TACTICS.clear()
    SM107_SPLITK_TACTICS.update(_SM107_SPLITK_TACTICS_BASE)
    if TEP8_DRAFTER_WIDE:
        SM107_SPLITK_TACTICS.update(
            {
                key: tactic
                for key, tactic in SM107_TP8_DRAFTER_WIDE_TACTICS.items()
                if key[0] <= 32 or TEP8_WIDE64
            }
        )
    if TEP8_FC_WIDE:
        SM107_SPLITK_TACTICS.update(
            {
                key: tactic
                for key, tactic in SM107_TP8_FC_WIDE_TACTICS.items()
                if key[0] <= 32 or TEP8_WIDE64
            }
        )
    if TEP8_DRAFTER_WIDE64 and TEP8_DRAFTER_MAX_ROWS > ONE_LAUNCH_ROWS:
        # 65..MAX_ROWS rows: the site's 64-row tactic on the WIDE_TILE tile, one launch.
        SM107_SPLITK_TACTICS.update(
            {
                (rows, *_TP8_DRAFTER_WIDE_SHAPES[site]): (
                    *TEP8_DRAFTER_WIDE_TILE,
                    *_TP8_DRAFTER_WIDE_TILES[ONE_LAUNCH_ROWS][site][2:],
                )
                for rows in range(ONE_LAUNCH_ROWS + 1, TEP8_DRAFTER_MAX_ROWS + 1)
                for site in TEP8_DRAFTER_WIDE_SITES
            }
        )
    TEP8_DRAFTER_DEEP_STAGES = bool(tep8_drafter_deep_stages)
    if TEP8_DRAFTER_DEEP_STAGES:
        # After the 65..MAX_ROWS entries (built from the 64-row cell as measured): only stages move.
        for rows, sites in _TP8_DRAFTER_DEEP_STAGES.items():
            for site, stages in sites.items():
                key = (rows, *_TP8_DRAFTER_WIDE_SHAPES[site])
                if key in SM107_SPLITK_TACTICS:
                    SM107_SPLITK_TACTICS[key] = (*SM107_SPLITK_TACTICS[key][:3], stages)
    MAX_M = 64 if TEP8_WIDE64 and (TEP8_DRAFTER_WIDE or TEP8_FC_WIDE) else 32
    if TEP8_DRAFTER_WIDE64 and TEP8_DRAFTER_MAX_ROWS > ONE_LAUNCH_ROWS:
        MAX_M = TEP8_DRAFTER_MAX_ROWS


_DEVICE_CAPABILITY: tuple[int, int] | None = None


def device_capability() -> tuple[int, int]:
    """This process's CUDA device compute capability, read once.

    On a host without CUDA, where no kernel launches, the target's own sm107.
    """
    global _DEVICE_CAPABILITY
    if _DEVICE_CAPABILITY is None:
        _DEVICE_CAPABILITY = (
            tuple(torch.cuda.get_device_capability()) if torch.cuda.is_available() else (10, 7)
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


@functools.cache
def one_launch_tactic(tactic: tuple[int, int, int, int], k: int, n: int):
    """The ``SplitKTactic`` a 65..``TEP8_DRAFTER_MAX_ROWS``-row call of ``[*, k] x [n, k]^T`` at
    ``tactic`` launches: ``tactic`` with its ``ab_stages`` lowered until the split-K kernel accepts
    it at ``ONE_LAUNCH_ROWS`` rows (``run_splitk_dense_tiles`` validates there); tile and split_k
    kept. The 64x64 tile is the 64-row tactic as is. Host code only; ``wide_launch_tactics``
    resolves every listed one at install.

    Raises:
        ValueError: when no stage count fits (or the tile or split_k is refused).
    """
    from .splitk_gemm import SplitKTactic, validate_tactic

    mma_m, mma_n, split_k, stages = tactic
    error = None
    for fit in range(stages, 0, -1):
        candidate = SplitKTactic(mma_m, mma_n, split_k, fit)
        try:
            validate_tactic(candidate, ONE_LAUNCH_ROWS, n, k)
        except ValueError as refused:
            error = refused
            continue
        return candidate
    raise ValueError(
        f"fixed_low_m_gemm: no stage count of tactic {tactic} serves {ONE_LAUNCH_ROWS} x {k} -> "
        f"{n}: {error}"
    )


def wide_launch_tactics(tactics: dict | None = None) -> dict:
    """``(k, n) -> SplitKTactic`` of every 65..``TEP8_DRAFTER_MAX_ROWS``-row entry of ``tactics``
    (the device's TP8 table when None), each resolved by ``one_launch_tactic``: ``spec_worker``'s
    install calls it so that a tile the kernel refuses fails there, not at a step. Empty at the
    default ``TEP8_DRAFTER_MAX_ROWS`` (64)."""
    table = splitk_tactics() if tactics is None else tactics
    return {
        (k, n): one_launch_tactic(tuple(tactic), k, n)
        for (m, k, n), tactic in table.items()
        if m > ONE_LAUNCH_ROWS
    }


def deep_stage_tactics(tactics: dict | None = None) -> dict:
    """``(rows, k, n) -> SplitKTactic`` of every deepened entry of ``tactics`` (the device's TP8 table when
    None), each validated at its rows: ``spec_worker``'s install calls it so that a stage count the device's
    shared memory cannot hold (no oversized mode) fails there, not at a step. Empty without
    ``TEP8_DRAFTER_DEEP_STAGES``.

    Raises:
        ValueError: naming ``tep8_drafter_wide_gemms__DEEP_STAGES`` when an entry does not fit.
    """
    if not TEP8_DRAFTER_DEEP_STAGES:
        return {}
    from .splitk_gemm import SplitKTactic, validate_tactic

    table = splitk_tactics() if tactics is None else tactics
    out = {}
    for rows, sites in _TP8_DRAFTER_DEEP_STAGES.items():
        for site in sites:
            k, n = _TP8_DRAFTER_WIDE_SHAPES[site]
            tactic = table.get((rows, k, n))
            if tactic is None:
                continue
            candidate = SplitKTactic(*tactic)
            try:
                validate_tactic(candidate, rows, n, k)
            except ValueError as refused:
                raise ValueError(
                    f"fixed_low_m_gemm: tep8_drafter_wide_gemms__DEEP_STAGES: the {rows}-row {site} tactic "
                    f"{tuple(tactic)} does not fit this device ({refused})"
                ) from refused
            out[(rows, k, n)] = candidate
    return out


class FixedLowMGemm(OpWrapper):
    """``x [M, K] @ weight [N, K].T -> [M, N]`` bf16, on the CuTe-DSL split-K kernel at ``tactic``."""

    ARCHS = frozenset({Arch.SM_107})

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
      M <= MAX_M (32; 64 with the wide entries; ``TEP8_DRAFTER_MAX_ROWS`` above it, those
      calls in one ``run_splitk_dense_tiles`` launch of the 64-row tactic), K % 128 == 0,
      bf16, contiguous, 32-byte aligned operands (``takes``), no bias.

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
        from .splitk_gemm import SplitKTactic, run_splitk_dense, run_splitk_dense_tiles

        out = torch.empty((x.shape[0], weight.shape[0]), dtype=x.dtype, device=x.device)
        if x.shape[0] > ONE_LAUNCH_ROWS:
            # tep8_drafter_max_rows: 65..MAX_ROWS rows in one launch of the 64-row tactic (the
            # token tiles on the grid's second dimension; each row's one-request bits).
            return run_splitk_dense_tiles(
                x.detach(),
                weight.detach().t(),
                out,
                enable_pdl,
                one_launch_tactic(tuple(tactic), int(weight.shape[1]), int(weight.shape[0])),
            )
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
            f"operands must be bf16, contiguous, 32-byte aligned, x [M <= {MAX_M}, K] with K % 128 == 0 "
            f"matching weight [N, K]; got x {tuple(x.shape)} {x.dtype} weight "
            f"{tuple(weight.shape)} {weight.dtype}"
        )
        assert len(tuple(tactic)) == 4, (
            f"tactic must be (mma_m, mma_n, split_k, ab_stages), got {tactic}"
        )


fixed_low_m_gemm = FixedLowMGemm()
