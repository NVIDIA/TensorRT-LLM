# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Build, load and launch the megakernel library (``msoMegakernel.cu``).

The source is compiled at run time with ``torch.utils.cpp_extension`` for the device's arch-specific target
(``sm_107a`` on Rubin), one torch extension per compile mode, and its ``extern "C"`` entry points are called through
``ctypes``:

* ``check``: the watchdog and the stamp sites read at run time (correctness runs, stamped timelines);
* ``lean``: no watchdog, no stamp code (timed runs; never time a stamped build).

``Workspace`` holds what every launch shares: the arena copies, the counter sets, the single-reader partial buffers
and the pinned watchdog record. ``Megakernel`` holds one plan's role tables and makes one parameter block per
(layer, copy); ``launch`` runs a block with a ``KernelIo`` of per-step pointers. Launches of one ``Megakernel`` must
cycle the copies in order (launch n uses copy n % copies), which ``Megakernel.launch`` enforces; a captured graph
must hold a multiple of ``copies`` launches.
"""

from __future__ import annotations

import ctypes
import functools
import hashlib
import json
import os
import re
import subprocess
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

import torch

from . import schedule

SOURCE = Path(__file__).resolve().parent / "msoMegakernel.cu"
LIBRARY_NAME = "musespark_open_megakernel"
NVCC_FLAGS = (
    "-O3",
    "-std=c++20",
    "--expt-relaxed-constexpr",
    "-lineinfo",
    "--maxrregcount=200",
)
#: Compile modes: extra defines.
MODES: Dict[str, Sequence[str]] = {
    "check": ("-DMSO_MK_LEAN=0", "-DMSO_MK_STAMPS_MODE=0"),
    "lean": ("-DMSO_MK_LEAN=1", "-DMSO_MK_STAMPS_MODE=1"),
    # Stamped lean build (no watchdog, run-time stamp sites): attribution timelines of lean code, never timed.
    "stlean": ("-DMSO_MK_LEAN=1", "-DMSO_MK_STAMPS_MODE=0"),
    # Attribution builds of the QKV run's inner loop (stamped, wrong results): no MMA / no shared-memory loads.
    "dbg_nomma": ("-DMSO_MK_LEAN=0", "-DMSO_MK_STAMPS_MODE=0", "-DMSO_MK_DBG=1"),
    "dbg_nolds": ("-DMSO_MK_LEAN=0", "-DMSO_MK_STAMPS_MODE=0", "-DMSO_MK_DBG=2"),
    "dbg_nomma_s4": (
        "-DMSO_MK_LEAN=0",
        "-DMSO_MK_STAMPS_MODE=0",
        "-DMSO_MK_DBG=1",
        "-DMSO_MK_QKV_SLICES=4",
    ),
    "dbg_nolds_s4": (
        "-DMSO_MK_LEAN=0",
        "-DMSO_MK_STAMPS_MODE=0",
        "-DMSO_MK_DBG=2",
        "-DMSO_MK_QKV_SLICES=4",
    ),
    # The QKV weight in 4 K slices (MSO_MK_QKV_SLICES = 4): check and lean builds.
    "check_s4": ("-DMSO_MK_LEAN=0", "-DMSO_MK_STAMPS_MODE=0", "-DMSO_MK_QKV_SLICES=4"),
    "lean_s4": ("-DMSO_MK_LEAN=1", "-DMSO_MK_STAMPS_MODE=1", "-DMSO_MK_QKV_SLICES=4"),
    # ---- full attention: ring-depth variants (12 / 20 pages = 3 / 5 tile slots; the default is 16) ----
    "lean_r12": ("-DMSO_MK_LEAN=1", "-DMSO_MK_STAMPS_MODE=1", "-DMSO_MK_FA_RING=12"),
    "check_r20": ("-DMSO_MK_LEAN=0", "-DMSO_MK_STAMPS_MODE=0", "-DMSO_MK_FA_RING=20"),
    "lean_r20": ("-DMSO_MK_LEAN=1", "-DMSO_MK_STAMPS_MODE=1", "-DMSO_MK_FA_RING=20"),
    # Attribution builds of the attention loop (timing only, wrong results): no MMAs / no softmax arithmetic.
    "lean_nomma": ("-DMSO_MK_LEAN=1", "-DMSO_MK_STAMPS_MODE=1", "-DMSO_MK_FA_DBG=1"),
    "lean_nosm": ("-DMSO_MK_LEAN=1", "-DMSO_MK_STAMPS_MODE=1", "-DMSO_MK_FA_DBG=2"),
    "lean_nomma_r20": (
        "-DMSO_MK_LEAN=1",
        "-DMSO_MK_STAMPS_MODE=1",
        "-DMSO_MK_FA_DBG=1",
        "-DMSO_MK_FA_RING=20",
    ),
    "lean_stream": ("-DMSO_MK_LEAN=1", "-DMSO_MK_STAMPS_MODE=1", "-DMSO_MK_FA_DBG=3"),
    # The KV tensor map with 128 B L2 promotion (the default is 256 B).
    "lean_p128": ("-DMSO_MK_LEAN=1", "-DMSO_MK_STAMPS_MODE=1", "-DMSO_MK_FA_L2PROMO=128"),
    "lean_stream_p128": (
        "-DMSO_MK_LEAN=1",
        "-DMSO_MK_STAMPS_MODE=1",
        "-DMSO_MK_FA_DBG=3",
        "-DMSO_MK_FA_L2PROMO=128",
    ),
    # V of a page as two 2 KiB boxes (the default is one 4 KiB box).
    "lean_v2box": ("-DMSO_MK_LEAN=1", "-DMSO_MK_STAMPS_MODE=1", "-DMSO_MK_FA_VBOX=0"),
    "lean_stream_v2box": (
        "-DMSO_MK_LEAN=1",
        "-DMSO_MK_STAMPS_MODE=1",
        "-DMSO_MK_FA_DBG=3",
        "-DMSO_MK_FA_VBOX=0",
    ),
    "check_v2box": ("-DMSO_MK_LEAN=0", "-DMSO_MK_STAMPS_MODE=0", "-DMSO_MK_FA_VBOX=0"),
    # ---- the FaAttn O rescale in 16-column chunks (MSO_MK_FA_RC; the stock's rescale_cols=16) ----
    "check_rc16": ("-DMSO_MK_LEAN=0", "-DMSO_MK_STAMPS_MODE=0", "-DMSO_MK_FA_RC=16"),
    "lean_rc16": ("-DMSO_MK_LEAN=1", "-DMSO_MK_STAMPS_MODE=1", "-DMSO_MK_FA_RC=16"),
    "check_s4_rc16": (
        "-DMSO_MK_LEAN=0",
        "-DMSO_MK_STAMPS_MODE=0",
        "-DMSO_MK_QKV_SLICES=4",
        "-DMSO_MK_FA_RC=16",
    ),
    "lean_s4_rc16": (
        "-DMSO_MK_LEAN=1",
        "-DMSO_MK_STAMPS_MODE=1",
        "-DMSO_MK_QKV_SLICES=4",
        "-DMSO_MK_FA_RC=16",
    ),
    # attribution build (timing only, wrong results): the softmax exps as FMULs (is the loop MUFU-bound?)
    "lean_noex2": ("-DMSO_MK_LEAN=1", "-DMSO_MK_STAMPS_MODE=1", "-DMSO_MK_FA_DBG=4"),
    "check_noex2": ("-DMSO_MK_LEAN=0", "-DMSO_MK_STAMPS_MODE=0", "-DMSO_MK_FA_DBG=4"),
    # ---- end FaAttn rescale / exp builds ----
    # ---- attribution builds (timing only, wrong results): QK without its Q_lo MMAs / no PV MMAs ----
    "lean_noqklo": ("-DMSO_MK_LEAN=1", "-DMSO_MK_STAMPS_MODE=1", "-DMSO_MK_FA_DBG=8"),
    "lean_nopv": ("-DMSO_MK_LEAN=1", "-DMSO_MK_STAMPS_MODE=1", "-DMSO_MK_FA_DBG=16"),
    # ---- end QK / PV attribution builds ----
    # Stamped attribution builds (profiles only, wrong results): no MMAs / no MMAs and no softmax arithmetic.
    "check_nomma": ("-DMSO_MK_LEAN=0", "-DMSO_MK_STAMPS_MODE=0", "-DMSO_MK_FA_DBG=1"),
    "check_stream": ("-DMSO_MK_LEAN=0", "-DMSO_MK_STAMPS_MODE=0", "-DMSO_MK_FA_DBG=3"),
    # The softmax exps two per instruction (ex2.approx.f16x2; MSO_MK_FA_EXP16).
    "check_x16": ("-DMSO_MK_LEAN=0", "-DMSO_MK_STAMPS_MODE=0", "-DMSO_MK_FA_EXP16=1"),
    "lean_x16": ("-DMSO_MK_LEAN=1", "-DMSO_MK_STAMPS_MODE=1", "-DMSO_MK_FA_EXP16=1"),
    # ---- end full attention ----
    # ---- MoE hops: the boundaries' other FMA contractions (stages/moe_hops.cuh), check builds ----
    "check_mhfma0": ("-DMSO_MK_LEAN=0", "-DMSO_MK_STAMPS_MODE=0", "-DMSO_MK_MH_FMA=0"),
    "check_mhfma2": ("-DMSO_MK_LEAN=0", "-DMSO_MK_STAMPS_MODE=0", "-DMSO_MK_MH_FMA=2"),
    # the MoE hop boundaries with 2 / 4 units per row (default 1)
    "check_mhu2": ("-DMSO_MK_LEAN=0", "-DMSO_MK_STAMPS_MODE=0", "-DMSO_MK_MH_ROW_UNITS=2"),
    "lean_mhu2": ("-DMSO_MK_LEAN=1", "-DMSO_MK_STAMPS_MODE=1", "-DMSO_MK_MH_ROW_UNITS=2"),
    "check_mhu1": ("-DMSO_MK_LEAN=0", "-DMSO_MK_STAMPS_MODE=0", "-DMSO_MK_MH_ROW_UNITS=1"),
    "lean_mhu1": ("-DMSO_MK_LEAN=1", "-DMSO_MK_STAMPS_MODE=1", "-DMSO_MK_MH_ROW_UNITS=1"),
    # ---- end MoE hops ----
    # ---- integration: stage-guarded builds (MSO_MK_LANES bits: 1 attention half, 2 full attention, 4 experts,
    # 8 MoE hops + hand-offs) ----
    "lean_ah_s4": (
        "-DMSO_MK_LEAN=1",
        "-DMSO_MK_STAMPS_MODE=1",
        "-DMSO_MK_QKV_SLICES=4",
        "-DMSO_MK_LANES=1",
    ),
    "lean_ahfa_s4": (
        "-DMSO_MK_LEAN=1",
        "-DMSO_MK_STAMPS_MODE=1",
        "-DMSO_MK_QKV_SLICES=4",
        "-DMSO_MK_LANES=3",
    ),
    "lean_layer_s4": (
        "-DMSO_MK_LEAN=1",
        "-DMSO_MK_STAMPS_MODE=1",
        "-DMSO_MK_QKV_SLICES=4",
        "-DMSO_MK_LANES=13",
    ),
    "check_layer_s4": (
        "-DMSO_MK_LEAN=0",
        "-DMSO_MK_STAMPS_MODE=0",
        "-DMSO_MK_QKV_SLICES=4",
        "-DMSO_MK_LANES=13",
    ),
    # ---- end integration ----
    # ---- the MoE boundaries' row-norm exchange (MSO_MK_MH_XCHG; 0 = every warp polls the arena, 1 = one polling warp
    # (the default), 2 = cluster DSMEM) ----
    "check_mhx0": ("-DMSO_MK_LEAN=0", "-DMSO_MK_STAMPS_MODE=0", "-DMSO_MK_MH_XCHG=0"),
    "lean_mhx0": ("-DMSO_MK_LEAN=1", "-DMSO_MK_STAMPS_MODE=1", "-DMSO_MK_MH_XCHG=0"),
    "check_mhx1": ("-DMSO_MK_LEAN=0", "-DMSO_MK_STAMPS_MODE=0", "-DMSO_MK_MH_XCHG=1"),
    "lean_mhx1": ("-DMSO_MK_LEAN=1", "-DMSO_MK_STAMPS_MODE=1", "-DMSO_MK_MH_XCHG=1"),
    "check_mhx2": ("-DMSO_MK_LEAN=0", "-DMSO_MK_STAMPS_MODE=0", "-DMSO_MK_MH_XCHG=2"),
    "lean_mhx2": ("-DMSO_MK_LEAN=1", "-DMSO_MK_STAMPS_MODE=1", "-DMSO_MK_MH_XCHG=2"),
    "check_mhu2x2": (
        "-DMSO_MK_LEAN=0",
        "-DMSO_MK_STAMPS_MODE=0",
        "-DMSO_MK_MH_ROW_UNITS=2",
        "-DMSO_MK_MH_XCHG=2",
    ),
    "lean_mhu2x2": (
        "-DMSO_MK_LEAN=1",
        "-DMSO_MK_STAMPS_MODE=1",
        "-DMSO_MK_MH_ROW_UNITS=2",
        "-DMSO_MK_MH_XCHG=2",
    ),
    # F1's first norm over the whole row in every F1 unit (MSO_MK_MH_F1FULL = 1), with the exchanges above
    "check_mhf": ("-DMSO_MK_LEAN=0", "-DMSO_MK_STAMPS_MODE=0", "-DMSO_MK_MH_F1FULL=1"),
    "lean_mhf": ("-DMSO_MK_LEAN=1", "-DMSO_MK_STAMPS_MODE=1", "-DMSO_MK_MH_F1FULL=1"),
    "check_mhx1f": (
        "-DMSO_MK_LEAN=0",
        "-DMSO_MK_STAMPS_MODE=0",
        "-DMSO_MK_MH_XCHG=1",
        "-DMSO_MK_MH_F1FULL=1",
    ),
    "lean_mhx1f": (
        "-DMSO_MK_LEAN=1",
        "-DMSO_MK_STAMPS_MODE=1",
        "-DMSO_MK_MH_XCHG=1",
        "-DMSO_MK_MH_F1FULL=1",
    ),
    "check_mhu2x2f": (
        "-DMSO_MK_LEAN=0",
        "-DMSO_MK_STAMPS_MODE=0",
        "-DMSO_MK_MH_ROW_UNITS=2",
        "-DMSO_MK_MH_XCHG=2",
        "-DMSO_MK_MH_F1FULL=1",
    ),
    "lean_mhu2x2f": (
        "-DMSO_MK_LEAN=1",
        "-DMSO_MK_STAMPS_MODE=1",
        "-DMSO_MK_MH_ROW_UNITS=2",
        "-DMSO_MK_MH_XCHG=2",
        "-DMSO_MK_MH_F1FULL=1",
    ),
    # ---- end row-norm exchange ----
    # ---- stamped lean K-slice build (lean code, run-time stamp sites; attribution only, never timed) ----
    "stlean_s4": ("-DMSO_MK_LEAN=1", "-DMSO_MK_STAMPS_MODE=0", "-DMSO_MK_QKV_SLICES=4"),
    # ---- the same with clock64 stamps (MSO_MK_SY_CLK = 1; per-CTA %globaltimer anchors; attribution only) ----
    "stlean_clk_s4": (
        "-DMSO_MK_LEAN=1",
        "-DMSO_MK_STAMPS_MODE=0",
        "-DMSO_MK_QKV_SLICES=4",
        "-DMSO_MK_SY_CLK=1",
    ),
    # the vector poll as one loop (MSO_MK_SY_POLL = 1): timed, clock-stamped and check builds
    "lean_sypoll_s4": (
        "-DMSO_MK_LEAN=1",
        "-DMSO_MK_STAMPS_MODE=1",
        "-DMSO_MK_QKV_SLICES=4",
        "-DMSO_MK_SY_POLL=1",
    ),
    "stlean_clk_sypoll_s4": (
        "-DMSO_MK_LEAN=1",
        "-DMSO_MK_STAMPS_MODE=0",
        "-DMSO_MK_QKV_SLICES=4",
        "-DMSO_MK_SY_CLK=1",
        "-DMSO_MK_SY_POLL=1",
    ),
    "check_sypoll_s4": (
        "-DMSO_MK_LEAN=0",
        "-DMSO_MK_STAMPS_MODE=0",
        "-DMSO_MK_QKV_SLICES=4",
        "-DMSO_MK_SY_POLL=1",
    ),
    # the stamped lean builds without the per-unit host-memory progress store (MSO_MK_SY_PROG = 0)
    "stlean_np_s4": (
        "-DMSO_MK_LEAN=1",
        "-DMSO_MK_STAMPS_MODE=0",
        "-DMSO_MK_QKV_SLICES=4",
        "-DMSO_MK_SY_PROG=0",
    ),
    "stlean_clk_np_s4": (
        "-DMSO_MK_LEAN=1",
        "-DMSO_MK_STAMPS_MODE=0",
        "-DMSO_MK_QKV_SLICES=4",
        "-DMSO_MK_SY_CLK=1",
        "-DMSO_MK_SY_PROG=0",
    ),
    # per-unit stamps only (MSO_MK_SY_LITE = 1: no sub-stamps, no poll probe, no stamped-only barriers),
    # no progress store
    "stlean_lite_s4": (
        "-DMSO_MK_LEAN=1",
        "-DMSO_MK_STAMPS_MODE=0",
        "-DMSO_MK_QKV_SLICES=4",
        "-DMSO_MK_SY_PROG=0",
        "-DMSO_MK_SY_LITE=1",
    ),
    # ---- end clock64 / poll builds ----
    # ---- FaAttn with P in shared memory and cross-warpgroup QK issue (MSO_MK_FA_XQ) ----
    "check_xq": ("-DMSO_MK_LEAN=0", "-DMSO_MK_STAMPS_MODE=0", "-DMSO_MK_FA_XQ=1"),
    "lean_xq": ("-DMSO_MK_LEAN=1", "-DMSO_MK_STAMPS_MODE=1", "-DMSO_MK_FA_XQ=1"),
    "check_xq_s4": (
        "-DMSO_MK_LEAN=0",
        "-DMSO_MK_STAMPS_MODE=0",
        "-DMSO_MK_QKV_SLICES=4",
        "-DMSO_MK_FA_XQ=1",
    ),
    "lean_xq_s4": (
        "-DMSO_MK_LEAN=1",
        "-DMSO_MK_STAMPS_MODE=1",
        "-DMSO_MK_QKV_SLICES=4",
        "-DMSO_MK_FA_XQ=1",
    ),
    "stlean_xq_s4": (
        "-DMSO_MK_LEAN=1",
        "-DMSO_MK_STAMPS_MODE=0",
        "-DMSO_MK_QKV_SLICES=4",
        "-DMSO_MK_FA_XQ=1",
    ),
    # ---- end FA_XQ builds ----
    # SM-stable roles: the CTA's logical index is %smid (MSO_MK_I2B_SMID)
    "lean_sm_s4": (
        "-DMSO_MK_LEAN=1",
        "-DMSO_MK_STAMPS_MODE=1",
        "-DMSO_MK_QKV_SLICES=4",
        "-DMSO_MK_I2B_SMID=1",
    ),
    "stlean_sm_s4": (
        "-DMSO_MK_LEAN=1",
        "-DMSO_MK_STAMPS_MODE=0",
        "-DMSO_MK_QKV_SLICES=4",
        "-DMSO_MK_I2B_SMID=1",
    ),
    "check_sm_s4": (
        "-DMSO_MK_LEAN=0",
        "-DMSO_MK_STAMPS_MODE=0",
        "-DMSO_MK_QKV_SLICES=4",
        "-DMSO_MK_I2B_SMID=1",
    ),
    # the prologue with MSO_MK_I2B_PRO = 0 (thread 0 also copies a parameter field after its entry weight requests)
    "lean_i2bp0_s4": (
        "-DMSO_MK_LEAN=1",
        "-DMSO_MK_STAMPS_MODE=1",
        "-DMSO_MK_QKV_SLICES=4",
        "-DMSO_MK_I2B_PRO=0",
    ),
    "stlean_i2bp0_s4": (
        "-DMSO_MK_LEAN=1",
        "-DMSO_MK_STAMPS_MODE=0",
        "-DMSO_MK_QKV_SLICES=4",
        "-DMSO_MK_I2B_PRO=0",
    ),
    # ---- end SM-stable / prologue builds ----
    # ---- ``*_z1*`` builds with the BF16 experts stage (namespace z1) of layers 0 / 61 (MSO_MK_LANES bit 16;
    # MSO_MK_BF_N = its MMA N) ----
    "check_z1": ("-DMSO_MK_LEAN=0", "-DMSO_MK_STAMPS_MODE=0", "-DMSO_MK_LANES=31"),
    "lean_z1": ("-DMSO_MK_LEAN=1", "-DMSO_MK_STAMPS_MODE=1", "-DMSO_MK_LANES=31"),
    "stlean_z1": ("-DMSO_MK_LEAN=1", "-DMSO_MK_STAMPS_MODE=0", "-DMSO_MK_LANES=31"),
    "check_z1n16": (
        "-DMSO_MK_LEAN=0",
        "-DMSO_MK_STAMPS_MODE=0",
        "-DMSO_MK_LANES=31",
        "-DMSO_MK_BF_N=16",
    ),
    "lean_z1n16": (
        "-DMSO_MK_LEAN=1",
        "-DMSO_MK_STAMPS_MODE=1",
        "-DMSO_MK_LANES=31",
        "-DMSO_MK_BF_N=16",
    ),
    "check_z1_s4": (
        "-DMSO_MK_LEAN=0",
        "-DMSO_MK_STAMPS_MODE=0",
        "-DMSO_MK_QKV_SLICES=4",
        "-DMSO_MK_LANES=31",
    ),
    "lean_z1_s4": (
        "-DMSO_MK_LEAN=1",
        "-DMSO_MK_STAMPS_MODE=1",
        "-DMSO_MK_QKV_SLICES=4",
        "-DMSO_MK_LANES=31",
    ),
    "stlean_z1_s4": (
        "-DMSO_MK_LEAN=1",
        "-DMSO_MK_STAMPS_MODE=0",
        "-DMSO_MK_QKV_SLICES=4",
        "-DMSO_MK_LANES=31",
    ),
    # two k-blocks (32 KiB) per weight request and unit (MSO_MK_BF_UKB)
    "check_z1u2": (
        "-DMSO_MK_LEAN=0",
        "-DMSO_MK_STAMPS_MODE=0",
        "-DMSO_MK_LANES=31",
        "-DMSO_MK_BF_UKB=2",
    ),
    "lean_z1u2": (
        "-DMSO_MK_LEAN=1",
        "-DMSO_MK_STAMPS_MODE=1",
        "-DMSO_MK_LANES=31",
        "-DMSO_MK_BF_UKB=2",
    ),
    "stlean_z1u2": (
        "-DMSO_MK_LEAN=1",
        "-DMSO_MK_STAMPS_MODE=0",
        "-DMSO_MK_LANES=31",
        "-DMSO_MK_BF_UKB=2",
    ),
    # four k-blocks (64 KiB) per request, two ring stages
    "check_z1u4": (
        "-DMSO_MK_LEAN=0",
        "-DMSO_MK_STAMPS_MODE=0",
        "-DMSO_MK_LANES=31",
        "-DMSO_MK_BF_UKB=4",
    ),
    "lean_z1u4": (
        "-DMSO_MK_LEAN=1",
        "-DMSO_MK_STAMPS_MODE=1",
        "-DMSO_MK_LANES=31",
        "-DMSO_MK_BF_UKB=4",
    ),
    # the whole-layer builds (QKV slices 4) with two / four k-blocks per BF16 request
    "check_z1u2_s4": (
        "-DMSO_MK_LEAN=0",
        "-DMSO_MK_STAMPS_MODE=0",
        "-DMSO_MK_QKV_SLICES=4",
        "-DMSO_MK_LANES=31",
        "-DMSO_MK_BF_UKB=2",
    ),
    "lean_z1u2_s4": (
        "-DMSO_MK_LEAN=1",
        "-DMSO_MK_STAMPS_MODE=1",
        "-DMSO_MK_QKV_SLICES=4",
        "-DMSO_MK_LANES=31",
        "-DMSO_MK_BF_UKB=2",
    ),
    "check_z1u4_s4": (
        "-DMSO_MK_LEAN=0",
        "-DMSO_MK_STAMPS_MODE=0",
        "-DMSO_MK_QKV_SLICES=4",
        "-DMSO_MK_LANES=31",
        "-DMSO_MK_BF_UKB=4",
    ),
    "lean_z1u4_s4": (
        "-DMSO_MK_LEAN=1",
        "-DMSO_MK_STAMPS_MODE=1",
        "-DMSO_MK_QKV_SLICES=4",
        "-DMSO_MK_LANES=31",
        "-DMSO_MK_BF_UKB=4",
    ),
    # ---- end BF16 experts builds ----
}

# ---- builds with the served experts flags fixed at compile time (stages/experts_defs.cuh MSO_MK_IC_MOE_FIX) ----
#: The experts descriptor flags of every served NVFP4 layer (``LayerKernels`` MoE half with the serving defaults):
#: MH_F2 | LOGITS_COOP | QUANT_FIRST | E3_FC2_PAIRS | W2_FC2_ONE_ELECT | cap (3 early requests, 6 units in flight).
IC_MOE_FLAGS = 0x0630040008308000
#: Every flag bit fixed (mask -1): a launch whose experts flags differ traps.
IC_MOE_DEFINES = (
    "-DMSO_MK_IC_MOE_FIX=1",
    "-DMSO_MK_IC_MOE_MASK=-1ll",
    f"-DMSO_MK_IC_MOE_VAL={IC_MOE_FLAGS:#x}ll",
)
#: ``kPIFlags`` bits no served layer sets (msoMegakernel.cu MSO_MK_IC_PI_ZERO): 26 (kIFlagZ1FinalNorm, edges only),
#: 32-36 (FaAttn variants outside the served 0xE030000000), 40-44 (the kIFlagT3Canary* canaries and kIFlagT3XchgVec,
#: outside the served set 45-49), 56-58 (kIFlagW1WarmQkv / kIFlagW1WarmOproj / kIFlagW1OprojAfterQ). A launch that sets
#: one traps.
IC_PI_ZERO = (1 << 26) | (0x1F << 32) | (0x1F << 40) | (0x7 << 56)
IC_PI_DEFINES = (f"-DMSO_MK_IC_PI_ZERO={IC_PI_ZERO:#x}ll",)
#: ``kPIFlags`` bits every served layer sets (msoMegakernel.cu MSO_MK_IC_PI_ONE): 1 (o_proj into the MoE hop), 2 (r'
#: through global memory) and 45-49 (kIFlagT3WaitFirst .. kIFlagT3WaitFirstQkv). A launch that clears one traps.
IC_PI_ONE = (1 << 1) | (1 << 2) | (0x1F << 45)
IC_PI_ONE_DEFINES = (f"-DMSO_MK_IC_PI_ONE={IC_PI_ONE:#x}ll",)
#: The ``IC_VARIANTS`` builds of a base build ``<kind>_s4`` are ``<kind>_<variant>_s4`` (the ``_s4`` suffix and the
#: ``lean`` / ``stlean`` prefix are what ``Library`` checks against the compiled QKV slices and lean / stamps settings).
IC_VARIANTS = {
    "icf": IC_MOE_DEFINES,
    "icp": IC_PI_DEFINES,
    "icfp": IC_MOE_DEFINES + IC_PI_DEFINES,
    "icfpo": IC_MOE_DEFINES + IC_PI_DEFINES + IC_PI_ONE_DEFINES,
}
# The combined served configuration. The experts flags of every served NVFP4 layer: the rebuilt issue loops
# (``serving.LX_MOE_FLAGS``, bits 33-34) and the stock queue at ``serving.PX_LEVEL`` 3 (``serving.PX_MOE_FLAGS``: queue
# + MMA loop + 8-stage ring, bits 35 / 17 / 18) on top of IC_MOE_FLAGS; the builds ``<kind>_icfpox_s4`` fix this word
# and the kPIFlags bits above, so the byte / tile cut and the default issue loops drop out of the build. A launch with
# other experts flags traps.
CB_MOE_FLAGS = IC_MOE_FLAGS | 0x600000000 | 0x800060000
CB_MOE_DEFINES = (
    "-DMSO_MK_IC_MOE_FIX=1",
    "-DMSO_MK_IC_MOE_MASK=-1ll",
    f"-DMSO_MK_IC_MOE_VAL={CB_MOE_FLAGS:#x}ll",
)
IC_VARIANTS["icfpox"] = CB_MOE_DEFINES + IC_PI_DEFINES + IC_PI_ONE_DEFINES
#: ``kPIFlags`` bits of the served one-launch full layers that only the full-attention stages read (msoMegakernel.cu
#: MSO_MK_IC_FA_ZERO / MSO_MK_IC_FA_ONE, folded by icFaFlags in stages/fullAttention.cuh): fixed to 0 = 6 (reversed unit
#: order), 9 (LateLight) and 30-31 (its weight); fixed to 1 = 0 (FaMerge output into the arena), 5 (QKV into the arena),
#: 28-29 (kIFlagF3FaPingPong, kIFlagF3FaRefillSib), 37-38 (kIFlagFaEarlyMeta, kIFlagFaQPrep). The FaAttn / FaMerge
#: entry points trap on a launch whose bits differ. ``<kind>_icfpoxa_s4`` = icfpox + these.
CB_FA_ZERO = (1 << 6) | (1 << 9) | (0x3 << 30)
CB_FA_ONE = (1 << 0) | (1 << 5) | (0x3 << 28) | (0x3 << 37)
CB_FA_DEFINES = (f"-DMSO_MK_IC_FA_ZERO={CB_FA_ZERO:#x}ll", f"-DMSO_MK_IC_FA_ONE={CB_FA_ONE:#x}ll")
IC_VARIANTS["icfpoxa"] = IC_VARIANTS["icfpox"] + CB_FA_DEFINES
# ---- the combined build with the one-launch full layer's FaMerge summing its row's gate itself (MSO_MK_FT_GATE;
# the full-layer plans then hold no gate QkvSum units, ``LayerKernels`` reads ``ft_gate(mode)``) ----
FT_GATE_DEFINES = ("-DMSO_MK_FT_GATE=1",)
IC_VARIANTS["icfpoxft"] = IC_VARIANTS["icfpox"] + FT_GATE_DEFINES
# ... and FaMerge's merge code once on zeros before its polls (MSO_MK_FT_DRY: the real pass runs warm)
FT_DRY_DEFINES = ("-DMSO_MK_FT_DRY=1",)
IC_VARIANTS["icfpoxftd"] = IC_VARIANTS["icfpoxft"] + FT_DRY_DEFINES
# ---- end FaMerge gate builds ----
# ---- the combined build with the full-attention fold (``icfpoxa``) and FaMerge's own gate (``icfpoxft``) ----
IC_VARIANTS["icfpoxaft"] = IC_VARIANTS["icfpoxa"] + FT_GATE_DEFINES
# ---- end icfpoxaft ----
for _kind in ("lean", "stlean", "check"):
    for _variant, _ic_defs in IC_VARIANTS.items():
        MODES[f"{_kind}_{_variant}_s4"] = tuple(MODES[f"{_kind}_s4"]) + _ic_defs
# ---- ``rt<m>`` builds: the experts front's own code (stages/experts_defs.cuh MSO_MK_RT, a bit mask: 1 the top 8 from
# sorted warp lanes, 2 the placement without CTA barriers, 4 the early requests from the placement, 8 the X gather
# inside route() with QuantFirst, 16 no debug-block code in the lean builds, 32 the window's per-expert scalars copied
# to shared memory at stage entry, 64 with 8: the X gather by cp.async into shared memory). ``<kind>_icfpox_rt<m>_s4``
# is the ``icfpox`` combined build with rt level m; ``check_rt<m>`` / ``lean_rt<m>`` the plain builds (the one-GPU
# stage harness). ----
RT_LEVELS = (1, 3, 7, 8, 11, 15, 16, 24, 40, 56, 72, 88)
for _rt in RT_LEVELS:
    for _kind in ("lean", "stlean", "check"):
        MODES[f"{_kind}_icfpox_rt{_rt}_s4"] = tuple(MODES[f"{_kind}_icfpox_s4"]) + (
            f"-DMSO_MK_RT={_rt}",
        )
    for _kind in ("check", "lean"):
        MODES[f"{_kind}_rt{_rt}"] = tuple(MODES[_kind]) + (f"-DMSO_MK_RT={_rt}",)
# ---- end rt builds ----
# ---- rt levels on the ``icfpoxaft`` combined build (``<kind>_icfpoxaft_rt<m>_s4`` = icfpoxaft + MSO_MK_RT=m) ----
for _rt in RT_LEVELS:
    for _kind in ("lean", "stlean", "check"):
        MODES[f"{_kind}_icfpoxaft_rt{_rt}_s4"] = tuple(MODES[f"{_kind}_icfpoxaft_s4"]) + (
            f"-DMSO_MK_RT={_rt}",
        )
# ---- end icfpoxaft rt builds ----
# ---- ``ld<m>`` builds: code a served lean launch never executes, compiled out of the lean builds (msoMegakernel.cu
# MSO_MK_LD, a bit mask; no effect in check builds): 1 the kernel Flags word fixed to the served one, 2 no check / test
# outputs, 4 the dispatchers with the served stages only, 8 the sliding-window stages' served bits and parameters, 16
# QkvSum and the full-attention stages with the heads-first pack and one o_proj replica.
# ``<kind>_icfpoxaft_rt16_ld<m>_s4`` is the ``icfpoxaft`` combined build with rt level 16 and ld level m. ----
LD_LEVELS = (1, 2, 3, 4, 7, 8, 9, 11, 16, 24, 25, 27, 31, 32, 59, 63)
# The higher bits, on top of ld31: 64 one runRole instance in the kernel body, 128 the one-piece weight request written
# out (needs bit 1), 256 served constants (copy neighbours without a division, the o_proj replica mask, no kMdOut copy
# under MhF2).
LD2_LEVELS = (95, 159, 223, 287, 479)
LD_LEVELS = LD_LEVELS + LD2_LEVELS
for _ld in LD_LEVELS:
    for _kind in ("lean", "stlean", "check"):
        MODES[f"{_kind}_icfpoxaft_rt16_ld{_ld}_s4"] = tuple(MODES[f"{_kind}_icfpoxaft_rt16_s4"]) + (
            f"-DMSO_MK_LD={_ld}",
        )
# ---- end ld ----
# ---- ``ue<m>`` builds: the px queue's per-unit pace (stages/experts_defs.cuh MSO_MK_UE, a bit mask; px instantiations
# only, the same tensor operations in the same order): 1 the h loader from the stage's start, 2 the h loader from the
# CTA's last FC1 tile (CTAs with two or more FC1 tiles; 1 and 2 exclude each other), 4 no in-flight cap on the weight
# ring, 8 one k-block per MMA issue. ``<kind>_icfpoxaft_rt16_ld479_ue<m>_s4`` is the ld479 build with ue level m;
# ``check_ue<m>`` / ``lean_ue<m>`` the plain builds (the one-GPU stage harness). ----
UE_LEVELS = tuple(m for m in range(1, 16) if m & 3 != 3)
for _ue in UE_LEVELS:
    for _kind in ("lean", "stlean", "check"):
        MODES[f"{_kind}_icfpoxaft_rt16_ld479_ue{_ue}_s4"] = tuple(
            MODES[f"{_kind}_icfpoxaft_rt16_ld479_s4"]
        ) + (f"-DMSO_MK_UE={_ue}",)
    for _kind in ("check", "lean"):
        MODES[f"{_kind}_ue{_ue}"] = tuple(MODES[_kind]) + (f"-DMSO_MK_UE={_ue}",)
    # the one-GPU stage harness's low-perturbation stamped chain (clock64 stamps, MSO_MK_SY_CLK)
    MODES[f"stlean_clk_ue{_ue}_s4"] = tuple(MODES["stlean_clk_s4"]) + (f"-DMSO_MK_UE={_ue}",)
# ---- end ue ----
# ---- ``et<m>`` builds: the px path's tail (stages/experts_defs.cuh MSO_MK_ET, a bit mask; px only, the same values):
# 1 the FC2 epilogue in phases across the 8 tokens, 2 the final unit's loads after its first pick's sums at once, 4
# every final load from the start (2 and 4 exclude each other). ``<kind>_icfpoxaft_rt16_ld479_ue10_et<m>_s4`` is the
# ld479 + ue 10 build with et level m; ``check_ue10_et<m>`` / ``lean_ue10_et<m>`` the plain builds (the one-GPU stage
# harness); ``stlean_clk_ue10_et<m>_s4`` its clock-stamped chain. ----
ET_LEVELS = (1, 2, 3, 4, 5)
for _et in ET_LEVELS:
    for _kind in ("lean", "stlean", "check"):
        MODES[f"{_kind}_icfpoxaft_rt16_ld479_ue10_et{_et}_s4"] = tuple(
            MODES[f"{_kind}_icfpoxaft_rt16_ld479_ue10_s4"]
        ) + (f"-DMSO_MK_ET={_et}",)
    for _kind in ("check", "lean"):
        MODES[f"{_kind}_ue10_et{_et}"] = tuple(MODES[f"{_kind}_ue10"]) + (f"-DMSO_MK_ET={_et}",)
    MODES[f"stlean_clk_ue10_et{_et}_s4"] = tuple(MODES["stlean_clk_ue10_s4"]) + (
        f"-DMSO_MK_ET={_et}",
    )
# ---- end et ----
# ---- per-unit-only stamped builds (MSO_MK_SY_LITE) of the combined variants: attribution only ----
for _variant in ("icfpox", "icfpoxft", "icfpoxftd", "icfpoxa", "icfpoxaft"):
    MODES[f"stlean_lite_{_variant}_s4"] = tuple(MODES["stlean_lite_s4"]) + IC_VARIANTS[_variant]
# ---- ``nb<m>`` builds: one-exchange MoE boundaries (stages/moe_hops_layout.cuh MSO_MK_NB, a bit mask: 1 MhH1, 2 MhF1;
# the first norm bitwise, the second norm's variance from the FP32 gated residual, so not bitwise).
# ``<kind>_icfpoxaft_rt16_ld479_ue10_nb<m>_s4`` is the ue 10 build with nb level m; ``check_nb<m>`` the plain check
# build (the 8-rank hop tests); ``stlean_lite_icfpoxaft_rt16_ld479_ue10[_nb<m>]_s4`` the per-unit-only stamped builds
# (attribution only). ----
NB_LEVELS = (1, 2, 3)
_NB_LITE_BASE = "stlean_lite_icfpoxaft_rt16_ld479_ue10_s4"
if _NB_LITE_BASE not in MODES:
    MODES[_NB_LITE_BASE] = tuple(MODES["stlean_lite_icfpoxaft_s4"]) + (
        "-DMSO_MK_RT=16",
        "-DMSO_MK_LD=479",
        "-DMSO_MK_UE=10",
    )
for _nb in NB_LEVELS:
    for _kind in ("lean", "check"):
        MODES[f"{_kind}_icfpoxaft_rt16_ld479_ue10_nb{_nb}_s4"] = tuple(
            MODES[f"{_kind}_icfpoxaft_rt16_ld479_ue10_s4"]
        ) + (f"-DMSO_MK_NB={_nb}",)
    MODES[f"stlean_lite_icfpoxaft_rt16_ld479_ue10_nb{_nb}_s4"] = tuple(MODES[_NB_LITE_BASE]) + (
        f"-DMSO_MK_NB={_nb}",
    )
    MODES[f"check_nb{_nb}"] = tuple(MODES["check"]) + (f"-DMSO_MK_NB={_nb}",)


def nb_level(mode: str) -> int:
    """A build mode's ``MSO_MK_NB`` mask (0 without it)."""
    for flag in MODES[mode]:
        if flag.startswith("-DMSO_MK_NB="):
            return int(flag.split("=", 1)[1])
    return 0


# ---- end nb ----
# ---- et and nb together on the ue 10 build: ``<kind>_icfpoxaft_rt16_ld479_ue10_et<m>_nb<n>_s4`` is the et mode
# ``<kind>_icfpoxaft_rt16_ld479_ue10_et<m>_s4`` with nb level n (the two masks own disjoint stages: et the experts
# stage's px tail, nb the MoE boundaries). ----
for _et in ET_LEVELS:
    for _nb in NB_LEVELS:
        for _kind in ("lean", "check"):
            MODES[f"{_kind}_icfpoxaft_rt16_ld479_ue10_et{_et}_nb{_nb}_s4"] = tuple(
                MODES[f"{_kind}_icfpoxaft_rt16_ld479_ue10_et{_et}_s4"]
            ) + (f"-DMSO_MK_NB={_nb}",)
# ---- end et + nb builds ----
# ---- per-unit-only stamped builds (MSO_MK_SY_LITE) of the et and et + nb builds, attribution only:
# ``stlean_lite_icfpoxaft_rt16_ld479_ue10_et<m>[_nb<n>]_s4`` is the ue 10 lite build with et level m
# (and nb level n). ----
for _et in ET_LEVELS:
    MODES[f"stlean_lite_icfpoxaft_rt16_ld479_ue10_et{_et}_s4"] = tuple(MODES[_NB_LITE_BASE]) + (
        f"-DMSO_MK_ET={_et}",
    )
    for _nb in NB_LEVELS:
        MODES[f"stlean_lite_icfpoxaft_rt16_ld479_ue10_et{_et}_nb{_nb}_s4"] = tuple(
            MODES[f"stlean_lite_icfpoxaft_rt16_ld479_ue10_et{_et}_s4"]
        ) + (f"-DMSO_MK_NB={_nb}",)
# ---- end stamped et builds ----
# ---- ``pb<p>`` builds: B-fragment input polls with one 256-bit load per fragment (msoMegakernel.cu MSO_MK_PB, a bit
# mask: 1 MhUnpool's m, 2 MhPool's x_ffn, 4 QkvRun's early x, 8 AhOproj's input; the same words, the same fragment
# values).
# ``<kind>_icfpoxaft_rt16_ld479_ue10_et5[_nb3]_pb<p>_s4`` is the et5 (+ nb3) build with pb level p (lean / check), and
# ``stlean_lite_icfpoxaft_rt16_ld479_ue10_et5_nb3_pb<p>_s4`` its per-unit-only stamped build (attribution only). ----
PB_LEVELS = (1, 2, 4, 5, 8, 13, 15)
for _pb in PB_LEVELS:
    for _base in ("icfpoxaft_rt16_ld479_ue10_et5", "icfpoxaft_rt16_ld479_ue10_et5_nb3"):
        for _kind in ("lean", "check"):
            MODES[f"{_kind}_{_base}_pb{_pb}_s4"] = tuple(MODES[f"{_kind}_{_base}_s4"]) + (
                f"-DMSO_MK_PB={_pb}",
            )
    MODES[f"stlean_lite_icfpoxaft_rt16_ld479_ue10_et5_nb3_pb{_pb}_s4"] = tuple(
        MODES["stlean_lite_icfpoxaft_rt16_ld479_ue10_et5_nb3_s4"]
    ) + (f"-DMSO_MK_PB={_pb}",)


def pb_level(mode: str) -> int:
    """A build mode's ``MSO_MK_PB`` mask (0 without it)."""
    for flag in MODES[mode]:
        if flag.startswith("-DMSO_MK_PB="):
            return int(flag.split("=", 1)[1])
    return 0


# ---- end pb builds ----
# ---- ``bh<m>`` builds: two-term pool / router consumers at H1 (stages/moe_hops_layout.cuh MSO_MK_BH, a bit mask on nb
# bit 1: 1 the branch term published right after MhH1's rank sum and the consumers' combine with inv1 / inv2 from the
# one-exchange slots (alone: a timing build without the residual term, attribution only), 2 the residual term
# published before the hop-1 poll, 4 the pool's lo planes). ``<kind>_icfpoxaft_rt16_ld479_ue10_et5_nb3_bh<m>_s4`` is the
# et5 + nb3 build with bh level m (lean / check / stlean_lite); ``check_nb3_bh<m>`` the plain check build (the 8-rank
# hop tests). ----
BH_LEVELS = (1, 3, 7)
for _bh in BH_LEVELS:
    for _kind in ("lean", "check", "stlean_lite"):
        MODES[f"{_kind}_icfpoxaft_rt16_ld479_ue10_et5_nb3_bh{_bh}_s4"] = tuple(
            MODES[f"{_kind}_icfpoxaft_rt16_ld479_ue10_et5_nb3_s4"]
        ) + (f"-DMSO_MK_BH={_bh}",)
    MODES[f"check_nb3_bh{_bh}"] = tuple(MODES["check_nb3"]) + (f"-DMSO_MK_BH={_bh}",)


def bh_level(mode: str) -> int:
    """A build mode's ``MSO_MK_BH`` mask (0 without it)."""
    for flag in MODES[mode]:
        if flag.startswith("-DMSO_MK_BH="):
            return int(flag.split("=", 1)[1])
    return 0


# Zero-exchange F1 (stages/moe_hops_layout.cuh MSO_MK_FZ = 1, on nb bit 2): MhUnpool publishes the stock thread leaves
# of the F1 row statistics and MhF1 sums them in nb's tree (no row exchange; inv1 / inv2 bitwise nb's).
# ``<kind>_icfpoxaft_rt16_ld479_ue10_et5_nb3[_bh<m>]_fz1_s4`` is the et5 + nb3 [+ bh m] build with it (lean / check /
# stlean_lite); ``check_nb3[_bh<m>]_fz1`` the plain check build (the 8-rank hop tests).
FZ_LEVELS = (1,)
for _fz in FZ_LEVELS:
    for _base in ("", "_bh3"):
        for _kind in ("lean", "check", "stlean_lite"):
            MODES[f"{_kind}_icfpoxaft_rt16_ld479_ue10_et5_nb3{_base}_fz{_fz}_s4"] = tuple(
                MODES[f"{_kind}_icfpoxaft_rt16_ld479_ue10_et5_nb3{_base}_s4"]
            ) + (f"-DMSO_MK_FZ={_fz}",)
        MODES[f"check_nb3{_base}_fz{_fz}"] = tuple(MODES[f"check_nb3{_base}"]) + (
            f"-DMSO_MK_FZ={_fz}",
        )


def fz_level(mode: str) -> int:
    """A build mode's ``MSO_MK_FZ`` (0 without it)."""
    for flag in MODES[mode]:
        if flag.startswith("-DMSO_MK_FZ="):
            return int(flag.split("=", 1)[1])
    return 0


# ---- end bh / fz builds ----
# ---- pb on bh, and the mode composition of the levers. ``<kind>_icfpoxaft_rt16_ld479_ue10_et5_nb3_bh3_pb<p>_s4``
# (lean / check / stlean_lite) is the et5 + nb3 + bh3 build with pb level p; under ``MSO_MK_BH`` the PB bit 2 site is
# bh's pool plane polls (``stages/moe_hops.cuh`` ``bhPollPlane`` / ``pollVecs256Tail``), so pb15 covers the same four
# hand-offs as without bh. ``check_nb3[_bh3]_pb<p>`` are the plain check builds (the 8-rank hop tests). ----
IG_BH_PB_LEVELS = (13, 15)
for _pb in IG_BH_PB_LEVELS:
    for _kind in ("lean", "check", "stlean_lite"):
        MODES[f"{_kind}_icfpoxaft_rt16_ld479_ue10_et5_nb3_bh3_pb{_pb}_s4"] = tuple(
            MODES[f"{_kind}_icfpoxaft_rt16_ld479_ue10_et5_nb3_bh3_s4"]
        ) + (f"-DMSO_MK_PB={_pb}",)
    for _base in ("check_nb3", "check_nb3_bh3"):
        MODES[f"{_base}_pb{_pb}"] = tuple(MODES[_base]) + (f"-DMSO_MK_PB={_pb}",)
# ---- ``ah<m>`` builds: the sliding-window attention front (msoMegakernel.cu MSO_MK_AH, a bit mask: 1 GM, the merging
# SwaAttn units sum the gate from the gate runs' partials and the plan holds no gate QkvFin unit; 2 FF, the q / k / v
# QkvFin units first on early CTAs with a pure-gate run, host plan only; 4 FFP, with 2 only, those CTAs run the fin and
# the run before griddepcontrol.wait; 8 GP (with 4; host plan + FFP's rule for a fin CTA without a QKV run), 16 WD (a
# dry merge pass in the early-x wait), 32 WD-Q (with 16: the dry pass in the Q wait), 128 GS (with 1, without 16: GM's
# gate over warps 4-7 of the merging unit, one gate value per thread), 256 SP (the SWA block pass's key-half staging
# rows padded against bank conflicts); value-identical).
# ``<kind>_icfpoxaft_rt16_ld479_ue10_et5_nb3[_bh3]_pb15_ah<m>_s4`` (lean / check / stlean_lite) is the et5 + nb3 + pb15
# build (with or without bh3) with ah level m; ``LayerKernels`` reads the plan bits from the build
# (``lever_level(mode, "ah")``). ----
AH_LEVELS = (1, 2, 3, 6, 7, 15, 23, 31, 55, 63, 135, 143, 263, 391)
for _ah in AH_LEVELS:
    for _base in (
        "icfpoxaft_rt16_ld479_ue10_et5_nb3_pb15",
        "icfpoxaft_rt16_ld479_ue10_et5_nb3_bh3_pb15",
    ):
        for _kind in ("lean", "check", "stlean_lite"):
            MODES[f"{_kind}_{_base}_ah{_ah}_s4"] = tuple(MODES[f"{_kind}_{_base}_s4"]) + (
                f"-DMSO_MK_AH={_ah}",
            )
# ---- end ah ----
# ---- the stamped builds with sub-stamps and the poll probe (MSO_MK_SY_LITE off; no host-memory progress store) of the
# et5 + nb3 + bh3 + pb15 build and its ah levels, attribution only:
# ``stlean_np_icfpoxaft_rt16_ld479_ue10_et5_nb3_bh3_pb15[_ah<m>]_s4`` is the stlean_lite build of the same name without
# ``-DMSO_MK_SY_LITE=1``. ----
for _ah in (0,) + AH_LEVELS:
    _lite = f"stlean_lite_icfpoxaft_rt16_ld479_ue10_et5_nb3_bh3_pb15{f'_ah{_ah}' if _ah else ''}_s4"
    MODES[_lite.replace("stlean_lite_", "stlean_np_", 1)] = tuple(
        d for d in MODES[_lite] if d != "-DMSO_MK_SY_LITE=1"
    )
# ---- end stlean_np builds ----

#: The levers a build mode composes (``compose_mode``; the numerics knobs: ``serving.LEVER_ENV_VARS``), in the order
#: their tokens take in a mode name (``..._et5_nb3_bh3_fz1_pb15_rp1_rl1_ah7_xp8_km8_s4``; the order the mode names
#: defined in this module use), with the define each token stands for.
LEVERS = ("nb", "bh", "fz", "pb", "rp", "rl", "ah", "xp", "km", "cf", "lm")
LEVER_DEFINES = {
    "nb": "MSO_MK_NB",
    "bh": "MSO_MK_BH",
    "fz": "MSO_MK_FZ",
    "pb": "MSO_MK_PB",
    "rp": "MSO_MK_RP",
    "rl": "MSO_MK_RL",
    "ah": "MSO_MK_AH",
    "xp": "MSO_MK_XP",
    "km": "MSO_MK_KM",
    "cf": "MSO_MK_CF",
    "lm": "MSO_MK_LM",
}
_LEVER_TOKEN = re.compile(r"^(nb|bh|fz|pb|rp|rl|ah|xp|km|cf|lm)([0-9]+)$")


def lever_level(mode: str, lever: str) -> int:
    """A build mode's level of ``lever`` (``LEVERS``) from its defines (0 without the define)."""
    prefix = f"-D{LEVER_DEFINES[lever]}="
    for flag in MODES[mode]:
        if flag.startswith(prefix):
            return int(flag[len(prefix) :])
    return 0


def compose_mode(base: str, levels: Mapping[str, Optional[int]]) -> str:
    """The build mode ``base`` with each lever of ``levels`` at its level.

    ``None`` keeps what ``base`` names, ``0`` removes the lever's token, ``n > 0`` names level n. A lever token of
    ``base`` is replaced (or removed) where it stands; a lever ``base`` does not name goes right after the last named
    lever that precedes it in ``LEVERS``, else right before the first one that follows it, else before a trailing
    ``_s4`` (else at the end). Raises ``ValueError`` when ``base`` or the composed name is not a build mode, or when
    the composed mode's defines do not carry the levels its name states.
    """
    if base not in MODES:
        raise ValueError(f"unknown megakernel build mode {base!r}")
    unknown = sorted(set(levels) - set(LEVERS))
    if unknown:
        raise ValueError(f"unknown megakernel lever(s) {unknown}; the levers are {LEVERS}")
    items: List[tuple] = []  # (lever or None, token)
    named: Dict[str, int] = {}
    for i, token in enumerate(base.split("_")):
        match = _LEVER_TOKEN.match(token) if i > 0 else None
        if match is None:
            items.append((None, token))
            continue
        named[match.group(1)] = int(match.group(2))
        items.append((match.group(1), token))
    want = dict(named)
    for lever, level in levels.items():
        if level is None:
            continue
        if int(level) < 0:
            raise ValueError(f"megakernel lever {lever} level {level}: a mask >= 0")
        if int(level) == 0:
            want.pop(lever, None)
        else:
            want[lever] = int(level)
    out = [
        (lever, token if lever is None else f"{lever}{want[lever]}")
        for lever, token in items
        if lever is None or lever in want
    ]
    for lever in LEVERS:
        if lever not in want or lever in named:
            continue
        order = LEVERS.index(lever)
        present = [k for k, (other, _) in enumerate(out) if other is not None]
        before = [k for k in present if LEVERS.index(out[k][0]) < order]
        after = [k for k in present if LEVERS.index(out[k][0]) > order]
        if before:
            at = before[-1] + 1
        elif after:
            at = after[0]
        else:
            at = len(out) - 1 if len(out) > 1 and out[-1][1] == "s4" else len(out)
        out.insert(at, (lever, f"{lever}{want[lever]}"))
    name = "_".join(token for _, token in out)
    if name not in MODES:
        given = ", ".join(f"{k}={v}" for k, v in sorted(levels.items()) if v is not None)
        raise ValueError(
            f"megakernel build mode {name!r} (from {base!r} with {given}) is not in library.MODES"
        )
    got = {lever: lever_level(name, lever) for lever in LEVERS}
    if got != {lever: want.get(lever, 0) for lever in LEVERS}:
        raise ValueError(
            f"megakernel build mode {name!r}: its defines carry {got}, its name {want}"
        )
    return name


# ---- end lever composition ----
# ---- ``cx<m>`` builds: the attribution-only experts-front oracle (stages/experts_defs.cuh MSO_MK_CX, bit 1: the
# routing state from the layer's oracle slot in the experts descriptor's Dbg word; lean rt16 builds only, never served).
# ``lean_icfpoxaft_rt16_ld479_ue10_et5[_nb3]_cx1_s4`` = the et5 (+ nb3) build with the oracle code. ----
CX_LEVELS = (1,)
for _cx in CX_LEVELS:
    for _base in (
        "lean_icfpoxaft_rt16_ld479_ue10_et5_s4",
        "lean_icfpoxaft_rt16_ld479_ue10_et5_nb3_s4",
    ):
        MODES[_base[: -len("_s4")] + f"_cx{_cx}_s4"] = tuple(MODES[_base]) + (f"-DMSO_MK_CX={_cx}",)


def cx_level(mode: str) -> int:
    """A build mode's ``MSO_MK_CX`` mask (0 without it)."""
    for flag in MODES[mode]:
        if flag.startswith("-DMSO_MK_CX="):
            return int(flag.split("=", 1)[1])
    return 0


#: The oracle slot (``kCxSlotBytes``): u32 want, have, level, early at 0, 4, 8, 12; E at 16; the image at 64.
CX_SLOT_BYTES = 2048
# ---- end cx ----
# ---- ``rp<m>`` builds: the routing state published once per rank (stages/experts_defs.cuh MSO_MK_RP, bit 1: CTA
# kCtas - 1 routes and places, every other CTA polls its tables; the same values).
# ``<kind>_icfpoxaft_rt16_ld479_ue10_et5[_nb3]_rp1_s4`` = the et5 (+ nb3) build with the publisher;
# ``lean_…_et5_nb3_cx1_rp1_s4`` its attribution form whose publisher takes the cx oracle slot (it publishes at stage
# entry). ----
RP_LEVELS = (1,)
for _rp in RP_LEVELS:
    for _kind in ("lean", "check"):
        for _base in (
            f"{_kind}_icfpoxaft_rt16_ld479_ue10_et5_s4",
            f"{_kind}_icfpoxaft_rt16_ld479_ue10_et5_nb3_s4",
        ):
            MODES[_base[: -len("_s4")] + f"_rp{_rp}_s4"] = tuple(MODES[_base]) + (
                f"-DMSO_MK_RP={_rp}",
            )
    for _cx in CX_LEVELS:
        _base = f"lean_icfpoxaft_rt16_ld479_ue10_et5_nb3_cx{_cx}_s4"
        MODES[_base[: -len("_s4")] + f"_rp{_rp}_s4"] = tuple(MODES[_base]) + (f"-DMSO_MK_RP={_rp}",)


def rp_level(mode: str) -> int:
    """A build mode's ``MSO_MK_RP`` mask (0 without it)."""
    for flag in MODES[mode]:
        if flag.startswith("-DMSO_MK_RP="):
            return int(flag.split("=", 1)[1])
    return 0


# ---- end rp ----
# ---- ``rl<m>`` builds: the router logits by linearity (msoMegakernel.cu MSO_MK_RL: 1 the build, 2 the record, 8 the
# logits formed by rp's publisher instead of MhH1; rl.py). The build is not bitwise the stock layers (the routing input
# differs from stock's; numerics contract in msoMegakernel.cu); switched on per layer by rl.IFLAG_RL_ON (serving:
# MUSESPARK_OPEN_MEGAKERNEL_RL). On the et5 + nb3 rp builds: ``lean_…_et5_nb3_rp1_rl1_s4`` (MhH1 forms the logits),
# ``lean_…_et5_nb3_rp1_rl9_s4`` (the publisher forms them), and their check twins with the record (rl3 / rl11). ----
RL_LEVELS = (1, 9)
for _rl in RL_LEVELS:
    for _kind, _rec in (("lean", 0), ("check", 2)):
        _base = f"{_kind}_icfpoxaft_rt16_ld479_ue10_et5_nb3_rp1_s4"
        MODES[_base[: -len("_s4")] + f"_rl{_rl | _rec}_s4"] = tuple(MODES[_base]) + (
            f"-DMSO_MK_RL={_rl | _rec}",
        )


def rl_level(mode: str) -> int:
    """A build mode's ``MSO_MK_RL`` mask (0 without it)."""
    for flag in MODES[mode]:
        if flag.startswith("-DMSO_MK_RL="):
            return int(flag.split("=", 1)[1])
    return 0


# ---- end rl ----
# ---- rp and rl on the pb15 / bh3 builds, and bh's pool-only level.
# ``<kind>_icfpoxaft_rt16_ld479_ue10_et5_nb3<b>_rp1_s4`` and ``..._rp1_rl<r>_s4`` (lean: rl 1; check: rl 3, with the
# record) for ``<b>`` in ``IG2_BASES`` (et5 + nb3 + pb15, + bh3, + bh3 + pb15; et5 + nb3 itself is defined above).
# bh 11 = bh 3 + bit 8 (``stages/moe_hops_layout.cuh``: pool-only two-term consumers, no router code; an rl build
# served with rl on every layer): ``..._bh11_pb15_rp1_rl<r>_s4``. Composition at H1 on an rl layer: MhH1 publishes bh's
# terms (pool) and forms the logits from rl's partials (rlFinalize, after every bh publication); the router slot runs
# rl's units, so bh's two-term router never runs there. ----
IG2_BASES = ("_pb15", "_bh3", "_bh3_pb15")
for _b in IG2_BASES:
    for _kind, _rl in (("lean", 1), ("check", 3)):
        _base = f"{_kind}_icfpoxaft_rt16_ld479_ue10_et5_nb3{_b}_s4"
        _rp = _base[: -len("_s4")] + "_rp1_s4"
        MODES[_rp] = tuple(MODES[_base]) + ("-DMSO_MK_RP=1",)
        MODES[_rp[: -len("_s4")] + f"_rl{_rl}_s4"] = tuple(MODES[_rp]) + (f"-DMSO_MK_RL={_rl}",)
for _kind, _rl in (("lean", 1), ("check", 3)):
    _src = f"{_kind}_icfpoxaft_rt16_ld479_ue10_et5_nb3_bh3_pb15_rp1_rl{_rl}_s4"
    MODES[_src.replace("_bh3_", "_bh11_")] = tuple(
        "-DMSO_MK_BH=11" if d == "-DMSO_MK_BH=3" else d for d in MODES[_src]
    )
# rl bit 16 (msoMegakernel.cu MSO_MK_RL: router-free, no router code; served with rl on every layer):
# ``<kind>_icfpoxaft_rt16_ld479_ue10_et5_nb3_pb15_rp1_rl<r | 16>_s4`` (lean rl 17, check rl 19).
for _kind, _rl in (("lean", 1), ("check", 3)):
    _src = f"{_kind}_icfpoxaft_rt16_ld479_ue10_et5_nb3_pb15_rp1_rl{_rl}_s4"
    MODES[_src.replace(f"_rl{_rl}_", f"_rl{_rl | 16}_")] = tuple(
        f"-DMSO_MK_RL={_rl | 16}" if d == f"-DMSO_MK_RL={_rl}" else d for d in MODES[_src]
    )
# ---- end rp / rl builds ----
# ---- ah on the rl1 build (et5 + nb3 + bh11 + pb15 + rp1 + rl1). ``stlean_lite_…_bh11_pb15_rp1_rl1_s4`` is the rl1
# build's per-unit-only stamped build (attribution only), and ``<kind>_…_bh11_pb15_rp1_rl<r>_ah<m>_s4`` (lean /
# stlean_lite: rl 1; check: rl 3, with the record) the rl1 build with ah level m. Composition: ah changes only the
# sliding-window layers' attention half (GM in ``ahMerge``, FF's plan, FFP's pre-wait roles); rl's units (the router
# slots), rp's publisher (CTA kCtas - 1) and bh11's pool consumers are MoE-half units, which run after the attention
# half on every CTA and after the dependency wait, so their inputs, writers and resets are the rl1 build's
# (test_megakernel_layer_plan.py). ----
IG3_AH_LEVELS = (7,)
for _kind, _src in (
    ("lean", "lean_icfpoxaft_rt16_ld479_ue10_et5_nb3_bh11_pb15_rp1_rl1_s4"),
    ("check", "check_icfpoxaft_rt16_ld479_ue10_et5_nb3_bh11_pb15_rp1_rl3_s4"),
):
    for _ah in IG3_AH_LEVELS:
        MODES[_src[: -len("_s4")] + f"_ah{_ah}_s4"] = tuple(MODES[_src]) + (f"-DMSO_MK_AH={_ah}",)
_IG3_LITE_H = "stlean_lite_icfpoxaft_rt16_ld479_ue10_et5_nb3_bh11_pb15_rp1_rl1_s4"
MODES[_IG3_LITE_H] = tuple(
    "-DMSO_MK_BH=11" if d == "-DMSO_MK_BH=3" else d
    for d in MODES["stlean_lite_icfpoxaft_rt16_ld479_ue10_et5_nb3_bh3_pb15_s4"]
) + ("-DMSO_MK_RP=1", "-DMSO_MK_RL=1")
for _ah in IG3_AH_LEVELS:
    MODES[_IG3_LITE_H[: -len("_s4")] + f"_ah{_ah}_s4"] = tuple(MODES[_IG3_LITE_H]) + (
        f"-DMSO_MK_AH={_ah}",
    )
# ---- end ah on rl1 ----
# ---- ``xp<m>`` builds: the experts front's X path (stages/experts_defs.cuh MSO_MK_XP, a bit mask: 1 row owners (8 CTAs
# quantize whole pooled rows), 2 two owners per row, 4 the state first on the CTAs without a group (needs rp), 8 the rp
# publisher's warm-up pass of route() / place() before its logit poll, 32 the cut table: the stock queue cut of every
# slot count tabled at the stage entry, 64 the early requests' tile 0 from that table (needs 32), 128 the hop-1 slots
# emptied by the next launch's MhClear instead of MhH1, 256 the same inside MhClear's one byte range, 512 row-aligned
# quantization shares, 1024 no share on the MhPool CTAs (needs 512)). Value-identical.
# On the rl1 build (``XP_BASE``): ``<kind>_<XP_BASE>[_xp<m>]_s4`` (lean rl 1 / check rl 3 with the record); the
# stamped twins (per-CTA timelines with sub-stamps; attribution only, never timed) ``stlean_<XP_BASE>[_xp<m>]_s4``
# (%globaltimer sub-stamps) and ``stlean_clk_<XP_BASE>[_xp<m>]_s4`` (MSO_MK_SY_CLK = 1: clock64 sub-stamps) are the
# stamped et5 build plus the same defines, in the same order, as the lean mode of the same name. ----
# 40 and up: with bits 32-1024
XP_LEVELS = (
    1,
    2,
    4,
    5,
    6,
    8,
    9,
    10,
    13,
    40,
    104,
    136,
    232,
    264,
    296,
    360,
    520,
    552,
    616,
    808,
    1544,
    1576,
    1640,
    1832,
    1896,
)
XP_BASE = "icfpoxaft_rt16_ld479_ue10_et5_nb3_bh11_pb15_rp1_rl1"
for _xp in XP_LEVELS:
    for _kind, _rl in (("lean", 1), ("check", 3)):
        _src = f"{_kind}_{XP_BASE[: -len('_rl1')]}_rl{_rl}_s4"
        MODES[_src[: -len("_s4")] + f"_xp{_xp}_s4"] = tuple(MODES[_src]) + (f"-DMSO_MK_XP={_xp}",)
_lean_et5 = tuple(MODES["lean_icfpoxaft_rt16_ld479_ue10_et5_s4"])
for _name in [XP_BASE] + [f"{XP_BASE}_xp{_xp}" for _xp in XP_LEVELS]:
    assert tuple(MODES[f"lean_{_name}_s4"])[: len(_lean_et5)] == _lean_et5, _name
    _extra = tuple(MODES[f"lean_{_name}_s4"])[len(_lean_et5) :]
    MODES[f"stlean_{_name}_s4"] = tuple(MODES["stlean_icfpoxaft_rt16_ld479_ue10_et5_s4"]) + _extra
    MODES[f"stlean_clk_{_name}_s4"] = tuple(MODES[f"stlean_{_name}_s4"]) + ("-DMSO_MK_SY_CLK=1",)


def xp_level(mode: str) -> int:
    """A build mode's ``MSO_MK_XP`` mask (0 without it)."""
    return lever_level(mode, "xp")


# ---- end xp ----
# ---- xp and the further ah levels on the ah7 build (rl1 + ah7). ``<kind>_…_bh11_pb15_rp1_rl<r>_ah<m>[_xp<x>]_s4``
# (lean / stlean_lite: rl 1; check: rl 3, with the record) is the rl1 build with ah level m in ``IG4_AH_LEVELS`` (7;
# 135 = 7 + GS; 391 = 7 + GS + SP) and xp level x in ``IG4_XP_LEVELS`` (8: rp's publisher warm pass); the defines are
# the rl1 build's, then ``-DMSO_MK_AH``, then ``-DMSO_MK_XP``, the order of the name's tokens (``LEVERS``). Composition:
# xp 8 adds code only to the experts stage of the publisher CTA (kCtas - 1; shared memory only, no global word, no
# wait), which hosts no ah unit change (its role list is the rl1 build's under every ah level) and no rl unit; GS / SP
# change only the merging AhSwaAttn unit's merge (``ahMerge``) and the block pass's staging (``stageAhSwaAttn``) of the
# sliding-window layers, and the plans of ah 135 / 391 are ah 7's (test_megakernel_layer_plan.py). ----
IG4_AH_LEVELS = (7, 135, 391)
IG4_XP_LEVELS = (8,)
for _kind, _src in (
    ("lean", "lean_icfpoxaft_rt16_ld479_ue10_et5_nb3_bh11_pb15_rp1_rl1_s4"),
    ("check", "check_icfpoxaft_rt16_ld479_ue10_et5_nb3_bh11_pb15_rp1_rl3_s4"),
    ("stlean_lite", "stlean_lite_icfpoxaft_rt16_ld479_ue10_et5_nb3_bh11_pb15_rp1_rl1_s4"),
):
    for _ah in IG4_AH_LEVELS:
        _ahm = _src[: -len("_s4")] + f"_ah{_ah}_s4"
        if _ahm not in MODES:  # ah 7 is defined above
            MODES[_ahm] = tuple(MODES[_src]) + (f"-DMSO_MK_AH={_ah}",)
        for _xp in IG4_XP_LEVELS:
            MODES[_ahm[: -len("_s4")] + f"_xp{_xp}_s4"] = tuple(MODES[_ahm]) + (
                f"-DMSO_MK_XP={_xp}",
            )
    if (
        _kind == "stlean_lite"
    ):  # rl1 + xp (no ah) in the lite form too, for the xp-alone attribution twin
        for _xp in IG4_XP_LEVELS:
            MODES[_src[: -len("_s4")] + f"_xp{_xp}_s4"] = tuple(MODES[_src]) + (
                f"-DMSO_MK_XP={_xp}",
            )
# ---- end xp / ah on ah7 ----
# ---- ``ly<m>`` builds: layout / no-op twins of the rl1 and ah7 builds (msoMegakernel.cu MSO_MK_LY, attribution only,
# never served): bit 1 / 2 move ahMerge / stageAhQkvFin to the end of the kernel's text section (the same code), bit 4
# compiles ah FFP's kernel-body block into the rl1 build without FF's plan (inert).
# ``lean_…_bh11_pb15_rp1_rl1[_ah7]_ly<m>_s4``. ----
AH3_LY_LEVELS = {
    "lean_icfpoxaft_rt16_ld479_ue10_et5_nb3_bh11_pb15_rp1_rl1_s4": (3, 4),
    "lean_icfpoxaft_rt16_ld479_ue10_et5_nb3_bh11_pb15_rp1_rl1_ah7_s4": (3,),
}
for _src, _lys in AH3_LY_LEVELS.items():
    for _ly in _lys:
        MODES[_src[: -len("_s4")] + f"_ly{_ly}_s4"] = tuple(MODES[_src]) + (f"-DMSO_MK_LY={_ly}",)
# ME (MSO_MK_AH bit 512, the merge early): the rl1 build's lean / check / stlean_lite builds at ah7 + GS (135), + SP
# (391), + ME (647) and + SP + ME (903): ``<kind>_…_bh11_pb15_rp1_rl<r>_ah<m>_s4``.
AH3_AH_LEVELS = (135, 391, 647, 903, 1159)  # 1159: ah7 + GS + MX
for _src in (
    "lean_icfpoxaft_rt16_ld479_ue10_et5_nb3_bh11_pb15_rp1_rl1_s4",
    "check_icfpoxaft_rt16_ld479_ue10_et5_nb3_bh11_pb15_rp1_rl3_s4",
    _IG3_LITE_H,
):
    for _ah in AH3_AH_LEVELS:
        MODES[_src[: -len("_s4")] + f"_ah{_ah}_s4"] = tuple(MODES[_src]) + (f"-DMSO_MK_AH={_ah}",)
# ME's attribution twin (MSO_MK_LY 8: the merge entered after the publication), lean and lite:
# ``<kind>_…_bh11_pb15_rp1_rl1_ah647_ly8_s4``.
AH3_LY_LEVELS["lean_icfpoxaft_rt16_ld479_ue10_et5_nb3_bh11_pb15_rp1_rl1_ah647_s4"] = (8,)
AH3_LY_LEVELS[_IG3_LITE_H[: -len("_s4")] + "_ah647_s4"] = (8,)
for _src in list(AH3_LY_LEVELS)[-2:]:
    MODES[_src[: -len("_s4")] + "_ly8_s4"] = tuple(MODES[_src]) + ("-DMSO_MK_LY=8",)
# ---- end ly / ah builds ----
# ---- ``km<k>`` builds: FaAttn's MMA issue (stages/fullAttentionConfig.cuh MSO_MK_KM, a bit mask: 1 PV of a full tile
# and its two commits as one statement, 2 QK and its commit likewise, 4 QK(t + 2) issued after PV(t) completed, 8 / 16
# the PV / QK statements with one code copy for every slot (and 16 a one-compare patch check), 32 with 8 / 16 their
# operands formed right before each MMA, 64 with 8 one PV statement for every tile (a tail tile's MMAs predicated);
# value-identical).
# ``<kind>_km<k>`` (lean / check / stlean) are the FaAttn stage test's builds with km level k; ``stlean[_km<k>]_kmtl``
# their attribution timeline builds (MSO_MK_KM_TL, stage test ``test_km_timeline``). ----
KM_LEVELS = (1, 2, 3, 4, 5, 8, 16, 24, 56, 72, 88, 120)
for _km in KM_LEVELS:
    for _kind in ("lean", "check", "stlean"):
        MODES[f"{_kind}_km{_km}"] = tuple(MODES[_kind]) + (f"-DMSO_MK_KM={_km}",)
MODES["stlean_kmtl"] = tuple(MODES["stlean"]) + ("-DMSO_MK_KM_TL=1",)
for _km in KM_LEVELS:
    MODES[f"stlean_km{_km}_kmtl"] = tuple(MODES[f"stlean_km{_km}"]) + ("-DMSO_MK_KM_TL=1",)
# Attribution only (timing, wrong results): the per-MMA timeline (TL 2), no S load (MSO_MK_FA_DBG 32), no P store (64).
MODES["stlean_kmtl2"] = tuple(MODES["stlean"]) + ("-DMSO_MK_KM_TL=2",)
for _dbg, _name in (
    (32, "nold"),
    (64, "nost"),
    (96, "noldst"),
    (128, "norf"),
    (384, "norfw"),
    (512, "nofence"),
):
    MODES[f"lean_{_name}"] = tuple(MODES["lean"]) + (f"-DMSO_MK_FA_DBG={_dbg}",)
    MODES[f"stlean_kmtl_{_name}"] = tuple(MODES["stlean"]) + (
        "-DMSO_MK_KM_TL=1",
        f"-DMSO_MK_FA_DBG={_dbg}",
    )
    MODES[f"stlean_kmtl2_{_name}"] = tuple(MODES["stlean"]) + (
        "-DMSO_MK_KM_TL=2",
        f"-DMSO_MK_FA_DBG={_dbg}",
    )
# km on the ah7 build (rl1 + ah7): ``lean_…_bh11_pb15_rp1_rl1_ah7_km<k>_s4``, its check twin
# ``check_…_rl3_ah7_km<k>_s4`` and the per-unit stamped ``stlean_lite_…_rl1_ah7_km<k>_s4``.
# km changes only stageFaAttn's issuer path (the full-attention layers' FaAttn units); every other unit, plan and flag
# is the ah7 build's.
KM_I_LEVELS = (8, 16, 24, 56, 72, 88, 120)
for _km in KM_I_LEVELS:
    for _src in (
        "lean_icfpoxaft_rt16_ld479_ue10_et5_nb3_bh11_pb15_rp1_rl1_ah7_s4",
        "check_icfpoxaft_rt16_ld479_ue10_et5_nb3_bh11_pb15_rp1_rl3_ah7_s4",
        "stlean_lite_icfpoxaft_rt16_ld479_ue10_et5_nb3_bh11_pb15_rp1_rl1_ah7_s4",
    ):
        MODES[_src[: -len("_s4")] + f"_km{_km}_s4"] = tuple(MODES[_src]) + (f"-DMSO_MK_KM={_km}",)
# ---- end km ----
# ---- ME, the cut table and km 8 on the ah391_xp8 build (rl1 + ah391 (= ah7 + GS + SP) + xp8).
# ``<kind>_…_bh11_pb15_rp1_rl<r>_ah<m>_xp<x>[_km<k>]_s4`` (lean / stlean_lite: rl 1; check: rl 3, with the record) is
# the rl1 build with ah level m in ``IG5_AH_LEVELS`` (391; 903 = 391 + ME, MSO_MK_AH bit 512), xp level x in
# ``IG5_XP_LEVELS`` (8; 40 = 8 + the cut table, MSO_MK_XP bit 32) and km level k in ``IG5_KM_LEVELS`` (8: FaAttn's PV as
# one statement); the defines are the rl1 build's, then ``-DMSO_MK_AH``, ``-DMSO_MK_XP``, ``-DMSO_MK_KM``, the order of
# the name's tokens (``LEVERS``). ``…_ah391_xp8_s4`` is the mode defined above, unchanged. Composition
# (test_megakernel_layer_plan.py): the three bits are kernel-only (the ah391_xp8 build's host tables); ME is code of the
# sliding-window layers' merging AhSwaAttn units (half 1's merge on bar 1 in the reduction area while half 0 combines
# from the attention scratch), the cut table of the experts stage's entry (the reduction area's free end, before the
# entry barrier; xp8's warm pass runs after it), km 8 of stageFaAttn's issuer (full-attention layers only), so no unit
# runs two of them and every unit boundary is a CTA barrier. ----
IG5_AH_LEVELS = (391, 903)
IG5_XP_LEVELS = (8, 40)
IG5_KM_LEVELS = (8,)
for _kind, _src in (
    ("lean", "lean_icfpoxaft_rt16_ld479_ue10_et5_nb3_bh11_pb15_rp1_rl1_s4"),
    ("check", "check_icfpoxaft_rt16_ld479_ue10_et5_nb3_bh11_pb15_rp1_rl3_s4"),
    ("stlean_lite", "stlean_lite_icfpoxaft_rt16_ld479_ue10_et5_nb3_bh11_pb15_rp1_rl1_s4"),
):
    for _ah in IG5_AH_LEVELS:
        _ahm = _src[: -len("_s4")] + f"_ah{_ah}_s4"  # both defined above (391 and 903)
        for _xp in IG5_XP_LEVELS:
            _axm = _ahm[: -len("_s4")] + f"_xp{_xp}_s4"
            if _axm not in MODES:  # ah 391 + xp 8 is defined above
                MODES[_axm] = tuple(MODES[_ahm]) + (f"-DMSO_MK_XP={_xp}",)
            for _km in IG5_KM_LEVELS:
                MODES[_axm[: -len("_s4")] + f"_km{_km}_s4"] = tuple(MODES[_axm]) + (
                    f"-DMSO_MK_KM={_km}",
                )
# ---- end ME / cut table / km 8 builds ----
# ---- ``cf<m>`` builds: the h loader's code (stages/experts_defs.cuh MSO_MK_CF, a bit mask: 1 the four in-flight
# k-blocks stored and released by one rolled body, 2 the workspace clear rolled, 4 the px slot scan in closed form) on
# the km8 build (``…_ah903_xp40_km8``). ``<kind>_…_bh11_pb15_rp1_rl<r>_ah903_xp40_km8_cf<m>_s4`` (lean / stlean_lite:
# rl 1; check: rl 3) is the km8 build with cf level m in ``CF_LEVELS``; the defines are the km8 build's, then
# ``-DMSO_MK_CF``. cf is kernel-only (loaderLoop, warps 2-3 of the experts stage): no host table changes. ----
CF_LEVELS = (1, 2, 3, 4, 5, 7)
for _kind, _src in (
    ("lean", "lean_icfpoxaft_rt16_ld479_ue10_et5_nb3_bh11_pb15_rp1_rl1_ah903_xp40_km8_s4"),
    ("check", "check_icfpoxaft_rt16_ld479_ue10_et5_nb3_bh11_pb15_rp1_rl3_ah903_xp40_km8_s4"),
    (
        "stlean_lite",
        "stlean_lite_icfpoxaft_rt16_ld479_ue10_et5_nb3_bh11_pb15_rp1_rl1_ah903_xp40_km8_s4",
    ),
):
    for _cf in CF_LEVELS:
        MODES[_src[: -len("_s4")] + f"_cf{_cf}_s4"] = tuple(MODES[_src]) + (f"-DMSO_MK_CF={_cf}",)
# ---- end cf builds ----
# ---- ``lyp<n>`` / ``lyx<n>`` builds: layout-only twins of the ah391_xp8 build (attribution only, never served, no
# knob). ``<kind>_…_rl<r>_ah391_xp8_lyp<n>_s4`` (lean rl 1 / check rl 3) = the ah391_xp8 build's defines +
# ``-DMSO_MK_LY_PAD=<n>`` (stages/moe_hops.cuh: n NANOSLEEPs at the head of the router-free stageMhRouter, which no unit
# of these builds runs, so every callee laid out after it moves by 16 n bytes); ``..._lyx<n>_s4`` = the same defines +
# ``-DMSO_MK_LY_PADX=<n>`` (the same pad in p4a::lyPadEpilog, between finalUnitPx and epilogueLoop: the experts'
# epilogue / producer / MMA loops move against the loader loop). No executed instruction changes in either. ----
LY_PAD_LEVELS = (4, 256)
LY_PADX_LEVELS = (128, 256)
for _kind, _src in (
    ("lean", "lean_icfpoxaft_rt16_ld479_ue10_et5_nb3_bh11_pb15_rp1_rl1_ah391_xp8_s4"),
    ("check", "check_icfpoxaft_rt16_ld479_ue10_et5_nb3_bh11_pb15_rp1_rl3_ah391_xp8_s4"),
):
    for _p in LY_PAD_LEVELS:
        MODES[_src[: -len("_s4")] + f"_lyp{_p}_s4"] = tuple(MODES[_src]) + (
            f"-DMSO_MK_LY_PAD={_p}",
        )
    for _p in LY_PADX_LEVELS:
        MODES[_src[: -len("_s4")] + f"_lyx{_p}_s4"] = tuple(MODES[_src]) + (
            f"-DMSO_MK_LY_PADX={_p}",
        )
# ---- end layout pads ----
# ---- the same layout-only pads on the km8 build and on km8 + cf 7 / cf 3 (attribution only, never served, no knob):
# ``<kind>_…_rl<r>_ah903_xp40_km8[_cf7|_cf3]_lyp<n>_s4`` / ``…_lyx<n>_s4`` = the base build's defines + the pad define
# (the km8 build has bh 11, so the router-free stageMhRouter the pads need). ----
for _kind, _src in (
    ("lean", "lean_icfpoxaft_rt16_ld479_ue10_et5_nb3_bh11_pb15_rp1_rl1_ah903_xp40_km8_s4"),
    ("check", "check_icfpoxaft_rt16_ld479_ue10_et5_nb3_bh11_pb15_rp1_rl3_ah903_xp40_km8_s4"),
    ("lean", "lean_icfpoxaft_rt16_ld479_ue10_et5_nb3_bh11_pb15_rp1_rl1_ah903_xp40_km8_cf7_s4"),
    ("check", "check_icfpoxaft_rt16_ld479_ue10_et5_nb3_bh11_pb15_rp1_rl3_ah903_xp40_km8_cf7_s4"),
    ("lean", "lean_icfpoxaft_rt16_ld479_ue10_et5_nb3_bh11_pb15_rp1_rl1_ah903_xp40_km8_cf3_s4"),
    ("check", "check_icfpoxaft_rt16_ld479_ue10_et5_nb3_bh11_pb15_rp1_rl3_ah903_xp40_km8_cf3_s4"),
):
    for _p in LY_PAD_LEVELS:
        MODES[_src[: -len("_s4")] + f"_lyp{_p}_s4"] = tuple(MODES[_src]) + (
            f"-DMSO_MK_LY_PAD={_p}",
        )
    for _p in LY_PADX_LEVELS:
        MODES[_src[: -len("_s4")] + f"_lyx{_p}_s4"] = tuple(MODES[_src]) + (
            f"-DMSO_MK_LY_PADX={_p}",
        )
# ---- end km8 layout pads ----
# ---- the FaAttn loop record (attribution only, stages/fullAttentionConfig.cuh MSO_MK_FA_ST) on the km8 build:
# ``stlean_lite_fs_…_rl1_ah903_xp40_km8_s4`` is the km8 build's per-unit stamped build plus ``-DMSO_MK_FA_ST=1``; the
# FaAttn stage test's ``stlean_lite_fs[_km<k>]`` (lean code, per-unit stamps, the record). ----
KM2_K_LITE = "stlean_lite_icfpoxaft_rt16_ld479_ue10_et5_nb3_bh11_pb15_rp1_rl1_ah903_xp40_km8_s4"
MODES[KM2_K_LITE.replace("stlean_lite_", "stlean_lite_fs_", 1)] = tuple(MODES[KM2_K_LITE]) + (
    "-DMSO_MK_FA_ST=1",
)
MODES["stlean_lite_fs"] = tuple(MODES["stlean"]) + (
    "-DMSO_MK_SY_PROG=0",
    "-DMSO_MK_SY_LITE=1",
    "-DMSO_MK_FA_ST=1",
)
for _km in KM_LEVELS:
    MODES[f"stlean_lite_fs_km{_km}"] = tuple(MODES["stlean_lite_fs"]) + (f"-DMSO_MK_KM={_km}",)
# ---- end FaAttn loop record builds ----
# ---- ``lm<m>`` builds: the full layer's merge hand-off (stages/fullAttentionConfig.cuh MSO_MK_LM, a bit mask: 1
# FaMerge's row is the CTA's arrival ticket, with ``layer_mk.lm_ticket_plan``'s candidates on every CTA; 2 the gate's
# run slots polled by the eight warps with the partials and summed by warp 0) on the km8 build.
# ``<kind>_…_rl<r>_ah903_xp40_km8_lm<m>_s4`` (lean / stlean_lite: rl 1; check: rl 3) is the km8 build with lm level m
# in ``LM_LEVELS`` (the lean lm 2 one is ``serving.DEFAULT_MODE``); the defines are the km8 build's, then
# ``-DMSO_MK_LM``. ``stlean_lite_fs_…_km8_lm<m>_s4`` adds the FaAttn / FaMerge loop record (attribution only), and
# ``<kind>_…_km8_lm<m>_lyp<n>_s4`` / ``…_lyx<n>_s4`` (lean / check) the layout-only pads (attribution only). ----
LM_LEVELS = (1, 2, 3)
LM_K = "icfpoxaft_rt16_ld479_ue10_et5_nb3_bh11_pb15_rp1_rl{rl}_ah903_xp40_km8"
for _lm in LM_LEVELS:
    for _kind, _rl in (("lean", 1), ("check", 3), ("stlean_lite", 1)):
        _src = f"{_kind}_{LM_K.format(rl=_rl)}_s4"
        _dst = _src[: -len("_s4")] + f"_lm{_lm}_s4"
        MODES[_dst] = tuple(MODES[_src]) + (f"-DMSO_MK_LM={_lm}",)
        if _kind == "stlean_lite":
            MODES[_dst.replace("stlean_lite_", "stlean_lite_fs_", 1)] = tuple(MODES[_dst]) + (
                "-DMSO_MK_FA_ST=1",
            )
        if _kind in ("lean", "check"):
            for _p in LY_PAD_LEVELS:
                MODES[_dst[: -len("_s4")] + f"_lyp{_p}_s4"] = tuple(MODES[_dst]) + (
                    f"-DMSO_MK_LY_PAD={_p}",
                )
            for _p in LY_PADX_LEVELS:
                MODES[_dst[: -len("_s4")] + f"_lyx{_p}_s4"] = tuple(MODES[_dst]) + (
                    f"-DMSO_MK_LY_PADX={_p}",
                )


# The km8 build and km8 + lm<m> with km level 88 (FaAttn's PV and QK statements with one code copy, a tail tile's PV
# predicated) in place of km 8: ``<kind>_…_ah903_xp40_km88[_lm<m>]_s4``.
LM_KM88_LM_LEVELS = (0, 3)
for _lm in LM_KM88_LM_LEVELS:
    for _kind, _rl in (("lean", 1), ("check", 3), ("stlean_lite", 1)):
        _src = f"{_kind}_{LM_K.format(rl=_rl)}" + (f"_lm{_lm}" if _lm else "") + "_s4"
        _dst = _src.replace("_km8_", "_km88_", 1) if _lm else _src.replace("_km8_s4", "_km88_s4")
        assert "-DMSO_MK_KM=8" in MODES[_src], _src
        MODES[_dst] = tuple("-DMSO_MK_KM=88" if f == "-DMSO_MK_KM=8" else f for f in MODES[_src])
        if _kind == "stlean_lite":
            MODES[_dst.replace("stlean_lite_", "stlean_lite_fs_", 1)] = tuple(MODES[_dst]) + (
                "-DMSO_MK_FA_ST=1",
            )
        if _kind == "lean":
            for _p in LY_PAD_LEVELS:
                MODES[_dst[: -len("_s4")] + f"_lyp{_p}_s4"] = tuple(MODES[_dst]) + (
                    f"-DMSO_MK_LY_PAD={_p}",
                )
            for _p in LY_PADX_LEVELS:
                MODES[_dst[: -len("_s4")] + f"_lyx{_p}_s4"] = tuple(MODES[_dst]) + (
                    f"-DMSO_MK_LY_PADX={_p}",
                )


def lm_level(mode: str) -> int:
    """A build mode's ``MSO_MK_LM`` mask (0 without it)."""
    return lever_level(mode, "lm")


# ---- end lm ----


def ft_gate(mode: str) -> bool:
    """Whether a build mode's FaMerge sums the gate itself (``MSO_MK_FT_GATE=1``), so its full-layer plans must not
    hold the gate QkvSum units."""
    return "-DMSO_MK_FT_GATE=1" in MODES[mode]


def fixed_moe_flags(mode: str) -> Optional[int]:
    """The experts flags word a build mode fixes at compile time (``MSO_MK_IC_MOE_FIX`` with mask -1), else None."""
    defs = MODES[mode]
    if "-DMSO_MK_IC_MOE_FIX=1" not in defs or "-DMSO_MK_IC_MOE_MASK=-1ll" not in defs:
        return None
    (val,) = [d for d in defs if d.startswith("-DMSO_MK_IC_MOE_VAL=")]
    return int(val.split("=", 1)[1].rstrip("l"), 0)


def fixed_fa_flags(mode: str) -> Optional[tuple]:
    """The (zero, one) ``kPIFlags`` masks a build mode fixes in the full-attention stages (``MSO_MK_IC_FA_*``),
    else None."""
    vals = {}
    for d in MODES[mode]:
        for key in ("ZERO", "ONE"):
            if d.startswith(f"-DMSO_MK_IC_FA_{key}="):
                vals[key] = int(d.split("=", 1)[1].rstrip("l"), 0)
    return (vals.get("ZERO", 0), vals.get("ONE", 0)) if vals else None


# ---- end compile-time-fixed flag builds ----

#: Parameter block fields, in the order of ``mso_mk::Param``.
PARAM_FIELDS = (
    "Magic",
    "NumCtas",
    "MaxRoles",
    "SmemBytes",
    "Copy",
    "NumCopies",
    "Counters",
    "Arena",
    "Err",
    "TimeoutNs",
    "Stamps",
    "Flags",
    "WQkv",
    "GammaQkv",
    "QkvPart",
    "Eps",
    "TestUnits",
    "TestBlob",
    "TestMail",
    "TestPubs",
    # ---- the attention half (stages/attnHalf.cuh) ----
    "AhWOproj",
    "AhSlot",
    "AhPool",
    "AhBlockTable",
    "AhBtStride",
    "AhSeqLen",
    "AhPositions",
    "AhCosSin",
    "AhKvScale",
    "AhQScale",
    "AhPrepEps",
    "AhGateEps",
    "AhPart",
    "AhWindow",
    "AhQFp8",
    "AhOReplicas",
    "AhAttnOut",
    "AhFlags",
    # ---- end attention half ----
    # ---- full attention (stages/fullAttention.cuh) ----
    "FaTmap",
    "FaPool",
    "FaTable",
    "FaTableStride",
    "FaKvLens",
    "FaKvScale",
    "FaPartO",
    "FaPartL",
    "FaUnits",
    "FaMinPages",
    "FaEntryPages",
    "FaQScale",
    "FaEps",
    "FaOut",
    "FaOutF32",
    "FaPrefetch",
    "FaDbgO",
    "FaDbgL",
    # ---- end full attention ----
    # ---- MoE experts (stages/experts.cuh) ----
    "Moe",
    "MoePad",
    # ---- end MoE experts ----
    # ---- MoE-side hops (stages/moe_hops.cuh) ----
    "MhUc",
    "MhMc",
    "MhArena",
    "MhRank",
    "MhWRouter",
    "MhWPool",
    "MhWUnpool",
    "MhCoefH1",
    "MhCoefF1",
    "MhPeers",
    "MhScale",
    "MhCapture",
    "MhCaptureStride",
    "MhInOproj",
    "MhInExpert",
    "MhOutRes1",
    "MhOutRes2",
    "MhOutXffn",
    "MhOutPool",
    "MhOutLogits",
    "MhOutM",
    "MhOutB",
    "MhOutXattn",
    "Rl",  # rl (MSO_MK_RL builds): kPRl, the layer's rl block (rl.py RlBlock), else 0 (the hop fields' even-count pad)
    # ---- end MoE-side hops ----
    # ---- integration (stages/integration.cuh) ----
    "IFlags",
    "IResOut",
    "IXBuf",
    "IPad",
    # ---- end integration ----
    # ---- ghost and warm-up units ----
    "I2bGhost",
    "I2bWarm",
    # ---- end ghost and warm-up units ----
)
PARAM = {name: i for i, name in enumerate(PARAM_FIELDS)}

#: ``mso_mk::Flag``.
FLAG_EVICT_FIRST = 1
FLAG_NO_STAMPS = 2
FLAG_X_PRIME = 4
FLAG_CHECK = 8
FLAG_L2_DEFERRED = 16
FLAG_ONE_PIECE = 32
FLAG_MH_UNICAST = 1 << 32  # MoE hops: hop publications as per-rank unicast stores

#: ``mso_mk::WaitCode``.
WAIT_CODES = {
    101: "weights",
    110: "qkv partials",
    111: "qkv ssq partials",
    120: "counter",
    130: "test sub",
    131: "test hang",
    # ---- attention half ----
    140: "attention half: qkv run partials",
    141: "attention half: rope rows",
    142: "attention half: q / step k v rows",
    143: "attention half: kv pages",
    144: "attention half: merge partials / gate",
    145: "attention half: o_proj input",
    # ---- end attention half ----
    # ---- full attention ----
    150: "fa ring stage (TMA)",
    151: "fa S buffer (QK)",
    152: "fa MMA drain",
    153: "fa merge partials",
    154: "fa ring refill (PV)",
    # ---- end full attention ----
    # ---- MoE hops ----
    180: "moe hops: hop1 slots",
    181: "moe hops: row-norm exchange",
    182: "moe hops: x_ffn",
    183: "moe hops: F2' pool / logits",
    184: "moe hops: finalize slots",
    185: "moe hops: m",
    186: "moe hops: F1 branch",
    # ---- end MoE hops ----
}

#: Stamp layout (``mso_mk::kStamp*``).
STAMP_ENTRY = schedule.MAX_ROLES * 4
STAMP_WAIT = STAMP_ENTRY + 1
STAMP_EXIT = STAMP_ENTRY + 2
STAMP_SMID = STAMP_ENTRY + 3
STAMP_ISSUED = STAMP_ENTRY + 4
STAMP_FIRST = STAMP_ENTRY + 5
# ---- prologue stamps (thread 0): the parameter block / context is in shared memory; warp 0's entry issue done ----
STAMP_CTX = STAMP_ENTRY + 6
STAMP_PRO = STAMP_ENTRY + 7
# ---- end prologue stamps ----
STAMP_SUB = STAMP_ENTRY + 8
# ---- clock64 at entry / exit (after the 32 sub-stamps) ----
STAMP_CLK0 = STAMP_SUB + 32
STAMP_CLK1 = STAMP_CLK0 + 1
# ---- end clock64 words ----
# ---- the experts stage's attribution words, after the clock words ----
STAMP_I2A_RAW6 = STAMP_CLK1 + 1
STAMP_I2A_XSTAGED = STAMP_CLK1 + 2
# ---- end experts attribution words ----
# ---- the experts stage's per-unit record (experts_defs.cuh kE3Tr*), after the attribution words ----
STAMP_E3 = STAMP_CLK1 + 3
STAMP_E3_WORDS = 104
# ---- end experts per-unit record ----
# ---- the deferred FaAttn prologue (start, warp 0's part done, ring issued, spare), after the per-unit record ----
STAMP_FA = STAMP_E3 + STAMP_E3_WORDS
STAMP_FA_WORDS = 4
# ---- end FaAttn prologue words ----
# ---- the poll probe (per role two poll classes, each [start, second round, end, (code << 32) | rounds]; clock64)
# and the anchors (clock64 before / after the wait's and the exit's %globaltimer read), after the FaAttn words ----
STAMP_SY = STAMP_FA + STAMP_FA_WORDS
STAMP_SY_ROLE = 8
STAMP_SY_ANCHOR = STAMP_SY + STAMP_SY_ROLE * schedule.MAX_ROLES
STAMP_SY_WORDS = STAMP_SY_ROLE * schedule.MAX_ROLES + 4
#: The fallback for a CTA whose anchors cannot give its SM cycles per ns: 1.0, i.e. no conversion. That CTA's
#: converted words are then its wait anchor (ns) plus a raw SM-cycle offset, not ns.
SY_CYCLES_PER_NS = 1.0
# ---- end poll probe ----

#: Test mailbox bytes (``kPTestMail``: 64 units x 256 B).
TEST_MAIL_BYTES = 64 * 256

_LOCK = threading.Lock()


def arch_target() -> str:
    """The arch-specific target of the current device (``107a`` on Rubin)."""
    major, minor = torch.cuda.get_device_capability()
    if (major, minor) == (10, 7):
        return "107a"
    raise RuntimeError(f"the megakernel is built for sm_107a, not sm_{major}{minor}")


class Library:
    """One compiled mode of ``msoMegakernel.cu`` (``ctypes`` handle + the ABI constants it reports)."""

    def __init__(self, path: str, mode: str):
        self.path = path
        self.mode = mode
        lib = ctypes.CDLL(path)
        self.lib = lib
        for name in (
            "msoMkSmemBytes",
            "msoMkSmemMain",
            "msoMkNumParams",
            "msoMkCtas",
            "msoMkMaxRoles",
            "msoMkTableWords",
            "msoMkWeightWords",
            "msoMkArenaBytes",
            "msoMkClearUnits",
            "msoMkNumCounters",
            "msoMkStampsPerCta",
            "msoMkQkvPartBytes",
            "msoMkLean",
            "msoMkQkvSlices",
            "msoMkStampsMode",
            "msoMkStampsClock",  # clock64 stamps
            "msoMkStampSy",  # the poll probe's first word
            "msoMkInit",
            "msoMkAhPartBytes",
        ):
            getattr(lib, name).restype = ctypes.c_int
            getattr(lib, name).argtypes = []
        lib.msoMkMagic.restype = ctypes.c_int64
        lib.msoMkMagic.argtypes = []
        lib.msoMkAttributes.restype = ctypes.c_int
        lib.msoMkAttributes.argtypes = [ctypes.c_void_p]
        lib.msoMkWeightTable.restype = ctypes.c_int
        lib.msoMkWeightTable.argtypes = [ctypes.c_void_p] * 3
        lib.msoMkLaunch.restype = ctypes.c_int
        lib.msoMkLaunch.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
        lib.msoMkSpin.restype = ctypes.c_int
        lib.msoMkSpin.argtypes = [
            ctypes.c_ulonglong,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_int,
        ]
        assert lib.msoMkNumParams() == len(PARAM_FIELDS), "parameter block mismatch"
        assert lib.msoMkMaxRoles() == schedule.MAX_ROLES, "role table mismatch"
        assert lib.msoMkSmemMain() == schedule.SMEM_MAIN, "shared-memory layout mismatch"
        assert lib.msoMkClearUnits() == schedule.CLEAR_UNITS, "clear units mismatch"
        lean = mode.startswith("lean")
        # lean and stlean builds compile lean code; only lean builds compile no stamp sites.
        assert lib.msoMkLean() == int(
            lean or mode.startswith("stlean")
        ) and lib.msoMkStampsMode() == int(lean), "compile mode mismatch"
        self.qkv_slices = lib.msoMkQkvSlices()
        assert self.qkv_slices == (4 if mode.endswith("_s4") else 1), "compile mode mismatch"
        self.ctas = lib.msoMkCtas()
        self.smem_bytes = lib.msoMkSmemBytes()
        self.table_words = lib.msoMkTableWords()
        self.weight_words = lib.msoMkWeightWords()
        self.arena_bytes = lib.msoMkArenaBytes()
        self.num_counters = lib.msoMkNumCounters()
        self.stamps_per_cta = lib.msoMkStampsPerCta()
        # ---- the poll probe's stamp layout and the clock64 stamps ----
        self.stamps_clock = lib.msoMkStampsClock()
        assert (
            lib.msoMkStampSy() == STAMP_SY and self.stamps_per_cta == STAMP_SY + STAMP_SY_WORDS
        ), "stamp layout"
        assert self.stamps_clock == int("_clk" in mode), "compile mode mismatch"
        # ---- end ----
        self.qkv_part_bytes = lib.msoMkQkvPartBytes()
        self.ah_part_bytes = lib.msoMkAhPartBytes()
        self.magic = int(lib.msoMkMagic())
        sms = torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
        assert sms == self.ctas, (
            f"built for {self.ctas} CTAs (one per SM); this device has {sms} SMs"
        )
        rc = lib.msoMkInit()
        assert rc == 0, f"msoMkInit: CUDA error {rc}"

    def attributes(self) -> Dict[str, int]:
        """Registers per thread, local (spill) bytes, max threads per block and binary version of the kernel."""
        out = (ctypes.c_int * 4)()
        rc = self.lib.msoMkAttributes(out)
        assert rc == 0, f"cudaFuncGetAttributes: CUDA error {rc}"
        return dict(regs=out[0], local_bytes=out[1], max_threads=out[2], binary_version=out[3])

    def spin(
        self,
        spin_ns: int,
        early: bool,
        src: torch.Tensor,
        dst: torch.Tensor,
        ctas: int = 64,
        threads: int = 128,
        pdl: bool = True,
        quiet: bool = True,
    ) -> None:
        """The benchmarks' latency-bound predecessor: waits for its predecessor, releases its dependents at entry
        (``early``) or exit, spins ``spin_ns``, copies ``src`` -> ``dst`` (same bytes, a multiple of 16). ``quiet``:
        one thread per CTA reads the clock with ``__nanosleep`` between reads (else all threads spin on
        ``%globaltimer``)."""
        nbytes = src.numel() * src.element_size()
        assert nbytes == dst.numel() * dst.element_size() and nbytes % 16 == 0
        rc = self.lib.msoMkSpin(
            spin_ns,
            int(early),
            int(quiet),
            src.data_ptr(),
            dst.data_ptr(),
            nbytes // 16,
            ctas,
            threads,
            torch.cuda.current_stream().cuda_stream,
            int(pdl),
        )
        assert rc == 0, f"msoMkSpin: CUDA error {rc}"


def _headers_digest() -> str:
    """A digest of the stage headers (``stages/*.cuh``): torch's extension versioner hashes only the listed sources
    and flags, so the digest goes into the flags to rebuild when a header changes."""

    h = hashlib.sha256()
    for path in sorted((SOURCE.parent / "stages").glob("*.cuh")):
        h.update(path.name.encode())
        h.update(path.read_bytes())
    return h.hexdigest()[:16]


def _build(mode: str, verbose: bool = False) -> str:
    from torch.utils.cpp_extension import load

    target = arch_target()
    path = load(
        name=f"{LIBRARY_NAME}_sm{target}_{mode}",
        sources=[str(SOURCE)],
        extra_cuda_cflags=list(NVCC_FLAGS)
        + list(MODES[mode])
        + [
            f"-gencode=arch=compute_{target},code=sm_{target}",
            f"-DMSO_MK_HEADERS={_headers_digest()}",
        ],
        is_python_module=False,
        verbose=verbose,
    )
    return str(path)


@functools.lru_cache(maxsize=None)
def _library(mode: str, device: int) -> Library:
    with _LOCK:
        return Library(_build(mode), mode)


def library(mode: str = "check") -> Library:
    """The library of compile mode ``mode`` (built on first use; one per process and device)."""
    assert mode in MODES, mode
    return _library(mode, torch.cuda.current_device())


def sass_resource_report(lib: Library) -> str:
    """``cuobjdump -res-usage`` of the library (registers, spills, shared memory per kernel), or why it is missing."""
    from torch.utils import cpp_extension

    cuda_home = cpp_extension.CUDA_HOME or "/usr/local/cuda"
    try:
        out = subprocess.run(
            [os.path.join(cuda_home, "bin", "cuobjdump"), "-res-usage", lib.path],
            capture_output=True,
            text=True,
            timeout=120,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return f"cuobjdump unavailable: {error!r}"
    return out.stdout + out.stderr


@dataclass
class KernelIo:
    """Per-step pointers (``mso_mk::Io``): device tensors or None."""

    x: Optional[torch.Tensor] = None
    res: Optional[torch.Tensor] = None
    qkv: Optional[torch.Tensor] = None
    qkv_f32: Optional[torch.Tensor] = None
    out0: Optional[torch.Tensor] = None
    out1: Optional[torch.Tensor] = None
    in0: Optional[torch.Tensor] = None
    in1: Optional[torch.Tensor] = None

    def pointers(self) -> ctypes.Array:
        vals = [
            0 if t is None else t.data_ptr()
            for t in (
                self.x,
                self.res,
                self.qkv,
                self.qkv_f32,
                self.out0,
                self.out1,
                self.in0,
                self.in1,
            )
        ]
        return (ctypes.c_uint64 * 8)(*vals)


class Workspace:
    """Buffers every launch of the library shares on one device."""

    def __init__(self, lib: Library, copies: int = 3):
        from tensorrt_llm._utils import prefer_pinned

        assert copies >= 2
        dev = torch.device("cuda", torch.cuda.current_device())
        self.lib = lib
        self.copies = int(copies)
        empty = lambda nbytes: torch.full((nbytes // 4,), -0.0, dtype=torch.float32, device=dev)  # noqa: E731
        self.arena = empty(self.copies * lib.arena_bytes)
        self.counters = torch.zeros(self.copies * lib.num_counters, dtype=torch.int32, device=dev)
        self.qkv_part = empty(lib.qkv_part_bytes)
        self.test_mail = empty(TEST_MAIL_BYTES)
        # Pinned host memory: a watchdog's record stays readable after the kernel traps; then per CTA the unit it is
        # in (stamped launches: (stage << 16) | (role << 8) | 1, 0xffff once done).
        self.err = torch.zeros(16 + 5 * lib.ctas, dtype=torch.int32, pin_memory=prefer_pinned())
        # The copy the next launch uses (launch n: copy n % copies), over every Megakernel of this workspace.
        self.next_copy = 0
        # The one copy ``Megakernel.make_layer`` builds a block for (None: every copy), for a caller whose layer
        # always runs on the same copy (a forward of a multiple of ``copies`` launches that starts on copy 0), or
        # ``(copy, num_copies)`` for a launch of a ``copy_plan``.
        self.build_copy = None
        # A cyclic per-launch plan ``[(copy, num_copies)]`` (``copy_plan``) in place of the ``n % copies`` rotation:
        # launch n of the cycle uses ``copy_plan[n][0]``; its ``num_copies`` makes ``(copy +- 1) % num_copies`` the
        # next / previous launch's copy (what the kernel's clears address).
        self.copy_plan = None
        self.plan_pos = 0

    def arena_copy(self, copy: int) -> torch.Tensor:
        """The arena copy ``copy`` as FP32 words."""
        n = self.lib.arena_bytes // 4
        return self.arena[copy * n : (copy + 1) * n]

    def error_record(self) -> Dict[str, object]:
        """The watchdog record: the trap (code, cta, thread, a, b), the CTAs that logged a wait at half the timeout,
        and the unit each CTA was in (stamped launches)."""
        rec = self.err.tolist()
        ctas = self.lib.ctas
        waiting = {
            c: tuple(rec[16 + 4 * c : 20 + 4 * c]) for c in range(ctas) if rec[16 + 4 * c] != 0
        }
        busy = {
            c: v for c, v in enumerate(rec[16 + 4 * ctas : 16 + 5 * ctas]) if v not in (0, 0xFFFF)
        }
        return dict(trap=rec[:5], trap_site=WAIT_CODES.get(rec[0]), waiting=waiting, busy=busy)


class ParamBlock:
    """One device parameter block: the fields, the role tables and the device-built weight table."""

    def __init__(
        self,
        tensor: torch.Tensor,
        copy: int,
        stamps: Optional[torch.Tensor],
        keep: Sequence[object] = (),
    ):
        self.tensor = tensor
        self.copy = copy
        self.stamps = stamps
        # The tensors whose addresses the block holds (weights, tables): alive as long as the block.
        self.keep = tuple(keep)


class Megakernel:
    """One plan of the library: its role tables, shared by every parameter block made from it.

    Args:
        lib: the compiled library (``library(mode)``).
        workspace: the shared buffers.
        tables: the plan's role tables (``schedule.build_tables``).
        timeout_ms: the watchdog's timeout (check builds).
    """

    def __init__(
        self, lib: Library, workspace: Workspace, tables: schedule.Tables, timeout_ms: int = 20000
    ):
        assert tables.ctas == lib.ctas, "the plan has one list per CTA"
        self.lib = lib
        self.ws = workspace
        self.tables = tables
        self.timeout_ns = int(timeout_ms) * 1_000_000

    def make_layer(
        self,
        fields: Dict[str, int],
        flags: int = FLAG_EVICT_FIRST,
        stamps: bool = False,
        keep: Sequence[object] = (),
    ) -> List[ParamBlock]:
        """The parameter blocks of one layer, one per copy (eager; never inside a capture).

        ``fields`` maps ``PARAM_FIELDS`` names (``WQkv``, ``GammaQkv``, ``Eps``, ...) to int64 values (pointers as
        ``data_ptr()``); the shared fields are filled here. ``keep``: the tensors those pointers point into, held by
        the blocks. With ``stamps``, the blocks share one stamp buffer ``[ctas, stamps_per_cta]`` u64."""
        assert not torch.cuda.is_current_stream_capturing(), "parameter blocks are made eagerly"
        lib = self.lib
        dev = self.ws.arena.device
        stamp_buf = (
            torch.zeros(lib.ctas * lib.stamps_per_cta, dtype=torch.int64, device=dev)
            if stamps
            else None
        )
        blocks = []
        # Build only one copy's block (``Workspace.build_copy``: a copy, or (copy, num_copies) of a copy plan)
        only, num_copies = self.ws.build_copy, self.ws.copies
        if isinstance(only, tuple):
            only, num_copies = only
        for copy in range(self.ws.copies):
            if only is not None and copy != only:  # a copy this layer never runs on
                blocks.append(None)
                continue
            host = torch.zeros(
                len(PARAM_FIELDS) + lib.table_words + lib.weight_words, dtype=torch.int64
            )
            vals = dict(
                Magic=lib.magic,
                NumCtas=lib.ctas,
                MaxRoles=schedule.MAX_ROLES,
                SmemBytes=lib.smem_bytes,
                Copy=copy,
                NumCopies=num_copies,  # the copy plan's per-launch count (else the workspace's)
                Counters=self.ws.counters.data_ptr(),
                Arena=self.ws.arena.data_ptr(),
                Err=self.ws.err.data_ptr(),
                TimeoutNs=self.timeout_ns,
                Stamps=0 if stamp_buf is None else stamp_buf.data_ptr(),
                Flags=int(flags),
                QkvPart=self.ws.qkv_part.data_ptr(),
                TestMail=self.ws.test_mail.data_ptr(),
            )
            vals.update(fields)
            for name, v in vals.items():
                host[PARAM[name]] = int(v)
            host[len(PARAM_FIELDS) : len(PARAM_FIELDS) + lib.table_words] = (
                self.tables.flat().view(-1).view(torch.int64)
            )
            block = host.to(dev)
            n = len(PARAM_FIELDS) + lib.table_words
            rc = lib.lib.msoMkWeightTable(
                block.data_ptr(), block[n:].data_ptr(), torch.cuda.current_stream().cuda_stream
            )
            assert rc == 0, f"msoMkWeightTable: CUDA error {rc}"
            blocks.append(ParamBlock(block, copy, stamp_buf, keep))
        torch.cuda.current_stream().synchronize()
        return blocks

    def weight_table(self, block: ParamBlock) -> torch.Tensor:
        """int64 ``[ctas, MAX_ROLES, 3]`` = (src0, src1, n0 | n1 << 32) of a block."""
        n = len(PARAM_FIELDS) + self.lib.table_words
        return block.tensor[n:].view(self.lib.ctas, schedule.MAX_ROLES, 3)

    def launch(self, layer: Sequence[ParamBlock], io: KernelIo, pdl: bool = True) -> ParamBlock:
        """Launch the layer's block of the next copy in the rotation on the current stream; returns the block."""
        ws = self.ws
        if ws.copy_plan is not None:  # the copy plan's copy for this launch position
            copy = ws.copy_plan[ws.plan_pos][0]
            ws.plan_pos = (ws.plan_pos + 1) % len(ws.copy_plan)
        else:
            copy = ws.next_copy
            ws.next_copy = (copy + 1) % ws.copies
        block = layer[copy]
        assert block is not None, (
            f"the layer has no block for copy {copy} (built for one copy only)"
        )
        assert block.copy == copy
        ptrs = io.pointers()
        rc = self.lib.lib.msoMkLaunch(
            block.tensor.data_ptr(),
            ctypes.cast(ptrs, ctypes.c_void_p),
            torch.cuda.current_stream().cuda_stream,
            int(pdl),
        )
        assert rc == 0, f"msoMkLaunch: CUDA error {rc}"
        return block


def stamps_of(lib: Library, block: ParamBlock) -> torch.Tensor:
    """The block's stamps as int64 ``[ctas, stamps_per_cta]`` (host)."""
    assert block.stamps is not None, "a block made with stamps=True"
    return block.stamps.view(lib.ctas, lib.stamps_per_cta).cpu()


# ---- cycle stamps (MSO_MK_SY_CLK builds) onto the device clock ----
#: Words of a CTA row that are %globaltimer anchors, ids or raw values in every build (never converted).
_SY_RAW = {STAMP_ENTRY, STAMP_WAIT, STAMP_EXIT, STAMP_SMID, STAMP_CLK0, STAMP_CLK1, STAMP_I2A_RAW6}


def sy_cycles_per_ns(r: List[int]) -> float:
    """A CTA's SM cycles per ns from its two anchors (the wait's and the exit's %globaltimer reads, each between two
    clock64 reads), or ``SY_CYCLES_PER_NS`` (1.0, no conversion: that CTA's offsets from its wait anchor stay in SM
    cycles) when they cannot give it."""
    a = STAMP_SY_ANCHOR
    cw, cx = (r[a] + r[a + 1]) / 2.0, (r[a + 2] + r[a + 3]) / 2.0
    dt = r[STAMP_EXIT] - r[STAMP_WAIT]
    if r[a] and r[a + 2] and dt > 2000:
        return (cx - cw) / dt
    return SY_CYCLES_PER_NS


def sy_rows_ns(lib: Library, stamps: torch.Tensor) -> List[List[int]]:
    """The stamp rows with every time word in ns of %globaltimer: unchanged for a %globaltimer build; for a clock64
    build every word except the anchors / ids / raw words maps through the CTA's wait anchor (the midpoint of the two
    clock64 reads around it) at the CTA's own cycles per ns, and the probe's round words stay raw. A CTA whose
    anchors cannot give its cycles per ns keeps SM-cycle offsets from its wait anchor (``SY_CYCLES_PER_NS``)."""
    s = stamps.tolist()
    if len(s) == 0 or len(s[0]) < STAMP_SY + STAMP_SY_WORDS:
        return s
    # The probe's clock words (every build), or every time word (clock64 builds).
    probe = {
        STAMP_SY + STAMP_SY_ROLE * i + 4 * c + k
        for i in range(schedule.MAX_ROLES)
        for c in (0, 1)
        for k in (0, 1, 2)
    }
    if getattr(lib, "stamps_clock", 0):
        raw = set(_SY_RAW)
        raw |= {
            STAMP_SY + STAMP_SY_ROLE * i + 4 * c + 3
            for i in range(schedule.MAX_ROLES)
            for c in (0, 1)
        }
        raw |= set(range(STAMP_SY_ANCHOR, STAMP_SY_ANCHOR + 4))
        conv = [k for k in range(len(s[0])) if k not in raw]
    else:
        conv = sorted(probe)
    out = []
    for r in s:
        if not r[STAMP_WAIT] or not r[STAMP_SY_ANCHOR]:
            out.append(r)
            continue
        f = sy_cycles_per_ns(r)
        cw = (r[STAMP_SY_ANCHOR] + r[STAMP_SY_ANCHOR + 1]) / 2.0
        g = r[STAMP_WAIT]
        r = list(r)
        for k in conv:
            if r[k]:
                r[k] = int(round(g + (r[k] - cw) / f))
        out.append(r)
    return out


# ---- end cycle stamps ----


def stamp_report(lib: Library, tables: schedule.Tables, stamps: torch.Tensor) -> Dict[str, object]:
    """Per stage (us from the first CTA's entry): units, first start, last input-ready, last weights-ready, last end,
    and the median / max of a unit's own time after its later of start, inputs and weights; plus the entry, the
    grid-dependency wait return, the entry requests issued and the exit (first / last), and the SMs the CTAs ran on."""
    s = sy_rows_ns(lib, stamps)  # cycle stamps in ns
    roles = tables.roles.tolist()
    ctas = lib.ctas
    entries = [r[STAMP_ENTRY] for r in s if r[STAMP_ENTRY] > 0]
    t0 = min(entries)

    def us(v):
        return round((v - t0) / 1000.0, 3) if v else None

    agg: Dict[str, dict] = {}
    for c in range(ctas):
        for i in range(schedule.MAX_ROLES):
            role = roles[c][i]
            if role == 0:
                break
            name = schedule.STAGE_NAMES.get(role >> 16, str(role >> 16))
            start, ready, wready, end = s[c][4 * i : 4 * i + 4]
            a = agg.setdefault(name, dict(n=0, start=None, ready=0, wready=0, end=0, own=[]))
            a["n"] += 1
            a["start"] = start if a["start"] is None else min(a["start"], start)
            a["ready"] = max(a["ready"], ready)
            a["wready"] = max(a["wready"], wready)
            a["end"] = max(a["end"], end)
            a["own"].append((end - max(start, ready, wready)) / 1000.0)
    col = lambda k: [r[k] for r in s if r[k]]  # noqa: E731
    out: Dict[str, object] = dict(
        t0_ns=t0,
        entry=[us(min(entries)), us(max(entries))],
        first=[us(min(col(STAMP_FIRST))), us(max(col(STAMP_FIRST)))] if col(STAMP_FIRST) else None,
        issued=[us(min(col(STAMP_ISSUED))), us(max(col(STAMP_ISSUED)))],
        wait=[us(min(col(STAMP_WAIT))), us(max(col(STAMP_WAIT)))],
        exit=[us(min(col(STAMP_EXIT))), us(max(col(STAMP_EXIT)))],
        distinct_sms=len({r[STAMP_SMID] for r in s}),
    )
    for name, a in sorted(agg.items(), key=lambda kv: kv[1]["start"]):
        own = sorted(a["own"])
        out[name] = dict(
            n=a["n"],
            start=us(a["start"]),
            ready=us(a["ready"]),
            w=us(a["wready"]),
            end=us(a["end"]),
            own_med=round(own[len(own) // 2], 3),
            own_max=round(own[-1], 3),
        )
    return out


def dump_timeline(
    path: str, lib: Library, tables: schedule.Tables, stamps: torch.Tensor, meta: Dict
) -> None:
    """The raw per-CTA record (stamps, role list) plus ``meta`` as JSON, for offline timelines: per CTA its entry,
    issued, wait, exit, SM id, and per unit (stage, unit, start, inputs, weights, end) in ns from the first entry."""
    s = sy_rows_ns(lib, stamps)  # cycle stamps in ns
    t0 = min(r[STAMP_ENTRY] for r in s if r[STAMP_ENTRY] > 0)
    rel = lambda v: (v - t0) if v else None  # noqa: E731
    ctas = []
    for c, r in enumerate(s):
        units = []
        for i in range(schedule.MAX_ROLES):
            role = int(tables.roles[c, i])
            if role == 0:
                break
            units.append(
                dict(
                    stage=schedule.STAGE_NAMES.get(role >> 16, role >> 16),
                    unit=role & 0xFFFF,
                    start=rel(r[4 * i]),
                    inputs=rel(r[4 * i + 1]),
                    weights=rel(r[4 * i + 2]),
                    end=rel(r[4 * i + 3]),
                )
            )
        ctas.append(
            dict(
                cta=c,
                sm=r[STAMP_SMID],
                entry=rel(r[STAMP_ENTRY]),
                first=rel(r[STAMP_FIRST]),
                issued=rel(r[STAMP_ISSUED]),
                ctx=rel(r[STAMP_CTX]),  # prologue: context in shared memory
                pro=rel(r[STAMP_PRO]),  # prologue: entry issue done
                wait=rel(r[STAMP_WAIT]),
                exit=rel(r[STAMP_EXIT]),
                sub=[
                    rel(v) for v in r[STAMP_SUB:STAMP_CLK0]
                ],  # the clock words follow the sub-stamps
                clk0=r[STAMP_CLK0] if len(r) > STAMP_CLK1 else None,  # clock64 at entry
                clk1=r[STAMP_CLK1] if len(r) > STAMP_CLK1 else None,  # clock64 at exit
                raw6=r[STAMP_I2A_RAW6]
                if len(r) > STAMP_I2A_XSTAGED
                else None,  # experts (E << 32) | units, raw
                xstaged=rel(r[STAMP_I2A_XSTAGED])
                if len(r) > STAMP_I2A_XSTAGED
                else None,  # experts: X staged
                # the experts' per-unit record (FC2 requests / h / full / issue, h k-blocks, tiles, FC1 tail)
                e3=[rel(v) for v in r[STAMP_E3 : STAMP_E3 + STAMP_E3_WORDS]]
                if len(r) >= STAMP_E3 + STAMP_E3_WORDS
                else None,
                # the deferred FaAttn prologue's stamps
                fa=[rel(v) for v in r[STAMP_FA : STAMP_FA + STAMP_FA_WORDS]]
                if len(r) >= STAMP_FA + STAMP_FA_WORDS
                else None,
                # the poll probe: per role, thread 0's last poll of each class (0: input polls, 1: poll1 / merge /
                # experts pool) as [start, end, wait code, rounds, second round start], and the anchors' clock64
                # brackets (cycles), the CTA's cycles per ns
                sy=[
                    [
                        [
                            rel(r[STAMP_SY + STAMP_SY_ROLE * i + 4 * c]),
                            rel(r[STAMP_SY + STAMP_SY_ROLE * i + 4 * c + 2]),
                            r[STAMP_SY + STAMP_SY_ROLE * i + 4 * c + 3] >> 32,
                            r[STAMP_SY + STAMP_SY_ROLE * i + 4 * c + 3] & 0xFFFFFFFF,
                            rel(r[STAMP_SY + STAMP_SY_ROLE * i + 4 * c + 1]),
                        ]
                        for c in (0, 1)
                    ]
                    for i in range(len(units))
                ]
                if len(r) >= STAMP_SY + STAMP_SY_WORDS
                else None,
                sy_anchor=dict(
                    wait_cyc=r[STAMP_SY_ANCHOR + 1] - r[STAMP_SY_ANCHOR],
                    exit_cyc=r[STAMP_SY_ANCHOR + 3] - r[STAMP_SY_ANCHOR + 2],
                    cyc_per_ns=round(sy_cycles_per_ns(r), 4),
                    clock=int(getattr(lib, "stamps_clock", 0)),
                )
                if len(r) >= STAMP_SY + STAMP_SY_WORDS
                else None,
                units=units,
            )
        )
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump(dict(meta=meta, t0_ns=t0, ctas=ctas), f)
