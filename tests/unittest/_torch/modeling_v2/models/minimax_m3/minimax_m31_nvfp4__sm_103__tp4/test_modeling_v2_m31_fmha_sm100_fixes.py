# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The two arch-independent fmha_sm100 correctness fixes the M3.1 targets carry in
3rdparty/patches/msa_strided_paged_kv.patch (from #20629 cf8a77e816), checked on the installed package:

1. Sm100FmhaFwdKernelTmaWarpspecialized is PDL-launched, so it must run griddepcontrol.wait before reading any
   input (the upstream wait is commented out: a race with its predecessor; the M3.1 prefill index proxy pass runs it).
2. The NVFP4-KV forward zeroes the V block scales / values of rows at or past kv_valid_cols, so a never-written pool
   slot cannot inject NaN through a zero probability.
"""

import importlib.util
import re
from pathlib import Path

import pytest

_SPEC = importlib.util.find_spec("fmha_sm100")
pytestmark = pytest.mark.skipif(_SPEC is None, reason="fmha_sm100 is not installed")


def _pkg() -> Path:
    return Path(_SPEC.origin).parent


def test_the_fwd_kernel_waits_on_its_grid_dependency_before_reading_inputs() -> None:
    src = (_pkg() / "csrc/include/sm100_fmha_fwd_kernel_tma_warpspecialized.hpp").read_text()
    body = src[src.index("CUTLASS_DEVICE void operator()(const Params& params, char* smem) {") :]
    first_lines = body.split("\n")[1:6]
    active = [
        line
        for line in first_lines
        if "griddepcontrol.wait" in line and not line.lstrip().startswith("//")
    ]
    assert active, "griddepcontrol.wait is missing or commented out at the top of the fwd kernel"


def test_the_nvfp4_kv_forward_zeroes_v_rows_past_the_valid_columns() -> None:
    src = (_pkg() / "cute/src/sm100/fwd/atten_fwd_nvfp4_kv.py").read_text()
    assert re.search(r"scale_e4m3_lo = scale_e4m3_lo if row < kv_valid_cols else Int32\(0\)", src)
    assert re.search(r"scale_e4m3_hi = scale_e4m3_hi if row < kv_valid_cols else Int32\(0\)", src)
    assert re.search(
        r"combined_bf16x2 = combined_bf16x2 if row < kv_valid_cols else Int32\(0\)", src
    )
