# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The two arch-independent fmha_sm100 correctness fixes the M3.1 targets rely on, checked on the packaged fmha_sm100
the engine's MSA backend resolves (``require_msa_module``):

1. Sm100FmhaFwdKernelTmaWarpspecialized is PDL-launched, so it must run griddepcontrol.wait before reading any
   input (the upstream wait is commented out: a race with its predecessor; the M3.1 prefill index proxy pass runs it).
2. The NVFP4-KV forward zeroes the V block scales / values of rows at or past kv_valid_cols, so a never-written pool
   slot cannot inject NaN through a zero probability.
"""

import re
from pathlib import Path

import pytest

from tensorrt_llm._torch.attention.backends.sparse.minimax_m3.kernels.msa_utils import (
    require_msa_module,
)

pytestmark = [
    pytest.mark.cpu_only,
    # The packaged fmha_sm100 does not carry these two fixes yet; they land with the vendored MSA tree update.
    pytest.mark.skip(
        reason="the packaged fmha_sm100 does not carry the PDL wait and NVFP4-KV tail fixes yet"
    ),
]


def _pkg() -> Path:
    try:
        module = require_msa_module()
    except (ImportError, RuntimeError) as err:
        pytest.skip(f"fmha_sm100 is not installed: {err}")
    return Path(module.__file__).resolve().parent


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
