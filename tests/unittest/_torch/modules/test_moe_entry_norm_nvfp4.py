# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""MoE-entry add-RMSNorm with NVFP4 output (TRTLLM_MOE_ENTRY_NORM_NVFP4): byte-exact against FlashInfer's CuTe DSL
fused add-RMSNorm followed by fp4_quantize (linear scale factors), eager and under CUDA-graph replay.

Runs tests/microbenchmarks/check_dsr1_moe_entry_norm_nvfp4.py in a fresh interpreter.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

_BENCH_DIR = Path(__file__).resolve().parents[3] / "microbenchmarks"


def _is_blackwell_family() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 10


@pytest.mark.skipif(not _is_blackwell_family(), reason="requires SM10x and the CuTeDSL runtime")
def test_fused_add_rmsnorm_nvfp4_matches_norm_then_quantize():
    result = subprocess.run(
        [
            sys.executable,
            str(_BENCH_DIR / "check_dsr1_moe_entry_norm_nvfp4.py"),
            "--tokens",
            "1",
            "7",
            "128",
            "1536",
        ],
        env=dict(os.environ),
        capture_output=True,
        text=True,
        timeout=1800,
        check=False,
    )
    assert result.returncode == 0, result.stdout[-4000:] + result.stderr[-4000:]
    assert "MOE_ENTRY_DONE ok=True" in result.stdout, result.stdout[-4000:]
