# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""NVFP4-out MLA decode epilogues (TRTLLM_MLA_NVFP4_EPILOGUE) on Rubin SM107: byte-exact against today's path.

The FP4 MLA module reads TRTLLM_FP4_MLA_MXFP4_2D at import, so the check runs the microbenchmark checker
(tests/microbenchmarks/check_dsr1_mla_nvfp4_out.py) in a fresh interpreter.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

_BENCH_DIR = Path(__file__).resolve().parents[3] / "microbenchmarks"


def _is_rubin() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability() == (10, 7)


def _run(*args: str) -> str:
    env = dict(os.environ)
    env.update({"TRTLLM_FP4_MLA_MXFP4_2D": "1", "TRTLLM_FP4_MLA_ATTENTION_BACKEND": "cutedsl"})
    result = subprocess.run(
        [sys.executable, str(_BENCH_DIR / "check_dsr1_mla_nvfp4_out.py"), *args],
        env=env,
        capture_output=True,
        text=True,
        timeout=1800,
        check=False,
    )
    assert result.returncode == 0, result.stdout[-4000:] + result.stderr[-4000:]
    return result.stdout


@pytest.mark.skipif(not _is_rubin(), reason="requires Rubin SM107 and the CuTeDSL runtime")
def test_bmm_o_nvfp4_epilogue_matches_bmm_and_fp4_quantize():
    out = _run("--parts", "O", "--tokens", "1", "16", "129", "992")
    assert "NVFP4OUT_DONE ok=True" in out, out[-4000:]


@pytest.mark.skipif(not _is_rubin(), reason="requires Rubin SM107 and the CuTeDSL runtime")
def test_bmm_q_fp4_prefix_epilogue_matches_fused_writer():
    out = _run("--parts", "Q", "--tokens", "16", "129", "992")
    assert "NVFP4OUT_DONE ok=True" in out, out[-4000:]
