# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""MXFP4-2D FP4 MLA KV: writer bit-exactness and decode numerics on SM107.

TRTLLM_FP4_MLA_MXFP4_2D is read when the FP4 MLA module is imported, so each
check runs the microbenchmark checker in a fresh interpreter.
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


def _run(script: str, *args: str, env_overrides: dict[str, str]) -> str:
    env = dict(os.environ)
    env.update(env_overrides)
    result = subprocess.run(
        [sys.executable, str(_BENCH_DIR / script), *args],
        env=env,
        capture_output=True,
        text=True,
        timeout=1800,
        check=False,
    )
    assert result.returncode == 0, result.stdout[-4000:] + result.stderr[-4000:]
    return result.stdout


@pytest.mark.skipif(not _is_rubin(), reason="requires Rubin SM107 and the CuTeDSL runtime")
def test_mxfp4_2d_writer_matches_direct_quantization():
    out = _run(
        "check_dsr1_mx2d_writer.py",
        "--steps",
        "40",
        env_overrides={"TRTLLM_FP4_MLA_MXFP4_2D": "1"},
    )
    assert "MX2D_WRITER_DONE ok=True" in out, out[-4000:]


@pytest.mark.skipif(not _is_rubin(), reason="requires Rubin SM107 and the CuTeDSL runtime")
def test_mxfp4_2d_decode_matches_reference():
    out = _run(
        "check_dsr1_fp4_mla_decode_numerics.py",
        "--backends",
        "cutedsl",
        "--seq-lens",
        "1",
        "31",
        "33",
        "128",
        "129",
        "1000",
        "3000",
        env_overrides={"TRTLLM_FP4_MLA_MXFP4_2D": "1"},
    )
    assert "NUMERICS_DONE ok=True" in out, out[-4000:]
    exact = [line for line in out.splitlines() if "cutedsl vs exact:" in line]
    assert exact, out[-4000:]
    rel_l2 = float(exact[0].split("rel_l2=")[1].split()[0])
    # MXFP4-2D decode vs FP32 math on the same dequantized MXFP4-2D cache.
    assert rel_l2 < 0.05, exact[0]


@pytest.mark.skipif(not _is_rubin(), reason="requires Rubin SM107 and the CuTeDSL runtime")
def test_mxfp4_2d_split_kv_matches_unsplit():
    # TRTLLM_FP4_MLA_MXFP4_2D_SPLITKV on/off through the production dispatch, eager and in
    # CUDA graphs (including a replay that re-plans for new KV lengths).
    batches = ("1", "8", "64")
    out = _run(
        "check_persistent_mx2d_decode.py",
        "mx2d_split",
        "--batch",
        *batches,
        "--time",
        "--soak",
        "20",
        "--iters",
        "5",
        env_overrides={},
    )
    assert "CHECK_DONE" in out and "CHECK_ERROR" not in out, out[-4000:]
    worst = [line for line in out.splitlines() if "worst_seq_rel_l2" in line]
    assert len(worst) == len(batches), out[-4000:]
    for line in worst:
        fields = dict(f.split("=", 1) for f in line.split() if "=" in f)
        assert fields["rerun"] == "BITWISE", line
        # Split-KV only changes the rounding (window anchors and merge order).
        assert float(fields["split"]) <= 1.01 * float(fields["unsplit"]) + 1e-4, line
    assert out.count("MX2D_SPLIT_TIMING") == len(batches), out[-4000:]
