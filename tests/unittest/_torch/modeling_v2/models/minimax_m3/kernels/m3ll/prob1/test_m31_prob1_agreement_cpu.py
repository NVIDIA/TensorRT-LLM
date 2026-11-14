# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The draft prob1 agreement and exchange protocols of the family library (csrc/m31/prob1Agreement.h,
prob1Exchange.h): their host-side transitions compiled into a CPU model and checked there, and the ordering the
device code and the op wrapper must keep around them."""

import shutil
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.cpu_only

# tests/unittest/_torch/modeling_v2/models/minimax_m3/kernels/m3ll/prob1/ -> the repository root.
ROOT = Path(__file__).resolve().parents[9]
MODELS = ROOT / "tensorrt_llm/_torch/_experimental/modeling_v2/models"
CSRC = MODELS / "minimax_m3/csrc/m31"


def test_the_shared_protocol_on_cpu(tmp_path):
    compiler = shutil.which("clang++") or shutil.which("g++")
    if compiler is None:
        pytest.skip("no C++17 host compiler")
    binary = tmp_path / "prob1_agreement"
    subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-DTRTLLM_MV2_SCOPE=minimax_m3::shared",
            "-I",
            str(ROOT / "cpp/include"),
            "-I",
            str(ROOT / "cpp"),
            "-I",
            str(MODELS),
            str(Path(__file__).with_name("prob1_agreement_cpu.cpp")),
            "-o",
            str(binary),
        ],
        check=True,
    )
    result = subprocess.run([str(binary)], check=True, capture_output=True, text=True)
    assert "PROB1_AGREEMENT_CPU_PASS" in result.stdout


def test_full_tags_and_reader_ack_wiring():
    exchange = (CSRC / "prob1Exchange.h").read_text()
    # A payload is written before its tag is released, and read after its tag is acquired.
    assert exchange.index("st.relaxed.sys") < exchange.index("st.release.sys")
    assert exchange.index("ld.acquire.sys") < exchange.index("ld.relaxed.sys")
    agreement = (CSRC / "prob1Agreement.h").read_text()
    assert "readerDone(load(peers[r] + kDoneOffset), priorSeq)" in agreement
    assert "proposal(observed, priorSeq + 1, Abort)" in agreement
    shared_header = '#include "minimax_m3/csrc/m31/prob1Agreement.h"'
    wired = []
    for tp in (4, 8):
        cu = (CSRC / f"draftProb1Tp{tp}.cu").read_text()
        if shared_header not in cu:
            # A kernel on its own agreement takes none of the shared protocol; it must not half-use it either.
            assert "prob1Agreement::" not in cu, tp
            continue
        wired.append(tp)
        assert "prob1Agreement::prepare(world, peers.p, seq, capNs)" in cu
        assert cu.index("writeKernel<<<") < cu.index(f"finishReaders<{tp}>")
        wrapper = (CSRC / f"draftProb1Tp{tp}Op.cpp").read_text()
        assert wrapper.index("CUDAGuard guard(base.device())") < wrapper.index(
            "int64_t const mg = dp::maxGrid"
        )
        assert "q.readerAck" in wrapper and "q.seq" in wrapper
    # The 4-rank-layout draft prob1 kernel, which the tp2 target runs, uses the shared protocol.
    assert 4 in wired, wired
