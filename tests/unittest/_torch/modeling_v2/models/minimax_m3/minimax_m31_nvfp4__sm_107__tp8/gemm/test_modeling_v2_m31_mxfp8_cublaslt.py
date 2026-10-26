# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the m3_mxfp8_cublaslt entry of the MiniMax-M3.1 sm_107 tp8 target (cuBLASLt's MXFP8 GEMM for the
generation steps past the decode bound, LINEAR_CUBLASLT)."""

import os
import subprocess
import sys
import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m31_nvfp4__sm_107__tp8.ops.gemm import (  # noqa: E501
    m3_mxfp8_cublaslt as entry,
)

__extra_import_path__ = ["../../../..", "."]
from _m3_mxfp8_operands import mxfp8  # noqa: E402
from _validating import assert_within_ulp, validating  # noqa: E402 — needs the path declared above

assert torch.cuda.is_available(), "m3_mxfp8_cublaslt requires a CUDA device"
op = entry.m3_mxfp8_cublaslt


def _operands(cell):
    spec = cell.spec
    g = torch.Generator(device="cuda").manual_seed(zlib.crc32(cell.why.encode()) % 2**31)
    w, w_sf = mxfp8(spec["n"], spec["k"], g)
    if spec["prequant"]:
        act, act_sf = mxfp8(spec["m"], spec["k"], g)
    else:
        act = (torch.randn(spec["m"], spec["k"], generator=g, device="cuda") / 4).to(torch.bfloat16)
        act_sf = None
    return act, act_sf, w, w_sf


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:48] for c in op.CELLS])
def test_certified_cells(cell) -> None:
    """Same E4M3 operands and power-of-two scales on both sides; accumulation order then one bf16 rounding: at most
    2 units of ULP[bf16] of the row, RMS well under one."""
    act, act_sf, w, w_sf = _operands(cell)
    with validating(op):
        out = op(act, act_sf, w, w_sf)
    assert out.shape == (cell.spec["m"], cell.spec["n"]) and out.dtype == torch.bfloat16
    assert_within_ulp(out, op.reference(act, act_sf, w, w_sf), element_ulp=2.0, rms_ulp=0.25)


def test_the_quantize_is_the_native_one() -> None:
    """The bf16 path's CuTe DSL quantize gives the bytes of ``torch.ops.trtllm.mxfp8_quantize(x, True)``, so the
    cuBLASLt GEMM sees the operands FlashInfer's quantize + GEMM would."""
    g = torch.Generator(device="cuda").manual_seed(7)
    x = (torch.randn(64, 6144, generator=g, device="cuda") / 4).to(torch.bfloat16)
    act, act_sf = entry.mxfp8_quantize(x)
    ref, ref_sf = torch.ops.trtllm.mxfp8_quantize(x, True)
    assert torch.equal(act.view(torch.uint8), ref.view(64, 6144).view(torch.uint8))
    assert torch.equal(act_sf.view(torch.uint8).reshape(-1), ref_sf.view(torch.uint8).reshape(-1))


@pytest.mark.parametrize("nk", entry.SHAPES, ids=str)
def test_repeats_and_graph_replay_are_bit_identical(nk) -> None:
    """Three eager calls and a CUDA-graph replay (captured on a fresh stream) give the same bits: cuBLASLt's
    algorithm is a pure function of the shape and the process's workspace."""
    n, k = nk
    g = torch.Generator(device="cuda").manual_seed(n + k)
    w, w_sf = mxfp8(n, k, g)
    act, act_sf = mxfp8(entry.MAX_TOKENS, k, g)
    first = op(act, act_sf, w, w_sf)
    for _ in range(2):
        assert torch.equal(op(act, act_sf, w, w_sf), first)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            out = op(act, act_sf, w, w_sf)
    torch.cuda.current_stream().wait_stream(stream)
    out.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(out, first)


# A fresh process: the target's prebuild warm-up (one eager call per shape, rows and activation), then the first
# capture of the entry on a fresh stream; prints the sha256 of the eager and the replayed output per (shape, rows).
_FRESH = """
import hashlib, sys, torch
sys.path[:0] = sys.argv[1:3]
from {mod} import m3_mxfp8_cublaslt as op, mxfp8_quantize, SHAPES, MIN_TOKENS, MAX_TOKENS
from _m3_mxfp8_operands import mxfp8
for (n, k) in SHAPES:
    g = torch.Generator(device="cuda").manual_seed(n * 7 + k)
    w, w_sf = mxfp8(n, k, g)
    for m in (MIN_TOKENS, MAX_TOKENS):
        x = (torch.randn(m, k, generator=g, device="cuda") / 4).to(torch.bfloat16)
        act, act_sf = mxfp8_quantize(torch.zeros(m, k, dtype=torch.bfloat16, device="cuda"))
        op(torch.zeros_like(x), None, w, w_sf)
        op(act, act_sf, w, w_sf)
        eager = op(x, None, w, w_sf)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                out = op(x, None, w, w_sf)
        torch.cuda.current_stream().wait_stream(stream)
        graph.replay()
        torch.cuda.synchronize()
        h = lambda t: hashlib.sha256(t.contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()[:16]
        print("SHA", n, k, m, h(eager), h(out))
"""


def _fresh_process() -> dict:
    here = os.path.dirname(os.path.abspath(__file__))
    code = _FRESH.format(mod=entry.__name__)
    out = subprocess.run(
        [sys.executable, "-c", code, here, os.path.join(here, "../../../..")],
        capture_output=True,
        text=True,
        check=True,
    )
    rows = [line.split() for line in out.stdout.splitlines() if line.startswith("SHA ")]
    assert len(rows) == 2 * len(entry.SHAPES), out.stdout + out.stderr
    return {tuple(r[1:4]): (r[4], r[5]) for r in rows}


def test_capture_after_the_prebuild_warmup_and_across_processes() -> None:
    """In two fresh processes: after the prebuild-style eager warm-up, the entry's first capture works and replays the
    eager bits (capture safety), and both processes give the same bits at every shape and its first / last row
    (cross-process determinism: cuBLASLt's heuristic algorithm with torch's fixed default workspace, no timing)."""
    first, second = _fresh_process(), _fresh_process()
    for key, (eager, replay) in first.items():
        assert eager == replay, f"{key}: the replay differs from the eager call"
        assert second[key] == (eager, replay), f"{key}: another process gave other bits"


def test_the_range_is_past_the_decode_bound() -> None:
    """The entry's rows start past the R1VR8 decode bound and stop at the K7 verify rows at batch 8."""
    assert entry.MIN_TOKENS == 33
    assert entry.MAX_TOKENS == 64
