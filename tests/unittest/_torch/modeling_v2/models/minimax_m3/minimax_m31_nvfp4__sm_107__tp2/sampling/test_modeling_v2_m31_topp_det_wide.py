# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for TOPP_DET at the TP2 WIDE row counts (9..448 rows, ``m31_topp_det.MAX_ROWS``): V4K8 under every switch
combination equals v1.2 (``minimax_m31_topp_renorm_det``) bit for bit and FlashInfer's det renorm on the top_p < 1
rows, at 2 / 1 CTAs per row, on one workspace alternating narrow and wide calls and in CUDA-graph replays. The entry
takes up to 448 rows with v3 and keeps v1.2 at 8 rows (TRTLLM_M31_DISABLE_TOPPDET_V3=1), and logs the cap it uses.
"""

import os
import subprocess
import sys

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m31_nvfp4__sm_107__tp2.ops.sampling import (  # noqa: E501
    m31_topp_det as entry,
)

TARGET = (
    "tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m31_nvfp4__sm_107__tp2"
)
D = 200064
CLASSES = ("randn", "peaked", "ties", "near_uniform", "all_equal", "one_hot", "tiny_tail")
COMBOS = [(True, True), (True, False), (False, True), (False, False)]
WIDE = (9, 16, 48, 64, 128, 192, 256, 448)


def _ops():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7):
        pytest.skip("this TOPP_DET target is sm_107")
    v12 = entry._load_op("minimax_m31_topp_renorm_det")
    v3 = entry._load_op("minimax_m31_topp_renorm_det_v3")
    if v12 is None or v3 is None or "list_path" not in str(v3.default._schema):
        pytest.skip("needs minimax_m31_topp_renorm_det and a V4K8 minimax_m31_topp_renorm_det_v3")
    pytest.importorskip("flashinfer.sampling")
    return v12, v3


def _make(cls, rows, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    if cls == "randn":
        logits = torch.randn(rows, D, generator=g, device="cuda") * 3
    elif cls == "peaked":
        logits = torch.randn(rows, D, generator=g, device="cuda") * 2
        idx = torch.randint(0, D, (rows, 8), generator=g, device="cuda")
        logits.scatter_(1, idx, 14 + 3 * torch.rand(rows, 8, generator=g, device="cuda"))
    elif cls == "ties":
        logits = torch.round(torch.randn(rows, D, generator=g, device="cuda") * 2) / 2
    elif cls == "near_uniform":
        logits = torch.randn(rows, D, generator=g, device="cuda") * 1e-3
    elif cls == "all_equal":
        logits = torch.zeros(rows, D, device="cuda")
    elif cls == "one_hot":
        logits = torch.full((rows, D), -1e4, device="cuda")
        logits[torch.arange(rows), torch.randint(0, D, (rows,), generator=g, device="cuda")] = 0
    elif cls == "tiny_tail":
        logits = torch.randn(rows, D, generator=g, device="cuda") * 30
    else:
        raise ValueError(cls)
    return torch.softmax(logits, -1).contiguous()


def _top_p(mode, rows, seed):
    g = torch.Generator(device="cuda").manual_seed(seed + 7)
    if mode == "p95":
        return torch.full((rows,), 0.95, device="cuda")
    if mode == "mixed":
        return (0.05 + 0.95 * torch.rand(rows, generator=g, device="cuda")).contiguous()
    tp = torch.full((rows,), 0.95, device="cuda")
    tp[seed % 2 :: 2] = 1.0  # with_p1: keep-all rows interleaved
    return tp


class _Runner:
    """v1.2 and V4K8 on separate zeroed workspaces sized for MAX_ROWS rows."""

    def __init__(self, v12, v3):
        self.v12, self.v3 = v12, v3
        n12 = int(torch.ops.trtllm.minimax_m31_topp_renorm_det_ws_bytes())
        n3 = int(torch.ops.trtllm.minimax_m31_topp_renorm_det_v3_ws_bytes())
        self.ws12 = torch.zeros(entry.MAX_ROWS * n12, dtype=torch.uint8, device="cuda")
        self.ws3 = torch.zeros(entry.MAX_ROWS * n3, dtype=torch.uint8, device="cuda")
        self.sms = torch.cuda.get_device_properties(0).multi_processor_count

    def g(self, rows):
        return entry.ctas_per_row(rows, self.sms)

    def ref(self, probs, tp):
        out = torch.empty_like(probs)
        self.v12(probs, tp, self.ws12, out, self.g(probs.shape[0]))
        return out

    def new(self, probs, tp, pdl, combo, out=None):
        out = torch.empty_like(probs) if out is None else out
        self.v3(probs, tp, self.ws3, out, self.g(probs.shape[0]), pdl, True, combo[0], combo[1])
        return out


def _bad_rows(a, b):
    return int((a.view(torch.int32) != b.view(torch.int32)).any(1).sum())


def _fi_det(probs, top_p):
    import flashinfer.sampling

    return flashinfer.sampling.top_p_renorm_probs(probs, top_p, is_deterministic=True)


@pytest.mark.parametrize("combo", COMBOS)
@pytest.mark.parametrize("rows", WIDE)
def test_wide_bitwise_v12_and_flashinfer(rows, combo):
    """Every class and top_p mode: all rows equal v1.2's, and the top_p < 1 rows FlashInfer's det renorm."""
    r = _Runner(*_ops())
    bad_v12 = bad_fi = 0
    for cls in CLASSES:
        for mode in ("p95", "mixed", "with_p1"):
            probs, tp = _make(cls, rows, 31 * rows + len(mode)), _top_p(mode, rows, rows)
            got = r.new(probs, tp, True, combo)
            bad_v12 += _bad_rows(got, r.ref(probs, tp))
            below = tp < 1
            bad_fi += _bad_rows(got[below], _fi_det(probs, tp)[below])
    assert bad_v12 == 0 and bad_fi == 0, (bad_v12, bad_fi)


def test_wide_alternating_rows_one_workspace():
    """48 launches on one workspace alternating 8 / 64 / 448 / 9 rows (16 / 2 / 1 / 16 CTAs per row), the four
    combinations and PDL: the barrier counters, parity buffers and lists stay consistent."""
    r = _Runner(*_ops())
    for i in range(48):
        rows = (8, 64, 448, 9)[i % 4]
        probs = _make(CLASSES[i % len(CLASSES)], rows, 500 + i)
        tp = _top_p(("p95", "mixed", "with_p1")[i % 3], rows, i)
        got = r.new(probs, tp, bool(i % 2), COMBOS[(i // 4) % 4])
        assert _bad_rows(got, r.ref(probs, tp)) == 0, (i, rows)


def test_wide_graph_replay_equals_eager():
    """The entry at 96 rows captured in a CUDA graph (workspace prebuilt outside the capture) gives the eager bits."""
    _ops()
    if not entry.is_v3():
        pytest.skip("v3 is not the engaged op")
    rows = 96
    probs = _make("peaked", rows, 5).clone()
    tp = _top_p("mixed", rows, 5).clone()
    entry.prebuild(probs.device)
    ref = entry.top_p_renorm_det(probs, tp)
    assert ref is not None, "96 rows are within the v3 cap"
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        entry.top_p_renorm_det(probs, tp)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            out = entry.top_p_renorm_det(probs, tp)
    assert out is not None, "a prebuilt workspace must serve a captured call"
    for k, cls in enumerate(("randn", "near_uniform", "peaked")):
        probs.copy_(_make(cls, rows, 70 + k))
        tp.copy_(_top_p("with_p1" if k == 2 else "p95", rows, k))
        out.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        assert _bad_rows(out, entry.top_p_renorm_det(probs, tp)) == 0, cls


def _probe(env):
    code = (
        f"import importlib, torch; m = importlib.import_module('{TARGET}.ops.sampling.m31_topp_det'); "
        "m.topp_det_op(); p = torch.softmax(torch.randn(9, 200064, device='cuda'), -1); "
        "t = torch.full((9,), 0.9, device='cuda'); "
        "print('CAP', m.max_rows(), m.top_p_renorm_det(p, t) is not None, "
        "m.prebuild().numel() == m.max_rows() * m.ws_bytes_per_row())"
    )
    out = subprocess.run(
        [sys.executable, "-c", code],
        env={**os.environ, "TLLM_LOG_LEVEL": "INFO", **env},
        capture_output=True,
        text=True,
        check=True,
    )
    text = out.stdout + out.stderr
    lines = [ln for ln in text.splitlines() if "TOPP_DET row cap:" in ln]
    return [ln for ln in out.stdout.splitlines() if ln.startswith("CAP ")][-1], lines


def test_row_cap_follows_the_engaged_op():
    """v3 takes 9 rows (cap 448, workspace for 448 rows); v1.2 declines them (cap 8); one cap line per process."""
    _ops()
    v3_cap, v3_lines = _probe({"TRTLLM_M31_DISABLE_TOPPDET_V3": "0"})
    v12_cap, v12_lines = _probe({"TRTLLM_M31_DISABLE_TOPPDET_V3": "1"})
    assert v3_cap == f"CAP {entry.MAX_ROWS} True True" and entry.MAX_ROWS == 448
    assert v12_cap == f"CAP {entry.V12_MAX_ROWS} False True" and entry.V12_MAX_ROWS == 8
    assert len(v3_lines) == 1 and "row cap: 448 rows" in v3_lines[0]
    assert len(v12_lines) == 1 and "row cap: 8 rows" in v12_lines[0]


def _engagement_lines(env, rows_seq):
    code = (
        f"import importlib, torch; m = importlib.import_module('{TARGET}.ops.sampling.m31_topp_det'); "
        f"[m.top_p_renorm_det(torch.softmax(torch.randn(r, {D}, device='cuda'), -1), "
        "torch.full((r,), 0.9, device='cuda')) "
        f"for r in {list(rows_seq)}]; torch.cuda.synchronize()"
    )
    out = subprocess.run(
        [sys.executable, "-c", code],
        env={**os.environ, "TLLM_LOG_LEVEL": "INFO", **env},
        capture_output=True,
        text=True,
        check=True,
    )
    text = out.stdout + out.stderr
    return [
        ln for ln in text.splitlines() if "TOPP_DET taken at" in ln or "TOPP_DET stock at" in ln
    ]


def test_taken_line_per_distinct_row_count():
    """One "TOPP_DET taken at N rows" line per distinct served row count, one "TOPP_DET stock at N rows" line per
    declined one (above the cap), never repeated; with v1.2 the 9-row call is logged stock."""
    _ops()
    lines = _engagement_lines(
        {"TRTLLM_M31_DISABLE_TOPPDET_V3": "0"}, (8, 8, 64, 9, 64, 449, 449, 8)
    )
    taken = sorted(
        ln.split("TOPP_DET taken at ")[1].split(" rows")[0] for ln in lines if "taken at" in ln
    )
    stock = [ln for ln in lines if "TOPP_DET stock at" in ln]
    assert taken == ["64", "8", "9"], lines
    assert len(stock) == 1 and "stock at 449 rows (above the 448-row cap)" in stock[0], lines
    assert all("ctas_per_row" in ln for ln in lines if "taken at" in ln)
    v12 = _engagement_lines({"TRTLLM_M31_DISABLE_TOPPDET_V3": "1"}, (8, 9, 9))
    assert len(v12) == 2, v12
    assert (
        "TOPP_DET taken at 8 rows" in v12[0]
        and "TOPP_DET stock at 9 rows (above the 8-row cap)" in v12[1]
    )
