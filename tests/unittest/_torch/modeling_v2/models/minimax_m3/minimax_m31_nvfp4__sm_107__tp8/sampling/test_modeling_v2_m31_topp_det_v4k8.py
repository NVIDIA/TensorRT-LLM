# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for TOPP_DET V4K8 (``minimax_m31_topp_renorm_det_v3`` with ``list_path`` / ``wide_hist``): every switch
combination's output must equal v1.2's (``minimax_m31_topp_renorm_det``) bit for bit, including rows whose >= 2^-12
list overflows (unnormalized rows), thresholds on either side of the 2^-12 bucket edge, keep-all rows, mixed
combinations on one workspace and CUDA-graph replays; ``ops/sampling/m31_topp_det`` logs the values it uses and
honours TRTLLM_M31_DISABLE_TOPPDET_V4 / _K8.
"""

import os
import subprocess
import sys

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m31_nvfp4__sm_107__tp8.ops.sampling import (  # noqa: E501
    m31_topp_det as entry,
)

TARGET = (
    "tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m31_nvfp4__sm_107__tp8"
)
D = 200064
EDGE = 2.0**-12
CLASSES = ("randn", "peaked", "ties", "near_uniform", "all_equal", "one_hot", "tiny_tail")
COMBOS = [(True, True), (True, False), (False, True), (False, False)]


def _ops():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7):
        pytest.skip("this TOPP_DET target is sm_107")
    v12 = entry._load_op("minimax_m31_topp_renorm_det")
    v3 = entry._load_op("minimax_m31_topp_renorm_det_v3")
    if v12 is None or v3 is None or "list_path" not in str(v3.default._schema):
        pytest.skip("needs minimax_m31_topp_renorm_det and a V4K8 minimax_m31_topp_renorm_det_v3")
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
    def __init__(self, v12, v3):
        self.v12, self.v3 = v12, v3
        n12 = int(torch.ops.trtllm.minimax_m31_topp_renorm_det_ws_bytes())
        n3 = int(torch.ops.trtllm.minimax_m31_topp_renorm_det_v3_ws_bytes())
        self.ws12 = torch.zeros(8 * n12, dtype=torch.uint8, device="cuda")
        self.ws3 = torch.zeros(8 * n3, dtype=torch.uint8, device="cuda")
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


@pytest.mark.parametrize("combo", COMBOS)
@pytest.mark.parametrize("pdl", [False, True])
@pytest.mark.parametrize("rows", [1, 7, 8])
def test_v4k8_bitwise_v12(rows, pdl, combo):
    r = _Runner(*_ops())
    bad = 0
    for cls in CLASSES:
        for mode in ("p95", "mixed", "with_p1"):
            for seed in range(3):
                probs, tp = _make(cls, rows, 100 * seed + rows), _top_p(mode, rows, seed)
                bad += _bad_rows(r.new(probs, tp, pdl, combo), r.ref(probs, tp))
    assert bad == 0


@pytest.mark.parametrize("listed", [4095, 4096, 4097, 6000])
def test_v4k8_list_overflow_unnormalized(listed):
    """Rows (not summing to 1) with `listed` elements >= 2^-12 and the threshold above the edge: at most 4096 fit the
    list, the rest take the radix fallback; both bitwise v1.2."""
    r = _Runner(*_ops())
    g = torch.Generator(device="cuda").manual_seed(listed)
    probs = torch.rand(8, D, generator=g, device="cuda") * 1e-6
    for i in range(8):
        idx = torch.randperm(D, generator=g, device="cuda")[:listed]
        probs[i, idx] = EDGE * (1.5 + torch.rand(listed, generator=g, device="cuda"))
    tp = torch.full((8,), 0.9, device="cuda")
    ref = r.ref(probs, tp)
    for combo in COMBOS:
        assert _bad_rows(r.new(probs, tp, True, combo), ref) == 0


@pytest.mark.parametrize("side", ["at_edge", "below_edge"])
def test_v4k8_threshold_at_the_edge(side):
    """The kept set ends inside the [2^-12, ...) bucket (list path) or just below 2^-12 (fallback)."""
    r = _Runner(*_ops())
    g = torch.Generator(device="cuda").manual_seed(3)
    probs = torch.rand(8, D, generator=g, device="cuda") * 1e-7
    v = EDGE if side == "at_edge" else EDGE * 0.999
    for i in range(8):
        idx = torch.randperm(D, generator=g, device="cuda")[:3000]
        probs[i, idx] = v
        probs[i, idx[:4]] = 0.05
    probs = probs.contiguous()
    tp = torch.full((8,), 0.6, device="cuda")
    ref = r.ref(probs, tp)
    for combo in COMBOS:
        assert _bad_rows(r.new(probs, tp, True, combo), ref) == 0


def test_v4k8_mixed_combos_one_workspace():
    """64 launches on one workspace cycling the four combinations, keep-all rows and row counts (the pass-0 parity
    buffers and list counters must stay consistent across modes)."""
    r = _Runner(*_ops())
    for i in range(64):
        rows = (1, 7, 8)[i % 3]
        probs = _make(CLASSES[i % len(CLASSES)], rows, 900 + i)
        tp = _top_p(("p95", "mixed", "with_p1")[i % 3], rows, i)
        assert (
            _bad_rows(r.new(probs, tp, bool(i % 2), COMBOS[(i // 2) % 4]), r.ref(probs, tp)) == 0
        ), i


def test_v4k8_graph_replays():
    """48 launches (all combinations) captured back to back on one workspace, replayed with fresh inputs copied in."""
    r = _Runner(*_ops())
    rows = 8
    probs = _make("peaked", rows, 1).clone()
    tp = _top_p("mixed", rows, 1).clone()
    outs = [torch.empty_like(probs) for _ in range(48)]
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for i, o in enumerate(outs):
            r.new(probs, tp, True, COMBOS[i % 4], out=o)
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for i, o in enumerate(outs):
            r.new(probs, tp, True, COMBOS[i % 4], out=o)
    for k, cls in enumerate(("randn", "peaked", "one_hot")):
        probs.copy_(_make(cls, rows, 50 + k))
        tp.copy_(_top_p("with_p1" if k == 2 else "p95", rows, k))
        graph.replay()
        torch.cuda.synchronize()
        ref = r.ref(probs, tp)
        assert all(_bad_rows(o, ref) == 0 for o in outs), cls


def _v4k8_line(env):
    code = f"import importlib; m = importlib.import_module('{TARGET}.ops.sampling.m31_topp_det'); m.topp_det_op()"
    out = subprocess.run(
        [sys.executable, "-c", code],
        env={**os.environ, "TLLM_LOG_LEVEL": "INFO", **env},
        capture_output=True,
        text=True,
        check=True,
    )
    return [ln for ln in (out.stdout + out.stderr).splitlines() if "TOPP_DET V4K8:" in ln]


def test_v4k8_engagement_and_switches():
    """Rule e: one line per process naming the values actually used; each kill switch turns only its item off."""
    _ops()
    base = {"TRTLLM_M31_DISABLE_TOPPDET_V4": "0", "TRTLLM_M31_DISABLE_TOPPDET_K8": "0"}
    on = _v4k8_line(base)
    no_list = _v4k8_line({**base, "TRTLLM_M31_DISABLE_TOPPDET_V4": "1"})
    no_k8 = _v4k8_line({**base, "TRTLLM_M31_DISABLE_TOPPDET_K8": "1"})
    assert len(on) == 1 and "list path on, 8 pass histograms on" in on[0]
    assert len(no_list) == 1 and "list path off, 8 pass histograms on" in no_list[0]
    assert len(no_k8) == 1 and "list path on, 8 pass histograms off" in no_k8[0]
