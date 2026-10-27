# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for TOPP_DET v3 (``torch.ops.trtllm.minimax_m31_topp_renorm_det_v3``): its output must equal v1.2's
(``minimax_m31_topp_renorm_det``) bit for bit on every row (top_p < 1 and the top_p >= 1 keep-all rows), with and
without the programmatic dependent launch, across back-to-back launches and CUDA-graph replays that reuse one
workspace; ``ops/sampling/m31_topp_det`` takes v3 when it is built unless TRTLLM_M31_DISABLE_TOPPDET_V3 is set.
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
CLASSES = ("randn", "peaked", "ties", "near_uniform", "all_equal", "one_hot", "tiny_tail")


def _ops():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7):
        pytest.skip("this TOPP_DET target is sm_107")
    # The entry loads only the op it engages; load both (in-tree or from tensorrt_llm/libs) for the comparison.
    v12 = entry._load_op("minimax_m31_topp_renorm_det")
    v3 = entry._load_op("minimax_m31_topp_renorm_det_v3")
    if v12 is None or v3 is None:
        pytest.skip("needs both minimax_m31_topp_renorm_det and minimax_m31_topp_renorm_det_v3")
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
    """v1.2 and v3 on separate zeroed workspaces sized for 8 rows."""

    def __init__(self, v12, v3):
        self.v12, self.v3 = v12, v3
        self.ws12 = torch.zeros(
            8 * int(torch.ops.trtllm.minimax_m31_topp_renorm_det_ws_bytes()),
            dtype=torch.uint8,
            device="cuda",
        )
        self.ws3 = torch.zeros(
            8 * int(torch.ops.trtllm.minimax_m31_topp_renorm_det_v3_ws_bytes()),
            dtype=torch.uint8,
            device="cuda",
        )
        self.sms = torch.cuda.get_device_properties(0).multi_processor_count

    def g(self, rows):
        return entry.ctas_per_row(rows, self.sms)

    def ref(self, probs, tp):
        out = torch.empty_like(probs)
        self.v12(probs, tp, self.ws12, out, self.g(probs.shape[0]))
        return out

    def new(self, probs, tp, pdl, out=None, cand=True):
        out = torch.empty_like(probs) if out is None else out
        self.v3(probs, tp, self.ws3, out, self.g(probs.shape[0]), pdl, cand)
        return out


def _bad_rows(a, b):
    return int((a.view(torch.int32) != b.view(torch.int32)).any(1).sum())


@pytest.mark.parametrize("cand", [False, True])
@pytest.mark.parametrize("pdl", [False, True])
@pytest.mark.parametrize("rows", [1, 7, 8])
def test_v3_bitwise_v12(rows, pdl, cand):
    r = _Runner(*_ops())
    bad = 0
    for cls in CLASSES:
        for mode in ("p95", "mixed", "with_p1"):
            for seed in range(4):
                probs, tp = _make(cls, rows, 100 * seed + rows), _top_p(mode, rows, seed)
                bad += _bad_rows(r.new(probs, tp, pdl, cand=cand), r.ref(probs, tp))
    assert bad == 0


def test_v3_back_to_back_graph_replays():
    """100 PDL launches captured back to back on one workspace, replayed twice with fresh inputs copied in."""
    r = _Runner(*_ops())
    rows = 7
    probs = _make("peaked", rows, 1).clone()
    tp = _top_p("mixed", rows, 1).clone()
    outs = [torch.empty_like(probs) for _ in range(100)]
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        r.new(probs, tp, True, outs[0])
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for o in outs:
            r.new(probs, tp, True, o)
    for k, cls in enumerate(("randn", "near_uniform")):
        probs.copy_(_make(cls, rows, 50 + k))
        tp.copy_(_top_p("with_p1", rows, k))
        graph.replay()
        torch.cuda.synchronize()
        ref = r.ref(probs, tp)
        assert sum(_bad_rows(o, ref) for o in outs) == 0


def test_v3_alternating_grid_one_workspace():
    """60 launches on one workspace alternating rows (1 / 7 / 8), ctas_per_row (16 / 8 / 4), PDL and the candidate
    list: the barrier counters and the list must stay consistent; every output bitwise v1.2's at the same rows."""
    r = _Runner(*_ops())
    bad = 0
    for i in range(60):
        rows = (1, 7, 8)[i % 3]
        g = (16, 8, 4)[(i // 3) % 3]
        probs = _make(CLASSES[i % len(CLASSES)], rows, 300 + i)
        tp = _top_p(("p95", "mixed", "with_p1")[i % 3], rows, i)
        out = torch.empty_like(probs)
        r.v3(probs, tp, r.ws3, out, g, bool(i % 2), bool((i // 2) % 2))
        bad += _bad_rows(out, r.ref(probs, tp))
    assert bad == 0


def _cap_row(members, seed):
    """A probability row whose top-p cut lands mid-way into a pass-0 radix bucket (the top 11 key bits) of exactly
    ``members`` elements: 10 head elements of 0.05, ``members`` values in [2^-14, 1.15 * 2^-14) (one bucket), and a
    tiny tail in other buckets; returns (row, top_p)."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    row = torch.full((D,), 1.0e-6, device="cuda")
    pos = torch.randperm(D, generator=g, device="cuda")
    row[pos[:10]] = 0.05
    c = 2.0**-14
    row[pos[10 : 10 + members]] = c * (1 + 0.15 * torch.arange(members, device="cuda") / members)
    bucket_mass = float(row[pos[10 : 10 + members]].sum())
    return row, 0.5 + 0.5 * bucket_mass


def _cut_bucket_size(row, top_p):
    vals, _ = torch.sort(row, descending=True)
    cut = int(
        torch.searchsorted(
            torch.cumsum(vals.double(), 0), torch.tensor(top_p, dtype=torch.float64, device="cuda")
        )
    )
    keys = row.view(torch.int32) >> 21
    return int((keys == (vals[cut : cut + 1].view(torch.int32) >> 21)).sum())


@pytest.mark.parametrize("members", [4095, 4096, 4097])
def test_t3_candidate_cap_boundary(members):
    """The candidate list holds a pass-0 bucket of at most 4096: rows whose cut bucket has 4095 / 4096 (list) and 4097
    (fallback to the full passes) members, mixed with randn rows, must all be bitwise v1.2's, list on and off."""
    r = _Runner(*_ops())
    rows = 8
    probs = _make("randn", rows, 900 + members).clone()
    tp = _top_p("p95", rows, members).clone()
    for i in (0, 3, 7):
        row, p = _cap_row(members, 10 * members + i)
        assert _cut_bucket_size(row, p) == members  # the construction's precondition
        probs[i], tp[i] = row, p
    ref = r.ref(probs, tp)
    for pdl in (False, True):
        for cand in (False, True):
            assert _bad_rows(r.new(probs, tp, pdl, cand=cand), ref) == 0


def _t3_line(env):
    code = f"import importlib; m = importlib.import_module('{TARGET}.ops.sampling.m31_topp_det'); m.topp_det_op()"
    out = subprocess.run(
        [sys.executable, "-c", code],
        env={**os.environ, "TLLM_LOG_LEVEL": "INFO", **env},
        capture_output=True,
        text=True,
        check=True,
    )
    lines = [
        ln for ln in (out.stdout + out.stderr).splitlines() if "TOPP_DET v3 T3 candidate-list" in ln
    ]
    return lines


def test_t3_engagement_line():
    """Rule e: one line per process naming the value actually used."""
    _ops()
    on = _t3_line({"TRTLLM_M31_DISABLE_TOPPDET_T3": "0"})
    off = _t3_line({"TRTLLM_M31_DISABLE_TOPPDET_T3": "1"})
    assert len(on) == 1 and "candidate-list fused passes: on (cap 4096)" in on[0]
    assert len(off) == 1 and "candidate-list fused passes: off (cap 4096)" in off[0]


def _engaged(env):
    code = (
        f"import importlib; m = importlib.import_module('{TARGET}.ops.sampling.m31_topp_det'); "
        "m.topp_det_op(); print('V3' if m.is_v3() else 'V12')"
    )
    out = subprocess.run(
        [sys.executable, "-c", code],
        env={**os.environ, **env},
        capture_output=True,
        text=True,
        check=True,
    )
    return out.stdout.strip().splitlines()[-1]


def test_switch_precedence():
    _ops()  # both ops present in this process: the subprocesses see the same build / libs
    assert _engaged({"TRTLLM_M31_DISABLE_TOPPDET_V3": "0"}) == "V3"
    assert _engaged({"TRTLLM_M31_DISABLE_TOPPDET_V3": "1"}) == "V12"
