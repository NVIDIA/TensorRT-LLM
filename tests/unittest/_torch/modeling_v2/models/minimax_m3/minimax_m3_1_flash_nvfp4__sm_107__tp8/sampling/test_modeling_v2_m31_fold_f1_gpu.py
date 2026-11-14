# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""FOLD_F1 on a GPU: ``minimax_m31_topp_renorm_det_v3_rows`` writes exactly what ``minimax_m31_topp_renorm_det_v3``
and then the spec sampler's store (``dst[slot_ids, :k, :d] = out.view(n, k, d)``) write: the same output, and the
same slot buffer (every other element untouched), for every V4K8 combination with and without PDL, the row classes
and top_p modes of the V4K8 test, request / row layouts (1, 7), (1, 8), (2, 4), (8, 1), and from a CUDA graph. The
op rejects the layouts it does not take; ``ops/sampling/m31_topp_det`` takes an armed row store only where it fits,
reports it once (``rows_stored``) and otherwise renorms as before."""

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_1_flash_nvfp4__sm_107__tp8.ops.sampling import (  # noqa: E501
    m31_topp_det as entry,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.native import family_ops

D = 200064
D_PAD = D + 64
SLOTS, KMAX = 9, 8
SENTINEL = -3.0
CLASSES = ("randn", "peaked", "ties", "near_uniform", "one_hot", "tiny_tail")
COMBOS = [(True, True), (True, False), (False, True), (False, False)]
LAYOUTS = [(1, 7), (1, 8), (2, 4), (8, 1)]


def _ops():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7):
        pytest.skip("this TOPP_DET target is sm_107")
    v3 = entry._load_op("minimax_m31_topp_renorm_det_v3")
    rows_op = entry._load_op("minimax_m31_topp_renorm_det_v3_rows")
    assert v3 is not None and rows_op is not None and "list_path" in str(v3.default._schema), (
        "this build has no V4K8 minimax_m31_topp_renorm_det_v3 / minimax_m31_topp_renorm_det_v3_rows"
    )
    return v3, rows_op


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
    elif cls == "one_hot":
        logits = torch.full((rows, D), -1e4, device="cuda")
        logits[torch.arange(rows), torch.randint(0, D, (rows,), generator=g, device="cuda")] = 0
    else:
        logits = torch.randn(rows, D, generator=g, device="cuda") * 30
    return torch.softmax(logits, -1).contiguous()


def _top_p(mode, rows, seed):
    g = torch.Generator(device="cuda").manual_seed(seed + 7)
    if mode == "p95":
        return torch.full((rows,), 0.95, device="cuda")
    if mode == "mixed":
        return (0.05 + 0.95 * torch.rand(rows, generator=g, device="cuda")).contiguous()
    tp = torch.full((rows,), 0.95, device="cuda")
    tp[seed % 2 :: 2] = 1.0
    return tp


class _Runner:
    def __init__(self, v3, rows_op):
        self.v3, self.rows_op = v3, rows_op
        n3 = int(family_ops().minimax_m31_topp_renorm_det_v3_ws_bytes())
        self.ws_ref = torch.zeros(8 * n3, dtype=torch.uint8, device="cuda")
        self.ws_new = torch.zeros(8 * n3, dtype=torch.uint8, device="cuda")
        self.sms = torch.cuda.get_device_properties(0).multi_processor_count

    def ref(self, probs, tp, pdl, combo, dst, slot_ids, k):
        out = torch.empty_like(probs)
        g = entry.ctas_per_row(probs.shape[0], self.sms)
        self.v3(probs, tp, self.ws_ref, out, g, pdl, True, combo[0], combo[1])
        dst[slot_ids, :k, :D] = out.view(slot_ids.numel(), k, D)
        return out

    def new(self, probs, tp, pdl, combo, dst, slot_ids, k, out=None):
        out = torch.empty_like(probs) if out is None else out
        g = entry.ctas_per_row(probs.shape[0], self.sms)
        self.rows_op(
            probs, tp, self.ws_new, out, g, pdl, True, combo[0], combo[1], dst, slot_ids, k
        )
        return out


def _slots(n, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    return torch.randperm(SLOTS, generator=g, device="cuda")[:n].contiguous()


def _dst():
    return torch.full((SLOTS, KMAX, D_PAD), SENTINEL, device="cuda")


def _same(a, b):
    return torch.equal(a.view(torch.int32), b.view(torch.int32))


@pytest.mark.parametrize("combo", COMBOS)
@pytest.mark.parametrize("pdl", [False, True])
@pytest.mark.parametrize("layout", LAYOUTS, ids=[f"n{n}k{k}" for n, k in LAYOUTS])
def test_rows_op_is_v3_then_the_store(layout, pdl, combo):
    r = _Runner(*_ops())
    n, k = layout
    for c, cls in enumerate(CLASSES):
        for m, mode in enumerate(("p95", "mixed", "with_p1")):
            seed = 1000 * c + 100 * m + 10 * n + k
            probs, tp = _make(cls, n * k, seed), _top_p(mode, n * k, seed)
            slot_ids = _slots(n, seed)
            dst_ref, dst_new = _dst(), _dst()
            ref = r.ref(probs, tp, pdl, combo, dst_ref, slot_ids, k)
            got = r.new(probs, tp, pdl, combo, dst_new, slot_ids, k)
            torch.cuda.synchronize()
            assert _same(got, ref), (cls, mode)
            assert _same(dst_new, dst_ref), (cls, mode)


def test_rows_op_in_a_cuda_graph():
    r = _Runner(*_ops())
    n, k = 1, 7
    probs, tp = _make("peaked", n * k, 1).clone(), _top_p("mixed", n * k, 1).clone()
    slot_ids = _slots(n, 2).clone()
    dst = _dst()
    out = torch.empty_like(probs)
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        r.new(probs, tp, True, COMBOS[0], dst, slot_ids, k, out=out)
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        r.new(probs, tp, True, COMBOS[0], dst, slot_ids, k, out=out)
    for step, cls in enumerate(("randn", "peaked", "one_hot", "tiny_tail")):
        probs.copy_(_make(cls, n * k, 50 + step))
        tp.copy_(_top_p("with_p1" if step == 2 else "p95", n * k, step))
        slot_ids.copy_(_slots(n, 60 + step))
        dst_ref = dst.clone()
        ref = r.ref(probs, tp, True, COMBOS[0], dst_ref, slot_ids, k)
        graph.replay()
        torch.cuda.synchronize()
        assert _same(out, ref) and _same(dst, dst_ref), cls


def test_rows_op_rejects_other_layouts():
    r = _Runner(*_ops())
    n, k = 1, 7
    probs, tp = _make("randn", n * k, 3), _top_p("p95", n * k, 3)
    slot_ids = _slots(n, 4)
    bad = {
        "unaligned row stride": (torch.zeros(SLOTS, KMAX, D + 2, device="cuda"), slot_ids, k),
        "int32 slot ids": (_dst(), slot_ids.to(torch.int32), k),
        "rows != slots * k": (_dst(), slot_ids, 6),
        "bf16 dst": (_dst().bfloat16(), slot_ids, k),
    }
    for name, (dst, ids, kk) in bad.items():
        with pytest.raises(RuntimeError, match="topp_renorm_det_v3_rows"):
            r.new(probs, tp, True, COMBOS[0], dst, ids, kk)


def test_the_entry_takes_an_armed_store_once():
    _ops()
    n, k = 1, 7
    probs, tp = _make("peaked", n * k, 5), _top_p("mixed", n * k, 5)
    slot_ids = _slots(n, 6)
    dst = _dst()
    entry.arm_row_store(dst, slot_ids, k)
    got = entry.top_p_renorm_det(probs, tp)
    assert got is not None
    stored = got.view(n, k, D)
    assert entry.rows_stored(dst, slot_ids, stored)
    assert not entry.rows_stored(dst, slot_ids, stored)  # one-shot
    want = _dst()
    want[slot_ids, :k, :D] = stored
    torch.cuda.synchronize()
    assert _same(dst, want)
    # The arm is consumed by that call: the next renorm stores nothing and reports nothing.
    again = entry.top_p_renorm_det(probs, tp)
    assert _same(again, got) and not entry.rows_stored(dst, slot_ids, again.view(n, k, D))


def test_the_entry_renorms_as_before_when_the_store_does_not_fit():
    _ops()
    n, k = 1, 7
    probs, tp = _make("randn", n * k, 8), _top_p("p95", n * k, 8)
    slot_ids = _slots(n, 9)
    for dst, ids in (
        (torch.full((SLOTS, KMAX, D + 2), SENTINEL, device="cuda"), slot_ids),
        (_dst(), slot_ids.to(torch.int32)),
    ):
        before = dst.clone()
        plain = entry.top_p_renorm_det(probs, tp)
        entry.arm_row_store(dst, ids, k)
        got = entry.top_p_renorm_det(probs, tp)
        torch.cuda.synchronize()
        assert _same(got, plain)
        assert not entry.rows_stored(dst, ids, got.view(n, k, D))
        assert _same(dst, before)
