# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""FC12's glue on its own stream (``Fc12Glue``; CPU only, on the source, CUDA streams and events
stood in by a recorder): each layer's zeroing after the last reader; the operands (FP32 weights
bitwise, NVFP4 latent) made on the glue stream; FC12 joining the glue and launching prezeroed; the
in-order path without the glue stream; the join when a layer raises.
"""

from __future__ import annotations

import contextlib
import types
from types import SimpleNamespace
from typing import Optional, Tuple

import pytest
import torch
from _helix16_wide_cpu import modeling_path, source_objects

pytestmark = pytest.mark.cpu_only

ROWS = 96


class _Cuda:
    """``torch.cuda``'s streams, events and stream context, recording each stream's operations."""

    def __init__(self):
        self.log: list = []
        self.main = _Stream("main", self)
        self.current = self.main

    def Stream(self, device=None):  # noqa: N802 (torch's name)
        return _Stream("glue", self)

    def Event(self):  # noqa: N802
        return _Event(self)

    def current_stream(self, device=None):
        return self.current

    @contextlib.contextmanager
    def stream(self, s):
        before, self.current = self.current, s
        try:
            yield
        finally:
            self.current = before

    def on(self, stream_name):
        return [entry[1:] for entry in self.log if entry[0] == stream_name]


class _Stream:
    def __init__(self, name, cuda):
        self.name, self.cuda = name, cuda

    def wait_event(self, event):
        self.cuda.log.append((self.name, "wait", event.name, event.version))


class _Event:
    def __init__(self, cuda):
        self.cuda, self.name, self.version = cuda, "?", 0

    def record(self, stream=None):
        stream = stream or self.cuda.current
        self.version += 1
        self.cuda.log.append((stream.name, "record", self.name, self.version))


class _Torch(types.ModuleType):
    def __getattr__(self, name):
        return getattr(torch, name)


class _Logged:
    """A tensor's slices whose ``zero_`` is recorded with the current stream."""

    def __init__(self, tensor, cuda):
        self.tensor, self.cuda = tensor, cuda

    def __getitem__(self, index):
        view = self.tensor[index]
        cuda = self.cuda

        class _View:
            def zero_(self):
                cuda.log.append((cuda.current.name, "zero", view.numel()))
                view.zero_()

        return _View()


class _Quantized:
    def __init__(self, name, cuda):
        self.name, self.cuda = name, cuda

    def record_stream(self, stream):
        self.cuda.log.append((self.cuda.current.name, "record_stream", self.name, stream.name))


@pytest.fixture
def ns():
    cuda = _Cuda()
    proxy = _Torch("torch")
    proxy.cuda = cuda
    names = (
        "_MOE_TOP_K", "_MOE_LATENT", "_MOE_CODE_BYTES", "_FC12_COUNTER_WORDS", "FRONT_ROWS",
        "Fc12Glue", "Fc12MoE",
    )  # fmt: skip
    prepared = SimpleNamespace(PreparedMoE=object, JOIN={16: ""})
    ns = {
        "torch": proxy,
        "contextlib": contextlib,
        "Optional": Optional,
        "Tuple": Tuple,
        "module": {"cutedsl.moe.prepared": prepared}.__getitem__,
    }
    source_objects(modeling_path(), names, ns)
    ns["cuda"] = cuda
    return ns


def _glue(ns, rows=128):
    glue = ns["Fc12Glue"](torch.device("cpu"), rows)
    for name in ("begin", "routed", "done", "free"):
        getattr(glue, name).name = name
    glue.buffer = _Logged(glue.buffer, ns["cuda"])
    return glue


def test_each_layer_zeroes_its_rows_after_the_last_reader(ns):
    cuda, glue = ns["cuda"], _glue(ns)
    real = glue.buffer.tensor
    real.fill_(-1)
    glue.zero(1, ROWS)
    words = ns["_FC12_COUNTER_WORDS"] + ROWS * 3584 // 2
    # The step's first MoE layer: after the layer's start.
    assert cuda.on("main") == [("record", "begin", 1)]
    assert cuda.on("glue") == [("wait", "begin", 1), ("zero", words)]
    assert bool((real[:words] == 0).all()) and bool((real[words:] == -1).all())
    assert bool((glue.partial[:ROWS].view(torch.int16) == 0).all())
    # Its LatentRows releases them; the next layer zeroes after that, not after its own start.
    cuda.log.clear()
    glue.release(1, ROWS)
    glue.zero(2, ROWS)
    assert cuda.on("main") == [("record", "free", 1)]
    assert cuda.on("glue") == [("wait", "free", 1), ("zero", words)]
    # No release by the layer before (or another row count): after the layer's start again.
    for layer, rows, released in ((4, ROWS, None), (3, 72, (2, ROWS)), (1, ROWS, (92, ROWS))):
        cuda.log.clear()
        glue.freed = released
        glue.zero(layer, rows)
        assert cuda.on("main") == [("record", "begin", cuda.on("main")[0][2])]
        assert cuda.on("glue")[0][:2] == ("wait", "begin")
        assert glue.freed is None


def test_the_operands_are_the_main_streams_and_made_beside_the_sort(ns):
    cuda, glue = ns["cuda"], _glue(ns)
    weights = torch.rand((ROWS, 16)).to(torch.bfloat16)
    latent = torch.zeros((ROWS, 3584), dtype=torch.bfloat16)
    calls = []

    def quantize(x):
        calls.append((cuda.current.name, x is latent))
        return _Quantized("x4", cuda), _Quantized("sf", cuda)

    x4, sf, scales = glue.operands(SimpleNamespace(quantize=quantize), latent, weights, ROWS)
    assert scales.data_ptr() == glue.scales.data_ptr() and scales.shape == (ROWS, 16)
    assert torch.equal(scales.view(torch.int32), weights.float().view(torch.int32))
    assert calls == [("glue", True)]
    assert cuda.on("main") == [("record", "routed", 1)]
    # No record_stream: the quantized latent's blocks go back to the glue stream's pool.
    assert cuda.on("glue") == [("wait", "routed", 1), ("record", "done", 1)]
    cuda.log.clear()
    glue.join()
    assert cuda.on("main") == [("wait", "done", 1)]
    # The sort's FP32 ones: written by nothing here (zeroing included).
    glue.zero(1, ROWS)
    assert bool((glue.ones == 1).all())


def test_the_operands_after_the_forks_recorded_at_the_routing(ns):
    """M2 ``stream``: ``run_wide`` records ``routed`` right after the routing launch, sorts,
    then calls ``operands(..., recorded=True)``: no second record, and the glue stream waits for
    that event, so for the routing only (not the sort launched after it)."""
    cuda, glue = ns["cuda"], _glue(ns)
    weights = torch.rand((ROWS, 16)).to(torch.bfloat16)
    latent = torch.zeros((ROWS, 3584), dtype=torch.bfloat16)

    def quantize(x):
        return _Quantized("x4", cuda), _Quantized("sf", cuda)

    glue.routed.record(cuda.current_stream())
    first = list(cuda.on("main"))
    glue.operands(SimpleNamespace(quantize=quantize), latent, weights, ROWS, recorded=True)
    assert cuda.on("main") == first and len(first) == 1 and first[0][:2] == ("record", "routed")
    assert [e[:2] for e in cuda.on("glue")] == [("wait", "routed"), ("record", "done")]


class _Experts:
    def __init__(self, cuda, log):
        self.cuda, self.log = cuda, log

    def sort(self, ids, weights):
        self.log.append(("sort", self.cuda.current.name, ids, weights))
        return ("tables",)

    def assemble(self, x4, sf, tables, scales, out):
        return ("operands", x4, sf, tables, scales, out)

    def launch(self, operands, counters):
        self.log.append(("fc12", self.cuda.current.name, operands, counters))

    def run(self, latent, ids, weights, out):
        self.log.append(("fc12 in order", self.cuda.current.name, latent, ids, weights, out))


class _Tail:
    def __init__(self, cuda, log):
        self.cuda, self.log = cuda, log

    def run(self, partial, b, join):
        self.log.append(("LatentRows", self.cuda.current.name, partial, b))
        join()
        self.log.append(("NormRows", self.cuda.current.name))


def _moe(ns, glue, log):
    cuda = ns["cuda"]
    moe = object.__new__(ns["Fc12MoE"])
    moe.experts, moe.glue, moe.layer, moe.layer_idx = _Experts(cuda, log), glue, 7, 7
    moe.partial = glue.partial if glue is not None else torch.zeros((128, 3584))
    moe.prefetch = SimpleNamespace(fork=lambda point, layer, b: log.append(("prefetch", point)))
    # Levers off: no tuned tail (T1), no routing-written FP32 weights (G1 b).
    moe.tuned, moe.feeds, moe.route32, moe.route32_rows = None, {}, None, frozenset()
    return moe


def test_fc12_joins_the_glue_and_launches_prezeroed(ns):
    cuda, glue, log = ns["cuda"], _glue(ns), []
    moe = _moe(ns, glue, log)
    ids, weights, latent = torch.zeros((ROWS, 16), dtype=torch.int32), torch.zeros(1), None
    glued = ("x4", "sf", glue.scales[:ROWS])
    joined = []
    moe.experts_and_tail(
        _Tail(cuda, log), latent, ids, weights, ROWS, lambda: joined.append(len(log)), glued
    )
    sort, fc12, prefetch, latent_rows, norm_rows = log
    # The sort's routing weights: the glue's FP32 ones (their shape), not the BF16 weights.
    assert sort[:3] == ("sort", "main", ids)
    assert sort[3].data_ptr() == glue.ones.data_ptr() and sort[3].shape == (ROWS, 16)
    assert sort[3].dtype == torch.float32 and bool((sort[3] == 1).all())
    assert fc12[:2] == ("fc12", "main") and fc12[3] is glue.counters
    _, x4, sf, tables, scales, out = fc12[2]
    assert (x4, sf, tables, scales) == ("x4", "sf", ("tables",), glued[2])
    assert out.data_ptr() == glue.partial.data_ptr() and out.shape == (ROWS, 3584)
    assert prefetch == ("prefetch", "next_attention")
    assert latent_rows[:2] == ("LatentRows", "main") and latent_rows[2] is out
    # Between FC12's sort and its launch the main stream waits for the glue stream; after the
    # tail's LatentRows it releases the partial, then joins the shared expert, then NormRows.
    assert cuda.on("main") == [("wait", "done", 0), ("record", "free", 1)]
    assert joined == [4] and glue.freed == (7, ROWS) and norm_rows == ("NormRows", "main")


def test_without_the_glue_stream_fc12_runs_in_order(ns):
    cuda, log = ns["cuda"], []
    moe = _moe(ns, None, log)
    ids, weights, latent = object(), object(), object()
    moe.experts_and_tail(_Tail(cuda, log), latent, ids, weights, ROWS, lambda: None, None)
    first = log[0]
    assert first[:5] == ("fc12 in order", "main", latent, ids, weights)
    assert first[5].data_ptr() == moe.partial.data_ptr() and first[5].shape[0] == ROWS
    assert [entry[0] for entry in log] == ["fc12 in order", "prefetch", "LatentRows", "NormRows"]
    assert cuda.log == []


def test_a_layer_zeroes_first_and_joins_the_glue_when_it_raises(ns):
    cuda, glue = ns["cuda"], _glue(ns)
    moe = _moe(ns, glue, [])
    moe.wide = None
    order = []

    def run_front(x, b):
        order.append(("front", list(cuda.log)))
        raise RuntimeError("boom")

    moe.run_front = run_front
    x = torch.zeros((48, 8))
    with pytest.raises(RuntimeError, match="boom"):
        moe.run(x, None, consume=lambda out: out)
    # The zeroing came first (at this layer's start), the join after the error.
    assert order[0][1][:2] == [("main", "record", "begin", 1), ("glue", "wait", "begin", 1)]
    assert cuda.on("glue")[-1] == ("record", "done", 1)
    assert cuda.on("main")[-1] == ("wait", "done", 1)
    assert glue.freed is None
