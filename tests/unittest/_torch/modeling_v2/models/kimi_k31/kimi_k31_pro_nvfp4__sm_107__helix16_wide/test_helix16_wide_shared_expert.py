# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The shared expert above 64 rows (CPU only, on the sources, streams stood in by a recorder): a
layer that raises still joins the side stream (``run_wide`` and ``run_front``); ``shared_halves``
refuses a fused front.
"""

from __future__ import annotations

import contextlib
import types
from types import SimpleNamespace

import pytest
import torch
from _helix16_wide_cpu import modeling_path, source_objects

pytestmark = pytest.mark.cpu_only


class _Cuda:
    def __init__(self):
        self.log: list = []
        self.main = _Stream("main", self)
        self.current = self.main

    def current_stream(self, device=None):
        return self.current

    @contextlib.contextmanager
    def stream(self, s):
        before, self.current = self.current, s
        try:
            yield
        finally:
            self.current = before

    def on(self, name):
        return [entry[1:] for entry in self.log if entry[0] == name]

    def note(self, *what):
        self.log.append((self.current.name, *what))


class _Stream:
    def __init__(self, name, cuda):
        self.name, self.cuda = name, cuda

    def wait_event(self, event):
        self.cuda.log.append((self.name, "wait", event.name))


class _Event:
    def __init__(self, name, cuda):
        self.name, self.cuda = name, cuda

    def record(self, stream=None):
        self.cuda.log.append(((stream or self.cuda.current).name, "record", self.name))


class _Kernel:
    def __init__(self, cuda, name, block):
        self.cuda, self.name, self.block = cuda, name, block

    def run(self, x, out):
        self.cuda.note(self.name, self.block, x.shape[0], x.data_ptr(), out.data_ptr())


def _front(cuda, block, rows, storage, fused_situ):
    """A row block's ``SharedFront`` stood in: its plans' views, its split-K kernels (recording),
    and ``shared_expert`` as the vendored method runs its plain path."""
    views = tuple(t[:rows] for t in storage)
    front = SimpleNamespace(native_shared=None, device="cpu")

    def situ(*args):
        cuda.note("situ", block, rows, args)

    front.plans = {rows: (views, None, None, situ, ("args",), None, None)}
    front.gate_up, front.down = _Kernel(cuda, "gate_up", block), _Kernel(cuda, "down", block)
    front.gate_situ = None
    if fused_situ:
        front.gate_situ = SimpleNamespace(kernels={rows}, run=_Kernel(cuda, "gate_situ", block).run)

    def shared_expert(x, tail):
        views, _, _, situ_, args, _, code = front.plans[x.shape[0]]
        s = "s"
        if front.gate_situ is not None and x.shape[0] in front.gate_situ.kernels:
            front.gate_situ.run(x, views[4])
        else:
            front.gate_up.run(x, views[1])
            situ_(*args, s)
        shared = front.down.run(views[4], views[5])
        tail.push(shared)
        return shared, None

    front.shared_expert = shared_expert
    return front


@pytest.fixture
def ns():
    cuda = _Cuda()

    class _Torch(types.ModuleType):
        def __getattr__(self, name):
            return getattr(torch, name)

    proxy = _Torch("torch")
    proxy.cuda = cuda
    host = SimpleNamespace(stream=lambda device: "s")
    modules = {
        "cutedsl.moe.prepared": SimpleNamespace(PreparedMoE=object, JOIN={16: ""}),
        "cutedsl.common.host": host,
    }
    ns = {"torch": proxy, "contextlib": contextlib, "module": modules.__getitem__}
    names = ("FRONT_ROWS", "_NO_PUSH", "shared_halves", "Fc12MoE")
    ns["SimpleNamespace"] = SimpleNamespace
    source_objects(modeling_path(), names, ns)
    ns["cuda"] = cuda
    return ns


def _storage(cap=128):
    """``MoEScratch.front``'s seven buffers (logits, gate/up, ids, weights, h, shared rows,
    latent)."""
    shapes = ((cap, 896), (cap, 768), (cap, 16), (cap, 16), (cap, 384), (cap, 7168), (cap, 3584))
    return tuple(torch.zeros(s) for s in shapes)


def _moe(ns, shared, rows, storage, fail=False):
    cuda = ns["cuda"]
    cap = storage[0].shape[0]
    first = _front(cuda, "first", 64, storage, False)
    rest_storage = tuple(t[64:] for t in storage)
    rest = _front(cuda, "rest", rows - 64, rest_storage, rows - 64 <= 8)

    class _Tail:
        def push(self, shared_rows):
            cuda.note("push", shared_rows.shape[0], shared_rows.data_ptr())

        def run(self, partial, b, join):
            cuda.note("LatentRows")
            join()
            cuda.note("NormRows")

    class _Once:
        def route(self, x):
            cuda.note("front+route", x.shape[0])
            return storage[2][: x.shape[0]], storage[3][: x.shape[0]], storage[6][: x.shape[0]]

    class _Experts:
        def run(self, latent, ids, weights, out):
            if fail:
                raise RuntimeError("fc12 failed")
            cuda.note("fc12", out.shape[0])

    moe = object.__new__(ns["Fc12MoE"])
    moe.state = SimpleNamespace(
        device="cpu", stream=_Stream("side", cuda), begin=_Event("begin", cuda),
        shared_done=_Event("shared_done", cuda), front=storage,
    )  # fmt: skip
    moe.wide = (first, rest, _Tail(), _Once())
    moe.glue, moe.fronted = None, _Event("fronted", cuda)
    moe.experts, moe.partial, moe.prefetch = _Experts(), torch.zeros((cap, 8)), None
    moe.layer = moe.layer_idx = 5
    # The K-split front, the shared expert blocks, the wide tail, no routing-written FP32 weights.
    moe.options = SimpleNamespace(front="once")
    moe.tuned, moe.feeds, moe.route32, moe.route32_rows = None, {}, None, frozenset()
    return moe


def test_a_failing_layer_still_joins_the_side_stream(ns):
    cuda = ns["cuda"]
    moe = _moe(ns, "split", 96, _storage(), fail=True)
    with pytest.raises(RuntimeError, match="fc12 failed"):
        moe.run_wide(torch.zeros((96, 8)), 96)
    main = cuda.on("main")
    assert main[-1] == ("wait", "shared_done")
    assert cuda.on("side")[-1] == ("record", "shared_done")


def test_run_front_joins_the_side_stream_when_it_raises(ns):
    """Up to 64 rows (``run_front``): a failure past the side stream's fork (here FC12)
    still records ``shared_done`` there and joins it on the main stream before it propagates."""
    cuda = ns["cuda"]
    moe = _moe(ns, "split", 72, _storage(), fail=True)
    storage = moe.state.front

    def route(x):
        cuda.note("route", x.shape[0])
        return storage[2][: x.shape[0]], storage[3][: x.shape[0]], storage[6][: x.shape[0]]

    def shared_expert(x, tail):
        cuda.note("shared", x.shape[0])
        return None, None

    front = dict(merged_batch=lambda b: False, route=route, shared_expert=shared_expert)
    moe.front = SimpleNamespace(**front)
    moe.state.gate_done, moe.rs, moe.join, moe.tail = _Event("gate_done", cuda), None, {}, None
    with pytest.raises(RuntimeError, match="fc12 failed"):
        moe.run_front(torch.zeros((48, 8)), 48)
    assert cuda.on("side") == [("wait", "begin"), ("shared", 48), ("record", "shared_done"),
                               ("record", "shared_done")]  # fmt: skip
    assert cuda.on("main")[-1] == ("wait", "shared_done")
    assert ("route", 48) in cuda.on("main")


def test_halves_refuse_a_fused_front():
    ns = {"module": {"cutedsl.common.host": SimpleNamespace()}.__getitem__}
    source_objects(modeling_path(), ("shared_halves",), ns)
    plans = {64: ((), None, None, None, (), None, "code")}
    fused = SimpleNamespace(plans=plans, native_shared=None)
    with pytest.raises(RuntimeError, match="plain split-K"):
        ns["shared_halves"](fused, torch.zeros((64, 1)))
