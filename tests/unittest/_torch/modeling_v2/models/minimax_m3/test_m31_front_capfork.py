# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""FRONT_CAPFORK (m31_capture_fork, default on in the sm_107 tp8 target): the Eagle3 capture copy of a decode
capture layer forked onto a capture stream and joined after the decoder stack. The captured bytes and what the drafter
reads from them are bit for bit the in-line capture's, eager and under CUDA-graph capture, also when the forked copy
is slow (a sleep on the capture stream) and the main stream keeps writing new residuals behind it.

The served sm_107 targets run one-model DSpark: their capture hook is M3DSparkWorker.capture_hidden_states (dspark.py)
over DFlashSpecMetadata's capture, not modeling.py's _eagle3_capture. The test_dspark_* cases drive that worker class:
it forks the plain decode capture, its bytes and the DSpark drafter's read (get_hidden_states) are bit for bit the
in-line capture's (eager and graph), and the kill switch restores the in-line copy."""

import importlib
import inspect

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3 import (
    m31_capture_fork as capfork,
)
from tensorrt_llm._torch.speculative.dflash import DFlashSpecMetadata
from tensorrt_llm._torch.speculative.eagle3 import Eagle3OneModelSpecMetadata
from tensorrt_llm._utils import get_sm_version

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or get_sm_version() != 107,
    reason="the capture fork of the MiniMax-M3.1-Flash tp8 target runs on sm_107 (Rubin) only",
)

H = 6144
LAYERS = (3, 17, 31, 45)
ROWS, CAPACITY = 8, 16
SLOW = 2_000_000  # cycles slept on the capture stream before a forked copy (~1 ms)


class _Spec:
    """The in-tree capture (Eagle3OneModelSpecMetadata.maybe_capture_hidden_states -> trtllm::inplace_slice_copy) on
    the fields it reads; ``slow`` sleeps on the stream the copy is issued on first."""

    def __init__(self, slow: int = 0):
        self.layers_to_capture = LAYERS
        self.hidden_size = H
        self.hidden_states = torch.empty(
            CAPACITY, H * len(LAYERS), dtype=torch.bfloat16, device="cuda"
        )
        self.num_capture_tokens = ROWS
        self.num_tokens = ROWS
        self.slow = slow

    def maybe_capture_hidden_states(self, layer_id, hidden_states, residual=None):
        if self.slow:
            torch.cuda._sleep(self.slow)
        Eagle3OneModelSpecMetadata.maybe_capture_hidden_states(
            self, layer_id, hidden_states, residual
        )


def _inputs(seed: int):
    g = torch.Generator(device="cuda").manual_seed(seed)
    xs = [
        torch.randn(ROWS, H, generator=g, device="cuda").bfloat16() for _ in range(len(LAYERS) + 1)
    ]
    w = torch.randn(H * len(LAYERS), 256, generator=g, device="cuda").bfloat16()
    return xs, w


def _stack(spec: _Spec, xs, w, fork: bool):
    """Four capture layers: each boundary writes a FRESH residual (as the multicast all-reduce does), the capture reads
    it, the main stream then overwrites the layer state; after the stack the drafter reads the capture buffer."""
    h = xs[0]
    outs = []
    for k, layer in enumerate(LAYERS):
        torch.cuda._sleep(200_000)  # a slow boundary: the forked copy must wait for it
        residual = h * 1.0009765625 + xs[k + 1]
        if fork:
            capfork.fork_capture(spec, layer, residual, "ut")
        else:
            spec.maybe_capture_hidden_states(layer, residual, None)
        h = residual * 0.5 - xs[k + 1]
        outs.append(h)
    if fork:
        assert capfork.pending() == len(LAYERS)
        assert capfork.join() == len(LAYERS)
    assert capfork.pending() == 0
    drafter_in = spec.hidden_states[:ROWS, : H * len(LAYERS)]
    return drafter_in.clone(), (drafter_in.float() @ w.float()), outs


def _same(a: torch.Tensor, b: torch.Tensor) -> bool:
    return torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))


def _expected(xs):
    h = xs[0]
    cols = []
    for k in range(len(LAYERS)):
        residual = h * 1.0009765625 + xs[k + 1]
        cols.append(residual)
        h = residual * 0.5 - xs[k + 1]
    return torch.cat(cols, dim=1)


@pytest.mark.parametrize("slow", [0, SLOW])
def test_eager_on_equals_off(slow: int) -> None:
    xs, w = _inputs(11)
    off, on = _Spec(), _Spec(slow)
    off.hidden_states.view(torch.int16).fill_(-1)
    on.hidden_states.view(torch.int16).fill_(-1)
    cap_off, fc_off, h_off = _stack(off, xs, w, fork=False)
    cap_on, fc_on, h_on = _stack(on, xs, w, fork=True)
    torch.cuda.synchronize()
    assert _same(cap_on, cap_off), "captured bytes differ ON vs OFF"
    assert _same(cap_on, _expected(xs)), "the capture is not the boundary's residual"
    assert _same(fc_on, fc_off), "the drafter input (fc of the captured rows) differs"
    assert all(_same(a, b) for a, b in zip(h_on, h_off))
    assert _same(on.hidden_states[ROWS:], off.hidden_states[ROWS:]), (
        "rows past num_capture_tokens were touched"
    )


@pytest.mark.parametrize("slow", [0, SLOW])
def test_cuda_graph_on_equals_off(slow: int) -> None:
    xs, w = _inputs(12)
    static = [x.clone() for x in xs]
    graphs = {}
    for name, fork in (("off", False), ("on", True)):
        spec = _Spec(slow if fork else 0)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            _stack(
                spec, static, w, fork
            )  # eager warm-up (creates the capture stream before the capture)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = _stack(spec, static, w, fork)
        graphs[name] = (graph, spec, out)
    for seed in (21, 22, 23):
        fresh, _ = _inputs(seed)
        for x, y in zip(static, fresh):
            x.copy_(y)
        got = {}
        for name, (graph, spec, out) in graphs.items():
            spec.hidden_states.view(torch.int16).fill_(-1)  # a stale or late copy cannot pass
            graph.replay()
            torch.cuda.synchronize()
            got[name] = (
                spec.hidden_states.clone(),
                out[0].clone(),
                out[1].clone(),
                [h.clone() for h in out[2]],
            )
        on, off = got["on"], got["off"]
        assert _same(on[0], off[0]), (
            f"seed {seed}: capture buffer differs ON vs OFF under graph replay"
        )
        assert _same(on[1], _expected(fresh)), (
            f"seed {seed}: the replayed capture is not this replay's residual"
        )
        assert _same(on[2], off[2]), (
            f"seed {seed}: the drafter input differs ON vs OFF under graph replay"
        )
        assert all(_same(a, b) for a, b in zip(on[3], off[3]))


def test_join_without_pending_is_a_noop() -> None:
    assert capfork.pending() == 0
    assert capfork.join() == 0


@pytest.mark.parametrize("target", ["minimax_m3_1_flash_nvfp4__sm_107__tp8"])
def test_target_wiring(monkeypatch: pytest.MonkeyPatch, target: str) -> None:
    """The target's _eagle3_capture forks only the plain capture of a decode capture layer (no capture norm, no
    residual addend, not the final hidden state), the kill switch restores the in-line copy, and the target forward /
    the early-fc fork join before reading the capture buffer."""
    import importlib

    mod = importlib.import_module(
        f"tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.{target}.modeling"
    )
    assert mod.FRONT_CAPFORK == (not mod.m31_disabled("FRONT_CAPFORK"))
    xs, _ = _inputs(31)
    spec = _Spec()
    monkeypatch.setattr(mod, "FRONT_CAPFORK", True)
    mod._eagle3_capture(spec, LAYERS[0], xs[0], None, None)
    assert capfork.pending() == 1
    mod._eagle3_capture(spec, LAYERS[1], xs[1], xs[2], None)  # with a residual addend: in line
    mod._eagle3_capture(
        spec, mod.FINAL_HIDDEN_LAYER_ID, xs[3], None, None
    )  # the final hidden state: in line
    assert capfork.pending() == 1
    assert capfork.join() == 1
    monkeypatch.setattr(mod, "FRONT_CAPFORK", False)
    mod._eagle3_capture(spec, LAYERS[2], xs[4], None, None)
    assert capfork.pending() == 0
    torch.cuda.synchronize()
    assert _same(spec.hidden_states[:ROWS, :H], xs[0]) and _same(
        spec.hidden_states[:ROWS, 2 * H : 3 * H], xs[4]
    )
    forward = [
        c
        for _, c in inspect.getmembers(mod, inspect.isclass)
        if c.__module__ == mod.__name__
        and "forward" in c.__dict__
        and "self.spec_worker" in inspect.getsource(c.__dict__["forward"])
    ]
    assert forward, "no target forward found"
    for cls in forward:
        src = inspect.getsource(cls.__dict__["forward"])
        assert src.index("m31_capture_fork.join()") > src.index("self.model("), cls.__name__
        assert src.index("m31_capture_fork.join()") < src.index("FINAL_HIDDEN_LAYER_ID"), (
            cls.__name__
        )
    src = inspect.getsource(mod.M3Eagle3Worker.capture_hidden_states)
    assert src.index("m31_capture_fork.join()") < src.index("self._m3_fc_fork("), (
        "early fc fork must join first"
    )


# ---- the served worker: M3DSparkWorker (dspark.py) over DFlashSpecMetadata's capture ---------------------------------

TARGETS = ["minimax_m3_1_flash_nvfp4__sm_107__tp8"]
_PKG = "tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3"


class _DFlashSpec:
    """The in-tree DSpark capture (DFlashSpecMetadata.maybe_capture_hidden_states -> trtllm::inplace_slice_copy, and
    get_hidden_states, the DSpark drafter's read) on the fields they use; ``slow`` sleeps on the stream the copy is
    issued on first."""

    def __init__(self, slow: int = 0):
        self.layers_to_capture = list(LAYERS)
        self._layer_to_idx = {lid: i for i, lid in enumerate(LAYERS)}
        self._capture_layer_set = frozenset(LAYERS)
        self.num_capture_layers = len(LAYERS)
        self.hidden_size = H
        self._warned_capture_dtype = False
        self.captured_hidden_states = torch.empty(
            CAPACITY, H * len(LAYERS), dtype=torch.bfloat16, device="cuda"
        )
        self.slow = slow

    def is_layer_capture(self, layer_id: int) -> bool:
        return DFlashSpecMetadata.is_layer_capture(self, layer_id)

    def maybe_capture_hidden_states(self, layer_id, hidden_states, residual=None):
        if self.slow:
            torch.cuda._sleep(self.slow)
        DFlashSpecMetadata.maybe_capture_hidden_states(self, layer_id, hidden_states, residual)

    def get_hidden_states(self, num_tokens: int):
        return DFlashSpecMetadata.get_hidden_states(self, num_tokens)


def _dspark(target: str):
    return importlib.import_module(f"{_PKG}.{target}.dspark")


def _worker(dspark):
    """The served worker class without its constructor: capture_hidden_states reads no instance state."""
    return dspark.M3DSparkWorker.__new__(dspark.M3DSparkWorker)


def _dspark_stack(spec: _DFlashSpec, worker, xs, w):
    """As _stack, through the worker's capture hook (the target decode loop's ``capture(spec_metadata, i, residual,
    None)``) and the target forward's join; then the DSpark drafter's read of the capture buffer."""
    h = xs[0]
    outs = []
    for k, layer in enumerate(LAYERS):
        torch.cuda._sleep(200_000)  # a slow boundary: the forked copy must wait for it
        residual = h * 1.0009765625 + xs[k + 1]
        worker.capture_hidden_states(spec, layer, residual, None)
        h = residual * 0.5 - xs[k + 1]
        outs.append(h)
    forked = capfork.join()
    assert capfork.pending() == 0
    drafter_in = spec.get_hidden_states(ROWS)
    return drafter_in.clone(), (drafter_in.float() @ w.float()), outs, forked


@pytest.mark.parametrize("target", TARGETS)
def test_dspark_worker_wiring(monkeypatch: pytest.MonkeyPatch, target: str) -> None:
    """M3DSparkWorker.capture_hidden_states forks only the plain capture of a decode capture layer (no residual
    addend, not the final hidden state), the kill switch restores the in-line copy, the switch and the final-hidden id
    match modeling.py's, and the target forward hands the standalone DSpark worker's hook to the decoder stack
    (``capture = worker.capture_hidden_states``, no capture_target) and joins before reading the buffer."""
    dspark = _dspark(target)
    mod = importlib.import_module(f"{_PKG}.{target}.modeling")
    assert dspark.FRONT_CAPFORK == (not dspark.m31_disabled("FRONT_CAPFORK")) == mod.FRONT_CAPFORK
    assert dspark._FINAL_HIDDEN_LAYER_ID == mod.FINAL_HIDDEN_LAYER_ID
    assert "m31_capture_fork.fork_capture(" in inspect.getsource(
        dspark.M3DSparkWorker.capture_hidden_states
    )
    xs, _ = _inputs(41)
    spec = _DFlashSpec()
    worker = _worker(dspark)
    monkeypatch.setattr(dspark, "FRONT_CAPFORK", True)
    worker.capture_hidden_states(spec, LAYERS[0], xs[0], None)
    assert capfork.pending() == 1
    worker.capture_hidden_states(spec, LAYERS[1], xs[1], xs[2])  # with a residual addend: in line
    worker.capture_hidden_states(
        spec, dspark._FINAL_HIDDEN_LAYER_ID, xs[3], None
    )  # final hidden: in line
    assert capfork.pending() == 1
    assert capfork.join() == 1
    monkeypatch.setattr(dspark, "FRONT_CAPFORK", False)
    worker.capture_hidden_states(spec, LAYERS[2], xs[4], None)
    assert capfork.pending() == 0
    torch.cuda.synchronize()
    buf = spec.captured_hidden_states
    assert _same(buf[:ROWS, :H], xs[0]) and _same(buf[:ROWS, 2 * H : 3 * H], xs[4])
    assert _same(buf[:ROWS, H : 2 * H], xs[1] + xs[2])
    for cls in [
        c
        for _, c in inspect.getmembers(mod, inspect.isclass)
        if c.__module__ == mod.__name__
        and "forward" in c.__dict__
        and "self.spec_worker" in inspect.getsource(c.__dict__["forward"])
    ]:
        src = inspect.getsource(cls.__dict__["forward"])
        assert "capture = worker.capture_hidden_states" in src, cls.__name__
        assert "_standalone_dspark" in src, (
            f"{cls.__name__}: no capture_target guard for standalone DSpark"
        )
        assert src.index("m31_capture_fork.join()") > src.index("self.model("), cls.__name__
        assert src.index("m31_capture_fork.join()") < src.index("FINAL_HIDDEN_LAYER_ID"), (
            cls.__name__
        )
        assert src.index("m31_capture_fork.join()") < src.index("outputs = worker("), cls.__name__


@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize("slow", [0, SLOW])
def test_dspark_worker_eager_on_equals_off(
    monkeypatch: pytest.MonkeyPatch, target: str, slow: int
) -> None:
    dspark = _dspark(target)
    worker = _worker(dspark)
    xs, w = _inputs(51)
    got = {}
    for name, on in (("off", False), ("on", True)):
        monkeypatch.setattr(dspark, "FRONT_CAPFORK", on)
        spec = _DFlashSpec(slow if on else 0)
        spec.captured_hidden_states.view(torch.int16).fill_(-1)
        cap, fc, outs, forked = _dspark_stack(spec, worker, xs, w)
        assert forked == (len(LAYERS) if on else 0), f"{name}: {forked} forked copies"
        torch.cuda.synchronize()
        got[name] = (cap, fc, outs, spec.captured_hidden_states.clone())
    on, off = got["on"], got["off"]
    assert _same(on[0], off[0]), "DSpark captured bytes differ ON vs OFF"
    assert _same(on[0], _expected(xs)), "the DSpark capture is not the boundary's residual"
    assert _same(on[1], off[1]), "the DSpark drafter input differs ON vs OFF"
    assert all(_same(a, b) for a, b in zip(on[2], off[2]))
    assert _same(on[3], off[3]), (
        "the capture buffer (rows past the step included) differs ON vs OFF"
    )


@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize("slow", [0, SLOW])
def test_dspark_worker_cuda_graph_on_equals_off(
    monkeypatch: pytest.MonkeyPatch, target: str, slow: int
) -> None:
    dspark = _dspark(target)
    worker = _worker(dspark)
    xs, w = _inputs(52)
    static = [x.clone() for x in xs]
    graphs = {}
    for name, on in (("off", False), ("on", True)):
        monkeypatch.setattr(dspark, "FRONT_CAPFORK", on)
        spec = _DFlashSpec(slow if on else 0)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            _dspark_stack(
                spec, worker, static, w
            )  # eager warm-up (creates the capture stream before the capture)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = _dspark_stack(spec, worker, static, w)
        assert out[3] == (len(LAYERS) if on else 0), (
            f"{name}: {out[3]} forked copies in the captured step"
        )
        graphs[name] = (graph, spec, out)
    for seed in (61, 62, 63):
        fresh, _ = _inputs(seed)
        for x, y in zip(static, fresh):
            x.copy_(y)
        got = {}
        for name, (graph, spec, out) in graphs.items():
            spec.captured_hidden_states.view(torch.int16).fill_(
                -1
            )  # a stale or late copy cannot pass
            graph.replay()
            torch.cuda.synchronize()
            got[name] = (
                spec.captured_hidden_states.clone(),
                out[0].clone(),
                out[1].clone(),
                [h.clone() for h in out[2]],
            )
        on, off = got["on"], got["off"]
        assert _same(on[0], off[0]), (
            f"seed {seed}: DSpark capture buffer differs ON vs OFF under graph replay"
        )
        assert _same(on[1], _expected(fresh)), (
            f"seed {seed}: the replayed DSpark capture is not this replay's residual"
        )
        assert _same(on[2], off[2]), (
            f"seed {seed}: the DSpark drafter input differs ON vs OFF under graph replay"
        )
        assert all(_same(a, b) for a, b in zip(on[3], off[3]))
