# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the eager_on_graph catalog entry.

Each cell captures (or, for a passthrough cell, eagerly runs) a body of `breaks` breaks with
row-wise ops between them, the shape of a breakable prefill body, and replays it at the cell's
live rows. Every break's `inner` reads the step's rows and scale from live state, mixes rows
`[:n]` of its operand across the batch, writes a fixed output allocated inside the body and
zeroes its rows `[n:]`. The replayed body must equal the same body run eagerly at the same
live state, bit for bit; the hazards the entry's note states are each shown by a test below.
"""

import ctypes
import logging
import zlib
from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog.graph.eager_on_graph import (
    eager_on_graph as op,
)
from tensorrt_llm._torch.pyexecutor.breakable_cuda_graph import (
    BreakableCUDAGraph,
    BreakableCUDAGraphCapture,
)

__extra_import_path__ = [".."]
from _validating import certifying  # noqa: E402,F401 — path above; fixture by name

assert torch.cuda.is_available(), "eager_on_graph requires a CUDA device"

_BCG_LOGGER = "tensorrt_llm._torch.pyexecutor.breakable_cuda_graph.breakable_cuda_graph"

#: `cudaError_t` values the pending-error checks compare against.
_CUDA_SUCCESS = 0
_CUDA_ERROR_ILLEGAL_STATE = 401


def _loaded_cudart() -> ctypes.CDLL:
    """The CUDA runtime library torch already loaded into this process. Its
    per-thread last error is the one torch's launch checks read, which
    `cuda.bindings.runtime` (a separate runtime instance) does not see."""
    soname = f"/libcudart.so.{torch.version.cuda.split('.')[0]}"
    with open("/proc/self/maps") as maps:
        paths = sorted({line.split()[-1] for line in maps if soname in line})
    assert len(paths) == 1, f"expected exactly one loaded {soname[1:]}, found {paths}"
    return ctypes.CDLL(paths[0])


_CUDART = _loaded_cudart()


def _take_pending_cuda_error() -> int:
    """Read and clear the CUDA runtime's pending last error on this thread."""
    return int(_CUDART.cudaGetLastError())


@pytest.fixture(autouse=True)
def _leaves_cuda_clean():
    """Every test here hands the process on as it found it: no stream
    capturing and no CUDA runtime error pending. A pending error is reported
    by whichever CUDA launch comes next -- in a shared pytest process, the
    first call of the next test file -- so it is taken here, after every test
    and whether or not the test passed, and a test that left one fails here
    instead of there."""
    yield
    pending = _take_pending_cuda_error()
    torch.cuda.synchronize()
    assert not torch.cuda.is_current_stream_capturing(), "a test left the stream capturing"
    assert pending == _CUDA_SUCCESS, f"the test left CUDA runtime error {pending} pending"


class _Live:
    """The per-step state a break reads, refreshed before every step."""

    def __init__(self, rows: int, scale: float) -> None:
        self.rows = rows
        self.scale = scale


def _inner(live: _Live):
    """A break's eager work: rows `[:n]` centred over the live batch, scaled by
    the live scale and offset by the static layer index, into `out` in place;
    rows `[n:]` zeroed. Returns None, as an in-place break does."""

    def inner(src: torch.Tensor, out: torch.Tensor, layer: int) -> None:
        n = live.rows
        rows = src[:n].float()
        mixed = (rows - rows.mean(0, keepdim=True)) * live.scale + layer
        out[:n].copy_(mixed.to(out.dtype))
        out[n:].zero_()

    return inner


def _body(x: torch.Tensor, breaks: int, call) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Row-wise ops around `breaks` breaks; each break's output is allocated in
    the body, as a captured body allocates it from the graph pool."""
    h = torch.sin(x) * 2.0
    outs = []
    for layer in range(breaks):
        out = torch.empty_like(h)
        call(h, out, layer)
        outs.append(out)
        h = torch.tanh(out) + h * 0.5
    return h, outs


def _capture(x_static: torch.Tensor, body) -> tuple:
    """Two eager warmups, then the capture, on a side stream as the engine runs it."""
    graph = BreakableCUDAGraph()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(2):
            body(x_static)
        with BreakableCUDAGraphCapture(graph, pool=torch.cuda.graph_pool_handle(), stream=stream):
            h, outs = body(x_static)
    torch.cuda.current_stream().wait_stream(stream)
    return graph, h, outs


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 — pytest resolves the fixture by this name
    certifying(op)
    spec = cell.spec
    tokens, width, breaks = spec["tokens"], spec["width"], spec["breaks"]
    gen = torch.Generator(device="cuda").manual_seed(zlib.crc32(cell.why.encode()) % 2**31)
    live = _Live(rows=tokens, scale=1.0)
    inner = _inner(live)

    def broken(*operands):
        return op(*operands, inner=inner)

    def eager(*operands):
        return op.reference(*operands, inner=inner)

    x_static = torch.randn([tokens, width], generator=gen, device="cuda").to(torch.bfloat16)
    graph = None
    if spec["capture"]:
        graph, h_static, outs_static = _capture(x_static, lambda x: _body(x, breaks, broken))
        assert graph.num_breaks == breaks and graph.num_segments == breaks + 1
        assert graph.num_segments == breaks + 1

    for step, rows in enumerate(spec["live_rows"]):
        assert 0 < rows <= tokens
        live.rows, live.scale = rows, 0.25 + 0.5 * step
        x_new = torch.randn([tokens, width], generator=gen, device="cuda").to(torch.bfloat16)
        if graph is not None:
            x_static.copy_(x_new)
            graph.replay()
            got, got_outs = h_static.clone(), [o.clone() for o in outs_static]
        else:
            got, got_outs = _body(x_new, breaks, broken)
        want, want_outs = _body(x_new, breaks, eager)
        assert len(got_outs) == len(want_outs) == breaks
        op.compare(got, want)
        for got_out, want_out in zip(got_outs, want_outs):
            op.compare(got_out, want_out)
            assert torch.count_nonzero(got_out[rows:]) == 0, "padding rows left unwritten"


def _single_break_graph(inner, x: torch.Tensor, *extra):
    """Capture `inner(h, out, *extra)` as one break between two segments that
    read `x` and copy `out` into the returned result. Every tensor the graph
    touches is allocated inside the capture and kept by the caller."""
    graph = BreakableCUDAGraph()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        with BreakableCUDAGraphCapture(graph, pool=torch.cuda.graph_pool_handle(), stream=stream):
            h = x * 1.0
            out = torch.empty_like(h)
            op(h, out, *extra, inner=inner)
            result = out + 0.0
    torch.cuda.current_stream().wait_stream(stream)
    return graph, result, (h, out)


def test_a_scalar_argument_is_frozen_at_capture() -> None:
    """A per-step count handed in as an argument replays at its capture-time value."""

    def inner(src, out, n):
        out[:n].copy_(src[:n] * 2)
        out[n:].zero_()

    x = torch.randn(64, 8, device="cuda")
    graph, result, _keep = _single_break_graph(inner, x, 64)
    # The step now carries 17 rows, but nothing can hand the replay that number.
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(result, x * 2), "the replay must run at the capture-time n"
    stale = torch.zeros_like(x)
    stale[:17] = x[:17] * 2
    assert not torch.equal(result, stale)


def test_a_tensor_argument_is_read_at_its_capture_address() -> None:
    """A tensor argument is re-read at its capture-time address: a new tensor the
    step computed is never seen, and the captured one is read as it now is."""

    def inner(src, out, scale):
        torch.mul(src, scale, out=out)

    x = torch.randn(32, 8, device="cuda")
    captured = torch.full((1,), 1.0, device="cuda")
    graph, result, _keep = _single_break_graph(inner, x, captured)
    fresh = torch.full((1,), 2.0, device="cuda")  # what a per-step computation would make
    assert fresh.data_ptr() != captured.data_ptr()
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(result, x * 1.0), "replay must read the capture-time tensor"
    captured.fill_(3.0)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(result, x * 3.0), "replay must read the capture-time address as it is now"


def test_a_wait_on_a_non_capturing_stream_is_dropped(caplog) -> None:
    """Inside the inner's own capture-time run, a wait on a stream that is not
    capturing is dropped, with a warning naming the call site."""
    side = torch.cuda.Stream()

    def inner(src, out, layer):
        torch.cuda.current_stream().wait_stream(side)
        out.copy_(src)

    with caplog.at_level(logging.WARNING, logger=_BCG_LOGGER):
        _single_break_graph(inner, torch.randn(8, 8, device="cuda"), 0)
    assert any("Dropping a wait" in record.message for record in caplog.records)


def test_an_object_argument_raises_under_capture() -> None:
    """Any argument the replay cannot hold weakly is refused at capture, after
    the inner already ran once. Outside capture the same call is a passthrough.
    The refusal is a break raising under capture, so it also leaves the error
    `test_a_raising_break_leaves_an_error_pending` shows, taken here."""
    ran = []

    def inner(src, out, config):
        ran.append(config.tag)
        out.copy_(src)

    config = SimpleNamespace(tag="step")
    x = torch.randn(8, 8, device="cuda")
    out = torch.empty_like(x)
    op(x, out, config, inner=inner)
    assert torch.equal(out, x) and ran == ["step"]

    with pytest.raises((TypeError, RuntimeError)) as info:
        _single_break_graph(inner, x, config)
    chain = []
    error = info.value
    while error is not None:
        chain.append(error)
        error = error.__context__
    assert any(isinstance(e, TypeError) and "Invalid type" in str(e) for e in chain)
    assert ran == ["step", "step"]
    assert _take_pending_cuda_error() == _CUDA_ERROR_ILLEGAL_STATE
    torch.cuda.synchronize()


def test_a_raising_break_leaves_an_error_pending() -> None:
    """A break that raises under capture has already ended the captured
    segment, and the capture's exit ends it again: the exit raises
    RuntimeError over the break's own error and leaves cudaErrorIllegalState
    pending, which the process's next CUDA launch reports as its own failure
    -- once, since that launch's check clears it."""

    def inner(src, out, layer):
        out.copy_(src)
        raise ValueError("the break's own failure")

    with pytest.raises(RuntimeError) as info:
        _single_break_graph(inner, torch.randn(8, 8, device="cuda"), 0)
    assert isinstance(info.value.__context__, ValueError), info.value.__context__
    torch.cuda.synchronize()

    assert _CUDART.cudaPeekAtLastError() == _CUDA_ERROR_ILLEGAL_STATE
    with pytest.raises(torch.AcceleratorError, match="cannot be performed in the present state"):
        torch.ones(1, device="cuda").add_(1)
    assert _take_pending_cuda_error() == _CUDA_SUCCESS
    assert torch.equal(torch.ones(1, device="cuda").add_(1).cpu(), torch.full((1,), 2.0))
