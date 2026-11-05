# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A copy plan's parallel scatter and its in-graph final flush (``CopyPlan.parallel_scatter``, ``InGraph``)."""

import time
import weakref
from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.pyexecutor import device_copy
from tensorrt_llm._torch.pyexecutor.cuda_graph_runner import CUDAGraphRunner, KeyType
from tensorrt_llm._torch.pyexecutor.device_copy import (
    CopyPlan,
    CopySiteEnum,
    DeviceCopier,
    Direction,
    InGraph,
)


def _scatter_op_registered() -> bool:
    return torch.cuda.is_available() and hasattr(torch.ops.trtllm, "device_copy_scatter")


pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
    pytest.mark.skipif(
        not _scatter_op_registered(),
        reason="torch.ops.trtllm.device_copy_scatter is not registered (built from "
        "cpp/tensorrt_llm/thop/deviceCopyScatterOp.cpp)",
    ),
]

# About half a second of GPU time: long enough that work queued behind it is still pending when
# the host checks, short enough for a unit test.
_SLEEP_CYCLES = 1_000_000_000


class _Site(CopySiteEnum):
    H2D = ("test_in_graph.h2d", Direction.H2D)
    D2D = ("test_in_graph.d2d", Direction.D2D)


@pytest.fixture(autouse=True)
def _no_check_mode(monkeypatch):
    # These tests read merged destinations directly; check mode (on for unit tests) would poison them.
    monkeypatch.setattr(device_copy, "CHECK_MODE", False)


def _device():
    return torch.device("cuda", torch.cuda.current_device())


def _host(values, dtype=torch.int32):
    return torch.tensor(values, dtype=dtype).pin_memory()


def _zeros(*shape, dtype=torch.int32):
    return torch.zeros(*shape, dtype=dtype, device="cuda")


def _copier(parallel_scatter=False, final_flush=None) -> DeviceCopier:
    plan = CopyPlan(
        merge={_Site.H2D, _Site.D2D}, parallel_scatter=parallel_scatter, final_flush=final_flush
    )
    return DeviceCopier(plan, _device())


def _copy(dst, src):
    device_copy.copy(_Site.D2D if src.is_cuda else _Site.H2D, dst, src)


# ---------------------------------------------------------------------------
# Parallel scatter
# ---------------------------------------------------------------------------


def _random_pack_run(parallel: bool, seed: int, overlap: bool):
    """Stage a fixed random mix of host copies (2-D page-table rows included) and device copies that read a host
    copy's destination; return every destination after the flush."""
    g = torch.Generator().manual_seed(seed)
    copier = _copier(parallel_scatter=parallel)
    big = _zeros(4096)
    table = torch.zeros((8, 64), dtype=torch.int32, device=_device())
    out = _zeros(256)
    with device_copy.active(copier):
        device_copy.begin()
        for k in range(24):
            n = int(torch.randint(1, 64, (1,), generator=g))
            _copy(
                big[k * 128 : k * 128 + n],
                torch.randint(0, 1 << 30, (n,), generator=g, dtype=torch.int32),
            )
        _copy(table[:3, :40], torch.randint(0, 1 << 30, (3, 40), generator=g, dtype=torch.int32))
        if overlap:
            # A later host copy overwrites part of an earlier one: order matters, so the serial scatter must run.
            _copy(big[0:4], torch.full((4,), -7, dtype=torch.int32))
        _copy(out[:64], big[:64])
        _copy(out[64:96], big[128:160])
        device_copy.flush()
    torch.cuda.synchronize()
    return big.cpu(), table.cpu(), out.cpu()


@pytest.mark.parametrize("overlap", [False, True], ids=["disjoint", "overlapping"])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_parallel_scatter_writes_what_the_serial_scatter_writes(seed, overlap):
    serial = _random_pack_run(False, seed, overlap)
    parallel = _random_pack_run(True, seed, overlap)
    for a, b in zip(serial, parallel):
        assert torch.equal(a, b)


def _compiled_variants(kernel) -> int:
    """Compiled variants of a Triton kernel on this device, or -1 where this Triton keeps them elsewhere."""
    caches = getattr(kernel, "device_caches", None)
    if not caches:
        return -1
    cache = caches.get(torch.cuda.current_device())
    return len(cache[0]) if cache else 0


@pytest.mark.parametrize("count", [1, 15, 16, 17, 120])  # 120: 240 of the 256 descriptors
def test_parallel_scatter_is_bitwise_for_every_copy_count(count):
    """``first`` (the device copies' descriptor offset) is the segment's host-copy count: one compiled variant serves
    every count, so a new count never compiles mid-serving."""
    copier = _copier(parallel_scatter=True)
    pack = copier._pack
    width = 3
    dst = _zeros(count * width)
    mirror = _zeros(count * width)
    g = torch.Generator().manual_seed(count)
    values = torch.randint(
        -(1 << 31), (1 << 31) - 1, (count, width), generator=g, dtype=torch.int32
    )
    with device_copy.active(copier):
        device_copy.begin()
        for i in range(count):
            _copy(dst[i * width : (i + 1) * width], values[i])
        for i in range(count):
            _copy(mirror[i * width : (i + 1) * width], dst[i * width : (i + 1) * width])
        assert len(pack.h2d) == count and len(pack.d2d) == count
        device_copy.flush()
    torch.cuda.synchronize()
    assert torch.equal(dst.cpu(), values.flatten())
    assert torch.equal(mirror.cpu(), values.flatten())
    variants = _compiled_variants(device_copy._parallel_scatter_kernel)
    assert variants in (-1, 1, 2), variants  # at most one per DEVICE_SRC


def test_parallel_scatter_only_for_order_free_packs():
    assert device_copy._parallel_ok([(0, 0, 8), (8, 0, 8)], [])
    assert not device_copy._parallel_ok([(0, 0, 8), (4, 0, 8)], [])
    # A device copy may read a host copy's destination (the launches run in order) ...
    assert device_copy._parallel_ok([(0, 0, 8)], [(100, 0, 8)])
    # ... but not another device copy's destination, and device destinations must be disjoint.
    assert not device_copy._parallel_ok([], [(100, 0, 8), (200, 100, 8)])
    assert not device_copy._parallel_ok([], [(100, 0, 8), (104, 300, 8)])


def test_parallel_scatter_is_off_by_default():
    assert not CopyPlan().parallel_scatter
    assert not _copier()._pack.parallel_scatter


def test_a_barrier_segment_scatters_in_parallel_too():
    """Every segment of a step takes the parallel scatter when order-free, not only the last one."""
    plan = CopyPlan(merge={_Site.H2D}, flush_after={_Site.H2D}, parallel_scatter=True)
    copier = DeviceCopier(plan, _device())
    a, b = _zeros(4), _zeros(4)
    with device_copy.active(copier):
        device_copy.begin()
        _copy(a, _host([1, 2, 3, 4]))
        _copy(b, _host([5, 6, 7, 8]))
        device_copy.flush()
    torch.cuda.synchronize()
    assert a.tolist() == [1, 2, 3, 4] and b.tolist() == [5, 6, 7, 8]


# ---------------------------------------------------------------------------
# In-graph final flush
# ---------------------------------------------------------------------------


def _graph_case(early=False, epoch=False):
    """A copier with the in-graph prologue and a graph that starts with it and then reads the destinations."""
    copier = _copier(parallel_scatter=True, final_flush=InGraph(early_copy=early, step_epoch=epoch))
    a, b, c = _zeros(16), _zeros(16), _zeros(16)
    out = _zeros(48)
    graph = torch.cuda.CUDAGraph()
    with device_copy.active(copier):
        torch.cuda.synchronize()
        with torch.cuda.graph(graph):
            assert device_copy.capture_prologue() is copier
            out.copy_(torch.cat([a, b, c]))
    return copier, graph, (a, b, c), out


def _step(copier, graph, dsts, values, overlap=False, flush=True):
    a, b, c = dsts
    with device_copy.active(copier):
        device_copy.begin()
        _copy(a, _host(values[0]))
        _copy(b, _host(values[1]))
        if overlap:
            _copy(a[:4], _host([-1, -2, -3, -4]))
        _copy(c, a)  # a device copy reading a host copy's destination
        if not flush:
            device_copy.discard()
        device_copy.flush(in_graph=True)
    graph.replay()
    copier.on_graph_replay()


def test_in_graph_pack_is_issued_by_the_replay():
    copier, graph, dsts, out = _graph_case()
    for step in range(5):
        va = [step * 100 + i for i in range(16)]
        vb = [step * 1000 - i for i in range(16)]
        _step(copier, graph, dsts, (va, vb))
        torch.cuda.synchronize()
        assert out.tolist() == va + vb + va, step
        # Nothing was issued outside the graph: the window carried the copies.
        assert int(copier._pack.ghost[:8].view(torch.int64)[0]) == 2


def test_in_graph_pack_not_order_free_is_issued_eagerly():
    copier, graph, dsts, out = _graph_case()
    va = list(range(16))
    vb = list(range(100, 116))
    _step(copier, graph, dsts, (va, vb), overlap=True)
    torch.cuda.synchronize()
    expect_a = [-1, -2, -3, -4] + va[4:]
    assert out.tolist() == expect_a + vb + expect_a
    # The window was zeroed, so the prologue wrote nothing over the eager result.
    assert not copier._pack.ghost[:16].any()


def test_in_graph_replay_without_a_pack_reapplies_nothing():
    copier, graph, dsts, out = _graph_case()
    _step(copier, graph, dsts, (list(range(16)), list(range(16))))
    torch.cuda.synchronize()
    dsts[0].fill_(7)
    _step(copier, graph, dsts, ([0] * 16, [0] * 16), flush=False)
    torch.cuda.synchronize()
    assert dsts[0].tolist() == [7] * 16


@pytest.mark.parametrize("early", [False, True], ids=["prologue_copy", "early_copy"])
def test_in_graph_pack_lands_every_step(early):
    copier, graph, dsts, out = _graph_case(early)
    assert copier._pack.early_copy is early
    for step in range(6):
        va = [step * 100 + i for i in range(16)]
        vb = [step * 1000 - i for i in range(16)]
        _step(copier, graph, dsts, (va, vb), overlap=step % 2 == 1)
        torch.cuda.synchronize()
        head = [-1, -2, -3, -4] if step % 2 == 1 else []
        expect_a = head + va[len(head) :]
        assert out.tolist() == expect_a + vb + expect_a, step


def test_only_the_last_segment_rides_the_window():
    """A barrier segment is issued eagerly before the replay; the window carries what is staged after it."""
    plan = CopyPlan(
        merge={_Site.H2D, _Site.D2D},
        flush_after={_Site.D2D},
        parallel_scatter=True,
        final_flush=InGraph(early_copy=True),
    )
    copier = DeviceCopier(plan, _device())
    a, b, c = _zeros(16), _zeros(16), _zeros(16)
    out = _zeros(48)
    graph = torch.cuda.CUDAGraph()
    with device_copy.active(copier):
        torch.cuda.synchronize()
        with torch.cuda.graph(graph):
            device_copy.capture_prologue()
            out.copy_(torch.cat([a, b, c]))
    for step in range(3):
        va = [step * 10 + i for i in range(16)]
        vb = [step * 100 + i for i in range(16)]
        with device_copy.active(copier):
            device_copy.begin()
            _copy(a, _host(va))
            _copy(c, a)  # a barrier: a and c land now, eagerly
            barrier_seen = c.clone()
            _copy(b, _host(vb))
            device_copy.flush(in_graph=True)
            # One host copy in the window.
            assert int(copier._pack.ghost[:8].view(torch.int64)[0]) == 1
        graph.replay()
        copier.on_graph_replay()
        torch.cuda.synchronize()
        assert barrier_seen.tolist() == va, step
        assert out.tolist() == va + vb + va, step


def _history_run(copier, graph, dsts, out, steps, before_replay=None, before_upload=None):
    """Steps back to back without host syncs; each replay's output is copied to a history row behind it (same
    stream), so a later upload that overwrote the window too early shows up as the wrong step's values."""
    hist = _zeros(steps, out.numel())
    expect = []
    for step in range(steps):
        va = [step * 100 + i for i in range(16)]
        vb = [step * 1000 - i for i in range(16)]
        expect.append(va + vb + va)
        a, b, c = dsts
        with device_copy.active(copier):
            device_copy.begin()
            _copy(a, _host(va))
            _copy(b, _host(vb))
            _copy(c, a)
            if before_upload is not None:
                before_upload(step)
            device_copy.flush(in_graph=True)
        if before_replay is not None:
            before_replay(step)
        graph.replay()
        hist[step].copy_(out)
    torch.cuda.synchronize()
    return hist.tolist(), expect


def test_early_copy_waits_for_the_previous_prologue_before_overwriting_the_window():
    """A slow GPU: the replay of step N (and its prologue) is held back behind a long kernel while the host already
    uploads step N+1; that upload must wait for step N's prologue to consume the window."""
    copier, graph, dsts, out = _graph_case(early=True)
    hist, expect = _history_run(
        copier,
        graph,
        dsts,
        out,
        4,
        before_replay=lambda step: torch.cuda._sleep(_SLEEP_CYCLES // 4),
    )
    assert hist == expect


def test_early_copy_prologue_waits_for_a_slow_upload():
    """A slow copy stream: each step's upload is held back behind a long kernel on the copy stream; the prologue must
    wait for it (staged_ready), and the next host fill must wait for the upload to have read the window."""
    copier, graph, dsts, out = _graph_case(early=True)

    def slow_copy(step):
        with torch.cuda.stream(copier._pack.copy_stream):
            torch.cuda._sleep(_SLEEP_CYCLES // 4)

    hist, expect = _history_run(copier, graph, dsts, out, 4, before_upload=slow_copy)
    assert hist == expect


def test_early_copy_with_a_slow_host():
    """A slow host: the GPU idles between steps (the window lands long before the replay)."""
    copier, graph, dsts, out = _graph_case(early=True)
    hist, expect = _history_run(
        copier, graph, dsts, out, 3, before_replay=lambda step: time.sleep(0.05)
    )
    assert hist == expect


@pytest.mark.parametrize("early", [False, True], ids=["prologue_copy", "early_copy"])
def test_in_graph_replay_after_quiesce_reapplies_nothing(early):
    copier, graph, dsts, out = _graph_case(early)
    _step(copier, graph, dsts, (list(range(16)), list(range(16))))
    torch.cuda.synchronize()
    dsts[0].fill_(7)
    copier.quiesce_graph_window()
    graph.replay()
    torch.cuda.synchronize()
    assert dsts[0].tolist() == [7] * 16


def test_prologue_op_needs_early_copy_and_is_off_by_default():
    def op(window, epoch, max_desc):
        raise AssertionError("not called")

    assert InGraph().prologue_op is None
    assert not InGraph().early_copy
    with pytest.raises(ValueError, match="needs early_copy"):
        InGraph(prologue_op=op)


def test_in_graph_needs_parallel_scatter_and_is_off_by_default():
    assert CopyPlan().final_flush is None
    assert _copier()._pack.in_graph is None
    with pytest.raises(ValueError, match="needs parallel_scatter"):
        CopyPlan(merge={_Site.H2D}, final_flush=InGraph())
    with pytest.raises(ValueError, match="must merge some"):
        CopyPlan(parallel_scatter=True, final_flush=InGraph())


def test_a_copier_without_the_in_graph_flush_captures_no_prologue():
    copier = _copier(parallel_scatter=True)
    graph = torch.cuda.CUDAGraph()
    with device_copy.active(copier):
        torch.cuda.synchronize()
        with torch.cuda.graph(graph):
            assert device_copy.capture_prologue() is None
    assert device_copy.capture_prologue() is None


def test_in_graph_replay_outside_the_pack_context_reapplies_nothing():
    copier, graph, dsts, out = _graph_case()
    _step(copier, graph, dsts, (list(range(16)), list(range(16))))
    torch.cuda.synchronize()
    dsts[0].fill_(7)
    # What the graph runner does for a replay outside the copier's context.
    assert not device_copy.is_active(copier)
    copier.quiesce_graph_window()
    graph.replay()
    torch.cuda.synchronize()
    assert dsts[0].tolist() == [7] * 16


def test_step_epoch_is_off_by_default_and_needs_in_graph():
    assert not InGraph().step_epoch
    copier = _copier(parallel_scatter=True, final_flush=InGraph())
    assert copier.epoch is None
    with device_copy.active(copier):
        assert device_copy.step_epoch() is None
    assert device_copy.step_epoch() is None
    assert _copier(parallel_scatter=True).epoch is None


@pytest.mark.parametrize("early", [False, True], ids=["prologue_copy", "early_copy"])
def test_step_epoch_advances_once_per_replay_and_never_resets(early):
    copier, graph, dsts, out = _graph_case(early, epoch=True)
    with device_copy.active(copier):
        assert device_copy.step_epoch() is copier.epoch
    torch.cuda.synchronize()
    start = int(copier.epoch.item())  # the warm-up compile launched the prologue once
    assert start == 1
    for step in range(5):
        va = [step * 100 + i for i in range(16)]
        vb = [step * 1000 - i for i in range(16)]
        # A replay with a new pack, one with an eagerly issued pack, and one with nothing to scatter.
        _step(copier, graph, dsts, (va, vb), overlap=step == 2, flush=step != 3)
        torch.cuda.synchronize()
        assert int(copier.epoch.item()) == start + step + 1, step
    # The copies are unchanged by the epoch.
    _step(copier, graph, dsts, (list(range(16)), list(range(16, 32))))
    torch.cuda.synchronize()
    assert out.tolist() == list(range(16)) + list(range(16, 32)) + list(range(16))
    assert int(copier.epoch.item()) == start + 6


def test_step_epoch_is_ordered_before_the_graph_body():
    """A kernel after the prologue in the same graph reads the bumped epoch of its own replay."""
    copier = _copier(parallel_scatter=True, final_flush=InGraph(step_epoch=True))
    seen = torch.zeros(4, dtype=torch.int64, device="cuda")
    slot = torch.zeros(1, dtype=torch.int64, device="cuda")
    graph = torch.cuda.CUDAGraph()
    with device_copy.active(copier):
        torch.cuda.synchronize()
        with torch.cuda.graph(graph):
            device_copy.capture_prologue()
            seen.index_copy_(0, slot, copier.epoch)
            slot.add_(1)
    torch.cuda.synchronize()
    start = int(copier.epoch.item())
    for _ in range(4):
        with device_copy.active(copier):
            copier.quiesce_graph_window()
        graph.replay()
    torch.cuda.synchronize()
    assert seen.tolist() == [start + 1, start + 2, start + 3, start + 4]


def test_eager_flush_does_not_advance_the_step_epoch():
    copier = _copier(parallel_scatter=True, final_flush=InGraph(step_epoch=True))
    torch.cuda.synchronize()
    start = int(copier.epoch.item())
    dst = _zeros(4)
    with device_copy.active(copier):
        device_copy.begin()
        _copy(dst, _host([1, 2, 3, 4]))
        device_copy.flush()
    torch.cuda.synchronize()
    assert dst.tolist() == [1, 2, 3, 4]
    assert int(copier.epoch.item()) == start


@pytest.mark.parametrize("early", [False, True])
def test_graph_window_keeps_temporary_device_sources_until_replay(early):
    copier, graph, (a, b, c), out = _graph_case(early)
    source = torch.arange(16, device="cuda", dtype=a.dtype)
    source_ref = weakref.ref(source)
    with device_copy.active(copier):
        device_copy.begin()
        _copy(a, source)
        del source
        device_copy.flush(in_graph=True)
    assert source_ref() is not None
    overwrite = torch.full((16,), -7, device="cuda", dtype=a.dtype)
    graph.replay()
    copier.on_graph_replay()
    torch.cuda.synchronize()
    assert out[:16].tolist() == list(range(16))
    assert overwrite.eq(-7).all()
    assert source_ref() is None


@pytest.mark.parametrize("early", [False, True])
def test_replay_with_a_different_copier_issues_its_copies_eagerly(early):
    captured, graph, (a, b, c), out = _graph_case(early)
    _step(captured, graph, (a, b, c), (list(range(16)), list(range(16))))
    active = _copier(parallel_scatter=True, final_flush=InGraph(early_copy=early))
    key = KeyType(batch_size=16, draft_len=0)
    metadata = object()
    runner = object.__new__(CUDAGraphRunner)
    runner.graph_metadata = {key: {"attn_metadata": metadata, "spec_metadata": None}}
    runner.shared_static_tensors = {"input_ids": a, "position_ids": b.view(1, 16)}
    runner.config = SimpleNamespace(use_mrope=False)
    runner._check_strict_buffer_stability = lambda meta: None
    runner._get_num_tokens_for_key = lambda key: 16
    runner._prologue_copiers = {key: captured}
    runner._pre_replay_hooks = []
    runner.graphs = {key: graph}
    runner.graph_outputs = {key: out}
    inputs = {
        "attn_metadata": metadata,
        "input_ids": torch.full_like(a, 77),
        "position_ids": torch.full_like(b.view(1, 16), 88),
    }
    with device_copy.active(active):
        device_copy.begin()
        runner.replay(key, inputs)
    torch.cuda.synchronize()
    assert out[:16].tolist() == [77] * 16
    assert out[16:32].tolist() == [88] * 16
