# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the m3_step_prologue entry (the decode graph's step-input prologue in one launch)."""

import random

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m31_nvfp4__sm_107__tp8.ops.spec.m3_step_prologue import (  # noqa: E501
    m3_step_prologue as op,
)
from tensorrt_llm._torch.pyexecutor import step_inputs
from tensorrt_llm._torch.pyexecutor.step_inputs import StepInputPack

__extra_import_path__ = ["../../../..", "."]
from _m3_spec_inputs import gen, seed_of  # noqa: E402 — needs the path declared above
from _validating import certifying, validating  # noqa: E402, F401

assert torch.cuda.is_available(), "m3_step_prologue requires a CUDA device"
# The reference and the pack tests run this branch's step_inputs (the step epoch, prologue_op): a run that mounts only
# the target's files gets the image's older module, whose graph scatter has no epoch argument.
assert "epoch" in step_inputs._graph_scatter_kernel.arg_names, (
    "tensorrt_llm/_torch/pyexecutor/step_inputs.py is not this branch's (no step epoch): mount the shared pyexecutor "
    "files (step_inputs.py, runtime_hooks.py, engine/runners/decoder/runner.py, h2d_cache.py) with the target"
)

MAX_DESC = StepInputPack.MAX_DESC
WINDOW = StepInputPack.GRAPH_WINDOW
DATA_START = StepInputPack.HEADER_BYTES + StepInputPack.DESC_BYTES * MAX_DESC
ARENA = 1 << 20


def _layout(spec, seed):
    """Copies as (dst, src, bytes) arena offsets (a host copy's src: a window offset) and the host copies' bytes."""
    rng = random.Random(seed)
    h2d, d2d, data = [], [], []
    used = DATA_START
    pos = 0
    for _ in range(spec["h2d"]):
        n = rng.choice([1, 3, 4, 7, 16, 24, 32, 48, 64, 5, 12])
        pos += rng.randint(0, 40)  # gaps and every alignment
        h2d.append((pos, used, n))
        data.append(bytes(rng.getrandbits(8) for _ in range(n)))
        pos += n
        used += -(-n // 16) * 16
    assert pos < ARENA // 2 and used <= WINDOW
    # Device copies write the last quarter; unchained ones read the third.
    dpos, spos = 3 * ARENA // 4, ARENA // 2
    for i in range(spec["d2d"]):
        if spec.get("chained") and h2d:
            dst_h, _, n_h = h2d[i % len(h2d)]
            off = rng.randint(0, n_h - 1)
            src, n = dst_h + off, n_h - off
        else:
            n = rng.choice([4, 8, 16, 32, 6, 64])
            spos += rng.randint(0, 24)
            src = spos
            spos += n
        dpos += rng.randint(0, 24)
        d2d.append((dpos, src, n))
        dpos += n
    assert dpos < ARENA and spos < 3 * ARENA // 4
    return h2d, d2d, data


def _window(layout, base: int) -> torch.Tensor:
    """The window of ``layout`` for an arena at device address ``base``."""
    h2d, d2d, data = layout
    w = torch.zeros(WINDOW, dtype=torch.uint8)
    words = [len(h2d), len(d2d)]
    for dst, src, n in h2d:
        words += [base + dst, src, n]
    for dst, src, n in d2d:
        words += [base + dst, base + src, n]
    w[: 8 * len(words)] = torch.tensor(words, dtype=torch.int64).view(torch.uint8)
    for (_, off, n), b in zip(h2d, data):
        w[off : off + n] = torch.frombuffer(bytearray(b), dtype=torch.uint8)
    return w.cuda()


def _arenas(seed):
    a = torch.randint(0, 256, (ARENA,), generator=gen(seed), device="cuda", dtype=torch.int32)
    a = a.to(torch.uint8)
    return a, a.clone()


@pytest.mark.parametrize("with_epoch", [True, False], ids=["epoch", "no_epoch"])
@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, with_epoch, certifying) -> None:  # noqa: F811 — pytest resolves the fixture by this name
    """Byte for byte the default prologue's destinations, header clear and epoch."""
    seed = seed_of(cell.why)
    layout = _layout(cell.spec, seed)
    arena, ref_arena = _arenas(seed)
    window, ref_window = _window(layout, arena.data_ptr()), _window(layout, ref_arena.data_ptr())
    epoch = torch.full((1,), 41, dtype=torch.int64, device="cuda") if with_epoch else None
    ref_epoch = epoch.clone() if with_epoch else None
    certifying(op)
    op(window, epoch, MAX_DESC)
    op.reference(ref_window, ref_epoch, MAX_DESC)
    torch.cuda.synchronize()
    assert torch.equal(arena, ref_arena)
    # Each window holds its own arena's addresses: past the consumed header, both are left as staged.
    for w, a in ((window, arena), (ref_window, ref_arena)):
        assert not w[:16].any(), "the header is consumed"
        assert torch.equal(w[16:], _window(layout, a.data_ptr())[16:])
    if with_epoch:
        assert int(epoch.item()) == int(ref_epoch.item()) == 42


def test_graph_replays_scatter_each_new_window() -> None:
    """Captured once on a static window: every replay writes that replay's window and bumps the epoch once."""
    arena, ref_arena = _arenas(7)
    window = torch.zeros(WINDOW, dtype=torch.uint8, device="cuda")
    epoch = torch.zeros(1, dtype=torch.int64, device="cuda")
    ref_epoch = epoch.clone()
    op(window, epoch, MAX_DESC)  # the empty window: writes nothing
    op.reference(window.clone(), ref_epoch, MAX_DESC)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        op(window, epoch, MAX_DESC)
    for step, spec in enumerate(
        [dict(h2d=24, d2d=8, chained=True), dict(h2d=0, d2d=0), dict(h2d=40, d2d=4)]
    ):
        layout = _layout(spec, 100 + step)
        window.copy_(_window(layout, arena.data_ptr()))
        graph.replay()
        op.reference(_window(layout, ref_arena.data_ptr()), ref_epoch, MAX_DESC)
        torch.cuda.synchronize()
        assert torch.equal(arena, ref_arena), step
        assert not window[:16].any(), step
        assert int(epoch.item()) == int(ref_epoch.item()) == step + 2, step


def _pack_case(prologue_op):
    pack = StepInputPack(
        torch.device("cuda", torch.cuda.current_device()),
        parallel_scatter=True,
        in_graph=True,
        early_copy=True,
        step_epoch=True,
        prologue_op=prologue_op,
    )
    a, b, c = (torch.zeros(16, dtype=torch.int32, device="cuda") for _ in range(3))
    out = torch.zeros(48, dtype=torch.int32, device="cuda")
    graph = torch.cuda.CUDAGraph()
    with step_inputs.active(pack):
        torch.cuda.synchronize()
        with torch.cuda.graph(graph):
            assert step_inputs.capture_prologue() is pack
            out.copy_(torch.cat([a, b, c]))
    return pack, graph, (a, b, c), out


def _pack_step(pack, graph, dsts, values, overlap=False, flush=True):
    a, b, c = dsts

    def host(v):
        return torch.tensor(v, dtype=torch.int32).pin_memory()

    with step_inputs.active(pack):
        step_inputs.begin()
        step_inputs.copy_(a, host(values[0]))
        step_inputs.copy_(b, host(values[1]))
        if overlap:
            step_inputs.copy_(a[:4], host([-1, -2, -3, -4]))
        step_inputs.copy_(c, a)  # a device copy reading a host copy's destination
        if not flush:
            step_inputs.discard()
        step_inputs.flush(in_graph=True)
    graph.replay()


def test_pack_with_the_one_launch_prologue_matches_the_default() -> None:
    """The same pack steps through the default prologue and the one-launch one: the same outputs and epochs, and the
    one-launch prologue is engaged (the default's warm-up compile is not what bumped its epoch)."""
    packs = [_pack_case(None), _pack_case(op)]
    assert packs[0][0].prologue_op is None and packs[1][0].prologue_op is op
    for step in range(6):
        va = [step * 100 + i for i in range(16)]
        vb = [step * 1000 - i for i in range(16)]
        outs = []
        for pack, graph, dsts, out in packs:
            # A replay with a new pack, one with an eagerly issued pack, and one with nothing to scatter.
            _pack_step(pack, graph, dsts, (va, vb), overlap=step == 2, flush=step != 3)
            torch.cuda.synchronize()
            outs.append((out.tolist(), int(pack.epoch.item())))
        assert outs[0] == outs[1], step
        assert outs[1][1] == step + 2, step  # one bump at warm-up, then one per replay


def _kernel_nodes_launch_plainly(capture) -> list:
    """The PDL attributes of every kernel node of a graph ``capture(stream)`` records: the step-input writers must be
    plain launches (MSA pre-wait contract: no programmatic serialization or event on them)."""
    from cuda.bindings import runtime as rt

    graph = torch.cuda.CUDAGraph(keep_graph=True)
    with torch.cuda.graph(graph):
        capture()
    raw = rt.cudaGraph_t(init_value=graph.raw_cuda_graph())
    # cuda-python changed the signature: (graph[, numNodes]) on older bindings, (graph, nodes, numNodes) on newer ones.
    try:
        err, _, n = rt.cudaGraphGetNodes(raw)
        err, nodes, n = rt.cudaGraphGetNodes(raw, n)
    except TypeError:
        err, _, n = rt.cudaGraphGetNodes(raw, None, 0)
        err, nodes, n = rt.cudaGraphGetNodes(raw, None, n)
    found = []
    for node in nodes[:n]:
        err, kind = rt.cudaGraphNodeGetType(node)
        if kind != rt.cudaGraphNodeType.cudaGraphNodeTypeKernel:
            continue
        err, attr = rt.cudaGraphKernelNodeGetAttribute(
            node, rt.cudaLaunchAttributeID.cudaLaunchAttributeProgrammaticStreamSerialization
        )
        # Older bindings return no attribute for a kernel launched without one (a plain launch).
        found.append(0 if attr is None else int(attr.programmaticStreamSerializationAllowed))
    return found


def test_the_prologue_launches_plainly() -> None:
    window = torch.zeros(WINDOW, dtype=torch.uint8, device="cuda")
    epoch = torch.zeros(1, dtype=torch.int64, device="cuda")
    op(window, epoch, MAX_DESC)
    torch.cuda.synchronize()
    flags = _kernel_nodes_launch_plainly(lambda: op(window, epoch, MAX_DESC))
    assert flags == [0], flags
