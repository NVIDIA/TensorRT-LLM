# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The attention-DP MoE round trip's transport switch (`DP_MOE_COMM_ENV`).

`nccl` (the default) keeps the all-gather / reduce-scatter pair; `lamport` and `lamport_step` send
every round trip the Lamport transport `takes` through `dep_lamport_round_trip` and every other one
through NCCL, the same branch on every rank. The build serves dep4 (symmetric memory) and dep8
(`PeerBuffers`), and agrees on a failure across the group; `lamport_step`'s launch indices are
counted by `LamportSteps` over regions the shell's forward opens. The collectives are recorders;
no kernel runs.
"""

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_10x__default import (  # noqa: E501
    modeling as M,
)

pytestmark = pytest.mark.cpu_only


@pytest.mark.parametrize(
    "value, expected",
    [
        (None, "nccl"),
        ("", "nccl"),
        ("NCCL", "nccl"),
        (" lamport ", "lamport"),
        ("Lamport_Step", "lamport_step"),
    ],
)
def test_switch_values(monkeypatch, value, expected):
    if value is None:
        monkeypatch.delenv(M.DP_MOE_COMM_ENV, raising=False)
    else:
        monkeypatch.setenv(M.DP_MOE_COMM_ENV, value)
    assert M._dp_moe_comm() == expected


def test_switch_refuses_another_value(monkeypatch):
    monkeypatch.setenv(M.DP_MOE_COMM_ENV, "nvls")
    with pytest.raises(ValueError, match="one of nccl, lamport, lamport_step"):
        M._dp_moe_comm()


class _Trip:
    """A stand-in transport that takes up to `max_rows` rows a rank."""

    def __init__(self, max_rows: int):
        self.max_rows = max_rows

    def takes(self, rows: int, x: torch.Tensor) -> bool:
        return 1 <= rows <= self.max_rows


def _round_trip(monkeypatch, comm: str, max_rows: int = 4):
    calls = []
    monkeypatch.setenv(M.DP_MOE_COMM_ENV, comm)
    monkeypatch.setattr(
        M, "allgather", lambda x, *a: calls.append(("nccl gather", x.shape[0])) or x.repeat(4, 1)
    )
    monkeypatch.setattr(
        M,
        "reducescatter",
        lambda x, *a: calls.append(("nccl reduce", x.shape[0])) or x[: x.shape[0] // 4],
    )

    def lamport(x, trip, reduce):
        calls.append(("lamport reduce" if reduce else "lamport gather", x.shape[0]))
        return x[: x.shape[0] // 4] if reduce else x.repeat(4, 1)

    monkeypatch.setattr(M, "dep_lamport_round_trip", lamport)
    core = SimpleNamespace(
        dp_size=4,
        dp_group=[0, 1, 2, 3],
        mapping=SimpleNamespace(tp_group_pg=None),
        _dp_lamport=lambda group: _Trip(max_rows),
    )
    return M.Qwen4ExpDefaultModelingV2._expert_sum_op(core), calls


@pytest.mark.parametrize(
    "comm, rows, pad, expected",
    [
        ("nccl", 3, 1, [("nccl gather", 4), ("nccl reduce", 16)]),
        ("lamport", 3, 1, [("lamport gather", 4), ("lamport reduce", 16)]),
        ("lamport_step", 3, 1, [("lamport gather", 4), ("lamport reduce", 16)]),
        ("lamport", 5, 0, [("nccl gather", 5), ("nccl reduce", 20)]),  # past the transport's rows
    ],
)
def test_round_trip_transport(monkeypatch, comm, rows, pad, expected):
    expert_sum, calls = _round_trip(monkeypatch, comm)
    x = torch.arange(rows * 8, dtype=torch.float32).reshape(rows, 8)
    out = expert_sum(lambda gathered: gathered, x, pad)
    assert calls == expected
    assert out.shape == x.shape and torch.equal(out, x)


# ------------------------------------------------------------------ the build (`_dp_lamport`)


def _core(monkeypatch, world: int, capability=(10, 3), comm="lamport_step"):
    monkeypatch.setenv(M.DP_MOE_COMM_ENV, comm)
    monkeypatch.setattr(M.torch.cuda, "get_device_capability", lambda *a: capability)
    monkeypatch.setattr(M.torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(M.torch.cuda, "synchronize", lambda *a: None)
    core = SimpleNamespace(
        dp_size=world,
        dp_rank=1,
        hidden=8,
        max_num_tokens=16,
        mapping=SimpleNamespace(world_size=world),
        _lamport_trip=None,
        _lamport_steps=None,
    )
    for name in ("_dp_lamport_symmetric", "_dp_lamport_peers"):
        setattr(core, name, getattr(M.Qwen4ExpDefaultModelingV2, name).__get__(core))
    return core


@pytest.mark.parametrize(
    "world, capability", [(2, (10, 3)), (16, (10, 3)), (4, (10, 7)), (8, (10, 0))]
)
def test_build_refuses_another_layout(monkeypatch, world, capability):
    core = _core(monkeypatch, world, capability)
    with pytest.raises(ValueError, match="dep4 or dep8 worker on SM103"):
        M.Qwen4ExpDefaultModelingV2._dp_lamport(core, list(range(world)))


def test_layouts():
    assert M._LAMPORT_LAYOUTS == {4: "dep4", 8: "dep8"}


class _Peers:
    peer_ptrs = [0x1000 * (r + 1) for r in range(8)]


class _Dist:
    """`Distributed.get(mapping)` with the other ranks' errors."""

    def __init__(self, others):
        self.others = others

    def tp_allgather(self, value):
        return [value] + list(self.others)


def _peers_path(monkeypatch, create, others=("",) * 7, build_error=None):
    from tensorrt_llm._torch.distributed import communicator, peer_buffers

    built = []
    monkeypatch.setattr(peer_buffers.PeerBuffers, "create", classmethod(lambda cls, m, n: create))
    monkeypatch.setattr(communicator.Distributed, "get", staticmethod(lambda m: _Dist(others)))
    monkeypatch.setattr(M, "dep_lamport_buffers_from_pointers", lambda ptrs, *a: list(ptrs))

    def build_on(hidden, rows, rank, world, buffers, steps=None, keepalive=None):
        if build_error is not None:
            raise build_error
        trip = SimpleNamespace(max_rows=rows, buffers=buffers, steps=steps, keepalive=keepalive)
        built.append((rank, world, trip))
        return trip

    monkeypatch.setattr(M, "dep_lamport_build_on", build_on)
    return built


@pytest.mark.parametrize("comm", ["lamport", "lamport_step"])
def test_dep8_builds_on_peer_buffers(monkeypatch, comm):
    core = _core(monkeypatch, 8, comm=comm)
    peers = _Peers()
    built = _peers_path(monkeypatch, peers)
    trip = M.Qwen4ExpDefaultModelingV2._dp_lamport(core, list(range(8)))
    assert built == [(1, 8, trip)] and trip.keepalive is peers
    assert trip.buffers == peers.peer_ptrs
    assert (core._lamport_steps is not None) == (comm == "lamport_step")
    assert trip.steps is core._lamport_steps
    # Once a load: a second build returns the same workspace and allocates nothing.
    assert M.Qwen4ExpDefaultModelingV2._dp_lamport(core, list(range(8))) is trip
    assert len(built) == 1


def test_dep8_without_peer_memory_raises(monkeypatch):
    core = _core(monkeypatch, 8)
    _peers_path(monkeypatch, None)
    with pytest.raises(RuntimeError, match="no peer memory"):
        M.Qwen4ExpDefaultModelingV2._dp_lamport(core, list(range(8)))


@pytest.mark.parametrize("mine", [None, RuntimeError("compile failed")])
def test_dep8_build_failure_raises_on_every_rank(monkeypatch, mine):
    """One rank's failure (this one's, or another rank's reported through the gather) raises
    here too, instead of leaving the others polling."""
    core = _core(monkeypatch, 8)
    others = ("",) * 6 + (("" if mine else "RuntimeError: rank 7 failed"),)
    _peers_path(monkeypatch, _Peers(), others=others, build_error=mine)
    with pytest.raises(RuntimeError, match="failed on 1 rank"):
        M.Qwen4ExpDefaultModelingV2._dp_lamport(core, list(range(8)))
    assert core._lamport_trip is None


# ------------------------------------------------------------------ `lamport_step` regions


class _StepTrip:
    """A host-index workspace whose launches and advances are recorded."""

    def __init__(self, log, name):
        self.log, self.name = log, name
        self.state = object()

    def advance(self, n, stream):
        self.log.append(("advance", self.name, n % 3))


def test_steps_count_per_workspace_and_advance_at_the_outermost_end():
    log = []
    steps = M.LamportSteps()
    a, b = _StepTrip(log, "a"), _StepTrip(log, "b")
    with steps.region():
        assert [steps.next(a, "s") for _ in range(2)] == [0, 1]
        with steps.region():  # nested: counts into the outermost
            assert steps.next(a, "s") == 2 and steps.next(b, "s") == 0
        assert log == []
        assert steps.next(a, "s") == 3
    # a: 4 launches -> advance by 1; b: 1 launch -> by 1.
    assert log == [("advance", "a", 1), ("advance", "b", 1)]
    log.clear()
    with steps.region():
        for _ in range(3):
            steps.next(a, "s")
    assert log == []  # a multiple of 3 needs no advance
    assert steps.next(a, "s") is None  # outside any region


def test_steps_region_that_raises_in_capture_advances_nothing(monkeypatch):
    log = []
    steps = M.LamportSteps()
    a = _StepTrip(log, "a")
    monkeypatch.setattr(M.torch.cuda, "is_current_stream_capturing", lambda: True)
    with pytest.raises(KeyError):
        with steps.region():
            steps.next(a, "s")
            raise KeyError("capture failed")
    assert log == [] and steps.depth == 0
    monkeypatch.setattr(M.torch.cuda, "is_current_stream_capturing", lambda: False)
    with pytest.raises(KeyError):
        with steps.region():
            steps.next(a, "s")
            raise KeyError("eager step failed")
    assert log == [("advance", "a", 1)]  # what ran eagerly keeps the slots in step


def test_launch_outside_a_region_advances_by_one(monkeypatch):
    from tensorrt_llm._torch.modeling_v2.models.qwen4_exp.ops.moe import (
        dep_lamport_round_trip as op,
    )

    launches = []
    monkeypatch.setattr(op, "_raw_stream", lambda index: "stream")
    steps = op.LamportSteps()
    trip = op.DepRoundTrip(
        hidden=8,
        max_rows=4,
        rank=0,
        world=4,
        buffer=None,
        peer_buffers=[],
        state=torch.zeros(6, dtype=torch.int32),
        gather_fn=lambda x, out, peers, state, index, stream: launches.append(("gather", index)),
        reduce_fn=lambda y, out, peers, state, index, stream: launches.append(("reduce", index)),
        steps=steps,
        advance_fn=lambda state, n, stream: launches.append(("advance", n)),
    )
    x = torch.zeros(2, 8, dtype=torch.bfloat16)
    trip.gather(x)
    assert launches == [("gather", 0), ("advance", 1)]
    launches.clear()
    with steps.region():
        trip.gather(x)
        trip.reduce_scatter(torch.zeros(8, 8, dtype=torch.bfloat16))
    assert launches == [("gather", 0), ("reduce", 1), ("advance", 2)]


@pytest.mark.parametrize("comm", ["lamport", "lamport_step"])
def test_shell_step_is_one_region(comm):
    """With `lamport_step` the shell's whole step runs inside one region of the core's registry."""
    seen = []
    steps = M.LamportSteps() if comm == "lamport_step" else None

    def step(*args, **kwargs):
        seen.append((steps.depth if steps is not None else None, args[0], kwargs))
        return "out"

    shell = SimpleNamespace(model=SimpleNamespace(_lamport_steps=steps), _step=step)
    out = M.ModelingV2Qwen38FlashNextNvfp4Sm10xDefault.forward(shell, "md", spec_metadata="sm")
    assert out == "out"
    assert seen == [(1 if steps is not None else None, "md", {})]
