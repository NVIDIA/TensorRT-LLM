# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi K3 Lamport mailbox layer: parity schedule, standalone all-reduce bitwise vs the MNNVL
summation order (and MNNVLAllReduce itself), emulated larger groups, NaN/Inf rows, CUDA-graph
replays interleaved with eager forwards."""

import os
import pickle
import sys
import traceback

import cloudpickle
import pytest
import torch
from mpi4py import MPI

import tensorrt_llm
from tensorrt_llm._torch.distributed.k3_lamport_mailbox import (
    SENTINEL_I32,
    K3LamportSchedule,
    lamport_parities,
)
from tensorrt_llm.mapping import Mapping

cloudpickle.register_pickle_by_value(sys.modules[__name__])
MPI.pickle.__init__(
    cloudpickle.dumps,
    cloudpickle.loads,
    pickle.HIGHEST_PROTOCOL,
)

pytestmark = pytest.mark.threadleak(enabled=False)

HIDDEN = 7168
BITWISE_M = (1, 3, 6, 16, 33, 64, 128)
# Polls trap after this many failed passes (tens of seconds), so a hang fails the test.
SPIN_LIMIT = 1 << 26


def _blackwell_or_rubin():
    return torch.cuda.is_available() and torch.cuda.get_device_capability() in (
        (10, 0),
        (10, 3),
        (10, 7),
    )


@pytest.mark.parametrize("num_uses", list(range(2, 40)))
def test_lamport_parities_any_forward_order_is_safe(num_uses):
    parities = lamport_parities(num_uses)
    assert len(parities) == num_uses
    assert parities[0] == 0 and parities[-1] != 0
    assert all(a != b for a, b in zip(parities, parities[1:]))
    # Any forward may follow any other: the boundary pair (last, first) differs too.
    other = lamport_parities(num_uses + 1)
    assert parities[-1] != other[0] and other[-1] != parities[0]


def test_lamport_parities_rejects_a_single_use():
    with pytest.raises(ValueError):
        lamport_parities(1)
    assert lamport_parities(0) == ()


def test_lamport_schedule_counts_uses():
    schedule = K3LamportSchedule(4)
    assert [schedule.take() for _ in range(4)] == [0, 1, 2, 1]
    schedule.finish()
    with pytest.raises(RuntimeError):
        schedule.take()
    short = K3LamportSchedule(3)
    short.take()
    with pytest.raises(RuntimeError):
        short.finish()


def test_sentinel_matches_the_device_primitives():
    pytest.importorskip("cutlass")
    from tensorrt_llm._torch.cute_dsl_kernels.rubin import k3_lamport

    assert k3_lamport.SENTINEL_I32 == SENTINEL_I32
    assert k3_lamport.SENTINEL == SENTINEL_I32 & 0xFFFFFFFF


def _payloads(seed, world, m, h, device):
    """[world, m, h] BF16 with +-0, subnormals, large magnitudes and pairwise cancellations."""
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    shape = (world, m, h)
    kw = dict(generator=g, device=device, dtype=torch.int32)
    bits = (torch.randn(shape, generator=g, device=device) * 4.0).to(torch.bfloat16).view(
        torch.int16
    ).to(torch.int32) & 0xFFFF
    sel = torch.randint(0, 100, shape, **kw)
    sign = torch.randint(0, 2, shape, **kw) << 15
    mant = torch.randint(0, 128, shape, **kw)
    expo = torch.randint(0xC0, 0xF1, shape, **kw) << 7
    bits = torch.where(sel < 4, torch.zeros_like(bits), bits)
    bits = torch.where((sel >= 4) & (sel < 8), torch.full_like(bits, 0x8000), bits)
    bits = torch.where((sel >= 8) & (sel < 16), sign | mant.clamp(min=1), bits)
    bits = torch.where((sel >= 16) & (sel < 20), sign | expo | mant, bits)
    pair = bits.clone()
    pair[1::2] = bits[0::2] ^ 0x8000
    bits = torch.where((sel >= 20) & (sel < 24), pair, bits)
    bits = torch.where(bits >= 0x8000, bits - 0x10000, bits).to(torch.int16)
    return bits.view(torch.bfloat16)


def _mnnvl_order_sum(payloads, chunk):
    acc = torch.zeros(payloads.shape[1:], dtype=torch.float32, device=payloads.device)
    for first in range(0, payloads.shape[0], chunk):
        part = torch.zeros_like(acc)
        for r in range(first, first + chunk):
            part = part + payloads[r].float()
        acc = acc + part
    return acc.to(torch.bfloat16)


def _bad(a, b, nan_equal=False):
    diff = a.view(torch.int16) != b.view(torch.int16)
    if nan_equal:  # torch rounds NaN to 0x7fc0, the device to the canonical 0x7fff
        diff &= ~(torch.isnan(a) & torch.isnan(b))
    return int(diff.sum().item())


class _EagerUses:
    """Parities of back-to-back single calls: consecutive calls differ (0, 1, 2, 0, ...)."""

    def __init__(self):
        self.count = 0

    def next(self):
        parity = self.count % 3
        self.count += 1
        return parity


def _all_zero(comm, bad, what):
    """Assert on every rank at once: a rank failing alone would leave its peers waiting."""
    per_rank = comm.allgather(bad)
    assert sum(per_rank) == 0, f"{what}: mismatches per rank {per_rank}"


def _bitwise_phase(tp, rank, arena, mnnvl, device, comm, uses):
    from tensorrt_llm._torch.cute_dsl_kernels.rubin import k3_lamport_allreduce as kern
    from tensorrt_llm._torch.cute_dsl_kernels.rubin.k3_lamport import oneshot_chunk, twoshot_chunk
    from tensorrt_llm._torch.distributed.k3_lamport_mailbox import K3LamportAllReduce
    from tensorrt_llm.functional import AllReduceFusionOp, AllReduceParams

    plain = AllReduceParams(fusion_op=AllReduceFusionOp.NONE)
    ran = skipped = 0
    for world in (tp, 2 * tp, 4 * tp):
        ar = K3LamportAllReduce(arena, HIDDEN, world=world, spin_limit=SPIN_LIMIT)
        # Fabric arenas map every rank at one stride; without fabric handles there is no two-shot.
        assert ar.twoshot_available == arena.fabric
        ar.compile()
        comm.Barrier()
        v = world // tp
        modes = [kern.MODE_ONESHOT] + ([kern.MODE_TWOSHOT] if ar.twoshot_available else [])
        for mode in modes:
            chunk = oneshot_chunk(world) if mode == kern.MODE_ONESHOT else twoshot_chunk(world)
            for m in BITWISE_M:
                if not ar.supports(m, mode):
                    # Only one-shot rows of emulated groups can outgrow the region.
                    assert mode == kern.MODE_ONESHOT and world > tp, (world, mode, m)
                    skipped += 1
                    continue
                pay = _payloads(1009 * world + 17 * m + len(mode), world, m, HIDDEN, device)
                x = pay[rank::tp].contiguous()
                out = ar(x if v > 1 else x[0], uses.next(), mode=mode)
                bad = _bad(out, _mnnvl_order_sum(pay, chunk))
                if world == tp and mnnvl is not None:
                    bad += _bad(out, mnnvl(x[0], all_reduce_params=plain))
                _all_zero(comm, bad, f"world {world} {mode} m {m}")
                ran += 1
    print(f"[rank {rank}] bitwise cases run {ran}, one-shot rows over the region {skipped}")
    # One-shot: 7 + 6 + 5 cases fit a 9 MiB region at 4 / 8 / 16 ranks; two-shot: all 21.
    assert ran == 18 + (21 if arena.fabric else 0), ran


def _special_values_phase(tp, rank, arena, mnnvl, device, comm, uses):
    """NaN and +-Inf in the payloads: bitwise vs MNNVLAllReduce, and vs the reference up to the
    NaN encoding."""
    from tensorrt_llm._torch.distributed.k3_lamport_mailbox import K3LamportAllReduce
    from tensorrt_llm.functional import AllReduceFusionOp, AllReduceParams

    ar = K3LamportAllReduce(arena, HIDDEN, spin_limit=SPIN_LIMIT)
    for m in (2, 24, 96):
        pay = _payloads(77 + m, tp, m, HIDDEN, device)
        bits = pay.view(torch.int16)
        bits[0, :, 0:8] = 0x7FC0  # NaN on rank 0
        bits[1 % tp, :, 8:16] = 0x7F80  # +Inf on rank 1
        bits[2 % tp, :, 16:24] = -128  # -Inf (0xff80) on rank 2
        bits[:, :, 24:32] = 0x7F80  # +Inf on every rank
        bits[tp - 1, :, 32:40] = 0x7F7F  # BF16 max: the sum overflows to Inf
        x = pay[rank].contiguous()
        out = ar(x, uses.next())
        bad = _bad(out, _mnnvl_order_sum(pay, tp), nan_equal=True)
        if mnnvl is not None:
            mn = mnnvl(x, all_reduce_params=AllReduceParams(fusion_op=AllReduceFusionOp.NONE))
            bad += _bad(out, mn)
        _all_zero(comm, bad, f"special values m {m}")


def _graph_phase(tp, rank, arena, device, comm):
    """CUDA graphs of two sizes replayed in a mixed order, with an eager two-use forward between
    replays (no host sync in between), fresh payloads every time."""
    from tensorrt_llm._torch.cute_dsl_kernels.rubin.k3_lamport import oneshot_chunk, twoshot_chunk
    from tensorrt_llm._torch.distributed.k3_lamport_mailbox import K3LamportAllReduce

    ar = K3LamportAllReduce(arena, HIDDEN, spin_limit=SPIN_LIMIT)

    def chunk(m):
        return oneshot_chunk(tp) if ar.mode_for(m) == "oneshot" else twoshot_chunk(tp)

    def forward(xs, outs):
        schedule = arena.begin_forward(len(xs))
        for x, out in zip(xs, outs):
            ar(x, schedule.take(), out=out)
        schedule.finish()

    graphs = []
    for m, uses in ((6, 4), (40, 5)):  # 40 rows: two-shot at the 1 MiB default
        xs = [torch.empty((m, HIDDEN), dtype=torch.bfloat16, device=device) for _ in range(uses)]
        outs = [torch.empty_like(xs[0]) for _ in range(uses)]
        forward(xs, outs)  # eager warmup
        torch.cuda.synchronize()
        comm.Barrier()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            forward(xs, outs)
        graphs.append((graph, xs, outs, m))
    torch.cuda.synchronize()
    comm.Barrier()
    eager_x = [torch.empty((3, HIDDEN), dtype=torch.bfloat16, device=device) for _ in range(2)]
    eager_out = [torch.empty_like(eager_x[0]) for _ in range(2)]
    for it in range(60):
        graph, xs, outs, m = graphs[(it * 7 // 3) % 2]
        refs = []
        for u, xu in enumerate(xs):
            pay = _payloads(31 * it + u, tp, m, HIDDEN, device)
            xu.copy_(pay[rank])
            refs.append(_mnnvl_order_sum(pay, chunk(m)))
        eager_refs = []
        for u, xu in enumerate(eager_x):
            pay = _payloads(100003 + 31 * it + u, tp, 3, HIDDEN, device)
            xu.copy_(pay[rank])
            eager_refs.append(_mnnvl_order_sum(pay, chunk(3)))
        graph.replay()
        forward(eager_x, eager_out)
        bad = sum(_bad(o, r) for o, r in zip(outs + eager_out, refs + eager_refs))
        _all_zero(comm, bad, f"graph replay {it}")
    torch.cuda.synchronize()
    comm.Barrier()
    _all_zero(comm, arena.unarmed_words(), "unarmed words after the replays")


def _lamport_rank_body(tp, rank):
    from tensorrt_llm._torch.distributed.k3_lamport_mailbox import K3LamportArena
    from tensorrt_llm._torch.distributed.ops import MNNVLAllReduce

    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    comm = MPI.COMM_WORLD
    mapping = Mapping(world_size=tp, rank=rank, gpus_per_node=tp, tp_size=tp)
    arena = K3LamportArena.get(mapping)
    assert K3LamportArena.get(mapping) is arena
    assert not arena.fabric or arena.peer_stride is not None
    mnnvl = None
    if MNNVLAllReduce.is_mnnvl(mapping, torch.bfloat16):
        mnnvl = MNNVLAllReduce(mapping, torch.bfloat16)
    uses = _EagerUses()
    _bitwise_phase(tp, rank, arena, mnnvl, device, comm, uses)
    _special_values_phase(tp, rank, arena, mnnvl, device, comm, uses)
    torch.cuda.synchronize()
    comm.Barrier()
    _graph_phase(tp, rank, arena, device, comm)
    return True


def _run_rank(tp, force_fabric):
    rank = tensorrt_llm.mpi_rank()
    env = {"TLLM_TEST_MNNVL": "1"}
    if force_fabric:  # fabric handles on one node, as across nodes (two-shot available)
        env["TRTLLM_FORCE_MNNVL_AR"] = "1"
    saved = {k: os.environ.get(k) for k in env}
    os.environ.update(env)  # spawned workers may not inherit the parent's environment
    try:
        return _lamport_rank_body(tp, rank)
    except Exception:
        traceback.print_exc()
        raise
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


@pytest.mark.skipif(
    torch.cuda.device_count() < 4 or not _blackwell_or_rubin(),
    reason="needs 4 Blackwell/Rubin GPUs with multicast",
)
@pytest.mark.parametrize("mpi_pool_executor", [4], indirect=True)
def test_k3_lamport_allreduce_bitwise(mpi_pool_executor):
    tp = mpi_pool_executor.num_workers
    force_fabric = os.environ.get("TRTLLM_FORCE_MNNVL_AR", "0") == "1"
    results = mpi_pool_executor.map(_run_rank, [tp] * tp, [force_fabric] * tp)
    for r in results:
        assert r is True
