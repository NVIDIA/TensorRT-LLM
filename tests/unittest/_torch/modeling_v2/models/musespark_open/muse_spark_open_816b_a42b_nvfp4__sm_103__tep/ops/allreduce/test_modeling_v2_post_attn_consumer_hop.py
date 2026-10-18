# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Multi-GPU test for the musespark_open post_attn_consumer_hop entry (``attention_output_into_hop``).

The entry claims to be ``post_attn_hop`` bit for bit when every rank's
partial was published into its slot of the direct transport by the partial's
producer. Every check below publishes the same partials both ways -- the
producer writing into ``PostAttnDirectWorkspace.slot`` (a copy kernel, or the
real producer: cuBLAS ``torch.mm(out=slot)``), then the consumer hop; and the
partial in a local tensor, then the fused hop on its own workspace -- and compares
both outputs (FP32 residual, BF16 prenorm) as bits. The cells are also gated
against the entry's reference and the built-in pair (MNNVL all-reduce +
cluster boundary). Every sequence of calls alternates the buffer parity, as the
target does (layer ``l`` uses ``l mod 2``).

Also: a concurrent poller that records the first value of every 16-bit half of
every slot while cuBLAS writes them through the multicast address and checks it
against the final value (at least 10,000 calls, rows 1-8); the poll guard's
error word under a deliberately broken parity, and the watchdog that logs it;
a call whose consumer never ran (the hazard, then the collective ``reset``);
the producer check on a capture that publishes nothing; graph replay with
changing inputs, a delayed rank, a 200-call burst. ``MODELING_V2_TIMING=1`` also runs
the chain benchmark: norm-gate -> o_proj -> hop -> pool GEMM, 62 layers of
cold weights in one CUDA graph, the hop against direct publication, printed by rank 0.

A collective test, run once per rank (4 ranks on one node, or 8 on two)::

    srun -N2 --segment=2 --ntasks-per-node=4 --mpi=pmix \\
        python3 -m pytest -q -p no:cacheprovider --noconftest \\
        -o addopts="-p test_common.magic_import_hooks" <this file>

Outside such a launch it is skipped.
"""

import os
import statistics
import zlib

import pytest
import torch
import triton
import triton.language as tl

from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.ops.allreduce import (  # noqa: E501
    post_attn_hop as hop_module,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.ops.allreduce.post_attn_hop import (  # noqa: E501
    DIRECT_SENTINEL,
    HIDDEN,
    MAX_RANKS,
    MAX_ROWS,
    PostAttnDirectWorkspace,
    PostAttnHopWorkspace,
    consumer_hop_supported,
    publish_copy,
    validated_producer,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.ops.allreduce.post_attn_hop import (  # noqa: E501
    post_attn_consumer_hop as op,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.ops.allreduce.post_attn_hop import (  # noqa: E501
    post_attn_hop as hop_op,
)

__extra_import_path__ = ["../../../../.."]
from _validating import (  # noqa: E402,F401 — path above; fixture by name
    assert_within_ulp,
    certifying,
)

POST_EPS = 1e-8
PRE_EPS = 1e-5
O_PROJ_K = 1024
TRANSPORTS = ("direct", "direct_early")
SENTINEL_I32 = DIRECT_SENTINEL - (1 << 32)
#: Calls of the concurrent-poller test.
POLLER_CALLS = int(os.environ.get("MODELING_V2_DIRECT_PUBLISH_CALLS", "10240"))


def _group():
    try:
        from tensorrt_llm._utils import mpi_comm

        comm = mpi_comm()
        return comm, comm.Get_size(), comm.Get_rank()
    except Exception:  # noqa: BLE001 - no MPI launch means no group
        return None, 1, 0


COMM, WORLD, RANK = _group()

pytestmark = pytest.mark.skipif(
    WORLD not in (4, 8)
    or not torch.cuda.is_available()
    or torch.cuda.get_device_capability() != (10, 3),
    reason="collective test: run under a 4- or 8-rank launch on GB300",
)


def _agree(ok: bool) -> bool:
    """True only if every rank says so (keeps ranks on the same control path)."""
    return COMM.allreduce(int(ok)) == WORLD


def _max_over_ranks(value: float) -> float:
    from mpi4py import MPI

    return COMM.allreduce(value, op=MPI.MAX)


def _coefficients(gate: torch.Tensor):
    scaled = gate.to(torch.float32) / 0.3
    beta = torch.sigmoid(scaled)
    alpha = torch.sqrt(torch.clamp(torch.sigmoid(-scaled) * (1.0 + beta), min=1e-3))
    return alpha, beta


def _cancelling(rows: int, g: torch.Generator):
    big = torch.exp2(torch.randint(0, 8, (rows, HIDDEN), generator=g).float())
    vals = torch.randn(WORLD, rows, HIDDEN, generator=g) * big * 2.0**-20
    perm = torch.argsort(torch.rand(WORLD, rows, HIDDEN, generator=g), dim=0)
    vals.scatter_(0, perm[0:1], big.unsqueeze(0))
    vals.scatter_(0, perm[1:2], -big.unsqueeze(0))
    small = big * 2.0**-26 * (1.0 + torch.rand(rows, HIDDEN, generator=g))
    vals.scatter_(0, perm[2:3], small.unsqueeze(0))
    return [v.to(torch.bfloat16) for v in vals]


def _inputs(rows: int, seed: int, zeros: bool = False, cancel: bool = False):
    """Every rank's partial (rank order) and the shared residual / coefficients / weight."""
    g = torch.Generator().manual_seed(seed)
    channel = torch.exp(0.75 * torch.randn(HIDDEN, generator=g))
    partials = [
        (0.5 * torch.randn(rows, HIDDEN, generator=g) * channel).to(torch.bfloat16)
        for _ in range(WORLD)
    ]
    if zeros:
        for part in partials:
            pick = torch.rand(rows, HIDDEN, generator=g)
            part[pick < 0.08] = 0.0
            part[(pick >= 0.08) & (pick < 0.16)] = -0.0
        for part in partials:
            part[:, 0:64] = -0.0
        # Words that are the old sentinel 0x80000000 (high half -0, low half +0).
        for part in partials:
            part[:, 64:128:2] = 0.0
            part[:, 65:128:2] = -0.0
        for r in range(0, WORLD, 2):
            partials[r + 1][:, 128:192] = -partials[r][:, 128:192]
    if cancel:
        partials = _cancelling(rows, g)
    residual = (
        4.0
        * torch.randn(rows, HIDDEN, generator=g)
        * torch.exp(0.5 * torch.randn(HIDDEN, generator=g))
    )
    alpha, beta = _coefficients(1.5 * torch.randn(HIDDEN, generator=g))
    weight = (0.2 * torch.randn(HIDDEN, generator=g)).to(torch.bfloat16)
    dev = torch.device("cuda", torch.cuda.current_device())
    return (
        [p.to(dev) for p in partials],
        residual.to(dev),
        alpha.to(dev),
        beta.to(dev),
        weight.to(dev),
    )


def _bits(t: torch.Tensor) -> torch.Tensor:
    return t.view(torch.int32) if t.dtype == torch.float32 else t.view(torch.int16)


def _differing(a: torch.Tensor, b: torch.Tensor) -> int:
    return int((_bits(a) != _bits(b)).sum())


@pytest.fixture(scope="module")
def env():
    from tensorrt_llm._torch.distributed import AllReduce
    from tensorrt_llm._torch.models.musespark_open_boundary import cluster_boundary
    from tensorrt_llm.functional import AllReduceStrategy
    from tensorrt_llm.mapping import Mapping

    local = int(os.environ.get("SLURM_LOCALID", RANK % torch.cuda.device_count()))
    torch.cuda.set_device(local)
    mapping = Mapping(world_size=WORLD, rank=RANK, gpus_per_node=4, tp_size=WORLD)
    variants = [(POST_EPS, PRE_EPS, False), (POST_EPS, PRE_EPS, True)]
    hop_on = PostAttnHopWorkspace(mapping, multicast=True, variants=variants)
    direct = {
        "direct": PostAttnDirectWorkspace(mapping, variants=variants),
        "direct_early": PostAttnDirectWorkspace(mapping, variants=variants, early_trigger=True),
    }
    guarded = PostAttnDirectWorkspace(
        mapping, variants=variants[:1], guard_ns=50_000_000, guard_break=True, watchdog_s=0
    )
    aborted = PostAttnDirectWorkspace(
        mapping, variants=variants[:1], guard_ns=50_000_000, guard_break=True, watchdog_s=0
    )
    watched = PostAttnDirectWorkspace(
        mapping, variants=variants[:1], guard_ns=50_000_000, guard_break=True, watchdog_s=0.05
    )
    legacy_ar = AllReduce(mapping=mapping, strategy=AllReduceStrategy.MNNVL, dtype=torch.bfloat16)
    assert legacy_ar.mnnvl_allreduce is not None, "the built-in MNNVL all-reduce is not active"

    def legacy(partial, residual, alpha, beta, weight, norm_fp32):
        return cluster_boundary(
            residual,
            legacy_ar(partial),
            None,
            alpha,
            beta,
            weight,
            POST_EPS,
            PRE_EPS,
            norm_fp32_residual=norm_fp32,
        )

    def fused(partial, residual, alpha, beta, weight, norm_fp32=False):
        return hop_op(partial, residual, alpha, beta, weight, POST_EPS, PRE_EPS, norm_fp32, hop_on)

    COMM.Barrier()
    return dict(
        mapping=mapping,
        hop_on=hop_on,
        direct=direct,
        guarded=guarded,
        aborted=aborted,
        watched=watched,
        legacy=legacy,
        fused=fused,
    )


def _consume(ws, partial, residual, alpha, beta, weight, parity, norm_fp32=False):
    """Publish ``partial`` with a copy kernel into this rank's slot, then the consumer hop."""
    slot = ws.slot(parity, partial.shape[0])
    publish_copy(slot, partial)
    return op(slot, residual, alpha, beta, weight, POST_EPS, PRE_EPS, norm_fp32, parity, ws)


def _gemm_consume(ws, x, w, residual, alpha, beta, weight, parity, norm_fp32=False):
    """The real producer: cuBLAS writes ``x @ w.T`` into the slot; then the consumer hop."""
    slot = ws.slot(parity, x.shape[0])
    torch.mm(x, w.t(), out=slot)
    return op(slot, residual, alpha, beta, weight, POST_EPS, PRE_EPS, norm_fp32, parity, ws)


@pytest.mark.parametrize("transport", TRANSPORTS)
@pytest.mark.parametrize("cell", op.CELLS, ids=lambda c: c.why[:40].replace(" ", "_"))
def test_cell_is_the_hop_hop_bit_for_bit(env, certifying, cell, transport):  # noqa: F811
    rows = cell.spec["rows"]
    norm_fp32 = cell.spec.get("norm_fp32", False)
    parity = cell.spec.get("parity", 0)
    seed = zlib.crc32(f"direct_publish|{cell.why}|{transport}".encode())
    partials, residual, alpha, beta, weight = _inputs(
        rows, seed, cell.spec.get("zeros", False), cell.spec.get("cancel", False)
    )
    ws = env["direct"][transport]
    slot = ws.slot(parity, rows)
    assert consumer_hop_supported(slot, residual, alpha, beta, weight, parity, ws)
    certifying(op)
    if parity == 1:
        # The target never calls buffer 1 first: one call on buffer 0 before it.
        _consume(ws, partials[RANK], residual, alpha, beta, weight, 0, norm_fp32)
    publish_copy(slot, partials[RANK])
    got = op(slot, residual, alpha, beta, weight, POST_EPS, PRE_EPS, norm_fp32, parity, ws)
    if parity == 0:
        # And back to buffer 1, so every sequence on the workspace alternates.
        _consume(ws, partials[RANK], residual, alpha, beta, weight, 1, norm_fp32)
    want = env["fused"](partials[RANK], residual, alpha, beta, weight, norm_fp32)
    builtin = env["legacy"](partials[RANK], residual, alpha, beta, weight, norm_fp32)
    torch.cuda.synchronize()
    mismatch = (_differing(got[0], want[0]), _differing(got[1], want[1]))
    mismatch_builtin = (_differing(got[0], builtin[0]), _differing(got[1], builtin[1]))
    exact_updated, exact_ffn = op.reference(
        partials, residual, alpha, beta, weight, POST_EPS, PRE_EPS, norm_fp32, parity, ws
    )
    assert _agree(mismatch == (0, 0)), f"rank {RANK}: {mismatch} values differ from the fused hop"
    assert _agree(mismatch_builtin == (0, 0)), f"rank {RANK}: {mismatch_builtin} vs built-in"
    assert_within_ulp(got[1], exact_ffn, element_ulp=2.0, rms_ulp=0.05)
    assert_within_ulp(got[0], exact_updated, element_ulp=2.0, rms_ulp=0.05, ulp=2.0**-8)


@pytest.mark.parametrize("rows", range(1, MAX_ROWS + 1))
def test_a_cublas_producer_publishes_the_bits_hop_reduces(env, rows):
    """o_proj's shape: cuBLAS ``[rows, 1024] x [1024, 8192]`` straight into the slot."""
    dev = torch.device("cuda", torch.cuda.current_device())
    g = torch.Generator().manual_seed(500 + 37 * rows + RANK)
    w = (0.03 * torch.randn(HIDDEN, O_PROJ_K, generator=g)).to(torch.bfloat16).to(dev)
    bad = 0
    for parity in (0, 1):
        x = torch.randn(rows, O_PROJ_K, generator=g).to(torch.bfloat16).to(dev)
        _, residual, alpha, beta, weight = _inputs(rows, 600 + rows * 2 + parity)
        local = torch.mm(x, w.t())
        got = _gemm_consume(env["direct"]["direct"], x, w, residual, alpha, beta, weight, parity)
        want = env["fused"](local, residual, alpha, beta, weight)
        bad += _differing(got[0], want[0]) + _differing(got[1], want[1])
    torch.cuda.synchronize()
    assert _agree(bad == 0), f"rank {RANK}: {bad} values differ with a cuBLAS producer"


@pytest.mark.parametrize("transport", TRANSPORTS)
def test_a_long_burst_of_mixed_shapes_stays_identical(env, transport):
    """200 back-to-back calls, rows and producer changing per call, parity alternating."""
    ws = env["direct"][transport]
    dev = torch.device("cuda", torch.cuda.current_device())
    g = torch.Generator().manual_seed(1234)
    w = (0.03 * torch.randn(HIDDEN, O_PROJ_K, generator=g)).to(torch.bfloat16).to(dev)
    bad = 0
    for call in range(200):
        rows = int(torch.randint(1, MAX_ROWS + 1, (1,), generator=g))
        norm_fp32 = bool(torch.randint(0, 2, (1,), generator=g))
        cancel = bool(torch.randint(0, 4, (1,), generator=g) == 0)
        partials, residual, alpha, beta, weight = _inputs(rows, 10_000 + call, cancel=cancel)
        if call % 3 == 2:
            x = partials[RANK][:, :O_PROJ_K].contiguous()
            partial = torch.mm(x, w.t())
            got = _gemm_consume(ws, x, w, residual, alpha, beta, weight, call % 2, norm_fp32)
        else:
            partial = partials[RANK]
            got = _consume(ws, partial, residual, alpha, beta, weight, call % 2, norm_fp32)
        want = env["fused"](partial, residual, alpha, beta, weight, norm_fp32)
        bad += _differing(got[0], want[0]) + _differing(got[1], want[1])
    torch.cuda.synchronize()
    assert _agree(bad == 0), f"rank {RANK}: {bad} values differ over the burst"


@pytest.mark.parametrize("transport", TRANSPORTS)
def test_a_delayed_rank_is_waited_for(env, transport):
    """One rank arrives late (a GPU spin before its producer); every rank still agrees."""
    ws = env["direct"][transport]
    bad = 0
    for late in range(WORLD):
        for parity in (0, 1):
            partials, residual, alpha, beta, weight = _inputs(8, 20_000 + 2 * late + parity)
            want = env["fused"](partials[RANK], residual, alpha, beta, weight)
            if RANK == late:
                torch.cuda._sleep(2_000_000)  # ~1 ms
            got = _consume(ws, partials[RANK], residual, alpha, beta, weight, parity)
            bad += _differing(got[0], want[0]) + _differing(got[1], want[1])
    torch.cuda.synchronize()
    assert _agree(bad == 0), f"rank {RANK}: {bad} values differ with a delayed rank"


@pytest.mark.parametrize("transport", TRANSPORTS)
def test_graph_replay_with_changing_inputs(env, transport):
    """A captured sequence of cuBLAS producer + consumer pairs replays identically with new
    inputs, with eager pairs on the same workspace between replays."""
    ws = env["direct"][transport]
    dev = torch.device("cuda", torch.cuda.current_device())
    g = torch.Generator().manual_seed(4242 + RANK)
    shapes = (8, 8, 1, 4, 8, 6)  # an even count: the parity alternates across replays
    weights = [(0.03 * torch.randn(HIDDEN, O_PROJ_K, generator=g)).to(torch.bfloat16).to(dev)]
    weights = weights * len(shapes)

    def make(i, rows, seed):
        _, residual, alpha, beta, weight = _inputs(rows, seed)
        x = (torch.randn(rows, O_PROJ_K, generator=torch.Generator().manual_seed(seed))).to(
            torch.bfloat16
        )
        return [x.to(dev), residual, alpha, beta, weight]

    static = [make(i, rows, 30_000 + i) for i, rows in enumerate(shapes)]

    def run():
        return [
            _gemm_consume(ws, x, weights[i], r, a, b, w, i % 2)
            for i, (x, r, a, b, w) in enumerate(static)
        ]

    run()  # warm
    torch.cuda.synchronize()
    COMM.Barrier()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        outs = run()
    bad = 0
    for replay in range(30):
        for i, rows in enumerate(shapes):
            for dst, src in zip(static[i], make(i, rows, 40_000 + 100 * replay + i)):
                dst.copy_(src)
        graph.replay()
        for parity in (0, 1):
            partials, residual, alpha, beta, weight = _inputs(8, 50_000 + 2 * replay + parity)
            eager = _consume(ws, partials[RANK], residual, alpha, beta, weight, parity)
            want = env["fused"](partials[RANK], residual, alpha, beta, weight)
            bad += _differing(eager[0], want[0]) + _differing(eager[1], want[1])
        for i, (x, r, a, b, w) in enumerate(static):
            want = env["fused"](torch.mm(x, weights[i].t()), r, a, b, w)
            bad += _differing(outs[i][0], want[0]) + _differing(outs[i][1], want[1])
    torch.cuda.synchronize()
    assert _agree(bad == 0), f"rank {RANK}: {bad} values differ across graph replays"


def test_unsupported_inputs_are_refused_before_any_launch(env):
    ws = env["direct"]["direct"]
    partials, residual, alpha, beta, weight = _inputs(8, 7)
    slot = ws.slot(0, 8)
    with pytest.raises(ValueError):
        op(partials[RANK], residual, alpha, beta, weight, POST_EPS, PRE_EPS, False, 0, ws)
    with pytest.raises(ValueError):
        op(ws.slot(1, 8), residual, alpha, beta, weight, POST_EPS, PRE_EPS, False, 0, ws)
    with pytest.raises(ValueError):
        op(slot, residual[:4], alpha, beta, weight, POST_EPS, PRE_EPS, False, 0, ws)
    with pytest.raises(ValueError):
        op(slot, residual, alpha, beta, weight, POST_EPS, PRE_EPS, False, 2, ws)
    with pytest.raises(ValueError):
        ws.slot(0, MAX_ROWS + 1)
    # Nothing was launched: the transport is in step, one more real pair agrees.
    bad = 0
    for parity in (0, 1):
        got = _consume(ws, partials[RANK], residual, alpha, beta, weight, parity)
        want = env["fused"](partials[RANK], residual, alpha, beta, weight)
        bad += _differing(got[0], want[0]) + _differing(got[1], want[1])
    torch.cuda.synchronize()
    assert _agree(bad == 0)


def test_the_guard_word_flags_a_broken_parity_sequence(env):
    """Rank 0 consumes buffer 1 while every rank published into buffer 0: its poll never
    completes, so its guard (50 ms here, breaking) sets the guard word; the others complete.
    """
    ws = env["guarded"]
    partials, residual, alpha, beta, weight = _inputs(8, 77)
    slot = ws.slot(0, 8)
    publish_copy(slot, partials[RANK])
    parity = 1 if RANK == 0 else 0
    torch.cuda.synchronize()
    COMM.Barrier()
    got = op(
        ws.slot(parity, 8), residual, alpha, beta, weight, POST_EPS, PRE_EPS, False, parity, ws
    )
    fired = ws.guard_fired()
    want = env["fused"](partials[RANK], residual, alpha, beta, weight)
    ok = _differing(got[0], want[0]) == 0 if RANK != 0 else True
    COMM.Barrier()
    # Put the transport back: every slot of both buffers holds the sentinel again.
    ws._receive.fill_(SENTINEL_I32)
    ws.control.zero_()
    torch.cuda.synchronize()
    COMM.Barrier()
    assert _agree(fired == (RANK == 0)), f"rank {RANK}: guard fired {fired}"
    assert _agree(ok), f"rank {RANK}: the undisturbed ranks' result differs"


def test_the_watchdog_logs_a_poll_that_outlasts_the_guard(env, monkeypatch):
    """The broken parity of the test above on a workspace whose watchdog reads the pinned
    guard word every 50 ms: rank 0 logs one error while (or after) its poll waits, the
    others none."""
    import time

    ws = env["watched"]
    errors = []
    monkeypatch.setattr(hop_module.logger, "error", lambda msg, *args: errors.append(msg))
    partials, residual, alpha, beta, weight = _inputs(8, 78)
    publish_copy(ws.slot(0, 8), partials[RANK])
    parity = 1 if RANK == 0 else 0
    torch.cuda.synchronize()
    COMM.Barrier()
    op(ws.slot(parity, 8), residual, alpha, beta, weight, POST_EPS, PRE_EPS, False, parity, ws)
    deadline = time.monotonic() + 5.0
    while RANK == 0 and not errors and time.monotonic() < deadline:
        time.sleep(0.02)
    time.sleep(0.2)  # every other rank's watchdog has looked at least twice
    fired = ws.guard_fired()
    COMM.Barrier()
    ws._receive.fill_(SENTINEL_I32)
    ws.control.zero_()
    torch.cuda.synchronize()
    COMM.Barrier()
    if RANK == 0 and errors:
        print(f"\nwatchdog (rank 0): {errors[0]}")
    assert _agree(fired == (RANK == 0)), f"rank {RANK}: guard fired {fired}"
    assert _agree(len(errors) == (1 if RANK == 0 else 0)), f"rank {RANK}: {errors}"
    assert _agree(all("parities differ" in message for message in errors))


def test_an_aborted_call_is_cleared_by_the_collective_reset(env):
    """A producer published and its consumer never ran -- what a host exception between
    o_proj and its join leaves. The next call on that buffer then takes the aborted call's
    words (rank 0, whose peers publish late, must see that: the hazard is real); after
    ``reset`` every call is the fused hop bit for bit again and no guard fires."""
    ws = env["aborted"]
    dev = torch.device("cuda", torch.cuda.current_device())
    g = torch.Generator().manual_seed(7000 + RANK)
    w = (0.03 * torch.randn(HIDDEN, O_PROJ_K, generator=g)).to(torch.bfloat16).to(dev)

    def call(k, parity, consume=True, late_peers=False):
        gg = torch.Generator().manual_seed(80_000 + 13 * k + 1000 * RANK)
        x = torch.randn(8, O_PROJ_K, generator=gg).to(torch.bfloat16).to(dev)
        _, residual, alpha, beta, weight = _inputs(8, 90_000 + k)
        if not consume:
            torch.mm(x, w.t(), out=ws.slot(parity, 8))
            return 0
        if late_peers and RANK != 0:
            torch.cuda._sleep(4_000_000)  # ~2 ms
        got = _gemm_consume(ws, x, w, residual, alpha, beta, weight, parity)
        want = env["fused"](torch.mm(x, w.t()), residual, alpha, beta, weight)
        torch.cuda.synchronize()
        return _differing(got[0], want[0]) + _differing(got[1], want[1])

    _reset_transport(ws)
    before = [call(k, k % 2) for k in range(4)]
    call(4, 0, consume=False)  # the step raised after layer 0's o_proj published
    torch.cuda.synchronize()
    COMM.Barrier()
    stale = call(5, 0, late_peers=True)  # no reset: the next step starts at layer 0
    ws.reset()
    ws.control.zero_()
    after = [call(6 + k, k % 2) for k in range(8)]
    fired = ws.guard_fired()
    _reset_transport(ws)
    ws.control.zero_()
    if RANK == 0:
        print(f"\nabort: before {before}, stale call on rank 0 {stale}, after reset {after}")
    assert _agree(sum(before) == 0), f"rank {RANK}: {before}"
    assert _agree(stale > 0 if RANK == 0 else True), "rank 0 did not take the aborted words"
    assert _agree(sum(after) == 0 and not fired), f"rank {RANK}: {after}, guard {fired}"


def test_the_producer_check_reads_a_capture_and_publishes_nothing(env):
    """``check_producer`` captures the producer into a graph it never replays: cuBLAS into
    the slot is one validated kernel, and no word of any rank's copy changes; a producer
    of two kernels (a GEMM elsewhere, then the copy) and flashinfer's split-K kernel (not
    validated) are refused on every rank."""
    ws = env["direct"]["direct"]
    dev = torch.device("cuda", torch.cuda.current_device())
    g = torch.Generator().manual_seed(6100 + RANK)
    w = (0.03 * torch.randn(HIDDEN, O_PROJ_K, generator=g)).to(torch.bfloat16).to(dev)
    x = torch.randn(8, O_PROJ_K, generator=g).to(torch.bfloat16).to(dev)
    local = torch.empty(8, HIDDEN, dtype=torch.bfloat16, device=dev)
    _reset_transport(ws)

    name = ws.check_producer(
        lambda: torch.mm(x, w.t(), out=ws.slot(0, 8)), lambda: torch.mm(x, w.t(), out=local)
    )
    refused = {}
    for case, produce, warm in (
        ("two kernels", lambda: publish_copy(ws.slot(0, 8), torch.mm(x, w.t())), None),
        (
            "split-K",
            lambda: _produce("splitk", x, w, ws.slot(0, 8)),
            lambda: _produce("splitk", x, w, local),
        ),
    ):
        try:
            ws.check_producer(produce, warm)
        except ValueError as error:
            refused[case] = str(error)
    torch.cuda.synchronize()
    COMM.Barrier()
    untouched = bool((ws._receive == SENTINEL_I32).all())
    if RANK == 0:
        print(f"\nproducer check: {name}; refused {refused}")
    assert _agree(validated_producer(name)), f"rank {RANK}: {name}"
    assert _agree(sorted(refused) == ["split-K", "two kernels"]), f"rank {RANK}: {refused}"
    assert _agree(untouched), f"rank {RANK}: a capture published into the transport"


@triton.jit
def _first_seen_kernel(
    receive,
    slot_words,
    buffer_words,
    counts,
    max_rounds,
    PART: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Poll ``PART`` words of one slot (program 0 of axis 0 the first ``PART``, ...) until no
    16-bit half is 0xFFFE; record the first non-sentinel value of every half, compare it with
    a final read, reset the words.

    ``counts`` (int64): [0] halves whose first value differs from the final one, [1] halves
    seen as the sentinel by the first read (writes the poll watched arrive), [2] blocks that
    hit ``max_rounds``, [3] halves still the sentinel in those blocks.
    """
    slot = tl.program_id(1)
    lane = tl.arange(0, BLOCK)
    offsets = tl.program_id(0) * PART + lane
    mask = lane < PART
    base = receive + slot.to(tl.int64) * slot_words + buffer_words
    words = tl.load(base + offsets, mask=mask, other=0, volatile=True)
    low = words & 0xFFFF
    high = (words >> 16) & 0xFFFF
    unseen = tl.sum(
        tl.where(mask, (low == 0xFFFE).to(tl.int32) + (high == 0xFFFE).to(tl.int32), 0), axis=0
    )
    first_low = low
    first_high = high
    pending = unseen
    rounds = 0
    while (pending > 0) & (rounds < max_rounds):
        words = tl.load(base + offsets, mask=mask, other=0, volatile=True)
        low = words & 0xFFFF
        high = (words >> 16) & 0xFFFF
        first_low = tl.where(first_low == 0xFFFE, low, first_low)
        first_high = tl.where(first_high == 0xFFFE, high, first_high)
        pending = tl.sum(
            tl.where(
                mask, (first_low == 0xFFFE).to(tl.int32) + (first_high == 0xFFFE).to(tl.int32), 0
            ),
            axis=0,
        )
        rounds += 1
    final = tl.load(base + offsets, mask=mask, other=0, volatile=True)
    changed = ((final & 0xFFFF) != first_low).to(tl.int32) + (
        ((final >> 16) & 0xFFFF) != first_high
    ).to(tl.int32)
    tl.atomic_add(counts + 0, tl.sum(tl.where(mask, changed, 0), axis=0).to(tl.int64))
    tl.atomic_add(counts + 1, unseen.to(tl.int64))
    tl.atomic_add(counts + 2, (pending > 0).to(tl.int64))
    tl.atomic_add(counts + 3, pending.to(tl.int64))
    tl.store(base + offsets, tl.full([BLOCK], -65538, tl.int32), mask=mask)


def _reset_transport(ws):
    """Every rank: wait for everything in flight, then every slot of both buffers back to
    the sentinel."""
    torch.cuda.synchronize()
    COMM.Barrier()
    ws._receive.fill_(SENTINEL_I32)
    torch.cuda.synchronize()
    COMM.Barrier()


#: GEMM producers of the poller test: o_proj's shape written straight into the slot.
GEMM_PRODUCERS = {
    "cublas": None,
    "splitk": "splitk:64,8,1,12",
    "direct": "direct:64,4",
}


def _produce(producer, x, w, slot):
    if producer == "copy":
        publish_copy(slot, x)
    elif producer == "cublas":
        torch.mm(x, w.t(), out=slot)
    else:
        from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.ops.gemm.dense_gemm import (  # noqa: E501
            _run,
            parse_kernel,
        )

        family, fields = parse_kernel(GEMM_PRODUCERS[producer])
        _run(x, w, slot, True, family, fields)


def _gemm_kernels(ws, w, rows, producer):
    """The kernels ``producer`` launches into the slot at ``rows`` rows (o_proj's shape)."""
    from torch.profiler import ProfilerActivity, profile

    dev = w.device
    x = torch.randn(rows, O_PROJ_K, device=dev).to(torch.bfloat16)
    _produce(producer, x, w, ws.slot(0, rows))  # heuristics / compile warm
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        _produce(producer, x, w, ws.slot(0, rows))
        torch.cuda.synchronize()
    names = [
        e.name
        for e in prof.events()
        if e.device_type == torch.autograd.DeviceType.CUDA and "memcpy" not in e.name.lower()
    ]
    return sorted(set(names))


#: Poller layouts: "narrow" (default) = 4 programs per slot, each holding a quarter of the
#: slot's words in registers (32 CTAs, one SM each at most); "wide" = one 4-warp program per
#: 1,024 words (up to 256 CTAs, every SM). MODELING_V2_DIRECT_PUBLISH_LAYOUTS=narrow+wide runs both
#: ("+" or ","; some launchers split exported values at commas).
POLLER_LAYOUTS = tuple(
    os.environ.get("MODELING_V2_DIRECT_PUBLISH_LAYOUTS", "narrow").replace("+", ",").split(",")
)


def _poller_launch(layout, words):
    """``(grid axis 0, PART, BLOCK, num_warps)`` of a poll of ``words`` words per slot."""
    if layout == "wide":
        return words // 1024, 1024, 1024, 4
    part = words // 4
    block = triton.next_power_of_2(part)
    return 4, part, block, max(4, min(16, block // 512))


@pytest.mark.parametrize("layout", POLLER_LAYOUTS)
@pytest.mark.parametrize("producer", ("copy", *GEMM_PRODUCERS))
def test_a_concurrent_poller_never_sees_a_value_before_the_final_one(env, producer, layout):
    """A producer writes every rank's slot through the multicast address while a poller on
    every rank, started before the producers, records the first non-sentinel value of each
    16-bit half: it must be the final value (one store per element, no partial value ever
    visible). Producers: the copy kernel (``publish_copy``), cuBLAS (``torch.mm(out=)``),
    flashinfer's split-K (``splitk:64,8,1,12``) and direct (``direct:64,4``) kernels, the
    last three at o_proj's shape. Rows 1-8, both buffers, at least 10,000 calls each; the
    poller resets the slots it read. Counted per row count.

    The poller must not keep the producer from running: a poller that occupies every SM
    (the "wide" layout at 5-8 rows) can delay a cluster-launched GEMM until the poll gives
    up, and the late stores then land after the reset (counted as stuck, then as changed in
    the next call on that buffer). The default "narrow" layout takes 32 SMs. Every poller
    variant is compiled before the timed calls (a first-call JIT on one rank would otherwise
    hold its producer back the same way), and one untimed call of every row count and
    buffer runs first: in a fresh process the first concurrent call can find one rank's
    producer late by a second or more (seen once per process, every rank, 1 row). Calls
    with a stuck block are counted and timed."""
    import time

    import numpy as np
    from mpi4py import MPI

    ws = env["direct"]["direct"]
    dev = torch.device("cuda", torch.cuda.current_device())
    g = torch.Generator(device=dev).manual_seed(9000 + RANK)
    w = (0.03 * torch.randn(HIDDEN, O_PROJ_K, generator=g, device=dev)).to(torch.bfloat16)
    kernels = {}
    if producer != "copy":
        kernels = {rows: _gemm_kernels(ws, w, rows, producer) for rows in range(1, MAX_ROWS + 1)}
    receive = ws._receive
    slab_words = MAX_ROWS * HIDDEN // 2
    max_rounds = 1 << 22

    def poll(rows, parity, counts):
        words = rows * HIDDEN // 2
        programs, part, block, warps = _poller_launch(layout, words)
        _first_seen_kernel[(programs, WORLD)](
            receive,
            slab_words,
            parity * MAX_RANKS * slab_words,
            counts,
            max_rounds,
            PART=part,
            BLOCK=block,
            num_warps=warps,
        )

    def make_x(rows):
        if producer != "copy":
            return torch.randn(rows, O_PROJ_K, generator=g, device=dev).to(torch.bfloat16)
        return (0.5 * torch.randn(rows, HIDDEN, generator=g, device=dev)).to(torch.bfloat16)

    side = torch.cuda.Stream(device=dev)

    def concurrent_call(rows, parity, counts):
        x = make_x(rows)
        torch.cuda.synchronize()
        COMM.Barrier()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            poll(rows, parity, counts)
        torch.cuda._sleep(20_000)  # let the pollers start before the stores
        _produce(producer, x, w, ws.slot(parity, rows))
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()

    # Untimed and uncounted: one call of every row count and buffer, exactly as timed below
    # (compiles every poller variant, warms the producer, the streams and the sleep kernel),
    # then every slot back to the sentinel with nothing in flight.
    warm = torch.zeros(4, dtype=torch.int64, device=dev)
    _reset_transport(ws)
    for call in range(2 * MAX_ROWS):
        concurrent_call(1 + call % MAX_ROWS, (call // MAX_ROWS) % 2, warm)
    _reset_transport(ws)

    counts = torch.zeros(MAX_ROWS, 4, dtype=torch.int64, device=dev)
    stuck_calls, stuck_seconds, call_seconds = 0, [], []
    stuck_before = 0
    for call in range(POLLER_CALLS):
        rows = 1 + call % MAX_ROWS
        began = time.perf_counter()
        concurrent_call(rows, (call // MAX_ROWS) % 2, counts[rows - 1])
        seconds = time.perf_counter() - began
        call_seconds.append(seconds)
        stuck_now = int(counts[:, 2].sum())
        if stuck_now != stuck_before:
            stuck_calls += 1
            stuck_seconds.append(seconds)
            stuck_before = stuck_now
    mine = counts.cpu().numpy()
    total = COMM.allreduce(mine, op=MPI.SUM)
    bad_ranks = COMM.allgather(int(mine[:, 0].sum() + mine[:, 2].sum()))
    stuck_calls_all = COMM.allgather(stuck_calls)
    stuck_seconds_max = COMM.allreduce(max(stuck_seconds, default=0.0), op=MPI.MAX)
    _reset_transport(ws)
    if RANK == 0:
        sms = torch.cuda.get_device_properties(dev).multi_processor_count
        programs = [_poller_launch(layout, r * HIDDEN // 2)[0] * WORLD for r in (1, 8)]
        print(
            f"\nconcurrent poller, producer {producer}, layout {layout} (CTAs {programs[0]} at 1 "
            f"row, {programs[1]} at 8 rows; {sms} SMs): {POLLER_CALLS} calls x {WORLD} slots; "
            f"call p50 {1e6 * statistics.median(call_seconds):.0f} us"
        )
        for rows in range(1, MAX_ROWS + 1):
            changed, watched, stuck, unseen = (int(v) for v in total[rows - 1])
            print(
                f"   rows {rows}: halves changed after first seen {changed}, watched arriving "
                f"{watched}, blocks stuck {stuck} ({unseen} halves never seen)"
                + (f"; kernels {kernels[rows]}" if kernels else "")
            )
        print(
            f"   changed + stuck per rank {bad_ranks}; calls with a stuck block per rank "
            f"{stuck_calls_all}, longest such call {stuck_seconds_max:.3f} s"
        )
    assert POLLER_CALLS >= 10_000 or os.environ.get("MODELING_V2_DIRECT_PUBLISH_CALLS")
    assert int(np.asarray(total)[:, 2].sum()) == 0, f"blocks never completed: {total[:, 2]}"
    assert int(np.asarray(total)[:, 0].sum()) == 0, f"halves changed: {total[:, 0]}"
    assert int(np.asarray(total)[:, 1].sum()) > 0, "no poller ever watched a store arrive"


def test_the_default_guard_never_fired(env):
    fired = [ws.guard_fired() for ws in env["direct"].values()]
    assert _agree(not any(fired)), f"rank {RANK}: guard fired {fired}"


@pytest.mark.skipif(os.environ.get("MODELING_V2_TIMING") != "1", reason="set MODELING_V2_TIMING=1")
def test_chain_benchmark(env):
    """norm-gate -> o_proj -> post-attention hop -> pool GEMM, 62 layers of distinct cold
    weights per CUDA graph, PDL on: the hop (o_proj into a fresh tensor, FULL hop) against direct
    publication (o_proj into the slot, consumer hop), each with o_proj by cuBLAS (``hop``,
    ``direct_publish``) and by flashinfer split-K ``(64,8,1,12)`` (``hop_sk``,
    ``direct_publish_sk``). Each layer's norm-gate reads the previous layer's prenorm output,
    so the chain is serial like the decode step. Per
    layer = replay time / 62; median of 5 x 20 replays per arm and round, rounds alternate
    the arm order; max over ranks."""
    from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.ops.gemm.dense_gemm import (  # noqa: E501
        dense_gemm,
    )
    from tensorrt_llm._torch.models.modeling_musespark_open import _attention_norm_gate_kernel

    layers = 62
    rows = int(os.environ.get("MODELING_V2_DIRECT_PUBLISH_TIMING_ROWS", "8"))
    dev = torch.device("cuda", torch.cuda.current_device())
    g = torch.Generator(device=dev).manual_seed(77 + RANK)
    o_w = [
        (0.03 * torch.randn(HIDDEN, O_PROJ_K, generator=g, device=dev)).to(torch.bfloat16)
        for _ in range(layers)
    ]
    pool_w = [
        (0.01 * torch.randn(512, HIDDEN, generator=g, device=dev)).to(torch.bfloat16)
        for _ in range(layers)
    ]
    _, residual0, alpha, beta, weight = _inputs(rows, 31337)
    attention0 = torch.randn(rows, 2 * O_PROJ_K, generator=g, device=dev).to(torch.bfloat16)
    ws_direct = env["direct"]["direct"]

    def norm_gate(x):
        out = torch.empty(rows, O_PROJ_K, dtype=torch.bfloat16, device=dev)
        _attention_norm_gate_kernel[(rows * 16,)](
            out, x, x[:, O_PROJ_K:], x.stride(0), x.stride(0), 64, 1e-6, HEADS=16, HEAD_DIM=64,
            num_warps=4,
        )  # fmt: skip
        return out

    def chain(direct: bool, producer: str):
        x, h = attention0, residual0
        for i in range(layers):
            gated = norm_gate(x)
            if direct:
                slot = ws_direct.slot(i % 2, rows)
                _produce(producer, gated, o_w[i], slot)
                h, ffn = op(
                    slot, h, alpha, beta, weight, POST_EPS, PRE_EPS, False, i % 2, ws_direct
                )
            else:
                partial = torch.empty(rows, HIDDEN, dtype=torch.bfloat16, device=dev)
                _produce(producer, gated, o_w[i], partial)
                h, ffn = env["fused"](partial, h, alpha, beta, weight)
            pooled = dense_gemm(ffn, pool_w[i], "splitk:64,8,4,12", True)
            x = ffn[:, : 2 * O_PROJ_K]
            del pooled
        return h

    arms = {
        "hop": (False, "cublas"),
        "direct_publish": (True, "cublas"),
        "hop_sk": (False, "splitk"),
        "direct_publish_sk": (True, "splitk"),
    }
    graphs = {}
    for name, (direct, producer) in arms.items():
        chain(direct, producer)
        torch.cuda.synchronize()
        COMM.Barrier()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            chain(direct, producer)
        for _ in range(10):
            graph.replay()
        graphs[name] = graph
    torch.cuda.synchronize()
    names = tuple(arms)
    samples = {name: [] for name in names}
    for round_ in range(int(os.environ.get("MODELING_V2_DIRECT_PUBLISH_TIMING_ROUNDS", "4"))):
        order = names if round_ % 2 == 0 else tuple(reversed(names))
        for name in order:
            for _ in range(5):
                COMM.Barrier()
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(20):
                    graphs[name].replay()
                end.record()
                end.synchronize()
                samples[name].append(start.elapsed_time(end) * 1000.0 / 20 / layers)
    result = {name: _max_over_ranks(statistics.median(samples[name])) for name in names}
    rounds = {name: [_max_over_ranks(v) for v in samples[name][::5]] for name in names}
    if RANK == 0:
        print(
            f"\ndirect_publish chain benchmark, world {WORLD}, rows {rows}, us per layer (max over ranks of "
            "the per-rank median): "
            + "  ".join(f"{name} {result[name]:.3f}" for name in names)
            + f"\n  direct_publish - hop {result['direct_publish'] - result['hop']:+.3f}  direct_publish_sk - hop_sk "
            f"{result['direct_publish_sk'] - result['hop_sk']:+.3f}  direct_publish_sk - hop "
            f"{result['direct_publish_sk'] - result['hop']:+.3f}"
            + "".join(
                f"\n  {name} per round (first sample, max over ranks) "
                + " ".join(f"{v:.3f}" for v in rounds[name])
                for name in names
            )
        )
