# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test of the direct unpool transport's buffer discipline (``unpool_into_boundary``; TP8, two GB300 nodes).

Run with eight MPI ranks on two NVLink-connected four-GPU GB300 nodes, e.g.
``srun -N2 --ntasks-per-node=4 --mpi=pmix python -m pytest <this file>``;
anywhere else every test skips.

The reuse order of the transport rests on one rule: no two consecutive calls on the
workspace use the same buffer, the last call of a cycle (a decode step) and the first
of the next included (``unpool_direct_boundary.buffer_sequence``). This file is the
negative control that can fail for that rule. Per layer, in one all-PDL graph:
a producer spin on a rotating rank (a PDL kernel that waits, then spins) -> the unpool
split-K GEMM into the layer's slot -> a consumer spin on a rotating rank (between the
GEMM and the boundary; the last layer always has a 100 us one on rank 1) -> the
direct boundary (the early-release entry trigger) -> a hop that waits. Each layer's inputs are
fixed tensors, so only the transport couples the layers; the reference is the gather boundary on the
tagged inbox, eager, layer by layer. Graph replays run back to back, so a fast rank's
next replay starts while the late consumer still reads.

* even count on alternating buffers: exact;
* **negative control**: an odd count on ``layer mod 2`` (the last call of a replay and
  the first of the next on buffer 0, no collective in between) -- a fast rank's next
  GEMM overwrites the late consumer's unread slot: must be detected (differing bits, or
  the 20 ms breaking guard);
* the same odd count on ``buffer_sequence`` (the last call on buffer 2): exact;
* the model's cycle, 61 fused layers on ``buffer_sequence(61)``: exact.
"""

import importlib

import pytest
import torch

try:
    from mpi4py import MPI
except ImportError:  # pragma: no cover - the test needs MPI ranks
    MPI = None

_TARGET = (
    "tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open."
    "muse_spark_open_816b_a42b_nvfp4__sm_103__tep"
)
D = importlib.import_module(_TARGET + ".ops.comm.unpool_direct_boundary")
GATHER_BOUNDARY = importlib.import_module(_TARGET + ".ops.comm.unpool_gather_boundary")
INBOX = importlib.import_module(_TARGET + ".ops.comm.projection_inbox")
GEMM = importlib.import_module(_TARGET + ".ops.gemm.dense_gemm")

try:
    import cuda.bindings.driver as _cuda_driver
    import cutlass
    import cutlass.cute as cute
    from cutlass import Int64
    from cutlass.cute.runtime import make_ptr
except ImportError:  # pragma: no cover - the test needs CuTe DSL
    cute = None

WORLD = 8
HIDDEN, LOCAL_COLUMNS = D.HIDDEN, D.LOCAL_COLUMNS
POST_EPS, PRE_EPS = 1e-6, 1e-5
UNPOOL_K = 4096
SPLITK = "splitk:64,8,4,12"
ROWS = 8
#: Graph replays back to back per check, and checks per case.
BACK_TO_BACK, CHECKS = 5, 6


def _world():
    if MPI is None or MPI.COMM_WORLD.Get_size() != WORLD or not torch.cuda.is_available():
        pytest.skip("needs eight MPI ranks with CUDA (two four-GPU nodes)")
    if cute is None or not D.available():
        pytest.skip("needs CuTe DSL")
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    torch.cuda.set_device(rank % 4)
    if torch.cuda.get_device_capability() not in ((10, 3), (10, 7)):
        pytest.skip("needs sm_103 or sm_107")
    return comm, rank


@pytest.fixture(scope="module")
def group():
    """The gather boundary's inbox and a direct workspace whose guard (20 ms) gives up (collective)."""
    comm, rank = _world()
    from tensorrt_llm.mapping import Mapping

    mapping = Mapping(world_size=WORLD, tp_size=WORLD, rank=rank, gpus_per_node=4)
    inbox = INBOX.ProjectionInbox(mapping, row_words=LOCAL_COLUMNS // 2)
    guarded = D.UnpoolDirectWorkspace(mapping, guard_ns=20_000_000, guard_break=True)
    comm.Barrier()
    yield comm, rank, inbox, guarded
    torch.cuda.synchronize()
    comm.Barrier()


def _settle(comm):
    torch.cuda.synchronize()
    comm.Barrier()


def _bits(t):
    return t.view({4: torch.int32, 2: torch.int16}[t.element_size()])


def _differing(got, want) -> int:
    return sum(int((_bits(g) != _bits(w)).sum().item()) for g, w in zip(got, want))


if cute is not None:

    class _Hop:
        """One CTA launched with PDL: wait on the predecessor, spin ``ns`` on the global
        timer, release the dependent."""

        def __init__(self, ns: int = 0):
            self.ns = int(ns)

        @cute.jit
        def __call__(self, sink: cute.Pointer, stream: _cuda_driver.CUstream):
            self.kernel(sink).launch(grid=[1, 1, 1], block=[32, 1, 1], stream=stream, use_pdl=True)

        @cute.kernel
        def kernel(self, sink: cute.Pointer):
            cute.arch.griddepcontrol_wait()
            if cutlass.const_expr(self.ns > 0):
                start = D._globaltimer()
                now = D._globaltimer()
                while now - start < Int64(self.ns):
                    now = D._globaltimer()
            cute.arch.griddepcontrol_launch_dependents()


def _launcher(kernel):
    sink = torch.zeros(4, dtype=torch.int32, device="cuda")
    stream = _cuda_driver.CUstream(torch.cuda.current_stream().cuda_stream)
    compiled = cute.compile(
        kernel,
        make_ptr(cutlass.Int32, 16, cute.AddressSpace.gmem, assumed_align=16),
        stream,
        options="--enable-tvm-ffi",
    )

    def launch():
        compiled(sink.data_ptr(), _cuda_driver.CUstream(torch.cuda.current_stream().cuda_stream))

    launch()
    return launch


class _Join:
    def __init__(self, seed: int):
        from tensorrt_llm._torch.models.modeling_musespark_open import MuseSparkOpenGatedResidual
        from tensorrt_llm._torch.modules.rms_norm import RMSNorm

        torch.manual_seed(seed)
        self.gate = MuseSparkOpenGatedResidual(HIDDEN, torch.bfloat16).cuda()
        self.gate.gate.normal_(0, 0.5)
        self.gate.cache_derived_state()
        self.gate._preqkv_norm_fp32 = False
        self.norm = RMSNorm(hidden_size=HIDDEN, eps=PRE_EPS, dtype=torch.bfloat16, use_gemma=True)
        self.norm = self.norm.cuda()
        self.norm.weight.normal_(0, 0.02)
        self.gain = torch.empty(HIDDEN, device="cuda", dtype=torch.bfloat16).normal_(1, 0.1)

    def tagged(self, inbox, local, residual):
        alpha, beta = self.gate.coefficients()
        return GATHER_BOUNDARY.unpool_gather_boundary(
            local,
            inbox,
            residual,
            self.gain,
            alpha,
            beta,
            self.norm.weight,
            POST_EPS,
            PRE_EPS,
            False,
            "entry",
        )

    def direct(self, ws, buffer, slot, residual):
        alpha, beta = self.gate.coefficients()
        return D.unpool_direct_boundary(
            slot,
            ws,
            buffer,
            residual,
            self.gain,
            alpha,
            beta,
            self.norm.weight,
            POST_EPS,
            PRE_EPS,
            False,
            "entry",
            None,
        )


class _Layers:
    def __init__(self, layers: int, rank: int):
        self.layers = layers
        gen = torch.Generator(device="cuda").manual_seed(5150 + rank)
        self.w = (0.03 * torch.randn(LOCAL_COLUMNS, UNPOOL_K, generator=gen, device="cuda")).to(
            torch.bfloat16
        )
        self.x = torch.zeros(layers, ROWS, UNPOOL_K, device="cuda", dtype=torch.bfloat16)
        self.residual = torch.zeros(layers, ROWS, HIDDEN, device="cuda", dtype=torch.float32)
        self.rank = rank

    def fill(self, seed: int):
        gen = torch.Generator().manual_seed(seed * 16 + self.rank)
        self.x.copy_(torch.randn(*self.x.shape, generator=gen).to(torch.bfloat16))
        shared = torch.Generator().manual_seed(seed * 16 + 99)
        self.residual.copy_(torch.randn(*self.residual.shape, generator=shared))


def _producer_spins(layers: int, rank: int) -> dict:
    """A producer spin on a rotating rank every layer (20 / 50 / 100 us), except the last
    (a late producer there would hold every consumer of it, hiding the late consumer)."""
    return {
        layer: (20_000, 50_000, 100_000)[layer % 3]
        for layer in range(layers - 1)
        if (layer * 3 + 1) % WORLD == rank
    }


def _consumer_spins(layers: int, rank: int) -> dict:
    """A consumer spin (between the GEMM and the boundary) on a rotating rank every layer,
    and 100 us on rank 1 at the last layer."""
    spins = {
        layer: (50_000, 100_000)[layer % 2]
        for layer in range(layers)
        if (layer * 5 + 3) % WORLD == rank
    }
    if rank == 1:
        spins[layers - 1] = 100_000
    return spins


def _run(comm, rank, inbox, ws, buffers):
    """Graph of ``len(buffers)`` layers on ``buffers``; ``CHECKS`` x ``BACK_TO_BACK`` replays.
    Returns this rank's differing bits per check and whether its guard fired."""
    layers = len(buffers)
    join = _Join(seed=7)
    ws.build_variants([D.variant(ROWS, True, POST_EPS, PRE_EPS, False, "entry", False)])
    hop = _launcher(_Hop())
    spin = {ns: _launcher(_Hop(ns)) for ns in (20_000, 50_000, 100_000)}
    producers = {layer: spin[ns] for layer, ns in _producer_spins(layers, rank).items()}
    consumers = {layer: spin[ns] for layer, ns in _consumer_spins(layers, rank).items()}
    lay = _Layers(layers, rank)
    lay.fill(1)
    _settle(comm)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        got = []
        for layer, buffer in enumerate(buffers):
            if layer in producers:
                producers[layer]()
            slot = ws.slot(buffer, ROWS)
            GEMM.dense_gemm(lay.x[layer], lay.w, SPLITK, True, out=slot)
            if layer in consumers:
                consumers[layer]()
            got.append(join.direct(ws, buffer, slot, lay.residual[layer]))
            hop()
    _settle(comm)
    bad, fired = [], False
    for check in range(CHECKS):
        lay.fill(100 + check)
        want = []
        for layer in range(layers):
            local = GEMM.dense_gemm(lay.x[layer], lay.w, SPLITK, True)
            want.append(tuple(t.clone() for t in join.tagged(inbox, local, lay.residual[layer])))
        _settle(comm)
        for _ in range(BACK_TO_BACK):
            graph.replay()
        torch.cuda.synchronize()
        bad.append(sum(_differing(g, w) for g, w in zip(got, want)))
        fired = fired or ws.guard_fired()
        _settle(comm)
    # Leave the workspace clean for the next case (nothing in flight).
    ws.reset()
    ws.clear_guard()
    _settle(comm)
    return bad, fired


def _totals(comm, rank, name, bad, fired):
    rows = comm.gather((rank, bad, fired), root=0)
    if rank == 0:
        for r, bits, f in rows:
            print(
                f"[unpool_direct buffers] {name}: rank {r} differing bits per check {bits} guard {f}"
            )
    total = comm.allreduce(sum(bad), op=MPI.SUM)
    guards = comm.allreduce(int(fired), op=MPI.SUM)
    if rank == 0:
        print(
            f"[unpool_direct buffers] {name}: total differing bits {total}, ranks with guard {guards}"
        )
    return total, guards


@torch.inference_mode()
def test_an_even_count_on_alternating_buffers_is_exact(group):
    comm, rank, inbox, ws = group
    buffers = tuple(layer % 2 for layer in range(16))
    total, guards = _totals(comm, rank, "16 on 0/1", *_run(comm, rank, inbox, ws, buffers))
    assert total == 0 and guards == 0


@torch.inference_mode()
def test_negative_control_two_consecutive_calls_on_one_buffer_are_detected(group):
    comm, rank, inbox, ws = group
    buffers = tuple(layer % 2 for layer in range(15))
    with pytest.raises(ValueError):
        D.check_buffer_sequence(buffers)
    total, guards = _totals(comm, rank, "15 on layer mod 2", *_run(comm, rank, inbox, ws, buffers))
    assert total > 0 or guards > 0, "two consecutive calls on one buffer went undetected"


@torch.inference_mode()
def test_an_odd_count_on_the_buffer_sequence_is_exact(group):
    comm, rank, inbox, ws = group
    buffers = D.buffer_sequence(15)
    assert buffers[-1] == 2
    total, guards = _totals(
        comm, rank, "15 on buffer_sequence", *_run(comm, rank, inbox, ws, buffers)
    )
    assert total == 0 and guards == 0


@torch.inference_mode()
def test_the_models_cycle_of_61_calls_is_exact(group):
    comm, rank, inbox, ws = group
    buffers = D.buffer_sequence(61)
    total, guards = _totals(
        comm, rank, "61 on buffer_sequence", *_run(comm, rank, inbox, ws, buffers)
    )
    assert total == 0 and guards == 0
