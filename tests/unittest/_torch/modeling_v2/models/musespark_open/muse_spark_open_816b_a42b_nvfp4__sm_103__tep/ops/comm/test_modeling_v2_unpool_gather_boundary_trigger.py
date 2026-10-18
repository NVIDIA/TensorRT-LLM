# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test of ``boundary_early_release``'s QKV edge: the fused unpool boundary releases the next QKV early.

Run with eight MPI ranks on two NVLink-connected four-GPU GB300 nodes, e.g.
``srun -N2 --ntasks-per-node=4 --mpi=pmix python -m pytest <this file>``;
anywhere else every test skips. (The boundary's own cells, including its
trigger variants, are certified bit for bit against the built-in gather +
boundary by ``test_modeling_v2_unpool_gather_boundary.py``.)

The decode chain of one layer edge is rebuilt with the real kernels:
[unpool split-K GEMM -> fused unpool gather + boundary -> next layer's QKV
split-K GEMM], flashinfer's ``(64,8,4,12)`` tactic for both GEMMs, all launched
with programmatic dependent launch. With the boundary triggering its
dependents at entry (the early release) or after its wait, the QKV GEMM issues its weight
loads while the boundary still polls its peers; it must still read the
boundary's output, never what the buffer held before. Negative control: one
rank sleeps before every replay, so every other rank's boundary waits for
milliseconds with the QKV GEMM already resident; the QKV outputs (and the
boundary's) must equal, as bits, those of the chain whose boundary triggers at
exit -- the fused gather boundary as the tier-1 set runs it -- through eager calls and CUDA-graph
replays with changing inputs.

A second control delays the boundary's own input: between the unpool GEMM and
the boundary sits a PDL kernel that releases its dependents at entry, waits,
spins and only then writes the boundary's input. With the boundary triggering
at entry the QKV GEMM is then launched while that writer still spins -- three
grids in flight before the QKV's own wait returns, the boundary's input not yet
written -- at rows 1, 2, 3, 5, 7 and 8 (the boundary's 16-CTA cluster at one
row, 8 above), with one rank sleeping per call and every intermediate
NaN-poisoned before each replay. The reference is the same arithmetic with no
overlap at all (a device synchronize after every kernel, the GEMMs without
PDL, the boundary at exit).

``MODELING_V2_TIMING=1`` also runs the chain microbenchmark (printed, never
gated): 62 layers of [rest-of-layer stand-in -> unpool -> boundary -> QKV] with
cold weights per layer, per trigger position, with and without the dense GEMM L2
prefetch of the next QKV weight, timed on every rank.
"""

import os
import statistics
import time

import pytest
import torch
import triton
import triton.language as tl

try:
    from mpi4py import MPI
except ImportError:  # pragma: no cover - the test needs MPI ranks
    MPI = None

from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.ops.comm.projection_inbox import (  # noqa: E501
    ProjectionInbox,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.ops.comm.unpool_gather_boundary import (  # noqa: E501
    HIDDEN,
    LOCAL_COLUMNS,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.ops.comm.unpool_gather_boundary import (  # noqa: E501
    unpool_gather_boundary as op,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.ops.gemm.dense_gemm import (  # noqa: E501
    dense_gemm,
    kernel_error,
    l2_prefetch,
)

WORLD = 8
POST_EPS, PRE_EPS = 1e-6, 1e-5
SPLITK = "splitk:64,8,4,12"
QKV_N = 2304
UNPOOL_K = 4096
ROWS = 8


def _world():
    if MPI is None or MPI.COMM_WORLD.Get_size() != WORLD or not torch.cuda.is_available():
        pytest.skip("needs eight MPI ranks with CUDA (two four-GPU GB300 nodes)")
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    torch.cuda.set_device(rank % 4)
    if torch.cuda.get_device_capability() != (10, 3):
        pytest.skip("needs GB300 (sm_103)")
    return comm, rank


@pytest.fixture(scope="module")
def group():
    comm, rank = _world()
    from tensorrt_llm.mapping import Mapping

    mapping = Mapping(world_size=WORLD, tp_size=WORLD, rank=rank, gpus_per_node=4)
    inbox = ProjectionInbox(mapping, row_words=LOCAL_COLUMNS // 2)
    comm.Barrier()
    return comm, rank, inbox


class _Edge:
    """One layer edge's operands: unpool input/weight, the boundary's vectors, the QKV weight."""

    def __init__(self, seed: int, rank: int, rows: int = ROWS):
        self.rows = rows
        gen = torch.Generator(device="cuda").manual_seed(seed * WORLD + rank)
        self.x = torch.empty(rows, UNPOOL_K, device="cuda", dtype=torch.bfloat16)
        self.unpool_w = (
            torch.randn(LOCAL_COLUMNS, UNPOOL_K, device="cuda", generator=gen) * 0.02
        ).to(torch.bfloat16)
        shared = torch.Generator(device="cuda").manual_seed(seed)
        self.residual = torch.empty(rows, HIDDEN, device="cuda", dtype=torch.float32)
        self.gain = (torch.randn(HIDDEN, device="cuda", generator=shared) * 0.1 + 1).to(
            torch.bfloat16
        )
        self.alpha = torch.rand(HIDDEN, device="cuda", generator=shared) + 0.5
        self.beta = torch.rand(HIDDEN, device="cuda", generator=shared) + 0.5
        self.weight = (torch.randn(HIDDEN, device="cuda", generator=shared) * 0.02).to(
            torch.bfloat16
        )
        self.qkv_w = (torch.randn(QKV_N, HIDDEN, device="cuda", generator=gen) * 0.02).to(
            torch.bfloat16
        )
        self.gen = gen
        # Split-K for both GEMMs wherever it serves the row count (it does at 1-8 rows).
        self.unpool_kernel = _split_k_or_cublas(rows, UNPOOL_K, LOCAL_COLUMNS)
        self.qkv_kernel = _split_k_or_cublas(rows, HIDDEN, QKV_N)
        self.late = torch.empty(rows, LOCAL_COLUMNS, device="cuda", dtype=torch.bfloat16)

    def load(self, seed: int) -> None:
        g = torch.Generator(device="cuda").manual_seed(1000 + seed)
        self.x.copy_(
            torch.randn(self.rows, UNPOOL_K, device="cuda", generator=g).to(torch.bfloat16)
        )
        self.residual.copy_(torch.randn(self.rows, HIDDEN, device="cuda", generator=g) * 4.0)

    def _boundary(self, inbox, local, trigger: str):
        return op(
            local,
            inbox,
            self.residual,
            self.gain,
            self.alpha,
            self.beta,
            self.weight,
            POST_EPS,
            PRE_EPS,
            False,
            trigger,
        )

    def run(self, inbox, trigger: str):
        local = dense_gemm(self.x, self.unpool_w, self.unpool_kernel, True)
        updated, normalized = self._boundary(inbox, local, trigger)
        return updated, normalized, dense_gemm(normalized, self.qkv_w, self.qkv_kernel, True)

    def run_late(self, inbox, trigger: str, delay_ns: int):
        """``run`` with a PDL writer that releases at entry and writes the boundary input late."""
        local = dense_gemm(self.x, self.unpool_w, self.unpool_kernel, True)
        _late_writer(local, self.late, delay_ns)
        updated, normalized = self._boundary(inbox, self.late, trigger)
        return updated, normalized, dense_gemm(normalized, self.qkv_w, self.qkv_kernel, True)

    def serial(self, inbox):
        """No overlap anywhere: a synchronize after every kernel, GEMMs without PDL, exit trigger."""
        local = dense_gemm(self.x, self.unpool_w, self.unpool_kernel, False)
        torch.cuda.synchronize()
        updated, normalized = self._boundary(inbox, local, "exit")
        torch.cuda.synchronize()
        qkv = dense_gemm(normalized, self.qkv_w, self.qkv_kernel, False)
        torch.cuda.synchronize()
        return updated.clone(), normalized.clone(), qkv.clone()


def _split_k_or_cublas(rows: int, k: int, n: int) -> str:
    return SPLITK if kernel_error(SPLITK, rows, k, n) is None else "cublas"


@triton.jit
def _globaltimer(dummy):
    return tl.inline_asm_elementwise(
        "mov.u64 $0, %globaltimer;", "=l,r", [dummy], dtype=tl.int64, is_pure=False, pack=1
    )


@triton.jit
def _late_writer_kernel(src, dst, n, delay_ns, BLOCK: tl.constexpr):
    """PDL: release the dependents at entry, wait for the predecessor, spin, copy src -> dst."""
    tl.extra.cuda.gdc_launch_dependents()
    tl.extra.cuda.gdc_wait()
    start = _globaltimer(0)
    now = start
    while now - start < delay_ns:
        now = _globaltimer(0)
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    tl.store(dst + offsets, tl.load(src + offsets, mask=mask), mask=mask)


def _late_writer(src: torch.Tensor, dst: torch.Tensor, delay_ns: int) -> None:
    n = src.numel()
    _late_writer_kernel[(triton.cdiv(n, 1024),)](
        src, dst, n, delay_ns, BLOCK=1024, num_warps=4, launch_pdl=True
    )


def _bits(t):
    return t.contiguous().view(torch.int32 if t.dtype == torch.float32 else torch.int16)


def _assert_same(got, want, what):
    for g, w, name in zip(got, want, ("residual", "next input", "qkv")):
        differ = (_bits(g) != _bits(w)).sum().item()
        assert differ == 0, f"{what}: {differ} {name} values differ from the exit-trigger chain"


@pytest.mark.parametrize("trigger", ["after_wait", "entry"])
@torch.inference_mode()
def test_an_early_boundary_trigger_keeps_every_dependent_bit(group, trigger):
    comm, rank, inbox = group
    edge = _Edge(seed=41, rank=rank)

    # Eager, with a sleeping rank per call: early chain vs exit chain on the same inputs.
    for step in range(4):
        edge.load(step)
        torch.cuda.synchronize()
        comm.Barrier()
        if rank == step % WORLD:
            time.sleep(0.003)
        want = [t.clone() for t in edge.run(inbox, "exit")]
        torch.cuda.synchronize()
        comm.Barrier()
        if rank == (step + 3) % WORLD:
            time.sleep(0.003)
        got = edge.run(inbox, trigger)
        torch.cuda.synchronize()
        _assert_same(got, want, f"eager step {step}, trigger {trigger}")

    # Graph replays: both chains captured once, replayed with changing inputs.
    graphs = {}
    outs = {}
    for name in ("exit", trigger):
        edge.run(inbox, name)
        torch.cuda.synchronize()
        comm.Barrier()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            outs[name] = edge.run(inbox, name)
        graphs[name] = graph
        torch.cuda.synchronize()
        comm.Barrier()
    for replay in range(8):
        edge.load(100 + replay)
        torch.cuda.synchronize()
        comm.Barrier()
        if rank == replay % WORLD:
            time.sleep(0.002)
        graphs["exit"].replay()
        torch.cuda.synchronize()
        want = [t.clone() for t in outs["exit"]]
        for t in outs[trigger]:
            t.fill_(float("nan"))
        torch.cuda.synchronize()
        comm.Barrier()
        if rank == (replay + 5) % WORLD:
            time.sleep(0.002)
        graphs[trigger].replay()
        torch.cuda.synchronize()
        _assert_same(outs[trigger], want, f"replay {replay}, trigger {trigger}")
    comm.Barrier()


def _differing(got, want) -> list:
    return [int((_bits(g) != _bits(w)).sum().item()) for g, w in zip(got, want)]


@pytest.mark.parametrize("trigger", ["entry", "after_wait"])
@pytest.mark.parametrize("rows", [1, 2, 3, 5, 7, 8])
@torch.inference_mode()
def test_a_late_boundary_input_and_a_sleeping_rank_keep_every_bit(group, rows, trigger):
    """The boundary's input lands 30 us after the QKV GEMM may launch; bits must not change."""
    comm, rank, inbox = group
    edge = _Edge(seed=50 + rows, rank=rank, rows=rows)
    delay_ns = 30_000
    report = []

    for step in range(3):
        edge.load(37 * step + rank)
        torch.cuda.synchronize()
        comm.Barrier()
        want = edge.serial(inbox)
        comm.Barrier()
        edge.late.fill_(float("nan"))
        torch.cuda.synchronize()
        comm.Barrier()
        if rank == (step * 3) % WORLD:
            time.sleep(0.002)
        got = edge.run_late(inbox, trigger, delay_ns)
        torch.cuda.synchronize()
        report.append(("eager", step, _differing(got, want)))

    edge.load(99)
    edge.run_late(inbox, trigger, delay_ns)
    torch.cuda.synchronize()
    comm.Barrier()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        outs = edge.run_late(inbox, trigger, delay_ns)
    torch.cuda.synchronize()
    comm.Barrier()
    for replay in range(8):
        edge.load(200 + 37 * replay + rank)
        torch.cuda.synchronize()
        comm.Barrier()
        want = edge.serial(inbox)
        comm.Barrier()
        edge.late.fill_(float("nan"))
        for t in outs:
            t.fill_(float("nan"))
        torch.cuda.synchronize()
        comm.Barrier()
        if rank == (replay * 5 + 1) % WORLD:
            time.sleep(0.002)
        graph.replay()
        torch.cuda.synchronize()
        report.append(("replay", replay, _differing(outs, want)))
    comm.Barrier()
    bad = [r for r in report if any(r[2])]
    print(
        f"[early_release late edge] rank {rank} rows {rows} trigger {trigger} unpool {edge.unpool_kernel} "
        f"qkv {edge.qkv_kernel}: {len(report)} checks, {len(bad)} with differing bits {bad[:3]}"
    )
    all_bad = comm.allgather(len(bad))
    assert sum(all_bad) == 0, f"differing bits on ranks {all_bad}; this rank: {bad[:3]}"


# --------------------------------------------------------------------------
# Chain microbenchmark (MODELING_V2_TIMING=1)
# --------------------------------------------------------------------------
@triton.jit
def _rest_of_layer_kernel(src, dst, n, delay_ns, BLOCK: tl.constexpr):
    """The rest of a layer between QKV and unpool: waits, reads the QKV output, spins."""
    tl.extra.cuda.gdc_wait()
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    value = tl.load(src + offsets, mask=mask)
    start = _globaltimer(0)
    now = start
    while now - start < delay_ns:
        now = _globaltimer(0)
    tl.store(dst + offsets, value, mask=mask)


LAYERS = 62


@pytest.mark.skipif(os.environ.get("MODELING_V2_TIMING") != "1", reason="MODELING_V2_TIMING=1")
@torch.inference_mode()
def test_bench_qkv_site_chain(group):
    comm, rank, inbox = group
    gen = torch.Generator(device="cuda").manual_seed(77 + rank)
    unpool_w = [
        (torch.randn(LOCAL_COLUMNS, UNPOOL_K, device="cuda", generator=gen) * 0.02).to(
            torch.bfloat16
        )
        for _ in range(LAYERS)
    ]
    qkv_w = [
        (torch.randn(QKV_N, HIDDEN, device="cuda", generator=gen) * 0.02).to(torch.bfloat16)
        for _ in range(LAYERS)
    ]
    x = torch.randn(ROWS, UNPOOL_K, device="cuda").to(torch.bfloat16)
    residual = torch.randn(ROWS, HIDDEN, device="cuda")
    gain = torch.ones(HIDDEN, device="cuda", dtype=torch.bfloat16)
    alpha = torch.ones(HIDDEN, device="cuda")
    beta = torch.ones(HIDDEN, device="cuda")
    weight = torch.zeros(HIDDEN, device="cuda", dtype=torch.bfloat16)
    rest = torch.empty(ROWS, QKV_N, device="cuda", dtype=torch.bfloat16)
    sink = torch.zeros(1, dtype=torch.int32, device="cuda")
    side = torch.cuda.Stream()
    rest_ns = int(os.environ.get("EARLY_RELEASE_TIMING_REST_NS", "10000"))

    def chain(trigger: str, prefetch: bool):
        qkv = torch.zeros(ROWS, QKV_N, device="cuda", dtype=torch.bfloat16)
        for i in range(LAYERS):
            main = torch.cuda.current_stream()
            _rest_of_layer_kernel[(triton.cdiv(ROWS * QKV_N, 1024),)](
                qkv, rest, ROWS * QKV_N, rest_ns, BLOCK=1024, num_warps=4, launch_pdl=True
            )
            local = dense_gemm(x, unpool_w[i], SPLITK, True)
            nxt = (i + 1) % LAYERS
            if prefetch:
                side.wait_stream(main)
                with torch.cuda.stream(side):
                    l2_prefetch(qkv_w[nxt], sink)
            _, normalized = op(
                local, inbox, residual, gain, alpha, beta, weight, POST_EPS, PRE_EPS, False, trigger
            )
            if prefetch:
                main.wait_stream(side)
            qkv = dense_gemm(normalized, qkv_w[nxt], SPLITK, True)

    arms = [
        ("exit", True),  # tier-1 baseline: gather boundary + L2 prefetch of the next QKV weight
        ("exit", False),
        ("after_wait", False),
        ("after_wait", True),
        ("entry", False),  # boundary_early_release
        ("entry", True),
    ]
    graphs = {}
    for arm in arms:
        chain(*arm)
        torch.cuda.synchronize()
        comm.Barrier()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            chain(*arm)
        graphs[arm] = graph
        torch.cuda.synchronize()
        comm.Barrier()
    samples = {arm: [] for arm in arms}
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    for r in range(10):
        order = arms if r % 2 == 0 else list(reversed(arms))
        for arm in order:
            torch.cuda.synchronize()
            comm.Barrier()
            start.record()
            graphs[arm].replay()
            end.record()
            end.synchronize()
            samples[arm].append(start.elapsed_time(end) * 1000.0 / LAYERS)
    medians = {arm: statistics.median(v) for arm, v in samples.items()}
    spreads = {arm: (min(v), max(v)) for arm, v in samples.items()}
    results = comm.gather((medians, spreads), root=0)
    if rank == 0:
        base = arms[0]
        print(
            f"\n[early_release bench qkv site] rest-of-layer stand-in {rest_ns / 1000:.1f} us; us per layer"
        )
        for arm in arms:
            per_rank = [res[0][arm] for res in results]
            worst = max(per_rank)
            print(
                f"[early_release bench] trigger={arm[0]:<10} prefetch={'on ' if arm[1] else 'off'} "
                f"max-rank median {worst:8.3f}  rank medians "
                + " ".join(f"{v:.3f}" for v in per_rank)
                + f"  vs baseline {worst - max(res[0][base] for res in results):+.3f}"
                + f"  rank0 range {results[0][1][arm][0]:.3f}..{results[0][1][arm][1]:.3f}"
            )
    comm.Barrier()
