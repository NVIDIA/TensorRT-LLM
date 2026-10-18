# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Multi-GPU test for the musespark_open drafter_hop entry.

The entry claims to be the built-in chain it replaces, bit for bit: the MNNVL
one-shot all-reduce of a drafter projection's partial (``AllReduce`` with
strategy MNNVL), the DFlash 2 finish-side block convolution
(``dflash2_block_conv``, itself bitwise the built-in torch chain), and the
built-in ``RMSNorm`` with a residual (flashinfer's fused add + RMSNorm, in
place). Every check runs both on the same inputs and compares the normed
output and the updated residual as bits, on both transports (the hop publishing
the partial, and ``direct``: the partial stored into the site's area through
the multicast address first, as the projection GEMM does with direct publication). The
cells are additionally gated against the entry's reference within a band.

It is a collective test: every rank of a 4-rank (one node) or 8-rank (two
nodes) group runs this same file and every test is a collective in the same
order on every rank. Under MPI, one process per rank::

    srun -N2 --segment=2 --ntasks-per-node=4 --mpi=pmix \\
        python3 -m pytest -q -p no:cacheprovider --noconftest \\
        -o addopts="-p test_common.magic_import_hooks" <this file>

or from one pytest process on 4 GPUs with ``MODELING_V2_DRAFTER_HOP_RANK_JOB=1`` set per
rank over ``torch.distributed`` (``TLLM_DISABLE_MPI=1``). Outside either it is
skipped. ``MODELING_V2_TIMING=1`` also runs the microbenchmark.
"""

import os
import statistics
import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.ops.allreduce.drafter_hop import (  # noqa: E501
    HIDDEN,
    MAX_ROWS,
    DrafterHopWorkspace,
    hop_layout_ok,
    hop_takes,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.ops.allreduce.drafter_hop import (  # noqa: E501
    drafter_hop as op,
)

__extra_import_path__ = ["../../../../.."]
from _validating import (  # noqa: E402,F401 — path above; fixture by name
    assert_within_ulp,
    certifying,
)

EPS = 1e-6  # the drafter checkpoint's rms_norm_eps
SITES = 10
GROUPS = HIDDEN // 16
RANK_JOB_ENV = "MODELING_V2_DRAFTER_HOP_RANK_JOB"
TRANSPORTS = ("mc", "direct", "mc_early")


class _TorchGroup:
    """The MPI-communicator calls this file makes, over torch.distributed's CPU backend."""

    def __init__(self):
        import torch.distributed as dist

        self._dist = dist

    def Get_size(self):
        return self._dist.get_world_size()

    def Get_rank(self):
        return self._dist.get_rank()

    def Barrier(self):
        self._dist.barrier()

    def allreduce(self, value, op="sum"):
        t = torch.tensor([value], dtype=torch.float64)
        reduce = self._dist.ReduceOp.MAX if op == "max" else self._dist.ReduceOp.SUM
        self._dist.all_reduce(t, op=reduce)
        return type(value)(t.item())


def _group():
    if os.environ.get(RANK_JOB_ENV) == "1":
        import torch.distributed as dist

        rank = int(os.environ["RANK"])
        torch.cuda.set_device(rank % torch.cuda.device_count())
        if not dist.is_initialized():
            dist.init_process_group(
                "cuda:nccl,cpu:gloo",
                init_method="env://",
                device_id=torch.device("cuda", torch.cuda.current_device()),
            )
        comm = _TorchGroup()
        return comm, comm.Get_size(), comm.Get_rank()
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
    or torch.cuda.get_device_capability() not in ((10, 3), (10, 7)),
    reason="collective test: run under a 4- or 8-rank launch on sm_103 or sm_107",
)


def _max_over_ranks(value: float) -> float:
    if isinstance(COMM, _TorchGroup):
        return COMM.allreduce(value, op="max")
    from mpi4py import MPI

    return COMM.allreduce(value, op=MPI.MAX)


def _bits(t: torch.Tensor) -> torch.Tensor:
    return t.view(torch.int32) if t.dtype == torch.float32 else t.view(torch.int16)


def _differing(a: torch.Tensor, b: torch.Tensor) -> int:
    return int((_bits(a) != _bits(b)).sum())


def _agree(ok: bool) -> bool:
    """True only if every rank says so (keeps ranks on the same control path)."""
    return COMM.allreduce(int(ok)) == WORLD


def _denormals(shape, g) -> torch.Tensor:
    """BF16 denormals of both signs (|x| < 2**-126)."""
    mag = torch.exp2(-127.0 - 6.0 * torch.rand(shape, generator=g))
    sign = torch.where(torch.rand(shape, generator=g) < 0.5, -1.0, 1.0)
    return (sign * mag).to(torch.bfloat16)


def _cancelling(rows: int, g: torch.Generator):
    """Partials whose FP32 sum depends on the order the ranks are added in (the post-attention hop's construction)."""
    big = torch.exp2(torch.randint(0, 8, (rows, HIDDEN), generator=g).float())
    vals = torch.randn(WORLD, rows, HIDDEN, generator=g) * big * 2.0**-20
    perm = torch.argsort(torch.rand(WORLD, rows, HIDDEN, generator=g), dim=0)
    vals.scatter_(0, perm[0:1], big.unsqueeze(0))
    vals.scatter_(0, perm[1:2], -big.unsqueeze(0))
    small = big * 2.0**-26 * (1.0 + torch.rand(rows, HIDDEN, generator=g))
    vals.scatter_(0, perm[2:3], small.unsqueeze(0))
    return [v.to(torch.bfloat16) for v in vals]


def _sum_bf16(partials, order) -> torch.Tensor:
    acc = torch.zeros(partials[0].shape, dtype=torch.float32)
    for r in order:
        acc = acc + partials[r].cpu().float()
    return acc.to(torch.bfloat16)


def _inputs(rows: int, seed: int, zeros: bool = False, cancel: bool = False):
    """Every rank's partial (rank order) and the shared residual / coefficients / base / weight.

    Each rank regenerates all of them from the seed. ``coefficients`` is the
    ``[rows, 2, 2, 512]`` view of a ``[rows, 2048]`` projection, as the
    drafter's ``prepare`` returns it; the hop reads its finish side ``[:, 1]``.
    """
    g = torch.Generator().manual_seed(seed)
    channel = torch.exp(0.75 * torch.randn(HIDDEN, generator=g))
    partials = [
        (0.5 * torch.randn(rows, HIDDEN, generator=g) * channel).to(torch.bfloat16)
        for _ in range(WORLD)
    ]
    if cancel:
        partials = _cancelling(rows, g)
    residual = (2.0 * torch.randn(rows, HIDDEN, generator=g)).to(torch.bfloat16)
    projection = (0.3 * torch.randn(rows, 2 * 2 * GROUPS, generator=g)).to(torch.bfloat16)
    base = (0.5 + 0.3 * torch.randn(2, 2, HIDDEN, generator=g)).to(torch.bfloat16)
    weight = (1.0 + 0.2 * torch.randn(HIDDEN, generator=g)).to(torch.bfloat16)
    if zeros:
        for part in partials:
            pick = torch.rand(rows, HIDDEN, generator=g)
            part[pick < 0.06] = 0.0
            part[(pick >= 0.06) & (pick < 0.12)] = -0.0
            den = (pick >= 0.12) & (pick < 0.18)
            part[den] = _denormals((int(den.sum()),), g)
        for part in partials:
            part[:, 0:64] = -0.0
        for r in range(0, WORLD, 2):
            partials[r + 1][:, 64:128] = -partials[r][:, 64:128]
        pick = torch.rand(rows, HIDDEN, generator=g)
        residual[pick < 0.05] = -0.0
        den = (pick >= 0.05) & (pick < 0.10)
        residual[den] = _denormals((int(den.sum()),), g)
        # Rows whose sum cancels exactly against the convolution's output are rare by
        # chance; make the residual the negated partial sum's rough scale in a band.
        residual[:, 128:192] = 0.0
        projection[:, ::7] = 0.0
        projection[:, 3::11] = _denormals(projection[:, 3::11].shape, g)
        base[:, :, 256:272] = -0.0
        base[:, :, 272:288] = _denormals((2, 2, 16), g)
        weight[300:310] = -0.0
        weight[310:320] = _denormals((10,), g)
    dev = torch.device("cuda", torch.cuda.current_device())
    return (
        [p.to(dev) for p in partials],
        residual.to(dev),
        projection.to(dev).view(rows, 2, 2, GROUPS),
        base.to(dev),
        weight.to(dev),
    )


class _BuiltinMnnvlAllReduce:
    """``AllReduce(strategy=MNNVL)``'s plain all-reduce for the ``torch.distributed`` launch.

    The same construction as the post-attention hop test's.
    """

    def __init__(self, mapping):
        from tensorrt_llm._torch.distributed.ops import MNNVLAllReduce

        self.mnnvl_allreduce = MNNVLAllReduce(mapping, torch.bfloat16)

    def __call__(self, tensor):
        from tensorrt_llm.functional import AllReduceParams

        out = self.mnnvl_allreduce(tensor.contiguous(), all_reduce_params=AllReduceParams())
        assert out is not None, "MNNVLAllReduce declined a plain all-reduce"
        return out


def _builtin_all_reduce(mapping):
    if isinstance(COMM, _TorchGroup):
        return _BuiltinMnnvlAllReduce(mapping)
    from tensorrt_llm._torch.distributed import AllReduce
    from tensorrt_llm.functional import AllReduceStrategy

    all_reduce = AllReduce(mapping=mapping, strategy=AllReduceStrategy.MNNVL, dtype=torch.bfloat16)
    assert all_reduce.mnnvl_allreduce is not None, "the built-in MNNVL all-reduce is not active"
    return all_reduce


@pytest.fixture(scope="module")
def env():
    from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.ops.spec.dflash2_block_conv import (  # noqa: E501
        dflash2_block_conv,
    )
    from tensorrt_llm._torch.models.modeling_dflash import dflash2_grouped_conv
    from tensorrt_llm._torch.modules.rms_norm import RMSNorm
    from tensorrt_llm.mapping import Mapping

    local = int(os.environ.get("SLURM_LOCALID", RANK % torch.cuda.device_count()))
    torch.cuda.set_device(local)
    mapping = Mapping(world_size=WORLD, rank=RANK, gpus_per_node=4, tp_size=WORLD)
    workspaces = {
        "mc": DrafterHopWorkspace(mapping, sites=SITES, eps=(EPS,)),
        "direct": DrafterHopWorkspace(mapping, sites=SITES, eps=(EPS,), direct=True),
        "mc_early": DrafterHopWorkspace(mapping, sites=SITES, eps=(EPS,), early_trigger=True),
    }
    legacy_ar = _builtin_all_reduce(mapping)
    with torch.device("cuda", local):
        norm = RMSNorm(hidden_size=HIDDEN, eps=EPS, dtype=torch.bfloat16)

    def legacy(partial, residual, coefficients, base, weight, block, torch_conv=False):
        """The built-in chain; returns (normed, updated residual). Does not modify its inputs."""
        reduced = legacy_ar(partial)
        conv = dflash2_grouped_conv if torch_conv else dflash2_block_conv
        hidden = conv(reduced, coefficients[:, 1], base[1], block, 16)
        residual = residual.clone()
        with torch.no_grad():
            norm.weight.copy_(weight)
        hidden, residual = norm(hidden, residual)
        return hidden, residual

    COMM.Barrier()
    return dict(mapping=mapping, workspaces=workspaces, legacy=legacy, legacy_ar=legacy_ar)


class _Sites:
    """Calls a workspace's sites in order, as the drafter does."""

    def __init__(self, ws: DrafterHopWorkspace):
        self.ws = ws

    def __call__(self, partial, residual, coefficients, base, weight, block):
        ws = self.ws
        site = ws.next_site
        residual = residual.clone()
        if ws.direct:
            ws.direct_output(site, partial.shape[0]).copy_(partial)
            partial = None
        out = op(site, partial, residual, coefficients[:, 1], base[1], weight, EPS, block, ws)
        return out, residual


def _block(cell_spec, rows):
    return cell_spec.get("block", rows)


@pytest.mark.parametrize("transport", TRANSPORTS)
@pytest.mark.parametrize("cell", op.CELLS, ids=lambda c: c.why[:40].replace(" ", "_"))
def test_cell_is_the_builtin_chain_bit_for_bit(env, certifying, cell, transport):  # noqa: F811
    rows = cell.spec["rows"]
    block = _block(cell.spec, rows)
    seed = zlib.crc32(f"{cell.why}|{transport}".encode())
    partials, residual, coefficients, base, weight = _inputs(
        rows, seed, cell.spec.get("zeros", False), cell.spec.get("cancel", False)
    )
    partial = partials[RANK]
    assert hop_takes(rows, residual, coefficients[:, 1], base[1], weight, block)
    assert hop_layout_ok(partial, residual, coefficients[:, 1], base[1], weight)

    certifying(op)
    got_out, got_res = _Sites(env["workspaces"][transport])(
        partial, residual, coefficients, base, weight, block
    )
    want_out, want_res = env["legacy"](partial, residual, coefficients, base, weight, block)
    torch_out, torch_res = env["legacy"](
        partial, residual, coefficients, base, weight, block, torch_conv=True
    )
    torch.cuda.synchronize()
    mismatch = (_differing(got_out, want_out), _differing(got_res, want_res))
    conv_bitwise = _differing(torch_out, want_out) + _differing(torch_res, want_res) == 0
    ref_out, ref_res = op.reference(
        partials, residual, coefficients[:, 1], base[1], weight, EPS, block
    )
    order_observable = True
    if cell.spec.get("cancel"):
        forward = _sum_bf16(partials, range(WORLD))
        backward = _sum_bf16(partials, reversed(range(WORLD)))
        order_observable = _differing(forward, backward) > 1000
    identical = _agree(mismatch == (0, 0))
    conv_agree = _agree(conv_bitwise)
    order_agree = _agree(order_observable)
    assert identical, (
        f"rank {RANK}: {mismatch[0]} normed and {mismatch[1]} residual values differ from the "
        "built-in all-reduce + block conv + fused add-RMSNorm"
    )
    assert conv_agree, "the block convolution is not bitwise the built-in torch chain"
    assert order_agree, "the cancelling inputs do not make the summation order observable"
    if not cell.spec.get("zeros"):
        # Both sides share the BF16 all-reduce, convolution and residual sum; what
        # remains is the norm's FP32 reduction and approximate rsqrt against the
        # float64 reference, and the final BF16 rounding.
        assert_within_ulp(got_out, ref_out, element_ulp=2.0, rms_ulp=0.1)
        # The residual is the FP32 sum rounded to BF16 on both sides; the reference
        # sums in float64, so at most a double-rounding ulp apart, rarely.
        assert_within_ulp(got_res, ref_res, element_ulp=1.0, rms_ulp=0.01)


@pytest.mark.parametrize("transport", TRANSPORTS)
def test_a_long_burst_over_the_sites_stays_identical(env, transport):
    """200 calls in site order, row counts changing between forwards, built-in calls between."""
    ws = env["workspaces"][transport]
    g = torch.Generator().manual_seed(1234)
    bad = 0
    rows = 8
    for call in range(200):
        if ws.next_site == 0:
            rows = int(torch.randint(1, MAX_ROWS + 1, (1,), generator=g))
        block = rows if rows < 5 or call % 3 else 4
        cancel = bool(torch.randint(0, 4, (1,), generator=g) == 0)
        partials, residual, coefficients, base, weight = _inputs(rows, 10_000 + call, cancel=cancel)
        got = _Sites(ws)(partials[RANK], residual, coefficients, base, weight, block)
        want = env["legacy"](partials[RANK], residual, coefficients, base, weight, block)
        bad += _differing(got[0], want[0]) + _differing(got[1], want[1])
    torch.cuda.synchronize()
    assert _agree(bad == 0), f"rank {RANK}: {bad} values differ over the burst"


def test_back_to_back_hops_with_no_kernel_between(env):
    """Hops on one stream with nothing between them: each one's dependent is the next hop."""
    g = torch.Generator().manual_seed(4321)
    calls = []
    for call in range(3 * SITES):
        rows = 8 if call < SITES else int(torch.randint(1, MAX_ROWS + 1, (1,), generator=g))
        calls.append((rows, _inputs(rows, 80_000 + call)))
    for transport in ("mc", "mc_early"):
        ws = env["workspaces"][transport]
        while ws.next_site != 0:  # align to a forward boundary
            p, r, c, b, w = _inputs(8, 1)
            _Sites(ws)(p[RANK], r, c, b, w, 8)
        torch.cuda.synchronize()
        COMM.Barrier()
        outs = [_Sites(ws)(p[RANK], r, c, b, w, rows) for rows, (p, r, c, b, w) in calls]
        bad = 0
        for (rows, (p, r, c, b, w)), got in zip(calls, outs):
            want = env["legacy"](p[RANK], r, c, b, w, rows)
            bad += _differing(got[0], want[0]) + _differing(got[1], want[1])
        torch.cuda.synchronize()
        assert _agree(bad == 0), f"rank {RANK}: {bad} values differ back to back ({transport})"


@pytest.mark.parametrize("transport", ["mc", "direct"])
def test_a_delayed_rank_is_waited_for(env, transport):
    """One rank arrives late (a GPU spin before its call); every rank still agrees."""
    ws = env["workspaces"][transport]
    bad = 0
    for late in range(WORLD):
        partials, residual, coefficients, base, weight = _inputs(8, 20_000 + late)
        want = env["legacy"](partials[RANK], residual, coefficients, base, weight, 8)
        if RANK == late:
            torch.cuda._sleep(2_000_000)  # ~1 ms
        got = _Sites(ws)(partials[RANK], residual, coefficients, base, weight, 8)
        bad += _differing(got[0], want[0]) + _differing(got[1], want[1])
    torch.cuda.synchronize()
    assert _agree(bad == 0), f"rank {RANK}: {bad} values differ with a delayed rank"


@pytest.mark.parametrize("transport", TRANSPORTS)
def test_graph_replay_of_a_whole_forward_with_eager_forwards_between(env, transport):
    """A captured forward's 10 sites replay identically with new inputs; eager forwards interleave."""
    ws = env["workspaces"][transport]
    while ws.next_site != 0:
        p, r, c, b, w = _inputs(8, 2)
        _Sites(ws)(p[RANK], r, c, b, w, 8)
    static = [_inputs(8, 30_000 + i) for i in range(SITES)]
    static_inputs = [(s[0][RANK].clone(), *[t.clone() for t in s[1:]]) for s in static]

    def forward():
        outs = []
        for partial, residual, coefficients, base, weight in static_inputs:
            outs.append(_Sites(ws)(partial, residual, coefficients, base, weight, 8))
        return outs

    forward()  # warm
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        outs = forward()
    bad = 0
    for replay in range(20):
        for i in range(SITES):
            fresh = _inputs(8, 40_000 + 100 * replay + i)
            for dst, src in zip(static_inputs[i], (fresh[0][RANK], *fresh[1:])):
                dst.copy_(src)
        graph.replay()
        wants = [env["legacy"](p, r, c, b, w, 8) for p, r, c, b, w in static_inputs]
        for (got_o, got_r), (want_o, want_r) in zip(outs, wants):
            bad += _differing(got_o, want_o) + _differing(got_r, want_r)
        # An eager forward on the same workspace between replays.
        for i in range(SITES):
            p, r, c, b, w = _inputs(8, 50_000 + 100 * replay + i)
            got = _Sites(ws)(p[RANK], r, c, b, w, 8)
            want = env["legacy"](p[RANK], r, c, b, w, 8)
            bad += _differing(got[0], want[0]) + _differing(got[1], want[1])
    torch.cuda.synchronize()
    assert _agree(bad == 0), f"rank {RANK}: {bad} values differ across graph replays"


def test_unsupported_inputs_and_out_of_order_sites_are_refused(env):
    ws = env["workspaces"]["mc"]
    partials, residual, coefficients, base, weight = _inputs(8, 7)
    partial = partials[RANK]
    site = ws.next_site
    with pytest.raises(ValueError):
        op(site, partial[:, :4096], residual, coefficients[:, 1], base[1], weight, EPS, 8, ws)
    with pytest.raises(ValueError):
        op(site, partial, residual.float(), coefficients[:, 1], base[1], weight, EPS, 8, ws)
    nine = torch.zeros(MAX_ROWS + 1, HIDDEN, dtype=torch.bfloat16, device=partial.device)
    with pytest.raises(ValueError):
        op(site, nine, nine, coefficients[:, 1], base[1], weight, EPS, 8, ws)
    with pytest.raises(ValueError):
        op(site, None, residual.clone(), coefficients[:, 1], base[1], weight, EPS, 8, ws)
    with pytest.raises(RuntimeError):
        op(site + 1, partial, residual.clone(), coefficients[:, 1], base[1], weight, EPS, 8, ws)
    assert ws.next_site == site
    got = _Sites(ws)(partial, residual, coefficients, base, weight, 8)
    want = env["legacy"](partial, residual, coefficients, base, weight, 8)
    torch.cuda.synchronize()
    assert _agree(_differing(got[0], want[0]) + _differing(got[1], want[1]) == 0)


def test_a_direct_slot_is_never_seen_before_its_final_value(env):
    """Under ``direct`` a GEMM stores into the slot that peers poll: no transient value may show.

    Each rank's projection (the drafter o_proj shape, ``[8, 1024] x [1024, 8192]``
    on the drafter's fixed split-K tactic) writes straight into the multicast slot,
    and the hop then reduces it; the result must equal the built-in chain fed
    the same GEMM's local output, over many forwards (a poller that saw a
    partial tile would sum it).
    """
    from flashinfer.gemm.kernels.dense_bf16_gemm_sm100_splitk import SplitKTactic, run_splitk_dense

    from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.ops.gemm.fixed_low_m_gemm import (  # noqa: E501
        tactic_for,
    )

    ws = env["workspaces"]["direct"]
    while ws.next_site != 0:
        p, r, c, b, w = _inputs(8, 3)
        _Sites(ws)(p[RANK], r, c, b, w, 8)
    dev = torch.device("cuda", torch.cuda.current_device())
    g = torch.Generator().manual_seed(77)
    weights = [(0.02 * torch.randn(HIDDEN, 1024, generator=g)).to(torch.bfloat16).to(dev)]
    tactic = SplitKTactic(*tactic_for(8, 1024, HIDDEN))
    bad = 0
    for call in range(6 * SITES):
        x = (torch.randn(8, 1024, generator=g) * (1 + RANK)).to(torch.bfloat16).to(dev)
        _, residual, coefficients, base, weight = _inputs(8, 60_000 + call)
        site = ws.next_site
        run_splitk_dense(x, weights[0].t(), None, ws.direct_output(site, 8), True, tactic)
        local = torch.empty(8, HIDDEN, dtype=torch.bfloat16, device=dev)
        run_splitk_dense(x, weights[0].t(), None, local, True, tactic)
        res = residual.clone()
        out = op(site, None, res, coefficients[:, 1], base[1], weight, EPS, 8, ws)
        want = env["legacy"](local, residual, coefficients, base, weight, 8)
        bad += _differing(out, want[0]) + _differing(res, want[1])
    torch.cuda.synchronize()
    assert _agree(bad == 0), f"rank {RANK}: {bad} values differ with the GEMM publishing directly"


def _align(ws: DrafterHopWorkspace) -> None:
    """Call sites until the workspace stands at a forward boundary (site 0)."""
    while ws.next_site != 0:
        p, r, c, b, w = _inputs(8, 1)
        _Sites(ws)(p[RANK], r, c, b, w, 8)


def _divisors(rows: int):
    return [d for d in range(1, rows + 1) if rows % d == 0]


@pytest.mark.parametrize("transport", TRANSPORTS)
def test_every_row_count_and_block_size_is_the_builtin_chain(env, transport):
    """Rows 1..8 x every block size dividing them x {plain, zeros/-0/denormals, cancelling}."""
    ws = env["workspaces"][transport]
    bad, cases = 0, 0
    for rows in range(1, MAX_ROWS + 1):
        for block in _divisors(rows):
            for variant, kwargs in (
                ("plain", {}),
                ("zeros", {"zeros": True}),
                ("cancel", {"cancel": True}),
            ):
                seed = zlib.crc32(f"{rows}|{block}|{variant}".encode())
                p, r, c, b, w = _inputs(rows, seed, **kwargs)
                got = _Sites(ws)(p[RANK], r, c, b, w, block)
                want = env["legacy"](p[RANK], r, c, b, w, block)
                bad += _differing(got[0], want[0]) + _differing(got[1], want[1])
                cases += 1
    torch.cuda.synchronize()
    assert cases == 60
    assert _agree(bad == 0), f"rank {RANK}: {bad} values differ over {cases} row/block cases"


@pytest.mark.parametrize("transport", ["mc", "mc_early"])
def test_independent_delays_on_every_rank_over_3000_calls(env, transport):
    """Each rank sleeps before ~30 % of its calls, independently; rows / block change per forward."""
    ws = env["workspaces"][transport]
    _align(ws)
    torch.cuda.synchronize()
    COMM.Barrier()
    shared = torch.Generator().manual_seed(77)
    own = torch.Generator().manual_seed(1000 + RANK)
    bad, rows, block = 0, 8, 8
    for call in range(3_000):
        if ws.next_site == 0:
            rows = int(torch.randint(1, MAX_ROWS + 1, (1,), generator=shared))
            choices = _divisors(rows)
            block = choices[int(torch.randint(0, len(choices), (1,), generator=shared))]
        p, r, c, b, w = _inputs(rows, 200_000 + call)
        if float(torch.rand(1, generator=own)) < 0.3:
            torch.cuda._sleep(int(torch.randint(1_000, 60_000, (1,), generator=own)))
        got = _Sites(ws)(p[RANK], r, c, b, w, block)
        want = env["legacy"](p[RANK], r, c, b, w, block)
        bad += _differing(got[0], want[0]) + _differing(got[1], want[1])
    torch.cuda.synchronize()
    assert _agree(bad == 0), f"rank {RANK}: {bad} values differ with independent rank delays"


@pytest.mark.parametrize("transport", ["mc", "mc_early"])
def test_two_graphs_of_different_row_counts_replay_alternately(env, transport):
    """Forwards of 8 rows / block 8 and 4 rows / block 2 captured, replayed in turn, eager 3-row forwards between."""
    ws = env["workspaces"][transport]
    _align(ws)
    torch.cuda.synchronize()
    static = {}
    for rows, block in ((8, 8), (4, 2)):
        items = []
        for i in range(SITES):
            p, r, c, b, w = _inputs(rows, 300_000 + 10 * rows + i)
            items.append([p[RANK].clone(), r, c, b, w])
        static[rows] = (block, items)

    def forward(rows):
        block, items = static[rows]
        return [_Sites(ws)(*item, block) for item in items]

    graphs, outs = {}, {}
    for rows in static:
        forward(rows)
        torch.cuda.synchronize()
        COMM.Barrier()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            outs[rows] = forward(rows)
        graphs[rows] = graph
    bad = 0
    for rep in range(40):
        rows = 8 if rep % 2 == 0 else 4
        block, items = static[rows]
        for i in range(SITES):
            fresh = _inputs(rows, 400_000 + 100 * rep + i, cancel=rep % 5 == 0)
            for dst, src in zip(items[i], (fresh[0][RANK], *fresh[1:])):
                dst.copy_(src)
        graphs[rows].replay()
        for (got_o, got_r), item in zip(outs[rows], items):
            want_o, want_r = env["legacy"](*item, block)
            bad += _differing(got_o, want_o) + _differing(got_r, want_r)
        if rep % 3 == 2:
            for i in range(SITES):
                p, r, c, b, w = _inputs(3, 500_000 + 100 * rep + i)
                got = _Sites(ws)(p[RANK], r, c, b, w, 3)
                want = env["legacy"](p[RANK], r, c, b, w, 3)
                bad += _differing(got[0], want[0]) + _differing(got[1], want[1])
    torch.cuda.synchronize()
    assert _agree(bad == 0), f"rank {RANK}: {bad} values differ across the two graphs"


class _CaptureAbort(Exception):
    pass


@pytest.mark.parametrize("transport", ["mc", "mc_early"])
def test_a_capture_that_fails_part_way_gives_the_site_order_back(env, transport):
    """A forward raising mid-capture leaves ``next_site`` past 0; the next forward resumes at site 0.

    Nothing of the failed capture runs, so the hops after it are bitwise the
    built-in chain, eager and replayed. An eager forward that stops part way
    is refused at the next forward's start instead.
    """
    ws = env["workspaces"][transport]
    _align(ws)
    static = []
    for i in range(SITES):
        p, r, c, b, w = _inputs(8, 600_000 + i)
        static.append([p[RANK].clone(), r, c, b, w])

    def forward(stop=None):
        ws.begin_forward()
        outs = []
        for i, item in enumerate(static):
            if i == stop:
                raise _CaptureAbort()
            outs.append(_Sites(ws)(*item, 8))
        return outs

    forward()
    torch.cuda.synchronize()
    COMM.Barrier()
    failed = torch.cuda.CUDAGraph()
    with pytest.raises(_CaptureAbort):
        with torch.cuda.graph(failed):
            forward(stop=4)
    del failed
    stopped_at = ws.next_site
    bad = 0
    # The next forward (eager) starts at site 0 and matches the built-in chain.
    for i, item in enumerate(static):
        if i == 0:
            ws.begin_forward()
            resumed_at = ws.next_site
        got = _Sites(ws)(*item, 8)
        want = env["legacy"](*item, 8)
        bad += _differing(got[0], want[0]) + _differing(got[1], want[1])
    # A complete capture afterwards, replayed with fresh inputs.
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        outs = forward()
    for rep in range(10):
        for i in range(SITES):
            fresh = _inputs(8, 610_000 + 100 * rep + i)
            for dst, src in zip(static[i], (fresh[0][RANK], *fresh[1:])):
                dst.copy_(src)
        graph.replay()
        for (got_o, got_r), item in zip(outs, static):
            want_o, want_r = env["legacy"](*item, 8)
            bad += _differing(got_o, want_o) + _differing(got_r, want_r)
    # An eager forward stopping part way: the next start refuses, the order stays put.
    with pytest.raises(_CaptureAbort):
        forward(stop=3)
    with pytest.raises(RuntimeError, match="outside graph capture"):
        ws.begin_forward()
    eager_stop = ws.next_site
    for item in static[eager_stop:]:  # finish that forward's sites in order
        got = _Sites(ws)(*item, 8)
        want = env["legacy"](*item, 8)
        bad += _differing(got[0], want[0]) + _differing(got[1], want[1])
    torch.cuda.synchronize()
    order = (stopped_at, resumed_at, eager_stop, ws.next_site)
    assert _agree(order == (4, 0, 3, 0) and bad == 0), (
        f"rank {RANK}: site order {order} (want (4, 0, 3, 0)); {bad} values differ around a "
        "failed capture"
    )


@pytest.mark.skipif(os.environ.get("MODELING_V2_TIMING") != "1", reason="set MODELING_V2_TIMING=1")
def test_microbenchmark(env):
    """The drafter's hop chain in one graph: built-in three kernels vs the fused hop.

    Per forward 5 layers x (o_proj GEMM [8, 1024] x [1024, 8192] -> hop -> the next
    conv projection [8, 8192] x [8192, 2048], down GEMM [8, 3584] x [3584, 8192]
    -> hop -> the next conv projection), every GEMM on the drafter's fixed split-K tactic
    with PDL and distinct cold weights, the hop's output feeding the next GEMM.
    Per arm the median of 2 x 5 x 50 replays, arms timed in order and reversed;
    max over ranks; microseconds per hop site. Also the hops alone.
    """
    from flashinfer.gemm.kernels.dense_bf16_gemm_sm100_splitk import SplitKTactic, run_splitk_dense

    from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.ops.gemm.fixed_low_m_gemm import (  # noqa: E501
        fixed_low_m_gemm,
        tactic_for,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.ops.spec.dflash2_block_conv import (  # noqa: E501
        dflash2_block_conv,
    )

    dev = torch.device("cuda", torch.cuda.current_device())
    g = torch.Generator().manual_seed(5)

    def w(n, k):
        return (0.02 * torch.randn(n, k, generator=g)).to(torch.bfloat16).to(dev)

    layers = 5
    o_w = [w(HIDDEN, 1024) for _ in range(layers)]
    d_w = [w(HIDDEN, 3584) for _ in range(layers)]
    c_w = [w(2048, HIDDEN) for _ in range(2 * layers)]
    partials, residual0, coefficients0, base, weight = _inputs(8, 99)
    xo = partials[RANK][:, :1024].contiguous()
    xd = (torch.randn(8, 3584, generator=g)).to(torch.bfloat16).to(dev)

    def gemm(x, weight_):
        return fixed_low_m_gemm(x, weight_, tactic_for(8, x.shape[1], weight_.shape[0]), True)

    def chain(arm, with_gemm):
        residual = residual0.clone()
        coefficients = coefficients0
        h = None
        for i in range(2 * layers):
            if with_gemm:
                # The projection's input is static (its producer is not part of the
                # hop chain); the chain is serial through the convolution coefficients.
                src = xo if i % 2 == 0 else xd
                wt = o_w[i // 2] if i % 2 == 0 else d_w[i // 2]
                if arm == "direct":
                    ws = env["workspaces"]["direct"]
                    out = ws.direct_output(ws.next_site, 8)
                    tactic = SplitKTactic(*tactic_for(8, src.shape[1], HIDDEN))
                    run_splitk_dense(src, wt.t(), None, out, True, tactic)
                    part = None
                else:
                    part = gemm(src, wt)
            else:
                part = partials[RANK]
                if arm == "direct":
                    ws = env["workspaces"]["direct"]
                    ws.direct_output(ws.next_site, 8).copy_(part)
                    part = None
            if arm == "builtin":
                reduced = env["legacy_ar"](part)
                hidden = dflash2_block_conv(reduced, coefficients[:, 1], base[1], 8, 16)
                torch.ops.trtllm.flashinfer_fused_add_rmsnorm(hidden, residual, weight, EPS)
                h = hidden
            else:
                ws = env["workspaces"][arm]
                h = op(
                    ws.next_site, part, residual, coefficients[:, 1], base[1], weight, EPS, 8, ws
                )
            if with_gemm:
                coefficients = gemm(h, c_w[i]).view(8, 2, 2, GROUPS)
        return h

    names = ("builtin", "mc", "mc_early", "direct")
    results = {}
    for with_gemm in (False, True):
        graphs = {}
        for name in names:
            if name != "builtin":
                ws = env["workspaces"][name]
                while ws.next_site != 0:
                    p, r, c, b, wt = _inputs(8, 4)
                    _Sites(ws)(p[RANK], r, c, b, wt, 8)
            chain(name, with_gemm)
            torch.cuda.synchronize()
            COMM.Barrier()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                chain(name, with_gemm)
            for _ in range(10):
                graph.replay()
            graphs[name] = graph
        torch.cuda.synchronize()
        samples = {name: [] for name in names}
        for order in (names, tuple(reversed(names))):
            for name in order:
                for _ in range(5):
                    COMM.Barrier()
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    start.record()
                    for _ in range(50):
                        graphs[name].replay()
                    end.record()
                    end.synchronize()
                    samples[name].append(start.elapsed_time(end) * 1000.0 / 50 / (2 * layers))
        for name in names:
            results[(with_gemm, name)] = _max_over_ranks(statistics.median(samples[name]))
        del graphs
    if RANK == 0:
        print(f"\ndrafter_hop microbenchmark, world {WORLD}, us per hop site (max over ranks)")
        for with_gemm in (False, True):
            label = "GEMM chain" if with_gemm else "hops only "
            print(label + " | " + " ".join(f"{n}={results[(with_gemm, n)]:7.2f}" for n in names))
