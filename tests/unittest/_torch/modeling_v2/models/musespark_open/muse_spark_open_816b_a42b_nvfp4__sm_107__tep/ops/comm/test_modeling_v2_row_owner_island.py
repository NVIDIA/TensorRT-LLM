# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Four-rank GPU test of the row-owner MoE island's transports (``row_owner_moe``, prototype).

* The owner hop (``post_attn_hop_owner``) must store, for every row ``t`` a rank owns
  (``t % 4 == rank``), the wide hop's updated residual and normalized row bit for bit, at
  its compact row ``t // 4``, for the decode row counts 584..768 and 1024, eagerly and in
  CUDA-graph replays with changing inputs.
* The finalize's ``phase1_only`` mode must store the two-shot finalize's row of every token
  a rank owns at its compact row, bit for bit.
* The owner exchange (``owner_dispatch``) must deliver every owner's compact rows at row
  ``4 i + owner`` of every rank's receive view, over calls of changing row and segment
  counts and in graph replays.
* The v2 LL dispatch (``OwnerLLDispatch``) must deliver every owner's rows bit for bit over
  calls of changing row counts and in graph replays with changing inputs, and its device
  epoch must count every executed call (eager, warm-up and replays; not the capture).
* The v2 row publish (``OwnerRowPublish``) must give the served FFN join's rows bit for bit:
  the owner's updated residual, every rank's normalized next-layer rows and, when asked, every
  rank's updated residual rows, over calls of changing row counts that mix calls with and
  without the residual, eagerly and in graph replays with changing inputs, including the -0
  words the served join produces.

Run with four MPI ranks on one four-GPU node, e.g.
``mpirun -n 4 python -m pytest --noconftest <this file>``;
anywhere else every test skips.
"""

import pytest
import torch

try:
    from mpi4py import MPI
except ImportError:  # pragma: no cover - the test needs MPI ranks
    MPI = None

_T = (
    "tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open."
    "muse_spark_open_816b_a42b_nvfp4__sm_107__tep"
)
WORLD = 4
HIDDEN = 8192
ROWS = (584, 648, 704, 744, 768, 1024)
VARIANT = (1e-6, 1e-6, False)
NEG_ZERO_NEXT_COLUMN = 1341
NEG_ZERO_RESIDUAL_COLUMN = 5729


def _world():
    if MPI is None or MPI.COMM_WORLD.Get_size() != WORLD or not torch.cuda.is_available():
        pytest.skip("needs four MPI ranks on one four-GPU node with CUDA")
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    torch.cuda.set_device(rank % 4)
    if torch.cuda.get_device_capability() != (10, 7):
        pytest.skip("needs sm_107")
    return comm, rank


def _agree(comm, ok: bool) -> bool:
    return all(comm.allgather(bool(ok)))


@pytest.fixture(scope="module")
def group():
    import importlib

    from tensorrt_llm.mapping import Mapping

    comm, rank = _world()
    mapping = Mapping(world_size=WORLD, tp_size=WORLD, rank=rank, gpus_per_node=4)
    wide = importlib.import_module(_T + ".ops.allreduce.post_attn_hop_wide")
    owner = importlib.import_module(_T + ".ops.allreduce.post_attn_hop_owner")
    served = wide.WideHopWorkspace(
        mapping, variants=[VARIANT], two_shot_rows=wide.MIN_ROWS, max_rows=wide.WIDE_MAX_ROWS
    )
    own = owner.OwnerHopWorkspace(mapping, variants=[VARIANT], max_rows=wide.WIDE_MAX_ROWS)
    comm.Barrier()
    return comm, rank, mapping, served, own


def _hop_inputs(rows, seed, rank):
    g = torch.Generator(device="cuda").manual_seed(seed)
    gr = torch.Generator(device="cuda").manual_seed(seed * 7 + 1 + rank)
    partial = (torch.randn(rows, HIDDEN, device="cuda", generator=gr) * 3).to(torch.bfloat16)
    residual = torch.randn(rows, HIDDEN, device="cuda", generator=g) * 4
    alpha = torch.rand(HIDDEN, device="cuda", generator=g) + 0.5
    beta = torch.rand(HIDDEN, device="cuda", generator=g) + 0.5
    weight = (torch.randn(HIDDEN, device="cuda", generator=g) * 0.1).to(torch.bfloat16)
    return partial, residual, alpha, beta, weight


def _bits_differ(a, b):
    return int(
        (
            a.contiguous().view(torch.int32 if a.element_size() == 4 else torch.int16)
            != b.contiguous().view(torch.int32 if b.element_size() == 4 else torch.int16)
        )
        .sum()
        .item()
    )


@torch.inference_mode()
def test_owner_hop_rows_are_the_wide_hop_rows_bit_for_bit(group):
    comm, rank, mapping, served, own = group
    bad = 0
    for rows in ROWS:
        for seed in (1, 2):
            p, r, a, b, w = _hop_inputs(rows, 100 * rows + seed, rank)
            upd, nrm = served.launch(p, r, a, b, w, *VARIANT)
            upd_o, nrm_o = own.launch(p, r[rank::WORLD].contiguous(), a, b, w, *VARIANT)
            torch.cuda.synchronize()
            bad += _bits_differ(upd[rank::WORLD], upd_o) + _bits_differ(nrm[rank::WORLD], nrm_o)
    assert _agree(comm, bad == 0), f"rank {rank}: {bad} values differ"


@torch.inference_mode()
def test_owner_hop_graph_replays_changing_inputs(group):
    comm, rank, mapping, served, own = group
    rows = 768
    p, r, a, b, w = _hop_inputs(rows, 5, rank)
    ro = r[rank::WORLD].contiguous()
    upd_o = torch.empty(rows // WORLD, HIDDEN, device="cuda")
    nrm_o = torch.empty(rows // WORLD, HIDDEN, device="cuda", dtype=torch.bfloat16)
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(2):
            own.launch(p, ro, a, b, w, *VARIANT, updated=upd_o, normalized=nrm_o)
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(3):
            own.launch(p, ro, a, b, w, *VARIANT, updated=upd_o, normalized=nrm_o)
    bad = 0
    for seed in (11, 12, 13):
        p2, r2, a2, b2, w2 = _hop_inputs(rows, seed, rank)
        p.copy_(p2)
        ro.copy_(r2[rank::WORLD])
        comm.Barrier()
        graph.replay()
        torch.cuda.synchronize()
        upd, nrm = served.launch(p2, r2, a, b, w, *VARIANT)
        torch.cuda.synchronize()
        bad += _bits_differ(upd[rank::WORLD], upd_o) + _bits_differ(nrm[rank::WORLD], nrm_o)
    assert _agree(comm, bad == 0), f"rank {rank}: {bad} values differ"


def _fin_inputs(tokens, seed, rank):
    gen = torch.Generator().manual_seed(seed * WORLD + rank)
    owners = torch.Generator().manual_seed(10_000 + seed)
    experts = torch.stack([torch.randperm(256, generator=owners)[:8] for _ in range(tokens)])
    local = experts // 64 == rank
    rows = max(1, int(local.sum()))
    index = torch.full((tokens, 8), -1, dtype=torch.int32)
    index[local] = torch.randperm(int(local.sum()), generator=gen).to(torch.int32)
    gemm = torch.randn(rows, 4096, generator=gen).to(torch.bfloat16)
    shared = torch.Generator().manual_seed(seed)
    weights = torch.rand(tokens, 8, generator=shared)
    weights /= weights.sum(-1, keepdim=True)
    norm = torch.randn(4096, generator=shared).to(torch.bfloat16) * 0.03
    return gemm.cuda(), index.reshape(-1).cuda(), weights.reshape(-1).cuda(), norm.cuda()


@torch.inference_mode()
def test_finalize_phase1_rows_are_the_two_shot_rows_bit_for_bit(group):
    import importlib

    comm, rank, mapping, served, own = group
    two = importlib.import_module(_T + ".ops.moe.finalize_two_shot")
    bad = 0
    for cap, counts in ((768, (584, 648, 744, 768)), (1024, (1024,))):
        full = two.FinalizeTwoShotWorkspace(None, mapping, max_rows=cap)
        ph1 = two.FinalizeTwoShotWorkspace(None, mapping, max_rows=cap, phase1_only=True)
        for tokens in counts:
            gemm, index, weights, norm = _fin_inputs(tokens, tokens, rank)
            want = full.launch(gemm, index, weights, norm, 1e-6, tokens)
            got = ph1.launch(gemm, index, weights, norm, 1e-6, tokens)
            torch.cuda.synchronize()
            bad += _bits_differ(want[rank::WORLD], got)
    assert _agree(comm, bad == 0), f"rank {rank}: {bad} values differ"


@torch.inference_mode()
def test_owner_exchange_delivers_every_owner_row(group):
    import importlib

    comm, rank, mapping, served, own = group
    od = importlib.import_module(_T + ".ops.comm.owner_dispatch")
    ex = od.OwnerExchange(mapping, [512, 64, 8, 8], max_rows=1024)
    bad = 0
    for call, (rows, nseg) in enumerate(
        ((768, 4), (744, 4), (584, 2), (1024, 4), (768, 1), (648, 3))
    ):
        full = [
            torch.randint(
                -(2**31),
                2**31 - 1,
                (rows, w),
                dtype=torch.int32,
                device="cuda",
                generator=torch.Generator(device="cuda").manual_seed(call * 10 + s),
            )
            for s, w in enumerate(ex.segments[:nseg])
        ]
        views = ex.push([f[rank::WORLD].contiguous() for f in full], rows)
        torch.cuda.synchronize()
        for f, v in zip(full, views):
            bad += int((f != v).sum().item())
    # Graph replay with changing inputs.
    rows = 768
    src = [torch.zeros(rows // WORLD, w, dtype=torch.int32, device="cuda") for w in ex.segments]
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(2):
            ex.push(src, rows)
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    outs = []
    with torch.cuda.graph(graph):
        for _ in range(2):
            outs.append([v for v in ex.push(src, rows)])
    for seed in (3, 4):
        full = [
            torch.randint(
                -(2**31),
                2**31 - 1,
                (rows, w),
                dtype=torch.int32,
                device="cuda",
                generator=torch.Generator(device="cuda").manual_seed(seed * 100 + i),
            )
            for i, w in enumerate(ex.segments)
        ]
        for d, f in zip(src, full):
            d.copy_(f[rank::WORLD])
        comm.Barrier()
        graph.replay()
        torch.cuda.synchronize()
        for f, v in zip(full, outs[-1]):
            bad += int((f != v).sum().item())
    assert _agree(comm, bad == 0), f"rank {rank}: {bad} words differ"


def _words(rows, widths, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    return [
        torch.randint(-(2**31), 2**31 - 1, (rows, w), dtype=torch.int32, device="cuda", generator=g)
        for w in widths
    ]


def _captured(fn, calls):
    """Two eager warm-up rounds of ``fn`` on a side stream, then a graph of ``calls`` rounds."""
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(2):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    outs = []
    with torch.cuda.graph(graph):
        for _ in range(calls):
            outs.append(fn())
    return graph, outs


@torch.inference_mode()
def test_ll_dispatch_delivers_every_owner_row(group):
    import importlib

    comm, rank, mapping, served, own = group
    od = importlib.import_module(_T + ".ops.comm.owner_dispatch")
    dx = od.OwnerLLDispatch(mapping, (512, 64, 8, 8), max_rows=1024)
    bad, executed = 0, 0
    for call, rows in enumerate((768, 744, 584, 1024, 648, 672)):
        full = _words(rows, dx.segments, 1000 + call)
        got = dx.dispatch([f[rank::WORLD].contiguous() for f in full], rows)
        torch.cuda.synchronize()
        executed += 1
        bad += sum(int((f != v).sum().item()) for f, v in zip(full, got))
    rows = 584
    src = [torch.zeros(rows // WORLD, w, dtype=torch.int32, device="cuda") for w in dx.segments]
    graph, outs = _captured(lambda: [v.clone() for v in dx.dispatch(src, rows)], 2)
    executed += 2
    for seed in (3, 4, 5):
        full = _words(rows, dx.segments, 2000 + seed)
        for d, f in zip(src, full):
            d.copy_(f[rank::WORLD])
        comm.Barrier()
        graph.replay()
        torch.cuda.synchronize()
        executed += 2
        bad += sum(int((f != v).sum().item()) for f, v in zip(full, outs[-1]))
    epoch = dx.epoch()
    assert _agree(comm, bad == 0 and epoch == executed), (
        f"rank {rank}: {bad} words differ, epoch {epoch} after {executed} calls"
    )


def _join_inputs(rows, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    residual = torch.randn(rows, HIDDEN, device="cuda", generator=g) * 2
    branch = torch.randn(rows, HIDDEN, device="cuda", generator=g).to(torch.bfloat16)
    gain = (torch.randn(HIDDEN, device="cuda", generator=g) * 0.1 + 1).to(torch.bfloat16)
    alpha = torch.rand(HIDDEN, device="cuda", generator=g) * 0.2 + 0.9
    beta = torch.rand(HIDDEN, device="cuda", generator=g) * 0.2 + 0.9
    weight = (torch.randn(HIDDEN, device="cuda", generator=g) * 0.1).to(torch.bfloat16)
    # Signed zeros through the join, as the checkpoint makes them: a next-norm weight of exactly
    # -1 (``w + 1 == 0``) gives -0 in the normalized row of every row whose input is negative
    # there, and a zero gain with a -0 residual gives a -0 updated residual.
    weight[NEG_ZERO_NEXT_COLUMN] = -1.0
    gain[NEG_ZERO_RESIDUAL_COLUMN] = 0.0
    residual[:, NEG_ZERO_RESIDUAL_COLUMN] = -0.0
    return residual, branch, gain, alpha, beta, weight


def _negative_zeros(t):
    words = t.contiguous().view(torch.int32 if t.element_size() == 4 else torch.int16)
    return int((words == torch.iinfo(words.dtype).min).sum().item())


def _served_join(residual, branch, gain, alpha, beta, weight):
    from tensorrt_llm._torch.models import modeling_musespark_open as builtin

    rows = residual.shape[0]
    out = torch.empty(rows, HIDDEN, device="cuda", dtype=torch.float32)
    nxt = torch.empty(rows, HIDDEN, device="cuda", dtype=torch.bfloat16)
    builtin._gated_residual_after_rmsnorm_kernel[(rows,)](
        out,
        residual,
        branch,
        gain,
        alpha,
        beta,
        1e-6,
        nxt,
        weight,
        1e-6,
        HIDDEN=HIDDEN,
        BLOCK_H=HIDDEN,
        HAS_GAIN=True,
        FUSE_NEXT_NORM=True,
        NORM_FP32_RESIDUAL=False,
        num_warps=builtin._RESIDUAL_NUM_WARPS,
    )
    return out, nxt


def _publish_bad(want, got_own, views, rank, with_residual):
    want_res, want_nxt = want
    bad = _bits_differ(want_res[rank::WORLD], got_own) + _bits_differ(want_nxt, views[0])
    if with_residual:
        bad += _bits_differ(want_res, views[1])
    return bad


@torch.inference_mode()
def test_row_publish_rows_are_the_join_rows_bit_for_bit(group):
    import importlib

    comm, rank, mapping, served, own = group
    od = importlib.import_module(_T + ".ops.comm.owner_dispatch")
    rp = od.OwnerRowPublish(mapping, max_rows=1024)
    bad = 0
    sequence = (
        (768, False),
        (768, True),
        (584, False),
        (744, True),
        (1024, True),
        (648, False),
        (672, True),
        (584, False),
    )
    for call, (rows, with_residual) in enumerate(sequence):
        r, b, g, a, be, w = _join_inputs(rows, 3000 + call)
        want = _served_join(r, b, g, a, be, w)
        # The served rows hold -0 words, which must travel unchanged.
        if not (_negative_zeros(want[0]) and _negative_zeros(want[1])):
            bad += 1
        got, views = rp.join_publish(
            r[rank::WORLD].contiguous(),
            b[rank::WORLD].contiguous(),
            g,
            a,
            be,
            1e-6,
            w,
            1e-6,
            rows,
            with_residual,
        )
        torch.cuda.synchronize()
        bad += _publish_bad(want, got, views, rank, with_residual)
    # Graph replays with changing inputs; the captured calls alternate the residual segment.
    rows = 744
    flags = (False, True, False, True)
    held = [_join_inputs(rows, 4000 + i) for i in range(len(flags))]
    owned = [
        (r[rank::WORLD].contiguous(), b[rank::WORLD].contiguous(), g, a, be, w)
        for r, b, g, a, be, w in held
    ]
    k = iter(range(10**9))

    def call():
        i = next(k) % len(flags)
        r, b, g, a, be, w = owned[i]
        got, views = rp.join_publish(r, b, g, a, be, 1e-6, w, 1e-6, rows, flags[i])
        return got, views

    graph, outs = _captured(lambda: [call() for _ in flags], 1)
    for seed in (5, 6, 7):
        fresh = [_join_inputs(rows, 5000 + 10 * seed + i) for i in range(len(flags))]
        for dst, src in zip(owned, fresh):
            r, b, g, a, be, w = src
            for d, s in zip(dst, (r[rank::WORLD], b[rank::WORLD], g, a, be, w)):
                d.copy_(s)
        comm.Barrier()
        graph.replay()
        torch.cuda.synchronize()
        last = outs[-1]
        for i, (src, (got, views)) in enumerate(zip(fresh, last)):
            want = _served_join(*src)
            want_res, want_nxt = want
            bad += _bits_differ(want_res[rank::WORLD], got)
            if i == len(flags) - 1:
                # Only the last call's receive rows survive the sequence (the next call of
                # the other parity clears them).
                bad += _publish_bad(want, got, views, rank, flags[i])
    assert _agree(comm, bad == 0), f"rank {rank}: {bad} values differ"
