# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU certification matrix for ``m3_mc_allreduce_norm`` and ``m3_mc_finalize_allreduce_norm``, 4 ranks.

Its own launcher, as the catalog's collective matrices are: run plainly it spawns
one rank process per device in CUDA_VISIBLE_DEVICES and kills them all on the
first failure or on the deadline. A rank whose peers never push polls forever,
so the deadline is what keeps a broken exchange from wedging the calling run.

    CUDA_VISIBLE_DEVICES=0,1,2,3 python _m3_mc_allreduce_norm_op_matrix.py

Each rank builds the three multicast workspaces (8 / 16 / 48 tokens) the target
builds, over the torch.distributed group (the non-MPI path of
``_make_mnnvl_mcast_buffer``). Every rank can regenerate every rank's inputs from
the shared seed convention, which is what keeps each reference arithmetic rather
than a second collective. Producer pushes (``prepushed`` 2 / 3) are emulated by
storing the canonicalized rows through the multicast mapping at the call parity
the flags select, which is exactly what the producing kernels store.

Not a pytest module, despite the ``check_*`` bodies: they are one fixed sequence
inside a single 4-rank job whose workspace state (call parities) carries from
one to the next. The collected entry point is
``test_modeling_v2_m3_mc_allreduce_norm_op_matrix.py``.
"""

import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.distributed as dist

assert torch.cuda.is_available(), "m3_mc_allreduce_norm requires CUDA devices"

_WORKER_FLAG = "--rank-worker"
DEADLINE_S = 900

HIDDEN = 6144
EPS = 1e-6
CLUSTER = 8
BUCKETS = (8, 16, 48)
EMPTY_WORD = -2147483648

# Bound inside the rank body: importing the entries pulls in tensorrt_llm.
allreduce_entry: Any = None
finalize_entry: Any = None
ar_module: Any = None
validating: Any = None
RANK = 0
WORLD = 1
WORKSPACES: Dict[int, Any] = {}


# ---------------------------------------------------------------- inputs


def _gen(seed: int, rank: int) -> torch.Generator:
    return torch.Generator(device="cuda").manual_seed(seed * 1009 + rank + 1)


def _rows(seed: int, rank: int, m: int, scale: float = 1.0) -> torch.Tensor:
    """One rank's [m, H] BF16 rows; ``rank`` -1 for tensors every rank holds alike."""
    g = _gen(seed, rank + 1)
    return (torch.randn(m, HIDDEN, generator=g, device="cuda") * scale).to(torch.bfloat16)


def _residual(seed: int, m: int) -> torch.Tensor:
    return _rows(seed + 7, -1, m, 4.0)


def _norm_w(seed: int) -> torch.Tensor:
    g = _gen(seed + 11, 0)
    return (1.0 + 0.25 * torch.randn(HIDDEN, generator=g, device="cuda")).to(torch.bfloat16)


def _finalize_inputs(seed: int, rank: int, m: int, topk: int, dropped: int):
    """One rank's unfinalized MoE outputs: fc2 [m * topk, H], perm [m, topk], fp32 scales, shared [m, H]."""
    g = _gen(seed + 23, rank)
    rows = m * topk
    fc2 = (torch.randn(rows, HIDDEN, generator=g, device="cuda") * 0.5).to(torch.bfloat16)
    perm = torch.randperm(rows, generator=g, device="cuda").to(torch.int32)
    if dropped:
        drop = torch.randperm(rows, generator=g, device="cuda")[:dropped]
        perm[drop] = -1
    perm = perm.view(m, topk).contiguous()
    scales = torch.rand(m, topk, generator=g, device="cuda", dtype=torch.float32)
    shared = _rows(seed + 29, rank, m, 0.5)
    return fc2, perm, scales, shared


# ---------------------------------------------------------------- workspace / producer emulation


def _make_workspace(maxt: int):
    """One bucket's exchange, as ``McAllReduceWorkspace.create`` builds it, over the torch.distributed group."""
    from tensorrt_llm._torch.distributed.ops import _make_mnnvl_mcast_buffer
    from tensorrt_llm.mapping import Mapping

    mapping = Mapping(world_size=WORLD, tp_size=WORLD, rank=RANK)
    words = 4 * maxt * WORLD * (HIDDEN // 2)
    group = dist.group.WORLD
    handle = _make_mnnvl_mcast_buffer(group, words * 4, mapping, False)
    uc = handle.get_uc_buffer(RANK, (words,), torch.int32, 0)
    mc = handle.get_mc_buffer((words,), torch.int32, 0)
    uc.fill_(EMPTY_WORD)
    torch.cuda.synchronize()
    dist.barrier()
    ws = ar_module.McAllReduceWorkspace(
        uc=uc,
        mc=mc,
        flags={},
        rank=RANK,
        world_size=WORLD,
        hidden=HIDDEN,
        handle=handle,
        comm=group,
        max_tokens=maxt,
    )
    ws.flags_for(CLUSTER)
    return ws


def _push(ws, rows: torch.Tensor, exchange_set: int) -> None:
    """Store this rank's rows into every rank's exchange, as a producer kernel does before a prepushed call.

    Slot of (token t, cluster CTA c): set * 2 * half + parity(t, c) * half + (t * world + rank) * H / 2 + c * H / 16
    words, parity read from the CTA's flag; -0.0 halves are stored as +0.0 (0x80000000 is the empty word).
    """
    m = rows.shape[0]
    torch.cuda.synchronize()
    parity = (ws.flags_for(CLUSTER).view(ws.max_tokens, CLUSTER)[:m] & 1).long()
    half = ws.max_tokens * WORLD * (HIDDEN // 2)
    chunk = HIDDEN // 2 // CLUSTER
    h16 = rows.contiguous().view(torch.int16)
    h16 = torch.where(h16 == -32768, torch.zeros_like(h16), h16)
    words = h16.view(torch.int32).view(m, CLUSTER, chunk)
    t = torch.arange(m, device="cuda").view(m, 1, 1)
    c = torch.arange(CLUSTER, device="cuda").view(1, CLUSTER, 1)
    w = torch.arange(chunk, device="cuda").view(1, 1, chunk)
    index = (
        exchange_set * 2 * half
        + parity.view(m, CLUSTER, 1) * half
        + (t * WORLD + RANK) * (HIDDEN // 2)
        + c * chunk
        + w
    )
    ws.mc[index.reshape(-1)] = words.reshape(-1)
    torch.cuda.synchronize()
    dist.barrier()


# ---------------------------------------------------------------- gates


def _assert_bitwise(out: torch.Tensor, ref: torch.Tensor, where: str) -> None:
    assert out.dtype == ref.dtype and out.shape == ref.shape, (
        where,
        out.dtype,
        ref.dtype,
        out.shape,
    )
    a, b = out.contiguous().view(torch.uint8), ref.contiguous().view(torch.uint8)
    bad = (a != b).sum().item()
    assert bad == 0, f"{where}: {bad} of {a.numel()} bytes differ"


#: Share of normed elements one BF16 step off the float64 reference: measured at most 1.1e-4 per call over
#: every call of this matrix on GB300 (see ``_assert_normed``). The gate sits ~18x above that and far below what a
#: wrong statistic produces (most elements move).
NORMED_STEP_FRACTION = 0.002
NORMED_STEP_SEEN: List[float] = []


def _assert_normed(out: torch.Tensor, ref: torch.Tensor, where: str) -> None:
    """At most one BF16 rounding step from the float64 reference, and rarely that.

    The kernel's RMS statistic is a shuffle / warp / cluster tree of FP32 partial sums and its ``rsqrtf`` is
    approximate (~2^-22 relative): the scaled value moves by ~1e-6 relative, which changes the BF16 rounding only
    for values within that distance of a rounding midpoint -- one step, never two. The fraction bound catches a
    statistic that is systematically off (a missing rank, a wrong H), which moves most elements.
    """
    assert out.dtype == ref.dtype == torch.bfloat16 and out.shape == ref.shape, where
    o, r = out.float(), ref.float()
    _, exponent = torch.frexp(torch.maximum(o.abs(), r.abs()))
    step = torch.ldexp(torch.ones_like(o), exponent - 8)
    diff = (o - r).abs()
    assert bool((diff <= step).all()), f"{where}: max diff {diff.max().item()} beyond one BF16 step"
    frac = (diff != 0).float().mean().item()
    NORMED_STEP_SEEN.append(frac)
    assert frac <= NORMED_STEP_FRACTION, f"{where}: {frac:.4f} of normed elements off by one step"


def _assert_quant(outs, ref, m: int, where: str) -> None:
    """The MXFP8 copy: bit-exact against the recipe applied to the op's own normed rows (a pure function of
    them), and equal to the reference's wherever the reference's normed 32-column block is the op's."""
    from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_nvfp4__sm_103__tp4.ops.comm.m3_mc_allreduce_norm import (  # noqa: E501
        mxfp8_rows,
        sf_written,
    )

    q, sf = outs[2], outs[3]
    own_q, own_sf = mxfp8_rows(outs[0])
    written = sf_written(m, HIDDEN // 32, sf.device)
    _assert_bitwise(q, own_q, f"{where} e4m3 rows")
    _assert_bitwise(sf[written], own_sf[written], f"{where} UE8M0 scales")
    same_block = (outs[0].view(m, -1, 32) == ref[0].view(m, -1, 32)).all(-1)
    qb = q.view(torch.uint8).view(m, -1, 32)
    rqb = ref[2].view(torch.uint8).view(m, -1, 32)
    assert bool((qb == rqb).all(-1)[same_block].all()), (
        f"{where}: e4m3 differs on a block whose normed rows agree"
    )


def _check_outputs(outs, ref, m: int, quant: bool, where: str) -> None:
    assert len(outs) == (4 if quant else 2), (where, len(outs))
    _assert_bitwise(outs[1], ref[1], f"{where} updated residual")
    _assert_normed(outs[0], ref[0], f"{where} normed")
    if quant:
        _assert_quant(outs, ref, m, where)


# ---------------------------------------------------------------- calls


def _call_plain(ws, x, x2, residual, norm_w, quant, prepushed, opts):
    return allreduce_entry.m3_mc_allreduce_norm(
        x,
        x2,
        residual,
        norm_w,
        EPS,
        ws.uc,
        ws.mc.data_ptr(),
        ws.flags_for(CLUSTER),
        RANK,
        WORLD,
        CLUSTER,
        False,
        ws.max_tokens,
        quant,
        prepushed,
        opts,
    )


def _plain_case(spec: Dict[str, Any], seed: int, where: str) -> None:
    m, maxt, quant, prepushed = spec["m"], spec["maxt"], spec["quant"], spec["prepushed"]
    ws = WORKSPACES[maxt]
    xs = [_rows(seed, r, m) for r in range(WORLD)]
    x2s = [_rows(seed + 3, r, m) for r in range(WORLD)] if spec["x2"] else None
    residual, norm_w = _residual(seed, m), _norm_w(seed)
    x, x2 = xs[RANK], (None if x2s is None else x2s[RANK])
    if prepushed >= 2:
        _push(ws, x, 0)
    if prepushed == 3:
        _push(ws, x2, 1)
    with validating(allreduce_entry.m3_mc_allreduce_norm):
        outs = _call_plain(ws, x, x2, residual, norm_w, quant, prepushed, spec["opts"])
    torch.cuda.synchronize()
    ref = allreduce_entry.m3_mc_allreduce_norm.reference(xs, x2s, residual, norm_w, EPS, quant)
    _check_outputs(outs, ref, m, quant, where)
    dist.barrier()


# ---------------------------------------------------------------- checks


def check_plain_cells() -> None:
    """Every cell of ``m3_mc_allreduce_norm.CELLS`` through ``__call__`` under ``validating``, gated on its
    ``reference``."""
    for i, cell in enumerate(allreduce_entry.m3_mc_allreduce_norm.CELLS):
        _plain_case(cell.spec, 1000 + 17 * i, cell.why)


def check_finalize_cells() -> None:
    """Every cell of ``m3_mc_finalize_allreduce_norm.CELLS``, gated on its ``reference``."""
    op = finalize_entry.m3_mc_finalize_allreduce_norm
    for i, cell in enumerate(op.CELLS):
        spec = cell.spec
        m, maxt, quant, topk = spec["m"], spec["maxt"], spec["quant"], spec["topk"]
        seed = 2000 + 17 * i
        ws = WORKSPACES[maxt]
        per_rank = [_finalize_inputs(seed, r, m, topk, spec["dropped"]) for r in range(WORLD)]
        residual, norm_w = _residual(seed, m), _norm_w(seed)
        fc2, perm, scales, shared = per_rank[RANK]
        with validating(op):
            outs = op(
                fc2,
                perm,
                scales,
                shared,
                residual,
                norm_w,
                EPS,
                ws.uc,
                ws.mc.data_ptr(),
                ws.flags_for(CLUSTER),
                RANK,
                WORLD,
                CLUSTER,
                False,
                maxt,
                quant,
                spec["opts"],
            )
        torch.cuda.synchronize()
        ref = op.reference(
            [p[0] for p in per_rank],
            [p[1] for p in per_rank],
            [p[2] for p in per_rank],
            [p[3] for p in per_rank],
            residual,
            norm_w,
            EPS,
            quant,
        )
        _check_outputs(outs, ref, m, quant, cell.why)
        dist.barrier()


def check_the_target_wrappers_select_bucket_and_options() -> None:
    """The target's own wrappers (bucket by token count, option table) on top of the entries, every token count of
    the 8-token bucket plus one of each larger bucket, and the finalize wrapper; the MXFP8 publication is the
    prequant slot's business and is left off."""
    buckets = ar_module.McBuckets([WORKSPACES[b] for b in BUCKETS])
    for m in (*range(1, 9), 11, 40):
        seed = 3000 + m
        xs = [_rows(seed, r, m) for r in range(WORLD)]
        x2s = [_rows(seed + 3, r, m) for r in range(WORLD)]
        residual, norm_w = _residual(seed, m), _norm_w(seed)
        normed, updated = ar_module.m3_mc_allreduce_norm(
            xs[RANK], residual, norm_w, EPS, buckets, x2=x2s[RANK]
        )
        torch.cuda.synchronize()
        ref = allreduce_entry.m3_mc_allreduce_norm.reference(xs, x2s, residual, norm_w, EPS, False)
        _check_outputs([normed, updated], ref, m, False, f"wrapper m={m}")
        dist.barrier()
    for m in (9, 33):
        seed = 3100 + m
        per_rank = [_finalize_inputs(seed, r, m, 4, 2) for r in range(WORLD)]
        residual, norm_w = _residual(seed, m), _norm_w(seed)
        fc2, perm, scales, shared = per_rank[RANK]
        normed, updated = ar_module.m3_mc_finalize_allreduce_norm(
            fc2, perm, scales, shared, residual, norm_w, EPS, buckets
        )
        torch.cuda.synchronize()
        ref = finalize_entry.m3_mc_finalize_allreduce_norm.reference(
            *[[p[j] for p in per_rank] for j in range(4)], residual, norm_w, EPS, False
        )
        _check_outputs([normed, updated], ref, m, False, f"finalize wrapper m={m}")
        dist.barrier()


def check_cuda_graph_replays() -> None:
    """Two boundaries captured in one graph (8- and 16-token buckets), replayed with fresh inputs.

    The decode graph holds every boundary call; each replay rotates the same per-CTA parities the eager calls do,
    so a replay that reused a stale half or skipped a re-arm reads the previous replay's rows and fails the
    bitwise residual gate (every replay gets new inputs).
    """
    shapes = ((3, 8, 15), (12, 16, 15))
    norm_w = _norm_w(4000)
    bufs = []
    for m, maxt, opts in shapes:
        bufs.append(
            dict(
                m=m,
                maxt=maxt,
                opts=opts,
                x=torch.empty(m, HIDDEN, dtype=torch.bfloat16, device="cuda"),
                x2=torch.empty(m, HIDDEN, dtype=torch.bfloat16, device="cuda"),
                residual=torch.empty(m, HIDDEN, dtype=torch.bfloat16, device="cuda"),
            )
        )

    def body():
        return [
            _call_plain(
                WORKSPACES[b["maxt"]], b["x"], b["x2"], b["residual"], norm_w, False, 0, b["opts"]
            )
            for b in bufs
        ]

    def fill(seed: int):
        refs = []
        for j, b in enumerate(bufs):
            s = seed + 31 * j
            xs = [_rows(s, r, b["m"]) for r in range(WORLD)]
            x2s = [_rows(s + 3, r, b["m"]) for r in range(WORLD)]
            b["x"].copy_(xs[RANK])
            b["x2"].copy_(x2s[RANK])
            b["residual"].copy_(_residual(s, b["m"]))
            refs.append(
                allreduce_entry.m3_mc_allreduce_norm.reference(
                    xs, x2s, b["residual"].clone(), norm_w, EPS, False
                )
            )
        torch.cuda.synchronize()
        dist.barrier()
        return refs

    refs = fill(4100)
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        warm = body()
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    for j, b in enumerate(bufs):
        _check_outputs(warm[j], refs[j], b["m"], False, f"graph warm-up m={b['m']}")
    dist.barrier()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = body()
    for rep in range(4):
        refs = fill(4200 + 100 * rep)
        graph.replay()
        torch.cuda.synchronize()
        for j, b in enumerate(bufs):
            _check_outputs(captured[j], refs[j], b["m"], False, f"replay {rep} m={b['m']}")
        dist.barrier()
    del graph
    dist.barrier()


def check_the_gate_sees_a_missing_rank() -> None:
    """The residual gate is not blind to the reduction: dropping any one rank's partial from the reference
    moves most elements of the updated residual (each rank's rows are distinct by construction)."""
    m, seed = 4, 5000
    xs = [_rows(seed, r, m) for r in range(WORLD)]
    residual, norm_w = _residual(seed, m), _norm_w(seed)
    full = allreduce_entry.m3_mc_allreduce_norm.reference(xs, None, residual, norm_w, EPS, False)
    for r in range(WORLD):
        partial = [x if i != r else torch.zeros_like(x) for i, x in enumerate(xs)]
        wrong = allreduce_entry.m3_mc_allreduce_norm.reference(
            partial, None, residual, norm_w, EPS, False
        )
        frac = (wrong[1] != full[1]).float().mean().item()
        assert frac > 0.9, f"dropping rank {r} changes only {frac:.3f} of the residual"
    dist.barrier()


def _expect_rejected(fn, match: str, where: str) -> None:
    try:
        fn()
    except AssertionError as exc:
        assert match in str(exc), f"{where}: rejected for the wrong reason: {exc}"
    else:
        raise AssertionError(f"{where}: the guard admitted it")


def check_guards() -> None:
    """``is_valid`` refuses what the op accepts and answers wrongly. Nothing here reaches the kernel: the guard
    raises first, on every rank alike, so no rank is left polling."""
    m, seed = 4, 6000
    ws8, ws16 = WORKSPACES[8], WORKSPACES[16]
    x, residual, norm_w = _rows(seed, RANK, m), _residual(seed, m), _norm_w(seed)
    op = allreduce_entry.m3_mc_allreduce_norm

    def call(ws_uc, flags, norm):
        with validating(op):
            op(
                x,
                None,
                residual,
                norm,
                EPS,
                ws_uc,
                ws8.mc.data_ptr(),
                flags,
                RANK,
                WORLD,
                CLUSTER,
                False,
                8,
                False,
                0,
                15,
            )

    strided = torch.empty(2 * HIDDEN, dtype=torch.bfloat16, device="cuda")[::2]
    strided.copy_(norm_w)
    _expect_rejected(
        lambda: call(ws8.uc, ws8.flags_for(CLUSTER), strided), "contiguous", "strided norm_w"
    )
    _expect_rejected(
        lambda: call(ws8.uc[: ws8.uc.numel() // 2], ws8.flags_for(CLUSTER), norm_w),
        "words",
        "short workspace",
    )
    _expect_rejected(
        lambda: call(ws8.uc, ws16.flags_for(CLUSTER), norm_w), "counters", "another bucket's flags"
    )

    fin = finalize_entry.m3_mc_finalize_allreduce_norm
    fc2, perm, scales, shared = _finalize_inputs(seed, RANK, 9, 4, 0)
    bad = perm.clone()
    bad[0, 0] = fc2.shape[0]

    def call_fin():
        with validating(fin):
            fin(
                fc2,
                bad,
                scales,
                shared,
                _residual(seed, 9),
                norm_w,
                EPS,
                ws16.uc,
                ws16.mc.data_ptr(),
                ws16.flags_for(CLUSTER),
                RANK,
                WORLD,
                CLUSTER,
                False,
                16,
                False,
                15,
            )

    _expect_rejected(call_fin, "perm spans", "perm past the FC2 rows")
    dist.barrier()


CHECKS = (
    check_plain_cells,
    check_finalize_cells,
    check_the_target_wrappers_select_bucket_and_options,
    check_cuda_graph_replays,
    check_the_gate_sees_a_missing_rank,
    check_guards,
)


def _run_one_rank() -> int:
    """Body of one spawned rank: run every check; a failing rank exits nonzero and the launcher kills its
    siblings."""
    global RANK, WORLD, allreduce_entry, finalize_entry, ar_module, validating
    os.environ.setdefault("TLLM_DISABLE_MPI", "1")
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(rank)
    dist.init_process_group(
        "cpu:gloo,cuda:nccl", init_method="env://", device_id=torch.device("cuda", rank)
    )
    RANK, WORLD = rank, world
    assert WORLD == 4, f"the entries are certified at 4 ranks, got {WORLD}"

    import tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_nvfp4__sm_103__tp4.ops.m3_mc_allreduce as _ar  # noqa: E501
    from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_nvfp4__sm_103__tp4.ops.comm import (  # noqa: E501
        m3_mc_allreduce_norm as _plain,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_nvfp4__sm_103__tp4.ops.comm import (  # noqa: E501
        m3_mc_finalize_allreduce_norm as _finalize,
    )

    allreduce_entry, finalize_entry, ar_module = _plain, _finalize, _ar
    # Spawned by file path, not collected by pytest: reach the sibling helper the plain way (confined to the
    # rank processes). <target>/comm -> tests/unittest/_torch/modeling_v2.
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
    from _validating import validating as _validating_cm

    validating = _validating_cm
    for maxt in BUCKETS:
        WORKSPACES[maxt] = _make_workspace(maxt)

    for check in CHECKS:
        try:
            check()
        except BaseException:
            import traceback

            print(f"[rank {RANK}] FAILED {check.__name__}", flush=True)
            traceback.print_exc()
            sys.stdout.flush()
            sys.stderr.flush()
            # A rank that leaves early wedges its peers' polls; the launcher kills them on this exit.
            os._exit(13)
        print(f"[rank {RANK}] {check.__name__} passed", flush=True)
    dist.barrier()
    torch.cuda.synchronize()
    print(
        f"[rank {RANK}] {len(CHECKS)} checks passed; normed one-step fraction max "
        f"{max(NORMED_STEP_SEEN):.5f} over {len(NORMED_STEP_SEEN)} calls",
        flush=True,
    )
    sys.stdout.flush()
    os._exit(0)


def _spawn_ranks() -> None:
    """One worker process per claimed device, each in its own session; on the first nonzero exit or on the
    deadline every session is killed."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    assert visible, "set CUDA_VISIBLE_DEVICES to the 4 devices this run owns"
    world_size = len([d for d in visible.split(",") if d.strip()])
    assert world_size == 4, (
        f"CUDA_VISIBLE_DEVICES names {world_size} device(s); the matrix runs at 4"
    )
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    env = dict(
        os.environ,
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(port),
        WORLD_SIZE=str(world_size),
        TLLM_DISABLE_MPI="1",
    )
    # Drop a launching srun step's PMIx / MPI namespace: children registered in it get the whole step SIGKILLed
    # when the first one exits hard. These ranks rendezvous through torch.distributed's env://.
    env = {k: v for k, v in env.items() if not k.startswith(("PMIX_", "PMI_", "OMPI_", "SLURM_"))}
    print(f"[launcher] spawning {world_size} ranks, rendezvous on :{port}", flush=True)
    children = [
        subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()), _WORKER_FLAG],
            env=dict(env, RANK=str(r), LOCAL_RANK=str(r)),
            start_new_session=True,
        )
        for r in range(world_size)
    ]

    def _kill_all() -> None:
        for p in children:
            try:
                os.killpg(p.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        for p in children:
            p.wait()

    deadline = time.monotonic() + DEADLINE_S
    live = set(range(world_size))
    failure: Optional[str] = None
    while live and failure is None:
        if time.monotonic() > deadline:
            failure = f"did not finish in {DEADLINE_S}s (wedged)"
            break
        for r in sorted(live):
            code = children[r].poll()
            if code is None:
                continue
            live.discard(r)
            if code != 0:
                failure = f"rank {r} exited {code}"
                break
        time.sleep(0.5)
    if failure is not None:
        _kill_all()
        raise AssertionError(f"the {world_size}-rank run failed: {failure}")


if __name__ == "__main__":
    if _WORKER_FLAG in sys.argv:
        sys.exit(_run_one_rank())
    _spawn_ranks()
    print("OK")
