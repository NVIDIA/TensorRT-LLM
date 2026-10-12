# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU certification matrix for the DeepSeek-V4.1 target's pp_send_tensors / pp_recv_tensors.

Its own launcher. Run plainly, it re-executes itself under ``mpirun`` with one rank per
device of the first 4 of CUDA_VISIBLE_DEVICES, under a deadline: world 4, PP2 x TP2 on one
node. With ``--two-node`` it starts the same rank body at world 8 (PP2 x TP4) through the
two-node launcher named by MODELING_V2_TWO_NODE_MPIRUN: a command prefix that starts 8 ranks
with ranks 0-3 on one node and 4-7 on another, so stage 0 is one node, stage 1 the other,
and every hand-over crosses between them. Each form drives the entries' cells of its world
size (`pp_send_tensors.CELLS`, which `pp_recv_tensors.CELLS` must equal).

    CUDA_VISIBLE_DEVICES=0,1,2,3 python _pp_send_recv_op_matrix.py
    MODELING_V2_TWO_NODE_MPIRUN="<prefix>" python _pp_send_recv_op_matrix.py --two-node

Unlike the other comm matrices this keeps ``mpirun``: the hand-over it certifies runs on the
engine's MPI-path PP communicator (``PPCommNCCL``), whose eager send copies each tensor and
sends the copy on its own stream -- which is what lets the sender overwrite its tensors
right after the call (phases 1, 2, 4, 5) -- and the no-copy control uses that
communicator's raw send. With MPI disabled the engine builds ``PPCommTorch`` instead, which
sends the caller's tensors with no copy, so those checks would not be about it.

Stage-0 rank r sends to rank r + TP (``Mapping.next_pp_rank``); a sender and its receiver
are a *pair*, and pairs run independently. Every step hands over what the DeepSeek-V4.1
target hands over at its stage boundary, in its order: the hyper-connection stream
h bf16 [T, 4, 5120], the last FFN mixes yF fp32 [T, 24] and their RMS statistic rF fp32 [T].

Delivery is checked bitwise, per tensor, on the receiver, and the payload is built so that
a stale or misrouted delivery cannot pass: every step's words are a fixed random base
(the same on every rank) plus, as int32 words, a tag unique to (phase, step, sender), so the
bytes of any other step, phase or sender differ from the expected ones in every 32-bit
word; and every receive buffer is filled with 0xFF bytes first, so an unwritten word shows.
The op moves bytes and the dtypes only size the transfer, so tagged words that are not
finite bf16 / fp32 values are fine. The gate is the entries' own `compare` against their
`reference` (the sender's bytes).

Phases, on every pair (1-3 are the cells, run inside `validating` so the guards are shown
to admit every certified input):
1. eager, every T in 1..16384, the attention-TP layout: all pairs hand over the same T in
   a step, T in a shuffled order;
2. eager, every T in 1..16384, the attention-DP layout (``enable_attention_dp``: each rank
   holds its own rows): the pairs hand over different T in the same step;
   in 1 and 2 the sender overwrites and drops its tensors right after each send;
3. CUDA graphs at every row count the target captures: T = G and 6 G (DSpark verify) over
   the 46 graph batch sizes, 82 values in 1..1536, each after an eager hand-over at that T;
   one graph per T on both stages, each replayed 3 times on new tags (ascending, descending,
   and a per-pair shuffle with the sender overwriting its static tensors after each replay);
4. eager steps and graph replays interleaved at random T with the sender never waiting in
   between (the engine's mix of eager context steps and captured decode steps);
5. a late receiver: its stream is held ~20 ms before the receive while the sender
   overwrites, drops and reallocates its tensors (eager), or overwrites its static tensors
   (graph);
6. controls: a send that does not copy (the communicator's raw send, held back, the tensors
   overwritten meanwhile) must be seen as wrong; so must a missed receive, the previous
   step's bytes and the other sender's bytes, by the entries' gates too; the guards
   (`is_valid`, under `validating`) must refuse a strided view (what
   the raw op does with one, eagerly and under capture, is recorded);
7. timing of one hand-over at T = 1, 6, 1536, 8192, 16384.
"""

import os
import random
import shlex
import shutil
import signal
import socket
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import torch

assert torch.cuda.is_available(), "pp_send_tensors requires CUDA devices"

_WORKER_FLAG = "--rank-worker"
_TWO_NODE_FLAG = "--two-node"
TWO_NODE_ENV = "MODELING_V2_TWO_NODE_MPIRUN"
_PID_DIR_ENV = "PP_SEND_RECV_MATRIX_PID_DIR"
DEADLINE_S = 1200
LOCAL_WORLD = 4
HIDDEN, HC, MIX = 5120, 4, 24
NAMES = ("h", "yF", "rF")
DTYPES = (torch.bfloat16, torch.float32, torch.float32)
# max_num_tokens up to 16384 per rank.
N_MAX = 16384
# The 46 CUDA-graph batch sizes: every G in 1..32 plus the padded default list up to 256.
GRAPH_BATCHES = list(range(1, 33)) + list(range(40, 129, 8)) + [192, 256]
# A captured step hands over T = G rows (DSpark off) or 6 G (DSpark on: 1 + 5 drafts).
GRAPH_T = sorted(set(GRAPH_BATCHES) | {6 * g for g in GRAPH_BATCHES})
LATE_T = (1, 6, 1536, 8192, 16384)
TIMING_T = (1, 6, 1536, 8192, 16384)
INTERLEAVED_STEPS = 400
SENTINEL = -1  # 0xFFFFFFFF: what a receive buffer holds before the receive
JUNK = 0x5A5A5A5A  # what a sender writes over its tensors after handing them over
LATE_CYCLES = 40_000_000  # about 20 ms of GPU clock
# Tag phases.
PH_SWEEP_TP, PH_SWEEP_DP, PH_CAPTURE, PH_REPLAY, PH_MIX, PH_LATE, PH_CONTROL, PH_TIMING = (
    1,
    2,
    3,
    4,
    8,
    9,
    10,
    11,
)

assert len(GRAPH_BATCHES) == 46 and len(GRAPH_T) == 82 and GRAPH_T[-1] == 1536


def _shapes(T):
    return ((T, HC, HIDDEN), (T, MIX), (T,))


def _tag(phase: int, step: int, sender: int) -> int:
    assert 0 < phase < 256 and 0 <= step < (1 << 18) and 0 <= sender < 15
    return (phase << 22) | (step << 4) | (sender + 1)


def _untag(value: int) -> str:
    value &= 0xFFFFFFFF
    return f"phase {value >> 22} step {(value >> 4) & 0x3FFFF} sender {(value & 15) - 1}"


def _diagnose(got, base, tag: int) -> str:
    """Say what a wrong delivery holds: unwritten words, or another step's / sender's words."""
    delta = got.reshape(-1) - base.reshape(-1)
    wrong = delta != tag
    n = int(wrong.sum())
    first = int(wrong.nonzero()[0, 0])
    unfilled = int((got.reshape(-1)[wrong] == SENTINEL).sum())
    values, counts = torch.unique(delta[wrong], return_counts=True)
    top = sorted(zip(counts.tolist(), values.tolist()), reverse=True)[:3]
    carried = "; ".join(f"{c} words carry +{v & 0xFFFFFFFF:#x} ({_untag(v)})" for c, v in top)
    return (
        f"{n}/{delta.numel()} words wrong, first at word {first}, {unfilled} still 0xFF; {carried}"
    )


def _rank_body() -> int:
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    # This node's ranks do not live in mpirun's process group: leave the pid where the
    # launcher's deadline can find it.
    pid_dir = os.environ.get(_PID_DIR_ENV)
    if pid_dir:
        Path(pid_dir, f"rank{rank}.pid").write_text(str(os.getpid()))
    try:
        return _run(comm)
    except BaseException:
        import traceback

        print(f"[rank {rank}] FAILED with an exception", flush=True)
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        # Abort rather than return: a rank that leaves the hand-over wedges its peer.
        comm.Abort(1)
        return 1


def _run(comm) -> int:
    from tensorrt_llm._torch._experimental.modeling_v2.models.deepseek_v41.deepseek_v4_1_flash_nvfp4__sm_103__unified.ops.comm import (  # noqa: E501
        pp_recv_tensors as recv_entry,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.models.deepseek_v41.deepseek_v4_1_flash_nvfp4__sm_103__unified.ops.comm import (  # noqa: E501
        pp_send_tensors as send_entry,
    )
    from tensorrt_llm._torch.distributed import communicator
    from tensorrt_llm.mapping import Mapping

    # This file runs as an mpirun rank, not under pytest, so the repo's
    # `__extra_import_path__` mechanism is not in play and the shared helper is
    # reached the plain way. The mutation is confined to these rank processes.
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
    from _validating import validating

    # The entries' instances: every call below goes through them.
    pp_send_tensors, pp_recv_tensors = send_entry.pp_send_tensors, recv_entry.pp_recv_tensors
    rank, world = comm.Get_rank(), comm.Get_size()
    assert world in (4, 8), f"PP2 x TP2 (4 ranks) or PP2 x TP4 (8 ranks), got {world}"
    tp = world // 2

    # The cells of this world size, which phases 1-3 drive: one matrix certifies both
    # halves, so the two entries must name the same cells.
    assert [c.spec for c in pp_recv_tensors.CELLS] == [c.spec for c in pp_send_tensors.CELLS], (
        "pp_send_tensors and pp_recv_tensors name different cells"
    )
    cells = {
        (c.spec["attention_dp"], c.spec["mode"]): c
        for c in pp_send_tensors.CELLS
        if c.spec["world_size"] == world
    }
    assert set(cells) == {(False, "eager"), (True, "eager"), (False, "graph")}, list(cells)
    handover = [(n, s[1:], d) for n, s, d in zip(NAMES, _shapes(1), DTYPES)]
    for c in cells.values():
        assert c.spec["tp_size"] == tp and c.spec["pp_size"] == 2, c.why
        assert [(n, tuple(t), d) for n, t, d in c.spec["tensors"]] == handover, c.why
    for key in ((False, "eager"), (True, "eager")):
        assert list(cells[key].spec["rows"]) == list(range(1, N_MAX + 1)), cells[key].why
    assert list(cells[(False, "graph")].spec["rows"]) == GRAPH_T, cells[(False, "graph")].why
    local = int(os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK", rank))
    per_node = int(os.environ.get("OMPI_COMM_WORLD_LOCAL_SIZE", world))
    torch.cuda.set_device(local)
    dev = torch.device("cuda", local)
    kw = dict(world_size=world, rank=rank, tp_size=tp, pp_size=2, gpus_per_node=per_node)
    layouts = {
        "attention-TP": Mapping(**kw),
        "attention-DP": Mapping(**kw, moe_ep_size=tp, enable_attention_dp=True),
    }
    m = layouts["attention-TP"]
    first = m.is_first_pp_rank()
    pair = m.tp_rank
    sender = rank if first else m.prev_pp_rank()
    receiver = m.next_pp_rank() if first else rank
    assert receiver == sender + tp and pair == sender, (rank, sender, receiver, pair)
    for mm in layouts.values():
        assert (mm.next_pp_rank() if first else mm.prev_pp_rank()) == (
            receiver if first else sender
        )
    communicator.init_pp_comm(layouts["attention-TP"])

    major, minor = torch.cuda.get_device_capability()
    who = comm.allgather(
        (
            socket.gethostname(),
            local,
            torch.cuda.get_device_name(),
            f"sm_{major}{minor}",
            send_entry.__file__,
            recv_entry.__file__,
        )
    )
    hosts = [w[0] for w in who]
    crosses = [hosts[s] != hosts[s + tp] for s in range(tp)]
    # World 8 certifies the two-tray form: every pair must cross between the nodes.
    assert all(crosses) if world == 8 else not any(crosses), (world, hosts)
    if rank == 0:
        print(f"[rank 0] world_size={world} PP2 x TP{tp}; pairs cross nodes: {crosses}", flush=True)
        for r, (host, dvc, name, sm, fs, fr) in enumerate(who):
            role = f"stage 0, sends to {r + tp}" if r < tp else f"stage 1, receives from {r - tp}"
            print(
                f"[rank 0]   rank {r} host {host} device {dvc} {name} {sm} ({role}) entry={fs} {fr}",
                flush=True,
            )

    g = torch.Generator(device=dev).manual_seed(0x5EED)
    base = [
        torch.randn((N_MAX, HC, HIDDEN), generator=g, device=dev)
        .to(torch.bfloat16)
        .view(torch.int32),
        torch.randn((N_MAX, MIX), generator=g, device=dev).view(torch.int32),
        (torch.rand((N_MAX,), generator=g, device=dev) + 0.5).view(torch.int32),
    ]
    digest = [int(b.to(torch.int64).sum()) for b in base]
    assert all(d == digest for d in comm.allgather(digest)), (
        "the base payload differs between ranks"
    )

    failures = []
    checks = 0

    def payload(T, tag):
        return [torch.add(b[:T], tag).view(d) for b, d in zip(base, DTYPES)]

    def fresh(T):
        bufs = [torch.empty(s, dtype=d, device=dev) for s, d in zip(_shapes(T), DTYPES)]
        for b in bufs:
            b.view(torch.int32).fill_(SENTINEL)
        return bufs

    def matches(bufs, T, tag):
        return [torch.equal(x.view(torch.int32), torch.add(b[:T], tag)) for x, b in zip(bufs, base)]

    def verify(bufs, T, tag, what):
        """The entries' own gate against their own reference (the sender's bytes, which
        every rank rebuilds from the shared base and the tag); on a failure, say what the
        wrong delivery holds instead."""
        nonlocal checks
        checks += 1
        sent = payload(T, tag)
        try:
            pp_recv_tensors.compare(bufs, pp_recv_tensors.reference(sent))
            pp_send_tensors.compare(bufs, pp_send_tensors.reference(sent))
        except AssertionError as exc:
            wrong = [k for k, ok in enumerate(matches(bufs, T, tag)) if not ok]
            failures.extend(
                f"[rank {rank}] {what} T={T} {NAMES[k]}: "
                f"{_diagnose(bufs[k].view(torch.int32), base[k][:T], tag)}"
                for k in wrong
            )
            if not wrong:
                failures.append(f"[rank {rank}] {what} T={T}: {exc}")

    def junk(bufs):
        for b in bufs:
            b.view(torch.int32).fill_(JUNK)

    def gate_rejects_each(bufs, T, tag):
        """Whether both entries' gates fail each tensor of `bufs` on its own."""
        rejected = []
        for x, want in zip(bufs, payload(T, tag)):
            refused = 0
            for op in (pp_recv_tensors, pp_send_tensors):
                try:
                    op.compare([x], op.reference([want]))
                except AssertionError:
                    refused += 1
            rejected.append(refused == 2)
        return rejected

    def eager(T, phase, step, *, late=False, sync=True, what="eager"):
        tag = _tag(phase, step, sender)
        if first:
            bufs = payload(T, tag)
            pp_send_tensors(bufs)
            # The eager send works on a copy: overwrite and drop the tensors at once.
            junk(bufs)
            del bufs
            if late:
                # Reallocate the freed sizes and fill them with junk: a copy that the
                # caching allocator handed out again before the send read it would show.
                again = [fresh(T) for _ in range(2)]
                for bufs in again:
                    junk(bufs)
                del again
            if sync:
                torch.cuda.synchronize()
        else:
            bufs = fresh(T)
            if late:
                torch.cuda._sleep(LATE_CYCLES)
            got = pp_recv_tensors(bufs)
            assert got is bufs, "pp_recv_tensors must return the list it filled"
            verify(bufs, T, tag, f"{what} step {step}")

    def sweep(layout, phase, order):
        assert sorted(order) == list(range(1, N_MAX + 1))
        communicator.init_pp_comm(layouts[layout])
        comm.Barrier()
        t0 = time.perf_counter()
        for step, T in enumerate(order):
            eager(T, phase, step, what=f"{layout} eager sweep")
        torch.cuda.synchronize()
        comm.Barrier()
        return time.perf_counter() - t0

    # Phases 1-3 are the cells: the guards run on every call, which shows they admit
    # every certified input.
    with validating(pp_send_tensors, pp_recv_tensors):
        # 1. eager, every T, the attention-TP layout: the same shuffled T on every pair.
        shared = list(cells[(False, "eager")].spec["rows"])
        random.Random(1).shuffle(shared)
        secs_tp = sweep("attention-TP", PH_SWEEP_TP, shared)
        # 2. eager, every T, the attention-DP layout: a different T on every pair in a step.
        per_pair = [
            list(range(1, N_MAX + 1)),
            list(range(N_MAX, 0, -1)),
            [(s * 7919) % N_MAX + 1 for s in range(N_MAX)],
            [(s * 4099 + 5000) % N_MAX + 1 for s in range(N_MAX)],
        ]
        secs_dp = sweep("attention-DP", PH_SWEEP_DP, per_pair[pair])
        communicator.init_pp_comm(layouts["attention-TP"])
        checks_eager = checks

        # 3. CUDA graphs: an eager hand-over at T, then one graph per T on both stages.
        graphs = {}
        capture = torch.cuda.Stream()
        for i, T in enumerate(GRAPH_T):
            eager(T, PH_CAPTURE, i, what="eager before capture")
            static = [torch.zeros(s, dtype=d, device=dev) for s, d in zip(_shapes(T), DTYPES)]
            comm.Barrier()
            capture.wait_stream(torch.cuda.current_stream())
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=capture):
                if first:
                    pp_send_tensors(static)
                else:
                    pp_recv_tensors(static)
            torch.cuda.current_stream().wait_stream(capture)
            torch.cuda.synchronize()
            graphs[T] = (graph, static)
        comm.Barrier()

        def replay(T, phase, step, *, late=False, clobber=False, what="replay"):
            graph, static = graphs[T]
            tag = _tag(phase, step, sender)
            if first:
                for s, b in zip(static, base):
                    torch.add(b[:T], tag, out=s.view(torch.int32))
                graph.replay()
                if clobber or late:
                    # Stream-ordered after the captured send: the replay must have sent the
                    # bytes the static tensors held when it ran.
                    junk(static)
            else:
                for s in static:
                    s.view(torch.int32).fill_(SENTINEL)
                if late:
                    torch.cuda._sleep(LATE_CYCLES)
                graph.replay()
                verify(static, T, tag, f"{what} step {step}")

        shuffled = list(GRAPH_T)
        random.Random(100 + pair).shuffle(shuffled)
        rounds = (
            ("ascending", GRAPH_T, False),
            ("descending", GRAPH_T[::-1], False),
            ("per-pair shuffle, sender overwrites", shuffled, True),
        )
        for r, (name, order, clobber) in enumerate(rounds):
            for step, T in enumerate(order):
                replay(T, PH_REPLAY + r, step, clobber=clobber, what=f"graph {name}")
            torch.cuda.synchronize()
            comm.Barrier()
        checks_graph = checks - checks_eager

    # 4. eager steps and replays interleaved, the sender not waiting in between.
    rng = random.Random(900 + pair)
    mix = {"eager": 0, "replay": 0}
    for step in range(INTERLEAVED_STEPS):
        if rng.random() < 0.5:
            replay(rng.choice(GRAPH_T), PH_MIX, step, what="interleaved replay")
            mix["replay"] += 1
        else:
            T = rng.choice(GRAPH_T) if rng.random() < 0.5 else int(round(2 ** rng.uniform(0, 14)))
            eager(T, PH_MIX, step, sync=False, what="interleaved eager")
            mix["eager"] += 1
        if first and step % 8 == 7:
            torch.cuda.synchronize()  # bound the copies in flight
    torch.cuda.synchronize()
    comm.Barrier()

    # 5. a late receiver.
    for i, T in enumerate(LATE_T):
        eager(T, PH_LATE, i, late=True, what="late receiver, eager")
    for i, T in enumerate(t for t in LATE_T if t in graphs):
        replay(T, PH_LATE, 100 + i, late=True, what="late receiver, replay")
        torch.cuda.synchronize()
    comm.Barrier()

    # 6. controls: the checks see what they guard against.
    control = []
    side = torch.cuda.Stream()
    for i, T in enumerate((1536, 16384)):
        tag = _tag(PH_CONTROL, i, sender)
        if first:
            # The communicator's own send without the eager path's copy, held back ~20 ms
            # on a side stream while the current stream overwrites the tensors.
            bufs = payload(T, tag)
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                torch.cuda._sleep(LATE_CYCLES)
                for t in bufs:
                    communicator._pp_comm.nccl_comm.send(t, receiver)
            junk(bufs)
            torch.cuda.synchronize()
            del bufs
        else:
            bufs = fresh(T)
            pp_recv_tensors(bufs)
            ok = matches(bufs, T, tag)
            overwritten = [bool((x.view(torch.int32) == JUNK).all()) for x in bufs]
            rejected = gate_rejects_each(bufs, T, tag)
            control.append(
                f"send without a copy, T={T}: delivered == payload {ok}, == overwrite {overwritten}, "
                f"gate rejects {rejected}"
            )
            if any(ok) or not all(rejected):
                failures.append(
                    f"[rank {rank}] control T={T}: an overwritten send passed the check"
                )
    if not first:
        T = 257
        now = _tag(PH_CONTROL, 10, sender)
        wrong = {
            "missed receive": fresh(T),
            "previous step's bytes": payload(T, _tag(PH_CONTROL, 9, sender)),
            "the other sender's bytes": payload(T, _tag(PH_CONTROL, 10, (sender + 1) % tp)),
        }
        for name, bufs in wrong.items():
            ok = matches(bufs, T, now)
            differ = [
                bool((x.view(torch.int32) != torch.add(b[:T], now)).all())
                for x, b in zip(bufs, base)
            ]
            rejected = gate_rejects_each(bufs, T, now)
            control.append(
                f"{name}: passes {ok}, differs in every word {differ}, gate rejects {rejected}"
            )
            if any(ok) or not all(differ) or not all(rejected):
                failures.append(f"[rank {rank}] control {name}: not seen in every word")
    # A strided view (every other hc slot, [T, 2, 5120] of [T, 4, 5120]): the guards must
    # refuse it. What the unguarded op (`raw_call`, outside `validating`) does with one is
    # recorded, eagerly and under capture.
    T = 64
    for j, captured in enumerate((False, True)):
        tag = _tag(PH_CONTROL, 20 + j, sender)
        how = "captured" if captured else "eager"
        if first:
            view = payload(T, tag)[0][:, ::2, :]
            if not captured:
                try:
                    with validating(pp_send_tensors):
                        pp_send_tensors([view])
                    failures.append(f"[rank {rank}] the send guard accepted a strided view")
                except AssertionError as e:
                    control.append(f"send guard, strided view: refused, AssertionError({e})")
                pp_send_tensors.raw_call([view])
            else:
                graph = torch.cuda.CUDAGraph()
                capture.wait_stream(torch.cuda.current_stream())
                with torch.cuda.graph(graph, stream=capture):
                    pp_send_tensors.raw_call([view])
                torch.cuda.current_stream().wait_stream(capture)
                comm.Barrier()
                graph.replay()
            torch.cuda.synchronize()
        else:
            got = torch.empty((T, HC // 2, HIDDEN), dtype=torch.bfloat16, device=dev)
            got.view(torch.int32).fill_(SENTINEL)
            if not captured:
                pp_recv_tensors([got])
            else:
                graph = torch.cuda.CUDAGraph()
                capture.wait_stream(torch.cuda.current_stream())
                with torch.cuda.graph(graph, stream=capture):
                    pp_recv_tensors([got])
                torch.cuda.current_stream().wait_stream(capture)
                comm.Barrier()
                graph.replay()
            torch.cuda.synchronize()
            whole = torch.add(base[0][:T], tag).view(torch.bfloat16)
            as_view = torch.equal(got, whole[:, ::2, :].contiguous())
            as_storage = torch.equal(got.reshape(-1), whole.reshape(-1)[: got.numel()])
            control.append(
                f"raw op send of a strided view, {how}: delivered the view's elements {as_view}, "
                f"the first {got.numel() * 2} bytes of its storage {as_storage}"
            )
    tag = _tag(PH_CONTROL, 22, sender)
    if first:
        pp_send_tensors([payload(T, tag)[0][:, : HC // 2, :].contiguous()])
        torch.cuda.synchronize()
    else:
        big = torch.empty((T, HC, HIDDEN), dtype=torch.bfloat16, device=dev)
        big.view(torch.int32).fill_(SENTINEL)
        try:
            with validating(pp_recv_tensors):
                pp_recv_tensors([big[:, ::2, :]])
            failures.append(f"[rank {rank}] the recv guard accepted a strided view")
        except AssertionError as e:
            control.append(f"recv guard, strided view: refused, AssertionError({e})")
        pp_recv_tensors.raw_call([big[:, ::2, :]])
        torch.cuda.synchronize()
        want = torch.add(base[0][:T], tag).view(torch.bfloat16)[:, : HC // 2, :].contiguous()
        in_view = torch.equal(big[:, ::2, :], want)
        in_storage = torch.equal(big.reshape(-1)[: want.numel()], want.reshape(-1))
        control.append(
            f"raw op receive into a strided view, eager: the view holds the payload {in_view}, "
            f"the first {want.numel() * 2} bytes of its storage hold it {in_storage}"
        )
    comm.Barrier()

    # 7. timing of one hand-over (host wall time from a barrier to the data in place).
    timing = {}
    for i, T in enumerate(TIMING_T):
        tag = _tag(PH_TIMING, i, sender)
        bufs = payload(T, tag) if first else fresh(T)
        samples = []
        for _ in range(10):
            torch.cuda.synchronize()
            comm.Barrier()
            t0 = time.perf_counter()
            if first:
                pp_send_tensors(bufs)
            else:
                pp_recv_tensors(bufs)
            torch.cuda.synchronize()
            samples.append(time.perf_counter() - t0)
        if not first:
            verify(bufs, T, tag, "timing")
        timing[T] = statistics.median(samples) * 1e6
    comm.Barrier()

    total_failures = comm.allreduce(len(failures))
    total_checks = comm.allreduce(checks)
    report = comm.gather(
        (
            rank,
            first,
            failures[:20],
            control,
            timing,
            checks,
            checks_eager,
            checks_graph,
            mix,
            secs_tp,
            secs_dp,
        ),
        root=0,
    )
    if rank == 0:
        for r, is_first, fails, ctrl, tim, n, ne, ng, mx, st, sd in report:
            role = "sender" if is_first else "receiver"
            print(
                f"[rank {r}] {role}: timing us {', '.join(f'T={t}: {v:.0f}' for t, v in tim.items())}; "
                f"sweeps {st:.1f} s / {sd:.1f} s",
                flush=True,
            )
            if not is_first:
                print(
                    f"[rank {r}] checks {n} (eager sweeps {ne}; capture phase {ng}: 82 eager hand-overs + 246 "
                    f"replays); interleaved {mx}",
                    flush=True,
                )
            for line in ctrl:
                print(f"[rank {r}] control: {line}", flush=True)
            for line in fails:
                print(f"FAILURE {line}", flush=True)
        status = "PP_SEND_RECV_OK" if total_failures == 0 else "PP_SEND_RECV_FAILED"
        print(
            f"{status} world_size={world} checks={total_checks} failures={total_failures} "
            f"eager_T=1..{N_MAX} x2 layouts graph_T={len(GRAPH_T)} values 1..{GRAPH_T[-1]} "
            f"x3 replays interleaved={INTERLEAVED_STEPS} late={list(LATE_T)}",
            flush=True,
        )
    return 0 if total_failures == 0 else 1


def _scrubbed_env(**extra):
    """The environment without the allocation's Slurm/PMI context (a rank attached to it
    that dies abnormally takes the allocation down)."""
    env = dict(os.environ, **extra)
    for v in list(env):
        if v.startswith(
            ("PMI", "PMIX", "OMPI", "SLURM", "PRTE", "PALS", "MPI_LOCALRANKID", "MPI_LOCALNRANKS")
        ):
            env.pop(v)
    return env


def _wait(cmd, env, pid_dir) -> int:
    print(f"[launcher] {' '.join(cmd)}", flush=True)
    # Own process group, so the deadline reaches every process it started.
    proc = subprocess.Popen(cmd, env=env, start_new_session=True)
    try:
        return proc.wait(timeout=DEADLINE_S)
    except subprocess.TimeoutExpired:
        # TERM first: a launcher script gets to run its cleanup.
        os.killpg(proc.pid, signal.SIGTERM)
        try:
            proc.wait(timeout=90)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
        if pid_dir:
            for pid_file in Path(pid_dir).glob("rank*.pid"):
                try:
                    os.kill(int(pid_file.read_text()), signal.SIGKILL)
                except (ProcessLookupError, ValueError):
                    pass
        print(f"pp_send_recv matrix wedged: stopped after {DEADLINE_S} s", flush=True)
        return 124
    finally:
        if pid_dir:
            shutil.rmtree(pid_dir, ignore_errors=True)


def _launch_one_node() -> int:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    devices = [d for d in visible.split(",") if d.strip()] or [
        str(i) for i in range(torch.cuda.device_count())
    ]
    assert len(devices) >= LOCAL_WORLD, f"needs {LOCAL_WORLD} devices, sees {devices}"
    pid_dir = tempfile.mkdtemp(prefix="pp-send-recv-matrix-")
    env = _scrubbed_env(
        CUDA_VISIBLE_DEVICES=",".join(devices[:LOCAL_WORLD]), **{_PID_DIR_ENV: pid_dir}
    )
    cmd = [
        "mpirun",
        "-np",
        str(LOCAL_WORLD),
        "--oversubscribe",
        "--allow-run-as-root",
        "-x",
        _PID_DIR_ENV,
        sys.executable,
        str(Path(__file__).resolve()),
        _WORKER_FLAG,
    ]
    return _wait(cmd, env, pid_dir)


def _launch_two_node() -> int:
    prefix = os.environ.get(TWO_NODE_ENV, "").strip()
    assert prefix, (
        f"{_TWO_NODE_FLAG} needs {TWO_NODE_ENV}: a command prefix that starts 8 ranks, "
        f"ranks 0-3 on one node and 4-7 on another"
    )
    cmd = shlex.split(prefix) + [sys.executable, str(Path(__file__).resolve()), _WORKER_FLAG]
    # The other node's ranks cannot be signalled from here; the prefix owns them.
    return _wait(cmd, _scrubbed_env(), None)


if __name__ == "__main__":
    if _WORKER_FLAG in sys.argv:
        sys.exit(_rank_body())
    sys.exit(_launch_two_node() if _TWO_NODE_FLAG in sys.argv else _launch_one_node())
