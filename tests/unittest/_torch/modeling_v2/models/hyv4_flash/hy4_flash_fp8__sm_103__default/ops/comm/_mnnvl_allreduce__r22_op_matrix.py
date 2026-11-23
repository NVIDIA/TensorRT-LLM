# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Certify the Hy4-Flash target's `comm/mnnvl_allreduce__r22` over a TP group
that spans two GB300 trays.

Run once per node of a two-node spool job (``spoolctl.py submit --nodes 2
--gpus all``), never on one node alone: the cells are world 8 and the group
must span the trays, which one node's four devices cannot. Each node's copy
spawns its four local ranks (``RANK = node_rank * 4 + local``) with a
torch.distributed ``env://`` rendezvous on ``MV2FLOW_JOB_MASTER`` and MPI
disabled, so the entry builds its workspace on the mapping's TP process group
-- the path a Ray-orchestrated engine takes; under MPI only the communicator
that carries the setup handshake differs.

Not a pytest module, for the same reason as its siblings in ``comm/``: the
``check_*`` bodies are one ordered sequence inside one job and read rank state
that only ``_run_one_rank`` binds. Rank 0 prints each cell's measured error as
``[measure]`` lines.
"""

import functools
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

GPUS_PER_NODE = 4
"""A GB300 tray. The mapping needs it to see the group span two nodes."""

NODES = 2
"""Trays the certified group spans: world 8 is two of them here."""

MAX_ROWS = 8192
"""The row budget the workspace is built for: the target's max_num_tokens."""

GRAPH_BATCH_SIZES = (1, 8, 13, 64)
"""Decode batch sizes a served engine captures, sampled: the smallest, a power
of two, a size that is not one, and the target's max_batch_size."""

DEADLINE_S = 1200
"""The launcher's bound: a broken collective hangs rather than raising."""

_MODELING_V2_TESTS = Path(__file__).resolve().parents[5]
"""tests/unittest/_torch/modeling_v2: `_validating` and the shared `comm/` helpers."""

_WORKER_FLAG = "--rank-worker"

sys.path.insert(0, str(_MODELING_V2_TESTS / "comm"))
import _allreduce_op_matrix  # noqa: E402
from _allreduce_op_matrix import _payload, _residual_and_weight  # noqa: E402

# Bound inside the rank body. `mnnvl_allreduce` is the call surface, a partial
# carrying the workspace arguments; `entry` is the module, whose
# `mnnvl_allreduce` instance holds `reference`, `is_valid` and `CELLS`.
mnnvl_allreduce: Any = None
entry: Any = None
validating: Any = None
WORKSPACE_ARGS: dict = {}
RANK = 0
WORLD = 1


def _compare(out, ref) -> None:
    """`comm/allreduce`'s band: 5 ulp per element against the row scale,
    1.25 ulp relative RMS, on both outputs.

    The shared `comm/mnnvl_allreduce` matrix measured the same kernel at world
    8 over two GB300 trays with the gpt-oss and deepseek widths: at most 2.98
    ulp per element and 0.88 ulp RMS on `norm_out`, 2.00 and 0.72 on
    `residual_out`, flat in the row count. A dropped rank moves the result by
    tens of ulp; `check_the_gate_discriminates_a_wrong_reduction` checks the
    band refuses one.
    """
    _allreduce_op_matrix._compare(out, ref)


def _call(x, residual, weight, eps):
    return mnnvl_allreduce(x, residual, norm_weight=weight, eps=eps)


def _reference(rows, hidden, seed, residual, weight, eps):
    inputs = [_payload(r, rows, hidden, seed) for r in range(WORLD)]
    return entry.mnnvl_allreduce.reference(inputs, residual, weight, eps)


def _ulp_stats(out, ref):
    """Worst element error against the row scale, and relative RMS, in bf16 ulp."""
    ulp = 2.0**-8
    o, r = out.float().reshape(-1, out.shape[-1]), ref.float().reshape(-1, ref.shape[-1])
    row = r.abs().amax(dim=1, keepdim=True).clamp_min(1e-9)
    elem = ((o - r).abs() / row).max().item() / ulp
    rms = ((o - r).pow(2).mean().sqrt() / r.pow(2).mean().sqrt().clamp_min(1e-9)).item() / ulp
    return elem, rms


def check_certified_cells() -> None:
    """Every world-matching cell, inside `validating`, against the reference
    and band. Rank 0 also reports the error it measured over several seeds."""
    cells = [c for c in entry.mnnvl_allreduce.CELLS if c.spec["world"] == WORLD]
    assert cells, f"no cell is certified at world size {WORLD}"
    for i, cell in enumerate(cells):
        spec = cell.spec
        worst = [0.0, 0.0, 0.0, 0.0]
        for s in range(5):
            seed = 7000 + 17 * i + 1000 * s
            x = _payload(RANK, spec["rows"], spec["hidden"], seed)
            residual, weight = _residual_and_weight(spec["rows"], spec["hidden"], seed)
            with validating(entry.mnnvl_allreduce):
                out = _call(x, residual, weight, spec["eps"])
            ref = _reference(spec["rows"], spec["hidden"], seed, residual, weight, spec["eps"])
            for k, (o, r) in enumerate(zip(out, ref)):
                e, rms = _ulp_stats(o, r)
                worst[2 * k] = max(worst[2 * k], e)
                worst[2 * k + 1] = max(worst[2 * k + 1], rms)
            try:
                _compare(out, ref)
            except AssertionError as exc:
                raise AssertionError(f"{cell.why}: {exc}") from exc
            dist.barrier()
        if RANK == 0:
            print(
                f"[measure] world={WORLD} hidden={spec['hidden']} rows={spec['rows']} "
                f"norm_out elem={worst[0]:.2f} rms={worst[1]:.3f} "
                f"residual_out elem={worst[2]:.2f} rms={worst[3]:.3f} (ulp, 5 seeds)",
                flush=True,
            )


def check_inputs_are_untouched() -> None:
    x = _payload(RANK, 64, 3072, 21)
    residual, weight = _residual_and_weight(64, 3072, 21)
    x0, r0 = x.clone(), residual.clone()
    _call(x, residual, weight, 1e-6)
    torch.cuda.synchronize()
    assert torch.equal(x, x0) and torch.equal(residual, r0), "the call wrote an input"
    dist.barrier()


def check_cuda_graph_replay_at_engine_batch_sizes() -> None:
    """Captured once per decode batch size, refilled and replayed: the Lamport
    buffers rotate across calls, and a replay must still read fresh data."""
    for rows in GRAPH_BATCH_SIZES:
        x = _payload(RANK, rows, 3072, 31)
        residual, weight = _residual_and_weight(rows, 3072, 31)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(2):
                _call(x, residual, weight, 1e-6)
        torch.cuda.current_stream().wait_stream(stream)
        dist.barrier()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = _call(x, residual, weight, 1e-6)
        for rep in range(3):
            seed = 31 + 100 * (rep + 1) + rows
            x.copy_(_payload(RANK, rows, 3072, seed))
            fresh_res, _ = _residual_and_weight(rows, 3072, seed)
            residual.copy_(fresh_res)
            dist.barrier()
            graph.replay()
            torch.cuda.synchronize()
            ref = _reference(rows, 3072, seed, residual, weight, 1e-6)
            try:
                _compare(out, ref)
            except AssertionError as exc:
                raise AssertionError(f"graph replay rows={rows} rep={rep}: {exc}") from exc
        dist.barrier()


def check_the_gate_discriminates_a_wrong_reduction() -> None:
    """A reference missing one rank must fail the band."""
    rows = 64
    x = _payload(RANK, rows, 3072, 81)
    residual, weight = _residual_and_weight(rows, 3072, 81)
    out = _call(x, residual, weight, 1e-6)
    short = [_payload(r, rows, 3072, 81) for r in range(WORLD - 1)]
    wrong = entry.mnnvl_allreduce.reference(short, residual, weight, 1e-6)
    try:
        _compare(out, wrong)
    except AssertionError:
        pass
    else:
        raise AssertionError("the band accepted a sum that is missing one rank")
    dist.barrier()


def check_the_guard_refuses_a_call_past_the_workspace() -> None:
    """The kernel never checks the row count against its buffer; the guard
    must. Presented a buffer view too small for the call, it refuses before
    the op runs, so no rank enters the collective."""
    rows = 2048
    x = _payload(RANK, rows, 3072, 91)
    residual, weight = _residual_and_weight(rows, 3072, 91)
    small = WORKSPACE_ARGS["comm_buffer"][:, :1024]
    try:
        with validating(entry.mnnvl_allreduce):
            mnnvl_allreduce(x, residual, norm_weight=weight, eps=1e-6, comm_buffer=small)
    except AssertionError as exc:
        assert "workspace" in str(exc), exc
    else:
        raise AssertionError("the guard admitted a call larger than the workspace")
    dist.barrier()


CHECKS = (
    check_certified_cells,
    check_inputs_are_untouched,
    check_cuda_graph_replay_at_engine_batch_sizes,
    check_the_gate_discriminates_a_wrong_reduction,
    check_the_guard_refuses_a_call_past_the_workspace,
)


def _run_one_rank(device: int) -> int:
    """Body of one rank process. `device` is this node's local index, so the
    ranks of one node sit on that node's four visible devices."""
    global RANK, WORLD, WORKSPACE_ARGS, mnnvl_allreduce, entry, validating
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    assert world == NODES * GPUS_PER_NODE, (world, NODES * GPUS_PER_NODE)
    torch.cuda.set_device(device)
    # cuda:nccl,cpu:gloo, as a Ray worker builds it: the workspace's setup
    # handshake runs on the TP group's CPU backend.
    dist.init_process_group(
        "cuda:nccl,cpu:gloo", init_method="env://", device_id=torch.device("cuda", device)
    )
    RANK, WORLD = rank, world

    from tensorrt_llm._torch.modeling_v2.models.hyv4_flash.hy4_flash_fp8__sm_103__default.ops.comm import (  # noqa: E501
        mnnvl_allreduce__r22 as entry,
    )
    from tensorrt_llm.mapping import Mapping

    mapping = Mapping(world_size=world, rank=rank, gpus_per_node=GPUS_PER_NODE, tp_size=world)
    assert mapping.is_multi_node(), "these cells certify a group that spans trays"
    # What the target does in __init__: build the workspace once and keep its
    # arguments. One workspace for every cell, sized by the widest hidden a
    # cell certifies. The partial is only the call surface; the guard is armed
    # on `entry.mnnvl_allreduce` itself.
    widest = max(cell.spec["hidden"] for cell in entry.mnnvl_allreduce.CELLS)
    WORKSPACE_ARGS = entry.mnnvl_workspace_args(
        mapping, max_rows=MAX_ROWS, hidden=widest, dtype=torch.bfloat16
    )
    mnnvl_allreduce = functools.partial(entry.mnnvl_allreduce, **WORKSPACE_ARGS)

    sys.path.insert(0, str(_MODELING_V2_TESTS))
    from _validating import validating as _validating_cm

    validating = _validating_cm

    for check in CHECKS:
        try:
            check()
        except BaseException:
            import traceback

            print(f"[rank {RANK}] FAILED {check.__name__}", flush=True)
            traceback.print_exc()
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(13)
    dist.barrier()
    torch.cuda.synchronize()
    print(f"[rank {RANK}] {len(CHECKS)} checks passed", flush=True)
    sys.stdout.flush()
    os._exit(0)


def _node_env() -> tuple[int, str]:
    """This node's rank in the gang and the rendezvous host, from the spool."""
    nodes = int(os.environ.get("MV2FLOW_JOB_NODES", "1"))
    assert nodes == NODES, (
        f"these cells are world {NODES * GPUS_PER_NODE} over {NODES} GB300 trays: "
        f"run this file as a --nodes {NODES} spool job"
    )
    node_rank = int(os.environ["MV2FLOW_JOB_NODE_RANK"])
    hosts = os.environ["MV2FLOW_JOB_HOSTS"].split(",")
    assert len(hosts) == nodes, hosts
    master = os.environ["MV2FLOW_JOB_MASTER"]
    assert master == hosts[0], (master, hosts)
    return node_rank, master


def _spawn_node_ranks(node_rank: int, master: str) -> None:
    """Start this node's four ranks on its visible devices, rendezvousing with
    the other node's on `master`; kill them all and raise on the first nonzero
    exit or past `DEADLINE_S`."""
    port = int(os.environ.get("MV2V_MNNVL_PORT", "29573"))
    base = node_rank * GPUS_PER_NODE
    env = dict(
        os.environ,
        MASTER_ADDR=master,
        MASTER_PORT=str(port),
        WORLD_SIZE=str(NODES * GPUS_PER_NODE),
        TLLM_DISABLE_MPI="1",
    )
    # Drop any PMIx/PMI/OMPI/SLURM namespace a parent step left behind: a child
    # that inherits it joins that step instead of this rendezvous.
    env = {k: v for k, v in env.items() if not k.startswith(("PMIX_", "PMI_", "OMPI_", "SLURM_"))}
    print(
        f"[launcher] node {node_rank}: ranks {base}..{base + GPUS_PER_NODE - 1} "
        f"rendezvous on {master}:{port}",
        flush=True,
    )
    children = [
        subprocess.Popen(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                _WORKER_FLAG,
                str(local),
            ],
            env=dict(env, RANK=str(base + local)),
            start_new_session=True,
        )
        for local in range(GPUS_PER_NODE)
    ]
    deadline = time.monotonic() + DEADLINE_S
    live = set(range(GPUS_PER_NODE))
    failure = None
    while live and failure is None:
        if time.monotonic() > deadline:
            failure = f"did not finish in {DEADLINE_S}s (wedged)"
            break
        for local in sorted(live):
            code = children[local].poll()
            if code is None:
                continue
            live.discard(local)
            if code != 0:
                failure = f"rank {base + local} exited {code}"
                break
        time.sleep(0.5)
    if failure is not None:
        for p in children:
            try:
                os.killpg(p.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        for p in children:
            p.wait()
        raise AssertionError(f"node {node_rank}: {failure}")


if __name__ == "__main__":
    if _WORKER_FLAG in sys.argv:
        sys.exit(_run_one_rank(device=int(sys.argv[sys.argv.index(_WORKER_FLAG) + 1])))
    assert torch.cuda.device_count() == GPUS_PER_NODE, (
        f"this node holds {torch.cuda.device_count()} devices, a tray has {GPUS_PER_NODE}"
    )
    node_rank, master = _node_env()
    _spawn_node_ranks(node_rank, master)
    print(
        f"OK: node {node_rank} of {NODES} passed ({NODES * GPUS_PER_NODE} ranks, "
        f"hidden 3072, rows 1..{MAX_ROWS})",
        flush=True,
    )
