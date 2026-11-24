# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU matrix for ``ops/comm/allreduce__r19.py``; run plainly, it is its own launcher.

    CUDA_VISIBLE_DEVICES=0,1,2,3 python _allreduce__r19_op_matrix.py

It spawns one rank per visible device and kills them all on the first failure or at
the deadline, because a broken collective hangs rather than raising. A world over two
nodes runs one copy per node with ``MV2FLOW_JOB_NODES``, ``MV2FLOW_JOB_NODE_RANK``,
``MV2FLOW_JOB_HOSTS`` and ``MV2FLOW_JOB_MASTER`` set; every node rendezvous at the master.
"""

import functools
import os
import random
import signal
import subprocess
import sys
import time
import zlib
from pathlib import Path
from typing import Any, List

import torch
import torch.distributed as dist

assert torch.cuda.is_available(), "allreduce__r19 requires CUDA devices"

_WORKER_FLAG = "--rank-worker"
DEADLINE_S = 900

# Bound in the rank body: importing the entry pulls in tensorrt_llm, which the launcher must not.
allreduce: Any = None
entry: Any = None
validating: Any = None
RANK = 0
WORLD = 1
HOST_GROUP: Any = None

# Decode graph batch sizes, sampled: the smallest, a power of two, a non-power, the cap.
GRAPH_BATCH_SIZES = (1, 8, 24, 64)


def _compare(out, ref) -> None:
    """5 ulp per element against the row scale, 1.25 ulp relative RMS."""
    from _validating import assert_within_ulp

    assert_within_ulp(out, ref, element_ulp=5.0, rms_ulp=1.25)


class _HostExchange:
    """Stands in for `Distributed` in `IpcMemory`: its one call, a TP-group host allgather."""

    def tp_allgather(self, obj: Any) -> List[Any]:
        gathered: List[Any] = [None] * WORLD
        dist.all_gather_object(gathered, obj, group=HOST_GROUP)
        return gathered


def _payload(rank: int, rows: int, hidden: int, seed: int) -> torch.Tensor:
    """One rank's partial sum; every rank can regenerate every other rank's from the seed."""
    gen = torch.Generator(device="cuda").manual_seed(seed * 977 + rank + 1)
    return torch.randn(rows, hidden, generator=gen, device="cuda").to(torch.bfloat16)


def _call(x, eps, op):
    return allreduce(
        x,
        residual=None,
        norm_weight=None,
        scale=None,
        bias=None,
        op=op,
        eps=eps,
        trigger_completion_at_end=False,
    )


def _reference(rows, hidden, seed, eps, op):
    inputs = [_payload(r, rows, hidden, seed) for r in range(WORLD)]
    return entry.allreduce.reference(inputs, None, None, eps, op)


def _cells():
    return [c for c in entry.allreduce.CELLS if c.spec["world"] == WORLD]


def check_certified_cells() -> None:
    cells = _cells()
    assert cells, f"no cell is certified at world size {WORLD}"
    for i, cell in enumerate(cells):
        spec = cell.spec
        seed = 4000 + 17 * i
        x = _payload(RANK, spec["rows"], spec["hidden"], seed)
        with validating(entry.allreduce):
            out = _call(x, spec["eps"], spec["op"])
        ref = _reference(spec["rows"], spec["hidden"], seed, spec["eps"], spec["op"])
        try:
            _compare(out, ref)
        except AssertionError as exc:
            raise AssertionError(f"{cell.why}: {exc}") from exc
        dist.barrier()


def check_cuda_graph_replay_at_engine_batch_sizes() -> None:
    """One graph per decode batch size in a shared pool, replayed after refilling the inputs."""
    spec = _cells()[0].spec
    hidden, eps, op = spec["hidden"], spec["eps"], spec["op"]
    pool = None
    graphs = {}
    for rows in GRAPH_BATCH_SIZES:
        x = _payload(RANK, rows, hidden, 700 + rows)
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(2):
                _call(x, eps, op)
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()
        dist.barrier()
        graph = torch.cuda.CUDAGraph()
        ctx = torch.cuda.graph(graph) if pool is None else torch.cuda.graph(graph, pool=pool)
        with ctx:
            out = _call(x, eps, op)
        if pool is None:
            pool = graph.pool()
        graphs[rows] = (graph, x, out)
        dist.barrier()

    for step, rows in enumerate(reversed(GRAPH_BATCH_SIZES)):
        graph, x, out = graphs[rows]
        seed = 900 + step
        x.copy_(_payload(RANK, rows, hidden, seed))
        graph.replay()
        torch.cuda.synchronize()
        try:
            _compare(out.clone(), _reference(rows, hidden, seed, eps, op))
        except AssertionError as exc:
            raise AssertionError(f"replay rows={rows}: {exc}") from exc
        dist.barrier()


CHECKS = (
    check_certified_cells,
    check_cuda_graph_replay_at_engine_batch_sizes,
)


def _run_one_rank() -> int:
    global RANK, WORLD, HOST_GROUP, allreduce, entry, validating
    os.environ.setdefault("TLLM_DISABLE_MPI", "1")
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    node_rank = int(os.environ["NODE_RANK"])
    per_node = int(os.environ["PER_NODE"])
    local = int(os.environ["LOCAL_DEVICE"])
    # Never LOCAL_RANK: every spawned rank inherits the launching shell's one value.
    torch.cuda.set_device(local)
    dist.init_process_group(
        "cuda:nccl,cpu:gloo", init_method="env://", device_id=torch.device("cuda", local)
    )
    RANK, WORLD = rank, world
    HOST_GROUP = dist.new_group(backend="gloo")

    from tensorrt_llm._torch.distributed import communicator
    from tensorrt_llm._torch.modeling_v2.models.hyv4_flash.hy4_flash_fp8__sm_103__default.ops.comm import (  # noqa: E501
        allreduce__r19 as entry,
    )
    from tensorrt_llm._utils import torch_pybind11_abi
    from tensorrt_llm.bindings.internal.process_group import init_pg
    from tensorrt_llm.mapping import Mapping

    # The `_pg` symbol reads the world and node-local groups registered on the C++ side
    # (as `TorchDist` does) and dereferences null without them. `new_group` is
    # collective: every rank creates every node's group, in the same order.
    local_group = None
    for start in range(0, world, per_node):
        pg = dist.new_group(
            ranks=list(range(start, start + per_node)),
            backend="cuda:nccl,cpu:gloo",
        )
        if start == node_rank * per_node:
            local_group = pg
    init_pg(dist.group.WORLD, local_group, torch_pybind11_abi())

    # `TorchDist` needs Ray; the IPC handle exchange is all these ranks need from it.
    communicator.Distributed.get = staticmethod(lambda mapping: _HostExchange())
    mapping = Mapping(world_size=world, rank=rank, gpus_per_node=per_node, tp_size=world)
    allreduce = functools.partial(entry.allreduce, **entry.allreduce_group_args(mapping))

    # parents[5] is tests/unittest/_torch/modeling_v2, where `_validating.py` lives.
    sys.path.insert(0, str(Path(__file__).resolve().parents[5]))
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
            # A rank that leaves a collective early wedges its siblings; exit hard.
            os._exit(13)
    dist.barrier()
    torch.cuda.synchronize()
    print(f"[rank {RANK}] {len(CHECKS)} checks passed", flush=True)
    sys.stdout.flush()
    os._exit(0)


def _visible_devices() -> List[str]:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    assert visible, "set CUDA_VISIBLE_DEVICES to the devices this run owns"
    return [d for d in visible.split(",") if d.strip()]


def _argv_world_size():
    argv = sys.argv[1:]
    if "--world-size" not in argv:
        return None
    return int(argv[argv.index("--world-size") + 1])


def _spawn_ranks() -> None:
    """One worker per visible device of this node, all killed on the first nonzero exit or at the deadline."""
    devices = _visible_devices()
    per_node = len(devices)
    assert per_node >= 2, f"a collective test needs at least 2 devices, got {per_node}"
    nodes = int(os.environ.get("MV2FLOW_JOB_NODES") or 1)
    node_rank = int(os.environ.get("MV2FLOW_JOB_NODE_RANK") or 0)
    master = os.environ["MV2FLOW_JOB_MASTER"] if nodes > 1 else "127.0.0.1"
    world_size = nodes * per_node
    asked = _argv_world_size()
    assert asked in (None, world_size), (
        f"--world-size {asked} was asked for, but {nodes} node(s) x {per_node} device(s) "
        f"form {world_size}"
    )

    def _port(attempt: int) -> int:
        """Below the ephemeral range; across nodes, derived from the host list every node shares."""
        if nodes > 1:
            hosts = os.environ["MV2FLOW_JOB_HOSTS"]
            return 20000 + (zlib.crc32(hosts.encode()) + attempt * 7919) % 10000
        return random.SystemRandom().randrange(20000, 30000)

    attempts = 3
    failure = None
    for attempt in range(attempts):
        port = _port(attempt)
        env = dict(
            os.environ,
            MASTER_ADDR=master,
            MASTER_PORT=str(port),
            WORLD_SIZE=str(world_size),
            NODE_RANK=str(node_rank),
            PER_NODE=str(per_node),
            TLLM_DISABLE_MPI="1",
        )
        # An inherited srun PMI namespace lets one failing rank kill the whole step unreported.
        env = {
            k: v for k, v in env.items() if not k.startswith(("PMIX_", "PMI_", "OMPI_", "SLURM_"))
        }
        print(
            f"[launcher] node {node_rank}/{nodes}: spawning {per_node} ranks of {world_size}, "
            f"rendezvous on {master}:{port} (attempt {attempt + 1}/{attempts})",
            flush=True,
        )
        children = [
            subprocess.Popen(
                [sys.executable, str(Path(__file__).resolve()), _WORKER_FLAG],
                env=dict(env, RANK=str(node_rank * per_node + local), LOCAL_DEVICE=str(local)),
                start_new_session=True,
            )
            for local in range(per_node)
        ]
        deadline = time.monotonic() + DEADLINE_S
        live = set(range(per_node))
        failure = None
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
                    failure = f"rank {node_rank * per_node + r} exited {code}"
                    break
            time.sleep(0.5)
        if failure is None:
            return
        for p in children:
            try:
                os.killpg(p.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        for p in children:
            p.wait()
        print(f"[launcher] attempt {attempt + 1} failed: {failure}", flush=True)
    raise AssertionError(f"the {world_size}-rank run failed on node {node_rank}: {failure}")


if __name__ == "__main__":
    if _WORKER_FLAG in sys.argv:
        sys.exit(_run_one_rank())
    _spawn_ranks()
    print("OK")
