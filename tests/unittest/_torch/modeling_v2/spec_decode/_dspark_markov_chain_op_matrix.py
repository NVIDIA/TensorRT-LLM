# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Certify `spec_decode/dspark_markov_chain`: the one-launch chain's draft tokens against the reference chain's, bit
for bit, on every cell certified at this run's world size.

Every rank builds the same full-vocabulary tensors from one seed and passes its padded shard to the op; rank 0 checks
the tokens against `reference` over every rank's shard. Per cell: random logits and logits on a coarse grid (many
exact bf16 ties, so the tie order is exercised), each of the cell's batches, several seeds, inside `validating`. Also:
three repeated calls give identical tokens and a CUDA-graph replay gives the eager tokens; `supports` refuses a batch
above the kernel's capacity and fp32 logits; the guard refuses a padding column that is not -inf.

World 4 (one GB300 tray) is spawned from one process:

    CUDA_VISIBLE_DEVICES=0,1,2,3 python _dspark_markov_chain_op_matrix.py

World 8 spans two trays and runs as one srun task per rank, like `comm/_mnnvl_allreduce_op_matrix.py`:

    MASTER_ADDR=<first node> MASTER_PORT=<port> python _dspark_markov_chain_op_matrix.py --srun

under ``srun --ntasks=8 --ntasks-per-node=4 --kill-on-bad-exit=1``.
"""

import importlib.util
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Optional

import torch
import torch.distributed as dist

assert torch.cuda.is_available(), "dspark_markov_chain requires CUDA devices"

_WORKER_FLAG = "--rank-worker"
DEADLINE_S = 900
RANK_DIM = 256
SEEDS = 8
GPUS_PER_NODE = 4
"""A GB300 tray. The mapping needs it to see a world-8 group span two nodes."""

RANK = 0
WORLD = 1
# Bound inside the rank body: `entry` is the module, `chain` its `dspark_markov_chain` instance (the call surface,
# `reference`, `is_valid` and `CELLS`), `WS` the workspace arguments of every call.
entry: Any = None
chain: Any = None
validating: Any = None
WS: dict = {}


def _weights(vocab: int):
    g = torch.Generator(device="cuda").manual_seed(1234)
    w1 = (torch.randn(vocab, RANK_DIM, device="cuda", generator=g) * 0.05).bfloat16()
    w2 = (torch.randn(vocab, RANK_DIM, device="cuda", generator=g) * 0.05).bfloat16()
    return w1, w2


def _inputs(seed: int, batch: int, block: int, vocab: int, ties: bool):
    g = torch.Generator(device="cuda").manual_seed(seed)
    base = torch.randn(batch, block, vocab, device="cuda", generator=g) * 3
    if ties:
        base = torch.round(base * 2) / 2
    prev0 = torch.randint(0, vocab, (batch,), device="cuda", generator=g)
    return base.bfloat16(), prev0


class _Shard:
    """One vocabulary's shards: this rank's padded `w2` rows and logits, and every rank's for the reference."""

    def __init__(self, vocab: int):
        self.vocab = vocab
        self.shard = vocab // WORLD
        self.rows = entry.padded_rows(self.shard, entry.CERTIFIED_BATCH)
        assert self.rows >= self.shard, f"no padded shard for {self.shard} rows"
        self.w1, w2 = _weights(vocab)
        self.w2s_all = [
            entry.pad_w2_shard(w2[r * self.shard : (r + 1) * self.shard].contiguous(), self.rows)
            for r in range(WORLD)
        ]
        self.w2s = self.w2s_all[RANK]

    def base(self, full: torch.Tensor, rank: int) -> torch.Tensor:
        b = entry.pad_logits_buffer(full.shape[0], full.shape[1], self.rows, full.device)
        b[:, :, : self.shard] = full[:, :, rank * self.shard : (rank + 1) * self.shard]
        return b

    def call(self, base: torch.Tensor, prev0: torch.Tensor) -> torch.Tensor:
        return chain(base, prev0, self.w1, self.w2s, RANK * self.shard, **WS)

    def reference(self, full: torch.Tensor, prev0: torch.Tensor) -> torch.Tensor:
        bases = [self.base(full, r) for r in range(WORLD)]
        return chain.reference(bases, prev0, self.w1, self.w2s_all, self.shard)


def check_certified_cells() -> None:
    cells = [c for c in chain.CELLS if c.spec["world"] == WORLD]
    assert cells, f"no cell is certified at world size {WORLD}"
    for cell in cells:
        spec = cell.spec
        sh = _Shard(spec["vocab"])
        for ties in (False, True):
            for batch in spec["batches"]:
                for seed in range(SEEDS):
                    full, prev0 = _inputs(seed, batch, spec["block"], sh.vocab, ties)
                    with validating(chain):
                        got = sh.call(sh.base(full, RANK), prev0)
                    if RANK == 0:
                        ref = sh.reference(full, prev0)
                        assert torch.equal(got, ref), (
                            f"{cell.why}: ties {ties} batch {batch} seed {seed}: "
                            f"{(got != ref).any(1).sum().item()} request(s) differ"
                        )
        dist.barrier()
        if RANK == 0:
            print(f"[cell] world={WORLD} {cell.why}: bitwise", flush=True)


def check_deterministic_and_graph() -> None:
    sh = _Shard(200064)
    full, prev0 = _inputs(77, entry.CERTIFIED_BATCH, 7, sh.vocab, False)
    base = sh.base(full, RANK)
    outs = [sh.call(base, prev0) for _ in range(3)]
    assert all(torch.equal(outs[0], o) for o in outs[1:]), "repeated calls differ"
    graph_out = {}
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    g = torch.cuda.CUDAGraph()
    with torch.cuda.stream(s):
        with torch.cuda.graph(g):
            graph_out["t"] = sh.call(base, prev0)
    torch.cuda.current_stream().wait_stream(s)
    for _ in range(3):
        g.replay()
        torch.cuda.synchronize()
        assert torch.equal(graph_out["t"], outs[0]), "graph replay differs from eager"
    dist.barrier()


def check_supports_refuses() -> None:
    sh = _Shard(200064)
    kern = entry._kernel_module()
    full, prev0 = _inputs(5, kern.MAX_BATCH + 1, 3, sh.vocab, False)
    assert not entry.supports(sh.base(full, RANK), sh.w1, sh.w2s), "a batch above MAX_BATCH"
    full, prev0 = _inputs(5, 2, 3, sh.vocab, False)
    assert not entry.supports(sh.base(full, RANK).float(), sh.w1, sh.w2s), "fp32 logits"
    assert not entry.supports(sh.base(full, RANK)[:, :, :-64], sh.w1, sh.w2s), (
        "logits narrower than w2_shard"
    )
    dist.barrier()


def check_the_guard_refuses_a_live_padding_column() -> None:
    sh = _Shard(200064)
    assert sh.rows > sh.shard, "the M3.1 shard is padded"
    full, prev0 = _inputs(9, 2, 3, sh.vocab, False)
    base = sh.base(full, RANK)
    base[:, :, -1] = 0.0
    try:
        chain.is_valid(base, prev0, sh.w1, sh.w2s, RANK * sh.shard, **WS)
    except AssertionError:
        pass
    else:
        raise AssertionError("the guard admitted a padding column that is not -inf")
    dist.barrier()


CHECKS = (
    check_certified_cells,
    check_deterministic_and_graph,
    check_supports_refuses,
    check_the_guard_refuses_a_live_padding_column,
)


def _run_one_rank(device: int) -> int:
    global RANK, WORLD, WS, entry, chain, validating
    os.environ.setdefault("TLLM_DISABLE_MPI", "1")
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(device)
    dist.init_process_group(
        "cuda:nccl,cpu:gloo", init_method="env://", device_id=torch.device("cuda", device)
    )
    RANK, WORLD = rank, world

    from tensorrt_llm._torch._experimental.modeling_v2.catalog.spec_decode import (
        dspark_markov_chain as entry_module,
    )
    from tensorrt_llm.mapping import Mapping

    entry = entry_module
    chain = entry.dspark_markov_chain
    mapping = Mapping(world_size=world, rank=rank, gpus_per_node=GPUS_PER_NODE, tp_size=world)
    WS = entry.workspace_args(mapping, dist.group.WORLD)

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
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
        print(f"[rank {RANK}] {check.__name__} passed", flush=True)
    dist.barrier()
    torch.cuda.synchronize()
    sys.stdout.flush()
    os._exit(0)


def _spawn_ranks() -> None:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    assert visible, "set CUDA_VISIBLE_DEVICES to the devices this run owns"
    world_size = len([d for d in visible.split(",") if d.strip()])
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    env = dict(os.environ, MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), WORLD_SIZE=str(world_size),
               TLLM_DISABLE_MPI="1")  # fmt: skip
    env = {k: v for k, v in env.items() if not k.startswith(("PMIX_", "PMI_", "OMPI_", "SLURM_"))}
    children = [
        subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()), _WORKER_FLAG],
            env=dict(env, RANK=str(r), LOCAL_RANK=str(r)),
            start_new_session=True,
        )  # fmt: skip
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


def _rank_job():
    """`_rank_job.py` beside this file, by path (a sibling directory ships its own module of that name)."""
    path = Path(__file__).resolve().with_name("_rank_job.py")
    spec = importlib.util.spec_from_file_location("_spec_decode_rank_job", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


if __name__ == "__main__":
    if "--srun" in sys.argv:
        sys.exit(_run_one_rank(device=_rank_job().adopt_srun_env()))
    if _WORKER_FLAG in sys.argv:
        sys.exit(_run_one_rank(device=int(os.environ["RANK"])))
    _spawn_ranks()
    print("OK")
