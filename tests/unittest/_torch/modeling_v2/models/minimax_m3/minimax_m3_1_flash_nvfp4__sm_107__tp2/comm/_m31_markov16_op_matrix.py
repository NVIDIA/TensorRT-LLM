# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Certify ``ops/spec/m31_markov16``: the TP2 16-row Markov chain's draft tokens against the entry's reference chain,
bit for bit, on every cell (world 2).

Every rank builds the same full-vocabulary tensors from one seed and passes its shard to the op: the unpadded shard
logits for a ``fold`` cell (``base_cols`` = the shard), the catalog's padded buffer otherwise (``base_cols`` = the
padded rows); rank 0 checks the tokens against ``reference`` over every rank's shard. Per cell: random logits and
logits on a coarse grid (many exact bf16 ties, so the tie order is exercised), each of the cell's batches, several
seeds, inside ``validating``. Also: three repeated calls give identical tokens and a CUDA-graph replay gives the eager
tokens.

Spawned from one process over two devices:

    CUDA_VISIBLE_DEVICES=0,1 python _m31_markov16_op_matrix.py
"""

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

assert torch.cuda.is_available(), "m31_markov16 requires CUDA devices"

_WORKER_FLAG = "--rank-worker"
DEADLINE_S = 900
RANK_DIM = 256
SEEDS = 4

RANK = 0
WORLD = 1
# Bound inside the rank body: `catalog` is the family's dspark_markov_chain module (padding and workspace),
# `entry` the m31_markov16 module, `op` its entry instance (the call surface, `reference`, `is_valid` and `CELLS`),
# `WS` the workspace arguments of every call.
catalog: Any = None
entry: Any = None
op: Any = None
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
    """One vocabulary's shards: this rank's padded ``w2`` rows and logits, and every rank's for the reference."""

    def __init__(self, vocab: int):
        self.vocab = vocab
        self.shard = vocab // WORLD
        self.rows = catalog.padded_rows(self.shard, catalog.CERTIFIED_BATCH_TP2)
        assert self.rows >= self.shard, f"no padded shard for {self.shard} rows"
        self.w1, w2 = _weights(vocab)
        self.w2s_all = [
            catalog.pad_w2_shard(w2[r * self.shard : (r + 1) * self.shard].contiguous(), self.rows)
            for r in range(WORLD)
        ]
        self.w2s = self.w2s_all[RANK]

    def base(self, full: torch.Tensor, rank: int, fold: bool) -> torch.Tensor:
        cols = full[:, :, rank * self.shard : (rank + 1) * self.shard]
        if fold:
            return cols.contiguous()
        b = catalog.pad_logits_buffer(full.shape[0], full.shape[1], self.rows, full.device)
        b[:, :, : self.shard] = cols
        return b

    def call(
        self, base: torch.Tensor, prev0: torch.Tensor, grid: int, w2_regs: bool
    ) -> torch.Tensor:
        kwargs = dict(WS, grid=grid, base_cols=base.shape[-1], w2_regs=w2_regs, fp32_sum=True)
        entry.compile_call(base, prev0, self.w1, self.w2s, RANK * self.shard, **kwargs)
        return op(base, prev0, self.w1, self.w2s, RANK * self.shard, **kwargs)

    def reference(self, full: torch.Tensor, prev0: torch.Tensor) -> torch.Tensor:
        bases = [self.base(full, r, fold=False) for r in range(WORLD)]
        return op.reference(bases, prev0, self.w1, self.w2s_all, self.shard, fp32_sum=True)


def check_certified_cells() -> None:
    cells = [c for c in op.CELLS if c.spec["world"] == WORLD]
    assert cells, f"no cell is certified at world size {WORLD}"
    for cell in cells:
        spec = cell.spec
        sh = _Shard(spec["vocab"])
        for ties in (False, True):
            for batch in spec["batches"]:
                for seed in range(SEEDS):
                    full, prev0 = _inputs(seed, batch, spec["block"], sh.vocab, ties)
                    base = sh.base(full, RANK, spec["fold"])
                    with validating(op):
                        got = sh.call(base, prev0, spec["grid"], spec["w2_regs"])
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
    spec = op.CELLS[0].spec
    sh = _Shard(spec["vocab"])
    full, prev0 = _inputs(77, max(spec["batches"]), spec["block"], sh.vocab, False)
    base = sh.base(full, RANK, spec["fold"])
    outs = [sh.call(base, prev0, spec["grid"], spec["w2_regs"]) for _ in range(3)]
    assert all(torch.equal(outs[0], o) for o in outs[1:]), "repeated calls differ"
    graph_out = {}
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    g = torch.cuda.CUDAGraph()
    with torch.cuda.stream(s):
        with torch.cuda.graph(g):
            graph_out["t"] = sh.call(base, prev0, spec["grid"], spec["w2_regs"])
    torch.cuda.current_stream().wait_stream(s)
    for _ in range(3):
        g.replay()
        torch.cuda.synchronize()
        assert torch.equal(graph_out["t"], outs[0]), "graph replay differs from eager"
    dist.barrier()


CHECKS = (check_certified_cells, check_deterministic_and_graph)


def _run_one_rank(device: int) -> int:
    global RANK, WORLD, WS, catalog, entry, op, validating
    os.environ.setdefault("TLLM_DISABLE_MPI", "1")
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(device)
    dist.init_process_group(
        "cuda:nccl,cpu:gloo", init_method="env://", device_id=torch.device("cuda", device)
    )
    RANK, WORLD = rank, world

    from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_1_flash_nvfp4__sm_107__tp2.ops.spec import (  # noqa: E501
        m31_markov16 as entry_module,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.ops.dspark import (
        dspark_markov_chain as catalog_module,
    )
    from tensorrt_llm.mapping import Mapping

    catalog, entry = catalog_module, entry_module
    op = entry.m31_markov16
    mapping = Mapping(world_size=world, rank=rank, gpus_per_node=world, tp_size=world)
    WS = catalog.workspace_args(mapping, dist.group.WORLD)

    # tests/unittest/_torch/modeling_v2, where _validating.py lives.
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
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
    env = dict(
        os.environ,
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(port),
        WORLD_SIZE=str(world_size),
        TLLM_DISABLE_MPI="1",
    )
    env = {k: v for k, v in env.items() if not k.startswith(("PMIX_", "PMI_", "OMPI_", "SLURM_"))}
    children = []

    def _kill_all() -> None:
        for p in children:
            try:
                os.killpg(p.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        for p in children:
            p.wait()

    failure: Optional[str] = None
    try:
        for r in range(world_size):
            children.append(
                subprocess.Popen(
                    [sys.executable, str(Path(__file__).resolve()), _WORKER_FLAG],
                    env=dict(env, RANK=str(r), LOCAL_RANK=str(r)),
                    start_new_session=True,
                )
            )
        deadline = time.monotonic() + DEADLINE_S
        live = set(range(world_size))
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
    finally:
        if failure is not None or any(p.poll() is None for p in children):
            _kill_all()
    if failure is not None:
        raise AssertionError(f"the {world_size}-rank run failed: {failure}")


if __name__ == "__main__":
    if _WORKER_FLAG in sys.argv:
        sys.exit(_run_one_rank(device=int(os.environ["RANK"])))
    _spawn_ranks()
    print("OK")
