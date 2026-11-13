# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""MARKOV_GRID on real ranks: the TP4 target's batch-1 chain launch on its wider grid (``dspark_markov_grid``, 198 x 256
on SM107) returns bitwise the catalog op's tokens (132 x 384) and KV-length rewind, over the catalog's multicast
workspace, with the M3.1 vocabulary split over 4 ranks.

Checks (batch 1; K 7 and 3 with the MARKOV_LL exchange on, the deployed K 7 with it off: the Lamport batch-1 kernel):
- random logits, logits on a coarse grid (many exact ties), all-equal values (zero ``markov_w2``: every token 0) and
  NaN-free extremes; rank 0 also against the catalog ``reference`` (the in-tree chain on the gathered logits);
- forced ties at fixed columns (their ``markov_w2`` rows zeroed): across ranks, rank boundaries, the boundaries of the
  wide grid's CTAs (every 256 rows) and of the catalog grid's (every 384 rows), two warps of one CTA, every rank's
  last real column; the token must be the lowest tied column;
- the rewind variant: tokens and ``kv_lens`` equal the catalog rewind call's (rows outside untouched, clamped rows);
- back to back without host syncs: wide-grid batch-1 launches interleaved with the catalog's batch-2 launches (the
  shared Lamport rotation) and its own batch-1 launches;
- a CUDA graph of wide-grid launches replayed with new inputs, against eager catalog calls;
- the worker's prebuild agreement on the grid (``S4WorkerMixin._s4_ranks_agree_value`` over the real TP collective):
  equal grids agree, one rank's other grid or no grid disagrees, on every rank.

World 4 (one tray) is spawned from one process, each rank in its own session; ``DEADLINE_S`` kills a wedged run. The
rank workers never outlive the launcher: it kills them on any exit (a failure, the deadline, SIGTERM / SIGINT /
SIGHUP, e.g. from ``timeout``), and each worker asks the kernel for SIGKILL when the launcher dies (PR_SET_PDEATHSIG),
which also covers a SIGKILLed launcher:

    CUDA_VISIBLE_DEVICES=0,1,2,3 python _m31_markov_grid_op_matrix.py
"""

import atexit
import ctypes
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, List, Optional

import torch
import torch.distributed as dist

_WORKER_FLAG = "--rank-worker"
_PARENT_ENV = "R6_MARKOV_GRID_MATRIX_PARENT"
# The whole matrix (4 ranks compiling their CuTe DSL variants at once, then the checks); its pytest wrapper
# (test_modeling_v2_m31_r6_markov_grid_matrix.py) allows DEADLINE_S + 120 s.
DEADLINE_S = 600
RANK_DIM = 256
VOCAB = 200064
SEEDS = 3
BF16_MAX = torch.finfo(torch.bfloat16).max
BF16_DENORM_MIN = 2.0**-133

RANK = 0
WORLD = 1
chain: Any = None
grid_mod: Any = None
WS: dict = {}


def _set_ll(on: bool) -> None:
    chain.enable_batch1_variant(True)
    chain.enable_ll_exchange(on)


def _weights(vocab: int, zero_rows: Optional[List[int]] = None, zero_w2: bool = False):
    g = torch.Generator(device="cuda").manual_seed(1234)
    w1 = (torch.randn(vocab, RANK_DIM, device="cuda", generator=g) * 0.05).bfloat16()
    w2 = (torch.randn(vocab, RANK_DIM, device="cuda", generator=g) * 0.05).bfloat16()
    if zero_w2:
        w2.zero_()
    if zero_rows:
        w2[torch.tensor(zero_rows, device="cuda")] = 0
    return w1, w2


class _Shard:
    def __init__(self, vocab: int, **weights):
        self.vocab = vocab
        self.shard = vocab // WORLD
        self.rows = chain.padded_rows(self.shard, chain.CERTIFIED_BATCH)
        self.w1, w2 = _weights(vocab, **weights)
        self.w2s_all = [
            chain.pad_w2_shard(w2[r * self.shard : (r + 1) * self.shard].contiguous(), self.rows)
            for r in range(WORLD)
        ]
        self.w2s = self.w2s_all[RANK]

    def base(self, full: torch.Tensor, rank: int) -> torch.Tensor:
        b = chain.pad_logits_buffer(full.shape[0], full.shape[1], self.rows, full.device)
        b[:, :, : self.shard] = full[:, :, rank * self.shard : (rank + 1) * self.shard]
        return b

    def grid(self, block: int) -> int:
        return grid_mod.plan(self.rows, block, torch.device("cuda"))

    def wide(self, base, prev0, **rewind):
        grid = self.grid(base.shape[1])
        assert grid > 0 and grid_mod.supports(base, self.w1, self.w2s, grid)
        return grid_mod.markov_chain_b1(
            base, prev0, self.w1, self.w2s, RANK * self.shard, grid, **rewind, **WS
        )

    def catalog(self, base, prev0):
        return chain.dspark_markov_chain(base, prev0, self.w1, self.w2s, RANK * self.shard, **WS)

    def reference(self, full, prev0):
        bases = [self.base(full, r) for r in range(WORLD)]
        return chain.dspark_markov_chain.reference(bases, prev0, self.w1, self.w2s_all, self.shard)


def _inputs(seed: int, batch: int, block: int, vocab: int, kind: str):
    g = torch.Generator(device="cuda").manual_seed(seed)
    base = torch.randn(batch, block, vocab, device="cuda", generator=g) * 3
    if kind == "ties":
        base = torch.round(base * 2) / 2
    elif kind == "equal":
        base = torch.ones_like(base)
    elif kind == "extremes":
        pool = torch.tensor(
            [
                BF16_MAX,
                -BF16_MAX,
                BF16_DENORM_MIN,
                -BF16_DENORM_MIN,
                0.0,
                -0.0,
                float("-inf"),
                1.0,
                -1.0,
            ],
            device="cuda",
        )
        base = pool[torch.randint(0, pool.numel(), base.shape, device="cuda", generator=g)]
    prev0 = torch.randint(0, vocab, (batch,), device="cuda", generator=g)
    return base.bfloat16(), prev0


def _eq(what: str, got: torch.Tensor, want: torch.Tensor) -> None:
    assert torch.equal(got, want), f"{what}: wide grid {got.tolist()} != catalog {want.tolist()}"


def check_grid_is_wider() -> None:
    sh = _Shard(VOCAB)
    for ll in (True, False):
        _set_ll(ll)
        for block in (3, 7) if ll else (7,):
            grid = sh.grid(block)
            cat = grid_mod.catalog_grid(sh.rows, block, torch.device("cuda"))
            assert grid > cat > 0, (grid, cat)
            assert sh.rows % grid == 0 and (sh.rows // grid) % 64 == 0
            assert grid_mod.fits_workspace(block, grid, WS)
            if RANK == 0:
                print(
                    f"[grid] ll={ll} K {block}: {grid} x {sh.rows // grid} (catalog {cat})",
                    flush=True,
                )
    dist.barrier()


def check_tokens() -> None:
    sh = _Shard(VOCAB)
    sh_eq = _Shard(VOCAB, zero_w2=True)
    for ll in (True, False):
        _set_ll(ll)
        for block in (7, 3) if ll else (7,):
            for kind in ("random", "ties", "equal", "extremes"):
                s = sh_eq if kind == "equal" else sh
                for seed in range(SEEDS):
                    full, prev0 = _inputs(seed, 1, block, s.vocab, kind)
                    base = s.base(full, RANK)
                    got, want = s.wide(base, prev0), s.catalog(base, prev0)
                    _eq(f"ll={ll} K {block} {kind} seed {seed}", got, want)
                    if kind == "equal":
                        assert got.eq(0).all()
                    if RANK == 0:
                        _eq(
                            f"ll={ll} K {block} {kind} seed {seed} vs reference",
                            got,
                            s.reference(full, prev0),
                        )
    dist.barrier()
    if RANK == 0:
        print(
            "[grid] tokens == catalog (LL and Lamport; K 7 / 3; random, ties, equal, extremes)",
            flush=True,
        )


def _tie_patterns(shard: int) -> List[List[int]]:
    last = shard - 1
    pats = [
        [r * shard + 5 for r in range(1, WORLD)],
        [last, shard],  # rank 0's last real column vs rank 1's first
        [(WORLD - 1) * shard, (WORLD - 2) * shard + last],
        [255, 256],  # wide grid: CTA 0's last row vs CTA 1's first
        [383, 384],  # catalog grid: the same at 384 rows
        [shard + 511, shard + 512, 2 * shard + 767, 2 * shard + 768],
        [
            shard + 32 + 3,
            shard + 3,
        ],  # rank 1, CTA 0: warp 1 vs warp 0 (32 rows per warp on the wide grid)
        [r * shard + last for r in range(WORLD)][::-1],
        [(WORLD - 1) * shard + 7, 7],
    ]
    return [sorted(set(p)) for p in pats]


def check_forced_ties() -> None:
    pats = _tie_patterns(VOCAB // WORLD)
    zero_rows = sorted({c for p in pats for c in p})
    sh = _Shard(VOCAB, zero_rows=zero_rows)
    checked = 0
    for ll in (True, False):
        _set_ll(ll)
        for block in (7, 3) if ll else (7,):
            for name, tie, neg_inf in (("large", 100.0, False), ("bf16max", BF16_MAX, False),
                                       ("denormal", BF16_DENORM_MIN, True), ("signed-zero", 0.0, True)):  # fmt: skip
                for offset in range(len(pats)):
                    full, prev0 = _inputs(1000 + offset, 1, block, sh.vocab, "random")
                    full = full.float()
                    expect = []
                    for k in range(block):
                        cols = pats[(offset + k) % len(pats)]
                        if neg_inf:
                            full[0, k, :] = float("-inf")
                        idx = torch.tensor(cols, device="cuda")
                        if name == "signed-zero":
                            full[0, k, idx] = torch.tensor(
                                [(-0.0 if i % 2 == 0 else 0.0) for i in range(len(cols))],
                                device="cuda",
                            )
                        else:
                            full[0, k, idx] = tie
                        expect.append(min(cols))
                    full = full.bfloat16()
                    base = sh.base(full, RANK)
                    got, want = sh.wide(base, prev0), sh.catalog(base, prev0)
                    what = f"forced ties ll={ll} {name} K {block} offset {offset}"
                    _eq(what, got, want)
                    assert torch.equal(
                        got, torch.tensor([expect], dtype=torch.int32, device="cuda")
                    ), what
                    checked += 1
    dist.barrier()
    if RANK == 0:
        print(f"[grid] forced ties: {checked} cases == catalog == lowest tied column", flush=True)


def check_rewind() -> None:
    sh = _Shard(VOCAB)
    for ll in (True, False):
        _set_ll(ll)
        for amount, start in ((3, 10), (-2, 5), (9, 4)):
            full, prev0 = _inputs(50 + amount, 1, 7, sh.vocab, "random")
            base = sh.base(full, RANK)
            kv_a = torch.arange(6, dtype=torch.int32, device="cuda") + start
            kv_b = kv_a.clone()
            rew = torch.tensor([amount], dtype=torch.int32, device="cuda")
            got = sh.wide(base, prev0, kv_lens=kv_a, rewind=rew, rewind_first=2)
            want = chain.dspark_markov_chain_rewind(
                base, prev0, sh.w1, sh.w2s, RANK * sh.shard, kv_b, rew, 2, **WS
            )
            torch.cuda.synchronize()
            _eq(f"rewind ll={ll} {amount}", got, want)
            assert torch.equal(kv_a, kv_b), (kv_a.tolist(), kv_b.tolist())
    dist.barrier()
    if RANK == 0:
        print("[grid] rewind: tokens and kv_lens == catalog", flush=True)


def check_back_to_back() -> None:
    sh = _Shard(VOCAB)
    _set_ll(True)
    plan = [
        ("wide", 7, 1),
        ("cat", 7, 2),
        ("wide", 3, 1),
        ("cat", 7, 1),
        ("wide", 7, 1),
        ("cat", 7, 2),
    ] * 8
    inputs = []
    for i, (kind, block, batch) in enumerate(plan):
        full, prev0 = _inputs(5000 + i, batch, block, sh.vocab, "ties" if i % 3 == 0 else "random")
        inputs.append((sh.base(full, RANK), prev0))
    for kind, block, batch in set(plan):  # compile outside the sequence
        base, prev0 = inputs[plan.index((kind, block, batch))]
        (sh.wide if kind == "wide" else sh.catalog)(base, prev0)
    torch.cuda.synchronize()
    dist.barrier()
    outs = [
        (sh.wide if k == "wide" else sh.catalog)(b, p) for (k, _, _), (b, p) in zip(plan, inputs)
    ]
    torch.cuda.synchronize()
    refs = [sh.catalog(b, p) for b, p in inputs]
    torch.cuda.synchronize()
    for i, (got, want) in enumerate(zip(outs, refs)):
        _eq(f"back-to-back launch {i} {plan[i]}", got, want)
    dist.barrier()
    if RANK == 0:
        print(f"[grid] back-to-back: {len(plan)} mixed launches == catalog", flush=True)


def check_cuda_graph() -> None:
    sh = _Shard(VOCAB)
    _set_ll(True)
    static = []
    for j, block in enumerate((7, 7, 3)):
        full, prev0 = _inputs(300 + j, 1, block, sh.vocab, "random")
        static.append((sh.base(full, RANK), prev0.clone()))
    for base, prev0 in static:
        sh.wide(base, prev0)
    torch.cuda.synchronize()
    dist.barrier()
    graph_out = []
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    g = torch.cuda.CUDAGraph()
    with torch.cuda.stream(s):
        with torch.cuda.graph(g):
            for base, prev0 in static:
                graph_out.append(sh.wide(base, prev0))
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    dist.barrier()
    for rep in range(8):
        for j, (base, prev0) in enumerate(static):
            full, p = _inputs(
                400 + rep * 8 + j, 1, base.shape[1], sh.vocab, "ties" if rep % 2 else "random"
            )
            base.copy_(sh.base(full, RANK))
            prev0.copy_(p)
        g.replay()
        torch.cuda.synchronize()
        got = [t.clone() for t in graph_out]
        refs = [sh.catalog(b, p) for b, p in static]
        torch.cuda.synchronize()
        for j, (a, b) in enumerate(zip(got, refs)):
            _eq(f"graph replay {rep} launch {j}", a, b)
    dist.barrier()
    if RANK == 0:
        print("[grid] CUDA graph: 8 replays x 3 launches == eager catalog", flush=True)


def check_rank_agreement() -> None:
    """The worker's prebuild agreement on the batch-1 grid over the real TP collective (TLLM_DISABLE_MPI: the TP
    process group's CPU backend, as _get_mnnvl_workspace_comm / _mnnvl_workspace_all_succeeded use it)."""
    from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_1_flash_nvfp4__sm_107__tp4.dspark_markov_worker import (  # noqa: E501
        S4WorkerMixin,
    )

    worker = SimpleNamespace(mapping=SimpleNamespace(tp_group_pg=dist.group.WORLD))
    for local, want in (
        (198, True),
        (132 if RANK == WORLD - 2 else 198, False),
        (0 if RANK == WORLD - 1 else 198, False),
        (0, False),
        (256, True),
    ):
        got = S4WorkerMixin._s4_ranks_agree_value(worker, local)
        assert got is want, f"rank agreement on {local}: {got}, want {want}"
    dist.barrier()
    if RANK == 0:
        print("[grid] prebuild rank agreement over the TP collective", flush=True)


CHECKS = (
    check_rank_agreement,
    check_grid_is_wider,
    check_tokens,
    check_forced_ties,
    check_rewind,
    check_back_to_back,
    check_cuda_graph,
)


def _die_with_parent() -> None:
    """SIGKILL this rank worker when the launcher dies (Linux PR_SET_PDEATHSIG), and exit now if it already did."""
    pr_set_pdeathsig = 1
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(pr_set_pdeathsig, int(signal.SIGKILL), 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), "prctl(PR_SET_PDEATHSIG) failed")
    parent = os.environ.get(_PARENT_ENV)
    if parent is not None and os.getppid() != int(parent):
        os._exit(14)


def _run_one_rank(device: int) -> int:
    global RANK, WORLD, WS, chain, grid_mod
    _die_with_parent()
    os.environ.setdefault("TLLM_DISABLE_MPI", "1")
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(device)
    dist.init_process_group(
        "cuda:nccl,cpu:gloo", init_method="env://", device_id=torch.device("cuda", device)
    )
    RANK, WORLD = rank, world

    from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_1_flash_nvfp4__sm_107__tp4 import (  # noqa: E501
        dspark_markov_grid,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.ops.dspark import (
        dspark_markov_chain as chain_module,
    )
    from tensorrt_llm.mapping import Mapping

    chain = chain_module
    grid_mod = dspark_markov_grid
    chain.enable_fp32_sum(True)
    chain.enable_cached_inputs(True)
    mapping = Mapping(world_size=world, rank=rank, gpus_per_node=4, tp_size=world)
    WS = chain.workspace_args(mapping, dist.group.WORLD)
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


def spawn(world_size: int = 4) -> None:
    """Run the matrix over ``world_size`` visible devices (one process per rank) and raise on any failure."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    devices = (
        [d for d in visible.split(",") if d.strip()]
        if visible
        else [str(i) for i in range(torch.cuda.device_count())]
    )
    assert len(devices) >= world_size, f"needs {world_size} devices, {len(devices)} visible"
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    env = dict(os.environ, MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), WORLD_SIZE=str(world_size),
               TLLM_DISABLE_MPI="1", CUDA_VISIBLE_DEVICES=",".join(devices[:world_size]),
               **{_PARENT_ENV: str(os.getpid())})  # fmt: skip
    env = {k: v for k, v in env.items() if not k.startswith(("PMIX_", "PMI_", "OMPI_", "SLURM_"))}
    children: List[subprocess.Popen] = []

    def _kill_all() -> None:
        for p in children:
            if p.poll() is None:
                try:
                    os.killpg(p.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
        for p in children:
            p.wait()

    def _on_signal(signum, frame) -> None:
        _kill_all()
        os._exit(128 + signum)

    # Registered before the first worker starts: a signal or an exit at any later point takes the workers along.
    atexit.register(_kill_all)
    previous = {
        s: signal.signal(s, _on_signal) for s in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP)
    }
    try:
        for r in range(world_size):
            children.append(
                subprocess.Popen(
                    [sys.executable, str(Path(__file__).resolve()), _WORKER_FLAG],
                    env=dict(env, RANK=str(r), LOCAL_RANK=str(r)),
                    start_new_session=True,
                )  # fmt: skip
            )
        _wait_all(children, world_size, _kill_all)
    finally:
        _kill_all()
        for s, handler in previous.items():
            signal.signal(s, handler)
        atexit.unregister(_kill_all)


def _wait_all(children: List[subprocess.Popen], world_size: int, kill_all) -> None:
    """Wait for every rank; on the first failure or at the deadline kill them all and raise."""
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
        kill_all()
        raise AssertionError(f"the {world_size}-rank MARKOV_GRID matrix failed: {failure}")


if __name__ == "__main__":
    if _WORKER_FLAG in sys.argv:
        sys.exit(_run_one_rank(device=int(os.environ["RANK"])))
    spawn()
    print("OK")
