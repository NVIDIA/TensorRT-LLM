# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""trtllm::k3_latent_reduce (the Kimi K3 latent all-reduce as the consumer of the push-only k3_moe), one process per
GPU over the TP group of this run (4 on one GB200 tray, 16 on four), at every M in 1..8 and every CTA split:
  each rank's partial is pushed as the push-only k3_moe stores it (bf16 pairs through the multicast mapping into slot
  [rank] of the call's half, -0.0 as +0.0); the reduce must equal MNNVLAllReduce's one-shot of the partials bit for
  bit, on every rank, run to run, with one rank's perturbed partial changing the result; after every call the whole
  buffer is empty again, the call count is +1 and the arrival word 0; a CUDA graph of four push + reduce pairs,
  replayed three times with new partials, matches the eager calls.

Run under pytest (a pool of 4 MPI workers) or directly, one process per GPU:
  srun -N1 -n4 --mpi=pmix python3 test_k3_latent_reduce.py
"""

import hashlib
import os
import pickle
import sys
import traceback
from types import SimpleNamespace

import pytest
import torch

try:
    import cloudpickle
    from mpi4py import MPI
except ImportError:  # the test is skipped below
    cloudpickle = MPI = None

if cloudpickle is not None:
    cloudpickle.register_pickle_by_value(sys.modules[__name__])
    MPI.pickle.__init__(cloudpickle.dumps, cloudpickle.loads, pickle.HIGHEST_PROTOCOL)

WORLD = 4
LATENT = 3584
M_CASES = list(range(1, 9))
CTAS = (4, 14, 28)
EMPTY_WORD = -(2**31)


def _supported() -> bool:
    if MPI is None or not torch.cuda.is_available() or torch.cuda.device_count() < WORLD:
        return False
    return torch.cuda.get_device_capability()[0] == 10


pytestmark = [
    pytest.mark.threadleak(enabled=False),
    pytest.mark.skipif(not _supported(), reason=f"needs {WORLD} SM100 GPUs with MNNVL and mpi4py"),
]


def _bits(t: torch.Tensor) -> torch.Tensor:
    return t.contiguous().view(torch.int16)


def _same(a: torch.Tensor, b: torch.Tensor) -> bool:
    return a.shape == b.shape and a.dtype == b.dtype and torch.equal(_bits(a), _bits(b))


def _digest(t: torch.Tensor) -> str:
    return hashlib.sha256(_bits(t).cpu().numpy().tobytes()).hexdigest()


def _context():
    os.environ.setdefault("TRTLLM_FORCE_MNNVL_AR", "1")
    comm = MPI.COMM_WORLD
    rank, world = comm.Get_rank(), comm.Get_size()
    gpus = torch.cuda.device_count()
    torch.cuda.set_device(rank % gpus)
    import tensorrt_llm  # noqa: F401
    from tensorrt_llm._torch.cute_dsl_kernels.k3_fused_moe import latent_op
    from tensorrt_llm._torch.distributed.ops import MNNVLAllReduce
    from tensorrt_llm.mapping import Mapping

    mapping = Mapping(world_size=world, rank=rank, gpus_per_node=gpus, tp_size=world)
    mnnvl = MNNVLAllReduce(mapping, torch.bfloat16)
    ex = latent_op.K3LatentExchange.create(mapping, fabric_handle=True)
    return SimpleNamespace(comm=comm, rank=rank, world=world, mnnvl=mnnvl, ex=ex, count=0)


def _allreduce(ctx, x):
    """MNNVLAllReduce sent one-shot (the order the reduce reproduces)."""
    from tensorrt_llm._torch.distributed import AllReduceParams

    return ctx.mnnvl(
        x, AllReduceParams(), one_shot_max_bytes=x.numel() * ctx.world * x.element_size()
    )


def _partial(m, seed, rank):
    gen = torch.Generator(device="cuda").manual_seed(seed * 131 + rank)
    x = (torch.randn(8, LATENT, generator=gen, device="cuda") * 0.5).bfloat16()
    x[:, ::97] = -0.0  # the pushes store these as +0.0
    return x[:m].contiguous()


def _push(ctx, x, half):
    """What the push-only k3_moe stores: the rows (-0.0 as +0.0) into slot [rank] of ``half`` of every rank's buffer,
    through the multicast mapping."""
    m = x.shape[0]
    words = x.clone().view(torch.int16)
    words[words == -32768] = 0
    rows = ctx.ex.mc.view(2, 8, ctx.world, LATENT // 2)
    rows[half, :m, ctx.rank].copy_(words.view(torch.int32).view(m, LATENT // 2))


def _reduce(ctx, m, ctas):
    out = torch.ops.trtllm.k3_latent_reduce(ctx.ex.uc, ctx.ex.flags, m, ctas)
    ctx.count += 1
    return out


def _call(ctx, x, ctas):
    _push(ctx, x, ctx.count & 1)
    return _reduce(ctx, x.shape[0], ctas)


def _state_ok(ctx) -> bool:
    """Every rank's last reduce done and nothing of the next call pushed yet: the whole buffer is empty, the count
    advanced, the arrival word cleared."""
    torch.cuda.synchronize()
    ctx.comm.Barrier()
    flags = ctx.ex.flags.tolist()
    ok = bool((ctx.ex.uc == EMPTY_WORD).all().item()) and flags[0] == ctx.count and flags[2] == 0
    ctx.comm.Barrier()
    return ok


def check_reduce(ctx):
    results = []
    for ctas in CTAS:
        for m in M_CASES:
            x = _partial(m, 1 + m + 10 * ctas, ctx.rank)
            ref = _allreduce(ctx, x)
            got = _call(ctx, x, ctas)
            state = _state_ok(ctx)
            again = [_call(ctx, x, ctas) for _ in range(2)]
            bad = x.clone()
            if ctx.rank == ctx.world - 1:
                bad[0, 1] += 1.0
            bad_out = _call(ctx, bad, ctas)
            row = dict(op="k3_latent_reduce", case=f"ctas{ctas}", M=m, exact=_same(got, ref),
                       det=all(_same(a, got) for a in again), control=not _same(bad_out, got),
                       ranks_agree=len(set(ctx.comm.allgather(_digest(got)))) == 1,
                       state=state and _state_ok(ctx))  # fmt: skip
            row["ok"] = all(ctx.comm.allgather(all(row[k] for k in ("exact", "det", "control", "ranks_agree",
                                                                        "state"))))  # fmt: skip
            results.append(row)
    return results


def check_graph(ctx):
    """Four push + reduce pairs (M 8, 3, 8, 1) captured in one CUDA graph, replayed three times with new partials; an
    even number of pairs, so every replay starts on the half the capture pushed into."""
    results = []
    ms = (8, 3, 8, 1)
    inputs = [_partial(m, 500 + i, ctx.rank) for i, m in enumerate(ms)]
    outs = [None] * len(ms)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        # eager first: every configuration compiled outside capture
        for i, m in enumerate(ms):
            outs[i] = _call(ctx, inputs[i], 0)
        torch.cuda.synchronize()
        base = ctx.count
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            for i, m in enumerate(ms):
                # the halves alternate, so the captured pairs follow the eager calls' parity
                _push(ctx, inputs[i], (base + i) & 1)
                outs[i] = torch.ops.trtllm.k3_latent_reduce(ctx.ex.uc, ctx.ex.flags, m, 0)
    torch.cuda.synchronize()
    ctx.comm.Barrier()
    for rep in range(3):
        fresh = [_partial(m, 900 + 10 * rep + i, ctx.rank) for i, m in enumerate(ms)]
        for i in range(len(ms)):
            inputs[i].copy_(fresh[i])
        refs = [_allreduce(ctx, x) for x in fresh]
        torch.cuda.synchronize()
        ctx.comm.Barrier()
        graph.replay()
        ctx.count += len(ms)
        torch.cuda.synchronize()
        exact = all(_same(o, r) for o, r in zip(outs, refs))
        row = dict(
            op="k3_latent_reduce", case=f"graph_replay{rep}", M=8, exact=exact, state=_state_ok(ctx)
        )
        row["ok"] = all(ctx.comm.allgather(exact and row["state"]))
        results.append(row)
    del graph
    return results


CHECKS = {"reduce": check_reduce, "graph": check_graph}


def _run_checks(names):
    try:
        ctx = _context()
        with torch.inference_mode():
            return [row for name in names for row in CHECKS[name](ctx)]
    except Exception:
        traceback.print_exc()
        raise


def _report(rows):
    for row in rows:
        fields = " ".join(f"{k}={v}" for k, v in row.items() if k not in ("op", "case", "M"))
        print(f"OPCHECK op={row['op']} case={row['case']} M={row['M']} {fields}", flush=True)


@pytest.mark.parametrize("mpi_pool_executor", [WORLD], indirect=True)
def test_k3_latent_reduce(mpi_pool_executor):
    per_rank = list(mpi_pool_executor.map(_run_checks, [list(CHECKS)] * WORLD))
    _report(per_rank[0])
    assert all(row["ok"] for rows in per_rank for row in rows)


def test_latent_exchange_refuses_graph_capture():
    """The exchange is created collectively (an MNNVL multicast allocation over the TP group): creating it under
    CUDA-graph capture raises instead of entering the collective. One process, a group of one."""
    import tensorrt_llm  # noqa: F401
    from tensorrt_llm._torch.cute_dsl_kernels.k3_fused_moe import latent_op
    from tensorrt_llm.mapping import Mapping

    mapping = Mapping(world_size=1, rank=0, tp_size=1)
    graph, stream = torch.cuda.CUDAGraph(), torch.cuda.Stream()
    with torch.cuda.graph(graph, stream=stream):
        with pytest.raises(RuntimeError, match="outside CUDA-graph capture"):
            latent_op.K3LatentExchange.create(mapping)


def main() -> int:
    names = sys.argv[1:] or list(CHECKS)
    rows = _run_checks(names)
    if MPI.COMM_WORLD.Get_rank() == 0:
        _report(rows)
        print("PASS" if all(r["ok"] for r in rows) else "FAIL", flush=True)
    return 0 if all(r["ok"] for r in rows) else 1


if __name__ == "__main__":
    sys.exit(main())
