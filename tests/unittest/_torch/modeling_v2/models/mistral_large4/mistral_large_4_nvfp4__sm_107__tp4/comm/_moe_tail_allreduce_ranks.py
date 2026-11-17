# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Rank job for the sm_107 tp4 target's MoE-tail all-reduces, against the ops they replace.

The decode target folds two ops of each MoE layer into the Lamport one-shot collectives
around it, and each fold must compute the same bits as the separate op:

- `allreduce_add_rmsnorm_nvfp4` against `cute_dsl_allreduce_add_rmsnorm` followed by
  `fp4_quantize` of its output: the norm output, the updated residual, the e2m1 codes and
  the e4m3 scales;
- `allreduce_add2_rmsnorm(routed, shared)` against `cute_dsl_allreduce_add_rmsnorm` over
  `torch.add(routed, shared)`: the norm output and the updated residual.

Every comparison is of bits. The partials carry outliers and signed zeros, the global scales
span small to saturating, and the row counts run from one row to `ONE_SHOT_ROWS`. Random scales
almost never sit where the order of the fp32 scale arithmetic decides an e4m3 rounding, so the
quantization is also driven at global scales that put a block's scale on a rounding midpoint
and a few fp32 ulp to either side. Both pairs also run inside one captured CUDA graph, replayed
on fresh inputs, as the decode target runs them. A call above `ONE_SHOT_ROWS` must be refused
before it launches.

A collective cannot be exercised in one process. Run plainly, this file is its own launcher:
it spawns one rank per device named in CUDA_VISIBLE_DEVICES (four) under a deadline it
enforces by killing every rank's session, since a Lamport collective whose ranks disagree
hangs rather than raising. Each rank collects its mismatches and the ranks agree on the
outcome only after the last collective, so no rank leaves the sequence early.

    CUDA_VISIBLE_DEVICES=0,1,2,3 python _moe_tail_allreduce_ranks.py

The collected entry point is `test_modeling_v2_moe_tail_allreduce.py` beside this file. This
file imports only torch and the standard library at module scope: the test executes it to read
`DEADLINE_S`.
"""

import math
import os
import signal
import socket
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import List

import torch
import torch.distributed as dist

_WORKER_FLAG = "--rank-worker"
#: One workspace of five CuTe DSL compiles per rank, then a few hundred calls of at most
#: `ONE_SHOT_ROWS` rows.
DEADLINE_S = 600
WORLD_SIZE = 4
HIDDEN = 6144
#: Mistral Large 4's `rms_norm_eps`.
EPS = 1e-5
SF_VEC = 16
ROWS = (1, 2, 3, 4, 8, 16, 24, 32, 64, 128, 1024)
GRAPH_ROWS = (4, 32)
GRAPH_REPLAYS = 3
#: Small, typical, and large enough that most blocks' scales saturate e4m3.
GLOBAL_SCALES = (0.5, 134.4, 448.0 * 6.0 / 0.37, 2000.0)
PARTIAL_STDS = (0.02, 0.3, 1.0, 4.0)
#: Rows whose first block's scale is put on an e4m3 rounding midpoint, one at a time, and the
#: fp32 global scales tried on each side of each.
BOUNDARY_ROWS = 32
BOUNDARY_ULPS = 3


def _partial(rows: int, seed: int, std: float) -> torch.Tensor:
    """A rank's bf16 partial: Gaussian, with one element in a thousand 50x out, and +0.0 and
    -0.0 one in a thousand each."""
    gen = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(rows, HIDDEN, generator=gen, device="cuda") * std
    pick = torch.rand(rows, HIDDEN, generator=gen, device="cuda")
    x = torch.where(pick < 0.001, x * 50, x)
    x = torch.where(pick > 0.999, torch.zeros_like(x), x)
    x = torch.where((pick > 0.998) & (pick <= 0.999), torch.full_like(x, -0.0), x)
    return x.to(torch.bfloat16)


def _shared(rows: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """The residual and the norm weight, the same on every rank."""
    gen = torch.Generator(device="cuda").manual_seed(seed)
    residual = (torch.randn(rows, HIDDEN, generator=gen, device="cuda") * 3).to(torch.bfloat16)
    weight = (1 + 0.2 * torch.randn(HIDDEN, generator=gen, device="cuda")).to(torch.bfloat16)
    return residual, weight


def _bits(t: torch.Tensor) -> torch.Tensor:
    if t.element_size() == 2:
        return t.view(torch.int16)
    return t.view(torch.uint8)


class _Mismatches:
    def __init__(self) -> None:
        self.found: List[str] = []

    def compare(self, where: str, **pairs: tuple[torch.Tensor, torch.Tensor]) -> None:
        for name, (got, want) in pairs.items():
            got, want = _bits(got).reshape(-1), _bits(want).reshape(-1)
            if got.shape != want.shape:
                self.found.append(
                    f"{where}: {name} shape {tuple(got.shape)} != {tuple(want.shape)}"
                )
                continue
            differ = int((got != want).sum())
            if differ:
                self.found.append(f"{where}: {name} differs in {differ} of {got.numel()}")


def _run_rank() -> None:
    os.environ.setdefault("TLLM_DISABLE_MPI", "1")
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    assert world == WORLD_SIZE, f"this job runs {WORLD_SIZE} ranks, got {world}"
    torch.cuda.set_device(rank)

    from tensorrt_llm._torch._experimental.modeling_v2.catalog.comm import _lamport
    from tensorrt_llm._torch._experimental.modeling_v2.catalog.comm.cute_dsl_allreduce_add_rmsnorm import (  # noqa: E501
        cute_dsl_allreduce_add_rmsnorm,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.catalog.quantization.fp4_quantize import (
        fp4_quantize,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.models.mistral_large4.mistral_large_4_nvfp4__sm_107__tp4.ops.comm.moe_tail_allreduce import (  # noqa: E501
        ONE_SHOT_ROWS,
        allreduce_add2_rmsnorm,
        allreduce_add_rmsnorm_nvfp4,
    )

    # Room past the one-shot limit, so the guard below is the one-shot kernels' own.
    ws = _lamport.build_workspace(
        HIDDEN, EPS, ONE_SHOT_ROWS + 64, rank, world, add_second_input=True
    )
    torch.cuda.synchronize()
    bad = _Mismatches()

    def separate_quant(partial, residual, weight, gscale):
        norm = cute_dsl_allreduce_add_rmsnorm(
            partial, residual, weight, torch.empty_like(partial), ws
        )
        codes, scales = fp4_quantize(norm, gscale, SF_VEC, False, False)
        return norm, codes, scales

    def separate_add(routed, shared, residual, weight):
        mlp = torch.add(routed, shared)
        return cute_dsl_allreduce_add_rmsnorm(mlp, residual, weight, torch.empty_like(mlp), ws)

    def folded_add(routed, shared, residual, weight):
        return allreduce_add2_rmsnorm(
            routed, shared, residual, weight, torch.empty_like(routed), ws
        )

    # Eager, row count by row count.
    for i, rows in enumerate((*ROWS, ONE_SHOT_ROWS)):
        for seed in range(len(GLOBAL_SCALES)):
            base = 1000 * i + 100 * seed
            std = PARTIAL_STDS[seed]
            partial = _partial(rows, base + 10 * rank, std)
            routed = _partial(rows, base + 10 * rank + 1, std)
            shared = _partial(rows, base + 10 * rank + 2, 0.3 * std)
            residual, weight = _shared(rows, base + 99)
            gscale = torch.tensor([GLOBAL_SCALES[seed]], dtype=torch.float32, device="cuda")
            where = f"rows={rows} seed={seed}"

            res_a, res_b = residual.clone(), residual.clone()
            norm_a, codes_a, scales_a = separate_quant(partial, res_a, weight, gscale)
            norm_b, codes_b, scales_b = allreduce_add_rmsnorm_nvfp4(
                partial, res_b, weight, gscale, ws
            )
            bad.compare(
                f"fused quant {where}",
                norm=(norm_b, norm_a),
                residual=(res_b, res_a),
                codes=(codes_b, codes_a),
                scales=(scales_b, scales_a),
            )
            res_a, res_b = residual.clone(), residual.clone()
            out_a = separate_add(routed, shared, res_a, weight)
            out_b = folded_add(routed, shared, res_b, weight)
            bad.compare(f"fused add {where}", norm=(out_b, out_a), residual=(res_b, res_a))
    torch.cuda.synchronize()

    # Global scales that put one block's scale on an e4m3 rounding midpoint, and the fp32 scales
    # a few ulp to either side: there the order of the fp32 scale arithmetic decides the
    # rounding, which random scales almost never reach. The partials are zero, so every call
    # writes the same norm rows, and a first call gives their block maxima.
    rows = BOUNDARY_ROWS
    zero = torch.zeros(rows, HIDDEN, dtype=torch.bfloat16, device="cuda")
    residual, weight = _shared(rows, 4242)
    norm = cute_dsl_allreduce_add_rmsnorm(
        zero, residual.clone(), weight, torch.empty_like(zero), ws
    )
    block_max = norm.float().abs().view(rows, HIDDEN // SF_VEC, SF_VEC).amax(-1)[:, 0].tolist()
    for row, vec_max in enumerate(block_max):
        scale = GLOBAL_SCALES[1] * vec_max / 6.0
        step = 2.0 ** (math.floor(math.log2(scale)) - 3)
        midpoint = (math.floor(scale / step) + 0.5) * step
        center = torch.tensor([6.0 * midpoint / vec_max], dtype=torch.float32, device="cuda")
        for ulps in range(-BOUNDARY_ULPS, BOUNDARY_ULPS + 1):
            gscale = center
            toward = torch.full_like(center, math.copysign(math.inf, ulps))
            for _ in range(abs(ulps)):
                gscale = torch.nextafter(gscale, toward)
            res_a, res_b = residual.clone(), residual.clone()
            norm_a, codes_a, scales_a = separate_quant(zero, res_a, weight, gscale)
            norm_b, codes_b, scales_b = allreduce_add_rmsnorm_nvfp4(zero, res_b, weight, gscale, ws)
            bad.compare(
                f"fused quant on a scale midpoint, row={row} ulps={ulps}",
                norm=(norm_b, norm_a),
                codes=(codes_b, codes_a),
                scales=(scales_b, scales_a),
            )
    torch.cuda.synchronize()

    # Both pairs in one captured graph per row count, replayed on fresh inputs.
    for rows in GRAPH_ROWS:
        names = ("partial", "routed", "shared", "res_qa", "res_qb", "res_aa", "res_ab")
        static = {k: torch.zeros(rows, HIDDEN, dtype=torch.bfloat16, device="cuda") for k in names}
        weight = torch.ones(HIDDEN, dtype=torch.bfloat16, device="cuda")
        gscale = torch.tensor([GLOBAL_SCALES[1]], dtype=torch.float32, device="cuda")
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.stream(stream):
            with torch.cuda.graph(graph, stream=stream):
                quant_a = separate_quant(static["partial"], static["res_qa"], weight, gscale)
                quant_b = allreduce_add_rmsnorm_nvfp4(
                    static["partial"], static["res_qb"], weight, gscale, ws
                )
                add_a = separate_add(static["routed"], static["shared"], static["res_aa"], weight)
                add_b = folded_add(static["routed"], static["shared"], static["res_ab"], weight)
        torch.cuda.current_stream().wait_stream(stream)
        for replay in range(GRAPH_REPLAYS):
            base = 100000 + 1000 * rows + 100 * replay
            static["partial"].copy_(_partial(rows, base + 10 * rank, 1.0))
            static["routed"].copy_(_partial(rows, base + 10 * rank + 1, 1.0))
            static["shared"].copy_(_partial(rows, base + 10 * rank + 2, 0.3))
            residual, _ = _shared(rows, base + 99)
            for k in ("res_qa", "res_qb", "res_aa", "res_ab"):
                static[k].copy_(residual)
            graph.replay()
            torch.cuda.synchronize()
            where = f"graph rows={rows} replay={replay}"
            bad.compare(
                f"fused quant {where}",
                norm=(quant_b[0], quant_a[0]),
                residual=(static["res_qb"], static["res_qa"]),
                codes=(quant_b[1], quant_a[1]),
                scales=(quant_b[2], quant_a[2]),
            )
            bad.compare(
                f"fused add {where}",
                norm=(add_b, add_a),
                residual=(static["res_ab"], static["res_aa"]),
            )

    # Above the one-shot limit, both refuse before launching anything (rank-local).
    over = torch.zeros(ONE_SHOT_ROWS + 1, HIDDEN, dtype=torch.bfloat16, device="cuda")
    for name, call in (
        ("fused quant", lambda: allreduce_add_rmsnorm_nvfp4(over, over, weight, gscale, ws)),
        ("fused add", lambda: folded_add(over, over, over, weight)),
    ):
        try:
            call()
            bad.found.append(f"{name} took {ONE_SHOT_ROWS + 1} rows")
        except AssertionError as exc:
            if "one-shot" not in str(exc):
                bad.found.append(
                    f"{name} refused {ONE_SHOT_ROWS + 1} rows for another reason: {exc}"
                )

    # Agree on the outcome after the last collective.
    torch.cuda.synchronize()
    failed = torch.tensor([len(bad.found)], dtype=torch.int32, device="cuda")
    dist.all_reduce(failed)
    for line in bad.found[:20]:
        print(f"[rank {rank}] MISMATCH {line}", flush=True)
    if rank == 0:
        print(f"[rank 0] {int(failed)} mismatch(es) over {world} ranks", flush=True)
    sys.stdout.flush()
    # Exit without tearing the symmetric memory down: these are throwaway processes.
    os._exit(0 if int(failed) == 0 else 13)


def _spawn_ranks() -> None:
    """Start one rank per claimed device; kill every rank's session on the first failure or at
    the deadline."""
    visible = [d for d in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if d.strip()]
    assert len(visible) == WORLD_SIZE, (
        f"set CUDA_VISIBLE_DEVICES to the {WORLD_SIZE} devices this run owns, got {visible}"
    )
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    env = dict(
        os.environ,
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(port),
        WORLD_SIZE=str(WORLD_SIZE),
        TLLM_DISABLE_MPI="1",
    )
    # A launching srun step's PMIx/PMI/OMPI namespace would register the ranks as its
    # participants; they rendezvous through torch.distributed instead.
    env = {k: v for k, v in env.items() if not k.startswith(("PMIX_", "PMI_", "OMPI_", "SLURM_"))}
    children = [
        subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()), _WORKER_FLAG],
            env=dict(env, RANK=str(r)),
            start_new_session=True,
        )
        for r in range(WORLD_SIZE)
    ]
    deadline = time.monotonic() + DEADLINE_S
    live = set(range(WORLD_SIZE))
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
                failure = f"rank {r} exited {code}"
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
        raise AssertionError(f"the {WORLD_SIZE}-rank run failed: {failure}")


if __name__ == "__main__":
    if _WORKER_FLAG in sys.argv:
        try:
            _run_rank()
        except BaseException:
            traceback.print_exc()
            sys.stdout.flush()
            sys.stderr.flush()
            # A rank that leaves early wedges its siblings; the launcher kills them on this.
            os._exit(13)
    _spawn_ranks()
    print("OK")
