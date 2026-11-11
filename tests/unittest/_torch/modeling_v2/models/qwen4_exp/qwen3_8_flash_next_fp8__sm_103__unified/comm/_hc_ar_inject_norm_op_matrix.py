# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Certification matrix for the FP8 target's fused TP sum + injection at 2 and 4 ranks: `hc_out`
and `normed` bitwise the IPC all-reduce then the Triton injection, with and without the MoE
combine, interleaved with plain all-reduces, and in a CUDA graph chain."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch

assert torch.cuda.is_available(), "the fused TP sum + injection requires CUDA devices"

DEADLINE_S = 1500
HC, HIDDEN = 4, 2560
WIDTH = HC * HIDDEN
EPS = 1e-6
#: Decode-site row counts on both sides of the switch to 2-warp programs at 512 rows: the entry's
#: CELLS at this world, set by `_run_one_rank`.
ROWS: tuple = ()
_MODELING_V2_TESTS = Path(__file__).resolve().parents[4]

modeling = fused_mod = tp_sum = MPI_RANK = WS = None
RANK = WORLD = 0


def _gen(seed: int) -> torch.Generator:
    return torch.Generator(device="cuda").manual_seed(seed)


def _inputs(rows: int, seed: int, combine: bool):
    g = _gen(seed)
    hc = (torch.randn(rows, WIDTH, generator=g, device="cuda") * 2.0).to(torch.bfloat16)
    # The injection weights as the mixer leaves them: a column slice of a wider row.
    inject = (torch.randn(rows, 336, generator=g, device="cuda") * 0.5 + 1.0).to(torch.bfloat16)[
        :, 320:324
    ]
    norm = (torch.randn(WIDTH, generator=g, device="cuda") * 0.477).to(torch.bfloat16)
    shared = gate = None
    if combine:
        shared = (torch.randn(rows, HIDDEN, generator=g, device="cuda") * 0.5).to(torch.bfloat16)
        gate = torch.randn(rows, 1288, generator=g, device="cuda").to(torch.bfloat16)[:, 1280:1281]
    pg = _gen(seed * 977 + RANK + 1)
    partial = (torch.randn(rows, HIDDEN, generator=pg, device="cuda") * 0.25).to(torch.bfloat16)
    # Signed zeros: a (+0, -0) bf16 pair is the -0.0f word, the not-arrived marker; every rank
    # -0.0 in columns 7..16 (the sanitized push makes their sum +0.0, as the all-reduce does).
    partial[:, :7] = 0.0
    partial[:, 7:17] = -0.0
    # A few large rows (residual-stream outliers) and an all-zero residual row (rstd of eps).
    if rows > 40:
        hc[3] *= 300.0
        hc[5].zero_()
    return hc, inject, norm, shared, gate, partial


def _warps(rows):
    return 2 if rows >= modeling._HC_INJECT_FEW_WARPS_MIN_ROWS else modeling._HC_INJECT_WARPS


def _plain(hc, inject, norm, shared, gate, partial):
    blk = tp_sum(partial)
    return modeling._hc_inject_norm_launch(
        hc, blk, inject, norm, EPS, HC, shared, gate, _warps(hc.shape[0])
    )


def _fused(hc, inject, norm, shared, gate, partial):
    assert fused_mod.fits(hc, partial, inject, shared, gate)
    return fused_mod.hc_allreduce_inject_norm(
        partial, WS, RANK, WORLD, hc, inject, norm, EPS, shared, gate, _warps(hc.shape[0])
    )


def _same(a, b, what: str) -> None:
    for name, x, y in zip(("hc_out", "normed"), a, b):
        if not torch.equal(x, y):
            idx = (x != y).nonzero()[:8].tolist()
            vals = [(i, float(x[i[0], i[1]]), float(y[i[0], i[1]])) for i in idx]
            raise AssertionError(
                f"[rank {RANK}] {what}: {name} differs ({(x != y).sum().item()} elements), first {vals}"
            )


def check_fused_is_bitwise_the_all_reduce_then_the_injection() -> None:
    n = 0
    for rows in ROWS:
        for combine in (False, True):
            for rep in range(2):
                args = _inputs(rows, 7000 + 10 * rows + 2 * combine + rep, combine)
                want = _plain(*args)
                got = _fused(*args)
                again = _plain(*args)
                torch.cuda.synchronize()
                what = f"{rows} rows combine {combine} rep {rep}"
                _same(got, want, what)
                _same(again, want, what + " (plain after fused)")
                n += 1
        MPI_RANK.barrier()
    if RANK == 0:
        print(f"[world {WORLD}] {n} fused calls bitwise the all-reduce + injection", flush=True)


def check_cuda_graph_chain_replays_bitwise() -> None:
    """A captured chain of mixed plain and fused sites (an odd call count, so replays start on
    every slot), replayed with fresh partials, is bitwise the plain chain run eagerly."""
    for rows in (96, 192, 480):
        n_sites = 7
        g = _gen(60 + rows)
        hc0 = (torch.randn(rows, WIDTH, generator=g, device="cuda") * 2.0).to(torch.bfloat16)
        inj = (torch.randn(rows, HC, generator=g, device="cuda") * 0.5 + 1.0).to(torch.bfloat16)
        norms = [
            (torch.randn(WIDTH, generator=g, device="cuda") * 0.4).to(torch.bfloat16)
            for _ in range(n_sites)
        ]
        shared = (torch.randn(rows, HIDDEN, generator=g, device="cuda") * 0.5).to(torch.bfloat16)
        gate = torch.randn(rows, 1, generator=g, device="cuda").to(torch.bfloat16)
        parts = [
            torch.empty(rows, HIDDEN, dtype=torch.bfloat16, device="cuda") for _ in range(n_sites)
        ]

        def chain(kinds):
            hc, outs = hc0, []
            for k in range(n_sites):
                fn = _fused if kinds[k] else _plain
                sh, gt = (shared, gate) if k % 2 else (None, None)
                hc, normed = fn(hc, inj, norms[k], sh, gt, parts[k])
                outs.append(normed)
            return hc, outs

        kinds = [k % 3 != 1 for k in range(n_sites)]
        for p in parts:
            p.copy_(
                torch.randn(rows, HIDDEN, generator=_gen(1 + RANK), device="cuda").to(
                    torch.bfloat16
                )
            )
        chain(kinds)  # warm the compiles
        torch.cuda.synchronize()
        MPI_RANK.barrier()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            g_hc, g_outs = chain(kinds)
        torch.cuda.synchronize()
        MPI_RANK.barrier()
        for rep in range(5):
            pg = _gen(1000 * rep + 31 * rows + RANK)
            for p in parts:
                p.copy_(
                    (torch.randn(rows, HIDDEN, generator=pg, device="cuda") * 0.3).to(
                        torch.bfloat16
                    )
                )
            graph.replay()
            torch.cuda.synchronize()
            got = (g_hc.clone(), [o.clone() for o in g_outs])
            MPI_RANK.barrier()
            want = chain([False] * n_sites)
            torch.cuda.synchronize()
            assert torch.equal(got[0], want[0]), f"{rows} rows replay {rep}: hc differs"
            for k in range(n_sites):
                assert torch.equal(got[1][k], want[1][k]), (
                    f"{rows} rows replay {rep}: site {k} normed differs"
                )
            MPI_RANK.barrier()
    if RANK == 0:
        print(f"[world {WORLD}] CUDA graph chains replay bitwise", flush=True)


CHECKS = (
    check_fused_is_bitwise_the_all_reduce_then_the_injection,
    check_cuda_graph_chain_replays_bitwise,
)


def _run_one_rank() -> int:
    global modeling, fused_mod, tp_sum, MPI_RANK, RANK, WORLD, WS, ROWS
    os.environ.pop("TLLM_DISABLE_MPI", None)
    os.environ.setdefault("TRTLLM_ENABLE_PDL", "1")
    sys.path.insert(0, str(_MODELING_V2_TESTS / "comm"))
    import _lockstep

    MPI_RANK = _lockstep.Rank(_lockstep.parse_args(sys.argv[1:]))
    RANK, WORLD = MPI_RANK.rank, MPI_RANK.world
    assert WORLD in (2, 4), "the fused TP sum + injection is certified at 2 and 4 ranks"
    from tensorrt_llm._torch._experimental.modeling_v2.catalog.comm import allreduce as entry
    from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_fp8__sm_103__unified import (  # noqa: E501
        modeling as _modeling,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_fp8__sm_103__unified.ops.comm import (  # noqa: E501
        hc_allreduce_inject_norm__tp_sum as _fused_mod,
    )
    from tensorrt_llm.functional import AllReduceFusionOp

    modeling, fused_mod = _modeling, _fused_mod
    ROWS = tuple(
        c.spec["rows"] for c in fused_mod.hc_allreduce_inject_norm.CELLS if c.spec["world"] == WORLD
    )
    assert ROWS, f"no cell is certified at world size {WORLD}"
    group = entry.allreduce_group_args(MPI_RANK.mapping)
    WS = group["workspace"]

    def _tp_sum(x):
        return entry.allreduce(
            x,
            None,
            None,
            scale=None,
            bias=None,
            op=int(AllReduceFusionOp.NONE),
            eps=EPS,
            trigger_completion_at_end=False,
            **group,
        )

    tp_sum = _tp_sum
    code = _lockstep.run_checks(MPI_RANK, list(CHECKS))
    torch.cuda.synchronize()
    sys.stdout.flush()
    sys.stderr.flush()
    import torch.distributed as dist

    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
    MPI_RANK.MPI.Finalize()
    os._exit(code)


if __name__ == "__main__":
    sys.path.insert(0, str(_MODELING_V2_TESTS / "comm"))
    import _lockstep

    args = _lockstep.parse_args(sys.argv[1:])
    if args.rank_worker:
        sys.exit(_run_one_rank())
    import _launch  # beside this file

    world = args.world_size or len(_launch.visible_devices())
    assert world in (2, 4), "the fused sum runs at 2 or 4 ranks"
    worker = [_lockstep.WORKER_FLAG, "--world-size", str(world)]
    worker += ["--fabric-handle", args.fabric_handle]
    _launch.mpirun_ranks(__file__, world, worker, DEADLINE_S)
    print("OK: fused TP sum + injection")
