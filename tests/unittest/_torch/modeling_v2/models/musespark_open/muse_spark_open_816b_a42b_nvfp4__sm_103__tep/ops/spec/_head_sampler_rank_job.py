# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""8-rank (two GB300 trays) check and chain timing of the ``lm_head_inbox`` head gather.

Run one process per rank under MPI (eight ranks, four per node), e.g.

    python3 _head_sampler_rank_job.py --check --bench --out result.json

Every rank multiplies the same rows by its own random ``[25256, 8192]`` BF16
LM-head shard with the split-K tactic the autotuner picks for it at this shape
(``SERVED_TACTICS``) and writes the result through the multicast address into
its inbox slot (``HeadGatherWorkspace.slot``); kernel A gathers, kernel B
samples.

``--check`` compares, per call: the gathered BF16 row with the rank-order
concatenation of every rank's local GEMM output of the same kernel (exchanged
over MPI on the host), bit for bit, on every rank; the tokens with the
built-in soft cap + ``fused_sample_from_logits`` on that row; the inbox reset
(every word back to the sentinel) and the status word. Calls cover eager, one
rank delayed before its GEMM (each rank in turn), slots published by a copy
holding negative zeros (instead of a GEMM), and CUDA-graph replay with PDL
on fresh inputs.

``--steps`` checks the inbox's cross-call order with nothing but device-side
collectives between calls (ownership item 3 of ``HeadGatherWorkspace``): CUDA
graphs of ``--step-count`` back-to-back head steps (GEMM into the slot,
kernel A, kernel B, copies of the gathered row and tokens, then one small
NCCL all-gather -- the only thing between two head calls), with no delay,
with rank ``r`` sleeping before its GEMM on odd steps, and with rank ``r``
sleeping between its GEMM and its kernel A on odd steps (a late consumer,
so a late reset), for every ``r``; each graph is replayed ``--step-replays``
times on fresh inputs and every step is verified as in ``--check``.

``--producers`` runs every runner and tactic the ``lm_head_inbox`` LM-head call can pick at
this shape (cuBLAS, every split-K tactic, every direct tactic: the same on
every rank per call) into the multicast slot, then kernel A, and compares the
gathered row with the rank-order concatenation of the same producer's local
outputs (and the tokens, on the first call of each), bit for bit.

``--bench`` times, in CUDA graphs of ``--repeat`` head steps, the built-in
chain (GEMM, NCCL all-gather, ``.float()`` and the four-kernel soft cap,
``fused_sample_from_logits``) against the ``lm_head_inbox`` chain (GEMM into the slot,
kernel A, kernel B), each followed by the ``fused_accept`` accept kernel, in alternating
order for ``--rounds`` rounds; per-step time is the max over ranks.
"""

from __future__ import annotations

import argparse
import json
import socket
import statistics
import sys
import time

import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.ops.spec import (  # noqa: E501
    dflash_accept as _accept,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.ops.spec import (  # noqa: E501
    head_sampler as hs,
)
from tensorrt_llm._utils import mpi_comm
from tensorrt_llm.mapping import Mapping

MULTIPLIER = 0.1767766952966369
SOFT_CAP = 20.0
SHARD = 25256
HIDDEN = 8192
ROWS = 8
#: The LM head's split-K tactic at 8 rows per rank, as the autotuner picks it for this shape.
SERVED_TACTICS = {0: (64, 16, 1, 11), 3: (128, 8, 2, 6), 7: (64, 8, 1, 11)}
DEFAULT_TACTIC = (64, 8, 1, 12)


def log(*a):
    print(f"[rank {RANK}]", *a, flush=True)


COMM = mpi_comm()
WORLD = COMM.Get_size()
RANK = COMM.Get_rank()


def agree(ok: bool) -> bool:
    return COMM.allreduce(int(ok)) == WORLD


def max_over_ranks(x: float) -> float:
    from mpi4py import MPI

    return COMM.allreduce(x, op=MPI.MAX)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--steps", action="store_true")
    ap.add_argument("--producers", action="store_true")
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--calls", type=int, default=64)
    ap.add_argument("--graph-calls", type=int, default=64)
    ap.add_argument("--step-count", type=int, default=24)
    ap.add_argument("--step-replays", type=int, default=8)
    ap.add_argument("--producer-calls", type=int, default=2)
    ap.add_argument("--repeat", type=int, default=20)
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    torch.cuda.set_device(RANK % torch.cuda.device_count())
    dev = torch.device("cuda", torch.cuda.current_device())
    mapping = Mapping(world_size=WORLD, rank=RANK, gpus_per_node=4, tp_size=WORLD)
    from tensorrt_llm._torch.flashinfer_utils import get_env_enable_pdl
    from tensorrt_llm._torch.modules.low_m_gemm import _SplitKGemmRunner

    pdl = get_env_enable_pdl()
    lib = hs.load_library()
    log("host", socket.gethostname(), "lib", lib, "pdl", pdl)
    ws = hs.HeadGatherWorkspace(mapping, SHARD)
    tactic = SERVED_TACTICS.get(RANK, DEFAULT_TACTIC)
    runner = _SplitKGemmRunner(has_bias=False, pdl=pdl)
    g = torch.Generator(device=dev).manual_seed(1000 + RANK)
    weight = (torch.randn(SHARD, HIDDEN, generator=g, device=dev) * 0.03).to(torch.bfloat16)
    weight_t = weight.t()
    x = torch.empty(ROWS, HIDDEN, dtype=torch.bfloat16, device=dev)
    local = torch.empty(ROWS, SHARD, dtype=torch.bfloat16, device=dev)
    temps = torch.ones(ROWS, device=dev)
    top_ks = torch.zeros(ROWS, dtype=torch.int32, device=dev)
    top_ps = torch.ones(ROWS, device=dev)
    min_ps = torch.zeros(ROWS, device=dev)
    seed = torch.zeros(ROWS, dtype=torch.int64, device=dev)
    offset = torch.zeros(ROWS, dtype=torch.int64, device=dev)
    result = {"world": WORLD, "tactics": {}}
    result["tactics"] = {str(r): list(SERVED_TACTICS.get(r, DEFAULT_TACTIC)) for r in range(WORLD)}

    def fresh_inputs(call: int):
        gc = torch.Generator(device=dev).manual_seed(77 + call)  # same on every rank
        x.copy_((torch.randn(ROWS, HIDDEN, generator=gc, device=dev) * 2.0).to(torch.bfloat16))
        gs = torch.Generator().manual_seed(99 + call)
        seed.copy_(torch.randint(0, 2**62, (ROWS,), generator=gs, dtype=torch.int64))
        offset.copy_(torch.randint(0, 2**40, (ROWS,), generator=gs, dtype=torch.int64))

    def host_row(shard_out: torch.Tensor) -> torch.Tensor:
        """Rank-order concatenation of every rank's ``[ROWS, SHARD]`` over MPI (host)."""
        mine = shard_out.view(torch.int16).cpu().numpy()
        parts = COMM.allgather(mine)
        row = torch.cat([torch.from_numpy(p) for p in parts], dim=1)
        return row.view(torch.bfloat16).to(dev)

    def lm_head_inbox_step():
        runner([x, weight_t], tactic=tactic, out=ws.slot(ROWS))
        return hs.gather_split_sample(
            ws.inbox,
            ws.gathered[:ROWS],
            temps,
            top_ks,
            top_ps,
            min_ps,
            seed,
            offset,
            MULTIPLIER,
            SOFT_CAP,
            ws.states,
            ws.status,
            SHARD,
            hs.MAX_ROWS,
            True,
            pdl,
        )

    def verify(tokens, published_row, tag, stats):
        torch.cuda.synchronize()
        gathered_bad = int(
            (ws.gathered[:ROWS].view(torch.int16) != published_row.view(torch.int16)).sum()
        )
        ref = torch.ops.trtllm.fused_sample_from_logits(
            hs.reference_capped(published_row, MULTIPLIER, SOFT_CAP),
            temps,
            top_ks,
            top_ps,
            min_ps,
            seed,
            offset,
        )
        token_bad = int((tokens != ref).sum())
        not_reset = int((ws.inbox != -1).sum())
        s = stats.setdefault(
            tag, {"calls": 0, "gathered_bad": 0, "token_bad": 0, "not_reset": 0, "draws": 0}
        )
        s["calls"] += 1
        s["draws"] += ROWS
        s["gathered_bad"] += gathered_bad
        s["token_bad"] += token_bad
        s["not_reset"] += not_reset

    if args.check:
        stats = {}
        # Warm up the GEMM (compile) on the local buffer, then on the slot.
        runner([x, weight_t], tactic=tactic, out=local)
        torch.cuda.synchronize()
        COMM.Barrier()
        for call in range(args.calls):
            fresh_inputs(call)
            runner([x, weight_t], tactic=tactic, out=local)
            torch.cuda.synchronize()
            row = host_row(local)
            COMM.Barrier()
            mode = ("eager", "delay", "zeros")[call % 3]
            if mode == "delay" and RANK == (call // 3) % WORLD:
                torch.cuda._sleep(2_000_000)  # about 1 ms on this rank before it publishes
            if mode == "zeros":
                # Publish a copy holding +0 / -0 / NaN-free values instead of a GEMM.
                gz = torch.Generator(device=dev).manual_seed(500 + call * WORLD + RANK)
                z = (torch.randn(ROWS, SHARD, generator=gz, device=dev) * 30).to(torch.bfloat16)
                pick = torch.rand(ROWS, SHARD, generator=gz, device=dev)
                z[pick < 0.05] = 0.0
                z[(pick >= 0.05) & (pick < 0.10)] = -0.0
                torch.cuda.synchronize()
                row = host_row(z)
                COMM.Barrier()
                ws.slot(ROWS).copy_(z)
                tokens = hs.gather_split_sample(
                    ws.inbox,
                    ws.gathered[:ROWS],
                    temps,
                    top_ks,
                    top_ps,
                    min_ps,
                    seed,
                    offset,
                    MULTIPLIER,
                    SOFT_CAP,
                    ws.states,
                    ws.status,
                    SHARD,
                    hs.MAX_ROWS,
                    True,
                    pdl,
                )
            else:
                tokens = lm_head_inbox_step()
            verify(tokens, row, mode, stats)
            COMM.Barrier()
        # CUDA graph with PDL, replayed on fresh inputs.
        fresh_inputs(10_000)
        lm_head_inbox_step()
        torch.cuda.synchronize()
        COMM.Barrier()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            g_tokens = lm_head_inbox_step()
        COMM.Barrier()
        for call in range(args.graph_calls):
            fresh_inputs(20_000 + call)
            runner([x, weight_t], tactic=tactic, out=local)
            torch.cuda.synchronize()
            row = host_row(local)
            COMM.Barrier()
            graph.replay()
            verify(g_tokens, row, "graph", stats)
            COMM.Barrier()
        status = int(ws.status.item())
        totals = {
            k: {kk: int(COMM.allreduce(vv)) for kk, vv in v.items()} for k, v in stats.items()
        }
        ok = all(
            v["gathered_bad"] == 0 and v["token_bad"] == 0 and v["not_reset"] == 0
            for v in totals.values()
        )
        ok = agree(ok and status == 0)
        result["check"] = {
            "ok": ok,
            "status_max": int(max_over_ranks(status)),
            "per_mode_sum_over_ranks": totals,
        }
        log("check", json.dumps(result["check"]))
        del graph

    if args.steps:
        from tensorrt_llm._torch.distributed import allgather

        steps = args.step_count
        xs = torch.empty(steps, ROWS, HIDDEN, dtype=torch.bfloat16, device=dev)
        seeds = torch.empty(steps, ROWS, dtype=torch.int64, device=dev)
        offsets = torch.empty(steps, ROWS, dtype=torch.int64, device=dev)
        got_rows = torch.empty(steps, ROWS, SHARD * WORLD, dtype=torch.bfloat16, device=dev)
        got_tokens = torch.empty(steps, ROWS, dtype=torch.int32, device=dev)
        box = torch.zeros(1, ROWS, dtype=torch.int32, device=dev)
        sleep_cycles = 200_000  # about 100 us

        def fresh_steps(tag: int):
            gc = torch.Generator(device=dev).manual_seed(tag)  # same on every rank
            xs.copy_(
                (torch.randn(steps, ROWS, HIDDEN, generator=gc, device=dev) * 2.0).to(
                    torch.bfloat16
                )
            )
            gs = torch.Generator().manual_seed(tag + 1)
            seeds.copy_(torch.randint(0, 2**62, (steps, ROWS), generator=gs, dtype=torch.int64))
            offsets.copy_(torch.randint(0, 2**40, (steps, ROWS), generator=gs, dtype=torch.int64))

        def body(late_rank: int, where: str):
            for s in range(steps):
                late = RANK == late_rank and s % 2 == 1
                if late and where == "before_gemm":
                    torch.cuda._sleep(sleep_cycles)
                runner([xs[s], weight_t], tactic=tactic, out=ws.slot(ROWS))
                if late and where == "before_consumer":
                    torch.cuda._sleep(sleep_cycles)
                tokens = hs.gather_split_sample(
                    ws.inbox,
                    ws.gathered[:ROWS],
                    temps,
                    top_ks,
                    top_ps,
                    min_ps,
                    seeds[s],
                    offsets[s],
                    MULTIPLIER,
                    SOFT_CAP,
                    ws.states,
                    ws.status,
                    SHARD,
                    hs.MAX_ROWS,
                    True,
                    pdl,
                )
                got_rows[s].copy_(ws.gathered[:ROWS])
                got_tokens[s].copy_(tokens)
                box.copy_(tokens.view(1, ROWS))
                allgather(box, mapping, dim=0)

        variants = [("none", -1)]
        variants += [("before_gemm", r) for r in range(WORLD)]
        variants += [("before_consumer", r) for r in range(WORLD)]
        stats = {}
        fresh_steps(1)
        for v, (where, late_rank) in enumerate(variants):
            body(late_rank, where)  # eager: compiles, and runs the path once
            torch.cuda.synchronize()
            COMM.Barrier()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                body(late_rank, where)
            torch.cuda.synchronize()
            COMM.Barrier()
            key = f"{where}:{late_rank}"
            st = stats.setdefault(key, {"steps": 0, "gathered_bad": 0, "token_bad": 0, "draws": 0})
            for rep in range(args.step_replays):
                fresh_steps(1000 * (v + 1) + rep)
                torch.cuda.synchronize()
                COMM.Barrier()
                graph.replay()
                torch.cuda.synchronize()
                for s in range(steps):
                    runner([xs[s], weight_t], tactic=tactic, out=local)
                    row = allgather(local, mapping, dim=-1)
                    ref = torch.ops.trtllm.fused_sample_from_logits(
                        hs.reference_capped(row, MULTIPLIER, SOFT_CAP),
                        temps,
                        top_ks,
                        top_ps,
                        min_ps,
                        seeds[s],
                        offsets[s],
                    )
                    st["gathered_bad"] += int(
                        (got_rows[s].view(torch.int16) != row.view(torch.int16)).sum()
                    )
                    st["token_bad"] += int((got_tokens[s] != ref).sum())
                    st["steps"] += 1
                    st["draws"] += ROWS
                torch.cuda.synchronize()
                COMM.Barrier()
            del graph
        not_reset = int((ws.inbox != -1).sum())
        status = int(ws.status.item())
        totals = {
            k: {kk: int(COMM.allreduce(vv)) for kk, vv in v.items()} for k, v in stats.items()
        }
        not_reset_all = int(COMM.allreduce(not_reset))
        ok = all(v["gathered_bad"] == 0 and v["token_bad"] == 0 for v in totals.values())
        ok = agree(ok and not_reset_all == 0 and status == 0)
        result["steps"] = {
            "ok": ok,
            "steps_per_graph": steps,
            "replays": args.step_replays,
            "not_reset": not_reset_all,
            "status_max": int(max_over_ranks(status)),
            "rank_steps": sum(v["steps"] for v in totals.values()),
            "draws": sum(v["draws"] for v in totals.values()),
            "per_variant_sum_over_ranks": totals,
        }
        log(
            "steps",
            json.dumps(
                {k: v for k, v in result["steps"].items() if k != "per_variant_sum_over_ranks"}
            ),
        )

    if args.producers:
        from tensorrt_llm._torch.modules.low_m_gemm import _CuBLASGemmRunner, _DirectGemmRunner

        cublas = _CuBLASGemmRunner(has_bias=False)
        direct = _DirectGemmRunner(pdl=pdl)
        probe = [x, weight_t]
        producers = [("cublas", cublas, t) for t in cublas.get_valid_tactics(probe, None)]
        producers += [("splitk", runner, tuple(t)) for t in runner.get_valid_tactics(probe, None)]
        producers += [("direct", direct, tuple(t)) for t in direct.get_valid_tactics(probe, None)]
        names = COMM.allgather([(n, str(t)) for n, _, t in producers])
        if any(n != names[0] for n in names):
            raise RuntimeError("ranks list different LM-head producers")
        stats = {}
        for i, (name, producer, ptactic) in enumerate(producers):
            st = stats.setdefault(
                name,
                {"producers": 0, "calls": 0, "gathered_bad": 0, "token_bad": 0, "not_reset": 0},
            )
            st["producers"] += 1
            producer([x, weight_t], tactic=ptactic, out=local)  # compile and warm
            torch.cuda.synchronize()
            COMM.Barrier()
            for call in range(args.producer_calls):
                fresh_inputs(50_000 + 16 * i + call)
                producer([x, weight_t], tactic=ptactic, out=local)
                torch.cuda.synchronize()
                row = host_row(local)
                COMM.Barrier()
                producer([x, weight_t], tactic=ptactic, out=ws.slot(ROWS))
                sample = call == 0
                tokens = hs.gather_split_sample(
                    ws.inbox,
                    ws.gathered[:ROWS],
                    temps,
                    top_ks,
                    top_ps,
                    min_ps,
                    seed,
                    offset,
                    MULTIPLIER,
                    SOFT_CAP,
                    ws.states,
                    ws.status,
                    SHARD,
                    hs.MAX_ROWS,
                    sample,
                    pdl,
                )
                torch.cuda.synchronize()
                st["calls"] += 1
                st["gathered_bad"] += int(
                    (ws.gathered[:ROWS].view(torch.int16) != row.view(torch.int16)).sum()
                )
                if sample:
                    ref = torch.ops.trtllm.fused_sample_from_logits(
                        hs.reference_capped(row, MULTIPLIER, SOFT_CAP),
                        temps,
                        top_ks,
                        top_ps,
                        min_ps,
                        seed,
                        offset,
                    )
                    st["token_bad"] += int((tokens != ref).sum())
                st["not_reset"] += int((ws.inbox != -1).sum())
                COMM.Barrier()
        status = int(ws.status.item())
        totals = {
            k: {kk: int(COMM.allreduce(vv)) for kk, vv in v.items()} for k, v in stats.items()
        }
        ok = all(
            v["gathered_bad"] == 0 and v["token_bad"] == 0 and v["not_reset"] == 0
            for v in totals.values()
        )
        ok = agree(ok and status == 0)
        result["producers"] = {
            "ok": ok,
            "count": len(producers),
            "status_max": int(max_over_ranks(status)),
            "per_runner_sum_over_ranks": totals,
        }
        log("producers", json.dumps(result["producers"]))

    if args.bench:
        from tensorrt_llm._torch.distributed import allgather

        accept_draft = torch.zeros(1, ROWS - 1, dtype=torch.int32, device=dev)
        token_box = torch.zeros(1, ROWS, dtype=torch.int32, device=dev)

        def accept(tokens):
            out = _accept.dflash_accept(
                tokens, accept_draft, 0, _accept.FORCE_FIXED, 4, 8, 0.0, None, None
            )
            # In the model the next step's forward (~250 collectives) separates two head
            # calls; the inbox protocol needs one collective between them. Both arms pay
            # the same small all-gather here.
            token_box.copy_(out[0][:, :ROWS])
            allgather(token_box, mapping, dim=0)
            return out

        def builtin_step():
            out = runner([x, weight_t], tactic=tactic, out=local)
            full = allgather(out, mapping, dim=-1)
            capped = hs.reference_capped(full, MULTIPLIER, SOFT_CAP)
            tokens = torch.ops.trtllm.fused_sample_from_logits(
                capped, temps, top_ks, top_ps, min_ps, seed, offset
            )
            return accept(tokens)

        def lm_head_inbox_full_step():
            return accept(lm_head_inbox_step())

        fresh_inputs(30_000)
        graphs = {}
        for name, fn in (("builtin", builtin_step), ("lm_head_inbox", lm_head_inbox_full_step)):
            for _ in range(2):
                fn()
            torch.cuda.synchronize()
            COMM.Barrier()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                for _ in range(args.repeat):
                    fn()
            torch.cuda.synchronize()
            COMM.Barrier()
            graphs[name] = graph
        times = {"builtin": [], "lm_head_inbox": []}
        for rnd in range(args.rounds):
            order = ("builtin", "lm_head_inbox") if rnd % 2 == 0 else ("lm_head_inbox", "builtin")
            for name in order:
                for _ in range(3):
                    COMM.Barrier()
                    graphs[name].replay()
                    torch.cuda.synchronize()
                samples = []
                for _ in range(10):
                    COMM.Barrier()
                    start, end = (
                        torch.cuda.Event(enable_timing=True),
                        torch.cuda.Event(enable_timing=True),
                    )
                    start.record()
                    graphs[name].replay()
                    end.record()
                    torch.cuda.synchronize()
                    samples.append(start.elapsed_time(end) * 1000.0 / args.repeat)
                us = statistics.median(samples)
                times[name].append(max_over_ranks(us))
        result["bench_us_per_step_max_over_ranks"] = times
        result["bench_delta_us"] = [z - b for z, b in zip(times["lm_head_inbox"], times["builtin"])]
        log("bench", json.dumps({"times": times, "delta": result["bench_delta_us"]}))
        assert int(ws.status.item()) == 0

    if RANK == 0 and args.out:
        with open(args.out, "w") as f:
            json.dump(result, f, indent=1)
    COMM.Barrier()
    log("done", time.strftime("%H:%M:%S"))


if __name__ == "__main__":
    try:
        main()
    except Exception:  # noqa: BLE001 - report and abort every rank
        import traceback

        traceback.print_exc()
        sys.stdout.flush()
        COMM.Abort(1)
