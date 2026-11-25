# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""dep2's expert all-to-all (`moe_a2a_dispatch` -> experts -> `moe_a2a_combine`) against the engine's `NVLinkOneSided`
(the sentinel transport for small steps, `use_sentinel_for_dispatch`), on two GPUs. Both pick the transport by the same
rule from the same vote, so each step runs ours on the transport stock takes, and ours on the fence transport too.

Its own launcher, as `_moe_a2a_op_matrix.py`: run it plainly and it re-executes itself under `mpirun`, one rank per
device named in CUDA_VISIBLE_DEVICES, under a deadline.

    CUDA_VISIBLE_DEVICES=0,1 python _moe_a2a_vs_stock.py

The dispatch input is dep2's: EP2, top-6 of 384 experts, NVFP4 rows (hidden 5120: 2560 code bytes, 320 scale bytes)
with fp32 routing weights. Both paths dispatch the same rows; every received row's expert output is a fixed function of
the row's bytes and the receiving rank (a stand-in for the experts with every mantissa bit in use); both combine. The
combined rows must be bitwise equal, and equal to the reference sum. Row counts per rank: a few, ~145 (a prefix-cache
hit), the engine's sentinel limit for this layout (`sentinel_max_tokens`, sentinel transport) and one more (the engine
falls back to the fence transport); each step's transport is checked to be the one named, on both. Then the kernels
alone are timed: ours on each transport, dispatch -> a copy into the combine region -> combine, 20 rounds in one CUDA
graph (no host work in the timing, as in the dep2 pass graphs), median of 15 replays per round.
"""

import os
import signal
import subprocess
import sys
import zlib
from pathlib import Path

import torch

_WORKER_FLAG = "--rank-worker"
DEADLINE_S = 900
WORLD = 2
MAX_ROWS = 1024
HIDDEN, TOP_K, EXPERTS = 5120, 6, 384
RANK = 0


def _say(msg: str) -> None:
    if RANK == 0:
        print(f"[a2a-vs-stock] {msg}", flush=True)


def _inputs(why: str, rows: int, src: int):
    """Rank `src`'s rows of a step: NVFP4 code bytes (the first 8 hold the row's id), scale bytes, expert ids
    (some rows routed to one rank only), fp32 weights; a pure function of the step and the rank."""
    gen = torch.Generator(device="cuda").manual_seed(
        (zlib.crc32(why.encode()) + 7919 * src) % 2**31
    )
    ids = torch.argsort(torch.rand(rows, EXPERTS, generator=gen, device="cuda"), dim=1)[
        :, :TOP_K
    ].to(torch.int32)
    half = EXPERTS // 2
    ids[0::4] = ids[0::4] % half  # all on rank 0's experts
    ids[1::4] = ids[1::4] % half + half  # all on rank 1's
    xq = torch.randint(0, 256, (rows, HIDDEN // 2), generator=gen, device="cuda", dtype=torch.uint8)
    xq[:, :8] = (
        (torch.arange(rows, device="cuda", dtype=torch.int64) + (src << 32))
        .view(torch.uint8)
        .view(rows, 8)
    )
    # The sentinel transport carries e2m1 values, not bytes: it sends a -0 nibble (1000b) as +0. Rows without one
    # arrive byte for byte on both transports, so the stand-in experts see the same bytes.
    lo, hi = xq & 0x0F, xq >> 4
    xq = torch.where(lo == 8, 0, lo) | (torch.where(hi == 8, 0, hi) << 4)
    xsf = torch.randint(
        0, 256, (rows, HIDDEN // 16), generator=gen, device="cuda", dtype=torch.uint8
    )
    w = torch.rand(rows, TOP_K, generator=gen, device="cuda")
    return xq, xsf, ids, w


def _expert_output(xq: torch.Tensor, rank: int) -> torch.Tensor:
    """A received row's partial sum on `rank`: bf16 `[n, hidden]`, a fixed function of the row's bytes and the rank."""
    v = torch.sin(xq.to(torch.float32) * 0.0173 + 0.37 * (rank + 1))
    return torch.cat([v, v * 0.5], dim=1).to(torch.bfloat16)


def _reference(mine, rank_rows: int) -> torch.Tensor:
    """This rank's combined rows: the sum over the ranks holding one of a row's experts of their partial for it."""
    xq, _, ids, _ = mine
    total = torch.zeros(rank_rows, HIDDEN, dtype=torch.float32, device="cuda")
    for r in range(WORLD):
        lo, hi = EXPERTS * r // WORLD, EXPERTS * (r + 1) // WORLD
        sent = ((ids >= lo) & (ids < hi)).any(dim=1, keepdim=True)
        total += torch.where(sent, _expert_output(xq, r).float(), 0.0)
    return total.to(torch.bfloat16)


def _ours(ws, mine, counts, sentinel_max_rows):
    """Ours on the transport `sentinel_max_rows` picks: the workspace's (stock's rule), or 0 (fence only)."""
    from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.moe import moe_a2a

    xq, xsf, ids, w = mine
    rows = max(max(1, c) for c in counts)
    payloads = [xq, xsf, ids, w]
    moe_a2a.moe_a2a_dispatch.is_valid(ids, payloads, 2, ws.workspace, ws.metainfo, rows, RANK, WORLD, EXPERTS,
                                      sentinel_max_rows)  # fmt: skip
    recv, offset = moe_a2a.moe_a2a_dispatch(ids, payloads, 2, ws.workspace, ws.metainfo, rows, RANK, WORLD, EXPERTS,
                                            sentinel_max_rows)  # fmt: skip
    # Dead rows carry garbage the combine must skip; on a sentinel step a row not routed here does too.
    region = moe_a2a.moe_a2a_combine_payload(ws.workspace, ws.metainfo, RANK, WORLD, rows, HIDDEN)
    region.copy_(
        _expert_output(recv[0], RANK)
    )  # as the target: the experts write the combine region
    return moe_a2a.moe_a2a_combine(region, xq.shape[0], ws.workspace, ws.metainfo, rows, RANK, WORLD, TOP_K, offset,
                                   sentinel_max_rows, True)  # fmt: skip


def _stock(comm, mine, counts):
    xq, xsf, ids, w = mine
    sentinel = comm.use_sentinel_for_dispatch(max(counts), xq, xsf, w, None)
    recv_x, _, recv_ids, _ = comm.dispatch(xq, xsf, ids, w, list(counts))
    # A sentinel step does not compact: every token of every rank has a row, and one routed to no expert here carries
    # invalid ids in every slot; the experts give such a row nothing (the fence step sends it nowhere).
    lo, hi = EXPERTS * RANK // WORLD, EXPERTS * (RANK + 1) // WORLD
    here = ((recv_ids >= lo) & (recv_ids < hi)).any(dim=-1, keepdim=True)
    partial = torch.where(here, _expert_output(recv_x, RANK), 0).to(torch.bfloat16)
    return comm.combine(partial), sentinel


def _graph_us(
    world, ws, rows: int, sentinel_max_rows: int, rounds: int = 20, reps: int = 15
) -> float:
    """Median GPU time per round (us) of ours on the transport `sentinel_max_rows` picks for `rows` rows per rank:
    dispatch -> a copy into the combine region (a stand-in for the experts) -> combine, `rounds` rounds captured in one
    CUDA graph, replayed `reps` times with both ranks starting together."""
    from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.moe import moe_a2a

    xq, xsf, ids, w = _inputs(f"timing {rows}", rows, RANK)
    src = _expert_output(torch.cat([xq, xq]), RANK)

    def round_():
        recv, offset = moe_a2a.moe_a2a_dispatch(ids, [xq, xsf, ids, w], 2, ws.workspace, ws.metainfo, rows, RANK,
                                                WORLD, EXPERTS, sentinel_max_rows)  # fmt: skip
        region = moe_a2a.moe_a2a_combine_payload(
            ws.workspace, ws.metainfo, RANK, WORLD, rows, HIDDEN
        )
        region.copy_(src)
        moe_a2a.moe_a2a_combine(region, rows, ws.workspace, ws.metainfo, rows, RANK, WORLD, TOP_K, offset,
                                sentinel_max_rows, True)  # fmt: skip

    round_()
    torch.cuda.synchronize()
    world.Barrier()
    graph, stream = torch.cuda.CUDAGraph(), torch.cuda.Stream()
    with torch.cuda.stream(stream), torch.cuda.graph(graph, stream=stream):
        for _ in range(rounds):
            round_()
    torch.cuda.synchronize()
    times = []
    for _ in range(reps):
        world.Barrier()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end) * 1000 / rounds)
    return sorted(times)[len(times) // 2]


def _run_one_rank() -> int:
    global RANK
    from mpi4py import MPI

    from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.moe import moe_a2a
    from tensorrt_llm._torch.moe.fused_moe.communication.nvlink_one_sided import NVLinkOneSided
    from tensorrt_llm.mapping import Mapping

    world = MPI.COMM_WORLD
    RANK = world.Get_rank()
    assert world.Get_size() == WORLD, f"written for {WORLD} ranks"
    torch.cuda.set_device(RANK)
    mapping = Mapping(world_size=WORLD, rank=RANK, gpus_per_node=WORLD, tp_size=WORLD, moe_ep_size=WORLD,
                      enable_attention_dp=True)  # fmt: skip
    ws = moe_a2a.moe_a2a_workspace(mapping, MAX_ROWS, TOP_K, [HIDDEN // 2, HIDDEN // 16, 4 * TOP_K, 4 * TOP_K],
                                   2 * HIDDEN, HIDDEN, EXPERTS)  # fmt: skip
    stock = NVLinkOneSided(mapping, EXPERTS, TOP_K, MAX_ROWS, hidden_size=HIDDEN, dtype=torch.bfloat16,
                           sentinel_payload_format="NVFP4")  # fmt: skip
    limit = stock.sentinel_max_tokens
    _say(f"stock sentinel_max_tokens {limit}, ours sentinel_max_rows {ws.sentinel_max_rows}")
    assert limit > 0, (
        "the engine's sentinel transport is off for this layout: nothing to compare against"
    )
    assert ws.sentinel_max_rows == limit, "our vote gives stock's limit"
    steps = [("a few rows", (5, 3), True), ("a prefix-cache hit", (145, 131), True),
             ("the sentinel limit", (limit, limit - 9), True),
             ("one past the limit", (limit + 1, limit), False)]  # fmt: skip
    try:
        for why, counts, want_sentinel in steps:
            mine = _inputs(why, counts[RANK], RANK)
            ours = _ours(ws, mine, counts, ws.sentinel_max_rows)
            ours_fence = _ours(ws, mine, counts, 0)
            theirs, sentinel = _stock(stock, mine, counts)
            torch.cuda.synchronize()
            assert sentinel == want_sentinel, (
                f"{why}: the engine took the {'sentinel' if sentinel else 'fence'} transport"
            )
            mode = "sentinel" if max(counts) <= ws.sentinel_max_rows else "fence"
            assert mode == ("sentinel" if sentinel else "fence"), (
                f"{why}: ours took the {mode} transport"
            )
            ref = _reference(mine, counts[RANK])
            same = bool(torch.equal(ours, theirs))
            _say(
                f"{why} {counts}: {mode}; ours == stock {same}, ours (fence) == stock "
                f"{bool(torch.equal(ours_fence, theirs))}, ours == reference {bool(torch.equal(ours, ref))}"
            )
            assert ours.shape == theirs.shape == ref.shape == ours_fence.shape
            assert same, (
                f"{why}: the combined rows differ from the engine's (max |diff| "
                f"{float((ours.float() - theirs.float()).abs().max()):.3e})"
            )
            assert torch.equal(ours_fence, theirs), (
                f"{why}: ours on the fence transport differs from the engine's"
            )
            assert torch.equal(ours, ref), f"{why}: the combined rows differ from the reference sum"
            world.Barrier()
        # dep2's decoder-pass buckets (multiples of the 140-row replay window), the limit and one past it
        for rows in (1, 140, 280, 420, limit, limit + 1):
            auto = _graph_us(world, ws, rows, ws.sentinel_max_rows)
            fence = _graph_us(world, ws, rows, 0)
            mode = "sentinel" if rows <= ws.sentinel_max_rows else "fence"
            _say(
                f"{rows} rows per rank: us per round, ours as stock ({mode}) {auto:.1f}, ours on fence {fence:.1f}"
            )
            world.Barrier()
    except BaseException:
        import traceback

        print(f"[rank {RANK}] FAILED", flush=True)
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        world.Abort(1)
    print(f"[rank {RANK}] all steps equal", flush=True)
    return 0


def _spawn_ranks() -> None:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    assert visible, "set CUDA_VISIBLE_DEVICES to the two devices this run owns, e.g. 0,1"
    n = len([d for d in visible.split(",") if d.strip()])
    assert n == WORLD, f"this check runs on {WORLD} devices; CUDA_VISIBLE_DEVICES names {n}"
    command = ["mpirun", "--oversubscribe", "--allow-run-as-root", "-n", str(WORLD), sys.executable,
               str(Path(__file__).resolve()), _WORKER_FLAG]  # fmt: skip
    print(f"[launcher] {' '.join(command)}", flush=True)
    process = subprocess.Popen(command, start_new_session=True)
    try:
        code = process.wait(timeout=DEADLINE_S)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()
        raise AssertionError(
            f"the {WORLD}-rank run did not finish in {DEADLINE_S}s (wedged)"
        ) from None
    assert code == 0, f"the {WORLD}-rank run exited {code}"


if __name__ == "__main__":
    if _WORKER_FLAG in sys.argv:
        sys.exit(_run_one_rank())
    _spawn_ranks()
    print("OK")
