# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Certification matrix for the FP8 target's gather of a decode step's QSA selection rows over the
TP group (`lamport_allgather__index_split`): every output bitwise the NCCL all-gather, over mixed
trunk and draft calls and with the split's row map, eagerly and replayed from a CUDA graph."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch

assert torch.cuda.is_available(), "the selection gather requires CUDA devices"

DEADLINE_S = 900
WIDTH = 2051
#: Requests a step (each rank scores `requests / world` of them): the entry's CELLS at this world
#: (tep2 / tep4 HT batches), set by `_run_one_rank`.
REQUESTS: tuple = ()
MAX_ROWS = 0
_MODELING_V2_TESTS = Path(__file__).resolve().parents[4]

lam = gather = MPI_RANK = None
RANK = WORLD = 0


def _share(rows: int, seed: int) -> torch.Tensor:
    """This rank's `[rows / WORLD, WIDTH]` selection rows: token indices, -1 padding, and the
    int32 extremes next to the not-arrived marker (int32 min itself never occurs)."""
    g = torch.Generator(device="cuda").manual_seed(seed * 131 + RANK)
    share = torch.randint(-1, 1 << 24, (rows // WORLD, WIDTH), generator=g, device="cuda")
    share = share.to(torch.int32)
    share[:, :3] = -1
    share[:, 3] = 2**31 - 1
    share[:, 4] = -(2**31) + 1
    return share


def _nccl(share: torch.Tensor) -> torch.Tensor:
    import torch.distributed as dist

    out = torch.empty((WORLD * share.shape[0], WIDTH), dtype=torch.int32, device="cuda")
    dist.all_gather_into_tensor(out, share)
    return out


def _ensure_process_group() -> None:
    """A process group for the NCCL reference, from the environment or a port rank 0 broadcasts."""
    import torch.distributed as dist

    if dist.is_initialized():
        return
    if not all(os.environ.get(k) for k in ("MASTER_ADDR", "MASTER_PORT", "RANK", "WORLD_SIZE")):
        addr = None
        if RANK == 0:
            import socket

            with socket.socket() as sock:
                sock.bind(("", 0))
                addr = (socket.gethostbyname(socket.gethostname()), sock.getsockname()[1])
        host, port = MPI_RANK.comm.bcast(addr, root=0)
        os.environ["MASTER_ADDR"] = host
        os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group(
        backend="nccl",
        rank=RANK,
        world_size=WORLD,
        device_id=torch.device("cuda", torch.cuda.current_device()),
    )


def _same(got: torch.Tensor, want: torch.Tensor, what: str) -> None:
    assert got.shape == want.shape and torch.equal(got, want), f"{what}: not bitwise NCCL's"


def _step_rows() -> list[int]:
    """One served step per batch: 12 trunk layers at 3 rows a request, 2 draft steps at 1."""
    seq = []
    for requests in REQUESTS:
        seq += [3 * requests] * 12 + [requests] * 2
    return seq


def check_eager_mixed_sequence_is_bitwise_nccl() -> None:
    """Every call of a long mixed sequence equals NCCL's while the buffers' three slots rotate."""
    for i, rows in enumerate(_step_rows() * 2):
        share = _share(rows, 100 + i)
        got = gather(share, lam)
        _same(got, _nccl(share), f"call {i} ({rows} rows)")
    torch.cuda.synchronize()
    assert MPI_RANK.same_on_ranks(lam.state[:2])


def _row_map(rows: int, seed: int) -> torch.Tensor:
    """A permutation of `rows` rows, the same on every rank (the split plan's `perm_rows`)."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    return torch.randperm(rows, generator=g, device="cuda").to(torch.int32)


def check_row_map_stores_each_gathered_row_at_its_row() -> None:
    """Gathered row `j` lands at row `dst_rows[j]`, bitwise NCCL's gather permuted the same way,
    interleaved with plain calls, eagerly and in a CUDA graph replayed with new rows."""
    for i, rows in enumerate([3 * 64, 64, 3 * 192, 3 * 16, 192, 3 * 128]):
        share = _share(rows, 7000 + i)
        row_map = _row_map(rows, 7100 + i)
        want = torch.empty_like(_nccl(share))
        want[row_map.long()] = _nccl(share)
        _same(gather(share, lam, row_map), want, f"mapped call {i} ({rows} rows)")
        plain = _share(rows, 7200 + i)
        _same(gather(plain, lam), _nccl(plain), f"plain call after mapped call {i}")
    torch.cuda.synchronize()
    for requests in (32, 96):
        rows_seq = [3 * requests] * 12 + [requests] * 2
        maps = [
            _row_map(r, 7300 + j + requests) if j < 12 else None for j, r in enumerate(rows_seq)
        ]
        shares = [
            torch.zeros((r // WORLD, WIDTH), dtype=torch.int32, device="cuda") for r in rows_seq
        ]
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for s_, m_ in zip(shares, maps):
                gather(s_, lam, m_)
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        MPI_RANK.barrier()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            outs = [gather(s_, lam, m_) for s_, m_ in zip(shares, maps)]
        torch.cuda.synchronize()
        MPI_RANK.barrier()
        for rep in range(6):
            for j, s_ in enumerate(shares):
                s_.copy_(_share(s_.shape[0] * WORLD, 7500 + 100 * rep + j + requests))
            torch.cuda.synchronize()
            MPI_RANK.barrier()
            graph.replay()
            torch.cuda.synchronize()
            for j, (s_, m_, o) in enumerate(zip(shares, maps, outs)):
                want = _nccl(s_)
                if m_ is not None:
                    w2 = torch.empty_like(want)
                    w2[m_.long()] = want
                    want = w2
                _same(o, want, f"{requests} requests, replay {rep}, call {j}")
            MPI_RANK.barrier()
        del graph


CHECKS = (
    check_eager_mixed_sequence_is_bitwise_nccl,
    check_row_map_stores_each_gathered_row_at_its_row,
)


def _run_one_rank() -> int:
    global lam, gather, MPI_RANK, RANK, WORLD, REQUESTS, MAX_ROWS
    os.environ.pop("TLLM_DISABLE_MPI", None)
    os.environ.setdefault("TRTLLM_ENABLE_PDL", "1")
    sys.path.insert(0, str(_MODELING_V2_TESTS / "comm"))
    import _lockstep

    MPI_RANK = _lockstep.Rank(_lockstep.parse_args(sys.argv[1:]))
    RANK, WORLD = MPI_RANK.rank, MPI_RANK.world
    from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_fp8__sm_103__unified.ops.comm import (  # noqa: E501
        lamport_allgather__index_split as entry,
    )

    gather = entry.lamport_allgather
    cells = [c.spec for c in gather.CELLS if c.spec["world"] == WORLD]
    assert cells and all(s["width"] == WIDTH for s in cells), f"no cell at world {WORLD}"
    REQUESTS = tuple(s["requests"] for s in cells)
    MAX_ROWS = 3 * max(REQUESTS)
    # The buffers' handles travel over MPI (the engine's TP communicator); the NCCL reference
    # needs a torch process group.
    lam = entry.build(MPI_RANK.mapping, MAX_ROWS * WIDTH)
    _ensure_process_group()
    code = _lockstep.run_checks(MPI_RANK, list(CHECKS))
    torch.cuda.synchronize()
    sys.stdout.flush()
    sys.stderr.flush()
    import torch.distributed as dist

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
    worker = [_lockstep.WORKER_FLAG, "--world-size", str(world)]
    _launch.mpirun_ranks(
        __file__, world, worker + ["--fabric-handle", args.fabric_handle], DEADLINE_S
    )
    print("OK: lamport selection gather")
