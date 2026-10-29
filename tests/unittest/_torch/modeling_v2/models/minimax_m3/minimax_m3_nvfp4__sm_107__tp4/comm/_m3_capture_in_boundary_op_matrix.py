# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CAPTURE_NORM_IN_BOUNDARY on the tp4 multicast all-reduce + RMSNorm op, at 4 ranks.

With ``cap_out`` the boundary also writes the Eagle3 capture of its updated residual. Asserted bit for bit against
``m3_spec_tail.capture_norm`` of the op's own updated residual, in a strided [M, H] block of a [rows, 3H] capture
buffer (the other bytes untouched), and the op's normed / updated / MXFP8 outputs bitwise the same call's without
``cap_out``. Every kernel a boundary can run:

- the 8-CTA cluster kernel in the stock oneshot order (options 39 / 47, plain calls up to 8 rows, x and x + x2,
  prepushed 2);
- the row kernel (plain options 7 / 15 and the carveout instance 71 / 79, 1 to 8 rows on the 8-token bucket, 12 rows
  on the 16-token bucket, 32 rows on the 48-token bucket, prepushed 2);

(The tp4 MoE finalize runs the shared GB300 op, which has no capture epilogue.)

at residual scales 1e-3 to 1e3 and the fc_norm eps 1e-6 / 1e-5.

Its own 4-rank launcher, as ``_m3_draft_ar_norm_op_matrix.py`` is; the collected entry point is
``test_modeling_v2_m3_capture_in_boundary.py``.

    CUDA_VISIBLE_DEVICES=0,1,2,3 python _m3_capture_in_boundary_op_matrix.py
"""

import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.distributed as dist

assert torch.cuda.is_available(), "m3_mc_allreduce_norm requires CUDA devices"

_WORKER_FLAG = "--rank-worker"
DEADLINE_S = 900

HIDDEN = 6144
EPS = 1e-6
CLUSTER = 8
BUCKETS = (8, 16, 48)
EMPTY_WORD = -2147483648

# Bound inside the rank body: importing the entries pulls in tensorrt_llm.
allreduce_entry: Any = None
ar_module: Any = None
RANK = 0
WORLD = 1
WORKSPACES: Dict[int, Any] = {}


def _gen(seed: int, rank: int) -> torch.Generator:
    return torch.Generator(device="cuda").manual_seed(seed * 1009 + rank + 1)


def _rows(seed: int, rank: int, m: int, scale: float = 1.0) -> torch.Tensor:
    g = _gen(seed, rank + 1)
    return (torch.randn(m, HIDDEN, generator=g, device="cuda") * scale).to(torch.bfloat16)


def _norm_w(seed: int) -> torch.Tensor:
    g = _gen(seed + 11, 0)
    return (1.0 + 0.25 * torch.randn(HIDDEN, generator=g, device="cuda")).to(torch.bfloat16)


def _make_workspace(maxt: int):
    """One bucket's exchange, as ``McAllReduceWorkspace.create`` builds it, over the torch.distributed group."""
    from tensorrt_llm._torch.distributed.ops import _make_mnnvl_mcast_buffer
    from tensorrt_llm.mapping import Mapping

    mapping = Mapping(world_size=WORLD, tp_size=WORLD, rank=RANK)
    words = 4 * maxt * WORLD * (HIDDEN // 2)
    handle = _make_mnnvl_mcast_buffer(dist.group.WORLD, words * 4, mapping, False)
    uc = handle.get_uc_buffer(RANK, (words,), torch.int32, 0)
    mc = handle.get_mc_buffer((words,), torch.int32, 0)
    uc.fill_(EMPTY_WORD)
    torch.cuda.synchronize()
    dist.barrier()
    ws = ar_module.McAllReduceWorkspace(
        uc=uc,
        mc=mc,
        flags={},
        rank=RANK,
        world_size=WORLD,
        hidden=HIDDEN,
        handle=handle,
        comm=dist.group.WORLD,
        max_tokens=maxt,
    )
    ws.flags_for(CLUSTER)
    return ws


def _push(ws, rows: torch.Tensor) -> None:
    """Store this rank's rows into every rank's exchange (set A), as a producer does before a prepushed call."""
    m = rows.shape[0]
    torch.cuda.synchronize()
    parity = (ws.flags_for(CLUSTER).view(ws.max_tokens, CLUSTER)[:m] & 1).long()
    half = ws.max_tokens * WORLD * (HIDDEN // 2)
    chunk = HIDDEN // 2 // CLUSTER
    h16 = rows.contiguous().view(torch.int16)
    h16 = torch.where(h16 == -32768, torch.zeros_like(h16), h16)
    words = h16.view(torch.int32).view(m, CLUSTER, chunk)
    t = torch.arange(m, device="cuda").view(m, 1, 1)
    c = torch.arange(CLUSTER, device="cuda").view(1, CLUSTER, 1)
    w = torch.arange(chunk, device="cuda").view(1, 1, chunk)
    index = parity.view(m, CLUSTER, 1) * half + (t * WORLD + RANK) * (HIDDEN // 2) + c * chunk + w
    ws.mc[index.reshape(-1)] = words.reshape(-1)
    torch.cuda.synchronize()
    dist.barrier()


def _assert_bitwise(out: torch.Tensor, ref: torch.Tensor, where: str) -> None:
    assert out.dtype == ref.dtype and out.shape == ref.shape, (where, out.dtype, ref.dtype)
    a, b = out.contiguous().view(torch.uint8), ref.contiguous().view(torch.uint8)
    bad = (a != b).sum().item()
    assert bad == 0, f"{where}: {bad} of {a.numel()} bytes differ"


def _capture_reference(updated, weight, eps, out) -> None:
    from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_nvfp4__sm_107__tp4.ops import (  # noqa: E501
        m3_spec_tail,
    )

    m3_spec_tail.capture_norm(updated, None, weight, eps, out)


def _one_case(
    kind: str, m: int, maxt: int, opts: int, scale: float, quant: bool, seed: int
) -> None:
    from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_nvfp4__sm_107__tp4.ops.comm.m3_mc_allreduce_norm import (  # noqa: E501
        sf_written,
    )

    cap_eps = (1e-6, 1e-5)[seed % 2]
    ws = WORKSPACES[maxt]
    residual = (_rows(seed + 7, -1, m).float() * 4.0 * scale).to(torch.bfloat16)
    norm_w, cap_w = _norm_w(seed), _norm_w(seed + 101)
    results = []
    for with_cap in (False, True):
        buf = torch.full((m + 3, 3 * HIDDEN), 7.0, dtype=torch.bfloat16, device="cuda")
        cap = buf[:m, HIDDEN : 2 * HIDDEN] if with_cap else None
        cw = cap_w if with_cap else None
        flags = ws.flags_for(CLUSTER)
        x = _rows(seed, RANK, m, scale)
        x2 = _rows(seed + 3, RANK, m, scale) if kind == "x2" else None
        prepushed = 2 if kind == "prepushed" else 0
        if prepushed:
            _push(ws, x)
        outs = allreduce_entry.m3_mc_allreduce_norm(
            x, x2, residual, norm_w, EPS, ws.uc, ws.mc.data_ptr(), flags, RANK, WORLD, CLUSTER, False, maxt,
            quant, prepushed, opts, cap, cw, cap_eps,
        )  # fmt: skip
        torch.cuda.synchronize()
        results.append((outs, buf))
        dist.barrier()
    (plain, _), (capped, buf) = results
    where = f"{kind} opts {opts} bucket {maxt} rows {m} scale {scale} quant {quant}"
    assert len(plain) == len(capped), where
    for i, (a, b) in enumerate(zip(plain, capped)):
        if i == 3:
            written = sf_written(m, HIDDEN // 32, a.device)
            a, b = a[written], b[written]
        _assert_bitwise(b, a, f"{where}: output {i} with vs without cap_out")
    ref = torch.empty((m, HIDDEN), dtype=torch.bfloat16, device="cuda")
    _capture_reference(capped[1], cap_w, cap_eps, ref)
    torch.cuda.synchronize()
    _assert_bitwise(buf[:m, HIDDEN : 2 * HIDDEN], ref, f"{where}: capture vs capture_norm")
    untouched = buf.clone()
    untouched[:m, HIDDEN : 2 * HIDDEN] = 7.0
    assert bool((untouched == 7.0).all()), f"{where}: bytes outside the capture block changed"
    dist.barrier()


def _cases():
    """(kind, rows, bucket, opts, scale, quant)."""
    cases = []
    scales = (1.0, 1e3, 1e-3)
    # The cluster kernel in the oneshot order (plain calls up to 8 rows): options 39 and 47.
    for i, m in enumerate((1, 4, 8)):
        for opts in (39, 47):
            cases.append(("x" if i != 1 else "x2", m, 8, opts, scales[i], opts == 47))
    cases.append(("prepushed", 4, 8, 39, 1.0, False))
    # The row kernel: plain options 7 / 15 and the carveout instance (+ 64).
    for i, (m, opts) in enumerate(((1, 7), (4, 15), (8, 7 | 64), (3, 15 | 64))):
        cases.append(("x2" if i % 2 else "x", m, 8, opts, scales[i % 3], i == 1))
    cases.append(("prepushed", 8, 8, 7, 1e-3, False))
    cases.append(("x", 12, 16, 15, 1.0, True))
    cases.append(("x2", 12, 16, 15 | 64, 1e3, False))
    cases.append(("x", 32, 48, 15, 1e-3, False))
    return cases


def _run_matrix() -> None:
    for n, (kind, m, maxt, opts, scale, quant) in enumerate(_cases()):
        _one_case(kind, m, maxt, opts, scale, quant, 7000 + 37 * n)
        if RANK == 0:
            print(
                f"[case] {kind} opts {opts} bucket {maxt} rows {m} scale {scale}: bitwise",
                flush=True,
            )


def _run_one_rank() -> int:
    global RANK, WORLD, allreduce_entry, ar_module
    os.environ.setdefault("TLLM_DISABLE_MPI", "1")
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(rank)
    dist.init_process_group(
        "cpu:gloo,cuda:nccl", init_method="env://", device_id=torch.device("cuda", rank)
    )
    RANK, WORLD = rank, world
    assert WORLD == 4, f"the tp4 boundary runs at 4 ranks, got {WORLD}"
    try:
        import tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_nvfp4__sm_107__tp4.ops.m3_mc_allreduce as _ar  # noqa: E501
        from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_nvfp4__sm_107__tp4.ops.comm import (  # noqa: E501
            m3_mc_allreduce_norm as _plain,
        )

        allreduce_entry, ar_module = _plain, _ar
        for maxt in BUCKETS:
            WORKSPACES[maxt] = _make_workspace(maxt)
        _run_matrix()
    except BaseException:
        import traceback  # only the rank subprocess's failure path needs it

        print(f"[rank {RANK}] FAILED", flush=True)
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(13)
    dist.barrier()
    torch.cuda.synchronize()
    print(f"[rank {RANK}] passed", flush=True)
    sys.stdout.flush()
    os._exit(0)


def _spawn_ranks() -> None:
    """One worker per claimed device, each in its own session; on the first nonzero exit or the deadline every
    session is killed (as ``_m3_draft_ar_norm_op_matrix.py`` does)."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    assert visible, "set CUDA_VISIBLE_DEVICES to the 4 devices this run owns"
    world_size = len([d for d in visible.split(",") if d.strip()])
    assert world_size == 4, (
        f"CUDA_VISIBLE_DEVICES names {world_size} device(s); the matrix runs at 4"
    )
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
    children = [
        subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()), _WORKER_FLAG],
            env=dict(env, RANK=str(r), LOCAL_RANK=str(r)),
            start_new_session=True,
        )
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


if __name__ == "__main__":
    if _WORKER_FLAG in sys.argv:
        sys.exit(_run_one_rank())
    _spawn_ranks()
    print("OK")
