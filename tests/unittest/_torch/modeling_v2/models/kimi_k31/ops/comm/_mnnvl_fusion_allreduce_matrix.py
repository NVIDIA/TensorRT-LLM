# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Certify the K3.1 ULL target's ``ops.comm.mnnvl_fusion_allreduce`` on one GB300 tray (4 ranks).

Its own launcher (``_k31_ull_comm_ranks``): ``python _mnnvl_fusion_allreduce_matrix.py`` with
``CUDA_VISIBLE_DEVICES`` naming 4 devices. Each rank builds the workspaces the way
``comm.FlatAllReduce`` does (a private ``McastGPUBuffer`` of three Lamport buffers, FP32 -0.0, flags
``[0, 2, bytes, 0...]``), one per dtype, and drives every cell of ``mnnvl_fusion_allreduce.CELLS``
several times in a row so the rotation re-arms every buffer.

The plain sums are gated bitwise: the inputs are multiples of 1/8 whose sums are exact in any order.
The fused norms are gated in BF16 ulp of the row scale (5 per element, 1.25 relative RMS:
``comm/mnnvl_allreduce``'s band for the same kernel).
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.distributed as dist

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _k31_ull_comm_ranks as ranks  # noqa: E402

DEADLINE_S = 600
MAX_ROWS = 64
"""The target's largest served batch: 8 requests x next_n 8."""
CALLS_PER_CELL = 4
"""More than the three Lamport buffers: every buffer is read after a re-arm."""

RANK = 0
WORLD = 1
WORKSPACES: dict = {}
op = None
entry = None
validating = None
within_ulp = None


def _workspace(mapping, dtype: torch.dtype, hidden: int):
    """``comm.FlatAllReduce.__init__``'s workspace, over the TP process group instead of MPI."""
    import math

    from tensorrt_llm._torch.distributed.ops import (
        _MNNVL_ONE_SHOT_THRESHOLD_BYTES,
        _make_mnnvl_mcast_buffer,
    )

    element = torch.empty((), dtype=dtype).element_size()
    need = max(entry.lamport_bytes(r, hidden, WORLD, element) for r in range(1, MAX_ROWS + 1))
    need = max(need, _MNNVL_ONE_SHOT_THRESHOLD_BYTES)
    capacity = math.ceil(need / (8 * 1024**2)) * (8 * 1024**2)
    handle = _make_mnnvl_mcast_buffer(
        mapping.tp_group_pg, 3 * capacity, mapping, mapping.is_multi_node()
    )
    raw = handle.get_uc_buffer(mapping.tp_rank, (3 * capacity // 4,), torch.float32, 0)
    raw.fill_(-0.0)
    torch.cuda.synchronize()
    dist.barrier()
    flags = torch.tensor([0, 2, capacity, 0, 0, 0, 0, 0, 0], dtype=torch.uint32, device=raw.device)
    return dict(comm_buffer=raw.view(dtype).view(3, -1), buffer_flags=flags, handle=handle)


def _args(spec):
    ws = WORKSPACES[spec["dtype"]]
    return ws["comm_buffer"], ws["buffer_flags"]


def _inputs(spec, seed: int, rank: int):
    rows, hidden, dtype = spec["rows"], spec["hidden"], spec["dtype"]
    if spec["fusion"] == entry.NONE:
        return ranks.eighths((rows, hidden), seed * 101 + rank, 32, dtype)
    return ranks.normal((rows, hidden), seed * 101 + rank, 1.0, dtype)


def _norm_operands(spec, seed: int):
    rows, hidden, dtype = spec["rows"], spec["hidden"], spec["dtype"]
    if spec["fusion"] == entry.NONE:
        return None, None
    weight = (1 + ranks.normal((hidden,), seed + 7, 0.1, torch.float32)).to(dtype)
    residual = None
    if spec["fusion"] == entry.RESIDUAL_RMS_NORM:
        residual = ranks.normal((rows, hidden), seed + 9, 2.0, dtype)
    return weight, residual


def _call(spec, x, weight, residual, comm_buffer=None):
    buffer, flags = _args(spec)
    return op(
        x,
        weight,
        residual,
        spec["eps"] if weight is not None else 0.0,
        buffer if comm_buffer is None else comm_buffer,
        flags,
        spec["fusion"],
        WORLD,
    )


def _check(spec, out, seed, weight, residual, where):
    inputs = [_inputs(spec, seed, r) for r in range(WORLD)]
    ref = op.reference(inputs, weight, residual, spec["eps"], spec["fusion"])
    if spec["fusion"] == entry.NONE:
        ranks.assert_bitwise(out, ref, where)
        return
    outs = (out,) if spec["fusion"] == entry.RMS_NORM else out
    refs = (ref,) if spec["fusion"] == entry.RMS_NORM else ref
    assert len(outs) == len(refs), f"{where}: {len(outs)} outputs, expected {len(refs)}"
    for o, r in zip(outs, refs):
        try:
            within_ulp(o, r, element_ulp=5.0, rms_ulp=1.25)
        except AssertionError as exc:
            raise AssertionError(f"{where}: {exc}") from exc


def check_certified_cells() -> None:
    """Every cell, guarded, ``CALLS_PER_CELL`` times in a row against the reference; each call
    advances the rotation by one buffer."""
    for i, cell in enumerate(op.CELLS):
        spec = cell.spec
        for call in range(CALLS_PER_CELL):
            seed = 1000 * i + call
            weight, residual = _norm_operands(spec, seed)
            x = _inputs(spec, seed, RANK)
            flags = _args(spec)[1]
            before = int(flags[0].item())
            with validating(op):
                out = _call(spec, x, weight, residual)
            torch.cuda.synchronize()
            after = int(flags[0].item())
            assert after == (before + 1) % 3, f"{cell.why}: buffer {before} -> {after}"
            _check(spec, out, seed, weight, residual, f"{cell.why} (call {call})")
            dist.barrier()


def check_inputs_are_untouched() -> None:
    spec = next(c.spec for c in op.CELLS if c.spec["fusion"] == entry.RESIDUAL_RMS_NORM)
    weight, residual = _norm_operands(spec, 21)
    x = _inputs(spec, 21, RANK)
    saved = (x.clone(), residual.clone(), weight.clone())
    _call(spec, x, weight, residual)
    torch.cuda.synchronize()
    for a, b in zip(saved, (x, residual, weight)):
        ranks.assert_bitwise(b, a, "an input after the call")


def check_cuda_graph_replay() -> None:
    """Captured once per row count, replayed with fresh inputs: the rotation advances inside the
    graph, and every replay reads that replay's data (one-shot at 8 rows, two-shot at 32)."""
    for rows in (8, 32):
        spec = dict(rows=rows, hidden=entry.HIDDEN, dtype=torch.bfloat16, fusion=entry.NONE)
        spec["eps"] = entry.EPS
        x = _inputs(spec, 31, RANK)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(2):
                _call(spec, x, None, None)
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        dist.barrier()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = _call(spec, x, None, None)
        for rep in range(4):
            seed = 300 + 10 * rep + rows
            x.copy_(_inputs(spec, seed, RANK))
            torch.cuda.synchronize()
            dist.barrier()
            graph.replay()
            torch.cuda.synchronize()
            _check(spec, out, seed, None, None, f"graph replay rows={rows} rep={rep}")
            dist.barrier()


def check_the_gate_discriminates_a_wrong_reduction() -> None:
    spec = dict(rows=8, hidden=entry.HIDDEN, dtype=torch.bfloat16, fusion=entry.NONE, eps=0.0)
    out = _call(spec, _inputs(spec, 41, RANK), None, None)
    torch.cuda.synchronize()
    short = [_inputs(spec, 41, r) for r in range(WORLD - 1)]
    wrong = op.reference(short, None, None, 0.0, entry.NONE)
    try:
        ranks.assert_bitwise(out, wrong, "missing rank")
    except AssertionError:
        return
    raise AssertionError("the gate accepted a sum missing one rank")


def check_the_guard_refuses_a_call_past_the_workspace() -> None:
    """The kernel never checks the call against the buffer; the guard does, before any rank enters
    the collective."""
    spec = dict(rows=64, hidden=entry.HIDDEN, dtype=torch.bfloat16, fusion=entry.NONE, eps=0.0)
    small = _args(spec)[0][:, :1024]
    try:
        with validating(op):
            _call(spec, _inputs(spec, 51, RANK), None, None, comm_buffer=small)
    except ValueError as exc:
        assert "Lamport buffer" in str(exc), exc
    else:
        raise AssertionError("the guard admitted a call larger than the workspace")


CHECKS = (
    check_certified_cells,
    check_inputs_are_untouched,
    check_cuda_graph_replay,
    check_the_gate_discriminates_a_wrong_reduction,
    check_the_guard_refuses_a_call_past_the_workspace,
)


def _worker() -> None:
    global RANK, WORLD, op, entry, validating, within_ulp
    RANK, WORLD, mapping = ranks.init_rank()
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.comm import (
        rubin_ull_mnnvl_fusion_allreduce as entry_module,  # noqa: E501
    )

    entry = entry_module
    op = entry_module.mnnvl_fusion_allreduce
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
    from _validating import assert_within_ulp
    from _validating import validating as validating_cm

    validating, within_ulp = validating_cm, assert_within_ulp
    for dtype in sorted({c.spec["dtype"] for c in op.CELLS}, key=str):
        hidden = max(c.spec["hidden"] for c in op.CELLS if c.spec["dtype"] == dtype)
        WORKSPACES[dtype] = _workspace(mapping, dtype, hidden)
    ranks.run_checks(CHECKS, RANK)


if __name__ == "__main__":
    if ranks.WORKER_FLAG in sys.argv:
        _worker()
    else:
        ranks.spawn_ranks(Path(__file__).resolve(), DEADLINE_S)
