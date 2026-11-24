# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the head-parallel DFlash2 window decode over the replicated ring: every certified cell
against the FP32 reference; on identical inputs against today's helix_mla path (16 CP ranks' partials
merged by log-sum-exp), both bounded by the same reference; and the call captured in a CUDA graph,
replayed bitwise with the block starts changed in place."""

import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Cell, true_fp32_matmul
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.spec.rubin_ull_dflash2_headtp_decode import (  # noqa: E501
    headtp_decode_workspace,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.spec.rubin_ull_dflash2_headtp_decode import (  # noqa: E501
    rubin_ull_dflash2_headtp_decode as op,
)

__extra_import_path__ = ["../../../.."]
from _cells import certified_cells  # noqa: E402 — needs the path declared above
from _validating import validating  # noqa: E402 — needs the path declared above

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="Triton kernels need a GPU")

TARGET_CELLS: dict[str, tuple[Cell, ...]] = {}
CELLS = certified_cells(op, TARGET_CELLS)

PAGE, WINDOW, DIM, LAT = 64, 1024, 576, 512
SCALE = 0.13086080
_BLACKWELL = torch.cuda.is_available() and torch.cuda.get_device_capability() in ((10, 0), (10, 3))


def _rel(got, want):
    return (torch.linalg.vector_norm(got.float() - want) / torch.linalg.vector_norm(want)).item()


def _build(spec, seed):
    """Ring rows of every slot with random FP8 latents (what the stores leave), and the queries."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    groups, block, heads, ring_rows = (
        len(spec["starts"]),
        spec["block"],
        spec["heads"],
        spec["ring"],
    )
    slots_n = max(spec["slots"]) + 1
    dtype = torch.float8_e4m3fn if spec["dtype"] == "fp8" else torch.bfloat16
    ring = (0.2 * torch.randn(slots_n * ring_rows, DIM, generator=g, device="cuda")).to(dtype)
    query = torch.randn(groups * block, heads, DIM, generator=g, device="cuda").to(torch.bfloat16)
    slots = torch.tensor(spec["slots"], dtype=torch.int32, device="cuda")
    starts = torch.tensor(spec["starts"], dtype=torch.int64, device="cuda")
    return query, ring, slots, starts, ring_rows


@pytest.mark.parametrize("cell", CELLS, ids=[c.why[:44] for c in CELLS])
@torch.inference_mode()
def test_certified_cells(cell) -> None:
    spec = cell.spec
    query, ring, slots, starts, ring_rows = _build(spec, seed=zlib.crc32(cell.why.encode()) % 2**31)
    ws = headtp_decode_workspace(
        len(spec["starts"]), spec["block"], spec["heads"], WINDOW, query.device
    )
    args = (query, ring, slots, starts, ring_rows, SCALE, 1.0, ws)
    with true_fp32_matmul():
        want = op.reference(*args)
    with validating(op):
        got = op(*args)
    torch.cuda.synchronize()
    assert _rel(got, want) < 0.01


@pytest.mark.skipif(
    not _BLACKWELL, reason="today's helix_mla path: the CuTe DSL MLA decode (Blackwell)"
)
@pytest.mark.parametrize("length", [5, 700, 1023, 1024, 4133, 190000])
@torch.inference_mode()
def test_matches_helix_mla_on_identical_inputs(length) -> None:
    """One request at block start ``length``: the history and block latents laid out as today's 16
    ranks' pages (helix_mla: per-rank pack + MLA decode over 96 heads, merged here as the helix
    combine weighs them) and as the ring; the head-parallel decode of heads 0..5 must sit in the
    same envelope as helix_mla's heads 0..5 against the FP32 reference."""
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.spec.rubin_ull_dflash2_helix_window import (  # noqa: E501
        helix_window_workspace,
        rubin_ull_dflash2_helix_window,
    )

    cp, block, heads, ring_rows = 16, 8, 96, 1088
    g = torch.Generator(device="cuda").manual_seed(length)
    span = -(-(length + block) // (PAGE * cp)) * PAGE * cp
    history = (0.2 * torch.randn(span, DIM, generator=g, device="cuda")).to(torch.float8_e4m3fn)
    block_kv = (0.2 * torch.randn(block, 1, DIM, generator=g, device="cuda")).to(
        torch.float8_e4m3fn
    )
    query = torch.randn(block, heads, DIM, generator=g, device="cuda").to(torch.bfloat16)
    lengths = torch.tensor([length], dtype=torch.int64, device="cuda")
    table_rows = torch.zeros(1, dtype=torch.int64, device="cuda")

    # Today: every rank's partial over the keys it owns, merged.
    ws = helix_window_workspace(block, heads, PAGE, WINDOW, cp, query.device)
    parts, lses = [], []
    for rank in range(cp):
        pool = history.view(-1, cp, PAGE, DIM)[:, rank].contiguous()
        table = torch.arange(pool.shape[0], dtype=torch.int32, device="cuda").view(1, -1)
        part, stats, empty = rubin_ull_dflash2_helix_window(
            query, pool, table, table_rows, lengths, block_kv, block, SCALE, cp, rank, ws
        )
        lse = torch.where(empty.view(-1, 1), float("-inf"), stats[..., 0])
        parts.append(part.float().clone())
        lses.append(lse.clone())
    lse = torch.stack(lses)
    weight = torch.exp(lse - lse.max(0).values).nan_to_num(0.0)
    today = (torch.stack(parts) * weight[..., None]).sum(0) / weight.sum(0)[..., None]

    # A: the same latents in the ring (history positions < length, then the block's own).
    ring = torch.zeros(ring_rows, DIM, dtype=torch.float8_e4m3fn, device="cuda")
    pos = torch.arange(max(0, length - WINDOW), length, device="cuda")
    ring.view(torch.uint8)[pos % ring_rows] = history.view(torch.uint8)[pos]
    own = torch.arange(length, length + block, device="cuda")
    ring.view(torch.uint8)[own % ring_rows] = block_kv.view(torch.uint8)[:, 0]
    q6 = query[:, :6].contiguous()
    ws6 = headtp_decode_workspace(1, block, 6, WINDOW, query.device)
    args = (
        q6,
        ring,
        torch.zeros(1, dtype=torch.int32, device="cuda"),
        lengths,
        ring_rows,
        SCALE,
        1.0,
        ws6,
    )
    got = op(*args)
    with true_fp32_matmul():
        want = op.reference(*args)
    torch.cuda.synchronize()
    err_a, err_today, between = _rel(got, want), _rel(today[:, :6], want), _rel(got, today[:, :6])
    assert err_a < 0.01 and err_today < 0.01, (err_a, err_today)
    assert err_a < 2 * err_today + 1e-3, (err_a, err_today)  # A no worse than today's envelope
    assert between < 0.01, between


@torch.inference_mode()
def test_graph_replay_reads_the_starts() -> None:
    spec = CELLS[0].spec | {"starts": (1023, 1024, 8192, 131072), "slots": (0, 1, 2, 3)}
    query, ring, slots, starts, ring_rows = _build(spec, seed=7)
    ws = headtp_decode_workspace(4, spec["block"], spec["heads"], WINDOW, query.device)
    args = (query, ring, slots, starts, ring_rows, SCALE, 1.0, ws)
    eager = op(*args).clone()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = op(*args)
    saved = starts.clone()
    starts.fill_(3)
    graph.replay()
    small = op(*args).clone()
    graph.replay()
    torch.testing.assert_close(out, small, rtol=0, atol=0)
    starts.copy_(saved)
    graph.replay()
    torch.testing.assert_close(out, eager, rtol=0, atol=0)
