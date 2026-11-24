# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the helix-layout DFlash2 window on the CuTe DSL MLA decode: every rank's partial,
stats and empty mask against the FP32 reference over FP8 history, then the same call captured in a
CUDA graph and replayed bitwise, with the lengths changed in place between replays."""

import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Cell, true_fp32_matmul
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.spec.rubin_ull_dflash2_helix_window import (  # noqa: E501
    helix_window_workspace,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.spec.rubin_ull_dflash2_helix_window import (  # noqa: E501
    rubin_ull_dflash2_helix_window as op,
)

__extra_import_path__ = ["../../../.."]
from _cells import certified_cells  # noqa: E402 — needs the path declared above
from _validating import validating  # noqa: E402 — needs the path declared above

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability() not in ((10, 0), (10, 3), (10, 7)),
    reason="the CuTe DSL MLA decode (Blackwell, Rubin)",
)


def _helix8(why: str, lengths: tuple, far: bool = False, block: int = 6) -> Cell:
    """helix8's drafter window (``draft_window_mla``): CP 8, block 6 (k5) unless named, 96 heads."""
    return Cell(why=why, spec=dict(lengths=lengths, block=block, cp=8, heads=96, far=far))


TARGET_CELLS: dict[str, tuple[Cell, ...]] = {
    "kimi_k31/kimi_k31_pro_nvfp4__sm_107__helix8": (
        _helix8("helix8 b2: CP8, block 6, a short and a long history", (4097, 130001)),
        _helix8(
            "helix8 b4 around the window: L = W - 1, W + 6, 8 W, 128 K", (1023, 1030, 8192, 131072)
        ),
        _helix8(
            "helix8 b9, a 5-layer interleaved pool past 2**31 bytes",
            (40000, 51234, 60001, 70000, 80123, 90000, 100001, 110000, 150000),
            far=True,
        ),
        # draft_window_mla_bands' upper band (from 70 rows): b13 at k5, b13 at k7 (block 8).
        _helix8(
            "helix8 b13, 78 rows (the upper band)",
            tuple(40000 + 9001 * i for i in range(13)),
        ),
        _helix8(
            "helix8 k7 b13, block 8, 104 rows",
            tuple(1017 + 11003 * i for i in range(13)),
            block=8,
        ),
    )
}
CELLS = certified_cells(op, TARGET_CELLS)

PAGE, WINDOW, DIM = 64, 1024, 576
SCALE = 0.13086080


def _build(spec, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    lengths = torch.tensor(spec["lengths"], dtype=torch.int64, device="cuda")
    batch, block, cp, heads = lengths.numel(), spec["block"], spec["cp"], spec["heads"]
    span = -(-(int(lengths.max()) + block) // (PAGE * cp)) * PAGE * cp
    history = (0.2 * torch.randn(span, DIM, generator=g, device="cuda")).to(torch.float8_e4m3fn)
    block_kv = (0.2 * torch.randn(batch * block, 1, DIM, generator=g, device="cuda")).to(
        torch.float8_e4m3fn
    )
    query = torch.randn(batch * block, heads, DIM, generator=g, device="cuda").to(torch.bfloat16)
    # Request b's table lives in row 2 b + 1: the call reads it through table_rows.
    table_rows = torch.arange(1, 2 * batch, 2, device="cuda")
    return lengths, history, block_kv, query, table_rows


# far cells: this rank's pages sit at FAR and up in a pool whose pages interleave 5 layers (the draft
# cache's layout), so page * page stride passes 2**31 bytes.
FAR, LAYERS = 12000, 5


def _rank(history, cp, rank, batch, far=False):
    local = history.view(-1, cp, PAGE, DIM)[:, rank]
    if not far:
        pool, first = local.contiguous(), 0
    else:
        big = torch.zeros(FAR + local.shape[0], LAYERS, PAGE, DIM, dtype=local.dtype, device="cuda")
        big[FAR:, 2] = local
        pool, first = big[:, 2], FAR
        assert (first * pool.stride(0)) * pool.element_size() >= 2**31
    table = first + torch.arange(local.shape[0], dtype=torch.int32, device="cuda")
    return pool, table.expand(2 * batch, -1).contiguous()


@pytest.mark.parametrize("cell", CELLS, ids=[c.why[:44] for c in CELLS])
@torch.inference_mode()
def test_certified_cells(cell) -> None:
    spec = cell.spec
    lengths, history, block_kv, query, table_rows = _build(
        spec, seed=zlib.crc32(cell.why.encode()) % 2**31
    )
    batch, block, cp, heads = lengths.numel(), spec["block"], spec["cp"], spec["heads"]
    rows = batch * block
    for rank in range(cp):
        pool, table = _rank(history, cp, rank, batch, spec["far"])
        ws = helix_window_workspace(rows, heads, PAGE, WINDOW, cp, query.device)
        args = (query, pool, table, table_rows, lengths, block_kv, block, SCALE, cp, rank, ws)
        with true_fp32_matmul():
            want, want_stats, want_empty = op.reference(*args)
        with validating(op):
            got, stats, empty = op(*args)
        torch.cuda.synchronize()
        assert torch.equal(empty, want_empty), rank
        live = ~want_empty
        if live.any():
            err = torch.linalg.vector_norm(got[live].float() - want[live])
            assert err / torch.linalg.vector_norm(want[live]) < 0.01, (rank, err)
            torch.testing.assert_close(stats[live], want_stats[live], rtol=2e-3, atol=2e-3)


@torch.inference_mode()
def test_graph_replay_reads_the_lengths() -> None:
    spec = CELLS[0].spec | {"lengths": (1023, 1024, 8192, 131072)}
    lengths, history, block_kv, query, table_rows = _build(spec, seed=7)
    batch, block, cp, heads = lengths.numel(), spec["block"], spec["cp"], spec["heads"]
    pool, table = _rank(history, cp, 1, batch)
    ws = helix_window_workspace(batch * block, heads, PAGE, WINDOW, cp, query.device)
    args = (query, pool, table, table_rows, lengths, block_kv, block, SCALE, cp, 1, ws)
    eager, eager_stats, _ = op(*args)
    eager, eager_stats = eager.clone(), eager_stats.clone()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out, out_stats, _ = op(*args)
    saved = lengths.clone()
    lengths.zero_()
    graph.replay()
    zero, zero_stats, _ = op(*args)
    zero, zero_stats = zero.clone(), zero_stats.clone()
    graph.replay()
    torch.testing.assert_close(out, zero, rtol=0, atol=0)
    torch.testing.assert_close(out_stats, zero_stats, rtol=0, atol=0)
    lengths.copy_(saved)
    graph.replay()
    torch.testing.assert_close(out, eager, rtol=0, atol=0)
    torch.testing.assert_close(out_stats, eager_stats, rtol=0, atol=0)
