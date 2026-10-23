# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the qsa_paged_index_scores_verify entry: every cell's visible columns bitwise
the row-wise `qsa_paged_index_scores`, on a verify step's layout (each request's rows
consecutive at consecutive positions) over the engine's table (request slots drawn from a
permutation, shuffled pages, -1 past each sequence) and, on the NVFP4 cells, the index
plane as a view of the trunk's padded page; columns past a row's bound stay untouched; a
graph replay with PDL reads the queries after its wait."""

import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_103__tp1.ops.attention.qsa_paged_index_scores_verify import (  # noqa: E501
    qsa_paged_index_scores_verify as op,
)

__extra_import_path__ = ["../../../.."]
from _validating import certifying  # noqa: E402,F401 — path above; fixture by name

assert torch.cuda.is_available(), "qsa_paged_index_scores_verify requires a CUDA device"

DEV = "cuda"
IH, ID, TPB, CR, WIDTH = 4, 128, 32, 4, 324
#: The NVFP4 trunk's index page: 12 index buffers of TPB x ID, then the position page.
BUFFERS, PAD = 12, 192


def _build(spec, seed):
    g = torch.Generator(device=DEV).manual_seed(seed)
    starts, per = spec["starts"], spec["per"]
    rows = len(starts) * per
    slots = len(starts) + 3
    slot_of = torch.randperm(slots, generator=torch.Generator().manual_seed(seed))[: len(starts)]
    need = [(s + per - 1) // TPB + 1 for s in starts]
    pages = sum(need) + 5
    perm = torch.randperm(pages, generator=g, device=DEV).to(torch.int32)
    block_table = torch.full((slots, WIDTH), -1, dtype=torch.int32, device=DEV)
    used = 0
    for slot, n in zip(slot_of.tolist(), need):
        block_table[slot, :n] = perm[used : used + n]
        used += n
    keys = torch.randn(pages, TPB, 1, ID, generator=g, device=DEV).to(torch.bfloat16)
    if spec.get("nvfp4"):
        page = torch.zeros(pages, BUFFERS * TPB * ID + PAD, dtype=torch.bfloat16, device=DEV)
        index_cache = page[:, 5 * TPB * ID : 6 * TPB * ID].view(pages, TPB, 1, ID)
        index_cache.copy_(keys)
    else:
        index_cache = keys
    scratch = torch.randn(rows, IH + 2, ID, generator=g, device=DEV).to(torch.bfloat16)
    positions = torch.tensor(
        [s + i for s in starts for i in range(per)], dtype=torch.int64, device=DEV
    )
    requests = slot_of.to(DEV).repeat_interleave(per).to(torch.int32)
    return dict(
        q=scratch[:, :IH], index_cache=index_cache, block_table=block_table,
        query_positions=positions, request_indices=requests, tokens_per_block=TPB,
        compress_ratio=CR, rows_per_request=per,
    )  # fmt: skip


def _visible_equal(got, want, positions) -> None:
    for r, p in enumerate(positions.tolist()):
        n = (p + 1) // CR
        assert torch.equal(got[r, :n], want[r, :n]), f"row {r} (position {p})"


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 — fixture by name
    certifying(op)
    args = _build(cell.spec, seed=zlib.crc32(cell.why.encode()) % (2**31))
    want = op.reference(**args)
    got = op(**args)
    torch.cuda.synchronize()
    assert got.shape == want.shape and got.dtype == torch.float32
    _visible_equal(got, want, args["query_positions"])


def test_reads_the_queries_after_the_wait_under_graph_replay() -> None:
    args = _build(dict(starts=(8697, 5000, 1203), per=4), seed=5)
    src = args["q"].clone()
    work = torch.empty_like(src)
    args["q"] = work

    def step():
        work.copy_(src * 1.0)
        return op(**args, use_pdl=True)

    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        step()
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = step()
    for trial in range(3):
        src.copy_(torch.randn(src.shape, device=DEV).to(src.dtype))
        graph.replay()
        torch.cuda.synchronize()
        want = op.reference(**{**args, "q": src})
        _visible_equal(out, want, args["query_positions"])
