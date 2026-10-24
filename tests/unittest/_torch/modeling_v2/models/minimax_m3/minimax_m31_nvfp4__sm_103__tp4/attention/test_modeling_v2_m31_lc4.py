# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for LC4: this target's copy of the long-context sparse decode
(``torch.ops.trtllm.minimax_m31_tp4_sparse_decode_lc``) with its two exact options, block-table staging (btsmem) and
the softmax sync trim (sync3), must write the same bits as the M3 op (``minimax_m3_sparse_decode_lc``): the BF16 rows,
the MXFP8 hand-off and its UE8M0 scales, for every option combination over the certified cells' shapes, a row longer
than the staged block table, repeated runs and a CUDA-graph replay.
"""

import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m31_nvfp4__sm_103__tp4.ops.attention import (  # noqa: E501
    m3_sparse_decode_lc as entry,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m31_nvfp4__sm_103__tp4.ops.attention.m3_sparse_decode_lc import (  # noqa: E501
    WIDTH,
)

__extra_import_path__ = ["../../../..", "."]
from _m3_nvfp4_cache import build_step  # noqa: E402 — needs the path declared above

assert torch.cuda.is_available(), "LC4 requires a CUDA device"

_HANDOFF_ROWS = 128
_OPTIONS = ((True, False), (False, True), (True, True))


def _lc4():
    op = getattr(torch.ops.trtllm, "minimax_m31_tp4_sparse_decode_lc", None)
    if op is None and entry._lc4_op() is not None:
        op = torch.ops.trtllm.minimax_m31_tp4_sparse_decode_lc
    if op is None:
        pytest.skip("minimax_m31_tp4_sparse_decode_lc is not built")
    return op


def _run(fn, case, linear_softmax, mxfp8, *opts):
    total_q = case.q.shape[0]
    out = torch.full((total_q, WIDTH), float("nan"), dtype=torch.bfloat16, device="cuda")
    q8 = torch.zeros(_HANDOFF_ROWS, WIDTH, dtype=torch.uint8, device="cuda") if mxfp8 else None
    sf = (
        torch.zeros(_HANDOFF_ROWS * (WIDTH // 32), dtype=torch.uint8, device="cuda")
        if mxfp8
        else None
    )
    fn(
        case.q.view(torch.uint8),
        case.k_paged,
        case.v_paged,
        case.k_block_scale,
        case.v_block_scale,
        case.k_global_scale,
        case.v_global_scale,
        case.topk[0],
        case.block_table,
        case.seq_lens,
        out,
        q8,
        sf,
        WIDTH // 128,
        case.decode_query_len,
        case.sm_scale,
        linear_softmax,
        False,
        True,
        *opts,
    )
    return out, q8, sf


def _assert_same(a, b, what):
    assert torch.equal(a[0].view(torch.int16), b[0].view(torch.int16)), f"{what}: BF16 rows differ"
    if a[1] is not None:
        assert torch.equal(a[1], b[1]), f"{what}: MXFP8 rows differ"
        assert torch.equal(a[2], b[2]), f"{what}: UE8M0 scales differ"


@pytest.mark.parametrize(
    "cell",
    entry.m3_sparse_decode_lc.CELLS,
    ids=[c.why[:44] for c in entry.m3_sparse_decode_lc.CELLS],
)
@pytest.mark.parametrize("linear_softmax", (False, True))
@pytest.mark.parametrize("mxfp8", (False, True))
@pytest.mark.parametrize("btsmem,sync3", _OPTIONS)
def test_lc4_matches_the_m3_op_bit_for_bit(cell, linear_softmax, mxfp8, btsmem, sync3) -> None:
    lc4 = _lc4()
    spec = cell.spec
    case = build_step(
        spec["seq_lens"],
        spec["decode_query_len"],
        identity=False,
        seed=zlib.crc32(cell.why.encode()) % 2**31,
    )
    ref = _run(torch.ops.trtllm.minimax_m3_sparse_decode_lc, case, linear_softmax, mxfp8)
    got = _run(lc4, case, linear_softmax, mxfp8, btsmem, sync3)
    _assert_same(got, ref, f"btsmem={btsmem} sync3={sync3}")


def test_lc4_row_longer_than_the_staged_block_table() -> None:
    """A request past 8192 blocks reads its block table from global memory under btsmem: same bits."""
    lc4 = _lc4()
    case = build_step((8193 * 128 + 5,), 1, identity=False, seed=31)
    ref = _run(torch.ops.trtllm.minimax_m3_sparse_decode_lc, case, False, True)
    for btsmem, sync3 in _OPTIONS:
        _assert_same(
            _run(lc4, case, False, True, btsmem, sync3),
            ref,
            f"long row btsmem={btsmem} sync3={sync3}",
        )


def test_lc4_graph_replay_and_repeats_are_bit_identical() -> None:
    """Three eager runs and a CUDA-graph replay of the shipped placement (both options, hand-off on) agree."""
    lc4 = _lc4()
    case = build_step((33000,), 4, identity=False, seed=21)
    first = _run(lc4, case, False, True, True, True)
    for _ in range(2):
        _assert_same(_run(lc4, case, False, True, True, True), first, "repeat")
    total_q = case.q.shape[0]
    out = torch.empty(total_q, WIDTH, dtype=torch.bfloat16, device="cuda")
    q8 = torch.zeros(_HANDOFF_ROWS, WIDTH, dtype=torch.uint8, device="cuda")
    sf = torch.zeros(_HANDOFF_ROWS * (WIDTH // 32), dtype=torch.uint8, device="cuda")
    args = (
        case.q.view(torch.uint8),
        case.k_paged,
        case.v_paged,
        case.k_block_scale,
        case.v_block_scale,
        case.k_global_scale,
        case.v_global_scale,
        case.topk[0],
        case.block_table,
        case.seq_lens,
        out,
        q8,
        sf,
        WIDTH // 128,
        case.decode_query_len,
        case.sm_scale,
        False,
        True,
        True,
        True,
        True,
    )
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        lc4(*args)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            lc4(*args)
    out.fill_(float("nan"))
    q8.zero_()
    sf.zero_()
    graph.replay()
    torch.cuda.synchronize()
    _assert_same((out, q8, sf), first, "graph replay")


def test_btsmem_is_scoped_to_small_steps() -> None:
    """btsmem costs larger grids their occupancy: the entry enables it only up to LC4_BTSMEM_MAX_ROWS rows."""
    assert entry.LC4_BTSMEM_MAX_ROWS == 8
    if entry.LC4_BTSMEM:
        assert entry._lc4_btsmem(4) and entry._lc4_btsmem(8)
    assert not entry._lc4_btsmem(9) and not entry._lc4_btsmem(32)
