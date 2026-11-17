# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for LC4: this target's copy of the long-context sparse decode
(``native_ops.minimax_m31_tp4_sparse_decode_lc``) with its two exact options, block-table staging (btsmem) and
the softmax sync trim (sync3), must write the same bits as the M3 op (``minimax_m3_sparse_decode_lc``): the BF16 rows,
the MXFP8 hand-off and its UE8M0 scales, for every option combination on the shortest and longest certified cells
and at the shipped placement on the rest, a row longer than the staged block table, repeated runs and a CUDA-graph
replay.
"""

import zlib

import pytest
import torch

from tensorrt_llm._torch.modeling_v2._native import NoTarget

try:
    from tensorrt_llm._torch.modeling_v2.models.minimax_m3.minimax_m3_1_flash_nvfp4__sm_103__tp4 import (
        native_ops,
    )
    from tensorrt_llm._torch.modeling_v2.models.minimax_m3.minimax_m3_1_flash_nvfp4__sm_103__tp4.ops.attention import (  # noqa: E501
        m3_sparse_decode_lc as entry,
    )
    from tensorrt_llm._torch.modeling_v2.models.minimax_m3.minimax_m3_1_flash_nvfp4__sm_103__tp4.ops.attention.m3_sparse_decode_lc import (  # noqa: E501
        WIDTH,
    )
except NoTarget as no_target:
    pytest.skip(str(no_target), allow_module_level=True)

__extra_import_path__ = ["../../../..", "."]
from _m3_nvfp4_cache import build_step  # noqa: E402 — needs the path declared above

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 3),
    reason="LC4 runs on sm_103 (GB300)",
)

_HANDOFF_ROWS = 128
_OPTIONS = ((True, False), (False, True), (True, True))


def _lc4():
    op = getattr(native_ops, "minimax_m31_tp4_sparse_decode_lc", None)
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


def _bit_for_bit_cases():
    """The full option matrix on the shortest and the longest cell (by KV tokens); every other cell at the shipped
    placement (log-domain softmax, MXFP8 hand-off, btsmem and sync3 on)."""
    cells = entry.m3_sparse_decode_lc.CELLS
    kv = [sum(c.spec["seq_lens"]) for c in cells]
    full = {kv.index(min(kv)), kv.index(max(kv))}
    cases = []
    for i, cell in enumerate(cells):
        combos = (
            [
                (ls, mx, bt, s3)
                for ls in (False, True)
                for mx in (False, True)
                for bt, s3 in _OPTIONS
            ]
            if i in full
            else [(False, True, True, True)]
        )
        for ls, mx, bt, s3 in combos:
            cases.append(
                pytest.param(cell, ls, mx, bt, s3, id=f"{cell.why[:44]}-{ls}-{mx}-{bt}-{s3}")
            )
    return cases


@pytest.mark.parametrize("cell,linear_softmax,mxfp8,btsmem,sync3", _bit_for_bit_cases())
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
