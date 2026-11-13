# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The dep2ctx target's minimax_m31_route_stage (DEP2CTX_NORMQ_STAGE's route-only stage op) bit for bit vs the routing
columns noaux_tc_fp4_quantize_stage_out writes for the same logits: the live rows' top-k ids and weights (both expert
tiers, a masked 144-slot layout, ties, strided logits), -1 / 0 on pad-mask rows, the appended slot / 1.0 in column
topk (SeAppendRouting.stage_appended), nothing past the live rows or the routing columns, and CUDA graph replay with
PDL. GPU (SM107)."""

import pytest
import torch

import tensorrt_llm  # noqa: F401  (registers the trtllm ops)
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_1_flash_nvfp4__sm_107__dep2ctx import (  # noqa: E501
    native_ops,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.native import family_ops
from tensorrt_llm._utils import get_sm_version

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or get_sm_version() != 107, reason="SM107 (Rubin) only"
)

HIDDEN = 64
RSF = 2.5
IDX_SENT = 12345
W_SENT = 7.5


def _inputs(rows, e, seed, ties=False, stride_pad=0):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    wide = torch.randn(max(rows, 1), e + stride_pad, generator=gen, device="cuda") * 2.0
    logits = wide[:rows, :e]
    bias = 0.05 * torch.randn(e, generator=gen, device="cuda")
    if ties:
        logits[:, ::3] = logits[:, :1]
        bias.zero_()
    if e == 144:
        bias[16::18] = -1.0e30
        bias[17::18] = -1.0e30
    return logits, bias


def _stage(cap, cols):
    return (
        torch.full((cap, cols), IDX_SENT, dtype=torch.int32, device="cuda"),
        torch.full((cap, cols), W_SENT, dtype=torch.float32, device="cuda"),
    )


def _reference(logits, bias, topk, cap, appended, pad):
    """noaux_tc_fp4_quantize_stage_out on the same logits (+ stage_appended's column on the live rows)."""
    rows = logits.shape[0]
    cols = topk + (1 if appended >= 0 else 0)
    ids, w = _stage(cap, cols)
    hidden = torch.zeros(rows, HIDDEN, dtype=torch.bfloat16, device="cuda")
    gs = torch.ones(1, device="cuda")
    fp4 = torch.empty(cap, HIDDEN // 2, dtype=torch.uint8, device="cuda")
    sf = torch.empty(cap, HIDDEN // 16, dtype=torch.uint8, device="cuda")
    empty = (
        torch.empty(0, topk, dtype=torch.int32, device="cuda"),
        torch.empty(0, topk, dtype=torch.float32, device="cuda"),
        torch.empty(0, HIDDEN // 2, dtype=torch.uint8, device="cuda"),
        torch.empty(0, HIDDEN // 16, dtype=torch.uint8, device="cuda"),
    )
    kw = {} if pad is None else {"pad_mask": pad}
    family_ops().noaux_tc_fp4_quantize_stage_out(
        logits.contiguous(),
        bias,
        hidden,
        gs,
        topk,
        RSF,
        rows,
        ids[:, :topk],
        w[:, :topk],
        fp4,
        sf,
        *empty,
        **kw,
    )
    if appended >= 0:
        ids[:rows, topk] = appended
        w[:rows, topk] = 1.0
    return ids, w


def _route_stage(logits, bias, topk, cap, appended, pad, pdl=False):
    ids, w = _stage(cap, topk + (1 if appended >= 0 else 0))
    native_ops.minimax_m31_route_stage(logits, bias, pad, topk, RSF, appended, ids, w, pdl)
    return ids, w


def _assert_live_rows_equal(ref, got, rows):
    assert torch.equal(ref[0][:rows], got[0][:rows]), "ids"
    assert torch.equal(ref[1][:rows].view(torch.int32), got[1][:rows].view(torch.int32)), "weights"
    # nothing past the live rows (the boundary owns the tail)
    assert bool((got[0][rows:] == IDX_SENT).all()) and bool((got[1][rows:] == W_SENT).all())


def _pad_mask(rows, pattern):
    if pattern is None:
        return None
    pm = torch.zeros(rows, dtype=torch.uint8, device="cuda")
    if pattern == "all":
        pm.fill_(1)
    else:
        pm[rows // 2 :: 3] = 1
    return pm


@pytest.mark.parametrize("e, topk", [(128, 4), (128, 8), (144, 4), (144, 8), (256, 8)])
@pytest.mark.parametrize("rows", [1, 3, 47, 192, 2048, 8193])
@pytest.mark.parametrize("ties", [False, True])
def test_route_stage_matches_the_fused_stage(e, topk, rows, ties):
    logits, bias = _inputs(rows, e, 100 * e + rows + topk, ties=ties)
    cap = rows + 37
    ref = _reference(logits, bias, topk, cap, -1, None)
    got = _route_stage(logits, bias, topk, cap, -1, None)
    torch.cuda.synchronize()
    _assert_live_rows_equal(ref, got, rows)


@pytest.mark.parametrize("pattern", ["some", "all"])
@pytest.mark.parametrize("appended", [-1, 35])
@pytest.mark.parametrize("rows", [7, 129, 4100])
def test_pad_mask_and_appended_slot(pattern, appended, rows):
    e, topk = 144, 4
    logits, bias = _inputs(rows, e, 7 + rows)
    pm = _pad_mask(rows, pattern)
    cap = rows + 64
    ref = _reference(logits, bias, topk, cap, appended, pm)
    got = _route_stage(logits, bias, topk, cap, appended, pm)
    torch.cuda.synchronize()
    _assert_live_rows_equal(ref, got, rows)


def test_strided_logits():
    rows, e, topk = 333, 128, 8
    logits, bias = _inputs(rows, e, 5, stride_pad=40)
    assert logits.stride(0) == e + 40
    ref = _reference(logits, bias, topk, rows, -1, None)
    got = _route_stage(logits, bias, topk, rows, -1, None)
    torch.cuda.synchronize()
    _assert_live_rows_equal(ref, got, rows)


def test_zero_rows_is_a_no_op():
    logits, bias = _inputs(0, 128, 1)
    got = _route_stage(logits, bias, 4, 16, -1, None)
    torch.cuda.synchronize()
    assert bool((got[0] == IDX_SENT).all()) and bool((got[1] == W_SENT).all())


@pytest.mark.parametrize("rows", [88, 4096])
def test_cuda_graph_replay_with_pdl(rows):
    e, topk = 128, 4
    logits, bias = _inputs(rows, e, 21 + rows)
    pm = _pad_mask(rows, "some")
    ref = _reference(logits, bias, topk, rows + 8, 35, pm)
    ids, w = _stage(rows + 8, topk + 1)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        native_ops.minimax_m31_route_stage(logits, bias, pm, topk, RSF, 35, ids, w, True)
    for _ in range(3):
        ids.fill_(IDX_SENT)
        w.fill_(W_SENT)
        graph.replay()
        torch.cuda.synchronize()
        _assert_live_rows_equal(ref, (ids, w), rows)


def test_rejects_bad_arguments():
    rows, e, topk = 64, 128, 4
    logits, bias = _inputs(rows, e, 3)
    small = _stage(32, topk)
    with pytest.raises(RuntimeError):
        native_ops.minimax_m31_route_stage(logits, bias, None, topk, RSF, -1, *small, False)
    ids, w = _stage(rows, topk)
    with pytest.raises(RuntimeError):  # no room for the appended column
        native_ops.minimax_m31_route_stage(logits, bias, None, topk, RSF, 3, ids, w, False)
    with pytest.raises(RuntimeError):  # bf16 logits
        native_ops.minimax_m31_route_stage(
            logits.to(torch.bfloat16), bias, None, topk, RSF, -1, ids, w, False
        )
    with pytest.raises(RuntimeError):  # bias size != E
        native_ops.minimax_m31_route_stage(logits, bias[:-1], None, topk, RSF, -1, ids, w, False)
    with pytest.raises(RuntimeError):  # topk > 32
        big = _stage(rows, 33)
        native_ops.minimax_m31_route_stage(logits, bias, None, 33, RSF, -1, *big, False)
    one_ids, one_w = _stage(1, topk)
    with pytest.raises(RuntimeError, match="rows not overlapping"):  # expanded (stride-0) outputs
        native_ops.minimax_m31_route_stage(
            logits,
            bias,
            None,
            topk,
            RSF,
            -1,
            one_ids.expand(rows, -1),
            one_w.expand(rows, -1),
            False,
        )


def test_dep2ctx_row_b_sees_the_route_stage_op():
    """The dep2ctx boundary's stage gate (ops_built) now finds both of its ops on an sm_10x build."""
    from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_1_flash_nvfp4__sm_107__dep2ctx import (  # noqa: E501
        modeling,
    )

    modeling._NORMQ_V2_STATE.clear()
    try:
        assert modeling._route_stage_op() is not None
        assert modeling._normq_v2_out_op() is not None
    finally:
        modeling._NORMQ_V2_STATE.clear()
