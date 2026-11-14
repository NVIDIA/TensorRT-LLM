# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU tests for TP2WIDE_LC2_SPLIT: mv2_minimax_m3::minimax_m3_sm107_tp2_sparse_decode_lc with ``allow_split=True``
runs LC2-split (LC2 without its thread-block cluster: a per-(row, block) grid storing LC2's partials, then a merge grid
in LC2's slot order); with ``allow_split=False`` it runs the cluster kernel. The target passes ``allow_split`` unless
TRTLLM_M31_DISABLE_TP2WIDE_LC2_SPLIT=1.

Contract: on every case of test_modeling_v2_m3_sparse_decode_lc.py (the M3 cases, the K3 / K7 wide steps to 448 rows,
the N1 cases), in every variant (log-domain / linear softmax, pre-wait staging, MXFP8 hand-off on / off, N1 on / off,
PDL on / off), the split writes the same BF16 rows, MXFP8 bytes, swizzled scales and N1 ids as the cluster kernel, bit
for bit, eagerly and replayed from a CUDA graph.
"""

import importlib.util
import os

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2._native import NoTarget

try:
    from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_1_flash_nvfp4__sm_107__tp2.ops.attention import (  # noqa: E501
        m3_sparse_decode_lc,
    )
except NoTarget as exc:
    pytest.skip(str(exc), allow_module_level=True)
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.native import family_ops

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7),
    reason="the sm_107 TP2 target runs on SM107 only",
)

# The sibling LC2 test's case builder (its own __extra_import_path__ resolves its _validating import).
_spec = importlib.util.spec_from_file_location(
    "lc2_cases", os.path.join(os.path.dirname(__file__), "test_modeling_v2_m3_sparse_decode_lc.py")
)
lc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(lc)

_ALL = {**lc._CASES, **lc._WIDE_CASES, **{f"n1-{k}": v for k, v in lc._N1_CASES.items()}}


def _split_enabled():
    return m3_sparse_decode_lc.LC2_SPLIT


def _call(allow_split, c, out, q8, sf, linear, use_pdl, prewait, n1):
    family_ops().minimax_m3_sm107_tp2_sparse_decode_lc(
        c["q"].view(torch.uint8),
        c["k"],
        c["v"],
        c["ksc"],
        c["vsc"],
        c["kgs"],
        c["vgs"],
        c["topk"],
        c["bt"],
        c["seq"],
        out,
        q8,
        sf,
        lc.WIDTH // 128,
        c["dql"],
        lc.SM_SCALE,
        linear,
        use_pdl,
        prewait,
        n1,
        allow_split,
    )


def _buffers(total_q, mxfp8):
    out = torch.full((total_q, lc.WIDTH), 7.0, dtype=torch.bfloat16, device="cuda")
    if not mxfp8:
        return out, None, None
    q8 = torch.full((total_q, lc.WIDTH), 0x55, dtype=torch.uint8, device="cuda")
    sf = torch.full(
        (128 * (lc.WIDTH // 32) * ((total_q + 127) // 128),), 0x33, dtype=torch.uint8, device="cuda"
    )
    return out, q8, sf


def _both(c, linear, use_pdl, prewait, mxfp8, n1_init):
    total_q = c["q"].shape[0]
    res = []
    for allow_split in (True, False):
        out, q8, sf = _buffers(total_q, mxfp8)
        n1 = n1_init.clone() if n1_init is not None else None
        _call(allow_split, c, out, q8, sf, linear, use_pdl, prewait, n1)
        torch.cuda.synchronize()
        res.append((out, q8, sf, n1))
    return res


def _assert_same(a, b, what):
    for x, y, name in zip(a, b, ("out", "q8", "sf", "n1")):
        if x is None:
            continue
        assert torch.equal(x.view(torch.uint8), y.view(torch.uint8)), (what, name)


@pytest.mark.skipif(not _split_enabled(), reason="TRTLLM_M31_DISABLE_TP2WIDE_LC2_SPLIT=1")
@pytest.mark.parametrize("name", list(_ALL))
@pytest.mark.parametrize("linear", [False, True])
@pytest.mark.parametrize("prewait", [False, True])
def test_split_bitwise_vs_cluster(name, linear, prewait) -> None:
    c = lc._case(seed=7, **_ALL[name])
    total_q = c["q"].shape[0]
    for mxfp8 in (False, True):
        for use_pdl in (False, True):
            for n1_init in (None, lc._n1_garbage(max(total_q, 1), 5)):
                split, cluster = _both(c, linear, use_pdl, prewait, mxfp8, n1_init)
                _assert_same(
                    split, cluster, (name, linear, prewait, mxfp8, use_pdl, n1_init is not None)
                )


@pytest.mark.skipif(not _split_enabled(), reason="TRTLLM_M31_DISABLE_TP2WIDE_LC2_SPLIT=1")
@pytest.mark.parametrize("name", ["k7-c24", "k3-c56", "k7-c56", "b4"])
def test_split_graph_replay_bitwise(name) -> None:
    """The split captured in a CUDA graph (its workspace is allocated by the eager warmup call) and replayed twice
    writes what the cluster kernel writes eagerly."""
    c = lc._case(seed=9, **_ALL[name])
    total_q = c["q"].shape[0]
    ref = _both(c, False, True, False, True, None)[1]
    out, q8, sf = _buffers(total_q, True)
    _call(True, c, out, q8, sf, False, True, False, None)  # eager warmup
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(stream), torch.cuda.graph(graph, stream=stream):
        _call(True, c, out, q8, sf, False, True, False, None)
    for _ in range(2):
        out.fill_(7.0)
        q8.fill_(0x55)
        sf.fill_(0x33)
        graph.replay()
        torch.cuda.synchronize()
        _assert_same((out, q8, sf, None), ref, name)
