# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""FOLD_F5B on a GPU: the draft head GEMM written into the S4 chain's padded logits buffer equals F.linear plus the
strided copy, bit for bit, at the served shapes (the TP8 shard 25008 of 200064, drafter hidden 6144, K = 7, batches
1..8, the chain's own padded row count), eagerly and replayed from a CUDA graph."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_1_flash_nvfp4__sm_107__tp8 import (  # noqa: E501
    dspark_markov_worker as w1,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.ops.dspark import (
    dspark_markov_chain as chain,
)
from tensorrt_llm._torch.modules.linear import UnquantizedLinearMethod

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7),
    reason="the sm_107 TP8 target runs on SM107 only",
)

SHARD, HIDDEN, K = 25008, 6144, 7


def _head(seed=11):
    g = torch.Generator(device="cuda").manual_seed(seed)
    weight = (torch.randn(SHARD, HIDDEN, device="cuda", generator=g) * 0.02).bfloat16()
    return SimpleNamespace(
        weight=weight,
        bias=None,
        quant_method=UnquantizedLinearMethod(),
        _locality_domain_weight_shards=None,
        use_cute_dsl_bf16_gemm=False,
        use_custom_cublas_mm=False,
    )


def _rows():
    rows = chain.padded_rows(SHARD, w1.MAX_FAST_BATCH, torch.device("cuda"))
    if rows == 0:
        pytest.skip("the chain's kernel does not split this shard on this device")
    return rows


def _hidden(seed, B):
    g = torch.Generator(device="cuda").manual_seed(seed)
    return (torch.randn(B * K, HIDDEN, device="cuda", generator=g) * 2).bfloat16()


def _reference(head, h, B, rows):
    want = chain.pad_logits_buffer(B, K, rows, h.device)
    logits = F.linear(h, head.weight)
    want[:, :, :SHARD].copy_(logits.view(B, K, SHARD))
    return want, logits


@pytest.mark.parametrize("B", list(range(1, w1.MAX_FAST_BATCH + 1)))
def test_head_into_buffer_bitwise(B):
    rows = _rows()
    head = _head()
    for seed in range(3):
        h = _hidden(100 * seed + B, B)
        got = chain.pad_logits_buffer(B, K, rows, h.device)
        out = w1._head_into(got, head, h, SHARD)
        assert out is not None
        want, logits = _reference(head, h, B, rows)
        assert torch.equal(got.view(torch.int16), want.view(torch.int16)), f"B {B} seed {seed}"
        assert torch.equal(out.view(torch.int16), logits.view(torch.int16)), f"B {B} seed {seed}"


def test_head_into_buffer_graph_replay():
    rows = _rows()
    head = _head()
    B = 1
    h = _hidden(1, B)
    got = chain.pad_logits_buffer(B, K, rows, h.device)
    w1._head_into(got, head, h, SHARD)
    torch.cuda.synchronize()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    g = torch.cuda.CUDAGraph()
    with torch.cuda.stream(s):
        with torch.cuda.graph(g):
            w1._head_into(got, head, h, SHARD)
    torch.cuda.current_stream().wait_stream(s)
    for rep in range(4):
        h.copy_(_hidden(500 + rep, B))
        g.replay()
        torch.cuda.synchronize()
        want, _ = _reference(head, h, B, rows)
        assert torch.equal(got.view(torch.int16), want.view(torch.int16)), f"replay {rep}"
