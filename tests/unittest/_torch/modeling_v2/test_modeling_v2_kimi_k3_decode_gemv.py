# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The Kimi K3 target's decode GEMVs, LM head and embedding on the catalog's single-GPU entries (``decode_gemv.py``
of ``kimi_k3_mxfp4__sm_100__tp16_moetp4ep4``), on one GPU.

* Each MLA projection site at 1..8 rows: the bits of its catalog entry's call, within 8e-3 of ``max |ref|`` of a
  float64 product; declined above 8 rows, at another shape or dtype, and under capture before an eager call.
* The LM head through ``K3LogitsProcessor`` on a real ``LMHead`` (one rank): fp32 logits from ``k3_head_gemv``
  within the same bound, the stock processor's rows selected, the stock path above 8 rows and before the state is
  built.
* The embedding: bit-identical to ``nn.Embedding`` followed by the stock RMSNorm, the rows in the bank's slot 0.
* CUDA-graph replays give the eager bits.
"""

import types

import pytest
import torch
from torch import nn

from tensorrt_llm._torch._experimental.modeling_v2.catalog.gemm.k3_ctm_gemv_wide import (
    k3_ctm_gemv_wide,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.gemm.k3_decode_gemv import k3_decode_gemv
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.kimi_k3_mxfp4__sm_100__tp16_moetp4ep4 import (  # noqa: E501
    decode_gemv,
)
from tensorrt_llm._torch.modules.embedding import LMHead
from tensorrt_llm._torch.modules.linear import TensorParallelMode
from tensorrt_llm._torch.modules.logits_processor import LogitsProcessor
from tensorrt_llm._torch.modules.rms_norm import RMSNorm
from tensorrt_llm.mapping import Mapping

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 0),
    reason="the K3 decode kernels run on sm_100 only",
)

TOL = 8e-3  # max |y - ref| / max |ref|, ref a float64 product
VOCAB_SHARD, HIDDEN = 10240, 7168
ROWS = range(1, 9)


def _weight(n, k, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    return (torch.randn(n, k, generator=g, device="cuda") * 0.02).to(torch.bfloat16)


def _rows(m, k, seed):
    g = torch.Generator(device="cuda").manual_seed(1000 + seed)
    return torch.randn(m, k, generator=g, device="cuda").to(torch.bfloat16)


def _rel_err(y, x, w):
    ref = x.double() @ w.double().t()
    return ((y.double() - ref).abs().max() / ref.abs().max()).item()


def _bits(t):
    return t.contiguous().view(torch.int16)


@pytest.mark.parametrize("site", list(decode_gemv.SITES))
def test_site(site):
    """Every row count 1..8 runs the site's catalog entry: its bits, within TOL of the float64 product."""
    n, k, kernel = decode_gemv.SITES[site]
    w = _weight(n, k, seed=len(site))
    gemvs = decode_gemv.K3DecodeGemvs.create(None, {site: w})
    entry = k3_decode_gemv if kernel == "decode" else k3_ctm_gemv_wide
    for m in ROWS:
        x = _rows(m, k, seed=m)
        y = gemvs.project(site, x, w)
        assert y is not None and y.shape == (m, n), (site, m)
        assert torch.equal(_bits(y), _bits(entry(x, w))), (site, m)
        assert _rel_err(y, x, w) <= TOL, (site, m)


def test_site_declines():
    """Above 8 rows, another weight shape, a non-bf16 input, and under capture before any eager call: None, nothing
    launched."""
    n, k, _ = decode_gemv.SITES["kv_a"]
    w = _weight(n, k, seed=1)
    gemvs = decode_gemv.K3DecodeGemvs.create(None, {"kv_a": w})
    assert gemvs.project("kv_a", _rows(9, k, seed=9), w) is None
    assert gemvs.project("kv_a", _rows(4, k, seed=4), _weight(n + 128, k, seed=2)) is None
    assert gemvs.project("kv_a", _rows(4, k, seed=4).float(), w) is None
    fresh = decode_gemv.K3DecodeGemvs()
    x = _rows(4, k, seed=4)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        y = fresh.project("kv_a", x, w)
    assert y is None


def test_site_capture_replays_the_eager_bits():
    """A captured call, its input rewritten in place before each replay, gives the eager call's bits."""
    n, k, _ = decode_gemv.SITES["g_proj"]
    w = _weight(n, k, seed=3)
    gemvs = decode_gemv.K3DecodeGemvs.create(None, {"g_proj": w})
    x = _rows(4, k, seed=4)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        y = gemvs.project("g_proj", x, w)
    assert y is not None
    for seed in (5, 6):
        x.copy_(_rows(4, k, seed=seed))
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(_bits(y), _bits(gemvs.project("g_proj", x, w)))


def _lm_head():
    head = LMHead(
        VOCAB_SHARD,
        HIDDEN,
        dtype=torch.bfloat16,
        mapping=Mapping(),
        tensor_parallel_mode=TensorParallelMode.COLUMN,
        gather_output=True,
    ).cuda()
    with torch.no_grad():
        head.weight.copy_(_weight(VOCAB_SHARD, HIDDEN, seed=7))
    return head


def test_lm_head():
    """At most 8 rows: fp32 logits from k3_head_gemv within TOL; 9 rows and an unbuilt processor: the stock bits."""
    head = _lm_head()
    stock = LogitsProcessor()
    processor = decode_gemv.K3LogitsProcessor(stock)
    for m in (1, 9):
        x = _rows(m, HIDDEN, seed=m)
        assert torch.equal(processor(x, head, None, True), stock(x, head, None, True))
    processor.gemvs = decode_gemv.K3DecodeGemvs.create(head, {})
    assert processor.gemvs.head_workspace is not None
    for m in ROWS:
        x = _rows(m, HIDDEN, seed=m)
        logits = processor(x, head, None, True)
        assert logits.dtype == torch.float32 and logits.shape == (m, VOCAB_SHARD)
        assert _rel_err(logits, x, head.weight) <= TOL, m
    x = _rows(9, HIDDEN, seed=9)
    assert torch.equal(processor(x, head, None, True), stock(x, head, None, True))


def test_lm_head_selects_the_stock_rows():
    """Without context logits, the stock processor's last-token selection feeds the head."""
    head = _lm_head()
    processor = decode_gemv.K3LogitsProcessor(LogitsProcessor())
    processor.gemvs = decode_gemv.K3DecodeGemvs.create(head, {})
    x = _rows(7, HIDDEN, seed=3)
    metadata = types.SimpleNamespace(
        seq_lens_cuda=torch.tensor([3, 1, 3], dtype=torch.int32, device="cuda")
    )
    logits = processor(x, head, metadata, False)
    want = processor(x[[2, 3, 6]].contiguous(), head, None, True)
    assert torch.equal(logits, want)


def test_lm_head_capture_replays_the_eager_bits():
    head = _lm_head()
    gemvs = decode_gemv.K3DecodeGemvs.create(head, {})
    x = _rows(8, HIDDEN, seed=8)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        y = gemvs.lm_head_logits(x, head)
    assert y is not None
    x.copy_(_rows(8, HIDDEN, seed=11))
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(_bits(y), _bits(gemvs.lm_head_logits(x, head)))


@pytest.mark.parametrize("n", [1, 8, 64])
def test_embed_norm(n):
    """Layer 0's normed input is bit-identical to nn.Embedding followed by the stock RMSNorm; the bank's slot 0 holds
    the embedding rows."""
    vocab = 4096
    table = _weight(vocab, HIDDEN, seed=21)
    norm = RMSNorm(hidden_size=HIDDEN, eps=1e-5, dtype=torch.bfloat16).cuda()
    with torch.no_grad():
        norm.weight.copy_(1 + _weight(1, HIDDEN, seed=22)[0])
    embedding = nn.Embedding(vocab, HIDDEN, dtype=torch.bfloat16).cuda()
    with torch.no_grad():
        embedding.weight.copy_(table)
    g = torch.Generator(device="cuda").manual_seed(n)
    ids = torch.randint(0, vocab, (n,), generator=g, device="cuda", dtype=torch.int32)
    gemvs = decode_gemv.K3DecodeGemvs()
    bank = table.new_empty(3, n, HIDDEN)
    normed = gemvs.embed_norm(ids, table, norm, bank)
    assert normed is not None
    raw = embedding(ids)
    assert torch.equal(_bits(bank[0]), _bits(raw))
    assert torch.equal(_bits(normed), _bits(norm(raw)))


def test_embed_norm_capture():
    """Under capture a token count that ran eagerly replays the eager bits; one that never ran is declined."""
    vocab = 4096
    table = _weight(vocab, HIDDEN, seed=31)
    norm = RMSNorm(hidden_size=HIDDEN, eps=1e-5, dtype=torch.bfloat16).cuda()
    gemvs = decode_gemv.K3DecodeGemvs()
    ids = torch.arange(4, dtype=torch.int32, device="cuda")
    bank = table.new_empty(2, 4, HIDDEN)
    eager = gemvs.embed_norm(ids, table, norm, bank).clone()
    other_ids = torch.arange(5, dtype=torch.int32, device="cuda")
    other_bank = table.new_empty(2, 5, HIDDEN)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = gemvs.embed_norm(ids, table, norm, bank)
        declined = gemvs.embed_norm(other_ids, table, norm, other_bank)
    assert captured is not None and declined is None
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(_bits(captured), _bits(eager))
