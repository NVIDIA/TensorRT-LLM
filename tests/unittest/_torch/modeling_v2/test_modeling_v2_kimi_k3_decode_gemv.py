# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The Kimi K3 target's decode GEMVs, LM head and embedding on the catalog's single-GPU entries (``decode_gemv.py``
of ``kimi_k3_mxfp4__sm_100__tp16_moetp4ep4``), on one GPU.

* Each GEMV site at every row count it takes (1..8, and 9..its ``wide_rows`` where k3_ctm_gemv_wide takes it): the
  bits of its catalog entry's call (fp32 at an ``out_fp32`` site), within 8e-3 of ``max |ref|`` of a float64 product
  (the sigmoid columns within 1e-2 of the sigmoid of it; a SiLU-and-mul site's product taken of torch's bf16
  ``silu(gate) * up``); declined above its rows, at another shape or dtype, and under capture before an eager call.
* The LM head through ``K3LogitsProcessor`` on a real ``LMHead`` (one rank): fp32 logits from ``k3_head_gemv``
  within the same bound, the stock processor's rows selected, the stock path above 8 rows and before the state is
  built.
* The embedding: bit-identical to ``nn.Embedding`` followed by the stock RMSNorm, the rows in the bank's slot 0.
* Layer 0's dense MLP at 1..8 rows: the bits of its three entries chained, within 3e-2 of ``max |ref|`` of the
  float64 SituAndMul MLP; declined above 8 rows, at linear_beta 0 and under capture before an eager call.
* CUDA-graph replays give the eager bits.
"""

import types

import pytest
import torch
from torch import nn

from tensorrt_llm._torch._experimental.modeling_v2.catalog.activation.k3_situ_mul import k3_situ_mul
from tensorrt_llm._torch._experimental.modeling_v2.catalog.gemm.k3_ctm_gemv import k3_ctm_gemv
from tensorrt_llm._torch._experimental.modeling_v2.catalog.gemm.k3_ctm_gemv_long import (
    k3_ctm_gemv_long,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.gemm.k3_ctm_gemv_swiglu import (
    k3_ctm_gemv_swiglu,
)
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


def _check_product(y, x, w, sig_col0=-1):
    """Within TOL of the float64 product (columns from ``sig_col0`` on: within 1e-2 of its sigmoid)."""
    ref = x.double() @ w.double().t()
    sig = ref.shape[1] if sig_col0 < 0 else sig_col0
    lin = ((y[:, :sig].double() - ref[:, :sig]).abs().max() / ref[:, :sig].abs().max()).item()
    assert lin <= TOL, lin
    if sig < ref.shape[1]:
        err = (y[:, sig:].double() - ref[:, sig:].sigmoid()).abs().max().item()
        assert err <= 1e-2, err


def _rel_err(y, x, w):
    ref = x.double() @ w.double().t()
    return ((y.double() - ref).abs().max() / ref.abs().max()).item()


def _bits(t):
    return t.contiguous().view(torch.int16)


def _entry(spec, rows, x, w):
    """The site's catalog entry called directly: what ``project`` must reproduce bit for bit."""
    if rows > decode_gemv.MAX_ROWS or spec.small == "wide":
        return k3_ctm_gemv_wide(x, w, sig_col0=spec.sig_col0, out_fp32=spec.out_fp32)
    if spec.small == "decode":
        return k3_decode_gemv(x, w)
    if spec.small == "ctm":
        return k3_ctm_gemv(x, w, split=spec.split, push=spec.push)
    if spec.small == "swiglu":
        return k3_ctm_gemv_swiglu(x, w, split=spec.split, push=spec.push)
    return k3_ctm_gemv_long(
        x, w, sig_col0=spec.sig_col0, split=spec.split, ring=spec.ring, push=spec.push
    )


def _site_rows(spec):
    return list(ROWS) + (
        [m for m in (9, 16, 24, 32, 40, 64) if m <= spec.wide_rows] if spec.wide else []
    )


def _product_input(spec, x):
    """The rows the site's weight multiplies: ``x``, or a SiLU-and-mul site's torch bf16 ``silu(gate) * up``."""
    if spec.small != "swiglu":
        return x
    gate, up = x.float().chunk(2, dim=-1)
    return (torch.nn.functional.silu(gate) * up).to(torch.bfloat16)


@pytest.mark.parametrize("site", list(decode_gemv.SITES))
def test_site(site):
    """Every row count the site takes runs its catalog entry: its bits, within TOL of the float64 product."""
    spec = decode_gemv.SITES[site]
    w = _weight(spec.n, spec.k, seed=len(site))
    gemvs = decode_gemv.K3DecodeGemvs.create(None, sites=[site])
    for m in _site_rows(spec):
        x = _rows(m, decode_gemv._width(spec), seed=m)
        y = gemvs.project(site, x, w)
        assert y is not None and y.shape == (m, spec.n), (site, m)
        assert y.dtype == (torch.float32 if spec.out_fp32 else torch.bfloat16), (site, y.dtype)
        assert torch.equal(_bits(y), _bits(_entry(spec, m, x, w))), (site, m)
        _check_product(y, _product_input(spec, x), w, spec.sig_col0)


@pytest.mark.parametrize("site", ["kv_a", "mla_ag", "moe_head"])
def test_site_declines(site):
    """More rows than the site takes, another weight shape, a non-bf16 input, and under capture before any eager
    call: None, nothing launched."""
    spec = decode_gemv.SITES[site]
    w = _weight(spec.n, spec.k, seed=1)
    gemvs = decode_gemv.K3DecodeGemvs.create(None, sites=[site])
    too_many = spec.wide_rows + 1 if spec.wide else decode_gemv.MAX_ROWS + 1
    assert gemvs.project(site, _rows(too_many, spec.k, seed=9), w) is None
    assert (
        gemvs.project(site, _rows(4, spec.k, seed=4), _weight(spec.n + 128, spec.k, seed=2)) is None
    )
    assert gemvs.project(site, _rows(4, spec.k, seed=4).float(), w) is None
    fresh = decode_gemv.K3DecodeGemvs()
    x = _rows(4, spec.k, seed=4)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        y = fresh.project(site, x, w)
    assert y is None


@pytest.mark.parametrize("rows", [4, 16])
def test_site_capture_replays_the_eager_bits(rows):
    """A captured call (the long kernel at 4 rows, the wide one at 16, the gate columns through the sigmoid), its
    input rewritten in place before each replay, gives the eager call's bits."""
    spec = decode_gemv.SITES["mla_ag"]
    w = _weight(spec.n, spec.k, seed=3)
    gemvs = decode_gemv.K3DecodeGemvs.create(None, sites=["mla_ag"])
    x = _rows(rows, spec.k, seed=4)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        y = gemvs.project("mla_ag", x, w)
    assert y is not None
    for seed in (5, 6):
        x.copy_(_rows(rows, spec.k, seed=seed))
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(_bits(y), _bits(gemvs.project("mla_ag", x, w)))


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
    processor.gemvs = decode_gemv.K3DecodeGemvs.create(head, sites=())
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
    processor.gemvs = decode_gemv.K3DecodeGemvs.create(head, sites=())
    x = _rows(7, HIDDEN, seed=3)
    metadata = types.SimpleNamespace(
        seq_lens_cuda=torch.tensor([3, 1, 3], dtype=torch.int32, device="cuda")
    )
    logits = processor(x, head, metadata, False)
    want = processor(x[[2, 3, 6]].contiguous(), head, None, True)
    assert torch.equal(logits, want)


def test_lm_head_capture_replays_the_eager_bits():
    head = _lm_head()
    gemvs = decode_gemv.K3DecodeGemvs.create(head, sites=())
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


SITU = (4.0, 25.0)  # the Kimi K3 checkpoint's (beta, linear_beta)


def _dense_weights():
    gate_up, down = decode_gemv.SITES["dense_gate_up"], decode_gemv.SITES["dense_down"]
    return _weight(gate_up.n, gate_up.k, seed=41), _weight(down.n, down.k, seed=42)


def _dense_reference(x, w_gu, w_down, beta, linear_beta):
    gu = x.double() @ w_gu.double().t()
    g, u = gu.chunk(2, dim=1)
    a = beta * torch.tanh(g / beta) * torch.sigmoid(g)
    v = u if linear_beta is None else linear_beta * torch.tanh(u / linear_beta)
    return (a * v) @ w_down.double().t()


@pytest.mark.parametrize("situ", [SITU, (1.0, None)], ids=["k3", "default"])
def test_dense_mlp(situ):
    """At 1..8 rows: the bits of the gate_up GEMV, k3_situ_mul and the down GEMV called directly, within 3e-2 of the
    float64 MLP."""
    w_gu, w_down = _dense_weights()
    gemvs = decode_gemv.K3DecodeGemvs.create(None, sites=["dense_gate_up", "dense_down"])
    gate_up, down = decode_gemv.SITES["dense_gate_up"], decode_gemv.SITES["dense_down"]
    for m in ROWS:
        x = _rows(m, HIDDEN, seed=m)
        y = gemvs.dense_mlp(x, w_gu, w_down, *situ)
        assert y is not None and y.shape == (m, down.n), m
        gu = k3_ctm_gemv_long(x, w_gu, split=gate_up.split, ring=gate_up.ring)
        want = k3_ctm_gemv_long(k3_situ_mul(gu, *situ), w_down, split=down.split, ring=down.ring)
        assert torch.equal(_bits(y), _bits(want)), m
        ref = _dense_reference(x, w_gu, w_down, *situ)
        err = ((y.double() - ref).abs().max() / ref.abs().max()).item()
        assert err <= 3e-2, (m, err)


def test_dense_mlp_declines():
    """9 rows, linear_beta 0 (the activation would run it as 1), and under capture before any eager call: None."""
    w_gu, w_down = _dense_weights()
    gemvs = decode_gemv.K3DecodeGemvs.create(None, sites=["dense_gate_up", "dense_down"])
    assert gemvs.dense_mlp(_rows(9, HIDDEN, seed=9), w_gu, w_down, *SITU) is None
    assert gemvs.dense_mlp(_rows(4, HIDDEN, seed=4), w_gu, w_down, 4.0, 0.0) is None
    fresh = decode_gemv.K3DecodeGemvs()
    x = _rows(4, HIDDEN, seed=4)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        y = fresh.dense_mlp(x, w_gu, w_down, *SITU)
    assert y is None


def test_dense_mlp_capture_replays_the_eager_bits():
    w_gu, w_down = _dense_weights()
    gemvs = decode_gemv.K3DecodeGemvs.create(None, sites=["dense_gate_up", "dense_down"])
    x = _rows(8, HIDDEN, seed=8)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        y = gemvs.dense_mlp(x, w_gu, w_down, *SITU)
    assert y is not None
    x.copy_(_rows(8, HIDDEN, seed=12))
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(_bits(y), _bits(gemvs.dense_mlp(x, w_gu, w_down, *SITU)))
