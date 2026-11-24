# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""PADFOLD (dspark_markov_worker): the S4 draft head's GEMM written straight into the padded logits buffer.

At every M3.1 TP4 shape PADFOLD serves (the K3 and K7 block rows of batches 1..8, M = 3..56; the 50,016-column vocab
shard of the 6144-wide head; the 50,688-column padded buffer) the folded GEMM must leave the buffer byte-identical to
the stock F.linear + copy (the -inf tail included, with logits that are all negative so a zero pad would win an
argmax) and run the same GEMM kernel (the same cuBLAS algorithm for the wider ldc); it must capture in a CUDA graph and
return a view the step's reshape and _refine_block_logits accept. The routing (greedy steps only, the head's plain
F.linear route only, the kill switch) runs without a GPU.
"""

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from tensorrt_llm._torch.modeling_v2.models.minimax_m3.minimax_m3_1_flash_nvfp4__sm_107__tp4 import (  # noqa: E501
    dspark_markov_worker as s4,
)
from tensorrt_llm._torch.modules.linear import UnquantizedLinearMethod

SHARD = 50016  # 200064 / 4
ROWS = 50688  # the S4 padded shard (132 x 384)
HIDDEN = 6144
BLOCK = 7
BLOCKS = (3, 7)  # the S4 draft lengths (K3, K7)

_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="PADFOLD's GEMM needs a CUDA device"
)


def _head(weight, **attrs):
    """The target head as the S4 logits processor sees it: UnquantizedLinearMethod's default F.linear route."""
    calls = []

    def apply_linear(h, bias):
        calls.append(tuple(h.shape))
        return F.linear(h, weight, bias)

    head = SimpleNamespace(
        weight=weight,
        bias=None,
        quant_method=UnquantizedLinearMethod(),
        use_custom_cublas_mm=False,
        use_cute_dsl_bf16_gemm=False,
        _locality_domain_weight_shards=None,
        apply_linear=apply_linear,
        calls=calls,
    )
    for k, v in attrs.items():
        setattr(head, k, v)
    return head


def _state(batches, shard=SHARD, rows=ROWS, block=BLOCK, device="cuda"):
    st = s4._S4State(SimpleNamespace(tp_rank=0), rows, shard, 0, None, None, block, {})
    for b in batches:
        st.bufs[(b, block)] = torch.full(
            (b, block, rows), float("-inf"), dtype=torch.bfloat16, device=device
        )
    return st


_WEIGHTS: dict = {}


def _weight(negative):
    """The [shard, hidden] head shard (built once per sign: the cases vary M, the rows)."""
    w = _WEIGHTS.get(negative)
    if w is None:
        g = torch.Generator(device="cuda").manual_seed(1000 + int(negative))
        w = (torch.randn(SHARD, HIDDEN, generator=g, device="cuda") * 0.02).bfloat16()
        if negative:
            w = -(w.abs() + 1e-3)
        w = _WEIGHTS[negative] = w.contiguous()
    return w


def _operands(batch, negative, seed=0, block=BLOCK):
    g = torch.Generator(device="cuda").manual_seed(seed)
    hidden = torch.randn(batch * block, HIDDEN, generator=g, device="cuda").bfloat16()
    if negative:
        # Every logit < 0: positive hidden rows against negative weights (a 0 in the tail would win the argmax).
        hidden = hidden.abs() + 0.01
    return hidden, _weight(negative)


def _stock(hidden, weight, batch, rows=ROWS, block=BLOCK):
    buf = torch.full((batch, block, rows), float("-inf"), dtype=torch.bfloat16, device="cuda")
    logits = F.linear(hidden, weight)
    buf[:, :, :SHARD].copy_(logits.view(batch, block, SHARD))
    return buf, logits


def _bits(t):
    return t.contiguous().view(torch.int16)


_BATCHES = [1, s4.MAX_FAST_BATCH]  # the smallest and largest M PADFOLD serves


@_cuda
@pytest.mark.parametrize("negative", [False, True], ids=["random", "all_negative"])
@pytest.mark.parametrize("block", BLOCKS, ids=[f"k{b}" for b in BLOCKS])
@pytest.mark.parametrize("batch", _BATCHES)
def test_padfold_bytes_equal_flinear_and_copy(batch, block, negative):
    hidden, weight = _operands(batch, negative, seed=batch * 10 + block, block=block)
    want, want_logits = _stock(hidden, weight, batch, block=block)
    st = _state([batch], block=block)
    buf = st.buffer(batch)
    proc = s4._ShardedDraftLogits(lambda *a: None, st, batch, fold=True)
    head = _head(weight)
    got = proc(hidden, head, None, True)
    torch.cuda.synchronize()
    assert head.calls == [], "the folded step must not run the head's own GEMM"
    assert torch.equal(_bits(buf), _bits(want)), "padded buffer differs from F.linear + copy"
    assert torch.isneginf(buf[..., SHARD:]).all(), "the tail past the shard must keep -inf"
    if negative:
        assert (buf[..., :SHARD] < 0).all()
        # The full row's argmax lies in the shard (a zero tail would have won).
        assert (buf.float().argmax(-1) < SHARD).all()
    # The returned logits: the strided [B*K, shard] view of the buffer, the stock logits' values.
    assert got.shape == (batch * block, SHARD) and got.stride() == (ROWS, 1)
    assert got.data_ptr() == buf.data_ptr()
    assert torch.equal(_bits(got), _bits(want_logits))
    # The step's reshape is a view, and _refine_block_logits sees the shard width.
    gen = got.reshape(batch, block, SHARD)
    assert gen.data_ptr() == buf.data_ptr() and gen.shape[-1] == st.shard


@_cuda
def test_padfold_captures_in_a_cuda_graph():
    batch = 1
    hidden, weight = _operands(batch, False, seed=5)
    st = _state([batch])
    buf = st.buffer(batch)
    proc = s4._ShardedDraftLogits(lambda *a: None, st, batch, fold=True)
    head = _head(weight)
    static_h = hidden.clone()
    proc(static_h, head, None, True)  # warm cuBLAS outside capture
    torch.cuda.synchronize()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(stream):
        with torch.cuda.graph(graph):
            out = proc(static_h, head, None, True)
    torch.cuda.current_stream().wait_stream(stream)
    for seed in (11, 12):
        new_h, _ = _operands(batch, seed == 12, seed=seed)
        static_h.copy_(new_h)
        buf.fill_(float("-inf"))
        graph.replay()
        torch.cuda.synchronize()
        want, _ = _stock(new_h, weight, batch)
        assert torch.equal(_bits(buf), _bits(want))
        assert out.data_ptr() == buf.data_ptr()
    assert head.calls == []


@_cuda
def test_refine_accepts_the_folded_view(monkeypatch):
    """The S4 worker's _refine_block_logits takes the folded view (shape check on the last dim only) and hands the
    chain the padded buffer itself."""

    class _Base:
        def _refine_block_logits(self, draft_model, gen_logits, inputs, spec_metadata):
            raise AssertionError("a served step must not reach the in-tree path")

    class _Worker(s4.S4WorkerMixin, _Base):
        pass

    batch = 2
    hidden, weight = _operands(batch, True, seed=3)
    st = _state([batch])
    proc = s4._ShardedDraftLogits(lambda *a: None, st, batch, fold=True)
    gen = proc(hidden, _head(weight), None, True).reshape(batch, BLOCK, SHARD)
    seen = {}

    def chain(base, first_prev, w1, w2s, offset, **ws):
        seen["base"] = base
        return torch.zeros(base.shape[0], base.shape[1], dtype=torch.int32, device=base.device)

    monkeypatch.setattr(s4, "chain", SimpleNamespace(supports=lambda *a: True, dspark_markov_chain=chain,
                                                     ll_exchange_engaged=lambda b: False))  # fmt: skip
    wk = _Worker()
    wk.mapping = SimpleNamespace(tp_rank=0)
    wk._s4_step = st
    wk._s4_step_mode = "tokens"
    out = wk._refine_block_logits(
        None, gen, {"first_prev_tokens": torch.zeros(batch, dtype=torch.long)}, None
    )
    assert out is gen and seen["base"] is st.buffer(batch)


# ----------------------------------------------------------------------------------------------- routing (no GPU)
