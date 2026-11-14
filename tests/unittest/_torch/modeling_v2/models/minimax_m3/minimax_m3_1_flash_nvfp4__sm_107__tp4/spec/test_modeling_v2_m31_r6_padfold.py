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

import ast
import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_1_flash_nvfp4__sm_107__tp4 import (  # noqa: E501
    dspark_markov_worker as s4,
)
from tensorrt_llm._torch.modules.linear import UnquantizedLinearMethod

SHARD = 50016  # 200064 / 4
ROWS = 50688  # the S4 padded shard (132 x 384)
HIDDEN = 6144
BLOCK = 7
BLOCKS = (3, 7)  # the S4 draft lengths (K3, K7)
_DIR = Path(s4.__file__).resolve().parent

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


_BATCHES = list(range(1, s4.MAX_FAST_BATCH + 1))


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


def _gemm_kernels(fn):
    """CUDA kernel names launched by ``fn`` (in order)."""
    from torch.profiler import ProfilerActivity, profile

    fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    names = [
        e.name
        for e in prof.events()
        if e.device_type == torch.autograd.DeviceType.CUDA and "memcpy" not in e.name.lower()
    ]
    return names


@_cuda
@pytest.mark.parametrize("block", BLOCKS, ids=[f"k{b}" for b in BLOCKS])
@pytest.mark.parametrize("batch", _BATCHES)
def test_padfold_runs_the_same_gemm_kernel(batch, block):
    """The same cuBLAS algorithm for ldc = ROWS as for the fresh output, at every M PADFOLD serves: the folded launch's
    kernel list is the stock path's minus the copy."""
    hidden, weight = _operands(batch, False, block=block)
    stock_buf = torch.full((batch, block, ROWS), float("-inf"), dtype=torch.bfloat16, device="cuda")
    logits = F.linear(hidden, weight)

    def stock_gemm():
        F.linear(hidden, weight)

    def stock_copy():
        stock_buf[:, :, :SHARD].copy_(logits.view(batch, block, SHARD))

    fold_buf = torch.full_like(stock_buf, float("-inf"))

    def fold():
        s4.padfold_linear(hidden, weight, fold_buf, SHARD)

    gemm = _gemm_kernels(stock_gemm)
    copy = _gemm_kernels(stock_copy)
    folded = _gemm_kernels(fold)
    assert gemm, "no GEMM kernel seen"
    assert folded == gemm, f"folded GEMM kernels {folded} != stock {gemm}"
    assert not set(copy) & set(folded), "the fold must not launch the stock copy"


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


@_cuda
def test_declined_refine_gets_contiguous_logits(monkeypatch):
    """A step the op declines after the fold: the in-tree path gets the same values, contiguous."""
    got = {}

    class _Base:
        def _refine_block_logits(self, draft_model, gen_logits, inputs, spec_metadata):
            got["logits"] = gen_logits
            return gen_logits

    class _Worker(s4.S4WorkerMixin, _Base):
        pass

    batch = 1
    hidden, weight = _operands(batch, False, seed=4)
    st = _state([batch])
    gen = s4._ShardedDraftLogits(lambda *a: None, st, batch, fold=True)(
        hidden, _head(weight), None, True
    )
    gen = gen.reshape(batch, BLOCK, SHARD)
    monkeypatch.setattr(s4, "chain", SimpleNamespace(supports=lambda *a: False))
    wk = _Worker()
    wk._s4_step = st
    wk._refine_block_logits(None, gen, {}, None)
    assert got["logits"].is_contiguous() and torch.equal(_bits(got["logits"]), _bits(gen))


# ----------------------------------------------------------------------------------------------- routing (no GPU)


def _cpu_state(batch=2, shard=8, rows=12, block=3):
    return _state([batch], shard=shard, rows=rows, block=block, device="cpu")


def _cpu_head(**attrs):
    head = SimpleNamespace(apply_linear=lambda h, b: h[:, :8] * 2, calls=[])
    for k, v in attrs.items():
        setattr(head, k, v)
    return head


def test_unfolded_processor_is_the_stock_copy():
    """fold=False (S4-logits steps, PADFOLD off): the head's own GEMM, then the copy; the tail keeps -inf."""
    st = _cpu_state()
    proc = s4._ShardedDraftLogits(lambda *a: "inner", st, batch=2, fold=False)
    h = torch.arange(6 * 10, dtype=torch.float32).view(6, 10)
    out = proc(h, _cpu_head(), None, True)
    assert torch.equal(out, h[:, :8] * 2) and out.is_contiguous()
    assert torch.equal(st.buffer(2)[:, :, :8].float(), (h[:, :8] * 2).view(2, 3, 8))
    assert torch.isinf(st.buffer(2)[:, :, 8:]).all()


@_cuda
def test_fold_declines_anything_but_the_plain_flinear_route(monkeypatch):
    """A head PADFOLD cannot reproduce keeps the stock path even with fold=True."""
    import tensorrt_llm._torch.modules.low_m_gemm as low_m

    w = torch.zeros(8, 10, dtype=torch.bfloat16, device="cuda")
    h = torch.zeros(6, 10, dtype=torch.bfloat16, device="cuda")
    base = dict(
        weight=w,
        bias=None,
        quant_method=UnquantizedLinearMethod(),
        use_custom_cublas_mm=False,
        use_cute_dsl_bf16_gemm=False,
        _locality_domain_weight_shards=None,
    )
    monkeypatch.setattr(low_m, "_should_apply_low_m_gemm", lambda x: False)
    assert s4.padfold_applies(SimpleNamespace(**base), h, 8)
    for change in (
        dict(bias=torch.zeros(8, dtype=torch.bfloat16, device="cuda")),
        dict(quant_method=object()),
        dict(use_custom_cublas_mm=True),
        dict(use_cute_dsl_bf16_gemm=True),
        dict(_locality_domain_weight_shards=[w]),
        dict(weight=w.float()),
        dict(weight=w.cpu()),
        dict(weight=torch.zeros(9, 10, dtype=torch.bfloat16, device="cuda")),
    ):
        assert not s4.padfold_applies(SimpleNamespace(**{**base, **change}), h, 8), change
    assert not s4.padfold_applies(SimpleNamespace(**base), h.float(), 8)
    assert not s4.padfold_applies(SimpleNamespace(**base), h.cpu(), 8)
    assert not s4.padfold_applies(SimpleNamespace(**base), h[:, :9], 8)
    assert not s4.padfold_applies(SimpleNamespace(), h, 8)
    # The low-M dispatcher would take this GEMM: the stock route is not F.linear, so no fold.
    monkeypatch.setattr(low_m, "_should_apply_low_m_gemm", lambda x: True)
    assert not s4.padfold_applies(SimpleNamespace(**base), h, 8)


def test_folded_processor_falls_back_for_a_foreign_head():
    """fold=True with a head that has no plain weight route: the stock GEMM + copy."""
    st = _cpu_state()
    proc = s4._ShardedDraftLogits(lambda *a: "inner", st, batch=2, fold=True)
    h = torch.arange(6 * 10, dtype=torch.float32).view(6, 10)
    out = proc(h, _cpu_head(), None, True)
    assert torch.equal(out, h[:, :8] * 2)
    assert torch.equal(st.buffer(2)[:, :, :8].float(), (h[:, :8] * 2).view(2, 3, 8))


def test_every_s4_step_folds(monkeypatch):
    """The worker builds the step's processor with fold on whenever PADFOLD is (S4-logits steps read the strided rows
    through the catalog's strided-logits arm)."""

    class _Base:
        def _forward_impl(self, *args):
            return args[6].logits_processor

    class _Worker(s4.S4WorkerMixin, _Base):
        pass

    for mode, on, want in (
        ("tokens", True, True),
        ("logits", True, True),
        ("tokens", False, False),
    ):
        monkeypatch.setattr(s4, "PADFOLD", on)
        wk = _Worker()
        wk._s4 = _cpu_state()
        monkeypatch.setattr(wk, "_s4_mode", lambda n, m, d, _mode=mode: _mode, raising=False)
        draft = SimpleNamespace(logits_processor="inner")
        proc = wk._forward_impl(
            None, None, None, None, SimpleNamespace(num_seqs=2, num_contexts=0), None, draft
        )
        assert isinstance(proc, s4._ShardedDraftLogits) and proc.fold is want, (mode, on)
        assert draft.logits_processor == "inner"


def test_padfold_engagement_logged_once_per_rank(monkeypatch):
    lines = []
    monkeypatch.setattr(s4.logger, "info", lambda msg, *a, **k: lines.append(msg))
    monkeypatch.setattr(s4, "padfold_applies", lambda head, h, shard: True)
    monkeypatch.setattr(
        s4, "padfold_linear", lambda h, w, buf, shard: buf.view(-1, buf.shape[-1])[:, :shard]
    )
    st = _cpu_state()
    st.mapping = SimpleNamespace(tp_rank=2)
    proc = s4._ShardedDraftLogits(lambda *a: None, st, batch=2, fold=True)
    for _ in range(3):
        proc(torch.zeros(6, 10), SimpleNamespace(weight=None), None, True)
    assert [m for m in lines if "PADFOLD" in m] == [
        "[M3 dispatch] PADFOLD taken: draft head GEMM into the S4 padded buffer (6 rows, ldc 12) on rank 2"
    ]


def _eval_switch(monkeypatch, env):
    """dspark_markov_worker.py's PADFOLD assignment evaluated against a private copy of m31_switches under ``env``."""
    for name in list(os.environ):
        if name.startswith(("TRTLLM_M31_DISABLE_", "TRTLLM_M31_ENABLE_")):
            monkeypatch.delenv(name)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    spec = importlib.util.spec_from_file_location(
        f"_m31_switches_padfold_{id(env)}", _DIR / "m31_switches.py"
    )
    switches = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(switches)
    tree = ast.parse((_DIR / "dspark_markov_worker.py").read_text())
    stmts = [
        n
        for n in tree.body
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "PADFOLD" for t in n.targets)
    ]
    assert len(stmts) == 1
    ns = {"m31_disabled": switches.m31_disabled, "m31_enabled": switches.m31_enabled}
    exec(compile(ast.Module(stmts, []), "dspark_markov_worker.py", "exec"), ns)
    return ns["PADFOLD"]


@pytest.mark.parametrize(
    "env,want",
    [
        ({}, True),
        ({"TRTLLM_M31_DISABLE_PADFOLD": "0"}, True),
        ({"TRTLLM_M31_DISABLE_PADFOLD": "1"}, False),
    ],
)
def test_padfold_switch(monkeypatch, env, want):
    """PADFOLD is on by default; TRTLLM_M31_DISABLE_PADFOLD=1 (read at import) turns it off."""
    assert _eval_switch(monkeypatch, env) is want
