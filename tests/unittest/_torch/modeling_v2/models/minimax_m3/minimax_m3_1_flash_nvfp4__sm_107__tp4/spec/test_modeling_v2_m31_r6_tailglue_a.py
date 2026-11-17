# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""TAILGLUE_A: the forked DSpark step's post-join glue in one launch (kernels/m3_tail_glue_op: the C++ op
trtllm::minimax_m31_tail_glue_a; dspark_tailfold.TailGlue).

The launch must leave every byte the stock four launches leave -- the DRAFTER_CTX_KV_STORE_FUSED store
(dspark_tailfork's store kernel: COL_FROM_POS on a bound pool), the DRAFTER_CTX_ADVANCE_FUSED advance + read view
(catalog dspark_ctx_advance, ADV_PAR's two launches and the one-launch op / its C++ op on SM107) and the
DRAFTER_FORWARD_INDICES_FUSED forward indices (catalog dspark_forward_indices: its C++ op on SM107, the Triton kernel
elsewhere): the paged cache of every layer, the context lengths (dummy slot included), the read table, the advertised
lengths and the seven index tensors. Covered at the live drafter shape (5 layers, 1 KV head, head dim 128, rotary 64, K7
+ 1 rows per request, a 16,416-page read table) and around it: accepted counts 1..8 on every request, 1..3 requests, a
request on the dummy slot, rebased read views, clamped columns, context lengths at the cap, the k_norm weight in the
kernel or not, stacked or per-layer caches, whole and partial rotary. In a CUDA graph the two sequences run from the
catalog drafter-inputs launch (which writes the positions both read) over replays whose accepted counts walk 1..8 and
whose context grows. The worker-side wiring: the store defers only its bound, column-from-position layout, the advance
fuses only over that store's operands, the forward-indices hook returns the fused outputs only for the exact tensors
they came from, and every other path is the stock one, bitwise. Its lines: the first captured take, then the first step
with a deferred store run eagerly after it and the first one captured with the stock path, once each. The C++ op is also
compared, every byte, with the Triton kernel it ports (kernels/m3_tail_glue, the oracle), eagerly and in a CUDA graph.
"""

import functools
import importlib
import importlib.util
from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.models.minimax_m3.minimax_m3_1_flash_nvfp4__sm_107__tp4 import (
    native_ops,
)
from tensorrt_llm._torch.modeling_v2.models.minimax_m3.ops.dspark import dspark_ctx_advance as adv
from tensorrt_llm._torch.modeling_v2.models.minimax_m3.ops.dspark import dspark_ctx_store as store
from tensorrt_llm._torch.modeling_v2.models.minimax_m3.ops.dspark import (
    dspark_drafter_inputs as dinp,
)
from tensorrt_llm._torch.modeling_v2.models.minimax_m3.ops.dspark import (
    dspark_forward_indices as fidx,
)

TARGET = "minimax_m3_1_flash_nvfp4__sm_107__tp4"
_PKG = "tensorrt_llm._torch.modeling_v2.models.minimax_m3." + TARGET

needs_gpu = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability() != (10, 7)
    or not hasattr(native_ops, "minimax_m31_tail_glue_a"),
    reason="compares against the catalog's C++ glue and context-store ops, built for SM107 only",
)

L, NKV, HD = 5, 1, 128
KP1 = 8  # K7 + 1 rows per request
BLOCK = 7  # the drafter's query block (shift-label, block width K)
SLOTS = 9
DUMMY = SLOTS - 1


def _mods():
    tfold = importlib.import_module(_PKG + ".dspark_tailfold")
    tfork = importlib.import_module(_PKG + ".dspark_tailfork")
    glue = importlib.import_module(_PKG + ".kernels.m3_tail_glue_op")
    return tfold, tfork, glue


def _triton_glue():
    """The Triton kernel the C++ op ports (the oracle; never on a served path)."""
    return importlib.import_module(_PKG + ".kernels.m3_tail_glue")


@pytest.fixture
def fold_flag():
    saved = store._fold_w[0]
    yield store._fold_w
    store._fold_w[0] = saved


def _drafter(dev, seed, hidden, rot):
    g = torch.Generator(device="cpu").manual_seed(seed)
    w = (torch.randn(L * 2 * NKV * HD, hidden, generator=g) * 0.05).to(torch.bfloat16).to(dev)
    kn = (1.0 + 0.3 * torch.randn(L, HD, generator=g)).to(torch.bfloat16).to(dev)
    pos = torch.arange(1 << 17, dtype=torch.float32)
    inv = 1.0 / (10000.0 ** (torch.arange(0, rot, 2, dtype=torch.float32) / rot))
    f = pos[:, None] * inv[None, :]
    cos_sin = torch.cat([f.cos(), f.sin()], dim=1).contiguous().to(dev)
    return SimpleNamespace(
        _fused_kv_weight=w,
        _fused_kv_bias=None,
        _num_attn_layers=L,
        _num_kv_heads=NKV,
        _head_dim=HD,
        _k_norm_stacked=kn,
        _k_norm_eps=1e-6,
        _is_neox=True,
        _input_ln_eps=None,
        _get_cos_sin_cache=lambda: cos_sin,
        _build_fused_kv_buffers=lambda: None,
    )


class _State:
    """One step's operands: the bound draft pool, the context lengths, the block / read tables, the accepted counts and
    the positions the drafter inputs wrote (ctx_len[slot] + j)."""

    def __init__(
        self, dev, seed, n, width, ps, hidden, per_layer, dummy_req, at_cap, rebase, small_counts
    ):
        g = torch.Generator(device="cpu").manual_seed(seed)
        self.n, self.width, self.ps = n, width, ps
        pages = 96
        self.max_ctx = width * ps - 1 if not at_cap else 40 + 3 * seed
        if per_layer:
            self.kv = [
                torch.zeros(pages, 2, NKV, ps, HD, dtype=torch.bfloat16, device=dev)
                for _ in range(L)
            ]
        else:
            self.kv = torch.zeros(L, pages, 2, NKV, ps, HD, dtype=torch.bfloat16, device=dev)
        rows_n = n + 2  # block-table rows beyond the step's requests stay untouched
        self.tables = torch.randint(0, pages, (rows_n, width), generator=g, dtype=torch.int32).to(
            dev
        )
        self.tables_read = torch.full((rows_n, width), -7, dtype=torch.int32, device=dev)
        cnt = (
            torch.randint(1, 4, (rows_n,), generator=g)
            if small_counts
            else torch.full((rows_n,), width)
        )
        self.counts = cnt.to(torch.int64).to(dev)
        slots = torch.randperm(SLOTS - 1, generator=g)[:n]
        if dummy_req:
            slots[-1] = DUMMY
        self.slots = slots.to(torch.int64).to(dev)
        self.rows = torch.arange(n, dtype=torch.int64, device=dev)
        ctx = torch.randint(0, max(1, min(self.max_ctx, ps * 3)), (SLOTS,), generator=g)
        if at_cap:
            ctx[: SLOTS - 1] = torch.clamp(
                self.max_ctx - torch.randint(0, 5, (SLOTS - 1,), generator=g), min=0
            )
        # The dummy slot holds what any step left there (the advance zeroes it either way).
        ctx[DUMMY] = 3 + seed % 4
        self.ctx_len = ctx.to(torch.int64).to(dev)
        if rebase == "deep":  # base pages past the context: the rebase follows the advanced length
            base = torch.full((SLOTS,), 50, dtype=torch.int64)
        elif rebase:
            base = torch.randint(0, 3, (SLOTS,), generator=g)
        else:
            base = torch.zeros(SLOTS, dtype=torch.int64)
        self.base = base.to(torch.int64).to(dev)
        self.nacc = torch.randint(1, KP1 + 1, (n,), generator=g).to(torch.int64).to(dev)
        self.projected = (torch.randn(n * KP1, hidden, generator=g) * 2).to(torch.bfloat16).to(dev)
        # Each request row's first `writable_cols` columns (every column its store can write) get
        # physical pages of their own: a page shared by two written columns would leave bytes that
        # depend on which store lands last. The other entries are only read and may repeat.
        self.writable_cols = min(width, pages // n)
        own = torch.randperm(pages, generator=g)[: n * self.writable_cols]
        own = own.view(n, self.writable_cols).to(torch.int32).to(dev)
        self.tables[:n, : self.writable_cols] = own
        self.refresh_positions()

    def refresh_positions(self):
        ctx = self.ctx_len[self.slots]
        j = torch.arange(KP1, device=ctx.device)
        self.positions = (ctx.unsqueeze(1) + j.unsqueeze(0)).reshape(-1).contiguous()
        last_col = int(self.positions.max()) // self.ps
        assert last_col < self.writable_cols, "a store column would share a physical page"
        now = torch.minimum(ctx + self.nacc, torch.full_like(ctx, self.max_ctx))
        q = torch.arange(BLOCK, device=ctx.device)
        self.qpos = (now.unsqueeze(1) + q.unsqueeze(0)).contiguous()

    def clone(self):
        c = _State.__new__(_State)
        for k, v in self.__dict__.items():
            if isinstance(v, torch.Tensor):
                v = v.clone()
            elif isinstance(v, list):
                v = [t.clone() for t in v]
            c.__dict__[k] = v
        return c

    def kv_bits(self):
        ts = self.kv if isinstance(self.kv, list) else [self.kv]
        return [t.view(torch.int16) for t in ts]


def _store_ops(tfork, draft, st):
    prep = tfork.ctx_store_prep(draft, st.projected)
    ops = tfork.ctx_store_operands(prep, draft, st.projected, st.positions, st.nacc, st.rows, None, st.tables, st.kv,
                                   KP1, capacity_counts=st.counts, max_col=None)  # fmt: skip
    assert isinstance(ops, tfork.StoreOperands)
    assert ops.meta["COL_FROM_POS"] and ops.meta["BOUND"]
    return ops


def _stock(tfork, draft, st, parallel=True):
    _store_ops(tfork, draft, st).launch()
    out = adv.advance_ctx_and_read_view(st.slots, st.rows, st.nacc, st.ctx_len, st.max_ctx, DUMMY, st.counts, st.base,
                                        st.tables, st.tables_read, st.ps, BLOCK, parallel=parallel)  # fmt: skip
    assert out is not None
    idx = fidx.forward_indices(out[: st.n], st.rows, st.qpos, BLOCK, st.width, st.ps)
    assert idx is not None
    return out, idx


def _fused(tfork, glue, draft, st):
    ops = _store_ops(tfork, draft, st)
    return glue.launch(ops, st.slots, st.ctx_len, st.max_ctx, DUMMY, st.base, st.tables_read, st.ps, BLOCK, st.qpos,
                       BLOCK, st.ps)  # fmt: skip


def _assert_same(a, b, out_a, out_b, idx_a, idx_b, what=""):
    for x, y in zip(a.kv_bits(), b.kv_bits()):
        assert torch.equal(x, y), f"cache bytes differ {what}"
    assert torch.equal(a.ctx_len, b.ctx_len), f"ctx_len differs {what}"
    assert torch.equal(a.tables_read, b.tables_read), f"read table differs {what}"
    assert torch.equal(out_a, out_b), f"advertised lengths differ {what}"
    names = ("cache_seqlens", "cache_batch_idx", "seq_lens_after", "kv_indptr", "kv_last_page_len",
             "append_positions", "query_positions_flat")  # fmt: skip
    for nm, x, y in zip(names, idx_a, idx_b):
        assert x.dtype == y.dtype == torch.int32 and x.shape == y.shape, nm
        assert torch.equal(x, y), f"{nm} differs {what}"


CASES = [
    # (n, width, ps, hidden, rot, per_layer, dummy_req, at_cap, rebase, small_counts, fold)
    pytest.param(1, 16416, 32, 6144, 64, True, False, False, True, False, True, id="live-b1-K7"),
    pytest.param(
        1, 16416, 32, 6144, 64, True, False, False, False, False, False, id="live-b1-nofold"
    ),
    pytest.param(2, 16416, 32, 6144, 64, True, False, False, True, False, True, id="live-b2-K7"),
    pytest.param(
        2, 2048, 32, 512, 64, False, False, False, True, False, True, id="b2-stacked-page32"
    ),
    pytest.param(2, 1000, 64, 512, 64, True, True, False, True, False, True, id="b2-dummy"),
    pytest.param(3, 40, 16, 512, 64, False, False, True, True, False, True, id="b3-stacked-at-cap"),
    pytest.param(
        2, 300, 32, 512, 128, True, False, False, True, True, True, id="b2-whole-rotary-clamped"
    ),
    pytest.param(1, 7, 64, 512, 64, False, True, True, False, True, False, id="b1-dummy-narrow"),
    pytest.param(
        2, 16416, 16, 512, 64, True, False, False, "deep", False, True, id="b2-deep-rebase"
    ),
    pytest.param(
        2, 600, 4, 512, 64, True, True, False, "deep", False, True, id="b2-dummy-page4-rebase"
    ),
]


# Every case on the advance-parallel launch, plus the deployed b1 K7 case on the one-launch path.
_GLUE_CASES = [pytest.param(*c.values, True, id=f"advpar-{c.id}") for c in CASES] + [
    pytest.param(*CASES[0].values, False, id=f"onelaunch-{CASES[0].id}")
]


@needs_gpu
@pytest.mark.parametrize(
    "n,width,ps,hidden,rot,per_layer,dummy_req,at_cap,rebase,small_counts,fold,parallel",
    _GLUE_CASES,
)
def test_fused_glue_equals_the_four_launches(fold_flag, n, width, ps, hidden, rot, per_layer, dummy_req, at_cap,
                                             rebase, small_counts, fold, parallel):  # fmt: skip
    tfold, tfork, glue = _mods()
    dev = torch.device("cuda")
    fold_flag[0] = fold
    draft = _drafter(dev, 3, hidden, rot)
    for accepted in range(1, KP1 + 1):  # every accepted count on request 0, the others shifted
        st = _State(
            dev,
            11 * accepted + n,
            n,
            width,
            ps,
            hidden,
            per_layer,
            dummy_req,
            at_cap,
            rebase,
            small_counts,
        )
        st.nacc = ((torch.arange(n, device=dev) + accepted - 1) % KP1 + 1).to(torch.int64)
        st.refresh_positions()
        a, b, c = st.clone(), st.clone(), st.clone()
        store._cache_arg(a.kv)
        store._cache_arg(b.kv)
        store._cache_arg(c.kv)
        out_a, idx_a = _stock(tfork, draft, a, parallel)
        out_b, idx_b = _fused(tfork, glue, draft, b)
        out_c, idx_c = _fused(tfork, _triton_glue(), draft, c)
        torch.cuda.synchronize()
        _assert_same(a, b, out_a, out_b, idx_a, idx_b, f"(accepted {accepted})")
        _assert_same(
            c, b, out_c, out_b, idx_c, idx_b, f"vs the Triton kernel (accepted {accepted})"
        )
        assert any(bool((t != 0).any()) for t in b.kv_bits()), "the store wrote nothing"
        assert not torch.equal(b.tables_read, st.tables_read), "the read view wrote nothing"


@needs_gpu
@pytest.mark.parametrize(
    "shift", [0, 1, -1], ids=["same-start", "read-after-table", "read-before-table"]
)
def test_op_rejects_a_read_view_overlapping_the_block_table(fold_flag, shift):
    _, tfork, glue = _mods()
    dev = torch.device("cuda")
    fold_flag[0] = True
    draft = _drafter(dev, 3, 512, 64)
    st = _State(dev, 7, 1, 40, 32, 512, False, False, False, False, False)
    rows, width = st.tables.shape
    buf = torch.zeros(rows * width + 1, dtype=torch.int32, device=dev)
    t0, r0 = (0, shift) if shift >= 0 else (1, 0)
    st.tables = buf[t0 : t0 + rows * width].view(rows, width)
    st.tables.copy_(torch.randint(0, 96, (rows, width), device=dev, dtype=torch.int32))
    st.tables_read = buf[r0 : r0 + rows * width].view(rows, width)
    store._cache_arg(st.kv)
    with pytest.raises(RuntimeError, match="overlap"):
        _fused(tfork, glue, draft, st)


def _embedding(dev, hidden, seed):
    g = torch.Generator(device="cpu").manual_seed(seed)
    table = (torch.randn(64, hidden, generator=g)).to(torch.bfloat16).to(dev)
    mask = (torch.randn(hidden, generator=g)).to(torch.bfloat16).to(dev)
    return table, mask


@needs_gpu
@pytest.mark.parametrize("n", [1, 2])
def test_fused_glue_in_a_cuda_graph_over_accepted_counts(fold_flag, n):
    """Both sequences captured from the catalog drafter-inputs launch (the positions the store and the read view read
    come from it, as in the step), replayed with accepted counts 1..8 and a growing context: equal after every
    replay."""
    tfold, tfork, glue = _mods()
    dev = torch.device("cuda")
    fold_flag[0] = True
    hidden, width, ps = 512, 2000, 32
    draft = _drafter(dev, 5, hidden, 64)
    table, mask = _embedding(dev, hidden, 5)
    base = _State(dev, 7, n, width, ps, hidden, True, n > 1, False, True, False)
    base.max_ctx = 1500
    accepted = torch.randint(0, 64, (n, KP1), device=dev, dtype=torch.int32)
    nacc32 = torch.ones(n, dtype=torch.int32, device=dev)
    sides = {"stock": base.clone(), "fused": base.clone(), "triton": base.clone()}
    outs = {}
    graphs = {}
    for name, st in sides.items():
        store._cache_arg(st.kv)

        def step(st=st, name=name):
            nacc = nacc32.long()  # dflash.py's gen_num_accepted.long()
            got = dinp.drafter_inputs(
                accepted, nacc32, st.slots, st.ctx_len, st.max_ctx, table, mask, BLOCK, KP1
            )
            st.nacc = nacc
            st.positions = got[2].long().reshape(-1)
            st.qpos = got[1].long()
            if name == "stock":
                return _stock(tfork, draft, st)
            return _fused(tfork, glue if name == "fused" else _triton_glue(), draft, st)

        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            snap = st.clone()
            step()  # warm-up compiles; then restore the state the replays start from
            st.ctx_len.copy_(snap.ctx_len)
            st.tables_read.copy_(snap.tables_read)
            for x, y in zip(st.kv if isinstance(st.kv, list) else [st.kv], snap.kv if isinstance(snap.kv, list)
                            else [snap.kv]):  # fmt: skip
                x.copy_(y)
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            outs[name] = step()
        graphs[name] = g
    for t in range(2 * KP1):
        nacc32.copy_(((torch.arange(n, device=dev) + t) % KP1 + 1).to(torch.int32))
        accepted.copy_(torch.randint(0, 64, (n, KP1), device=dev, dtype=torch.int32))
        proj = (torch.randn(n * KP1, hidden, device=dev) * 2).to(torch.bfloat16)
        for name in sides:
            sides[name].projected.copy_(proj)
        for name in sides:
            graphs[name].replay()
        torch.cuda.synchronize()
        (out_a, idx_a), (out_b, idx_b) = outs["stock"], outs["fused"]
        _assert_same(sides["stock"], sides["fused"], out_a, out_b, idx_a, idx_b, f"(replay {t})")
        out_c, idx_c = outs["triton"]
        _assert_same(
            sides["triton"], sides["fused"], out_c, out_b, idx_c, idx_b, f"vs Triton (replay {t})"
        )
    assert int(sides["fused"].ctx_len[sides["fused"].slots[0]]) > int(base.ctx_len[base.slots[0]])


# ------------------------------------------------------------------------------------------------ TailGlue (worker)


def _glue_obj(tfold):
    return tfold.TailGlue(
        functools.partial(adv.advance_ctx_and_read_view, parallel=True), fidx.forward_indices, 0
    )


def _inputs_for(tfold, st):
    """This step's DRAFTER_INPUTS_FUSED call as the recording hook keeps it: its outputs and the slots / context lengths
    it read."""
    return tfold.StepInputs(
        (None, st.qpos, st.positions.view(st.n, KP1), None), st.slots, st.ctx_len
    )


@needs_gpu
def test_tailglue_defers_fuses_and_hands_back_the_indices(fold_flag):
    tfold, tfork, glue = _mods()
    dev = torch.device("cuda")
    fold_flag[0] = True
    draft = _drafter(dev, 9, 512, 64)
    st = _State(dev, 21, 2, 600, 32, 512, True, False, False, True, False)
    a, b = st.clone(), st.clone()
    store._cache_arg(a.kv)
    store._cache_arg(b.kv)
    out_a, idx_a = _stock(tfork, draft, a)
    tg = _glue_obj(tfold)
    tg.inputs = _inputs_for(tfold, b)
    prep = tfork.ctx_store_prep(draft, b.projected)
    done = tg.store(prep, draft, b.projected, b.positions, b.nacc, b.rows, None, b.tables, b.kv, KP1,
                    capacity_counts=b.counts, max_col=None)  # fmt: skip
    assert done is True and tg.pending is not None
    torch.cuda.synchronize()
    assert not any(bool((t != 0).any()) for t in b.kv_bits()), "the store ran before the advance"
    out_b = tg.advance(b.slots, b.rows, b.nacc, b.ctx_len, b.max_ctx, DUMMY, b.counts, b.base, b.tables,
                       b.tables_read, b.ps, BLOCK)  # fmt: skip
    assert tg.pending is None and tg.fused == 1
    idx_b = tg.indices(out_b[: b.n], b.rows, b.qpos, BLOCK, b.width, b.ps)
    assert idx_b is not None and tg.idx is None and tg.handed == 1
    torch.cuda.synchronize()
    _assert_same(a, b, out_a, out_b, idx_a, idx_b)
    # One-shot: the next call computes (the stock op), with equal values.
    again = tg.indices(out_b[: b.n], b.rows, b.qpos, BLOCK, b.width, b.ps)
    assert all(x is not y for x, y in zip(again, idx_b))
    assert all(torch.equal(x, y) for x, y in zip(again, idx_b))


class _Log:
    def __init__(self):
        self.lines = []

    def info(self, msg):
        self.lines.append(msg)

    warning = info


@needs_gpu
@pytest.mark.parametrize("err", [RuntimeError, TypeError], ids=["launch-error", "other-error"])
def test_a_failed_fused_launch_keeps_the_stock_order(fold_flag, monkeypatch, err):
    """A compile / launch error of the fused kernel: the deferred store, then the stock advance and indices (the same
    bytes), and this rank stays on the stock ops. Any other exception propagates with the store still pending, for the
    drafter inputs' flush (store first, as in the stock order)."""
    tfold, tfork, glue = _mods()
    dev = torch.device("cuda")
    fold_flag[0] = True
    draft = _drafter(dev, 29, 512, 64)
    st = _State(dev, 61, 1, 300, 32, 512, True, False, False, True, False)
    a, b, c = st.clone(), st.clone(), st.clone()
    for s in (a, b, c):
        store._cache_arg(s.kv)
    log = _Log()
    monkeypatch.setattr(tfold, "logger", log)
    tg = _glue_obj(tfold)
    tg.inputs = _inputs_for(tfold, b)
    prep = tfork.ctx_store_prep(draft, b.projected)
    assert tg.store(prep, draft, b.projected, b.positions, b.nacc, b.rows, None, b.tables, b.kv, KP1,
                    capacity_counts=b.counts, max_col=None)  # fmt: skip
    assert tg.pending is not None

    def boom(*args, **kwargs):
        raise err("the fused kernel failed to compile")

    monkeypatch.setattr(glue, "launch", boom)
    adv_args = (b.slots, b.rows, b.nacc, b.ctx_len, b.max_ctx, DUMMY, b.counts, b.base, b.tables, b.tables_read,
                b.ps, BLOCK)  # fmt: skip
    if err is TypeError:
        with pytest.raises(TypeError):
            tg.advance(*adv_args)
        assert tg.pending is not None and tg.broken is None
        assert tg.flush() and tg.pending is None
        _store_ops(tfork, draft, a).launch()  # the stock store alone
        torch.cuda.synchronize()
        for x, y in zip(a.kv_bits(), b.kv_bits()):
            assert torch.equal(x, y)
        assert torch.equal(b.ctx_len, st.ctx_len)
        return
    out_a, idx_a = _stock(tfork, draft, a)
    out_b = tg.advance(*adv_args)
    assert tg.broken and tg.pending is None and tg.fused == 0 and tg.declined == 1
    idx_b = tg.indices(out_b[: b.n], b.rows, b.qpos, BLOCK, b.width, b.ps)
    torch.cuda.synchronize()
    _assert_same(a, b, out_a, out_b, idx_a, idx_b, "after a failed fused launch")
    assert any("TAILGLUE_A: the fused launch failed" in x for x in log.lines)
    # From now on the store launches at once.
    prep_c = tfork.ctx_store_prep(draft, c.projected)
    assert tg.store(prep_c, draft, c.projected, c.positions, c.nacc, c.rows, None, c.tables, c.kv, KP1,
                    capacity_counts=c.counts, max_col=None)  # fmt: skip
    assert tg.pending is None


# ------------------------------------------------------------------------------------------------ wiring + switches
