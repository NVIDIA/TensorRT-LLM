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
import re
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3 as _family
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_1_flash_nvfp4__sm_107__tp4 import (
    native_ops,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.ops.dspark import (
    dspark_ctx_advance as adv,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.ops.dspark import (
    dspark_ctx_store as store,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.ops.dspark import (
    dspark_drafter_inputs as dinp,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.ops.dspark import (
    dspark_forward_indices as fidx,
)

TARGET = "minimax_m3_1_flash_nvfp4__sm_107__tp4"
_PKG = "tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3." + TARGET
_DIR = Path(_family.__file__).resolve().parent / TARGET

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


@needs_gpu
@pytest.mark.parametrize(
    "n,width,ps,hidden,rot,per_layer,dummy_req,at_cap,rebase,small_counts,fold", CASES
)
@pytest.mark.parametrize("parallel", [True, False], ids=["advpar", "onelaunch"])
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


def _offset_view(t, elems):
    """A contiguous copy of t that starts `elems` elements into a fresh allocation."""
    buf = torch.zeros(t.numel() + elems, dtype=t.dtype, device=t.device)
    v = buf[elems:].view(t.shape)
    v.copy_(t)
    return v


@needs_gpu
def test_misaligned_pointer_list_cache_equals_the_four_launches(fold_flag):
    """Per-layer caches at a 2-byte offset (no 8-byte alignment): the launch takes the generic store, same bytes."""
    _, tfork, glue = _mods()
    dev = torch.device("cuda")
    fold_flag[0] = True
    draft = _drafter(dev, 3, 512, 64)
    st = _State(dev, 5, 2, 300, 32, 512, True, False, False, True, False)
    a, b = st.clone(), st.clone()
    b.kv = [_offset_view(t, 1) for t in b.kv]
    assert all(t.data_ptr() % 8 != 0 for t in b.kv)
    store._cache_arg(a.kv)
    store._cache_arg(b.kv)
    out_a, idx_a = _stock(tfork, draft, a)
    out_b, idx_b = _fused(tfork, glue, draft, b)
    torch.cuda.synchronize()
    _assert_same(a, b, out_a, out_b, idx_a, idx_b, "(misaligned per-layer caches)")
    assert any(bool((t != 0).any()) for t in b.kv_bits()), "the store wrote nothing"


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


def _dot():
    spec = importlib.util.spec_from_file_location(
        "_m31_r6_graph_dot", Path(__file__).parent / "_m31_r6_graph_dot.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@needs_gpu
def test_graph_holds_one_glue_kernel_for_four(fold_flag):
    tfold, tfork, glue = _mods()
    dot = _dot()
    dev = torch.device("cuda")
    fold_flag[0] = True
    draft = _drafter(dev, 23, 512, 64)
    st = _State(dev, 51, 1, 16416, 32, 512, True, False, False, True, False)
    names = {}
    for side in ("stock", "fused"):
        s = st.clone()
        store._cache_arg(s.kv)
        fn = (
            (lambda s=s: _stock(tfork, draft, s))
            if side == "stock"
            else (lambda s=s: _fused(tfork, glue, draft, s))
        )
        _, nodes, _ = dot.capture(fn)
        names[side] = [v["name"] for v in nodes.values() if v["type"] == "KERNEL"]
    # The catalog's stock glue kernels, by their Triton or C++ (S10c / ADV_PAR) names.
    glue_keys = ("_ctx_store_kernel", "dsparkCtxStore", "_advance_only_kernel", "dsparkAdvanceOnly",
                 "_read_view_kernel", "dsparkReadView", "_indices_kernel", "dsparkIndices")  # fmt: skip
    stock_glue = [n for n in names["stock"] if any(k in n for k in glue_keys)]
    assert len(stock_glue) == 4, names["stock"]
    assert [n for n in names["fused"] if "tailGlueKernel" in n] and len(names["fused"]) == len(
        names["stock"]
    ) - 3
    assert not any(any(k in n for k in glue_keys) for n in names["fused"])


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


@needs_gpu
@pytest.mark.parametrize(
    "variant",
    [
        "no-store",
        "other-counts",
        "no-inputs",
        "explicit-cols",
        "indices-other-qpos",
        "flush",
        "other-slots",
        "other-ctx-len",
        "other-positions",
    ],
)
def test_tailglue_every_other_path_is_the_stock_one(fold_flag, variant):
    tfold, tfork, glue = _mods()
    dev = torch.device("cuda")
    fold_flag[0] = True
    draft = _drafter(dev, 13, 512, 64)
    st = _State(dev, 31, 1, 300, 32, 512, True, False, False, True, False)
    a, b = st.clone(), st.clone()
    store._cache_arg(a.kv)
    store._cache_arg(b.kv)
    out_a, idx_a = _stock(tfork, draft, a)
    tg = _glue_obj(tfold)
    tg.inputs = None if variant == "no-inputs" else _inputs_for(tfold, b)
    if (
        variant == "other-slots"
    ):  # DRAFTER_INPUTS_FUSED read other slots than the advance's (equal values, another tensor)
        tg.inputs.slots = b.slots.clone()
    elif variant == "other-ctx-len":
        tg.inputs.ctx_len = b.ctx_len.clone()
    elif (
        variant == "other-positions"
    ):  # the store's positions are not DRAFTER_INPUTS_FUSED's context positions
        tg.inputs.outs = (None, b.qpos, b.positions.clone().view(b.n, KP1), None)
    prep = tfork.ctx_store_prep(draft, b.projected)
    counts = b.counts.clone() if variant == "other-counts" else b.counts
    if variant == "explicit-cols":
        cols = torch.minimum(b.positions, b.counts[b.rows].repeat_interleave(KP1) * b.ps - 1).clamp(
            min=0
        )
        done = tg.store(
            prep, draft, b.projected, b.positions, b.nacc, b.rows, cols, b.tables, b.kv, KP1
        )
        assert done is True and tg.pending is None  # launched at once
    elif variant != "no-store":
        done = tg.store(prep, draft, b.projected, b.positions, b.nacc, b.rows, None, b.tables, b.kv, KP1,
                        capacity_counts=b.counts, max_col=None)  # fmt: skip
        assert done is True and tg.pending is not None
    else:
        _store_ops(tfork, draft, b).launch()
    if variant == "flush":
        assert tg.flush() and tg.pending is None and not tg.flush()
    out_b = tg.advance(b.slots, b.rows, b.nacc, b.ctx_len, b.max_ctx, DUMMY, counts, b.base, b.tables,
                       b.tables_read, b.ps, BLOCK)  # fmt: skip
    fused = variant in ("indices-other-qpos",)
    assert tg.fused == (1 if fused else 0)
    if variant in ("other-counts", "no-inputs", "other-slots", "other-ctx-len", "other-positions"):
        assert tg.declined == 1 and tg.last_decline
    qpos = b.qpos.clone() if variant == "indices-other-qpos" else b.qpos
    idx_b = tg.indices(out_b[: b.n], b.rows, qpos, BLOCK, b.width, b.ps)
    assert tg.handed == 0
    torch.cuda.synchronize()
    _assert_same(a, b, out_a, out_b, idx_a, idx_b, variant)


@needs_gpu
def test_store_operands_launch_is_ctx_store_launch(fold_flag):
    """The refactored store launch: StoreOperands.launch == ctx_store_launch == the whole fused_ctx_store."""
    tfold, tfork, glue = _mods()
    dev = torch.device("cuda")
    fold_flag[0] = True
    draft = _drafter(dev, 17, 512, 64)
    st = _State(dev, 41, 2, 64, 32, 512, True, False, False, False, False)
    a, b, c = st.clone(), st.clone(), st.clone()
    for s in (a, b, c):
        store._cache_arg(s.kv)
    assert store.fused_ctx_store(draft, a.projected, a.positions, a.nacc, a.rows, None, a.tables, a.kv, KP1,
                                 capacity_counts=a.counts, max_col=None)  # fmt: skip
    prep = tfork.ctx_store_prep(draft, b.projected)
    assert tfork.ctx_store_launch(prep, draft, b.projected, b.positions, b.nacc, b.rows, None, b.tables, b.kv, KP1,
                                  capacity_counts=b.counts, max_col=None)  # fmt: skip
    _store_ops(tfork, draft, c).launch()
    torch.cuda.synchronize()
    for x, y, z in zip(a.kv_bits(), b.kv_bits(), c.kv_bits()):
        assert torch.equal(x, y) and torch.equal(x, z)
    # Declines pass through unchanged.
    assert tfork.ctx_store_operands(None, draft, b.projected, b.positions, b.nacc, b.rows, None, b.tables, b.kv,
                                    KP1, capacity_counts=b.counts) is None  # fmt: skip
    assert tfork.ctx_store_operands(prep, draft, b.projected, b.positions, b.nacc, b.rows, None, b.tables, b.kv,
                                    KP1) is False  # fmt: skip


@needs_gpu
def test_the_fused_launch_runs_only_over_s10c_c_positions_and_store_rows(fold_flag):
    """The read view's context length before the advance is the store's first position of each request: the fused
    launch needs those positions to be DRAFTER_INPUTS_FUSED's, from the advance's own slots and context lengths, and the
    store's rows (its grid) to be the requests' K + 1 rows. Host checks only; nothing is launched here."""
    tfold, tfork, glue = _mods()
    dev = torch.device("cuda")
    fold_flag[0] = True
    draft = _drafter(dev, 19, 512, 64)
    b = _State(dev, 43, 2, 300, 32, 512, True, False, False, True, False)
    store._cache_arg(b.kv)
    tg = _glue_obj(tfold)
    p = _store_ops(tfork, draft, b)

    def why(**over):
        args = dict(slots=b.slots, ctx_len=b.ctx_len)
        args.update(over)
        return tg._unfit(p, args["slots"], b.rows, b.nacc, args["ctx_len"], b.counts, b.base, b.tables,
                         b.tables_read, b.ps)  # fmt: skip

    tg.inputs = None
    assert why() == "no DRAFTER_INPUTS_FUSED call this step"
    tg.inputs = _inputs_for(tfold, b)
    assert why() is None
    assert "slots or context lengths" in why(slots=b.slots.clone())
    assert "slots or context lengths" in why(ctx_len=b.ctx_len.clone())
    tg.inputs.outs = (None, b.qpos, b.positions.clone().view(b.n, KP1), None)
    assert "context positions" in why()
    tg.inputs = _inputs_for(tfold, b)
    grid = p.grid
    p.grid = (grid[0] + 1,) + tuple(grid[1:])
    assert "store rows" in why()
    p.grid = grid
    tg.broken = "RuntimeError: x"
    assert "failed earlier" in why()


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


@needs_gpu
def test_engagement_lines_report_the_captured_path(fold_flag, monkeypatch):
    """The fused launch's and the hand-back's lines are logged once, the first time each is taken under capture (the
    eager warmup before a capture does not count)."""
    tfold, tfork, glue = _mods()
    dev = torch.device("cuda")
    fold_flag[0] = True
    draft = _drafter(dev, 37, 512, 64)
    log = _Log()
    monkeypatch.setattr(tfold, "logger", log)
    tg = _glue_obj(tfold)
    for capt in (False, True, True):
        monkeypatch.setattr(tfold, "capturing", lambda capt=capt: capt)
        b = _State(dev, 71, 1, 300, 32, 512, True, False, False, True, False)
        store._cache_arg(b.kv)
        tg.inputs = _inputs_for(tfold, b)
        prep = tfork.ctx_store_prep(draft, b.projected)
        tg.store(prep, draft, b.projected, b.positions, b.nacc, b.rows, None, b.tables, b.kv, KP1,
                 capacity_counts=b.counts, max_col=None)  # fmt: skip
        out = tg.advance(b.slots, b.rows, b.nacc, b.ctx_len, b.max_ctx, DUMMY, b.counts, b.base, b.tables,
                         b.tables_read, b.ps, BLOCK)  # fmt: skip
        tg.indices(out[: b.n], b.rows, b.qpos, BLOCK, b.width, b.ps)
        tg.end_step()
        taken = [x for x in log.lines if x.startswith("[M3 dispatch] TAILGLUE_A taken: ")]
        handed = [
            x for x in log.lines if x.startswith("[M3 dispatch] TAILGLUE_A indices handed back: ")
        ]
        assert len(taken) == len(handed) == (1 if capt else 0)
    assert tg.fused == 3 and tg.handed == 3
    assert taken[0].endswith("(1 requests, width 300) on rank 0") and handed[0].endswith(
        "(1 requests) on rank 0"
    )
    torch.cuda.synchronize()


class _Deferred:
    """A deferred store (the advance launches it on the stock path)."""

    def __init__(self):
        self.launched = 0

    def launch(self):
        self.launched += 1


def _host_glue(tfold, glue, monkeypatch, rank=2):
    """A TailGlue whose fused launch and fit check are host stand-ins (the lines are host logic only); returns it, the
    advance's arguments and a step function step(capturing, why, deferred=True) -> the advance's result."""
    tg = tfold.TailGlue(lambda *a: "stock", None, rank)
    q = torch.zeros(1, BLOCK, dtype=torch.int64)
    rows = torch.zeros(1, dtype=torch.int64)
    tables = torch.zeros(1, 300, dtype=torch.int32)
    monkeypatch.setattr(glue, "launch", lambda *a: (torch.zeros(1, dtype=torch.int64), ("i",) * 7))
    adv_args = (None, rows, None, None, None, DUMMY, None, None, tables, None, 32, BLOCK)

    def step(capt, why, deferred=True):
        monkeypatch.setattr(tfold, "capturing", lambda: capt)
        monkeypatch.setattr(tg, "_unfit", lambda *a: why)
        tg.inputs = tfold.StepInputs((None, q, None, None), None, None)
        tg.pending = _Deferred() if deferred else None
        out = tg.advance(*adv_args)
        tg.end_step()
        return out

    return tg, step


def test_eager_and_captured_stock_lines(monkeypatch):
    """Beside the captured take's line, once per rank each: the first step with a deferred store run eagerly after
    that take (taken, or the stock path with why) and the first one captured with the stock path. Nothing before the
    captured take (the eager warmup), nothing for a step without a deferred store."""
    tfold, _, glue = _mods()
    log = _Log()
    monkeypatch.setattr(tfold, "logger", log)
    tg, step = _host_glue(tfold, glue, monkeypatch)
    eager = "[M3 dispatch] TAILGLUE_A at an eager step after the capture: "
    capt_stock = "[M3 dispatch] TAILGLUE_A stock path while a CUDA graph is captured ("

    def lines(prefix):
        return [x for x in log.lines if x.startswith(prefix)]

    assert (
        step(False, None) is not None
        and step(False, "no DRAFTER_INPUTS_FUSED call this step") == "stock"
    )
    assert step(False, None, deferred=False) == "stock"  # no deferred store: not counted, no line
    assert log.lines == [] and (tg.fused, tg.declined) == (1, 1)
    step(True, None)  # the captured take
    assert len(lines("[M3 dispatch] TAILGLUE_A taken: ")) == 1 and len(log.lines) == 1
    step(True, None)
    assert len(log.lines) == 1
    step(True, "no DRAFTER_INPUTS_FUSED call this step")
    step(True, "another reason")
    assert lines(capt_stock) == [
        capt_stock + "no DRAFTER_INPUTS_FUSED call this step; fused 3, stock 2) on rank 2"
    ]
    step(False, "the advance's slots or context lengths are not the ones DRAFTER_INPUTS_FUSED read")
    step(False, None)
    step(False, "a later reason")
    assert lines(eager) == [
        eager
        + "stock path (the advance's slots or context lengths are not the ones DRAFTER_INPUTS_FUSED read) "
        "(fused 3, stock 4) on rank 2"
    ]
    assert len(log.lines) == 3
    # The eager step after the capture that takes the fused launch.
    log.lines.clear()
    tg2, step2 = _host_glue(tfold, glue, monkeypatch, rank=0)
    step2(True, None)
    step2(False, None)
    step2(False, "x")
    assert lines(eager) == [eager + "taken (fused 2, stock 0) on rank 0"]
    assert not lines(capt_stock)


# ------------------------------------------------------------------------------------------------ wiring + switches


def _src(name):
    return (_DIR / name).read_text()


def _flat(text):
    """Source text without blanks and parentheses (the formatter's line wraps)."""
    return re.sub(r"[\s()]", "", text)


def test_worker_wiring():
    src = _flat(_src("dspark.py"))
    for needle in (
        "self._tfold_hook_inputs(s10c_c_drafter_inputs)",
        "else self._tfold_hook_advance(",
        "(self._tg.indices if self._tg is not None else forward_indices)",
        "(self._tg.store if self._tg is not None else ctx_store_launch)(",
        "self._tg.flush()",
    ):
        assert _flat(needle) in src, needle


def test_hook_advance_follows_the_switch(monkeypatch):
    tfold, _, _ = _mods()
    dspark = importlib.import_module(_PKG + ".dspark")
    w = dspark.M3DSparkWorker.__new__(dspark.M3DSparkWorker)
    torch.nn.Module.__init__(w)
    stock = functools.partial(adv.advance_ctx_and_read_view, parallel=True)
    # Pin TAILGLUE_A on for the taken case (default ON; the environment may carry its kill switch).
    monkeypatch.setattr(tfold, "TAILGLUE_A", True)
    assert w._tfold_hook_advance(stock, True, True, True) == w._tg.advance
    assert w._tg.stock_advance is stock and w._tg.stock_indices is fidx.forward_indices
    for args in ((False, True, True), (True, False, True), (True, True, False)):
        assert w._tfold_hook_advance(stock, *args) is stock and w._tg is None
    monkeypatch.setattr(tfold, "TAILGLUE_A", False)
    assert w._tfold_hook_advance(stock, True, True, True) is stock and w._tg is None
    inputs_fn = dinp.drafter_inputs
    monkeypatch.setattr(tfold, "MEMCPY86K", False)
    assert w._tfold_hook_inputs(inputs_fn) is inputs_fn
    monkeypatch.setattr(tfold, "MEMCPY86K", True)
    hook = w._tfold_hook_inputs(inputs_fn)
    assert hook is not inputs_fn and hook.inner is inputs_fn


# Each item's default: TAILGLUE_A and MEMCPY86K on; MEMCPY4B off (opt-in).
_DEFAULTS = {"TAILGLUE_A": True, "MEMCPY86K": True, "MEMCPY4B": False}


@pytest.mark.parametrize("item", ["TAILGLUE_A", "MEMCPY86K", "MEMCPY4B"])
def test_kill_switches(monkeypatch, item):
    sw = importlib.import_module(_PKG + ".m31_switches")
    tfold = importlib.import_module(_PKG + ".dspark_tailfold")
    try:
        for o in _DEFAULTS:
            monkeypatch.delenv(f"TRTLLM_M31_ENABLE_{o}", raising=False)
            monkeypatch.delenv(f"TRTLLM_M31_DISABLE_{o}", raising=False)
        importlib.reload(sw)
        importlib.reload(tfold)
        assert {o: getattr(tfold, o) for o in _DEFAULTS} == _DEFAULTS
        if not _DEFAULTS[item]:
            monkeypatch.setenv(f"TRTLLM_M31_ENABLE_{item}", "1")
            importlib.reload(sw)
            importlib.reload(tfold)
            assert getattr(tfold, item) is True
            assert all(getattr(tfold, o) is _DEFAULTS[o] for o in _DEFAULTS if o != item)
        monkeypatch.setenv(f"TRTLLM_M31_DISABLE_{item}", "1")
        importlib.reload(sw)
        importlib.reload(tfold)
        assert getattr(tfold, item) is False
        assert all(getattr(tfold, o) is _DEFAULTS[o] for o in _DEFAULTS if o != item)
    finally:
        monkeypatch.undo()
        importlib.reload(sw)
        importlib.reload(tfold)
    assert tfold.TAILGLUE_A == (not sw.m31_disabled("TAILGLUE_A"))


def test_engagement_and_setup_lines():
    tfold, _, _ = _mods()
    w = SimpleNamespace(mapping=SimpleNamespace(tp_rank=3))
    on = tfold.tailfold_setup_line(w, True, True, True)
    off = tfold.tailfold_setup_line(w, False, False, False)
    assert on[0].startswith("M3.1 DSpark TAILGLUE_A: ") and on[0].endswith(" on on rank 3")
    assert on[1].startswith("M3.1 DSpark MEMCPY86K: ") and on[1].endswith(" on on rank 3")
    assert on[2].startswith("M3.1 DSpark MEMCPY4B: ") and on[2].endswith(" on on rank 3")
    assert all(x.endswith(" off on rank 3") for x in off[:2])
    assert off[2].startswith("M3.1 DSpark MEMCPY4B: ")
    assert off[2].endswith(" off on rank 3" + tfold._memcpy4b_off_reason())
    src = _src("dspark_tailfold.py")
    assert (
        "[M3 dispatch] TAILGLUE_A taken: ctx store + context advance + read view + forward indices in one"
        in src
    )
    assert (
        "[M3 dispatch] TAILGLUE_A indices handed back: the drafter's forward indices come from the fused"
        in src
    )
    dsrc = _src("dspark.py")
    assert (
        "[M3 dispatch] MEMCPY86K taken: drafter residual stream starts in the DRAFTER_INPUTS_FUSED "
        in dsrc
    )
    assert (
        "[M3 dispatch] MEMCPY4B taken: seq_lens copy for the draft forward taken on the fork's stream"
        in dsrc
    )


class _Pending:
    def __init__(self):
        self.launched = 0

    def launch(self):
        self.launched += 1


@pytest.mark.parametrize("raises", [False, True])
def test_drafter_inputs_never_return_with_a_deferred_store(raises):
    """A store deferred but not taken by an advance (any path, an exception included) is launched before the drafter
    inputs return: the drafter forward reads the cache next."""
    tfold, _, _ = _mods()
    dspark = importlib.import_module(_PKG + ".dspark")
    w = dspark.M3DSparkWorker.__new__(dspark.M3DSparkWorker)
    torch.nn.Module.__init__(w)
    w._tg = tfold.TailGlue(None, None, 0)
    p = _Pending()

    def inner(*args, **kwargs):
        w._tg.pending = p
        if raises:
            raise RuntimeError("inner failed")
        return {"noise_embedding": None}

    w._tf_prepare_1st_drafter_inputs = inner
    if raises:
        with pytest.raises(RuntimeError):
            w.prepare_1st_drafter_inputs(draft_model=None)
    else:
        assert w.prepare_1st_drafter_inputs(draft_model=None) == {"noise_embedding": None}
    assert p.launched == 1 and w._tg.pending is None
    w._tg.pending = p
    w._tfold_end_step(None)
    assert p.launched == 2 and w._tg.pending is None and w._tg.inputs is None
