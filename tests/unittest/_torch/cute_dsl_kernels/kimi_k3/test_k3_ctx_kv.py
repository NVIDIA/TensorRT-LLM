# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""``trtllm::k3_ctx_kv`` (the block drafter's context K/V of a decode step) against the DFlashWorker Python path, for
B requests of K + 1 tokens (N = B (K + 1) <= 64), at the in-model TP16 shape (5 drafter layers, 1 KV head, hidden
7168; TP4's 4 KV heads where the kernel takes it).

The Python path, as DFlashWorker runs it (modeling_dflash precompute_context_kv, dflash _store_context_kv_paged):
F.linear with the stacked K/V weight, K/V .contiguous(), F.rms_norm + k_norm, flashinfer NeoX RoPE in place, the write
mask, column clamps, one flashinfer paged append per layer, ctx_len += num_accepted (clamped), num_ctx = min(ctx_len,
counts page - block).

Every split with mixed per-request lengths (columns crossing a page), accepted counts and slots / table rows, a column
clamp at the allocation, page-major and arena pools: every written pool row against the Python path and an fp32
reference (the kernel's error must be the Python path's), masked rows zero, the pool's other elements untouched,
ctx_len and num_ctx exact, reruns bit-identical; CUDA-graph replays with rewritten inputs.

Timing: ``python3 test_k3_ctx_kv.py time``; error table: ``python3 test_k3_ctx_kv.py report``.
"""

import statistics
import sys

import pytest
import torch
import torch.nn.functional as F

HIDDEN = 7168
HEAD = 64
LAYERS = 5
PAGE = 64
EPS = 1e-6
MAX_POS = 8192
MAX_CTX = 4096
THETA = 1.0e6
BLOCK = 8

# (B requests, K + 1 tokens each): every split with B (K + 1) <= 8, DSpark's B x 8, and other draft lengths.
SPLITS = (
    [(1, 8), (2, 4), (4, 2), (8, 1), (1, 1), (2, 1), (3, 1), (4, 1), (5, 1)]
    + [(b, 8) for b in range(2, 9)]
    + [(8, 7), (2, 7), (8, 4), (3, 4)]
)


def _sm100() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 10


pytestmark = pytest.mark.skipif(not _sm100(), reason="needs SM100 (tcgen05, TMA, clusters)")


def _op():
    import tensorrt_llm  # noqa: F401  (registers the trtllm ops)
    from tensorrt_llm._torch.cute_dsl_kernels.k3_ctx_kv import op

    return op


def cos_sin_cache(device):
    inv = 1.0 / (THETA ** (torch.arange(0, HEAD, 2, dtype=torch.float64, device=device) / HEAD))
    ang = torch.arange(MAX_POS, dtype=torch.float64, device=device)[:, None] * inv[None, :]
    return torch.cat([ang.cos(), ang.sin()], dim=1).float().contiguous()


def make_pool(style, pages, nkv, gen):
    """Per-layer HND views [pages, 2, nkv, PAGE, 64] of one allocation, filled with a pattern."""
    per_page = 2 * nkv * PAGE * HEAD
    if style == "arena":
        buf = torch.randn(
            LAYERS, pages, 2, nkv, PAGE, HEAD, generator=gen, device="cuda"
        ).bfloat16()
        return buf, [buf[layer] for layer in range(LAYERS)]
    # Page-major: each layer at offset layer * per_page inside every page group.
    buf = torch.randn(pages * LAYERS * per_page, generator=gen, device="cuda").bfloat16()
    views = [
        buf.as_strided(
            (pages, 2, nkv, PAGE, HEAD),
            (LAYERS * per_page, nkv * PAGE * HEAD, PAGE * HEAD, HEAD, 1),
            layer * per_page,
        )  # fmt: skip
        for layer in range(LAYERS)
    ]
    return buf, views


def views_of(buf, like, style):
    if style == "arena":
        return [buf[layer] for layer in range(LAYERS)]
    return [buf.as_strided(t.shape, t.stride(), t.storage_offset()) for t in like]


def python_path(
    x, w, k_norm, cs, cpos, num_acc, ctx_len, slots, rows, table, counts, layers, block_size
):
    """DFlashWorker's ops for the step (the model's functions)."""
    from tensorrt_llm._torch.custom_ops import (
        flashinfer_apply_rope_with_cos_sin_cache_inplace as rope,
    )
    from tensorrt_llm._torch.speculative.dflash_attention import get_dflash_paged_append

    n, k1 = x.shape[0], cpos.shape[1]
    nkv = w.shape[0] // (LAYERS * 2 * HEAD)
    kv = F.linear(x, w).view(n, LAYERS, 2, nkv, HEAD)
    k = kv[:, :, 0].contiguous()
    v = kv[:, :, 1].contiguous()
    k = F.rms_norm(k, (HEAD,), eps=EPS)
    k = k * k_norm.view(1, LAYERS, 1, HEAD)
    pos = cpos.reshape(-1).to(torch.int32).repeat_interleave(LAYERS)
    dummy_q = k.new_zeros(n * LAYERS, HEAD)
    rope(pos, dummy_q, k.view(n * LAYERS, nkv * HEAD), HEAD, cs, True)
    offs = torch.arange(k1, device="cuda")
    mask = (offs[None, :] < num_acc.long()[:, None]).reshape(-1).view(-1, 1, 1, 1).to(k.dtype)
    k.mul_(mask)
    v.mul_(mask)
    col = ctx_len[slots][:, None] + offs[None, :]
    cap = counts[rows] * PAGE
    col = torch.minimum(col, (cap - 1)[:, None]).clamp_(min=0)
    rows_i32 = rows[:, None].expand(-1, k1).reshape(-1).to(torch.int32)
    col_i32 = col.reshape(-1).to(torch.int32)
    width = table.shape[1]
    indptr = torch.arange(0, (table.shape[0] + 1) * width, width, dtype=torch.int32, device="cuda")
    last = torch.full((table.shape[0],), PAGE, dtype=torch.int32, device="cuda")
    append = get_dflash_paged_append()
    for layer in range(LAYERS):
        append(append_key=k[:, layer].contiguous(), append_value=v[:, layer].contiguous(), batch_indices=rows_i32,
               positions=col_i32, paged_kv_cache=layers[layer], kv_indices=table.flatten().contiguous(),
               kv_indptr=indptr, kv_last_page_len=last, kv_layout="HND")  # fmt: skip
    ctx_len[slots] += num_acc.long()
    ctx_len.clamp_(max=MAX_CTX)
    allocated = (counts * PAGE - block_size).clamp(min=0)
    return torch.minimum(ctx_len[slots], allocated[rows])


def fp32_rows(x, w, k_norm, cs, cpos, nkv):
    """High-precision reference of every (token, layer, K/V, head) row: [N, L, 2, nkv, 64] fp32, no bf16 rounding
    (computed in float64: no TF32 even where cuBLAS is told to use it)."""
    n = x.shape[0]
    kv = (x.double() @ w.double().t()).view(n, LAYERS, 2, nkv, HEAD)
    k = kv[:, :, 0]
    k = (
        k
        * torch.rsqrt(k.pow(2).mean(-1, keepdim=True) + EPS)
        * k_norm.double().view(1, LAYERS, 1, HEAD)
    )
    c = cs.double()[cpos.reshape(-1)][:, None, None, : HEAD // 2]
    s = cs.double()[cpos.reshape(-1)][:, None, None, HEAD // 2 :]
    k1_, k2_ = k[..., : HEAD // 2], k[..., HEAD // 2 :]
    k = torch.cat([k1_ * c - k2_ * s, k2_ * c + k1_ * s], dim=-1)
    return torch.stack([k, kv[:, :, 1]], dim=2).float()


_views = {}


def pool_view(layers):
    """The kernel's view of a pool, built once per pool outside capture (as the worker does); a few kept."""
    key = tuple(t.data_ptr() for t in layers)
    view = _views.get(key)
    if view is None:
        if len(_views) >= 4:
            _views.clear()
        view = _views[key] = _op().pool_view(layers)
    return view


def kernel_call(
    x, w, k_norm, cs, cpos, num_acc, ctx_len, slots, rows, table, counts, layers, block_size, nkv
):
    flat, layer_off, ps, kvs, hs = pool_view(layers)
    return torch.ops.trtllm.k3_ctx_kv(x, w, k_norm, cs, cpos, num_acc, ctx_len, slots, rows, table, counts, flat,
                                      layer_off, ps, kvs, hs, EPS, MAX_CTX, PAGE, block_size, nkv)  # fmt: skip


def gather_rows(layers, table, rows, col):
    """The pool rows of each (token, layer, K/V, head): [N, L, 2, nkv, 64]."""
    page = table[rows[:, None].expand_as(col).reshape(-1), col.reshape(-1) // PAGE].long()
    off = col.reshape(-1) % PAGE
    return torch.stack([layers[layer][page, :, :, off, :] for layer in range(LAYERS)], dim=1)


class Step:
    """One decode step's inputs: B requests (distinct slots and table rows, lengths whose columns cross pages, mixed
    accepted counts), the weights, the table and the pools."""

    def __init__(self, gen, nkv, batch, k1, style="v1", clamp=False, seed=0):
        self.nkv, self.batch, self.k1, self.style = nkv, batch, k1, style
        self.n = batch * k1
        n_rows = LAYERS * 2 * nkv * HEAD
        self.w = (torch.randn(n_rows, HIDDEN, generator=gen, device="cuda") * 0.02).bfloat16()
        self.k_norm = (
            1.0 + 0.1 * torch.randn(LAYERS, HEAD, generator=gen, device="cuda")
        ).bfloat16()
        self.cs = cos_sin_cache("cuda")
        width, table_rows, slots_total, self.pages = 64, batch + 3, batch + 4, 64 * (batch + 3) + 16
        self.table = (
            torch.randperm(self.pages, generator=torch.Generator().manual_seed(seed))[: table_rows * width]
            .view(table_rows, width).to(torch.int32).cuda()
        )  # fmt: skip
        self.counts = torch.full((table_rows,), width, dtype=torch.int64, device="cuda")
        g = torch.Generator().manual_seed(seed + 1)
        self.slots = torch.randperm(slots_total, generator=g)[:batch].cuda()
        self.rows = torch.randperm(table_rows, generator=g)[:batch].cuda()
        # Lengths: a page end inside the block for every other request (64 m - 3), else random.
        ctx0 = torch.randint(100, 2000, (slots_total,), generator=g)
        for b in range(0, batch, 2):
            ctx0[int(self.slots[b])] = 64 * int(torch.randint(2, 30, (1,), generator=g)) - 3
        if clamp:  # the first request's last columns clamp to its allocation's end
            self.counts[int(self.rows[0])] = 20
            ctx0[int(self.slots[0])] = 20 * PAGE - 3
        self.ctx0 = ctx0.cuda()
        self.num_acc = (torch.arange(batch) * 3 + seed) % k1 + 1
        self.num_acc = self.num_acc.to(torch.int32).cuda()
        self.cpos = (
            self.ctx0[self.slots][:, None] + torch.arange(k1, device="cuda")[None, :]
        ).contiguous()
        self.x = (torch.randn(self.n, HIDDEN, generator=gen, device="cuda") * 0.5).bfloat16()
        self.buf, self.layers = make_pool(style, self.pages, nkv, gen)

    def cols(self):
        return torch.minimum(self.ctx0[self.slots][:, None] + torch.arange(self.k1, device="cuda")[None, :],
                             (self.counts[self.rows] * PAGE - 1)[:, None]).clamp(min=0)  # fmt: skip

    def run_kernel(self, buf=None, ctx=None, fn=None):
        buf = self.buf.clone() if buf is None else buf
        ctx = self.ctx0.clone() if ctx is None else ctx
        layers = views_of(buf, self.layers, self.style)
        call = kernel_call if fn is None else fn
        num_ctx = call(self.x, self.w, self.k_norm, self.cs, self.cpos, self.num_acc, ctx, self.slots, self.rows,
                       self.table, self.counts, layers, BLOCK, self.nkv)  # fmt: skip
        return buf, ctx, num_ctx.long(), layers

    def run_python(self):
        buf = self.buf.clone()
        ctx = self.ctx0.clone()
        layers = views_of(buf, self.layers, self.style)
        num_ctx = python_path(self.x, self.w, self.k_norm, self.cs, self.cpos, self.num_acc, ctx, self.slots,
                              self.rows, self.table, self.counts, layers, BLOCK)  # fmt: skip
        return buf, ctx, num_ctx.long(), layers


def measure(nkv, batch, k1, style="v1", clamp=False, seed=0):
    gen = torch.Generator(device="cuda").manual_seed(20260929 + 100 * batch + k1 + seed)
    st = Step(gen, nkv, batch, k1, style, clamp, seed)
    buf_p, ctx_p, nc_p, lay_p = st.run_python()
    buf_k, ctx_k, nc_k, lay_k = st.run_kernel()
    torch.cuda.synchronize()
    col = st.cols()
    ref32 = fp32_rows(st.x, st.w, st.k_norm, st.cs, st.cpos, nkv)
    got = gather_rows(lay_k, st.table, st.rows, col).float()
    want = gather_rows(lay_p, st.table, st.rows, col).float()
    n = st.n
    live = (torch.arange(k1, device="cuda")[None, :] < st.num_acc.long()[:, None]).reshape(n)
    # A clamped column is written by several tokens; compare only the tokens whose column is their own.
    cflat = col.reshape(-1)
    full = st.rows[:, None].expand_as(col).reshape(-1) * 1_000_000 + cflat
    own = torch.tensor([int((full == full[t]).sum()) == 1 for t in range(n)], device="cuda")
    cmp = (live & own).view(n, 1, 1, 1, 1).expand_as(got)
    scale = ref32.abs().amax(dim=-1, keepdim=True).clamp(min=1e-6).expand_as(got)
    e_k = ((got - ref32).abs() / scale)[cmp].max().item() if cmp.any() else 0.0
    e_p = ((want - ref32).abs() / scale)[cmp].max().item() if cmp.any() else 0.0
    e_kp = ((got - want).abs() / scale)[cmp].max().item() if cmp.any() else 0.0
    masked = (~live & own).view(n, 1, 1, 1, 1).expand_as(got)
    masked_zero = bool((got[masked] == 0).all()) if masked.any() else True
    # Every pool element the Python path left alone is left alone (and the kernel wrote no others).
    before = st.buf.view(torch.int16)
    changed_k = buf_k.view(torch.int16) != before
    changed_p = buf_p.view(torch.int16) != before
    written = torch.zeros(st.buf.numel(), dtype=torch.bool, device="cuda")
    idx = torch.arange(st.buf.numel(), device="cuda")
    pages = st.table[st.rows[:, None].expand_as(col).reshape(-1), cflat // PAGE].long()
    for layer in range(LAYERS):
        idx_l = idx.view(st.buf.shape)[layer] if st.style == "arena" else idx.as_strided(
            lay_p[layer].shape, lay_p[layer].stride(), lay_p[layer].storage_offset())  # fmt: skip
        written[idx_l[pages, :, :, cflat % PAGE, :].reshape(-1)] = True
    written = written.view(st.buf.shape)
    untouched = bool((~((changed_k | changed_p) & ~written)).all())
    buf_r, ctx_r, nc_r, _ = st.run_kernel()
    rerun = torch.equal(buf_r.view(torch.int16), buf_k.view(torch.int16)) and torch.equal(ctx_r, ctx_k) and \
        torch.equal(nc_r, nc_k)  # fmt: skip
    res = dict(split=f"{batch}x{k1}", nkv=nkv, pool=style, clamp=clamp, num_acc=st.num_acc.tolist(),
               ctx=st.ctx0[st.slots].tolist(), e_kernel=e_k, e_python=e_p, e_kernel_vs_python=e_kp,
               ctx_len=torch.equal(ctx_p, ctx_k), num_ctx=torch.equal(nc_p, nc_k), masked_zero=masked_zero,
               untouched=untouched, rerun=rerun)  # fmt: skip
    res["ok"] = (res["ctx_len"] and res["num_ctx"] and masked_zero and untouched and rerun
                 and e_k <= max(2.0 * e_p, 1.6e-2))  # fmt: skip
    return res


@pytest.mark.parametrize("batch,k1", SPLITS, ids=[f"{b}x{k}" for b, k in SPLITS])
@pytest.mark.parametrize("style", ["v1", "arena"])
def test_split(batch, k1, style):
    with torch.inference_mode():
        res = measure(1, batch, k1, style, seed=1 if style == "arena" else 0)
    assert res["ok"], res


@pytest.mark.parametrize("batch,k1", [(1, 8), (2, 8), (8, 8), (8, 1), (4, 4)])
def test_clamp(batch, k1):
    with torch.inference_mode():
        res = measure(1, batch, k1, "v1", clamp=True, seed=2)
    assert res["ok"], res


TP4_SPLITS = [(1, 8), (2, 4), (8, 1), (2, 8), (4, 8)]


@pytest.mark.parametrize("batch,k1", TP4_SPLITS, ids=[f"{b}x{k}" for b, k in TP4_SPLITS])
def test_split_tp4(batch, k1):
    with torch.inference_mode():
        res = measure(4, batch, k1, "v1", seed=3)
    assert res["ok"], res


def test_supported_shapes():
    """TP16 takes every step up to 8 x 8; TP4 (shared memory for the resident tokens) up to 32 tokens."""
    op = _op()
    dev = torch.device("cuda")
    for b, k1 in SPLITS:
        assert op.pick_split(LAYERS * 2 * HEAD, HIDDEN, 1, k1, b * k1, dev) == 8, (b, k1)
    assert op.pick_split(LAYERS * 2 * 4 * HEAD, HIDDEN, 4, 8, 32, dev) == 4
    assert op.pick_split(LAYERS * 2 * 4 * HEAD, HIDDEN, 4, 8, 64, dev) == 0
    assert op.pick_split(LAYERS * 2 * HEAD, HIDDEN, 1, 8, 72, dev) == 0  # 9 requests
    assert op.pick_split(LAYERS * 2 * HEAD, HIDDEN, 1, 1, 9, dev) == 0  # 9 requests of 1


@pytest.mark.parametrize("batch,k1", [(1, 8), (4, 8), (8, 8), (8, 1)])
def test_graph_replay(batch, k1):
    """20 replays of a captured call with the inputs rewritten in place (x, ctx_len, num_acc, cpos)."""
    with torch.inference_mode():
        gen = torch.Generator(device="cuda").manual_seed(7 + batch)
        st = Step(gen, 1, batch, k1, "v1", seed=4)
        x = st.x.clone()
        ctx = st.ctx0.clone()
        num_acc = st.num_acc.clone()
        cpos = st.cpos.clone()
        buf = st.buf.clone()
        lay = views_of(buf, st.layers, "v1")
        out = {}

        def body():
            out["nc"] = kernel_call(x, st.w, st.k_norm, st.cs, cpos, num_acc, ctx, st.slots, st.rows, st.table,
                                    st.counts, lay, BLOCK, 1)  # fmt: skip

        body()
        torch.cuda.synchronize()
        stream = torch.cuda.Stream()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            body()
        for rep in range(20):
            x.copy_((torch.randn(x.shape, generator=gen, device="cuda") * 0.5).bfloat16())
            ctx.copy_(torch.randint(100, 2000, ctx.shape, generator=gen, device="cuda"))
            num_acc.copy_(((torch.arange(batch, device="cuda") + rep) % k1 + 1).to(torch.int32))
            cpos.copy_(ctx[st.slots][:, None] + torch.arange(k1, device="cuda")[None, :])
            ref_buf = buf.clone()
            ref_lay = views_of(ref_buf, st.layers, "v1")
            ref_ctx = ctx.clone()
            want_nc = python_path(x, st.w, st.k_norm, st.cs, cpos, num_acc, ref_ctx, st.slots, st.rows, st.table,
                                  st.counts, ref_lay, BLOCK)  # fmt: skip
            graph.replay()
            torch.cuda.synchronize()
            col = torch.minimum(cpos, (st.counts[st.rows] * PAGE - 1)[:, None]).clamp(min=0)
            got = gather_rows(lay, st.table, st.rows, col).float()
            want = gather_rows(ref_lay, st.table, st.rows, col).float()
            ref32 = fp32_rows(x, st.w, st.k_norm, st.cs, cpos, 1)
            live = (torch.arange(k1, device="cuda")[None, :] < num_acc.long()[:, None]).reshape(-1)
            live = live.view(-1, 1, 1, 1, 1).expand_as(got)
            scale = ref32.abs().amax(dim=-1, keepdim=True).clamp(min=1e-6).expand_as(got)
            e_k = ((got - ref32).abs() / scale)[live].max().item()
            e_p = ((want - ref32).abs() / scale)[live].max().item()
            assert torch.equal(ctx, ref_ctx), f"replay {rep}: ctx_len"
            assert torch.equal(out["nc"].long(), want_nc.long()), f"replay {rep}: num_ctx"
            assert e_k <= max(2.0 * e_p, 1.6e-2), f"replay {rep}: {e_k} vs {e_p}"


# The elements before an offset view: no index, length or table entry of the tests holds it.
PAD = -1


def _offset(t: torch.Tensor, k: int, pads: list) -> torch.Tensor:
    """``t``'s values ``k`` elements into a buffer of their own, after ``k`` PAD elements, as a view: where the
    drafter's per-request slices (``num_accepted_tokens[num_contexts:]``, ``_batch_to_slot[num_contexts:]``) can
    start. The buffer's pad goes to ``pads`` for the caller to check."""
    flat = torch.full((k + t.numel(),), PAD, dtype=t.dtype, device=t.device)
    flat[k:] = t.reshape(-1)
    pads.append(flat[:k])
    return flat[k:].view(t.shape)


def _kernel_call_layer_off_at(k, pads):
    def call(
        x,
        w,
        k_norm,
        cs,
        cpos,
        num_acc,
        ctx_len,
        slots,
        rows,
        table,
        counts,
        layers,
        block_size,
        nkv,
    ):
        flat, layer_off, ps, kvs, hs = pool_view(layers)
        return torch.ops.trtllm.k3_ctx_kv(x, w, k_norm, cs, cpos, num_acc, ctx_len, slots, rows, table, counts, flat,
                                          _offset(layer_off, k, pads), ps, kvs, hs, EPS, MAX_CTX, PAGE, block_size,
                                          nkv)  # fmt: skip

    return call


@pytest.mark.parametrize(
    "arg", ["cpos", "num_acc", "ctx_len", "slots", "rows", "table", "counts", "layer_off"]
)
def test_index_offset(arg):
    """Each int index / length / table argument as a view 1, 2 and 3 elements into its buffer: the pool, ctx_len and
    num_ctx equal those of the call on tensors of their own, bit for bit, and the elements before the view keep their
    PAD values (no read or write outside it)."""
    with torch.inference_mode():
        gen = torch.Generator(device="cuda").manual_seed(20260929 + 37)
        st = Step(gen, 1, 3, 4, "v1", seed=5)
        want_buf, want_ctx, want_nc, _ = st.run_kernel()
        for k in (1, 2, 3):
            ctx, fn, saved, pads = None, None, None, []
            if arg == "ctx_len":
                ctx = _offset(st.ctx0, k, pads)
            elif arg == "layer_off":
                fn = _kernel_call_layer_off_at(k, pads)
            else:
                saved = getattr(st, arg)
                setattr(st, arg, _offset(saved, k, pads))
            try:
                buf, got_ctx, got_nc, _ = st.run_kernel(ctx=ctx, fn=fn)
                torch.cuda.synchronize()
            finally:
                if saved is not None:
                    setattr(st, arg, saved)
            assert torch.equal(buf.view(torch.int16), want_buf.view(torch.int16)), (
                f"{arg} at {k}: pool"
            )
            assert torch.equal(got_ctx, want_ctx), f"{arg} at {k}: ctx_len"
            assert torch.equal(got_nc, want_nc), f"{arg} at {k}: num_ctx"
            assert len(pads) == 1 and bool((pads[0] == PAD).all()), f"{arg} at {k}: pad"


# ----------------------------------------------------------------------------------------------------------------
# Timing (python3 test_k3_ctx_kv.py time) and the error table (report)
# ----------------------------------------------------------------------------------------------------------------


def time_graph(body, calls, replays=15):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        body(0)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            for i in range(calls):
                body(i)
    torch.cuda.synchronize()
    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()
    per_call = []
    for _ in range(replays):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        torch.cuda.synchronize()
        per_call.append(start.elapsed_time(end) * 1e3 / calls)
    return statistics.median(per_call), min(per_call), max(per_call)


def timing() -> None:
    """Graphs of back-to-back calls with the weight rotating over 160 MB of copies (HBM-cold), TP16."""
    _op()
    gen = torch.Generator(device="cuda").manual_seed(11)
    n_rows = LAYERS * 2 * HEAD
    copies = max(2, -(-(160 << 20) // (n_rows * HIDDEN * 2)))
    ws = [
        (torch.randn(n_rows, HIDDEN, generator=gen, device="cuda") * 0.02).bfloat16()
        for _ in range(copies)
    ]
    calls = 2 * copies
    print(f"{torch.cuda.get_device_name()}; graphs of {calls} calls, weights rotating over {copies} copies, "
          "15 replays: median (min-max) us per call")  # fmt: skip
    print("| split | N | k3_ctx_kv |")
    print("| :-- | --: | --: |")
    with torch.inference_mode():
        for b, k1 in SPLITS:
            st = Step(gen, 1, b, k1, "v1", seed=5)
            st.num_acc.fill_(max(1, k1 - 2))
            ctx = st.ctx0.clone()
            arms = [lambda i, fn=fn: fn(st.x, ws[i % copies], st.k_norm, st.cs, st.cpos, st.num_acc, ctx, st.slots,
                                         st.rows, st.table, st.counts, st.layers, BLOCK, 1)
                    for fn in (kernel_call,)]  # fmt: skip
            res = [[] for _ in arms]
            for rep in range(3):
                for a in range(len(arms)) if rep % 2 == 0 else reversed(range(len(arms))):
                    ctx.copy_(st.ctx0)
                    res[a].append(time_graph(arms[a], calls))
            cells = []
            for a in range(len(arms)):
                meds = sorted(x[0] for x in res[a])
                cells.append(
                    f"{meds[1]:.2f} ({min(x[1] for x in res[a]):.2f}-{max(x[2] for x in res[a]):.2f})"
                )
            print(f"| {b}x{k1} | {b * k1} | " + " | ".join(cells) + " |", flush=True)


def report() -> int:
    print(f"{torch.cuda.get_device_name()}")
    print(
        "| nkv | split | pool | clamp | kernel vs fp32 | Python vs fp32 | kernel vs Python | ctx_len | num_ctx | "
        "masked 0 | untouched | rerun | result |"
    )
    print("| --: | :-- | :-- | :-- | --: | --: | --: | :-- | :-- | :-- | :-- | :-- | :-- |")
    ok_all = True
    cases = [(1, b, k, s, False) for b, k in SPLITS for s in ("v1", "arena")]
    cases += [(1, b, k, "v1", True) for b, k in [(1, 8), (2, 8), (8, 8), (8, 1), (4, 4)]]
    cases += [(4, b, k, "v1", False) for b, k in TP4_SPLITS]
    with torch.inference_mode():
        for nkv, b, k, style, clamp in cases:
            r = measure(
                nkv, b, k, style, clamp, seed=1 if style == "arena" else (2 if clamp else 0)
            )
            ok_all &= r["ok"]
            print(f"| {nkv} | {b}x{k} | {style} | {clamp} | {r['e_kernel']:.2e} | {r['e_python']:.2e} | "
                  f"{r['e_kernel_vs_python']:.2e} | {r['ctx_len']} | {r['num_ctx']} | {r['masked_zero']} | "
                  f"{r['untouched']} | {r['rerun']} | {'PASS' if r['ok'] else 'FAIL'} |", flush=True)  # fmt: skip
    print("ALL PASS" if ok_all else "FAIL")
    return 0 if ok_all else 1


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "time":
        timing()
    elif len(sys.argv) > 1 and sys.argv[1] == "report":
        sys.exit(report())
    else:
        sys.exit(pytest.main([__file__, "-q", "-p", "no:cacheprovider", *sys.argv[1:]]))
