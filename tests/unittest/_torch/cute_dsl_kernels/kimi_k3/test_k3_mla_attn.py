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
"""trtllm::k3_mla_attn / k3_mla_attn_out / k3_mla_attn_vb_out (Kimi K3 MLA decode attention) at the TP16 shape (6
heads per rank, latent 512 + rope 64, bf16 pool of 64-row pages) for decode steps of R requests x T tokens: against a
torch float64 reference with per-request bottom-right causal masks and against the stock CuTe DSL MLA decode
(trtllm::cute_dsl_mla_decode_fp16_blackwell); request i's rows bit-identical to the one-request call on its own
rows, pages and length; reruns bit-identical.

Batch-1 identity against the unmodified kernel: set ``K3_BASE_TRTLLM`` to an unmodified ``tensorrt_llm`` package
directory (one request's outputs must match its single-request kernel bit for bit)."""

import importlib.util
import math
import os

import pytest
import torch


def _is_sm100() -> bool:
    if not torch.cuda.is_available():
        return False
    major, minor = torch.cuda.get_device_capability()
    return major * 10 + minor in (100, 103)


pytestmark = pytest.mark.skipif(not _is_sm100(), reason="needs an SM100 / SM103 GPU")

H, LAT, ROPE, PAGE, V = 6, 512, 64, 64, 128
DQK = LAT + ROPE
SCALE = 1.0 / math.sqrt(128 + ROPE)
GATE_COL0 = 2112

# Every split of up to 8 tokens and the DSpark verify steps of R requests x 8 tokens.
SPLITS = [(1, 1), (2, 1), (3, 1), (4, 1), (5, 1), (8, 1), (1, 8), (2, 4), (4, 2)] + [
    (r, 8) for r in range(2, 9)
]
SPLIT_IDS = [f"{r}x{t}" for r, t in SPLITS]
# The TP4 shape (24 heads: 4 clusters per request, the workspace slots of several head groups) on a few splits.
ATTN_CASES = [(r, t, H) for r, t in SPLITS] + [
    (r, t, 4 * H) for r, t in ((2, 4), (8, 1), (3, 8), (8, 8))
]
# KV lengths, assigned to the requests of a step in turn: the step's rows crossing a page boundary (579, 1989), only
# the step's tokens (8, 1), one 128-row tile minus one, > 16 tiles (several per CTA of the cluster), short contexts.
LENGTHS = (1100, 64 * 9 + 3, 8, 2049, 127, 4100, 64 * 31 + 5, 300, 64, 1)


def _bits(t: torch.Tensor) -> torch.Tensor:
    return t.contiguous().view(torch.int16)


def _make_case(seed, num_requests, tokens, heads=H, row_stride=DQK, layers=1, slot=0, lens=None):
    """A step of `num_requests` x `tokens`: q [M, heads, 576] (request-major), a pool of `layers` interleaved layer
    slots with rows of `row_stride` elements, the page table as rows of an int32 [R, 2, W] buffer (the layout of
    kv_cache_block_offsets: row stride 2 W; entries past a request's pages name a page no request owns), lengths
    (`lens`, or LENGTHS in turn)."""
    gen = torch.Generator(device="cuda").manual_seed(seed)
    cpu_gen = torch.Generator().manual_seed(seed)
    if lens is None:
        lens = [max(tokens, LENGTHS[(i + seed) % len(LENGTHS)]) for i in range(num_requests)]
    pages = [(n + PAGE - 1) // PAGE for n in lens]
    total_pages = sum(pages) + 3
    width = max(pages) + 2
    pool = torch.randn(total_pages * layers, PAGE, row_stride, generator=gen, device="cuda") * 0.5
    pool = pool.bfloat16()
    perm = (torch.randperm(total_pages, generator=cpu_gen) * layers).to(torch.int32)
    offsets = torch.empty(num_requests, 2, width, dtype=torch.int32).fill_(int(perm[-1]))
    start = 0
    for i, n in enumerate(pages):
        offsets[i, :, :n] = perm[start : start + n]
        start += n
    page_table = offsets.cuda()[:, 0, :]
    seq_len = torch.tensor(lens, dtype=torch.int32, device="cuda")
    q = torch.randn(num_requests * tokens, heads, DQK, generator=gen, device="cuda") * 0.5
    return q.bfloat16(), pool, page_table, slot, seq_len


def _reference(q, pool, page_table, page_offset, seq_len, tokens):
    """float64 attention of each request's tokens over its pages; token t sees rows <= L - T + t. [M, heads, 512]."""
    outs = []
    for i, length in enumerate(seq_len.tolist()):
        pages = page_table[i, : (length + PAGE - 1) // PAGE].long() + page_offset
        kv = pool[pages].reshape(-1, pool.shape[-1])[:length, :DQK].double()
        qi = q[i * tokens : (i + 1) * tokens].double()
        s = torch.einsum("thd,ld->thl", qi, kv) * SCALE
        limit = length - tokens + torch.arange(tokens, device=q.device)
        hidden = torch.arange(length, device=q.device)[None, :] > limit[:, None]
        s = s.masked_fill(hidden[:, None, :], float("-inf"))
        outs.append(torch.einsum("thl,ld->thd", torch.softmax(s, dim=-1), kv[:, :LAT]))
    return torch.cat(outs)


def _max_rel(a, b, tokens, num_requests):
    """max over requests of max |a - b| / max |b| (per request, so a short context is not hidden by a long one)."""
    err = 0.0
    for i in range(num_requests):
        ai, bi = (
            a[i * tokens : (i + 1) * tokens].double(),
            b[i * tokens : (i + 1) * tokens].double(),
        )
        err = max(err, (ai - bi).abs().max().item() / max(bi.abs().max().item(), 1e-6))
    return err


def _attn_out(q, pool, row_stride, page_table, page_offset, seq_len):
    from tensorrt_llm._torch.cute_dsl_kernels.k3_mla import op  # noqa: F401

    m, heads = q.shape[0], q.shape[1]
    out = torch.empty(m, heads * LAT, dtype=torch.bfloat16, device="cuda")
    torch.ops.trtllm.k3_mla_attn_out(
        q.reshape(m, -1), pool.view(-1), row_stride, page_table, page_offset, seq_len, SCALE, out
    )
    return out.view(m, heads, LAT)


def _attn_vb(q, pool, row_stride, page_table, page_offset, seq_len, w_vb, gate=None):
    from tensorrt_llm._torch.cute_dsl_kernels.k3_mla import op  # noqa: F401

    m, heads = q.shape[0], q.shape[1]
    y = torch.empty(m, heads * V, dtype=torch.bfloat16, device="cuda")
    torch.ops.trtllm.k3_mla_attn_vb_out(
        q.reshape(m, -1),
        pool.view(-1),
        row_stride,
        page_table,
        page_offset,
        seq_len,
        SCALE,
        w_vb,
        y,
        gate,
        GATE_COL0,
    )
    return y


@pytest.mark.parametrize(
    "num_requests,tokens,heads", ATTN_CASES, ids=[f"{r}x{t}-h{h}" for r, t, h in ATTN_CASES]
)
def test_attn_out(num_requests, tokens, heads):
    """k3_mla_attn_out against the float64 reference, request by request against the one-request call, and rerun. Odd
    seeds use an interleaved pool (2 layers, the layer's slot as the page offset) with rows of 640 elements."""
    seed = 11 * num_requests + tokens
    layers, slot, row_stride = (2, 1, 640) if seed % 2 else (1, 0, DQK)
    q, pool, page_table, page_offset, seq_len = _make_case(
        seed, num_requests, tokens, heads, row_stride, layers, slot
    )
    out = _attn_out(q, pool, row_stride, page_table, page_offset, seq_len)
    again = _attn_out(q, pool, row_stride, page_table, page_offset, seq_len)
    rows = [slice(i * tokens, (i + 1) * tokens) for i in range(num_requests)]
    alone = torch.cat(
        [_attn_out(q[r], pool, row_stride, page_table[i : i + 1], page_offset, seq_len[i : i + 1])
         for i, r in enumerate(rows)]
    )  # fmt: skip
    ref = _reference(q, pool, page_table, page_offset, seq_len, tokens)
    torch.cuda.synchronize()
    assert _max_rel(out, ref, tokens, num_requests) <= 1e-2
    assert torch.equal(_bits(out), _bits(again))
    assert torch.equal(_bits(out), _bits(alone))


@pytest.mark.parametrize("num_requests,tokens", SPLITS, ids=SPLIT_IDS)
def test_attn_returns(num_requests, tokens):
    """k3_mla_attn (its own output; a flat page-table row when R = 1) gives k3_mla_attn_out's bits."""
    from tensorrt_llm._torch.cute_dsl_kernels.k3_mla import op  # noqa: F401

    q, pool, page_table, _, seq_len = _make_case(5 + num_requests, num_requests, tokens)
    m = q.shape[0]
    table = page_table[0] if num_requests == 1 else page_table
    out = torch.ops.trtllm.k3_mla_attn(q.view(m, -1), pool.view(-1), DQK, table, seq_len, SCALE)
    ref = _attn_out(q, pool, DQK, page_table, 0, seq_len)
    assert torch.equal(_bits(out), _bits(ref.view(m, -1)))


@pytest.mark.parametrize("gated", [False, True], ids=["plain", "gated"])
@pytest.mark.parametrize("num_requests,tokens", SPLITS, ids=SPLIT_IDS)
def test_attn_vb(num_requests, tokens, gated):
    """k3_mla_attn_vb_out against the reference (attention output rounded to bf16, v_b in float64), request by request
    against the one-request call, rerun; with the gate, bit-identical to torch's bf16 y * s."""
    seed = 7 * num_requests + tokens
    q, pool, page_table, page_offset, seq_len = _make_case(seed, num_requests, tokens)
    gen = torch.Generator(device="cuda").manual_seed(seed + 1)
    m, heads = q.shape[0], q.shape[1]
    w_vb = (torch.randn(heads, V, LAT, generator=gen, device="cuda") * 0.05).bfloat16()
    ag = (
        torch.rand(m, GATE_COL0 + heads * V, generator=gen, device="cuda").bfloat16()
        if gated
        else None
    )
    y = _attn_vb(q, pool, DQK, page_table, page_offset, seq_len, w_vb, ag)
    again = _attn_vb(q, pool, DQK, page_table, page_offset, seq_len, w_vb, ag)
    rows = [slice(i * tokens, (i + 1) * tokens) for i in range(num_requests)]
    alone = torch.cat(
        [_attn_vb(q[r], pool, DQK, page_table[i : i + 1], page_offset, seq_len[i : i + 1], w_vb,
                  ag[r] if gated else None) for i, r in enumerate(rows)]
    )  # fmt: skip
    o_ref = _reference(q, pool, page_table, page_offset, seq_len, tokens).bfloat16().double()
    y_ref = torch.einsum("thc,hvc->thv", o_ref, w_vb.double()).reshape(m, heads * V)
    torch.cuda.synchronize()
    if gated:
        plain = _attn_vb(q, pool, DQK, page_table, page_offset, seq_len, w_vb)
        assert torch.equal(_bits(y), _bits(plain * ag[:, GATE_COL0:]))
        y_ref = y_ref.bfloat16().double() * ag[:, GATE_COL0:].double()
    assert _max_rel(y, y_ref, tokens, num_requests) <= 1e-2
    assert torch.equal(_bits(y), _bits(again))
    assert torch.equal(_bits(y), _bits(alone))


@pytest.mark.parametrize("num_requests,tokens", SPLITS, ids=SPLIT_IDS)
def test_attn_vs_stock(num_requests, tokens):
    """k3_mla_attn_out against the stock CuTe DSL MLA decode on the same step (batch R, seq_len_q T); both within 1e-2
    of the float64 reference."""
    import cutlass

    from tensorrt_llm._torch.custom_ops.cute_dsl_custom_ops import CuteDSLNVMlaDecodeBlackwellRunner

    q, pool, page_table, page_offset, seq_len = _make_case(
        3 * num_requests + tokens, num_requests, tokens
    )
    m, heads = q.shape[0], q.shape[1]
    size = CuteDSLNVMlaDecodeBlackwellRunner.get_max_padded_workspace_size(
        heads, tokens, LAT, num_requests, cutlass.Float32
    )
    workspace = torch.empty(max(size, 1), dtype=torch.int8, device="cuda")
    kv = pool[:, :, :DQK]
    stock = torch.empty(num_requests, tokens, heads, LAT, dtype=torch.bfloat16, device="cuda")
    qv = q.view(num_requests, tokens, heads, DQK)
    torch.ops.trtllm.cute_dsl_mla_decode_fp16_blackwell(
        qv[..., :LAT].permute(2, 3, 1, 0), qv[..., LAT:].permute(2, 3, 1, 0), kv[..., :LAT].permute(1, 2, 0),
        kv[..., LAT:].permute(1, 2, 0), (page_table + page_offset).transpose(0, 1), seq_len,
        stock.permute(2, 3, 1, 0), workspace, heads, tokens, PAGE, SCALE, 1.0, num_requests, None, None,
    )  # fmt: skip
    out = _attn_out(q, pool, DQK, page_table, page_offset, seq_len)
    ref = _reference(q, pool, page_table, page_offset, seq_len, tokens)
    torch.cuda.synchronize()
    stock = stock.view(m, heads, LAT)
    assert _max_rel(stock, ref, tokens, num_requests) <= 1e-2
    assert _max_rel(out, stock, tokens, num_requests) <= 1e-2


def test_attn_rejects():
    """Calls outside R <= 8 requests of T <= 8 tokens, or page-table rows / lengths that do not match, are refused."""
    from tensorrt_llm._torch.cute_dsl_kernels.k3_mla import op

    q, pool, page_table, _, seq_len = _make_case(1, 2, 4)
    q2, flat = q.view(8, -1), pool.view(-1)
    assert op.supports_attn(q2, flat, DQK, page_table, seq_len)
    assert not op.supports_attn(q2, flat, DQK, page_table[:1], seq_len)  # 1 row, 2 lengths
    assert not op.supports_attn(q2, flat, DQK, page_table[0], seq_len)  # a flat row, 2 lengths
    assert op.supports_attn(q2[:6], flat, DQK, page_table, seq_len)  # 2 requests x 3 tokens
    assert not op.supports_attn(q2[:7], flat, DQK, page_table, seq_len)  # 7 tokens over 2 requests
    q16, _, table16, _, len16 = _make_case(2, 1, 16)
    assert not op.supports_attn(q16.view(16, -1), flat, DQK, table16, len16)  # T = 16
    q9, _, table9, _, len9 = _make_case(3, 9, 1)
    assert not op.supports_attn(q9.view(9, -1), flat, DQK, table9, len9)  # R = 9
    with pytest.raises(ValueError):
        _attn_out(q[:7], pool, DQK, page_table, 0, seq_len)


# Both launch modes (clusters; no_cluster past CLUSTER_WAVE clusters) at the TP16 and TP4 head counts, with folds
# (3x5: 4100 rows, 8x1, 8x8) and CTAs without a tile (1x8-h24: 5 tiles).
POISON_CASES = [(1, 1, H), (3, 5, H), (8, 1, H), (8, 8, H), (1, 8, 4 * H), (2, 4, 4 * H)]


def _fill_workspace(heads, value):
    """Fill the data words of the attention workspace: the per-CTA partial slots (fp16) and the no_cluster (m, l)
    exchange (fp32). The arrival counters after them keep their values."""
    from tensorrt_llm._torch.cute_dsl_kernels.k3_mla import k3_mla_attn_kernel as kernel
    from tensorrt_llm._torch.cute_dsl_kernels.k3_mla import op

    groups = heads // kernel.HEADS
    ws = op._attn_workspace(torch.device("cuda", torch.cuda.current_device()), groups)
    slots = kernel.MAX_REQUESTS * groups * kernel.CLUSTER
    partials = slots * kernel.WS_SLOT_ELEMS
    exchange = slots * kernel.ROWS * 2  # fp32 words
    ws[:partials].fill_(value)
    ws[partials : partials + 2 * exchange].view(torch.float32).fill_(value)


@pytest.mark.parametrize(
    "num_requests,tokens,heads", POISON_CASES, ids=[f"{r}x{t}-h{h}" for r, t, h in POISON_CASES]
)
def test_attn_workspace_poison(num_requests, tokens, heads):
    """A call reads only workspace words it wrote itself: with the partial slots and the (m, l) exchange refilled
    with NaN before each call, k3_mla_attn_out and the gated k3_mla_attn_vb_out give the bits of the same calls on a
    zero-filled workspace."""
    seed = 13 * num_requests + tokens
    q, pool, page_table, page_offset, seq_len = _make_case(seed, num_requests, tokens, heads)
    gen = torch.Generator(device="cuda").manual_seed(seed + 1)
    m = q.shape[0]
    w_vb = (torch.randn(heads, V, LAT, generator=gen, device="cuda") * 0.05).bfloat16()
    ag = torch.rand(m, GATE_COL0 + heads * V, generator=gen, device="cuda").bfloat16()
    outs = {}
    try:
        for value in (0.0, float("nan")):
            _fill_workspace(heads, value)
            o = _attn_out(q, pool, DQK, page_table, page_offset, seq_len)
            _fill_workspace(heads, value)
            y = _attn_vb(q, pool, DQK, page_table, page_offset, seq_len, w_vb, ag)
            outs[value == 0.0] = (o, y)
    finally:
        _fill_workspace(heads, 0.0)
    torch.cuda.synchronize()
    for got, want in zip(outs[False], outs[True]):
        assert not torch.isnan(got).any()
        assert torch.equal(_bits(got), _bits(want))


# no_cluster launches (more clusters than co-reside): 8 requests at 6 heads, 2 requests at 24 heads.
WRAP_CASES = [(8, 1, H), (8, 8, H), (2, 4, 4 * H)]


def _set_counters(heads, value):
    """Set every no_cluster arrival counter of the attention workspace (int32 words after the (m, l) exchange)."""
    from tensorrt_llm._torch.cute_dsl_kernels.k3_mla import k3_mla_attn_kernel as kernel
    from tensorrt_llm._torch.cute_dsl_kernels.k3_mla import op

    groups = heads // kernel.HEADS
    ws = op._attn_workspace(torch.device("cuda", torch.cuda.current_device()), groups)
    slots = kernel.MAX_REQUESTS * groups * kernel.CLUSTER
    start = (
        slots * kernel.WS_SLOT_ELEMS + 2 * slots * kernel.ROWS * 2
    )  # fp16 elements before the counters
    ws[start:].view(torch.int32).fill_(value)


@pytest.mark.parametrize(
    "num_requests,tokens,heads", WRAP_CASES, ids=[f"{r}x{t}-h{h}" for r, t, h in WRAP_CASES]
)
def test_attn_counter_wrap(num_requests, tokens, heads):
    """The no_cluster arrival counters only grow (16 per launch) and are compared by signed difference: calls with the
    counters just below the int32 wrap (2^31 - 16, and -16 just below 0) give the bits of calls on zeroed ones."""
    seed = 19 * num_requests + tokens
    q, pool, page_table, page_offset, seq_len = _make_case(seed, num_requests, tokens, heads)
    gen = torch.Generator(device="cuda").manual_seed(seed + 1)
    w_vb = (torch.randn(heads, V, LAT, generator=gen, device="cuda") * 0.05).bfloat16()
    outs = {}
    try:
        for start in (0, 2**31 - 16, -16):
            _set_counters(heads, start)
            # Three launches: each crosses the wrap point once the counters start 16 below it.
            outs[start] = [
                _attn_out(q, pool, DQK, page_table, page_offset, seq_len),
                _attn_vb(q, pool, DQK, page_table, page_offset, seq_len, w_vb),
                _attn_out(q, pool, DQK, page_table, page_offset, seq_len),
            ]
    finally:
        _set_counters(heads, 0)
    torch.cuda.synchronize()
    for start in (2**31 - 16, -16):
        for got, want in zip(outs[start], outs[0]):
            assert torch.equal(_bits(got), _bits(want))


# ----------------------------------------------------------------------------------------------------------------
# Batch-1 identity against the unmodified kernel (K3_BASE_TRTLLM: an unmodified tensorrt_llm package directory).
# ----------------------------------------------------------------------------------------------------------------

_base = {}


def _base_kernel():
    """The unmodified single-request kernel module, loaded from K3_BASE_TRTLLM."""
    if "mod" not in _base:
        path = os.path.join(
            os.environ["K3_BASE_TRTLLM"],
            "_torch",
            "cute_dsl_kernels",
            "k3_mla",
            "k3_mla_attn_kernel.py",
        )
        spec = importlib.util.spec_from_file_location("k3_mla_attn_kernel_base", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _base["mod"] = mod
    return _base["mod"]


def _base_attn(q, pool, row_stride, page_row, page_offset, seq_len, out, w_vb=None, gate=None):
    """The unmodified op's launch (one request: one 16-byte aligned page-table row, seq_len [1]); q [M, heads * 576]."""
    import cuda.bindings.driver as cuda_driver
    import cutlass.cute as cute
    from cutlass.cute.runtime import from_dlpack

    kern = _base_kernel()

    def arg(t):
        return from_dlpack(t.detach(), assumed_align=16).mark_layout_dynamic(
            leading_dim=t.dim() - 1
        )

    heads = q.shape[1] // kern.QK
    groups = heads // kern.HEADS
    if ("ws", groups) not in _base:
        _base["ws", groups] = torch.empty(
            groups * kern.CLUSTER * kern.WS_SLOT_ELEMS, dtype=torch.float16, device=q.device
        )
    fuse_vb, apply_gate = w_vb is not None, gate is not None
    gate_flat = gate.as_strided((gate.numel(),), (1,)) if apply_gate else q.view(-1)
    args = (arg(q.view(-1)), arg(pool.view(-1)[: PAGE * row_stride]), arg(page_row.reshape(-1)),
            arg(seq_len.reshape(-1)), arg(_base["ws", groups]), arg(out.view(-1)),
            arg((w_vb if fuse_vb else q).view(-1)), arg(gate_flat))  # fmt: skip
    scalars = (q.shape[0], SCALE * kern.LOG2E, pool.numel() // row_stride, page_offset, GATE_COL0,
               gate.stride(0) if apply_gate else 0)  # fmt: skip
    use_pdl = os.environ.get("TRTLLM_ENABLE_PDL", "1") == "1"
    stream = cuda_driver.CUstream(torch.cuda.current_stream().cuda_stream)
    key = (row_stride, heads, fuse_vb, apply_gate, use_pdl)
    fn = _base.get(key)
    if fn is None:
        fn = _base[key] = cute.compile(kern.k3_mla_attn, *args, *scalars, row_stride, heads, fuse_vb, apply_gate,
                                       use_pdl, stream)  # fmt: skip
    fn(*args, *scalars, stream)
    return out


@pytest.mark.skipif(
    not os.environ.get("K3_BASE_TRTLLM"), reason="K3_BASE_TRTLLM (unmodified package) not set"
)
@pytest.mark.parametrize("tokens", [8, 1, 2, 4, 7])
def test_batch1_identity(tokens):
    """One request: k3_mla_attn_out and k3_mla_attn_vb_out (plain and gated) bit-identical to the unmodified kernel
    for every length in LENGTHS, rows of 576 and interleaved rows of 640 with a page offset, the page-table row given
    flat, as a [1, W] row, and at a 4-byte offset (a row of a wider table)."""
    from tensorrt_llm._torch.cute_dsl_kernels.k3_mla import op  # noqa: F401

    for i, length in enumerate(LENGTHS):
        layers, slot, row_stride = (2, 1, 640) if i % 2 else (1, 0, DQK)
        q, pool, table, off, seq_len = _make_case(
            900 + 10 * tokens + i,
            1,
            tokens,
            H,
            row_stride,
            layers,
            slot,
            lens=[max(tokens, length)],
        )
        gen = torch.Generator(device="cuda").manual_seed(1900 + 10 * tokens + i)
        w_vb = (torch.randn(H, V, LAT, generator=gen, device="cuda") * 0.05).bfloat16()
        ag = torch.rand(tokens, GATE_COL0 + H * V, generator=gen, device="cuda").bfloat16()
        shifted = torch.zeros(table.shape[1] + 1, dtype=torch.int32, device="cuda")
        shifted[1:] = table[0]
        q2 = q.view(tokens, -1)
        want = _base_attn(q2, pool, row_stride, table[0], off, seq_len,
                          torch.empty(tokens, H * LAT, dtype=torch.bfloat16, device="cuda"))  # fmt: skip
        for form, row in (
            ("flat", table[0]),
            ("[1, W]", table[:1]),
            ("4-byte offset", shifted[1:]),
        ):
            got = _attn_out(q, pool, row_stride, row, off, seq_len).view(tokens, -1)
            assert torch.equal(_bits(got), _bits(want)), f"attn_out L {length} {form}"
        for gate in (None, ag):
            want = _base_attn(q2, pool, row_stride, table[0], off, seq_len,
                              torch.empty(tokens, H * V, dtype=torch.bfloat16, device="cuda"), w_vb, gate)  # fmt: skip
            got = _attn_vb(q, pool, row_stride, table[:1], off, seq_len, w_vb, gate)
            assert torch.equal(_bits(got), _bits(want)), (
                f"attn_vb L {length} gate {gate is not None}"
            )
