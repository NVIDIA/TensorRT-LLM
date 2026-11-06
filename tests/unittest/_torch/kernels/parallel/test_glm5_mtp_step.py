# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The GLM-5.3-Flash fused MTP draft step kernels (``trtllm::glm5_mtp_*``, one rank) against torch references at the
model's shapes with random weights: every kernel's outputs, the all-reduce buffers it pushes to and re-marks."""

import pytest
import torch
from utils.util import skip_pre_hopper

import tensorrt_llm  # noqa: F401  (registers the trtllm ops)

pytestmark = skip_pre_hopper

ops = torch.ops.trtllm

H = 4096
MAX_ROWS, MAX_RANKS, BUFS = 8, 8, 3
AR_ELEMS = BUFS * MAX_RANKS * MAX_ROWS * H
EPS = 1e-5
SCRATCH_WORDS = 256 * 8 + 2
dev = "cuda"


def bf(x):
    return x.to(torch.bfloat16)


def rand(*shape, scale=1.0):
    return bf(torch.randn(*shape, device=dev) * scale)


def rms(x, g, eps=EPS):
    xf = x.float()
    return bf(xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps) * g.float())


def close(got, want, tol=1e-2):
    got, want = got.float(), want.float()
    err = (got - want).abs().max().item()
    scale = want.abs().max().item() + 1e-6
    assert err / scale < tol, f"rel err {err / scale:.3e} (abs {err:.3e}, scale {scale:.3e})"


def ar_buffer():
    return torch.full((AR_ELEMS,), -0.0, dtype=torch.bfloat16, device=dev)


def ar_slot(buf, b, src, rows):
    return buf.view(BUFS, MAX_RANKS, MAX_ROWS, H)[b, src, :rows]


def is_empty(t):
    return bool((t.view(torch.int16) == -32768).all())


def push(buf, b, partial):
    """What the pushing kernels write (single rank): the bf16 partial, -0.0 as +0.0."""
    p = partial.clone()
    p[p == 0] = 0.0
    ar_slot(buf, b, 0, p.shape[0]).copy_(p)


@pytest.fixture(autouse=True)
def seed():
    torch.manual_seed(0)


@pytest.mark.parametrize("m", [1, 3, 8])
@pytest.mark.parametrize("use_embed,col", [(True, 0), (True, 2048), (False, 0), (False, 2048)])
def test_pre(m, use_embed, col):
    vocab = 1000
    embed = rand(vocab, H)
    hidden = rand(m, H)
    tokens = torch.randint(0, vocab, (m,), dtype=torch.int32, device=dev)
    if m > 1:
        tokens[1] = -1  # a rejected slot embeds as zeros
    gamma = rand(H, scale=0.1) + 1
    w = rand(H, H // 2, scale=0.02)
    buf = ar_buffer()
    ops.glm5_mtp_pre(
        m, tokens, 1, hidden, embed, use_embed, gamma, EPS, col, w, [buf.data_ptr()], 0
    )
    if use_embed:
        src = torch.where(
            (tokens >= 0)[:, None], embed[tokens.clamp(min=0).long()], torch.zeros_like(embed[:1])
        )
    else:
        src = hidden
    x = rms(src, gamma)[:, col : col + H // 2]
    want = bf(x.float() @ w.float().T)
    torch.cuda.synchronize()
    close(ar_slot(buf, 0, 0, m), want)


@pytest.mark.parametrize("m", [1, 4, 8])
def test_qkv_a(m):
    buf = ar_buffer()
    partial = rand(m, H)
    push(buf, 0, partial)
    ar_slot(buf, 2, 0, MAX_ROWS).fill_(
        1.0
    )  # the previous all-reduce's buffer: re-marked empty by the reader
    gamma = rand(H, scale=0.1) + 1
    wq = rand(2048, H, scale=0.02)
    wi = rand(288, H, scale=0.02)
    resid = torch.empty(m, H, dtype=torch.bfloat16, device=dev)
    out = torch.empty(m, 2048 + 256, dtype=torch.bfloat16, device=dev)
    ops.glm5_mtp_qkv_a(m, [buf.data_ptr()], 0, gamma, EPS, resid, wq, wi, 256, out)
    torch.cuda.synchronize()
    assert torch.equal(resid, partial)
    x = rms(partial, gamma)
    close(out, bf(x.float() @ torch.cat([wq, wi[:256]]).float().T))
    assert is_empty(ar_slot(buf, 2, 0, MAX_ROWS))


@pytest.mark.parametrize("m", [1, 5, 8])
def test_q_b(m):
    qkv = rand(m, 1536 + 512 + 256)
    qa_g, kv_g = rand(1536, scale=0.1) + 1, rand(512, scale=0.1) + 1
    ln_w, ln_b = rand(128, scale=0.1) + 1, rand(128, scale=0.1)
    w = rand(4096, 1536, scale=0.02)
    latent = torch.empty(m, 512, dtype=torch.bfloat16, device=dev)
    packed = torch.empty(m, 256, dtype=torch.bfloat16, device=dev)
    query = torch.empty(m, 4096, dtype=torch.bfloat16, device=dev)
    wq = rand(4096, 1536, scale=0.02)  # the indexer's wq_b (step 0)
    q_index = torch.empty(m, 4096, dtype=torch.bfloat16, device=dev)
    ops.glm5_mtp_q_b(
        qkv,
        1536,
        512,
        128,
        qa_g,
        kv_g,
        EPS,
        ln_w,
        ln_b,
        1e-6,
        latent,
        packed,
        w,
        query,
        wq,
        q_index,
    )
    torch.cuda.synchronize()
    q = rms(qkv[:, :1536], qa_g)
    close(query, bf(q.float() @ w.float().T))
    close(q_index, bf(q.float() @ wq.float().T))
    close(latent, rms(qkv[:, 1536:2048], kv_g), tol=5e-3)
    k = torch.nn.functional.layer_norm(
        qkv[:, 2048:2176].float(), (128,), ln_w.float(), ln_b.float(), 1e-6
    )
    close(packed[:, :128], bf(k), tol=5e-3)
    assert torch.equal(packed[:, 128:], qkv[:, 2176:])


@pytest.mark.parametrize("m", [1, 8])
def test_absorb_and_v_up(m):
    heads, nope, kv, v = 16, 256, 512, 256
    kv_b = rand(heads * (nope + v), kv, scale=0.05)
    per_head = kv_b.view(heads, nope + v, kv)
    w_k = per_head[:, :nope, :]
    w_v_t = per_head[:, nope:, :].transpose(1, 2)
    wk_t = w_k.transpose(1, 2).contiguous()  # [heads, kv, nope]
    query = rand(m, heads * nope)
    q_lat = torch.empty(m, heads * kv, dtype=torch.bfloat16, device=dev)
    ops.glm5_mtp_absorb(query, wk_t, q_lat)
    want = torch.bmm(query.view(m, heads, nope).transpose(0, 1).float(), w_k.float()).transpose(
        0, 1
    )
    torch.cuda.synchronize()
    close(q_lat.view(m, heads, kv), bf(want))
    o_lat = rand(m, heads * kv)
    vout = torch.empty(m, heads * v, dtype=torch.bfloat16, device=dev)
    ops.glm5_mtp_v_up(o_lat, kv_b, heads, nope, v, vout)
    want = torch.bmm(o_lat.view(m, heads, kv).transpose(0, 1).float(), w_v_t.float()).transpose(
        0, 1
    )
    torch.cuda.synchronize()
    close(vout.view(m, heads, v), bf(want))


@pytest.mark.parametrize("m", [1, 8])
def test_o_proj(m):
    buf = ar_buffer()
    x = rand(m, 4096)
    w = rand(H, 4096, scale=0.02)
    ops.glm5_mtp_o_proj(x, w, [buf.data_ptr()], 0)
    torch.cuda.synchronize()
    close(ar_slot(buf, 1, 0, m), bf(x.float() @ w.float().T))


def swiglu(g, u, lim=10.0):
    return torch.nn.functional.silu(g.clamp(max=lim)) * u.clamp(-lim, lim)


ROUTE_WORDS = 4096


def run_router(m, buf, bias, start, local=72):
    """The router kernel on a pushed all-reduce buffer 1: (inputs, outputs, route table)."""
    partial = rand(m, H)
    push(buf, 1, partial)
    ar_slot(buf, 0, 0, MAX_ROWS).fill_(1.0)
    res_in = rand(m, H)
    gamma = rand(H, scale=0.1) + 1
    gate_w = rand(288, H, scale=0.02)
    gate_up = rand(1024, H, scale=0.02)
    res_out = torch.empty_like(res_in)
    norm = torch.empty_like(res_in)
    logits = torch.empty(m, 288, dtype=torch.float32, device=dev)
    shared = torch.empty(m, 512, dtype=torch.bfloat16, device=dev)
    route = torch.zeros(ROUTE_WORDS, dtype=torch.int32, device=dev)
    ops.glm5_mtp_router([buf.data_ptr()], 0, res_in, res_out, gamma, EPS, norm, gate_w, logits, gate_up, 10.0, shared,
                        bias, 8, 2.5, start, local, route)  # fmt: skip
    torch.cuda.synchronize()
    return (partial, res_in, gamma, gate_w, gate_up), (res_out, norm, logits, shared), route


@pytest.mark.parametrize("m", [1, 6, 8])
def test_router(m):
    buf = ar_buffer()
    (partial, res_in, gamma, gate_w, gate_up), (res_out, norm, logits, shared), route = run_router(
        m, buf, torch.zeros(288, device=dev), 72
    )
    summed = bf(partial.float() + res_in.float())
    assert torch.equal(res_out, summed)
    x = rms(summed, gamma)
    close(norm, x, tol=5e-3)
    close(logits, x.float() @ gate_w.float().T)
    gu = bf(x.float() @ gate_up.float().T).float()
    close(shared, bf(swiglu(gu[:, :512], gu[:, 512:])))
    assert is_empty(ar_slot(buf, 0, 0, MAX_ROWS))


def fp8_block_cast(w, block=128):
    rows, cols = w.shape[-2:]
    t = w.float().reshape(*w.shape[:-2], rows // block, block, cols // block, block)
    s = t.abs().amax(dim=(-3, -1), keepdim=True).clamp(min=1e-4) / 448.0
    q = (t / s).to(torch.float8_e4m3fn).reshape(w.shape)
    return q, s.reshape(*w.shape[:-2], rows // block, cols // block)


def dequant(q, s, block=128):
    rows, cols = q.shape[-2:]
    t = q.float().reshape(*q.shape[:-2], rows // block, block, cols // block, block)
    return (t * s.reshape(*s.shape[:-2], rows // block, 1, cols // block, 1)).reshape(q.shape)


def quant_rows(x, block=128):
    t = x.float().view(x.shape[0], -1, block)
    s = t.abs().amax(-1, keepdim=True) / 448.0
    s = torch.where(s > 0, s, torch.ones_like(s))
    return (t / s).to(torch.float8_e4m3fn).float().mul(s).view_as(x)


def routing_ref(logits, bias, top_k, scale):
    s = torch.sigmoid(logits)
    idx = torch.topk(s + bias, top_k, dim=-1).indices
    w = s.gather(1, idx)
    return idx, w / w.sum(-1, keepdim=True) * scale


@pytest.mark.parametrize("m", [1, 4, 8])
def test_experts(m):
    E, local, rank = 288, 72, 1
    start = rank * local
    bias = torch.randn(E, device=dev) * 0.1
    # Steer every row to a few of this rank's experts so each exercises local slots.
    bias[start : start + 4] += 4.0
    buf = ar_buffer()
    for _ in range(2):  # twice: the router's tile counter must be back at zero
        _, (_, x, logits, _), route = run_router(m, buf, bias, start, local)
        buf = ar_buffer()
    w13_bf = rand(local, 4096, H, scale=0.02)
    w2_bf = rand(local, H, 2048, scale=0.02)
    w13, w13_s = fp8_block_cast(w13_bf)
    w2, w2_s = fp8_block_cast(w2_bf)
    shared_in = rand(m, 512)
    shared_down = rand(H, 512, scale=0.05)
    inter = torch.zeros(64, 8, 2048, dtype=torch.float32, device=dev)
    shared_out = torch.empty(m, H, dtype=torch.bfloat16, device=dev)
    ops.glm5_mtp_fc1(
        route, 8, start, x, w13, w13_s, 10.0, inter, shared_in, shared_down, shared_out
    )
    ops.glm5_mtp_fc2(m, route, 8, start, w2, w2_s, inter, shared_out, [buf.data_ptr()], 0)
    torch.cuda.synchronize()
    close(shared_out, bf(shared_in.float() @ shared_down.float().T))
    # Reference: same input quantization, dequantized weights, fp32 math.
    idx, wts = routing_ref(logits, bias, 8, 2.5)
    xq = quant_rows(x)
    w13_f, w2_f = dequant(w13, w13_s), dequant(w2, w2_s)
    routed = torch.zeros(m, H, device=dev)
    for t in range(m):
        for k in range(8):
            e = int(idx[t, k])
            if not start <= e < start + local:
                continue
            h = w13_f[e - start] @ xq[t]
            a = swiglu(h[2048:], h[:2048])
            a = quant_rows(a[None])[0]
            routed[t] += wts[t, k] * (w2_f[e - start] @ a)
    want = bf(bf(routed).float() + shared_out.float())
    close(ar_slot(buf, 2, 0, m), want, tol=2e-2)


@pytest.mark.parametrize("m", [1, 3, 8])
def test_head(m):
    buf = ar_buffer()
    partial = rand(m, H)
    push(buf, 2, partial)
    ar_slot(buf, 1, 0, MAX_ROWS).fill_(1.0)
    res_in = rand(m, H)
    gamma = rand(H, scale=0.1) + 1
    n, offset = 38720, 77440
    w = rand(n, H, scale=0.02)
    hidden = torch.empty(m, H, dtype=torch.bfloat16, device=dev)
    tokens = torch.full((m,), -1, dtype=torch.int32, device=dev)
    scratch = torch.zeros(SCRATCH_WORDS, dtype=torch.int64, device=dev)
    for _ in range(2):
        ops.glm5_mtp_head(
            [buf.data_ptr()], 0, res_in, gamma, EPS, hidden, w, offset, tokens, 1, scratch, [], 0
        )
    torch.cuda.synchronize()
    x = rms(bf(partial.float() + res_in.float()), gamma)
    close(hidden, x, tol=5e-3)
    logits = hidden.float() @ w.float().T
    top2 = logits.topk(2, dim=-1)
    clear = (top2.values[:, 0] - top2.values[:, 1]) > 1e-2
    assert torch.equal(tokens.long()[clear], (top2.indices[:, 0] + offset)[clear])
    assert is_empty(ar_slot(buf, 1, 0, MAX_ROWS))
