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
"""Bit-exactness of the fused Kimi K3 MLA decode q path (``k3_mla_fused_qkv``).

* The BF16 query equals the stock chain bit for bit: FlashInfer RMSNorm of q_a,
  ``fp8_quantize_1x128_r128c4`` and ``cute_dsl_mxfp8_gemm_rubin`` with the same in-cluster K split;
  the FP8 query equals the absorption of that query (``k3_mla_q_absorb``).
* The KV rows of the step are the RoPE kernel's bytes (normalized latent and rotated ``k_pe``,
  FP8) at row ``pos % 64`` of their page, for a pool interleaving several layers addressed by
  (block, layer) slots (page ids plus the layer's slot) or by the layer's own view and a page
  stride of several layers; every other byte of the pool is untouched.
* The FMHA scales and the scheduler buffers hold the RoPE kernel's values.
* The q_b tactic's K partition: the base kernel is one chain, an in-cluster stream-K split its
  cluster_k; any other tactic has none the kernel reproduces (the stock path runs).
"""

import math

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7),
    reason="needs an SM107 GPU",
)

Q_LORA, KV_LORA, ROPE, QB_ROWS, PAGE, ROW = 1536, 512, 64, 2304, 64, 576
EPS = 1e-6
STREAMK = ("streamk", (256, 64, 128), (256, 64, 64), (2, 1), True, False, "static", "m")


def _fqkv():
    from tensorrt_llm._torch.cute_dsl_kernels.rubin import k3_mla_fused_qkv

    return k3_mla_fused_qkv


def _weights(seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    w = (torch.randn(QB_ROWS, Q_LORA, device="cuda", generator=g) * 2).to(torch.float8_e4m3fn)
    # Random UE8M0 scale per (row, 32 columns): the R128c4 layout is a byte permutation of them.
    w_sf = torch.randint(
        118, 128, (QB_ROWS * Q_LORA // 32,), device="cuda", generator=g, dtype=torch.int32
    ).to(torch.uint8)
    q_ln = (1 + 0.25 * torch.randn(Q_LORA, device="cuda", generator=g)).to(torch.bfloat16)
    kv_ln = (1 + 0.25 * torch.randn(KV_LORA, device="cuda", generator=g)).to(torch.bfloat16)
    return w, w_sf, q_ln, kv_ln


def _fused_a(m, seed, scale=0.3):
    g = torch.Generator(device="cuda").manual_seed(seed)
    # A view with the fused GEMM's row stride (gate columns first), as the module passes it.
    full = (torch.randn(m, 1536 + Q_LORA + KV_LORA + ROPE, device="cuda", generator=g) * scale).to(
        torch.bfloat16
    )
    return full[:, 1536:]


def _stock_q(fa, q_ln, w, w_sf, tactic):
    from tensorrt_llm._torch.custom_ops import cute_dsl_custom_ops as cdo
    from tensorrt_llm._torch.modules.kimi_k3_fp8_quant import fp8_quantize_1x128_r128c4

    normed = torch.ops.trtllm.flashinfer_rmsnorm(fa[:, :Q_LORA], q_ln, EPS)
    act, act_sf = fp8_quantize_1x128_r128c4(normed)
    runner = cdo.CuteDSLKimiK3MXFP8RubinLinear(output_dtype=torch.bfloat16, use_tvm_ffi=True)
    inputs = [act, w, act_sf, w_sf, cdo._get_mxfp8_gemm_alpha(fa.device)]
    return runner(inputs, tactic=tactic)


def _buffers(batch):
    dev = torch.device("cuda")
    return (
        torch.full((2,), float("nan"), dtype=torch.float32, device=dev),
        torch.full((1,), float("nan"), dtype=torch.float32, device=dev),
        torch.full((batch + 1,), -7, dtype=torch.int32, device=dev),
        torch.full((batch + 1,), -7, dtype=torch.int32, device=dev),
        torch.full((1,), 7, dtype=torch.uint32, device=dev),
    )


def _w_kc(seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    return (torch.randn(12, KV_LORA, 128, device="cuda", generator=g) * 0.05).to(torch.bfloat16)


def _run(
    fa, w, w_sf, w_kc, q_ln, kv_ln, pool, page_bytes, page_table, slot, kv_lens, cos_sin, q_len, ks
):
    fq = _fqkv()
    m = fa.shape[0]
    q = torch.empty((m, QB_ROWS), dtype=torch.bfloat16, device="cuda")
    quant_q = torch.full((m, 12, KV_LORA + ROPE), 0x7F, dtype=torch.uint8, device="cuda")
    bufs = _buffers(m // q_len)
    fq.run_k3_mla_fused_qkv(
        fa,
        q_ln,
        kv_ln,
        EPS,
        EPS,
        w,
        w_sf,
        w_kc,
        quant_q,
        pool.data_ptr(),
        page_bytes,
        page_table,
        slot,
        kv_lens,
        cos_sin,
        *bufs,
        fq.host_bmm1_scale(1.0, 192),
        q_len,
        ks,
        q_out=q,
    )
    torch.cuda.synchronize()
    return q, quant_q, bufs


@pytest.mark.parametrize("m", [1, 2, 3, 5, 7, 8, 9, 12, 14, 16, 17, 21, 24, 28, 31, 32])
@pytest.mark.parametrize("k_splits", [2, 3, 4])
def test_query_matches_stock_chain(m, k_splits, monkeypatch):
    from tensorrt_llm._torch.cute_dsl_kernels.blackwell import k3_mla_q_absorb

    monkeypatch.setenv("TRTLLM_K3_DENSE_STREAMK", "1")
    w, w_sf, q_ln, kv_ln = _weights(m)
    w_kc = _w_kc(200 + m)
    fa = _fused_a(m, 100 + m)
    ref = _stock_q(fa, q_ln, w, w_sf, STREAMK + (k_splits,))
    ref_quant_q = torch.empty((m, 12, KV_LORA + ROPE), dtype=torch.uint8, device="cuda")
    k3_mla_q_absorb.k3_mla_q_absorb(
        ref.view(m, 12, 192), w_kc, ref_quant_q.view(torch.float8_e4m3fn), 128, ROPE
    )
    pool = torch.zeros((2, PAGE * ROW), dtype=torch.uint8, device="cuda")
    page_table = torch.zeros((m, 1), dtype=torch.int32, device="cuda")
    kv_lens = torch.ones((m,), dtype=torch.int32, device="cuda")
    cos_sin = torch.zeros((2, 2 * ROPE), dtype=torch.float32, device="cuda")
    cos_sin[:, 0::2] = 1.0
    q, quant_q, _ = _run(
        fa,
        w,
        w_sf,
        w_kc,
        q_ln,
        kv_ln,
        pool,
        PAGE * ROW,
        page_table,
        0,
        kv_lens,
        cos_sin,
        1,
        k_splits,
    )
    torch.testing.assert_close(q.view(torch.int16), ref.view(torch.int16), rtol=0, atol=0)
    torch.testing.assert_close(quant_q, ref_quant_q, rtol=0, atol=0)


def _kv_reference(fa, kv_ln, cos_sin, kv_lens, q_len):
    """RoPE kernel KV rows: normalized latent and GPT-J rotated k_pe, BF16, then FP8."""
    latent = torch.ops.trtllm.flashinfer_rmsnorm(fa[:, Q_LORA : Q_LORA + KV_LORA], kv_ln, EPS)
    k_pe = fa[:, Q_LORA + KV_LORA :].float()
    m = fa.shape[0]
    t = torch.arange(m, device="cuda")
    pos = kv_lens.long()[t // q_len] - q_len + t % q_len
    coef = cos_sin.view(cos_sin.shape[0], ROPE, 2)[pos, : ROPE // 2]  # [m, 32, (cos, sin)]
    x0, x1 = k_pe[:, 0::2], k_pe[:, 1::2]
    rot = torch.empty_like(k_pe)
    # cos / sin in {0, +-0.5, +-1}: every product and sum is exact, any FMA contraction agrees.
    rot[:, 0::2] = coef[..., 0] * x0 - coef[..., 1] * x1
    rot[:, 1::2] = coef[..., 0] * x1 + coef[..., 1] * x0
    rows = torch.cat([latent.float(), rot.to(torch.bfloat16).float()], dim=1)
    return rows.to(torch.float8_e4m3fn).view(torch.uint8), pos


@pytest.mark.parametrize(
    "q_len,batch,kv_len,layers,slot,addressing",
    [
        (7, 1, 67, 1, 0, "slots"),
        (7, 3, 4093, 2, 1, "slots"),
        (6, 5, 130, 2, 0, "stride"),
        (1, 32, 5, 1, 0, "slots"),
        (4, 8, 1000, 3, 2, "slots"),
        (5, 6, 700, 3, 1, "stride"),
    ],
)
def test_kv_rows_and_buffers(q_len, batch, kv_len, layers, slot, addressing):
    """Request b holds ``kv_len + 13 b`` tokens including this step's ``q_len`` (rows straddle
    page boundaries for several requests). The pool interleaves ``layers`` layers; ``slots``
    addresses it as the CuTe DSL MLA decode does (pages of 64 rows counted over (block, layer)
    slots, the page table holding each block's first slot, the layer's slot added), ``stride``
    by the layer's own view (block ids, a page stride of ``layers`` pages)."""
    m = q_len * batch
    w, w_sf, q_ln, kv_ln = _weights(7)
    fa = _fused_a(m, 300 + m, scale=1.0)
    pages_per_seq = (kv_len + 13 * (batch - 1) + PAGE - 1) // PAGE
    num_pages = batch * pages_per_seq + 3
    pool = torch.randint(0, 256, (num_pages, layers, PAGE * ROW), dtype=torch.int32, device="cuda")
    pool = pool.to(torch.uint8)
    before = pool.clone()
    g = torch.Generator(device="cuda").manual_seed(batch)
    perm = torch.randperm(num_pages, device="cuda", generator=g).to(torch.int32)
    blocks = perm[: batch * pages_per_seq].view(batch, pages_per_seq)
    if addressing == "slots":
        view, layer_in_pool = pool, slot
        page_bytes, page_table = PAGE * ROW, (blocks * layers).contiguous()
    else:
        view, layer_in_pool = pool[:, slot], 0
        page_bytes, page_table = layers * PAGE * ROW, blocks.contiguous()
    kv_lens = kv_len + 13 * torch.arange(batch, dtype=torch.int32, device="cuda")
    positions = kv_len + 13 * batch + 8
    choices = torch.tensor([0.0, 0.5, -0.5, 1.0, -1.0], device="cuda")
    cos_sin = choices[torch.randint(0, 5, (positions, 2 * ROPE), device="cuda", generator=g)]
    _, _, (bmm1, bmm2, cu_q, cu_kv, counter) = _run(
        fa,
        w,
        w_sf,
        _w_kc(9),
        q_ln,
        kv_ln,
        view,
        page_bytes,
        page_table,
        layer_in_pool,
        kv_lens,
        cos_sin,
        q_len,
        3,
    )
    ref_rows, pos = _kv_reference(fa, kv_ln, cos_sin, kv_lens, q_len)
    expect = before.clone()
    flat = expect.view(-1)
    for t in range(m):
        p = int(pos[t])
        page = int(blocks[t // q_len, p // PAGE]) * layers + slot
        start = page * PAGE * ROW + (p % PAGE) * ROW
        flat[start : start + ROW] = ref_rows[t]
    assert torch.equal(pool, expect)
    assert bmm1[0].item() == _fqkv().host_bmm1_scale(1.0, 192)
    assert bmm1[1].cpu() == (
        torch.tensor(bmm1[0].item(), dtype=torch.float32)
        * torch.tensor(1.4426950408889634, dtype=torch.float32)
    )
    assert abs(bmm1[0].item() - 1.0 / math.sqrt(192.0)) < 1e-7
    assert bmm2.item() == 1.0
    assert counter.item() == 0
    lens = kv_lens.long().cpu()
    assert cu_q.tolist() == [12 * q_len * b for b in range(batch + 1)]
    assert cu_kv.tolist() == [0] + torch.cumsum(lens, 0).tolist()


@pytest.mark.parametrize(
    "tactic,expected",
    [
        (None, 1),
        (("base", (256, 64, 128), (256, 64, 64), (2, 1), True, False, "static", "m"), 1),
        (STREAMK + (2,), 2),
        (STREAMK + (3,), 3),
        (("mixed_clusters", (256, 64, 128), (256, 64, 64), (4, 1), (2, 1), True, False, "m"), None),
        (("splitk", (256, 64, 128), (256, 64, 64), (2, 1), True, False, "static", "m", 3), None),
    ],
)
def test_qb_tactic_k_splits(tactic, expected, monkeypatch):
    """The K partition of the q_b tactic: the base kernel (and the untuned default) is one chain,
    an in-cluster stream-K split its cluster_k; every other tactic (mixed clusters, split-K, ...)
    maps to None, i.e. the stock path."""
    from types import SimpleNamespace

    from tensorrt_llm._torch.autotuner import AutoTuner
    from tensorrt_llm._torch.modules.kimi_k3_mla import kimi_k3_mla_attention as k3mla

    tuner = AutoTuner.get()
    monkeypatch.setattr(
        type(tuner), "choose_one", lambda self, op, runners, cfg, inputs, **kw: (runners[0], tactic)
    )
    w, w_sf, _, _ = _weights(1)
    q_b_proj = SimpleNamespace(weight=w, weight_scale=w_sf)
    assert k3mla._qb_gemm_k_splits(q_b_proj, 7) == expected
