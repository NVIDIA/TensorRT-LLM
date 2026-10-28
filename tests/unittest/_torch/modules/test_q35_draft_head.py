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
"""Qwen3.5 MXFP8 draft head: streaming GEMV, fused shard argmax and key decode."""

import os

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)),
    reason="needs SM100/SM103",
)

N_VOCAB_SHARD = 31040  # Qwen3.5 248320 / TP8
K = 4096
INT64_MIN = -(1 << 63)


def _mod():
    from tensorrt_llm._torch.modules import q35_draft_head as m

    return m


def _weights(n, k=K, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    w = (torch.randn((n, k), generator=g, device="cuda") * 0.02).to(torch.bfloat16)
    return w


def _acts(m, k=K, seed=1, scale=3.0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    return (torch.randn((m, k), generator=g, device="cuda") * scale).to(torch.bfloat16)


def _ref_logits(x, values, scales):
    deq = _mod().dequantize_mxfp8_rows(values, scales).double()
    return x.double() @ deq.T


def _keys_to_ids(keys):
    return ((~keys) & 0xFFFFFFFF).to(torch.int64)


@pytest.mark.parametrize("m", [1, 9, 16])
def test_logits_match_dequantized_reference(m):
    mod = _mod()
    w = _weights(N_VOCAB_SHARD)
    values, scales = mod.quantize_mxfp8_rows(w)
    x = _acts(m)
    out = torch.empty((m, N_VOCAB_SHARD), dtype=torch.float32, device="cuda")
    mod.mxfp8_head_gemv(x, values, scales, out, argmax=False)
    torch.cuda.synchronize()
    ref = _ref_logits(x, values, scales)
    # Exact products, FP32 sums of 4096 terms: bound by a few ulps of sum |x||w|.
    bound = x.double().abs() @ mod.dequantize_mxfp8_rows(values, scales).double().abs().T
    err = (out.double() - ref).abs()
    assert torch.all(err <= 1e-5 * bound + 1e-6), float((err / (bound + 1e-9)).max())


@pytest.mark.parametrize("m", [1, 16])
def test_argmax_keys_match_reference(m):
    mod = _mod()
    w = _weights(N_VOCAB_SHARD, seed=3)
    values, scales = mod.quantize_mxfp8_rows(w)
    x = _acts(m, seed=4)
    keys = torch.full((m,), INT64_MIN, dtype=torch.int64, device="cuda")
    offset = 5 * N_VOCAB_SHARD
    mod.mxfp8_head_gemv(
        x, values, scales, keys.view(torch.int32).view(m, 2), argmax=True, vocab_offset=offset
    )
    torch.cuda.synchronize()
    ref = _ref_logits(x, values, scales)
    ids = _keys_to_ids(keys) - offset
    ref_max = ref.max(dim=1).values
    got = ref.gather(1, ids[:, None].clamp(0, N_VOCAB_SHARD - 1)).squeeze(1)
    # The chosen column must be a maximum up to FP32 summation noise.
    assert torch.all((ids >= 0) & (ids < N_VOCAB_SHARD))
    assert torch.all(ref_max - got <= 1e-5 * ref_max.abs() + 1e-5), (ids, ref.argmax(1))
    # Value word: the orderable FP32 bits of the logit.
    hi = (keys >> 32).to(torch.int32)
    bits = torch.where(hi < 0, hi ^ 0x7FFFFFFF, hi)
    vals = bits.view(torch.float32).double()
    assert torch.allclose(vals, got, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("m", [1, 16])
def test_keys_match_ll_exchange_packing(m):
    """The fused keys are bit-identical to comm's pack_argmax_keys of the same kernel's FP32 logits."""
    from tensorrt_llm._torch.distributed.ll_exchange import pack_argmax_keys

    mod = _mod()
    w = _weights(N_VOCAB_SHARD, seed=41)
    values, scales = mod.quantize_mxfp8_rows(w)
    x = _acts(m, seed=42)
    logits = torch.empty((m, N_VOCAB_SHARD), dtype=torch.float32, device="cuda")
    mod.mxfp8_head_gemv(x, values, scales, logits, argmax=False)
    keys = torch.full((m,), INT64_MIN, dtype=torch.int64, device="cuda")
    offset = 3 * N_VOCAB_SHARD
    mod.mxfp8_head_gemv(
        x, values, scales, keys.view(torch.int32).view(m, 2), argmax=True, vocab_offset=offset
    )
    vmax, imax = torch.max(logits, dim=-1)
    ref = pack_argmax_keys(vmax, imax, offset)
    torch.cuda.synchronize()
    assert torch.equal(keys, ref), (keys, ref)


def test_ties_resolve_to_smallest_id_across_ctas_and_shards():
    mod = _mod()
    n = N_VOCAB_SHARD
    w = _weights(n, seed=5)
    x = _acts(1, seed=6)
    # Make two far-apart rows (different CTAs) identical winners.
    win = (x.float() / x.float().norm()).to(torch.bfloat16)[0] * 4
    w[9000] = win
    w[20000] = win
    values, scales = mod.quantize_mxfp8_rows(w)
    keys = torch.full((1,), INT64_MIN, dtype=torch.int64, device="cuda")
    mod.mxfp8_head_gemv(x, values, scales, keys.view(torch.int32).view(1, 2), argmax=True)
    torch.cuda.synchronize()
    assert int(_keys_to_ids(keys)[0]) == 9000
    # Shards: the same winner in shards 2 and 6 -> the global id of shard 2.
    shard_keys = []
    for r in range(8):
        k = torch.full((1,), INT64_MIN, dtype=torch.int64, device="cuda")
        wr = _weights(n, seed=100 + r)
        if r in (2, 6):
            wr[123] = win
        v, s = mod.quantize_mxfp8_rows(wr)
        mod.mxfp8_head_gemv(
            x, v, s, k.view(torch.int32).view(1, 2), argmax=True, vocab_offset=r * n
        )
        shard_keys.append(k)
    gathered = torch.cat(shard_keys)
    local = torch.full((1,), 7, dtype=torch.int64, device="cuda")
    tokens = mod.decode_keys(gathered, 8, local)
    torch.cuda.synchronize()
    assert int(tokens[0]) == 2 * n + 123
    assert int(local[0]) == INT64_MIN  # re-armed


@pytest.mark.parametrize("m", [4])
@pytest.mark.parametrize("ranks,shard", [(16, 15520), (8, 31040)])
def test_sharded_argmax_equals_global_argmax(m, ranks, shard):
    """TP8 / TP16 shards of the 248320-entry vocabulary."""
    mod = _mod()
    w = _weights(ranks * shard, seed=7)
    n = shard
    x = _acts(m, seed=8)
    keys = []
    refs = []
    for r in range(ranks):
        v, s = mod.quantize_mxfp8_rows(w[r * n : (r + 1) * n].contiguous())
        k = torch.full((m,), INT64_MIN, dtype=torch.int64, device="cuda")
        mod.mxfp8_head_gemv(
            x, v, s, k.view(torch.int32).view(m, 2), argmax=True, vocab_offset=r * n
        )
        keys.append(k)
        refs.append(_ref_logits(x, v, s))
    local = torch.empty((m,), dtype=torch.int64, device="cuda")
    tokens = mod.decode_keys(torch.cat(keys), ranks, local)
    torch.cuda.synchronize()
    ref = torch.cat(refs, dim=1)
    got = ref.gather(1, tokens.long()[:, None]).squeeze(1)
    assert torch.all(ref.max(1).values - got <= 1e-5 * ref.max(1).values.abs() + 1e-5)


def test_negative_control_scales_are_used():
    """Positive/negative control: blowing up one row's scales must move the argmax to that row
    (proves the kernel reads the per-block scales and every CTA's tiles)."""
    mod = _mod()
    w = _weights(N_VOCAB_SHARD, seed=9)
    x = _acts(1, seed=10)
    values, scales = mod.quantize_mxfp8_rows(w)
    keys = torch.full((1,), INT64_MIN, dtype=torch.int64, device="cuda")
    mod.mxfp8_head_gemv(x, values, scales, keys.view(torch.int32).view(1, 2), argmax=True)
    base = int(_keys_to_ids(keys)[0])
    ref = _ref_logits(x, values, scales)[0]
    target = int(torch.argmin(ref))  # the smallest logit; make it the largest by flipping sign
    values2 = values.clone()
    values2[target] = (-(values[target].float())).to(torch.float8_e4m3fn)
    scales2 = scales.clone()
    scales2[target] = (scales[target].int() + 6).clamp(max=254).to(torch.uint8)
    keys.fill_(INT64_MIN)
    mod.mxfp8_head_gemv(x, values2, scales2, keys.view(torch.int32).view(1, 2), argmax=True)
    torch.cuda.synchronize()
    moved = int(_keys_to_ids(keys)[0])
    assert moved == target != base


def test_graph_replay_rearms_keys():
    mod = _mod()
    w = _weights(N_VOCAB_SHARD, seed=11)
    values, scales = mod.quantize_mxfp8_rows(w)

    class _Head(mod.Q35DraftHead):
        def __init__(self):  # noqa: D401 - test double without an LMHead
            self.values, self.scales = values, scales
            self.vocab_offset = 0
            self.k = K
            self.mapping = None
            self.keys = torch.full((mod.MAX_ROWS,), INT64_MIN, dtype=torch.int64, device="cuda")
            self.slice_blocks = mod.default_slice_blocks(K)

    head = _Head()
    xs = torch.empty((2, K), dtype=torch.bfloat16, device="cuda")
    xs.copy_(_acts(2, seed=12))
    head.sample(xs)  # warmup / compile
    stream = torch.cuda.Stream()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(stream):
        with torch.cuda.graph(graph, stream=stream):
            tokens = head.sample(xs)
    for seed in range(13, 18):
        xs.copy_(_acts(2, seed=seed))
        graph.replay()
        torch.cuda.synchronize()
        ref = _ref_logits(xs, values, scales)
        got = ref.gather(1, tokens.long()[:, None]).squeeze(1)
        assert torch.all(ref.max(1).values - got <= 1e-5 * ref.max(1).values.abs() + 1e-5)
        assert torch.all(head.keys == INT64_MIN)


def test_pdl_waits_for_a_delayed_producer():
    """A producer that releases its dependents immediately and writes the activation ~200 us later:
    the PDL-launched head must still read the new activation (no load above griddepcontrol.wait).
    Positive control: a PDL-launched copy without a dependency wait reads the old activation."""
    import triton
    import triton.language as tl

    if os.environ.get("TRTLLM_ENABLE_PDL", "1") != "1":
        pytest.skip("PDL disabled")
    try:
        from triton.language.extra.cuda import gdc_launch_dependents
    except ImportError:
        pytest.skip("Triton without gdc_launch_dependents")

    @triton.jit
    def _late_writer(src, dst, n, delay_ns: tl.constexpr, BLOCK: tl.constexpr):
        gdc_launch_dependents()
        t0 = tl.inline_asm_elementwise(
            "mov.u64 $0, %globaltimer;", "=l,r", [n], dtype=tl.int64, is_pure=False, pack=1
        )
        t = t0
        while t - t0 < delay_ns:
            t = tl.inline_asm_elementwise(
                "mov.u64 $0, %globaltimer;", "=l,r", [n], dtype=tl.int64, is_pure=False, pack=1
            )
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        tl.store(dst + offs, tl.load(src + offs, mask=offs < n), mask=offs < n)

    @triton.jit
    def _copy_no_wait(src, dst, n, BLOCK: tl.constexpr):
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        tl.store(dst + offs, tl.load(src + offs, mask=offs < n), mask=offs < n)

    mod = _mod()
    w = _weights(N_VOCAB_SHARD, seed=21)
    values, scales = mod.quantize_mxfp8_rows(w)
    x_old = _acts(1, seed=22)
    x_new = _acts(1, seed=23)
    ref_new = int(_ref_logits(x_new, values, scales).argmax(1)[0])
    ref_old = int(_ref_logits(x_old, values, scales).argmax(1)[0])
    assert ref_new != ref_old
    xs = x_old.clone()
    keys = torch.full((1,), INT64_MIN, dtype=torch.int64, device="cuda")
    snap = torch.empty_like(xs)

    def head():
        mod.mxfp8_head_gemv(xs, values, scales, keys.view(torch.int32).view(1, 2), argmax=True)

    def copy_no_wait():
        _copy_no_wait[(K // 1024,)](xs, snap, K, BLOCK=1024, launch_pdl=True)

    def run(consumer, observe):
        # Producer and consumer back to back in one CUDA graph (eager host launch gaps exceed the delay).
        consumer()
        _late_writer[(K // 1024,)](x_new, xs, K, delay_ns=200_000, BLOCK=1024)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            with torch.cuda.graph(graph, stream=stream):
                _late_writer[(K // 1024,)](x_new, xs, K, delay_ns=200_000, BLOCK=1024)
                consumer()
        got = []
        for _ in range(5):
            xs.copy_(x_old)
            keys.fill_(INT64_MIN)
            torch.cuda.synchronize()
            graph.replay()
            torch.cuda.synchronize()
            got.append(observe())
        return got

    stale = run(copy_no_wait, lambda: torch.equal(snap, x_old))
    assert any(stale), "the delayed producer did not overlap a PDL dependent: the test is not valid"
    assert run(head, lambda: int(_keys_to_ids(keys)[0])) == [ref_new] * 5


def _mx_head(monkeypatch, mx_rows=64, seed=51):
    """Q35DraftHead (TP 1) with TRTLLM_Q35_DRAFT_HEAD_MX_ROWS=mx_rows."""
    import types

    mod = _mod()
    monkeypatch.setenv("TRTLLM_Q35_DRAFT_HEAD_MX_ROWS", str(mx_rows))
    w = _weights(N_VOCAB_SHARD, seed=seed)
    head = types.SimpleNamespace(
        weight=w, out_features=N_VOCAB_SHARD, num_embeddings=N_VOCAB_SHARD, tp_size=1
    )
    return mod, mod.Q35DraftHead(head, None)


@pytest.mark.parametrize("rows,cols", [(64, 128), (130, 9), (N_VOCAB_SHARD, K // 32)])
def test_blocked_scales_layout(rows, cols):
    """blocked_scales against the closed form of cuBLAS's 128 x 4 tiled block-scale layout."""
    mod = _mod()
    s = torch.randint(0, 255, (rows, cols), dtype=torch.uint8, device="cuda")
    got = mod.blocked_scales(s)
    nc = -(-cols // 4)
    r = torch.arange(rows, device="cuda")[:, None]
    c = torch.arange(cols, device="cuda")[None, :]
    off = ((r // 128) * nc + c // 4) * 512 + (r % 32) * 16 + ((r % 128) // 32) * 4 + c % 4
    assert got.numel() == -(-rows // 128) * 128 * nc * 4
    assert torch.equal(got[off.flatten()], s.flatten())
    pad = torch.ones(got.numel(), dtype=torch.bool, device="cuda")
    pad[off.flatten()] = False
    assert torch.all(got[pad] == 127)


@pytest.mark.parametrize("m", [17, 33, 48, 64, 128])
def test_mx_quant_matches_quantize_mxfp8_rows(m):
    """The Triton activation quantizer = quantize_mxfp8_rows, bit for bit (values; scales in the tiled layout),
    including an all-zero block and block maxima with significand exactly 1.75 and just above it."""
    mod = _mod()
    x = _acts(m, seed=60 + m)
    x[0, :32] = 0
    x[1, 32:64] = 0
    x[1, 40] = 1.75 * 2**3  # amax / 448 = 2^-5 exactly: e = -5
    x[2, 64:96] = 0.01
    x[2, 70] = -1.7578125 * 2**-5  # significand above 1.75: e rounds up
    x[3, 96:128] = 1e-30  # small exponents (e = -108)
    ref_v, ref_s = mod.quantize_mxfp8_rows(x)
    q = torch.empty((m, K), dtype=torch.float8_e4m3fn, device="cuda")
    blk = torch.full((128 * (K // 32),), 127, dtype=torch.uint8, device="cuda")
    mod.mxfp8_quantize_rows_blocked(x, q, blk)
    torch.cuda.synchronize()
    assert torch.equal(q.view(torch.uint8), ref_v.view(torch.uint8))
    assert torch.equal(blk, mod.blocked_scales(ref_s))


@pytest.mark.parametrize("m", [17, 48, 64])
def test_mx_logits_match_reference(monkeypatch, m):
    """cuBLAS MXFP8 logits = dequantized MXFP8 activation x dequantized head copy (exact products, FP32 sums)."""
    mod, dh = _mx_head(monkeypatch)
    x = _acts(m, seed=70 + m)
    logits = dh.mx_logits(x)
    torch.cuda.synchronize()
    xd = mod.dequantize_mxfp8_rows(*mod.quantize_mxfp8_rows(x)).double()
    wd = mod.dequantize_mxfp8_rows(dh.values, dh.scales).double()
    ref = xd @ wd.T
    bound = xd.abs() @ wd.abs().T
    err = (logits.double() - ref).abs()
    assert logits.dtype == torch.float32 and logits.shape == (m, N_VOCAB_SHARD)
    assert torch.all(err <= 1e-5 * bound + 1e-6), float((err / (bound + 1e-9)).max())


@pytest.mark.parametrize("m", [17, 33, 48, 64])
def test_mx_sample_is_argmax_of_mx_logits(monkeypatch, m):
    """Q35DraftHead.sample above 16 rows (TP 1): the argmax (lowest id on ties) of its MXFP8 logits."""
    _, dh = _mx_head(monkeypatch)
    assert dh.supports(m) and dh.supports(16) and not dh.supports(65)
    x = _acts(m, seed=80 + m)
    out = torch.empty(m, dtype=torch.int32, device="cuda")
    tokens = dh.sample(x, out=out)
    logits = dh.mx_logits(x)
    torch.cuda.synchronize()
    assert tokens.data_ptr() == out.data_ptr()
    assert torch.equal(tokens.long(), logits.argmax(dim=-1))


def test_mx_rows_default_off(monkeypatch):
    import types

    mod = _mod()
    monkeypatch.delenv("TRTLLM_Q35_DRAFT_HEAD_MX_ROWS", raising=False)
    w = _weights(N_VOCAB_SHARD, seed=52)
    head = types.SimpleNamespace(
        weight=w, out_features=N_VOCAB_SHARD, num_embeddings=N_VOCAB_SHARD, tp_size=1
    )
    dh = mod.Q35DraftHead(head, None)
    assert dh.mx_rows == 0 and dh.supports(16) and not dh.supports(17)


def test_mx_graph_replay(monkeypatch):
    """The 17..64-row path captured in a CUDA graph follows new activations on replay."""
    _, dh = _mx_head(monkeypatch)
    m = 48
    xs = torch.empty((m, K), dtype=torch.bfloat16, device="cuda")
    xs.copy_(_acts(m, seed=90))
    dh.sample(xs)  # warmup / compile
    torch.cuda.synchronize()
    stream = torch.cuda.Stream()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(stream):
        with torch.cuda.graph(graph, stream=stream):
            tokens = dh.sample(xs)
    for seed in range(91, 95):
        xs.copy_(_acts(m, seed=seed))
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(tokens.long(), dh.mx_logits(xs).argmax(dim=-1))
