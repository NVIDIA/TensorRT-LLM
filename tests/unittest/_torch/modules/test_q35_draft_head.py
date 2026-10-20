# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the Qwen3.5 MXFP8 draft head (streaming GEMV + fused shard argmax + key decode).

Run: pytest tests/unittest/_torch/modules/test_q35_draft_head.py
Microbenchmark: python tests/unittest/_torch/modules/test_q35_draft_head.py bench
"""

import os
import sys

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


@pytest.mark.parametrize("m", [1, 2, 3, 4, 8, 9, 16])
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
    bound = (x.double().abs() @ mod.dequantize_mxfp8_rows(values, scales).double().abs().T)
    err = (out.double() - ref).abs()
    assert torch.all(err <= 1e-5 * bound + 1e-6), float((err / (bound + 1e-9)).max())


@pytest.mark.parametrize("m", [1, 2, 4, 8, 16])
def test_argmax_keys_match_reference(m):
    mod = _mod()
    w = _weights(N_VOCAB_SHARD, seed=3)
    values, scales = mod.quantize_mxfp8_rows(w)
    x = _acts(m, seed=4)
    keys = torch.full((m, ), INT64_MIN, dtype=torch.int64, device="cuda")
    offset = 5 * N_VOCAB_SHARD
    mod.mxfp8_head_gemv(x, values, scales, keys.view(torch.int32).view(m, 2), argmax=True,
                        vocab_offset=offset)
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


@pytest.mark.parametrize("m", [1, 4, 16])
def test_keys_match_ll_exchange_packing(m):
    """The fused keys are bit-identical to comm's pack_argmax_keys of the same kernel's FP32 logits."""
    from tensorrt_llm._torch.distributed.ll_exchange import pack_argmax_keys

    mod = _mod()
    w = _weights(N_VOCAB_SHARD, seed=41)
    values, scales = mod.quantize_mxfp8_rows(w)
    x = _acts(m, seed=42)
    logits = torch.empty((m, N_VOCAB_SHARD), dtype=torch.float32, device="cuda")
    mod.mxfp8_head_gemv(x, values, scales, logits, argmax=False)
    keys = torch.full((m, ), INT64_MIN, dtype=torch.int64, device="cuda")
    offset = 3 * N_VOCAB_SHARD
    mod.mxfp8_head_gemv(x, values, scales, keys.view(torch.int32).view(m, 2), argmax=True,
                        vocab_offset=offset)
    vmax, imax = torch.max(logits, dim=-1)
    ref = pack_argmax_keys(vmax, imax, offset)
    torch.cuda.synchronize()
    assert torch.equal(keys, ref), (keys, ref)


def test_nvfp4_quantization_grid_and_error():
    """E2M1 rounding to nearest with ties to the even code; block scale = amax / 6 in E4M3 units of g."""
    mod = _mod()
    vals = torch.tensor([0.0, 0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, 5.9, -0.75, -6.0, 6.0, 0.1, 1.1, 2.9, 4.4],
                        device="cuda")
    # Row 1 sets the tensor amax to 10.5, so g = 10.5 / 2688 = 2^-8 and row 0's block scale (amax 6) is
    # 256 = 2^8 exactly: one E2M1 unit is exactly 1.0 and the ties below are exact ties.
    other = torch.zeros(16, device="cuda")
    other[0] = 10.5
    w = torch.stack([vals, other]).to(torch.bfloat16)
    packed, scales, g = mod.quantize_nvfp4_rows(w)
    assert g == 2.0**-8, g
    deq = mod.dequantize_nvfp4_rows(packed, scales, g)
    unit = float(scales.view(torch.float8_e4m3fn).float()[0, 0]) * g
    assert unit == 1.0, unit
    expect = torch.tensor([0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 4.0, 4.0, 6.0, -1.0, -6.0, 6.0, 0.0, 1.0, 3.0, 4.0],
                          device="cuda")
    assert torch.equal(deq[0], expect), deq[0]
    # Gaussian rows: NVFP4 relative error ~10 %.
    w = _weights(512, seed=21)
    packed, scales, g = mod.quantize_nvfp4_rows(w)
    deq = mod.dequantize_nvfp4_rows(packed, scales, g)
    rel = float((deq - w.float()).norm() / w.float().norm())
    assert 0.03 < rel < 0.15, rel


def _ref_logits_nvfp4(x, packed, scales, g):
    deq = _mod().dequantize_nvfp4_rows(packed, scales, g).double()
    return x.double() @ deq.T, deq


@pytest.mark.parametrize("m", [1, 3, 8, 9, 16])
def test_nvfp4_logits_match_dequantized_reference(m):
    mod = _mod()
    w = _weights(N_VOCAB_SHARD, seed=31)
    packed, scales, g = mod.quantize_nvfp4_rows(w)
    x = _acts(m, seed=32)
    out = torch.empty((m, N_VOCAB_SHARD), dtype=torch.float32, device="cuda")
    mod.nvfp4_head_gemv(x, packed, scales, g, out, argmax=False)
    torch.cuda.synchronize()
    ref, deq = _ref_logits_nvfp4(x, packed, scales, g)
    bound = x.double().abs() @ deq.abs().T
    err = (out.double() - ref).abs()
    assert torch.all(err <= 1e-5 * bound + 1e-6), float((err / (bound + 1e-9)).max())
    # Negative control: the MXFP8 kernel path on the same bytes is far off.
    assert not torch.allclose(out.double(), ref * 1.5, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("m", [1, 4, 16])
def test_nvfp4_argmax_keys_match_logits(m):
    """argmax=True keys are bit-identical to comm's packing of the same kernel's logits (NVFP4)."""
    from tensorrt_llm._torch.distributed.ll_exchange import pack_argmax_keys

    mod = _mod()
    w = _weights(N_VOCAB_SHARD, seed=33)
    packed, scales, g = mod.quantize_nvfp4_rows(w)
    x = _acts(m, seed=34)
    logits = torch.empty((m, N_VOCAB_SHARD), dtype=torch.float32, device="cuda")
    mod.nvfp4_head_gemv(x, packed, scales, g, logits, argmax=False)
    keys = torch.full((m, ), INT64_MIN, dtype=torch.int64, device="cuda")
    offset = 2 * N_VOCAB_SHARD
    mod.nvfp4_head_gemv(x, packed, scales, g, keys.view(torch.int32).view(m, 2), argmax=True,
                        vocab_offset=offset)
    vmax, imax = torch.max(logits, dim=-1)
    ref = pack_argmax_keys(vmax, imax, offset)
    torch.cuda.synchronize()
    assert torch.equal(keys, ref), (keys, ref)


def test_nvfp4_draft_head_object(monkeypatch):
    """Q35DraftHead with TRTLLM_Q35_DRAFT_HEAD_NVFP4=1 picks the global argmax of its dequantized shard."""
    import types

    mod = _mod()
    monkeypatch.setenv("TRTLLM_Q35_DRAFT_HEAD_NVFP4", "1")
    w = _weights(N_VOCAB_SHARD, seed=35)
    head = types.SimpleNamespace(weight=w, out_features=N_VOCAB_SHARD, num_embeddings=N_VOCAB_SHARD, tp_size=1)
    dh = mod.Q35DraftHead(head, None)
    assert dh.nvfp4 and dh.values.shape == (N_VOCAB_SHARD, K // 2)
    x = _acts(4, seed=36)
    tokens = dh.sample(x)
    torch.cuda.synchronize()
    ref, _ = _ref_logits_nvfp4(x, dh.values, dh.scales, dh.global_scale)
    got = ref.gather(1, tokens.long()[:, None]).squeeze(1)
    assert torch.all(ref.max(1).values - got <= 1e-5 * ref.max(1).values.abs() + 1e-5)


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
    keys = torch.full((1, ), INT64_MIN, dtype=torch.int64, device="cuda")
    mod.mxfp8_head_gemv(x, values, scales, keys.view(torch.int32).view(1, 2), argmax=True)
    torch.cuda.synchronize()
    assert int(_keys_to_ids(keys)[0]) == 9000
    # Shards: the same winner in shards 2 and 6 -> the global id of shard 2.
    shard_keys = []
    for r in range(8):
        k = torch.full((1, ), INT64_MIN, dtype=torch.int64, device="cuda")
        wr = _weights(n, seed=100 + r)
        if r in (2, 6):
            wr[123] = win
        v, s = mod.quantize_mxfp8_rows(wr)
        mod.mxfp8_head_gemv(x, v, s, k.view(torch.int32).view(1, 2), argmax=True,
                            vocab_offset=r * n)
        shard_keys.append(k)
    gathered = torch.cat(shard_keys)
    local = torch.full((1, ), 7, dtype=torch.int64, device="cuda")
    tokens = mod.decode_keys(gathered, 8, local)
    torch.cuda.synchronize()
    assert int(tokens[0]) == 2 * n + 123
    assert int(local[0]) == INT64_MIN  # re-armed


@pytest.mark.parametrize("m", [1, 4])
@pytest.mark.parametrize("ranks,shard", [(8, 7760), (16, 15520), (8, 31040)])
def test_sharded_argmax_equals_global_argmax(m, ranks, shard):
    """TP4 / TP8 / TP16 vocabulary shards (TP16: 248320 / 16 = 15520 rows per rank)."""
    mod = _mod()
    w = _weights(ranks * shard, seed=7)
    n = shard
    x = _acts(m, seed=8)
    keys = []
    refs = []
    for r in range(ranks):
        v, s = mod.quantize_mxfp8_rows(w[r * n:(r + 1) * n].contiguous())
        k = torch.full((m, ), INT64_MIN, dtype=torch.int64, device="cuda")
        mod.mxfp8_head_gemv(x, v, s, k.view(torch.int32).view(m, 2), argmax=True,
                            vocab_offset=r * n)
        keys.append(k)
        refs.append(_ref_logits(x, v, s))
    local = torch.empty((m, ), dtype=torch.int64, device="cuda")
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
    keys = torch.full((1, ), INT64_MIN, dtype=torch.int64, device="cuda")
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
            self.keys = torch.full((mod.MAX_ROWS, ), INT64_MIN, dtype=torch.int64,
                                   device="cuda")
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
    """A producer that releases its dependents immediately and writes the activation ~50 us later:
    the PDL-launched head must still read the new activation (no load above griddepcontrol.wait)."""
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
        t0 = tl.inline_asm_elementwise("mov.u64 $0, %globaltimer;", "=l,r", [n], dtype=tl.int64,
                                       is_pure=False, pack=1)
        t = t0
        while t - t0 < delay_ns:
            t = tl.inline_asm_elementwise("mov.u64 $0, %globaltimer;", "=l,r", [n],
                                          dtype=tl.int64, is_pure=False, pack=1)
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
    keys = torch.full((1, ), INT64_MIN, dtype=torch.int64, device="cuda")
    mod.mxfp8_head_gemv(xs, values, scales, keys.view(torch.int32).view(1, 2), argmax=True)
    def run(skip_wait):
        # Producer and head back to back in one CUDA graph (eager host launch gaps exceed the delay).
        mod.mxfp8_head_gemv(xs, values, scales, keys.view(torch.int32).view(1, 2), argmax=True,
                            _debug_skip_wait=skip_wait)
        _late_writer[(K // 1024, )](x_new, xs, K, delay_ns=200_000, BLOCK=1024)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            with torch.cuda.graph(graph, stream=stream):
                _late_writer[(K // 1024, )](x_new, xs, K, delay_ns=200_000, BLOCK=1024)
                mod.mxfp8_head_gemv(xs, values, scales, keys.view(torch.int32).view(1, 2),
                                    argmax=True, _debug_skip_wait=skip_wait)
        got = []
        for _ in range(5):
            xs.copy_(x_old)
            keys.fill_(INT64_MIN)
            torch.cuda.synchronize()
            graph.replay()
            torch.cuda.synchronize()
            got.append(int(_keys_to_ids(keys)[0]))
        return got

    # Positive control: without the wait the head runs ahead of the producer and sees the old row.
    assert ref_old in run(True), "delayed producer did not overlap the head: the test is not valid"
    assert run(False) == [ref_new] * 5


def _bench():
    """Per-call time of the BF16 head (F.linear + torch.max) vs the MXFP8 GEMV + argmax + decode,
    in CUDA graphs of 50 calls, at M = 1, 2, 4, 8."""
    mod = _mod()
    w = _weights(N_VOCAB_SHARD, seed=31)
    values, scales = mod.quantize_mxfp8_rows(w)
    reps = 50
    for m in (1, 2, 4, 8):
        x = _acts(m, seed=32)
        keys = torch.full((m, ), INT64_MIN, dtype=torch.int64, device="cuda")

        def bf16():
            logits = torch.nn.functional.linear(x, w)
            return torch.max(logits, dim=-1)

        def fp8_logits():
            out = torch.empty((m, N_VOCAB_SHARD), dtype=torch.bfloat16, device="cuda")
            return mod.mxfp8_head_gemv(x, values, scales, out, argmax=False)

        def fp8_argmax():
            mod.mxfp8_head_gemv(x, values, scales, keys.view(torch.int32).view(m, 2), argmax=True)
            return mod.decode_keys(keys, 1, keys)

        def fp8_argmax_only():
            return mod.mxfp8_head_gemv(x, values, scales, keys.view(torch.int32).view(m, 2),
                                       argmax=True)

        res = {}
        for name, fn in (("bf16_linear+max", bf16), ("fp8_gemv_logits", fp8_logits),
                         ("fp8_gemv_argmax", fp8_argmax_only), ("fp8_argmax+decode", fp8_argmax)):
            fn()
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            s = torch.cuda.Stream()
            with torch.cuda.stream(s):
                with torch.cuda.graph(g, stream=s):
                    for _ in range(reps):
                        fn()
            g.replay()
            torch.cuda.synchronize()
            # Flush L2 between measurements is not needed: 127+ MB weights exceed L2.
            ts = []
            for _ in range(10):
                e0 = torch.cuda.Event(enable_timing=True)
                e1 = torch.cuda.Event(enable_timing=True)
                e0.record()
                g.replay()
                e1.record()
                torch.cuda.synchronize()
                ts.append(e0.elapsed_time(e1) * 1000 / reps)
            ts.sort()
            res[name] = ts[len(ts) // 2]
        print(f"M={m}: " + ", ".join(f"{k} {v:.1f} us" for k, v in res.items()), flush=True)
    print(f"bytes: bf16 {w.numel() * 2 / 1e6:.1f} MB, fp8 {(values.numel() + scales.numel()) / 1e6:.1f} MB")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "bench":
        _bench()
    else:
        sys.exit(pytest.main([__file__, "-v", "-x"] + sys.argv[1:]))
