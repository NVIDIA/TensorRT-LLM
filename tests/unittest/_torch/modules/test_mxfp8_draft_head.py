# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""MXFP8 draft head: streaming GEMV, fused shard argmax and key decode."""

import os

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)),
    reason="needs SM100/SM103",
)

N_VOCAB_SHARD = 31040  # 248320 / TP8
K = 4096
INT64_MIN = -(1 << 63)


def _mod():
    from tensorrt_llm._torch.modules import mxfp8_draft_head as m

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
def test_logits_and_argmax_keys_match_reference(m):
    from tensorrt_llm._torch.distributed.ll_exchange import pack_argmax_keys

    mod = _mod()
    values, scales = mod.quantize_mxfp8_rows(_weights(N_VOCAB_SHARD, seed=m))
    x = _acts(m, seed=m + 1)
    logits = torch.empty((m, N_VOCAB_SHARD), dtype=torch.float32, device="cuda")
    mod.mxfp8_head_gemv(x, values, scales, logits, argmax=False)
    keys = torch.full((m,), INT64_MIN, dtype=torch.int64, device="cuda")
    offset = 5 * N_VOCAB_SHARD
    mod.mxfp8_head_gemv(
        x, values, scales, keys.view(torch.int32).view(m, 2), argmax=True, vocab_offset=offset
    )
    torch.cuda.synchronize()
    ref = _ref_logits(x, values, scales)
    # Exact products, FP32 sums of 4096 terms: bound by a few ulps of sum |x||w|.
    bound = x.double().abs() @ mod.dequantize_mxfp8_rows(values, scales).double().abs().T
    err = (logits.double() - ref).abs()
    assert torch.all(err <= 1e-5 * bound + 1e-6), float((err / (bound + 1e-9)).max())
    # The fused keys are bit-identical to comm's packing of the same kernel's FP32 logits.
    vmax, imax = torch.max(logits, dim=-1)
    assert torch.equal(keys, pack_argmax_keys(vmax, imax, offset))


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


@pytest.mark.parametrize("ranks,shard", [(16, 15520), (8, 31040)])
def test_sharded_argmax_equals_global_argmax(ranks, shard):
    mod = _mod()
    m = 4
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


def test_graph_replay_rearms_keys():
    mod = _mod()
    w = _weights(N_VOCAB_SHARD, seed=11)
    values, scales = mod.quantize_mxfp8_rows(w)

    class _Head(mod.MXFP8DraftHead):
        def __init__(self):  # noqa: D401 - test double without an LMHead
            self.values, self.scales = values, scales
            self.vocab_offset = 0
            self.k = K
            self.mapping = None
            self.keys = torch.full((mod.MAX_ROWS,), INT64_MIN, dtype=torch.int64, device="cuda")
            self.slice_blocks = mod.default_slice_blocks(K)
            self.use_pdl = True

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
