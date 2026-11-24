# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fused convolution raw Q/K/V, state ownership, and CUDA graph parity
(``kernels/ssm/rubin_fused_prefill_conv.py``).

The kernel accumulates in fp32 and evaluates SiLU exactly (a few fp32 ulps), then rounds once to
BF16, so its outputs are checked in BF16 ulps: against the engine's convolution (also fp32), and
against an fp64 SiLU of the kernel's own fp32 accumulation over pre-activations from |x| ~ 0 to
beyond the fp32 underflow of exp(-|x|).
"""

from itertools import accumulate
from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kernels.ssm.rubin_fused_prefill_conv import (
    fused_prefill_conv,
)
from tensorrt_llm._torch.modules.mamba.causal_conv1d import causal_conv1d_fn
from tensorrt_llm._torch.modules.mamba.fuse_elementwise_ops import extract_transpose_prefill_slice

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7),
    reason="the family's Rubin kernels: needs an SM107 GPU",
)

# Two fp32 evaluations of the same SiLU round to the same BF16 value except near a rounding
# boundary: at most one BF16 ulp apart, and rarely.
MIN_EQUAL_FRACTION = 0.999


def bf16_order(t: torch.Tensor) -> torch.Tensor:
    """BF16 bit patterns as integers in the order of the values (adjacent BF16 values differ by 1)."""
    bits = t.contiguous().view(torch.int16).to(torch.int32)
    return torch.where(bits < 0, -32768 - bits, bits)


def assert_bf16_within_one_ulp(
    name: str, actual: torch.Tensor, expected: torch.Tensor, atol: float = 0.0
) -> None:
    """At most one BF16 ulp apart (or within `atol`, for outputs that cancel in the accumulation), and
    at least MIN_EQUAL_FRACTION of the elements bit for bit equal."""
    assert actual.dtype == expected.dtype == torch.bfloat16 and actual.shape == expected.shape
    assert torch.isfinite(actual).all(), f"{name}: non-finite outputs"
    ulps = (bf16_order(actual) - bf16_order(expected)).abs()
    near = (actual.float() - expected.float()).abs() <= atol
    bad = (ulps > 1) & ~near
    assert not bad.any(), (
        f"{name}: {int(bad.sum())} elements more than one BF16 ulp apart, max {int(ulps.max())} ulps"
    )
    equal = (ulps == 0).double().mean().item()
    assert equal >= MIN_EQUAL_FRACTION, f"{name}: only {equal:.6f} of the elements are equal"


def fp32_conv_fp64_silu(
    src: torch.Tensor,
    weight: torch.Tensor,
    history: torch.Tensor,
    starts: list[int],
    slots: list[int],
    init: list[bool],
) -> torch.Tensor:
    """BF16 ``silu(x)`` of every token, ``[T, C]``: x is the kernel's fp32 accumulation (a multiply,
    then fused multiply-adds, oldest tap first; emulated exactly in fp64), SiLU is evaluated in
    fp64 and rounded once (through fp32, which the kernel's own result also passes)."""
    channels, width = weight.shape
    w = weight.double()
    out = torch.empty(src.shape[0], channels, dtype=torch.float64, device=src.device)
    for lo, hi, slot, has in zip(starts, starts[1:], slots, init):
        hist = (
            history[slot].double().t()
            if has
            else torch.zeros(width - 1, channels, dtype=torch.float64, device=src.device)
        )
        window = torch.cat((hist, src[lo:hi, :channels].double()))
        acc = (w[:, 0] * window[: hi - lo]).float().double()
        for tap in range(1, width):
            acc = (w[:, tap] * window[tap : tap + hi - lo] + acc).float().double()
        out[lo:hi] = acc * torch.sigmoid(acc)
    return out.float().to(torch.bfloat16)


def fused_kda_post_conv(packed, num_heads, head_dim, normalize_qk=False):
    """The reference layout: the ``[3 * H * D, T]`` conv output as raw ``[1, T, H, D]`` Q, K, V
    (the unfused fallback's conversion without its Q/K normalization)."""
    assert not normalize_qk
    return tuple(
        t.t().reshape(1, packed.shape[1], num_heads, head_dim).contiguous()
        for t in packed.chunk(3, dim=0)
    )


def make_batch(
    heads: int,
    lengths: list[int],
    hidden_size: int = 7168,
    *,
    initial: str = "fresh",
    input_seed: int = 23,
    padded: bool = False,
) -> SimpleNamespace:
    """Packed-batch inputs prepared outside execution: a copy of the KDA layer tests'
    ``make_batch``, so that this kernel test needs no model-test helper."""
    n, total = len(lengths), sum(lengths)
    generator = torch.Generator(device="cuda").manual_seed(input_seed)
    offsets = [0]
    for length in lengths:
        offsets.append(offsets[-1] + length)
    cu = torch.tensor(offsets, dtype=torch.int64, device="cuda")
    indices = torch.arange(n, 0, -1, dtype=torch.int32, device="cuda")
    has_init = [initial == "continuation" or (initial == "mixed" and i % 2 == 0) for i in range(n)]
    metadata = SimpleNamespace(
        has_initial_states=torch.tensor(has_init, dtype=torch.bool, device="cuda"),
        use_initial_states=any(has_init),
        state_indices=indices,
        query_start_loc=cu.to(torch.int32),
        query_start_loc_long=cu,
        kda_max_seqlen=max(lengths, default=0),
        _arange_buffer=torch.arange(n + 1, dtype=torch.int32, device="cuda"),
    )
    hidden = (
        torch.randn(total, hidden_size, device="cuda", dtype=torch.bfloat16, generator=generator)
        * 0.1
    )
    conv = (
        torch.randn(
            n + 2, 3 * heads * 128, 3, device="cuda", dtype=torch.bfloat16, generator=generator
        )
        * 0.05
    )
    state = torch.empty_strided(
        (n + 2, heads, 128, 128),
        (heads * 16384 + (8 if padded else 0), 16384, 128, 1),
        device="cuda",
        dtype=torch.bfloat16,
    )
    state.copy_(torch.randn(state.shape, device="cuda", generator=generator) * 0.05)
    return SimpleNamespace(
        hidden=hidden,
        cu=cu,
        indices=indices,
        metadata=metadata,
        conv=conv,
        state=state,
        n=n,
        lengths=lengths,
    )


def clone_pool(pool: torch.Tensor) -> torch.Tensor:
    """Copy pool values while preserving padding between rows."""
    return torch.empty_strided(
        pool.shape, pool.stride(), device=pool.device, dtype=pool.dtype
    ).copy_(pool)


@pytest.mark.parametrize("heads", [1, 3, 6, 12, 96])
@pytest.mark.parametrize("initial", ["fresh", "continuation", "mixed"])
@torch.inference_mode()
def test_fused_conv_raw_outputs_and_dynamic_graph(heads: int, initial: str) -> None:
    # Include empty/short sequences and tile-boundary crossings. Deliberately
    # offset the input and pad pool rows to exercise the actual stride contract.
    lengths = [0, 1, 2, 3, 15, 16, 17, 31, 32, 33, 63, 64, 65]
    batch = make_batch(heads, lengths, hidden_size=8, initial=initial)
    total, channels = sum(lengths), 3 * heads * 128
    backing = (
        torch.randn(total * (4 * heads * 128 + 8) + 1, device="cuda", dtype=torch.bfloat16) * 0.1
    )
    src = torch.as_strided(backing, (total, 4 * heads * 128), (4 * heads * 128 + 8, 1), 1)
    weight = torch.randn(channels, 4, device="cuda", dtype=torch.bfloat16) * 0.1
    state = torch.empty_strided(
        batch.conv.shape, (channels * 3 + 8, 3, 1), device="cuda", dtype=torch.bfloat16
    ).copy_(batch.conv)
    seed = clone_pool(state)
    qkv = tuple(
        torch.empty(1, total, heads, 128, device="cuda", dtype=torch.bfloat16) for _ in range(3)
    )
    immutable = [t.clone() for t in (src, weight)]

    def run() -> None:
        fused_prefill_conv(
            src,
            weight,
            state,
            batch.metadata.query_start_loc,
            batch.indices,
            batch.metadata.has_initial_states,
            total,
            *qkv,
        )

    run()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    for current in (lengths, list(reversed(lengths)), [total] + [0] * (len(lengths) - 1)):
        offsets = torch.tensor([0] + list(accumulate(current)), device="cuda", dtype=torch.int64)
        batch.cu.copy_(offsets)
        batch.metadata.query_start_loc.copy_(offsets)
        expected_state = clone_pool(seed)
        expected_packed = extract_transpose_prefill_slice(src, total, 0, channels)
        causal_conv1d_fn(
            expected_packed,
            weight,
            query_start_loc=batch.metadata.query_start_loc,
            cache_indices=batch.indices,
            has_initial_state=batch.metadata.has_initial_states,
            conv_states=expected_state,
            activation="silu",
        )
        expected = fused_kda_post_conv(expected_packed, heads, 128, normalize_qk=False)
        for launch in (run, graph.replay):
            state.copy_(seed)
            launch()
            for name, actual, ref in zip("qkv", qkv, expected):
                assert_bf16_within_one_ulp(name, actual, ref, atol=1e-6)
            torch.testing.assert_close(state, expected_state, rtol=0, atol=0)
            for tensor, original in zip((src, weight), immutable):
                torch.testing.assert_close(tensor, original, rtol=0, atol=0)


@torch.inference_mode()
def test_fused_conv_rejects_output_aliasing() -> None:
    heads, total = 6, 16
    batch = make_batch(heads, [total], hidden_size=8)
    src = torch.randn(total, 3 * heads * 128, dtype=torch.bfloat16, device="cuda")
    weight = torch.randn(3 * heads * 128, 4, dtype=torch.bfloat16, device="cuda")
    # A contiguous output in the source allocation must be rejected before launch.
    q = src.view(-1)[: total * heads * 128].view(1, total, heads, 128)
    with pytest.raises(ValueError, match="disjoint storage"):
        fused_prefill_conv(
            src,
            weight,
            batch.conv,
            batch.metadata.query_start_loc,
            batch.indices,
            batch.metadata.has_initial_states,
            total,
            q,
            torch.empty_like(q),
            torch.empty_like(q),
        )


@pytest.mark.parametrize("source_offset,padded_state", [(0, False), (1, True)])
@torch.inference_mode()
def test_fused_conv_warmed_layout_accepts_new_buffers_in_graph(
    source_offset: int, padded_state: bool
) -> None:
    heads, lengths = 1, [17, 31]
    total, channels = sum(lengths), 384
    previous_outputs = None
    previous_values = None
    for seed in (701, 702):
        torch.manual_seed(seed)
        batch = make_batch(heads, lengths, hidden_size=8, initial="mixed", input_seed=seed)
        storage = torch.randn(total * channels + source_offset, device="cuda", dtype=torch.bfloat16)
        src = storage[source_offset:].view(total, channels)
        weight = torch.randn(channels, 4, device="cuda", dtype=torch.bfloat16) * 0.1
        state = torch.empty_strided(
            batch.conv.shape,
            (channels * 3 + (8 if padded_state else 0), 3, 1),
            device="cuda",
            dtype=torch.bfloat16,
        ).copy_(batch.conv)
        qkv = tuple(
            torch.empty(1, total, heads, 128, device="cuda", dtype=torch.bfloat16) for _ in range(3)
        )
        expected_state = clone_pool(state)
        packed = extract_transpose_prefill_slice(src, total, 0, channels)
        causal_conv1d_fn(
            packed,
            weight,
            query_start_loc=batch.metadata.query_start_loc,
            cache_indices=batch.indices,
            has_initial_state=batch.metadata.has_initial_states,
            conv_states=expected_state,
            activation="silu",
        )
        expected = fused_kda_post_conv(packed, heads, 128, normalize_qk=False)

        def run() -> None:
            fused_prefill_conv(
                src,
                weight,
                state,
                batch.metadata.query_start_loc,
                batch.indices,
                batch.metadata.has_initial_states,
                max(lengths),
                *qkv,
            )

        if previous_outputs is None:
            run()
        else:
            # Only the first allocation was warmed. New pointers with the same
            # layout must reuse that specialization without compiling in capture.
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                run()
            state.copy_(batch.conv)
            graph.replay()
            for old, snapshot in zip(previous_outputs, previous_values):
                torch.testing.assert_close(old, snapshot, rtol=0, atol=0)
        for name, actual, reference in zip("qkv", qkv, expected):
            assert_bf16_within_one_ulp(name, actual, reference, atol=1e-6)
        torch.testing.assert_close(state, expected_state, rtol=0, atol=0)
        previous_outputs = qkv
        previous_values = tuple(t.clone() for t in qkv)


@pytest.mark.parametrize("heads", [1, 6])
@pytest.mark.parametrize("source_offset", [0, 1])
@torch.inference_mode()
def test_fused_conv_exact_silu_over_wide_range(heads: int, source_offset: int) -> None:
    # Per-sequence input scales put the pre-activations x from |x| << 1 to beyond -100, where
    # exp(-|x|) is subnormal or zero in fp32: every value must stay within one BF16 ulp of the fp64
    # SiLU of the kernel's own fp32 accumulation, finite, with the sign of x.
    scales = [0.05, 0.6, 2.0, 6.0, 40.0]
    lengths = [37, 64, 33, 96, 45]
    init = [False, True, True, False, True]
    total, channels = sum(lengths), 3 * heads * 128
    gen = torch.Generator(device="cuda").manual_seed(4242 + heads + source_offset)
    width = channels + 128 * heads  # extra columns, as in the [q | k | v | g] projection
    storage = torch.zeros(total * width + source_offset, device="cuda", dtype=torch.bfloat16)
    src = storage[source_offset:].view(total, width)
    starts = [0]
    for length, scale in zip(lengths, scales):
        rows = slice(starts[-1], starts[-1] + length)
        src[rows].copy_(torch.randn(length, width, device="cuda", generator=gen) * scale)
        starts.append(starts[-1] + length)
    weight = (torch.randn(channels, 4, device="cuda", generator=gen) * 0.5).to(torch.bfloat16)
    slots = [3, 0, 5, 1, 4]
    state = torch.empty_strided(
        (6, channels, 3), (channels * 3 + 8, 3, 1), device="cuda", dtype=torch.bfloat16
    ).copy_(torch.randn(6, channels, 3, device="cuda", generator=gen) * 4.0)
    seed = clone_pool(state)
    qkv = tuple(
        torch.empty(1, total, heads, 128, device="cuda", dtype=torch.bfloat16) for _ in range(3)
    )
    fused_prefill_conv(
        src,
        weight,
        state,
        torch.tensor(starts, dtype=torch.int32, device="cuda"),
        torch.tensor(slots, dtype=torch.int32, device="cuda"),
        torch.tensor(init, dtype=torch.bool, device="cuda"),
        max(lengths),
        *qkv,
    )
    expected = fp32_conv_fp64_silu(src, weight, seed, starts, slots, init)
    expected_state = clone_pool(seed)
    for lo, hi, slot, has in zip(starts, starts[1:], slots, init):
        hist = seed[slot].t() if has else torch.zeros_like(seed[slot].t())
        expected_state[slot] = torch.cat((hist, src[lo:hi, :channels]))[-3:].t()
    for i, (name, actual) in enumerate(zip("qkv", qkv)):
        ref = expected[:, i * heads * 128 : (i + 1) * heads * 128].reshape(actual.shape)
        assert_bf16_within_one_ulp(name, actual, ref)
        assert torch.equal(actual.signbit(), ref.signbit()), f"{name}: sign differs"
    torch.testing.assert_close(state, expected_state, rtol=0, atol=0)
