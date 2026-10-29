# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fused convolution raw Q/K/V, state ownership, and CUDA graph parity."""

import random
from itertools import accumulate
from types import SimpleNamespace

import cuda.bindings.driver as cuda
import pytest
import torch

from tensorrt_llm._torch.cute_dsl_kernels.blackwell.kimi_k3_kda.fused_prefill_conv import (
    _conv_layouts,
    _grid_runs,
    _specialization,
    compile_prefill_conv,
    fused_prefill_conv,
)
from tensorrt_llm._torch.modules.kimi_kda._kda_flashinfer_prefill import (
    kda_layout_qkv as fused_kda_post_conv,
)
from tensorrt_llm._torch.modules.mamba.causal_conv1d import causal_conv1d_fn
from tensorrt_llm._torch.modules.mamba.fuse_elementwise_ops import extract_transpose_prefill_slice
from tests.unittest._torch.modules.kimi_kda.kda_module_test_utils import clone_pool, make_batch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability() not in ((10, 0), (10, 3), (10, 7)),
    reason="BT16 requires SM100/SM103/SM107",
)


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
            for actual, ref in zip(qkv, expected):
                assert torch.isfinite(actual).all()
                torch.testing.assert_close(actual, ref, rtol=0.008, atol=0.0001)
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
        for actual, reference in zip(qkv, expected):
            torch.testing.assert_close(actual, reference, rtol=0.008, atol=0.0001)
        torch.testing.assert_close(state, expected_state, rtol=0, atol=0)
        previous_outputs = qkv
        previous_values = tuple(t.clone() for t in qkv)


def _conv_inputs(heads: int, lengths: list[int], seed: int, initial: str = "mixed"):
    """One launch: the batch (state pool and metadata), packed source, weights and fresh outputs."""
    torch.manual_seed(seed)
    batch = make_batch(heads, lengths, hidden_size=8, initial=initial, input_seed=seed)
    total, channels = sum(lengths), 3 * heads * 128
    src = torch.randn(total, 4 * heads * 128, device="cuda", dtype=torch.bfloat16) * 0.1
    weight = torch.randn(channels, 4, device="cuda", dtype=torch.bfloat16) * 0.1
    qkv = tuple(
        torch.empty(1, total, heads, 128, device="cuda", dtype=torch.bfloat16) for _ in range(3)
    )
    return batch, src, weight, qkv


def _reference(batch, src: torch.Tensor, weight: torch.Tensor, heads: int):
    """Reference q/k/v and final conv state of one launch (the batch's pool is left untouched)."""
    total, channels = src.shape[0], weight.shape[0]
    expected_state = clone_pool(batch.conv)
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
    return fused_kda_post_conv(packed, heads, 128, normalize_qk=False), expected_state


def _launch(batch, src, weight, qkv, lengths: list[int]) -> None:
    fused_prefill_conv(
        src,
        weight,
        batch.conv,
        batch.metadata.query_start_loc,
        batch.indices,
        batch.metadata.has_initial_states,
        max(lengths),
        *qkv,
    )


def _launch_per_shape(batch, src, weight, qkv, lengths: list[int]) -> None:
    """The same launch through a kernel compiled for exactly these shapes: every dimension is a constant."""
    operands = (
        src,
        weight,
        batch.conv,
        batch.metadata.query_start_loc,
        batch.indices,
        batch.metadata.has_initial_states,
        *qkv,
    )
    spec, grid_y = _specialization(src, weight, batch.conv, *qkv, len(lengths), max(lengths))
    # Plain ints in the layouts make every dimension a compile-time constant.
    layouts = tuple((tuple(t.shape), tuple(t.stride()), t.dtype) for t in operands)
    compile_prefill_conv(src.device, layouts, *spec)(
        *operands, grid_y, cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    )


def _bits(t: torch.Tensor) -> torch.Tensor:
    """The raw 16-bit patterns of a BF16 tensor: equality of bits, whatever the values (NaN included)."""
    return t.view(torch.int16)


def _shifted(t: torch.Tensor) -> torch.Tensor:
    """A contiguous copy of t that starts one element into its allocation: never 16-byte aligned."""
    backing = torch.empty(t.numel() + 1, dtype=t.dtype, device=t.device)
    shifted = backing[1:].view(t.shape)
    shifted.copy_(t)
    return shifted


# Token counts, sequence counts (also more sequences than tokens) and, through make_batch,
# state-pool sizes all differ between the cases; the first one is the warm-up.
BATCH_SHAPES = [
    [17, 31],
    [1],
    [3],
    [0, 0, 5],
    [64] * 13,
    [33, 33, 0],
    [1] * 40,
    [129, 7, 4000],
    [8192],
    [5, 4, 3, 2, 1],
    [2048] * 3,
    [1, 1],
    [0, 0, 0, 1],
    [0] * 5 + [2],
]


@pytest.mark.parametrize("heads", [1, 6])
@torch.inference_mode()
def test_fused_conv_new_batch_shapes_reuse_the_compilation(heads: int, cute_compiles) -> None:
    for position, lengths in enumerate(BATCH_SHAPES):
        batch, src, weight, qkv = _conv_inputs(heads, lengths, seed=900 + position)
        expected, expected_state = _reference(batch, src, weight, heads)
        if position == 1:
            # The first launch has compiled this layout class; nothing may compile again.
            misses, compiles = compile_prefill_conv.cache_info().misses, len(cute_compiles)
        _launch(batch, src, weight, qkv, lengths)
        for actual, reference in zip(qkv, expected):
            torch.testing.assert_close(actual, reference, rtol=0.008, atol=0.0001)
        torch.testing.assert_close(batch.conv, expected_state, rtol=0, atol=0)
    assert compile_prefill_conv.cache_info().misses == misses
    assert len(cute_compiles) == compiles


@pytest.mark.parametrize("heads", [1, 6])
@pytest.mark.parametrize("initial", ["fresh", "continuation", "mixed"])
@torch.inference_mode()
def test_fused_conv_symbolic_shapes_match_per_shape_compilation_bitwise(
    heads: int, initial: str
) -> None:
    """One symbolic compilation and a compilation of the exact shapes give the same bits."""
    for position, lengths in enumerate(([17, 31], [1], [0, 0, 5], [64] * 13, [4097, 3])):
        batch, src, weight, qkv = _conv_inputs(heads, lengths, 950 + position, initial)
        seed_state = clone_pool(batch.conv)
        _launch(batch, src, weight, qkv, lengths)
        symbolic_state = clone_pool(batch.conv)

        batch.conv.copy_(seed_state)
        static_qkv = tuple(torch.empty_like(t) for t in qkv)
        _launch_per_shape(batch, src, weight, static_qkv, lengths)
        for symbolic, static in zip(qkv, static_qkv):
            assert torch.equal(_bits(symbolic), _bits(static))
        assert torch.equal(_bits(symbolic_state), _bits(batch.conv))


@pytest.mark.parametrize("operand", ["src", "weight", "outputs", "state"])
@torch.inference_mode()
def test_fused_conv_unaligned_operands_take_elementwise_paths_and_stay_symbolic(
    operand: str, cute_compiles
) -> None:
    """An operand that is not 16-byte aligned selects the elementwise memory path of that operand,
    and that specialization is shape-symbolic like the vector one."""
    heads = 2

    def prepare(lengths: list[int], seed: int):
        batch, src, weight, qkv = _conv_inputs(heads, lengths, seed)
        seed_state = clone_pool(batch.conv)
        expected = _reference(batch, src, weight, heads)
        if operand == "src":
            src = _shifted(src)
        elif operand == "weight":
            weight = _shifted(weight)
        elif operand == "outputs":
            qkv = tuple(_shifted(t) for t in qkv)
        else:
            batch.conv = _shifted(batch.conv)
            batch.conv.copy_(seed_state)
        spec, _ = _specialization(src, weight, batch.conv, *qkv, len(lengths), max(lengths))
        flags = dict(zip(("src", "weight", "outputs", "state"), spec[5:9]))
        assert flags[operand] is False
        assert all(flag is True for name, flag in flags.items() if name != operand), flags
        return lengths, batch, src, weight, qkv, seed_state, expected

    cases = [prepare([37, 0, 5, 64], seed=980), prepare([3, 90, 1, 0, 8, 200], seed=981)]
    mark = None
    for lengths, batch, src, weight, qkv, _, (expected, expected_state) in cases:
        _launch(batch, src, weight, qkv, lengths)
        if mark is None:
            mark = (compile_prefill_conv.cache_info().misses, len(cute_compiles))
        for actual, reference in zip(qkv, expected):
            torch.testing.assert_close(actual, reference, rtol=0.008, atol=0.0001)
        torch.testing.assert_close(batch.conv, expected_state, rtol=0, atol=0)
    # The second batch shape reused the compilation of the first.
    assert (compile_prefill_conv.cache_info().misses, len(cute_compiles)) == mark

    for lengths, batch, src, weight, qkv, seed_state, _ in cases:
        symbolic_state = clone_pool(batch.conv)
        batch.conv.copy_(seed_state)
        static_qkv = tuple(
            _shifted(torch.empty_like(t)) if operand == "outputs" else torch.empty_like(t)
            for t in qkv
        )
        _launch_per_shape(batch, src, weight, static_qkv, lengths)
        for symbolic, static in zip(qkv, static_qkv):
            assert torch.equal(_bits(symbolic), _bits(static))
        assert torch.equal(_bits(symbolic_state), _bits(batch.conv))


def _single_token_sequences(n: int, heads: int, seed: int):
    """n sequences of one token each, slots in reverse order, a state pool with two spare rows."""
    channels = 3 * heads * 128
    generator = torch.Generator(device="cuda").manual_seed(seed)
    src = torch.randn(
        n, 4 * channels // 3, device="cuda", dtype=torch.bfloat16, generator=generator
    )
    weight = torch.randn(channels, 4, device="cuda", dtype=torch.bfloat16, generator=generator)
    conv = torch.randn(n + 2, channels, 3, device="cuda", dtype=torch.bfloat16, generator=generator)
    qkv = tuple(
        torch.empty(1, n, heads, 128, device="cuda", dtype=torch.bfloat16) for _ in range(3)
    )
    batch = SimpleNamespace(
        conv=conv,
        indices=torch.arange(n, 0, -1, device="cuda", dtype=torch.int32),
        metadata=SimpleNamespace(
            query_start_loc=torch.arange(n + 1, device="cuda", dtype=torch.int32),
            has_initial_states=torch.arange(n, device="cuda") % 2 == 0,
        ),
    )
    return batch, src, weight, qkv


@torch.inference_mode()
def test_fused_conv_more_sequences_than_grid_z_stays_symbolic(cute_compiles) -> None:
    """More than 65535 sequences select the specialization that grid-strides over sequences."""
    heads = 1
    runs = []
    for n, seed in ((65537, 990), (70001, 991)):
        batch, src, weight, qkv = _single_token_sequences(n, heads, seed)
        spec, _ = _specialization(src, weight, batch.conv, *qkv, n, 1)
        assert spec[-1] is True  # seq_loop
        expected, expected_state = _reference(batch, src, weight, heads)
        seed_state = clone_pool(batch.conv)
        _launch(batch, src, weight, qkv, [1] * n)
        runs.append((n, batch, src, weight, qkv, seed_state))
        for actual, reference in zip(qkv, expected):
            torch.testing.assert_close(actual, reference, rtol=0.008, atol=0.0001)
        torch.testing.assert_close(batch.conv, expected_state, rtol=0, atol=0)
        if len(runs) == 1:
            mark = (compile_prefill_conv.cache_info().misses, len(cute_compiles))
    # Another sequence count in this specialization did not compile again.
    assert (compile_prefill_conv.cache_info().misses, len(cute_compiles)) == mark

    n, batch, src, weight, qkv, seed_state = runs[0]
    symbolic_state = clone_pool(batch.conv)
    batch.conv.copy_(seed_state)
    static_qkv = tuple(torch.empty_like(t) for t in qkv)
    _launch_per_shape(batch, src, weight, static_qkv, [1] * n)
    for symbolic, static in zip(qkv, static_qkv):
        assert torch.equal(_bits(symbolic), _bits(static))
    assert torch.equal(_bits(symbolic_state), _bits(batch.conv))


@torch.inference_mode()
def test_fused_conv_layout_key_ignores_batch_shape() -> None:
    def key(tokens: int, seqs: int, pool: int, heads: int = 6, src_stride: int = 4 * 6 * 128):
        meta = {"device": "meta", "dtype": torch.bfloat16}
        channels = 3 * heads * 128
        src = torch.empty_strided((tokens, channels), (src_stride, 1), **meta)
        state = torch.empty(pool, channels, 3, **meta)
        qkv = [torch.empty(1, tokens, heads, 128, **meta) for _ in range(3)]
        starts, slots = (torch.empty(n, device="meta", dtype=torch.int32) for n in (seqs + 1, seqs))
        has_init = torch.empty(seqs, device="meta", dtype=torch.bool)
        weight = torch.empty(channels, 4, **meta)
        return _conv_layouts(src, weight, state, starts, slots, has_init, *qkv)

    base = key(4096, 8, 10)
    assert {key(t, n, p) for t in (1, 2, 33, 32768) for n in (1, 2, 64) for p in (2, 70)} == {base}
    # What the kernel bakes into its address arithmetic must still select a compilation.
    assert key(4096, 8, 10, heads=3, src_stride=4 * 3 * 128) != base


def test_grid_y_balances_the_longest_chain_against_idle_ctas() -> None:
    run, channels = 32, 3 * 96 * 128
    # One sequence, or equal ones: a CTA per run, nothing idles.
    assert _grid_runs(24576, 1, 24576, channels, run) == 768
    assert _grid_runs(24576, 8, 3072, channels, run) == 96
    assert _grid_runs(24576, 64, 384, channels, run) == 12
    assert _grid_runs(32000, 2, 24000, channels, run) == 250  # the other sequence has 250 runs
    # Skewed batches: far fewer CTAs than the longest sequence has runs (359, 938, 127).
    assert _grid_runs(32768, 32, 11469, channels, run) == 25
    assert _grid_runs(30063, 64, 30000, channels, run) == 28
    assert _grid_runs(4096, 64, 4033, channels, run) == 11
    # Fewer heads mean fewer channel blocks per run, so more CTAs per sequence are needed.
    assert _grid_runs(32768, 32, 11469, 3 * 12 * 128, run) == 70
    assert _grid_runs(4096, 64, 4033, 3 * 12 * 128, run) == 29
    # Degenerate shapes: a bound below the mean wins, the launch limit caps.
    assert _grid_runs(64, 64, 1, channels, run) == 1
    assert _grid_runs(1000, 4, 10, channels, run) == 1
    assert _grid_runs(100, 1, 100, channels, run) == 4
    assert _grid_runs(10_000_000, 1, 10_000_000, channels, run) == 65535


def test_grid_y_never_exceeds_the_longest_sequence() -> None:
    """Between one CTA and one per run of the longest sequence (what sizing by the longest gives)."""
    rng = random.Random(20261020)
    for _ in range(2000):
        tokens, batch = rng.randint(1, 40000), rng.randint(1, 80)
        bound, channels = rng.randint(1, 40000), 384 * rng.choice([1, 3, 12, 24, 96])
        longest = -(-min(bound, tokens) // 32)
        assert 1 <= _grid_runs(tokens, batch, bound, channels, 32) <= min(longest, 65535)


@torch.inference_mode()
def test_fused_conv_addresses_pool_rows_past_int32() -> None:
    """Pool rows whose element offset slot * stride exceeds int32.

    KV cache manager V2 coalesces every layer's state of a slot, so a large pool
    passes 2**31 elements as a whole; slot 16 here starts exactly there. The
    launch must give the bits of the same launch on a compact pool and leave
    every element outside the batch's rows (NaN) alone.
    """
    lengths = [40, 24]
    batch, src, weight, qkv = _conv_inputs(heads=1, lengths=lengths, seed=971)
    seed = clone_pool(batch.conv)
    expected_qkv = tuple(torch.empty_like(t) for t in qkv)
    _launch(batch, src, weight, expected_qkv, lengths)
    expected_rows = batch.conv[batch.indices.long()].clone()
    row_stride = 2**27
    backing = torch.full(
        (16 * row_stride + 3 * 128 * 3,), float("nan"), device="cuda", dtype=torch.bfloat16
    )
    pool = backing.as_strided((17, 3 * 128, 3), (row_stride, 3, 1))
    slots = torch.tensor([16, 3], device="cuda", dtype=torch.int32)
    pool[slots.long()] = seed[batch.indices.long()]
    fused_prefill_conv(
        src,
        weight,
        pool,
        batch.metadata.query_start_loc,
        slots,
        batch.metadata.has_initial_states,
        max(lengths),
        *qkv,
    )
    assert all(torch.equal(_bits(a), _bits(b)) for a, b in zip(qkv, expected_qkv))
    assert torch.equal(_bits(pool[slots.long()]), _bits(expected_rows))
    pool[slots.long()] = float("nan")
    assert bool(torch.isnan(backing).all()), "the convolution wrote outside the batch's rows"


@pytest.mark.parametrize("lengths", [[40, 24], [3000, 100, 5], [1024], [1] * 33 + [700]])
@torch.inference_mode()
def test_fused_conv_grid_y_only_sets_parallelism(lengths: list[int]) -> None:
    """Any grid.y >= 1 covers every token with the same bits: CTAs loop over their sequence's runs."""
    batch, src, weight, qkv = _conv_inputs(heads=3, lengths=lengths, seed=970)
    seed_state = clone_pool(batch.conv)
    operands = (
        src,
        weight,
        batch.conv,
        batch.metadata.query_start_loc,
        batch.indices,
        batch.metadata.has_initial_states,
        *qkv,
    )
    spec, grid_y = _specialization(src, weight, batch.conv, *qkv, len(lengths), max(lengths))
    kernel = compile_prefill_conv(src.device, _conv_layouts(*operands), *spec)
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    runs = -(-max(lengths) // 32)
    reference = None
    for grid in sorted({1, 2, 3, max(1, runs // 2), grid_y, runs, runs + 5}):
        batch.conv.copy_(seed_state)
        for out in qkv:
            out.fill_(float("nan"))
        kernel(*operands, grid, stream)
        result = [_bits(t).clone() for t in (*qkv, batch.conv)]
        reference = reference or result
        assert all(torch.equal(a, b) for a, b in zip(result, reference)), f"grid.y={grid}"
