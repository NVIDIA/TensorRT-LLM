# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fused convolution raw Q/K/V, state ownership, and CUDA graph parity."""

from itertools import accumulate

import pytest
import torch
from kda_module_test_utils import clone_pool, make_batch

from tensorrt_llm._torch.attention.kernels.blackwell.kimi_k3_kda.fused_prefill_conv import (
    fused_prefill_conv,
)
from tensorrt_llm._torch.modules.kimi_kda._kda_kernels import fused_kda_post_conv
from tensorrt_llm._torch.modules.mamba.causal_conv1d import causal_conv1d_fn
from tensorrt_llm._torch.modules.mamba.fuse_elementwise_ops import extract_transpose_prefill_slice

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
