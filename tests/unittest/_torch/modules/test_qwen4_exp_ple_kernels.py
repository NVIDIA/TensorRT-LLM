# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bitwise and CUDA-graph tests for fused Qwen4-Exp PLE kernels."""

import pytest
import torch
import torch.nn.functional as F

from tensorrt_llm._torch.modules.qwen4_exp.ple_kernels import (
    can_use_ple_decode_short_conv,
    can_use_ple_gate_value,
    can_use_ple_ngram_hash,
    can_use_ple_packed_short_conv,
    can_use_ple_short_conv_state,
    ple_decode_short_conv,
    ple_gate_value,
    ple_ngram_hash,
    ple_packed_short_conv,
    ple_short_conv_state,
)

EOS_TOKEN_ID = 248044
MULTIPLIERS = [23703573157769, 20109073645365, 8052911324071]


def _inputs(num_tokens: int):
    generator = torch.Generator(device="cuda").manual_seed(20260827 + num_tokens)
    contexts = torch.randint(
        0,
        248320,
        (num_tokens, 3),
        device="cuda",
        generator=generator,
    )
    contexts[0, 0] = EOS_TOKEN_ID
    if num_tokens > 1:
        contexts[1, 1] = EOS_TOKEN_ID
    multipliers = torch.tensor(MULTIPLIERS, dtype=torch.long, device="cuda")
    vocab_sizes = 20000003 + 2 * torch.arange(16, device="cuda")
    offsets = torch.cat([vocab_sizes.new_zeros(1), vocab_sizes.cumsum(0)[:-1]])
    gate = torch.randn(
        num_tokens,
        4,
        1,
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    )
    value = torch.randn(
        num_tokens,
        2560,
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    )
    state = torch.randn(
        num_tokens + 1,
        32,
        9,
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    )
    state_indices = torch.arange(1, num_tokens + 1, device="cuda")
    conv_value = torch.randn(
        num_tokens,
        32,
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    )
    return (
        contexts,
        multipliers,
        vocab_sizes,
        offsets,
        gate,
        value,
        state,
        state_indices,
        conv_value,
    )


def _hash_reference(
    contexts: torch.Tensor,
    multipliers: torch.Tensor,
    vocab_sizes: torch.Tensor,
    offsets: torch.Tensor,
) -> torch.Tensor:
    token_0, token_1, token_2 = contexts.unbind(1)
    mixed_2 = token_2 * multipliers[0] ^ token_1 * multipliers[1]
    token_0 = torch.where(
        (token_0 == EOS_TOKEN_ID) | (token_1 == EOS_TOKEN_ID),
        EOS_TOKEN_ID,
        token_0,
    )
    mixed_3 = mixed_2 ^ token_0 * multipliers[2]
    return torch.cat(
        [
            mixed_2[:, None].remainder(vocab_sizes[:8]) + offsets[:8],
            mixed_3[:, None].remainder(vocab_sizes[8:]) + offsets[8:],
        ],
        dim=1,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("num_tokens", [1, 64, 128, 256])
def test_ple_decode_kernels_are_bitwise_exact(num_tokens: int) -> None:
    contexts, multipliers, vocab_sizes, offsets, gate, value, state, indices, conv_value = _inputs(
        num_tokens
    )
    expected_hash = _hash_reference(contexts, multipliers, vocab_sizes, offsets)
    expected_gate = torch.sigmoid(
        gate.abs().clamp_min(1e-6).sqrt() * gate.sign()
    ) * value.unsqueeze(1)
    original_state = state.clone()
    expected_conv_input = torch.cat(
        [original_state.index_select(0, indices), conv_value.unsqueeze(-1)],
        dim=-1,
    )

    assert can_use_ple_ngram_hash(contexts, multipliers, vocab_sizes, offsets)
    assert can_use_ple_gate_value(gate, value)
    assert can_use_ple_short_conv_state(state, indices, conv_value)
    assert torch.equal(
        ple_ngram_hash(
            contexts,
            multipliers,
            vocab_sizes,
            offsets,
            EOS_TOKEN_ID,
        ),
        expected_hash,
    )
    assert torch.equal(ple_gate_value(gate, value), expected_gate)
    assert torch.equal(
        ple_short_conv_state(state, indices, conv_value),
        expected_conv_input,
    )
    assert torch.equal(state[indices, :, :-1], original_state[indices, :, 1:])
    assert torch.equal(state[indices, :, -1], conv_value)
    assert torch.equal(state[0], original_state[0])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("num_tokens", [1, 64, 128, 256])
def test_ple_decode_short_conv_is_bitwise_exact(num_tokens: int) -> None:
    generator = torch.Generator(device="cuda").manual_seed(20260828 + num_tokens)
    channels = 32
    indices = torch.arange(1, num_tokens + 1, device="cuda")
    state = torch.randn(
        num_tokens + 1,
        channels,
        9,
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    )
    value = torch.randn(
        num_tokens,
        channels,
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    )
    weight = torch.randn(
        channels,
        1,
        4,
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    )
    original_state = state.clone()
    expected_input = torch.cat(
        [original_state.index_select(0, indices), value.unsqueeze(-1)], dim=-1
    )
    expected = F.silu(F.conv1d(expected_input, weight, dilation=3, groups=channels).squeeze(-1))

    assert can_use_ple_decode_short_conv(state, indices, value, weight)
    actual = ple_decode_short_conv(state, indices, value, weight)
    assert torch.equal(actual, expected)
    assert torch.equal(state[indices, :, :-1], original_state[indices, :, 1:])
    assert torch.equal(state[indices, :, -1], value)
    assert torch.equal(state[0], original_state[0])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_ple_decode_kernels_replay_in_cuda_graph() -> None:
    contexts, multipliers, vocab_sizes, offsets, gate, value, state, indices, conv_value = _inputs(
        64
    )
    # Compile each Triton specialization before capture.
    ple_ngram_hash(contexts, multipliers, vocab_sizes, offsets, EOS_TOKEN_ID)
    ple_gate_value(gate, value)
    ple_short_conv_state(state, indices, conv_value)
    conv_weight = torch.randn(32, 1, 4, dtype=torch.bfloat16, device="cuda")
    ple_decode_short_conv(state, indices, conv_value, conv_weight)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        hash_output = ple_ngram_hash(
            contexts,
            multipliers,
            vocab_sizes,
            offsets,
            EOS_TOKEN_ID,
        )
        gate_output = ple_gate_value(gate, value)
        conv_input = ple_short_conv_state(state, indices, conv_value)
        conv_output = ple_decode_short_conv(state, indices, conv_value, conv_weight)
    graph.replay()
    torch.cuda.synchronize()

    assert torch.equal(
        hash_output,
        _hash_reference(contexts, multipliers, vocab_sizes, offsets),
    )
    assert torch.equal(
        gate_output,
        torch.sigmoid(gate.abs().clamp_min(1e-6).sqrt() * gate.sign()) * value.unsqueeze(1),
    )
    assert conv_input.shape == (64, 32, 10)
    assert conv_output.shape == (64, 32)


def _packed_layout(lengths: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
    length_tensor = torch.tensor(lengths, device="cuda")
    req_indices = torch.repeat_interleave(torch.arange(len(lengths), device="cuda"), length_tensor)
    starts = torch.cumsum(length_tensor, 0) - length_tensor
    token_offsets = torch.arange(sum(lengths), device="cuda") - starts[req_indices]
    return req_indices, token_offsets


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("channels", [32, 10240])
@pytest.mark.parametrize(
    "lengths",
    [[1], [5, 1, 9, 2], [37, 10, 3, 64], [8, 0, 11]],
    ids=["single", "shorter-than-state", "long", "empty-request"],
)
def test_ple_packed_short_conv_matches_per_request_conv(channels: int, lengths: list[int]) -> None:
    generator = torch.Generator(device="cuda").manual_seed(20261106 + sum(lengths))
    dilation, state_len = 3, 9
    # Distinct, non-monotonic slots; slot 0 and the last slot stay unused.
    indices = torch.tensor([4, 1, 5, 2][: len(lengths)], device="cuda")
    state = torch.randn(
        7, channels, state_len, dtype=torch.bfloat16, device="cuda", generator=generator
    )
    value = torch.randn(
        sum(lengths), channels, dtype=torch.bfloat16, device="cuda", generator=generator
    )
    weight = torch.randn(channels, 1, 4, dtype=torch.bfloat16, device="cuda", generator=generator)
    req_indices, token_offsets = _packed_layout(lengths)
    length_tensor = torch.tensor(lengths, device="cuda")

    expected_state = state.clone()
    outputs = []
    start = 0
    for slot, length in zip(indices.tolist(), lengths):
        conv_input = torch.cat((state[slot], value[start : start + length].T), dim=1)
        if length:
            conv = F.conv1d(conv_input.unsqueeze(0), weight, dilation=dilation, groups=channels)
            outputs.append(F.silu(conv).squeeze(0).T)
        expected_state[slot] = conv_input[:, length:]
        start += length

    assert can_use_ple_packed_short_conv(
        state, indices, value, weight, req_indices, token_offsets, length_tensor, dilation
    )
    actual = ple_packed_short_conv(
        state, indices, value, weight, req_indices, token_offsets, length_tensor, dilation
    )
    assert torch.equal(actual, torch.cat(outputs))
    assert torch.equal(state, expected_state)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_ple_packed_short_conv_rejects_unsupported_inputs() -> None:
    state = torch.zeros(2, 32, 9, dtype=torch.bfloat16, device="cuda")
    indices = torch.arange(2, device="cuda")
    value = torch.zeros(3, 32, dtype=torch.bfloat16, device="cuda")
    weight = torch.zeros(32, 1, 4, dtype=torch.bfloat16, device="cuda")
    req_indices, token_offsets = _packed_layout([2, 1])
    lengths = torch.tensor([2, 1], device="cuda")
    args = (state, indices, value, weight, req_indices, token_offsets, lengths)

    assert can_use_ple_packed_short_conv(*args, 3)
    assert not can_use_ple_packed_short_conv(*args, 2)
    assert not can_use_ple_packed_short_conv(state.float(), *args[1:], 3)
    assert not can_use_ple_packed_short_conv(*args[:4], req_indices[:2], *args[5:], 3)
    assert not can_use_ple_packed_short_conv(*args[:6], lengths.int(), 3)
    with pytest.raises(ValueError, match="unsupported tensor geometry"):
        ple_packed_short_conv(*args[:2], value.float(), *args[3:], 3)
