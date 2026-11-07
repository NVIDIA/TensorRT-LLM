# SPDX-FileCopyrightText: Copyright 2024-2026 SGLang Team
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Adapted from SGLang's Qwen4 PLE kernels.
"""Shape-specialized fused kernels for decode-sized Qwen4-Exp PLE.

Each entry point has a matching ``can_use_*`` guard; other legal model
geometries remain on the generic PyTorch path.
"""

import torch
import triton
import triton.language as tl

_NGRAM_SIZE = 3
_HEADS_PER_NGRAM = 8
_NGRAM_HEADS = 16
_HC_COUNT = 4
_HIDDEN_SIZE = 2560
_MAX_SHORT_CONV_STATE_LEN = 16
_SHORT_CONV_KERNEL_SIZE = 4
_SHORT_CONV_STATE_LEN = 9


def _same_cuda_device(*tensors: torch.Tensor) -> bool:
    """Return whether every specialized-kernel operand shares one CUDA device."""
    return bool(tensors) and all(
        tensor.is_cuda and tensor.device == tensors[0].device for tensor in tensors
    )


@triton.jit
def _round_bf16_to_fp32(value):
    """RNE-round an FP32 register to BF16 precision."""
    bits = value.to(tl.int32, bitcast=True)
    rounding_bias = 0x7FFF + ((bits >> 16) & 1)
    rounded_bits = (bits + rounding_bias) & -65536
    return rounded_bits.to(tl.float32, bitcast=True)


@triton.jit
def _ngram_hash_kernel(
    contexts_ptr,
    multipliers_ptr,
    vocab_sizes_ptr,
    offsets_ptr,
    output_ptr,
    num_outputs,
    eos_token_id,
    ngram_size: tl.constexpr,
    heads_per_ngram: tl.constexpr,
    ngram_heads: tl.constexpr,
    block_size: tl.constexpr,
) -> None:
    output_idx = tl.program_id(0) * block_size + tl.arange(0, block_size)
    mask = output_idx < num_outputs
    token_idx = output_idx // ngram_heads
    head_idx = output_idx % ngram_heads
    context_base = token_idx * ngram_size

    token_0 = tl.load(contexts_ptr + context_base, mask=mask, other=0)
    token_1 = tl.load(contexts_ptr + context_base + 1, mask=mask, other=0)
    token_2 = tl.load(contexts_ptr + context_base + 2, mask=mask, other=0)
    multiplier_0 = tl.load(multipliers_ptr)
    multiplier_1 = tl.load(multipliers_ptr + 1)
    multiplier_2 = tl.load(multipliers_ptr + 2)

    # Evaluate the eos-delimited shifts only at the final position of each
    # three-token window. The bigram sees token_1; the trigram sees token_0
    # only when neither preceding token is a segment boundary.
    previous_2 = tl.where(
        (token_0 == eos_token_id) | (token_1 == eos_token_id),
        eos_token_id,
        token_0,
    )
    mixed = (token_2 * multiplier_0) ^ (token_1 * multiplier_1)
    mixed_3 = mixed ^ (previous_2 * multiplier_2)
    mixed = tl.where(head_idx < heads_per_ngram, mixed, mixed_3)

    vocab_size = tl.load(vocab_sizes_ptr + head_idx, mask=mask, other=1)
    offset = tl.load(offsets_ptr + head_idx, mask=mask, other=0)
    tl.store(output_ptr + output_idx, mixed % vocab_size + offset, mask=mask)


def can_use_ple_ngram_hash(
    contexts: torch.Tensor,
    multipliers: torch.Tensor,
    vocab_sizes: torch.Tensor,
    offsets: torch.Tensor,
) -> bool:
    """Return whether tensors match the released PLE hash contract."""
    return (
        _same_cuda_device(contexts, multipliers, vocab_sizes, offsets)
        and contexts.dtype == torch.long
        and contexts.dim() == 2
        and contexts.shape[1] == _NGRAM_SIZE
        and contexts.is_contiguous()
        and multipliers.dtype == torch.long
        and multipliers.numel() == _NGRAM_SIZE
        and multipliers.is_contiguous()
        and vocab_sizes.dtype == torch.long
        and vocab_sizes.numel() == _NGRAM_HEADS
        and vocab_sizes.is_contiguous()
        and offsets.dtype == torch.long
        and offsets.numel() == _NGRAM_HEADS
        and offsets.is_contiguous()
    )


@torch.library.custom_op("trtllm::qwen4_exp_ple_ngram_hash", mutates_args=())
def ple_ngram_hash(
    contexts: torch.Tensor,
    multipliers: torch.Tensor,
    vocab_sizes: torch.Tensor,
    offsets: torch.Tensor,
    eos_token_id: int,
) -> torch.Tensor:
    """Hash every three-token PLE context into its 16 embedding rows."""
    if not can_use_ple_ngram_hash(contexts, multipliers, vocab_sizes, offsets):
        raise ValueError("PLE n-gram hash received unsupported tensor geometry")
    output = torch.empty(
        (contexts.shape[0], _NGRAM_HEADS),
        dtype=torch.long,
        device=contexts.device,
    )
    num_outputs = output.numel()
    if num_outputs:
        block_size = 256
        with torch.cuda.device(contexts.device.index):
            _ngram_hash_kernel[(triton.cdiv(num_outputs, block_size),)](
                contexts,
                multipliers,
                vocab_sizes,
                offsets,
                output,
                num_outputs,
                eos_token_id,
                ngram_size=_NGRAM_SIZE,
                heads_per_ngram=_HEADS_PER_NGRAM,
                ngram_heads=_NGRAM_HEADS,
                block_size=block_size,
                num_warps=4,
            )
    return output


@ple_ngram_hash.register_fake
def _(
    contexts: torch.Tensor,
    multipliers: torch.Tensor,
    vocab_sizes: torch.Tensor,
    offsets: torch.Tensor,
    eos_token_id: int,
) -> torch.Tensor:
    del multipliers, vocab_sizes, offsets, eos_token_id
    return contexts.new_empty((contexts.shape[0], _NGRAM_HEADS))


@triton.jit
def _gate_value_kernel(
    gate_ptr,
    value_ptr,
    output_ptr,
    num_tokens,
    hc_count: tl.constexpr,
    hidden_size: tl.constexpr,
    block_size: tl.constexpr,
) -> None:
    token_group = tl.program_id(0)
    token = token_group // hc_count
    hidden = tl.arange(0, block_size)
    mask = (token < num_tokens) & (hidden < hidden_size)

    # The input is already the BF16 output of multiply, reduction, and division.
    # Reproduce each remaining eager BF16 rounding boundary before broadcasting.
    gate = tl.load(gate_ptr + token_group).to(tl.float32)
    magnitude = tl.maximum(tl.abs(gate), 1.0e-6)
    root = _round_bf16_to_fp32(tl.sqrt(magnitude))
    sign = tl.where(gate > 0.0, 1.0, tl.where(gate < 0.0, -1.0, 0.0))
    transformed = _round_bf16_to_fp32(root * sign)
    activated = _round_bf16_to_fp32(tl.sigmoid(transformed))

    value = tl.load(
        value_ptr + token * hidden_size + hidden,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    output_offset = token_group * hidden_size + hidden
    tl.store(output_ptr + output_offset, activated * value, mask=mask)


def can_use_ple_gate_value(gate: torch.Tensor, value: torch.Tensor) -> bool:
    """Return whether tensors match the released BF16 PLE gate contract."""
    return (
        _same_cuda_device(gate, value)
        and gate.dtype == torch.bfloat16
        and gate.dim() == 3
        and gate.shape[1:] == (_HC_COUNT, 1)
        and gate.is_contiguous()
        and value.dtype == gate.dtype
        and value.shape == (gate.shape[0], _HIDDEN_SIZE)
        and value.is_contiguous()
    )


@torch.library.custom_op("trtllm::qwen4_exp_ple_gate_value", mutates_args=())
def ple_gate_value(gate: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    """Apply signed-sqrt sigmoid and broadcast the PLE value in one launch."""
    if not can_use_ple_gate_value(gate, value):
        raise ValueError("PLE gate/value fusion received unsupported tensor geometry")
    output = torch.empty(
        (gate.shape[0], _HC_COUNT, _HIDDEN_SIZE),
        dtype=value.dtype,
        device=value.device,
    )
    if gate.shape[0]:
        with torch.cuda.device(gate.device.index):
            _gate_value_kernel[(gate.shape[0] * _HC_COUNT,)](
                gate,
                value,
                output,
                gate.shape[0],
                hc_count=_HC_COUNT,
                hidden_size=_HIDDEN_SIZE,
                block_size=4096,
                num_warps=8,
            )
    return output


@ple_gate_value.register_fake
def _(gate: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    return value.new_empty((gate.shape[0], _HC_COUNT, _HIDDEN_SIZE))


@triton.jit
def _short_conv_state_kernel(
    state_ptr,
    state_indices_ptr,
    value_ptr,
    conv_input_ptr,
    num_tokens,
    channels: tl.constexpr,
    state_len: tl.constexpr,
    block_channels: tl.constexpr,
    block_state_len: tl.constexpr,
) -> None:
    token = tl.program_id(0)
    channel = tl.program_id(1) * block_channels + tl.arange(0, block_channels)[:, None]
    state_col = tl.arange(0, block_state_len)[None, :]
    channel_mask = (token < num_tokens) & (channel < channels)
    state_mask = channel_mask & (state_col < state_len)
    state_index = tl.load(state_indices_ptr + token, mask=token < num_tokens, other=0)
    state_base = state_index * channels * state_len
    state_offset = state_base + channel * state_len + state_col
    output_base = token * channels * (state_len + 1)
    output_offset = output_base + channel * (state_len + 1) + state_col

    old_state = tl.load(state_ptr + state_offset, mask=state_mask, other=0.0)
    tl.store(conv_input_ptr + output_offset, old_state, mask=state_mask)
    value = tl.load(
        value_ptr + token * channels + channel,
        mask=channel_mask,
        other=0.0,
    )
    tl.store(
        conv_input_ptr + output_base + channel * (state_len + 1) + state_len,
        value,
        mask=channel_mask,
    )
    tl.debug_barrier()

    update_mask = state_mask & (state_col < state_len - 1)
    next_value = tl.load(
        conv_input_ptr + output_offset + 1,
        mask=update_mask,
        other=0.0,
    )
    tl.store(state_ptr + state_offset, next_value, mask=update_mask)
    tl.store(
        state_ptr + state_base + channel * state_len + state_len - 1,
        value,
        mask=channel_mask,
    )


def can_use_ple_short_conv_state(
    state: torch.Tensor,
    state_indices: torch.Tensor,
    value: torch.Tensor,
) -> bool:
    """Return whether decode state movement can use the fused kernel."""
    return (
        _same_cuda_device(state, state_indices, value)
        and state.dtype in (torch.bfloat16, torch.float16)
        and state.dim() == 3
        and state.is_contiguous()
        and 0 < state.shape[2] <= _MAX_SHORT_CONV_STATE_LEN
        and state_indices.dtype == torch.long
        and state_indices.dim() == 1
        and state_indices.is_contiguous()
        and value.dtype == state.dtype
        and value.dim() == 2
        and value.is_contiguous()
        and value.shape == (state_indices.shape[0], state.shape[1])
    )


@torch.library.custom_op(
    "trtllm::qwen4_exp_ple_short_conv_state",
    mutates_args=("state",),
)
def ple_short_conv_state(
    state: torch.Tensor,
    state_indices: torch.Tensor,
    value: torch.Tensor,
) -> torch.Tensor:
    """Build ``[selected state, value]`` and advance unique decode slots."""
    if not can_use_ple_short_conv_state(state, state_indices, value):
        raise ValueError("PLE short-convolution state update received unsupported geometry")
    state_len = state.shape[2]
    conv_input = torch.empty(
        (value.shape[0], value.shape[1], state_len + 1),
        dtype=value.dtype,
        device=value.device,
    )
    if value.shape[0]:
        block_channels = 128
        block_state_len = triton.next_power_of_2(state_len)
        with torch.cuda.device(state.device.index):
            _short_conv_state_kernel[(value.shape[0], triton.cdiv(value.shape[1], block_channels))](
                state,
                state_indices,
                value,
                conv_input,
                value.shape[0],
                channels=state.shape[1],
                state_len=state_len,
                block_channels=block_channels,
                block_state_len=block_state_len,
                num_warps=8,
            )
    return conv_input


@ple_short_conv_state.register_fake
def _(
    state: torch.Tensor,
    state_indices: torch.Tensor,
    value: torch.Tensor,
) -> torch.Tensor:
    del state_indices
    return value.new_empty((value.shape[0], value.shape[1], state.shape[2] + 1))


@triton.jit
def _decode_short_conv_kernel(
    state_ptr,
    state_indices_ptr,
    value_ptr,
    weight_ptr,
    output_ptr,
    num_tokens,
    channels: tl.constexpr,
    kernel_size: tl.constexpr,
    state_len: tl.constexpr,
    block_channels: tl.constexpr,
) -> None:
    token = tl.program_id(0)
    channel = tl.program_id(1) * block_channels + tl.arange(0, block_channels)
    mask = (token < num_tokens) & (channel < channels)
    state_index = tl.load(state_indices_ptr + token, mask=token < num_tokens, other=0)
    state_base = state_index * channels * state_len + channel * state_len

    state_0 = tl.load(state_ptr + state_base, mask=mask, other=0.0).to(tl.float32)
    state_1 = tl.load(state_ptr + state_base + 1, mask=mask, other=0.0)
    state_2 = tl.load(state_ptr + state_base + 2, mask=mask, other=0.0)
    state_3 = tl.load(state_ptr + state_base + 3, mask=mask, other=0.0).to(tl.float32)
    state_4 = tl.load(state_ptr + state_base + 4, mask=mask, other=0.0)
    state_5 = tl.load(state_ptr + state_base + 5, mask=mask, other=0.0)
    state_6 = tl.load(state_ptr + state_base + 6, mask=mask, other=0.0).to(tl.float32)
    state_7 = tl.load(state_ptr + state_base + 7, mask=mask, other=0.0)
    state_8 = tl.load(state_ptr + state_base + 8, mask=mask, other=0.0)
    value = tl.load(value_ptr + token * channels + channel, mask=mask, other=0.0)

    weight_base = channel * kernel_size
    weight_0 = tl.load(weight_ptr + weight_base, mask=mask, other=0.0).to(tl.float32)
    weight_1 = tl.load(weight_ptr + weight_base + 1, mask=mask, other=0.0).to(tl.float32)
    weight_2 = tl.load(weight_ptr + weight_base + 2, mask=mask, other=0.0).to(tl.float32)
    weight_3 = tl.load(weight_ptr + weight_base + 3, mask=mask, other=0.0).to(tl.float32)
    conv = state_0 * weight_0
    conv += state_3 * weight_1
    conv += state_6 * weight_2
    conv += value.to(tl.float32) * weight_3

    # F.conv1d materializes a BF16 output before F.silu. Preserve that
    # otherwise-observable rounding boundary while keeping both operations in
    # this launch.
    conv = _round_bf16_to_fp32(conv)
    tl.store(
        output_ptr + token * channels + channel,
        conv * tl.sigmoid(conv),
        mask=mask,
    )

    tl.store(state_ptr + state_base, state_1, mask=mask)
    tl.store(state_ptr + state_base + 1, state_2, mask=mask)
    tl.store(state_ptr + state_base + 2, state_3, mask=mask)
    tl.store(state_ptr + state_base + 3, state_4, mask=mask)
    tl.store(state_ptr + state_base + 4, state_5, mask=mask)
    tl.store(state_ptr + state_base + 5, state_6, mask=mask)
    tl.store(state_ptr + state_base + 6, state_7, mask=mask)
    tl.store(state_ptr + state_base + 7, state_8, mask=mask)
    tl.store(state_ptr + state_base + 8, value, mask=mask)


def can_use_ple_decode_short_conv(
    state: torch.Tensor,
    state_indices: torch.Tensor,
    value: torch.Tensor,
    weight: torch.Tensor,
) -> bool:
    """Return whether decode tensors match the bitwise-exact fused contract."""
    return (
        _same_cuda_device(state, state_indices, value, weight)
        and state.dtype == torch.bfloat16
        and state.dim() == 3
        and state.shape[2] == _SHORT_CONV_STATE_LEN
        and state.is_contiguous()
        and state_indices.dtype == torch.long
        and state_indices.dim() == 1
        and state_indices.is_contiguous()
        and value.dtype == state.dtype
        and value.dim() == 2
        and value.is_contiguous()
        and value.shape == (state_indices.shape[0], state.shape[1])
        and weight.dtype == state.dtype
        and weight.shape == (state.shape[1], 1, _SHORT_CONV_KERNEL_SIZE)
        and weight.is_contiguous()
    )


@torch.library.custom_op(
    "trtllm::qwen4_exp_ple_decode_short_conv",
    mutates_args=("state",),
)
def ple_decode_short_conv(
    state: torch.Tensor,
    state_indices: torch.Tensor,
    value: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    """Advance decode state and compute dilated short-conv SiLU in one launch."""
    if not can_use_ple_decode_short_conv(state, state_indices, value, weight):
        raise ValueError("PLE decode short convolution received unsupported tensor geometry")
    output = torch.empty_like(value)
    if value.shape[0]:
        block_channels = 64
        with torch.cuda.device(state.device.index):
            _decode_short_conv_kernel[
                (value.shape[0], triton.cdiv(value.shape[1], block_channels))
            ](
                state,
                state_indices,
                value,
                weight,
                output,
                value.shape[0],
                channels=value.shape[1],
                kernel_size=_SHORT_CONV_KERNEL_SIZE,
                state_len=_SHORT_CONV_STATE_LEN,
                block_channels=block_channels,
                num_warps=4,
            )
    return output


@ple_decode_short_conv.register_fake
def _(
    state: torch.Tensor,
    state_indices: torch.Tensor,
    value: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    del state, state_indices, weight
    return torch.empty_like(value)


__all__ = [
    "can_use_ple_decode_short_conv",
    "can_use_ple_gate_value",
    "can_use_ple_ngram_hash",
    "can_use_ple_short_conv_state",
    "ple_gate_value",
    "ple_decode_short_conv",
    "ple_ngram_hash",
    "ple_short_conv_state",
]
