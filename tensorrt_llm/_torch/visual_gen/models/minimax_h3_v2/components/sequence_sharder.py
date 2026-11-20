# Copyright 2026 The MiniMax and HuggingFace Teams. All rights reserved.

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

"""H3 packed-row partitioning for the text refiner and audio-video DiT."""

from typing import TYPE_CHECKING

import torch
import torch.distributed as dist

if TYPE_CHECKING:
    from tensorrt_llm._torch.visual_gen.mapping import VisualGenMapping


class H3SequenceSharder:
    """Partition padded H3 rows for the token refiner and DiT."""

    def __init__(self, mapping: "VisualGenMapping | None", *, num_attention_heads: int):
        if mapping is None:
            self.size, self.rank, self.group = 1, 0, None
            return
        self.size, self.rank, self.group = mapping.seq_size, mapping.seq_rank, mapping.seq_group()
        if self.size > 1 and self.group is None:
            raise ValueError("H3 sequence sharding requires a sequence process group.")
        if num_attention_heads % mapping.ulysses_size:
            raise ValueError(
                f"num_attention_heads={num_attention_heads} must be divisible by "
                f"ulysses_size={mapping.ulysses_size}"
            )

    def shard(self, tensor: torch.Tensor, dim: int) -> torch.Tensor:
        if self.size == 1:
            return tensor
        rows, remainder = divmod(tensor.size(dim), self.size)
        if remainder:
            raise ValueError(
                f"H3 sequence length {tensor.size(dim)} must be divisible by {self.size}."
            )
        return tensor.narrow(dim, self.rank * rows, rows).contiguous()

    def shard_rope(
        self, rope: tuple[torch.Tensor, torch.Tensor], *, seq_len: int, seq_dim: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.size == 1:
            return rope
        cos, sin = rope
        if cos.size(seq_dim) != seq_len:
            raise ValueError(f"H3 RoPE has {cos.size(seq_dim)} rows, expected {seq_len}.")
        return self.shard(cos, seq_dim), self.shard(sin, seq_dim)

    def gather(self, tensor: torch.Tensor, dim: int, *, unpad_to: int) -> torch.Tensor:
        if self.size == 1:
            return tensor
        tensor = tensor.contiguous()
        parts = [torch.empty_like(tensor) for _ in range(self.size)]
        dist.all_gather(parts, tensor, group=self.group)
        return torch.cat(parts, dim=dim).narrow(dim, 0, unpad_to)


def _pad_tensor_dim(
    tensor: torch.Tensor,
    dim: int,
    pad: int,
    value: float | int | bool = 0,
) -> torch.Tensor:
    if pad == 0:
        return tensor
    pad_shape = list(tensor.shape)
    pad_shape[dim] = pad
    return torch.cat([tensor, tensor.new_full(pad_shape, value)], dim=dim)


def _padding_to_multiple(seq_len: int, multiple: int) -> int:
    if multiple <= 1:
        return 0
    return (multiple - seq_len % multiple) % multiple


def _pad_rotary_emb(
    rotary_emb: tuple[torch.Tensor, torch.Tensor],
    pad: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    cos, sin = rotary_emb
    return (
        _pad_tensor_dim(cos, 0, pad, 1.0),
        _pad_tensor_dim(sin, 0, pad, 0.0),
    )


def shard_refiner_inputs(
    hidden_states: torch.Tensor,
    sharder: H3SequenceSharder,
    backend: str,
    supports_key_padding_mask: bool,
) -> tuple[torch.Tensor, torch.Tensor | None, int | None]:
    seq_len = hidden_states.shape[1]
    pad = _padding_to_multiple(seq_len, sharder.size)
    key_padding_mask = None
    if pad:
        if not supports_key_padding_mask:
            raise NotImplementedError(
                "Padded token-refiner sequences require key_padding_mask support; "
                "use VANILLA, FA4, TRTLLM or FLASHINFER."
            )
        if backend != "TRTLLM":
            # Attention caches use the mask version to detect changes.
            with torch.inference_mode(False):
                valid = hidden_states.new_ones((hidden_states.shape[0], seq_len), dtype=torch.bool)
                key_padding_mask = _pad_tensor_dim(valid, 1, pad, False)
        hidden_states = _pad_tensor_dim(hidden_states, 1, pad)
    hidden_states = sharder.shard(hidden_states, dim=1)
    return hidden_states, key_padding_mask, seq_len if pad else None


def shard_dit_inputs(
    packed_hidden_states: torch.Tensor,
    adaln_indices: torch.Tensor,
    timestep_indices: torch.Tensor,
    token_tags: torch.Tensor,
    rotary_emb: tuple[torch.Tensor, torch.Tensor],
    sharder: H3SequenceSharder,
    backend: str,
    supports_key_padding_mask: bool,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    tuple[torch.Tensor, torch.Tensor],
    torch.Tensor | None,
    int | None,
]:
    sequence_length = token_tags.shape[0]
    pad = _padding_to_multiple(sequence_length, sharder.size)
    has_existing_padding = bool((token_tags < 0).any())
    valid_seq_len = None
    if backend == "TRTLLM" and (pad or has_existing_padding):
        valid_seq_len = sequence_length
        if has_existing_padding:
            valid_seq_len = int((token_tags >= 0).sum().item())
    token_tags_for_mask = token_tags
    if pad:
        packed_hidden_states = _pad_tensor_dim(packed_hidden_states, 1, pad)
        adaln_indices = _pad_tensor_dim(adaln_indices, 0, pad)
        timestep_indices = _pad_tensor_dim(timestep_indices, 0, pad)
        token_tags_for_mask = _pad_tensor_dim(token_tags, 0, pad, -1)
        rotary_emb = _pad_rotary_emb(rotary_emb, pad)

    key_padding_mask = None
    if pad or has_existing_padding:
        if not supports_key_padding_mask:
            raise NotImplementedError(
                "Padded packed sequences (negative token_tags) need an attention "
                "backend that honours key_padding_mask; the "
                f"{backend} backend cannot express it. Use "
                "VANILLA, FA4, TRTLLM or FLASHINFER."
            )
        # FA4 seqused_k and TRTLLM prefix trimming require trailing padding.
        # Ulysses appends padding in this layout.
        if backend in ("FA4", "TRTLLM") and bool(
            ((token_tags_for_mask[:-1] < 0) & (token_tags_for_mask[1:] >= 0)).any()
        ):
            raise NotImplementedError(
                f"{backend} requires trailing packed-row padding; use VANILLA for interior padding."
            )
        if backend != "TRTLLM":
            # Attention caches use the mask version to detect changes.
            with torch.inference_mode(False):
                key_padding_mask = (
                    (token_tags_for_mask >= 0)
                    .unsqueeze(0)
                    .expand(packed_hidden_states.shape[0], -1)
                )

    padded_sequence_length = packed_hidden_states.shape[1]
    packed_hidden_states = sharder.shard(packed_hidden_states, dim=1)
    adaln_indices = sharder.shard(adaln_indices, dim=0)
    timestep_indices = sharder.shard(timestep_indices, dim=0)
    rotary_emb = sharder.shard_rope(
        rotary_emb,
        seq_len=padded_sequence_length,
        seq_dim=0,
    )
    return (
        packed_hidden_states,
        adaln_indices,
        timestep_indices,
        rotary_emb,
        key_padding_mask,
        valid_seq_len,
    )
