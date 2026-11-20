# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The family Transformer forward with one shared input for the two FP32 heads."""

from typing import Optional

import torch

from ..transformer_minimax_h3 import (
    MINIMAX_H3_MODALITY_NUM,
    MiniMaxH3StaticContext,
    MiniMaxH3TransformerOutput,
    _pad_rotary_emb,
    _pad_tensor_dim,
    _padding_to_multiple,
)


def shared_output_forward(
    self,
    hidden_states: torch.Tensor,
    audio_hidden_states: torch.Tensor,
    encoder_hidden_states: Optional[torch.Tensor],
    timestep: torch.Tensor,
    conditioning_timesteps: torch.Tensor,
    timestep_indices: torch.Tensor,
    token_tags: torch.Tensor,
    position_ids: Optional[torch.Tensor],
    video_indices: torch.Tensor,
    audio_indices: torch.Tensor,
    text_indices: torch.Tensor,
    attention_kwargs: Optional[dict[str, object]] = None,
    return_dict: bool = True,
    static_context: Optional[MiniMaxH3StaticContext] = None,
) -> MiniMaxH3TransformerOutput | tuple[torch.Tensor, torch.Tensor]:
    del attention_kwargs
    if position_ids is None:
        if static_context is None:
            raise ValueError("position_ids is required when static_context is not provided.")
        sequence_length = static_context.sequence_length
    else:
        if position_ids.ndim != 2 or position_ids.shape[-1] != 3:
            raise ValueError(
                f"position_ids must have shape [sequence_length, 3], got {list(position_ids.shape)}."
            )
        sequence_length = position_ids.shape[0]
    self._validate_layout(sequence_length, token_tags, timestep_indices, conditioning_timesteps)
    if static_context is None:
        if encoder_hidden_states is None or position_ids is None:
            raise ValueError(
                "encoder_hidden_states and position_ids are required to prepare static context."
            )
        static_context = self.prepare_static_context(encoder_hidden_states, position_ids)
    if static_context.sequence_length != sequence_length:
        raise ValueError("static_context RoPE length must match the packed sequence length.")
    if static_context.text_embeds.shape[1] != text_indices.numel():
        raise ValueError("text_indices must contain one entry for every cached text embedding.")
    video_embeds = self.proj_in(self._projection_input(self.proj_in, hidden_states))
    audio_embeds = self.audio_proj_in(
        self._projection_input(self.audio_proj_in, audio_hidden_states)
    )
    text_embeds = static_context.text_embeds
    packed_hidden_states = text_embeds.new_zeros(
        (text_embeds.shape[0], sequence_length, text_embeds.shape[-1])
    )
    packed_hidden_states = packed_hidden_states.index_copy(1, text_indices, text_embeds)
    packed_hidden_states = packed_hidden_states.index_copy(
        1, video_indices, video_embeds.to(text_embeds.dtype)
    )
    packed_hidden_states = packed_hidden_states.index_copy(
        1, audio_indices, audio_embeds.to(text_embeds.dtype)
    )
    temb = self.time_proj(conditioning_timesteps)
    temb = self.time_embedder(temb.to(self.time_embedder.linear_1.weight.dtype))
    adaln_indices = timestep_indices * MINIMAX_H3_MODALITY_NUM + token_tags.clamp(min=0)
    timestep_indices_for_norm = timestep_indices
    rotary_emb = static_context.rotary_emb
    pad = _padding_to_multiple(sequence_length, self.sharder.size)
    has_existing_padding = bool((token_tags < 0).any())
    valid_seq_len = None
    if self._attn_backend == "TRTLLM" and (pad or has_existing_padding):
        valid_seq_len = sequence_length
        if has_existing_padding:
            valid_seq_len = int((token_tags >= 0).sum().item())
    token_tags_for_mask = token_tags
    if pad:
        packed_hidden_states = _pad_tensor_dim(packed_hidden_states, 1, pad)
        adaln_indices = _pad_tensor_dim(adaln_indices, 0, pad)
        timestep_indices_for_norm = _pad_tensor_dim(timestep_indices, 0, pad)
        token_tags_for_mask = _pad_tensor_dim(token_tags, 0, pad, -1)
        rotary_emb = _pad_rotary_emb(rotary_emb, pad)
    key_padding_mask = None
    if pad or has_existing_padding:
        if not self._supports_key_padding_mask:
            raise NotImplementedError(
                "Padded packed sequences (negative token_tags) need an attention backend that honours "
                f"key_padding_mask; the {self._attn_backend} backend cannot express it. "
                "Use VANILLA, FA4, TRTLLM, FLASHINFER or CUDNN."
            )
        if self._attn_backend in ("FA4", "TRTLLM", "CUDNN") and bool(
            ((token_tags_for_mask[:-1] < 0) & (token_tags_for_mask[1:] >= 0)).any()
        ):
            raise NotImplementedError(
                f"{self._attn_backend} requires trailing packed-row padding; use VANILLA for interior padding."
            )
        if self._attn_backend != "TRTLLM":
            with torch.inference_mode(False):
                key_padding_mask = (
                    (token_tags_for_mask >= 0)
                    .unsqueeze(0)
                    .expand(packed_hidden_states.shape[0], -1)
                )
    padded_sequence_length = packed_hidden_states.shape[1]
    packed_hidden_states = self.sharder.shard(packed_hidden_states, dim=1)
    adaln_indices = self.sharder.shard(adaln_indices, dim=0)
    timestep_indices_for_norm = self.sharder.shard(timestep_indices_for_norm, dim=0)
    rotary_emb = self.sharder.shard_rope(rotary_emb, seq_len=padded_sequence_length, seq_dim=0)
    for block in self.transformer_blocks:
        packed_hidden_states = block(
            packed_hidden_states,
            temb,
            adaln_indices,
            rotary_emb,
            key_padding_mask,
            timestep,
            valid_seq_len=valid_seq_len,
        )
    packed_hidden_states = self.norm_out(packed_hidden_states, temb, timestep_indices_for_norm)
    packed_hidden_states = self.sharder.gather(
        packed_hidden_states, dim=1, unpad_to=sequence_length
    )
    shared_projection_input = self._projection_input(self.proj_out, packed_hidden_states)
    video_output = self.proj_out(shared_projection_input).index_select(1, video_indices)
    audio_output = self.audio_proj_out(shared_projection_input).index_select(1, audio_indices)
    if not return_dict:
        return (video_output, audio_output)
    return MiniMaxH3TransformerOutput(sample=video_output, audio_sample=audio_output)
