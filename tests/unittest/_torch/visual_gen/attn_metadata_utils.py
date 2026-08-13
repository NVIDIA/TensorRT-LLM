# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Test helpers that derive attention metadata from the tensors a test passes in."""

from typing import Optional

import torch

from tensorrt_llm._torch.attention_backend.interface import AttentionMetadata
from tensorrt_llm._torch.visual_gen.attention_backend.interface import AttentionTensorLayout
from tensorrt_llm._torch.visual_gen.attention_backend.metadata import make_diffusion_attn_metadata
from tensorrt_llm._torch.visual_gen.attention_backend.utils import get_visual_gen_attention_backend


def make_attn_metadata(
    backend: str,
    hidden_states: torch.Tensor,
    encoder_hidden_states: Optional[torch.Tensor] = None,
    *,
    q_seq_len: Optional[int] = None,
    kv_seq_len: Optional[int] = None,
) -> AttentionMetadata:
    """Metadata for one site, sized from ``hidden_states`` and an optional KV stream.

    ``q_seq_len`` / ``kv_seq_len`` override the lengths read off the tensors.
    """
    if kv_seq_len is None and encoder_hidden_states is not None:
        kv_seq_len = encoder_hidden_states.shape[1]

    return make_diffusion_attn_metadata(
        get_visual_gen_attention_backend(backend).Metadata,
        batch_size=hidden_states.shape[0],
        q_seq_lens=q_seq_len if q_seq_len is not None else hidden_states.shape[1],
        kv_seq_lens=kv_seq_len,
    )


def make_backend_attn_metadata(
    backend,
    q: torch.Tensor,
    k: Optional[torch.Tensor] = None,
) -> AttentionMetadata:
    """Metadata for ``q``/``k`` passed straight to a backend, in its own layout."""
    seq_axis = 2 if backend.preferred_layout == AttentionTensorLayout.HND else 1
    return make_diffusion_attn_metadata(
        type(backend).Metadata,
        batch_size=q.shape[0],
        q_seq_lens=q.shape[seq_axis],
        kv_seq_lens=None if k is None else k.shape[seq_axis],
    )


def flux_attn_metadata(model, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor):
    """The single joint text+image site a FLUX / FLUX.2 transformer forward needs."""
    return model.create_attn_metadata(
        batch_size=hidden_states.shape[0],
        text_seq_len=encoder_hidden_states.shape[1],
        image_seq_len=hidden_states.shape[1],
    )["self"]


def cosmos3_attn_metadata_kwargs(
    model,
    hidden_states: torch.Tensor,
    text_mask: torch.Tensor,
    video_shape,
    audio_latents: Optional[torch.Tensor] = None,
) -> dict:
    """Cosmos3's ``und`` / ``mixed`` sites as ``forward`` keyword arguments."""
    sites = model.create_attn_metadata(
        batch_size=hidden_states.shape[0],
        text_seq_len=text_mask.shape[1],
        text_lens=text_mask.sum(dim=1).tolist(),
        video_shape=video_shape,
        num_audio_tokens=(audio_latents.shape[2] if audio_latents is not None else 0),
    )
    return {
        "attn_metadata_und": sites["und"],
        "attn_metadata_mixed": sites["mixed"],
        "attn_metadata_mixed_ragged": sites["mixed_ragged"],
    }


def ltx2_attn_metadata(model, video, audio, text_cache) -> dict:
    """LTX-2's sites, unwrapping the lengths from the ``Modality`` bundles."""
    batch_size = 1
    if video is not None:
        batch_size = video.latent.shape[0]
    elif audio is not None:
        batch_size = audio.latent.shape[0]
    return model.create_attn_metadata(
        batch_size=batch_size,
        video_seq_len=0 if video is None else video.latent.shape[1],
        audio_seq_len=0 if audio is None else audio.latent.shape[1],
        text_cache=text_cache,
    )
