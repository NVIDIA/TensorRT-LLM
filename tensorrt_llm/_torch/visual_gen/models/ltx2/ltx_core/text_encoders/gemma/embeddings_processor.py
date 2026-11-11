from typing import NamedTuple

import torch
from torch import nn

from tensorrt_llm._torch.visual_gen.models.ltx2.ltx_core.model.disposable import Disposable
from tensorrt_llm._torch.visual_gen.models.ltx2.ltx_core.text_encoders.gemma.embeddings_connector import Embeddings1DConnector


class EmbeddingsProcessorOutput(NamedTuple):
    video_encoding: torch.Tensor
    audio_encoding: torch.Tensor | None
    attention_mask: torch.Tensor


def convert_to_additive_mask(attention_mask: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Convert binary attention mask to additive form for transformer masking."""
    return (attention_mask.to(torch.int64) - 1).to(dtype).reshape(
        (attention_mask.shape[0], 1, -1, attention_mask.shape[-1])
    ) * torch.finfo(dtype).max


def _compute_right_pad_order(additive_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute the index permutation that places valid tokens before pads in each row.
    Stable sort: valid tokens keep their relative order. Idempotent for inputs already
    right-padded. The sort and reordered mask depend only on the mask, so they can be
    computed once and reused across multiple feature tensors that share the mask.
    Args:
        additive_mask: (B, 1, 1, S) additive mask, ``0.0`` for valid, ``-finfo.max`` for pad.
    Returns:
        ``(sort_idx, reordered_additive_mask)``: ``sort_idx`` is (B, S); the reordered mask
        has the same shape as the input.
    """
    binary = (additive_mask[:, 0, 0, :] >= 0).to(torch.int32)  # (B, S)
    sort_idx = torch.argsort(binary, dim=-1, descending=True, stable=True)  # (B, S)
    new_binary = torch.gather(binary, 1, sort_idx)
    new_additive = (new_binary.to(additive_mask.dtype) - 1) * torch.finfo(additive_mask.dtype).max
    return sort_idx, new_additive[:, None, None, :]


def _apply_right_pad_order(features: torch.Tensor, sort_idx: torch.Tensor) -> torch.Tensor:
    """Apply a precomputed right-pad permutation (from ``_compute_right_pad_order``) to features."""
    return torch.gather(features, 1, sort_idx.unsqueeze(-1).expand_as(features))


def _to_binary_mask(encoded_mask: torch.Tensor, lead_shape: tuple[int, int]) -> torch.Tensor:
    """Convert connector output mask to a binary (0/1) mask shaped ``(B, S, 1)`` for broadcasting."""
    return (encoded_mask < 0.000001).to(torch.int64).reshape([lead_shape[0], lead_shape[1], 1])


class EmbeddingsProcessor(nn.Module, Disposable):
    """Wraps feature extractor + video connector + optional audio connector.
    Can operate in two modes:
    1. create_embeddings(): Takes pre-computed features + additive mask (backward compat, used by trainer)
    2. process_hidden_states(): Takes raw Gemma hidden states, runs feature extraction + connectors
    """

    def __init__(
        self,
        *,
        feature_extractor: nn.Module | None = None,
        video_connector: Embeddings1DConnector,
        audio_connector: Embeddings1DConnector | None = None,
    ):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.video_connector = video_connector
        self.audio_connector = audio_connector

    def create_embeddings(
        self,
        video_features: torch.Tensor,
        audio_features: torch.Tensor | None,
        additive_attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
        if self.audio_connector is not None and audio_features is None:
            raise ValueError("Audio connector is configured but no audio features were provided.")
        if self.audio_connector is None and audio_features is not None:
            raise ValueError("Audio features were provided but no audio connector is configured.")

        # Connectors expect right-padded input ([valid, pad]). Normalize layout here so the
        # upstream tokenizer can keep using either side without coupling to the connector.
        # The sort index depends only on the mask, so compute it once and reuse for audio.
        sort_idx, mask_for_connector = _compute_right_pad_order(additive_attention_mask)
        video_features = _apply_right_pad_order(video_features, sort_idx)
        video_encoded, video_mask = self.video_connector(video_features, mask_for_connector)
        binary_mask = _to_binary_mask(video_mask, video_encoded.shape[:2])
        video_encoded = video_encoded * binary_mask

        audio_encoded = None
        if self.audio_connector is not None:
            audio_features = _apply_right_pad_order(audio_features, sort_idx)
            audio_encoded, _ = self.audio_connector(audio_features, mask_for_connector)

        return video_encoded, audio_encoded, binary_mask.squeeze(-1)

    def process_hidden_states(
        self,
        hidden_states: tuple[torch.Tensor, ...],
        attention_mask: torch.Tensor,
        padding_side: str = "left",
    ) -> EmbeddingsProcessorOutput:
        """Full pipeline: feature extraction -> connectors -> final embeddings.
        Args:
            hidden_states: Raw Gemma hidden states (tuple of tensors per layer).
            attention_mask: Binary attention mask [B, seq_len].
            padding_side: Padding side used during tokenization.
        Returns:
            EmbeddingsProcessorOutput with video_encoding, audio_encoding, and attention_mask.
        """
        if self.feature_extractor is None:
            raise ValueError("feature_extractor is required for process_hidden_states()")

        video_feats, audio_feats = self.feature_extractor(hidden_states, attention_mask, padding_side)
        additive_mask = convert_to_additive_mask(attention_mask, video_feats.dtype)
        video_enc, audio_enc, binary_mask = self.create_embeddings(video_feats, audio_feats, additive_mask)
        return EmbeddingsProcessorOutput(video_enc, audio_enc, binary_mask)
