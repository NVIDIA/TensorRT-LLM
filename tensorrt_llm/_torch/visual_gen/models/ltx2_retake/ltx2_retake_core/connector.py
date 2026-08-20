# SPDX-FileCopyrightText: Copyright (c) 2025-2026 Lightricks Ltd.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: LicenseRef-LTX-2

import math
from typing import Any

import torch

from ...ltx2.ltx2_core.connector import Embeddings1DConnector
from ...ltx2.ltx2_core.rope import LTXRopeType


def _connector_from_config(config: dict[str, Any], prefix: str = "") -> Embeddings1DConnector:
    transformer_config = config.get("transformer", {})

    def value(name: str, default: Any) -> Any:
        key = f"{prefix}{name}" if prefix else f"connector_{name}"
        return transformer_config.get(key, transformer_config.get(f"connector_{name}", default))

    rope_type = value("rope_type", LTXRopeType.INTERLEAVED)
    if isinstance(rope_type, str):
        rope_type = rope_type.lower()

    return Embeddings1DConnector(
        num_attention_heads=value("num_attention_heads", 30),
        attention_head_dim=value("attention_head_dim", 128),
        num_layers=value("num_layers", 2),
        positional_embedding_max_pos=transformer_config.get(
            "connector_positional_embedding_max_pos", [1]
        ),
        rope_type=LTXRopeType(rope_type),
        double_precision_rope=transformer_config.get("frequencies_precision") == "float64",
        apply_gated_attention=transformer_config.get("connector_apply_gated_attention", False),
    )


class Embeddings1DConnectorConfigurator:
    @classmethod
    def from_config(cls, config: dict[str, Any]) -> Embeddings1DConnector:
        return _connector_from_config(config)


class AudioEmbeddings1DConnectorConfigurator:
    @classmethod
    def from_config(cls, config: dict[str, Any]) -> Embeddings1DConnector:
        return _connector_from_config(config, prefix="audio_connector_")


# The stacked Gemma-3-12b states contain 48 transformer layers plus the
# embedding layer, each with width 3840.
_GEMMA_EMBEDDING_DIM = 3840
_GEMMA_NUM_LAYERS = 49
_GEMMA_FLAT_DIM = _GEMMA_EMBEDDING_DIM * _GEMMA_NUM_LAYERS

_NORM_EPS = 1e-6


def _norm_and_concat_per_token_rms(
    encoded_text: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Normalize each token and layer over the Gemma hidden dimension."""
    b, t, d, num_layers = encoded_text.shape
    variance = torch.mean(encoded_text**2, dim=2, keepdim=True)  # [B, T, 1, L]
    normed = encoded_text * torch.rsqrt(variance + _NORM_EPS)
    normed = normed.reshape(b, t, d * num_layers)
    mask_3d = attention_mask.bool().unsqueeze(-1)  # [B, T, 1]
    return torch.where(mask_3d, normed, torch.zeros_like(normed))


def _rescale_norm(x: torch.Tensor, target_dim: int, source_dim: int) -> torch.Tensor:
    """Rescale normalization: ``x * sqrt(target_dim / source_dim)``."""
    return x * math.sqrt(target_dim / source_dim)


class GemmaFeaturesExtractor(torch.nn.Module):
    """Project normalized Gemma states into video and audio connector inputs."""

    def __init__(
        self,
        video_aggregate_embed: torch.nn.Linear,
        embedding_dim: int,
        audio_aggregate_embed: torch.nn.Linear,
    ) -> None:
        super().__init__()
        self.video_aggregate_embed = video_aggregate_embed
        self.audio_aggregate_embed = audio_aggregate_embed
        self.embedding_dim = embedding_dim

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        normed = _norm_and_concat_per_token_rms(hidden_states, attention_mask)
        v_dim = self.video_aggregate_embed.out_features
        video = self.video_aggregate_embed(_rescale_norm(normed, v_dim, self.embedding_dim))
        a_dim = self.audio_aggregate_embed.out_features
        audio = self.audio_aggregate_embed(_rescale_norm(normed, a_dim, self.embedding_dim))
        return video, audio


_EXPECTED_CONNECTOR_CONFIG = {
    "caption_proj_before_connector": True,
    "caption_projection_first_linear": False,
    "caption_proj_input_norm": False,
    "caption_projection_second_linear": False,
}


class GemmaFeaturesExtractorConfigurator:
    """Build the dual-head feature extractor used by LTX-2.3 retake."""

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> torch.nn.Module:
        transformer_config = config.get("transformer", {})

        mismatched = {
            key: transformer_config.get(key)
            for key, expected in _EXPECTED_CONNECTOR_CONFIG.items()
            if transformer_config.get(key) != expected
        }
        if mismatched:
            raise NotImplementedError(
                "LTX-2.3 retake requires the dual-head connector config; got "
                + ", ".join(
                    f"{key}={value!r} (expected {_EXPECTED_CONNECTOR_CONFIG[key]!r})"
                    for key, value in sorted(mismatched.items())
                )
            )

        video_inner_dim = (
            transformer_config["num_attention_heads"] * transformer_config["attention_head_dim"]
        )
        audio_inner_dim = (
            transformer_config["audio_num_attention_heads"]
            * transformer_config["audio_attention_head_dim"]
        )
        return GemmaFeaturesExtractor(
            video_aggregate_embed=torch.nn.Linear(_GEMMA_FLAT_DIM, video_inner_dim, bias=True),
            embedding_dim=_GEMMA_EMBEDDING_DIM,
            audio_aggregate_embed=torch.nn.Linear(_GEMMA_FLAT_DIM, audio_inner_dim, bias=True),
        )
