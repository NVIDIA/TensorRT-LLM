# SPDX-FileCopyrightText: Copyright (c) 2025-2026 Lightricks Ltd.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: LicenseRef-LTX-2
"""Split Gemma feature extractor and LTX-2.3 connector configurators."""

import math

import torch

from ...ltx2.ltx2_core.connector import Embeddings1DConnector
from ...ltx2.ltx2_core.rope import LTXRopeType

# Gemma-3-12b-it exposes 49 hidden states (48 layers plus embeddings).
_NUM_GEMMA_HIDDEN_STATES = 49


def _transformer_cfg(config: dict) -> dict:
    return config.get("transformer", config)


class LTX23GemmaFeaturesExtractor(torch.nn.Module):
    """Split Gemma feature extractor (video + audio), pre-connector."""

    def __init__(
        self,
        caption_channels: int = 3840,
        video_dim: int = 4096,
        audio_dim: int = 2048,
        num_hidden_states: int = _NUM_GEMMA_HIDDEN_STATES,
    ) -> None:
        super().__init__()
        in_features = caption_channels * num_hidden_states
        # Rescale uses the Gemma hidden width, not the flattened 3840*49 size.
        self.embedding_dim = caption_channels
        self.video_aggregate_embed = torch.nn.Linear(in_features, video_dim, bias=True)
        self.audio_aggregate_embed = torch.nn.Linear(in_features, audio_dim, bias=True)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        v_scale = math.sqrt(self.video_aggregate_embed.out_features / self.embedding_dim)
        a_scale = math.sqrt(self.audio_aggregate_embed.out_features / self.embedding_dim)
        return (
            self.video_aggregate_embed(x * v_scale),
            self.audio_aggregate_embed(x * a_scale),
        )

    @classmethod
    def from_config(cls, config: dict) -> "LTX23GemmaFeaturesExtractor":
        cfg = _transformer_cfg(config)
        return cls(
            caption_channels=cfg.get("caption_channels", 3840),
            video_dim=cfg.get("cross_attention_dim", 4096),
            audio_dim=cfg.get("audio_cross_attention_dim", 2048),
        )


def _build_connector(
    cfg: dict,
    num_attention_heads: int,
    attention_head_dim: int,
) -> Embeddings1DConnector:
    rope_type = LTXRopeType(cfg.get("rope_type", "split"))
    double_precision_rope = cfg.get("frequencies_precision", False) == "float64"
    return Embeddings1DConnector(
        attention_head_dim=attention_head_dim,
        num_attention_heads=num_attention_heads,
        num_layers=cfg.get("connector_num_layers", 8),
        positional_embedding_max_pos=cfg.get("connector_positional_embedding_max_pos", [4096]),
        num_learnable_registers=cfg.get("connector_num_learnable_registers", 128),
        rope_type=rope_type,
        double_precision_rope=double_precision_rope,
        apply_gated_attention=cfg.get("connector_apply_gated_attention", True),
    )


class LTX23VideoConnectorConfigurator:
    """Video connector: 32 x 128 = 4096, 8 layers, gated, 128 registers."""

    @classmethod
    def from_config(cls, config: dict) -> Embeddings1DConnector:
        cfg = _transformer_cfg(config)
        return _build_connector(
            cfg,
            num_attention_heads=cfg.get("connector_num_attention_heads", 32),
            attention_head_dim=cfg.get("connector_attention_head_dim", 128),
        )


class LTX23AudioConnectorConfigurator:
    """Audio connector: 32 x 64 = 2048, 8 layers, gated, 128 registers."""

    @classmethod
    def from_config(cls, config: dict) -> Embeddings1DConnector:
        cfg = _transformer_cfg(config)
        return _build_connector(
            cfg,
            num_attention_heads=cfg.get("audio_connector_num_attention_heads", 32),
            attention_head_dim=cfg.get("audio_connector_attention_head_dim", 64),
        )
