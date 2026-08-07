# SPDX-FileCopyrightText: Copyright (c) 2025-2026 Lightricks Ltd.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: LicenseRef-LTX-2
"""LTX-2.3 ("V2") embeddings connector + split feature extractor.

Differences from LTX-2 that this module encodes (all verified against the
LTX-2.3 checkpoint weight shapes):

* Feature extraction is **split per modality and happens before the connector**
  (``caption_proj_before_connector=True``):
    - ``video_aggregate_embed``: Linear(3840*49 -> 4096, bias=True)
    - ``audio_aggregate_embed``: Linear(3840*49 -> 2048, bias=True)
  vs LTX-2's single shared ``aggregate_embed`` Linear(3840*49 -> 3840, bias=False).

* Two separate connectors with distinct dims / depth / gating:
    - video: 32 heads x 128 = 4096, 8 layers, gated, 128 registers
    - audio: 32 heads x 64  = 2048, 8 layers, gated, 128 registers
  vs LTX-2's single 30x128=3840, 2-layer, ungated connector consuming a shared
  projection.

The ``Embeddings1DConnector`` nn.Module itself is structurally identical to
LTX-2 (it already supports configurable heads/dim/layers/registers and gated
attention), so we reuse it and only provide V2 configurators + the split
feature extractor here.
"""

import math

import torch

from ...ltx2.ltx2_core.connector import Embeddings1DConnector
from ...ltx2.ltx2_core.rope import LTXRopeType

# Gemma-3-12b-it exposes 49 hidden states (48 transformer layers + embeddings),
# each of width ``caption_channels`` (3840). The feature extractor flattens all
# of them, matching the checkpoint's [out, 3840*49] projection weights.
_NUM_GEMMA_HIDDEN_STATES = 49


def _transformer_cfg(config: dict) -> dict:
    return config.get("transformer", config)


class LTX23GemmaFeaturesExtractor(torch.nn.Module):
    """Split Gemma feature extractor (video + audio), pre-connector.

    Maps to checkpoint keys ``text_embedding_projection.video_aggregate_embed``
    and ``text_embedding_projection.audio_aggregate_embed``.
    """

    def __init__(
        self,
        caption_channels: int = 3840,
        video_dim: int = 4096,
        audio_dim: int = 2048,
        num_hidden_states: int = _NUM_GEMMA_HIDDEN_STATES,
    ) -> None:
        super().__init__()
        in_features = caption_channels * num_hidden_states
        # Per-modality norm rescale reference: the flattened, per-token-RMS'd
        # features are rescaled by sqrt(out_dim / embedding_dim) before each
        # projection (ltx-core FeatureExtractorV2._rescale_norm). embedding_dim
        # is the Gemma hidden width (caption_channels), NOT the flattened size.
        self.embedding_dim = caption_channels
        self.video_aggregate_embed = torch.nn.Linear(in_features, video_dim, bias=True)
        self.audio_aggregate_embed = torch.nn.Linear(in_features, audio_dim, bias=True)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """x: flattened Gemma features [..., caption_channels * num_hidden_states].

        Returns (video_features -> video_dim, audio_features -> audio_dim).
        Applies the modality-specific ``sqrt(out_dim / embedding_dim)`` rescale
        (matching ltx-core ``FeatureExtractorV2``) before each projection.
        """
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
    # LTX-2.3 default is SPLIT (matches ltx-core); the 2.3 checkpoint sets this
    # explicitly, so the default only guards future/partial configs.
    rope_type = LTXRopeType(cfg.get("rope_type", "split"))
    double_precision_rope = cfg.get("frequencies_precision", False) == "float64"
    return Embeddings1DConnector(
        attention_head_dim=attention_head_dim,
        num_attention_heads=num_attention_heads,
        num_layers=cfg.get("connector_num_layers", 8),
        positional_embedding_max_pos=cfg.get(
            "connector_positional_embedding_max_pos", [4096]
        ),
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
