# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

import torch

from tensorrt_llm._torch.visual_gen.models.ltx_core.text_encoders.gemma.embeddings_connector import (
    Embeddings1DConnector as UpstreamConnector,
)
from tensorrt_llm._torch.visual_gen.models.ltx_core.text_encoders.gemma.embeddings_connector import (
    _BasicTransformerBlock1D as _BasicTransformerBlock1D,
)
from tensorrt_llm._torch.visual_gen.models.ltx_core.text_encoders.gemma.embeddings_processor import (
    _apply_right_pad_order,
    _compute_right_pad_order,
)
from tensorrt_llm._torch.visual_gen.models.ltx_core.text_encoders.gemma.feature_extractor import (
    FeatureExtractorV1,
)

from .rope import LTXRopeType


class Embeddings1DConnector(UpstreamConnector):
    def __init__(self, *args, rope_type=LTXRopeType.INTERLEAVED, **kwargs):
        super().__init__(*args, rope_type=rope_type, **kwargs)

    def forward(self, hidden_states, attention_mask=None):
        # Upstream moved layout normalization from the connector into its processor.
        if self.num_learnable_registers and attention_mask is not None:
            order, attention_mask = _compute_right_pad_order(attention_mask)
            hidden_states = _apply_right_pad_order(hidden_states, order)
        return super().forward(hidden_states, attention_mask)


class Embeddings1DConnectorConfigurator:
    @classmethod
    def from_config(cls, config: dict) -> Embeddings1DConnector:
        config = config.get("transformer", {})
        rope_type = LTXRopeType(config.get("rope_type", "interleaved"))
        double_precision_rope = config.get("frequencies_precision", False) == "float64"
        pe_max_pos = config.get("connector_positional_embedding_max_pos", [1])
        return Embeddings1DConnector(
            positional_embedding_max_pos=pe_max_pos,
            rope_type=rope_type,
            double_precision_rope=double_precision_rope,
            apply_gated_attention=config.get("connector_apply_gated_attention", False),
        )


class GemmaFeaturesExtractorProjLinear(FeatureExtractorV1):
    """The TRTLLM pipeline supplies features normalized before this boundary."""

    def __init__(self):
        super().__init__(torch.nn.Linear(3840 * 49, 3840, bias=False), is_av=True)

    def forward(self, x):
        return self.aggregate_embed(x)

    @classmethod
    def from_config(cls, config):
        return cls()
